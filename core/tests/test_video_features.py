"""Tests for the v2 motion stream extractor (plan katvad-v2-e0-e2.md P1).

The failures that matter are silent: a clip that peeks past its CLIP step
(lookahead), a feature row count that drifts from the CLIP cache, a checkpoint
that loads with a layer missing, weights that are not the pinned bytes, and a
cache extended under a different transform. Tiny random encoders; no download.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from core import constants
from core.data.dataset_files import num_sampled_frames
from core.models import videomae_v2
from core.tools import extract_video_features as evf

TINY_DIM, TINY_DEPTH, TINY_HEADS = 32, 1, 2
FRAME_SIZE = 8
STRIDE = 8
N_FRAMES = 50


class LastFrameProbe(nn.Module):
    """``(B,3,T,H,W)`` -> the newest frame's mean and the oldest frame's mean."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([x[:, 0, -1].mean(dim=(1, 2)), x[:, 0, 0].mean(dim=(1, 2))], dim=1)


def _write_frames(folder: Path, count: int, bump_from: int | None = None) -> None:
    """Grey frames whose intensity encodes the frame index; optional later change."""
    from torchvision.io import write_png

    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        value = index * 4 % 256 if bump_from is None or index < bump_from else 255
        image = torch.full((3, FRAME_SIZE, FRAME_SIZE), value, dtype=torch.uint8)
        write_png(image, str(folder / f"{index:06d}.png"))


def _tiny_encoder() -> videomae_v2.VideoMAEv2Encoder:
    return videomae_v2.VideoMAEv2Encoder(TINY_DIM, TINY_DEPTH, TINY_HEADS)


class TestCausalIndices:
    def test_full_clip_is_every_third_frame_ending_at_the_step(self) -> None:
        assert evf.causal_clip_indices(45) == list(range(0, 46, 3))

    def test_start_of_video_repeats_frame_zero(self) -> None:
        clip = evf.causal_clip_indices(8)
        assert len(clip) == constants.VIDEOMAE_CLIP_FRAMES
        assert clip[-3:] == [2, 5, 8]
        assert set(clip[:-3]) == {0}

    @pytest.mark.parametrize("end", [0, 7, 44, 45, 300])
    def test_never_looks_ahead(self, end: int) -> None:
        clip = evf.causal_clip_indices(end)
        assert max(clip) == end == clip[-1]
        assert clip == sorted(clip)

    @pytest.mark.parametrize("total", [1, 8, 9, 50, 322])
    def test_steps_match_the_clip_cache_row_count(self, total: int) -> None:
        assert len(evf.step_end_frames(total, STRIDE)) == num_sampled_frames(total, STRIDE)


class TestSquash:
    def test_shape_and_imagenet_normalization(self) -> None:
        frames = np.full((2, 40, 90, 3), 255, dtype=np.uint8)  # non-square: squashed
        out = evf.squash_frames(frames)
        assert out.shape == (2, 3, constants.CROP_SIZE, constants.CROP_SIZE)
        expected = (1.0 - np.array(constants.VIDEOMAE_IMAGE_MEAN)) / np.array(
            constants.VIDEOMAE_IMAGE_STD
        )
        np.testing.assert_allclose(out[0, :, 5, 5].numpy(), expected, rtol=1e-5)


class TestEncoder:
    def test_forward_shape_and_token_guard(self) -> None:
        model = _tiny_encoder().eval()
        clip = torch.randn(2, 3, constants.VIDEOMAE_CLIP_FRAMES, constants.CROP_SIZE,
                           constants.CROP_SIZE)
        with torch.no_grad():
            assert model(clip).shape == (2, TINY_DIM)
            with pytest.raises(ValueError, match="tokens"):
                model(clip[:, :, :8])

    def test_arch_table_matches_upstream(self) -> None:
        assert videomae_v2.build_encoder(constants.VIDEOMAE_ENCODER_S).dim == 384
        with pytest.raises(KeyError):
            videomae_v2.build_encoder("vit_x")

    def test_head_is_the_only_key_dropped_and_load_is_strict(self) -> None:
        model = _tiny_encoder()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        checkpoint = {"module": {**state, "head.weight": torch.zeros(3, TINY_DIM),
                                 "head.bias": torch.zeros(3)}}
        unwrapped = videomae_v2.unwrap_state(checkpoint)
        assert set(unwrapped) == set(state)
        _tiny_encoder().load_state_dict(unwrapped, strict=True)
        unwrapped.pop("fc_norm.weight")
        with pytest.raises(RuntimeError, match=r"fc_norm\.weight"):
            _tiny_encoder().load_state_dict(unwrapped, strict=True)

    def test_position_table_is_not_a_checkpoint_key(self) -> None:
        assert "pos_embed" not in _tiny_encoder().state_dict()

    def test_sinusoid_table_values(self) -> None:
        table = videomae_v2.sinusoid_table(3, 4)[0]
        assert table.shape == (3, 4)
        np.testing.assert_allclose(table[1].numpy(),
                                   [np.sin(1.0), np.cos(1.0), np.sin(0.01), np.cos(0.01)],
                                   rtol=1e-6)

    def test_weights_that_are_not_the_pinned_bytes_raise(self, tmp_path: Path) -> None:
        fake = tmp_path / "w.pth"
        fake.write_bytes(b"not the weights")
        with pytest.raises(ValueError, match="sha256"):
            videomae_v2.verify_weights(fake, constants.VIDEOMAE_ENCODER_B)


class TestEncodeFrameDir:
    def test_rows_align_with_the_clip_cache(self, tmp_path: Path) -> None:
        _write_frames(tmp_path / "v", N_FRAMES)
        out = evf.encode_frame_dir(tmp_path / "v", _tiny_encoder().eval(),
                                   torch.device("cpu"), STRIDE, batch_size=3)
        assert out.shape == (num_sampled_frames(N_FRAMES, STRIDE), TINY_DIM)
        assert out.dtype == np.float32

    def test_newest_frame_is_the_step_frame_and_oldest_is_45_back(self, tmp_path: Path) -> None:
        _write_frames(tmp_path / "v", N_FRAMES)
        out = evf.encode_frame_dir(tmp_path / "v", LastFrameProbe(), torch.device("cpu"), STRIDE)
        frames = evf.decode_frames(sorted((tmp_path / "v").iterdir()), list(range(N_FRAMES)))
        for step, end in enumerate(evf.step_end_frames(N_FRAMES, STRIDE)):
            oldest = max(0, end - 45)
            assert out[step, 0] == pytest.approx(frames[end][0].mean().item(), abs=1e-5)
            assert out[step, 1] == pytest.approx(frames[oldest][0].mean().item(), abs=1e-5)

    def test_changing_future_frames_leaves_earlier_steps_unchanged(self, tmp_path: Path) -> None:
        bump = 30
        _write_frames(tmp_path / "a", N_FRAMES)
        _write_frames(tmp_path / "b", N_FRAMES, bump_from=bump)
        model = _tiny_encoder().eval()
        a = evf.encode_frame_dir(tmp_path / "a", model, torch.device("cpu"), STRIDE)
        b = evf.encode_frame_dir(tmp_path / "b", model, torch.device("cpu"), STRIDE)
        before = [i for i, end in enumerate(evf.step_end_frames(N_FRAMES, STRIDE)) if end < bump]
        np.testing.assert_array_equal(a[before], b[before])
        assert not np.array_equal(a[len(before):], b[len(before):])


class TestShuffleControl:
    """D6 / N11: the shuffle cache keeps each window's frames and loses only their order."""

    def test_order_is_a_fixed_non_identity_permutation(self) -> None:
        order = evf.shuffled_frame_order(constants.V2_D6_SHUFFLE_SEED)
        assert sorted(order) == list(range(constants.VIDEOMAE_CLIP_FRAMES))
        assert order != sorted(order)
        assert order == evf.shuffled_frame_order(constants.V2_D6_SHUFFLE_SEED)

    def test_shuffled_clip_moves_the_newest_frame(self, tmp_path: Path) -> None:
        _write_frames(tmp_path / "v", N_FRAMES)
        order = evf.shuffled_frame_order(constants.V2_D6_SHUFFLE_SEED)
        cpu = torch.device("cpu")
        out = evf.encode_frame_dir(tmp_path / "v", LastFrameProbe(), cpu, STRIDE,
                                   frame_order=order)
        frames = evf.decode_frames(sorted((tmp_path / "v").iterdir()), list(range(N_FRAMES)))
        for step, end in enumerate(evf.step_end_frames(N_FRAMES, STRIDE)):
            clip = evf.causal_clip_indices(end)
            assert out[step, 0] == pytest.approx(frames[clip[order[-1]]][0].mean().item(), abs=1e-5)
            assert out[step, 1] == pytest.approx(frames[clip[order[0]]][0].mean().item(), abs=1e-5)

    def test_order_invariant_encoder_sees_the_same_frames(self, tmp_path: Path) -> None:
        class MeanProbe(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.mean(dim=(1, 2, 3, 4)).unsqueeze(1)

        _write_frames(tmp_path / "v", N_FRAMES)
        cpu = torch.device("cpu")
        plain = evf.encode_frame_dir(tmp_path / "v", MeanProbe(), cpu, STRIDE)
        shuffled = evf.encode_frame_dir(tmp_path / "v", MeanProbe(), cpu, STRIDE,
                                        frame_order=evf.shuffled_frame_order(7))
        np.testing.assert_allclose(plain, shuffled, rtol=1e-5)

    def test_a_non_permutation_is_refused(self, tmp_path: Path) -> None:
        _write_frames(tmp_path / "v", 10)
        with pytest.raises(ValueError, match="permutation"):
            evf.encode_frame_dir(tmp_path / "v", LastFrameProbe(), torch.device("cpu"), STRIDE,
                                 frame_order=[0] * constants.VIDEOMAE_CLIP_FRAMES)


class TestExtractDirectory:
    def test_resume_and_ids_filter(self, tmp_path: Path) -> None:
        for vid in ("t01_v001", "t01_v002"):
            _write_frames(tmp_path / "flat" / vid, 20)
        out_dir = tmp_path / "cache"
        model = _tiny_encoder().eval()
        written = evf.extract_frame_directory(tmp_path / "flat", out_dir, model,
                                              torch.device("cpu"), STRIDE,
                                              video_ids={"t01_v001"})
        assert [p.name for p in written] == ["t01_v001.npy"]
        again = evf.extract_frame_directory(tmp_path / "flat", out_dir, model,
                                            torch.device("cpu"), STRIDE)
        assert [p.name for p in again] == ["t01_v002.npy"]

    def test_manifest_mismatch_raises_unless_forced(self, tmp_path: Path) -> None:
        sha = constants.VIDEOMAE_WEIGHTS_SHA256[constants.VIDEOMAE_ENCODER_B]
        first = evf.build_manifest(constants.VIDEOMAE_ENCODER_B, STRIDE, sha)
        evf.check_or_write_manifest(tmp_path, first, force=False)
        evf.check_or_write_manifest(tmp_path, first, force=False)  # same settings: fine
        other = evf.build_manifest(constants.VIDEOMAE_ENCODER_B, 3, sha)
        with pytest.raises(ValueError, match="stride"):
            evf.check_or_write_manifest(tmp_path, other, force=False)
        evf.check_or_write_manifest(tmp_path, other, force=True)
        saved = json.loads((tmp_path / constants.VIDEO_MANIFEST_FILENAME).read_text())
        assert saved["stride"] == 3

    def test_default_output_dir_names_encoder_stride_and_transform(self) -> None:
        path = evf.default_output_dir(constants.VIDEOMAE_ENCODER_B, "DADA2000_orig", STRIDE)
        assert path.parts[-2:] == (constants.VIDEOMAE_ENCODER_B, "DADA2000_orig_s8_squash")
