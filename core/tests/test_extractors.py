"""Tests for the CLIP feature and RAFT flow extraction CLIs.

Real tiny videos (cv2-written), random-weight models — no downloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants
from core.data.video_io import list_videos, read_sampled_frames
from core.flow import raft_extract
from core.tests.fixtures import build_fixture
from core.tools import extract_clip_features

CPU = torch.device("cpu")
STRIDE = constants.FRAME_STRIDE


@pytest.fixture(scope="module")
def video_fixture(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("msad_videos")
    return build_fixture(root, num_abnormal=2, num_normal=2, with_videos=True)


@pytest.fixture(scope="module")
def tiny_clip_encoder() -> object:
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=constants.CROP_SIZE,
        patch_size=32,
        projection_dim=64,
    )
    torch.manual_seed(0)
    return CLIPVisionModelWithProjection(config).eval()


class TestVideoIO:
    def test_frame_count_matches_stride_formula(self, video_fixture) -> None:
        video = list_videos(video_fixture.videos_dir)[0]
        frames = read_sampled_frames(video, STRIDE)
        meta_total = _total_frames_of(video_fixture, video.stem)
        assert frames.shape[0] == (meta_total + STRIDE - 1) // STRIDE
        assert frames.shape[-1] == 3 and frames.dtype == np.uint8


def _total_frames_of(fixture, video_id: str) -> int:
    import json

    meta = json.loads((fixture.data_dir / constants.META_FILENAME).read_text())
    return int(meta[video_id]["total_frames"])


class TestClipExtraction:
    def test_shapes_and_resume(self, video_fixture, tiny_clip_encoder, tmp_path: Path) -> None:
        out = tmp_path / "clip"
        written = extract_clip_features.extract_directory(
            video_fixture.videos_dir, out, tiny_clip_encoder, CPU, batch_size=16
        )
        assert len(written) == 4
        features = np.load(written[0])
        assert features.dtype == np.float32
        assert features.shape[1] == 64  # tiny projection_dim
        # resume: second run writes nothing
        assert (
            extract_clip_features.extract_directory(
                video_fixture.videos_dir, out, tiny_clip_encoder, CPU
            )
            == []
        )

    def test_preprocess_shape_and_range(self) -> None:
        frames = np.random.default_rng(0).integers(
            0, 255, size=(3, 60, 90, 3), dtype=np.uint8
        )
        pixels = extract_clip_features.preprocess_frames(frames)
        assert pixels.shape == (3, 3, constants.CROP_SIZE, constants.CROP_SIZE)
        assert torch.isfinite(pixels).all()

    def test_no_center_crop_same_shape_different_content(self) -> None:
        """Both transforms give CLIP its 224 square, but not the same pixels.

        A wide frame is the case that matters: center-crop discards the left
        and right thirds, which on dashcam footage is where lateral motion is.
        """
        frames = np.random.default_rng(0).integers(
            0, 255, size=(2, 60, 180, 3), dtype=np.uint8
        )
        cropped = extract_clip_features.preprocess_frames(frames, center_crop=True)
        squashed = extract_clip_features.preprocess_frames(frames, center_crop=False)
        assert cropped.shape == squashed.shape
        assert torch.isfinite(squashed).all()
        assert not torch.allclose(cropped, squashed)

    def test_no_center_crop_is_identity_on_square_input(self) -> None:
        """Square frames have nothing to crop, so the transforms must agree."""
        frames = np.random.default_rng(1).integers(
            0, 255, size=(2, 64, 64, 3), dtype=np.uint8
        )
        assert torch.allclose(
            extract_clip_features.preprocess_frames(frames, center_crop=True),
            extract_clip_features.preprocess_frames(frames, center_crop=False),
            atol=1e-5,
        )

    def test_center_crop_flag_reaches_the_cache(
        self, tiny_clip_encoder, tmp_path: Path
    ) -> None:
        """The flag must change the written features, not just the signature.

        Uses a wide (16:9-ish) frame folder: on a square input the transforms
        agree by construction, so a square fixture would pass vacuously.
        """
        from torchvision.io import write_jpeg

        frames_dir = tmp_path / "frames"
        folder = frames_dir / "wide_000001" / "images"
        folder.mkdir(parents=True)
        torch.manual_seed(0)
        for index in range(4):
            write_jpeg(
                (torch.rand(3, 45, 80) * 255).to(torch.uint8),
                str(folder / f"{index:06d}.jpg"),
            )

        caches = {}
        for name, center_crop in (("default", True), ("ncc", False)):
            out = tmp_path / name
            extract_clip_features.extract_frame_directory(
                frames_dir, out, tiny_clip_encoder, CPU, stride=1, batch_size=4,
                subdir="images", center_crop=center_crop,
            )
            caches[name] = np.load(out / "wide_000001.npy")

        assert caches["default"].shape == caches["ncc"].shape
        assert not np.allclose(caches["default"], caches["ncc"])


class TestFlowStatistics:
    def test_stats_dim_and_determinism(self) -> None:
        flow = torch.randn(4, 2, 24, 32)
        stats = raft_extract.flow_statistics(flow)
        assert stats.shape == (4, constants.FLOW_STATS_DIM)
        assert torch.allclose(stats, raft_extract.flow_statistics(flow))
        # histogram part is L1-normalized
        assert torch.allclose(stats[:, 7:].sum(dim=1), torch.ones(4))

    def test_projection_roundtrip_and_failfast(self, tmp_path: Path) -> None:
        matrix = raft_extract.make_projection()
        assert matrix.shape == (constants.FLOW_STATS_DIM, constants.FLOW_DIM)
        assert np.allclose(matrix, raft_extract.make_projection())  # seeded
        path = tmp_path / constants.FLOW_PROJECTION_FILENAME
        raft_extract.save_projection(path, matrix, constants.FLOW_PROJECTION_SEED)
        assert np.allclose(raft_extract.load_projection(path), matrix)
        with pytest.raises(FileNotFoundError, match="missing"):
            raft_extract.load_projection(tmp_path / "nope.npz")


class TestRaftExtraction:
    def test_end_to_end_shapes_and_resume(self, video_fixture, tmp_path: Path) -> None:
        model = raft_extract.load_raft_model(small=True, random_weights=True)
        cache_root = tmp_path / "flow" / "v1"
        written = raft_extract.extract_directory(
            videos_dir=video_fixture.videos_dir,
            dataset=constants.MSAD_DATASET,
            model=model,
            device=CPU,
            cache_root=cache_root,
            batch_size=4,
            size=(128, 128),  # keep the random-weight CPU run fast
        )
        assert len(written) == 4
        embeddings = np.load(written[0])
        stats = np.load(
            written[0].parent / f"{written[0].stem}{constants.FLOW_STATS_SUFFIX}"
        )
        assert embeddings.shape[1] == constants.FLOW_DIM
        assert stats.shape == (embeddings.shape[0], constants.FLOW_STATS_DIM)
        # L rows for L sampled frames (last one duplicated, spec §1)
        video = next(
            v for v in list_videos(video_fixture.videos_dir) if v.stem == written[0].stem
        )
        assert embeddings.shape[0] == len(read_sampled_frames(video, STRIDE))
        assert np.allclose(stats[-1], stats[-2])
        assert (cache_root / constants.FLOW_PROJECTION_FILENAME).exists()
        # resume
        assert (
            raft_extract.extract_directory(
                videos_dir=video_fixture.videos_dir,
                dataset=constants.MSAD_DATASET,
                model=model,
                device=CPU,
                cache_root=cache_root,
            )
            == []
        )

    def test_single_frame_video_zero_stats(self) -> None:
        frames = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        model = raft_extract.load_raft_model(small=True, random_weights=True)
        stats = raft_extract.video_flow_stats(frames, model, CPU, size=(128, 128))
        assert stats.shape == (1, constants.FLOW_STATS_DIM)
        assert not stats.any()
