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


class TestRaftFrameExtraction:
    """P2: datasets that ship frames (TAD) instead of videos.

    The parity test is the load-bearing one. A second extraction path is only
    safe if it is the *same* pipeline with a different reader — anything else
    silently produces a flow cache that is not comparable to the MSAD one every
    v1 number was measured against.
    """

    @staticmethod
    def _frames_from_video(video_path: Path, folder: Path, subdir: str | None = None):
        """Dump a video's stride-sampled frames as lossless PNGs, one per file."""
        from torchvision.io import write_png

        target = folder / subdir if subdir else folder
        target.mkdir(parents=True, exist_ok=True)
        frames = read_sampled_frames(video_path, 1)
        for index, frame in enumerate(frames):
            write_png(
                torch.from_numpy(frame).permute(2, 0, 1).contiguous(),
                str(target / f"{index:06d}.png"),
            )
        return frames

    def test_matches_the_video_path_exactly(self, video_fixture, tmp_path: Path) -> None:
        """Same pixels in, bit-identical e_O out — whatever the source was."""
        model = raft_extract.load_raft_model(small=True, random_weights=True)
        video = list_videos(video_fixture.videos_dir)[0]
        frames_dir = tmp_path / "frames"
        self._frames_from_video(video, frames_dir / f"{video.stem}.mp4")

        from_frames = raft_extract.extract_frame_directory(
            frames_dir=frames_dir, dataset=constants.TAD_DATASET, model=model,
            device=CPU, cache_root=tmp_path / "flow_frames", batch_size=4,
            size=(128, 128),
        )
        from_video = raft_extract.extract_directory(
            videos_dir=video_fixture.videos_dir, dataset=constants.TAD_DATASET,
            model=model, device=CPU, cache_root=tmp_path / "flow_video",
            batch_size=4, size=(128, 128),
        )
        assert len(from_frames) == 1
        video_target = next(p for p in from_video if p.stem == video.stem)
        assert np.array_equal(np.load(from_frames[0]), np.load(video_target))
        stats_name = f"{video.stem}{constants.FLOW_STATS_SUFFIX}"
        assert np.array_equal(
            np.load(from_frames[0].parent / stats_name),
            np.load(video_target.parent / stats_name),
        )

    def test_shapes_resume_and_id_layout(self, video_fixture, tmp_path: Path) -> None:
        model = raft_extract.load_raft_model(small=True, random_weights=True)
        frames_dir = tmp_path / "frames"
        # TAD's split-directory layout, and its ".mp4" folder names
        for index, video in enumerate(list_videos(video_fixture.videos_dir)[:2]):
            split = "abnormal" if index == 0 else "normal"
            self._frames_from_video(video, frames_dir / split / f"{video.stem}.mp4")

        cache_root = tmp_path / "flow" / "v1"
        written = raft_extract.extract_frame_directory(
            frames_dir=frames_dir, dataset=constants.TAD_DATASET, model=model,
            device=CPU, cache_root=cache_root, batch_size=4, size=(128, 128),
        )
        assert len(written) == 2
        # the split directory does not leak into the id (video_id_from_path)
        assert {p.stem for p in written} == {
            v.stem for v in list_videos(video_fixture.videos_dir)[:2]
        }
        embeddings = np.load(written[0])
        assert embeddings.shape[1] == constants.FLOW_DIM
        stats = np.load(
            written[0].parent / f"{written[0].stem}{constants.FLOW_STATS_SUFFIX}"
        )
        assert stats.shape == (embeddings.shape[0], constants.FLOW_STATS_DIM)
        assert np.allclose(stats[-1], stats[-2])  # last flow duplicated, spec §1
        assert (cache_root / constants.FLOW_PROJECTION_FILENAME).exists()
        assert (
            raft_extract.extract_frame_directory(
                frames_dir=frames_dir, dataset=constants.TAD_DATASET, model=model,
                device=CPU, cache_root=cache_root, size=(128, 128),
            )
            == []
        )

    def test_stride_matches_the_clip_cache_row_for_row(
        self, video_fixture, tmp_path: Path
    ) -> None:
        """The training dataset asserts len(flow) == len(features); prove it holds."""
        model = raft_extract.load_raft_model(small=True, random_weights=True)
        video = list_videos(video_fixture.videos_dir)[0]
        frames_dir = tmp_path / "frames"
        frames = self._frames_from_video(video, frames_dir / f"{video.stem}.mp4")

        written = raft_extract.extract_frame_directory(
            frames_dir=frames_dir, dataset=constants.TAD_DATASET, model=model,
            device=CPU, cache_root=tmp_path / "flow", stride=STRIDE, batch_size=4,
            size=(128, 128),
        )
        expected = (len(frames) + STRIDE - 1) // STRIDE
        assert np.load(written[0]).shape[0] == expected

    def test_ids_file_scopes_the_run_and_missing_ids_raise(
        self, video_fixture, tmp_path: Path
    ) -> None:
        model = raft_extract.load_raft_model(small=True, random_weights=True)
        videos = list_videos(video_fixture.videos_dir)[:2]
        frames_dir = tmp_path / "frames"
        for video in videos:
            self._frames_from_video(video, frames_dir / f"{video.stem}.mp4")

        written = raft_extract.extract_frame_directory(
            frames_dir=frames_dir, dataset=constants.TAD_DATASET, model=model,
            device=CPU, cache_root=tmp_path / "flow", batch_size=4, size=(128, 128),
            video_ids={videos[0].stem},
        )
        assert [p.stem for p in written] == [videos[0].stem]

        with pytest.raises(ValueError, match="have no frame folder under"):
            raft_extract.extract_frame_directory(
                frames_dir=frames_dir, dataset=constants.TAD_DATASET, model=model,
                device=CPU, cache_root=tmp_path / "flow2", size=(128, 128),
                video_ids={"not_a_clip"},
            )

    def test_frames_subdir_layout(self, video_fixture, tmp_path: Path) -> None:
        """DoTA's frames/{id}/images/*.png shape, on the flow path too."""
        model = raft_extract.load_raft_model(small=True, random_weights=True)
        video = list_videos(video_fixture.videos_dir)[0]
        frames_dir = tmp_path / "frames"
        self._frames_from_video(video, frames_dir / video.stem, subdir="images")

        written = raft_extract.extract_frame_directory(
            frames_dir=frames_dir, dataset=constants.DOTA_DATASET, model=model,
            device=CPU, cache_root=tmp_path / "flow", batch_size=4, size=(128, 128),
            subdir="images",
        )
        assert [p.stem for p in written] == [video.stem]

    def test_cli_requires_exactly_one_source(self) -> None:
        parser = raft_extract.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset", constants.TAD_DATASET])
        with pytest.raises(SystemExit):
            parser.parse_args(["--videos-dir", "a", "--frames-dir", "b"])
        assert parser.parse_args(["--frames-dir", "b"]).videos_dir is None

    def test_cli_dispatches_to_the_frame_path(
        self, video_fixture, tmp_path: Path
    ) -> None:
        video = list_videos(video_fixture.videos_dir)[0]
        frames_dir = tmp_path / "frames"
        self._frames_from_video(video, frames_dir / "abnormal" / f"{video.stem}.mp4")
        ids_file = tmp_path / "train_ids.txt"
        ids_file.write_text(f"{video.stem}\n", encoding="utf-8")
        cache_root = tmp_path / "flow" / "v1"

        raft_extract.main([
            "--frames-dir", str(frames_dir),
            "--ids-file", str(ids_file),
            "--dataset", constants.TAD_DATASET,
            "--cache-root", str(cache_root),
            "--batch-size", "4",
            "--device", "cpu",
            "--small", "--random-weights",
        ])
        assert (cache_root / constants.TAD_DATASET / f"{video.stem}.npy").exists()
