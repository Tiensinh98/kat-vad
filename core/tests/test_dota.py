"""Tests for the DoTA preprocessor and frame-folder feature extraction.

Synthetic frame folders in the real on-disk layout
(``frames/{video_id}/images/000000.jpg``) — no dataset download.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision.io import write_jpeg

from core import constants
from core.data import dota
from core.data.dataset_files import TEST_IDS_FILENAME
from core.data.video_io import (
    list_frame_folders,
    list_frame_images,
    read_sampled_frames_from_dir,
    video_id_from_path,
)
from core.tools import extract_clip_features

CPU = torch.device("cpu")
SUBDIR = "images"


def _write_clip(root: Path, video_id: str, num_frames: int, size: int = 48) -> Path:
    """One clip's frames on disk, zero-padded like DoTA's extractor emits."""
    folder = root / video_id / SUBDIR
    folder.mkdir(parents=True)
    torch.manual_seed(abs(hash(video_id)) % 2**31)
    for index in range(num_frames):
        image = (torch.rand(3, size, size) * 255).to(torch.uint8)
        write_jpeg(image, str(folder / f"{index:06d}.jpg"))
    return root / video_id


@pytest.fixture(scope="module")
def dota_fixture(tmp_path_factory: pytest.TempPathFactory):
    """Three clips with the annotation files DoTA ships."""
    root = tmp_path_factory.mktemp("dota")
    frames_dir = root / "frames"
    entries: dict[str, dict[str, object]] = {
        "aaa_000001": {"num_frames": 24, "anomaly_start": 8, "anomaly_end": 16,
                       "anomaly_class": "ego: turning"},
        "bbb_000002": {"num_frames": 16, "anomaly_start": 4, "anomaly_end": 12,
                       "anomaly_class": "other: lateral"},
        "ccc_000003": {"num_frames": 20, "anomaly_start": 19, "anomaly_end": 20,
                       "anomaly_class": "ego: obstacle"},
    }
    for video_id, entry in entries.items():
        num_frames = int(str(entry["num_frames"]))
        _write_clip(frames_dir, video_id, num_frames)
        entry.update({"video_start": 0, "video_end": num_frames, "subset": "val"})
    (root / "metadata_val.json").write_text(json.dumps(entries), encoding="utf-8")
    (root / "val_split.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    return root


class TestFrameFolderIO:
    def test_lists_video_folders_not_image_folders(self, dota_fixture) -> None:
        folders = list_frame_folders(dota_fixture / "frames", SUBDIR)
        assert [video_id_from_path(p) for p in folders] == [
            "aaa_000001",
            "bbb_000002",
            "ccc_000003",
        ]

    def test_video_id_keeps_dotless_folder_names_whole(self) -> None:
        assert video_id_from_path(Path("0RJPQ_97dcs_000387")) == "0RJPQ_97dcs_000387"
        assert video_id_from_path(Path("01_Accident_106.mp4")) == "01_Accident_106"

    def test_images_are_in_temporal_order(self, dota_fixture) -> None:
        folder = dota_fixture / "frames" / "aaa_000001"
        names = [p.stem for p in list_frame_images(folder, SUBDIR)]
        assert names == sorted(names) == [f"{i:06d}" for i in range(24)]

    def test_strided_read_matches_sampling_formula(self, dota_fixture) -> None:
        folder = dota_fixture / "frames" / "aaa_000001"
        frames = read_sampled_frames_from_dir(folder, stride=8, subdir=SUBDIR)
        assert frames.shape == (3, 48, 48, 3)
        assert frames.dtype == np.uint8


class TestLabelConstruction:
    def test_stride_one_reproduces_raw_window(self, dota_fixture) -> None:
        records = dota.parse_metadata(
            dota_fixture / "metadata_val.json",
            dota.read_split_ids(dota_fixture / "val_split.txt"),
        )
        labels = dota.build_frame_labels(records, stride=1)
        expected = [0] * 8 + [1] * 8 + [0] * 8
        assert labels["aaa_000001"] == expected

    def test_matches_baseline_normalized_span_arithmetic(self, dota_fixture) -> None:
        """LaGoVAD base.py: round(normed_span * feature_length), filled [s, e)."""
        records = dota.parse_metadata(
            dota_fixture / "metadata_val.json",
            dota.read_split_ids(dota_fixture / "val_split.txt"),
        )
        for stride in (1, 2, 4, 8):
            labels = dota.build_frame_labels(records, stride)
            for record in records:
                length = (record.total_frames + stride - 1) // stride
                baseline = [0] * length
                start, end = (round(f * length) for f in record.span)
                for index in range(start, end):
                    baseline[index] = 1
                assert labels[record.video_id] == baseline

    def test_short_window_rounds_away_and_is_reported(self, dota_fixture, caplog) -> None:
        records = dota.parse_metadata(
            dota_fixture / "metadata_val.json",
            dota.read_split_ids(dota_fixture / "val_split.txt"),
        )
        with caplog.at_level("WARNING"):
            labels = dota.build_frame_labels(records, stride=8)
        assert sum(labels["ccc_000003"]) == 0
        assert "lose their anomaly window" in caplog.text

    def test_strict_refuses_a_vanished_window(self, dota_fixture) -> None:
        records = dota.parse_metadata(
            dota_fixture / "metadata_val.json",
            dota.read_split_ids(dota_fixture / "val_split.txt"),
        )
        with pytest.raises(ValueError, match="lose their anomaly window"):
            dota.build_frame_labels(records, stride=8, strict=True)

    def test_rejects_inverted_window(self, tmp_path: Path) -> None:
        bad = {"x_1": {"num_frames": 10, "anomaly_start": 6, "anomaly_end": 3,
                       "anomaly_class": "ego: turning"}}
        (tmp_path / "m.json").write_text(json.dumps(bad), encoding="utf-8")
        (tmp_path / "s.txt").write_text("x_1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="half-open interval"):
            dota.parse_metadata(tmp_path / "m.json", dota.read_split_ids(tmp_path / "s.txt"))

    def test_rejects_split_id_missing_from_metadata(self, dota_fixture, tmp_path: Path) -> None:
        (tmp_path / "s.txt").write_text("aaa_000001\nnot_annotated\n", encoding="utf-8")
        with pytest.raises(ValueError, match="absent from"):
            dota.parse_metadata(
                dota_fixture / "metadata_val.json", dota.read_split_ids(tmp_path / "s.txt")
            )


class TestPreprocess:
    def test_writes_the_standard_dataset_files(self, dota_fixture, tmp_path: Path) -> None:
        records = dota.preprocess(
            metadata=dota_fixture / "metadata_val.json",
            split_file=dota_fixture / "val_split.txt",
            out_dir=tmp_path,
            frames_dir=dota_fixture / "frames",
            frames_subdir=SUBDIR,
            stride=1,
        )
        assert len(records) == 3
        defs = json.loads((tmp_path / constants.DEFS_FILENAME).read_text())
        assert defs == ["Normal", dota.DOTA_CLASS_NAME]
        assert json.loads((tmp_path / constants.LABELS_TRAIN_FILENAME).read_text()) == {}
        frame_labels = json.loads((tmp_path / constants.FRAME_LABELS_TEST_FILENAME).read_text())
        assert set(frame_labels) == {"aaa_000001", "bbb_000002", "ccc_000003"}
        assert [len(v) for v in frame_labels.values()] == [24, 16, 20]
        ids = (tmp_path / TEST_IDS_FILENAME).read_text().split()
        assert ids == sorted(frame_labels)

    def test_meta_carries_the_ego_flag(self, dota_fixture, tmp_path: Path) -> None:
        dota.preprocess(
            metadata=dota_fixture / "metadata_val.json",
            split_file=dota_fixture / "val_split.txt",
            out_dir=tmp_path,
            stride=1,
        )
        meta = json.loads((tmp_path / constants.META_FILENAME).read_text())
        assert meta["aaa_000001"]["ego_involve"] is True
        assert meta["bbb_000002"]["ego_involve"] is False

    def test_disk_frame_count_wins_over_annotation(self, dota_fixture, tmp_path: Path) -> None:
        """A truncated unzip must shift nothing: labels follow the images."""
        frames_dir = tmp_path / "frames"
        _write_clip(frames_dir, "aaa_000001", 12)  # annotation says 24
        _write_clip(frames_dir, "bbb_000002", 16)
        _write_clip(frames_dir, "ccc_000003", 20)
        records = dota.preprocess(
            metadata=dota_fixture / "metadata_val.json",
            split_file=dota_fixture / "val_split.txt",
            out_dir=tmp_path / "out",
            frames_dir=frames_dir,
            frames_subdir=SUBDIR,
            stride=1,
        )
        by_id = {r.video_id: r for r in records}
        assert by_id["aaa_000001"].total_frames == 12
        labels = json.loads(
            (tmp_path / "out" / constants.FRAME_LABELS_TEST_FILENAME).read_text()
        )
        assert len(labels["aaa_000001"]) == 12
        assert labels["aaa_000001"] == [0] * 4 + [1] * 4 + [0] * 4

    def test_missing_frame_folder_is_fatal(self, dota_fixture, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        _write_clip(frames_dir, "aaa_000001", 24)
        with pytest.raises(ValueError, match="no frame folder"):
            dota.preprocess(
                metadata=dota_fixture / "metadata_val.json",
                split_file=dota_fixture / "val_split.txt",
                out_dir=tmp_path / "out",
                frames_dir=frames_dir,
                frames_subdir=SUBDIR,
                stride=1,
            )

    def test_allow_missing_frames_evaluates_the_present_subset(
        self, dota_fixture, tmp_path: Path
    ) -> None:
        """Partial unzip: keep the clips on disk, drop the rest, say so."""
        frames_dir = tmp_path / "frames"
        _write_clip(frames_dir, "aaa_000001", 24)
        _write_clip(frames_dir, "ccc_000003", 20)
        records = dota.preprocess(
            metadata=dota_fixture / "metadata_val.json",
            split_file=dota_fixture / "val_split.txt",
            out_dir=tmp_path / "out",
            frames_dir=frames_dir,
            frames_subdir=SUBDIR,
            stride=1,
            allow_missing_frames=True,
        )
        assert [r.video_id for r in records] == ["aaa_000001", "ccc_000003"]
        labels = json.loads(
            (tmp_path / "out" / constants.FRAME_LABELS_TEST_FILENAME).read_text()
        )
        assert set(labels) == {"aaa_000001", "ccc_000003"}
        ids = (tmp_path / "out" / TEST_IDS_FILENAME).read_text().split()
        assert ids == ["aaa_000001", "ccc_000003"]

    def test_empty_image_folder_is_fatal(self, dota_fixture, tmp_path: Path) -> None:
        """A folder whose images/ is empty is a truncated unzip, not a clip."""
        frames_dir = tmp_path / "frames"
        _write_clip(frames_dir, "aaa_000001", 24)
        _write_clip(frames_dir, "bbb_000002", 16)
        (frames_dir / "ccc_000003" / SUBDIR).mkdir(parents=True)
        with pytest.raises(ValueError, match="empty one"):
            dota.preprocess(
                metadata=dota_fixture / "metadata_val.json",
                split_file=dota_fixture / "val_split.txt",
                out_dir=tmp_path / "out",
                frames_dir=frames_dir,
                frames_subdir=SUBDIR,
                stride=1,
            )

    def test_allow_missing_frames_drops_empty_folders_too(
        self, dota_fixture, tmp_path: Path
    ) -> None:
        frames_dir = tmp_path / "frames"
        _write_clip(frames_dir, "aaa_000001", 24)
        (frames_dir / "bbb_000002" / SUBDIR).mkdir(parents=True)  # empty
        records = dota.preprocess(  # ccc_000003 has no folder at all
            metadata=dota_fixture / "metadata_val.json",
            split_file=dota_fixture / "val_split.txt",
            out_dir=tmp_path / "out",
            frames_dir=frames_dir,
            frames_subdir=SUBDIR,
            stride=1,
            allow_missing_frames=True,
        )
        assert [r.video_id for r in records] == ["aaa_000001"]
        ids = (tmp_path / "out" / TEST_IDS_FILENAME).read_text().split()
        assert ids == ["aaa_000001"]

    def test_folders_outside_the_split_are_ignored(
        self, dota_fixture, tmp_path: Path, caplog
    ) -> None:
        frames_dir = tmp_path / "frames"
        for video_id in ("aaa_000001", "bbb_000002", "ccc_000003"):
            _write_clip(frames_dir, video_id, 16)
        _write_clip(frames_dir, "zzz_000009", 16)  # in the archive, not the split
        with caplog.at_level(logging.INFO, logger=dota.LOGGER.name):
            records = dota.preprocess(
                metadata=dota_fixture / "metadata_val.json",
                split_file=dota_fixture / "val_split.txt",
                out_dir=tmp_path / "out",
                frames_dir=frames_dir,
                frames_subdir=SUBDIR,
                stride=1,
            )
        assert [r.video_id for r in records] == [
            "aaa_000001",
            "bbb_000002",
            "ccc_000003",
        ]
        assert "1 frame folders" in caplog.text

    def test_allow_missing_frames_still_fails_when_nothing_is_on_disk(
        self, dota_fixture, tmp_path: Path
    ) -> None:
        frames_dir = tmp_path / "frames"
        _write_clip(frames_dir, "ghost_000009", 8)
        with pytest.raises(ValueError, match="No annotated clip"):
            dota.preprocess(
                metadata=dota_fixture / "metadata_val.json",
                split_file=dota_fixture / "val_split.txt",
                out_dir=tmp_path / "out",
                frames_dir=frames_dir,
                frames_subdir=SUBDIR,
                stride=1,
                allow_missing_frames=True,
            )


@pytest.fixture(scope="module")
def tiny_encoder() -> object:
    """Random-weight CLIP vision tower — no download, 64-d projection."""
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    config = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, image_size=constants.CROP_SIZE,
        patch_size=32, projection_dim=64,
    )
    torch.manual_seed(0)
    return CLIPVisionModelWithProjection(config).eval()


class TestFrameDirectoryExtraction:
    def test_feature_length_matches_label_length(
        self, dota_fixture, tiny_encoder, tmp_path: Path
    ) -> None:
        out = tmp_path / "clip"
        written = extract_clip_features.extract_frame_directory(
            dota_fixture / "frames", out, tiny_encoder, CPU, stride=8,
            batch_size=2, subdir=SUBDIR,
        )
        assert len(written) == 3
        features = np.load(out / "aaa_000001.npy")
        assert features.shape == (3, 64)  # ceil(24 / 8), tiny projection_dim
        assert features.dtype == np.float32

    def test_batching_does_not_change_features(
        self, dota_fixture, tiny_encoder, tmp_path: Path
    ) -> None:
        """Streaming in chunks must be numerically identical to one big batch."""
        folder = dota_fixture / "frames" / "aaa_000001"
        small = extract_clip_features.encode_frame_dir(
            folder, tiny_encoder, CPU, stride=2, batch_size=2, subdir=SUBDIR
        )
        big = extract_clip_features.encode_frame_dir(
            folder, tiny_encoder, CPU, stride=2, batch_size=64, subdir=SUBDIR
        )
        np.testing.assert_allclose(small, big, atol=1e-5)

    def test_ids_file_restricts_the_run(
        self, dota_fixture, tiny_encoder, tmp_path: Path
    ) -> None:
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("aaa_000001\nbbb_000002\n", encoding="utf-8")
        written = extract_clip_features.extract_frame_directory(
            dota_fixture / "frames", tmp_path / "clip", tiny_encoder, CPU,
            stride=8, batch_size=4, subdir=SUBDIR,
            video_ids=extract_clip_features.read_ids_file(ids_file),
        )
        assert sorted(p.stem for p in written) == ["aaa_000001", "bbb_000002"]

    def test_unknown_id_is_fatal(self, dota_fixture, tiny_encoder, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no frame folder"):
            extract_clip_features.extract_frame_directory(
                dota_fixture / "frames", tmp_path / "clip", tiny_encoder, CPU,
                stride=8, subdir=SUBDIR, video_ids={"aaa_000001", "ghost_000009"},
            )

    def test_resume_skips_existing(self, dota_fixture, tiny_encoder, tmp_path: Path) -> None:
        out = tmp_path / "clip"
        first = extract_clip_features.extract_frame_directory(
            dota_fixture / "frames", out, tiny_encoder, CPU, stride=8, subdir=SUBDIR
        )
        second = extract_clip_features.extract_frame_directory(
            dota_fixture / "frames", out, tiny_encoder, CPU, stride=8, subdir=SUBDIR
        )
        assert len(first) == 3 and second == []

    def test_resume_redoes_an_interrupted_write(
        self, dota_fixture, tiny_encoder, tmp_path: Path
    ) -> None:
        """A clip killed mid-`np.save` must be re-extracted, not skipped forever."""
        out = tmp_path / "clip"
        extract_clip_features.extract_frame_directory(
            dota_fixture / "frames", out, tiny_encoder, CPU, stride=8, subdir=SUBDIR
        )
        killed = out / "bbb_000002.npy"
        payload = killed.read_bytes()
        killed.write_bytes(payload[: len(payload) // 2])
        redone = extract_clip_features.extract_frame_directory(
            dota_fixture / "frames", out, tiny_encoder, CPU, stride=8, subdir=SUBDIR
        )
        assert [p.name for p in redone] == ["bbb_000002.npy"]
        assert np.load(killed).shape[1] == 64


class TestCrossDatasetEval:
    """The real DoTA run: an MSAD-trained checkpoint scored on DoTA data.

    Cheap insurance against burning a Colab feature-extraction run and only
    then discovering that eval trips on a DoTA-specific path — the ``dota``
    verbalizer key, an empty ``labels_train.json``, or the 12-class-checkpoint
    / 2-class-definitions mismatch (definition conditioning means there is
    none, and this proves it).
    """

    def test_msad_checkpoint_scores_dota(self, dota_fixture, tmp_path: Path) -> None:
        from core import evaluate, train
        from core.tests.fixtures import build_fixture

        torch.set_num_threads(1)
        msad = build_fixture(tmp_path / "msad", num_abnormal=2, num_normal=2)
        ckpt_dir = tmp_path / "trained"
        train.main([
            "--data-dir", str(msad.data_dir),
            "--clip-dir", str(msad.clip_dir),
            "--flow-dir", str(msad.flow_dir),
            "--knn-cache", str(msad.knn_cache_path),
            "--output-dir", str(ckpt_dir),
            "--text-encoder", "stub",
            "--set", "train.num_epochs=1",
            "--set", "train.batch_size=4",
            "--set", "train.device=cpu",
            "--set", "loss.captions_from_definitions=true",
        ])

        data_dir = tmp_path / "dota_data"
        dota.preprocess(
            metadata=dota_fixture / "metadata_val.json",
            split_file=dota_fixture / "val_split.txt",
            out_dir=data_dir,
            stride=8,
        )
        clip_dir = tmp_path / "dota_clip"
        clip_dir.mkdir()
        frame_labels = json.loads(
            (data_dir / constants.FRAME_LABELS_TEST_FILENAME).read_text()
        )
        rng = np.random.default_rng(0)
        for video_id, labels in frame_labels.items():
            features = rng.standard_normal((len(labels), constants.CLIP_FEATURE_DIM))
            np.save(clip_dir / f"{video_id}.npy", features.astype(np.float32))

        out = tmp_path / "eval_dota"
        evaluate.main([
            "--ckpt", str(ckpt_dir / "checkpoint_last.pt"),
            "--data-dir", str(data_dir),
            "--clip-dir", str(clip_dir),
            "--output-dir", str(out),
            "--text-encoder", "stub",
            "--set", "data.dataset=DoTA",
            "--set", "train.device=cpu",
            "--save-scores",
        ])
        results = json.loads((out / evaluate.RESULTS_FILENAME).read_text())
        assert results["dataset"] == constants.DOTA_DATASET
        assert results["num_videos"] == 3
        assert 0.0 <= results["auc"] <= 1.0
        scores = np.load(out / evaluate.SCORES_DIRNAME / "aaa_000001.npz")
        assert scores["score"].shape == (3,)  # ceil(24 / 8)
        assert list(scores["class_names"]) == ["Normal", dota.DOTA_CLASS_NAME]
