"""Tests for the TAD preprocessor (P1) and the TAD training path.

Two things are under test here:

* ``core.data.tad`` in **both** modes — the eval-only default that every TAD
  artifact before 2026-09-02 was built with must be bit-for-bit unchanged, and
  the new ``--with-train-split`` mode must derive weak labels from the split
  directory and refuse to guess when it cannot.
* That the files it writes actually train, under the **v1** gate this branch
  ships (the frozen MLP over the flow norm) — so a TAD run cannot be blocked
  for a data reason that was never exercised on this dataset.

No downloads: tiny PNG frame folders, synthetic CLIP/flow caches, stub text
encoder, CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants, train
from core.data import tad
from core.data.dataset_files import TEST_IDS_FILENAME, TRAIN_IDS_FILENAME
from core.data.knn_cache import build_knn_cache

torch.set_num_threads(1)

STRIDE = 2
RAW_FRAMES = 24  # -> 12 sampled frames, comfortably above the top-k denominators
FRAME_SIZE = 8
SEED = 11

# (id, split dirname, annotated?) — 3 abnormal / 3 normal outside the annotation
CLIPS = (
    ("01_Accident_001", constants.TAD_ABNORMAL_DIRNAME, True),
    ("01_Accident_002", constants.TAD_ABNORMAL_DIRNAME, True),
    ("01_Accident_003", constants.TAD_ABNORMAL_DIRNAME, False),
    ("01_Accident_004", constants.TAD_ABNORMAL_DIRNAME, False),
    ("01_Accident_005", constants.TAD_ABNORMAL_DIRNAME, False),
    ("Normal_001", constants.TAD_NORMAL_DIRNAME, True),
    ("Normal_002", constants.TAD_NORMAL_DIRNAME, False),
    ("Normal_003", constants.TAD_NORMAL_DIRNAME, False),
    ("Normal_004", constants.TAD_NORMAL_DIRNAME, False),
)
ANNOTATED = tuple(c for c in CLIPS if c[2])
UNANNOTATED = tuple(c for c in CLIPS if not c[2])


def write_frames(frames_dir: Path, clips=CLIPS) -> Path:
    """TAD's on-disk layout: ``frames/{abnormal,normal}/{id}.mp4/{i:06d}.png``."""
    from torchvision.io import write_png

    generator = torch.Generator().manual_seed(SEED)
    for video_id, split_dirname, _ in clips:
        folder = frames_dir / split_dirname / f"{video_id}.mp4"
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(RAW_FRAMES):
            image = (
                torch.rand(3, FRAME_SIZE, FRAME_SIZE, generator=generator) * 255
            ).to(torch.uint8)
            write_png(image, str(folder / f"{index:06d}.png"))
    return frames_dir


def write_annotation(path: Path) -> Path:
    """The shipped test-annotation shape: normalized spans, one entry per video."""
    entries = []
    for video_id, split_dirname, _ in ANNOTATED:
        is_abnormal = split_dirname == constants.TAD_ABNORMAL_DIRNAME
        entries.append(
            {
                "path": f"other_datasets/tad_features/{video_id}.npy",
                "video_path": f"other_datasets/tad_videos/{video_id}.mp4",
                "class_name": (
                    constants.TAD_ABNORMAL_CLASS if is_abnormal else "Normal"
                ),
                "descriptions": None,
                "anomaly_span": [[0.25, 0.75]] if is_abnormal else [],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def tad_tree(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """``(frames_dir, annotation, out_dir)`` — built once, preprocessed per test."""
    root = tmp_path_factory.mktemp("tad")
    frames_dir = write_frames(root / "frames")
    annotation = write_annotation(root / "annotations" / "tad_test_anno.json")
    return frames_dir, annotation, root / "data"


def _load(out_dir: Path, name: str):
    return json.loads((out_dir / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def out_dir(tad_tree, tmp_path_factory) -> Path:
    """Eval-only preprocessing — the pre-P1 default."""
    frames_dir, annotation, _ = tad_tree
    out: Path = tmp_path_factory.mktemp("evalonly")
    tad.preprocess(annotation, frames_dir, out, stride=STRIDE)
    return out


@pytest.fixture(scope="module")
def built(tad_tree, tmp_path_factory):
    """``--with-train-split`` preprocessing: ``(out_dir, train, test)``."""
    frames_dir, annotation, _ = tad_tree
    out = tmp_path_factory.mktemp("withtrain")
    train_records, test_records = tad.preprocess(
        annotation, frames_dir, out, stride=STRIDE, with_train_split=True
    )
    return out, train_records, test_records


class TestEvalOnlyDefaultIsUnchanged:
    """The pre-P1 contract. A TAD eval run must not notice that P1 happened."""

    def test_labels_train_is_empty(self, out_dir: Path) -> None:
        assert _load(out_dir, constants.LABELS_TRAIN_FILENAME) == {}

    def test_only_annotated_videos_are_scored(self, out_dir: Path) -> None:
        labels = _load(out_dir, constants.FRAME_LABELS_TEST_FILENAME)
        assert set(labels) == {c[0] for c in ANNOTATED}

    def test_meta_holds_only_the_test_split(self, out_dir: Path) -> None:
        meta = _load(out_dir, constants.META_FILENAME)
        assert {info["split"] for info in meta.values()} == {tad.TEST_SPLIT}

    def test_no_train_ids_file(self, out_dir: Path) -> None:
        assert (out_dir / TEST_IDS_FILENAME).exists()
        assert not (out_dir / TRAIN_IDS_FILENAME).exists()

    def test_frame_labels_align_with_the_stride_formula(self, out_dir: Path) -> None:
        labels = _load(out_dir, constants.FRAME_LABELS_TEST_FILENAME)
        expected = (RAW_FRAMES + STRIDE - 1) // STRIDE
        assert {len(v) for v in labels.values()} == {expected}
        # abnormal clips keep a positive frame; normal clips have none
        assert any(labels["01_Accident_001"])
        assert not any(labels["Normal_001"])


class TestTrainSplit:
    def test_unannotated_folders_become_the_train_split(self, built) -> None:
        _, train_records, test_records = built
        assert {r.video_id for r in train_records} == {c[0] for c in UNANNOTATED}
        assert {r.video_id for r in test_records} == {c[0] for c in ANNOTATED}
        assert not (
            {r.video_id for r in train_records} & {r.video_id for r in test_records}
        )

    def test_the_directory_is_the_video_level_label(self, built) -> None:
        out, _, _ = built
        labels = _load(out, constants.LABELS_TRAIN_FILENAME)
        for video_id, split_dirname, _ in UNANNOTATED:
            expected = int(split_dirname == constants.TAD_ABNORMAL_DIRNAME)
            assert labels[video_id] == expected, video_id
        assert sum(labels.values()) == 3

    def test_abnormal_train_clips_carry_the_defined_class_name(self, built) -> None:
        out, _, _ = built
        meta = _load(out, constants.META_FILENAME)
        assert meta["01_Accident_003"]["class_name"] == constants.TAD_ABNORMAL_CLASS
        assert meta["Normal_002"]["class_name"] == "Normal"
        assert _load(out, constants.DEFS_FILENAME) == [
            "Normal",
            constants.TAD_ABNORMAL_CLASS,
        ]

    def test_train_meta_carries_no_anomaly_window(self, built) -> None:
        """Weak supervision: TAD's train split has no frame-level annotation."""
        out, _, _ = built
        meta = _load(out, constants.META_FILENAME)
        train_meta = [i for i in meta.values() if i["split"] == tad.TRAIN_SPLIT]
        assert len(train_meta) == len(UNANNOTATED)
        assert all(not info["normalized_spans"] for info in train_meta)

    def test_both_id_files_are_written(self, built) -> None:
        out, _, _ = built
        train_ids = (out / TRAIN_IDS_FILENAME).read_text().split()
        test_ids = (out / TEST_IDS_FILENAME).read_text().split()
        assert set(train_ids) == {c[0] for c in UNANNOTATED}
        assert set(test_ids) == {c[0] for c in ANNOTATED}

    def test_test_split_is_identical_to_the_eval_only_mode(
        self, built, tad_tree, tmp_path: Path
    ) -> None:
        """Adding a train split must not perturb the scored ground truth."""
        out, _, _ = built
        frames_dir, annotation, _ = tad_tree
        tad.preprocess(annotation, frames_dir, tmp_path, stride=STRIDE)
        assert _load(out, constants.FRAME_LABELS_TEST_FILENAME) == _load(
            tmp_path, constants.FRAME_LABELS_TEST_FILENAME
        )


class TestFailsLoud:
    def test_folder_outside_either_split_directory_raises(
        self, tad_tree, tmp_path: Path
    ) -> None:
        """No default bucket: an unlabelled anomaly in the normal half is silent."""
        _, annotation, _ = tad_tree
        stray = tmp_path / "frames"
        write_frames(stray, CLIPS)
        orphan = stray / "unsorted" / "01_Accident_099.mp4"
        orphan.mkdir(parents=True)
        (orphan / "000000.png").write_bytes(
            (stray / constants.TAD_NORMAL_DIRNAME / "Normal_001.mp4" / "000000.png")
            .read_bytes()
        )
        with pytest.raises(ValueError, match="neither of the split directories"):
            tad.preprocess(
                annotation, stray, tmp_path / "out", stride=STRIDE, with_train_split=True
            )

    def test_single_class_train_split_raises(self, tmp_path: Path) -> None:
        """DVSFeatureDataset raises on this later and far less clearly."""
        frames_dir = tmp_path / "frames"
        abnormal_only = tuple(
            c for c in CLIPS if c[1] == constants.TAD_ABNORMAL_DIRNAME
        ) + tuple(c for c in CLIPS if c[1] == constants.TAD_NORMAL_DIRNAME and c[2])
        write_frames(frames_dir, abnormal_only)
        annotation = write_annotation(tmp_path / "anno.json")
        with pytest.raises(ValueError, match="needs both classes"):
            tad.preprocess(
                annotation, frames_dir, tmp_path / "out", stride=STRIDE,
                with_train_split=True,
            )

    def test_undefined_class_name_raises(self, tad_tree, tmp_path: Path) -> None:
        """Lesson 19: the verbalizer's fallback would swallow this silently."""
        frames_dir, _, _ = tad_tree
        entries = json.loads(
            write_annotation(tmp_path / "anno.json").read_text(encoding="utf-8")
        )
        entries[0]["class_name"] = "Vehicle Fire"
        (tmp_path / "anno.json").write_text(json.dumps(entries), encoding="utf-8")
        with pytest.raises(ValueError, match="no definition sentences"):
            tad.preprocess(
                tmp_path / "anno.json", frames_dir, tmp_path / "out", stride=STRIDE
            )

    def test_annotated_video_without_frames_raises(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        write_frames(frames_dir, tuple(c for c in CLIPS if c[0] != "01_Accident_001"))
        annotation = write_annotation(tmp_path / "anno.json")
        with pytest.raises(ValueError, match="have no frame folder"):
            tad.preprocess(annotation, frames_dir, tmp_path / "out", stride=STRIDE)

    def test_dry_run_writes_nothing(self, tad_tree, tmp_path: Path) -> None:
        frames_dir, annotation, _ = tad_tree
        out = tmp_path / "dry"
        tad.preprocess(
            annotation, frames_dir, out, stride=STRIDE, with_train_split=True,
            dry_run=True,
        )
        assert not out.exists()


class TestCli:
    def test_with_train_split_flag_reaches_the_labels(
        self, tad_tree, tmp_path: Path
    ) -> None:
        frames_dir, annotation, _ = tad_tree
        out = tmp_path / "cli"
        tad.main([
            "--annotation", str(annotation),
            "--frames-dir", str(frames_dir),
            "--out-dir", str(out),
            "--stride", str(STRIDE),
            "--with-train-split",
        ])
        assert sum(_load(out, constants.LABELS_TRAIN_FILENAME).values()) == 3

    def test_default_cli_stays_eval_only(self, tad_tree, tmp_path: Path) -> None:
        frames_dir, annotation, _ = tad_tree
        out = tmp_path / "cli_default"
        tad.main([
            "--annotation", str(annotation),
            "--frames-dir", str(frames_dir),
            "--out-dir", str(out),
            "--stride", str(STRIDE),
        ])
        assert _load(out, constants.LABELS_TRAIN_FILENAME) == {}


# ---------------------------------------------------------------------------
# The point of P1: these files have to train, under every gate in the enum.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trainable_tad(tad_tree, tmp_path_factory) -> dict[str, Path]:
    """Real label files + synthetic caches, in the on-disk contract's shape."""
    frames_dir, annotation, _ = tad_tree
    root = tmp_path_factory.mktemp("tad_train")
    data_dir = root / "data" / constants.TAD_DATASET
    clip_dir = root / "cache" / "clip" / constants.TAD_DATASET
    flow_root = root / "cache" / "flow" / constants.FLOW_CACHE_VERSION
    flow_dir = flow_root / constants.TAD_DATASET
    for directory in (clip_dir, flow_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_records, test_records = tad.preprocess(
        annotation, frames_dir, data_dir, stride=STRIDE, with_train_split=True
    )

    from core.flow.raft_extract import make_projection, save_projection

    projection = make_projection()
    save_projection(
        flow_root / constants.FLOW_PROJECTION_FILENAME,
        projection,
        constants.FLOW_PROJECTION_SEED,
    )
    rng = np.random.default_rng(SEED)
    # flow is a fixed function of the features (+ noise), so the PMG head has
    # something learnable to regress — same construction as the MSAD fixture
    feat_to_stats = rng.standard_normal(
        (constants.CLIP_FEATURE_DIM, constants.FLOW_STATS_DIM)
    ) / np.sqrt(constants.CLIP_FEATURE_DIM)
    length = (RAW_FRAMES + STRIDE - 1) // STRIDE
    video_ids = [r.video_id for r in train_records] + [r.video_id for r in test_records]
    for video_id in video_ids:
        features = rng.standard_normal((length, constants.CLIP_FEATURE_DIM))
        np.save(clip_dir / f"{video_id}.npy", features.astype(np.float32))
        stats = np.abs(
            features @ feat_to_stats
            + 0.05 * rng.standard_normal((length, constants.FLOW_STATS_DIM))
        )
        np.save(
            flow_dir / f"{video_id}{constants.FLOW_STATS_SUFFIX}",
            stats.astype(np.float32),
        )
        np.save(flow_dir / f"{video_id}.npy", (stats @ projection).astype(np.float32))

    knn_cache_path = root / "cache" / "knn" / constants.TAD_DATASET / constants.KNN_CACHE_FILENAME
    build_knn_cache(
        labels={r.video_id: int(r.is_abnormal) for r in train_records},
        clip_dir=clip_dir,
        output_path=knn_cache_path,
        k=2,
    )
    return {
        "data_dir": data_dir,
        "clip_dir": clip_dir,
        "flow_dir": flow_dir,
        "knn_cache": knn_cache_path,
    }


def _train_argv(paths: dict[str, Path], out_dir: Path, extra: list[str]) -> list[str]:
    return [
        "--data-dir", str(paths["data_dir"]),
        "--clip-dir", str(paths["clip_dir"]),
        "--flow-dir", str(paths["flow_dir"]),
        "--knn-cache", str(paths["knn_cache"]),
        "--output-dir", str(out_dir),
        "--text-encoder", "stub",
        "--set", f"data.dataset={constants.TAD_DATASET}",
        "--set", "data.is_egocentric=false",
        "--set", "train.num_epochs=1",
        "--set", "train.batch_size=4",
        "--set", "train.device=cpu",
        "--set", "loss.captions_from_definitions=true",
        *extra,
    ]


# `main` is KAT-VAD v1 and ships exactly one gate: the frozen MLP over the flow
# norm. The v3 gate matrix (`kip.gate_type` in {rank, mlp_frozen, mlp_ste,
# constant}) lives on branch `v3`; `core/config.py` here has no such field and
# raises on it by design, so parametrizing over it belongs on that branch.
V1_KIP_ARM = ["--set", "kip.gate_signal=flow_norm"]


class TestTadTrains:
    def test_stage2_trains(self, trainable_tad, tmp_path: Path) -> None:
        out = tmp_path / "v1"
        train.main(_train_argv(trainable_tad, out, V1_KIP_ARM))

        assert (out / train.CHECKPOINT_LAST).exists()
        records = [
            json.loads(line)
            for line in (out / train.METRICS_FILENAME)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert records, "no metrics logged"
        for record in records:
            for key, value in record.items():
                if key not in ("epoch", "batch", "global_step"):
                    assert np.isfinite(value), f"{key} not finite: {record}"

    def test_stage1_warmup_runs(self, trainable_tad, tmp_path: Path) -> None:
        """Stage 1 is the precondition for A1/A2b/A3/A4 and needs the flow cache."""
        out = tmp_path / "v1_stage1"
        train.main(
            _train_argv(trainable_tad, out, [*V1_KIP_ARM, "--set", "train.stage=1"])
        )
        assert (out / train.CHECKPOINT_LAST).exists()

    def test_kip_off_trains(self, trainable_tad, tmp_path: Path) -> None:
        """Arm A0 — the baseline every delta subtracts from."""
        out = tmp_path / "a0"
        train.main(
            _train_argv(trainable_tad, out, ["--set", "kip.enabled=false"])
        )
        assert (out / train.CHECKPOINT_LAST).exists()

    def test_the_config_records_the_gate(self, trainable_tad, tmp_path: Path) -> None:
        """`config.yaml` is the only durable record of which arm a run was."""
        import yaml

        out = tmp_path / "recorded"
        train.main(_train_argv(trainable_tad, out, V1_KIP_ARM))
        cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["kip"]["gate_signal"] == "flow_norm"
        assert cfg["data"]["dataset"] == constants.TAD_DATASET
