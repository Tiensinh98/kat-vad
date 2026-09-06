"""Tests for the DADA-2000 preprocessor (core.data.dada).

Covers the three record kinds (annotated abnormal, weak abnormal, normal), the
seeded train/test split, the CSV/frame-folder join and its failure modes, and
that the files it writes actually train under every KIP gate in the enum --
the same discipline ``core/tests/test_tad.py`` applies to TAD.

No downloads: tiny PNG frame folders, a synthetic CSV, synthetic CLIP/flow
caches, stub text encoder, CPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants, train
from core.data import dada
from core.data.dataset_files import TEST_IDS_FILENAME, TRAIN_IDS_FILENAME
from core.data.knn_cache import build_knn_cache

torch.set_num_threads(1)

STRIDE = 2
RAW_FRAMES = 24  # -> 12 sampled frames
FRAME_SIZE = 8
SEED = 11

CSV_COLUMNS = (
    "video", "weather(sunny,rainy,snowy,foggy)1-4", "light(day,night)1-2",
    "scenes(highway,tunnel,mountain,urban,rural)1-5",
    "linear(arterials,curve,intersection,t-junction,ramp) 1-5", "type",
    "whether an accident occurred (1/0)", "abnormal start frame", "accident frame",
    "abnormal end frame", "total frames", "[0,tai]", "[tai,tco]", "[tai,tae]",
    "[tco,tae]", "[tae,end]", "texts", "causes", "measures", "Fault_Label",
)

# (fault_dirname, type, vid, annotated?) -- 4 annotated + 1 weak per fault dir
NON_EGO = [
    (constants.DADA_NON_EGO_FAULT_DIRNAME, 1, v, v <= 4) for v in range(1, 6)
]
EGO = [(constants.DADA_EGO_FAULT_DIRNAME, 2, v, v <= 4) for v in range(1, 6)]
NORMAL = [(constants.DADA_NORMAL_DIRNAME, 3, v, False) for v in range(1, 7)]
ABNORMAL_FOLDERS = NON_EGO + EGO
ANNOTATED = [c for c in ABNORMAL_FOLDERS if c[3]]
WEAK = [c for c in ABNORMAL_FOLDERS if not c[3]]


def _folder_name(type_id: int, vid: int) -> str:
    return f"type{type_id}_vid{vid:03d}"


def _video_id(fault_dir: str, type_id: int, vid: int) -> str:
    return f"{fault_dir}{constants.DADA_ID_SEPARATOR}{_folder_name(type_id, vid)}"


def write_frames(frames_dir: Path, folders=ABNORMAL_FOLDERS + NORMAL) -> Path:
    from torchvision.io import write_png

    generator = torch.Generator().manual_seed(SEED)
    for fault_dir, type_id, vid, _ in folders:
        folder = frames_dir / fault_dir / _folder_name(type_id, vid)
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(RAW_FRAMES):
            image = (torch.rand(3, FRAME_SIZE, FRAME_SIZE, generator=generator) * 255).to(
                torch.uint8
            )
            write_png(image, str(folder / f"{index:06d}.png"))
    return frames_dir


def write_metadata_csv(path: Path, rows=ANNOTATED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for fault_dir, type_id, vid, _ in rows:
            writer.writerow(
                {
                    "video": vid,
                    "weather(sunny,rainy,snowy,foggy)1-4": 1,
                    "light(day,night)1-2": 1,
                    "scenes(highway,tunnel,mountain,urban,rural)1-5": 1,
                    "linear(arterials,curve,intersection,t-junction,ramp) 1-5": 1,
                    "type": type_id,
                    "whether an accident occurred (1/0)": 1,
                    "abnormal start frame": 6,
                    "accident frame": 12,
                    "abnormal end frame": 18,
                    "total frames": RAW_FRAMES,
                    "[0,tai]": 6, "[tai,tco]": 6, "[tai,tae]": 6, "[tco,tae]": 12,
                    "[tae,end]": 6,
                    "texts": "[CLS]x[SEP]", "causes": "c", "measures": "m",
                    "Fault_Label": fault_dir,
                }
            )
    return path


@pytest.fixture(scope="module")
def dada_tree(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("dada")
    frames_dir = write_frames(root / "frames")
    metadata = write_metadata_csv(root / "Cleaned_Metadata.csv")
    return frames_dir, metadata


def _load(out_dir: Path, name: str):
    return json.loads((out_dir / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built(dada_tree, tmp_path_factory):
    frames_dir, metadata = dada_tree
    out = tmp_path_factory.mktemp("out")
    train_records, test_records = dada.preprocess(
        metadata, frames_dir, out, stride=STRIDE, seed=SEED,
        test_ratio_abnormal=0.25, test_ratio_normal=1 / 3,
    )
    return out, train_records, test_records


class TestRecordResolution:
    def test_all_folders_are_accounted_for(self, built) -> None:
        _, tr, te = built
        ids = {r.video_id for r in tr} | {r.video_id for r in te}
        expected = {_video_id(fd, t, v) for fd, t, v, _ in ABNORMAL_FOLDERS + NORMAL}
        assert ids == expected

    def test_weak_abnormal_is_always_train(self, built) -> None:
        _, tr, te = built
        train_ids = {r.video_id for r in tr}
        test_ids = {r.video_id for r in te}
        for fault_dir, type_id, vid, _ in WEAK:
            vid_id = _video_id(fault_dir, type_id, vid)
            assert vid_id in train_ids
            assert vid_id not in test_ids

    def test_no_leakage(self, built) -> None:
        _, tr, te = built
        assert not ({r.video_id for r in tr} & {r.video_id for r in te})

    def test_annotated_span_is_normalized(self, built) -> None:
        _, tr, te = built
        fault_dirs = (constants.DADA_NON_EGO_FAULT_DIRNAME, constants.DADA_EGO_FAULT_DIRNAME)
        for r in tr + te:
            if r.fault_label in fault_dirs:
                is_weak = r.span is None
                is_annotated_weak = any(
                    _video_id(fd, t, v) == r.video_id for fd, t, v, ok in WEAK if not ok
                )
                assert is_weak == is_annotated_weak or not is_weak
        annotated_test = [r for r in te if r.is_abnormal]
        for r in annotated_test:
            assert r.span == pytest.approx((6 / RAW_FRAMES, 18 / RAW_FRAMES))
            assert r.accident_frac == pytest.approx(12 / RAW_FRAMES)

    def test_ego_flag_matches_fault_dir(self, built) -> None:
        _, tr, te = built
        for r in tr + te:
            assert r.is_ego == (r.fault_label == constants.DADA_EGO_FAULT_DIRNAME)

    def test_test_split_has_frame_level_labels(self, built) -> None:
        out, _, te = built
        labels = _load(out, constants.FRAME_LABELS_TEST_FILENAME)
        assert set(labels) == {r.video_id for r in te}
        for r in te:
            if r.is_abnormal:
                assert any(labels[r.video_id])
            else:
                assert not any(labels[r.video_id])

    def test_labels_train_is_video_level_only(self, built) -> None:
        out, tr, _ = built
        labels = _load(out, constants.LABELS_TRAIN_FILENAME)
        assert set(labels) == {r.video_id for r in tr}
        assert all(v in (0, 1) for v in labels.values())
        for r in tr:
            assert labels[r.video_id] == int(r.is_abnormal)

    def test_meta_never_used_as_supervision_but_carries_train_windows(self, built) -> None:
        """meta.json may carry a known train-split window; labels_train.json never does."""
        out, tr, _ = built
        meta = _load(out, constants.META_FILENAME)
        annotated_train = [
            r for r in tr
            if r.span is not None
        ]
        assert annotated_train, "fixture should split some annotated rows into train"
        for r in annotated_train:
            assert meta[r.video_id]["normalized_span"] is not None
        labels = _load(out, constants.LABELS_TRAIN_FILENAME)
        assert set(labels.values()) <= {0, 1}

    def test_defs_and_ids_files(self, built) -> None:
        out, tr, te = built
        assert _load(out, constants.DEFS_FILENAME) == ["Normal", constants.DADA_CLASS_NAME]
        assert set((out / TRAIN_IDS_FILENAME).read_text().split()) == {r.video_id for r in tr}
        assert set((out / TEST_IDS_FILENAME).read_text().split()) == {r.video_id for r in te}


class TestFailsLoud:
    def test_unknown_fault_label_raises(self, tmp_path: Path) -> None:
        frames_dir = write_frames(tmp_path / "frames")
        bad_row = [(NON_EGO[0][0], NON_EGO[0][1], NON_EGO[0][2], True)]
        metadata = write_metadata_csv(tmp_path / "meta.csv", bad_row)
        rows = metadata.read_text(encoding="utf-8").replace(
            constants.DADA_NON_EGO_FAULT_DIRNAME, "Bogus_Label"
        )
        metadata.write_text(rows, encoding="utf-8")
        with pytest.raises(ValueError, match="not one of"):
            dada.preprocess(metadata, frames_dir, tmp_path / "out", stride=STRIDE)

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "meta.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["video", "type"])
            writer.writeheader()
            writer.writerow({"video": 1, "type": 1})
        with pytest.raises(ValueError, match="missing required columns"):
            dada.parse_metadata_csv(path)

    def test_bad_span_ordering_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "meta.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(
                {
                    "video": 1, "weather(sunny,rainy,snowy,foggy)1-4": 1,
                    "light(day,night)1-2": 1,
                    "scenes(highway,tunnel,mountain,urban,rural)1-5": 1,
                    "linear(arterials,curve,intersection,t-junction,ramp) 1-5": 1,
                    "type": 1, "whether an accident occurred (1/0)": 1,
                    "abnormal start frame": 20, "accident frame": 21,
                    "abnormal end frame": 10, "total frames": RAW_FRAMES,
                    "[0,tai]": 0, "[tai,tco]": 0, "[tai,tae]": 0, "[tco,tae]": 0,
                    "[tae,end]": 0, "texts": "x", "causes": "c", "measures": "m",
                    "Fault_Label": constants.DADA_NON_EGO_FAULT_DIRNAME,
                }
            )
        with pytest.raises(ValueError, match=r"window \[20,10\]"):
            dada.parse_metadata_csv(path)

    def test_annotated_without_frames_raises(self, dada_tree, tmp_path: Path) -> None:
        _, metadata = dada_tree
        frames_dir = write_frames(
            tmp_path / "frames", [c for c in ABNORMAL_FOLDERS + NORMAL if c[1:3] != (1, 1)]
        )
        with pytest.raises(ValueError, match="no readable frame folder"):
            dada.preprocess(metadata, frames_dir, tmp_path / "out", stride=STRIDE)

    def test_allow_missing_frames_drops_instead_of_raising(self, dada_tree, tmp_path: Path) -> None:
        _, metadata = dada_tree
        frames_dir = write_frames(
            tmp_path / "frames", [c for c in ABNORMAL_FOLDERS + NORMAL if c[1:3] != (1, 1)]
        )
        train_records, test_records = dada.preprocess(
            metadata, frames_dir, tmp_path / "out", stride=STRIDE, allow_missing_frames=True,
        )
        missing_id = _video_id(constants.DADA_NON_EGO_FAULT_DIRNAME, 1, 1)
        assert missing_id not in {r.video_id for r in train_records + test_records}

    def test_folder_name_not_matching_pattern_raises(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        stray = frames_dir / constants.DADA_NON_EGO_FAULT_DIRNAME / "not_a_valid_name"
        stray.mkdir(parents=True)
        (stray / "000000.png").write_bytes(b"\x89PNG\r\n")
        with pytest.raises(ValueError, match="does not match"):
            dada.frame_folders_by_type_vid(frames_dir / constants.DADA_NON_EGO_FAULT_DIRNAME)

    def test_colliding_bare_folder_names_are_disambiguated(self, tmp_path: Path) -> None:
        frames_dir = write_frames(tmp_path / "frames")
        # a normal folder whose bare name collides with an existing non-ego-fault one --
        # confirmed on the real archive, not a hypothetical (video_id is fault-dir-prefixed
        # precisely so this resolves instead of raising)
        clash = frames_dir / constants.DADA_NORMAL_DIRNAME / _folder_name(1, 1)
        clash.mkdir(parents=True, exist_ok=True)
        from torchvision.io import write_png
        blank = torch.zeros(3, FRAME_SIZE, FRAME_SIZE, dtype=torch.uint8)
        write_png(blank, str(clash / "000000.png"))
        metadata = write_metadata_csv(tmp_path / "meta.csv")
        train_records, test_records = dada.preprocess(
            metadata, frames_dir, tmp_path / "out", stride=STRIDE,
        )
        ids = {r.video_id for r in train_records + test_records}
        fault_id = _video_id(constants.DADA_NON_EGO_FAULT_DIRNAME, 1, 1)
        normal_id = _video_id(constants.DADA_NORMAL_DIRNAME, 1, 1)
        assert fault_id in ids
        assert normal_id in ids
        assert fault_id != normal_id

    def test_undefined_class_name_raises(self, monkeypatch, dada_tree, tmp_path: Path) -> None:
        frames_dir, metadata = dada_tree
        monkeypatch.setattr(dada, "DATASET_CLS_DEFS", {"dada": {"Normal": ["Normal"]}})
        with pytest.raises(ValueError, match="no definition sentences"):
            dada.preprocess(metadata, frames_dir, tmp_path / "out", stride=STRIDE)

    def test_strict_raises_on_vanished_window(self, tmp_path: Path) -> None:
        # a window narrow enough (1% of the clip) to round away at any stride
        frames_dir = write_frames(tmp_path / "frames")
        narrow = [(NON_EGO[0][0], NON_EGO[0][1], NON_EGO[0][2], True)]
        path = tmp_path / "meta.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for fault_dir, type_id, vid, _ in narrow:
                writer.writerow(
                    {
                        "video": vid, "weather(sunny,rainy,snowy,foggy)1-4": 1,
                        "light(day,night)1-2": 1,
                        "scenes(highway,tunnel,mountain,urban,rural)1-5": 1,
                        "linear(arterials,curve,intersection,t-junction,ramp) 1-5": 1,
                        "type": type_id, "whether an accident occurred (1/0)": 1,
                        "abnormal start frame": 10, "accident frame": 15,
                        "abnormal end frame": 20, "total frames": 1000,
                        "[0,tai]": 0, "[tai,tco]": 0, "[tai,tae]": 0, "[tco,tae]": 0,
                        "[tae,end]": 0, "texts": "x", "causes": "c", "measures": "m",
                        "Fault_Label": fault_dir,
                    }
                )
        with pytest.raises(ValueError, match="lose their anomaly window"):
            dada.preprocess(
                path, frames_dir, tmp_path / "out", stride=STRIDE, strict=True,
                test_ratio_abnormal=1.0, test_ratio_normal=1.0,
            )

    def test_dry_run_writes_nothing(self, dada_tree, tmp_path: Path) -> None:
        frames_dir, metadata = dada_tree
        out = tmp_path / "dry"
        dada.preprocess(metadata, frames_dir, out, stride=STRIDE, dry_run=True)
        assert not out.exists()

    def test_exclude_unannotated_abnormal_drops_weak_records(
        self, dada_tree, tmp_path: Path
    ) -> None:
        frames_dir, metadata = dada_tree
        tr, te = dada.preprocess(
            metadata, frames_dir, tmp_path / "out", stride=STRIDE,
            exclude_unannotated_abnormal=True,
        )
        weak_ids = {_video_id(fd, t, v) for fd, t, v, ok in WEAK if not ok}
        assert not (weak_ids & {r.video_id for r in tr + te})


class TestFlatFramesDir:
    def test_symlinks_resolve_to_the_real_folder(self, dada_tree, tmp_path: Path) -> None:
        frames_dir, metadata = dada_tree
        flat = tmp_path / "flat"
        train_records, test_records = dada.preprocess(
            metadata, frames_dir, tmp_path / "out", stride=STRIDE, flat_frames_dir=flat,
        )
        for r in train_records + test_records:
            link = flat / r.video_id
            assert link.is_symlink()
            assert link.resolve() == (frames_dir / r.fault_label / r.folder_name).resolve()

    def test_disambiguates_colliding_bare_names(self, tmp_path: Path) -> None:
        frames_dir = write_frames(tmp_path / "frames")
        clash = frames_dir / constants.DADA_NORMAL_DIRNAME / _folder_name(1, 1)
        clash.mkdir(parents=True, exist_ok=True)
        from torchvision.io import write_png
        blank = torch.zeros(3, FRAME_SIZE, FRAME_SIZE, dtype=torch.uint8)
        write_png(blank, str(clash / "000000.png"))
        metadata = write_metadata_csv(tmp_path / "meta.csv")
        flat = tmp_path / "flat"
        dada.preprocess(
            metadata, frames_dir, tmp_path / "out", stride=STRIDE, flat_frames_dir=flat,
        )
        fault_id = _video_id(constants.DADA_NON_EGO_FAULT_DIRNAME, 1, 1)
        normal_id = _video_id(constants.DADA_NORMAL_DIRNAME, 1, 1)
        assert (flat / fault_id).resolve() == (
            frames_dir / constants.DADA_NON_EGO_FAULT_DIRNAME / _folder_name(1, 1)
        ).resolve()
        assert (flat / normal_id).resolve() == clash.resolve()

    def test_dry_run_materializes_nothing(self, dada_tree, tmp_path: Path) -> None:
        frames_dir, metadata = dada_tree
        flat = tmp_path / "flat"
        dada.preprocess(
            metadata, frames_dir, tmp_path / "out", stride=STRIDE,
            flat_frames_dir=flat, dry_run=True,
        )
        assert not flat.exists()


class TestCli:
    def test_cli_reaches_the_labels(self, dada_tree, tmp_path: Path) -> None:
        frames_dir, metadata = dada_tree
        out = tmp_path / "cli"
        dada.main([
            "--metadata", str(metadata),
            "--frames-dir", str(frames_dir),
            "--out-dir", str(out),
            "--stride", str(STRIDE),
        ])
        assert (out / constants.LABELS_TRAIN_FILENAME).exists()
        assert (out / constants.FRAME_LABELS_TEST_FILENAME).exists()


# ---------------------------------------------------------------------------
# The files dada.py writes have to train, under every gate in the enum.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trainable_dada(dada_tree, tmp_path_factory) -> dict[str, Path]:
    frames_dir, metadata = dada_tree
    root = tmp_path_factory.mktemp("dada_train")
    data_dir = root / "data" / constants.DADA_DATASET
    clip_dir = root / "cache" / "clip" / constants.DADA_DATASET
    flow_root = root / "cache" / "flow" / constants.FLOW_CACHE_VERSION
    flow_dir = flow_root / constants.DADA_DATASET
    for directory in (clip_dir, flow_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_records, test_records = dada.preprocess(
        metadata, frames_dir, data_dir, stride=STRIDE, seed=SEED,
        test_ratio_abnormal=0.25, test_ratio_normal=1 / 3,
    )

    from core.flow.raft_extract import make_projection, save_projection

    projection = make_projection()
    save_projection(
        flow_root / constants.FLOW_PROJECTION_FILENAME, projection, constants.FLOW_PROJECTION_SEED,
    )
    rng = np.random.default_rng(SEED)
    feat_to_stats = rng.standard_normal(
        (constants.CLIP_FEATURE_DIM, constants.FLOW_STATS_DIM)
    ) / np.sqrt(constants.CLIP_FEATURE_DIM)
    length = (RAW_FRAMES + STRIDE - 1) // STRIDE
    video_ids = [r.video_id for r in train_records] + [r.video_id for r in test_records]
    for video_id in video_ids:
        features = rng.standard_normal((length, constants.CLIP_FEATURE_DIM))
        np.save(clip_dir / f"{video_id}.npy", features.astype(np.float32))
        noise = 0.05 * rng.standard_normal((length, constants.FLOW_STATS_DIM))
        stats = np.abs(features @ feat_to_stats + noise)
        np.save(flow_dir / f"{video_id}{constants.FLOW_STATS_SUFFIX}", stats.astype(np.float32))
        np.save(flow_dir / f"{video_id}.npy", (stats @ projection).astype(np.float32))

    knn_cache_path = (
        root / "cache" / "knn" / constants.DADA_DATASET / constants.KNN_CACHE_FILENAME
    )
    build_knn_cache(
        labels={r.video_id: int(r.is_abnormal) for r in train_records},
        clip_dir=clip_dir,
        output_path=knn_cache_path,
        k=2,
    )
    return {
        "data_dir": data_dir, "clip_dir": clip_dir, "flow_dir": flow_dir,
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
        "--set", f"data.dataset={constants.DADA_DATASET}",
        "--set", "data.is_egocentric=true",
        "--set", "train.num_epochs=1",
        "--set", "train.batch_size=4",
        "--set", "train.device=cpu",
        "--set", "loss.captions_from_definitions=true",
        *extra,
    ]


GATE_ARMS = {
    "v1_mlp_frozen": ["--set", "kip.gate_type=mlp_frozen", "--set", "kip.gate_signal=flow_norm"],
    "v1_mlp_ste": ["--set", "kip.gate_type=mlp_ste", "--set", "kip.gate_signal=flow_norm"],
    "v3_rank": ["--set", "kip.gate_type=rank"],
    "v3_constant": ["--set", "kip.gate_type=constant", "--set", "kip.const_shift_ratio=0.5"],
}


class TestDadaTrainsUnderEveryGate:
    @pytest.mark.parametrize("arm", sorted(GATE_ARMS))
    def test_stage2_trains(self, trainable_dada, tmp_path: Path, arm: str) -> None:
        out = tmp_path / arm
        train.main(_train_argv(trainable_dada, out, GATE_ARMS[arm]))

        assert (out / train.CHECKPOINT_LAST).exists()
        records = [
            json.loads(line)
            for line in (out / train.METRICS_FILENAME).read_text(encoding="utf-8").splitlines()
        ]
        assert records, "no metrics logged"
        for record in records:
            for key, value in record.items():
                if key not in ("epoch", "batch", "global_step"):
                    assert np.isfinite(value), f"{key} not finite in {arm}: {record}"

    def test_kip_off_trains(self, trainable_dada, tmp_path: Path) -> None:
        out = tmp_path / "a0"
        train.main(_train_argv(trainable_dada, out, ["--set", "kip.enabled=false"]))
        assert (out / train.CHECKPOINT_LAST).exists()

    def test_the_config_records_the_dataset(self, trainable_dada, tmp_path: Path) -> None:
        import yaml

        out = tmp_path / "recorded"
        train.main(_train_argv(trainable_dada, out, GATE_ARMS["v3_rank"]))
        cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["kip"]["gate_type"] == "rank"
        assert cfg["data"]["dataset"] == constants.DADA_DATASET
        assert cfg["data"]["is_egocentric"] is True
