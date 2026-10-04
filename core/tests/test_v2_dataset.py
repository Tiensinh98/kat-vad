"""The v2 dataset dir (proposal §7.2): train on T2-train minus T2-val, evaluate T2-val.

The failures that matter are silent: a T2-val window left in training (the pilot reads
T2-val in-sample), T2-test still in the evaluation file, or T2-val labelled by an arithmetic
that differs from the one that labelled T2-test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import constants
from core.data.dada import DadaRecord, sampled_frame_labels
from core.data.dataset_files import write_dataset_files, write_windows
from core.data.v2_splits import lines_sha1
from core.data.windows import Window
from core.tools import v2_dataset

N_TYPES = 3
TRAIN_PER_TYPE = 6
TEST_PER_TYPE = 2
TOTAL_FRAMES = 800  # 100 rows at stride 8
WINDOW = 20
HOP = 8
SPAN = (0.40, 0.50)


def _source_labels(total: int, span: tuple[float, float]) -> list[int]:
    record = DadaRecord("s", "s", "CarAccident", "origin", total, span, span[1] - span[0])
    return sampled_frame_labels(record, constants.FRAME_STRIDE)


def _build(root: Path) -> tuple[Path, list[str]]:
    windows: dict[str, Window] = {}
    labels: dict[str, int] = {}
    frame_labels: dict[str, list[int]] = {}
    meta: dict[str, dict[str, object]] = {}
    val: list[str] = []
    for t in range(N_TYPES):
        for v in range(TRAIN_PER_TYPE + TEST_PER_TYPE):
            source = f"t{t:02d}_v{v:03d}"
            is_test = v >= TRAIN_PER_TYPE
            if v == 0:
                val.append(source)
            rows = _source_labels(TOTAL_FRAMES, SPAN)
            for w, start in enumerate(range(0, len(rows) - WINDOW + 1, HOP)):
                wid = f"{source}{constants.WINDOW_ID_SEPARATOR}w{w:03d}"
                windows[wid] = Window(source, start, start + WINDOW)
                meta[wid] = {
                    "source": source, "class_name": "CarAccident", "type": t,
                    "split": "test" if is_test else "train", "fault_label": "origin",
                    "total_frames": TOTAL_FRAMES, "normalized_span": list(SPAN),
                    "accident_frac": SPAN[1] - SPAN[0],
                    "start": start, "end": start + WINDOW, "sampled_frames": WINDOW,
                }
                window_labels = rows[start : start + WINDOW]
                if is_test:
                    frame_labels[wid] = window_labels
                else:
                    labels[wid] = int(any(window_labels))
    data_dir = root / "T2"
    write_dataset_files(data_dir, labels, frame_labels, ["Normal", "CarAccident"], meta)
    write_windows(data_dir, windows)
    split_dir = root / "splits"
    split_dir.mkdir()
    (split_dir / "t2_val_sources.txt").write_text("".join(f"{s}\n" for s in sorted(val)))
    manifest = {"splits": {constants.V2_SPLIT_T2_VAL: {"count": len(val), "sha1": lines_sha1(val)}}}
    (split_dir / constants.V2_SPLITS_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return data_dir, sorted(val)


def _read(path: Path) -> dict:
    payload: dict = json.loads(path.read_text(encoding="utf-8"))
    return payload


class TestV2Dataset:
    def test_val_leaves_training_and_becomes_the_eval_set(self, tmp_path: Path) -> None:
        data_dir, val = _build(tmp_path)
        out = tmp_path / "v2"
        manifest = v2_dataset.make_v2_dataset(data_dir, out, tmp_path / "splits")
        train = _read(out / constants.LABELS_TRAIN_FILENAME)
        test = _read(out / constants.FRAME_LABELS_TEST_FILENAME)
        source = {w: w.split(constants.WINDOW_ID_SEPARATOR)[0] for w in (*train, *test)}
        assert not {source[w] for w in train} & set(val)
        assert {source[w] for w in test} == set(val)  # T2-test is gone from evaluation
        assert manifest["train_sources"] == N_TYPES * (TRAIN_PER_TYPE - 1)
        assert manifest["label_arithmetic_checked_on_parent_test_windows"] > 0
        assert (out / v2_dataset.MANIFEST_FILENAME).exists()
        assert not (out / constants.SUBSET_MANIFEST_FILENAME).exists()

    def test_the_unseeded_split_logs_cleanly(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The v2 dir has no seed (the split is frozen); the shared subset log must format it."""
        data_dir, _ = _build(tmp_path)
        with caplog.at_level("INFO", logger="core.tools.subset_train"):
            v2_dataset.make_v2_dataset(data_dir, tmp_path / "v2", tmp_path / "splits")
        assert "seed None" in caplog.text

    def test_val_frame_labels_match_the_window_labels(self, tmp_path: Path) -> None:
        data_dir, _ = _build(tmp_path)
        out = tmp_path / "v2"
        v2_dataset.make_v2_dataset(data_dir, out, tmp_path / "splits")
        parent = _read(data_dir / constants.LABELS_TRAIN_FILENAME)
        for wid, frames in _read(out / constants.FRAME_LABELS_TEST_FILENAME).items():
            assert len(frames) == WINDOW
            assert int(any(frames)) == parent[wid]

    def test_a_label_arithmetic_drift_is_refused(self, tmp_path: Path) -> None:
        data_dir, _ = _build(tmp_path)
        path = data_dir / constants.FRAME_LABELS_TEST_FILENAME
        labels = _read(path)
        first = sorted(labels)[0]
        labels[first] = [1 - x for x in labels[first]]
        path.write_text(json.dumps(labels))
        with pytest.raises(ValueError, match="not reproduced"):
            v2_dataset.make_v2_dataset(data_dir, tmp_path / "v2", tmp_path / "splits")
