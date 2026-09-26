"""Tests for the train-source subset tool (plan katvad-t2-learning-curve.md §4.2).

The failures that matter are silent ones: a window-level draw that keeps pieces
of every accident, a kept source that also has test windows, a nested draw that
escapes its parent, or a parent file that changes on the way through.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import constants
from core.data.dataset_files import TRAIN_IDS_FILENAME, write_dataset_files, write_windows
from core.data.windows import Window
from core.tools import subset_train

N_TYPES = 4
SOURCES_PER_TYPE = 12
WINDOWS_PER_SOURCE = 4
TEST_SOURCES_PER_TYPE = 3
WINDOW_LEN = 20


def _build_t2(root: Path) -> Path:
    """A T2-shaped dir: per source, 2 abnormal + 2 normal windows; test by source."""
    windows: dict[str, Window] = {}
    labels: dict[str, int] = {}
    frame_labels: dict[str, list[int]] = {}
    meta: dict[str, dict[str, object]] = {}
    train_sources: list[str] = []
    for t in range(N_TYPES):
        for v in range(SOURCES_PER_TYPE + TEST_SOURCES_PER_TYPE):
            source = f"{t:02d}_{v:03d}"
            is_test = v >= SOURCES_PER_TYPE
            if not is_test:
                train_sources.append(source)
            for w in range(WINDOWS_PER_SOURCE):
                wid = f"{source}{constants.WINDOW_ID_SEPARATOR}{w:03d}"
                windows[wid] = Window(source, w * 8, w * 8 + WINDOW_LEN)
                meta[wid] = {"class_name": f"type{t}", "source": source}
                label = int(w >= WINDOWS_PER_SOURCE // 2)
                if is_test:
                    frame_labels[wid] = [label] * WINDOW_LEN
                else:
                    labels[wid] = label
    data_dir = root / "T2"
    write_dataset_files(data_dir, labels, frame_labels, ["Normal", "type0"], meta)
    write_windows(data_dir, windows)
    (data_dir / TRAIN_IDS_FILENAME).write_text("".join(f"{s}\n" for s in train_sources))
    (data_dir / "extra.txt").write_text("untouched\n")
    return data_dir


@pytest.fixture
def t2(tmp_path: Path) -> Path:
    return _build_t2(tmp_path)


def _labels(d: Path) -> dict[str, int]:
    labels: dict[str, int] = json.loads((d / constants.LABELS_TRAIN_FILENAME).read_text())
    return labels


def _sources(d: Path) -> set[str]:
    return {i.split(constants.WINDOW_ID_SEPARATOR)[0] for i in _labels(d)}


class TestDraw:
    def test_fraction_is_per_type_and_by_source(self, t2: Path, tmp_path: Path) -> None:
        out = tmp_path / "f50"
        manifest = subset_train.make_subset(t2, out, 0.5, seed=2024)
        assert manifest["kept_by_type"] == {f"type{t}": 6 for t in range(N_TYPES)}
        # whole sources only: every kept source keeps all its train windows
        labels = _labels(out)
        for source in _sources(out):
            kept = [i for i in labels if i.startswith(source)]
            assert len(kept) == WINDOWS_PER_SOURCE
        assert manifest["items_kept"] == 6 * N_TYPES * WINDOWS_PER_SOURCE

    def test_deterministic_per_seed(self, t2: Path, tmp_path: Path) -> None:
        a = subset_train.make_subset(t2, tmp_path / "a", 0.5, seed=7)
        b = subset_train.make_subset(t2, tmp_path / "b", 0.5, seed=7)
        c = subset_train.make_subset(t2, tmp_path / "c", 0.5, seed=8)
        assert a["kept_sources_sha1"] == b["kept_sources_sha1"]
        assert a["kept_sources_sha1"] != c["kept_sources_sha1"]

    def test_never_keeps_a_test_source(self, t2: Path, tmp_path: Path) -> None:
        out = tmp_path / "f100"
        subset_train.make_subset(t2, out, 1.0, seed=1)
        test_ids = json.loads((t2 / constants.FRAME_LABELS_TEST_FILENAME).read_text())
        test_sources = {i.split(constants.WINDOW_ID_SEPARATOR)[0] for i in test_ids}
        assert not _sources(out) & test_sources
        assert _labels(out) == _labels(t2)

    def test_type_rounding_to_zero_is_reported(self, t2: Path, tmp_path: Path) -> None:
        # leave type3 with a single train source: round(1 * 0.25) == 0
        labels = _labels(t2)
        small = {i: v for i, v in labels.items() if not i.startswith("03_") or
                 i.startswith("03_000")}
        (t2 / constants.LABELS_TRAIN_FILENAME).write_text(json.dumps(small))
        manifest = subset_train.make_subset(t2, tmp_path / "q", 0.25, seed=1)
        assert manifest["dropped_types"] == ["type3"]
        assert manifest["total_by_type"]["type3"] == 1
        assert not any(s.startswith("03_") for s in _sources(tmp_path / "q"))

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_rejects_bad_fraction(self, t2: Path, tmp_path: Path, fraction: float) -> None:
        with pytest.raises(ValueError, match="fraction"):
            subset_train.make_subset(t2, tmp_path / "x", fraction, seed=1)


class TestNesting:
    def test_quarter_is_inside_half(self, t2: Path, tmp_path: Path) -> None:
        half = tmp_path / "f50"
        subset_train.make_subset(t2, half, 0.5, seed=2024)
        quarter = tmp_path / "f25"
        manifest = subset_train.make_subset(t2, quarter, 0.25, seed=2024, nest_in=half)
        assert _sources(quarter) <= _sources(half)
        # the quota is computed on the FULL type count: round(12 * 0.25) = 3
        assert manifest["kept_by_type"] == {f"type{t}": 3 for t in range(N_TYPES)}

    def test_rejects_a_foreign_parent(self, t2: Path, tmp_path: Path) -> None:
        other = _build_t2(tmp_path / "other")
        (other / constants.LABELS_TRAIN_FILENAME).write_text(json.dumps({"zz__w000": 1}))
        with pytest.raises(ValueError, match="not in this dataset"):
            subset_train.make_subset(t2, tmp_path / "x", 0.25, seed=1, nest_in=other)


class TestGuards:
    def test_raises_when_a_class_vanishes(self, t2: Path, tmp_path: Path) -> None:
        labels = _labels(t2)
        (t2 / constants.LABELS_TRAIN_FILENAME).write_text(
            json.dumps(dict.fromkeys(labels, 1))
        )
        with pytest.raises(ValueError, match="both classes"):
            subset_train.make_subset(t2, tmp_path / "x", 0.5, seed=1)

    def test_raises_on_a_leaked_source(self, t2: Path, tmp_path: Path) -> None:
        test_ids = json.loads((t2 / constants.FRAME_LABELS_TEST_FILENAME).read_text())
        train_wid = next(iter(_labels(t2)))
        test_ids[train_wid] = [0] * WINDOW_LEN
        (t2 / constants.FRAME_LABELS_TEST_FILENAME).write_text(json.dumps(test_ids))
        with pytest.raises(ValueError, match="test items"):
            subset_train.make_subset(t2, tmp_path / "x", 1.0, seed=1)

    def test_refuses_to_overwrite_the_parent(self, t2: Path) -> None:
        with pytest.raises(ValueError, match="must differ"):
            subset_train.make_subset(t2, t2, 0.5, seed=1)


class TestFiles:
    def test_other_files_are_byte_identical(self, t2: Path, tmp_path: Path) -> None:
        out = tmp_path / "f50"
        subset_train.make_subset(t2, out, 0.5, seed=3)
        for name in (constants.WINDOWS_FILENAME, constants.META_FILENAME,
                     constants.FRAME_LABELS_TEST_FILENAME, constants.DEFS_FILENAME,
                     "extra.txt"):
            assert (out / name).read_bytes() == (t2 / name).read_bytes(), name
        assert not list(out.glob(f"*{constants.CACHE_PART_SUFFIX}"))

    def test_train_ids_file_is_filtered(self, t2: Path, tmp_path: Path) -> None:
        out = tmp_path / "f50"
        subset_train.make_subset(t2, out, 0.5, seed=3)
        kept = set((out / TRAIN_IDS_FILENAME).read_text().split())
        assert kept == _sources(out)

    def test_manifest_counts(self, t2: Path, tmp_path: Path) -> None:
        out = tmp_path / "f50"
        subset_train.make_subset(t2, out, 0.5, seed=3)
        manifest = json.loads((out / constants.SUBSET_MANIFEST_FILENAME).read_text())
        labels = _labels(out)
        assert manifest["items_kept"] == len(labels)
        assert manifest["abnormal_kept"] == sum(labels.values())
        assert manifest["normal_kept"] == len(labels) - sum(labels.values())
        assert manifest["sources_total"] == N_TYPES * SOURCES_PER_TYPE
        assert manifest["item_fraction"] == pytest.approx(0.5)

    def test_cli(self, t2: Path, tmp_path: Path) -> None:
        out = tmp_path / "cli"
        subset_train.main(["--data-dir", str(t2), "--out-dir", str(out),
                           "--fraction", "0.5", "--seed", "11"])
        assert (out / constants.SUBSET_MANIFEST_FILENAME).exists()
