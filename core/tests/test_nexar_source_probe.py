"""Tests for :mod:`core.tools.nexar_source_probe` (``KIM_RUN.md`` §38 S0), CPU, synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.data.windows import Window, window_id, windows_payload
from core.tools import nexar_source_probe as probe

DIM = 8
ROWS = 40
WINDOW = 10
VIDEOS_PER_CLASS = 12
SCENES = ("Urban", "Highway")


def _write_corpus(root: Path, negative_shift: float, seed: int = 0) -> tuple[Path, Path]:
    """Positive videos: windows 0-1 normal, 2 positive; negative videos: 3 normal windows."""
    rng = np.random.default_rng(seed)
    data_dir, clip_dir = root / "corpus", root / "clip"
    data_dir.mkdir()
    clip_dir.mkdir()
    windows: dict[str, Window] = {}
    meta: dict[str, dict[str, object]] = {}
    for label in (1, 0):
        for k in range(VIDEOS_PER_CLASS):
            source = f"{'pos' if label else 'neg'}{k:02d}"
            rows = rng.normal(size=(ROWS, DIM))
            if not label:
                rows[:, 0] += negative_shift
            np.save(clip_dir / f"{source}.npy", rows)
            for index in range(3):
                wid = window_id(source, index)
                windows[wid] = Window(source, index * WINDOW, (index + 1) * WINDOW)
                positive = WINDOW if (label and index == 2) else 0
                meta[wid] = {"split": "train", "source": source, "positive_frames": positive,
                             "source_is_abnormal": bool(label),
                             "scene": SCENES[k % 2], "weather": "Clear"}
    meta[window_id("val00", 0)] = {"split": "test", "source": "val00", "positive_frames": 0,
                                   "source_is_abnormal": True}
    (data_dir / constants.META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")
    (data_dir / constants.WINDOWS_FILENAME).write_text(
        json.dumps(windows_payload(windows)), encoding="utf-8")
    return data_dir, clip_dir


def test_pools_split_by_source_class_and_positive_frames(tmp_path: Path) -> None:
    data_dir, _ = _write_corpus(tmp_path, negative_shift=0.0)
    meta = json.loads((data_dir / constants.META_FILENAME).read_text(encoding="utf-8"))
    pools = probe.split_pools(meta)
    assert len(pools[probe.POOL_A]) == 2 * VIDEOS_PER_CLASS
    assert len(pools[probe.POOL_B]) == 3 * VIDEOS_PER_CLASS
    assert len(pools[probe.POOL_P]) == VIDEOS_PER_CLASS
    assert not any(w.startswith("val") for pool in pools.values() for w in pool)


def test_empty_pool_raises() -> None:
    meta = {"a": {"split": "train", "source": "a", "positive_frames": 3,
                  "source_is_abnormal": True}}
    with pytest.raises(ValueError, match="Nexar_window_neg"):
        probe.split_pools(meta)


def test_shifted_negative_pool_is_flagged(tmp_path: Path) -> None:
    data_dir, clip_dir = _write_corpus(tmp_path, negative_shift=6.0)
    report = probe.run(data_dir, clip_dir, folds=3, seed=0)
    assert report["S1_A_vs_B"] >= constants.NEXAR_SOURCE_PROBE_SHORTCUT_AUC
    assert report["shortcut"] is True
    assert "SHORTCUT" in probe.render_markdown(report)


def test_same_distribution_is_not_flagged(tmp_path: Path) -> None:
    data_dir, clip_dir = _write_corpus(tmp_path, negative_shift=0.0)
    report = probe.run(data_dir, clip_dir, folds=3, seed=0)
    assert report["S1_A_vs_B"] < constants.NEXAR_SOURCE_PROBE_SHORTCUT_AUC
    assert report["shortcut"] is False
    assert report["videos"] == {probe.POOL_A: VIDEOS_PER_CLASS, probe.POOL_B: VIDEOS_PER_CLASS,
                                probe.POOL_P: VIDEOS_PER_CLASS}


def test_window_past_cache_raises(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()
    np.save(clip_dir / "v.npy", np.zeros((5, DIM)))
    slicer = probe.FeatureSlicer({"v__000": Window("v", 0, WINDOW)})
    with pytest.raises(ValueError, match="C2"):
        probe.window_means(clip_dir, ["v__000"], slicer)


def test_stratified_match_copies_a_shares() -> None:
    strata = {**{f"a{k}": "Urban" for k in range(6)},
              **{f"a{k}": "Highway" for k in range(6, 8)},
              **{f"b{k}": "Urban" for k in range(20)},
              **{f"b{k}": "Highway" for k in range(20, 24)},
              "a8": "Rain"}
    ids_a = [f"a{k}" for k in range(9)]
    ids_b = [f"b{k}" for k in range(24)]
    kept_a, kept_b = probe.stratified_match(ids_a, ids_b, strata, seed=0)
    assert "a8" not in kept_a  # B has no Rain window
    counts = {s: sum(strata[w] == s for w in kept_b) for s in ("Urban", "Highway")}
    assert counts["Urban"] == 3 * counts["Highway"]  # A's 6 : 2
    assert len(set(kept_b)) == len(kept_b)


def test_row_stride_reads_strided_rows(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()
    np.save(clip_dir / "v.npy", np.arange(4 * WINDOW, dtype=np.float64)[:, None])
    slicer = probe.FeatureSlicer({"v__000": Window("v", 0, 2)})
    means = probe.window_means(clip_dir, ["v__000"], slicer, row_stride=8)
    assert means[0, 0] == pytest.approx((0 + 8) / 2)  # rows 0 and 8 of the s1 file


def test_main_writes_named_reports(tmp_path: Path) -> None:
    data_dir, clip_dir = _write_corpus(tmp_path, negative_shift=0.0)
    out = tmp_path / "out"
    probe.main(["--data-dir", str(data_dir), "--clip-dir", str(clip_dir), "--out-dir", str(out),
                "--folds", "3", "--name", "clip"])
    stem = f"{constants.NEXAR_SOURCE_PROBE_STEM}_clip"
    report = json.loads((out / f"{stem}.json").read_text(encoding="utf-8"))
    assert report["row_stride"] == 1 and report["feature_dim"] == DIM
    assert (out / f"{stem}.md").read_text(encoding="utf-8").startswith("# Nexar source-shortcut")
