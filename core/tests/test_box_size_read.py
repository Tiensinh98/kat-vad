"""Tests for ``core.tools.box_size_read`` (KIM_RUN §47)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from core.tools import box_size_read as bs


def _doc(frames: dict[int, list[tuple[int, list[float]]]]) -> dict[str, Any]:
    return {"anomaly_start": 2, "anomaly_end": 6, "labels": [
        {"frame_id": k, "objects": [{"obj_track_id": t, "bbox": b} for t, b in objs]}
        for k, objs in frames.items()]}


def test_clip_boxes_takes_the_largest_box_inside_the_window() -> None:
    big, small = [0.0, 0.0, 128.0, 72.0], [0.0, 0.0, 12.8, 7.2]
    out = bs.clip_boxes(_doc({0: [(1, [0.0, 0.0, 1280.0, 720.0])], 2: [(1, big), (2, small)],
                              3: [(1, [10.0, 0.0, 138.0, 72.0])]}))
    assert out is not None
    assert out[bs.SIZE] == pytest.approx(0.01)
    assert out[bs.SPEED] == pytest.approx(10.0)
    assert out[bs.SIZE_SPEED] == pytest.approx(0.1)


def test_clip_boxes_without_a_box_in_the_window_is_none() -> None:
    assert bs.clip_boxes(_doc({0: [(1, [0.0, 0.0, 9.0, 9.0])], 3: []})) is None


def test_spearman_is_rank_based_and_skips_nan() -> None:
    x = np.array([1.0, 2.0, 3.0, np.nan])
    assert bs.spearman(x, np.array([1.0, 10.0, 100.0, 0.0])) == pytest.approx(1.0)


def test_partial_spearman_removes_a_shared_cause() -> None:
    rng = np.random.default_rng(0)
    c = rng.normal(size=800)
    x, y = c + 0.3 * rng.normal(size=800), c + 0.3 * rng.normal(size=800)
    assert bs.spearman(x, y) > 0.8
    assert abs(bs.partial_spearman(x, y, [c])) < 0.2


def _size_driven() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low only depends on the bin; the groups differ in bin mix alone."""
    bins = np.concatenate([np.repeat(np.arange(6), [50, 10, 10, 10, 10, 10]),
                           np.repeat(np.arange(6), [10, 10, 10, 10, 10, 50])])
    involved = np.array([False] * 100 + [True] * 100)
    return (bins == 0).astype(np.float64), involved, bins


def test_gap_parts_is_all_mix_when_low_depends_on_size_only() -> None:
    parts = bs.gap_parts(*_size_driven())
    assert parts is not None
    assert parts["G_raw"] == pytest.approx(0.4)
    assert parts["G_adj"] == pytest.approx(0.0)
    assert parts["R"] == pytest.approx(0.4)


def test_gap_parts_is_none_on_an_empty_bin() -> None:
    low, involved, bins = _size_driven()
    assert bs.gap_parts(low, involved, np.where(bins == 5, 4, bins)) is None


def _iv(mean: float, low: float, high: float) -> dict[str, float]:
    return {"mean": mean, "low": low, "high": high}


@pytest.mark.parametrize(("rho", "r", "want"), [
    (_iv(0.3, 0.2, 0.4), _iv(0.15, 0.05, 0.25), bs.CONSISTENT),
    (_iv(0.3, 0.2, 0.4), _iv(0.03, -0.02, 0.08), bs.RELATED),
    (_iv(0.05, -0.05, 0.15), _iv(0.03, -0.02, 0.08), bs.NO_RELATION),
    (_iv(0.3, 0.2, 0.4), _iv(0.08, 0.01, 0.15), bs.NOT_SETTLED),
    (_iv(0.15, -0.02, 0.3), _iv(0.03, -0.02, 0.08), bs.NOT_SETTLED),
    (_iv(0.05, -0.05, 0.15), _iv(0.15, 0.05, 0.25), bs.NOT_SETTLED),
])
def test_verdicts(rho: dict[str, float], r: dict[str, float], want: str) -> None:
    assert bs.verdict(rho, {"G_raw": _iv(0.3, 0.2, 0.4), "R": r}) == want
