"""Tests for ``core.tools.micro_level_read`` (KIM_RUN §44.2)."""

from __future__ import annotations

import numpy as np
import pytest

from core.tools import micro_level_read as ml


def _curves() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    scores, labels = {}, {}
    for i in range(12):
        n = 30 + i
        y = np.zeros(n, dtype=np.int64)
        y[10:18] = 1
        offset = 0.0 if i < 4 else 0.5  # four quiet clips
        s = offset + 0.3 * y + 0.05 * rng.random(n)
        scores[f"c{i}"], labels[f"c{i}"] = s.astype(np.float32), y
    return scores, labels


def test_share_below_counts_ties_half() -> None:
    ref = np.array([0.0, 1.0, 1.0, 2.0])
    np.testing.assert_allclose(ml.share_below(ref, np.array([1.0, 3.0, -1.0])), [0.5, 1.0, 0.0])


def test_weighted_level_equals_pooled_micro() -> None:
    scores, labels = _curves()
    lv = ml.levels(scores, labels)  # raises if the L2 gate fails
    assert 0.0 < lv["micro_raw"] < 1.0


def test_quiet_clips_are_low_raw_but_not_after_minmax() -> None:
    scores, labels = _curves()
    lv = ml.levels(scores, labels)
    low = {v for v, a in lv["A"].items() if a < 0.5}
    assert low == {"c0", "c1", "c2", "c3"}
    assert all(a > 0.5 for a in lv["A_mm"].values())


def test_overlap_and_parse_rows() -> None:
    o = ml.overlap(["a", "b"], ["b", "c"])
    assert (o["both"], o["first_only"], o["second_only"]) == (1, 1, 1)
    assert o["jaccard"] == pytest.approx(1 / 3)
    assert ml.parse_rows(["K5f=/x"]) == {"K5f": ml.Path("/x")}
    with pytest.raises(ValueError):
        ml.parse_rows(["nopath"])
