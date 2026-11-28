"""core.tools.window_geometry_read (KIM_RUN §67): windowing, window labels, verdict. CPU."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from core import constants
from core.tools import window_geometry_read as wg

GAIN, COST, NONE = wg.GAIN, wg.COST, constants.KIM_DVS_IGNORE_NO_DIFFERENCE


class _StartModel(nn.Module):
    """Scores 1.0 logit at the first row of whatever sequence it is given, else -1."""

    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, class_feats: Any) -> dict[str, Any]:
        logits = torch.full(x.shape[:2], -1.0)
        logits[:, 0] = 1.0
        return {"cls_bin_logits": logits, "cls_sim_mat": torch.zeros(*x.shape[:2], 2)}


def test_windowed_steps_cover_every_row_and_follow_the_geometry() -> None:
    feats = torch.zeros(30, 4)
    steps, parts = wg.windowed_steps(_StartModel(), feats, lambda: None)
    assert len(steps) == 30 and [s for s, _ in parts] == [0, 6, 12, 18]
    assert all(len(p) == wg.WINDOW for _, p in parts)
    high = torch.sigmoid(torch.tensor(1.0)).item()
    # row 6 is the start of window 2 and row 6 of window 1: mean of high and low
    low = torch.sigmoid(torch.tensor(-1.0)).item()
    assert steps[0] == pytest.approx(high) and steps[6] == pytest.approx((high + low) / 2)
    assert wg.position_profile(parts, wg.WINDOW)[0] == pytest.approx(high)


def test_short_clip_is_one_window() -> None:
    steps, parts = wg.windowed_steps(_StartModel(), torch.zeros(7, 4), lambda: None)
    assert len(parts) == 1 and len(steps) == 7


def test_topk_mean_reads_like_l_mil() -> None:
    scores = np.arange(12, dtype=float)
    assert wg.topk_mean(scores, 5) == pytest.approx(10.5)  # k = 12 // 5 = 2
    assert wg.topk_mean(np.array([0.3, 0.9]), 5) == pytest.approx(0.9)  # k >= 1


def test_window_records_label_any_abnormal_row() -> None:
    labels = np.zeros(18, dtype=np.int64)
    labels[15] = 1
    parts = [(0, np.zeros(12)), (6, np.ones(12))]
    assert wg.window_records(parts, labels, 5) == [(0.0, 0), (1.0, 1)]


def test_spike_ratio() -> None:
    steps = np.array([0.1, 0.8, 0.8, 0.1, 0.1, 0.1])
    assert wg.spike_ratio(steps) == pytest.approx(8.0)


@pytest.mark.parametrize(("g", "r", "want"), [
    (GAIN, NONE, constants.KIM_WG_GEOMETRY),
    (GAIN, GAIN, constants.KIM_WG_GEOMETRY),
    (GAIN, COST, constants.KIM_WG_PART),
    (NONE, GAIN, constants.KIM_WG_NOT),
    (COST, NONE, constants.KIM_WG_NOT),
])
def test_verdict(g: str, r: str, want: str) -> None:
    assert wg.verdict({"G": g, "R": r}) == want


def test_paired_uses_common_clips() -> None:
    aucs = {(wg.L5, wg.WINDOWED): {"a": 0.7, "b": 0.6}, (wg.L5, wg.WHOLE): {"a": 0.4}}
    assert wg.paired(aucs, wg.TERMS["G"]) == {"a": pytest.approx(0.3)}


def test_render_prints_the_verdict() -> None:
    ci = {"mean": 0.1, "low": 0.05, "high": 0.15}
    row = {"name": "x", "macro": {wg.WHOLE: ci, wg.WINDOWED: ci}, "window_auc": 0.6,
           "windows": 10, "positive_windows": 7, "start_spike_whole": 3.0,
           "position_profile": [0.1] * wg.WINDOW}
    report = {"label": wg.LABEL, "clips": 569, "two_class": 569, "stride": 3, "window": 12,
              "hop": 6, "verdict": constants.KIM_WG_PART,
              "rows": dict.fromkeys(constants.KIM_WG_ROWS, row),
              "contrasts": dict.fromkeys(wg.TERMS, ci), "labels": dict.fromkeys(wg.TERMS, GAIN)}
    text = wg.render(report)
    assert "PART GEOMETRY" in text and "G = L5 windowed - L5 whole" in text
