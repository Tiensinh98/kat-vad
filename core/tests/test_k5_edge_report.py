"""``core.tools.k5_edge_report`` (``KIM_RUN.md`` §49): targets, labels and verdicts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.metrics import frame_auc
from core.tools import k5_edge_report as rep
from core.tools import pre_onset_read as po

FPS = 10.0
LENGTH, ONSET, END = 200, 100, 130  # 10 s pre-roll, a 3 s span
SETS = constants.KIM_LNEG_CAP_SETS
GAIN, COST, NONE = (constants.KIM_DVS_IGNORE_GAIN, constants.KIM_DVS_IGNORE_COST,
                    constants.KIM_DVS_IGNORE_NO_DIFFERENCE)


def _gt() -> np.ndarray:
    gt = np.zeros(LENGTH, dtype=np.int64)
    gt[ONSET:END] = 1
    return gt


def _curve(lead: int = 0, tail: int = 0) -> np.ndarray:
    """0.1 outside, 0.9 in the span, above the span's level ``lead`` / ``tail`` frames outside."""
    score: np.ndarray = np.full(LENGTH, 0.1) + np.linspace(0, 1e-3, LENGTH)
    score[ONSET:END] = 0.9
    score[ONSET - lead:ONSET] = 0.95
    score[END:END + tail] = 0.95
    return score


def _labels(**given: str) -> dict[str, str]:
    return {**dict.fromkeys((*SETS, "tad"), NONE), **given}


def test_onset_auc_reads_only_its_side() -> None:
    labels = {"v": _gt()}
    early, late = {"v": _curve(lead=20)}, {"v": _curve(tail=20)}
    assert rep.onset_aucs(early, labels, FPS, rep.PRE) == {"v": 0.0}
    assert rep.onset_aucs(early, labels, FPS, rep.POST) == {"v": 1.0}
    assert rep.onset_aucs(late, labels, FPS, rep.PRE) == {"v": 1.0}
    assert rep.onset_aucs(late, labels, FPS, rep.POST) == {"v": 0.0}


def test_onset_auc_zone_is_edge_seconds_wide_and_skips_videos_without_it() -> None:
    far = {"v": _curve()}
    far["v"][: ONSET - round(constants.EDGE_ONSET_S * FPS)] = 0.95  # beyond the zone: not read
    assert rep.onset_aucs(far, {"v": _gt()}, FPS, rep.PRE) == {"v": 1.0}
    no_pre = np.zeros(LENGTH, dtype=np.int64)
    no_pre[:END] = 1
    assert rep.onset_aucs({"v": _curve()}, {"v": no_pre}, FPS, rep.PRE) == {}
    assert rep.onset_aucs({"v": _curve()}, {"v": np.zeros(LENGTH)}, FPS, rep.PRE) == {}


def _nexar(lead: int, seed: int) -> tuple[po.Curves, po.Curves]:
    rng = np.random.default_rng(seed)
    scores, labels = {}, {}
    for i in range(8):
        scores[f"p{i}"] = _curve(lead) + rng.normal(0, 0.3, LENGTH)
        scores[f"n{i}"] = np.full(LENGTH, 0.1) + rng.normal(0, 0.3, LENGTH)
        labels[f"p{i}"], labels[f"n{i}"] = _gt(), np.zeros(LENGTH, dtype=np.int64)
    return scores, labels


def test_zone_auc_equals_the_pooled_auc_of_pre_onset_read() -> None:
    scores, labels = _nexar(lead=5, seed=0)
    got = rep.zone_delta({rep.K5F: rep.zone_wins(scores, labels, FPS)}, rep.K5F)
    key = po._zone_key(*constants.EDGE_TARGET_ZONE_S)
    assert got[rep.K5F]["auc"] == pytest.approx(po.zone_aucs(scores, labels, FPS)[key], abs=1e-12)
    zone = np.concatenate([scores[f"p{i}"][ONSET - 10:ONSET] for i in range(8)])
    neg = np.concatenate([scores[f"n{i}"] for i in range(8)])
    direct = frame_auc(np.concatenate([zone, neg]), np.r_[np.ones(zone.size), np.zeros(neg.size)])
    assert got[rep.K5F]["auc"] == pytest.approx(direct, abs=1e-12)


def test_zone_delta_sign_and_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(constants, "PRE_ONSET_BOOTSTRAP", 300)
    late, labels = _nexar(lead=0, seed=1)
    early, _ = _nexar(lead=10, seed=1)
    wins = {rep.K5F: rep.zone_wins(early, labels, FPS), "x": rep.zone_wins(late, labels, FPS)}
    delta = rep.zone_delta(wins, rep.K5F)["x"]["delta"]
    assert delta["high"] < 0 and delta["low"] <= delta["mean"] <= delta["high"]
    same = rep.zone_delta({rep.K5F: wins["x"], "x": wins["x"]}, rep.K5F)["x"]["delta"]
    assert same == {"mean": 0.0, "low": 0.0, "high": 0.0}


def test_zone_wins_needs_negative_videos() -> None:
    with pytest.raises(ValueError, match="positive and negative"):
        rep.zone_wins({"p": _curve()}, {"p": _gt()}, FPS)


def test_paired_ci_refuses_different_videos() -> None:
    with pytest.raises(ValueError, match="different videos"):
        rep.paired_ci({"a": 1.0}, {"b": 1.0}, None)


@pytest.mark.parametrize(("t1", "t2", "labels", "want"), [
    (True, True, _labels(), rep.REDUCED),
    (True, True, _labels(tad=COST), rep.REDUCED),  # TAD never decides
    (True, True, _labels(nexar=COST), rep.TRADE_OFF),
    (True, False, _labels(dad=COST), rep.TRADE_OFF),
    (True, False, _labels(), rep.PARTIAL),
    (False, True, _labels(dota=GAIN), rep.PARTIAL),
    (False, False, _labels(dota=GAIN), rep.NOT_REDUCED),
    (False, False, _labels(nexar=COST), rep.NOT_REDUCED),
])
def test_verdict_pre(t1: bool, t2: bool, labels: dict[str, str], want: str) -> None:
    assert rep.verdict_pre(t1, t2, labels) == want


@pytest.mark.parametrize(("labels", "want"), [
    (_labels(dota=GAIN), rep.GOES_ON), (_labels(dada=GAIN), rep.GOES_ON),
    (_labels(nexar=GAIN), rep.CLOSED), (_labels(dota=GAIN, dad=COST), rep.CLOSED),
    (_labels(), rep.CLOSED),
])
def test_verdict_macro(labels: dict[str, str], want: str) -> None:
    assert rep.verdict_macro(labels) == want


def _run(folder: Path, mil: float, step: int = 1) -> Path:
    folder.mkdir()
    (folder / "metrics.jsonl").write_text(json.dumps({"global_step": step, "mil": mil}) + "\n")
    return folder


def test_start_check(tmp_path: Path) -> None:
    base = _run(tmp_path / "a", 0.6878)
    assert rep.start_check(base, _run(tmp_path / "b", 0.68781))["same_draws"] is True
    assert rep.start_check(base, _run(tmp_path / "c", 0.70))["same_draws"] is False
    with pytest.raises(ValueError, match="global_step 1"):
        rep.start_check(base, _run(tmp_path / "d", 0.7, step=2))


def test_the_rows_flag_takes_a_subset() -> None:
    args = rep.build_arg_parser().parse_args(
        ["--kim-root", "k", "--k5f-nexar-dir", "n", "--edge-root", "e", "--out-dir", "o",
         "--rows", constants.KIM_EDGE_PRE_ROW])
    assert args.rows == [constants.KIM_EDGE_PRE_ROW]


def test_render_prints_every_row_verdict_and_flag() -> None:
    rows = list(constants.KIM_EDGE_ROWS)
    delta = {"mean": -0.02, "low": -0.03, "high": -0.01}
    macro = {"mean": 0.8}
    cell = {"delta": delta, "sentence": "below",
            "rows": {key: {"macro": macro, "macro_crops": macro} for key in (rep.kw.K5, *rows)}}
    read = {"auc": 0.7, "delta": delta, "videos": 5}
    every = {rep.K5F: {**read, "delta": None}, **dict.fromkeys(rows, read)}
    report = {
        "rule": rep.RULE, "seed": 2024, "rows": rows, "fixed_mul_mil": {rep.K5F: 0.1},
        "start_check": {"control": 0.68, "new": 0.70, "same_draws": False},
        "sets": {row: dict.fromkeys(rep.ALL_SETS, cell) for row in rows},
        "labels": {row: dict.fromkeys(rep.ALL_SETS, COST) for row in rows},
        "decision_sets": list(SETS), "t1": every,
        "onset": {s: {rep.PRE: every, rep.POST: every} for s in rep.ONSET_SETS},
        "hits": {row: {"T1": True, "T2": False} for row in rows},
        "verdicts": dict.fromkeys(rows, rep.TRADE_OFF),
        "false_alarms": dict.fromkeys((rep.K5F, *rows), 0.5),
    }
    text = rep.render(report)
    assert all(f"**{row}: {rep.TRADE_OFF}.**" in text for row in rows)
    assert "FLAG: not the same draws" in text and "| tad (printed) |" in text
