"""``core.tools.k5_bottomk_report`` (``KIM_RUN.md`` §50): targets, verdict, start gate, render."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.tools import k5_bottomk_report as rep
from core.tools import k5_edge_report as edge

GAIN, COST, NONE = (constants.KIM_DVS_IGNORE_GAIN, constants.KIM_DVS_IGNORE_COST,
                    constants.KIM_DVS_IGNORE_NO_DIFFERENCE)


def _ci(low: float, high: float) -> dict[str, float]:
    return {"mean": (low + high) / 2, "low": low, "high": high}


def _labels(**given: str) -> dict[str, str]:
    return {**dict.fromkeys(rep.ALL_SETS, NONE), **given}


def test_first_third_aucs_reads_only_the_first_part() -> None:
    gt = np.zeros(30, dtype=np.int64)
    gt[9:18] = 1  # thirds 9-11, 12-14, 15-17
    score = np.full(30, 0.1)
    score[12:18] = 0.9
    score[9:12] = 0.0  # below every normal row
    assert rep.first_third_aucs({"v": score}, {"v": gt}) == {"v": 0.0}
    score[9:12], score[12:18] = 0.9, 0.0
    assert rep.first_third_aucs({"v": score}, {"v": gt}) == {"v": 1.0}


def test_first_third_skips_short_spans_and_one_class_videos() -> None:
    two = np.zeros(10, dtype=np.int64)
    two[3:5] = 1  # two abnormal rows
    curve = np.arange(10, dtype=float)
    labels = {"two": two, "none": np.zeros(10, dtype=np.int64), "all": np.ones(10, dtype=np.int64)}
    assert rep.first_third_aucs(dict.fromkeys(labels, curve), labels) == {}


@pytest.mark.parametrize(("nexar", "dada", "ok"), [
    (_ci(-0.03, 0.01), _ci(-0.02, 0.0), True),
    (_ci(-0.03, -0.001), _ci(-0.02, 0.01), False),
    (_ci(-0.03, 0.01), _ci(-0.02, -0.001), False),
    (_ci(-0.03, 0.01), None, False),
])
def test_target_a_is_not_wholly_below_zero_on_both_sets(
    nexar: dict[str, float] | None, dada: dict[str, float] | None, ok: bool
) -> None:
    assert rep.target_a_ok({rep.NEXAR: nexar, rep.DADA: dada, rep.DOTA: _ci(-1, -1)}) is ok


@pytest.mark.parametrize(("a", "b", "labels", "want"), [
    (True, True, _labels(), rep.HELD),
    (True, True, _labels(tad=COST), rep.HELD),  # TAD never decides
    (True, True, _labels(dad=COST), rep.TRADE_OFF),
    (False, True, _labels(), rep.TRADE_OFF),
    (True, False, _labels(dota=GAIN), rep.CLOSED),
    (False, False, _labels(), rep.CLOSED),
])
def test_verdict(a: bool, b: bool, labels: dict[str, str], want: str) -> None:
    assert rep.verdict(a, b, labels) == want


def test_read_env(tmp_path: Path) -> None:
    assert rep.read_env(tmp_path) is None
    (tmp_path / constants.KIM_RUN_ENV_FILENAME).write_text(json.dumps({"gpu": "T4"}))
    assert rep.read_env(tmp_path) == {"gpu": "T4"}


def test_start_check_compares_the_two_rows(tmp_path: Path) -> None:
    for name, mil in (("a", 0.6925), ("b", 0.69251), ("c", 0.6878)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.jsonl").write_text(
            json.dumps({"global_step": 1, "mil": mil}) + "\n")
    assert edge.start_check(tmp_path / "a", tmp_path / "b")["same_draws"] is True
    assert edge.start_check(tmp_path / "a", tmp_path / "c")["same_draws"] is False


def test_onset_reads_accepts_another_control() -> None:
    assert inspect.signature(edge.onset_reads).parameters["control"].default == edge.K5F


def _report(**over: Any) -> dict[str, Any]:
    cell = {"delta": _ci(-0.03, -0.01), "sentence": "below",
            "rows": {k: {"macro": {"mean": 0.8}, "macro_crops": {"mean": 0.8}}
                     for k in (rep.kw.K5, rep.NEW, rep.CONTROL)}}
    read = {"auc": 0.7, "delta": _ci(-0.02, 0.02), "videos": 9}
    base = {"auc": 0.7, "delta": None, "videos": 9}
    two = {rep.CONTROL: base, rep.NEW: read}
    return {
        "rule": rep.RULE, "seed": 2024, "fixed_mul_mil": {rep.CONTROL: 0.1},
        "start_check": {"control": 0.69, "new": 0.69, "same_draws": True},
        "start_vs_k5f": {"control": 0.6878, "new": 0.69, "same_draws": False},
        "env": {rep.CONTROL: {"gpu": "T4"}, rep.NEW: {"gpu": "T4"}},
        "sets": dict.fromkeys(rep.ALL_SETS, cell), "labels": _labels(),
        "rerun": dict.fromkeys(rep.ALL_SETS, cell),
        "thirds": dict.fromkeys(edge.ONSET_SETS, two),
        "onset": {s: {edge.PRE: two, edge.POST: two} for s in edge.ONSET_SETS},
        "targets": {"A": True, "B": True}, "a_strict": {rep.NEXAR: True, rep.DADA: False},
        "verdict": rep.HELD, "decision_sets": list(rep.dvs.DECISION_SETS),
        "false_alarms": {rep.CONTROL: 0.5}, **over,
    }


def test_render_prints_verdict_targets_and_rerun_column() -> None:
    text = rep.render(_report())
    assert f"**Verdict: {rep.HELD}.**" in text and "FLAG" not in text
    assert f"{rep.CONTROL} - {rep.K5F}" in text and "tad (printed)" in text


def test_render_flags_a_different_environment() -> None:
    env = {rep.CONTROL: {"gpu": "T4"}, rep.NEW: {"gpu": "A100"}}
    assert "FLAG: the two rows were trained in different environments" in rep.render(
        _report(env=env))
