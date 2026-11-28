"""``core.tools.k5_clip_encoder_report`` (``KIM_RUN.md`` §59): gates, verdict, render."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import constants
from core.tools import k5_clip_encoder_report as rep

GAIN, COST, NONE = (constants.KIM_DVS_IGNORE_GAIN, constants.KIM_DVS_IGNORE_COST,
                    constants.KIM_DVS_IGNORE_NO_DIFFERENCE)
OK = {"same_draws": True, "trained": True}


def _labels(**given: str) -> dict[str, str]:
    return {**dict.fromkeys(rep.ALL_SETS, NONE), **given}


def _metrics(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_rows_and_contrasts_are_the_rule_s() -> None:
    assert (rep.CONTROL, rep.NEW, rep.CAP, rep.K5F) == ("K5dfirr", "K5dfice", "K5dficx", "K5f")
    assert rep.CONTRASTS == {"P": ("K5dfirr", "K5dfice"), "C": ("K5dficx", "K5dfice"),
                             "X": ("K5dfirr", "K5dficx"), "R": ("K5f", "K5dfice")}


def test_trains_check_reads_the_last_epoch_and_final_drift(tmp_path: Path) -> None:
    rows = [{"epoch": 0, "global_step": 1, "mil": 0.7},  # a non-boundary batch has no key
            {"epoch": 0, rep.GRAD_KEY: 0.0, rep.DRIFT_KEY: 1e-4},
            {"epoch": 1, rep.GRAD_KEY: 0.2, rep.DRIFT_KEY: 0.01},
            {"epoch": 1, rep.GRAD_KEY: 0.4, rep.DRIFT_KEY: 0.02}]
    got = rep.trains_check(_metrics(tmp_path / "a", rows))
    assert got == {"grad_norm": pytest.approx(0.3), "drift": 0.02, "trained": True}


@pytest.mark.parametrize("last", [
    {"epoch": 1, rep.GRAD_KEY: 0.0, rep.DRIFT_KEY: 0.02},  # no gradient reached it
    {"epoch": 1, rep.GRAD_KEY: float("inf"), rep.DRIFT_KEY: 0.02},
    {"epoch": 1, rep.GRAD_KEY: 0.3, rep.DRIFT_KEY: 1e-4},  # it barely moved
])
def test_trains_check_flags_an_encoder_that_did_not_train(tmp_path: Path,
                                                          last: dict[str, Any]) -> None:
    assert rep.trains_check(_metrics(tmp_path / "a", [last]))["trained"] is False


def test_trains_check_without_the_key_is_untrained(tmp_path: Path) -> None:
    got = rep.trains_check(_metrics(tmp_path / "a", [{"epoch": 0, "mil": 0.7}]))
    assert got == {"grad_norm": None, "drift": None, "trained": False}


@pytest.mark.parametrize(("labels", "want"), [
    (_labels(), rep.NO_DIFFERENCE),
    (_labels(dada=GAIN), rep.HELD),
    (_labels(dada=GAIN, dad=COST), rep.MODULE_COST),
    (_labels(tad=COST), rep.NO_DIFFERENCE),  # TAD never decides
    (_labels(tad=GAIN), rep.NO_DIFFERENCE),
])
def test_module_verdict(labels: dict[str, str], want: str) -> None:
    assert rep.module_verdict(labels) == want


def test_attribution_needs_c_to_gain_where_p_gains() -> None:
    p = _labels(dada=GAIN, dota=GAIN)
    assert rep.attribution(_labels(), _labels(dada=GAIN)) is None  # P not HELD
    assert rep.attribution(p, _labels()) == rep.NOT_SEPARATED
    assert rep.attribution(p, _labels(nexar=GAIN)) == rep.NOT_SEPARATED  # not where P gains
    assert rep.attribution(p, _labels(dada=GAIN)) == rep.CLIP_CARRIES.format(sets="dada")


def test_candidate_needs_held_and_no_cost_against_k5f() -> None:
    assert rep.candidate(rep.HELD, _labels()) is True
    assert rep.candidate(rep.HELD, _labels(dad=COST)) is False
    assert rep.candidate(rep.HELD, _labels(tad=COST)) is True
    assert rep.candidate(rep.NO_DIFFERENCE, _labels()) is False


@pytest.mark.parametrize(("new", "cap", "want_verdict", "want_attr"), [
    (OK, OK, rep.HELD, rep.CLIP_CARRIES.format(sets="dada")),
    ({"same_draws": False, "trained": True}, OK, rep.UNMATCHED, None),
    ({"same_draws": True, "trained": False}, OK, rep.UNTRAINED, None),
    (OK, {"same_draws": True, "trained": False}, rep.HELD, f"not read: {rep.CAP} misses a gate"),
])
def test_decide_applies_the_gates_first(new: dict[str, bool], cap: dict[str, bool],
                                        want_verdict: str, want_attr: str | None) -> None:
    labels = {"P": _labels(dada=GAIN), "C": _labels(dada=GAIN), "X": _labels(),
              "R": _labels()}
    got = rep.decide({rep.NEW: new, rep.CAP: cap}, labels)
    assert (got["verdict"], got["attribution"]) == (want_verdict, want_attr)
    assert got["candidate"] is (want_verdict == rep.HELD)


def _report(**over: Any) -> dict[str, Any]:
    rows = {k: {"macro": {"mean": 0.8}, "macro_crops": {"mean": 0.8}}
            for k in (rep.kw.K5, rep.NEW, rep.CAP)}
    cell = {"delta": {"mean": -0.02, "low": -0.03, "high": -0.01}, "sentence": "below",
            "rows": rows}
    gate = {"same_draws": True, "trained": True,
            "start": {"control": 0.69, "new": 0.69, "same_draws": True},
            "trains": {"grad_norm": 0.01, "drift": 0.05, "trained": True}}
    return {
        "rule": rep.RULE, "seed": 2024, "fixed_mul_mil": {rep.NEW: 0.15},
        "gates": {rep.NEW: gate, rep.CAP: gate},
        "env": {row: {"gpu": "T4"} for row in (rep.CONTROL, rep.NEW, rep.CAP)},
        "sets": {n: dict.fromkeys(rep.ALL_SETS, cell) for n in rep.CONTRASTS},
        "labels": {n: _labels() for n in rep.CONTRASTS},
        "decision_sets": list(rep.dvs.DECISION_SETS),
        "verdict": rep.NO_DIFFERENCE, "attribution": None, "candidate": False,
        "false_alarms": {rep.CONTROL: 0.5, rep.NEW: 0.5}, **over,
    }


def test_render_prints_every_contrast_and_the_gates() -> None:
    text = rep.render(_report())
    assert f"**Verdict: {rep.NO_DIFFERENCE}.**" in text and "FLAG" not in text
    for name in rep.CONTRASTS:
        assert f"## {rep.TITLES[name]}" in text
    assert "tad (printed)" in text and "trained = True" in text and "drift 0.0500" in text


def test_render_flags_a_different_environment() -> None:
    env = {rep.CONTROL: {"gpu": "T4"}, rep.NEW: {"gpu": "L4"}, rep.CAP: {"gpu": "L4"}}
    assert "FLAG: the rows were trained in different environments" in rep.render(
        _report(env=env))
