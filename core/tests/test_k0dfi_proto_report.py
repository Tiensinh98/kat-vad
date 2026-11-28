"""``core.tools.k0dfi_proto_report`` (``KIM_RUN.md`` §61): rows, verdict, gates, render."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import constants
from core.tools import k0dfi_proto_report as rep

GAIN, COST, NONE = (constants.KIM_DVS_IGNORE_GAIN, constants.KIM_DVS_IGNORE_COST,
                    constants.KIM_DVS_IGNORE_NO_DIFFERENCE)


def _labels(**given: str) -> dict[str, str]:
    return {**dict.fromkeys(rep.SETS, NONE), **given}


def _metrics(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_rows_contrasts_and_sets_are_the_rule_s() -> None:
    assert rep.ROWS == ("K0dfiP", "K0dfiL", "K0dfiPL", "K0dfiPLb")
    assert rep.CONTRASTS["D"] == (rep.CONTROL, "K0dfiPL")
    assert rep.CONTRASTS["b"] == ("K0dfiPL", "K0dfiPLb")
    assert rep.CONTRASTS["R"] == ("K5f", "K0dfiPL")
    assert rep.SETS == ("nexar", "dota", "dada", "dad")  # no TAD
    assert "dad" not in rep.DECISION_SETS


@pytest.mark.parametrize(("labels", "same", "want"), [
    (_labels(), True, rep.NO_DIFFERENCE),
    (_labels(dada=GAIN), True, rep.HELD),
    (_labels(dada=GAIN, nexar=COST), True, rep.D_COST),
    (_labels(dad=COST), True, rep.NO_DIFFERENCE),  # DAD never decides
    (_labels(dad=GAIN), True, rep.NO_DIFFERENCE),
    (_labels(dada=GAIN), False, rep.UNMATCHED),
])
def test_verdict(labels: dict[str, str], same: bool, want: str) -> None:
    assert rep.verdict(labels, same) == want


def test_candidate_needs_held_and_no_cost_against_k5f_on_a_deciding_set() -> None:
    assert rep.candidate(rep.HELD, _labels()) is True
    assert rep.candidate(rep.HELD, _labels(dad=COST)) is True
    assert rep.candidate(rep.HELD, _labels(dada=COST)) is False
    assert rep.candidate(rep.NO_DIFFERENCE, _labels()) is False


def test_proto_trace_reads_the_last_step(tmp_path: Path) -> None:
    rows = [{"epoch": 0, "proto_ready": 0.0},
            {"epoch": 1, "proto_ready": 1.0, "proto_lambda": 0.3, "proto_cos": -0.2,
             "proto_anchor_share": 0.5}]
    got = rep.proto_trace(_metrics(tmp_path / "a", rows))
    assert got == {"proto_lambda": 0.3, "proto_cos": -0.2, "proto_ready": 1.0,
                   "proto_anchor_share": 0.5}


def test_proto_trace_raises_when_the_prototypes_never_existed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="never both existed"):
        rep.proto_trace(_metrics(tmp_path / "a", [{"epoch": 0, "proto_ready": 0.0}]))


def _cell(mean: float) -> dict[str, Any]:
    delta = {"mean": mean, "low": mean - 0.01, "high": mean + 0.01}
    return {"rows": {"K5": {"macro": {"mean": 0.7}, "macro_crops": {"mean": 0.7}},
                     **{row: {"macro": {"mean": 0.7 + mean}, "macro_crops": {"mean": 0.7 + mean}}
                        for row in (*rep.ROWS,)}},
            "delta": delta, "sentence": "above" if mean > 0 else "not separated"}


def _report(env: dict[str, Any] | None = None) -> dict[str, Any]:
    start = {"control": 0.689790, "new": 0.689790, "same_draws": True}
    trace = {"proto_lambda": 0.2, "proto_cos": -0.3, "proto_ready": 1.0, "proto_anchor_share": 0.4}
    one = {"gpu": "NVIDIA L4"}
    return {
        "rule": rep.RULE, "seed": 2024, "control": "K0dfi",
        "fixed_mul_mil": {rep.CONTROL: 0.1, **dict.fromkeys(rep.ROWS, 0.1)},
        "starts": dict.fromkeys(rep.ROWS, start), "proto": dict.fromkeys(rep.ROWS, trace),
        "env": env or {row: dict(one) for row in (rep.CONTROL, *rep.ROWS)},
        "sets": {name: dict.fromkeys(rep.SETS, _cell(0.02)) for name in rep.CONTRASTS},
        "labels": {name: _labels(dada=GAIN) for name in rep.CONTRASTS},
        "decision_sets": list(rep.DECISION_SETS), "verdict": rep.HELD, "candidate": True,
        "false_alarms": {rep.CONTROL: 0.6, **dict.fromkeys(rep.ROWS, 0.5)},
    }


def test_render_prints_every_contrast_the_gates_and_names_the_control() -> None:
    text = rep.render(_report())
    for name in rep.CONTRASTS:
        assert rep.TITLES[name].split(" ")[0] in text
    assert "D (decides): K0dfiPL - K0dfi" in text and "control" not in text.split("\n")[0]
    assert "λ +0.2000" in text and "dad (never decides)" in text
    assert "FLAG" not in text


def test_render_flags_a_different_environment() -> None:
    env = {rep.CONTROL: {"gpu": "NVIDIA L4"}, **{row: {"gpu": "Tesla T4"} for row in rep.ROWS}}
    assert "different environments" in rep.render(_report(env))
