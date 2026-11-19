"""``core.tools.k5_dvs_ignore_report`` (``KIM_RUN.md`` §34): R4 labels, verdict, start check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import constants
from core.tools import k5_dvs_ignore_report as rep
from core.tools.zs_lagovad_readout import ABOVE, BELOW, NOT_SEPARATED

SETS = constants.KIM_LNEG_CAP_SETS
GAIN, COST, NONE = (constants.KIM_DVS_IGNORE_GAIN, constants.KIM_DVS_IGNORE_COST,
                    constants.KIM_DVS_IGNORE_NO_DIFFERENCE)


def _cell(mean: float, low: float, high: float) -> dict[str, dict[str, float]]:
    return {"delta": {"mean": mean, "low": low, "high": high}}


def _labels(**given: str) -> dict[str, str]:
    out = dict.fromkeys((*SETS, "tad"), NONE)
    out.update(given)
    return out


@pytest.mark.parametrize(("cell", "want"), [
    (_cell(0.012, 0.002, 0.022), GAIN),
    (_cell(0.0098, 0.001, 0.018), NONE),  # K5d - W0's miss (§12.1): below the margin is not a pass
    (_cell(0.020, -0.001, 0.041), NONE),  # margin met, interval spans 0
    (_cell(-0.015, -0.025, -0.005), COST),
    (_cell(-0.004, -0.010, 0.002), NONE),
    ({"delta": None}, NONE),
])
def test_label_needs_the_margin_and_the_interval(cell: dict, want: str) -> None:
    assert rep.label(cell) == want


@pytest.mark.parametrize(("labels", "want"), [
    (_labels(dota=GAIN), GAIN),
    (_labels(dada=GAIN), GAIN),
    (_labels(nexar=GAIN), NONE),  # a gain off the DVS-sensitive sets does not count
    (_labels(dota=GAIN, dad=COST), COST),  # a cost anywhere decides
    (_labels(nexar=COST), COST),
    (_labels(tad=COST), NONE),  # TAD never decides
    (_labels(), NONE),
])
def test_verdict(labels: dict[str, str], want: str) -> None:
    assert rep.verdict(labels) == want


def _map(**sentences: str) -> dict[str, str]:
    out = dict.fromkeys((*SETS, "tad"), NOT_SEPARATED)
    out.update(sentences)
    return out


@pytest.mark.parametrize(("vs_ref", "want"), [
    (_map(dada=ABOVE), True),
    (_map(dada=ABOVE, dad=BELOW), False),
    (_map(), False),
    (_map(tad=ABOVE), False),
])
def test_candidate_needs_one_above_and_none_below(vs_ref: dict[str, str], want: bool) -> None:
    assert rep.candidate(vs_ref)["candidate"] is want


def _run(tmp_path: Path, name: str, **step1: float) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    row = {"global_step": 1, "mil": 0.69, "dvs_sup": 0.70, "dvs_sup_mil": 0.78, **step1}
    (run_dir / "metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return run_dir


def test_start_check_passes_with_a_live_mask_and_same_draws(tmp_path: Path) -> None:
    got = rep.start_check(_run(tmp_path, "df"), _run(tmp_path, "i", dvs_sup=0.55))
    assert got["same_draws"] is True and got["step1"]["dvs_sup"]["K5dfi"] == 0.55


def test_start_check_flags_different_draws_without_raising(tmp_path: Path) -> None:
    got = rep.start_check(_run(tmp_path, "df"), _run(tmp_path, "i", dvs_sup=0.55, mil=0.71))
    assert got["same_draws"] is False and got["equal"]["mil"] is False


def test_start_check_raises_when_the_mask_removed_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ignore mask removed nothing"):
        rep.start_check(_run(tmp_path, "df"), _run(tmp_path, "i"))


def _synthetic_report() -> dict:
    """A hand-built report with the shape ``run`` writes (per-clip scores are not local)."""
    macro = {"mean": 0.8}
    delta = {"mean": 0.012, "low": 0.002, "high": 0.022}
    cell = {"delta": delta, "sentence": ABOVE,
            "rows": {key: {"macro": macro, "macro_crops": macro}
                     for key in (rep.kw.K5, *rep.ROWS)}}
    sets = {name: dict.fromkeys(rep.ALL_SETS, cell) for name, _, _ in rep.CONTRASTS}
    labels = {n: {s: rep.label(sets[n][s]) for s in rep.ALL_SETS} for n in rep.DECIDING}
    step1 = {k: {"K5df": 0.7, "K5dfi": 0.6} for k in (*rep.START_KEYS, rep.MASK_KEY)}
    sentences = {n: dict.fromkeys(rep.ALL_SETS, ABOVE) for n in sets}
    return {"rule": rep.RULE, "seed": 2024, "fixed_mul_mil": dict.fromkeys(rep.ROWS, 0.1),
            "start_check": {"same_draws": False, "equal": {"mil": False}, "step1": step1},
            "sets": sets, "sentences": sentences, "labels": labels,
            "decision_sets": list(SETS),
            "verdicts": {n: rep.verdict(labels[n]) for n in rep.DECIDING},
            "candidates": {"K5dfi": rep.candidate(sentences["ignore_vs_reference"]),
                           "K5dfie": rep.candidate(sentences["ego_vs_reference"])},
            "false_alarms": dict.fromkeys(rep.ROWS, 0.5)}


def test_render_prints_verdicts_flag_and_every_contrast() -> None:
    text = rep.render(_synthetic_report())
    assert "K5dfi - K5df" in text and "K5dfie - K5dfi" in text and "K5dfie - K5df" in text
    assert f"`dvs_anchor_mode` ignore vs span): {GAIN}" in text
    assert "FLAG: not the same draws" in text
    assert text.count("| nexar") >= 3 and "CANDIDATE" in text
