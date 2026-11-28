"""core.tools.kim_edge_fill_report (KIM_RUN §68): guards, slot ratio, contrasts, verdict. CPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import constants
from core.tools import kim_edge_fill_report as ef

KEY = constants.KIM_EDGE_FILL_KEY
GAIN, COST, NONE = ef.GAIN, ef.COST, constants.KIM_DVS_IGNORE_NO_DIFFERENCE
BASE = {"kim.enabled": True, "kim.edge_mask": True, KEY: False}


def test_config_allows_only_the_fill_flag() -> None:
    assert ef.check_config(BASE, BASE | {KEY: True}) == {KEY: [False, True]}
    with pytest.raises(ValueError, match="allows exactly"):
        ef.check_config(BASE, BASE | {KEY: True, "kim.edge_mask": False})
    with pytest.raises(ValueError, match="false -> true"):
        ef.check_config(BASE | {KEY: True}, BASE)


def test_slot_ratio() -> None:
    profile = [0.155, 0.644, 0.485] + [0.0565] * 9
    assert ef.slot_ratio(profile) == pytest.approx(0.5645 / 0.0565)
    assert ef.slot_ratio([0.1, 0.2]) is None
    assert ef.slot_ratio([0.1, 0.2, 0.2] + [0.0] * 9) is None


def test_read_slot_takes_the_l5_row(tmp_path: Path) -> None:
    ci = {"mean": 0.6, "low": 0.5, "high": 0.7}
    row = {"name": "L5dfiDoTAef", "position_profile": [0.2] * 12,
           "macro": {"whole": ci, "windowed": ci}}
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"rows": {constants.KIM_WG_ROWS[0]: row}}), encoding="utf-8")
    got = ef.read_slot(path)
    assert got is not None and got["row"] == "L5dfiDoTAef" and got["ratio"] == pytest.approx(1.0)
    assert ef.read_slot(None) is None


def test_macro_terms_pair_over_common_items() -> None:
    items = {ef.TRUNK: {"a": 0.6, "b": 0.6, "c": 0.6}, ef.FAILED: {"a": 0.4, "b": 0.4, "c": 0.4},
             ef.NEW: {"a": 0.7, "b": 0.7, "c": 0.7, "z": 0.1}}
    got = ef.macro_terms(items, {})
    assert got["E"] is not None and got["E"]["mean"] == pytest.approx(0.1)
    assert got["F"] is not None and got["F"]["mean"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="share no"):
        ef.macro_terms({ef.TRUNK: {"a": 1.0}, ef.FAILED: {"b": 1.0}, ef.NEW: {"a": 1.0}}, {})


@pytest.mark.parametrize(("dota", "dada", "want"), [
    (GAIN, NONE, constants.KIM_EDGE_FILL_HELPS),
    (NONE, NONE, constants.KIM_EDGE_FILL_NO_DIFFERENCE),
    (COST, NONE, constants.KIM_EDGE_FILL_COSTS),
    (GAIN, COST, constants.KIM_EDGE_FILL_MIXED),
])
def test_verdict(dota: str, dada: str, want: str) -> None:
    assert ef.verdict({"dota": dota, "dada": dada}, ("dota", "dada")) == want


def test_render() -> None:
    ci = {"mean": 0.05, "low": 0.01, "high": 0.09}
    cells = {k: {ef.MICRO: 0.6, ef.MACRO: 0.65} for k in ef.ROW_KEYS}
    per_set = {s: {"cells": cells, ef.MICRO: dict.fromkeys(ef.TERMS, ci),
                   ef.MACRO: dict.fromkeys(ef.TERMS, ci)} for s in ("dota", "dada")}
    report = {"label": ef.LABEL, "seed": 2024, "sets": ["dota", "dada"],
              "rows": {k: {"name": k, "env": "T4"} for k in ef.ROW_KEYS},
              "config_diff": {KEY: [False, True]},
              "start": {"control": 0.69, "new": 0.69}, "start_failed": {"same_draws": True},
              "verdicts": {ef.MICRO: "KIM HELPS", ef.MACRO: "NO DIFFERENCE"},
              "per_set": per_set,
              "labels": {m: {t: {"dota": GAIN, "dada": GAIN} for t in ef.TERMS}
                         for m in (ef.MICRO, ef.MACRO)},
              "slot": {"row": "fill", "ratio": 1.1, "whole_macro": ci, "windowed_macro": ci}}
    text = ef.render(report)
    assert "E = fill - trunk" in text and "1.10x" in text and "KIM HELPS" in text
