"""core.tools.kim_nkc_report (KIM_RUN §66 NK4-NK7): guards, drift print, verdict. Data-free, CPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import constants
from core.config import Config
from core.tools import kim_nkc_report as nk
from core.tools.v5_stage1 import _flat

GAIN, COST, NONE = nk.GAIN, nk.COST, constants.KIM_NKC_NO_DIFFERENCE
DECIDING = constants.KIM_NKC_SETS[0]
KEY = constants.KIM_NKC_WEIGHT_KEY


def _l5() -> dict[str, Any]:
    return {**_flat(Config().to_dict()), "v2.motion": constants.V2_OFF, "model.motion_dim": 0,
            "kim.enabled": True, "kim.prototypes": True, "kim.edge_mask": True,
            "kim.seam_mask": True}


def test_config_allows_only_the_weight_to_zero() -> None:
    ctrl = _l5()
    out = nk.check_config(ctrl, {**ctrl, KEY: 0.0})
    assert out == {KEY: [ctrl[KEY], 0.0]} and ctrl[KEY] > 0


@pytest.mark.parametrize(("change", "match"), [
    ({KEY: 0.0, "dvs.theta": 0.5}, "config diff"),
    ({KEY: 0.5}, "needs > 0"),
    ({}, "config diff"),
])
def test_config_refuses_anything_else(change: dict[str, Any], match: str) -> None:
    ctrl = _l5()
    with pytest.raises(ValueError, match=match):
        nk.check_config(ctrl, {**ctrl, **change})


def test_config_refuses_a_control_without_prototypes() -> None:
    ctrl = {**_l5(), "kim.prototypes": False}
    with pytest.raises(ValueError, match="never ran"):
        nk.check_config(ctrl, {**ctrl, KEY: 0.0})


def _metrics(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.mkdir(parents=True)
    (path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_pre_nkc_drift_stops_at_the_first_nkc_step(tmp_path: Path) -> None:
    a = _metrics(tmp_path / "a", [{"global_step": 1, "total": 3.0},
                                  {"global_step": 2, "total": 2.0},
                                  {"global_step": 3, "total": 1.0, "kim_nkc": 2.0}])
    b = _metrics(tmp_path / "b", [{"global_step": 1, "total": 3.0},
                                  {"global_step": 2, "total": 2.5},
                                  {"global_step": 3, "total": 9.0, "kim_nkc": 2.0}])
    out = nk.pre_nkc_drift(a, b)
    assert out == {"first_nkc_step": 3, "steps_compared": 2, "max_abs_total": pytest.approx(0.5)}


def _lab(**sets: str) -> dict[str, str]:
    return {s: sets.get(s, NONE) for s in (*DECIDING, "dad")}


@pytest.mark.parametrize(("labels", "want"), [
    (_lab(dada=GAIN), constants.KIM_NKC_HELPS),
    (_lab(nexar=COST), constants.KIM_NKC_COSTS),
    (_lab(), constants.KIM_NKC_NO_DIFFERENCE),
    (_lab(dota=GAIN, dada=COST), constants.KIM_NKC_MIXED),
    (_lab(dad=GAIN), constants.KIM_NKC_NO_DIFFERENCE),
])
def test_verdict(labels: dict[str, str], want: str) -> None:
    assert nk.verdict(labels, DECIDING) == want


def test_macro_contrast_is_control_minus_row_on_common_items() -> None:
    items = {nk.CTRL: {"a": 0.7, "b": 0.8, "c": 0.9}, nk.NEW: {"a": 0.6, "b": 0.6}}
    out = nk.macro_contrast(items, {})
    assert out is not None and out["mean"] == pytest.approx(0.15)


def test_render_names_both_rows() -> None:
    cell = {"mean": 0.01, "low": -0.01, "high": 0.03}
    sets = {s: {"cells": {k: {nk.MICRO: 0.7, nk.MACRO: 0.7} for k in (nk.CTRL, nk.NEW)},
                nk.MICRO: cell, nk.MACRO: cell} for s in (*DECIDING, "dad")}
    report = {"rows": {nk.CTRL: {"name": "L5dfi", "env": None},
                       nk.NEW: {"name": "L5dfin0", "env": None}},
              "seed": 2024, "label": nk.LABEL, "deciding_sets": list(DECIDING),
              "labelled_sets": ["dad"], "config_diff": {KEY: [0.1, 0.0]},
              "start": {"control": 0.72, "new": 0.72}, "verdicts": {nk.MICRO: NONE, nk.MACRO: NONE},
              "pre_nkc": {"first_nkc_step": 221, "steps_compared": 220, "max_abs_total": 0.0},
              "sets": sets, "labels": {m: _lab() for m in (nk.MICRO, nk.MACRO)}}
    text = nk.render(report)
    assert "L5dfi - L5dfin0" in text and "NO DIFFERENCE" in text
