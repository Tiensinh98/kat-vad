"""core.tools.kim_clip_host_report (KIM_RUN §64): guards, contrasts, questions. Data-free, CPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.config import Config
from core.tools import kim_ablation_report as ka
from core.tools import kim_clip_host_report as ch
from core.tools.v5_stage1 import _flat

GAIN, COST, NONE = ch.GAIN, ch.COST, ch.NO_DIFFERENCE
DECIDING = ("nexar", "dota", "dada")


def _base() -> dict[str, Any]:
    return _flat(Config().to_dict())


def _cfgs() -> dict[str, dict[str, Any]]:
    l0 = {**_base(), "v2.motion": constants.V2_OFF, "model.motion_dim": 0}
    kim = {"kim.enabled": True, "kim.prototypes": True, "kim.edge_mask": True,
           "kim.seam_mask": True}
    cap = dict.fromkeys(constants.KIM_CLIP_HOST_CAP_FLAGS, True)
    k0 = {**l0, "v2.motion": constants.V5_ENCODER, "model.motion_dim": constants.V5_GRID_DIM}
    return {ch.L0: l0, ch.L5: {**l0, **kim}, ch.L5CAP: {**l0, **cap}, ch.K0: k0,
            ch.K5: {**k0, **kim}}


def test_configs_of_the_campaign_pass() -> None:
    out = ch.check_configs(_cfgs())
    assert out["L5cap_vs_L0"] == sorted(constants.KIM_CLIP_HOST_CAP_FLAGS)
    assert set(out["K0_vs_L0"]) == set(constants.KIM_CLIP_HOST_MOTION_KEYS)


@pytest.mark.parametrize(("rows", "change", "match"), [
    ((ch.L5CAP,), {"kim.edge_mask": True}, "L5cap vs L0"),
    ((ch.K0, ch.K5), {"dvs.theta": 0.5}, "K0 vs L0"),
    ((ch.L5,), {"v2.motion": constants.V5_ENCODER}, "config diff"),
])
def test_configs_refuse_a_second_factor(rows: tuple[str, ...], change: dict[str, Any],
                                        match: str) -> None:
    cfgs = _cfgs()
    for row in rows:
        cfgs[row] = {**cfgs[row], **change}
    with pytest.raises(ValueError, match=match):
        ch.check_configs(cfgs)


def test_interaction_term_is_a_minus_b() -> None:
    for row, sign in ch.TERMS["I"].items():
        a, b = ch.TERMS["A"].get(row, 0), ch.TERMS["B"].get(row, 0)
        assert sign == a - b


def test_micro_interaction_equals_a_minus_b_on_the_point() -> None:
    rng = np.random.default_rng(0)
    labels = {f"v{i}": (np.arange(20) >= 10).astype(np.int64) for i in range(10)}
    curves = {k: {v: rng.normal(size=20) + (0.5 + j / 4) * y for v, y in labels.items()}
              for j, k in enumerate(ch.ROW_KEYS)}
    groups = {v: v for v in labels}
    out = ka.micro_contrasts(curves, labels, groups, ch.TERMS, 30, 1)
    assert out["I"]["mean"] == pytest.approx(out["A"]["mean"] - out["B"]["mean"])
    assert out["I"]["low"] <= out["I"]["mean"] <= out["I"]["high"]


def test_macro_terms_use_the_common_items() -> None:
    items = {k: {"a": 0.5 + j / 10, "b": 0.6 + j / 10, "c": 0.7} for j, k in enumerate(ch.ROW_KEYS)}
    items[ch.K0].pop("c")
    out = ch.macro_terms(items, {"a": "a", "b": "b", "c": "c"})
    a = items[ch.L5]["a"] - items[ch.L0]["a"]
    b = items[ch.K5]["a"] - items[ch.K0]["a"]
    assert out["I"] is not None and out["I"]["mean"] == pytest.approx(a - b)


def _lab(**sets: str) -> dict[str, str]:
    return {s: sets.get(s, NONE) for s in DECIDING}


@pytest.mark.parametrize(("labels", "want"), [
    (_lab(dada=GAIN), ch.SUPPORTED),
    (_lab(dada=GAIN, dota=COST), ch.NO_CLAIM),
    (_lab(nexar=COST), ch.REVERSED),
    (_lab(), ch.NO_CLAIM),
])
def test_q2(labels: dict[str, str], want: str) -> None:
    assert ch.question(labels, DECIDING, ch.SUPPORTED, ch.REVERSED, ch.NO_CLAIM) == want


def test_labelled_only_set_never_decides() -> None:
    labels = {**_lab(), "dad": GAIN}
    assert ch.question(labels, DECIDING, ch.YES, ch.NO, ch.NO) == ch.NO


def test_unmatched_start_is_not_decided() -> None:
    gain = {t: _lab(dada=GAIN) for t in ch.TERMS}
    labels = {ch.MICRO: gain, ch.MACRO: gain}
    got = ch.verdicts(labels, DECIDING, {"L5": True, "L5cap": False, "K5": True})
    assert got[ch.MACRO][ch.Q1] == ch.YES and got[ch.MACRO][ch.Q2] == ch.SUPPORTED
    assert got[ch.MACRO][ch.Q3] == ch.UNMATCHED
    got = ch.verdicts(labels, DECIDING, {"L5": False, "L5cap": True, "K5": True})
    assert {got[ch.MICRO][q] for q in (ch.Q1, ch.Q2, ch.Q3)} == {ch.UNMATCHED}


def test_start_gates_read_step_one(tmp_path: Path) -> None:
    dirs = {}
    for k, mil in zip(ch.ROW_KEYS, (0.70, 0.70, 0.70, 0.69, 0.68), strict=True):
        dirs[k] = tmp_path / k
        dirs[k].mkdir()
        (dirs[k] / "metrics.jsonl").write_text(json.dumps({"global_step": 1, "mil": mil}) + "\n")
    starts = ch.start_gates(dirs)
    assert starts["L5"]["same_draws"] and starts["L5cap"]["same_draws"]
    assert not starts["K5"]["same_draws"]


def test_rows_and_sets_are_defined_for_both_hosts() -> None:
    for host, rows in constants.KIM_CLIP_HOST_ROWS.items():
        assert set(rows) == set(ch.ROW_KEYS) and len(set(rows.values())) == len(rows)
        deciding, labelled = constants.KIM_CLIP_HOST_SETS[host]
        assert deciding and not set(deciding) & set(labelled)
