"""E3 read-out (Amendment 9): the rules are applied mechanically and nothing is read twice.

The failures that matter are silent: a contrast paired across different seeds, an arm adopted
with one collapsed seed, a costly arm adopted that beats A0 but not the free arm, or a
protocol-B read-out of another checkpoint than the one diagnosed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.tools import e3_readout as e3
from core.tools.position_prior import PRIOR_NPZ
from core.tools.position_prior import READOUT_JSON as PRIOR_READOUT_JSON
from core.tools.protocol_b_eval import (
    CLIP_AUCS_JSON,
    CLIP_SCORES_NPZ,
    write_clip_scores,
)
from core.tools.protocol_b_eval import READOUT_JSON as PB_READOUT_JSON
from core.tools.v2_diagnostics import DIAG_JSON

SEEDS = ["s2024", "s2025", "s2026", "s2027", "s2028"]
STEP = 1740


def _interval(mean: float, low: float, high: float) -> dict[str, Any]:
    return {"mean": mean, "low": low, "high": high, "sd": 0.01, "n": 5, "per_seed": {}}


class TestPairedInterval:
    def test_matches_the_t95_formula(self) -> None:
        x = dict(zip(SEEDS, [0.70, 0.71, 0.72, 0.69, 0.73], strict=True))
        y = dict(zip(SEEDS, [0.68, 0.70, 0.70, 0.68, 0.70], strict=True))
        r = e3.paired_interval(x, y, constants.V2_E3_T95)
        d = np.array([0.02, 0.01, 0.02, 0.01, 0.03])
        half = constants.V2_E3_T95 * d.std(ddof=1) / np.sqrt(5)
        assert r["mean"] == pytest.approx(d.mean())
        assert r["low"] == pytest.approx(d.mean() - half)
        assert r["high"] == pytest.approx(d.mean() + half)

    def test_different_seeds_are_refused(self) -> None:
        with pytest.raises(ValueError, match="different seeds"):
            e3.paired_interval({"s1": 0.1, "s2": 0.2}, {"s1": 0.1, "s3": 0.2}, 2.776)

    def test_positive_needs_the_lower_bound_above_zero(self) -> None:
        assert e3.positive(_interval(0.02, 0.001, 0.04))
        assert not e3.positive(_interval(0.02, -0.001, 0.04))
        assert not e3.positive(_interval(-0.02, -0.04, -0.01))
        assert e3.excludes_zero(_interval(-0.02, -0.04, -0.01))

    def test_mde_pools_variances(self) -> None:
        a = {"sd": 0.03, "n": 5}
        b = {"sd": 0.04, "n": 5}
        expected = 2.776 * np.sqrt((0.03**2 + 0.04**2) / 2) / np.sqrt(5)
        assert e3.mde_e3([a, b], 2.776) == pytest.approx(expected)


class TestSpearman:
    def test_ties_share_their_mean_rank(self) -> None:
        np.testing.assert_allclose(e3.average_ranks(np.array([3.0, 1.0, 3.0, 2.0])),
                                   [2.5, 0.0, 2.5, 1.0])

    def test_monotone_is_one_and_reversed_is_minus_one(self) -> None:
        a = np.array([0.1, 0.5, 0.2, 0.9])
        assert e3.spearman(a, a ** 3) == pytest.approx(1.0)
        assert e3.spearman(a, -a) == pytest.approx(-1.0)

    def test_constant_and_single_class_clips_are_not_scored(self) -> None:
        scores = {"a": np.array([1.0, 1.0, 1.0]), "b": np.array([0.1, 0.2, 0.3]),
                  "c": np.array([0.3, 0.2, 0.1])}
        prior = {v: np.array([0.0, 1.0, 2.0]) for v in scores}
        labels = {"a": np.array([0, 1, 0]), "b": np.array([0, 1, 1]), "c": np.array([0, 0, 0])}
        rho, dropped = e3.clip_spearman(scores, prior, labels)
        assert rho == {"b": pytest.approx(1.0)}
        assert dropped == 1


def _diag(o1: bool, o1p: bool | None, macro: float = 0.67) -> dict[str, Any]:
    g: dict[str, Any] = {"pass": o1, "macro": macro, "micro": 0.66, "window_level_auc": 0.69,
                         "clip_oracle_micro": 0.70}
    if o1p is not None:
        g["o1_prime"] = {"pass": o1p}
    return {"guardrails": g, "global_step": STEP}


class TestGuardrails:
    def test_one_failed_seed_breaks_the_arm(self) -> None:
        diags = {s: _diag(False, s != "s2026") for s in SEEDS}
        g = e3.guardrails("A2", diags)
        assert not g["hold"]
        assert g["per_seed"]["s2026"] is False

    def test_o1_registered_is_printed_not_decided(self) -> None:
        diags = {s: _diag(False, True) for s in SEEDS}
        assert e3.guardrails("A3", diags)["hold"]

    def test_a_diag_without_o1_prime_is_refused(self) -> None:
        with pytest.raises(ValueError, match="a0-diag"):
            e3.guardrails("A1", {s: _diag(True, None) for s in SEEDS})

    def test_a0_is_read_under_o1(self) -> None:
        g = e3.guardrails("A0", {s: _diag(True, None) for s in SEEDS})
        assert g["hold"] and g["rule"].startswith("O1")


class TestFreeRuleAndDecision:
    def test_a_reversed_bin_fails_the_free_rule(self) -> None:
        bins = {"<30": {"mean": -0.05, "low": -0.08, "high": -0.01}, "30-50": None}
        free = e3.free_rule(_interval(0.01, -0.01, 0.03), 0.002, bins)
        assert not free["pass"] and free["reversed_bins"] == ["<30"]

    def test_negative_t2_fails_the_free_rule(self) -> None:
        free = e3.free_rule(_interval(0.01, -0.01, 0.03), -0.001, {})
        assert not free["pass"]

    @staticmethod
    def _contrasts(a3_a0: Any, a3_a1: Any, a2_a0: Any) -> dict[str, Any]:
        return {"A1 - A0": _interval(0.01, -0.01, 0.03), "A3 - A0": a3_a0,
                "A3 - A1": a3_a1, "A2 - A0": a2_a0, "A2 - A1": a2_a0}

    def test_a3_must_beat_f_not_only_a0(self) -> None:
        up, flat = _interval(0.04, 0.01, 0.07), _interval(0.01, -0.02, 0.04)
        guards = {a: {"hold": True} for a in constants.V2_E3_ARMS}
        decision = e3.decide(self._contrasts(up, flat, flat), guards, {"pass": True})
        assert decision == {"adopt": "A1", "F": "A1"}

    def test_a3_wins_when_f_is_a0(self) -> None:
        up, flat = _interval(0.04, 0.01, 0.07), _interval(0.01, -0.02, 0.04)
        guards = {a: {"hold": True} for a in constants.V2_E3_ARMS}
        decision = e3.decide(self._contrasts(up, flat, flat), guards, {"pass": False})
        assert decision == {"adopt": "A3", "F": "A0"}

    def test_a_broken_guardrail_blocks_a_significant_arm(self) -> None:
        up = _interval(0.04, 0.01, 0.07)
        guards = {a: {"hold": a != "A3"} for a in constants.V2_E3_ARMS}
        decision = e3.decide(self._contrasts(up, up, up), guards, {"pass": False})
        assert decision["adopt"] == "A2"

    def test_nothing_clears_is_a_bounded_null(self) -> None:
        flat = _interval(0.01, -0.02, 0.04)
        guards = {a: {"hold": True} for a in constants.V2_E3_ARMS}
        decision = e3.decide(self._contrasts(flat, flat, flat), guards, {"pass": False})
        assert decision["adopt"] == "A0"
        claims = e3.claims("A0", self._contrasts(flat, flat, flat), 0.03)
        assert "bounded null" in claims[0]

    def test_claims_never_say_motion_as_a_finding(self) -> None:
        up = _interval(0.04, 0.01, 0.07)
        for arm in ("A3", "A2"):
            text = " ".join(e3.claims(arm, self._contrasts(up, up, up), 0.03))
            assert "video stream (V2-S)" in text
            assert "improves motion" not in text


# ---------------------------------------------------------------------------
# end to end on a synthetic E3 dir
# ---------------------------------------------------------------------------
CLIPS = [f"vid{k}_{j:06d}" for k in range(6) for j in range(3)]
FRAMES = 12


def _labels() -> dict[str, np.ndarray]:
    out = {}
    for i, v in enumerate(CLIPS):
        lab = np.zeros(FRAMES, dtype=np.int64)
        lab[3 + i % 4: 7 + i % 4] = 1
        out[v] = lab
    return out


def _write_pb(root: Path, arm: str, split: str, bump: float, steps: dict[str, int]) -> None:
    d = root / e3.REPORTS / e3.PB_DIR / arm / split
    d.mkdir(parents=True)
    labels = _labels()
    rng = np.random.default_rng(len(arm) + int(bump * 1000))
    scores = {v: labels[v] * (0.5 + bump) + rng.random(FRAMES) * 0.6 for v in CLIPS}
    write_clip_scores(d / CLIP_SCORES_NPZ, scores, labels)
    aucs = {v: 0.6 + bump + 0.01 * (i % 3) for i, v in enumerate(CLIPS)}
    (d / CLIP_AUCS_JSON).write_text(json.dumps(aucs))
    per_run = {s: 0.62 + bump + 0.003 * k for k, s in enumerate(SEEDS)}
    readout = {"runs": SEEDS, "global_steps": steps, "per_run_macro": per_run,
               "macro": {"mean": 0.62 + bump, "low": 0.6, "high": 0.64},
               "position_ruler": {"mean": 0.55, "low": 0.5, "high": 0.6}}
    (d / PB_READOUT_JSON).write_text(json.dumps(readout))


def _write_diag(root: Path, arm: str, seed: str, macro: float) -> None:
    d = root / arm / seed / e3.DIAG_DIR
    d.mkdir(parents=True)
    diag = _diag(True, None if arm == "A0" else True, macro)
    diag["source_shortcut_auc"] = {"v_t": 0.9, "y_bin": 0.55}
    diag["position_r2"] = {"v_t": -0.1}
    diag["motion_share"] = None
    (d / DIAG_JSON).write_text(json.dumps(diag))


def _write_prior(root: Path, split: str) -> None:
    d = root / e3.REPORTS / e3.PRIOR_DIR / split
    d.mkdir(parents=True)
    tau = (np.arange(FRAMES) + 0.5) / FRAMES
    write_clip_scores(d / PRIOR_NPZ, dict.fromkeys(CLIPS, -(tau - 0.5) ** 2), _labels())
    (d / PRIOR_READOUT_JSON).write_text(json.dumps(
        {"p_t2_macro": {"mean": 0.7, "low": 0.68, "high": 0.72}}))


def _e3_dir(root: Path, bumps: dict[str, float], bad_step: bool = False) -> Path:
    steps = dict.fromkeys(SEEDS, STEP)
    for arm in constants.V2_E3_ARMS:
        for split in e3.SPLITS_OF[arm]:
            arm_steps = dict(steps)
            if bad_step and arm == "A2":
                arm_steps["s2025"] = 510
            _write_pb(root, arm, split, bumps[arm], arm_steps)
        for seed in SEEDS:
            _write_diag(root, arm, seed, 0.67 + bumps[arm])
    for split in (e3.DEV, e3.CAP):
        _write_prior(root, split)
    return root


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(e3_dir=root, seeds=list(constants.V2_E3_SEEDS), seed=2024)


class TestEndToEnd:
    def test_uniform_gains_adopt_a3_and_write_the_readout(self, tmp_path: Path) -> None:
        root = _e3_dir(tmp_path, {"A0": 0.0, "A1": 0.01, "A2": 0.03, "A3": 0.05})
        # per-seed deltas are constant -> SD 0 -> the interval is the point
        r = e3.run(_args(root))
        assert r["free_rule"]["pass"]
        assert r["decision"] == {"adopt": "A3", "F": "A1"}
        assert r["contrasts"]["A3 - A1"]["split"] == e3.CAP
        assert r["contrasts"]["A1 - A0"]["split"] == e3.DEV
        md = (root / e3.REPORTS / e3.E3_READOUT_MD).read_text(encoding="utf-8")
        assert "adopt A3" in md and "MDE_E3" in md and "motion" in md
        assert (root / e3.REPORTS / e3.E3_READOUT_JSON).exists()

    def test_a_step_mismatch_is_refused(self, tmp_path: Path) -> None:
        root = _e3_dir(tmp_path, {"A0": 0.0, "A1": 0.01, "A2": 0.03, "A3": 0.05}, bad_step=True)
        with pytest.raises(ValueError, match="not the same checkpoint"):
            e3.run(_args(root))

    def test_a_missing_run_is_refused(self, tmp_path: Path) -> None:
        root = _e3_dir(tmp_path, {"A0": 0.0, "A1": 0.01, "A2": 0.03, "A3": 0.05})
        (root / "A3" / "s2028" / e3.DIAG_DIR / DIAG_JSON).unlink()
        with pytest.raises(FileNotFoundError, match="E3 input missing"):
            e3.run(_args(root))
