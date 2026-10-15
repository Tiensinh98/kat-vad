"""v2.2 SC-2: the A3-B adoption rule as pre-registered (v2 addendum §25).

The failures that matter are silent: adopting on a gain below the +0.01 bar, adopting when only
one of the two intervals clears 0, a guardrail read against the wrong arm, and a read-out that
pairs two arms scored on different labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.tools import protocol_b_eval as pb
from core.tools import sc2_readout as sc2
from core.tools.e3_readout import DIAG_DIR, PB_DIR, PRIOR_DIR, REPORTS
from core.tools.position_prior import PRIOR_NPZ
from core.tools.v2_diagnostics import DIAG_JSON

CONTROL, ARM = constants.V22_SC2_CONTROL, constants.V22_SC2_ARM
SEEDS = [f"s{s}" for s in constants.V2_E3_SEEDS]
FRAMES, ONSET, END = 60, 20, 30


def _labels() -> np.ndarray:
    lab = np.zeros(FRAMES, dtype=np.int64)
    lab[ONSET:END] = 1
    return lab


def _interval(mean: float, low: float) -> dict[str, Any]:
    return {"mean": mean, "low": low, "high": mean + (mean - low), "n": 5, "per_seed": {}}


def _ci(mean: float, low: float) -> dict[str, float]:
    return {"mean": mean, "low": low, "high": mean + (mean - low)}


def _diag(macro: float, o1p: bool = True) -> dict[str, Any]:
    return {
        "global_step": 1740,
        "checkpoint": "/nonexistent/checkpoint_last.pt",
        "guardrails": {"macro": macro, "o1_prime": {"pass": o1p}},
        "t2_val_reads": {"normal_window_peak": 0.3,
                         "span_profile": {"clips": 1, "argmax_in_span": 1.0}},
        "source_shortcut_auc": {"v_t": 0.55},
        "position_r2": {"v_t": 0.1},
    }


HOLD = {"hold": True}


class TestDecide:
    def test_every_check_passing_adopts(self) -> None:
        d = sc2.decide(_interval(0.02, 0.005), _ci(0.02, 0.01), _ci(0.01, -0.01), HOLD)
        assert d["base"] == ARM and d["sentence"] == sc2.ADOPT_SENTENCE

    def test_a_gain_below_the_bar_keeps_the_control(self) -> None:
        d = sc2.decide(_interval(0.009, 0.004), _ci(0.009, 0.003), _ci(0.01, 0.0), HOLD)
        assert not d["checks"]["y5c_point_ge_min_gain"] and d["base"] == CONTROL
        assert "does not improve" in d["sentence"]

    def test_both_intervals_must_clear_zero(self) -> None:
        paired_only = sc2.decide(_interval(0.02, 0.005), _ci(0.02, -0.001), _ci(0.0, -0.1), HOLD)
        cluster_only = sc2.decide(_interval(0.02, -0.001), _ci(0.02, 0.01), _ci(0.0, -0.1), HOLD)
        assert paired_only["base"] == CONTROL and cluster_only["base"] == CONTROL

    def test_negative_f2_point_or_a_guardrail_blocks(self) -> None:
        f2 = sc2.decide(_interval(0.02, 0.005), _ci(0.02, 0.01), _ci(-0.001, -0.02), HOLD)
        guard = sc2.decide(_interval(0.02, 0.005), _ci(0.02, 0.01), _ci(0.0, -0.1),
                           {"hold": False})
        assert f2["base"] == CONTROL and guard["base"] == CONTROL


class TestGuardrails:
    def test_inside_the_margin_holds(self) -> None:
        control = {s: _diag(0.73) for s in SEEDS}
        arm = {s: _diag(0.73 - 0.009) for s in SEEDS}
        assert sc2.guardrails(control, arm)["hold"]

    def test_macro_drop_beyond_the_margin_fails(self) -> None:
        control = {s: _diag(0.73) for s in SEEDS}
        arm = {s: _diag(0.73 - 0.011) for s in SEEDS}
        assert not sc2.guardrails(control, arm)["checks"]["t2_val_macro"]

    def test_one_o1_prime_failure_breaks_it(self) -> None:
        control = {s: _diag(0.73) for s in SEEDS}
        arm = {s: _diag(0.73, o1p=(s != SEEDS[0])) for s in SEEDS}
        assert not sc2.guardrails(control, arm)["hold"]

    def test_a_diag_without_o1_prime_is_refused(self) -> None:
        control = {s: _diag(0.73) for s in SEEDS}
        arm = {s: _diag(0.73) for s in SEEDS}
        del arm[SEEDS[0]]["guardrails"]["o1_prime"]
        with pytest.raises(ValueError, match="O1'"):
            sc2.guardrails(control, arm)


# ---------------------------------------------------------------------------
# end to end on synthetic E3 / E4-0 / SC-2 dirs
# ---------------------------------------------------------------------------
CLIPS = [f"vid{i % 7}_{i:06d}" for i in range(40)]


def _write_pb(root: Path, arm: str, scores: dict[str, np.ndarray],
              labels: dict[str, np.ndarray], per_seed: dict[str, float]) -> None:
    d = root / REPORTS / PB_DIR / arm / constants.V2_SPLIT_DOTA_CAP_DEV
    d.mkdir(parents=True, exist_ok=True)
    pb.write_clip_scores(d / pb.CLIP_SCORES_NPZ, scores, labels)
    (d / pb.READOUT_JSON).write_text(json.dumps({
        "runs": SEEDS, "global_steps": dict.fromkeys(SEEDS, 1740), "per_run_macro": per_seed,
    }))


def _write_diags(root: Path, arm: str, macro: float) -> None:
    for s in SEEDS:
        d = root / arm / s / DIAG_DIR
        d.mkdir(parents=True, exist_ok=True)
        (d / DIAG_JSON).write_text(json.dumps(_diag(macro)))


def _dirs(tmp_path: Path, gain: float) -> tuple[Path, Path, Path, dict[str, Any]]:
    rng = np.random.default_rng(1)
    labels = {v: _labels() for v in CLIPS}
    control, arm = {}, {}
    for v in CLIPS:
        base = rng.random(FRAMES) * 0.3
        c = base.copy()
        c[ONSET:ONSET + 3] = 1.0
        a = base.copy()
        a[ONSET:END] = 1.0  # holds the whole span -> a better per-clip read
        control[v], arm[v] = c, a
    e3, e4, out = tmp_path / "e3", tmp_path / "e4_0", tmp_path / "sc2"
    _write_pb(e4, CONTROL, control, labels, {s: 0.79 + 0.001 * i for i, s in enumerate(SEEDS)})
    _write_pb(out, ARM, arm, labels,
              {s: 0.79 + gain + 0.0012 * i for i, s in enumerate(SEEDS)})
    p = e3 / REPORTS / PRIOR_DIR / constants.V2_SPLIT_DOTA_CAP_DEV
    p.mkdir(parents=True)
    pb.write_clip_scores(p / PRIOR_NPZ, {v: np.linspace(0.0, 1.0, FRAMES) for v in CLIPS}, labels)
    _write_diags(e4, CONTROL, 0.73)
    _write_diags(out, ARM, 0.73)
    alignment = {"clips": {v: {"rate": 1.0 if i % 2 else 3.0} for i, v in enumerate(CLIPS)}}
    return e3, e4, out, alignment


def _argv(e3: Path, e4: Path, out: Path) -> list[str]:
    return ["--e3-dir", str(e3), "--e4-dir", str(e4), "--sc2-dir", str(out)]


class TestRun:
    def test_a_clear_gain_is_adopted(self, tmp_path: Path) -> None:
        e3, e4, out, alignment = _dirs(tmp_path, 0.03)
        (tmp_path / "align.json").write_text(json.dumps(alignment))
        sc2.main([*_argv(e3, e4, out), "--alignment", str(tmp_path / "align.json")])
        r = json.loads((out / REPORTS / sc2.READOUT_JSON).read_text())
        assert r["decision"]["base"] == ARM
        assert set(r["rate_groups"]) == {"r1", "r3"}
        assert "Base = A3B" in (out / REPORTS / sc2.READOUT_MD).read_text()

    def test_a_per_seed_gain_below_the_bar_is_not_adopted(self, tmp_path: Path) -> None:
        e3, e4, out, _ = _dirs(tmp_path, 0.005)
        r = sc2.run(sc2.build_arg_parser().parse_args(_argv(e3, e4, out)))
        assert r["decision"]["checks"]["y5a_paired_t95_above_0"]
        assert not r["decision"]["checks"]["y5c_point_ge_min_gain"]
        assert r["decision"]["base"] == CONTROL

    def test_different_labels_are_refused(self, tmp_path: Path) -> None:
        e3, e4, out, _ = _dirs(tmp_path, 0.03)
        d = out / REPORTS / PB_DIR / ARM / constants.V2_SPLIT_DOTA_CAP_DEV
        scores, labels = pb.read_clip_scores(d / pb.CLIP_SCORES_NPZ)
        labels[CLIPS[0]] = np.roll(labels[CLIPS[0]], 1)
        pb.write_clip_scores(d / pb.CLIP_SCORES_NPZ, scores, labels)
        with pytest.raises(ValueError, match="labels differ"):
            sc2.main(_argv(e3, e4, out))
