"""v2 Amendment 20: the drop-CRN non-inferiority rule as pre-registered (addendum §31).

The failures that matter are silent: dropping CRN when only the total macro is non-inferior and
the beyond-position read is not, reading "> -M" as "> 0" (or as ">= -M"), a guardrail read
against the wrong arm name, and pairing two arms scored on different labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.tools import crn_drop_readout as cd
from core.tools import protocol_b_eval as pb
from core.tools.e3_readout import DIAG_DIR, PB_DIR, PRIOR_DIR, REPORTS
from core.tools.position_prior import PRIOR_NPZ
from core.tools.v2_diagnostics import DIAG_JSON

CONTROL, ARM, M = cd.CONTROL, cd.ARM, constants.V2_CRNDROP_MARGIN
SEEDS = [f"s{s}" for s in constants.V2_E3_SEEDS]
FRAMES, ONSET, END = 60, 20, 30
HOLD = {"hold": True}


def _ci(mean: float, low: float) -> dict[str, float]:
    return {"mean": mean, "low": low, "high": mean + (mean - low)}


def _interval(mean: float, low: float) -> dict[str, Any]:
    return {**_ci(mean, low), "n": 5, "per_seed": {}}


def _diag(macro: float, o1p: bool = True) -> dict[str, Any]:
    return {
        "global_step": 1740,
        "checkpoint": "/nonexistent/checkpoint_last.pt",
        "guardrails": {"macro": macro, "o1_prime": {"pass": o1p}},
        "t2_val_reads": {"normal_window_peak": 0.3,
                         "span_profile": {"clips": 1, "argmax_in_span": 1.0}},
        "source_shortcut_auc": {"v_t": 0.99},
        "position_r2": {"v_t": 0.1},
    }


class TestConstants:
    def test_the_arms_are_e4_0s_control_and_its_no_crn_twin(self) -> None:
        assert CONTROL == constants.V2_E4_0_ARM == "A3ign"
        assert ARM == "A2ign" and M == 0.01


class TestDecide:
    def test_a_small_loss_inside_the_margin_drops_crn(self) -> None:
        d = cd.decide(_interval(-0.003, -0.008), _ci(-0.002, -0.009), _ci(-0.004, -0.009), HOLD)
        assert d["drop_crn"] and d["base"] == ARM and d["sentence"] == cd.DROP_SENTENCE
        assert d["failed"] == []

    def test_macro_non_inferior_but_f2_not_keeps_crn(self) -> None:
        # E3's prediction: total macro ties, beyond position A2 trails A3 by ~0.016.
        d = cd.decide(_interval(0.0, -0.006), _ci(0.002, -0.005), _ci(-0.016, -0.025), HOLD)
        assert not d["drop_crn"] and d["base"] == CONTROL
        assert d["failed"] == ["z5_f2_ci_low_gt_minus_margin"]
        assert d["sentence"] == cd.KEEP_SENTENCE and "helps" not in d["sentence"]

    def test_the_bound_is_strict_and_at_minus_the_margin(self) -> None:
        at = cd.decide(_interval(0.0, -M), _ci(0.0, -0.001), _ci(0.0, -0.001), HOLD)
        assert not at["checks"]["z4a_paired_t95_low_gt_minus_margin"]
        below_zero = cd.decide(_interval(-0.002, -0.009), _ci(0.0, -0.001), _ci(0.0, -0.001), HOLD)
        assert below_zero["drop_crn"]  # non-inferiority, not superiority

    def test_both_detection_intervals_and_the_guardrails_must_hold(self) -> None:
        cluster = cd.decide(_interval(0.0, -0.005), _ci(0.0, -0.02), _ci(0.0, -0.001), HOLD)
        guard = cd.decide(_interval(0.0, -0.005), _ci(0.0, -0.001), _ci(0.0, -0.001),
                          {"hold": False})
        missing = cd.decide(_interval(0.0, -0.005), None, _ci(0.0, -0.001), HOLD)
        assert not (cluster["drop_crn"] or guard["drop_crn"] or missing["drop_crn"])


# ---------------------------------------------------------------------------
# end to end on synthetic E3 / E4-0 / Amendment 20 dirs
# ---------------------------------------------------------------------------
CLIPS = [f"vid{i % 7}_{i:06d}" for i in range(40)]


def _labels() -> np.ndarray:
    lab = np.zeros(FRAMES, dtype=np.int64)
    lab[ONSET:END] = 1
    return lab


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


def _dirs(tmp_path: Path, arm_holds_span: bool, gain: float,
          with_e3_twin: bool = True) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(1)
    labels = {v: _labels() for v in CLIPS}
    control, arm = {}, {}
    for v in CLIPS:
        base = rng.random(FRAMES) * 0.3
        c = base.copy()
        c[ONSET:END] = 1.0
        a = base.copy()
        a[ONSET:END if arm_holds_span else ONSET + 2] = 1.0
        if not arm_holds_span:
            a[:ONSET] += 0.6  # a worse per-clip read -> beyond the margin
        control[v], arm[v] = c, a
    e3, e4, out = tmp_path / "e3", tmp_path / "e4_0", tmp_path / "crndrop"
    _write_pb(e4, CONTROL, control, labels, {s: 0.79 + 0.001 * i for i, s in enumerate(SEEDS)})
    _write_pb(out, ARM, arm, labels, {s: 0.79 + gain + 0.0012 * i for i, s in enumerate(SEEDS)})
    if with_e3_twin:
        _write_pb(e3, cd.E3_TWIN, arm, labels, dict.fromkeys(SEEDS, 0.75))
    p = e3 / REPORTS / PRIOR_DIR / constants.V2_SPLIT_DOTA_CAP_DEV
    p.mkdir(parents=True)
    pb.write_clip_scores(p / PRIOR_NPZ, {v: np.linspace(0.0, 1.0, FRAMES) for v in CLIPS}, labels)
    _write_diags(e4, CONTROL, 0.73)
    _write_diags(out, ARM, 0.73)
    return e3, e4, out


def _argv(e3: Path, e4: Path, out: Path) -> list[str]:
    return ["--e3-dir", str(e3), "--e4-dir", str(e4), "--out-dir", str(out)]


class TestRun:
    def test_an_identical_read_drops_crn(self, tmp_path: Path) -> None:
        e3, e4, out = _dirs(tmp_path, arm_holds_span=True, gain=0.0)
        cd.main(_argv(e3, e4, out))
        r = json.loads((out / REPORTS / cd.READOUT_JSON).read_text())
        assert r["decision"]["base"] == ARM and r["margin"] == M
        assert r["e3_a2_span_macro"] == pytest.approx(r["macro"][ARM])
        assert set(r["guardrails"]["t2_val_macro"]) == {CONTROL, ARM}
        assert f"Base = {ARM}" in (out / REPORTS / cd.READOUT_MD).read_text()

    def test_a_worse_per_clip_read_keeps_crn(self, tmp_path: Path) -> None:
        e3, e4, out = _dirs(tmp_path, arm_holds_span=False, gain=0.0, with_e3_twin=False)
        r = cd.run(cd.build_arg_parser().parse_args(_argv(e3, e4, out)))
        assert r["decision"]["checks"]["z4a_paired_t95_low_gt_minus_margin"]
        assert not r["decision"]["checks"]["z4b_cluster_ci_low_gt_minus_margin"]
        assert r["decision"]["base"] == CONTROL and r["e3_a2_span_macro"] is None

    def test_different_labels_are_refused(self, tmp_path: Path) -> None:
        e3, e4, out = _dirs(tmp_path, arm_holds_span=True, gain=0.0)
        d = out / REPORTS / PB_DIR / ARM / constants.V2_SPLIT_DOTA_CAP_DEV
        scores, labels = pb.read_clip_scores(d / pb.CLIP_SCORES_NPZ)
        labels[CLIPS[0]] = np.roll(labels[CLIPS[0]], 1)
        pb.write_clip_scores(d / pb.CLIP_SCORES_NPZ, scores, labels)
        with pytest.raises(ValueError, match="labels differ"):
            cd.main(_argv(e3, e4, out))

    def test_a_diag_from_another_step_is_refused(self, tmp_path: Path) -> None:
        e3, e4, out = _dirs(tmp_path, arm_holds_span=True, gain=0.0)
        path = out / ARM / SEEDS[0] / DIAG_DIR / DIAG_JSON
        diag = json.loads(path.read_text())
        diag["global_step"] = 1060
        path.write_text(json.dumps(diag))
        with pytest.raises(ValueError, match="not the same checkpoint"):
            cd.main(_argv(e3, e4, out))
