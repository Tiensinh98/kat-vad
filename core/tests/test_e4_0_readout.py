"""v2.1 E4-0: the span reads, the guardrails and the base rule as pre-registered (§11.4).

The failures that matter are silent: a span read that a global rescale can move (``ignore``
lowers every positive score, so a raw-scale read would show a "falling pre-anomaly level" for
free), a guardrail that compares against the wrong arm, a base rule that adopts A3-ign on a
point estimate, and a read-out that pairs diagnostics and protocol B from different checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from core import constants
from core.metrics import normal_window_peak, span_reads, span_summary
from core.tools import e4_0_readout as e4
from core.tools import protocol_b_eval as pb
from core.tools.e3_readout import DIAG_DIR, PB_DIR, PRIOR_DIR, REPORTS
from core.tools.position_prior import PRIOR_NPZ
from core.tools.v2_diagnostics import DIAG_JSON

CONTROL, ARM = constants.V2_E4_0_CONTROL, constants.V2_E4_0_ARM
SEEDS = [f"s{s}" for s in constants.V2_E3_SEEDS]
FRAMES, ONSET, END = 60, 20, 30  # every synthetic clip has >= 30 normal frames after its span


def _labels() -> np.ndarray:
    lab = np.zeros(FRAMES, dtype=np.int64)
    lab[ONSET:END] = 1
    return lab


def _interval(mean: float, low: float, high: float) -> dict[str, Any]:
    return {"mean": mean, "low": low, "high": high, "n": 5, "per_seed": {}}


def _ci(mean: float, low: float, high: float) -> dict[str, float]:
    return {"mean": mean, "low": low, "high": high}


class TestSpanReads:
    def test_global_rescale_cannot_move_them(self) -> None:
        rng = np.random.default_rng(0)
        scores = {"v_000001": rng.random(FRAMES)}
        labels = {"v_000001": _labels()}
        a = span_reads(scores, labels)
        b = span_reads({k: 0.3 * v + 0.1 for k, v in scores.items()}, labels)
        for key in a["v_000001"]:
            assert a["v_000001"][key] == pytest.approx(b["v_000001"][key])

    def test_reads_locate_the_peak(self) -> None:
        score = np.zeros(FRAMES)
        score[ONSET + 2] = 1.0
        read = span_reads({"v_000001": score}, {"v_000001": _labels()})["v_000001"]
        assert read["argmax_in_span"] == 1.0
        assert read["pre_mean"] == pytest.approx(0.0)
        assert read["in_mean"] == pytest.approx(1.0 / (END - ONSET), rel=1e-6)

    def test_no_pre_frames_means_no_pre_mean(self) -> None:
        lab = np.zeros(FRAMES, dtype=np.int64)
        lab[:5] = 1
        read = span_reads({"v_000001": np.arange(FRAMES, dtype=float)}, {"v_000001": lab})
        assert "pre_mean" not in read["v_000001"]
        assert span_summary(read)["pre_mean"] is None

    def test_single_class_clips_are_skipped(self) -> None:
        reads = span_reads({"v_000001": np.ones(FRAMES)},
                           {"v_000001": np.ones(FRAMES, dtype=np.int64)})
        assert reads == {}

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="scores vs"):
            span_reads({"v_000001": np.ones(FRAMES - 1)}, {"v_000001": _labels()})


class TestNormalWindowPeak:
    def test_mean_top_k_over_normal_windows_only(self) -> None:
        normal = np.linspace(0.0, 1.0, 20)  # k = 20 // 5 = 4 -> mean of the top 4
        abnormal = np.ones(20)
        lab_n, lab_a = np.zeros(20, dtype=np.int64), np.ones(20, dtype=np.int64)
        got = normal_window_peak([normal, abnormal], [lab_n, lab_a], 5)
        assert got == pytest.approx(float(np.sort(normal)[-4:].mean()))

    def test_no_normal_window_is_none(self) -> None:
        assert normal_window_peak([np.ones(20)], [np.ones(20, dtype=np.int64)], 5) is None


def _diag(macro: float, peak: float | None, o1p: bool = True, step: int = 1740) -> dict[str, Any]:
    return {
        "global_step": step,
        "checkpoint": "/nonexistent/checkpoint_last.pt",
        "guardrails": {"macro": macro, "o1_prime": {"pass": o1p}},
        "t2_val_reads": {"normal_window_peak": peak,
                         "span_profile": {"clips": 1, "argmax_in_span": 1.0,
                                          "in_mean": 0.8, "pre_mean": 0.2}},
        "source_shortcut_auc": {"v_t": 0.55},
        "position_r2": {"v_t": 0.1},
    }


class TestGuardrails:
    def test_inside_both_margins_holds(self) -> None:
        control = {s: _diag(0.70, 0.30) for s in SEEDS}
        arm = {s: _diag(0.70 - 0.009, 0.30 + 0.019) for s in SEEDS}
        assert e4.guardrails(control, arm)["hold"]

    def test_macro_drop_beyond_margin_fails(self) -> None:
        control = {s: _diag(0.70, 0.30) for s in SEEDS}
        arm = {s: _diag(0.70 - 0.011, 0.30) for s in SEEDS}
        g = e4.guardrails(control, arm)
        assert not g["checks"]["t2_val_macro"] and not g["hold"]

    def test_peak_rise_beyond_margin_fails(self) -> None:
        control = {s: _diag(0.70, 0.30) for s in SEEDS}
        arm = {s: _diag(0.70, 0.30 + 0.021) for s in SEEDS}
        assert not e4.guardrails(control, arm)["checks"]["normal_window_peak"]

    def test_one_o1_prime_failure_breaks_it(self) -> None:
        control = {s: _diag(0.70, 0.30) for s in SEEDS}
        arm = {s: _diag(0.70, 0.30, o1p=(s != SEEDS[0])) for s in SEEDS}
        assert not e4.guardrails(control, arm)["hold"]

    def test_a_diag_without_o1_prime_is_refused(self) -> None:
        control = {s: _diag(0.70, 0.30) for s in SEEDS}
        arm = {s: _diag(0.70, 0.30) for s in SEEDS}
        del arm[SEEDS[0]]["guardrails"]["o1_prime"]
        with pytest.raises(ValueError, match="O1'"):
            e4.guardrails(control, arm)

    def test_undefined_peak_is_refused(self) -> None:
        control = {s: _diag(0.70, None) for s in SEEDS}
        arm = {s: _diag(0.70, 0.30) for s in SEEDS}
        with pytest.raises(ValueError, match="all-normal"):
            e4.guardrails(control, arm)


class TestDecide:
    GOOD: ClassVar[dict[str, bool]] = {"hold": True}

    def test_all_three_adopt_a3_ign(self) -> None:
        d = e4.decide(_interval(0.02, 0.005, 0.035), _ci(0.01, -0.01, 0.03), self.GOOD)
        assert d["base"] == ARM

    def test_point_estimate_alone_does_not_adopt(self) -> None:
        d = e4.decide(_interval(0.02, -0.001, 0.041), _ci(0.01, 0.0, 0.02), self.GOOD)
        assert d["base"] == CONTROL and not d["checks"]["macro_t95_above_0"]

    def test_negative_f2_point_keeps_a3(self) -> None:
        d = e4.decide(_interval(0.02, 0.005, 0.035), _ci(-0.001, -0.02, 0.02), self.GOOD)
        assert d["base"] == CONTROL

    def test_broken_guardrail_keeps_a3(self) -> None:
        d = e4.decide(_interval(0.02, 0.005, 0.035), _ci(0.01, 0.0, 0.02), {"hold": False})
        assert d["base"] == CONTROL


# ---------------------------------------------------------------------------
# end to end on synthetic E3 / E4-0 dirs
# ---------------------------------------------------------------------------
def _clips(n: int) -> list[str]:
    return [f"vid{i % 7}_{i:06d}" for i in range(n)]


def _write_pb(root: Path, arm: str, scores: dict[str, np.ndarray],
              labels: dict[str, np.ndarray], per_seed: dict[str, float]) -> None:
    d = root / REPORTS / PB_DIR / arm / constants.V2_SPLIT_DOTA_CAP_DEV
    d.mkdir(parents=True)
    pb.write_clip_scores(d / pb.CLIP_SCORES_NPZ, scores, labels)
    (d / pb.READOUT_JSON).write_text(json.dumps({
        "runs": SEEDS, "global_steps": dict.fromkeys(SEEDS, 1740), "per_run_macro": per_seed,
    }))


def _write_diags(root: Path, arm: str, macro: float, peak: float) -> None:
    for s in SEEDS:
        d = root / arm / s / DIAG_DIR
        d.mkdir(parents=True)
        (d / DIAG_JSON).write_text(json.dumps(_diag(macro, peak)))


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    clips = _clips(40)
    monkeypatch.setattr(constants, "V2_E4_0_TAIL_CLIPS", len(clips))
    rng = np.random.default_rng(1)
    labels = {v: _labels() for v in clips}
    # A3 peaks at onset and fades; A3-ign holds the span -> a strictly better per-clip read
    control, arm = {}, {}
    for v in clips:
        base = rng.random(FRAMES) * 0.3
        c = base.copy()
        c[ONSET:ONSET + 3] = 1.0
        a = base.copy()
        a[ONSET:END] = 1.0
        control[v], arm[v] = c, a
    prior = {v: np.linspace(0.0, 1.0, FRAMES) for v in clips}
    e3, e40 = tmp_path / "e3", tmp_path / "e4_0"
    _write_pb(e3, CONTROL, control, labels, {s: 0.70 + 0.001 * i for i, s in enumerate(SEEDS)})
    _write_pb(e40, ARM, arm, labels, {s: 0.73 + 0.002 * i for i, s in enumerate(SEEDS)})
    p = e3 / REPORTS / PRIOR_DIR / constants.V2_SPLIT_DOTA_CAP_DEV
    p.mkdir(parents=True)
    pb.write_clip_scores(p / PRIOR_NPZ, prior, labels)
    _write_diags(e40, CONTROL, 0.70, 0.30)
    _write_diags(e40, ARM, 0.70, 0.30)
    return e3, e40


class TestRun:
    def test_better_arm_becomes_the_base(self, dirs: tuple[Path, Path]) -> None:
        e3, e40 = dirs
        e4.main(["--e3-dir", str(e3), "--e4-dir", str(e40)])
        r = json.loads((e40 / REPORTS / e4.READOUT_JSON).read_text())
        assert r["decision"]["base"] == ARM
        assert r["mechanism"]["conditions"]["argmax_in_span_rises"] is False  # both peak in span
        assert r["mechanism"]["summary"][ARM]["in_mean"] > r["mechanism"]["summary"][CONTROL][
            "in_mean"]
        assert "Base B = A3ign" in (e40 / REPORTS / e4.READOUT_MD).read_text()

    def test_a_different_checkpoint_is_refused(self, dirs: tuple[Path, Path]) -> None:
        e3, e40 = dirs
        path = e40 / ARM / SEEDS[0] / DIAG_DIR / DIAG_JSON
        path.write_text(json.dumps(_diag(0.70, 0.30, step=870)))
        with pytest.raises(ValueError, match="not the same checkpoint"):
            e4.main(["--e3-dir", str(e3), "--e4-dir", str(e40)])

    def test_an_e3_era_diag_is_refused(self, dirs: tuple[Path, Path]) -> None:
        e3, e40 = dirs
        path = e40 / CONTROL / SEEDS[0] / DIAG_DIR / DIAG_JSON
        d = _diag(0.70, 0.30)
        del d["t2_val_reads"]
        path.write_text(json.dumps(d))
        with pytest.raises(ValueError, match="t2_val_reads"):
            e4.main(["--e3-dir", str(e3), "--e4-dir", str(e40)])

    def test_the_tail_clip_gate_fires(self, dirs: tuple[Path, Path],
                                      monkeypatch: pytest.MonkeyPatch) -> None:
        e3, e40 = dirs
        monkeypatch.setattr(constants, "V2_E4_0_TAIL_CLIPS", 277)
        with pytest.raises(ValueError, match="tail clips"):
            e4.main(["--e3-dir", str(e3), "--e4-dir", str(e40)])
