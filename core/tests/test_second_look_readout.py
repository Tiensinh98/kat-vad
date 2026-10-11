"""Addendum §23 (Amendment 12): the second look of A3-ign -- S7 gates, S3-S6 reads, S5 sentences.

CPU-only, data-free: a synthetic E3 / E4-0 / Final / second-look layout in the shapes the real
tools write.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.tools import second_look_readout as sl
from core.tools.e3_readout import PB_DIR, PRIOR_DIR, REPORTS, seed_name
from core.tools.final_readout import T2_TEST_DIR
from core.tools.position_prior import PRIOR_NPZ
from core.tools.position_prior import READOUT_JSON as PRIOR_READOUT_JSON
from core.tools.protocol_b_eval import CLIP_AUCS_JSON, READOUT_JSON, clip_aucs, write_clip_scores
from core.tools.v2_guard_test import GUARD_JSON

SEEDS = [seed_name(s) for s in constants.V2_E3_SEEDS]
STEP = 1740
CAP_EVAL, CAP_DEV = constants.V2_SPLIT_DOTA_CAP_EVAL, constants.V2_SPLIT_DOTA_CAP_DEV


def _labels(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {}
    for k in range(n):
        lab = np.zeros(20, dtype=np.int64)
        start = int(rng.integers(3, 12))
        lab[start : start + 5] = 1
        out[f"vid{k // 7:03d}_{k:06d}"] = lab
    return out


def _pb(root: Path, arm: str, split: str, labels: dict[str, np.ndarray], gain: float,
        rng: np.random.Generator, steps: dict[str, int] | None = None) -> None:
    d = root / REPORTS / PB_DIR / arm / split
    d.mkdir(parents=True)
    per_run, runs = {}, []
    for s in SEEDS:
        scores = {v: gain * lab + rng.normal(0, 1, len(lab)) for v, lab in labels.items()}
        per_run[s] = float(np.mean(list(clip_aucs(scores, labels).values())))
        runs.append(scores)
    averaged = {v: np.mean([r[v] for r in runs], axis=0) for v in labels}
    write_clip_scores(d / "clip_scores.npz", averaged, labels)
    (d / CLIP_AUCS_JSON).write_text(json.dumps(clip_aucs(averaged, labels)))
    (d / READOUT_JSON).write_text(json.dumps({
        "global_steps": steps or dict.fromkeys(SEEDS, STEP), "per_run_macro": per_run}))


def _guard(
    path: Path, arms: dict[str, float], o1p: bool = True, windows: int | None = None
) -> None:
    n = windows or constants.V2_SL_T2_TEST_WINDOWS
    row = {"micro": 0.7, "clip_oracle_micro": 0.70, "window_level_auc": 0.75}
    path.mkdir(parents=True)
    path.joinpath(GUARD_JSON).write_text(json.dumps({
        "runs": {a: {"global_step": STEP, "windows": n} for a in arms},
        "t2_test": {a: {**row, "macro": m} for a, m in arms.items()},
        "arms": {a: {"o1_prime_t2_test": {"pass": o1p}} for a in arms if a != "A0"},
    }))


def _layout(tmp: Path, ign_gain: float, **kw: Any) -> argparse.Namespace:
    rng = np.random.default_rng(0)
    ev, dev = _labels(constants.V2_SL_CAP_EVAL_CLIPS, rng), _labels(70, rng)
    e3, e4, final, slp = (tmp / x for x in ("e3", "e4", "final", "sl"))
    _pb(e3, "A3", CAP_DEV, dev, 1.0, rng)
    _pb(e4, sl.ARM, CAP_DEV, dev, 1.4, rng)
    _pb(final, "A3", CAP_EVAL, ev, 1.0, rng)
    _pb(final, "A0", CAP_EVAL, ev, 0.5, rng)
    _pb(slp, sl.ARM, CAP_EVAL, ev, ign_gain, rng, kw.get("ign_steps"))
    prior_dir = final / REPORTS / PRIOR_DIR / CAP_EVAL
    prior_dir.mkdir(parents=True)
    write_clip_scores(prior_dir / PRIOR_NPZ, {v: np.linspace(0, 1, 20) for v in ev}, ev)
    (prior_dir / PRIOR_READOUT_JSON).write_text("{}")
    for s in SEEDS:
        _guard(final / T2_TEST_DIR / s, {"A0": 0.63, "A3": 0.695})
        _guard(slp / T2_TEST_DIR / s, {"A0": kw.get("a0_here", 0.63), sl.ARM: 0.71},
               windows=kw.get("windows"))
    return sl.build_arg_parser().parse_args([
        "--e3-dir", str(e3), "--e4-dir", str(e4), "--final-dir", str(final), "--sl-dir", str(slp)])



class TestSentences:
    def test_each_outcome_licenses_its_own_sentence(self) -> None:
        up, flat, down = ({"low": 0.01, "high": 0.05}, {"low": -0.01, "high": 0.02},
                          {"low": -0.05, "high": -0.01})
        assert "improves" in sl.sentences(up, None)[0]
        assert "not detectable" in sl.sentences(flat, None)[0]
        assert "did not hold" in sl.sentences(down, None)[0]

    def test_beyond_position_needs_the_whole_interval_above_zero(self) -> None:
        up = {"low": 0.01, "high": 0.05}
        assert sl.sentences(up, {"mean": 0.01, "low": 0.002, "high": 0.02})[1].startswith("...")
        assert "No beyond" in sl.sentences(up, {"mean": 0.01, "low": -0.001, "high": 0.02})[1]


class TestReadout:
    def test_a_better_arm_reads_as_an_improvement(self, tmp_path: Path) -> None:
        r = sl.run(_layout(tmp_path, ign_gain=1.6))
        assert r["dota_cap_eval"]["delta_vs_a3"]["low"] > 0
        assert "improves" in r["sentences"][0] and r["label"] == sl.LABEL
        assert r["t2_test"]["o1_prime_every_seed"]
        assert r["t2_test"]["delta_macro_vs_a3"]["mean"] == pytest.approx(0.015)
        md = (tmp_path / "sl" / REPORTS / sl.READOUT_MD).read_text()
        assert "second look at an opened set" in md and "Nothing re-decides" in md

    def test_a_worse_arm_still_does_not_re_decide(self, tmp_path: Path) -> None:
        r = sl.run(_layout(tmp_path, ign_gain=0.6))
        assert r["dota_cap_eval"]["delta_vs_a3"]["high"] < 0
        assert "stays the recipe" in r["sentences"][0]


class TestIntegrityGates:
    def test_other_checkpoints_are_refused(self, tmp_path: Path) -> None:
        steps = dict.fromkeys(SEEDS, STEP) | {SEEDS[0]: 510}
        with pytest.raises(ValueError, match=r"S7 FAILED.*steps"):
            sl.run(_layout(tmp_path, 1.4, ign_steps=steps))

    def test_a_different_a0_read_on_t2_test_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not the same read"):
            sl.run(_layout(tmp_path, 1.4, a0_here=0.64))

    def test_a_short_t2_test_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="windows"):
            sl.run(_layout(tmp_path, 1.4, windows=1000))

    def test_clip_sets_must_match(self) -> None:
        a = {"x": np.array([0, 1])}
        with pytest.raises(ValueError, match="clip sets"):
            sl.check_clips(a, {"y": np.array([0, 1])}, "t")
        with pytest.raises(ValueError, match="expected"):
            sl.check_clips(a, {"x": np.array([0, 1])}, "t")
