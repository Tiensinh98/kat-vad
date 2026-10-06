"""Final read-out (Amendment 10): sealed sets are read only on purpose, and only E3's checkpoints.

The failures that matter are silent: a sealed split scored without ``--final``, a fusion that
is not within-clip (so clip level leaks into a macro), a P7 gate that passes on numbers that
moved, a Final that re-decides the adoption, or an eval read of other checkpoints than E3's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from core import constants
from core.tools import final_readout as fr
from core.tools import position_prior as pp
from core.tools import protocol_b_eval as pb
from core.tools.e3_readout import E3_READOUT_JSON, PB_DIR, PRIOR_DIR, REPORTS
from core.tools.v2_guard_test import GUARD_JSON

SEEDS = ["s2024", "s2025", "s2026", "s2027", "s2028"]
STEP = 1740
CLIPS = 24
FRAMES = 30


class TestSplitGate:
    def test_open_splits_pass_without_final(self) -> None:
        for split in pb.OPEN_SPLITS:
            pb.check_split(split, False, "t")

    def test_a_sealed_split_needs_final(self) -> None:
        for split in pb.FINAL_SPLITS:
            with pytest.raises(ValueError, match="--final"):
                pb.check_split(split, False, "t")
            pb.check_split(split, True, "t")

    def test_an_unknown_split_is_refused_even_with_final(self) -> None:
        with pytest.raises(ValueError, match="reads"):
            pb.check_split("t2_val", True, "t")

    def test_every_cap_split_is_named_dota_cap(self) -> None:
        assert pb.set_name(constants.V2_SPLIT_DOTA_CAP_EVAL).startswith("DoTA-CAP")
        assert pb.set_name(constants.V2_SPLIT_DOTA_EVAL) == "DoTA"

    def test_both_clis_accept_final_only_as_a_flag(self) -> None:
        for parser in (pb.build_arg_parser(), pp.build_arg_parser()):
            dests = {a.dest: a for a in parser._actions}
            assert dests["final"].default is False
            assert constants.V2_SPLIT_DOTA_CAP_EVAL in list(dests["split"].choices or [])

    def test_position_prior_refuses_a_sealed_split_before_reading_anything(
        self, tmp_path: Path
    ) -> None:
        args = pp.build_arg_parser().parse_args([
            "--data-dir", str(tmp_path / "missing"), "--s1-dir", str(tmp_path),
            "--metadata", str(tmp_path / "m.json"), "--split-file", str(tmp_path / "s.txt"),
            "--split", constants.V2_SPLIT_DOTA_CAP_EVAL, "--out-dir", str(tmp_path / "o"),
        ])
        with pytest.raises(ValueError, match="sealed"):
            pp.run(args)


class TestFusion:
    def test_zscore_of_a_constant_is_zero(self) -> None:
        np.testing.assert_array_equal(fr.zscore(np.ones(4)), np.zeros(4))

    def test_raw_weight_is_the_plain_clip_auc(self) -> None:
        s = {"v_000001": np.array([0.1, 0.9, 0.2, 0.8])}
        lab = {"v_000001": np.array([0, 1, 0, 1])}
        assert fr.fused_clip_aucs(s, s, lab, None) == {"v_000001": 1.0}

    def test_fusion_is_within_clip_so_a_rescaled_score_reads_the_same(self) -> None:
        rng = np.random.default_rng(0)
        prior = {"a_000001": rng.random(20)}
        lab = {"a_000001": (np.arange(20) >= 10).astype(int)}
        s = {"a_000001": rng.random(20)}
        scaled = {"a_000001": 100.0 * s["a_000001"] + 7.0}
        assert fr.fused_clip_aucs(s, prior, lab, 2.0) == fr.fused_clip_aucs(scaled, prior, lab, 2.0)

    def test_a_large_weight_reads_the_prior(self) -> None:
        lab = {"a_000001": np.array([0, 0, 1, 1, 0])}
        prior = {"a_000001": np.array([0.0, 0.1, 0.9, 0.8, 0.2])}
        anti = {"a_000001": -prior["a_000001"]}
        assert fr.fused_clip_aucs(anti, prior, lab, None)["a_000001"] == 0.0
        assert fr.fused_clip_aucs(anti, prior, lab, 10.0)["a_000001"] == 1.0

    def test_single_class_clips_are_skipped_and_lengths_must_match(self) -> None:
        lab = {"a_000001": np.zeros(3, dtype=int)}
        ones = {"a_000001": np.ones(3)}
        assert fr.fused_clip_aucs(ones, ones, lab, 1.0) == {}
        with pytest.raises(ValueError, match="prior frames"):
            fr.fused_clip_aucs({"a_000001": np.ones(3)}, {"a_000001": np.ones(4)},
                               {"a_000001": np.array([0, 1, 0])}, 1.0)


def _interval(low: float) -> dict[str, float]:
    return {"mean": low + 0.01, "low": low, "high": low + 0.02}


class TestSentences:
    def _control(self, a3a0: float, a3a1: float, over: float) -> dict[str, Any]:
        w = "w=2"
        return {"delta": {"A3 - A0": {w: _interval(a3a0)}, "A3 - A1": {w: _interval(a3a1)}},
                "over_prior": {"A3": _interval(over)}}

    def test_each_sentence_follows_its_lower_bound(self) -> None:
        yes = fr.sentences(self._control(0.01, 0.01, 0.01))
        assert yes[0].startswith("A3's gain over A0 survives")
        assert "survives" in yes[1] and yes[2] == "A3 carries signal beyond position."
        no = fr.sentences(self._control(-0.01, -0.01, -0.01))
        assert "not separable" in no[0] and "not separable" in no[1]
        assert no[2].startswith("A3 adds nothing")

    def test_no_sentence_says_motion(self) -> None:
        for s in fr.sentences(self._control(0.01, -0.01, 0.01)):
            assert "motion" not in s.lower()


# ---------------------------------------------------------------------------
# synthetic E3 + Final dirs
# ---------------------------------------------------------------------------
def _clip_ids(prefix: str) -> list[str]:
    return [f"{prefix}{k // 3:03d}_{k:06d}" for k in range(CLIPS)]


def _write_pb(root: Path, arm: str, split: str, gain: float, rng: np.random.Generator,
              ids: list[str], step: int = STEP) -> None:
    d = root / REPORTS / PB_DIR / arm / split
    d.mkdir(parents=True, exist_ok=True)
    labels = {v: (np.arange(FRAMES) >= FRAMES // 2).astype(np.int64) for v in ids}
    scores = {v: gain * labels[v] + rng.random(FRAMES) for v in ids}
    pb.write_clip_scores(d / pb.CLIP_SCORES_NPZ, scores, labels)
    aucs = pb.clip_aucs(scores, labels)
    (d / pb.CLIP_AUCS_JSON).write_text(json.dumps(aucs))
    macro = float(np.mean(list(aucs.values())))
    per_run = {s: macro + 0.001 * k for k, s in enumerate(SEEDS)}
    (d / pb.READOUT_JSON).write_text(json.dumps({
        "runs": SEEDS, "global_steps": dict.fromkeys(SEEDS, step), "clips": len(ids),
        "macro": {"mean": macro, "low": macro - 0.01, "high": macro + 0.01},
        "per_run_macro": per_run,
    }))


def _write_prior(root: Path, split: str, ids: list[str], rng: np.random.Generator) -> None:
    d = root / REPORTS / PRIOR_DIR / split
    d.mkdir(parents=True, exist_ok=True)
    labels = {v: (np.arange(FRAMES) >= FRAMES // 2).astype(np.int64) for v in ids}
    pb.write_clip_scores(d / pp.PRIOR_NPZ, {v: rng.random(FRAMES) for v in ids}, labels)
    ci = {"mean": 0.6, "low": 0.55, "high": 0.65}
    (d / pp.READOUT_JSON).write_text(json.dumps({"p_t2_macro": ci, "monotone_ruler_macro": ci}))


def _guard_row(macro: float, window: float) -> dict[str, Any]:
    return {"micro": 0.66, "macro": macro, "clip_oracle_micro": 0.70,
            "window_level_auc": window, "pass": True}


def _write_t2(root: Path, steps: dict[str, int]) -> None:
    for s in SEEDS:
        d = root / fr.T2_TEST_DIR / s
        d.mkdir(parents=True, exist_ok=True)
        rows = {"A0": _guard_row(0.63, 0.66), "A1": _guard_row(0.64, 0.70),
                "A2": _guard_row(0.70, 0.76), "A3": _guard_row(0.69, 0.78)}
        runs = {a: {"global_step": steps[a]} for a in rows}
        (d / GUARD_JSON).write_text(json.dumps({"t2_test": rows, "runs": runs}))


GAINS = {"A0": 0.3, "A1": 0.25, "A2": 0.6, "A3": 0.65}


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    e3_dir, final_dir = tmp_path / "e3", tmp_path / "final"
    splits = {fr.DEV: "dv", fr.CAP_DEV: "cd", fr.EVAL: "ev", fr.CAP_EVAL: "ce"}
    for arm, gain in GAINS.items():
        for split in fr.EVAL_SPLITS_OF[arm]:
            _write_pb(final_dir, arm, split, gain, rng, _clip_ids(splits[split]))
            dev = fr.DEV_OF[split]
            _write_pb(e3_dir, arm, dev, gain, rng, _clip_ids(splits[dev]))
    for split in (fr.DEV, fr.CAP_DEV):
        _write_prior(e3_dir, split, _clip_ids(splits[split]), rng)
    for split in (fr.EVAL, fr.CAP_EVAL):
        _write_prior(final_dir, split, _clip_ids(splits[split]), rng)
    (e3_dir / REPORTS / E3_READOUT_JSON).write_text(
        json.dumps({"decision": {"adopt": "A3", "F": "A0"}}))
    _write_t2(final_dir, dict.fromkeys(GAINS, STEP))
    return e3_dir, final_dir


def _args(e3_dir: Path, final_dir: Path) -> argparse.Namespace:
    return fr.build_arg_parser().parse_args(
        ["--e3-dir", str(e3_dir), "--final-dir", str(final_dir)])


def _passing_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    real = fr.dev_gate

    def gate(e3_dir: Path, seed: int) -> dict[str, Any]:
        return {**real(e3_dir, seed), "pass": True, "failed": []}

    monkeypatch.setattr(fr, "dev_gate", gate)


class TestDevGate:
    def test_moved_numbers_fail_the_gate(self, dirs: tuple[Path, Path]) -> None:
        gate = fr.dev_gate(dirs[0], constants.SEED)
        assert not gate["pass"] and gate["failed"]
        assert len(gate["checks"]) == 1 + 4 * 3 + 3 * 3

    def test_the_recorded_numbers_pass_when_they_are_what_the_tool_reads(
        self, dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        control = fr.dev_gate(dirs[0], constants.SEED)["control"]
        keys = ("raw", "w=1", "w=2")
        monkeypatch.setattr(constants, "V2_FINAL_DEV_P_T2", control["p_t2_macro"])
        monkeypatch.setattr(constants, "V2_FINAL_DEV_ARM_MACRO", {
            a: tuple(control["macro"][a][k] for k in keys) for a in GAINS})
        monkeypatch.setattr(constants, "V2_FINAL_DEV_DELTA", {
            (x, y): tuple(control["delta"][f"{x} - {y}"][k]["mean"] for k in keys)
            for x, y in fr.FUSION_PAIRS})
        assert fr.dev_gate(dirs[0], constants.SEED)["pass"]

    def test_run_refuses_to_read_sealed_outputs_when_the_gate_fails(
        self, dirs: tuple[Path, Path]
    ) -> None:
        with pytest.raises(ValueError, match="P7 gate FAILS"):
            fr.run(_args(*dirs))


class TestRun:
    def test_end_to_end(self, dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        _passing_gate(monkeypatch)
        r = fr.run(_args(*dirs))
        assert set(r["arms"]) == set(GAINS)
        assert set(r["arms"]["A2"]) == {fr.CAP_EVAL}
        assert set(r["arms"]["A0"]) == {fr.EVAL, fr.CAP_EVAL}
        assert r["contrasts"]["A3 - A0"]["mean"] > 0.0
        assert r["contrasts"]["A1 - A0"]["split"] == fr.EVAL
        assert len(r["sentences"]) == 3
        assert r["t2_test"]["o1_prime_pass"] == {"A1": 5, "A2": 5, "A3": 5}
        assert set(r["position_control"]) == {fr.CAP_EVAL, fr.CAP_DEV}
        md = (dirs[1] / REPORTS / fr.FINAL_READOUT_MD).read_text()
        assert "DoTA-CAP-eval (n/1397)" in md and "never compare them" in md

    def test_a_different_decision_is_refused(
        self, dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _passing_gate(monkeypatch)
        (dirs[0] / REPORTS / E3_READOUT_JSON).write_text(
            json.dumps({"decision": {"adopt": "A2", "F": "A0"}}))
        with pytest.raises(ValueError, match="P1"):
            fr.run(_args(*dirs))

    def test_eval_scores_of_other_checkpoints_are_refused(
        self, dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _passing_gate(monkeypatch)
        _write_pb(dirs[1], "A3", fr.CAP_EVAL, 0.6, np.random.default_rng(1),
                  _clip_ids("ce"), step=STEP - 1)
        with pytest.raises(ValueError, match="not E3's checkpoints"):
            fr.run(_args(*dirs))

    def test_t2_test_of_other_checkpoints_is_refused(
        self, dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _passing_gate(monkeypatch)
        _write_t2(dirs[1], {"A0": STEP, "A1": STEP, "A2": 510, "A3": STEP})
        with pytest.raises(ValueError, match="T2-test A2"):
            fr.run(_args(*dirs))

    def test_t2_test_collapse_is_read_against_a0_of_the_same_seed(
        self, dirs: tuple[Path, Path]
    ) -> None:
        d = dirs[1] / fr.T2_TEST_DIR / "s2026" / GUARD_JSON
        r = json.loads(d.read_text())
        r["t2_test"]["A2"] = _guard_row(0.60, 0.80)  # window up, macro < A0 - 0.01
        d.write_text(json.dumps(r))
        out = fr.t2_test(dirs[1], SEEDS, {a: dict.fromkeys(SEEDS, STEP) for a in GAINS})
        assert out["o1_prime_pass"]["A2"] == 4
        assert out["per_seed"]["s2026"]["A2"]["o1_prime"] is False


class TestFeaturelessExclusion:
    IDS: ClassVar[list[str]] = ["a_000001", "b_000002", "c_000003"]

    def _s1(self, tmp_path: Path, present: list[str]) -> Path:
        for v in present:
            np.save(tmp_path / f"{v}.npy", np.zeros((2, 4), dtype=np.float32))
        return tmp_path

    def test_the_fixed_list_is_dropped_under_final(self, tmp_path: Path) -> None:
        s1 = self._s1(tmp_path, ["a_000001", "c_000003"])
        assert pb.featureless_ids(self.IDS, ("b_000002",), s1, True) == ["a_000001", "c_000003"]

    def test_a_clip_with_features_cannot_be_excluded(self, tmp_path: Path) -> None:
        s1 = self._s1(tmp_path, self.IDS)
        with pytest.raises(ValueError, match="have CLIP features"):
            pb.featureless_ids(self.IDS, ("b_000002",), s1, True)

    def test_another_featureless_clip_is_refused(self, tmp_path: Path) -> None:
        s1 = self._s1(tmp_path, ["a_000001"])
        with pytest.raises(ValueError, match="outside the fixed list"):
            pb.featureless_ids(self.IDS, ("b_000002",), s1, True)

    def test_exclusion_needs_final_and_ids_in_the_split(self, tmp_path: Path) -> None:
        s1 = self._s1(tmp_path, ["a_000001", "c_000003"])
        with pytest.raises(ValueError, match="Final-step"):
            pb.featureless_ids(self.IDS, ("b_000002",), s1, False)
        with pytest.raises(ValueError, match="not in the split"):
            pb.featureless_ids(self.IDS, ("z_000009",), s1, True)

    def test_the_list_applies_to_dota_eval_only(self) -> None:
        assert pb.excluded_for(constants.V2_SPLIT_DOTA_EVAL, True) == \
            constants.V2_FINAL_DOTA_EVAL_NO_FEATURES
        assert pb.excluded_for(constants.V2_SPLIT_DOTA_CAP_EVAL, True) == ()
        assert pb.excluded_for(constants.V2_SPLIT_DOTA_EVAL, False) == ()

    def test_the_list_is_the_five_clips_missing_from_the_frozen_split(self) -> None:
        from core.data.v2_splits import load_split

        ev = load_split(constants.V2_SPLIT_DOTA_EVAL, final=True)
        cap_ev = load_split(constants.V2_SPLIT_DOTA_CAP_EVAL, final=True)
        listed = set(constants.V2_FINAL_DOTA_EVAL_NO_FEATURES)
        assert len(listed) == 5 and listed <= set(ev) and not listed & set(cap_ev)
