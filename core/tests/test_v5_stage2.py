"""v5 Stage 2 read-out: S2-7 sentences, S2-9 guardrails, S2-10 decision (PREREG_V5 §2.3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import protocol_b_eval as pb
from core.tools import v5_stage2

SEEDS = [f"s{s}" for s in constants.V5_RUN_SEEDS]
FRAMES = 60
CLIPS = [f"vid{g:02d}_{k:06d}" for g in range(12) for k in range(3)]  # 12 source videos


def _labels() -> dict[str, np.ndarray]:
    out = {}
    for i, v in enumerate(CLIPS):
        y = np.zeros(FRAMES, dtype=np.int64)
        start = 15 + i % 10
        y[start : start + 20] = 1
        out[v] = y
    return out


def _scores(labels: dict[str, np.ndarray], signal: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {v: rng.random(FRAMES) + signal * y for v, y in labels.items()}


def _write_pb(root: Path, row: str, signal: float, seed_offsets: dict[str, float]) -> None:
    labels = _labels()
    per_seed = {s: _scores(labels, signal + seed_offsets.get(s, 0.0), i)
                for i, s in enumerate(SEEDS)}
    averaged = {v: np.mean([per_seed[s][v] for s in SEEDS], axis=0) for v in labels}
    d = root / row
    d.mkdir(parents=True)
    pb.write_clip_scores(d / pb.CLIP_SCORES_NPZ, averaged, labels)
    (d / pb.CLIP_AUCS_JSON).write_text(json.dumps(pb.clip_aucs(averaged, labels)))
    per_run = {s: float(np.mean(list(pb.clip_aucs(per_seed[s], labels).values()))) for s in SEEDS}
    (d / pb.READOUT_JSON).write_text(json.dumps({"per_run_macro": per_run}))


def _write_nexar(root: Path, row: str, endpoint: float) -> None:
    d = root / row
    d.mkdir(parents=True)
    whole = {"mean": 0.7, "low": 0.6, "high": 0.8, "clips": 10, "clusters": 10}
    (d / "nexar_readout.json").write_text(json.dumps(
        {"per_run_crop_endpoint": dict.fromkeys(SEEDS, endpoint), "whole": {"macro": whole}}))


def _ci(low: float) -> dict[str, float]:
    return {"mean": low + 0.01, "low": low, "high": low + 0.02, "clips": 10, "clusters": 5}


def _layout(
    tmp_path: Path,
    signals: dict[str, float],
    points: dict[str, bool],
    endpoints: dict[str, float] | None = None,
    prov_fail: tuple[str, ...] = (),
) -> list[str]:
    pb_dir, nexar_dir = tmp_path / "pb", tmp_path / "nexar"
    endpoints = endpoints or {}
    for row, signal in signals.items():
        _write_pb(pb_dir, row, signal, {})
        _write_nexar(nexar_dir, row, endpoints.get(row, 0.88))
    maps = {"rows": {r: {"hit_minus_chance": _ci(0.02 if p else -0.01),
                         "hit_minus_pre": _ci(0.02 if p else -0.01),
                         "hit_minus_centre": _ci(0.02 if p else -0.01),
                         "points": p, "stauc": {"stauc": 0.5, "auc": 0.6}}
                     for r, p in points.items()}}
    (tmp_path / "maps_readout.json").write_text(json.dumps(maps))
    runs = {f"{r}/{s}": {"pass": f"{r}/{s}" not in prov_fail} for r in signals for s in SEEDS}
    (tmp_path / "g4.json").write_text(json.dumps({"runs": runs}))
    return ["--pb-dir", str(pb_dir), "--nexar-dir", str(nexar_dir),
            "--maps-readout", str(tmp_path / "maps_readout.json"),
            "--provenance", str(tmp_path / "g4.json"), "--out-dir", str(tmp_path / "out")]


def _run(argv: list[str]) -> dict:
    return v5_stage2.run(v5_stage2.build_arg_parser().parse_args(argv))


class TestSentences:
    def test_improves_needs_all_three(self) -> None:
        ci = {"mean": 0.02, "low": 0.005, "high": 0.04}
        assert v5_stage2.detection_sentence(ci, {"a": 0.01, "b": 0.02, "c": 0.03}) == \
            v5_stage2.IMPROVES
        # (ii) one seed <= 0
        assert v5_stage2.detection_sentence(ci, {"a": -0.001, "b": 0.02, "c": 0.03}) == \
            v5_stage2.NO_DIFF
        # (iii) point below +0.01
        small = {"mean": 0.009, "low": 0.001, "high": 0.02}
        assert v5_stage2.detection_sentence(small, {"a": 0.01, "b": 0.01, "c": 0.01}) == \
            v5_stage2.NO_DIFF

    def test_costs(self) -> None:
        ci = {"mean": -0.03, "low": -0.05, "high": -0.01}
        assert v5_stage2.detection_sentence(ci, {"a": -0.03}) == v5_stage2.COSTS

    def test_paired_t_uses_n3_quantile(self) -> None:
        t = v5_stage2.paired_t({"a": 0.0, "b": 0.01, "c": 0.02})
        assert t is not None
        assert t["mean"] == pytest.approx(0.01)
        assert t["high"] - t["mean"] == pytest.approx(constants.V5_T95_N3 * 0.01 / np.sqrt(3))

    def test_paired_t_undefined_on_one_seed(self) -> None:
        assert v5_stage2.paired_t({"a": 0.01}) is None


class TestDecide:
    @staticmethod
    def _det(sentence: str) -> dict:
        return {"sentence": sentence}

    @staticmethod
    def _guard(hold: bool = True) -> dict:
        return {"hold": hold}

    def test_table(self) -> None:
        for improves, point, outcome in ((True, True, "GO-MQR"), (True, False, "DETECT-ONLY"),
                                         (False, True, "MAPS-ONLY"), (False, False, "NULL")):
            det = {"R5": self._det(v5_stage2.IMPROVES if improves else v5_stage2.NO_DIFF),
                   "R5d": self._det(v5_stage2.NO_DIFF)}
            d = v5_stage2.decide(det, {"R5": point, "R5d": False},
                                 {"R5": self._guard(), "R5d": self._guard()})
            assert (d["candidate"], d["outcome"]) == ("R5", outcome)

    def test_r5d_replaces_only_when_r5_has_no_sentence(self) -> None:
        guards = {"R5": self._guard(), "R5d": self._guard()}
        det = {"R5": self._det(v5_stage2.NO_DIFF), "R5d": self._det(v5_stage2.IMPROVES)}
        d = v5_stage2.decide(det, {"R5": False, "R5d": False}, guards)
        assert d["candidate"] == "R5d" and d["r5d_replaced_r5"] and d["outcome"] == "DETECT-ONLY"
        # R5 has a maps sentence -> R5 stays, even though R5Δ improves detection
        d = v5_stage2.decide(det, {"R5": True, "R5d": False}, guards)
        assert d["candidate"] == "R5" and d["outcome"] == "MAPS-ONLY"

    def test_guardrail_failure_is_not_adopted(self) -> None:
        det = {"R5": self._det(v5_stage2.IMPROVES), "R5d": self._det(v5_stage2.NO_DIFF)}
        d = v5_stage2.decide(det, {"R5": True, "R5d": False},
                             {"R5": self._guard(False), "R5d": self._guard()})
        assert d["outcome"] == v5_stage2.GUARD_FAIL[0] and d["adopted"] is None

    def test_fewer_seeds_is_preliminary_and_adopts_nothing(self) -> None:
        det = {"R5": self._det(v5_stage2.IMPROVES), "R5d": self._det(v5_stage2.NO_DIFF)}
        d = v5_stage2.decide(det, {"R5": True, "R5d": False},
                             {"R5": self._guard(), "R5d": self._guard()}, n_seeds=1)
        assert d["outcome"].startswith("PRELIMINARY GO-MQR") and d["preliminary"]
        assert d["adopted"] is None


class TestRun:
    def test_end_to_end_go(self, tmp_path: Path) -> None:
        argv = _layout(tmp_path, {"R1": 0.3, "R3": 0.3, "R5": 1.5, "R5d": 0.3},
                       {"R3": False, "R5": True, "R5d": False})
        r = _run(argv)
        assert r["detection"]["R5"]["sentence"] == v5_stage2.IMPROVES
        assert r["decision"]["outcome"] == "GO-MQR" and r["decision"]["adopted"] == "R5"
        assert r["detection"]["R5"]["clip_delta"]["clusters"] == 12
        md = (tmp_path / "out" / v5_stage2.READOUT_MD).read_text()
        assert "GO-MQR" in md and "R3 (printed)" in md

    def test_nexar_guardrail_blocks(self, tmp_path: Path) -> None:
        argv = _layout(tmp_path, {"R1": 0.3, "R3": 0.3, "R5": 1.5, "R5d": 0.3},
                       {"R3": False, "R5": True, "R5d": False},
                       endpoints={"R1": 0.88, "R5": 0.86})
        r = _run(argv)
        assert not r["guardrails"]["R5"]["checks"]["nexar_val_crop_endpoint"]
        assert r["decision"]["outcome"] == v5_stage2.GUARD_FAIL[0]

    def test_provenance_failure_blocks(self, tmp_path: Path) -> None:
        argv = _layout(tmp_path, {"R1": 0.3, "R3": 0.3, "R5": 1.5, "R5d": 0.3},
                       {"R3": False, "R5": True, "R5d": False}, prov_fail=("R5/s2025",))
        assert _run(argv)["decision"]["outcome"] == v5_stage2.GUARD_FAIL[0]

    def test_null(self, tmp_path: Path) -> None:
        argv = _layout(tmp_path, {"R1": 0.3, "R3": 0.3, "R5": 0.3, "R5d": 0.3},
                       {"R3": False, "R5": False, "R5d": False})
        assert _run(argv)["decision"]["outcome"] == "NULL"

    def test_seed_mismatch_raises(self, tmp_path: Path) -> None:
        argv = _layout(tmp_path, {"R1": 0.3, "R3": 0.3, "R5": 0.3, "R5d": 0.3},
                       {"R3": False, "R5": False, "R5d": False})
        with pytest.raises(ValueError, match="seeds"):
            _run([*argv, "--seeds", "2024", "2025"])

    def test_clip_mismatch_raises(self, tmp_path: Path) -> None:
        argv = _layout(tmp_path, {"R1": 0.3, "R3": 0.3, "R5": 0.3, "R5d": 0.3},
                       {"R3": False, "R5": False, "R5d": False})
        d = tmp_path / "pb" / "R5"
        scores, labels = pb.read_clip_scores(d / pb.CLIP_SCORES_NPZ)
        drop = CLIPS[0]
        pb.write_clip_scores(d / pb.CLIP_SCORES_NPZ,
                             {k: v for k, v in scores.items() if k != drop},
                             {k: v for k, v in labels.items() if k != drop})
        with pytest.raises(ValueError, match="different clips"):
            _run(argv)

    def test_missing_primary_maps_raises(self, tmp_path: Path) -> None:
        argv = _layout(tmp_path, {"R1": 0.3, "R3": 0.3, "R5": 0.3, "R5d": 0.3},
                       {"R3": False, "R5": False})
        with pytest.raises(ValueError, match="maps"):
            _run(argv)

    def test_f2_and_borrowed_are_printed(self, tmp_path: Path) -> None:
        argv = _layout(tmp_path, {"R1": 0.3, "R3": 0.3, "R5": 0.3, "R5d": 0.3},
                       {"R3": False, "R5": False, "R5d": False})
        labels = _labels()
        prior = {v: -np.abs(np.arange(FRAMES) / FRAMES - 0.5) for v in labels}
        pb.write_clip_scores(tmp_path / "prior.npz", prior, labels)
        (tmp_path / "a0.json").write_text(json.dumps({
            "macro": {"mean": 0.71}, "per_run_macro": {"s2024": 0.69, "s2025": 0.71,
                                                       "s2026": 0.70, "s2027": 0.70}}))
        r = _run([*argv, "--prior", str(tmp_path / "prior.npz"), "--r0-pb",
                  str(tmp_path / "a0.json")])
        p = r["printed"]
        assert set(p["f2"]["delta_vs_r1"]) == {"R3", "R5", "R5d"}
        assert p["r0_v2_nexar_a0"]["per_seed_mean"] == pytest.approx(0.70)
        assert r["decision"]["outcome"] == "NULL"  # printed reads never move the decision


class TestAP:
    """§2.4 B12: AP is printed beside AUC and never moves the decision."""

    def test_perfect_scores_and_minmax_pooling(self) -> None:
        labels = _labels()
        perfect = {v: y.astype(float) for v, y in labels.items()}
        assert set(v5_stage2.clip_aps(perfect, labels).values()) == {1.0}
        # per-clip offsets and scales vanish under min-max pooling, not under raw pooling
        shifted = {v: (i + 1.0) * s + 10.0 * i for i, (v, s) in
                   enumerate(_scores(labels, 0.5, 0).items())}
        base = _scores(labels, 0.5, 0)
        assert v5_stage2.pooled_minmax_ap(shifted, labels) == pytest.approx(
            v5_stage2.pooled_minmax_ap(base, labels))

    def test_run_prints_ap(self, tmp_path: Path) -> None:
        argv = _layout(tmp_path, {"R1": 0.3, "R3": 0.3, "R5": 0.9, "R5d": 0.3},
                       {"R3": False, "R5": False, "R5d": False})
        ap = _run(argv)["printed"]["ap"]
        assert set(ap["clip_delta_vs_r1"]) == {"R3", "R5", "R5d"}
        assert ap["clip_delta_vs_r1"]["R5"]["low"] > 0
        assert ap["pooled_minmax"]["R5"] > ap["pooled_minmax"]["R1"]
        assert 0 < ap["positive_frame_share"] < 1
        md = (tmp_path / "out" / v5_stage2.READOUT_MD).read_text()
        assert "AP (§2.4 B12)" in md


class TestPreliminary:
    def test_one_seed_without_r3(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """§2.4 B11: one seed, R5 / R5Δ only -> outcome PRELIMINARY, nothing adopted."""
        monkeypatch.setitem(globals(), "SEEDS", ["s2024"])
        argv = _layout(tmp_path, {"R1": 0.3, "R5": 0.3, "R5d": 0.3}, {"R5": False, "R5d": False})
        r = _run([*argv, "--seeds", "2024", "--rows", "R5", "R5d"])
        assert r["decision"]["outcome"] == "PRELIMINARY NULL (n = 1 seed)"
        assert r["decision"]["adopted"] is None
        assert r["detection"]["R5"]["paired_t_printed"] is None
        assert r["detection"]["R5"]["sentence"] in (v5_stage2.IMPROVES, v5_stage2.COSTS,
                                                    "no difference shown at n = 1")
        assert "PRELIMINARY" in (tmp_path / "out" / v5_stage2.READOUT_MD).read_text()
