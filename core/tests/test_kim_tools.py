"""KIM pilot (KP1-KP4) and first-pass read-out (``core/docs/final/KIM_RUN.md`` §3.1, §5, §6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants, evaluate
from core.config import Config
from core.models.kat_vad import KATVAD
from core.tools import dada_whole_eval, kim_pilot, kim_readout
from core.tools.protocol_b_eval import CLIP_AUCS_JSON, READOUT_JSON, write_clip_scores
from core.tools.stp_readout import COST, GAIN, NONE

MOTION = 384
BASE = constants.HIDDEN_DIM + MOTION
CPU = torch.device("cpu")


def _w0() -> KATVAD:
    cfg = Config()
    cfg.kip.enabled = False
    cfg.model.motion_dim = MOTION
    cfg.v2.motion = constants.V5_ENCODER
    cfg.model.score_head_kernel = 3
    torch.manual_seed(0)
    return KATVAD.from_config(cfg).eval()


class TestPilotPieces:
    def test_change_scores_by_hand(self) -> None:
        z = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]])
        s = kim_pilot.change_scores(z)
        # d1 per step: 0, 1, 2, 0 -> rho = median(1, 2, 0) = 1
        np.testing.assert_allclose(s[1], [0, 1, 2, 0], rtol=1e-5)
        # d2: 0, 0, |2 - 1|, |0 - 2| = 0, 0, 1, 2
        np.testing.assert_allclose(s[2], [0, 0, 1, 2], rtol=1e-5)

    def test_rank01(self) -> None:
        np.testing.assert_allclose(kim_pilot.rank01(np.array([3.0, 1.0, 2.0])), [1.0, 0.0, 0.5])
        np.testing.assert_allclose(kim_pilot.rank01(np.ones(4)), [0.5] * 4)

    def test_edge_share(self) -> None:
        curve = np.array([0.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        share, chance = kim_pilot.edge_share(curve)  # type: ignore[misc]
        assert share == 1.0 and chance == pytest.approx(0.2)  # top-1 of 10 diffs is the first
        assert kim_pilot.edge_share(np.zeros(3)) is None

    def test_onset_lag(self) -> None:
        labels = np.array([0, 0, 1, 1, 0])
        assert kim_pilot.onset_lag(np.array([0, 0, 0, 5.0, 0]), labels, 0.1) == pytest.approx(0.1)
        assert kim_pilot.onset_lag(np.zeros(5), np.zeros(5), 0.1) is None
        summary = kim_pilot.lag_summary([0.1, -0.2, 2.0])
        assert summary is not None and summary["share_within"] == pytest.approx(2 / 3)

    def test_feature_sequences_are_the_host_path(self) -> None:
        model = _w0()
        rows = np.random.default_rng(0).standard_normal((7, BASE)).astype(np.float32)
        seqs = kim_pilot.feature_sequences(model, rows, CPU)
        out = model(torch.from_numpy(rows)[None], torch.tensor([7]))
        np.testing.assert_allclose(seqs["vt"], out["vis_feats"][0].detach().numpy(), atol=1e-5)
        np.testing.assert_allclose(seqs["x"], rows[:, : constants.HIDDEN_DIM], atol=1e-6)  # W_u = 0
        assert seqs["motion"].shape == (7, MOTION)

    def test_check_model_refuses_kim(self) -> None:
        cfg = Config()
        cfg.kip.enabled, cfg.kim.enabled = False, True
        with pytest.raises(ValueError, match="KIM and KIP off"):
            kim_pilot.check_model(KATVAD.from_config(cfg))


def _labels(n_clips: int, frames: int) -> dict[str, np.ndarray]:
    out = {}
    for i in range(n_clips):
        gt = np.zeros(frames, dtype=np.int64)
        gt[frames // 2 : frames // 2 + 4] = 1
        out[f"src{i % 3}_{i:06d}"] = gt
    return out


class TestPilotSets:
    @pytest.mark.parametrize(("kind", "stride"), [("native", 3), ("rows", 1)])
    def test_score_and_summarize(self, tmp_path: Path, kind: str, stride: int) -> None:
        model = _w0()
        frames = 30
        labels = _labels(6, frames)
        rng = np.random.default_rng(1)
        n_rows = len(range(0, frames, stride)) if kind == "native" else frames
        for v in labels:
            np.save(tmp_path / f"{v}.npy", rng.standard_normal((n_rows, BASE)).astype(np.float32))
        curves, steps = kim_pilot.score_set(kind, tmp_path, labels, model, CPU, stride)
        assert set(curves) == {f"{f}{o}" for f in kim_pilot.FEATURES for o in kim_pilot.ORDERS}
        assert all(len(c) == frames for c in curves["vt2"].values())
        w0 = {v: rng.random(frames) for v in labels}
        name = kim_pilot.DOTA_SET if kind == "native" else kim_pilot.DADA_SET
        ego = {v: i % 2 == 0 for i, v in enumerate(labels)} if kind == "native" else None
        s = kim_pilot.summarize_set(name, curves, steps, w0, labels, ego, 0.1)
        assert s["two_class"] == 6 and s["kp3"] is not None and s["kp4"] is not None
        assert kim_pilot.RANK_NAME in s["rows"] and s["kp2_delta"] is not None
        if kind == "native":
            assert "ego" in s["rows"]["vt2"]

    def test_row_count_mismatch_raises(self, tmp_path: Path) -> None:
        labels = _labels(1, 10)
        v = next(iter(labels))
        np.save(tmp_path / f"{v}.npy", np.zeros((9, BASE), dtype=np.float32))
        with pytest.raises(ValueError, match="input rows"):
            kim_pilot.score_set("rows", tmp_path, labels, _w0(), CPU, 1)

    def test_read_labels_from_either_output(self, tmp_path: Path) -> None:
        labels = _labels(2, 8)
        scores = {v: np.linspace(0, 1, 8) for v in labels}
        write_clip_scores(tmp_path / kim_pilot.CLIP_SCORES_NPZ, scores, labels)
        _, got_l = kim_pilot.read_labels(tmp_path)
        assert set(got_l) == set(labels)
        tad = tmp_path / "tad"
        (tad / evaluate.SCORES_DIRNAME).mkdir(parents=True)
        np.savez(tad / evaluate.SCORES_DIRNAME / "v0.npz", score=np.zeros(4), gt=np.ones(4))
        assert set(kim_pilot.read_labels(tad)[1]) == {"v0"}

    def test_decide(self) -> None:
        def row(m: float) -> dict[str, dict[str, float]]:
            return {"macro": {"mean": m, "low": m - 0.01, "high": m + 0.01},
                    "ego": {"mean": m, "low": m, "high": m}}

        def one_set(vt: float, clip: float) -> dict[str, object]:
            rows = {f"{f}2": row(0.5) for f in kim_pilot.FEATURES}
            rows["vt2"], rows["clip2"] = row(vt), row(clip)
            return {"rows": rows, "kp2_delta": {"mean": 0.01, "low": 0.001, "high": 0.02},
                    "kp3": {"share": 0.3, "chance": 0.1, "videos": 5},
                    "kp4": {"clips": 5, "median_abs_s": 0.3, "share_within": 0.6}}

        sets = {kim_pilot.DOTA_SET: one_set(0.65, 0.70), kim_pilot.DADA_SET: one_set(0.6, 0.63)}
        d = kim_pilot.decide(sets)
        assert d["KP1"]["verdict"] == "pass" and d["KP2"]["verdict"] == "pass"
        assert d["input"] == "clip"  # +0.05 and +0.03 >= 0.02 on both sets
        assert d["KP3"]["verdict"] == "fail" and d["KP4"]["verdict"] == "pass"
        sets[kim_pilot.DADA_SET] = one_set(0.6, 0.61)
        assert kim_pilot.decide(sets)["input"] == "vt"  # +0.01 on DADA: no switch


# --------------------------------------------------------------------------- read-out
DOTA_IDS = [f"src{i % 5}_{i:06d}" for i in range(40)]
DADA_IDS = [f"d{i}" for i in range(40)]


def _dota(d: Path, aucs: dict[str, float]) -> None:
    d.mkdir(parents=True)
    ci = {"mean": float(np.mean(list(aucs.values()))), "low": 0.0, "high": 1.0, "clips": 40}
    (d / CLIP_AUCS_JSON).write_text(json.dumps(aucs))
    (d / READOUT_JSON).write_text(json.dumps({"macro": ci, "position_ruler": ci,
                                              "global_steps": {"s2024": 1749}}))


def _dada(d: Path, aucs: dict[str, float]) -> None:
    d.mkdir(parents=True)
    ci = {"mean": float(np.mean(list(aucs.values()))), "low": 0.0, "high": 1.0, "clips": 40}
    (d / dada_whole_eval.CLIP_AUCS_JSON).write_text(json.dumps(aucs))
    (d / dada_whole_eval.READOUT_JSON).write_text(json.dumps({
        "per_run_macro": {"s2024": ci["mean"]}, "macro": ci, "micro": 0.7, "video_auc": None,
        "monotone_ruler_macro": ci, "middle_ruler_macro": ci}))


def _tad(d: Path, shift: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = d / "s2024" / evaluate.SCORES_DIRNAME
    scores.mkdir(parents=True)
    for i in range(constants.V2_TADZ_VIDEOS):
        gt = np.zeros(12, dtype=np.float32)
        if i < constants.V2_TADZ_ABNORMAL:
            gt[4:8] = 1.0
        np.savez(scores / f"v{i:03d}.npz", score=(rng.random(12) + shift * gt).astype(np.float32),
                 gt=gt)


def _aucs(ids: list[str], base: float, shift: float) -> dict[str, float]:
    rng = np.random.default_rng(int(base * 100))
    return {v: base + shift + 0.002 * rng.standard_normal() for v in ids}


def _runs(tmp_path: Path, rows: dict[str, tuple[float, float]]) -> list[str]:
    """rows: name -> (DoTA shift, DADA shift) over the same per-video base."""
    argv: list[str] = []
    for name, (s_dota, s_dada) in rows.items():
        _dota(tmp_path / "dota" / name, _aucs(DOTA_IDS, 0.8, s_dota))
        _dada(tmp_path / "dada" / name, _aucs(DADA_IDS, 0.7, s_dada))
        argv += ["--dota", name, str(tmp_path / "dota" / name),
                 "--dada", name, str(tmp_path / "dada" / name)]
    return argv


class TestReadout:
    def test_kernel_9_needs_a_gain_on_both_endpoints(self, tmp_path: Path) -> None:
        argv = _runs(tmp_path, {"W0": (0.0, 0.0), "K0": (-0.02, -0.02), "K0k9": (0.01, 0.01)})
        r = kim_readout.run(kim_readout.build_arg_parser().parse_args(
            [*argv, "--out-dir", str(tmp_path / "rep")]))
        dota = r["sets"]["dota"]["contrasts"]
        assert dota["K0k9 - K0"]["sentence"] == GAIN and dota["K0 - W0"]["sentence"] == COST
        assert r["decisions"]["kernel"] == kim_readout.KERNEL_9
        assert r["decisions"]["stop_rule"] == kim_readout.PENDING
        md = (tmp_path / "rep" / "kim_readout.md").read_text()
        assert "PRELIMINARY (n = 1)" in md and constants.KIM_PILOT_DISCLOSURE in md

    def test_kernel_3_unless_both(self, tmp_path: Path) -> None:
        argv = _runs(tmp_path, {"K0": (0.0, 0.0), "K0k9": (0.03, 0.0)})
        r = kim_readout.run(kim_readout.build_arg_parser().parse_args(
            [*argv, "--out-dir", str(tmp_path / "rep")]))
        assert r["sets"]["dada"]["contrasts"]["K0k9 - K0"]["sentence"] == NONE
        assert r["decisions"]["kernel"] == kim_readout.KERNEL_3

    @pytest.mark.parametrize(("k5", "verdict"), [
        ((-0.03, -0.03), kim_readout.CLOSED),
        ((-0.03, 0.0), kim_readout.CONTINUE),
    ])
    def test_stop_rule(self, tmp_path: Path, k5: tuple[float, float], verdict: str) -> None:
        argv = _runs(tmp_path, {"K0": (0.0, 0.0), "K2": (0.0, 0.0), "K5": k5})
        r = kim_readout.run(kim_readout.build_arg_parser().parse_args(
            [*argv, "--out-dir", str(tmp_path / "rep")]))
        assert r["decisions"]["stop_rule"] == verdict
        assert "K5 - K2" in r["sets"]["dota"]["contrasts"]

    def test_tad_is_printed_and_diagnostics_read(self, tmp_path: Path) -> None:
        argv = _runs(tmp_path, {"K0": (0.0, 0.0), "K5": (0.0, 0.0)})
        _tad(tmp_path / "tad" / "K0", 0.3, 0)
        _tad(tmp_path / "tad" / "K5", 3.0, 1)
        run_dir = tmp_path / "runs" / "K5"
        run_dir.mkdir(parents=True)
        lines = [{"epoch": e, "global_step": s, "mil": m, "kim_mil": 0.6,
                  "kim_w_out_norm": 0.01 * s}
                 for e, s, m in ((0, 1, 0.7), (1, 2, 0.4), (1, 3, 0.2))]
        (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
        r = kim_readout.run(kim_readout.build_arg_parser().parse_args([
            *argv, "--tad", "K0", str(tmp_path / "tad" / "K0"), "--tad", "K5",
            str(tmp_path / "tad" / "K5"), "--run", "K5", str(run_dir),
            "--out-dir", str(tmp_path / "rep")]))
        assert r["sets"]["tad"]["contrasts"]["K5 - K0"]["sentence"] == GAIN
        assert r["decisions"]["stop_rule"] == kim_readout.CONTINUE  # TAD never decides
        d = r["diagnostics"]["K5"]
        assert d["last_step"] == 3 and d["mil_last_epoch_mean"] == pytest.approx(0.3)
        assert d["kim_w_out_norm"] == {"first": 0.01, "last": 0.03}

    def test_needs_a_row(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one"):
            kim_readout.run(kim_readout.build_arg_parser().parse_args(
                ["--out-dir", str(tmp_path)]))
