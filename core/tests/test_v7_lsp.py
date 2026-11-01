"""v7 Step 0: the LSP module and the D1-D5 tool (PREREG_V7 §1). CPU, data-free."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from core import constants
from core.models.lsp import (
    LatentSurprisePredictor,
    SurpriseSeries,
    clip_unit_vectors,
    context_windows,
    evidence_vector,
    first_target,
    parameter_count,
    surprise_series,
    unit_vectors,
)
from core.tools import v7_lsp

D = constants.V7_DIM
H = constants.V7_HORIZON
K = constants.V7_CONTEXTS[0]
CENSUS = constants.OUTPUT_ROOT / "v2" / "REPORTS" / "nexar_n0" / "census.json"


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _series(s: np.ndarray, p: np.ndarray, valid: np.ndarray | None = None) -> SurpriseSeries:
    valid = np.ones(len(s), bool) if valid is None else valid
    rho = np.log((s + constants.V7_EPS) / (p + constants.V7_EPS))
    return SurpriseSeries(
        s=s.astype(np.float32),
        p=p.astype(np.float32),
        rho=rho.astype(np.float32),
        r=np.zeros((len(s), D), np.float32),
        valid=valid,
    )


# --------------------------------------------------------------------------- model


def test_constants_fixed_by_prereg() -> None:
    assert H == 16 and constants.V7_CONTEXTS == (16, 32)
    assert constants.V7_HELDOUT_FIRST_TARGET == 47 >= 2 * H
    assert constants.V7_NEXAR_LATTICE_STEP == 3 and constants.V7_DOTA_LATTICE_STEP == 1


def test_unit_vectors_recover_affine_free_layernorm() -> None:
    m = torch.as_tensor(_rng().normal(size=(20, D)), dtype=torch.float32)
    ln = torch.nn.LayerNorm(D, eps=constants.VIDEOMAE_LN_EPS)
    with torch.no_grad():
        ln.weight.uniform_(0.5, 2.0)
        ln.bias.normal_()
        cached = ln(m).numpy()
    u = unit_vectors(cached, ln.weight.detach().numpy(), ln.bias.detach().numpy())
    ref = F.layer_norm(m, (D,), eps=constants.VIDEOMAE_LN_EPS).numpy()
    np.testing.assert_allclose(u, ref, atol=1e-4)


def test_unit_vectors_refuse_zero_gamma() -> None:
    gamma = np.ones(D, np.float32)
    gamma[3] = 0.0
    with pytest.raises(ValueError, match="not recoverable"):
        unit_vectors(np.zeros((2, D)), gamma, np.zeros(D, np.float32))


def test_predictor_starts_as_copy_last_and_size() -> None:
    model = LatentSurprisePredictor(context=K)
    ctx = torch.randn(4, K, D)
    torch.testing.assert_close(model.predict(ctx), ctx[:, -1])
    assert (
        3_500_000 < parameter_count(model) < 3_800_000
    )  # proposal "about 3.5 M"; 3.70 M with readout


def test_predictor_rejects_wrong_context() -> None:
    with pytest.raises(ValueError, match="context"):
        LatentSurprisePredictor(context=K)(torch.randn(2, K + 1, D))


def test_context_windows_alignment() -> None:
    u = torch.arange(60, dtype=torch.float32)[:, None].repeat(1, D)
    w = context_windows(u, K, H)
    assert w.shape == (60 - H - K + 1, K, D)
    first = first_target(K, H)
    for i in (0, 5, w.shape[0] - 1):
        assert float(w[i, -1, 0]) == first + i - H  # last context row = u_{t - Δ}
        assert float(w[i, 0, 0]) == first + i - H - K + 1


def test_surprise_series_at_init_equals_copy_last() -> None:
    model = LatentSurprisePredictor(context=K)
    u = _rng().normal(size=(80, D)).astype(np.float32)
    ser = surprise_series(model, u)
    first = first_target(K)
    assert not ser.valid[:first].any() and ser.valid[first:].all()
    np.testing.assert_allclose(ser.s[first:], ser.p[first:], rtol=1e-6)
    np.testing.assert_allclose(ser.rho[first:], 0.0, atol=1e-6)
    assert (ser.s[:first] == 0).all() and (ser.r[:first] == 0).all()


def test_evidence_vector_masks_and_shape() -> None:
    model = LatentSurprisePredictor(context=K)
    ser = surprise_series(model, _rng().normal(size=(70, D)).astype(np.float32))
    e = evidence_vector(ser, s_mean=0.5, s_std=0.1)
    assert e.shape == (70, D + 2)
    assert (e[~ser.valid] == 0).all()
    first = first_target(K)
    np.testing.assert_allclose(e[first:, D], (ser.s[first:] - 0.5) / 0.1, rtol=1e-5)
    np.testing.assert_allclose(e[first:, :D].mean(1), 0.0, atol=1e-5)


# --------------------------------------------------------------------------- stage 1


def test_packed_batch_matches_context_windows() -> None:
    seqs = [_rng(i).normal(size=(n, D)).astype(np.float32) for i, n in enumerate((60, 75))]
    packed = v7_lsp.Packed(seqs, torch.device("cpu"))
    first = first_target(K)
    rows = packed.targets(first)
    assert len(rows) == (60 - first) + (75 - first)
    ctx, target = packed.batch(rows, K, H)
    w0 = context_windows(torch.as_tensor(seqs[0]), K, H)
    torch.testing.assert_close(ctx[: w0.shape[0]], w0)
    torch.testing.assert_close(target[0], torch.as_tensor(seqs[0][first]))
    torch.testing.assert_close(
        ctx[w0.shape[0]], torch.as_tensor(seqs[1][first - H - K + 1 : first - H + 1])
    )


def test_packed_sample_stays_inside_valid_targets() -> None:
    packed = v7_lsp.Packed(
        [np.zeros((40, D), np.float32), np.zeros((20, D), np.float32)], torch.device("cpu")
    )
    rows = packed.sample(first_target(K), 50, _rng())
    assert len(rows) == 50  # the 20-row sequence has no valid target
    assert rows.min() >= first_target(K) and rows.max() < 40


def test_heldout_fixed_predictors_on_a_linear_trend() -> None:
    t = np.arange(90, dtype=np.float32)[:, None]
    seq = (t * _rng().normal(size=(1, D)) * 0.01).astype(np.float32)
    packed = v7_lsp.Packed([seq], torch.device("cpu"))
    losses = v7_lsp.heldout_losses(LatentSurprisePredictor(context=K), packed, with_fixed=True)
    assert losses["const_velocity"] < 1e-10
    assert losses["copy_last"] > 0
    assert losses["lsp"] == pytest.approx(losses["copy_last"])  # zero-init readout


def test_heldout_refuses_an_early_first_target() -> None:
    packed = v7_lsp.Packed([np.zeros((90, D), np.float32)], torch.device("cpu"))
    with pytest.raises(ValueError, match="too early"):
        v7_lsp.heldout_losses(LatentSurprisePredictor(context=32), packed, first=40)


def test_train_lsp_never_ends_worse_than_copy_last(tmp_path: Path) -> None:
    rng = _rng(3)
    basis = rng.normal(size=(2, D)).astype(np.float32)

    def seq(phase: float) -> np.ndarray:
        t = np.arange(120)[:, None] * 0.37 + phase
        out: np.ndarray = (np.sin(t) * basis[0] + np.cos(0.5 * t) * basis[1]).astype(np.float32)
        return out

    train = v7_lsp.Packed([seq(x) for x in rng.uniform(0, 6, 6)], torch.device("cpu"))
    held = v7_lsp.Packed([seq(x) for x in rng.uniform(0, 6, 2)], torch.device("cpu"))
    start = v7_lsp.heldout_losses(LatentSurprisePredictor(context=K), held)["lsp"]
    fit = v7_lsp.train_lsp(train, held, K, epochs=4)
    assert len(fit.history) == 4 and fit.best_heldout <= start
    path = tmp_path / "m.pt"
    v7_lsp.save_model(path, fit.model, {"best_epoch": fit.best_epoch})
    again = v7_lsp.load_model(path, torch.device("cpu"))
    ctx = torch.randn(3, K, D)
    torch.testing.assert_close(again.predict(ctx), fit.model.predict(ctx))


def test_choose_context_tie_rule() -> None:
    assert v7_lsp.choose_context({16: 1.0, 32: 0.995}) == 16  # 0.5 % better -> tie -> smaller
    assert v7_lsp.choose_context({16: 1.0, 32: 0.98}) == 32
    assert v7_lsp.choose_context({16: 0.9, 32: 1.0}) == 16


# --------------------------------------------------------------------------- diagnostics


def test_d1_constant_sequence_fails_and_noise_passes() -> None:
    const = [np.ones((60, D), np.float32)]
    assert v7_lsp.d1_verdict(v7_lsp.lag_cosines(const, constants.V7_D1_MAX_LAG))["failed"]
    noise = [
        _rng().normal(size=(200, D)).astype(np.float32)
        + np.linspace(0, 5, 200)[:, None] * _rng(1).normal(size=(1, D)).astype(np.float32)
    ]
    curve = v7_lsp.lag_cosines(noise, constants.V7_D1_MAX_LAG)
    assert set(curve) == set(range(1, constants.V7_D1_MAX_LAG + 1))
    assert all(-1.0 <= c <= 1.0 for c in curve.values())


def test_lattices_cover_every_phase() -> None:
    u = np.arange(10)[:, None].repeat(D, 1).astype(np.float32)
    phases = v7_lsp.lattices(u, 3)
    assert [p[:, 0].tolist() for p in phases] == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]


def test_d2_pass_rule() -> None:
    class Fake:
        context = 16

    out = {"lsp": 0.8, "copy_last": 1.0, "const_velocity": 1.5, "context_mean": 0.95}
    orig = v7_lsp.heldout_losses
    v7_lsp.heldout_losses = lambda *a, **k: out
    try:
        res = v7_lsp.d2(Fake(), None)  # type: ignore[arg-type]
    finally:
        v7_lsp.heldout_losses = orig
    assert res["best_fixed"] == "context_mean"
    assert res["rel_gain"] == pytest.approx(1 - 0.8 / 0.95) and res["passed"]


def test_d3_nexar_reads_video_max() -> None:
    labels = {f"n{i}": i % 2 for i in range(20)}
    series = {}
    for v, y in labels.items():
        s = np.full(30, 0.1 + 0.5 * y)
        series[v] = _series(s, np.full(30, 0.3))
    out = v7_lsp.d3_nexar(series, labels)
    assert out["videos"] == 20 and out["auc_s"] == 1.0 and out["auc_p"] == 0.5


def _clip(onset: int, n: int, s_peak: int, p_peak: int, ego: bool) -> v7_lsp.DotaClip:
    s, p = np.full(n, 0.1), np.full(n, 0.1)
    s[s_peak], p[p_peak] = 1.0, 1.0
    valid = np.arange(n) >= 4
    labels = (np.arange(n) >= onset).astype(np.int64)
    return v7_lsp.DotaClip(_series(s, p, valid), labels, ego)


def test_d3_dota_and_d5_on_planted_clips() -> None:
    clips = {
        f"vid{i % 3}_{i:06d}": _clip(20, 40, s_peak=25, p_peak=10, ego=bool(i % 2))
        for i in range(12)
    }
    d3 = v7_lsp.d3_dota(clips)
    assert d3["clips_read"] == 12
    assert d3["all"]["s_minus_p"]["mean"] > 0
    assert d3["ego"]["clips"] + d3["non_ego"]["clips"] == 12
    nexar = {"a": _series(np.full(20, 0.2), np.full(20, 0.3))}
    d5 = v7_lsp.d5(nexar, {"a": 0}, clips)
    assert d5["lead_in_max_share"] == {"s": 0.0, "p": 1.0} and d5["clips"] == 12


def test_d4_runs_and_flags() -> None:
    rng = _rng(5)
    nexar = {f"n{i}": _series(rng.uniform(0, 0.2, 40), np.full(40, 0.3)) for i in range(12)}
    labels = dict.fromkeys(nexar, 0)
    clips = {
        f"vid{i}_{i:06d}": v7_lsp.DotaClip(
            _series(rng.uniform(0.8, 1.0, 40), np.full(40, 0.3)), np.zeros(40, np.int64), None
        )
        for i in range(12)
    }
    out = v7_lsp.d4(nexar, labels, clips, anomaly_auc=0.6)
    assert out["source_auc"] == pytest.approx(1.0) and out["flag"] is True


def test_verdict_needs_every_check() -> None:
    d1 = {"failed": False}
    d2 = {"passed": True}
    d3n = {"auc_s": 0.7, "auc_p": 0.6}
    d3d = {"all": {"s_minus_p": {"mean": 0.01}}}
    assert v7_lsp.verdict(d1, d2, d3n, d3d)[0] == constants.V7_VERDICT_GO
    assert v7_lsp.verdict(None, d2, d3n, d3d)[0] == constants.V7_VERDICT_NO_GO
    assert (
        v7_lsp.verdict(d1, d2, {"auc_s": 0.6, "auc_p": 0.6}, d3d)[0] == constants.V7_VERDICT_NO_GO
    )
    assert (
        v7_lsp.verdict(d1, d2, d3n, {"all": {"s_minus_p": None}})[0] == constants.V7_VERDICT_NO_GO
    )


def test_render_full_readout() -> None:
    clips = {f"vid{i % 3}_{i:06d}": _clip(20, 40, 25, 10, bool(i % 2)) for i in range(12)}
    nexar = {
        f"n{i}": _series(np.full(30, 0.1 + 0.5 * (i % 2)), np.full(30, 0.3)) for i in range(10)
    }
    labels = {v: i % 2 for i, v in enumerate(nexar)}
    d3n = v7_lsp.d3_nexar(nexar, labels)
    d3d = v7_lsp.d3_dota(clips)
    d2 = {
        "losses": {"lsp": 0.8, "copy_last": 1.0},
        "best_fixed": "copy_last",
        "rel_gain": 0.2,
        "passed": True,
    }
    d1 = {"cos_k1": 0.99, "cos_horizon": 0.8, "failed": False}
    decision, checks = v7_lsp.verdict(d1, d2, d3n, d3d)
    text = v7_lsp.render(
        {
            "rule": v7_lsp.RULE,
            "decision": decision,
            "checks": checks,
            "selected": {"context": 16, "horizon": 16, "s_mean": 0.1, "s_std": 0.01},
            "d1": d1,
            "d2": d2,
            "d3_nexar": d3n,
            "d3_dota": d3d,
            "d4": v7_lsp.d4(nexar, labels, clips, d3n["auc_s"]),
            "d5": v7_lsp.d5(nexar, labels, clips),
            "dota_missing": [],
        }
    )
    assert "Decision: GO" in text and "| non_ego |" in text


def test_cli_parses_every_command() -> None:
    parser = v7_lsp.build_arg_parser()
    for cmd in ("d1", "train", "diag"):
        args = parser.parse_args([cmd, "--census", "c.json"])
        assert isinstance(args, argparse.Namespace) and args.command == cmd


@pytest.mark.skipif(not CENSUS.exists(), reason="local Nexar census absent")
def test_frozen_holdout_matches_the_seeded_draw() -> None:
    sets = v7_lsp.nexar_sets(
        CENSUS, constants.V2_SPLITS_DIR, constants.V7_SPLITS_DIR / constants.V7_HOLDOUT_FILENAME
    )
    assert (len(sets.train), len(sets.holdout), len(sets.val)) == (404, 45, 225)
    assert not set(sets.train) & set(sets.holdout)
    assert all(sets.labels[v] == 0 for v in sets.train + sets.holdout)


# --------------------------------------------------------------------------- §1.1 CLIP stream


def test_clip_unit_vectors_are_affine_free_layernorm() -> None:
    c = _rng(7).normal(size=(9, constants.CLIP_FEATURE_DIM)).astype(np.float32) * 3 + 1
    u = clip_unit_vectors(c)
    np.testing.assert_allclose(u.mean(1), 0.0, atol=1e-5)
    np.testing.assert_allclose(u.std(1), 1.0, atol=1e-3)


def test_clip_stream_trains_saves_and_loads(tmp_path: Path) -> None:
    dim = constants.V7_STREAM_DIMS[constants.V7_STREAM_CLIP]
    heads = constants.V7_STREAM_HEADS[constants.V7_STREAM_CLIP]
    seqs = [_rng(i).normal(size=(80, dim)).astype(np.float32) for i in range(3)]
    train = v7_lsp.Packed(seqs[:2], torch.device("cpu"))
    held = v7_lsp.Packed(seqs[2:], torch.device("cpu"))
    fit = v7_lsp.train_lsp(train, held, K, epochs=1, heads=heads)
    assert (fit.model.dim, fit.model.heads) == (dim, heads)
    assert 6_000_000 < parameter_count(fit.model) < 7_000_000
    path = tmp_path / "clip.pt"
    v7_lsp.save_model(path, fit.model, {})
    again = v7_lsp.load_model(path, torch.device("cpu"))
    assert (again.dim, again.heads, again.context) == (dim, heads, K)


def test_load_model_reads_a_v2s_checkpoint_without_dim(tmp_path: Path) -> None:
    model = LatentSurprisePredictor(context=K)
    path = tmp_path / "old.pt"
    torch.save({"state_dict": model.state_dict(), "context": K, "best_epoch": 3}, path)
    again = v7_lsp.load_model(path, torch.device("cpu"))
    assert (again.dim, again.heads) == (constants.V7_DIM, constants.V7_HEADS)


def test_resolve_defaults_per_stream() -> None:
    parser = v7_lsp.build_arg_parser()
    clip = v7_lsp.resolve_defaults(parser.parse_args(["diag", "--census", "c", "--stream", "clip"]))
    assert clip.video_dir.name == constants.NEXAR_CLIP_S1_DATASET
    assert clip.dota_video_dir.name == "DoTA_s1_ncc"
    assert clip.out_dir.name == "step0_clip"
    v2s = v7_lsp.resolve_defaults(parser.parse_args(["train", "--census", "c"]))
    assert v2s.video_dir.name == "Nexar_s1_squash" and v2s.out_dir.name == "step0"
    assert not hasattr(v2s, "dota_video_dir")
