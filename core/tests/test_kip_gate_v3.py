"""Phase 1 tests: v3 gate primitives (ECMR + rank map), CPU-only, data-free.

Covers plan T1.1-T1.3. The mask-invariance test is the load-bearing one: if a
clip's shift schedule depends on what else shared its batch, every arm measured
with this gate is irreproducible.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest
import torch

from core import constants
from core.kip.ecmr import ego_compensated_residual, rank_map, shift_counts_from_ratio

LOGGER = logging.getLogger(__name__)

FLOW_DIM = constants.FLOW_DIM
MAX_SHIFT = constants.HIDDEN_DIM // constants.FOLDING_FACTOR  # 128
LAM = constants.ECMR_EMA_LAMBDA


def ema_oracle(eo: np.ndarray, valid: np.ndarray, lam: float = LAM) -> np.ndarray:
    """Hand-rolled reference for one clip: ``(L, d)``, ``(L,)`` bool -> ``m (L,)``."""
    length, dim = eo.shape
    mu = np.zeros(dim, dtype=np.float64)
    started = False
    out = np.zeros(length, dtype=np.float64)
    for t in range(length):
        if valid[t]:
            mu = eo[t].astype(np.float64) if not started else lam * mu + (1.0 - lam) * eo[t]
            started = True
        # padded positions hold the prototype
        out[t] = np.linalg.norm(eo[t] - mu) if valid[t] else 0.0
    return out


def make_batch(
    lengths: tuple[int, ...], total: int | None = None, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random ``ê_O`` plus a left-aligned mask; padding filled with a loud value."""
    torch.manual_seed(seed)
    length = total if total is not None else max(lengths)
    eo = torch.randn(len(lengths), length, FLOW_DIM, dtype=torch.float64)
    mask = torch.zeros(len(lengths), length, dtype=torch.float64)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1.0
        eo[i, n:] = 999.0  # padding must never influence a valid position
    return eo, mask


class TestEgoCompensatedResidual:
    @pytest.mark.parametrize("length", [1, 2, 7, 33])
    def test_matches_numpy_oracle(self, length: int):
        eo, mask = make_batch((length,), total=length)
        m, _ = ego_compensated_residual(eo, mask)
        expected = ema_oracle(eo[0].numpy(), mask[0].numpy().astype(bool))
        assert np.allclose(m[0].numpy(), expected, atol=1e-10)

    def test_matches_oracle_with_padding(self):
        eo, mask = make_batch((33, 12), total=40)
        m, _ = ego_compensated_residual(eo, mask)
        for i in range(2):
            expected = ema_oracle(eo[i].numpy(), mask[i].numpy().astype(bool))
            assert np.allclose(m[i].numpy(), expected, atol=1e-10)

    def test_first_valid_position_seeds_prototype(self):
        """mu_0 = eo_0 => m_0 = 0. Seeding at zero would make frame 0 the loudest."""
        eo, mask = make_batch((16,), total=16)
        m, mu_norm = ego_compensated_residual(eo, mask)
        assert float(m[0, 0]) == pytest.approx(0.0, abs=1e-12)
        assert float(mu_norm[0, 0]) == pytest.approx(float(eo[0, 0].norm()), abs=1e-12)

    def test_constant_flow_gives_zero_residual(self):
        """Static-camera degradation: mu_t == eo_t, so the residual vanishes."""
        eo = torch.ones(2, 20, FLOW_DIM, dtype=torch.float64) * 3.5
        m, _ = ego_compensated_residual(eo, None)
        assert torch.allclose(m, torch.zeros_like(m), atol=1e-12)

    def test_residual_equals_lambda_times_previous_deviation(self):
        """Documented identity: delta_t == lam * (eo_t - mu_{t-1}) for t >= 1."""
        eo, mask = make_batch((10,), total=10)
        m, _ = ego_compensated_residual(eo, mask)
        # rebuild mu_{t-1} from the oracle recurrence
        arr = eo[0].numpy().astype(np.float64)
        mu = arr[0].copy()
        for t in range(1, 10):
            expected = LAM * np.linalg.norm(arr[t] - mu)
            assert float(m[0, t]) == pytest.approx(expected, abs=1e-10)
            mu = LAM * mu + (1.0 - LAM) * arr[t]

    def test_padded_positions_zeroed_and_state_held(self):
        eo, mask = make_batch((6, 6), total=20)
        m, mu_norm = ego_compensated_residual(eo, mask)
        assert torch.all(m[:, 6:] == 0)
        assert torch.all(mu_norm[:, 6:] == 0)

    def test_padding_content_does_not_leak(self):
        eo, mask = make_batch((9,), total=25)
        m_a, _ = ego_compensated_residual(eo, mask)
        corrupted = eo.clone()
        corrupted[0, 9:] = -12345.0
        m_b, _ = ego_compensated_residual(corrupted, mask)
        assert torch.allclose(m_a, m_b, atol=1e-12)

    def test_rejects_invalid_lambda(self):
        eo, mask = make_batch((4,), total=4)
        for bad in (-0.1, 1.0, 1.5):
            with pytest.raises(ValueError):
                ego_compensated_residual(eo, mask, lam=bad)

    def test_gradient_flows_to_eo(self):
        """m is differentiable in eo; the gradient stops later, at the floor()."""
        eo, mask = make_batch((8,), total=8)
        eo = eo.float().requires_grad_(True)
        m, _ = ego_compensated_residual(eo, mask.float())
        m.sum().backward()
        assert eo.grad is not None
        assert torch.isfinite(eo.grad).all()


class TestRankMap:
    def test_is_normalized_permutation(self):
        eo, mask = make_batch((30,), total=30)
        m, _ = ego_compensated_residual(eo, mask)
        r = rank_map(m, mask)
        expected = torch.arange(30, dtype=torch.float64) / 29.0
        assert torch.allclose(r[0].sort().values, expected, atol=1e-12)

    def test_spans_full_range_on_every_clip(self):
        """The property that rules out H4' by construction."""
        eo, mask = make_batch((30, 17, 2), total=30, seed=3)
        m, _ = ego_compensated_residual(eo, mask)
        r = rank_map(m, mask)
        for i, n in enumerate((30, 17, 2)):
            valid = r[i, :n]
            assert float(valid.min()) == pytest.approx(0.0, abs=1e-12)
            assert float(valid.max()) == pytest.approx(1.0, abs=1e-12)

    def test_uses_n_valid_not_l(self):
        """Same clip, two padding widths -> identical ranks at valid positions."""
        eo_a, mask_a = make_batch((12,), total=12, seed=5)
        eo_b = torch.cat([eo_a, torch.full((1, 40, FLOW_DIM), 999.0, dtype=torch.float64)], dim=1)
        mask_b = torch.cat([mask_a, torch.zeros(1, 40, dtype=torch.float64)], dim=1)
        m_a, _ = ego_compensated_residual(eo_a, mask_a)
        m_b, _ = ego_compensated_residual(eo_b, mask_b)
        assert torch.allclose(rank_map(m_a, mask_a)[0, :12], rank_map(m_b, mask_b)[0, :12])

    def test_is_scale_free(self):
        """Rank consumes only ordering, so a global gain leaves r unchanged."""
        eo, mask = make_batch((21,), total=21, seed=7)
        m, _ = ego_compensated_residual(eo, mask)
        m_scaled, _ = ego_compensated_residual(eo * 1000.0, mask)
        assert torch.allclose(rank_map(m, mask), rank_map(m_scaled, mask))

    def test_single_valid_position_is_zero(self):
        eo, mask = make_batch((1,), total=10)
        m, _ = ego_compensated_residual(eo, mask)
        r = rank_map(m, mask)
        assert torch.all(r == 0)
        assert torch.isfinite(r).all()

    def test_padded_positions_zeroed(self):
        eo, mask = make_batch((5,), total=18)
        m, _ = ego_compensated_residual(eo, mask)
        assert torch.all(rank_map(m, mask)[0, 5:] == 0)

    def test_constant_residual_warns(self, caplog):
        m = torch.zeros(1, 12, dtype=torch.float64)
        with caplog.at_level(logging.WARNING, logger="core.kip.ecmr"):
            rank_map(m, None)
        assert "effectively constant" in caplog.text

    def test_no_mask_is_all_valid(self):
        eo, _ = make_batch((14,), total=14)
        m, _ = ego_compensated_residual(eo, None)
        ones = torch.ones(1, 14, dtype=torch.float64)
        assert torch.allclose(rank_map(m, None), rank_map(m, ones))


class TestShiftCountsFromRatio:
    def test_endpoints(self):
        r = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64)
        counts = shift_counts_from_ratio(r, MAX_SHIFT)
        assert counts.tolist() == [[0, MAX_SHIFT // 2, MAX_SHIFT]]

    def test_range_and_dtype(self):
        eo, mask = make_batch((30, 17), total=30, seed=11)
        m, _ = ego_compensated_residual(eo, mask)
        counts = shift_counts_from_ratio(rank_map(m, mask), MAX_SHIFT, mask)
        assert counts.dtype == torch.long
        assert int(counts.min()) == 0
        assert int(counts.max()) == MAX_SHIFT
        assert int(counts[:, :].max()) <= MAX_SHIFT

    def test_clamps_out_of_range_ratio(self):
        r = torch.tensor([[-0.5, 2.0]], dtype=torch.float64)
        counts = shift_counts_from_ratio(r, MAX_SHIFT)
        assert counts.tolist() == [[0, MAX_SHIFT]]

    def test_padded_positions_never_shift(self):
        eo, mask = make_batch((9,), total=24)
        m, _ = ego_compensated_residual(eo, mask)
        counts = shift_counts_from_ratio(rank_map(m, mask), MAX_SHIFT, mask)
        assert torch.all(counts[0, 9:] == 0)

    def test_slices_never_overlap(self):
        """2*s_t <= 256 < 512, so past/future/present slices stay disjoint."""
        assert 2 * MAX_SHIFT < constants.HIDDEN_DIM


class TestMaskInvariance:
    """The load-bearing guard: batch composition must not move any valid position."""

    @pytest.mark.parametrize("total", [40, 64, 128])
    def test_m_r_and_s_invariant_to_padding_width(self, total: int):
        lengths = (31, 18, 5)
        eo_ref, mask_ref = make_batch(lengths, total=max(lengths), seed=13)
        eo_pad = torch.full((len(lengths), total, FLOW_DIM), 777.0, dtype=torch.float64)
        eo_pad[:, : max(lengths)] = eo_ref
        mask_pad = torch.zeros(len(lengths), total, dtype=torch.float64)
        mask_pad[:, : max(lengths)] = mask_ref

        m_ref, mu_ref = ego_compensated_residual(eo_ref, mask_ref)
        m_pad, mu_pad = ego_compensated_residual(eo_pad, mask_pad)
        r_ref = rank_map(m_ref, mask_ref)
        r_pad = rank_map(m_pad, mask_pad)
        s_ref = shift_counts_from_ratio(r_ref, MAX_SHIFT, mask_ref)
        s_pad = shift_counts_from_ratio(r_pad, MAX_SHIFT, mask_pad)

        for i, n in enumerate(lengths):
            assert torch.allclose(m_ref[i, :n], m_pad[i, :n], atol=1e-12)
            assert torch.allclose(mu_ref[i, :n], mu_pad[i, :n], atol=1e-12)
            assert torch.allclose(r_ref[i, :n], r_pad[i, :n], atol=1e-12)
            assert torch.equal(s_ref[i, :n], s_pad[i, :n])

    def test_clip_is_invariant_to_its_batch_neighbours(self):
        """A clip scored alone and scored beside others must get the same s_t."""
        eo_solo, mask_solo = make_batch((23,), total=23, seed=17)
        neighbour = torch.randn(1, 50, FLOW_DIM, dtype=torch.float64)
        eo_batch = torch.full((2, 50, FLOW_DIM), 5.0, dtype=torch.float64)
        eo_batch[0, :23] = eo_solo[0]
        eo_batch[1] = neighbour[0]
        mask_batch = torch.zeros(2, 50, dtype=torch.float64)
        mask_batch[0, :23] = 1.0
        mask_batch[1, :] = 1.0

        m_solo, _ = ego_compensated_residual(eo_solo, mask_solo)
        m_batch, _ = ego_compensated_residual(eo_batch, mask_batch)
        s_solo = shift_counts_from_ratio(rank_map(m_solo, mask_solo), MAX_SHIFT, mask_solo)
        s_batch = shift_counts_from_ratio(rank_map(m_batch, mask_batch), MAX_SHIFT, mask_batch)
        assert torch.equal(s_solo[0], s_batch[0, :23])


class TestParameterFree:
    def test_gate_path_has_no_parameters(self):
        """Phases 3b/3c are functions, not modules: nothing to train, nothing to seed."""
        import core.kip.ecmr as ecmr

        assert not any(isinstance(v, torch.nn.Module) for v in vars(ecmr).values())
        assert not any(isinstance(v, torch.nn.Parameter) for v in vars(ecmr).values())


class TestPerformance:
    def test_recurrence_cost_is_logged(self):
        """The EMA loop is O(L) sequential; record its cost rather than assume it."""
        eo = torch.randn(4, 512, FLOW_DIM)
        mask = torch.ones(4, 512)
        start = time.perf_counter()
        for _ in range(5):
            ego_compensated_residual(eo, mask)
        elapsed = (time.perf_counter() - start) / 5.0
        LOGGER.info("ECMR forward B=4 L=512 d=256: %.1f ms", elapsed * 1e3)
        assert elapsed < 2.0  # generous; a regression past this is pathological
