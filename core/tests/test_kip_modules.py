"""Phase 1 tests: KIP modules (spec §2-5.1) on random tensors, CPU-only."""

from __future__ import annotations

import logging
import time

import pytest
import torch

from core.config import KIPConfig
from core.kip import (
    GATE_SIGNAL_FEAT_VAR,
    GATE_TYPE_MLP_FROZEN,
    KIP,
    KinematicShift,
    MotionScoreHead,
    PMGFlowHead,
    shift_channels_reference,
    shift_channels_vectorized,
)

LOGGER = logging.getLogger(__name__)

BATCH, LENGTH, DIM, FLOW_DIM = 2, 50, 512, 256
MAX_SHIFT = DIM // 4


def make_inputs(lengths: tuple[int, ...] = (50, 37)) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    vt = torch.randn(len(lengths), LENGTH, DIM)
    mask = torch.zeros(len(lengths), LENGTH)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1.0
    return vt, mask


class TestPMGFlowHead:
    def test_output_shape(self):
        vt, _ = make_inputs()
        eo = PMGFlowHead()(vt)
        assert eo.shape == (BATCH, LENGTH, FLOW_DIM)

    def test_gradients_flow(self):
        vt, _ = make_inputs()
        head = PMGFlowHead()
        head(vt).sum().backward()
        assert all(p.grad is not None for p in head.parameters())

    def test_valid_positions_invariant_to_padding_content(self):
        vt, mask = make_inputs()
        head = PMGFlowHead()
        eo_a = head(vt, mask)
        corrupted = vt.clone()
        corrupted[1, 37:] = 999.0
        eo_b = head(corrupted, mask)
        assert torch.allclose(eo_a[1, :37], eo_b[1, :37])


class TestShiftEquivalence:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_vectorized_matches_reference(self, seed: int):
        torch.manual_seed(seed)
        vt = torch.randn(BATCH, LENGTH, DIM)
        counts = torch.randint(0, MAX_SHIFT + 1, (BATCH, LENGTH))
        assert torch.equal(
            shift_channels_vectorized(vt, counts), shift_channels_reference(vt, counts)
        )

    def test_forward_matches_reference_path(self):
        vt, mask = make_inputs()
        module = KinematicShift()
        eo = torch.randn(BATCH, LENGTH, FLOW_DIM)
        vk = module(vt, eo, mask)
        masked_vt = vt * mask.unsqueeze(-1)
        counts = module.compute_shift_counts(masked_vt, eo, mask)
        expected = shift_channels_reference(masked_vt, counts) * mask.unsqueeze(-1)
        assert torch.allclose(vk, expected)

    def test_benchmark_vectorized_vs_loop(self):
        torch.manual_seed(0)
        vt = torch.randn(2, 400, DIM)
        counts = torch.randint(0, MAX_SHIFT + 1, (2, 400))
        t0 = time.perf_counter()
        ref = shift_channels_reference(vt, counts)
        t_loop = time.perf_counter() - t0
        t0 = time.perf_counter()
        vec = shift_channels_vectorized(vt, counts)
        t_vec = time.perf_counter() - t0
        LOGGER.info("shift benchmark (2x400x512): loop %.4fs, vectorized %.4fs", t_loop, t_vec)
        assert torch.equal(ref, vec)


class TestKinematicShift:
    def test_shape_preserved(self):
        vt, mask = make_inputs()
        eo = torch.randn(BATCH, LENGTH, FLOW_DIM)
        assert KinematicShift()(vt, eo, mask).shape == vt.shape

    def test_boundary_zero_fill(self):
        torch.manual_seed(0)
        vt = torch.rand(1, 10, DIM) + 1.0  # strictly positive → zeros only from fill
        counts = torch.full((1, 10), 5, dtype=torch.long)
        vk = shift_channels_vectorized(vt, counts)
        assert torch.equal(vk[0, 0, :5], torch.zeros(5))  # t=0 has no past
        assert torch.equal(vk[0, -1, 5:10], torch.zeros(5))  # t=L-1 has no future
        assert torch.equal(vk[0, 0, 5:10], vt[0, 1, 5:10])  # future slice intact at t=0

    def test_present_slice_preserved(self):
        torch.manual_seed(0)
        vt = torch.randn(BATCH, LENGTH, DIM)
        counts = torch.randint(0, MAX_SHIFT + 1, (BATCH, LENGTH))
        vk = shift_channels_vectorized(vt, counts)
        assert torch.equal(vk[..., 2 * MAX_SHIFT :], vt[..., 2 * MAX_SHIFT :])

    def test_shift_counts_range_and_padding(self):
        vt, mask = make_inputs()
        module = KinematicShift()
        counts = module.compute_shift_counts(vt, torch.randn(BATCH, LENGTH, FLOW_DIM), mask)
        assert counts.min() >= 0
        assert counts.max() <= MAX_SHIFT
        assert (counts[1, 37:] == 0).all()  # padded positions never shift

    def test_padded_content_does_not_leak(self):
        vt, mask = make_inputs()
        eo = torch.randn(BATCH, LENGTH, FLOW_DIM)
        module = KinematicShift()
        vk_a = module(vt, eo, mask)
        corrupted = vt.clone()
        corrupted[1, 37:] = 999.0
        vk_b = module(corrupted, eo, mask)
        assert torch.allclose(vk_a, vk_b)
        assert (vk_a[1, 37:] == 0).all()

    def test_feat_var_gate_signal(self):
        vt, mask = make_inputs()
        module = KinematicShift(
            gate_type=GATE_TYPE_MLP_FROZEN, gate_signal=GATE_SIGNAL_FEAT_VAR
        )
        eo = torch.randn(BATCH, LENGTH, FLOW_DIM)
        assert module(vt, eo, mask).shape == vt.shape

    def test_unknown_gate_signal_rejected(self):
        with pytest.raises(ValueError):
            KinematicShift(gate_signal="bogus")


class TestMotionScoreHead:
    def test_shape_and_range(self):
        eo = torch.randn(BATCH, LENGTH, FLOW_DIM)
        yo = MotionScoreHead()(eo)
        assert yo.shape == (BATCH, LENGTH)
        assert yo.min() > 0.0
        assert yo.max() < 1.0

    def test_padded_scores_zero(self):
        _, mask = make_inputs()
        yo = MotionScoreHead()(torch.randn(BATCH, LENGTH, FLOW_DIM), mask)
        assert (yo[1, 37:] == 0).all()

    def test_gradients_flow(self):
        head = MotionScoreHead()
        head(torch.randn(BATCH, LENGTH, FLOW_DIM)).sum().backward()
        assert all(p.grad is not None for p in head.parameters())


class TestKIP:
    def test_output_shapes(self):
        vt, mask = make_inputs()
        vk, eo, yo, _ = KIP()(vt, mask)
        assert yo is not None
        assert vk.shape == (BATCH, LENGTH, DIM)
        assert eo.shape == (BATCH, LENGTH, FLOW_DIM)
        assert yo.shape == (BATCH, LENGTH)

    def test_gradients_reach_pmg_and_motion_head(self):
        vt, mask = make_inputs()
        kip = KIP()
        vk, eo, yo, _ = kip(vt, mask)
        assert yo is not None  # training graph builds the motion head
        (vk.sum() + eo.sum() + yo.sum()).backward()
        assert kip.mhead is not None
        assert all(p.grad is not None for p in kip.pmg.parameters())
        assert all(p.grad is not None for p in kip.mhead.parameters())

    def test_gate_mlp_receives_no_gradient_spec_as_written(self):
        """Hard floor → integer counts: the v1 gate MLP gets no gradient.

        Pins the defect that motivated the v3 rank gate. Must name
        ``mlp_frozen`` explicitly now that ``rank`` is the default and owns no
        MLP at all.
        """
        vt, mask = make_inputs()
        kip = KIP(gate_type=GATE_TYPE_MLP_FROZEN)
        vk, eo, yo, _ = kip(vt, mask)
        (vk.sum() + eo.sum() + yo.sum()).backward()
        assert kip.shift is not None
        assert kip.shift.mlp is not None
        assert all(p.grad is None for p in kip.shift.mlp.parameters())

    def test_align_projections(self):
        vt, mask = make_inputs()
        kip = KIP()
        _, eo, _, _ = kip(vt, mask)
        a, b = kip.project_for_align(eo, vt)
        assert a.shape == (BATCH, LENGTH, 128)
        assert b.shape == (BATCH, LENGTH, 128)
        (a.sum() + b.sum()).backward()
        assert kip.proj_flow is not None
        assert kip.proj_rgb is not None
        assert kip.proj_flow.weight.grad is not None
        assert kip.proj_rgb.weight.grad is not None

    def test_gate_shift_disabled_passthrough(self):
        vt, mask = make_inputs()
        kip = KIP(use_gate_shift=False)
        vk, _, _, _ = kip(vt, mask)
        assert kip.shift is None
        assert torch.equal(vk, vt * mask.unsqueeze(-1))

    def test_from_config_pmg_only_disables_shift(self):
        cfg = KIPConfig(pmg_only=True)
        assert KIP.from_config(cfg).shift is None

    def test_from_config_gate_signal(self):
        cfg = KIPConfig(gate_type=GATE_TYPE_MLP_FROZEN, gate_signal=GATE_SIGNAL_FEAT_VAR)
        kip = KIP.from_config(cfg)
        assert kip.shift is not None
        assert kip.shift.gate_signal == GATE_SIGNAL_FEAT_VAR

    def test_mask_optional(self):
        vt, _ = make_inputs()
        vk, eo, yo, _ = KIP()(vt)
        assert yo is not None
        assert vk.shape == vt.shape
        assert torch.isfinite(vk).all()
        assert torch.isfinite(eo).all()
        assert torch.isfinite(yo).all()
