"""Phase 2 tests: KIP losses (spec §5.2) on random tensors, CPU-only."""

from __future__ import annotations

import pytest
import torch

from core.kip import kinematic_loss, kip_alignment_loss, kip_reconstruction_loss

BATCH, LENGTH, FLOW_DIM, PROJ_DIM = 2, 50, 256, 128
VALID_LENGTHS = (50, 37)


def make_mask(lengths: tuple[int, ...] = VALID_LENGTHS) -> torch.Tensor:
    mask = torch.zeros(len(lengths), LENGTH)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1.0
    return mask


def assert_finite_scalar(loss: torch.Tensor) -> None:
    assert loss.dim() == 0
    assert torch.isfinite(loss)


class TestReconstructionLoss:
    def test_finite_scalar_and_gradient(self):
        torch.manual_seed(0)
        eo_hat = torch.randn(BATCH, LENGTH, FLOW_DIM, requires_grad=True)
        eo = torch.randn(BATCH, LENGTH, FLOW_DIM)
        loss = kip_reconstruction_loss(eo_hat, eo, make_mask())
        assert_finite_scalar(loss)
        loss.backward()
        assert eo_hat.grad is not None
        assert torch.isfinite(eo_hat.grad).all()

    def test_zero_on_perfect_reconstruction(self):
        eo = torch.randn(BATCH, LENGTH, FLOW_DIM)
        assert kip_reconstruction_loss(eo, eo.clone(), make_mask()).item() == 0.0

    def test_padded_positions_ignored(self):
        torch.manual_seed(0)
        mask = make_mask()
        eo_hat = torch.randn(BATCH, LENGTH, FLOW_DIM)
        eo = torch.randn(BATCH, LENGTH, FLOW_DIM)
        loss_a = kip_reconstruction_loss(eo_hat, eo, mask)
        corrupted = eo_hat.clone()
        corrupted[1, 37:] = 999.0
        loss_b = kip_reconstruction_loss(corrupted, eo, mask)
        assert torch.allclose(loss_a, loss_b)

    def test_matches_manual_computation(self):
        torch.manual_seed(0)
        mask = make_mask()
        eo_hat = torch.randn(BATCH, LENGTH, FLOW_DIM)
        eo = torch.randn(BATCH, LENGTH, FLOW_DIM)
        expected = torch.cat(
            [
                (eo_hat[0, :50] - eo[0, :50]).pow(2).mean(-1),
                (eo_hat[1, :37] - eo[1, :37]).pow(2).mean(-1),
            ]
        ).mean()
        assert torch.allclose(kip_reconstruction_loss(eo_hat, eo, mask), expected)

    def test_no_mask(self):
        eo_hat = torch.randn(BATCH, LENGTH, FLOW_DIM)
        eo = torch.randn(BATCH, LENGTH, FLOW_DIM)
        assert torch.allclose(
            kip_reconstruction_loss(eo_hat, eo), (eo_hat - eo).pow(2).mean()
        )


class TestAlignmentLoss:
    def make_projs(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        flow = torch.randn(BATCH, LENGTH, PROJ_DIM, requires_grad=True)
        rgb = torch.randn(BATCH, LENGTH, PROJ_DIM, requires_grad=True)
        return flow, rgb

    def test_finite_scalar_and_gradient(self):
        flow, rgb = self.make_projs()
        loss = kip_alignment_loss(flow, rgb, make_mask())
        assert_finite_scalar(loss)
        loss.backward()
        assert flow.grad is not None
        assert rgb.grad is not None
        assert torch.isfinite(flow.grad).all()

    def test_aligned_pairs_beat_random(self):
        torch.manual_seed(0)
        flow = torch.randn(BATCH, LENGTH, PROJ_DIM)
        aligned = kip_alignment_loss(flow, flow.clone(), make_mask())
        random = kip_alignment_loss(flow, torch.randn(BATCH, LENGTH, PROJ_DIM), make_mask())
        assert aligned.item() < random.item()

    def test_padded_positions_ignored(self):
        torch.manual_seed(0)
        mask = make_mask()
        flow = torch.randn(BATCH, LENGTH, PROJ_DIM)
        rgb = torch.randn(BATCH, LENGTH, PROJ_DIM)
        loss_a = kip_alignment_loss(flow, rgb, mask)
        corrupted = flow.clone()
        corrupted[1, 37:] = 999.0
        loss_b = kip_alignment_loss(corrupted, rgb, mask)
        assert torch.allclose(loss_a, loss_b)

    def test_subsample_flag(self):
        torch.manual_seed(0)
        flow = torch.randn(BATCH, LENGTH, PROJ_DIM)
        rgb = torch.randn(BATCH, LENGTH, PROJ_DIM)
        loss = kip_alignment_loss(flow, rgb, make_mask(), subsample=10)
        assert_finite_scalar(loss)
        # subsample larger than n_valid must be a no-op
        full = kip_alignment_loss(flow, rgb, make_mask(), subsample=LENGTH + 1)
        assert torch.allclose(full, kip_alignment_loss(flow, rgb, make_mask()))

    def test_exclude_window_changes_loss(self):
        torch.manual_seed(0)
        flow = torch.randn(BATCH, LENGTH, PROJ_DIM)
        rgb = torch.randn(BATCH, LENGTH, PROJ_DIM)
        default = kip_alignment_loss(flow, rgb, make_mask())
        excluded = kip_alignment_loss(flow, rgb, make_mask(), exclude_window=3)
        assert_finite_scalar(excluded)
        assert not torch.allclose(default, excluded)

    def test_single_valid_frame(self):
        flow = torch.randn(1, LENGTH, PROJ_DIM)
        rgb = torch.randn(1, LENGTH, PROJ_DIM)
        loss = kip_alignment_loss(flow, rgb, make_mask((1,)))
        assert_finite_scalar(loss)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)  # one class → CE = 0


class TestKinematicLoss:
    def make_scores(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        motion = torch.rand(BATCH, LENGTH).clamp(0.01, 0.99).requires_grad_()
        main = torch.rand(BATCH, LENGTH, requires_grad=True)
        labels = torch.tensor([1.0, 0.0])
        return motion, main, labels, make_mask()

    def test_finite_scalar_and_gradient_to_motion_only(self):
        motion, main, labels, mask = self.make_scores()
        loss = kinematic_loss(motion, main, labels, mask)
        assert_finite_scalar(loss)
        loss.backward()
        assert motion.grad is not None
        assert torch.isfinite(motion.grad).all()
        assert main.grad is None  # y^bin is detached inside the loss

    def test_all_normal_batch(self):
        torch.manual_seed(0)
        motion = torch.rand(BATCH, LENGTH).clamp(0.01, 0.99)
        main = torch.rand(BATCH, LENGTH)
        labels = torch.zeros(BATCH)
        loss = kinematic_loss(motion, main, labels, make_mask())
        assert_finite_scalar(loss)  # consistency term empty → pure BCE

    def test_short_video_k_clamped(self):
        torch.manual_seed(0)
        motion = torch.rand(1, LENGTH).clamp(0.01, 0.99)
        main = torch.rand(1, LENGTH)
        loss = kinematic_loss(motion, main, torch.ones(1), make_mask((3,)))
        assert_finite_scalar(loss)  # n_valid=3 < topk_pct → k=1

    def test_padded_positions_ignored(self):
        motion, main, labels, mask = self.make_scores()
        loss_a = kinematic_loss(motion, main, labels, mask)
        corrupted_main = main.clone()
        corrupted_main[1, 37:] = 999.0
        loss_b = kinematic_loss(motion, corrupted_main, labels, mask)
        assert torch.allclose(loss_a, loss_b)

    def test_yp_anchor_used_for_synthesized(self):
        torch.manual_seed(0)
        motion = torch.rand(1, LENGTH).clamp(0.01, 0.99)
        main = torch.rand(1, LENGTH)
        labels = torch.ones(1)
        yp = (torch.rand(1, LENGTH) > 0.5).float()
        mask = make_mask((50,))
        synth = torch.tensor([True])
        with_yp = kinematic_loss(
            motion, main, labels, mask, pseudo_labels=yp, is_synthesized=synth
        )
        without_yp = kinematic_loss(
            motion, main, labels, mask, pseudo_labels=yp, is_synthesized=synth,
            use_yp_anchor=False,
        )
        assert_finite_scalar(with_yp)
        assert not torch.allclose(with_yp, without_yp)

    def test_yp_anchor_ignored_for_real_videos(self):
        torch.manual_seed(0)
        motion = torch.rand(1, LENGTH).clamp(0.01, 0.99)
        main = torch.rand(1, LENGTH)
        labels = torch.ones(1)
        yp = (torch.rand(1, LENGTH) > 0.5).float()
        mask = make_mask((50,))
        real = torch.tensor([False])
        with_flag = kinematic_loss(
            motion, main, labels, mask, pseudo_labels=yp, is_synthesized=real
        )
        baseline = kinematic_loss(motion, main, labels, mask)
        assert torch.allclose(with_flag, baseline)

    def test_mil_term_matches_plain_bce_reference(self):
        # The logit-trick (autocast-safe) MIL term must equal plain BCE on the
        # per-video top-k means exactly.
        motion, main, labels, mask = self.make_scores()
        loss = kinematic_loss(motion, main, labels, mask, beta=0.0)
        topk_means = []
        for b, n in enumerate(VALID_LENGTHS):
            k = max(1, n // 16)
            topk_means.append(motion[b, :n].topk(k).values.mean())
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.logit(torch.stack(topk_means), eps=1e-6), labels
        )
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_runs_under_cpu_autocast(self):
        # Regression for the stage-2 AMP crash: plain F.binary_cross_entropy is
        # rejected inside CUDA autocast regions. CPU autocast can't reproduce the
        # raise, but this at least exercises the loss under an autocast context;
        # the CUDA-only ban is enforced statically (ruff TID251 banned-api).
        motion, main, labels, mask = self.make_scores()
        with torch.autocast("cpu", enabled=True):
            loss = kinematic_loss(motion, main, labels, mask)
        assert_finite_scalar(loss)
        loss.backward()
        assert motion.grad is not None
        assert torch.isfinite(motion.grad).all()
        assert torch.allclose(
            loss, kinematic_loss(motion.detach(), main.detach(), labels, mask), atol=1e-5
        )

    def test_beta_scales_consistency_term(self):
        torch.manual_seed(0)
        motion = torch.rand(1, LENGTH).clamp(0.01, 0.99)
        main = torch.rand(1, LENGTH)
        labels = torch.ones(1)
        mask = make_mask((50,))
        loss_b0 = kinematic_loss(motion, main, labels, mask, beta=0.0)
        loss_b1 = kinematic_loss(motion, main, labels, mask, beta=1.0)
        loss_b05 = kinematic_loss(motion, main, labels, mask, beta=0.5)
        cons = loss_b1 - loss_b0
        assert torch.allclose(loss_b05, loss_b0 + 0.5 * cons, atol=1e-6)
