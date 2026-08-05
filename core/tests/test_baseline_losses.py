"""Phase 3 unit tests: ported baseline losses (finiteness, gradients, edge cases).

Numerical parity with the baseline lives in ``test_baseline_parity.py``.
"""

from __future__ import annotations

import pytest
import torch

from core.losses import (
    CapContrastLoss,
    mil_loss,
    multi_class_mil_loss,
    multi_class_mil_loss_v2,
    pseudo_sup_mil_loss,
    supervised_loss,
)


def _finite_scalar(loss: torch.Tensor) -> None:
    assert loss.dim() == 0 and torch.isfinite(loss)


class TestMILLosses:
    def test_finite_and_differentiable(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4, 20, requires_grad=True)
        loss = mil_loss(
            logits, torch.tensor([0.0, 1.0, 1.0, 0.0]), torch.tensor([20, 12, 5, 20])
        )
        _finite_scalar(loss)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()

    def test_short_video_uses_k_of_one(self) -> None:
        logits = torch.zeros(1, 20)
        logits[0, 2] = 5.0
        # length 3 with topk_pct 16 → k = 1: only the max logit counts
        loss = mil_loss(logits, torch.tensor([1.0]), torch.tensor([3]))
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([5.0]), torch.tensor([1.0])
        )
        torch.testing.assert_close(loss, expected)

    def test_multi_class_variants(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4, 20, 5, requires_grad=True)
        labels = torch.tensor([0, 2, 4, 0])
        lengths = torch.tensor([20, 15, 8, 20])
        for fn in (multi_class_mil_loss, multi_class_mil_loss_v2):
            loss = fn(logits, labels, lengths)
            _finite_scalar(loss)

    def test_v2_differs_on_normal_videos(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(2, 20, 3)
        labels = torch.tensor([0, 1])
        lengths = torch.tensor([20, 20])
        v1 = multi_class_mil_loss(logits, labels, lengths, topk_pct=4)
        v2 = multi_class_mil_loss_v2(logits, labels, lengths, topk_pct=4)
        assert not torch.allclose(v1, v2)

    def test_topk_requires_some_k(self) -> None:
        with pytest.raises(ValueError, match="topk"):
            mil_loss(
                torch.randn(1, 5), torch.tensor([1.0]), torch.tensor([5]),
                topk_num=None, topk_pct=None,
            )


class TestDVSLosses:
    def test_supervised_loss_ignores_padding(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, 10)
        frame_labels = torch.zeros(2, 10)
        lengths = torch.tensor([10, 4])
        base = supervised_loss(logits, frame_labels, lengths)
        scribbled = logits.clone()
        scribbled[1, 4:] = 99.0
        torch.testing.assert_close(
            base, supervised_loss(scribbled, frame_labels, lengths)
        )

    def test_pseudo_sup_mil_abnormal_selects_in_span(self) -> None:
        logits = torch.zeros(1, 10)
        logits[0, 7] = 9.0  # huge logit OUTSIDE the annotated span
        logits[0, 2] = 1.0  # best logit inside the span
        frame_labels = torch.zeros(1, 10)
        frame_labels[0, 1:4] = 1.0
        loss = pseudo_sup_mil_loss(logits, frame_labels, torch.tensor([10]), topk_pct=4)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([1.0]), torch.tensor([1.0])
        )
        torch.testing.assert_close(loss, expected)

    def test_pseudo_sup_mil_all_normal_batch(self) -> None:
        torch.manual_seed(0)
        loss = pseudo_sup_mil_loss(
            torch.randn(3, 12), torch.zeros(3, 12), torch.tensor([12, 6, 12])
        )
        _finite_scalar(loss)


class TestCapContrastLoss:
    def _inputs(self) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(0)
        logits = torch.randn(4, 15) * 3
        lengths = torch.tensor([15, 10, 15, 7])
        v_feats = torch.randn(4, 15, 32)
        t_feats = torch.randn(2, 32)
        cls_label_idx = torch.tensor([0, 1, 0, 2])
        pseudo = torch.zeros(4, 15)
        pseudo[1, 2:6] = 1.0
        pseudo[3, 0:3] = 1.0
        return logits, lengths, v_feats, t_feats, cls_label_idx, pseudo

    @pytest.mark.parametrize("contrast_type", ["vanilla", "n3"])
    def test_finite_and_differentiable(self, contrast_type: str) -> None:
        logits, lengths, v_feats, t_feats, cls_label_idx, pseudo = self._inputs()
        v_feats.requires_grad_(True)
        loss = CapContrastLoss(contrast_type)(
            logits, lengths, v_feats, t_feats, cls_label_idx, pseudo
        )
        _finite_scalar(loss)
        loss.backward()
        assert v_feats.grad is not None and torch.isfinite(v_feats.grad).all()

    def test_unknown_contrast_type_raises(self) -> None:
        with pytest.raises(ValueError, match="contrast_type"):
            CapContrastLoss("n1")
