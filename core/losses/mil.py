"""MIL losses re-implemented from the LaGoVAD baseline (``L_MIL``, ``L_MIL-align``).

Behavioral parity with ``LaGoVAD-PreVAD/src/models/LaGoVAD/losses.py`` is
asserted by ``core/tests/test_baseline_losses.py`` on identical random inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from core import constants


def _topk_k(n_valid: int, topk_pct: int | None, topk_num: int | None) -> int:
    """Per-video k: ``max(1, n_valid // topk_pct)`` or a fixed ``topk_num``."""
    if topk_num is not None:
        return max(1, int(topk_num))
    if topk_pct is None:
        raise ValueError("Either topk_num or topk_pct must be provided")
    return max(1, n_valid // topk_pct)


def mil_loss(
    logits: Tensor,
    labels: Tensor,
    lengths: Tensor,
    topk_num: int | None = None,
    topk_pct: int | None = constants.MIL_TOPK_PCT,
) -> Tensor:
    """Vanilla top-k MIL: BCE(mean of top-k logits per video, video label).

    ``logits (B, T)``, ``labels (B,)`` in {0, 1}, ``lengths (B,)``.
    """
    batch = logits.shape[0]
    video_logits = []
    for b in range(batch):
        n = int(lengths[b])
        k = _topk_k(n, topk_pct, topk_num)
        video_logits.append(logits[b, :n].topk(k).values.mean())
    return F.binary_cross_entropy_with_logits(
        torch.stack(video_logits), labels.to(logits.dtype)
    )


def multi_class_mil_loss(
    logits: Tensor,
    labels: Tensor,
    lengths: Tensor,
    topk_num: int | None = None,
    topk_pct: int | None = constants.MUL_MIL_TOPK_PCT,
) -> Tensor:
    """Per-class top-k aggregation + cross entropy (``L_MIL-align``, v1).

    ``logits (B, T, C)``, ``labels (B,)`` class indices incl. normal=0.
    """
    batch = logits.shape[0]
    video_logits = []
    for b in range(batch):
        n = int(lengths[b])
        k = _topk_k(n, topk_pct, topk_num)
        video_logits.append(logits[b, :n].topk(k, dim=0).values.mean(dim=0))  # (C,)
    return F.cross_entropy(torch.stack(video_logits), labels)


def multi_class_mil_loss_v2(
    logits: Tensor,
    labels: Tensor,
    lengths: Tensor,
    topk_num: int | None = None,
    topk_pct: int | None = constants.MUL_MIL_TOPK_PCT,
) -> Tensor:
    """v2: normal videos aggregate their bottom-k instances instead of top-k."""
    batch = logits.shape[0]
    video_logits = []
    for b in range(batch):
        n = int(lengths[b])
        k = _topk_k(n, topk_pct, topk_num)
        largest = bool(labels[b] != 0)
        video_logits.append(
            logits[b, :n].topk(k, dim=0, largest=largest).values.mean(dim=0)
        )
    return F.cross_entropy(torch.stack(video_logits), labels)


__all__ = ["mil_loss", "multi_class_mil_loss", "multi_class_mil_loss_v2"]
