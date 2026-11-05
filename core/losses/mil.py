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


def patch_mil_loss(
    cell_logits: Tensor,
    frame_logits: Tensor,
    labels: Tensor,
    lengths: Tensor,
    topk_pct: int | None = constants.MIL_TOPK_PCT,
    top_cells: int = constants.SV2_TOP_CELLS,
) -> tuple[Tensor, Tensor]:
    """``L_patch``: the same selection in every bag (proposal §3.2.3).

    Top-k steps by host frame score (no gradient through the choice), top-``top_cells`` cells
    in each; BCE pushes them to the bag label, 1 or 0. Dense negatives could be met by
    sharpening the map alone, so a normal bag gets exactly this selection too.

    ``cell_logits (B, T, P)``, ``frame_logits (B, T)``, ``labels (B,)``, ``lengths (B,)``.
    Returns the loss and each bag's mean selected ``q`` (detached, for the C3 read).
    """
    losses, means = [], []
    for b in range(cell_logits.shape[0]):
        n = int(lengths[b])
        steps = frame_logits[b, :n].detach().topk(_topk_k(n, topk_pct, None)).indices
        chosen = cell_logits[b, steps].topk(top_cells, dim=-1).values
        losses.append(F.binary_cross_entropy_with_logits(chosen, chosen.new_full(
            chosen.shape, float(labels[b]))))
        means.append(chosen.detach().sigmoid().mean())
    return torch.stack(losses).mean(), torch.stack(means)


def abnormal_bottomk_loss(
    logits: Tensor,
    labels: Tensor,
    lengths: Tensor,
    topk_num: int | None = None,
    topk_pct: int | None = constants.BOTTOMK_MIL_TOPK_PCT,
) -> Tensor:
    """Push the **lowest**-k frames of each *abnormal* video toward 0 (Phase 1.2).

    ``logits (B, T)``, ``labels (B,)`` in {0, 1}, ``lengths (B,)``.

    The wired objective contains no term that lowers *any* frame of an abnormal
    clip: :func:`mil_loss` raises its top-k, :func:`pseudo_sup_mil_loss` raises
    the top-k inside the span, and :func:`multi_class_mil_loss` raises the top-k
    toward the anomaly class. A per-clip constant therefore satisfies the loss
    completely, which is what DADA-2000 measured (normal frames inside abnormal
    clips scored 0.4078 against 0.4076 for true positives). This adds the
    missing pressure: an abnormal bag must contain *some* normal instance.

    Normal videos are excluded — :func:`mil_loss` already drives their maximum
    to 0, which bounds every frame; adding them here would double-count.

    Weight defaults to 0 (``constants.BOTTOMK_WEIGHT``): this is an arm.
    """
    rows = [b for b in range(logits.shape[0]) if float(labels[b]) > 0.5]
    if not rows:  # all-normal batch: contribute 0, keep the graph connected
        return logits.sum() * 0.0
    video_logits = []
    for b in rows:
        n = int(lengths[b])
        k = _topk_k(n, topk_pct, topk_num)
        video_logits.append(logits[b, :n].topk(k, largest=False).values.mean())
    stacked = torch.stack(video_logits)
    return F.binary_cross_entropy_with_logits(stacked, torch.zeros_like(stacked))


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


__all__ = [
    "abnormal_bottomk_loss",
    "mil_loss",
    "multi_class_mil_loss",
    "multi_class_mil_loss_v2",
]
