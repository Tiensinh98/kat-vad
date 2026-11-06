"""KIM's two training terms — rule ``core/docs/final/KIM_RUN.md`` §2 D7-D10.

* :func:`kim_mil_loss` — ``kim_mil``: per bag, the mean of the top-k **probabilities**
  ``kim_score`` over valid steps (k = ``max(1, n_valid // topk_pct)``, the host's k), then BCE
  against the bag label (proposal §3.2.4; user U1). Not KIP v1's ``kinematic_loss``.
* :func:`kim_anchors` — the anchor set (D9): per abnormal bag, its top-k valid steps by
  ``kim_score`` that also exceed ``anchor_thresh``.
* :func:`kim_nkc_loss` — ``kim_nkc`` (D10): InfoNCE per anchor with ``mu_1`` as the positive and
  ``mu_0`` plus the anchor's M most similar valid steps of normal bags in the batch as
  negatives, positive kept in the denominator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from core import constants


def _topk_count(n_valid: int, topk_pct: int) -> int:
    return max(1, n_valid // topk_pct)


def kim_mil_loss(score: Tensor, labels: Tensor, lengths: Tensor, topk_pct: int) -> Tensor:
    """``BCE(mean(top-k kim_score), y)`` averaged over bags with at least one valid step."""
    means, kept = [], []
    for b in range(score.shape[0]):
        n = int(lengths[b])
        if n == 0:
            continue
        means.append(score[b, :n].topk(_topk_count(n, topk_pct)).values.mean())
        kept.append(labels[b])
    if not means:
        return score.sum() * 0.0
    # BCE on probabilities as BCE-with-logits: autocast-safe (as core/kip/losses.py)
    return F.binary_cross_entropy_with_logits(
        torch.logit(torch.stack(means).float(), eps=constants.LOGIT_EPS),
        torch.stack(kept).float(),
    )


def kim_anchors(
    score: Tensor, labels: Tensor, lengths: Tensor, topk_pct: int, thresh: float
) -> Tensor:
    """``(B, L)`` bool: top-k valid steps of each abnormal bag with ``kim_score > thresh``."""
    anchors = torch.zeros_like(score, dtype=torch.bool)
    detached = score.detach()
    for b in range(score.shape[0]):
        n = int(lengths[b])
        if n == 0 or labels[b] < 0.5:
            continue
        top = detached[b, :n].topk(_topk_count(n, topk_pct))
        keep = top.indices[top.values > thresh]
        anchors[b, keep] = True
    return anchors


def normal_steps(labels: Tensor, mask: Tensor) -> Tensor:
    """``(B, L)`` bool: valid steps of label-0 bags (exact negatives)."""
    return mask.bool() & (labels < 0.5).unsqueeze(1)


def kim_nkc_loss(
    kin_feat: Tensor,
    anchors: Tensor,
    negatives: Tensor,
    prototypes: Tensor,
    num_negatives: int,
    temperature: float,
) -> Tensor:
    """Normal-anchored kinematic contrast; 0 (graph-connected) when there is no anchor."""
    anchor_feat = kin_feat[anchors].float()  # (N, F)
    if anchor_feat.shape[0] == 0:
        return kin_feat.sum() * 0.0
    anchor_n = F.normalize(anchor_feat, dim=-1)
    proto_n = F.normalize(prototypes.detach().float(), dim=-1)  # (2, F)
    logits = [anchor_n @ proto_n[1:2].T, anchor_n @ proto_n[0:1].T]  # positive first
    pool = kin_feat[negatives].float()
    if pool.shape[0]:
        sim = anchor_n @ F.normalize(pool, dim=-1).T  # (N, P)
        logits.append(sim.topk(min(num_negatives, pool.shape[0]), dim=-1).values)
    all_logits = torch.cat(logits, dim=-1) / temperature
    return (torch.logsumexp(all_logits, dim=-1) - all_logits[:, 0]).mean()


__all__ = ["kim_anchors", "kim_mil_loss", "kim_nkc_loss", "normal_steps"]
