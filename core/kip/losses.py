"""KIP losses — spec §5.2.

Three losses, all mask-aware, all returning finite scalars on any batch:

- :func:`kip_reconstruction_loss` — ``L_KIP_rec``, masked MSE(ê_O, cached e_O).
- :func:`kip_alignment_loss` — ``L_KIP_align``, bidirectional per-video InfoNCE
  between projected pseudo-flow and RGB features, with optional A11 mitigations
  (position subsampling, ±w temporal-neighbor exclusion; defaults off).
- :func:`kinematic_loss` — ``L_kin``, top-k MIL on the motion curve plus a
  smooth-L1 consistency term on abnormal videos (``y^p`` anchor on synthesized
  samples when ``use_yp_anchor``).

Loss weights (lambda_rec, lambda_align, gamma_kin) are applied by the training
loop, not here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from core import constants


def _zero_like_scalar(reference: Tensor) -> Tensor:
    """Differentiable zero scalar on the right device/dtype (degenerate cases)."""
    return reference.sum() * 0.0


def kip_reconstruction_loss(eo_hat: Tensor, eo: Tensor, mask: Tensor | None = None) -> Tensor:
    """``L_KIP_rec``: mean over valid ``(b, l)`` of the per-position feature MSE.

    ``eo_hat, eo (B, L, d_flow)``, ``mask (B, L)`` with 1 = valid.
    """
    sq = (eo_hat - eo).pow(2).mean(dim=-1)  # (B, L)
    if mask is None:
        return sq.mean()
    valid = mask.bool()
    if not bool(valid.any()):
        return _zero_like_scalar(sq)
    return sq[valid].mean()


def kip_alignment_loss(
    flow_proj: Tensor,
    rgb_proj: Tensor,
    mask: Tensor | None = None,
    tau: float = constants.TAU_ALIGN,
    subsample: int = 0,
    exclude_window: int = 0,
) -> Tensor:
    """``L_KIP_align``: bidirectional snippet-level InfoNCE, averaged over videos.

    ``flow_proj = proj_flow(ê_O)``, ``rgb_proj = proj_rgb(v^t)``, both
    ``(B, L, d_c)`` and unnormalized (L2 normalization happens here).

    A11 mitigations (default off = spec-as-written):
    - ``subsample > 0``: random subset of that many valid positions per video.
    - ``exclude_window > 0``: temporal neighbors within ±w of the positive are
      dropped from the negative set (masked to ``NEG_INF_MASK_VALUE``).
    """
    batch, length, _ = flow_proj.shape
    per_video: list[Tensor] = []
    for b in range(batch):
        if mask is not None:
            positions = mask[b].bool().nonzero(as_tuple=True)[0]
        else:
            positions = torch.arange(length, device=flow_proj.device)
        n = int(positions.numel())
        if n == 0:
            continue
        if 0 < subsample < n:
            keep = torch.randperm(n, device=positions.device)[:subsample]
            positions = positions[keep.sort().values]
            n = subsample
        a = F.normalize(flow_proj[b, positions], dim=-1)  # (n, d_c)
        r = F.normalize(rgb_proj[b, positions], dim=-1)  # (n, d_c)
        sim = a @ r.T / tau  # (n, n)
        if exclude_window > 0:
            dist = (positions.view(-1, 1) - positions.view(1, -1)).abs()
            neighbors = (dist <= exclude_window) & (dist > 0)
            sim = sim.masked_fill(neighbors, constants.NEG_INF_MASK_VALUE)
        targets = torch.arange(n, device=sim.device)
        loss_f2r = F.cross_entropy(sim, targets)
        loss_r2f = F.cross_entropy(sim.T, targets)
        per_video.append(0.5 * (loss_f2r + loss_r2f))
    if not per_video:
        return _zero_like_scalar(flow_proj)
    return torch.stack(per_video).mean()


def _topk_k(n_valid: int, topk_pct: int) -> int:
    """Dynamic per-video k (A12): ``max(1, n_valid // topk_pct)``, never > n_valid."""
    return max(1, n_valid // topk_pct)


def kinematic_loss(
    motion_scores: Tensor,
    main_scores: Tensor,
    video_labels: Tensor,
    mask: Tensor | None = None,
    pseudo_labels: Tensor | None = None,
    is_synthesized: Tensor | None = None,
    use_yp_anchor: bool = True,
    beta: float = constants.BETA_CONS,
    topk_pct: int = constants.MIL_TOPK_PCT,
) -> Tensor:
    """``L_kin = BCE(topk_mean(ŷ_O), ŷ) + β · consistency`` (spec §5.2c).

    ``motion_scores = ŷ_O (B, L)`` in [0, 1]; ``main_scores = y^bin (B, L)``
    (detached here — never a gradient path); ``video_labels (B,)`` in {0, 1}.
    On abnormal videos the smooth-L1 consistency term compares ``ŷ_O`` to the
    anchor curve on the union of both top-k index sets; the anchor is
    ``pseudo_labels`` (``y^p``) for synthesized samples when ``use_yp_anchor``,
    otherwise the detached main scores.
    """
    main_scores = main_scores.detach()
    batch, length = motion_scores.shape
    topk_means: list[Tensor] = []
    kept_labels: list[Tensor] = []
    cons_terms: list[Tensor] = []
    for b in range(batch):
        if mask is not None:
            positions = mask[b].bool().nonzero(as_tuple=True)[0]
        else:
            positions = torch.arange(length, device=motion_scores.device)
        n = int(positions.numel())
        if n == 0:
            continue  # all-padded video: no evidence, no loss contribution
        k = _topk_k(n, topk_pct)
        yo = motion_scores[b, positions]  # (n,)
        topk_means.append(yo.topk(k).values.mean())
        kept_labels.append(video_labels[b])

        if video_labels[b] < 0.5:
            continue  # consistency term is abnormal-only
        anchor_src = main_scores[b, positions]
        if (
            use_yp_anchor
            and pseudo_labels is not None
            and is_synthesized is not None
            and bool(is_synthesized[b])
        ):
            anchor_src = pseudo_labels[b, positions].detach().to(motion_scores.dtype)
        idx_main = anchor_src.topk(k).indices
        idx_mot = yo.topk(k).indices
        union = torch.cat([idx_main, idx_mot]).unique()
        cons_terms.append(F.smooth_l1_loss(yo[union], anchor_src[union]))

    if not topk_means:
        return _zero_like_scalar(motion_scores)
    # BCE(p, y) == BCE_with_logits(logit(p), y); plain binary_cross_entropy is
    # rejected inside autocast regions (AMP), the with_logits form is not.
    mil_motion = F.binary_cross_entropy_with_logits(
        torch.logit(torch.stack(topk_means).float(), eps=constants.LOGIT_EPS),
        torch.stack(kept_labels).float(),
    )
    if cons_terms:
        return mil_motion + beta * torch.stack(cons_terms).mean()
    return mil_motion


__all__ = ["kinematic_loss", "kip_alignment_loss", "kip_reconstruction_loss"]
