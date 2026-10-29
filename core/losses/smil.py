"""v5 spatial MIL loss ``L_sMIL`` (PREREG_V5 §2.1 S2-6) and v5.1's ``L_tile`` (§3).

Per step the strongest query decides, ``S_t = max_m s_{t,m}``; per sequence the top-k steps
decide, ``s_hat = mean of the top-k valid S_t`` with ``k = max(floor(L / 16), 1)``;
``L_sMIL = BCE(s_hat, y)``. It sees every sequence the trunk's ``L_MIL`` sees (DVS-synthesized
ones included), with the same label and valid lengths.

The scores are probabilities, not logits; ``binary_cross_entropy`` is unsafe under autocast, so
``BCE(s_hat, y)`` is taken as ``binary_cross_entropy_with_logits(logit(s_hat, eps), y)`` in float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from core import constants


def spatial_mil_loss(
    probs: Tensor,
    labels: Tensor,
    lengths: Tensor,
    topk_pct: int = constants.V5_SMIL_TOPK_PCT,
) -> Tensor:
    """``probs (B, L, M)`` in ``(0, 1)``, ``labels (B,)`` in {0, 1}, ``lengths (B,)`` -> scalar."""
    with torch.autocast(probs.device.type, enabled=False):
        step = probs.float().amax(dim=-1)
        pooled = []
        for b in range(step.shape[0]):
            n = int(lengths[b])
            k = max(n // topk_pct, 1)
            pooled.append(step[b, :n].topk(k).values.mean())
        logits = torch.logit(torch.stack(pooled), eps=constants.V5_SMIL_EPS)
        return F.binary_cross_entropy_with_logits(logits, labels.float())


def boundary_steps(pseudo: Tensor, rows: int) -> Tensor:
    """``(L,)`` bool: steps within ``rows`` of a change in ``pseudo`` (an inserted DVS boundary)."""
    near = torch.zeros_like(pseudo, dtype=torch.bool)
    if rows < 1 or len(pseudo) < 2:
        return near
    edges = (pseudo[1:] != pseudo[:-1]).nonzero(as_tuple=True)[0]  # change between e and e + 1
    for edge in edges.tolist():
        near[max(0, edge + 1 - rows) : edge + 1 + rows] = True
    return near


def tile_mil_loss(
    logits: Tensor,
    labels: Tensor,
    lengths: Tensor,
    pseudo: Tensor,
    is_synth: Tensor,
    dense_negatives: bool,
    topk_cells: int = constants.V5_TILE_TOPK_CELLS,
    beta: float = constants.V5_TILE_BETA,
    boundary_rows: int = 0,
    topk_pct: int = constants.V5_SMIL_TOPK_PCT,
) -> dict[str, Tensor]:
    """``L_tile`` (PREREG_V5 §3, proposal §3.2) on scorer logits ``e (B, L, cells)``.

    Per sequence, ``p = sigmoid(e)``, ``S_t = max_i p_{t,i}`` and
    ``s_hat = mean of the top-K cells of the top-k steps`` (``k = max(floor(L / 16), 1)``):

    * abnormal (``label 1``, DVS-synthesized included): ``BCE(s_hat, 1)``; with
      ``dense_negatives`` a synthesized sequence's pseudo-normal steps, minus those within
      ``boundary_rows`` of an inserted boundary, add ``mean BCE(p, 0)`` over their cells;
    * normal: ``mean BCE(p, 0)`` over every valid cell (``dense_negatives``, R6c) or
      ``BCE(s_hat, 0)`` (bag negatives only, R5c);
    * ``+ beta * (mean p`` on abnormal, ``mean p^2`` on normal``)``.

    Returns ``tile`` (the batch mean) and detached diagnostics.
    """
    with torch.autocast(logits.device.type, enabled=False):
        e_all = logits.float()
        total, p_abn, p_nor = [], [], []
        for b in range(e_all.shape[0]):
            n = int(lengths[b])
            e = e_all[b, :n]
            p = e.sigmoid()
            k = max(n // topk_pct, 1)
            steps = p.amax(dim=-1).topk(k).indices
            s_hat = p[steps].topk(min(topk_cells, p.shape[-1]), dim=-1).values.mean()
            bag = torch.logit(s_hat, eps=constants.V5_SMIL_EPS)
            if bool(labels[b] > 0.5):
                term = F.binary_cross_entropy_with_logits(bag, bag.new_ones(()))
                if dense_negatives and bool(is_synth[b]):
                    keep = (pseudo[b, :n] < 0.5) & ~boundary_steps(pseudo[b, :n], boundary_rows)
                    if keep.any():
                        term = term + F.softplus(e[keep]).mean()
                term = term + beta * p.mean()
                p_abn.append(p.mean().detach())
            else:
                term = (
                    F.softplus(e).mean()
                    if dense_negatives
                    else F.binary_cross_entropy_with_logits(bag, bag.new_zeros(()))
                )
                term = term + beta * p.square().mean()
                p_nor.append(p.mean().detach())
            total.append(term)
        out = {"tile": torch.stack(total).mean()}
        if p_abn:
            out["tile_p_abnormal"] = torch.stack(p_abn).mean()
        if p_nor:
            out["tile_p_normal"] = torch.stack(p_nor).mean()
        return out


__all__ = ["boundary_steps", "spatial_mil_loss", "tile_mil_loss"]
