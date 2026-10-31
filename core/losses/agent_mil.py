"""v6 agent-level MIL loss ``L_agent`` (proposal §3.2, PREREG_V6 §2.3).

Per sequence with candidate probabilities ``p_{t,i} = sigmoid(e_{t,i})`` over the valid candidates
``C_t`` (agents, ego, and no-agent when no agent is detected):

* ``S_t = max_i p_{t,i}``; ``s_hat`` = mean of the top-``K`` candidates of each of the top-``k``
  steps by ``S_t`` (``k = max(floor(L / 16), 1)``);
* abnormal (label 1, DVS-synthesized included): ``BCE(s_hat, 1) + beta * mean p``; a synthesized
  sequence's pseudo-normal steps (minus those within ``boundary_rows`` of an inserted boundary)
  also enter the dense negative term;
* normal: ``mean BCE(p, 0)`` over every candidate of every valid step (dense negatives), or
  ``BCE(s_hat, 0)`` with ``dense_negatives=False`` (ladder row 9).

Ground-truth boxes and frame labels are never used.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

from core import constants
from core.losses.smil import boundary_steps


def agent_mil_loss(
    logits: Tensor,
    candidates: Tensor,
    labels: Tensor,
    lengths: Tensor,
    pseudo: Tensor,
    is_synth: Tensor,
    dense_negatives: bool = True,
    topk_agents: int = constants.V6_TOPK_AGENTS,
    beta: float = constants.V6_BETA,
    boundary_rows: int = 0,
    topk_pct: int = constants.V6_MIL_TOPK_PCT,
) -> dict[str, Tensor]:
    """``L_agent`` on ``logits (B, L, C)`` with ``candidates (B, L, C)`` bool; returns ``agent``
    (the batch mean) and detached mechanics (top-candidate probability, top-pick shares)."""
    with torch.autocast(logits.device.type, enabled=False):
        e_all = logits.float()
        total: list[Tensor] = []
        top_abn, top_nor, ego_share, none_share = [], [], [], []
        k_slots = e_all.shape[-1] - 2
        for b in range(e_all.shape[0]):
            n = int(lengths[b])
            valid = candidates[b, :n]
            e = e_all[b, :n]
            p = e.sigmoid().masked_fill(~valid, 0.0)
            k = max(n // topk_pct, 1)
            s_t = p.amax(-1)
            steps = s_t.topk(min(k, n)).indices
            cand = p[steps]
            top = cand.topk(min(topk_agents, cand.shape[-1]), dim=-1)
            s_hat = top.values.mean()
            bag = torch.logit(s_hat, eps=constants.V6_SMIL_EPS)
            picks = top.indices[:, 0]
            ego_share.append((picks == k_slots).float().mean())
            none_share.append((picks == k_slots + 1).float().mean())
            if bool(labels[b] > 0.5):
                term = functional.binary_cross_entropy_with_logits(bag, bag.new_ones(()))
                if dense_negatives and bool(is_synth[b]):
                    keep = (pseudo[b, :n] < 0.5) & ~boundary_steps(pseudo[b, :n], boundary_rows)
                    neg = valid & keep[:, None]
                    if neg.any():
                        term = term + functional.softplus(e[neg]).mean()
                term = term + beta * p[valid].mean()
                top_abn.append(s_t.mean().detach())
            else:
                term = (
                    functional.softplus(e[valid]).mean()
                    if dense_negatives
                    else functional.binary_cross_entropy_with_logits(bag, bag.new_zeros(()))
                )
                top_nor.append(s_t.mean().detach())
            total.append(term)
        out = {
            "agent": torch.stack(total).mean(),
            "agent_ego_pick": torch.stack(ego_share).mean().detach(),
            "agent_none_pick": torch.stack(none_share).mean().detach(),
        }
        if top_abn:
            out["agent_top_abnormal"] = torch.stack(top_abn).mean()
        if top_nor:
            out["agent_top_normal"] = torch.stack(top_nor).mean()
        return out


__all__ = ["agent_mil_loss"]
