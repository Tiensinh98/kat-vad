"""Spatial Attention Aggregation (SA²) in the CLIP branch -- LaGoVAD-STP's visual part.

Proposal ``core/docs/final/`` §3.2.2 (STPrompt Eqs. 1-3). The motion prior -- which ``K`` of
the 49 window-crop patches of a frame changed most against its neighbouring sampled frames --
is fixed and computed offline (``core.tools.stp_patches``); this module only aggregates the
selected patches ``P_{t,k}`` with the frame's CLS feature ``c_t`` as the query::

    a_{t,k} = softmax_k( (W_q c_t)^T (W_k P_{t,k}) / sqrt(d_a) ),   x^AS_t = sum_k a_{t,k} P_{t,k}
    c~_t   = c_t + x^AS_t

``W_q`` starts at zero (§3.2.4), so at initialisation every selected patch weighs ``1/K`` and
SA² is STPrompt's "selection + average" variant; ``W_k`` is Xavier-random. ``x^AS_t`` is added
at full weight and stays in CLIP's embedding space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from core import constants


@dataclass(frozen=True)
class SA2Output:
    """``c~ (B, L, D)`` and the attention ``a (B, L, K)`` over the selected patches."""

    features: Tensor
    attn: Tensor


class SpatialAttentionAggregation(nn.Module):
    """CLS-queried attention over the ``K`` motion-selected patch features of each step."""

    def __init__(
        self,
        feature_dim: int = constants.STP_PATCH_DIM,
        top_k: int = constants.STP_TOP_K,
        attn_dim: int = constants.STP_ATTN_DIM,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.attn_dim = attn_dim
        self.w_q = nn.Linear(feature_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(feature_dim, attn_dim, bias=False)
        nn.init.zeros_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)

    @property
    def width(self) -> int:
        """Columns the selected patches occupy at the end of an input row."""
        return self.top_k * self.feature_dim

    def forward(
        self,
        cls: Tensor,
        patches: Tensor,
        mask: Tensor | None = None,
        add_cls: bool = True,
    ) -> SA2Output:
        """``cls (B, L, D)``, ``patches (B, L, K * D)`` -> ``c~`` and the attention.

        ``mask`` (``(B, L)``, 1 = valid) zeroes ``x^AS`` on padded steps, so a padded row stays
        exactly what the host model sees there. ``add_cls=False`` returns ``x^AS`` alone (the
        local branch, ``LOCAL_STEP1.md`` §2): ``c_t`` is then only the query, never an input.
        """
        if patches.shape[-1] != self.width:
            raise ValueError(
                f"patch columns {patches.shape[-1]} != top_k {self.top_k} x {self.feature_dim}"
            )
        p = patches.reshape(*patches.shape[:-1], self.top_k, self.feature_dim)
        query = self.w_q(cls).unsqueeze(-2)  # (B, L, 1, d_a)
        keys = self.w_k(p)  # (B, L, K, d_a)
        logits = (query * keys).sum(-1) / math.sqrt(self.attn_dim)
        attn = logits.softmax(dim=-1)
        x_as = (attn.unsqueeze(-1) * p).sum(-2)
        if mask is not None:
            x_as = x_as.masked_fill(~mask.bool()[..., None], 0.0)
        return SA2Output(cls + x_as if add_cls else x_as, attn)

    def query_norm(self) -> Tensor:
        """``||W_q||_F`` (detached): 0 until the attention leaves its uniform start."""
        norm: Tensor = self.w_q.weight.detach().norm()
        return norm


def attention_entropy(attn: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean entropy of ``attn (B, L, K)`` over valid steps, divided by ``log K`` (1 = uniform)."""
    with torch.no_grad():
        ent = -(attn * attn.clamp_min(constants.STP_NORM_EPS).log()).sum(-1)
        if mask is not None:
            ent = ent[mask.bool()]
        return ent.mean() / math.log(attn.shape[-1])


__all__ = ["SA2Output", "SpatialAttentionAggregation", "attention_entropy"]
