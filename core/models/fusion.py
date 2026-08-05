"""Co-attention fusion ``U`` — port of the baseline's ``FusionV1('co_attn')``.

Only the co-attention variant is ported: it is the one the shipped LaGoVAD
checkpoint/config uses. Module names (``layers.N.text_attn`` ...) match the
baseline for 1:1 state-dict mapping.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from core import constants


class CoAttnFusionLayer(nn.Module):
    """One bidirectional cross-attention block over (video, text) features."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = constants.FUSION_DROPOUT,
    ) -> None:
        super().__init__()
        self.text_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)

        self.vis_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.vis_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.vis_norm1 = nn.LayerNorm(d_model)
        self.vis_norm2 = nn.LayerNorm(d_model)

    def forward(
        self, v_feat: Tensor, t_feat: Tensor, attn_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``v (B, T, E)``, ``t (B, C, E)``, ``attn_mask (B, T)`` True = padded."""
        attn_t = self.text_norm1(
            t_feat + self.text_attn(t_feat, v_feat, v_feat, key_padding_mask=attn_mask)[0]
        )
        attn_v = self.vis_norm1(v_feat + self.vis_attn(v_feat, t_feat, t_feat)[0])
        attn_t = self.text_norm2(attn_t + self.text_ffn(attn_t))
        attn_v = self.vis_norm2(attn_v + self.vis_ffn(attn_v))
        return attn_v, attn_t


class CoAttentionFusion(nn.Module):
    """``U(v^k, z^t)`` → ``(v^u, z^u)``: stacked co-attention layers."""

    def __init__(
        self,
        d_model: int = constants.HIDDEN_DIM,
        nhead: int = constants.FUSION_HEADS,
        num_layers: int = constants.FUSION_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CoAttnFusionLayer(d_model, nhead, d_model * 4) for _ in range(num_layers)]
        )

    def forward(
        self, v_feat: Tensor, t_feat: Tensor, lengths: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``v (B, T, E)``, ``t (B, C, E)`` (pre-expanded), ``lengths (B,)``."""
        attn_mask = (
            torch.arange(v_feat.shape[1], device=v_feat.device)[None, :]
            >= lengths[:, None].to(v_feat.device)
        )
        for layer in self.layers:
            v_feat, t_feat = layer(v_feat, t_feat, attn_mask)
        return v_feat, t_feat


__all__ = ["CoAttentionFusion", "CoAttnFusionLayer"]
