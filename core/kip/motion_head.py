"""Motion-only score head — spec §4.

Reads the pseudo-flow embedding alone and produces the per-frame anomaly curve
``ŷ_O`` consumed only by ``L_kin``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from core import constants


class MotionScoreHead(nn.Module):
    """Linear 256→128 → GELU → Linear 128→1 → sigmoid; ``(B, L, d_flow)`` → ``(B, L)``."""

    def __init__(
        self,
        d_flow: int = constants.FLOW_DIM,
        d_hidden: int = constants.MOTION_HEAD_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_flow, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, eo: Tensor, mask: Tensor | None = None) -> Tensor:
        """Score ``ê_O (B, L, d_flow)`` → ``ŷ_O (B, L)`` in [0, 1].

        Padded positions are forced to 0 so downstream top-k never selects them.
        """
        scores = torch.sigmoid(self.net(eo)).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores * mask
        return scores


__all__ = ["MotionScoreHead"]
