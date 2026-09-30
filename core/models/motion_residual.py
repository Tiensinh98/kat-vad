"""v2 motion-stream fusion: the only new trainable block (architecture §5).

The input rows carry the CLIP stream (``hidden_dim`` columns, CRN-normalized or
raw) followed by the pre-scaled motion stream ``c * u~ / sigma_u`` (``motion_dim``
columns); both transforms are fixed statistics baked into the input cache by
``core.tools.build_v2_inputs``. This block adds ``W_u`` of that motion term onto
the CLIP stream:

    h_t = x_t + W_u (c * u~_t / sigma_u)

``W_u`` (weight and bias) starts at zero, so at initialization ``h == x`` exactly
and the model is the no-motion arm. There is deliberately no LayerNorm: it would
erase the per-step deviation magnitude CRN produces (architecture §5).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from core import constants


class MotionResidual(nn.Module):
    """Split ``[x ; u]`` rows and return ``x + W_u u`` plus the motion share."""

    def __init__(self, hidden_dim: int, motion_dim: int) -> None:
        super().__init__()
        if motion_dim <= 0:
            raise ValueError(f"motion_dim must be positive, got {motion_dim}")
        self.hidden_dim = hidden_dim
        self.motion_dim = motion_dim
        self.proj = nn.Linear(motion_dim, hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, rows: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``rows (B, L, hidden+motion)`` -> ``h (B, L, hidden)`` and ``rho_u`` (scalar).

        ``mask`` follows :func:`core.data.collate.padding_mask`: ``(B, L)``, 1 for a
        valid step and 0 for padding. ``rho_u = ||W_u u|| / ||x||`` over the valid
        steps, detached: the logged evidence that the stream is used (architecture §5).
        """
        expected = self.hidden_dim + self.motion_dim
        if rows.shape[-1] != expected:
            raise ValueError(
                f"input rows have {rows.shape[-1]} columns; a motion model needs "
                f"{self.hidden_dim} (CLIP) + {self.motion_dim} (motion) = {expected} -- "
                "is the run pointed at the v2 input cache?"
            )
        x, u = rows[..., : self.hidden_dim], rows[..., self.hidden_dim :]
        motion = self.proj(u)
        valid = torch.ones(rows.shape[:-1], dtype=torch.bool, device=rows.device)
        if mask is not None:
            valid = mask.bool()
            # padded rows stay exactly as the no-motion model sees them (no bias leak)
            motion = motion.masked_fill(~valid[..., None], 0.0)
        with torch.no_grad():
            clip_norm = x.detach()[valid].norm().clamp_min(constants.V2_NORM_EPS)
            rho = motion.detach()[valid].norm() / clip_norm
        return x + motion, rho

    def weight_norm(self) -> Tensor:
        """``||W_u||_F`` (detached), logged beside ``rho_u``."""
        norm: Tensor = self.proj.weight.detach().norm()
        return norm


__all__ = ["MotionResidual"]
