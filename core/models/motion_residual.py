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

Arm K5ln (``core/docs/final/KIM_RUN.md`` §25) tests that choice on the CRN-off host:
with ``motion_norm="layernorm"`` the motion columns get a per-step LayerNorm without
affine (``W_u`` absorbs any gain and shift) and are rescaled by the cache's ``c``, so
their per-channel RMS stays at the baked stream's and ``W_u``'s effective step size
matches K5's. ``c`` is a buffer set once from the input manifest
(:meth:`MotionResidual.set_norm_scale`) and travels in the checkpoint.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from core import constants


class MotionResidual(nn.Module):
    """Split ``[x ; u]`` rows and return ``x + W_u u`` plus the motion share."""

    def __init__(
        self,
        hidden_dim: int,
        motion_dim: int,
        init_std: float = 0.0,
        init_seed: int = 0,
        norm: str = constants.MOTION_NORM_OFF,
    ) -> None:
        super().__init__()
        if motion_dim <= 0:
            raise ValueError(f"motion_dim must be positive, got {motion_dim}")
        if init_std < 0:
            raise ValueError(f"init_std must be >= 0, got {init_std}")
        if norm not in constants.MOTION_NORM_CHOICES:
            raise ValueError(f"norm must be one of {constants.MOTION_NORM_CHOICES}, got {norm!r}")
        self.hidden_dim = hidden_dim
        self.motion_dim = motion_dim
        self.norm = norm
        if norm != constants.MOTION_NORM_OFF:
            # 0 = not set: forward refuses to run rather than train at an unmatched scale.
            # Registered only for the LN arm, so every earlier checkpoint loads unchanged (C5).
            self.register_buffer("norm_scale", torch.zeros(()))
        self.proj = nn.Linear(motion_dim, hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if init_std > 0:
            # a private generator: the global RNG stream (and so every other module's init
            # and the data order) stays identical to the zero-init control (WU_ARMS.md)
            generator = torch.Generator().manual_seed(init_seed)
            with torch.no_grad():
                self.proj.weight.copy_(
                    torch.randn(self.proj.weight.shape, generator=generator) * init_std
                )

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
        motion = self.proj(self.normalize(u))
        valid = torch.ones(rows.shape[:-1], dtype=torch.bool, device=rows.device)
        if mask is not None:
            valid = mask.bool()
            # padded rows stay exactly as the no-motion model sees them (no bias leak)
            motion = motion.masked_fill(~valid[..., None], 0.0)
        with torch.no_grad():
            clip_norm = x.detach()[valid].norm().clamp_min(constants.V2_NORM_EPS)
            rho = motion.detach()[valid].norm() / clip_norm
        return x + motion, rho

    def set_norm_scale(self, c: float) -> None:
        """Fix the LN arm's rescale to the input cache's ``c`` (its manifest's ``"c"``)."""
        if self.norm == constants.MOTION_NORM_OFF:
            raise ValueError("set_norm_scale is only defined for motion_norm != 'none'")
        if not c > 0:
            raise ValueError(f"norm scale c must be > 0, got {c}")
        self.norm_scale.fill_(float(c))

    def normalize(self, u: Tensor) -> Tensor:
        """``u`` unchanged (norm off) or ``c * LayerNorm(u)`` per step, no affine."""
        if self.norm == constants.MOTION_NORM_OFF:
            return u
        scale: Tensor = self.norm_scale
        if not bool(scale > 0):
            raise RuntimeError(
                "motion_norm=layernorm but norm_scale was never set: call set_norm_scale(c) "
                "with the input cache's c, or load a checkpoint that carries it"
            )
        normed = functional.layer_norm(u, (self.motion_dim,), eps=constants.MOTION_NORM_EPS)
        return scale * normed

    def weight_norm(self) -> Tensor:
        """``||W_u||_F`` (detached), logged beside ``rho_u``."""
        norm: Tensor = self.proj.weight.detach().norm()
        return norm


__all__ = ["MotionResidual"]
