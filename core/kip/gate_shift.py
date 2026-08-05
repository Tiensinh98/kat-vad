"""Kinematic gate + adaptive temporal channel shift — spec §3.

Derived from RefineVAD MoTAR steps 2-4 with the gate driven by the pseudo-flow
norm (``gate_signal="flow_norm"``, ours) or feature variance
(``"feat_var"``, RefineVAD original — spec §10 ablation 5).

Two shift implementations with identical semantics (plan Phase 1, A13/R6):

- :func:`shift_channels_reference` — the spec's per-``(b, t)`` loop, the oracle.
- :func:`shift_channels_vectorized` — broadcast channel-index masks + nested
  ``torch.where``; used by :meth:`KinematicShift.forward`.

Note on gradients (spec-as-written): the channel count ``s_t`` comes from a hard
``floor`` and integer slicing, so no gradient reaches the gate MLP — its weights
stay at initialization and the gate is input-adaptive but untrained. Kept
deliberately faithful to the spec; revisit only with an explicit ablation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from core import constants

GATE_SIGNAL_FLOW_NORM = "flow_norm"
GATE_SIGNAL_FEAT_VAR = "feat_var"
MINMAX_EPS = 1e-6


def shift_channels_reference(vt: Tensor, shift_counts: Tensor) -> Tensor:
    """Oracle per-``(b, t)`` loop implementation of spec §3 step 5.

    ``vt (B, L, D)``, ``shift_counts (B, L)`` int — channels per direction.
    """
    batch, length, dim = vt.shape
    past = torch.zeros_like(vt)
    fut = torch.zeros_like(vt)
    past[:, 1:, :] = vt[:, :-1, :]
    fut[:, :-1, :] = vt[:, 1:, :]
    vk = vt.clone()
    for b in range(batch):
        for t in range(length):
            st = int(shift_counts[b, t].item())
            st2 = min(2 * st, dim)
            if st > 0:
                vk[b, t, :st] = past[b, t, :st]
            if st2 > st:
                vk[b, t, st:st2] = fut[b, t, st:st2]
    return vk


def shift_channels_vectorized(vt: Tensor, shift_counts: Tensor) -> Tensor:
    """Vectorized equivalent of :func:`shift_channels_reference`."""
    _, _, dim = vt.shape
    past = torch.zeros_like(vt)
    fut = torch.zeros_like(vt)
    past[:, 1:, :] = vt[:, :-1, :]
    fut[:, :-1, :] = vt[:, 1:, :]
    channel = torch.arange(dim, device=vt.device).view(1, 1, dim)  # (1, 1, D)
    s = shift_counts.unsqueeze(-1)  # (B, L, 1)
    return torch.where(channel < s, past, torch.where(channel < 2 * s, fut, vt))


class KinematicShift(nn.Module):
    """Recalibrate ``v^t`` by shifting up to ``D/K`` channels per direction.

    Shift counts are gated per time step by the kinematic intensity of the
    pseudo-flow ``ê_O`` (or feature variance under ablation 5).
    """

    def __init__(
        self,
        d: int = constants.HIDDEN_DIM,
        folding_factor: int = constants.FOLDING_FACTOR,
        gate_signal: str = GATE_SIGNAL_FLOW_NORM,
        gate_hidden: int = constants.GATE_MLP_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if gate_signal not in (GATE_SIGNAL_FLOW_NORM, GATE_SIGNAL_FEAT_VAR):
            raise ValueError(f"Unknown gate_signal: {gate_signal!r}")
        self.d = d
        self.max_shift = d // folding_factor
        self.gate_signal = gate_signal
        self.mlp = nn.Sequential(
            nn.Linear(1, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

    def _intensity(self, vt: Tensor, eo: Tensor) -> Tensor:
        """Kinematic intensity ``m (B, L)`` — spec §3 step 1 (or ablation 5)."""
        if self.gate_signal == GATE_SIGNAL_FLOW_NORM:
            norm: Tensor = eo.norm(dim=-1)
            return norm
        return vt.var(dim=-1, unbiased=False)

    def compute_shift_counts(self, vt: Tensor, eo: Tensor, mask: Tensor | None = None) -> Tensor:
        """Steps 1-4: intensity → mask-aware min-max → ratio MLP → ``s (B, L)`` int."""
        m = self._intensity(vt, eo)  # (B, L)
        if mask is not None:
            valid = mask.bool()
            m_min = m.masked_fill(~valid, float("inf")).amin(dim=1, keepdim=True)
            m_max = m.masked_fill(~valid, float("-inf")).amax(dim=1, keepdim=True)
        else:
            m_min = m.amin(dim=1, keepdim=True)
            m_max = m.amax(dim=1, keepdim=True)
        m_hat = (m - m_min) / (m_max - m_min + MINMAX_EPS)
        ratio = torch.sigmoid(self.mlp(m_hat.unsqueeze(-1))).squeeze(-1)  # (B, L) in [0, 1]
        counts = (ratio * self.max_shift).floor().long()
        if mask is not None:
            counts = counts * mask.long()
        return counts

    def forward(self, vt: Tensor, eo: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return ``v^k (B, L, D)``; padded positions zeroed, never pulled from."""
        if mask is not None:
            vt = vt * mask.unsqueeze(-1)  # padded neighbors contribute zeros
        counts = self.compute_shift_counts(vt, eo, mask)
        vk = shift_channels_vectorized(vt, counts)
        if mask is not None:
            vk = vk * mask.unsqueeze(-1)
        return vk


__all__ = [
    "GATE_SIGNAL_FEAT_VAR",
    "GATE_SIGNAL_FLOW_NORM",
    "KinematicShift",
    "shift_channels_reference",
    "shift_channels_vectorized",
]
