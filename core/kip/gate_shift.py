"""Kinematic gate + adaptive temporal channel shift — v3 Phases 3c/3d.

The shift (3d) is RefineVAD MoTAR Eq. 3 and is **unchanged** between v1 and v3:
``s_t`` channels from the past position, ``s_t`` from the future, the rest from
the present. ``s_t <= D/K = 128`` so ``2*s_t <= 256 < 512`` and the three slices
are always disjoint.

What v3 changes is only *how* ``s_t`` is produced. Four gate types are
selectable so that every cross-gate comparison shares one implementation of the
floor/clamp/mask semantics (:func:`core.kip.ecmr.shift_counts_from_ratio`)
rather than four drifting copies:

- ``rank`` — **v3 default.** Ego-compensated motion residual → within-clip rank
  map → ``s_t``. Parameter-free, deterministic, seed-independent, spans the full
  ``[0, 128]`` on every clip.
- ``mlp_frozen`` — **v1, bit-identical.** Min-max normalized intensity through a
  randomly-initialized MLP that never receives gradient. Kept so every number in
  ``core/docs/RESULTS_*.md`` stays reproducible. See the warning below.
- ``mlp_ste`` — ``mlp_frozen``'s forward pass exactly, with the backward pass
  unblocked (spec §4.2 variant). Ablation only.
- ``constant`` — fixed ratio, ``ê_O`` ignored entirely. The plain-TSM control
  (spec §10.3 ablation 4), which has never been run.

**Measured warning about ``mlp_frozen``.** The gate MLP never trains, so its
random initialization *is* the deployed function. Its input is min-max
normalized, so ``[0, 1]`` is the whole reachable input domain — and sweeping that
domain moves ``s_t`` by 0-4 channels out of 128 (8 seeds; seed 0 gives exactly
0). ``mlp_frozen`` is therefore a **near-constant smoother at s ≈ 58-69**, not an
adaptive gate. Do not describe it as motion-gated. See ``core/docs/v3/`` and the
plan's Appendix C.

**Gradient note.** Under ``rank``, ``mlp_frozen`` and ``constant`` no gradient
reaches ``ê_O`` or the PMG head through the shift: ``s_t`` is a hard integer used
as a slice index. This was already true in v1 (report §12.5) and the rank gate
does not change it — it removes 321 already-dead parameters. PMG is trained by
``L_KIP_rec``, ``L_KIP_align`` and ``L_kin``-via-``mhead``.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from core import constants
from core.kip.ecmr import ego_compensated_residual, rank_map, shift_counts_from_ratio

LOGGER = logging.getLogger(__name__)

GATE_SIGNAL_FLOW_NORM = "flow_norm"
GATE_SIGNAL_FEAT_VAR = "feat_var"
GATE_SIGNALS = (GATE_SIGNAL_FLOW_NORM, GATE_SIGNAL_FEAT_VAR)

GATE_TYPE_RANK = "rank"
GATE_TYPE_MLP_FROZEN = "mlp_frozen"
GATE_TYPE_MLP_STE = "mlp_ste"
GATE_TYPE_CONSTANT = "constant"
GATE_TYPES = (GATE_TYPE_RANK, GATE_TYPE_MLP_FROZEN, GATE_TYPE_MLP_STE, GATE_TYPE_CONSTANT)
MLP_GATE_TYPES = (GATE_TYPE_MLP_FROZEN, GATE_TYPE_MLP_STE)

MINMAX_EPS = 1e-6


def shift_channels_reference(vt: Tensor, shift_counts: Tensor) -> Tensor:
    """Oracle per-``(b, t)`` loop implementation of the shift.

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


def shift_channels_straight_through(vt: Tensor, shift_float: Tensor) -> Tensor:
    """Shift with a straight-through estimator — forward exact, backward live.

    ``shift_float (B, L)`` is the *unrounded* channel count ``u_t = r_t * D/K``.
    The forward pass reproduces :func:`shift_channels_vectorized` on
    ``floor(u_t)`` bit-for-bit; the backward pass sees smooth sigmoid boundaries,
    so a gradient reaches ``u_t`` and through it the gate MLP and PMG head.

    Why this exists: the spec writes the STE as
    ``s_t = u_t + (floor(u_t) - u_t).detach()``, which does make ``s_t`` carry a
    gradient — but ``s_t`` is then consumed only inside ``channel < s``
    comparisons, and a comparison emits a boolean that no gradient flows through.
    The formula as written is a no-op. Applying the straight-through estimator to
    the channel *selection weights* instead is what actually unblocks the
    backward pass while keeping the forward pass identical.
    """
    _, _, dim = vt.shape
    past = torch.zeros_like(vt)
    fut = torch.zeros_like(vt)
    past[:, 1:, :] = vt[:, :-1, :]
    fut[:, :-1, :] = vt[:, 1:, :]

    channel = torch.arange(dim, device=vt.device, dtype=vt.dtype).view(1, 1, dim)
    u = shift_float.unsqueeze(-1)  # (B, L, 1)
    s_hard = u.detach().floor()

    # Hard partition (exactly what shift_channels_vectorized computes).
    hard_past = (channel < s_hard).to(vt.dtype)
    hard_upto2 = (channel < 2.0 * s_hard).to(vt.dtype)
    # Smooth surrogate with the same partition-of-unity structure.
    soft_past = torch.sigmoid(u - channel)
    soft_upto2 = torch.sigmoid(2.0 * u - channel)

    w_past = hard_past + (soft_past - soft_past.detach())
    w_upto2 = hard_upto2 + (soft_upto2 - soft_upto2.detach())
    vk: Tensor = w_past * past + (w_upto2 - w_past) * fut + (1.0 - w_upto2) * vt
    return vk


class KinematicShift(nn.Module):
    """Recalibrate ``v^t`` by shifting up to ``D/K`` channels per direction.

    ``gate_type`` selects how the per-position shift count is produced; see the
    module docstring. Only the ``mlp_*`` types own parameters (321); ``rank`` and
    ``constant`` are parameter-free.
    """

    def __init__(
        self,
        d: int = constants.HIDDEN_DIM,
        folding_factor: int = constants.FOLDING_FACTOR,
        gate_type: str = GATE_TYPE_RANK,
        gate_signal: str | None = None,
        gate_hidden: int = constants.GATE_MLP_HIDDEN_DIM,
        ecmr_lambda: float = constants.ECMR_EMA_LAMBDA,
        const_shift_ratio: float = constants.CONST_SHIFT_RATIO,
    ) -> None:
        super().__init__()
        if gate_type not in GATE_TYPES:
            raise ValueError(f"Unknown gate_type: {gate_type!r}; expected one of {GATE_TYPES}")
        if gate_type in MLP_GATE_TYPES:
            if gate_signal is None:
                gate_signal = GATE_SIGNAL_FLOW_NORM
            if gate_signal not in GATE_SIGNALS:
                raise ValueError(f"Unknown gate_signal: {gate_signal!r}")
        elif gate_signal is not None:
            raise ValueError(
                f"gate_signal={gate_signal!r} is meaningless for gate_type={gate_type!r}: "
                f"only {MLP_GATE_TYPES} consume an intensity signal. Leave it unset."
            )
        if not 0.0 <= const_shift_ratio <= 1.0:
            raise ValueError(f"const_shift_ratio must be in [0, 1], got {const_shift_ratio!r}")

        self.d = d
        self.max_shift = d // folding_factor
        self.gate_type = gate_type
        self.gate_signal = gate_signal
        self.ecmr_lambda = ecmr_lambda
        self.const_shift_ratio = const_shift_ratio
        # Parameters exist only for the MLP variants; rank/constant are 0-param.
        self.mlp: nn.Sequential | None = (
            nn.Sequential(
                nn.Linear(1, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            if gate_type in MLP_GATE_TYPES
            else None
        )

    def _intensity(self, vt: Tensor, eo: Tensor) -> Tensor:
        """Kinematic intensity ``m (B, L)`` for the MLP gates (v1 behaviour)."""
        if self.gate_signal == GATE_SIGNAL_FLOW_NORM:
            norm: Tensor = eo.norm(dim=-1)
            return norm
        return vt.var(dim=-1, unbiased=False)

    def _mlp_ratio(self, vt: Tensor, eo: Tensor, mask: Tensor | None) -> Tensor:
        """v1 gate ratio: intensity → mask-aware min-max → sigmoid(MLP(·))."""
        if self.mlp is None:  # unreachable: guarded by gate_type in __init__
            raise RuntimeError(f"gate_type={self.gate_type!r} has no gate MLP")
        m = self._intensity(vt, eo)  # (B, L)
        if mask is not None:
            valid = mask.bool()
            m_min = m.masked_fill(~valid, float("inf")).amin(dim=1, keepdim=True)
            m_max = m.masked_fill(~valid, float("-inf")).amax(dim=1, keepdim=True)
        else:
            m_min = m.amin(dim=1, keepdim=True)
            m_max = m.amax(dim=1, keepdim=True)
        m_hat = (m - m_min) / (m_max - m_min + MINMAX_EPS)
        ratio: Tensor = torch.sigmoid(self.mlp(m_hat.unsqueeze(-1))).squeeze(-1)
        return ratio

    def gate_ratio(self, vt: Tensor, eo: Tensor, mask: Tensor | None = None) -> Tensor:
        """Per-position shift ratio ``r (B, L)`` in ``[0, 1]``, before the floor."""
        if self.gate_type == GATE_TYPE_RANK:
            m, _ = ego_compensated_residual(eo, mask, self.ecmr_lambda)
            return rank_map(m, mask)
        if self.gate_type == GATE_TYPE_CONSTANT:
            return torch.full_like(eo[..., 0], self.const_shift_ratio)
        return self._mlp_ratio(vt, eo, mask)

    def compute_shift_counts(self, vt: Tensor, eo: Tensor, mask: Tensor | None = None) -> Tensor:
        """Integer shift counts ``s (B, L)``; padded positions never shift."""
        return shift_counts_from_ratio(self.gate_ratio(vt, eo, mask), self.max_shift, mask)

    def diagnostics(self, vt: Tensor, eo: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
        """Detached gate internals for logging (v3 §9: "also log ``s_t``")."""
        if self.gate_type == GATE_TYPE_RANK:
            m, mu_norm = ego_compensated_residual(eo, mask, self.ecmr_lambda)
        else:
            m = self._intensity(vt, eo) if self.gate_type in MLP_GATE_TYPES else eo.norm(dim=-1)
            mu_norm = torch.zeros_like(m)
        ratio = self.gate_ratio(vt, eo, mask)
        counts = shift_counts_from_ratio(ratio, self.max_shift, mask)
        return {
            "s": counts.detach(),
            "gate_ratio": ratio.detach(),
            "m": m.detach(),
            "mu_norm": mu_norm.detach(),
            "eo_norm": eo.norm(dim=-1).detach(),
        }

    def forward(self, vt: Tensor, eo: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return ``v^k (B, L, D)``; padded positions zeroed, never pulled from."""
        if mask is not None:
            vt = vt * mask.unsqueeze(-1)  # padded neighbors contribute zeros
        if self.gate_type == GATE_TYPE_MLP_STE:
            ratio = self.gate_ratio(vt, eo, mask)
            shift_float = (ratio * self.max_shift).clamp(0.0, float(self.max_shift))
            if mask is not None:
                shift_float = shift_float * mask
            vk = shift_channels_straight_through(vt, shift_float)
        else:
            vk = shift_channels_vectorized(vt, self.compute_shift_counts(vt, eo, mask))
        if mask is not None:
            vk = vk * mask.unsqueeze(-1)
        return vk


__all__ = [
    "GATE_SIGNALS",
    "GATE_SIGNAL_FEAT_VAR",
    "GATE_SIGNAL_FLOW_NORM",
    "GATE_TYPES",
    "GATE_TYPE_CONSTANT",
    "GATE_TYPE_MLP_FROZEN",
    "GATE_TYPE_MLP_STE",
    "GATE_TYPE_RANK",
    "MLP_GATE_TYPES",
    "KinematicShift",
    "shift_channels_reference",
    "shift_channels_straight_through",
    "shift_channels_vectorized",
]
