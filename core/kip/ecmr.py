"""ECMR + rank map — KAT-VAD v3 architecture Phases 3b and 3c.

Two parameter-free stages that sit between the PMG head (3a) and the adaptive
temporal shift (3d), replacing v1's frozen randomly-initialized gate MLP:

- :func:`ego_compensated_residual` — 3b. Causal EMA prototype of the clip's own
  dominant motion, subtracted from the pseudo-flow embedding. Converts a global,
  uninformative flow magnitude into a local, differential one.
- :func:`rank_map` — 3c. Within-clip ordinal rank of that residual, normalized to
  ``[0, 1]``. Scale-free by construction, so a dashcam at highway speed and a
  static parking-lot camera get the same shift schedule shape.
- :func:`shift_counts_from_ratio` — the shared ``⌊r·D/K⌋`` step. Every gate type
  routes through this one function so the floor/clamp/mask semantics cannot drift
  apart between arms (see plan T1.3).

**Zero parameters.** Nothing here is learned; these are deterministic functions
of ``ê_O``. Under ``gate_type="rank"`` this makes KIP's score-path parameter
count exactly 0 — the entire inference-time effect of the pathway is a
deterministic reindexing driven by a reconstruction-trained flow estimator.

**Gradient note (unchanged from v1, not introduced here).**
:func:`shift_counts_from_ratio` ends in ``floor().long()`` and its output is used
as a slice index, so no gradient reaches ``ê_O`` or the PMG head through the
shift. That was already true of v1's MLP gate — see ``core/docs/v3/`` and the
report §12.5 finding that all 6 gate tensors had ``grad is None``. PMG is trained
by ``L_KIP_rec``, ``L_KIP_align`` and ``L_kin``-via-``mhead``, and by nothing
else. Stage-1 convergence is therefore a precondition, not a nicety.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from core import constants

LOGGER = logging.getLogger(__name__)


def _as_bool_mask(mask: Tensor | None, batch: int, length: int, device: torch.device) -> Tensor:
    """Normalize an optional float/bool mask to a bool ``(B, L)`` tensor."""
    if mask is None:
        return torch.ones(batch, length, dtype=torch.bool, device=device)
    return mask.bool()


def ego_compensated_residual(
    eo: Tensor,
    mask: Tensor | None = None,
    lam: float = constants.ECMR_EMA_LAMBDA,
) -> tuple[Tensor, Tensor]:
    """Phase 3b — ego-compensated motion residual.

    ``mu_t = lam * mu_{t-1} + (1 - lam) * eo_t`` (the recurrence form of the
    spec's ``mu_t = sum_{tau<=t} (1-lam) lam^(t-tau) eo_tau``), then
    ``m_t = ||eo_t - mu_t||_2``.

    Args:
        eo: pseudo-flow embedding ``ê_O (B, L, d_flow)``.
        mask: ``(B, L)``, 1 = valid. Padded positions **hold** the EMA state
            rather than updating it with zeros, so a prototype never decays
            toward zero inside padding.
        lam: EMA decay. Higher = longer memory of the clip's dominant motion.

    Returns:
        ``(m, mu_norm)``, both ``(B, L)`` and both zeroed at padded positions.
        ``mu_norm`` is ``||mu_t||_2``, carried for Phase 4 diagnostics only.

    Notes:
        The first valid position seeds the prototype with the sample itself
        (``mu = eo``), giving ``m = 0`` there. Seeding at zero instead would make
        frame 0 the loudest position of every clip, which the rank map would then
        hand the maximum shift — a boundary position whose past neighbour is
        zero-padded. That is a bias, not noise.

        Because the current sample is included in ``mu_t``, the residual equals
        ``lam * (eo_t - mu_{t-1})`` exactly. The constant factor is immaterial:
        :func:`rank_map` consumes only the ordering.

        Implemented as a sequential recurrence rather than the closed form. The
        closed form needs ``lam^-tau``, which overflows float32 near ``t = 835``
        for ``lam = 0.9`` while ``lam^t`` underflows to zero at the same point,
        and ``L`` may reach ``constants.TEMPORAL_MAX_POSITIONS``.
    """
    if not 0.0 <= lam < 1.0:
        raise ValueError(f"ecmr lam must be in [0, 1), got {lam!r}")
    batch, length, _ = eo.shape
    valid = _as_bool_mask(mask, batch, length, eo.device)

    mu = torch.zeros_like(eo[:, 0])  # (B, d_flow)
    started = torch.zeros(batch, 1, dtype=torch.bool, device=eo.device)
    prototypes: list[Tensor] = []
    for t in range(length):
        sample = eo[:, t]  # (B, d_flow)
        is_valid = valid[:, t : t + 1]  # (B, 1)
        ema = lam * mu + (1.0 - lam) * sample
        # Before the first valid position the prototype is the sample itself.
        candidate = torch.where(started, ema, sample)
        mu = torch.where(is_valid, candidate, mu)  # hold state through padding
        started = started | is_valid
        prototypes.append(mu)

    mu_seq = torch.stack(prototypes, dim=1)  # (B, L, d_flow)
    residual = eo - mu_seq
    mask_f = valid.to(eo.dtype)
    m = residual.norm(dim=-1) * mask_f
    mu_norm = mu_seq.norm(dim=-1) * mask_f
    return m, mu_norm


def rank_map(m: Tensor, mask: Tensor | None = None) -> Tensor:
    """Phase 3c — within-clip ordinal rank of ``m``, normalized to ``[0, 1]``.

    ``r_t = rank_t(m) / (n_valid - 1)`` over the **valid** positions only. Using
    ``n_valid`` rather than ``L`` matters: with ``L`` the rank range would be
    compressed by the padding fraction, so a clip's own scores would shift with
    whatever else happened to share its batch.

    Args:
        m: residual magnitude ``(B, L)`` from :func:`ego_compensated_residual`.
        mask: ``(B, L)``, 1 = valid. Padded positions rank last and return 0.

    Returns:
        ``r (B, L)`` in ``[0, 1]``, zeroed at padded positions.

    Notes:
        Ties break by index order (stable double ``argsort``), which is
        deterministic but arbitrary. On a clip whose residual is effectively
        constant the map still spans ``[0, 1]``, so the shift schedule is
        arbitrary rather than absent; a warning is logged when that happens.

        Spanning the full range on every clip with ``n_valid >= 2`` is the
        property that rules out H4' by construction: a constant smoother is not
        expressible by this gate.
    """
    batch, length = m.shape
    valid = _as_bool_mask(mask, batch, length, m.device)

    # Padded positions sort last, so valid positions occupy ranks 0..n_valid-1.
    ordered = m.masked_fill(~valid, float("inf"))
    order = ordered.argsort(dim=1, stable=True)
    ranks = order.argsort(dim=1, stable=True).to(m.dtype)  # (B, L)

    n_valid = valid.sum(dim=1, keepdim=True).to(m.dtype)  # (B, 1)
    denom = (n_valid - 1.0).clamp(min=1.0)  # n_valid <= 1 -> r = 0 below
    r = ranks / denom
    r = torch.where(n_valid > 1.0, r, torch.zeros_like(r))
    r = r * valid.to(m.dtype)

    if bool(valid.any()):
        spread = torch.where(valid, m, torch.zeros_like(m)).std(dim=1)
        if bool((spread < constants.RANK_TIE_EPS).any()):
            LOGGER.warning(
                "ECMR residual is effectively constant on %d clip(s) "
                "(std < %g); the rank map still spans [0, 1], so their shift "
                "schedule is deterministic but arbitrary.",
                int((spread < constants.RANK_TIE_EPS).sum().item()),
                constants.RANK_TIE_EPS,
            )
    return r


def shift_counts_from_ratio(
    ratio: Tensor,
    max_shift: int,
    mask: Tensor | None = None,
) -> Tensor:
    """``s_t = floor(r_t * max_shift)``, clamped and mask-zeroed — ``(B, L)`` long.

    Shared by every gate type so the floor/clamp/padding semantics stay identical
    across arms; a per-gate reimplementation would put an implementation confound
    inside every cross-gate comparison.

    Args:
        ratio: ``r (B, L)``, expected in ``[0, 1]`` but clamped defensively.
        max_shift: ``D // K``; channels shiftable per direction.
        mask: ``(B, L)``, 1 = valid. Padded positions never shift.
    """
    counts = (ratio * max_shift).floor().long().clamp(0, max_shift)
    if mask is not None:
        counts = counts * mask.long()
    return counts


__all__ = ["ego_compensated_residual", "rank_map", "shift_counts_from_ratio"]
