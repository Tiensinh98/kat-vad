"""KIP — Kinematic Induction Pathway wrapper — spec §5.1.

Wires PMG → KinematicShift → MotionScoreHead and owns the ``L_KIP_align``
projections (``proj_flow`` 256→128, ``proj_rgb`` 512→128, spec §5.2b).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from torch import Tensor, nn

from core import constants
from core.kip.gate_shift import GATE_TYPE_RANK, KinematicShift
from core.kip.motion_head import MotionScoreHead
from core.kip.pmg import PMGFlowHead

if TYPE_CHECKING:
    from core.config import KIPConfig


class KIPOutput(NamedTuple):
    """What :meth:`KIP.forward` returns.

    ``motion_scores`` is ``None`` on the inference graph (3e not instantiated).
    ``diagnostics`` is ``None`` unless explicitly requested; it carries detached
    gate internals (``s``, ``gate_ratio``, ``m``, ``mu_norm``, ``eo_norm``), which
    v3 §9 asks to be logged at inference and which ablation-ladder rows 1-2 read
    off existing checkpoints with no training.
    """

    vk: Tensor
    eo_hat: Tensor
    motion_scores: Tensor | None
    diagnostics: dict[str, Tensor] | None = None


class KIP(nn.Module):
    """``v^t (B, L, D)`` → ``(v^k (B, L, D), ê_O (B, L, d_flow), ŷ_O (B, L) | None)``.

    With ``use_gate_shift=False`` (spec §10 ablation 2) the shift is skipped and
    ``v^k = v^t`` (masked); PMG and the motion head still run.

    With ``train_only_modules=False`` the v3 inference graph is built: the motion
    head (3e) and the alignment projections (3f) are **not instantiated**, and
    ``forward`` returns ``ŷ_O = None``. Nothing at inference consumes either.
    See :func:`core.models.ckpt_compat.load_kip_state_dict` for loading a
    training checkpoint into this graph without weakening strict loading.
    """

    def __init__(
        self,
        d: int = constants.HIDDEN_DIM,
        d_flow: int = constants.FLOW_DIM,
        pmg_latent_dim: int = constants.PMG_LATENT_DIM,
        folding_factor: int = constants.FOLDING_FACTOR,
        align_proj_dim: int = constants.ALIGN_PROJ_DIM,
        gate_type: str = GATE_TYPE_RANK,
        gate_signal: str | None = None,
        ecmr_lambda: float = constants.ECMR_EMA_LAMBDA,
        const_shift_ratio: float = constants.CONST_SHIFT_RATIO,
        use_gate_shift: bool = True,
        train_only_modules: bool = True,
    ) -> None:
        super().__init__()
        self.pmg = PMGFlowHead(d_in=d, d_lat=pmg_latent_dim, d_flow=d_flow)
        self.shift: KinematicShift | None = (
            KinematicShift(
                d=d,
                folding_factor=folding_factor,
                gate_type=gate_type,
                gate_signal=gate_signal,
                ecmr_lambda=ecmr_lambda,
                const_shift_ratio=const_shift_ratio,
            )
            if use_gate_shift
            else None
        )
        # v3 Phases 3e/3f: train-time only, off the inference graph entirely.
        self.train_only_modules = train_only_modules
        self.mhead: MotionScoreHead | None = (
            MotionScoreHead(d_flow=d_flow) if train_only_modules else None
        )
        self.proj_flow: nn.Linear | None = (
            nn.Linear(d_flow, align_proj_dim) if train_only_modules else None
        )
        self.proj_rgb: nn.Linear | None = (
            nn.Linear(d, align_proj_dim) if train_only_modules else None
        )

    @classmethod
    def from_config(
        cls, cfg: KIPConfig, d: int = constants.HIDDEN_DIM, training: bool = True
    ) -> KIP:
        """Build from :class:`core.config.KIPConfig` (ablation flags applied).

        ``training=False`` builds the v3 inference graph (no ``mhead``, no
        alignment projections).
        """
        return cls(
            d=d,
            d_flow=cfg.d_flow,
            pmg_latent_dim=cfg.pmg_latent_dim,
            folding_factor=cfg.folding_factor,
            align_proj_dim=cfg.align_proj_dim,
            gate_type=cfg.gate_type,
            gate_signal=cfg.gate_signal,
            ecmr_lambda=cfg.ecmr_lambda,
            const_shift_ratio=cfg.const_shift_ratio,
            use_gate_shift=cfg.use_gate_shift and not cfg.pmg_only,
            train_only_modules=training,
        )

    def forward(
        self,
        vt: Tensor,
        mask: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> KIPOutput:
        """Forward pass; padded positions zeroed in every output.

        ``ŷ_O`` is ``None`` on the inference graph (``train_only_modules=False``),
        where the motion head is not instantiated. ``return_diagnostics`` costs a
        second gate evaluation, so it is off on the training path.
        """
        if mask is not None:
            vt = vt * mask.unsqueeze(-1)
        eo = self.pmg(vt)  # (B, L, d_flow)
        vk = self.shift(vt, eo, mask) if self.shift is not None else vt
        yo = self.mhead(eo, mask) if self.mhead is not None else None  # (B, L)
        diag: dict[str, Tensor] | None = None
        if return_diagnostics and self.shift is not None:
            diag = self.shift.diagnostics(vt, eo, mask)
        return KIPOutput(vk=vk, eo_hat=eo, motion_scores=yo, diagnostics=diag)

    def project_for_align(self, eo: Tensor, vt: Tensor) -> tuple[Tensor, Tensor]:
        """Project both modalities to the common ``d_c`` space for ``L_KIP_align``.

        Returns unnormalized ``(proj_flow(ê_O), proj_rgb(v^t))``; the loss handles
        L2 normalization and masking. Raises on the inference graph, where the
        projections do not exist — ``L_KIP_align`` is a training-only objective.
        """
        if self.proj_flow is None or self.proj_rgb is None:
            raise RuntimeError(
                "project_for_align requires the training graph; this KIP was built "
                "with train_only_modules=False (Phases 3e/3f are not instantiated)"
            )
        return self.proj_flow(eo), self.proj_rgb(vt)


__all__ = ["KIP", "KIPOutput"]
