"""KIP — Kinematic Induction Pathway wrapper — spec §5.1.

Wires PMG → KinematicShift → MotionScoreHead and owns the ``L_KIP_align``
projections (``proj_flow`` 256→128, ``proj_rgb`` 512→128, spec §5.2b).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from core import constants
from core.kip.gate_shift import GATE_SIGNAL_FLOW_NORM, KinematicShift
from core.kip.motion_head import MotionScoreHead
from core.kip.pmg import PMGFlowHead

if TYPE_CHECKING:
    from core.config import KIPConfig


class KIP(nn.Module):
    """``v^t (B, L, D)`` → ``(v^k (B, L, D), ê_O (B, L, d_flow), ŷ_O (B, L))``.

    With ``use_gate_shift=False`` (spec §10 ablation 2) the shift is skipped and
    ``v^k = v^t`` (masked); PMG and the motion head still run.
    """

    def __init__(
        self,
        d: int = constants.HIDDEN_DIM,
        d_flow: int = constants.FLOW_DIM,
        pmg_latent_dim: int = constants.PMG_LATENT_DIM,
        folding_factor: int = constants.FOLDING_FACTOR,
        align_proj_dim: int = constants.ALIGN_PROJ_DIM,
        gate_signal: str = GATE_SIGNAL_FLOW_NORM,
        use_gate_shift: bool = True,
    ) -> None:
        super().__init__()
        self.pmg = PMGFlowHead(d_in=d, d_lat=pmg_latent_dim, d_flow=d_flow)
        self.shift: KinematicShift | None = (
            KinematicShift(d=d, folding_factor=folding_factor, gate_signal=gate_signal)
            if use_gate_shift
            else None
        )
        self.mhead = MotionScoreHead(d_flow=d_flow)
        self.proj_flow = nn.Linear(d_flow, align_proj_dim)
        self.proj_rgb = nn.Linear(d, align_proj_dim)

    @classmethod
    def from_config(cls, cfg: KIPConfig, d: int = constants.HIDDEN_DIM) -> KIP:
        """Build from :class:`core.config.KIPConfig` (ablation flags applied)."""
        return cls(
            d=d,
            d_flow=cfg.d_flow,
            pmg_latent_dim=cfg.pmg_latent_dim,
            folding_factor=cfg.folding_factor,
            align_proj_dim=cfg.align_proj_dim,
            gate_signal=cfg.gate_signal,
            use_gate_shift=cfg.use_gate_shift and not cfg.pmg_only,
        )

    def forward(self, vt: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Spec §5.1 forward; padded positions zeroed in every output."""
        if mask is not None:
            vt = vt * mask.unsqueeze(-1)
        eo = self.pmg(vt)  # (B, L, d_flow)
        vk = self.shift(vt, eo, mask) if self.shift is not None else vt
        yo = self.mhead(eo, mask)  # (B, L)
        return vk, eo, yo

    def project_for_align(self, eo: Tensor, vt: Tensor) -> tuple[Tensor, Tensor]:
        """Project both modalities to the common ``d_c`` space for ``L_KIP_align``.

        Returns unnormalized ``(proj_flow(ê_O), proj_rgb(v^t))``; the loss handles
        L2 normalization and masking.
        """
        return self.proj_flow(eo), self.proj_rgb(vt)


__all__ = ["KIP"]
