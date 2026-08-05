"""KIP - Kinematic Induction Pathway (novel; spec 2-5)."""

from core.kip.gate_shift import (
    GATE_SIGNAL_FEAT_VAR,
    GATE_SIGNAL_FLOW_NORM,
    KinematicShift,
    shift_channels_reference,
    shift_channels_vectorized,
)
from core.kip.kip_module import KIP
from core.kip.losses import kinematic_loss, kip_alignment_loss, kip_reconstruction_loss
from core.kip.motion_head import MotionScoreHead
from core.kip.pmg import PMGFlowHead

__all__ = [
    "GATE_SIGNAL_FEAT_VAR",
    "GATE_SIGNAL_FLOW_NORM",
    "KIP",
    "KinematicShift",
    "MotionScoreHead",
    "PMGFlowHead",
    "kinematic_loss",
    "kip_alignment_loss",
    "kip_reconstruction_loss",
    "shift_channels_reference",
    "shift_channels_vectorized",
]
