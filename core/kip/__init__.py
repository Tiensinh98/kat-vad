"""KIP - Kinematic Induction Pathway (novel; spec 2-5)."""

from core.kip.ecmr import ego_compensated_residual, rank_map, shift_counts_from_ratio
from core.kip.gate_shift import (
    GATE_SIGNAL_FEAT_VAR,
    GATE_SIGNAL_FLOW_NORM,
    GATE_SIGNALS,
    GATE_TYPE_CONSTANT,
    GATE_TYPE_MLP_FROZEN,
    GATE_TYPE_MLP_STE,
    GATE_TYPE_RANK,
    GATE_TYPES,
    MLP_GATE_TYPES,
    KinematicShift,
    shift_channels_reference,
    shift_channels_straight_through,
    shift_channels_vectorized,
)
from core.kip.kip_module import KIP, KIPOutput
from core.kip.losses import kinematic_loss, kip_alignment_loss, kip_reconstruction_loss
from core.kip.motion_head import MotionScoreHead
from core.kip.pmg import PMGFlowHead

__all__ = [
    "GATE_SIGNALS",
    "GATE_SIGNAL_FEAT_VAR",
    "GATE_SIGNAL_FLOW_NORM",
    "GATE_TYPES",
    "GATE_TYPE_CONSTANT",
    "GATE_TYPE_MLP_FROZEN",
    "GATE_TYPE_MLP_STE",
    "GATE_TYPE_RANK",
    "KIP",
    "MLP_GATE_TYPES",
    "KIPOutput",
    "KinematicShift",
    "MotionScoreHead",
    "PMGFlowHead",
    "ego_compensated_residual",
    "kinematic_loss",
    "kip_alignment_loss",
    "kip_reconstruction_loss",
    "rank_map",
    "shift_channels_reference",
    "shift_channels_straight_through",
    "shift_channels_vectorized",
    "shift_counts_from_ratio",
]
