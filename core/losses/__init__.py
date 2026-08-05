"""Baseline (LaGoVAD) losses re-implemented for KAT-VAD.

KIP-specific losses live in :mod:`core.kip.losses`.
"""

from core.losses.contrastive import CapContrastLoss, asymmetric_infonce_loss
from core.losses.dvs import pseudo_sup_mil_loss, supervised_loss
from core.losses.mil import mil_loss, multi_class_mil_loss, multi_class_mil_loss_v2

__all__ = [
    "CapContrastLoss",
    "asymmetric_infonce_loss",
    "mil_loss",
    "multi_class_mil_loss",
    "multi_class_mil_loss_v2",
    "pseudo_sup_mil_loss",
    "supervised_loss",
]
