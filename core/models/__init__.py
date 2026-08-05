"""Baseline (LaGoVAD re-implementation) model components + assembled KAT-VAD."""

from core.models.clip_text import SoftPromptCLIPTextModel
from core.models.fusion import CoAttentionFusion, CoAttnFusionLayer
from core.models.heads import BinaryHead, ConvScoreHead, MultiClassHead, SimScoreHead
from core.models.kat_vad import KATVAD
from core.models.temporal_encoder import (
    RoFormerEncoder,
    TemporalEncoder,
    extended_attention_mask,
    extended_local_attention_mask,
)

__all__ = [
    "KATVAD",
    "BinaryHead",
    "CoAttentionFusion",
    "CoAttnFusionLayer",
    "ConvScoreHead",
    "MultiClassHead",
    "RoFormerEncoder",
    "SimScoreHead",
    "SoftPromptCLIPTextModel",
    "TemporalEncoder",
    "extended_attention_mask",
    "extended_local_attention_mask",
]
