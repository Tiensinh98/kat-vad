"""LaGoVAD ``best.ckpt`` → :class:`core.models.kat_vad.KATVAD` weight mapping.

The Phase-3 parity tests pin every submodule's key layout 1:1 against the
baseline, so the only differences between a LaGoVAD Lightning checkpoint and
our model's state dict are structural:

- baseline ``temporal_encoder.*``      → ours ``temporal_encoder.encoder.*``
  (our :class:`TemporalEncoder` wraps the RoFormer stack and owns the residual)
- baseline top-level ``gate_alpha``    → ours ``temporal_encoder.gate_alpha``
- baseline ``clip_text_model.model.*`` is *absent* from checkpoints (frozen
  CLIP body is stripped on save) and comes from HF weights on our side
- ours-only ``kip.*`` keeps its fresh initialization (baseline has no KIP)

``fusion.*``, ``bin_head.*``, ``sim_head.*`` and
``clip_text_model.prompt_embedding.*`` map 1:1.

Anything in the checkpoint that maps to no parameter of ours is an error —
silent partial loads are exactly what reproduction gate (a) exists to catch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

    from core.models.kat_vad import KATVAD

LOGGER = logging.getLogger(__name__)

_TEMPORAL_PREFIX = "temporal_encoder."
_OURS_TEMPORAL_PREFIX = "temporal_encoder.encoder."
_GATE_ALPHA_KEY = "gate_alpha"
_OURS_GATE_ALPHA_KEY = "temporal_encoder.gate_alpha"
_CLIP_BODY_PREFIX = "clip_text_model.model."
_KIP_PREFIX = "kip."
_PASSTHROUGH_PREFIXES = (
    "fusion.",
    "bin_head.",
    "sim_head.",
    "clip_text_model.prompt_embedding.",
)


@dataclass
class LoadReport:
    """Outcome of a compat load; ``loaded`` counts checkpoint keys consumed."""

    loaded: list[str] = field(default_factory=list)
    skipped_clip_body: list[str] = field(default_factory=list)
    missing_ours: list[str] = field(default_factory=list)  # ours, not in ckpt


def map_baseline_key(key: str) -> str | None:
    """Map one baseline state-dict key to ours; ``None`` = intentionally skipped."""
    if key.startswith(_CLIP_BODY_PREFIX):
        return None
    if key == _GATE_ALPHA_KEY:
        return _OURS_GATE_ALPHA_KEY
    if key.startswith(_TEMPORAL_PREFIX):
        return _OURS_TEMPORAL_PREFIX + key[len(_TEMPORAL_PREFIX) :]
    if key.startswith(_PASSTHROUGH_PREFIXES):
        return key
    raise KeyError(f"Unrecognized baseline checkpoint key: {key!r}")


def baseline_key_for(our_key: str) -> str | None:
    """Inverse of :func:`map_baseline_key` (test/tooling helper); ``None`` = ours-only."""
    if our_key.startswith(_KIP_PREFIX) or our_key.startswith(_CLIP_BODY_PREFIX):
        return None
    if our_key == _OURS_GATE_ALPHA_KEY:
        return _GATE_ALPHA_KEY
    if our_key.startswith(_OURS_TEMPORAL_PREFIX):
        return _TEMPORAL_PREFIX + our_key[len(_OURS_TEMPORAL_PREFIX) :]
    if our_key.startswith(_PASSTHROUGH_PREFIXES):
        return our_key
    raise KeyError(f"Unrecognized KATVAD state-dict key: {our_key!r}")


def load_baseline_state_dict(
    model: KATVAD, state_dict: dict[str, Tensor]
) -> LoadReport:
    """Load a LaGoVAD ``state_dict`` (Lightning ``['state_dict']``) into ``model``."""
    ours = model.state_dict()
    report = LoadReport()
    mapped: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        target = map_baseline_key(key)
        if target is None:
            report.skipped_clip_body.append(key)
            continue
        if target not in ours:
            raise KeyError(
                f"Baseline key {key!r} maps to {target!r} which does not exist "
                "in this KATVAD configuration (check model config vs checkpoint)"
            )
        if ours[target].shape != value.shape:
            raise ValueError(
                f"Shape mismatch for {key!r} -> {target!r}: "
                f"{tuple(value.shape)} vs {tuple(ours[target].shape)}"
            )
        mapped[target] = value
        report.loaded.append(key)

    for our_key in ours:
        if our_key in mapped:
            continue
        if baseline_key_for(our_key) is None:
            continue  # kip.* / frozen CLIP body: expected to be ours-only
        report.missing_ours.append(our_key)
    if report.missing_ours:
        raise KeyError(
            "Checkpoint is missing weights for mapped parameters: "
            f"{report.missing_ours[:10]}{'...' if len(report.missing_ours) > 10 else ''}"
        )

    result = model.load_state_dict(mapped, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:  # cannot happen after the checks above, but fail loudly anyway
        raise KeyError(f"Unexpected keys after mapping: {unexpected}")
    LOGGER.info(
        "Baseline checkpoint loaded: %d tensors mapped, %d frozen-CLIP keys skipped",
        len(report.loaded),
        len(report.skipped_clip_body),
    )
    return report


def load_baseline_checkpoint(model: KATVAD, path: Path) -> LoadReport:
    """Load LaGoVAD ``best.ckpt`` (or any Lightning ckpt of it) from disk."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = payload.get("state_dict", payload)
    return load_baseline_state_dict(model, state_dict)


__all__ = [
    "LoadReport",
    "baseline_key_for",
    "load_baseline_checkpoint",
    "load_baseline_state_dict",
    "map_baseline_key",
]
