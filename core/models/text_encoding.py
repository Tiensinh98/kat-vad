"""Text-feature providers for the train/inference/evaluate CLIs.

Two modes (CLI flag ``--text-encoder``):

- ``clip``: the model's frozen CLIP text tower + soft prompts
  (:meth:`KATVAD.encode_text`; requires HF weights → Phase-6 runs).
- ``stub``: a deterministic per-string Gaussian embedding (seeded from a
  SHA-256 of the text). Zero downloads — used by the synthetic E2E test and
  the Colab dry-run. Same string ⇒ same vector, across processes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from core import constants

if TYPE_CHECKING:
    from core.models.kat_vad import KATVAD

TEXT_ENCODER_CLIP = "clip"
TEXT_ENCODER_STUB = "stub"
TEXT_ENCODER_CHOICES = (TEXT_ENCODER_CLIP, TEXT_ENCODER_STUB)

TextEncodeFn = Callable[[list[str]], Tensor]

_STUB_SEED_BYTES = 8


def stub_text_features(
    texts: list[str], dim: int = constants.CLIP_FEATURE_DIM
) -> Tensor:
    """Deterministic pseudo-embeddings: one seeded ``randn(dim)`` per string."""
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:_STUB_SEED_BYTES], "big") % (2**63)
        generator = torch.Generator().manual_seed(seed)
        rows.append(torch.randn(dim, generator=generator))
    return torch.stack(rows)


def make_text_encoder(
    model: KATVAD,
    mode: str,
    device: torch.device,
    dim: int = constants.CLIP_FEATURE_DIM,
    use_soft_prompt: bool = True,
) -> TextEncodeFn:
    """Build the ``texts -> (len(texts), D)`` callable for the chosen mode."""
    if mode == TEXT_ENCODER_STUB:

        def encode_stub(texts: list[str]) -> Tensor:
            return stub_text_features(texts, dim).to(device)

        return encode_stub
    if mode == TEXT_ENCODER_CLIP:

        def encode_clip(texts: list[str]) -> Tensor:
            return model.encode_text(texts, use_soft_prompt=use_soft_prompt)

        return encode_clip
    raise ValueError(f"Unknown text encoder mode: {mode!r} (use {TEXT_ENCODER_CHOICES})")


__all__ = [
    "TEXT_ENCODER_CHOICES",
    "TEXT_ENCODER_CLIP",
    "TEXT_ENCODER_STUB",
    "TextEncodeFn",
    "make_text_encoder",
    "stub_text_features",
]
