"""Dynamic Video Synthesis primitives (spec §6.1) — pure, rng-injected.

Generalization of the baseline ``PreVADDatasetOnline`` splicing: an anchor
clip is inserted at a random slot among normal filler clips; features, flow
embeddings (``e_O``) and the frame-level pseudo label ``y^p`` are concatenated
in the same order so they stay aligned. θ is the **no-synthesis probability**
(research decision OQ5): with probability θ the anchor is returned alone and
``is_synthesized=False``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from core import constants


@dataclass
class SynthesizedSample:
    """Aligned outputs of one DVS draw (variable length, un-padded)."""

    features: Tensor  # (T, D) float32
    flow: Tensor | None  # (T, d_O) float32, None when flow not requested
    pseudo_label: Tensor  # (T,) float32, 1 on the anchor span iff synthesized+abnormal
    is_synthesized: bool
    clip_ids: list[str]  # composition order, anchor included


def choose_filler_ids(
    rng: random.Random,
    num_fillers: int,
    normal_ids: list[str],
    knn_neighbors: list[str] | None,
    knn_ratio: float = constants.DVS_KNN_FILLER_RATIO,
) -> list[str]:
    """Pick filler video ids: ``knn_ratio`` of draws from the KNN neighbors of
    the anchor (when available), the rest uniformly from all normal videos."""
    if not normal_ids:
        raise ValueError("No normal videos available as DVS fillers")
    fillers: list[str] = []
    for _ in range(num_fillers):
        if knn_neighbors and rng.random() < knn_ratio:
            fillers.append(rng.choice(knn_neighbors))
        else:
            fillers.append(rng.choice(normal_ids))
    return fillers


def compose_sequence(
    anchor_id: str,
    anchor_features: Tensor,
    anchor_flow: Tensor | None,
    anchor_is_abnormal: bool,
    fillers: Sequence[tuple[str, Tensor, Tensor | None]],
    insert_index: int,
) -> SynthesizedSample:
    """Concatenate fillers with the anchor inserted at ``insert_index``.

    ``y^p`` marks the anchor span only for a *synthesized* (fillers present)
    abnormal sequence — that is the known-window supervision of spec §6.1.
    An un-synthesized abnormal clip has an unknown anomaly window (weak
    supervision), so its ``y^p`` stays all-zero and losses fall back to the
    ``y^bin.detach()`` anchor.
    """
    if not 0 <= insert_index <= len(fillers):
        raise ValueError(f"insert_index {insert_index} out of range 0..{len(fillers)}")

    parts: list[tuple[str, Tensor, Tensor | None]] = list(fillers)
    parts.insert(insert_index, (anchor_id, anchor_features, anchor_flow))

    features = torch.cat([p[1] for p in parts], dim=0)
    flows = [p[2] for p in parts]
    flow = torch.cat(flows, dim=0) if all(f is not None for f in flows) else None  # type: ignore[arg-type]

    pseudo = torch.zeros(len(features), dtype=torch.float32)
    if anchor_is_abnormal and fillers:
        start = sum(len(p[1]) for p in parts[:insert_index])
        pseudo[start : start + len(anchor_features)] = 1.0

    return SynthesizedSample(
        features=features,
        flow=flow,
        pseudo_label=pseudo,
        is_synthesized=len(fillers) > 0,
        clip_ids=[p[0] for p in parts],
    )


def truncate_sample(sample: SynthesizedSample, max_len: int) -> SynthesizedSample:
    """Hard-truncate all aligned fields to ``max_len`` rows (baseline behavior)."""
    if len(sample.features) <= max_len:
        return sample
    return SynthesizedSample(
        features=sample.features[:max_len],
        flow=sample.flow[:max_len] if sample.flow is not None else None,
        pseudo_label=sample.pseudo_label[:max_len],
        is_synthesized=sample.is_synthesized,
        clip_ids=sample.clip_ids,
    )


__all__ = [
    "SynthesizedSample",
    "choose_filler_ids",
    "compose_sequence",
    "truncate_sample",
]
