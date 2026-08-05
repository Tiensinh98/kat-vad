"""Variable-length padding / masking utilities and the batch collate fn.

Ported from the LaGoVAD dataset utilities; the multi-dim numpy padding bug in
the baseline (`np.pad` with a scalar pad-width pads *every* axis) is fixed here
by always padding along the time axis only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

RESAMPLE_UNIFORM = "uniform"
RESAMPLE_RANDOM = "random"


def _pad_time_axis(feature: Tensor | np.ndarray, pad_len: int) -> Tensor | np.ndarray:
    """Zero-pad ``feature (L, *)`` to ``(L + pad_len, *)`` along axis 0."""
    if isinstance(feature, torch.Tensor):
        pad = torch.zeros(
            (pad_len, *feature.shape[1:]), dtype=feature.dtype, device=feature.device
        )
        return torch.cat([feature, pad], dim=0)
    widths = [(0, pad_len)] + [(0, 0)] * (feature.ndim - 1)
    return np.pad(feature, widths, mode="constant", constant_values=0)


def resample_or_pad_feature_length(
    feature: Tensor | np.ndarray,
    target_length: int,
    resample_mode: str = RESAMPLE_UNIFORM,
) -> tuple[Tensor | np.ndarray, int]:
    """Fit ``feature (L, *)`` to ``target_length`` rows.

    Longer inputs are subsampled (uniform grid or a random crop); shorter inputs
    are zero-padded. Returns ``(feature, valid_length)`` where ``valid_length``
    is the un-padded row count.
    """
    length = len(feature)
    if length > target_length:
        if resample_mode == RESAMPLE_UNIFORM:
            rows = np.linspace(0, length - 1, target_length).astype(np.int64)
            index = torch.as_tensor(rows) if isinstance(feature, torch.Tensor) else rows
            return feature[index], target_length
        if resample_mode == RESAMPLE_RANDOM:
            rng = np.random.default_rng()
            start = int(rng.integers(length - target_length))
            return feature[start : start + target_length], target_length
        raise ValueError(f"Unknown resample_mode: {resample_mode!r}")
    if length < target_length:
        return _pad_time_axis(feature, target_length - length), length
    return feature, length


def truncate_or_pad_feature_length(
    feature: Tensor | np.ndarray, target_length: int
) -> tuple[Tensor | np.ndarray, int]:
    """Hard-truncate or zero-pad ``feature (L, *)`` to exactly ``target_length``."""
    length = len(feature)
    if length >= target_length:
        return feature[:target_length], target_length
    return _pad_time_axis(feature, target_length - length), length


def padding_mask(lengths: Tensor, max_len: int) -> Tensor:
    """``(B, max_len)`` float mask: 1 for valid positions, 0 for padding."""
    return (
        torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    ).to(torch.float32)


def pad_and_stack(features: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Stack variable-length ``(L_i, *)`` tensors into ``(B, L_max, *)`` + lengths."""
    lengths = torch.tensor([len(f) for f in features], dtype=torch.long)
    max_len = int(lengths.max())
    padded: list[Tensor] = []
    for feature in features:
        if len(feature) < max_len:
            pad = torch.zeros(
                (max_len - len(feature), *feature.shape[1:]),
                dtype=feature.dtype,
                device=feature.device,
            )
            feature = torch.cat([feature, pad], dim=0)
        padded.append(feature)
    return torch.stack(padded), lengths


def collate_variable_length(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate dataset samples with variable-length tensor fields.

    Tensor/ndarray fields with a leading time axis (``v_feat``, ``e_o``,
    ``pseudo_frame_label``) are padded to the batch max and stacked; the batch
    gains ``v_feat_l`` (valid lengths) and ``mask``. Scalar tensors are stacked;
    any other field is collected into a list.
    """
    time_keys = ("v_feat", "e_o", "pseudo_frame_label")
    batch: dict[str, Any] = {}
    keys = samples[0].keys()
    for key in keys:
        values = [s[key] for s in samples]
        if key in time_keys and values[0] is not None:
            tensors = [torch.as_tensor(v) for v in values]
            batch[key], lengths = pad_and_stack(tensors)
            if "v_feat_l" not in batch:
                batch["v_feat_l"] = lengths
                batch["mask"] = padding_mask(lengths, int(lengths.max()))
        elif isinstance(values[0], torch.Tensor) and values[0].dim() == 0:
            batch[key] = torch.stack(values)
        elif isinstance(values[0], (int, float, bool)):
            batch[key] = torch.tensor(values)
        else:
            batch[key] = values
    return batch


__all__ = [
    "collate_variable_length",
    "pad_and_stack",
    "padding_mask",
    "resample_or_pad_feature_length",
    "truncate_or_pad_feature_length",
]
