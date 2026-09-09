"""The ``L_dvs`` pair (A2): frame-wise supervised BCE + pseudo-supervised MIL.

Both consume the DVS pseudo frame labels ``y^p``; re-implemented from the
LaGoVAD baseline (``supervised_loss`` / ``pseudo_sup_mil_loss``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from core import constants


def supervised_loss(
    logits: Tensor,
    frame_labels: Tensor,
    lengths: Tensor,
    *,
    ignore_positive: bool = False,
) -> Tensor:
    """Length-masked frame-level BCE against DVS pseudo labels.

    ``logits (B, T)``, ``frame_labels (B, T)`` in {0, 1}, ``lengths (B,)``.

    ``ignore_positive`` (lesson **C29**, Phase 1.1) drops the pseudo-*positive*
    frames from the dense BCE, turning ``y^p`` into a 3-valued target
    ``{0, ignore}``. ``compose_sequence`` marks the **entire** spliced anchor
    clip positive, which is right only when an abnormal clip is ~all anomaly;
    on DADA-2000 the mean true positive fraction is 0.351, so 64.9 % of those
    frames are trained to 1 against a 0 annotation. With this flag the filler
    frames stay hard negatives and the positive pressure comes from
    :func:`pseudo_sup_mil_loss`, whose top-k is already restricted to the span.

    Keyword-only with a ``False`` default: the three-positional-argument call
    is byte-identical to the baseline, which ``test_baseline_parity`` asserts.
    """
    mask = (
        torch.arange(logits.shape[1], device=logits.device)[None, :]
        < lengths[:, None]
    ).to(logits.dtype)
    if ignore_positive:
        mask = mask * (frame_labels < 0.5).to(mask.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits, frame_labels.to(logits.dtype), reduction="none"
    )
    # An all-positive row under ``ignore_positive`` leaves an empty mask; the
    # numerator is then 0 too, so clamping keeps the term at 0 *with* a grad_fn
    # instead of returning NaN. Never reached when ignore_positive is False
    # (lengths >= 1), so baseline parity is exact.
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def pseudo_sup_mil_loss(
    logits: Tensor,
    frame_labels: Tensor,
    lengths: Tensor,
    topk_pct: int = constants.SUP_MIL_TOPK_PCT,
) -> Tensor:
    """Top-k MIL restricted to the annotated (pseudo-positive) span.

    Abnormal videos select top-k inside ``y^p = 1`` frames (others pushed out of
    the top-k by an additive ``PSEUDO_LABEL_IGNORE`` mask); normal videos use
    the plain top-k over valid frames. BCE against the derived video label.
    """
    batch = logits.shape[0]
    span_mask = torch.zeros_like(frame_labels, dtype=logits.dtype)
    span_mask[frame_labels == 0] = constants.PSEUDO_LABEL_IGNORE

    video_logits = []
    video_labels = []
    for b in range(batch):
        n_pos = int(frame_labels[b].sum())
        if n_pos == 0:  # normal video
            k = max(1, int(lengths[b]) // topk_pct)
            top_values = logits[b, : int(lengths[b])].topk(k).values
            video_labels.append(0.0)
        else:  # abnormal: top-k within the annotated span (full padded row, as baseline)
            k = max(1, n_pos // topk_pct)
            top_values = (logits[b] + span_mask[b]).topk(k).values
            video_labels.append(1.0)
        video_logits.append(top_values.mean())
    return F.binary_cross_entropy_with_logits(
        torch.stack(video_logits),
        torch.tensor(video_labels, device=logits.device, dtype=logits.dtype),
    )


__all__ = ["pseudo_sup_mil_loss", "supervised_loss"]
