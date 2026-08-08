"""Frame-level VAD metrics and per-video score pooling (lesson C12).

Kept free of torch/model imports so both :mod:`core.evaluate` (which scores) and
:mod:`core.tools.rescore` (which only reads saved ``.npz`` files) can use it.

The pooling rule is the part that matters. Micro AUC concatenates every video's
frames into a single ranking, so each clip's *absolute* score scale enters the
metric. On a test set containing normal videos that scale is genuine signal. On
an **all-abnormal** set (DoTA: every clip contains an anomaly, ~33 % of frames
positive) the task is purely within-clip temporal localization, the between-clip
scale carries no label information, and pooling raw scores buries the signal --
measured at 0.5055 raw vs 0.6142 min-max for LaGoVAD's own released checkpoint,
against a published 0.6260. LaGoVAD's ``offline_dota_eval.py`` min-max
normalizes each clip before accumulating; its generic ``full_length_eval.py``
does not, which is why the two scripts disagree.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from core import constants

LOGGER = logging.getLogger(__name__)


def frame_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Micro frame-level ROC AUC (labels in {0,1}); tested against torchmetrics."""
    return float(roc_auc_score(labels.astype(np.int64), scores))


def frame_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Micro frame-level average precision; tested against torchmetrics."""
    return float(average_precision_score(labels.astype(np.int64), scores))


def normalize_scores(scores: np.ndarray, method: str) -> np.ndarray:
    """Rescale one video's score curve. ``none`` returns it unchanged.

    ``minmax`` maps to [0,1] (LaGoVAD's DoTA convention), ``zscore`` to zero
    mean / unit variance. Both are strictly increasing, so neither changes the
    *within*-video ranking -- only how this video's frames interleave with
    other videos' frames in the pooled ranking.
    """
    if method == constants.SCORE_NORM_NONE:
        return scores
    if method == constants.SCORE_NORM_MINMAX:
        low, high = scores.min(), scores.max()
        scaled: np.ndarray = (scores - low) / (high - low + constants.SCORE_NORM_EPS)
        return scaled
    if method == constants.SCORE_NORM_ZSCORE:
        centered: np.ndarray = (scores - scores.mean()) / (
            scores.std() + constants.SCORE_NORM_EPS
        )
        return centered
    raise ValueError(
        f"Unknown score normalization {method!r}; "
        f"choose from {list(constants.SCORE_NORM_CHOICES)}"
    )


def resolve_score_norm(method: str, labels: list[np.ndarray]) -> str:
    """Resolve ``auto`` against the label set; other methods pass through.

    ``auto`` picks ``minmax`` when normal videos are below
    ``SCORE_NORM_AUTO_NORMAL_FRACTION`` of the test set, because pooling raw
    scores over an effectively all-abnormal set measures between-clip
    confidence rather than localization. Deciding from the labels rather than
    from a dataset name means a new all-abnormal benchmark cannot silently
    inherit the wrong protocol.

    The threshold is a fraction, not zero: DoTA val has 3 normal clips out of
    1,397 (windows that round away at stride 8), and demanding zero would hand
    those 3 clips a veto over the whole protocol.
    """
    if method != constants.SCORE_NORM_AUTO:
        return method
    abnormal = sum(int(gt.max() > 0) for gt in labels)
    normal_fraction = (len(labels) - abnormal) / len(labels)
    resolved = (
        constants.SCORE_NORM_MINMAX
        if normal_fraction < constants.SCORE_NORM_AUTO_NORMAL_FRACTION
        else constants.SCORE_NORM_NONE
    )
    LOGGER.info(
        "score-norm auto -> %s (%d/%d videos abnormal; %.2f%% normal, threshold %.0f%%)",
        resolved, abnormal, len(labels), 100 * normal_fraction,
        100 * constants.SCORE_NORM_AUTO_NORMAL_FRACTION,
    )
    return resolved


def macro_video_auc(
    scores: list[np.ndarray], labels: list[np.ndarray]
) -> tuple[float, int]:
    """Mean of per-video AUCs, over videos that contain both classes.

    Reported alongside the micro number because it needs no normalization at
    all: each video is ranked only against itself. On an all-abnormal set this
    is the honest localization metric; videos that are entirely positive or
    entirely negative have no defined AUC and are skipped (count returned).
    """
    per_video = [
        frame_auc(s, gt)
        for s, gt in zip(scores, labels, strict=True)
        if 0 < int(gt.sum()) < len(gt)
    ]
    if not per_video:
        raise ValueError("No video contains both classes; macro AUC undefined")
    return float(np.mean(per_video)), len(per_video)


def pooled_metrics(
    scores: list[np.ndarray], labels: list[np.ndarray], score_norm: str
) -> dict[str, float | int | str]:
    """Every headline number for one eval run, under a resolved pooling rule.

    Always reports the raw-pooled AUC too: it is what the previous protocol
    measured, and keeping both in ``results.json`` makes the difference between
    protocols auditable instead of a silent re-definition of the metric.
    """
    resolved = resolve_score_norm(score_norm, labels)
    normed = np.concatenate([normalize_scores(s, resolved) for s in scores])
    raw = np.concatenate(scores)
    labels_cat = np.concatenate(labels)
    if labels_cat.min() == labels_cat.max():
        raise ValueError("Test set has a single class; AUC/AP undefined")
    macro_auc, macro_n = macro_video_auc(scores, labels)
    return {
        "score_norm": resolved,
        "num_frames": len(labels_cat),
        "auc": frame_auc(normed, labels_cat),
        "ap": frame_ap(normed, labels_cat),
        "auc_raw": frame_auc(raw, labels_cat),
        "ap_raw": frame_ap(raw, labels_cat),
        "auc_macro": macro_auc,
        "auc_macro_videos": macro_n,
    }


__all__ = [
    "frame_ap",
    "frame_auc",
    "macro_video_auc",
    "normalize_scores",
    "pooled_metrics",
    "resolve_score_norm",
]
