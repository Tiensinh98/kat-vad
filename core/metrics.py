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


def cluster_bootstrap_ci(
    values: dict[str, float],
    groups: dict[str, str],
    resamples: int,
    seed: int,
    level: float,
) -> dict[str, float] | None:
    """Mean of per-item ``values`` and a percentile CI resampling their clusters.

    Clusters are drawn with replacement and the statistic is the item mean over
    the drawn clusters (v2 addendum D3: DoTA clips of one source video are
    correlated, so a clip bootstrap understates the variance). ``None`` when
    ``values`` is empty.
    """
    if not values:
        return None
    names = sorted({groups[item] for item in values})
    index = {name: i for i, name in enumerate(names)}
    sums = np.zeros(len(names))
    counts = np.zeros(len(names))
    for item, value in values.items():
        sums[index[groups[item]]] += value
        counts[index[groups[item]]] += 1
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(names), size=(resamples, len(names)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return {
        "mean": float(np.mean(list(values.values()))),
        "low": float(low),
        "high": float(high),
        "clips": len(values),
        "clusters": len(names),
    }


def span_reads(
    scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Per two-class clip: where the score sits relative to the anomalous span.

    Each clip is min-max normalized first, so a global rescale of an arm's scores
    cannot move these reads. Keys: ``argmax_in_span`` (1.0 if the clip's first
    argmax is a positive frame), ``in_mean`` (mean over positive frames) and, when
    the clip has normal frames before its first positive one, ``pre_mean``.
    """
    out: dict[str, dict[str, float]] = {}
    for clip in sorted(scores):
        lab = np.asarray(labels[clip]).astype(bool)
        if not 0 < int(lab.sum()) < len(lab):
            continue
        if len(scores[clip]) != len(lab):
            raise ValueError(f"{clip}: {len(scores[clip])} scores vs {len(lab)} labels")
        norm = normalize_scores(
            np.asarray(scores[clip], dtype=np.float64), constants.SCORE_NORM_MINMAX
        )
        read = {
            "argmax_in_span": float(lab[int(np.argmax(norm))]),
            "in_mean": float(norm[lab].mean()),
        }
        first = int(np.flatnonzero(lab)[0])
        if first > 0:
            read["pre_mean"] = float(norm[:first].mean())
        out[clip] = read
    return out


def span_summary(reads: dict[str, dict[str, float]]) -> dict[str, float | int | None]:
    """Mean of each :func:`span_reads` key over the clips that have it."""
    summary: dict[str, float | int | None] = {"clips": len(reads)}
    for key in ("argmax_in_span", "in_mean", "pre_mean"):
        values = [r[key] for r in reads.values() if key in r]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def normal_window_peak(
    scores: list[np.ndarray], labels: list[np.ndarray], topk_pct: int
) -> float | None:
    """Mean over all-normal windows of the mean top-k raw score, k = max(1, L // topk_pct).

    The quantity ``L_MIL`` pushes down on a normal bag; ``None`` without a normal window.
    """
    peaks = []
    for score, lab in zip(scores, labels, strict=True):
        if np.asarray(lab).any():
            continue
        k = max(1, len(score) // topk_pct)
        peaks.append(float(np.sort(np.asarray(score, dtype=np.float64))[-k:].mean()))
    return float(np.mean(peaks)) if peaks else None


__all__ = [
    "cluster_bootstrap_ci",
    "frame_ap",
    "frame_auc",
    "macro_video_auc",
    "normal_window_peak",
    "normalize_scores",
    "pooled_metrics",
    "resolve_score_norm",
    "span_reads",
    "span_summary",
]
