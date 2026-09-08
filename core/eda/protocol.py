"""What a micro AUC on this label distribution is actually measuring.

Lesson **C12** has two faces and this corpus set shows both:

* **DoTA** is ~all-abnormal, so pooling raw scores ranks clips by confidence
  rather than localizing anything — hence the per-clip min-max protocol.
* **DADA-2000** is the mirror: 74 % of its test frames come from clips that are
  negative in their entirety, so a model that emits **one constant score per
  clip** and does zero localization scores **0.9086** micro. Any micro number
  there is mostly a video-classification score.

:func:`clip_constant_oracle` computes that ceiling exactly, and
:func:`pair_decomposition` says what fraction of the metric it can reach — both
from labels alone, with no model involved.

:func:`clip_length_leak` asks a harder question (lesson **C28**): can the clip's
*length* alone predict its label? On the reconstructed DADA-2000 it can — a
constant score of ``-T`` per clip reaches **micro AUC 0.8654**, within 0.009 of
the best trained arm — because the build trims accident videos and leaves normal
ones full length. A corpus that fails this check leaks its label through a
channel no model change can remove.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.eda.corpus import DatasetFiles, describe
from core.metrics import frame_ap, frame_auc, macro_video_auc, resolve_score_norm

LOGGER = logging.getLogger(__name__)

SCORE_NPZ_SCORE_KEY = "score"
SCORE_NPZ_GT_KEY = "gt"


def frame_share(labels: list[np.ndarray]) -> dict[str, Any]:
    """How the test frames divide between all-normal, all-abnormal and mixed clips."""
    total = int(sum(a.size for a in labels))
    buckets = {"all_normal": 0, "all_positive": 0, "mixed": 0}
    clips = {"all_normal": 0, "all_positive": 0, "mixed": 0}
    for array in labels:
        positives = int(array.sum())
        key = (
            "all_normal" if positives == 0
            else "all_positive" if positives == array.size
            else "mixed"
        )
        buckets[key] += int(array.size)
        clips[key] += 1
    return {
        "total_frames": total,
        "clips": clips,
        "frames": buckets,
        "frame_fraction": {k: (v / total if total else None) for k, v in buckets.items()},
    }


def clip_constant_oracle(labels: list[np.ndarray]) -> dict[str, Any]:
    """Micro AUC of a *perfect clip classifier that does no localization*.

    Every frame of a clip gets the same score: 1 if the clip contains any
    positive frame, else 0. Within-clip ranking is therefore chance by
    construction (``auc_macro`` = 0.5), so whatever this scores is the part of
    the micro metric that clip classification alone can buy. Print it beside
    every micro number on a test set that contains all-normal clips.
    """
    if not labels:
        raise ValueError("No label arrays; cannot compute the clip oracle")
    scores = np.concatenate([np.full(a.size, float(a.max() > 0)) for a in labels])
    flat = np.concatenate(labels).astype(np.int64)
    if flat.min() == flat.max():
        raise ValueError("Test set has a single class; oracle AUC undefined")
    return {
        "auc_micro": frame_auc(scores, flat),
        "ap_micro": frame_ap(scores, flat),
        "auc_macro": 0.5,
        "definition": "constant score per clip = clip label; zero within-clip ranking",
    }


def clip_length_leak(labels: list[np.ndarray]) -> dict[str, Any]:
    """Can the clip's **length** alone predict its label? (lesson **C28**)

    A corpus reconstructed by trimming abnormal videos around the event, while
    leaving normal videos at full length, encodes its label in the frame count.
    That channel survives every model, loss and backbone change, and a model can
    read it: attention masks and positional encodings are length-dependent, and
    clips are scored at their true length.

    Scores every frame of a clip with the clip's own length (signed so that the
    predictive direction is positive), i.e. a constant-score-per-clip detector
    that has seen no pixels at all. Whatever this reaches is free, and any arm
    that fails to beat it is unmeasured.
    """
    if not labels:
        raise ValueError("No label arrays; cannot compute the length leak")
    lengths = np.array([float(a.size) for a in labels], dtype=np.float64)
    clip_labels = np.array([int(a.max() > 0) for a in labels], dtype=np.int64)
    if clip_labels.min() == clip_labels.max():
        raise ValueError("Test set has a single clip class; length leak undefined")
    flat = np.concatenate(labels).astype(np.int64)
    if flat.min() == flat.max():
        raise ValueError("Test set has a single frame class; length leak undefined")

    auc_longer = frame_auc(lengths, clip_labels)
    longer_is_abnormal = auc_longer >= 0.5
    signed = lengths if longer_is_abnormal else -lengths
    frame_scores = np.concatenate(
        [np.full(a.size, s) for a, s in zip(labels, signed, strict=True)]
    )

    abnormal = lengths[clip_labels == 1]
    normal = lengths[clip_labels == 0]
    # Normal clips lying outside the abnormal length range on the normal side:
    # separable for free, by a threshold, with no pixels read.
    disjoint = (
        normal < abnormal.min() if longer_is_abnormal else normal > abnormal.max()
    )
    disjoint_frames = int(lengths[clip_labels == 0][disjoint].sum())
    return {
        "auc_clip_level": max(auc_longer, 1.0 - auc_longer),
        "direction": "longer" if longer_is_abnormal else "shorter",
        "auc_micro": frame_auc(frame_scores, flat),
        "ap_micro": frame_ap(frame_scores, flat),
        "auc_macro": 0.5,
        "abnormal_length": describe(abnormal.tolist(), "abnormal clip length T"),
        "normal_length": describe(normal.tolist(), "normal clip length T"),
        "disjoint_normal_clips": int(disjoint.sum()),
        "disjoint_normal_frames": disjoint_frames,
        "definition": (
            "constant score per clip = clip length (signed to the predictive "
            "direction); zero within-clip ranking, no pixels read"
        ),
    }


def pair_decomposition(labels: list[np.ndarray]) -> dict[str, Any]:
    """Split the micro-AUC positive/negative pairs into within-clip and cross-clip.

    Micro ROC AUC is the fraction of (positive, negative) frame pairs ranked
    correctly. A pair whose two frames sit in the *same* clip can only be decided
    by localization; a pair spanning two clips is decided by the clips' relative
    score scale. The cross-clip share is therefore the fraction of the metric that
    a clip classifier can win without localizing anything.
    """
    positives = np.array([int(a.sum()) for a in labels], dtype=np.int64)
    negatives = np.array([int(a.size - a.sum()) for a in labels], dtype=np.int64)
    total_pairs = int(positives.sum()) * int(negatives.sum())
    within_pairs = int((positives * negatives).sum())
    if total_pairs == 0:
        raise ValueError("Test set has a single class; pair decomposition undefined")
    return {
        "total_pairs": total_pairs,
        "within_clip_pairs": within_pairs,
        "cross_clip_pairs": total_pairs - within_pairs,
        "within_clip_fraction": within_pairs / total_pairs,
        "cross_clip_fraction": (total_pairs - within_pairs) / total_pairs,
    }


def macro_resolution(labels: list[np.ndarray]) -> dict[str, Any]:
    """How coarse a per-clip AUC is on this corpus.

    A clip with ``p`` positives and ``n`` negatives has exactly ``p * n``
    orderable pairs, so its AUC can only take values on a 1/(p*n) grid. Report it
    with ``auc_macro``: a mean of coarse per-clip AUCs is honest but low
    resolution, and quoting it as a bare point estimate oversells it.
    """
    grids: list[float] = []
    two_class = 0
    for array in labels:
        positives = int(array.sum())
        negatives = int(array.size - positives)
        if positives and negatives:
            two_class += 1
            grids.append(1.0 / (positives * negatives))
    return {
        "two_class_clips": two_class,
        "single_class_clips": len(labels) - two_class,
        "auc_grid_step": describe(grids, "per-clip AUC resolution (1 / (pos * neg))"),
    }


def score_norm_resolution(labels: list[np.ndarray]) -> dict[str, Any]:
    """What ``--score-norm auto`` resolves to here, and why (lesson **C12**)."""
    normal_clips = sum(1 for a in labels if int(a.sum()) == 0)
    fraction = normal_clips / len(labels) if labels else 0.0
    return {
        "normal_clip_fraction": fraction,
        "threshold": constants.SCORE_NORM_AUTO_NORMAL_FRACTION,
        "resolves_to": resolve_score_norm(constants.SCORE_NORM_AUTO, labels),
    }


def load_score_curves(scores_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Read an eval run's per-clip ``.npz`` files. Raises on a truncated write."""
    files = sorted(scores_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz score files under {scores_dir}")
    empty = [p.name for p in files if p.stat().st_size == 0]
    if empty:
        raise ValueError(
            f"{len(empty)} zero-byte score files in {scores_dir} (lesson C11b): {empty[:3]}"
        )
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for path in files:
        with np.load(path) as payload:
            scores.append(np.asarray(payload[SCORE_NPZ_SCORE_KEY], dtype=np.float64))
            labels.append(np.asarray(payload[SCORE_NPZ_GT_KEY], dtype=np.int8))
    return scores, labels, [p.stem for p in files]


def curve_flatness(scores: list[np.ndarray]) -> dict[str, Any]:
    """Between-clip vs within-clip variance of a model's score curves.

    ``RESULTS_DADA.md`` §4/§7.1 measured Spearman(this ratio, DADA micro) = +0.68
    and Spearman(this ratio, DoTA micro) = -0.82 across seven arms: the flatter a
    curve, the better it separates *clips* and the worse it localizes. A high
    ratio is the signature of a model that has learned clip classification.
    """
    means = np.array([s.mean() for s in scores], dtype=np.float64)
    sizes = np.array([s.size for s in scores], dtype=np.float64)
    within = np.array([s.var() for s in scores], dtype=np.float64)
    frame_weighted_within = float((within * sizes).sum() / sizes.sum())
    between = float(means.var())
    ranges = [float(s.max() - s.min()) for s in scores]
    return {
        "between_clip_variance": between,
        "within_clip_variance_frame_weighted": frame_weighted_within,
        "between_over_within": (
            between / frame_weighted_within if frame_weighted_within > 0 else None
        ),
        "within_clip_range": describe(ranges, "per-clip score range (max - min)"),
        "constant_curve_clips": int(sum(1 for r in ranges if r == 0.0)),
    }


def protocol_report(files: DatasetFiles, scores_dir: Path | None = None) -> dict[str, Any]:
    """Everything in this module. ``scores_dir`` adds the model-curve flatness block."""
    labels = files.test_label_arrays()
    try:
        length_leak: dict[str, Any] = clip_length_leak(labels)
    except ValueError as exc:  # single-class test split: the check has no meaning
        LOGGER.warning("Skipping the clip-length leak check: %s", exc)
        length_leak = {"skipped": str(exc)}
    report: dict[str, Any] = {
        "frame_share": frame_share(labels),
        "clip_constant_oracle": clip_constant_oracle(labels),
        "clip_length_leak": length_leak,
        "pair_decomposition": pair_decomposition(labels),
        "macro_resolution": macro_resolution(labels),
        "score_norm_auto": score_norm_resolution(labels),
    }
    if scores_dir is not None:
        scores, gt, ids = load_score_curves(scores_dir)
        macro, macro_n = macro_video_auc(scores, gt)
        report["scored_run"] = {
            "scores_dir": str(scores_dir),
            "clips": len(ids),
            "auc_macro": macro,
            "auc_macro_videos": macro_n,
            "flatness": curve_flatness(scores),
        }
    return report


__all__ = [
    "clip_constant_oracle",
    "clip_length_leak",
    "curve_flatness",
    "frame_share",
    "load_score_curves",
    "macro_resolution",
    "pair_decomposition",
    "protocol_report",
    "score_norm_resolution",
]
