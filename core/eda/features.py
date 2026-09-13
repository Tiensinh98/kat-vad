"""What the cached frozen-CLIP features contain, before any model touches them.

The centrepiece is :func:`frame_linear_probe` — item **B** of
``core/docs/v3/RESULTS_DADA.md`` §10, the decisive experiment the campaign left
open. Every arm on DADA-2000 scored `auc_macro` at chance, and two explanations
were indistinguishable from the training numbers alone:

1. **Representation.** Frozen CLIP image features cannot separate an accident
   frame from its neighbours, so no head could.
2. **Supervision.** They can, and video-level MIL over 9-frame clips cannot find
   it.

A supervised linear probe on the *same cached features* against the *real frame
labels* answers this directly: if the probe localizes and the trained arms do
not, the deficit is supervision, not representation — and no amount of KIP will
fix it. If the probe is also at chance, frame-level localization on this corpus
needs a different backbone (see §6's SimpleTAD comparison), not a better head.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from core import constants
from core.data.windows import FeatureSlicer
from core.eda.corpus import DatasetFiles, describe
from core.metrics import frame_ap, frame_auc

LOGGER = logging.getLogger(__name__)

FLOW_STATS_SUFFIX = ".stats.npy"


def load_clip_features(
    clip_dir: Path,
    video_ids: list[str],
    expected: dict[str, int] | None = None,
    slicer: FeatureSlicer | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load each id's feature rows; return the features and the missing ids.

    When ``expected`` is given, a feature array whose row count disagrees with the
    label vector raises: that mismatch means the cache and the labels were built
    at different strides or from different frame folders (lessons **C2**/**C13**),
    and every downstream number would be measured on misaligned rows.

    On a **windowed** corpus the ids are window ids while the cache is keyed by
    source clip, so the rows must come through the ``slicer``. Reading
    ``clip_dir / f"{window_id}.npy"`` directly reports every clip missing and
    silently kills sections 4.1-4.2, the linear probe included.
    """
    slicer = slicer if slicer is not None else FeatureSlicer()
    features: dict[str, np.ndarray] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    for video_id in video_ids:
        path = clip_dir / f"{slicer.source_of(video_id)}.npy"
        if not path.is_file():
            missing.append(video_id)
            continue
        array = slicer.load(clip_dir, video_id)
        if expected is not None and video_id in expected and len(array) != expected[video_id]:
            mismatched.append(f"{video_id}: features {len(array)} vs labels {expected[video_id]}")
            continue
        features[video_id] = np.asarray(array, dtype=np.float32)
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} clips have feature rows that do not match their label "
            f"rows -- the cache was built at a different stride or transform "
            f"(lessons C2/C13): {mismatched[:3]}"
        )
    if missing:
        LOGGER.warning(
            "%d/%d ids have no feature file under %s, e.g. %s",
            len(missing), len(video_ids), clip_dir, missing[:5],
        )
    return features, missing


def feature_stats(features: dict[str, np.ndarray]) -> dict[str, Any]:
    """L2-norm and dimension-variance summary of the cached embeddings."""
    if not features:
        return {"clips": 0}
    stacked = np.concatenate(list(features.values()), axis=0)
    norms = np.linalg.norm(stacked, axis=1)
    per_dim_var = stacked.var(axis=0)
    return {
        "clips": len(features),
        "frames": int(stacked.shape[0]),
        "dim": int(stacked.shape[1]),
        "l2_norm": describe(norms, "per-frame L2 norm"),
        "per_dim_variance": describe(per_dim_var, "per-dimension variance"),
        "effective_dims_90pct_variance": int(
            np.searchsorted(np.cumsum(np.sort(per_dim_var)[::-1]) / per_dim_var.sum(), 0.90) + 1
        ),
    }


def temporal_autocorrelation(
    features: dict[str, np.ndarray], max_lag: int = constants.EDA_AUTOCORR_MAX_LAG
) -> dict[str, Any]:
    """Mean cosine similarity between frame ``t`` and frame ``t + lag``, per lag.

    Reads as "how much new information does one more sampled frame carry". A
    curve that is still near 1.0 at lag 1 says the stride is finer than the
    content changes; one that drops fast says the opposite. This is the evidence
    to bring to any ``frame_stride`` change, which fires lesson **C2**.
    """
    lags = range(1, max_lag + 1)
    result: dict[str, Any] = {"max_lag": max_lag, "cosine_by_lag": {}}
    for lag in lags:
        sims: list[float] = []
        for array in features.values():
            if array.shape[0] <= lag:
                continue
            head, tail = array[:-lag], array[lag:]
            numerator = (head * tail).sum(axis=1)
            denominator = np.linalg.norm(head, axis=1) * np.linalg.norm(tail, axis=1)
            valid = denominator > 0
            sims.extend((numerator[valid] / denominator[valid]).tolist())
        result["cosine_by_lag"][f"lag{lag}"] = (
            {"mean": float(np.mean(sims)), "std": float(np.std(sims)), "pairs": len(sims)}
            if sims else {"mean": None, "std": None, "pairs": 0}
        )
    return result


def variance_decomposition(features: dict[str, np.ndarray]) -> dict[str, Any]:
    """Split total feature variance into between-clip and within-clip parts.

    A high between/within ratio means the embedding mostly encodes *which scene
    this is* rather than *what is happening in it* — the representational version
    of the flat-curve problem :mod:`core.eda.protocol` measures on model outputs.
    """
    usable = {k: v for k, v in features.items() if v.shape[0] > 0}
    if not usable:
        return {"clips": 0}
    means = np.stack([v.mean(axis=0) for v in usable.values()])
    sizes = np.array([v.shape[0] for v in usable.values()], dtype=np.float64)
    within = np.stack([v.var(axis=0) for v in usable.values()])
    within_pooled = float((within * sizes[:, None]).sum() / sizes.sum() / within.shape[1])
    between = float(means.var(axis=0).mean())
    return {
        "clips": len(usable),
        "between_clip_variance": between,
        "within_clip_variance": within_pooled,
        "between_over_within": between / within_pooled if within_pooled > 0 else None,
    }


def _probe_fit_predict(
    matrix: np.ndarray, target: np.ndarray, groups: np.ndarray, folds: int, seed: int
) -> np.ndarray:
    """Out-of-fold decision scores from a grouped-CV logistic probe.

    Grouping is by clip: a frame never appears in the fold that scored it, so a
    clip's own frames cannot leak across the split.
    """
    predictions = np.zeros(target.shape[0], dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for train_idx, test_idx in splitter.split(matrix, target, groups):
        if len(np.unique(target[train_idx])) < 2:
            LOGGER.warning("A probe fold has a single class; its scores stay at 0")
            continue
        scaler = StandardScaler().fit(matrix[train_idx])
        model = LogisticRegression(
            C=constants.EDA_PROBE_C,
            max_iter=constants.EDA_PROBE_MAX_ITER,
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(scaler.transform(matrix[train_idx]), target[train_idx])
        predictions[test_idx] = model.decision_function(scaler.transform(matrix[test_idx]))
    return predictions


def _source_groups(ids: list[str], slicer: FeatureSlicer | None) -> list[int]:
    """One integer per id, shared by every item that came from the same clip."""
    slicer = slicer if slicer is not None else FeatureSlicer()
    index: dict[str, int] = {}
    groups: list[int] = []
    for item_id in ids:
        source = slicer.source_of(item_id)
        groups.append(index.setdefault(source, len(index)))
    return groups


def frame_linear_probe(
    features: dict[str, np.ndarray],
    frame_labels: dict[str, list[int]],
    folds: int = constants.EDA_PROBE_FOLDS,
    seed: int = constants.SEED,
    slicer: FeatureSlicer | None = None,
) -> dict[str, Any]:
    """Supervised frame-level ceiling of the cached features (``RESULTS_DADA`` §10-B).

    Grouped cross-validation over clips, logistic regression on standardized
    features, scored with the same micro/macro pair the model eval uses so the
    numbers are directly comparable to an arm's ``results.json``.

    The fold group is the **source clip**, not the item. On a windowed corpus two
    overlapping windows of one clip share real frames (12 of 24 at the default
    50 % hop), so grouping by window id would train and test on the same rows and
    inflate the very ceiling this probe exists to establish -- the number the
    backbone decision hangs on (``DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md`` §7).
    """
    ids = [v for v in sorted(features) if v in frame_labels and len(frame_labels[v]) > 0]
    if len(ids) < constants.EDA_PROBE_MIN_CLIPS:
        return {"ran": False, "reason": f"only {len(ids)} clips with features and labels"}
    matrix = np.concatenate([features[v] for v in ids], axis=0).astype(np.float64)
    target = np.concatenate(
        [np.asarray(frame_labels[v], dtype=np.int64) for v in ids]
    )
    groups = np.concatenate(
        [np.full(len(frame_labels[v]), g, dtype=np.int64)
         for v, g in zip(ids, _source_groups(ids, slicer), strict=True)]
    )
    if len(np.unique(target)) < 2:
        return {"ran": False, "reason": "test frames are single-class"}
    usable_folds = min(folds, len(np.unique(groups)))
    scores = _probe_fit_predict(matrix, target, groups, usable_folds, seed)

    per_clip: list[float] = []
    offset = 0
    for video_id in ids:
        length = len(frame_labels[video_id])
        clip_gt = target[offset:offset + length]
        clip_scores = scores[offset:offset + length]
        if 0 < int(clip_gt.sum()) < length:
            per_clip.append(frame_auc(clip_scores, clip_gt))
        offset += length
    return {
        "ran": True,
        "folds": usable_folds,
        "clips": len(ids),
        "frames": int(target.size),
        "positive_frames": int(target.sum()),
        "auc_micro": frame_auc(scores, target),
        "ap_micro": frame_ap(scores, target),
        "ap_baseline": float(target.mean()),
        "auc_macro": float(np.mean(per_clip)) if per_clip else None,
        "auc_macro_videos": len(per_clip),
        "interpretation": (
            "auc_macro well above 0.5 means the frozen features DO carry a "
            "frame-level accident signal and the deficit is supervision; at 0.5 "
            "the representation itself is the ceiling (RESULTS_DADA.md §10-B)"
        ),
    }


def clip_linear_probe(
    features: dict[str, np.ndarray],
    frame_labels: dict[str, list[int]],
    folds: int = constants.EDA_PROBE_FOLDS,
    seed: int = constants.SEED,
    slicer: FeatureSlicer | None = None,
) -> dict[str, Any]:
    """The same probe on mean-pooled clips against clip labels — the contrast arm.

    A high clip AUC beside a chance frame AUC is the representational form of the
    result :mod:`core.eda.protocol` finds in the metric: the corpus supports video
    classification and not localization.
    """
    ids = [v for v in sorted(features) if v in frame_labels and len(frame_labels[v]) > 0]
    if len(ids) < constants.EDA_PROBE_MIN_CLIPS:
        return {"ran": False, "reason": f"only {len(ids)} clips with features and labels"}
    matrix = np.stack([features[v].mean(axis=0) for v in ids]).astype(np.float64)
    target = np.array([int(any(frame_labels[v])) for v in ids], dtype=np.int64)
    if len(np.unique(target)) < 2:
        return {"ran": False, "reason": "every clip has the same clip-level label"}
    groups = np.asarray(_source_groups(ids, slicer), dtype=np.int64)
    usable_folds = min(folds, int(np.bincount(target).min()), len(np.unique(groups)))
    if usable_folds < 2:
        return {"ran": False, "reason": "too few clips in the minority class"}
    scores = _probe_fit_predict(matrix, target, groups, usable_folds, seed)
    return {
        "ran": True,
        "folds": usable_folds,
        "clips": len(ids),
        "abnormal_clips": int(target.sum()),
        "auc": frame_auc(scores, target),
        "ap": frame_ap(scores, target),
        "ap_baseline": float(target.mean()),
    }


def flow_stats(
    flow_dir: Path,
    video_ids: list[str],
    limit: int = 0,
    slicer: FeatureSlicer | None = None,
) -> dict[str, Any]:
    """Distribution of the cached RAFT descriptors.

    ``{id}.stats.npy`` is ``(L, 23)`` raw frame-global scalars; ``{id}.npy`` is
    those 23 numbers lifted to 256-d by a fixed seeded projection. There are **23
    effective dimensions** and no spatial content — never read localization into
    them. Flow is a *train-time* cache, and the train split carries no frame
    labels, so this block is distributional only: no flow-versus-label
    correlation is computable from what is on disk.
    """
    candidates = video_ids[:limit] if limit else video_ids
    slicer = slicer if slicer is not None else FeatureSlicer()
    rows: list[np.ndarray] = []
    found = 0
    for video_id in candidates:
        path = flow_dir / f"{slicer.source_of(video_id)}{FLOW_STATS_SUFFIX}"
        if not path.is_file():
            continue
        found += 1
        rows.append(
            np.asarray(slicer.load(flow_dir, video_id, FLOW_STATS_SUFFIX), dtype=np.float64)
        )
    if not rows:
        return {"clips": 0, "note": f"no {FLOW_STATS_SUFFIX} files under {flow_dir}"}
    stacked = np.concatenate(rows, axis=0)
    return {
        "clips": found,
        "frames": int(stacked.shape[0]),
        "raw_dims": int(stacked.shape[1]),
        "per_dim_mean": [float(x) for x in stacked.mean(axis=0)],
        "per_dim_std": [float(x) for x in stacked.std(axis=0)],
        "dead_dims": int((stacked.std(axis=0) == 0).sum()),
        "nonfinite_frames": int((~np.isfinite(stacked)).any(axis=1).sum()),
        "note": "23 effective dimensions, frame-global, no spatial content",
    }


def feature_report(
    files: DatasetFiles,
    clip_dir: Path,
    flow_dir: Path | None = None,
    max_clips: int = 0,
    probe: bool = True,
    seed: int = constants.SEED,
) -> dict[str, Any]:
    """Everything in this module, for one dataset's test split."""
    ids = files.test_ids[:max_clips] if max_clips else files.test_ids
    expected = {v: len(files.frame_labels_test[v]) for v in ids}
    features, missing = load_clip_features(clip_dir, ids, expected, files.slicer)
    report: dict[str, Any] = {
        "clip_dir": str(clip_dir),
        "coverage": {
            "requested": len(ids),
            "found": len(features),
            "missing": len(missing),
            "missing_examples": missing[:10],
        },
        "stats": feature_stats(features),
        "temporal_autocorrelation": temporal_autocorrelation(features),
        "variance_decomposition": variance_decomposition(features),
    }
    if probe and features:
        report["frame_linear_probe"] = frame_linear_probe(
            features, files.frame_labels_test, seed=seed, slicer=files.slicer
        )
        report["clip_linear_probe"] = clip_linear_probe(
            features, files.frame_labels_test, seed=seed, slicer=files.slicer
        )
    if flow_dir is not None:
        report["flow"] = flow_stats(
            flow_dir, files.train_ids, limit=max_clips, slicer=files.slicer
        )
    return report


__all__ = [
    "clip_linear_probe",
    "feature_report",
    "feature_stats",
    "flow_stats",
    "frame_linear_probe",
    "load_clip_features",
    "temporal_autocorrelation",
    "variance_decomposition",
]
