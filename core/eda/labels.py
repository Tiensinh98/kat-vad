"""Label geometry: where the positives are, and which windows survived the stride.

Two measurements here fed defects in ``core/docs/v3/RESULTS_DADA.md``:

* :func:`vanished_windows` finds abnormal clips whose anomaly span rounded away
  to nothing at the sampling stride. ``build_frame_labels`` only *warns* about
  them unless ``--strict`` is passed, and ``--strict`` **raises** rather than
  repairing, so without this audit they sit in the test split with an all-zero
  label vector and read as genuine normal clips (§8.3, ``DADA_SETUP.md`` §5.1).
* :func:`span_stats` measures how many positives a per-clip AUC actually has to
  work with. On DADA-2000 the median abnormal test clip has 2 positive frames,
  which caps per-clip AUC resolution at roughly 1/14.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.eda.corpus import DatasetFiles, describe

LOGGER = logging.getLogger(__name__)

ABNORMAL_META_KEYS = ("normalized_span", "anomaly_span")


def find_spans(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of 1 in a frame-label vector, as half-open ``[start, end)``."""
    if labels.size == 0:
        return []
    padded = np.concatenate(([0], labels.astype(np.int8), [0]))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def positives_per_clip(files: DatasetFiles) -> dict[str, Any]:
    """Positive-frame counts, over all test clips and over abnormal ones only."""
    per_clip = {v: int(sum(files.frame_labels_test[v])) for v in files.test_ids}
    abnormal = [n for n in per_clip.values() if n > 0]
    fractions = [
        sum(files.frame_labels_test[v]) / len(files.frame_labels_test[v])
        for v in files.test_ids
        if len(files.frame_labels_test[v]) > 0 and sum(files.frame_labels_test[v]) > 0
    ]
    return {
        "all_clips": describe(list(per_clip.values()), "positive frames per test clip"),
        "abnormal_clips": describe(abnormal, "positive frames per abnormal test clip"),
        "positive_fraction_within_abnormal": describe(
            fractions, "positive fraction within abnormal clips"
        ),
        "abnormal_clips_with_one_positive": int(sum(1 for n in abnormal if n == 1)),
        "abnormal_clips_with_two_or_fewer": int(sum(1 for n in abnormal if n <= 2)),
    }


def span_stats(files: DatasetFiles) -> dict[str, Any]:
    """Anomaly-span count and length per abnormal test clip.

    Multi-span clips matter because ``core/data/dota.py`` fills exactly one span
    while PreVAD ships up to four (lesson **C18**); a corpus that turns out to be
    multi-span here would need the same treatment.
    """
    counts: list[int] = []
    lengths: list[int] = []
    multi: list[str] = []
    for video_id in files.test_ids:
        spans = find_spans(np.asarray(files.frame_labels_test[video_id], dtype=np.int8))
        if not spans:
            continue
        counts.append(len(spans))
        lengths.extend(end - start for start, end in spans)
        if len(spans) > 1:
            multi.append(video_id)
    return {
        "spans_per_abnormal_clip": describe(counts, "spans per abnormal test clip"),
        "span_length": describe(lengths, "span length (sampled frames)"),
        "multi_span_clips": len(multi),
        "multi_span_examples": sorted(multi)[:10],
    }


def vanished_windows(files: DatasetFiles) -> dict[str, Any]:
    """Abnormal **source clips** whose label vector is all zero everywhere.

    These entered the split through ``build_frame_labels``' default warn-and-continue
    path. They are not neutral: on a test set already dominated by all-normal
    clips they are counted as normal by every metric, inflating both micro AUC and
    the constant-score clip oracle (:mod:`core.eda.protocol`).

    **On a windowed corpus the unit is the source clip, not the item.** A window of
    an abnormal clip that holds no positive frame is a *correct negative window* --
    producing them is the point of re-sharding (an abnormal clip's normal stretch
    becomes genuine negatives). Flagging those would report the feature as a
    defect; the real defect is an abnormal clip **none** of whose windows carries
    the anomaly, which means the span rounded away at this stride.
    """
    by_source: dict[str, list[str]] = {}
    abnormal_sources: set[str] = set()
    unknown_flag = 0
    for video_id in files.test_ids:
        entry = files.meta.get(video_id, {})
        span = next((entry[k] for k in ABNORMAL_META_KEYS if entry.get(k) is not None), None)
        declared_abnormal = span is not None
        if span is None and entry.get("class_name") not in (None, "Normal"):
            # A corpus that records no span but a non-Normal class still counts.
            declared_abnormal = True
            unknown_flag += 1
        source = files.source_of(video_id)
        by_source.setdefault(source, []).append(video_id)
        if declared_abnormal:
            abnormal_sources.add(source)

    vanished = sorted(
        source for source in abnormal_sources
        if not any(any(files.frame_labels_test[v]) for v in by_source[source])
    )
    negative_windows = 0
    if files.is_windowed:
        negative_windows = sum(
            1
            for source in abnormal_sources - set(vanished)
            for v in by_source[source]
            if not any(files.frame_labels_test[v])
        )
    if vanished:
        LOGGER.warning(
            "%d abnormal test %s carry an all-zero label vector; they read as "
            "normal to every metric. Exclude them at scoring time -- see "
            "core/docs/DADA_SETUP.md §5.1. First few: %s",
            len(vanished), "source clips" if files.is_windowed else "clips", vanished[:5],
        )
    if negative_windows:
        LOGGER.info(
            "%d windows of abnormal clips hold no positive frame -- these are "
            "correct negatives, not vanished windows (that is what re-sharding is for)",
            negative_windows,
        )
    return {
        "count": len(vanished),
        "video_ids": vanished,
        "clips_without_explicit_span_field": unknown_flag,
        "unit": "source clip" if files.is_windowed else "clip",
        "negative_windows_of_abnormal_clips": negative_windows,
    }


def label_report(files: DatasetFiles) -> dict[str, Any]:
    """Everything in this module, for one dataset."""
    return {
        "positives": positives_per_clip(files),
        "spans": span_stats(files),
        "vanished_windows": vanished_windows(files),
    }


__all__ = [
    "find_spans",
    "label_report",
    "positives_per_clip",
    "span_stats",
    "vanished_windows",
]
