"""Corpus shape: how many clips, how long, and what the model can see of them.

The clip-length distribution is not a curiosity on this project. Lesson **C27**
was written because ``ConvScoreHead`` is a single ``Conv1d(kernel_size=9)`` and
DADA-2000's median clip is 9 sampled frames, which makes every output timestep a
function of the whole clip — a clip classifier wearing a frame detector's API.
:func:`kernel_coverage` and :func:`mil_topk_floor` compute that collapse
directly from the label files, before a single GPU-hour is spent.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import constants

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetFiles:
    """The four on-disk dataset files, loaded (``core/docs/DATA_LAYOUT.md``)."""

    dataset: str
    data_dir: Path
    labels_train: dict[str, int]
    frame_labels_test: dict[str, list[int]]
    defs: list[str]
    meta: dict[str, dict[str, Any]]

    @property
    def test_ids(self) -> list[str]:
        return sorted(self.frame_labels_test)

    @property
    def train_ids(self) -> list[str]:
        return sorted(self.labels_train)

    def test_label_arrays(self) -> list[np.ndarray]:
        """Per-clip frame labels as ``int8`` arrays, in :attr:`test_ids` order."""
        return [np.asarray(self.frame_labels_test[v], dtype=np.int8) for v in self.test_ids]


def load_dataset_files(data_dir: Path, dataset: str) -> DatasetFiles:
    """Load the four standard dataset files; raise if any is missing."""

    def read(filename: str) -> Any:
        target = data_dir / filename
        if not target.is_file():
            raise FileNotFoundError(
                f"{target} not found. Build it first: DADA-2000 -> "
                "core/docs/DADA_SETUP.md §5; DoTA -> core/docs/DOTA_EVAL.md."
            )
        with target.open(encoding="utf-8") as fh:
            payload: Any = json.load(fh)
        return payload

    files = DatasetFiles(
        dataset=dataset,
        data_dir=data_dir,
        labels_train=read(constants.LABELS_TRAIN_FILENAME),
        frame_labels_test=read(constants.FRAME_LABELS_TEST_FILENAME),
        defs=read(constants.DEFS_FILENAME),
        meta=read(constants.META_FILENAME),
    )
    LOGGER.info(
        "%s: %d train / %d test clips, %d classes",
        dataset, len(files.labels_train), len(files.frame_labels_test), len(files.defs),
    )
    return files


def describe(values: list[int] | list[float] | np.ndarray, name: str) -> dict[str, Any]:
    """Percentile summary of one distribution. Empty input yields nulls, not a raise."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"name": name, "n": 0, "mean": None, "std": None, "percentiles": {}}
    return {
        "name": name,
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "percentiles": {
            f"p{p}": float(np.percentile(array, p)) for p in constants.EDA_PERCENTILES
        },
    }


def split_sizes(files: DatasetFiles) -> dict[str, Any]:
    """Clip and frame counts per split, with the train-split class balance."""
    train_abnormal = sum(1 for v in files.labels_train.values() if v)
    test_lengths = [len(v) for v in files.frame_labels_test.values()]
    test_abnormal = sum(1 for v in files.frame_labels_test.values() if any(v))
    overlap = sorted(set(files.labels_train) & set(files.frame_labels_test))
    if overlap:
        LOGGER.error("SPLIT LEAK: %d ids in both splits, e.g. %s", len(overlap), overlap[:5])
    return {
        "train_clips": len(files.labels_train),
        "train_abnormal": train_abnormal,
        "train_normal": len(files.labels_train) - train_abnormal,
        "train_abnormal_fraction": (
            train_abnormal / len(files.labels_train) if files.labels_train else None
        ),
        "test_clips": len(files.frame_labels_test),
        "test_abnormal": test_abnormal,
        "test_normal": len(files.frame_labels_test) - test_abnormal,
        "test_sampled_frames": int(sum(test_lengths)),
        "test_positive_frames": int(sum(sum(v) for v in files.frame_labels_test.values())),
        "split_leak_ids": overlap,
        # DVSFeatureDataset.__len__ is 2 x abnormal-train; the training budget
        # follows from it (DADA_SETUP.md §10), so it belongs in the report.
        "dvs_dataset_len": 2 * train_abnormal,
        "steps_per_epoch_at_batch_64": (
            int(np.ceil(2 * train_abnormal / constants.BATCH_SIZE)) if train_abnormal else 0
        ),
    }


def length_distribution(files: DatasetFiles) -> dict[str, Any]:
    """Sampled-clip-length distribution for the test split and, from ``meta``, train."""
    test_lengths = [len(files.frame_labels_test[v]) for v in files.test_ids]
    train_lengths = [
        int(entry["sampled_frames"])
        for vid, entry in files.meta.items()
        if entry.get("split") == "train" and entry.get("sampled_frames") is not None
    ]
    raw_lengths = [
        int(entry["total_frames"])
        for entry in files.meta.values()
        if entry.get("total_frames") is not None
    ]
    return {
        "test_sampled": describe(test_lengths, "test sampled frames per clip"),
        "train_sampled": describe(train_lengths, "train sampled frames per clip"),
        "raw_total_frames": describe(raw_lengths, "raw frames per clip (pre-stride)"),
    }


def kernel_coverage(
    lengths: list[int], kernel: int = constants.SCORE_HEAD_KERNEL
) -> dict[str, Any]:
    """How much of each clip one score-head output timestep sees (lesson **C27**).

    ``ConvScoreHead`` is a single ``Conv1d(kernel_size=kernel)`` with replicate
    padding, so one output frame is a function of ``kernel`` input frames. When
    ``kernel >= T`` every output timestep sees the entire clip and the head is a
    clip-pooling operator, whatever its output shape suggests.
    """
    if not lengths:
        return {"kernel": kernel, "clips": 0}
    array = np.asarray(lengths, dtype=np.float64)
    coverage = np.minimum(1.0, kernel / array)
    return {
        "kernel": kernel,
        "clips": int(array.size),
        "median_length": float(np.median(array)),
        "median_coverage": float(np.median(coverage)),
        "mean_coverage": float(coverage.mean()),
        "clips_fully_covered": int((array <= kernel).sum()),
        "fraction_fully_covered": float((array <= kernel).mean()),
        "frames_in_fully_covered_clips": int(array[array <= kernel].sum()),
        "fraction_frames_fully_covered": float(array[array <= kernel].sum() / array.sum()),
        "clips_at_or_below": {
            f"T<={t}": int((array <= t).sum()) for t in constants.EDA_SHORT_CLIP_THRESHOLDS
        },
    }


def mil_topk_floor(
    lengths: list[int], topk_pct: int = constants.MIL_TOPK_PCT
) -> dict[str, Any]:
    """Distribution of ``k = max(1, T // topk_pct)``, the MIL top-k per clip.

    ``core/losses/mil.py`` degenerates to a plain max whenever ``k == 1``: the
    clip contributes exactly one frame of supervision per step regardless of how
    many frames it has.
    """
    if not lengths:
        return {"topk_pct": topk_pct, "clips": 0}
    array = np.asarray(lengths, dtype=np.int64)
    k = np.maximum(1, array // topk_pct)
    values, counts = np.unique(k, return_counts=True)
    return {
        "topk_pct": topk_pct,
        "clips": int(array.size),
        "k_distribution": {
            str(int(v)): int(c) for v, c in zip(values, counts, strict=True)
        },
        "clips_at_k1": int((k == 1).sum()),
        "fraction_at_k1": float((k == 1).mean()),
        "median_k": float(np.median(k)),
    }


def subgroup_table(files: DatasetFiles, key: str) -> dict[str, Any]:
    """Clip / frame / positive-frame counts grouped by one ``meta.json`` field.

    Returns an empty payload when the field is absent, so one report function
    serves DADA-2000 (``fault_label``) and DoTA (``anomaly_class``) alike.
    """
    present = [v for v, entry in files.meta.items() if key in entry and entry[key] is not None]
    if not present:
        return {"key": key, "present": False, "groups": {}}
    groups: dict[str, dict[str, Any]] = {}
    counter = Counter(str(files.meta[v][key]) for v in present)
    for group in sorted(counter):
        members = [v for v in present if str(files.meta[v][key]) == group]
        test_members = [v for v in members if v in files.frame_labels_test]
        frames = sum(len(files.frame_labels_test[v]) for v in test_members)
        positives = sum(sum(files.frame_labels_test[v]) for v in test_members)
        groups[group] = {
            "clips": len(members),
            "test_clips": len(test_members),
            "train_clips": len(members) - len(test_members),
            "test_frames": int(frames),
            "test_positive_frames": int(positives),
            "test_positive_fraction": float(positives / frames) if frames else None,
        }
    return {"key": key, "present": True, "num_groups": len(groups), "groups": groups}


def corpus_report(files: DatasetFiles, kernel: int, topk_pct: int) -> dict[str, Any]:
    """Everything in this module, for one dataset."""
    test_lengths = [len(files.frame_labels_test[v]) for v in files.test_ids]
    return {
        "splits": split_sizes(files),
        "lengths": length_distribution(files),
        "kernel_coverage_test": kernel_coverage(test_lengths, kernel),
        "mil_topk_floor_test": mil_topk_floor(test_lengths, topk_pct),
        "subgroups": {
            key: subgroup_table(files, key)
            for key in ("fault_label", "anomaly_class", "ego_involve", "class_name", "split")
        },
        "classes": files.defs,
    }


__all__ = [
    "DatasetFiles",
    "corpus_report",
    "describe",
    "kernel_coverage",
    "length_distribution",
    "load_dataset_files",
    "mil_topk_floor",
    "split_sizes",
    "subgroup_table",
]
