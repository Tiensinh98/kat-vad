"""The on-disk dataset contract shared by every preprocessor (spec §7).

Each dataset preprocessor -- :mod:`core.data.msad`, :mod:`core.data.tad` -- ends
by writing the same four files under ``data/{DATASET}/`` (see
``core/docs/DATA_LAYOUT.md``)::

    labels_train.json        {video_id: 0|1}          video-level, weak supervision
    frame_labels_test.json   {video_id: [0,1,...]}    per SAMPLED frame
    defs.json                [class_name, ...]        Normal first
    meta.json                {video_id: {...}}        diagnostics only

Only the *derivation* of those payloads is dataset-specific; the layout,
frame-sampling arithmetic and JSON conventions live here so the two
preprocessors cannot drift apart.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core import constants

LOGGER = logging.getLogger(__name__)

NORMAL_CLASS = "Normal"  # first entry of defs.json for every dataset
TEST_IDS_FILENAME = "test_ids.txt"  # scored ids, for extract_clip_features --ids-file
TRAIN_IDS_FILENAME = "train_ids.txt"  # train ids, for raft_extract --ids-file (flow is train-only)


def class_name_list(class_names: set[str]) -> list[str]:
    """Global class-name list for defs.json: ``Normal`` first, then sorted rest."""
    rest = set(class_names)
    rest.discard(NORMAL_CLASS)
    return [NORMAL_CLASS, *sorted(rest)]


def num_sampled_frames(total_frames: int, stride: int) -> int:
    """Number of frames produced by ``range(0, total_frames, stride)`` sampling.

    The single source of truth for label/feature alignment: the CLIP and RAFT
    extractors sample the same raw indices (:mod:`core.data.video_io`), so a
    label vector of this length lines up row-for-row with ``{id}.npy``.
    """
    return (total_frames + stride - 1) // stride


def write_dataset_files(
    out_dir: Path,
    labels_train: dict[str, int],
    frame_labels_test: dict[str, list[int]],
    defs: list[str],
    meta: dict[str, Any],
) -> None:
    """Write the four standard dataset files into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads: tuple[tuple[str, object], ...] = (
        (constants.LABELS_TRAIN_FILENAME, labels_train),
        (constants.FRAME_LABELS_TEST_FILENAME, frame_labels_test),
        (constants.DEFS_FILENAME, defs),
        (constants.META_FILENAME, meta),
    )
    for filename, payload in payloads:
        target = out_dir / filename
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        LOGGER.info("Wrote %s", target)


def _write_ids(target: Path, video_ids: list[str]) -> Path:
    """Write one video id per line, newline-terminated."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{v}\n" for v in video_ids), encoding="utf-8")
    LOGGER.info("Wrote %s (%d ids)", target, len(video_ids))
    return target


def write_test_ids(out_dir: Path, video_ids: list[str]) -> Path:
    """Emit the scored video ids for ``extract_clip_features --ids-file``."""
    return _write_ids(out_dir / TEST_IDS_FILENAME, video_ids)


def write_train_ids(out_dir: Path, video_ids: list[str]) -> Path:
    """Emit the train video ids for ``raft_extract --ids-file``.

    Flow targets are train-time only (spec §1), so extracting ``e_O`` for the
    scored split is wasted GPU hours on a dataset whose train split dominates
    the corpus (TAD: 410 of 510 videos).
    """
    return _write_ids(out_dir / TRAIN_IDS_FILENAME, video_ids)


__all__ = [
    "NORMAL_CLASS",
    "TEST_IDS_FILENAME",
    "TRAIN_IDS_FILENAME",
    "class_name_list",
    "num_sampled_frames",
    "write_dataset_files",
    "write_test_ids",
    "write_train_ids",
]
