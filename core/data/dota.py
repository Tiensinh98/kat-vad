"""DoTA preprocessor: official val split -> the standard dataset files.

DoTA ships extracted frames, not videos. After unzipping ``DoTA_full.zip``::

    frames/{video_id}/images/000000.jpg ...

and two annotation files travel in ``data/DoTA/``: ``metadata_val.json``
(1,402 clips) and ``val_split.txt`` (the same ids, one per line)::

    {video_id: {"video_start": int, "video_end": int, "anomaly_start": int,
                "anomaly_end": int, "anomaly_class": str, "num_frames": int,
                "subset": "val"}}

``anomaly_start``/``anomaly_end`` are **clip-relative, half-open** frame
indices at DoTA's native 10 fps.

This module is **evaluation-only**: DoTA's train split carries the same
annotation but the KAT-VAD DoTA protocol is zero-shot transfer, so
``labels_train.json`` is written empty and every split id lands in the test set.

**Label parity with the baseline.** LaGoVAD's ``dota_test_anno.json`` stores
``anomaly_span`` as normalized ``[start/num_frames, end/num_frames]`` and
rebuilds frame labels as ``round(frac x feature_length)`` filled half-open
(``LaGoVAD-PreVAD/src/datasets/base.py:57-69``). This module reproduces that
arithmetic exactly -- verified equal on all 1,402 val clips -- so our AUC is
measured against the same ground truth as the published DoTA 62.60.

Every DoTA clip is abnormal (each is an anomaly clip with a normal lead-in), so
frame-level AUC is pooled over clips, never over a normal-video pool: the
positive base rate is ~33% at any stride.

CLI::

    python -m core.data.dota --metadata data/DoTA/metadata_val.json
        --split-file data/DoTA/val_split.txt --out-dir data/DoTA
        [--frames-dir data/DoTA/frames --frames-subdir images]
        [--stride 8] [--strict] [--allow-missing-frames] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import constants
from core.data.dataset_files import (
    class_name_list,
    num_sampled_frames,
    write_dataset_files,
    write_test_ids,
)
from core.data.video_io import list_frame_folders, list_frame_images, video_id_from_path

LOGGER = logging.getLogger(__name__)

DOTA_CLASS_NAME = "CarAccident"  # the definition key DoTA/DADA share (spec §7.4)
EGO_PREFIX = "ego:"
NUM_FRAMES_KEY = "num_frames"
ANOMALY_START_KEY = "anomaly_start"
ANOMALY_END_KEY = "anomaly_end"
ANOMALY_CLASS_KEY = "anomaly_class"


@dataclass(frozen=True)
class DotaRecord:
    """One annotated DoTA val clip.

    ``total_frames`` is the clip length used for *sampling* (the on-disk image
    count when ``--frames-dir`` is given, else the annotation's ``num_frames``);
    ``span`` is the normalized anomaly window, which is invariant to that
    choice and is what the baseline stores.
    """

    video_id: str
    anomaly_class: str
    total_frames: int
    span: tuple[float, float]

    @property
    def is_ego(self) -> bool:
        """DoTA marks ego-involved anomalies with an ``ego:`` class prefix."""
        return self.anomaly_class.startswith(EGO_PREFIX)


def read_split_ids(path: Path) -> list[str]:
    """Video ids from ``val_split.txt``, order preserved, blanks dropped."""
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [video_id for video_id in ids if video_id]
    if not ids:
        raise ValueError(f"{path} contains no video ids")
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise ValueError(f"{path} has {duplicates} duplicate ids")
    LOGGER.info("Read %d split ids from %s", len(ids), path)
    return ids


def parse_metadata(path: Path, split_ids: list[str]) -> list[DotaRecord]:
    """Build one record per split id from ``metadata_val.json``."""
    with path.open("r", encoding="utf-8") as fh:
        metadata: dict[str, dict[str, Any]] = json.load(fh)
    missing = [video_id for video_id in split_ids if video_id not in metadata]
    if missing:
        raise ValueError(f"{len(missing)} split ids absent from {path}: {missing[:5]}")
    records = [
        _record_from_entry(video_id, metadata[video_id]) for video_id in split_ids
    ]
    LOGGER.info("Parsed %d annotated clips from %s", len(records), path)
    return sorted(records, key=lambda r: r.video_id)


def _record_from_entry(video_id: str, entry: dict[str, Any]) -> DotaRecord:
    """Validate one metadata entry and normalize its anomaly window."""
    total_frames = int(entry[NUM_FRAMES_KEY])
    start = int(entry[ANOMALY_START_KEY])
    end = int(entry[ANOMALY_END_KEY])
    if total_frames <= 0:
        raise ValueError(f"{video_id}: num_frames must be positive, got {total_frames}")
    if not 0 <= start < end <= total_frames:
        raise ValueError(
            f"{video_id}: anomaly window [{start}, {end}) is not a non-empty "
            f"half-open interval inside [0, {total_frames})"
        )
    return DotaRecord(
        video_id=video_id,
        anomaly_class=str(entry[ANOMALY_CLASS_KEY]),
        total_frames=total_frames,
        span=(start / total_frames, end / total_frames),
    )


def resolve_frame_counts(
    records: list[DotaRecord],
    frames_dir: Path,
    subdir: str | None,
    allow_missing: bool = False,
) -> list[DotaRecord]:
    """Replace annotation frame counts with the image counts actually on disk.

    The extractor reads those images, so their count -- not the annotation's --
    determines the feature length that labels must match. Any disagreement is
    logged: it means the unzip is incomplete or the clip was re-extracted at a
    different fps, and it silently shifts the anomaly window if ignored.

    A clip the extractor cannot read -- no frame folder at all, or a folder whose
    image directory is empty -- is fatal by default. Both mean a truncated
    unzip, and quietly evaluating on whatever survived would move the
    denominator without saying so. ``allow_missing`` drops those clips instead,
    counting each cause separately, for deliberate partial-coverage runs (see
    ``core/docs/DOTA_EVAL.md`` §3.6).
    """
    folders = {video_id_from_path(p): p for p in list_frame_folders(frames_dir, subdir)}
    annotated = {r.video_id for r in records}
    unannotated = len(folders.keys() - annotated)
    if unannotated:
        LOGGER.info(
            "%d frame folders under %s are not in the split -- ignored",
            unannotated,
            frames_dir,
        )
    resolved: list[DotaRecord] = []
    absent: list[str] = []
    empty: list[str] = []
    mismatched = 0
    for record in records:
        folder = folders.get(record.video_id)
        if folder is None:
            absent.append(record.video_id)
            continue
        on_disk = len(list_frame_images(folder, subdir))
        if on_disk == 0:
            empty.append(record.video_id)
            continue
        if on_disk != record.total_frames:
            mismatched += 1
            LOGGER.warning(
                "%s: %d images on disk but annotation says %d frames; using disk",
                record.video_id,
                on_disk,
                record.total_frames,
            )
        resolved.append(
            DotaRecord(
                video_id=record.video_id,
                anomaly_class=record.anomaly_class,
                total_frames=on_disk,
                span=record.span,
            )
        )
    if mismatched:
        LOGGER.warning(
            "%d/%d clips disagree with the annotation frame count",
            mismatched,
            len(records),
        )
    if absent or empty:
        _report_unreadable(records, absent, empty, frames_dir, subdir, allow_missing)
    if not resolved:
        raise ValueError(f"No annotated clip has readable frames under {frames_dir}")
    return resolved


def _report_unreadable(
    records: list[DotaRecord],
    absent: list[str],
    empty: list[str],
    frames_dir: Path,
    subdir: str | None,
    allow_missing: bool,
) -> None:
    """Raise or warn about clips the extractor cannot read, cause by cause."""
    message = (
        f"{len(absent) + len(empty)}/{len(records)} annotated clips are "
        f"unreadable under {frames_dir} (subdir={subdir!r}): "
        f"{len(absent)} with no frame folder {sorted(absent)[:5]}, "
        f"{len(empty)} with an empty one {sorted(empty)[:5]}"
    )
    if not allow_missing:
        raise ValueError(
            f"{message} -- finish the unzip (`unzip -n` resumes), or pass "
            "--allow-missing-frames to evaluate on the clips that are present "
            "(the absolute AUC is then no longer comparable to the published "
            "number)"
        )
    LOGGER.warning("%s -- dropped from the evaluation set", message)


def sampled_frame_labels(record: DotaRecord, stride: int) -> list[int]:
    """Per-sampled-frame 0/1 labels from the normalized span.

    Mirrors ``LaGoVAD-PreVAD/src/datasets/base.py``: scale both fractions by the
    sampled length, round, fill the half-open interval. At ``stride=1`` this is
    exactly the raw ``[anomaly_start, anomaly_end)`` window.
    """
    length = num_sampled_frames(record.total_frames, stride)
    labels = [0] * length
    start = max(0, min(length, round(record.span[0] * length)))
    end = max(0, min(length, round(record.span[1] * length)))
    for index in range(start, end):
        labels[index] = 1
    return labels


def build_frame_labels(
    records: list[DotaRecord], stride: int, strict: bool = False
) -> dict[str, list[int]]:
    """Frame labels for every record, warning on windows that round away.

    An anomaly shorter than one sampled step collapses to an all-normal clip,
    which then only contributes negatives. The baseline's own label code does
    the same thing silently, so the default is to match it and report the count;
    ``strict`` refuses instead. At ``stride=8``, 5 of the 1,402 val clips
    collapse (anomaly windows of 1-7 raw frames); at ``stride=1``, none do.
    """
    frame_labels = {r.video_id: sampled_frame_labels(r, stride) for r in records}
    vanished = sorted(r.video_id for r in records if not any(frame_labels[r.video_id]))
    if vanished:
        message = (
            f"{len(vanished)} clips lose their anomaly window at stride {stride} "
            f"(window shorter than one sampled frame): {vanished[:5]}"
        )
        if strict:
            raise ValueError(message)
        LOGGER.warning("%s -- they contribute only negative frames", message)
    return frame_labels


def build_meta(records: list[DotaRecord], stride: int) -> dict[str, dict[str, Any]]:
    """Per-video diagnostics, including the ego/non-ego flag for subgroup AUC."""
    return {
        r.video_id: {
            "video_id": r.video_id,
            "class_name": DOTA_CLASS_NAME,
            "anomaly_class": r.anomaly_class,
            "ego_involve": r.is_ego,
            "total_frames": r.total_frames,
            "sampled_frames": num_sampled_frames(r.total_frames, stride),
            "anomaly_span": list(r.span),
            "split": "test",
        }
        for r in records
    }


def preprocess(
    metadata: Path,
    split_file: Path,
    out_dir: Path,
    frames_dir: Path | None = None,
    frames_subdir: str | None = None,
    stride: int = constants.FRAME_STRIDE,
    strict: bool = False,
    dry_run: bool = False,
    allow_missing_frames: bool = False,
) -> list[DotaRecord]:
    """Build the standard dataset files for the official DoTA val split."""
    records = parse_metadata(metadata, read_split_ids(split_file))
    annotated = len(records)
    if frames_dir is not None:
        records = resolve_frame_counts(
            records, frames_dir, frames_subdir, allow_missing_frames
        )
    if len(records) != annotated:
        LOGGER.warning(
            "Coverage %d/%d clips (%.1f%%) -- report Delta(on-off) on this subset "
            "only; the absolute AUC is not the published protocol",
            len(records),
            annotated,
            100.0 * len(records) / annotated,
        )
    frame_labels = build_frame_labels(records, stride, strict)
    total = sum(len(v) for v in frame_labels.values())
    positive = sum(sum(v) for v in frame_labels.values())
    ego = sum(1 for r in records if r.is_ego)
    LOGGER.info(
        "DoTA val split: %d clips (%d ego / %d other), %d sampled frames at "
        "stride %d, %d positive (%.4f)",
        len(records),
        ego,
        len(records) - ego,
        total,
        stride,
        positive,
        positive / total,
    )
    if dry_run:
        LOGGER.info("Dry run: no files written")
        return records
    write_dataset_files(
        out_dir=out_dir,
        labels_train={},
        frame_labels_test=frame_labels,
        defs=class_name_list({DOTA_CLASS_NAME}),
        meta=build_meta(records, stride),
    )
    write_test_ids(out_dir, [r.video_id for r in records])
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--metadata", type=Path, required=True, help="metadata_val.json"
    )
    parser.add_argument("--split-file", type=Path, required=True, help="val_split.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="unzipped frames/ root; reconciles on-disk frame counts",
    )
    parser.add_argument(
        "--frames-subdir",
        default="images",
        help="image subfolder inside each clip folder",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=constants.FRAME_STRIDE,
        help="must match extract_clip_features --stride",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of warning when an anomaly window rounds away",
    )
    parser.add_argument(
        "--allow-missing-frames",
        action="store_true",
        help=(
            "drop annotated clips that have no frame folder instead of failing; "
            "partial coverage -- absolute AUC stops being comparable"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_arg_parser().parse_args(argv)
    preprocess(
        metadata=args.metadata,
        split_file=args.split_file,
        out_dir=args.out_dir,
        frames_dir=args.frames_dir,
        frames_subdir=args.frames_subdir,
        stride=args.stride,
        strict=args.strict,
        dry_run=args.dry_run,
        allow_missing_frames=args.allow_missing_frames,
    )


if __name__ == "__main__":
    main()
