"""TAD preprocessor: official test protocol -> the standard dataset files.

TAD ships extracted frames, not videos::

    data/TAD/frames/abnormal/01_Accident_106.mp4/0.jpg ...
    data/TAD/frames/normal/Normal_001.mp4/0.jpg ...

and the official **test** annotation is the JSON the LaGoVAD baseline ships as
``data/other_datasets/tad_test_anno.json`` -- 100 videos (60 ``Car Accident``
+ 40 ``Normal``), the split its reported TAD 89.56 AUC is measured on. Copy it
to ``data/TAD/annotations/`` (the baseline tree is git-ignored, so it does not
travel to Colab).

Each entry carries ``anomaly_span`` as **normalized** ``[start_frac, end_frac]``
pairs -- fractions of the clip, so they convert to any sampling rate. Ten of the
60 abnormal videos carry more than one span.

This module is **evaluation-only**: TAD's remaining 410 videos are its train
split and have no public frame-level annotation, so ``labels_train.json`` is
written empty and every annotated video lands in the test split. Frame folders
on disk that the annotation does not mention are ignored (with a log line).

The whole corpus is far larger than the scored split, so ``test_ids.txt`` is
emitted alongside for ``extract_clip_features --ids-file`` -- extracting all 510
videos would waste most of the run.

CLI::

    python -m core.data.tad --annotation data/TAD/annotations/tad_test_anno.json
        --frames-dir data/TAD/frames --out-dir data/TAD [--stride 8] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from core import constants
from core.data.dataset_files import (
    NORMAL_CLASS,
    TEST_IDS_FILENAME,
    class_name_list,
    num_sampled_frames,
    write_dataset_files,
)
from core.data.video_io import list_frame_folders, list_frame_images, video_id_from_path

LOGGER = logging.getLogger(__name__)

VIDEO_PATH_KEY = "video_path"
CLASS_NAME_KEY = "class_name"
ANOMALY_SPAN_KEY = "anomaly_span"


@dataclass(frozen=True)
class TadRecord:
    """One annotated TAD test video, resolved against the frames on disk.

    ``spans`` are normalized ``(start_frac, end_frac)`` pairs verbatim from the
    annotation; ``total_frames`` is the image count in the frame folder, which
    is authoritative because the extractor reads those same images.
    """

    video_id: str
    class_name: str
    total_frames: int
    spans: tuple[tuple[float, float], ...]

    @property
    def is_abnormal(self) -> bool:
        return self.class_name != NORMAL_CLASS


def parse_annotation_file(path: Path) -> list[tuple[str, str, tuple[tuple[float, float], ...]]]:
    """Read ``(video_id, class_name, spans)`` triples from the TAD test JSON."""
    with path.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must be a non-empty JSON list of annotation entries")

    parsed: list[tuple[str, str, tuple[tuple[float, float], ...]]] = []
    seen: set[str] = set()
    for entry in entries:
        video_id = video_id_from_path(Path(str(entry[VIDEO_PATH_KEY])))
        if video_id in seen:
            raise ValueError(f"Duplicate annotation entry for {video_id}")
        seen.add(video_id)
        class_name = str(entry[CLASS_NAME_KEY])
        spans = tuple(
            (float(start), float(end)) for start, end in entry[ANOMALY_SPAN_KEY] or ()
        )
        _validate_spans(video_id, class_name, spans)
        parsed.append((video_id, class_name, spans))
    LOGGER.info("Parsed %d annotated videos from %s", len(parsed), path)
    return parsed


def _validate_spans(
    video_id: str, class_name: str, spans: tuple[tuple[float, float], ...]
) -> None:
    """Reject class/span disagreements before they become silent label errors."""
    is_abnormal = class_name != NORMAL_CLASS
    if is_abnormal and not spans:
        raise ValueError(
            f"{video_id}: class {class_name!r} but no anomaly span; frame-level "
            "eval would score it as entirely normal"
        )
    if not is_abnormal and spans:
        raise ValueError(f"{video_id}: class Normal but carries spans {spans}")
    for start, end in spans:
        if not 0.0 <= start < end <= 1.0:
            raise ValueError(
                f"{video_id}: span ({start}, {end}) is not a normalized, "
                "increasing fraction pair within [0, 1]"
            )


def resolve_records(
    annotated: list[tuple[str, str, tuple[tuple[float, float], ...]]],
    frames_dir: Path,
) -> list[TadRecord]:
    """Attach on-disk frame counts to the annotated videos, sorted by id."""
    folders = {video_id_from_path(p): p for p in list_frame_folders(frames_dir)}
    missing = sorted(video_id for video_id, _, _ in annotated if video_id not in folders)
    if missing:
        raise ValueError(
            f"{len(missing)} annotated videos have no frame folder under "
            f"{frames_dir}: {missing[:5]}"
        )
    records = [
        TadRecord(
            video_id=video_id,
            class_name=class_name,
            total_frames=len(list_frame_images(folders[video_id])),
            spans=spans,
        )
        for video_id, class_name, spans in annotated
    ]
    unannotated = len(folders) - len(records)
    if unannotated:
        LOGGER.info(
            "Ignoring %d frame folders absent from the annotation (TAD train split)",
            unannotated,
        )
    return sorted(records, key=lambda r: r.video_id)


def sampled_frame_labels(record: TadRecord, stride: int) -> list[int]:
    """Per-sampled-frame 0/1 labels from normalized spans.

    Mirrors the baseline (``LaGoVAD-PreVAD/src/datasets/base.py``): scale each
    fraction by the sampled length, round, and fill the half-open interval, so
    our micro AUC/AP are computed over the same ground truth.
    """
    length = num_sampled_frames(record.total_frames, stride)
    labels = [0] * length
    for start_frac, end_frac in record.spans:
        start = max(0, min(length, round(start_frac * length)))
        end = max(0, min(length, round(end_frac * length)))
        for index in range(start, end):
            labels[index] = 1
    return labels


def build_frame_labels(
    records: list[TadRecord], stride: int
) -> dict[str, list[int]]:
    """Frame labels for every record; abnormal videos must keep a positive frame.

    A span shorter than one stride at this sampling rate rounds away to nothing,
    turning an abnormal video into a negative-only one -- silently depressing
    recall rather than erroring. Refuse it instead.
    """
    frame_labels = {r.video_id: sampled_frame_labels(r, stride) for r in records}
    vanished = sorted(
        r.video_id for r in records if r.is_abnormal and not any(frame_labels[r.video_id])
    )
    if vanished:
        raise ValueError(
            f"{len(vanished)} abnormal videos lose their anomaly window at stride "
            f"{stride} (span shorter than one sampled frame): {vanished[:5]}"
        )
    return frame_labels


def write_test_ids(out_dir: Path, records: list[TadRecord]) -> Path:
    """Emit the scored video ids for ``extract_clip_features --ids-file``."""
    target = out_dir / TEST_IDS_FILENAME
    target.write_text(
        "".join(f"{r.video_id}\n" for r in records), encoding="utf-8"
    )
    LOGGER.info("Wrote %s (%d ids)", target, len(records))
    return target


def preprocess(
    annotation: Path,
    frames_dir: Path,
    out_dir: Path,
    stride: int = constants.FRAME_STRIDE,
    dry_run: bool = False,
) -> list[TadRecord]:
    """Build the standard dataset files for the official TAD test split."""
    records = resolve_records(parse_annotation_file(annotation), frames_dir)
    frame_labels = build_frame_labels(records, stride)
    abnormal = [r for r in records if r.is_abnormal]
    LOGGER.info(
        "TAD test split: %d videos (%d abnormal), %d sampled frames, %d positive",
        len(records),
        len(abnormal),
        sum(len(v) for v in frame_labels.values()),
        sum(sum(v) for v in frame_labels.values()),
    )
    if dry_run:
        LOGGER.info("Dry run: no files written")
        return records

    write_dataset_files(
        out_dir,
        labels_train={},  # eval-only: TAD's train split has no public annotation
        frame_labels_test=frame_labels,
        defs=class_name_list({r.class_name for r in records}),
        meta={
            r.video_id: {
                "class_name": r.class_name,
                "split": "test",
                "total_frames": r.total_frames,
                "sampled_frames": len(frame_labels[r.video_id]),
                "positive_frames": sum(frame_labels[r.video_id]),
                "normalized_spans": [list(s) for s in r.spans],
            }
            for r in records
        },
    )
    write_test_ids(out_dir, records)
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--annotation", type=Path, required=True,
                        help="tad_test_anno.json (official test protocol)")
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="root of the extracted-frame folders")
    parser.add_argument("--out-dir", type=Path,
                        default=constants.DATA_ROOT / constants.TAD_DATASET)
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and report, write nothing")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    preprocess(
        annotation=args.annotation,
        frames_dir=args.frames_dir,
        out_dir=args.out_dir,
        stride=args.stride,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
