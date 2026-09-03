"""TAD preprocessor: official test protocol (+ optional train split) -> dataset files.

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

**Two modes, and the default is unchanged from the eval-only original:**

* Default (``with_train_split=False``): evaluation only. ``labels_train.json``
  is written empty and every annotated video lands in the test split; frame
  folders the annotation does not mention are ignored (with a log line). This
  is what every TAD artifact before 2026-09-02 was built with.
* ``with_train_split=True``: the ~410 frame folders the annotation does **not**
  name become the weakly-supervised train split. TAD's train videos carry no
  public frame-level annotation, so the **directory is the label** --
  ``frames/abnormal/*`` -> 1, ``frames/normal/*`` -> 0 -- and TAD has exactly
  one anomaly class, ``Car Accident``, which is both the annotation's own name
  for it and the key in ``_TAD_CLS_DEFS``. Nothing frame-level is inferred, so
  weak supervision is preserved by construction.

Both modes emit ``test_ids.txt``; the train mode also emits ``train_ids.txt``.
The whole corpus is far larger than the scored split, so those files exist to
scope ``extract_clip_features --ids-file`` and ``raft_extract --ids-file``
(flow targets are train-time only) rather than paying for all 510 videos twice.

CLI::

    python -m core.data.tad --annotation data/TAD/annotations/tad_test_anno.json
        --frames-dir data/TAD/frames --out-dir data/TAD [--stride 8]
        [--with-train-split] [--abnormal-dirname abnormal]
        [--normal-dirname normal] [--dry-run]
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
    class_name_list,
    num_sampled_frames,
    write_dataset_files,
    write_test_ids,
    write_train_ids,
)
from core.data.definitions import DATASET_CLS_DEFS, dataset_abbr
from core.data.video_io import list_frame_folders, list_frame_images, video_id_from_path

LOGGER = logging.getLogger(__name__)

VIDEO_PATH_KEY = "video_path"
CLASS_NAME_KEY = "class_name"
ANOMALY_SPAN_KEY = "anomaly_span"

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


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


@dataclass(frozen=True)
class TadTrainRecord:
    """One TAD **train** video: a video-level label and nothing more.

    Deliberately not a :class:`TadRecord` with empty ``spans`` -- that type's
    invariant is "abnormal implies at least one span", enforced by
    :func:`_validate_spans`, and a train video legitimately has none. Keeping
    the two apart means a train record can never be mistaken for an annotated
    one on the frame-level eval path.
    """

    video_id: str
    class_name: str
    total_frames: int

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


def frame_folders_by_id(frames_dir: Path) -> dict[str, Path]:
    """``{video_id: folder}`` for every frame folder under ``frames_dir``.

    One scan, shared by the test and train resolvers: on a Drive FUSE mount a
    recursive walk of ~510 clip folders is not free, and doing it twice per run
    is the kind of waste that turns a two-minute cell into a ten-minute one.
    """
    return {video_id_from_path(p): p for p in list_frame_folders(frames_dir)}


def resolve_records(
    annotated: list[tuple[str, str, tuple[tuple[float, float], ...]]],
    frames_dir: Path,
    folders: dict[str, Path] | None = None,
) -> list[TadRecord]:
    """Attach on-disk frame counts to the annotated videos, sorted by id."""
    folders = folders if folders is not None else frame_folders_by_id(frames_dir)
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
            "%d frame folders are absent from the annotation (TAD train split)",
            unannotated,
        )
    return sorted(records, key=lambda r: r.video_id)


def _label_from_directory(
    folder: Path, frames_dir: Path, abnormal_dirname: str, normal_dirname: str
) -> str:
    """Class name implied by the split directory a frame folder sits under.

    The train split has no annotation file, so this *is* the supervision. It
    refuses to guess: a folder under neither directory (or, absurdly, both) is
    an error, not a default -- silently bucketing it as normal would poison the
    MIL objective with unlabelled anomalies and nothing downstream would raise.
    """
    parents = folder.relative_to(frames_dir).parts[:-1]
    is_abnormal = abnormal_dirname in parents
    is_normal = normal_dirname in parents
    if is_abnormal == is_normal:
        got = "both" if is_abnormal else "neither"
        raise ValueError(
            f"{folder}: {got} of the split directories {abnormal_dirname!r} / "
            f"{normal_dirname!r} appears in its path relative to {frames_dir}. "
            "The directory is the only video-level label TAD's train split has; "
            "fix the layout rather than defaulting it."
        )
    return constants.TAD_ABNORMAL_CLASS if is_abnormal else NORMAL_CLASS


def resolve_train_records(
    frames_dir: Path,
    test_ids: set[str],
    abnormal_dirname: str = constants.TAD_ABNORMAL_DIRNAME,
    normal_dirname: str = constants.TAD_NORMAL_DIRNAME,
    folders: dict[str, Path] | None = None,
) -> list[TadTrainRecord]:
    """Every frame folder outside ``test_ids``, labelled by its split directory."""
    folders = folders if folders is not None else frame_folders_by_id(frames_dir)
    records = sorted(
        (
            TadTrainRecord(
                video_id=video_id,
                class_name=_label_from_directory(
                    folder, frames_dir, abnormal_dirname, normal_dirname
                ),
                total_frames=len(list_frame_images(folder)),
            )
            for video_id, folder in folders.items()
            if video_id not in test_ids
        ),
        key=lambda r: r.video_id,
    )
    abnormal = sum(1 for r in records if r.is_abnormal)
    if not abnormal or abnormal == len(records):
        raise ValueError(
            f"TAD train split needs both classes, got {abnormal} abnormal / "
            f"{len(records) - abnormal} normal under {frames_dir}. "
            "DVSFeatureDataset raises on this later and less clearly."
        )
    return records


def check_definition_coverage(class_names: set[str]) -> None:
    """Every emitted class must have definition sentences (lesson 19).

    ``verbalize_class_name`` returns the bare class name for an unknown class
    instead of raising, so a taxonomy mismatch does not crash training -- it
    quietly conditions the text branch on ``"Car Accident"`` the string rather
    than on a definition, voiding the definition-conditioning claim with no
    error anywhere.
    """
    known = set(DATASET_CLS_DEFS[dataset_abbr(constants.TAD_DATASET)])
    undefined = sorted(class_names - known)
    if undefined:
        raise ValueError(
            f"{len(undefined)} TAD classes have no definition sentences: {undefined}. "
            "Add them to _TAD_CLS_DEFS in core/data/definitions.py -- without them "
            "the text branch is conditioned on bare class names."
        )


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


def _meta(
    test: list[TadRecord],
    train: list[TadTrainRecord],
    frame_labels: dict[str, list[int]],
    stride: int,
) -> dict[str, dict[str, object]]:
    """Per-video diagnostics -- never training supervision (data-layout contract)."""
    meta: dict[str, dict[str, object]] = {
        r.video_id: {
            "class_name": r.class_name,
            "split": TEST_SPLIT,
            "total_frames": r.total_frames,
            "sampled_frames": len(frame_labels[r.video_id]),
            "positive_frames": sum(frame_labels[r.video_id]),
            "normalized_spans": [list(s) for s in r.spans],
        }
        for r in test
    }
    meta.update(
        {
            r.video_id: {
                "class_name": r.class_name,
                "split": TRAIN_SPLIT,
                "total_frames": r.total_frames,
                "sampled_frames": num_sampled_frames(r.total_frames, stride),
                # no window: TAD's train split has no frame-level annotation,
                # which is what keeps the supervision weak by construction
                "normalized_spans": [],
            }
            for r in train
        }
    )
    return meta


def preprocess(
    annotation: Path,
    frames_dir: Path,
    out_dir: Path,
    stride: int = constants.FRAME_STRIDE,
    dry_run: bool = False,
    with_train_split: bool = False,
    abnormal_dirname: str = constants.TAD_ABNORMAL_DIRNAME,
    normal_dirname: str = constants.TAD_NORMAL_DIRNAME,
) -> tuple[list[TadTrainRecord], list[TadRecord]]:
    """Build the standard dataset files for TAD; returns ``(train, test)``.

    ``with_train_split=False`` (default) reproduces the eval-only behaviour this
    module shipped with: ``train`` comes back empty and ``labels_train.json`` is
    written empty.
    """
    folders = frame_folders_by_id(frames_dir)
    test = resolve_records(parse_annotation_file(annotation), frames_dir, folders)
    frame_labels = build_frame_labels(test, stride)
    train: list[TadTrainRecord] = []
    if with_train_split:
        train = resolve_train_records(
            frames_dir,
            {r.video_id for r in test},
            abnormal_dirname,
            normal_dirname,
            folders,
        )
    class_names = {r.class_name for r in test} | {r.class_name for r in train}
    check_definition_coverage(class_names)

    abnormal_test = sum(1 for r in test if r.is_abnormal)
    LOGGER.info(
        "TAD test split: %d videos (%d abnormal), %d sampled frames, %d positive",
        len(test),
        abnormal_test,
        sum(len(v) for v in frame_labels.values()),
        sum(sum(v) for v in frame_labels.values()),
    )
    if with_train_split:
        abnormal_train = sum(1 for r in train if r.is_abnormal)
        LOGGER.info(
            "TAD train split: %d videos (%d abnormal / %d normal), video-level "
            "labels only",
            len(train),
            abnormal_train,
            len(train) - abnormal_train,
        )
    else:
        LOGGER.info(
            "Eval-only run: labels_train.json will be empty. Pass --with-train-split "
            "to build the weakly-supervised train split from the frame folders."
        )
    if dry_run:
        LOGGER.info("Dry run: no files written")
        return train, test

    write_dataset_files(
        out_dir,
        labels_train={r.video_id: int(r.is_abnormal) for r in train},
        frame_labels_test=frame_labels,
        defs=class_name_list(class_names),
        meta=_meta(test, train, frame_labels, stride),
    )
    write_test_ids(out_dir, [r.video_id for r in test])
    if with_train_split:
        write_train_ids(out_dir, [r.video_id for r in train])
    return train, test


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--annotation", type=Path, required=True,
                        help="tad_test_anno.json (official test protocol)")
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="root of the extracted-frame folders")
    parser.add_argument("--out-dir", type=Path,
                        default=constants.DATA_ROOT / constants.TAD_DATASET)
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--with-train-split", action="store_true",
                        help="build labels_train.json from the frame folders the "
                             "annotation does not name; the split directory is the "
                             "video-level label (default: off, eval-only)")
    parser.add_argument("--abnormal-dirname", default=constants.TAD_ABNORMAL_DIRNAME,
                        help="split directory whose train clips are abnormal")
    parser.add_argument("--normal-dirname", default=constants.TAD_NORMAL_DIRNAME,
                        help="split directory whose train clips are normal")
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
        with_train_split=args.with_train_split,
        abnormal_dirname=args.abnormal_dirname,
        normal_dirname=args.normal_dirname,
    )


if __name__ == "__main__":
    main()
