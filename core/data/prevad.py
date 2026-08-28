"""PreVAD preprocessor: the released train/test CSVs -> the standard dataset files.

PreVAD is LaGoVAD's training set (35,279 videos). The HuggingFace release ships
**no raw video** — only per-clip CLIP features already sampled at interval 8
(``features/ViT-B-16-8p-features.zip``) plus two annotation tables::

    video_id,path,video_path,class_name,superclass_name,descriptions,anomaly_span

``video_id`` equals the ``.npy`` stem for all 35,279 rows, so a flat unzip
satisfies the ``cache/clip/{DATASET}/{video_id}.npy`` contract with no renaming.

**There is no ``--stride`` here, and that is deliberate.** Every other
preprocessor derives the label length from a raw frame count via
:func:`core.data.dataset_files.num_sampled_frames`. PreVAD ships no frames: the
``.npy`` *is* the sampled sequence. Feature lengths are therefore read from the
array header, which is the only value that can be right — guessing one from the
timestamps in the video id would silently shift every anomaly window (C2/C13).

**Label parity with the baseline.** ``LaGoVAD-PreVAD/src/datasets/base.py:57-69``
scales each normalized span by the feature length, rounds, and fills the
half-open interval into a ``vis_max_len``-long zero buffer. Python slicing on
that buffer clamps out-of-range ends and drops reversed spans, so this module
clamps to ``[0, L]`` explicitly and reproduces the same arithmetic — the same
convention that reproduced the published DoTA number (:mod:`core.data.dota`).

Both quirks the clamp absorbs are present in the shipped test table and are
counted, never repaired: **440 spans end past 1.0** (up to 1.2104) and **one is
reversed** (``RrUW8ITUqx0_aug3``). Refusing them would make the released test
split unusable; silently "fixing" them would diverge from the baseline's own
ground truth.

**Multi-span is the norm, not an edge case.** 104 of the 1,306 abnormal test
clips carry 2-4 disjoint anomaly windows. Collapsing them to their hull would
label the normal gaps between windows abnormal, so every span is filled
independently.

CLI::

    python -m core.data.prevad --train-csv PreVAD/train.csv
        --test-csv PreVAD/test.csv --clip-dir "$PREVAD_CLIP"
        --out-dir data/PreVAD [--strict] [--allow-missing-features] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.dataset_files import (
    NORMAL_CLASS,
    class_name_list,
    write_dataset_files,
    write_test_ids,
)
from core.data.definitions import DATASET_CLS_DEFS

LOGGER = logging.getLogger(__name__)

PREVAD_DEFS_KEY = "prevad"  # DATASET_CLS_DEFS key holding the v6 taxonomy
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"

VIDEO_ID_FIELD = "video_id"
CLASS_NAME_FIELD = "class_name"
SUPERCLASS_NAME_FIELD = "superclass_name"
DESCRIPTIONS_FIELD = "descriptions"
ANOMALY_SPAN_FIELD = "anomaly_span"
REQUIRED_CSV_FIELDS = (
    VIDEO_ID_FIELD,
    CLASS_NAME_FIELD,
    SUPERCLASS_NAME_FIELD,
    DESCRIPTIONS_FIELD,
    ANOMALY_SPAN_FIELD,
)

# The released features are unnormalized ``encode_image`` output, so their L2
# norms sit around 8-12. A mean near 1.0 means the release is pre-normalized and
# our pipeline (which is not) would be feeding the model a different scale.
NORM_CHECK_SAMPLES = 32
MIN_EXPECTED_FEATURE_NORM = 2.0


@dataclass(frozen=True)
class PrevadRecord:
    """One annotated PreVAD clip.

    ``spans`` are normalized ``[start, end]`` fractions of the clip, exactly as
    the release stores them: possibly several, possibly ending past 1.0, and in
    one shipped case reversed. They are carried verbatim and only clamped where
    frame labels are built.
    """

    video_id: str
    class_name: str
    superclass_name: str
    description: str
    spans: tuple[tuple[float, float], ...]
    split: str

    @property
    def is_abnormal(self) -> bool:
        """Video-level weak label: anything the release does not call Normal."""
        return self.class_name != NORMAL_CLASS


def parse_spans(raw: str, video_id: str) -> tuple[tuple[float, float], ...]:
    """Parse the ``anomaly_span`` cell -- a JSON list of ``[start, end]`` pairs.

    Rejects only what cannot be a span at all (bad JSON, wrong arity, non-finite
    or negative values). Spans that end past 1.0 or run backwards are shipped by
    the release and are accepted here; :func:`build_frame_labels` counts them.
    """
    text = raw.strip()
    if not text:
        return ()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{video_id}: anomaly_span is not valid JSON: {raw!r}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{video_id}: anomaly_span must be a list, got {type(payload)}")
    spans: list[tuple[float, float]] = []
    for entry in payload:
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            raise ValueError(f"{video_id}: anomaly span {entry!r} is not a [start, end] pair")
        start, end = (float(value) for value in entry)
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"{video_id}: non-finite anomaly span {entry!r}")
        if start < 0.0:
            raise ValueError(f"{video_id}: negative anomaly span start in {entry!r}")
        spans.append((start, end))
    return tuple(spans)


def read_split_csv(path: Path, split: str) -> list[PrevadRecord]:
    """Read one release CSV into records, validating the schema and the split."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing_fields = [f for f in REQUIRED_CSV_FIELDS if f not in (reader.fieldnames or ())]
        if missing_fields:
            raise ValueError(
                f"{path} is missing required columns {missing_fields}; "
                f"found {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no rows")

    records: list[PrevadRecord] = []
    for row in rows:
        video_id = (row[VIDEO_ID_FIELD] or "").strip()
        if not video_id:
            raise ValueError(f"{path}: a row has an empty {VIDEO_ID_FIELD}")
        class_name = (row[CLASS_NAME_FIELD] or "").strip()
        if not class_name:
            raise ValueError(f"{path}: {video_id} has an empty {CLASS_NAME_FIELD}")
        records.append(
            PrevadRecord(
                video_id=video_id,
                class_name=class_name,
                superclass_name=(row[SUPERCLASS_NAME_FIELD] or "").strip(),
                description=(row[DESCRIPTIONS_FIELD] or "").strip(),
                spans=parse_spans(row[ANOMALY_SPAN_FIELD] or "", video_id),
                split=split,
            )
        )
    _check_unique_ids(records, path)
    _check_label_span_agreement(records, path)
    abnormal = sum(1 for r in records if r.is_abnormal)
    LOGGER.info(
        "Read %d %s rows from %s (%d normal / %d abnormal)",
        len(records),
        split,
        path,
        len(records) - abnormal,
        abnormal,
    )
    return sorted(records, key=lambda r: r.video_id)


def _check_unique_ids(records: list[PrevadRecord], path: Path) -> None:
    """One row per video id -- duplicates would collide in every output file."""
    counts = Counter(r.video_id for r in records)
    duplicates = sorted(video_id for video_id, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"{path} has {len(duplicates)} duplicate video ids: {duplicates[:5]}")


def _check_label_span_agreement(records: list[PrevadRecord], path: Path) -> None:
    """The class name and the presence of a span must tell the same story.

    A Normal row with an anomaly window, or an abnormal row without one, means
    the table disagrees with itself: the video-level weak label written to
    ``labels_train.json`` and the frame labels written to
    ``frame_labels_test.json`` would then contradict each other.
    """
    normal_with_span = [r.video_id for r in records if not r.is_abnormal and r.spans]
    abnormal_without_span = [r.video_id for r in records if r.is_abnormal and not r.spans]
    problems = []
    if normal_with_span:
        problems.append(f"{len(normal_with_span)} Normal rows carry a span {normal_with_span[:5]}")
    if abnormal_without_span:
        problems.append(
            f"{len(abnormal_without_span)} abnormal rows carry no span "
            f"{abnormal_without_span[:5]}"
        )
    if problems:
        raise ValueError(f"{path}: class name and anomaly span disagree -- " + "; ".join(problems))


def check_definition_coverage(records: list[PrevadRecord]) -> None:
    """Every observed class must have definition sentences (gap G3).

    ``verbalize_class_name`` falls back to the bare class name for an unknown
    class instead of raising, so a missing definition does not crash training --
    it quietly conditions the text branch on ``"Store Robbery"`` rather than on
    a definition, which is precisely the claim LaGoVAD is testing. Catch it here,
    where it is loud and cheap.
    """
    known = set(DATASET_CLS_DEFS[PREVAD_DEFS_KEY])
    observed = {r.class_name for r in records}
    undefined = sorted(observed - known)
    if undefined:
        raise ValueError(
            f"{len(undefined)} PreVAD classes have no definition sentences: {undefined}. "
            f"Add them to _PREVAD_CLS_DEFS in core/data/definitions.py -- without them "
            f"the text branch is conditioned on bare class names and the "
            f"definition-conditioning claim is silently void."
        )
    unused = sorted(known - observed)
    if unused:
        LOGGER.info(
            "%d defined classes do not occur in the data and stay out of defs.json: %s",
            len(unused),
            unused,
        )


def feature_length(path: Path) -> int:
    """Rows in a cached feature array, validating the layout without loading it.

    ``mmap_mode="r"`` reads the ``.npy`` header only, so this stays cheap over
    35k files. The width check is §4.3's sanity check made unskippable: a 768-d
    array means the ViT-L/14 release was unzipped instead of ViT-B/16.
    """
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D (L, D) feature array, got shape {array.shape}")
    if array.shape[1] != constants.CLIP_FEATURE_DIM:
        raise ValueError(
            f"{path}: feature width {array.shape[1]} != {constants.CLIP_FEATURE_DIM}; "
            f"this is not the ViT-B/16 release"
        )
    if array.shape[0] == 0:
        raise ValueError(f"{path}: feature array is empty")
    return int(array.shape[0])


def resolve_feature_lengths(
    records: list[PrevadRecord],
    clip_dir: Path,
    allow_missing: bool = False,
) -> tuple[list[PrevadRecord], dict[str, int]]:
    """Read every record's feature length, dropping or refusing unusable clips.

    A clip whose ``.npy`` is absent or unreadable is fatal by default: for the
    train split it would crash mid-epoch hours in, and for the test split it
    would move the evaluation denominator without saying so (C10). ``allow_missing``
    drops them instead and logs the resulting coverage.
    """
    kept: list[PrevadRecord] = []
    lengths: dict[str, int] = {}
    absent: list[str] = []
    unreadable: list[tuple[str, str]] = []
    for record in records:
        path = clip_dir / f"{record.video_id}.npy"
        if not path.exists():
            absent.append(record.video_id)
            continue
        try:
            lengths[record.video_id] = feature_length(path)
        except (ValueError, OSError) as exc:
            unreadable.append((record.video_id, str(exc)))
            continue
        kept.append(record)
    if absent or unreadable:
        _report_unusable(records, absent, unreadable, clip_dir, allow_missing)
    if not kept:
        raise ValueError(f"No PreVAD clip has a readable feature array under {clip_dir}")
    return kept, lengths


def _report_unusable(
    records: list[PrevadRecord],
    absent: list[str],
    unreadable: list[tuple[str, str]],
    clip_dir: Path,
    allow_missing: bool,
) -> None:
    """Raise or warn about clips with no usable feature array, cause by cause."""
    message = (
        f"{len(absent) + len(unreadable)}/{len(records)} clips have no usable feature "
        f"array under {clip_dir}: {len(absent)} absent {sorted(absent)[:5]}, "
        f"{len(unreadable)} unreadable {[v for v, _ in unreadable[:3]]}"
    )
    for video_id, reason in unreadable[:3]:
        LOGGER.warning("%s is unreadable: %s", video_id, reason)
    if not allow_missing:
        raise ValueError(
            f"{message} -- finish the unzip (`unzip -n` resumes), or pass "
            "--allow-missing-features to build labels for the clips that are "
            "present (the split is then a subset and absolute numbers stop "
            "being comparable)"
        )
    LOGGER.warning("%s -- dropped", message)


def check_feature_scale(
    records: list[PrevadRecord],
    clip_dir: Path,
    samples: int = NORM_CHECK_SAMPLES,
) -> float:
    """Mean L2 norm over a sample of feature rows; warns if they look normalized.

    Both LaGoVAD's extractor and ours save raw ``encode_image`` output, whose
    rows have norms around 8-12. A mean near 1.0 means the release stores
    L2-normalized features, and every downstream number would be measured on a
    scale the model was never trained at (C13).
    """
    norms: list[float] = []
    for record in records[:samples]:
        array = np.load(clip_dir / f"{record.video_id}.npy").astype(np.float32)
        norms.append(float(np.linalg.norm(array, axis=1).mean()))
    if not norms:
        return 0.0
    mean_norm = float(np.mean(norms))
    if mean_norm < MIN_EXPECTED_FEATURE_NORM:
        LOGGER.warning(
            "Mean feature L2 norm is %.3f over %d clips -- expected ~8-12 for raw "
            "encode_image output. These features look pre-normalized; our pipeline "
            "is not, so checkpoints would be scored at a scale they never saw.",
            mean_norm,
            len(norms),
        )
    else:
        LOGGER.info(
            "Mean feature L2 norm %.3f over %d clips (raw, as expected)",
            mean_norm,
            len(norms),
        )
    return mean_norm


def sampled_frame_labels(record: PrevadRecord, length: int) -> list[int]:
    """Per-feature-row 0/1 labels, filling **every** span (gap G2).

    Mirrors ``LaGoVAD-PreVAD/src/datasets/base.py``: scale both fractions by the
    feature length, round, fill half-open. Each span is filled independently, so
    the normal gaps between two anomaly windows stay normal. Bounds are clamped
    to ``[0, length]``, which is what the baseline gets for free by slicing into
    a longer buffer -- an end past 1.0 stops at the clip end and a reversed span
    contributes nothing.
    """
    labels = [0] * length
    for start_frac, end_frac in record.spans:
        start = max(0, min(length, round(start_frac * length)))
        end = max(0, min(length, round(end_frac * length)))
        for index in range(start, end):
            labels[index] = 1
    return labels


def build_frame_labels(
    records: list[PrevadRecord],
    lengths: dict[str, int],
    strict: bool = False,
) -> dict[str, list[int]]:
    """Frame labels for the test split, reporting every span quirk it absorbed.

    Three things are counted rather than repaired, because the baseline's own
    label code absorbs all three and the released ground truth is defined by
    what it produces: spans ending past the clip (440 shipped rows), reversed
    spans (1 shipped row), and abnormal clips whose windows all round away.
    ``strict`` refuses the last of those instead.
    """
    frame_labels = {r.video_id: sampled_frame_labels(r, lengths[r.video_id]) for r in records}

    overflow = sum(1 for r in records for _, end in r.spans if end > 1.0)
    reversed_spans = sum(1 for r in records for start, end in r.spans if end <= start)
    if overflow:
        LOGGER.warning(
            "%d anomaly spans end past the clip (max end %.4f) -- clamped to the "
            "clip length, matching the baseline's buffer slicing",
            overflow,
            max((end for r in records for _, end in r.spans), default=0.0),
        )
    if reversed_spans:
        LOGGER.warning(
            "%d anomaly spans run backwards (end <= start) and contribute no "
            "positive frames, matching the baseline",
            reversed_spans,
        )

    vanished = sorted(
        r.video_id for r in records if r.is_abnormal and not any(frame_labels[r.video_id])
    )
    if vanished:
        message = (
            f"{len(vanished)} abnormal clips have no positive frame after rounding "
            f"(window shorter than one feature row, or reversed): {vanished[:5]}"
        )
        if strict:
            raise ValueError(message)
        LOGGER.warning("%s -- they contribute only negative frames", message)
    return frame_labels


def build_train_labels(records: list[PrevadRecord]) -> dict[str, int]:
    """Video-level weak labels for the train split -- windows are never used."""
    return {r.video_id: int(r.is_abnormal) for r in records}


def build_meta(
    records: list[PrevadRecord],
    lengths: dict[str, int],
    frame_labels: dict[str, list[int]],
) -> dict[str, dict[str, Any]]:
    """Per-video diagnostics.

    ``class_name`` is the field :class:`~core.data.dataset.DVSFeatureDataset`
    reads to label each training sample. ``description`` is carried for the
    per-video caption arm (gap G4) that is deliberately not wired yet -- this is
    the only place it would survive.
    """
    meta: dict[str, dict[str, Any]] = {}
    for record in records:
        entry: dict[str, Any] = {
            "video_id": record.video_id,
            "class_name": record.class_name,
            "superclass_name": record.superclass_name,
            "description": record.description,
            "anomaly_span": [list(span) for span in record.spans],
            "feature_frames": lengths[record.video_id],
            "split": record.split,
        }
        if record.video_id in frame_labels:
            entry["positive_frames"] = sum(frame_labels[record.video_id])
        meta[record.video_id] = entry
    return meta


def preprocess(
    train_csv: Path,
    test_csv: Path,
    clip_dir: Path,
    out_dir: Path,
    strict: bool = False,
    dry_run: bool = False,
    allow_missing_features: bool = False,
    norm_check_samples: int = NORM_CHECK_SAMPLES,
) -> tuple[list[PrevadRecord], list[PrevadRecord]]:
    """Build the standard dataset files for the released PreVAD split."""
    train = read_split_csv(train_csv, SPLIT_TRAIN)
    test = read_split_csv(test_csv, SPLIT_TEST)
    overlap = sorted({r.video_id for r in train} & {r.video_id for r in test})
    if overlap:
        raise ValueError(
            f"{len(overlap)} video ids appear in both splits: {overlap[:5]} -- "
            f"they share one feature cache and one meta entry"
        )
    check_definition_coverage(train + test)

    annotated_train, annotated_test = len(train), len(test)
    train, train_lengths = resolve_feature_lengths(train, clip_dir, allow_missing_features)
    test, test_lengths = resolve_feature_lengths(test, clip_dir, allow_missing_features)
    lengths = {**train_lengths, **test_lengths}
    if len(train) != annotated_train or len(test) != annotated_test:
        LOGGER.warning(
            "Coverage: train %d/%d (%.1f%%), test %d/%d (%.1f%%) -- this is a "
            "subset of the released split; absolute numbers are not comparable "
            "to the published protocol",
            len(train), annotated_train, 100.0 * len(train) / annotated_train,
            len(test), annotated_test, 100.0 * len(test) / annotated_test,
        )
    check_feature_scale(test, clip_dir, norm_check_samples)

    frame_labels = build_frame_labels(test, test_lengths, strict)
    labels_train = build_train_labels(train)
    defs = class_name_list({r.class_name for r in train + test})

    total_frames = sum(len(v) for v in frame_labels.values())
    positive = sum(sum(v) for v in frame_labels.values())
    normal_test = sum(1 for r in test if not r.is_abnormal)
    LOGGER.info(
        "PreVAD: %d train (%d abnormal), %d test (%d normal / %d abnormal), "
        "%d classes, %d test feature rows, %d positive (%.4f)",
        len(train),
        sum(labels_train.values()),
        len(test),
        normal_test,
        len(test) - normal_test,
        len(defs),
        total_frames,
        positive,
        positive / total_frames if total_frames else 0.0,
    )
    LOGGER.info(
        "Test split is %.1f%% normal videos -- core.evaluate --score-norm auto "
        "resolves to raw pooling (threshold %.0f%%), the MSAD protocol, not DoTA's "
        "per-clip min-max (C12)",
        100.0 * normal_test / len(test),
        100.0 * constants.SCORE_NORM_AUTO_NORMAL_FRACTION,
    )
    if dry_run:
        LOGGER.info("Dry run: no files written")
        return train, test

    write_dataset_files(
        out_dir=out_dir,
        labels_train=labels_train,
        frame_labels_test=frame_labels,
        defs=defs,
        meta=build_meta(train + test, lengths, frame_labels),
    )
    write_test_ids(out_dir, [r.video_id for r in test])
    return train, test


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--train-csv", type=Path, required=True, help="PreVAD/train.csv")
    parser.add_argument("--test-csv", type=Path, required=True, help="PreVAD/test.csv")
    parser.add_argument(
        "--clip-dir",
        type=Path,
        required=True,
        help=(
            "unzipped ViT-B-16-8p features; label lengths are read from these "
            "arrays, so they are required rather than optional"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of warning when an abnormal clip's spans all round away",
    )
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help=(
            "drop clips whose .npy is absent or unreadable instead of failing; "
            "partial coverage -- absolute numbers stop being comparable"
        ),
    )
    parser.add_argument(
        "--norm-check-samples",
        type=int,
        default=NORM_CHECK_SAMPLES,
        help="clips sampled for the feature-scale check (0 disables it)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    preprocess(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        clip_dir=args.clip_dir,
        out_dir=args.out_dir,
        strict=args.strict,
        dry_run=args.dry_run,
        allow_missing_features=args.allow_missing_features,
        norm_check_samples=args.norm_check_samples,
    )


if __name__ == "__main__":
    main()
