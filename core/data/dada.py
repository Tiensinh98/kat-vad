"""DADA-2000 preprocessor: frame folders + Cleaned_Metadata.csv -> standard files.

DADA-2000 ships extracted frames split by **fault-attribution directory**, not a
train/test split::

    0_Non_Ego_Fault/type10_vid001/*.png ...
    0_Normal_Driving/type10_vid001/*.png ...
    1_Ego_Fault/type10_vid023/*.png ...

and ``Cleaned_Metadata.csv`` -- one row per **accident** clip (``0_Non_Ego_Fault``
or ``1_Ego_Fault``), carrying a single raw-frame accident window
(``abnormal start frame``, ``accident frame``, ``abnormal end frame``,
``total frames``) plus a per-type/per-video index (``type``, ``video``) that
identifies the on-disk folder. ``0_Normal_Driving`` carries **no CSV rows at
all** -- exactly like TAD's train split, the directory is the whole label.

Unlike TAD/DoTA, DADA-2000 ships frame-level ground truth for (almost) every
abnormal clip, not just a held-out test slice, so this module can build a real
train/test split with genuine frame-level test labels (MSAD-style), rather than
TAD's directory-only weak train. Three record kinds land in one corpus:

* **Annotated abnormal** -- a CSV row resolved against its on-disk folder;
  known window; eligible for **either** split.
* **Weak abnormal** -- a frame folder under a fault-attribution directory with
  *no* matching CSV row (a handful per type in the shipped file, e.g. type 1 has
  53 folders but 48 CSV rows). Unknown window, so it is **forced into train**
  the same way DoTA and MSAD refuse a windowless *test* video.
* **Normal** -- every folder under ``0_Normal_Driving``; no window; eligible
  for either split.

Weak supervision is preserved throughout: any window known for a *train*
record is written to ``meta.json`` only, never into ``labels_train.json``.

``type<N>_vid<N>`` folder names repeat across the three fault-attribution
directories (confirmed on the real archive), so ``video_id`` is
``{fault_label}{DADA_ID_SEPARATOR}folder_name`` -- globally unique by
construction, not just asserted so. ``--flat-frames-dir`` materializes a
symlink farm under that id, for feeding ``extract_clip_features.py``/
``raft_extract.py --frames-dir`` unchanged (both key their folder scan on
bare ``Path.name``, which is not unique here).

Split: seeded, stratified by ``(is_abnormal, fault_label)`` (MSAD's
``split_records`` pattern) -- an explicit ``--split-file`` (test ids) overrides.

CLI::

    python -m core.data.dada --metadata data/DADA2000/Cleaned_Metadata.csv
        --frames-dir data/DADA2000/frames --out-dir data/DADA2000
        [--stride 8] [--seed 2024]
        [--test-ratio-abnormal 0.2] [--test-ratio-normal 0.2]
        [--split-file test_ids.txt] [--exclude-unannotated-abnormal]
        [--allow-missing-frames] [--strict]
        [--flat-frames-dir data/DADA2000/frames_flat] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
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

_FOLDER_RE = re.compile(r"^type(\d+)_vid(\d+)$")

# Cleaned_Metadata.csv column names, verbatim.
_COL_VIDEO = "video"
_COL_TYPE = "type"
_COL_ACCIDENT_FLAG = "whether an accident occurred (1/0)"
_COL_START = "abnormal start frame"
_COL_ACCIDENT = "accident frame"
_COL_END = "abnormal end frame"
_COL_TOTAL = "total frames"
_COL_FAULT_LABEL = "Fault_Label"
_REQUIRED_COLUMNS = (
    _COL_VIDEO, _COL_TYPE, _COL_ACCIDENT_FLAG, _COL_START, _COL_ACCIDENT,
    _COL_END, _COL_TOTAL, _COL_FAULT_LABEL,
)

FAULT_DIRNAMES = (constants.DADA_NON_EGO_FAULT_DIRNAME, constants.DADA_EGO_FAULT_DIRNAME)
ALL_DIRNAMES = (*FAULT_DIRNAMES, constants.DADA_NORMAL_DIRNAME)

TEST_RATIO_ABNORMAL = 0.2
TEST_RATIO_NORMAL = 0.2


@dataclass(frozen=True)
class DadaRecord:
    """One DADA-2000 clip, resolved against its on-disk frame folder.

    ``span``/``accident_frac`` are ``None`` for a normal clip and for a
    **weak** abnormal clip (frame folder with no CSV row) -- the latter is
    never a test candidate, only ``labels_train.json`` may hold it.

    ``type<N>_vid<N>`` folder names repeat across the three fault-attribution
    directories (confirmed on the real archive, not a hypothetical), so
    ``video_id`` is ``{fault_label}{DADA_ID_SEPARATOR}{folder_name}`` --
    globally unique by construction. ``folder_name`` keeps the bare on-disk
    name for re-deriving the ``(type, video)`` key and for locating the
    source folder (:func:`materialize_flat_dir`).
    """

    video_id: str
    folder_name: str
    class_name: str
    fault_label: str
    total_frames: int
    span: tuple[float, float] | None
    accident_frac: float | None

    @property
    def is_abnormal(self) -> bool:
        return self.class_name != NORMAL_CLASS

    @property
    def is_ego(self) -> bool:
        return self.fault_label == constants.DADA_EGO_FAULT_DIRNAME

    @property
    def has_window(self) -> bool:
        return self.span is not None


def _make_video_id(fault_label: str, folder_name: str) -> str:
    """``{fault_label}__{folder_name}`` -- globally unique, unlike ``folder_name`` alone."""
    return f"{fault_label}{constants.DADA_ID_SEPARATOR}{folder_name}"


def _type_and_vid(folder_name: str) -> tuple[int, int]:
    match = _FOLDER_RE.match(folder_name.strip())
    if not match:
        raise ValueError(
            f"Frame folder {folder_name!r} does not match the expected "
            "'type<N>_vid<N>' layout DADA-2000 ships"
        )
    return int(match.group(1)), int(match.group(2))


def frame_folders_by_type_vid(fault_dir: Path) -> dict[tuple[int, int], Path]:
    """``{(type, video): folder}`` for one fault-attribution directory.

    Ints, not the padded on-disk string, so the CSV's un-padded ``video``
    column joins regardless of the archive's zero-padding width.
    """
    folders = {}
    for folder in list_frame_folders(fault_dir):
        key = _type_and_vid(folder.name)
        if key in folders:
            raise ValueError(f"Duplicate frame folder for type/video {key} under {fault_dir}")
        folders[key] = folder
    return folders


class _CsvRow:
    __slots__ = ("accident", "end", "fault_label", "start", "total_frames", "video_id_key")

    def __init__(
        self,
        video_id_key: tuple[int, int],
        fault_label: str,
        start: int,
        accident: int,
        end: int,
        total_frames: int,
    ) -> None:
        self.video_id_key = video_id_key
        self.fault_label = fault_label
        self.start = start
        self.accident = accident
        self.end = end
        self.total_frames = total_frames


def parse_metadata_csv(path: Path) -> list[_CsvRow]:
    """Read ``Cleaned_Metadata.csv`` into validated per-row accident windows.

    Rows whose accident flag is not ``1`` are skipped with a warning (defensive
    only -- every row in the file this was written against carries ``1``;
    ``0_Normal_Driving`` has no rows at all).
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        rows: list[_CsvRow] = []
        seen: set[tuple[str, tuple[int, int]]] = set()
        skipped_flag = 0
        for line_no, raw in enumerate(reader, start=2):
            if raw[_COL_ACCIDENT_FLAG].strip() != "1":
                skipped_flag += 1
                continue
            fault_label = raw[_COL_FAULT_LABEL].strip()
            if fault_label not in FAULT_DIRNAMES:
                raise ValueError(
                    f"{path}:{line_no}: Fault_Label {fault_label!r} is not one of "
                    f"{FAULT_DIRNAMES}"
                )
            key = (int(raw[_COL_TYPE]), int(raw[_COL_VIDEO]))
            dedup_key = (fault_label, key)
            if dedup_key in seen:
                raise ValueError(f"{path}:{line_no}: duplicate row for {fault_label}/{key}")
            seen.add(dedup_key)

            start, accident, end, total = (
                int(raw[_COL_START]), int(raw[_COL_ACCIDENT]),
                int(raw[_COL_END]), int(raw[_COL_TOTAL]),
            )
            if total <= 0 or not 0 <= start < end <= total:
                raise ValueError(
                    f"{path}:{line_no}: window [{start},{end}] not inside "
                    f"[0,{total}] for {fault_label}/{key}"
                )
            rows.append(_CsvRow(key, fault_label, start, accident, end, total))
    if not rows:
        raise ValueError(f"No accident rows parsed from {path}")
    if skipped_flag:
        LOGGER.warning("Skipped %d rows with accident flag != 1", skipped_flag)
    LOGGER.info("Parsed %d accident rows from %s", len(rows), path)
    return rows


def resolve_annotated_records(
    rows: list[_CsvRow],
    frames_dir: Path,
    folders_by_fault: dict[str, dict[tuple[int, int], Path]],
    allow_missing: bool = False,
) -> list[DadaRecord]:
    """Attach on-disk frame counts to CSV rows; normalize the window to a fraction.

    ``total_frames`` in the returned record is the **on-disk** image count (the
    extractor reads those images); ``span`` is the fraction implied by the
    CSV's own ``total_frames``, invariant to any disagreement between the two
    (mirrors ``core.data.dota.resolve_frame_counts``).
    """
    records: list[DadaRecord] = []
    missing: list[str] = []
    for row in rows:
        folder = folders_by_fault[row.fault_label].get(row.video_id_key)
        if folder is None:
            missing.append(f"{row.fault_label}/type{row.video_id_key[0]}_vid{row.video_id_key[1]}")
            continue
        on_disk = len(list_frame_images(folder))
        if on_disk == 0:
            missing.append(video_id_from_path(folder))
            continue
        if on_disk != row.total_frames:
            LOGGER.warning(
                "%s: %d images on disk but annotation says %d frames; using disk",
                folder.name, on_disk, row.total_frames,
            )
        folder_name = video_id_from_path(folder)
        records.append(
            DadaRecord(
                video_id=_make_video_id(row.fault_label, folder_name),
                folder_name=folder_name,
                class_name=constants.DADA_CLASS_NAME,
                fault_label=row.fault_label,
                total_frames=on_disk,
                span=(row.start / row.total_frames, row.end / row.total_frames),
                accident_frac=row.accident / row.total_frames,
            )
        )
    if missing:
        message = (
            f"{len(missing)}/{len(rows)} annotated clips have no readable frame "
            f"folder under {frames_dir}: {sorted(missing)[:5]}"
        )
        if not allow_missing:
            raise ValueError(f"{message}. Finish the unzip, or pass --allow-missing-frames")
        LOGGER.warning("%s -- dropped from the corpus", message)
    if not records:
        raise ValueError(f"No annotated clip has readable frames under {frames_dir}")
    return records


def resolve_weak_abnormal_records(
    folders_by_fault: dict[str, dict[tuple[int, int], Path]],
    annotated_keys: set[tuple[str, tuple[int, int]]],
    exclude: bool = False,
) -> list[DadaRecord]:
    """Frame folders under a fault directory with no CSV row: known label, no window."""
    if exclude:
        dropped = sum(
            1
            for fault_label in FAULT_DIRNAMES
            for key in folders_by_fault.get(fault_label, {})
            if (fault_label, key) not in annotated_keys
        )
        if dropped:
            LOGGER.info("Excluded %d unannotated abnormal frame folders", dropped)
        return []

    records: list[DadaRecord] = []
    for fault_label in FAULT_DIRNAMES:
        for key, folder in folders_by_fault.get(fault_label, {}).items():
            if (fault_label, key) in annotated_keys:
                continue
            folder_name = video_id_from_path(folder)
            records.append(
                DadaRecord(
                    video_id=_make_video_id(fault_label, folder_name),
                    folder_name=folder_name,
                    class_name=constants.DADA_CLASS_NAME,
                    fault_label=fault_label,
                    total_frames=len(list_frame_images(folder)),
                    span=None,
                    accident_frac=None,
                )
            )
    if records:
        LOGGER.info(
            "%d abnormal frame folders have no CSV row -- weak-labeled, forced to train",
            len(records),
        )
    return records


def _normal_record(folder: Path) -> DadaRecord:
    folder_name = video_id_from_path(folder)
    return DadaRecord(
        video_id=_make_video_id(constants.DADA_NORMAL_DIRNAME, folder_name),
        folder_name=folder_name,
        class_name=NORMAL_CLASS,
        fault_label=constants.DADA_NORMAL_DIRNAME,
        total_frames=len(list_frame_images(folder)),
        span=None,
        accident_frac=None,
    )


def resolve_normal_records(normal_dir: Path) -> list[DadaRecord]:
    """Every frame folder under ``0_Normal_Driving`` -- no window, no CSV row."""
    records = [_normal_record(folder) for folder in list_frame_folders(normal_dir)]
    LOGGER.info("%d normal frame folders under %s", len(records), normal_dir)
    return records


def _assert_no_id_collisions(*record_groups: list[DadaRecord]) -> None:
    """Defensive only: ``video_id`` is fault-dir-prefixed and unique by construction."""
    seen: set[str] = set()
    for group in record_groups:
        for record in group:
            if record.video_id in seen:
                raise ValueError(f"video id {record.video_id!r} is duplicated -- internal bug")
            seen.add(record.video_id)


def _source_folder(
    record: DadaRecord,
    folders_by_fault: dict[str, dict[tuple[int, int], Path]],
    normal_dir: Path,
) -> Path:
    """The real on-disk folder a record's ``video_id`` was derived from."""
    if record.fault_label == constants.DADA_NORMAL_DIRNAME:
        return normal_dir / record.folder_name
    return folders_by_fault[record.fault_label][_type_and_vid(record.folder_name)]


def materialize_flat_dir(
    records: list[DadaRecord],
    folders_by_fault: dict[str, dict[tuple[int, int], Path]],
    normal_dir: Path,
    flat_dir: Path,
) -> None:
    """Symlink farm: ``flat_dir/{video_id} -> real frame folder``.

    ``extract_clip_features.py``/``raft_extract.py --frames-dir`` key their
    (dataset-generic) folder scan on bare ``Path.name``, so they cannot resolve
    a fault-dir-prefixed ``video_id`` against ``frames_dir`` directly. Pointing
    them at this flat dir instead needs no change to either tool.
    """
    flat_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for record in records:
        target = flat_dir / record.video_id
        if target.is_symlink() or target.exists():
            continue
        source = _source_folder(record, folders_by_fault, normal_dir)
        target.symlink_to(source.resolve(), target_is_directory=True)
        written += 1
    LOGGER.info("Materialized %d new symlinks (of %d records) under %s",
                written, len(records), flat_dir)


def sampled_frame_labels(record: DadaRecord, stride: int) -> list[int]:
    """Per-sampled-frame 0/1 labels from the normalized span (all-zero if none)."""
    length = num_sampled_frames(record.total_frames, stride)
    labels = [0] * length
    if record.span is None:
        return labels
    start = max(0, min(length, round(record.span[0] * length)))
    end = max(0, min(length, round(record.span[1] * length)))
    for index in range(start, end):
        labels[index] = 1
    return labels


def build_frame_labels(
    records: list[DadaRecord], stride: int, strict: bool = False
) -> dict[str, list[int]]:
    """Frame labels for the test split; warns (or raises) on windows that vanish."""
    frame_labels = {r.video_id: sampled_frame_labels(r, stride) for r in records}
    vanished = sorted(
        r.video_id for r in records if r.is_abnormal and not any(frame_labels[r.video_id])
    )
    if vanished:
        message = (
            f"{len(vanished)} abnormal test clips lose their anomaly window at "
            f"stride {stride}: {vanished[:5]}"
        )
        if strict:
            raise ValueError(message)
        LOGGER.warning("%s -- they contribute only negative frames", message)
    return frame_labels


def split_records(
    candidates: list[DadaRecord],
    seed: int = constants.SEED,
    test_ratio_abnormal: float = TEST_RATIO_ABNORMAL,
    test_ratio_normal: float = TEST_RATIO_NORMAL,
    split_file: Path | None = None,
) -> tuple[list[DadaRecord], list[DadaRecord]]:
    """Split annotated-abnormal + normal candidates; stratified by (abnormal, fault_label).

    Weak-abnormal records never reach this function -- they are forced into
    train by the caller. With ``split_file`` the listed ids form the test set.
    """
    if split_file is not None:
        test_ids = _read_split_file(split_file)
        unknown = test_ids - {r.video_id for r in candidates}
        if unknown:
            raise ValueError(
                f"Split file lists unknown/windowless video ids: {sorted(unknown)[:5]}"
            )
        train = [r for r in candidates if r.video_id not in test_ids]
        test = [r for r in candidates if r.video_id in test_ids]
    else:
        # deterministic split shuffling, not security-sensitive
        rng = random.Random(seed)  # nosec B311
        groups: dict[tuple[bool, str], list[DadaRecord]] = {}
        for rec in candidates:
            groups.setdefault((rec.is_abnormal, rec.fault_label), []).append(rec)
        train, test = [], []
        for (is_abn, _fault), group in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1])
        ):
            group = sorted(group, key=lambda r: r.video_id)
            rng.shuffle(group)
            ratio = test_ratio_abnormal if is_abn else test_ratio_normal
            n_test = round(len(group) * ratio)
            test.extend(group[:n_test])
            train.extend(group[n_test:])
    LOGGER.info(
        "Split: train=%d (%d abnormal) test=%d (%d abnormal)",
        len(train), sum(r.is_abnormal for r in train),
        len(test), sum(r.is_abnormal for r in test),
    )
    return train, test


def _read_split_file(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {Path(line.strip()).stem for line in text.splitlines() if line.strip()}


def check_definition_coverage(class_names: set[str]) -> None:
    """Every emitted class must have definition sentences (lesson 19)."""
    known = set(DATASET_CLS_DEFS[dataset_abbr(constants.DADA_DATASET)])
    undefined = sorted(class_names - known)
    if undefined:
        raise ValueError(
            f"{len(undefined)} DADA-2000 classes have no definition sentences: "
            f"{undefined}. Add them to DATASET_CLS_DEFS['dada'] in "
            "core/data/definitions.py"
        )


def _meta(
    train: list[DadaRecord], test: list[DadaRecord], frame_labels_test: dict[str, list[int]],
    stride: int,
) -> dict[str, dict[str, object]]:
    def entry(r: DadaRecord, split: str) -> dict[str, object]:
        sampled = frame_labels_test.get(r.video_id) if split == "test" else None
        sampled_frames = (
            len(sampled) if sampled is not None else num_sampled_frames(r.total_frames, stride)
        )
        return {
            "class_name": r.class_name,
            "fault_label": r.fault_label,
            "ego_involve": r.is_ego,
            "split": split,
            "total_frames": r.total_frames,
            "sampled_frames": sampled_frames,
            "positive_frames": sum(sampled) if sampled is not None else None,
            "normalized_span": list(r.span) if r.span is not None else None,
            "accident_frac": r.accident_frac,
        }

    meta: dict[str, dict[str, object]] = {r.video_id: entry(r, "train") for r in train}
    meta.update({r.video_id: entry(r, "test") for r in test})
    return meta


def preprocess(
    metadata: Path,
    frames_dir: Path,
    out_dir: Path,
    stride: int = constants.FRAME_STRIDE,
    seed: int = constants.SEED,
    test_ratio_abnormal: float = TEST_RATIO_ABNORMAL,
    test_ratio_normal: float = TEST_RATIO_NORMAL,
    split_file: Path | None = None,
    exclude_unannotated_abnormal: bool = False,
    allow_missing_frames: bool = False,
    strict: bool = False,
    flat_frames_dir: Path | None = None,
    dry_run: bool = False,
) -> tuple[list[DadaRecord], list[DadaRecord]]:
    """Build the standard dataset files for DADA-2000; returns ``(train, test)``."""
    folders_by_fault = {
        name: frame_folders_by_type_vid(frames_dir / name) for name in FAULT_DIRNAMES
    }
    normal_dir = frames_dir / constants.DADA_NORMAL_DIRNAME

    rows = parse_metadata_csv(metadata)
    annotated = resolve_annotated_records(rows, frames_dir, folders_by_fault, allow_missing_frames)
    annotated_keys = {(r.fault_label, _type_and_vid(r.folder_name)) for r in annotated}
    weak = resolve_weak_abnormal_records(
        folders_by_fault, annotated_keys, exclude_unannotated_abnormal
    )
    normal = resolve_normal_records(normal_dir)
    _assert_no_id_collisions(annotated, weak, normal)

    train_candidates, test = split_records(
        annotated + normal, seed, test_ratio_abnormal, test_ratio_normal, split_file
    )
    train = train_candidates + weak
    frame_labels_test = build_frame_labels(test, stride, strict)

    class_names = {r.class_name for r in train} | {r.class_name for r in test}
    check_definition_coverage(class_names)

    LOGGER.info(
        "DADA-2000: %d train (%d abnormal: %d annotated + %d weak / %d normal), "
        "%d test (%d abnormal / %d normal)",
        len(train), sum(r.is_abnormal for r in train), len(train_candidates) - sum(
            1 for r in train_candidates if not r.is_abnormal
        ), len(weak), sum(1 for r in train if not r.is_abnormal),
        len(test), sum(r.is_abnormal for r in test), sum(1 for r in test if not r.is_abnormal),
    )
    if dry_run:
        LOGGER.info("Dry run: no files written")
        return train, test

    write_dataset_files(
        out_dir,
        labels_train={r.video_id: int(r.is_abnormal) for r in train},
        frame_labels_test=frame_labels_test,
        defs=class_name_list(class_names),
        meta=_meta(train, test, frame_labels_test, stride),
    )
    write_train_ids(out_dir, [r.video_id for r in train])
    write_test_ids(out_dir, [r.video_id for r in test])
    if flat_frames_dir is not None:
        materialize_flat_dir(train + test, folders_by_fault, normal_dir, flat_frames_dir)
    return train, test


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Cleaned_Metadata.csv")
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="root holding 0_Non_Ego_Fault/, 0_Normal_Driving/, 1_Ego_Fault/")
    parser.add_argument("--out-dir", type=Path,
                        default=constants.DATA_ROOT / constants.DADA_DATASET)
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--seed", type=int, default=constants.SEED)
    parser.add_argument("--test-ratio-abnormal", type=float, default=TEST_RATIO_ABNORMAL)
    parser.add_argument("--test-ratio-normal", type=float, default=TEST_RATIO_NORMAL)
    parser.add_argument("--split-file", type=Path, default=None,
                        help="test video ids, one per line (overrides the seeded split)")
    parser.add_argument("--exclude-unannotated-abnormal", action="store_true",
                        help="drop abnormal frame folders with no CSV row instead of "
                             "folding them into train as weak-labeled")
    parser.add_argument("--allow-missing-frames", action="store_true",
                        help="drop annotated clips with no readable frame folder "
                             "instead of failing")
    parser.add_argument("--strict", action="store_true",
                        help="fail instead of warning when a test window rounds away")
    parser.add_argument("--flat-frames-dir", type=Path, default=None,
                        help="materialize a flat directory of symlinks named by the "
                             "disambiguated video id (fault_dir__type<N>_vid<N>), for "
                             "--frames-dir on extract_clip_features.py/raft_extract.py -- "
                             "works around type<N>_vid<N> folder names repeating across "
                             "the three fault-attribution directories")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    preprocess(
        metadata=args.metadata,
        frames_dir=args.frames_dir,
        out_dir=args.out_dir,
        stride=args.stride,
        seed=args.seed,
        test_ratio_abnormal=args.test_ratio_abnormal,
        test_ratio_normal=args.test_ratio_normal,
        split_file=args.split_file,
        exclude_unannotated_abnormal=args.exclude_unannotated_abnormal,
        allow_missing_frames=args.allow_missing_frames,
        strict=args.strict,
        flat_frames_dir=args.flat_frames_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
