"""DADA-2000 **original** release -> the T2 windowed corpus (Phase 2).

`.project/plans/katvad-dada-original-phase2-t2.md`. A different corpus from
:mod:`core.data.dada`, not a variant of it:

============  =========================================  ==============================
              trimmed archive (:mod:`core.data.dada`)    original release (this module)
============  =========================================  ==============================
annotation    ``Cleaned_Metadata.csv``, 975 rows          ``dada标注.xlsx``, 1,962 rows
key           ``(Fault_Label, type, video)``              ``(type, video)`` -- no such column
layout        ``{fault_dir}/type<N>_vid<N>/``             ``DADA2000/{type}/{video:03d}/images/``
normal clips  ``0_Normal_Driving`` (~3x longer)           **none at all**
frames        trimmed, abnormal median 56 vs orig. 322    untrimmed, aggregate delta +0.30 %
============  =========================================  ==============================

**Why a separate module.** ``dada.preprocess`` already takes 18 parameters and
holds every measured trimmed-corpus number; five more conditionals inside it can
silently change a corpus that is nobody's to change any more (lesson **C2**).
Everything reusable *is* reused: :class:`~core.data.dada.DadaRecord`,
``sampled_frame_labels``, ``plan_record_windows``, ``window_label``,
``write_windowed`` and ``check_definition_coverage`` all come from there, so the
two corpora cannot drift apart in label convention or window geometry.

**Negatives come from inside the accident videos.** That is the whole point of
T2. The archive's negatives were a separate ~3x longer pool, which let clip
length carry the label (a frame-count-only ruler scored micro **0.8654**, lesson
**C28**); cutting fixed-length windows out of the accident videos themselves puts
the leak at **0.5000** by construction. The negatives share camera, weather and
scene with the positives and sit seconds apart.

The window length is what decides whether the corpus is measurable at all, and it
was **measured, not predicted** (Gate W, 2026-09-16, real archive census): at
W=16/hop 8 the clip oracle is **0.7529** and Gate W FAILS; at
``DADA_ORIGIN_WINDOW_LENGTH`` = **20**/hop 8 it is **0.7037** with 95.7 % of the
abnormal sources kept and 798 two-class test windows, and every criterion passes.
W=24 would reach 0.656 but drops retention to 0.899 (C32). The per-clip cap is
**not** the lever here: it spans 0.7527-0.7529 across caps 2-6 (lesson C35).

**Labels use absolute frame indices** (plan §3.1). ``span`` is normalized against
the **on-disk** count, not the annotation's ``total frames``: where the release is
trimmed the two disagree, and the fraction convention then rescales by ``D/A``
and drags an end-of-clip anomaly into the middle -- 17 sampled frames on
``t05_v040``, measured. Expressing absolute indices as an on-disk fraction reuses
``sampled_frame_labels`` unchanged, so no second label code path exists; it
differs from a strict ``floor``/``ceil`` mapping by at most one sampled frame per
boundary, the same rounding band Phase 1 derived and measured.

**Ids are ``t{type:02d}_v{video:03d}``** -- identical to Phase 1's, so the
existing ``clip/DADA2000_orig`` feature cache is reused rather than re-extracted.
The bare on-disk folder name is the video number alone, which collides across all
52 types (**1,962 clips -> 255 ids**, lesson **C26**); ``--flat-frames-dir``
materializes a symlink farm of unique ids for the extractors, linked at the
``images`` directory itself because ``pathlib`` will not recurse into a symlinked
directory.

Usage::

    python -m core.data.dada_origin \\
        --annotation data/DADA/dada标注.xlsx \\
        --frames-dir /content/dada_origin \\
        --out-dir data/DADA2000_orig \\
        --flat-frames-dir /content/flat

    # sharded (94 GiB of images do not fit): census per shard, then labels once
    python -m core.data.dada_origin ... --counts-out counts/03.json --census-only
    python -m core.data.dada_origin ... --counts-file counts/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from core import constants
from core.data.dada import (
    DadaRecord,
    check_definition_coverage,
    write_windowed,
)
from core.data.video_io import list_frame_images
from core.data.xlsx import Sheet, find_sheet, normalize_header

LOGGER = logging.getLogger(__name__)

# --- annotation columns, verbatim (normalized by core.data.xlsx) -----------
COL_VIDEO = "video"
COL_TYPE = "type"
COL_ACCIDENT_FLAG = "whether an accident occurred (1/0)"
COL_START = "abnormal start frame"
COL_ACCIDENT = "accident frame"
COL_END = "abnormal end frame"
COL_TOTAL = "total frames"

#: Columns that define a row. The sheet is detected by carrying all of these.
REQUIRED_COLUMNS = (COL_TYPE, COL_VIDEO, COL_START, COL_END, COL_TOTAL)

#: Carried into ``meta.json`` only -- never into ``labels_train.json`` (plan §3.4).
#: ``texts`` is a 81-value category label, not a per-video description, so it
#: cannot drive ``L_neg``: two clips sharing a string give
#: ``asymmetric_infonce_loss`` two identical rows against a diagonal target, an
#: unsatisfiable objective, and at ``BATCH_SIZE`` 64 that is ~78 % of a batch.
#: Kept because the composite of all seven is 57.3 % unique, which would be
#: usable -- as a declared extra-supervision arm, after Gate W, never by default.
META_COLUMNS = (
    "weather(sunny,rainy,snowy,foggy)1-4",
    "light(day,night)1-2",
    "scenes(highway,tunnel,mountain,urban,rural)1-5",
    "linear(arterials,curve,intersection,t-junction,ramp) 1-5",
    "texts",
    "causes",
    "measures",
)
#: ``meta.json`` key for each of the above, in the same order.
META_KEYS = ("weather", "light", "scenes", "linear", "texts", "causes", "measures")


@dataclass(frozen=True)
class OriginRow:
    """One annotated accident clip, straight from the workbook."""

    type_id: int
    video: int
    start: int
    accident: int
    end: int
    total_frames: int
    attributes: dict[str, str]

    @property
    def video_id(self) -> str:
        return video_id(self.type_id, self.video)


def video_id(type_id: int, video: int) -> str:
    """``t{type:02d}_v{video:03d}`` -- unique across the 52 types (lesson **C26**).

    Deliberately unlike :func:`core.data.dada._make_video_id`'s
    ``{fault_label}__{folder}``: the original release has no ``Fault_Label``, and
    the two corpora must never share a feature cache (lesson **C2**).
    """
    return f"t{type_id:02d}_v{video:03d}"


def clip_folder(frames_dir: Path, type_id: int, video: int) -> Path:
    """The measured on-disk layout (``core/docs/DADA_ORIGIN_PHASE0.md`` §3.1)."""
    return frames_dir / constants.DADA_ORIGIN_ROOT_DIRNAME / str(type_id) / f"{video:03d}"


def _cell(sheet: Sheet, row: dict[int, str], column: str) -> str:
    return row.get(sheet.column(column), "").strip()


def parse_annotation(path: Path) -> list[OriginRow]:
    """Read the workbook into validated per-clip accident windows.

    Rows are skipped, with a count, when the accident flag is not ``1`` or when
    the window is not inside ``[0, total]`` -- the file is hand-maintained and a
    handful of rows carry blanks. A row that parses is trusted; a sheet that does
    not carry every required column raises with each sheet's header listed.
    """
    sheet = find_sheet(path, REQUIRED_COLUMNS)
    has_flag = normalize_header(COL_ACCIDENT_FLAG) in sheet.header
    has_accident = normalize_header(COL_ACCIDENT) in sheet.header
    if not has_flag:
        LOGGER.warning(
            "%s has no %r column; every row with a valid window is treated as an "
            "accident row", path.name, COL_ACCIDENT_FLAG,
        )

    rows: list[OriginRow] = []
    seen: set[tuple[int, int]] = set()
    skipped_flag = skipped_window = skipped_parse = 0
    for raw in sheet.rows:
        if has_flag and _cell(sheet, raw, COL_ACCIDENT_FLAG) != "1":
            skipped_flag += 1
            continue
        try:
            type_id = int(_cell(sheet, raw, COL_TYPE))
            video = int(_cell(sheet, raw, COL_VIDEO))
            start = int(_cell(sheet, raw, COL_START))
            end = int(_cell(sheet, raw, COL_END))
            total = int(_cell(sheet, raw, COL_TOTAL))
            accident = int(_cell(sheet, raw, COL_ACCIDENT)) if has_accident else start
        except ValueError:
            skipped_parse += 1
            continue
        if not 0 <= start < end <= total:
            skipped_window += 1
            continue
        key = (type_id, video)
        if key in seen:
            raise ValueError(f"{path}: duplicate annotation row for type {type_id} video {video}")
        seen.add(key)
        rows.append(
            OriginRow(
                type_id=type_id,
                video=video,
                start=start,
                accident=accident,
                end=end,
                total_frames=total,
                attributes={
                    key_name: _cell(sheet, raw, column)
                    for key_name, column in zip(META_KEYS, META_COLUMNS, strict=True)
                    if normalize_header(column) in sheet.header
                },
            )
        )
    if not rows:
        raise ValueError(f"No valid accident rows parsed from {path}")
    LOGGER.info(
        "Parsed %d accident rows from %s (skipped %d by flag, %d by window, %d unparsable)",
        len(rows), path.name, skipped_flag, skipped_window, skipped_parse,
    )
    return rows


def make_record(row: OriginRow, frames_on_disk: int) -> DadaRecord | None:
    """A :class:`DadaRecord` whose ``span`` is **absolute indices** over the disk count.

    ``None`` when the anomaly starts at or past the last frame on disk: the
    release trimmed that clip before its anomaly, so it carries no positive frame
    and windowing it would add only negatives under an abnormal id. The condition
    is the threshold (lesson **C33**) -- there is no separate tolerance to keep in
    sync with this code.
    """
    if frames_on_disk <= 0 or row.start >= frames_on_disk:
        return None
    end = min(row.end, frames_on_disk)
    return DadaRecord(
        video_id=row.video_id,
        folder_name=row.video_id,
        class_name=constants.DADA_CLASS_NAME,
        fault_label=constants.DADA_ORIGIN_FAULT_SENTINEL,
        total_frames=frames_on_disk,
        span=(row.start / frames_on_disk, end / frames_on_disk),
        accident_frac=min(row.accident, frames_on_disk) / frames_on_disk,
    )


def frame_census(rows: list[OriginRow], frames_dir: Path) -> dict[str, int]:
    """``{video_id: images on disk}`` for every row whose folder holds images.

    An existing directory is not evidence of data (lesson **C10**): a clip with a
    folder but no readable image is simply absent from the census, and every
    caller treats absent as missing.
    """
    counts: dict[str, int] = {}
    for row in rows:
        folder = clip_folder(frames_dir, row.type_id, row.video)
        images = list_frame_images(folder, constants.DADA_ORIGIN_IMAGES_SUBDIR)
        if images:
            counts[row.video_id] = len(images)
    LOGGER.info("Frame census: %d/%d clips have readable images under %s",
                len(counts), len(rows), frames_dir)
    return counts


def resolve_records(
    rows: list[OriginRow], counts: dict[str, int], allow_missing: bool = False
) -> list[DadaRecord]:
    """Join annotation rows to a frame census; report both ways a clip is lost."""
    records: list[DadaRecord] = []
    missing: list[str] = []
    trimmed_away: list[str] = []
    delta = 0
    for row in rows:
        on_disk = counts.get(row.video_id)
        if on_disk is None:
            missing.append(row.video_id)
            continue
        if on_disk != row.total_frames:
            delta += 1
        record = make_record(row, on_disk)
        if record is None:
            trimmed_away.append(row.video_id)
            continue
        records.append(record)
    if missing:
        message = (
            f"{len(missing)}/{len(rows)} annotated clips have no readable frames: "
            f"{sorted(missing)[:5]}"
        )
        if not allow_missing:
            raise ValueError(f"{message}. Finish the extraction, or pass --allow-missing-frames")
        LOGGER.warning("%s -- dropped from the corpus", message)
    if trimmed_away:
        LOGGER.warning(
            "%d clips are trimmed before their anomaly starts and carry no positive "
            "frame: %s -- dropped", len(trimmed_away), sorted(trimmed_away)[:5],
        )
    if delta:
        LOGGER.info(
            "%d/%d clips disagree with the annotation's frame count; labels use the "
            "ON-DISK count (absolute indices), which is what makes them correct there",
            delta, len(rows),
        )
    if not records:
        raise ValueError("No annotated clip survived the frame census")
    return records


def split_by_type(
    records: list[DadaRecord],
    rows_by_id: dict[str, OriginRow],
    seed: int = constants.SEED,
    test_ratio: float = constants.DADA_ORIGIN_TEST_RATIO,
) -> tuple[list[DadaRecord], list[DadaRecord]]:
    """Stratify the split by accident **type**, over SOURCE VIDEOS.

    The original release ships zero normal videos, so
    :func:`core.data.dada.split_records`' ``(is_abnormal, fault_label)`` key
    collapses to a single group and the 52-type taxonomy can drift between the
    splits. Splitting on **windows** instead of sources would be worse still --
    T2 gives ~3.8 windows per video, so the same accident would appear in both;
    ``write_windowed`` re-asserts that and raises.
    """
    groups: dict[int, list[DadaRecord]] = {}
    for record in records:
        groups.setdefault(rows_by_id[record.video_id].type_id, []).append(record)

    # deterministic split shuffling, not security-sensitive
    rng = random.Random(seed)  # nosec B311
    train: list[DadaRecord] = []
    test: list[DadaRecord] = []
    for type_id in sorted(groups):
        group = sorted(groups[type_id], key=lambda r: r.video_id)
        rng.shuffle(group)
        n_test = round(len(group) * test_ratio)
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    if not test or not train:
        raise ValueError(
            f"Split at ratio {test_ratio} left train={len(train)} test={len(test)}; "
            "one side is empty"
        )
    LOGGER.info(
        "Split by type: train=%d test=%d source videos over %d types",
        len(train), len(test), len(groups),
    )
    return train, test


def annotation_meta(rows_by_id: dict[str, OriginRow]) -> dict[str, dict[str, object]]:
    """Per-source annotation columns for ``meta.json`` (plan §3.4).

    ``meta.json`` only. These never reach ``labels_train.json`` and nothing in the
    training path reads them; they exist so a later caption experiment does not
    have to rebuild 94 GiB of corpus to get them.
    """
    meta: dict[str, dict[str, object]] = {}
    for vid, row in rows_by_id.items():
        entry: dict[str, object] = {
            "type": row.type_id,
            "video": row.video,
            "annotation_total_frames": row.total_frames,
            "annotation_span_frames": [row.start, row.end],
            "annotation_accident_frame": row.accident,
        }
        entry.update(row.attributes)
        meta[vid] = entry
    return meta


def materialize_flat_dir(
    records: list[DadaRecord], frames_dir: Path, flat_dir: Path,
    rows_by_id: dict[str, OriginRow],
) -> int:
    """``flat_dir/{video_id} -> DADA2000/{type}/{video:03d}/images`` (lesson **C26**).

    The link points at the ``images`` directory **itself**, and the extractors are
    then run *without* ``--frames-subdir``. ``pathlib`` refuses to recurse into a
    symlinked directory (cycle guard, every version through 3.13), so a clip-level
    farm plus ``--frames-subdir images`` makes ``list_frame_folders`` walk
    ``rglob("images")``, never descend through the link, and raise "No frame
    folders found". Measured in Phase 1, not assumed.
    """
    flat_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for record in records:
        target = flat_dir / record.video_id
        if target.is_symlink() or target.exists():
            continue
        row = rows_by_id[record.video_id]
        source = clip_folder(frames_dir, row.type_id, row.video)
        target.symlink_to(
            (source / constants.DADA_ORIGIN_IMAGES_SUBDIR).resolve(), target_is_directory=True
        )
        written += 1
    LOGGER.info("Flat dir: %d new symlinks, %d records total under %s",
                written, len(records), flat_dir)
    return written


def load_counts(patterns: list[str]) -> dict[str, int]:
    """Merge shard censuses written by ``--counts-out``; a later shard wins."""
    counts: dict[str, int] = {}
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    if not paths:
        raise ValueError(f"No census file matched {patterns}")
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        counts.update({str(k): int(v) for k, v in payload.items()})
    LOGGER.info("Merged %d census files -> %d clips", len(paths), len(counts))
    return counts


def census_pass(
    annotation: Path,
    frames_dir: Path,
    counts_out: Path | None = None,
    flat_frames_dir: Path | None = None,
) -> dict[str, int]:
    """One shard's pass: census what is on disk, and farm symlinks for it.

    The sharded route needs both, and it needs them **without** building a
    dataset: a shard holds ~150 of the corpus's clips, so any split computed from
    it would be wrong, and the four dataset files are written once at the end
    from the merged censuses instead.
    """
    rows = parse_annotation(annotation)
    counts = frame_census(rows, frames_dir)
    if counts_out is not None:
        counts_out.parent.mkdir(parents=True, exist_ok=True)
        part = counts_out.with_suffix(counts_out.suffix + constants.CACHE_PART_SUFFIX)
        part.write_text(json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8")
        part.replace(counts_out)  # atomic, lesson C11
        LOGGER.info("Wrote census of %d clips to %s", len(counts), counts_out)
    if flat_frames_dir is not None:
        records = resolve_records(rows, counts, allow_missing=True)
        rows_by_id = {row.video_id: row for row in rows}
        materialize_flat_dir(records, frames_dir, flat_frames_dir, rows_by_id)
    return counts


def preprocess(
    annotation: Path,
    out_dir: Path,
    frames_dir: Path | None = None,
    counts: dict[str, int] | None = None,
    stride: int = constants.FRAME_STRIDE,
    seed: int = constants.SEED,
    test_ratio: float = constants.DADA_ORIGIN_TEST_RATIO,
    window_length: int = constants.DADA_ORIGIN_WINDOW_LENGTH,
    window_stride: int = constants.DADA_ORIGIN_WINDOW_STRIDE,
    window_min_positive: int = constants.WINDOW_MIN_POSITIVE,
    window_max_per_clip: int = constants.WINDOW_MAX_PER_CLIP,
    allow_missing_frames: bool = False,
    flat_frames_dir: Path | None = None,
    dry_run: bool = False,
) -> tuple[list[DadaRecord], list[DadaRecord]]:
    """Build the T2 windowed corpus; returns ``(train, test)`` **source** records.

    Exactly one of ``frames_dir`` (measure the disk) or ``counts`` (a census from
    a previous sharded pass) supplies the frame counts. The sharded route exists
    because ``images`` is 94.01 GiB over six spanned-zip volumes and no runtime
    holds it: encode a shard, record its census, delete the frames, repeat.
    """
    if (frames_dir is None) == (counts is None):
        raise ValueError("Pass exactly one of frames_dir (measure) or counts (census)")
    # Checked before anything is parsed or written: on the sharded route the
    # frames are already deleted, and finding that out after the build is a
    # wasted pass over a corpus whose images are 94 GiB.
    if flat_frames_dir is not None and frames_dir is None:
        raise ValueError("--flat-frames-dir needs --frames-dir: symlinks need real folders")

    rows = parse_annotation(annotation)
    if counts is None:
        assert frames_dir is not None  # nosec B101 -- guarded above, for the type checker
        counts = frame_census(rows, frames_dir)
    records = resolve_records(rows, counts, allow_missing_frames)
    rows_by_id = {row.video_id: row for row in rows}

    train, test = split_by_type(records, rows_by_id, seed, test_ratio)
    class_names = {r.class_name for r in records}
    check_definition_coverage(class_names)

    if dry_run:
        LOGGER.info("Dry run: no files written")
        return train, test

    write_windowed(
        out_dir,
        train,
        test,
        stride,
        class_names,
        window_length,
        window_stride,
        window_min_positive,
        # Every record here is annotated: the original release has no clip whose
        # label is known but whose window is not, so weak_mode can never fire.
        "drop",
        window_max_per_clip,
        annotation_meta(rows_by_id),
    )
    if flat_frames_dir is not None and frames_dir is not None:
        materialize_flat_dir(records, frames_dir, flat_frames_dir, rows_by_id)
    return train, test


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--annotation", type=Path, required=True,
                        help=f"the original {constants.DADA_ORIGIN_ANNOTATION_FILENAME}")
    parser.add_argument("--frames-dir", type=Path, default=None,
                        help=f"root holding {constants.DADA_ORIGIN_ROOT_DIRNAME}/"
                             "{type}/{video:03d}/images/")
    parser.add_argument("--counts-file", nargs="+", default=None,
                        help="shard censuses (globs allowed) to use INSTEAD of reading "
                             "the disk -- the sharded route, when the frames are gone")
    parser.add_argument("--counts-out", type=Path, default=None,
                        help="write this pass's {video_id: frames} census here")
    parser.add_argument("--census-only", action="store_true",
                        help="write --counts-out and stop; builds no dataset")
    parser.add_argument("--out-dir", type=Path,
                        default=constants.DATA_ROOT / constants.DADA_ORIGIN_DATASET)
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--seed", type=int, default=constants.SEED)
    parser.add_argument("--test-ratio", type=float, default=constants.DADA_ORIGIN_TEST_RATIO,
                        help="fraction of SOURCE VIDEOS held out, stratified by type")
    parser.add_argument("--window-length", type=int,
                        default=constants.DADA_ORIGIN_WINDOW_LENGTH,
                        help=f"T2 = {constants.DADA_ORIGIN_WINDOW_LENGTH} sampled frames")
    parser.add_argument("--window-stride", type=int,
                        default=constants.DADA_ORIGIN_WINDOW_STRIDE,
                        help=f"T2 = hop {constants.DADA_ORIGIN_WINDOW_STRIDE}")
    parser.add_argument("--window-min-positive", type=int,
                        default=constants.WINDOW_MIN_POSITIVE)
    parser.add_argument("--window-max-per-clip", type=int,
                        default=constants.WINDOW_MAX_PER_CLIP,
                        help="cap on windows from ONE source clip, evenly spaced (C32)")
    parser.add_argument("--allow-missing-frames", action="store_true",
                        help="drop annotated clips with no readable frames instead of failing")
    parser.add_argument("--flat-frames-dir", type=Path, default=None,
                        help="materialize a symlink farm of unique ids for "
                             "extract_clip_features.py/raft_extract.py --frames-dir; "
                             "links point at the images directory itself (C26)")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if (args.frames_dir is None) == (args.counts_file is None):
        raise SystemExit("Pass exactly one of --frames-dir or --counts-file")

    if args.counts_out is not None or args.census_only:
        if args.frames_dir is None:
            raise SystemExit("--counts-out/--census-only need --frames-dir")
        census_pass(
            annotation=args.annotation,
            frames_dir=args.frames_dir,
            counts_out=args.counts_out,
            # On a census pass the farm covers this shard only -- that is the
            # point: the extractor is pointed at it before the frames are deleted.
            flat_frames_dir=args.flat_frames_dir if args.census_only else None,
        )
        if args.census_only:
            return

    preprocess(
        annotation=args.annotation,
        out_dir=args.out_dir,
        frames_dir=args.frames_dir,
        counts=load_counts(args.counts_file) if args.counts_file else None,
        stride=args.stride,
        seed=args.seed,
        test_ratio=args.test_ratio,
        window_length=args.window_length,
        window_stride=args.window_stride,
        window_min_positive=args.window_min_positive,
        window_max_per_clip=args.window_max_per_clip,
        allow_missing_frames=args.allow_missing_frames,
        flat_frames_dir=args.flat_frames_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
