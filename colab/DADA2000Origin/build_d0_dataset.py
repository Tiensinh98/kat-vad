"""Build the Gate-D0 dataset directories for the DADA-2000 **original** release.

Phase 1 of `.project/plans/katvad-dada-original-phase1-gate-d0.md`. This script
lives OUTSIDE ``core/`` on purpose: D0 is a hard-stop gate, so nothing is written
into the package until the gate is clear (plan §2, option B).

It writes the four files ``core.eda.corpus.load_dataset_files`` requires, plus a
symlink farm that lets the stock extractor see globally unique ids.

**It does not restate the label convention — it imports it.** ``DadaRecord`` and
``sampled_frame_labels`` come from ``core.data.dada``, so ``d0_frac`` carries
exactly the labels the Phase 4 pipeline would build:
``total_frames`` is the **on-disk** count and ``span`` is the fraction implied by
the annotation's own total (``core/data/dada.py:resolve_annotated_records``).
``d0_abs`` is the sanity arm: absolute indices, ``start // stride``.

Why the symlink farm (lesson **C26**): ``extract_clip_features.py:194`` keys its
folder scan on ``Path.name``, which on ``DADA2000/{type}/{video:03d}/images`` is
the video number alone — **1,962 clips collapse to 255 ids**, silently, and
``pending_items`` then reports a clean resume over a cache that is 87 % missing.

Three modes, because a GPU Colab runtime has far less disk than the CPU one
Phase 0 measured, and 400 clips of frames (~20 GiB) do not fit beside everything
else:

``full``
    resolve from disk, build the farm, write both dataset dirs. One pass; needs
    every clip's frames present at once.
``shard``
    resolve **this shard** from disk, build a shard farm, dump
    ``{video_id: frames_on_disk}`` to ``--counts-out``. Writes no dataset dir, so
    the caller may delete the frames straight after encoding features.
``labels``
    read the merged ``--counts-in`` files, touch **no** frames and build **no**
    farm, write both dataset dirs. Run once, after every shard is encoded.

Usage::

    # one pass (plenty of disk)
    python build_d0_dataset.py --mode full \\
        --clips /content/d0_clips.json --root /content/d0/frames \\
        --out-dir /content/d0/dataset --stride 8

    # per shard
    python build_d0_dataset.py --mode shard \\
        --clips /content/shard_03.json --root /content/d0/frames \\
        --out-dir /content/d0/shard_03 --counts-out /content/counts/03.json

    # once, at the end
    python build_d0_dataset.py --mode labels \\
        --clips /content/d0_clips.json --out-dir /content/d0/dataset \\
        --counts-in /content/counts/*.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.data.dada import DadaRecord, sampled_frame_labels
from core.data.dataset_files import class_name_list, num_sampled_frames
from core.data.video_io import list_frame_folders, list_frame_images, video_id_from_path

LOGGER = logging.getLogger("build_d0_dataset")

IMAGES_SUBDIR = "images"
FRAC_DIRNAME = "d0_frac"
ABS_DIRNAME = "d0_abs"
FLAT_DIRNAME = "flat"
#: Pre-registered bar of plan §3.2, with its derivation (lesson **C33**: derive
#: the attainable range BEFORE writing the threshold).
#: ``sampled_frame_labels`` rounds each boundary (``round(frac * length)``) while
#: the absolute arm floors the start and ceils the end, so the two can differ by
#: at most **one sampled frame per boundary** = 2 per clip. More than that means
#: the frame counts, not the rounding, disagree.
MAX_LABEL_DISAGREEMENT_FRAMES = 2


def video_id(row: dict[str, int]) -> str:
    """``t{type:02d}_v{video:03d}`` — unique, and it keeps the (type, video) key.

    Deliberately unlike ``core.data.dada._make_video_id``'s
    ``{fault_label}__{folder}``: the original release has no ``Fault_Label``, and
    the two corpora must never share a feature cache (lesson **C2**).
    """
    return f"t{int(row['type']):02d}_v{int(row['video']):03d}"


def clip_folder(root: Path, row: dict[str, int]) -> Path:
    """The measured on-disk layout (``DADA_ORIGIN_PHASE0.md`` §3.1)."""
    return root / "DADA2000" / str(int(row["type"])) / f"{int(row['video']):03d}"


def resolve_rows(
    rows: list[dict[str, int]], root: Path, allow_missing: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    """Attach the on-disk frame count to every picked row.

    Mirrors ``core.data.dada.resolve_annotated_records``: a clip with no readable
    images is dropped, not silently zero-length (lesson **C10** — an existing
    directory is not evidence of data).
    """
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        folder = clip_folder(root, row)
        images = list_frame_images(folder / IMAGES_SUBDIR)
        if not images:
            missing.append(f"{video_id(row)} ({folder})")
            continue
        entry: dict[str, Any] = dict(row)
        entry["video_id"] = video_id(row)
        entry["folder"] = folder
        entry["frames_on_disk"] = len(images)
        resolved.append(entry)
    if missing and not allow_missing:
        raise ValueError(
            f"{len(missing)}/{len(rows)} picked clips have no readable images "
            f"under {root}: {sorted(missing)[:5]}. Finish the extraction, or "
            "pass --allow-missing-frames"
        )
    if missing:
        LOGGER.warning("%d/%d clips dropped (no images): %s",
                       len(missing), len(rows), sorted(missing)[:5])
    if not resolved:
        raise ValueError(f"No picked clip has readable images under {root}")
    return resolved, missing


def entries_from_counts(
    rows: list[dict[str, int]], counts: dict[str, int]
) -> list[dict[str, Any]]:
    """Rebuild the resolved entries from a shard-time frame census.

    ``labels`` mode runs after the frames are gone, so the on-disk count cannot
    be re-measured — it is read back from what the shards recorded. A clip absent
    from the census was never encoded and is dropped, loudly: a label row with no
    feature row would fail ``core/eda/features.py:346`` much later (lesson C10 —
    the census, not the directory, is the evidence).
    """
    out: list[dict[str, Any]] = []
    absent: list[str] = []
    for row in rows:
        vid = video_id(row)
        if vid not in counts:
            absent.append(vid)
            continue
        entry: dict[str, Any] = dict(row)
        entry["video_id"] = vid
        entry["folder"] = None
        entry["frames_on_disk"] = int(counts[vid])
        out.append(entry)
    if absent:
        LOGGER.warning("%d/%d clips missing from the counts census (never "
                       "encoded?): %s", len(absent), len(rows), sorted(absent)[:5])
    if not out:
        raise ValueError("No clip survived the counts census")
    return out


def make_record(entry: dict[str, Any]) -> DadaRecord:
    """A real ``DadaRecord``, so the project's own label function applies.

    ``accident_frac`` is ``None``: the original annotation's accident-frame column
    is not carried by ``pick_probe.py`` and nothing in the features section reads
    it. ``fault_label`` is a sentinel — the original release has no such column.
    """
    total = int(entry["total"])
    return DadaRecord(
        video_id=entry["video_id"],
        folder_name=entry["video_id"],
        class_name=constants.DADA_CLASS_NAME,
        fault_label="origin",
        total_frames=int(entry["frames_on_disk"]),
        span=(int(entry["start"]) / total, int(entry["end"]) / total),
        accident_frac=None,
    )


def absolute_frame_labels(entry: dict[str, Any], stride: int) -> list[int]:
    """Labels from absolute frame indices — the sanity arm of plan §3.2.

    Strictly more correct on an untrimmed release, and therefore the check that
    the fraction mapping (which P1 validated to ±3 frames on one clip of 30) has
    not moved the window.
    """
    length = num_sampled_frames(int(entry["frames_on_disk"]), stride)
    labels = [0] * length
    start = max(0, min(length, int(entry["start"]) // stride))
    end = max(0, min(length, -(-int(entry["end"]) // stride)))
    for index in range(start, end):
        labels[index] = 1
    return labels


def build_meta(
    entries: list[dict[str, Any]],
    frame_labels: dict[str, list[int]],
    records: dict[str, DadaRecord],
) -> dict[str, dict[str, object]]:
    """Per-clip diagnostics — every field that defines the run (lesson **C17**)."""
    meta: dict[str, dict[str, object]] = {}
    for entry in entries:
        vid = entry["video_id"]
        labels = frame_labels[vid]
        record = records[vid]
        meta[vid] = {
            "class_name": record.class_name,
            "split": "test",
            "type": int(entry["type"]),
            "video": int(entry["video"]),
            "total_frames": int(entry["frames_on_disk"]),
            "annotation_total_frames": int(entry["total"]),
            "frame_delta": int(entry["frames_on_disk"]) - int(entry["total"]),
            "sampled_frames": len(labels),
            "positive_frames": sum(labels),
            "annotation_span_frames": [int(entry["start"]), int(entry["end"])],
            "normalized_span": list(record.span) if record.span else None,
        }
    return meta


def write_dataset_dir(
    out_dir: Path,
    frame_labels: dict[str, list[int]],
    meta: dict[str, dict[str, object]],
) -> None:
    """The four files ``load_dataset_files`` demands; ``labels_train`` is empty.

    Phase 1 trains nothing, so an empty ``labels_train.json`` is the honest
    statement — not a copy of the test ids, which would read as a split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, object] = {
        constants.FRAME_LABELS_TEST_FILENAME: frame_labels,
        constants.LABELS_TRAIN_FILENAME: {},
        constants.DEFS_FILENAME: class_name_list({constants.DADA_CLASS_NAME}),
        constants.META_FILENAME: meta,
    }
    for filename, payload in payloads.items():
        target = out_dir / filename
        part = target.with_suffix(target.suffix + ".part")
        part.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        part.replace(target)          # atomic, lesson C11c
    LOGGER.info("wrote %s (%d test clips)", out_dir, len(frame_labels))


def materialize_flat(entries: list[dict[str, Any]], flat_dir: Path) -> int:
    """``flat/{video_id} -> DADA2000/{type}/{video:03d}/images`` (lesson **C26**).

    The same trick as ``core.data.dada.materialize_flat_dir``, with one
    difference that is **not** cosmetic: the link points at the ``images``
    directory itself, and the extractor is then run **without**
    ``--frames-subdir``.

    ``pathlib`` refuses to recurse into a symlinked directory (cycle guard, every
    version incl. 3.10 and 3.13), so ``list_frame_folders(flat, "images")`` walks
    ``flat.rglob("images")``, never descends through the link, and raises
    *"No frame folders found"*. A link at the images level is matched by the
    final component of ``rglob("*")`` instead, and ``list_frame_images`` follows
    it happily. Measured, not assumed.
    """
    flat_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for entry in entries:
        target = flat_dir / entry["video_id"]
        if target.is_symlink() or target.exists():
            continue
        source = (Path(entry["folder"]) / IMAGES_SUBDIR).resolve()
        target.symlink_to(source, target_is_directory=True)
        written += 1
    LOGGER.info("flat dir: %d new symlinks, %d entries total under %s",
                written, len(entries), flat_dir)
    return written


def check(
    entries: list[dict[str, Any]],
    frac: dict[str, list[int]],
    absolute: dict[str, list[int]],
    flat_dir: Path | None,
    stride: int,
) -> None:
    """Plan §4.3's five assertions. Loud, before 20 GiB of extraction is spent."""
    ids = [e["video_id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("video_id collision — the (type, video) key is not unique")

    for entry in entries:
        vid = entry["video_id"]
        expected = num_sampled_frames(int(entry["frames_on_disk"]), stride)
        if len(frac[vid]) != expected or len(absolute[vid]) != expected:
            raise ValueError(
                f"{vid}: label length {len(frac[vid])}/{len(absolute[vid])} != "
                f"{expected} sampled frames — cache and labels would not align"
            )

    deltas = [int(e["frames_on_disk"]) - int(e["total"]) for e in entries]
    exact = sum(1 for d in deltas if d == 0)
    within2 = sum(1 for d in deltas if abs(d) <= 2)
    LOGGER.info("P1 at n=%d: %d exact (%.1f%%), %d within +-2 (%.1f%%), "
                "mean signed delta %+.2f",
                len(deltas), exact, 100 * exact / len(deltas),
                within2, 100 * within2 / len(deltas),
                sum(deltas) / len(deltas))
    worst = sorted(zip(ids, deltas, strict=True), key=lambda p: -abs(p[1]))[:5]
    LOGGER.info("largest frame deltas: %s", worst)

    if flat_dir is not None:
        broken = [e["video_id"] for e in entries
                  if not list_frame_images(flat_dir / e["video_id"])]
        if broken:
            raise ValueError(
                f"{len(broken)} symlinks do not resolve to a folder of images: "
                f"{broken[:5]}"
            )
        scanned = list_frame_folders(flat_dir)
        scanned_ids = {video_id_from_path(f) for f in scanned}
        if len(scanned_ids) != len(entries):
            raise ValueError(
                f"the extractor would see {len(scanned_ids)} unique ids for "
                f"{len(entries)} clips — lesson C26 is not solved"
            )
        LOGGER.info("extractor scan of the flat dir: %d folders, %d unique ids",
                    len(scanned), len(scanned_ids))

    vanished = sorted(v for v in frac if not any(frac[v]))
    if vanished:
        LOGGER.warning("%d clips lose their window at stride %d: %s",
                       len(vanished), stride, vanished[:5])

    disagree = [
        (v, sum(a != b for a, b in zip(frac[v], absolute[v], strict=True)))
        for v in frac
    ]
    worst_frames = max(n for _, n in disagree)
    total_diff = sum(n for _, n in disagree)
    LOGGER.info("frac vs abs labels: %d clips differ, worst %d frames, %d frames total",
                sum(1 for _, n in disagree if n), worst_frames, total_diff)
    if worst_frames > MAX_LABEL_DISAGREEMENT_FRAMES:
        LOGGER.warning(
            "worst-case disagreement %d frames exceeds the pre-registered %d "
            "(plan §3.2) — compare the two eda reports before trusting either",
            worst_frames, MAX_LABEL_DISAGREEMENT_FRAMES,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--mode", choices=("full", "shard", "labels"), default="full")
    parser.add_argument("--clips", type=Path, required=True,
                        help="pick_probe.py output: [{type, video, start, end, total}]")
    parser.add_argument("--root", type=Path, default=None,
                        help="extraction root holding DADA2000/{type}/{video:03d}/images "
                             "(full and shard modes)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--counts-out", type=Path, default=None,
                        help="shard mode: where to record {video_id: frames_on_disk}")
    parser.add_argument("--counts-in", type=Path, nargs="+", default=None,
                        help="labels mode: the shard censuses to merge")
    parser.add_argument("--allow-missing-frames", action="store_true",
                        help="drop picked clips with no images instead of raising")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.mode in ("full", "shard") and args.root is None:
        parser.error(f"--root is required in {args.mode} mode")
    if args.mode == "shard" and args.counts_out is None:
        parser.error("--counts-out is required in shard mode")
    if args.mode == "labels" and not args.counts_in:
        parser.error("--counts-in is required in labels mode")

    rows = json.loads(args.clips.read_text(encoding="utf-8"))
    LOGGER.info("mode %s: %d clips, stride %d", args.mode, len(rows), args.stride)

    if args.mode == "labels":
        counts: dict[str, int] = {}
        for path in args.counts_in:
            counts.update(json.loads(path.read_text(encoding="utf-8")))
        LOGGER.info("merged %d shard censuses -> %d clips",
                    len(args.counts_in), len(counts))
        entries = entries_from_counts(rows, counts)
        flat_dir: Path | None = None
    else:
        entries, _ = resolve_rows(rows, args.root, args.allow_missing_frames)
        farm = args.out_dir / FLAT_DIRNAME
        materialize_flat(entries, farm)
        flat_dir = farm

    records = {e["video_id"]: make_record(e) for e in entries}
    frac = {vid: sampled_frame_labels(rec, args.stride) for vid, rec in records.items()}
    absolute = {e["video_id"]: absolute_frame_labels(e, args.stride) for e in entries}
    check(entries, frac, absolute, flat_dir, args.stride)

    if args.mode == "shard":
        args.counts_out.parent.mkdir(parents=True, exist_ok=True)
        census = {e["video_id"]: int(e["frames_on_disk"]) for e in entries}
        args.counts_out.write_text(json.dumps(census, indent=2), encoding="utf-8")
        LOGGER.info("census -> %s (%d clips). Encode features from %s, then "
                    "DELETE the frames.", args.counts_out, len(census), flat_dir)
        return

    meta = build_meta(entries, frac, records)
    write_dataset_dir(args.out_dir / FRAC_DIRNAME, frac, meta)
    write_dataset_dir(args.out_dir / ABS_DIRNAME, absolute,
                      build_meta(entries, absolute, records))

    positives = sum(sum(v) for v in frac.values())
    sampled = sum(len(v) for v in frac.values())
    LOGGER.info("D0 population: %d clips, %d sampled frames, %d positive (%.1f%%)",
                len(frac), sampled, positives, 100 * positives / sampled)
    if args.mode == "full":
        LOGGER.info("next: extract features from %s (NO --frames-subdir, see "
                    "materialize_flat), then core.tools.eda report --sections "
                    "features --probe", flat_dir)
    else:
        LOGGER.info("next: core.tools.eda report --sections features --probe")


if __name__ == "__main__":
    main()
