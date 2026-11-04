"""DAD (Dashcam Accident Dataset) for zero-shot reads: census, labels, frame decode.

Rule: ``core/docs/final/ZS_DAD.md`` (written before any DAD number). Layout on Drive::

    <data>/DAD/videos/testing/{positive,negative}/<stem>.mp4
    <data>/DAD/annotation/<stem>.txt     frame, track id, class, x1, y1, x2, y2, accident flag

Video ids are ``pos_<stem>`` / ``neg_<stem>`` (the two folders may reuse a stem). A positive
video's abnormal frames are the frames where **any** annotated object carries flag 1 (frame
numbers are 1-based in the file); a negative video is all-normal and its annotation, if any,
must carry no flag. The annotation dir is flat, so a stem present in both folders is attributed to
the positive video (counted as ``shared_stems`` in the census).

Subcommands::

    census   decode every video once (exact frame count, C10), measure fps, parse annotations,
             write dad_census.json and exit non-zero on any gate failure (ZS_DAD.md §3)
    prepare  census -> a core.evaluate data dir: frame_labels_test.json at the stride (row
             labels = native[::stride]), defs.json (the arms' definitions), dad_manifest.json
    decode   mp4 -> one JPEG folder per video (VM disk): every extractor reads the same pixels

CLI::

    python -m core.tools.dad census --root data/DAD --out REPORTS/dad_census.json
    python -m core.tools.dad prepare --census REPORTS/dad_census.json \\
        --defs data/Nexar_window/defs.json --out-dir data/DAD_s5
    python -m core.tools.dad decode --root data/DAD --census REPORTS/dad_census.json \\
        --out-dir /content/dad
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image

from core import constants
from core.data.v2_splits import lines_sha1
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

FRAME_NAME = "{:05d}.jpg"
OK = "ok"
NO_ANNOTATION = "positive_without_annotation"
NO_FLAG = "positive_without_flagged_frame"
NEGATIVE_FLAGGED = "negative_with_flagged_frame"
FPS_MISMATCH = "fps_mismatch"
ANNOTATION_PAST_END = "annotation_frame_past_video_end"
BAD_ANNOTATION = "unparseable_annotation"


# ---------------------------------------------------------------------------
# pure pieces
# ---------------------------------------------------------------------------
def video_id(cls: str, stem: str) -> str:
    return f"{constants.DAD_ID_PREFIX[cls]}_{stem}"


def parse_annotation(text: str) -> tuple[set[int], int]:
    """``(flagged 1-based frames, last annotated frame)`` of one DAD annotation file."""
    flagged: set[int] = set()
    last = 0
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < constants.DAD_ANNOTATION_COLUMNS:
            raise ValueError(f"line {n}: {len(parts)} columns, need "
                             f"{constants.DAD_ANNOTATION_COLUMNS}: {line!r}")
        frame, flag = int(parts[0]), int(parts[-1])
        if frame < 1 or flag not in (0, 1):
            raise ValueError(f"line {n}: frame {frame} / flag {flag} out of range: {line!r}")
        last = max(last, frame)
        if flag:
            flagged.add(frame)
    return flagged, last


def native_labels(frames: int, flagged: set[int] | list[int]) -> np.ndarray:
    """``(frames,)`` 0/1: frame ``f`` (1-based in the annotation) -> index ``f - 1``."""
    labels = np.zeros(frames, dtype=np.int64)
    for f in flagged:
        if f > frames:
            raise ValueError(f"flagged frame {f} past the video's {frames} frames")
        labels[f - 1] = 1
    return labels


def row_labels(native: np.ndarray, stride: int) -> list[int]:
    """Labels of the CLIP rows ``frames[::stride]`` (the protocol's step = native frame)."""
    return [int(x) for x in native[::stride]]


def entry_reason(cls: str, fps: float, frames: int, annotated: bool, flagged: set[int],
                 last: int) -> str:
    if abs(fps - constants.DAD_ASSUMED_FPS) > constants.DAD_FPS_TOL:
        return FPS_MISMATCH
    if annotated and last > frames:
        return ANNOTATION_PAST_END
    if cls == "positive":
        if not annotated:
            return NO_ANNOTATION
        if not flagged:
            return NO_FLAG
    elif flagged:
        return NEGATIVE_FLAGGED
    return OK


def summarize(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Counts, frame / fps spread, the flag timing, and the census verdict (ZS_DAD.md §3)."""
    pos = [e for e in entries.values() if e["cls"] == "positive"]
    first = [min(e["flagged"]) for e in pos if e["flagged"]]
    bad = {v: e["reason"] for v, e in entries.items() if e["reason"] != OK}
    median_first = statistics.median(first) if first else None
    timing_ok = median_first is not None and median_first >= constants.DAD_FLAG_TIME_MIN_FRAME
    return {
        "videos": len(entries),
        "positive": len(pos),
        "negative": len(entries) - len(pos),
        "frames": sorted({e["frames"] for e in entries.values()}),
        "fps": sorted({round(e["fps"], 2) for e in entries.values()}),
        "first_flag_frame": {"min": min(first, default=None), "median": median_first,
                             "max": max(first, default=None)},
        "flagged_frames": sorted({len(e["flagged"]) for e in pos}),
        "flag_after_first_is_contiguous": all(
            sorted(e["flagged"]) == list(range(min(e["flagged"]), max(e["flagged"]) + 1))
            for e in pos if e["flagged"]),
        "shared_stems": sum(1 for e in entries.values() if e.get("shared_stem")),
        "failures": bad,
        "flag_timing_ok": timing_ok,
        "pass": not bad and timing_ok,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def list_dad_videos(root: Path) -> dict[str, tuple[str, Path]]:
    """``{id: (class, mp4)}`` under ``root/videos/testing/{positive,negative}``."""
    out: dict[str, tuple[str, Path]] = {}
    for cls in constants.DAD_CLASSES:
        folder = root / "videos" / constants.DAD_SPLIT_DIRNAME / cls
        for path in sorted(folder.glob("*.mp4")):
            out[video_id(cls, path.stem)] = (cls, path)
    if not out:
        split = root / "videos" / constants.DAD_SPLIT_DIRNAME
        raise FileNotFoundError(f"no DAD videos under {split}")
    return out


def probe_video(path: Path) -> tuple[int, float]:
    """``(decoded frame count, average fps)``; the count is from a full decode (C10)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        frames = sum(1 for _ in container.decode(stream))
    return frames, fps


def census(root: Path) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    ann_dir = root / constants.DAD_ANNOTATION_DIRNAME
    videos = list_dad_videos(root)
    positive_stems = {p.stem for cls, p in videos.values() if cls == "positive"}
    for vid, (cls, path) in videos.items():
        frames, fps = probe_video(path)
        ann = ann_dir / f"{path.stem}.txt"
        # one flat annotation dir: a stem in both folders belongs to the positive video
        shared = cls == "negative" and path.stem in positive_stems
        flagged: set[int] = set()
        last = 0
        reason = None
        if ann.is_file() and not shared:
            try:
                flagged, last = parse_annotation(ann.read_text(encoding="utf-8"))
            except ValueError as exc:
                reason = f"{BAD_ANNOTATION}: {exc}"
        entries[vid] = {
            "cls": cls, "video": str(path.relative_to(root)), "frames": frames, "fps": fps,
            "annotated": ann.is_file() and not shared, "shared_stem": shared,
            "flagged": sorted(flagged), "last_annotated": last,
        }
        entries[vid]["reason"] = reason or entry_reason(
            cls, fps, frames, entries[vid]["annotated"], flagged, last)
        LOGGER.info("%s: %d frames %.2f fps, %d flagged -> %s", vid, frames, fps, len(flagged),
                    entries[vid]["reason"])
    return {"root": str(root), "summary": summarize(entries), "videos": entries}


def prepare(census_doc: dict[str, Any], defs: Path, stride: int, out_dir: Path) -> dict[str, Any]:
    """Write the core.evaluate data dir; refuses a census that did not pass."""
    summary = census_doc["summary"]
    if not summary["pass"]:
        raise ValueError(f"the DAD census did not pass (ZS_DAD.md §3): {summary}")
    videos = census_doc["videos"]
    labels = {v: row_labels(native_labels(e["frames"], e["flagged"]), stride)
              for v, e in sorted(videos.items())}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / constants.FRAME_LABELS_TEST_FILENAME, labels)
    shutil.copy2(defs, out_dir / constants.DEFS_FILENAME)
    write_json_atomic(out_dir / constants.DAD_CENSUS_FILENAME, census_doc)
    two_class = sum(1 for g in labels.values() if 0 < sum(g) < len(g))
    manifest = {
        "dataset": constants.DAD_DATASET, "stride": stride, "fps": constants.DAD_ASSUMED_FPS,
        "videos": len(labels), "abnormal": sum(1 for g in labels.values() if max(g) > 0),
        "two_class_at_stride": two_class, "ids_sha1": lines_sha1(sorted(labels)),
        "label_rule": "rows = native[::stride]; native frame f-1 = 1 iff any object flag at f",
        "defs_from": str(defs),
    }
    write_json_atomic(out_dir / constants.DAD_MANIFEST_FILENAME, manifest)
    LOGGER.info("DAD data dir -> %s: %s", out_dir, manifest)
    return manifest


def decode_video(path: Path, out: Path, frames: int,
                 quality: int = constants.DAD_JPEG_QUALITY) -> bool:
    """mp4 -> ``out/00000.jpg ...``; ``False`` if already complete (resumable)."""
    if out.is_dir() and len(list(out.glob("*.jpg"))) == frames:
        return False
    tmp = out.with_name(out.name + constants.CACHE_PART_SUFFIX)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for n, frame in enumerate(container.decode(stream), 1):
            Image.fromarray(frame.to_ndarray(format="rgb24")).save(
                tmp / FRAME_NAME.format(n - 1), quality=quality)
    if n != frames:
        shutil.rmtree(tmp)
        raise ValueError(f"{path}: decoded {n} frames, census {frames} (C10)")
    shutil.rmtree(out, ignore_errors=True)
    tmp.replace(out)
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("census", help="decode-count, fps, annotations -> dad_census.json + gate")
    c.add_argument("--root", type=Path, required=True, help="data/DAD")
    c.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("prepare", help="census -> core.evaluate data dir")
    p.add_argument("--census", type=Path, required=True)
    p.add_argument("--defs", type=Path, required=True, help="the arms' defs.json (Nexar_window)")
    p.add_argument("--stride", type=int, default=constants.DAD_STRIDE)
    p.add_argument("--out-dir", type=Path, required=True)
    d = sub.add_parser("decode", help="mp4 -> JPEG folders (VM disk)")
    d.add_argument("--root", type=Path, required=True)
    d.add_argument("--census", type=Path, required=True)
    d.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "census":
        doc = census(args.root)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.out, doc)
        LOGGER.info("census summary: %s", json.dumps(doc["summary"]))
        if not doc["summary"]["pass"]:
            raise SystemExit(f"DAD census FAILED (see {args.out}): {doc['summary']}")
        return
    doc = json.loads(args.census.read_text(encoding="utf-8"))
    if args.command == "prepare":
        prepare(doc, args.defs, args.stride, args.out_dir)
        return
    if not doc["summary"]["pass"]:
        raise SystemExit("the DAD census did not pass; nothing is decoded")
    root = Path(args.root)
    done = 0
    for vid, e in sorted(doc["videos"].items()):
        done += decode_video(root / e["video"], args.out_dir / vid, int(e["frames"]))
    LOGGER.info("decoded %d new videos -> %s (%d total)", done, args.out_dir, len(doc["videos"]))


if __name__ == "__main__":
    main()
