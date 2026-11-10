"""MSAD test videos for the zero-shot read of ``KIM_RUN.md`` §18: census, identity, frame decode.

The ``mp4`` come from the official blurred release through the user's Drive shortcut
(``MSAD_anomaly/<class>/<id>.mp4``, ``MSAD_normal_testing/<id>.mp4``); labels, definitions and
the CLIP cache are the ones ``full_gate_a`` read (``data/MSAD``, ``cache/clip/MSAD_ncc``).

Subcommands::

    census     find each test id's mp4 exactly once under MSAD_ZS_SOURCE_DIRS (never VALU's
               modified copy), decode-count it (C10), read its fps, and gate M2:
               ceil(decoded / stride) == rows(MSAD_ncc). rows != len(labels) is counted only
               (core.evaluate truncates both rows alike). Exit non-zero on failure
    pick       the 12 ids of gate M1 (first test id of each class + first normal) -> ids file
    identity   gate M1: re-extracted CLIP rows == MSAD_ncc rows, max abs <= MSAD_ZS_IDENTITY_ATOL
    decode     mp4 -> one JPEG folder per test video (VM disk), for V2-S

CLI::

    python -m core.tools.msad_zs census --videos-root "Thesis-Final/data/MSAD Videos blurred" \\
        --labels data/MSAD/frame_labels_test.json --clip-dir cache/clip/MSAD_ncc --out census.json
    python -m core.tools.msad_zs pick --census census.json --out m1_ids.txt
    python -m core.tools.msad_zs identity --ref cache/clip/MSAD_ncc --new /content/m1_clip \\
        --ids-file m1_ids.txt --out identity.json
    python -m core.tools.msad_zs decode --videos-root ... --census census.json --out-dir /content/f
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools.dad import decode_video, probe_video
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

OK = "ok"
MISSING = "mp4_missing"
DUPLICATE = "mp4_found_twice"
ROWS_MISMATCH = "decoded_rows_differ_from_clip_rows"
VIDEO_SUFFIX = ".mp4"


# ---------------------------------------------------------------------------
# pure pieces
# ---------------------------------------------------------------------------
def video_class(video_id: str) -> str:
    """``Traffic_accident_20`` -> ``Traffic_accident``; every normal -> ``Normal``."""
    if video_id.startswith(constants.MSAD_ZS_NORMAL_PREFIX):
        return "Normal"
    return video_id.rsplit("_", 1)[0]


def index_videos(paths: list[Path]) -> dict[str, list[Path]]:
    """``{stem: [paths]}``; a stem with two paths is a duplicate the census refuses."""
    found: dict[str, list[Path]] = {}
    for path in sorted(paths):
        found.setdefault(path.stem, []).append(path)
    return found


def stride_rows(frames: int, stride: int) -> int:
    """Rows of ``frames[::stride]``."""
    return math.ceil(frames / stride)


def entry_reason(paths: list[Path], rows: int | None, clip_rows: int | None) -> str:
    if not paths:
        return MISSING
    if len(paths) > 1:
        return DUPLICATE
    if clip_rows is not None and rows != clip_rows:
        return ROWS_MISMATCH
    return OK


def summarize(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Counts, fps spread, label-length mismatches (printed) and the M2 verdict."""
    bad = {v: e["reason"] for v, e in entries.items() if e["reason"] != OK}
    fps = [round(float(e["fps"]), 2) for e in entries.values() if e.get("fps") is not None]
    spread: dict[str, int] = {}
    for f in fps:
        spread[str(f)] = spread.get(str(f), 0) + 1
    label_diff = {v: e["label_rows"] - e["clip_rows"] for v, e in entries.items()
                  if e.get("clip_rows") is not None and e["label_rows"] != e["clip_rows"]}
    return {
        "videos": len(entries),
        "found": sum(1 for e in entries.values() if e["reason"] != MISSING),
        "fps_spread": dict(sorted(spread.items(), key=lambda kv: -kv[1])),
        "fps_off_nominal": sum(1 for f in fps if abs(f - constants.MSAD_ZS_ASSUMED_FPS) > 1.0),
        "label_rows_differ": len(label_diff),
        "label_rows_differ_examples": dict(sorted(label_diff.items())[:10]),
        "failures": bad,
        "pass": not bad and len(entries) == constants.MSAD_ZS_TEST_VIDEOS,
    }


def pick_identity_ids(ids: list[str]) -> list[str]:
    """M1's sample: the first test id (natural order) of every class, normals included."""
    def order(v: str) -> tuple[str, int]:
        head, _, tail = v.rpartition("_")
        return head, int(tail) if tail.isdigit() else -1

    first: dict[str, str] = {}
    for v in sorted(ids, key=order):
        first.setdefault(video_class(v), v)
    return [first[c] for c in sorted(first)]


def max_abs_diff(ref: np.ndarray, new: np.ndarray) -> float | None:
    """``None`` when the row counts differ (a different video, not a rounding)."""
    if ref.shape != new.shape:
        return None
    return float(np.max(np.abs(ref.astype(np.float64) - new.astype(np.float64))))


def identity_report(diffs: dict[str, float | None], atol: float) -> dict[str, Any]:
    bad = {v: d for v, d in diffs.items() if d is None or d > atol}
    known = [d for d in diffs.values() if d is not None]
    return {
        "videos": len(diffs),
        "atol": atol,
        "max_abs": max(known) if known else None,
        "per_video": diffs,
        "failures": bad,
        "pass": bool(diffs) and not bad,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _rows(directory: Path, video_id: str) -> int | None:
    path = directory / f"{video_id}.npy"
    return int(np.load(path, mmap_mode="r").shape[0]) if path.is_file() else None


def source_videos(root: Path,
                  dirs: tuple[str, ...] = constants.MSAD_ZS_SOURCE_DIRS) -> list[Path]:
    """Every mp4 under the official folders only (never VALU's modified copies)."""
    missing = [d for d in dirs if not (root / d).is_dir()]
    if missing:
        raise FileNotFoundError(f"no {missing} under {root}")
    return [p for d in dirs for p in (root / d).rglob(f"*{VIDEO_SUFFIX}")]


def census(root: Path, labels: dict[str, list[int]], clip_dir: Path | None,
           stride: int) -> dict[str, Any]:
    found = index_videos(source_videos(root))
    entries: dict[str, dict[str, Any]] = {}
    for vid in sorted(labels):
        paths = found.get(vid, [])
        frames: int | None = None
        fps: float | None = None
        rows: int | None = None
        if len(paths) == 1:
            frames, fps = probe_video(paths[0])
            rows = stride_rows(frames, stride)
        clip_rows = _rows(clip_dir, vid) if clip_dir is not None else None
        reason = entry_reason(paths, rows, clip_rows)
        entries[vid] = {
            "cls": video_class(vid),
            "video": str(paths[0].relative_to(root)) if len(paths) == 1 else None,
            "frames": frames, "fps": fps, "rows": rows, "clip_rows": clip_rows,
            "label_rows": len(labels[vid]), "reason": reason,
        }
        LOGGER.info("%s: %s frames, %s fps, rows %s / clip %s / labels %d -> %s", vid, frames,
                    fps, rows, clip_rows, len(labels[vid]), reason)
    return {"root": str(root), "stride": stride, "source_dirs": list(constants.MSAD_ZS_SOURCE_DIRS),
            "summary": summarize(entries), "videos": entries}


def census_current(doc: dict[str, Any]) -> bool:
    """A census that passed and read only the official folders; anything else is re-run."""
    return bool(doc["summary"]["pass"]) and doc.get("source_dirs") == list(
        constants.MSAD_ZS_SOURCE_DIRS)


def identity(ref_dir: Path, new_dir: Path, ids: list[str], atol: float) -> dict[str, Any]:
    diffs = {v: max_abs_diff(np.load(ref_dir / f"{v}.npy"), np.load(new_dir / f"{v}.npy"))
             for v in ids}
    return identity_report(diffs, atol)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("census", help="find, decode-count and gate the 240 test mp4 (M2)")
    c.add_argument("--videos-root", type=Path, required=True)
    c.add_argument("--labels", type=Path, required=True, help="data/MSAD/frame_labels_test.json")
    c.add_argument("--clip-dir", type=Path, default=None, help="cache/clip/MSAD_ncc")
    c.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    c.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("pick", help="M1's 12 ids -> ids file")
    p.add_argument("--census", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("identity", help="M1: re-extracted CLIP vs MSAD_ncc")
    i.add_argument("--ref", type=Path, required=True)
    i.add_argument("--new", type=Path, required=True)
    i.add_argument("--ids-file", type=Path, required=True)
    i.add_argument("--atol", type=float, default=constants.MSAD_ZS_IDENTITY_ATOL)
    i.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("decode", help="mp4 -> JPEG folders (VM disk)")
    d.add_argument("--videos-root", type=Path, required=True)
    d.add_argument("--census", type=Path, required=True)
    d.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "census":
        doc = census(args.videos_root, _read_json(args.labels), args.clip_dir, args.stride)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.out, doc)
        LOGGER.info("census summary: %s", json.dumps(doc["summary"]))
        if not doc["summary"]["pass"]:
            raise SystemExit(f"MSAD census FAILED (M2, see {args.out}): {doc['summary']}")
        return
    if args.command == "identity":
        ids = [v for v in args.ids_file.read_text(encoding="utf-8").split() if v]
        rep = identity(args.ref, args.new, ids, args.atol)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.out, rep)
        LOGGER.info("identity: %s", json.dumps({k: rep[k] for k in ("videos", "max_abs", "pass")}))
        if not rep["pass"]:
            raise SystemExit(f"M1 FAILED: these are not MSAD_ncc's videos: {rep['failures']}")
        return
    doc = _read_json(args.census)
    if not doc["summary"]["pass"]:
        raise SystemExit("the MSAD census did not pass; nothing is picked or decoded")
    if args.command == "pick":
        ids = pick_identity_ids(sorted(doc["videos"]))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(f"{v}\n" for v in ids), encoding="utf-8")
        LOGGER.info("M1 ids (%d): %s", len(ids), ids)
        return
    done = 0
    for vid, e in sorted(doc["videos"].items()):
        done += decode_video(args.videos_root / e["video"], args.out_dir / vid, int(e["frames"]))
    LOGGER.info("decoded %d new videos -> %s (%d total)", done, args.out_dir, len(doc["videos"]))


if __name__ == "__main__":
    main()
