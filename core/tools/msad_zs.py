"""MSAD videos for ``KIM_RUN.md`` §18 (zero-shot test) and §41 (train): census, identity, decode.

The videos come from the official blurred release through the user's Drive shortcut
(``MSAD_anomaly/<class>/<id>.mp4``, ``MSAD_normal_testing/<id>.mp4`` for the test split,
``MSAD_normal_training/<scenario>/<id>.mp4`` for the train normals); labels, definitions and the
CLIP cache are the ones ``full_gate_a`` read (``data/MSAD``, ``cache/clip/MSAD_ncc``).

Subcommands::

    census     find each id's video exactly once under the split's official folders (never
               VALU's modified copy), decode-count it (C10), read its fps, and gate M2:
               ceil(decoded / stride) == rows(MSAD_ncc). rows != len(labels) is counted only
               (core.evaluate truncates both rows alike). ``--split train`` (§41) reads
               ``labels_train.json`` ({id: 0|1}), also accepts ``.avi``, requires each label to
               agree with the folder it was found in, and prints the train length ruler.
               Exit non-zero on failure
    pick       the 12 ids of gate M1 (first test id of each class + first normal) -> ids file
    identity   gate M1: re-extracted CLIP rows are MSAD_ncc's rows -- per-row cosine >=
               MSAD_ZS_IDENTITY_MIN_COS and the nearest video-mean is the video's own; max abs
               is printed against MSAD_ZS_IDENTITY_ATOL, not gated (§18 note 2026-11-11)
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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from av.error import FFmpegError

from core import constants
from core.data.definitions import DATASET_CLS_DEFS
from core.tools.dad import decode_video, probe_video
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

# the 11 MSAD anomaly classes (definitions.py), without "Normal"
MSAD_ANOMALY_CLASSES = frozenset(c for c in DATASET_CLS_DEFS["msad"] if c != "Normal")

OK = "ok"
MISSING = "mp4_missing"
DUPLICATE = "mp4_found_twice"
ROWS_MISMATCH = "decoded_rows_differ_from_clip_rows"
LABEL_FOLDER = "label_disagrees_with_folder"
VIDEO_SUFFIX = ".mp4"
SPLIT_TEST = "test"
SPLIT_TRAIN = "train"


# ---------------------------------------------------------------------------
# pure pieces
# ---------------------------------------------------------------------------
def video_class(video_id: str) -> str:
    """``Traffic_accident_20`` -> ``Traffic_accident``; every normal -> ``Normal``.

    Test normals are ``MSAD_normal_testing_<n>``; train normals are ``<scenario><n>``
    (``frontdoor1``, ``street_highview1``): a normal is every id whose prefix is not a class.
    """
    if video_id.startswith(constants.MSAD_ZS_NORMAL_PREFIX):
        return "Normal"
    head = video_id.rsplit("_", 1)[0]
    return head if head in MSAD_ANOMALY_CLASSES else "Normal"


def folder_agrees_with_label(label: int, relative_video: Path) -> bool:
    """Abnormal (1) iff the video is under ``MSAD_anomaly``; a swapped label inverts supervision."""
    return (relative_video.parts[0] == constants.MSAD_ANOMALY_DIR) == bool(label)


def length_auc(abnormal_rows: list[int], normal_rows: list[int]) -> float:
    """P(an abnormal video is shorter than a normal one), ties half: C28's length ruler."""
    a = np.asarray(abnormal_rows, dtype=np.float64)[:, None]
    n = np.asarray(normal_rows, dtype=np.float64)[None, :]
    return float(((a < n).sum() + 0.5 * (a == n).sum()) / (a.size * n.size))


def canonical_id(stem: str, aliases: dict[str, str] | None = None) -> str:
    """A file stem spelled as the release spells it -> the id ``labels_train.json`` uses.

    ``perdestrian_street7`` -> ``pedestrian_street7`` (the release's typo, §41 note 2026-11-23).
    """
    for typo, correct in (aliases or {}).items():
        if stem.startswith(typo):
            return correct + stem[len(typo):]
    return stem


def index_videos(paths: list[Path]) -> dict[str, list[Path]]:
    """``{stem: [paths]}``; a stem with two paths is a duplicate the census refuses."""
    found: dict[str, list[Path]] = {}
    for path in sorted(paths):
        found.setdefault(path.stem, []).append(path)
    return found


def stride_rows(frames: int, stride: int) -> int:
    """Rows of ``frames[::stride]``."""
    return math.ceil(frames / stride)


def entry_reason(paths: list[Path], rows: int | None, clip_rows: int | None,
                 label_ok: bool = True) -> str:
    if not paths:
        return MISSING
    if len(paths) > 1:
        return DUPLICATE
    if clip_rows is not None and rows != clip_rows:
        return ROWS_MISMATCH
    if not label_ok:
        return LABEL_FOLDER
    return OK


def train_stats(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Train split only (§41): the length ruler and rows by label. Printed, never gated.

    ``length_ruler_auc`` is P(abnormal shorter than normal) over the 480 videos at the census
    stride: above 0.5 a clip-length shortcut exists in the supervision (lesson C28).
    """
    rows: dict[int, list[int]] = {1: [], 0: []}
    for e in entries.values():
        if e.get("rows") is not None and e.get("label") is not None:
            rows[int(e["label"])].append(int(e["rows"]))
    if not rows[1] or not rows[0]:
        return {}
    return {
        "abnormal": len(rows[1]), "normal": len(rows[0]),
        "rows_median_abnormal": float(np.median(rows[1])),
        "rows_median_normal": float(np.median(rows[0])),
        "rows_min": min(rows[1] + rows[0]), "rows_max": max(rows[1] + rows[0]),
        "length_ruler_auc": length_auc(rows[1], rows[0]),
    }


def summarize(entries: dict[str, dict[str, Any]], expected: int | None = None,
              expected_abnormal: int | None = None) -> dict[str, Any]:
    """Counts, fps spread, label-length mismatches (printed) and the M2 verdict.

    ``expected`` defaults to the test split's size (read at call time); ``expected_abnormal``
    (train split) also gates the number of label-1 ids.
    """
    target = constants.MSAD_ZS_TEST_VIDEOS if expected is None else expected
    bad = {v: e["reason"] for v, e in entries.items() if e["reason"] != OK}
    fps = [round(float(e["fps"]), 2) for e in entries.values() if e.get("fps") is not None]
    spread: dict[str, int] = {}
    for f in fps:
        spread[str(f)] = spread.get(str(f), 0) + 1
    label_diff = {v: e["label_rows"] - e["clip_rows"] for v, e in entries.items()
                  if e.get("clip_rows") is not None and e.get("label_rows") is not None
                  and e["label_rows"] != e["clip_rows"]}
    abnormal = sum(1 for e in entries.values() if e.get("label") == 1)
    abnormal_ok = expected_abnormal is None or abnormal == expected_abnormal
    out: dict[str, Any] = {
        "videos": len(entries),
        "found": sum(1 for e in entries.values() if e["reason"] != MISSING),
        "fps_spread": dict(sorted(spread.items(), key=lambda kv: -kv[1])),
        "fps_off_nominal": sum(1 for f in fps if abs(f - constants.MSAD_ZS_ASSUMED_FPS) > 1.0),
        "label_rows_differ": len(label_diff),
        "label_rows_differ_examples": dict(sorted(label_diff.items())[:10]),
        "failures": bad,
        "pass": not bad and len(entries) == target and abnormal_ok,
    }
    if expected_abnormal is not None:
        out["abnormal"] = abnormal
        out["alias_matched"] = sorted(v for v, e in entries.items() if e.get("alias"))
        out["train"] = train_stats(entries)
    return out


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


def default_workers() -> int:
    """Decode threads: every core up to ``MSAD_DECODE_MAX_WORKERS`` (a worker holds one frame)."""
    return max(1, min(os.cpu_count() or 1, constants.MSAD_DECODE_MAX_WORKERS))


def decode_many(jobs: list[tuple[Path, Path, int]], workers: int) -> int:
    """``decode_video`` over ``(video, out_dir, frames)`` jobs on ``workers`` threads; returns
    how many were decoded (not already complete).

    Threads, not processes: PyAV's decoder and Pillow's JPEG encoder release the GIL (measured
    3.5x on 4 cores, 1080p), so there is no fork and each worker holds one frame. Every job
    writes its own folder, so the output is bit-identical to the sequential loop. The first
    failure cancels the jobs not yet started; all failures are named in the raised error.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    failed: dict[str, str] = {}
    decoded = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(decode_video, src, out, frames): out.name
                   for src, out, frames in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            if future.cancelled():  # cancelled below after a failure; result() would raise
                continue            # CancelledError and hide the failure being reported
            try:
                decoded += bool(future.result())
            except (ValueError, OSError, FFmpegError) as err:
                failed[futures[future]] = str(err)
                for pending in futures:
                    pending.cancel()
            if n % 20 == 0 or n == len(futures):
                LOGGER.info("decode: %d / %d videos finished", n, len(futures))
    if failed:
        raise RuntimeError(f"decode failed on {len(failed)} video(s): "
                           f"{dict(sorted(failed.items())[:5])}")
    return decoded


def _unit(rows: np.ndarray) -> np.ndarray:
    """Row-normalise; a zero row stays zero (its cosine then reads 0, never NaN)."""
    norm = np.linalg.norm(rows, axis=-1, keepdims=True)
    return np.asarray(rows / np.where(norm > 0, norm, 1.0))


def min_row_cosine(ref: np.ndarray, new: np.ndarray) -> float | None:
    """Lowest per-row cosine; ``None`` when the row counts differ."""
    if ref.shape != new.shape:
        return None
    ref64, new64 = ref.astype(np.float64), new.astype(np.float64)
    return float(np.min(np.sum(_unit(ref64) * _unit(new64), axis=1)))


def identity_report(pairs: dict[str, tuple[np.ndarray, np.ndarray]], atol: float,
                    min_cos: float) -> dict[str, Any]:
    """Gate M1 over ``{video: (MSAD_ncc rows, re-extracted rows)}``.

    PASS needs, for every video: the same row count, per-row cosine >= ``min_cos``, and the
    video-mean embedding of the re-extraction closest to its own ``MSAD_ncc`` mean among all the
    sampled videos. ``max_abs`` and ``bit_identical`` (every max abs <= ``atol``) are printed,
    never gated: the original M1 (max abs <= 1e-3) measured bit identity, not "same video"
    (``KIM_RUN.md`` §18, note 2026-11-11).
    """
    ids = list(pairs)
    per_video: dict[str, dict[str, float | bool | None]] = {}
    for v, (ref, new) in pairs.items():
        per_video[v] = {"max_abs": max_abs_diff(ref, new), "min_cos": min_row_cosine(ref, new)}
    matched: dict[str, bool] = {}
    if ids:
        ref_mean = _unit(np.stack([pairs[v][0].astype(np.float64).mean(axis=0) for v in ids]))
        new_mean = _unit(np.stack([pairs[v][1].astype(np.float64).mean(axis=0) for v in ids]))
        nearest = (ref_mean @ new_mean.T).argmax(axis=1)
        matched = {v: bool(nearest[i] == i) for i, v in enumerate(ids)}
    for v in ids:
        per_video[v]["nearest_is_own"] = matched[v]
    failures: dict[str, dict[str, float | bool | None]] = {}
    for v, stats in per_video.items():
        cos = stats["min_cos"]
        if not isinstance(cos, float) or not cos >= min_cos or not stats["nearest_is_own"]:
            failures[v] = stats
    abs_known = [s["max_abs"] for s in per_video.values() if isinstance(s["max_abs"], float)]
    cos_known = [s["min_cos"] for s in per_video.values() if isinstance(s["min_cos"], float)]
    return {
        "videos": len(ids),
        "atol": atol,
        "min_cos_floor": min_cos,
        "max_abs": max(abs_known) if abs_known else None,
        "min_cos": min(cos_known) if cos_known else None,
        "bit_identical": bool(abs_known) and len(abs_known) == len(ids) and max(abs_known) <= atol,
        "per_video": per_video,
        "failures": failures,
        "pass": bool(ids) and not failures,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _rows(directory: Path, video_id: str) -> int | None:
    path = directory / f"{video_id}.npy"
    return int(np.load(path, mmap_mode="r").shape[0]) if path.is_file() else None


def source_videos(root: Path,
                  dirs: tuple[str, ...] = constants.MSAD_ZS_SOURCE_DIRS,
                  suffixes: tuple[str, ...] = (VIDEO_SUFFIX,)) -> list[Path]:
    """Every video under the official folders only (never VALU's modified copies)."""
    missing = [d for d in dirs if not (root / d).is_dir()]
    if missing:
        raise FileNotFoundError(f"no {missing} under {root}")
    return [p for d in dirs for s in suffixes for p in (root / d).rglob(f"*{s}")]


def census(root: Path, labels: dict[str, Any], clip_dir: Path | None, stride: int,
           dirs: tuple[str, ...] = constants.MSAD_ZS_SOURCE_DIRS,
           suffixes: tuple[str, ...] = (VIDEO_SUFFIX,),
           expected: int | None = None,
           expected_abnormal: int | None = None,
           aliases: dict[str, str] | None = None) -> dict[str, Any]:
    """Census of ``labels``' ids. A list value is the test split's per-frame labels; an int
    (``labels_train.json``) is the video-level label, which must agree with the folder.

    ``aliases`` (train split) maps a typo'd file prefix to the id's spelling; a file spelled
    exactly as the id always wins, an alias match is flagged ``alias`` in the entry.
    """
    found = index_videos(source_videos(root, dirs, suffixes))
    by_alias: dict[str, list[Path]] = {}
    for stem, stem_paths in found.items():
        by_alias.setdefault(canonical_id(stem, aliases), []).extend(stem_paths)
    entries: dict[str, dict[str, Any]] = {}
    for vid in sorted(labels):
        paths = found.get(vid) or by_alias.get(vid, [])
        frames: int | None = None
        fps: float | None = None
        rows: int | None = None
        if len(paths) == 1:
            frames, fps = probe_video(paths[0])
            rows = stride_rows(frames, stride)
        clip_rows = _rows(clip_dir, vid) if clip_dir is not None else None
        video_level = not isinstance(labels[vid], list)
        label_ok = True
        if video_level and len(paths) == 1:
            label_ok = folder_agrees_with_label(int(labels[vid]), paths[0].relative_to(root))
        reason = entry_reason(paths, rows, clip_rows, label_ok)
        entries[vid] = {
            "cls": video_class(vid),
            "video": str(paths[0].relative_to(root)) if len(paths) == 1 else None,
            "frames": frames, "fps": fps, "rows": rows, "clip_rows": clip_rows,
            "label_rows": None if video_level else len(labels[vid]),
            "label": int(labels[vid]) if video_level else None, "reason": reason,
            "alias": len(paths) == 1 and paths[0].stem != vid,
        }
        LOGGER.info("%s: %s frames, %s fps, rows %s / clip %s -> %s", vid, frames, fps, rows,
                    clip_rows, reason)
    return {"root": str(root), "stride": stride, "source_dirs": list(dirs),
            "summary": summarize(entries, expected, expected_abnormal), "videos": entries}


def census_current(doc: dict[str, Any],
                   dirs: tuple[str, ...] = constants.MSAD_ZS_SOURCE_DIRS) -> bool:
    """A census that passed and read only the official folders; anything else is re-run."""
    return bool(doc["summary"]["pass"]) and doc.get("source_dirs") == list(dirs)


def identity(ref_dir: Path, new_dir: Path, ids: list[str], atol: float,
             min_cos: float = constants.MSAD_ZS_IDENTITY_MIN_COS) -> dict[str, Any]:
    pairs = {v: (np.load(ref_dir / f"{v}.npy"), np.load(new_dir / f"{v}.npy")) for v in ids}
    return identity_report(pairs, atol, min_cos)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("census", help="find, decode-count and gate the split's videos (M2)")
    c.add_argument("--videos-root", type=Path, required=True)
    c.add_argument("--split", choices=(SPLIT_TEST, SPLIT_TRAIN), default=SPLIT_TEST,
                   help="test: 240 videos from frame_labels_test.json (§18); train: 480 from "
                        "labels_train.json (§41)")
    c.add_argument("--labels", type=Path, required=True,
                   help="data/MSAD/frame_labels_test.json (test) or labels_train.json (train)")
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
    i.add_argument("--atol", type=float, default=constants.MSAD_ZS_IDENTITY_ATOL,
                   help="bit-identical threshold on max abs: printed, not gated")
    i.add_argument("--min-cos", type=float, default=constants.MSAD_ZS_IDENTITY_MIN_COS,
                   help="every row's cosine(MSAD_ncc, re-extracted) must reach this")
    i.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("decode", help="mp4 -> JPEG folders (VM disk)")
    d.add_argument("--videos-root", type=Path, required=True)
    d.add_argument("--census", type=Path, required=True)
    d.add_argument("--out-dir", type=Path, required=True)
    d.add_argument("--workers", type=int, default=None,
                   help="decode threads (default: every core up to MSAD_DECODE_MAX_WORKERS)")
    d.add_argument("--ids-file", type=Path, default=None,
                   help="decode only these ids (one per line): the runbook decodes, extracts and "
                        "deletes in chunks so the 480 train videos never sit on VM disk at once")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "census":
        if args.split == SPLIT_TRAIN:
            doc = census(args.videos_root, _read_json(args.labels), args.clip_dir, args.stride,
                         constants.MSAD_TRAIN_SOURCE_DIRS, constants.MSAD_VIDEO_SUFFIXES,
                         constants.MSAD_TRAIN_VIDEOS, constants.MSAD_TRAIN_ABNORMAL,
                         constants.MSAD_FILE_STEM_ALIASES)
        else:
            doc = census(args.videos_root, _read_json(args.labels), args.clip_dir, args.stride)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.out, doc)
        LOGGER.info("census summary: %s", json.dumps(doc["summary"]))
        if not doc["summary"]["pass"]:
            raise SystemExit(f"MSAD census FAILED (M2, see {args.out}): {doc['summary']}")
        return
    if args.command == "identity":
        ids = [v for v in args.ids_file.read_text(encoding="utf-8").split() if v]
        rep = identity(args.ref, args.new, ids, args.atol, args.min_cos)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.out, rep)
        LOGGER.info("identity: %s", json.dumps(
            {k: rep[k] for k in ("videos", "max_abs", "min_cos", "bit_identical", "pass")}))
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
    keep = {v for v in args.ids_file.read_text(encoding="utf-8").split() if v} \
        if args.ids_file is not None else None
    unknown = sorted(keep - set(doc["videos"])) if keep is not None else []
    if unknown:
        raise SystemExit(f"--ids-file names ids the census does not hold: {unknown[:5]}")
    wanted = {v: e for v, e in doc["videos"].items() if keep is None or v in keep}
    workers = args.workers if args.workers is not None else default_workers()
    jobs = [(args.videos_root / e["video"], args.out_dir / vid, int(e["frames"]))
            for vid, e in sorted(wanted.items())]
    done = decode_many(jobs, workers)
    LOGGER.info("decoded %d new videos -> %s (%d wanted, %d threads)", done, args.out_dir,
                len(wanted), workers)


if __name__ == "__main__":
    main()
