"""Zero-shot frame archives: only the stride-8 row frames of TAD test / DADA-2000 whole sources.

SA² needs pixels (``clip_windows`` cuts 1 x 1 + 7 x 7 cells per row), but TAD's frames live only
in the Kaggle copy and DADA-2000's in a 117 GiB spanned zip on Drive. Both are read **once**, on
a CPU runtime, and only the frames that become rows are kept: ``frames[::8]`` of each video, the
rows of ``TAD_ncc`` (``ceil(N / 8)``) and of the per-source ``DADA2000_orig`` CLIP cache. The GPU
notebook then unpacks ~2.5 GB instead of re-reading ~18 GiB of DADA PNGs through the mount
(the DoTA-CAP pattern, ``core.tools.dota_cap_frames``).

Layout: ``{archive_dir}/{video_id}.zip``, flat, entry ``{r:06d}{ext}`` = the **original bytes**
of row ``r``'s frame (stored, never re-encoded), so an unpacked archive lists in row order.

Subcommands::

    # DADA: from a ``7z l -slt`` listing of DADA2000.zip -> the member paths to extract
    python -m core.tools.zs_frames dada-plan --listing dada_listing.txt \\
        --census resolved_census.json --ids-file dada_sources.txt \\
        --out-list members.txt --out-plan plan.json
    # pack a per-video frame farm ({frames_dir}/{id}/[subdir/]*.png)
    python -m core.tools.zs_frames pack --frames-dir /content/farm --ids-file ids.txt \\
        --clip-dir cache/clip/TAD_ncc --archive-dir .../ZS_frames/TAD --report pack.json \\
        [--prestrided]
    python -m core.tools.zs_frames verify --dataset TAD --ids-file ids.txt \\
        --clip-dir cache/clip/TAD_ncc --archive-dir .../ZS_frames/TAD --report verify.json

``pack`` refuses a video whose row frames do not number its CLIP rows (C13) and skips one whose
archive is already complete (resumable; writes atomic, C11). ``verify`` writes the manifest only
when every id has a complete archive; ``clip_windows packed`` refuses an archive dir without it.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.dada_origin import load_counts
from core.data.v2_splits import lines_sha1
from core.data.video_io import IMAGE_EXTENSIONS, list_frame_images
from core.tools.dota_cap import OK
from core.tools.dota_cap_frames import archive_frames, archive_path, write_archive
from core.tools.feature_cache import read_ids_file
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

NO_FRAMES = "no_frames"
NO_CLIP_ROWS = "no_clip_rows"
ROWS_MISMATCH = "rows_mismatch"
CENSUS_MISMATCH = "census_mismatch"
NO_ARCHIVE = "no_archive"
COUNT_MISMATCH = "count_mismatch"
LISTING_PATH = re.compile(r"^Path = (.+)$")
DADA_ID = re.compile(r"^t(\d+)_v(\d+)$")


def clip_rows(clip_dir: Path, video_id: str) -> int | None:
    path = clip_dir / f"{video_id}.npy"
    return int(np.load(path, mmap_mode="r").shape[0]) if path.is_file() else None


# --------------------------------------------------------------------------- DADA plan


def dada_folder(video_id: str) -> str:
    """``t{type}_v{video}`` -> the source's image dir inside DADA2000.zip (PHASE0 §3.1)."""
    match = DADA_ID.match(video_id)
    if match is None:
        raise ValueError(f"not a DADA-2000 source id: {video_id!r}")
    kind, video = int(match.group(1)), int(match.group(2))
    return (f"{constants.DADA_ORIGIN_ROOT_DIRNAME}/{kind}/{video:03d}/"
            f"{constants.DADA_ORIGIN_IMAGES_SUBDIR}")


def listing_images(listing: str) -> dict[str, list[str]]:
    """``7z l -slt`` text -> {image dir: sorted image file names} (folders and others dropped)."""
    by_dir: dict[str, list[str]] = {}
    for line in listing.splitlines():
        match = LISTING_PATH.match(line.strip())
        if match is None:
            continue
        path = match.group(1).replace("\\", "/")
        folder, _, name = path.rpartition("/")
        if folder and Path(name).suffix.lower() in IMAGE_EXTENSIONS:
            by_dir.setdefault(folder, []).append(name)
    return {k: sorted(v) for k, v in by_dir.items()}


def dada_plan(
    images: dict[str, list[str]], census: dict[str, int], ids: list[str], stride: int
) -> dict[str, dict[str, Any]]:
    """Per source: the member paths of its row frames, ``sorted(images)[::stride]``.

    The order is :func:`list_frame_images`'s (lexicographic in one folder), the one the CLIP
    cache was built with; the archive's image count must equal the census (C10).
    """
    plan: dict[str, dict[str, Any]] = {}
    for video_id in ids:
        folder = dada_folder(video_id)
        names = images.get(folder, [])
        if not names:
            plan[video_id] = {"reason": NO_FRAMES, "folder": folder}
            continue
        if len(names) != census.get(video_id):
            plan[video_id] = {"reason": CENSUS_MISMATCH, "frames": len(names),
                              "census": census.get(video_id)}
            continue
        plan[video_id] = {"reason": OK, "folder": folder, "native_frames": len(names),
                          "members": [f"{folder}/{n}" for n in names[::stride]]}
    return plan


def run_dada_plan(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    images = listing_images(args.listing.read_text(encoding="utf-8", errors="replace"))
    census = load_counts([str(args.census)])
    plan = dada_plan(images, census, sorted(read_ids_file(args.ids_file)), args.stride)
    failed = {k: v["reason"] for k, v in plan.items() if v["reason"] != OK}
    write_json_atomic(args.out_plan, plan)
    if failed:
        first = dict(list(failed.items())[:5])
        raise ValueError(f"{len(failed)} sources cannot be planned: {first}")
    members = [m for v in plan.values() for m in v["members"]]
    args.out_list.write_text("".join(f"{m}\n" for m in members), encoding="utf-8")
    LOGGER.info("plan: %d sources, %d row frames to extract", len(plan), len(members))
    return plan


# --------------------------------------------------------------------------- pack


def row_frames(folder: Path, subdir: str | None, stride: int, prestrided: bool) -> list[Path]:
    paths = list_frame_images(folder, subdir)
    return paths if prestrided else paths[::stride]


def pack_video(
    video_id: str, frames: list[Path], rows: int | None, archive_dir: Path
) -> dict[str, Any]:
    """Zip one video's row frames, or say why not; a complete archive is left alone."""
    if not frames:
        return {"reason": NO_FRAMES}
    if rows is None:
        return {"reason": NO_CLIP_ROWS}
    if len(frames) != rows:
        return {"reason": ROWS_MISMATCH, "frames": len(frames), "clip_rows": rows}
    dest = archive_path(archive_dir, video_id)
    if archive_frames(dest) == rows:
        return {"reason": OK, "frames": rows, "bytes": dest.stat().st_size, "resumed": True}
    size = write_archive(dest, frames, list(range(len(frames))))
    return {"reason": OK, "frames": rows, "bytes": size}


def run_pack(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = (
        json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    )
    for video_id in sorted(read_ids_file(args.ids_file)):
        frames = row_frames(args.frames_dir / video_id, args.frames_subdir, args.stride,
                            args.prestrided)
        report[video_id] = pack_video(video_id, frames, clip_rows(args.clip_dir, video_id),
                                      args.archive_dir)
        LOGGER.info("%s: %s", video_id, report[video_id])
    write_json_atomic(args.report, report)
    return report


# --------------------------------------------------------------------------- verify


def run_verify(args: argparse.Namespace) -> dict[str, Any]:
    ids = sorted(read_ids_file(args.ids_file))
    per_video: dict[str, dict[str, Any]] = {}
    for video_id in ids:
        n = archive_frames(archive_path(args.archive_dir, video_id))
        rows = clip_rows(args.clip_dir, video_id)
        if n is None:
            per_video[video_id] = {"reason": NO_ARCHIVE}
        else:
            per_video[video_id] = {"reason": OK if n == rows else COUNT_MISMATCH,
                                   "frames": n, "clip_rows": rows}
    failed = {k: v["reason"] for k, v in per_video.items() if v["reason"] != OK}
    total = sum(archive_path(args.archive_dir, k).stat().st_size
                for k, v in per_video.items() if v["reason"] == OK)
    result = {
        "dataset": args.dataset,
        "stride": args.stride,
        "videos": len(ids),
        "ids_sha1": lines_sha1(ids),
        "ok": len(ids) - len(failed),
        "frames": sum(v.get("frames", 0) for v in per_video.values() if v["reason"] == OK),
        "gb": round(total / constants.BYTES_PER_GB, 3),
        "failed": failed,
        "pass": not failed,
    }
    if result["pass"]:
        manifest = {k: v for k, v in result.items() if k != "failed"} | {
            "layout": "{id}.zip, entries {r:06d}{ext} = original bytes of row r's frame, stored",
            "rows": f"frames[::{args.stride}] == the rows of {args.clip_dir.name}",
        }
        write_json_atomic(args.archive_dir / constants.ZS_FRAMES_MANIFEST, manifest)
    write_json_atomic(args.report, {**result, "per_video": per_video})
    LOGGER.info("verify: %s", {k: v for k, v in result.items() if k != "failed"})
    return result


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("dada-plan", help="7z listing -> the DADA row-frame members (CPU)")
    plan.add_argument("--listing", type=Path, required=True, help="output of `7z l -slt`")
    plan.add_argument("--census", type=Path, required=True, help="resolved_census.json")
    plan.add_argument("--out-list", type=Path, required=True, help="7z @listfile")
    plan.add_argument("--out-plan", type=Path, required=True, help="per-source JSON")
    pack = sub.add_parser("pack", help="per-video frame farm -> {id}.zip of the row frames")
    pack.add_argument("--frames-dir", type=Path, required=True, help="{frames_dir}/{id}/")
    pack.add_argument("--frames-subdir", default=None)
    pack.add_argument("--prestrided", action="store_true",
                      help="the folders hold only the row frames (DADA plan): no slicing")
    pack.add_argument("--report", type=Path, required=True)
    verify = sub.add_parser("verify", help="every id has a complete archive -> manifest")
    verify.add_argument("--dataset", required=True)
    verify.add_argument("--report", type=Path, required=True)
    for p in (pack, verify):
        p.add_argument("--clip-dir", type=Path, required=True,
                       help="the CLIP _ncc cache whose rows the frames must number")
        p.add_argument("--archive-dir", type=Path, required=True)
    for p in (plan, pack, verify):
        p.add_argument("--ids-file", type=Path, required=True)
        p.add_argument("--stride", type=int, default=constants.ZS_STRIDE)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "dada-plan":
        run_dada_plan(args)
    elif args.command == "pack":
        failed = {k: v for k, v in run_pack(args).items() if v["reason"] != OK}
        if failed:
            raise SystemExit(f"{len(failed)} videos not packed: {dict(list(failed.items())[:5])}")
    elif not run_verify(args)["pass"]:
        raise SystemExit("archive incomplete -- see the verify report")


if __name__ == "__main__":
    main()
