"""DoTA-CAP frame archive: one zip of DoTA-native frames per clip, so the CAP tars are read once.

Every DoTA-CAP tool so far (``dota_cap extract``, ``clip_windows cap``, ``v5_stage0 extract``)
streamed the CAP-DATA tar parts again, because a split gzipped tar cannot be seeked (group
``11`` alone is ~96 GB extracted). ``pack`` streams them **once** and writes, per DoTA clip of
the frozen alignment, ``{dota_id}.zip`` holding the clip's DoTA frame sequence rebuilt by its
frozen frame map: entry ``{d:06d}.jpg`` = the **original bytes** of CAP frame ``frame_map[d]``
(stored, not re-encoded). That is exactly the folder :func:`core.tools.dota_cap.rebuild_folder`
symlinks, so every cache built from an unpacked archive equals one built from the stream.

CPU only (no model). Resumable per clip: a clip whose zip opens and holds ``dota_frames``
entries is done. Writes are atomic (VM temp file -> ``.partial`` on the archive dir -> rename).

Subcommands::

    python -m core.tools.dota_cap_frames pack --alignment .../dota_cap_alignment.json \\
        --parts .../11/11.part_* --work-dir /content/pack/11 --archive-dir .../DoTA_CAP_frames \\
        [--cap-ids-file cap_ids_11.txt] --report .../pack_11.json
    python -m core.tools.dota_cap_frames verify --alignment ... --archive-dir ... \\
        --dota-clip-dir cache/clip/DoTA_s1_ncc --report .../verify.json

``verify`` checks every aligned clip has a zip whose entry count equals both the frame map and
the clip's ``DoTA_s1_ncc`` rows (the native frame count every protocol uses).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.video_io import list_frame_images
from core.tools.dota_cap import OK, file_sha256
from core.tools.feature_cache import read_ids_file
from core.tools.kill_switch_probe import write_json_atomic
from core.tools.stream_frames_clip import stream_video_batches

LOGGER = logging.getLogger(__name__)

CAP_FRAMES_MISMATCH = "cap_frames_mismatch"
NO_ARCHIVE = "no_archive"
BAD_ARCHIVE = "bad_archive"
COUNT_MISMATCH = "count_mismatch"
FRAME_SUFFIX = ".jpg"


def archive_path(archive_dir: Path, dota_id: str) -> Path:
    return archive_dir / f"{dota_id}.zip"


def entry_name(index: int, suffix: str = FRAME_SUFFIX) -> str:
    return f"{index:06d}{suffix}"


def archive_frames(path: Path) -> int | None:
    """Entry count of a readable archive, ``None`` if absent or unreadable (C11)."""
    if not path.is_file():
        return None
    try:
        with zipfile.ZipFile(path) as zf:
            return len(zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return None


def write_archive(dest: Path, cap_paths: list[Path], frame_map: list[int]) -> int:
    """Zip DoTA's frame sequence (original CAP bytes, stored) atomically; returns bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        local = Path(tmp.name)
    try:
        with zipfile.ZipFile(local, "w", compression=zipfile.ZIP_STORED) as zf:
            for d, j in enumerate(frame_map):
                source = cap_paths[j]
                zf.write(source, entry_name(d, source.suffix.lower() or FRAME_SUFFIX))
        staged = dest.with_name(dest.name + constants.CACHE_PART_SUFFIX)
        shutil.copyfile(local, staged)
        with staged.open("rb+") as fh:
            os.fsync(fh.fileno())
        staged.replace(dest)
        return dest.stat().st_size
    finally:
        local.unlink(missing_ok=True)


def unpack(path: Path, dest: Path) -> Path:
    """Extract one clip archive to ``dest`` (a plain frame folder); returns ``dest``."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if Path(name).name != name:  # flat archives only: no path can escape dest
                raise ValueError(f"{path}: unexpected entry {name!r}")
        zf.extractall(dest)
    return dest


def aligned_clips(alignment: Path) -> dict[str, dict[str, Any]]:
    clips: dict[str, dict[str, Any]] = json.loads(alignment.read_text(encoding="utf-8"))["clips"]
    return {k: v for k, v in clips.items() if v["reason"] == OK}


# --------------------------------------------------------------------------- pack


@dataclass
class Packer:
    """Per-batch callback of :func:`run_pack`: zip every DoTA clip of each streamed CAP video."""

    by_cap: dict[str, list[dict[str, Any]]]
    archive_dir: Path
    subdir: str
    report: dict[str, dict[str, Any]]
    save_report: Callable[[], None]

    def __call__(self, ready_dir: Path) -> None:
        for folder in sorted(p for p in ready_dir.iterdir() if p.is_dir()):
            cap_paths = list_frame_images(folder, self.subdir)
            for clip in self.by_cap.get(folder.name, []):
                self.report[clip["dota_id"]] = self.one_clip(clip, cap_paths)
        self.save_report()

    def one_clip(self, clip: dict[str, Any], cap_paths: list[Path]) -> dict[str, Any]:
        if len(cap_paths) != clip["cap_frames"]:
            return {"reason": CAP_FRAMES_MISMATCH, "cap_frames_streamed": len(cap_paths)}
        size = write_archive(
            archive_path(self.archive_dir, clip["dota_id"]), cap_paths, clip["frame_map"]
        )
        LOGGER.info("%s <- CAP %s: %d frames, %.1f MB", clip["dota_id"], clip["cap_id"],
                    len(clip["frame_map"]), size / 1e6)
        return {"reason": OK, "frames": len(clip["frame_map"]), "bytes": size,
                "cap_id": clip["cap_id"]}


def run_pack(args: argparse.Namespace) -> dict[str, Any]:
    clips = aligned_clips(args.alignment)
    report: dict[str, dict[str, Any]] = (
        json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    )
    todo = [
        c for c in clips.values()
        if archive_frames(archive_path(args.archive_dir, c["dota_id"])) != c["dota_frames"]
    ]
    if args.cap_ids_file is not None:
        group = read_ids_file(args.cap_ids_file)
        todo = [c for c in todo if c["cap_id"] in group]
    if not todo:
        LOGGER.info("Nothing to pack for %s", args.parts[0].parent)
        return report
    by_cap: dict[str, list[dict[str, Any]]] = {}
    for clip in todo:
        by_cap.setdefault(clip["cap_id"], []).append(clip)
    packer = Packer(by_cap, args.archive_dir, args.frames_subdir, report,
                    lambda: write_json_atomic(args.report, report))
    LOGGER.info("Packing %d DoTA clips from %d CAP videos", len(todo), len(by_cap))
    stream_video_batches(
        sorted(args.parts), args.work_dir / "stream", args.frames_subdir,
        args.max_batch_gb * constants.BYTES_PER_GB, packer, keep_ids=set(by_cap),
    )
    shutil.rmtree(args.work_dir, ignore_errors=True)
    return report


# --------------------------------------------------------------------------- verify


def run_verify(args: argparse.Namespace) -> dict[str, Any]:
    clips = aligned_clips(args.alignment)
    per_clip: dict[str, dict[str, Any]] = {}
    for dota_id, clip in sorted(clips.items()):
        n = archive_frames(archive_path(args.archive_dir, dota_id))
        if n is None:
            exists = archive_path(args.archive_dir, dota_id).exists()
            reason = BAD_ARCHIVE if exists else NO_ARCHIVE
            per_clip[dota_id] = {"reason": reason}
            continue
        native = len(np.load(args.dota_clip_dir / f"{dota_id}.npy", mmap_mode="r"))
        ok = n == clip["dota_frames"] == len(clip["frame_map"]) == native
        per_clip[dota_id] = {"reason": OK if ok else COUNT_MISMATCH, "frames": n,
                             "dota_s1_rows": native}
    failed = {k: v["reason"] for k, v in per_clip.items() if v["reason"] != OK}
    total_bytes = sum(
        archive_path(args.archive_dir, k).stat().st_size for k, v in per_clip.items()
        if v["reason"] == OK
    )
    result = {
        "alignment_sha256": file_sha256(args.alignment),
        "clips": len(clips),
        "ok": len(clips) - len(failed),
        "frames": sum(v.get("frames", 0) for v in per_clip.values() if v["reason"] == OK),
        "gb": round(total_bytes / 1e9, 2),
        "failed": failed,
        "pass": not failed,
    }
    manifest = {k: v for k, v in result.items() if k != "failed"} | {
        "layout": "{dota_id}.zip, entries {d:06d}.jpg = original CAP bytes of frame_map[d], stored",
        "rows": "DoTA native frames (== DoTA_s1_ncc rows)",
    }
    if result["pass"]:
        write_json_atomic(args.archive_dir / constants.DOTA_CAP_FRAMES_MANIFEST, manifest)
    write_json_atomic(args.report, {**result, "per_clip": per_clip})
    LOGGER.info("verify: %s", {k: v for k, v in result.items() if k != "failed"})
    return result


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack", help="stream CAP tar parts once -> one zip per DoTA clip (CPU)")
    pack.add_argument("--parts", type=Path, nargs="+", required=True)
    pack.add_argument("--work-dir", type=Path, required=True, help="VM disk, not Drive")
    pack.add_argument("--cap-ids-file", type=Path, default=None,
                      help="CAP ids in this group: restricts the work, stops the stream early")
    pack.add_argument("--frames-subdir", default="images")
    pack.add_argument("--max-batch-gb", type=float, default=constants.MMAU_STREAM_BATCH_GB)
    verify = sub.add_parser("verify", help="every aligned clip has a complete archive")
    verify.add_argument("--dota-clip-dir", type=Path, required=True, help="DoTA_s1_ncc")
    for p in (pack, verify):
        p.add_argument("--alignment", type=Path, required=True, help="dota_cap_alignment.json")
        p.add_argument("--archive-dir", type=Path, required=True)
        p.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "pack":
        run_pack(args)
    elif not run_verify(args)["pass"]:
        raise SystemExit("archive incomplete -- see the verify report")


if __name__ == "__main__":
    main()
