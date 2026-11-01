"""LaGoVAD-STP patch caches: 7 x 7 window-crop CLIP features -> motion top-K (proposal §3.2.2).

Two subcommands, one output format (:mod:`core.data.stp_patches`):

``nexar`` (Colab, GPU)
    One PyAV decode per Nexar mp4; the frames of the training rows ``s1[::stride]`` are squashed
    to 224^2 exactly as ``Nexar_s1_ncc`` was (:func:`core.tools.nexar_extract.resize_frames`),
    cut into the 1 x 1 + 7 x 7 cells and encoded by frozen CLIP. **Gate**: the 1 x 1 cell must
    reproduce the video's own ``Nexar_s1_ncc[::stride]`` rows (same row count, every row cos
    >= ``MMAU_MATCH_NEAR``, mean >= ``MMAU_MATCH_EXACT``); a video that fails stores nothing.
    Only the selected top-K is stored (the full grid would be ~11 GB). Resumable per video.
``select`` (CPU)
    A ``core.tools.clip_windows --grids 1 7 --cells resized224`` window cache (DoTA-CAP-dev,
    protocol-B rows ``s1[::3]``) -> the same patch-cache format, keeping the window cache for maps
    and ablations.

Mo uses the neighbouring rows **of the cache** (the sampled steps), per source video, before any
item is cut or spliced. ``--amp`` runs the CLIP tower under fp16 autocast (features are stored
in fp16 anyway); the gate still reads the 1 x 1 cell against the fp32 CLS cache.

CLI::

    python -m core.tools.stp_patches nexar --root data/Nexar/raw --census census.json \\
        --data-dir data/Nexar_window --clip-s1-dir cache/clip/Nexar_s1_ncc \\
        --out-dir cache/clip_patches/Nexar_s8_k12_raw --report REPORTS/stp_extract_nexar.json
    python -m core.tools.stp_patches select \\
        --windows-dir cache/clip_windows/DoTA_CAP_s3_g17_resized224_ncc \\
        --ids-file dota_cap_dev.txt --out-dir cache/clip_patches/DoTA_CAP_s3_k12_raw
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core import constants
from core.data.stp_patches import build_manifest, cache_dir, select_video
from core.data.windows import FeatureSlicer, load_windows
from core.device import resolve_device
from core.tools.clip_windows import encode_frame_windows, global_window
from core.tools.dota_cap import frame_cosines
from core.tools.extract_clip_features import ImageEncoder, load_pretrained_encoder
from core.tools.extract_video_features import check_or_write_manifest
from core.tools.feature_cache import Progress, is_complete, read_ids_file, save_array
from core.tools.kill_switch_probe import write_json_atomic
from core.tools.nexar_extract import iter_frame_chunks, resize_frames, video_paths

LOGGER = logging.getLogger(__name__)

OK = "ok"
ROWS_MISMATCH = "rows_mismatch"
GATE_FAILED = "cls_cell_cos"
NEXAR_SOURCE = "Nexar mp4, PyAV decode, frames s1[::stride], squash224 (== Nexar_s1_ncc)"


# --------------------------------------------------------------------------- shared


def write_video(out_dir: Path, video_id: str, arrays: dict[str, np.ndarray]) -> None:
    """Index and magnitude first, the feature file last: it is the completion marker (C11)."""
    save_array(out_dir / constants.STP_INDEX_SUBDIR / f"{video_id}.npy", arrays["index"])
    save_array(out_dir / constants.STP_MAGNITUDE_SUBDIR / f"{video_id}.npy", arrays["magnitude"])
    save_array(out_dir / f"{video_id}.npy", arrays["features"])


def video_done(out_dir: Path, video_id: str) -> bool:
    return is_complete(out_dir / f"{video_id}.npy") and all(
        (out_dir / sub / f"{video_id}.npy").exists()
        for sub in (constants.STP_INDEX_SUBDIR, constants.STP_MAGNITUDE_SUBDIR)
    )


def prepare_out(out_dir: Path, manifest: dict[str, Any], force: bool) -> None:
    check_or_write_manifest(out_dir, manifest, force)
    for sub in (constants.STP_INDEX_SUBDIR, constants.STP_MAGNITUDE_SUBDIR):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def cls_gate(windows: np.ndarray, clip_rows: np.ndarray) -> dict[str, Any]:
    """The 1 x 1 cell against the video's own CLS rows (same pixels, same transform)."""
    if len(windows) != len(clip_rows):
        return {"reason": ROWS_MISMATCH, "rows": len(windows), "clip_rows": len(clip_rows)}
    cos = frame_cosines(global_window(windows, constants.STP_WINDOW_GRIDS), clip_rows)
    entry = {"cls_mean_cos": float(cos.mean()), "cls_min_cos": float(cos.min())}
    passes = (
        entry["cls_min_cos"] >= constants.MMAU_MATCH_NEAR
        and entry["cls_mean_cos"] >= constants.MMAU_MATCH_EXACT
    )
    return {**entry, "reason": OK if passes else GATE_FAILED}


def patch_cells(windows: np.ndarray) -> np.ndarray:
    """``(L, 1 + 49, D)`` windows -> the ``(L, 49, D)`` 7 x 7 cells (row-major)."""
    if windows.shape[1] != 1 + constants.STP_PATCHES:
        raise ValueError(f"expected 1 + {constants.STP_PATCHES} windows, got {windows.shape[1]}")
    out: np.ndarray = windows[:, 1:]
    return out


# --------------------------------------------------------------------------- nexar


def decode_strided(path: Path, stride: int) -> torch.Tensor:
    """``(ceil(F / stride), 3, 224, 224)`` in [0, 1]: frames ``0, stride, ...`` of ``path``.

    Same resize as :func:`core.tools.nexar_extract.decode_resized`, applied to the kept frames
    only (the resize is per frame, so the values are identical).
    """
    kept: list[torch.Tensor] = []
    offset = 0
    for chunk in iter_frame_chunks(path):
        index = np.arange(offset, offset + len(chunk))
        keep = (index % stride) == 0
        if keep.any():
            kept.append(resize_frames(chunk[keep]))
        offset += len(chunk)
    if not kept:
        raise ValueError(f"No frames decoded from {path}")
    return torch.cat(kept)


def decoded_ahead(
    paths: dict[str, Path], ids: list[str], stride: int
) -> Iterator[tuple[str, torch.Tensor]]:
    """``(id, decode_strided)`` in order, the next video decoding on a worker thread."""
    if not ids:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(decode_strided, paths[ids[0]], stride)
        for i, video_id in enumerate(ids):
            frames = pending.result()
            if i + 1 < len(ids):
                pending = pool.submit(decode_strided, paths[ids[i + 1]], stride)
            yield video_id, frames


@torch.no_grad()
def encode_video(
    frames: torch.Tensor,
    encoder: ImageEncoder,
    device: torch.device,
    frames_per_batch: int,
    amp: bool,
) -> np.ndarray:
    """``(L, 3, 224, 224)`` -> ``(L, 50, 512)`` float32 windows (1 x 1 + 7 x 7, resized cells)."""
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if amp and device.type == "cuda"
        else contextlib.nullcontext()
    )
    chunks: list[np.ndarray] = []
    with autocast:
        for start in range(0, len(frames), frames_per_batch):
            batch = frames[start : start + frames_per_batch].to(device)
            chunks.append(
                encode_frame_windows(
                    batch, encoder, constants.STP_WINDOW_GRIDS, constants.STP_CELLS_RESIZED
                )
            )
    out: np.ndarray = np.concatenate(chunks).astype(np.float32)
    return out


def training_sources(data_dir: Path) -> set[str]:
    """Source videos behind every training item of a (windowed) data dir."""
    labels = json.loads((data_dir / constants.LABELS_TRAIN_FILENAME).read_text(encoding="utf-8"))
    slicer = FeatureSlicer(load_windows(data_dir))
    return {slicer.source_of(item) for item in labels}


def run_nexar(args: argparse.Namespace) -> dict[str, Any]:
    census = json.loads(args.census.read_text(encoding="utf-8"))
    ids: set[str] = set()
    if args.data_dir is not None:
        ids |= training_sources(args.data_dir)
    if args.ids_file is not None:
        ids |= read_ids_file(args.ids_file)
    if not ids:
        raise ValueError("give --data-dir and/or --ids-file")
    paths = video_paths(args.root, census, ids)
    out: Path = args.out_dir or cache_dir(
        constants.NEXAR_DATASET, args.stride, args.motion, args.top_k
    )
    prepare_out(out, build_manifest(constants.NEXAR_DATASET, args.stride, args.motion,
                                    args.top_k, constants.STP_CELLS_RESIZED, NEXAR_SOURCE),
                args.force)
    report: dict[str, Any] = (
        json.loads(args.report.read_text(encoding="utf-8"))
        if args.report is not None and args.report.exists() else {}
    )
    videos: dict[str, Any] = report.setdefault("videos", {})
    todo = sorted(v for v in paths if args.force or not video_done(out, v))
    LOGGER.info("Resume: %d / %d videos done, %d to extract", len(paths) - len(todo), len(paths),
                len(todo))
    device = resolve_device(args.device)
    encoder = load_pretrained_encoder(device)
    progress = Progress(len(todo))
    for video_id, frames in decoded_ahead(paths, todo, args.stride):
        windows = encode_video(frames, encoder, device, args.frames_per_batch, args.amp)
        clip_rows = np.load(args.clip_s1_dir / f"{video_id}.npy", mmap_mode="r")[:: args.stride]
        gate = cls_gate(windows, np.asarray(clip_rows, dtype=np.float32))
        if gate["reason"] == OK:
            write_video(out, video_id, select_video(patch_cells(windows), args.motion, args.top_k))
        videos[video_id] = gate
        LOGGER.info("%s %d rows: %s -- %s", video_id, len(windows), gate, progress.step())
        if args.report is not None:
            write_json_atomic(args.report, report)
    return finish_report(report, paths, out, args.report)


def finish_report(
    report: dict[str, Any], ids: Any, out: Path, path: Path | None
) -> dict[str, Any]:
    videos = report.get("videos", {})
    failed = sorted(v for v in ids if not video_done(out, v))
    report["summary"] = {
        "videos": len(ids),
        "done": len(ids) - len(failed),
        "failed": {v: videos.get(v, {}).get("reason", "not_run") for v in failed},
        "min_cls_cos": min(
            (e["cls_min_cos"] for e in videos.values() if "cls_min_cos" in e), default=None
        ),
        "out_dir": str(out),
        "pass": not failed,
    }
    if path is not None:
        write_json_atomic(path, report)
    LOGGER.info("summary: %s", {k: v for k, v in report["summary"].items() if k != "failed"})
    return report


# --------------------------------------------------------------------------- select


def run_select(args: argparse.Namespace) -> dict[str, Any]:
    window_manifest = json.loads(
        (args.windows_dir / constants.VIDEO_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    if window_manifest.get("grids") != list(constants.STP_WINDOW_GRIDS):
        raise ValueError(f"{args.windows_dir} has grids {window_manifest.get('grids')}, "
                         f"STP needs {list(constants.STP_WINDOW_GRIDS)} (clip_windows --grids 1 7)")
    cells = window_manifest.get("cells", constants.STP_CELLS_NATIVE)
    if cells != args.cells:
        raise ValueError(
            f"{args.windows_dir} cut its cells from {cells!r}, run asks {args.cells!r}"
        )
    stride = int(window_manifest["stride"])
    dataset = str(window_manifest["dataset"])
    out: Path = args.out_dir or cache_dir(dataset, stride, args.motion, args.top_k)
    source = f"clip_windows cache {args.windows_dir.name}: {window_manifest.get('frame_source')}"
    prepare_out(out, build_manifest(dataset, stride, args.motion, args.top_k, cells, source),
                args.force)
    ids = (
        sorted(read_ids_file(args.ids_file)) if args.ids_file is not None
        else sorted(p.stem for p in args.windows_dir.glob("*.npy"))
    )
    report: dict[str, Any] = {"videos": {}}
    for video_id in ids:
        src = args.windows_dir / f"{video_id}.npy"
        if not src.exists():
            report["videos"][video_id] = {"reason": "no_window_cache"}
            continue
        if not args.force and video_done(out, video_id):
            continue
        rows = np.load(src).astype(np.float32)
        windows = rows.reshape(len(rows), 1 + constants.STP_PATCHES, -1)
        write_video(out, video_id, select_video(patch_cells(windows), args.motion, args.top_k))
    return finish_report(report, ids, out, args.report)


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    nexar = sub.add_parser("nexar", help="decode Nexar mp4s -> selected patch cache (GPU)")
    nexar.add_argument("--root", type=Path, required=True, help="data/Nexar/raw")
    nexar.add_argument("--census", type=Path, required=True, help="nexar_n0/census.json")
    nexar.add_argument("--data-dir", type=Path, default=None,
                       help="training data dir: every source behind labels_train is extracted")
    nexar.add_argument("--ids-file", type=Path, default=None, help="extra source ids")
    nexar.add_argument("--clip-s1-dir", type=Path, required=True,
                       help="Nexar_s1_ncc: the CLS gate for the 1 x 1 cell")
    nexar.add_argument("--stride", type=int, default=constants.STP_NEXAR_STRIDE)
    nexar.add_argument("--frames-per-batch", type=int, default=constants.STP_FRAMES_PER_BATCH)
    nexar.add_argument("--amp", action="store_true", help="fp16 autocast for the CLIP tower")
    nexar.add_argument("--device", default="auto")

    select = sub.add_parser("select", help="window cache (grids 1 7) -> selected patch cache")
    select.add_argument("--windows-dir", type=Path, required=True)
    select.add_argument("--ids-file", type=Path, default=None)
    select.add_argument("--cells", choices=constants.STP_CELL_SOURCES,
                        default=constants.STP_CELLS_RESIZED)

    for p in (nexar, select):
        p.add_argument("--motion", choices=constants.STP_MOTIONS, default=constants.STP_MOTION_RAW)
        p.add_argument("--top-k", type=int, default=constants.STP_TOP_K)
        p.add_argument("--out-dir", type=Path, default=None,
                       help="default: cache/clip_patches/<dataset>_s<stride>_k<K>_<motion>")
        p.add_argument("--report", type=Path, default=None, help="JSON report (resumable)")
        p.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    report = run_nexar(args) if args.command == "nexar" else run_select(args)
    if not report["summary"]["pass"]:
        raise SystemExit(f"{len(report['summary']['failed'])} videos have no patch cache")


if __name__ == "__main__":
    main()
