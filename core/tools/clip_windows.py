"""CLIP ViT-B/16 embeddings of multi-scale frame windows (exploratory probe, pending (ba)).

AnyAnomaly's WinCLIP-based attention (WACV 2026, §3.3) splits a frame into windows at several
scales and embeds each window with CLIP. Here the windows are the non-overlapping cells of the
``g x g`` grids in :data:`constants.TW_GRIDS` laid over the **whole** frame, and each cell is
squashed to 224^2 and CLIP-normalized -- the ``no_center_crop`` transform of every v2 CLIP cache,
applied per cell (lesson C2/C13). The 1 x 1 grid is the whole frame, i.e. the existing CLIP row
``x``; it is extracted only so every clip can be **gated against its own CLIP cache**: a window
cache that disagrees there was built with another transform, frame map or row geometry. Per clip,
every row must reach ``DOTA_CAP_MIN_COS`` (a shifted frame map or row geometry breaks rows); the
mean must reach ``DOTA_CAP_MEAN_COS`` per clip on T2 (same pixels as the cache) but only over the
**corpus** on DoTA-CAP (:func:`corpus_gate`). There the reference is DoTA's own cache, not the CAP
pixels, so a clip admitted at 0.990x over all s1 frames can read 0.989x over its s3 rows -- a
transform error drops every clip, not the margin of a few (15 of 496 dev clips, 2026-10-08).

Rows: one per sampled frame, ``frames[::stride]`` -- stride 8 on T2 (the K sources, K's
geometry) and stride 3 on DoTA-CAP (protocol B, the rows of ``DoTA_s1_ncc[::3]``). A row stores
all windows flattened, ``(L, W * D)`` in ``TW_STORE_DTYPE``; :func:`unflatten` restores
``(L, W, D)``. The manifest records grids, transform, model revision, stride and frame source.

Two subcommands:

``frames`` -- per-video frame folders (T2: the per-shard DADA symlink farm of the notebooks).
``cap``    -- DoTA-CAP: streams the CAP tar parts, rebuilds each DoTA clip from its frozen frame
              map (:func:`core.tools.dota_cap.rebuild_folder`), encodes it at stride 3.

CLI::

    python -m core.tools.clip_windows frames --frames-dir /content/tw/flat --ids-file ids.txt \\
        --dataset DADA2000_orig --clip-dir cache/clip/DADA2000_orig [--stride 8]
    python -m core.tools.clip_windows cap --alignment .../dota_cap_alignment.json \\
        --parts .../11/11.part_* --work-dir /content/tw/work \\
        --dota-clip-dir cache/clip/DoTA_s1_ncc --dota-ids-file dota_cap_dev.txt \\
        --report .../tw_extract_11.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from core import constants
from core.data.video_io import (
    list_frame_folders,
    list_frame_images,
    read_images,
    video_id_from_path,
)
from core.device import resolve_device
from core.tools import extract_clip_features, extract_video_features
from core.tools.dota_cap import (
    FARM_DIR,
    OK,
    file_sha256,
    frame_cosines,
    rebuild_folder,
)
from core.tools.feature_cache import (
    Progress,
    is_complete,
    pending_items,
    read_ids_file,
    save_array,
    select_ids,
)
from core.tools.kill_switch_probe import write_json_atomic
from core.tools.stream_frames_clip import stream_video_batches

LOGGER = logging.getLogger(__name__)

TRANSFORM_NAME = "grid cells of the full frame, each squash224 + CLIP norm (no_center_crop)"
FEATURE_NAME = "CLIPVisionModelWithProjection.image_embeds per window"
FRAME_SOURCE_CAP = "CAP-DATA frames rebuilt by the frozen DoTA-CAP frame map"
FRAME_SOURCE_FOLDERS = "per-video frame folders"
GATE_FAILED = "global_window_cos"
ROWS_MISMATCH = "rows_mismatch"
CAP_FRAMES_MISMATCH = "cap_frames_mismatch"

Box = tuple[int, int, int, int]  # top, bottom, left, right


def windows_per_frame(grids: tuple[int, ...] = constants.TW_GRIDS) -> int:
    return sum(g * g for g in grids)


def window_boxes(
    height: int, width: int, grids: tuple[int, ...] = constants.TW_GRIDS
) -> list[Box]:
    """Row-major cells of every grid, grids in order; edges are rounded fractions of the frame."""
    boxes: list[Box] = []
    for g in grids:
        rows = np.rint(np.linspace(0, height, g + 1)).astype(int)
        cols = np.rint(np.linspace(0, width, g + 1)).astype(int)
        boxes += [
            (int(rows[i]), int(rows[i + 1]), int(cols[j]), int(cols[j + 1]))
            for i in range(g)
            for j in range(g)
        ]
    return boxes


def window_pixels(
    frames: Tensor, boxes: list[Box], size: int = constants.CROP_SIZE
) -> Tensor:
    """``(N, 3, H, W)`` in [0, 1] -> ``(N * len(boxes), 3, size, size)``, frame-major.

    Each cell goes through exactly what :func:`extract_clip_features.preprocess_frames` does with
    ``center_crop=False``: bilinear antialiased resize to ``size``^2, then CLIP mean/std.
    """
    cells = [
        nn.functional.interpolate(
            frames[:, :, top:bottom, left:right], size=(size, size), mode="bilinear",
            antialias=True,
        )
        for top, bottom, left, right in boxes
    ]
    stacked = torch.stack(cells, dim=1).flatten(0, 1)
    mean = torch.tensor(constants.CLIP_IMAGE_MEAN, device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor(constants.CLIP_IMAGE_STD, device=frames.device).view(1, 3, 1, 1)
    normalized: Tensor = (stacked - mean) / std
    return normalized


@torch.no_grad()
def encode_windows(
    paths: list[Path],
    encoder: extract_clip_features.ImageEncoder,
    device: torch.device,
    frames_per_batch: int = constants.TW_FRAMES_PER_BATCH,
    grids: tuple[int, ...] = constants.TW_GRIDS,
) -> np.ndarray:
    """``(L, W, D)`` float32 window embeddings for ``paths`` (already strided), streamed (C9)."""
    if not paths:
        raise ValueError("No frame images given")
    chunks: list[np.ndarray] = []
    for start in range(0, len(paths), frames_per_batch):
        frames = torch.from_numpy(read_images(paths[start : start + frames_per_batch]))
        frames = frames.to(device).permute(0, 3, 1, 2).float() / 255.0
        boxes = window_boxes(frames.shape[-2], frames.shape[-1], grids)
        output = encoder(pixel_values=window_pixels(frames, boxes))
        embeds = output.image_embeds.float().cpu()  # type: ignore[attr-defined]
        chunks.append(embeds.view(len(frames), len(boxes), -1).numpy())
    out: np.ndarray = np.concatenate(chunks).astype(np.float32)
    return out


def flatten(windows: np.ndarray) -> np.ndarray:
    """``(L, W, D)`` -> the stored ``(L, W * D)`` (2-D, so :func:`is_complete` reads it)."""
    return windows.reshape(len(windows), -1).astype(constants.TW_STORE_DTYPE)


def unflatten(rows: np.ndarray, grids: tuple[int, ...] = constants.TW_GRIDS) -> np.ndarray:
    """The stored ``(L, W * D)`` -> ``(L, W, D)`` float32."""
    out: np.ndarray = rows.astype(np.float32).reshape(len(rows), windows_per_frame(grids), -1)
    return out


def global_window(windows: np.ndarray, grids: tuple[int, ...] = constants.TW_GRIDS) -> np.ndarray:
    """The whole-frame cell -- equal to the CLIP row ``x`` under the same transform."""
    offset = 0
    for g in grids:
        if g == constants.TW_GLOBAL_GRID:
            return windows[:, offset]
        offset += g * g
    raise ValueError(f"grids {grids} have no whole-frame cell ({constants.TW_GLOBAL_GRID})")


def gate_rows(
    windows: np.ndarray, clip_rows: np.ndarray, min_mean_cos: float | None = None
) -> dict[str, Any]:
    """Row count and whole-frame cosine against the clip's own CLIP cache rows.

    Every row must reach ``DOTA_CAP_MIN_COS``; the clip mean is gated only when ``min_mean_cos``
    is given (same-pixel caches) -- otherwise it is left to :func:`corpus_gate`.
    """
    if len(windows) != len(clip_rows):
        return {"reason": ROWS_MISMATCH, "rows": len(windows), "clip_rows": len(clip_rows)}
    cos = frame_cosines(global_window(windows), clip_rows)
    entry = {"global_mean_cos": float(cos.mean()), "global_min_cos": float(cos.min())}
    passes = entry["global_min_cos"] >= constants.DOTA_CAP_MIN_COS and (
        min_mean_cos is None or entry["global_mean_cos"] >= min_mean_cos
    )
    return {**entry, "reason": OK if passes else GATE_FAILED}


def corpus_gate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Transform check over the gated clips: mean of clip means >= ``DOTA_CAP_MEAN_COS``."""
    if not entries:
        raise ValueError("No gated clips to check")
    means = np.array([e["global_mean_cos"] for e in entries])
    out = {
        "clips": len(entries),
        "mean_of_clip_means": float(means.mean()),
        "min_clip_mean": float(means.min()),
        "min_row_cos": float(min(e["global_min_cos"] for e in entries)),
    }
    return {**out, "reason": OK if out["mean_of_clip_means"] >= constants.DOTA_CAP_MEAN_COS
            else GATE_FAILED}


def build_manifest(dataset: str, stride: int, frame_source: str) -> dict[str, Any]:
    """Everything that makes two window caches comparable (C2)."""
    return {
        "dataset": dataset,
        "stride": stride,
        "grids": list(constants.TW_GRIDS),
        "windows_per_frame": windows_per_frame(),
        "model": constants.CLIP_MODEL_NAME,
        "revision": constants.CLIP_MODEL_REVISION,
        "transform": TRANSFORM_NAME,
        "feature": FEATURE_NAME,
        "dtype": constants.TW_STORE_DTYPE,
        "layout": "(L, W * D), windows grid-major then row-major",
        "frame_source": frame_source,
    }


def cache_dir(dataset: str, stride: int) -> Path:
    grids = "".join(str(g) for g in constants.TW_GRIDS)
    return constants.TW_CACHE_DIR / f"{dataset}_s{stride}_g{grids}_ncc"


# --------------------------------------------------------------------------- frames


def run_frames(args: argparse.Namespace, encoder: extract_clip_features.ImageEncoder) -> None:
    out = args.output_dir or cache_dir(args.dataset, args.stride)
    extract_video_features.check_or_write_manifest(
        out, build_manifest(args.dataset, args.stride, FRAME_SOURCE_FOLDERS), args.force
    )
    device = resolve_device(args.device)
    folders = select_ids(
        {video_id_from_path(p): p for p in list_frame_folders(args.frames_dir, args.frames_subdir)},
        read_ids_file(args.ids_file) if args.ids_file else None,
        args.frames_dir,
        "frame folder",
    )
    pending = pending_items(folders, out, args.force)
    progress = Progress(len(pending))
    for video_id, folder in pending:
        paths = list_frame_images(folder, args.frames_subdir)[:: args.stride]
        windows = encode_windows(paths, encoder, device, args.frames_per_batch)
        gate = gate_rows(
            windows, np.load(args.clip_dir / f"{video_id}.npy"), constants.DOTA_CAP_MEAN_COS
        )
        if gate["reason"] != OK:
            raise ValueError(f"{video_id}: window cache disagrees with its CLIP rows: {gate}")
        save_array(out / f"{video_id}.npy", flatten(windows))
        LOGGER.info("%s %s, global cos min %.4f -- %s", video_id, windows.shape,
                    gate["global_min_cos"], progress.step())


# --------------------------------------------------------------------------- cap


@dataclass
class CapExtractor:
    """Per-batch callback of :func:`run_cap`: rebuild, encode, gate and save each DoTA clip."""

    by_cap: dict[str, list[dict[str, Any]]]
    encoder: extract_clip_features.ImageEncoder
    out_dir: Path
    dota_clip_dir: Path
    farm: Path
    device: torch.device
    subdir: str
    stride: int
    frames_per_batch: int
    report: dict[str, dict[str, Any]]
    save_report: Callable[[], None]

    def __call__(self, ready_dir: Path) -> None:
        for folder in sorted(p for p in ready_dir.iterdir() if p.is_dir()):
            cap_paths = list_frame_images(folder, self.subdir)
            for clip in self.by_cap.get(folder.name, []):
                self.report[clip["dota_id"]] = self.one_clip(clip, cap_paths)
        self.save_report()

    def one_clip(self, clip: dict[str, Any], cap_paths: list[Path]) -> dict[str, Any]:
        dota_id = clip["dota_id"]
        if len(cap_paths) != clip["cap_frames"]:
            return {"reason": CAP_FRAMES_MISMATCH, "cap_frames_streamed": len(cap_paths)}
        farm = rebuild_folder(self.farm / dota_id, cap_paths, clip["frame_map"])
        try:
            paths = list_frame_images(farm)[:: self.stride]
            windows = encode_windows(paths, self.encoder, self.device, self.frames_per_batch)
            clip_rows = np.load(self.dota_clip_dir / f"{dota_id}.npy")[:: self.stride]
            gate = gate_rows(windows, clip_rows)
            if gate["reason"] == OK:
                save_array(self.out_dir / f"{dota_id}.npy", flatten(windows))
        finally:
            shutil.rmtree(farm, ignore_errors=True)
        LOGGER.info("%s <- CAP %s: %s", dota_id, clip["cap_id"], gate)
        return gate


def run_cap(args: argparse.Namespace, encoder: extract_clip_features.ImageEncoder) -> None:
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    out = args.output_dir or cache_dir(constants.DOTA_CAP_DATASET, args.stride)
    source = f"{FRAME_SOURCE_CAP} (alignment sha256 {file_sha256(args.alignment)})"
    extract_video_features.check_or_write_manifest(
        out, build_manifest(constants.DOTA_CAP_DATASET, args.stride, source), args.force
    )
    report: dict[str, dict[str, Any]] = (
        json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    )
    wanted = read_ids_file(args.dota_ids_file)
    todo = [
        clip for dota_id, clip in sorted(alignment["clips"].items())
        if dota_id in wanted and clip["reason"] == OK
        and not is_complete(out / f"{dota_id}.npy")
    ]
    if args.cap_ids_file is not None:
        in_group = read_ids_file(args.cap_ids_file)
        todo = [c for c in todo if c["cap_id"] in in_group]
    if not todo:
        LOGGER.info("Nothing to extract for %s", args.parts[0].parent)
        return
    by_cap: dict[str, list[dict[str, Any]]] = {}
    for clip in todo:
        by_cap.setdefault(clip["cap_id"], []).append(clip)
    extractor = CapExtractor(
        by_cap=by_cap, encoder=encoder, out_dir=out, dota_clip_dir=args.dota_clip_dir,
        farm=args.work_dir / FARM_DIR, device=resolve_device(args.device),
        subdir=args.frames_subdir, stride=args.stride, frames_per_batch=args.frames_per_batch,
        report=report, save_report=lambda: write_json_atomic(args.report, report),
    )
    LOGGER.info("Window-encoding %d DoTA clips from %d CAP videos", len(todo), len(by_cap))
    stream_video_batches(
        sorted(args.parts), args.work_dir / "stream", args.frames_subdir,
        args.max_batch_gb * constants.BYTES_PER_GB, extractor, keep_ids=set(by_cap),
    )
    shutil.rmtree(args.work_dir, ignore_errors=True)


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    frames = sub.add_parser("frames", help="per-video frame folders (T2 K sources)")
    frames.add_argument("--frames-dir", type=Path, required=True)
    frames.add_argument("--frames-subdir", default=None)
    frames.add_argument("--ids-file", type=Path, default=None)
    frames.add_argument("--dataset", required=True)
    frames.add_argument("--clip-dir", type=Path, required=True,
                        help="the dataset's CLIP _ncc cache at the same stride (row gate)")
    frames.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)

    cap = sub.add_parser("cap", help="DoTA-CAP clips rebuilt from the CAP tar parts")
    cap.add_argument("--alignment", type=Path, required=True, help="dota_cap_alignment.json")
    cap.add_argument("--parts", type=Path, nargs="+", required=True)
    cap.add_argument("--work-dir", type=Path, required=True, help="VM disk, not Drive")
    cap.add_argument("--dota-clip-dir", type=Path, required=True, help="DoTA_s1_ncc (row gate)")
    cap.add_argument("--dota-ids-file", type=Path, required=True, help="dota_cap_dev ids")
    cap.add_argument("--cap-ids-file", type=Path, default=None,
                     help="CAP ids in this group: restricts the work, stops the stream early")
    cap.add_argument("--report", type=Path, required=True, help="per-group JSON, resumable")
    cap.add_argument("--frames-subdir", default="images")
    cap.add_argument("--max-batch-gb", type=float, default=constants.MMAU_STREAM_BATCH_GB)
    cap.add_argument("--stride", type=int, default=constants.TW_DOTA_STRIDE)

    for p in (frames, cap):
        p.add_argument("--output-dir", type=Path, default=None,
                       help="default: cache/clip_windows/<dataset>_s<stride>_g<grids>_ncc")
        p.add_argument("--frames-per-batch", type=int, default=constants.TW_FRAMES_PER_BATCH)
        p.add_argument("--device", default="auto")
        p.add_argument("--force", action="store_true",
                       help="rewrite the manifest / re-extract existing files")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    encoder = extract_clip_features.load_pretrained_encoder(resolve_device(args.device))
    if args.command == "frames":
        run_frames(args, encoder)
    else:
        run_cap(args, encoder)


if __name__ == "__main__":
    main()
