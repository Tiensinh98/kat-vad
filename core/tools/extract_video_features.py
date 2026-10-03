"""VideoMAE V2 motion-stream feature extraction CLI (v2 proposal §4.1, plan P1).

One ``u_t`` per CLIP step, row-aligned with ``cache/clip/{DATASET}/{id}.npy``:
CLIP step ``i`` reads frame ``paths[::stride][i]`` = raw frame ``stride * i``;
``u_i`` encodes the **causal** clip of ``VIDEOMAE_CLIP_FRAMES`` frames taken every
``VIDEOMAE_CLIP_FRAME_STEP`` raw frames and **ending at that same frame**
(16 x 3 = 1.5 s at the assumed 30 fps, plan A1). No lookahead: ``u_i`` never sees
a frame after ``stride * i``. Before the first 45 frames the clip is padded by
repeating frame 0 (index clamping) -- the proposal's rule.

Transform: **squash** -- anisotropic resize of the full frame to 224^2, the CLIP
stream's ``no_center_crop`` field of view -- then ImageNet normalization, the
statistics VideoMAE V2 was fine-tuned under. It is a different normalization
from CLIP's, so this module has its own resize instead of widening
``extract_clip_features.preprocess_frames`` (the v1 cache's transform, C2/C13).

The transform, stride, clip geometry and pinned weights are recorded in
``video_manifest.json`` in the output directory; a rerun whose settings differ
from the manifest raises instead of mixing two caches (lesson C2). Writes are
atomic and resumable through :mod:`core.tools.feature_cache` (C11).

CLI::

    python -m core.tools.extract_video_features --frames-dir /content/farm/flat
        --dataset DADA2000_orig --ids-file k_sources.txt
        [--encoder vit_b_k710_dl_from_giant] [--weights local.pth]
        [--stride 8] [--batch-size 16] [--device auto] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
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
from core.models.videomae_v2 import load_pretrained
from core.tools.feature_cache import (
    Progress,
    pending_items,
    read_ids_file,
    save_array,
    select_ids,
)

LOGGER = logging.getLogger(__name__)

TRANSFORM_NAME = "squash224_imagenet"
FEATURE_NAME = "fc_norm(mean over tokens)"
PAD_RULE = "clamp to frame 0 (repeat first frame)"
DECODE_CHUNK = 64  # frames decoded per read_images call


def causal_clip_indices(
    end_frame: int,
    frames: int = constants.VIDEOMAE_CLIP_FRAMES,
    step: int = constants.VIDEOMAE_CLIP_FRAME_STEP,
) -> list[int]:
    """Raw frame indices of the clip ending at ``end_frame``, oldest first."""
    return [max(0, end_frame - step * (frames - 1 - k)) for k in range(frames)]


def step_end_frames(total_frames: int, stride: int) -> list[int]:
    """Raw frame of each CLIP step: ``range(0, total, stride)`` (``num_sampled_frames``)."""
    return list(range(0, total_frames, stride))


def squash_frames(frames: np.ndarray, size: int = constants.CROP_SIZE) -> Tensor:
    """``(N,H,W,3)`` uint8 RGB -> ``(N,3,size,size)`` float32, ImageNet-normalized."""
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    tensor = nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", antialias=True
    )
    mean = torch.tensor(constants.VIDEOMAE_IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(constants.VIDEOMAE_IMAGE_STD).view(1, 3, 1, 1)
    normalized: Tensor = (tensor - mean) / std
    return normalized


def decode_frames(paths: list[Path], indices: list[int]) -> dict[int, Tensor]:
    """Squashed frames for ``indices`` only, decoded in bounded chunks (C9)."""
    decoded: dict[int, Tensor] = {}
    for start in range(0, len(indices), DECODE_CHUNK):
        chunk = indices[start : start + DECODE_CHUNK]
        pixels = squash_frames(read_images([paths[i] for i in chunk]))
        decoded.update(zip(chunk, pixels, strict=True))
    return decoded


@torch.no_grad()
def encode_frame_dir(
    folder: Path,
    encoder: nn.Module,
    device: torch.device,
    stride: int = constants.FRAME_STRIDE,
    batch_size: int = 16,
    subdir: str | None = None,
    clip_step: int = constants.VIDEOMAE_CLIP_FRAME_STEP,
) -> np.ndarray:
    """``(L, D)`` float32 motion features, ``L = ceil(frames / stride)``.

    ``clip_step`` = raw frames between the clip's 16 frames: 3 at DADA's 30 fps, 1 at DoTA's
    native 10 fps (``DOTA_VIDEOMAE_FRAME_STEP``); both give the same causal 1.5 s.
    """
    paths = list_frame_images(folder, subdir)
    if not paths:
        raise ValueError(f"No frame images under {folder}")
    clips = [
        causal_clip_indices(end, step=clip_step) for end in step_end_frames(len(paths), stride)
    ]
    frames = decode_frames(paths, sorted({i for clip in clips for i in clip}))
    chunks: list[Tensor] = []
    for start in range(0, len(clips), batch_size):
        batch = torch.stack([
            torch.stack([frames[i] for i in clip], dim=1)  # (3, T, H, W)
            for clip in clips[start : start + batch_size]
        ])
        chunks.append(encoder(batch.to(device)).float().cpu())
    return torch.cat(chunks).numpy().astype(np.float32)


def build_manifest(encoder_name: str, stride: int, weights_sha256: str) -> dict[str, Any]:
    """Everything that makes two caches comparable (C2); compared on every rerun."""
    return {
        "encoder": encoder_name,
        "repo_id": constants.VIDEOMAE_REPO_ID,
        "revision": constants.VIDEOMAE_REVISION,
        "weights_sha256": weights_sha256,
        "stride": stride,
        "clip_frames": constants.VIDEOMAE_CLIP_FRAMES,
        "clip_frame_step": constants.VIDEOMAE_CLIP_FRAME_STEP,
        "assumed_fps": constants.DADA_ASSUMED_FPS,
        "transform": TRANSFORM_NAME,
        "image_mean": list(constants.VIDEOMAE_IMAGE_MEAN),
        "image_std": list(constants.VIDEOMAE_IMAGE_STD),
        "feature": FEATURE_NAME,
        "pad": PAD_RULE,
    }


def check_or_write_manifest(output_dir: Path, manifest: dict[str, Any], force: bool) -> None:
    """Refuse to extend a cache built with different settings unless ``force``."""
    path = output_dir / constants.VIDEO_MANIFEST_FILENAME
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        diff = sorted(k for k in manifest if existing.get(k) != manifest[k])
        if diff:
            raise ValueError(
                f"{path} was built with different {diff}; use a new --output-dir "
                "or --force to rebuild (lesson C2)"
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    save_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    staged = path.with_name(path.name + constants.CACHE_PART_SUFFIX)
    staged.write_text(save_text, encoding="utf-8")
    staged.replace(path)


def extract_frame_directory(
    frames_dir: Path,
    output_dir: Path,
    encoder: nn.Module,
    device: torch.device,
    stride: int = constants.FRAME_STRIDE,
    batch_size: int = 16,
    force: bool = False,
    subdir: str | None = None,
    video_ids: set[str] | None = None,
) -> list[Path]:
    """Encode every (selected) per-video frame folder; resumable, atomic writes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = select_ids(
        {video_id_from_path(p): p for p in list_frame_folders(frames_dir, subdir)},
        video_ids,
        frames_dir,
        "frame folder",
    )
    pending = pending_items(folders, output_dir, force)
    progress = Progress(len(pending))
    written: list[Path] = []
    steps = 0
    started = time.monotonic()
    for video_id, folder in pending:
        target = output_dir / f"{video_id}.npy"
        features = encode_frame_dir(folder, encoder, device, stride, batch_size, subdir)
        save_array(target, features)
        written.append(target)
        steps += features.shape[0]
        LOGGER.info("Saved %s %s -- %s", target.name, features.shape, progress.step())
    elapsed = time.monotonic() - started
    LOGGER.info(
        "Extracted %d files into %s; throughput %.1f steps/s (decode included)",
        len(written), output_dir, steps / elapsed if elapsed > 0 else 0.0,
    )
    return written


def default_output_dir(encoder_name: str, dataset: str, stride: int) -> Path:
    """``cache/video/<encoder>/<dataset>_s<stride>_squash`` (plan P1)."""
    return constants.VIDEO_CACHE_DIR / encoder_name / f"{dataset}_s{stride}_squash"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="directory of per-video frame folders (DADA: the flat symlink farm)")
    parser.add_argument("--frames-subdir", default=None,
                        help="image subfolder inside each frame folder")
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="restrict to these video ids, one per line")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder", default=constants.VIDEOMAE_ENCODER_B,
                        choices=sorted(constants.VIDEOMAE_ARCH))
    parser.add_argument("--weights", type=Path, default=None,
                        help="local .pth (sha256-checked); default: download at the pinned commit")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: cache/video/<encoder>/<dataset>_s<stride>_squash")
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true",
                        help="re-extract existing files and rewrite the manifest")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    output_dir = args.output_dir or default_output_dir(args.encoder, args.dataset, args.stride)
    encoder = load_pretrained(args.encoder, device, args.weights)
    manifest = build_manifest(
        args.encoder, args.stride, constants.VIDEOMAE_WEIGHTS_SHA256[args.encoder]
    )
    check_or_write_manifest(output_dir, manifest, args.force)
    LOGGER.info("Transform: %s, stride %d -> %s", TRANSFORM_NAME, args.stride, output_dir)
    extract_frame_directory(
        frames_dir=args.frames_dir,
        output_dir=output_dir,
        encoder=encoder,
        device=device,
        stride=args.stride,
        batch_size=args.batch_size,
        force=args.force,
        subdir=args.frames_subdir,
        video_ids=read_ids_file(args.ids_file) if args.ids_file else None,
    )


if __name__ == "__main__":
    main()
