"""CLIP ViT-B/16 feature-extraction CLI (spec §7 common step).

Per video: sample every ``stride`` frames, resize + center-crop to 224,
normalize with CLIP statistics, encode with the frozen HF CLIP vision tower
(projection head included -> 512-d), save ``cache/clip/{DATASET}/{id}.npy``
as ``(L, 512)`` float32.

``--no-center-crop`` switches to LaGoVAD's transform (anisotropic resize to a
224 square, full field of view). The two are incompatible caches (lesson C2) --
give each its own ``--output-dir``, and feed a checkpoint only features built
with the transform it was trained under.

Resumable per video: rerun the same command after a disconnect and it extracts
only what is missing, reporting ``Resume: done/total`` and a per-clip ETA.
Writes are atomic and existing outputs are verified by reading their header, so
a run killed mid-write is redone rather than skipped (:mod:`core.tools.feature_cache`).
Stride and transform are *not* part of that check — changing either needs
``--force`` (lesson C2).

The encoder is injectable (any module returning ``.image_embeds``) so tests
run on a tiny random-weight ``CLIPVisionModelWithProjection`` with no download.

Datasets that ship extracted frames instead of videos (TAD, DoTA) use
``--frames-dir``; ``--frames-subdir`` names the per-video image folder
(DoTA: ``frames/{video_id}/images/*.jpg``). Frame folders are encoded in
streaming batches, so a long clip at ``--stride 1`` never materializes as one
array.

CLI::

    python -m core.tools.extract_clip_features --videos-dir data/MSAD/videos
        --dataset MSAD [--stride 8] [--batch-size 32] [--device auto] [--force]

    python -m core.tools.extract_clip_features --frames-dir data/DoTA/frames
        --frames-subdir images --dataset DoTA --stride 8
        [--ids-file data/DoTA/test_ids.txt]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from core import constants
from core.data.video_io import (
    list_frame_folders,
    list_frame_images,
    list_videos,
    read_images,
    read_sampled_frames,
    video_id_from_path,
)
from core.device import resolve_device
from core.tools.feature_cache import Progress, pending_items, save_array

LOGGER = logging.getLogger(__name__)


class ImageEncoder(Protocol):
    """Anything CLIP-vision-shaped: pixel_values (B,3,H,W) -> .image_embeds (B,D)."""

    def __call__(self, pixel_values: Tensor) -> object: ...


def preprocess_frames(
    frames: np.ndarray,
    crop_size: int = constants.CROP_SIZE,
    center_crop: bool = True,
) -> Tensor:
    """``(L,H,W,3)`` uint8 RGB -> ``(L,3,crop,crop)`` float32, CLIP-normalized.

    ``center_crop=True`` (default) resizes the shorter side to ``crop_size``
    then center crops — the CLIPImageProcessor convention, re-implemented
    deterministically so tests need no downloaded processor config.

    ``center_crop=False`` resizes anisotropically to ``crop_size`` square,
    keeping the full field of view. This is LaGoVAD's ``no_center_crop``
    augmentation (``LaGoVAD-PreVAD/src/utils/video_loader.py:79``) and the
    transform its released ``best.ckpt`` was trained under. It also matches
    :func:`core.flow.raft_extract.preprocess_for_raft`, which is full-frame —
    under the default, KIP is asked to regress motion evidence from the whole
    frame using features that only saw its middle.

    The two produce incompatible feature caches (lesson C2): a cache built with
    one is invalid for a checkpoint trained under the other.
    """
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    if center_crop:
        height, width = tensor.shape[-2:]
        scale = crop_size / min(height, width)
        new_h, new_w = round(height * scale), round(width * scale)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(new_h, new_w), mode="bilinear", antialias=True
        )
        top = (new_h - crop_size) // 2
        left = (new_w - crop_size) // 2
        tensor = tensor[:, :, top : top + crop_size, left : left + crop_size]
    else:
        tensor = torch.nn.functional.interpolate(
            tensor, size=(crop_size, crop_size), mode="bilinear", antialias=True
        )
    mean = torch.tensor(constants.CLIP_IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(constants.CLIP_IMAGE_STD).view(1, 3, 1, 1)
    normalized: Tensor = (tensor - mean) / std
    return normalized


@torch.no_grad()
def encode_video(
    video_path: Path,
    encoder: ImageEncoder,
    device: torch.device,
    stride: int = constants.FRAME_STRIDE,
    batch_size: int = 32,
    crop_size: int = constants.CROP_SIZE,
    center_crop: bool = True,
) -> np.ndarray:
    """Extract ``(L, D)`` float32 CLIP features for one video."""
    frames = read_sampled_frames(video_path, stride)
    pixels = preprocess_frames(frames, crop_size, center_crop)
    chunks: list[Tensor] = []
    for start in range(0, len(pixels), batch_size):
        batch = pixels[start : start + batch_size].to(device)
        output = encoder(pixel_values=batch)
        chunks.append(output.image_embeds.float().cpu())  # type: ignore[attr-defined]
    return torch.cat(chunks).numpy().astype(np.float32)


@torch.no_grad()
def encode_frame_dir(
    folder: Path,
    encoder: ImageEncoder,
    device: torch.device,
    stride: int = constants.FRAME_STRIDE,
    batch_size: int = 32,
    crop_size: int = constants.CROP_SIZE,
    subdir: str | None = None,
    center_crop: bool = True,
) -> np.ndarray:
    """``(L, D)`` CLIP features for a folder of extracted frames.

    Decodes and encodes ``batch_size`` images at a time: a 284-frame 720p clip
    at ``stride=1`` is ~3 GB as one float32 array, which is how this OOMs on a
    Colab T4 if written like :func:`encode_video`.
    """
    paths = list_frame_images(folder, subdir)[::stride]
    if not paths:
        raise ValueError(f"No frame images under {folder}")
    chunks: list[Tensor] = []
    for start in range(0, len(paths), batch_size):
        pixels = preprocess_frames(
            read_images(paths[start : start + batch_size]), crop_size, center_crop
        )
        output = encoder(pixel_values=pixels.to(device))
        chunks.append(output.image_embeds.float().cpu())  # type: ignore[attr-defined]
    return torch.cat(chunks).numpy().astype(np.float32)


def extract_frame_directory(
    frames_dir: Path,
    output_dir: Path,
    encoder: ImageEncoder,
    device: torch.device,
    stride: int = constants.FRAME_STRIDE,
    batch_size: int = 32,
    force: bool = False,
    subdir: str | None = None,
    video_ids: set[str] | None = None,
    center_crop: bool = True,
) -> list[Path]:
    """Encode every per-video frame folder under ``frames_dir``.

    ``video_ids`` restricts the run to a split (``--ids-file``); folders absent
    from it are skipped, and ids with no folder on disk raise -- a silently
    short feature cache would show up much later as a missing-`.npy` crash
    inside eval.

    Resumable: only the clips still missing from ``output_dir`` are encoded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = {video_id_from_path(p): p for p in list_frame_folders(frames_dir, subdir)}
    if video_ids is not None:
        missing = sorted(video_ids - set(folders))
        if missing:
            raise ValueError(
                f"{len(missing)} requested ids have no frame folder under "
                f"{frames_dir}: {missing[:5]}"
            )
        folders = {vid: path for vid, path in folders.items() if vid in video_ids}
    pending = pending_items(folders, output_dir, force)
    progress = Progress(len(pending))
    written: list[Path] = []
    for video_id, folder in pending:
        target = output_dir / f"{video_id}.npy"
        features = encode_frame_dir(
            folder, encoder, device, stride, batch_size,
            subdir=subdir, center_crop=center_crop,
        )
        save_array(target, features)
        written.append(target)
        LOGGER.info("Saved %s %s -- %s", target.name, features.shape, progress.step())
    LOGGER.info("Extracted %d new feature files into %s", len(written), output_dir)
    return written


def read_ids_file(path: Path) -> set[str]:
    """Video ids, one per line; blank lines ignored."""
    ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}
    ids.discard("")
    if not ids:
        raise ValueError(f"{path} contains no video ids")
    return ids


def extract_directory(
    videos_dir: Path,
    output_dir: Path,
    encoder: ImageEncoder,
    device: torch.device,
    stride: int = constants.FRAME_STRIDE,
    batch_size: int = 32,
    force: bool = False,
    center_crop: bool = True,
) -> list[Path]:
    """Encode every video under ``videos_dir``; skip completed outputs unless forced."""
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = {path.stem: path for path in list_videos(videos_dir)}
    pending = pending_items(videos, output_dir, force)
    progress = Progress(len(pending))
    written: list[Path] = []
    for video_id, video_path in pending:
        target = output_dir / f"{video_id}.npy"
        features = encode_video(
            video_path, encoder, device, stride, batch_size, center_crop=center_crop
        )
        save_array(target, features)
        written.append(target)
        LOGGER.info("Saved %s %s -- %s", target.name, features.shape, progress.step())
    LOGGER.info("Extracted %d new feature files into %s", len(written), output_dir)
    return written


def load_pretrained_encoder(device: torch.device) -> ImageEncoder:
    """The real frozen CLIP vision tower (network access on first call)."""
    from transformers import CLIPVisionModelWithProjection

    model = CLIPVisionModelWithProjection.from_pretrained(
        constants.CLIP_MODEL_NAME,
        revision=constants.CLIP_MODEL_REVISION,
        use_safetensors=True,
    )
    return model.to(device).eval()  # type: ignore[no-any-return]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--videos-dir", type=Path, help="directory of video files")
    source.add_argument("--frames-dir", type=Path,
                        help="directory of per-video extracted-frame folders")
    parser.add_argument("--frames-subdir", default=None,
                        help="image subfolder inside each frame folder (DoTA: images)")
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="restrict --frames-dir to these video ids, one per line")
    parser.add_argument("--dataset", default=constants.MSAD_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: cache/clip/{dataset}")
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true", help="re-extract existing files")
    parser.add_argument("--no-center-crop", action="store_true",
                        help="anisotropic resize to a square instead of resize+center "
                             "crop (LaGoVAD's transform; keeps the full field of view). "
                             "Produces a cache incompatible with the default -- write it "
                             "to its own --output-dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    output_dir = args.output_dir or constants.CLIP_CACHE_DIR / args.dataset
    if args.ids_file is not None and args.frames_dir is None:
        raise SystemExit("--ids-file only applies to --frames-dir")
    center_crop = not args.no_center_crop
    LOGGER.info(
        "Transform: %s -> %s",
        "resize-shorter-side + center crop" if center_crop else "anisotropic resize",
        output_dir,
    )
    encoder = load_pretrained_encoder(device)
    if args.frames_dir is not None:
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
            center_crop=center_crop,
        )
        return
    extract_directory(
        videos_dir=args.videos_dir,
        output_dir=output_dir,
        encoder=encoder,
        device=device,
        stride=args.stride,
        batch_size=args.batch_size,
        force=args.force,
        center_crop=center_crop,
    )


if __name__ == "__main__":
    main()
