"""Nexar N2: one decode per mp4 -> the CLIP ``s1`` cache and the VideoMAE V2 ``s1`` cache.

Plan ``katvad-v2-nexar-feasibility.md`` N2, addendum §21. Nexar ships pixels (unlike DoTA), and
the lesson from DoTA is to keep **every native frame**: any stride, crop or window is then a
slice of these two caches, never a re-extraction.

* ``cache/clip/Nexar_s1_ncc/{id}.npy`` -- ``(F, 512)`` CLIP image embeddings of every frame,
  ``no_center_crop`` (the whole project's transform, C2/C13).
* ``cache/video/<encoder>/Nexar_s1_squash/{id}.npy`` -- ``(F, d)`` VideoMAE V2 features, row
  ``i`` = the causal 16-frame clip every 3rd frame ending at frame ``i`` (1.5 s at 30 fps,
  clamped at frame 0) -- :mod:`core.tools.extract_video_features`' geometry at stride 1.

**One resize, two normalizations.** Both transforms are "float / 255 -> anisotropic bilinear
(antialias) resize of the full frame to 224^2 -> per-channel normalization"; they differ only
in the mean/std (CLIP vs ImageNet). Each frame is therefore decoded and resized once and the
two normalizations are applied to the same resized tensor. ``core/tests/test_nexar_extract.py``
pins both against :func:`core.tools.extract_clip_features.preprocess_frames` (``center_crop=
False``) and :func:`core.tools.extract_video_features.squash_frames` bit for bit.

Resumable by video (a video is done when both of its files read back, C11); writes are atomic.
``--report`` records decoded frames per video; the corpus builder compares them with the census.

CLI::

    python -m core.tools.nexar_extract --root data/Nexar/raw \\
        --census outputs/v2/REPORTS/nexar_n0/census.json \\
        [--ids-file ids.txt] [--encoder vit_s_k710_dl_from_giant] [--weights local.pth] \\
        [--clip-batch 256] [--video-batch 32] \\
        [--report outputs/v2/REPORTS/nexar_n2/extract_<k>.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from torch import Tensor, nn

from core import constants
from core.device import resolve_device
from core.models.videomae_v2 import load_pretrained
from core.tools.extract_clip_features import ImageEncoder, load_pretrained_encoder
from core.tools.extract_video_features import (
    build_manifest,
    causal_clip_indices,
    check_or_write_manifest,
)
from core.tools.feature_cache import Progress, is_complete, read_ids_file, save_array
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

CLIP_MANIFEST_FILENAME = "clip_manifest.json"
CLIP_TRANSFORM = "no_center_crop: bilinear antialias resize to 224^2, CLIP mean/std"


def resize_frames(frames: np.ndarray, size: int = constants.CROP_SIZE) -> Tensor:
    """``(N,H,W,3)`` uint8 RGB -> ``(N,3,size,size)`` float32 in ``[0,1]``: the shared step."""
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    resized: Tensor = nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", antialias=True
    )
    return resized


def normalize(resized: Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> Tensor:
    """Per-channel ``(x - mean) / std`` on a ``(..., 3, H, W)`` tensor."""
    shape = (3, 1, 1)
    out: Tensor = (resized - torch.tensor(mean).view(shape)) / torch.tensor(std).view(shape)
    return out


def iter_frame_chunks(
    path: Path, chunk: int = constants.NEXAR_EXTRACT_CHUNK
) -> Iterator[np.ndarray]:
    """Every decoded frame of ``path`` as ``(<=chunk, H, W, 3)`` uint8 RGB arrays (C9)."""
    buffer: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            buffer.append(frame.to_ndarray(format="rgb24"))
            if len(buffer) == chunk:
                yield np.stack(buffer)
                buffer = []
    if buffer:
        yield np.stack(buffer)


def decode_resized(path: Path) -> Tensor:
    """``(F, 3, 224, 224)`` float32 in ``[0,1]`` for every frame of ``path``."""
    parts = [resize_frames(chunk) for chunk in iter_frame_chunks(path)]
    if not parts:
        raise ValueError(f"No frames decoded from {path}")
    return torch.cat(parts)


@torch.no_grad()
def clip_rows(
    resized: Tensor, encoder: ImageEncoder, device: torch.device, batch_size: int
) -> np.ndarray:
    """``(F, 512)`` CLIP embeddings; identical to ``preprocess_frames(center_crop=False)``."""
    chunks: list[Tensor] = []
    for start in range(0, len(resized), batch_size):
        pixels = normalize(
            resized[start : start + batch_size], constants.CLIP_IMAGE_MEAN, constants.CLIP_IMAGE_STD
        )
        output = encoder(pixel_values=pixels.to(device))
        chunks.append(output.image_embeds.float().cpu())  # type: ignore[attr-defined]
    return torch.cat(chunks).numpy().astype(np.float32)


def motion_clips(
    num_frames: int, step: int = constants.VIDEOMAE_CLIP_FRAME_STEP
) -> list[list[int]]:
    """The causal clip of every frame (stride 1), as ``extract_video_features`` builds them."""
    return [causal_clip_indices(end, step=step) for end in range(num_frames)]


@torch.no_grad()
def motion_rows(
    resized: Tensor, encoder: nn.Module, device: torch.device, batch_size: int
) -> np.ndarray:
    """``(F, d)`` VideoMAE features; each clip is ``(3, 16, 224, 224)`` squash-normalized."""
    clips = motion_clips(len(resized))
    chunks: list[Tensor] = []
    for start in range(0, len(clips), batch_size):
        index = torch.tensor(clips[start : start + batch_size])  # (B, 16)
        frames = normalize(
            resized[index], constants.VIDEOMAE_IMAGE_MEAN, constants.VIDEOMAE_IMAGE_STD
        )  # (B, 16, 3, H, W)
        chunks.append(encoder(frames.permute(0, 2, 1, 3, 4).to(device)).float().cpu())
    return torch.cat(chunks).numpy().astype(np.float32)


def video_paths(root: Path, census: dict[str, Any], ids: set[str] | None) -> dict[str, Path]:
    """``{id: mp4}`` for the census videos (or ``ids``); a requested id that is absent raises."""
    rows: dict[str, dict[str, Any]] = census["videos"]
    wanted = sorted(rows) if ids is None else sorted(ids)
    unknown = [v for v in wanted if v not in rows]
    if unknown:
        raise KeyError(f"{len(unknown)} ids are not in the census: {unknown[:5]}")
    paths = {v: root / rows[v]["path"] for v in wanted}
    missing = [v for v, p in paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} mp4 files missing under {root}: {missing[:5]}")
    return paths


def clip_manifest() -> dict[str, Any]:
    """What makes the CLIP s1 cache comparable to ``DoTA_s1_ncc`` (C2)."""
    return {
        "model": constants.CLIP_MODEL_NAME,
        "revision": constants.CLIP_MODEL_REVISION,
        "transform": CLIP_TRANSFORM,
        "stride": 1,
        "source": "mp4 decoded with PyAV, every frame",
    }


def check_clip_manifest(output_dir: Path) -> None:
    """Write the CLIP manifest, or refuse to extend a cache built otherwise."""
    path = output_dir / CLIP_MANIFEST_FILENAME
    manifest = clip_manifest()
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"{path} was built with different settings: {existing} (C2)")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, manifest)


def extract(args: argparse.Namespace) -> dict[str, Any]:
    census = json.loads(args.census.read_text(encoding="utf-8"))
    ids = read_ids_file(args.ids_file) if args.ids_file else None
    paths = video_paths(args.root, census, ids)
    clip_dir: Path = args.clip_out
    video_dir: Path = args.video_out or (
        constants.VIDEO_CACHE_DIR / args.encoder / f"{constants.NEXAR_VIDEO_S1_DATASET}_squash"
    )
    check_clip_manifest(clip_dir)
    check_or_write_manifest(
        video_dir,
        build_manifest(args.encoder, 1, constants.VIDEOMAE_WEIGHTS_SHA256[args.encoder]),
        force=False,
    )
    todo = [
        v for v in paths
        if not (is_complete(clip_dir / f"{v}.npy") and is_complete(video_dir / f"{v}.npy"))
    ]
    LOGGER.info("Resume: %d / %d videos done, %d to extract", len(paths) - len(todo),
                len(paths), len(todo))
    device = resolve_device(args.device)
    clip_encoder = load_pretrained_encoder(device)
    video_encoder = load_pretrained(args.encoder, device, args.weights)
    progress = Progress(len(todo))
    report: dict[str, Any] = {"census_sha1_inputs": census.get("inputs"), "videos": {}}
    frames_total = 0
    started = time.monotonic()
    for video_id in todo:
        resized = decode_resized(paths[video_id])
        x = clip_rows(resized, clip_encoder, device, args.clip_batch)
        u = motion_rows(resized, video_encoder, device, args.video_batch)
        if len(x) != len(u):
            raise RuntimeError(f"{video_id}: CLIP rows {len(x)} != motion rows {len(u)}")
        save_array(clip_dir / f"{video_id}.npy", x)
        save_array(video_dir / f"{video_id}.npy", u)
        header = census["videos"][video_id].get("frames_header")
        report["videos"][video_id] = {"frames_decoded": len(x), "frames_header": header}
        frames_total += len(x)
        LOGGER.info("%s: %d frames -- %s", video_id, len(x), progress.step())
    elapsed = time.monotonic() - started
    report["summary"] = {
        "extracted": len(todo),
        "frames": frames_total,
        "frames_per_s": round(frames_total / elapsed, 1) if elapsed > 0 else None,
        "header_mismatch": sorted(
            v for v, r in report["videos"].items() if r["frames_decoded"] != r["frames_header"]
        ),
        "clip_dir": str(clip_dir),
        "video_dir": str(video_dir),
        "encoder": args.encoder,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.report, report)
    LOGGER.info("Extracted %d videos (%d frames, %s frames/s); header mismatches %d",
                len(todo), frames_total, report["summary"]["frames_per_s"],
                len(report["summary"]["header_mismatch"]))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--root", type=Path, required=True, help="Nexar repo mirror (holds train/)")
    parser.add_argument("--census", type=Path, required=True, help="N0 census.json")
    parser.add_argument("--ids-file", type=Path, default=None, help="restrict to these ids")
    parser.add_argument("--encoder", default=constants.VIDEOMAE_ENCODER_S,
                        choices=sorted(constants.VIDEOMAE_WEIGHTS_SHA256))
    parser.add_argument("--weights", type=Path, default=None, help="local VideoMAE weights")
    parser.add_argument("--clip-out", type=Path,
                        default=constants.CLIP_CACHE_DIR / constants.NEXAR_CLIP_S1_DATASET)
    parser.add_argument("--video-out", type=Path, default=None,
                        help="default cache/video/<encoder>/Nexar_s1_squash")
    parser.add_argument("--clip-batch", type=int, default=256)
    parser.add_argument("--video-batch", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", type=Path, default=None, help="per-video frame counts (json)")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    extract(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
