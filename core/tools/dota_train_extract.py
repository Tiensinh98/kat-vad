"""Recovered DoTA train clips (``KIM_RUN.md`` §55): frame archives -> CLIP + VideoMAE rows at s3.

Reads ``{dota_id}.zip`` (DoTA's native 10 fps frames, written by ``dota_cap_frames pack``) in
memory and applies Nexar's two transforms unchanged (``core.tools.nexar_extract``). The VideoMAE
clip is 16 frames, step 1 = the same causal 1.5 s as everywhere else (``DOTA_VIDEOMAE_FRAME_STEP``,
the geometry of the DoTA-CAP cache the arms are read on).

* ``cache/clip/DoTA_train_s3_ncc/{id}.npy`` -- ``(ceil(F / 3), 512)``
* ``cache/video/<encoder>/DoTA_train_s3_squash/{id}.npy`` -- ``(ceil(F / 3), d)``, row-aligned

The clips are the frozen map minus ``DOTA_TRAIN_EXCLUDED``. Resumable by clip; atomic writes. CLI::

    python -m core.tools.dota_train_extract --archive-dir data/DoTA_CAP_train_frames \\
        --clip-out cache/clip/DoTA_train_s3_ncc \\
        --video-out cache/video/vit_s_k710_dl_from_giant/DoTA_train_s3_squash
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
import zipfile
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torchvision.io import ImageReadMode, decode_image

from core import constants
from core.data.dataset_files import num_sampled_frames
from core.device import resolve_device
from core.models.videomae_v2 import load_pretrained
from core.tools.dota_cap_frames import archive_path
from core.tools.dota_train_match import load_map
from core.tools.extract_clip_features import load_pretrained_encoder
from core.tools.extract_video_features import build_manifest, check_or_write_manifest
from core.tools.feature_cache import Progress, is_complete, read_ids_file, save_array
from core.tools.kill_switch_probe import write_json_atomic
from core.tools.nexar_extract import (
    CLIP_TRANSFORM,
    check_clip_manifest,
    clip_rows,
    motion_rows,
    resize_frames,
)

LOGGER = logging.getLogger(__name__)

FRAME_SOURCE = "CAP-DATA frames of the located DoTA train clips (KIM_RUN.md §52), 10 fps"


def train_clips(map_path: Path) -> dict[str, dict[str, Any]]:
    """The frozen map's clips minus the excluded ones (an unknown exclusion raises)."""
    clips: dict[str, dict[str, Any]] = load_map(map_path)["clips"]
    unknown = sorted(set(constants.DOTA_TRAIN_EXCLUDED) - set(clips))
    if unknown:
        raise ValueError(f"excluded ids are not in {map_path}: {unknown}")
    return {k: clips[k] for k in sorted(clips) if k not in constants.DOTA_TRAIN_EXCLUDED}


def decode_archive(path: Path, frames: int) -> Tensor:
    """``(F, 3, 224, 224)`` float32 in ``[0,1]``; a frame count other than ``frames`` raises."""
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as zf:  # one sequential Drive read
        names = sorted(zf.namelist())
        if len(names) != frames:
            raise ValueError(f"{path}: {len(names)} frames, the map says {frames}")
        parts: list[Tensor] = []
        for start in range(0, frames, constants.DOTA_TRAIN_DECODE_CHUNK):
            images = [
                decode_image(torch.frombuffer(bytearray(zf.read(n)), dtype=torch.uint8),
                             mode=ImageReadMode.RGB)
                for n in names[start : start + constants.DOTA_TRAIN_DECODE_CHUNK]
            ]
            parts.append(resize_frames(torch.stack(images).permute(0, 2, 3, 1).numpy()))
    return torch.cat(parts)


def decoded_clips(
    archive_dir: Path, todo: list[str], clips: dict[str, dict[str, Any]], workers: int
) -> Iterator[tuple[str, Tensor]]:
    """Clips in order, read and decoded on ``workers`` threads, at most ``workers`` clips ahead.

    The image ops release the GIL, so threads scale with the cores. The caller limits torch to
    one intra-op thread each (:func:`extract`): pixels do not depend on either number.
    """
    pending: deque[tuple[str, Future[Tensor]]] = deque()
    rest = iter(todo)
    with ThreadPoolExecutor(workers) as pool:
        while True:
            while len(pending) < workers and (k := next(rest, None)):
                pending.append((k, pool.submit(decode_archive, archive_path(archive_dir, k),
                                               int(clips[k]["dota_frames"]))))
            if not pending:
                return
            dota_id, future = pending.popleft()
            yield dota_id, future.result()


def clip_manifest() -> dict[str, Any]:
    return {
        "model": constants.CLIP_MODEL_NAME,
        "revision": constants.CLIP_MODEL_REVISION,
        "transform": CLIP_TRANSFORM,
        "stride": constants.DOTA_TRAIN_STRIDE,
        "source": FRAME_SOURCE,
    }


def extract(args: argparse.Namespace) -> dict[str, Any]:
    clips = train_clips(args.map)
    wanted = sorted(read_ids_file(args.ids_file)) if args.ids_file else list(clips)
    unknown = [v for v in wanted if v not in clips]
    if unknown:
        raise KeyError(f"{len(unknown)} ids are not train clips: {unknown[:5]}")
    check_clip_manifest(args.clip_out, clip_manifest())
    check_or_write_manifest(
        args.video_out,
        {**build_manifest(args.encoder, constants.DOTA_TRAIN_STRIDE,
                          constants.VIDEOMAE_WEIGHTS_SHA256[args.encoder],
                          clip_frame_step=constants.DOTA_VIDEOMAE_FRAME_STEP,
                          assumed_fps=constants.DOTA_FPS),
         "frame_source": FRAME_SOURCE},
        force=False,
    )
    todo = [v for v in wanted if not (is_complete(args.clip_out / f"{v}.npy")
                                      and is_complete(args.video_out / f"{v}.npy"))]
    missing = [v for v in todo if not archive_path(args.archive_dir, v).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} archives missing: {missing[:5]}")
    LOGGER.info("%d / %d clips done, %d to extract", len(wanted) - len(todo), len(wanted),
                len(todo))
    device = resolve_device(args.device)
    clip_encoder = load_pretrained_encoder(device)
    video_encoder = load_pretrained(args.encoder, device, args.weights)
    torch.set_num_threads(1)  # decode threads already fill the cores
    progress = Progress(len(todo))
    frames_total = 0
    started = time.monotonic()
    for dota_id, resized in decoded_clips(args.archive_dir, todo, clips, args.workers):
        x = clip_rows(resized[:: constants.DOTA_TRAIN_STRIDE], clip_encoder, device,
                      args.clip_batch)
        u = motion_rows(resized, video_encoder, device, args.video_batch,
                        stride=constants.DOTA_TRAIN_STRIDE,
                        clip_step=constants.DOTA_VIDEOMAE_FRAME_STEP)
        rows = num_sampled_frames(len(resized), constants.DOTA_TRAIN_STRIDE)
        if not len(x) == len(u) == rows:
            raise RuntimeError(f"{dota_id}: {len(x)} CLIP and {len(u)} motion rows, need {rows}")
        save_array(args.clip_out / f"{dota_id}.npy", np.ascontiguousarray(x))
        save_array(args.video_out / f"{dota_id}.npy", u)
        frames_total += len(resized)
        LOGGER.info("%s: %d frames -> %d rows -- %s", dota_id, len(resized), rows, progress.step())
    elapsed = time.monotonic() - started
    report = {
        "extracted": len(todo), "frames": frames_total, "encoder": args.encoder,
        "frames_per_s": round(frames_total / elapsed, 1) if elapsed > 0 else None,
        "workers": args.workers,
        "clip_dir": str(args.clip_out), "video_dir": str(args.video_out),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.report, report)
    LOGGER.info("Extracted %s", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--archive-dir", type=Path, required=True, help="{dota_id}.zip folder")
    parser.add_argument("--map", type=Path, default=(
        constants.DOTA_TRAIN_SPLITS_DIR / constants.DOTA_TRAIN_MAP_FILENAME))
    parser.add_argument("--ids-file", type=Path, default=None, help="restrict to these ids")
    parser.add_argument("--encoder", default=constants.VIDEOMAE_ENCODER_S,
                        choices=sorted(constants.VIDEOMAE_WEIGHTS_SHA256))
    parser.add_argument("--weights", type=Path, default=None, help="local .pth (sha256-checked)")
    parser.add_argument("--clip-out", type=Path, required=True)
    parser.add_argument("--video-out", type=Path, required=True)
    parser.add_argument("--clip-batch", type=int, default=128)
    parser.add_argument("--video-batch", type=int, default=16)
    parser.add_argument("--workers", type=int,
                        default=(os.cpu_count() or 1) + constants.DOTA_TRAIN_IO_THREADS,
                        help="clips read and decoded at once")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    extract(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
