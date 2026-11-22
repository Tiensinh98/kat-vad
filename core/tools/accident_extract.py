"""ACCIDENT (``KIM_RUN.md`` §39 A2): mp4 -> 30 fps CFR in memory -> CLIP s1 + VideoMAE s8 caches.

ACCIDENT's native frame rate varies from 3.9 to 49.7 fps (median 15), and VideoMAE's causal clip
(16 frames, step 3) is 1.5 s only at 30 fps. Each clip is therefore resampled to **30 fps CFR by
timestamp**: CFR frame ``k`` is the last decoded frame stamped at ``t <= k / 30`` (``t`` measured
from the first decoded frame). No re-encode: the decoded frames are resized once (Nexar's shared
step) and indexed. Then Nexar's two transforms run unchanged
(:func:`core.tools.nexar_extract.clip_rows` / :func:`~core.tools.nexar_extract.motion_rows`).

Only what training reads is computed (§39 A2, "extract to the cut"):

* frames past ``t_impact + post`` are not kept (the corpus cuts them; ``F`` = the cut length,
  identical to a full decode followed by :func:`core.tools.accident_build.cut_frames`);
* V2-S runs at the s8 rows only (training, the bake and every read use ``s1[::8]``);
* a clip whose cut is shorter than one window is skipped (it gives no window, train or val).

* ``cache/clip/ACCIDENT_s1_ncc/{id}.npy`` -- ``(F, 512)`` CLIP rows, one per CFR frame.
* ``cache/video/<encoder>/ACCIDENT_s8_squash/{id}.npy`` -- ``(ceil(F / 8), d)`` V2-S rows, row
  ``k`` = CFR frame ``8k`` (row-aligned with the CLIP cache's ``[::8]``).

CLIP is computed on the distinct decoded frames and indexed (a 15 fps clip repeats each one).
Resumable by clip (both files read back, C11); writes are atomic. CLI::

    python -m core.tools.accident_extract --videos-dir <download>/real_videos \\
        --metadata metadata-real.csv --ids-file ids.txt --clip-out cache/clip/ACCIDENT_s1_ncc \\
        --video-out cache/video/vit_s_k710_dl_from_giant/ACCIDENT_s8_squash --report extract.json
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from torch import Tensor

from core import constants
from core.device import resolve_device
from core.models.videomae_v2 import load_pretrained
from core.tools.accident_build import cut_frames, read_metadata, windowable
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

CLIP_SOURCE = (
    f"mp4 decoded with PyAV, resampled to {constants.ACCIDENT_FPS} fps CFR by timestamp "
    "(CFR frame k = last decoded frame with t <= k / fps)"
)


def cfr_index(times: np.ndarray, fps: int = constants.ACCIDENT_FPS) -> np.ndarray:
    """Index into the decoded frames of every CFR frame ``k = 0 .. floor(t_last * fps)``.

    ``times`` are the decoded frames' timestamps in decode order (must be non-decreasing);
    they are measured from the first frame, so a stream that starts at ``t0 > 0`` maps the same.
    """
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("need a non-empty 1-D array of frame times")
    if np.any(np.diff(times) < 0):
        raise ValueError("frame times decrease: decode order is not presentation order")
    rel = times - times[0]
    count = int(np.floor(rel[-1] * fps + constants.ACCIDENT_CFR_TIME_EPS)) + 1
    grid = np.arange(count) / fps
    index = np.searchsorted(rel, grid + constants.ACCIDENT_CFR_TIME_EPS, side="right") - 1
    return np.clip(index, 0, len(times) - 1).astype(np.int64)


def decode_resized_with_times(
    path: Path, until_s: float | None = None
) -> tuple[Tensor, np.ndarray]:
    """``(N, 3, 224, 224)`` resized decoded frames and their timestamps (s), chunked (C9).

    With ``until_s``, decoding stops after the first frame stamped later than ``until_s`` (from the
    first frame): that frame is kept so the CFR grid up to ``until_s`` matches a full decode.
    """
    parts: list[Tensor] = []
    times: list[float] = []
    buffer: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError(f"{path}: a decoded frame has no timestamp")
            times.append(float(frame.time))
            buffer.append(frame.to_ndarray(format="rgb24"))
            if len(buffer) == constants.NEXAR_EXTRACT_CHUNK:
                parts.append(resize_frames(np.stack(buffer)))
                buffer = []
            if until_s is not None and times[-1] - times[0] > until_s:
                break
    if buffer:
        parts.append(resize_frames(np.stack(buffer)))
    if not parts:
        raise ValueError(f"No frames decoded from {path}")
    return torch.cat(parts), np.asarray(times, dtype=np.float64)


def clip_manifest() -> dict[str, Any]:
    """What makes the ACCIDENT CLIP cache comparable to Nexar's (C2): same model and transform."""
    return {
        "model": constants.CLIP_MODEL_NAME,
        "revision": constants.CLIP_MODEL_REVISION,
        "transform": CLIP_TRANSFORM,
        "stride": 1,
        "source": CLIP_SOURCE,
    }


def extract(args: argparse.Namespace) -> dict[str, Any]:
    meta = read_metadata(args.metadata)
    requested = sorted(read_ids_file(args.ids_file))
    unknown = [v for v in requested if v not in meta]
    if unknown:
        raise KeyError(f"{len(unknown)} ids are not in {args.metadata}: {unknown[:5]}")
    ids = [v for v in requested if windowable(meta[v])]
    skipped = sorted(set(requested) - set(ids))
    paths = {v: args.videos_dir / f"{v}.mp4" for v in ids}
    missing = [v for v, p in paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} mp4 files missing in {args.videos_dir}: {missing[:5]}")
    check_clip_manifest(args.clip_out, clip_manifest())
    check_or_write_manifest(
        args.video_out,
        build_manifest(args.encoder, constants.FRAME_STRIDE,
                       constants.VIDEOMAE_WEIGHTS_SHA256[args.encoder]),
        force=False,
    )
    todo = [v for v in ids if not (is_complete(args.clip_out / f"{v}.npy")
                                   and is_complete(args.video_out / f"{v}.npy"))]
    LOGGER.info("Skipped %d clips shorter than one window; resume: %d / %d done, %d to extract",
                len(skipped), len(ids) - len(todo), len(ids), len(todo))
    device = resolve_device(args.device)
    clip_encoder = load_pretrained_encoder(device)
    video_encoder = load_pretrained(args.encoder, device, args.weights)
    progress = Progress(len(todo))
    report: dict[str, Any] = {"fps": constants.ACCIDENT_FPS, "video_stride": constants.FRAME_STRIDE,
                              "skipped_short": skipped, "videos": {}}
    started = time.monotonic()
    frames_total = 0
    for clip_id in todo:
        until = float(meta[clip_id]["accident_time"]) + args.post_impact_s
        decoded, times = decode_resized_with_times(paths[clip_id], until)
        index = cfr_index(times)
        index = index[: cut_frames(meta[clip_id], len(index), args.post_impact_s)]
        decoded = decoded[: int(index[-1]) + 1]           # frames the kept grid never reads go
        x = clip_rows(decoded, clip_encoder, device, args.clip_batch)[index]
        u = motion_rows(decoded[torch.from_numpy(index)], video_encoder, device, args.video_batch,
                        stride=constants.FRAME_STRIDE)
        if len(u) != len(x[:: constants.FRAME_STRIDE]):
            raise RuntimeError(f"{clip_id}: motion rows {len(u)} != CLIP s1[::8] rows")
        save_array(args.clip_out / f"{clip_id}.npy", np.ascontiguousarray(x))
        save_array(args.video_out / f"{clip_id}.npy", u)
        span = float(times[-1] - times[0])
        report["videos"][clip_id] = {
            "frames_decoded": len(times),
            "frames_cfr": len(index),
            "motion_rows": len(u),
            "first_time_s": round(float(times[0]), 4),
            "native_fps": round((len(times) - 1) / span, 3) if span > 0 else None,
        }
        frames_total += len(index)
        LOGGER.info("%s: %d decoded -> %d CFR frames -- %s", clip_id, len(times), len(index),
                    progress.step())
    elapsed = time.monotonic() - started
    report["summary"] = {
        "extracted": len(todo),
        "skipped_short": len(skipped),
        "frames_cfr": frames_total,
        "frames_per_s": round(frames_total / elapsed, 1) if elapsed > 0 else None,
        "clip_dir": str(args.clip_out),
        "video_dir": str(args.video_out),
        "encoder": args.encoder,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.report, report)
    LOGGER.info("Extracted %d clips (%d CFR frames, %s frames/s)", len(todo), frames_total,
                report["summary"]["frames_per_s"])
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--videos-dir", type=Path, required=True, help="<download>/real_videos")
    parser.add_argument("--metadata", type=Path, required=True, help="metadata-real.csv")
    parser.add_argument("--ids-file", type=Path, required=True, help="clip ids to extract")
    parser.add_argument("--post-impact-s", type=float, default=constants.ACCIDENT_POST_IMPACT_S,
                        help="keep frames to t_impact + this (the corpus cut)")
    parser.add_argument("--encoder", default=constants.V5_ENCODER,
                        choices=sorted(constants.VIDEOMAE_WEIGHTS_SHA256))
    parser.add_argument("--weights", type=Path, default=None, help="local VideoMAE weights")
    parser.add_argument("--clip-out", type=Path,
                        default=constants.CLIP_CACHE_DIR / constants.ACCIDENT_CLIP_S1_DATASET)
    parser.add_argument("--video-out", type=Path, required=True,
                        help="cache/video/<encoder>/ACCIDENT_s8_squash")
    parser.add_argument("--clip-batch", type=int, default=256)
    parser.add_argument("--video-batch", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", type=Path, default=None, help="per-clip frame counts (json)")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    extract(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
