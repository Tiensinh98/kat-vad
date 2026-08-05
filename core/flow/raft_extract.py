"""Offline RAFT flow-target extraction CLI (spec §1, train-time only).

Per training video: sample frames exactly like the RGB pipeline (stride 8),
run torchvision RAFT on each adjacent sampled pair, pool every flow field
into deterministic statistics (magnitude mean/std/max, u/v mean/std, a
magnitude-weighted angle histogram) and project them with a **seeded fixed
linear map** to ``d_O = 256`` (A10). The last frame duplicates the previous
flow so each video yields ``e_O in R^{L x 256}``.

Cache layout (versioned, spec/DATA_LAYOUT):

    cache/flow/v1/{DATASET}/{id}.npy         (L, 256) float32  e_O
    cache/flow/v1/{DATASET}/{id}.stats.npy   (L, 23)  float32  raw stats
    cache/flow/v1/flow_projection.npz        the fixed projection (fails loudly
                                             if missing/mismatched on load)

The raw stats are kept so the motion-aware KNN key (spec §6.1) can recover
mean magnitude/direction — unrecoverable from the projected ``e_O`` alone.

Resumable per video, on the same terms as the CLIP extractor: rerunning encodes
only what is missing, and an interrupted write is redone rather than skipped
(:mod:`core.tools.feature_cache`).

CLI::

    python -m core.flow.raft_extract --videos-dir data/MSAD/videos
        --dataset MSAD [--stride 8] [--batch-size 8] [--device auto]
        [--small] [--random-weights] [--force]
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from core import constants
from core.data.video_io import list_videos, read_sampled_frames
from core.device import resolve_device
from core.tools.feature_cache import Progress, pending_items, save_array

LOGGER = logging.getLogger(__name__)

_PROJECTION_KEYS = ("matrix", "seed", "stats_dim", "flow_dim")


def flow_statistics(flow: Tensor, angle_bins: int = constants.FLOW_ANGLE_BINS) -> Tensor:
    """Pool flow fields ``(B, 2, H, W)`` into ``(B, FLOW_STATS_DIM)`` statistics.

    Layout: [mag mean, mag std, mag max, u mean, u std, v mean, v std,
    magnitude-weighted angle histogram (``angle_bins``, L1-normalized)].
    """
    u, v = flow[:, 0], flow[:, 1]
    magnitude = torch.sqrt(u**2 + v**2 + 1e-12)
    base = torch.stack(
        [
            magnitude.mean(dim=(1, 2)),
            magnitude.std(dim=(1, 2)),
            magnitude.amax(dim=(1, 2)),
            u.mean(dim=(1, 2)),
            u.std(dim=(1, 2)),
            v.mean(dim=(1, 2)),
            v.std(dim=(1, 2)),
        ],
        dim=1,
    )
    angle = torch.atan2(v, u)  # [-pi, pi]
    bin_index = ((angle + math.pi) / (2 * math.pi) * angle_bins).long().clamp_(
        max=angle_bins - 1
    )
    histogram = torch.zeros(flow.shape[0], angle_bins, device=flow.device)
    histogram.scatter_add_(
        1, bin_index.flatten(1), magnitude.flatten(1)
    )
    histogram = histogram / histogram.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.cat([base, histogram], dim=1)


def make_projection(
    seed: int = constants.FLOW_PROJECTION_SEED,
    stats_dim: int = constants.FLOW_STATS_DIM,
    flow_dim: int = constants.FLOW_DIM,
) -> np.ndarray:
    """Seeded fixed Gaussian map ``(stats_dim, flow_dim)`` — deterministic (A10)."""
    rng = np.random.default_rng(seed)
    matrix: np.ndarray = rng.standard_normal((stats_dim, flow_dim)) / np.sqrt(stats_dim)
    return matrix.astype(np.float32)


def save_projection(path: Path, matrix: np.ndarray, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, matrix=matrix, seed=seed, stats_dim=matrix.shape[0], flow_dim=matrix.shape[1]
    )
    LOGGER.info("Flow projection saved to %s", path)


def load_projection(path: Path) -> np.ndarray:
    """Load the persisted projection; fail loudly if absent or malformed."""
    if not path.exists():
        raise FileNotFoundError(
            f"Flow projection {path} is missing for this cache version; "
            "re-run raft_extract or copy the projection that produced the cache"
        )
    payload = np.load(path)
    missing = [k for k in _PROJECTION_KEYS if k not in payload]
    if missing:
        raise ValueError(f"Flow projection {path} is missing keys {missing}")
    matrix: np.ndarray = payload["matrix"]
    return matrix.astype(np.float32)


def get_or_create_projection(cache_root: Path) -> np.ndarray:
    path = cache_root / constants.FLOW_PROJECTION_FILENAME
    if path.exists():
        return load_projection(path)
    matrix = make_projection()
    save_projection(path, matrix, constants.FLOW_PROJECTION_SEED)
    return matrix


def preprocess_for_raft(
    frames: np.ndarray, size: tuple[int, int] | None = None
) -> Tensor:
    """``(L,H,W,3)`` uint8 -> ``(L,3,H',W')`` float in [-1,1] at the RAFT size.

    ``size`` (H, W) must be divisible by 8; defaults to the production
    constants — tests pass a small size to keep CPU runs fast.
    """
    height, width = size or (constants.FLOW_RAFT_HEIGHT, constants.FLOW_RAFT_WIDTH)
    # torchvision RAFT: /8 feature maps must be >=16 -> inputs >=128 and /8
    if height % 8 or width % 8 or height < 128 or width < 128:
        raise ValueError(
            f"RAFT input size must be divisible by 8 and >=128, got {(height, width)}"
        )
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    tensor = torch.nn.functional.interpolate(
        tensor, size=(height, width), mode="bilinear", antialias=True
    )
    scaled: Tensor = tensor * 2.0 - 1.0
    return scaled


@torch.no_grad()
def video_flow_stats(
    frames: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Per-sampled-frame flow statistics ``(L, FLOW_STATS_DIM)`` for one video.

    Pair ``i`` supervises frame ``i``; the last frame duplicates the previous
    pair's statistics (spec §1). ``L == 1`` videos get a zero row.
    """
    pixels = preprocess_for_raft(frames, size)
    length = len(pixels)
    if length == 1:
        LOGGER.warning("Single-frame video; emitting zero flow statistics")
        return np.zeros((1, constants.FLOW_STATS_DIM), dtype=np.float32)
    chunks: list[Tensor] = []
    for start in range(0, length - 1, batch_size):
        end = min(start + batch_size, length - 1)
        first = pixels[start:end].to(device)
        second = pixels[start + 1 : end + 1].to(device)
        flow = model(first, second)[-1]  # final RAFT iteration, (B,2,H,W)
        chunks.append(flow_statistics(flow).cpu())
    stats = torch.cat(chunks)
    stats = torch.cat([stats, stats[-1:]], dim=0)  # duplicate for the last frame
    return stats.numpy().astype(np.float32)


def extract_directory(
    videos_dir: Path,
    dataset: str,
    model: nn.Module,
    device: torch.device,
    cache_root: Path | None = None,
    stride: int = constants.FRAME_STRIDE,
    batch_size: int = 8,
    force: bool = False,
    size: tuple[int, int] | None = None,
) -> list[Path]:
    """Cache ``e_O`` (+ raw stats) for every video; resumable per video."""
    cache_root = cache_root if cache_root is not None else constants.FLOW_CACHE_DIR
    projection = get_or_create_projection(cache_root)
    output_dir = cache_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()

    videos = {path.stem: path for path in list_videos(videos_dir)}
    pending = pending_items(videos, output_dir, force)
    progress = Progress(len(pending))
    written: list[Path] = []
    for video_id, video_path in pending:
        target = output_dir / f"{video_id}.npy"
        frames = read_sampled_frames(video_path, stride)
        stats = video_flow_stats(frames, model, device, batch_size, size)
        embeddings = (stats @ projection).astype(np.float32)
        # e_O last: it is what resume keys on, so a crash in between redoes both
        save_array(output_dir / f"{video_id}{constants.FLOW_STATS_SUFFIX}", stats)
        save_array(target, embeddings)
        written.append(target)
        LOGGER.info("Saved %s %s -- %s", target.name, embeddings.shape, progress.step())
    LOGGER.info("Extracted flow for %d new videos into %s", len(written), output_dir)
    return written


def load_raft_model(small: bool = False, random_weights: bool = False) -> nn.Module:
    """torchvision RAFT; ``random_weights`` skips the download (dev/tests only)."""
    from torchvision.models import optical_flow as of

    model: nn.Module
    if small:
        small_weights = None if random_weights else of.Raft_Small_Weights.DEFAULT
        model = of.raft_small(weights=small_weights)
    else:
        large_weights = (
            None
            if random_weights
            else of.Raft_Large_Weights[constants.RAFT_WEIGHTS_NAME]
        )
        model = of.raft_large(weights=large_weights)
    return model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--dataset", default=constants.MSAD_DATASET)
    parser.add_argument("--cache-root", type=Path, default=None,
                        help="default: cache/flow/v1")
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--small", action="store_true", help="use raft_small")
    parser.add_argument("--random-weights", action="store_true",
                        help="random-init RAFT (shape/dev runs only)")
    parser.add_argument("--force", action="store_true", help="re-extract existing files")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    model = load_raft_model(small=args.small, random_weights=args.random_weights)
    extract_directory(
        videos_dir=args.videos_dir,
        dataset=args.dataset,
        model=model,
        device=device,
        cache_root=args.cache_root,
        stride=args.stride,
        batch_size=args.batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
