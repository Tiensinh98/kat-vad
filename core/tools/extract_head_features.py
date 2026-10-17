"""v4 Stage A: per-head VideoMAE V2 features, row-aligned with the v2 caches (PREREG I1-I4).

One ``{id}.npy`` of float32 ``(T, depth * heads * head_dim)`` per source video (2-D, so the
cache's resume check reads it; :func:`load_head_rows` restores ``(T, heads, head_dim)``): for every
CLIP step, the token mean of each block's ``attn.proj`` input (:class:`HeadTap`), for the
**same** causal clip v2's ``u`` encodes. The clip geometry, transform and weights are
v2's, through :func:`core.tools.extract_video_features.encode_frame_dir` called unchanged;
the taps only read.

**Row check (I4).** The pooled ``u`` of the same forward pass is compared with the
existing V2-S cache (``--reference-dir``) row by row; the per-row cosines go to
``ucos/{id}.npy`` *before* the head file is written, so a head file implies its check.
:func:`ucos_failures` is the gate the selection runs first. The reference cache's
manifest must equal this pass's geometry (C2), and so must this cache's on a rerun.

CLI::

    python -m core.tools.extract_head_features --frames-dir /content/v4/flat \\
        --ids-file shard.txt --dataset DADA2000_orig \\
        --reference-dir cache/video/vit_s_k710_dl_from_giant/DADA2000_orig_s8_squash \\
        --output-dir cache/v4/heads/vit_s_k710_dl_from_giant/DADA2000_orig_s8_squash \\
        [--stride 8] [--batch-size 32] [--device auto] [--weights local.pth] [--force]
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

from core import constants
from core.data.video_io import list_frame_folders, video_id_from_path
from core.device import resolve_device
from core.models.videomae_heads import HeadTap
from core.models.videomae_v2 import VideoMAEv2Encoder, load_pretrained
from core.tools import extract_video_features as evf
from core.tools.feature_cache import (
    Progress,
    is_complete,
    pending_items,
    read_ids_file,
    save_array,
    select_ids,
)

LOGGER = logging.getLogger(__name__)

LAYOUT = (
    "(T, depth*heads*head_dim); reshape (T, depth*heads, head_dim); "
    "head index = block*heads + head"
)


def build_manifest(
    encoder_name: str, stride: int, weights_sha256: str, depth: int, heads: int, head_dim: int
) -> dict[str, Any]:
    """v2's geometry manifest with the head feature in place of ``u`` (C2)."""
    manifest = evf.build_manifest(encoder_name, stride, weights_sha256)
    manifest.update({
        "feature": constants.V4_HEAD_FEATURE,
        "layout": LAYOUT,
        "depth": depth,
        "heads": heads,
        "head_dim": head_dim,
        "u_check": f"per-row cosine vs the V2-S cache >= {constants.V4_U_COS_MIN}",
    })
    return manifest


def check_reference_manifest(reference_dir: Path, encoder_name: str, stride: int) -> None:
    """The ``u`` cache we check against must have been built with this exact geometry (C2)."""
    path = reference_dir / constants.VIDEO_MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: an unverifiable reference cache")
    found = json.loads(path.read_text(encoding="utf-8"))
    expected = evf.build_manifest(
        encoder_name, stride, constants.VIDEOMAE_WEIGHTS_SHA256[encoder_name]
    )
    diff = sorted(k for k in expected if found.get(k) != expected[k])
    if diff:
        raise ValueError(f"{reference_dir} was built with different {diff} than this pass")


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine of two ``(T, D)`` arrays; a length mismatch raises (C13)."""
    if a.shape != b.shape:
        raise ValueError(f"pooled u {a.shape} vs cached u {b.shape}")
    num = (a.astype(np.float64) * b.astype(np.float64)).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out: np.ndarray = num / np.maximum(den, np.finfo(np.float64).tiny)
    return out


@torch.no_grad()
def encode_heads(
    folder: Path,
    encoder: VideoMAEv2Encoder,
    device: torch.device,
    stride: int,
    batch_size: int,
    subdir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(heads (T, depth*heads, head_dim), u (T, dim))`` of one frame folder, one pass."""
    with HeadTap(encoder) as tap:
        u = evf.encode_frame_dir(folder, encoder, device, stride, batch_size, subdir)
        heads = tap.rows()
    if len(heads) != len(u):
        raise RuntimeError(f"{folder}: {len(heads)} tapped rows vs {len(u)} pooled rows")
    return heads, u


def ucos_path(output_dir: Path, video_id: str) -> Path:
    return output_dir / constants.V4_UCOS_SUBDIR / f"{video_id}.npy"


def extract_heads_directory(
    frames_dir: Path,
    output_dir: Path,
    reference_dir: Path,
    encoder: VideoMAEv2Encoder,
    device: torch.device,
    stride: int = constants.FRAME_STRIDE,
    batch_size: int = 16,
    force: bool = False,
    subdir: str | None = None,
    video_ids: set[str] | None = None,
) -> list[Path]:
    """Encode every (selected) frame folder; ``ucos`` first, then heads, both atomic (C11)."""
    (output_dir / constants.V4_UCOS_SUBDIR).mkdir(parents=True, exist_ok=True)
    folders = select_ids(
        {video_id_from_path(p): p for p in list_frame_folders(frames_dir, subdir)},
        video_ids, frames_dir, "frame folder",
    )
    pending = pending_items(folders, output_dir, force)
    progress = Progress(len(pending))
    written: list[Path] = []
    started = time.monotonic()
    for video_id, folder in pending:
        reference = reference_dir / f"{video_id}.npy"
        if not reference.is_file():
            raise FileNotFoundError(f"{video_id}: no cached u in {reference_dir} to check against")
        heads, u = encode_heads(folder, encoder, device, stride, batch_size, subdir)
        cosines = row_cosines(u, np.load(reference))
        save_array(ucos_path(output_dir, video_id), cosines.astype(np.float32)[:, None])
        target = output_dir / f"{video_id}.npy"
        save_array(target, heads.reshape(len(heads), -1))
        written.append(target)
        LOGGER.info("Saved %s %s min cos %.6f -- %s",
                    target.name, heads.shape, float(cosines.min()), progress.step())
    LOGGER.info("Extracted %d head files into %s in %.1f min",
                len(written), output_dir, (time.monotonic() - started) / 60)
    return written


def load_head_rows(heads_dir: Path, ids: list[str]) -> dict[str, np.ndarray]:
    """``{id: (T, depth*heads, head_dim)}`` float32; the manifest gives ``head_dim`` (C2)."""
    manifest = json.loads((heads_dir / constants.VIDEO_MANIFEST_FILENAME).read_text("utf-8"))
    head_dim = int(manifest["head_dim"])
    missing = [v for v in ids if not (heads_dir / f"{v}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} sources have no head file: {missing[:5]}")
    out: dict[str, np.ndarray] = {}
    for v in ids:
        flat = np.load(heads_dir / f"{v}.npy")
        out[v] = flat.reshape(len(flat), -1, head_dim)
    return out


def ucos_failures(
    output_dir: Path, ids: list[str], bar: float = constants.V4_U_COS_MIN
) -> dict[str, float]:
    """``{id: min cosine}`` for every id whose check is missing (-1) or below ``bar`` (I4)."""
    bad: dict[str, float] = {}
    for video_id in ids:
        path = ucos_path(output_dir, video_id)
        if not (is_complete(path) and is_complete(output_dir / f"{video_id}.npy")):
            bad[video_id] = -1.0
            continue
        low = float(np.load(path).min())
        if low < bar:
            bad[video_id] = low
    return bad


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--frames-subdir", default=None)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="the existing V2-S u cache the pooled rows are checked against (I4)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", default=constants.V4_ENCODER,
                        choices=sorted(constants.VIDEOMAE_ARCH))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    check_reference_manifest(args.reference_dir, args.encoder, args.stride)
    device = resolve_device(args.device)
    encoder = load_pretrained(args.encoder, device, args.weights)
    dim, depth, heads = constants.VIDEOMAE_ARCH[args.encoder]
    manifest = build_manifest(
        args.encoder, args.stride, constants.VIDEOMAE_WEIGHTS_SHA256[args.encoder],
        depth, heads, dim // heads,
    )
    evf.check_or_write_manifest(args.output_dir, manifest, args.force)
    LOGGER.info("%s heads (%d x %d x %d) of %s -> %s",
                args.encoder, depth, heads, dim // heads, args.dataset, args.output_dir)
    extract_heads_directory(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        reference_dir=args.reference_dir,
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
