"""SA²-v2's inputs: 7 x 7 pooled CLIP patch tokens, level constants, the C2 label permutation.

Rule ``core/docs/final/SA2V2_RUN.md``. Subcommands:

* ``pool``: existing 14 x 14 token grids (branch ``v5-r7``'s ``cgrid14`` caches) -> cells; every row
  count is checked against the host bake (lesson C13).
* ``extract``: ``zs_frames`` row-frame archives -> one CLIP pass per frame -> cells. A video passes
  only if the projected CLS equals ``image_embeds`` and reproduces the video's own CLIP rows.
* ``stats``: ``m, s`` of Mo and rho over the training sources' cells (``level_stats.json``).
* ``shuffle-labels``: a copy of a data dir with labels and class names permuted across bags (C2).

CLI::

    python -m core.tools.sa2v2_cells pool --grid-dir .../Nexar_s8_cgrid14 --ids-file ids.txt \\
        --input-dir .../A2_S/Nexar_s8 --dataset Nexar --stride 8 --out-dir .../Nexar_s8_g7
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from core import constants
from core.data.stp_patches import cell_columns, cell_manifest, pool_token_grid
from core.data.v2_splits import lines_sha1
from core.data.video_io import list_frame_images, read_images
from core.data.windows import FeatureSlicer, load_windows
from core.device import resolve_device
from core.models.sa2v2 import cell_priors
from core.tools import dota_cap_frames, extract_clip_features, extract_video_features
from core.tools.clip_windows import (
    corpus_gate,
    gate_rows,
    squash,
    window_boxes,
    window_pixels,
)
from core.tools.feature_cache import Progress, is_complete, read_ids_file, save_array
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

OK = "ok"
ROWS_MISMATCH, G1_FAILED = "rows_mismatch", "g1_cls"
LEVEL_KEYS = ("mo_mean", "mo_std", "rho_mean", "rho_std")
STATS_BATCH_ROWS = 1024  # rows per prior computation in ``stats``


def flatten_cells(cells: np.ndarray) -> np.ndarray:
    """``(L, 49, D)`` -> the stored ``(L, 49 * D)`` (2-D, so :func:`is_complete` reads it)."""
    return cells.reshape(len(cells), -1).astype(constants.SV2_STORE_DTYPE)


# --------------------------------------------------------------------------- pool


def run_pool(args: argparse.Namespace) -> dict[str, Any]:
    ids = sorted(read_ids_file(args.ids_file))
    extract_video_features.check_or_write_manifest(
        args.out_dir,
        cell_manifest(args.dataset, args.stride, f"pooled from {args.grid_dir.name}"),
        args.force,
    )
    bad: dict[str, str] = {}
    done = 0
    for video_id in ids:
        target = args.out_dir / f"{video_id}.npy"
        host = np.load(args.input_dir / f"{video_id}.npy", mmap_mode="r").shape[0]
        if not args.force and is_complete(target):
            if np.load(target, mmap_mode="r").shape[0] != host:
                bad[video_id] = ROWS_MISMATCH
            done += 1
            continue
        path = args.grid_dir / f"{video_id}.npy"
        if not path.is_file():
            bad[video_id] = "no_grid"
            continue
        grid = np.load(path)
        if len(grid) != host:
            bad[video_id] = ROWS_MISMATCH
            continue
        save_array(target, flatten_cells(pool_token_grid(grid)))
        done += 1
    out = {"videos": len(ids), "done": done, "bad": bad}
    if bad:
        raise ValueError(f"{len(bad)} of {len(ids)} videos failed: {dict(list(bad.items())[:8])}")
    LOGGER.info("pooled %d videos -> %s", done, args.out_dir)
    return out


# --------------------------------------------------------------------------- extract


@torch.no_grad()
def clip_cells(frames: Tensor, model: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """``(N, 3, H, W)`` in [0, 1] -> ``image_embeds (N, D)``, cells ``(N, 49, D)``, G1's gap.

    One pass per frame. Patch tokens take the CLS token's path, so they live in ``image_embeds``'
    space; the pixels are the CLIP cache's (squash to 224^2, CLIP mean / std).
    """
    resized = squash(frames)
    side = resized.shape[-1]
    pixels = window_pixels(resized, window_boxes(side, side, (constants.TW_GLOBAL_GRID,)))
    out = model(pixel_values=pixels)
    vision, project = model.vision_model, model.visual_projection
    hidden = out.last_hidden_state
    gap = float((project(vision.post_layernorm(hidden[:, 0])) - out.image_embeds).abs().max())
    tokens = project(vision.post_layernorm(hidden[:, 1:])).float().cpu().numpy()
    grid = constants.SV2_SOURCE_GRID
    if tokens.shape[1] != grid * grid:
        raise ValueError(f"{tokens.shape[1]} patch tokens, expected {grid} x {grid}")
    return out.image_embeds.float().cpu().numpy(), pool_token_grid(tokens), gap


def encode_cells(
    paths: list[Path], model: Any, device: torch.device, batch: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Streamed (C9) ``image_embeds``, cells and the worst G1 gap of one video's frames."""
    embeds, cells, worst = [], [], 0.0
    for start in range(0, len(paths), batch):
        frames = torch.from_numpy(read_images(paths[start : start + batch]))
        frames = frames.to(device).permute(0, 3, 1, 2).float() / 255.0
        embed, cell, gap = clip_cells(frames, model)
        embeds.append(embed)
        cells.append(cell)
        worst = max(worst, gap)
    return np.concatenate(embeds), np.concatenate(cells), worst


def gate_video(embeds: np.ndarray, clip_rows: np.ndarray, gap: float) -> dict[str, Any]:
    """G1 + the row and cosine gate against the video's own CLIP rows (same pixels)."""
    if gap > constants.SV2_G1_MAX_ABS:
        return {"reason": G1_FAILED, "g1_max_abs": gap}
    gate = gate_rows(
        embeds[:, None], clip_rows, constants.DOTA_CAP_MEAN_COS, (constants.TW_GLOBAL_GRID,)
    )
    return {**gate, "g1_max_abs": gap}


def run_extract(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.archive_dir / constants.ZS_FRAMES_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} missing: run zs_frames verify first")
    archive = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset, stride = str(archive["dataset"]), int(archive["stride"])
    extract_video_features.check_or_write_manifest(
        args.out_dir,
        cell_manifest(dataset, stride, f"zs_frames (ids sha1 {archive['ids_sha1']})"),
        args.force,
    )
    report: dict[str, dict[str, Any]] = (
        json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    )
    ids = sorted(read_ids_file(args.ids_file))
    todo = [v for v in ids if args.force or not is_complete(args.out_dir / f"{v}.npy")]
    device = resolve_device(args.device)
    model = extract_clip_features.load_pretrained_encoder(device)
    progress = Progress(len(todo))
    for video_id in todo:
        folder = dota_cap_frames.unpack(
            dota_cap_frames.archive_path(args.archive_dir, video_id), args.work_dir / video_id
        )
        try:
            embeds, cells, gap = encode_cells(
                list_frame_images(folder), model, device, args.frames_per_batch
            )
            gate = gate_video(embeds, np.load(args.clip_dir / f"{video_id}.npy"), gap)
            if gate["reason"] == OK:
                save_array(args.out_dir / f"{video_id}.npy", flatten_cells(cells))
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        report[video_id] = gate
        write_json_atomic(args.report, report)
        LOGGER.info("%s: %s -- %s", video_id, gate, progress.step())
    entries = [report[v] for v in ids if v in report and "global_mean_cos" in report[v]]
    out = corpus_gate(entries) if entries else {"clips": 0, "reason": OK}
    missing = {v: report.get(v, {}).get("reason", "not_run")
               for v in ids if not is_complete(args.out_dir / f"{v}.npy")}
    if missing or out["reason"] != OK:
        raise ValueError(f"extract failed: corpus {out}, missing {dict(list(missing.items())[:8])}")
    LOGGER.info("extracted %d videos, corpus gate %s", len(ids), out)
    return out


# --------------------------------------------------------------------------- stats


def training_sources(data_dir: Path) -> list[str]:
    """Source videos behind every training item of a (windowed) data dir, sorted."""
    labels = json.loads((data_dir / constants.LABELS_TRAIN_FILENAME).read_text(encoding="utf-8"))
    slicer = FeatureSlicer(load_windows(data_dir))
    return sorted({slicer.source_of(item) for item in labels})


def level_stats(cell_dir: Path, sources: list[str]) -> dict[str, Any]:
    """Mean and std of Mo and rho over all cells and steps of ``sources``."""
    count = 0
    totals = {"mo": [0.0, 0.0], "rho": [0.0, 0.0]}  # [sum, sum of squares]
    for video_id in sources:
        columns = cell_columns(np.load(cell_dir / f"{video_id}.npy", mmap_mode="r"))
        flat = torch.from_numpy(columns).reshape(-1, 2, constants.SV2_CELLS, constants.SV2_DIM)
        for start in range(0, len(flat), STATS_BATCH_ROWS):
            part = flat[start : start + STATS_BATCH_ROWS]
            mo, rho = cell_priors(part[:, 0], part[:, 1])
            count += mo.numel()
            for name, values in (("mo", mo), ("rho", rho)):
                totals[name][0] += float(values.double().sum())
                totals[name][1] += float(values.double().pow(2).sum())
    out: dict[str, Any] = {"sources": len(sources), "cells": count,
                           "ids_sha1": lines_sha1(sources)}
    for name, (total, squares) in totals.items():
        mean = total / count
        out[f"{name}_mean"] = mean
        out[f"{name}_std"] = math.sqrt(max(squares / count - mean * mean, 0.0))
    return out


def run_stats(args: argparse.Namespace) -> dict[str, Any]:
    stats = level_stats(args.cell_dir, training_sources(args.data_dir))
    write_json_atomic(args.cell_dir / constants.SV2_LEVEL_FILENAME, stats)
    LOGGER.info("level stats: %s", stats)
    return stats


# --------------------------------------------------------------------------- shuffle-labels


def run_shuffle(args: argparse.Namespace) -> dict[str, Any]:
    """A copy of ``data_dir`` whose bag labels and class names follow a permutation of the bags."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.data_dir.glob("*.json"):
        shutil.copy2(path, args.out_dir / path.name)
    labels: dict[str, int] = json.loads(
        (args.data_dir / constants.LABELS_TRAIN_FILENAME).read_text(encoding="utf-8")
    )
    ids = sorted(labels)
    order = np.random.default_rng(args.seed).permutation(len(ids))
    donor = {v: ids[int(i)] for v, i in zip(ids, order, strict=True)}
    shuffled = {v: labels[donor[v]] for v in ids}
    write_json_atomic(args.out_dir / constants.LABELS_TRAIN_FILENAME, shuffled)
    meta_path = args.data_dir / constants.META_FILENAME
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        names = {v: meta[v]["class_name"] for v in ids if v in meta}
        for v in ids:
            if v in meta and donor[v] in names:
                meta[v]["class_name"] = names[donor[v]]
        write_json_atomic(args.out_dir / constants.META_FILENAME, meta)
    out = {
        "seed": args.seed,
        "bags": len(ids),
        "abnormal": sum(shuffled.values()),
        "unchanged_label_share": float(np.mean([shuffled[v] == labels[v] for v in ids])),
    }
    write_json_atomic(args.out_dir / "shuffle_report.json", out)
    LOGGER.info("shuffled labels: %s", out)
    return out


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    pool = sub.add_parser("pool", help="14 x 14 token grids -> 7 x 7 cells")
    pool.add_argument("--grid-dir", type=Path, required=True)
    pool.add_argument("--ids-file", type=Path, required=True)
    pool.add_argument("--input-dir", type=Path, required=True, help="host bake: row counts (C13)")
    pool.add_argument("--dataset", required=True)
    pool.add_argument("--stride", type=int, required=True)
    pool.add_argument("--out-dir", type=Path, required=True)
    pool.add_argument("--force", action="store_true")

    extract = sub.add_parser("extract", help="zs_frames archives -> one CLIP pass -> 7 x 7 cells")
    extract.add_argument("--archive-dir", type=Path, required=True)
    extract.add_argument("--work-dir", type=Path, required=True, help="VM disk, not Drive")
    extract.add_argument("--clip-dir", type=Path, required=True, help="the videos' own CLIP rows")
    extract.add_argument("--ids-file", type=Path, required=True)
    extract.add_argument("--report", type=Path, required=True, help="per-video JSON, resumable")
    extract.add_argument("--out-dir", type=Path, required=True)
    extract.add_argument("--frames-per-batch", type=int, default=constants.SV2_FRAMES_PER_BATCH)
    extract.add_argument("--device", default="auto")
    extract.add_argument("--force", action="store_true")

    stats = sub.add_parser("stats", help="level constants of Mo and rho")
    stats.add_argument("--cell-dir", type=Path, required=True)
    stats.add_argument("--data-dir", type=Path, required=True, help="training data dir")

    shuffle = sub.add_parser("shuffle-labels", help="control C2's data dir")
    shuffle.add_argument("--data-dir", type=Path, required=True)
    shuffle.add_argument("--out-dir", type=Path, required=True)
    shuffle.add_argument("--seed", type=int, default=constants.SV2_SHUFFLE_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    args = build_arg_parser().parse_args(argv)
    handlers = {"pool": run_pool, "extract": run_extract, "stats": run_stats,
                "shuffle-labels": run_shuffle}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
