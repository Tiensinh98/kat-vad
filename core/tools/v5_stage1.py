"""v5 Stage 1 -- Nexar grids and the hard gates before any Stage 2 number (PREREG_V5 §2.2).

Subcommands (each writes ``<name>.json`` + ``<name>.md`` under ``--out-dir``):

``extract`` (Colab, GPU)
    One PyAV decode per Nexar mp4 (N2's ``decode_resized``: one resize, the squash normalization
    bit-exact to ``Nexar_s1_squash``), V2-S tokens of the causal window ending at each row
    ``s1[::stride]``, the V7 grid (S0-4) stored fp16. **S1-G1** (§2.4 B9): a video whose decoded
    length differs from ``Nexar_s1_squash`` stores nothing (fails); the per-row
    ``cos(fc_norm(mean_j G_t(j)), Nexar_s1_squash[t])`` on the stored grid is printed only.
    Training positives at stride 8, ``nexar_val`` at stride 1 (§2.4 B1). Resumable per video.
    ``--clip-grid`` (R7, PREREG_V5 §4 Phase 1): CLIP ViT-B/16 patch grids ``(R, 14, 14, 512)`` of
    the same rows instead, stored unpooled; G1 = length, projected CLS == ``image_embeds``, and
    every stored row's ``image_embeds`` vs ``Nexar_s1_ncc`` >= ``V5_R7_G1_MIN_COS``.
``head``
    V2-S's ``fc_norm`` + Nexar A2's baked ``c``, ``m_u``, ``sigma_u`` -> ``v5_head.npz`` in the
    training grid cache (deviation D1).
``init-check`` -- **S1-G2**
    (a) on fp32 synthetic grids, every query readout at init equals the tile mean (|Δ| <= 1e-6),
    M in {1, 8}, both key types; a DVS splice carries each item's own ``ΔG``;
    (b) on ``V5_G2_WINDOWS`` real Nexar training windows, every row's motion term at init equals
    R1's (the tile-mean readout of the same grid; |Δ| <= 1e-3; §2.4 B9).
``provenance`` -- **S1-G4**: J10 (checkpoint step == last metrics step == the plan's steps) and
    ``config.yaml`` == v2 Nexar A2's of the same seed except the ``v5`` section (= the row's).
``collapse`` -- **S1-G6** (read before S2-7): last-step ``mil`` / ``smil`` finite; query entropy;
    "did not leave the mean" iff every query's entropy >= 0.999 log 49.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from core import constants
from core.config import Config, load_config
from core.data.synthesis import compose_sequence
from core.data.v2_inputs import load_stats, read_input_manifest
from core.data.v5_grids import (
    centred_reference,
    change_lag,
    clip_grid_manifest,
    grid_change,
    read_grid_geometry,
    save_head,
    v5_rows,
)
from core.data.windows import FeatureSlicer, load_windows
from core.device import resolve_device
from core.models.kat_vad import KATVAD
from core.models.mqr import GridReadout
from core.models.videomae_v2 import VideoMAEv2Encoder, load_pretrained
from core.tools.dota_cap import frame_cosines
from core.tools.extract_clip_features import load_pretrained_encoder
from core.tools.extract_video_features import (
    causal_clip_indices,
    centred_clip_indices,
    check_or_write_manifest,
)
from core.tools.feature_cache import Progress, read_ids_file, save_array
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_extract import CLIP_TRANSFORM, decode_resized, normalize, video_paths
from core.tools.v5_stage0 import clip_patch_tokens, grid_complete, tile_grid, token_grids

LOGGER = logging.getLogger(__name__)

OK = "ok"
RULE = "core/docs/v5/PREREG_V5.md §2.2"
METRICS_FILENAME = "metrics.jsonl"
CKPT_FILENAME = "checkpoint_last.pt"
CONFIG_FILENAME = "config.yaml"
A2_OVERRIDES = [  # v2 Nexar A2's recipe, the base every v5 row is built on (S2-1)
    "kip.enabled=false",
    f"v2.motion={constants.V5_ENCODER}",
    f"model.motion_dim={constants.V5_GRID_DIM}",
]


def _write(out_dir: Path, name: str, payload: dict[str, Any], md: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / f"{name}.json", payload)
    write_text_atomic(out_dir / f"{name}.md", md)
    return payload


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------------------- extract (S1-G1)


@torch.no_grad()
def exact_fp32() -> None:
    """Plain fp32 matmul and convolution, no TF32: how v2's ``Nexar_s1_squash`` was computed.

    With cuDNN's default TF32 a few A100 videos fell to cos 0.9998 against the v2 rows; with it
    off they read 1.00000000 (§2.4 B8).
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def grid_rows(
    resized: torch.Tensor,
    rows: list[int],
    encoder: VideoMAEv2Encoder,
    device: torch.device,
    batch_size: int,
    frame_step: int = constants.VIDEOMAE_CLIP_FRAME_STEP,
    tile: bool = False,
) -> np.ndarray:
    """Float32 grids per row: V7 ``(R, 7, 7, D)`` of the causal window ending at the row, or
    (``tile``, v5.1) ``(R, 4, 7, 7, D)`` of the window centred on it.

    The video goes to ``device`` once; windows are gathered and normalized there (elementwise
    fp32, same values as on the CPU), so the CPU only decodes.
    """
    video = resized.to(device)
    out: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        clips = (
            [centred_clip_indices(t, len(resized), step=frame_step) for t in chunk]
            if tile
            else [causal_clip_indices(t, step=frame_step) for t in chunk]
        )
        index = torch.tensor(clips, device=device)
        frames = normalize(
            video[index], constants.VIDEOMAE_IMAGE_MEAN, constants.VIDEOMAE_IMAGE_STD
        )  # (B, 16, 3, H, W): N2's squash transform on N2's resize
        tokens = encoder.forward_tokens(frames.permute(0, 2, 1, 3, 4)).float()
        grid = tile_grid(tokens) if tile else token_grids(tokens)[1]
        out.append(grid.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def clip_grid_rows(
    resized: torch.Tensor, rows: list[int], model: Any, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """R7: ``image_embeds (R, 512)``, CLIP patch grids ``(R, 14, 14, 512)`` float32 of frames
    ``rows`` and G1's max |projected CLS - image_embeds|.

    Pixels are N2's CLIP rows exactly (``nexar_extract.clip_rows``: the shared resize, CLIP
    mean/std on the CPU), so ``image_embeds`` reproduces ``Nexar_s1_ncc``.
    """
    embeds, grids, worst = [], [], 0.0
    for start in range(0, len(rows), batch_size):
        pixels = normalize(
            resized[rows[start : start + batch_size]],
            constants.CLIP_IMAGE_MEAN,
            constants.CLIP_IMAGE_STD,
        )
        embed, grid, gap = clip_patch_tokens(pixels.to(device), model)
        worst = max(worst, gap)
        embeds.append(embed)
        grids.append(grid)
    return torch.cat(embeds).numpy(), torch.cat(grids).numpy(), worst


@torch.no_grad()
def g1_cosines(grid16: np.ndarray, reference: np.ndarray, encoder: VideoMAEv2Encoder) -> np.ndarray:
    """Per-row ``cos(fc_norm(mean tiles of the stored fp16 grid), reference row)`` (S1-G1)."""
    device = encoder.fc_norm.weight.device
    flat = torch.from_numpy(grid16.astype(np.float32)).reshape(len(grid16), -1, grid16.shape[-1])
    pooled = encoder.fc_norm(flat.to(device).mean(1)).double().cpu().numpy()
    ref = reference.astype(np.float64)
    num = (pooled * ref).sum(-1)
    den = np.linalg.norm(pooled, axis=-1) * np.linalg.norm(ref, axis=-1)
    cos: np.ndarray = num / np.maximum(den, np.finfo(np.float64).tiny)
    return cos


def grid_manifest(encoder: str, stride: int, tile: bool = False) -> dict[str, Any]:
    if tile:
        return {
            "encoder": encoder,
            "weights_sha256": constants.VIDEOMAE_WEIGHTS_SHA256[encoder],
            "stride": stride,
            "frame_step": constants.VIDEOMAE_CLIP_FRAME_STEP,
            "slots": constants.V5_SLOTS_TILE,
            "window_kind": constants.V5_WINDOW_CENTRED,
            "rows": f"s1[::{stride}] of the decoded mp4 (Nexar N2 frame order)",
            "window": "centred 16 frames t-8*step..t+7*step, edge frames repeated",
            "grid": "4x7x7: pairs of the 8 slots, then 2x2, last layer before fc_norm",
            "dtype": "float16",
            "transform": "N2 decode_resized + squash normalize",
            "rule": "core/docs/v5/PREREG_V5.md §3",
        }
    return {
        "encoder": encoder,
        "weights_sha256": constants.VIDEOMAE_WEIGHTS_SHA256[encoder],
        "stride": stride,
        "frame_step": constants.VIDEOMAE_CLIP_FRAME_STEP,
        "rows": f"s1[::{stride}] of the decoded mp4 (Nexar N2 frame order)",
        "window": "causal 16 frames ending at the row (Nexar_s1_squash geometry)",
        "grid": f"{constants.V5_GRID_COARSE}x{constants.V5_GRID_COARSE}, mean over T' then 2x2, "
        "last layer before fc_norm",
        "dtype": "float16",
        "transform": "N2 decode_resized + squash normalize",
        "rule": "core/docs/v5/PREREG_V5.md S2-2",
    }


def decoded_ahead(paths: dict[str, Path], ids: list[str]) -> Iterator[tuple[str, torch.Tensor]]:
    """``(video_id, decode_resized(path))`` in order, the next video decoding on a worker thread.

    PyAV decode and the CPU resize overlap the GPU forward of the current video; one video ahead
    only, so at most two decoded videos are in memory.
    """
    if not ids:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(decode_resized, paths[ids[0]])
        for i, video_id in enumerate(ids):
            frames = pending.result()
            if i + 1 < len(ids):
                pending = pool.submit(decode_resized, paths[ids[i + 1]])
            yield video_id, frames


def clip_grid_entry(
    resized: torch.Tensor,
    reference: np.ndarray,
    stride: int,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """R7 G1 on one video: its report entry and the fp16 grid to store (``None`` = failed)."""
    entry: dict[str, Any] = {"frames": len(resized)}
    if len(resized) != len(reference):
        return {**entry, "reason": "g1_length"}, None
    rows = list(range(0, len(resized), stride))
    embeds, grid, worst = clip_grid_rows(resized, rows, model, device, batch_size)
    cos = frame_cosines(embeds, np.asarray(reference[rows]))
    entry |= {
        "rows": len(rows),
        "g1_max_abs": worst,
        "g1_min_cos": float(cos.min()),
        "g1_mean_cos": float(cos.mean()),
    }
    if worst > constants.V5_G1_MAX_ABS:
        return {**entry, "reason": "g1_cls"}, None
    if entry["g1_min_cos"] < constants.V5_R7_G1_MIN_COS:
        return {**entry, "reason": "g1_cos"}, None
    return {**entry, "reason": OK}, grid.astype(np.float16)


def run_extract(args: argparse.Namespace) -> dict[str, Any]:
    if args.clip_grid:
        return run_extract_clip(args)
    if args.video_s1_dir is None:
        raise ValueError("--video-s1-dir is required without --clip-grid (S1-G1)")
    census = json.loads(args.census.read_text(encoding="utf-8"))
    ids = sorted(read_ids_file(args.ids_file))
    paths = video_paths(args.root, census, set(ids))
    tile = bool(args.tile)
    check_or_write_manifest(
        args.grid_dir, grid_manifest(args.encoder, args.stride, tile), force=False
    )
    report: dict[str, Any] = (
        json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    )
    todo = [
        v
        for v in ids
        if not (report.get(v, {}).get("reason") == OK and grid_complete(args.grid_dir / f"{v}.npy"))
    ]
    LOGGER.info(
        "grids: %d / %d videos done, %d to extract", len(ids) - len(todo), len(ids), len(todo)
    )
    device = resolve_device(args.device)
    exact_fp32()
    encoder = load_pretrained(args.encoder, device, args.weights) if todo else None
    progress = Progress(len(todo))
    for video_id, resized in decoded_ahead(paths, todo):
        assert encoder is not None  # nosec B101 - loaded whenever there is work
        reference = np.load(args.video_s1_dir / f"{video_id}.npy", mmap_mode="r")
        entry: dict[str, Any] = {"frames": len(resized)}
        if len(resized) != len(reference):
            report[video_id] = {**entry, "reason": "g1_length"}
            continue
        rows = list(range(0, len(resized), args.stride))
        grid16 = grid_rows(resized, rows, encoder, device, args.batch_size, tile=tile).astype(
            np.float16
        )
        if tile:  # G-c: centred rows == the v2 causal row of the same 16 frames (shifted)
            keep, index = centred_reference(
                np.asarray(rows), constants.VIDEOMAE_CLIP_FRAME_STEP, len(resized)
            )
            cos = g1_cosines(grid16[keep], np.asarray(reference[index]), encoder)
        else:
            cos = g1_cosines(grid16, np.asarray(reference[rows]), encoder)
        entry |= {
            "rows": len(rows),
            "g1_min_cos": float(cos.min()) if len(cos) else None,
            "g1_mean_cos": float(cos.mean()) if len(cos) else None,
        }
        save_array(args.grid_dir / f"{video_id}.npy", grid16)
        report[video_id] = {**entry, "reason": OK}
        write_json_atomic(args.report, report)
        LOGGER.info(
            "%s: %d rows, G1 min cos %.8f -- %s",
            video_id,
            len(rows),
            entry["g1_min_cos"],
            progress.step(),
        )
    write_json_atomic(args.report, report)
    failed = {v: e["reason"] for v, e in report.items() if v in ids and e["reason"] != OK}
    mins = [
        report[v]["g1_min_cos"]
        for v in ids
        if report.get(v, {}).get("g1_min_cos") is not None
    ]
    summary = {
        "rule": "S1-G1 (S2-2, §2.4 B9: length gates; cos vs v2 printed only)"
        + (" + v5.1 G-c (centred, shifted rows; §3)" if tile else ""),
        "tile": tile,
        "tf32": False,
        "grid_dir": str(args.grid_dir),
        "stride": args.stride,
        "videos": len(ids),
        "ok": sum(1 for v in ids if report.get(v, {}).get("reason") == OK),
        "failed": failed,
        "g1_min_cos": min(mins, default=None),
        "videos_min_cos_below_0.9999": sum(1 for c in mins if c < constants.V5_G0_MIN_COS),
        "pass": not failed and all(v in report for v in ids),
    }
    md = (
        f"# S1-G1 grid extraction (stride {args.stride})\n\n**{_mark(summary['pass'])}** -- "
        f"{summary['ok']} / {len(ids)} videos stored, failed (length) {len(failed)} "
        f"{sorted(failed.items())[:10]}\n\nPrinted only: min cos vs v2 `Nexar_s1_squash` "
        f"{summary['g1_min_cos']}; videos with a row < {constants.V5_G0_MIN_COS}: "
        f"{summary['videos_min_cos_below_0.9999']}\n"
    )
    name = f"g1_extract{'_tile' if tile else ''}_s{args.stride}"
    return _write(args.out_dir, name, summary, md)


def run_extract_clip(args: argparse.Namespace) -> dict[str, Any]:
    """R7 Phase 1 (PREREG_V5 §4): CLIP 14 x 14 x 512 grids of rows ``s1[::stride]``, G1."""
    if args.tile:
        raise ValueError("--tile and --clip-grid build different caches; pass one")
    if args.clip_s1_dir is None:
        raise ValueError("--clip-grid needs --clip-s1-dir (Nexar_s1_ncc, G1)")
    census = json.loads(args.census.read_text(encoding="utf-8"))
    ids = sorted(read_ids_file(args.ids_file))
    paths = video_paths(args.root, census, set(ids))
    rows = f"s1[::{args.stride}] of the decoded mp4 (Nexar N2 frame order)"
    check_or_write_manifest(
        args.grid_dir, clip_grid_manifest(args.stride, rows, CLIP_TRANSFORM), force=False
    )
    report: dict[str, Any] = (
        json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    )
    todo = [
        v
        for v in ids
        if not (report.get(v, {}).get("reason") == OK and grid_complete(args.grid_dir / f"{v}.npy"))
    ]
    LOGGER.info(
        "CLIP grids: %d / %d videos done, %d to go", len(ids) - len(todo), len(ids), len(todo)
    )
    device = resolve_device(args.device)  # TF32 left at torch's defaults, as N2 built Nexar_s1_ncc
    model = load_pretrained_encoder(device) if todo else None
    progress = Progress(len(todo))
    for video_id, resized in decoded_ahead(paths, todo):
        reference = np.load(args.clip_s1_dir / f"{video_id}.npy", mmap_mode="r")
        entry, grid16 = clip_grid_entry(
            resized, reference, args.stride, model, device, args.batch_size
        )
        if grid16 is not None:
            save_array(args.grid_dir / f"{video_id}.npy", grid16)
        report[video_id] = entry
        write_json_atomic(args.report, report)
        LOGGER.info("%s: %s, G1 min cos %s -- %s", video_id, entry["reason"],
                    entry.get("g1_min_cos"), progress.step())
    write_json_atomic(args.report, report)
    failed = {v: e["reason"] for v, e in report.items() if v in ids and e["reason"] != OK}
    mins = [report[v]["g1_min_cos"] for v in ids if report.get(v, {}).get("g1_min_cos") is not None]
    gaps = [report[v]["g1_max_abs"] for v in ids if report.get(v, {}).get("g1_max_abs") is not None]
    summary = {
        "rule": "PREREG_V5 §4 R7 G1 (length; projected CLS; image_embeds vs Nexar_s1_ncc)",
        "grid_dir": str(args.grid_dir),
        "stride": args.stride,
        "videos": len(ids),
        "ok": sum(1 for v in ids if report.get(v, {}).get("reason") == OK),
        "rows": sum(report[v].get("rows", 0) for v in ids if report.get(v, {}).get("reason") == OK),
        "failed": failed,
        "g1_min_cos": min(mins, default=None),
        "g1_max_abs": max(gaps, default=None),
        "min_cos_bar": constants.V5_R7_G1_MIN_COS,
        "pass": not failed and all(v in report for v in ids),
    }
    md = (
        f"# R7 G1 CLIP grid extraction (stride {args.stride})\n\n**{_mark(summary['pass'])}** -- "
        f"{summary['ok']} / {len(ids)} videos, {summary['rows']} rows stored; failed "
        f"{len(failed)} {sorted(failed.items())[:10]}\n\nmin cos vs `Nexar_s1_ncc` "
        f"{summary['g1_min_cos']} (bar {constants.V5_R7_G1_MIN_COS}); max |CLS - image_embeds| "
        f"{summary['g1_max_abs']} (bar {constants.V5_G1_MAX_ABS})\n"
    )
    return _write(args.out_dir, f"g1_extract_cgrid_s{args.stride}", summary, md)


# --------------------------------------------------------------------------- head (D1)


def run_head(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_input_manifest(args.stats_dir)
    if manifest is None or (manifest.get("crn"), manifest.get("motion")) != (
        constants.V2_OFF,
        constants.V5_ENCODER,
    ):
        raise ValueError(f"{args.stats_dir} is not Nexar A2's bake (crn none, motion V2-S)")
    stats = load_stats(args.stats_dir)
    assert stats.m_u is not None and stats.sigma_u is not None  # nosec B101 - motion arm
    encoder = load_pretrained(args.encoder, torch.device("cpu"), args.weights)
    path = save_head(
        args.grid_dir / constants.V5_HEAD_FILENAME,
        encoder.fc_norm.weight.detach().numpy(),
        encoder.fc_norm.bias.detach().numpy(),
        float(encoder.fc_norm.eps),
        stats.c,
        stats.m_u,
        stats.sigma_u,
    )
    payload = {
        "head": str(path),
        "stats_dir": str(args.stats_dir),
        "train_ids_sha1": manifest.get("train_ids_sha1"),
        "c": stats.c,
        "encoder": args.encoder,
    }
    return _write(args.out_dir, "head", payload, f"# v5 head\n\n`{path}` from `{args.stats_dir}`\n")


# --------------------------------------------------------------------------- S1-G2


ALL_ROWS: dict[str, dict[str, Any]] = {
    **constants.V5_ROWS,
    **constants.V5_TILE_ROWS,
    **constants.V5_R7_ROWS,  # R7 (§4): A2's recipe, the CLIP grid read instead of u
}


def row_config(row: str, overrides: list[str] | None = None) -> Config:
    """v2 Nexar A2's architecture with ``row``'s v5 section (S2-3; v5.1 rows, §3)."""
    extra = [f"v5.{k}={v}" for k, v in ALL_ROWS[row].items()]
    return load_config(None, [*A2_OVERRIDES, *extra, *(overrides or [])])


def g2_synthetic(seed: int = constants.V5_BOOTSTRAP_SEED) -> dict[str, Any]:
    """S1-G2(a): query readouts at init == tile mean; DVS splices keep per-item ΔG."""
    gen = torch.Generator().manual_seed(seed)
    grid = torch.randn(2, 5, constants.V5_GRID_TILES, constants.V5_GRID_DIM, generator=gen)
    change = torch.randn(grid.shape, generator=gen)
    mean = grid.mean(-2)
    worst: dict[str, float] = {}
    for queries in (1, constants.V5_QUERIES):
        for keys in constants.V5_KEYS:
            readout = GridReadout(constants.V5_READOUT_MQR, queries, keys)
            r, _, _ = readout.read(grid, change if keys == constants.V5_KEYS_CHANGE else None)
            worst[f"M{queries}_{keys}"] = float((r - mean).abs().max())
    rng = np.random.default_rng(seed)
    lag, items = 3, []
    for length in (7, 9):
        g = rng.standard_normal((length, constants.V5_GRID_TILES, constants.V5_GRID_DIM))
        x = rng.standard_normal((length, constants.HIDDEN_DIM + constants.V5_GRID_DIM))
        items.append((g, torch.from_numpy(v5_rows(x, g, lag))))
    spliced = compose_sequence("a", items[0][1], None, True, [("b", items[1][1], None)], 0)
    start = constants.HIDDEN_DIM + constants.V5_GRID_DIM + constants.V5_GRID_WIDTH
    per_item = np.concatenate([grid_change(g.reshape(len(g), -1), lag) for g, _ in items])
    splice_ok = bool(np.array_equal(spliced.features[:, start:].numpy(), per_item))
    return {
        "readout_vs_mean_max_abs": worst,
        "readout_pass": max(worst.values()) <= constants.V5_G2_UNIT_MAX_ABS,
        "splice_keeps_item_change": splice_ok,
    }


def g2_real(args: argparse.Namespace) -> dict[str, Any]:
    """S1-G2(b): every row's motion term at init vs R1's on the same real windows."""
    windows = load_windows(args.data_dir)
    if windows is None:
        raise ValueError(f"{args.data_dir} has no windows.json (Nexar `window` corpus)")
    slicer = FeatureSlicer(windows)
    pick = sorted(windows)
    random.Random(constants.V5_BOOTSTRAP_SEED).shuffle(pick)  # nosec B311 - window sampling
    pick = sorted(pick[: constants.V5_G2_WINDOWS])
    geometry = read_grid_geometry(args.grid_dir)
    lag = change_lag(geometry.stride, geometry.frame_step)

    def motion(row: str, window_id: str) -> torch.Tensor:
        cfg = row_config(row)
        rows = v5_rows(
            slicer.load(args.clip_dir, window_id),
            slicer.load(args.grid_dir, window_id, mmap=True),
            lag if cfg.v5.keys == constants.V5_KEYS_CHANGE else None,
        )
        with torch.no_grad():
            out, _ = models[row]._grid_rows(torch.from_numpy(rows)[None], {})  # V5_ROWS: no W_c
        return out[0, :, constants.HIDDEN_DIM :]

    models: dict[str, KATVAD] = {}
    for row in constants.V5_ROWS:
        models[row] = KATVAD.from_config(row_config(row)).eval()
        readout = models[row].grid_readout
        assert readout is not None  # nosec B101 - every row has a readout
        readout.load_head(args.grid_dir / constants.V5_HEAD_FILENAME)
    reference = {w: motion("R1", w) for w in pick}
    worst = {
        row: max(float((motion(row, w) - reference[w]).abs().max()) for w in pick)
        for row in constants.V5_ROWS
    }
    return {
        "windows": len(pick),
        "max_abs": worst,
        "pass": max(worst.values()) <= constants.V5_G2_MAX_ABS,
    }


def run_init_check(args: argparse.Namespace) -> dict[str, Any]:
    synthetic = g2_synthetic()
    real = g2_real(args)
    ok = synthetic["readout_pass"] and synthetic["splice_keeps_item_change"] and real["pass"]
    payload = {"rule": "S1-G2", "a": synthetic, "b": real, "pass": ok}
    md = (
        f"# S1-G2 same start: **{_mark(ok)}**\n\n"
        f"- (a) readout vs tile mean max |Δ| {synthetic['readout_vs_mean_max_abs']} "
        f"(≤ {constants.V5_G2_UNIT_MAX_ABS}); DVS splice keeps per-item ΔG: "
        f"{synthetic['splice_keeps_item_change']}\n"
        f"- (b) {real['windows']} Nexar windows, motion term vs R1's max |Δ| {real['max_abs']} "
        f"(≤ {constants.V5_G2_MAX_ABS})\n"
    )
    return _write(args.out_dir, "g2_init", payload, md)


# --------------------------------------------------------------------------- S1-G4


def _last_metrics(path: Path) -> dict[str, Any] | None:
    lines = (
        [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if path.exists()
        else []
    )
    return json.loads(lines[-1]) if lines else None


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out |= _flat(v, f"{prefix}{k}.")
        else:
            out[f"{prefix}{k}"] = v
    return out


def provenance(run_dir: Path, row: str, a2_dir: Path, steps: int) -> dict[str, Any]:
    """J10 + config diff of one run against v2 Nexar A2 of the same seed."""
    payload = torch.load(run_dir / CKPT_FILENAME, map_location="cpu", weights_only=False)  # nosec B614
    ckpt_step = int(payload.get("global_step", -1))
    del payload
    last = _last_metrics(run_dir / METRICS_FILENAME)
    metrics_step = None if last is None else int(last["global_step"])
    mine = _flat(yaml.safe_load((run_dir / CONFIG_FILENAME).read_text(encoding="utf-8")))
    theirs = _flat(yaml.safe_load((a2_dir / CONFIG_FILENAME).read_text(encoding="utf-8")))
    diff = sorted(
        k
        for k in set(mine) | set(theirs)
        if not k.startswith("v5.") and mine.get(k) != theirs.get(k)
    )
    want_v5 = {f"v5.{k}": v for k, v in ALL_ROWS[row].items()}
    bad_v5 = sorted(k for k, v in want_v5.items() if mine.get(k) != v)
    ok = ckpt_step == metrics_step == steps and not diff and not bad_v5
    return {
        "ckpt_step": ckpt_step,
        "metrics_step": metrics_step,
        "want_steps": steps,
        "config_diff_vs_a2": diff,
        "v5_mismatch": bad_v5,
        "pass": ok,
    }


def run_provenance(args: argparse.Namespace) -> dict[str, Any]:
    runs = {
        f"{row}/s{seed}": provenance(
            args.root / row / f"s{seed}", row, args.a2_root / f"s{seed}", args.steps
        )
        for row in args.rows
        for seed in args.seeds
    }
    ok = all(r["pass"] for r in runs.values())
    lines = [
        f"| {k} | {r['ckpt_step']} | {r['metrics_step']} | {r['config_diff_vs_a2'] or '—'} | "
        f"{r['v5_mismatch'] or '—'} | {_mark(r['pass'])} |"
        for k, r in runs.items()
    ]
    md = (
        "# S1-G4 provenance: **"
        + _mark(ok)
        + "**\n\n| run | ckpt step | metrics step | config Δ vs A2 "
        "| v5 mismatch | |\n|---|---|---|---|---|---|\n" + "\n".join(lines) + "\n"
    )
    return _write(args.out_dir, "g4_provenance", {"rule": "S1-G4", "runs": runs, "pass": ok}, md)


# --------------------------------------------------------------------------- S1-G6


def collapse(run_dir: Path, require_moved: bool = False) -> dict[str, Any]:
    """S1-G6 on a run's last step; with ``require_moved`` also §2.4 B13 G0 (a query moved)."""
    last = _last_metrics(run_dir / METRICS_FILENAME)
    if last is None:
        raise FileNotFoundError(f"no metrics in {run_dir}")
    ent = {k: float(v) for k, v in sorted(last.items()) if k.startswith("v5_entropy_q")}
    finite = {k: math.isfinite(float(last[k])) for k in ("mil", "smil") if k in last}
    log_tiles = math.log(constants.V5_GRID_TILES)
    bound = constants.V5_G6_ENTROPY_FRACTION * log_tiles
    moved_bound = constants.V5_B13_MOVED_FRACTION * log_tiles
    moved = any(e < moved_bound for e in ent.values())
    return {
        "step": last["global_step"],
        "finite": finite,
        "entropy": ent,
        "entropy_bound": bound,
        "did_not_leave_mean": bool(ent) and all(e >= bound for e in ent.values()),
        "moved_bound": moved_bound,
        "moved": moved,
        "logit_std": {
            k: float(v) for k, v in sorted(last.items()) if k.startswith("v5_logit_std_q")
        },
        "pass": all(finite.values()) and (moved or not require_moved),
    }


def run_collapse(args: argparse.Namespace) -> dict[str, Any]:
    runs = {
        f"{row}/s{seed}": collapse(args.root / row / f"s{seed}", args.require_moved)
        for row in args.rows
        for seed in args.seeds
    }
    ok = all(r["pass"] for r in runs.values())
    lines = [
        f"| {k} | {r['finite']} | "
        + ", ".join(f"{e:.3f}" for e in r["entropy"].values())
        + f" | {'yes' if r['did_not_leave_mean'] else 'no'} | {'yes' if r['moved'] else 'no'} | "
        + (", ".join(f"{e:.3f}" for e in r["logit_std"].values()) or "—")
        + " |"
        for k, r in runs.items()
    ]
    log_tiles = math.log(constants.V5_GRID_TILES)
    rule = "S1-G6 + §2.4 B13 G0" if args.require_moved else "S1-G6"
    stay, move = constants.V5_G6_ENTROPY_FRACTION, constants.V5_B13_MOVED_FRACTION
    md = (
        f"# {rule}: **{_mark(ok)}**\n\nEntropy bound {stay:.3f} log 49 = "
        f"{stay * log_tiles:.4f}; moved = some query below "
        f"{move:.2f} log 49 = {move * log_tiles:.4f}"
        + (" (**required**, B13 G0)" if args.require_moved else " (printed)")
        + ".\n\n| run | finite | entropy per query | did not leave the mean | moved | "
        "logit std per query |\n|---|---|---|---|---|---|\n"
        + "\n".join(lines)
        + "\n"
    )
    payload = {"rule": rule, "require_moved": args.require_moved, "runs": runs, "pass": ok}
    return _write(args.out_dir, "g6_collapse", payload, md)


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="Nexar V7 / 4x7x7 / CLIP 14x14 grids + G1 (GPU)")
    ex.add_argument(
        "--root", type=Path, required=True, help="Nexar raw root (train/{positive,negative})"
    )
    ex.add_argument("--census", type=Path, required=True)
    ex.add_argument("--ids-file", type=Path, required=True)
    ex.add_argument(
        "--video-s1-dir", type=Path, default=None,
        help="cache/video/<V2-S>/Nexar_s1_squash (S1-G1); not read with --clip-grid",
    )
    ex.add_argument("--clip-s1-dir", type=Path, default=None,
                    help="cache/clip/Nexar_s1_ncc (R7 G1); --clip-grid only")
    ex.add_argument("--grid-dir", type=Path, required=True)
    ex.add_argument("--stride", type=int, required=True)
    ex.add_argument("--report", type=Path, required=True)
    ex.add_argument("--encoder", default=constants.V5_ENCODER)
    ex.add_argument("--weights", type=Path, default=None)
    ex.add_argument("--batch-size", type=int, default=32)
    ex.add_argument("--device", default="auto")
    ex.add_argument("--tile", action="store_true",
                    help="v5.1 (PREREG_V5 §3): centred 4x7x7 grids; G-c on the shifted v2 rows")
    ex.add_argument("--clip-grid", action="store_true",
                    help="R7 (PREREG_V5 §4): CLIP 14x14x512 grids, unpooled; G1 vs Nexar_s1_ncc")

    hd = sub.add_parser("head", help="fc_norm + A2 statistics -> v5_head.npz")
    hd.add_argument("--stats-dir", type=Path, required=True, help="cache/v2/A2_S/Nexar_s8")
    hd.add_argument("--grid-dir", type=Path, required=True, help="the training grid cache")
    hd.add_argument("--encoder", default=constants.V5_ENCODER)
    hd.add_argument("--weights", type=Path, default=None)

    g2 = sub.add_parser("init-check", help="S1-G2")
    g2.add_argument("--data-dir", type=Path, required=True, help="data/Nexar_window")
    g2.add_argument("--clip-dir", type=Path, required=True, help="cache/v2/A2_S/Nexar_s8")
    g2.add_argument("--grid-dir", type=Path, required=True, help="training grid cache (+ head)")

    g4 = sub.add_parser("provenance", help="S1-G4")
    g4.add_argument("--root", type=Path, required=True, help="v5 runs: <row>/s<seed>/")
    g4.add_argument("--a2-root", type=Path, required=True, help="v2 Nexar A2 runs: s<seed>/")
    g4.add_argument("--rows", nargs="+", choices=list(ALL_ROWS), required=True)
    g4.add_argument("--seeds", nargs="+", type=int, default=list(constants.V5_RUN_SEEDS))
    g4.add_argument("--steps", type=int, required=True, help="the build plan's steps (1,749)")

    g6 = sub.add_parser("collapse", help="S1-G6")
    g6.add_argument("--root", type=Path, required=True)
    g6.add_argument("--rows", nargs="+", choices=list(ALL_ROWS), required=True)
    g6.add_argument("--seeds", nargs="+", type=int, default=list(constants.V5_RUN_SEEDS))
    g6.add_argument("--require-moved", action="store_true",
                    help="§2.4 B13 G0: fail a run none of whose queries left the mean")

    for p in (ex, hd, g2, g4, g6):
        p.add_argument("--out-dir", type=Path, required=True)
    return parser


COMMANDS = {
    "extract": run_extract,
    "head": run_head,
    "init-check": run_init_check,
    "provenance": run_provenance,
    "collapse": run_collapse,
}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
