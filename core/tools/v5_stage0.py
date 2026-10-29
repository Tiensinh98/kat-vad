"""v5 Stage 0 -- do VideoMAE V2-S tokens carry local anomaly evidence? (inference only).

Rules: ``core/docs/v5/PREREG_V5.md`` §1 (S0-1 ... S0-12), signed before any grid existed.

``extract`` (Colab, GPU): one stream over the CAP-DATA tars (the path of ``core.tools.dota_cap
extract``). For every finalized DoTA-CAP clip it stores the 7 x 7 V2-S token grid at protocol B's
rows (S0-3/S0-4); for DoTA-CAP-dev clips it also stores the change maps of ``V7``, ``V14`` and the
CLIP patch grid (S0-5) plus the clip's ``DoTA_s1_ncc`` rows for D0. Gates G0-G2 (S0-10) are checked
per clip and recorded in the report; a failing clip stores nothing.

``readout`` (local, CPU): D1 / D3 / D2 against the DoTA box annotations, the S0-11 decision table.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from core import constants
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_splits import dota_group, load_split
from core.data.v5_grids import centred_reference
from core.data.video_io import list_frame_images, read_images
from core.device import resolve_device
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.models.videomae_v2 import VideoMAEv2Encoder, load_pretrained
from core.tools import extract_clip_features, extract_video_features
from core.tools.dota_cap import file_sha256, frame_cosines, rebuild_folder
from core.tools.feature_cache import read_ids_file, save_array
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.rate_matched_eval import native_labels, to_native
from core.tools.stream_frames_clip import stream_video_batches

LOGGER = logging.getLogger(__name__)

OK = "ok"
FARM_DIR = "farm"
MAP_SUFFIX = ".npz"
READOUT_JSON = "stage0_readout.json"
READOUT_MD = "stage0_readout.md"
REPS = ("V7", "V14", "C14")  # S0-4; C14 is printed only (S0-12)
GATED = ("V7", "V14")
ASPECT = 16 / 9
BIN_NAMES = ("<0.5%", "0.5-2%", "2-8%", ">8%")  # S0-12, edges V5_BOX_SHARE_BINS
RULE = "core/docs/v5/PREREG_V5.md §1 + Amendments 1-2 (§1.1-§1.2)"
LAG_JSON = "stage0_lag.json"
LAG_MD = "stage0_lag.md"


# --------------------------------------------------------------------------- grids + change


def token_grids(tokens: Tensor, fine: int = constants.V5_GRID_FINE) -> tuple[Tensor, Tensor]:
    """``(B, T' * f * f, D)`` tokens -> ``V14 (B, f, f, D)``, ``V7 (B, f/2, f/2, D)``.

    ``V14`` averages the temporal slots, ``V7`` averages 2 x 2 tiles of ``V14`` (S0-4). Both are
    equal-weight means, so each grid's mean is the mean of all tokens.
    """
    batch, count, dim = tokens.shape
    slots = count // (fine * fine)
    if slots * fine * fine != count:
        raise ValueError(f"{count} tokens are not T' x {fine} x {fine}")
    fine_grid = tokens.reshape(batch, slots, fine, fine, dim).mean(1)
    half = fine // 2
    coarse = fine_grid.reshape(batch, half, 2, half, 2, dim).mean((2, 4))
    return fine_grid, coarse


def tile_grid(tokens: Tensor, fine: int = constants.V5_GRID_FINE) -> Tensor:
    """``(B, 8 * f * f, D)`` tokens -> ``(B, 4, f/2, f/2, D)`` (v5.1, PREREG_V5 §3).

    Consecutive pairs of the 8 temporal slots and 2 x 2 tiles are averaged; both are equal-weight
    means, so the mean over all 196 cells is the mean of all tokens (the same starting function).
    """
    batch, count, dim = tokens.shape
    slots = count // (fine * fine)
    if slots * fine * fine != count or slots != constants.V5_SLOTS_TILE * constants.V5_SLOT_POOL:
        raise ValueError(f"{count} tokens are not 8 x {fine} x {fine}")
    half = fine // 2
    grid = tokens.reshape(
        batch, constants.V5_SLOTS_TILE, constants.V5_SLOT_POOL, half, 2, half, 2, dim
    )
    out: Tensor = grid.mean((2, 4, 6))
    return out


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    num = (a64 * b64).sum(-1)
    den = np.linalg.norm(a64, axis=-1) * np.linalg.norm(b64, axis=-1)
    out: np.ndarray = num / np.maximum(den, np.finfo(np.float64).tiny)
    return out


def change_map(grid: np.ndarray, lag: int = 1) -> np.ndarray:
    """``(K, g, g, D)`` -> ``(K, g*g)``: ``1 - cos`` of each tile to the step ``lag`` rows back.

    Rows ``k < lag`` have no earlier step and are NaN (S0-5; ``lag`` > 1 is check L, §1.3).
    """
    flat = grid.reshape(len(grid), -1, grid.shape[-1])
    out = np.full(flat.shape[:2], np.nan)
    if len(flat) > lag:
        out[lag:] = 1.0 - _cos_rows(flat[lag:], flat[:-lag])
    return out


def global_change(vectors: np.ndarray, lag: int = 1) -> np.ndarray:
    """``(K, D)`` -> ``(K,)``: ``1 - cos`` to the step ``lag`` rows back; rows ``k < lag`` NaN."""
    out = np.full(len(vectors), np.nan)
    if len(vectors) > lag:
        out[lag:] = 1.0 - _cos_rows(vectors[lag:], vectors[:-lag])
    return out


def _fill(score: np.ndarray, first: int, last: int) -> np.ndarray:
    """Undefined edge rows take the nearest defined row's value (S0-5, D2; §1.3 L-3)."""
    out = score.copy()
    out[:first] = out[first]
    out[last + 1 :] = out[last]
    return out


def shift_rows(score: np.ndarray, shift: int) -> np.ndarray:
    """Row ``k`` takes row ``k + shift``; the last ``shift`` rows are undefined (§1.3 L-3)."""
    out = np.full(score.shape, np.nan)
    out[: len(score) - shift] = score[shift:]
    return out


def tile_weights(
    boxes: list[list[float]],
    grid: int,
    size: int = constants.CROP_SIZE,
    frame: tuple[int, int] = constants.V5_BOX_FRAME_SIZE,
) -> np.ndarray:
    """Covered fraction of each tile by the union of ``boxes`` (x1, y1, x2, y2 in ``frame`` px)."""
    mask = np.zeros((size, size), dtype=bool)
    sx, sy = size / frame[0], size / frame[1]
    for x1, y1, x2, y2 in boxes:
        c0, c1 = max(0, math.floor(x1 * sx)), min(size, math.ceil(x2 * sx))
        r0, r1 = max(0, math.floor(y1 * sy)), min(size, math.ceil(y2 * sy))
        if c1 > c0 and r1 > r0:
            mask[r0:r1, c0:c1] = True
    tile = size // grid
    out: np.ndarray = mask.reshape(grid, tile, grid, tile).mean((1, 3)).reshape(-1)
    return out


# --------------------------------------------------------------------------- encoders


def _clip_tokens(
    paths: list[Path],
    clips: list[list[int]],
    encoder: VideoMAEv2Encoder,
    device: torch.device,
    batch_size: int,
) -> Iterator[Tensor]:
    """Last-layer tokens ``(B, 1568, D)`` of each batch of clips (frame indices into ``paths``)."""
    frames = extract_video_features.decode_frames(paths, sorted({i for c in clips for i in c}))
    for start in range(0, len(clips), batch_size):
        batch = torch.stack(
            [
                torch.stack([frames[i] for i in clip], dim=1)
                for clip in clips[start : start + batch_size]
            ]
        ).to(device)
        yield encoder.forward_tokens(batch).float()


@torch.no_grad()
def video_grids(
    paths: list[Path],
    steps: list[int],
    encoder: VideoMAEv2Encoder,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``V14``, ``V7`` and ``fc_norm(mean tokens)`` for the causal window ending at each step."""
    clips = [
        extract_video_features.causal_clip_indices(t, step=constants.DOTA_VIDEOMAE_FRAME_STEP)
        for t in steps
    ]
    fine, coarse, pooled = [], [], []
    for tokens in _clip_tokens(paths, clips, encoder, device, batch_size):
        v14, v7 = token_grids(tokens)
        fine.append(v14.cpu())
        coarse.append(v7.cpu())
        pooled.append(encoder.fc_norm(tokens.mean(1)).float().cpu())
    return (torch.cat(fine).numpy(), torch.cat(coarse).numpy(), torch.cat(pooled).numpy())


@torch.no_grad()
def video_tile_grids(
    paths: list[Path],
    steps: list[int],
    encoder: VideoMAEv2Encoder,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """v5.1: ``4 x 7 x 7`` grids and ``fc_norm(mean tokens)`` of the clip centred on each step."""
    clips = [
        extract_video_features.centred_clip_indices(
            t, len(paths), step=constants.DOTA_VIDEOMAE_FRAME_STEP
        )
        for t in steps
    ]
    tiles, pooled = [], []
    for tokens in _clip_tokens(paths, clips, encoder, device, batch_size):
        tiles.append(tile_grid(tokens).cpu())
        pooled.append(encoder.fc_norm(tokens.mean(1)).float().cpu())
    return torch.cat(tiles).numpy(), torch.cat(pooled).numpy()


@torch.no_grad()
def clip_patches(
    paths: list[Path],
    steps: list[int],
    model: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """CLIP ``image_embeds (K, P)``, patch grid ``(K, g, g, P)``, G1's max |CLS - image_embeds|."""
    embeds, grids, worst = [], [], 0.0
    vision, project = model.vision_model, model.visual_projection
    for start in range(0, len(steps), batch_size):
        pixels = extract_clip_features.preprocess_frames(
            read_images([paths[t] for t in steps[start : start + batch_size]]), center_crop=False
        ).to(device)
        out = model(pixel_values=pixels)
        hidden = out.last_hidden_state
        cls = project(vision.post_layernorm(hidden[:, 0]))
        worst = max(worst, float((cls - out.image_embeds).abs().max()))
        patches = project(vision.post_layernorm(hidden[:, 1:]))
        side = math.isqrt(patches.shape[1])
        embeds.append(out.image_embeds.float().cpu())
        grids.append(patches.reshape(len(patches), side, side, -1).float().cpu())
    return torch.cat(embeds).numpy(), torch.cat(grids).numpy(), worst


# --------------------------------------------------------------------------- extract


def passes_row_gate(cos: np.ndarray) -> bool:
    """Amendment 2 (§1.2) G1-pixel on the Stage 0 rows: every row keeps v2's per-frame minimum.

    v2's clip mean (``passes_frame_gate``) is a verdict over all frames and already holds for every
    finalized id (:func:`run_extract`); re-taking it on a third of the rows fails by sampling.
    """
    return bool(cos.min() >= constants.DOTA_CAP_MIN_COS)


def grid_out_dir(encoder: str, tile: bool = False) -> Path:
    tag = (
        f"{constants.DOTA_CAP_DATASET}_s{constants.V5_TILE_DOTA_STRIDE}_{constants.V5_TILE_GRID_TAG}"
        if tile
        else constants.V5_GRID_DATASET_TAG
    )
    return constants.CACHE_ROOT / constants.V5_GRID_CACHE_DIRNAME / encoder / tag


def stage0_dir() -> Path:
    return constants.OUTPUT_ROOT / constants.V5_OUTPUT_DIRNAME / constants.V5_STAGE0_DIRNAME


def frame_aspect(path: Path) -> float:
    image = read_images([path])[0]
    return float(image.shape[1]) / float(image.shape[0])


@dataclass
class Stage0Extractor:
    """Per-batch callback of :func:`stream_video_batches` (one CAP batch on VM disk)."""

    by_cap: dict[str, list[dict[str, Any]]]
    dev: set[str]
    encoder: VideoMAEv2Encoder
    clip_model: Any
    v2_cache: Path
    dota_clip_dir: Path | None
    grid_dir: Path
    maps_dir: Path
    farm: Path
    device: torch.device
    subdir: str
    batch_size: int
    report: dict[str, dict[str, Any]]
    save_report: Callable[[], None]
    stride: int = constants.V5_STAGE0_STRIDE
    tile: bool = False  # v5.1: centred 4 x 7 x 7 grids, G0 on the shifted rows, no dev maps
    done: list[str] = field(default_factory=list)

    def __call__(self, ready_dir: Path) -> None:
        for folder in sorted(p for p in ready_dir.iterdir() if p.is_dir()):
            cap_paths = list_frame_images(folder, self.subdir)
            for clip in self.by_cap.get(folder.name, []):
                self.report[clip["dota_id"]] = self.one_clip(clip, cap_paths)
        self.save_report()

    def one_clip(self, clip: dict[str, Any], cap_paths: list[Path]) -> dict[str, Any]:
        dota_id = clip["dota_id"]
        entry: dict[str, Any] = {"cap_id": clip["cap_id"], "dev": dota_id in self.dev}
        if len(cap_paths) != clip["cap_frames"]:
            return {**entry, "reason": "cap_frames_mismatch"}
        farm = rebuild_folder(self.farm / dota_id, cap_paths, clip["frame_map"])
        try:
            return {**entry, **self._encode(dota_id, list_frame_images(farm))}
        finally:
            shutil.rmtree(farm, ignore_errors=True)

    def _encode(self, dota_id: str, paths: list[Path]) -> dict[str, Any]:
        entry: dict[str, Any] = {"aspect": frame_aspect(paths[0])}
        if abs(entry["aspect"] / ASPECT - 1.0) > constants.V5_ASPECT_TOLERANCE:
            return {**entry, "reason": "g2_aspect"}
        steps = extract_video_features.step_end_frames(len(paths), self.stride)
        ref = np.load(self.v2_cache / f"{dota_id}.npy")
        if len(ref) != len(paths):
            return {**entry, "reason": "g0_length"}
        if self.tile:
            return {**entry, **self._encode_tile(dota_id, paths, steps, ref)}
        v14, v7, pooled = video_grids(paths, steps, self.encoder, self.device, self.batch_size)
        entry["g0_min_cos"] = float(_cos_rows(pooled, ref[steps]).min())
        if entry["g0_min_cos"] < constants.V5_G0_MIN_COS:
            return {**entry, "reason": "g0_cos"}
        dev = dota_id in self.dev
        if dev:
            embeds, cgrid, worst = clip_patches(
                paths, steps, self.clip_model, self.device, self.batch_size
            )
            assert self.dota_clip_dir is not None  # nosec B101 - run_extract requires it
            s1 = np.load(self.dota_clip_dir / f"{dota_id}.npy")
            cos = frame_cosines(embeds, s1[steps])
            entry |= {"g1_max_abs": worst, "pixel_min_cos": float(cos.min())}
            if worst > constants.V5_G1_MAX_ABS:
                return {**entry, "reason": "g1_cls"}
            entry["pixel_mean_cos"] = float(cos.mean())
            if not passes_row_gate(cos):
                return {**entry, "reason": "g1_pixel"}
            tmp = self.maps_dir / f"{dota_id}.partial{MAP_SUFFIX}"
            np.savez_compressed(
                tmp,
                steps=np.asarray(steps),
                num_frames=len(paths),
                d7=change_map(v7).astype(np.float32),
                d14=change_map(v14).astype(np.float32),
                dc14=change_map(cgrid).astype(np.float32),
                g_v=global_change(v14.reshape(len(v14), -1, v14.shape[-1]).mean(1)),
                g_c=global_change(embeds),
                clip_s1=s1.astype(np.float16),
            )
            tmp.replace(self.maps_dir / f"{dota_id}{MAP_SUFFIX}")
        save_array(self.grid_dir / f"{dota_id}.npy", v7.astype(np.float16))
        LOGGER.info(
            "%s: %d steps, G0 min cos %.6f%s",
            dota_id,
            len(steps),
            entry["g0_min_cos"],
            " (dev maps)" if dev else "",
        )
        return {**entry, "steps": len(steps), "reason": OK}

    def _encode_tile(
        self, dota_id: str, paths: list[Path], steps: list[int], ref: np.ndarray
    ) -> dict[str, Any]:
        """v5.1 gate G-c: ``fc_norm`` of the cell mean == the v2 row of the same frames."""
        tiles, pooled = video_tile_grids(paths, steps, self.encoder, self.device, self.batch_size)
        keep, index = centred_reference(
            np.asarray(steps), constants.DOTA_VIDEOMAE_FRAME_STEP, len(paths)
        )
        entry: dict[str, Any] = {"g0_rows": int(keep.sum())}
        if keep.any():
            entry["g0_min_cos"] = float(_cos_rows(pooled[keep], ref[index]).min())
            if entry["g0_min_cos"] < constants.V5_TILE_G_C_MIN_COS:
                return {**entry, "reason": "g0_cos"}
        save_array(self.grid_dir / f"{dota_id}.npy", tiles.astype(np.float16))
        LOGGER.info("%s: %d steps (centred 4x7x7), G-c rows %d", dota_id, len(steps), keep.sum())
        return {**entry, "steps": len(steps), "reason": OK}


def grid_manifest(encoder: str, alignment_sha256: str, tile: bool = False) -> dict[str, Any]:
    if tile:
        return {
            "encoder": encoder,
            "weights_sha256": constants.VIDEOMAE_WEIGHTS_SHA256[encoder],
            "stride": constants.V5_TILE_DOTA_STRIDE,
            "frame_step": constants.DOTA_VIDEOMAE_FRAME_STEP,
            "slots": constants.V5_SLOTS_TILE,
            "window_kind": constants.V5_WINDOW_CENTRED,
            "rows": f"s1[::{constants.V5_TILE_DOTA_STRIDE}] (protocol B)",
            "window": "centred 16 frames t-8..t+7, edge frames repeated",
            "grid": "4x7x7: pairs of the 8 slots, then 2x2, last layer before fc_norm",
            "dtype": "float16",
            "transform": extract_video_features.TRANSFORM_NAME,
            "alignment_sha256": alignment_sha256,
            "rule": "core/docs/v5/PREREG_V5.md §3",
        }
    return {
        "encoder": encoder,
        "weights_sha256": constants.VIDEOMAE_WEIGHTS_SHA256[encoder],
        "rows": f"s1[::{constants.V5_STAGE0_STRIDE}] (protocol B)",
        "window": "causal 16 frames, DOTA_VIDEOMAE_FRAME_STEP",
        "grid": f"{constants.V5_GRID_COARSE}x{constants.V5_GRID_COARSE}, mean over T' then 2x2, "
        "last layer before fc_norm",
        "dtype": "float16",
        "transform": extract_video_features.TRANSFORM_NAME,
        "alignment_sha256": alignment_sha256,
        "rule": "core/docs/v5/PREREG_V5.md S0-4",
    }


def grid_complete(path: Path) -> bool:
    """A readable ``(T, [4,] 7, 7, D)`` grid (``feature_cache.is_complete`` only accepts 2-D)."""
    if not path.exists():
        return False
    try:
        array = np.load(path, mmap_mode="r")
    except (OSError, ValueError, EOFError):
        return False
    return bool(array.ndim in (4, 5) and array.shape[0] > 0)


def todo(
    alignment: dict[str, Any],
    ids: list[str],
    grid_dir: Path,
    maps_dir: Path,
    dev: set[str],
    report: dict[str, dict[str, Any]],
    retry: frozenset[str] = frozenset(),
    maps: bool = True,
) -> list[dict[str, Any]]:
    """Clips still to extract; a clip that failed for a reason in ``retry`` is redone."""
    out = []
    for dota_id in ids:
        clip = alignment["clips"][dota_id]
        done = report.get(dota_id)
        if done is not None and done["reason"] in retry:
            done = None
        if done is not None and (
            done["reason"] != OK
            or (
                grid_complete(grid_dir / f"{dota_id}.npy")
                and (
                    not maps or dota_id not in dev or (maps_dir / f"{dota_id}{MAP_SUFFIX}").exists()
                )
            )
        ):
            continue
        out.append(clip)
    return out


def run_extract(args: argparse.Namespace) -> None:
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    ids = sorted(json.loads(args.final_readout.read_text(encoding="utf-8"))["ids"])
    if any(alignment["clips"][i]["reason"] != OK for i in ids):
        raise ValueError("finalized DoTA-CAP id not aligned ok -- wrong alignment file")
    dev = set(load_split(constants.V2_SPLIT_DOTA_CAP_DEV))
    tile = bool(args.tile)
    grid_dir = args.grid_dir or grid_out_dir(args.encoder, tile)
    maps_dir = args.out_dir / "maps"
    grid_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    manifest = grid_manifest(args.encoder, file_sha256(args.alignment), tile)
    extract_video_features.check_or_write_manifest(grid_dir, manifest, force=False)
    report: dict[str, dict[str, Any]] = (
        json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    )
    work = todo(
        alignment, ids, grid_dir, maps_dir, dev, report, frozenset(args.retry), maps=not tile
    )
    if args.cap_ids_file is not None:
        group = read_ids_file(args.cap_ids_file)
        work = [c for c in work if c["cap_id"] in group]
    if not work:
        LOGGER.info("Nothing to extract for %s", args.parts[0].parent)
        return
    need = [c["dota_id"] for c in work]
    missing = [v for v in need if not (args.v2_cache / f"{v}.npy").exists()]
    if not tile:
        if args.dota_clip_dir is None:
            raise ValueError("--dota-clip-dir is required without --tile (G1, D0)")
        missing += [
            v for v in need if v in dev and not (args.dota_clip_dir / f"{v}.npy").exists()
        ]
    if missing:
        raise FileNotFoundError(f"{len(missing)} reference rows missing (G0/G1): {missing[:5]}")
    by_cap: dict[str, list[dict[str, Any]]] = {}
    for clip in work:
        by_cap.setdefault(clip["cap_id"], []).append(clip)
    device = resolve_device(args.device)
    extractor = Stage0Extractor(
        by_cap=by_cap,
        dev=dev,
        encoder=load_pretrained(args.encoder, device, None),
        clip_model=None if tile else extract_clip_features.load_pretrained_encoder(device),
        v2_cache=args.v2_cache,
        dota_clip_dir=args.dota_clip_dir,
        grid_dir=grid_dir,
        maps_dir=maps_dir,
        farm=args.work_dir / FARM_DIR,
        device=device,
        subdir=args.frames_subdir,
        batch_size=args.batch_size,
        report=report,
        save_report=lambda: write_json_atomic(args.report, report),
        tile=tile,
    )
    LOGGER.info(
        "Stage 0: %d DoTA-CAP clips (%d dev) from %d CAP videos",
        len(work),
        sum(c["dota_id"] in dev for c in work),
        len(by_cap),
    )
    stream_video_batches(
        sorted(args.parts),
        args.work_dir / "stream",
        args.frames_subdir,
        args.max_batch_gb * constants.BYTES_PER_GB,
        extractor,
        keep_ids=set(by_cap),
    )
    shutil.rmtree(args.work_dir, ignore_errors=True)


# --------------------------------------------------------------------------- readout


@dataclass(frozen=True)
class ClipRead:
    """Per-clip reads of one representation (S0-6 ... S0-8)."""

    effect: float | None
    rho: float | None
    hit: float | None
    chance: float | None
    hit_pre: float | None
    auc: dict[str, float]
    chance_pre: float | None = None


def _weighted(delta: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    inside = float((w * delta).sum() / w.sum())
    outside = float(((1.0 - w) * delta).sum() / (1.0 - w).sum())
    return inside, outside


def _rho(deltas: list[np.ndarray], masks: list[np.ndarray]) -> float:
    pairs = [_weighted(d, w) for d, w in zip(deltas, masks, strict=True)]
    return math.log(np.mean([p[0] for p in pairs]) / np.mean([p[1] for p in pairs]))


def _hits(deltas: list[np.ndarray], masks: list[np.ndarray]) -> float:
    return float(np.mean([w[int(np.argmax(d))] > 0 for d, w in zip(deltas, masks, strict=True)]))


def clip_read(
    delta: np.ndarray,
    glob: np.ndarray,
    steps: np.ndarray,
    labels: np.ndarray,
    boxes: dict[int, list[list[float]]],
    topk: int,
) -> ClipRead:
    """D1 / D3 / D2 of one clip on one representation's change map ``delta (K, grid*grid)``."""
    grid = math.isqrt(delta.shape[1])
    first_pos = int(np.argmax(labels)) if labels.any() else len(labels)
    defined = [k for k in range(len(steps)) if np.isfinite(delta[k]).all()]
    anom = [k for k in defined if labels[steps[k]] and boxes.get(int(steps[k]))]
    pre = [k for k in defined if steps[k] < first_pos]
    masks = {k: tile_weights(boxes[int(steps[k])], grid) for k in anom}
    anom = [k for k in anom if 0 < masks[k].sum() < grid * grid]
    rho = effect = hit = chance = hit_pre = chance_pre = None
    if len(anom) >= constants.V5_MIN_STEPS:
        d_a, m_a = [delta[k] for k in anom], [masks[k] for k in anom]
        rho = _rho(d_a, m_a)
        hit = _hits(d_a, m_a)
        chance = float(np.mean([(m > 0).mean() for m in m_a]))
        if len(pre) >= constants.V5_MIN_STEPS:
            fixed = masks[anom[0]]
            d_p = [delta[k] for k in pre]
            effect = rho - _rho(d_p, [fixed] * len(pre))
            hit_pre = _hits(d_p, [fixed] * len(pre))
            chance_pre = float((fixed > 0).mean())
    auc: dict[str, float] = {}
    if 0 < int(labels.sum()) < len(labels):
        if defined:
            ends = defined[0], defined[-1]
            filled, g = _fill(delta, *ends), _fill(glob, *ends)
        else:
            filled, g = np.zeros_like(delta), np.zeros_like(glob)
        part = np.sort(filled, axis=1)[:, -topk:]
        for name, score in (("global", g), ("max", filled.max(1)), ("topk", part.mean(1))):
            auc[name] = frame_auc(to_native(score, constants.V5_STAGE0_STRIDE, len(labels)), labels)
    return ClipRead(effect, rho, hit, chance, hit_pre, auc, chance_pre)


def load_boxes(annotation: Path) -> tuple[dict[int, list[list[float]]], bool]:
    data = json.loads(annotation.read_text(encoding="utf-8"))
    boxes = {
        int(f["frame_id"]): [o["bbox"] for o in f["objects"]]
        for f in data["labels"]
        if f["objects"]
    }
    return boxes, bool(data["ego_involve"])


def box_share(boxes: dict[int, list[list[float]]], labels: np.ndarray) -> float | None:
    area = constants.V5_BOX_FRAME_SIZE[0] * constants.V5_BOX_FRAME_SIZE[1]
    shares = [
        max((b[2] - b[0]) * (b[3] - b[1]) for b in bs) / area
        for f, bs in boxes.items()
        if f < len(labels) and labels[f]
    ]
    return float(np.median(shares)) if shares else None


def share_bin(share: float | None) -> str | None:
    if share is None:
        return None
    edges = constants.V5_BOX_SHARE_BINS
    return BIN_NAMES[int(np.searchsorted(edges, share, side="right"))]


def ci(values: dict[str, float]) -> dict[str, float] | None:
    return cluster_bootstrap_ci(
        values,
        {v: dota_group(v) for v in values},
        constants.V5_BOOTSTRAP,
        constants.V5_BOOTSTRAP_SEED,
        constants.V5_BOOTSTRAP_LEVEL,
    )


Read = Callable[[ClipRead], float | None]
Keep = Callable[[str], bool]


def _effect(r: ClipRead) -> float | None:
    return r.effect


def _rho_of(r: ClipRead) -> float | None:
    return r.rho


def _lift(r: ClipRead) -> float | None:
    return None if r.hit is None or r.chance is None else r.hit - r.chance


def _vs_pre(r: ClipRead) -> float | None:
    """Amendment 1 (S0-7'): lift over chance during the anomaly minus the same before it."""
    if r.hit is None or r.chance is None or r.hit_pre is None or r.chance_pre is None:
        return None
    return (r.hit - r.chance) - (r.hit_pre - r.chance_pre)


def _d2(r: ClipRead) -> float | None:
    return r.auc["topk"] - r.auc["global"] if r.auc else None


def _auc_of(score: str) -> Read:
    def read(r: ClipRead) -> float | None:
        return r.auc.get(score)

    return read


def _pick(reads: dict[str, ClipRead], fn: Read, keep: Keep) -> dict[str, float]:
    out: dict[str, float] = {}
    for v, r in reads.items():
        value = fn(r)
        if value is not None and keep(v):
            out[v] = value
    return out


def summarize_rep(
    reads: dict[str, ClipRead], ego: dict[str, bool], bins: dict[str, str | None]
) -> dict[str, Any]:
    """Every S0-11 / S0-12 interval of one representation."""

    def stat(fn: Read, keep: Keep) -> dict[str, float] | None:
        return ci(_pick(reads, fn, keep))

    def every(_: str) -> bool:
        return True

    def non_ego(v: str) -> bool:
        return not ego[v]

    def in_bin(name: str) -> Keep:
        return lambda v: bins[v] == name

    def is_ego(flag: bool) -> Keep:
        return lambda v: ego[v] == flag

    return {
        "effect": stat(_effect, every),
        "effect_non_ego": stat(_effect, non_ego),
        "rho": stat(_rho_of, every),
        "hit_minus_chance": stat(_lift, every),
        "hit_minus_pre": stat(_vs_pre, every),
        "d2_topk_minus_global_non_ego": stat(_d2, non_ego),
        "d2_topk_minus_global": stat(_d2, every),
        "macro": {s: stat(_auc_of(s), every) for s in ("global", "max", "topk")},
        "by_bin": {
            b: {"effect": stat(_effect, in_bin(b)), "hit_minus_chance": stat(_lift, in_bin(b))}
            for b in BIN_NAMES
        },
        "by_ego": {
            label: {
                "effect": stat(_effect, is_ego(flag)),
                "hit_minus_chance": stat(_lift, is_ego(flag)),
                "d2": stat(_d2, is_ego(flag)),
            }
            for flag, label in ((True, "ego"), (False, "non_ego"))
        },
    }


def _low(stat: dict[str, float] | None) -> float:
    return -math.inf if stat is None else stat["low"]


def gates(rep: dict[str, Any]) -> dict[str, bool]:
    """S0-11 G-A / G-B / G-C of one representation."""
    non_ego = rep["effect_non_ego"]
    return {
        "G-A": _low(rep["effect"]) > 0 and non_ego is not None and non_ego["mean"] > 0,
        "G-B": _low(rep["hit_minus_chance"]) > 0 and _low(rep["hit_minus_pre"]) > 0,
        "G-C": _low(rep["d2_topk_minus_global_non_ego"]) > 0,
    }


def decide(by_rep: dict[str, dict[str, bool]], valid: bool) -> str:
    """S0-11's table; INVALID when a validity gate failed (S0-10)."""
    if not valid:
        return "INVALID"
    if all(by_rep["V7"].values()):
        return "GO-7"
    if all(by_rep["V14"].values()):
        return "GO-14"
    if by_rep["V7"]["G-A"] or by_rep["V14"]["G-A"]:
        return "PARTIAL"
    return "KILL"


def d0(maps: dict[str, dict[str, np.ndarray]], seed: int) -> dict[str, float]:
    """S0-12 D0: CLIP cosine ``V5_D0_LAG`` frames apart within a clip vs random cross-clip pairs."""
    rows = {v: m["clip_s1"].astype(np.float64) for v, m in maps.items()}
    lag = constants.V5_D0_LAG
    allrows = np.concatenate(list(rows.values()))
    centre = allrows.mean(0)
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for name, shift in (("raw", 0.0), ("centred", centre)):
        within = [
            _cos_rows(r[lag:] - shift, r[:-lag] - shift).mean()
            for r in rows.values()
            if len(r) > lag
        ]
        a = rng.integers(0, len(allrows), size=20000)
        b = rng.integers(0, len(allrows), size=20000)
        out[f"within_{name}"] = float(np.mean(within))
        out[f"across_{name}"] = float(_cos_rows(allrows[a] - shift, allrows[b] - shift).mean())
    return out


@dataclass
class DevInputs:
    """DoTA-CAP-dev clips with stored maps, their labels, boxes and validity (S0-10)."""

    report: dict[str, dict[str, Any]]
    failed: dict[str, str]
    missing: list[str]
    maps: dict[str, dict[str, np.ndarray]]
    labels: dict[str, np.ndarray]
    boxes: dict[str, dict[int, list[list[float]]]]
    ego: dict[str, bool]
    bins: dict[str, str | None]

    def validity(self) -> dict[str, Any]:
        cosines = (e.get("g0_min_cos", 1.0) for e in self.report.values())
        return {
            "failed": self.failed,
            "missing_dev": self.missing,
            "dev_maps": len(self.maps),
            "g0_min_cos": min(cosines, default=None),
            "clips_reported": len(self.report),
        }


def load_dev(args: argparse.Namespace) -> DevInputs:
    report: dict[str, dict[str, Any]] = {}
    for path in args.reports:
        report |= json.loads(path.read_text(encoding="utf-8"))
    dev = sorted(load_split(constants.V2_SPLIT_DOTA_CAP_DEV))
    records = {
        r.video_id: r for r in parse_metadata(args.metadata, read_split_ids(args.split_file))
    }
    out = DevInputs(
        report=report,
        failed={v: e["reason"] for v, e in report.items() if e["reason"] != OK},
        missing=[v for v in dev if v not in report],
        maps={},
        labels={},
        boxes={},
        ego={},
        bins={},
    )
    for v in dev:
        path = args.maps_dir / f"{v}{MAP_SUFFIX}"
        if report.get(v, {}).get("reason") != OK or not path.exists():
            continue
        with np.load(path) as data:
            out.maps[v] = {k: data[k] for k in data.files}
        out.boxes[v], out.ego[v] = load_boxes(args.annotations_dir / f"{v}.json")
        out.labels[v] = native_labels(records[v], int(out.maps[v]["num_frames"]))
        out.bins[v] = share_bin(box_share(out.boxes[v], out.labels[v]))
    return out


def _topk(cells: int) -> int:
    return constants.V5_TOPK.get(math.isqrt(cells), constants.V5_TOPK[constants.V5_GRID_COARSE])


def run_readout(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_dev(args)
    maps, ego, bins = inputs.maps, inputs.ego, inputs.bins
    reads: dict[str, dict[str, ClipRead]] = {r: {} for r in REPS}
    ruler: dict[str, float] = {}
    for v, m in maps.items():
        labels, boxes = inputs.labels[v], inputs.boxes[v]
        if 0 < int(labels.sum()) < len(labels):
            ruler[v] = frame_auc(np.arange(len(labels)) / len(labels), labels)
        for rep, key, glob in (("V7", "d7", "g_v"), ("V14", "d14", "g_v"), ("C14", "dc14", "g_c")):
            topk = _topk(m[key].shape[1])
            reads[rep][v] = clip_read(m[key], m[glob], m["steps"], labels, boxes, topk)
    summary = {rep: summarize_rep(reads[rep], ego, bins) for rep in REPS}
    by_rep = {rep: gates(summary[rep]) for rep in GATED}
    valid = not inputs.failed and not inputs.missing
    readout = {
        "rule": RULE,
        "decision": decide(by_rep, valid),
        "gates": by_rep,
        "validity": inputs.validity(),
        "summary": summary,
        "ruler_t_over_n": ci(ruler),
        "p_t2_cap_dev": constants.V5_P_T2_CAP_DEV,
        "d0": d0(maps, constants.V5_BOOTSTRAP_SEED) if maps else None,
        "bins": {b or "no_box": sum(x == b for x in bins.values()) for b in set(bins.values())},
        "ego_count": {"ego": sum(ego.values()), "non_ego": sum(not e for e in ego.values())},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render(readout))
    return readout


def _fmt(stat: dict[str, float] | None) -> str:
    if stat is None:
        return "—"
    return f"{stat['mean']:+.4f} [{stat['low']:+.4f}, {stat['high']:+.4f}] (n = {stat['clips']})"


def render(readout: dict[str, Any]) -> str:
    s = readout["summary"]
    valid = readout["validity"]
    lines = [
        "# v5 Stage 0 — VideoMAE token evidence on DoTA-CAP-dev",
        "",
        f"Rule: {readout['rule']} (§1 signed before any grid existed; Amendment 2 after the "
        "first read-out).",
        "",
        f"**Decision: {readout['decision']}**",
        "",
        f"Validity (S0-10): dev maps {valid['dev_maps']}, failed {len(valid['failed'])} "
        f"{valid['failed'] or ''}, missing dev {len(valid['missing_dev'])}, "
        f"G0 min cos {valid['g0_min_cos']}",
        "",
        "## Gates (S0-11)",
        "",
        "| rep | G-A | G-B | G-C |",
        "|---|---|---|---|",
    ]
    for rep, g in readout["gates"].items():
        lines.append(f"| {rep} | {g['G-A']} | {g['G-B']} | {g['G-C']} |")
    lines += [
        "",
        "## Reads (cluster bootstrap over source videos)",
        "",
        "| read | V7 | V14 | C14 (printed) |",
        "|---|---|---|---|",
    ]
    for key, name in (
        ("effect", "D1 e = rho - rho_pre"),
        ("effect_non_ego", "D1 e, non-ego"),
        ("rho", "D1 rho"),
        ("hit_minus_chance", "D3 hit - chance"),
        ("hit_minus_pre", "D3 lift - lift_pre (Am. 1)"),
        ("d2_topk_minus_global_non_ego", "D2 top-k - global, non-ego"),
        ("d2_topk_minus_global", "D2 top-k - global, all"),
    ):
        lines.append(f"| {name} | " + " | ".join(_fmt(s[r][key]) for r in REPS) + " |")
    lines += ["", "D2 macros (training-free; printed): ", ""]
    for rep in REPS:
        lines.append(
            f"- {rep}: " + " · ".join(f"{k} {_fmt(v)}" for k, v in s[rep]["macro"].items())
        )
    lines += [
        f"- `t/N` ruler {_fmt(readout['ruler_t_over_n'])} · v2 `p_T2` {readout['p_t2_cap_dev']}",
        "",
        "## By box share and ego (printed; confounded, S0-12)",
        "",
        "| group | V7 effect | V7 hit - chance | V14 effect |",
        "|---|---|---|---|",
    ]
    for b in BIN_NAMES:
        lines.append(
            f"| {b} | {_fmt(s['V7']['by_bin'][b]['effect'])} | "
            f"{_fmt(s['V7']['by_bin'][b]['hit_minus_chance'])} | "
            f"{_fmt(s['V14']['by_bin'][b]['effect'])} |"
        )
    for e in ("ego", "non_ego"):
        lines.append(
            f"| {e} | {_fmt(s['V7']['by_ego'][e]['effect'])} | "
            f"{_fmt(s['V7']['by_ego'][e]['hit_minus_chance'])} | "
            f"{_fmt(s['V14']['by_ego'][e]['effect'])} |"
        )
    if readout["d0"]:
        d = readout["d0"]
        lines += [
            "",
            f"D0 (CLIP, {constants.V5_D0_LAG} frames): within raw {d['within_raw']:.3f} vs "
            f"across {d['across_raw']:.3f}; centred within {d['within_centred']:.3f} vs across "
            f"{d['across_centred']:.3f}",
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- check L (§1.3)


def lag_read(
    grid: np.ndarray,
    steps: np.ndarray,
    labels: np.ndarray,
    boxes: dict[int, list[list[float]]],
    lag: int,
    shift: int = 0,
) -> ClipRead:
    """S0-6 ... S0-8 on a stored ``(K, g, g, D)`` grid: change ``lag`` rows back, read ``shift``
    rows ahead of step ``k``'s box and label (§1.3 L-1 / L-3)."""
    delta = change_map(grid, lag)
    glob = global_change(grid.reshape(len(grid), -1, grid.shape[-1]).mean(1), lag)
    if shift:
        delta, glob = shift_rows(delta, shift), shift_rows(glob, shift)
    return clip_read(delta, glob, steps, labels, boxes, _topk(delta.shape[1]))


def cell_key(lag: int, shift: int) -> str:
    return f"lag{lag}_shift{shift}"


def centre_verdict(cells: dict[str, dict[str, Any]], lags: list[int]) -> str:
    """§1.3 L-3: centred iff, at one lag, G-B passes at shifts 2 and 3 and fails at 0."""
    for lag in lags:
        passes = {s: cells[cell_key(lag, s)]["gates"]["G-B"] for s in constants.V5_SHIFT_ROWS}
        if not passes[0] and all(passes[s] for s in constants.V5_SHIFT_ROWS if s):
            return f"centred (lag {lag})"
    return "causal"


def run_lag(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_dev(args)
    combos = [(lag, shift) for lag in args.lags for shift in args.shifts]
    reads: dict[str, dict[str, ClipRead]] = {cell_key(*c): {} for c in combos}
    worst = 0.0
    for v, m in inputs.maps.items():
        grid = np.load(args.grid_dir / f"{v}.npy").astype(np.float64)
        if len(grid) != len(m["steps"]):
            raise ValueError(f"{v}: grid has {len(grid)} rows, maps {len(m['steps'])} steps")
        worst = max(worst, float(np.nanmax(np.abs(change_map(grid) - m["d7"]))))
        for lag, shift in combos:
            reads[cell_key(lag, shift)][v] = lag_read(
                grid, m["steps"], inputs.labels[v], inputs.boxes[v], lag, shift
            )
    cells: dict[str, dict[str, Any]] = {}
    for lag, shift in combos:
        summary = summarize_rep(reads[cell_key(lag, shift)], inputs.ego, inputs.bins)
        cells[cell_key(lag, shift)] = {
            "lag": lag,
            "shift": shift,
            "gates": gates(summary),
            "summary": summary,
        }
    full = set(args.shifts) >= set(constants.V5_SHIFT_ROWS)
    out = {
        "rule": "core/docs/v5/PREREG_V5.md §1.3 (check L, L-1..L-3; printed, never gating)",
        "validity": inputs.validity(),
        "lag1_vs_stored_d7_max_abs": worst,
        "frames_per_row": constants.V5_STAGE0_STRIDE,
        "window_frames": constants.VIDEOMAE_CLIP_FRAMES * constants.DOTA_VIDEOMAE_FRAME_STEP,
        "cells": cells,
        "l3_window": centre_verdict(cells, args.lags) if full else None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / LAG_JSON, out)
    write_text_atomic(args.out_dir / LAG_MD, render_lag(out))
    return out


LAG_COLUMNS = (
    ("effect", "D1 e"),
    ("effect_non_ego", "D1 e non-ego"),
    ("rho", "D1 rho"),
    ("hit_minus_chance", "D3 hit - chance"),
    ("hit_minus_pre", "D3 lift - lift_pre"),
    ("d2_topk_minus_global_non_ego", "D2 top-k - global non-ego"),
)


def _short(stat: dict[str, float] | None) -> str:
    if stat is None:
        return "—"
    return f"{stat['mean']:+.3f} [{stat['low']:+.3f}, {stat['high']:+.3f}]"


def render_lag(out: dict[str, Any]) -> str:
    stride = out["frames_per_row"]
    valid = out["validity"]
    centre_offset = (out["window_frames"] - 1) / 2
    names = " | ".join(name for _, name in LAG_COLUMNS)
    lines = [
        "# v5 Stage 0 — check L: V7 change by lag (window overlap) and centre shift",
        "",
        f"Rule: {out['rule']}.",
        "",
        f"Validity: dev maps {valid['dev_maps']}, failed {len(valid['failed'])}, "
        f"missing dev {len(valid['missing_dev'])}; lag 1 (fp16 grid) vs stored fp32 `d7`: "
        f"max |Δ| {out['lag1_vs_stored_d7_max_abs']:.2e}",
        "",
        f"**L-3 window for Stage 1: {out['l3_window']}**",
        "",
        f"| lag (frames) | shift (window centre vs t) | {names} | macro top-k | G-A | G-B | G-C |",
        "|---" * (len(LAG_COLUMNS) + 6) + "|",
    ]
    for cell in out["cells"].values():
        s, g = cell["summary"], cell["gates"]
        centre = cell["shift"] * stride - centre_offset
        reads = " | ".join(_short(s[key]) for key, _ in LAG_COLUMNS)
        head = f"| {cell['lag']} ({cell['lag'] * stride}) | {cell['shift']} ({centre:+.1f}) |"
        flags = " | ".join(str(g[name]) for name in ("G-A", "G-B", "G-C"))
        lines.append(f"{head} {reads} | {_short(s['macro']['topk'])} | {flags} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI


def _add_read_args(read: argparse.ArgumentParser) -> None:
    read.add_argument("--reports", type=Path, nargs="+", required=True)
    read.add_argument("--maps-dir", type=Path, default=stage0_dir() / "maps")
    read.add_argument(
        "--annotations-dir",
        type=Path,
        default=constants.DATA_ROOT / "DoTA" / "raw_annotations" / "annotations",
    )
    read.add_argument(
        "--metadata", type=Path, default=constants.DATA_ROOT / "DoTA" / "metadata_val.json"
    )
    read.add_argument(
        "--split-file", type=Path, default=constants.DATA_ROOT / "DoTA" / "val_split.txt"
    )
    read.add_argument("--out-dir", type=Path, default=stage0_dir() / "REPORTS")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ext = sub.add_parser("extract", help="Colab: stream CAP tars, store V7 grids + dev change maps")
    ext.add_argument("--alignment", type=Path, required=True)
    ext.add_argument("--final-readout", type=Path, required=True, help="dota_cap_readout.json")
    ext.add_argument("--parts", type=Path, nargs="+", required=True)
    ext.add_argument("--work-dir", type=Path, required=True)
    ext.add_argument("--dota-clip-dir", type=Path, default=None,
                     help="DoTA_s1_ncc (G1, D0); not read with --tile")
    ext.add_argument("--v2-cache", type=Path, required=True, help="V2-S DoTA_CAP_s1_squash (G0)")
    ext.add_argument("--report", type=Path, required=True)
    ext.add_argument("--out-dir", type=Path, default=stage0_dir())
    ext.add_argument("--cap-ids-file", type=Path, default=None)
    ext.add_argument("--grid-dir", type=Path, default=None, help="default: grid_out_dir(encoder)")
    ext.add_argument("--encoder", default=constants.V5_STAGE0_ENCODER)
    ext.add_argument("--frames-subdir", default="images")
    ext.add_argument("--batch-size", type=int, default=16)
    ext.add_argument("--max-batch-gb", type=float, default=constants.MMAU_STREAM_BATCH_GB)
    ext.add_argument("--device", default="auto")
    ext.add_argument(
        "--retry", nargs="*", default=[], help="redo clips that failed for these reasons (Am. 2)"
    )
    ext.add_argument(
        "--tile", action="store_true",
        help="v5.1 (PREREG_V5 §3): centred 4x7x7 grids, gate G-c, no dev maps / CLIP patches",
    )
    _add_read_args(sub.add_parser("readout", help="local: D1/D3/D2 + the S0-11 decision"))
    lag = sub.add_parser("lag", help="check L (§1.3): V7 change at row lags, printed only")
    _add_read_args(lag)
    lag.add_argument("--grid-dir", type=Path, default=grid_out_dir(constants.V5_STAGE0_ENCODER))
    lag.add_argument("--lags", type=int, nargs="+", default=list(constants.V5_LAG_ROWS))
    lag.add_argument("--shifts", type=int, nargs="+", default=list(constants.V5_SHIFT_ROWS))
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "extract":
        run_extract(args)
    elif args.command == "lag":
        out = run_lag(args)
        LOGGER.info("check L: L-3 window %s", out["l3_window"])
    else:
        readout = run_readout(args)
        LOGGER.info("Stage 0 decision: %s", readout["decision"])


if __name__ == "__main__":
    main()
