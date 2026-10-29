"""v5 grid inputs: V7 grids appended to v2's A2 rows (``core/docs/v5/PREREG_V5.md`` §2.1).

A v5 input row is v2's A2 row followed by the flattened grid of the same step::

    [ x (512) ; c * (u - m_u) / sigma_u (384) ; G_t (49 * 384) ; (G_t - G_{t-l}) (49 * 384) ]

The last block exists only for change keys (R5Δ). The baked ``u`` columns stay in the row so the
input cache is exactly A2's (its manifest check is unchanged) and S1-G2(b) can compare the v5
motion term against them; the v5 model never reads them.

``ΔG`` is computed **inside one dataset item** -- a training window, an eval crop, a DoTA clip --
before DVS splices items together, so a splice boundary never reads as change (S2-4). Its first
``l`` rows are zero. ``l`` is the smallest row lag whose VideoMAE window shares no frame with row
``t``'s: ``ceil((15 * step + 1) / stride)`` (:func:`change_lag`).

The head file (:data:`constants.V5_HEAD_FILENAME`) carries what turns a readout ``r_t`` into A2's
motion term: V2-S's ``fc_norm`` and A2's baked ``c``, ``m_u``, ``sigma_u`` (deviation D1).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import constants

_HEAD_KEYS = ("fc_weight", "fc_bias", "eps", "c", "m_u", "sigma_u")


def change_lag(stride: int, frame_step: int) -> int:
    """Rows between two steps whose VideoMAE windows share no frame (S2-4)."""
    if stride < 1 or frame_step < 1:
        raise ValueError(f"stride and frame_step must be >= 1, got {stride}, {frame_step}")
    return math.ceil((constants.V5_SPAN_GAPS * frame_step + 1) / stride)


GRID_WIDTHS = (constants.V5_GRID_WIDTH, constants.V5_TILE_GRID_WIDTH)  # V7 | 4 x 7 x 7


def centred_reference(
    rows: np.ndarray, frame_step: int, num_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Which centred rows equal a causal ``*_s1_squash`` row, and that row (v5.1 gate G-c).

    The clip centred on ``t`` is the causal clip ending ``(16 - 1 - lead) * step`` frames later,
    unless either end is clamped. Returns ``(keep (len(rows),) bool, s1 index of the kept rows)``.
    """
    rows = np.asarray(rows, dtype=np.int64)
    lead = constants.V5_CENTRE_LEAD * frame_step
    lag = (constants.VIDEOMAE_CLIP_FRAMES - 1 - constants.V5_CENTRE_LEAD) * frame_step
    keep: np.ndarray = (rows - lead >= 0) & (rows + lag < num_frames)
    index: np.ndarray = rows[keep] + lag
    return keep, index


def flat_grid(grid: np.ndarray) -> np.ndarray:
    """``(T, [4,] 7, 7, D)`` or ``(T, cells, D)`` -> ``(T, cells * D)`` (slot, tile, channel).

    V7 rows (Stages 0-2) hold 49 cells, v5.1 rows 196 (4 slots x 49 tiles, PREREG_V5 §3).
    """
    rows = np.asarray(grid)
    if rows.ndim not in (3, 4, 5):
        raise ValueError(f"a grid is (T, [S,] 7, 7, D) or (T, cells, D), got shape {rows.shape}")
    flat: np.ndarray = rows.reshape(len(rows), -1)
    if flat.shape[1] not in GRID_WIDTHS:
        raise ValueError(f"grid rows have {flat.shape[1]} values, expected one of {GRID_WIDTHS}")
    return flat


def grid_change(flat: np.ndarray, lag: int) -> np.ndarray:
    """``G_t - G_{t-lag}`` of one item's flattened grid; rows ``t < lag`` are zero (S2-4)."""
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    base = np.asarray(flat, dtype=np.float32)
    out = np.zeros_like(base)
    if len(base) > lag:
        out[lag:] = base[lag:] - base[:-lag]
    return out


def v5_rows(rows: np.ndarray, grid: np.ndarray, lag: int | None) -> np.ndarray:
    """One item's A2 rows with its grid (and, given ``lag``, its change) appended, float32."""
    flat = flat_grid(grid)
    if len(flat) != len(rows):
        raise ValueError(f"grid rows {len(flat)} != input rows {len(rows)} (lesson C13)")
    parts = [np.asarray(rows, dtype=np.float32), flat.astype(np.float32)]
    if lag is not None:
        parts.append(grid_change(flat, lag))
    out: np.ndarray = np.concatenate(parts, axis=1)
    return out


def row_width(base_width: int, keys: str) -> int:
    """Columns of a v5 row given the A2 row width and the key type."""
    blocks = 2 if keys == constants.V5_KEYS_CHANGE else 1
    return base_width + blocks * constants.V5_GRID_WIDTH


@dataclass(frozen=True)
class GridGeometry:
    """How a grid cache's rows map to native frames."""

    stride: int  # cache row k = native frame k * stride
    frame_step: int  # VideoMAE frame step of the window at each row
    slots: int = constants.V5_SLOTS_V7  # temporal slots kept per row (v5.1: 4)
    window: str = constants.V5_WINDOW_CAUSAL  # causal (Stages 0-2) | centred (v5.1)


def read_grid_geometry(grid_dir: Path) -> GridGeometry:
    """The cache's stride and window step, from its manifest.

    Stage 0's DoTA-CAP cache predates the numeric fields; its ``rows`` string is matched exactly
    and mapped to protocol B's geometry (stride 3, DoTA frame step 1).
    """
    path = grid_dir / constants.V5_GRID_MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no grid manifest in {grid_dir}")
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "stride" in manifest and "frame_step" in manifest:
        return GridGeometry(
            int(manifest["stride"]),
            int(manifest["frame_step"]),
            int(manifest.get("slots", constants.V5_SLOTS_V7)),
            str(manifest.get("window_kind", constants.V5_WINDOW_CAUSAL)),
        )
    stage0_rows = f"s1[::{constants.V5_STAGE0_STRIDE}] (protocol B)"
    if manifest.get("rows") == stage0_rows:
        return GridGeometry(constants.V5_STAGE0_STRIDE, constants.DOTA_VIDEOMAE_FRAME_STEP)
    raise ValueError(f"{path}: no stride / frame_step and not Stage 0's DoTA-CAP cache")


def tile_boundary_rows(geometry: GridGeometry, stride: int) -> int:
    """Rows around an inserted DVS boundary left out of L_tile's dense negatives (PREREG_V5 §3).

    A centred clip reaches ``V5_CENTRE_LEAD * frame_step`` native frames before its row; at
    ``stride`` that is ``ceil(lead * step / stride)`` rows (Nexar s8: 3).
    """
    return math.ceil(constants.V5_CENTRE_LEAD * geometry.frame_step / stride)


def item_grid(
    grid: np.ndarray, geometry: GridGeometry, start: int, end: int, stride: int
) -> np.ndarray:
    """Grid rows of native frames ``range(start, end, stride)`` from a cache at ``geometry``.

    v5.1 (PREREG_V5 §3, ``nexar_val`` at s3): a centred cache at the item's own stride serves an
    item of any phase by snapping it to the phase-0 rows, ``start - start % stride`` (< ``stride``
    frames earlier); any other off-grid read raises (Stages 0-2 caches unchanged).
    """
    frames = np.arange(start, end, stride)
    snap = geometry.window == constants.V5_WINDOW_CENTRED and stride == geometry.stride
    if len(frames) and snap:
        frames = frames - start % stride
    if len(frames) and (frames % geometry.stride).any():
        raise ValueError(
            f"frames {start}:{end}:{stride} are not on the cache's s1[::{geometry.stride}] rows"
        )
    index = frames // geometry.stride
    if len(index) and index[-1] >= len(grid):
        raise ValueError(
            f"grid cache has {len(grid)} rows, frame {frames[-1]} needs row {index[-1]}"
        )
    out: np.ndarray = np.asarray(grid[index])
    return out


def eval_lag(keys: str, frame_step: int, stride: int) -> int | None:
    """The change lag an eval item at ``stride`` uses (None for content keys, S2-4)."""
    return change_lag(stride, frame_step) if keys == constants.V5_KEYS_CHANGE else None


@dataclass(frozen=True)
class EvalGrid:
    """What an eval tool needs to append one checkpoint's grid columns to its items."""

    grid_dir: Path
    geometry: GridGeometry
    stride: int  # the eval protocol's stride over native frames
    lag: int | None  # change lag at that stride (None for content keys)

    def append(self, rows: np.ndarray, video_id: str, start: int, end: int) -> np.ndarray:
        """``rows`` = the item ``range(start, end, stride)``'s A2 rows -> its v5 rows."""
        grid = np.load(self.grid_dir / f"{video_id}.npy", mmap_mode="r")
        return v5_rows(rows, item_grid(grid, self.geometry, start, end, self.stride), self.lag)


def eval_grid(readout: str, keys: str, grid_dir: Path | None, stride: int) -> EvalGrid | None:
    """``None`` for a non-v5 checkpoint; refuses a v5 checkpoint without a grid cache and back."""
    if readout == constants.V5_READOUT_OFF:
        if grid_dir is not None:
            raise ValueError("a grid cache was given for a checkpoint with v5.readout=none")
        return None
    if grid_dir is None:
        raise ValueError(f"checkpoint has v5.readout={readout!r}: pass its grid cache")
    geometry = read_grid_geometry(grid_dir)
    return EvalGrid(grid_dir, geometry, stride, eval_lag(keys, geometry.frame_step, stride))


def v5_arm(v5: Any) -> dict[str, Any]:
    """A checkpoint's ``cfg.v5`` as read-out provenance (empty for a v1 / v2 checkpoint)."""
    if v5.readout == constants.V5_READOUT_OFF:
        return {}
    return {
        "v5_readout": v5.readout,
        "v5_queries": v5.queries,
        "v5_keys": v5.keys,
        "v5_smil_weight": v5.smil_weight,
        "v5_grid_slots": v5.grid_slots,
        "v5_window": v5.window,
    }


def save_head(
    path: Path,
    fc_weight: np.ndarray,
    fc_bias: np.ndarray,
    eps: float,
    c: float,
    m_u: np.ndarray,
    sigma_u: np.ndarray,
) -> Path:
    """Write the motion head (``fc_norm`` + A2's statistics), atomically."""
    arrays: dict[str, np.ndarray] = {
        "fc_weight": np.asarray(fc_weight, dtype=np.float32),
        "fc_bias": np.asarray(fc_bias, dtype=np.float32),
        "eps": np.array(eps, dtype=np.float64),
        "c": np.array(c, dtype=np.float64),
        "m_u": np.asarray(m_u, dtype=np.float64),
        "sigma_u": np.asarray(sigma_u, dtype=np.float64),
    }
    for name in ("fc_weight", "fc_bias", "m_u", "sigma_u"):
        if arrays[name].shape != (constants.V5_GRID_DIM,):
            raise ValueError(
                f"{name} has shape {arrays[name].shape}, expected ({constants.V5_GRID_DIM},)"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return path


def load_head(path: Path) -> dict[str, np.ndarray]:
    """Read a head file back; every key must be present."""
    with np.load(path) as data:
        missing = [k for k in _HEAD_KEYS if k not in data.files]
        if missing:
            raise KeyError(f"{path} lacks {missing}")
        return {k: np.asarray(data[k]) for k in _HEAD_KEYS}


__all__ = [
    "GRID_WIDTHS",
    "EvalGrid",
    "GridGeometry",
    "centred_reference",
    "change_lag",
    "eval_grid",
    "eval_lag",
    "flat_grid",
    "grid_change",
    "item_grid",
    "load_head",
    "read_grid_geometry",
    "row_width",
    "save_head",
    "tile_boundary_rows",
    "v5_arm",
    "v5_rows",
]
