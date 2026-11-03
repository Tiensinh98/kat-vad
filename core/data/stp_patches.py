"""STP patch inputs: motion-selected CLIP patch features appended to the host's rows.

Proposal ``core/docs/final/`` §3.2.2. Each sampled frame is cut into the 7 x 7 grid of windows,
each window is encoded by the frozen CLIP image tower (``core.tools.stp_patches``), and the
**fixed** motion prior picks ``K`` of the 49 patch features per step (STPrompt Eq. 1)::

    D_{t,j} = 2 P^_{t,j} - P^_{t-1,j} - P^_{t+1,j},  Mo_{t,j} = ||D_{t,j}||,  S_t = TopK_j Mo_{t,j}

with ``P^`` the channel-wise L2-normalised patches of the neighbouring **cache rows** (the
sampled steps). The first and last rows repeat their neighbour. Mo is computed per source video
-- before LaGoVAD's dynamic video synthesis splices items -- so a seam is never motion.
The ego-compensated variant (R7) subtracts the frame-mean ``D`` before the norm.

A patch cache stores per video ``{id}.npy`` = ``(L, K * 512)`` selected features (fp16, ordered
by decreasing Mo), ``idx/{id}.npy`` = ``(L, K)`` grid indices (row-major) for evidence maps and
``mo/{id}.npy`` = ``(L, 49)`` magnitudes. An STP input row is the host row followed by the
selected features::

    [ c_t (512) ; c * (u - m_u) / sigma_u (384) ; P_{t,S_t} (K * 512) ]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import constants


def motion_magnitude(patches: np.ndarray, motion: str = constants.STP_MOTION_RAW) -> np.ndarray:
    """``(L, 49, D)`` patch features of one video -> ``(L, 49)`` Mo (STPrompt Eq. 1)."""
    if motion not in constants.STP_MOTIONS:
        raise ValueError(f"motion must be one of {constants.STP_MOTIONS}, got {motion!r}")
    p = np.asarray(patches, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"patches must be (L, P, D), got shape {p.shape}")
    unit = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), constants.STP_NORM_EPS)
    prev = np.concatenate([unit[:1], unit[:-1]])  # the first row repeats its neighbour
    nxt = np.concatenate([unit[1:], unit[-1:]])
    change = 2 * unit - prev - nxt
    if motion == constants.STP_MOTION_EGO:
        change = change - change.mean(axis=1, keepdims=True)
    out: np.ndarray = np.linalg.norm(change, axis=-1)
    return out


def select_top_k(
    patches: np.ndarray, magnitude: np.ndarray, top_k: int = constants.STP_TOP_K
) -> tuple[np.ndarray, np.ndarray]:
    """``(L, K, D)`` features and ``(L, K)`` indices of the ``K`` largest Mo per row.

    Ordered by decreasing Mo; ties broken by the lower grid index (stable, deterministic).
    """
    if not 1 <= top_k <= magnitude.shape[1]:
        raise ValueError(f"top_k must be in [1, {magnitude.shape[1]}], got {top_k}")
    order = np.argsort(-magnitude, axis=1, kind="stable")[:, :top_k]
    selected = np.take_along_axis(np.asarray(patches), order[..., None], axis=1)
    return selected, order


def select_video(
    patches: np.ndarray, motion: str, top_k: int = constants.STP_TOP_K
) -> dict[str, np.ndarray]:
    """One video's ``(L, 49, D)`` patches -> the three arrays a patch cache stores."""
    magnitude = motion_magnitude(patches, motion)
    selected, index = select_top_k(patches, magnitude, top_k)
    return {
        "features": selected.reshape(len(selected), -1).astype(constants.STP_STORE_DTYPE),
        "index": index.astype(np.int16),
        "magnitude": magnitude.astype(np.float16),
    }


def build_manifest(
    dataset: str, stride: int, motion: str, top_k: int, cells: str, source: str
) -> dict[str, Any]:
    """Everything that makes two patch caches interchangeable (C2 / C13)."""
    return {
        "dataset": dataset,
        "stride": stride,
        "motion": motion,
        "top_k": top_k,
        "grid": constants.STP_GRID,
        "cells": cells,
        "model": constants.CLIP_MODEL_NAME,
        "revision": constants.CLIP_MODEL_REVISION,
        "feature": "CLIPVisionModelWithProjection.image_embeds per window, squash224 + CLIP norm",
        "neighbours": "+-1 cache row, edge rows repeat their neighbour",
        "dtype": constants.STP_STORE_DTYPE,
        "layout": "(L, K * 512) selected features, decreasing Mo; idx/ (L, K); mo/ (L, 49)",
        "source": source,
    }


def cache_dir(dataset: str, stride: int, motion: str, top_k: int = constants.STP_TOP_K) -> Path:
    name = f"{dataset}_s{stride}_k{top_k}_{motion}"
    return constants.CACHE_ROOT / constants.STP_PATCH_CACHE_DIRNAME / name


def read_manifest(patch_dir: Path) -> dict[str, Any]:
    path = patch_dir / constants.STP_PATCH_MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {constants.STP_PATCH_MANIFEST_FILENAME} in {patch_dir}")
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def check_patch_cache(patch_dir: Path, stride: int, motion: str, top_k: int) -> dict[str, Any]:
    """Refuse a cache built at another stride, motion prior or K (lesson C2)."""
    manifest = read_manifest(patch_dir)
    want = {"stride": stride, "motion": motion, "top_k": top_k}
    got = {k: manifest.get(k) for k in want}
    if got != want:
        raise ValueError(f"{patch_dir} was built with {got}; this run needs {want}")
    return manifest


def stp_rows(rows: np.ndarray, patches: np.ndarray) -> np.ndarray:
    """One item's host rows with its selected patch features appended, float32."""
    if len(patches) != len(rows):
        raise ValueError(f"patch rows {len(patches)} != input rows {len(rows)} (lesson C13)")
    out: np.ndarray = np.concatenate(
        [np.asarray(rows, dtype=np.float32), np.asarray(patches, dtype=np.float32)], axis=1
    )
    return out


@dataclass(frozen=True)
class EvalPatches:
    """What an eval tool needs to append one checkpoint's patch columns to its clips."""

    patch_dir: Path

    def append(self, rows: np.ndarray, video_id: str) -> np.ndarray:
        """Whole-clip rows (already at the cache's stride) -> STP rows."""
        return stp_rows(rows, np.load(self.patch_dir / f"{video_id}.npy", mmap_mode="r"))


def eval_patches(
    sa2: bool, motion: str, top_k: int, patch_dir: Path | None, stride: int
) -> EvalPatches | None:
    """``None`` for a checkpoint without SA²; refuses a missing or mismatched cache and back."""
    if not sa2:
        if patch_dir is not None:
            raise ValueError("a patch cache was given for a checkpoint with stp.sa2=false")
        return None
    if patch_dir is None:
        raise ValueError("checkpoint has stp.sa2=true: pass its patch cache (--patch-dir)")
    check_patch_cache(patch_dir, stride, motion, top_k)
    return EvalPatches(patch_dir)


def stp_arm(stp: Any) -> dict[str, Any]:
    """A checkpoint's ``cfg.stp`` as read-out provenance (empty for a pre-STP checkpoint)."""
    if not stp.sa2 and stp.text_prompt == constants.STP_PROMPT_RELEASED:
        return {}
    arm = {
        "stp_sa2": stp.sa2,
        "stp_top_k": stp.top_k if stp.sa2 else None,
        "stp_motion": stp.motion if stp.sa2 else None,
        "stp_text_prompt": stp.text_prompt,
    }
    if stp.branch != constants.STP_BRANCH_HOST:  # host arms keep their pre-L1 provenance keys
        arm["stp_branch"] = stp.branch
    return arm


__all__ = [
    "EvalPatches",
    "build_manifest",
    "cache_dir",
    "check_patch_cache",
    "eval_patches",
    "motion_magnitude",
    "read_manifest",
    "select_top_k",
    "select_video",
    "stp_arm",
    "stp_rows",
]
