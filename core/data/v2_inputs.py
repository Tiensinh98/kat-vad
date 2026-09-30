"""v2 input cache: CRN and the fixed motion scaling, baked per source video.

Architecture §4-§5 and §11. Both transforms are parameter-free given four
statistics fitted once on T2-train minus T2-val, so they are applied offline and
the model sees one row per step:

=====  ==========================================================  =========
arm    row                                                         width
=====  ==========================================================  =========
A0     ``x`` (the CLIP cache itself; nothing is baked)             512
A1     ``s * (x - mu^x)``                                          512
A2     ``[x ; c * (u - m_u) / sigma_u]``                           512 + d_v
A3     ``[s * (x - mu^x) ; c * (u - mu^u) / sigma_u]``             512 + d_v
=====  ==========================================================  =========

``mu`` is the CRN reference (:func:`core.crn.reference.reference`) of the file
being baked: a **source video** for T2 (train windows and eval windows are
slices of it) and a **clip** for DoTA. ``s`` makes the mean step norm of the CRN
stream equal the raw one, ``c`` is the raw CLIP stream's per-channel RMS
``E||x|| / sqrt(D)``, ``m_u`` the T2-train channel mean of ``u`` and ``sigma_u``
the per-channel std of the centred motion stream the arm feeds. Every baked cache
carries a manifest (:data:`constants.V2_INPUT_MANIFEST_FILENAME`) naming its arm
and the split its statistics were fitted on; training and evaluation refuse a
cache whose manifest disagrees with the run's ``v2`` config
(:func:`check_input_manifest`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.crn.reference import reference

LOGGER = logging.getLogger(__name__)

STATS_FILENAME = "v2_input_stats.npz"


@dataclass(frozen=True)
class V2Stats:
    """The fixed statistics of one arm, fitted on T2-train minus T2-val."""

    crn: str
    motion: str
    s: float  # CRN scale (1.0 without CRN)
    c: float  # CLIP per-channel RMS
    m_u: np.ndarray | None  # (d_v,) T2-train channel mean of u (A2's centring)
    sigma_u: np.ndarray | None  # (d_v,) per-channel std of the centred motion stream

    @property
    def has_crn(self) -> bool:
        return self.crn != constants.V2_OFF

    @property
    def has_motion(self) -> bool:
        return self.motion != constants.V2_OFF


def centred_clip(x: np.ndarray, crn: str) -> np.ndarray:
    """``x - mu^x`` under reference ``crn`` (``x`` itself when CRN is off)."""
    base = np.asarray(x, dtype=np.float64)
    if crn == constants.V2_OFF:
        return base
    centred: np.ndarray = base - reference(x, crn)
    return centred


def centred_motion(u: np.ndarray, crn: str, m_u: np.ndarray) -> np.ndarray:
    """``u - mu^u`` with CRN (A3), ``u - m_u`` without (A2)."""
    base = np.asarray(u, dtype=np.float64)
    centred: np.ndarray = base - (m_u if crn == constants.V2_OFF else reference(u, crn))
    return centred


def fit_stats(
    clips: dict[str, np.ndarray], motions: dict[str, np.ndarray] | None, crn: str, motion: str
) -> V2Stats:
    """Fit ``s``, ``c``, ``m_u``, ``sigma_u`` on the given (T2-train) sources."""
    if not clips:
        raise ValueError("fit_stats needs at least one source")
    ids = sorted(clips)
    raw = np.concatenate([np.asarray(clips[v], dtype=np.float64) for v in ids])
    mean_norm = float(np.linalg.norm(raw, axis=1).mean())
    c = mean_norm / float(np.sqrt(raw.shape[1]))
    s = 1.0
    if crn != constants.V2_OFF:
        dev = np.concatenate([centred_clip(clips[v], crn) for v in ids])
        s = mean_norm / max(float(np.linalg.norm(dev, axis=1).mean()), constants.V2_MIN_SIGMA)
    m_u = sigma_u = None
    if motion != constants.V2_OFF:
        if motions is None:
            raise ValueError(f"v2.motion={motion!r} needs motion features")
        missing = [v for v in ids if v not in motions]
        if missing:
            raise ValueError(f"{len(missing)} sources have no motion features: {missing[:5]}")
        m_u = np.concatenate([np.asarray(motions[v], dtype=np.float64) for v in ids]).mean(axis=0)
        centred = np.concatenate([centred_motion(motions[v], crn, m_u) for v in ids])
        sigma_u = np.maximum(centred.std(axis=0), constants.V2_MIN_SIGMA)
    return V2Stats(crn, motion, s, c, m_u, sigma_u)


def bake_rows(x: np.ndarray, u: np.ndarray | None, stats: V2Stats) -> np.ndarray:
    """One source's ``(T, 512 [+ d_v])`` float32 input rows for the arm in ``stats``."""
    clip = stats.s * centred_clip(x, stats.crn)
    if not stats.has_motion:
        return clip.astype(np.float32)
    if u is None or stats.m_u is None or stats.sigma_u is None:
        raise ValueError("a motion arm needs u, m_u and sigma_u")
    if len(u) != len(x):
        raise ValueError(f"motion rows {len(u)} != CLIP rows {len(x)} (lesson C13)")
    scaled = stats.c * centred_motion(u, stats.crn, stats.m_u) / stats.sigma_u
    rows: np.ndarray = np.concatenate([clip, scaled], axis=1).astype(np.float32)
    return rows


def save_stats(out_dir: Path, stats: V2Stats) -> Path:
    """Write the arrays beside the manifest (``np.savez``; small)."""
    arrays: dict[str, np.ndarray] = {"s": np.array(stats.s), "c": np.array(stats.c)}
    if stats.m_u is not None and stats.sigma_u is not None:
        arrays["m_u"] = stats.m_u
        arrays["sigma_u"] = stats.sigma_u
    path = out_dir / STATS_FILENAME
    np.savez(path, **arrays)
    return path


def load_stats(stats_dir: Path) -> V2Stats:
    """Read a fitted arm back from a baked cache (its manifest + stats file)."""
    manifest = read_input_manifest(stats_dir)
    if manifest is None:
        raise FileNotFoundError(f"no {constants.V2_INPUT_MANIFEST_FILENAME} in {stats_dir}")
    with np.load(stats_dir / STATS_FILENAME) as data:
        m_u = data.get("m_u", None)
        sigma_u = data.get("sigma_u", None)
        return V2Stats(
            manifest["crn"], manifest["motion"], float(data["s"]), float(data["c"]), m_u, sigma_u
        )


def read_input_manifest(cache_dir: Path) -> dict[str, Any] | None:
    """The cache's manifest, or ``None`` for a plain (A0) CLIP cache."""
    path = cache_dir / constants.V2_INPUT_MANIFEST_FILENAME
    if not path.is_file():
        return None
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def check_input_manifest(cache_dir: Path, crn: str, motion: str) -> None:
    """Refuse a feature cache that is not the input the run's ``v2`` config names.

    * v2 off (A0): a baked v2 cache is refused; a plain CLIP cache passes.
    * v2 on: the cache must carry a manifest whose ``crn`` and ``motion`` match.
    """
    manifest = read_input_manifest(cache_dir)
    wants_v2 = crn != constants.V2_OFF or motion != constants.V2_OFF
    if manifest is None:
        if wants_v2:
            raise ValueError(
                f"v2.crn={crn!r} v2.motion={motion!r} but {cache_dir} has no "
                f"{constants.V2_INPUT_MANIFEST_FILENAME}: it is a plain CLIP cache, not "
                "a v2 input cache (build it with core.tools.build_v2_inputs)"
            )
        return
    got = (manifest.get("crn"), manifest.get("motion"))
    if got != (crn, motion):
        raise ValueError(
            f"{cache_dir} was baked for crn={got[0]!r} motion={got[1]!r}, but the run "
            f"names crn={crn!r} motion={motion!r}"
        )
    LOGGER.info("v2 input cache %s matches crn=%s motion=%s", cache_dir, crn, motion)


__all__ = [
    "STATS_FILENAME",
    "V2Stats",
    "bake_rows",
    "centred_clip",
    "centred_motion",
    "check_input_manifest",
    "fit_stats",
    "load_stats",
    "read_input_manifest",
    "save_stats",
]
