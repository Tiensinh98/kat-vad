"""Standardizing the flow target: moments, the fitted statistics, the transform.

Numpy only, and imported by both :mod:`core.eda.features` (which measures the
scale of ``e_O``) and :mod:`core.flow.zscore_cache` (which rebuilds it), so the
``V`` that licensed Option A and the ``mu, sigma`` that implement it come from
one piece of arithmetic over the same rows. It imports nothing from
:mod:`core.eda`, which is what keeps that pair free of an import cycle.

Why the 23 raw statistics and not the 256-d ``e_O``
(``.project/plans/katvad-flow-zscore-option-a.md`` decision D-1): every output
dimension of ``e_O = s @ M`` is a mixture of all 23 stats, and ``mag_max`` holds
83 % of ``E[s^2]`` on T2. Standardizing after the projection fixes the units but
leaves every dimension dominated by that one stat; standardizing ``s`` first
gives each stat equal weight and then reuses the **same** ``M``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import constants


class MomentAccumulator:
    """Streaming per-dimension moments, with the within-item pool kept apart.

    Concatenating every cached array to call ``np.var`` once costs
    O(frames x dims) float64 and grows with the corpus; both passes here are
    O(dims) per item, the same reason the extractors stream (lesson **C9**).

    ``variance`` is the population variance over *all* frames pooled — the MSE a
    single global-mean vector would score. ``within_variance`` is the pooled
    per-item variance — the MSE an oracle that knew each item's own mean would
    score. The gap between them is the share of the target that is item
    identity rather than within-item dynamics.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.items = 0
        self.total = np.zeros(dim, dtype=np.float64)
        self.total_sq = np.zeros(dim, dtype=np.float64)
        self.within_sq = np.zeros(dim, dtype=np.float64)

    def add(self, rows: np.ndarray) -> None:
        block = np.asarray(rows, dtype=np.float64)
        self.items += 1
        self.count += int(block.shape[0])
        self.total += block.sum(axis=0)
        self.total_sq += np.square(block).sum(axis=0)
        self.within_sq += np.square(block - block.mean(axis=0)).sum(axis=0)

    @property
    def mean(self) -> np.ndarray:
        return self.total / max(self.count, 1)

    @property
    def second_moment(self) -> np.ndarray:
        moment: np.ndarray = self.total_sq / max(self.count, 1)
        return moment

    @property
    def variance(self) -> np.ndarray:
        spread: np.ndarray = np.maximum(self.second_moment - np.square(self.mean), 0.0)
        return spread

    @property
    def within_variance(self) -> np.ndarray:
        pooled: np.ndarray = self.within_sq / max(self.count, 1)
        return pooled


@dataclass(frozen=True)
class ZScoreStats:
    """Per-stat train-split moments, plus what they were fitted on.

    ``train_ids_sha1`` binds the statistics to one split the way a transform is
    bound to a feature cache (lesson **C2**): the same numbers applied to another
    corpus, or to a re-split of this one, are a silent mismatch.
    """

    mean: np.ndarray
    std: np.ndarray
    n_items: int
    n_rows: int
    train_ids_sha1: str
    src_version: str

    def matches(self, other: ZScoreStats) -> bool:
        """Same split and numerically the same moments."""
        return (
            self.train_ids_sha1 == other.train_ids_sha1
            and self.mean.shape == other.mean.shape
            and bool(np.allclose(self.mean, other.mean))
            and bool(np.allclose(self.std, other.std))
        )


def ids_digest(item_ids: Iterable[str]) -> str:
    """Order-independent SHA-1 of a set of item ids."""
    joined = "\n".join(sorted(item_ids)).encode("utf-8")
    return hashlib.sha1(joined, usedforsecurity=False).hexdigest()


def fit_zscore(
    accumulator: MomentAccumulator, train_ids_sha1: str, src_version: str
) -> ZScoreStats:
    """Freeze an accumulator's moments; raise on an empty pool or a dead stat.

    A dead stat is not clamped: D1 measured none on T2, so one appearing means the
    input changed, and dividing by an epsilon would hide that.
    """
    if accumulator.count == 0:
        raise ValueError("No rows to fit z-score statistics on")
    std = np.sqrt(accumulator.variance)
    dead = np.flatnonzero(std < constants.FLOW_ZSCORE_MIN_STD)
    if dead.size:
        names = [_stat_name(int(j)) for j in dead]
        raise ValueError(
            f"Dead flow stats (train std < {constants.FLOW_ZSCORE_MIN_STD}): {names}"
        )
    return ZScoreStats(
        mean=accumulator.mean.copy(),
        std=std,
        n_items=accumulator.items,
        n_rows=accumulator.count,
        train_ids_sha1=train_ids_sha1,
        src_version=src_version,
    )


def standardize(stats: np.ndarray, zscore: ZScoreStats) -> np.ndarray:
    """``(stats - mean) / std`` in float64, shape preserved."""
    rows = np.asarray(stats, dtype=np.float64)
    if rows.shape[-1] != zscore.mean.shape[0]:
        raise ValueError(
            f"Stats have {rows.shape[-1]} columns, z-score was fitted on "
            f"{zscore.mean.shape[0]}"
        )
    standardized: np.ndarray = (rows - zscore.mean) / zscore.std
    return standardized


def save_zscore_stats(path: Path, zscore: ZScoreStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=zscore.mean,
        std=zscore.std,
        n_items=zscore.n_items,
        n_rows=zscore.n_rows,
        train_ids_sha1=zscore.train_ids_sha1,
        src_version=zscore.src_version,
        stat_names=np.asarray(constants.FLOW_STAT_NAMES),
    )


def load_zscore_stats(path: Path) -> ZScoreStats:
    """Read :func:`save_zscore_stats` output; a missing key raises."""
    with np.load(path) as payload:
        return ZScoreStats(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            n_items=int(payload["n_items"]),
            n_rows=int(payload["n_rows"]),
            train_ids_sha1=str(payload["train_ids_sha1"]),
            src_version=str(payload["src_version"]),
        )


def _stat_name(index: int) -> str:
    names = constants.FLOW_STAT_NAMES
    return names[index] if index < len(names) else f"dim_{index}"


__all__ = [
    "MomentAccumulator",
    "ZScoreStats",
    "fit_zscore",
    "ids_digest",
    "load_zscore_stats",
    "save_zscore_stats",
    "standardize",
]
