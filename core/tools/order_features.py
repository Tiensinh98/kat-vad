"""v2.2 order diagnostics: change features, the block shuffle and the order-free twin (numpy only).

Architecture ``core/docs/v2.2/KAT_VAD_v2.2_ARCHITECTURE.md``:

* §2 ``block_permutation`` -- the adjacency-free block shuffle ``Π_b``: blocks of 5 steps (the last
  one 5-9), each reordered so that no two originally adjacent steps sit in neighbouring slots, in
  either direction, also across a block boundary; ``n <= 3`` is left unshuffled.
* §3 ``change_features`` -- κ per sequence: rate-normalised CLIP differences ``Δx̄``, their norm
  ``s``, causal max ``m5`` and trace ``e``, PCA velocity ``v̄`` (fixed causal 3-step mean) and
  acceleration ``ā``, with κ-level replicate padding at the sequence start.
* §4 ``far_ordering`` + ``twin_features`` -- the order-free twin: the whole pipeline run in an order
  whose neighbours are >= 6 original steps apart, written back, then quantile-matched per channel
  to the true-order features.

Every operation is per sequence (a source video in training, a clip at test), so nothing ever reads
across a window start or a DVS splice. Statistics fitted on T2-train live in :class:`OrderStats`.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np

from core import constants

KAPPA_PARTS = ("v_bar", "a_bar", "s", "m5", "e")  # arch §3 row 9, in this order
SUMMARY_CHANNELS = ("s", "m5", "e", "v_bar_mean", "a_bar_mean")  # S1c


@dataclass(frozen=True)
class OrderStats:
    """Fixed T2-train statistics of one stream (arch §3, §8)."""

    sigma_delta: np.ndarray  # (D,) per-channel std of Δ / m on T2-train
    basis: np.ndarray  # (d_q, D) top principal directions of Δx̄ on T2-train
    kappa_mean: np.ndarray | None = None  # (d_κ,) μ_κ, true-order κ on T2-train
    kappa_std: np.ndarray | None = None  # (d_κ,) sigma_κ


def sequence_rng(tag: str, sequence_id: str) -> np.random.Generator:
    """A generator seeded from sha1(tag, id): stable across processes and machines (O9)."""
    digest = hashlib.sha1(f"{tag}\0{sequence_id}".encode(), usedforsecurity=False).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


# ---------------------------------------------------------------------------
# §3: change features
# ---------------------------------------------------------------------------
def rate_normalised_differences(x: np.ndarray, sigma_delta: np.ndarray | None) -> np.ndarray:
    """``Δx_t / m ⊘ sigma_Δ`` for ``t >= 1`` (rows 1-3); ``m`` = median ‖Δx_t‖ over ``t >= 1``.

    ``sigma_delta=None`` returns ``Δx / m`` (used to fit ``sigma_Δ`` itself).
    """
    if len(x) < 2:
        raise ValueError(f"a sequence of {len(x)} steps has no difference")
    delta = np.diff(np.asarray(x, dtype=np.float64), axis=0)
    rate = float(np.median(np.linalg.norm(delta, axis=1)))
    if rate == 0.0:
        raise ValueError("every step-to-step change is zero: the rate scalar is undefined")
    scaled = delta / rate
    return scaled if sigma_delta is None else scaled / sigma_delta


def _causal(values: np.ndarray, width: int, reduce: str) -> np.ndarray:
    """Causal max / mean over the last ``width`` rows (available rows at the start)."""
    out = np.empty_like(values)
    for t in range(len(values)):
        window = values[max(0, t - width + 1) : t + 1]
        out[t] = window.max(axis=0) if reduce == "max" else window.mean(axis=0)
    return out


def _trace(values: np.ndarray, half_life: float) -> np.ndarray:
    decay = 2.0 ** (-1.0 / half_life)
    out = np.empty_like(values)
    out[0] = values[0]
    for t in range(1, len(values)):
        out[t] = decay * out[t - 1] + (1.0 - decay) * values[t]
    return out


def change_parts(x: np.ndarray, stats: OrderStats) -> dict[str, np.ndarray]:
    """Every κ part of one sequence, all ``(n, ·)``, replicate-padded at the start (rows 1-8)."""
    if len(x) < constants.V22_MIN_STEPS:
        raise ValueError(f"κ needs >= {constants.V22_MIN_STEPS} steps, got {len(x)} (O10)")
    dbar = rate_normalised_differences(x, stats.sigma_delta)  # (n-1, D), step t = row t-1
    s = np.linalg.norm(dbar, axis=1, keepdims=True)
    s = np.concatenate([s[:1], s])  # velocity-type: t = 0 copies t = 1
    v = np.abs(dbar @ stats.basis.T)
    v = np.concatenate([v[:1], v])
    v_bar = _causal(v, constants.V22_JITTER, "mean")
    a = np.abs(np.diff(v_bar, axis=0))[1:]  # t >= 2 (t = 1 would read the copied row)
    a = np.concatenate([a[:1], a[:1], a])  # acceleration-type: t <= 1 copies t = 2
    return {
        "v_bar": v_bar,
        "a_bar": _causal(a, constants.V22_JITTER, "mean"),
        "s": s,
        "m5": _causal(s, constants.V22_MAX_WINDOW, "max"),
        "e": _trace(s, constants.V22_TRACE_HALF_LIFE),
    }


def kappa(parts: dict[str, np.ndarray]) -> np.ndarray:
    """``κ = [v̄ ; ā ; s ; m5 ; e]`` (row 9)."""
    return np.concatenate([parts[name] for name in KAPPA_PARTS], axis=1)


def standardise(k: np.ndarray, stats: OrderStats) -> np.ndarray:
    """Row 10: the fixed per-channel standardisation baked into the cache."""
    if stats.kappa_mean is None or stats.kappa_std is None:
        raise ValueError("OrderStats has no κ statistics: fit them first (fit_kappa_stats)")
    return np.asarray((k - stats.kappa_mean) / stats.kappa_std, dtype=np.float64)


def summary_channels(parts: dict[str, np.ndarray]) -> np.ndarray:
    """S1c's five summaries: ``s``, ``m5``, ``e``, channel-mean ``v̄``, channel-mean ``ā``."""
    return np.concatenate([
        parts["s"], parts["m5"], parts["e"],
        parts["v_bar"].mean(axis=1, keepdims=True), parts["a_bar"].mean(axis=1, keepdims=True),
    ], axis=1)


def fit_stats(train: list[np.ndarray], dim: int = constants.V22_PCA_DIM) -> OrderStats:
    """``sigma_Δ`` and the PCA basis ``Q`` from every T2-train sequence (O13); no κ stats."""
    scaled = np.concatenate([rate_normalised_differences(x, None) for x in train])
    sigma = scaled.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    dbar = scaled / sigma
    _, _, vt = np.linalg.svd(dbar - dbar.mean(axis=0), full_matrices=False)
    return OrderStats(sigma_delta=sigma, basis=vt[:dim])


def fit_kappa_stats(stats: OrderStats, train: list[np.ndarray]) -> OrderStats:
    """``μ_κ``, ``sigma_κ`` over true-order κ of every T2-train sequence (row 10)."""
    rows = np.concatenate([kappa(change_parts(x, stats)) for x in train])
    std = rows.std(axis=0)
    std[std == 0.0] = 1.0
    return OrderStats(stats.sigma_delta, stats.basis, rows.mean(axis=0), std)


# ---------------------------------------------------------------------------
# §2: adjacency-free block shuffle
# ---------------------------------------------------------------------------
def _no_adjacent(perm: tuple[int, ...]) -> bool:
    return all(abs(a - b) != 1 for a, b in itertools.pairwise(perm))


@cache
def valid_block_permutations(m: int) -> tuple[tuple[int, ...], ...]:
    """Every permutation of ``range(m)`` with no originally adjacent neighbours (A002464)."""
    return tuple(p for p in itertools.permutations(range(m)) if _no_adjacent(p))


def block_sizes(n: int) -> list[int]:
    """Arch §2: blocks of 5, the last one 5 + (n mod 5); one block of 4; none below 4."""
    if n < constants.V22_MIN_SHUFFLE_BLOCK:
        return []
    if n < 2 * constants.V22_BLOCK:
        return [n]
    q = n // constants.V22_BLOCK
    return [constants.V22_BLOCK] * (q - 1) + [constants.V22_BLOCK + n % constants.V22_BLOCK]


def block_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    """``π``: slot -> original index; identity for ``n <= 3`` (arch §2)."""
    sizes = block_sizes(n)
    if not sizes:
        return np.arange(n)
    order: list[int] = []
    offset = 0
    for m in sizes:
        candidates = valid_block_permutations(m)
        while True:
            local = candidates[int(rng.integers(len(candidates)))]
            if not order or abs(order[-1] - (offset + local[0])) != 1:
                break
        order.extend(offset + i for i in local)
        offset += m
    return np.asarray(order)


# ---------------------------------------------------------------------------
# §4: the order-free twin
# ---------------------------------------------------------------------------
def _warnsdorff(n: int, gap: int, rng: np.random.Generator) -> list[int] | None:
    """One randomised attempt: always step to the far node with the fewest onward options."""
    free = np.ones(n, dtype=bool)
    current = int(rng.integers(n))
    free[current] = False
    order = [current]
    for _ in range(n - 1):
        index = np.arange(n)
        moves = index[free & (np.abs(index - current) >= gap)]
        if len(moves) == 0:
            return None
        # onward options of j = free steps outside (j - gap, j + gap); j itself is free and near
        remaining = int(free.sum())
        near = np.array([free[max(0, j - gap + 1) : j + gap].sum() for j in moves])
        degree = remaining - near
        ties = moves[degree == degree.min()]
        current = int(ties[int(rng.integers(len(ties)))])
        free[current] = False
        order.append(current)
    return order


def far_ordering(n: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """An order of ``range(n)`` whose neighbours are >= 6 apart, or the largest feasible gap."""
    for gap in range(min(constants.V22_TWIN_GAP, max(n - 1, 1)), 0, -1):
        for _ in range(constants.V22_TWIN_RESTARTS):
            order = _warnsdorff(n, gap, rng)
            if order is not None:
                return np.asarray(order), gap
    return np.arange(n), 1 if n > 1 else 0


def quantile_match(true: np.ndarray, twin: np.ndarray) -> np.ndarray:
    """Per channel: ``true``'s sorted values handed out in the rank order of ``twin`` (§4 row 4)."""
    out = np.empty_like(true)
    for j in range(true.shape[1]):
        ranks = np.argsort(np.argsort(twin[:, j], kind="stable"), kind="stable")
        out[:, j] = np.sort(true[:, j])[ranks]
    return out


def twin_rows(
    x: np.ndarray,
    true_rows: np.ndarray,
    rows_of: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Run ``rows_of`` in a far ordering, write back, quantile-match to ``true_rows``.

    ``rows_of`` maps a sequence to its per-step feature matrix (e.g. κ, or one score column).
    Returns the twin and the gap achieved.
    """
    order, gap = far_ordering(len(x), rng)
    permuted = rows_of(x[order])
    written = np.empty_like(permuted)
    written[order] = permuted
    return quantile_match(true_rows, written), gap


__all__ = [
    "KAPPA_PARTS",
    "SUMMARY_CHANNELS",
    "OrderStats",
    "block_permutation",
    "block_sizes",
    "change_parts",
    "far_ordering",
    "fit_kappa_stats",
    "fit_stats",
    "kappa",
    "quantile_match",
    "rate_normalised_differences",
    "sequence_rng",
    "standardise",
    "summary_channels",
    "twin_rows",
    "valid_block_permutations",
]
