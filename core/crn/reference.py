"""CRN references R1-R4 and the parameter-free deviation (proposal §4.2).

One implementation, shared by the E2 selection probe (``core/tools/crn_select.py``)
and, later, by the v2 model (plan P5), so the reference chosen in E2 is the one
the model trains with.

=====  =============================================================
R1     mean over all steps
R2     per-dimension median
R3     mean of the ``V2_CRN_ROBUST_KEEP`` share of steps closest (l2) to R2
R4     mean of the **strictly past** steps ``tau < t`` for ``t >= N_w``;
       for ``t < N_w`` the mean of the first ``N_w`` steps (warm-up).
       A clip shorter than ``N_w`` uses all its steps as the warm-up.
=====  =============================================================

R1-R3 are one vector per clip, broadcast over time; R4 varies with ``t`` and never
reads a step at or after ``t`` once past the warm-up.
"""

from __future__ import annotations

import numpy as np

from core import constants


def reference(x: np.ndarray, kind: str, warmup: int = constants.V2_CRN_WARMUP_STEPS) -> np.ndarray:
    """``(T, D)`` reference ``mu_t^ref`` of clip ``x`` (``(T, D)``) under ``kind``."""
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"expected a non-empty (T, D) clip, got shape {x.shape}")
    steps = x.shape[0]
    if kind == "R1":
        return np.broadcast_to(x.mean(axis=0), x.shape)
    if kind == "R2":
        return np.broadcast_to(np.median(x, axis=0), x.shape)
    if kind == "R3":
        median = np.median(x, axis=0)
        keep = max(1, int(np.ceil(constants.V2_CRN_ROBUST_KEEP * steps)))
        closest = np.argsort(np.linalg.norm(x - median, axis=1), kind="stable")[:keep]
        return np.broadcast_to(x[closest].mean(axis=0), x.shape)
    if kind == "R4":
        if warmup < 1:
            raise ValueError(f"R4 warm-up must be >= 1 step, got {warmup}")
        head = min(warmup, steps)
        out = np.empty_like(x, dtype=np.float64)
        out[:head] = x[:head].mean(axis=0)
        if steps > head:
            past_sum = np.cumsum(x, axis=0, dtype=np.float64)[head - 1 : steps - 1]
            counts = np.arange(head, steps, dtype=np.float64)[:, None]
            out[head:] = past_sum / counts  # step t: mean of x[0 .. t-1]
        return out
    raise ValueError(
        f"unknown CRN reference {kind!r}; expected one of {constants.V2_CRN_REFERENCES}"
    )


def deviation(x: np.ndarray, kind: str, warmup: int = constants.V2_CRN_WARMUP_STEPS) -> np.ndarray:
    """``d_t = ||x_t - mu_t^ref||`` (``(T,)``), the E2 parameter-free score."""
    return np.asarray(np.linalg.norm(x - reference(x, kind, warmup), axis=1), dtype=np.float64)
