"""v7 Latent Surprise Predictor (LSP): forecast the V2-S vector one non-overlapping clip ahead.

Rules: ``core/docs/v7/PREREG_V7.md`` §1; proposal §3.2.2-§3.2.3. Everything runs on a fixed 10 fps
lattice of one source video (X6): ``u`` is ``(N, D)`` with consecutive rows 0.1 s apart.

* ``u_t = LN_no-affine(mean tokens)``, recovered from the cached ``fc_norm(mean)`` as
  ``(v - b) / gamma`` (:func:`unit_vectors`, X5).
* ``û_t = u_{t-Δ} + g(u_{t-Δ-K+1}, ..., u_{t-Δ})``; ``g`` = 2 pre-LN causal transformer layers
  (width D, 6 heads, MLP 4x), learned positions over K, a linear readout of the last position
  that starts at zero -- so the predictor starts as copy-last (proposal §3.2.3).
* ``s_t = mean_d l(u_t - û_t)``, ``p_t = mean_d l(u_t - u_{t-Δ})`` (Smooth-L1, β = 1),
  ``rho_t = log((s_t + ε) / (p_t + ε))``; a target is valid iff its full context exists
  (``t >= Δ + K - 1``).
* ``e_t = m_t [LN(r_t); ŝ_t; rho_t]`` (:func:`evidence_vector`), the stage-2 input (P2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from core import constants


def unit_vectors(cached: np.ndarray, fc_weight: np.ndarray, fc_bias: np.ndarray) -> np.ndarray:
    """``LN_no-affine(mean tokens)`` from the cached ``fc_norm(mean tokens)`` (X5).

    ``fc_norm(m) = gamma * LN(m) + b`` with ``LN`` affine-free, so ``LN(m) = (v - b) / gamma``
    exactly (up to the cache's float precision).
    """
    if np.any(np.abs(fc_weight) < constants.V7_EPS):
        raise ValueError("fc_norm weight has a ~0 entry; LN(mean) is not recoverable")
    out: np.ndarray = (np.asarray(cached, dtype=np.float32) - fc_bias) / fc_weight
    return out.astype(np.float32)


def clip_unit_vectors(cached: np.ndarray) -> np.ndarray:
    """``LN_no-affine(c)`` of CLIP image embeddings (PREREG_V7 §1.1): the V2-S rule of L0-2."""
    c = torch.as_tensor(np.asarray(cached, dtype=np.float32))
    out: np.ndarray = F.layer_norm(c, (c.shape[-1],)).numpy()
    return out


class LatentSurprisePredictor(nn.Module):
    """``g``: K context vectors ``(B, K, D)`` -> a correction ``(B, D)`` added to copy-last."""

    def __init__(
        self,
        dim: int = constants.V7_DIM,
        context: int = constants.V7_CONTEXTS[0],
        layers: int = constants.V7_LAYERS,
        heads: int = constants.V7_HEADS,
        mlp_ratio: int = constants.V7_MLP_RATIO,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.context = context
        self.pos = nn.Parameter(torch.zeros(context, dim))
        nn.init.trunc_normal_(self.pos, std=constants.V7_POS_INIT_STD)
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.readout = nn.Linear(dim, dim)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer(
            "causal", nn.Transformer.generate_square_subsequent_mask(context), persistent=False
        )

    def forward(self, ctx: Tensor) -> Tensor:
        if ctx.shape[-2:] != (self.context, self.dim):
            raise ValueError(f"context {tuple(ctx.shape)} is not (B, {self.context}, {self.dim})")
        h = self.encoder(ctx + self.pos, mask=self.causal, is_causal=True)
        out: Tensor = self.readout(self.norm(h[:, -1]))
        return out

    def predict(self, ctx: Tensor) -> Tensor:
        """``û = u_{t-Δ} + g(ctx)``; ``ctx[:, -1]`` is ``u_{t-Δ}``."""
        out: Tensor = ctx[:, -1] + self(ctx)
        return out


def first_target(context: int, horizon: int = constants.V7_HORIZON) -> int:
    """First lattice index whose full context exists: ``Δ + K - 1``."""
    return horizon + context - 1


def context_windows(u: Tensor, context: int, horizon: int = constants.V7_HORIZON) -> Tensor:
    """``(N, D)`` -> ``(N - Δ - K + 1, K, D)``; window ``i`` predicts target ``i + Δ + K - 1``."""
    n = u.shape[0] - horizon - context + 1
    if n <= 0:
        return u.new_zeros((0, context, u.shape[1]))
    out: Tensor = u[: u.shape[0] - horizon].unfold(0, context, 1).permute(0, 2, 1)
    return out


def row_loss(pred: Tensor, target: Tensor, beta: float = constants.V7_SMOOTH_L1_BETA) -> Tensor:
    """Smooth-L1 averaged over the feature axis: one value per row."""
    out: Tensor = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(-1)
    return out


@dataclass
class SurpriseSeries:
    """Per-lattice-step evidence of one sequence; invalid steps hold 0 (and ``valid`` False)."""

    s: np.ndarray  # (N,) LSP surprise
    p: np.ndarray  # (N,) copy-last surprise
    rho: np.ndarray  # (N,) log((s + ε) / (p + ε))
    r: np.ndarray  # (N, D) residual u - û
    valid: np.ndarray  # (N,) bool: t >= Δ + K - 1


@torch.no_grad()
def surprise_series(
    model: LatentSurprisePredictor,
    u: np.ndarray,
    horizon: int = constants.V7_HORIZON,
    batch: int = constants.V7_BATCH * 4,
) -> SurpriseSeries:
    """Run the frozen predictor over every valid target of one lattice sequence ``u (N, D)``."""
    n, dim = u.shape
    device = next(model.parameters()).device
    s, p = np.zeros(n, np.float32), np.zeros(n, np.float32)
    r = np.zeros((n, dim), np.float32)
    valid = np.zeros(n, bool)
    first = first_target(model.context, horizon)
    ut = torch.as_tensor(u, dtype=torch.float32, device=device)
    windows = context_windows(ut, model.context, horizon)
    for lo in range(0, windows.shape[0], batch):
        ctx = windows[lo : lo + batch]
        target = ut[first + lo : first + lo + ctx.shape[0]]
        pred = model.predict(ctx)
        span = slice(first + lo, first + lo + ctx.shape[0])
        s[span] = row_loss(pred, target).cpu().numpy()
        p[span] = row_loss(ctx[:, -1], target).cpu().numpy()
        r[span] = (target - pred).cpu().numpy()
        valid[span] = True
    rho = np.where(valid, np.log((s + constants.V7_EPS) / (p + constants.V7_EPS)), 0.0)
    return SurpriseSeries(s=s, p=p, rho=rho.astype(np.float32), r=r, valid=valid)


def evidence_vector(series: SurpriseSeries, s_mean: float, s_std: float) -> np.ndarray:
    """``e_t = m_t [LN(r_t); ŝ_t; rho_t]`` -> ``(N, D + 2)``, ``LN`` affine-free (§3.2.2)."""
    r = torch.as_tensor(series.r)
    ln = F.layer_norm(r, (r.shape[-1],)).numpy()
    s_hat = (series.s - s_mean) / max(s_std, constants.V7_EPS)
    out = np.concatenate([ln, s_hat[:, None], series.rho[:, None]], axis=1).astype(np.float32)
    out[~series.valid] = 0.0
    result: np.ndarray = out
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "LatentSurprisePredictor",
    "SurpriseSeries",
    "clip_unit_vectors",
    "context_windows",
    "evidence_vector",
    "first_target",
    "parameter_count",
    "row_loss",
    "surprise_series",
    "unit_vectors",
]
