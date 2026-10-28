"""v5 LaGoVAD-MQR: a readout of the V7 token grid that feeds v2's motion path (PREREG_V5 §2).

Per step ``t`` the grid ``G_t`` holds 49 tiles of 384 channels (V2-S's last layer before
``fc_norm``). The readout ``r_t`` replaces the token mean that produced A2's ``u``:

* ``mean`` (R1): ``r_t`` = the tile mean -- exactly the mean of all tokens, so ``fc_norm(r_t)``
  is A2's ``u`` and the row computes A2's function (S1-G1 / S1-G3);
* ``mqr`` (R3 / R5 / R5Δ, S2-4): ``M`` learned queries attend over the tiles,
  ``a_{t,m,j} = softmax_j(u_m . K_{t,j} / sqrt(d_K))`` with keys ``LN(G)`` (content) or
  ``LN([G ; ΔG])`` (change, R5Δ) and no affine LN; values are always ``G``.
  ``z_{t,m} = [W_V sum_j a_{t,m,j} G_{t,j}]_m`` (the ``m``-th ``D/M`` slice),
  ``r_t = [z_1; ...; z_M]``.

Init ``u_m = 0``, ``W_V = I``, bias 0 makes every query uniform and ``r_t`` the tile mean, so every
row starts as R1 (S1-G2). The motion term is then v2's (deviation D1)::

    c * (fc_norm(r_t) - m_u) / sigma_u        -> MotionResidual (W_u zero-init)

with V2-S's frozen ``fc_norm`` and Nexar A2's baked ``c``, ``m_u``, ``sigma_u`` held as buffers
(:meth:`GridReadout.load_head`), so a checkpoint carries them.

:class:`SpatialMILHead` scores each query, ``s_{t,m} = sigmoid(g(z_{t,m}))``, ``g`` shared (S2-6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from core import constants
from core.data.v5_grids import load_head


@dataclass
class GridReadoutOutput:
    """What one forward of the readout returns."""

    motion: Tensor  # (B, L, D) the pre-scaled motion term c * (fc_norm(r) - m_u) / sigma_u
    attn: Tensor | None  # (B, L, M, tiles) query attention; None for the mean readout
    z: Tensor | None  # (B, L, M, D / M) per-query slices; None for the mean readout


class GridReadout(nn.Module):
    """``G_t (B, L, tiles, D)`` -> A2's motion term from a mean or query readout."""

    def __init__(
        self,
        readout: str,
        queries: int = 1,
        keys: str = constants.V5_KEYS_CONTENT,
        dim: int = constants.V5_GRID_DIM,
        tiles: int = constants.V5_GRID_TILES,
        eps: float = constants.VIDEOMAE_LN_EPS,
    ) -> None:
        super().__init__()
        if readout not in (constants.V5_READOUT_MEAN, constants.V5_READOUT_MQR):
            raise ValueError(f"GridReadout needs readout mean or mqr, got {readout!r}")
        if dim % queries:
            raise ValueError(f"{queries} queries do not split {dim} channels")
        self.readout = readout
        self.num_queries = queries
        self.keys = keys
        self.dim = dim
        self.tiles = tiles
        self.eps = eps
        # frozen motion head (V2-S fc_norm + A2's statistics); identity until load_head
        self.register_buffer("fc_weight", torch.ones(dim))
        self.register_buffer("fc_bias", torch.zeros(dim))
        self.register_buffer("c", torch.ones(()))
        self.register_buffer("m_u", torch.zeros(dim))
        self.register_buffer("sigma_u", torch.ones(dim))
        self.register_buffer("head_loaded", torch.zeros((), dtype=torch.bool))
        self.query: nn.Parameter | None = None
        self.value: nn.Linear | None = None
        if readout == constants.V5_READOUT_MQR:
            key_dim = 2 * dim if keys == constants.V5_KEYS_CHANGE else dim
            self.query = nn.Parameter(torch.zeros(queries, key_dim))
            self.value = nn.Linear(dim, dim)
            with torch.no_grad():
                self.value.weight.copy_(torch.eye(dim))
                self.value.bias.zero_()

    @property
    def uses_change(self) -> bool:
        return self.readout == constants.V5_READOUT_MQR and self.keys == constants.V5_KEYS_CHANGE

    def load_head(self, path: Path) -> None:
        """Fill the frozen buffers from a :data:`constants.V5_HEAD_FILENAME` file."""
        head = load_head(path)
        with torch.no_grad():
            self.fc_weight.copy_(torch.from_numpy(head["fc_weight"]).float())
            self.fc_bias.copy_(torch.from_numpy(head["fc_bias"]).float())
            self.c.fill_(float(head["c"]))
            self.m_u.copy_(torch.from_numpy(head["m_u"]).float())
            self.sigma_u.copy_(torch.from_numpy(head["sigma_u"]).float())
            self.head_loaded.fill_(True)
        if abs(float(head["eps"]) - self.eps) > 0:
            raise ValueError(f"{path}: fc_norm eps {float(head['eps'])} != {self.eps}")

    def attention(self, grid: Tensor, change: Tensor | None) -> Tensor:
        """``(B, L, M, tiles)`` softmax over tiles of ``u_m . LN(K) / sqrt(d_K)``."""
        if self.query is None:
            raise RuntimeError("the mean readout has no attention")
        if self.uses_change != (change is not None):
            raise ValueError(f"keys={self.keys!r} but ΔG {'absent' if change is None else 'given'}")
        keys = grid if change is None else torch.cat([grid, change], dim=-1)
        keys = F.layer_norm(keys, (keys.shape[-1],))
        logits = torch.einsum("bltk,mk->blmt", keys, self.query.to(keys.dtype))
        attn: Tensor = (logits / math.sqrt(keys.shape[-1])).softmax(dim=-1)
        return attn

    def read(
        self, grid: Tensor, change: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """``r_t (B, L, D)``, attention and per-query slices (S2-4); S1-G2(a) compares ``r_t``."""
        if self.readout == constants.V5_READOUT_MEAN:
            return grid.mean(dim=-2), None, None
        assert self.value is not None  # nosec B101 - narrowing for type-checkers
        attn = self.attention(grid, change)
        pooled = torch.einsum("blmt,bltd->blmd", attn, grid)
        per = self.dim // self.num_queries
        weight = self.value.weight.view(self.num_queries, per, self.dim).to(pooled.dtype)
        bias = self.value.bias.view(self.num_queries, per).to(pooled.dtype)
        z = torch.einsum("blmd,med->blme", pooled, weight) + bias
        return z.reshape(*z.shape[:-2], self.dim), attn, z

    def forward(self, grid: Tensor, change: Tensor | None = None) -> GridReadoutOutput:
        r, attn, z = self.read(grid, change)
        u = F.layer_norm(r.float(), (self.dim,), self.fc_weight, self.fc_bias, self.eps)
        motion = self.c * (u - self.m_u) / self.sigma_u
        return GridReadoutOutput(motion.to(grid.dtype), attn, z)


class SpatialMILHead(nn.Module):
    """``g``: one linear score per query slice, shared across queries (S2-6)."""

    def __init__(self, slice_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(slice_dim, 1)

    def forward(self, z: Tensor) -> Tensor:
        """``z (B, L, M, D/M)`` -> ``s (B, L, M)`` in ``(0, 1)``."""
        probs: Tensor = self.score(z).squeeze(-1).sigmoid()
        return probs


def query_entropy(attn: Tensor, mask: Tensor) -> Tensor:
    """Mean attention entropy per query over valid steps, ``(M,)``, detached (S1-G6)."""
    with torch.no_grad():
        ent = -(attn.float().clamp_min(torch.finfo(torch.float32).tiny).log() * attn.float()).sum(
            -1
        )
        valid = mask.bool()[..., None].expand_as(ent)
        per_query: Tensor = ent.masked_fill(~valid, 0.0).sum((0, 1)) / valid.sum((0, 1)).clamp_min(
            1
        )
    return per_query


def query_logit_spread(attn: Tensor, mask: Tensor) -> Tensor:
    """Mean over valid steps of the std over tiles of each query's logits, ``(M,)`` (§2.4 B13).

    ``log a`` equals the logits up to a per-step constant, so its std over tiles is the logits'.
    0 for a uniform query (Stage 2's R5 / R5Δ read about 0.05).
    """
    with torch.no_grad():
        logits = attn.float().clamp_min(torch.finfo(torch.float32).tiny).log()
        spread = logits.std(dim=-1, unbiased=False)
        valid = mask.bool()[..., None].expand_as(spread)
        per_query: Tensor = spread.masked_fill(~valid, 0.0).sum((0, 1)) / valid.sum(
            (0, 1)
        ).clamp_min(1)
    return per_query


__all__ = [
    "GridReadout",
    "GridReadoutOutput",
    "SpatialMILHead",
    "query_entropy",
    "query_logit_spread",
]
