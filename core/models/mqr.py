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

v5.1 (PREREG_V5 §3, proposal 2026-10-29 §3.2) reads the centred ``4 x 7 x 7`` grid, 196 cells:

* :class:`TileReadout` (R5c / R6c): a shared scorer ``h`` rates every cell from
  ``F = [LN(R) ; LN(Δ)]`` (:func:`tile_features`; frame-wide median removed before LN), and the
  readout pools the raw cells with ``softmax(e / T)``. ``h``'s last layer is zero, so every
  weight starts uniform and ``r_t`` is the cell mean -- R1c's function at init.
* :class:`ChangeTopKReadout` (R3c): no scorer; the ``K`` cells of largest ``|R| + |Δ|`` are
  averaged (a fixed motion ranking).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from core import constants
from core.config import V5Config
from core.data.v5_grids import load_head


@dataclass
class GridReadoutOutput:
    """What one forward of the readout returns."""

    motion: Tensor  # (B, L, D) the pre-scaled motion term c * (fc_norm(r) - m_u) / sigma_u
    attn: Tensor | None  # (B, L, M, tiles) query attention; None for the mean readout
    z: Tensor | None  # (B, L, M, D / M) per-query slices; None for the mean readout
    tile_logits: Tensor | None = None  # (B, L, cells) v5.1 scorer logits e; None otherwise


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

    def motion(self, r: Tensor) -> Tensor:
        """A2's motion term of a readout ``r``: ``c * (fc_norm(r) - m_u) / sigma_u``, float32."""
        u = F.layer_norm(r.float(), (self.dim,), self.fc_weight, self.fc_bias, self.eps)
        out: Tensor = self.c * (u - self.m_u) / self.sigma_u
        return out

    def forward(self, grid: Tensor, change: Tensor | None = None) -> GridReadoutOutput:
        r, attn, z = self.read(grid, change)
        return GridReadoutOutput(self.motion(r).to(grid.dtype), attn, z)


def tile_residuals(
    grid: Tensor, slots: int = constants.V5_SLOTS_TILE
) -> tuple[Tensor, Tensor]:
    """Raw ``R`` and ``Δ`` of each cell, ``(..., slots * tiles, D)`` float32 (proposal §3.2).

    ``R = Q - med_j Q`` over the tiles of a slot; ``Δ~_s = Q_{s+1} - Q_s`` (the last slot repeats
    ``Δ~_{S-1}``), ``Δ = Δ~ - med_j Δ~``. The median is taken per channel.
    """
    q = grid.float().unflatten(-2, (slots, -1))
    r = q - q.median(dim=-2, keepdim=True).values
    step = torch.diff(q, dim=-3)
    step = torch.cat([step, step[..., -1:, :, :]], dim=-3)
    d = step - step.median(dim=-2, keepdim=True).values
    return r.flatten(-3, -2), d.flatten(-3, -2)


def tile_features(grid: Tensor, slots: int = constants.V5_SLOTS_TILE) -> Tensor:
    """``F = [LN(R) ; LN(Δ)]``, ``(..., cells, 2D)``; LN without affine (proposal §3.2)."""
    r, d = tile_residuals(grid, slots)
    feats: Tensor = torch.cat(
        [F.layer_norm(r, (r.shape[-1],)), F.layer_norm(d, (d.shape[-1],))], dim=-1
    )
    return feats


def change_magnitude(grid: Tensor, slots: int = constants.V5_SLOTS_TILE) -> Tensor:
    """``|R| + |Δ|`` per cell, ``(..., cells)``: R3c's ranking and the untrained change map."""
    r, d = tile_residuals(grid, slots)
    mag: Tensor = r.norm(dim=-1) + d.norm(dim=-1)
    return mag


class TileScorer(nn.Module):
    """``h(F) = w_2 . GELU(W_1 F + b_1) + b_2``, shared by all cells; last layer zero (§3.2)."""

    def __init__(self, in_dim: int, hidden: int = constants.V5_TILE_HIDDEN) -> None:
        super().__init__()
        self.hidden = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats: Tensor) -> Tensor:
        """``(..., cells, 2D)`` -> logits ``e (..., cells)``."""
        logits: Tensor = self.out(F.gelu(self.hidden(feats))).squeeze(-1)
        return logits


class _CellReadout(GridReadout):
    """Shared base of the v5.1 readouts: the frozen head, ``W_V = I`` and 196 cells."""

    def __init__(self, readout: str, dim: int, cells: int, eps: float) -> None:
        super().__init__(constants.V5_READOUT_MEAN, 1, constants.V5_KEYS_CONTENT, dim, cells, eps)
        self.readout = readout
        self.value = nn.Linear(dim, dim)
        with torch.no_grad():
            self.value.weight.copy_(torch.eye(dim))
            self.value.bias.zero_()

    def pool(self, grid: Tensor, weights: Tensor) -> Tensor:
        """``W_V sum_i a_i Q_i`` -- pooling before projecting equals projecting every cell."""
        assert self.value is not None  # nosec B101 - set in __init__
        pooled = torch.einsum("bli,blid->bld", weights.to(grid.dtype), grid)
        out: Tensor = self.value(pooled)
        return out


class TileReadout(_CellReadout):
    """R5c / R6c: weights ``softmax_i(e_{t,i} / T)`` from the tile scorer (proposal §3.2)."""

    def __init__(
        self,
        dim: int = constants.V5_GRID_DIM,
        cells: int = constants.V5_TILE_CELLS,
        eps: float = constants.VIDEOMAE_LN_EPS,
    ) -> None:
        super().__init__(constants.V5_READOUT_TILE, dim, cells, eps)
        self.scorer = TileScorer(2 * dim)
        self.log_temp = nn.Parameter(torch.zeros(()))  # theta; T = exp(theta) >= T_min

    def temperature(self) -> Tensor:
        temp: Tensor = self.log_temp.exp().clamp_min(constants.V5_TILE_TEMP_MIN)
        return temp

    def forward(self, grid: Tensor, change: Tensor | None = None) -> GridReadoutOutput:
        if change is not None:
            raise ValueError("the tile readout computes its own change features")
        logits = self.scorer(tile_features(grid))
        attn = (logits.float() / self.temperature()).softmax(dim=-1)
        r = self.pool(grid, attn)
        return GridReadoutOutput(
            self.motion(r).to(grid.dtype), attn.unsqueeze(-2), None, tile_logits=logits
        )


class ChangeTopKReadout(_CellReadout):
    """R3c: the mean of the ``K`` cells of largest ``|R| + |Δ|`` per step, then ``W_V``."""

    def __init__(
        self,
        k: int = constants.V5_TILE_TOPK_CELLS,
        dim: int = constants.V5_GRID_DIM,
        cells: int = constants.V5_TILE_CELLS,
        eps: float = constants.VIDEOMAE_LN_EPS,
    ) -> None:
        super().__init__(constants.V5_READOUT_CHANGE_TOPK, dim, cells, eps)
        self.k = k

    def forward(self, grid: Tensor, change: Tensor | None = None) -> GridReadoutOutput:
        if change is not None:
            raise ValueError("the change top-K readout computes its own change features")
        with torch.no_grad():
            top = change_magnitude(grid).topk(self.k, dim=-1).indices
            weights = torch.zeros(grid.shape[:-1], device=grid.device, dtype=torch.float32)
            weights.scatter_(-1, top, 1.0 / self.k)
        r = self.pool(grid, weights)
        return GridReadoutOutput(self.motion(r).to(grid.dtype), weights.unsqueeze(-2), None)


def build_readout(v5: V5Config) -> GridReadout:
    """The readout ``cfg.v5`` names (``v5.readout != none``)."""
    if v5.readout == constants.V5_READOUT_TILE:
        return TileReadout()
    if v5.readout == constants.V5_READOUT_CHANGE_TOPK:
        return ChangeTopKReadout(v5.tile_topk_cells)
    return GridReadout(
        v5.readout, v5.queries, v5.keys, tiles=v5.grid_slots * constants.V5_GRID_TILES
    )


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
    "ChangeTopKReadout",
    "GridReadout",
    "GridReadoutOutput",
    "SpatialMILHead",
    "TileReadout",
    "TileScorer",
    "build_readout",
    "change_magnitude",
    "query_entropy",
    "query_logit_spread",
    "tile_features",
    "tile_residuals",
]
