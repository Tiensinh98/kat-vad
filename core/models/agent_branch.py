"""v6 Agent Prompt (AP) branch (proposal §3.2, PREREG_V6 §2.2).

Per model step the candidates are the ``K`` agent slots of the agent cache, an ego token and a
no-agent token (valid only when the step has no agent). Each agent token sums a Fourier encoding
of its box corners, an agent-type and class embedding, an MLP of (log-area, aspect, confidence,
ego distance), ``W_a LN(crop)``, ``W_m LN(RoI motion)`` and an MLP of its kinematics. One instance
self-attention layer with an additive pairwise bias runs over the candidates; a relation module
(``N_K`` learnable queries, ``L_R`` blocks: queries <- agents, queries <- scene, agents <- queries)
builds the frame token ``z_t`` (mean of the queries). A shared scorer gives one logit per
candidate; its last layer is zero, so every probability starts at 0.5.

The branch has no time index: it sees one step at a time (temporal position reaches the model only
through the host's RoFormer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from core import constants
from core.agents import layout

EGO, NONE = 0, 1  # offsets of the ego / no-agent candidates after the K agent slots
GEOM_DIM = 4  # log-area, aspect, confidence, ego distance
PAIR_HIDDEN = 32


@dataclass(frozen=True)
class BranchOutput:
    z: Tensor  # (B, L, d) frame token
    logits: Tensor  # (B, L, K + 2) candidate logits (agents, ego, no-agent)
    candidates: Tensor  # (B, L, K + 2) bool: which candidates exist


class Fourier(nn.Module):
    """SAM-style random Fourier features of a 2-D point in ``[0, 1]``: ``(..., 2) -> (..., d)``."""

    def __init__(self, d: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(constants.V6_FOURIER_SEED)
        freqs = torch.randn(2, d // 2, generator=generator) * constants.V6_FOURIER_SCALE
        self.register_buffer("freqs", freqs)

    def forward(self, point: Tensor) -> Tensor:
        proj = 2 * math.pi * (2 * point - 1) @ self.freqs
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


def _ffn(d: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d),
        nn.Linear(d, d * constants.V6_FFN_RATIO),
        nn.GELU(),
        nn.Linear(d * constants.V6_FFN_RATIO, d),
    )


def pair_features(boxes: Tensor) -> Tensor:
    """``(N, T, 4)`` boxes -> ``(N, T, T, 6)``: box difference, IoU and centre gap of each pair."""
    diff = boxes[:, None, :, :] - boxes[:, :, None, :]
    lt = torch.maximum(boxes[:, :, None, :2], boxes[:, None, :, :2])
    rb = torch.minimum(boxes[:, :, None, 2:], boxes[:, None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    area = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)
    union = area[:, :, None] + area[:, None, :] - inter
    iou = torch.where(union > 0, inter / union.clamp_min(1e-9), torch.zeros_like(inter))
    c = (boxes[..., :2] + boxes[..., 2:]) / 2
    gap = (c[:, :, None, :] - c[:, None, :, :]).norm(dim=-1)
    return torch.cat([diff, iou[..., None], gap[..., None]], dim=-1)


class BiasedSelfAttention(nn.Module):
    """Pre-LN self-attention over a step's candidates with an additive pairwise bias, then FFN."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.bias = nn.Sequential(
            nn.Linear(constants.V6_PAIR_DIM, PAIR_HIDDEN), nn.GELU(), nn.Linear(PAIR_HIDDEN, heads)
        )
        self.ffn = _ffn(d)

    def forward(self, x: Tensor, pairs: Tensor, valid: Tensor) -> Tensor:
        n, t, d = x.shape
        qkv = self.qkv(self.norm(x)).reshape(n, t, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        bias = self.bias(pairs).permute(0, 3, 1, 2)  # (N, heads, T, T)
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        attn = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.out(attn.transpose(1, 2).reshape(n, t, d))
        out: Tensor = x + self.ffn(x)
        return out


class CrossAttention(nn.Module):
    """Pre-LN residual cross-attention ``x <- context``."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, x: Tensor, context: Tensor, valid: Tensor | None = None) -> Tensor:
        kv = self.norm_kv(context)
        out, _ = self.attn(
            self.norm_q(x),
            kv,
            kv,
            key_padding_mask=None if valid is None else ~valid,
            need_weights=False,
        )
        result: Tensor = x + out
        return result


class RelationBlock(nn.Module):
    """PromptTAD-style relation block: queries <- agents, queries <- scene, agents <- queries."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.q_agents = CrossAttention(d, heads)
        self.q_scene = CrossAttention(d, heads)
        self.agents_q = CrossAttention(d, heads)
        self.ffn_q = _ffn(d)
        self.ffn_o = _ffn(d)

    def forward(self, q: Tensor, o: Tensor, scene: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        q = self.q_agents(q, o, valid)
        q = self.q_scene(q, scene)
        o = self.agents_q(o, q)
        return q + self.ffn_q(q), o + self.ffn_o(o)


class AgentBranch(nn.Module):
    """Agent tokens -> instance self-attention -> relations -> ``z_t`` and candidate logits."""

    def __init__(
        self,
        clip_dim: int = constants.HIDDEN_DIM,
        motion_dim: int = constants.V5_GRID_DIM,
        d: int = constants.V6_D,
        heads: int = constants.V6_HEADS,
        queries: int = constants.V6_QUERIES,
        blocks: int = constants.V6_RELATION_BLOCKS,
        k_max: int = constants.V6_K_MAX,
        appearance: bool = True,
        motion: bool = True,
        relation: bool = True,
    ) -> None:
        super().__init__()
        self.clip_dim, self.motion_dim, self.d, self.k_max = clip_dim, motion_dim, d, k_max
        self.appearance, self.motion, self.relation = appearance, motion, relation
        self.fourier = Fourier(d)
        self.e_agent = nn.Parameter(torch.zeros(d))
        self.e_ego = nn.Parameter(torch.zeros(d))
        self.e_none = nn.Parameter(torch.zeros(d))
        for p in (self.e_agent, self.e_ego, self.e_none):
            nn.init.normal_(p, std=0.02)
        self.cls_embed = nn.Embedding(len(constants.V6_CLASS_NAMES), d)
        self.geom = nn.Sequential(nn.Linear(GEOM_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.kin = nn.Sequential(nn.Linear(constants.V6_KIN_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.ego_mlp = nn.Sequential(nn.Linear(constants.V6_EGO_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.crop = nn.Sequential(
            nn.LayerNorm(constants.V6_AGENT_CROP_DIM), nn.Linear(constants.V6_AGENT_CROP_DIM, d)
        )
        self.roi = nn.Sequential(
            nn.LayerNorm(constants.V6_AGENT_MOTION_DIM), nn.Linear(constants.V6_AGENT_MOTION_DIM, d)
        )
        self.scene = nn.Sequential(
            nn.LayerNorm(clip_dim + motion_dim), nn.Linear(clip_dim + motion_dim, d)
        )
        self.instance = BiasedSelfAttention(d, heads)
        self.query = nn.Parameter(torch.randn(queries, d) * 0.02)
        self.blocks = (
            nn.ModuleList(RelationBlock(d, heads) for _ in range(blocks)) if relation else None
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, constants.V6_SCORER_HIDDEN),
            nn.GELU(),
            nn.Linear(constants.V6_SCORER_HIDDEN, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)
        self.register_buffer("ego_box", torch.tensor(constants.V6_EGO_BOX, dtype=torch.float32))

    # ------------------------------------------------------------------ tokens
    def agent_tokens(self, a: Tensor) -> Tensor:
        """``(N, K, AGENT_DIM)`` -> ``(N, K, d)``."""
        box = a[..., layout.BOX]
        wh = (box[..., 2:] - box[..., :2]).clamp_min(1e-4)
        c = (box[..., :2] + box[..., 2:]) / 2
        ego = torch.tensor(constants.V6_EGO_POINT, device=a.device, dtype=a.dtype)
        geom = torch.stack(
            [
                wh.prod(-1).log(),
                (wh[..., 0] / wh[..., 1]).log(),
                a[..., layout.CONF],
                (c - ego).norm(dim=-1),
            ],
            dim=-1,
        )
        cls = a[..., layout.CLASS].round().long().clamp(0, len(constants.V6_CLASS_NAMES) - 1)
        tok = (
            self.fourier(box[..., :2])
            + self.fourier(box[..., 2:])
            + self.e_agent
            + self.cls_embed(cls)
            + self.geom(geom)
            + self.kin(a[..., layout.KIN])
        )
        if self.appearance:
            tok = tok + self.crop(a[..., layout.CROP])
        if self.motion:
            tok = tok + self.roi(a[..., layout.MOTION])
        out: Tensor = tok
        return out

    # ------------------------------------------------------------------ forward
    def forward(self, base: Tensor, agents: Tensor, ego: Tensor, mask: Tensor) -> BranchOutput:
        """``base (B, L, clip+motion)``, ``agents (B, L, K, AGENT_DIM)``, ``ego (B, L, 48)``,
        ``mask (B, L)`` 1 = valid step."""
        b, length, k, _ = agents.shape
        t = k + 2
        step = mask.bool()
        z = base.new_zeros(b, length, self.d)
        logits = base.new_zeros(b, length, t)
        cand = torch.zeros(b, length, t, dtype=torch.bool, device=base.device)
        if not step.any():
            return BranchOutput(z, logits, cand)
        a, e, s = agents[step].float(), ego[step].float(), base[step].float()  # (N, ...)
        n = a.shape[0]
        has = a[..., layout.VALID] > 0.5
        valid = torch.cat(
            [has, torch.ones(n, 1, dtype=torch.bool, device=a.device), ~has.any(-1, keepdim=True)],
            dim=-1,
        )
        ego_tok = (
            self.fourier(self.ego_box[:2].expand(n, 2))
            + self.fourier(self.ego_box[2:].expand(n, 2))
            + self.e_ego
            + self.ego_mlp(e)
        )
        tokens = torch.cat(
            [self.agent_tokens(a), ego_tok[:, None], self.e_none.expand(n, 1, self.d)], dim=1
        )
        boxes = torch.cat(
            [
                a[..., layout.BOX],
                self.ego_box.expand(n, 1, 4),
                torch.zeros(n, 1, 4, device=a.device),
            ],
            dim=1,
        )
        o = self.instance(tokens, pair_features(boxes), valid)
        scene = self.scene(s)[:, None]
        if self.blocks is not None:
            q = self.query.expand(n, -1, -1)
            for block in self.blocks:
                q, o = block(q, o, scene, valid)
            frame = q.mean(1)
        else:  # ladder row 4: z_t = score-weighted mean of the candidate tokens
            w = torch.sigmoid(self.scorer(o)[..., 0]).masked_fill(~valid, 0.0)
            frame = (w[..., None] * o).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-6)
        z[step] = frame.to(z.dtype)
        logits[step] = self.scorer(o)[..., 0].to(logits.dtype)
        cand[step] = valid
        return BranchOutput(z, logits, cand)


class AgentGate(nn.Module):
    """``x_t + tanh(alpha_o) W_o LN(z_t)``, ``alpha_o`` = 0 at init (proposal §3.2 Fusion)."""

    def __init__(self, hidden_dim: int, d: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, hidden_dim)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor, z: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        term = torch.tanh(self.alpha) * self.proj(self.norm(z))
        term = term.masked_fill(~mask.bool()[..., None], 0.0)
        with torch.no_grad():
            valid = mask.bool()
            share = term.detach()[valid].norm() / x.detach()[valid].norm().clamp_min(
                constants.V2_NORM_EPS
            )
        return x + term, share


__all__ = ["AgentBranch", "AgentGate", "BranchOutput", "Fourier", "pair_features"]
