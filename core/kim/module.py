"""Kinematic Intention Module (KIM) — rule ``core/docs/final/KIM_RUN.md``.

Reads ``v^t`` (the RoFormer output, ``(B, L, D)``) and returns ``v^t + W_out k_t`` for fusion
and the pre-fusion head (KIP's slot). Per step (proposal §3.2.2-§3.2.3, code names of
``KIM_RUN.md`` §1)::

    state_t  = W_state LN(v^t_t)                                  (B, L, w)
    d1_t     = pair_t * |state_t - state_{t-1}| / (kim_scale + eps)
    vel_t    = sigmoid(Conv_vel(d1))_t * d1_t
    d2_t     = pair_t * pair_{t-1} * |vel_t - vel_{t-1}|
    acc_t    = sigmoid(Conv_acc(d2))_t * d2_t
    kin_feat = [state ; vel ; acc] * valid                         (B, L, 3w)
    kim_logits = W_2 ReLU(W_1 kin_feat)                            (B, L, 2)
    kim_score  = softmax(w_t * kim_logits)[..., 1]

``pair_t = valid_t * valid_{t-1}`` (``pair_0 = 0``; D5) and ``kim_scale`` is the detached median
over valid pairs of ``mean_c |state_t - state_{t-1}|`` per bag (D6). ``w_t = (1 + cos(kin_feat_t,
mu[argmax logits_t])) / 2`` once both prototypes exist (arm K5), else 1 (arm K2). The prototypes
are buffers with no gradient, updated by :meth:`KIM.update_prototypes` (D8).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from core import constants


@dataclass
class KIMOutput:
    """One KIM forward. ``features`` replaces ``v^t`` downstream; the rest is for the losses."""

    features: Tensor  # (B, L, D) = v^t + W_out kin_feat
    kin_feat: Tensor  # (B, L, 3w), zero on padded steps
    logits: Tensor  # (B, L, 2)
    score: Tensor  # (B, L) in (0, 1)
    scale: Tensor  # (B,) detached kim_scale
    vel_gate: Tensor  # scalar, mean gate on valid steps (detached)
    acc_gate: Tensor  # scalar, mean gate on valid steps (detached)


def pair_mask(mask: Tensor, drop_first: bool = False) -> Tensor:
    """``(B, L)`` float, 1 where step t and t-1 are both valid (column 0 is always 0).

    ``drop_first`` also zeroes each bag's first valid pair (``KIM_RUN.md`` §5.3, KP3).
    """
    pair = torch.zeros_like(mask)
    pair[:, 1:] = mask[:, 1:] * mask[:, :-1]
    if drop_first:
        second_valid = (mask.argmax(dim=1) + 1).clamp(max=mask.shape[1] - 1)
        pair[torch.arange(mask.shape[0], device=mask.device), second_valid] = 0.0
    return pair


def first_difference(x: Tensor) -> Tensor:
    """``|x_t - x_{t-1}|`` along dim 1, zero at t = 0."""
    out = torch.zeros_like(x)
    out[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()
    return out


def kim_scale(d1: Tensor, pair: Tensor) -> Tensor:
    """Per-bag median over valid pairs of ``mean_c d1`` (``(B,)``, detached; 1 if no pair)."""
    per_step = d1.detach().mean(dim=-1)  # (B, L)
    scales = []
    for b in range(per_step.shape[0]):
        values = per_step[b][pair[b].bool()]
        scales.append(values.median() if values.numel() else per_step.new_ones(()))
    return torch.stack(scales)


class KIM(nn.Module):
    """State / velocity / acceleration of ``v^t``, scored and returned through zero-init W_out."""

    def __init__(
        self,
        d_model: int = constants.HIDDEN_DIM,
        width: int = constants.KIM_WIDTH,
        prototypes: bool = False,
        edge_mask: bool = False,
    ) -> None:
        super().__init__()
        self.width = width
        self.use_prototypes = prototypes
        self.edge_mask = edge_mask
        feat_dim = 3 * width
        self.norm = nn.LayerNorm(d_model)
        self.state_proj = nn.Linear(d_model, width)
        pad = (constants.KIM_GATE_KERNEL - 1) // 2
        self.vel_gate = nn.Conv1d(width, width, constants.KIM_GATE_KERNEL, padding=pad)
        self.acc_gate = nn.Conv1d(width, width, constants.KIM_GATE_KERNEL, padding=pad)
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, constants.KIM_SCORER_HIDDEN),
            nn.ReLU(),
            nn.Linear(constants.KIM_SCORER_HIDDEN, constants.KIM_CLASSES),
        )
        self.w_out = nn.Linear(feat_dim, d_model, bias=False)
        # prototypes[0] = normal, [1] = abnormal; proto_init marks which one exists (D8)
        self.register_buffer("prototypes", torch.zeros(constants.KIM_CLASSES, feat_dim))
        self.register_buffer("proto_init", torch.zeros(constants.KIM_CLASSES, dtype=torch.bool))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier weights, zero biases (gates start at 0.5), ``W_out = 0`` (D13)."""
        for layer in (self.state_proj, self.vel_gate, self.acc_gate, *self.scorer):
            if isinstance(layer, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.w_out.weight)

    @property
    def prototypes_ready(self) -> bool:
        return bool(self.use_prototypes and self.proto_init.all())

    def _gated(self, conv: nn.Conv1d, diff: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        gate = torch.sigmoid(conv(diff.transpose(1, 2)).transpose(1, 2))
        valid = mask.bool()
        mean_gate = gate.detach()[valid].mean() if valid.any() else gate.new_zeros(())
        return gate * diff, mean_gate

    def forward(self, vt: Tensor, mask: Tensor) -> KIMOutput:
        """``vt (B, L, D)``, ``mask (B, L)`` float with 1 = valid."""
        mask = mask.to(vt.dtype)
        pair = pair_mask(mask, drop_first=self.edge_mask)
        state = self.state_proj(self.norm(vt))
        d1 = first_difference(state)
        scale = kim_scale(d1, pair)
        d1 = d1 * pair.unsqueeze(-1) / (scale[:, None, None] + constants.KIM_EPS)
        vel, vel_gate = self._gated(self.vel_gate, d1, mask)
        pair2 = torch.zeros_like(pair)
        pair2[:, 1:] = pair[:, 1:] * pair[:, :-1]
        d2 = first_difference(vel) * pair2.unsqueeze(-1)
        acc, acc_gate = self._gated(self.acc_gate, d2, mask)
        kin_feat = torch.cat([state, vel, acc], dim=-1) * mask.unsqueeze(-1)
        logits = self.scorer(kin_feat)
        score = self._score(kin_feat, logits)
        return KIMOutput(
            features=vt + self.w_out(kin_feat),
            kin_feat=kin_feat,
            logits=logits,
            score=score,
            scale=scale,
            vel_gate=vel_gate,
            acc_gate=acc_gate,
        )

    def _score(self, kin_feat: Tensor, logits: Tensor) -> Tensor:
        if not self.prototypes_ready:
            return torch.softmax(logits, dim=-1)[..., 1]
        predicted = logits.detach().argmax(dim=-1)  # (B, L)
        proto = self.prototypes.to(kin_feat.dtype)[predicted]  # (B, L, F)
        confidence = 0.5 * (1.0 + F.cosine_similarity(kin_feat, proto, dim=-1))
        return torch.softmax(confidence.unsqueeze(-1) * logits, dim=-1)[..., 1]

    @torch.no_grad()
    def update_prototypes(
        self, kin_feat: Tensor, normal_steps: Tensor, anchors: Tensor, momentum: float
    ) -> None:
        """Momentum update (D8): ``mu_0`` from valid normal steps, ``mu_1`` from the anchors.

        A class absent from the batch keeps its prototype; its first appearance sets it.
        """
        feats = kin_feat.detach().float()
        for cls, chosen in ((0, normal_steps), (1, anchors)):
            picked = feats[chosen.bool()]
            if not picked.numel():
                continue
            mean = picked.mean(dim=0)
            if bool(self.proto_init[cls]):
                self.prototypes[cls].mul_(1.0 - momentum).add_(momentum * mean)
            else:
                self.prototypes[cls].copy_(mean)
                self.proto_init[cls] = True

    def prototype_cosine(self) -> Tensor:
        """``cos(mu_0, mu_1)`` (0 until both exist)."""
        if not bool(self.proto_init.all()):
            zero: Tensor = self.prototypes.new_zeros(())
            return zero
        cos: Tensor = F.cosine_similarity(self.prototypes[0], self.prototypes[1], dim=0)
        return cos

    def w_out_norm(self) -> Tensor:
        norm: Tensor = self.w_out.weight.detach().norm()
        return norm

    def w_state_norm(self) -> Tensor:
        norm: Tensor = self.state_proj.weight.detach().norm()
        return norm


__all__ = ["KIM", "KIMOutput", "first_difference", "kim_scale", "pair_mask"]
