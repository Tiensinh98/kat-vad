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

``pair_t = valid_t * valid_{t-1}`` (``pair_0 = 0``; D5; arm K5d also zeroes it at DVS joins, §12)
and ``kim_scale`` is the detached median
over valid pairs of ``mean_c |state_t - state_{t-1}|`` per bag (D6). ``w_t = (1 + cos(kin_feat_t,
mu[argmax logits_t])) / 2`` once both prototypes exist (arm K5), else 1 (arm K2). The prototypes
are buffers with no gradient, updated by :meth:`KIM.update_prototypes` (D8).

:class:`KIMCapacity` is arm Kcap: ``kin_feat = MLP(LN(v^t))`` with no differences, the same width,
scorer, ``W_out`` and losses, and about the same parameter count (``KIM_RUN.md`` §3, §6.3).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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


def fill_from_nearest(x: Tensor, have: Tensor, valid: Tensor) -> Tensor:
    """``x (B, L, F)`` with every valid row lacking a value (``valid & ~have``) set to the nearest
    row that has one, the next first, else the previous (``KIM_RUN.md`` §68 EF1). Rows with no
    such row anywhere, and invalid rows, are left as they are.
    """
    length = have.shape[1]
    idx = torch.arange(length, device=x.device).expand_as(have)
    has = have.bool()
    nxt = torch.where(has, idx, torch.full_like(idx, length))
    nxt = torch.flip(torch.cummin(torch.flip(nxt, [1]), dim=1).values, [1])
    prv = torch.cummax(torch.where(has, idx, torch.full_like(idx, -1)), dim=1).values
    src = torch.where(nxt < length, nxt, prv)
    need = valid.bool() & ~has & (src >= 0)
    src = torch.where(need, src, idx)
    return torch.gather(x, 1, src.unsqueeze(-1).expand_as(x))


def kim_scale(d1: Tensor, pair: Tensor) -> Tensor:
    """Per-bag median over valid pairs of ``mean_c d1`` (``(B,)``, detached; 1 if no pair)."""
    per_step = d1.detach().mean(dim=-1)  # (B, L)
    scales = []
    for b in range(per_step.shape[0]):
        values = per_step[b][pair[b].bool()]
        scales.append(values.median() if values.numel() else per_step.new_ones(()))
    return torch.stack(scales)


class PrivateRNG:
    """A random stream of its own for one module's dropout (``KIM_RUN.md`` §59 CE2).

    Inside :meth:`active` the device's default generator holds this stream's state; on exit the
    stream's state is kept and the caller's is put back, so the host draws exactly what it would
    have drawn without the module. Not checkpointed: a resumed run restarts the stream.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._states: dict[str, Tensor] = {}

    @staticmethod
    def _get(device: torch.device) -> Tensor:
        state: Tensor = (torch.cuda.get_rng_state(device) if device.type == "cuda"
                         else torch.get_rng_state())
        return state

    @staticmethod
    def _set(device: torch.device, state: Tensor) -> None:
        if device.type == "cuda":
            torch.cuda.set_rng_state(state, device)
        else:
            torch.set_rng_state(state)

    def _initial(self, device: torch.device) -> Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        state: Tensor = generator.get_state()
        return state

    @contextmanager
    def active(self, device: torch.device) -> Iterator[None]:
        key = str(device)
        host = self._get(device)
        self._set(device, self._states.get(key, self._initial(device)))
        try:
            yield
        finally:
            self._states[key] = self._get(device)
            self._set(device, host)


class KIM(nn.Module):
    """State / velocity / acceleration of ``v^t``, scored and returned through zero-init W_out.

    With :meth:`attach_encoder` (§59) the differences are taken on a second temporal encoder's
    output over ``source`` rows instead of on ``v^t``; the return path stays ``v^t + W_out k``.
    """

    encoder: nn.Module | None

    def __init__(
        self,
        d_model: int = constants.HIDDEN_DIM,
        width: int = constants.KIM_WIDTH,
        prototypes: bool = False,
        edge_mask: bool = False,
        edge_fill: bool = False,
    ) -> None:
        super().__init__()
        self.width = width
        self.use_prototypes = prototypes
        self.edge_mask = edge_mask
        self.edge_fill = edge_fill
        feat_dim = 3 * width
        self.norm = nn.LayerNorm(d_model)
        self._build_feature_layers(d_model, width)
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
        self.encoder = None
        self.encoder_rng: PrivateRNG | None = None

    def attach_encoder(self, encoder: nn.Module, dropout_seed: int) -> None:
        """§59 CE1/CE2: ``encoder(source, lengths)`` feeds the differences; own dropout stream.

        Called after construction, so KIM's own initial draws are those of an arm without it.
        """
        self.encoder = encoder
        self.encoder_rng = PrivateRNG(dropout_seed)

    def _encode(self, source: Tensor, lengths: Tensor) -> Tensor:
        assert self.encoder is not None and self.encoder_rng is not None  # nosec B101
        if not self.training:
            encoded: Tensor = self.encoder(source, lengths)
            return encoded
        with self.encoder_rng.active(source.device):
            encoded = self.encoder(source, lengths)
        return encoded

    def _build_feature_layers(self, d_model: int, width: int) -> None:
        self.state_proj = nn.Linear(d_model, width)
        pad = (constants.KIM_GATE_KERNEL - 1) // 2
        # §68 EF1 (c): replicate the edge row instead of zero padding (same weights, same shapes)
        mode = constants.KIM_GATE_PADDING_FILL if self.edge_fill else "zeros"
        self.vel_gate = nn.Conv1d(width, width, constants.KIM_GATE_KERNEL, padding=pad,
                                  padding_mode=mode)
        self.acc_gate = nn.Conv1d(width, width, constants.KIM_GATE_KERNEL, padding=pad,
                                  padding_mode=mode)

    def _feature_layers(self) -> tuple[nn.Module, ...]:
        return (self.state_proj, self.vel_gate, self.acc_gate)

    def reset_parameters(self) -> None:
        """Xavier weights, zero biases (gates start at 0.5), ``W_out = 0`` (D13)."""
        for layer in (*self._feature_layers(), *self.scorer):
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

    def _features(
        self, vt: Tensor, mask: Tensor, seam: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``kin_feat (B, L, 3w)`` before masking, ``scale (B,)``, velocity and acc gates."""
        pair = pair_mask(mask, drop_first=self.edge_mask)
        if seam is not None:
            pair = pair * (1.0 - seam.to(pair.dtype))
        state = self.state_proj(self.norm(vt))
        d1 = first_difference(state)
        scale = kim_scale(d1, pair)
        d1 = d1 * pair.unsqueeze(-1) / (scale[:, None, None] + constants.KIM_EPS)
        if self.edge_fill:  # §68 EF1 (a): no structural zero where a pair is masked
            d1 = fill_from_nearest(d1, pair, mask)
        vel, vel_gate = self._gated(self.vel_gate, d1, mask)
        pair2 = torch.zeros_like(pair)
        pair2[:, 1:] = pair[:, 1:] * pair[:, :-1]
        d2 = first_difference(vel) * pair2.unsqueeze(-1)
        if self.edge_fill:  # §68 EF1 (b)
            d2 = fill_from_nearest(d2, pair2, mask)
        acc, acc_gate = self._gated(self.acc_gate, d2, mask)
        return torch.cat([state, vel, acc], dim=-1), scale, vel_gate, acc_gate

    def forward(
        self,
        vt: Tensor,
        mask: Tensor,
        seam: Tensor | None = None,
        source: Tensor | None = None,
        lengths: Tensor | None = None,
    ) -> KIMOutput:
        """``vt (B, L, D)``, ``mask (B, L)``: 1 = valid, ``seam (B, L)``: 1 starts a part.

        With an attached encoder, ``source (B, L, D)`` and ``lengths (B,)`` are required: the
        features are taken on ``encoder(source)``, and ``vt`` is only the return path.
        """
        mask = mask.to(vt.dtype)
        kin_in = vt
        if self.encoder is not None:
            if source is None or lengths is None:
                raise ValueError("KIM with an input encoder needs source rows and lengths (§59)")
            kin_in = self._encode(source, lengths)
        elif source is not None:
            raise ValueError("source rows given to a KIM without an input encoder")
        raw, scale, vel_gate, acc_gate = self._features(kin_in, mask, seam)
        kin_feat = raw * mask.unsqueeze(-1)
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

    def encoder_params(self) -> list[Tensor]:
        """The attached encoder's trainable parameters (empty without one), in a fixed order."""
        if self.encoder is None:
            return []
        return [p for p in self.encoder.parameters() if p.requires_grad]


class KIMCapacity(KIM):
    """Arm Kcap: K5 with ``kin_feat = MLP(LN(v^t))``, per step, no differences and no pairs."""

    def _build_feature_layers(self, d_model: int, width: int) -> None:
        self.cap_mlp = nn.Sequential(
            nn.Linear(d_model, constants.KIM_CAP_HIDDEN),
            nn.ReLU(),
            nn.Linear(constants.KIM_CAP_HIDDEN, 3 * width),
        )

    def _feature_layers(self) -> tuple[nn.Module, ...]:
        return tuple(self.cap_mlp)

    def _features(
        self, vt: Tensor, mask: Tensor, seam: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        none = vt.new_zeros(())  # no velocity or acceleration gate exists; logged as 0
        return self.cap_mlp(self.norm(vt)), vt.new_ones(vt.shape[0]), none, none

    def w_state_norm(self) -> Tensor:
        norm: Tensor = next(self.cap_mlp.parameters()).detach().norm()  # the first layer's weight
        return norm


__all__ = ["KIM", "KIMCapacity", "KIMOutput", "fill_from_nearest", "first_difference", "kim_scale",
           "pair_mask"]
