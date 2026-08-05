"""Temporal encoder — RoFormer (RoPE) re-implementation + windowed attention.

Encoder-only port of the baseline's vendored HF RoFormer: decoder/cross-
attention, KV-cache, head pruning and gradient checkpointing are stripped;
math and module names (``layer.N.attention.self.query`` ...) are identical so
state dicts round-trip with the baseline (parity-tested) and the LaGoVAD
checkpoint maps 1:1.

:class:`TemporalEncoder` adds what the baseline keeps in ``lagovad.py`` (A8):
the window-25 local attention mask and the (optionally gated) residual
``v^t = F + encoder(F)``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from core import constants


class RoFormerSinusoidalPositionalEmbedding(nn.Module):
    """Non-interleaved sinusoidal positions: sin in the first half, cos in the second.

    Stored as a frozen ``weight`` parameter so the state-dict key matches the
    baseline's ``nn.Embedding``-based implementation.
    """

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__()
        position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
        freq_idx = torch.arange(embedding_dim, dtype=torch.float32)
        angles = position / torch.pow(10000.0, 2 * (freq_idx // 2) / embedding_dim)
        dim = embedding_dim
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        weight = torch.empty(num_positions, embedding_dim)
        weight[:, :sentinel] = torch.sin(angles[:, 0::2])
        weight[:, sentinel:] = torch.cos(angles[:, 1::2])
        self.weight = nn.Parameter(weight, requires_grad=False)

    @torch.no_grad()
    def forward(self, input_shape: torch.Size) -> Tensor:
        """``(B, L, ...)`` shape → ``(L, dim)`` position table slice."""
        return self.weight[: input_shape[1]].detach()


def apply_rotary_position_embeddings(
    sinusoidal_pos: Tensor, query_layer: Tensor, key_layer: Tensor
) -> tuple[Tensor, Tensor]:
    """RoPE on q/k: identical pairing/rotation math to the baseline."""
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
    rotate_half_q = torch.stack(
        [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
    ).reshape_as(query_layer)
    query_layer = query_layer * cos_pos + rotate_half_q * sin_pos
    rotate_half_k = torch.stack(
        [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
    ).reshape_as(key_layer)
    key_layer = key_layer * cos_pos + rotate_half_k * sin_pos
    return query_layer, key_layer


class RoFormerSelfAttention(nn.Module):
    """Multi-head self-attention with rotary position embeddings on q/k."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _transpose_for_scores(self, x: Tensor) -> Tensor:
        new_shape = (*x.size()[:-1], self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        sinusoidal_pos: Tensor,
    ) -> Tensor:
        query_layer = self._transpose_for_scores(self.query(hidden_states))
        key_layer = self._transpose_for_scores(self.key(hidden_states))
        value_layer = self._transpose_for_scores(self.value(hidden_states))
        query_layer, key_layer = apply_rotary_position_embeddings(
            sinusoidal_pos, query_layer, key_layer
        )

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = self.dropout(torch.softmax(scores, dim=-1))

        context = torch.matmul(probs, value_layer).permute(0, 2, 1, 3).contiguous()
        return context.view(*context.size()[:-2], -1)


class RoFormerSelfOutput(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        out: Tensor = self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)
        return out


class RoFormerAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, dropout: float, eps: float
    ) -> None:
        super().__init__()
        self.self = RoFormerSelfAttention(hidden_size, num_heads, dropout)
        self.output = RoFormerSelfOutput(hidden_size, dropout, eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        sinusoidal_pos: Tensor,
    ) -> Tensor:
        context = self.self(hidden_states, attention_mask, sinusoidal_pos)
        out: Tensor = self.output(context, hidden_states)
        return out


class RoFormerIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        out: Tensor = nn.functional.gelu(self.dense(hidden_states))
        return out


class RoFormerOutput(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, dropout: float, eps: float
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        out: Tensor = self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)
        return out


class RoFormerLayer(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int,
        dropout: float, eps: float,
    ) -> None:
        super().__init__()
        self.attention = RoFormerAttention(hidden_size, num_heads, dropout, eps)
        self.intermediate = RoFormerIntermediate(hidden_size, intermediate_size)
        self.output = RoFormerOutput(hidden_size, intermediate_size, dropout, eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        sinusoidal_pos: Tensor,
    ) -> Tensor:
        attention_output = self.attention(hidden_states, attention_mask, sinusoidal_pos)
        out: Tensor = self.output(self.intermediate(attention_output), attention_output)
        return out


class RoFormerEncoder(nn.Module):
    """Stack of RoPE transformer layers; returns the last hidden state."""

    def __init__(
        self,
        hidden_size: int = constants.HIDDEN_DIM,
        num_layers: int = constants.TEMPORAL_LAYERS,
        num_heads: int = constants.TEMPORAL_HEADS,
        max_position_embeddings: int = constants.TEMPORAL_MAX_POSITIONS,
        dropout: float = constants.TEMPORAL_DROPOUT,
        layer_norm_eps: float = constants.LAYER_NORM_EPS,
    ) -> None:
        super().__init__()
        intermediate_size = hidden_size * 4
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            max_position_embeddings, hidden_size // num_heads
        )
        self.layer = nn.ModuleList(
            [
                RoFormerLayer(hidden_size, num_heads, intermediate_size, dropout, layer_norm_eps)
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        sinusoidal_pos: Tensor = self.embed_positions(hidden_states.shape[:-1])[
            None, None, :, :
        ]
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, sinusoidal_pos)
        return hidden_states


def extended_attention_mask(attention_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """``(B, L)`` 1/0 padding mask → additive ``(B, 1, 1, L)`` mask (0 / dtype-min)."""
    extended = attention_mask[:, None, None, :].to(dtype=dtype)
    inverted: Tensor = 1.0 - extended
    return inverted * torch.finfo(dtype).min


def extended_local_attention_mask(
    attention_mask: Tensor,
    local_window_size: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Additive band mask: each position attends within ±(window-1)/2 valid positions.

    Port of the baseline's ``get_extended_local_attention_mask``:
    ``(B, L)`` 1/0 padding mask → ``(B, 1, L, L)`` additive mask.
    """
    device = attention_mask.device
    length = attention_mask.shape[-1]
    half_window = (local_window_size - 1) // 2
    band = torch.triu(torch.ones(length, length, device=device), -half_window)
    band = band * torch.tril(torch.ones(length, length, device=device), half_window)
    combined = attention_mask[:, None, None, :].to(dtype=dtype) * band[None, None, :, :]
    inverted: Tensor = 1.0 - combined
    return inverted * torch.finfo(dtype).min


class TemporalEncoder(nn.Module):
    """``F (B, L, D)`` + lengths → ``v^t (B, L, D)`` (windowed RoFormer + residual)."""

    def __init__(
        self,
        hidden_size: int = constants.HIDDEN_DIM,
        num_layers: int = constants.TEMPORAL_LAYERS,
        num_heads: int = constants.TEMPORAL_HEADS,
        window_size: int = constants.TEMPORAL_WINDOW,
        max_position_embeddings: int = constants.TEMPORAL_MAX_POSITIONS,
        use_gate: bool = False,
        gate_weight: float = constants.TEMP_GATE_WEIGHT,
        gate_init: float = constants.TEMP_GATE_INIT,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.encoder = RoFormerEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
        )
        self.gate_weight = gate_weight
        self.gate_alpha: nn.Parameter | None = (
            nn.Parameter(torch.full((1,), gate_init)) if use_gate else None
        )

    def forward(self, v_feat: Tensor, lengths: Tensor) -> Tensor:
        pad_mask = (
            torch.arange(v_feat.shape[1], device=v_feat.device)[None, :]
            < lengths[:, None].to(v_feat.device)
        ).to(v_feat.dtype)
        if self.window_size and self.window_size > 0:
            attn_mask = extended_local_attention_mask(
                pad_mask, self.window_size, dtype=v_feat.dtype
            )
        else:
            attn_mask = extended_attention_mask(pad_mask, dtype=v_feat.dtype)
        encoded: Tensor = self.encoder(v_feat, attn_mask)
        if self.gate_alpha is not None:
            return v_feat + encoded * torch.tanh(self.gate_alpha) * self.gate_weight
        return v_feat + encoded


__all__ = [
    "RoFormerEncoder",
    "TemporalEncoder",
    "extended_attention_mask",
    "extended_local_attention_mask",
]
