"""Detection heads — port of the baseline's ``heads.py``.

``H_bin`` = :class:`BinaryHead` (adaptive fuse of pre-/post-fusion conv scores,
α₀ = 0.25, scale 10). ``H_mul`` = :class:`MultiClassHead` (cosine similarity /
temperature 0.2). Module names match the baseline state dict.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from core import constants

BIN_HEAD_VANILLA = "vanilla"
BIN_HEAD_FUSED = "fused_vanilla"
BIN_HEAD_ADAPTIVE = "adaptive"


class ConvScoreHead(nn.Module):
    """Stacked Conv1d score head; halves channels per layer, final conv → 1."""

    def __init__(
        self,
        d_model: int = constants.HIDDEN_DIM,
        num_layers: int = constants.SCORE_HEAD_LAYERS,
        kernel_size: int = constants.SCORE_HEAD_KERNEL,
    ) -> None:
        super().__init__()
        n_pad = (kernel_size - 1) // 2
        dim_list = [d_model // (2**i) for i in range(num_layers)]
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    dim_list[i],
                    dim_list[i + 1],
                    kernel_size=kernel_size,
                    padding=n_pad,
                    padding_mode="replicate",
                )
                for i in range(num_layers - 1)
            ]
        )
        self.convs.append(
            nn.Conv1d(
                dim_list[-1], 1, kernel_size=kernel_size,
                padding=n_pad, padding_mode="replicate",
            )
        )
        self.norm = nn.ModuleList([nn.LayerNorm(d) for d in dim_list[1:]])

    def forward(self, x: Tensor) -> Tensor:
        """``x (B, C, T)`` → logits ``(B, 1, T)``."""
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i != len(self.convs) - 1:
                x = self.norm[i](F.gelu(x).permute(0, 2, 1)).permute(0, 2, 1)
        return x


class BinaryHead(nn.Module):
    """``H_bin``: per-frame binary logits from pre- and/or post-fusion features."""

    def __init__(
        self,
        bin_head_type: str = BIN_HEAD_ADAPTIVE,
        d_model: int = constants.HIDDEN_DIM,
        num_layers: int = constants.SCORE_HEAD_LAYERS,
        kernel_size: int = constants.SCORE_HEAD_KERNEL,
        adp_alpha_init: float = constants.ADAPTIVE_FUSE_ALPHA0,
        adp_weight: float = constants.ADAPTIVE_FUSE_SCALE,
    ) -> None:
        super().__init__()
        kwargs = {"d_model": d_model, "num_layers": num_layers, "kernel_size": kernel_size}
        self.bin_head_type = bin_head_type
        self.adp_weight = adp_weight
        if bin_head_type == BIN_HEAD_VANILLA:
            self.bin_head = ConvScoreHead(**kwargs)
        elif bin_head_type == BIN_HEAD_FUSED:
            self.bin_fused_head = ConvScoreHead(**kwargs)
        elif bin_head_type == BIN_HEAD_ADAPTIVE:
            self.bin_head = ConvScoreHead(**kwargs)
            self.bin_fused_head = ConvScoreHead(**kwargs)
            self.adp_alpha = nn.Parameter(torch.full((1,), adp_alpha_init))
        else:
            raise ValueError(f"Unknown bin_head_type: {bin_head_type!r}")

    def forward(
        self, before_fused: Tensor | None = None, after_fused: Tensor | None = None
    ) -> Tensor:
        """Inputs ``(B, T, C)`` → logits ``(B, T)``; adaptive blends both paths."""
        if self.bin_head_type == BIN_HEAD_VANILLA:
            if before_fused is None:
                raise ValueError("vanilla bin head requires before_fused")
            pre: Tensor = self.bin_head(before_fused.permute(0, 2, 1)).squeeze(1)
            return pre
        if self.bin_head_type == BIN_HEAD_FUSED:
            if after_fused is None:
                raise ValueError("fused_vanilla bin head requires after_fused")
            post: Tensor = self.bin_fused_head(after_fused.permute(0, 2, 1)).squeeze(1)
            return post
        if before_fused is None or after_fused is None:
            raise ValueError("adaptive bin head requires both before_fused and after_fused")
        w = torch.sigmoid(self.adp_alpha * self.adp_weight)
        blended: Tensor = (
            self.bin_head(before_fused.permute(0, 2, 1)).squeeze(1) * w
            + self.bin_fused_head(after_fused.permute(0, 2, 1)).squeeze(1) * (1 - w)
        )
        return blended


class SimScoreHead(nn.Module):
    """Cosine similarity / learnable temperature between video and text features."""

    def __init__(self, temperature_init: float = constants.MULTICLASS_TEMP) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.full((1,), temperature_init))

    def forward(self, x_feat: Tensor, y_feat: Tensor) -> Tensor:
        """``x (B, T, E)``, ``y (B, C, E)`` → similarity ``(B, T, C)``."""
        x_feat = x_feat / x_feat.norm(dim=-1, keepdim=True)
        y_feat = y_feat / y_feat.norm(dim=-1, keepdim=True)
        sim: Tensor = torch.einsum("bte,bce->btc", x_feat, y_feat) / self.temperature
        return sim


class MultiClassHead(nn.Module):
    """``H_mul``: per-frame class-similarity matrix (baseline ``sim`` head)."""

    def __init__(self, temperature_init: float = constants.MULTICLASS_TEMP) -> None:
        super().__init__()
        self.sim_head = SimScoreHead(temperature_init=temperature_init)

    def forward(self, x_feat: Tensor, y_feat: Tensor) -> Tensor:
        sim: Tensor = self.sim_head(x_feat, y_feat)
        return sim


__all__ = [
    "BIN_HEAD_ADAPTIVE",
    "BIN_HEAD_FUSED",
    "BIN_HEAD_VANILLA",
    "BinaryHead",
    "ConvScoreHead",
    "MultiClassHead",
    "SimScoreHead",
]
