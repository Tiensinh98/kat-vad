"""Per-head taps inside the frozen VideoMAE V2 encoder (v4 PREREG I2).

``Attention.forward`` runs fused SDPA, so ``attn @ v`` is never materialised as a
tensor of its own. Its merged form is: the input of ``attn.proj``,
``(B, N, dim)`` with channel ``head * head_dim + d`` (the module's
``transpose(1, 2).reshape(B, N, dim)``). :class:`HeadTap` registers a
``forward_pre_hook`` on every block's ``attn.proj``, mean-pools that input over
the N tokens (the same token mean that gives ``u``) and, after each encoder
forward, stores one ``(B, depth * heads, head_dim)`` array. Head index =
``block * heads + head``.

The hooks only read: the encoder's output is unchanged (tested), and the
encoder module is never edited, so every v2 cache stays valid (C2).
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from core.models.videomae_v2 import VideoMAEv2Encoder


class HeadTap:
    """Context manager collecting every block's per-head token mean, batch by batch."""

    def __init__(self, encoder: VideoMAEv2Encoder) -> None:
        self.encoder = encoder
        self.depth = len(encoder.blocks)
        self.heads = int(encoder.blocks[0].attn.heads)
        self.head_dim = encoder.dim // self.heads
        self._current: dict[int, Tensor] = {}
        self._batches: list[np.ndarray] = []
        self._handles: list[RemovableHandle] = []

    @property
    def num_heads(self) -> int:
        """Candidate pool size: ``depth * heads`` (72 for V2-S)."""
        return int(self.depth * self.heads)

    def _block_hook(self, index: int) -> Any:
        def hook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
            self._current[index] = args[0].detach().float().mean(dim=1)  # (B, dim)
        return hook

    def _forward_hook(self, _module: nn.Module, _args: Any, _output: Any) -> None:
        if sorted(self._current) != list(range(self.depth)):
            raise RuntimeError(f"tapped blocks {sorted(self._current)}, expected {self.depth}")
        # (B, depth, dim)
        stacked = torch.stack([self._current[i] for i in range(self.depth)], dim=1)
        batch = stacked.shape[0]
        rows = stacked.reshape(batch, self.depth * self.heads, self.head_dim)
        self._batches.append(rows.cpu().numpy().astype(np.float32))
        self._current = {}

    def __enter__(self) -> HeadTap:
        self._batches = []
        self._current = {}
        for index, block in enumerate(self.encoder.blocks):
            self._handles.append(block.attn.proj.register_forward_pre_hook(self._block_hook(index)))
        self._handles.append(self.encoder.register_forward_hook(self._forward_hook))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def rows(self) -> np.ndarray:
        """``(N, depth * heads, head_dim)`` float32, in forward order; resets the buffer."""
        if not self._batches:
            return np.zeros((0, self.num_heads, self.head_dim), dtype=np.float32)
        out: np.ndarray = np.concatenate(self._batches, axis=0)
        self._batches = []
        return out


__all__ = ["HeadTap"]
