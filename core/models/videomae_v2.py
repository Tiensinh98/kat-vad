"""Frozen VideoMAE V2 distilled encoder -- the v2 motion stream's backbone.

Proposal ``core/docs/v2/KAT_VAD_PROPOSAL_v2.md`` §4.1; addendum §3/§6.1.

**Why vendored.** The distilled checkpoints (``vit_{b,s}_k710_dl_from_giant``)
are not in ``transformers``; upstream builds them with ``timm.create_model`` from
``OpenGVLab/VideoMAEv2`` ``models/modeling_finetune.py``. That file drags in a
timm registry and a training-only surface (drop-path, gradient checkpointing,
cosine attention, layer scale), none of which the distilled B/S checkpoints use.
This module keeps only the inference path of that ``VisionTransformer`` with its
**parameter names unchanged**, so the upstream ``state_dict`` loads strictly
(lesson C5) and nothing is re-mapped by hand.

What the upstream forward does, kept here exactly:

* ``Conv3d`` tubelet embedding (2 x 16 x 16), a **fixed** sinusoid position table
  (not a parameter -- absent from the checkpoint);
* pre-LN blocks; the qkv projection has no bias of its own -- ``q_bias`` and
  ``v_bias`` are concatenated around a zero ``k`` bias;
* ``fc_norm(tokens.mean(1))`` -- the feature the K710 head reads. That is ``u``.

The classification ``head`` is dropped at load time and **only** it: every other
checkpoint key must match a parameter and vice versa, or the load raises.

Weights are pinned by HF commit **and** sha256 (lesson C4) and are read with
``torch.load(weights_only=True)`` -- the file is a plain ``{"module": state_dict}``
(measured 2026-09-28), so no pickle of a library object is needed (lesson C15).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from core import constants

LOGGER = logging.getLogger(__name__)

HEAD_PREFIX = "head."
STATE_KEYS = ("module", "model")
SHA_CHUNK_BYTES = 1 << 20
SINUSOID_BASE = 10000.0


def sinusoid_table(positions: int, dim: int) -> Tensor:
    """Upstream ``get_sinusoid_encoding_table``: ``(1, positions, dim)`` float32."""
    position = torch.arange(positions, dtype=torch.float64).unsqueeze(1)
    exponent = 2.0 * torch.div(torch.arange(dim), 2, rounding_mode="floor") / dim
    angles = position / torch.pow(torch.tensor(SINUSOID_BASE, dtype=torch.float64), exponent)
    angles[:, 0::2] = torch.sin(angles[:, 0::2])
    angles[:, 1::2] = torch.cos(angles[:, 1::2])
    return angles.float().unsqueeze(0)


class PatchEmbed(nn.Module):
    """``(B, 3, T, H, W)`` -> ``(B, T/t * H/p * W/p, D)`` tubelet tokens."""

    def __init__(self, dim: int, patch: int, tubelet: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            3, dim, kernel_size=(tubelet, patch, patch), stride=(tubelet, patch, patch)
        )

    def forward(self, x: Tensor) -> Tensor:
        tokens: Tensor = self.proj(x).flatten(2).transpose(1, 2)
        return tokens


class Attention(nn.Module):
    """Multi-head self-attention with upstream's split ``q_bias`` / ``v_bias``."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = nn.functional.linear(x, self.qkv.weight, bias)
        qkv = qkv.reshape(batch, tokens, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        # SDPA's default scale is head_dim ** -0.5, upstream's `qk_scale=None` value.
        out = nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        projected: Tensor = self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))
        return projected


class Mlp(nn.Module):
    """``fc1 -> GELU -> fc2``, upstream names."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.fc2(self.act(self.fc1(x)))
        return out


class Block(nn.Module):
    """Pre-LN transformer block (no layer scale: ``init_values=0`` upstream)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=constants.VIDEOMAE_LN_EPS)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=constants.VIDEOMAE_LN_EPS)
        self.mlp = Mlp(dim, dim * constants.VIDEOMAE_MLP_RATIO)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        out: Tensor = x + self.mlp(self.norm2(x))
        return out


class VideoMAEv2Encoder(nn.Module):
    """Headless VideoMAE V2 ViT: ``(B, 3, 16, 224, 224)`` -> ``(B, D)``."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        frames: int = constants.VIDEOMAE_CLIP_FRAMES,
        image_size: int = constants.CROP_SIZE,
        patch: int = constants.VIDEOMAE_PATCH_SIZE,
        tubelet: int = constants.VIDEOMAE_TUBELET_SIZE,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.patch_embed = PatchEmbed(dim, patch, tubelet)
        positions = (frames // tubelet) * (image_size // patch) ** 2
        self.register_buffer("pos_embed", sinusoid_table(positions, dim), persistent=False)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.fc_norm = nn.LayerNorm(dim, eps=constants.VIDEOMAE_LN_EPS)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.patch_embed(x)
        if tokens.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"input gives {tokens.shape[1]} tokens, the position table has "
                f"{self.pos_embed.shape[1]} -- clip must be {constants.VIDEOMAE_CLIP_FRAMES} "
                f"frames of {constants.CROP_SIZE}^2"
            )
        tokens = tokens + self.pos_embed.to(tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        pooled: Tensor = self.fc_norm(tokens.mean(1))
        return pooled


def build_encoder(name: str) -> VideoMAEv2Encoder:
    """Random-weight encoder with ``name``'s architecture (tests; before loading)."""
    if name not in constants.VIDEOMAE_ARCH:
        known = sorted(constants.VIDEOMAE_ARCH)
        raise KeyError(f"unknown VideoMAE encoder {name!r}; known: {known}")
    dim, depth, heads = constants.VIDEOMAE_ARCH[name]
    return VideoMAEv2Encoder(dim, depth, heads)


def unwrap_state(checkpoint: Mapping[str, object]) -> dict[str, Tensor]:
    """The state dict inside an upstream checkpoint, minus the K710 head."""
    state: object = checkpoint
    for key in STATE_KEYS:
        if key in checkpoint:
            state = checkpoint[key]
            break
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint holds no state dict under {STATE_KEYS}")
    return {
        str(k): v
        for k, v in state.items()
        if isinstance(v, Tensor) and not str(k).startswith(HEAD_PREFIX)
    }


def file_sha256(path: Path) -> str:
    """Streaming sha256 of a weights file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(SHA_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(path: Path, name: str) -> None:
    """Raise unless ``path`` is byte-identical to the pinned ``name`` weights."""
    expected = constants.VIDEOMAE_WEIGHTS_SHA256[name]
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{path} sha256 {actual} != pinned {expected} for {name}")


def fetch_weights(name: str) -> Path:
    """Download ``name`` at the pinned commit (HF cache) and verify its sha256."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=constants.VIDEOMAE_REPO_ID,
            filename=f"{constants.VIDEOMAE_WEIGHTS_SUBDIR}/{name}.pth",
            revision=constants.VIDEOMAE_REVISION,
        )
    )
    verify_weights(path, name)
    return path


def load_pretrained(
    name: str, device: torch.device, weights: Path | None = None
) -> VideoMAEv2Encoder:
    """Frozen, eval-mode encoder with the pinned weights; strict load (C5)."""
    path = weights if weights is not None else fetch_weights(name)
    if weights is not None:
        verify_weights(path, name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_encoder(name)
    model.load_state_dict(unwrap_state(checkpoint), strict=True)
    model.requires_grad_(False)
    LOGGER.info("VideoMAE V2 %s loaded from %s (dim %d)", name, path, model.dim)
    return model.to(device).eval()
