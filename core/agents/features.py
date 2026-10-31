"""Per-row agent blocks (PREREG_V6 §2.1): what the agent cache stores for one video.

For each model row ``t`` (a native frame, ``s1[::stride]``), at most ``K_MAX`` tracked agents
alive at ``t`` (by track-mean confidence) each get

* geometry: the tracker box at ``t`` (normalized), track-mean confidence, class, track id;
* kinematics: ego-compensated velocity / acceleration (box heights / s), looming, age, gaps;
* motion: VideoMAE V2-S tokens of the clip centred on ``t``, 2 x 2 RoIAlign of the agent's box on
  each of the 8 temporal slots (the union of its boxes at the slot's two frames, at least one
  16-px token wide), averaged over the slots where the track exists;
* appearance: CLIP ``image_embeds`` of a square crop of the native frame (1.5 x the long side,
  at least 32 px, zero-padded outside the frame), in the space of the frame features and the text.

The ego block is the camera warp across each slot of the same centred clip. Missing agents are
zero slots. Everything is computed from frozen encoders and the frozen tracker.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.ops import roi_align

from core import constants
from core.agents import layout
from core.agents.geometry import (
    Track,
    TrackLog,
    box_at,
    build_tracks,
    crop_square,
    cut_square,
    ego_block,
    enlarge_to,
    kinematics,
    select_agents,
)
from core.tools.extract_video_features import centred_clip_indices

TokenFn = Callable[[list[list[int]]], Tensor]  # clips (B x 16 frame indices) -> (B, 1568, D)
FrameFn = Callable[[int], np.ndarray]  # native frame index -> RGB uint8 (H, W, 3)
MIN_TOKEN_SIDE = 1.0 / constants.V6_TOKEN_GRID
CROP_BATCH = 64


@dataclass
class RowSources:
    """How one video's rows, clips and frames are read."""

    num_frames: int
    stride: int  # model rows = s1[::stride]
    clip_step: int  # native frames between VideoMAE clip frames
    det_step: int  # native frames between detection frames
    tokens: TokenFn
    frame: FrameFn

    @property
    def rows(self) -> list[int]:
        return list(range(0, self.num_frames, self.stride))


def slot_boxes(track: Track, clip: list[int]) -> list[np.ndarray | None]:
    """The agent's box on each temporal slot: union of its boxes at the slot's two frames."""
    out: list[np.ndarray | None] = []
    for slot in range(constants.V6_TOKEN_SLOTS):
        boxes = [b for f in clip[2 * slot : 2 * slot + 2] if (b := box_at(track, f)) is not None]
        if not boxes:
            out.append(None)
            continue
        stack = np.stack(boxes)
        union = np.concatenate([stack[:, :2].min(0), stack[:, 2:].max(0)])
        out.append(np.clip(enlarge_to(union, MIN_TOKEN_SIDE), 0.0, 1.0))
    return out


def roi_motion(tokens: Tensor, boxes: list[list[list[np.ndarray | None]]]) -> np.ndarray:
    """``tokens (B, 1568, D)`` + per row, per agent, per slot boxes -> ``(B, K_MAX, D)`` means."""
    batch, _, dim = tokens.shape
    grid = constants.V6_TOKEN_GRID
    slots = constants.V6_TOKEN_SLOTS
    feat = (
        tokens.float()
        .reshape(batch, slots, grid, grid, dim)
        .permute(0, 1, 4, 2, 3)
        .reshape(batch * slots, dim, grid, grid)
    )
    rois, owner = [], []
    for b, agents in enumerate(boxes):
        for k, per_slot in enumerate(agents):
            for s, box in enumerate(per_slot):
                if box is not None:
                    rois.append([b * slots + s, *(box * grid).tolist()])
                    owner.append(b * constants.V6_K_MAX + k)
    out = torch.zeros(batch * constants.V6_K_MAX, dim, device=tokens.device)
    if rois:
        pooled = roi_align(  # pyright: ignore[reportCallIssue]
            feat,
            torch.tensor(rois, dtype=feat.dtype, device=feat.device),
            output_size=constants.V6_ROI_OUT,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        ).mean((2, 3))
        index = torch.tensor(owner, device=feat.device)
        out.index_add_(0, index, pooled)
        count = torch.zeros(batch * constants.V6_K_MAX, device=feat.device)
        count.index_add_(0, index, torch.ones(len(owner), device=feat.device))
        out = out / count.clamp_min(1.0)[:, None]
    result: np.ndarray = out.reshape(batch, constants.V6_K_MAX, dim).cpu().numpy()
    return result


@torch.no_grad()
def crop_embeds(crops: list[np.ndarray], encoder: Any, device: torch.device) -> np.ndarray:
    """CLIP ``image_embeds (N, 512)`` of square uint8 crops (squash to 224, CLIP mean / std)."""
    out: list[np.ndarray] = []
    mean = torch.tensor(constants.CLIP_IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(constants.CLIP_IMAGE_STD, device=device).view(1, 3, 1, 1)
    for start in range(0, len(crops), CROP_BATCH):
        pixels = torch.stack(
            [
                nn.functional.interpolate(
                    torch.from_numpy(c).permute(2, 0, 1)[None].float().to(device) / 255.0,
                    size=(constants.CROP_SIZE, constants.CROP_SIZE),
                    mode="bilinear",
                    antialias=True,
                )[0]
                for c in crops[start : start + CROP_BATCH]
            ]
        )
        embeds = encoder(pixel_values=(pixels - mean) / std).image_embeds
        out.append(embeds.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, constants.V6_AGENT_CROP_DIM), np.float32)


def agent_rows(
    log: TrackLog,
    sources: RowSources,
    clip_encoder: Any,
    device: torch.device,
    batch_rows: int = 16,
    k_max: int = constants.V6_K_MAX,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``(R, V6_ROW_WIDTH)`` float16 agent rows of one video and a summary for the report."""
    if k_max != constants.V6_K_MAX:
        raise ValueError("the cache layout is fixed at V6_K_MAX agents per row")
    tracks = build_tracks(log)
    rows = sources.rows
    agents = np.zeros((len(rows), k_max, constants.V6_AGENT_DIM), dtype=np.float32)
    ego = np.zeros((len(rows), constants.V6_EGO_DIM), dtype=np.float32)
    width, height = log.size
    counts: list[int] = []
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start : start + batch_rows]
        clips = [centred_clip_indices(t, sources.num_frames, step=sources.clip_step) for t in chunk]
        chosen = [select_agents(tracks, t, k_max) for t in chunk]
        alive = [[o for o in tracks.values() if o.alive(t)] for t in chunk]
        boxes = [
            [slot_boxes(a, clip) for a in sel] for sel, clip in zip(chosen, clips, strict=True)
        ]
        motion = roi_motion(sources.tokens(clips), boxes)
        crops, owner = [], []
        for b, (t, sel) in enumerate(zip(chunk, chosen, strict=True)):
            if not sel:
                continue
            frame = sources.frame(t)
            for k, track in enumerate(sel):
                raw = box_at(track, t, raw=True)
                assert raw is not None  # nosec B101 - select_agents keeps tracks alive at t
                crops.append(cut_square(frame, *crop_square(raw, width, height)))
                owner.append((b, k))
        embeds = crop_embeds(crops, clip_encoder, device)
        for (b, k), e in zip(owner, embeds, strict=True):
            agents[start + b, k, layout.CROP] = e
        for b, (t, sel, others, clip) in enumerate(zip(chunk, chosen, alive, clips, strict=True)):
            r = start + b
            counts.append(len(sel))
            ego[r] = ego_block(log, clip)
            for k, track in enumerate(sel):
                box = box_at(track, t)
                assert box is not None  # nosec B101 - alive at t
                agents[r, k, layout.VALID] = 1.0
                agents[r, k, layout.BOX] = box
                agents[r, k, layout.CONF] = track.mean_conf
                agents[r, k, layout.CLASS] = track.cls
                agents[r, k, layout.TRACK] = track.tid
                agents[r, k, layout.KIN] = kinematics(track, t, log, others, sources.det_step)
                agents[r, k, layout.MOTION] = motion[b, k]
    packed = layout.pack_rows(agents, ego).astype(np.float16)
    if not np.isfinite(packed).all():
        raise ValueError("non-finite agent row (float16 overflow?)")
    summary = {
        "rows": len(rows),
        "tracks": len(tracks),
        "agents_mean": float(np.mean(counts)) if counts else 0.0,
        "agents_max": int(max(counts, default=0)),
        "rows_without_agents": int(sum(c == 0 for c in counts)),
        "rows_at_k_max": int(sum(c == k_max for c in counts)),
    }
    return packed, summary


def agent_manifest(stride: int, clip_step: int, det_step: int, source: str) -> dict[str, Any]:
    """What a model row's agent columns mean (checked by train / eval, lesson C2)."""
    return {
        "layout": "v6.1: K x [valid, box4, conf, class, track, kin9, motion384, crop512] + ego48",
        "k_max": constants.V6_K_MAX,
        "agent_dim": constants.V6_AGENT_DIM,
        "width": constants.V6_ROW_WIDTH,
        "stride": stride,
        "clip_step": clip_step,
        "det_step": det_step,
        "window": "centred 16 frames t-8*step..t+7*step, edge frames repeated",
        "motion": f"{constants.V5_ENCODER} last layer, 2x2 RoIAlign per slot, slot mean",
        "crop": "CLIP ViT-B/16 image_embeds of a 1.5x square crop (>= 32 px), squash 224",
        "dtype": "float16",
        "source": source,
        "rule": "core/docs/v6/PREREG_V6.md §2.1",
    }


__all__ = ["RowSources", "agent_manifest", "agent_rows", "crop_embeds", "roi_motion", "slot_boxes"]
