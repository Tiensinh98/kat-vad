"""Column layout of one agent-cache row (PREREG_V6 §2.1).

A row is ``K_MAX`` agent slots followed by the ego block::

    slot k: [valid, x1, y1, x2, y2, conf, class, track_id, kinematics (9), motion (384), crop (512)]
    ego:    camera affine minus identity, 6 values per slot of the centred clip (8 slots)

Boxes are normalized to the frame (``[0, 1]``). An empty slot is all zeros (``valid = 0``),
which is also what batch padding produces, so a padded row and a row without agents read alike.
The same layout is read by the model (torch) and by the Step 0 / read-out tools (numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch

from core import constants

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)

VALID = 0
BOX = slice(1, 5)
CONF = 5
CLASS = 6
TRACK = 7
KIN = slice(constants.V6_AGENT_HEAD, constants.V6_AGENT_HEAD + constants.V6_KIN_DIM)
MOTION = slice(KIN.stop, KIN.stop + constants.V6_AGENT_MOTION_DIM)
CROP = slice(MOTION.stop, MOTION.stop + constants.V6_AGENT_CROP_DIM)
if CROP.stop != constants.V6_AGENT_DIM:  # pragma: no cover - a constants edit broke the layout
    raise RuntimeError("agent slot layout does not add up to V6_AGENT_DIM")


@dataclass(frozen=True)
class RowLayout:
    """Where the agent columns sit in a model input row."""

    base: int  # columns before the agent block (A2 row: CLIP 512 + motion 384)
    k_max: int = constants.V6_K_MAX

    @property
    def agents_width(self) -> int:
        return self.k_max * constants.V6_AGENT_DIM

    @property
    def width(self) -> int:
        return self.agents_width + constants.V6_EGO_DIM

    @property
    def total(self) -> int:
        return self.base + self.width


def split_row(rows: ArrayT, layout: RowLayout) -> tuple[ArrayT, ArrayT, ArrayT]:
    """``(..., total)`` rows -> base ``(..., base)``, agents ``(..., K, AGENT_DIM)``, ego
    ``(..., 48)``."""
    if rows.shape[-1] != layout.total:
        raise ValueError(
            f"rows have {rows.shape[-1]} columns, expected {layout.base} (base) + "
            f"{layout.width} (agents) -- is the agent cache appended (lesson C13)?"
        )
    lead = tuple(rows.shape[:-1])
    base = rows[..., : layout.base]
    agents = rows[..., layout.base : layout.base + layout.agents_width].reshape(
        *lead, layout.k_max, constants.V6_AGENT_DIM
    )
    ego = rows[..., layout.base + layout.agents_width :]
    return base, agents, ego


def pack_rows(agents: np.ndarray, ego: np.ndarray) -> np.ndarray:
    """``(R, K, AGENT_DIM)`` + ``(R, EGO_DIM)`` -> ``(R, K * AGENT_DIM + EGO_DIM)``."""
    if agents.ndim != 3 or agents.shape[-1] != constants.V6_AGENT_DIM:
        raise ValueError(f"agents must be (R, K, {constants.V6_AGENT_DIM}), got {agents.shape}")
    if ego.shape != (len(agents), constants.V6_EGO_DIM):
        raise ValueError(f"ego must be ({len(agents)}, {constants.V6_EGO_DIM}), got {ego.shape}")
    out: np.ndarray = np.concatenate([agents.reshape(len(agents), -1), ego], axis=1)
    return out


def unpack_rows(rows: np.ndarray, k_max: int = constants.V6_K_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_rows` for a stored ``(R, width)`` agent array."""
    _, agents, ego = split_row(rows, RowLayout(base=0, k_max=k_max))
    return agents, ego


__all__ = [
    "BOX",
    "CLASS",
    "CONF",
    "CROP",
    "KIN",
    "MOTION",
    "TRACK",
    "VALID",
    "RowLayout",
    "pack_rows",
    "split_row",
    "unpack_rows",
]
