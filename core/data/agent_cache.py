"""v6 agent cache (PREREG_V6 §2.1): ``<agent_dir>/{video_id}.npy`` + ``agent_manifest.json``.

Each file holds one float16 row of ``V6_ROW_WIDTH`` columns per model row ``range(0, N, stride)``
of the video -- the same rows as the clip cache it is appended to (Nexar ``A2_S/Nexar_s8``,
DoTA-CAP's protocol-B ``s1[::3]``). A windowed corpus slices it like the clip cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core import constants

MANIFEST = "agent_manifest.json"


def write_agent_manifest(agent_dir: Path, manifest: dict[str, Any]) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / MANIFEST
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        same = {k: v for k, v in old.items() if k != "ultralytics"}
        new = {k: v for k, v in manifest.items() if k != "ultralytics"}
        if same != new:
            raise ValueError(f"{path} was built differently: {old} vs {manifest}")
        return
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_agent_manifest(agent_dir: Path) -> dict[str, Any]:
    path = agent_dir / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: not an agent cache (core.tools.v6_agents)")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("width") != constants.V6_ROW_WIDTH or data.get("k_max") != constants.V6_K_MAX:
        raise ValueError(f"{agent_dir}: layout {data.get('layout')!r} is not this build's")
    return data


def check_agent_dir(agent_enabled: bool, agent_dir: Path | None, stride: int) -> None:
    """A v6 model needs its agent cache at the stride it reads; a non-v6 model refuses one."""
    if not agent_enabled:
        if agent_dir is not None:
            raise ValueError("an agent cache was given for a model with agent.enabled=false")
        return
    if agent_dir is None:
        raise ValueError("agent.enabled=true: pass --agent-dir")
    built = read_agent_manifest(agent_dir)["stride"]
    if built != stride:
        raise ValueError(f"{agent_dir} rows are s1[::{built}], the model reads s1[::{stride}]")


def append_agents(rows: np.ndarray, agents: np.ndarray, video_id: str) -> np.ndarray:
    """``[rows ; agents]`` along the columns, float32; row counts must match exactly."""
    if len(agents) != len(rows):
        raise ValueError(f"{video_id}: {len(agents)} agent rows vs {len(rows)} input rows")
    if agents.shape[1] != constants.V6_ROW_WIDTH:
        raise ValueError(f"{video_id}: agent rows have {agents.shape[1]} columns")
    out: np.ndarray = np.concatenate(
        [np.asarray(rows, dtype=np.float32), np.asarray(agents, dtype=np.float32)], axis=1
    )
    return out


def load_agents(agent_dir: Path, video_id: str) -> np.ndarray:
    out: np.ndarray = np.load(agent_dir / f"{video_id}.npy", mmap_mode="r")
    return out


__all__ = [
    "MANIFEST",
    "append_agents",
    "check_agent_dir",
    "load_agents",
    "read_agent_manifest",
    "write_agent_manifest",
]
