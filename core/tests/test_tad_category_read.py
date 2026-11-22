"""Tests for :mod:`core.tools.tad_category_read` (``KIM_RUN.md`` §38.2 N4), synthetic scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import tad_category_read as tcr

FRAMES = 10


def _video(scores_dir: Path, name: str, score: list[float], gt: list[int]) -> None:
    np.savez(scores_dir / f"{name}.npz", score=np.array(score), gt=np.array(gt),
             sim=np.zeros((len(score), 2)), class_names=np.array(["Normal", "CarAccident"]))


def _run(root: Path, accident_peak: float) -> Path:
    scores = root / tcr.SCORES_DIRNAME
    scores.mkdir(parents=True)
    gt = [0] * 6 + [1] * 4
    _video(scores, "01_Accident_001", [0.1] * 6 + [accident_peak] * 4, gt)
    _video(scores, "01_Accident_002", [0.2] * 6 + [accident_peak] * 4, gt)
    _video(scores, "06_PedestrianOnRoad_001", [0.5] * 6 + [0.1] * 4, gt)  # inverted
    _video(scores, "Normal_001", [0.3] * FRAMES, [0] * FRAMES)
    _video(scores, "Normal_002", [0.4] * FRAMES, [0] * FRAMES)
    return root


def test_category_of_strips_the_index() -> None:
    assert tcr.category_of("01_Accident_004") == "01_Accident"
    assert tcr.category_of("Normal_128") == tcr.NORMAL_CATEGORY
    assert tcr.category_of("05_else_013") == "05_else"


def test_cells_per_category(tmp_path: Path) -> None:
    cells = tcr.read_categories(_run(tmp_path, accident_peak=0.9))
    acc = cells["01_Accident"]
    assert (acc["videos"], acc["two_class"]) == (2, 2)
    assert acc["macro"] == pytest.approx(1.0)
    assert acc["clip_auc"] == pytest.approx(1.0)  # peaks 0.9 above the normals' 0.3 / 0.4
    ped = cells["06_PedestrianOnRoad"]
    assert ped["macro"] == pytest.approx(0.0)
    assert cells[tcr.ALL_ABNORMAL]["videos"] == 3
    assert tcr.NORMAL_CATEGORY not in cells


def test_low_peaks_lose_the_clip_auc(tmp_path: Path) -> None:
    cells = tcr.read_categories(_run(tmp_path, accident_peak=0.25))
    assert cells["01_Accident"]["macro"] == pytest.approx(1.0)  # still ranks within its video
    assert cells["01_Accident"]["clip_auc"] < 0.5  # but its peak sits below the normals'


def test_missing_scores_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no score files"):
        tcr.read_categories(tmp_path)


def test_main_writes_both_rows(tmp_path: Path) -> None:
    a, b = _run(tmp_path / "a", 0.9), _run(tmp_path / "b", 0.25)
    out = tmp_path / "out"
    tcr.main(["--read", "K5f", str(a), "--read", "K5fno", str(b), "--out-dir", str(out),
              "--name", "t"])
    stem = out / f"{constants.TAD_CATEGORY_STEM}_t"
    payload = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(payload) == {"K5f", "K5fno"}
    text = stem.with_suffix(".md").read_text(encoding="utf-8")
    assert "K5fno macro / clip AUC / micro" in text and "01_Accident" in text
