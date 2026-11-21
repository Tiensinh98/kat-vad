"""``core.tools.dota_cap_pool``: one row's dev + eval reads -> whole DoTA-CAP (KIM_RUN.md §36)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.tools import dota_cap_pool as dcp
from core.tools import protocol_b_eval as pb
from core.tools import zs_lagovad_readout as zlr

DEV, EVAL = constants.V2_SPLIT_DOTA_CAP_DEV, constants.V2_SPLIT_DOTA_CAP_EVAL
FRAMES = 12
POSITIVE = slice(4, 8)


def _readout(split: str, clips: int, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "protocol": "B: s1[::3], whole clip, native frames (addendum §8.1 J1-J5)",
        "split": split, "clips": clips, "two_class": clips,
        "arm": {"crn": "none", "motion": "none"}, "runs": ["s2024"],
        "global_steps": {"s2024": 1749}, "released_baseline": False,
        "checkpoints": {"s2024": "/drive/L0dfi/K5/s2024/checkpoint_last.pt"},
        "per_run_macro": {"s2024": 0.7},
    }
    base.update(over)
    return base


def _write_read(d: Path, ids: list[str], split: str, seed: int, **over: Any) -> Path:
    rng = np.random.default_rng(seed)
    labels = {}
    for v in ids:
        y = np.zeros(FRAMES, dtype=np.int64)
        y[POSITIVE] = 1
        labels[v] = y
    scores = {v: rng.random(FRAMES) for v in ids}
    d.mkdir(parents=True)
    pb.write_clip_scores(d / pb.CLIP_SCORES_NPZ, scores, labels)
    (d / pb.CLIP_AUCS_JSON).write_text(json.dumps(pb.clip_aucs(scores, labels)))
    (d / pb.READOUT_JSON).write_text(json.dumps(_readout(split, len(ids), **over)))
    return d


@pytest.fixture
def reads(tmp_path: Path) -> tuple[Path, Path]:
    dev = _write_read(tmp_path / "dev", [f"dvid{i}_001" for i in range(5)], DEV, 0)
    ev = _write_read(tmp_path / "eval", [f"evid{i}_001" for i in range(4)], EVAL, 1)
    return dev, ev


def test_pool_concatenates_without_rescoring(reads: tuple[Path, Path], tmp_path: Path) -> None:
    dev, ev = reads
    out = tmp_path / "all"
    readout = dcp.write_pooled(dev, ev, out, (5, 4))
    scores, labels = pb.read_clip_scores(out / pb.CLIP_SCORES_NPZ)
    sd, _ = pb.read_clip_scores(dev / pb.CLIP_SCORES_NPZ)
    se, _ = pb.read_clip_scores(ev / pb.CLIP_SCORES_NPZ)
    assert set(scores) == set(sd) | set(se) and len(labels) == 9
    assert all(np.array_equal(scores[v], s[v]) for s in (sd, se) for v in s)
    aucs = json.loads((out / pb.CLIP_AUCS_JSON).read_text())
    assert readout["clips"] == 9 and readout["two_class"] == len(aucs) == 9
    assert readout["per_run_macro"]["pooled"] == pytest.approx(np.mean(list(aucs.values())))


def test_pooled_dir_is_a_protocol_b_read(reads: tuple[Path, Path], tmp_path: Path) -> None:
    """``zs_arm_table`` reads DoTA via ``zs_lagovad_readout.read_dota``: it accepts the output."""
    dev, ev = reads
    out = tmp_path / "all"
    dcp.write_pooled(dev, ev, out, None)
    got = zlr.read_dota(out)
    assert len(got["items"]) == 9 and 0.0 <= got["micro"] <= 1.0


@pytest.mark.parametrize("key, value", [
    ("arm", {"crn": "none", "motion": "vit_s_k710_dl_from_giant"}),
    ("global_steps", {"s2024": 1530}),
    ("checkpoints", {"s2024": "/drive/K5dfi/K5/s2024/checkpoint_last.pt"}),
    ("runs", ["s2025"]),
    ("protocol", "B: s1[::8], whole clip, native frames (addendum §8.1 J1-J5)"),
    ("released_baseline", True),
])
def test_refuses_a_different_model(tmp_path: Path, key: str, value: Any) -> None:
    dev = _write_read(tmp_path / "dev", ["dvid0_001", "dvid1_001"], DEV, 0)
    ev = _write_read(tmp_path / "eval", ["evid0_001"], EVAL, 1, **{key: value})
    with pytest.raises(ValueError):
        dcp.write_pooled(dev, ev, tmp_path / "all", None)


def test_refuses_wrong_splits(tmp_path: Path) -> None:
    dev = _write_read(tmp_path / "dev", ["dvid0_001"], DEV, 0)
    ev = _write_read(tmp_path / "eval", ["evid0_001"], DEV, 1)
    with pytest.raises(ValueError, match="splits"):
        dcp.write_pooled(dev, ev, tmp_path / "all", None)


def test_refuses_overlapping_clips(tmp_path: Path) -> None:
    dev = _write_read(tmp_path / "dev", ["avid_001", "bvid_001"], DEV, 0)
    ev = _write_read(tmp_path / "eval", ["bvid_001", "cvid_001"], EVAL, 1)
    with pytest.raises(ValueError, match="in both reads"):
        dcp.write_pooled(dev, ev, tmp_path / "all", None)


def test_refuses_wrong_counts(reads: tuple[Path, Path], tmp_path: Path) -> None:
    dev, ev = reads
    with pytest.raises(ValueError, match="want 569 / 560"):
        dcp.write_pooled(dev, ev, tmp_path / "all",
                         (constants.DOTA_CAP_DEV_CLIPS, constants.DOTA_CAP_EVAL_CLIPS))


def test_cli_writes_the_three_files(reads: tuple[Path, Path], tmp_path: Path) -> None:
    dev, ev = reads
    out = tmp_path / "all"
    dcp.main(["--dev", str(dev), "--eval", str(ev), "--out-dir", str(out), "--any-size"])
    files = (pb.CLIP_SCORES_NPZ, pb.CLIP_AUCS_JSON, pb.READOUT_JSON)
    assert all((out / f).is_file() for f in files)


def test_real_reads_reproduce_the_section_35_pool() -> None:
    """On disk (skipped elsewhere): K5f's §35 dev + eval reads pool to the committed §35 file."""
    root = constants.REPO_ROOT / "outputs" / "v_final" / "kim_cap_eval" / "K5f" / "eval"
    dev = constants.REPO_ROOT / "outputs" / "v_final" / "kim_fixed" / "K5f" / "eval" / "dota" / "K5"
    paths = (dev / pb.CLIP_SCORES_NPZ, root / EVAL / "K5" / pb.CLIP_SCORES_NPZ,
             root / constants.DOTA_CAP_ALL_DIRNAME / "K5" / pb.CLIP_SCORES_NPZ)
    if not all(p.is_file() for p in paths):
        pytest.skip("the §35 reads are not on this disk")
    scores, _, aucs, _ = dcp.pool(dev, root / EVAL / "K5",
                                  (constants.DOTA_CAP_DEV_CLIPS, constants.DOTA_CAP_EVAL_CLIPS))
    ref, _ = pb.read_clip_scores(paths[2])
    assert set(scores) == set(ref) and all(np.array_equal(scores[v], ref[v]) for v in ref)
    assert np.mean(list(aucs.values())) == pytest.approx(0.8226300616592576)
