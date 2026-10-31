"""Tests for the v6 post-hoc DADA read (PREREG_V6 §4): official ids, R1 strip, gate G-S.

The failures that matter are silent: a SimpleTAD ``type/video`` line mapped to the wrong
``t{type}_v{video}`` id (another clip's labels), a dropped official clip that vanishes from the
denominator unnamed, a "stripped" checkpoint that throws away a *trained* readout, and a gate
that passes two different models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants
from core.data.dada_origin import OriginRow
from core.tools import dada_whole_eval, protocol_b_eval, v6_r1_dada


def _row(type_id: int, video: int, start: int = 10) -> OriginRow:
    return OriginRow(type_id, video, start, start + 5, start + 20, 100, {})


# ---------------------------------------------------------------------------
# ids
# ---------------------------------------------------------------------------
def test_parse_official_maps_type_video_to_project_ids(tmp_path: Path) -> None:
    p = tmp_path / "validation.txt"
    p.write_text("1/001\n12/045\n\n1/001\n")
    assert v6_r1_dada.parse_official(p) == ["t01_v001", "t12_v045"]


@pytest.mark.parametrize("bad", ["1_001", "a/001", "1/2/3"])
def test_parse_official_rejects_other_formats(tmp_path: Path, bad: str) -> None:
    p = tmp_path / "validation.txt"
    p.write_text(f"{bad}\n")
    with pytest.raises(ValueError, match="type/video"):
        v6_r1_dada.parse_official(p)


def test_filter_sources_names_every_exclusion(tmp_path: Path) -> None:
    clip, video = tmp_path / "clip", tmp_path / "video"
    clip.mkdir()
    video.mkdir()
    for v in ("t01_v001", "t01_v002", "t01_v005"):
        np.save(clip / f"{v}.npy", np.zeros((2, 2)))
    for v in ("t01_v001", "t01_v002"):
        np.save(video / f"{v}.npy", np.zeros((2, 2)))
    annotated = {r.video_id: r for r in
                 (_row(1, 1), _row(1, 2), _row(1, 3, start=500), _row(1, 5), _row(1, 6))}
    census = {"t01_v001": 100, "t01_v002": 100, "t01_v003": 100, "t01_v005": 100}
    official = ["t01_v001", "t01_v002", "t01_v003", "t01_v004", "t01_v005", "t01_v006"]
    kept, dropped = v6_r1_dada.filter_sources(official, annotated, census,
                                              {"clip": clip, "video": video})
    assert kept == ["t01_v001", "t01_v002"]
    assert dropped == {"anomaly_past_frames": ["t01_v003"], "no_annotation_row": ["t01_v004"],
                       "no_video_cache": ["t01_v005"], "no_frames_on_disk": ["t01_v006"]}
    assert len(kept) + sum(map(len, dropped.values())) == len(official)


# ---------------------------------------------------------------------------
# strip
# ---------------------------------------------------------------------------
def _payload(readout: str = constants.V5_READOUT_MEAN, extra: str | None = None,
             loaded: bool = True) -> dict:
    state = {"motion_residual.w.weight": torch.ones(2, 2)}
    for b in v6_r1_dada.MEAN_READOUT_BUFFERS:
        state[f"grid_readout.{b}"] = torch.ones(())
    state["grid_readout.head_loaded"] = torch.tensor(loaded)
    if extra:
        state[f"grid_readout.{extra}"] = torch.ones(3)
    v5 = {"readout": readout, "queries": 1, "keys": "content", "smil_weight": 0.0}
    return {"model": state, "config": {"model": {"motion_dim": 384}, "v5": v5},
            "global_step": 1749}


def test_strip_drops_only_the_frozen_readout_and_turns_it_off() -> None:
    out, dropped = v6_r1_dada.strip_payload(_payload())
    assert out["config"]["v5"]["readout"] == constants.V5_READOUT_OFF
    assert not any(k.startswith("grid_readout.") for k in out["model"])
    assert torch.equal(out["model"]["motion_residual.w.weight"], torch.ones(2, 2))
    assert len(dropped) == len(v6_r1_dada.MEAN_READOUT_BUFFERS)
    assert out["global_step"] == 1749


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"readout": constants.V5_READOUT_MQR}, "parameter-free"),
    ({"extra": "query"}, "non-buffer"),
    ({"loaded": False}, "never loaded"),
])
def test_strip_refuses_a_trained_or_unloaded_readout(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        v6_r1_dada.strip_payload(_payload(**kwargs))


def test_strip_cli_keeps_metrics_for_j10(tmp_path: Path) -> None:
    src = tmp_path / "R1" / "s2024"
    src.mkdir(parents=True)
    torch.save(_payload(), src / "checkpoint_last.pt")
    (src / "metrics.jsonl").write_text('{"global_step": 1749}\n')
    (src / "config.yaml").write_text("v5:\n  readout: mean\n  queries: 1\n")
    out = tmp_path / "strip"
    v6_r1_dada.run_strip(argparse.Namespace(ckpt=src / "checkpoint_last.pt", out_dir=out))
    stored = torch.load(out / "checkpoint_last.pt", weights_only=False)  # nosec B614 - test ckpt
    assert stored["config"]["v5"]["readout"] == constants.V5_READOUT_OFF
    assert (out / "metrics.jsonl").read_text() == (src / "metrics.jsonl").read_text()
    assert "readout: none" in (out / "config.yaml").read_text()


# ---------------------------------------------------------------------------
# gate G-S
# ---------------------------------------------------------------------------
def _scores(shift: float = 0.0) -> tuple[dict, dict]:
    labels = {"a": np.array([0, 0, 1, 1]), "b": np.array([0, 1, 1, 0])}
    scores = {"a": np.array([0.1, 0.2, 0.8, 0.9]) + shift, "b": np.array([0.3, 0.6, 0.7, 0.2])}
    return scores, labels


def test_gate_passes_identical_and_fails_a_different_model() -> None:
    assert v6_r1_dada.compare_scores(_scores(), _scores())["passes"]
    flipped = ({"a": np.array([0.9, 0.8, 0.2, 0.1]), "b": _scores()[0]["b"]}, _scores()[1])
    r = v6_r1_dada.compare_scores(_scores(), flipped)
    assert not r["passes"] and r["macro_abs_delta"] > 0.1


def test_gate_fails_on_score_drift_even_when_ranks_agree() -> None:
    r = v6_r1_dada.compare_scores(_scores(), _scores(shift=0.05))
    assert r["macro_abs_delta"] == 0.0 and not r["passes"]


def test_gate_cli_stops_on_fail(tmp_path: Path) -> None:
    for name, shift in (("ref", 0.0), ("cand", 0.5)):
        protocol_b_eval.write_clip_scores(tmp_path / f"{name}.npz", *_scores(shift))
    with pytest.raises(SystemExit, match="G-S FAIL"):
        v6_r1_dada.run_gate(argparse.Namespace(reference=tmp_path / "ref.npz",
                                               candidate=tmp_path / "cand.npz",
                                               out=tmp_path / "g.json"))
    assert (tmp_path / "g.json").is_file()


# ---------------------------------------------------------------------------
# dada_whole_eval --ids-file
# ---------------------------------------------------------------------------
def test_read_ids(tmp_path: Path) -> None:
    p = tmp_path / "ids.txt"
    p.write_text("t02_v001\n\nt01_v003\nt01_v003\n")
    assert dada_whole_eval.read_ids(p) == ["t01_v003", "t02_v001"]
    p.write_text("\n")
    with pytest.raises(ValueError, match="names no source"):
        dada_whole_eval.read_ids(p)


def test_run_needs_exactly_one_set(tmp_path: Path) -> None:
    args = dada_whole_eval.build_arg_parser().parse_args([
        "--run", "s2024", "x.pt", "--annotation", "a.xlsx", "--census", "c.json",
        "--input-dir", str(tmp_path), "--data-dir", str(tmp_path), "--out-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="exactly one"):
        dada_whole_eval.run(args)


# ---------------------------------------------------------------------------
# integration: a real R1 model, stripped, loads strictly and scores the same
# ---------------------------------------------------------------------------
def test_stripped_r1_on_baked_u_scores_like_r1_on_its_grid(tmp_path: Path) -> None:
    from core.config import load_config
    from core.data.v5_grids import save_head
    from core.inference import adopt_checkpoint_architecture
    from core.models.kat_vad import KATVAD
    from core.tools import v5_stage1

    tiles, dim = constants.V5_GRID_TILES, constants.V5_GRID_DIM
    rng = np.random.default_rng(0)
    head = save_head(tmp_path / "h.npz", 1 + 0.1 * rng.standard_normal(dim),
                     0.1 * rng.standard_normal(dim), constants.VIDEOMAE_LN_EPS, 0.4,
                     0.1 * rng.standard_normal(dim), 0.5 + rng.random(dim))
    torch.manual_seed(0)
    cfg = v5_stage1.row_config("R1")
    r1 = KATVAD.from_config(cfg).eval()
    assert r1.grid_readout is not None and r1.motion_residual is not None
    r1.grid_readout.load_head(head)
    with torch.no_grad():  # a trained W_u (zero at init would hide a wrong motion input)
        for p in r1.motion_residual.parameters():
            p.normal_(0, 0.05)
    payload = {"model": r1.state_dict(), "config": cfg.to_dict(), "global_step": 7}

    stripped, _ = v6_r1_dada.strip_payload(payload)
    cfg2 = load_config(None, [])
    adopt_checkpoint_architecture(cfg2, stripped, [])
    a2 = KATVAD.from_config(cfg2).eval()
    a2.load_state_dict(stripped["model"])  # strict (C5)
    assert a2.grid_readout is None

    grid = torch.randn(1, 6, tiles, dim)
    x = torch.randn(1, 6, constants.HIDDEN_DIM)
    with torch.no_grad():
        baked = r1.grid_readout(grid).motion
        lengths = torch.tensor([6])
        feats = torch.randn(2, constants.HIDDEN_DIM)
        want = r1(torch.cat([x, baked, grid.reshape(1, 6, -1)], -1), lengths, class_feats=feats)
        got = a2(torch.cat([x, baked], -1), lengths, class_feats=feats)
    torch.testing.assert_close(got["cls_bin_logits"], want["cls_bin_logits"])
