"""Tests for ``core.tools.k5_lag_shift`` (``KIM_RUN.md`` §23). Synthetic, data-free."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import k5_lag_shift as ls
from core.tools import protocol_b_eval
from core.tools import tadz_readout as td

FPS = 10.0  # synthetic Nexar videos: 10 steps per second
LAG_S = 0.5  # the synthetic response arrives this late


def _late_curve(gt: np.ndarray, lag_steps: int, rng: np.random.Generator) -> np.ndarray:
    """A response equal to the label delayed by ``lag_steps``, plus small noise."""
    late = np.r_[np.zeros(lag_steps), gt[: len(gt) - lag_steps]]
    return np.asarray(late + 0.05 * rng.random(len(gt)), dtype=np.float64)


def _span_curves(n: int, length: int, lag_steps: int, seed: int,
                 ids: list[str] | None = None) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    scores, labels = {}, {}
    for i in range(n):
        gt = np.zeros(length)
        start = int(rng.integers(length // 4, length // 2))
        gt[start: start + length // 5] = 1
        key = ids[i] if ids else f"{i:05d}"
        scores[key] = _late_curve(gt, lag_steps, rng)
        labels[key] = gt.astype(np.int64)
    return scores, labels


def _write_nexar(root: Path, lag_steps: int) -> tuple[Path, Path, Path]:
    scores, labels = _span_curves(12, 80, lag_steps, seed=1)
    run = root / "nexar"
    (run / td.SCORES_DIRNAME).mkdir(parents=True)
    for v in scores:
        np.savez(run / td.SCORES_DIRNAME / f"{v}.npz", score=scores[v].astype(np.float32),
                 gt=labels[v].astype(np.float32))
    loaded, gts = td.load_run(run)
    readout = run / "nexar_readout.json"
    readout.write_text(json.dumps({"whole": {"macro": {"mean": ls.macro(loaded, gts)}}}))
    census = root / "census.json"
    census.write_text(json.dumps({"videos": {v: {"duration_s": len(scores[v]) / FPS}
                                             for v in scores}}))
    return run, readout, census


def _write_zero_shot(run_dir: Path, name: str, lag_steps: int, seed: int) -> None:
    ids = ([f"src{i // 3}_{i:06d}" for i in range(12)] if name == ls.DOTA
           else [f"t01_v{i:03d}" for i in range(12)])
    scores, labels = _span_curves(12, 60, lag_steps, seed, ids)
    run_dir.mkdir(parents=True)
    protocol_b_eval.write_clip_scores(run_dir / protocol_b_eval.CLIP_SCORES_NPZ, scores, labels)
    name_json = protocol_b_eval.READOUT_JSON if name == ls.DOTA else ls.DADA_READOUT_JSON
    doc: dict[str, object] = {"macro": {"mean": ls.macro(scores, labels)}}
    if name == ls.DADA:
        doc["micro"] = td.micro(scores, labels)
    (run_dir / name_json).write_text(json.dumps(doc))


def _args(tmp_path: Path, lag_steps: int = 5) -> argparse.Namespace:
    run, readout, census = _write_nexar(tmp_path, lag_steps)
    _write_zero_shot(tmp_path / "dota", ls.DOTA, 3, seed=2)
    _write_zero_shot(tmp_path / "dada", ls.DADA, 2, seed=3)
    return ls.build_arg_parser().parse_args([
        "--nexar-run", str(run), "--nexar-readout", str(readout), "--census", str(census),
        "--dota", str(tmp_path / "dota"), "--dada", str(tmp_path / "dada"),
        "--out-dir", str(tmp_path / "out")])


class TestShift:
    def test_grid_is_fixed(self) -> None:
        g = ls.grid()
        assert len(g) == 31 and g[0] == 0.0 and g[-1] == constants.LAGSHIFT_GRID_MAX_S
        assert g[1] == constants.LAGSHIFT_GRID_STEP_S

    def test_zero_is_identity_and_integer_shift_advances(self) -> None:
        x = np.arange(6, dtype=float)
        assert np.array_equal(ls.shift_curve(x, 0), x)
        assert ls.shift_curve(x, 2).tolist() == [2, 3, 4, 5, 5, 5]

    def test_fractional_shift_interpolates(self) -> None:
        assert ls.shift_curve(np.array([0.0, 2.0, 4.0]), 0.5).tolist() == [1.0, 3.0, 4.0]

    def test_negative_shift_refused(self) -> None:
        with pytest.raises(ValueError, match="advances"):
            ls.shift_curve(np.zeros(3), -1)


class TestFit:
    def test_recovers_a_known_lag(self) -> None:
        scores, labels = _span_curves(12, 80, int(LAG_S * FPS), seed=4)
        fit = ls.fit_delta(scores, labels, dict.fromkeys(scores, FPS))
        assert fit["delta"] == pytest.approx(LAG_S, abs=constants.LAGSHIFT_GRID_STEP_S)
        assert fit["macro_by_delta"][fit["delta"]] > fit["macro_by_delta"][0.0]

    def test_tie_takes_the_smallest_delta(self) -> None:
        scores, labels = _span_curves(4, 40, 0, seed=5)
        flat = {v: np.ones_like(s) for v, s in scores.items()}  # every delta ties at 0.5
        assert ls.fit_delta(flat, labels, dict.fromkeys(flat, FPS))["delta"] == 0.0


class TestRun:
    def test_report_and_follow_up(self, tmp_path: Path) -> None:
        report = ls.run(_args(tmp_path))
        assert report["fit"]["delta"] > 0
        for name in (ls.DOTA, ls.DADA):
            s = report["sets"][name]
            assert s["macro_sentence"] in ("above", "below", "not separated")
            assert len(s["grid_printed"]) == len(ls.grid())
        assert report["sets"][ls.DOTA]["clusters"] == 4
        text = (tmp_path / "out" / f"{constants.LAGSHIFT_REPORT_STEM}.md").read_text()
        assert "LS7" in text and "offline protocol change" in text

    def test_gate_g1_refuses_other_scores(self, tmp_path: Path) -> None:
        args = _args(tmp_path)
        readout = args.dada / ls.DADA_READOUT_JSON
        doc = json.loads(readout.read_text())
        doc["macro"]["mean"] += 0.01
        readout.write_text(json.dumps(doc))
        with pytest.raises(ValueError, match="G1"):
            ls.run(args)

    def test_census_must_cover_every_video(self, tmp_path: Path) -> None:
        args = _args(tmp_path)
        census = json.loads(args.census.read_text())
        census["videos"].pop(sorted(census["videos"])[0])
        args.census.write_text(json.dumps(census))
        with pytest.raises(ValueError, match="G3"):
            ls.run(args)

    def test_follow_up_sentences(self) -> None:
        both = {ls.DOTA: {"macro_sentence": "above"}, ls.DADA: {"macro_sentence": "above"}}
        one = {ls.DOTA: {"macro_sentence": "above"}, ls.DADA: {"macro_sentence": "below"}}
        none = {ls.DOTA: {"macro_sentence": "not separated"},
                ls.DADA: {"macro_sentence": "below"}}
        assert "may propose" in ls.follow_up(both)
        assert "only" in ls.follow_up(one)
        assert "not supported" in ls.follow_up(none)
