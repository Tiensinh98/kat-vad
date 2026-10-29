"""v5 check C: change maps vs the centre prior on the same grid and steps (PREREG_V5.md §3.7)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import v5_centre_check as cc
from core.tools import v5_stage0 as s0
from core.tools.v5_maps import centre_prior

GRID14 = constants.V5_GRID_FINE
CENTRE_BOX = (560.0, 300.0, 720.0, 420.0)  # around the frame centre (1280 x 720)
LEFT_BOX = (0.0, 0.0, 300.0, 720.0)  # left edge, away from the centre


def _clip(
    box: tuple[float, ...], n: int = 60
) -> tuple[np.ndarray, np.ndarray, dict[int, list[list[float]]]]:
    steps = np.arange(0, n, constants.V5_STAGE0_STRIDE)
    labels = np.zeros(n, dtype=np.int64)
    labels[n // 2 :] = 1
    return steps, labels, {f: [list(box)] for f in range(n // 2, n)}


def _left_hot(steps: int, grid: int = GRID14) -> np.ndarray:
    rng = np.random.default_rng(3)
    delta = rng.uniform(0.1, 0.2, size=(steps, grid, grid))
    delta[:, :, :2] += 1.0  # the two left columns change most
    out = delta.reshape(steps, -1)
    out[0] = np.nan  # lag-1 change: the first row is undefined
    return out


class TestPool:
    def test_block_mean_and_nan_rows(self) -> None:
        delta = np.arange(2 * 196, dtype=float).reshape(2, 196)
        delta[0] = np.nan
        pooled = cc.pool_map(delta, 2)
        assert pooled.shape == (2, 49) and np.isnan(pooled[0]).all()
        grid = delta[1].reshape(14, 14)
        assert pooled[1, 0] == pytest.approx(grid[:2, :2].mean())

    def test_identity_and_bad_factor(self) -> None:
        delta = np.ones((3, 196))
        assert cc.pool_map(delta, 1) is delta
        with pytest.raises(ValueError, match="divisible"):
            cc.pool_map(np.ones((3, 49)), 2)


class TestMatchedPrior:
    def test_nan_where_the_map_is_undefined(self) -> None:
        delta = _left_hot(5)
        prior = cc.matched_prior(delta)
        assert np.isnan(prior[0]).all() and np.isfinite(prior[1:]).all()
        np.testing.assert_allclose(prior[1], centre_prior(GRID14))

    def test_tie_on_14_takes_tile_6_6(self) -> None:
        assert int(np.argmax(centre_prior(GRID14))) == 6 * GRID14 + 6


class TestClipReads:
    def test_centre_map_ties_the_prior(self) -> None:
        steps, labels, boxes = _clip(CENTRE_BOX)
        delta = cc.matched_prior(_left_hot(len(steps)))
        read = cc.clip_reads(delta, steps, labels, boxes)
        assert read is not None
        assert read["hit"] == read["hit_centre"] == 1.0
        assert read["mass"] == pytest.approx(read["mass_centre"])

    def test_off_centre_agent_map_beats_the_prior(self) -> None:
        steps, labels, boxes = _clip(LEFT_BOX)
        read = cc.clip_reads(_left_hot(len(steps)), steps, labels, boxes)
        assert read is not None
        assert read["hit"] == 1.0 and read["hit_centre"] == 0.0
        assert read["mass"] > read["mass_centre"]

    def test_centre_agent_map_loses_to_the_prior(self) -> None:
        steps, labels, boxes = _clip(CENTRE_BOX)
        read = cc.clip_reads(_left_hot(len(steps)), steps, labels, boxes)
        assert read is not None
        assert read["hit"] == 0.0 and read["hit_centre"] == 1.0

    def test_no_boxed_anomaly_is_none(self) -> None:
        steps, labels, _ = _clip(LEFT_BOX)
        assert cc.clip_reads(_left_hot(len(steps)), steps, labels, {}) is None


class TestDecide:
    def test_rule(self) -> None:
        def primary(low: float) -> dict:
            return {"hit_minus_centre": {"mean": low + 0.01, "low": low, "high": low + 0.02}}

        assert cc.decide(primary(0.001), valid=True) == cc.BEATS
        assert cc.decide(primary(0.0), valid=True) == cc.LOSES
        assert cc.decide(primary(-0.1), valid=True) == cc.LOSES
        assert cc.decide({"hit_minus_centre": None}, valid=True) == cc.LOSES
        assert cc.decide(primary(0.5), valid=False) == cc.INVALID

    def test_g_c0(self) -> None:
        ref = {"summary": {cc.PRIMARY: {"hit_minus_chance": {"mean": 0.2878}}}}
        ok = cc.g_c0({"hit_minus_chance": {"mean": 0.28785, "low": 0.0, "high": 0.0}}, ref)
        bad = cc.g_c0({"hit_minus_chance": {"mean": 0.29, "low": 0.0, "high": 0.0}}, ref)
        assert ok["pass"] and not bad["pass"]
        assert not cc.g_c0({"hit_minus_chance": None}, ref)["pass"]


class TestRunCheck:
    def _inputs(self, clips: int) -> s0.DevInputs:
        maps: dict[str, dict[str, np.ndarray]] = {}
        labels: dict[str, np.ndarray] = {}
        boxes: dict[str, dict[int, list[list[float]]]] = {}
        ego: dict[str, bool] = {}
        bins: dict[str, str | None] = {}
        for i in range(clips):
            v = f"v{i:03d}_x_{i:06d}"  # one DoTA source per clip
            steps, lab, box = _clip(LEFT_BOX)
            d14 = _left_hot(len(steps))
            maps[v] = {
                "steps": steps,
                "dc14": d14,
                "d14": d14,
                "d7": cc.pool_map(d14, 2),
            }
            labels[v], boxes[v], ego[v], bins[v] = lab, box, i % 2 == 0, s0.BIN_NAMES[0]
        return s0.DevInputs({}, {}, [], maps, labels, boxes, ego, bins)

    def test_end_to_end(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs = self._inputs(12)
        monkeypatch.setattr(s0, "load_dev", lambda _args: inputs)
        chance = float(np.mean([(s0.tile_weights([list(LEFT_BOX)], GRID14) > 0).mean()]))
        ref = tmp_path / "stage0_readout.json"
        ref.write_text(
            json.dumps({"summary": {cc.PRIMARY: {"hit_minus_chance": {"mean": 1.0 - chance}}}})
        )
        args = argparse.Namespace(stage0_readout=ref, out_dir=tmp_path / "REPORTS")
        out = cc.run_check(args)
        assert out["g_c0"]["pass"]
        assert out["decision"] == cc.BEATS
        assert set(out["summary"]) == {name for name, _, _ in cc.REPS}
        assert out["summary"][cc.PRIMARY]["hit_minus_centre"]["mean"] == pytest.approx(1.0)
        md = (tmp_path / "REPORTS" / cc.READOUT_MD).read_text()
        assert md.startswith("# v5 check C") and cc.BEATS in md

    def test_failed_gate_is_invalid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs = self._inputs(4)
        monkeypatch.setattr(s0, "load_dev", lambda _args: inputs)
        ref = tmp_path / "stage0_readout.json"
        ref.write_text(json.dumps({"summary": {cc.PRIMARY: {"hit_minus_chance": {"mean": -1}}}}))
        out = cc.run_check(argparse.Namespace(stage0_readout=ref, out_dir=tmp_path / "R"))
        assert out["decision"] == cc.INVALID

    def test_cli_parses(self, tmp_path: Path) -> None:
        args = cc.build_arg_parser().parse_args(
            ["--reports", str(tmp_path / "a.json"), "--stage0-readout", str(tmp_path / "r.json")]
        )
        assert args.out_dir.parts[-3:] == (
            constants.V5_OUTPUT_DIRNAME,
            constants.V5_CHECK_C_DIRNAME,
            "REPORTS",
        )
