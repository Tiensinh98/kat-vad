"""K5 vs LaGoVAD on MSAD test (``KIM_RUN.md`` §18; data-free)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import k5_msad_report as rep
from core.tools import tadz_readout as td

IDS = {"Traffic_accident_1": 12, "Traffic_accident_2": 10, "Fire_1": 14,
       "MSAD_normal_testing_1": 30, "MSAD_normal_testing_2": 20}


def _labels(vid: str, n: int) -> np.ndarray:
    g = np.zeros(n, dtype=np.int64)
    if not vid.startswith(constants.MSAD_ZS_NORMAL_PREFIX):
        g[n // 3: 2 * n // 3] = 1
    return g


def _write_run(d: Path, scores: dict[str, np.ndarray]) -> None:
    (d / "scores").mkdir(parents=True)
    for vid, n in IDS.items():
        np.savez(d / "scores" / f"{vid}.npz", score=scores[vid], gt=_labels(vid, n))


def _runs(root: Path, k5_shift: float = 0.0, k5_labels_off: bool = False) -> None:
    rng = np.random.default_rng(0)
    lag = {v: rng.random(n) for v, n in IDS.items()}
    k5 = {v: s + k5_shift * _labels(v, len(s)) for v, s in lag.items()}
    _write_run(root / "eval" / rep.LAG / constants.ZS_LAGOVAD_RUN, lag)
    _write_run(root / "eval" / rep.K5 / "s2024", k5)
    if k5_labels_off:
        p = root / "eval" / rep.K5 / "s2024" / "scores" / "Fire_1.npz"
        np.savez(p, score=k5["Fire_1"], gt=np.zeros(IDS["Fire_1"], dtype=np.int64))


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(constants, "MSAD_ZS_TEST_VIDEOS", len(IDS))
    monkeypatch.setattr(rep, "TWO_CLASS", {rep.TRAFFIC: 2, rep.FULL: 3})


def _args(root: Path) -> argparse.Namespace:
    return rep.build_arg_parser().parse_args(["--root", str(root)])


def _gate_to(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s, g = td.load_run(root / "eval" / rep.LAG / constants.ZS_LAGOVAD_RUN)
    monkeypatch.setattr(constants, "MSAD_ZS_GATE_A_MICRO", td.micro(s, g))


class TestPure:
    def test_traffic_set_is_class_plus_every_normal(self) -> None:
        assert rep.set_ids(list(IDS), rep.TRAFFIC) == [
            "MSAD_normal_testing_1", "MSAD_normal_testing_2", "Traffic_accident_1",
            "Traffic_accident_2"]
        assert len(rep.set_ids(list(IDS), rep.FULL)) == len(IDS)

    def test_equal_video_micro_removes_length_weight(self) -> None:
        # one long normal video scored high; frame weighting punishes it more than video weighting
        labels = {"a": np.array([0, 1]), "n": np.zeros(98, dtype=np.int64)}
        scores = {"a": np.array([0.0, 0.5]), "n": np.full(98, 0.9)}
        assert rep.equal_video_micro(scores, labels) > td.micro(scores, labels)

    def test_gate_m0_raises(self) -> None:
        assert rep.gate_m0(constants.MSAD_ZS_GATE_A_MICRO)["pass"]
        with pytest.raises(ValueError, match="M0"):
            rep.gate_m0(constants.MSAD_ZS_GATE_A_MICRO + 0.01)


class TestRun:
    def test_identical_rows_not_separated(self, tmp_path: Path, small: None,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        _runs(tmp_path)
        _gate_to(tmp_path, monkeypatch)
        report = rep.run(_args(tmp_path))
        for s in rep.SET_ORDER:
            assert report["sets"][s]["delta_macro"]["mean"] == pytest.approx(0.0)
            assert report["sets"][s]["sentence"] == "not separated"
        assert report["sets"][rep.TRAFFIC]["videos"] == 4
        md = (tmp_path / "REPORTS" / f"{rep.REPORT_STEM}.md").read_text()
        assert "MSAD-traffic" in md and str(constants.MSAD_ZS_PAPER_MICRO) in md

    def test_better_k5_reads_above(self, tmp_path: Path, small: None,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        _runs(tmp_path, k5_shift=5.0)
        _gate_to(tmp_path, monkeypatch)
        report = rep.run(_args(tmp_path))
        assert report["sets"][rep.FULL]["delta_macro"]["low"] > 0
        assert report["sets"][rep.FULL]["sentence"] == "above"

    def test_refuses_other_labels(self, tmp_path: Path, small: None,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
        _runs(tmp_path, k5_labels_off=True)
        _gate_to(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="labels differ"):
            rep.run(_args(tmp_path))

    def test_refuses_m0_miss(self, tmp_path: Path, small: None) -> None:
        _runs(tmp_path)
        with pytest.raises(ValueError, match="M0"):
            rep.run(_args(tmp_path))

    def test_refuses_protocol_size(self, tmp_path: Path, small: None,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        _runs(tmp_path)
        _gate_to(tmp_path, monkeypatch)
        monkeypatch.setattr(rep, "TWO_CLASS", {rep.TRAFFIC: 2, rep.FULL: 96})
        with pytest.raises(ValueError, match="two-class"):
            rep.run(_args(tmp_path))
