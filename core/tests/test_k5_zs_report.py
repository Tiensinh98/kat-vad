"""The K5 vs LaGoVAD advisor table (data-free)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tests.test_zs_lagovad import _dota_read
from core.tests.test_zs_tad_dada import _tad_run
from core.tools import dada_whole_eval, k5_zs_report, protocol_b_eval, zs_compare_table
from core.tools.nexar_transfer_readout import read_tad_runs

LAG, K5 = constants.ZS_LAGOVAD_ARM, constants.KIM_MAIN_ARM
CKPT = "/drive/kim/K5/s2024/checkpoint_last.pt"
DAD_ROWS = 15
DAD_ACCIDENT = [13, 14]


def _dada(arm_dir: Path, shift: float, rng: np.random.Generator, per_run: dict) -> None:
    """A DADA read whose micro and macro agree with its own score file."""
    arm_dir.mkdir(parents=True)
    scores, labels = {}, {}
    for i in range(30):
        g = np.zeros(20, dtype=np.int64)
        g[6:12] = 1
        labels[f"d{i}"] = g
        scores[f"d{i}"] = rng.random(20) + shift * g
    aucs = protocol_b_eval.clip_aucs(scores, labels)
    ci = {"mean": 0.7, "low": 0.6, "high": 0.8}
    protocol_b_eval.write_clip_scores(arm_dir / protocol_b_eval.CLIP_SCORES_NPZ, scores, labels)
    (arm_dir / dada_whole_eval.CLIP_AUCS_JSON).write_text(json.dumps(aucs))
    (arm_dir / dada_whole_eval.READOUT_JSON).write_text(json.dumps({
        "per_run_macro": per_run, "macro": ci, "video_auc": None,
        "micro": zs_compare_table.pooled_reads(scores, labels, False)["micro"],
        "monotone_ruler_macro": ci, "middle_ruler_macro": ci, "checkpoints": {"s2024": CKPT},
    }))


def _dad(run_dir: Path, bump: float, rng: np.random.Generator) -> None:
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True)
    for prefix, videos in (("pos", 12), ("neg", 20)):
        for i in range(videos):
            g = np.zeros(DAD_ROWS, dtype=np.int64)
            if prefix == "pos":
                g[DAD_ACCIDENT] = 1
            np.savez(scores_dir / f"{prefix}_{i}.npz", score=rng.random(DAD_ROWS) + bump * g, gt=g)


def _results(run_dir: Path, ckpt: str = CKPT) -> None:
    (run_dir / k5_zs_report.RESULTS_JSON).write_text(json.dumps({"checkpoint": ckpt}))


def _layout(root: Path, with_dad: bool = True) -> argparse.Namespace:
    rng = np.random.default_rng(0)
    lag, kim = root / "lag" / "eval", root / "kim" / "eval"
    dota_ids = [f"src{i // 3}_{i:06d}" for i in range(30)]
    _dota_read(lag / "dota" / LAG / "B", 0.2, rng, dota_ids)
    _dota_read(kim / "dota" / K5, 1.0, rng, dota_ids)
    readout = kim / "dota" / K5 / protocol_b_eval.READOUT_JSON
    doc = json.loads(readout.read_text())
    readout.write_text(json.dumps({**doc, "checkpoints": {"s2024": CKPT}}))
    _dada(lag / "dada" / LAG, 0.2, rng, {"released": 0.6})
    _dada(kim / "dada" / K5, 1.0, rng, {"s2024": 0.7})
    _tad_run(lag / "tad" / LAG / "released", 0.5, rng)
    _tad_run(kim / "tad" / K5 / "s2024", 0.3, rng)
    _results(kim / "tad" / K5 / "s2024")
    _dad(lag / "dad" / LAG / "released", 0.2, rng)
    if with_dad:
        _dad(kim / "dad" / K5 / "s2024", 1.0, rng)
        _results(kim / "dad" / K5 / "s2024")
    return k5_zs_report.build_arg_parser().parse_args(
        ["--lagovad-root", str(root / "lag"), "--kim-root", str(root / "kim"),
         "--out-dir", str(root / "out")])


def _pass_t0(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    read = read_tad_runs(root / "lag" / "eval" / "tad" / LAG, ["released"], 0)
    monkeypatch.setattr(constants, "ZS_LAGOVAD_T0_MICRO", read["micro"])


class TestReport:
    def test_every_cell_and_only_two_rows(self, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        args = _layout(tmp_path)
        _pass_t0(tmp_path, monkeypatch)
        report = k5_zs_report.run(args)
        assert report["missing"] == []
        for s in report["sets"].values():
            assert set(s["rows"]) == {LAG, K5}
            assert all(s["rows"][r]["ap"] is not None for r in (LAG, K5))
        assert report["sets"]["dota"]["sentence"] == "above"
        assert report["sets"]["dad"]["rows"][K5]["macro"]["mean"] > 0.5
        text = (tmp_path / "out" / "k5_zs_report.md").read_text()
        assert "| **DAD** | AUC macro |" in text and "W0" not in text and "R4" not in text
        assert "PRELIMINARY (n = 1 seed)" in text and "Not a matched comparison" in text

    def test_dad_ruler_and_placebo_are_printed(self, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
        args = _layout(tmp_path)
        _pass_t0(tmp_path, monkeypatch)
        d = k5_zs_report.run(args)["dad_diagnostics"]
        assert d["ruler_macro"] == 1.0
        assert d["rows"][K5]["all_rows"]["positive"] > d["rows"][K5]["all_rows"]["placebo"]
        assert "_boot" not in d["rows"][K5]["same_position"]
        assert json.loads((tmp_path / "out" / "k5_zs_report.json").read_text())["missing"] == []

    def test_dad_not_read_yet_is_listed_never_filled(self, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
        args = _layout(tmp_path, with_dad=False)
        _pass_t0(tmp_path, monkeypatch)
        report = k5_zs_report.run(args)
        assert report["missing"] == ["dad"] and "dad" not in report["sets"]
        assert report["dad_diagnostics"] is None
        assert "**Not read yet:** DAD." in (tmp_path / "out" / "k5_zs_report.md").read_text()


class TestRefusals:
    def test_reads_from_two_checkpoints(self, tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        args = _layout(tmp_path)
        _pass_t0(tmp_path, monkeypatch)
        _results(tmp_path / "kim" / "eval" / "dad" / K5 / "s2024", "/other/checkpoint.pt")
        with pytest.raises(ValueError, match="different checkpoints"):
            k5_zs_report.run(args)
        assert not (tmp_path / "out").exists()

    def test_rows_on_other_items(self, tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
        args = _layout(tmp_path)
        _pass_t0(tmp_path, monkeypatch)
        path = tmp_path / "kim" / "eval" / "dada" / K5 / dada_whole_eval.CLIP_AUCS_JSON
        path.write_text(json.dumps({"d0": 0.6}))
        with pytest.raises(ValueError, match="not paired"):
            k5_zs_report.run(args)

    def test_micro_from_other_scores(self, tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        args = _layout(tmp_path)
        _pass_t0(tmp_path, monkeypatch)
        path = tmp_path / "kim" / "eval" / "dada" / K5 / dada_whole_eval.READOUT_JSON
        doc = json.loads(path.read_text())
        path.write_text(json.dumps({**doc, "micro": doc["micro"] + 0.01}))
        with pytest.raises(ValueError, match="not the same scores"):
            k5_zs_report.run(args)

    def test_gate_t0(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Gate T0"):
            k5_zs_report.run(_layout(tmp_path))
