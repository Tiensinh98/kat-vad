"""The K5-on-T2 table and ``nexar_eval --baseline`` (data-free)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from core import constants
from core.tests.test_k5_zs_report import CKPT, _layout, _pass_t0
from core.tools import k5_t2_report, nexar_eval, protocol_b_eval
from core.tools.rate_matched_eval import runs_from_args

LAG, K5, K5T2 = k5_t2_report.LAG, k5_t2_report.K5, k5_t2_report.K5T2
PLACEMENTS = constants.NEXAR_CROP_PLACEMENTS
T2_CKPT = "/drive/kim_t2/K5/s2024/checkpoint_last.pt"
NEXAR_CKPT = "/drive/kim/K5/s2024/checkpoint_last.pt"
CI = {"mean": 0.7, "low": 0.6, "high": 0.8}


def _nexar_row(root: Path, row: str, auc: float, split: str = constants.V2_SPLIT_NEXAR_VAL,
               ckpt: str = T2_CKPT) -> None:
    d = root / row
    d.mkdir(parents=True)
    (d / nexar_eval.ITEM_AUCS_JSON).write_text(json.dumps(
        {f"v{i}{nexar_eval.CROP_SEP}{p}": auc + 0.01 * i for i in range(20) for p in PLACEMENTS}))
    (d / nexar_eval.READOUT_JSON).write_text(json.dumps({
        "split": split, "videos": 225, "released_baseline": row == LAG,
        "checkpoints": {"s2024": ckpt},
        "crops": {"mean_over_placements": {"macro": CI}},
        "whole": {"macro": CI, "micro": 0.7, "micro_ap": 0.2, "clip_auc_max": 0.6,
                  "positive_frame_share": 0.03, "clip_oracle_micro": 0.75,
                  "middle_ruler_macro": CI},
    }))


def _rewrite(run_root: Path, old: str, new: str) -> None:
    for name in ("results.json", protocol_b_eval.READOUT_JSON):
        for path in run_root.rglob(name):
            path.write_text(path.read_text().replace(old, new))


def _roots(root: Path) -> argparse.Namespace:
    _layout(root)
    kim = root / "kim"
    t2 = root / "kim_t2"
    shutil.copytree(kim, t2)
    _rewrite(kim, CKPT, NEXAR_CKPT)
    _rewrite(t2, CKPT, T2_CKPT)
    shutil.rmtree(t2 / "eval" / "dada")
    nexar = t2 / "eval" / "nexar"
    _nexar_row(nexar, LAG, 0.6, ckpt="/drive/best.ckpt")
    _nexar_row(nexar, K5, 0.9, ckpt=NEXAR_CKPT)
    _nexar_row(nexar, K5T2, 0.8)
    return k5_t2_report.build_arg_parser().parse_args(
        ["--lagovad-root", str(root / "lag"), "--kim-root", str(kim), "--t2-root", str(t2),
         "--out-dir", str(root / "out")])


class TestReport:
    def test_every_set_and_three_rows(self, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
        args = _roots(tmp_path)
        _pass_t0(tmp_path, monkeypatch)
        report = k5_t2_report.run(args)
        assert list(report["sets"]) == ["nexar", "dota", "tad", "dad"]
        for s in report["sets"].values():
            assert set(s["rows"]) == {LAG, K5, K5T2}
        nexar = report["sets"]["nexar"]
        assert nexar["t2_minus_lagovad"]["delta"]["mean"] == pytest.approx(0.2)
        assert nexar["t2_minus_lagovad"]["sentence"] == "above"
        assert nexar["t2_minus_nexar"]["sentence"] == "below"
        assert report["sets"]["dad"]["t2_minus_nexar"]["delta"]["mean"] == pytest.approx(0.0)
        text = (tmp_path / "out" / "k5_t2_report.md").read_text()
        assert "| **Nexar val** | AUC macro, crops |" in text and "| **DAD** | AUC macro |" in text
        assert "decides nothing" in text and "W0" not in text
        assert "01_Accident" not in text  # synthetic ids: one category, no real names
        assert set(report["dad_diagnostics"]) == {"trained on T2", "trained on Nexar"}

    def test_tad_categories_strip_the_clip_number(self) -> None:
        name = k5_t2_report.CATEGORY_SUFFIX.sub("", "06_PedestrianOnRoad_027")
        assert name == "06_PedestrianOnRoad"


class TestRefusals:
    def test_t2_reads_from_the_nexar_checkpoint(self, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
        args = _roots(tmp_path)
        _pass_t0(tmp_path, monkeypatch)
        _rewrite(tmp_path / "kim_t2", T2_CKPT, NEXAR_CKPT)
        path = tmp_path / "kim_t2" / "eval" / "nexar" / K5T2 / nexar_eval.READOUT_JSON
        path.write_text(path.read_text().replace(T2_CKPT, NEXAR_CKPT))
        with pytest.raises(ValueError, match="share a checkpoint"):
            k5_t2_report.run(args)
        assert not (tmp_path / "out").exists()

    def test_nexar_test_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        args = _roots(tmp_path)
        _pass_t0(tmp_path, monkeypatch)
        nexar = tmp_path / "kim_t2" / "eval" / "nexar"
        shutil.rmtree(nexar / K5T2)
        _nexar_row(nexar, K5T2, 0.8, split=constants.V2_SPLIT_NEXAR_TEST)
        with pytest.raises(ValueError, match="nexar_val"):
            k5_t2_report.run(args)

    def test_lagovad_row_must_be_a_baseline_read(self, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        args = _roots(tmp_path)
        _pass_t0(tmp_path, monkeypatch)
        path = tmp_path / "kim_t2" / "eval" / "nexar" / LAG / nexar_eval.READOUT_JSON
        path.write_text(json.dumps({**json.loads(path.read_text()), "released_baseline": False}))
        with pytest.raises(ValueError, match="--baseline"):
            k5_t2_report.run(args)


class TestNexarEvalBaseline:
    BASE = ("--census", "c", "--clip-s1-dir", "x", "--data-dir", "d", "--out-dir", "o")

    def test_cli_takes_baseline_without_run(self) -> None:
        args = nexar_eval.build_arg_parser().parse_args(
            ["--baseline", "released", "best.ckpt", *self.BASE])
        assert args.run is None
        (run,) = runs_from_args(args.run, args.baseline)
        assert run.baseline is True

    def test_neither_refuses_before_any_file_is_read(self) -> None:
        args = nexar_eval.build_arg_parser().parse_args(list(self.BASE))
        with pytest.raises(ValueError, match="not both"):
            nexar_eval.run(args)
