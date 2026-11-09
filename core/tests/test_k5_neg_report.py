"""The K5neg vs K5 table (``KIM_RUN.md`` §17; data-free)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from core import constants
from core.tests.test_k5_whole_report import _nexar_row, _rewrite
from core.tests.test_k5_zs_report import CKPT, _layout
from core.tools import k5_neg_report, nexar_eval

K5, NEG = k5_neg_report.K5, k5_neg_report.NEG
NEG_CKPT = "/drive/kim_neg/K5/s2024/checkpoint_last.pt"


def _stats(d: Path, fa_peaks: dict[str, float]) -> None:
    """10 positives + negatives named by scene, peak as given (others 0.1)."""
    stats = {f"p{i}": {"label": 1, "scene": "Urban", "peak": 0.9, "mean": 0.3, "range": 0.8}
             for i in range(10)}
    scenes = ["Urban"] * 4 + ["Sub-urban"] * 4 + ["Highway"] * 2
    for i, scene in enumerate(scenes):
        v = f"n{i}"
        stats[v] = {"label": 0, "scene": scene, "peak": fa_peaks.get(v, 0.1), "mean": 0.05,
                    "range": 0.05}
    (d / nexar_eval.VIDEO_STATS_JSON).write_text(json.dumps(stats))


def _roots(root: Path, neg_crops: float = 0.9, k5_fa: dict[str, float] | None = None,
           neg_fa: dict[str, float] | None = None) -> argparse.Namespace:
    _layout(root)
    kim, neg = root / "kim", root / "kim_neg"
    shutil.copytree(kim, neg)
    _rewrite(neg, CKPT, NEG_CKPT)
    nexar = neg / "eval" / "nexar"
    _nexar_row(root / "ref", 0.9, CKPT)
    _nexar_row(nexar / K5, 0.9, CKPT)
    _nexar_row(nexar / NEG, neg_crops, NEG_CKPT)
    _stats(nexar / K5, k5_fa or {})
    _stats(nexar / NEG, neg_fa or {})
    return k5_neg_report.build_arg_parser().parse_args(
        ["--kim-root", str(kim), "--neg-root", str(neg), "--k5-reference-dir", str(root / "ref"),
         "--out-dir", str(root / "out")])


class TestReport:
    def test_no_difference_reads_no_claim(self, tmp_path: Path) -> None:
        report = k5_neg_report.run(_roots(tmp_path))
        assert list(report["sets"]) == ["nexar", "dota", "dada", "tad", "dad"]
        assert report["follow_up"]["verdict"] == "no claim"
        assert report["repro"]["gap"] == pytest.approx(0.0)
        text = (tmp_path / "out" / "k5_neg_report.md").read_text()
        assert "False alarms on Nexar val's 10 negative videos" in text
        assert "Verdict: **no claim**" in text and "W0" not in text

    def test_false_alarm_drop_makes_a_candidate(self, tmp_path: Path) -> None:
        k5_fa = {"n4": 0.8, "n5": 0.8, "n0": 0.7}  # 3 of 10 negatives alarm under K5
        report = k5_neg_report.run(_roots(tmp_path, k5_fa=k5_fa, neg_fa={"n0": 0.7}))
        fa = report["false_alarms"]
        assert fa[K5]["share"] == pytest.approx(0.3) and fa[NEG]["share"] == pytest.approx(0.1)
        assert fa[K5]["by_scene"]["Sub-urban"] == {"videos": 4, "share": pytest.approx(0.5)}
        assert fa[NEG]["by_scene"]["Sub-urban"]["share"] == pytest.approx(0.0)
        assert fa[K5]["by_scene"]["other"] == {"videos": 0, "share": None}
        assert report["follow_up"]["fa_drop"] == pytest.approx(0.2)
        assert report["follow_up"]["verdict"] == "candidate"

    def test_a_set_below_keeps_k5(self, tmp_path: Path) -> None:
        report = k5_neg_report.run(_roots(tmp_path, neg_crops=0.6, k5_fa={"n0": 0.9}))
        assert report["sets"]["nexar"]["sentence"] == "below"
        assert report["follow_up"]["verdict"] == "K5 stays"  # even with fewer false alarms

    def test_ranges_are_printed_per_set(self, tmp_path: Path) -> None:
        report = k5_neg_report.run(_roots(tmp_path))
        assert list(report["ranges"][NEG]) == ["nexar", "dota", "dada", "tad", "dad"]
        assert report["ranges"][K5]["nexar"] == pytest.approx(0.425)  # median of 0.8 x10, 0.05 x10


class TestRefusals:
    def test_k5_reread_must_reproduce_section_13(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        shutil.rmtree(tmp_path / "ref")
        _nexar_row(tmp_path / "ref", 0.85, CKPT)
        with pytest.raises(ValueError, match="NG4"):
            k5_neg_report.run(args)
        assert not (tmp_path / "out").exists()

    def test_missing_video_stats(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        (tmp_path / "kim_neg" / "eval" / "nexar" / NEG / nexar_eval.VIDEO_STATS_JSON).unlink()
        with pytest.raises(FileNotFoundError, match="video_stats"):
            k5_neg_report.run(args)

    def test_shared_checkpoint(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        _rewrite(tmp_path / "kim_neg", NEG_CKPT, CKPT)
        path = tmp_path / "kim_neg" / "eval" / "nexar" / NEG / nexar_eval.READOUT_JSON
        path.write_text(path.read_text().replace(NEG_CKPT, CKPT))
        with pytest.raises(ValueError, match="share a checkpoint"):
            k5_neg_report.run(args)

    def test_nexar_test_is_refused(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        d = tmp_path / "kim_neg" / "eval" / "nexar" / NEG
        shutil.rmtree(d)
        _nexar_row(d, 0.9, NEG_CKPT, split=constants.V2_SPLIT_NEXAR_TEST)
        _stats(d, {})
        with pytest.raises(ValueError, match="nexar_val"):
            k5_neg_report.run(args)
