"""The K5nc2 vs K5 / K5neg table (``KIM_RUN.md`` §27; data-free)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tests.test_k5_whole_report import _nexar_row, _rewrite
from core.tests.test_k5_zs_report import CKPT, _layout
from core.tools import k5_nc2_report, nexar_eval

K5, NEG, NC = k5_nc2_report.K5, k5_nc2_report.NEG, k5_nc2_report.NC
LNEG = constants.KIM_LNEG_ROW
NEG_CKPT = "/drive/kim_neg/K5/s2024/checkpoint_last.pt"
NC_CKPT = "/drive/kim_negcap/K5/s2024/checkpoint_last.pt"
FPS = int(constants.NEXAR_NOMINAL_FPS)
LENGTH = 30 * FPS  # 30 s per video
ONSET = 20 * FPS   # positives: GT from 20 s


def _stats(d: Path, pos_peaks: list[float], neg_peaks: list[float]) -> None:
    stats = {f"p{i}": {"label": 1, "scene": "Urban", "peak": p, "mean": 0.3, "range": 0.8}
             for i, p in enumerate(pos_peaks)}
    stats |= {f"n{i}": {"label": 0, "scene": "Highway", "peak": p, "mean": 0.05, "range": 0.05}
              for i, p in enumerate(neg_peaks)}
    (d / nexar_eval.VIDEO_STATS_JSON).write_text(json.dumps(stats))


def _scores(d: Path, n_pos: int, n_neg: int, spike_neg: int = 0, spike_pre: int = 0) -> None:
    """Flat 0.1 curves; ``spike_neg`` negatives / ``spike_pre`` positives get a 0.9 spike."""
    (d / "scores").mkdir(parents=True, exist_ok=True)
    for i in range(n_pos):
        s, gt = np.full(LENGTH, 0.1), np.zeros(LENGTH, dtype=np.int64)
        gt[ONSET:] = 1
        s[ONSET:] = 0.9
        if i < spike_pre:
            s[2 * FPS] = 0.9            # far before onset (first chunk)
            s[ONSET - FPS] = 0.9        # inside the last 5 s
        np.savez(d / "scores" / f"p{i}.npz", score=s, gt=gt)
    for i in range(n_neg):
        s = np.full(LENGTH, 0.1)
        if i < spike_neg:
            s[FPS] = 0.9
        np.savez(d / "scores" / f"n{i}.npz", score=s, gt=np.zeros(LENGTH, dtype=np.int64))


POS = [0.9] * 10
NEG_PEAKS = [0.1 * i for i in range(10)]  # 0.0 ... 0.9


def _roots(root: Path, nc_crops: float = 0.9, nc_pos: list[float] | None = None,
           k5_pos: list[float] | None = None, new: str = NC) -> argparse.Namespace:
    _layout(root)
    dirname = {NC: "kim_negcap", LNEG: constants.KIM_LNEG_OUTPUT_DIRNAME}[new]
    new_ckpt = NC_CKPT.replace("kim_negcap", dirname)
    kim, neg, nc = root / "kim", root / "kim_neg", root / dirname
    for target, ckpt in ((neg, NEG_CKPT), (nc, new_ckpt)):
        shutil.copytree(kim, target)
        _rewrite(target, CKPT, ckpt)
    _nexar_row(root / "ref", 0.9, CKPT)
    rows = {K5: (neg / "eval" / "nexar" / K5, 0.9, CKPT, k5_pos or POS),
            NEG: (neg / "eval" / "nexar" / NEG, 0.9, NEG_CKPT, POS),
            new: (nc / "eval" / "nexar" / new, nc_crops, new_ckpt, nc_pos or POS)}
    for d, auc, ckpt, pos in rows.values():
        _nexar_row(d, auc, ckpt)
        _stats(d, pos, NEG_PEAKS)
    _scores(root / "k5_scores", 10, 10, spike_neg=4, spike_pre=2)
    _scores(root / "neg_scores", 10, 10, spike_neg=1)
    _scores(rows[new][0], 10, 10, spike_neg=2, spike_pre=1)
    return k5_nc2_report.build_arg_parser().parse_args(
        ["--kim-root", str(kim), "--neg-root", str(neg), "--nc-root", str(nc), "--new-row", new,
         "--k5-reference-dir", str(root / "ref"), "--k5-scores-dir", str(root / "k5_scores"),
         "--neg-scores-dir", str(root / "neg_scores"), "--out-dir", str(root / "out")])


class TestPieces:
    def test_matched_detection_is_section_17_1_arithmetic(self) -> None:
        stats = {f"n{i}": {"label": 0, "peak": 0.1 * i} for i in range(10)}
        stats |= {f"p{i}": {"label": 1, "peak": 0.05 + 0.1 * i} for i in range(10)}
        out = k5_nc2_report.matched_detection(stats, (0.2,))["0.2"]
        assert out["cut"] == pytest.approx(np.quantile([0.1 * i for i in range(10)], 0.8))
        assert out["fa"] == pytest.approx(0.2)
        assert out["detect"] == pytest.approx(0.3)  # 0.75, 0.85, 0.95 pass the cut

    def test_chunk_rates_split_far_last_and_negatives(self, tmp_path: Path) -> None:
        _scores(tmp_path, 4, 2, spike_neg=1, spike_pre=1)
        c = k5_nc2_report.chunk_rates(tmp_path)
        far_chunks = (ONSET - 5 * FPS) // (5 * FPS)        # 3 chunks per positive
        assert c["far_pre_onset"] == {"chunks": 4 * far_chunks, "share": pytest.approx(1 / 12)}
        assert c["negatives"] == {"chunks": 2 * 6, "share": pytest.approx(1 / 12)}
        assert c["last_pre_onset"] == {"videos": 4, "share": pytest.approx(0.25)}


class TestReport:
    def test_no_difference_reads_no_claim(self, tmp_path: Path) -> None:
        report = k5_nc2_report.run(_roots(tmp_path))
        assert list(report["sets"]) == ["nexar", "dota", "dada", "tad", "dad"]
        assert set(report["sets"]["dota"]["rows"]) == {K5, NC}
        assert set(report["vs_neg"]["dota"]["rows"]) == {K5, NC}  # K5neg sits in the control slot
        assert report["follow_up"]["verdict"] == "no claim"
        chunks = report["chunks"]
        assert chunks[K5]["negatives"]["share"] > chunks[NEG]["negatives"]["share"]
        text = (tmp_path / "out" / "k5_nc2_report.md").read_text()
        assert "Verdict: **no claim**" in text and "K5nc2 - K5neg" in text

    def test_matched_detection_gain_makes_a_candidate(self, tmp_path: Path) -> None:
        k5_pos = [0.95, 0.95] + [0.5] * 8   # at FA 0.2 (cut 0.72) K5 detects 0.2, K5nc2 1.0
        report = k5_nc2_report.run(_roots(tmp_path, k5_pos=k5_pos))
        assert report["matched"][K5]["0.2"]["detect"] == pytest.approx(0.2)
        assert report["follow_up"]["detect_gain"] == pytest.approx(0.8)
        assert report["follow_up"]["verdict"] == "candidate"

    def test_lneg_row_reads_with_section_28_labels(self, tmp_path: Path) -> None:
        """§28 reuses the read: same verdict logic, its own row, file and NL clauses."""
        report = k5_nc2_report.run(_roots(tmp_path, new=LNEG))
        assert report["new_row"] == LNEG and report["rule"] == "§28"
        assert set(report["sets"]["dota"]["rows"]) == {K5, LNEG}
        assert report["follow_up"]["verdict"] == "no claim"
        text = (tmp_path / "out" / f"{constants.KIM_LNEG_REPORT_STEM}.md").read_text()
        assert "§28 NL7" in text and "K5Lneg - K5neg" in text and "NC" not in text
        assert not (tmp_path / "out" / "k5_nc2_report.md").exists()

    def test_a_set_below_keeps_k5(self, tmp_path: Path) -> None:
        report = k5_nc2_report.run(_roots(tmp_path, nc_crops=0.6, k5_pos=[0.5] * 10))
        assert report["sets"]["nexar"]["sentence"] == "below"
        assert report["follow_up"]["verdict"] == "K5 stays"  # even with a detection gain


class TestRefusals:
    def test_k5_reread_must_reproduce_section_13(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        shutil.rmtree(tmp_path / "ref")
        _nexar_row(tmp_path / "ref", 0.85, CKPT)
        with pytest.raises(ValueError, match="NG4"):
            k5_nc2_report.run(args)

    def test_nc_sharing_k5negs_checkpoint(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        _rewrite(tmp_path / "kim_negcap", NC_CKPT, NEG_CKPT)
        path = tmp_path / "kim_negcap" / "eval" / "nexar" / NC / nexar_eval.READOUT_JSON
        path.write_text(path.read_text().replace(NC_CKPT, NEG_CKPT))
        with pytest.raises(ValueError, match="share a checkpoint"):
            k5_nc2_report.run(args)

    def test_missing_curves(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        shutil.rmtree(tmp_path / "neg_scores")
        with pytest.raises(FileNotFoundError, match="no saved scores"):
            k5_nc2_report.run(args)
        assert not (tmp_path / "out").exists()
