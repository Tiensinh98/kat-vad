"""v2 figures (core.tools.v2_plots): input checks, clip picking, and the numbers on the figures.

CPU-only, data-free; renders real PNGs into tmp_path (Agg backend).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.metrics import pooled_metrics
from core.tools import v2_plots
from core.tools.protocol_b_eval import write_clip_scores


def _labels(n: int = 12, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(n):
        lab = np.zeros(40, dtype=np.int64)
        start = int(rng.integers(5, 25))
        lab[start : start + 10] = 1
        out[f"vid{k:02d}_{k:06d}"] = lab
    return out


def _write(path: Path, labels: dict[str, np.ndarray], gain: float, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    write_clip_scores(path, {v: gain * lab + rng.random(len(lab)) for v, lab in labels.items()},
                      labels)
    return path


@pytest.fixture()
def files(tmp_path: Path) -> dict[str, Path]:
    labels = _labels()
    return {
        "A0": _write(tmp_path / "a0.npz", labels, 0.2, 1),
        "A3-ign": _write(tmp_path / "a3ign.npz", labels, 1.0, 2),
        "prior": _write(tmp_path / "prior.npz", labels, 0.5, 3),
    }


def _argv(cmd: str, files: dict[str, Path], out: Path, *extra: str) -> list[str]:
    return [cmd, "--out-dir", str(out), "--arm", "A0", str(files["A0"]),
            "--arm", "A3-ign", str(files["A3-ign"]), "--prior", str(files["prior"]), *extra]


class TestInputs:
    def test_files_on_different_clips_are_refused(self, tmp_path: Path, files: dict) -> None:
        other = _write(tmp_path / "other.npz", _labels(seed=9), 1.0, 4)
        with pytest.raises(ValueError, match=r"different clips|disagree"):
            v2_plots.load_series([["A0", str(files["A0"])], ["X", str(other)]], None)

    def test_more_arms_than_palette_slots_are_refused(self, files: dict) -> None:
        arms = [[f"a{i}", str(files["A0"])] for i in range(len(v2_plots.SERIES_COLORS) + 1)]
        with pytest.raises(ValueError, match="at most"):
            v2_plots.load_series(arms, None)


class TestPick:
    def test_picks_spread_worst_to_best(self, files: dict) -> None:
        series, _ = v2_plots.load_series([["A3-ign", str(files["A3-ign"])]], None)
        picks = v2_plots.pick_clips(series[0], 3)
        aucs = [v2_plots.frame_auc(series[0].scores[v], series[0].labels[v]) for v in picks]
        assert len(picks) == 3 and aucs == sorted(aucs)
        assert len(v2_plots.pick_clips(series[0], 100)) == 12


class TestRoc:
    def test_writes_the_figure_and_the_read_outs_own_numbers(
        self, tmp_path: Path, files: dict
    ) -> None:
        v2_plots.main(_argv("roc", files, tmp_path / "roc", "--title", "t"))
        rec = json.loads((tmp_path / "roc" / v2_plots.ROC_JSON).read_text())
        assert (tmp_path / "roc" / v2_plots.ROC_PNG).stat().st_size > 0
        series, _ = v2_plots.load_series([["A3-ign", str(files["A3-ign"])]], None)
        s = series[0]
        ids = sorted(s.labels)
        want = pooled_metrics([s.scores[v] for v in ids], [s.labels[v] for v in ids],
                              constants.SCORE_NORM_MINMAX)
        assert rec["series"]["A3-ign"]["auc"] == pytest.approx(want["auc"])
        assert rec["series"]["A3-ign"]["auc_macro"] == pytest.approx(want["auc_macro"])
        assert set(rec["series"]) == {"A0", "A3-ign", v2_plots.PRIOR_NAME}
        assert rec["series"]["A3-ign"]["auc"] > rec["series"]["A0"]["auc"]


class TestCurves:
    def test_writes_one_png_per_clip_and_a_grid(self, tmp_path: Path, files: dict) -> None:
        out = tmp_path / "curves"
        v2_plots.main(_argv("curves", files, out, "--n", "3", "--sort-by", "A3-ign"))
        pngs = sorted(p.name for p in out.glob("*.png"))
        assert len(pngs) == 4 and v2_plots.GRID_PNG in pngs

    def test_explicit_clips_must_exist(self, tmp_path: Path, files: dict) -> None:
        with pytest.raises(ValueError, match="not in the score files"):
            v2_plots.main(_argv("curves", files, tmp_path / "c", "--clips", "nope_000001"))

    def test_sort_by_must_name_an_arm(self, tmp_path: Path, files: dict) -> None:
        with pytest.raises(ValueError, match="--sort-by"):
            v2_plots.main(_argv("curves", files, tmp_path / "c", "--sort-by", "A9"))
