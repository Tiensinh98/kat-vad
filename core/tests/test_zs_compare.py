"""The zero-shot comparison table and the DAD position diagnostics (data-free)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.metrics import frame_ap, frame_auc, normalize_scores
from core.tools import dad_position_diag as diag
from core.tools import protocol_b_eval, zs_compare_table
from core.tools.extract_video_features import causal_clip_indices

LAG, W0, R4 = constants.ZS_LAGOVAD_ARM, constants.ZS_CONTROL_ARM, constants.ZS_SA2_ARM
ROWS_PER_VIDEO = 15
ACCIDENT = [13, 14]
RESAMPLES = 50


def _clips(seed: int, clips: int = 6, length: int = 20) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    labels = {f"v{i}": (np.arange(length) >= length // 2).astype(np.int64) for i in range(clips)}
    scores = {v: rng.random(length) + 0.5 * g for v, g in labels.items()}
    return scores, labels


def _eval_run(run_dir: Path, scores: dict, labels: dict) -> None:
    (run_dir / "scores").mkdir(parents=True)
    for v in labels:
        np.savez(run_dir / "scores" / f"{v}.npz", score=scores[v], gt=labels[v])


def _micro(scores: dict, labels: dict, minmax: bool) -> float:
    return zs_compare_table.pooled_reads(scores, labels, minmax)["micro"]


def _layout(root: Path, skip_w0_dota: bool = False) -> list[str]:
    """Every row's scores for the four sets, plus a read-out JSON that agrees with them."""
    lag, zs = root / "lag", root / "zs"
    sets: dict = {}
    dota_dirs = {W0: root / "w_u" / "W0", R4: root / "stp" / "R4"}
    for s, set_name in enumerate(zs_compare_table.SET_ORDER):
        rows: dict = {}
        for r, row in enumerate(zs_compare_table.ROWS):
            scores, labels = _clips(100 * s + r)
            if set_name == zs_compare_table.DOTA:
                base = lag / "eval" / "dota" / LAG / "B" if row == LAG else dota_dirs[row]
            elif set_name == zs_compare_table.DADA:
                base = lag / "eval" / "dada" / LAG if row == LAG else zs / "eval" / "dada" / row
            elif set_name == zs_compare_table.TAD:
                base = (lag / "eval" / "tad" / LAG / "released" if row == LAG
                        else zs / "eval" / "tad" / row / "s2024")
            else:
                base = lag / "eval" / "dad" / row / ("released" if row == LAG else "s2024")
            if set_name in (zs_compare_table.DOTA, zs_compare_table.DADA):
                if not (skip_w0_dota and set_name == zs_compare_table.DOTA and row == W0):
                    base.mkdir(parents=True)
                    protocol_b_eval.write_clip_scores(
                        base / protocol_b_eval.CLIP_SCORES_NPZ, scores, labels)
            else:
                _eval_run(base, scores, labels)
            rows[row] = {"macro": {"mean": 0.7, "low": 0.6, "high": 0.8},
                         "micro": _micro(scores, labels, set_name == zs_compare_table.DOTA)}
        sets[set_name] = {"rows": rows}
    (lag / "REPORTS").mkdir(parents=True)
    (lag / "REPORTS" / "zs_lagovad_readout.json").write_text(json.dumps({"sets": sets}))
    return ["--lagovad-root", str(lag), "--zs-root", str(zs), "--dota", W0, str(dota_dirs[W0]),
            "--dota", R4, str(dota_dirs[R4]), "--out", str(root / "table.md")]


def _args(argv: list[str]) -> argparse.Namespace:
    return zs_compare_table.build_arg_parser().parse_args(argv)


class TestCompareTable:
    def test_dota_pools_after_minmax_and_others_raw(self) -> None:
        scores, labels = _clips(1)
        scores = {v: s * (i + 1) for i, (v, s) in enumerate(scores.items())}
        ids = sorted(labels)
        truth = np.concatenate([labels[v] for v in ids])
        raw = np.concatenate([scores[v] for v in ids])
        mm = np.concatenate([normalize_scores(scores[v], constants.SCORE_NORM_MINMAX) for v in ids])
        assert zs_compare_table.pooled_reads(scores, labels, False)["ap"] == frame_ap(raw, truth)
        assert zs_compare_table.pooled_reads(scores, labels, True)["ap"] == frame_ap(mm, truth)
        assert zs_compare_table.pooled_reads(scores, labels, True)["base_rate"] == 0.5

    def test_table_has_every_cell(self, tmp_path: Path) -> None:
        table = zs_compare_table.run(_args(_layout(tmp_path)))
        for set_name in zs_compare_table.SET_ORDER:
            assert all(table[set_name][r]["ap"] is not None for r in zs_compare_table.ROWS)
        text = (tmp_path / "table.md").read_text()
        assert "| **DoTA-CAP-dev** | AUC macro |" in text and "| **DAD** | AUC macro |" in text

    def test_missing_scores_leave_only_that_cell_blank(self, tmp_path: Path) -> None:
        table = zs_compare_table.run(_args(_layout(tmp_path, skip_w0_dota=True)))
        assert table["dota"][W0]["ap"] is None
        assert table["dota"][W0]["micro"] is not None
        assert table["dota"][R4]["ap"] is not None
        ap_row = next(line for line in (tmp_path / "table.md").read_text().splitlines()
                      if line.startswith("| | AP"))
        assert zs_compare_table.BLANK in ap_row

    def test_scores_that_disagree_with_the_readout_refuse(self, tmp_path: Path) -> None:
        argv = _layout(tmp_path)
        path = tmp_path / "lag" / "REPORTS" / "zs_lagovad_readout.json"
        readout = json.loads(path.read_text())
        readout["sets"]["dad"]["rows"][R4]["micro"] += 0.01
        path.write_text(json.dumps(readout))
        with pytest.raises(ValueError, match="not the same scores"):
            zs_compare_table.run(_args(argv))
        assert not (tmp_path / "table.md").exists()


def _dad_curves(trend: float, seed: int, bump: float = 1.0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    curves, labels = {}, {}
    t = np.arange(ROWS_PER_VIDEO, dtype=np.float64)
    for prefix, videos in (("pos", 12), ("neg", 20)):
        for i in range(videos):
            g = np.zeros(ROWS_PER_VIDEO, dtype=np.int64)
            if prefix == "pos":
                g[ACCIDENT] = 1
            curves[f"{prefix}_{i}"] = trend * t + bump * g + 0.1 * rng.random(ROWS_PER_VIDEO)
            labels[f"{prefix}_{i}"] = g
    return curves, labels


class TestPositionDiag:
    def test_first_full_window_row_matches_the_clamp(self) -> None:
        row = diag.first_full_window_row()
        step, stride = constants.DAD_VIDEOMAE_FRAME_STEP, constants.DAD_STRIDE
        assert causal_clip_indices(row * stride, step=step)[0] >= 0
        assert (row * stride) - step * (constants.VIDEOMAE_CLIP_FRAMES - 1) >= 0
        assert ((row - 1) * stride) - step * (constants.VIDEOMAE_CLIP_FRAMES - 1) < 0
        assert row == 7

    def test_varying_accident_rows_refuse(self) -> None:
        labels = {"a": np.array([0, 0, 1]), "b": np.array([0, 1, 0])}
        with pytest.raises(ValueError, match="fixed-toa"):
            diag.accident_rows(labels)

    def test_mann_whitney_matches_frame_auc(self) -> None:
        rng = np.random.default_rng(0)
        a, b = rng.random(30), rng.random(40)
        truth = np.r_[np.ones(30), np.zeros(40)]
        assert diag.mann_whitney(a, b) == pytest.approx(frame_auc(np.r_[a, b], truth))

    def test_a_pure_time_drift_shows_on_the_placebo(self) -> None:
        curves, labels = _dad_curves(trend=1.0, seed=0, bump=0.0)
        r = diag.diagnose({LAG: curves, W0: curves, R4: curves}, labels, RESAMPLES, 0)
        assert r["ruler_macro"] == 1.0
        assert r["rows"][W0]["all_rows"]["placebo"] > 0.9
        assert abs(r["rows"][W0]["all_rows"]["gap"]["mean"]) < 0.1
        assert abs(r["rows"][W0]["same_position"]["accident_rows"]["mean"] - 0.5) < 0.2

    def test_a_real_reaction_shows_without_a_placebo(self) -> None:
        curves, labels = _dad_curves(trend=0.0, seed=1, bump=1.0)
        flat, _ = _dad_curves(trend=0.0, seed=2, bump=0.0)
        r = diag.diagnose({LAG: flat, W0: curves, R4: curves}, labels, RESAMPLES, 0)
        assert r["rows"][W0]["all_rows"]["positive"] == 1.0
        assert abs(r["rows"][W0]["all_rows"]["placebo"] - 0.5) < 0.2
        assert r["rows"][W0]["same_position"]["accident_rows"]["mean"] == 1.0
        assert r["same_position_contrasts"][f"{W0} - {LAG}"]["low"] > 0
        assert r["same_position_contrasts"][f"{R4} - {W0}"]["mean"] == 0.0
        assert "_boot" not in r["rows"][W0]["same_position"]
        assert "No-pixel `t/N` ruler macro = 1.0000" in diag.render(r)
