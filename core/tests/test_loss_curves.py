"""core.tools.loss_curves: loading, smoothing, total check, summary and the CLI. Data-free, CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from core import constants
from core.tools import loss_curves as lc

STEPS = 60
NKC_FROM = 21
NKC_W = 0.1


def _write_run(root: Path, name: str, steps_per_epoch: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    run_dir = root / name
    run_dir.mkdir(parents=True)
    rows = []
    for step in range(1, STEPS + 1):
        row = {
            "epoch": (step - 1) // steps_per_epoch,
            "global_step": step,
            "lr": 1e-4,
            "mil": float(rng.uniform()),
            "mul_mil": float(rng.uniform()),
            "kim_mil": float(rng.uniform()),
        }
        if step >= NKC_FROM:
            row["kim_nkc"] = float(rng.uniform())
        row["total"] = (
            row["mil"] + row["mul_mil"] + row["kim_mil"] + NKC_W * row.get("kim_nkc", 0.0)
        )
        rows.append(row)
    (run_dir / constants.LOSS_CURVES_METRICS_FILENAME).write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    cfg = {
        "loss": {"mul_weight": 1.0},
        "kim": {"mil_weight": 1.0, "nkc_weight": NKC_W},
        "data": {"dataset": name},
    }
    (run_dir / constants.LOSS_CURVES_CONFIG_FILENAME).write_text(yaml.safe_dump(cfg))
    return run_dir


def test_rolling_mean_is_centred_and_unbiased_at_edges() -> None:
    x = np.arange(10, dtype=float)
    sm = lc.rolling_mean(x, 3)
    assert (
        sm[0] == pytest.approx(0.5) and sm[5] == pytest.approx(5.0) and sm[-1] == pytest.approx(8.5)
    )
    assert np.array_equal(lc.rolling_mean(x, 1), x)


def test_rolling_std_matches_numpy_in_the_interior() -> None:
    x = np.random.default_rng(0).normal(size=40)
    sd = lc.rolling_std(x, 5)
    assert sd[10] == pytest.approx(np.std(x[8:13])) and (sd >= 0).all()
    assert np.array_equal(lc.rolling_std(x, 1), np.zeros_like(x))


def test_total_residual_is_zero_when_every_term_is_logged(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "a", 10, 0)
    rows, cfg = lc.load_metrics(run_dir), lc.load_config(run_dir)
    assert lc.total_residual(rows, cfg) == pytest.approx(0.0, abs=1e-12)
    assert lc.term_weight(cfg, "kim_nkc") == NKC_W and lc.term_weight(cfg, "mil") == 1.0
    assert lc.total_residual(rows, {}) is None  # unknown weight: no claim


def test_summary_shape_and_late_key(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "a", 10, 0)
    s = lc.summarize(lc.load_metrics(run_dir), lc.load_config(run_dir), ("mil", "kim_nkc"), 5, 10)
    assert (s["steps"], s["epochs"], s["steps_per_epoch"], s["dataset"]) == (STEPS, 6, 10, "a")
    assert s["keys"]["kim_nkc"]["first_step"] == NKC_FROM


def test_cli_writes_png_and_json(tmp_path: Path) -> None:
    a, b = _write_run(tmp_path, "a", 10, 0), _write_run(tmp_path, "b", 30, 1)
    out = tmp_path / "out"
    lc.main(
        [
            "--run",
            "A",
            str(a),
            "--run",
            "B",
            str(b),
            "--name",
            "t",
            "--out-dir",
            str(out),
            "--keys",
            "total",
            "mil",
            "kim_nkc",
            "lr",
        ]
    )
    assert (out / "loss_curves_t.png").stat().st_size > 0
    payload = json.loads((out / "loss_curves_t.json").read_text())
    assert set(payload["runs"]) == {"A", "B"} and payload["runs"]["B"]["steps_per_epoch"] == 30


def test_refusals(tmp_path: Path) -> None:
    a = _write_run(tmp_path, "a", 10, 0)
    with pytest.raises(lc.LossCurvesError, match="no run logged"):
        lc.run([("A", a)], ("dvs_sup",), "t", tmp_path / "o", 5, 10, "x")
    with pytest.raises(lc.LossCurvesError, match="unique"):
        lc.run([("A", a), ("A", a)], ("mil",), "t", tmp_path / "o", 5, 10, "x")
    with pytest.raises(lc.LossCurvesError, match="no metrics"):
        lc.load_metrics(tmp_path / "missing")
