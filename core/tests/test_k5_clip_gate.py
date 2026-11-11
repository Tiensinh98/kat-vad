"""Tests for ``core.tools.k5_clip_gate`` (``KIM_RUN.md`` §19)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.tools import k5_clip_gate as kg
from core.tools import nexar_eval
from core.tools.tadz_readout import load_run


def _toy() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Two abnormal videos that localize but score low; two normal videos that score high."""
    scores = {
        "a1": np.array([0.10, 0.12, 0.30, 0.11]),
        "a2": np.array([0.20, 0.40, 0.21, 0.22]),
        "n1": np.array([0.80, 0.85, 0.82, 0.81]),
        "n2": np.array([0.60, 0.70, 0.65, 0.66]),
    }
    labels = {
        "a1": np.array([0, 0, 1, 0]),
        "a2": np.array([0, 1, 0, 0]),
        "n1": np.zeros(4, dtype=np.int64),
        "n2": np.zeros(4, dtype=np.int64),
    }
    return scores, labels


class TestHeadroom:
    def test_oracle_gate_puts_every_normal_frame_below(self) -> None:
        scores, labels = _toy()
        h = kg.headroom(scores, labels)
        assert h["micro_raw"] < 0.5
        assert h["micro_oracle_gate"] > h["micro_raw"]

    def test_offsets_never_move_macro(self) -> None:
        scores, labels = _toy()
        shifted = kg.offset(scores, {v: float(i) for i, v in enumerate(sorted(scores))})
        assert kg.macro(shifted, labels) == pytest.approx(kg.macro(scores, labels), abs=1e-12)

    def test_offset_constant_exceeds_spread(self) -> None:
        scores, _ = _toy()
        out = kg.offset(scores, {"a1": 1.0, "a2": 1.0, "n1": 0.0, "n2": 0.0})
        assert min(out["a1"].min(), out["a2"].min()) > max(out["n1"].max(), out["n2"].max())


class TestGate:
    def test_fusion_keeps_macro_and_can_lift_micro(self) -> None:
        scores, labels = _toy()
        gate = {"a1": 0.9, "a2": 0.8, "n1": 0.1, "n2": 0.2}
        g = kg.gate_read(scores, labels, gate)
        assert g["macro_drift"] <= 1e-9
        assert g["rows"]["K5 x gate"]["micro"] > g["rows"]["K5 raw"]["micro"]

    def test_missing_gate_raises(self) -> None:
        scores, labels = _toy()
        with pytest.raises(ValueError, match="no gate score"):
            kg.gate_read(scores, labels, {"a1": 0.5})

    def test_gate_rows_must_match_curve(self, tmp_path: Path) -> None:
        scores, _ = _toy()
        path = tmp_path / "c.json"
        path.write_text(json.dumps({v: {"c": 0.5, "rows": 3.0} for v in scores}))
        with pytest.raises(ValueError, match="gate rows"):
            kg.read_gate_file(path, scores)


class TestParsing:
    def test_parse_set_reads_named_gates(self) -> None:
        name, run_dir, gates = kg.parse_set(["TAD", "run", "NG=a.json", "MS=b.json"])
        assert (name, run_dir) == ("TAD", Path("run"))
        assert gates == {"NG": Path("a.json"), "MS": Path("b.json")}

    def test_parse_set_refuses_a_bare_path(self) -> None:
        with pytest.raises(ValueError, match="GATE=JSON"):
            kg.parse_set(["TAD", "run", "a.json"])


class TestStrata:
    def test_one_class_stratum_has_no_auc(self) -> None:
        scores, labels = _toy()
        gate = {"a1": 0.9, "a2": 0.8, "n1": 0.1, "n2": 0.2}
        strata = {"a1": {"scene": "x"}, "n1": {"scene": "x"}, "a2": {"scene": "y"},
                  "n2": {"scene": "z"}}
        out = kg.strata_read(scores, labels, {"NG": gate}, strata)
        assert out["scene"]["x"]["NG"] == pytest.approx(1.0)
        assert "NG" not in out["scene"]["y"]


def _write_run(root: Path) -> Path:
    scores, labels = _toy()
    run_dir = root / "run"
    (run_dir / "scores").mkdir(parents=True)
    for v in scores:
        np.savez(run_dir / "scores" / f"{v}.npz", score=scores[v], gt=labels[v])
    return run_dir


def _write_gate(path: Path, good: float) -> Path:
    scores, _ = _toy()
    path.write_text(json.dumps({v: {"c": good if v.startswith("a") else 1 - good, "rows": 4.0}
                                for v in scores}))
    return path


def test_cli_writes_report(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    ng = _write_gate(tmp_path / "ng.json", 0.9)
    ms = _write_gate(tmp_path / "ms.json", 0.7)
    strata = tmp_path / "strata.json"
    strata.write_text(json.dumps({"a1": {"scene": "x"}, "n1": {"scene": "x"}}))
    kg.main(["--set", "TOY", str(run_dir), f"NG={ng}", f"MS={ms}",
             "--set", "TOY2", str(run_dir), f"MS={ms}",
             "--in-domain", "MS:TOY2", "--strata", "TOY", str(strata),
             "--out-dir", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out" / "k5_clip_gate.json").read_text())
    toy = report["sets"]["TOY"]
    assert set(toy["gates"]) == {"NG", "MS"} and "MS - NG" in toy["cross_gate"]
    assert toy["gates"]["NG"]["sentence"] in ("above", "below", "not separated")
    assert report["sets"]["TOY2"]["in_domain_gates"] == ["MS"]
    md = (tmp_path / "out" / "k5_clip_gate.md").read_text()
    assert "in-domain for this gate" in md and "within strata" in md


def test_nexar_eval_curves_read_back(tmp_path: Path) -> None:
    scores, labels = _toy()
    n = nexar_eval.save_whole_scores(tmp_path, scores, labels, sorted(scores))
    got_s, got_g = load_run(tmp_path)
    assert n == len(scores) and set(got_s) == set(scores)
    np.testing.assert_allclose(got_s["a1"], scores["a1"], rtol=1e-6)
    np.testing.assert_array_equal(got_g["a2"], labels["a2"])
