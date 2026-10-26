"""Tests for addendum §30: the source-video CRN reference and the CRN cost diagnostic.

The failures that matter are silent: a "source" path whose rows are not the item's rows (so the
read changes more than the reference), a whole item that differs between the two paths, a gate
that lets a read of other checkpoints through, and a sentence that skips C6's order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.crn.reference import reference
from core.data.v2_inputs import fit_stats
from core.tools import crn_cost_diag, nexar_eval

SEED = 7
STRIDE = constants.NEXAR_EVAL_STRIDE


def _x(rows: int = 301, dim: int = 16) -> np.ndarray:
    return np.random.default_rng(SEED).normal(size=(rows, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# nexar_eval --crn-reference source (C3)
# ---------------------------------------------------------------------------
def test_source_rows_equal_item_rows_on_a_whole_item() -> None:
    x1 = _x()
    stats = fit_stats({"a": x1[::8]}, None, "R2", constants.V2_OFF)
    np.testing.assert_array_equal(
        nexar_eval.source_item_rows(x1, None, stats, 0, len(x1)),
        nexar_eval.item_rows(x1, None, stats, 0, len(x1)),
    )


@pytest.mark.parametrize("start", [30, 31, 32])
def test_source_rows_keep_the_crop_rows_and_change_only_the_reference(start: int) -> None:
    x1 = _x()
    stats = fit_stats({"a": x1[::8]}, None, "R2", constants.V2_OFF)
    end = start + 120
    rows = nexar_eval.source_item_rows(x1, None, stats, start, end)
    crop = x1[start:end:STRIDE].astype(np.float64)
    assert len(rows) == len(crop)
    source_ref = reference(x1[start % STRIDE :: STRIDE], "R2")[0]
    np.testing.assert_allclose(rows, stats.s * (crop - source_ref), rtol=1e-5, atol=1e-5)
    item = nexar_eval.item_rows(x1, None, stats, start, end)
    assert not np.allclose(rows, item)  # the crop's own median differs from the source's


def test_source_rows_centre_the_motion_stream_by_the_source_too() -> None:
    x1, u1 = _x(), _x(dim=8) + 3.0
    stats = fit_stats({"a": x1[::8]}, {"a": u1[::8]}, "R2", "enc")
    rows = nexar_eval.source_item_rows(x1, u1, stats, 31, 151)
    u_crop = u1[31:151:STRIDE].astype(np.float64)
    u_ref = reference(u1[1::STRIDE], "R2")[0]
    assert stats.sigma_u is not None
    np.testing.assert_allclose(rows[:, x1.shape[1]:], stats.c * (u_crop - u_ref) / stats.sigma_u,
                               rtol=1e-5, atol=1e-5)


def test_source_rows_refuse_an_arm_without_crn() -> None:
    x1, u1 = _x(), _x(dim=8)
    stats = fit_stats({"a": x1[::8]}, {"a": u1[::8]}, constants.V2_OFF, "enc")
    with pytest.raises(ValueError, match="CRN arm"):
        nexar_eval.source_item_rows(x1, u1, stats, 0, 90)


def test_cli_defaults_to_the_item_reference() -> None:
    args = nexar_eval.build_arg_parser().parse_args([
        "--run", "s", "c.pt", "--census", "c.json", "--clip-s1-dir", ".", "--data-dir", ".",
        "--out-dir", "o",
    ])
    assert args.crn_reference == constants.CRN_REFERENCE_ITEM


# ---------------------------------------------------------------------------
# share (C7)
# ---------------------------------------------------------------------------
def test_spearman_handles_ties_and_constants() -> None:
    np.testing.assert_array_equal(crn_cost_diag.average_ranks(np.array([3.0, 1.0, 3.0])),
                                  [2.5, 1.0, 2.5])
    assert crn_cost_diag.spearman(np.arange(5.0), np.arange(5.0) ** 3) == pytest.approx(1.0)
    assert crn_cost_diag.spearman(np.ones(4), np.arange(4.0)) is None


def test_cost_table_bins_by_share() -> None:
    share = np.linspace(0.05, 0.95, 8)
    a = {f"c{i}": (0.5 - s, float(s)) for i, s in enumerate(share)}
    b = {f"c{i}": (0.5, float(s)) for i, s in enumerate(share)}
    t = crn_cost_diag.cost_table(a, b)
    assert t["clips"] == 8 and t["spearman_share_delta"] == pytest.approx(-1.0)
    means = [x["mean_delta"] for x in t["delta_by_share_bin"]]
    assert [x["clips"] for x in t["delta_by_share_bin"]] == [2, 2, 2, 2]
    assert means == sorted(means, reverse=True)


def test_read_npz_keeps_two_class_clips_only(tmp_path: Path) -> None:
    path = tmp_path / crn_cost_diag.CLIP_SCORES
    np.savez(path, **{"score/a": np.array([0.1, 0.9, 0.2, 0.8]),
                      "label/a": np.array([0, 1, 0, 1]),
                      "score/b": np.array([0.1, 0.2]), "label/b": np.array([1, 1])})
    assert crn_cost_diag.read_npz(path) == {"a": (1.0, 0.5)}


def test_nexar_shares_of_whole_videos_and_crops() -> None:
    row = {"label": 1, "duration_s": 40.0, "frames_header": 1200,
           "time_of_alert": 18.0, "time_of_event": 19.6}
    shares = crn_cost_diag.nexar_shares(["v", "v@0.5"], {"v": row})
    span_s = 19.6 + constants.NEXAR_POST_EVENT_S - 18.0
    assert shares["v"] == pytest.approx(span_s / 40.0, abs=2 / 1200)
    assert shares["v@0.5"] == pytest.approx(span_s / constants.NEXAR_CROP_S, abs=2 / 240)


# ---------------------------------------------------------------------------
# ref (C4-C6)
# ---------------------------------------------------------------------------
def _readout(endpoints: list[float], reference_unit: str,
             ckpt_root: str = "/ckpt") -> dict[str, Any]:
    seeds = [f"s{s}" for s in constants.NEXAR_RUN_SEEDS]
    crops = {str(p): {"macro": {"mean": float(np.mean(endpoints))}}
             for p in constants.NEXAR_CROP_PLACEMENTS}
    return {
        "split": constants.V2_SPLIT_NEXAR_VAL,
        "crn_reference": reference_unit,
        "global_steps": dict.fromkeys(seeds, 1749),
        "checkpoints": {s: f"{ckpt_root}/{s}.pt" for s in seeds},
        "per_run_crop_endpoint": dict(zip(seeds, endpoints, strict=True)),
        "crops": crops,
    }


AUCS = {"v1": 0.8, "v1@0.5": 0.7, "v2": None}


def _write(arm_dir: Path, readout: dict[str, Any], aucs: dict[str, float | None]) -> None:
    arm_dir.mkdir(parents=True)
    (arm_dir / nexar_eval.READOUT_JSON).write_text(json.dumps(readout))
    (arm_dir / nexar_eval.ITEM_AUCS_JSON).write_text(json.dumps(aucs))


def test_parity_gate_passes_identical_whole_items() -> None:
    item = (_readout([0.8] * 5, constants.CRN_REFERENCE_ITEM), AUCS)
    source = (_readout([0.82] * 5, constants.CRN_REFERENCE_SOURCE), {**AUCS, "v1@0.5": 0.75})
    assert crn_cost_diag.parity_gate(item, source) == []


def test_parity_gate_catches_each_failure() -> None:
    item = (_readout([0.8] * 5, constants.CRN_REFERENCE_ITEM), AUCS)
    wrong_ref = (_readout([0.8] * 5, constants.CRN_REFERENCE_ITEM), AUCS)
    moved = (_readout([0.8] * 5, constants.CRN_REFERENCE_SOURCE), {**AUCS, "v1": 0.79})
    other = (_readout([0.8] * 5, constants.CRN_REFERENCE_SOURCE, "/other"), AUCS)
    assert crn_cost_diag.parity_gate(item, wrong_ref)
    assert any("whole item v1" in f for f in crn_cost_diag.parity_gate(item, moved))
    assert any("checkpoints" in f for f in crn_cost_diag.parity_gate(item, other))


@pytest.mark.parametrize(("source", "sentence"), [
    ([0.86, 0.87, 0.86, 0.87, 0.86], "explains most"),
    ([0.81, 0.82, 0.81, 0.82, 0.81], "explains part"),
    ([0.75, 0.76, 0.75, 0.76, 0.75], "costs more"),
    ([0.79, 0.81, 0.80, 0.78, 0.82], "no change shown"),
])
def test_ref_sentence_follows_c6_in_order(source: list[float], sentence: str) -> None:
    item = _readout([0.80, 0.80, 0.80, 0.80, 0.80], constants.CRN_REFERENCE_ITEM)
    twin = _readout([0.88] * 5, constants.CRN_REFERENCE_ITEM)
    effect = crn_cost_diag.reference_effect(
        item, _readout(source, constants.CRN_REFERENCE_SOURCE), twin)
    assert effect["gap_twin_minus_item"] == pytest.approx(0.08)
    assert sentence in crn_cost_diag.ref_sentence(effect)


def _ref_args(tmp_path: Path) -> argparse.Namespace:
    return crn_cost_diag.build_arg_parser().parse_args([
        "ref", "--item-dir", str(tmp_path / "item"), "--source-dir", str(tmp_path / "src"),
        "--out-dir", str(tmp_path / "out"),
    ])


def test_run_ref_writes_the_sentence(tmp_path: Path) -> None:
    for arm in ("A0", "A1", "A2", "A3", "A3ign"):
        _write(tmp_path / "item" / arm, _readout([0.80] * 4 + [0.81], "item"), AUCS)
    for arm in ("A1", "A3", "A3ign"):
        _write(tmp_path / "src" / arm, _readout([0.80] * 4 + [0.81], "source"), AUCS)
    out = crn_cost_diag.run_ref(_ref_args(tmp_path))
    assert out["gate"] == "pass" and "does not explain" in out["sentence"]
    assert (tmp_path / "out" / crn_cost_diag.REF_MD).exists()


def test_run_ref_prints_nothing_when_the_gate_fails(tmp_path: Path) -> None:
    for arm in ("A0", "A1", "A2", "A3", "A3ign"):
        _write(tmp_path / "item" / arm, _readout([0.8] * 5, "item"), AUCS)
    for arm in ("A1", "A3", "A3ign"):
        _write(tmp_path / "src" / arm, _readout([0.8] * 5, "source"), {**AUCS, "v1": 0.5})
    with pytest.raises(SystemExit, match="C4 gate failed"):
        crn_cost_diag.run_ref(_ref_args(tmp_path))
    assert not (tmp_path / "out").exists()
