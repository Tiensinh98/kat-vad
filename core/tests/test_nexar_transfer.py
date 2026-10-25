"""Tests for Amendment 18 (§29): the per-seed crop endpoint, DADA whole-video reads, the read-out.

The failures that matter are silent: a crop endpoint that averages videos missing a placement,
a DADA label that is one row off its cache, DoTA's two splits pooled by an unweighted mean, a
primary interval read on seed-averaged scores instead of per seed, and a sentence that fires
on an interval containing 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.tools import dada_whole_eval, nexar_eval, nexar_transfer_readout, protocol_b_eval
from core.tools.e3_readout import REPORTS

SEED = 5
SEEDS = list(constants.NEXAR_RUN_SEEDS)
P = constants.NEXAR_CROP_PLACEMENTS


def _ci(m: float) -> dict[str, float]:
    return {"mean": m, "low": m - 0.01, "high": m + 0.01}


# ---------------------------------------------------------------------------
# nexar_eval: crop endpoint
# ---------------------------------------------------------------------------
def test_crop_video_means_keeps_only_videos_with_every_placement() -> None:
    aucs = {f"a{nexar_eval.CROP_SEP}{p}": 0.8 for p in P}
    aucs |= {f"b{nexar_eval.CROP_SEP}{p}": 0.6 for p in P[:-1]}  # b misses one placement
    aucs["a"] = 0.1  # the whole video is not a crop
    assert nexar_eval.crop_video_means(aucs) == {"a": pytest.approx(0.8)}


def test_crop_endpoint_is_the_mean_over_videos_of_their_crop_mean() -> None:
    labels, scores, items = {}, {}, {}
    for v, good in (("a", True), ("b", False)):
        for p in P:
            name = f"{v}{nexar_eval.CROP_SEP}{p}"
            g = np.array([0, 0, 1, 1, 0])
            labels[name], items[name] = g, (v, 0, 5)
            scores[name] = g.astype(float) if good else 1.0 - g
    assert nexar_eval.crop_endpoint(scores, labels, items) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# dada_whole_eval
# ---------------------------------------------------------------------------
def test_test_sources_reads_only_test_windows(tmp_path: Path) -> None:
    meta = {"w1": {"split": "test", "source": "s2"}, "w2": {"split": "test", "source": "s1"},
            "w3": {"split": "train", "source": "s9"}, "w4": {"split": "test", "source": "s1"}}
    (tmp_path / constants.META_FILENAME).write_text(json.dumps(meta))
    assert dada_whole_eval.test_sources(tmp_path) == ["s1", "s2"]
    (tmp_path / constants.META_FILENAME).write_text(json.dumps({"w": {"split": "train",
                                                                       "source": "s"}}))
    with pytest.raises(ValueError, match="no test window"):
        dada_whole_eval.test_sources(tmp_path)


def test_check_rows_refuses_a_missing_source_and_an_off_by_one() -> None:
    labels = {"a": np.zeros(10, dtype=np.int64), "b": np.zeros(7, dtype=np.int64)}
    dada_whole_eval.check_rows(labels, {"a": 10, "b": 7})
    with pytest.raises(FileNotFoundError):
        dada_whole_eval.check_rows(labels, {"a": 10})
    with pytest.raises(ValueError, match="label length"):
        dada_whole_eval.check_rows(labels, {"a": 10, "b": 8})


def test_dada_summarize_seed_averages_and_reads_per_run() -> None:
    g = np.array([0, 0, 1, 1, 1, 0, 0, 0])
    labels = {"a": g, "b": np.roll(g, 2)}
    perfect = {v: x.astype(float) for v, x in labels.items()}
    inverted = {v: 1.0 - x for v, x in labels.items()}
    out = dada_whole_eval.summarize({"s1": perfect, "s2": perfect, "s3": inverted}, labels, SEED)
    assert out["per_run_macro"] == {"s1": 1.0, "s2": 1.0, "s3": 0.0}
    assert out["macro"]["mean"] == 1.0  # (1 + 1 + 0) / 3 per frame still ranks perfectly
    assert out["video_auc"] is None  # every DADA source is abnormal
    assert out["two_class"] == 2


# ---------------------------------------------------------------------------
# nexar_transfer_readout
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _nexar(root: Path, src: str, arm: str, per_seed: dict[str, float], crop: float) -> None:
    d = root / nexar_transfer_readout.EVAL_DIR / nexar_transfer_readout.NEXAR_TEST / src / arm
    crops: dict[str, Any] = {str(p): {"macro": _ci(crop), "middle_ruler_macro": _ci(0.5)}
                             for p in P}
    crops["mean_over_placements"] = {"videos": 3, "macro": _ci(crop)}
    _write_json(d / nexar_eval.READOUT_JSON, {
        "split": constants.V2_SPLIT_NEXAR_TEST, "per_run_crop_endpoint": per_seed,
        "crops": crops,
        "whole": {"macro": _ci(0.7), "micro": 0.7, "clip_auc_max": 0.6, "clip_oracle_micro": 0.75},
    })
    _write_json(d / nexar_eval.ITEM_AUCS_JSON,
                {f"v{i}{nexar_eval.CROP_SEP}{p}": crop + 0.01 * i for i in range(3) for p in P})


def _per_seed(base: float, step: float = 0.002) -> dict[str, float]:
    return {f"s{s}": base + step * i for i, s in enumerate(SEEDS)}


def _args(root: Path) -> argparse.Namespace:
    return nexar_transfer_readout.build_arg_parser().parse_args(["--root", str(root)])


def test_primary_reads_the_per_seed_endpoint_and_writes_the_sentence(tmp_path: Path) -> None:
    _nexar(tmp_path, "nexar", "A3ign", _per_seed(0.82), 0.82)
    _nexar(tmp_path, "nexar", "A0", {f"s{s}": 0.68 + 0.001 * (i % 2)
                                     for i, s in enumerate(SEEDS)}, 0.68)
    _nexar(tmp_path, "t2", "A3ign", _per_seed(0.69), 0.69)
    r = nexar_transfer_readout.run(_args(tmp_path))
    assert r["primary"]["sentence"] == "A3-ign helps on Nexar"
    assert r["primary"]["paired_t95"]["mean"] == pytest.approx(0.82 + 0.004 - 0.6804, abs=1e-6)
    assert r["reads"][nexar_transfer_readout.TAD]["nexar"]["A0"] is None  # not run is not fatal
    assert "A3ign" in r["nexar_minus_t2_on_nexar_test"]
    md = (tmp_path / REPORTS / nexar_transfer_readout.READOUT_MD).read_text()
    assert "not run" in md and "A3-ign helps on Nexar" in md


def test_primary_refuses_missing_arms_or_seeds(tmp_path: Path) -> None:
    _nexar(tmp_path, "nexar", "A3ign", _per_seed(0.8), 0.8)
    with pytest.raises(FileNotFoundError, match="primary"):
        nexar_transfer_readout.run(_args(tmp_path))
    _nexar(tmp_path, "nexar", "A0", {"s2024": 0.7, "s2025": 0.7}, 0.7)
    with pytest.raises(ValueError, match="runs"):
        nexar_transfer_readout.run(_args(tmp_path))


def test_sentence_needs_the_whole_interval_on_one_side() -> None:
    assert nexar_transfer_readout.primary_sentence(_ci(0.02)) == "A3-ign helps on Nexar"
    assert nexar_transfer_readout.primary_sentence(_ci(-0.02)) == "A3-ign hurts on Nexar"
    assert nexar_transfer_readout.primary_sentence(_ci(0.005)) == "no difference shown at n = 5"


def test_dota_splits_pool_by_two_class_count(tmp_path: Path) -> None:
    arm = tmp_path / "A0"
    for split, n, macro in ((constants.V2_SPLIT_DOTA_CAP_DEV, 3, 0.9),
                            (constants.V2_SPLIT_DOTA_CAP_EVAL, 1, 0.5)):
        ids = [f"{split}_{i}_000" for i in range(n)]
        _write_json(arm / split / protocol_b_eval.READOUT_JSON, {
            "split": split, "two_class": n, "macro": _ci(macro),
            "per_run_macro": {"s1": macro, "s2": macro}})
        _write_json(arm / split / protocol_b_eval.CLIP_AUCS_JSON, dict.fromkeys(ids, macro))
        g = np.array([0, 1, 1, 0])
        protocol_b_eval.write_clip_scores(arm / split / protocol_b_eval.CLIP_SCORES_NPZ,
                                          {v: g.astype(float) for v in ids},
                                          dict.fromkeys(ids, g))
    r = nexar_transfer_readout.read_dota(arm, SEED)
    assert r["per_seed"] == {"s1": pytest.approx(0.8), "s2": pytest.approx(0.8)}
    assert len(r["items"]) == 4 and r["micro"] == 1.0 and r["clip_auc"] is None
