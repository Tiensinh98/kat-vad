"""Tests for ``core.tools.k5_fail_tail`` (``KIM_RUN.md`` §24). Synthetic, data-free."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import k5_fail_tail as ft
from core.tools import protocol_b_eval

N_CLIPS = 60
LENGTH = 40


def _clips(seed: int, failing: set[int]) -> tuple[dict, dict]:
    """Clips whose span is ranked correctly except the ``failing`` ones (span scored low)."""
    rng = np.random.default_rng(seed)
    scores, labels = {}, {}
    for i in range(N_CLIPS):
        gt = np.zeros(LENGTH, dtype=np.int64)
        gt[10:20] = 1
        s = rng.random(LENGTH) * 0.1 + (1.0 - gt if i in failing else gt)
        key = f"src{i // 2}_{i:06d}"
        scores[key], labels[key] = s, gt
    return scores, labels


def _write(run_dir: Path, scores: dict, labels: dict) -> None:
    run_dir.mkdir(parents=True)
    protocol_b_eval.write_clip_scores(run_dir / protocol_b_eval.CLIP_SCORES_NPZ, scores, labels)


class TestHelpers:
    def test_actor_takes_the_first_named(self) -> None:
        assert ft.dada_actor("[CLS]a pedestrian is hit by a vehicle[SEP]") == "pedestrian"
        assert ft.dada_actor("[CLS]a vehicle hits a cyclist[SEP]") == "vehicle"
        assert ft.dada_actor("[CLS]ego-car is out of control[SEP]") == ft.OTHER

    def test_small_strata_pool(self) -> None:
        values = {f"v{i}": "big" for i in range(30)} | {"a": "x", "b": "y"}
        pooled = ft.pool_small(values)
        assert pooled["v0"] == "big" and pooled["a"] == ft.SMALL == pooled["b"]

    def test_average_ranks_share_ties(self) -> None:
        assert ft.average_ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]

    def test_covariates_from_labels(self) -> None:
        gt = np.zeros(20, dtype=np.int64)
        gt[5:10] = 1
        c = ft.covariates({"v": gt}, rate=10.0)
        assert c["span share"]["v"] == 0.25 and c["span seconds"]["v"] == 0.5
        assert c["span centre"]["v"] == pytest.approx(7 / 20)


class TestReads:
    def test_a_concentrated_stratum_is_a_lead(self) -> None:
        fail = np.array([True] * 10 + [False] * 10 + [False] * 40)
        auc = np.where(fail, 0.3, 0.9)
        col = np.array(["night"] * 20 + ["day"] * 40)
        ids = [str(i) for i in range(60)]
        draws = ft.draws_of(ids, {v: v for v in ids})
        reads = ft.stratum_reads(fail, auc, col, draws)
        assert reads["night"]["lead"] and not reads["day"]["lead"]
        assert reads["night"]["fail_share"] == 0.5

    def test_a_separating_covariate_is_a_lead(self) -> None:
        fail = np.array([True] * 20 + [False] * 40)
        x = np.r_[np.full(20, 1.0), np.full(40, 3.0)] + np.linspace(0, 0.1, 60)
        ids = [str(i) for i in range(60)]
        r = ft.covariate_read(x, fail, ft.draws_of(ids, {v: v for v in ids}))
        assert r["lead"] and r["auc"] < 0.5

    def test_read_set_end_to_end(self, tmp_path: Path) -> None:
        failing = set(range(0, 20, 2))  # every failing clip is ego-uninvolved
        scores, labels = _clips(1, failing)
        _write(tmp_path / "k5", scores, labels)
        lag_scores, _ = _clips(2, set(range(30)))
        _write(tmp_path / "lag", lag_scores, labels)
        ann = tmp_path / "ann"
        ann.mkdir()
        for i, v in enumerate(sorted(scores, key=lambda k: int(k.rsplit("_", 1)[1]))):
            ann.joinpath(f"{v}.json").write_text(json.dumps(
                {"ego_involve": i not in set(range(20)), "night": False,
                 "accident_name": "turning"}))
        strata = ft.dota_strata(sorted(scores), ann)
        s = ft.read_set(ft.DOTA, tmp_path / "k5", tmp_path / "lag", strata, {})
        assert s["failing"] == len(failing) and s["clusters"] == N_CLIPS // 2
        assert s["strata"]["ego_involve"]["False"]["lead"]
        assert s["peak_of_failing"]["inside span"] == 0.0
        assert ft.leads({ft.DOTA: s}) == [f"{ft.DOTA} ego_involve = False"]

    def test_other_items_refused(self, tmp_path: Path) -> None:
        scores, labels = _clips(1, set())
        _write(tmp_path / "k5", scores, labels)
        drop = sorted(scores)[0]
        _write(tmp_path / "lag", {k: v for k, v in scores.items() if k != drop},
               {k: v for k, v in labels.items() if k != drop})
        with pytest.raises(ValueError, match="same items"):
            ft.read_set(ft.DOTA, tmp_path / "k5", tmp_path / "lag", {}, {})

    def test_missing_annotation_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ft.dota_strata(["nope_000001"], tmp_path)

    def test_constants_match_the_rule(self) -> None:
        assert constants.FAILTAIL_FAIL_AUC == 0.5 and constants.FAILTAIL_MIN_STRATUM == 20
        assert constants.FAILTAIL_LEAD_RATIO == 2.0
