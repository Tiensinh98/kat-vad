"""Tests for DVS synthesis primitives, the training/eval datasets and fixtures."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from core import constants
from core.config import DVSConfig
from core.data.collate import collate_variable_length
from core.data.dataset import DVSFeatureDataset, FeatureEvalDataset
from core.data.knn_cache import load_knn_cache
from core.data.synthesis import choose_filler_ids, compose_sequence, truncate_sample
from core.tests.fixtures import build_fixture

D, DO = constants.CLIP_FEATURE_DIM, constants.FLOW_DIM


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory):
    return build_fixture(tmp_path_factory.mktemp("msad_fix"), num_abnormal=6, num_normal=8)


def _dataset(fixture, **overrides) -> DVSFeatureDataset:
    kwargs = {
        "data_dir": fixture.data_dir,
        "clip_dir": fixture.clip_dir,
        "flow_dir": fixture.flow_dir,
        "knn_cache": load_knn_cache(fixture.knn_cache_path),
        "seed": 3,
    }
    kwargs.update(overrides)
    return DVSFeatureDataset(**kwargs)


class TestSynthesisPrimitives:
    def test_compose_alignment_and_span(self) -> None:
        anchor = torch.ones(5, D)
        fillers = [
            ("n0", torch.zeros(3, D), torch.zeros(3, DO)),
            ("n1", torch.zeros(4, D), torch.zeros(4, DO)),
        ]
        sample = compose_sequence("a0", anchor, torch.ones(5, DO), True, fillers, 1)
        assert sample.features.shape == (12, D)
        assert sample.flow is not None and sample.flow.shape == (12, DO)
        assert sample.pseudo_label.tolist() == [0] * 3 + [1] * 5 + [0] * 4
        assert sample.clip_ids == ["n0", "a0", "n1"]
        assert sample.is_synthesized

    def test_normal_anchor_has_zero_pseudo(self) -> None:
        sample = compose_sequence(
            "n9",
            torch.ones(4, D),
            None,
            False,
            [("n0", torch.zeros(2, D), None)],
            0,
        )
        assert not sample.pseudo_label.any()
        assert sample.flow is None

    def test_no_fillers_not_synthesized(self) -> None:
        sample = compose_sequence("a0", torch.ones(4, D), torch.ones(4, DO), True, [], 0)
        assert not sample.is_synthesized
        # window unknown without synthesis (weak supervision) -> zero y^p
        assert not sample.pseudo_label.any()

    def test_truncate_keeps_alignment(self) -> None:
        sample = compose_sequence(
            "a0",
            torch.ones(6, D),
            torch.ones(6, DO),
            True,
            [("n0", torch.zeros(4, D), torch.zeros(4, DO))],
            1,
        )
        cut = truncate_sample(sample, 7)
        assert len(cut.features) == len(cut.pseudo_label) == 7
        assert cut.flow is not None and len(cut.flow) == 7
        assert cut.pseudo_label.tolist() == [0, 0, 0, 0, 1, 1, 1]

    def test_choose_fillers_respects_knn_ratio(self) -> None:
        rng = random.Random(0)
        fillers = choose_filler_ids(rng, 200, ["n0", "n1"], ["knn0"], knn_ratio=1.0)
        assert set(fillers) == {"knn0"}
        fillers = choose_filler_ids(rng, 200, ["n0", "n1"], ["knn0"], knn_ratio=0.0)
        assert "knn0" not in fillers


class TestDVSFeatureDataset:
    def test_length_is_twice_anomaly_count(self, fixture) -> None:
        dataset = _dataset(fixture)
        train_abnormal = [v for v in fixture.train_ids if v.startswith("Traffic")]
        assert len(dataset) == 2 * len(train_abnormal)

    def test_item_fields_and_alignment(self, fixture) -> None:
        dataset = _dataset(fixture, dvs=DVSConfig(theta=0.0))  # always synthesize
        item = dataset[0]
        length = len(item["v_feat"])
        assert item["v_feat"].shape == (length, D)
        assert item["e_o"].shape == (length, DO)
        assert item["pseudo_frame_label"].shape == (length,)
        assert item["label"].item() == 1
        assert bool(item["is_synthesized"])
        assert item["pseudo_frame_label"].sum() > 0  # anchor span marked
        assert item["cls_label"] == "Traffic_accident"

    def test_theta_one_never_synthesizes(self, fixture) -> None:
        dataset = _dataset(fixture, dvs=DVSConfig(theta=1.0))
        for index in range(len(dataset)):
            item = dataset[index]
            assert not bool(item["is_synthesized"])
            if item["label"].item() == 1:
                # un-synthesized abnormal: window unknown -> zero pseudo label
                assert not item["pseudo_frame_label"].any()

    def test_normal_half_label_zero(self, fixture) -> None:
        dataset = _dataset(fixture)
        item = dataset[len(dataset) - 1]
        assert item["label"].item() == 0
        assert not item["pseudo_frame_label"].any()

    def test_truncation_to_vis_max_len(self, fixture) -> None:
        dataset = _dataset(fixture, dvs=DVSConfig(theta=0.0), vis_max_len=16)
        for index in range(len(dataset)):
            assert len(dataset[index]["v_feat"]) <= 16

    def test_require_flow_false_yields_zeros(self, fixture) -> None:
        dataset = _dataset(fixture, flow_dir=None, require_flow=False)
        item = dataset[0]
        assert item["e_o"].shape == (len(item["v_feat"]), DO)
        assert not item["e_o"].any()

    def test_missing_flow_fails_loudly(self, fixture, tmp_path) -> None:
        dataset = _dataset(fixture, flow_dir=tmp_path)  # empty dir
        with pytest.raises(FileNotFoundError, match="raft_extract"):
            dataset[0]

    def test_collate_integration(self, fixture) -> None:
        dataset = _dataset(fixture, dvs=DVSConfig(theta=0.5))
        batch = collate_variable_length([dataset[i] for i in range(4)])
        batch_size, max_len = batch["v_feat"].shape[:2]
        assert batch_size == 4
        assert batch["e_o"].shape == (4, max_len, DO)
        assert batch["mask"].shape == (4, max_len)
        assert batch["pseudo_frame_label"].shape == (4, max_len)
        assert batch["is_synthesized"].dtype == torch.bool or (
            batch["is_synthesized"].dtype == torch.int64
        )
        # padding rows are zero wherever the mask is zero
        assert torch.allclose(
            batch["v_feat"] * (1 - batch["mask"])[..., None], torch.zeros(1)
        )

    def test_deterministic_given_seed(self, fixture) -> None:
        first = _dataset(fixture, seed=11)
        second = _dataset(fixture, seed=11)
        item_a, item_b = first[1], second[1]
        assert torch.equal(item_a["v_feat"], item_b["v_feat"])
        assert item_a["video_id"] == item_b["video_id"]


class TestFeatureEvalDataset:
    def test_serves_full_length_with_frame_labels(self, fixture) -> None:
        dataset = FeatureEvalDataset(fixture.data_dir, fixture.clip_dir)
        assert len(dataset) == len(fixture.test_ids)
        item = dataset[0]
        assert item["v_feat"].shape[0] == item["frame_label"].shape[0]
        assert item["label"].item() in (0, 1)

    def test_video_level_label_consistent(self, fixture) -> None:
        dataset = FeatureEvalDataset(fixture.data_dir, fixture.clip_dir)
        for index in range(len(dataset)):
            item = dataset[index]
            assert item["label"].item() == int(item["frame_label"].max() > 0)


class TestFixture:
    def test_deterministic(self, tmp_path) -> None:
        first = build_fixture(tmp_path / "a", num_abnormal=3, num_normal=3)
        second = build_fixture(tmp_path / "b", num_abnormal=3, num_normal=3)
        assert first.train_ids == second.train_ids
        video_id = first.train_ids[0]
        assert np.allclose(
            np.load(first.clip_dir / f"{video_id}.npy"),
            np.load(second.clip_dir / f"{video_id}.npy"),
        )

    def test_flow_matches_projection(self, fixture) -> None:
        from core.flow.raft_extract import load_projection

        projection = load_projection(
            fixture.flow_cache_root / constants.FLOW_PROJECTION_FILENAME
        )
        video_id = fixture.train_ids[0]
        stats = np.load(fixture.flow_dir / f"{video_id}{constants.FLOW_STATS_SUFFIX}")
        flow = np.load(fixture.flow_dir / f"{video_id}.npy")
        assert np.allclose(flow, stats @ projection, atol=1e-5)
