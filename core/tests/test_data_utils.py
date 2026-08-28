"""Phase 3 tests: collate/padding utilities and the definitions verbalizer."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from core.data import (
    DATASET_CLS_DEFS,
    SPECIAL_ABNORMAL_CLS,
    DatasetSpecVerbalizer,
    collate_variable_length,
    pad_and_stack,
    padding_mask,
    resample_or_pad_feature_length,
    truncate_or_pad_feature_length,
)


class TestPaddingUtils:
    def test_pad_and_stack(self) -> None:
        feats = [torch.ones(5, 3), torch.ones(2, 3) * 2]
        padded, lengths = pad_and_stack(feats)
        assert padded.shape == (2, 5, 3)
        assert lengths.tolist() == [5, 2]
        assert (padded[1, 2:] == 0).all()

    def test_padding_mask(self) -> None:
        mask = padding_mask(torch.tensor([3, 1]), 4)
        assert mask.tolist() == [[1, 1, 1, 0], [1, 0, 0, 0]]

    def test_resample_uniform_downsamples(self) -> None:
        feature = torch.arange(10).unsqueeze(1).float()
        out, length = resample_or_pad_feature_length(feature, 5)
        assert out.shape == (5, 1) and length == 5
        assert out[0, 0] == 0 and out[-1, 0] == 9

    def test_resample_pads_short_input(self) -> None:
        feature = torch.ones(3, 4)
        out, length = resample_or_pad_feature_length(feature, 6)
        assert out.shape == (6, 4) and length == 3
        assert (out[3:] == 0).all()

    def test_numpy_2d_padding_only_pads_time_axis(self) -> None:
        feature = np.ones((3, 4), dtype=np.float32)
        out, length = resample_or_pad_feature_length(feature, 6)
        assert out.shape == (6, 4) and length == 3  # baseline np.pad bug fixed
        assert (out[3:] == 0).all() and (out[:3] == 1).all()

    def test_truncate_or_pad(self) -> None:
        feature = torch.ones(8, 2)
        out, length = truncate_or_pad_feature_length(feature, 4)
        assert out.shape == (4, 2) and length == 4
        out, length = truncate_or_pad_feature_length(feature, 10)
        assert out.shape == (10, 2) and length == 8

    def test_resample_random_mode_and_unknown_mode(self) -> None:
        feature = torch.arange(10).unsqueeze(1).float()
        out, length = resample_or_pad_feature_length(feature, 4, resample_mode="random")
        assert out.shape == (4, 1) and length == 4
        with pytest.raises(ValueError, match="resample_mode"):
            resample_or_pad_feature_length(feature, 4, resample_mode="nope")


class TestCollate:
    def test_collate_variable_length(self) -> None:
        samples = [
            {
                "v_feat": torch.ones(5, 3),
                "e_o": torch.ones(5, 2),
                "pseudo_frame_label": torch.zeros(5),
                "cls_label_idx": 1,
                "video_id": "a",
            },
            {
                "v_feat": torch.ones(3, 3),
                "e_o": torch.ones(3, 2),
                "pseudo_frame_label": torch.ones(3),
                "cls_label_idx": 0,
                "video_id": "b",
            },
        ]
        batch = collate_variable_length(samples)
        assert batch["v_feat"].shape == (2, 5, 3)
        assert batch["e_o"].shape == (2, 5, 2)
        assert batch["pseudo_frame_label"].shape == (2, 5)
        assert batch["v_feat_l"].tolist() == [5, 3]
        assert batch["mask"].shape == (2, 5)
        assert batch["cls_label_idx"].tolist() == [1, 0]
        assert batch["video_id"] == ["a", "b"]


class TestVerbalizer:
    def test_returns_string_from_known_definitions(self) -> None:
        verbalizer = DatasetSpecVerbalizer("tad", rng=random.Random(0))
        text = verbalizer("Car Accident")
        assert text in DATASET_CLS_DEFS["tad"]["Car Accident"]

    def test_list_input_maps_each_class(self) -> None:
        verbalizer = DatasetSpecVerbalizer("dota", rng=random.Random(0))
        texts = verbalizer(["Normal", "CarAccident"])
        assert isinstance(texts, list) and len(texts) == 2

    def test_special_abnormal_expands_all_classes(self) -> None:
        verbalizer = DatasetSpecVerbalizer("prevad", rng=random.Random(0))
        texts = verbalizer(["Normal", SPECIAL_ABNORMAL_CLS])
        assert isinstance(texts, list)
        assert len(texts) == len(DATASET_CLS_DEFS["prevad"])

    def test_set_dataset_switches_definitions(self) -> None:
        verbalizer = DatasetSpecVerbalizer("prevad", rng=random.Random(0))
        verbalizer.set_dataset("tad")
        assert verbalizer.cls2text is DATASET_CLS_DEFS["tad"]
        with pytest.raises(KeyError, match="No definitions"):
            verbalizer.set_dataset("unknown")

    def test_seeded_rng_is_deterministic(self) -> None:
        # "Car Accident" (spaced) is the PreVAD v6 class name shipped in the
        # release CSVs; the CamelCase "CarAccident" belongs to dota/dada.
        first = DatasetSpecVerbalizer("prevad", rng=random.Random(7))("Car Accident")
        second = DatasetSpecVerbalizer("prevad", rng=random.Random(7))("Car Accident")
        assert first == second

    def test_dada_shares_dota_definitions(self) -> None:
        assert DATASET_CLS_DEFS["dada"] is DATASET_CLS_DEFS["dota"]
