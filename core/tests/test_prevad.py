"""Tests for the PreVAD preprocessor (gaps G1-G3 of ``core/docs/PREVAD_SETUP.md``).

Synthetic CSVs and feature caches in the release's real layout -- no download.
The multi-span, span-overflow and reversed-span cases are modelled on quirks
actually present in the shipped ``test.csv`` (104 multi-span clips, 440 spans
ending past 1.0, one reversed span).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants
from core.data import prevad
from core.data.dataset import DVSFeatureDataset, FeatureEvalDataset
from core.data.dataset_files import TEST_IDS_FILENAME
from core.data.definitions import (
    DATASET_CLS_DEFS,
    DatasetSpecVerbalizer,
    verbalize_class_name,
)
from core.train import class_index_tensor

CSV_HEADER = "video_id,path,video_path,class_name,superclass_name,descriptions,anomaly_span\n"


def _row(
    video_id: str,
    class_name: str,
    superclass: str = "Violence",
    description: str = "",
    spans: str = "[]",
) -> str:
    """One release CSV row; ``descriptions`` and ``anomaly_span`` are quoted."""
    return (
        f"{video_id},ViT-B-16-8p-features/{video_id}.npy,"
        f'NormalVideos/{video_id}.mp4,{class_name},{superclass},"{description}","{spans}"\n'
    )


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text(CSV_HEADER + "".join(rows), encoding="utf-8")
    return path


def _write_features(clip_dir: Path, video_id: str, length: int, dim: int | None = None) -> Path:
    clip_dir.mkdir(parents=True, exist_ok=True)
    width = constants.CLIP_FEATURE_DIM if dim is None else dim
    rng = np.random.default_rng(abs(hash(video_id)) % 2**31)
    # released features are unnormalized encode_image output (L2 norm ~8-12)
    array = (rng.standard_normal((length, width)) * 0.45).astype(np.float32)
    path = clip_dir / f"{video_id}.npy"
    np.save(path, array)
    return path


@pytest.fixture()
def release(tmp_path: Path) -> dict[str, Path]:
    """A miniature PreVAD release: 4 train rows, 4 test rows, features on disk."""
    clip_dir = tmp_path / "clip"
    train_rows = [
        _row("tr_norm_a", "Normal", "Normal"),
        _row("tr_norm_b", "Normal", "Normal"),
        _row("tr_fire", "Fire", "Fire-related Accident", "A building burns.", "[[0.25, 0.75]]"),
        _row("tr_mug", "Mugging", "Robbery", "A bag is snatched.", "[[0.1, 0.4]]"),
    ]
    test_rows = [
        _row("te_norm", "Normal", "Normal"),
        _row("te_single", "Car Accident", "Vehicle Accident", "Two cars hit.", "[[0.25, 0.5]]"),
        # two disjoint windows -- the gap between them must stay normal (G2)
        _row("te_multi", "War", "Violence", "Shelling.", "[[0.0, 0.25], [0.5, 0.75]]"),
        # ends past the clip, as 440 shipped rows do
        _row("te_over", "Explosion", "Fire-related Accident", "A blast.", "[[0.5, 1.2]]"),
    ]
    for video_id in ("tr_norm_a", "tr_norm_b", "tr_fire", "tr_mug"):
        _write_features(clip_dir, video_id, 20)
    for video_id in ("te_norm", "te_single", "te_multi", "te_over"):
        _write_features(clip_dir, video_id, 20)
    return {
        "train_csv": _write_csv(tmp_path / "train.csv", train_rows),
        "test_csv": _write_csv(tmp_path / "test.csv", test_rows),
        "clip_dir": clip_dir,
        "out_dir": tmp_path / "data",
    }


def _run(release: dict[str, Path], **kwargs: object):
    return prevad.preprocess(
        train_csv=release["train_csv"],
        test_csv=release["test_csv"],
        clip_dir=release["clip_dir"],
        out_dir=release["out_dir"],
        **kwargs,  # type: ignore[arg-type]
    )


class TestSpanParsing:
    def test_empty_cell_is_no_span(self) -> None:
        assert prevad.parse_spans("", "v") == ()
        assert prevad.parse_spans("[]", "v") == ()

    def test_parses_multiple_spans_in_order(self) -> None:
        spans = prevad.parse_spans("[[0.1, 0.2], [0.5, 0.9]]", "v")
        assert spans == ((0.1, 0.2), (0.5, 0.9))

    def test_accepts_the_shipped_overflow_and_reversed_quirks(self) -> None:
        """The release contains both; refusing them makes the test split unusable."""
        over = prevad.parse_spans("[[0.82, 1.2103512147543891]]", "v")
        assert over == ((0.82, 1.2103512147543891),)
        assert prevad.parse_spans("[[0.9814, 0.7110]]", "v") == ((0.9814, 0.7110),)

    def test_rejects_malformed_json(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            prevad.parse_spans("[[0.1, 0.2]", "v")

    def test_rejects_wrong_arity(self) -> None:
        with pytest.raises(ValueError, match="not a \\[start, end\\] pair"):
            prevad.parse_spans("[[0.1, 0.2, 0.3]]", "v")

    def test_rejects_negative_start(self) -> None:
        with pytest.raises(ValueError, match="negative anomaly span start"):
            prevad.parse_spans("[[-0.1, 0.2]]", "v")


class TestFrameLabels:
    def _record(self, spans: tuple[tuple[float, float], ...]) -> prevad.PrevadRecord:
        return prevad.PrevadRecord("v", "War", "Violence", "", spans, prevad.SPLIT_TEST)

    def test_matches_baseline_rounded_half_open_fill(self) -> None:
        labels = prevad.sampled_frame_labels(self._record(((0.25, 0.5),)), 20)
        assert labels == [0] * 5 + [1] * 5 + [0] * 10

    def test_every_span_is_filled_and_the_gap_stays_normal(self) -> None:
        """G2: collapsing to the hull would mark frames 5-9 abnormal."""
        labels = prevad.sampled_frame_labels(self._record(((0.0, 0.25), (0.5, 0.75))), 20)
        assert labels == [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5

    def test_span_past_the_clip_is_clamped_not_dropped(self) -> None:
        labels = prevad.sampled_frame_labels(self._record(((0.5, 1.2),)), 20)
        assert labels == [0] * 10 + [1] * 10

    def test_reversed_span_contributes_nothing(self) -> None:
        labels = prevad.sampled_frame_labels(self._record(((0.9814, 0.7110),)), 20)
        assert sum(labels) == 0

    def test_label_length_equals_feature_length(self) -> None:
        for length in (1, 7, 20, 133):
            assert len(prevad.sampled_frame_labels(self._record(((0.1, 0.9),)), length)) == length

    def test_overlapping_spans_do_not_double_count(self) -> None:
        labels = prevad.sampled_frame_labels(self._record(((0.0, 0.5), (0.25, 0.75))), 20)
        assert labels == [1] * 15 + [0] * 5

    def test_vanished_window_is_warned_and_strict_refuses(self, caplog) -> None:
        records = [
            prevad.PrevadRecord("v", "War", "Violence", "", ((0.10, 0.11),), prevad.SPLIT_TEST)
        ]
        with caplog.at_level(logging.WARNING):
            labels = prevad.build_frame_labels(records, {"v": 4}, strict=False)
        assert sum(labels["v"]) == 0
        assert "no positive frame after rounding" in caplog.text
        with pytest.raises(ValueError, match="no positive frame after rounding"):
            prevad.build_frame_labels(records, {"v": 4}, strict=True)

    def test_normal_clip_is_all_zero(self) -> None:
        record = prevad.PrevadRecord("v", "Normal", "Normal", "", (), prevad.SPLIT_TEST)
        assert prevad.sampled_frame_labels(record, 12) == [0] * 12


class TestCsvReading:
    def test_rejects_a_missing_column(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("video_id,class_name\nv,Normal\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            prevad.read_split_csv(path, prevad.SPLIT_TRAIN)

    def test_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path / "d.csv", [_row("v", "Normal", "Normal")] * 2)
        with pytest.raises(ValueError, match="duplicate video ids"):
            prevad.read_split_csv(path, prevad.SPLIT_TRAIN)

    def test_rejects_normal_row_carrying_a_span(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path / "n.csv", [_row("v", "Normal", "Normal", "", "[[0.1, 0.2]]")])
        with pytest.raises(ValueError, match="Normal rows carry a span"):
            prevad.read_split_csv(path, prevad.SPLIT_TEST)

    def test_rejects_abnormal_row_without_a_span(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path / "a.csv", [_row("v", "Fire", "Fire-related Accident")])
        with pytest.raises(ValueError, match="abnormal rows carry no span"):
            prevad.read_split_csv(path, prevad.SPLIT_TEST)

    def test_reads_description_and_class_fields(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path / "ok.csv",
            [_row("v", "Mugging", "Robbery", "A bag is snatched.", "[[0.1, 0.4]]")],
        )
        (record,) = prevad.read_split_csv(path, prevad.SPLIT_TRAIN)
        assert record.class_name == "Mugging"
        assert record.superclass_name == "Robbery"
        assert record.description == "A bag is snatched."
        assert record.is_abnormal


class TestFeatureResolution:
    def test_length_comes_from_the_array_header(self, tmp_path: Path) -> None:
        path = _write_features(tmp_path, "v", 37)
        assert prevad.feature_length(path) == 37

    def test_rejects_the_vit_l_release(self, tmp_path: Path) -> None:
        path = _write_features(tmp_path, "v", 10, dim=768)
        with pytest.raises(ValueError, match="not the ViT-B/16 release"):
            prevad.feature_length(path)

    def test_rejects_an_empty_array(self, tmp_path: Path) -> None:
        path = _write_features(tmp_path, "v", 0)
        with pytest.raises(ValueError, match="empty"):
            prevad.feature_length(path)

    def test_missing_feature_is_fatal(self, release: dict[str, Path]) -> None:
        (release["clip_dir"] / "te_multi.npy").unlink()
        with pytest.raises(ValueError, match="no usable feature array"):
            _run(release)

    def test_allow_missing_drops_and_reports_coverage(
        self, release: dict[str, Path], caplog
    ) -> None:
        (release["clip_dir"] / "te_multi.npy").unlink()
        with caplog.at_level(logging.WARNING):
            _train, test = _run(release, allow_missing_features=True)
        assert [r.video_id for r in test] == ["te_norm", "te_over", "te_single"]
        assert "Coverage" in caplog.text

    def test_warns_when_features_look_prenormalized(self, tmp_path: Path, caplog) -> None:
        clip_dir = tmp_path / "clip"
        clip_dir.mkdir()
        array = np.ones((5, constants.CLIP_FEATURE_DIM), dtype=np.float32)
        array /= np.linalg.norm(array, axis=1, keepdims=True)
        np.save(clip_dir / "v.npy", array)
        records = [prevad.PrevadRecord("v", "Normal", "Normal", "", (), prevad.SPLIT_TEST)]
        with caplog.at_level(logging.WARNING):
            mean = prevad.check_feature_scale(records, clip_dir)
        assert mean == pytest.approx(1.0, abs=1e-5)
        assert "look pre-normalized" in caplog.text


class TestDefinitionCoverage:
    def test_the_v6_taxonomy_is_the_space_separated_one(self) -> None:
        """G3: the CamelCase list was the *commented-out* one in the baseline."""
        defs = DATASET_CLS_DEFS["prevad"]
        assert len(defs) == 36
        assert next(iter(defs)) == "Normal"
        spaced = ("Store Robbery", "Fume", "Stunt Fail", "Shooting Accident", "Fall to the Ground")
        for name in spaced:
            assert name in defs, name
        for name in ("StoreRobbery", "FallDown", "WarScene", "Accident"):
            assert name not in defs, name

    def test_every_class_verbalizes_through_the_lookup_not_the_fallback(self) -> None:
        """``verbalize_class_name`` returns the bare name for an unknown class.

        Some definition lists legitimately *contain* the bare class name as one
        sampled variant (``"Robbery"``, ``"Explosion"`` -- the baseline's own
        style), so the fallback cannot be detected from the returned string.
        Assert on the lookup itself instead.
        """
        verbalizer = DatasetSpecVerbalizer("prevad")
        for name, defs in DATASET_CLS_DEFS["prevad"].items():
            assert name in verbalizer.cls2text, name
            assert verbalize_class_name(verbalizer, name) in defs, name
            assert any(len(text.split()) > 3 for text in defs), name

    def test_undefined_class_is_fatal(self) -> None:
        records = [
            prevad.PrevadRecord("v", "Teleportation", "Violence", "", ((0.1, 0.2),), "train")
        ]
        with pytest.raises(ValueError, match="no definition sentences"):
            prevad.check_definition_coverage(records)


class TestPreprocess:
    def test_writes_the_standard_dataset_files(self, release: dict[str, Path]) -> None:
        _run(release)
        out = release["out_dir"]
        labels = json.loads((out / constants.LABELS_TRAIN_FILENAME).read_text())
        frame_labels = json.loads((out / constants.FRAME_LABELS_TEST_FILENAME).read_text())
        defs = json.loads((out / constants.DEFS_FILENAME).read_text())
        meta = json.loads((out / constants.META_FILENAME).read_text())

        assert labels == {"tr_norm_a": 0, "tr_norm_b": 0, "tr_fire": 1, "tr_mug": 1}
        assert set(frame_labels) == {"te_norm", "te_single", "te_multi", "te_over"}
        assert defs[0] == "Normal"
        assert defs == sorted(defs[1:]) or defs[0] == "Normal"
        assert set(defs) == {"Normal", "Fire", "Mugging", "Car Accident", "War", "Explosion"}
        assert set(meta) == set(labels) | set(frame_labels)
        assert (out / TEST_IDS_FILENAME).read_text().split() == sorted(frame_labels)

    def test_frame_labels_match_the_feature_lengths(self, release: dict[str, Path]) -> None:
        _run(release)
        frame_labels = json.loads(
            (release["out_dir"] / constants.FRAME_LABELS_TEST_FILENAME).read_text()
        )
        for video_id, labels in frame_labels.items():
            array = np.load(release["clip_dir"] / f"{video_id}.npy", mmap_mode="r")
            assert len(labels) == array.shape[0], video_id

    def test_multi_span_survives_to_disk(self, release: dict[str, Path]) -> None:
        _run(release)
        frame_labels = json.loads(
            (release["out_dir"] / constants.FRAME_LABELS_TEST_FILENAME).read_text()
        )
        assert frame_labels["te_multi"] == [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5

    def test_meta_carries_class_description_and_split(self, release: dict[str, Path]) -> None:
        _run(release)
        meta = json.loads((release["out_dir"] / constants.META_FILENAME).read_text())
        assert meta["tr_mug"]["class_name"] == "Mugging"
        assert meta["tr_mug"]["superclass_name"] == "Robbery"
        assert meta["tr_mug"]["description"] == "A bag is snatched."
        assert meta["tr_mug"]["split"] == "train"
        assert meta["te_multi"]["split"] == "test"
        assert meta["te_multi"]["positive_frames"] == 10
        assert "positive_frames" not in meta["tr_mug"]

    def test_defs_omits_classes_absent_from_the_data(self, release: dict[str, Path]) -> None:
        _run(release)
        defs = json.loads((release["out_dir"] / constants.DEFS_FILENAME).read_text())
        assert "Fire-related Accident" not in defs
        assert "Fire-related Accident" in DATASET_CLS_DEFS["prevad"]

    def test_dry_run_writes_nothing(self, release: dict[str, Path]) -> None:
        _run(release, dry_run=True)
        assert not release["out_dir"].exists()

    def test_id_in_both_splits_is_fatal(self, release: dict[str, Path], tmp_path: Path) -> None:
        clash = _row("tr_fire", "Fire", "Fire-related Accident", "x", "[[0.1, 0.2]]")
        _write_csv(release["test_csv"], [clash])
        with pytest.raises(ValueError, match="appear in both splits"):
            _run(release)


class TestConsumedByTheTrainingPipeline:
    """The output contract is only real if the loaders it feeds accept it."""

    def test_dvs_dataset_reads_labels_and_class_names(self, release: dict[str, Path]) -> None:
        _run(release)
        dataset = DVSFeatureDataset(
            data_dir=release["out_dir"],
            clip_dir=release["clip_dir"],
            require_flow=False,
        )
        assert len(dataset) == 4  # 2 x 2 abnormal train videos (A9 balance)
        classes = {dataset[i]["cls_label"] for i in range(len(dataset))}
        defs = json.loads((release["out_dir"] / constants.DEFS_FILENAME).read_text())
        assert classes <= set(defs)

    def test_class_names_resolve_to_indices(self, release: dict[str, Path]) -> None:
        """class_index_tensor raises on any class absent from defs.json."""
        _run(release)
        defs = json.loads((release["out_dir"] / constants.DEFS_FILENAME).read_text())
        meta = json.loads((release["out_dir"] / constants.META_FILENAME).read_text())
        names = [entry["class_name"] for entry in meta.values()]
        indices = class_index_tensor(names, defs, torch.device("cpu"))
        assert len(indices) == len(names)
        assert defs[int(indices[names.index("Normal")])] == "Normal"

    def test_eval_dataset_aligns_labels_with_features(
        self, release: dict[str, Path], caplog
    ) -> None:
        _run(release)
        with caplog.at_level(logging.WARNING):
            dataset = FeatureEvalDataset(release["out_dir"], release["clip_dir"])
            items = [dataset[i] for i in range(len(dataset))]
        assert "truncating" not in caplog.text  # lengths agree exactly
        for item in items:
            assert item["v_feat"].shape[0] == item["frame_label"].shape[0]
