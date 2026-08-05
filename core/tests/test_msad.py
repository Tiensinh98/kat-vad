"""Tests for the MSAD preprocessor (core/data/msad.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import constants
from core.data import msad

ANNOTATION = """\
name\tscenario\ttotal frames\tstarting frame of anomaly\tending frame of anomaly
Traffic_accident_001.mp4\thighway\t240\t80\t159
Traffic_accident_002.mp4\tparking lot\t320\t0\t319
Fighting_003.mp4\tstreet highview\t160\t40\t80
Normal_001.mp4\thighway\t200\t\t
Normal_002.mp4\tparking lot\t240\t-1\t-1
Normal_003.mp4\toffice\t180\t\t
"""


@pytest.fixture()
def annotation_file(tmp_path: Path) -> Path:
    path = tmp_path / "anno.tsv"
    path.write_text(ANNOTATION, encoding="utf-8")
    return path


class TestParsing:
    def test_parses_all_rows_and_header(self, annotation_file: Path) -> None:
        records = msad.parse_annotation_file(annotation_file)
        assert len(records) == 6
        by_id = {r.video_id: r for r in records}
        assert by_id["Traffic_accident_001"].anomaly_start == 80
        assert by_id["Traffic_accident_001"].anomaly_end == 159
        assert by_id["Traffic_accident_001"].scenario == "highway"

    def test_normal_detection(self, annotation_file: Path) -> None:
        records = msad.parse_annotation_file(annotation_file)
        by_id = {r.video_id: r for r in records}
        assert not by_id["Normal_001"].is_abnormal  # empty fields
        assert not by_id["Normal_002"].is_abnormal  # -1 markers
        assert by_id["Traffic_accident_002"].is_abnormal

    def test_comma_separated_variant(self, tmp_path: Path) -> None:
        path = tmp_path / "anno.csv"
        path.write_text(
            "Traffic_accident_001.mp4,highway,240,80,159\n", encoding="utf-8"
        )
        records = msad.parse_annotation_file(path)
        assert records[0].scenario == "highway"
        assert records[0].is_abnormal

    def test_one_indexed_and_end_exclusive(self, tmp_path: Path) -> None:
        path = tmp_path / "anno.tsv"
        path.write_text("v1.mp4\thighway\t100\t9\t20\n", encoding="utf-8")
        rec = msad.parse_annotation_file(path, one_indexed=True, end_exclusive=True)[0]
        assert rec.anomaly_start == 8
        assert rec.anomaly_end == 18  # (20-1 for 1-indexed) -1 for exclusive end

    def test_end_clamped_to_total(self, tmp_path: Path) -> None:
        path = tmp_path / "anno.tsv"
        path.write_text("v1.mp4\thighway\t100\t50\t500\n", encoding="utf-8")
        rec = msad.parse_annotation_file(path)[0]
        assert rec.anomaly_end == 99

    def test_single_space_rows_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "anno.txt"
        path.write_text("v1.mp4 highway 100 50 60\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Cannot split"):
            msad.parse_annotation_file(path)

    def test_class_inference(self) -> None:
        assert msad.infer_class_name("Traffic_accident_001", True) == "Traffic_accident"
        assert msad.infer_class_name("Fighting_003", True) == "Fighting"
        assert msad.infer_class_name("mystery_clip", True) == msad.ABNORMAL_CLASS
        assert msad.infer_class_name("Normal_001", False) == "Normal"


class TestInferAbnormalFromName:
    """Full-MSAD mode: abnormal train videos have no temporal window."""

    WINDOWLESS = "Fighting_004.mp4\tstreet highview\t160\t\t\n"

    def test_windowless_abnormal_row(self, tmp_path: Path) -> None:
        path = tmp_path / "anno.tsv"
        path.write_text(self.WINDOWLESS, encoding="utf-8")
        default = msad.parse_annotation_file(path)[0]
        assert not default.is_abnormal  # unchanged default: window-less = normal
        flagged = msad.parse_annotation_file(path, infer_abnormal_from_name=True)[0]
        assert flagged.is_abnormal
        assert flagged.class_name == "Fighting"
        assert flagged.anomaly_start is None and flagged.anomaly_end is None

    def test_windowless_normal_row_stays_normal(self, tmp_path: Path) -> None:
        path = tmp_path / "anno.tsv"
        path.write_text("road2.mp4\troad\t200\t\t\n", encoding="utf-8")
        rec = msad.parse_annotation_file(path, infer_abnormal_from_name=True)[0]
        assert not rec.is_abnormal
        assert rec.class_name == msad.NORMAL_CLASS

    def test_windowless_abnormal_in_test_split_refused(
        self, annotation_file: Path, tmp_path: Path
    ) -> None:
        anno = annotation_file.read_text(encoding="utf-8") + self.WINDOWLESS
        path = tmp_path / "anno_full.tsv"
        path.write_text(anno, encoding="utf-8")
        split = tmp_path / "test_ids.txt"
        split.write_text("Fighting_004\nNormal_001\n", encoding="utf-8")
        with pytest.raises(ValueError, match="without an anomaly window"):
            msad.preprocess(
                path,
                tmp_path / "out",
                split_file=split,
                infer_abnormal_from_name=True,
            )

    def test_windowless_abnormal_in_train_split_ok(
        self, annotation_file: Path, tmp_path: Path
    ) -> None:
        anno = annotation_file.read_text(encoding="utf-8") + self.WINDOWLESS
        path = tmp_path / "anno_full.tsv"
        path.write_text(anno, encoding="utf-8")
        split = tmp_path / "test_ids.txt"
        split.write_text("Traffic_accident_001\nNormal_001\n", encoding="utf-8")
        msad.preprocess(
            path, tmp_path / "out", split_file=split, infer_abnormal_from_name=True
        )
        out = tmp_path / "out"
        labels = json.loads((out / constants.LABELS_TRAIN_FILENAME).read_text())
        assert labels["Fighting_004"] == 1  # video-level weak label preserved
        meta = json.loads((out / constants.META_FILENAME).read_text())
        assert meta["Fighting_004"]["anomaly_start"] is None
        assert meta["Fighting_004"]["class_name"] == "Fighting"


class TestFrameLabels:
    def test_alignment_matches_stride_sampling(self) -> None:
        rec = msad.VideoRecord("v", "highway", 240, 80, 159, "Traffic_accident")
        labels = msad.sampled_frame_labels(rec, stride=8)
        assert len(labels) == msad.num_sampled_frames(240, 8) == 30
        # sampled index i covers raw frame 8*i: 80..159 -> indices 10..19
        assert labels[9] == 0 and labels[10] == 1 and labels[19] == 1 and labels[20] == 0

    def test_normal_video_all_zeros(self) -> None:
        rec = msad.VideoRecord("v", "highway", 100, None, None, "Normal")
        assert msad.sampled_frame_labels(rec, stride=8) == [0] * 13

    def test_ceil_length(self) -> None:
        assert msad.num_sampled_frames(1, 8) == 1
        assert msad.num_sampled_frames(8, 8) == 1
        assert msad.num_sampled_frames(9, 8) == 2


class TestSplit:
    def test_deterministic(self, annotation_file: Path) -> None:
        records = msad.parse_annotation_file(annotation_file)
        first = msad.split_records(records, seed=1)
        second = msad.split_records(records, seed=1)
        assert [r.video_id for r in first[0]] == [r.video_id for r in second[0]]

    def test_split_file_override(self, annotation_file: Path, tmp_path: Path) -> None:
        records = msad.parse_annotation_file(annotation_file)
        split = tmp_path / "test_ids.txt"
        split.write_text("Traffic_accident_001\nNormal_001\n", encoding="utf-8")
        train, test = msad.split_records(records, split_file=split)
        assert {r.video_id for r in test} == {"Traffic_accident_001", "Normal_001"}
        assert len(train) == 4

    def test_unknown_split_id_fails(self, annotation_file: Path, tmp_path: Path) -> None:
        records = msad.parse_annotation_file(annotation_file)
        split = tmp_path / "test_ids.txt"
        split.write_text("nope\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown video ids"):
            msad.split_records(records, split_file=split)


class TestEndToEnd:
    def test_preprocess_writes_contract_files(
        self, annotation_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "MSAD"
        train, test = msad.preprocess(annotation_file, out, seed=0)
        labels = json.loads((out / constants.LABELS_TRAIN_FILENAME).read_text())
        frame_labels = json.loads((out / constants.FRAME_LABELS_TEST_FILENAME).read_text())
        defs = json.loads((out / constants.DEFS_FILENAME).read_text())
        meta = json.loads((out / constants.META_FILENAME).read_text())

        assert set(labels.values()) <= {0, 1}
        assert set(labels) == {r.video_id for r in train}
        for record in test:
            expected = msad.num_sampled_frames(record.total_frames, 8)
            assert len(frame_labels[record.video_id]) == expected
        assert defs[0] == "Normal"
        assert len(meta) == 6
        # weak supervision: train windows live only in meta.json
        assert all(isinstance(v, int) for v in labels.values())

    def test_traffic_scenario_filter(self, annotation_file: Path, tmp_path: Path) -> None:
        records = msad.parse_annotation_file(annotation_file)
        kept = msad.filter_scenarios(records, msad.TRAFFIC_SCENARIOS)
        assert {r.video_id for r in kept} == {
            "Traffic_accident_001",
            "Traffic_accident_002",
            "Fighting_003",
            "Normal_001",
            "Normal_002",
        }  # office scenario dropped
