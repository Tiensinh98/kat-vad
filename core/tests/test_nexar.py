"""Tests for the Nexar N0 tools: annotation parsing, census, frozen split.

The failures that matter are silent: a negative's empty time cell parsed as 0.0, a video id in
both class folders, a header frame count trusted without a decode, a census summary whose
abnormal shares drift from their videos, a video on two sides of the split, or a harness that
reads the sealed ``nexar_test`` list.
"""

from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from core import constants
from core.data import nexar
from core.data.v2_splits import SealedSplitError, frozen_splits, lines_sha1, load_split
from core.tools import nexar_census, nexar_splits

FPS = 30
FRAME_SIZE = 32
SEED = 7


def _write_mp4(path: Path, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width = stream.height = FRAME_SIZE
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, FPS)
        for i in range(frames):
            img = np.full((FRAME_SIZE, FRAME_SIZE, 3), (i * 7) % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file_name", "time_of_event", "time_of_alert", "weather", "scene"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_root(root: Path) -> Path:
    """2 positives (3 s, event at 1.5 s / 2.0 s) + 2 negatives (2 s / 4 s)."""
    pos = [
        {
            "file_name": "00001.mp4",
            "time_of_event": "1.5",
            "time_of_alert": "1.0",
            "weather": "Cloudy",
            "scene": "Urban",
        },
        {
            "file_name": "00002.mp4",
            "time_of_event": "2.0",
            "time_of_alert": "1.4",
            "weather": "Clear",
            "scene": "Urban",
        },
    ]
    neg = [
        {
            "file_name": "01001.mp4",
            "time_of_event": "",
            "time_of_alert": "",
            "weather": "Clear",
            "scene": "Rural",
        },
        {
            "file_name": "01002.mp4",
            "time_of_event": "None",
            "time_of_alert": "nan",
            "weather": "Clear",
            "scene": "Urban",
        },
    ]
    train = root / constants.NEXAR_TRAIN_DIR
    _write_metadata(train / "positive" / constants.NEXAR_METADATA_FILENAME, pos)
    _write_metadata(train / "negative" / constants.NEXAR_METADATA_FILENAME, neg)
    for name, frames in (("00001", 90), ("00002", 90)):
        _write_mp4(train / "positive" / f"{name}.mp4", frames)
    for name, frames in (("01001", 60), ("01002", 120)):
        _write_mp4(train / "negative" / f"{name}.mp4", frames)
    return root


class TestParse:
    def test_empty_and_none_cells_are_none_not_zero(self, tmp_path: Path) -> None:
        records = nexar.load_train_records(_build_root(tmp_path))
        by_id = {r.video_id: r for r in records}
        assert [r.video_id for r in records] == ["00001", "00002", "01001", "01002"]
        assert by_id["00001"].label == 1 and by_id["00001"].time_of_event == pytest.approx(1.5)
        assert by_id["01001"].time_of_event is None and by_id["01002"].time_of_alert is None
        assert by_id["01001"].extras == {"weather": "Clear", "scene": "Rural"}
        assert by_id["01002"].relative_path == "train/negative/01002.mp4"

    def test_duplicate_id_across_folders_raises(self, tmp_path: Path) -> None:
        root = _build_root(tmp_path)
        _write_metadata(
            root / "train" / "negative" / constants.NEXAR_METADATA_FILENAME,
            [
                {
                    "file_name": "00001.mp4",
                    "time_of_event": "",
                    "time_of_alert": "",
                    "weather": "",
                    "scene": "",
                }
            ],
        )
        with pytest.raises(ValueError, match="appears twice"):
            nexar.load_train_records(root)

    @pytest.mark.parametrize("name", ["a/00001.mp4", "00001.avi", ".mp4"])
    def test_bad_file_name_raises(self, name: str) -> None:
        with pytest.raises(ValueError):
            nexar.video_id_from_file_name(name)

    def test_missing_file_name_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "metadata.csv"
        path.write_text("video,time_of_event\n1.mp4,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="file_name"):
            nexar.parse_metadata(path, "positive")

    def test_issues(self) -> None:
        def rec(label: int, event: float | None, alert: float | None) -> nexar.NexarRecord:
            return nexar.NexarRecord("x", label, "positive", event, alert)

        assert nexar.annotation_issues(rec(1, 2.0, 1.0)) == []
        assert nexar.annotation_issues(rec(1, None, 1.0)) == [nexar.ISSUE_MISSING_TIMES]
        assert nexar.annotation_issues(rec(1, 1.0, 2.0)) == [nexar.ISSUE_ALERT_AFTER_EVENT]
        assert nexar.annotation_issues(rec(0, 1.0, None)) == [nexar.ISSUE_NEGATIVE_HAS_TIMES]
        assert nexar.ISSUE_NEGATIVE_TIME in nexar.annotation_issues(rec(1, 1.0, -0.5))


class TestCensus:
    def test_end_to_end(self, tmp_path: Path) -> None:
        root = _build_root(tmp_path / "raw")
        census = nexar_census.run_census(root, workers=2, verify_decode=4)
        rows = census["videos"]
        assert rows["00001"]["frames_header"] == 90
        assert rows["00001"]["duration_s"] == pytest.approx(3.0, abs=0.05)
        assert rows["01002"]["fps"] == pytest.approx(FPS)
        assert census["decode_check"] == {"checked": 4, "max_abs_diff": 0}
        s = census["summary"]
        assert s["per_label"] == {"0": 2, "1": 2} and s["issues"] == {}
        assert s["positives_clean"] == 2
        # positives 3 s each, negatives 2 s and 4 s -> duration ranks: 2 < 3 = 3 < 4
        assert s["length_ruler_auc"]["auc"] == pytest.approx(0.5)
        assert s["event_relative_position"]["p50"] == pytest.approx((0.5 + 2 / 3) / 2, abs=0.02)
        share0 = s["abnormal_share_by_post_event_s"]["0"]
        # [1.0, 1.5] and [1.4, 2.0] over 12 s of video
        assert share0["corpus_frame_share"] == pytest.approx(1.1 / 12.0, abs=0.01)
        assert s["categories"]["meta_scene"] == {"Rural": {"0": 1}, "Urban": {"0": 1, "1": 2}}
        assert "Nexar N0 census" in nexar_census.render_markdown(census)

    def test_missing_video_is_recorded_not_raised(self, tmp_path: Path) -> None:
        root = _build_root(tmp_path / "raw")
        (root / "train" / "negative" / "01002.mp4").unlink()
        census = nexar_census.run_census(root, workers=1, verify_decode=0)
        assert census["videos"]["01002"]["issues"] == [nexar_census.ISSUE_PROBE_FAILED]
        assert census["summary"]["probed"] == 3

    def test_event_after_end_and_share_clamp(self) -> None:
        row = {"duration_s": 10.0, "time_of_alert": 8.0, "time_of_event": 9.5}
        assert nexar_census.abnormal_share(row, 2.0) == pytest.approx(0.2)
        assert (
            nexar_census.abnormal_share({**row, "time_of_alert": 9.9, "time_of_event": 9.0}, 0)
            is None
        )
        record = nexar.NexarRecord("v", 1, "positive", 12.0, 11.0)
        rows = nexar_census.build_rows([record], {"v": {"duration_s": 10.0}})
        assert rows["v"]["issues"] == [nexar_census.ISSUE_EVENT_AFTER_END]

    def test_verify_ids_spread_over_both_classes(self) -> None:
        records = [nexar.NexarRecord(f"{i:05d}", i % 2, "positive", None, None) for i in range(40)]
        chosen = nexar_census.verify_ids(records, 6)
        labels = [int(c) % 2 for c in chosen]
        assert len(chosen) == 6 and labels.count(0) == 3 and labels.count(1) == 3


def _synthetic_census(n: int = constants.NEXAR_VIDEOS) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    videos: dict[str, dict[str, Any]] = {}
    for i in range(n):
        label = i % 2
        duration = float(rng.choice([20.0, 40.0]) + rng.uniform(-1, 1))
        event = float(duration * rng.uniform(0.2, 0.8)) if label else None
        issues = ["positive_missing_times"] if label and i % 97 == 1 else []
        videos[f"{i:05d}"] = {
            "label": label,
            "duration_s": duration,
            "issues": issues,
            "time_of_event": None if issues else event,
            "time_of_alert": None if (issues or event is None) else event - 1.0,
        }
    return {"videos": videos, "inputs": {}}


class TestSplits:
    def test_disjoint_cover_fractions_deterministic(self) -> None:
        census = _synthetic_census()
        splits, manifest = nexar_splits.freeze_nexar(census, constants.V2_SPLIT_SEED)
        train, val, test = (
            set(splits[k])
            for k in (
                constants.V2_SPLIT_NEXAR_TRAIN,
                constants.V2_SPLIT_NEXAR_VAL,
                constants.V2_SPLIT_NEXAR_TEST,
            )
        )
        assert not (train & val) and not (train & test) and not (val & test)
        assert train | val | test == set(census["videos"])
        n = constants.NEXAR_VIDEOS
        assert abs(len(test) - constants.NEXAR_TEST_FRACTION * n) <= 10
        assert abs(len(val) - constants.NEXAR_VAL_FRACTION * n) <= 10
        for name in splits:
            per = manifest["per_label"][name]
            assert abs(per["0"] - per["1"]) <= 10, (name, per)
        assert any("|pna" in s for s in manifest["per_stratum"])
        again, _ = nexar_splits.freeze_nexar(census, constants.V2_SPLIT_SEED)
        assert again == splits

    def test_partial_census_refused(self) -> None:
        with pytest.raises(ValueError, match="complete release"):
            nexar_splits.freeze_nexar(_synthetic_census(100), constants.V2_SPLIT_SEED)

    def test_stratum(self) -> None:
        row = {"label": 1, "duration_s": 40.0, "time_of_event": 30.0, "issues": []}
        assert nexar_splits.stratum_of(row, 30.0, [0.4, 0.6]) == "pos|long|p2"
        assert nexar_splits.stratum_of({**row, "label": 0}, 30.0, []) == "neg|long"
        assert (
            nexar_splits.stratum_of({**row, "issues": ["x"]}, 45.0, [0.4, 0.6]) == "pos|short|pna"
        )

    def test_write_check_and_sealed(self, tmp_path: Path) -> None:
        census_path = tmp_path / "census.json"
        census_path.write_text(json.dumps(_synthetic_census()), encoding="utf-8")
        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        (split_dir / constants.V2_SPLITS_MANIFEST_FILENAME).write_text(
            json.dumps({"splits": {}}), encoding="utf-8"
        )
        argv = ["--census", str(census_path), "--split-dir", str(split_dir)]
        nexar_splits.main(argv)
        nexar_splits.main([*argv, "--check"])
        with pytest.raises(FileExistsError):
            nexar_splits.main(argv)
        frozen = frozen_splits(split_dir)
        train = load_split(constants.V2_SPLIT_NEXAR_TRAIN, split_dir)
        assert lines_sha1(train) == frozen[constants.V2_SPLIT_NEXAR_TRAIN]["sha1"]
        with pytest.raises(SealedSplitError):
            load_split(constants.V2_SPLIT_NEXAR_TEST, split_dir)
        assert len(load_split(constants.V2_SPLIT_NEXAR_TEST, split_dir, final=True)) > 0
