"""MSAD zero-shot census / identity / decode helpers (``KIM_RUN.md`` §18; data-free)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import msad_zs


class TestPure:
    def test_video_class(self) -> None:
        assert msad_zs.video_class("Traffic_accident_20") == "Traffic_accident"
        assert msad_zs.video_class("Object_falling_3") == "Object_falling"
        assert msad_zs.video_class("MSAD_normal_testing_61") == "Normal"

    def test_index_keeps_duplicates(self, tmp_path: Path) -> None:
        a, b = tmp_path / "x" / "Fire_1.mp4", tmp_path / "y" / "Fire_1.mp4"
        found = msad_zs.index_videos([a, b, tmp_path / "Fire_2.mp4"])
        assert len(found["Fire_1"]) == 2 and len(found["Fire_2"]) == 1

    def test_stride_rows(self) -> None:
        assert msad_zs.stride_rows(426, 8) == 54
        assert msad_zs.stride_rows(8, 8) == 1

    @pytest.mark.parametrize(("paths", "rows", "clip", "want"), [
        ([], None, 5, msad_zs.MISSING),
        ([Path("a"), Path("b")], None, 5, msad_zs.DUPLICATE),
        ([Path("a")], 6, 5, msad_zs.ROWS_MISMATCH),
        ([Path("a")], 5, 5, msad_zs.OK),
        ([Path("a")], 5, None, msad_zs.OK),
    ])
    def test_entry_reason(self, paths: list[Path], rows: int | None, clip: int | None,
                          want: str) -> None:
        assert msad_zs.entry_reason(paths, rows, clip) == want

    def test_pick_first_per_class_natural_order(self) -> None:
        ids = ["Fire_10", "Fire_2", "MSAD_normal_testing_100", "MSAD_normal_testing_7",
               "Traffic_accident_3"]
        assert msad_zs.pick_identity_ids(ids) == ["Fire_2", "MSAD_normal_testing_7",
                                                  "Traffic_accident_3"]

    def test_identity_report(self) -> None:
        ref = np.zeros((4, 3), dtype=np.float32)
        assert msad_zs.max_abs_diff(ref, ref + 5e-4) == pytest.approx(5e-4, rel=1e-3)
        assert msad_zs.max_abs_diff(ref, np.zeros((5, 3))) is None

    @staticmethod
    def _rows(seed: int, n: int = 6) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(n, 16)).astype(np.float32)

    def test_identity_same_video_small_drift_passes_but_is_not_bit_identical(self) -> None:
        """The 2026-11-10 read: max abs 1.6e-2 over 8 of 12 videos, cosine >= 0.999992."""
        a, b = self._rows(0), self._rows(1)
        noise = 1e-2 * np.random.default_rng(2).normal(size=a.shape).astype(np.float32)
        rep = msad_zs.identity_report({"a": (a, a + noise), "b": (b, b)}, 1e-3, 0.999)
        assert rep["pass"] and not rep["failures"]
        assert not rep["bit_identical"] and rep["max_abs"] > 1e-3
        assert rep["min_cos"] >= 0.999
        exact = msad_zs.identity_report({"a": (a, a), "b": (b, b)}, 1e-3, 0.999)
        assert exact["pass"] and exact["bit_identical"]

    def test_identity_fails_other_video_row_mismatch_and_swap(self) -> None:
        a, b = self._rows(0), self._rows(1)
        other = msad_zs.identity_report({"a": (a, self._rows(9))}, 1e-3, 0.999)
        assert not other["pass"] and set(other["failures"]) == {"a"}
        rows = msad_zs.identity_report({"a": (a, a[:-1])}, 1e-3, 0.999)
        assert not rows["pass"] and rows["failures"]["a"]["min_cos"] is None
        swapped = msad_zs.identity_report({"a": (a, b), "b": (b, a)}, 1e-3, -1.0)
        assert not swapped["pass"] and set(swapped["failures"]) == {"a", "b"}  # nearest_is_own
        zero = msad_zs.identity_report({"a": (np.zeros((3, 4)), np.ones((3, 4)))}, 1e-3, 0.999)
        assert not zero["pass"]
        assert not msad_zs.identity_report({}, 1e-3, 0.999)["pass"]


class TestCensus:
    def _setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
               frames: dict[str, int]) -> tuple[Path, dict[str, list[int]], Path]:
        root, clip = tmp_path / "videos", tmp_path / "clip"
        clip.mkdir()
        for d in constants.MSAD_ZS_SOURCE_DIRS:
            (root / d).mkdir(parents=True)
        for vid in frames:
            sub = root / ("MSAD_normal_testing" if vid.startswith("MSAD") else "MSAD_anomaly")
            sub.mkdir(parents=True, exist_ok=True)
            (sub / f"{vid}.mp4").write_bytes(b"")
            np.save(clip / f"{vid}.npy", np.zeros((msad_zs.stride_rows(16, 8), 2), np.float32))
        monkeypatch.setattr(msad_zs, "probe_video", lambda p: (frames[p.stem], 30.0))
        monkeypatch.setattr(constants, "MSAD_ZS_TEST_VIDEOS", len(frames))
        labels = {v: [0, 1] for v in frames}
        return root, labels, clip

    def test_pass_and_label_rows_counted(self, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
        frames = {"Fire_1": 16, "MSAD_normal_testing_1": 15}
        root, labels, clip = self._setup(tmp_path, monkeypatch, frames)
        labels["Fire_1"] = [0, 1, 1]  # 3 label rows vs 2 CLIP rows: printed, not gated
        doc = msad_zs.census(root, labels, clip, 8)
        assert doc["summary"]["pass"]
        assert doc["summary"]["label_rows_differ"] == 1
        assert doc["videos"]["Fire_1"]["video"] == "MSAD_anomaly/Fire_1.mp4"

    def test_valu_copy_is_ignored(self, tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
        root, labels, clip = self._setup(tmp_path, monkeypatch, {"Fire_1": 16})
        valu = root / "VALU_MSAD_subset_2026" / "MSAD_split_mask_test_videos"
        valu.mkdir(parents=True)
        (valu / "Fire_1.mp4").write_bytes(b"")
        doc = msad_zs.census(root, labels, clip, 8)
        assert doc["summary"]["pass"]
        assert doc["videos"]["Fire_1"]["video"] == "MSAD_anomaly/Fire_1.mp4"

    def test_missing_source_dir_raises(self, tmp_path: Path) -> None:
        (tmp_path / "MSAD_anomaly").mkdir()
        with pytest.raises(FileNotFoundError, match="MSAD_normal_testing"):
            msad_zs.source_videos(tmp_path)

    def test_census_current(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root, labels, clip = self._setup(tmp_path, monkeypatch, {"Fire_1": 16})
        doc = msad_zs.census(root, labels, clip, 8)
        assert msad_zs.census_current(doc)
        assert not msad_zs.census_current({**doc, "source_dirs": None})  # pre-fix census
        assert not msad_zs.census_current({**doc, "summary": {**doc["summary"], "pass": False}})

    def test_rows_mismatch_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root, labels, clip = self._setup(tmp_path, monkeypatch, {"Fire_1": 24})
        doc = msad_zs.census(root, labels, clip, 8)
        assert not doc["summary"]["pass"]
        assert doc["summary"]["failures"] == {"Fire_1": msad_zs.ROWS_MISMATCH}

    def test_missing_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root, labels, clip = self._setup(tmp_path, monkeypatch, {"Fire_1": 16})
        labels["Fire_9"] = [0]
        monkeypatch.setattr(constants, "MSAD_ZS_TEST_VIDEOS", 2)
        doc = msad_zs.census(root, labels, clip, 8)
        assert doc["summary"]["failures"] == {"Fire_9": msad_zs.MISSING}

    def test_cli_census_exits_on_failure(self, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
        root, labels, clip = self._setup(tmp_path, monkeypatch, {"Fire_1": 24})
        lab = tmp_path / "labels.json"
        lab.write_text(json.dumps(labels))
        with pytest.raises(SystemExit):
            msad_zs.main(["census", "--videos-root", str(root), "--labels", str(lab),
                          "--clip-dir", str(clip), "--out", str(tmp_path / "c.json")])
        assert (tmp_path / "c.json").is_file()


class TestIdentityCli:
    def test_fails_on_other_video(self, tmp_path: Path) -> None:
        ref, new = tmp_path / "ref", tmp_path / "new"
        ref.mkdir()
        new.mkdir()
        np.save(ref / "Fire_1.npy", np.zeros((3, 2), np.float32))
        np.save(new / "Fire_1.npy", np.ones((3, 2), np.float32))
        ids = tmp_path / "ids.txt"
        ids.write_text("Fire_1\n")
        with pytest.raises(SystemExit):
            msad_zs.main(["identity", "--ref", str(ref), "--new", str(new), "--ids-file",
                          str(ids), "--out", str(tmp_path / "i.json")])
        assert not json.loads((tmp_path / "i.json").read_text())["pass"]
