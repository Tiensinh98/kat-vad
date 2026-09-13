"""Tests for fixed-length windows over a clip-level feature cache (lesson C28).

The invariants that matter are the ones a wrong window would break silently:
no padding, no window crossing a clip boundary, appearance and flow sliced by
the *same* window, and the unwindowed path byte-identical to before.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.data.dataset_files import write_windows
from core.data.windows import (
    FeatureSlicer,
    Window,
    cap_windows,
    load_windows,
    plan_windows,
    slice_labels,
    window_id,
    windows_payload,
)


class TestWindow:
    def test_length(self) -> None:
        assert Window("a", 4, 12).length == 8

    @pytest.mark.parametrize(("start", "end"), [(-1, 5), (5, 5), (5, 4)])
    def test_rejects_degenerate(self, start: int, end: int) -> None:
        with pytest.raises(ValueError):
            Window("a", start, end)


class TestPlanWindows:
    def test_exact_fit_gives_one_window(self) -> None:
        assert plan_windows("a", 32, 32, 16) == [Window("a", 0, 32)]

    def test_overlapping_windows(self) -> None:
        assert plan_windows("a", 64, 32, 16) == [
            Window("a", 0, 32), Window("a", 16, 48), Window("a", 32, 64),
        ]

    def test_no_overlap_when_stride_equals_length(self) -> None:
        assert plan_windows("a", 64, 32, 32) == [Window("a", 0, 32), Window("a", 32, 64)]

    def test_never_pads_a_short_clip(self) -> None:
        """A clip shorter than one window contributes nothing -- lesson C27."""
        assert plan_windows("a", 31, 32, 16) == []

    def test_last_window_never_runs_past_the_clip(self) -> None:
        for window in plan_windows("a", 40, 32, 4):
            assert window.end <= 40

    @pytest.mark.parametrize(("length", "stride"), [(0, 16), (-1, 16), (32, 0), (32, -3)])
    def test_rejects_bad_geometry(self, length: int, stride: int) -> None:
        with pytest.raises(ValueError):
            plan_windows("a", 64, length, stride)


class TestSliceLabels:
    def test_slices(self) -> None:
        assert slice_labels([0, 0, 1, 1, 0], Window("a", 1, 4)) == [0, 1, 1]

    def test_raises_when_the_window_crosses_the_clip(self) -> None:
        with pytest.raises(ValueError, match="never cross a clip boundary"):
            slice_labels([0, 1, 0], Window("a", 1, 5))


class TestWindowId:
    def test_is_sortable_within_a_source(self) -> None:
        ids = [window_id("clip", i) for i in (0, 2, 10)]
        assert ids == ["clip__w000", "clip__w002", "clip__w010"]
        assert sorted(ids) == ids

    def test_uses_the_shared_separator(self) -> None:
        assert constants.WINDOW_ID_SEPARATOR in window_id("clip", 1)


class TestLoadWindows:
    def test_absent_file_means_not_windowed(self, tmp_path: Path) -> None:
        assert load_windows(tmp_path) is None

    def test_round_trips_through_write_windows(self, tmp_path: Path) -> None:
        windows = {"c__w000": Window("c", 0, 4), "c__w001": Window("c", 4, 8)}
        write_windows(tmp_path, windows)
        assert load_windows(tmp_path) == windows

    def test_payload_is_sorted_and_typed(self) -> None:
        payload = windows_payload({"b__w000": Window("b", 2, 6), "a__w000": Window("a", 0, 4)})
        assert list(payload) == ["a__w000", "b__w000"]
        assert payload["a__w000"] == {"source": "a", "start": 0, "end": 4}
        json.dumps(payload)  # must be JSON-serializable as-is

    def test_refuses_to_write_an_empty_map(self, tmp_path: Path) -> None:
        """An empty windows.json would make every id unresolvable; omit the file."""
        with pytest.raises(ValueError, match=r"empty windows\.json"):
            write_windows(tmp_path, {})
        assert not (tmp_path / constants.WINDOWS_FILENAME).exists()

    def test_warns_when_windows_are_not_equal_length(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        write_windows(tmp_path, {"a__w000": Window("a", 0, 4), "b__w000": Window("b", 0, 9)})
        with caplog.at_level(logging.WARNING):
            load_windows(tmp_path)
        assert "carry the label" in caplog.text


def _cache(tmp_path: Path, rows: int = 10, dim: int = 3, suffix: str = ".npy") -> Path:
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    np.save(cache / f"clip0{suffix}", np.arange(rows * dim, dtype=np.float32).reshape(rows, dim))
    return cache


class TestFeatureSlicerIdentityPath:
    def test_unwindowed_returns_the_whole_array(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        slicer = FeatureSlicer()
        assert not slicer.is_windowed
        assert slicer.load(cache, "clip0").shape == (10, 3)

    def test_unwindowed_source_is_the_id(self) -> None:
        assert FeatureSlicer().source_of("clip0") == "clip0"
        assert FeatureSlicer().window_of("clip0") is None


class TestFeatureSlicerWindowedPath:
    def test_slices_to_the_window(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        slicer = FeatureSlicer({"clip0__w001": Window("clip0", 4, 7)})
        rows = slicer.load(cache, "clip0__w001")
        assert rows.shape == (3, 3)
        np.testing.assert_array_equal(rows, np.load(cache / "clip0.npy")[4:7])

    def test_source_of_resolves_the_cache_stem(self) -> None:
        slicer = FeatureSlicer({"clip0__w001": Window("clip0", 4, 7)})
        assert slicer.source_of("clip0__w001") == "clip0"

    def test_suffix_slices_sibling_caches_identically(self, tmp_path: Path) -> None:
        """Flow stats must be sliced by the same window as the features (C13)."""
        cache = _cache(tmp_path, suffix=constants.FLOW_STATS_SUFFIX)
        slicer = FeatureSlicer({"clip0__w000": Window("clip0", 2, 5)})
        rows = slicer.load(cache, "clip0__w000", constants.FLOW_STATS_SUFFIX)
        expected = np.load(cache / f"clip0{constants.FLOW_STATS_SUFFIX}")[2:5]
        np.testing.assert_array_equal(rows, expected)

    def test_unknown_window_id_raises(self, tmp_path: Path) -> None:
        slicer = FeatureSlicer({"clip0__w000": Window("clip0", 0, 4)})
        with pytest.raises(KeyError, match=r"windows\.json"):
            slicer.load(_cache(tmp_path), "clip0__w999")

    def test_raises_when_the_cache_is_shorter_than_the_window(self, tmp_path: Path) -> None:
        """A stride mismatch between cache and windows must not slice silently (C2)."""
        slicer = FeatureSlicer({"clip0__w000": Window("clip0", 8, 40)})
        with pytest.raises(ValueError, match="different stride"):
            slicer.load(_cache(tmp_path, rows=10), "clip0__w000")


class TestCapWindows:
    """Lesson C32: a fixed hop gives each clip windows in proportion to its length."""

    def _plan(self, length: int) -> list[Window]:
        return plan_windows("a", length, 24, 12)

    def test_short_clip_is_untouched(self) -> None:
        planned = self._plan(49)  # DADA's median abnormal clip, raw frames
        assert len(planned) == 3
        assert cap_windows(planned, 4) == planned

    def test_long_clip_is_capped(self) -> None:
        planned = self._plan(139)  # DADA's median normal clip
        assert len(planned) == 10
        capped = cap_windows(planned, 4)
        assert len(capped) == 4
        assert all(w in planned for w in capped)

    def test_cap_is_evenly_spaced_and_keeps_both_ends(self) -> None:
        """Keeping the first N would drop the accident, which sits at the clip end."""
        planned = self._plan(139)
        capped = cap_windows(planned, 4)
        assert capped[0] == planned[0]
        assert capped[-1] == planned[-1]
        gaps = [b.start - a.start for a, b in itertools.pairwise(capped)]
        assert max(gaps) - min(gaps) <= 12  # one hop of slack from rounding

    def test_cap_equalizes_the_class_ratio(self) -> None:
        """The defect it exists to prevent, in one assertion."""
        abnormal, normal = self._plan(49), self._plan(139)
        assert len(normal) / len(abnormal) > 3
        ratio = len(cap_windows(normal, 4)) / len(cap_windows(abnormal, 4))
        assert ratio <= 1.5

    def test_rejects_a_nonpositive_cap(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            cap_windows(self._plan(139), 0)
