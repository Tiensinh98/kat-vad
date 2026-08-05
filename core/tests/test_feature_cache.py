"""Resume semantics for the per-video feature caches (`core/tools/feature_cache.py`).

The cases that matter are the ones a Colab disconnect actually produces: a
target that exists but is truncated, and a leftover ``.part`` file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core import constants
from core.tools import feature_cache


def _write(target: Path, rows: int = 4) -> np.ndarray:
    array = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    feature_cache.save_array(target, array)
    return array


class TestAtomicWrite:
    def test_roundtrip_leaves_no_part_file(self, tmp_path: Path) -> None:
        target = tmp_path / "clip_000001.npy"
        array = _write(target)
        np.testing.assert_array_equal(np.load(target), array)
        assert list(tmp_path.iterdir()) == [target]  # .part renamed, not left behind

    def test_part_path_appends_rather_than_replaces_the_suffix(self, tmp_path: Path) -> None:
        staged = feature_cache.part_path(tmp_path / "id.npy")
        assert staged.name == f"id.npy{constants.CACHE_PART_SUFFIX}"


class TestIsComplete:
    def test_absent_and_empty_and_truncated_are_all_incomplete(self, tmp_path: Path) -> None:
        assert not feature_cache.is_complete(tmp_path / "nope.npy")

        empty = tmp_path / "empty.npy"  # mkdir succeeded, write did not (lesson C10)
        empty.touch()
        assert not feature_cache.is_complete(empty)

        truncated = tmp_path / "truncated.npy"
        _write(truncated, rows=64)
        payload = truncated.read_bytes()
        truncated.write_bytes(payload[: len(payload) // 2])
        assert not feature_cache.is_complete(truncated)

    def test_complete_file_reads_back(self, tmp_path: Path) -> None:
        target = tmp_path / "good.npy"
        _write(target)
        assert feature_cache.is_complete(target)

    def test_zero_length_feature_array_is_incomplete(self, tmp_path: Path) -> None:
        target = tmp_path / "zero.npy"
        feature_cache.save_array(target, np.zeros((0, 3), dtype=np.float32))
        assert not feature_cache.is_complete(target)


class TestPendingItems:
    def test_reports_done_and_pending(self, tmp_path: Path, caplog) -> None:
        sources = {f"v{i}": tmp_path / f"v{i}.mp4" for i in range(4)}
        out = tmp_path / "clip"
        out.mkdir()
        _write(out / "v0.npy")
        _write(out / "v1.npy")
        with caplog.at_level(logging.INFO, logger=feature_cache.LOGGER.name):
            pending = feature_cache.pending_items(sources, out)
        assert [vid for vid, _ in pending] == ["v2", "v3"]
        assert "Resume: 2/4 already cached" in caplog.text

    def test_truncated_output_is_re_extracted_with_a_warning(
        self, tmp_path: Path, caplog
    ) -> None:
        out = tmp_path / "clip"
        out.mkdir()
        (out / "v0.npy").write_bytes(b"\x93NUMPY truncated")
        with caplog.at_level(logging.WARNING, logger=feature_cache.LOGGER.name):
            pending = feature_cache.pending_items({"v0": tmp_path / "v0.mp4"}, out)
        assert [vid for vid, _ in pending] == ["v0"]
        assert "unreadable (interrupted write)" in caplog.text

    def test_stale_part_file_is_cleared(self, tmp_path: Path) -> None:
        out = tmp_path / "clip"
        out.mkdir()
        stale = feature_cache.part_path(out / "v0.npy")
        stale.write_bytes(b"half a write")
        feature_cache.pending_items({"v0": tmp_path / "v0.mp4"}, out)
        assert not stale.exists()

    def test_force_ignores_complete_outputs(self, tmp_path: Path) -> None:
        out = tmp_path / "clip"
        out.mkdir()
        _write(out / "v0.npy")
        pending = feature_cache.pending_items({"v0": tmp_path / "v0.mp4"}, out, force=True)
        assert [vid for vid, _ in pending] == ["v0"]


class TestProgress:
    def test_counts_up_and_renders(self) -> None:
        progress = feature_cache.Progress(total=2)
        assert progress.step().startswith("[1/2] elapsed ")
        assert progress.step().startswith("[2/2] ")

    def test_duration_switches_to_hours(self) -> None:
        assert feature_cache.format_duration(65.4) == "1m05s"
        assert feature_cache.format_duration(3 * 3600 + 7 * 60) == "3h07m"
