"""Tests for the DVS KNN filler cache builder (core/data/knn_cache.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.data import knn_cache


def _write_features(
    clip_dir: Path, video_id: str, vector: np.ndarray, length: int = 9
) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(video_id)) % 2**32)
    features = rng.standard_normal((length, len(vector))).astype(np.float32) * 0.01
    features[length // 2] = vector  # central frame carries the identity
    np.save(clip_dir / f"{video_id}.npy", features)


class TestBuilder:
    def test_nearest_normal_is_the_semantic_match(self, tmp_path: Path) -> None:
        clip_dir = tmp_path / "clip"
        base = np.eye(8, dtype=np.float32)
        # abnormal a0 points along axis 0; normal n0 is nearly parallel to it
        _write_features(clip_dir, "a0", base[0])
        _write_features(clip_dir, "n0", base[0] * 0.9 + base[1] * 0.1)
        _write_features(clip_dir, "n1", base[2])
        _write_features(clip_dir, "n2", base[3])
        labels = {"a0": 1, "n0": 0, "n1": 0, "n2": 0}
        mapping = knn_cache.build_knn_cache(
            labels, clip_dir, tmp_path / "knn.npz", k=2
        )
        assert mapping["a0"][0] == "n0"
        assert len(mapping["a0"]) == 2

    def test_k_clamped_to_normal_count(self, tmp_path: Path) -> None:
        clip_dir = tmp_path / "clip"
        base = np.eye(4, dtype=np.float32)
        _write_features(clip_dir, "a0", base[0])
        _write_features(clip_dir, "n0", base[1])
        mapping = knn_cache.build_knn_cache(
            {"a0": 1, "n0": 0}, clip_dir, tmp_path / "knn.npz", k=10
        )
        assert mapping["a0"] == ["n0"]

    def test_single_class_fails(self, tmp_path: Path) -> None:
        clip_dir = tmp_path / "clip"
        _write_features(clip_dir, "n0", np.ones(4, dtype=np.float32))
        with pytest.raises(ValueError, match="both classes"):
            knn_cache.build_knn_cache({"n0": 0}, clip_dir, tmp_path / "knn.npz")

    def test_roundtrip_via_npz(self, tmp_path: Path) -> None:
        clip_dir = tmp_path / "clip"
        base = np.eye(4, dtype=np.float32)
        _write_features(clip_dir, "a0", base[0])
        _write_features(clip_dir, "n0", base[1])
        _write_features(clip_dir, "n1", base[2])
        output = tmp_path / "knn.npz"
        built = knn_cache.build_knn_cache(
            {"a0": 1, "n0": 0, "n1": 0}, clip_dir, output, k=2
        )
        assert knn_cache.load_knn_cache(output) == built


class TestMotionKey:
    def test_motion_key_changes_neighbors(self, tmp_path: Path) -> None:
        clip_dir, flow_dir = tmp_path / "clip", tmp_path / "flow"
        flow_dir.mkdir(parents=True)
        base = np.eye(4, dtype=np.float32)
        # visually identical normals, distinct motion statistics
        _write_features(clip_dir, "a0", base[0])
        _write_features(clip_dir, "n0", base[0])
        _write_features(clip_dir, "n1", base[0])
        for vid, magnitude in (("a0", 5.0), ("n0", 5.0), ("n1", 0.01)):
            stats = np.zeros((6, constants.FLOW_STATS_DIM), dtype=np.float32)
            stats[:, 0] = magnitude  # magnitude mean
            stats[:, 7] = 1.0  # all motion in the first angle bin
            np.save(flow_dir / f"{vid}{constants.FLOW_STATS_SUFFIX}", stats)
        mapping = knn_cache.build_knn_cache(
            {"a0": 1, "n0": 0, "n1": 0},
            clip_dir,
            tmp_path / "knn.npz",
            k=2,
            motion_key=True,
            flow_dir=flow_dir,
        )
        assert mapping["a0"][0] == "n0"  # same motion profile wins

    def test_motion_key_without_stats_fails(self, tmp_path: Path) -> None:
        clip_dir, flow_dir = tmp_path / "clip", tmp_path / "flow"
        flow_dir.mkdir(parents=True)
        _write_features(clip_dir, "a0", np.ones(4, dtype=np.float32))
        _write_features(clip_dir, "n0", np.ones(4, dtype=np.float32))
        with pytest.raises(FileNotFoundError, match="raft_extract"):
            knn_cache.build_knn_cache(
                {"a0": 1, "n0": 0},
                clip_dir,
                tmp_path / "knn.npz",
                motion_key=True,
                flow_dir=flow_dir,
            )
