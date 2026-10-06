"""T2 position prior ``p_T2`` (Amendment 6 G6 (a)): fitted on T2 windows, scored per DoTA length."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.tools import position_prior as pp
from core.tools.protocol_b_eval import read_clip_scores, write_clip_scores


def _mid_labels(n: int) -> list[int]:
    lab = [0] * n
    for t in range(n // 3, 2 * n // 3):
        lab[t] = 1
    return lab


class TestTrainingFrames:
    def test_reads_every_train_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        train = {"w0": 1, "w1": 0, "w2": 1}
        (tmp_path / constants.LABELS_TRAIN_FILENAME).write_text(json.dumps(train))
        (tmp_path / constants.META_FILENAME).write_text(json.dumps({w: {} for w in train}))
        frames = {"w0": _mid_labels(20), "w1": [0] * 20, "w2": _mid_labels(15)}

        def fake(_entry: dict[str, Any], window_id: str) -> list[int]:
            return frames[window_id]

        monkeypatch.setattr(pp, "window_frame_labels", fake)
        matrix, target, windows = pp.training_frames(tmp_path)
        assert windows == 3
        assert matrix.shape == (55, constants.V2_K_POSITION_DEGREE)
        assert int(target.sum()) == sum(map(sum, frames.values()))

    def test_a_bag_label_disagreeing_with_frames_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / constants.LABELS_TRAIN_FILENAME).write_text(json.dumps({"w0": 0}))
        (tmp_path / constants.META_FILENAME).write_text(json.dumps({"w0": {}}))
        monkeypatch.setattr(pp, "window_frame_labels", lambda _e, _w: _mid_labels(20))
        with pytest.raises(ValueError, match="disagree"):
            pp.training_frames(tmp_path)


class TestPriorScores:
    def test_a_mid_window_prior_peaks_mid_clip_at_every_length(self) -> None:
        feats = np.concatenate([pp.position_features(20) for _ in range(30)])
        target = np.concatenate([np.array(_mid_labels(20)) for _ in range(30)])
        scores = pp.prior_scores(feats, target, {"a": 30, "b": 99, "c": 30}, seed=0)
        assert {len(scores["a"]), len(scores["b"])} == {30, 99}
        np.testing.assert_array_equal(scores["a"], scores["c"])
        for v in scores.values():
            mid = len(v) // 2
            assert v[mid] > v[0] and v[mid] > v[-1]


class TestClipScoresRoundTrip:
    def test_write_then_read_is_identity(self, tmp_path: Path) -> None:
        scores = {"x_000001": np.array([0.1, 0.2]), "y_000002": np.array([0.5, 0.4, 0.3])}
        labels = {"x_000001": np.array([0, 1]), "y_000002": np.array([1, 1, 0])}
        path = tmp_path / "s.npz"
        write_clip_scores(path, scores, labels)
        got_s, got_l = read_clip_scores(path)
        assert sorted(got_s) == sorted(scores)
        for v in scores:
            np.testing.assert_allclose(got_s[v], scores[v])
            np.testing.assert_array_equal(got_l[v], labels[v])

    def test_mismatched_clips_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="same clips"):
            write_clip_scores(tmp_path / "s.npz", {"a": np.zeros(2)}, {"b": np.zeros(2)})
