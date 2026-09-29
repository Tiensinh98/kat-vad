"""Tests for the v2 E1 harness (addendum §8, J1-J9) — data-free, model-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from core import constants
from core.data.dota import DotaRecord
from core.metrics import cluster_bootstrap_ci
from core.tools import rate_matched_eval as e1


def _record(total: int, start: int, end: int) -> DotaRecord:
    return DotaRecord("vid_000001", "ego: turning", total, (start / total, end / total))


class TestWindows:
    def test_divisible_length_needs_no_tail(self) -> None:
        assert e1.window_starts(28, 20, 4) == [0, 4, 8]

    def test_non_divisible_length_gets_tail_window(self) -> None:
        starts = e1.window_starts(30, 20, 4)
        assert starts == [0, 4, 8, 10]
        assert starts[-1] + 20 == 30

    def test_short_clip_is_one_window(self) -> None:
        assert e1.window_starts(13, 20, 4) == [0]
        assert e1.window_starts(20, 20, 4) == [0]

    def test_overlap_average_of_consistent_windows_is_identity(self) -> None:
        truth = np.linspace(0.0, 1.0, 30)
        parts = [(s, truth[s : s + 20]) for s in e1.window_starts(30, 20, 4)]
        np.testing.assert_allclose(e1.overlap_average(parts, 30), truth)

    def test_overlap_average_means_disagreeing_windows(self) -> None:
        out = e1.overlap_average([(0, np.array([1.0, 1.0])), (1, np.array([3.0, 3.0]))], 3)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])

    def test_uncovered_step_raises(self) -> None:
        with pytest.raises(ValueError, match="covered by no window"):
            e1.overlap_average([(0, np.ones(2))], 4)


class TestNative:
    @pytest.mark.parametrize(("steps", "stride", "frames"), [(13, 8, 100), (34, 3, 100), (1, 3, 2)])
    def test_length_is_native_frame_count(self, steps: int, stride: int, frames: int) -> None:
        assert len(e1.to_native(np.random.default_rng(0).random(steps), stride, frames)) == frames

    def test_sampled_frames_keep_their_score_and_tail_is_held(self) -> None:
        steps = np.array([0.0, 1.0, 0.5])
        out = e1.to_native(steps, 3, 9)
        np.testing.assert_allclose(out[[0, 3, 6]], steps)
        np.testing.assert_allclose(out[1:3], [1 / 3, 2 / 3])
        np.testing.assert_allclose(out[6:], 0.5)

    def test_labels_equal_raw_window_when_length_matches(self) -> None:
        labels = e1.native_labels(_record(100, 40, 70), 100)
        assert labels.sum() == 30 and labels[40] == 1 and labels[39] == 0 and labels[70] == 0

    def test_labels_rescale_to_cache_length(self) -> None:
        labels = e1.native_labels(_record(100, 40, 70), 50)
        assert len(labels) == 50 and labels.sum() == 15


class TestRegression:
    def test_match_within_tolerance(self) -> None:
        per_video = {"a": {"max_score": 0.5}, "b": {"max_score": 0.9}}
        assert e1.regression_mismatches({"a": 0.50001, "b": 0.9}, per_video, 1e-4) == []

    def test_mismatch_and_absent_reported(self) -> None:
        bad = e1.regression_mismatches({"a": 0.6, "c": 0.1}, {"a": {"max_score": 0.5}}, 1e-4)
        assert len(bad) == 2 and "absent" in bad[1]

    def test_check_regression_raises(self, tmp_path: Path) -> None:
        results = tmp_path / "results.json"
        results.write_text(json.dumps({"per_video": {"a": {"max_score": 0.5}}}))
        run = e1.Run("s1", tmp_path / "ckpt.pt", results)
        with pytest.raises(RuntimeError, match="J9 regression FAILED"):
            e1.check_regression(run, {"a": 0.7})
        e1.check_regression(run, {"a": 0.5})


def _ci(mean: float, low: float) -> dict[str, float]:
    return {"mean": mean, "low": low, "high": mean + 0.05}


class TestDecide:
    def test_none_eligible_keeps_a(self) -> None:
        assert e1.decide({"B": _ci(0.02, -0.01), "C": None})["adopted"] == "A"

    def test_negative_interval_is_not_a_switch(self) -> None:
        assert e1.decide({"B": _ci(-0.05, -0.08), "C": _ci(-0.04, -0.07)})["adopted"] == "A"

    def test_single_eligible_is_adopted(self) -> None:
        assert e1.decide({"B": _ci(0.02, -0.01), "C": _ci(0.03, 0.01)})["adopted"] == "C"

    def test_both_eligible_close_means_prefer_b(self) -> None:
        out = e1.decide({"B": _ci(0.030, 0.01), "C": _ci(0.035, 0.01)})
        assert out == {"eligible": ["B", "C"], "adopted": "B"}

    def test_both_eligible_far_means_take_larger(self) -> None:
        assert e1.decide({"B": _ci(0.02, 0.01), "C": _ci(0.05, 0.02)})["adopted"] == "C"


class TestClusterBootstrap:
    def test_single_cluster_has_zero_width(self) -> None:
        ci = cluster_bootstrap_ci({"a": 0.2, "b": 0.4}, {"a": "g", "b": "g"}, 200, 0, 0.95)
        assert ci is not None
        assert ci["mean"] == pytest.approx(0.3)
        assert ci["low"] == pytest.approx(0.3) and ci["high"] == pytest.approx(0.3)
        assert ci["clusters"] == 1

    def test_empty_is_none(self) -> None:
        assert cluster_bootstrap_ci({}, {}, 10, 0, 0.95) is None


def _pointwise_scores(
    _model: Any, feats: torch.Tensor, class_feats_fn: Any, max_vis_len: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    class_feats_fn()
    return feats[:, 0].clone(), torch.zeros(len(feats), 1)


class TestScoreSteps:
    def test_windowed_equals_whole_for_a_pointwise_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(e1, "sliding_window_scores", _pointwise_scores)
        feats = torch.rand(37, 4)
        names = ["Normal", "CarAccident"]

        def text_fn(batch: list[str]) -> torch.Tensor:
            return torch.zeros(len(batch), 4)

        model: Any = None
        whole = e1.score_steps(model, feats, text_fn, names, "v", e1.PROTOCOLS["B"])
        windowed = e1.score_steps(model, feats, text_fn, names, "v", e1.PROTOCOLS["C"])
        np.testing.assert_allclose(windowed, whole, rtol=1e-6)


def _clip(video_id: str, labels: np.ndarray, group: str) -> e1.Clip:
    return e1.Clip(video_id, np.zeros((len(labels), 2)), labels, float(labels.mean()), group)


class TestSummarize:
    def test_perfect_arm_is_adopted_and_extras_printed(self) -> None:
        rng = np.random.default_rng(0)
        clips = {}
        for i in range(12):
            labels = np.zeros(40, dtype=np.int64)
            labels[10 + i : 25 + i] = 1
            clips[f"src{i // 2}_{i:06d}"] = _clip(f"src{i // 2}_{i:06d}", labels, f"src{i // 2}")
        per_run: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for seed in ("s1", "s2"):
            per_run[seed] = {
                "A": {v: rng.random(40) for v in clips},
                "B": {v: c.labels + 0.1 * rng.random(40) for v, c in clips.items()},
                "C": {v: rng.random(40) for v in clips},
            }
        out = e1.summarize(per_run, clips, constants.SEED)
        assert out["macro"]["B"]["mean"] == pytest.approx(1.0)
        assert out["decision"]["adopted"] == "B"
        assert out["delta_vs_A"]["B"]["clusters"] == 6
        printed = {"position_ruler", "macro_by_share_bin", "per_seed_macro", "micro_minmax"}
        assert set(out["printed"]) == printed
        assert "adopt B" in e1.render_markdown(
            {"runs": ["s1", "s2"], "clips": 12, "off_length": 0, "summary": out}
        )
