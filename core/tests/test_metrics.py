"""Tests for per-video score pooling and the rescore CLI (lesson C12).

The regression these lock down: on an all-abnormal test set, pooling raw scores
measures between-clip confidence instead of within-clip localization. That is
what put LaGoVAD's released checkpoint at chance on DoTA.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.evaluate import RESULTS_FILENAME, SCORES_DIRNAME
from core.metrics import (
    macro_video_auc,
    normalize_scores,
    pooled_metrics,
    resolve_score_norm,
)
from core.tools import rescore


def _offset_clips() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Two all-abnormal clips, each perfectly localized, on disjoint scales.

    Clip A scores in [0.0, 0.1], clip B in [0.9, 1.0]. Within each clip the
    ranking is perfect (AUC 1.0), but pooled raw, every frame of B outranks
    every frame of A -- including A's positives -- so the micro AUC collapses.
    """
    scores = [np.array([0.00, 0.02, 0.08, 0.10]), np.array([0.90, 0.92, 0.98, 1.00])]
    labels = [np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])]
    return scores, labels


class TestNormalizeScores:
    def test_minmax_maps_to_unit_interval(self) -> None:
        out = normalize_scores(np.array([2.0, 4.0, 6.0]), constants.SCORE_NORM_MINMAX)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0, abs=1e-6)

    def test_zscore_centers_and_scales(self) -> None:
        out = normalize_scores(np.array([1.0, 2.0, 3.0]), constants.SCORE_NORM_ZSCORE)
        assert out.mean() == pytest.approx(0.0, abs=1e-9)
        assert out.std() == pytest.approx(1.0, abs=1e-6)

    def test_none_is_identity(self) -> None:
        scores = np.array([0.3, 0.1, 0.9])
        assert normalize_scores(scores, constants.SCORE_NORM_NONE) is scores

    def test_constant_clip_does_not_divide_by_zero(self) -> None:
        """238 of 1,397 DoTA clips were flat under the saturated MSAD ckpt."""
        flat = np.full(5, 0.7)
        for method in (constants.SCORE_NORM_MINMAX, constants.SCORE_NORM_ZSCORE):
            assert np.isfinite(normalize_scores(flat, method)).all()

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown score normalization"):
            normalize_scores(np.array([1.0]), "softmax")

    def test_normalization_preserves_within_video_ranking(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.random(32)
        order = np.argsort(scores)
        for method in (constants.SCORE_NORM_MINMAX, constants.SCORE_NORM_ZSCORE):
            assert np.array_equal(np.argsort(normalize_scores(scores, method)), order)


class TestResolveScoreNorm:
    def test_auto_picks_minmax_when_every_video_is_abnormal(self) -> None:
        labels = [np.array([0, 1]), np.array([1, 1])]
        assert resolve_score_norm(constants.SCORE_NORM_AUTO, labels) == (
            constants.SCORE_NORM_MINMAX
        )

    def test_auto_picks_none_on_a_balanced_test_set(self) -> None:
        """MSAD is ~50 % normal: there the between-video scale is real signal."""
        labels = [np.array([0, 1]), np.array([0, 0])]
        assert resolve_score_norm(constants.SCORE_NORM_AUTO, labels) == (
            constants.SCORE_NORM_NONE
        )

    def test_a_handful_of_normal_clips_cannot_flip_the_protocol(self) -> None:
        """DoTA's real shape: 1394 abnormal + 3 normal must still be minmax.

        Those 3 are clips whose anomaly window rounds away at stride 8. An
        all-or-nothing rule let them silently restore raw pooling.
        """
        labels = [np.array([0, 1])] * 1394 + [np.array([0, 0])] * 3
        assert resolve_score_norm(constants.SCORE_NORM_AUTO, labels) == (
            constants.SCORE_NORM_MINMAX
        )

    def test_threshold_boundary_prefers_raw(self) -> None:
        """At exactly the threshold there are enough normals to calibrate."""
        labels = [np.array([0, 1])] * 95 + [np.array([0, 0])] * 5
        assert resolve_score_norm(constants.SCORE_NORM_AUTO, labels) == (
            constants.SCORE_NORM_NONE
        )

    def test_explicit_method_passes_through(self) -> None:
        labels = [np.array([0, 1])]
        assert resolve_score_norm(constants.SCORE_NORM_ZSCORE, labels) == (
            constants.SCORE_NORM_ZSCORE
        )


class TestPooledMetrics:
    def test_normalization_rescues_scale_separated_clips(self) -> None:
        scores, labels = _offset_clips()
        raw = pooled_metrics(scores, labels, constants.SCORE_NORM_NONE)
        normed = pooled_metrics(scores, labels, constants.SCORE_NORM_MINMAX)
        # Every clip is perfectly localized, so the only honest answer is 1.0.
        # Raw pooling gets 0.75: clip B's *negatives* outrank clip A's positives.
        assert raw["auc"] == pytest.approx(0.75)
        assert normed["auc"] == pytest.approx(1.0)
        assert normed["auc_macro"] == pytest.approx(1.0)

    def test_auc_raw_is_reported_under_every_rule(self) -> None:
        scores, labels = _offset_clips()
        raw_auc = pooled_metrics(scores, labels, constants.SCORE_NORM_NONE)["auc"]
        for method in (constants.SCORE_NORM_MINMAX, constants.SCORE_NORM_ZSCORE):
            assert pooled_metrics(scores, labels, method)["auc_raw"] == pytest.approx(
                raw_auc
            )

    def test_auto_resolves_and_is_recorded(self) -> None:
        scores, labels = _offset_clips()
        assert pooled_metrics(scores, labels, constants.SCORE_NORM_AUTO)[
            "score_norm"
        ] == constants.SCORE_NORM_MINMAX

    def test_single_class_test_set_raises(self) -> None:
        with pytest.raises(ValueError, match="single class"):
            pooled_metrics([np.array([0.1, 0.2])], [np.array([1, 1])],
                           constants.SCORE_NORM_NONE)

    def test_macro_skips_single_class_videos(self) -> None:
        scores = [np.array([0.1, 0.9]), np.array([0.2, 0.3])]
        labels = [np.array([0, 1]), np.array([1, 1])]
        auc, counted = macro_video_auc(scores, labels)
        assert counted == 1 and auc == pytest.approx(1.0)

    def test_macro_needs_one_mixed_video(self) -> None:
        with pytest.raises(ValueError, match="macro AUC undefined"):
            macro_video_auc([np.array([0.1, 0.2])], [np.array([1, 1])])


class TestRescoreCli:
    @staticmethod
    def _write_run(root: Path) -> Path:
        scores, labels = _offset_clips()
        scores_dir = root / SCORES_DIRNAME
        scores_dir.mkdir(parents=True)
        for i, (s, gt) in enumerate(zip(scores, labels, strict=True)):
            np.savez(scores_dir / f"clip{i}.npz", score=s.astype(np.float32),
                     gt=gt.astype(np.float32), sim=np.zeros((len(s), 2), np.float32),
                     class_names=np.array(["Normal", "CarAccident"]))
        (root / RESULTS_FILENAME).write_text(
            json.dumps({"dataset": "DoTA", "auc": 0.5, "checkpoint": "x",
                        "per_video": {"clip0": {"num_frames": 4.0}}}),
            encoding="utf-8",
        )
        return root

    def test_load_scores_roundtrip(self, tmp_path: Path) -> None:
        run = self._write_run(tmp_path / "run")
        ids, scores, labels = rescore.load_scores(run / SCORES_DIRNAME)
        assert ids == ["clip0", "clip1"]
        assert len(scores) == len(labels) == 2

    def test_missing_gt_raises(self, tmp_path: Path) -> None:
        scores_dir = tmp_path / SCORES_DIRNAME
        scores_dir.mkdir(parents=True)
        np.savez(scores_dir / "a.npz", score=np.zeros(3, np.float32))
        with pytest.raises(ValueError, match="no 'gt' array"):
            rescore.load_scores(scores_dir)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"No \.npz score files"):
            rescore.load_scores(tmp_path)

    def test_write_updates_results_and_keeps_per_video(self, tmp_path: Path) -> None:
        run = self._write_run(tmp_path / "run")
        rescore.main(["--run-dir", str(run), "--write"])
        results = json.loads((run / RESULTS_FILENAME).read_text(encoding="utf-8"))
        assert results["score_norm"] == constants.SCORE_NORM_MINMAX
        assert results["auc"] == pytest.approx(1.0)
        assert results["auc_raw"] == pytest.approx(0.75)
        assert results["per_video"]["clip0"]["num_frames"] == 4.0  # preserved
        assert results["checkpoint"] == "x"

    def test_write_without_run_dir_exits(self, tmp_path: Path) -> None:
        run = self._write_run(tmp_path / "run")
        with pytest.raises(SystemExit):
            rescore.main(["--scores", str(run / SCORES_DIRNAME), "--write"])

    def test_dry_run_leaves_results_untouched(self, tmp_path: Path) -> None:
        run = self._write_run(tmp_path / "run")
        before = (run / RESULTS_FILENAME).read_text(encoding="utf-8")
        rescore.main(["--run-dir", str(run)])
        assert (run / RESULTS_FILENAME).read_text(encoding="utf-8") == before
