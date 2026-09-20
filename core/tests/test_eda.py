"""Tests for the EDA package (core.eda, core.tools.eda).

The properties under test are the ones that decide whether a published number
means what it says, so each is pinned against a hand-computed value rather than
a golden file:

* the constant-score-per-clip oracle reproduces the DADA-2000 test split's 0.9086
  (``RESULTS_DADA.md`` §4 printed 0.9069 from the wrong frame split; corrected 2026-09-08);
* the length-only baseline (lesson **C28**) fires on DADA's trimmed-clip shape and
  stays silent on a corpus whose classes share a length distribution;
* the kernel-coverage and MIL-k tables reproduce §5's DADA / DoTA / MSAD rows;
* a vanished anomaly window is *found*, because the campaign shipped four of
  them undetected.

No data, no downloads, no GPU: synthetic dataset dirs under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.eda import corpus, features, labels, protocol, report
from core.flow import raft_extract
from core.tools import eda as eda_cli

DIM = 16
SEED = 7


def write_dataset(
    root: Path,
    frame_labels: dict[str, list[int]],
    labels_train: dict[str, int] | None = None,
    meta_extra: dict[str, dict[str, object]] | None = None,
) -> Path:
    """Materialize the four standard dataset files for a synthetic corpus."""
    root.mkdir(parents=True, exist_ok=True)
    labels_train = labels_train or {}
    meta: dict[str, dict[str, object]] = {}
    for video_id, vector in frame_labels.items():
        meta[video_id] = {
            "class_name": "CarAccident" if any(vector) else "Normal",
            "split": "test",
            "total_frames": len(vector) * constants.FRAME_STRIDE,
            "sampled_frames": len(vector),
            "ego_involve": True,
            "anomaly_span": [0.1, 0.2] if any(vector) else None,
        }
    for video_id, value in labels_train.items():
        meta[video_id] = {
            "class_name": "CarAccident" if value else "Normal",
            "split": "train",
            "total_frames": 40 * constants.FRAME_STRIDE,
            "sampled_frames": 40,
            "ego_involve": False,
            "anomaly_span": None,
        }
    for video_id, patch in (meta_extra or {}).items():
        meta.setdefault(video_id, {}).update(patch)
    payloads = {
        constants.LABELS_TRAIN_FILENAME: labels_train,
        constants.FRAME_LABELS_TEST_FILENAME: frame_labels,
        constants.DEFS_FILENAME: ["Normal", "CarAccident"],
        constants.META_FILENAME: meta,
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    return root


@pytest.fixture
def tiny(tmp_path: Path) -> corpus.DatasetFiles:
    frame_labels = {
        "mixed_a": [0, 0, 1, 1, 0, 0],
        "mixed_b": [0, 1, 0, 0],
        "normal_a": [0, 0, 0, 0, 0],
        "normal_b": [0, 0, 0],
        "vanished": [0, 0, 0, 0],
    }
    train_labels = {f"tr{i}": int(i % 2 == 0) for i in range(10)}
    root = write_dataset(
        tmp_path / "ds",
        frame_labels,
        train_labels,
        # 'vanished' is annotated abnormal in meta.json -- its span exists in the
        # source annotation -- but rounds away at this stride, leaving an all-zero
        # label vector. That is the shape of RESULTS_DADA.md §8.3's four clips.
        meta_extra={"vanished": {"class_name": "CarAccident", "anomaly_span": [0.31, 0.33]}},
    )
    return corpus.load_dataset_files(root, "SYNTH")


class TestCorpusShape:
    def test_split_sizes_and_dvs_budget(self, tiny) -> None:
        sizes = corpus.split_sizes(tiny)
        assert sizes["test_clips"] == 5
        assert sizes["test_abnormal"] == 2
        assert sizes["test_sampled_frames"] == 22
        assert sizes["test_positive_frames"] == 3
        assert sizes["train_abnormal"] == 5
        # DVSFeatureDataset.__len__ is 2 x abnormal-train.
        assert sizes["dvs_dataset_len"] == 10
        assert sizes["split_leak_ids"] == []

    def test_split_leak_is_detected(self, tmp_path: Path) -> None:
        root = write_dataset(
            tmp_path / "leak", {"shared": [0, 1, 0]}, {"shared": 1, "other": 0}
        )
        files = corpus.load_dataset_files(root, "LEAK")
        assert corpus.split_sizes(files)["split_leak_ids"] == ["shared"]

    def test_kernel_coverage_matches_results_dada_table(self) -> None:
        # RESULTS_DADA.md §5: a clip at or below the kernel is fully covered.
        cov = corpus.kernel_coverage([4, 9, 9, 20, 100], kernel=9)
        assert cov["clips_fully_covered"] == 3
        assert cov["fraction_fully_covered"] == pytest.approx(0.6)
        assert cov["median_coverage"] == pytest.approx(1.0)
        # A clip 10x the kernel sees 9 % of itself per output timestep.
        assert corpus.kernel_coverage([90], kernel=9)["median_coverage"] == pytest.approx(0.1)

    def test_kernel_coverage_frame_weighting(self) -> None:
        cov = corpus.kernel_coverage([2, 2, 96], kernel=9)
        assert cov["frames_in_fully_covered_clips"] == 4
        assert cov["fraction_frames_fully_covered"] == pytest.approx(4 / 100)

    def test_mil_topk_floor_is_a_plain_max_on_short_clips(self) -> None:
        floor = corpus.mil_topk_floor([9, 13, 86], topk_pct=16)
        # k = max(1, T // 16): 9 -> 1, 13 -> 1, 86 -> 5
        assert floor["clips_at_k1"] == 2
        assert floor["k_distribution"] == {"1": 2, "5": 1}

    def test_subgroup_table_absent_field_is_not_an_error(self, tiny) -> None:
        assert corpus.subgroup_table(tiny, "fault_label") == {
            "key": "fault_label", "present": False, "groups": {}
        }
        assert corpus.subgroup_table(tiny, "class_name")["present"] is True

    def test_describe_handles_empty(self) -> None:
        assert corpus.describe([], "nothing")["n"] == 0

    def test_missing_file_names_the_runbook(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="DADA_SETUP"):
            corpus.load_dataset_files(tmp_path, "MISSING")


class TestLabelGeometry:
    def test_find_spans(self) -> None:
        assert labels.find_spans(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]
        assert labels.find_spans(np.array([1, 1])) == [(0, 2)]
        assert labels.find_spans(np.array([0, 0])) == []
        assert labels.find_spans(np.array([], dtype=np.int8)) == []

    def test_positive_counts(self, tiny) -> None:
        pos = labels.positives_per_clip(tiny)
        assert pos["abnormal_clips"]["n"] == 2
        assert pos["abnormal_clips_with_one_positive"] == 1

    def test_vanished_window_is_found(self, tiny) -> None:
        # 'vanished' is declared abnormal in meta but has an all-zero vector --
        # exactly the four DADA clips of RESULTS_DADA.md §8.3.
        found = labels.vanished_windows(tiny)
        assert found["count"] == 1
        assert found["video_ids"] == ["vanished"]

    def test_normal_clips_are_not_reported_as_vanished(self, tiny) -> None:
        assert "normal_a" not in labels.vanished_windows(tiny)["video_ids"]

    def test_multi_span_clips_are_counted(self, tmp_path: Path) -> None:
        root = write_dataset(tmp_path / "multi", {"m": [1, 0, 1, 0, 1], "n": [0, 0]}, {"t": 1})
        files = corpus.load_dataset_files(root, "MULTI")
        assert labels.span_stats(files)["multi_span_clips"] == 1


class TestProtocol:
    def test_clip_oracle_reproduces_the_dada_test_split(self) -> None:
        """The DADA-2000 test split: 476 positives, 872 negatives inside abnormal
        clips, 3,896 frames in all-normal clips -> the constant-score oracle
        scores **0.9086**, which is what the run on disk measures.

        RESULTS_DADA.md §4 printed 0.9069 because it used the 3,880-frame
        ``0_Normal_Driving`` *subgroup* count instead of the all-normal-*clip*
        count. The 16-frame difference is exactly the four vanished-window clips
        (§8.3): abnormal in ``meta.json``, all-zero after stride-8 rounding, so
        they are all-normal clips for every metric. Corrected 2026-09-08.
        """
        abnormal = np.concatenate([np.ones(476, dtype=np.int8), np.zeros(872, dtype=np.int8)])
        normal = np.zeros(3896, dtype=np.int8)
        oracle = protocol.clip_constant_oracle([abnormal, normal])
        assert oracle["auc_micro"] == pytest.approx(0.9086, abs=5e-5)
        assert oracle["auc_macro"] == 0.5

    def test_oracle_is_one_when_no_all_normal_clips_exist(self) -> None:
        # DoTA's shape: every clip mixed -> a clip classifier wins nothing extra,
        # because there is no clip whose frames are all negative to rank below.
        mixed = [np.array([0, 1, 0], dtype=np.int8) for _ in range(4)]
        assert protocol.clip_constant_oracle(mixed)["auc_micro"] == pytest.approx(0.5)

    def test_length_leak_fires_on_the_dada_shape(self) -> None:
        """Lesson C28. DADA-2000's abnormal clips are trimmed (T <= 17) and its
        normal clips are not (median 19), so a detector reading only the frame
        count separates them. Shape reproduced in miniature: short abnormal
        clips, long normal ones, disjoint ranges."""
        labels = [np.array([0, 1, 0], dtype=np.int8) for _ in range(5)]
        labels += [np.zeros(20, dtype=np.int8) for _ in range(5)]
        leak = protocol.clip_length_leak(labels)
        assert leak["direction"] == "shorter"
        assert leak["auc_clip_level"] == pytest.approx(1.0)
        assert leak["auc_micro"] > 0.9
        assert leak["auc_macro"] == 0.5
        # every normal clip is longer than the longest abnormal one
        assert leak["disjoint_normal_clips"] == 5
        assert leak["disjoint_normal_frames"] == 100

    def test_length_leak_is_chance_when_lengths_carry_nothing(self) -> None:
        """A corpus whose classes share a length distribution must not fire C28."""
        labels = [np.array([0, 1, 0, 0], dtype=np.int8) for _ in range(6)]
        labels += [np.zeros(4, dtype=np.int8) for _ in range(6)]
        leak = protocol.clip_length_leak(labels)
        assert leak["auc_clip_level"] == pytest.approx(0.5)
        assert leak["disjoint_normal_clips"] == 0

    def test_length_leak_detects_the_longer_direction_too(self) -> None:
        """The leak is not always 'shorter = abnormal'; report whichever way it goes."""
        labels = [np.concatenate([np.zeros(19, np.int8), np.ones(1, np.int8)]) for _ in range(4)]
        labels += [np.zeros(3, dtype=np.int8) for _ in range(4)]
        leak = protocol.clip_length_leak(labels)
        assert leak["direction"] == "longer"
        assert leak["auc_clip_level"] == pytest.approx(1.0)

    def test_length_leak_raises_on_a_single_clip_class(self) -> None:
        with pytest.raises(ValueError, match="single clip class"):
            protocol.clip_length_leak([np.array([0, 1], dtype=np.int8) for _ in range(3)])

    def test_protocol_report_survives_a_corpus_with_no_normal_clip(self, tmp_path: Path) -> None:
        """The leak check must degrade, never break protocol_report (DoTA-like)."""
        files = corpus.load_dataset_files(
            write_dataset(tmp_path / "ds", {f"m{i}": [0, 1, 0] for i in range(4)}), "SYNTH"
        )
        report = protocol.protocol_report(files)
        assert "skipped" in report["clip_length_leak"]
        assert report["clip_constant_oracle"]["auc_micro"] == pytest.approx(0.5)

    def test_pair_decomposition_sums(self) -> None:
        arrays = [np.array([1, 0, 0], dtype=np.int8), np.zeros(5, dtype=np.int8)]
        pairs = protocol.pair_decomposition(arrays)
        assert pairs["total_pairs"] == 1 * 7
        assert pairs["within_clip_pairs"] == 1 * 2
        assert pairs["cross_clip_pairs"] == 5
        assert pairs["within_clip_fraction"] + pairs["cross_clip_fraction"] == pytest.approx(1.0)

    def test_frame_share(self, tiny) -> None:
        share = protocol.frame_share(tiny.test_label_arrays())
        assert share["clips"]["all_normal"] == 3  # normal_a, normal_b, vanished
        assert share["clips"]["mixed"] == 2
        assert share["frames"]["all_normal"] == 12
        assert share["frame_fraction"]["all_normal"] == pytest.approx(12 / 22)

    def test_macro_resolution_counts_two_class_clips(self, tiny) -> None:
        res = protocol.macro_resolution(tiny.test_label_arrays())
        assert res["two_class_clips"] == 2
        assert res["single_class_clips"] == 3

    def test_score_norm_auto_follows_the_label_distribution(self, tiny) -> None:
        # 60 % normal clips -> raw pooling, the MSAD/DADA protocol (lesson C12).
        assert protocol.score_norm_resolution(tiny.test_label_arrays())["resolves_to"] == (
            constants.SCORE_NORM_NONE
        )
        all_abnormal = [np.array([0, 1], dtype=np.int8) for _ in range(20)]
        assert protocol.score_norm_resolution(all_abnormal)["resolves_to"] == (
            constants.SCORE_NORM_MINMAX
        )

    def test_curve_flatness_flags_a_constant_curve(self) -> None:
        flat = protocol.curve_flatness([np.full(5, 0.2), np.full(5, 0.8)])
        assert flat["constant_curve_clips"] == 2
        assert flat["within_clip_variance_frame_weighted"] == pytest.approx(0.0)
        assert flat["between_over_within"] is None
        varied = protocol.curve_flatness([np.linspace(0, 1, 8), np.linspace(0, 1, 8)])
        assert varied["between_over_within"] == pytest.approx(0.0)

    def test_zero_byte_score_file_raises(self, tmp_path: Path) -> None:
        (tmp_path / "a.npz").write_bytes(b"")
        with pytest.raises(ValueError, match="C11b"):
            protocol.load_score_curves(tmp_path)

    def test_load_score_curves_roundtrip(self, tmp_path: Path) -> None:
        np.savez(tmp_path / "clip.npz", score=np.array([0.1, 0.9]), gt=np.array([0, 1]))
        scores, gts, ids = protocol.load_score_curves(tmp_path)
        assert ids == ["clip"] and scores[0][1] == pytest.approx(0.9) and gts[0][1] == 1


class TestFeatures:
    @staticmethod
    def _write_features(clip_dir: Path, frame_labels: dict[str, list[int]],
                        separable: bool, rng: np.random.Generator) -> None:
        clip_dir.mkdir(parents=True, exist_ok=True)
        for video_id, vector in frame_labels.items():
            array = rng.normal(size=(len(vector), DIM)).astype(np.float32)
            if separable:
                array[np.asarray(vector) == 1, 0] += 12.0
            np.save(clip_dir / f"{video_id}.npy", array)

    @pytest.fixture
    def probe_corpus(self, tmp_path: Path):
        rng = np.random.default_rng(SEED)
        frame_labels = {
            f"clip{i:02d}": [int(t == 5 or t == 6) for t in range(12)] for i in range(24)
        }
        frame_labels.update({f"norm{i:02d}": [0] * 12 for i in range(12)})
        root = write_dataset(tmp_path / "probe", frame_labels, {"tr0": 1, "tr1": 0})
        return corpus.load_dataset_files(root, "PROBE"), frame_labels, rng

    def test_length_mismatch_raises_lesson_c2(self, tmp_path: Path, tiny) -> None:
        clip_dir = tmp_path / "clip"
        clip_dir.mkdir()
        np.save(clip_dir / "mixed_a.npy", np.zeros((99, DIM), dtype=np.float32))
        with pytest.raises(ValueError, match="C2/C13"):
            features.load_clip_features(
                clip_dir, ["mixed_a"], {"mixed_a": 6}
            )

    def test_missing_features_are_reported_not_raised(self, tmp_path: Path) -> None:
        found, missing = features.load_clip_features(tmp_path, ["nope"], None)
        assert found == {} and missing == ["nope"]

    def test_autocorrelation_is_one_for_a_constant_clip(self) -> None:
        constant = {"a": np.ones((6, DIM), dtype=np.float32)}
        result = features.temporal_autocorrelation(constant, max_lag=2)
        assert result["cosine_by_lag"]["lag1"]["mean"] == pytest.approx(1.0)

    def test_autocorrelation_skips_clips_shorter_than_the_lag(self) -> None:
        result = features.temporal_autocorrelation({"a": np.ones((2, DIM))}, max_lag=4)
        assert result["cosine_by_lag"]["lag4"]["pairs"] == 0
        assert result["cosine_by_lag"]["lag4"]["mean"] is None

    def test_variance_decomposition_separates_scene_from_dynamics(self) -> None:
        rng = np.random.default_rng(SEED)
        far_apart = {
            "a": rng.normal(size=(20, DIM)) * 0.01,
            "b": rng.normal(size=(20, DIM)) * 0.01 + 50.0,
        }
        assert features.variance_decomposition(far_apart)["between_over_within"] > 100

    def test_frame_probe_finds_a_planted_signal(self, probe_corpus) -> None:
        files, frame_labels, rng = probe_corpus
        clip_dir = files.data_dir / "clip"
        self._write_features(clip_dir, frame_labels, separable=True, rng=rng)
        loaded, _ = features.load_clip_features(clip_dir, files.test_ids, None)
        result = features.frame_linear_probe(loaded, files.frame_labels_test, seed=SEED)
        assert result["ran"] is True
        assert result["auc_macro"] > 0.9, "a planted linear signal must be recoverable"

    def test_frame_probe_reports_chance_on_noise(self, probe_corpus) -> None:
        files, frame_labels, rng = probe_corpus
        clip_dir = files.data_dir / "noise"
        self._write_features(clip_dir, frame_labels, separable=False, rng=rng)
        loaded, _ = features.load_clip_features(clip_dir, files.test_ids, None)
        result = features.frame_linear_probe(loaded, files.frame_labels_test, seed=SEED)
        assert result["ran"] is True
        assert 0.25 < result["auc_macro"] < 0.75, "pure noise must not localize"

    def test_probe_declines_on_too_few_clips(self, tiny) -> None:
        result = features.frame_linear_probe({"mixed_a": np.zeros((6, DIM))},
                                             tiny.frame_labels_test)
        assert result["ran"] is False

    def test_flow_stats_reads_the_23_dim_descriptor(self, tmp_path: Path) -> None:
        flow_dir = tmp_path / "flow"
        flow_dir.mkdir()
        np.save(flow_dir / f"tr0{features.FLOW_STATS_SUFFIX}", np.ones((7, 23)))
        stats = features.flow_stats(flow_dir, ["tr0", "absent"])
        assert stats["clips"] == 1 and stats["raw_dims"] == 23
        assert stats["dead_dims"] == 23 and stats["nonfinite_frames"] == 0

    def test_flow_stats_missing_dir_is_not_an_error(self, tmp_path: Path) -> None:
        assert features.flow_stats(tmp_path, ["x"])["clips"] == 0

    def test_target_block_absent_when_only_raw_stats_are_cached(self, tmp_path: Path) -> None:
        flow_dir = tmp_path / "flow"
        flow_dir.mkdir()
        np.save(flow_dir / f"tr0{features.FLOW_STATS_SUFFIX}", np.ones((7, 23)))
        assert "target" not in features.flow_stats(flow_dir, ["tr0"])

    @staticmethod
    def _write_flow(flow_dir: Path, item_id: str, target: np.ndarray) -> None:
        flow_dir.mkdir(exist_ok=True)
        np.save(flow_dir / f"{item_id}{features.FLOW_STATS_SUFFIX}",
                np.ones((len(target), constants.FLOW_STATS_DIM)))
        np.save(flow_dir / f"{item_id}.npy", target)

    def test_target_baselines_split_between_and_within_item_variance(
        self, tmp_path: Path
    ) -> None:
        # Two items, each internally constant: every bit of the target's variance
        # is item identity, so an item-mean oracle scores exactly 0.
        flow_dir = tmp_path / "flow"
        dim = constants.FLOW_DIM
        self._write_flow(flow_dir, "a", np.full((4, dim), 1.0))
        self._write_flow(flow_dir, "b", np.full((4, dim), 3.0))
        target = features.flow_stats(flow_dir, ["a", "b"])["target"]
        assert target["items"] == 2 and target["frames"] == 8
        assert target["mse_zero_predictor"] == pytest.approx(5.0)  # (1 + 9) / 2
        assert target["mse_global_mean_predictor"] == pytest.approx(1.0)
        assert target["mse_item_mean_predictor"] == pytest.approx(0.0)
        assert target["between_item_share"] == pytest.approx(1.0)

    def test_target_between_item_share_is_zero_when_items_share_a_mean(
        self, tmp_path: Path
    ) -> None:
        flow_dir = tmp_path / "flow"
        dim = constants.FLOW_DIM
        rows = np.concatenate([np.zeros((2, dim)), np.full((2, dim), 2.0)])
        self._write_flow(flow_dir, "a", rows)
        self._write_flow(flow_dir, "b", rows)
        target = features.flow_stats(flow_dir, ["a", "b"])["target"]
        assert target["mse_global_mean_predictor"] == pytest.approx(1.0)
        assert target["mse_item_mean_predictor"] == pytest.approx(1.0)
        assert target["between_item_share"] == pytest.approx(0.0)

    def test_raw_energy_share_names_the_unnormalized_stat(self, tmp_path: Path) -> None:
        # mag_max in pixel units against an L1-normalized angle histogram: the
        # shape that makes e_O's scale a property of the corpus, not of the head.
        flow_dir = tmp_path / "flow"
        flow_dir.mkdir()
        raw = np.full((5, constants.FLOW_STATS_DIM), 0.05)
        raw[:, constants.FLOW_STAT_NAMES.index("mag_max")] = 30.0
        np.save(flow_dir / f"tr0{features.FLOW_STATS_SUFFIX}", raw)
        share = features.flow_stats(flow_dir, ["tr0"])["raw_energy_share"]
        assert share[0]["name"] == "mag_max"
        assert share[0]["energy_share"] > 0.99

    def test_projection_check_matches_a_real_seeded_projection(
        self, tmp_path: Path
    ) -> None:
        # e_O = s @ M with M ~ N(0, 1/23): mean_j E[s_j^2] predicts the per-dim
        # second moment of the target. If this drifts, cache and projection
        # disagree and every kip_rec measured on them is uninterpretable.
        rng = np.random.default_rng(SEED)
        raw = rng.normal(0.0, 3.0, size=(512, constants.FLOW_STATS_DIM))
        matrix = raft_extract.make_projection()
        flow_dir = tmp_path / "flow"
        flow_dir.mkdir()
        np.save(flow_dir / f"tr0{features.FLOW_STATS_SUFFIX}", raw)
        np.save(flow_dir / "tr0.npy", (raw @ matrix).astype(np.float32))
        target = features.flow_stats(flow_dir, ["tr0"])["target"]
        ratio = target["predicted_second_moment_from_raw"] / target["mse_zero_predictor"]
        assert ratio == pytest.approx(1.0, abs=0.1)


class TestReport:
    def test_verdicts_fire_on_the_dada_shape(self, tmp_path: Path) -> None:
        # 9-frame clips under a kernel-9 head, dominated by all-normal clips:
        # the exact geometry RESULTS_DADA.md §4-§5 describes.
        frame_labels = {f"ab{i:02d}": [0] * 4 + [1] + [0] * 4 for i in range(10)}
        frame_labels.update({f"no{i:02d}": [0] * 9 for i in range(30)})
        root = write_dataset(tmp_path / "dada_like", frame_labels, {"tr0": 1, "tr1": 0})
        payload = report.build_report(
            dataset="DADA_LIKE", data_dir=root, clip_dir=None, flow_dir=None,
            scores_dir=None, sections=(constants.EDA_SECTION_CORPUS,
                                       constants.EDA_SECTION_LABELS,
                                       constants.EDA_SECTION_PROTOCOL),
            kernel=9, topk_pct=16, max_clips=0, probe=False, seed=SEED,
        )
        titles = {v["title"] for v in payload["verdicts"]}
        assert any("lesson C27" in t for t in titles)
        assert any("clip classification" in t for t in titles)
        assert any("plain max" in t for t in titles)
        assert payload["corpus"]["kernel_coverage_test"]["fraction_fully_covered"] == 1.0

    def test_verdicts_stay_quiet_on_a_healthy_corpus(self, tmp_path: Path) -> None:
        frame_labels = {
            f"c{i:02d}": [0] * 40 + [1] * 40 + [0] * 20 for i in range(20)
        }
        root = write_dataset(tmp_path / "healthy", frame_labels, {"tr0": 1, "tr1": 0})
        payload = report.build_report(
            dataset="HEALTHY", data_dir=root, clip_dir=None, flow_dir=None, scores_dir=None,
            sections=(constants.EDA_SECTION_CORPUS, constants.EDA_SECTION_LABELS,
                      constants.EDA_SECTION_PROTOCOL),
            kernel=9, topk_pct=16, max_clips=0, probe=False, seed=SEED,
        )
        assert [v["level"] for v in payload["verdicts"]] == ["OK"]

    @staticmethod
    def _flow_report(baseline: float, between_share: float) -> dict[str, object]:
        return {
            "features": {
                "flow": {
                    "clips": 4,
                    "raw_energy_share": [{"name": "mag_max", "energy_share": 0.93}],
                    "target": {
                        "mse_zero_predictor": baseline * 2,
                        "mse_global_mean_predictor": baseline,
                        "mse_item_mean_predictor": baseline * (1.0 - between_share),
                        "between_item_share": between_share,
                        "predicted_second_moment_from_raw": baseline * 2,
                    },
                }
            }
        }

    def test_unnormalized_kip_target_is_flagged(self) -> None:
        # The measured DADA2000_orig shape: a constant predictor already scores an
        # MSE in the tens while every task loss is O(1).
        verdicts = report._verdicts(self._flow_report(14.5, 0.8))
        titles = {v["title"] for v in verdicts}
        assert any("reconstruction target is unnormalized" in t for t in titles)
        assert any("item identity, not dynamics" in t for t in titles)
        flagged = next(v for v in verdicts if "unnormalized" in v["title"])
        assert "grad_probe" in flagged["action"]

    def test_normalized_kip_target_is_not_flagged(self) -> None:
        verdicts = report._verdicts(self._flow_report(1.0, 0.2))
        assert [v["level"] for v in verdicts] == ["OK"]

    def test_markdown_renders_and_json_roundtrips(self, tmp_path: Path, tiny) -> None:
        payload = report.build_report(
            dataset="SYNTH", data_dir=tiny.data_dir, clip_dir=None, flow_dir=None,
            scores_dir=None, sections=constants.EDA_SECTIONS,
            kernel=9, topk_pct=16, max_clips=0, probe=False, seed=SEED,
        )
        json_path, md_path = report.write_report(payload, tmp_path / "out")
        text = md_path.read_text(encoding="utf-8")
        assert "# EDA — SYNTH" in text
        assert "clip-level oracle" in text.lower()
        assert "vanished" in text.lower()
        with json_path.open(encoding="utf-8") as fh:
            assert json.load(fh)["dataset"] == "SYNTH"

    def test_compare_renders_every_report(self, tmp_path: Path, tiny) -> None:
        payload = report.build_report(
            dataset="SYNTH", data_dir=tiny.data_dir, clip_dir=None, flow_dir=None,
            scores_dir=None, sections=constants.EDA_SECTIONS, kernel=9, topk_pct=16,
            max_clips=0, probe=False, seed=SEED,
        )
        text = report.compare_reports([payload, {**payload, "dataset": "OTHER"}])
        assert "| property | SYNTH | OTHER |" in text
        assert "clip-oracle micro AUC" in text

    def test_plots_are_written(self, tmp_path: Path, tiny) -> None:
        payload = report.build_report(
            dataset="SYNTH", data_dir=tiny.data_dir, clip_dir=None, flow_dir=None,
            scores_dir=None, sections=constants.EDA_SECTIONS, kernel=9, topk_pct=16,
            max_clips=0, probe=False, seed=SEED,
        )
        written = report.write_plots(payload, tiny, tmp_path / "plots")
        assert written and all(p.is_file() and p.stat().st_size > 0 for p in written)


class TestCli:
    def test_report_command_writes_both_files(self, tmp_path: Path, tiny) -> None:
        out = tmp_path / "cli"
        eda_cli.main([
            "report", "--dataset", "SYNTH", "--data-dir", str(tiny.data_dir),
            "--output-dir", str(out), "--sections", "corpus,labels,protocol", "--no-probe",
        ])
        assert (out / constants.EDA_REPORT_JSON_FILENAME).is_file()
        assert (out / constants.EDA_REPORT_MD_FILENAME).is_file()

    def test_unknown_section_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown section"):
            eda_cli._resolve_sections("corpus,nonsense")

    def test_compare_command(self, tmp_path: Path, tiny) -> None:
        out = tmp_path / "cli2"
        eda_cli.main([
            "report", "--dataset", "SYNTH", "--data-dir", str(tiny.data_dir),
            "--output-dir", str(out), "--no-probe",
        ])
        target = tmp_path / "compare.md"
        eda_cli.main([
            "compare", str(out / constants.EDA_REPORT_JSON_FILENAME), "--output", str(target),
        ])
        assert "cross-corpus comparison" in target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Windowed corpora (Phase 2a) — every id is a window id, the cache is by source
# ---------------------------------------------------------------------------

WINDOW_LEN = 6


@pytest.fixture
def windowed_eda(tmp_path: Path) -> tuple[corpus.DatasetFiles, Path]:
    """Two source clips -> overlapping windows, plus a source-keyed feature cache."""
    from core.data.dataset_files import write_windows
    from core.data.windows import Window

    # abn: 12 sampled frames, anomaly in frames 8-11 -> window 0 is a NEGATIVE
    # window of an abnormal clip (correct), window 1 holds the anomaly.
    windows = {
        "abn__w000": Window("abn", 0, 6),
        "abn__w001": Window("abn", 3, 9),
        "abn__w002": Window("abn", 6, 12),
        "gone__w000": Window("gone", 0, 6),
        "nrm__w000": Window("nrm", 0, 6),
        "nrm__w001": Window("nrm", 6, 12),
    }
    source_labels = {
        "abn": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        "gone": [0] * 6,
        "nrm": [0] * 12,
    }
    frame_labels = {
        wid: source_labels[w.source][w.start : w.end] for wid, w in windows.items()
    }
    root = write_dataset(
        tmp_path / "wds",
        frame_labels,
        {f"tr{i}": int(i % 2 == 0) for i in range(10)},
        meta_extra={
            wid: {
                "class_name": "CarAccident" if w.source in ("abn", "gone") else "Normal",
                "anomaly_span": [0.6, 0.9] if w.source in ("abn", "gone") else None,
                "source": w.source, "start": w.start, "end": w.end,
            }
            for wid, w in windows.items()
        },
    )
    write_windows(root, windows)
    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()
    rng = np.random.default_rng(0)
    for source, vector in source_labels.items():
        rows = rng.standard_normal((len(vector), 16)).astype(np.float32)
        rows[np.asarray(vector) == 1] += 3.0  # a signal the probe can find
        np.save(clip_dir / f"{source}.npy", rows)
    return corpus.load_dataset_files(root, "SYNTH"), clip_dir


class TestWindowedCorpusIsRecognized:
    def test_load_dataset_files_picks_up_windows_json(self, windowed_eda) -> None:
        files, _ = windowed_eda
        assert files.is_windowed
        assert files.source_of("abn__w001") == "abn"

    def test_unwindowed_corpus_stays_unwindowed(self, tiny) -> None:
        assert not tiny.is_windowed
        assert tiny.source_of("mixed_a") == "mixed_a"


class TestWindowedFeatureLookup:
    def test_features_resolve_through_the_source_keyed_cache(self, windowed_eda) -> None:
        """The bug this pins reported 760 of 760 test clips missing."""
        files, clip_dir = windowed_eda
        feats, missing = features.load_clip_features(
            clip_dir, files.test_ids,
            {v: len(files.frame_labels_test[v]) for v in files.test_ids},
            files.slicer,
        )
        assert missing == []
        assert len(feats) == len(files.test_ids)
        assert all(len(f) == WINDOW_LEN for f in feats.values())

    def test_each_window_gets_its_own_rows(self, windowed_eda) -> None:
        files, clip_dir = windowed_eda
        feats, _ = features.load_clip_features(clip_dir, files.test_ids, None, files.slicer)
        source = np.load(clip_dir / "abn.npy")
        np.testing.assert_array_equal(feats["abn__w001"], source[3:9])

    def test_without_the_slicer_everything_reads_as_missing(self, windowed_eda) -> None:
        """Why the slicer argument exists, stated as a test."""
        files, clip_dir = windowed_eda
        _feats, missing = features.load_clip_features(clip_dir, files.test_ids)
        assert len(missing) == len(files.test_ids)


class TestWindowedProbeGrouping:
    def test_probe_folds_group_by_source_not_by_window(self, windowed_eda) -> None:
        """Overlapping windows share real frames; splitting them across folds leaks."""
        files, _clip_dir = windowed_eda
        ids = ["abn__w000", "abn__w001", "nrm__w000", "nrm__w001"]
        groups = features._source_groups(ids, files.slicer)
        assert groups[0] == groups[1], "two windows of one clip must share a fold"
        assert groups[2] == groups[3]
        assert groups[0] != groups[2]

    def test_unwindowed_grouping_is_one_group_per_clip(self, tiny) -> None:
        ids = tiny.test_ids
        assert features._source_groups(ids, tiny.slicer) == list(range(len(ids)))


class TestWindowedVanishedCheck:
    def test_a_negative_window_of_an_abnormal_clip_is_not_vanished(
        self, windowed_eda
    ) -> None:
        """Producing those windows is the point of re-sharding, not a defect."""
        files, _ = windowed_eda
        result = labels.vanished_windows(files)
        assert result["unit"] == "source clip"
        assert "abn__w000" not in result["video_ids"]
        assert result["negative_windows_of_abnormal_clips"] >= 1

    def test_an_abnormal_source_with_no_positive_anywhere_is_still_flagged(
        self, windowed_eda
    ) -> None:
        files, _ = windowed_eda
        assert labels.vanished_windows(files)["video_ids"] == ["gone"]

    def test_unwindowed_behaviour_is_unchanged(self, tiny) -> None:
        result = labels.vanished_windows(tiny)
        assert result["unit"] == "clip"
        assert result["video_ids"] == ["vanished"]
        assert result["negative_windows_of_abnormal_clips"] == 0
