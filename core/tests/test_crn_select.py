"""Tests for the CRN references (``core.crn.reference``) and the E2 selection tool.

What must not happen silently: R4 reading the present or the future, a reference that
is not the proposal's definition, a trend fitted on accident steps, a cluster bootstrap
that resamples clips, and a choice rule that differs from §4.2 steps 4-5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.crn.reference import deviation, reference
from core.data.dota import DotaRecord, resized_frame_labels, sampled_frame_labels
from core.tools import crn_select as cs

SEED = 5
DIM = 4


class TestReference:
    def test_r1_r2_r3_are_constant_over_time(self) -> None:
        x = np.random.default_rng(SEED).normal(size=(12, DIM))
        for kind in ("R1", "R2", "R3"):
            ref = reference(x, kind)
            assert ref.shape == x.shape
            np.testing.assert_allclose(ref, ref[:1].repeat(12, axis=0))
        np.testing.assert_allclose(reference(x, "R1")[0], x.mean(axis=0))
        np.testing.assert_allclose(reference(x, "R2")[0], np.median(x, axis=0))

    def test_r3_ignores_a_minority_of_outliers(self) -> None:
        x = np.zeros((10, DIM))
        x[:3] = 100.0  # 30 % outliers
        np.testing.assert_allclose(reference(x, "R3")[0], 0.0)
        assert reference(x, "R1")[0, 0] == pytest.approx(30.0)

    def test_r4_is_strictly_past_after_warmup(self) -> None:
        x = np.arange(12, dtype=np.float64)[:, None].repeat(DIM, axis=1)
        ref = reference(x, "R4", warmup=4)
        np.testing.assert_allclose(ref[:4, 0], x[:4, 0].mean())  # warm-up
        for t in range(4, 12):
            assert ref[t, 0] == pytest.approx(x[:t, 0].mean())  # tau < t only

    def test_r4_never_reads_the_future(self) -> None:
        rng = np.random.default_rng(SEED)
        x = rng.normal(size=(15, DIM))
        future = x.copy()
        future[10:] += 50.0
        np.testing.assert_allclose(reference(x, "R4")[:11], reference(future, "R4")[:11])

    def test_r4_short_clip_uses_all_steps_as_warmup(self) -> None:
        x = np.random.default_rng(SEED).normal(size=(5, DIM))
        np.testing.assert_allclose(reference(x, "R4", warmup=8), reference(x, "R1"))

    def test_deviation_and_bad_input(self) -> None:
        x = np.random.default_rng(SEED).normal(size=(6, DIM))
        np.testing.assert_allclose(deviation(x, "R1"), np.linalg.norm(x - x.mean(0), axis=1))
        with pytest.raises(ValueError, match="unknown CRN reference"):
            reference(x, "R9")
        with pytest.raises(ValueError, match="non-empty"):
            reference(np.zeros((0, DIM)), "R1")


def _corpus(n: int, steps: int, rng: np.random.Generator, name: str = "c") -> cs.Corpus:
    x, y, share, group = {}, {}, {}, {}
    for i in range(n):
        v = f"vid{i // 2}_{i:06d}"
        lab = np.zeros(steps, dtype=np.int64)
        start = int(rng.integers(2, steps - 4))
        lab[start : start + 3] = 1
        x[v] = rng.normal(size=(steps, DIM)) + 2.0 * lab[:, None]
        y[v], share[v], group[v] = lab, float(lab.mean()), f"vid{i // 2}"
    return cs.Corpus(name, x, y, share, group)


class TestMetrics:
    def test_trend_is_fitted_on_normal_steps_only(self) -> None:
        corpus = _corpus(20, 20, np.random.default_rng(SEED))
        d = {v: np.where(corpus.y[v] == 1, 1e6, 1.0) for v in corpus.ids}
        trend = cs.fit_trend(d, corpus)
        np.testing.assert_allclose(trend(np.linspace(0, 1, 7)), 1.0, atol=1e-6)

    def test_trend_is_held_outside_the_clamp(self) -> None:
        trend = cs.Trend(np.array([1.0, 0.0]), 0.2, 0.8)  # f(tau) = tau
        np.testing.assert_allclose(trend(np.array([0.0, 0.5, 1.0])), [0.2, 0.5, 0.8])

    def test_stratified_auc_is_one_for_a_perfect_score(self) -> None:
        corpus = _corpus(10, 20, np.random.default_rng(SEED))
        perfect = {v: corpus.y[v].astype(np.float64) for v in corpus.ids}
        assert cs.stratified_auc(perfect, corpus, corpus.ids) == pytest.approx(1.0)
        assert cs.stratified_auc(perfect, corpus, []) is None

    def test_cluster_ci_resamples_groups(self) -> None:
        corpus = _corpus(10, 20, np.random.default_rng(SEED))
        values = {v: float(i) for i, v in enumerate(corpus.ids)}
        ci = cs.cluster_ci(values, corpus, 200, SEED)
        assert ci is not None and ci["clusters"] == 5 and ci["clips"] == 10
        assert ci["low"] <= ci["mean"] <= ci["high"]
        assert cs.cluster_ci({}, corpus, 200, SEED) is None

    def test_position_ruler_is_t_over_t(self) -> None:
        np.testing.assert_allclose(cs.rel_position(4), [0.0, 0.25, 0.5, 0.75])


def _block(
    overall: float, b5070: float | None, b70: float | None, strat: float
) -> dict[str, object]:
    def ci(v: float | None) -> dict[str, float] | None:
        return None if v is None else {"mean": v, "low": v, "high": v, "clips": 1, "clusters": 1}

    bins = {b: ci(0.6) for b in constants.V2_SHARE_BIN_LABELS}
    bins["50-70"], bins[">70"] = ci(b5070), ci(b70)
    return {
        "DoTA-dev": {
            "r": {
                "macro": {"overall": ci(overall), "bins": bins},
                "stratified": {"overall": 0.99, "bins": {}},  # must NOT be read
            },
            "d": {"stratified": {"overall": strat, "bins": {}}},
        }
    }


class TestChoose:
    def test_best_eligible_overall_wins(self) -> None:
        refs = {
            "R1": _block(0.70, 0.40, 0.60, 0.6),  # reversal in 50-70
            "R2": _block(0.65, 0.55, 0.55, 0.6),
            "R3": _block(0.66, 0.52, 0.51, 0.6),
            "R4": _block(0.80, 0.60, 0.60, 0.45),
        }  # stratified < 0.5
        choice = cs.choose(refs, "DoTA-dev", cs.METRIC_R)
        assert choice["chosen"] == "R3"
        assert not choice["rows"]["R1"]["eligible"] and not choice["rows"]["R4"]["eligible"]

    def test_none_eligible_drops_crn(self) -> None:
        refs = {"R1": _block(0.7, 0.4, 0.4, 0.6)}
        choice = cs.choose(refs, "DoTA-dev", cs.METRIC_R)
        assert choice["chosen"] is None
        assert cs.verdict(choice, {}) == cs.VERDICT_DROPPED

    def test_a_fail_on_gt70_alone_is_flagged(self) -> None:
        choice = cs.choose({"R1": _block(0.7, 0.6, 0.45, 0.6)}, "DoTA-dev", cs.METRIC_R)
        assert choice["rows"]["R1"]["decided_by_gt70"] is True

    def test_empty_high_bin_is_not_a_pass(self) -> None:
        choice = cs.choose({"R1": _block(0.7, 0.6, None, 0.6)}, "DoTA-dev", cs.METRIC_R)
        assert choice["chosen"] is None

    def test_veto_needs_the_whole_interval_below_zero(self) -> None:
        choice = {"chosen": "R2"}
        assert cs.verdict(choice, {"R2": {"delta_vs_raw": {"high": -0.001}}}) == cs.VERDICT_VETOED
        assert cs.verdict(choice, {"R2": {"delta_vs_raw": {"high": 0.0}}}) == "R2"


class TestEndToEnd:
    def test_references_and_transfer_run_on_synthetic_corpora(self) -> None:
        rng = np.random.default_rng(SEED)
        val, dota, train = (_corpus(16, 20, rng, n) for n in ("T2-val", "DoTA-dev", "T2-train"))
        result = cs.evaluate_reference("R4", val, dota, 100, SEED)
        assert result["DoTA-dev"]["r"]["macro"]["overall"]["mean"] > 0.5
        assert "minus_f" in result["DoTA-dev"]
        transfer = cs.transfer_probe(["R1"], train, dota, 100, SEED)
        assert transfer["raw"]["mean"] > 0.8 and "delta_vs_raw" in transfer["R1"]


class TestDotaStrideLoader:
    """D11: DoTA-dev from s1[::stride], labels rounded onto the s1 length (addendum J1)."""

    def _write(self, tmp_path: Path, lengths: dict[str, int]) -> tuple[Path, Path, Path]:
        s1, rng = tmp_path / "s1", np.random.default_rng(SEED)
        s1.mkdir()
        meta = {}
        for v, n in lengths.items():
            np.save(s1 / f"{v}.npy", rng.normal(size=(n, DIM)).astype(np.float32))
            meta[v] = {"video_start": 0, "video_end": n, "anomaly_start": n // 3,
                       "anomaly_end": 2 * n // 3, "anomaly_class": "ego: turning",
                       "num_frames": n, "subset": "val"}
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        (tmp_path / "split.txt").write_text("\n".join(lengths))
        return s1, tmp_path / "meta.json", tmp_path / "split.txt"

    def test_rows_labels_share_and_groups(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lengths = {"vidA_000010": 100, "vidA_000200": 61, "vidB_000005": 40}
        s1, meta, split = self._write(tmp_path, lengths)
        monkeypatch.setattr(cs, "load_split", lambda name: sorted(lengths))
        corpus, unlabelled = cs.load_dota_dev_s1(s1, meta, split, 3)
        assert unlabelled == 0 and corpus.name == "DoTA-dev-s3"
        for v, n in lengths.items():
            assert len(corpus.x[v]) == len(corpus.y[v]) == -(-n // 3)
            np.testing.assert_allclose(corpus.x[v], np.load(s1 / f"{v}.npy")[::3], rtol=1e-6)
        group = corpus.group
        assert group["vidA_000010"] == group["vidA_000200"] != group["vidB_000005"]
        assert corpus.share["vidA_000010"] == pytest.approx((66 - 33) / 100)

    def test_missing_s1_file_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        s1, meta, split = self._write(tmp_path, {"vidA_000010": 50})
        monkeypatch.setattr(cs, "load_split", lambda name: ["vidA_000010", "vidC_000001"])
        with pytest.raises(FileNotFoundError, match="no s1 feature"):
            cs.load_dota_dev_s1(s1, meta, split, 3)

    def test_resized_labels_equal_sampled_at_matching_length(self) -> None:
        record = DotaRecord("v", "ego: x", 100, (0.3, 0.55))
        for stride in (1, 3, 8):
            assert resized_frame_labels(record, 100, stride) == sampled_frame_labels(record, stride)
        assert len(resized_frame_labels(record, 91, 3)) == 31

    def test_cli_path_flags_are_exclusive(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            cs.main(["--dota-s1-dir", str(tmp_path), "--t2-meta", "m", "--annotation", "a",
                     "--census", "c", "--dada-clip-dir", "d", "--out-dir", str(tmp_path)])
        with pytest.raises(SystemExit):
            cs.main(["--t2-meta", "m", "--annotation", "a", "--census", "c",
                     "--dada-clip-dir", "d", "--out-dir", str(tmp_path)])

    def test_rate_audit_uses_each_corpus_stride(self) -> None:
        corpus = cs.Corpus("c", {}, {"a": np.zeros(10)}, {"a": 0.1}, {"a": "a"})
        out = cs.rate_audit({"dota": (corpus, 10, 3), "t2": (corpus, 30, 8)})
        assert out["dota"]["seconds_per_step"] == pytest.approx(0.3)
        assert out["step_ratio"] == pytest.approx(0.3 / (8 / 30))
