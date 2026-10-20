"""v2 TAD zero-shot probe: the read-out as pre-registered (v2 addendum §26).

The failures that matter are silent: a transfer sentence written when only one of the two
intervals clears 0, arms paired on different labels, a micro AUC read without its clip oracle,
and a T0 that no longer reproduces (the labels or cache drifted under every v2 number).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.tools import tadz_readout as tz
from core.tools.e3_readout import REPORTS

SEEDS = list(constants.V2_E3_SEEDS)
FRAMES, ONSET, END = 40, 15, 25


def _interval(low: float, high: float) -> dict[str, Any]:
    return {"mean": (low + high) / 2, "low": low, "high": high}


def _contrast(t_low: float, t_high: float, b_low: float, b_high: float) -> dict[str, Any]:
    return {"paired_t95": _interval(t_low, t_high), "clip_bootstrap": _interval(b_low, b_high)}


class TestTransferSentence:
    def test_both_above_zero_transfers(self) -> None:
        assert "transfer to" in tz.transfer_sentence(_contrast(0.01, 0.05, 0.002, 0.04))

    def test_only_seeds_above_zero_is_no_gain(self) -> None:
        sentence = tz.transfer_sentence(_contrast(0.01, 0.05, -0.01, 0.04))
        assert sentence.startswith("no transfer gain")

    def test_only_clips_above_zero_is_no_gain(self) -> None:
        sentence = tz.transfer_sentence(_contrast(-0.01, 0.05, 0.01, 0.04))
        assert sentence.startswith("no transfer gain")

    def test_both_below_zero_costs(self) -> None:
        assert "cost" in tz.transfer_sentence(_contrast(-0.05, -0.01, -0.04, -0.002))

    def test_missing_bootstrap_never_transfers(self) -> None:
        c = {"paired_t95": _interval(0.01, 0.05), "clip_bootstrap": None}
        assert tz.transfer_sentence(c).startswith("no transfer gain")


class TestReferenceSentence:
    @pytest.mark.parametrize(
        ("low", "high", "word"),
        [(0.01, 0.05, "above"), (-0.05, -0.01, "below"), (-0.01, 0.02, "not separated")],
    )
    def test_reads_the_bootstrap_only(self, low: float, high: float, word: str) -> None:
        assert word in tz.reference_sentence(_interval(low, high))


def _video_labels(abnormal: bool) -> np.ndarray:
    gt = np.zeros(FRAMES, dtype=np.int64)
    if abnormal:
        gt[ONSET:END] = 1
    return gt


def _write_run(run_dir: Path, scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> None:
    out = run_dir / tz.SCORES_DIRNAME
    out.mkdir(parents=True)
    for v, s in scores.items():
        np.savez(out / f"{v}.npz", score=s.astype(np.float32), gt=labels[v].astype(np.float32))


def _labels() -> dict[str, np.ndarray]:
    labels = {f"a{i:03d}": _video_labels(True) for i in range(constants.V2_TADZ_ABNORMAL)}
    normals = constants.V2_TADZ_VIDEOS - constants.V2_TADZ_ABNORMAL
    labels.update({f"n{i:03d}": _video_labels(False) for i in range(normals)})
    return labels


def _scores(
    labels: dict[str, np.ndarray], gain: float, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    return {v: rng.normal(size=FRAMES) + gain * g for v, g in labels.items()}


def _t0_scores(labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Tests that need T9(a) to pass pin the registered micro to these curves' own."""
    rng = np.random.default_rng(0)
    return _scores(labels, 0.8, rng)


def _build(
    root: Path, gains: dict[str, float], labels: dict[str, np.ndarray] | None = None
) -> None:
    labels = labels or _labels()
    rng = np.random.default_rng(1)
    _write_run(root / constants.V2_TADZ_REFERENCE, _t0_scores(labels), labels)
    for arm, gain in gains.items():
        for s in SEEDS:
            _write_run(root / arm / f"s{s}", _scores(labels, gain, rng), labels)


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(root=root, t2_v2_dir=None, seeds=SEEDS, seed=0)


def _pin_t0(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    labels = tz.load_run(root / constants.V2_TADZ_REFERENCE)
    monkeypatch.setattr(constants, "V2_TADZ_T0_MICRO", tz.micro(*labels))


GAINS = {constants.V2_TADZ_CONTROL: 0.5, constants.V2_TADZ_PRINTED_ARM: 0.5,
         constants.V2_TADZ_ARM: 2.0}


class TestRun:
    def test_clear_gain_writes_the_transfer_sentence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _build(tmp_path, GAINS)
        _pin_t0(monkeypatch, tmp_path)
        r = tz.run(_args(tmp_path))
        assert "transfer to" in r["endpoint_sentence"]
        assert r["two_class_videos"] == constants.V2_TADZ_ABNORMAL
        assert set(r["arms"]) == set(constants.V2_TADZ_ARMS)
        assert (tmp_path / REPORTS / tz.READOUT_MD).read_text().count("89.56") == 1

    def test_rulers_are_printed_beside_micro(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _build(tmp_path, GAINS)
        _pin_t0(monkeypatch, tmp_path)
        rl = tz.run(_args(tmp_path))["rulers"]
        # one constant per video -> the oracle ranks every abnormal video's frames together
        assert 0.5 < rl["clip_oracle_micro"] < 1.0
        assert rl["length_ruler_micro"] == pytest.approx(0.5)  # every video has the same length
        assert rl["p_t2_macro"] is None

    def test_t0_that_does_not_reproduce_stops(self, tmp_path: Path) -> None:
        _build(tmp_path, GAINS)
        with pytest.raises(ValueError, match="T9\\(a\\)"):
            tz.run(_args(tmp_path))

    def test_runs_on_different_labels_stop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _build(tmp_path, GAINS)
        _pin_t0(monkeypatch, tmp_path)
        bad = tmp_path / constants.V2_TADZ_ARM / f"s{SEEDS[0]}" / tz.SCORES_DIRNAME / "a000.npz"
        with np.load(bad) as data:
            score = data["score"]
        np.savez(bad, score=score, gt=np.ones(FRAMES, dtype=np.float32))
        with pytest.raises(ValueError, match="labels differ"):
            tz.run(_args(tmp_path))

    def test_wrong_protocol_size_stops(self, tmp_path: Path) -> None:
        labels = dict(list(_labels().items())[:-1])
        _build(tmp_path, GAINS, labels)
        with pytest.raises(ValueError, match="TAD protocol"):
            tz.run(_args(tmp_path))

    def test_missing_seed_stops(self, tmp_path: Path) -> None:
        _build(tmp_path, {a: g for a, g in GAINS.items() if a != constants.V2_TADZ_PRINTED_ARM})
        with pytest.raises(FileNotFoundError):
            tz.run(_args(tmp_path))


class TestMetrics:
    def test_all_normal_and_all_abnormal_videos_leave_the_macro(self) -> None:
        labels = {"two": _video_labels(True), "normal": _video_labels(False),
                  "full": np.ones(FRAMES, dtype=np.int64)}
        scores = {v: np.arange(FRAMES, dtype=float) for v in labels}
        assert set(tz.clip_aucs(scores, labels)) == {"two"}

    def test_video_auc_reads_the_max(self) -> None:
        labels = {"a": _video_labels(True), "n": _video_labels(False)}
        scores = {"a": np.zeros(FRAMES), "n": np.zeros(FRAMES)}
        scores["a"][0] = 1.0  # the peak is outside the span: clip-level still separates
        assert tz.video_auc(scores, labels) == 1.0
