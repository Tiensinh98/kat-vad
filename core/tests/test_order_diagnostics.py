"""v2.2 order diagnostics S1-S3 (proposal §5-§6, plan O1-O15).

The silent failures: a gate that drifts from the one bar (§6.1), probe scores left in shuffled slot
order when they are compared with labels and ``p_T2`` (``f_2`` would be read against the wrong
position), S3 running although S1a or S2 failed, an echo that admits Δu, and a clip-start spike
that keeps the early-onset prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import order_diagnostics as od
from core.tools import order_probe as op

DX, DU, N_T2, N_DOTA, L = 16, 8, 40, 33, 20


def _clip(rng: np.random.Generator, n: int, onset: int, span: int, echo: bool = False) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """A slow random walk with a jump at the onset; ``echo`` repeats the jump in ``u`` 5 steps
    after the span ends."""
    y = np.zeros(n, dtype=np.int64)
    y[onset : onset + span] = 1
    x = 0.1 * rng.normal(size=(n, DX)).cumsum(axis=0)
    u = 0.1 * rng.normal(size=(n, DU)).cumsum(axis=0)
    x[onset:] += 3.0 * rng.normal(size=DX)
    u[onset:] += 3.0 * rng.normal(size=DU)
    if echo and onset + span + 5 < n:
        u[onset + span + 5 :] += 3.0 * rng.normal(size=DU)
    return x, u, y


def _corpus(prefix: str, count: int, n: int, seed: int, echo: bool = False) -> od.Corpus:
    rng = np.random.default_rng(seed)
    x, u, y, group, ego = {}, {}, {}, {}, {}
    for i in range(count):
        v = f"{prefix}{i // 3:02d}_{i:04d}"
        x[v], u[v], y[v] = _clip(rng, n, int(rng.integers(6, n - 16)), 6, echo)
        group[v], ego[v] = v.rpartition("_")[0], i % 2 == 0
    return od.Corpus(x, u, y, group, ego)


def _data(echo: bool = False) -> od.Data:
    train = _corpus("t", 30, N_T2, 1)
    windows = [(v, s, s + L) for v in train.x for s in range(0, N_T2 - L + 1, 8)]
    dota = _corpus("d", 24, N_DOTA, 3, echo)
    tau = (np.arange(N_DOTA) + 0.5) / N_DOTA
    prior = dict.fromkeys(dota.y, -(tau - 0.5) ** 2)
    return od.Data(train, windows, _corpus("v", 9, N_T2, 2), dota, prior)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(resamples=200, seed=0, seeds=[1, 2], device="cpu", out_dir=tmp_path)


@pytest.fixture(autouse=True)
def _fast_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(constants, "V22_PROBE_EPOCHS", 2)


class TestBar:
    @pytest.mark.parametrize(("mean", "low", "ok"), [
        (0.02, 0.001, True), (0.02, 0.0, False), (0.009, 0.001, False), (0.01, 0.0001, True),
    ])
    def test_one_bar_low_above_zero_and_mean_at_least_one_point(
        self, mean: float, low: float, ok: bool
    ) -> None:
        assert od.passes({"mean": mean, "low": low, "high": mean + 0.01}) is ok

    def test_no_interval_never_passes(self) -> None:
        assert od.passes(None) is False


class TestWindows:
    def _dir(self, tmp_path: Path) -> Path:
        meta = {"w1": {"source": "s", "start": 0, "end": 4}, "w2": {"source": "s", "start": 2,
                                                                     "end": 6}}
        (tmp_path / constants.META_FILENAME).write_text(json.dumps(meta))
        (tmp_path / constants.LABELS_TRAIN_FILENAME).write_text(json.dumps({"w1": 0, "w2": 1}))
        return tmp_path

    def test_windows_come_from_meta(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        y = {"s": np.array([0, 0, 0, 1, 1, 1])}
        monkeypatch.setattr(od, "window_frame_labels",
                            lambda e, _w: list(y["s"][e["start"]:e["end"]]))
        assert od.training_windows(self._dir(tmp_path), y) == [("s", 0, 4), ("s", 2, 6)]

    def test_a_window_whose_labels_differ_from_its_source_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(od, "window_frame_labels", lambda e, _w: [1] * (e["end"] - e["start"]))
        with pytest.raises(ValueError, match="O2"):
            od.training_windows(self._dir(tmp_path), {"s": np.zeros(6, dtype=np.int64)})


class TestS1:
    def test_s1_reads_every_check_and_the_jump_is_ranked(self, tmp_path: Path) -> None:
        r = od.run_s1(_data(), _args(tmp_path))
        a = r["s1a"][od.DOTA]
        assert a["macro"]["m5"]["mean"] > 0.5
        assert r["s1a"]["verdict"] in (constants.V22_GATE_OK, constants.V22_GATE_NO)
        assert a["sequences"]["gap_counts"] == {constants.V22_TWIN_GAP: 24}
        assert set(r["s1c"]["summary_smd"]) == set(od.of.SUMMARY_CHANNELS)
        md = od.render("s1", r)
        assert "S1a" in md and "S1b" in md and "S1c" in md

    def test_an_exit_echo_in_u_is_not_admitted(self, tmp_path: Path) -> None:
        r = od.s1b(_data(echo=True), _args(tmp_path))
        assert r["u"]["delta"]["low"] > 0
        assert r["admit_du"] is False

    def test_a_clip_start_spike_withdraws_the_early_onset_prediction(self) -> None:
        data = _data()
        for x in data.dota.x.values():
            x[1:] += 5.0  # a jump between steps 0 and 1 of every clip
        stats = od.of.fit_kappa_stats(od.of.fit_stats(od.long_rows(data.train.x)),
                                      od.long_rows(data.train.x))
        assert od.s1c(data, stats)["h6_kept"] is False


class TestProbeOrder:
    def test_shuffled_scores_are_mapped_back_to_original_steps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fake probe that scores each slot by its original index must come back as arange."""

        def fake(train: object, test: dict[str, op.Sequence], *_a: object) -> tuple[
                dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
            scores = {v: seq.index.astype(np.float64) for v, seq in test.items()}
            return scores, {1: scores}

        monkeypatch.setattr(od.op, "seed_averaged_scores", fake)
        data = _data()
        scores, _per_seed, unshuffled = od.probe_scores(
            data, od.f0_bases(data, clip_only=False), True, True, _args(tmp_path))
        assert unshuffled == 0
        for v in data.dota.y:
            assert np.array_equal(scores[v], np.arange(N_DOTA))

    def test_s2_reads_both_sets_and_decides_on_f0(self, tmp_path: Path) -> None:
        r = od.run_s2(_data(), _args(tmp_path))
        for name in (od.F0, od.CLIP_ONLY):
            assert r[name][od.DOTA]["delta"] is not None
            assert r[name][od.DOTA]["f2_delta"] is not None
            assert set(r[name]["per_seed_dota_delta"]) == {1, 2}
        assert r["verdict"] == (constants.V22_GATE_OK if od.passes(r[od.F0][od.DOTA]["delta"])
                                else constants.V22_GATE_NO)
        assert "decides" in od.render("s2", r)


class TestS3:
    def _write(self, out: Path, s1a: str, s2: str, admit: bool = False) -> None:
        (out / od.readout_json("s1")).write_text(
            json.dumps({"s1a": {"verdict": s1a}, "s1b": {"admit_du": admit}}))
        (out / od.readout_json("s2")).write_text(json.dumps({"verdict": s2}))

    @pytest.mark.parametrize(("s1a", "s2"), [
        (constants.V22_GATE_NO, constants.V22_GATE_OK),
        (constants.V22_GATE_OK, constants.V22_GATE_NO),
    ])
    def test_s3_refuses_unless_s1a_and_s2_passed(self, tmp_path: Path, s1a: str, s2: str) -> None:
        self._write(tmp_path, s1a, s2)
        with pytest.raises(RuntimeError, match="S3 is not run"):
            od.run_s3(_data(), _args(tmp_path))

    @pytest.mark.parametrize("admit", [False, True])
    def test_s3_runs_value_and_order(self, tmp_path: Path, admit: bool) -> None:
        self._write(tmp_path, constants.V22_GATE_OK, constants.V22_GATE_OK, admit)
        r = od.run_s3(_data(), _args(tmp_path))
        pcs = min(constants.V22_PCA_DIM, DX)
        width = 2 * pcs + 3 + (2 * min(constants.V22_PCA_DIM, DU) + 3 if admit else 0)
        assert r["kappa_dim"] == width
        assert set(r["verdict"]) == {"value", "order", "s5_eligible"}
        assert 0.0 <= r["shortcut_auc_kappa"] <= 1.0
        assert "VALUE" in od.render("s3", r)
