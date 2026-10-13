"""Tests for core.tools.v22_gates: ΔT sets, the detrended read, rate groups, demotion, r0-saved."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import v22_gates as g
from core.tools.protocol_b_eval import write_clip_scores

DT = constants.V22_DOTA_STEP_SECONDS


def _labels(n: int, start: int, end: int) -> np.ndarray:
    y = np.zeros(n, dtype=np.int64)
    y[start:end] = 1
    return y


class TestOnsetSets:
    def test_windows_at_dota_step(self) -> None:
        pos, neg = g.onset_sets(_labels(40, 20, 35), DT)  # type: ignore[misc]
        # positives: tau in [6.0, 8.0) -> steps 20..26; negatives: tau in [3.0, 5.5) -> steps 10..18
        assert pos.tolist() == list(range(20, 27))
        assert neg.tolist() == list(range(10, 19))

    def test_short_span_caps_positives(self) -> None:
        pos, _ = g.onset_sets(_labels(40, 20, 22), DT)  # type: ignore[misc]
        assert pos.tolist() == [20, 21]

    @pytest.mark.parametrize("y", [_labels(40, 2, 30), _labels(40, 20, 21), np.zeros(40),
                                   np.ones(40)])
    def test_ineligible(self, y: np.ndarray) -> None:
        assert g.onset_sets(y, DT) is None

    def test_negatives_skip_abnormal_steps(self) -> None:
        y = _labels(40, 20, 35)
        y[12] = 1  # an earlier abnormal step moves t_s to step 12
        sets = g.onset_sets(y, DT)
        assert sets is None or 12 not in sets[1]


class TestOnsetAucs:
    def test_rising_ramp_wins(self) -> None:
        """The R0-saved finding: any score rising through the onset scores ΔT = 1."""
        y = {"a": _labels(40, 20, 35)}
        ramp = {"a": np.arange(40, dtype=np.float64)}
        assert g.onset_aucs(ramp, y, DT)["a"] == pytest.approx(1.0)

    def test_detrended_ramp_is_chance(self) -> None:
        y = {"a": _labels(40, 20, 35)}
        ramp = {"a": 0.3 * np.arange(40, dtype=np.float64) + 2.0}
        assert g.detrended_onset_aucs(ramp, y, DT)["a"] == pytest.approx(0.5)

    def test_detrended_step_wins(self) -> None:
        y = {"a": _labels(40, 20, 35)}
        step = {"a": np.where(np.arange(40) >= 20, 1.0, 0.0)}
        assert g.detrended_onset_aucs(step, y, DT)["a"] == pytest.approx(1.0)

    def test_ineligible_clips_dropped(self) -> None:
        y = {"a": _labels(40, 20, 35), "b": _labels(40, 1, 30)}
        s = {k: np.arange(40, dtype=np.float64) for k in y}
        assert set(g.onset_aucs(s, y, DT)) == {"a"}


class TestHelpers:
    def test_step_grid_reads_every_third(self) -> None:
        assert g.step_grid({"a": np.arange(10)}, 3)["a"].tolist() == [0, 3, 6, 9]

    def test_rate_groups(self) -> None:
        align = {"clips": {"a": {"rate": 1.0}, "b": {"rate": 2.98}, "c": {"rate": 2.0}}}
        assert g.rate_groups(align, ["a", "b", "c"]) == {"a": "r1", "b": "r3", "c": g.OTHER_GROUP}

    def test_causal_stats_shift(self) -> None:
        raw = np.arange(8, dtype=np.float64).reshape(4, 2)
        out = g.causal_stats(raw)
        assert out[0].tolist() == raw[0].tolist()
        assert out[1:].tolist() == raw[:-1].tolist()

    @pytest.mark.parametrize(
        ("eligible", "a3ign", "f0", "demoted"),
        [(557, 0.85, 0.80, False), (150, 0.85, 0.80, True), (557, 0.91, None, True),
         (557, 0.85, 0.95, True)],
    )
    def test_demotion(self, eligible: int, a3ign: float, f0: float | None, demoted: bool) -> None:
        assert g.demotion(eligible, a3ign, f0)["demoted"] is demoted

    def test_parse_scores_requires_a3ign(self) -> None:
        with pytest.raises(ValueError, match="A3ign"):
            g.parse_scores(["A3=x.npz"])
        with pytest.raises(ValueError, match="ARM=path"):
            g.parse_scores(["A3ign"])


def test_r0_saved_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = [f"vid{i:02d}_{100 + i:06d}" for i in range(12)]
    monkeypatch.setattr(g, "load_split", lambda name, split_dir: ids)
    rng = np.random.default_rng(0)
    labels = {v: np.repeat(_labels(40, 20, 35), 3)[:118] for v in ids}  # native frames
    prior = {v: np.linspace(0.0, 1.0, 118) for v in ids}
    arm = {v: rng.normal(size=118) + labels[v] for v in ids}
    write_clip_scores(tmp_path / "prior.npz", prior, labels)
    write_clip_scores(tmp_path / "arm.npz", arm, labels)
    align = {"clips": {v: {"rate": 1.0 if i % 2 else 3.0} for i, v in enumerate(ids)}}
    (tmp_path / "align.json").write_text(json.dumps(align))
    args = argparse.Namespace(
        step=g.R0_SAVED, scores=[f"A3ign={tmp_path / 'arm.npz'}"], prior=tmp_path / "prior.npz",
        alignment=tmp_path / "align.json", split_dir=tmp_path, seed=0, resamples=200,
        out_dir=tmp_path / "out",
    )
    r = g.run(args)
    assert r["eligible"] == len(ids)
    assert r["reads"][g.PRIOR]["onset"]["mean"] == pytest.approx(1.0)  # the ramp wins ΔT
    assert set(r["reads"][g.A3IGN]["onset_by_rate"]) == {"r1", "r3"}
    assert (tmp_path / "out" / g.readout_md(g.R0_SAVED)).is_file()


class TestColabSteps:
    """r0-probe and k1 on the synthetic corpus of test_order_diagnostics (jump at the onset)."""

    @pytest.fixture(autouse=True)
    def _fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(constants, "V22_PROBE_EPOCHS", 2)

    def _args(self, tmp_path: Path) -> argparse.Namespace:
        from core.tests.test_order_diagnostics import _args, _data

        args = _args(tmp_path)
        data = _data()
        align = {"clips": {v: {"rate": 1.0} for v in data.dota.y}}
        (tmp_path / "align.json").write_text(json.dumps(align))
        args.alignment = tmp_path / "align.json"
        args.data = data
        return args

    def test_r0_probe_reads_every_score_and_completes_with_saved(self, tmp_path: Path) -> None:
        args = self._args(tmp_path)
        saved = {"reads": {g.A3IGN: {"onset": {"mean": 0.85, "low": 0.8, "high": 0.9}}}}
        (tmp_path / g.readout_json(g.R0_SAVED)).write_text(json.dumps(saved))
        r = g.run_r0_probe(args.data, args)
        assert set(r["reads_dota"]) == {"m5", "s", g.F0, g.F0_K, g.P_ONLY}
        assert r["demotion"]["complete"]
        assert r["reads_dota"]["m5"]["onset"]["mean"] > 0.5  # the jump sits at the onset

    def test_k1_reads_t2_val_only_with_causal_stats(self, tmp_path: Path) -> None:
        args = self._args(tmp_path)
        data = args.data
        flow = tmp_path / "flow"
        flow.mkdir()
        rng = np.random.default_rng(0)
        for corpus in (data.train, data.val):
            for v, x in corpus.x.items():
                np.save(flow / f"{v}{constants.FLOW_STATS_SUFFIX}",
                        rng.normal(size=(len(x), constants.FLOW_STATS_DIM)))
        args.flow_dir = flow
        r = g.run_k1(data, args)
        assert r["span"]["delta"] is not None and r["onset"]["delta"] is not None
        assert isinstance(r["soft_stop"], bool)
        assert "K1" in g.render_k1(r)

    def test_k1_refuses_misaligned_stats(self, tmp_path: Path) -> None:
        rows = np.zeros((5, constants.FLOW_STATS_DIM))
        np.save(tmp_path / f"a{constants.FLOW_STATS_SUFFIX}", rows)
        with pytest.raises(ValueError, match="CLIP rows"):
            g.load_stats(tmp_path, ["a"], {"a": 6})
