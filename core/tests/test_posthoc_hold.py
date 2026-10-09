"""H2(a) post-hoc hold: causal, its length from T2 alone, and its rule as pre-registered.

The failures that matter are silent: a hold that peeks at future frames (it would score a
streaming-impossible curve), a hold length that leaks T2-val or DoTA, a verdict that ignores one
of G1-G3, or a read-out that runs on scores the E3 dev gate did not reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.tools import posthoc_hold as ph
from core.tools import protocol_b_eval as pb
from core.tools.e3_readout import PB_DIR, PRIOR_DIR, REPORTS
from core.tools.final_readout import CAP_DEV
from core.tools.position_prior import PRIOR_NPZ
from core.tools.position_prior import READOUT_JSON as PRIOR_READOUT_JSON

FRAMES = 40
ONSET = 10


def _window(source: str, split: str, span: tuple[int, int]) -> dict[str, Any]:
    return {"source": source, "split": split, "annotation_span_frames": list(span)}


def _fading_clip(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Label abnormal from ONSET to the end; score spikes at onset, then falls back to noise."""
    labels = (np.arange(FRAMES) >= ONSET).astype(np.int64)
    scores = rng.random(FRAMES) * 0.5
    scores[ONSET:ONSET + 3] = 1.0
    return scores, labels


class TestHolds:
    def test_max_hold_is_causal(self) -> None:
        x = np.array([0.1, 0.9, 0.2, 0.3, 0.1])
        np.testing.assert_allclose(ph.max_hold(x, 2), [0.1, 0.9, 0.9, 0.3, 0.3])

    def test_max_hold_never_reads_the_future(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.random(30)
        y = x.copy()
        y[20:] = 10.0
        np.testing.assert_array_equal(ph.max_hold(x, 5)[:20], ph.max_hold(y, 5)[:20])
        np.testing.assert_array_equal(ph.ema(x, 5)[:20], ph.ema(y, 5)[:20])

    def test_window_one_is_the_identity(self) -> None:
        x = np.array([0.3, 0.1, 0.7])
        np.testing.assert_array_equal(ph.max_hold(x, 1), x)
        np.testing.assert_array_equal(ph.ema(x, 1), x)

    def test_ema_uses_the_span_alpha(self) -> None:
        out = ph.ema(np.array([0.0, 1.0]), 3)
        np.testing.assert_allclose(out, [0.0, 0.5])

    def test_hold_keeps_the_length(self) -> None:
        x = np.arange(7, dtype=np.float64)
        for fn in ph.HOLDS.values():
            assert fn(x, 4).shape == x.shape


class TestHoldLength:
    def test_median_of_train_sources_only_in_dota_frames(self) -> None:
        fps = constants.DADA_ASSUMED_FPS
        meta = {
            "a__w0": _window("a", "train", (0, 2 * fps)),
            "a__w1": _window("a", "train", (0, 2 * fps)),  # same source counted once
            "b__w0": _window("b", "train", (0, 4 * fps)),
            "c__w0": _window("c", "train", (0, 6 * fps)),
            "v__w0": _window("v", "train", (0, 100 * fps)),  # T2-val source: excluded
            "t__w0": _window("t", "test", (0, 100 * fps)),  # T2-test: excluded
        }
        w = ph.hold_length(meta, {"v"})
        assert w["sources"] == 3
        assert w["median_s"] == pytest.approx(4.0)
        assert w["frames"] == 4 * constants.DOTA_FPS

    def test_windows_of_one_source_must_agree(self) -> None:
        meta = {"a__w0": _window("a", "train", (0, 30)), "a__w1": _window("a", "train", (0, 60))}
        with pytest.raises(ValueError, match="disagree"):
            ph.hold_length(meta, set())

    def test_no_train_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no T2-train"):
            ph.hold_length({"v__w0": _window("v", "train", (0, 30))}, {"v"})

    def test_the_frozen_t2_val_split_is_the_one_removed(self) -> None:
        from core.data.v2_splits import load_split
        val = load_split(constants.V2_SPLIT_T2_VAL)
        assert len(val) > 0


class TestTailsAndRule:
    def test_tail_clips_need_w_normal_frames_after_the_span(self) -> None:
        labels = {
            "x_000001": np.array([0, 1, 1, 0, 0, 0]),  # 3 normal after
            "x_000002": np.array([0, 1, 1, 1, 0, 0]),  # 2 normal after
            "x_000003": np.ones(6, dtype=np.int64),  # one-class: never a tail clip
        }
        assert ph.tail_clips(labels, 3) == {"x_000001"}

    @pytest.mark.parametrize(
        ("long_low", "f2_low", "all_mean", "verdict"),
        [
            (0.01, 0.01, 0.0, "GO"),
            (-0.01, 0.01, 0.01, "KILL"),
            (0.01, -0.01, 0.01, "KILL"),
            (0.01, 0.01, -0.001, "KILL"),
        ],
    )
    def test_go_needs_all_three_gates(
        self, long_low: float, f2_low: float, all_mean: float, verdict: str
    ) -> None:
        read = {
            "delta_long": {"mean": 0.1, "low": long_low, "high": 0.2},
            "delta_f2": {"mean": 0.1, "low": f2_low, "high": 0.2},
            "delta_all": {"mean": all_mean, "low": -1.0, "high": 1.0},
        }
        assert ph.decide(read)["verdict"] == verdict

    def test_an_empty_long_bin_is_a_kill(self) -> None:
        read = {"delta_long": None, "delta_f2": {"mean": 1, "low": 1, "high": 1},
                "delta_all": {"mean": 1, "low": 1, "high": 1}}
        assert ph.decide(read)["verdict"] == "KILL"


class TestReadout:
    def _corpus(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        rng = np.random.default_rng(1)
        scores: dict[str, np.ndarray] = {}
        labels: dict[str, np.ndarray] = {}
        for i in range(24):
            v = f"src{i // 2}_{i:06d}"
            scores[v], labels[v] = _fading_clip(rng)
        prior = {v: rng.random(FRAMES) for v in scores}
        return scores, labels, prior

    def test_a_hold_recovers_a_fading_score_on_long_spans(self) -> None:
        scores, labels, prior = self._corpus()
        r = ph.readout({constants.H2A_ARM: scores}, prior, labels, 10, constants.SEED)
        a = r["variants"][constants.H2A_PRIMARY][constants.H2A_ARM]
        assert a["macro_held"] > a["macro_raw"]
        assert a["n_long"] == 24  # share 0.75 everywhere
        assert r["decision"]["verdict"] == "GO"

    def test_every_variant_and_arm_is_read(self) -> None:
        scores, labels, prior = self._corpus()
        r = ph.readout({"A0": scores, "A3": scores}, prior, labels, 4, constants.SEED)
        assert set(r["variants"]) == set(constants.H2A_VARIANTS)
        for arms in r["variants"].values():
            assert set(arms) == {"A0", "A3"}
        assert "p_T2" in ph.render_markdown({**r, "plan": "p", "set": "s", "g0_dev_gate": True,
                                             "hold": {"frames": 4, "median_s": 0.4, "sources": 1}})


class TestRun:
    def _layout(self, root: Path) -> None:
        rng = np.random.default_rng(2)
        scores: dict[str, np.ndarray] = {}
        labels: dict[str, np.ndarray] = {}
        for i in range(12):
            v = f"src{i // 2}_{i:06d}"
            scores[v], labels[v] = _fading_clip(rng)
        for arm in constants.V2_E3_ARMS:
            d = root / REPORTS / PB_DIR / arm / CAP_DEV
            d.mkdir(parents=True)
            pb.write_clip_scores(d / pb.CLIP_SCORES_NPZ, scores, labels)
            (d / pb.READOUT_JSON).write_text("{}", encoding="utf-8")
            (d / pb.CLIP_AUCS_JSON).write_text("{}", encoding="utf-8")
        d = root / REPORTS / PRIOR_DIR / CAP_DEV
        d.mkdir(parents=True)
        pb.write_clip_scores(d / PRIOR_NPZ, {v: rng.random(FRAMES) for v in scores}, labels)
        (d / PRIOR_READOUT_JSON).write_text("{}", encoding="utf-8")
        meta = {"a__w0": _window("a", "train", (0, 3 * constants.DADA_ASSUMED_FPS))}
        (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def _args(self, root: Path) -> Any:
        return ph.build_arg_parser().parse_args(
            ["--e3-dir", str(root), "--t2-meta", str(root / "meta.json")]
        )

    def test_a_failed_dev_gate_reads_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ph, "dev_gate", lambda *_: {"pass": False, "failed": [{}]})
        with pytest.raises(SystemExit, match="G0"):
            ph.run(self._args(tmp_path))
        assert not (tmp_path / REPORTS / ph.OUT_DIR).exists()

    def test_end_to_end_writes_the_readout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._layout(tmp_path)
        monkeypatch.setattr(ph, "dev_gate", lambda *_: {"pass": True, "failed": []})
        r = ph.run(self._args(tmp_path))
        out = tmp_path / REPORTS / ph.OUT_DIR
        assert json.loads((out / ph.READOUT_JSON).read_text())["decision"] == r["decision"]
        assert "Verdict" in (out / ph.READOUT_MD).read_text()
        assert r["hold"]["frames"] == 3 * constants.DOTA_FPS
