"""Tests for kill-switch K on T2 (addendum §6.1, plan P1).

What must not happen silently: a T2-val source in K's subset, a verdict that is
not the pre-registered rule, a probe over misaligned rows, and a motion stream
killed because the pipeline was broken (the positive control).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.data.dada_origin import OriginRow
from core.tools import kill_switch_probe as ksp

N_SOURCES = 40
N_TYPES = 8
STEPS = 30
DIM_X, DIM_U = 12, 10
SEED = 3
FAST_BOOT = 200


def _t2_meta(train_per_type: int = 30) -> dict[str, dict[str, object]]:
    meta: dict[str, dict[str, object]] = {}
    for type_id in range(1, N_TYPES + 1):
        for video in range(train_per_type + 2):
            source = f"t{type_id:02d}_v{video:03d}"
            split = "train" if video < train_per_type else "test"
            for w, positive in enumerate((4, 0)):
                meta[f"{source}__w{w:03d}"] = {"source": source, "split": split,
                                               "type": type_id, "positive_frames": positive}
    return meta


def _corpus(
    u_mode: str, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Two-class sources; ``x`` carries the label; ``u`` per ``u_mode``."""
    labels: dict[str, np.ndarray] = {}
    types: dict[str, int] = {}
    x: dict[str, np.ndarray] = {}
    u: dict[str, np.ndarray] = {}
    mixing = rng.normal(size=(DIM_X, DIM_U))
    for i in range(N_SOURCES):
        vid = f"t{i % N_TYPES + 1:02d}_v{i:03d}"
        start = int(rng.integers(5, 20))
        y = np.zeros(STEPS, dtype=np.int64)
        y[start:start + 8] = 1
        labels[vid], types[vid] = y, i % N_TYPES + 1
        x[vid] = rng.normal(size=(STEPS, DIM_X)) + 0.8 * y[:, None]
        if u_mode == "redundant":  # u is a view of x: signal, but nothing new
            u[vid] = x[vid] @ mixing
        elif u_mode == "extra":  # u sees a second, independent copy of the label
            u[vid] = rng.normal(size=(STEPS, DIM_U)) + 1.5 * y[:, None]
        else:
            u[vid] = rng.normal(size=(STEPS, DIM_U))
    return labels, types, x, u


class TestPick:
    def test_excludes_t2_val_and_is_deterministic(self) -> None:
        meta = _t2_meta()
        val = {f"t{t:02d}_v{v:03d}" for t in range(1, N_TYPES + 1) for v in range(5)}
        ids, block = ksp.pick_sources(meta, val, seed=SEED, target=60)
        assert not set(ids) & val
        assert block["picked"] == len(ids) and 50 <= len(ids) <= 70
        assert ksp.pick_sources(meta, val, seed=SEED, target=60)[0] == ids
        assert all(i.endswith(tuple(f"v{v:03d}" for v in range(5, 30))) for i in ids)


class TestDecide:
    @pytest.mark.parametrize(
        ("delta_a", "delta_b", "control_low", "expected"),
        [
            ((0.0, -0.02, 0.02), (-0.01, -0.03, 0.01), 0.6, ksp.VERDICT_KILL),
            ((0.0, -0.02, 0.03), (-0.01, -0.03, 0.01), 0.6, ksp.VERDICT_GO),  # upper == bar
            ((0.01, -0.02, 0.02), (0.005, -0.03, 0.01), 0.6, ksp.VERDICT_KILL),  # point > 0 ok
            ((0.01, -0.02, 0.02), (0.02, 0.01, 0.04), 0.6, ksp.VERDICT_GO),  # one upper >= bar
            ((-0.01, -0.02, 0.0), (-0.01, -0.03, 0.01), 0.5, ksp.VERDICT_SUSPECT),
            ((0.05, 0.01, 0.09), (0.04, 0.0, 0.08), 0.49, ksp.VERDICT_SUSPECT),
        ],
    )
    def test_rule_is_the_pre_registered_one(
        self, delta_a: tuple[float, float, float], delta_b: tuple[float, float, float],
        control_low: float, expected: str,
    ) -> None:
        def ci(t: tuple[float, float, float]) -> dict[str, float]:
            return {"mean": t[0], "low": t[1], "high": t[2]}

        deltas = {"i_source": ci(delta_a), "ii_type": ci(delta_b)}
        control = {"mean": control_low + 0.05, "low": control_low, "high": 0.9}
        assert ksp.decide(deltas, control) == expected


class TestRunK:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_redundant_u_is_killed_on_every_seed(self, seed: int) -> None:
        """A u that is a view of x: signal alone, nothing added (|Δ| ~ 1e-4).

        Under the P0 rule's extra ``point <= 0`` leg this was a sign coin-flip
        (KILL on 1 of seeds 0-3, pending lesson (ae)); the amended rule
        (addendum §6.1, option A) must KILL it on every seed.
        """
        readout = ksp.run_k(*_corpus("redundant", np.random.default_rng(seed)),
                            seed=seed, resamples=FAST_BOOT)
        assert readout["arms"]["i_source"]["u"]["low"] > constants.V2_K_CONTROL_FLOOR
        assert all(abs(d["mean"]) < 0.01 for d in readout["delta_xu_minus_x"].values())
        assert readout["verdict"] == ksp.VERDICT_KILL

    def test_informative_u_goes(self) -> None:
        readout = ksp.run_k(*_corpus("extra", np.random.default_rng(SEED)),
                            seed=SEED, resamples=FAST_BOOT)
        assert readout["delta_xu_minus_x"]["i_source"]["low"] > 0
        assert readout["verdict"] == ksp.VERDICT_GO

    def test_signal_free_u_is_a_suspect_pipeline_not_a_kill(self) -> None:
        readout = ksp.run_k(*_corpus("noise", np.random.default_rng(SEED)),
                            seed=SEED, resamples=FAST_BOOT)
        assert readout["verdict"] == ksp.VERDICT_SUSPECT

    def test_misaligned_rows_raise(self) -> None:
        labels, types, x, u = _corpus("noise", np.random.default_rng(SEED))
        first = next(iter(u))
        u[first] = u[first][:-1]
        with pytest.raises(ValueError, match="misaligned"):
            ksp.run_k(labels, types, x, u, resamples=FAST_BOOT)

    def test_markdown_carries_the_verdict_and_provenance(self) -> None:
        readout = ksp.run_k(*_corpus("extra", np.random.default_rng(SEED)),
                            seed=SEED, resamples=FAST_BOOT)
        readout.update({"commit": "abc123", "ids_sha1": "f00", "video_manifest": {"stride": 8}})
        text = ksp.render_markdown(readout)
        assert "**GO**" in text and "abc123" in text and "NOT a domain transfer" in text


class TestSourceLabels:
    def test_labels_follow_the_annotation_span_at_stride(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        row = OriginRow(type_id=3, video=7, start=40, accident=50, end=80,
                        total_frames=160, attributes={})
        monkeypatch.setattr(ksp, "parse_annotation", lambda _path: [row])
        labels, types = ksp.source_labels(tmp_path / "a.xlsx", {"t03_v007": 160},
                                          ["t03_v007"], stride=8)
        assert types == {"t03_v007": 3}
        assert labels["t03_v007"].tolist() == [0] * 5 + [1] * 5 + [0] * 10

    def test_missing_census_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(ksp, "parse_annotation", lambda _path: [])
        with pytest.raises(ValueError, match="lack an annotation"):
            ksp.source_labels(tmp_path / "a.xlsx", {}, ["t01_v001"], stride=8)


def _diag_corpus(
    u_mode: str, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Mid-video anomalies after the padded steps; ``x`` weakly carries the label.

    ``u_mode``: ``pad`` = u only flags the padded leading steps (always normal);
    ``position`` = u is a noisy view of relative position, and the anomaly sits at a
    near-fixed relative position (so position explains it); ``extra`` = u sees the label.
    """
    n_pad = ksp.padded_steps(constants.FRAME_STRIDE)
    labels: dict[str, np.ndarray] = {}
    types: dict[str, int] = {}
    x: dict[str, np.ndarray] = {}
    u: dict[str, np.ndarray] = {}
    mixing = rng.normal(size=(constants.V2_K_POSITION_DEGREE, DIM_U))
    for i in range(N_SOURCES):
        vid = f"t{i % N_TYPES + 1:02d}_v{i:03d}"
        low, high = (n_pad + 6, n_pad + 10) if u_mode == "position" else (n_pad, STEPS - 8)
        start = int(rng.integers(low, high))
        y = np.zeros(STEPS, dtype=np.int64)
        y[start:start + 8] = 1
        labels[vid], types[vid] = y, i % N_TYPES + 1
        x[vid] = rng.normal(size=(STEPS, DIM_X)) + 0.5 * y[:, None]
        noise = rng.normal(size=(STEPS, DIM_U))
        if u_mode == "pad":
            flag = (np.arange(STEPS) < n_pad).astype(np.float64)
            u[vid] = noise + 3.0 * flag[:, None]
        elif u_mode == "position":
            u[vid] = ksp.position_features(STEPS) @ mixing + 0.01 * noise
        else:
            u[vid] = noise + 1.5 * y[:, None]
    return labels, types, x, u


class TestDiag:
    @pytest.mark.parametrize(("stride", "expected"), [(8, 6), (3, 15), (45, 1), (46, 1)])
    def test_padded_steps_match_the_extractor_clamp(self, stride: int, expected: int) -> None:
        assert ksp.padded_steps(stride) == expected
        from core.tools.extract_video_features import causal_clip_indices
        clamped = [i for i in range(expected + 3)
                   if causal_clip_indices(stride * i)[0] == 0 and stride * i < 45]
        assert len(clamped) == expected

    def test_position_features_are_relative(self) -> None:
        p = ksp.position_features(10)
        assert p.shape == (10, constants.V2_K_POSITION_DEGREE)
        np.testing.assert_allclose(p[:, 0], (np.arange(10) + 0.5) / 10)
        np.testing.assert_allclose(p[:, 2], p[:, 0] ** 3)

    def test_pad_only_u_reads_pad_explains(self) -> None:
        diag = ksp.run_diag(*_diag_corpus("pad", np.random.default_rng(SEED)),
                            seed=SEED, resamples=FAST_BOOT)
        assert diag["pad"]["steps_dropped_per_source"] == 6
        assert diag["pad"]["reading"] == ksp.READ_PAD_EXPLAINS

    def test_position_only_u_reads_position_proxy(self) -> None:
        diag = ksp.run_diag(*_diag_corpus("position", np.random.default_rng(SEED)),
                            seed=SEED, resamples=FAST_BOOT)
        assert diag["position"]["reading"] == ksp.READ_POSITION_PROXY

    def test_informative_u_survives_both(self) -> None:
        diag = ksp.run_diag(*_diag_corpus("extra", np.random.default_rng(SEED)),
                            seed=SEED, resamples=FAST_BOOT)
        assert diag["pad"]["reading"] == ksp.READ_NOT_PAD
        assert diag["position"]["reading"] == ksp.READ_BEYOND_POSITION
        text = ksp.render_diag_markdown(diag)
        assert "not gated" in text and "BEYOND_POSITION" in text

    def test_run_k_is_unchanged_by_the_refactor(self) -> None:
        corpus = _corpus("extra", np.random.default_rng(SEED))
        readout = ksp.run_k(*corpus, seed=SEED, resamples=FAST_BOOT)
        assert set(readout["arms"]["i_source"]) == set(ksp.FEATURE_SETS)
        assert set(readout["delta_xu_minus_x"]) == set(ksp.PROBES)

    def test_cli_parses_diag_without_commit(self) -> None:
        args = ksp.build_arg_parser().parse_args(
            ["diag", "--ids-file", "i", "--annotation", "a", "--census", "c",
             "--clip-dir", "x", "--video-dir", "v", "--out-dir", "o"])
        assert args.command == "diag" and args.commit == "n/a"
