"""Tests for core.tools.position_shift (DF8 / SC-1, v2 addendum §24)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from core import constants
from core.data.v2_inputs import V2Stats
from core.tools import position_shift as ps
from core.tools.protocol_b_eval import write_clip_scores

EARLY, LATE, CONTROL, FULL = (constants.V22_DF8_EARLY, constants.V22_DF8_LATE,
                              constants.V22_DF8_CONTROL, constants.V22_DF8_FULL)


def _labels(n: int, s: int, e: int) -> np.ndarray:
    y = np.zeros(n, dtype=np.int64)
    y[s:e] = 1
    return y


class TestGeometry:
    def test_four_arms(self) -> None:
        y = _labels(100, 40, 70)  # pre 40, post 30 -> m 33, L 63
        arms = ps.crop_geometry(y)
        assert arms is not None
        assert arms[FULL] == ps.Crop(0, 100, 40, 30)
        assert arms[EARLY] == ps.Crop(37, 63, 3, 30)
        assert arms[LATE] == ps.Crop(10, 63, 30, 30)
        assert arms[CONTROL] == ps.Crop(21, 63, 19, 30)  # round(33 * 40 / 70) = 19
        assert arms[EARLY].centre < arms[CONTROL].centre < arms[LATE].centre

    @pytest.mark.parametrize("arm", list(constants.V22_DF8_ARMS))
    def test_crop_labels_are_the_label_slice(self, arm: str) -> None:
        y = _labels(100, 40, 70)
        arms = ps.crop_geometry(y)
        assert arms is not None
        c = arms[arm]
        assert ps.crop_labels(c).tolist() == y[c.c0 : c.c0 + c.length].tolist()

    @pytest.mark.parametrize("y", [_labels(100, 5, 70), _labels(100, 40, 95), np.zeros(50),
                                   np.ones(50)])
    def test_ineligible(self, y: np.ndarray) -> None:
        assert ps.crop_geometry(y) is None

    def test_non_contiguous_span_raises(self) -> None:
        y = _labels(100, 40, 70)
        y[50] = 0
        with pytest.raises(ValueError, match="contiguous"):
            ps.crop_geometry(y)

    def test_build_table_counts(self) -> None:
        labels = {"a": _labels(100, 40, 70), "b": _labels(100, 2, 70), "c": np.zeros(30)}
        t = ps.build_table(labels)
        assert set(t["clips"]) == {"a"} and t["excluded"] == ["b"] and t["one_class"] == ["c"]


class TestHeads:
    def test_head_rows(self) -> None:
        assert ps.head_rows(ps.Crop(37, 63, 3, 30)) == [0, 3, 6, 9, 12]
        assert ps.head_rows(ps.Crop(0, 100, 40, 30)) == []
        assert ps.head_rows(ps.Crop(5, 7, 3, 2)) == [0, 3, 6]

    def test_windows_clamp_at_crop_start(self) -> None:
        crop = ps.Crop(37, 63, 3, 30)
        w = ps.head_windows(crop, [0, 12])
        assert w[0] == [37] * 16
        assert w[1] == [37] * 4 + list(range(38, 50))
        assert all(len(x) == constants.VIDEOMAE_CLIP_FRAMES for x in w)

    def test_clip_windows_and_gate(self) -> None:
        arms = ps.crop_geometry(_labels(100, 40, 70))
        assert arms is not None
        windows, heads, gates = ps.clip_head_windows(arms, 100)
        assert set(heads) == set(constants.V22_DF8_SHIFTED) == set(gates)
        idx, frame = gates[EARLY]
        assert frame == 37 + ps.REACH and windows[idx] == list(range(frame - 15, frame + 1))
        assert len(windows) == 3 * (5 + 1)

    def test_motion_rows_patch(self) -> None:
        u = np.arange(100, dtype=np.float32).reshape(100, 1)
        crop = ps.Crop(37, 63, 3, 30)
        heads = -np.ones((5, 1), dtype=np.float32)
        out = ps.motion_rows(u, crop, heads)
        assert out[[0, 3, 6, 9, 12], 0].tolist() == [-1.0] * 5
        assert out[1, 0] == 38 and out[15, 0] == 52
        leaky = ps.motion_rows(u, crop, None)
        assert leaky[0, 0] == 37

    def test_load_heads_gate(self, tmp_path: Path) -> None:
        arms = ps.crop_geometry(_labels(100, 40, 70))
        assert arms is not None
        rows = {a: np.zeros((5, 2)) for a in constants.V22_DF8_SHIFTED}
        np.savez(tmp_path / "v.npz", **rows, **{f"{ps.GATE_KEY}{EARLY}": np.asarray([0.5])})
        with pytest.raises(RuntimeError, match="X10"):
            ps.load_heads(tmp_path, "v", arms)
        np.savez(tmp_path / "v.npz", **rows, **{f"{ps.GATE_KEY}{EARLY}": np.asarray([0.9999])})
        assert set(ps.load_heads(tmp_path, "v", arms)) >= set(constants.V22_DF8_SHIFTED)


class TestSentence:
    @pytest.mark.parametrize(
        ("r1", "ni", "outcome"),
        [((0.01, 0.03), True, "R1>0, X7 holds"), ((0.01, 0.03), False, "R1>0, X7 fails"),
         ((-0.03, -0.01), True, "R1<0"), ((-0.01, 0.02), True, "R1 contains 0")],
    )
    def test_outcomes(self, r1: tuple[float, float], ni: bool, outcome: str) -> None:
        got, _ = ps.sentence({"low": r1[0], "high": r1[1], "mean": sum(r1) / 2}, ni)
        assert got == outcome

    def test_span_bins(self) -> None:
        assert [ps.span_bin(n) for n in (10, 30, 50)] == ["<2s", "2-4s", ">=4s"]


# ---------------------------------------------------------------------------
# end to end on a synthetic corpus (fake model: score = first input channel)
# ---------------------------------------------------------------------------
IDS = [f"vid{i:02d}_{100 + i:06d}" for i in range(8)]


def _corpus(tmp_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    labels, x = {}, {}
    for i, v in enumerate(IDS):
        n = 90 + 3 * i
        y = _labels(n, 30 + i, 60 + i)
        labels[v] = y
        x[v] = (rng.normal(size=(n, 4)) + 2.0 * y[:, None]).astype(np.float32)
    (tmp_path / "s1").mkdir()
    (tmp_path / "u").mkdir()
    for v in IDS:
        np.save(tmp_path / "s1" / f"{v}.npy", x[v])
        np.save(tmp_path / "u" / f"{v}.npy", x[v][:, :2].copy())
    return labels, x


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(constants, "V22_DF8_EXPECTED_CLIPS", len(IDS))
    labels, x = _corpus(tmp_path)
    table = ps.build_table(labels)
    (tmp_path / ps.CROPS_JSON).write_text(json.dumps(table))
    return {"labels": labels, "x": x, "crops": tmp_path / ps.CROPS_JSON}


def _fake_model(monkeypatch: pytest.MonkeyPatch, crn: str, motion: str) -> None:
    cfg = SimpleNamespace(v2=SimpleNamespace(crn=crn, motion=motion))
    monkeypatch.setattr(ps, "load_finished_model", lambda run, device: (None, 7, None, cfg))
    monkeypatch.setattr(ps, "load_class_names", lambda data_dir: ["a"])
    monkeypatch.setattr(
        ps, "score_steps", lambda model, feats, fn, names, vid, protocol: feats[:, 0].numpy()
    )


def _score_args(
    tmp_path: Path, corpus: dict[str, Any], model: str, ref: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        crops=corpus["crops"], model=model, run=[["s1", "a.pt"], ["s2", "b.pt"]],
        s1_dir=tmp_path / "s1", video_dir=tmp_path / "u", stats_dir=tmp_path / "stats",
        heads_dir=tmp_path / "heads", data_dir=tmp_path, reference=ref, device="cpu",
        out_dir=tmp_path / "out",
    )


def _reference(tmp_path: Path, corpus: dict[str, Any], name: str) -> Path:
    from core.tools.rate_matched_eval import to_native

    scores = {v: to_native(corpus["x"][v][::3, 0].astype(np.float64), 3, len(corpus["x"][v]))
              for v in IDS}
    path = tmp_path / name
    write_clip_scores(path, scores, corpus["labels"])
    return path


def test_score_a0_full_arm_reproduces_protocol_b(
    tmp_path: Path, corpus: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_model(monkeypatch, constants.V2_OFF, constants.V2_OFF)
    ref = _reference(tmp_path, corpus, "ref.npz")
    meta = ps.run_score(_score_args(tmp_path, corpus, "A0", ref))
    assert meta["x10b_max_abs_diff"] == 0.0
    scores = ps.read_arm_scores(tmp_path / "out" / ps.scores_npz("A0"))
    assert set(scores) == set(constants.V22_DF8_ARMS)
    crops, _ = ps.load_crops(corpus["crops"])
    v = IDS[0]
    assert len(scores[EARLY][v]) == crops[v][EARLY].length


def test_score_reference_gate_fails_loud(
    tmp_path: Path, corpus: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_model(monkeypatch, constants.V2_OFF, constants.V2_OFF)
    ref = _reference(tmp_path, corpus, "ref.npz")
    from core.tools.protocol_b_eval import read_clip_scores

    s, y = read_clip_scores(ref)
    s[IDS[0]] = s[IDS[0]] + 0.01
    write_clip_scores(ref, s, y)
    with pytest.raises(RuntimeError, match="X10"):
        ps.run_score(_score_args(tmp_path, corpus, "A0", ref))


def test_score_motion_model_patches_heads_and_scores_leaky(
    tmp_path: Path, corpus: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    enc = constants.VIDEOMAE_ENCODER_S
    _fake_model(monkeypatch, constants.V2_OFF, enc)
    stats = V2Stats(constants.V2_OFF, enc, 1.0, 1.0, np.zeros(2), np.ones(2))
    monkeypatch.setattr(ps, "load_stats", lambda d: stats)
    crops, _ = ps.load_crops(corpus["crops"])
    (tmp_path / "heads").mkdir()
    for v, arms in crops.items():
        rows = {a: np.zeros((len(ps.head_rows(arms[a])), 2)) for a in constants.V22_DF8_SHIFTED}
        np.savez(tmp_path / "heads" / f"{v}.npz", **rows)
    ref = _reference(tmp_path, corpus, "ref.npz")
    meta = ps.run_score(_score_args(tmp_path, corpus, ps.PRIMARY, ref))
    assert {EARLY + ps.LEAKY, LATE + ps.LEAKY} <= set(meta["per_run_macro"])
    assert meta["input"] == {"crn": constants.V2_OFF, "motion": enc}


def test_readout_end_to_end(tmp_path: Path, corpus: dict[str, Any]) -> None:
    crops, _ = ps.load_crops(corpus["crops"])
    out = tmp_path / "out"
    out.mkdir()
    model = {a: {v: ps.crop_labels(c[a]).astype(np.float64)
                 for v, c in crops.items()} for a in constants.V22_DF8_ARMS}
    prior = {a: {v: -np.abs(np.arange(c[a].length) / c[a].length - 0.6)
                 for v, c in crops.items()} for a in constants.V22_DF8_ARMS}
    ps.write_arm_scores(out / ps.scores_npz(ps.PRIMARY), model)
    ps.write_arm_scores(out / ps.scores_npz(ps.PRIOR_NAME), prior)
    per_run = {a: {"s1": 0.9} for a in constants.V22_DF8_ARMS}
    (out / ps.scores_json(ps.PRIMARY)).write_text(json.dumps(
        {"x10b_max_abs_diff": 0.0, "global_steps": {"s1": 7}, "per_run_macro": per_run}))
    (out / ps.scores_json(ps.PRIOR_NAME)).write_text(json.dumps({"x10b_max_abs_diff": 0.0}))
    ref_m = _reference(tmp_path, corpus, "ref_m.npz")
    ref_p = _reference(tmp_path, corpus, "ref_p.npz")
    args = argparse.Namespace(crops=corpus["crops"], scores_dir=out, reference_model=ref_m,
                              reference_prior=ref_p, seed=0, out_dir=out)
    r = ps.run_readout(args)
    assert r["outcome"] == "R1>0, X7 holds"  # a perfect detector: beats the prior, no arm cost
    assert r["r1_leaky"] is None
    assert "Verdict" in (out / ps.READOUT_MD).read_text()


def test_prior_full_arm_gate(tmp_path: Path, corpus: dict[str, Any],
                             monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    tau = rng.uniform(size=400)
    matrix = np.stack([tau, tau**2, tau**3], axis=1)
    target = (np.abs(tau - 0.6) < 0.15).astype(np.int64)
    monkeypatch.setattr(ps, "training_frames", lambda d: (matrix, target, 10))
    full = ps.prior_scores(matrix, target, {v: len(corpus["x"][v]) for v in IDS}, 0)
    ref = tmp_path / "prior_ref.npz"
    write_clip_scores(ref, full, corpus["labels"])
    args = argparse.Namespace(crops=corpus["crops"], data_dir=tmp_path, reference=ref, seed=0,
                              out_dir=tmp_path / "out")
    meta = ps.run_prior(args)
    assert meta["x10b_max_abs_diff"] == 0.0


def _same_frame_scores(
    crops: dict[str, dict[str, Any]], fn: Any
) -> dict[str, dict[str, np.ndarray]]:
    """Arm scores from ``fn(native_frame_index, clip_full_length, crop) -> score``."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for a in constants.V22_DF8_ARMS:
        out[a] = {}
        for v, arms in crops.items():
            c = arms[a]
            out[a][v] = np.array([fn(c.c0 + i, arms[FULL].length, c, i)
                                  for i in range(c.length)], dtype=np.float64)
    return out


def test_same_frame_content_model_has_zero_shift(tmp_path: Path, corpus: dict[str, Any]) -> None:
    crops, _ = ps.load_crops(corpus["crops"])
    out = tmp_path / "out"
    out.mkdir()
    rng = np.random.default_rng(1)
    noise = {v: rng.normal(size=c[FULL].length) for v, c in crops.items()}
    label = {v: corpus["labels"][v] for v in crops}
    owner = {id(c): v for v, arms in crops.items() for c in arms.values()}

    def content(n: int, total: int, c: ps.Crop, i: int) -> float:
        v = owner[id(c)]
        return float(label[v][n] + noise[v][n])

    def position(n: int, total: int, c: ps.Crop, i: int) -> float:
        return -abs(i / c.length - 0.6)

    ps.write_arm_scores(out / ps.scores_npz(ps.PRIMARY), _same_frame_scores(crops, content))
    ps.write_arm_scores(out / ps.scores_npz(ps.PRIOR_NAME), _same_frame_scores(crops, position))
    ps.write_arm_scores(out / ps.scores_npz("A3"), _same_frame_scores(crops, position))
    args = argparse.Namespace(crops=corpus["crops"], scores_dir=out, seed=0, out_dir=out)
    r = ps.run_same_frame(args)
    for arm in constants.V22_DF8_SHIFTED:
        assert r["models"][ps.PRIMARY]["arms"][arm]["shift"]["mean"] == 0.0
        assert r["models"][ps.PRIMARY]["tracking"][arm] is None  # no shift to track
        track = r["models"]["A3"]["tracking"][arm]
        if track is not None:  # a position-only model tracks the ruler exactly
            assert track["r"] == pytest.approx(1.0) and track["slope"] == pytest.approx(1.0)
    el = r["models"][ps.PRIMARY]["early_late"]
    assert el["total"]["mean"] == pytest.approx(el["frame_set"]["mean"] + el["move"]["mean"])
    assert "same frames" in (out / ps.SAME_FRAME_MD).read_text()


def test_same_frame_needs_the_ruler(tmp_path: Path, corpus: dict[str, Any]) -> None:
    args = argparse.Namespace(crops=corpus["crops"], scores_dir=tmp_path, seed=0, out_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="ruler"):
        ps.run_same_frame(args)
