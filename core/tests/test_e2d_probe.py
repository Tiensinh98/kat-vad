"""E2(d) on DoTA-CAP-dev (addendum §12, N1-N12).

The failures that matter are silent: a VideoMAE row that is not the CLIP row it is paired
with, A3 probed on raw instead of CRN'd streams, a Δ that does not isolate ``u``, and a pick
rule that drifts from N6/N7.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.crn.reference import reference
from core.data.dota import DotaRecord
from core.data.v2_splits import lines_sha1
from core.tools import dota_cap, extract_video_features
from core.tools import e2d_probe as e2d

B = constants.VIDEOMAE_ENCODER_B
S = constants.VIDEOMAE_ENCODER_S
DIM_X = 6
DIM_U = 4
STEPS = 24
SEED = 2024


def _corpus(n_clips: int, seed: int, u_signal: float, prefix: str = "vid") -> e2d.ProbeCorpus:
    """Clips whose accident span is readable from ``u`` (with weight ``u_signal``) only."""
    rng = np.random.default_rng(seed)
    x: dict[str, np.ndarray] = {}
    y: dict[str, np.ndarray] = {}
    u: dict[str, dict[str, np.ndarray]] = {B: {}, S: {}}
    group: dict[str, str] = {}
    for i in range(n_clips):
        clip_id = f"{prefix}{i // 3:02d}_{i:06d}"
        start = int(rng.integers(4, STEPS - 8))
        labels = np.zeros(STEPS, dtype=np.int64)
        labels[start : start + 6] = 1
        y[clip_id] = labels
        x[clip_id] = rng.normal(size=(STEPS, DIM_X))
        for enc in (B, S):
            noise = rng.normal(size=(STEPS, DIM_U))
            noise[:, 0] += u_signal * labels
            u[enc][clip_id] = noise
        group[clip_id] = clip_id.rpartition("_")[0]
    return e2d.ProbeCorpus(x, u, y, group)


def _block(in_domain: float, transfer: float) -> dict[str, dict[str, dict[str, float]]]:
    return {
        "in_domain": {"delta": {"mean": in_domain}},
        "transfer": {"delta": {"mean": transfer}},
    }


class TestRepresentations:
    def test_a2_is_raw_and_a3_is_r2_centred_per_clip(self) -> None:
        corpus = _corpus(3, 0, 0.0)
        clip = sorted(corpus.x)[0]
        a2 = e2d.arm_sets(corpus, B, "A2")
        a3 = e2d.arm_sets(corpus, B, "A3")
        np.testing.assert_array_equal(a2["base"][clip], corpus.x[clip])
        np.testing.assert_array_equal(a2["with"][clip][:, DIM_X:], corpus.u[B][clip])
        x_tilde = corpus.x[clip] - reference(corpus.x[clip], "R2")
        u_tilde = corpus.u[B][clip] - reference(corpus.u[B][clip], "R2")
        np.testing.assert_allclose(a3["base"][clip], x_tilde)
        np.testing.assert_allclose(a3["with"][clip], np.concatenate([x_tilde, u_tilde], axis=1))

    def test_the_only_difference_between_base_and_with_is_u(self) -> None:
        corpus = _corpus(2, 1, 0.0)
        for arm in e2d.ARMS:
            sets = e2d.arm_sets(corpus, S, arm)
            for clip in corpus.x:
                np.testing.assert_array_equal(sets["with"][clip][:, :DIM_X], sets["base"][clip])
                np.testing.assert_array_equal(sets["with_p"][clip][:, -3:], sets["p"][clip])


class TestProbes:
    def test_signal_in_u_shows_as_a_positive_delta_both_probes(self) -> None:
        dota = _corpus(30, 2, 3.0)
        t2 = _corpus(30, 3, 3.0, prefix="src")
        share = dict.fromkeys(dota.x, 0.25)
        out = e2d.probe_arm(dota, t2, B, "A2", folds=5, seed=SEED, resamples=200, share=share)
        assert out["in_domain"]["delta"]["mean"] > e2d.constants.V2_E2D_INDOMAIN_MIN
        assert out["transfer"]["delta"]["mean"] > e2d.constants.V2_E2D_TRANSFER_MIN
        assert out["transfer"]["macro"]["u"]["mean"] > 0.8

    def test_no_signal_gives_no_delta(self) -> None:
        dota = _corpus(30, 4, 0.0)
        t2 = _corpus(30, 5, 0.0, prefix="src")
        share = dict.fromkeys(dota.x, 0.25)
        out = e2d.probe_arm(dota, t2, S, "A3", folds=5, seed=SEED, resamples=200, share=share)
        assert abs(out["transfer"]["delta"]["mean"]) < e2d.constants.V2_E2D_TRANSFER_MIN

    def test_in_domain_folds_never_split_a_source_video(self) -> None:
        corpus = _corpus(12, 6, 3.0)
        ids = corpus.two_class()
        aucs = e2d.in_domain_aucs(corpus.x, corpus, ids, folds=4, seed=SEED)
        assert set(aucs) == set(ids)
        assert len({corpus.group[v] for v in ids}) == 4  # 12 clips, 3 per source video


class TestDecide:
    def test_none_eligible_drops_the_motion_stream(self) -> None:
        arms = {"A2": _block(0.09, 0.02), "A3": _block(0.05, 0.029)}
        assert e2d.decide({B: arms, S: arms})["pick"] == e2d.MOTION_DROPPED

    def test_in_domain_alone_makes_an_encoder_eligible(self) -> None:
        decision = e2d.decide({
            B: {"A2": _block(0.10, 0.0), "A3": _block(0.0, 0.0)},
            S: {"A2": _block(0.0, 0.0), "A3": _block(0.0, 0.0)},
        })
        assert decision["eligible"] == {B: ["A2"], S: []}
        assert decision["pick"] == B

    def test_within_the_tie_margin_the_cheaper_encoder_wins(self) -> None:
        decision = e2d.decide({
            B: {"A2": _block(0.0, 0.0), "A3": _block(0.0, 0.050)},
            S: {"A2": _block(0.0, 0.0), "A3": _block(0.0, 0.035)},
        })
        assert decision["pick"] == S
        assert decision["tie"]

    def test_beyond_the_margin_the_better_encoder_wins(self) -> None:
        decision = e2d.decide({
            B: {"A2": _block(0.0, 0.0), "A3": _block(0.0, 0.060)},
            S: {"A2": _block(0.0, 0.0), "A3": _block(0.0, 0.035)},
        })
        assert decision["pick"] == B
        assert not decision["tie"]


class TestGates:
    def test_videomae_rows_must_match_the_clip_rows(self, tmp_path: Path) -> None:
        s1, video = tmp_path / "s1", tmp_path / "video"
        s1.mkdir()
        video.mkdir()
        np.save(s1 / "vid_000010.npy", np.zeros((30, DIM_X), dtype=np.float32))
        np.save(video / "vid_000010.npy", np.zeros((29, DIM_U), dtype=np.float32))
        record = DotaRecord("vid_000010", "ego: turning", 30, (0.2, 0.5))
        with pytest.raises(ValueError, match="L6"):
            e2d.load_dota(["vid_000010"], {"vid_000010": record}, s1, {B: video}, 3)

    def test_stride_three_rows_and_labels_line_up(self, tmp_path: Path) -> None:
        s1, video = tmp_path / "s1", tmp_path / "video"
        s1.mkdir()
        video.mkdir()
        rows = np.arange(30 * DIM_X, dtype=np.float32).reshape(30, DIM_X)
        np.save(s1 / "vid_000010.npy", rows)
        np.save(video / "vid_000010.npy", np.ones((30, DIM_U), dtype=np.float32))
        record = DotaRecord("vid_000010", "ego: turning", 30, (0.2, 0.5))
        corpus = e2d.load_dota(["vid_000010"], {"vid_000010": record}, s1, {B: video}, 3)
        np.testing.assert_array_equal(corpus.x["vid_000010"], rows[::3])
        assert len(corpus.y["vid_000010"]) == len(corpus.u[B]["vid_000010"]) == 10

    def test_a_cache_built_with_other_settings_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / constants.VIDEO_MANIFEST_FILENAME).write_text(
            json.dumps({"encoder": B, "stride": 8}), encoding="utf-8"
        )
        e2d.check_manifest(tmp_path, {"encoder": B, "stride": 8})
        with pytest.raises(ValueError, match="stride"):
            e2d.check_manifest(tmp_path, {"encoder": B, "stride": 1})
        with pytest.raises(FileNotFoundError):
            e2d.check_manifest(tmp_path / "missing", {"encoder": B})

    def test_committed_alignment_sha_is_the_frozen_one(self) -> None:
        assert e2d.dota_cap_alignment_sha256(constants.V2_SPLITS_DIR).startswith("5e8690e031ca")


class TestRunSmoke:
    """The CLI wiring end to end on synthetic caches: gates, loading, decision, D15, read-out."""

    def test_run_writes_a_readout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        rng = np.random.default_rng(9)
        dev = [f"vid{g:02d}_{i:06d}" for g in range(12) for i in range(3)]
        cap = dev[: 2 * len(dev) // 3]
        k_ids = [f"t01_v{i:03d}" for i in range(15)]
        split_dir, s1_dir, video, clip = (tmp_path / n for n in ("splits", "s1", "video", "clip"))
        for d in (split_dir, s1_dir, clip):
            d.mkdir()
        (split_dir / "dota_dev.txt").write_text("".join(f"{v}\n" for v in dev))
        (split_dir / "dota_cap_dev.txt").write_text("".join(f"{v}\n" for v in cap))
        (split_dir / constants.V2_SPLITS_MANIFEST_FILENAME).write_text(json.dumps(
            {"splits": {"dota_dev": {"count": len(dev), "sha1": lines_sha1(dev)}}}))
        (split_dir / constants.V2_DOTA_CAP_MANIFEST_FILENAME).write_text(json.dumps({
            "source": {"alignment_sha256": "a" * 64},
            "splits": {"dota_cap_dev": {"count": len(cap), "sha1": lines_sha1(cap)}},
        }))
        records = {}
        for v in dev:
            frames = int(rng.integers(30, 60))
            np.save(s1_dir / f"{v}.npy", rng.normal(size=(frames, DIM_X)).astype(np.float32))
            records[v] = DotaRecord(v, "ego: turning" if v[3] < "5" else "other: turning",
                                    frames, (0.4, 0.7))
        y_t2: dict[str, np.ndarray] = {}
        for enc in (B, S):
            dota_dir, t2_dir = e2d.dota_video_dir(video, enc), e2d.t2_video_dir(video, enc)
            for d in (dota_dir, t2_dir):
                d.mkdir(parents=True)
            weights = constants.VIDEOMAE_WEIGHTS_SHA256[enc]
            (dota_dir / constants.VIDEO_MANIFEST_FILENAME).write_text(
                json.dumps(dota_cap.video_manifest(enc, "a" * 64)))
            (t2_dir / constants.VIDEO_MANIFEST_FILENAME).write_text(json.dumps(
                extract_video_features.build_manifest(enc, constants.FRAME_STRIDE, weights)))
            for v in cap:
                n = len(np.load(s1_dir / f"{v}.npy"))
                np.save(dota_dir / f"{v}.npy", rng.normal(size=(n, DIM_U)).astype(np.float32))
            for v in k_ids:
                np.save(t2_dir / f"{v}.npy", rng.normal(size=(30, DIM_U)).astype(np.float32))
        for v in k_ids:
            np.save(clip / f"{v}.npy", rng.normal(size=(30, DIM_X)).astype(np.float32))
            y_t2[v] = (np.arange(30) >= 15).astype(np.int64)
        k_file = tmp_path / "k.txt"
        k_file.write_text("".join(f"{v}\n" for v in k_ids))
        (tmp_path / "k.json").write_text(json.dumps({"sha1": lines_sha1(k_ids)}))
        a0 = tmp_path / "a0.json"
        a0.write_text(json.dumps(dict.fromkeys(dev, 0.6)))

        monkeypatch.setattr(e2d, "parse_metadata", lambda *_a: list(records.values()))
        monkeypatch.setattr(e2d, "read_split_ids", lambda *_a: set(dev))
        monkeypatch.setattr(e2d, "load_counts", lambda *_a: dict.fromkeys(k_ids, 240))
        monkeypatch.setattr(e2d, "source_labels", lambda *_a: (y_t2, dict.fromkeys(k_ids, 1)))
        out = tmp_path / "out"
        e2d.main([
            "--dota-s1-dir", str(s1_dir), "--metadata", "m", "--split-file", "s",
            "--k-ids-file", str(k_file), "--k-manifest", str(tmp_path / "k.json"),
            "--annotation", "x", "--census", "c", "--t2-clip-dir", str(clip),
            "--video-root", str(video), "--a0-clip-aucs", str(a0), "--split-dir", str(split_dir),
            "--resamples", "50", "--folds", "3", "--out-dir", str(out),
        ])
        readout = json.loads((out / e2d.READOUT_JSON).read_text())
        assert readout["decision"]["pick"] in (B, S, e2d.MOTION_DROPPED)
        assert readout["dota_cap_dev"]["clips"] == len(cap)
        d15 = readout["printed"]["d15"]
        assert d15["composition"]["cap"]["clips"] == len(cap)
        assert d15["a0_protocol_b"]["dev"]["mean"] == pytest.approx(0.6)
        assert "Decision" in (out / e2d.READOUT_MD).read_text()
