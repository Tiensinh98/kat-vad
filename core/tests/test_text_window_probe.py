"""Text-guided multi-scale CLIP window probe (pending (ba)): extraction and probe.

The silent failures: a window cache whose whole-frame cell is not the CLIP row it is paired with
(another transform or frame map), cells that do not tile the frame, text weights that leak onto the
global cell or do not sum to one per grid, a "text" stream that differs from the base by more than
the pooled windows, and a verdict rule that drifts from the docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants
from core.data.dota import DotaRecord
from core.data.v2_splits import lines_sha1
from core.tests.test_extractors import tiny_clip_encoder  # noqa: F401  (pytest fixture)
from core.tools import clip_windows as cw
from core.tools import dota_cap, extract_clip_features, extract_video_features
from core.tools import e2d_probe as e2d
from core.tools import text_window_probe as tw
from core.tools.feature_cache import is_complete

CPU = torch.device("cpu")
W = cw.windows_per_frame()
DIM = 8
DIM_U = 4
DIM_WIN = 32  # smoke-test windows: wide enough that a planted cell out-scores random ones
S = constants.VIDEOMAE_ENCODER_S


def _rows_at_cosine(cos: np.ndarray) -> np.ndarray:
    """Unit rows whose cosine with ``eye(DIM)[i]`` is ``cos[i]`` (the last axis takes the rest)."""
    rows: np.ndarray = np.eye(DIM)[: len(cos)] * cos[:, None]
    rows[:, -1] = np.sqrt(1.0 - cos**2)
    return rows


def _write_frames(folder: Path, count: int, height: int = 45, width: int = 80) -> None:
    from torchvision.io import write_png

    folder.mkdir(parents=True)
    torch.manual_seed(1)
    for index in range(count):
        write_png((torch.rand(3, height, width) * 255).to(torch.uint8),
                  str(folder / f"{index:06d}.png"))


class TestWindows:
    def test_every_grid_tiles_the_frame(self) -> None:
        boxes = cw.window_boxes(45, 80)
        assert len(boxes) == W == 1 + 4 + 9 + 25
        offset = 0
        for g in constants.TW_GRIDS:
            cells = boxes[offset : offset + g * g]
            assert sum((bot - top) * (rgt - lft) for top, bot, lft, rgt in cells) == 45 * 80
            assert all(bot > top and rgt > lft for top, bot, lft, rgt in cells)
            offset += g * g

    def test_whole_frame_cell_is_the_ncc_transform(self) -> None:
        frames = (torch.rand(2, 45, 80, 3) * 255).to(torch.uint8).numpy()
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        pixels = cw.window_pixels(tensor, cw.window_boxes(45, 80)).view(2, W, 3, 224, 224)
        reference = extract_clip_features.preprocess_frames(frames, center_crop=False)
        assert torch.allclose(pixels[:, 0], reference, atol=1e-5)

    def test_encode_matches_the_clip_extractor_on_the_global_cell(
        self, tiny_clip_encoder: object, tmp_path: Path  # noqa: F811
    ) -> None:
        folder = tmp_path / "frames" / "vid_000001"
        _write_frames(folder, 5)
        paths = sorted(folder.iterdir())[::2]
        windows = cw.encode_windows(paths, tiny_clip_encoder, CPU, frames_per_batch=2)  # type: ignore[arg-type]
        assert windows.shape[:2] == (3, W)
        rows = extract_clip_features.encode_frame_dir(
            folder, tiny_clip_encoder, CPU, stride=2, center_crop=False  # type: ignore[arg-type]
        )
        assert cw.gate_rows(windows, rows)["reason"] == cw.OK
        assert np.allclose(cw.global_window(windows), rows, atol=1e-4)

    def test_stored_layout_round_trips_and_reads_as_complete(self, tmp_path: Path) -> None:
        windows = np.random.default_rng(0).normal(size=(4, W, DIM)).astype(np.float32)
        target = tmp_path / "a.npy"
        cw.save_array(target, cw.flatten(windows))
        assert is_complete(target)
        assert np.allclose(cw.unflatten(np.load(target)), windows, atol=1e-2)

    def test_gate_refuses_other_rows(self) -> None:
        rng = np.random.default_rng(0)
        windows = rng.normal(size=(4, W, DIM))
        assert cw.gate_rows(windows, rng.normal(size=(5, DIM)))["reason"] == cw.ROWS_MISMATCH
        assert cw.gate_rows(windows, rng.normal(size=(4, DIM)))["reason"] == cw.GATE_FAILED
        assert cw.gate_rows(windows, windows[:, 0])["reason"] == cw.OK

    def test_clip_mean_is_gated_only_when_asked(self) -> None:
        # a DoTA-CAP clip admitted at 0.990x over s1 can read 0.989x over its s3 rows (2026-10-08)
        rows = _rows_at_cosine(np.full(6, 0.985))
        windows = np.zeros((6, W, DIM))
        windows[:, 0] = np.eye(DIM)[:6]
        gate = cw.gate_rows(windows, rows)
        assert gate["reason"] == cw.OK and gate["global_mean_cos"] == pytest.approx(0.985)
        assert cw.gate_rows(windows, rows, constants.DOTA_CAP_MEAN_COS)["reason"] == cw.GATE_FAILED
        low_row = _rows_at_cosine(np.array([0.999] * 5 + [constants.DOTA_CAP_MIN_COS - 0.01]))
        assert cw.gate_rows(windows, low_row)["reason"] == cw.GATE_FAILED

    def test_corpus_gate_reads_the_mean_of_clip_means(self) -> None:
        margin = {"global_mean_cos": 0.989, "global_min_cos": 0.96}
        clean = {"global_mean_cos": 0.995, "global_min_cos": 0.98}
        gate = cw.corpus_gate([margin, clean, clean])
        assert gate["reason"] == cw.OK and gate["min_clip_mean"] == pytest.approx(0.989)
        assert cw.corpus_gate([margin] * 3)["reason"] == cw.GATE_FAILED
        with pytest.raises(ValueError, match="No gated clips"):
            cw.corpus_gate([])

    def test_frames_cli_writes_a_gated_cache(
        self, tiny_clip_encoder: object, tmp_path: Path  # noqa: F811
    ) -> None:
        frames = tmp_path / "frames"
        _write_frames(frames / "t01_v001", 9)
        clip_dir = tmp_path / "clip"
        extract_clip_features.extract_frame_directory(
            frames, clip_dir, tiny_clip_encoder, CPU, stride=4, center_crop=False,  # type: ignore[arg-type]
        )
        out = tmp_path / "win"
        args = cw.build_arg_parser().parse_args([
            "frames", "--frames-dir", str(frames), "--dataset", "T", "--clip-dir", str(clip_dir),
            "--stride", "4", "--output-dir", str(out), "--device", "cpu",
        ])
        cw.run_frames(args, tiny_clip_encoder)  # type: ignore[arg-type]
        stored = cw.unflatten(np.load(out / "t01_v001.npy"))
        assert stored.shape[:2] == (3, W)
        manifest = json.loads((out / constants.VIDEO_MANIFEST_FILENAME).read_text())
        assert manifest["grids"] == list(constants.TW_GRIDS) and manifest["stride"] == 4


class TestPooling:
    def test_text_weights_sum_to_one_per_grid_and_skip_the_global_cell(self) -> None:
        margin = np.random.default_rng(0).normal(size=(3, W))
        slices = tw.local_slices()
        weights = tw.text_weights(margin, slices)
        assert np.all(weights[:, 0] == 0.0)
        for s in slices:
            assert np.allclose(weights[:, s].sum(axis=1), 1.0)

    def test_a_cold_softmax_picks_the_most_anomalous_window(self) -> None:
        margin = np.zeros((1, W))
        margin[0, 20] = 1.0  # a 5x5 cell
        weights = tw.text_weights(margin, tw.local_slices(), tau=1e-3)
        assert weights[0, 20] == pytest.approx(1.0)

    def test_uniform_pooling_is_the_mean_of_each_grid_then_of_grids(self) -> None:
        windows = np.random.default_rng(1).normal(size=(2, W, DIM))
        slices = tw.local_slices()
        pooled = tw.pool(windows, tw.uniform_weights(2, W, slices), slices)
        expected = np.mean([windows[:, s].mean(axis=1) for s in slices], axis=0)
        assert np.allclose(pooled, expected)

    def test_margin_follows_the_text_anchors(self) -> None:
        z_abn, z_norm = np.eye(DIM)[0], np.eye(DIM)[1]
        windows = np.zeros((1, W, DIM))
        windows[0, :, 1] = 1.0
        windows[0, 7, :] = np.eye(DIM)[0]
        margin = tw.margins(windows, z_abn, z_norm)
        assert margin[0, 7] == pytest.approx(1.0) and margin[0, 8] == pytest.approx(-1.0)

    def test_flat_weights_read_as_entropy_one(self) -> None:
        slices = tw.local_slices()
        assert tw.weight_entropy(tw.uniform_weights(3, W, slices), slices) == pytest.approx(1.0)

    def test_prior_is_the_mean_weight_over_every_frame(self) -> None:
        a, b = np.full((2, W), 1.0), np.full((3, W), 6.0)
        assert np.allclose(tw.spatial_prior({"a": a, "b": b}), 4.0)


class TestProbeSets:
    def test_only_the_pooled_stream_differs_from_the_base(self) -> None:
        rng = np.random.default_rng(2)
        x = {"v_000001": rng.normal(size=(10, DIM))}
        u = {S: {"v_000001": rng.normal(size=(10, DIM_U))}}
        corpus = e2d.ProbeCorpus(x, u, {"v_000001": np.zeros(10, dtype=np.int64)}, {})
        streams = {name: {"v_000001": rng.normal(size=(10, DIM))} for name in (*tw.POOLS, "score")}
        sets = tw.probe_sets(corpus, streams)
        base, text = sets[tw.BASE]["v_000001"], sets["text"]["v_000001"]
        p = constants.V2_K_POSITION_DEGREE
        assert base.shape[1] == DIM + DIM_U + p and text.shape[1] == 2 * DIM + DIM_U + p
        assert np.allclose(text[:, : DIM + DIM_U], base[:, : DIM + DIM_U])
        assert np.allclose(text[:, -p:], base[:, -p:])
        assert np.allclose(np.median(text[:, DIM + DIM_U : -p], axis=0), 0.0)  # R2-centred


class TestVerdict:
    @staticmethod
    def _block(mean: float, low: float) -> dict[str, object]:
        delta = {name: {"mean": 0.0, "low": -1.0, "high": 1.0} for name in tw.CONTRASTS}
        delta[tw.DECISION] = {"mean": mean, "low": low, "high": mean + 0.01}
        return {"delta": {"all": delta}}

    def test_go_needs_both_the_size_and_a_positive_lower_bound(self) -> None:
        assert tw.verdict(self._block(0.012, 0.001))["verdict"] == tw.GO
        assert tw.verdict(self._block(0.009, 0.001))["verdict"] == tw.KILL
        assert tw.verdict(self._block(0.05, -0.001))["verdict"] == tw.KILL

    def test_sentences_are_negated_when_their_interval_includes_zero(self) -> None:
        sentences = tw.verdict(self._block(0.05, 0.01))["sentences"]
        assert all(s.startswith("NOT:") for s in sentences.values())


class TestRunSmoke:
    def test_run_writes_a_readout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        rng = np.random.default_rng(9)
        cap = [f"vid{g:02d}_{i:06d}" for g in range(10) for i in range(3)]
        k_ids = [f"t01_v{i:03d}" for i in range(12)]
        root = {n: tmp_path / n for n in ("splits", "s1", "video", "clip", "win")}
        for d in root.values():
            d.mkdir()
        (root["splits"] / "dota_cap_dev.txt").write_text("".join(f"{v}\n" for v in cap))
        (root["splits"] / constants.V2_SPLITS_MANIFEST_FILENAME).write_text(
            json.dumps({"splits": {}}))
        (root["splits"] / constants.V2_DOTA_CAP_MANIFEST_FILENAME).write_text(json.dumps({
            "source": {"alignment_sha256": "a" * 64},
            "splits": {"dota_cap_dev": {"count": len(cap), "sha1": lines_sha1(cap)}},
        }))
        z_abn = np.eye(DIM_WIN)[0]
        dota_u, t2_u = e2d.dota_video_dir(root["video"], S), e2d.t2_video_dir(root["video"], S)
        t2_w = root["win"] / cw.cache_dir(
            constants.DADA_ORIGIN_DATASET, constants.FRAME_STRIDE).name
        dota_w = root["win"] / cw.cache_dir(
            constants.DOTA_CAP_DATASET, constants.TW_DOTA_STRIDE).name
        for d in (dota_u, t2_u, t2_w, dota_w):
            d.mkdir(parents=True)
        (dota_u / constants.VIDEO_MANIFEST_FILENAME).write_text(
            json.dumps(dota_cap.video_manifest(S, "a" * 64)))
        (t2_u / constants.VIDEO_MANIFEST_FILENAME).write_text(json.dumps(
            extract_video_features.build_manifest(
                S, constants.FRAME_STRIDE, constants.VIDEOMAE_WEIGHTS_SHA256[S])))
        for d in (t2_w, dota_w):
            (d / constants.VIDEO_MANIFEST_FILENAME).write_text(
                json.dumps(cw.build_manifest("T", 1, "test")))

        def windows_for(labels: np.ndarray) -> np.ndarray:
            win = rng.normal(size=(len(labels), W, DIM_WIN))
            win[:, 30, :] += 6.0 * labels[:, None] * z_abn  # a small cell carries the event
            return win

        records, y_t2 = {}, {}
        for v in cap:
            frames = int(rng.integers(30, 60))
            np.save(root["s1"] / f"{v}.npy", rng.normal(size=(frames, DIM)).astype(np.float32))
            np.save(dota_u / f"{v}.npy", rng.normal(size=(frames, DIM_U)).astype(np.float32))
            records[v] = DotaRecord(v, "ego: turning" if v[4] < "5" else "other: turning",
                                    frames, (0.4, 0.7))
            labels = np.asarray(
                e2d.resized_frame_labels(records[v], frames, constants.TW_DOTA_STRIDE))
            cw.save_array(dota_w / f"{v}.npy", cw.flatten(windows_for(labels)))
        for v in k_ids:
            y_t2[v] = (np.arange(30) >= 15).astype(np.int64)
            np.save(root["clip"] / f"{v}.npy", rng.normal(size=(30, DIM)).astype(np.float32))
            np.save(t2_u / f"{v}.npy", rng.normal(size=(30, DIM_U)).astype(np.float32))
            cw.save_array(t2_w / f"{v}.npy", cw.flatten(windows_for(y_t2[v])))
        k_file = tmp_path / "k.txt"
        k_file.write_text("".join(f"{v}\n" for v in k_ids))
        (tmp_path / "k.json").write_text(json.dumps({"sha1": lines_sha1(k_ids)}))
        text = tmp_path / "text.npz"
        np.savez(text, abnormal=np.stack([z_abn, z_abn]), normal=np.eye(DIM_WIN)[1:3])

        monkeypatch.setattr(tw, "parse_metadata", lambda *_a: list(records.values()))
        monkeypatch.setattr(tw, "read_split_ids", lambda *_a: set(cap))
        monkeypatch.setattr(tw, "load_counts", lambda *_a: dict.fromkeys(k_ids, 240))
        monkeypatch.setattr(e2d, "source_labels", lambda *_a: (y_t2, dict.fromkeys(k_ids, 1)))
        out = tmp_path / "out"
        tw.main([
            "--dota-s1-dir", str(root["s1"]), "--metadata", "m", "--split-file", "s",
            "--k-ids-file", str(k_file), "--k-manifest", str(tmp_path / "k.json"),
            "--annotation", "x", "--census", "c", "--t2-clip-dir", str(root["clip"]),
            "--video-root", str(root["video"]), "--window-root", str(root["win"]),
            "--text-npz", str(text), "--split-dir", str(root["splits"]),
            "--resamples", "50", "--folds", "3", "--out-dir", str(out),
        ])
        readout = json.loads((out / tw.READOUT_JSON).read_text())
        assert readout["decision"]["verdict"] == tw.GO  # the planted cell is found by the text map
        delta = readout["transfer"]["delta"]["all"]
        assert delta["text - mean"]["mean"] > 0.0
        assert readout["dota_cap_dev"]["clips"] == len(cap)
        assert readout["dota_cap_dev"]["ego"] == readout["dota_cap_dev"]["non-ego"] == 15
        assert "Verdict" in (out / tw.READOUT_MD).read_text()
