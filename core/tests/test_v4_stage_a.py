"""Tests for v4 Stage A (core/docs/v4/PREREG.md I1-I4, S1-S14, G1a).

The failures that matter are silent: a tap that reads the wrong head or changes the pooled
feature, a step label taken from conflicting windows, a detrend that leaves the trend in,
a selection that keeps a pure "late in the video" head, a random draw that overlaps the
selected heads, and a gate that runs on an uncommitted selection. Tiny encoders; no download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch
from torch import Tensor

from core import constants
from core.models.videomae_heads import HeadTap
from core.models.videomae_v2 import VideoMAEv2Encoder
from core.tools import extract_head_features as ehf
from core.tools import extract_video_features as evf
from core.tools import select_heads as sh
from core.tools import v4_gate

DIM, DEPTH, HEADS = 32, 2, 2
HEAD_DIM = DIM // HEADS
FRAME_SIZE = 8
STRIDE = 8
N_FRAMES = 20


def _encoder() -> VideoMAEv2Encoder:
    torch.manual_seed(0)
    return VideoMAEv2Encoder(DIM, DEPTH, HEADS).eval()


def _clip(batch: int = 2) -> Tensor:
    torch.manual_seed(1)
    size = constants.CROP_SIZE
    return torch.randn(batch, 3, constants.VIDEOMAE_CLIP_FRAMES, size, size)


def _write_frames(folder: Path, count: int) -> None:
    from torchvision.io import write_png

    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = torch.full((3, FRAME_SIZE, FRAME_SIZE), index * 9 % 256, dtype=torch.uint8)
        image[0, : index % FRAME_SIZE] = 255
        write_png(image, str(folder / f"{index:06d}.png"))


# ---------------------------------------------------------------------------
# I2: the tap
# ---------------------------------------------------------------------------
class TestHeadTap:
    def test_pooled_output_unchanged_and_hooks_removed(self) -> None:
        encoder, clip = _encoder(), _clip()
        with torch.no_grad():
            plain = encoder(clip)
            with HeadTap(encoder) as tap:
                tapped = encoder(clip)
                rows = tap.rows()
            again = encoder(clip)
        assert torch.equal(plain, tapped) and torch.equal(plain, again)
        assert rows.shape == (2, DEPTH * HEADS, HEAD_DIM)
        assert not encoder._forward_hooks
        assert all(not b.attn.proj._forward_pre_hooks for b in encoder.blocks)

    def test_tap_is_token_mean_of_proj_input_and_proj_gives_attention_output(self) -> None:
        encoder, clip = _encoder(), _clip()
        seen: dict[str, Tensor] = {}
        block = encoder.blocks[1]
        h1 = block.attn.proj.register_forward_pre_hook(
            lambda _m, a: seen.__setitem__("in", a[0].detach().clone()))
        h2 = block.attn.register_forward_hook(
            lambda _m, _a, out: seen.__setitem__("out", out.detach().clone()))
        with torch.no_grad(), HeadTap(encoder) as tap:
            encoder(clip)
            rows = tap.rows()
        h1.remove()
        h2.remove()
        merged = seen["in"].mean(dim=1).reshape(2, HEADS, HEAD_DIM).numpy()
        np.testing.assert_allclose(rows[:, HEADS:], merged, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(block.attn.proj(seen["in"]), seen["out"])

    def test_head_slices_match_a_per_head_sdpa(self) -> None:
        encoder, clip = _encoder(), _clip(1)
        attn = encoder.blocks[0].attn
        seen: dict[str, Tensor] = {}
        hook = encoder.blocks[0].norm1.register_forward_hook(
            lambda _m, _a, out: seen.__setitem__("x", out.detach().clone()))
        with torch.no_grad(), HeadTap(encoder) as tap:
            encoder(clip)
            rows = tap.rows()
        hook.remove()
        x = seen["x"]
        bias = torch.cat((attn.q_bias, torch.zeros_like(attn.v_bias), attn.v_bias))
        qkv = torch.nn.functional.linear(x, attn.qkv.weight, bias)
        q, k, v = qkv.split(DIM, dim=-1)
        for head in range(HEADS):
            part = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
            scores = (q[..., part] @ k[..., part].transpose(1, 2)) * HEAD_DIM ** -0.5
            out = (scores.softmax(dim=-1) @ v[..., part]).mean(dim=1)
            np.testing.assert_allclose(rows[0, head], out[0].detach().numpy(), rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# I3-I4: extraction and the row check
# ---------------------------------------------------------------------------
class TestExtraction:
    def test_heads_rows_align_and_u_check_passes(self, tmp_path: Path) -> None:
        encoder, device = _encoder(), torch.device("cpu")
        _write_frames(tmp_path / "frames" / "t01_v001", N_FRAMES)
        reference = tmp_path / "ref"
        reference.mkdir()
        u = evf.encode_frame_dir(tmp_path / "frames" / "t01_v001", encoder, device, STRIDE, 4)
        np.save(reference / "t01_v001.npy", u)
        out = tmp_path / "heads"
        ehf.extract_heads_directory(tmp_path / "frames", out, reference, encoder, device, STRIDE, 4)
        flat = np.load(out / "t01_v001.npy")
        assert flat.shape == (len(u), DEPTH * HEADS * HEAD_DIM) and flat.dtype == np.float32
        manifest = ehf.build_manifest(constants.V4_ENCODER, STRIDE, "sha", DEPTH, HEADS, HEAD_DIM)
        (out / constants.VIDEO_MANIFEST_FILENAME).write_text(json.dumps(manifest))
        assert ehf.load_head_rows(out, ["t01_v001"])["t01_v001"].shape == (3, 4, 16)
        assert ehf.pending_items({"t01_v001": tmp_path}, out) == []  # resume sees it as done (C11)
        assert ehf.ucos_failures(out, ["t01_v001"]) == {}

    def test_u_check_reports_a_mismatch_and_a_missing_source(self, tmp_path: Path) -> None:
        encoder, device = _encoder(), torch.device("cpu")
        _write_frames(tmp_path / "frames" / "t01_v001", N_FRAMES)
        reference = tmp_path / "ref"
        reference.mkdir()
        np.save(reference / "t01_v001.npy", np.random.default_rng(0).normal(size=(3, DIM)))
        out = tmp_path / "heads"
        ehf.extract_heads_directory(tmp_path / "frames", out, reference, encoder, device, STRIDE, 4)
        bad = ehf.ucos_failures(out, ["t01_v001", "t01_v002"])
        assert bad["t01_v002"] == -1.0
        assert bad["t01_v001"] < constants.V4_U_COS_MIN

    def test_reference_manifest_must_match(self, tmp_path: Path) -> None:
        enc = constants.V4_ENCODER
        manifest = evf.build_manifest(enc, STRIDE, constants.VIDEOMAE_WEIGHTS_SHA256[enc])
        (tmp_path / constants.VIDEO_MANIFEST_FILENAME).write_text(json.dumps(manifest))
        ehf.check_reference_manifest(tmp_path, enc, STRIDE)
        with pytest.raises(ValueError, match="stride"):
            ehf.check_reference_manifest(tmp_path, enc, 3)

    def test_row_cosines_refuse_misaligned_rows(self) -> None:
        with pytest.raises(ValueError):
            ehf.row_cosines(np.ones((3, 4)), np.ones((4, 4)))


# ---------------------------------------------------------------------------
# S1-S3: labels and representations
# ---------------------------------------------------------------------------
class TestLabels:
    def test_conflicting_and_uncovered_steps_are_not_scored(self) -> None:
        labels = sh.step_labels([(0, 4, 0), (2, 6, 1), (6, 8, 1)], 10)
        assert labels.tolist() == [0, 0, -1, -1, 1, 1, 1, 1, -1, -1]

    def test_window_outside_the_source_raises(self) -> None:
        with pytest.raises(ValueError):
            sh.step_labels([(5, 12, 1)], 10)

    def test_train_windows_take_train_split_and_positive_frames(self) -> None:
        meta = {
            "w0": {"source": "a", "split": "train", "start": 0, "end": 4, "positive_frames": 0},
            "w1": {"source": "a", "split": "train", "start": 4, "end": 8, "positive_frames": 3},
            "w2": {"source": "b", "split": "train", "start": 0, "end": 4, "positive_frames": 1},
            "w3": {"source": "a", "split": "test", "start": 0, "end": 4, "positive_frames": 1},
        }
        assert sh.train_windows(meta, {"a"}) == {"a": [(0, 4, 0), (4, 8, 1)]}


class TestRepresentations:
    def test_detrend_removes_a_cubic_exactly(self) -> None:
        tau = (np.arange(40) + 0.5) / 40
        rows = np.stack([1 + 2 * tau - 3 * tau ** 3, tau ** 2], axis=1)[:, :, None]
        np.testing.assert_allclose(sh.detrend_rows(rows), 0.0, atol=1e-10)

    def test_detrend_keeps_a_bump(self) -> None:
        rows = np.zeros((40, 1, 1))
        rows[18:24] = 1.0
        residual = sh.detrend_rows(rows)[:, 0, 0]
        # measured 0.78: the cubic absorbs ~22 % of a 6-of-40-step bump (the F5 cost, quantified)
        assert residual[18:24].mean() - np.delete(residual, range(18, 24)).mean() > 0.7

    def test_crn_rows_zero_median_per_channel(self) -> None:
        rows = np.random.default_rng(0).normal(size=(9, 3, 2)) + 5.0
        np.testing.assert_allclose(np.median(sh.crn_rows(rows), axis=0), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# S7-S12: measures, ranking, stability, draws
# ---------------------------------------------------------------------------
class TestMeasures:
    @pytest.mark.parametrize("measure", ["lda", "kl", "mmd", "nmi"])
    def test_separated_classes_score_higher(self, measure: str) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(size=(200, 4))
        same, shifted = rng.normal(size=(200, 4)), rng.normal(size=(200, 4)) + 2.0
        fn = {
            "lda": sh.lda_separability,
            "kl": sh.sym_gaussian_kl,
            "mmd": sh.mmd_rbf,
            "nmi": lambda x, y: sh.kmeans_nmi(x, y, 0),
        }[measure]
        assert fn(a, shifted) > fn(a, same) + 0.1

    def test_normalise_and_rss(self) -> None:
        norm = sh.normalise(np.array([[1.0, 5.0], [3.0, 5.0], [2.0, 5.0]]))
        assert norm[:, 0].tolist() == [0.0, 1.0, 0.5] and norm[:, 1].tolist() == [0.0, 0.0, 0.0]
        rss = sh.robust_score(np.array([[1.0, 0.5], [1.0, 0.9]]))
        np.testing.assert_allclose(rss, [1.0, 0.7 - 0.5 * 0.2])

    def test_top_k_breaks_ties_by_lower_index(self) -> None:
        assert sh.top_k(np.array([0.5, 0.9, 0.9, 0.1, 0.9]), 2) == [1, 2]

    def test_chance_jaccard(self) -> None:
        assert sh.chance_jaccard(72, 5) == pytest.approx(0.0397, abs=1e-4)
        rng = np.random.default_rng(0)
        sim = np.mean([sh.jaccard(list(rng.choice(72, 5, replace=False)),
                                  list(rng.choice(72, 5, replace=False))) for _ in range(20000)])
        assert sim == pytest.approx(sh.chance_jaccard(72, 5), abs=0.003)

    def test_folds_are_disjoint_and_cover(self) -> None:
        folds = sh.make_folds([f"s{i}" for i in range(23)], 5, 2024)
        flat = [v for f in folds for v in f]
        assert sorted(flat) == sorted(f"s{i}" for i in range(23)) and len(set(flat)) == 23
        assert folds == sh.make_folds([f"s{i}" for i in range(23)], 5, 2024)


Rows = dict[str, np.ndarray]


def _synthetic_sources(n: int = 30, length: int = 40) -> tuple[Rows, Rows]:
    """12 heads (2 blocks x 6) of 4-d noise; head 3 = a late-in-video ramp, head 7 = a mid bump."""
    rng = np.random.default_rng(0)
    tau = (np.arange(length) + 0.5) / length
    rows, labels = {}, {}
    for i in range(n):
        x = rng.normal(size=(length, 12, 4))
        x[:, 3, :] += 6.0 * tau[:, None]
        x[22:32, 7, :] += 2.0
        rows[f"s{i:02d}"] = x
        labels[f"s{i:02d}"] = sh.step_labels([(0, 20, 0), (20, 40, 1)], length)
    return rows, labels


class TestRanking:
    def test_detrending_drops_the_trend_head_and_keeps_the_bump(self) -> None:
        rows, labels = _synthetic_sources()
        folds = sh.make_folds(sorted(rows), 5, 2024)
        det = sh.rank({v: sh.detrend_rows(r) for v, r in rows.items()}, labels, folds, 2024)
        raw = sh.rank({v: sh.crn_rows(r) for v, r in rows.items()}, labels, folds, 2024)
        assert 7 in det["top"]
        assert 3 in raw["top"]
        assert det["rss"][7] > det["rss"][3]
        assert int(np.argmax(raw["rss"])) == 3

    def test_assemble_draws_avoid_the_selected_heads(self) -> None:
        rows, labels = _synthetic_sources()
        folds = sh.make_folds(sorted(rows), 5, 2024)
        det = sh.rank({v: sh.detrend_rows(r) for v, r in rows.items()}, labels, folds, 2024)
        raw = sh.rank({v: sh.crn_rows(r) for v, r in rows.items()}, labels, folds, 2024)
        out = sh.assemble(det, raw, 6, 2024)
        sel = set(out["selected"])
        assert len(out["a4_rand"]) == 5 and not sel & set(out["a4_rand"])
        assert len(out["g1a_draws"]) == constants.V4_RAND_DRAWS
        assert all(len(d) == 5 and not sel & set(d) for d in out["g1a_draws"])
        assert out["full_heads"] == list(range(out["best_block"] * 6, out["best_block"] * 6 + 6))
        assert out["stability"]["verdict"] in (constants.V4_PASS, constants.V4_FAIL)
        json.dumps(out)  # serialisable as committed


# ---------------------------------------------------------------------------
# G1a
# ---------------------------------------------------------------------------
class TestG1a:
    # median (lower) draw = rand_p_09 = 0.69
    DRAWS: ClassVar[dict[str, float]] = {f"rand_p_{j:02d}": 0.60 + 0.01 * j for j in range(20)}

    def test_pass(self) -> None:
        d = v4_gate.decide_g1a(constants.V4_PASS, 0.70, 0.65, self.DRAWS)
        assert d["verdict"] == constants.V4_PASS and d["median_draw"] == "rand_p_09"

    @pytest.mark.parametrize(("stability", "sel", "full"), [
        (constants.V4_FAIL, 0.70, 0.65),
        (constants.V4_PASS, 0.70, 0.71),
        (constants.V4_PASS, 0.685, 0.60),
    ])
    def test_each_check_can_fail(self, stability: str, sel: float, full: float) -> None:
        assert v4_gate.decide_g1a(stability, sel, full, self.DRAWS)["verdict"] == constants.V4_FAIL

    def test_flag_is_margin_below_half_width(self) -> None:
        assert v4_gate.flag({"mean": 0.01, "low": -0.01, "high": 0.03})
        assert not v4_gate.flag({"mean": 0.03, "low": 0.02, "high": 0.04})

    def test_uncommitted_selection_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "head_selection.json"
        path.write_text(json.dumps({"train_ids_sha1": "x"}))
        with pytest.raises(SystemExit, match="sha256"):
            v4_gate.check_selection(path, "0" * 64, ["a"])

    def test_feature_widths(self) -> None:
        rng = np.random.default_rng(0)
        streams = v4_gate.Streams(
            {"a": rng.normal(size=(10, 512))}, {"a": rng.normal(size=(10, 384))},
            {"a": rng.normal(size=(10, 72, 64)).astype(np.float32)},
            {"a": np.zeros(10, dtype=np.int64)},
        )
        selection = {"selected": [0, 1, 2, 3, 4], "full_heads": list(range(6, 12)),
                     "raw": {"top": [5, 6, 7, 8, 9]}, "g1a_draws": [[20, 21, 22, 23, 24]]}
        sets = v4_gate.feature_sets(selection)
        deg = constants.V2_K_POSITION_DEGREE
        widths = {name: build(streams, "a").shape[1] for name, build in sets.items()}
        assert widths[v4_gate.SEL] == 320 + deg
        assert widths[v4_gate.FULL] == 384 + deg
        assert widths[v4_gate.P] == deg
        assert widths[v4_gate.WITH_XUE] == 512 + 384 + 320 + deg
        assert widths["rand_p_00"] == 320 + deg and widths["head_00_p"] == 64 + deg

    def test_cli_parses(self) -> None:
        args = v4_gate.build_arg_parser().parse_args([
            "g1a", "--t2-dir", "t", "--annotation", "a", "--census", "c*", "--clip-dir", "x",
            "--video-dir", "u", "--heads-dir", "h", "--selection", "s", "--selection-sha256", "0",
            "--out-dir", "o",
        ])
        assert isinstance(args, argparse.Namespace) and args.seed == constants.V4_SEED


# ---------------------------------------------------------------------------
# integration: the glue the notebook runs (meta, manifest, I4, S14, read-outs)
# ---------------------------------------------------------------------------
def _fake_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    rows, _labels = _synthetic_sources(n=36)
    ids = sorted(rows)
    train, val = ids[:30], ids[30:]
    t2 = tmp_path / "t2"
    t2.mkdir()
    meta = {}
    for v in ids:
        for k, (start, end, pos) in enumerate([(0, 20, 0), (20, 40, 3)]):
            meta[f"{v}_w{k}"] = {"source": v, "split": "train", "start": start, "end": end,
                                 "positive_frames": pos}
    (t2 / constants.META_FILENAME).write_text(json.dumps(meta))
    heads = tmp_path / "heads"
    (heads / constants.V4_UCOS_SUBDIR).mkdir(parents=True)
    manifest = ehf.build_manifest(constants.V4_ENCODER, STRIDE, "sha", 2, 6, 4)
    (heads / constants.VIDEO_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    for v in ids:
        np.save(heads / f"{v}.npy", rows[v].reshape(40, -1).astype(np.float32))
        np.save(ehf.ucos_path(heads, v), np.ones((40, 1), dtype=np.float32))
    monkeypatch.setattr(sh, "t2_sources", lambda _meta: (val, train))
    monkeypatch.setattr(v4_gate, "t2_sources", lambda _meta: (val, train))
    return argparse.Namespace(t2_dir=t2, heads_dir=heads, out_dir=tmp_path / "out",
                              seed=2024, commit="test")


class TestIntegration:
    def test_select_then_g1a(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        args = _fake_corpus(tmp_path, monkeypatch)
        sel = sh.run(args)
        assert 7 in sel["selected"] and sel["train_sources"] == 30
        assert sel["steps"] == {"unlabelled": 0, "normal": 600, "abnormal": 600}
        path = args.out_dir / constants.V4_SELECTION_FILENAME
        assert path.is_file() and (args.out_dir / constants.V4_SELECTION_MD).is_file()

        rng = np.random.default_rng(1)
        meta = json.loads((args.t2_dir / constants.META_FILENAME).read_text())
        ids = sorted({entry["source"] for entry in meta.values()})
        for name, width in (("clip", 512), ("video", 384)):
            (tmp_path / name).mkdir()
            for v in ids:
                np.save(tmp_path / name / f"{v}.npy", rng.normal(size=(40, width)))
        y = np.r_[np.zeros(22), np.ones(10), np.zeros(8)].astype(np.int64)  # position-only labels
        monkeypatch.setattr(v4_gate, "load_counts", lambda _p: {})
        monkeypatch.setattr(v4_gate, "source_labels",
                            lambda _a, _c, ids, _s: ({v: y.copy() for v in ids}, {}))
        g1a = argparse.Namespace(
            t2_dir=args.t2_dir, annotation=tmp_path / "x.xlsx", census=["c"],
            clip_dir=tmp_path / "clip", video_dir=tmp_path / "video", heads_dir=args.heads_dir,
            selection=path, selection_sha256=v4_gate.sha256_file(path), out_dir=args.out_dir,
            seed=2024, resamples=200, commit="test",
        )
        out = v4_gate.run_g1a(g1a)
        assert out["decision"]["verdict"] in (constants.V4_PASS, constants.V4_FAIL)
        assert out["val_sources"] == 6 and (args.out_dir / constants.V4_G1A_MD).is_file()
        assert len([k for k in out["macro"] if k.startswith("rand_p_")]) == constants.V4_RAND_DRAWS

    def test_selection_refuses_a_failed_row_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args = _fake_corpus(tmp_path, monkeypatch)
        np.save(ehf.ucos_path(args.heads_dir, "s00"), np.full((40, 1), 0.9, dtype=np.float32))
        with pytest.raises(SystemExit, match="I4"):
            sh.run(args)
