"""DoTA-CAP: DoTA pixels recovered from CAP-DATA (addendum §11, L1-L7)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision.io import encode_jpeg

from core import constants
from core.data.v2_splits import SealedSplitError, load_split
from core.models import videomae_v2
from core.tools import dota_cap, e2d_probe, extract_clip_features, mmau_match
from core.tools import extract_video_features as evf
from core.tools.stream_frames_clip import stream_video_batches

CPU = torch.device("cpu")
DIM = 64
SUBDIR = "images"
TINY_VIDEOMAE = (32, 1, 2)


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    return mmau_match.normalize_rows(rng.normal(size=(n, DIM)))


class TestMonotoneAlignment:
    @pytest.mark.parametrize(("start", "step", "n"), [(5, 3, 15), (0, 1, 20), (2, 2, 9)])
    def test_recovers_a_subsampled_clip(self, start: int, step: int, n: int) -> None:
        cap = _unit(np.random.default_rng(0), 60)
        query = cap[start::step][:n]
        path = dota_cap.monotone_alignment(query @ cap.T)
        assert path is not None
        np.testing.assert_array_equal(path, np.arange(n) * step + start)

    def test_static_stretch_stays_strictly_increasing(self) -> None:
        cap = _unit(np.random.default_rng(1), 30)
        cap[10:20] = cap[10]  # ten identical frames
        query = cap[5:25]
        path = dota_cap.monotone_alignment(query @ cap.T)
        assert path is not None and np.all(np.diff(path) > 0)

    def test_shorter_cap_has_no_alignment(self) -> None:
        rows = _unit(np.random.default_rng(2), 10)
        assert dota_cap.monotone_alignment(rows @ rows[:6].T) is None


class TestGate:
    def test_exact_subsample_passes(self) -> None:
        cap = _unit(np.random.default_rng(3), 90)
        a = dota_cap.gate_alignment("q", "c", cap[1::3], cap)
        assert a.reason == dota_cap.OK
        assert a.median_step == 3 and a.irregular_share == 0
        assert a.mean_cos == pytest.approx(1.0, abs=1e-5)

    def test_unrelated_clip_fails_on_cosine(self) -> None:
        rng = np.random.default_rng(4)
        a = dota_cap.gate_alignment("q", "c", _unit(rng, 10), _unit(rng, 40))
        assert a.reason == "mean_cos"

    def test_one_bad_frame_fails_min_cos(self) -> None:
        rng = np.random.default_rng(5)
        cap = _unit(rng, 200)
        query = cap[::2][:100].copy()
        query[50] = _unit(rng, 1)[0]  # mean stays >= 0.99, one frame does not exist in CAP
        assert dota_cap.gate_alignment("q", "c", query, cap).reason == "min_cos"

    def test_non_uniform_clip_fails_once_the_map_is_linear(self) -> None:
        cap = _unit(np.random.default_rng(6), 200)
        positions = np.concatenate([np.arange(0, 60, 3), np.arange(61, 70), np.arange(72, 150, 3)])
        a = dota_cap.gate_alignment("q", "c", cap[positions], cap)
        assert a.reason != dota_cap.OK

    def test_jittery_dp_path_is_replaced_by_the_line(self) -> None:
        """30 fps CAP frames CLIP cannot tell apart: the DP may jitter, the line does not."""
        rng = np.random.default_rng(8)
        base = _unit(rng, 40)
        cap = mmau_match.normalize_rows(
            np.repeat(base, 3, axis=0) + 1e-3 * rng.normal(size=(120, DIM))
        )
        a = dota_cap.gate_alignment("q", "c", base, cap)
        assert a.reason == dota_cap.OK
        assert a.rate == pytest.approx(3.0, abs=0.05)
        # every DoTA frame lands on one of its own three CAP copies (the band is +-1 at rate 3)
        np.testing.assert_array_equal(np.asarray(a.frame_map) // 3, np.arange(40))

    def test_half_up_rounding_keeps_the_line_at_rate_three(self) -> None:
        path = np.arange(40) * 3 + np.arange(40) % 2
        assert dota_cap.theil_sen(path.astype(np.float64)) == pytest.approx((3.0, 0.5))
        line, _ = dota_cap.linear_map(path)
        assert set(np.diff(line)) == {3}  # np.round's half-to-even would give 4, 2, 4, 2

    def test_theil_sen_matches_scipy(self) -> None:
        scipy_stats = pytest.importorskip("scipy.stats")
        path = np.array([0, 3, 7, 9, 12, 16, 18, 21, 30, 27], dtype=np.float64)
        rate, offset = dota_cap.theil_sen(path)
        ref = scipy_stats.theilslopes(path, np.arange(len(path)))
        assert (rate, offset) == pytest.approx((ref.slope, ref.intercept))

    def test_line_one_frame_past_the_end_is_shifted_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 30 fps CAP that ends on the DoTA clip's last frame; a fit offset of +1 overshoots."""
        rng = np.random.default_rng(10)
        base = _unit(rng, 40)
        cap = _unit(rng, 118)
        cap[::3] = base  # DoTA frame d is CAP frame 3d; the clip ends there
        monkeypatch.setattr(dota_cap, "theil_sen", lambda path: (3.0, 1.0))
        a = dota_cap.gate_alignment("q", "c", base, cap)
        assert a.reason == dota_cap.OK
        assert (a.overshoot, a.phase_shift) == (1, -1)
        np.testing.assert_array_equal(a.frame_map, np.arange(40) * 3)

    def test_resampled_source_residual_is_absorbed_by_the_band(self) -> None:
        """25 fps source -> CAP 30 fps and DoTA 10 fps: DoTA frame d is CAP 3d + {0, +1}."""
        rng = np.random.default_rng(11)
        cap = _unit(rng, 130)
        truth = np.arange(40) * 3 + (np.arange(40) % 2)  # periodic residual, no drift
        a = dota_cap.gate_alignment("q", "c", cap[truth], cap)
        assert a.reason == dota_cap.OK
        np.testing.assert_array_equal(a.frame_map, truth)
        assert a.line_mean_cos is not None and a.mean_cos is not None
        assert a.line_mean_cos < a.mean_cos

    def test_band_is_under_half_a_dota_frame(self) -> None:
        assert [dota_cap.band_width(r) for r in (1.0, 2.0, 2.5, 3.0)] == [0, 0, 1, 1]
        line = np.arange(5) * 3
        sims = np.zeros((5, 15))
        sims[np.arange(5), line + 1] = 0.5
        sims[np.arange(5), line + 2] = 1.0  # better, but two off the line: outside a band of 1
        np.testing.assert_array_equal(dota_cap.banded_path(sims, line, 1), line + 1)

    def test_rate_one_line_is_never_shifted(self) -> None:
        line = np.arange(10)
        assert [k for k, _ in dota_cap.phase_candidates(line, 1.0, 10)] == [0]
        assert dota_cap.phase_candidates(line + 1, 1.0, 10) == []
        assert [k for k, _ in dota_cap.phase_candidates(line * 3 + 1, 3.0, 29)] == [0, -1]

    def test_line_leaving_the_cap_clip_fails(self) -> None:
        cap = _unit(np.random.default_rng(9), 30)
        query = np.concatenate([cap[:20], cap[29:30]])  # last frame far off the line
        assert dota_cap.monotone_alignment(query @ cap.T) is not None
        a = dota_cap.gate_alignment("q", "c", query, cap)
        assert a.reason != dota_cap.OK


def _write_caches(root: Path, clips: dict[str, np.ndarray]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for vid, rows in clips.items():
        np.save(root / f"{vid}.npy", rows.astype(np.float32))
    return root


class TestAlignCli:
    def test_only_exact_pairs_are_aligned(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(7)
        dev = load_split(constants.V2_SPLIT_DOTA_DEV)[:2]
        cap = {"000001": _unit(rng, 60), "000002": _unit(rng, 60)}
        dota = {dev[0]: cap["000001"][::3], dev[1]: cap["000002"][::2], "x_000001": _unit(rng, 8)}
        matches = tmp_path / "m.json"
        matches.write_text(json.dumps({"dota": [
            {"query": dev[0], "cap_id": "000001", "grade": "exact", "frames": 20},
            {"query": dev[1], "cap_id": "000002", "grade": "exact", "frames": 30},
            {"query": "x_000001", "cap_id": "000001", "grade": "near", "frames": 8},
        ]}))
        out = tmp_path / "out"
        dota_cap.main([
            "align", "--matches", str(matches), "--out-dir", str(out),
            "--dota-clip-dir", str(_write_caches(tmp_path / "dota", dota)),
            "--cap-clip-dir", str(_write_caches(tmp_path / "cap", cap)),
        ])
        clips = json.loads((out / dota_cap.ALIGNMENT_JSON).read_text())["clips"]
        assert {k: c["reason"] for k, c in clips.items()} == {
            dev[0]: "ok", dev[1]: "ok", "x_000001": "p0_near",
        }
        assert clips[dev[0]]["frame_map"] == list(range(0, 60, 3))
        assert "| dev | 2 |" in (out / dota_cap.ALIGN_MD).read_text()


# --------------------------------------------------------------------------- extract


def _frames(seed: int, n: int, size: int = 40) -> list[bytes]:
    gen = torch.Generator().manual_seed(seed)
    return [
        bytes(encode_jpeg((torch.rand(3, size, size, generator=gen) * 255).to(torch.uint8)))
        for _ in range(n)
    ]


def _write_tar(path: Path, videos: dict[str, list[bytes]]) -> list[Path]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for vid, frames in videos.items():
            for i, data in enumerate(frames):
                info = tarfile.TarInfo(f"11/11/{vid}/{SUBDIR}/{i:06d}.jpg")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return [path]


@pytest.fixture(scope="module")
def tiny_clip() -> extract_clip_features.ImageEncoder:
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    config = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        image_size=constants.CROP_SIZE, patch_size=32, projection_dim=64,
    )
    torch.manual_seed(0)
    model: extract_clip_features.ImageEncoder = CLIPVisionModelWithProjection(config).eval()
    return model


def _encode_folder(frames: list[bytes], folder: Path) -> Path:
    (folder / SUBDIR).mkdir(parents=True, exist_ok=True)
    for i, data in enumerate(frames):
        (folder / SUBDIR / f"{i:06d}.jpg").write_bytes(data)
    return folder


class TestExtract:
    def test_rebuilt_dota_frames_gate_and_encode(
        self, tmp_path: Path, tiny_clip: extract_clip_features.ImageEncoder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cap_frames = {"000010": _frames(1, 30), "000011": _frames(2, 12), "000012": _frames(3, 9)}
        frame_map = list(range(2, 30, 3))  # a 30 fps CAP copy of a 10-frame DoTA clip
        dota_frames = [cap_frames["000010"][j] for j in frame_map]
        dota_dir = tmp_path / "dota_s1"
        dota_dir.mkdir()
        good = extract_clip_features.encode_frame_dir(
            _encode_folder(dota_frames, tmp_path / "ref" / "good"), tiny_clip, CPU, stride=1,
            subdir=SUBDIR, center_crop=False,
        )
        np.save(dota_dir / "good_000001.npy", good)
        np.save(dota_dir / "bad_000001.npy", -good[:4])  # pixels exist but are not DoTA's
        np.save(dota_dir / "short_000001.npy", good[:2])
        alignment = {"clips": {
            "good_000001": {"dota_id": "good_000001", "cap_id": "000010", "cap_frames": 30,
                            "frame_map": frame_map, "reason": "ok"},
            "bad_000001": {"dota_id": "bad_000001", "cap_id": "000011", "cap_frames": 12,
                           "frame_map": [0, 3, 6, 9], "reason": "ok"},
            "short_000001": {"dota_id": "short_000001", "cap_id": "000012", "cap_frames": 10,
                             "frame_map": [0, 1], "reason": "ok"},
            "skip_000001": {"dota_id": "skip_000001", "cap_id": "000013", "cap_frames": 5,
                            "frame_map": [], "reason": "mean_cos"},
        }}
        align_path = tmp_path / "align.json"
        align_path.write_text(json.dumps(alignment))
        torch.manual_seed(1)
        tiny = videomae_v2.VideoMAEv2Encoder(*TINY_VIDEOMAE).eval()
        monkeypatch.setattr(constants, "VIDEO_CACHE_DIR", tmp_path / "video")
        monkeypatch.setattr(extract_clip_features, "load_pretrained_encoder", lambda d: tiny_clip)
        monkeypatch.setattr(dota_cap, "load_pretrained", lambda n, d, w: tiny)
        report = tmp_path / "rep" / "extract_11.json"
        args = [
            "extract", "--alignment", str(align_path), "--work-dir", str(tmp_path / "work"),
            "--parts", *map(str, _write_tar(tmp_path / "parts" / "11.part_aa", cap_frames)),
            "--dota-clip-dir", str(dota_dir), "--report", str(report), "--batch-size", "4",
            "--device", "cpu",
        ]
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="3 DoTA CLIP rows missing"):
            dota_cap.main([*args, "--dota-clip-dir", str(empty)])  # argparse: last one wins
        assert not report.exists()
        dota_cap.main(args)

        got = json.loads(report.read_text())
        assert {k: v["reason"] for k, v in got.items()} == {
            "good_000001": "ok", "bad_000001": "pixel_clip_cos",
            "short_000001": "cap_frames_mismatch",
        }
        assert got["good_000001"]["clip_min_cos"] == pytest.approx(1.0, abs=1e-5)
        out = dota_cap.video_out_dir(constants.VIDEOMAE_ENCODER_B)
        features = np.load(out / "good_000001.npy")
        expected = evf.encode_frame_dir(
            tmp_path / "ref" / "good", tiny, CPU, stride=1, subdir=SUBDIR, clip_step=1,
        )
        np.testing.assert_array_equal(features, expected)
        assert len(features) == len(good)
        assert not (out / "bad_000001.npy").exists()
        manifest = json.loads((out / constants.VIDEO_MANIFEST_FILENAME).read_text())
        assert manifest["clip_frame_step"] == 1 and manifest["assumed_fps"] == constants.DOTA_FPS
        assert not (tmp_path / "work").exists()

        dota_cap.main(args)  # resume: everything decided, nothing re-streamed
        assert json.loads(report.read_text()) == got

        dev = load_split(constants.V2_SPLIT_DOTA_DEV)
        readout = dota_cap.run_finalize(dota_cap.build_arg_parser().parse_args([
            "finalize", "--alignment", str(align_path), "--reports", str(report),
            "--out-dir", str(tmp_path / "final"),
        ]))
        assert readout["ids"] == ["good_000001"]
        assert readout["coverage"]["all"] == {"kept": 1, "of": 4}
        assert readout["coverage"]["dev"]["of"] == len(dev)
        assert readout["reasons"]["align:mean_cos"] == 1


class TestShuffleExtract:
    """D6 / N11: ``--shuffle-seed`` writes its own cache, restricted to ``--dota-ids-file``."""

    def test_shuffle_cache_is_separate_and_permuted(
        self, tmp_path: Path, tiny_clip: extract_clip_features.ImageEncoder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cap_frames = {"000010": _frames(1, 20), "000011": _frames(2, 20)}
        dota_dir = tmp_path / "dota_s1"
        dota_dir.mkdir()
        clips = {}
        for dota_id, cap_id in (("keep_000001", "000010"), ("skip_000001", "000011")):
            ref = _encode_folder(cap_frames[cap_id], tmp_path / "ref" / dota_id)
            np.save(dota_dir / f"{dota_id}.npy", extract_clip_features.encode_frame_dir(
                ref, tiny_clip, CPU, stride=1, subdir=SUBDIR, center_crop=False))
            clips[dota_id] = {"dota_id": dota_id, "cap_id": cap_id, "cap_frames": 20,
                              "frame_map": list(range(20)), "reason": "ok"}
        align_path = tmp_path / "align.json"
        align_path.write_text(json.dumps({"clips": clips}))
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("keep_000001\n")
        torch.manual_seed(1)
        tiny = videomae_v2.VideoMAEv2Encoder(*TINY_VIDEOMAE).eval()
        monkeypatch.setattr(constants, "VIDEO_CACHE_DIR", tmp_path / "video")
        monkeypatch.setattr(extract_clip_features, "load_pretrained_encoder", lambda d: tiny_clip)
        monkeypatch.setattr(dota_cap, "load_pretrained", lambda n, d, w: tiny)
        seed = constants.V2_D6_SHUFFLE_SEED
        dota_cap.main([
            "extract", "--alignment", str(align_path), "--work-dir", str(tmp_path / "work"),
            "--parts", *map(str, _write_tar(tmp_path / "parts" / "11.part_aa", cap_frames)),
            "--dota-clip-dir", str(dota_dir), "--report", str(tmp_path / "rep.json"),
            "--dota-ids-file", str(ids_file), "--shuffle-seed", str(seed),
            "--batch-size", "4", "--device", "cpu",
        ])
        enc = constants.VIDEOMAE_ENCODER_B
        shuf = dota_cap.video_out_dir(enc, seed)
        assert shuf != dota_cap.video_out_dir(enc)
        assert not dota_cap.video_out_dir(enc).exists()
        assert sorted(p.name for p in shuf.glob("*.npy")) == ["keep_000001.npy"]
        order = evf.shuffled_frame_order(seed)
        expected = evf.encode_frame_dir(tmp_path / "ref" / "keep_000001", tiny, CPU, stride=1,
                                        subdir=SUBDIR, clip_step=1, frame_order=order)
        np.testing.assert_array_equal(np.load(shuf / "keep_000001.npy"), expected)
        manifest = json.loads((shuf / constants.VIDEO_MANIFEST_FILENAME).read_text())
        assert manifest["frame_order"] == order and manifest["shuffle_seed"] == seed

    def test_ordered_and_shuffled_manifests_cannot_be_confused(self, tmp_path: Path) -> None:
        enc = constants.VIDEOMAE_ENCODER_S
        ordered = dota_cap.video_manifest(enc, "a" * 64)
        shuffled = dota_cap.video_manifest(enc, "a" * 64, constants.V2_D6_SHUFFLE_SEED)
        legacy = {k: v for k, v in ordered.items() if k not in ("frame_order", "shuffle_seed")}
        (tmp_path / constants.VIDEO_MANIFEST_FILENAME).write_text(json.dumps(legacy))
        e2d_probe.check_manifest(tmp_path, ordered)  # caches built before D6 stay readable
        with pytest.raises(ValueError, match="frame_order"):
            e2d_probe.check_manifest(tmp_path, shuffled)
        (tmp_path / constants.VIDEO_MANIFEST_FILENAME).write_text(json.dumps(shuffled))
        with pytest.raises(ValueError, match="frame_order"):
            e2d_probe.check_manifest(tmp_path, ordered)


class TestStreamKeepIds:
    def test_only_kept_videos_are_written_and_the_stream_stops(self, tmp_path: Path) -> None:
        videos = {f"{i:06d}": _frames(i, 2, size=8) for i in range(1, 6)}
        seen: list[str] = []
        census = stream_video_batches(
            _write_tar(tmp_path / "p" / "g.part_aa", videos), tmp_path / "w", SUBDIR, 1e12,
            lambda ready: seen.extend(p.name for p in ready.iterdir()),
            keep_ids={"000002", "000003"},
        )
        assert sorted(seen) == ["000002", "000003"]
        assert "000005" not in census  # stopped after the last kept video closed


class TestClipStep:
    def test_dota_geometry_is_16_consecutive_native_frames(self) -> None:
        clip = evf.causal_clip_indices(20, step=constants.DOTA_VIDEOMAE_FRAME_STEP)
        assert constants.DOTA_VIDEOMAE_FRAME_STEP == 1
        assert clip == list(range(5, 21))


# --------------------------------------------------------------------------- freeze (L7)

FOREIGN_IDS = ("zzzNotDev_000001", "zzzNotDev_000050")


def _split_dir(tmp_path: Path) -> Path:
    """The committed base splits, without DoTA-eval's id file (freeze must not need it)."""
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for name in (constants.V2_SPLITS_MANIFEST_FILENAME, "dota_dev.txt"):
        (split_dir / name).write_bytes((constants.V2_SPLITS_DIR / name).read_bytes())
    return split_dir


def _readout(tmp_path: Path, ids: list[str], n_dev: int) -> tuple[Path, Path]:
    ids = sorted(ids)
    readout = {
        "ids": ids,
        "ids_sha1": dota_cap.lines_sha1(ids),
        "alignment_sha256": "0" * 64,
        "encoders": [constants.VIDEOMAE_ENCODER_B],
        "clip_cos_min_of_kept": 0.96,
        "reasons": {"ok": len(ids)},
        "coverage": {
            "all": {"kept": len(ids), "of": 10},
            "dev": {"kept": n_dev, "of": 5},
            "rest_eval": {"kept": len(ids) - n_dev, "of": 5},
        },
    }
    readout_path = tmp_path / "dota_cap_readout.json"
    readout_path.write_text(json.dumps(readout), encoding="utf-8")
    ids_path = tmp_path / "dota_cap_ids.txt"
    ids_path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    return readout_path, ids_path


class TestFreeze:
    @pytest.fixture
    def frozen(self, tmp_path: Path) -> tuple[Path, list[str], list[str]]:
        dev = load_split(constants.V2_SPLIT_DOTA_DEV)[:3]
        readout, ids = _readout(tmp_path, [*dev, *FOREIGN_IDS], len(dev))
        split_dir = _split_dir(tmp_path)
        dota_cap.main(["freeze", "--readout", str(readout), "--ids-file", str(ids),
                       "--split-dir", str(split_dir)])
        return split_dir, dev, list(FOREIGN_IDS)

    def test_sides_are_dev_intersection_and_its_complement(
        self, frozen: tuple[Path, list[str], list[str]]
    ) -> None:
        split_dir, dev, foreign = frozen
        assert load_split(constants.V2_SPLIT_DOTA_CAP_DEV, split_dir) == sorted(dev)
        assert load_split(constants.V2_SPLIT_DOTA_CAP_EVAL, split_dir, final=True) == foreign
        # the base splits still read, from the base manifest
        assert len(load_split(constants.V2_SPLIT_DOTA_DEV, split_dir)) == 702

    def test_eval_side_is_sealed(self, frozen: tuple[Path, list[str], list[str]]) -> None:
        with pytest.raises(SealedSplitError, match="sealed"):
            load_split(constants.V2_SPLIT_DOTA_CAP_EVAL, frozen[0])

    def test_refuses_to_overwrite_and_check_passes(
        self, frozen: tuple[Path, list[str], list[str]], tmp_path: Path
    ) -> None:
        args = ["freeze", "--readout", str(tmp_path / "dota_cap_readout.json"),
                "--ids-file", str(tmp_path / "dota_cap_ids.txt"), "--split-dir", str(frozen[0])]
        with pytest.raises(FileExistsError, match="frozen"):
            dota_cap.main(args)
        dota_cap.main([*args, "--check"])

    def test_check_fails_once_the_list_changes(
        self, frozen: tuple[Path, list[str], list[str]], tmp_path: Path
    ) -> None:
        split_dir, dev, _ = frozen
        other = tmp_path / "other"
        other.mkdir()
        readout, ids = _readout(other, [*dev, FOREIGN_IDS[0]], len(dev))
        with pytest.raises(ValueError, match="differs"):
            dota_cap.main(["freeze", "--readout", str(readout), "--ids-file", str(ids),
                           "--split-dir", str(split_dir), "--check"])

    def test_an_id_file_that_is_not_the_readouts_list_is_refused(self, tmp_path: Path) -> None:
        dev = load_split(constants.V2_SPLIT_DOTA_DEV)[:2]
        readout, ids = _readout(tmp_path, [*dev, *FOREIGN_IDS], len(dev))
        ids.write_text(f"{dev[0]}\n{FOREIGN_IDS[0]}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not the read-out's list"):
            dota_cap.main(["freeze", "--readout", str(readout), "--ids-file", str(ids),
                           "--split-dir", str(_split_dir(tmp_path))])

    def test_coverage_disagreement_is_refused(self, tmp_path: Path) -> None:
        dev = load_split(constants.V2_SPLIT_DOTA_DEV)[:2]
        readout, ids = _readout(tmp_path, [*dev, *FOREIGN_IDS], n_dev=3)
        with pytest.raises(ValueError, match="disagree"):
            dota_cap.main(["freeze", "--readout", str(readout), "--ids-file", str(ids),
                           "--split-dir", str(_split_dir(tmp_path))])

    def test_a_derived_manifest_cannot_refreeze_a_base_split(self, tmp_path: Path) -> None:
        split_dir = _split_dir(tmp_path)
        clash = {"splits": {constants.V2_SPLIT_DOTA_DEV: {"count": 0, "sha1": "x"}}}
        (split_dir / constants.V2_DOTA_CAP_MANIFEST_FILENAME).write_text(
            json.dumps(clash), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="re-freezes"):
            load_split(constants.V2_SPLIT_DOTA_DEV, split_dir)


class TestCommittedDotaCap:
    """The committed DoTA-CAP splits: the 2026-10-03 read-out (ids sha1 e1a37bb…)."""

    def test_committed_splits_match_the_readout(self) -> None:
        manifest_path = constants.V2_SPLITS_DIR / constants.V2_DOTA_CAP_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cap_dev = load_split(constants.V2_SPLIT_DOTA_CAP_DEV)
        cap_eval = load_split(constants.V2_SPLIT_DOTA_CAP_EVAL, final=True)
        assert (len(cap_dev), len(cap_eval)) == (569, 560)
        assert set(cap_dev) <= set(load_split(constants.V2_SPLIT_DOTA_DEV))
        assert not set(cap_eval) & set(cap_dev)
        assert dota_cap.lines_sha1([*cap_dev, *cap_eval]) == manifest["source"]["ids_sha1"]
        assert manifest["source"]["ids_sha1"].startswith("e1a37bb")
