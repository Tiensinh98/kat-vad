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
from core.data.v2_splits import load_split
from core.models import videomae_v2
from core.tools import dota_cap, extract_clip_features, mmau_match
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

    def test_warped_time_axis_fails(self) -> None:
        cap = _unit(np.random.default_rng(6), 200)
        positions = np.concatenate([np.arange(0, 60, 3), np.arange(61, 70), np.arange(72, 150, 3)])
        assert dota_cap.gate_alignment("q", "c", cap[positions], cap).reason == "irregular_steps"


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
