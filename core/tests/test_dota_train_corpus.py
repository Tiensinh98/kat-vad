"""KIM_RUN §55: DoTA train features and windows.

Silent failures guarded: an excluded clip that still trains, a row label off by one, VideoMAE
clips at 30 fps geometry on 10 fps frames, frames read out of order from the archive.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from core import constants
from core.data.dada import sampled_frame_labels
from core.data.definitions import DATASET_CLS_DEFS, dataset_abbr
from core.tools import dota_train_build as dtb
from core.tools import dota_train_extract as dte
from core.tools import nexar_extract
from core.tools.dota_train_match import DotaClip

MAP = constants.DOTA_TRAIN_SPLITS_DIR / constants.DOTA_TRAIN_MAP_FILENAME
STRIDE = constants.DOTA_TRAIN_STRIDE


def _clip(frames: int, start: int, end: int, dota_id: str = "a_000001") -> DotaClip:
    return DotaClip(dota_id, frames, start, end, False)


def _archive(path: Path, frames: int) -> None:
    """Frame ``d`` is a flat grey of value ``d * 8``; entries are written in reverse order."""
    with zipfile.ZipFile(path, "w") as zf:
        for d in reversed(range(frames)):
            buffer = io.BytesIO()
            Image.new("RGB", (64, 36), (d * 8,) * 3).save(buffer, "JPEG", quality=95)
            zf.writestr(f"{d:06d}.jpg", buffer.getvalue())


def test_excluded_clips_never_train_and_val_is_disjoint() -> None:
    clips = dte.train_clips(MAP)
    assert len(clips) == 3023 - len(constants.DOTA_TRAIN_EXCLUDED) == 3020
    assert not set(constants.DOTA_TRAIN_EXCLUDED) & set(clips)
    val = {p for name in (constants.V2_SPLIT_DOTA_CAP_DEV, constants.V2_SPLIT_DOTA_CAP_EVAL)
           for p in (constants.V2_SPLITS_DIR / f"{name}.txt").read_text().split()}
    assert not val & set(clips)


def test_text_side_is_the_one_k5dfi_trained_with() -> None:
    nexar = DATASET_CLS_DEFS[dataset_abbr(constants.NEXAR_DATASET)]
    assert DATASET_CLS_DEFS[dataset_abbr(constants.DOTA_TRAIN_DATASET)] is nexar


@pytest.mark.parametrize("frames", [36, 37, 38, 101])
@pytest.mark.parametrize("start, end", [(0, 5), (1, 2), (10, 22), (11, 24), (12, 36)])
def test_row_j_is_abnormal_iff_frame_3j_is_in_the_span(frames: int, start: int, end: int) -> None:
    clip = _clip(frames, start, end)
    want = [int(start <= f < end) for f in range(0, frames, STRIDE)]
    assert sampled_frame_labels(dtb.make_record(clip), STRIDE) == want


def test_an_empty_span_raises() -> None:
    with pytest.raises(ValueError, match="annotated span"):
        dtb.make_record(_clip(60, 20, 20))


def test_windows_and_normals_come_from_the_same_clip(tmp_path: Path) -> None:
    stats = dtb.build(tmp_path, [_clip(90, 60, 90), _clip(30, 5, 20, "b_000002")])
    labels = json.loads((tmp_path / constants.LABELS_TRAIN_FILENAME).read_text())
    windows = json.loads((tmp_path / constants.WINDOWS_FILENAME).read_text())
    got = {(w["start"], w["end"]): labels[k] for k, w in windows.items()}
    # 30 rows, rows >= 20 abnormal: windows [0,12) [6,18) normal, [12,24) [18,30) abnormal
    assert got == {(0, 12): 0, (6, 18): 0, (12, 24): 1, (18, 30): 1}
    assert {w["source"] for w in windows.values()} == {"a_000001"}  # b has 10 rows: no window
    assert (stats["train_abnormal"], stats["train_normal"]) == (2, 2)
    assert stats["clips_with_windows"] == 1


def test_build_refuses_cached_rows_of_another_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _clip(90, 60, 90)
    monkeypatch.setattr(dtb, "train_clips", lambda _: {"a_000001": {"dota_frames": 90}})
    monkeypatch.setattr(dtb, "read_dota", lambda _: {"a_000001": clip})
    argv = ["--annotations", "x", "--clip-dir", str(tmp_path), "--out-dir", str(tmp_path / "d"),
            "--report", str(tmp_path / "r.json")]
    np.save(tmp_path / "a_000001.npy", np.zeros((29, 4), np.float32))
    with pytest.raises(ValueError, match="rows do not match"):
        dtb.main(argv)
    np.save(tmp_path / "a_000001.npy", np.zeros((30, 4), np.float32))
    dtb.main(argv)
    assert json.loads((tmp_path / "r.json").read_text())["construction"]["train_items"] == 4


def test_archive_is_decoded_in_frame_order_and_length_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "a.zip"
    _archive(path, 12)
    out = dte.decode_archive(path, 12)
    assert out.shape == (12, 3, constants.CROP_SIZE, constants.CROP_SIZE)
    level = (out.mean(dim=(1, 2, 3)) * 255).round()
    assert torch.allclose(level, torch.arange(12) * 8.0, atol=2.0)
    with pytest.raises(ValueError, match="the map says"):
        dte.decode_archive(path, 13)


@pytest.mark.parametrize("workers", [1, 4, 16])
def test_parallel_decode_keeps_order_and_pixels(tmp_path: Path, workers: int) -> None:
    ids = [f"c_{i:06d}" for i in range(7)]
    for i, k in enumerate(ids):
        _archive(tmp_path / f"{k}.zip", 3 + i)
    clips = {k: {"dota_frames": 3 + i} for i, k in enumerate(ids)}
    out = list(dte.decoded_clips(tmp_path, ids, clips, workers))
    assert [k for k, _ in out] == ids
    for i, (k, frames) in enumerate(out):
        assert torch.equal(frames, dte.decode_archive(tmp_path / f"{k}.zip", 3 + i))


class _LastMinusFirst(nn.Module):
    def forward(self, pixels: Tensor) -> Tensor:  # (B, 3, 16, H, W)
        return torch.stack([pixels[:, 0, -1, 0, 0], pixels[:, 0, 0, 0, 0]], dim=1)


def test_motion_clip_spans_15_frames_at_10_fps() -> None:
    frames = torch.arange(60.0).view(60, 1, 1, 1).expand(60, 3, 4, 4) / 60
    mean, std = constants.VIDEOMAE_IMAGE_MEAN[0], constants.VIDEOMAE_IMAGE_STD[0]
    rows = nexar_extract.motion_rows(
        frames, _LastMinusFirst(), torch.device("cpu"), 8, stride=STRIDE,
        clip_step=constants.DOTA_VIDEOMAE_FRAME_STEP)
    back = np.rint((rows * std + mean) * 60).astype(int)
    assert len(back) == 20
    assert back[:, 0].tolist() == list(range(0, 60, STRIDE))  # row k ends at frame 3k
    assert back[10].tolist() == [30, 15]  # 1.5 s back at 10 fps
    assert back[2].tolist() == [6, 0]  # clamped at the first frame


class _FakeClip(nn.Module):
    def forward(self, pixel_values: Tensor) -> object:
        return type("Out", (), {"image_embeds": pixel_values.mean(dim=(2, 3))})()


def test_extract_end_to_end_is_row_aligned_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(tmp_path / "a_000001.zip", 25)
    monkeypatch.setattr(dte, "train_clips", lambda _: {"a_000001": {"dota_frames": 25}})
    monkeypatch.setattr(dte, "load_pretrained_encoder", lambda _: _FakeClip())
    monkeypatch.setattr(dte, "load_pretrained", lambda *_: _LastMinusFirst())
    argv = ["--archive-dir", str(tmp_path), "--clip-out", str(tmp_path / "x"),
            "--video-out", str(tmp_path / "u"), "--report", str(tmp_path / "r.json"),
            "--device", "cpu"]
    dte.main(argv)
    x, u = (np.load(tmp_path / d / "a_000001.npy") for d in ("x", "u"))
    assert x.shape == (9, 3) and u.shape == (9, 2)
    assert np.all(np.diff(x[:, 0]) > 0) and np.all(np.diff(u[:, 0]) > 0)
    manifest = json.loads((tmp_path / "u" / constants.VIDEO_MANIFEST_FILENAME).read_text())
    assert (manifest["clip_frame_step"], manifest["assumed_fps"], manifest["stride"]) == (1, 10, 3)
    dte.main(argv)
    assert json.loads((tmp_path / "r.json").read_text())["extracted"] == 0
