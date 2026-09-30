"""Streamed tar -> bounded CLIP batches (core/tools/stream_frames_clip.py)."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision.io import encode_jpeg

from core import constants
from core.tools import extract_clip_features
from core.tools.extract_clip_features import ImageEncoder
from core.tools.stream_frames_clip import ConcatReader, frame_member, stream_video_batches

CPU = torch.device("cpu")
SUBDIR = "images"
FRAMES = {"000001": 5, "000002": 3, "000003": 4, "000004": 2}
PART_BYTES = 4096  # small enough that every test tar spans several parts


def _jpeg(seed: int, size: int = 48) -> bytes:
    gen = torch.Generator().manual_seed(seed)
    return bytes(encode_jpeg((torch.rand(3, size, size, generator=gen) * 255).to(torch.uint8)))


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _members(order: list[tuple[str, int]]) -> list[tuple[str, bytes]]:
    return [
        (f"1-10/3/{vid}/{SUBDIR}/{i:06d}.jpg", _jpeg(int(vid) * 100 + i)) for vid, i in order
    ]


def _write_parts(root: Path, members: list[tuple[str, bytes]]) -> list[Path]:
    """One gzipped tar, cut into ``PART_BYTES`` pieces like CAP's ``{group}.part_*``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        _add(tar, "1-10/README.txt", b"not a frame")
        for name, data in members:
            _add(tar, name, data)
    blob = buf.getvalue()
    root.mkdir(parents=True, exist_ok=True)
    parts = []
    for k, start in enumerate(range(0, len(blob), PART_BYTES)):
        part = root / f"g.part_{k:03d}"
        part.write_bytes(blob[start : start + PART_BYTES])
        parts.append(part)
    assert len(parts) > 1
    return parts


def _contiguous() -> list[tuple[str, bytes]]:
    return _members([(vid, i) for vid, n in FRAMES.items() for i in range(n)])


def _disk_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


@pytest.fixture(scope="module")
def tiny_encoder() -> object:
    """Random-weight CLIP vision tower — no download, 64-d projection."""
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    config = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, image_size=constants.CROP_SIZE,
        patch_size=32, projection_dim=64,
    )
    torch.manual_seed(0)
    return CLIPVisionModelWithProjection(config).eval()


class TestConcatReader:
    def test_reads_parts_as_one_stream(self, tmp_path: Path) -> None:
        paths = []
        for k, chunk in enumerate((b"abc", b"", b"defg")):
            paths.append(tmp_path / f"p{k}")
            paths[-1].write_bytes(chunk)
        with ConcatReader(paths) as reader:
            assert reader.read() == b"abcdefg"


class TestFrameMember:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("1-10/3/000001/images/000000.jpg", ("000001", "000000.jpg")),
            ("images/000000.jpg", None),  # no video folder above the subdir
            ("1-10/3/000001/other/000000.jpg", None),
            ("1-10/3/../images/x.jpg", None),  # path escape
            ("1-10/README.txt", None),
        ],
    )
    def test_parse(self, name: str, expected: tuple[str, str] | None) -> None:
        assert frame_member(name, SUBDIR) == expected


class TestStream:
    def test_batches_are_bounded_and_videos_whole(self, tmp_path: Path) -> None:
        parts = _write_parts(tmp_path / "parts", _contiguous())
        work = tmp_path / "work"
        one_video = max(len(_jpeg(0)) * n for n in FRAMES.values()) * 2  # jpeg size varies
        batches: list[dict[str, int]] = []
        peaks: list[int] = []

        def on_batch(ready: Path) -> None:
            batches.append({p.name: len(list((p / SUBDIR).iterdir())) for p in ready.iterdir()})
            peaks.append(_disk_bytes(work))

        census = stream_video_batches(parts, work, SUBDIR, one_video, on_batch)
        assert census == FRAMES
        assert len(batches) > 1
        merged = {vid: n for b in batches for vid, n in b.items()}
        assert merged == FRAMES and sum(len(b) for b in batches) == len(FRAMES)
        assert max(peaks) <= 3 * one_video  # one full batch + the video being written
        assert _disk_bytes(work) == 0

    def test_cached_videos_are_counted_not_written(self, tmp_path: Path) -> None:
        parts = _write_parts(tmp_path / "parts", _contiguous())
        seen: list[str] = []
        census = stream_video_batches(
            parts, tmp_path / "work", SUBDIR, 1e12,
            lambda ready: seen.extend(p.name for p in ready.iterdir()),
            skip_ids={"000002", "000004"},
        )
        assert census == FRAMES
        assert sorted(seen) == ["000001", "000003"]

    def test_non_contiguous_tar_is_refused(self, tmp_path: Path) -> None:
        order = [("000001", 0), ("000002", 0), ("000001", 1)]
        parts = _write_parts(tmp_path / "parts", _members(order))
        with pytest.raises(ValueError, match="reappears"):
            stream_video_batches(parts, tmp_path / "work", SUBDIR, 1e12, lambda ready: None)

    def test_features_equal_extracting_the_untarred_folders(
        self, tmp_path: Path, tiny_encoder: ImageEncoder
    ) -> None:
        """The streamed path is the --frames-dir path: same bytes, same transform, same array."""
        members = _contiguous()
        parts = _write_parts(tmp_path / "parts", members)
        flat = tmp_path / "flat"
        for name, data in members:
            target = flat / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        ref, got = tmp_path / "ref", tmp_path / "got"

        def encode(frames_dir: Path, out: Path) -> None:
            extract_clip_features.extract_frame_directory(
                frames_dir, out, tiny_encoder, CPU, stride=1, batch_size=3, subdir=SUBDIR,
                center_crop=False,
            )

        encode(flat, ref)
        stream_video_batches(  # 1 byte -> one video per batch
            parts, tmp_path / "work", SUBDIR, 1.0, lambda ready: encode(ready, got)
        )
        for vid in FRAMES:
            np.testing.assert_array_equal(np.load(got / f"{vid}.npy"), np.load(ref / f"{vid}.npy"))
