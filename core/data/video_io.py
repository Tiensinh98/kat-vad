"""Strided video-frame reading via PyAV (decord replacement, macOS-safe).

Shared by the CLIP feature extractor and the RAFT flow extractor so both
pipelines sample the *same* frames: raw indices ``range(0, N, stride)``,
matching :func:`core.data.msad.num_sampled_frames` label alignment.

Datasets that ship **extracted frames** instead of videos (TAD, DoTA) are read
through the frame-folder helpers below. They sample the same raw indices, so a
label vector built with ``num_sampled_frames`` lines up either way.
"""

from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np
import torch
from torchvision.io import ImageReadMode, read_image

LOGGER = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def read_sampled_frames(path: Path, stride: int) -> np.ndarray:
    """Decode ``path`` and keep every ``stride``-th frame.

    Returns ``(L, H, W, 3)`` uint8 RGB. Raises ``ValueError`` on empty video.
    """
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index % stride == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames)


def list_videos(videos_dir: Path) -> list[Path]:
    """All video files under ``videos_dir`` (recursive), sorted by name."""
    paths = sorted(
        p for p in videos_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No videos found under {videos_dir}")
    LOGGER.info("Found %d videos under %s", len(paths), videos_dir)
    return paths


def video_id_from_path(path: Path) -> str:
    """Video id for a video file, a frame folder, or an annotation path.

    Strips a *video* extension only (``01_Accident_106.mp4`` -> that stem);
    folder names without one are kept whole, so DoTA ids that carry dots or
    underscores (``0RJPQ_97dcs_000387``) survive untouched.
    """
    return path.stem if path.suffix.lower() in VIDEO_EXTENSIONS else path.name


def list_frame_images(folder: Path, subdir: str | None = None) -> list[Path]:
    """Sorted image files directly inside ``folder`` (or ``folder/subdir``).

    Frame extractors emit zero-padded names (``000000.jpg``), so lexicographic
    order is temporal order. Non-padded names would sort wrong -- callers own
    that contract.
    """
    target = folder / subdir if subdir else folder
    if not target.is_dir():
        return []
    return sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def list_frame_folders(root: Path, subdir: str | None = None) -> list[Path]:
    """Per-video frame folders under ``root``, sorted by video id.

    With ``subdir`` (DoTA: ``frames/{video_id}/images/*.jpg``) the returned
    path is the *video* folder, not the image folder, so
    :func:`video_id_from_path` yields the id in both layouts. Without it, any
    directory that directly holds images qualifies, at any depth (TAD:
    ``frames/abnormal/{video_id}.mp4/*.jpg``).
    """
    if not root.is_dir():
        raise ValueError(f"Frames directory {root} does not exist")
    if subdir:
        folders = [p.parent for p in sorted(root.rglob(subdir)) if p.is_dir()]
    else:
        folders = [
            p for p in sorted(root.rglob("*")) if p.is_dir() and list_frame_images(p)
        ]
    if not folders:
        raise ValueError(f"No frame folders found under {root}")
    LOGGER.info("Found %d frame folders under %s", len(folders), root)
    return sorted(folders, key=video_id_from_path)


def read_images(paths: list[Path]) -> np.ndarray:
    """Decode image files to one ``(L, H, W, 3)`` uint8 RGB array."""
    if not paths:
        raise ValueError("No image paths given")
    images = [read_image(str(p), mode=ImageReadMode.RGB) for p in paths]
    return torch.stack(images).permute(0, 2, 3, 1).numpy()


def read_sampled_frames_from_dir(
    folder: Path, stride: int, subdir: str | None = None
) -> np.ndarray:
    """Every ``stride``-th image of a frame folder as ``(L, H, W, 3)`` uint8 RGB.

    Loads the whole clip at once -- fine for short, small-resolution clips; the
    CLIP extractor streams in batches instead (:func:`encode_frame_dir`).

    The result is made **C-contiguous**, which :func:`read_images` is not: it
    returns a permuted view. Values are identical either way, but a
    non-contiguous buffer sends torch's batched convolutions down a different
    kernel path, and RAFT flow statistics then differ from the video path's by
    ~3e-5 on the same pixels. That is numerically irrelevant and operationally
    poisonous: ``cache/flow/v1/{DATASET}/`` is supposed to mean the same thing
    whether the dataset shipped videos (MSAD) or frames (TAD), and a cache that
    quietly depends on its source is a cache you cannot compare across datasets.
    Measured 2026-09-02; ``core/tests/test_extractors.py`` asserts bit-equality.
    """
    paths = list_frame_images(folder, subdir)
    if not paths:
        raise ValueError(f"No frame images under {folder}")
    return np.ascontiguousarray(read_images(paths[::stride]))


__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "list_frame_folders",
    "list_frame_images",
    "list_videos",
    "read_images",
    "read_sampled_frames",
    "read_sampled_frames_from_dir",
    "video_id_from_path",
]
