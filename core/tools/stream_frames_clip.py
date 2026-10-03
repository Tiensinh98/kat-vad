"""Stream a split ``.tar.gz`` of per-video frame folders into a CLIP cache in bounded batches.

CAP-DATA groups ship as ``{group}.part_*`` pieces of one gzipped tar (group ``11`` is ~96 GB
extracted). Extracting a whole group before encoding needs that much VM disk; this tool reads
the parts as one stream (never merged, never fully extracted), writes each video's frames to
``work_dir``, and as soon as the finished videos on disk reach ``--max-batch-gb`` it encodes
them with :func:`core.tools.extract_clip_features.extract_frame_directory` -- the same code
path, transform and resume rule as ``extract_clip_features --frames-dir`` -- then deletes them.
Peak disk = one batch + the video being written.

- A member belongs to video ``v`` when its path is ``.../v/{subdir}/{frame}``; anything else
  (README, directories) is skipped. Output paths are rebuilt from the validated video id and
  frame basename, so a member name can never escape ``work_dir``.
- The tar must keep each video's frames contiguous (``tar -c`` of a directory tree does); a
  video that reappears after its folder was closed raises instead of being silently split.
- Videos already in ``--output-dir`` are counted but not written (resume after a disconnect).
- ``--census-json`` records frames per video as seen in the stream (compare with the
  annotation afterwards).

CLI::

    python -m core.tools.stream_frames_clip \\
        --parts data/MMAU/raw/CAP-DATA_chunks/11/11.part_* --work-dir /content/mmau/work \\
        --frames-subdir images --dataset MMAU_CAP --output-dir cache/clip/MMAU_CAP_s1_ncc \\
        --stride 1 --no-center-crop --batch-size 256 [--max-batch-gb 8] \\
        [--census-json outputs/.../census_11.json]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import shutil
import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import IO

from core import constants
from core.device import resolve_device
from core.tools.extract_clip_features import extract_frame_directory, load_pretrained_encoder

LOGGER = logging.getLogger(__name__)

OPEN_DIR, READY_DIR = "open", "ready"
VIDEO_ID_PATTERN = re.compile(r"^\w[\w.-]*$")  # no "." or ".." (path escape)
COPY_CHUNK_BYTES = 1 << 20


class ConcatReader(io.RawIOBase):
    """Read-only stream over several files in order, as if they were one (``cat part_*``)."""

    def __init__(self, paths: Iterable[Path]) -> None:
        super().__init__()
        self._paths = list(paths)
        self._index = 0
        self._handle: IO[bytes] | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: memoryview) -> int:  # type: ignore[override]
        while self._index < len(self._paths):
            if self._handle is None:
                self._handle = self._paths[self._index].open("rb")
            n = self._handle.readinto(buffer)  # type: ignore[attr-defined]
            if n:
                return int(n)
            self._handle.close()
            self._handle = None
            self._index += 1
        return 0

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def frame_member(name: str, subdir: str) -> tuple[str, str] | None:
    """``(video_id, frame_basename)`` of a tar member ``.../video_id/subdir/frame``, else None."""
    parts = PurePosixPath(name).parts
    if len(parts) < 3 or parts[-2] != subdir:
        return None
    video_id, frame = parts[-3], parts[-1]
    if not VIDEO_ID_PATTERN.match(video_id) or not VIDEO_ID_PATTERN.match(frame):
        return None
    return video_id, frame


class _Batcher:
    """Moves closed videos from ``open/`` to ``ready/`` and flushes ``ready/`` when full."""

    def __init__(
        self, work_dir: Path, subdir: str, max_bytes: float, on_batch: Callable[[Path], None]
    ) -> None:
        self.open_dir, self.ready_dir = work_dir / OPEN_DIR, work_dir / READY_DIR
        self.subdir, self.max_bytes, self.on_batch = subdir, max_bytes, on_batch
        for d in (self.open_dir, self.ready_dir):
            shutil.rmtree(d, ignore_errors=True)  # stale half-written frames from a killed run
            d.mkdir(parents=True)
        self.ready_bytes = 0
        self.batches = 0

    def write(self, video_id: str, frame: str, data: IO[bytes]) -> int:
        target = self.open_dir / video_id / self.subdir / frame
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            shutil.copyfileobj(data, out, COPY_CHUNK_BYTES)
        return target.stat().st_size

    def close(self, video_id: str, size: int) -> None:
        src = self.open_dir / video_id
        if not src.is_dir():  # skipped (already cached) video: nothing on disk
            return
        src.rename(self.ready_dir / video_id)
        self.ready_bytes += size
        if self.ready_bytes >= self.max_bytes:
            self.flush()

    def flush(self) -> None:
        if not any(self.ready_dir.iterdir()):
            return
        self.batches += 1
        LOGGER.info("Batch %d: encoding %.2f GB of frames", self.batches,
                    self.ready_bytes / constants.BYTES_PER_GB)
        self.on_batch(self.ready_dir)
        shutil.rmtree(self.ready_dir)
        self.ready_dir.mkdir()
        self.ready_bytes = 0


def stream_video_batches(
    parts: list[Path],
    work_dir: Path,
    subdir: str,
    max_bytes: float,
    on_batch: Callable[[Path], None],
    skip_ids: set[str] | None = None,
    keep_ids: set[str] | None = None,
) -> dict[str, int]:
    """Stream ``parts`` (one gzipped tar) into bounded batches; return frames per video.

    ``skip_ids`` are counted, not written. With ``keep_ids`` only those videos are written, and
    the stream stops as soon as every one of them is closed (the census then covers only the
    videos read so far).
    """
    if not parts:
        raise ValueError("No tar parts given")
    skip = skip_ids or set()
    wanted = None if keep_ids is None else set(keep_ids) - skip
    batcher = _Batcher(work_dir, subdir, max_bytes, on_batch)
    census: dict[str, int] = {}
    closed: set[str] = set()
    current: str | None = None
    current_bytes = 0
    with ConcatReader(parts) as raw, tarfile.open(fileobj=raw, mode="r|gz") as tar:
        for member in tar:
            parsed = frame_member(member.name, subdir) if member.isfile() else None
            if parsed is None:
                continue
            video_id, frame = parsed
            if video_id != current:
                if video_id in closed:
                    raise ValueError(
                        f"video {video_id} reappears after its folder was closed: the tar does "
                        "not keep each video contiguous, so a bounded batch would split it"
                    )
                if current is not None:
                    batcher.close(current, current_bytes)
                    closed.add(current)
                if wanted is not None and wanted <= closed:
                    current = None
                    break
                current, current_bytes = video_id, 0
            census[video_id] = census.get(video_id, 0) + 1
            if video_id in skip or (wanted is not None and video_id not in wanted):
                continue
            data = tar.extractfile(member)
            if data is None:
                continue
            current_bytes += batcher.write(video_id, frame, data)
    if current is not None:
        batcher.close(current, current_bytes)
    batcher.flush()
    LOGGER.info("Streamed %d videos (%d skipped as cached) in %d batches",
                len(census), len(skip & census.keys()), batcher.batches)
    return census


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--parts", type=Path, nargs="+", required=True,
                        help="the tar.gz pieces, in order (sorted if given as a glob)")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="local scratch for one batch of frames (VM disk, not Drive)")
    parser.add_argument("--frames-subdir", default="images")
    parser.add_argument("--dataset", default=constants.MMAU_CAP_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: cache/clip/{dataset}")
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-center-crop", action="store_true",
                        help="anisotropic resize (the project's _ncc transform, lesson C2)")
    parser.add_argument("--max-batch-gb", type=float, default=constants.MMAU_STREAM_BATCH_GB)
    parser.add_argument("--census-json", type=Path, default=None,
                        help="write {video_id: frames seen in the stream} here")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    output_dir = args.output_dir or constants.CLIP_CACHE_DIR / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = load_pretrained_encoder(device)
    center_crop = not args.no_center_crop

    def encode(batch_dir: Path) -> None:
        extract_frame_directory(
            frames_dir=batch_dir, output_dir=output_dir, encoder=encoder, device=device,
            stride=args.stride, batch_size=args.batch_size, subdir=args.frames_subdir,
            center_crop=center_crop,
        )

    census = stream_video_batches(
        sorted(args.parts), args.work_dir, args.frames_subdir,
        args.max_batch_gb * constants.BYTES_PER_GB, encode,
        skip_ids={p.stem for p in output_dir.glob("*.npy")},
    )
    if args.census_json is not None:
        args.census_json.parent.mkdir(parents=True, exist_ok=True)
        args.census_json.write_text(json.dumps(census, indent=1, sort_keys=True))
    shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
