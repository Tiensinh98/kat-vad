"""Crash-safe, resumable per-video feature caches (CLIP features, flow ``e_O``).

One ``.npy`` per video id is both the unit of work and the unit of resume: a run
that dies halfway is restarted by rerunning the same command. Two things make
that true rather than merely plausible, and both live here so the CLIP and RAFT
extractors cannot drift apart:

* **Atomic writes.** ``np.save`` straight onto the target leaves a truncated
  file when the process is killed mid-write — routine on Colab, and Drive's
  FUSE layer can fail a write *after* creating the file (lesson C10). Writing to
  a ``.part`` sibling and renaming makes the target either absent or complete,
  never half.
* **Reading the header, not just ``exists()``.** A truncated ``.npy`` left by an
  older run passes an existence check, is skipped by every resume that follows,
  and finally surfaces hours later as a crash inside eval. Cheap to detect: the
  header carries shape and dtype, so ``mmap_mode`` reads it without touching the
  payload — which matters when the cache lives on Drive.

Resume does **not** re-derive the stride or transform a cached file was built
with; nothing on disk records them (lesson C2). Changing either means
``--force``, not a rerun.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from core import constants

LOGGER = logging.getLogger(__name__)


def part_path(target: Path) -> Path:
    """Sibling path an in-flight write goes to."""
    return target.with_name(target.name + constants.CACHE_PART_SUFFIX)


def save_array(target: Path, array: np.ndarray) -> None:
    """Write ``array`` to ``target`` atomically."""
    staged = part_path(target)
    with staged.open("wb") as handle:  # explicit handle: np.save would append .npy
        np.save(handle, array)
    staged.replace(target)


def is_complete(target: Path) -> bool:
    """True if ``target`` holds a readable, non-empty 2-D feature array.

    Header-only read, so this stays cheap over a slow filesystem.
    """
    if not target.exists():
        return False
    try:
        array = np.load(target, mmap_mode="r")
    except (OSError, ValueError, EOFError):  # truncated, empty, or not a .npy
        return False
    return bool(array.ndim == 2 and array.shape[0] > 0)


def pending_items(
    sources: dict[str, Path], output_dir: Path, force: bool = False
) -> list[tuple[str, Path]]:
    """``(video_id, source)`` still to extract, sorted; logs the resume state.

    Outputs that exist but do not read back are treated as unwritten and
    re-extracted, with a warning — they are the partial writes above, and
    skipping them is how a short cache reaches eval.
    """
    pending: list[tuple[str, Path]] = []
    done = 0
    partial: list[str] = []
    for video_id, source in sorted(sources.items()):
        target = output_dir / f"{video_id}.npy"
        if not force and is_complete(target):
            done += 1
            continue
        if not force and target.exists():
            partial.append(video_id)
        part_path(target).unlink(missing_ok=True)
        pending.append((video_id, source))
    if partial:
        LOGGER.warning(
            "%d cached files are unreadable (interrupted write) and will be "
            "re-extracted: %s",
            len(partial),
            sorted(partial)[:5],
        )
    LOGGER.info(
        "Resume: %d/%d already cached in %s, %d to extract",
        done,
        len(sources),
        output_dir,
        len(pending),
    )
    return pending


def read_ids_file(path: Path) -> set[str]:
    """Video ids, one per line; blank lines ignored.

    Shared by both extractors: ``--ids-file`` scopes a run to one split, and the
    CLIP and flow caches must be scoped by the same rule or they disagree about
    which videos exist.
    """
    ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}
    ids.discard("")
    if not ids:
        raise ValueError(f"{path} contains no video ids")
    return ids


def select_ids(
    sources: dict[str, Path],
    video_ids: set[str] | None,
    origin: Path,
    what: str = "source",
) -> dict[str, Path]:
    """Restrict ``sources`` to ``video_ids``; a requested id with no source raises.

    A silently short cache does not fail here -- it fails hours later as a
    missing-``.npy`` crash inside training or eval, by which point the run that
    caused it is gone. ``what`` names the thing that is missing ("frame folder",
    "video") so the message points at the directory the caller actually passed.
    """
    if video_ids is None:
        return sources
    missing = sorted(video_ids - set(sources))
    if missing:
        raise ValueError(
            f"{len(missing)} requested ids have no {what} under {origin}: {missing[:5]}"
        )
    return {vid: path for vid, path in sources.items() if vid in video_ids}


class Progress:
    """Per-item ``[i/N] elapsed .. eta ..`` for runs long enough to worry about."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.started = time.monotonic()

    def step(self) -> str:
        """Record one finished item and render the progress prefix."""
        self.done += 1
        elapsed = time.monotonic() - self.started
        remaining = elapsed / self.done * (self.total - self.done)
        return (
            f"[{self.done}/{self.total}] "
            f"elapsed {format_duration(elapsed)}, eta {format_duration(remaining)}"
        )


def format_duration(seconds: float) -> str:
    """``93m07s`` style — readable in a Colab log without a date library."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"
