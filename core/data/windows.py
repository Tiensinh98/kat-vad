"""Fixed-length windows over a clip-level feature cache (lesson **C28**, Phase 2a).

A corpus whose *clip length* predicts its label cannot be measured: on the
reconstructed DADA-2000, abnormal test clips run to at most 17 sampled frames
while all 107 clips of 18+ frames are normal, so a detector whose only input is
the frame count scores **micro AUC 0.8654** — within 0.009 of the best trained
arm (``core/docs/DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md`` §2).
``core.evaluate --equalize-length`` removes that channel at *scoring* time; this
module removes it at the *source*, by making every trained and scored item the
same length.

**The cache is not re-sharded.** Features stay one ``{source_id}.npy`` per clip,
exactly as the extractors write them; a window is a *slice* of one. So changing
the window geometry is a cheap re-run of a preprocessor, not a re-extraction,
and no frame is ever stored twice. The mapping lives in a fifth, **optional**
dataset file:

    windows.json   {window_id: {"source": video_id, "start": int, "end": int}}

When it is absent -- every corpus shipped before Phase 2a -- :class:`FeatureSlicer`
is the identity and every loader behaves exactly as it did.

Two invariants this module exists to hold:

* **No padding, ever.** A window is only ever a real slice of real frames. Padding
  short clips would fabricate evidence and interact with ``ConvScoreHead``'s
  ``padding_mode="replicate"`` (lesson **C27**).
* **Appearance and flow are sliced identically.** ``e_O`` is cached per source
  clip too, so a window must index both with the same ``[start, end)`` -- an
  offset between the two branches is lesson **C13** in miniature.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import constants

LOGGER = logging.getLogger(__name__)

_KEY_SOURCE = "source"
_KEY_START = "start"
_KEY_END = "end"


@dataclass(frozen=True)
class Window:
    """A ``[start, end)`` slice of the source clip's cached feature rows."""

    source: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Window start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"Window must be non-empty: got [{self.start}, {self.end}) "
                f"for source {self.source!r}"
            )

    @property
    def length(self) -> int:
        return self.end - self.start


def window_id(source: str, index: int) -> str:
    """Deterministic window id, sortable within a source clip."""
    return f"{source}{constants.WINDOW_ID_SEPARATOR}{index:03d}"


def plan_windows(
    source: str, length: int, window_length: int, window_stride: int
) -> list[Window]:
    """Every full-length window of a ``length``-frame clip, left to right.

    Returns ``[]`` when the clip is shorter than one window -- the caller decides
    whether that is a drop or an error. The last window is emitted only if it
    fits entirely inside the clip, so no window is ever padded or truncated.
    """
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}")
    if window_stride <= 0:
        raise ValueError(f"window_stride must be positive, got {window_stride}")
    starts = range(0, max(0, length - window_length + 1), window_stride)
    return [Window(source, s, s + window_length) for s in starts]


def cap_windows(windows: list[Window], max_per_clip: int) -> list[Window]:
    """Keep at most ``max_per_clip`` windows, **evenly spaced** across the clip.

    Without this, a fixed hop hands every source clip a window count proportional
    to its length -- so on a corpus whose normal clips are 3x longer than its
    abnormal ones (DADA-2000: raw median 139 vs 49), windowing *manufactures* a
    class imbalance the clip-level corpus did not have. Measured 2026-09-13: a
    32-frame window at stride 2 produced 3,244 normal windows against 327
    abnormal ones, and the constant-score clip oracle rose from 0.9086 to 0.9766.

    Evenly spaced, not the first ``N``: the anomaly sits at the end of a DADA
    clip, and keeping the head would drop it.
    """
    if max_per_clip <= 0:
        raise ValueError(f"max_per_clip must be positive, got {max_per_clip}")
    if len(windows) <= max_per_clip:
        return windows
    picks = np.linspace(0, len(windows) - 1, max_per_clip).round().astype(int)
    return [windows[i] for i in picks]


def slice_labels(labels: list[int], window: Window) -> list[int]:
    """The window's own frame labels, asserting the slice is in range."""
    if window.end > len(labels):
        raise ValueError(
            f"Window {window.start}:{window.end} exceeds {window.source!r}'s "
            f"{len(labels)} label rows -- windows must never cross a clip boundary"
        )
    return labels[window.start : window.end]


def load_windows(data_dir: Path) -> dict[str, Window] | None:
    """Read ``windows.json``; ``None`` when the corpus is not windowed."""
    path = data_dir / constants.WINDOWS_FILENAME
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    windows = {
        wid: Window(str(entry[_KEY_SOURCE]), int(entry[_KEY_START]), int(entry[_KEY_END]))
        for wid, entry in payload.items()
    }
    lengths = {w.length for w in windows.values()}
    LOGGER.info(
        "Loaded %d windows over %d source clips from %s (length%s %s)",
        len(windows), len({w.source for w in windows.values()}), path,
        "" if len(lengths) == 1 else "s", sorted(lengths),
    )
    if len(lengths) > 1:
        LOGGER.warning(
            "Windows are not all the same length (%s) -- clip length can still "
            "carry the label (lesson C28)", sorted(lengths),
        )
    return windows


def windows_payload(windows: dict[str, Window]) -> dict[str, dict[str, object]]:
    """``windows.json``'s JSON payload for :func:`core.data.dataset_files.write_windows`."""
    return {
        wid: {_KEY_SOURCE: w.source, _KEY_START: w.start, _KEY_END: w.end}
        for wid, w in sorted(windows.items())
    }


class FeatureSlicer:
    """Resolves an item id to cached rows, slicing when the corpus is windowed.

    One object serves the appearance cache, the flow cache and the KNN keys, so
    the three cannot drift apart. ``windows=None`` is the identity path: the id
    *is* the source id and the whole array is returned.
    """

    def __init__(self, windows: dict[str, Window] | None = None) -> None:
        self.windows = windows

    @property
    def is_windowed(self) -> bool:
        return self.windows is not None

    def window_of(self, item_id: str) -> Window | None:
        if self.windows is None:
            return None
        try:
            return self.windows[item_id]
        except KeyError as exc:
            raise KeyError(
                f"{item_id!r} is not in windows.json; a windowed corpus must key "
                "labels_train/frame_labels_test/meta by window id"
            ) from exc

    def source_of(self, item_id: str) -> str:
        """The cache filename stem backing ``item_id`` (the id itself if unwindowed)."""
        window = self.window_of(item_id)
        return item_id if window is None else window.source

    def load(self, cache_dir: Path, item_id: str, suffix: str = ".npy") -> np.ndarray:
        """``cache_dir/{source}{suffix}``, sliced to the window when there is one.

        ``suffix`` covers the sibling caches keyed by the same id -- RAFT's
        ``.stats.npy`` -- so every per-frame array backing an item is sliced by
        the one window, never by two different ones.
        """
        window = self.window_of(item_id)
        source = item_id if window is None else window.source
        rows: np.ndarray = np.load(cache_dir / f"{source}{suffix}")
        if window is None:
            return rows
        if window.end > len(rows):
            raise ValueError(
                f"Window {item_id} needs rows [{window.start}, {window.end}) but "
                f"{source}{suffix} has {len(rows)} -- the cache was built at a "
                "different stride than the windows (lesson C2)"
            )
        return rows[window.start : window.end]


__all__ = [
    "FeatureSlicer",
    "Window",
    "cap_windows",
    "load_windows",
    "plan_windows",
    "slice_labels",
    "window_id",
    "windows_payload",
]
