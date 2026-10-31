"""Off-the-shelf detector + tracker (PREREG_V6 §2.1): COCO YOLOv9-C, BoT-SORT with camera-motion
compensation and no ReID, at 10 fps. Frozen preprocessing: their boxes are inputs, never labels.

ultralytics is imported lazily (Colab only); :func:`track_video` takes the detector, tracker and
camera-motion estimator as objects so the bookkeeping is testable without it. The ultralytics
API used (``YOLO.predict``, ``BOTSORT.update`` -> ``[x1, y1, x2, y2, track_id, score, cls, idx]``,
``GMC.apply`` -> a ``2 x 3`` pixel warp) was checked against its documentation (context7,
2026-10-31); the version that ran is written into every track-cache manifest.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Protocol

import numpy as np

from core import constants
from core.agents.geometry import (
    LOG_BOX,
    LOG_CLS,
    LOG_COLS,
    LOG_CONF,
    LOG_DET,
    LOG_RAW,
    LOG_TID,
    TrackLog,
    normalize_boxes,
    pixel_to_normalized_affine,
)

LOGGER = logging.getLogger(__name__)

TRACK_OUT_COLS = 8  # BYTETracker / BOTSORT output: x1 y1 x2 y2 track_id score cls idx


class Tracker(Protocol):
    def update(self, results: Any, img: np.ndarray | None = None) -> np.ndarray: ...


class MotionEstimator(Protocol):
    def apply(self, raw_frame: np.ndarray) -> np.ndarray: ...


Detect = Callable[[list[np.ndarray]], list[Any]]  # BGR frames -> per-frame numpy Boxes


def ultralytics_version() -> str:
    import ultralytics  # pyright: ignore[reportMissingImports]

    return str(ultralytics.__version__)


def load_detector(device: str, weights: str = constants.V6_DETECTOR_WEIGHTS) -> Detect:
    """``YOLO(weights).predict`` on a batch of BGR frames: the six traffic classes, conf 0.10."""
    from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

    model = YOLO(weights)
    half = device != "cpu"

    def detect(frames: list[np.ndarray]) -> list[Any]:
        results = model.predict(
            frames,
            imgsz=constants.V6_DETECTOR_IMGSZ,
            conf=constants.V6_DETECTOR_CONF,
            classes=list(constants.V6_COCO_CLASSES),
            half=half,
            device=device,
            verbose=False,
        )
        return [r.boxes.cpu().numpy() for r in results]

    return detect


def make_tracker() -> Tracker:
    """A fresh BoT-SORT (one per video): CMC on, ReID off, lost after 10 detection frames."""
    from ultralytics.trackers.bot_sort import BOTSORT  # pyright: ignore[reportMissingImports]
    from ultralytics.utils import IterableSimpleNamespace  # pyright: ignore[reportMissingImports]
    from ultralytics.utils.checks import check_yaml  # pyright: ignore[reportMissingImports]

    path = check_yaml(constants.V6_TRACKER)
    try:  # ultralytics >= 8.3.1xx
        from ultralytics.utils import YAML  # pyright: ignore[reportMissingImports]

        raw = YAML.load(path)
    except ImportError:  # older releases
        from ultralytics.utils import yaml_load  # pyright: ignore[reportMissingImports]

        raw = yaml_load(path)
    args = IterableSimpleNamespace(
        **{**raw, "with_reid": False, "gmc_method": constants.V6_GMC_METHOD}
    )
    try:
        tracker = BOTSORT(args, frame_rate=constants.V6_DET_FPS)
    except TypeError:  # releases whose constructor takes only args
        tracker = BOTSORT(args)
    if not hasattr(tracker, "max_time_lost"):
        raise RuntimeError("this ultralytics BOTSORT has no max_time_lost; pin the tested release")
    tracker.max_time_lost = constants.V6_TRACK_LOST_FRAMES
    return tracker  # type: ignore[no-any-return]


def make_motion_estimator() -> MotionEstimator:
    from ultralytics.trackers.utils.gmc import GMC  # pyright: ignore[reportMissingImports]

    return GMC(method=constants.V6_GMC_METHOD)  # type: ignore[no-any-return]


def detection_frames(num_frames: int, det_step: int) -> list[int]:
    """Native frames the detector runs on: every ``det_step``-th (10 fps)."""
    return list(range(0, num_frames, det_step))


def _batches(
    items: Iterable[tuple[int, np.ndarray]], size: int
) -> Iterator[list[tuple[int, np.ndarray]]]:
    batch: list[tuple[int, np.ndarray]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def track_video(
    frames: Iterable[tuple[int, np.ndarray]],
    num_frames: int,
    detect: Detect,
    tracker: Tracker,
    motion: MotionEstimator,
    batch_size: int = constants.V6_DETECTOR_BATCH,
) -> TrackLog:
    """Detect + track one video.

    ``frames`` yields ``(native_index, RGB uint8 frame)`` for the detection frames, in order.
    Detection is batched; tracking and the camera affine run frame by frame in order.
    """
    det_frames: list[int] = []
    affines: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    for batch in _batches(frames, batch_size):
        bgr = [np.ascontiguousarray(f[:, :, ::-1]) for _, f in batch]
        boxes_list = detect(bgr)
        for (index, _), img, boxes in zip(batch, bgr, boxes_list, strict=True):
            h, w = img.shape[:2]
            if size is None:
                size = (w, h)
            elif size != (w, h):
                raise ValueError(f"frame {index} is {w}x{h}, the video started at {size}")
            if det_frames and index <= det_frames[-1]:
                raise ValueError("detection frames must be strictly increasing")
            warp = np.asarray(motion.apply(img), dtype=np.float64)
            affines.append(
                np.eye(2, 3) if not det_frames else pixel_to_normalized_affine(warp, w, h)
            )
            out = np.asarray(tracker.update(boxes, img), dtype=np.float64).reshape(
                -1, TRACK_OUT_COLS
            )
            if len(out):
                raw = np.asarray(boxes.xyxy, dtype=np.float64)[out[:, 7].astype(int)]
                r = np.zeros((len(out), LOG_COLS))
                r[:, LOG_DET] = len(det_frames)
                r[:, LOG_TID] = out[:, 4]
                r[:, LOG_BOX] = normalize_boxes(out[:, :4], w, h)
                r[:, LOG_CONF] = out[:, 5]
                r[:, LOG_CLS] = out[:, 6]
                r[:, LOG_RAW] = normalize_boxes(raw, w, h)
                rows.append(r)
            det_frames.append(index)
    if size is None:
        raise ValueError("no frames to track")
    return TrackLog(
        det_frames=np.asarray(det_frames, dtype=np.int64),
        affine=np.stack(affines),
        rows=np.concatenate(rows) if rows else np.zeros((0, LOG_COLS)),
        size=size,
        num_frames=num_frames,
    )


def track_manifest(det_step: int, source: str) -> dict[str, Any]:
    return {
        "detector": constants.V6_DETECTOR_WEIGHTS,
        "imgsz": constants.V6_DETECTOR_IMGSZ,
        "conf": constants.V6_DETECTOR_CONF,
        "classes": list(constants.V6_COCO_CLASSES),
        "tracker": constants.V6_TRACKER,
        "with_reid": False,
        "gmc": constants.V6_GMC_METHOD,
        "lost_frames": constants.V6_TRACK_LOST_FRAMES,
        "det_fps": constants.V6_DET_FPS,
        "det_step": det_step,
        "source": source,
        "ultralytics": ultralytics_version(),
        "rule": "core/docs/v6/PREREG_V6.md §2.1",
    }


__all__ = [
    "detection_frames",
    "load_detector",
    "make_motion_estimator",
    "make_tracker",
    "track_manifest",
    "track_video",
]
