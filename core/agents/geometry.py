"""Box, track and camera-affine arithmetic for the agent branch (PREREG_V6 §2.1).

Every box here is normalized ``xyxy`` in ``[0, 1]`` of its frame, so a box from a 1280 x 720 DoTA
annotation, a CAP frame and a Nexar frame compare directly (all are 16:9 and squash-resized, never
cropped, on the way into CLIP and VideoMAE). Frames are native frame indices of the video; the
detector ran on a sub-grid of them (``det_frames``, 10 fps).

A track log (``TrackLog``) is what the detector + tracker wrote for one video: one row per tracked
box per detection frame, plus the camera affine between consecutive detection frames.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core import constants

# track-log row columns (one tracked box at one detection frame)
LOG_DET, LOG_TID = 0, 1
LOG_BOX = slice(2, 6)  # tracker (Kalman) box, normalized
LOG_CONF, LOG_CLS = 6, 7  # detection confidence, COCO class id
LOG_RAW = slice(8, 12)  # the raw detection box, normalized
LOG_COLS = 12
FP16_EXACT_INT = 2048  # every integer up to 2^11 is exact in float16


@dataclass
class TrackLog:
    """Detector + tracker output of one video."""

    det_frames: np.ndarray  # (F,) native frame index of each detection frame, increasing
    affine: np.ndarray  # (F, 2, 3) normalized affine: previous detection frame -> this one
    rows: np.ndarray  # (N, LOG_COLS) float32
    size: tuple[int, int]  # native (width, height)
    num_frames: int  # native frames of the video

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            tmp,
            det_frames=self.det_frames.astype(np.int64),
            affine=self.affine.astype(np.float32),
            rows=self.rows.astype(np.float32),
            size=np.asarray(self.size, dtype=np.int64),
            num_frames=np.asarray(self.num_frames, dtype=np.int64),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> TrackLog:
        with np.load(path) as data:
            size = tuple(int(x) for x in data["size"])
            return cls(
                det_frames=data["det_frames"].astype(np.int64),
                affine=data["affine"].astype(np.float64),
                rows=data["rows"].astype(np.float64),
                size=(size[0], size[1]),
                num_frames=int(data["num_frames"]),
            )


@dataclass
class Track:
    """One kept track: its detection frames (native), boxes, confidences and class."""

    tid: int
    frames: np.ndarray  # (n,) native frame indices, increasing
    boxes: np.ndarray  # (n, 4) tracker boxes
    raw: np.ndarray  # (n, 4) raw detection boxes
    conf: np.ndarray  # (n,)
    cls: int  # V6_CLASS_NAMES index
    mean_conf: float = field(init=False)

    def __post_init__(self) -> None:
        self.mean_conf = float(self.conf.mean())

    @property
    def first(self) -> int:
        return int(self.frames[0])

    @property
    def last(self) -> int:
        return int(self.frames[-1])

    def alive(self, frame: int) -> bool:
        return self.first <= frame <= self.last


# --------------------------------------------------------------------------- boxes


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(N, 4) x (M, 4) -> (N, M)`` IoU of ``xyxy`` boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0.0, None).prod(-1)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0.0, None).prod(-1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0.0, None).prod(-1)
    union = area_a[:, None] + area_b[None, :] - inter
    out: np.ndarray = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    return out


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Share of ``inner``'s area inside ``outer``."""
    lt = np.maximum(inner[:2], outer[:2])
    rb = np.minimum(inner[2:], outer[2:])
    inter = float(np.clip(rb - lt, 0.0, None).prod())
    area = float(np.clip(inner[2:] - inner[:2], 0.0, None).prod())
    return inter / area if area > 0 else 0.0


def normalize_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel ``xyxy`` -> normalized, clipped to the frame."""
    scale = np.array([width, height, width, height], dtype=np.float64)
    out: np.ndarray = np.clip(np.asarray(boxes, dtype=np.float64) / scale, 0.0, 1.0)
    return out


def centre(box: np.ndarray) -> np.ndarray:
    out: np.ndarray = (np.asarray(box[..., :2]) + np.asarray(box[..., 2:])) / 2.0
    return out


def enlarge_to(box: np.ndarray, min_side: float) -> np.ndarray:
    """Grow ``box`` about its centre so each side is at least ``min_side`` (normalized)."""
    c = centre(box)
    half = np.maximum((box[2:] - box[:2]) / 2.0, min_side / 2.0)
    return np.concatenate([c - half, c + half])


def crop_square(box: np.ndarray, width: int, height: int) -> tuple[int, int, int]:
    """Pixel square ``(x0, y0, side)`` around ``box``: 1.5 x its long side, at least 32 px.

    The square may extend past the frame; the caller pads with zeros (PREREG_V6 §2.1).
    """
    px = box * np.array([width, height, width, height], dtype=np.float64)
    c = centre(px)
    side = max(
        float(max(px[2] - px[0], px[3] - px[1])) * constants.V6_CROP_SCALE,
        float(constants.V6_CROP_MIN_PX),
    )
    side_i = round(side)
    return round(float(c[0]) - side_i / 2), round(float(c[1]) - side_i / 2), side_i


def cut_square(frame: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """``(side, side, 3)`` uint8 cut of ``frame``; outside the frame is zero."""
    out = np.zeros((side, side, frame.shape[2]), dtype=frame.dtype)
    h, w = frame.shape[:2]
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out


# --------------------------------------------------------------------------- camera affine


def pixel_to_normalized_affine(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    """A pixel-space ``2 x 3`` warp ``p' = M p + t`` as the same warp on normalized coordinates."""
    m = np.asarray(matrix, dtype=np.float64)
    out = np.empty((2, 3), dtype=np.float64)
    out[0, 0], out[0, 1] = m[0, 0], m[0, 1] * height / width
    out[1, 0], out[1, 1] = m[1, 0] * width / height, m[1, 1]
    out[0, 2], out[1, 2] = m[0, 2] / width, m[1, 2] / height
    return out


def _as3(a: np.ndarray) -> np.ndarray:
    out = np.eye(3)
    out[:2] = a
    return out


def det_index(det_frames: np.ndarray, frame: int) -> int:
    """Index of the last detection frame ``<= frame`` (0 before the first)."""
    i = int(np.searchsorted(det_frames, frame, side="right")) - 1
    return max(i, 0)


def compose_affine(affine: np.ndarray, det_frames: np.ndarray, start: int, end: int) -> np.ndarray:
    """Camera warp from native frame ``start`` to ``end`` (``start <= end``), via the detection
    frames at or before each (identity when both fall on the same detection frame)."""
    a, b = det_index(det_frames, start), det_index(det_frames, end)
    out = np.eye(3)
    for i in range(a + 1, b + 1):
        out = _as3(affine[i]) @ out
    result: np.ndarray = out[:2]
    return result


def warp_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    out: np.ndarray = matrix[:, :2] @ point + matrix[:, 2]
    return out


# --------------------------------------------------------------------------- tracks


def build_tracks(log: TrackLog, min_dets: int = constants.V6_MIN_TRACK_DETS) -> dict[int, Track]:
    """Kept tracks of a log (>= ``min_dets`` detections), riders merged, ids renumbered 1..n.

    Track ids are renumbered in order of first appearance so they stay exact in a float16 cache.
    """
    rows = log.rows
    out: list[Track] = []
    for tid in np.unique(rows[:, LOG_TID]).astype(int).tolist():
        r = rows[rows[:, LOG_TID] == tid]
        r = r[np.argsort(r[:, LOG_DET], kind="stable")]
        _, keep = np.unique(r[:, LOG_DET], return_index=True)
        r = r[keep]
        if len(r) < min_dets:
            continue
        coco = Counter(int(c) for c in r[:, LOG_CLS]).most_common(1)[0][0]
        if coco not in constants.V6_COCO_TO_CLASS:
            continue
        out.append(
            Track(
                tid=tid,
                frames=log.det_frames[r[:, LOG_DET].astype(int)],
                boxes=r[:, LOG_BOX].copy(),
                raw=r[:, LOG_RAW].copy(),
                conf=r[:, LOG_CONF].copy(),
                cls=constants.V6_COCO_TO_CLASS[coco],
            )
        )
    merged = merge_riders(out)
    merged.sort(key=lambda t: (t.first, t.tid))
    for i, track in enumerate(merged, start=1):
        track.tid = i
    if len(merged) > FP16_EXACT_INT:
        raise ValueError(f"{len(merged)} tracks: ids would not be exact in float16")
    return {t.tid: t for t in merged}


def merge_riders(tracks: list[Track]) -> list[Track]:
    """A person track riding a two-wheeler track becomes part of it, as DoTA's ``rider`` class.

    The person is merged when its box overlaps the two-wheeler's (IoU or containment above
    ``V6_RIDER_CONTAIN``) on at least ``V6_RIDER_SHARE`` of their co-visible frames; the
    two-wheeler's boxes become the union on those frames and its class ``rider``.
    """
    wheels = [t for t in tracks if t.cls in constants.V6_TWO_WHEELERS]
    keep: list[Track] = []
    for person in (t for t in tracks if t.cls == 0):
        best, best_share = None, 0.0
        for wheel in wheels:
            shared, ip, iw = np.intersect1d(person.frames, wheel.frames, return_indices=True)
            if not len(shared):
                continue
            hits = [
                max(
                    float(iou_matrix(person.boxes[a], wheel.boxes[b])[0, 0]),
                    containment(person.boxes[a], wheel.boxes[b]),
                )
                > constants.V6_RIDER_CONTAIN
                for a, b in zip(ip, iw, strict=True)
            ]
            share = float(np.mean(hits))
            if share >= constants.V6_RIDER_SHARE and share > best_share:
                best, best_share = wheel, share
        if best is None:
            keep.append(person)
            continue
        _, ip, iw = np.intersect1d(person.frames, best.frames, return_indices=True)
        for a, b in zip(ip, iw, strict=True):
            best.boxes[b] = _union(best.boxes[b], person.boxes[a])
            best.raw[b] = _union(best.raw[b], person.raw[a])
        best.cls = constants.V6_RIDER_CLASS
    return keep + [t for t in tracks if t.cls != 0]


def _union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:])])


def box_at(track: Track, frame: int, raw: bool = False) -> np.ndarray | None:
    """The track's box at a native frame: linear between its detections, ``None`` outside."""
    if not track.alive(frame):
        return None
    boxes = track.raw if raw else track.boxes
    i = int(np.searchsorted(track.frames, frame))
    if i < len(track.frames) and track.frames[i] == frame:
        out: np.ndarray = boxes[i].copy()
        return out
    f0, f1 = track.frames[i - 1], track.frames[i]
    w = (frame - f0) / (f1 - f0)
    interp: np.ndarray = (1 - w) * boxes[i - 1] + w * boxes[i]
    return interp


# --------------------------------------------------------------------------- kinematics


def kinematics(
    track: Track,
    frame: int,
    log: TrackLog,
    others: list[Track],
    det_step: int,
) -> np.ndarray:
    """``V6_KIN_NAMES`` of a track at a native frame, read at its last detection ``<= frame``.

    Velocity is ego-compensated (the previous centre is warped by the camera affine) and in box
    heights per second; ``rho`` = d log(h) / dt (looming); gaps are normalized centre distances
    to the nearest other agent alive at the frame and to the ego point (bottom centre).
    Undefined values (a track's first detection) are 0; every value is capped at ``V6_KIN_CLIP``.
    """
    i = int(np.searchsorted(track.frames, frame, side="right")) - 1
    if i < 0:
        return np.zeros(constants.V6_KIN_DIM)
    dt_unit = det_step * constants.V6_DET_FPS  # native frames per second

    def velocity(j: int) -> tuple[np.ndarray, float]:
        if j < 1:
            return np.zeros(2), 0.0
        f0, f1 = int(track.frames[j - 1]), int(track.frames[j])
        dt = (f1 - f0) / dt_unit
        prev = warp_point(
            compose_affine(log.affine, log.det_frames, f0, f1), centre(track.boxes[j - 1])
        )
        h1 = max(float(track.boxes[j, 3] - track.boxes[j, 1]), 1e-6)
        h0 = max(float(track.boxes[j - 1, 3] - track.boxes[j - 1, 1]), 1e-6)
        u = (centre(track.boxes[j]) - prev) / (h1 * dt)
        return u, (np.log(h1) - np.log(h0)) / dt

    u, rho = velocity(i)
    u_prev, _ = velocity(i - 1)
    accel = np.zeros(2)
    if i >= 2:
        accel = (u - u_prev) / ((track.frames[i] - track.frames[i - 1]) / dt_unit)
    age = (int(track.frames[i]) - track.first) / dt_unit

    def gap_at(f: int) -> float:
        me = box_at(track, f)
        if me is None:
            return 0.0
        dists = [
            float(np.linalg.norm(centre(me) - centre(b)))
            for o in others
            if o.tid != track.tid and (b := box_at(o, f)) is not None
        ]
        return min(dists) if dists else 0.0

    gap = gap_at(frame)
    gap_rate = 0.0
    if i >= 1:
        f_prev = int(track.frames[i - 1])
        gap_rate = (gap - gap_at(f_prev)) / max((frame - f_prev) / dt_unit, 1e-6)
    me = box_at(track, frame)
    gap_ego = (
        float(np.linalg.norm(centre(me) - np.asarray(constants.V6_EGO_POINT)))
        if me is not None
        else 0.0
    )
    out = np.array(
        [u[0], u[1], accel[0], accel[1], rho, age, gap, gap_rate, gap_ego], dtype=np.float64
    )
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    clipped: np.ndarray = np.clip(out, -constants.V6_KIN_CLIP, constants.V6_KIN_CLIP)
    return clipped


def ego_block(log: TrackLog, clip_frames: list[int]) -> np.ndarray:
    """``(48,)``: per slot of the centred clip, the camera warp across the slot minus identity."""
    out = np.zeros((constants.V6_TOKEN_SLOTS, 6))
    eye = np.eye(2, 3)
    for slot in range(constants.V6_TOKEN_SLOTS):
        a, b = clip_frames[2 * slot], clip_frames[2 * slot + 1]
        out[slot] = (compose_affine(log.affine, log.det_frames, min(a, b), max(a, b)) - eye).ravel()
    clipped: np.ndarray = np.clip(out.ravel(), -constants.V6_KIN_CLIP, constants.V6_KIN_CLIP)
    return clipped


def select_agents(tracks: dict[int, Track], frame: int, k_max: int) -> list[Track]:
    """Tracks alive at ``frame``, at most ``k_max``, by track-mean confidence (ties: track id)."""
    alive = [t for t in tracks.values() if t.alive(frame)]
    alive.sort(key=lambda t: (-t.mean_conf, t.tid))
    return alive[:k_max]


__all__ = [
    "LOG_COLS",
    "Track",
    "TrackLog",
    "box_at",
    "build_tracks",
    "compose_affine",
    "containment",
    "crop_square",
    "cut_square",
    "ego_block",
    "enlarge_to",
    "iou_matrix",
    "kinematics",
    "merge_riders",
    "normalize_boxes",
    "pixel_to_normalized_affine",
    "select_agents",
]
