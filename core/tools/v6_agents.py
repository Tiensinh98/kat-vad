"""KAT-VAD v6 (LaGoVAD-AP) agent tool (``core/docs/v6/PREREG_V6.md``).

Subcommands
-----------
``extract-dota``  DoTA-CAP clips rebuilt from the MM-AU CAP stream (as v5 Stage 0): detect + track
                  every DoTA frame (10 fps), agent rows at protocol B's ``s1[::3]``.
``extract-nexar`` Nexar window sources (native mp4, ~30 fps): detect + track every 3rd frame,
                  agent rows at A2's training ``s1[::8]``.
``step0``         Go / no-go on DoTA-CAP-dev (PREREG_V6 §1): D0 recall, D1 load / ceiling,
                  D2 training-free per-agent ranking vs the centre / largest / lowest priors.
``dump``          Per-row candidate logits of a trained AP checkpoint on DoTA-CAP clips.
``readout``       Detection (macro / micro / strat-50, ego vs non-ego, paired Δ vs the control) and
                  box-level hit vs the priors (PREREG_V6 §3).

Detection and tracking need ultralytics + a GPU (Colab); the read-outs are CPU-only.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from core import constants
from core.agents import layout
from core.agents.features import RowSources, agent_manifest, agent_rows
from core.agents.geometry import Track, TrackLog, box_at, build_tracks, iou_matrix
from core.agents.tracking import (
    Detect,
    load_detector,
    make_motion_estimator,
    make_tracker,
    track_manifest,
    track_video,
)
from core.data.agent_cache import append_agents, load_agents, write_agent_manifest
from core.data.v2_splits import dota_group, load_split
from core.data.video_io import list_frame_images, read_images
from core.device import resolve_device
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.models.videomae_v2 import VideoMAEv2Encoder, load_pretrained
from core.tools import extract_clip_features
from core.tools.dota_cap import rebuild_folder
from core.tools.feature_cache import read_ids_file, save_array
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_extract import iter_frame_chunks, resize_frames, video_paths

LOGGER = logging.getLogger(__name__)

OK = "ok"
FARM_DIR = "farm"
ASPECT = 16 / 9
ASPECT_TOLERANCE = 0.02
DOTA_BOX_SIZE = (1280.0, 720.0)  # DoTA annotation boxes are in 1280 x 720 pixels
TOKEN_BATCH = 8
NEXAR_CLIP_STEP = constants.VIDEOMAE_CLIP_FRAME_STEP  # 3 native frames (A2's V2-S rows)
DOTA_CLIP_STEP = constants.DOTA_VIDEOMAE_FRAME_STEP  # 1 (10 fps)
BOX_SHARE_SMALL = constants.V6_SMALL_BOX_PX / (DOTA_BOX_SIZE[0] * DOTA_BOX_SIZE[1])
SIZE_BINS = (BOX_SHARE_SMALL, 0.02, 0.08)
SIZE_NAMES = ("<2267px", "2267px-2%", "2-8%", ">8%")


# =========================================================================== extraction


@dataclass
class Encoders:
    detect: Detect
    videomae: VideoMAEv2Encoder
    clip: Any
    device: torch.device


def load_encoders(device: torch.device, weights: str = constants.V6_DETECTOR_WEIGHTS) -> Encoders:
    return Encoders(
        detect=load_detector(str(device), weights),
        videomae=load_pretrained(constants.V5_ENCODER, device, None),
        clip=extract_clip_features.load_pretrained_encoder(device),
        device=device,
    )


@dataclass
class Decoded:
    """What one pass over a video keeps: VideoMAE inputs for every frame, native row frames."""

    resized: list[Tensor] = field(default_factory=list)  # (3, 224, 224) float16 in [0, 1]
    natives: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.resized)


def _detection_stream(
    frames: Iterator[np.ndarray], keep: Decoded, det_step: int, stride: int
) -> Iterator[tuple[int, np.ndarray]]:
    """Walk every frame once: store its VideoMAE input and, on row frames, the native frame;
    yield the detection frames to the tracker."""
    for index, frame in enumerate(frames):
        keep.resized.append(resize_frames(frame[None])[0].half())
        if index % stride == 0:
            keep.natives[index] = frame
        if index % det_step == 0:
            yield index, frame


def token_fn(
    decoded: Decoded, encoder: VideoMAEv2Encoder, device: torch.device
) -> Callable[[list[list[int]]], Tensor]:
    mean = torch.tensor(constants.VIDEOMAE_IMAGE_MEAN, device=device).view(3, 1, 1, 1)
    std = torch.tensor(constants.VIDEOMAE_IMAGE_STD, device=device).view(3, 1, 1, 1)
    dtype = next(encoder.parameters()).dtype

    @torch.no_grad()
    def tokens(clips: list[list[int]]) -> Tensor:
        out = []
        for start in range(0, len(clips), TOKEN_BATCH):
            batch = torch.stack(
                [
                    (
                        torch.stack([decoded.resized[i] for i in clip], dim=1).to(device).float()
                        - mean
                    )
                    / std
                    for clip in clips[start : start + TOKEN_BATCH]
                ]
            )
            out.append(encoder.forward_tokens(batch.to(dtype)).float())
        return torch.cat(out)

    return tokens


def process_video(
    frames: Iterator[np.ndarray],
    enc: Encoders,
    det_step: int,
    stride: int,
    clip_step: int,
) -> tuple[TrackLog, np.ndarray, dict[str, Any]]:
    """Detect + track, then the agent rows of one video."""
    decoded = Decoded()
    stream = _detection_stream(frames, decoded, det_step, stride)
    log = track_video(stream, 0, enc.detect, make_tracker(), make_motion_estimator())
    log.num_frames = decoded.num_frames
    sources = RowSources(
        num_frames=decoded.num_frames,
        stride=stride,
        clip_step=clip_step,
        det_step=det_step,
        tokens=token_fn(decoded, enc.videomae, enc.device),
        frame=lambda t: decoded.natives[t],
    )
    rows, summary = agent_rows(log, sources, enc.clip, enc.device)
    return log, rows, summary


def _save(video_id: str, log: TrackLog, rows: np.ndarray, agent_dir: Path, track_dir: Path) -> None:
    log.save(track_dir / f"{video_id}.npz")
    save_array(agent_dir / f"{video_id}.npy", rows)


def _done(video_id: str, agent_dir: Path, track_dir: Path) -> bool:
    return (agent_dir / f"{video_id}.npy").exists() and (track_dir / f"{video_id}.npz").exists()


def _write_track_manifest(track_dir: Path, det_step: int, source: str) -> None:
    track_dir.mkdir(parents=True, exist_ok=True)
    path = track_dir / "track_manifest.json"
    if not path.exists():
        write_json_atomic(path, track_manifest(det_step, source))


@dataclass
class DotaExtractor:
    """Per-batch callback of :func:`stream_video_batches` (one CAP batch on VM disk)."""

    by_cap: dict[str, list[dict[str, Any]]]
    enc: Encoders
    s1_dir: Path
    agent_dir: Path
    track_dir: Path
    farm: Path
    subdir: str
    report: dict[str, dict[str, Any]]
    report_path: Path

    def __call__(self, ready_dir: Path) -> None:
        for folder in sorted(p for p in ready_dir.iterdir() if p.is_dir()):
            cap_paths = list_frame_images(folder, self.subdir)
            for clip in self.by_cap.get(folder.name, []):
                self.report[clip["dota_id"]] = self.one_clip(clip, cap_paths)
        write_json_atomic(self.report_path, self.report)

    def one_clip(self, clip: dict[str, Any], cap_paths: list[Path]) -> dict[str, Any]:
        dota_id = clip["dota_id"]
        entry: dict[str, Any] = {"cap_id": clip["cap_id"]}
        if len(cap_paths) != clip["cap_frames"]:
            return {**entry, "reason": "cap_frames_mismatch"}
        farm = rebuild_folder(self.farm / dota_id, cap_paths, clip["frame_map"])
        try:
            paths = list_frame_images(farm)
            native = len(np.load(self.s1_dir / f"{dota_id}.npy", mmap_mode="r"))
            if len(paths) != native:
                return {**entry, "reason": "frame_count", "frames": len(paths), "s1": native}
            first = read_images(paths[:1])[0]
            aspect = first.shape[1] / first.shape[0]
            if abs(aspect / ASPECT - 1.0) > ASPECT_TOLERANCE:
                return {**entry, "reason": "aspect", "aspect": aspect}
            frames = (read_images([p])[0] for p in paths)
            log, rows, summary = process_video(
                frames,
                self.enc,
                det_step=1,
                stride=constants.V6_DOTA_STRIDE,
                clip_step=DOTA_CLIP_STEP,
            )
            _save(dota_id, log, rows, self.agent_dir, self.track_dir)
            LOGGER.info("%s: %d rows, %d tracks", dota_id, summary["rows"], summary["tracks"])
            return {**entry, **summary, "reason": OK}
        finally:
            shutil.rmtree(farm, ignore_errors=True)


def run_extract_dota(args: argparse.Namespace) -> None:
    from core.tools.stream_frames_clip import stream_video_batches

    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    ids = sorted(json.loads(args.final_readout.read_text(encoding="utf-8"))["ids"])
    if args.dev_only:
        dev = set(load_split(constants.V2_SPLIT_DOTA_CAP_DEV))
        ids = [v for v in ids if v in dev]
    write_agent_manifest(
        args.agent_dir,
        agent_manifest(constants.V6_DOTA_STRIDE, DOTA_CLIP_STEP, 1, "DoTA-CAP (MM-AU CAP frames)"),
    )
    _write_track_manifest(args.track_dir, 1, "DoTA-CAP (MM-AU CAP frames)")
    report: dict[str, dict[str, Any]] = (
        json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    )
    work = [
        {"dota_id": v, **alignment["clips"][v]}
        for v in ids
        if not _done(v, args.agent_dir, args.track_dir)
        and report.get(v, {}).get("reason") in (None, OK)
    ]
    if args.cap_ids_file is not None:
        group = read_ids_file(args.cap_ids_file)
        work = [c for c in work if c["cap_id"] in group]
    if not work:
        LOGGER.info("Nothing to extract for %s", args.parts[0].parent)
        return
    by_cap: dict[str, list[dict[str, Any]]] = {}
    for clip in work:
        by_cap.setdefault(clip["cap_id"], []).append(clip)
    extractor = DotaExtractor(
        by_cap=by_cap,
        enc=load_encoders(resolve_device(args.device), str(args.weights)),
        s1_dir=args.dota_clip_dir,
        agent_dir=args.agent_dir,
        track_dir=args.track_dir,
        farm=args.work_dir / FARM_DIR,
        subdir=args.frames_subdir,
        report=report,
        report_path=args.report,
    )
    stream_video_batches(
        sorted(args.parts),
        args.work_dir / "stream",
        args.frames_subdir,
        args.max_batch_gb * constants.BYTES_PER_GB,
        extractor,
        keep_ids=set(by_cap),
    )


def nexar_sources(data_dir: Path) -> list[str]:
    """Source videos of the Nexar ``window`` corpus (every window's source)."""
    windows = json.loads((data_dir / constants.WINDOWS_FILENAME).read_text(encoding="utf-8"))
    return sorted({str(w["source"]) for w in windows.values()})


def _native_frames(path: Path) -> Iterator[np.ndarray]:
    for chunk in iter_frame_chunks(path):
        yield from chunk


def run_extract_nexar(args: argparse.Namespace) -> None:
    census = json.loads(args.census.read_text(encoding="utf-8"))
    ids = nexar_sources(args.data_dir)
    if args.ids_file is not None:
        wanted = read_ids_file(args.ids_file)
        ids = [v for v in ids if v in wanted]
    shard, shards = args.shard
    ids = ids[shard::shards]
    stride = constants.V6_NEXAR_STRIDE
    write_agent_manifest(
        args.agent_dir, agent_manifest(stride, NEXAR_CLIP_STEP, -1, "Nexar native mp4")
    )
    _write_track_manifest(args.track_dir, -1, "Nexar native mp4 (det_step per video: fps / 10)")
    report_path = args.report.with_name(f"{args.report.stem}_{shard}of{shards}.json")
    report: dict[str, dict[str, Any]] = (
        json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    )
    todo = [v for v in ids if not _done(v, args.agent_dir, args.track_dir)]
    if not todo:
        LOGGER.info("Nexar shard %d/%d: nothing to extract", shard, shards)
        return
    paths = video_paths(args.raw, census, set(todo))
    enc = load_encoders(resolve_device(args.device), str(args.weights))
    for i, video_id in enumerate(todo):
        fps = census["videos"][video_id].get("fps") or constants.DADA_ASSUMED_FPS
        det_step = max(round(fps / constants.V6_DET_FPS), 1)
        log, rows, summary = process_video(
            _native_frames(paths[video_id]), enc, det_step, stride, NEXAR_CLIP_STEP
        )
        expected = len(np.load(args.clip_dir / f"{video_id}.npy", mmap_mode="r"))
        if len(rows) != expected:
            report[video_id] = {"reason": "row_count", "rows": len(rows), "a2_rows": expected}
        else:
            _save(video_id, log, rows, args.agent_dir, args.track_dir)
            report[video_id] = {**summary, "fps": fps, "det_step": det_step, "reason": OK}
        if i % 10 == 0 or i == len(todo) - 1:
            write_json_atomic(report_path, report)
            LOGGER.info("Nexar shard %d/%d: %d / %d", shard, shards, i + 1, len(todo))
    bad = [v for v, r in report.items() if r["reason"] != OK]
    if bad:
        raise RuntimeError(f"{len(bad)} Nexar sources failed: {bad[:5]} (see {report_path})")


# =========================================================================== annotations


@dataclass(frozen=True)
class GroundTruth:
    """DoTA's anomalous-agent boxes of one clip (normalized), by frame and by track."""

    by_frame: dict[int, np.ndarray]  # frame -> (n, 4)
    tracks: dict[int, list[int]]  # obj_track_id -> its annotated frames
    track_boxes: dict[tuple[int, int], np.ndarray]  # (track, frame) -> box
    ego: bool
    night: bool
    category: str
    start: int
    end: int


def load_ground_truth(path: Path) -> GroundTruth:
    data = json.loads(path.read_text(encoding="utf-8"))
    scale = np.array([*DOTA_BOX_SIZE, *DOTA_BOX_SIZE])
    by_frame: dict[int, list[np.ndarray]] = {}
    tracks: dict[int, list[int]] = {}
    track_boxes: dict[tuple[int, int], np.ndarray] = {}
    for frame in data["labels"]:
        f = int(frame["frame_id"])
        for obj in frame["objects"]:
            box = np.clip(np.asarray(obj["bbox"], dtype=np.float64) / scale, 0.0, 1.0)
            by_frame.setdefault(f, []).append(box)
            tid = int(obj.get("obj_track_id", -1))
            tracks.setdefault(tid, []).append(f)
            track_boxes[(tid, f)] = box
    return GroundTruth(
        by_frame={f: np.stack(b) for f, b in by_frame.items()},
        tracks=tracks,
        track_boxes=track_boxes,
        ego=bool(data["ego_involve"]),
        night=bool(data.get("night", False)),
        category=str(data.get("accident_name", "")),
        start=int(data["anomaly_start"]),
        end=int(data["anomaly_end"]),
    )


def size_bin(box: np.ndarray) -> str:
    share = float((box[2] - box[0]) * (box[3] - box[1]))
    return SIZE_NAMES[int(np.searchsorted(SIZE_BINS, share, side="right"))]


def agentless(gt: GroundTruth) -> bool:
    return gt.category in constants.V6_AGENTLESS_CATEGORIES or not gt.tracks


# =========================================================================== Step 0


def track_recall(
    gt: GroundTruth, tracks: dict[int, Track], selected: Callable[[int], list[Track]]
) -> list[dict[str, Any]]:
    """Per GT track: matched share of its frames by any kept track / by a selected one."""
    out = []
    for tid, frames in gt.tracks.items():
        hits = {thr: [0, 0] for thr in (constants.V6_IOU_LOOSE, constants.V6_IOU_STRICT)}
        boxes = [gt.track_boxes[(tid, f)] for f in frames]
        for f, box in zip(frames, boxes, strict=True):
            kept = [b for t in tracks.values() if (b := box_at(t, f)) is not None]
            sel = [b for t in selected(f) if (b := box_at(t, f)) is not None]
            for thr, h in hits.items():
                h[0] += bool(kept) and float(iou_matrix(box, np.stack(kept)).max()) >= thr
                h[1] += bool(sel) and float(iou_matrix(box, np.stack(sel)).max()) >= thr
        n = len(frames)
        med = np.median(np.stack(boxes), axis=0)
        out.append(
            {
                "track": tid,
                "frames": n,
                "size": size_bin(med),
                **{
                    f"found@{thr}": h[0] / n >= constants.V6_TRACK_FOUND_SHARE
                    for thr, h in hits.items()
                },
                **{
                    f"kept@{thr}": h[1] / n >= constants.V6_TRACK_FOUND_SHARE
                    for thr, h in hits.items()
                },
                **{f"frame_share@{thr}": h[0] / n for thr, h in hits.items()},
            }
        )
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    out: np.ndarray = (a * b).sum(-1)
    return out


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cos_change(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(_cos(a, b))


def agent_scores(rows: np.ndarray, text: np.ndarray) -> dict[str, np.ndarray]:
    """Training-free per-agent scores ``(R, K)`` (D2); invalid slots are -inf."""
    agents, _ = layout.unpack_rows(np.asarray(rows, dtype=np.float32))
    valid = agents[..., layout.VALID] > 0.5
    tid = agents[..., layout.TRACK]
    motion, crop = agents[..., layout.MOTION], agents[..., layout.CROP]
    kin = agents[..., layout.KIN]
    names = constants.V6_KIN_NAMES

    def lagged(
        feat: np.ndarray, lag: int, fn: Callable[[np.ndarray, np.ndarray], float]
    ) -> np.ndarray:
        out = np.zeros(valid.shape)
        for r in range(lag, len(rows)):
            for k in np.flatnonzero(valid[r]):
                prev = np.flatnonzero(valid[r - lag] & (tid[r - lag] == tid[r, k]))
                if len(prev):
                    out[r, k] = fn(feat[r, k], feat[r - lag, prev[0]])
        return out

    box = agents[..., layout.BOX]
    c = (box[..., :2] + box[..., 2:]) / 2
    scores = {
        "motion_change": lagged(motion, constants.V6_CHANGE_LAG_ROWS, _l2),
        "crop_change": lagged(crop, constants.V6_CROP_CHANGE_LAG_ROWS, _cos_change),
        "crop_text": _cos(crop, text[0]) - _cos(crop, text[1]),
        "loom": kin[..., names.index("rho")],
        "approach": -kin[..., names.index("gap_rate")],
        "centre": -np.linalg.norm(c - 0.5, axis=-1),
        "largest": (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1]),
        "lowest": box[..., 3],
    }
    return {k: np.where(valid, v, -np.inf) for k, v in scores.items()}


def text_embeds(device: torch.device) -> np.ndarray:
    """CLIP text ``[abnormal, normal]`` embeddings, the space of the crop features."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(  # nosec B615 - pinned revision
        constants.CLIP_MODEL_NAME, revision=constants.CLIP_MODEL_REVISION
    )
    model: Any = CLIPTextModelWithProjection.from_pretrained(  # nosec B615 - pinned revision
        constants.CLIP_MODEL_NAME, revision=constants.CLIP_MODEL_REVISION
    )
    model = model.to(device).eval()
    with torch.no_grad():
        inputs = tok(
            [constants.V6_TEXT_ABNORMAL, constants.V6_TEXT_NORMAL],
            padding=True,
            return_tensors="pt",
        ).to(device)
        out: np.ndarray = model(**inputs).text_embeds.float().cpu().numpy()
    return out


def row_hits(
    rows: np.ndarray,
    gt: GroundTruth,
    stride: int,
    scores: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, list[float]]:
    """Per abnormal annotated row: 1 if the top agent under each score overlaps a GT box;
    ``ceiling`` = any selected agent overlaps; ``chance`` = share of overlapping agents."""
    agents, _ = layout.unpack_rows(np.asarray(rows, dtype=np.float32))
    out: dict[str, list[float]] = {k: [] for k in [*scores, "ceiling", "chance"]}
    for r in range(len(rows)):
        frame = r * stride
        if frame not in gt.by_frame or not gt.start <= frame <= gt.end:
            continue
        valid = agents[r, :, layout.VALID] > 0.5
        if not valid.any():
            for k in out:
                out[k].append(0.0)
            continue
        hit = iou_matrix(agents[r, :, layout.BOX], gt.by_frame[frame]).max(1) >= threshold
        hit &= valid
        out["ceiling"].append(float(hit.any()))
        out["chance"].append(float(hit.sum() / valid.sum()))
        for name, s in scores.items():
            out[name].append(float(hit[int(np.argmax(s[r]))]))
    return out


def _ci(values: dict[str, float]) -> dict[str, float] | None:
    return cluster_bootstrap_ci(
        values,
        {v: dota_group(v) for v in values},
        constants.V2_E1_BOOTSTRAP,
        constants.V5_BOOTSTRAP_SEED,
        constants.V2_E1_CI,
    )


def _share(rows: list[dict[str, Any]], key: str) -> float | None:
    return float(np.mean([r[key] for r in rows])) if rows else None


def run_step0(args: argparse.Namespace) -> dict[str, Any]:
    ids = sorted(load_split(constants.V2_SPLIT_DOTA_CAP_DEV))
    ids = [v for v in ids if (args.agent_dir / f"{v}.npy").exists()]
    if not ids:
        raise FileNotFoundError(f"no CAP-dev agent rows under {args.agent_dir}")
    text = (
        np.load(args.text_embeds) if args.text_embeds else text_embeds(resolve_device(args.device))
    )
    stride = constants.V6_DOTA_STRIDE
    recall_rows: list[dict[str, Any]] = []
    loads: list[float] = []
    at_cap: list[float] = []
    hits: dict[float, dict[str, dict[str, float]]] = {
        constants.V6_IOU_LOOSE: {},
        constants.V6_IOU_STRICT: {},
    }
    for v in ids:
        gt = load_ground_truth(args.annotations / f"{v}.json")
        log = TrackLog.load(args.track_dir / f"{v}.npz")
        tracks = build_tracks(log)
        rows = np.asarray(load_agents(args.agent_dir, v), dtype=np.float32)
        agents, _ = layout.unpack_rows(rows)
        n_valid = (agents[..., layout.VALID] > 0.5).sum(1)
        loads.append(float(n_valid.mean()))
        at_cap.append(float((n_valid == constants.V6_K_MAX).mean()))
        if agentless(gt):
            continue

        def selected(f: int, tracks: dict[int, Track] = tracks) -> list[Track]:
            alive = sorted(
                (t for t in tracks.values() if t.alive(f)), key=lambda t: (-t.mean_conf, t.tid)
            )
            return alive[: constants.V6_K_MAX]

        for r in track_recall(gt, tracks, selected):
            recall_rows.append(
                {**r, "clip": v, "ego": gt.ego, "night": gt.night, "category": gt.category}
            )
        scores = agent_scores(rows, text)
        for thr, per in hits.items():
            h = row_hits(rows, gt, stride, scores, thr)
            for name, values in h.items():
                if values:
                    per.setdefault(name, {})[v] = float(np.mean(values))

    def recall(rows_: list[dict[str, Any]], thr: float = constants.V6_IOU_LOOSE) -> dict[str, Any]:
        return {
            "tracks": len(rows_),
            "recall": _share(rows_, f"found@{thr}"),
            "kept_recall": _share(rows_, f"kept@{thr}"),
        }

    d0 = {
        "overall@0.3": recall(recall_rows),
        "overall@0.5": recall(recall_rows, constants.V6_IOU_STRICT),
        "ego": recall([r for r in recall_rows if r["ego"]]),
        "non_ego": recall([r for r in recall_rows if not r["ego"]]),
        "night": recall([r for r in recall_rows if r["night"]]),
        "by_size": {s: recall([r for r in recall_rows if r["size"] == s]) for s in SIZE_NAMES},
        "by_category": {
            c: recall([r for r in recall_rows if r["category"] == c])
            for c in sorted({r["category"] for r in recall_rows})
        },
    }
    d1 = {
        "agents_per_row": float(np.mean(loads)),
        "rows_at_k_max": float(np.mean(at_cap)),
        "kept_given_found": _share(
            [r for r in recall_rows if r[f"found@{constants.V6_IOU_LOOSE}"]],
            f"kept@{constants.V6_IOU_LOOSE}",
        ),
    }
    d2: dict[str, Any] = {}
    for thr, per in hits.items():
        block: dict[str, Any] = {k: _ci(vals) for k, vals in per.items()}
        centre = per.get("centre", {})
        for name in constants.V6_RANK_SCORES:
            common = sorted(set(per.get(name, {})) & set(centre))
            diff = {v: per[name][v] - centre[v] for v in common}
            block[f"{name}-centre"] = _ci(diff)
        d2[f"iou{thr}"] = block
    verdict = decide(d0, d2)
    payload = {
        "clips": len(ids),
        "D0": d0,
        "D1": d1,
        "D2": d2,
        "decision": verdict,
        "rule": "core/docs/v6/PREREG_V6.md §1",
    }
    write_json_atomic(args.out_dir / "step0.json", payload)
    write_text_atomic(args.out_dir / "step0.md", render_step0(payload))
    LOGGER.info("Step 0: %s", verdict)
    return payload


def decide(d0: dict[str, Any], d2: dict[str, Any]) -> dict[str, Any]:
    """PREREG_V6 §1 (proposal §4.3), applied mechanically."""
    overall = d0["overall@0.3"]["recall"] or 0.0
    smallest = d0["by_size"][SIZE_NAMES[0]]["recall"]
    strata = {"non_ego": d0["non_ego"]["recall"], "smallest_box": smallest}
    rank_wins = [
        n
        for n in constants.V6_RANK_SCORES
        if (ci := d2[f"iou{constants.V6_IOU_LOOSE}"].get(f"{n}-centre")) and ci["low"] > 0
    ]
    strata_ok = all(r is None or r >= constants.V6_GO_RECALL_STRATUM for r in strata.values())
    if overall < constants.V6_NARROW_RECALL:
        label = "NO-GO"
    elif overall >= constants.V6_GO_RECALL and strata_ok and rank_wins:
        label = "GO"
    else:
        label = "NARROW"
    dropped = [
        k
        for k, r in {
            **strata,
            "night": d0["night"]["recall"],
            **{f"size {s}": d0["by_size"][s]["recall"] for s in SIZE_NAMES},
        }.items()
        if r is not None and r < constants.V6_DROP_STRATUM
    ]
    return {
        "label": label,
        "overall_recall": overall,
        "strata": strata,
        "rank_scores_beating_centre": rank_wins,
        "h2_rests_on_learning": not rank_wins,
        "dropped_strata": dropped,
    }


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, dict) and "mean" in x:
        return f"{x['mean']:.3f} [{x['low']:+.3f}, {x['high']:+.3f}]"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def render_step0(p: dict[str, Any]) -> str:
    d0, d1, dec = p["D0"], p["D1"], p["decision"]
    lines = [
        f"# v6 Step 0 — DoTA-CAP-dev ({p['clips']} clips)",
        "",
        f"**Decision: {dec['label']}** (overall track recall @0.3 = {_fmt(dec['overall_recall'])})",
        "",
        "## D0 track recall (VO / OO excluded)",
        "",
        "| stratum | tracks | recall (any kept) | recall (top-16) |",
        "|---|---|---|---|",
    ]
    for name in ("overall@0.3", "overall@0.5", "ego", "non_ego", "night"):
        r = d0[name]
        lines.append(f"| {name} | {r['tracks']} | {_fmt(r['recall'])} | {_fmt(r['kept_recall'])} |")
    for s, r in d0["by_size"].items():
        lines.append(
            f"| size {s} | {r['tracks']} | {_fmt(r['recall'])} | {_fmt(r['kept_recall'])} |"
        )
    lines += [
        "",
        "## D1 load",
        "",
        f"- agents per row {_fmt(d1['agents_per_row'])}, "
        f"rows at K_max {_fmt(d1['rows_at_k_max'])}, kept | found {_fmt(d1['kept_given_found'])}",
        "",
        "## D2 box hit of the top agent (abnormal annotated rows)",
        "",
    ]
    for key, block in p["D2"].items():
        lines += [f"### {key}", "", "| score | hit | - centre |", "|---|---|---|"]
        for name in [*constants.V6_RANK_SCORES, *constants.V6_PRIORS, "chance", "ceiling"]:
            lines.append(
                f"| {name} | {_fmt(block.get(name))} | {_fmt(block.get(f'{name}-centre'))} |"
            )
        lines.append("")
    lines += [
        f"- rank scores above centre (CI > 0): {dec['rank_scores_beating_centre'] or 'none'}",
        f"- dropped strata (< {constants.V6_DROP_STRATUM}): {dec['dropped_strata'] or 'none'}",
    ]
    return "\n".join(lines) + "\n"


# =========================================================================== dump


@torch.no_grad()
def run_dump(args: argparse.Namespace) -> None:
    from core.tools.protocol_b_eval import check_baked_stride, input_rows
    from core.tools.rate_matched_eval import Run, load_finished_model

    device = resolve_device(args.device)
    model, step, _, cfg = load_finished_model(Run("AP", args.ckpt), device)
    if not cfg.agent.enabled:
        raise ValueError(f"{args.ckpt} has agent.enabled=false: nothing to dump")
    baked = check_baked_stride(args.input_dir, constants.V6_DOTA_STRIDE)
    ids = sorted(load_split(constants.V2_SPLIT_DOTA_CAP_DEV))
    out: dict[str, np.ndarray] = {}
    for v in ids:
        if not (args.agent_dir / f"{v}.npy").exists():
            continue
        native = len(np.load(args.s1_dir / f"{v}.npy", mmap_mode="r"))
        rows = input_rows(args.input_dir, v, native, constants.V6_DOTA_STRIDE, baked)
        rows = append_agents(rows, load_agents(args.agent_dir, v), v)
        x = torch.from_numpy(rows)[None].to(device)
        outputs = model(x, torch.tensor([len(rows)], device=device))
        out[v] = outputs["agent_logits"][0].float().cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp.npz")
    np.savez_compressed(tmp, step=np.asarray(step), **out)
    tmp.replace(args.out)
    LOGGER.info("dumped candidate logits of %d clips (step %d) -> %s", len(out), step, args.out)


# =========================================================================== readout


def _norm(s: np.ndarray) -> np.ndarray:
    out: np.ndarray = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return out


def detection_block(
    scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], ids: list[str]
) -> dict[str, Any]:
    """Macro (cluster CI), micro and position-stratified AUC over ``ids``."""
    per = {v: frame_auc(scores[v], labels[v]) for v in ids if 0 < labels[v].sum() < len(labels[v])}
    s = np.concatenate([_norm(scores[v]) for v in ids])
    y = np.concatenate([labels[v] for v in ids])
    pos = np.concatenate([np.arange(len(labels[v])) / len(labels[v]) for v in ids])
    bins = np.minimum((pos * constants.V6_STRAT_BINS).astype(int), constants.V6_STRAT_BINS - 1)
    strat = [
        frame_auc(s[bins == b], y[bins == b])
        for b in range(constants.V6_STRAT_BINS)
        if 0 < y[bins == b].sum() < (bins == b).sum()
    ]
    return {
        "clips": len(per),
        "macro": _ci(per),
        "micro": frame_auc(s, y),
        "strat50": float(np.mean(strat)) if strat else None,
        "per_clip": per,
    }


def box_hits(
    logits: dict[str, np.ndarray], agent_dir: Path, annotations: Path, threshold: float
) -> dict[str, Any]:
    """Box-level hit of the top-scored agent vs priors and chance, unconditional and | detected."""
    per: dict[str, dict[str, float]] = {}
    cond: dict[str, dict[str, float]] = {}
    for v, e in logits.items():
        gt = load_ground_truth(annotations / f"{v}.json")
        if agentless(gt):
            continue
        rows = np.asarray(load_agents(agent_dir, v), dtype=np.float32)
        agents, _ = layout.unpack_rows(rows)
        box = agents[..., layout.BOX]
        valid = agents[..., layout.VALID] > 0.5
        c = (box[..., :2] + box[..., 2:]) / 2
        cand = {
            "AP": np.where(valid, e[:, : constants.V6_K_MAX], -np.inf),
            "centre": np.where(valid, -np.linalg.norm(c - 0.5, axis=-1), -np.inf),
            "largest": np.where(
                valid, (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1]), -np.inf
            ),
            "lowest": np.where(valid, box[..., 3], -np.inf),
        }
        h = row_hits(rows, gt, constants.V6_DOTA_STRIDE, cand, threshold)
        # the AP map can point at the ego / no-agent candidate: that is a miss
        top_is_agent = e.argmax(-1) < constants.V6_K_MAX
        ap, k = [], 0
        for r in range(len(rows)):
            frame = r * constants.V6_DOTA_STRIDE
            if frame in gt.by_frame and gt.start <= frame <= gt.end:
                ap.append(h["AP"][k] * float(top_is_agent[r]))
                k += 1
        h["AP"] = ap
        detected = [i for i, x in enumerate(h["ceiling"]) if x > 0]
        for name, values in h.items():
            if values:
                per.setdefault(name, {})[v] = float(np.mean(values))
            if detected and name != "ceiling":
                cond.setdefault(name, {})[v] = float(np.mean([values[i] for i in detected]))
    out: dict[str, Any] = {
        "unconditional": {k: _ci(x) for k, x in per.items()},
        "given_detected": {k: _ci(x) for k, x in cond.items()},
    }
    for name in constants.V6_PRIORS:
        common = sorted(set(per.get("AP", {})) & set(per.get(name, {})))
        out[f"AP-{name}"] = _ci({v: per["AP"][v] - per[name][v] for v in common})
    return out


def run_readout(args: argparse.Namespace) -> dict[str, Any]:
    from core.tools.protocol_b_eval import read_clip_scores

    rows: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {
        name: read_clip_scores(Path(d) / "clip_scores.npz") for name, d in args.row
    }
    common = sorted(set.intersection(*(set(s) for s, _ in rows.values())))
    ego = {v: load_ground_truth(args.annotations / f"{v}.json").ego for v in common}
    subsets = {
        "all": common,
        "ego": [v for v in common if ego[v]],
        "non_ego": [v for v in common if not ego[v]],
    }
    det: dict[str, Any] = {}
    for name, (scores, labels) in rows.items():
        det[name] = {k: detection_block(scores, labels, ids) for k, ids in subsets.items()}
    paired: dict[str, Any] = {}
    for a, b in args.pair:
        paired[f"{a}-{b}"] = {}
        for k in subsets:
            pa, pb = det[a][k]["per_clip"], det[b][k]["per_clip"]
            paired[f"{a}-{b}"][k] = _ci({v: pa[v] - pb[v] for v in sorted(set(pa) & set(pb))})
    for block in det.values():
        for sub in block.values():
            sub.pop("per_clip")
    payload: dict[str, Any] = {"clips": len(common), "detection": det, "paired": paired}
    if args.logits is not None:
        with np.load(args.logits) as z:
            logits = {k: z[k] for k in z.files if k != "step"}
        payload["box_hit"] = {
            f"iou{thr}": box_hits(logits, args.agent_dir, args.annotations, thr)
            for thr in (constants.V6_IOU_LOOSE, constants.V6_IOU_STRICT)
        }
    write_json_atomic(args.out_dir / "v6_readout.json", payload)
    write_text_atomic(args.out_dir / "v6_readout.md", render_readout(payload))
    return payload


def render_readout(p: dict[str, Any]) -> str:
    lines = [
        f"# v6 read-out — DoTA-CAP ({p['clips']} common clips, protocol B s3)",
        "",
        "| row | subset | clips | macro [95% CI] | micro | strat-50 |",
        "|---|---|---|---|---|---|",
    ]
    for name, block in p["detection"].items():
        for sub, d in block.items():
            lines.append(
                f"| {name} | {sub} | {d['clips']} | {_fmt(d['macro'])} | "
                f"{_fmt(d['micro'])} | {_fmt(d['strat50'])} |"
            )
    lines += [
        "",
        "## Paired macro Δ (cluster bootstrap by source video)",
        "",
        "| pair | all | ego | non-ego |",
        "|---|---|---|---|",
    ]
    for pair, d in p["paired"].items():
        lines.append(f"| {pair} | {_fmt(d['all'])} | {_fmt(d['ego'])} | {_fmt(d['non_ego'])} |")
    for key, block in p.get("box_hit", {}).items():
        lines += [
            "",
            f"## Box-level hit, {key}",
            "",
            "| map | hit | hit given detected |",
            "|---|---|---|",
        ]
        for name, ci in block["unconditional"].items():
            lines.append(f"| {name} | {_fmt(ci)} | {_fmt(block['given_detected'].get(name))} |")
        for name in constants.V6_PRIORS:
            lines.append(f"| AP - {name} | {_fmt(block[f'AP-{name}'])} | |")
    return "\n".join(lines) + "\n"


# =========================================================================== CLI


def _pair(text: str) -> tuple[str, str]:
    a, sep, b = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return a, b


def _shard(text: str) -> tuple[int, int]:
    i, sep, n = text.partition("/")
    if not sep or not (0 <= int(i) < int(n)):
        raise argparse.ArgumentTypeError(f"expected i/n with 0 <= i < n, got {text!r}")
    return int(i), int(n)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    dota = sub.add_parser("extract-dota", help="agent rows of DoTA-CAP clips (CAP stream)")
    dota.add_argument("--alignment", type=Path, required=True)
    dota.add_argument("--final-readout", type=Path, required=True, help="dota_cap_readout.json")
    dota.add_argument("--parts", type=Path, nargs="+", required=True)
    dota.add_argument("--work-dir", type=Path, required=True)
    dota.add_argument("--dota-clip-dir", type=Path, required=True, help="DoTA_s1_ncc (frame count)")
    dota.add_argument("--agent-dir", type=Path, required=True)
    dota.add_argument("--track-dir", type=Path, required=True)
    dota.add_argument("--report", type=Path, required=True)
    dota.add_argument("--cap-ids-file", type=Path, default=None)
    dota.add_argument("--dev-only", action="store_true", help="CAP-dev only (Step 0 / pilot)")
    dota.add_argument("--frames-subdir", default="images")
    dota.add_argument("--max-batch-gb", type=float, default=constants.MMAU_STREAM_BATCH_GB)
    dota.add_argument("--device", default="auto")
    dota.add_argument("--weights", type=Path, default=Path(constants.V6_DETECTOR_WEIGHTS))

    nexar = sub.add_parser("extract-nexar", help="agent rows of the Nexar window sources")
    nexar.add_argument("--raw", type=Path, required=True)
    nexar.add_argument("--census", type=Path, required=True)
    nexar.add_argument("--data-dir", type=Path, required=True, help="Nexar_window (windows.json)")
    nexar.add_argument("--clip-dir", type=Path, required=True, help="A2_S/Nexar_s8 (row counts)")
    nexar.add_argument("--agent-dir", type=Path, required=True)
    nexar.add_argument("--track-dir", type=Path, required=True)
    nexar.add_argument("--report", type=Path, required=True)
    nexar.add_argument("--ids-file", type=Path, default=None)
    nexar.add_argument("--shard", type=_shard, default=(0, 1), help="i/n: every n-th source from i")
    nexar.add_argument("--device", default="auto")
    nexar.add_argument("--weights", type=Path, default=Path(constants.V6_DETECTOR_WEIGHTS))

    step0 = sub.add_parser("step0", help="PREREG_V6 §1 go / no-go on DoTA-CAP-dev")
    step0.add_argument("--agent-dir", type=Path, required=True)
    step0.add_argument("--track-dir", type=Path, required=True)
    step0.add_argument("--annotations", type=Path, required=True)
    step0.add_argument("--out-dir", type=Path, required=True)
    step0.add_argument("--text-embeds", type=Path, default=None, help="(2, 512) npy; default: CLIP")
    step0.add_argument("--device", default="auto")

    dump = sub.add_parser("dump", help="candidate logits of an AP checkpoint on DoTA-CAP-dev")
    dump.add_argument("--ckpt", type=Path, required=True)
    dump.add_argument("--input-dir", type=Path, required=True, help="A2's DoTA-CAP s3 bake")
    dump.add_argument("--s1-dir", type=Path, required=True)
    dump.add_argument("--agent-dir", type=Path, required=True)
    dump.add_argument("--out", type=Path, required=True)
    dump.add_argument("--device", default="auto")

    read = sub.add_parser("readout", help="PREREG_V6 §3 detection + box-hit read-out")
    read.add_argument(
        "--row",
        type=_pair,
        action="append",
        required=True,
        help="NAME=protocol_b_eval out dir (repeatable)",
    )
    read.add_argument("--pair", type=_pair, action="append", default=[], help="A=B: paired Δ A - B")
    read.add_argument("--annotations", type=Path, required=True)
    read.add_argument("--logits", type=Path, default=None, help="dump output (box-level hit)")
    read.add_argument("--agent-dir", type=Path, default=None)
    read.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    args = build_parser().parse_args(argv)
    if args.command == "readout" and args.logits is not None and args.agent_dir is None:
        raise SystemExit("--logits needs --agent-dir")
    {
        "extract-dota": run_extract_dota,
        "extract-nexar": run_extract_nexar,
        "step0": run_step0,
        "dump": run_dump,
        "readout": run_readout,
    }[args.command](args)


if __name__ == "__main__":
    main()
