"""DF8 / SC-1: does ``p_T2`` still beat the detector when the accident is moved inside its clip?

v2 addendum §24 (Amendment 13, X1-X10); plan ``.project/plans/katvad-v2-df8-position-shift.md``.
No training. Every DoTA-CAP-dev clip with at least 1 s of context on both sides of its accident
is cropped four ways at the same length (``full`` is the whole clip): ``control`` keeps the
clip's own pre:post ratio, ``early`` puts the accident near the crop start (centre ≈ 0.30),
``late`` near the end (≈ 0.70). Each crop is scored as a clip of its own (protocol B).

Subcommands, in run order:

``plan`` (CPU)
    The crop table from the frozen labels (X2): ``df8_crops.json``. Refuses a clip count other
    than :data:`constants.V22_DF8_EXPECTED_CLIPS`.
``heads`` (GPU, CAP tar parts)
    D2 / X3: the V2-S rows whose 1.5 s causal window reaches before a crop's first frame are
    re-extracted from the CAP frames, clamped at that frame, plus one unclamped gate window per
    shifted arm (X10(c)). One ``heads/{clip}.npz`` per clip; resumable.
``score`` (GPU)
    One model's checkpoints on every arm: CLIP (+ V2-S, heads patched) rows of the crop,
    ``[::3]``, CRN + motion scaling with the arm's T2-train statistics on the crop's own rows,
    seed-averaged native scores. Gate X10(b): the ``full`` arm must reproduce the model's saved
    protocol-B scores.
``prior`` (CPU)
    ``p_T2`` (E3's T2 fit) at ``t / L`` of every crop; its ``full`` arm must reproduce E3's
    saved prior.
``readout`` (CPU)
    R1-R4, X7's non-inferiority, X8's sentence, X9's printed reads.
``same-frame`` (CPU, post-hoc, printed only — decides nothing)
    Same frames, different position: each shifted crop's AUC with its own scores minus the AUC of
    the *full* clip's scores on the identical frames. Content and labels are fixed, so the shift
    is what moving the frames (position + crop context) does. ``p_T2`` is the pure-position ruler;
    per clip, a model's shift is regressed on the prior's (``r``, slope). Also splits R2's
    ``early - late`` into a frame-set part (full-clip scores) and a move part.

CLI::

    python -m core.tools.position_shift plan --s1-dir cache/clip/DoTA_s1_ncc \\
        --metadata data/DoTA/metadata_val.json --split-file data/DoTA/val_split.txt --out-dir OUT
    python -m core.tools.position_shift heads --crops OUT/df8_crops.json \\
        --alignment .../dota_cap_alignment.json --parts .../11.part_* --cap-ids-file ids.txt \\
        --video-dir cache/video/<V2-S>/DoTA_CAP_s1_squash --work-dir /content/w --out-dir OUT/heads
    python -m core.tools.position_shift score --crops OUT/df8_crops.json --model A3ign \\
        --run s2024 CKPT ... --s1-dir ... --video-dir ... \\
        --stats-dir cache/v2/A3_R2_S/DADA2000_orig --heads-dir OUT/heads \\
        --data-dir data/DoTA/labels_s8 --reference .../clip_scores.npz \\
        --out-dir OUT
    python -m core.tools.position_shift prior --crops OUT/df8_crops.json \\
        --data-dir data/DADA2000_orig_v2 --reference .../position_prior.npz --out-dir OUT
    python -m core.tools.position_shift readout --crops OUT/df8_crops.json --scores-dir OUT \\
        --reference-model .../clip_scores.npz --reference-prior .../position_prior.npz --out-dir OUT
    python -m core.tools.position_shift same-frame --crops OUT/df8_crops.json --scores-dir OUT \\
        --out-dir OUT
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from core import constants
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_inputs import V2Stats, bake_rows, load_stats
from core.data.v2_splits import dota_group, load_split
from core.data.video_io import list_frame_images
from core.device import resolve_device
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.models.videomae_v2 import load_pretrained
from core.tools.dota_cap import rebuild_folder
from core.tools.extract_video_features import causal_clip_indices, encode_clips
from core.tools.feature_cache import read_ids_file
from core.tools.final_readout import fused_clip_aucs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.position_prior import prior_scores, training_frames
from core.tools.protocol_b_eval import clip_aucs, read_clip_scores
from core.tools.rate_matched_eval import (
    Protocol,
    Run,
    load_finished_model,
    native_labels,
    score_steps,
    seed_average,
    to_native,
)
from core.tools.stream_frames_clip import stream_video_batches
from core.train import load_class_names

LOGGER = logging.getLogger(__name__)

PLAN, HEADS, SCORE, PRIOR, READOUT = "plan", "heads", "score", "prior", "readout"
SAME_FRAME = "same-frame"
CROPS_JSON = "df8_crops.json"
CROPS_MD = "df8_crops.md"
HEADS_REPORT = "heads_report.json"
READOUT_JSON = "df8_readout.json"
READOUT_MD = "df8_readout.md"
SAME_FRAME_JSON = "df8_same_frame.json"
SAME_FRAME_MD = "df8_same_frame.md"
PRIOR_NAME = "pT2"
PRIMARY = "A3ign"
LEAKY = "_leaky"  # X9: D2 sensitivity arm suffix (cache slice, no re-extracted heads)
GATE_KEY = "gate_cos/"
OK = "ok"
# frames a protocol-B row's causal window reaches back (15 at DoTA's 10 fps)
REACH = (constants.VIDEOMAE_CLIP_FRAMES - 1) * constants.DOTA_VIDEOMAE_FRAME_STEP


def scores_npz(model: str) -> str:
    return f"df8_{model}.npz"


def scores_json(model: str) -> str:
    return f"df8_{model}.json"


# ---------------------------------------------------------------------------
# geometry (X2), pure
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Crop:
    """One arm of one clip: ``[c0, c0 + length)`` of the native frames, span starts ``before``."""

    c0: int
    length: int
    before: int
    span: int

    @property
    def centre(self) -> float:
        return (self.before + self.span / 2) / self.length


def span_of(labels: np.ndarray) -> tuple[int, int] | None:
    """``(s, e)`` of a two-class clip's single contiguous span; ``None`` if not two-class."""
    idx = np.flatnonzero(labels)
    if len(idx) == 0 or len(idx) == len(labels):
        return None
    s, e = int(idx[0]), int(idx[-1]) + 1
    if not labels[s:e].all():
        raise ValueError("the span is not contiguous: X2 assumes one span per clip")
    return s, e


def crop_geometry(
    labels: np.ndarray, k: int = constants.V22_DF8_K,
    min_context: int = constants.V22_DF8_MIN_CONTEXT,
) -> dict[str, Crop] | None:
    """The four arms of one clip, or ``None`` if it is not eligible (X2)."""
    span = span_of(labels)
    if span is None:
        return None
    s, e = span
    n = len(labels)
    pre, post = s, n - e
    if min(pre, post) < min_context:
        return None
    m = min(pre, post) + k
    width = e - s
    length = width + m
    before = {
        constants.V22_DF8_CONTROL: round(m * pre / (pre + post)),
        constants.V22_DF8_EARLY: k,
        constants.V22_DF8_LATE: m - k,
    }
    arms = {constants.V22_DF8_FULL: Crop(0, n, s, width)}
    for arm, b in before.items():
        if not (0 <= b <= pre and 0 <= m - b <= post):
            raise ValueError(
                f"{arm}: {b} before / {m - b} after does not fit pre {pre} post {post}"
            )
        arms[arm] = Crop(s - b, length, b, width)
    return arms


def crop_labels(crop: Crop) -> np.ndarray:
    y = np.zeros(crop.length, dtype=np.int64)
    y[crop.before : crop.before + crop.span] = 1
    return y


def head_rows(crop: Crop, stride: int = constants.V2_E1_STRIDE_BC) -> list[int]:
    """Crop-local protocol-B rows whose causal window reaches before ``c0`` (X3)."""
    if crop.c0 == 0:
        return []
    return list(range(0, min(REACH, crop.length), stride))


def head_windows(crop: Crop, rows: list[int]) -> list[list[int]]:
    """Absolute frame indices of each row's window, clamped at the crop's first frame."""
    step = constants.DOTA_VIDEOMAE_FRAME_STEP
    return [[crop.c0 + i for i in causal_clip_indices(r, step=step)] for r in rows]


def gate_frame(crop: Crop, native_frames: int) -> int | None:
    """X10(c): the first frame whose window is unclamped in both the crop and the cache."""
    end = crop.c0 + REACH
    return end if crop.c0 > 0 and end < native_frames else None


def motion_rows(
    u: np.ndarray, crop: Crop, heads: np.ndarray | None, stride: int = constants.V2_E1_STRIDE_BC
) -> np.ndarray:
    """The crop's V2-S rows with the re-extracted heads patched in (``None`` = the leaky slice)."""
    rows = np.array(u[crop.c0 : crop.c0 + crop.length], copy=True)
    idx = head_rows(crop, stride)
    if heads is not None and idx:
        if len(heads) != len(idx):
            raise ValueError(f"{len(heads)} head rows for {len(idx)} reaching rows")
        rows[idx] = heads
    return rows


def load_crops(path: Path) -> tuple[dict[str, dict[str, Crop]], dict[str, Any]]:
    table = json.loads(path.read_text(encoding="utf-8"))
    crops = {
        v: {arm: Crop(**c) for arm, c in entry["arms"].items()}
        for v, entry in table["clips"].items()
    }
    if len(crops) != constants.V22_DF8_EXPECTED_CLIPS:
        raise ValueError(f"X10(d): {len(crops)} crops, expected {constants.V22_DF8_EXPECTED_CLIPS}")
    return crops, table


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def build_table(labels: dict[str, np.ndarray]) -> dict[str, Any]:
    clips: dict[str, Any] = {}
    excluded: list[str] = []
    one_class: list[str] = []
    for v in sorted(labels):
        if span_of(labels[v]) is None:
            one_class.append(v)
            continue
        arms = crop_geometry(labels[v])
        if arms is None:
            excluded.append(v)
            continue
        clips[v] = {
            "native_frames": len(labels[v]),
            "arms": {a: c.__dict__ for a, c in arms.items()},
        }
    centres = {
        arm: np.percentile(
            [Crop(**e["arms"][arm]).centre for e in clips.values()], [10, 50, 90]
        ).round(3).tolist()
        for arm in constants.V22_DF8_ARMS
    } if clips else {}
    lengths = [e["arms"][constants.V22_DF8_EARLY]["length"] for e in clips.values()]
    return {
        "addendum": "v2 PREREG_ADDENDUM.md §24 (Amendment 13) X2",
        "k": constants.V22_DF8_K,
        "min_context": constants.V22_DF8_MIN_CONTEXT,
        "clips": clips,
        "excluded": excluded,
        "one_class": one_class,
        "centre_p10_p50_p90": centres,
        "crop_length_median": float(np.median(lengths)) if lengths else None,
    }


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    ids = load_split(constants.V2_SPLIT_DOTA_CAP_DEV, args.split_dir)
    parsed = parse_metadata(args.metadata, read_split_ids(args.split_file))
    records = {r.video_id: r for r in parsed}
    labels = {
        v: native_labels(records[v], len(np.load(args.s1_dir / f"{v}.npy", mmap_mode="r")))
        for v in ids
    }
    table = build_table(labels)
    n = len(table["clips"])
    if n != constants.V22_DF8_EXPECTED_CLIPS:
        raise ValueError(f"X10(d): {n} eligible clips, §24 X2 measured "
                         f"{constants.V22_DF8_EXPECTED_CLIPS}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / CROPS_JSON, table)
    write_text_atomic(args.out_dir / CROPS_MD, "\n".join([
        "# DF8 crop table (§24 X2)", "",
        f"Eligible {n} / {len(ids)} (excluded {len(table['excluded'])}, one-class "
        f"{len(table['one_class'])}); crop length median {table['crop_length_median']} frames.", "",
        "| arm | centre p10 / p50 / p90 |", "|---|---|",
        *(f"| {a} | {c} |" for a, c in table["centre_p10_p50_p90"].items()), "",
    ]))
    LOGGER.info("DF8 plan: %d clips -> %s", n, args.out_dir)
    return table


# ---------------------------------------------------------------------------
# heads (D2 / X3 / X10(c))
# ---------------------------------------------------------------------------
def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return np.asarray(
        (a64 * b64).sum(1) / (np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1))
    )


def clip_head_windows(
    arms: dict[str, Crop], native_frames: int
) -> tuple[list[list[int]], dict[str, slice], dict[str, tuple[int, int]]]:
    """Every window one clip needs: ``(windows, heads per arm, (gate idx, frame) per arm)``."""
    windows: list[list[int]] = []
    heads: dict[str, slice] = {}
    gates: dict[str, tuple[int, int]] = {}
    step = constants.DOTA_VIDEOMAE_FRAME_STEP
    for arm in constants.V22_DF8_SHIFTED:
        crop = arms[arm]
        rows = head_rows(crop)
        if not rows:
            continue
        start = len(windows)
        windows += head_windows(crop, rows)
        heads[arm] = slice(start, len(windows))
        frame = gate_frame(crop, native_frames)
        if frame is not None:
            gates[arm] = (len(windows), frame)
            windows.append(causal_clip_indices(frame, step=step))
    return windows, heads, gates


@dataclass
class HeadExtractor:
    """Per-batch callback of :func:`stream_video_batches` for ``heads``."""

    by_cap: dict[str, list[str]]
    alignment: dict[str, Any]
    crops: dict[str, dict[str, Crop]]
    native: dict[str, int]
    encoder: nn.Module
    video_dir: Path
    out_dir: Path
    farm: Path
    device: torch.device
    subdir: str
    batch_size: int
    report: dict[str, dict[str, Any]]
    save_report: Callable[[], None]

    def __call__(self, ready_dir: Path) -> None:
        for folder in sorted(p for p in ready_dir.iterdir() if p.is_dir()):
            cap_paths = list_frame_images(folder, self.subdir)
            for dota_id in self.by_cap.get(folder.name, []):
                self.report[dota_id] = self.one_clip(dota_id, cap_paths)
        self.save_report()

    def one_clip(self, dota_id: str, cap_paths: list[Path]) -> dict[str, Any]:
        clip = self.alignment["clips"][dota_id]
        if len(cap_paths) != clip["cap_frames"]:
            return {"reason": "cap_frames_mismatch", "cap_frames_streamed": len(cap_paths)}
        if len(clip["frame_map"]) != self.native[dota_id]:
            return {"reason": "frame_map_length"}
        windows, heads, gates = clip_head_windows(self.crops[dota_id], self.native[dota_id])
        farm = rebuild_folder(self.farm / dota_id, cap_paths, clip["frame_map"])
        try:
            feats = encode_clips(
                list_frame_images(farm, None), windows, self.encoder, self.device, self.batch_size
            )
        finally:
            shutil.rmtree(farm, ignore_errors=True)
        cache = np.load(self.video_dir / f"{dota_id}.npy", mmap_mode="r")
        arrays = {arm: feats[sl] for arm, sl in heads.items()}
        cos: dict[str, float] = {}
        for arm, (i, frame) in gates.items():
            cos[arm] = float(cosine_rows(feats[i : i + 1], np.asarray(cache[frame : frame + 1]))[0])
            arrays[f"{GATE_KEY}{arm}"] = np.asarray([cos[arm]])
        tmp = self.out_dir / f"{dota_id}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.replace(self.out_dir / f"{dota_id}.npz")
        low = min(cos.values()) if cos else None
        reason = OK if low is None or low >= constants.V22_DF8_HEAD_GATE_COS else "head_gate"
        return {"reason": reason, "windows": len(windows), "gate_min_cos": low}


def run_heads(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    crops, table = load_crops(args.crops)
    native = {v: int(e["native_frames"]) for v, e in table["clips"].items()}
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / HEADS_REPORT
    report: dict[str, dict[str, Any]] = (
        json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    )
    todo = [v for v in sorted(crops) if not (args.out_dir / f"{v}.npz").is_file()]
    if args.cap_ids_file is not None:
        group = read_ids_file(args.cap_ids_file)
        todo = [v for v in todo if alignment["clips"][v]["cap_id"] in group]
    if not todo:
        LOGGER.info("heads: nothing to extract for %s", args.parts[0].parent)
        return report
    by_cap: dict[str, list[str]] = {}
    for v in todo:
        by_cap.setdefault(alignment["clips"][v]["cap_id"], []).append(v)
    device = resolve_device(args.device)
    extractor = HeadExtractor(
        by_cap=by_cap, alignment=alignment, crops=crops, native=native,
        encoder=load_pretrained(args.encoder, device, None), video_dir=args.video_dir,
        out_dir=args.out_dir, farm=args.work_dir / "farm", device=device,
        subdir=args.frames_subdir, batch_size=args.batch_size, report=report,
        save_report=lambda: write_json_atomic(report_path, report),
    )
    LOGGER.info("heads: %d DoTA clips from %d CAP videos", len(todo), len(by_cap))
    stream_video_batches(
        sorted(args.parts), args.work_dir / "stream", args.frames_subdir,
        args.max_batch_gb * constants.BYTES_PER_GB, extractor, keep_ids=set(by_cap),
    )
    unseen = sorted(v for v in todo if v not in report)
    if unseen and args.cap_ids_file is not None:
        raise ValueError(f"{len(unseen)} clips' CAP videos were not in the tar: {unseen[:5]}")
    shutil.rmtree(args.work_dir, ignore_errors=True)
    return report


def load_heads(heads_dir: Path, video_id: str, arms: dict[str, Crop]) -> dict[str, np.ndarray]:
    """The re-extracted heads of one clip, after X10(c)'s gate."""
    path = heads_dir / f"{video_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{video_id}: no re-extracted heads at {path}")
    with np.load(path) as data:
        out = {k: data[k] for k in data.files}
    for arm in constants.V22_DF8_SHIFTED:
        if head_rows(arms[arm]) and arm not in out:
            raise ValueError(f"{video_id}: heads file has no {arm!r} rows")
        cos = out.get(f"{GATE_KEY}{arm}")
        if cos is not None and float(cos[0]) < constants.V22_DF8_HEAD_GATE_COS:
            raise RuntimeError(
                f"X10(c) FAILED {video_id}/{arm}: re-extracted window cos {float(cos[0]):.5f} "
                f"< {constants.V22_DF8_HEAD_GATE_COS} vs the cache"
            )
    return out


# ---------------------------------------------------------------------------
# score / prior
# ---------------------------------------------------------------------------
def write_arm_scores(path: Path, scores: dict[str, dict[str, np.ndarray]]) -> None:
    arrays = {f"{arm}/{v}": np.asarray(s, dtype=np.float64)
              for arm, per in scores.items() for v, s in per.items()}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def read_arm_scores(path: Path) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path) as data:
        for key in data.files:
            arm, v = key.split("/", 1)
            out.setdefault(arm, {})[v] = data[key]
    return out


def reference_gate(
    full: dict[str, np.ndarray], reference: Path, tol: float = constants.V2_FINAL_DEV_TOL
) -> float:
    """X10(b): the ``full`` arm must reproduce the saved protocol-B / prior scores per frame."""
    ref, _ = read_clip_scores(reference)
    worst = 0.0
    for v, s in full.items():
        if v not in ref or len(ref[v]) != len(s):
            raise RuntimeError(f"X10(b) FAILED: {v} missing or of another length in {reference}")
        worst = max(worst, float(np.abs(ref[v] - s).max()))
    if worst > tol:
        raise RuntimeError(
            f"X10(b) FAILED: full arm differs from {reference} by {worst:.2e} > {tol}"
        )
    LOGGER.info("X10(b) PASSED vs %s: max |diff| %.2e", reference, worst)
    return worst


def model_input(
    crop: Crop, x: np.ndarray, u: np.ndarray | None,
    heads: np.ndarray | None, stats: V2Stats | None,
) -> np.ndarray:
    """``[::3]`` rows of one crop, baked with the arm's T2-train statistics (X3)."""
    stride = constants.V2_E1_STRIDE_BC
    clip = x[crop.c0 : crop.c0 + crop.length][::stride]
    if stats is None:
        return np.ascontiguousarray(clip, dtype=np.float32)
    motion = None if u is None else motion_rows(u, crop, heads)[::stride]
    return bake_rows(clip, motion, stats)


def arm_plan(arms: dict[str, Crop], motion: bool) -> list[tuple[str, Crop, bool]]:
    """``(name, crop, patched)``; a motion model also scores X9's leaky slice of every shifted arm.

    A shifted arm that starts at frame 0 has no heads, so its leaky score equals the patched one.
    """
    out = [(a, arms[a], True) for a in constants.V22_DF8_ARMS]
    if motion:
        out += [(a + LEAKY, arms[a], False) for a in constants.V22_DF8_SHIFTED]
    return out


def score_one_run(
    run: Run, crops: dict[str, dict[str, Crop]], args: argparse.Namespace, device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], int, dict[str, str]]:
    model, step, text_encode_fn, cfg = load_finished_model(run, device)
    plain = cfg.v2.crn == constants.V2_OFF and cfg.v2.motion == constants.V2_OFF
    stats = None if plain else load_stats(args.stats_dir)
    if stats is not None and (stats.crn, stats.motion) != (cfg.v2.crn, cfg.v2.motion):
        raise ValueError(f"{run.name}: stats are {stats.crn}/{stats.motion}, checkpoint "
                         f"{cfg.v2.crn}/{cfg.v2.motion}")
    has_motion = stats is not None and stats.has_motion
    class_names = load_class_names(args.data_dir)
    protocol = Protocol(constants.V2_E1_STRIDE_BC, None)
    out: dict[str, dict[str, np.ndarray]] = {}
    for v, arms in sorted(crops.items()):
        x = np.load(args.s1_dir / f"{v}.npy", mmap_mode="r")
        u = np.load(args.video_dir / f"{v}.npy", mmap_mode="r") if has_motion else None
        if u is not None and len(u) != len(x):
            raise ValueError(f"{v}: V2-S rows {len(u)} != CLIP rows {len(x)} (C13)")
        heads = load_heads(args.heads_dir, v, arms) if has_motion else {}
        for name, crop, patched in arm_plan(arms, has_motion):
            arm = name.removesuffix(LEAKY)
            rows = model_input(crop, x, u, heads.get(arm) if patched else None, stats)
            steps = score_steps(
                model, torch.from_numpy(rows), text_encode_fn, class_names, v, protocol
            )
            out.setdefault(name, {})[v] = to_native(steps, protocol.stride, crop.length)
    LOGGER.info("%s (step %d): scored %d clips x %d arms", run.name, step, len(crops), len(out))
    return out, step, {"crn": cfg.v2.crn, "motion": cfg.v2.motion}


def macro(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> float:
    return float(np.mean([frame_auc(scores[v], labels[v]) for v in scores]))


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    crops, _ = load_crops(args.crops)
    runs = [Run(name, Path(ckpt)) for name, ckpt in args.run]
    device = resolve_device(args.device)
    per_run: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    steps: dict[str, int] = {}
    inputs: set[str] = set()
    for r in runs:
        per_run[r.name], steps[r.name], arm = score_one_run(r, crops, args, device)
        inputs.add(json.dumps(arm, sort_keys=True))
    if len(inputs) != 1:
        raise ValueError(f"checkpoints of different arms cannot be seed-averaged: {inputs}")
    names = sorted(per_run)
    arms = sorted(per_run[names[0]])
    averaged = {a: seed_average([per_run[n][a] for n in names]) for a in arms}
    worst = reference_gate(averaged[constants.V22_DF8_FULL], args.reference)
    labels = {a: {v: crop_labels(crops[v][a.removesuffix(LEAKY)]) for v in crops} for a in arms}
    meta = {
        "model": args.model,
        "input": json.loads(next(iter(inputs))),
        "runs": names,
        "checkpoints": {r.name: str(r.ckpt) for r in runs},
        "global_steps": steps,
        "reference": str(args.reference),
        "x10b_max_abs_diff": worst,
        "per_run_macro": {
            a: {n: macro(per_run[n][a], labels[a]) for n in names} for a in arms
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_arm_scores(args.out_dir / scores_npz(args.model), averaged)
    write_json_atomic(args.out_dir / scores_json(args.model), meta)
    return meta


def run_prior(args: argparse.Namespace) -> dict[str, Any]:
    crops, _ = load_crops(args.crops)
    matrix, target, windows = training_frames(args.data_dir)
    scores = {
        arm: prior_scores(matrix, target, {v: c[arm].length for v, c in crops.items()}, args.seed)
        for arm in constants.V22_DF8_ARMS
    }
    worst = reference_gate(scores[constants.V22_DF8_FULL], args.reference)
    meta = {"model": PRIOR_NAME, "train_windows": windows, "reference": str(args.reference),
            "x10b_max_abs_diff": worst}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_arm_scores(args.out_dir / scores_npz(PRIOR_NAME), scores)
    write_json_atomic(args.out_dir / scores_json(PRIOR_NAME), meta)
    return meta


# ---------------------------------------------------------------------------
# readout (X5-X9)
# ---------------------------------------------------------------------------
def arm_aucs(
    scores: dict[str, dict[str, np.ndarray]], crops: dict[str, dict[str, Crop]]
) -> dict[str, dict[str, float]]:
    return {
        a: {v: frame_auc(s, crop_labels(crops[v][a.removesuffix(LEAKY)])) for v, s in per.items()}
        for a, per in scores.items()
    }


def shifted(aucs: dict[str, dict[str, float]]) -> dict[str, float]:
    """D6: per clip, the mean of the early and late AUCs."""
    early, late = aucs[constants.V22_DF8_EARLY], aucs[constants.V22_DF8_LATE]
    return {v: (early[v] + late[v]) / 2 for v in early}


def diff(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {v: a[v] - b[v] for v in a}


def sentence(r1: dict[str, float], noninferior: bool) -> tuple[str, str]:
    """X8, mechanically: ``(outcome, sentence)``."""
    if r1["low"] > 0:
        if noninferior:
            return "R1>0, X7 holds", (
                "The position prior's advantage is a property of where DoTA-CAP places its "
                "accidents; it does not survive a position swap, whereas the detector's score "
                "does.")
        return "R1>0, X7 fails", (
            "The prior's advantage does not survive a position swap, but the detector also depends "
            "on where the accident sits.")
    if r1["high"] < 0:
        return "R1<0", (
            "A position prior remains competitive even when positions are swapped; the detector's "
            "use of position is an open problem.")
    return "R1 contains 0", (
        "Once the accident is moved, neither the detector nor the position prior is ahead at this "
        "size.")


def span_bin(span_frames: int) -> str:
    seconds = span_frames / constants.DOTA_FPS
    lo, hi = constants.V22_DF8_SPAN_BINS_SECONDS
    return f"<{lo:g}s" if seconds < lo else (f"{lo:g}-{hi:g}s" if seconds < hi else f">={hi:g}s")


def leaky_r1(
    p: dict[str, dict[str, float]], prior: dict[str, dict[str, float]],
    ci: Callable[[dict[str, float]], dict[str, float] | None],
) -> dict[str, float] | None:
    """X9 D2 sensitivity: R1 with the leaky cache slice in place of the re-extracted heads."""
    early, late = constants.V22_DF8_EARLY + LEAKY, constants.V22_DF8_LATE + LEAKY
    if early not in p or late not in p:
        return None
    return ci(diff(shifted({constants.V22_DF8_EARLY: p[early], constants.V22_DF8_LATE: p[late]}),
                   shifted(prior)))


def run_readout(args: argparse.Namespace) -> dict[str, Any]:
    crops, table = load_crops(args.crops)
    groups = {v: dota_group(v) for v in crops}

    def ci(values: dict[str, float]) -> dict[str, float] | None:
        return cluster_bootstrap_ci(
            values, {v: dota_group(v) for v in values}, constants.V2_E1_BOOTSTRAP, args.seed,
            constants.V2_E1_CI,
        )

    models = [m for m in (PRIMARY, "A3", "A0")
              if (args.scores_dir / scores_npz(m)).is_file()]
    if PRIMARY not in models or not (args.scores_dir / scores_npz(PRIOR_NAME)).is_file():
        raise FileNotFoundError(f"readout needs {scores_npz(PRIMARY)} and {scores_npz(PRIOR_NAME)}")
    metas = {m: json.loads((args.scores_dir / scores_json(m)).read_text(encoding="utf-8"))
             for m in [*models, PRIOR_NAME]}
    scores = {m: read_arm_scores(args.scores_dir / scores_npz(m)) for m in [*models, PRIOR_NAME]}
    for m, per in scores.items():
        for a, s in per.items():
            if set(s) != set(crops):
                raise ValueError(f"{m}/{a} covers {len(s)} clips, not the {len(crops)} crops")
    aucs = {m: arm_aucs(per, crops) for m, per in scores.items()}
    prior = aucs[PRIOR_NAME]
    p = aucs[PRIMARY]
    early, late, control = (p[constants.V22_DF8_EARLY], p[constants.V22_DF8_LATE],
                            p[constants.V22_DF8_CONTROL])
    r1_values = diff(shifted(p), shifted(prior))
    r1 = ci(r1_values)
    r2 = {
        "early - control": ci(diff(early, control)),
        "late - control": ci(diff(late, control)),
        "early - late": ci(diff(early, late)),
    }
    if r1 is None or r2["early - control"] is None or r2["late - control"] is None:
        raise ValueError("empty read")
    noninferior = all(
        r2[k]["low"] > -constants.V22_DF8_NONINFERIOR  # type: ignore[index]
        for k in ("early - control", "late - control")
    )
    outcome, text = sentence(r1, noninferior)
    weight = constants.V2_FINAL_FUSION_PRIMARY
    labels_by_arm = {
        a: {v: crop_labels(crops[v][a]) for v in crops} for a in constants.V22_DF8_ARMS
    }
    f2 = {
        m: {a: ci(fused_clip_aucs(scores[m][a], scores[PRIOR_NAME][a], labels_by_arm[a], weight))
            for a in constants.V22_DF8_ARMS}
        for m in models
    }
    bins: dict[str, dict[str, float]] = {}
    for v, d in r1_values.items():
        bins.setdefault(span_bin(crops[v][constants.V22_DF8_FULL].span), {})[v] = d
    ref_model, ref_labels = read_clip_scores(args.reference_model)
    ref_prior, _ = read_clip_scores(args.reference_prior)
    dropped = [v for v in table["excluded"] if v in ref_model]
    kept_vs_dropped = {
        name: {
            "kept": ci(clip_aucs({v: ref[v] for v in crops}, ref_labels)),
            "dropped": ci(clip_aucs({v: ref[v] for v in dropped}, ref_labels)),
        }
        for name, ref in ((PRIMARY, ref_model), (PRIOR_NAME, ref_prior))
    }
    readout: dict[str, Any] = {
        "addendum": "v2 PREREG_ADDENDUM.md §24 (Amendment 13)",
        "set": f"DoTA-CAP-dev shift subset, n = {len(crops)}",
        "clusters": len(set(groups.values())),
        "centres": table["centre_p10_p50_p90"],
        "gates": {m: metas[m]["x10b_max_abs_diff"] for m in metas},
        "global_steps": {m: metas[m].get("global_steps") for m in models},
        "macro": {m: {a: ci(v) for a, v in aucs[m].items()} for m in [*models, PRIOR_NAME]},
        "per_run_macro": {m: metas[m]["per_run_macro"] for m in models},
        "r1": r1,
        "r1_full": ci(diff(p[constants.V22_DF8_FULL], prior[constants.V22_DF8_FULL])),
        "r2": r2,
        "noninferior": noninferior,
        "r3_full_minus_control": ci(diff(p[constants.V22_DF8_FULL], control)),
        "outcome": outcome,
        "sentence": f"{text} (DoTA-CAP-dev shift subset, n = {len(crops)}; accident centre "
                    "0.30-0.70)",
        "f2": f2,
        "r1_by_span": {b: ci(v) for b, v in sorted(bins.items())},
        "r1_leaky": leaky_r1(p, prior, ci),
        "kept_vs_dropped_full_clip": kept_vs_dropped,
        "dropped_clips": len(dropped),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render(readout))
    LOGGER.info("DF8 readout: %s", outcome)
    return readout


def _f(c: dict[str, float] | None, signed: bool = False) -> str:
    if c is None:
        return "—"
    fmt = "{:+.4f}" if signed else "{:.4f}"
    return f"{fmt.format(c['mean'])} [{fmt.format(c['low'])}, {fmt.format(c['high'])}]"


def render(r: dict[str, Any]) -> str:
    arms = list(constants.V22_DF8_ARMS)
    models = [m for m in r["macro"] if m != PRIOR_NAME]
    lines = [
        "# DF8 / SC-1 — position swap (v2 addendum §24)", "",
        f"{r['set']}, {r['clusters']} source clusters, protocol B per crop. Centres (p10/p50/p90): "
        + "; ".join(f"{a} {c}" for a, c in r["centres"].items()) + ".", "",
        "X10(b) full-arm max |diff| vs saved scores: "
        + ", ".join(f"{m} {d:.1e}" for m, d in r["gates"].items()) + ".", "",
        f"## Verdict (X8): **{r['outcome']}**", "", f"> {r['sentence']}", "",
        f"* **R1** shifted(A3-ign) - shifted(`p_T2`): **{_f(r['r1'], True)}** "
        f"(same Δ on the full clips: {_f(r['r1_full'], True)})",
        *(f"* R2 {k}: {_f(v, True)}" for k, v in r["r2"].items()),
        f"* X7 non-inferiority (both moved - control lower bounds > "
        f"-{constants.V22_DF8_NONINFERIOR}): **{'holds' if r['noninferior'] else 'fails'}**",
        f"* R3 full - control (context cost): {_f(r['r3_full_minus_control'], True)}", "",
        "## Macro per arm (seed-averaged; R4 = the `p_T2` row)", "",
        "| model | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms),
        *(f"| {m} | " + " | ".join(_f(r["macro"][m].get(a)) for a in arms) + " |"
          for m in [*models, PRIOR_NAME]), "",
        "## Printed only (X9)", "",
        "`f_2` per arm: " + "; ".join(
            f"{m}: " + ", ".join(f"{a} {_f(r['f2'][m][a])}" for a in arms) for m in models),
        "",
        "R1 by span length: " + "; ".join(f"{b} {_f(c, True)}" for b, c in r["r1_by_span"].items()),
        "",
        f"D2 sensitivity, R1 with the leaky cache slice: {_f(r['r1_leaky'], True)}", "",
        f"Full-clip macros, kept vs dropped ({r['dropped_clips']} dropped): "
        + "; ".join(f"{m} kept {_f(c['kept'])} / dropped {_f(c['dropped'])}"
                    for m, c in r["kept_vs_dropped_full_clip"].items()), "",
        "Per-seed macros: " + "; ".join(
            f"{m}/{a}: " + ", ".join(f"{n} {x:.4f}" for n, x in per.items())
            for m, by_arm in r["per_run_macro"].items() for a, per in by_arm.items()
            if a in (constants.V22_DF8_EARLY, constants.V22_DF8_LATE)), "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# same-frame (post-hoc, printed only; the §24 verdict is X8's)
# ---------------------------------------------------------------------------
def same_frame_aucs(
    arm_scores: dict[str, np.ndarray], full_scores: dict[str, np.ndarray],
    crops: dict[str, dict[str, Crop]], arm: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per clip: the crop's AUC on its own scores and on the full clip's scores of its frames."""
    own: dict[str, float] = {}
    full: dict[str, float] = {}
    for v, s in arm_scores.items():
        c = crops[v][arm]
        y = crop_labels(c)
        own[v] = frame_auc(s, y)
        full[v] = frame_auc(full_scores[v][c.c0 : c.c0 + c.length], y)
    return own, full


def tracking(shift: dict[str, float], ruler: dict[str, float]) -> dict[str, float] | None:
    """Per-clip Pearson ``r`` and OLS slope of a model's shift on the position ruler's shift."""
    x = np.array([ruler[v] for v in shift])
    y = np.array([shift[v] for v in shift])
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return {"r": float(np.corrcoef(x, y)[0, 1]), "slope": float(np.polyfit(x, y, 1)[0])}


def run_same_frame(args: argparse.Namespace) -> dict[str, Any]:
    crops, _ = load_crops(args.crops)

    def ci(values: dict[str, float]) -> dict[str, float] | None:
        return cluster_bootstrap_ci(
            values, {v: dota_group(v) for v in values}, constants.V2_E1_BOOTSTRAP, args.seed,
            constants.V2_E1_CI,
        )

    names = [m for m in (PRIMARY, "A3", "A0", PRIOR_NAME)
             if (args.scores_dir / scores_npz(m)).is_file()]
    if PRIOR_NAME not in names:
        raise FileNotFoundError(f"same-frame needs {scores_npz(PRIOR_NAME)} as the ruler")
    shifts: dict[str, dict[str, dict[str, float]]] = {}
    out: dict[str, Any] = {
        "note": "post-hoc, printed only; decides nothing (v2 addendum §24's verdict is X8's)",
        "set": f"DoTA-CAP-dev shift subset, n = {len(crops)}",
        "models": {},
    }
    early, late = constants.V22_DF8_EARLY, constants.V22_DF8_LATE
    for m in names:
        scores = read_arm_scores(args.scores_dir / scores_npz(m))
        per_arm: dict[str, Any] = {}
        pairs: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
        for arm in constants.V22_DF8_SHIFTED:
            own, full = same_frame_aucs(scores[arm], scores[constants.V22_DF8_FULL], crops, arm)
            pairs[arm] = own, full
            shifts.setdefault(m, {})[arm] = diff(own, full)
            per_arm[arm] = {"crop": ci(own), "full_same_frames": ci(full),
                            "shift": ci(shifts[m][arm])}
        out["models"][m] = {
            "arms": per_arm,
            "early_late": {
                "total": ci(diff(pairs[early][0], pairs[late][0])),
                "frame_set": ci(diff(pairs[early][1], pairs[late][1])),
                "move": ci(diff(shifts[m][early], shifts[m][late])),
            },
        }
    for m in names:
        if m != PRIOR_NAME:
            out["models"][m]["tracking"] = {
                arm: tracking(shifts[m][arm], shifts[PRIOR_NAME][arm])
                for arm in constants.V22_DF8_SHIFTED
            }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / SAME_FRAME_JSON, out)
    write_text_atomic(args.out_dir / SAME_FRAME_MD, render_same_frame(out))
    LOGGER.info("DF8 same-frame: %s", ", ".join(names))
    return out


def render_same_frame(r: dict[str, Any]) -> str:
    arms = list(constants.V22_DF8_SHIFTED)
    lines = [
        "# DF8 — same frames, different position (post-hoc)", "",
        f"{r['set']}. {r['note']}.", "",
        "Shift = AUC of the crop's own scores - AUC of the full clip's scores on the same frames "
        "(content and labels fixed). `pT2` is the pure-position ruler.", "",
        "| model | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms),
        *(f"| {m} | " + " | ".join(_f(d["arms"][a]["shift"], True) for a in arms) + " |"
          for m, d in r["models"].items()), "",
        "## Per-clip tracking of the ruler's shift (r, slope)", "",
        *(f"* {m}: " + "; ".join(
            f"{a} " + ("—" if t is None else f"r {t['r']:+.3f}, slope {t['slope']:+.3f}")
            for a, t in d["tracking"].items())
          for m, d in r["models"].items() if "tracking" in d), "",
        "## early - late = frame set + move", "",
        "| model | total (crop scores) | frame set (full-clip scores) "
        "| move (shift early - late) |",
        "|---|---|---|---|",
        *(f"| {m} | " + " | ".join(_f(d["early_late"][k], True)
                                    for k in ("total", "frame_set", "move")) + " |"
          for m, d in r["models"].items()), "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser(PLAN, help="crop table from the frozen labels (X2)")
    plan.add_argument("--s1-dir", type=Path, required=True)
    plan.add_argument("--metadata", type=Path, required=True)
    plan.add_argument("--split-file", type=Path, required=True)
    plan.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    plan.add_argument("--out-dir", type=Path, required=True)
    heads = sub.add_parser(HEADS, help="re-extract the clamped V2-S heads (D2, X10(c))")
    heads.add_argument("--crops", type=Path, required=True)
    heads.add_argument("--alignment", type=Path, required=True)
    heads.add_argument("--parts", type=Path, nargs="+", required=True)
    heads.add_argument("--cap-ids-file", type=Path, default=None)
    heads.add_argument("--video-dir", type=Path, required=True, help="the DoTA-CAP V2-S cache")
    heads.add_argument("--encoder", default=constants.VIDEOMAE_ENCODER_S)
    heads.add_argument("--work-dir", type=Path, required=True)
    heads.add_argument("--frames-subdir", default="images")
    heads.add_argument("--batch-size", type=int, default=32)
    heads.add_argument("--max-batch-gb", type=float, default=constants.MMAU_STREAM_BATCH_GB)
    heads.add_argument("--device", default="auto")
    heads.add_argument("--out-dir", type=Path, required=True)
    score = sub.add_parser(SCORE, help="one model's checkpoints on every arm")
    score.add_argument("--crops", type=Path, required=True)
    score.add_argument("--model", required=True, choices=(PRIMARY, "A3", "A0"))
    score.add_argument("--run", nargs=2, action="append", required=True, metavar=("NAME", "CKPT"))
    score.add_argument("--s1-dir", type=Path, required=True)
    score.add_argument("--video-dir", type=Path, default=None)
    score.add_argument("--stats-dir", type=Path, default=None, help="the arm's T2 `fit` cache")
    score.add_argument("--heads-dir", type=Path, default=None)
    score.add_argument("--data-dir", type=Path, required=True, help="DoTA labels dir (defs.json)")
    score.add_argument("--reference", type=Path, required=True,
                       help="the model's saved protocol-B clip_scores.npz on dota_cap_dev")
    score.add_argument("--device", default="auto")
    score.add_argument("--out-dir", type=Path, required=True)
    prior = sub.add_parser(PRIOR, help="p_T2 at t / L of every crop")
    prior.add_argument("--crops", type=Path, required=True)
    prior.add_argument("--data-dir", type=Path, required=True, help="the T2 v2 dataset dir")
    prior.add_argument("--reference", type=Path, required=True, help="E3's position_prior.npz")
    prior.add_argument("--seed", type=int, default=constants.SEED)
    prior.add_argument("--out-dir", type=Path, required=True)
    readout = sub.add_parser(READOUT, help="R1-R4, X7, X8")
    readout.add_argument("--crops", type=Path, required=True)
    readout.add_argument("--scores-dir", type=Path, required=True)
    readout.add_argument("--reference-model", type=Path, required=True,
                         help="A3-ign's saved clip_scores.npz on dota_cap_dev (kept vs dropped)")
    readout.add_argument("--reference-prior", type=Path, required=True)
    readout.add_argument("--seed", type=int, default=constants.SEED)
    readout.add_argument("--out-dir", type=Path, required=True)
    same = sub.add_parser(SAME_FRAME, help="post-hoc: same frames, different position")
    same.add_argument("--crops", type=Path, required=True)
    same.add_argument("--scores-dir", type=Path, required=True)
    same.add_argument("--seed", type=int, default=constants.SEED)
    same.add_argument("--out-dir", type=Path, required=True)
    return parser


RUNNERS: dict[str, Callable[[argparse.Namespace], Any]] = {
    PLAN: run_plan, HEADS: run_heads, SCORE: run_score, PRIOR: run_prior, READOUT: run_readout,
    SAME_FRAME: run_same_frame,
}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == SCORE and args.model != "A0" and args.stats_dir is None:
        raise SystemExit("a CRN / motion model needs --stats-dir")
    RUNNERS[args.command](args)


if __name__ == "__main__":
    main()
