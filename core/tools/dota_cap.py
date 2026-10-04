"""DoTA-CAP: recover DoTA pixels from CAP-DATA and extract VideoMAE features on them.

Addendum §11 (Amendment 4, D13-D15; L1-L7 fixed before any motion score). MM-AU Phase 0 found
1,248 / 1,397 DoTA clips inside CAP-DATA at containment ``kappa >= 0.99`` (``exact``). This
module turns those pairs into a per-frame map and a VideoMAE cache row-aligned with
``DoTA_s1_ncc``. Three subcommands:

``align`` (CPU, CLIP caches only, no pixels, no labels)
    For every ``exact`` pair: the strictly increasing CAP frame for each DoTA frame ``d`` that
    maximizes the summed cosine (dynamic programming over the two s1 CLIP caches; independent of
    fps -- CAP stores some DoTA clips at 10 fps, some at 30). **Amendment 4a (L1'):** CLIP cannot
    tell two 30 fps frames 33 ms apart, so the DP path jitters by 1-2 frames around the true one;
    the map used is ``j_d = round(rate * d + offset)`` from a Theil-Sen fit on that path, which
    is uniform in time by construction. **Amendment 4b:** the line's end can fall a CAP frame or
    two past the clip, so it may move by ``|k| < rate`` CAP frames (less than one DoTA frame);
    the in-range shift with the best mean cosine is used. **Amendment 4c:** within that clock each
    DoTA frame takes the best CAP frame at most ``band_width(rate)`` frames off the line (1 at
    rate 3, 0 at rate 1) -- a resampled source leaves a bounded rounding residual. A clip is kept
    iff (L1) CAP has at least as many frames and the line stays inside the CAP clip, strictly
    increasing, (L2) mean cosine of the mapped frames >= ``DOTA_CAP_MEAN_COS`` and every frame >=
    ``DOTA_CAP_MIN_COS``
    (a CAP stream that is not a uniform resample of DoTA fails here), (L3) at most
    ``DOTA_CAP_MAX_IRREGULAR`` of its steps differ from the median step by more than
    ``DOTA_CAP_STEP_TOL``.

``extract`` (GPU, one CAP group's tar parts)
    Streams only the kept CAP videos (:func:`stream_video_batches` with ``keep_ids``), builds a
    symlink folder ``{d:06d}.jpg -> CAP frame j_d`` per DoTA clip, and gates it at pixel level:
    (L4) the streamed CAP folder has exactly the CAP cache's row count, (L5) CLIP of the
    rebuilt frames (``no_center_crop``) against ``DoTA_s1_ncc`` passes L2's thresholds. Only then
    is VideoMAE run: DoTA's native 10 fps, 16 frames every ``DOTA_VIDEOMAE_FRAME_STEP`` frame,
    causal, clamped at DoTA frame 0, stride 1 (L6). Several encoders share one pass over the tar.

``finalize`` (CPU)
    The DoTA-CAP id list = clips that passed L1-L5 and have a feature file for every encoder
    (L7); counts on all DoTA, DoTA-dev and the rest (DoTA-eval ids are never read).

``freeze`` (CPU, local, then commit -- L7)
    Splits the finalized list into ``dota_cap_dev`` (DoTA-CAP ∩ DoTA-dev) and the sealed
    ``dota_cap_eval`` (DoTA-CAP minus DoTA-dev; DoTA-eval's ids are not read) and writes them with
    ``DOTA_CAP_MANIFEST.json`` beside the base splits. Refuses an id list whose sha1, ids or
    coverage disagree with the read-out, and refuses to overwrite without ``--force``;
    ``--check`` recomputes and raises if the committed files differ.

CLI::

    python -m core.tools.dota_cap align --matches .../mmau_p0/all/mmau_p0_matches.json \\
        --dota-clip-dir cache/clip/DoTA_s1_ncc --cap-clip-dir cache/clip/MMAU_CAP_s1_ncc \\
        --out-dir outputs/v2/REPORTS/dota_cap
    python -m core.tools.dota_cap extract --alignment .../dota_cap_alignment.json \\
        --parts data/MMAU/raw/CAP-DATA_chunks/11/11.part_* --work-dir /content/dcap/work \\
        --dota-clip-dir cache/clip/DoTA_s1_ncc --report .../extract_11.json \\
        [--encoders vit_b_k710_dl_from_giant vit_s_k710_dl_from_giant] [--cap-ids-file ids.txt]
    python -m core.tools.dota_cap finalize --alignment ... --reports .../extract_*.json \\
        [--encoders ...] --out-dir outputs/v2/REPORTS/dota_cap
    python -m core.tools.dota_cap freeze --readout .../dota_cap_readout.json \\
        --ids-file .../dota_cap_ids.txt [--check | --force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from core import constants
from core.data.v2_splits import dota_group, lines_sha1, load_split, read_manifest
from core.data.video_io import list_frame_images
from core.device import resolve_device
from core.models.videomae_v2 import load_pretrained
from core.tools import extract_clip_features, extract_video_features
from core.tools.feature_cache import is_complete, read_ids_file, save_array
from core.tools.freeze_splits import check_splits, file_sha1, write_splits
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.mmau_match import GRADE_EXACT, normalize_rows
from core.tools.stream_frames_clip import stream_video_batches

LOGGER = logging.getLogger(__name__)

ALIGNMENT_JSON = "dota_cap_alignment.json"
ALIGN_MD = "dota_cap_align.md"
IDS_TXT = "dota_cap_ids.txt"
FINAL_JSON = "dota_cap_readout.json"
FINAL_MD = "dota_cap_readout.md"
FARM_DIR = "farm"
OK = "ok"
FRAME_SOURCE = "CAP-DATA frames aligned to DoTA_s1_ncc (core.tools.dota_cap align)"


@dataclass(frozen=True)
class Alignment:
    """One DoTA clip's CAP frame map and the L1-L3 verdict."""

    dota_id: str
    cap_id: str
    dota_frames: int
    cap_frames: int
    reason: str  # OK, or the first gate that failed
    frame_map: list[int]
    mean_cos: float | None = None
    min_cos: float | None = None
    median_step: float | None = None
    irregular_share: float | None = None
    rate: float | None = None  # Theil-Sen CAP frames per DoTA frame
    dp_jitter_share: float | None = None  # DP frames > 2 CAP frames off the line (printed only)
    phase_shift: int | None = None  # Amendment 4b: CAP frames the line was moved (|k| < rate)
    overshoot: int | None = None  # CAP frames the unshifted line leaves the clip by (0 = inside)
    line_mean_cos: float | None = None  # straight line's mean cosine, before 4c's band (printed)


def monotone_alignment(sims: np.ndarray) -> np.ndarray | None:
    """Strictly increasing ``j_d`` maximizing ``sum_d sims[d, j_d]``; None if CAP is shorter."""
    n_query, n_cap = sims.shape
    if n_cap < n_query:
        return None
    index = np.arange(n_cap)
    score = sims[0].astype(np.float64)
    back = np.zeros((n_query, n_cap), dtype=np.int64)
    for d in range(1, n_query):
        running = np.maximum.accumulate(score)
        arg = np.maximum.accumulate(np.where(score >= running, index, 0))
        prev = np.full(n_cap, -np.inf)
        prev[1:] = running[:-1]  # best path ending strictly before j
        back[d, 1:] = arg[:-1]
        score = sims[d] + prev
    path = np.empty(n_query, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for d in range(n_query - 1, 0, -1):
        path[d - 1] = back[d, path[d]]
    return path


def theil_sen(path: np.ndarray) -> tuple[float, float]:
    """``(rate, offset)`` of ``j ~ rate * d + offset``: median pairwise slope, then
    ``median(j) - rate * median(d)`` (scipy ``theilslopes``' default intercept)."""
    d = np.arange(len(path), dtype=np.float64)
    if len(path) < 2:
        return 1.0, float(path[0])
    i, k = np.triu_indices(len(path), 1)
    rate = float(np.median((path[k] - path[i]) / (d[k] - d[i])))
    return rate, float(np.median(path) - rate * np.median(d))


def linear_map(path: np.ndarray) -> tuple[np.ndarray, float]:
    """Amendment 4a: ``j_d = round(rate * d + offset)`` from a Theil-Sen fit on the DP path."""
    rate, offset = theil_sen(path.astype(np.float64))
    # half up, not np.round's half-to-even: with an odd integer rate and a .5 offset (an even-length
    # clip's median d) half-to-even alternates the steps 4, 2, 4, 2 instead of 3, 3, 3
    line = rate * np.arange(len(path)) + offset
    return np.floor(line + 0.5).astype(np.int64), rate


def phase_candidates(line: np.ndarray, rate: float, n_cap: int) -> list[tuple[int, np.ndarray]]:
    """Amendment 4b: the line moved by ``k`` CAP frames, ``|k| < rate``, kept iff inside the clip.

    A shift shorter than one DoTA frame interval re-picks the sub-frame phase only; it cannot hand
    a DoTA frame its neighbour's content. At rate 1 that leaves ``k = 0`` alone.
    """
    reach = max(0, int(np.ceil(rate)) - 1)
    out = []
    for k in sorted(range(-reach, reach + 1), key=abs):
        frames = line + k
        if frames.min() >= 0 and frames.max() < n_cap:
            out.append((k, frames))
    return out


def band_width(rate: float) -> int:
    """Amendment 4c: CAP frames a DoTA frame may sit off the line -- under half a DoTA interval."""
    return max(0, int(np.ceil(rate / 2)) - 1)


def banded_path(sims: np.ndarray, line: np.ndarray, band: int) -> np.ndarray:
    """Strictly increasing max-``sum cos`` path with ``|j_d - line_d| <= band`` (Amendment 4c).

    The line fixes the clock (no drift); the band absorbs the bounded rounding residual of two
    nearest-frame resamplings of one source. ``line`` itself is always feasible (steps >= 1).
    """
    cols = np.arange(sims.shape[1])
    inside = np.abs(cols[None, :] - line[:, None]) <= band
    path = monotone_alignment(np.where(inside, sims, -np.inf))
    if path is None:  # unreachable: the caller only passes in-range lines
        raise ValueError("banded alignment needs a CAP clip at least as long as the query")
    return path


def gate_alignment(dota_id: str, cap_id: str, query: np.ndarray, cap: np.ndarray) -> Alignment:
    """L1-L3 on unit CLIP rows of one (DoTA clip, CAP clip) pair (L1' = Amendments 4a-4c)."""
    sims = query @ cap.T
    path = monotone_alignment(sims)
    if path is None:
        return Alignment(dota_id, cap_id, len(query), len(cap), "cap_shorter", [])
    line, rate = linear_map(path)
    jitter = float(np.mean(np.abs(line - path) > constants.DOTA_CAP_STEP_TOL + 1))
    overshoot = int(max(0, -line.min(), line.max() - (len(cap) - 1)))
    rows = np.arange(len(query))
    candidates = phase_candidates(line, rate, len(cap))
    if not candidates:
        return Alignment(dota_id, cap_id, len(query), len(cap), "out_of_range", [],
                         rate=rate, dp_jitter_share=jitter, overshoot=overshoot)
    band = band_width(rate)
    banded = [(k, line_k, banded_path(sims, line_k, band)) for k, line_k in candidates]
    # the phase is the one free parameter CLIP leaves at 30 fps: take the best-matching one
    shift, straight, frames = max(banded, key=lambda c: float(sims[rows, c[2]].mean()))
    line_cos = float(sims[rows, straight].mean())
    steps = np.diff(frames)
    if np.any(steps <= 0):
        return Alignment(dota_id, cap_id, len(query), len(cap), "not_increasing", [],
                         rate=rate, dp_jitter_share=jitter, phase_shift=shift,
                         overshoot=overshoot, line_mean_cos=line_cos)
    cos = sims[rows, frames]
    median = float(np.median(steps)) if len(steps) else 1.0
    # L3 under 4c: steps vs the fitted rate, widened by the band's two ends (a guard -- the band
    # already bounds the clock; at band 0 it is L3 as first written)
    tol = constants.DOTA_CAP_STEP_TOL + 2 * band
    irregular = float(np.mean(np.abs(steps - rate) > tol)) if len(steps) else 0.0
    if float(cos.mean()) < constants.DOTA_CAP_MEAN_COS:
        reason = "mean_cos"
    elif float(cos.min()) < constants.DOTA_CAP_MIN_COS:
        reason = "min_cos"
    elif irregular > constants.DOTA_CAP_MAX_IRREGULAR:
        reason = "irregular_steps"
    else:
        reason = OK
    return Alignment(
        dota_id, cap_id, len(query), len(cap), reason, [int(j) for j in frames],
        float(cos.mean()), float(cos.min()), median, irregular, rate, jitter, shift, overshoot,
        line_cos,
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_cosines(rebuilt: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-frame cosine of two equally long CLIP row arrays."""
    if rebuilt.shape != reference.shape:
        raise ValueError(f"row mismatch: rebuilt {rebuilt.shape} vs reference {reference.shape}")
    cos: np.ndarray = np.sum(normalize_rows(rebuilt) * normalize_rows(reference), axis=1)
    return cos


def passes_frame_gate(cos: np.ndarray) -> bool:
    return bool(
        cos.mean() >= constants.DOTA_CAP_MEAN_COS and cos.min() >= constants.DOTA_CAP_MIN_COS
    )


# --------------------------------------------------------------------------- align


def run_align(args: argparse.Namespace) -> dict[str, Any]:
    matches = json.loads(args.matches.read_text(encoding="utf-8"))["dota"]
    rows: list[Alignment] = []
    for m in sorted(matches, key=lambda r: r["query"]):
        if m["grade"] != GRADE_EXACT:
            rows.append(Alignment(m["query"], m["cap_id"], m["frames"], 0, f"p0_{m['grade']}", []))
            continue
        query = normalize_rows(np.load(args.dota_clip_dir / f"{m['query']}.npy"))
        cap = normalize_rows(np.load(args.cap_clip_dir / f"{m['cap_id']}.npy"))
        rows.append(gate_alignment(m["query"], m["cap_id"], query, cap))
    return {
        "matches": str(args.matches),
        "matches_sha256": file_sha256(args.matches),
        "thresholds": {
            "mean_cos": constants.DOTA_CAP_MEAN_COS,
            "min_cos": constants.DOTA_CAP_MIN_COS,
            "step_tol": constants.DOTA_CAP_STEP_TOL,
            "max_irregular": constants.DOTA_CAP_MAX_IRREGULAR,
        },
        "clips": {r.dota_id: asdict(r) for r in rows},
    }


def coverage(passed: set[str], all_ids: set[str], dev: set[str]) -> dict[str, dict[str, int]]:
    """Kept / total on all DoTA, DoTA-dev and the rest (eval ids are the complement)."""
    rest = all_ids - dev
    return {
        name: {"kept": len(passed & ids), "of": len(ids)}
        for name, ids in (("all", all_ids), ("dev", dev), ("rest_eval", rest))
    }


def _cov_lines(cov: dict[str, dict[str, int]]) -> list[str]:
    return [
        "| set | kept | of | share |",
        "|---|---|---|---|",
        *(
            f"| {k} | {v['kept']} | {v['of']} | {v['kept'] / v['of']:.3f} |"
            for k, v in cov.items()
            if v["of"]
        ),
    ]


def _hist(clips: dict[str, Any], reason: str, key: str) -> str:
    counts = Counter(c.get(key) for c in clips.values() if c["reason"] == reason)
    return ", ".join(f"{k} → {v}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])))


def _band_line(clips: dict[str, Any]) -> str:
    out = []
    for reason in (OK, "mean_cos", "min_cos"):
        rows = [c for c in clips.values()
                if c["reason"] == reason and (c.get("rate") or 0) > 1 and c.get("line_mean_cos")]
        if rows:
            line = np.median([c["line_mean_cos"] for c in rows])
            band = np.median([c["mean_cos"] for c in rows])
            out.append(f"{reason} n={len(rows)} {line:.4f} → {band:.4f}")
    return "; ".join(out)


def render_align(alignment: dict[str, Any], dev: set[str]) -> str:
    clips = alignment["clips"]
    reasons = Counter(c["reason"] for c in clips.values())
    passed = {k for k, c in clips.items() if c["reason"] == OK}
    steps = Counter(c["median_step"] for c in clips.values() if c["reason"] == OK)
    th = alignment["thresholds"]
    return "\n".join([
        "# DoTA-CAP — alignment (L1-L3, CLIP caches only)",
        "",
        f"Rule: addendum §11-§11.4. mean cos ≥ {th['mean_cos']}, every frame ≥ {th['min_cos']}, "
        f"irregular steps (|Δj - rate| > {th['step_tol']} + 2·band) ≤ {th['max_irregular']:.0%}.",
        f"Matches: `{alignment['matches']}` (sha256 `{alignment['matches_sha256'][:12]}`)",
        "",
        "Reasons: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())),
        "Median CAP step of kept clips (Amendments 4a-4c): "
        + ", ".join(f"{k:g} → {v}" for k, v in sorted(steps.items())),
        "Phase shift of kept clips (Amendment 4b, CAP frames): " + _hist(clips, OK, "phase_shift"),
        "Rate > 1, mean cos straight line → band, q50 (Amendment 4c): " + _band_line(clips),
        "Overshoot of `out_of_range` clips (CAP frames): "
        + _hist(clips, "out_of_range", "overshoot"),
        "",
        *_cov_lines(coverage(passed, set(clips), dev)),
        "",
    ])


# --------------------------------------------------------------------------- extract


def rebuild_folder(farm: Path, cap_paths: list[Path], frame_map: list[int]) -> Path:
    """``farm/{d:06d}{suffix}`` -> ``cap_paths[j_d]``: DoTA's frame sequence, without copying."""
    shutil.rmtree(farm, ignore_errors=True)
    farm.mkdir(parents=True)
    for d, j in enumerate(frame_map):
        source = cap_paths[j]
        (farm / f"{d:06d}{source.suffix}").symlink_to(source.resolve())
    return farm


@dataclass
class Extractor:
    """Per-batch callback of :func:`run_extract`: L4-L6 for every DoTA clip in the batch."""

    by_cap: dict[str, list[dict[str, Any]]]
    clip_encoder: extract_clip_features.ImageEncoder
    encoders: dict[str, nn.Module]
    out_dirs: dict[str, Path]
    dota_clip_dir: Path
    farm: Path
    device: torch.device
    subdir: str
    batch_size: int
    report: dict[str, dict[str, Any]]
    save_report: Callable[[], None]
    frame_order: list[int] | None = None  # D6: permute every window's 16 frames

    def __call__(self, ready_dir: Path) -> None:
        for folder in sorted(p for p in ready_dir.iterdir() if p.is_dir()):
            cap_paths = list_frame_images(folder, self.subdir)
            for clip in self.by_cap.get(folder.name, []):
                self.report[clip["dota_id"]] = self.one_clip(clip, cap_paths)
        self.save_report()

    def one_clip(self, clip: dict[str, Any], cap_paths: list[Path]) -> dict[str, Any]:
        entry: dict[str, Any] = {"cap_id": clip["cap_id"], "cap_frames_streamed": len(cap_paths)}
        if len(cap_paths) != clip["cap_frames"]:
            return {**entry, "reason": "cap_frames_mismatch"}
        farm = rebuild_folder(self.farm / clip["dota_id"], cap_paths, clip["frame_map"])
        try:
            rebuilt = extract_clip_features.encode_frame_dir(
                farm, self.clip_encoder, self.device, stride=1, batch_size=self.batch_size,
                center_crop=False,
            )
            cos = frame_cosines(rebuilt, np.load(self.dota_clip_dir / f"{clip['dota_id']}.npy"))
            entry |= {"clip_mean_cos": float(cos.mean()), "clip_min_cos": float(cos.min())}
            if not passes_frame_gate(cos):
                return {**entry, "reason": "pixel_clip_cos"}
            for name, encoder in self.encoders.items():
                features = extract_video_features.encode_frame_dir(
                    farm, encoder, self.device, stride=1, batch_size=self.batch_size,
                    clip_step=constants.DOTA_VIDEOMAE_FRAME_STEP, frame_order=self.frame_order,
                )
                save_array(self.out_dirs[name] / f"{clip['dota_id']}.npy", features)
        finally:
            shutil.rmtree(farm, ignore_errors=True)
        LOGGER.info("%s <- CAP %s: %d frames, CLIP cos mean %.4f min %.4f",
                    clip["dota_id"], clip["cap_id"], len(clip["frame_map"]),
                    entry["clip_mean_cos"], entry["clip_min_cos"])
        return {**entry, "reason": OK}


def video_out_dir(encoder: str, shuffle_seed: int | None = None) -> Path:
    """The ordered cache, or the D6 shuffle-control cache of ``shuffle_seed``."""
    name = f"{constants.DOTA_CAP_DATASET}_s1_squash"
    if shuffle_seed is not None:
        name += f"_{constants.V2_D6_SHUFFLE_TAG}{shuffle_seed}"
    return constants.VIDEO_CACHE_DIR / encoder / name


def video_manifest(
    encoder: str, alignment_sha256: str, shuffle_seed: int | None = None
) -> dict[str, Any]:
    """The P1 manifest with DoTA's geometry and the frame source (lesson C2).

    A shuffle-control cache also records its seed and the permutation, so it can never be
    read as the ordered cache.
    """
    manifest = extract_video_features.build_manifest(
        encoder, 1, constants.VIDEOMAE_WEIGHTS_SHA256[encoder]
    ) | {
        "clip_frame_step": constants.DOTA_VIDEOMAE_FRAME_STEP,
        "assumed_fps": constants.DOTA_FPS,
        "frame_source": FRAME_SOURCE,
        "alignment_sha256": alignment_sha256,
        "frame_order": None,
        "shuffle_seed": None,
    }
    if shuffle_seed is not None:
        manifest |= {
            "frame_order": extract_video_features.shuffled_frame_order(shuffle_seed),
            "shuffle_seed": shuffle_seed,
        }
    return manifest


def todo_clips(
    alignment: dict[str, Any], out_dirs: dict[str, Path], report: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aligned clips not yet decided: no report entry, or an ok entry missing a feature file."""
    todo = []
    for dota_id, clip in sorted(alignment["clips"].items()):
        if clip["reason"] != OK:
            continue
        done = report.get(dota_id)
        if done is not None and (
            done["reason"] != OK
            or all(is_complete(d / f"{dota_id}.npy") for d in out_dirs.values())
        ):
            continue
        todo.append(clip)
    return todo


def run_extract(args: argparse.Namespace) -> None:
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    alignment_sha = file_sha256(args.alignment)
    seed = args.shuffle_seed
    out_dirs = {name: video_out_dir(name, seed) for name in args.encoders}
    manifests = {name: video_manifest(name, alignment_sha, seed) for name in args.encoders}
    for name, out in out_dirs.items():
        extract_video_features.check_or_write_manifest(out, manifests[name], force=False)
    report: dict[str, dict[str, Any]] = (
        json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    )
    todo = todo_clips(alignment, out_dirs, report)
    if args.cap_ids_file is not None:
        in_group = read_ids_file(args.cap_ids_file)
        todo = [c for c in todo if c["cap_id"] in in_group]
    if args.dota_ids_file is not None:
        wanted = read_ids_file(args.dota_ids_file)
        todo = [c for c in todo if c["dota_id"] in wanted]
    if not todo:
        LOGGER.info("Nothing to extract for %s", args.parts[0].parent)
        return
    missing = [
        c["dota_id"] for c in todo if not (args.dota_clip_dir / f"{c['dota_id']}.npy").exists()
    ]
    if missing:  # fail before loading models or reading a tar part
        raise FileNotFoundError(
            f"{len(missing)} DoTA CLIP rows missing under {args.dota_clip_dir}: {missing[:5]}"
        )
    by_cap: dict[str, list[dict[str, Any]]] = {}
    for clip in todo:
        by_cap.setdefault(clip["cap_id"], []).append(clip)
    device = resolve_device(args.device)
    extractor = Extractor(
        by_cap=by_cap,
        clip_encoder=extract_clip_features.load_pretrained_encoder(device),
        encoders={n: load_pretrained(n, device, None) for n in args.encoders},
        out_dirs=out_dirs,
        dota_clip_dir=args.dota_clip_dir,
        farm=args.work_dir / FARM_DIR,
        device=device,
        subdir=args.frames_subdir,
        batch_size=args.batch_size,
        report=report,
        save_report=lambda: write_json_atomic(args.report, report),
        frame_order=None if seed is None else manifests[args.encoders[0]]["frame_order"],
    )
    LOGGER.info("Extracting %d DoTA clips from %d CAP videos", len(todo), len(by_cap))
    stream_video_batches(
        sorted(args.parts), args.work_dir / "stream", args.frames_subdir,
        args.max_batch_gb * constants.BYTES_PER_GB, extractor, keep_ids=set(by_cap),
    )
    unseen = sorted(c["dota_id"] for c in todo if c["dota_id"] not in report)
    if unseen and args.cap_ids_file is not None:
        raise ValueError(f"{len(unseen)} DoTA clips' CAP videos were not in the tar: {unseen[:5]}")
    shutil.rmtree(args.work_dir, ignore_errors=True)


# --------------------------------------------------------------------------- finalize


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    report: dict[str, dict[str, Any]] = {}
    for path in args.reports:
        report |= json.loads(path.read_text(encoding="utf-8"))
    clips = alignment["clips"]
    reasons: Counter[str] = Counter()
    passed: set[str] = set()
    for dota_id, clip in clips.items():
        if clip["reason"] != OK:
            reasons[f"align:{clip['reason']}"] += 1
            continue
        entry = report.get(dota_id)
        if entry is None:
            reasons["extract:not_run"] += 1
        elif entry["reason"] != OK:
            reasons[f"extract:{entry['reason']}"] += 1
        elif not all(
            is_complete(video_out_dir(n) / f"{dota_id}.npy") for n in args.encoders
        ):
            reasons["extract:feature_missing"] += 1
        else:
            passed.add(dota_id)
            reasons[OK] += 1
    dev = set(load_split(constants.V2_SPLIT_DOTA_DEV, args.split_dir))
    ids = sorted(passed)
    return {
        "alignment_sha256": file_sha256(args.alignment),
        "encoders": list(args.encoders),
        "reasons": dict(sorted(reasons.items())),
        "coverage": coverage(passed, set(clips), dev),
        "ids_sha1": lines_sha1(ids),
        "ids": ids,
        "clip_cos_min_of_kept": min(
            (report[i]["clip_min_cos"] for i in ids), default=None
        ),
    }


def render_final(readout: dict[str, Any]) -> str:
    return "\n".join([
        "# DoTA-CAP — final subset (L1-L7)",
        "",
        f"Encoders: {', '.join(readout['encoders'])} · ids sha1 `{readout['ids_sha1']}` · "
        f"alignment sha256 `{readout['alignment_sha256'][:12]}`",
        f"Lowest per-frame CLIP cosine among kept clips: {readout['clip_cos_min_of_kept']}",
        "",
        "Reasons: " + ", ".join(f"{k} {v}" for k, v in readout["reasons"].items()),
        "",
        *_cov_lines(readout["coverage"]),
        "",
        "Name it **DoTA-CAP (n/1397)**; never beside a full-DoTA number or LaGoVAD's 62.60.",
        "",
    ])


# --------------------------------------------------------------------------- freeze


def freeze_dota_cap(
    readout: dict[str, Any], ids: list[str], dev: set[str]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """The two DoTA-CAP splits and their manifest block (L7).

    ``ids`` is the finalized id file; it must be the read-out's list, fingerprint and coverage.
    The eval side is the complement of DoTA-dev, so DoTA-eval's ids are never read.
    """
    if len(set(ids)) != len(ids):
        raise ValueError("the DoTA-CAP id file has duplicate ids")
    if sorted(ids) != sorted(readout["ids"]) or lines_sha1(ids) != readout["ids_sha1"]:
        raise ValueError(
            f"the id file (sha1 {lines_sha1(ids)}) is not the read-out's list "
            f"(sha1 {readout['ids_sha1']})"
        )
    for clip_id in ids:
        dota_group(clip_id)  # raises on anything that is not a DoTA clip id
    cap_dev = sorted(set(ids) & dev)
    cap_eval = sorted(set(ids) - dev)
    coverage_ = readout["coverage"]
    if (len(cap_dev), len(cap_eval)) != (
        coverage_["dev"]["kept"], coverage_["rest_eval"]["kept"]
    ):
        raise ValueError(
            f"dev / eval sides {len(cap_dev)} / {len(cap_eval)} disagree with the read-out's "
            f"coverage {coverage_['dev']['kept']} / {coverage_['rest_eval']['kept']}"
        )
    splits = {
        constants.V2_SPLIT_DOTA_CAP_DEV: cap_dev,
        constants.V2_SPLIT_DOTA_CAP_EVAL: cap_eval,
    }
    manifest = {
        "amendment": "core/docs/v2/PREREG_ADDENDUM.md §11 (Amendment 4, 4a-4c), L7",
        "rule": (
            f"{constants.V2_SPLIT_DOTA_CAP_DEV} = DoTA-CAP ∩ {constants.V2_SPLIT_DOTA_DEV}; "
            f"{constants.V2_SPLIT_DOTA_CAP_EVAL} = DoTA-CAP minus {constants.V2_SPLIT_DOTA_DEV} "
            "(DoTA-eval's ids are not read)"
        ),
        "naming": "DoTA-CAP (n/1397); never beside a full-DoTA number or LaGoVAD's 62.60 (D14)",
        "sealed": sorted(constants.V2_SEALED_SPLITS & splits.keys()),
        "source": {
            "ids_sha1": readout["ids_sha1"],
            "alignment_sha256": readout["alignment_sha256"],
            "encoders": list(readout["encoders"]),
            "clip_cos_min_of_kept": readout["clip_cos_min_of_kept"],
            "reasons": readout["reasons"],
            "coverage": coverage_,
        },
        "splits": {
            name: {"count": len(split), "sha1": lines_sha1(split)}
            for name, split in splits.items()
        },
    }
    return splits, manifest


def run_freeze(args: argparse.Namespace) -> None:
    readout: dict[str, Any] = json.loads(args.readout.read_text(encoding="utf-8"))
    ids = [
        line.strip()
        for line in args.ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev = set(load_split(constants.V2_SPLIT_DOTA_DEV, args.split_dir))
    splits, manifest = freeze_dota_cap(readout, ids, dev)
    base = read_manifest(args.split_dir)["splits"]
    manifest["derived_from"] = {
        constants.V2_SPLIT_DOTA_DEV: base[constants.V2_SPLIT_DOTA_DEV]["sha1"],
        "readout_sha1": file_sha1(args.readout),
    }
    sides = {name: len(split) for name, split in splits.items()}
    if args.check:
        check_splits(splits, manifest, args.split_dir, constants.V2_DOTA_CAP_MANIFEST_FILENAME)
        LOGGER.info("Frozen DoTA-CAP splits %s match the read-out", sides)
        return
    write_splits(
        splits, manifest, args.split_dir, args.force, constants.V2_DOTA_CAP_MANIFEST_FILENAME
    )
    LOGGER.info("Froze DoTA-CAP splits %s -> %s", sides, args.split_dir)


# --------------------------------------------------------------------------- CLI


def _add_encoders(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoders", nargs="+", default=[constants.VIDEOMAE_ENCODER_B],
                        choices=sorted(constants.VIDEOMAE_ARCH))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    align = sub.add_parser("align", help="L1-L3 on the CLIP caches")
    align.add_argument("--matches", type=Path, required=True, help="mmau_p0_matches.json")
    align.add_argument("--dota-clip-dir", type=Path, required=True, help="DoTA_s1_ncc")
    align.add_argument("--cap-clip-dir", type=Path, required=True, help="MMAU_CAP_s1_ncc")
    align.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    align.add_argument("--out-dir", type=Path, required=True)

    extract = sub.add_parser("extract", help="L4-L6 on one CAP group's tar parts")
    extract.add_argument("--alignment", type=Path, required=True)
    extract.add_argument("--parts", type=Path, nargs="+", required=True)
    extract.add_argument("--work-dir", type=Path, required=True, help="VM disk, not Drive")
    extract.add_argument("--dota-clip-dir", type=Path, required=True, help="DoTA_s1_ncc")
    extract.add_argument("--report", type=Path, required=True, help="per-group JSON, resumable")
    extract.add_argument("--cap-ids-file", type=Path, default=None,
                         help="CAP ids in this group: restricts the work, stops the stream early")
    extract.add_argument("--dota-ids-file", type=Path, default=None,
                         help="restrict to these DoTA ids (D6: dota_cap_dev only)")
    extract.add_argument("--shuffle-seed", type=int, default=None,
                         help="D6 shuffle control: permute every window's frames with this seed; "
                              "writes the *_shuf<seed> cache, never the ordered one")
    _add_encoders(extract)
    extract.add_argument("--frames-subdir", default="images")
    extract.add_argument("--batch-size", type=int, default=16)
    extract.add_argument("--max-batch-gb", type=float, default=constants.MMAU_STREAM_BATCH_GB)
    extract.add_argument("--device", default="auto")

    final = sub.add_parser("finalize", help="L7: the DoTA-CAP id list and coverage")
    final.add_argument("--alignment", type=Path, required=True)
    final.add_argument("--reports", type=Path, nargs="+", required=True)
    _add_encoders(final)
    final.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    final.add_argument("--out-dir", type=Path, required=True)

    freeze = sub.add_parser("freeze", help="L7: freeze dota_cap_dev / dota_cap_eval")
    freeze.add_argument("--readout", type=Path, required=True, help="dota_cap_readout.json")
    freeze.add_argument("--ids-file", type=Path, required=True, help="dota_cap_ids.txt")
    freeze.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    mode = freeze.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="recompute and verify the committed DoTA-CAP splits; write nothing")
    mode.add_argument("--force", action="store_true",
                      help="overwrite (only before any motion-arm score is read)")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "align":
        alignment = run_align(args)
        dev = set(load_split(constants.V2_SPLIT_DOTA_DEV, args.split_dir))
        write_json_atomic(args.out_dir / ALIGNMENT_JSON, alignment)
        write_text_atomic(args.out_dir / ALIGN_MD, render_align(alignment, dev))
        LOGGER.info("alignment -> %s", args.out_dir / ALIGNMENT_JSON)
    elif args.command == "extract":
        run_extract(args)
    elif args.command == "freeze":
        run_freeze(args)
    else:
        readout = run_finalize(args)
        write_json_atomic(args.out_dir / FINAL_JSON, readout)
        write_text_atomic(args.out_dir / IDS_TXT, "".join(f"{i}\n" for i in readout["ids"]))
        write_text_atomic(args.out_dir / FINAL_MD, render_final(readout))
        LOGGER.info("DoTA-CAP: %d clips -> %s", len(readout["ids"]), args.out_dir / IDS_TXT)


if __name__ == "__main__":
    main()
