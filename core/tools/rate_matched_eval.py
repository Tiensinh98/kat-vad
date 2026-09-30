"""v2 E1: rate-matched DoTA evaluation on existing checkpoints (no training).

Proposal §7.3 / §10.1, pre-registered in ``core/docs/v2/PREREG_ADDENDUM.md`` §8
(Amendment 2, choices J1-J8) and §9 (Amendment 3: D12, J9', J10). Every arm reads
the same stride-1 CLIP cache ``DoTA_s1_ncc`` and is scored against the same
native-frame labels:

* **A** = ``s1[::8]``, whole clip (today's protocol, 0.8 s/step);
* **B** = ``s1[::3]``, whole clip (0.3 s/step, T2's rate);
* **C** = ``s1[::3]`` in windows of 20 steps, hop 4, overlap-averaged.

Step ``t`` sits at native frame ``t * stride`` and is linearly interpolated to
every native frame (J3). Per-frame scores are averaged over the checkpoints (J4);
the decision is the per-clip paired macro Δ vs A with a source-video cluster
bootstrap (J6), under the J7 rule. A checkpoint is scored only if its stored
``global_step`` equals the last step of the ``metrics.jsonl`` beside it (J10), and
before B or C is scored, A's step-level curve from ``s1[::8]`` must equal the same
checkpoint's curve on the ``DoTA_s8_ncc`` cache on every DoTA-dev clip (J9').
DoTA-eval is sealed: only DoTA-dev ids are loaded.

CLI::

    python -m core.tools.rate_matched_eval \\
        --run s2024 runs/s2024/stage2_kip_off/checkpoint_last.pt \\
        --run s2025 ... --run s2026 ... \\
        --s1-dir cache/clip/DoTA_s1_ncc --s8-dir cache/clip/DoTA_s8_ncc \\
        --metadata data/DoTA/metadata_val.json \\
        --split-file data/DoTA/val_split.txt --data-dir data/DoTA/labels_s8 \\
        --out-dir outputs/v2/REPORTS/v2_E1
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core import constants
from core.config import load_config
from core.data.definitions import dataset_abbr, item_verbalizer
from core.data.dota import DotaRecord, parse_metadata, read_split_ids, resized_frame_labels
from core.data.v2_splits import dota_group, load_split, share_bin
from core.device import resolve_device
from core.inference import load_model_for_scoring, make_class_feats_fn, sliding_window_scores
from core.metrics import cluster_bootstrap_ci, frame_auc, macro_video_auc, pooled_metrics
from core.models.kat_vad import KATVAD
from core.models.text_encoding import TEXT_ENCODER_CLIP, TextEncodeFn, make_text_encoder
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.train import METRICS_FILENAME, load_class_names

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "e1_readout.json"
READOUT_MD = "e1_readout.md"
DOTA_OVERRIDES = [f"data.dataset={constants.DOTA_DATASET}"]
SCORE_NORM_MINMAX = "minmax"
KEEP_A = "A"


@dataclass(frozen=True)
class Protocol:
    """One E1 arm: sampling stride and window (``None`` = whole clip)."""

    stride: int
    window: int | None


PROTOCOLS = {
    "A": Protocol(constants.V2_E1_STRIDE_A, None),
    "B": Protocol(constants.V2_E1_STRIDE_BC, None),
    "C": Protocol(constants.V2_E1_STRIDE_BC, constants.V2_E1_WINDOW),
}


@dataclass(frozen=True)
class Run:
    """One checkpoint to re-score."""

    name: str
    ckpt: Path


@dataclass(frozen=True)
class Clip:
    """One DoTA-dev clip: stride-1 features, native labels, share and cluster."""

    video_id: str
    s1: np.ndarray
    labels: np.ndarray
    share: float
    group: str


# ---------------------------------------------------------------------------
# pure pieces (tested without a model)
# ---------------------------------------------------------------------------
def window_starts(length: int, window: int, hop: int) -> list[int]:
    """Window starts covering ``[0, length)``: every ``hop`` steps plus a tail window (J2)."""
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, hop))
    if starts[-1] + window < length:
        starts.append(length - window)
    return starts


def overlap_average(parts: list[tuple[int, np.ndarray]], length: int) -> np.ndarray:
    """Per-step mean over the windows covering each step; an uncovered step raises."""
    total = np.zeros(length)
    count = np.zeros(length)
    for start, scores in parts:
        total[start : start + len(scores)] += scores
        count[start : start + len(scores)] += 1
    if (count == 0).any():
        raise ValueError(f"{int((count == 0).sum())} of {length} steps are covered by no window")
    return total / count


def to_native(step_scores: np.ndarray, stride: int, num_frames: int) -> np.ndarray:
    """Linear interpolation of step ``t`` (at frame ``t * stride``) to frames ``0..N-1`` (J3)."""
    positions = np.arange(len(step_scores)) * stride
    return np.interp(np.arange(num_frames), positions, step_scores)


def native_labels(record: DotaRecord, num_frames: int) -> np.ndarray:
    """Stride-1 labels over the cache's frame count, by the baseline's arithmetic (J1)."""
    return np.asarray(resized_frame_labels(record, num_frames, 1), dtype=np.int64)


def regression_mismatches(
    harness: dict[str, np.ndarray], reference: dict[str, np.ndarray], atol: float
) -> list[str]:
    """Clips whose step-level A curve (``s1[::8]``) differs from the s8-cache curve (J9')."""
    bad = []
    for video_id, curve in sorted(harness.items()):
        ref = reference.get(video_id)
        if ref is None:
            bad.append(f"{video_id}: no reference curve")
        elif ref.shape != curve.shape:
            bad.append(f"{video_id}: {curve.shape} vs {ref.shape} steps")
        elif not np.allclose(curve, ref, rtol=0.0, atol=atol):
            bad.append(f"{video_id}: max |diff| {np.abs(curve - ref).max():.6f}")
    return bad


def metrics_last_step(metrics: Path) -> int:
    """The last ``global_step`` logged in a run's ``metrics.jsonl``."""
    lines = [line for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{metrics} is empty")
    return int(json.loads(lines[-1])["global_step"])


def check_finished(ckpt_step: int, metrics: Path, name: str) -> None:
    """J10: refuse a checkpoint whose step is not the run's last logged step."""
    if not metrics.is_file():
        raise FileNotFoundError(f"J10: {name} has no {metrics.name} beside its checkpoint")
    last = metrics_last_step(metrics)
    if ckpt_step != last:
        raise RuntimeError(
            f"J10 FAILED for {name}: checkpoint global_step {ckpt_step} != "
            f"{metrics.name} last step {last} (a mid-training snapshot)"
        )
    LOGGER.info("J10 PASSED for %s: finished checkpoint at step %d", name, last)


def decide(deltas: dict[str, dict[str, float] | None]) -> dict[str, Any]:
    """The J7 rule on the B and C paired-Δ intervals, applied mechanically."""
    eligible = {
        arm: ci for arm, ci in deltas.items() if ci is not None and ci["low"] > 0.0
    }
    if not eligible:
        adopted = KEEP_A
    elif len(eligible) == 1:
        adopted = next(iter(eligible))
    else:
        gap = abs(eligible["C"]["mean"] - eligible["B"]["mean"])
        if gap < constants.V2_E1_TIE_MARGIN:
            adopted = "B"
        else:
            adopted = max(eligible, key=lambda arm: eligible[arm]["mean"])
    return {"eligible": sorted(eligible), "adopted": adopted}


def clip_aucs(scores: dict[str, np.ndarray], clips: dict[str, Clip]) -> dict[str, float]:
    """Per-clip native-frame AUC over the two-class clips."""
    return {
        v: frame_auc(scores[v], clips[v].labels)
        for v in sorted(clips)
        if 0 < int(clips[v].labels.sum()) < len(clips[v].labels)
    }


def seed_average(per_run: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Per-frame mean over checkpoints (J4)."""
    return {v: np.mean([run[v] for run in per_run], axis=0) for v in per_run[0]}


# ---------------------------------------------------------------------------
# data + scoring
# ---------------------------------------------------------------------------
def load_dev_clips(s1_dir: Path, metadata: Path, split_file: Path) -> tuple[dict[str, Clip], int]:
    """DoTA-dev clips from the s1 cache; returns them and the count off the annotation length."""
    dev = load_split(constants.V2_SPLIT_DOTA_DEV)
    records = {r.video_id: r for r in parse_metadata(metadata, read_split_ids(split_file))}
    missing = [v for v in dev if not (s1_dir / f"{v}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} DoTA-dev ids have no s1 feature: {missing[:5]}")
    clips: dict[str, Clip] = {}
    off_length = 0
    for video_id in dev:
        record = records[video_id]
        s1 = np.load(s1_dir / f"{video_id}.npy")
        off_length += int(len(s1) != record.total_frames)
        clips[video_id] = Clip(
            video_id,
            s1,
            native_labels(record, len(s1)),
            record.span[1] - record.span[0],
            dota_group(video_id),
        )
    LOGGER.info("DoTA-dev: %d clips, %d with s1 length != annotation", len(clips), off_length)
    return clips, off_length


def score_steps(
    model: KATVAD,
    feats: torch.Tensor,
    text_encode_fn: TextEncodeFn,
    class_names: list[str],
    video_id: str,
    protocol: Protocol,
) -> np.ndarray:
    """Step-level sigmoid curve of one clip under one protocol (fresh per-item verbalizer)."""
    class_feats_fn = make_class_feats_fn(
        text_encode_fn,
        class_names,
        item_verbalizer(dataset_abbr(constants.DOTA_DATASET), video_id),
    )
    if protocol.window is None:
        return sliding_window_scores(model, feats, class_feats_fn)[0].numpy()
    parts = [
        (start, sliding_window_scores(
            model, feats[start : start + protocol.window], class_feats_fn
        )[0].numpy())
        for start in window_starts(len(feats), protocol.window, constants.V2_E1_HOP)
    ]
    return overlap_average(parts, len(feats))


def load_finished_model(run: Run, device: torch.device) -> tuple[KATVAD, int, TextEncodeFn]:
    """Model in eval mode, after the J10 gate on its stored ``global_step``."""
    payload = torch.load(run.ckpt, map_location="cpu", weights_only=False)  # nosec B614 - own ckpt
    step = int(payload.get("global_step", -1)) if isinstance(payload, dict) else -1
    del payload
    check_finished(step, run.ckpt.parent / METRICS_FILENAME, run.name)
    cfg = load_config(None, DOTA_OVERRIDES)
    model = load_model_for_scoring(cfg, device, run.ckpt, None, TEXT_ENCODER_CLIP, DOTA_OVERRIDES)
    text_encode_fn = make_text_encoder(model, TEXT_ENCODER_CLIP, device, dim=cfg.model.hidden_dim)
    return model, step, text_encode_fn


def score_run(
    run: Run,
    clips: dict[str, Clip],
    data_dir: Path,
    device: torch.device,
    arms: tuple[str, ...],
    s8_dir: Path | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    """Native-frame scores per arm for one checkpoint and its ``global_step``.

    With ``s8_dir`` (and arm A requested) the J9' gate runs before returning: A's
    step-level curve from ``s1[::8]`` must equal the curve on the s8 cache.
    """
    model, step, text_encode_fn = load_finished_model(run, device)
    class_names = load_class_names(data_dir)
    native: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in arms}
    harness_a: dict[str, np.ndarray] = {}
    reference_a: dict[str, np.ndarray] = {}
    protocol_a = PROTOCOLS[KEEP_A]
    for video_id, clip in sorted(clips.items()):
        for arm in arms:
            protocol = PROTOCOLS[arm]
            feats = torch.from_numpy(np.ascontiguousarray(clip.s1[:: protocol.stride]))
            steps = score_steps(model, feats, text_encode_fn, class_names, video_id, protocol)
            if arm == KEEP_A:
                harness_a[video_id] = steps
            native[arm][video_id] = to_native(steps, protocol.stride, len(clip.s1))
        if s8_dir is not None and KEEP_A in arms:
            s8 = torch.from_numpy(np.load(s8_dir / f"{video_id}.npy").astype(np.float32))
            reference_a[video_id] = score_steps(
                model, s8, text_encode_fn, class_names, video_id, protocol_a
            )
    LOGGER.info("%s: scored %d clips x arms %s", run.name, len(clips), ",".join(arms))
    if s8_dir is not None and KEEP_A in arms:
        check_regression(run.name, harness_a, reference_a)
    return native, step


def check_regression(
    name: str, harness: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> None:
    """J9': stop before B/C if the s1 input path differs from the evaluator's s8 path."""
    bad = regression_mismatches(harness, reference, constants.V2_E1_REGRESSION_ATOL)
    if bad:
        raise RuntimeError(
            f"J9' regression FAILED for {name}: {len(bad)} DoTA-dev clips differ "
            f"between s1[::8] and the s8 cache: {bad[:5]}"
        )
    LOGGER.info("J9' regression PASSED for %s (%d clips)", name, len(harness))


# ---------------------------------------------------------------------------
# read-out
# ---------------------------------------------------------------------------
def summarize(
    per_run: dict[str, dict[str, dict[str, np.ndarray]]], clips: dict[str, Clip], seed: int
) -> dict[str, Any]:
    """Seed-averaged macro per arm, paired Δ vs A, J7 decision, printed extras (J8)."""
    groups = {v: c.group for v, c in clips.items()}
    labels = [clips[v].labels for v in sorted(clips)]
    averaged = {
        arm: seed_average([per_run[r][arm] for r in sorted(per_run)])
        for arm in constants.V2_E1_ARMS
    }
    aucs = {arm: clip_aucs(averaged[arm], clips) for arm in constants.V2_E1_ARMS}

    def ci(values: dict[str, float]) -> dict[str, float] | None:
        return cluster_bootstrap_ci(
            values, groups, constants.V2_E1_BOOTSTRAP, seed, constants.V2_E1_CI
        )

    deltas = {
        arm: ci({v: aucs[arm][v] - aucs["A"][v] for v in aucs["A"]}) for arm in ("B", "C")
    }
    by_bin = {
        arm: {
            label: ci({v: a for v, a in aucs[arm].items() if share_bin(clips[v].share) == label})
            for label in constants.V2_SHARE_BIN_LABELS
        }
        for arm in constants.V2_E1_ARMS
    }
    ruler = {v: np.arange(len(c.labels)) / len(c.labels) for v, c in clips.items()}
    position = ci(clip_aucs(ruler, clips))
    per_seed = {
        r: {
            arm: macro_video_auc([per_run[r][arm][v] for v in sorted(clips)], labels)[0]
            for arm in constants.V2_E1_ARMS
        }
        for r in sorted(per_run)
    }
    micro = {
        arm: pooled_metrics(
            [averaged[arm][v] for v in sorted(clips)], labels, SCORE_NORM_MINMAX
        )["auc"]
        for arm in constants.V2_E1_ARMS
    }
    return {
        "macro": {arm: ci(aucs[arm]) for arm in constants.V2_E1_ARMS},
        "delta_vs_A": deltas,
        "decision": decide(deltas),
        "printed": {
            "position_ruler": position,
            "macro_by_share_bin": by_bin,
            "per_seed_macro": per_seed,
            "micro_minmax": micro,
        },
    }


def _fmt(ci: dict[str, float] | None) -> str:
    if ci is None:
        return "—"
    return f"{ci['mean']:.4f} [{ci['low']:+.4f}, {ci['high']:+.4f}]"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_markdown(readout: dict[str, Any]) -> str:
    """Human read-out; the JSON is the record."""
    s = readout["summary"]
    arms = list(constants.V2_E1_ARMS)
    rule = "|---" * (len(arms) + 1) + "|"
    decision = s["decision"]
    lines = [
        "# v2 E1: rate-matched DoTA-dev evaluation",
        "",
        f"Addendum §8-§9 (J1-J8, J9', J10). Runs: {', '.join(readout['runs'])}. "
        f"Clips: {readout['clips']} DoTA-dev; "
        f"s1 length != annotation: {readout['off_length']}.",
        "J10 finished checkpoints (global_step): "
        + ", ".join(f"{k} {v}" for k, v in readout["global_steps"].items())
        + ". J9' regression (s1[::8] == s8 cache): PASSED on every run.",
        "",
        _row(["Arm", "Macro (seed-avg) [95 % cluster CI]", "Δ vs A", "Micro (min-max)"]),
        "|---|---|---|---|",
    ]
    for arm in arms:
        delta = _fmt(s["delta_vs_A"][arm]) if arm != KEEP_A else "—"
        micro = f"{s['printed']['micro_minmax'][arm]:.4f}"
        lines.append(_row([arm, _fmt(s["macro"][arm]), delta, micro]))
    eligible = decision["eligible"] or "none"
    lines += [
        "",
        f"**Position ruler `t/N` (no pixels): {_fmt(s['printed']['position_ruler'])}**",
        "",
        f"**Decision (J7): eligible {eligible} → adopt {decision['adopted']}.**",
        "",
        "Per-seed macro (printed, not decided on):",
        "",
        _row(["Run", *arms]),
        rule,
    ]
    for run_name, row in s["printed"]["per_seed_macro"].items():
        lines.append(_row([run_name, *(f"{row[a]:.4f}" for a in arms)]))
    lines += ["", "Macro by share bin (printed):", "", _row(["Bin", *arms]), rule]
    for label in constants.V2_SHARE_BIN_LABELS:
        by_bin = s["printed"]["macro_by_share_bin"]
        lines.append(_row([label, *(_fmt(by_bin[a][label]) for a in arms)]))
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    runs = [Run(name, Path(ckpt)) for name, ckpt in args.run]
    device = resolve_device(args.device)
    clips, off_length = load_dev_clips(args.s1_dir, args.metadata, args.split_file)
    per_run: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    steps: dict[str, int] = {}
    for r in runs:
        native, steps[r.name] = score_run(
            r, clips, args.data_dir, device, (KEEP_A,), s8_dir=args.s8_dir
        )
        per_run[r.name] = native
    for r in runs:
        native, _ = score_run(r, clips, args.data_dir, device, ("B", "C"))
        per_run[r.name].update(native)
    readout: dict[str, Any] = {
        "addendum": "core/docs/v2/PREREG_ADDENDUM.md §8-§9",
        "runs": [r.name for r in runs],
        "checkpoints": {r.name: str(r.ckpt) for r in runs},
        "global_steps": steps,
        "clips": len(clips),
        "off_length": off_length,
        "summary": summarize(per_run, clips, args.seed),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(readout))
    LOGGER.info("E1 decision: %s -> %s", readout["summary"]["decision"], args.out_dir)
    return readout


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--run", nargs=2, action="append", required=True, metavar=("NAME", "CKPT"),
        help="finished checkpoint to re-score, metrics.jsonl beside it (repeat per seed)",
    )
    parser.add_argument("--s1-dir", type=Path, required=True, help="cache/clip/DoTA_s1_ncc")
    parser.add_argument(
        "--s8-dir", type=Path, required=True, help="cache/clip/DoTA_s8_ncc (J9' reference)"
    )
    parser.add_argument("--metadata", type=Path, required=True, help="metadata_val.json")
    parser.add_argument("--split-file", type=Path, required=True, help="val_split.txt")
    parser.add_argument("--data-dir", type=Path, required=True, help="DoTA labels dir (defs.json)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=constants.SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
