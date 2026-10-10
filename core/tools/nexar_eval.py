"""Nexar read-out of one arm's checkpoints: whole videos and shifted crops (addendum §21).

Protocol N-B (E1's protocol B on Nexar): rows ``s1[::3]`` of the N2 caches, the arm's input
baked **per scored item** with the stats of its ``build_v2_inputs fit-ids`` cache (each item is
its own CRN reference, as DoTA clips are), whole item, step ``t`` at native frame ``3t``, scores
interpolated to every native frame and averaged over seeds. Labels = D-N1's span on native frames.

Two reads, both printed beside what position alone scores (G6 mirror):

* **whole** -- every video of the split. Macro over positives (two-class), micro over all videos
  (negatives included), clip-level AUC (max score, positive vs negative video), and the
  "middle" ruler ``-|t/T - 0.5|``. On Nexar the event sits at ~0.5 of every video, so whole-video
  macro is a position test (census §1.1) -- printed, never the endpoint.
* **crops** (D-N2, the endpoint) -- from each positive, an ``NEXAR_CROP_S`` crop with the event
  centre (mid-span) at each of ``NEXAR_CROP_PLACEMENTS``; CRN re-referenced per crop. Macro per
  placement and the mean over the videos that fit all placements; the middle ruler per placement
  shows the position prior losing everywhere but 0.5.

``nexar_test`` is sealed: refused without ``--final``. Checkpoints must be finished (J10) and of
one arm; the stats cache must name that arm (K4).

CLI::

    python -m core.tools.nexar_eval --run s2024 runs/window/A3/s2024/checkpoint_last.pt ... \\
        --census .../nexar_n0/census.json --clip-s1-dir cache/clip/Nexar_s1_ncc \\
        [--video-s1-dir cache/video/<enc>/Nexar_s1_squash --stats-dir cache/v2/A3_R2_S/Nexar_s8] \\
        --data-dir data/Nexar_window --split nexar_val \\
        --out-dir outputs/v2/REPORTS/nexar_val/window_A3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core import constants
from core.data.v2_inputs import V2Stats, bake_rows, load_stats, read_input_manifest
from core.data.v2_splits import load_split
from core.device import resolve_device
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_build import native_labels, span_seconds
from core.tools.rate_matched_eval import (
    Protocol,
    Run,
    load_finished_model,
    score_steps,
    seed_average,
    to_native,
)
from core.train import load_class_names

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "nexar_readout.json"
READOUT_MD = "nexar_readout.md"
ITEM_AUCS_JSON = "item_aucs.json"
CROP_SEP = "@"


def crop_bounds(
    row: dict[str, Any], frames: int, placement: float, crop_s: float = constants.NEXAR_CROP_S
) -> tuple[int, int] | None:
    """Native ``[start, end)`` of the crop putting the span centre at ``placement``; ``None``
    when it does not fit inside the video."""
    fps = frames / float(row["duration_s"])
    start_s, end_s = span_seconds(row)
    length = round(crop_s * fps)
    start = round((start_s + end_s) / 2 * fps - placement * length)
    if start < 0 or start + length > frames:
        return None
    return start, start + length


def middle_ruler(length: int) -> np.ndarray:
    """``-|t/T - 0.5|``: the no-pixel score that wins when events sit mid-item."""
    ruler: np.ndarray = -np.abs(np.arange(length) / length - 0.5)
    return ruler


def item_rows(
    x1: np.ndarray, u1: np.ndarray | None, stats: V2Stats | None, start: int, end: int,
    stride: int = constants.NEXAR_EVAL_STRIDE,
) -> np.ndarray:
    """The arm's input for native frames ``[start, end)`` at ``stride``, CRN over this item."""
    x = x1[start:end:stride]
    if stats is None:
        return np.ascontiguousarray(x, dtype=np.float32)
    u = None if u1 is None else u1[start:end:stride]
    return bake_rows(x, u, stats)


def items_of(
    ids: list[str], rows: dict[str, dict[str, Any]], frames: dict[str, int]
) -> tuple[dict[str, tuple[str, int, int]], dict[str, list[str]]]:
    """``{item: (video, start, end)}`` for whole videos and every fitting crop; skipped crops."""
    items: dict[str, tuple[str, int, int]] = {v: (v, 0, frames[v]) for v in ids}
    skipped: dict[str, list[str]] = {}
    for v in ids:
        if rows[v]["label"] != 1:
            continue
        for p in constants.NEXAR_CROP_PLACEMENTS:
            bounds = crop_bounds(rows[v], frames[v], p)
            if bounds is None:
                skipped.setdefault(str(p), []).append(v)
            else:
                items[f"{v}{CROP_SEP}{p}"] = (v, *bounds)
    return items, skipped


def score_run(
    run: Run, items: dict[str, tuple[str, int, int]], args: argparse.Namespace,
    stats: V2Stats | None, device: torch.device,
) -> tuple[dict[str, np.ndarray], int, dict[str, str]]:
    model, step, text_encode_fn, cfg = load_finished_model(run, device)
    expected = (constants.V2_OFF, constants.V2_OFF) if stats is None else (stats.crn, stats.motion)
    if (cfg.v2.crn, cfg.v2.motion) != expected:
        raise ValueError(f"{run.name} is crn={cfg.v2.crn} motion={cfg.v2.motion}, "
                         f"the stats are {expected} (K4)")
    class_names = load_class_names(args.data_dir)
    protocol = Protocol(constants.NEXAR_EVAL_STRIDE, None)
    by_video: dict[str, list[str]] = {}
    for item, (video, _, _) in items.items():
        by_video.setdefault(video, []).append(item)
    native: dict[str, np.ndarray] = {}
    for video, names in sorted(by_video.items()):
        x1 = np.load(args.clip_s1_dir / f"{video}.npy")
        u1 = None if stats is None or not stats.has_motion else np.load(
            args.video_s1_dir / f"{video}.npy")
        for item in names:
            _, start, end = items[item]
            feats = torch.from_numpy(item_rows(x1, u1, stats, start, end))
            steps = score_steps(model, feats, text_encode_fn, class_names, video, protocol)
            native[item] = to_native(steps, constants.NEXAR_EVAL_STRIDE, end - start)
    LOGGER.info("%s (step %d): scored %d items", run.name, step, len(native))
    return native, step, {"crn": cfg.v2.crn, "motion": cfg.v2.motion}


def two_class_aucs(
    scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], names: list[str]
) -> dict[str, float]:
    return {
        n: frame_auc(scores[n], labels[n])
        for n in names
        if 0 < int(labels[n].sum()) < len(labels[n])
    }


def summarize(
    scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], ids: list[str],
    rows: dict[str, dict[str, Any]], items: dict[str, tuple[str, int, int]], seed: int,
) -> dict[str, Any]:
    """Every number of the read-out from seed-averaged native scores."""
    groups = {n: items[n][0] for n in items}

    def ci(values: dict[str, float]) -> dict[str, float] | None:
        return cluster_bootstrap_ci(
            values, groups, constants.V2_E1_BOOTSTRAP, seed, constants.V2_E1_CI
        )

    ruler = {n: middle_ruler(len(labels[n])) for n in items}
    whole_aucs = two_class_aucs(scores, labels, ids)
    video_label = np.array([rows[v]["label"] for v in ids])
    out: dict[str, Any] = {
        "whole": {
            "macro": ci(whole_aucs),
            "middle_ruler_macro": ci(two_class_aucs(ruler, labels, ids)),
            "micro": frame_auc(np.concatenate([scores[v] for v in ids]),
                               np.concatenate([labels[v] for v in ids])),
            "clip_oracle_micro": frame_auc(
                np.concatenate([np.full(len(labels[v]), rows[v]["label"], float) for v in ids]),
                np.concatenate([labels[v] for v in ids]),
            ) if 0 < video_label.sum() < len(ids) else None,
            "clip_auc_max": frame_auc(np.array([scores[v].max() for v in ids]), video_label)
            if 0 < video_label.sum() < len(ids) else None,
        },
        "crops": {},
    }
    per_video: dict[str, list[float]] = {}
    for p in constants.NEXAR_CROP_PLACEMENTS:
        names = [n for n in items if n.endswith(f"{CROP_SEP}{p}")]
        aucs = two_class_aucs(scores, labels, names)
        out["crops"][str(p)] = {
            "items": len(names),
            "macro": ci(aucs),
            "middle_ruler_macro": ci(two_class_aucs(ruler, labels, names)),
        }
        for n, a in aucs.items():
            per_video.setdefault(items[n][0], []).append(a)
    complete = {v: float(np.mean(a)) for v, a in per_video.items()
                if len(a) == len(constants.NEXAR_CROP_PLACEMENTS)}
    out["crops"]["mean_over_placements"] = {
        "videos": len(complete),
        "macro": cluster_bootstrap_ci(complete, {v: v for v in complete},
                                      constants.V2_E1_BOOTSTRAP, seed, constants.V2_E1_CI),
    }
    out["item_aucs"] = two_class_aucs(scores, labels, sorted(items))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    final = args.split == constants.V2_SPLIT_NEXAR_TEST
    if final and not args.final:
        raise SystemExit("nexar_test is sealed: pass --final only for the §21 final read")
    census = json.loads(args.census.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = census["videos"]
    ids = load_split(args.split, args.split_dir, final=final)
    frames = {v: int(np.load(args.clip_s1_dir / f"{v}.npy", mmap_mode="r").shape[0]) for v in ids}
    stats = load_stats(args.stats_dir) if args.stats_dir is not None else None
    if stats is not None and stats.has_motion and args.video_s1_dir is None:
        raise SystemExit("a motion arm needs --video-s1-dir")
    items, skipped = items_of(ids, rows, frames)
    labels: dict[str, np.ndarray] = {}
    for name, (video, start, end) in items.items():
        labels[name] = native_labels(rows[video], frames[video])[start:end]

    runs = [Run(name, Path(ckpt)) for name, ckpt in args.run]
    device = resolve_device(args.device)
    per_run: dict[str, dict[str, np.ndarray]] = {}
    steps: dict[str, int] = {}
    arms: dict[str, dict[str, str]] = {}
    for r in runs:
        per_run[r.name], steps[r.name], arms[r.name] = score_run(r, items, args, stats, device)
    averaged = seed_average([per_run[n] for n in sorted(per_run)])
    summary = summarize(averaged, labels, ids, rows, items, args.seed)
    readout: dict[str, Any] = {
        "protocol": "N-B: s1[::3] per item, CRN per item, native frames, seed-averaged",
        "split": args.split,
        "videos": len(ids),
        "positives": sum(rows[v]["label"] for v in ids),
        "crop_s": constants.NEXAR_CROP_S,
        "crops_skipped": {p: len(v) for p, v in skipped.items()},
        "data_dir": str(args.data_dir),
        "stats_dir": None if args.stats_dir is None else str(args.stats_dir),
        "stats_manifest": None if args.stats_dir is None else read_input_manifest(args.stats_dir),
        "arm": next(iter(arms.values())),
        "checkpoints": {r.name: str(r.ckpt) for r in runs},
        "global_steps": steps,
        "per_run_whole_macro": {
            n: float(np.mean(list(two_class_aucs(s, labels, ids).values())))
            for n, s in per_run.items()
        },
        **{k: v for k, v in summary.items() if k != "item_aucs"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / ITEM_AUCS_JSON, summary["item_aucs"])
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(readout))
    LOGGER.info("whole macro %s; crop mean %s -> %s", readout["whole"]["macro"],
                readout["crops"]["mean_over_placements"]["macro"], args.out_dir)
    return readout


def _fmt(c: dict[str, float] | None) -> str:
    if not c:
        return "—"
    return f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def render_markdown(r: dict[str, Any]) -> str:
    w = r["whole"]
    lines = [
        f"# Nexar read-out — {r['split']} ({r['videos']} videos, {r['positives']} positive)",
        "",
        f"Arm `{r['arm']}`, data `{r['data_dir']}`, runs {sorted(r['checkpoints'])} "
        f"(steps {r['global_steps']}). {r['protocol']}.",
        "",
        "## Crops (endpoint, D-N2)",
        "",
        "| event centre at | items | macro | middle ruler |",
        "|---|---:|---|---|",
    ]
    for p in constants.NEXAR_CROP_PLACEMENTS:
        c = r["crops"][str(p)]
        lines.append(
            f"| {p} | {c['items']} | {_fmt(c['macro'])} | {_fmt(c['middle_ruler_macro'])} |"
        )
    m = r["crops"]["mean_over_placements"]
    lines += [
        f"| **mean (videos with all {len(constants.NEXAR_CROP_PLACEMENTS)})** | {m['videos']} | "
        f"**{_fmt(m['macro'])}** | |",
        "",
        f"Crops skipped (did not fit): {r['crops_skipped']}",
        "",
        "## Whole videos (printed — a position test on Nexar)",
        "",
        f"* macro (positives): {_fmt(w['macro'])} · middle ruler {_fmt(w['middle_ruler_macro'])}",
        f"* micro (all videos): {_num(w['micro'])} · clip oracle {_num(w['clip_oracle_micro'])}",
        f"* clip-level AUC (max score): {_num(w['clip_auc_max'])}",
        f"* per-run whole macro: { {k: round(v, 4) for k, v in r['per_run_whole_macro'].items()} }",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--run", nargs=2, action="append", required=True,
                        metavar=("NAME", "CKPT"), help="one finished checkpoint (repeat per seed)")
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--clip-s1-dir", type=Path, required=True)
    parser.add_argument("--video-s1-dir", type=Path, default=None)
    parser.add_argument("--stats-dir", type=Path, default=None,
                        help="the arm's fit-ids cache (A1-A3); omit for A0")
    parser.add_argument("--data-dir", type=Path, required=True, help="training dir (defs.json)")
    parser.add_argument("--split", default=constants.V2_SPLIT_NEXAR_VAL,
                        choices=(constants.V2_SPLIT_NEXAR_VAL, constants.V2_SPLIT_NEXAR_TEST))
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--final", action="store_true", help="unseal nexar_test (§21, once)")
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap seed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
