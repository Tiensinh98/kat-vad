"""Per-video zero-shot score curves over the ground truth, plus the onset-aligned mean curve.

Reads the per-frame score files the zero-shot reads already wrote (nothing is re-scored):
``clip_scores.npz`` for DoTA-CAP-dev (protocol B) and DADA-2000 whole, a ``core.evaluate
--save-scores`` run dir (``<dir>/scores/*.npz``) for TAD and DAD -- the same sources
``zs_arm_table`` reads, through the same loader. Every model must have scored the same videos
with identical frame labels, or the tool raises.

Per set it writes ``grid_<set>.png``: ``--n`` videos spread evenly over the rank model's
per-video frame AUC (worst -> best, so the grid is not a best-case pick; on TAD / DAD the last
cell is one all-normal video, where false alarms show). Curves are drawn as the tables pool
them: DoTA / DADA after per-video min-max (raw drawn faint beneath), TAD / DAD raw (min-max
dashed, shape only). ``onset_aligned.png`` = each model's mean score in seconds from the GT
onset over every two-class video (+-1 SEM); bins with too few videos are hidden. The numbers
behind both go to ``zs_curves_<name>.json``. ``--all`` replaces the grid with one PNG per
video, ``<out-dir>/videos/<set>/<rank>_<video>.png`` (two-class worst -> best by the rank
model's AUC, then the all-normal ones). Illustration only (n = 1 seed): decides nothing.

CLI::

    python -m core.tools.zs_curves --rank-by K5 --name k5_lagovad_s2024 --out-dir figs \\
        --read dota K5 outputs/v_final/kim/eval/dota/K5/clip_scores.npz \\
        --read dota LaGoVAD outputs/v_final/zs_lagovad/eval/dota/LaGoVAD/B/clip_scores.npz \\
        --read tad K5 outputs/v_final/kim/eval/tad/K5/s2024 \\
        --read tad LaGoVAD outputs/v_final/zs_lagovad/eval/tad/LaGoVAD/released
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: render straight to file
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from core import constants
from core.metrics import frame_auc, normalize_scores
from core.tools import zs_compare_table
from core.tools.kill_switch_probe import write_json_atomic
from core.tools.zs_arm_table import check_same_labels

LOGGER = logging.getLogger(__name__)

SET_ORDER = zs_compare_table.SET_ORDER
SET_NAME = zs_compare_table.SET_NAME
MINMAX_SETS = zs_compare_table.TABLE_MINMAX_SETS
LABEL = "illustration (n = 1 seed); decides nothing"

Curves = dict[str, np.ndarray]
Loaded = dict[str, tuple[Curves, Curves]]  # model -> (scores, labels)


def minmax(scores: np.ndarray) -> np.ndarray:
    """One video's curve mapped to [0, 1] (the tables' per-video min-max)."""
    return normalize_scores(np.asarray(scores, dtype=np.float64), constants.SCORE_NORM_MINMAX)


def is_two_class(gt: np.ndarray) -> bool:
    """The video holds both normal and abnormal rows (its own frame AUC is defined)."""
    positive = int((gt > 0).sum())
    return 0 < positive < len(gt)


def check_lengths(loaded: Loaded, set_name: str) -> None:
    """Every model's curve has one score per label row."""
    for model, (scores, labels) in loaded.items():
        bad = [v for v in labels if len(scores[v]) != len(labels[v])]
        if bad:
            raise ValueError(f"{set_name}/{model}: score and label lengths differ on {bad[:5]}")


def video_aucs(scores: Curves, labels: Curves) -> dict[str, float]:
    """Frame AUC of each two-class video."""
    return {v: frame_auc(scores[v], labels[v]) for v in sorted(labels) if is_two_class(labels[v])}


def pick_videos(aucs: dict[str, float], labels: Curves, n: int) -> list[str]:
    """``n`` videos spread evenly over the AUC rank; the last swapped for an all-normal one."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    ranked = sorted(aucs, key=lambda v: (aucs[v], v))
    normals = sorted(v for v, gt in labels.items() if not (gt > 0).any())
    slots = n - 1 if normals and n > 1 else n
    picks: list[str] = []
    if ranked and slots:
        idx = np.linspace(0, len(ranked) - 1, min(slots, len(ranked))).round().astype(int)
        picks = [ranked[i] for i in idx]
    if normals and len(picks) < n:
        picks.append(normals[0])
    return picks


def rank_all(aucs: dict[str, float], labels: Curves) -> list[str]:
    """Every video: two-class ones worst -> best AUC, then the all-normal ones by id."""
    ranked = sorted(aucs, key=lambda v: (aucs[v], v))
    return ranked + sorted(v for v in labels if v not in aucs)


def onset_curve(scores: Curves, labels: Curves, step_s: float, use_minmax: bool,
                window_s: float, bin_rows: int, min_videos: int) -> dict[str, Any]:
    """Mean score per time bin from the GT onset, over the two-class videos.

    Each video contributes its own mean per bin first, so long videos do not outweigh short
    ones. Bins averaged over fewer than ``min_videos`` videos are ``None``.
    """
    bin_s = bin_rows * step_s
    n_bins = math.ceil(window_s / bin_s)
    edges = np.arange(-n_bins, n_bins + 1) * bin_s
    per_video = []
    for v in sorted(labels):
        gt = labels[v]
        if not is_two_class(gt):
            continue
        curve = minmax(scores[v]) if use_minmax else np.asarray(scores[v], dtype=np.float64)
        rel = (np.arange(len(curve)) - int(np.argmax(gt > 0))) * step_s
        which = np.digitize(rel, edges) - 1
        row = np.full(len(edges) - 1, np.nan)
        for b in np.unique(which[(which >= 0) & (which < len(row))]):
            row[b] = curve[which == b].mean()
        per_video.append(row)
    centers = (edges[:-1] + edges[1:]) / 2
    if not per_video:
        empty: list[float | None] = [None] * len(centers)
        return {"centers_s": centers.tolist(), "mean": empty, "sem": empty,
                "n_per_bin": [0] * len(centers), "videos": 0, "bin_s": bin_s}
    stack = np.vstack(per_video)
    count = np.sum(~np.isnan(stack), axis=0)
    mean: list[float | None] = []
    sem: list[float | None] = []
    for b in range(stack.shape[1]):
        if count[b] < min_videos:
            mean.append(None)
            sem.append(None)
            continue
        col = stack[~np.isnan(stack[:, b]), b]
        mean.append(float(col.mean()))
        sem.append(float(col.std() / math.sqrt(len(col))))
    return {"centers_s": centers.tolist(), "mean": mean, "sem": sem,
            "n_per_bin": count.astype(int).tolist(), "videos": len(per_video), "bin_s": bin_s}


def _nan(values: list[float | None]) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in values], dtype=np.float64)


def draw_video(ax: Axes, video: str, loaded: Loaded, set_name: str,
               aucs: dict[str, dict[str, float]]) -> None:
    """One video: GT shading and every model's curve, per-video AUC in the legend."""
    step_s = constants.ZS_CURVES_STEP_SECONDS[set_name]
    use_minmax = set_name in MINMAX_SETS
    gt = next(iter(loaded.values()))[1][video]
    t = np.arange(len(gt)) * step_s
    ax.fill_between(t, 0.0, 1.0, where=(gt > 0).tolist(), step="mid", color="tab:red",
                    alpha=constants.ZS_CURVES_GT_ALPHA, label="ground truth")
    models = loaded.items()
    for color, (model, (scores, _)) in zip(constants.ZS_CURVES_COLORS, models, strict=False):
        raw = np.asarray(scores[video], dtype=np.float64)
        auc = aucs[model].get(video)
        tag = f"{model} (AUC {auc:.3f})" if auc is not None else model
        if use_minmax:
            ax.plot(t, raw, color=color, lw=1.0, alpha=constants.ZS_CURVES_RAW_ALPHA)
            ax.plot(t, minmax(raw), color=color, lw=1.6, label=f"{tag} min-max")
        else:
            ax.plot(t, raw, color=color, lw=1.6, label=f"{tag} raw")
            ax.plot(t, minmax(raw), color=color, lw=0.9, ls="--", alpha=0.6)
    title = video if is_two_class(gt) else f"{video} (all-normal)"
    ax.set_title(title, fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0.0, max(float(t[-1]), step_s))
    ax.set_xlabel("time (s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=6)


def render_grid(path: Path, set_name: str, loaded: Loaded, picks: list[str],
                aucs: dict[str, dict[str, float]], rank_by: str, seed_label: str) -> None:
    cols = min(constants.ZS_CURVES_GRID_COLS, len(picks))
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.0 * rows), squeeze=False)
    for ax, video in zip(axes.flat, picks, strict=False):
        draw_video(ax, video, loaded, set_name, aucs)
    for ax in list(axes.flat)[len(picks):]:
        ax.set_visible(False)
    faint = ("faint = raw" if set_name in MINMAX_SETS else "dashed = min-max, shape only")
    n_videos = len(next(iter(loaded.values()))[1])
    fig.suptitle(f"{SET_NAME[set_name]} zero-shot, {n_videos} videos: picks spread worst -> best "
                 f"{rank_by} per-video AUC; {faint} ({seed_label})", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=constants.ZS_CURVES_DPI)
    plt.close(fig)


def render_videos(out_dir: Path, set_name: str, loaded: Loaded, order: list[str],
                  aucs: dict[str, dict[str, float]], rank_by: str, seed_label: str) -> int:
    """One PNG per video, ``<rank>_<video>.png`` so the folder sorts worst -> best."""
    out_dir.mkdir(parents=True, exist_ok=True)
    width = len(str(len(order)))
    for rank, video in enumerate(order, start=1):
        fig, ax = plt.subplots(figsize=constants.ZS_CURVES_VIDEO_FIGSIZE)
        draw_video(ax, video, loaded, set_name, aucs)
        fig.suptitle(f"{SET_NAME[set_name]} zero-shot, {rank}/{len(order)} by {rank_by} "
                     f"per-video AUC ({seed_label})", fontsize=9)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        fig.savefig(out_dir / f"{rank:0{width}d}_{video}.png", dpi=constants.ZS_CURVES_DPI)
        plt.close(fig)
    return len(order)


def render_onset(path: Path, onset: dict[str, dict[str, dict[str, Any]]], seed_label: str) -> None:
    sets = list(onset)
    fig, axes = plt.subplots(1, len(sets), figsize=(4.0 * len(sets), 3.4), squeeze=False)
    for ax, set_name in zip(axes.flat, sets, strict=True):
        ax.axvspan(0.0, constants.ZS_CURVES_ONSET_WINDOW_S, color="tab:red", alpha=0.12)
        ax.axvline(0.0, color="tab:red", lw=1.0)
        videos = 0
        for color, (model, c) in zip(constants.ZS_CURVES_COLORS, onset[set_name].items(),
                                     strict=False):
            x = np.asarray(c["centers_s"])
            m, s = _nan(c["mean"]), _nan(c["sem"])
            ax.plot(x, m, color=color, lw=1.8, label=model)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)
            videos = c["videos"]
        pooling = "min-max" if set_name in MINMAX_SETS else "raw"
        ax.set_title(f"{SET_NAME[set_name]} ({pooling}, n = {videos})", fontsize=10)
        ax.set_xlabel("seconds from GT onset", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=7)
    axes.flat[0].set_ylabel("mean score")
    fig.suptitle(f"Mean score aligned at the GT onset, two-class videos, +-1 SEM ({seed_label})",
                 fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(path, dpi=constants.ZS_CURVES_DPI)
    plt.close(fig)


def load_reads(reads: list[tuple[str, str, Path]]) -> dict[str, Loaded]:
    """``{set: {model: (scores, labels)}}`` in the tables' set order, checked per set."""
    by_set: dict[str, Loaded] = {}
    for set_name, model, source in reads:
        if set_name not in SET_ORDER:
            raise ValueError(f"unknown set {set_name!r}; one of {SET_ORDER}")
        got = zs_compare_table.load_scores(source)
        if got is None:
            raise FileNotFoundError(f"{set_name}/{model}: no per-frame scores at {source}")
        by_set.setdefault(set_name, {})[model] = got
    out = {s: by_set[s] for s in SET_ORDER if s in by_set}
    for set_name, loaded in out.items():
        if len(loaded) > len(constants.ZS_CURVES_COLORS):
            raise ValueError(f"{set_name}: {len(loaded)} models, at most "
                             f"{len(constants.ZS_CURVES_COLORS)} can be drawn")
        check_same_labels(loaded, set_name)
        check_lengths(loaded, set_name)
    return out


def build(data: dict[str, Loaded], rank_by: str, n: int,
          clips: list[str] | None, every_video: bool = False) -> dict[str, Any]:
    """Picks, per-video AUCs and onset curves of every set (pure: nothing is drawn here).

    ``every_video`` picks every video in rank order (``rank_all``) instead of ``n`` of them.
    """
    if every_video and clips:
        raise ValueError("every_video and clips are exclusive")
    out: dict[str, Any] = {}
    for set_name, loaded in data.items():
        if rank_by not in loaded:
            raise ValueError(f"{set_name}: the rank model {rank_by} has no read")
        labels = loaded[rank_by][1]
        aucs = {m: video_aucs(s, lab) for m, (s, lab) in loaded.items()}
        if every_video:
            picks = rank_all(aucs[rank_by], labels)
        elif clips:
            picks = [c for c in clips if c in labels]
        else:
            picks = pick_videos(aucs[rank_by], labels, n)
        onset = {m: onset_curve(s, lab, constants.ZS_CURVES_STEP_SECONDS[set_name],
                                set_name in MINMAX_SETS, constants.ZS_CURVES_ONSET_WINDOW_S,
                                constants.ZS_CURVES_ONSET_BIN_ROWS,
                                constants.ZS_CURVES_MIN_VIDEOS_PER_BIN)
                 for m, (s, lab) in loaded.items()}
        out[set_name] = {
            "videos": len(labels), "two_class": len(aucs[rank_by]), "picks": picks,
            "step_s": constants.ZS_CURVES_STEP_SECONDS[set_name],
            "pooling": "min-max" if set_name in MINMAX_SETS else "raw",
            "median_video_auc": {m: float(np.median(list(a.values()))) if a else None
                                 for m, a in aucs.items()},
            "pick_aucs": {m: {v: aucs[m].get(v) for v in picks} for m in loaded},
            "onset": onset,
        }
        LOGGER.info("%s: %d videos, %d two-class, %d picks %s", set_name, len(labels),
                    len(aucs[rank_by]), len(picks), picks if not every_video else "(all)")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-video zero-shot score curves over GT.")
    parser.add_argument("--read", nargs=3, action="append", required=True,
                        metavar=("SET", "MODEL", "PATH"),
                        help=f"one model's scores on one set; SET in {SET_ORDER}. PATH = "
                             "clip_scores.npz (dota / dada) or the run dir holding scores/ "
                             "(tad / dad)")
    parser.add_argument("--rank-by", required=True, help="model whose per-video AUC ranks picks")
    parser.add_argument("--n", type=int, default=constants.ZS_CURVES_N_PICK)
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--clips", nargs="+", default=None,
                       help="video ids to draw instead of the rank picks (per set, if present)")
    which.add_argument("--all", action="store_true",
                       help="draw EVERY video, one PNG each, in <out-dir>/"
                            f"{constants.ZS_CURVES_VIDEO_SUBDIR}/<set>/ (no grid)")
    parser.add_argument("--seed-label", default=f"s{constants.KIM_SEED}, n = 1")
    parser.add_argument("--name", required=True, help="json = zs_curves_<name>.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = load_reads([(s, m, Path(p)) for s, m, p in args.read])
    table = build(data, args.rank_by, args.n, args.clips, every_video=args.all)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for set_name, t in table.items():
        if not t["picks"]:
            LOGGER.warning("%s: no video to draw", set_name)
            continue
        aucs = {m: {v: a for v, a in t["pick_aucs"][m].items() if a is not None}
                for m in data[set_name]}
        if args.all:
            video_dir = args.out_dir / constants.ZS_CURVES_VIDEO_SUBDIR / set_name
            n = render_videos(video_dir, set_name, data[set_name], t["picks"], aucs,
                              args.rank_by, args.seed_label)
            LOGGER.info("%s: %d per-video figures -> %s", set_name, n, video_dir)
            continue
        render_grid(args.out_dir / f"grid_{set_name}.png", set_name, data[set_name],
                    t["picks"], aucs, args.rank_by, args.seed_label)
    render_onset(args.out_dir / "onset_aligned.png",
                 {s: t["onset"] for s, t in table.items()}, args.seed_label)
    write_json_atomic(args.out_dir / f"{constants.ZS_CURVES_STEM}_{args.name}.json",
                      {"rank_by": args.rank_by, "label": LABEL, "seed_label": args.seed_label,
                       "sets": table})
    return table


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()

__all__ = ["build", "load_reads", "onset_curve", "pick_videos", "rank_all", "run",
           "video_aucs"]
