"""v2 figures from protocol-B score files: per-clip score curves and pooled ROC / PR.

Reads the ``clip_scores.npz`` files the v2 harness already writes (``protocol_b_eval``:
seed-averaged native-frame scores, keys ``score/<clip>`` and ``label/<clip>``) and, optionally,
``position_prior.npz`` (``p_T2``) as a no-pixel reference. Nothing is re-scored.

* ``curves`` -- one PNG per clip plus a grid: every arm's curve over the ground-truth span, the
  per-clip AUC of each in its legend. Clips are given with ``--clips`` or picked evenly over the
  rank of ``--sort-by``'s per-clip AUC (worst -> best), so the figure is not a best-case pick.
* ``roc`` -- pooled ROC and PR under per-clip min-max (lesson 12: DoTA is ~all-abnormal, so the
  between-clip scale is noise). Legend numbers come from ``core.metrics.pooled_metrics``, the
  same code the read-outs use; ``roc_metrics.json`` records them.

Label every DoTA-CAP figure "DoTA-CAP (n/1397)"; never beside full DoTA or 62.60 (D14). An
A3-ign eval figure is a second look at an opened set (addendum §23).

CLI::

    python -m core.tools.v2_plots roc --out-dir figs/cap_eval --title "DoTA-CAP (n/1397) eval" \\
        --arm A0 outputs/v2_final/REPORTS/pb/A0/dota_cap_eval/clip_scores.npz \\
        --arm A3-ign outputs/v2_e4_0/second_look/REPORTS/pb/A3ign/dota_cap_eval/clip_scores.npz \\
        --prior outputs/v2_final/REPORTS/position_prior/dota_cap_eval/position_prior.npz
    python -m core.tools.v2_plots curves --out-dir figs/curves --n 6 --arm ... --prior ...
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: render straight to file
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import precision_recall_curve, roc_curve

from core import constants
from core.metrics import frame_auc, normalize_scores, pooled_metrics
from core.tools.kill_switch_probe import write_json_atomic
from core.tools.protocol_b_eval import read_clip_scores

LOGGER = logging.getLogger(__name__)

# Categorical slots in fixed order (dataviz reference palette, validated on #fcfcfb: all
# checks pass; slots 3-4 sit below 3:1 contrast, so every series is named in the legend).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
PRIOR_COLOR = "#52514e"  # text-secondary gray: a reference, not a series hue
GT_COLOR = "#d9d8d3"  # neutral band behind the curves
GRID_COLOR = "#e6e5e1"
SURFACE = "#fcfcfb"
LINE_WIDTH = 2.0
DPI = 200
CURVE_SIZE = (8.0, 2.6)
ROC_SIZE = (10.0, 4.4)
GRID_COLUMNS = 2
LEGEND_BAND = 0.12  # figure fraction kept free under the axes for the shared legend
PRIOR_NAME = "p_T2 (position only)"
ROC_JSON = "roc_metrics.json"
ROC_PNG = "roc_pr.png"
GRID_PNG = "curves_grid.png"
NORMS = (constants.SCORE_NORM_MINMAX, constants.SCORE_NORM_NONE)


@dataclass(frozen=True)
class Series:
    """One named score file: native-frame curves and labels per clip."""

    name: str
    scores: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]


def load_series(arms: list[list[str]], prior: Path | None) -> tuple[list[Series], Series | None]:
    """Every ``--arm NAME NPZ`` (+ the prior); all must cover the same clips with equal labels."""
    if not arms:
        raise ValueError("give at least one --arm")
    if len(arms) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} arms per figure (fixed palette order)")
    series = [Series(name, *read_clip_scores(Path(path))) for name, path in arms]
    ref = Series(PRIOR_NAME, *read_clip_scores(prior)) if prior else None
    first = series[0]
    for other in [*series[1:], *([ref] if ref else [])]:
        if set(other.labels) != set(first.labels):
            raise ValueError(f"{other.name} and {first.name} cover different clips")
        for clip, lab in first.labels.items():
            if not np.array_equal(other.labels[clip], lab) or len(other.scores[clip]) != len(lab):
                raise ValueError(f"{other.name} and {first.name} disagree on clip {clip}")
    return series, ref


def two_class(labels: dict[str, np.ndarray]) -> list[str]:
    return sorted(v for v, lab in labels.items() if 0 < int(lab.sum()) < len(lab))


def pick_clips(target: Series, n: int) -> list[str]:
    """``n`` clips evenly spaced over the rank of ``target``'s per-clip AUC, worst first."""
    ranked = sorted(two_class(target.labels),
                    key=lambda v: (frame_auc(target.scores[v], target.labels[v]), v))
    if n >= len(ranked):
        return ranked
    picks = np.linspace(0, len(ranked) - 1, n).round().astype(int)
    return [ranked[i] for i in picks]


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _curve(scores: np.ndarray, norm: str) -> np.ndarray:
    return normalize_scores(np.asarray(scores, dtype=np.float64), norm)


def draw_clip(ax: Axes, clip: str, series: list[Series], prior: Series | None, norm: str,
              fps: float) -> None:
    """One clip: GT band, each arm's curve, the prior dashed. Per-clip AUCs go in the title so
    the legend (identity only) can sit outside the plot area and never cover a curve."""
    lab = series[0].labels[clip]
    t = np.arange(len(lab)) / fps
    ax.fill_between(t, 0, 1, where=(lab > 0).tolist(), color=GT_COLOR, step="mid",
                    label="ground truth", transform=ax.get_xaxis_transform())
    aucs = []
    for color, s in zip(SERIES_COLORS, series, strict=False):
        ax.plot(t, _curve(s.scores[clip], norm), color=color, linewidth=LINE_WIDTH, label=s.name)
        aucs.append(f"{s.name} {frame_auc(s.scores[clip], lab):.2f}")
    if prior is not None:
        ax.plot(t, _curve(prior.scores[clip], norm), color=PRIOR_COLOR, linewidth=1.4,
                linestyle="--", label=prior.name)
        aucs.append(f"p_T2 {frame_auc(prior.scores[clip], lab):.2f}")
    _style(ax)
    ax.set_xlim(0, t[-1] if len(t) > 1 else 1)
    if norm == constants.SCORE_NORM_MINMAX:
        ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{clip}   AUC: " + " · ".join(aucs), fontsize="small", loc="left")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score" + (" (per-clip min-max)" if norm == constants.SCORE_NORM_MINMAX else ""))


def _shared_legend(fig: Any, ax: Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize="small", bbox_to_anchor=(0.5, 0.0))


def run_curves(args: argparse.Namespace) -> list[Path]:
    series, prior = load_series(args.arm, args.prior)
    by_name = {s.name: s for s in series}
    target = by_name.get(args.sort_by or series[0].name)
    if target is None:
        raise ValueError(f"--sort-by {args.sort_by!r} is not an --arm name {sorted(by_name)}")
    clips = args.clips or pick_clips(target, args.n)
    missing = [c for c in clips if c not in series[0].labels]
    if missing:
        raise ValueError(f"clips not in the score files: {missing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for clip in clips:
        fig, ax = plt.subplots(figsize=CURVE_SIZE)
        draw_clip(ax, clip, series, prior, args.norm, args.fps)
        _shared_legend(fig, ax)
        fig.tight_layout(rect=(0, LEGEND_BAND, 1, 1))
        written.append(args.out_dir / f"{clip}.png")
        fig.savefig(written[-1], dpi=DPI, facecolor=SURFACE)
        plt.close(fig)
    rows = -(-len(clips) // GRID_COLUMNS)
    fig, axes = plt.subplots(rows, GRID_COLUMNS, squeeze=False,
                             figsize=(CURVE_SIZE[0] * GRID_COLUMNS * 0.75, CURVE_SIZE[1] * rows))
    for ax, clip in zip(axes.flat, clips, strict=False):
        draw_clip(ax, clip, series, prior, args.norm, args.fps)
    for ax in list(axes.flat)[len(clips):]:
        ax.set_visible(False)
    _shared_legend(fig, axes.flat[0])
    if args.title:
        fig.suptitle(args.title, x=0.01, ha="left")
    fig.tight_layout(rect=(0, LEGEND_BAND / rows, 1, 1))
    written.append(args.out_dir / GRID_PNG)
    fig.savefig(written[-1], dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    LOGGER.info("%d clip figures + grid -> %s", len(clips), args.out_dir)
    return written


# ---------------------------------------------------------------------------
# ROC / PR
# ---------------------------------------------------------------------------
def pooled(series: Series, norm: str) -> dict[str, Any]:
    """Pooled frames under ``norm`` (per clip), with the read-outs' own metric numbers."""
    ids = sorted(series.labels)
    scores = [np.asarray(series.scores[v], dtype=np.float64) for v in ids]
    labels = [np.asarray(series.labels[v]) for v in ids]
    metrics = pooled_metrics(scores, labels, norm)
    y = np.concatenate(labels)
    s = np.concatenate([normalize_scores(x, str(metrics["score_norm"])) for x in scores])
    fpr, tpr, _ = roc_curve(y, s)
    precision, recall, _ = precision_recall_curve(y, s)
    return {"metrics": metrics, "fpr": fpr, "tpr": tpr, "precision": precision,
            "recall": recall, "positive_rate": float(y.mean())}


def run_roc(args: argparse.Namespace) -> dict[str, Any]:
    series, prior = load_series(args.arm, args.prior)
    entries = [(s, SERIES_COLORS[i], "-") for i, s in enumerate(series)]
    if prior is not None:
        entries.append((prior, PRIOR_COLOR, "--"))
    reads = {s.name: pooled(s, args.norm) for s, _, _ in entries}

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=ROC_SIZE)
    for s, color, style in entries:
        r, m = reads[s.name], reads[s.name]["metrics"]
        ax_roc.plot(r["fpr"], r["tpr"], color=color, linestyle=style, linewidth=LINE_WIDTH,
                    label=f"{s.name}: micro {m['auc']:.3f} · macro {m['auc_macro']:.3f}")
        ax_pr.plot(r["recall"], r["precision"], color=color, linestyle=style,
                   linewidth=LINE_WIDTH, label=f"{s.name}: AP {m['ap']:.3f}")
    ax_roc.plot([0, 1], [0, 1], color=GRID_COLOR, linewidth=1.0, linestyle=":")
    base = next(iter(reads.values()))["positive_rate"]
    ax_pr.axhline(base, color=GRID_COLOR, linewidth=1.0, linestyle=":")
    for ax, (x, y) in ((ax_roc, ("false positive rate", "true positive rate")),
                       (ax_pr, ("recall", "precision"))):
        _style(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend(loc="lower right" if ax is ax_roc else "upper right", fontsize="x-small",
                  frameon=False)
    ax_roc.set_title("ROC (pooled frames)", loc="left", fontsize="medium")
    ax_pr.set_title(f"Precision-recall (positives {base:.2f})", loc="left", fontsize="medium")
    if args.title:
        fig.suptitle(f"{args.title} — score norm: {args.norm}", x=0.01, ha="left")
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / ROC_PNG, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    record = {"title": args.title, "norm": args.norm,
              "series": {name: r["metrics"] for name, r in reads.items()}}
    write_json_atomic(args.out_dir / ROC_JSON, record)
    LOGGER.info("ROC/PR -> %s", args.out_dir / ROC_PNG)
    return record


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, helptext in (("curves", "per-clip score curves"), ("roc", "pooled ROC + PR")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--arm", nargs=2, action="append", required=True, metavar=("NAME", "NPZ"),
                       help="a protocol-B clip_scores.npz; repeat (≤ 4, legend order)")
        p.add_argument("--prior", type=Path, default=None, help="position_prior.npz (p_T2)")
        p.add_argument("--norm", choices=NORMS, default=constants.SCORE_NORM_MINMAX,
                       help="per-clip score normalization (default min-max, lesson 12)")
        p.add_argument("--title", default="")
        p.add_argument("--out-dir", type=Path, required=True)
    curves = sub.choices["curves"]
    curves.add_argument("--clips", nargs="+", default=None, help="clip ids (default: --n picks)")
    curves.add_argument("--n", type=int, default=6, help="clips spread over the AUC rank")
    curves.add_argument("--sort-by", default=None, help="arm whose per-clip AUC ranks the picks")
    curves.add_argument("--fps", type=float, default=constants.DOTA_FPS)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "curves":
        run_curves(args)
    else:
        run_roc(args)


if __name__ == "__main__":
    main()
