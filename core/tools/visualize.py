"""Anomaly-curve visualization CLI — one PNG per scored video.

Consumes the ``.npz`` files written by ``core/inference.py`` /
``core/evaluate.py --save-scores`` (keys: ``score (L,)``, optional ``gt (L,)``,
optional ``sim (L, C)`` + ``class_names``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: render straight to file
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

FIGSIZE = (10.0, 3.0)
DPI = 150
GT_ALPHA = 0.25


def render_video_scores(npz_path: Path, out_dir: Path, threshold: float | None) -> Path:
    """Render one score file to ``<out_dir>/<video_id>.png``."""
    data = np.load(npz_path, allow_pickle=False)
    score = data["score"]
    frames = np.arange(len(score))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if "gt" in data:
        ax.fill_between(
            frames, 0.0, 1.0, where=data["gt"] > 0.5,
            color="tab:red", alpha=GT_ALPHA, label="ground truth",
        )
    ax.plot(frames, score, color="tab:blue", linewidth=1.5, label="anomaly score")
    if threshold is not None:
        ax.axhline(threshold, color="tab:orange", linestyle="--",
                   linewidth=1.0, label=f"threshold {threshold:.2f}")
    ax.set_xlim(0, max(1, len(score) - 1))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("sampled frame")
    ax.set_ylabel("anomaly score")
    ax.set_title(npz_path.stem)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{npz_path.stem}.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render anomaly-curve PNGs")
    parser.add_argument("--scores", type=Path, required=True,
                        help="score .npz file or directory of them")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None,
                        help="draw a horizontal decision-threshold line")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    files = (
        sorted(args.scores.glob("*.npz")) if args.scores.is_dir() else [args.scores]
    )
    if not files:
        raise FileNotFoundError(f"No .npz score files under {args.scores}")
    for npz_path in files:
        out_path = render_video_scores(npz_path, args.output_dir, args.threshold)
        LOGGER.info("%s -> %s", npz_path.name, out_path)


if __name__ == "__main__":
    main()
