"""Recompute eval metrics from a saved ``scores/`` directory (lesson C12).

Model outputs are deterministic given the features, so changing the *pooling
rule* never needs re-running inference -- the per-video ``.npz`` files written
by ``core.evaluate --save-scores`` already contain the score curve and the
ground truth. This CLI re-derives the headline numbers from them, which is what
makes a protocol correction cost minutes instead of GPU hours.

Prints every pooling rule side by side so the choice is auditable, and
optionally rewrites ``results.json`` in place (``--write``), preserving
``per_video`` and ``checkpoint`` from the existing file.

CLI::

    python -m core.tools.rescore --scores outputs/DoTA/gate_a/scores
    python -m core.tools.rescore --run-dir outputs/DoTA/gate_a --write
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from core import constants
from core.evaluate import RESULTS_FILENAME, SCORES_DIRNAME
from core.metrics import pooled_metrics

LOGGER = logging.getLogger(__name__)

SCORE_KEY = "score"
GT_KEY = "gt"


def load_scores(scores_dir: Path) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Read every ``{video_id}.npz`` in id order -> (ids, scores, labels).

    Raises on a file without ``gt``: those were written by ``core.inference``
    (no labels available) and cannot be scored, and silently skipping them
    would report a metric over an unknown subset.
    """
    paths = sorted(scores_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"No .npz score files under {scores_dir}")
    ids: list[str] = []
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for path in paths:
        payload = np.load(path)
        if GT_KEY not in payload:
            raise ValueError(
                f"{path} has no '{GT_KEY}' array -- it was written without ground "
                "truth and cannot be rescored"
            )
        score = payload[SCORE_KEY].astype(np.float64)
        gt = payload[GT_KEY].astype(np.float64)
        if score.shape != gt.shape:
            raise ValueError(
                f"{path}: score {score.shape} and gt {gt.shape} disagree"
            )
        ids.append(path.stem)
        scores.append(score)
        labels.append(gt)
    return ids, scores, labels


def compare_norms(
    scores: list[np.ndarray], labels: list[np.ndarray]
) -> dict[str, dict[str, float | int | str]]:
    """Metrics under every concrete pooling rule, keyed by rule name."""
    concrete = [m for m in constants.SCORE_NORM_CHOICES if m != constants.SCORE_NORM_AUTO]
    return {method: pooled_metrics(scores, labels, method) for method in concrete}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scores", type=Path, help="directory of per-video .npz files")
    source.add_argument("--run-dir", type=Path,
                        help=f"eval output dir containing {SCORES_DIRNAME}/")
    parser.add_argument("--score-norm", choices=constants.SCORE_NORM_CHOICES,
                        default=constants.SCORE_NORM_AUTO,
                        help="rule recorded when --write is given")
    parser.add_argument("--write", action="store_true",
                        help=f"update {RESULTS_FILENAME} in the run dir in place")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    run_dir = args.run_dir
    scores_dir = args.scores if args.scores else run_dir / SCORES_DIRNAME
    if args.write and run_dir is None:
        raise SystemExit("--write needs --run-dir (it rewrites that run's results.json)")

    ids, scores, labels = load_scores(scores_dir)
    abnormal = sum(int(gt.max() > 0) for gt in labels)
    LOGGER.info(
        "%d videos, %d frames, %d abnormal", len(ids), sum(len(g) for g in labels), abnormal
    )
    for method, metrics in compare_norms(scores, labels).items():
        LOGGER.info(
            "norm=%-7s AUC %.4f | AP %.4f | macro AUC %.4f (%d videos)",
            method, metrics["auc"], metrics["ap"],
            metrics["auc_macro"], metrics["auc_macro_videos"],
        )

    if not args.write:
        return
    results_path = run_dir / RESULTS_FILENAME
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results.update(pooled_metrics(scores, labels, args.score_norm))
    results["num_videos"] = len(ids)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info(
        "Rewrote %s with norm=%s AUC %.4f",
        results_path, results["score_norm"], results["auc"],
    )


if __name__ == "__main__":
    main()
