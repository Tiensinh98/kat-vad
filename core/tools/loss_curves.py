"""Training-loss curves of one or more runs, read from the ``metrics.jsonl`` each run wrote.

Nothing is re-run: every value is the per-step record ``core.train`` logged. One panel per key
(``constants.LOSS_CURVES_KEYS``: ``total``, each loss term, ``lr``), every run overlaid on global
step: a centred rolling mean, inside a shaded band of +-1 rolling standard deviation of the
per-step values over the same window (minibatch noise). A key a run never
logged is left out of that run's panel (``kim_nkc`` starts once the prototype is ready). Panel
titles carry the term's weight in ``total`` when every run trained it with the same weight
(read from the ``config.yaml`` beside the metrics).

``loss_curves_<name>.json`` holds, per run and key, the mean over the first and last
``LOSS_CURVES_EDGE_STEPS`` logged steps, the smoothed minimum and its step, plus the run's
epochs / steps per epoch / dataset and ``total_residual`` = max |total - sum(w * term)| over
the logged terms (≈ 0 when every term in ``total`` is logged). Illustration only (n = 1 seed):
decides nothing.

CLI::

    python -m core.tools.loss_curves --name k5dfi_nexar_vs_dota_s2024 --out-dir figs \\
        --run "K5dfi (Nexar)" outputs/v_final/kim_dvs_ignore/K5dfi/K5/s2024 \\
        --run "K5dfiDoTA (DoTA)" outputs/v_final/kim_dota_train/K5dfiDoTA/K5/s2024
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")  # headless: render straight to file
import matplotlib.pyplot as plt

from core import constants
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

LABEL = "illustration (n = 1 seed); decides nothing"
STEP_KEY = "global_step"
EPOCH_KEY = "epoch"
LR_KEY = "lr"
TOTAL_KEY = "total"


class LossCurvesError(ValueError):
    """A run dir or key the tool cannot read."""


def load_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """Every record of ``<run_dir>/metrics.jsonl`` in step order; raises if absent or empty."""
    path = run_dir / constants.LOSS_CURVES_METRICS_FILENAME
    if not path.is_file():
        raise LossCurvesError(f"no {constants.LOSS_CURVES_METRICS_FILENAME} in {run_dir}")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise LossCurvesError(f"{path} is empty")
    return sorted(rows, key=lambda r: int(r[STEP_KEY]))


def load_config(run_dir: Path) -> dict[str, Any]:
    """The run's ``config.yaml`` (empty when absent: weights are then unknown, not assumed)."""
    path = run_dir / constants.LOSS_CURVES_CONFIG_FILENAME
    if not path.is_file():
        LOGGER.warning(
            "no %s in %s: term weights unknown", constants.LOSS_CURVES_CONFIG_FILENAME, run_dir
        )
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def term_weight(config: dict[str, Any], key: str) -> float | None:
    """``key``'s weight in ``total``; None when it is not a loss term or the config lacks it."""
    if key not in constants.LOSS_CURVES_WEIGHTS:
        return None
    dotted = constants.LOSS_CURVES_WEIGHTS[key]
    if dotted is None:
        return 1.0
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, int | float) else None


def series(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """(steps, values) of the records that logged ``key``."""
    kept = [(int(r[STEP_KEY]), float(r[key])) for r in rows if key in r and r[key] is not None]
    if not kept:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    steps, values = zip(*kept, strict=True)
    return np.asarray(steps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def _window_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred window mean via a cumulative sum; the window shrinks at both ends (no padding)."""
    cum = np.concatenate([[0.0], np.cumsum(values)])
    half = window // 2
    idx = np.arange(values.size)
    lo = np.clip(idx - half, 0, values.size)
    hi = np.clip(idx - half + window, 0, values.size)
    out: np.ndarray = (cum[hi] - cum[lo]) / (hi - lo)
    return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean; the window shrinks at both ends (no padding bias)."""
    if window <= 1 or values.size == 0:
        return values.copy()
    return _window_mean(values, window)


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling (population) standard deviation over the same windows as ``rolling_mean``."""
    if window <= 1 or values.size == 0:
        return np.zeros_like(values)
    var = _window_mean(values**2, window) - _window_mean(values, window) ** 2
    out: np.ndarray = np.sqrt(np.clip(var, 0.0, None))  # clip: float round-off below 0
    return out


def total_residual(rows: list[dict[str, Any]], config: dict[str, Any]) -> float | None:
    """max |total - sum(w * term)| over the steps; None when a logged term's weight is unknown."""
    worst = 0.0
    for row in rows:
        if TOTAL_KEY not in row:
            continue
        acc = 0.0
        for key in constants.LOSS_CURVES_WEIGHTS:
            if key not in row:
                continue
            w = term_weight(config, key)
            if w is None:
                return None
            acc += w * float(row[key])
        worst = max(worst, abs(float(row[TOTAL_KEY]) - acc))
    return worst


def summarize(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    keys: tuple[str, ...],
    smooth: int,
    edge: int,
) -> dict[str, Any]:
    """Per-key first / last means and smoothed minimum, plus the run's shape."""
    per_key: dict[str, Any] = {}
    for key in keys:
        steps, values = series(rows, key)
        if values.size == 0:
            continue
        sm = rolling_mean(values, smooth)
        i_min = int(np.argmin(sm))
        per_key[key] = {
            "first_step": int(steps[0]),
            "first_mean": float(values[:edge].mean()),
            "last_mean": float(values[-edge:].mean()),
            "smoothed_min": float(sm[i_min]),
            "smoothed_min_step": int(steps[i_min]),
            "weight": term_weight(config, key),
        }
    epochs = sorted({int(r[EPOCH_KEY]) for r in rows if EPOCH_KEY in r})
    first_epoch = epochs[0] if epochs else None
    return {
        "steps": int(rows[-1][STEP_KEY]),
        "epochs": len(epochs),
        "steps_per_epoch": sum(1 for r in rows if r.get(EPOCH_KEY) == first_epoch)
        if epochs
        else None,
        "dataset": (config.get("data") or {}).get("dataset"),
        "total_residual": total_residual(rows, config),
        "keys": per_key,
    }


def panel_title(key: str, configs: list[dict[str, Any]]) -> str:
    """``key``, plus its weight in ``total`` when every run used the same known one."""
    if key == TOTAL_KEY:
        return "total (as trained)"
    weights = {term_weight(c, key) for c in configs}
    if (
        len(weights) == 1
        and (w := weights.pop()) is not None
        and key in constants.LOSS_CURVES_WEIGHTS
    ):
        return key if w == 1.0 else f"{key} (x{w:g} in total)"
    return key


def plot(
    runs: list[tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]],
    keys: tuple[str, ...],
    smooth: int,
    title: str,
    out: Path,
) -> None:
    """One panel per key, every run overlaid (rolling mean solid, +-1 rolling std shaded)."""
    cols = min(constants.LOSS_CURVES_COLS, len(keys))
    nrows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(
        nrows,
        cols,
        squeeze=False,
        figsize=(constants.LOSS_CURVES_PANEL_W * cols, constants.LOSS_CURVES_PANEL_H * nrows),
    )
    configs = [cfg for _, _, cfg, _ in runs]
    for ax, key in zip(axes.flat, keys, strict=False):
        for color, (label, rows, _, summary) in zip(constants.ZS_CURVES_COLORS, runs, strict=False):
            steps, values = series(rows, key)
            if values.size == 0:
                continue
            mean, std = rolling_mean(values, smooth), rolling_std(values, smooth)
            ax.fill_between(
                steps,
                mean - std,
                mean + std,
                color=color,
                alpha=constants.LOSS_CURVES_BAND_ALPHA,
                linewidth=0,
            )
            ax.plot(
                steps,
                mean,
                color=color,
                linewidth=1.6,
                label=f"{label} ({summary['epochs']} ep x {summary['steps_per_epoch']} steps)",
            )
        ax.set_title(panel_title(key, configs), fontsize=10)
        ax.set_xlabel("global step")
        ax.grid(alpha=0.3)
        if key != LR_KEY:
            ax.set_ylim(bottom=0.0)
    axes.flat[0].legend(fontsize=7, loc="upper right")
    for ax in list(axes.flat)[len(keys) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"{title}\nline = rolling mean over {smooth} steps, band = +-1 rolling std "
        f"(minibatch noise) ({LABEL})",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=constants.ZS_CURVES_DPI)
    plt.close(fig)


def run(
    run_specs: list[tuple[str, Path]],
    keys: tuple[str, ...],
    name: str,
    out_dir: Path,
    smooth: int,
    edge: int,
    title: str,
) -> dict[str, Any]:
    """Load every run, draw ``loss_curves_<name>.png``, write ``loss_curves_<name>.json``."""
    if len(run_specs) > len(constants.ZS_CURVES_COLORS):
        raise LossCurvesError(
            f"at most {len(constants.ZS_CURVES_COLORS)} runs, got {len(run_specs)}"
        )
    if len({label for label, _ in run_specs}) != len(run_specs):
        raise LossCurvesError("run labels must be unique")
    runs = []
    for label, run_dir in run_specs:
        rows, config = load_metrics(run_dir), load_config(run_dir)
        runs.append((label, rows, config, summarize(rows, config, keys, smooth, edge)))
    missing = [k for k in keys if not any(k in r for _, rows, _, _ in runs for r in rows)]
    if missing:
        raise LossCurvesError(f"no run logged {missing}")
    stem = f"{constants.LOSS_CURVES_STEM}_{name}"
    plot(runs, keys, smooth, title, out_dir / f"{stem}.png")
    payload = {
        "label": LABEL,
        "smooth_steps": smooth,
        "edge_steps": edge,
        "keys": list(keys),
        "runs": {
            label: {"run_dir": str(run_dir), **summary}
            for (label, run_dir), (_, _, _, summary) in zip(run_specs, runs, strict=True)
        },
    }
    write_json_atomic(out_dir / f"{stem}.json", payload)
    for label, _, _, summary in runs:
        LOGGER.info(
            "%s: %d steps, %d epochs, dataset %s, total residual %s",
            label,
            summary["steps"],
            summary["epochs"],
            summary["dataset"],
            summary["total_residual"],
        )
    LOGGER.info("wrote %s.{png,json} in %s", stem, out_dir)
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run",
        nargs=2,
        action="append",
        required=True,
        metavar=("LABEL", "RUN_DIR"),
        help="a run dir holding metrics.jsonl (+ config.yaml); repeat per run",
    )
    p.add_argument("--keys", nargs="+", default=list(constants.LOSS_CURVES_KEYS))
    p.add_argument("--name", required=True, help="file tag: loss_curves_<name>.{png,json}")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--smooth", type=int, default=constants.LOSS_CURVES_SMOOTH_STEPS)
    p.add_argument("--edge", type=int, default=constants.LOSS_CURVES_EDGE_STEPS)
    p.add_argument("--title", default="Training losses")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    run(
        [(label, Path(d)) for label, d in args.run],
        tuple(args.keys),
        args.name,
        args.out_dir,
        args.smooth,
        args.edge,
        args.title,
    )


if __name__ == "__main__":
    main()
