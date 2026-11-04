"""DAD position diagnostics (``core/docs/final/ZS_DAD.md`` §8): printed, never gated.

Under ``toa = 90`` every positive video is abnormal on the same last rows, so the macro cannot
separate "reacts to the accident" from "scores rise with time". Pure aggregation over the saved
``core.evaluate`` scores of each row, no model is loaded. It prints:

* **ruler**: the no-pixel ``t/N`` macro;
* **placebo**: the positives' accident-row mask applied to the negative videos, which hold no
  accident, on all rows and on the rows whose V2-S window holds no repeated first frame;
* **same-position AUC**: positive vs negative videos on the same rows (position cannot help),
  on the accident rows and on the full-window rows before them, with the paired row contrasts.

::

    python -m core.tools.dad_position_diag --dad-root outputs/v_final/zs_lagovad/eval/dad \\
        --out-dir outputs/v_final/zs_lagovad/REPORTS
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import frame_auc
from core.tools import tadz_readout
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

REPORT_STEM = "dad_position_diag"
ROWS = (constants.ZS_LAGOVAD_ARM, constants.ZS_CONTROL_ARM, constants.ZS_SA2_ARM)
CONTRASTS = (
    (constants.ZS_CONTROL_ARM, constants.ZS_LAGOVAD_ARM),
    (constants.ZS_SA2_ARM, constants.ZS_LAGOVAD_ARM),
    (constants.ZS_SA2_ARM, constants.ZS_CONTROL_ARM),
)
LABEL = "POST-HOC, printed not gated, n = 1 seed"

Interval = dict[str, float]
Draws = tuple[np.ndarray, np.ndarray]


def first_full_window_row(
    stride: int = constants.DAD_STRIDE,
    step: int = constants.DAD_VIDEOMAE_FRAME_STEP,
    frames: int = constants.VIDEOMAE_CLIP_FRAMES,
) -> int:
    """First row whose causal V2-S window needs no clamp to frame 0 (``causal_clip_indices``)."""
    return math.ceil(step * (frames - 1) / stride)


def accident_rows(labels: dict[str, np.ndarray]) -> np.ndarray:
    """The one abnormal-row pattern shared by every positive; raises if they differ."""
    patterns = {tuple(np.flatnonzero(g > 0).tolist()) for g in labels.values() if g.sum() > 0}
    lengths = {len(g) for g in labels.values()}
    if len(patterns) != 1 or len(lengths) != 1:
        raise ValueError(f"not a fixed-toa set: {len(patterns)} patterns, lengths {lengths}")
    return np.array(patterns.pop())


def mann_whitney(pos: np.ndarray, neg: np.ndarray) -> float:
    """``P(pos > neg) + 0.5 P(pos == neg)``, the AUC of two samples."""
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def _interval(point: float, samples: np.ndarray) -> Interval:
    tail = (1.0 - constants.V2_E1_CI) / 2.0 * 100.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return {"mean": float(point), "low": float(low), "high": float(high)}


def within_video(curves: np.ndarray, rows: np.ndarray, abnormal: np.ndarray) -> np.ndarray:
    """Each video's frame AUC of ``abnormal`` against the other rows of ``rows``."""
    truth = np.isin(rows, abnormal).astype(np.int64)
    return np.array([frame_auc(c[rows], truth) for c in curves])


def row_reads(pos: np.ndarray, neg: np.ndarray, abnormal: np.ndarray, full_from: int,
              draws: Draws) -> dict[str, Any]:
    """Every diagnostic of one row from its ``(videos, rows)`` score matrices."""
    every = np.arange(pos.shape[1])
    full = every[full_from:]
    before = np.setdiff1d(full, abnormal)
    out: dict[str, Any] = {}
    for name, rows in (("all_rows", every), ("full_window_rows", full)):
        a, b = within_video(pos, rows, abnormal), within_video(neg, rows, abnormal)
        out[name] = {
            "positive": float(a.mean()),
            "placebo": float(b.mean()),
            "gap": _interval(a.mean() - b.mean(),
                             a[draws[0]].mean(axis=1) - b[draws[1]].mean(axis=1)),
            "placebo_ci": _interval(b.mean(), b[draws[1]].mean(axis=1)),
        }
    at_pos, at_neg = pos[:, abnormal].mean(axis=1), neg[:, abnormal].mean(axis=1)
    pre_pos, pre_neg = pos[:, before].mean(axis=1), neg[:, before].mean(axis=1)
    at_boot = np.array([mann_whitney(at_pos[i], at_neg[j]) for i, j in zip(*draws, strict=True)])
    pre_boot = np.array([mann_whitney(pre_pos[i], pre_neg[j]) for i, j in zip(*draws, strict=True)])
    at, pre = mann_whitney(at_pos, at_neg), mann_whitney(pre_pos, pre_neg)
    out["same_position"] = {
        "accident_rows": _interval(at, at_boot),
        "rows_before": _interval(pre, pre_boot),
        "accident_minus_before": _interval(at - pre, at_boot - pre_boot),
        "_boot": at_boot,
    }
    return out


def diagnose(curves: dict[str, dict[str, np.ndarray]], labels: dict[str, np.ndarray],
             resamples: int, seed: int) -> dict[str, Any]:
    """The full report from ``{row: {video: score}}`` and the shared labels."""
    abnormal = accident_rows(labels)
    ids = sorted(labels)
    pos_ids = [v for v in ids if labels[v].sum() > 0]
    neg_ids = [v for v in ids if labels[v].sum() == 0]
    length = len(labels[ids[0]])
    full_from = first_full_window_row()
    rng = np.random.default_rng(seed)
    draws: Draws = (rng.integers(0, len(pos_ids), (resamples, len(pos_ids))),
                    rng.integers(0, len(neg_ids), (resamples, len(neg_ids))))
    rows = {
        row: row_reads(np.stack([curves[row][v] for v in pos_ids]),
                       np.stack([curves[row][v] for v in neg_ids]), abnormal, full_from, draws)
        for row in ROWS
    }
    boots = {row: rows[row]["same_position"].pop("_boot") for row in ROWS}
    ruler = np.arange(length, dtype=np.float64)
    truth = np.isin(np.arange(length), abnormal).astype(np.int64)
    return {
        "label": LABEL,
        "positives": len(pos_ids),
        "negatives": len(neg_ids),
        "rows_per_video": length,
        "accident_rows": abnormal.tolist(),
        "first_full_window_row": full_from,
        "ruler_macro": frame_auc(ruler, truth),
        "rows": rows,
        "same_position_contrasts": {
            f"{a} - {b}": _interval(
                rows[a]["same_position"]["accident_rows"]["mean"]
                - rows[b]["same_position"]["accident_rows"]["mean"],
                boots[a] - boots[b],
            )
            for a, b in CONTRASTS
        },
    }


def _ci(c: Interval, signed: bool = False) -> str:
    form = "+.4f" if signed else ".4f"
    return f"{c['mean']:{form}} [{c['low']:{form}}, {c['high']:{form}}]"


def render(r: dict[str, Any]) -> str:
    full = r["first_full_window_row"]
    lines = [
        f"# DAD position diagnostics ({r['label']})",
        "",
        f"{r['positives']} positives, {r['negatives']} negatives, {r['rows_per_video']} rows each; "
        f"accident rows {r['accident_rows']}; first V2-S window without a repeated frame = row "
        f"{full}. **No-pixel `t/N` ruler macro = {r['ruler_macro']:.4f}.**",
        "",
        "## Within video: positives vs the same mask on negatives (placebo)",
        "",
        "| row | macro, all rows | placebo, all rows | gap | "
        f"macro, rows {full}+ | placebo, rows {full}+ | gap |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ROWS:
        a, f = r["rows"][row]["all_rows"], r["rows"][row]["full_window_rows"]
        lines.append(f"| {row} | {a['positive']:.4f} | {_ci(a['placebo_ci'])} | "
                     f"{_ci(a['gap'], True)} | {f['positive']:.4f} | "
                     f"{_ci(f['placebo_ci'])} | {_ci(f['gap'], True)} |")
    lines += [
        "",
        "## Same position: positive vs negative videos (position cannot help)",
        "",
        f"| row | accident rows | rows {full} up to the accident | difference |",
        "|---|---|---|---|",
    ]
    for row in ROWS:
        s = r["rows"][row]["same_position"]
        lines.append(f"| {row} | {_ci(s['accident_rows'])} | {_ci(s['rows_before'])} | "
                     f"{_ci(s['accident_minus_before'], True)} |")
    lines += ["", "Accident rows, paired:", ""]
    lines += [f"* {k}: {_ci(c, True)}" for k, c in r["same_position_contrasts"].items()]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--dad-root", type=Path, required=True,
                        help="<zs_lagovad root>/eval/dad (<row>/<run>/scores under it)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=constants.ZS_SEED, help="the arms' seed")
    parser.add_argument("--bootstrap", type=int, default=constants.V2_E1_BOOTSTRAP)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    runs = {row: tadz_readout.load_run(
        args.dad_root / row
        / (constants.ZS_LAGOVAD_RUN if row == constants.ZS_LAGOVAD_ARM else f"s{args.seed}"))
        for row in ROWS}
    labels = tadz_readout.check_labels(runs, None)
    report = diagnose({row: runs[row][0] for row in ROWS}, labels, args.bootstrap,
                      constants.V5_BOOTSTRAP_SEED)
    write_json_atomic(args.out_dir / f"{REPORT_STEM}.json", report)
    write_text_atomic(args.out_dir / f"{REPORT_STEM}.md", render(report))
    LOGGER.info("position diagnostics -> %s", args.out_dir)
    return report


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
