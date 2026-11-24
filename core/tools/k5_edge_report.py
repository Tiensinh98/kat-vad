"""K5fepre / K5fepost / K5fbk against K5f (``KIM_RUN.md`` §49): early rise, macro, verdicts.

Pure aggregation over saved reads, PRELIMINARY (n = 1). Rows: K5f (control) and any of K5fepre
(pre-span edge normal windows), K5fepost (post-span ones), K5fbk (bottom-k MIL). Every contrast is
row - K5f, paired per item with ``k5_whole_report``'s readers.

* **T1** (ED8): zone AUC of the ``EDGE_TARGET_ZONE_S`` frames of ``nexar_val``'s positive videos
  against every negative-video frame; paired bootstrap over positive and negative videos. Hit iff
  the interval of row - K5f is below 0.
* **T2** (ED8): per DoTA-CAP-dev clip, the onset AUC = the span's frames against the normal frames
  of the last ``EDGE_ONSET_S`` before it; source-video clusters. Hit iff the interval is above 0.
  The same read on Nexar val / DADA and its post-span twin are printed.
* **Labels** (ED9): ``k5_dvs_ignore_report.label`` per set (margin ``V5_MIN_GAIN``).
* **Verdicts**: ED10 for K5fepre, ED11 for K5fepost / K5fbk.

Guards (raise): last-epoch ``mul_mil`` above the §31 floor; one checkpoint per row, none shared.
Flag (printed): K5fbk's step-1 ``mil`` equal to K5f's within ``KIM_START_TOL``. CLI::

    python -m core.tools.k5_edge_report --kim-root outputs/kim_fixed/K5f \\
        --k5f-nexar-dir outputs/kim_fixed/K5f/eval/nexar/K5f \\
        --edge-root outputs/kim_edge --out-dir outputs/kim_edge/REPORTS [--rows K5fepre K5fbk]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools import k5_dvs_ignore_report as dvs
from core.tools import k5_neg_report as kn
from core.tools import k5_whole_report as kw
from core.tools import k5_zs_report as zs_reads
from core.tools import zs_compare_table as zs
from core.tools.k5_lneg_cap_report import _cell, check_fixed_runs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.pre_onset_read import Curves, load_rows, onset_seconds
from core.train import METRICS_FILENAME

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/KIM_RUN.md §49"
K5F = constants.KIM_REFERENCE_ARM
PRE_ROW, POST_ROW = constants.KIM_EDGE_PRE_ROW, constants.KIM_EDGE_POST_ROW
BK_ROW = constants.KIM_BOTTOMK_ROW
NEXAR, DOTA = kw.NEXAR, zs_reads.DOTA
ALL_SETS = (NEXAR, *kw.ZS_SETS)
ONSET_SETS = (NEXAR, DOTA, zs_reads.DADA)  # sets with a frame rate in ZS_CURVES_STEP_SECONDS
PRE, POST = constants.NEXAR_EDGE_SIDE_PRE, constants.NEXAR_EDGE_SIDE_POST
GAIN, COST = constants.KIM_DVS_IGNORE_GAIN, constants.KIM_DVS_IGNORE_COST
REDUCED = "REDUCED (s2025 may run; K5f stays)"
TRADE_OFF = "TRADE-OFF (closed)"
PARTIAL = "PARTIAL (one target hit, closed)"
NOT_REDUCED = "NOT REDUCED (closed)"
GOES_ON = "GAIN and no COST (s2025 may run; K5f stays)"
CLOSED = "closed (no gain, or a cost)"
START_KEY = "mil"

Interval = dict[str, float]


def edge_seconds(gt: np.ndarray, fps: float, side: str) -> np.ndarray:
    """Seconds outside the span on one side (> 0 there, <= 0 elsewhere)."""
    if side == PRE:
        return -onset_seconds(gt, fps)
    last = int(np.flatnonzero(gt)[-1])
    return (np.arange(len(gt)) - last) / fps


def onset_aucs(scores: Curves, labels: Curves, fps: float, side: str) -> dict[str, float]:
    """Per two-class video: AUC of the span's frames vs the normal frames within
    ``EDGE_ONSET_S`` of the span on ``side``; a video without such frames is left out."""
    out: dict[str, float] = {}
    for v, gt in labels.items():
        if not 0 < gt.sum() < len(gt):
            continue
        gap = edge_seconds(gt, fps, side)
        near = (gt == 0) & (gap > 0) & (gap <= constants.EDGE_ONSET_S)
        if near.any():
            keep = near | (gt > 0)
            out[v] = frame_auc(scores[v][keep], gt[keep])
    return out


def zone_wins(scores: Curves, labels: Curves, fps: float) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray]:
    """``(wins, zone frames per positive, frames per negative)`` of T1.

    ``wins[p, n]`` = frame pairs (zone frame of positive ``p``, frame of negative ``n``) the zone
    frame wins, ties half, so any video reweighting of the pooled AUC is a quadratic form.
    """
    low, high = constants.EDGE_TARGET_ZONE_S
    positive = [v for v in sorted(labels) if labels[v].any()]
    negative = [np.sort(scores[v]) for v in sorted(labels) if not labels[v].any()]
    if not positive or not negative:
        raise ValueError("T1 needs positive and negative videos")
    wins = np.zeros((len(positive), len(negative)))
    zone_n = np.zeros(len(positive))
    for p, v in enumerate(positive):
        t = onset_seconds(labels[v], fps)
        zone = scores[v][(t >= low) & (t < high) & (labels[v] == 0)]
        zone_n[p] = zone.size
        for n, curve in enumerate(negative):
            wins[p, n] = (np.searchsorted(curve, zone, "left")
                          + np.searchsorted(curve, zone, "right")).sum() / 2
    return wins, zone_n, np.array([len(c) for c in negative], dtype=float)


def zone_delta(rows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], control: str
               ) -> dict[str, dict[str, Any]]:
    """Zone AUC per row and its paired difference from ``control`` (T1)."""
    _, zone_n, neg_n = rows[control]

    def auc(wins: np.ndarray, wp: np.ndarray, wn: np.ndarray) -> float:
        return float(wp @ wins @ wn / ((wp @ zone_n) * (wn @ neg_n)))

    ones_p, ones_n = np.ones(len(zone_n)), np.ones(len(neg_n))
    point = {name: auc(r[0], ones_p, ones_n) for name, r in rows.items()}
    rng = np.random.default_rng(constants.PRE_ONSET_BOOTSTRAP_SEED)
    draws: dict[str, list[float]] = {name: [] for name in rows if name != control}
    for _ in range(constants.PRE_ONSET_BOOTSTRAP):
        wp = np.bincount(rng.integers(0, len(zone_n), len(zone_n)),
                         minlength=len(zone_n)).astype(float)
        wn = np.bincount(rng.integers(0, len(neg_n), len(neg_n)),
                         minlength=len(neg_n)).astype(float)
        if wp @ zone_n == 0:
            continue
        base = auc(rows[control][0], wp, wn)
        for name, values in draws.items():
            values.append(auc(rows[name][0], wp, wn) - base)
    tail = (1.0 - constants.PRE_ONSET_LEVEL) / 2
    out: dict[str, dict[str, Any]] = {control: {"auc": point[control], "delta": None}}
    for name, values in draws.items():
        low, high = np.quantile(values, [tail, 1.0 - tail])
        out[name] = {"auc": point[name], "delta": {
            "mean": point[name] - point[control], "low": float(low), "high": float(high)}}
    return out


def paired_ci(new: dict[str, float], base: dict[str, float], groups: dict[str, str] | None
              ) -> Interval | None:
    """Mean of ``new - base`` per video; ``groups`` = its clusters (``None`` = the video itself)."""
    if set(new) != set(base):
        raise ValueError("onset AUC: the two rows are defined on different videos")
    d = {v: new[v] - base[v] for v in new}
    clusters = {v: groups[v] if groups else v for v in d}
    return cluster_bootstrap_ci(d, clusters, constants.PRE_ONSET_BOOTSTRAP,
                                constants.PRE_ONSET_BOOTSTRAP_SEED, constants.PRE_ONSET_LEVEL)


def verdict_pre(t1_hit: bool, t2_hit: bool, labels: dict[str, str]) -> str:
    """ED10."""
    if not (t1_hit or t2_hit):
        return NOT_REDUCED
    if any(labels[s] == COST for s in dvs.DECISION_SETS):
        return TRADE_OFF
    return REDUCED if t1_hit and t2_hit else PARTIAL


def verdict_macro(labels: dict[str, str]) -> str:
    """ED11: GAIN on DoTA-CAP-dev or DADA and COST on no deciding set."""
    return GOES_ON if dvs.verdict(labels) == GAIN else CLOSED


def start_check(control_run: Path, new_run: Path) -> dict[str, Any]:
    """K5fbk vs K5f at step 1 (ED6): the same draws give the same ``mil`` (a flag)."""
    a, b = (json.loads((d / METRICS_FILENAME).read_text(encoding="utf-8").splitlines()[0])
            for d in (control_run, new_run))
    if a["global_step"] != 1 or b["global_step"] != 1:
        raise ValueError("metrics.jsonl does not start at global_step 1")
    return {"control": a[START_KEY], "new": b[START_KEY],
            "same_draws": abs(a[START_KEY] - b[START_KEY]) <= constants.KIM_START_TOL}


def _score_path(view: argparse.Namespace, nexar_dir: Path, set_name: str) -> Path:
    return nexar_dir if set_name == NEXAR else zs_reads.score_source(view, set_name, kw.K5)


def onset_reads(views: dict[str, argparse.Namespace], dirs: dict[str, Path],
                groups: dict[str, dict[str, str]], control: str = K5F
                ) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(per-set onset AUC and deltas on both sides, T1)`` from the saved per-frame scores."""
    onset: dict[str, Any] = {}
    t1: dict[str, Any] = {}
    for set_name in ONSET_SETS:
        fps = 1.0 / constants.ZS_CURVES_STEP_SECONDS[set_name]
        scores, labels = load_rows(
            [(row, _score_path(views[row], dirs[row], set_name)) for row in views])
        if set_name == NEXAR:
            t1 = zone_delta({r: zone_wins(scores[r], labels, fps) for r in views}, control)
        onset[set_name] = {}
        for side in (PRE, POST):
            aucs = {row: onset_aucs(scores[row], labels, fps, side) for row in views}
            onset[set_name][side] = {row: {
                "videos": len(a), "auc": float(np.mean(list(a.values()))),
                "delta": (None if row == control
                          else paired_ci(a, aucs[control], groups.get(set_name))),
            } for row, a in aucs.items()}
    return onset, t1


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = f"s{args.seed}"
    rows = list(args.rows)
    roots = {K5F: args.kim_root, **{row: args.edge_root / row for row in rows}}
    run_dirs = {row: root / kw.K5 / run_root for row, root in roots.items()}
    mul_mil = check_fixed_runs(run_dirs)
    views = {row: kw._view(root, args.seed) for row, root in roots.items()}
    dirs = {K5F: args.k5f_nexar_dir,
            **{row: roots[row] / zs.EVAL_DIR / NEXAR / row for row in rows}}

    sets: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_views = {kw.K5: views[K5F], row: views[row]}  # the readers key the control as K5
        pair_dirs = {kw.K5: dirs[K5F], row: dirs[row]}
        kw.check_checkpoints(pair_views, pair_dirs, args.seed, new=row)
        sets[row] = {NEXAR: kw.nexar_set(pair_dirs, new=row),
                     **{s: kw.zero_shot_set(pair_views, s, new=row) for s in kw.ZS_SETS}}
    control_reads = {s: zs_reads.read_row(views[K5F], s, kw.K5)
                     for s in ONSET_SETS if s != NEXAR}
    groups = {s: r["groups"] for s, r in control_reads.items()}
    onset, t1 = onset_reads(views, dirs, groups)

    labels = {row: {s: dvs.label(sets[row][s]) for s in ALL_SETS} for row in rows}
    hits: dict[str, dict[str, bool]] = {}
    verdicts: dict[str, str] = {}
    for row in rows:
        t2 = onset[DOTA][PRE][row]["delta"]
        hits[row] = {"T1": t1[row]["delta"]["high"] < 0, "T2": t2 is not None and t2["low"] > 0}
        verdicts[row] = (verdict_pre(hits[row]["T1"], hits[row]["T2"], labels[row])
                         if row == PRE_ROW else verdict_macro(labels[row]))
    report: dict[str, Any] = {
        "rule": RULE, "seed": args.seed, "rows": rows, "fixed_mul_mil": mul_mil,
        "start_check": start_check(run_dirs[K5F], run_dirs[BK_ROW]) if BK_ROW in rows else None,
        "sets": sets, "labels": labels, "decision_sets": list(dvs.DECISION_SETS),
        "t1": t1, "onset": onset, "hits": hits, "verdicts": verdicts,
        "false_alarms": {row: kn.false_alarms(kn.load_video_stats(d))["share"]
                         for row, d in dirs.items()},
    }
    stem = constants.KIM_EDGE_REPORT_STEM
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    LOGGER.info("verdicts %s -> %s", verdicts, args.out_dir / f"{stem}.md")
    return report


def _signed(c: Interval | None) -> str:
    return "-" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _macro(cell: dict[str, Any], set_name: str, key: str) -> str:
    cells = cell["rows"][key]
    return f"{cells['macro_crops' if set_name == NEXAR else 'macro']['mean']:.4f}"


def render(r: dict[str, Any]) -> str:
    rows = r["rows"]
    lines = [f"# {' / '.join(rows)} vs {K5F} -- s{r['seed']}", "",
             f"Rule: `{r['rule']}`. PRELIMINARY, n = 1; nothing is adopted, {K5F} stays the "
             "reference. Last-epoch mul_mil: "
             + ", ".join(f"{n} {x:.4f}" for n, x in r["fixed_mul_mil"].items()) + ".", ""]
    for row in rows:
        h = r["hits"][row]
        lines.append(f"* **{row}: {r['verdicts'][row]}.** T1 hit {h['T1']}, T2 hit {h['T2']}"
                     + ("" if row == PRE_ROW else " (printed)") + ".")
    sc = r["start_check"]
    if sc is not None:
        flag = "" if sc["same_draws"] else " **FLAG: not the same draws as K5f.**"
        lines.append(f"* {BK_ROW} step-1 mil {sc['new']:.6f} vs {K5F} {sc['control']:.6f}; same "
                     f"draws = {sc['same_draws']}.{flag}")
    low, high = constants.EDGE_TARGET_ZONE_S
    lines += ["", f"## T1: Nexar val zone AUC [{low:g}, {high:g}) s vs negative-video frames "
              "(lower = less early rise)", "", "| row | AUC | row - K5f |", "|---|---|---|"]
    for row in (K5F, *rows):
        lines.append(f"| {row} | {r['t1'][row]['auc']:.4f} | {_signed(r['t1'][row]['delta'])} |")
    lines += ["", f"## Onset AUC: span vs the normal frames within {constants.EDGE_ONSET_S:g} s "
              f"(higher = sharper edge; T2 = {DOTA}, {PRE})", "",
              "| set | side | row | videos | AUC | row - K5f |", "|---|---|---|---|---|---|"]
    for s in ONSET_SETS:
        for side in (PRE, POST):
            for row in (K5F, *rows):
                c = r["onset"][s][side][row]
                lines.append(f"| {s} | {side} | {row} | {c['videos']} | {c['auc']:.4f} | "
                             f"{_signed(c['delta'])} |")
    lines += ["", "## Macro per row (Nexar = val crops) and row - K5f, paired, 95 % cluster "
              "bootstrap", "", "| set | " + K5F + " | " + " | ".join(
                  f"{row} | {row} - {K5F} | label" for row in rows) + " |",
              "|---|---|" + "---|" * (3 * len(rows))]
    for s in ALL_SETS:
        tag = "" if s in r["decision_sets"] else " (printed)"
        cells = " | ".join(f"{_macro(r['sets'][row][s], s, row)} | {_cell(r['sets'][row][s])} | "
                           f"{r['labels'][row][s]}" for row in rows)
        lines.append(f"| {s}{tag} | {_macro(r['sets'][rows[0]][s], s, kw.K5)} | {cells} |")
    fa = r["false_alarms"]
    lines += ["", "Printed: Nexar val negatives with peak > 0.5: "
              + ", ".join(f"{row} {fa[row]:.3f}" for row in fa) + ".", "",
              "Never from this report: that edge windows reduce the early rise (one hit or a "
              "TRADE-OFF), which loss acts, \"K5f anticipates\".", ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--kim-root", type=Path, required=True, help="kim_fixed/K5f")
    parser.add_argument("--k5f-nexar-dir", type=Path, required=True,
                        help="K5f's nexar_val read with scores/ (kim_fixed/K5f/eval/nexar/K5f)")
    parser.add_argument("--edge-root", type=Path, required=True, help="kim_edge (holds the rows)")
    parser.add_argument("--rows", nargs="+", default=list(constants.KIM_EDGE_ROWS),
                        choices=constants.KIM_EDGE_ROWS, help="rows already trained and read")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
