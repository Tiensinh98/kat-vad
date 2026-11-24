"""K5fbk25 (bottom-k weight 0.25) against K5frr, the K5f of the same notebook (``KIM_RUN.md`` §50).

Pure aggregation over saved reads, PRELIMINARY (n = 1). Every deciding contrast is K5fbk25 - K5frr,
paired per item with ``k5_whole_report``'s readers; K5frr - K5f (the re-run spread) is printed.

* **A** (BQ5): per video, AUC of the span's first third against all normal rows; the interval of
  K5fbk25 - K5frr must not lie wholly below 0 on Nexar val and DADA. DoTA printed.
* **B** (BQ5): DoTA-CAP-dev onset AUC (``k5_edge_report.onset_aucs``, pre side): interval above 0.
* **Cost** (BQ6): ``k5_dvs_ignore_report.label`` on the deciding sets. **Verdict** (BQ7).

Guards (raise): last-epoch ``mul_mil`` above the §31 floor; one checkpoint per row, none shared.
The start gate (BQ2) does not raise: an unmatched start replaces the verdict. CLI::

    python -m core.tools.k5_bottomk_report --kim-root outputs/kim_fixed/K5f \\
        --k5f-nexar-dir outputs/kim_fixed/K5f/eval/nexar/K5f \\
        --root outputs/kim_bottomk25 --out-dir outputs/kim_bottomk25/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import frame_auc
from core.tools import k5_dvs_ignore_report as dvs
from core.tools import k5_edge_report as edge
from core.tools import k5_neg_report as kn
from core.tools import k5_whole_report as kw
from core.tools import k5_zs_report as zs_reads
from core.tools import zs_compare_table as zs
from core.tools.k5_lneg_cap_report import _cell, check_fixed_runs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.pre_onset_read import Curves, load_rows

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/KIM_RUN.md §50"
K5F = constants.KIM_REFERENCE_ARM
CONTROL, NEW = constants.KIM_BOTTOMK25_CONTROL, constants.KIM_BOTTOMK25_ROW
NEXAR, DOTA, DADA = kw.NEXAR, zs_reads.DOTA, zs_reads.DADA
ALL_SETS = (NEXAR, *kw.ZS_SETS)
TARGET_A_SETS = (NEXAR, DADA)
COST = constants.KIM_DVS_IGNORE_COST
HELD = "HELD (s2025 may run; K5f stays)"
TRADE_OFF = "TRADE-OFF (closed)"
CLOSED = "CLOSED (target B not met)"
UNMATCHED = "UNMATCHED START (not decided)"

Interval = dict[str, float]


def first_third_aucs(scores: Curves, labels: Curves) -> dict[str, float]:
    """Per two-class video with ``SPAN_THIRDS`` abnormal rows or more: AUC of the span's first
    part against all normal rows (BQ5)."""
    out: dict[str, float] = {}
    for v, gt in labels.items():
        positive = np.flatnonzero(gt)
        if len(positive) < constants.SPAN_THIRDS or len(positive) == len(gt):
            continue
        first = np.array_split(positive, constants.SPAN_THIRDS)[0]
        keep = gt == 0
        keep[first] = True
        truth = np.zeros(len(gt), dtype=np.int64)
        truth[first] = 1
        out[v] = frame_auc(scores[v][keep], truth[keep])
    return out


def target_a_ok(deltas: dict[str, Interval | None]) -> bool:
    """Interval not wholly below 0 on every ``TARGET_A_SETS`` set (BQ5, lenient on purpose)."""
    picked = [deltas[s] for s in TARGET_A_SETS]
    return all(d is not None and d["high"] >= 0 for d in picked)


def verdict(a_ok: bool, b_ok: bool, labels: dict[str, str]) -> str:
    """BQ7."""
    if not b_ok:
        return CLOSED
    cost = any(labels[s] == COST for s in dvs.DECISION_SETS)
    return HELD if a_ok and not cost else TRADE_OFF


def read_env(run_dir: Path) -> dict[str, str] | None:
    path = run_dir / constants.KIM_RUN_ENV_FILENAME
    if not path.exists():
        return None
    env: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return env


def pair_sets(views: dict[str, argparse.Namespace], dirs: dict[str, Path], control: str, new: str,
              seed: int) -> dict[str, Any]:
    """Both rows on the five sets; the readers key the control as ``K5``."""
    pair_views = {kw.K5: views[control], new: views[new]}
    pair_dirs = {kw.K5: dirs[control], new: dirs[new]}
    kw.check_checkpoints(pair_views, pair_dirs, seed, new=new)
    return {NEXAR: kw.nexar_set(pair_dirs, new=new),
            **{s: kw.zero_shot_set(pair_views, s, new=new) for s in kw.ZS_SETS}}


def third_reads(views: dict[str, argparse.Namespace], dirs: dict[str, Path],
                groups: dict[str, dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for set_name in edge.ONSET_SETS:
        scores, labels = load_rows(
            [(row, edge._score_path(views[row], dirs[row], set_name)) for row in views])
        aucs = {row: first_third_aucs(scores[row], labels) for row in views}
        out[set_name] = {row: {
            "videos": len(a), "auc": float(np.mean(list(a.values()))),
            "delta": (None if row == CONTROL
                      else edge.paired_ci(a, aucs[CONTROL], groups.get(set_name))),
        } for row, a in aucs.items()}
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    roots = {CONTROL: args.root / CONTROL, NEW: args.root / NEW, K5F: args.kim_root}
    run_dirs = {row: root / kw.K5 / f"s{args.seed}" for row, root in roots.items()}
    mul_mil = check_fixed_runs(run_dirs)
    views = {row: kw._view(root, args.seed) for row, root in roots.items()}
    dirs = {row: roots[row] / zs.EVAL_DIR / NEXAR / row for row in (CONTROL, NEW)}
    dirs[K5F] = args.k5f_nexar_dir

    sets = pair_sets(views, dirs, CONTROL, NEW, args.seed)
    rerun = pair_sets(views, dirs, K5F, CONTROL, args.seed)
    two = {row: views[row] for row in (CONTROL, NEW)}
    groups = {s: zs_reads.read_row(views[CONTROL], s, kw.K5)["groups"]
              for s in edge.ONSET_SETS if s != NEXAR}
    onset, t1 = edge.onset_reads(two, dirs, groups, control=CONTROL)
    thirds = third_reads(two, dirs, groups)

    labels = {s: dvs.label(sets[s]) for s in ALL_SETS}
    start = edge.start_check(run_dirs[CONTROL], run_dirs[NEW])
    a_deltas = {s: thirds[s][NEW]["delta"] for s in edge.ONSET_SETS}
    onset_b = onset[DOTA][edge.PRE][NEW]["delta"]
    a_ok = target_a_ok(a_deltas)
    b_ok = onset_b is not None and onset_b["low"] > 0
    report: dict[str, Any] = {
        "rule": RULE, "seed": args.seed, "fixed_mul_mil": mul_mil, "start_check": start,
        "start_vs_k5f": edge.start_check(run_dirs[K5F], run_dirs[CONTROL]),
        "env": {row: read_env(run_dirs[row]) for row in (CONTROL, NEW)},
        "sets": sets, "labels": labels, "rerun": rerun,
        "thirds": thirds, "onset": onset, "t1": t1,
        "targets": {"A": a_ok, "B": b_ok},
        "a_strict": {s: a_deltas[s] is not None
                     and a_deltas[s]["low"] >= constants.BOTTOMK25_STRICT_LOW
                     for s in TARGET_A_SETS},
        "verdict": verdict(a_ok, b_ok, labels) if start["same_draws"] else UNMATCHED,
        "decision_sets": list(dvs.DECISION_SETS),
        "false_alarms": {row: kn.false_alarms(kn.load_video_stats(d))["share"]
                         for row, d in dirs.items()},
    }
    stem = constants.KIM_BOTTOMK25_REPORT_STEM
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    LOGGER.info("verdict %s -> %s", report["verdict"], args.out_dir / f"{stem}.md")
    return report


def render(r: dict[str, Any]) -> str:
    sc, vs = r["start_check"], r["start_vs_k5f"]
    lines = [f"# {NEW} vs {CONTROL} (K5f re-trained in the same notebook) -- s{r['seed']}", "",
             f"Rule: `{r['rule']}`. PRELIMINARY, n = 1; nothing is adopted, {K5F} stays the "
             "project reference. Last-epoch mul_mil: "
             + ", ".join(f"{n} {x:.4f}" for n, x in r["fixed_mul_mil"].items()) + ".", "",
             f"* **Verdict: {r['verdict']}.** Target A (first third, Nexar val and DADA) "
             f"{r['targets']['A']}; target B (DoTA onset) {r['targets']['B']}.",
             f"* A, strict (lower bound >= {constants.BOTTOMK25_STRICT_LOW:g}, printed): "
             + ", ".join(f"{s} {ok}" for s, ok in r["a_strict"].items()) + ".",
             f"* Start gate: step-1 mil {sc['new']:.6f} vs {CONTROL} {sc['control']:.6f}, same "
             f"draws = {sc['same_draws']}.",
             f"* {CONTROL} vs the old {K5F}: step-1 mil {vs['new']:.6f} vs {vs['control']:.6f}, "
             f"same draws = {vs['same_draws']}.",
             f"* Environment (saved beside the runs): {r['env']}."]
    if r["env"][CONTROL] != r["env"][NEW]:
        lines.append("* **FLAG: the two rows were trained in different environments.**")
    lines += ["", "## First third of the span vs all normal rows (per video)", "",
              f"| set | videos | {CONTROL} | {NEW} | {NEW} - {CONTROL} |", "|---|---|---|---|---|"]
    for s in edge.ONSET_SETS:
        c = r["thirds"][s]
        tag = "" if s in TARGET_A_SETS else " (printed)"
        lines.append(f"| {s}{tag} | {c[NEW]['videos']} | {c[CONTROL]['auc']:.4f} | "
                     f"{c[NEW]['auc']:.4f} | {edge._signed(c[NEW]['delta'])} |")
    lines += ["", f"## Onset AUC: span vs the normal rows within {constants.EDGE_ONSET_S:g} s",
              "", f"| set | side | {CONTROL} | {NEW} | {NEW} - {CONTROL} |",
              "|---|---|---|---|---|"]
    for s in edge.ONSET_SETS:
        for side in (edge.PRE, edge.POST):
            c = r["onset"][s][side]
            lines.append(f"| {s} | {side} | {c[CONTROL]['auc']:.4f} | {c[NEW]['auc']:.4f} | "
                         f"{edge._signed(c[NEW]['delta'])} |")
    lines += ["", f"## Macro (Nexar = val crops): {NEW} - {CONTROL} decides; {CONTROL} - {K5F} is "
              "the re-run spread (printed)", "",
              f"| set | {CONTROL} | {NEW} | {NEW} - {CONTROL} | label | {CONTROL} - {K5F} |",
              "|---|---|---|---|---|---|"]
    for s in ALL_SETS:
        tag = "" if s in r["decision_sets"] else " (printed)"
        lines.append(f"| {s}{tag} | {edge._macro(r['sets'][s], s, kw.K5)} | "
                     f"{edge._macro(r['sets'][s], s, NEW)} | {_cell(r['sets'][s])} | "
                     f"{r['labels'][s]} | {_cell(r['rerun'][s])} |")
    fa = r["false_alarms"]
    lines += ["", "Printed: Nexar val negatives with peak > 0.5: "
              + ", ".join(f"{row} {fa[row]:.3f}" for row in fa) + ".", "",
              "Never from this report: \"bottom-k fixes the early rise\"; any read of "
              f"{constants.KIM_BOTTOMK_ROW} (weight 1.0) against {CONTROL}; a runtime conclusion "
              "from one re-run.", ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--kim-root", type=Path, required=True, help="kim_fixed/K5f")
    parser.add_argument("--k5f-nexar-dir", type=Path, required=True,
                        help="K5f's nexar_val read with scores/ (kim_fixed/K5f/eval/nexar/K5f)")
    parser.add_argument("--root", type=Path, required=True,
                        help="kim_bottomk25 (holds K5frr and K5fbk25)")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
