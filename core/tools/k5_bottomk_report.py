"""Bottom-k weight 0.25 against its own control, re-trained in the same notebook.

``KIM_RUN.md`` §50 (``--spec k5f``: K5fbk25 - K5frr) and §51 (``--spec k5dfi``: K5dfibk25 -
K5dfirr). Pure aggregation over saved reads, PRELIMINARY (n = 1). Every deciding contrast is
new - control, paired per item with ``k5_whole_report``'s readers; control - the old run (the
re-run) is printed.

* **A** (BQ5): per video, AUC of the span's first third against all normal rows; the interval of
  new - control must not lie wholly below 0 on Nexar val and DADA. DoTA printed.
* **B** (BQ5): DoTA-CAP-dev onset AUC (``k5_edge_report.onset_aucs``, pre side): interval above 0.
* **Cost** (BQ6): ``k5_dvs_ignore_report.label`` on the deciding sets. **Verdict** (BQ7). §51 adds
  the cost of new - K5f (BD6): a COST against either control fails it.

Guards (raise): last-epoch ``mul_mil`` above the §31 floor; one checkpoint per row, none shared.
The start gate (BQ2) does not raise: an unmatched start replaces the verdict. CLI::

    python -m core.tools.k5_bottomk_report --kim-root outputs/kim_fixed/K5f \\
        --k5f-nexar-dir outputs/kim_fixed/K5f/eval/nexar/K5f \\
        --root outputs/kim_bottomk25 --out-dir outputs/kim_bottomk25/REPORTS
    python -m core.tools.k5_bottomk_report --spec k5dfi --kim-root outputs/kim_dvs_ignore/K5dfi \\
        --k5f-nexar-dir outputs/kim_dvs_ignore/K5dfi/eval/nexar/K5dfi \\
        --reference-root outputs/kim_fixed/K5f \\
        --reference-nexar-dir outputs/kim_fixed/K5f/eval/nexar/K5f \\
        --root outputs/kim_dfi_bottomk25 --out-dir outputs/kim_dfi_bottomk25/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
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

K5F = constants.KIM_REFERENCE_ARM


@dataclass(frozen=True)
class Spec:
    """One bottom-k arm: ``new`` - ``control`` decides, ``old`` is the run ``control`` repeats.

    ``reference`` (§51): a further row whose COST also fails the arm; ``None`` = none (§50).
    """

    rule: str
    control: str
    new: str
    old: str
    stem: str
    reference: str | None
    title: str


SPECS = {
    "k5f": Spec("core/docs/final/KIM_RUN.md §50", constants.KIM_BOTTOMK25_CONTROL,
                constants.KIM_BOTTOMK25_ROW, K5F, constants.KIM_BOTTOMK25_REPORT_STEM, None,
                "K5f re-trained in the same notebook"),
    "k5dfi": Spec("core/docs/final/KIM_RUN.md §51", constants.KIM_DFI_BOTTOMK25_CONTROL,
                  constants.KIM_DFI_BOTTOMK25_ROW, constants.KIM_DVS_IGNORE_ROW,
                  constants.KIM_DFI_BOTTOMK25_REPORT_STEM, K5F,
                  "K5dfi re-trained in the same notebook"),
}
SPEC50 = SPECS["k5f"]
RULE, CONTROL, NEW = SPEC50.rule, SPEC50.control, SPEC50.new
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
                groups: dict[str, dict[str, str]], control: str = CONTROL) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for set_name in edge.ONSET_SETS:
        scores, labels = load_rows(
            [(row, edge._score_path(views[row], dirs[row], set_name)) for row in views])
        aucs = {row: first_third_aucs(scores[row], labels) for row in views}
        out[set_name] = {row: {
            "videos": len(a), "auc": float(np.mean(list(a.values()))),
            "delta": (None if row == control
                      else edge.paired_ci(a, aucs[control], groups.get(set_name))),
        } for row, a in aucs.items()}
    return out


def merge_cost(labels: dict[str, str], reference: dict[str, str] | None) -> dict[str, str]:
    """BD6: a set is a COST when either contrast (vs the control, vs the reference) is one."""
    if reference is None:
        return labels
    return {s: COST if COST in (labels[s], reference[s]) else labels[s] for s in labels}


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec = SPECS[args.spec]
    roots = {spec.control: args.root / spec.control, spec.new: args.root / spec.new,
             spec.old: args.kim_root}
    dirs = {row: roots[row] / zs.EVAL_DIR / NEXAR / row for row in (spec.control, spec.new)}
    dirs[spec.old] = args.k5f_nexar_dir
    if spec.reference is not None:
        if args.reference_root is None or args.reference_nexar_dir is None:
            raise ValueError(f"--spec {args.spec} needs --reference-root and --reference-nexar-dir")
        roots[spec.reference] = args.reference_root
        dirs[spec.reference] = args.reference_nexar_dir
    run_dirs = {row: root / kw.K5 / f"s{args.seed}" for row, root in roots.items()}
    mul_mil = check_fixed_runs(run_dirs)
    views = {row: kw._view(root, args.seed) for row, root in roots.items()}

    sets = pair_sets(views, dirs, spec.control, spec.new, args.seed)
    rerun = pair_sets(views, dirs, spec.old, spec.control, args.seed)
    ref_sets = (pair_sets(views, dirs, spec.reference, spec.new, args.seed)
                if spec.reference is not None else None)
    two = {row: views[row] for row in (spec.control, spec.new)}
    groups = {s: zs_reads.read_row(views[spec.control], s, kw.K5)["groups"]
              for s in edge.ONSET_SETS if s != NEXAR}
    onset, t1 = edge.onset_reads(two, dirs, groups, control=spec.control)
    thirds = third_reads(two, dirs, groups, spec.control)

    labels = {s: dvs.label(sets[s]) for s in ALL_SETS}
    ref_labels = None if ref_sets is None else {s: dvs.label(ref_sets[s]) for s in ALL_SETS}
    start = edge.start_check(run_dirs[spec.control], run_dirs[spec.new])
    a_deltas = {s: thirds[s][spec.new]["delta"] for s in edge.ONSET_SETS}
    onset_b = onset[DOTA][edge.PRE][spec.new]["delta"]
    a_ok = target_a_ok(a_deltas)
    b_ok = onset_b is not None and onset_b["low"] > 0
    report: dict[str, Any] = {
        "rule": spec.rule, "spec": args.spec, "seed": args.seed, "fixed_mul_mil": mul_mil,
        "start_check": start,
        "start_vs_k5f": edge.start_check(run_dirs[spec.old], run_dirs[spec.control]),
        "env": {row: read_env(run_dirs[row]) for row in (spec.control, spec.new)},
        "sets": sets, "labels": labels, "rerun": rerun,
        "reference_sets": ref_sets, "reference_labels": ref_labels,
        "thirds": thirds, "onset": onset, "t1": t1,
        "targets": {"A": a_ok, "B": b_ok},
        "a_strict": {s: a_deltas[s] is not None
                     and a_deltas[s]["low"] >= constants.BOTTOMK25_STRICT_LOW
                     for s in TARGET_A_SETS},
        "verdict": (verdict(a_ok, b_ok, merge_cost(labels, ref_labels))
                    if start["same_draws"] else UNMATCHED),
        "decision_sets": list(dvs.DECISION_SETS),
        "false_alarms": {row: kn.false_alarms(kn.load_video_stats(d))["share"]
                         for row, d in dirs.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{spec.stem}.json", report)
    write_text_atomic(args.out_dir / f"{spec.stem}.md", render(report))
    LOGGER.info("verdict %s -> %s", report["verdict"], args.out_dir / f"{spec.stem}.md")
    return report


def render(r: dict[str, Any]) -> str:
    spec = SPECS[r["spec"]]
    control, new, old = spec.control, spec.new, spec.old
    sc, vs = r["start_check"], r["start_vs_k5f"]
    lines = [f"# {new} vs {control} ({spec.title}) -- s{r['seed']}", "",
             f"Rule: `{r['rule']}`. PRELIMINARY, n = 1; nothing is adopted, {K5F} stays the "
             "project reference. Last-epoch mul_mil: "
             + ", ".join(f"{n} {x:.4f}" for n, x in r["fixed_mul_mil"].items()) + ".", "",
             f"* **Verdict: {r['verdict']}.** Target A (first third, Nexar val and DADA) "
             f"{r['targets']['A']}; target B (DoTA onset) {r['targets']['B']}.",
             f"* A, strict (lower bound >= {constants.BOTTOMK25_STRICT_LOW:g}, printed): "
             + ", ".join(f"{s} {ok}" for s, ok in r["a_strict"].items()) + ".",
             f"* Start gate: step-1 mil {sc['new']:.6f} vs {control} {sc['control']:.6f}, same "
             f"draws = {sc['same_draws']}.",
             f"* {control} vs the old {old}: step-1 mil {vs['new']:.6f} vs {vs['control']:.6f}, "
             f"same draws = {vs['same_draws']}.",
             f"* Environment (saved beside the runs): {r['env']}."]
    if r["env"][control] != r["env"][new]:
        lines.append("* **FLAG: the two rows were trained in different environments.**")
    lines += ["", "## First third of the span vs all normal rows (per video)", "",
              f"| set | videos | {control} | {new} | {new} - {control} |", "|---|---|---|---|---|"]
    for s in edge.ONSET_SETS:
        c = r["thirds"][s]
        tag = "" if s in TARGET_A_SETS else " (printed)"
        lines.append(f"| {s}{tag} | {c[new]['videos']} | {c[control]['auc']:.4f} | "
                     f"{c[new]['auc']:.4f} | {edge._signed(c[new]['delta'])} |")
    lines += ["", f"## Onset AUC: span vs the normal rows within {constants.EDGE_ONSET_S:g} s",
              "", f"| set | side | {control} | {new} | {new} - {control} |",
              "|---|---|---|---|---|"]
    for s in edge.ONSET_SETS:
        for side in (edge.PRE, edge.POST):
            c = r["onset"][s][side]
            lines.append(f"| {s} | {side} | {c[control]['auc']:.4f} | {c[new]['auc']:.4f} | "
                         f"{edge._signed(c[new]['delta'])} |")
    lines += ["", f"## Macro (Nexar = val crops): {new} - {control} decides; {control} - {old} is "
              "the re-run spread (printed)", "",
              f"| set | {control} | {new} | {new} - {control} | label | {control} - {old} |",
              "|---|---|---|---|---|---|"]
    for s in ALL_SETS:
        tag = "" if s in r["decision_sets"] else " (printed)"
        lines.append(f"| {s}{tag} | {edge._macro(r['sets'][s], s, kw.K5)} | "
                     f"{edge._macro(r['sets'][s], s, new)} | {_cell(r['sets'][s])} | "
                     f"{r['labels'][s]} | {_cell(r['rerun'][s])} |")
    if r["reference_sets"] is not None:
        ref = spec.reference
        lines += ["", f"## Macro against {ref} (the project reference): a COST here also fails "
                  "(BD6); starts not matched (another recipe)", "",
                  f"| set | {ref} | {new} | {new} - {ref} | label |", "|---|---|---|---|---|"]
        for s in ALL_SETS:
            tag = "" if s in r["decision_sets"] else " (printed)"
            cell = r["reference_sets"][s]
            lines.append(f"| {s}{tag} | {edge._macro(cell, s, kw.K5)} | "
                         f"{edge._macro(cell, s, new)} | {_cell(cell)} | "
                         f"{r['reference_labels'][s]} |")
    fa = r["false_alarms"]
    lines += ["", "Printed: Nexar val negatives with peak > 0.5: "
              + ", ".join(f"{row} {fa[row]:.3f}" for row in fa) + ".", "",
              "Never from this report: \"bottom-k fixes the early rise\"; any read of "
              f"{constants.KIM_BOTTOMK_ROW} (weight 1.0) against {control}; a runtime conclusion "
              "from one re-run.", ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--spec", choices=sorted(SPECS), default="k5f", help="§50 or §51")
    parser.add_argument("--kim-root", type=Path, required=True,
                        help="the old run the control repeats (k5f: kim_fixed/K5f)")
    parser.add_argument("--k5f-nexar-dir", type=Path, required=True,
                        help="that run's nexar_val read with scores/")
    parser.add_argument("--reference-root", type=Path, help="§51: kim_fixed/K5f")
    parser.add_argument("--reference-nexar-dir", type=Path, help="§51: K5f's nexar_val read")
    parser.add_argument("--root", type=Path, required=True,
                        help="holds the control and the new row")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
