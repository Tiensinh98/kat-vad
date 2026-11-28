"""KIM on its own CLIP-only temporal encoder (K5dfice) against K5dfirr, ``KIM_RUN.md`` §59.

Pure aggregation over saved reads, PRELIMINARY (n = 1). Contrasts (CE7), each paired per item with
``k5_whole_report``'s readers, macro on Nexar val crops and the four zero-shot sets:

* **P** (module) K5dfice - K5dfirr: decides (CE9).
* **C** (input beyond capacity) K5dfice - K5dficx: attribution, read only when P is HELD.
* **X** (the second encoder alone) K5dficx - K5dfirr: printed.
* **R** (reference) K5dfice - K5f: a COST here means "not a candidate" (starts not matched).

Labels are §34's (``k5_dvs_ignore_report.label``). Gates: the step-1 start (CE4) and the trains
check (CE5) do not raise; a row that misses either is not labelled. Guards (raise): last-epoch
``mul_mil`` above the §31 floor, one checkpoint per row. CLI::

    python -m core.tools.k5_clip_encoder_report \\
        --control-root outputs/kim_dfi_bottomk25/K5dfirr \\
        --reference-root outputs/kim_fixed/K5f \\
        --reference-nexar-dir outputs/kim_fixed/K5f/eval/nexar/K5f \\
        --root outputs/kim_dfi_clip_encoder --out-dir outputs/kim_dfi_clip_encoder/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from core import constants
from core.tools import k5_dvs_ignore_report as dvs
from core.tools import k5_edge_report as edge
from core.tools import k5_neg_report as kn
from core.tools import k5_whole_report as kw
from core.tools import zs_compare_table as zs
from core.tools.k5_bottomk_report import ALL_SETS, NEXAR, pair_sets, read_env
from core.tools.k5_lneg_cap_report import _cell, check_fixed_runs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/KIM_RUN.md §59"
CONTROL = constants.KIM_CLIP_ENC_CONTROL
NEW = constants.KIM_CLIP_ENC_ROW
CAP = constants.KIM_CLIP_ENC_CAP_ROW
K5F = constants.KIM_REFERENCE_ARM
GAIN, COST = constants.KIM_DVS_IGNORE_GAIN, constants.KIM_DVS_IGNORE_COST
CONTRASTS = {"P": (CONTROL, NEW), "C": (CAP, NEW), "X": (CONTROL, CAP), "R": (K5F, NEW)}
HELD = "HELD (s2025 may run; K5f stays)"
MODULE_COST = "COST (closed)"
NO_DIFFERENCE = "NO DIFFERENCE (closed)"
UNMATCHED = "UNMATCHED START (not decided)"
UNTRAINED = "ENCODER DID NOT TRAIN (not decided)"
CLIP_CARRIES = "the CLIP-only input carries it on {sets}"
NOT_SEPARATED = "not separated from the extra encoder"
GRAD_KEY, DRIFT_KEY = "kim_enc_grad_norm", "kim_enc_drift"


def trains_check(run_dir: Path) -> dict[str, Any]:
    """CE5 from ``metrics.jsonl``: last-epoch mean grad norm, final drift, trained or not."""
    rows = [json.loads(x) for x in (run_dir / edge.METRICS_FILENAME).read_text(
        encoding="utf-8").splitlines() if x.strip()]
    stepped = [r for r in rows if GRAD_KEY in r]
    if not stepped:
        return {"grad_norm": None, "drift": None, "trained": False}
    last_epoch = max(r["epoch"] for r in stepped)
    grads = [r[GRAD_KEY] for r in stepped if r["epoch"] == last_epoch]
    grad = sum(grads) / len(grads)
    drift = stepped[-1][DRIFT_KEY]
    trained = math.isfinite(grad) and grad > 0.0 and drift >= constants.KIM_ENC_MIN_DRIFT
    return {"grad_norm": grad, "drift": drift, "trained": trained}


def module_verdict(labels: dict[str, str]) -> str:
    """CE9 on P's labels over the decision sets."""
    picked = [labels[s] for s in dvs.DECISION_SETS]
    if COST in picked:
        return MODULE_COST
    return HELD if GAIN in picked else NO_DIFFERENCE


def attribution(p_labels: dict[str, str], c_labels: dict[str, str]) -> str | None:
    """CE9: only when P is HELD; C must gain where P gains."""
    if module_verdict(p_labels) != HELD:
        return None
    both = [s for s in dvs.DECISION_SETS if p_labels[s] == GAIN and c_labels[s] == GAIN]
    return CLIP_CARRIES.format(sets=", ".join(both)) if both else NOT_SEPARATED


def candidate(verdict: str, r_labels: dict[str, str]) -> bool:
    """CE9: HELD and no COST against K5f on a decision set."""
    return verdict == HELD and not any(r_labels[s] == COST for s in dvs.DECISION_SETS)


def decide(gates: dict[str, dict[str, bool]], labels: dict[str, dict[str, str]]
           ) -> dict[str, Any]:
    """The verdict, attribution and candidacy from the gates (CE4, CE5) and the labels."""
    if not gates[NEW]["same_draws"]:
        verdict = UNMATCHED
    elif not gates[NEW]["trained"]:
        verdict = UNTRAINED
    else:
        verdict = module_verdict(labels["P"])
    cap_ok = gates[CAP]["same_draws"] and gates[CAP]["trained"]
    attr = attribution(labels["P"], labels["C"]) if verdict == HELD and cap_ok else None
    if verdict == HELD and not cap_ok:
        attr = f"not read: {CAP} misses a gate"
    return {"verdict": verdict, "attribution": attr,
            "candidate": candidate(verdict, labels["R"])}


def run(args: argparse.Namespace) -> dict[str, Any]:
    roots = {CONTROL: args.control_root, NEW: args.root / NEW, CAP: args.root / CAP,
             K5F: args.reference_root}
    dirs = {row: roots[row] / zs.EVAL_DIR / NEXAR / row for row in (CONTROL, NEW, CAP)}
    dirs[K5F] = args.reference_nexar_dir
    run_dirs = {row: root / kw.K5 / f"s{args.seed}" for row, root in roots.items()}
    mul_mil = check_fixed_runs(run_dirs)
    views = {row: kw._view(root, args.seed) for row, root in roots.items()}

    sets = {name: pair_sets(views, dirs, control, new, args.seed)
            for name, (control, new) in CONTRASTS.items()}
    labels = {name: {s: dvs.label(cell[s]) for s in ALL_SETS} for name, cell in sets.items()}
    gates: dict[str, dict[str, Any]] = {}
    for row in (NEW, CAP):
        start = edge.start_check(run_dirs[CONTROL], run_dirs[row])
        trains = trains_check(run_dirs[row])
        gates[row] = {"same_draws": start["same_draws"], "trained": trains["trained"],
                      "start": start, "trains": trains}
    report: dict[str, Any] = {
        "rule": RULE, "seed": args.seed, "fixed_mul_mil": mul_mil, "gates": gates,
        "env": {row: read_env(run_dirs[row]) for row in (CONTROL, NEW, CAP)},
        "sets": sets, "labels": labels, "decision_sets": list(dvs.DECISION_SETS),
        **decide(gates, labels),
        "false_alarms": {row: kn.false_alarms(kn.load_video_stats(d))["share"]
                         for row, d in dirs.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = constants.KIM_CLIP_ENC_REPORT_STEM
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    LOGGER.info("verdict %s -> %s", report["verdict"], args.out_dir / f"{stem}.md")
    return report


def _gate_line(row: str, gate: dict[str, Any]) -> str:
    start, trains = gate["start"], gate["trains"]
    grad = "none" if trains["grad_norm"] is None else f"{trains['grad_norm']:.3e}"
    drift = "none" if trains["drift"] is None else f"{trains['drift']:.4f}"
    return (f"* {row}: step-1 mil {start['new']:.6f} vs {CONTROL} {start['control']:.6f}, same "
            f"draws = {start['same_draws']}; encoder last-epoch grad norm {grad}, final drift "
            f"{drift} (floor {constants.KIM_ENC_MIN_DRIFT:g}), trained = {trains['trained']}.")


TITLES = {"P": f"P (decides): {NEW} - {CONTROL}",
          "C": f"C (input beyond capacity; read only when P is HELD): {NEW} - {CAP}",
          "X": f"X (the second encoder alone, printed): {CAP} - {CONTROL}",
          "R": f"R (the project reference; a COST = not a candidate; starts not matched): "
               f"{NEW} - {K5F}"}


def render(r: dict[str, Any]) -> str:
    lines = [f"# {NEW} vs {CONTROL} (KIM on its own CLIP-only temporal encoder) -- s{r['seed']}",
             "", f"Rule: `{r['rule']}`. PRELIMINARY, n = 1; nothing is adopted, {K5F} stays the "
             "project reference. Last-epoch mul_mil: "
             + ", ".join(f"{n} {x:.4f}" for n, x in r["fixed_mul_mil"].items()) + ".", "",
             f"* **Verdict: {r['verdict']}.** Attribution (C): {r['attribution'] or 'not read'}. "
             f"Candidate for the main model: {r['candidate']}.",
             *(_gate_line(row, r["gates"][row]) for row in (NEW, CAP)),
             f"* Environment (saved beside the runs): {r['env']}."]
    if len({json.dumps(e, sort_keys=True) for e in r["env"].values()}) > 1:
        lines.append("* **FLAG: the rows were trained in different environments.**")
    for name, (control, new) in CONTRASTS.items():
        lines += ["", f"## {TITLES[name]}", "",
                  f"| set | {control} | {new} | {new} - {control} | label |",
                  "|---|---|---|---|---|"]
        for s in ALL_SETS:
            cell = r["sets"][name][s]
            tag = "" if s in r["decision_sets"] else " (printed)"
            lines.append(f"| {s}{tag} | {edge._macro(cell, s, kw.K5)} | "
                         f"{edge._macro(cell, s, new)} | {_cell(cell)} | "
                         f"{r['labels'][name][s]} |")
    fa = r["false_alarms"]
    lines += ["", "Printed: Nexar val negatives with peak > 0.5: "
              + ", ".join(f"{row} {fa[row]:.3f}" for row in fa) + ".", "",
              "Never from this report: \"KIM now matches LAS-VAD's IAM\"; \"pure CLIP is the "
              "better input for KIM\" unless C separates; \"KIM works / does not work\" from P "
              "alone; any claim about a K5f-hosted KIM; a runtime conclusion.", ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--control-root", type=Path, required=True,
                        help="K5dfirr's row dir (reused from §51 or re-trained here)")
    parser.add_argument("--reference-root", type=Path, required=True, help="kim_fixed/K5f")
    parser.add_argument("--reference-nexar-dir", type=Path, required=True,
                        help="K5f's nexar_val read with scores/")
    parser.add_argument("--root", type=Path, required=True, help="holds K5dfice and K5dficx")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
