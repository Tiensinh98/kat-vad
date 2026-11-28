"""Class prototypes on ``v^t`` in the main score (K0dfiP / L / PL / PLb) vs K0dfi (§61).

Pure aggregation over saved reads, PRELIMINARY (n = 1). Contrasts (PR9), each paired per item
with ``k5_whole_report``'s readers, macro on Nexar val crops, DoTA-CAP-dev, DADA and DAD (no TAD):

* **D** K0dfiPL - control: decides (PR11) on Nexar val, DoTA-CAP-dev, DADA (PR10).
* **P**, **L** each row - control; **PL_P** K0dfiPL - K0dfiP; **b** K0dfiPLb - K0dfiPL: printed.
* **R** K0dfiPL - K5f: a COST on a deciding set = not a candidate.
* **KIM** K0dfiPL - K5dfi: the prototype term against KIM on the same host, printed.

Labels are §34's (``k5_dvs_ignore_report.label``); DAD is labelled and never decides. Gates: the
step-1 start (PR7) does not raise, a row that misses it is not labelled; a row whose prototypes
never both existed raises. Guards (raise): last-epoch ``mul_mil`` above the §31 floor, one
checkpoint per row. CLI::

    python -m core.tools.k0dfi_proto_report --control-root outputs/kim_ablation/K0dfi \\
        --root outputs/kim_dfi_proto --reference-root outputs/kim_fixed/K5f \\
        --reference-nexar-dir outputs/kim_fixed/K5f/eval/nexar/K5f \\
        --kim-root outputs/kim_dvs_ignore/K5dfi \\
        --kim-nexar-dir outputs/kim_dvs_ignore/K5dfi/eval/nexar/K5dfi --out-dir <REPORTS>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.tools import k5_dvs_ignore_report as dvs
from core.tools import k5_edge_report as edge
from core.tools import k5_neg_report as kn
from core.tools import k5_whole_report as kw
from core.tools import k5_zs_report as zs
from core.tools import nexar_eval
from core.tools.k5_bottomk_report import read_env
from core.tools.k5_lneg_cap_report import _cell, check_fixed_runs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/KIM_RUN.md §61"
NEXAR = kw.NEXAR
ZS_SETS = (zs.DOTA, zs.DADA, zs.DAD)  # no TAD (user, §61)
SETS = (NEXAR, *ZS_SETS)
DECISION_SETS = constants.PROTO_DECISION_SETS
ROWS = tuple(constants.PROTO_ROWS)
P_ROW, L_ROW, PL_ROW, PLB_ROW = ROWS
CONTROL = "control"  # K0dfi or K0dfirr: named in the report from --control-name
K5F, K5DFI = constants.KIM_REFERENCE_ARM, constants.KIM_DVS_IGNORE_ROW
CONTRASTS = {"D": (CONTROL, PL_ROW), "P": (CONTROL, P_ROW), "L": (CONTROL, L_ROW),
             "PL_P": (P_ROW, PL_ROW), "b": (PL_ROW, PLB_ROW), "R": (K5F, PL_ROW),
             "KIM": (K5DFI, PL_ROW)}
# the contrasts whose two sides share a start (PR7); the others are printed as they are
GATED = {"D": (PL_ROW,), "P": (P_ROW,), "L": (L_ROW,), "PL_P": (P_ROW, PL_ROW),
         "b": (PL_ROW, PLB_ROW)}
GAIN, COST = constants.KIM_DVS_IGNORE_GAIN, constants.KIM_DVS_IGNORE_COST
HELD = "HELD (s2025 may run; K5f stays)"
D_COST = "COST (closed)"
NO_DIFFERENCE = "NO DIFFERENCE (closed)"
UNMATCHED = "UNMATCHED START (not decided)"
NOT_LABELLED = "not labelled (start)"


def proto_trace(run_dir: Path) -> dict[str, Any]:
    """PR7: the last step's ``proto_*`` keys; raises if the prototypes never both existed."""
    rows = [json.loads(x) for x in (run_dir / edge.METRICS_FILENAME).read_text(
        encoding="utf-8").splitlines() if x.strip()]
    last = rows[-1]
    if last.get("proto_ready") != 1.0:
        raise ValueError(f"{run_dir}: proto_ready {last.get('proto_ready')} on the last step: "
                         "the prototypes never both existed, the row is its host")
    return {k: last.get(k) for k in constants.PROTO_LOG_KEYS}


def verdict(labels: dict[str, str], same_start: bool) -> str:
    """PR11 on D's labels over the deciding sets."""
    if not same_start:
        return UNMATCHED
    picked = [labels[s] for s in DECISION_SETS]
    if COST in picked:
        return D_COST
    return HELD if GAIN in picked else NO_DIFFERENCE


def candidate(decision: str, r_labels: dict[str, str]) -> bool:
    """PR11: HELD and no COST against K5f on a deciding set."""
    return decision == HELD and not any(r_labels[s] == COST for s in DECISION_SETS)


def check_checkpoints(views: dict[str, argparse.Namespace], dirs: dict[str, Path],
                      seed: int, new: str) -> None:
    """One checkpoint per row across its four reads; the two rows use different ones."""
    found: dict[str, set[str]] = {}
    for row, view in views.items():
        seen = zs.k5_checkpoints(view, list(ZS_SETS))
        readout = json.loads((dirs[row] / nexar_eval.READOUT_JSON).read_text(encoding="utf-8"))
        seen[NEXAR] = str(readout["checkpoints"][f"s{seed}"])
        zs.check_one_checkpoint(seen)
        found[row] = set(seen.values())
    if found[kw.K5] & found[new]:
        raise ValueError(f"{new} reads share a checkpoint with its control")


def pair(views: dict[str, argparse.Namespace], dirs: dict[str, Path], control: str, new: str,
         seed: int) -> dict[str, Any]:
    """Both rows on the four sets; the readers key the control as ``K5``."""
    pair_views = {kw.K5: views[control], new: views[new]}
    pair_dirs = {kw.K5: dirs[control], new: dirs[new]}
    check_checkpoints(pair_views, pair_dirs, seed, new)
    return {NEXAR: kw.nexar_set(pair_dirs, new=new),
            **{s: kw.zero_shot_set(pair_views, s, new=new) for s in ZS_SETS}}


def run(args: argparse.Namespace) -> dict[str, Any]:
    roots = {CONTROL: args.control_root, **{row: args.root / row for row in ROWS},
             K5F: args.reference_root, K5DFI: args.kim_root}
    dirs = {CONTROL: args.control_root / zs.EVAL_DIR / NEXAR / args.control_name,
            **{row: args.root / row / zs.EVAL_DIR / NEXAR / row for row in ROWS},
            K5F: args.reference_nexar_dir, K5DFI: args.kim_nexar_dir}
    run_dirs = {row: root / kw.K5 / f"s{args.seed}" for row, root in roots.items()}
    mul_mil = check_fixed_runs(run_dirs)
    views = {row: kw._view(root, args.seed) for row, root in roots.items()}
    starts = {row: edge.start_check(run_dirs[CONTROL], run_dirs[row]) for row in ROWS}
    traces = {row: proto_trace(run_dirs[row]) for row in ROWS}

    sets = {name: pair(views, dirs, control, new, args.seed)
            for name, (control, new) in CONTRASTS.items()}
    labels: dict[str, dict[str, str]] = {}
    for name, cell in sets.items():
        matched = all(starts[row]["same_draws"] for row in GATED.get(name, ()))
        labels[name] = {s: dvs.label(cell[s]) if matched else NOT_LABELLED for s in SETS}
    decision = verdict(labels["D"], starts[PL_ROW]["same_draws"])
    report: dict[str, Any] = {
        "rule": RULE, "seed": args.seed, "control": args.control_name,
        "fixed_mul_mil": mul_mil, "starts": starts, "proto": traces,
        "env": {row: read_env(run_dirs[row]) for row in (CONTROL, *ROWS)},
        "sets": sets, "labels": labels, "decision_sets": list(DECISION_SETS),
        "verdict": decision, "candidate": candidate(decision, labels["R"]),
        "false_alarms": {row: kn.false_alarms(kn.load_video_stats(d))["share"]
                         for row, d in dirs.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = constants.PROTO_REPORT_STEM
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    LOGGER.info("verdict %s -> %s", decision, args.out_dir / f"{stem}.md")
    return report


TITLES = {"D": "D (decides): {pl} - {c}", "P": "P (score term alone): {p} - {c}",
          "L": "L (loss alone): {l} - {c}", "PL_P": "PL - P (the loss on top of the term)",
          "b": "b (bottom-k in mu0): {plb} - {pl}",
          "R": "R (candidacy; a COST = not a candidate; starts not matched): {pl} - K5f",
          "KIM": "KIM (printed; same host, KIM on; envs may differ): {pl} - K5dfi"}


def _name(row: str, control: str) -> str:
    return control if row == CONTROL else row


def render(r: dict[str, Any]) -> str:
    c = r["control"]
    fmt = {"c": c, "p": P_ROW, "l": L_ROW, "pl": PL_ROW, "plb": PLB_ROW}
    lines = [f"# Class prototypes in the main score (§61) vs {c} -- s{r['seed']}", "",
             f"Rule: `{r['rule']}`. PRELIMINARY, n = 1; nothing is adopted, {K5F} stays the "
             "project reference. Last-epoch mul_mil: "
             + ", ".join(f"{_name(n, c)} {x:.4f}" for n, x in r["fixed_mul_mil"].items()) + ".",
             "", f"* **Verdict (D): {r['verdict']}.** Candidate for the main model: "
             f"{r['candidate']}. Deciding sets: {', '.join(r['decision_sets'])}; DAD labelled, "
             "never decides (read beside `dad_position_diag`)."]
    for row in ROWS:
        s, p = r["starts"][row], r["proto"][row]
        lam = "n/a" if p.get("proto_lambda") is None else f"{p['proto_lambda']:+.4f}"
        lines.append(f"* {row}: step-1 mil {s['new']:.6f} vs {c} {s['control']:.6f}, same "
                     f"draws = {s['same_draws']}; last step λ {lam}, cos(μ0, μ1) "
                     f"{p['proto_cos']:+.3f}, anchor share {p['proto_anchor_share']:.3f}.")
    env = {_name(k, c): v for k, v in r["env"].items()}
    lines.append(f"* Environment (saved beside the runs): {env}.")
    if len({json.dumps(e, sort_keys=True) for e in env.values()}) > 1:
        lines.append("* **FLAG: the rows were trained in different environments.**")
    for name, (control, new) in CONTRASTS.items():
        lines += ["", f"## {TITLES[name].format(**fmt)}", "",
                  f"| set | {_name(control, c)} | {new} | {new} - {_name(control, c)} | label |",
                  "|---|---|---|---|---|"]
        for s in SETS:
            cell = r["sets"][name][s]
            tag = "" if s in r["decision_sets"] else " (never decides)"
            lines.append(f"| {s}{tag} | {edge._macro(cell, s, kw.K5)} | "
                         f"{edge._macro(cell, s, new)} | {_cell(cell)} | "
                         f"{r['labels'][name][s]} |")
    fa = r["false_alarms"]
    lines += ["", "Printed: Nexar val negatives with peak > 0.5: "
              + ", ".join(f"{_name(row, c)} {fa[row]:.3f}" for row in fa) + ".", "",
              "Never from this report: \"prototypes make the model confident\"; \"the prototype "
              "term replaces KIM\" from KIM alone; a DAD macro difference without the placebo; "
              "\"μ1 helps / hurts\" (no row has μ0 without μ1).", ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--control-root", type=Path, required=True,
                        help="K0dfi's row dir (kim_ablation/K0dfi) or K0dfirr's")
    parser.add_argument("--control-name", default=constants.PROTO_HOST_ROW,
                        help="the control's row name (its nexar read folder)")
    parser.add_argument("--root", type=Path, required=True, help="holds the four §61 rows")
    parser.add_argument("--reference-root", type=Path, required=True, help="kim_fixed/K5f")
    parser.add_argument("--reference-nexar-dir", type=Path, required=True)
    parser.add_argument("--kim-root", type=Path, required=True, help="kim_dvs_ignore/K5dfi")
    parser.add_argument("--kim-nexar-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
