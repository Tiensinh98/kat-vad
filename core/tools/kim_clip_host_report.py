"""KIM without VideoMAE: the 2 x 2 {VideoMAE on / off} x {KIM on / off} on one host (§64).

``KIM_RUN.md`` §64. Pure aggregation over saved reads, PRELIMINARY (n = 1). Five rows per host:
L5 (KIM on the CLIP-only trunk), L5cap (its capacity control: a branch without differences),
L0 (the CLIP-only trunk), K5 / K0 (KIM on / off on the VideoMAE host, existing rows). Per set and
metric (micro first, macro second):

* **A** = L5 - L0, **B** = K5 - K0, **I** = A - B (interaction), **C** = L5 - L5cap; printed:
  L5cap - L0, K0 - L0, K5 - L5. Micro by ``kim_ablation_report``'s pooled-AUC cluster bootstrap
  (shared resamples, so I and C are paired); macro by the read-outs' bootstrap over the items
  every row scored.
* Labels §53 KA5 (GAIN / COST / NO DIFFERENCE), questions Q1-Q3 (CH8) on the deciding sets.

Guards (raise): §31 floor, config diffs CH9, one checkpoint per row and none shared, same items.
The start gates (CH3) do not raise: an unequal step-1 ``mil`` turns that pair into UNMATCHED
START and the questions that need it are not decided. CLI::

    python -m core.tools.kim_clip_host_report --host nexar \\
        --row L5 L5dfi .../kim_clip_host/L5dfi --row L5cap L5dficap ... --row L0 L0dfi ... \\
        --row K5 K5dfirr .../kim_ablation/K5dfirr --row K0 K0dfi .../kim_ablation/K0dfi \\
        --out-dir .../kim_clip_host/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.metrics import cluster_bootstrap_ci
from core.tools import k5_dvs_ignore_report as dvs
from core.tools import k5_edge_report as edge
from core.tools import k5_whole_report as kw
from core.tools import k5_zs_report as zs_reads
from core.tools import kim_ablation_report as ka
from core.tools import nexar_eval
from core.tools.k5_lneg_cap_report import check_fixed_runs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/KIM_RUN.md §64"
LABEL = "PRELIMINARY (n = 1 seed); decides nothing, K5f stays the reference"
L5, L5CAP, L0, K5, K0 = "L5", "L5cap", "L0", "K5", "K0"
ROW_KEYS = (L5, L5CAP, L0, K5, K0)
MICRO, MACRO = ka.MICRO, ka.MACRO
GAIN, COST = constants.KIM_DVS_IGNORE_GAIN, constants.KIM_DVS_IGNORE_COST
NO_DIFFERENCE = constants.KIM_DVS_IGNORE_NO_DIFFERENCE
UNMATCHED = "UNMATCHED START (not decided)"
# CH5: contrast -> signed rows
TERMS: dict[str, dict[str, int]] = {
    "A": {L5: 1, L0: -1},
    "B": {K5: 1, K0: -1},
    "I": {L5: 1, L0: -1, K5: -1, K0: 1},
    "C": {L5: 1, L5CAP: -1},
    "cap": {L5CAP: 1, L0: -1},
    "motion": {K0: 1, L0: -1},
    "motion_given_kim": {K5: 1, L5: -1},
}
TERM_TEXT = {"A": "L5 - L0 (KIM, CLIP host)", "B": "K5 - K0 (KIM, VideoMAE host)",
             "I": "I = A - B (interaction)", "C": "L5 - L5cap (differences)",
             "cap": "L5cap - L0 (branch alone)", "motion": "K0 - L0 (VideoMAE alone)",
             "motion_given_kim": "K5 - L5 (VideoMAE given KIM)"}
PRINTED_TERMS = ("cap", "motion", "motion_given_kim")
# CH3: the start pairs each contrast needs (control run, new run)
START_PAIRS = {"L5": (L0, L5), "L5cap": (L0, L5CAP), "K5": (K0, K5)}
NEEDS = {"A": ("L5",), "B": ("K5",), "I": ("L5", "K5"), "C": ("L5", "L5cap"),
         "cap": ("L5cap",), "motion": (), "motion_given_kim": ()}
Q1, Q2, Q3 = "Q1 KIM helps the CLIP host", "Q2 VideoMAE masks KIM", "Q3 what carries A"
YES, NO = "YES", "NO"
SUPPORTED, REVERSED, NO_CLAIM = "SUPPORTED", "REVERSED", "NO CLAIM"
DIFFERENCES, DIFFERENCES_COST = "DIFFERENCES", "DIFFERENCES COST"
BRANCH = "BRANCH + LOSSES"


# --------------------------------------------------------------------------- guards
def check_configs(cfgs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """CH9: L5 - L0 = the KIM flags, L5cap - L0 = Kcap's, K5 - K0 = KIM flags, K0 - L0 = motion."""
    out: dict[str, Any] = {
        "L5_vs_L0": ka.check_kim_only(cfgs[L5], cfgs[L0], "L5 vs L0"),
        "K5_vs_K0": ka.check_kim_only(cfgs[K5], cfgs[K0], "K5 vs K0"),
    }
    cap = ka.config_diff(cfgs[L0], cfgs[L5CAP])
    want_cap = dict.fromkeys(constants.KIM_CLIP_HOST_CAP_FLAGS, (False, True))
    if cap != want_cap:
        raise ValueError(f"L5cap vs L0: config diff {cap}, §64 allows exactly {want_cap}")
    out["L5cap_vs_L0"] = sorted(cap)
    motion = ka.config_diff(cfgs[L0], cfgs[K0])
    if set(motion) != set(constants.KIM_CLIP_HOST_MOTION_KEYS):
        raise ValueError(f"K0 vs L0: config diff {motion}, §64 allows exactly "
                         f"{constants.KIM_CLIP_HOST_MOTION_KEYS}")
    out["K0_vs_L0"] = {k: list(v) for k, v in motion.items()}
    return out


def check_checkpoints(rows: dict[str, tuple[Path, str]], sets: list[str], seed: int) -> None:
    """CH9: every read of a row names one checkpoint, and no two rows share one."""
    owner: dict[str, str] = {}
    zero_shot = [s for s in sets if s != ka.NEXAR]
    for key, (root, row) in rows.items():
        seen = zs_reads.k5_checkpoints(kw._view(root, seed), zero_shot)
        if ka.NEXAR in sets:
            path = root / "eval" / ka.NEXAR / row / nexar_eval.READOUT_JSON
            doc = json.loads(path.read_text(encoding="utf-8"))
            seen[ka.NEXAR] = str(doc["checkpoints"][f"s{seed}"])
        zs_reads.check_one_checkpoint(seen)
        ckpt = next(iter(seen.values()))
        if ckpt in owner:
            raise ValueError(f"{key} and {owner[ckpt]} read the same checkpoint {ckpt}")
        owner[ckpt] = key


def start_gates(run_dirs: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """CH3: step-1 ``mil`` of each new row against its control."""
    return {name: edge.start_check(run_dirs[ctrl], run_dirs[new])
            for name, (ctrl, new) in START_PAIRS.items()}


# --------------------------------------------------------------------------- contrasts
def macro_terms(items: dict[str, dict[str, float]], groups: dict[str, str]
                ) -> dict[str, dict[str, float] | None]:
    """Per-item signed sums over the items every row scored, read-outs' bootstrap."""
    common = sorted(set.intersection(*(set(items[k]) for k in ROW_KEYS)))
    if not common:
        raise ValueError("the five rows share no scored item")
    clusters: dict[str, str] = {v: str(groups.get(v, v)) for v in common}
    out: dict[str, dict[str, float] | None] = {}
    for term, signs in TERMS.items():
        values = {v: sum(sign * items[k][v] for k, sign in signs.items()) for v in common}
        out[term] = cluster_bootstrap_ci(values, clusters,
                                         constants.V2_E1_BOOTSTRAP, constants.V5_BOOTSTRAP_SEED,
                                         constants.V2_E1_CI)
    return out


def label(delta: dict[str, float] | None) -> str:
    return dvs.label({"delta": delta})


def question(labels: dict[str, str], deciding: tuple[str, ...], positive: str, negative: str,
             otherwise: str) -> str:
    """CH8: ``positive`` iff GAIN on >= 1 deciding set and COST on none; ``negative`` mirrors."""
    gain = any(labels[s] == GAIN for s in deciding)
    cost = any(labels[s] == COST for s in deciding)
    if gain and not cost:
        return positive
    if cost and not gain:
        return negative
    return otherwise


def verdicts(labels: dict[str, dict[str, dict[str, str]]], deciding: tuple[str, ...],
             matched: dict[str, bool]) -> dict[str, dict[str, str]]:
    """Q1-Q3 per metric; a question whose start pairs did not match is not decided."""
    def needs(term: str) -> bool:
        return all(matched[p] for p in NEEDS[term])

    out: dict[str, dict[str, str]] = {}
    for metric in (MICRO, MACRO):
        lab = labels[metric]
        out[metric] = {
            Q1: question(lab["A"], deciding, YES, NO, NO) if needs("A") else UNMATCHED,
            Q2: (question(lab["I"], deciding, SUPPORTED, REVERSED, NO_CLAIM)
                 if needs("I") else UNMATCHED),
            Q3: (question(lab["C"], deciding, DIFFERENCES, DIFFERENCES_COST, BRANCH)
                 if needs("C") else UNMATCHED),
        }
    return out


# --------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = {key: (Path(root), name) for key, name, root in args.row}
    if set(rows) != set(ROW_KEYS):
        raise ValueError(f"need exactly the rows {ROW_KEYS}, got {sorted(rows)}")
    deciding, labelled = constants.KIM_CLIP_HOST_SETS[args.host]
    sets = [*deciding, *labelled]
    run_dirs = {k: root / kw.K5 / f"s{args.seed}" for k, (root, _) in rows.items()}
    mul_mil = check_fixed_runs(run_dirs)
    cfgs = {k: ka._load_config(d) for k, d in run_dirs.items()}
    diffs = check_configs(cfgs)
    check_checkpoints(rows, sets, args.seed)
    starts = start_gates(run_dirs)
    matched = {k: bool(v["same_draws"]) for k, v in starts.items()}
    per_set: dict[str, Any] = {}
    for set_name in sets:
        data = ka.read_set(set_name, rows, args.seed)
        ka.check_items(data["items"], set_name)
        per_set[set_name] = {
            "cells": data["cells"], "videos": len(data["labels"]),
            "clip_oracle_micro": data["oracle"],
            MICRO: ka.micro_contrasts(data["curves"], data["labels"], data["groups"], TERMS,
                                      args.draws, constants.V5_BOOTSTRAP_SEED),
            MACRO: macro_terms(data["items"], data["item_groups"]),
        }
        LOGGER.info("%s: %d rows, %d draws", set_name, len(rows), args.draws)
    labels = {m: {t: {s: label(per_set[s][m][t]) for s in sets} for t in TERMS}
              for m in (MICRO, MACRO)}
    report: dict[str, Any] = {
        "rule": RULE, "label": LABEL, "host": args.host, "seed": args.seed, "draws": args.draws,
        "rows": {k: {"name": name, "root": str(root),
                     "env": ka.read_env(run_dirs[k])} for k, (root, name) in rows.items()},
        "deciding_sets": list(deciding), "labelled_sets": list(labelled),
        "fixed_mul_mil": mul_mil, "config_diffs": diffs, "starts": starts,
        "sets": per_set, "labels": labels, "verdicts": verdicts(labels, deciding, matched),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{constants.KIM_CLIP_HOST_REPORT_STEM}_{args.host}"
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    return report


# --------------------------------------------------------------------------- render
def _signed(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def render(r: dict[str, Any]) -> str:
    names = {k: v["name"] for k, v in r["rows"].items()}
    deciding, labelled = r["deciding_sets"], r["labelled_sets"]
    sets = [*deciding, *labelled]
    lines = [f"# §64 KIM without VideoMAE — host {r['host']}, s{r['seed']} — **{r['label']}**", "",
             "Rows: " + ", ".join(f"{k} = **{n}**" for k, n in names.items()) + ".",
             f"Deciding sets: {', '.join(deciding)}"
             + (f"; labelled only: {', '.join(labelled)}" if labelled else "") + ".", "",
             "## Start gates (CH3: step-1 `mil`)", "", "| pair | control | new | status |",
             "|---|---|---|---|"]
    for name, (ctrl, new) in START_PAIRS.items():
        s = r["starts"][name]
        lines.append(f"| {names[new]} vs {names[ctrl]} | {s['control']:.6f} | {s['new']:.6f} | "
                     f"{'matched' if s['same_draws'] else UNMATCHED} |")
    lines += ["", "## Verdicts (CH8)", "", "| question | micro | macro |", "|---|---|---|"]
    for q in (Q1, Q2, Q3):
        lines.append(f"| {q} | {r['verdicts'][MICRO][q]} | {r['verdicts'][MACRO][q]} |")
    lines += ["", "## Cells (micro / macro)", "",
              "| row | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
    for k in ROW_KEYS:
        cells = [f"{r['sets'][s]['cells'][k][MICRO]:.4f} / {r['sets'][s]['cells'][k][MACRO]:.4f}"
                 for s in sets]
        lines.append(f"| {k} = {names[k]} | " + " | ".join(cells) + " |")
    for metric in (MICRO, MACRO):
        lines += ["", f"## Contrasts, {metric} (labels §53 KA5; printed rows never decide)", "",
                  "| contrast | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
        for term in TERMS:
            tag = " *(printed)*" if term in PRINTED_TERMS else ""
            cells = [f"{_signed(r['sets'][s][metric][term])} {r['labels'][metric][term][s]}"
                     for s in sets]
            lines.append(f"| {TERM_TEXT[term]}{tag} | " + " | ".join(cells) + " |")
    lines += ["", "Never: \"KIM models motion\"; \"VideoMAE hides KIM's motion signal\" "
              "unless Q2 is SUPPORTED and Q3 is DIFFERENCES on the same host and metric; "
              "a sentence from a labelled-only set; a seed claim."]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--host", required=True, choices=sorted(constants.KIM_CLIP_HOST_SETS))
    parser.add_argument("--row", nargs=3, action="append", required=True,
                        metavar=("KEY", "NAME", "ROOT"),
                        help=f"KEY in {ROW_KEYS}; ROOT holds K5/s<seed>/ and eval/")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--draws", type=int, default=constants.BASEMICRO_BOOTSTRAP_DRAWS)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(build_arg_parser().parse_args(argv))
    LOGGER.info("verdicts: %s", report["verdicts"])


if __name__ == "__main__":
    main()
