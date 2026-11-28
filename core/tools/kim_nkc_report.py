"""Does ``kim_nkc`` contribute? L5dfi vs L5dfin0 on the CLIP-only Nexar host (``KIM_RUN.md`` §66).

Pure aggregation over saved reads, PRELIMINARY (n = 1). Two rows: the control (L5dfi, or L5dfirr
re-trained on the session's GPU, NK3) and L5dfin0 = the control's command + ``kim.nkc_weight=0``
(NK2). Per set and metric: **N** = control - L5dfin0 (what ``kim_nkc`` at its weight adds), micro by
``kim_ablation_report``'s pooled-AUC cluster bootstrap, macro by the read-outs' bootstrap over the
items both rows scored; labels §53 KA5; verdict NK7 on the deciding sets.

Guards (raise, NK4): §31 floor, config diff exactly ``{kim.nkc_weight: w -> 0}`` with ``w > 0``,
step-1 ``mil`` equal, one checkpoint per row and none shared, same items. Printed: the largest
|Δ total| before the first logged ``kim_nkc``. CLI::

    python -m core.tools.kim_nkc_report --control L5dfi .../kim_clip_host/L5dfi \\
        --row L5dfin0 .../kim_nkc/L5dfin0 --out-dir .../kim_nkc/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.metrics import cluster_bootstrap_ci
from core.tools import k5_edge_report as edge
from core.tools import k5_whole_report as kw
from core.tools import kim_ablation_report as ka
from core.tools import kim_clip_host_report as ch
from core.tools.k5_bottomk_report import read_env
from core.tools.k5_lneg_cap_report import check_fixed_runs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/KIM_RUN.md §66"
LABEL = "PRELIMINARY (n = 1 seed); decides nothing, K5f stays the reference"
CTRL, NEW = "control", "nkc0"
TERMS = {"N": {CTRL: 1, NEW: -1}}
MICRO, MACRO = ka.MICRO, ka.MACRO
GAIN, COST = ch.GAIN, ch.COST
METRICS_FILENAME = "metrics.jsonl"


# --------------------------------------------------------------------------- guards
def check_config(ctrl: dict[str, Any], new: dict[str, Any]) -> dict[str, list[float]]:
    """NK4: the row differs from its control by ``kim.nkc_weight`` alone, from > 0 to 0."""
    key = constants.KIM_NKC_WEIGHT_KEY
    diff = ka.config_diff(ctrl, new)
    if set(diff) != {key}:
        raise ValueError(f"config diff {sorted(diff)}, §66 NK4 allows exactly {{{key}}}")
    was, now = diff[key]
    if not (float(was) > 0 and float(now) == constants.KIM_NKC_OFF_WEIGHT):
        raise ValueError(f"{key}: {was} -> {now}, §66 NK4 needs > 0 -> "
                         f"{constants.KIM_NKC_OFF_WEIGHT}")
    if not ctrl.get("kim.enabled") or not ctrl.get("kim.prototypes"):
        raise ValueError("the control has no KIM with prototypes: kim_nkc never ran")
    return {key: [float(was), float(now)]}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def pre_nkc_drift(ctrl_run: Path, new_run: Path) -> dict[str, Any]:
    """Printed: largest |Δ total| over the steps before the control's first ``kim_nkc``."""
    a = {r["global_step"]: r for r in _rows(ctrl_run / METRICS_FILENAME)}
    b = {r["global_step"]: r for r in _rows(new_run / METRICS_FILENAME)}
    first = min((s for s, r in a.items() if "kim_nkc" in r), default=None)
    steps = sorted(s for s in set(a) & set(b) if first is None or s < first)
    drift = max((abs(a[s]["total"] - b[s]["total"]) for s in steps), default=0.0)
    return {"first_nkc_step": first, "steps_compared": len(steps), "max_abs_total": drift}


# --------------------------------------------------------------------------- contrasts
def macro_contrast(items: dict[str, dict[str, float]], groups: dict[str, str]
                   ) -> dict[str, float] | None:
    common = sorted(set(items[CTRL]) & set(items[NEW]))
    if not common:
        raise ValueError("the two rows share no scored item")
    values = {v: items[CTRL][v] - items[NEW][v] for v in common}
    return cluster_bootstrap_ci(values, {v: str(groups.get(v, v)) for v in common},
                                constants.V2_E1_BOOTSTRAP, constants.V5_BOOTSTRAP_SEED,
                                constants.V2_E1_CI)


def verdict(labels: dict[str, str], deciding: tuple[str, ...]) -> str:
    """NK7 on the deciding sets."""
    gain = any(labels[s] == GAIN for s in deciding)
    cost = any(labels[s] == COST for s in deciding)
    if gain and not cost:
        return constants.KIM_NKC_HELPS
    if cost and not gain:
        return constants.KIM_NKC_COSTS
    if not gain and not cost:
        return constants.KIM_NKC_NO_DIFFERENCE
    return constants.KIM_NKC_MIXED


# --------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> dict[str, Any]:
    (ctrl_name, ctrl_root), (new_name, new_root) = args.control, args.row
    rows = {CTRL: (Path(ctrl_root), ctrl_name), NEW: (Path(new_root), new_name)}
    deciding, labelled = constants.KIM_NKC_SETS
    sets = [*deciding, *labelled]
    run_dirs = {k: root / kw.K5 / f"s{args.seed}" for k, (root, _) in rows.items()}
    mul_mil = check_fixed_runs(run_dirs)
    diff = check_config(ka._load_config(run_dirs[CTRL]), ka._load_config(run_dirs[NEW]))
    ch.check_checkpoints(rows, sets, args.seed)
    start = edge.start_check(run_dirs[CTRL], run_dirs[NEW])
    if not start["same_draws"]:
        raise ValueError(f"step-1 mil differs ({start}): §66 NK4, re-train the control here")
    per_set: dict[str, Any] = {}
    for set_name in sets:
        data = ka.read_set(set_name, rows, args.seed)
        ka.check_items(data["items"], set_name)
        per_set[set_name] = {
            "cells": data["cells"], "videos": len(data["labels"]),
            MICRO: ka.micro_contrasts(data["curves"], data["labels"], data["groups"], TERMS,
                                      args.draws, constants.V5_BOOTSTRAP_SEED)["N"],
            MACRO: macro_contrast(data["items"], data["item_groups"]),
        }
        LOGGER.info("%s read", set_name)
    labels = {m: {s: ch.label(per_set[s][m]) for s in sets} for m in (MICRO, MACRO)}
    report: dict[str, Any] = {
        "rule": RULE, "label": LABEL, "seed": args.seed, "draws": args.draws,
        "rows": {k: {"name": n, "root": str(r), "env": read_env(run_dirs[k])}
                 for k, (r, n) in rows.items()},
        "deciding_sets": list(deciding), "labelled_sets": list(labelled),
        "fixed_mul_mil": mul_mil, "config_diff": diff, "start": start,
        "pre_nkc": pre_nkc_drift(run_dirs[CTRL], run_dirs[NEW]),
        "sets": per_set, "labels": labels,
        "verdicts": {m: verdict(labels[m], deciding) for m in (MICRO, MACRO)},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = constants.KIM_NKC_REPORT_STEM
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    return report


# --------------------------------------------------------------------------- render
def _signed(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def render(r: dict[str, Any]) -> str:
    c, n = r["rows"][CTRL]["name"], r["rows"][NEW]["name"]
    sets = [*r["deciding_sets"], *r["labelled_sets"]]
    st, pre = r["start"], r["pre_nkc"]
    lines = [f"# §66 Does `kim_nkc` contribute? {c} vs {n}, s{r['seed']} — **{r['label']}**", "",
             f"Control **{c}** ({r['rows'][CTRL]['env']}), row **{n}** "
             f"({r['rows'][NEW]['env']}). Config diff: {r['config_diff']}. Deciding sets: "
             f"{', '.join(r['deciding_sets'])}; labelled only: {', '.join(r['labelled_sets'])}.",
             "", f"Start (step-1 `mil`): {st['control']:.6f} vs {st['new']:.6f}, matched. "
             f"Before the first `kim_nkc` (step {pre['first_nkc_step']}): max |Δ total| "
             f"{pre['max_abs_total']:.2e} over {pre['steps_compared']} steps (printed).", "",
             "## Verdict (NK7)", "", "| metric | verdict |", "|---|---|",
             f"| micro | {r['verdicts'][MICRO]} |", f"| macro | {r['verdicts'][MACRO]} |", "",
             "## Cells (micro / macro)", "", "| row | " + " | ".join(sets) + " |",
             "|---|" + "---|" * len(sets)]
    for key in (CTRL, NEW):
        cells = [f"{r['sets'][s]['cells'][key][MICRO]:.4f} / "
                 f"{r['sets'][s]['cells'][key][MACRO]:.4f}" for s in sets]
        lines.append(f"| {r['rows'][key]['name']} | " + " | ".join(cells) + " |")
    lines += ["", f"## N = {c} - {n} (what `kim_nkc` adds; labels §53 KA5)", "",
              "| metric | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
    for metric in (MICRO, MACRO):
        cells = [f"{_signed(r['sets'][s][metric])} {r['labels'][metric][s]}" for s in sets]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines += ["", "Never: \"the weight is too small\" unless NKC HELPS; \"prototypes do not "
              "matter\" (the row keeps them); a sentence from a labelled-only set; anything "
              "about the VideoMAE host; a seed claim."]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--control", nargs=2, required=True, metavar=("NAME", "ROOT"),
                        help="ROOT holds K5/s<seed>/ and eval/")
    parser.add_argument("--row", nargs=2, required=True, metavar=("NAME", "ROOT"))
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--draws", type=int, default=constants.BASEMICRO_BOOTSTRAP_DRAWS)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(build_arg_parser().parse_args(argv))
    LOGGER.info("verdicts: %s", report["verdicts"])


__all__ = ["check_config", "macro_contrast", "pre_nkc_drift", "render", "run", "verdict"]


if __name__ == "__main__":
    main()
