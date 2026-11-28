"""Edge fill on the CLIP-only DoTA host: L5dfiDoTAef vs the trunk (``KIM_RUN.md`` §68).

Pure aggregation over saved reads, PRELIMINARY (n = 1, DoTA in-domain). Three rows: the trunk
(L0dfiDoTArr, or L0dfiDoTArr2 re-trained on the session's GPU, EF3), the failed row L5dfiDoTA
(§64.2) and L5dfiDoTAef = L5dfiDoTA's command + ``kim.edge_fill=true`` (EF2). Per set and metric:
**E** = L5dfiDoTAef - trunk (decides, EF7) and **F** = L5dfiDoTAef - L5dfiDoTA (printed). Micro by
``kim_ablation_report``'s pooled-AUC cluster bootstrap, macro by the read-outs' bootstrap over the
items every row scored; labels §53 KA5.

Guards (raise, EF4): §31 floor, config diff L5dfiDoTA -> L5dfiDoTAef exactly
``{kim.edge_fill: false -> true}``, step-1 ``mil`` of the new row equal to the trunk's, one
checkpoint per row and none shared, same items. Printed: the slot ratio from §67's window-geometry
read of the new row (``--geometry-json``). CLI::

    python -m core.tools.kim_edge_fill_report --trunk L0dfiDoTArr .../kim_clip_host/L0dfiDoTArr \\
        --failed L5dfiDoTA .../L5dfiDoTA --row L5dfiDoTAef .../L5dfiDoTAef \\
        --geometry-json .../REPORTS/window_geometry_ef/window_geometry_read.json \\
        --out-dir .../kim_clip_host/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

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

RULE = "core/docs/final/KIM_RUN.md §68"
LABEL = "PRELIMINARY (n = 1 seed, DoTA in-domain); decides nothing beyond §68, K5f stays"
TRUNK, FAILED, NEW = "trunk", "failed", "fill"
ROW_KEYS = (TRUNK, FAILED, NEW)
TERMS: dict[str, dict[str, int]] = {"E": {NEW: 1, TRUNK: -1}, "F": {NEW: 1, FAILED: -1}}
TERM_TEXT = {"E": "E = fill - trunk (decides)", "F": "F = fill - failed (printed)"}
MICRO, MACRO = ka.MICRO, ka.MACRO
GAIN, COST = ch.GAIN, ch.COST
SLOT, REST = (1, 2), slice(3, None)  # EF7: in-window positions 1-2 over positions 3-11


# --------------------------------------------------------------------------- guards
def check_config(failed: dict[str, Any], new: dict[str, Any]) -> dict[str, list[bool]]:
    """EF4: the row is the failed row's command plus ``kim.edge_fill=true``, nothing else."""
    key = constants.KIM_EDGE_FILL_KEY
    diff = ka.config_diff(failed, new)
    if set(diff) != {key}:
        raise ValueError(f"config diff {sorted(diff)}, §68 EF4 allows exactly {{{key}}}")
    was, now = diff[key]
    if bool(was) or not bool(now):
        raise ValueError(f"{key}: {was} -> {now}, §68 EF4 needs false -> true")
    return {key: [bool(was), bool(now)]}


def slot_ratio(profile: list[float]) -> float | None:
    """EF7 printed: mean at positions 1-2 over the mean at positions 3-11 (None if absent)."""
    if len(profile) <= max(SLOT):
        return None
    rest = float(np.mean(profile[REST]))
    return float(np.mean([profile[i] for i in SLOT]) / rest) if rest > 0 else None


def read_slot(path: Path | None) -> dict[str, Any] | None:
    """The new row's slot ratio from §67's read (its L5 row must be the new row)."""
    if path is None:
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    row = doc["rows"][constants.KIM_WG_ROWS[0]]
    return {"row": row["name"], "ratio": slot_ratio(row["position_profile"]),
            "windowed_macro": row["macro"]["windowed"], "whole_macro": row["macro"]["whole"]}


# --------------------------------------------------------------------------- contrasts
def macro_terms(items: dict[str, dict[str, float]], groups: dict[str, str]
                ) -> dict[str, dict[str, float] | None]:
    common = sorted(set.intersection(*(set(items[k]) for k in ROW_KEYS)))
    if not common:
        raise ValueError("the three rows share no scored item")
    clusters = {v: str(groups.get(v, v)) for v in common}
    return {t: cluster_bootstrap_ci({v: sum(s * items[k][v] for k, s in signs.items())
                                     for v in common}, clusters, constants.V2_E1_BOOTSTRAP,
                                    constants.V5_BOOTSTRAP_SEED, constants.V2_E1_CI)
            for t, signs in TERMS.items()}


def verdict(labels: dict[str, str], deciding: tuple[str, ...]) -> str:
    """EF7 on E's labels over the deciding sets."""
    gain = any(labels[s] == GAIN for s in deciding)
    cost = any(labels[s] == COST for s in deciding)
    if gain and not cost:
        return constants.KIM_EDGE_FILL_HELPS
    if cost and not gain:
        return constants.KIM_EDGE_FILL_COSTS
    if not gain and not cost:
        return constants.KIM_EDGE_FILL_NO_DIFFERENCE
    return constants.KIM_EDGE_FILL_MIXED


# --------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = {TRUNK: (Path(args.trunk[1]), args.trunk[0]),
            FAILED: (Path(args.failed[1]), args.failed[0]),
            NEW: (Path(args.row[1]), args.row[0])}
    sets: list[str] = list(constants.KIM_EDGE_FILL_SETS)
    run_dirs = {k: root / kw.K5 / f"s{args.seed}" for k, (root, _) in rows.items()}
    mul_mil = check_fixed_runs(run_dirs)
    diff = check_config(ka._load_config(run_dirs[FAILED]), ka._load_config(run_dirs[NEW]))
    ch.check_checkpoints(rows, sets, args.seed)
    start = edge.start_check(run_dirs[TRUNK], run_dirs[NEW])
    if not start["same_draws"]:
        raise ValueError(f"step-1 mil differs from the trunk ({start}): §68 EF3, re-train "
                         f"{constants.KIM_EDGE_FILL_TRUNK_RERUN} in this session")
    start_failed = edge.start_check(run_dirs[FAILED], run_dirs[NEW])
    per_set: dict[str, Any] = {}
    for set_name in sets:
        data = ka.read_set(set_name, rows, args.seed)
        ka.check_items(data["items"], set_name)
        per_set[set_name] = {
            "cells": data["cells"], "videos": len(data["labels"]),
            MICRO: ka.micro_contrasts(data["curves"], data["labels"], data["groups"], TERMS,
                                      args.draws, constants.V5_BOOTSTRAP_SEED),
            MACRO: macro_terms(data["items"], data["item_groups"]),
        }
        LOGGER.info("%s read", set_name)
    labels = {m: {t: {s: ch.label(per_set[s][m][t]) for s in sets} for t in TERMS}
              for m in (MICRO, MACRO)}
    report: dict[str, Any] = {
        "rule": RULE, "label": LABEL, "seed": args.seed, "draws": args.draws, "sets": sets,
        "rows": {k: {"name": n, "root": str(r), "env": read_env(run_dirs[k])}
                 for k, (r, n) in rows.items()},
        "fixed_mul_mil": mul_mil, "config_diff": diff, "start": start,
        "start_failed": start_failed, "slot": read_slot(args.geometry_json),
        "per_set": per_set, "labels": labels,
        "verdicts": {m: verdict(labels[m]["E"], tuple(sets)) for m in (MICRO, MACRO)},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = constants.KIM_EDGE_FILL_REPORT_STEM
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    return report


# --------------------------------------------------------------------------- render
def _signed(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _ci(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:.4f}"


def render(r: dict[str, Any]) -> str:
    names = {k: r["rows"][k]["name"] for k in ROW_KEYS}
    sets, st = r["sets"], r["start"]
    lines = [f"# §68 Edge fill on the CLIP-only DoTA host, s{r['seed']} — **{r['label']}**", "",
             "Rows: " + "; ".join(f"{k} = **{names[k]}** ({r['rows'][k]['env']})"
                                  for k in ROW_KEYS) + f". Config diff: {r['config_diff']}.",
             "", f"Start (step-1 `mil`) trunk vs fill: {st['control']:.6f} vs {st['new']:.6f}, "
             f"matched. Failed vs fill: same draws = {r['start_failed']['same_draws']} "
             "(printed). DoTA-CAP-dev is in-domain for every row.", "",
             "## Verdict (EF7, on E)", "", "| metric | verdict |", "|---|---|",
             f"| micro | {r['verdicts'][MICRO]} |", f"| macro | {r['verdicts'][MACRO]} |", "",
             "## Cells (micro / macro)", "", "| row | " + " | ".join(sets) + " |",
             "|---|" + "---|" * len(sets)]
    for key in ROW_KEYS:
        cells = [f"{r['per_set'][s]['cells'][key][MICRO]:.4f} / "
                 f"{r['per_set'][s]['cells'][key][MACRO]:.4f}" for s in sets]
        lines.append(f"| {names[key]} | " + " | ".join(cells) + " |")
    lines += ["", "## Contrasts (labels §53 KA5)", "",
              "| contrast | metric | " + " | ".join(sets) + " |", "|---|---|" + "---|" * len(sets)]
    for term in TERMS:
        for metric in (MICRO, MACRO):
            cells = [f"{_signed(r['per_set'][s][metric][term])} {r['labels'][metric][term][s]}"
                     for s in sets]
            lines.append(f"| {TERM_TEXT[term]} | {metric} | " + " | ".join(cells) + " |")
    slot = r["slot"]
    lines += ["", "## Slot check (EF7, printed)", ""]
    if slot is None:
        lines.append("No window-geometry read given.")
    else:
        ratio = "—" if slot["ratio"] is None else f"{slot['ratio']:.2f}x"
        lines.append(f"{slot['row']}: in-window mean at positions 1-2 / 3-11 = **{ratio}** "
                     "(L5dfiDoTA 9.5x, L0 0.84x); whole / windowed macro "
                     f"{_ci(slot['whole_macro'])} / {_ci(slot['windowed_macro'])}.")
    lines += ["", "Never: \"the edge zeros caused §64.2\" unless the slot is gone and F GAINs "
              "(one seed, one host even then); \"edge fill improves KIM\" on another host; any "
              "zero-shot DoTA sentence; a seed claim."]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    for flag in ("--trunk", "--failed", "--row"):
        parser.add_argument(flag, nargs=2, required=True, metavar=("NAME", "ROOT"),
                            help="ROOT holds K5/s<seed>/ and eval/")
    parser.add_argument("--geometry-json", type=Path, default=None,
                        help="§67 window_geometry_read.json with the new row as L5 (printed)")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--draws", type=int, default=constants.BASEMICRO_BOOTSTRAP_DRAWS)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(build_arg_parser().parse_args(argv))
    LOGGER.info("verdicts: %s", report["verdicts"])


__all__ = ["check_config", "macro_terms", "read_slot", "render", "run", "slot_ratio", "verdict"]


if __name__ == "__main__":
    main()
