"""Macro and micro (plus AP) of KIM arms on the four zero-shot sets, paired against one control.

``kim_readout`` decides on macro and prints micro only where its reader records one (raw DADA,
raw TAD, nothing on DoTA, no DAD). This tool prints both on every set, pooled as the comparison
tables pool them (``zs_compare_table``): **DoTA-CAP-dev and DADA-2000 after per-video min-max**
(every item holds an accident, C12 / ``KIM_RUN.md`` §21.1), **TAD and DAD raw** (they hold normal
videos). Every delta is paired per item against the control: macro by ``delta_ci`` (the
read-outs' bootstrap), micro by a cluster bootstrap of the pooled AUC (DoTA: source video; else
the video). It refuses rows that scored different items or labels, and a score file whose
as-read micro disagrees with its read-out. Printed only: it decides nothing. CLI::

    python -m core.tools.zs_arm_table --control K5 --seed 2024 \\
        --read dota K5 .../eval/dota/K5 --read dada K5 .../eval/dada/K5 \\
        --read tad K5 .../eval/tad/K5 --read dad K5 .../eval/dad/K5 \\
        --read dota K5ln ... --name k5ln_s2024 --out-dir .../REPORTS
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import normalize_scores
from core.tools import protocol_b_eval, tadz_readout, zs_compare_table
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_transfer_readout import ArmRead, read_dada, read_tad_runs
from core.tools.tad_clip_gate import micro_bootstrap_pairs
from core.tools.zs_lagovad_readout import check_same_items, delta_ci, read_dota, versus_sentence

LOGGER = logging.getLogger(__name__)

DOTA, DADA, TAD, DAD = (zs_compare_table.DOTA, zs_compare_table.DADA, zs_compare_table.TAD,
                        zs_compare_table.DAD)
SET_ORDER = zs_compare_table.SET_ORDER
SET_NAME = zs_compare_table.SET_NAME
MINMAX_SETS = zs_compare_table.TABLE_MINMAX_SETS
RUN_SETS = (TAD, DAD)  # core.evaluate outputs under <dir>/s<seed>; the others are clip_scores.npz
MEAN_TOL = 1e-9  # the bootstrap's point delta must equal the difference of the printed cells
POOLING = {
    DOTA: "569 clips, protocol B; micro / AP after per-clip min-max",
    DADA: "370 whole sources, s8; micro / AP after per-video min-max",
    TAD: "100 videos (macro over the 60 two-class), s8; raw",
    DAD: "466 videos (macro over the 165 positives), s7; raw",
}
DESCRIPTION = "Macro and micro of KIM arms on the four zero-shot sets, paired vs a control."
LABEL = "PRELIMINARY (n = 1 seed); printed, decides nothing"

Curves = dict[str, np.ndarray]


def read_set(set_name: str, arm_dir: Path, seed: int) -> ArmRead:
    """One arm's read of one set, through the read-outs' own readers."""
    if set_name == DOTA:
        return read_dota(arm_dir)
    if set_name == DADA:
        return read_dada(arm_dir)
    expect = tadz_readout.TAD_PROTOCOL_SIZE if set_name == TAD else None
    return read_tad_runs(arm_dir, [f"s{seed}"], constants.V5_BOOTSTRAP_SEED, expect)


def score_source(set_name: str, arm_dir: Path, seed: int) -> Path:
    """Where one arm's per-frame scores of one set live."""
    if set_name in RUN_SETS:
        return arm_dir / f"s{seed}"
    return arm_dir / protocol_b_eval.CLIP_SCORES_NPZ


def table_curves(scores: Curves, set_name: str) -> Curves:
    """The curves the table pools: per-video min-max on the all-abnormal sets, raw otherwise."""
    if set_name not in MINMAX_SETS:
        return scores
    return {v: normalize_scores(s, constants.SCORE_NORM_MINMAX) for v, s in scores.items()}


def check_same_labels(loaded: dict[str, tuple[Curves, Curves]], set_name: str) -> Curves:
    """Every arm's score file holds the same items with identical frame labels."""
    names = list(loaded)
    reference = loaded[names[0]][1]
    for name in names[1:]:
        labels = loaded[name][1]
        if set(labels) != set(reference):
            raise ValueError(f"{set_name}: {name} scored other items than {names[0]}")
        bad = [v for v in reference if not np.array_equal(labels[v], reference[v])]
        if bad:
            raise ValueError(f"{set_name}: {name} has other labels than {names[0]} on {bad[:5]}")
    return reference


def set_table(set_name: str, rows: dict[str, ArmRead], loaded: dict[str, tuple[Curves, Curves]],
              control: str, draws: int) -> dict[str, Any]:
    """Cells and paired deltas of one set (pure: no file is read here)."""
    if control not in rows:
        raise ValueError(f"{set_name}: the control {control} has no read")
    check_same_items(rows, set_name)
    labels = check_same_labels(loaded, set_name)
    cells: dict[str, Any] = {}
    curves: dict[str, Curves] = {}
    for arm, read in rows.items():
        pooled = zs_compare_table.gated_reads(loaded[arm], set_name, read["micro"],
                                              f"{set_name}/{arm}")
        cells[arm] = {"macro": read["macro"], "micro": pooled["micro"], "ap": pooled["ap"],
                      "base_rate": pooled["base_rate"]}
        curves[arm] = table_curves(loaded[arm][0], set_name)
    groups = {v: rows[control]["groups"].get(v, v) for v in labels}
    pairs = {arm: (arm, control) for arm in rows if arm != control}
    micro_d = micro_bootstrap_pairs(curves, pairs, labels, constants.V5_BOOTSTRAP_SEED,
                                    groups=groups, draws=draws)
    deltas: dict[str, Any] = {}
    for arm in pairs:
        expected = cells[arm]["micro"] - cells[control]["micro"]
        if abs(micro_d[arm]["mean"] - expected) > MEAN_TOL:
            raise ValueError(f"{set_name}/{arm}: bootstrap micro delta {micro_d[arm]['mean']} "
                             f"!= cell difference {expected}: not the pooled curves")
        macro_d = delta_ci(rows[arm], rows[control])
        deltas[arm] = {"macro": macro_d, "macro_sentence": versus_sentence(macro_d),
                       "micro": micro_d[arm], "micro_sentence": versus_sentence(micro_d[arm])}
    oracle = rows[control].get("extra", {}).get("clip_oracle_micro")
    return {"pooling": POOLING[set_name], "cells": cells, "deltas": deltas,
            "clip_oracle_micro": oracle, "items": len(labels)}


def build(reads: list[tuple[str, str, Path]], control: str, seed: int,
          draws: int) -> dict[str, Any]:
    """``{set: set_table}`` for every set that has reads, in the tables' order."""
    by_set: dict[str, dict[str, Path]] = {}
    for set_name, arm, arm_dir in reads:
        if set_name not in SET_ORDER:
            raise ValueError(f"unknown set {set_name!r}; one of {SET_ORDER}")
        by_set.setdefault(set_name, {})[arm] = arm_dir
    out: dict[str, Any] = {}
    for set_name in SET_ORDER:
        if set_name not in by_set:
            continue
        dirs = by_set[set_name]
        rows = {arm: read_set(set_name, d, seed) for arm, d in dirs.items()}
        loaded = {}
        for arm, d in dirs.items():
            got = zs_compare_table.load_scores(score_source(set_name, d, seed))
            if got is None:
                raise FileNotFoundError(f"{set_name}/{arm}: no per-frame scores under {d}")
            loaded[arm] = got
        LOGGER.info("%s: %d arms, bootstrap %d draws", set_name, len(rows), draws)
        out[set_name] = set_table(set_name, rows, loaded, control, draws)
    return out


def _ci(c: dict[str, float]) -> str:
    return f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _signed(c: dict[str, float]) -> str:
    return f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def render(table: dict[str, Any], control: str, seed: int, title: str) -> str:
    """The markdown report: one summary grid, then per set the cells and the paired deltas."""
    arms = list(dict.fromkeys(a for t in table.values() for a in t["cells"]))
    lines = [f"# {title} — macro and micro on the zero-shot sets, s{seed} — **{LABEL}**", "",
             f"Control = **{control}**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after "
             "per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per "
             f"item; macro by the read-outs' bootstrap, micro by a cluster bootstrap "
             f"(seed {constants.V5_BOOTSTRAP_SEED}).", "", "## Summary (macro / micro)", "",
             "| arm | " + " | ".join(SET_NAME[s] for s in table) + " |",
             "| --- |" + " --- |" * len(table)]
    for arm in arms:
        row = []
        for t in table.values():
            c = t["cells"].get(arm)
            row.append(f"{c['macro']['mean']:.4f} / {c['micro']:.4f}" if c else "—")
        lines.append(f"| {arm} | " + " | ".join(row) + " |")
    for set_name, t in table.items():
        lines += ["", f"## {SET_NAME[set_name]} ({t['pooling']})", "",
                  "| arm | macro [95 %] | micro | AP | base rate |",
                  "| --- | --- | --- | --- | --- |"]
        for arm, c in t["cells"].items():
            lines.append(f"| {arm} | {_ci(c['macro'])} | {c['micro']:.4f} | {c['ap']:.4f} | "
                         f"{c['base_rate']:.4f} |")
        if t["clip_oracle_micro"] is not None:
            lines.append(f"\nClip oracle micro (a constant score per video): "
                         f"{t['clip_oracle_micro']:.4f}. Micro here measures clip classification.")
        if t["deltas"]:
            lines += ["", "| contrast | Δ macro | sentence | Δ micro | sentence |",
                      "| --- | --- | --- | --- | --- |"]
            for arm, d in t["deltas"].items():
                lines.append(f"| {arm} - {control} | {_signed(d['macro'])} | "
                             f"{d['macro_sentence']} | {_signed(d['micro'])} | "
                             f"{d['micro_sentence']} |")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--read", nargs=3, action="append", required=True,
                        metavar=("SET", "ARM", "DIR"),
                        help=f"one arm's read of one set; SET in {SET_ORDER}. DIR = the "
                             "protocol_b / dada_whole_eval output dir, or the parent of s<seed>/ "
                             "for tad / dad")
    parser.add_argument("--control", required=True)
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--draws", type=int, default=constants.BASEMICRO_BOOTSTRAP_DRAWS)
    parser.add_argument("--name", required=True, help="report = <stem>_<name>.{md,json}")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    reads = [(s, a, Path(d)) for s, a, d in args.read]
    table = build(reads, args.control, args.seed, args.draws)
    stem = f"{constants.ZS_ARM_TABLE_STEM}_{args.name}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{stem}.json",
                      {"control": args.control, "seed": args.seed, "draws": args.draws,
                       "label": LABEL, "sets": table})
    write_text_atomic(args.out_dir / f"{stem}.md",
                      render(table, args.control, args.seed, args.name))
    return table


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()

__all__ = ["build", "render", "run", "set_table", "table_curves"]
