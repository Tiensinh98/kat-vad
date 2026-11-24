"""Does the level gap between third-party and ego accidents shrink on T2-trained rows?

``KIM_RUN.md`` §46 (rule written 2026-11-24, before any number of this check). Per row, every
DoTA-CAP clip has a level ``A_c`` and a low flag (§44.2, ``micro_level_read.levels``). The gap
``G`` is low share(ego-uninvolved) minus low share(ego-involved), ``G_A`` the same in mean
``A_c`` with the sign reversed so that a positive gap means "third-party clips sit lower". Each
pair compares a T2-trained row with the Nexar-trained row of the same recipe; differences are
paired over clips with a cluster bootstrap over source videos. Saved scores and annotations
only; printed, decides nothing.

Inputs::

    --dota ROW=DIR ...       whole DoTA-CAP read dirs (clip_scores.npz), every row of the pairs
    --dota-annotations       DoTA raw_annotations/annotations/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.v2_splits import dota_group
from core.metrics import frame_auc
from core.tools import k5_fail_tail as ft
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.micro_level_read import levels, parse_rows
from core.tools.protocol_b_eval import CLIP_SCORES_NPZ, read_clip_scores

LOGGER = logging.getLogger(__name__)

SUPPORT = "SUPPORT-CONSISTENT"
OTHER_SIDE = "NARROWED BY THE OTHER SIDE"
NO_SHRINK_EXCLUDED = "NO SHRINK SHOWN (a third or more excluded)"
NO_SHRINK_OPEN = "NO SHRINK SHOWN (not settled)"
WIDENED = "WIDENED"
MIXED = "MIXED"

Row = dict[str, np.ndarray]


def row_arrays(run_dir: Path, ids: list[str]) -> Row:
    """EG2: ``low``, ``A`` and own AUC per clip of ``ids``; another clip set raises."""
    scores, labels = read_clip_scores(run_dir / CLIP_SCORES_NPZ)
    if sorted(labels) != ids:
        raise ValueError(f"{run_dir}: clips differ from the first row's")
    lv = levels(scores, labels)
    a = np.array([lv["A"][v] for v in ids])
    return {"low": (a < constants.MICRO_LEVEL_LOW).astype(np.float64), "A": a,
            "auc": np.array([frame_auc(scores[v], labels[v]) for v in ids])}


def gate_reference(row: str, low: np.ndarray, involved: np.ndarray) -> None:
    """EG2 gate: a Nexar row must reproduce §44.3's low shares."""
    want = constants.EGO_GAP_REFERENCE_LOW.get(row)
    if want is None:
        return
    got = (float(low[~involved].mean()), float(low[involved].mean()))
    if any(abs(g - w) > constants.EGO_GAP_GATE_ATOL for g, w in zip(got, want, strict=True)):
        raise ValueError(f"EG2 gate: {row} low shares {got}, §44.3 reads {want}")


def gaps(row: Row, involved: np.ndarray) -> dict[str, float]:
    """EG3 on one sample of clips: group means and the two gaps."""
    out = {f"{k}_{g}": float(row[k][m].mean())
           for k in ("low", "A", "auc") for g, m in (("uninvolved", ~involved),
                                                     ("involved", involved))}
    out["G"] = out["low_uninvolved"] - out["low_involved"]
    out["G_A"] = out["A_involved"] - out["A_uninvolved"]
    return out


def paired(nexar: Row, t2: Row, involved: np.ndarray,
           draws: list[np.ndarray]) -> dict[str, dict[str, float]]:
    """EG3: T2 - Nexar per statistic with a cluster-bootstrap interval (same draws, both rows)."""
    point_n, point_t = gaps(nexar, involved), gaps(t2, involved)
    boot: dict[str, list[float]] = {k: [] for k in point_n}
    for d in draws:
        if involved[d].all() or not involved[d].any():
            continue
        g_n = gaps({k: x[d] for k, x in nexar.items()}, involved[d])
        g_t = gaps({k: x[d] for k, x in t2.items()}, involved[d])
        for k in boot:
            boot[k].append(g_t[k] - g_n[k])
    out = {}
    for k, values in boot.items():
        low, high = np.quantile(values, [ft.ALPHA, 1.0 - ft.ALPHA])
        out[k] = {"mean": point_t[k] - point_n[k], "low": float(low), "high": float(high)}
    return out


def verdict(d: dict[str, dict[str, float]]) -> str:
    """EG4 on the paired differences."""
    if d["G"]["low"] > 0:
        return WIDENED
    if d["G"]["high"] < 0:
        if d["G_A"]["high"] >= 0:
            return MIXED
        return SUPPORT if d["low_uninvolved"]["high"] < 0 else OTHER_SIDE
    return NO_SHRINK_EXCLUDED if d["G"]["low"] > -constants.EGO_GAP_THIRD else NO_SHRINK_OPEN


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = parse_rows(args.dota)
    names = [r for p in constants.EGO_GAP_PAIRS for r in p]
    if missing := [r for r in names if r not in dirs]:
        raise ValueError(f"rows not given: {missing}")
    ids = sorted(read_clip_scores(dirs[names[0]] / CLIP_SCORES_NPZ)[1])
    strata = ft.dota_strata(ids, args.dota_annotations)["ego_involve"]
    involved = np.array([strata[v].lower() == "true" for v in ids])
    rows = {r: row_arrays(dirs[r], ids) for r in names}
    for r, arr in rows.items():
        gate_reference(r, arr["low"], involved)
    draws = ft.draws_of(ids, {v: dota_group(v) for v in ids})
    pairs = {}
    for nexar, t2 in constants.EGO_GAP_PAIRS:
        diff = paired(rows[nexar], rows[t2], involved, draws)
        pairs[f"{t2} - {nexar}"] = {"diff": diff, "verdict": verdict(diff)}
    report = {"rule": "KIM_RUN.md §46", "seed": ft.SEED, "draws": ft.DRAWS,
              "n": {"uninvolved": int((~involved).sum()), "involved": int(involved.sum())},
              "rows": {r: gaps(a, involved) for r, a in rows.items()}, "pairs": pairs}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / constants.EGO_GAP_REPORT_STEM
    write_json_atomic(stem.with_suffix(".json"), report)
    write_text_atomic(stem.with_suffix(".md"), render_markdown(report))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    n = report["n"]
    lines = [f"# Ego gap in level ({report['rule']}) — printed, decides nothing, n = 1", "",
             f"DoTA-CAP {n['uninvolved']} uninvolved / {n['involved']} involved; "
             f"{report['draws']} draws, seed {report['seed']}.", "",
             "| row | low U | low I | G | mean A_c U | mean A_c I | G_A | own AUC U | own AUC I |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row, g in report["rows"].items():
        lines.append(f"| {row} | {g['low_uninvolved']:.3f} | {g['low_involved']:.3f} | "
                     f"{g['G']:.3f} | {g['A_uninvolved']:.3f} | {g['A_involved']:.3f} | "
                     f"{g['G_A']:.3f} | {g['auc_uninvolved']:.3f} | {g['auc_involved']:.3f} |")
    for pair, p in report["pairs"].items():
        lines += ["", f"## {pair}: {p['verdict']}", "", "| statistic | Δ [95 %] |",
                  "| --- | --- |"]
        lines += [f"| {k} | {v['mean']:+.3f} [{v['low']:+.3f}, {v['high']:+.3f}] |"
                  for k, v in p["diff"].items()]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument("--dota", nargs="+", required=True, help="ROW=DIR")
    for flag in ("--dota-annotations", "--out-dir"):
        parser.add_argument(flag, type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = run(build_arg_parser().parse_args(argv))
    LOGGER.info("%s", {k: p["verdict"] for k, p in report["pairs"].items()})


if __name__ == "__main__":
    main()
