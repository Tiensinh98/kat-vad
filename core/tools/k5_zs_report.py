"""K5 vs LaGoVAD's released checkpoint on the four zero-shot sets: macro, micro, AP.

The advisor's table: one seed of the main model (``KIM_RUN.md`` §12.3) beside the public checkpoint,
same protocols as ``ZS_COMPARISON.md``. Pure aggregation, no model is loaded. It refuses to write
when two rows did not score the same items, when a recomputed micro disagrees with its read-out,
when K5's reads did not come from one checkpoint, or when LaGoVAD misses Gate T0 on TAD. A set K5
has not read yet is listed as missing, never filled. Inputs::

    <lagovad-root>/eval/dota/LaGoVAD/B/, eval/dada/LaGoVAD/, eval/{tad,dad}/LaGoVAD/released/
    <kim-root>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/

DAD also gets ``ZS_DAD.md`` §8's reads, because its macro cannot separate a reaction to the accident
from a score that rises with time. CLI::

    python -m core.tools.k5_zs_report --lagovad-root outputs/v_final/zs_lagovad \\
        --kim-root outputs/v_final/kim --out-dir outputs/v_final/kim/REPORTS
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
from core.tools import dad_position_diag as diag
from core.tools import dada_whole_eval, protocol_b_eval, tadz_readout, zs_compare_table
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_transfer_readout import ArmRead, read_dada, read_tad_runs
from core.tools.zs_lagovad_readout import (
    PROTOCOL_B,
    delta_ci,
    read_dota,
    t0_check,
    versus_sentence,
)

LOGGER = logging.getLogger(__name__)

REPORT_STEM = "k5_zs_report"
LABEL = "PRELIMINARY (n = 1 seed)"
EVAL_DIR = "eval"
RESULTS_JSON = "results.json"
LAG, K5 = constants.ZS_LAGOVAD_ARM, constants.KIM_MAIN_ARM
DOTA, DADA, TAD, DAD = (zs_compare_table.DOTA, zs_compare_table.DADA, zs_compare_table.TAD,
                        zs_compare_table.DAD)
SET_ORDER = zs_compare_table.SET_ORDER
SET_NAME = zs_compare_table.SET_NAME
MICRO_TOLERANCE = zs_compare_table.MICRO_TOLERANCE
RUN_SETS = (TAD, DAD)  # core.evaluate outputs; the others are clip_scores.npz reads
PROTOCOL = {
    DOTA: "569 two-class clips; protocol B (`s1[::3]`, native frames); per-clip min-max",
    DADA: "370 two-class T2-test sources; stride 8, whole video; raw scores",
    TAD: "100 test videos, macro over the 60 two-class; stride 8; raw scores",
    DAD: ("466 test videos, macro over the 165 positives; stride 7 at 25 fps, `toa = 90`; "
          "raw scores"),
}
NOTES = """## How to read this table

* **PRELIMINARY (n = 1 seed, s2024).** K5 is the chosen main model, not the outcome of a rule.
* **Not a matched comparison.** The public checkpoint trained on PreVAD, K5 on Nexar, and its
  zero-shot status on these sets is unverified (`ZS_TAD_DADA.md` §7). Never write "beats LaGoVAD".
  Every cell is measured here; no paper number may sit beside these (lesson C8).
* **Bracket:** 95 % cluster bootstrap of the macro. **Δ:** paired per item, same bootstrap.
* **Micro on TAD and DAD measures clip classification**: read the clip oracle printed below.
* **DAD macro is not localization (`ZS_DAD.md` §8).** Every positive is abnormal on the same last
  two rows. Read the ruler, placebo and same-position lines before quoting it.
* **Read AP against the base rate in its row**: what a random scorer gets.
"""

Curves = dict[str, np.ndarray]


def run_name(row: str, seed: int) -> str:
    """The run directory of one row of a ``core.evaluate`` set."""
    return constants.ZS_LAGOVAD_RUN if row == LAG else f"s{seed}"


def row_dir(args: argparse.Namespace, set_name: str, row: str) -> Path:
    """Where one row of one set was read (for TAD / DAD the parent of its run directory)."""
    root: Path = args.lagovad_root if row == LAG else args.kim_root
    base = root / EVAL_DIR / set_name / row
    return base / PROTOCOL_B if set_name == DOTA and row == LAG else base


def score_source(args: argparse.Namespace, set_name: str, row: str) -> Path:
    """The saved scores of one row: a ``clip_scores.npz`` or a ``core.evaluate`` run dir."""
    base = row_dir(args, set_name, row)
    if set_name in RUN_SETS:
        return base / run_name(row, args.seed)
    return base / protocol_b_eval.CLIP_SCORES_NPZ


def read_row(args: argparse.Namespace, set_name: str, row: str) -> ArmRead:
    """Macro, micro and items of one row, through the read-out's own readers."""
    base = row_dir(args, set_name, row)
    if set_name == DOTA:
        return read_dota(base)
    if set_name == DADA:
        return read_dada(base)
    expect = tadz_readout.TAD_PROTOCOL_SIZE if set_name == TAD else None
    return read_tad_runs(base, [run_name(row, args.seed)], constants.V5_BOOTSTRAP_SEED, expect)


def k5_checkpoints(args: argparse.Namespace, sets: list[str]) -> dict[str, str]:
    """The checkpoint behind each of K5's reads, from the read's own record."""
    found: dict[str, str] = {}
    for set_name in sets:
        base = row_dir(args, set_name, K5)
        if set_name in RUN_SETS:
            doc = json.loads((base / f"s{args.seed}" / RESULTS_JSON).read_text(encoding="utf-8"))
            found[set_name] = str(doc["checkpoint"])
        else:
            name = (protocol_b_eval.READOUT_JSON if set_name == DOTA
                    else dada_whole_eval.READOUT_JSON)
            doc = json.loads((base / name).read_text(encoding="utf-8"))
            found[set_name] = str(doc["checkpoints"][f"s{args.seed}"])
    return found


def check_one_checkpoint(checkpoints: dict[str, str]) -> None:
    """K5's sets must all be reads of the same checkpoint file."""
    if len(set(checkpoints.values())) > 1:
        raise ValueError(f"K5's reads come from different checkpoints: {checkpoints}")


def set_report(args: argparse.Namespace, set_name: str) -> dict[str, Any]:
    """Both rows of one set, gated on same items and on the micro of the read-out."""
    reads = {row: read_row(args, set_name, row) for row in (LAG, K5)}
    if set(reads[LAG]["items"]) != set(reads[K5]["items"]):
        raise ValueError(f"{set_name}: {LAG} scored {len(reads[LAG]['items'])} items, "
                         f"{K5} {len(reads[K5]['items'])}: the contrast is not paired")
    rows: dict[str, Any] = {}
    base_rate = None
    for row, read in reads.items():
        loaded = zs_compare_table.load_scores(score_source(args, set_name, row))
        if loaded is None:
            raise FileNotFoundError(f"{set_name}/{row}: no scores, AP cannot be computed")
        pooled = zs_compare_table.pooled_reads(*loaded, minmax=set_name == DOTA)
        if abs(pooled["micro"] - read["micro"]) > MICRO_TOLERANCE:
            raise ValueError(f"{set_name}/{row}: micro {pooled['micro']:.6f} from the scores != "
                             f"{read['micro']:.6f} in the read: not the same scores")
        rows[row] = {"macro": read["macro"], "micro": read["micro"], "ap": pooled["ap"],
                     "clip_auc": read["clip_auc"]}
        base_rate = pooled["base_rate"]
    delta = delta_ci(reads[K5], reads[LAG])
    extra = reads[K5]["extra"]
    return {
        "rows": rows, "base_rate": base_rate, "protocol": PROTOCOL[set_name],
        "delta_macro": delta, "sentence": versus_sentence(delta),
        "clip_oracle_micro": extra.get("clip_oracle_micro"),
    }


def dad_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    """``ZS_DAD.md`` §8's reads for LaGoVAD and K5: ruler, placebo, same position."""
    runs = {row: tadz_readout.load_run(score_source(args, DAD, row)) for row in (LAG, K5)}
    labels = tadz_readout.check_labels(runs, None)
    abnormal = diag.accident_rows(labels)
    ids = sorted(labels)
    pos_ids = [v for v in ids if labels[v].sum() > 0]
    neg_ids = [v for v in ids if labels[v].sum() == 0]
    full_from = diag.first_full_window_row()
    rng = np.random.default_rng(constants.V5_BOOTSTRAP_SEED)
    resamples = constants.V2_E1_BOOTSTRAP
    draws = (rng.integers(0, len(pos_ids), (resamples, len(pos_ids))),
             rng.integers(0, len(neg_ids), (resamples, len(neg_ids))))
    rows = {row: diag.row_reads(np.stack([runs[row][0][v] for v in pos_ids]),
                                np.stack([runs[row][0][v] for v in neg_ids]),
                                abnormal, full_from, draws)
            for row in (LAG, K5)}
    boots = {row: rows[row]["same_position"].pop("_boot") for row in (LAG, K5)}
    length = len(labels[ids[0]])
    truth = np.isin(np.arange(length), abnormal).astype(np.int64)
    k5_at, lag_at = (rows[r]["same_position"]["accident_rows"]["mean"] for r in (K5, LAG))
    return {
        "label": diag.LABEL, "first_full_window_row": full_from,
        "ruler_macro": frame_auc(np.arange(length, dtype=np.float64), truth),
        "rows": rows,
        "same_position_delta": diag._interval(k5_at - lag_at, boots[K5] - boots[LAG]),
    }


def _macro(c: dict[str, float]) -> str:
    return f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _signed(c: dict[str, float]) -> str:
    return f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def render(report: dict[str, Any]) -> str:
    """The comparison as one markdown document."""
    lines = [
        f"# Zero-shot: K5 vs LaGoVAD public checkpoint — **{report['label']}**",
        "",
        f"K5 = VideoMAE + KIM, DVS off, trained on Nexar, seed {report['seed']}. Generated by "
        "`core.tools.k5_zs_report` from the saved scores.",
        "",
        "| Dataset | Metric | LaGoVAD (public ckpt) | K5 | Base rate |",
        "|---|---|---|---|---|",
    ]
    for name, s in report["sets"].items():
        lag, k5 = s["rows"][LAG], s["rows"][K5]
        macros = f"{_macro(lag['macro'])} | {_macro(k5['macro'])}"
        lines += [
            f"| **{SET_NAME[name]}** | AUC macro | {macros} | |",
            f"| | AUC micro | {lag['micro']:.4f} | {k5['micro']:.4f} | |",
            f"| | AP | {lag['ap']:.4f} | {k5['ap']:.4f} | {s['base_rate']:.4f} |",
        ]
    lines += ["", "## K5 - LaGoVAD, macro, paired per item", ""]
    lines += [f"* {SET_NAME[n]}: {_signed(s['delta_macro'])} → **{s['sentence']}**"
              for n, s in report["sets"].items()]
    lines += ["", "## Protocol per set", "", "| Set | Items and pooling | Clip oracle micro |",
              "|---|---|---|"]
    for name, s in report["sets"].items():
        oracle = s["clip_oracle_micro"]
        lines.append(f"| {SET_NAME[name]} | {s['protocol']} | "
                     f"{'—' if oracle is None else f'{oracle:.4f}'} |")
    if report["missing"]:
        lines += ["", f"**Not read yet:** {', '.join(SET_NAME[n] for n in report['missing'])}."]
    if report["dad_diagnostics"] is not None:
        lines += ["", *_render_dad(report["dad_diagnostics"])]
    return "\n".join(lines) + "\n\n" + NOTES


def _render_dad(d: dict[str, Any]) -> list[str]:
    full = d["first_full_window_row"]
    lines = [
        f"## DAD: what the macro does not measure ({d['label']})",
        "",
        f"No-pixel `t/N` ruler macro = **{d['ruler_macro']:.4f}**. Placebo = the same mask on the "
        "negatives, which hold no accident. Same position = positive vs negative videos on the "
        "same rows.",
        "",
        f"| row | macro | placebo, all rows | macro, rows {full}+ | placebo, rows {full}+ | "
        f"same position, accident rows | same position, rows {full} up to the accident |",
        "|---|---|---|---|---|---|---|",
    ]
    for row, r in d["rows"].items():
        a, f, s = r["all_rows"], r["full_window_rows"], r["same_position"]
        lines.append(f"| {row} | {a['positive']:.4f} | {_macro(a['placebo_ci'])} | "
                     f"{f['positive']:.4f} | {_macro(f['placebo_ci'])} | "
                     f"{_macro(s['accident_rows'])} | {_macro(s['rows_before'])} |")
    contrast = _signed(d["same_position_delta"])
    lines += ["", f"Same position, accident rows, K5 - LaGoVAD: {contrast}"]
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--lagovad-root", type=Path, required=True,
                        help="outputs/zs_lagovad (eval/ under it)")
    parser.add_argument("--kim-root", type=Path, required=True,
                        help="outputs/kim (eval/ under it)")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    present = [s for s in SET_ORDER if s not in RUN_SETS or (
        row_dir(args, s, K5) / f"s{args.seed}" / RESULTS_JSON).is_file()]
    missing = [s for s in SET_ORDER if s not in present]
    check_one_checkpoint(k5_checkpoints(args, present))
    sets = {s: set_report(args, s) for s in present}
    if TAD in sets:
        check = t0_check(read_row(args, TAD, LAG))
        if not check["pass"]:
            raise ValueError(f"released checkpoint does not reproduce Gate T0 on TAD: {check}")
    report: dict[str, Any] = {
        "label": LABEL, "seed": args.seed, "sets": sets, "missing": missing,
        "dad_diagnostics": dad_diagnostics(args) if DAD in sets else None,
    }
    write_json_atomic(args.out_dir / f"{REPORT_STEM}.json", report)
    write_text_atomic(args.out_dir / f"{REPORT_STEM}.md", render(report))
    LOGGER.info("report -> %s (missing: %s)", args.out_dir / f"{REPORT_STEM}.md", missing)
    return report


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
