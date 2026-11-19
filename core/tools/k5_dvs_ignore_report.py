"""K5dfi / K5dfie: does DVS anchor mode ``ignore`` (and the ego DVS setting) help K5df? (§34).

``KIM_RUN.md`` §34.

Pure aggregation over saved reads, PRELIMINARY (n = 1). Rows: K5f (reference), K5df (§32, DVS on,
``span``), K5dfi (= K5df + ``loss.dvs_anchor_mode=ignore``), K5dfie (= K5dfi +
``data.is_egocentric=true``: ``theta_ego`` 0.85, ``delta_m_ego`` 2: fewer and shorter splices).
Per set every contrast is paired per item with ``k5_whole_report``'s readers.

R4 (margin ``V5_MIN_GAIN``, fixed before any number): a contrast is a **GAIN** iff its mean is at
least +0.01 with the interval above 0 on a set of ``KIM_DVS_IGNORE_GAIN_SETS`` and it is a COST on
none of ``KIM_LNEG_CAP_SETS``; a **COST** iff its mean is at most -0.01 with the interval below 0 on
any of them; otherwise **NO DIFFERENCE**. Primary: K5dfi - K5df; secondary (same rule):
K5dfie - K5dfi.
A row is a **candidate** iff its contrast with K5f is above on >= 1 set of ``KIM_LNEG_CAP_SETS`` and
below on none (C7's form); K5f stays the reference unless the user decides.

Guards (raise): every row's last-epoch ``mul_mil`` above ``KIM_FIXED_MUL_MIL_FLOOR``; K5dfi's step-1
``dvs_sup`` differs from K5df's (the mask is live). Flags (printed, never stop): K5dfi's step-1
``mil`` / ``dvs_sup_mil`` equal to K5df's within ``KIM_START_TOL`` (same draws; ``ignore`` changes
neither). K5dfie has no matched start (its splice draws differ). CLI::

    python -m core.tools.k5_dvs_ignore_report --kim-root outputs/v_final/kim_fixed/K5f \\
        --k5f-nexar-dir outputs/v_final/kim_fixed/K5f/eval/nexar/K5f \\
        --df-root outputs/v_final/kim_lneg_cap_fixed --ign-root outputs/v_final/kim_dvs_ignore \\
        --out-dir outputs/v_final/kim_dvs_ignore/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.tools import k5_neg_report as kn
from core.tools import k5_whole_report as kw
from core.tools import zs_compare_table as zs
from core.tools.k5_lneg_cap_report import _cell, check_fixed_runs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.zs_lagovad_readout import ABOVE, BELOW
from core.train import METRICS_FILENAME

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/KIM_RUN.md §34"
K5F = constants.KIM_REFERENCE_ARM
K5DF = constants.KIM_FIXED_DVS_ROW
K5DFI = constants.KIM_DVS_IGNORE_ROW
K5DFIE = constants.KIM_DVS_IGNORE_EGO_ROW
ROWS = (K5F, K5DF, K5DFI, K5DFIE)
NEXAR = kw.NEXAR
ALL_SETS = (NEXAR, *kw.ZS_SETS)
DECISION_SETS = constants.KIM_LNEG_CAP_SETS
# (name, control, new): new - control. The first two decide (R4); the rest are printed or feed
# the candidate test
CONTRASTS = (("ignore", K5DF, K5DFI), ("ego", K5DFI, K5DFIE), ("ego_vs_df", K5DF, K5DFIE),
             ("ignore_vs_reference", K5F, K5DFI), ("ego_vs_reference", K5F, K5DFIE),
             ("df_vs_reference", K5F, K5DF))
DECIDING = ("ignore", "ego")
LINE_CANDIDATE = "CANDIDATE (user decides; K5f stays until then)"
LINE_NOT_CANDIDATE = "not a candidate (K5f stays)"
START_KEYS = ("mil", "dvs_sup_mil")  # equal at step 1 between K5df and K5dfi (same draws)
MASK_KEY = "dvs_sup"  # differs at step 1 between K5df and K5dfi (the mask is live)


def label(cell: dict[str, Any]) -> str:
    """One set's GAIN / COST / NO DIFFERENCE from the contrast's mean and interval."""
    d = cell["delta"]
    if d is None:
        return constants.KIM_DVS_IGNORE_NO_DIFFERENCE
    gain = constants.V5_MIN_GAIN
    if d["mean"] >= gain and d["low"] > 0:
        return constants.KIM_DVS_IGNORE_GAIN
    if d["mean"] <= -gain and d["high"] < 0:
        return constants.KIM_DVS_IGNORE_COST
    return constants.KIM_DVS_IGNORE_NO_DIFFERENCE


def verdict(labels: dict[str, str]) -> str:
    """R4 for one deciding contrast, from its per-set labels (keys = ``KIM_LNEG_CAP_SETS``)."""
    if any(labels[s] == constants.KIM_DVS_IGNORE_COST for s in DECISION_SETS):
        return constants.KIM_DVS_IGNORE_COST
    if any(labels[s] == constants.KIM_DVS_IGNORE_GAIN for s in constants.KIM_DVS_IGNORE_GAIN_SETS):
        return constants.KIM_DVS_IGNORE_GAIN
    return constants.KIM_DVS_IGNORE_NO_DIFFERENCE


def candidate(sentences: dict[str, str]) -> dict[str, Any]:
    """C7's form on one row's contrast with K5f: above on >= 1 decision set, below on none."""
    above = [s for s in DECISION_SETS if sentences[s] == ABOVE]
    below = [s for s in DECISION_SETS if sentences[s] == BELOW]
    return {"candidate": bool(above) and not below, "above": above, "below": below}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def start_check(control_run: Path, new_run: Path) -> dict[str, Any]:
    """K5dfi vs K5df at step 1: same draws (a flag), live mask (raises if ``dvs_sup`` is equal)."""
    a = _rows(control_run / METRICS_FILENAME)[0]
    b = _rows(new_run / METRICS_FILENAME)[0]
    if a["global_step"] != 1 or b["global_step"] != 1:
        raise ValueError("metrics.jsonl does not start at global_step 1")
    if abs(a[MASK_KEY] - b[MASK_KEY]) <= constants.KIM_START_TOL:
        raise ValueError(f"step-1 {MASK_KEY} equal ({a[MASK_KEY]:.6f} vs {b[MASK_KEY]:.6f}): the "
                         "ignore mask removed nothing, the arm is not what §34 specifies")
    same = {k: abs(a[k] - b[k]) <= constants.KIM_START_TOL for k in START_KEYS}
    return {"same_draws": all(same.values()), "equal": same,
            "step1": {k: {"K5df": a[k], "K5dfi": b[k]} for k in (*START_KEYS, MASK_KEY)}}


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = f"s{args.seed}"
    roots = {K5F: args.kim_root, K5DF: args.df_root / K5DF,
             K5DFI: args.ign_root / K5DFI, K5DFIE: args.ign_root / K5DFIE}
    run_dirs = {row: root / kw.K5 / run_root for row, root in roots.items()}
    mul_mil = check_fixed_runs(run_dirs)
    start = start_check(run_dirs[K5DF], run_dirs[K5DFI])
    views = {row: kw._view(root, args.seed) for row, root in roots.items()}
    dirs = {K5F: args.k5f_nexar_dir,
            **{row: roots[row] / zs.EVAL_DIR / NEXAR / row for row in (K5DF, K5DFI, K5DFIE)}}

    sets: dict[str, dict[str, Any]] = {}
    for name, control, new in CONTRASTS:
        pair_views = {kw.K5: views[control], new: views[new]}  # the readers key the control as K5
        pair_dirs = {kw.K5: dirs[control], new: dirs[new]}
        kw.check_checkpoints(pair_views, pair_dirs, args.seed, new=new)
        sets[name] = {NEXAR: kw.nexar_set(pair_dirs, new=new),
                      **{s: kw.zero_shot_set(pair_views, s, new=new) for s in kw.ZS_SETS}}
    sentences = {n: {s: c[s]["sentence"] for s in ALL_SETS} for n, c in sets.items()}
    labels = {n: {s: label(sets[n][s]) for s in ALL_SETS} for n in DECIDING}
    report: dict[str, Any] = {
        "rule": RULE, "seed": args.seed, "fixed_mul_mil": mul_mil, "start_check": start,
        "sets": sets, "sentences": sentences, "labels": labels,
        "decision_sets": list(DECISION_SETS),
        "verdicts": {n: verdict(labels[n]) for n in DECIDING},
        "candidates": {"K5dfi": candidate(sentences["ignore_vs_reference"]),
                       "K5dfie": candidate(sentences["ego_vs_reference"])},
        "false_alarms": {row: kn.false_alarms(kn.load_video_stats(d))["share"]
                         for row, d in dirs.items()},
    }
    stem = constants.KIM_DVS_IGNORE_REPORT_STEM
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    LOGGER.info("verdicts %s -> %s", report["verdicts"], args.out_dir / f"{stem}.md")
    return report


def _macro(r: dict[str, Any], set_name: str, row: str) -> str:
    """The row's macro on a set, from the first contrast that holds it."""
    for name, control, new in CONTRASTS:
        if row in (control, new):
            cells = r["sets"][name][set_name]["rows"][kw.K5 if row == control else new]
            m = cells["macro_crops"] if set_name == NEXAR else cells["macro"]
            return f"{m['mean']:.4f}"
    return "n/a"


def render(r: dict[str, Any]) -> str:
    v, cand, sc = r["verdicts"], r["candidates"], r["start_check"]

    def cand_line(row: str, key: str) -> str:
        c = cand[key]
        text = LINE_CANDIDATE if c["candidate"] else LINE_NOT_CANDIDATE
        return f"Candidate ({row}): {text} (above K5f on {c['above']}, below on {c['below']})."

    mask = sc["step1"][MASK_KEY]
    equal = ", ".join(f"{k} {e}" for k, e in sc["equal"].items())
    flag = "" if sc["same_draws"] else "**FLAG: not the same draws; no matched start.** "
    lines = [
        f"# {' / '.join(ROWS)}: DVS anchor mode and ego setting -- s{r['seed']}", "",
        f"Rule: `{r['rule']}`. PRELIMINARY, n = 1. Last-epoch mul_mil: "
        + ", ".join(f"{n} {x:.4f}" for n, x in r["fixed_mul_mil"].items()) + ".", "",
        f"**R4 primary ({K5DFI} - {K5DF}, `dvs_anchor_mode` ignore vs span): {v['ignore']}.** "
        f"Secondary ({K5DFIE} - {K5DFI}, `is_egocentric` true vs false): {v['ego']}.", "",
        cand_line(K5DFI, "K5dfi"), cand_line(K5DFIE, "K5dfie"), "",
        f"Start check ({K5DFI} vs {K5DF}, step 1): mask live ({MASK_KEY} {mask['K5df']:.4f} -> "
        f"{mask['K5dfi']:.4f}); same draws = {sc['same_draws']} ({equal}). {flag}"
        f"{K5DFIE} has no matched start (`theta_ego` / `delta_m_ego` change the splice draws).", "",
        "## Macro per row (Nexar = val crops)", "",
        "| set | " + " | ".join(ROWS) + " |", "|---" * (len(ROWS) + 1) + "|",
    ]
    for s in ALL_SETS:
        tag = "" if s in r["decision_sets"] else " (printed)"
        lines.append(f"| {s}{tag} | " + " | ".join(_macro(r, s, row) for row in ROWS) + " |")
    titles = [f"{new} - {control}" for _, control, new in CONTRASTS]
    lines += ["", "## Contrasts, paired per item, 95 % cluster bootstrap", "",
              "| set | " + " | ".join(titles) + " |", "|---" * (len(titles) + 1) + "|"]
    for s in ALL_SETS:
        lines.append(f"| {s} | " + " | ".join(_cell(r["sets"][name][s])
                                             for name, _, _ in CONTRASTS) + " |")
    gain_sets = list(constants.KIM_DVS_IGNORE_GAIN_SETS)
    lines += ["", f"## R4 labels (margin {constants.V5_MIN_GAIN}; GAIN needs one of {gain_sets}; "
              f"deciding sets {r['decision_sets']})", "",
              "| set | " + " | ".join(f"{n}" for n in DECIDING) + " |", "|---|---|---|"]
    for s in ALL_SETS:
        lines.append(f"| {s} | " + " | ".join(r["labels"][n][s] for n in DECIDING) + " |")
    fa = r["false_alarms"]
    lines += ["", "Printed: Nexar val negatives with peak > 0.5: "
              + ", ".join(f"{row} {fa[row]:.3f}" for row in fa) + ".", ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--kim-root", type=Path, required=True, help="kim_fixed/K5f")
    parser.add_argument("--k5f-nexar-dir", type=Path, required=True,
                        help="K5f's nexar_val read (kim_fixed/K5f/eval/nexar/K5f)")
    parser.add_argument("--df-root", type=Path, required=True,
                        help="kim_lneg_cap_fixed (holds K5df)")
    parser.add_argument("--ign-root", type=Path, required=True,
                        help="kim_dvs_ignore (holds K5dfi, K5dfie)")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
