"""K5cfn: does the dashcam Normal caption change K5cf? (``KIM_RUN.md`` §33, §32.1 A7).

Last arm of the caption ``L_neg`` line. Pure aggregation over saved reads, PRELIMINARY (n = 1).
Per set: K5cfn - K5cf (primary: only row 0 of the caption set differs) and K5cfn - K5f, each
paired per item with ``k5_whole_report``'s readers. R4: K5cfn is a candidate iff K5cfn - K5f is
above on >= 1 set of ``KIM_LNEG_CAP_SETS`` and below on none; otherwise the line closes with no
candidate. R5 guards (raise): every row's last-epoch ``mul_mil`` above
``KIM_FIXED_MUL_MIL_FLOOR``, and K5cfn's ``cap_same_source_neg`` equal to K5cf's at every step
(same seed, same batches). Inputs::

    <kim-root>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/, K5/s<seed>/   K5f
    <k5f-nexar-dir>/                                                          K5f's nexar_val read
    <cap-root>/<ROW>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/, eval/nexar/<ROW>/,
        K5/s<seed>/  for ROW in K5cf, K5cfn

CLI::

    python -m core.tools.k5_a7_report --kim-root outputs/v_final/kim_fixed/K5f \\
        --cap-root outputs/v_final/kim_lneg_cap_fixed \\
        --k5f-nexar-dir outputs/v_final/kim_fixed/K5f/eval/nexar/K5f \\
        --out-dir outputs/v_final/kim_lneg_cap_fixed/REPORTS
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

RULE = "core/docs/final/KIM_RUN.md §33"
K5F = constants.KIM_REFERENCE_ARM
K5CF = constants.KIM_FIXED_CAP_ROW
K5CFN = constants.KIM_FIXED_CAP_DASHCAM_ROW
ROWS = (K5F, K5CF, K5CFN)
NEXAR = kw.NEXAR
ALL_SETS = (NEXAR, *kw.ZS_SETS)
GUARD_KEY = "cap_same_source_neg"
# (name, control, new): new - control
CONTRASTS = (("normal_caption", K5CF, K5CFN), ("vs_reference", K5F, K5CFN))
LINE_CLOSED = "CLOSED"
LINE_CANDIDATE = "CANDIDATE (user decides; K5f stays until then)"


def verdict(primary: dict[str, str], vs_reference: dict[str, str]) -> dict[str, Any]:
    """R4 on K5cfn - K5f over ``KIM_LNEG_CAP_SETS``; the primary contrast decides nothing."""
    sets = constants.KIM_LNEG_CAP_SETS
    above = [s for s in sets if vs_reference[s] == ABOVE]
    below = [s for s in sets if vs_reference[s] == BELOW]
    candidate = bool(above) and not below
    return {"candidate": candidate, "line": LINE_CANDIDATE if candidate else LINE_CLOSED,
            "decision_sets": list(sets), "above_reference": above, "below_reference": below,
            "normal_caption_above": [s for s in sets if primary[s] == ABOVE],
            "normal_caption_below": [s for s in sets if primary[s] == BELOW]}


def _per_step(metrics: Path, key: str) -> dict[int, float]:
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return {int(r["global_step"]): float(r[key]) for r in rows if key in r}


def check_same_batches(control_run: Path, new_run: Path) -> int:
    """R5: ``cap_same_source_neg`` identical at every step (it depends on the batch only)."""
    a = _per_step(control_run / METRICS_FILENAME, GUARD_KEY)
    b = _per_step(new_run / METRICS_FILENAME, GUARD_KEY)
    if not a or set(a) != set(b):
        raise ValueError(f"{GUARD_KEY}: steps differ or are missing ({len(a)} vs {len(b)})")
    diff = [s for s in sorted(a) if a[s] != b[s]]
    if diff:
        raise ValueError(f"{GUARD_KEY} differs at {len(diff)} steps (first {diff[0]}): the two "
                         "runs did not see the same batches (§33 R5)")
    return len(a)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = f"s{args.seed}"
    run_dirs = {K5F: args.kim_root / kw.K5 / run_root,
                **{row: args.cap_root / row / kw.K5 / run_root for row in (K5CF, K5CFN)}}
    mul_mil = check_fixed_runs(run_dirs)
    steps = check_same_batches(run_dirs[K5CF], run_dirs[K5CFN])
    views = {K5F: kw._view(args.kim_root, args.seed),
             **{row: kw._view(args.cap_root / row, args.seed) for row in (K5CF, K5CFN)}}
    dirs = {K5F: args.k5f_nexar_dir,
            **{row: args.cap_root / row / zs.EVAL_DIR / NEXAR / row for row in (K5CF, K5CFN)}}

    sets: dict[str, dict[str, Any]] = {}
    for name, control, new in CONTRASTS:
        # the readers key the control row as K5
        pair_views = {kw.K5: views[control], new: views[new]}
        pair_dirs = {kw.K5: dirs[control], new: dirs[new]}
        kw.check_checkpoints(pair_views, pair_dirs, args.seed, new=new)
        sets[name] = {NEXAR: kw.nexar_set(pair_dirs, new=new),
                      **{s: kw.zero_shot_set(pair_views, s, new=new) for s in kw.ZS_SETS}}
    sentences = {name: {s: c[s]["sentence"] for s in ALL_SETS} for name, c in sets.items()}
    report: dict[str, Any] = {
        "rule": RULE, "seed": args.seed, "fixed_mul_mil": mul_mil, "same_batch_steps": steps,
        "normal_caption_text": constants.NORMAL_CAPTION_DASHCAM_TEXT,
        "sets": sets, "sentences": sentences,
        "false_alarms": {row: kn.false_alarms(kn.load_video_stats(d))["share"]
                         for row, d in dirs.items()},
        **verdict(sentences["normal_caption"], sentences["vs_reference"]),
    }
    stem = constants.KIM_A7_REPORT_STEM
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    LOGGER.info("line %s -> %s", report["line"], args.out_dir / f"{stem}.md")
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
    titles = [f"{new} - {control}" for _, control, new in CONTRASTS]
    lines = [
        f"# {' / '.join(ROWS)}: dashcam Normal caption -- s{r['seed']}", "",
        f"Rule: `{r['rule']}`. PRELIMINARY, n = 1. Row 0 = \"{r['normal_caption_text']}\". "
        "Last-epoch mul_mil: " + ", ".join(f"{n} {v:.4f}" for n, v in r["fixed_mul_mil"].items())
        + f"; {GUARD_KEY} equal to {K5CF}'s at all {r['same_batch_steps']} steps.", "",
        f"**R4: {K5CFN} candidate = {r['candidate']} -> caption L_neg line {r['line']}** "
        f"({K5CFN} - {K5F} above on {r['above_reference']}, below on {r['below_reference']}; "
        f"sets {r['decision_sets']}).", "",
        f"Primary ({K5CFN} - {K5CF}, decides nothing): the Normal sentence is above on "
        f"{r['normal_caption_above']}, below on {r['normal_caption_below']}.", "",
        "## Macro per row (Nexar = val crops)", "",
        "| set | " + " | ".join(ROWS) + " |", "|---" * (len(ROWS) + 1) + "|",
    ]
    for s in ALL_SETS:
        tag = "" if s in r["decision_sets"] else " (printed)"
        lines.append(f"| {s}{tag} | " + " | ".join(_macro(r, s, row) for row in ROWS) + " |")
    lines += ["", "## Contrasts, paired per item, 95 % cluster bootstrap", "",
              "| set | " + " | ".join(titles) + " |", "|---" * (len(titles) + 1) + "|"]
    for s in ALL_SETS:
        lines.append(f"| {s} | " + " | ".join(_cell(r["sets"][name][s])
                                             for name, _, _ in CONTRASTS) + " |")
    fa = r["false_alarms"]
    lines += ["", "Printed: Nexar val negatives with peak > 0.5: "
              + ", ".join(f"{row} {fa[row]:.3f}" for row in ROWS) + ".", ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--kim-root", type=Path, required=True, help="kim_fixed/K5f")
    parser.add_argument("--cap-root", type=Path, required=True,
                        help="kim_lneg_cap_fixed (K5cf, K5cfn, one folder each)")
    parser.add_argument("--k5f-nexar-dir", type=Path, required=True,
                        help="K5f's nexar_val read (kim_fixed/K5f/eval/nexar/K5f)")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
