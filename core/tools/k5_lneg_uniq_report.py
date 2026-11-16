"""K5Lnegu beside K5 and K5Lneg: which ingredient of ``L_neg`` costs (``KIM_RUN.md`` §28.3).

Pure aggregation over saved reads. Two paired contrasts per set (U1), reusing
``k5_whole_report``'s set readers (same items, one checkpoint per row, ``nexar_val`` only):
K5Lnegu - K5 and K5Lnegu - K5Lneg; K5Lneg - K5 is re-derived and must equal §28.1's read.
U2 turns the sentences on the four sets K5Lneg fell on into A / B / MIXED. Inputs::

    <kim-root>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/        K5
    <lneg-root>/eval/{dota,dada}/K5/, ..., eval/nexar/K5Lneg/          K5Lneg (§28)
    <uniq-root>/eval/{dota,dada}/K5/, ..., eval/nexar/K5Lnegu/         K5Lnegu (§28.3)
    <k5-nexar-dir>/        K5's nexar_val re-read with the same eval code (§17)
    <k5-reference-dir>/    §13's K5 read on nexar_val (NG4)

CLI::

    python -m core.tools.k5_lneg_uniq_report --kim-root outputs/v_final/kim \\
        --lneg-root outputs/v_final/kim_lneg --uniq-root outputs/v_final/kim_lneg_uniq \\
        --k5-nexar-dir outputs/v_final/kim_neg/eval/nexar/K5 \\
        --k5-reference-dir outputs/v_final/kim_t2/eval/nexar/K5 \\
        [--featdiv-json .../feature_diversity.json] --out-dir outputs/v_final/kim_lneg_uniq/REPORTS
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
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.zs_lagovad_readout import ABOVE, BELOW

LOGGER = logging.getLogger(__name__)

K5, LNEG, UNIQ, NEXAR = kw.K5, constants.KIM_LNEG_ROW, constants.KIM_LNEG_UNIQ_ROW, kw.NEXAR
RULE = "core/docs/final/KIM_RUN.md §28.3"
ALL_SETS = (NEXAR, *kw.ZS_SETS)


def verdict(vs_k5: dict[str, str], vs_lneg: dict[str, str]) -> dict[str, Any]:
    """U2 over ``KIM_LNEG_UNIQ_SETS``: A, B or MIXED from the two sentence maps."""
    sets = constants.KIM_LNEG_UNIQ_SETS
    below = [s for s in sets if vs_k5[s] == BELOW]
    recovered = [s for s in sets if vs_lneg[s] == ABOVE]
    if not below:
        call = constants.KIM_LNEG_UNIQ_A
    elif (len(below) >= constants.KIM_LNEG_UNIQ_B_MIN_BELOW
          and len(recovered) <= constants.KIM_LNEG_UNIQ_B_MAX_ABOVE):
        call = constants.KIM_LNEG_UNIQ_B
    else:
        call = constants.KIM_LNEG_UNIQ_MIXED
    return {"verdict": call, "below_k5": below, "above_lneg": recovered, "sets": list(sets)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    views = {K5: kw._view(args.kim_root, args.seed), LNEG: kw._view(args.lneg_root, args.seed),
             UNIQ: kw._view(args.uniq_root, args.seed)}
    dirs = {K5: args.k5_nexar_dir, LNEG: args.lneg_root / zs.EVAL_DIR / NEXAR / LNEG,
            UNIQ: args.uniq_root / zs.EVAL_DIR / NEXAR / UNIQ}
    repro = kn.reproduce_gate(dirs[K5], args.k5_reference_dir)
    for control, new in ((K5, LNEG), (K5, UNIQ), (LNEG, UNIQ)):
        kw.check_checkpoints({K5: views[control], new: views[new]},
                             {K5: dirs[control], new: dirs[new]}, args.seed, new=new)

    def contrast(control: str, new: str) -> dict[str, Any]:
        """``new - control`` per set; the readers key the control row as K5."""
        return {NEXAR: kw.nexar_set({K5: dirs[control], new: dirs[new]}, new=new),
                **{s: kw.zero_shot_set({K5: views[control], new: views[new]}, s, new=new)
                   for s in kw.ZS_SETS}}

    sets = {"uniq_vs_k5": contrast(K5, UNIQ), "uniq_vs_lneg": contrast(LNEG, UNIQ),
            "lneg_vs_k5": contrast(K5, LNEG)}
    sentences = {name: {s: c[s]["sentence"] for s in ALL_SETS} for name, c in sets.items()}
    report: dict[str, Any] = {
        "rule": RULE, "seed": args.seed, "repro": repro, "sets": sets, "sentences": sentences,
        **verdict(sentences["uniq_vs_k5"], sentences["uniq_vs_lneg"]),
        "feature_diversity": (json.loads(args.featdiv_json.read_text(encoding="utf-8"))
                              if args.featdiv_json is not None else None),
    }
    stem = constants.KIM_LNEG_UNIQ_REPORT_STEM
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    LOGGER.info("verdict %s -> %s", report["verdict"], args.out_dir / f"{stem}.md")
    return report


def _cell(c: dict[str, Any]) -> str:
    d = c["delta"]
    if d is None:
        return "n/a"
    return f"{d['mean']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}] {c['sentence']}"


def render(r: dict[str, Any]) -> str:
    lines = [
        f"# K5Lnegu (one meaningless target per video) vs K5 and K5Lneg -- s{r['seed']}", "",
        f"Rule: `{r['rule']}`. PRELIMINARY, n = 1. NG4: K5's nexar_val re-read "
        f"{r['repro']['reread']:.4f} vs §13 {r['repro']['reference']:.4f}.", "",
        f"**U2 verdict: {r['verdict']}** -- A iff K5Lnegu - K5 is below on none of "
        f"{r['sets']}; B iff below on >= {constants.KIM_LNEG_UNIQ_B_MIN_BELOW} and "
        f"K5Lnegu - K5Lneg above on <= {constants.KIM_LNEG_UNIQ_B_MAX_ABOVE}; else MIXED. "
        f"Below K5 on {r['below_k5']}; above K5Lneg on {r['above_lneg']}.", "",
        "| set (macro; Nexar = val crops) | K5Lnegu - K5 | K5Lnegu - K5Lneg | "
        "K5Lneg - K5 (= §28.1) |", "|---|---|---|---|",
    ]
    for s in ALL_SETS:
        tag = "" if s in r["sets"] else " (printed)"
        lines.append(f"| {s}{tag} | {_cell(r['sets']['uniq_vs_k5'][s])} | "
                     f"{_cell(r['sets']['uniq_vs_lneg'][s])} | "
                     f"{_cell(r['sets']['lneg_vs_k5'][s])} |")
    fd = r["feature_diversity"]
    if fd is not None:
        m = fd["metrics"]["sep"]
        lo, hi = m["delta_ci95"]
        lines += ["", f"Printed: P2 `sep` {fd['new']['name']} - {fd['base']['name']} = "
                  f"{m['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] ({m['base']:.4f} -> {m['new']:.4f}); "
                  f"FD6 {fd['verdict']}."]
    lines += ["", "A: the shared class target is what costs; good per-video captions *may* help "
              "(uniq is the extreme: real captions are correlated). B: the pooled-feature "
              "objective costs whatever the text; per-video captions under this loss form are "
              "closed. MIXED: no claim.", ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--kim-root", type=Path, required=True, help="outputs/kim (K5)")
    parser.add_argument("--lneg-root", type=Path, required=True, help="outputs/kim_lneg (§28)")
    parser.add_argument("--uniq-root", type=Path, required=True,
                        help="outputs/kim_lneg_uniq (§28.3)")
    parser.add_argument("--k5-nexar-dir", type=Path, required=True,
                        help="K5 nexar_val read, current eval code (kim_neg/eval/nexar/K5)")
    parser.add_argument("--k5-reference-dir", type=Path, required=True,
                        help="§13's K5 read on nexar_val (NG4)")
    parser.add_argument("--featdiv-json", type=Path, default=None,
                        help="feature_diversity.json of K5Lnegu vs K5 (printed only)")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
