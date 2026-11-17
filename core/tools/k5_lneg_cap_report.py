"""K5 / K5d / K5c / K5dc: do caption ``L_neg`` and DVS complement each other? (``KIM_RUN.md`` §30).

Pure aggregation over saved reads, PRELIMINARY (n = 1). Per set (C5): K5c - K5 (``L_neg``
alone), K5d - K5 (DVS alone), K5dc - K5 (both), K5dc - K5d (``L_neg`` with DVS), each paired per
item with ``k5_whole_report``'s readers, and the interaction I = (K5dc - K5d) - (K5c - K5) per
item over the items all four rows scored, cluster-bootstrapped (DoTA: source video). C6 turns
the I sentences on S = Nexar / DoTA-CAP-dev / DADA / DAD into COMPLEMENTARY / ANTAGONISTIC /
NO CLAIM; C7 says whether K5dc is a candidate. Inputs::

    <kim-root>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/      K5
    <k5-nexar-dir>/                                                  K5's nexar_val re-read (§17)
    <k5-reference-dir>/                                              §13's K5 read (NG4)
    <cap-root>/<ROW>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/, eval/nexar/<ROW>/
        for ROW in K5d, K5c, K5dc

CLI::

    python -m core.tools.k5_lneg_cap_report --kim-root outputs/v_final/kim \\
        --cap-root outputs/v_final/kim_lneg_cap \\
        --k5-nexar-dir outputs/v_final/kim_neg/eval/nexar/K5 \\
        --k5-reference-dir outputs/v_final/kim_t2/eval/nexar/K5 \\
        --out-dir outputs/v_final/kim_lneg_cap/REPORTS
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.metrics import cluster_bootstrap_ci
from core.tools import k5_neg_report as kn
from core.tools import k5_whole_report as kw
from core.tools import zs_compare_table as zs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.zs_lagovad_readout import ABOVE, BELOW, versus_sentence

LOGGER = logging.getLogger(__name__)

K5, NEXAR = kw.K5, kw.NEXAR
K5D, K5C, K5DC = (constants.KIM_DVS_ARM, constants.KIM_LNEG_CAP_ROW,
                  constants.KIM_LNEG_CAP_DVS_ROW)
NEW_ROWS = (K5D, K5C, K5DC)
RULE = "core/docs/final/KIM_RUN.md §30"
ALL_SETS = (NEXAR, *kw.ZS_SETS)
# (name, control, new): the C5 contrasts, new - control
CONTRASTS = (("lneg_alone", K5, K5C), ("dvs_alone", K5, K5D), ("both", K5, K5DC),
             ("lneg_with_dvs", K5D, K5DC))
CONTRAST_TITLE = {"lneg_alone": "K5c - K5", "dvs_alone": "K5d - K5", "both": "K5dc - K5",
                  "lneg_with_dvs": "K5dc - K5d", "interaction": "I"}


def verdict(interaction: dict[str, str], both: dict[str, str]) -> dict[str, Any]:
    """C6 on the interaction sentences, C7 on K5dc - K5, both over ``KIM_LNEG_CAP_SETS``."""
    sets = constants.KIM_LNEG_CAP_SETS
    above = [s for s in sets if interaction[s] == ABOVE]
    below = [s for s in sets if interaction[s] == BELOW]
    if len(above) >= constants.KIM_LNEG_CAP_MIN_SETS and not below:
        call = constants.KIM_LNEG_CAP_COMPLEMENTARY
    elif len(below) >= constants.KIM_LNEG_CAP_MIN_SETS:
        call = constants.KIM_LNEG_CAP_ANTAGONISTIC
    else:
        call = constants.KIM_LNEG_CAP_NO_CLAIM
    both_above = [s for s in sets if both[s] == ABOVE]
    both_below = [s for s in sets if both[s] == BELOW]
    return {"verdict": call, "interaction_above": above, "interaction_below": below,
            "decision_sets": list(sets), "k5dc_candidate": bool(both_above) and not both_below,
            "k5dc_above_k5": both_above, "k5dc_below_k5": both_below}


def interaction(items: dict[str, dict[str, float]], groups: dict[str, str]) -> dict[str, Any]:
    """Per item ``(K5dc - K5d) - (K5c - K5)`` over the items all four rows scored."""
    common = sorted(set.intersection(*(set(items[r]) for r in (K5, *NEW_ROWS))))
    if not common:
        raise ValueError("the four rows share no scored item")
    values = {v: (items[K5DC][v] - items[K5D][v]) - (items[K5C][v] - items[K5][v])
              for v in common}
    delta = cluster_bootstrap_ci(values, {v: groups.get(v, v) for v in common},
                                 constants.V2_E1_BOOTSTRAP, constants.V5_BOOTSTRAP_SEED,
                                 constants.V2_E1_CI)
    return {"delta": delta, "sentence": versus_sentence(delta), "items": len(common)}


def _set_items(set_name: str, views: dict[str, argparse.Namespace],
               dirs: dict[str, Path]) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Per-item macro inputs of every row of one set, and the clusters (K5's read)."""
    if set_name == NEXAR:
        items = {row: kw.nexar_read(d)[1] for row, d in dirs.items()}
        return items, {}
    reads = {row: kw.zero_shot_row(view, set_name)[0] for row, view in views.items()}
    return {row: r["items"] for row, r in reads.items()}, dict(reads[K5]["groups"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    views = {K5: kw._view(args.kim_root, args.seed),
             **{row: kw._view(args.cap_root / row, args.seed) for row in NEW_ROWS}}
    dirs = {K5: args.k5_nexar_dir,
            **{row: args.cap_root / row / zs.EVAL_DIR / NEXAR / row for row in NEW_ROWS}}
    repro = kn.reproduce_gate(dirs[K5], args.k5_reference_dir)
    for name, control, new in CONTRASTS:
        kw.check_checkpoints({K5: views[control], new: views[new]},
                             {K5: dirs[control], new: dirs[new]}, args.seed, new=new)
        LOGGER.info("checkpoints distinct: %s", name)

    def contrast(control: str, new: str) -> dict[str, Any]:
        """``new - control`` per set; the readers key the control row as K5."""
        return {NEXAR: kw.nexar_set({K5: dirs[control], new: dirs[new]}, new=new),
                **{s: kw.zero_shot_set({K5: views[control], new: views[new]}, s, new=new)
                   for s in kw.ZS_SETS}}

    sets = {name: contrast(control, new) for name, control, new in CONTRASTS}
    sets["interaction"] = {s: interaction(*_set_items(s, views, dirs)) for s in ALL_SETS}
    sentences = {name: {s: c[s]["sentence"] for s in ALL_SETS} for name, c in sets.items()}
    report: dict[str, Any] = {
        "rule": RULE, "seed": args.seed, "repro": repro, "sets": sets, "sentences": sentences,
        "false_alarms": {row: kn.false_alarms(kn.load_video_stats(d))["share"]
                         for row, d in dirs.items()},
        **verdict(sentences["interaction"], sentences["both"]),
    }
    stem = constants.KIM_LNEG_CAP_REPORT_STEM
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    LOGGER.info("verdict %s -> %s", report["verdict"], args.out_dir / f"{stem}.md")
    return report


def _cell(c: dict[str, Any]) -> str:
    d = c["delta"]
    if d is None:
        return "n/a"
    return f"{d['mean']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}] {c['sentence']}"


def _macro(r: dict[str, Any], set_name: str, row: str) -> str:
    """The row's macro on a set, from the contrast that holds it."""
    for name, control, new in CONTRASTS:
        if row in (control, new):
            rows = r["sets"][name][set_name]["rows"]
            key = K5 if row == control else new
            cells = rows[key]
            m = cells["macro_crops"] if set_name == NEXAR else cells["macro"]
            return f"{m['mean']:.4f}"
    return "n/a"


def render(r: dict[str, Any]) -> str:
    rows = (K5, *NEW_ROWS)
    names = ("lneg_alone", "dvs_alone", "both", "lneg_with_dvs", "interaction")
    lines = [
        f"# K5 / K5d / K5c / K5dc: caption L_neg x DVS -- s{r['seed']}", "",
        f"Rule: `{r['rule']}`. PRELIMINARY, n = 1. Captions: §29.3 prompt v6, Qwen2.5-VL-72B "
        "NF4, 32 frames, **QC waived by the user** (2026-11-17). NG4: K5's nexar_val re-read "
        f"{r['repro']['reread']:.4f} vs §13 {r['repro']['reference']:.4f}.", "",
        f"**C6 verdict: {r['verdict']}** -- COMPLEMENTARY iff I is above on >= "
        f"{constants.KIM_LNEG_CAP_MIN_SETS} of {r['decision_sets']} and below on none; "
        f"ANTAGONISTIC iff below on >= {constants.KIM_LNEG_CAP_MIN_SETS}; else NO CLAIM. "
        f"I above on {r['interaction_above']}, below on {r['interaction_below']}.", "",
        f"**C7: K5dc candidate = {r['k5dc_candidate']}** (K5dc - K5 above on "
        f"{r['k5dc_above_k5']}, below on {r['k5dc_below_k5']}). K5 stays unless this is True "
        "and the user decides.", "",
        "## Macro per row (Nexar = val crops)", "",
        "| set | " + " | ".join(rows) + " |", "|---" * (len(rows) + 1) + "|",
    ]
    for s in ALL_SETS:
        tag = "" if s in r["decision_sets"] else " (printed)"
        lines.append(f"| {s}{tag} | " + " | ".join(_macro(r, s, row) for row in rows) + " |")
    lines += ["", "## Contrasts, paired per item, 95 % cluster bootstrap", "",
              "| set | " + " | ".join(CONTRAST_TITLE[n] for n in names) + " |",
              "|---" * (len(names) + 1) + "|"]
    for s in ALL_SETS:
        lines.append(f"| {s} | " + " | ".join(_cell(r["sets"][n][s]) for n in names) + " |")
    fa = r["false_alarms"]
    lines += ["", "Printed: Nexar val negatives with peak > 0.5: "
              + ", ".join(f"{row} {fa[row]:.3f}" for row in rows) + ".",
              "", "Direction only, for comparison: LaGoVAD Tab. G Det. Avg. (one run, PreVAD): "
              "L_neg +2.46 with DVS, +2.20 without; DVS +2.91 with L_neg, +2.65 without; "
              "interaction about +0.26.", ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--kim-root", type=Path, required=True, help="outputs/kim (K5)")
    parser.add_argument("--cap-root", type=Path, required=True,
                        help="outputs/kim_lneg_cap (K5d, K5c, K5dc, one folder each)")
    parser.add_argument("--k5-nexar-dir", type=Path, required=True,
                        help="K5 nexar_val read, current eval code (kim_neg/eval/nexar/K5)")
    parser.add_argument("--k5-reference-dir", type=Path, required=True,
                        help="§13's K5 read on nexar_val (NG4)")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
