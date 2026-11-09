"""K5 with windows from Nexar's negative videos (K5neg) beside K5 (``KIM_RUN.md`` §17).

Pure aggregation over saved reads, with ``k5_whole_report``'s gates (same items, micro equal to
its read, one checkpoint per row, rows on different checkpoints, ``nexar_val`` only), plus §17
NG4: K5's re-read on ``nexar_val`` must reproduce its §13 crop endpoint. Inputs::

    <kim-root>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/     K5
    <neg-root>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/     K5neg
    <neg-root>/eval/nexar/{K5,K5neg}/                               nexar_eval + video_stats.json
    <k5-reference-dir>/                                             §13's K5 read on nexar_val

CLI::

    python -m core.tools.k5_neg_report --kim-root outputs/v_final/kim \\
        --neg-root outputs/v_final/kim_neg \\
        --k5-reference-dir outputs/v_final/kim_t2/eval/nexar/K5 \\
        --out-dir outputs/v_final/kim_neg/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools import k5_whole_report as kw
from core.tools import k5_zs_report as zs
from core.tools import nexar_eval, zs_compare_table
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.zs_lagovad_readout import ABOVE, BELOW

LOGGER = logging.getLogger(__name__)

REPORT_STEM = "k5_neg_report"
K5, NEG, NEXAR = kw.K5, constants.KIM_NEG_ROW, kw.NEXAR
ROWS = (K5, NEG)
ROW_NAME = {K5: "K5", NEG: "K5neg"}
SCENES = ("Urban", "Sub-urban", "Highway")
OTHER_SCENE = "other"
NOTES = """## How to read this

* **PRELIMINARY (n = 1 seed, s2024)** (`KIM_RUN.md` §17). One factor changes: the normal pool
  (windows from the 449 negative videos added, ~55 % of the pool). K5 stays the main model unless
  NG7 reads "candidate" and the user decides.
* **False alarms are printed, never gated** (NG5). A drop concentrated in one scene (Sub-urban
  has 41 positive vs 122 negative training videos) may be a scene shortcut, not detection.
* **Nexar val is in-domain**; crops are the endpoint, whole-video macro is a position test.
  Never beside the `nexar_test` table of `RESULTS_NEXAR.md`.
* **Micro on TAD and DAD measures clip classification. DAD macro is not localization**
  (`ZS_DAD.md` §8).
"""


def reproduce_gate(reread_dir: Path, reference_dir: Path) -> dict[str, float]:
    """NG4: K5's re-read crop endpoint equals its §13 read within ``KIM_NEG_REPRO_TOL``."""
    got = kw.nexar_read(reread_dir)[0]["crops"]["mean_over_placements"]["macro"]["mean"]
    ref = kw.nexar_read(reference_dir)[0]["crops"]["mean_over_placements"]["macro"]["mean"]
    gap = float(got) - float(ref)
    if abs(gap) > constants.KIM_NEG_REPRO_TOL:
        raise ValueError(f"K5's nexar_val re-read {got:.6f} != its §13 read {ref:.6f} (NG4)")
    return {"reread": float(got), "reference": float(ref), "gap": gap}


def load_video_stats(row_dir: Path) -> dict[str, dict[str, Any]]:
    path = row_dir / nexar_eval.VIDEO_STATS_JSON
    if not path.is_file():
        raise FileNotFoundError(f"{path}: re-run nexar_eval with this code (§17 NG4)")
    stats: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return stats


def _scene(name: str) -> str:
    return name if name in SCENES else OTHER_SCENE


def false_alarms(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """NG5 (a)(b): over the split's negative videos, share with peak > ``NORMAL_VIDEO_PEAK``."""
    negatives = {v: s for v, s in stats.items() if s["label"] == 0}
    if not negatives:
        raise ValueError("nexar: no negative video in the read")

    def share(vs: list[dict[str, Any]]) -> float:
        return float(np.mean([s["peak"] > constants.NORMAL_VIDEO_PEAK for s in vs]))

    by_scene: dict[str, dict[str, Any]] = {}
    for scene in (*SCENES, OTHER_SCENE):
        group = [s for s in negatives.values() if _scene(s["scene"]) == scene]
        by_scene[scene] = {"videos": len(group), "share": share(group) if group else None}
    return {"videos": len(negatives), "share": share(list(negatives.values())),
            "mean_score": float(np.mean([s["mean"] for s in negatives.values()])),
            "by_scene": by_scene}


def nexar_range(stats: dict[str, dict[str, Any]]) -> float:
    """NG5 (c) on Nexar val: median within-video score range over every val video."""
    return float(np.median([s["range"] for s in stats.values()]))


def zero_shot_range(view: argparse.Namespace, set_name: str) -> float:
    """NG5 (c): median within-video range (max - min) of the saved raw scores of one set."""
    loaded = zs_compare_table.load_scores(zs.score_source(view, set_name, K5))
    if loaded is None:
        raise FileNotFoundError(f"{set_name}: no scores under {view.kim_root}")
    scores = loaded[0]
    return float(np.median([np.ptp(np.asarray(scores[v], dtype=float)) for v in scores]))


def follow_up(report: dict[str, Any]) -> dict[str, Any]:
    """NG7: candidate iff no set below and (a set above or false alarms fall by the margin)."""
    sentences = {name: s["sentence"] for name, s in report["sets"].items()}
    below = [n for n, s in sentences.items() if s == BELOW]
    above = [n for n, s in sentences.items() if s == ABOVE]
    fa = report["false_alarms"]
    fa_drop = fa[K5]["share"] - fa[NEG]["share"]
    if below:
        verdict = "K5 stays"
    elif above or fa_drop >= constants.KIM_NEG_FA_MARGIN:
        verdict = "candidate"
    else:
        verdict = "no claim"
    return {"below": below, "above": above, "fa_drop": fa_drop, "verdict": verdict}


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def render(report: dict[str, Any]) -> str:
    """The comparison as one markdown document."""
    n = report["sets"][NEXAR]
    places = constants.NEXAR_CROP_PLACEMENTS
    lines = [
        f"# K5 with windows from Nexar's negative videos vs K5 — **{kw.LABEL}**",
        "",
        f"Seed {report['seed']}. Rule `KIM_RUN.md` §17, written before these numbers. Generated by "
        "`core.tools.k5_neg_report`. `nexar_test` is not read. NG4: K5's re-read "
        f"{report['repro']['reread']:.4f} vs §13 {report['repro']['reference']:.4f} "
        f"(gap {report['repro']['gap']:+.1e}).",
        "",
        f"## Nexar val ({n['videos']} videos, in-domain)",
        "",
        "| Row | crops (endpoint) | " + " | ".join(f"@{p}" for p in places)
        + " | whole macro | micro | AP | clip AUC |",
        "|---|---|" + "---|" * len(places) + "---|---|---|---|",
    ]
    for row in ROWS:
        c = n["rows"][row]
        lines.append(
            f"| {ROW_NAME[row]} | {kw._macro(c['macro_crops'])} | "
            + " | ".join(f"{c['placements'][str(p)]:.4f}" for p in places)
            + f" | {kw._macro(c['macro_whole'])} | {_num(c['micro'])} | {_num(c['ap'])} | "
            f"{_num(c['clip_auc'])} |")
    fa = report["false_alarms"]
    scenes = (*SCENES, OTHER_SCENE)
    lines += [
        "",
        f"## False alarms on Nexar val's {fa[K5]['videos']} negative videos (NG5, printed)",
        "",
        f"Share of negative videos whose peak score > {constants.NORMAL_VIDEO_PEAK}.",
        "",
        "| Row | all | mean score | " + " | ".join(
            f"{s} (n = {fa[K5]['by_scene'][s]['videos']})" for s in scenes) + " |",
        "|---|---|---|" + "---|" * len(scenes),
    ]
    for row in ROWS:
        f = fa[row]
        lines.append(f"| {ROW_NAME[row]} | {f['share']:.4f} | {f['mean_score']:.4f} | "
                     + " | ".join(_num(f["by_scene"][s]["share"]) for s in scenes) + " |")
    lines += ["", "## Zero-shot sets", "",
              "| Set | Metric | K5 | K5neg | Base rate |", "|---|---|---|---|---|"]
    for name in kw.ZS_SETS:
        s = report["sets"][name]
        r = s["rows"]
        lines += [
            f"| **{kw.SET_NAME[name]}** | AUC macro | {kw._macro(r[K5]['macro'])} | "
            f"{kw._macro(r[NEG]['macro'])} | |",
            f"| | AUC micro | {r[K5]['micro']:.4f} | {r[NEG]['micro']:.4f} | |",
            f"| | AP | {r[K5]['ap']:.4f} | {r[NEG]['ap']:.4f} | {s['base_rate']:.4f} |",
        ]
    lines += ["", "## Median within-video score range (NG5 c, flat-curve check)", "",
              "| Row | " + " | ".join(kw.SET_NAME[s] for s in report["ranges"][K5]) + " |",
              "|---|" + "---|" * len(report["ranges"][K5])]
    for row in ROWS:
        lines.append(f"| {ROW_NAME[row]} | "
                     + " | ".join(f"{v:.3f}" for v in report["ranges"][row].values()) + " |")
    lines += ["", "## K5neg - K5, macro, paired per item (§17 NG6)", "",
              "| Set | Δ [95 % CI] | Sentence |", "|---|---|---|"]
    for name, s in report["sets"].items():
        what = "crops" if name == NEXAR else "macro"
        lines.append(f"| {kw.SET_NAME[name]} ({what}) | {kw._signed(s['delta'])} | "
                     f"{s['sentence']} |")
    f = report["follow_up"]
    lines += [
        "",
        f"**§17 NG7:** below = {f['below'] or 'none'}; above = {f['above'] or 'none'}; "
        f"false-alarm drop (K5 - K5neg) {f['fa_drop']:+.4f} (margin "
        f"{constants.KIM_NEG_FA_MARGIN}). Verdict: **{f['verdict']}**.",
        "",
    ]
    return "\n".join(lines) + "\n" + NOTES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--kim-root", type=Path, required=True, help="outputs/kim (K5)")
    parser.add_argument("--neg-root", type=Path, required=True, help="outputs/kim_neg")
    parser.add_argument("--k5-reference-dir", type=Path, required=True,
                        help="§13's K5 nexar_val read (kim_t2/eval/nexar/K5)")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    views = {K5: kw._view(args.kim_root, args.seed), NEG: kw._view(args.neg_root, args.seed)}
    nexar_root = args.neg_root / zs.EVAL_DIR / NEXAR
    dirs = {K5: nexar_root / K5, NEG: nexar_root / NEG}
    repro = reproduce_gate(dirs[K5], args.k5_reference_dir)
    kw.check_checkpoints(views, dirs, args.seed, new=NEG)
    sets = {NEXAR: kw.nexar_set(dirs, new=NEG),
            **{s: kw.zero_shot_set(views, s, new=NEG) for s in kw.ZS_SETS}}
    stats = {row: load_video_stats(dirs[row]) for row in ROWS}
    if set(stats[K5]) != set(stats[NEG]):
        raise ValueError("nexar: the two rows' video_stats cover different videos")
    report: dict[str, Any] = {
        "label": kw.LABEL, "seed": args.seed, "repro": repro, "sets": sets,
        "false_alarms": {row: false_alarms(stats[row]) for row in ROWS},
        "ranges": {row: {NEXAR: nexar_range(stats[row]),
                         **{s: zero_shot_range(views[row], s) for s in kw.ZS_SETS}}
                   for row in ROWS},
    }
    report["follow_up"] = follow_up(report)
    write_json_atomic(args.out_dir / f"{REPORT_STEM}.json", report)
    write_text_atomic(args.out_dir / f"{REPORT_STEM}.md", render(report))
    LOGGER.info("report -> %s", args.out_dir / f"{REPORT_STEM}.md")
    return report


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
