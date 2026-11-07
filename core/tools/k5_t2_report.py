"""K5 trained on T2 beside K5 on Nexar and LaGoVAD's public checkpoint (``KIM_RUN.md`` §13).

Printed, decides nothing. Pure aggregation over saved reads, with ``k5_zs_report``'s gates: same
items per set, micro equal to its read, one checkpoint per K5 row, Gate T0 on TAD. It also refuses
when T2's reads came from the Nexar checkpoint, or a Nexar read is not on ``nexar_val``. Inputs::

    <lagovad-root>/eval/{dota,tad,dad}/LaGoVAD/...    public checkpoint (zs_lagovad)
    <kim-root>/eval/{dota,tad,dad}/K5/...             K5 trained on Nexar
    <t2-root>/eval/{dota,tad,dad}/K5/...              K5 trained on T2
    <t2-root>/eval/nexar/{LaGoVAD,K5,K5T2}/           nexar_eval on nexar_val

CLI::

    python -m core.tools.k5_t2_report --lagovad-root outputs/v_final/zs_lagovad \\
        --kim-root outputs/v_final/kim --t2-root outputs/v_final/kim_t2 \\
        --out-dir outputs/v_final/kim_t2/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools import k5_zs_report as zs
from core.tools import nexar_eval, tadz_readout
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.stp_readout import paired
from core.tools.zs_lagovad_readout import delta_ci, t0_check, versus_sentence

LOGGER = logging.getLogger(__name__)

REPORT_STEM = "k5_t2_report"
LABEL = "PRELIMINARY (n = 1 seed)"
LAG, K5, K5T2 = constants.ZS_LAGOVAD_ARM, constants.KIM_MAIN_ARM, constants.KIM_T2_ROW
ROWS = (LAG, K5, K5T2)
ROW_NAME = {LAG: "LaGoVAD (public ckpt)", K5: "K5 (Nexar)", K5T2: "K5 (T2)"}
ZS_SETS = (zs.DOTA, zs.TAD, zs.DAD)
NEXAR = "nexar"
SET_NAME = {NEXAR: "Nexar val", **zs.SET_NAME}
NORMAL_PEAK = 0.5  # a normal video whose peak passes this counts as a false alarm in the print
CATEGORY_SUFFIX = re.compile(r"_\d+$")
NOTES = """## How to read this

* **PRELIMINARY (n = 1 seed, s2024). Printed, decides nothing** (`KIM_RUN.md` §13). K5 on Nexar
  stays the main model. No row may be called adopted.
* **Not matched.** LaGoVAD trained on PreVAD; K5 on Nexar or T2. K5 (T2) - K5 (Nexar) changes the
  corpus and the input statistics together, so it never names one cause. T2 is dashcam too: it
  does not test the fixed-camera explanation on TAD.
* **Nexar val:** K5 (Nexar) is in-domain, the other two rows are zero-shot. Macro on whole videos
  is a position test (the event sits near the middle of every video); crops are the endpoint.
  These numbers never sit beside the `nexar_test` table of `RESULTS_NEXAR.md`.
* **Micro on TAD and DAD measures clip classification. DAD macro is not localization**
  (`ZS_DAD.md` §8): read its ruler, placebo and same-position lines.
"""

Interval = dict[str, float]


def _view(args: argparse.Namespace, kim_root: Path) -> argparse.Namespace:
    """The arguments ``k5_zs_report`` reads, with K5's reads under ``kim_root``."""
    return argparse.Namespace(lagovad_root=args.lagovad_root, kim_root=kim_root, seed=args.seed)


def _sided(delta: Interval | None) -> dict[str, Any]:
    return {"delta": delta, "sentence": versus_sentence(delta)}


def zero_shot_set(
    nexar_view: argparse.Namespace, t2_view: argparse.Namespace, set_name: str
) -> dict[str, Any]:
    """The three rows of a set, with ``k5_zs_report``'s gates applied to each K5 read."""
    on_nexar = zs.set_report(nexar_view, set_name)
    on_t2 = zs.set_report(t2_view, set_name)
    read_t2 = zs.read_row(t2_view, set_name, K5)
    read_nexar = zs.read_row(nexar_view, set_name, K5)
    if set(read_t2["items"]) != set(read_nexar["items"]):
        raise ValueError(f"{set_name}: K5 (T2) and K5 (Nexar) scored different items")
    return {
        "rows": {LAG: on_t2["rows"][LAG], K5: on_nexar["rows"][K5], K5T2: on_t2["rows"][K5]},
        "base_rate": on_t2["base_rate"],
        "protocol": on_t2["protocol"],
        "clip_oracle_micro": on_t2["clip_oracle_micro"],
        "t2_minus_lagovad": _sided(on_t2["delta_macro"]),
        "t2_minus_nexar": _sided(delta_ci(read_t2, read_nexar)),
    }


def _nexar_read(row_dir: Path) -> tuple[dict[str, Any], dict[str, float]]:
    readout = json.loads((row_dir / nexar_eval.READOUT_JSON).read_text(encoding="utf-8"))
    if readout["split"] != constants.V2_SPLIT_NEXAR_VAL:
        raise ValueError(f"{row_dir}: read on {readout['split']}, the rule says nexar_val (§13 T4)")
    items = json.loads((row_dir / nexar_eval.ITEM_AUCS_JSON).read_text(encoding="utf-8"))
    return readout, nexar_eval.crop_video_means(items)


def _paired_ci(a: dict[str, float], b: dict[str, float]) -> Interval | None:
    d = paired(a, b)
    return cluster_bootstrap_ci(d, {v: v for v in d}, constants.V2_E1_BOOTSTRAP,
                                constants.V5_BOOTSTRAP_SEED, constants.V2_E1_CI)


def nexar_set(root: Path) -> dict[str, Any]:
    """Crops (endpoint), whole videos, micro and AP of the three rows on ``nexar_val``."""
    reads = {row: _nexar_read(root / row) for row in ROWS}
    if set(reads[LAG][1]) != set(reads[K5][1]) or set(reads[LAG][1]) != set(reads[K5T2][1]):
        raise ValueError("nexar: the rows scored different crops")
    if not reads[LAG][0].get("released_baseline"):
        raise ValueError("nexar: the LaGoVAD row is not a --baseline read")
    rows = {}
    for row, (r, _) in reads.items():
        w = r["whole"]
        rows[row] = {"macro_crops": r["crops"]["mean_over_placements"]["macro"],
                     "macro_whole": w["macro"], "micro": w["micro"], "ap": w["micro_ap"],
                     "clip_auc": w["clip_auc_max"]}
    w = reads[LAG][0]["whole"]
    return {
        "rows": rows, "base_rate": w["positive_frame_share"],
        "clip_oracle_micro": w["clip_oracle_micro"],
        "middle_ruler_whole": w["middle_ruler_macro"],
        "videos": reads[LAG][0]["videos"],
        "t2_minus_lagovad": _sided(_paired_ci(reads[K5T2][1], reads[LAG][1])),
        "t2_minus_nexar": _sided(_paired_ci(reads[K5T2][1], reads[K5][1])),
    }


def check_checkpoints(
    nexar_view: argparse.Namespace, t2_view: argparse.Namespace, nexar_root: Path
) -> None:
    """One checkpoint per K5 row across its sets, and the two rows use different ones."""
    found: dict[str, dict[str, str]] = {}
    for row, view in ((K5, nexar_view), (K5T2, t2_view)):
        seen = zs.k5_checkpoints(view, list(ZS_SETS))
        path = nexar_root / row / nexar_eval.READOUT_JSON
        readout = json.loads(path.read_text(encoding="utf-8"))
        seen[NEXAR] = str(readout["checkpoints"][f"s{view.seed}"])
        zs.check_one_checkpoint(seen)
        found[row] = seen
    if set(found[K5].values()) & set(found[K5T2].values()):
        raise ValueError(f"K5 (T2) reads share a checkpoint with K5 (Nexar): {found[K5T2]}")


def tad_breakdown(views: dict[str, argparse.Namespace]) -> dict[str, Any]:
    """TAD per category, and false alarms on the normal videos (the §13 question, printed)."""
    runs = {K5: tadz_readout.load_run(zs.score_source(views[K5], zs.TAD, K5)),
            K5T2: tadz_readout.load_run(zs.score_source(views[K5T2], zs.TAD, K5)),
            LAG: tadz_readout.load_run(zs.score_source(views[K5T2], zs.TAD, LAG))}
    labels = tadz_readout.check_labels(runs, None)
    categories: dict[str, dict[str, Any]] = {}
    for video, truth in labels.items():
        if truth.max() > 0:
            name = CATEGORY_SUFFIX.sub("", video)
            categories.setdefault(name, {"videos": []})["videos"].append(video)
    out: dict[str, Any] = {"categories": {}, "normal_peak_share": {}}
    for name in sorted(categories):
        videos = categories[name]["videos"]
        out["categories"][name] = {
            "videos": len(videos),
            **{row: float(np.mean([frame_auc(runs[row][0][v], labels[v]) for v in videos]))
               for row in ROWS},
        }
    normals = [v for v, truth in labels.items() if truth.max() == 0]
    for row in ROWS:
        out["normal_peak_share"][row] = float(
            np.mean([runs[row][0][v].max() > NORMAL_PEAK for v in normals]))
    return out


def _macro(c: Interval | None) -> str:
    return "—" if c is None else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _signed(c: Interval | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _cells(rows: dict[str, Any], key: str) -> str:
    return " | ".join(f"{rows[r][key]:.4f}" for r in ROWS)


def _table(report: dict[str, Any]) -> list[str]:
    head = " | ".join(ROW_NAME[r] for r in ROWS)
    lines = [f"| Dataset | Metric | {head} | Base rate |", "|---|---|---|---|---|---|"]
    n = report["sets"][NEXAR]
    crops = " | ".join(_macro(n["rows"][r]["macro_crops"]) for r in ROWS)
    whole = " | ".join(_macro(n["rows"][r]["macro_whole"]) for r in ROWS)
    lines += [
        f"| **{SET_NAME[NEXAR]}** | AUC macro, crops | {crops} | |",
        f"| | AUC macro, whole video | {whole} | |",
        f"| | AUC micro | {_cells(n['rows'], 'micro')} | |",
        f"| | AP | {_cells(n['rows'], 'ap')} | {n['base_rate']:.4f} |",
    ]
    for name in ZS_SETS:
        s = report["sets"][name]
        macro = " | ".join(_macro(s["rows"][r]["macro"]) for r in ROWS)
        lines += [
            f"| **{SET_NAME[name]}** | AUC macro | {macro} | |",
            f"| | AUC micro | {_cells(s['rows'], 'micro')} | |",
            f"| | AP | {_cells(s['rows'], 'ap')} | {s['base_rate']:.4f} |",
        ]
    return lines


def render(report: dict[str, Any]) -> str:
    """The comparison as one markdown document."""
    lines = [
        f"# K5 trained on T2 vs K5 on Nexar vs LaGoVAD — **{report['label']}**",
        "",
        f"Seed {report['seed']}. Rule `KIM_RUN.md` §13, written before these numbers. Generated by "
        "`core.tools.k5_t2_report`. DADA-2000 is not read (§13 T6).",
        "",
        *_table(report),
        "",
        "## Macro contrasts, paired per item",
        "",
        "| Set | K5 (T2) - LaGoVAD | K5 (T2) - K5 (Nexar) |",
        "|---|---|---|",
    ]
    for name, s in report["sets"].items():
        a, b = s["t2_minus_lagovad"], s["t2_minus_nexar"]
        lines.append(f"| {SET_NAME[name]} | {_signed(a['delta'])} {a['sentence']} | "
                     f"{_signed(b['delta'])} {b['sentence']} |")
    n = report["sets"][NEXAR]
    lines += [
        "",
        f"Nexar val: {n['videos']} videos; contrasts on crops. Whole-video middle ruler "
        f"{_macro(n['middle_ruler_whole'])}, clip oracle micro {n['clip_oracle_micro']:.4f}. "
        f"TAD clip oracle micro {report['sets'][zs.TAD]['clip_oracle_micro']:.4f}, DAD "
        f"{report['sets'][zs.DAD]['clip_oracle_micro']:.4f}.",
        "",
        "## TAD by category (post-hoc, small groups, no intervals)",
        "",
        f"| Category | Videos | {' | '.join(ROW_NAME[r] for r in ROWS)} |",
        "|---|---|---|---|---|",
    ]
    tad = report["tad_breakdown"]
    for name, c in tad["categories"].items():
        lines.append(f"| {name} | {c['videos']} | {_cells({r: {'v': c[r]} for r in ROWS}, 'v')} |")
    peak = tad["normal_peak_share"]
    lines += ["", f"Normal videos with a peak above {NORMAL_PEAK}: "
              + ", ".join(f"{ROW_NAME[r]} {peak[r]:.2f}" for r in ROWS) + ".", ""]
    for title, diagnostics in report["dad_diagnostics"].items():
        lines += [f"**K5 row below = {title}.**", "", *zs._render_dad(diagnostics), ""]
    return "\n".join(lines) + "\n" + NOTES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--lagovad-root", type=Path, required=True, help="outputs/zs_lagovad")
    parser.add_argument("--kim-root", type=Path, required=True, help="outputs/kim (K5 on Nexar)")
    parser.add_argument("--t2-root", type=Path, required=True, help="outputs/kim_t2 (K5 on T2)")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    nexar_view, t2_view = _view(args, args.kim_root), _view(args, args.t2_root)
    nexar_root = args.t2_root / zs.EVAL_DIR / NEXAR
    check_checkpoints(nexar_view, t2_view, nexar_root)
    check = t0_check(zs.read_row(t2_view, zs.TAD, LAG))
    if not check["pass"]:
        raise ValueError(f"released checkpoint does not reproduce Gate T0 on TAD: {check}")
    sets = {NEXAR: nexar_set(nexar_root),
            **{s: zero_shot_set(nexar_view, t2_view, s) for s in ZS_SETS}}
    report: dict[str, Any] = {
        "label": LABEL, "seed": args.seed, "sets": sets,
        "tad_breakdown": tad_breakdown({K5: nexar_view, K5T2: t2_view}),
        "dad_diagnostics": {"trained on T2": zs.dad_diagnostics(t2_view),
                            "trained on Nexar": zs.dad_diagnostics(nexar_view)},
    }
    write_json_atomic(args.out_dir / f"{REPORT_STEM}.json", report)
    write_text_atomic(args.out_dir / f"{REPORT_STEM}.md", render(report))
    LOGGER.info("report -> %s", args.out_dir / f"{REPORT_STEM}.md")
    return report


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
