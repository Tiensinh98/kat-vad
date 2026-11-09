"""K5 trained on uncut Nexar videos (``whole``) beside K5 on ``Nexar_window`` (``KIM_RUN.md`` §16).

Printed, decides nothing (n = 1). Pure aggregation over saved reads. It refuses to write when two
rows did not score the same items, when a recomputed micro disagrees with its read, when a row's
reads come from more than one checkpoint or both rows share one, or when a Nexar read is not on
``nexar_val``. Inputs::

    <kim-root>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/     K5 (window)
    <whole-root>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/   K5 (whole)
    <k5-nexar-dir>/                                                 nexar_eval of K5 on nexar_val
    <whole-root>/eval/nexar/K5whole/                                nexar_eval of K5 (whole)
    [--pilot-dir]                                                   v2 whole/A3-ign pilot, printed

CLI::

    python -m core.tools.k5_whole_report --kim-root outputs/v_final/kim \\
        --whole-root outputs/v_final/kim_whole \\
        --k5-nexar-dir outputs/v_final/kim_t2/eval/nexar/K5 \\
        --out-dir outputs/v_final/kim_whole/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.metrics import cluster_bootstrap_ci
from core.tools import k5_zs_report as zs
from core.tools import nexar_eval, nexar_trigger, zs_compare_table
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_transfer_readout import ArmRead
from core.tools.stp_readout import paired
from core.tools.zs_lagovad_readout import ABOVE, delta_ci, versus_sentence

LOGGER = logging.getLogger(__name__)

REPORT_STEM = "k5_whole_report"
LABEL = "PRELIMINARY (n = 1 seed)"
K5, WHOLE = constants.KIM_MAIN_ARM, constants.KIM_WHOLE_ROW
ROWS = (K5, WHOLE)
ROW_NAME = {K5: "K5 (window)", WHOLE: "K5 (whole)"}
NEXAR = "nexar"
ZS_SETS = (zs.DOTA, zs.DADA, zs.TAD, zs.DAD)
SET_NAME = {NEXAR: "Nexar val", **zs.SET_NAME}
NOTES = """## How to read this

* **PRELIMINARY (n = 1 seed, s2024). Printed, decides nothing** (`KIM_RUN.md` §16). K5 on
  `Nexar_window` stays the main model. No row may be called adopted.
* **Three things change together** (§16 WH7): the corpus (uncut videos, video-level labels, the
  negative videos included), `mil_topk_pct` 5 -> 16, and the steps per epoch. A contrast never
  names one cause and is never written "whole is better / worse".
* **Nexar val is in-domain for both rows.** Whole-video macro is a position test (middle ruler
  printed); crops are the endpoint. Never beside the `nexar_test` table of `RESULTS_NEXAR.md`.
* **Micro on TAD and DAD measures clip classification. DAD macro is not localization**
  (`ZS_DAD.md` §8).
* The v2 pilot row is a **different host** (CLIP + CRN + V2-S, DVS on, `mil_topk_pct` 5, s2099).
"""

Interval = dict[str, float]


def _view(root: Path, seed: int) -> argparse.Namespace:
    """The arguments ``k5_zs_report`` reads, with the K5 row's reads under ``root``."""
    return argparse.Namespace(lagovad_root=root, kim_root=root, seed=seed)


def zero_shot_row(view: argparse.Namespace, set_name: str) -> tuple[ArmRead, dict[str, Any]]:
    """One row of one set, with its micro re-derived from the saved scores (same gate as zs)."""
    read = zs.read_row(view, set_name, K5)
    loaded = zs_compare_table.load_scores(zs.score_source(view, set_name, K5))
    if loaded is None:
        raise FileNotFoundError(f"{set_name}: no scores under {view.kim_root}: no AP")
    pooled = zs_compare_table.pooled_reads(*loaded, minmax=set_name == zs.DOTA)
    if abs(pooled["micro"] - read["micro"]) > zs.MICRO_TOLERANCE:
        raise ValueError(f"{set_name} ({view.kim_root}): micro {pooled['micro']:.6f} from the "
                         f"scores != {read['micro']:.6f} in the read: not the same scores")
    cells = {"macro": read["macro"], "micro": read["micro"], "ap": pooled["ap"],
             "base_rate": pooled["base_rate"]}
    return read, cells


def zero_shot_set(
    views: dict[str, argparse.Namespace], set_name: str, new: str = WHOLE
) -> dict[str, Any]:
    """Both rows of a set and ``new`` - K5 (window), paired per item."""
    reads, rows = {}, {}
    for row, view in views.items():
        reads[row], rows[row] = zero_shot_row(view, set_name)
    if set(reads[K5]["items"]) != set(reads[new]["items"]):
        raise ValueError(f"{set_name}: the two rows scored different items: not paired")
    delta = delta_ci(reads[new], reads[K5])
    return {"rows": rows, "base_rate": rows[K5]["base_rate"], "protocol": zs.PROTOCOL[set_name],
            "delta": delta, "sentence": versus_sentence(delta)}


def nexar_read(row_dir: Path) -> tuple[dict[str, Any], dict[str, float]]:
    """A ``nexar_eval`` read on ``nexar_val`` and its per-video crop means."""
    readout = json.loads((row_dir / nexar_eval.READOUT_JSON).read_text(encoding="utf-8"))
    if readout["split"] != constants.V2_SPLIT_NEXAR_VAL:
        raise ValueError(f"{row_dir}: read on {readout['split']}, §16 WH5 says nexar_val")
    items = json.loads((row_dir / nexar_eval.ITEM_AUCS_JSON).read_text(encoding="utf-8"))
    return readout, nexar_eval.crop_video_means(items)


def nexar_cells(readout: dict[str, Any]) -> dict[str, Any]:
    """Endpoint, each placement, edge drop (W1's read), whole video and clip AUC of one read."""
    w = readout["whole"]
    clip_auc = w["clip_auc_max"]
    flag = clip_auc is not None and clip_auc >= constants.NEXAR_TRIGGER_CLIP_AUC
    return {
        "macro_crops": readout["crops"]["mean_over_placements"]["macro"],
        "placements": {str(p): nexar_trigger.crop_mean(readout, p)
                       for p in constants.NEXAR_CROP_PLACEMENTS},
        "edge_drop": nexar_trigger.edge_drop(readout),
        "macro_whole": w["macro"], "micro": w["micro"], "ap": w["micro_ap"],
        "clip_auc": clip_auc,
        "clip_classifier_flag": flag,
    }


def nexar_set(dirs: dict[str, Path], new: str = WHOLE) -> dict[str, Any]:
    """Both rows on ``nexar_val``; the contrast is on the per-video crop means (D-N2)."""
    reads = {row: nexar_read(d) for row, d in dirs.items()}
    if set(reads[K5][1]) != set(reads[new][1]):
        raise ValueError("nexar: the two rows scored different crops")
    d = paired(reads[new][1], reads[K5][1])
    delta = cluster_bootstrap_ci(d, {v: v for v in d}, constants.V2_E1_BOOTSTRAP,
                                 constants.V5_BOOTSTRAP_SEED, constants.V2_E1_CI)
    w = reads[K5][0]["whole"]
    return {
        "rows": {row: nexar_cells(r) for row, (r, _) in reads.items()},
        "base_rate": w["positive_frame_share"], "middle_ruler_whole": w["middle_ruler_macro"],
        "videos": reads[K5][0]["videos"], "delta": delta, "sentence": versus_sentence(delta),
    }


def check_checkpoints(
    views: dict[str, argparse.Namespace], dirs: dict[str, Path], seed: int, new: str = WHOLE
) -> None:
    """One checkpoint per row across its five reads, and the two rows use different ones."""
    found: dict[str, dict[str, str]] = {}
    for row, view in views.items():
        seen = zs.k5_checkpoints(view, list(ZS_SETS))
        readout = json.loads((dirs[row] / nexar_eval.READOUT_JSON).read_text(encoding="utf-8"))
        seen[NEXAR] = str(readout["checkpoints"][f"s{seed}"])
        zs.check_one_checkpoint(seen)
        found[row] = seen
    if set(found[K5].values()) & set(found[new].values()):
        raise ValueError(f"{new} reads share a checkpoint with K5 (window): {found[new]}")


def pilot_row(pilot_dir: Path | None) -> dict[str, Any] | None:
    """The v2 ``whole``/A3-ign pilot on ``nexar_val`` (different host), printed only."""
    if pilot_dir is None:
        return None
    readout, _ = nexar_read(pilot_dir)
    return nexar_cells(readout)


def follow_up(report: dict[str, Any]) -> dict[str, Any]:
    """§16 WH8: may Claude propose s2025? Above K5 on crops and on >= N of the zero-shot sets."""
    above = [s for s in ZS_SETS if report["sets"][s]["sentence"] == ABOVE]
    crops_above = report["sets"][NEXAR]["sentence"] == ABOVE
    return {"crops_above": crops_above, "zero_shot_above": above,
            "may_propose_s2025": crops_above and len(above) >= constants.KIM_WHOLE_PROPOSE_MIN_SETS}


def _macro(c: Interval | None) -> str:
    return "—" if c is None else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _signed(c: Interval | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def _nexar_lines(report: dict[str, Any]) -> list[str]:
    n = report["sets"][NEXAR]
    rows = dict(n["rows"])
    names = dict(ROW_NAME)
    if report["pilot"] is not None:
        rows["pilot"], names["pilot"] = report["pilot"], "v2 pilot whole/A3-ign (other host)"
    places = constants.NEXAR_CROP_PLACEMENTS
    lines = [
        f"## Nexar val ({n['videos']} videos, in-domain for both K5 rows)",
        "",
        "| Row | crops (endpoint) | " + " | ".join(f"@{p}" for p in places)
        + " | edge drop | whole macro | micro | AP | clip AUC |",
        "|---|---|" + "---|" * len(places) + "---|---|---|---|---|",
    ]
    for row, c in rows.items():
        bar = constants.NEXAR_TRIGGER_CLIP_AUC
        flag = f" **(>= {bar:.2f})**" if c["clip_classifier_flag"] else ""
        lines.append(
            f"| {names[row]} | {_macro(c['macro_crops'])} | "
            + " | ".join(f"{c['placements'][str(p)]:.4f}" for p in places)
            + f" | {c['edge_drop']:+.4f} | {_macro(c['macro_whole'])} | {_num(c['micro'])} | "
            f"{_num(c['ap'])} | {_num(c['clip_auc'])}{flag} |")
    lines += ["", f"Whole-video middle ruler {_macro(n['middle_ruler_whole'])}; positive-frame "
              f"share {n['base_rate']:.4f}. Edge drop = centre crop - worse edge crop "
              "(W1's read, printed).", ""]
    return lines


def render(report: dict[str, Any]) -> str:
    """The comparison as one markdown document."""
    lines = [
        f"# K5 trained on uncut Nexar videos vs K5 on windows — **{report['label']}**",
        "",
        f"Seed {report['seed']}. Rule `KIM_RUN.md` §16, written before these numbers. Generated by "
        "`core.tools.k5_whole_report`. `nexar_test` is not read.",
        "",
        *_nexar_lines(report),
        "## Zero-shot sets",
        "",
        "| Set | Metric | K5 (window) | K5 (whole) | Base rate |",
        "|---|---|---|---|---|",
    ]
    for name in ZS_SETS:
        s = report["sets"][name]
        r = s["rows"]
        lines += [
            f"| **{SET_NAME[name]}** | AUC macro | {_macro(r[K5]['macro'])} | "
            f"{_macro(r[WHOLE]['macro'])} | |",
            f"| | AUC micro | {r[K5]['micro']:.4f} | {r[WHOLE]['micro']:.4f} | |",
            f"| | AP | {r[K5]['ap']:.4f} | {r[WHOLE]['ap']:.4f} | {s['base_rate']:.4f} |",
        ]
    lines += ["", "## K5 (whole) - K5 (window), macro, paired per item (§16 WH7)", "",
              "| Set | Δ [95 % CI] | Sentence |", "|---|---|---|"]
    for name, s in report["sets"].items():
        what = "crops" if name == NEXAR else "macro"
        lines.append(f"| {SET_NAME[name]} ({what}) | {_signed(s['delta'])} | {s['sentence']} |")
    f = report["follow_up"]
    lines += [
        "",
        f"**§16 WH8:** crops above = {f['crops_above']}; zero-shot sets above = "
        f"{f['zero_shot_above'] or 'none'} (needs ≥ {constants.KIM_WHOLE_PROPOSE_MIN_SETS}). "
        f"Claude may propose s2025: **{'yes' if f['may_propose_s2025'] else 'no'}**.",
        "",
    ]
    return "\n".join(lines) + "\n" + NOTES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--kim-root", type=Path, required=True, help="outputs/kim (K5 on windows)")
    parser.add_argument("--whole-root", type=Path, required=True, help="outputs/kim_whole")
    parser.add_argument("--k5-nexar-dir", type=Path, required=True,
                        help="nexar_eval of K5 (window) on nexar_val (kim_t2/eval/nexar/K5)")
    parser.add_argument("--pilot-dir", type=Path, default=None,
                        help="optional: v2 REPORTS/nexar_val_ignore/pilot_whole_A3 (printed only)")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    views = {K5: _view(args.kim_root, args.seed), WHOLE: _view(args.whole_root, args.seed)}
    dirs = {K5: args.k5_nexar_dir, WHOLE: args.whole_root / zs.EVAL_DIR / NEXAR / WHOLE}
    check_checkpoints(views, dirs, args.seed)
    sets = {NEXAR: nexar_set(dirs), **{s: zero_shot_set(views, s) for s in ZS_SETS}}
    report: dict[str, Any] = {"label": LABEL, "seed": args.seed, "sets": sets,
                              "pilot": pilot_row(args.pilot_dir)}
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
