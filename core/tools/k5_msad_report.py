"""K5 vs LaGoVAD's released checkpoint on the MSAD test set (``KIM_RUN.md`` §18).

Pure aggregation over two ``core.evaluate --save-scores`` runs; no model is loaded. Refuses to
write when the two rows did not score the same videos with the same labels, when the protocol size
is not 240 videos / 96 two-class (19 on MSAD-traffic), or when LaGoVAD misses gate M0 (its micro
on the legacy shared-verbalizer read must reproduce ``full_gate_a``'s 0.8949; the per-item read the
table uses is printed against it, not gated, ``KIM_RUN.md`` §18 MS4 / lesson C30). Two sets (MS5):
**MSAD-traffic** (``Traffic_accident`` + every normal; answers the CCTV question) and
**MSAD-full** (paper comparison). Inputs::

    <root>/eval/LaGoVAD/released/   <root>/eval/LaGoVAD_legacy/released/   (M0 gate only)
    <root>/eval/K5/s<seed>/   [<root>/REPORTS/msad_census.json]

CLI::

    python -m core.tools.k5_msad_report --root outputs/v_final/kim_msad --seed 2024
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from core import constants
from core.metrics import frame_ap
from core.tools import tadz_readout as td
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.msad_zs import video_class
from core.tools.stp_readout import paired
from core.tools.zs_lagovad_readout import versus_sentence

LOGGER = logging.getLogger(__name__)

REPORT_STEM = "k5_msad_report"
LABEL = "PRELIMINARY (n = 1), not matched"
LAG, K5 = constants.ZS_LAGOVAD_ARM, constants.KIM_MAIN_ARM
LAG_LEGACY = f"{LAG}_legacy"  # the notebook's row name for the `--legacy-shared-verbalizer` read
FULL, TRAFFIC = constants.MSAD_ZS_FULL_SET, constants.MSAD_ZS_TRAFFIC_SET
SET_ORDER = (TRAFFIC, FULL)
SET_NAME = {TRAFFIC: "MSAD-traffic", FULL: "MSAD-full"}
TWO_CLASS = {TRAFFIC: constants.MSAD_ZS_TRAFFIC_TWO_CLASS, FULL: constants.MSAD_ZS_TWO_CLASS}
SEED = constants.V5_BOOTSTRAP_SEED

Curves = dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# pure pieces
# ---------------------------------------------------------------------------
def set_ids(ids: list[str], set_name: str) -> list[str]:
    """MS5: MSAD-traffic = ``Traffic_accident`` + every normal; MSAD-full = all."""
    if set_name == FULL:
        return sorted(ids)
    keep = (constants.MSAD_ZS_TRAFFIC_CLASS, "Normal")
    return sorted(v for v in ids if video_class(v) in keep)


def _pool(scores: Curves, labels: Curves) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = sorted(labels)
    s = np.concatenate([scores[v] for v in ids])
    g = np.concatenate([labels[v] for v in ids])
    w = np.concatenate([np.full(len(labels[v]), 1.0 / len(labels[v])) for v in ids])
    return s, g, w


def equal_video_micro(scores: Curves, labels: Curves) -> float:
    """Pooled frame AUC with every video's frames summing to one (length weight removed)."""
    s, g, w = _pool(scores, labels)
    return float(roc_auc_score(g.astype(np.int64), s, sample_weight=w))


def row_read(scores: Curves, labels: Curves) -> dict[str, Any]:
    s, g, _ = _pool(scores, labels)
    items = td.clip_aucs(scores, labels)
    return {
        "items": items,
        "macro": td.bootstrap(items, SEED),
        "micro": td.micro(scores, labels),
        "micro_equal_video": equal_video_micro(scores, labels),
        "ap": frame_ap(s, g),
        "clip_auc": td.video_auc(scores, labels),
    }


def per_class_macro(items: dict[str, float]) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for v, a in items.items():
        out.setdefault(video_class(v), []).append(a)
    return {c: float(np.mean(x)) for c, x in sorted(out.items())}


def gate_m0(legacy_micro: float, per_item_micro: float) -> dict[str, Any]:
    """M0 on the legacy-stream read; the per-item micro (the table's row) is printed, not gated."""
    gap = float(legacy_micro) - constants.MSAD_ZS_GATE_A_MICRO
    out = {"micro": float(legacy_micro), "reference": constants.MSAD_ZS_GATE_A_MICRO,
           "gap": gap, "pass": abs(gap) <= constants.MSAD_ZS_GATE_A_TOL,
           "per_item_micro": float(per_item_micro),
           "per_item_gap": float(per_item_micro) - constants.MSAD_ZS_GATE_A_MICRO}
    if not out["pass"]:
        raise ValueError(f"M0: LaGoVAD legacy-stream micro {legacy_micro:.4f} does not reproduce "
                         f"full_gate_a {constants.MSAD_ZS_GATE_A_MICRO} (gap {gap:+.5f}): "
                         "checkpoint or MSAD_ncc drifted")
    return out


def read_set(runs: dict[str, tuple[Curves, Curves]], set_name: str) -> dict[str, Any]:
    labels_all = runs[LAG][1]
    ids = set_ids(list(labels_all), set_name)
    labels = {v: labels_all[v] for v in ids}
    rows = {r: row_read({v: runs[r][0][v] for v in ids}, labels) for r in (LAG, K5)}
    n_two = len(rows[LAG]["items"])
    if n_two != TWO_CLASS[set_name]:
        raise ValueError(f"{SET_NAME[set_name]}: {n_two} two-class videos, protocol "
                         f"{TWO_CLASS[set_name]}")
    rulers = td.rulers(labels, None, SEED)
    delta = td.bootstrap(paired(rows[K5]["items"], rows[LAG]["items"]), SEED)
    _, g, _ = _pool(labels, labels)
    return {
        "videos": len(ids),
        "abnormal": sum(int(labels[v].max() > 0) for v in ids),
        "two_class": n_two,
        "base_rate": float(g.mean()),
        "rows": {r: {k: v for k, v in rows[r].items() if k != "items"} for r in rows},
        "per_class_macro": {r: per_class_macro(rows[r]["items"]) for r in rows},
        "delta_macro": delta,
        "sentence": versus_sentence(delta),
        "delta_micro": rows[K5]["micro"] - rows[LAG]["micro"],
        "rulers": {"clip_oracle_micro": rulers["clip_oracle_micro"],
                   "length_ruler_micro": rulers["length_ruler_micro"],
                   "monotone_ruler_macro": rulers["monotone_ruler_macro"]},
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _ci(c: dict[str, float] | None, signed: bool = False) -> str:
    if c is None:
        return "n/a"
    f = "{:+.4f}" if signed else "{:.4f}"
    return f"{f.format(c['mean'])} [{f.format(c['low'])}, {f.format(c['high'])}]"


def render(report: dict[str, Any]) -> str:
    m0 = report["m0"]
    lines = [
        "# K5 vs LaGoVAD (released ckpt), zero-shot MSAD test — KIM_RUN.md §18",
        "",
        f"**{LABEL}.** K5 s{report['seed']}; nothing trained. Gate M0 (legacy shared-verbalizer "
        f"read): LaGoVAD micro {m0['micro']:.4f} vs `full_gate_a` {m0['reference']} "
        f"(gap {m0['gap']:+.5f}) PASS. The table's per-item read (definitions drawn per video, "
        f"same draws for both rows) is {m0['per_item_micro']:.4f} (gap {m0['per_item_gap']:+.5f}, "
        "printed, not gated: the draws differ, C30). "
        f"LaGoVAD's printed MSAD AUC is {constants.MSAD_ZS_PAPER_MICRO} (beside its re-read only).",
        "",
        "| Set | Metric | LaGoVAD | K5 | K5 - LaGoVAD |",
        "|---|---|---|---|---|",
    ]
    for s in SET_ORDER:
        r = report["sets"][s]
        lag, k5 = r["rows"][LAG], r["rows"][K5]
        head = f"**{SET_NAME[s]}** ({r['videos']} videos, macro over {r['two_class']})"
        lines += [
            f"| {head} | AUC macro | {_ci(lag['macro'])} | {_ci(k5['macro'])} | "
            f"{_ci(r['delta_macro'], True)} **{r['sentence']}** |",
            f"| | AUC micro | {lag['micro']:.4f} | {k5['micro']:.4f} | {r['delta_micro']:+.4f} |",
            f"| | micro, equal video weight | {lag['micro_equal_video']:.4f} | "
            f"{k5['micro_equal_video']:.4f} | |",
            f"| | AP (base rate {r['base_rate']:.4f}) | {lag['ap']:.4f} | {k5['ap']:.4f} | |",
            f"| | clip AUC (peak) | {lag['clip_auc']:.4f} | {k5['clip_auc']:.4f} | |",
        ]
    lines += ["", "## Rulers (no pixel)", "",
              "| Set | clip oracle (micro) | length ruler (micro) | `t/N` ruler (macro) |",
              "|---|---|---|---|"]
    for s in SET_ORDER:
        ru = report["sets"][s]["rulers"]
        lines.append(f"| {SET_NAME[s]} | {ru['clip_oracle_micro']:.4f} | "
                     f"{ru['length_ruler_micro']:.4f} | {_ci(ru['monotone_ruler_macro'])} |")
    full = report["sets"][FULL]["per_class_macro"]
    lines += ["", "## Macro per class (MSAD-full, printed)", "", "| Class | LaGoVAD | K5 |",
              "|---|---|---|"]
    lines += [f"| {c} | {full[LAG][c]:.4f} | {full[K5][c]:.4f} |" for c in full[LAG]]
    census = report.get("census")
    if census:
        lines += ["", f"Census: {census['videos']} videos, fps spread {census['fps_spread']}, "
                  f"{census['fps_off_nominal']} off 30 ± 1; label rows ≠ CLIP rows on "
                  f"{census['label_rows_differ']} (truncated alike, §18 M2)."]
    lines += [
        "",
        "## How to read this",
        "",
        "* The CCTV-traffic question is answered by **MSAD-traffic macro only** (MS6). "
        "MSAD-full is mostly classes outside the thesis scope: a cost there is not attributed "
        "to the camera.",
        "* **Micro on MSAD-traffic says nothing**: the length ruler alone beats it. On MSAD-full "
        "the clip oracle sits at LaGoVAD's micro.",
        "* Never put the paper's number beside K5; never write \"beats LaGoVAD\" (not matched).",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    runs = {LAG: td.load_run(args.root / "eval" / LAG / constants.ZS_LAGOVAD_RUN),
            K5: td.load_run(args.root / "eval" / K5 / f"s{args.seed}")}
    legacy_scores, legacy_labels = td.load_run(
        args.root / "eval" / LAG_LEGACY / constants.ZS_LAGOVAD_RUN)
    labels = td.check_labels(runs, None)
    if legacy_labels.keys() != labels.keys() or any(
            not np.array_equal(legacy_labels[v], labels[v]) for v in labels):
        raise ValueError("M0: the legacy read scored other videos or labels than the table's rows")
    if len(labels) != constants.MSAD_ZS_TEST_VIDEOS:
        raise ValueError(f"{len(labels)} videos scored, protocol {constants.MSAD_ZS_TEST_VIDEOS}")
    report: dict[str, Any] = {
        "seed": args.seed,
        "m0": gate_m0(td.micro(legacy_scores, labels), td.micro(runs[LAG][0], labels)),
        "sets": {s: read_set(runs, s) for s in SET_ORDER},
    }
    census = args.root / "REPORTS" / constants.MSAD_ZS_CENSUS_FILENAME
    if census.is_file():
        report["census"] = json.loads(census.read_text(encoding="utf-8"))["summary"]
    out = args.out_dir or args.root / "REPORTS"
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / f"{REPORT_STEM}.json", report)
    write_text_atomic(out / f"{REPORT_STEM}.md", render(report))
    LOGGER.info("wrote %s", out / f"{REPORT_STEM}.md")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--root", type=Path, required=True, help="OUTPUT_ROOT/kim_msad")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
