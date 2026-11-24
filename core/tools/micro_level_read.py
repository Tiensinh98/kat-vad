"""Where raw micro is lost on the all-abnormal sets: clips whose accident frames sit low.

``KIM_RUN.md`` §44.2 (rule written 2026-11-24, before any number of this check). On a set where
every clip holds an accident, raw micro = the positive-frame-weighted mean of each clip's level
``A_c`` (the share of the set's negative frames below its positive frames). A clip is **low** when
``A_c < MICRO_LEVEL_LOW``. The tool reads how many clips are low, whether they still rank well
inside themselves (level vs ranking), how K5f's and K5dfi's low sets overlap, and whether low
clips concentrate in a recorded attribute (strata, pooling, bootstrap and lead rule of §24).
Saved scores and annotations only; printed, decides nothing.

Inputs::

    --dota ROW=DIR ...       protocol-B read dirs (clip_scores.npz), one per row
    --dada ROW=DIR ...       DADA whole read dirs (clip_scores.npz), one per row
    --dota-annotations       DoTA raw_annotations/annotations/
    --dada-workbook          dada标注.xlsx
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
from core.tools.protocol_b_eval import CLIP_SCORES_NPZ, read_clip_scores

LOGGER = logging.getLogger(__name__)

Curves = dict[str, np.ndarray]
DOTA, DADA = "DoTA-CAP", "DADA-2000"
TIE = 0.5


def share_below(sorted_ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per value of ``x``: share of ``sorted_ref`` strictly below it, ties counted ½."""
    lo = np.searchsorted(sorted_ref, x, side="left")
    hi = np.searchsorted(sorted_ref, x, side="right")
    return np.asarray((lo + TIE * (hi - lo)) / len(sorted_ref), dtype=np.float64)


def minmax(x: np.ndarray) -> np.ndarray:
    span = float(x.max() - x.min())
    return np.asarray((x - x.min()) / span if span > 0 else np.zeros_like(x, dtype=np.float64))


def levels(scores: Curves, labels: Curves) -> dict[str, Any]:
    """§44.2 L2/L3 per clip: ``A_c`` (positive side), ``N_c`` (negative side), and the gate."""
    ids = sorted(labels)
    pos = np.sort(np.concatenate([scores[v][labels[v] > 0] for v in ids]).astype(np.float64))
    neg = np.sort(np.concatenate([scores[v][labels[v] == 0] for v in ids]).astype(np.float64))
    a_c = {v: float(share_below(neg, scores[v][labels[v] > 0]).mean()) for v in ids}
    n_c = {v: float(share_below(pos, scores[v][labels[v] == 0]).mean())
           for v in ids if (labels[v] == 0).any()}
    n_pos = {v: int((labels[v] > 0).sum()) for v in ids}
    weighted = sum(a_c[v] * n_pos[v] for v in ids) / sum(n_pos.values())
    pooled = frame_auc(np.concatenate([scores[v] for v in ids]),
                       np.concatenate([labels[v] for v in ids]).astype(np.int64))
    if abs(weighted - pooled) > constants.MICRO_LEVEL_GATE_ATOL:
        raise ValueError(f"L2 gate: weighted A_c {weighted:.8f} != pooled micro {pooled:.8f}")
    mm = {v: minmax(scores[v].astype(np.float64)) for v in ids}
    mm_neg = np.sort(np.concatenate([mm[v][labels[v] == 0] for v in ids]))
    a_mm = {v: float(share_below(mm_neg, mm[v][labels[v] > 0]).mean()) for v in ids}
    return {"ids": ids, "A": a_c, "N": n_c, "n_pos": n_pos, "micro_raw": pooled, "A_mm": a_mm}


def read_row(name: str, run_dir: Path, strata: dict[str, dict[str, str]],
             rate: float) -> dict[str, Any]:
    """Every L4/L5 read of one row on one set."""
    scores, labels = read_clip_scores(run_dir / CLIP_SCORES_NPZ)
    lv = levels(scores, labels)
    ids = lv["ids"]
    if any(not 0 < int(labels[v].sum()) < len(labels[v]) for v in ids):
        raise ValueError(f"{name}: §44.2 needs every clip two-class (all-abnormal set)")
    a = np.array([lv["A"][v] for v in ids])
    low = a < constants.MICRO_LEVEL_LOW
    auc = np.array([frame_auc(scores[v], labels[v]) for v in ids])
    n_pos = np.array([lv["n_pos"][v] for v in ids])
    loud = np.array([lv["N"].get(v, 0.0) > constants.MICRO_LEVEL_LOW for v in ids])
    groups = {v: dota_group(v) for v in ids} if name == DOTA else {v: v for v in ids}
    draws = ft.draws_of(ids, groups)
    attrs = {key: ft.stratum_reads(low, auc, np.array([ft.pool_small(vals)[v] for v in ids]),
                                   draws) for key, vals in strata.items()}
    covs = ft.covariates({v: labels[v] for v in ids}, rate)
    cov_reads = {k: ft.covariate_read(np.array([c[v] for v in ids]), low, draws)
                 for k, c in covs.items()}
    a_mm = np.array([lv["A_mm"][v] for v in ids])
    return {
        "items": len(ids), "micro_raw": lv["micro_raw"],
        "low_share": float(low.mean()), "low": int(low.sum()),
        "low_share_mm": float((a_mm < constants.MICRO_LEVEL_LOW).mean()),
        "positive_frames_in_low": float(n_pos[low].sum() / n_pos.sum()),
        "loud_share": float(loud.mean()),
        "level_vs_ranking": {
            "low_mean_own_auc": float(auc[low].mean()) if low.any() else float("nan"),
            "rest_mean_own_auc": float(auc[~low].mean()) if (~low).any() else float("nan"),
            "low_share_own_auc_ge_half":
                float((auc[low] >= constants.FAILTAIL_FAIL_AUC).mean()) if low.any()
                else float("nan"),
        },
        "strata": attrs, "covariates": cov_reads,
        "low_ids": [v for v, f in zip(ids, low, strict=True) if f],
    }


def overlap(a: list[str], b: list[str]) -> dict[str, float]:
    sa, sb = set(a), set(b)
    union = sa | sb
    return {"both": len(sa & sb), "first_only": len(sa - sb), "second_only": len(sb - sa),
            "jaccard": len(sa & sb) / len(union) if union else float("nan")}


def leads(rows: dict[str, Any]) -> list[str]:
    out = []
    for row, r in rows.items():
        for key, strata in r["strata"].items():
            out += [f"{row} {key} = {st}" for st, s in strata.items() if s["lead"]]
        out += [f"{row} {k}" for k, c in r["covariates"].items() if c["lead"]]
    return out


def parse_rows(pairs: list[str]) -> dict[str, Path]:
    out = {}
    for p in pairs:
        row, sep, path = p.partition("=")
        if not sep:
            raise ValueError(f"expected ROW=DIR, got {p!r}")
        out[row] = Path(path)
    return out


def read_set(name: str, dirs: dict[str, Path], strata_of: Any, rate: float) -> dict[str, Any]:
    first = next(iter(dirs.values()))
    ids = sorted(read_clip_scores(first / CLIP_SCORES_NPZ)[1])
    strata = strata_of(ids)
    rows = {row: read_row(name, d, strata, rate) for row, d in dirs.items()}
    if len({r["items"] for r in rows.values()}) != 1:
        raise ValueError(f"{name}: rows scored different item counts")
    names = list(rows)
    ov = overlap(rows[names[0]]["low_ids"], rows[names[1]]["low_ids"]) if len(names) > 1 else {}
    return {"set": name, "rows": rows, "overlap": ov, "overlap_rows": names[:2],
            "leads": leads(rows)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    sets = {
        DOTA: read_set(DOTA, parse_rows(args.dota),
                       lambda ids: ft.dota_strata(ids, args.dota_annotations),
                       ft.STEPS_PER_S[ft.DOTA]),
        DADA: read_set(DADA, parse_rows(args.dada),
                       lambda ids: ft.dada_strata(ids, args.dada_workbook)[0],
                       ft.STEPS_PER_S[ft.DADA]),
    }
    report = {"rule": "KIM_RUN.md §44.2", "seed": ft.SEED, "draws": ft.DRAWS,
              "low": constants.MICRO_LEVEL_LOW, "sets": sets}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / constants.MICRO_LEVEL_REPORT_STEM
    write_json_atomic(stem.with_suffix(".json"), report)
    write_text_atomic(stem.with_suffix(".md"), render_markdown(report))
    return report


def _ci(d: dict[str, float]) -> str:
    return f"{d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Micro level read ({report['rule']}) — printed, decides nothing, n = 1", ""]
    for name, s in report["sets"].items():
        lines += [f"## {name}", "",
                  "| row | items | micro raw | low share | low share (min-max) "
                  "| pos. frames in low | loud share | own AUC low / rest "
                  "| low with own AUC ≥ 0.5 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row, r in s["rows"].items():
            lr = r["level_vs_ranking"]
            lines.append(f"| {row} | {r['items']} | {r['micro_raw']:.4f} | {r['low_share']:.3f} "
                         f"({r['low']}) | {r['low_share_mm']:.3f} | "
                         f"{r['positive_frames_in_low']:.3f} | {r['loud_share']:.3f} | "
                         f"{lr['low_mean_own_auc']:.3f} / {lr['rest_mean_own_auc']:.3f} | "
                         f"{lr['low_share_own_auc_ge_half']:.3f} |")
        if s["overlap"]:
            o = s["overlap"]
            a, b = s["overlap_rows"]
            lines += ["", f"Overlap of low sets: both {o['both']}, {a} only {o['first_only']}, "
                      f"{b} only {o['second_only']}, Jaccard {o['jaccard']:.3f}"]
        for row, r in s["rows"].items():
            lines += ["", f"### {row}: low share by stratum (Δ vs rest, 95 %)", "",
                      "| attribute | stratum | n | low share | mean own AUC | Δ | lead |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for key, strata in r["strata"].items():
                for st, x in strata.items():
                    lines.append(f"| {key} | {st} | {x['n']} | {x['fail_share']:.3f} | "
                                 f"{x['mean_auc']:.3f} | {_ci(x['delta_share'])} | "
                                 f"{'LEAD' if x['lead'] else ''} |")
            lines += ["", "| covariate | AUC low vs rest [95 %] | mean low / rest | lead |",
                      "| --- | --- | --- | --- |"]
            for k, c in r["covariates"].items():
                lines.append(f"| {k} | {c['auc']:.3f} [{c['low']:.3f}, {c['high']:.3f}] | "
                             f"{c['mean_fail']:.3f} / {c['mean_rest']:.3f} | "
                             f"{'LEAD' if c['lead'] else ''} |")
        lines += ["", f"Leads: {', '.join(s['leads']) or 'none'}", ""]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument("--dota", nargs="+", required=True, help="ROW=DIR, first = reference")
    parser.add_argument("--dada", nargs="+", required=True, help="ROW=DIR, first = reference")
    for flag in ("--dota-annotations", "--dada-workbook", "--out-dir"):
        parser.add_argument(flag, type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
