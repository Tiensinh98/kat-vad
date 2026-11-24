"""Does the size of the accident's boxes explain K5f's ego gap in level on DoTA-CAP?

``KIM_RUN.md`` §47 (rule written 2026-11-24, before any score was paired with a box). Within the
ego-uninvolved clips: Spearman of box size against the clip level ``A_c``. Across groups: the gap
in low share re-read at matched size (fixed bins, the uninvolved group's bin weights), whose
difference from the raw gap is the part the size mix carries. Saved scores and DoTA annotations
only; printed, decides nothing.

Inputs::

    --dota ROW=DIR ...       whole DoTA-CAP read dirs (clip_scores.npz), every row of BOX_SIZE_ROWS
    --dota-annotations       DoTA raw_annotations/annotations/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.v2_splits import dota_group
from core.tools import ego_gap_read as eg
from core.tools import k5_fail_tail as ft
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.micro_level_read import parse_rows
from core.tools.protocol_b_eval import CLIP_SCORES_NPZ, read_clip_scores

LOGGER = logging.getLogger(__name__)

SIZE, SPEED, SIZE_SPEED = "size", "speed", "size x speed"
FEATURES = (SIZE, SPEED, SIZE_SPEED)
CONSISTENT = "SIZE-CONSISTENT"
RELATED = "RELATED, GAP NOT EXPLAINED"
NO_RELATION = "NO SIZE RELATION"
NOT_SETTLED = "NOT SETTLED"


def clip_boxes(doc: dict[str, Any]) -> dict[str, float] | None:
    """BS2 for one annotation: size, speed and their product; None if no frame has a box."""
    shares, steps = [], []
    last: dict[int, tuple[int, float, float]] = {}
    for label in doc["labels"]:
        frame = int(label["frame_id"])
        if not doc["anomaly_start"] <= frame < doc["anomaly_end"] or not label["objects"]:
            continue
        areas = []
        for obj in label["objects"]:
            x0, y0, x1, y1 = obj["bbox"]
            areas.append(max(x1 - x0, 0.0) * max(y1 - y0, 0.0))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if (prev := last.get(obj["obj_track_id"])) and frame > prev[0]:
                steps.append(float(np.hypot(cx - prev[1], cy - prev[2])) / (frame - prev[0]))
            last[obj["obj_track_id"]] = (frame, cx, cy)
        shares.append(max(areas) / constants.BOX_SIZE_FRAME_AREA)
    if not shares:
        return None
    size = float(np.median(shares))
    speed = float(np.median(steps)) if steps else float("nan")
    return {SIZE: size, SPEED: speed, SIZE_SPEED: size * speed}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(ft.average_ranks(x[ok]), ft.average_ranks(y[ok]))[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    """Rank correlation with the linear rank effect of ``controls`` removed from both sides."""
    ok = np.isfinite(x) & np.isfinite(y)
    design = np.column_stack([np.ones(int(ok.sum()))]
                             + [ft.average_ranks(c[ok]) for c in controls])
    out = []
    for v in (x, y):
        r = ft.average_ranks(v[ok])
        coef, *_ = np.linalg.lstsq(design, r, rcond=None)
        out.append(r - design @ coef)
    return float(np.corrcoef(out[0], out[1])[0, 1])


def bin_of(size: np.ndarray) -> np.ndarray:
    return np.digitize(size, constants.BOX_SIZE_EDGES)


def gap_parts(low: np.ndarray, involved: np.ndarray, bins: np.ndarray) -> dict[str, float] | None:
    """BS4 on one sample: ``G_raw``, ``G_adj``, ``R``; None when a group has an empty bin."""
    n_bins = len(constants.BOX_SIZE_EDGES) + 1
    u, i = ~involved, involved
    n_u = np.bincount(bins[u], minlength=n_bins)
    n_i = np.bincount(bins[i], minlength=n_bins)
    if (n_u == 0).any() or (n_i == 0).any():
        return None
    low_u = np.bincount(bins[u], weights=low[u], minlength=n_bins) / n_u
    low_i = np.bincount(bins[i], weights=low[i], minlength=n_bins) / n_i
    g_raw = float(low[u].mean() - low[i].mean())
    g_adj = float((n_u / n_u.sum() * (low_u - low_i)).sum())
    return {"G_raw": g_raw, "G_adj": g_adj, "R": g_raw - g_adj}


def interval(point: float, values: list[float]) -> dict[str, float]:
    arr = np.asarray([x for x in values if np.isfinite(x)])
    low, high = np.quantile(arr, [ft.ALPHA, 1.0 - ft.ALPHA])
    return {"mean": point, "low": float(low), "high": float(high)}


def read_rho(x: np.ndarray, y: np.ndarray, draws: list[np.ndarray]) -> dict[str, float]:
    return interval(spearman(x, y), [spearman(x[d], y[d]) for d in draws])


def verdict(rho: dict[str, float], parts: dict[str, dict[str, float]]) -> str:
    """BS5: every state is conditioned on its own interval."""
    third = constants.BOX_SIZE_THIRD * parts["G_raw"]["mean"]
    r = parts["R"]
    if rho["low"] > 0 and r["low"] > 0 and r["mean"] >= third:
        return CONSISTENT
    if rho["low"] > 0 and r["high"] < third:
        return RELATED
    if rho["low"] <= 0 <= rho["high"] < constants.BOX_SIZE_RHO_FLOOR and r["high"] < third:
        return NO_RELATION
    return NOT_SETTLED


def read_row(row: eg.Row, feats: dict[str, np.ndarray], involved: np.ndarray,
             controls: list[np.ndarray], groups: list[str]) -> dict[str, Any]:
    """BS3 / BS4 for one row on the boxed clips (all arrays are restricted already)."""
    idx = [str(k) for k in range(len(involved))]
    draws = ft.draws_of(idx, dict(zip(idx, groups, strict=True)))
    out: dict[str, Any] = {"rho": {}}
    for name, mask in (("uninvolved", ~involved), ("involved", involved)):
        pos = np.flatnonzero(mask)
        sub_groups = {str(k): groups[p] for k, p in enumerate(pos)}
        sub_draws = ft.draws_of([str(k) for k in range(len(pos))], sub_groups)
        out["rho"][name] = {
            f: {"A": read_rho(feats[f][pos], row["A"][pos], sub_draws),
                "auc": read_rho(feats[f][pos], row["auc"][pos], sub_draws),
                "A_partial": partial_spearman(feats[f][pos], row["A"][pos],
                                              [c[pos] for c in controls])}
            for f in FEATURES}
    bins = bin_of(feats[SIZE])
    point = gap_parts(row["low"], involved, bins)
    if point is None:
        raise ValueError("BS4: a size bin is empty for one ego group")
    boot = [gap_parts(row["low"][d], involved[d], bins[d]) for d in draws]
    kept = [b for b in boot if b is not None]
    out["dropped_draws"] = len(boot) - len(kept)
    out["gap"] = {k: interval(point[k], [b[k] for b in kept]) for k in point}
    out["bins"] = [
        {"n_u": int(((bins == b) & ~involved).sum()), "n_i": int(((bins == b) & involved).sum()),
         "low_u": float(row["low"][(bins == b) & ~involved].mean()),
         "low_i": float(row["low"][(bins == b) & involved].mean()),
         "auc_u": float(row["auc"][(bins == b) & ~involved].mean()),
         "auc_i": float(row["auc"][(bins == b) & involved].mean())}
        for b in range(len(constants.BOX_SIZE_EDGES) + 1)]
    out["verdict"] = verdict(out["rho"]["uninvolved"][SIZE]["A"], out["gap"])
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = parse_rows(args.dota)
    if missing := [r for r in constants.BOX_SIZE_ROWS if r not in dirs]:
        raise ValueError(f"rows not given: {missing}")
    labels = read_clip_scores(dirs[constants.BOX_SIZE_ROWS[0]] / CLIP_SCORES_NPZ)[1]
    ids = sorted(labels)
    docs = {v: json.loads((args.dota_annotations / f"{v}.json").read_text(encoding="utf-8"))
            for v in ids}
    involved = np.array([bool(docs[v]["ego_involve"]) for v in ids])
    boxes = [clip_boxes(docs[v]) for v in ids]
    boxed = np.array([b is not None for b in boxes])
    feats = {f: np.array([b[f] for b in boxes if b is not None]) for f in FEATURES}
    covs = ft.covariates({v: labels[v] for v in ids}, ft.STEPS_PER_S[ft.DOTA])
    controls = [np.array([c[v] for v in ids])[boxed] for c in covs.values()]
    groups = [dota_group(v) for v in np.array(ids)[boxed]]
    rows = {}
    for name in constants.BOX_SIZE_ROWS:
        row = eg.row_arrays(dirs[name], ids)
        eg.gate_reference(name, row["low"], involved)
        rows[name] = read_row({k: x[boxed] for k, x in row.items()}, feats, involved[boxed],
                              controls, groups)
        rows[name]["unboxed_involved_low"] = float(row["low"][involved & ~boxed].mean())
    report = {"rule": "KIM_RUN.md §47", "seed": ft.SEED, "draws": ft.DRAWS,
              "n": {"boxed_uninvolved": int((boxed & ~involved).sum()),
                    "boxed_involved": int((boxed & involved).sum()),
                    "unboxed": int((~boxed).sum())}, "rows": rows}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / constants.BOX_SIZE_REPORT_STEM
    write_json_atomic(stem.with_suffix(".json"), report)
    write_text_atomic(stem.with_suffix(".md"), render_markdown(report))
    return report


def _ci(d: dict[str, float]) -> str:
    return f"{d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Box size read ({report['rule']}) — printed, decides nothing, n = 1", "",
             f"{report['n']}; {report['draws']} draws, seed {report['seed']}.", ""]
    edges = ("0", *(str(e) for e in constants.BOX_SIZE_EDGES), "∞")
    for name, r in report["rows"].items():
        g = r["gap"]
        lines += [f"## {name}: {r['verdict']}", "",
                  f"G_raw {_ci(g['G_raw'])} · G_adj {_ci(g['G_adj'])} · R {_ci(g['R'])} · "
                  f"dropped draws {r['dropped_draws']} · unboxed involved low "
                  f"{r['unboxed_involved_low']:.3f}", "",
                  "| group | feature | rho(A_c) | rho(own AUC) | rho(A_c), covariates removed |",
                  "| --- | --- | --- | --- | --- |"]
        for grp, fs in r["rho"].items():
            lines += [f"| {grp} | {f} | {_ci(x['A'])} | {_ci(x['auc'])} | {x['A_partial']:+.3f} |"
                      for f, x in fs.items()]
        lines += ["", "| size bin | n U / I | low U / I | own AUC U / I |",
                  "| --- | --- | --- | --- |"]
        lines += [f"| {edges[k]} to {edges[k + 1]} | {b['n_u']} / {b['n_i']} | "
                  f"{b['low_u']:.3f} / {b['low_i']:.3f} | {b['auc_u']:.3f} / {b['auc_i']:.3f} |"
                  for k, b in enumerate(r["bins"])]
        lines.append("")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument("--dota", nargs="+", required=True, help="ROW=DIR")
    for flag in ("--dota-annotations", "--out-dir"):
        parser.add_argument(flag, type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = run(build_arg_parser().parse_args(argv))
    LOGGER.info("%s", {k: r["verdict"] for k, r in report["rows"].items()})


if __name__ == "__main__":
    main()
