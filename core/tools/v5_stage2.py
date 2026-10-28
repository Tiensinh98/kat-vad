"""v5 Stage 2 read-out: detection, guardrails and the decision (PREREG_V5 S2-7, S2-9, S2-10, S2-14).

Pure aggregation -- no model is loaded. Inputs (``colab/v5/stage2_train_read.ipynb``)::

    <pb-dir>/<row>/{protocol_b_readout.json, clip_aucs.json, clip_scores.npz}   protocol_b_eval
    <nexar-dir>/<row>/nexar_readout.json                                        nexar_eval
    <maps-readout>  = maps_readout.json                                         v5_maps read (S2-8)
    <provenance>    = g4_provenance.json (one or more; every row/seed once)     v5_stage1 provenance
    <collapse>      = g6_collapse.json                                          v5_stage1 collapse
    <runs-root>/<row>/s<seed>/metrics.jsonl                                     motion share
    <prior>         = v2 E3 position_prior.npz on DoTA-CAP-dev (p_T2, f_2)      printed
    <r0-pb>, <a2-pb> = v2 Nexar A0 / A2 protocol_b_readout.json on DoTA-CAP-dev printed

**S2-7** per row R against R1 (R5 and R5Δ primary; R3 printed): the cluster bootstrap over DoTA
source videos of the per-clip AUC difference on seed-averaged curves, B = 10,000, seed 2026, a
97.5 % interval. "improves" iff (i) low > 0, (ii) per-seed paired Δ > 0 on every seed, (iii) point
>= +0.01; "costs" iff high < 0; otherwise "no difference shown at n = 3". The paired t over seeds
is printed only.

**S2-9** a row is adoptable iff its ``nexar_val`` crop endpoint (seed mean) >= R1's - 0.01 and
every one of its runs passed S1-G4.

**S2-10** the candidate is R5; R5Δ replaces it only if R5 has neither sentence (detection
"improves", maps "point") and R5Δ has one. The candidate's two sentences pick the outcome. A
candidate that fails S2-9 is not adopted: outcome ``NOT-ADOPTED (S2-9)``, which the table does
not cover, so it goes to the user (§2.4 B10).

**Preliminary read (§2.4 B11).** With fewer seeds than ``V5_RUN_SEEDS`` the same statistics are
computed, but S2-7 (ii) cannot be met as signed, so the outcome is labelled ``PRELIMINARY``, no row
is adopted, and S2-11 stays closed.

**§2.4 B13.** ``--primary R5q R5dq`` reads the query-LR rows in place of R5 / R5Δ (same
statistics, same table); ``--split dota_cap_eval`` reads B13's decision set (protocol B under
``--final``); the recorded CAP-dev ``p_T2`` is printed only on CAP-dev.

**AP (§2.4 B12, printed only).** Per-clip AP macro (R - R1 at 95 %) and pooled frame AP after
per-clip min-max on DoTA-CAP-dev; pooled raw frame AP on ``nexar_val``. Never gating.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.v2_splits import dota_group
from core.metrics import cluster_bootstrap_ci, frame_ap, normalize_scores
from core.tools import nexar_eval
from core.tools import protocol_b_eval as pb
from core.tools.final_readout import fused_clip_aucs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.v2_diagnostics import motion_share

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/v5/PREREG_V5.md S2-7, S2-9, S2-10, S2-14 (+ §2.4 B10)"
READOUT_JSON = "stage2_readout.json"
READOUT_MD = "stage2_readout.md"
CONTROL = constants.V5_CONTROL_ROW
IMPROVES, COSTS, NO_DIFF = "improves", "costs", "no difference shown at n = 3"
OUTCOMES = {  # S2-10: (detection improves, maps point) -> outcome
    (True, True): ("GO-MQR", "seeds 2027-2028 for the adopted row; proposal edited; Stage 3"),
    (True, False): ("DETECT-ONLY", "H2 dropped from the claim; proposal edited; Stage 3 on the "
                    "user's word"),
    (False, True): ("MAPS-ONLY", "claim = weakly supervised maps that point, at no detection "
                    "gain; no Stage 3 for H1"),
    (False, False): ("NULL", "MQR closed; v5 written up as a measured negative with Stage 0"),
}
SPLIT_LABEL = {constants.V2_SPLIT_DOTA_CAP_DEV: "dev", constants.V2_SPLIT_DOTA_CAP_EVAL: "eval"}
GUARD_FAIL = ("NOT-ADOPTED (S2-9)", "the candidate failed a guardrail; S2-10's table does not "
              "cover this -- the user decides (§2.4 B10)")


def seed_name(seed: int) -> str:
    return f"s{seed}"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- inputs


def load_pb_row(pb_dir: Path, row: str, seeds: list[str]) -> dict[str, Any]:
    """One row's protocol B read: per-seed macro, seed-averaged clip AUCs and scores."""
    d = pb_dir / row
    readout = _json(d / pb.READOUT_JSON)
    if sorted(readout["per_run_macro"]) != sorted(seeds):
        raise ValueError(f"{row}: protocol B read seeds {sorted(readout['per_run_macro'])}, "
                         f"expected {sorted(seeds)}")
    scores, labels = pb.read_clip_scores(d / pb.CLIP_SCORES_NPZ)
    return {
        "readout": readout,
        "clip_aucs": {k: float(v) for k, v in _json(d / pb.CLIP_AUCS_JSON).items()},
        "scores": scores,
        "labels": labels,
    }


def check_same_clips(rows: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """Every row read the same clips with the same labels; returns those labels."""
    first = next(iter(rows))
    labels: dict[str, np.ndarray] = rows[first]["labels"]
    for row, r in rows.items():
        if set(r["labels"]) != set(labels):
            raise ValueError(f"{first} and {row} protocol B reads cover different clips")
        for v, y in labels.items():
            if not np.array_equal(y, r["labels"][v]):
                raise ValueError(f"{v}: {first} and {row} protocol B labels differ")
    return labels


def load_nexar_row(nexar_dir: Path, row: str, seeds: list[str]) -> dict[str, Any]:
    readout = _json(nexar_dir / row / nexar_eval.READOUT_JSON)
    endpoint = readout["per_run_crop_endpoint"]
    if sorted(endpoint) != sorted(seeds):
        raise ValueError(f"{row}: nexar_val seeds {sorted(endpoint)}, expected {sorted(seeds)}")
    return {
        "crop_endpoint_per_seed": {s: float(endpoint[s]) for s in seeds},
        "crop_endpoint_mean": float(np.mean([endpoint[s] for s in seeds])),
        "whole_macro": readout["whole"]["macro"],
        "whole_micro_ap": readout["whole"].get("micro_ap"),
        "positive_frame_share": readout["whole"].get("positive_frame_share"),
    }


def provenance_by_run(paths: list[Path]) -> dict[str, bool]:
    """``{"<row>/s<seed>": pass}`` over every S1-G4 report; a run listed twice must agree."""
    out: dict[str, bool] = {}
    for path in paths:
        for run, r in _json(path)["runs"].items():
            if run in out and out[run] != bool(r["pass"]):
                raise ValueError(f"{run}: S1-G4 disagrees between reports")
            out[run] = bool(r["pass"])
    return out


# --------------------------------------------------------------------------- statistics


def clip_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    if set(a) != set(b):
        raise ValueError("per-clip AUCs of the two rows cover different clips")
    return {v: a[v] - b[v] for v in sorted(a)}


def bootstrap(values: dict[str, float], level: float) -> dict[str, float] | None:
    groups = {v: dota_group(v) for v in values}
    return cluster_bootstrap_ci(
        values, groups, constants.V5_BOOTSTRAP, constants.V5_BOOTSTRAP_SEED, level
    )


def paired_t(per_seed: dict[str, float]) -> dict[str, float] | None:
    """Mean ± t(0.975, 2) · SD / √n of the per-seed Δ (printed only, S2-7); None below 2 seeds."""
    if len(per_seed) < 2:
        return None
    deltas = np.array([per_seed[s] for s in sorted(per_seed)], dtype=np.float64)
    mean = float(deltas.mean())
    half = constants.V5_T95_N3 * float(deltas.std(ddof=1)) / math.sqrt(len(deltas))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": len(deltas)}


def detection_sentence(ci: dict[str, float] | None, per_seed: dict[str, float]) -> str:
    """S2-7 (i)-(iii); no tie clause."""
    if ci is None:
        raise ValueError("no clips to read")
    if (ci["low"] > 0 and all(d > 0 for d in per_seed.values())
            and ci["mean"] >= constants.V5_MIN_GAIN):
        return IMPROVES
    if ci["high"] < 0:
        return COSTS
    return NO_DIFF


def detection(row: dict[str, Any], control: dict[str, Any], seeds: list[str]) -> dict[str, Any]:
    """One row's S2-7 read against R1."""
    per_seed = {
        s: float(row["readout"]["per_run_macro"][s] - control["readout"]["per_run_macro"][s])
        for s in seeds
    }
    ci = bootstrap(clip_delta(row["clip_aucs"], control["clip_aucs"]),
                   constants.V5_PRIMARY_LEVEL)
    return {
        "clip_delta": ci,
        "per_seed_delta": per_seed,
        "paired_t_printed": paired_t(per_seed),
        "sentence": detection_sentence(ci, per_seed),
    }


def guardrails(
    row: str, nexar: dict[str, dict[str, Any]], prov: dict[str, bool], seeds: list[str]
) -> dict[str, Any]:
    """S2-9 for one row."""
    delta = nexar[row]["crop_endpoint_mean"] - nexar[CONTROL]["crop_endpoint_mean"]
    runs = {f"{row}/{s}": prov.get(f"{row}/{s}") for s in seeds}
    missing = sorted(k for k, v in runs.items() if v is None)
    if missing:
        raise ValueError(f"no S1-G4 result for {missing}")
    checks = {
        "nexar_val_crop_endpoint": delta >= -constants.V5_NEXAR_GUARD_MARGIN,
        "s1_g4_every_run": all(bool(v) for v in runs.values()),
    }
    return {"nexar_val_crop_delta": delta, "s1_g4": runs, "checks": checks,
            "hold": all(checks.values())}


def has_sentence(det: dict[str, Any], points: bool) -> bool:
    return det["sentence"] == IMPROVES or points


def decide(
    det: dict[str, dict[str, Any]],
    points: dict[str, bool],
    guards: dict[str, dict[str, Any]],
    n_seeds: int = len(constants.V5_RUN_SEEDS),
    primary: tuple[str, str] = constants.V5_PRIMARY_ROWS,
) -> dict[str, Any]:
    """S2-10: the candidate row, its outcome and the next step (``PRELIMINARY`` below 3 seeds).

    ``primary`` = (content row, change row): R5 / R5Δ, or B13's R5q / R5Δq.
    """
    first, second = primary
    replaced = (not has_sentence(det[first], points[first])
                and has_sentence(det[second], points[second]))
    candidate = second if replaced else first
    improves = det[candidate]["sentence"] == IMPROVES
    outcome, step = OUTCOMES[(improves, points[candidate])]
    adoptable = guards[candidate]["hold"]
    if outcome != "NULL" and not adoptable:
        outcome, step = GUARD_FAIL
    preliminary = n_seeds < len(constants.V5_RUN_SEEDS)
    if preliminary:
        outcome = f"PRELIMINARY {outcome} (n = {n_seeds} seed{'s' if n_seeds > 1 else ''})"
        step = ("not the S2-10 decision (§2.4 B11): train the remaining seeds of "
                f"{', '.join(map(str, constants.V5_RUN_SEEDS))} and re-read")
    return {
        "candidate": candidate,
        "r5d_replaced_r5": replaced,
        "detection": det[candidate]["sentence"],
        "maps_point": points[candidate],
        "guardrails_hold": adoptable,
        "outcome": outcome,
        "next": step,
        "adopted": candidate if outcome in ("GO-MQR", "DETECT-ONLY", "MAPS-ONLY") else None,
        "preliminary": preliminary,
    }


# --------------------------------------------------------------------------- printed (S2-14)


def f2_reads(
    rows: dict[str, dict[str, Any]], prior: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> dict[str, Any]:
    """v2's beyond-position read ``z(y) + 2 z(p_T2)``: macro per row, Δ vs R1 (95 %)."""
    aucs = {r: fused_clip_aucs(d["scores"], prior, labels, constants.V5_F2_WEIGHT)
            for r, d in rows.items()}
    return {
        "weight": constants.V5_F2_WEIGHT,
        "macro": {r: float(np.mean(list(a.values()))) for r, a in aucs.items()},
        "delta_vs_r1": {r: bootstrap(clip_delta(aucs[r], aucs[CONTROL]),
                                     constants.V5_BOOTSTRAP_LEVEL)
                        for r in aucs if r != CONTROL},
    }


def clip_aps(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-clip frame AP over the two-class clips (the clips S2-7's macro AUC reads)."""
    return {v: frame_ap(scores[v], labels[v]) for v in sorted(scores)
            if 0 < int(labels[v].sum()) < len(labels[v])}


def pooled_minmax_ap(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> float:
    """Frame AP pooled over every clip after per-clip min-max (LaGoVAD's DoTA pooling)."""
    ids = sorted(scores)
    return frame_ap(
        np.concatenate([normalize_scores(scores[v], constants.SCORE_NORM_MINMAX) for v in ids]),
        np.concatenate([labels[v] for v in ids]),
    )


def ap_reads(
    curves: dict[str, dict[str, np.ndarray]], labels: dict[str, np.ndarray]
) -> dict[str, Any]:
    """§2.4 B12: AP beside AUC on DoTA-CAP-dev (printed). ``curves`` must hold ``CONTROL``."""
    aps = {r: clip_aps(c, labels) for r, c in curves.items()}
    return {
        "clip_macro": {r: float(np.mean(list(a.values()))) for r, a in aps.items()},
        "clip_delta_vs_r1": {r: bootstrap(clip_delta(aps[r], aps[CONTROL]),
                                          constants.V5_BOOTSTRAP_LEVEL)
                             for r in aps if r != CONTROL},
        "pooled_minmax": {r: pooled_minmax_ap(c, labels) for r, c in curves.items()},
        "positive_frame_share": float(np.concatenate(list(labels.values())).mean()),
    }


def share_reads(runs_root: Path, rows: list[str], seeds: list[str]) -> dict[str, Any]:
    """Last-step ``motion_share`` / ``w_u_norm`` per run (printed)."""
    out: dict[str, Any] = {}
    for row in rows:
        for s in seeds:
            path = runs_root / row / s / "metrics.jsonl"
            share = motion_share(path) if path.is_file() else None
            out[f"{row}/{s}"] = None if share is None else {k: v["last"] for k, v in share.items()}
    return out


def borrowed_macro(path: Path | None, seeds: list[str]) -> dict[str, Any] | None:
    """A v2 run's DoTA-CAP-dev read, restricted to ``seeds`` (printed)."""
    if path is None:
        return None
    readout = _json(path)
    per = {s: float(readout["per_run_macro"][s]) for s in seeds if s in readout["per_run_macro"]}
    return {
        "seed_averaged_macro_all_seeds": readout["macro"]["mean"],
        "seeds_in_file": sorted(readout["per_run_macro"]),
        "per_seed_macro": per,
        "per_seed_mean": float(np.mean(list(per.values()))) if per else None,
    }


# --------------------------------------------------------------------------- run


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = [seed_name(s) for s in args.seeds]
    rows = [CONTROL, *args.rows]
    pbs = {r: load_pb_row(args.pb_dir, r, seeds) for r in rows}
    labels = check_same_clips(pbs)
    nexar = {r: load_nexar_row(args.nexar_dir, r, seeds) for r in rows}
    maps = _json(args.maps_readout)["rows"]
    prov = provenance_by_run(args.provenance)
    collapse = _json(args.collapse)["runs"] if args.collapse else {}

    det = {r: detection(pbs[r], pbs[CONTROL], seeds) for r in args.rows}
    if len(seeds) < len(constants.V5_RUN_SEEDS):
        for d in det.values():
            if d["sentence"] == NO_DIFF:
                d["sentence"] = f"no difference shown at n = {len(seeds)}"
    primary: tuple[str, str] = (args.primary[0], args.primary[1])
    if not set(primary) <= set(args.rows):
        raise ValueError(f"--primary {list(primary)} must be among --rows {args.rows}")
    points = {r: bool(maps[r]["points"]) for r in args.rows if r in maps}
    missing_maps = [r for r in primary if r not in points]
    if missing_maps:
        raise ValueError(f"no S2-8 maps read for {missing_maps}")
    guards = {r: guardrails(r, nexar, prov, seeds) for r in args.rows}
    decision = decide(det, points, guards, len(seeds), primary)

    prior = None
    if args.prior is not None:
        prior, _ = pb.read_clip_scores(args.prior)
    result: dict[str, Any] = {
        "rule": RULE,
        "set": f"DoTA-CAP (n/1397) {SPLIT_LABEL[args.split]}, protocol B",
        "split": args.split,
        "primary": list(primary),
        "seeds": seeds,
        "clips_two_class": len(pbs[CONTROL]["clip_aucs"]),
        "macro": {r: float(np.mean(list(d["clip_aucs"].values()))) for r, d in pbs.items()},
        "per_seed_macro": {r: d["readout"]["per_run_macro"] for r, d in pbs.items()},
        "detection": det,
        "maps": {r: {k: maps[r][k] for k in ("hit_minus_chance", "hit_minus_pre",
                                             "hit_minus_centre", "points", "stauc")}
                 for r in maps},
        "nexar_val": nexar,
        "guardrails": guards,
        "decision": decision,
        "printed": {
            "p_t2_macro": (None if prior is None else
                           float(np.mean(list(pb.clip_aucs(prior, labels).values())))),
            "p_t2_recorded": (constants.V5_P_T2_CAP_DEV
                              if args.split == constants.V2_SPLIT_DOTA_CAP_DEV else None),
            "f2": None if prior is None else f2_reads(pbs, prior, labels),
            "ap": ap_reads({r: d["scores"] for r, d in pbs.items()}, labels),
            "ap_p_t2": (None if prior is None else
                        {"clip_macro": float(np.mean(list(clip_aps(prior, labels).values()))),
                         "pooled_minmax": pooled_minmax_ap(prior, labels)}),
            "r0_v2_nexar_a0": borrowed_macro(args.r0_pb, seeds),
            "v2_nexar_a2": borrowed_macro(args.a2_pb, seeds),
            "collapse": collapse,
            "motion_share_last": (None if args.runs_root is None
                                  else share_reads(args.runs_root, rows, seeds)),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / READOUT_JSON, result)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(result))
    LOGGER.info("Stage 2: %s (candidate %s) -> %s", decision["outcome"], decision["candidate"],
                args.out_dir)
    return result


# --------------------------------------------------------------------------- markdown


def _ci(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def render_markdown(r: dict[str, Any]) -> str:
    d = r["decision"]
    lines = [
        "# v5 Stage 2 read-out",
        "",
        f"Rule: {r['rule']}. Set: {r['set']}, {r['clips_two_class']} two-class clips, seeds "
        f"{', '.join(r['seeds'])}.",
        "",
        f"## Decision (S2-10): **{d['outcome']}**",
        "",
        *(["> **PRELIMINARY (§2.4 B11):** fewer seeds than signed; S2-7 (ii) is read on these "
           "seeds only and the paired t is not defined at n = 1. Report it as preliminary; "
           "the S2-10 decision needs every seed.", ""] if d.get("preliminary") else []),
        f"- candidate **{d['candidate']}**"
        + (f" ({r['primary'][1]} replaced {r['primary'][0]}: {r['primary'][0]} had no sentence)"
           if d["r5d_replaced_r5"] else ""),
        f"- detection: {d['detection']}; maps point: {'yes' if d['maps_point'] else 'no'}; "
        f"guardrails hold: {'yes' if d['guardrails_hold'] else 'no'}",
        f"- next: {d['next']}",
        "",
        "## Detection (S2-7) — R - R1, cluster bootstrap 97.5 % over source videos",
        "",
        "| row | macro | Δ per clip [97.5 %] | per-seed Δ | paired t (printed) | sentence |",
        "|---|---|---|---|---|---|",
        f"| {CONTROL} | {r['macro'][CONTROL]:.4f} | — | — | — | control |",
    ]
    for row, det in r["detection"].items():
        tag = " (printed)" if row not in r["primary"] else ""
        lines.append(
            f"| {row}{tag} | {r['macro'][row]:.4f} | {_ci(det['clip_delta'])} | "
            + ", ".join(f"{v:+.4f}" for v in det["per_seed_delta"].values())
            + f" | {_ci(det['paired_t_printed'])} | {det['sentence']} |"
        )
    lines += [
        "",
        "## Maps (S2-8) — 95 % cluster bootstrap; see `maps_readout.md`",
        "",
        "| row | hit - chance | lift - lift_pre | hit - centre | points? | STAUC |",
        "|---|---|---|---|---|---|",
    ]
    for row, m in r["maps"].items():
        lines.append(
            f"| {row} | {_ci(m['hit_minus_chance'])} | {_ci(m['hit_minus_pre'])} | "
            f"{_ci(m['hit_minus_centre'])} | {'**yes**' if m['points'] else 'no'} | "
            f"{m['stauc']['stauc']:.4f} |"
        )
    lines += [
        "",
        "## Guardrails (S2-9)",
        "",
        "| row | nexar_val crop endpoint (seed mean) | Δ vs R1 | S1-G4 | hold |",
        "|---|---|---|---|---|",
        f"| {CONTROL} | {r['nexar_val'][CONTROL]['crop_endpoint_mean']:.4f} | — | — | — |",
    ]
    for row, g in r["guardrails"].items():
        lines.append(
            f"| {row} | {r['nexar_val'][row]['crop_endpoint_mean']:.4f} | "
            f"{g['nexar_val_crop_delta']:+.4f} | "
            f"{'pass' if g['checks']['s1_g4_every_run'] else 'FAIL'} | "
            f"{'yes' if g['hold'] else '**no**'} |"
        )
    p = r["printed"]
    ap = p["ap"]
    lines += [
        "",
        "## AP (§2.4 B12) — printed, never gating",
        "",
        "| row | DoTA per-clip AP macro | Δ vs R1 [95 %] | DoTA pooled AP (min-max) | "
        "nexar_val pooled AP (raw) |",
        "|---|---|---|---|---|",
    ]
    for row in ap["clip_macro"]:
        lines.append(
            f"| {row} | {ap['clip_macro'][row]:.4f} | {_ci(ap['clip_delta_vs_r1'].get(row))} | "
            f"{ap['pooled_minmax'][row]:.4f} | {_num(r['nexar_val'][row]['whole_micro_ap'])} |"
        )
    if p["ap_p_t2"] is not None:
        lines.append(f"| `p_T2` (printed) | {p['ap_p_t2']['clip_macro']:.4f} | — | "
                     f"{p['ap_p_t2']['pooled_minmax']:.4f} | — |")
    lines += [
        "",
        f"Positive-frame share (an all-equal score's pooled AP): DoTA-CAP {r['split']} "
        f"{ap['positive_frame_share']:.4f}, nexar_val "
        f"{_num(r['nexar_val'][CONTROL]['positive_frame_share'])}.",
        "",
        "## Printed, never gating (S2-14)",
        "",
    ]
    lines.append(f"- `p_T2` on these clips: {_num(p['p_t2_macro'])}"
                 + ("" if p["p_t2_recorded"] is None else f" (recorded {p['p_t2_recorded']:.4f})"))
    for name, label in (("v2_nexar_a2", "v2 Nexar A2"), ("r0_v2_nexar_a0", "R0 = v2 Nexar A0")):
        b = p[name]
        if b is not None:
            lines.append(f"- {label}: seed-averaged macro over {len(b['seeds_in_file'])} seeds "
                         f"{b['seed_averaged_macro_all_seeds']:.4f}; mean of "
                         f"{', '.join(b['per_seed_macro'])} {_num(b['per_seed_mean'])}")
    if p["f2"] is not None:
        lines.append("- `f_2` (DoTA-CAP placement) macro: "
                     + ", ".join(f"{k} {v:.4f}" for k, v in p["f2"]["macro"].items())
                     + "; Δ vs R1 (95 %): "
                     + ", ".join(f"{k} {_ci(v)}" for k, v in p["f2"]["delta_vs_r1"].items()))
    lines.append("- `nexar_val` whole-video macro: "
                 + ", ".join(f"{k} {_ci(v['whole_macro'])}" for k, v in r["nexar_val"].items()))
    for run_name, c in p["collapse"].items():
        ent = ", ".join(f"{e:.3f}" for e in c["entropy"].values())
        spread = ", ".join(f"{e:.3f}" for e in c.get("logit_std", {}).values())
        lines.append(f"- S1-G6 {run_name}: entropy [{ent}]"
                     + (f", logit std [{spread}]" if spread else "")
                     + (" — **did not leave the mean**" if c["did_not_leave_mean"] else ""))
    if p["motion_share_last"]:
        lines.append("- last-step motion share / ‖W_u‖: " + "; ".join(
            f"{k} {v}" for k, v in p["motion_share_last"].items()))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--pb-dir", type=Path, required=True,
                        help="<row>/ protocol_b_eval outputs on dota_cap_dev")
    parser.add_argument("--nexar-dir", type=Path, required=True,
                        help="<row>/ nexar_eval outputs on nexar_val")
    parser.add_argument("--maps-readout", type=Path, required=True, help="maps_readout.json")
    parser.add_argument("--provenance", type=Path, nargs="+", required=True,
                        help="g4_provenance.json of Stage 1 (R1) and Stage 2 (new rows)")
    parser.add_argument("--collapse", type=Path, default=None, help="g6_collapse.json")
    parser.add_argument("--runs-root", type=Path, default=None,
                        help="<row>/s<seed>/metrics.jsonl (motion share, printed)")
    parser.add_argument("--prior", type=Path, default=None,
                        help="v2 E3 position_prior.npz on dota_cap_dev (p_T2, f_2; printed)")
    parser.add_argument("--r0-pb", type=Path, default=None,
                        help="v2 Nexar A0 protocol_b_readout.json on dota_cap_dev (printed)")
    parser.add_argument("--a2-pb", type=Path, default=None,
                        help="v2 Nexar A2 protocol_b_readout.json on dota_cap_dev (printed)")
    parser.add_argument("--rows", nargs="+", default=list(constants.V5_STAGE2_ROWS),
                        choices=[r for r in constants.V5_ROWS if r != CONTROL])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(constants.V5_RUN_SEEDS))
    parser.add_argument("--primary", nargs=2, default=list(constants.V5_PRIMARY_ROWS),
                        metavar=("CONTENT", "CHANGE"),
                        help="S2-10's (R5, R5Δ); §2.4 B13 reads R5q R5dq")
    parser.add_argument("--split", choices=list(SPLIT_LABEL),
                        default=constants.V2_SPLIT_DOTA_CAP_DEV, help="the CAP split --pb-dir / --maps-readout were read on (B13: eval)")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
