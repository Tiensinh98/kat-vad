"""v5.1 pilot read-out: position-controlled detection, maps gate, guardrail (PREREG_V5 §3).

Pure aggregation -- no model is loaded. Inputs (``colab/v5/v51_tile_pilot.ipynb``)::

    <pb-dir>/<row>/{protocol_b_readout.json, clip_aucs.json, clip_scores.npz}      protocol_b_eval
    <pb-rev-dir>/<row>/protocol_b_readout.json   (optional)                         --reverse-time
    <nexar-dir>/<row>/nexar_readout.json                                            nexar_eval
    <maps-readout> = maps_readout.json                                              v5_maps read

Per row R against the control (R1c), on DoTA-CAP-dev at protocol B:

* **clip macro** -- Δ of per-clip AUC, 95 % cluster bootstrap over DoTA source videos; TOST at
  ±``V5_TILE_TOST_MARGIN`` (90 % interval): "equivalent", "different" (95 % excludes 0) or
  "inconclusive".
* **position-controlled (a)** -- AUC over (anomalous, normal) frame pairs from *different* clips
  in the same relative-position bin, scores min-max per clip (point estimate, printed).
* **position-controlled (b)** -- per-clip AUC of ``z(y) + 2 z(p)`` with ``p`` a cubic in
  ``t / (n - 1)`` fitted on the same clips' labels (position only, no pixels); its macro minus
  ``p``'s alone is the gain beyond position; Δ vs the control at 95 %. **Gated** (§3).
* **shift test** -- macro of the score shifted by ``o`` eval rows; the peak must sit within
  ``V5_TILE_SHIFT_TOLERANCE`` rows of 0 (centred windows).
* **time-reversed control** (printed) -- macro with the steps fed last-first.
* **subsets** (printed, proposal §4.4) -- AUC and (b) by ego-involved / ego-uninvolved and by
  clip box-share bin, Δ vs the control (95 %); AP (clip macro + pooled after min-max).
* **mechanics** (printed) -- each run's last-step losses finite, tile scores on abnormal vs
  normal videos, temperature, readout entropy / log 196 (a diagnostic, never a criterion).

**Pilot gate (§3, one seed, PRELIMINARY):** a candidate passes iff (i) its maps beat the centre
prior *and* the untrained change map on pointing (``beats_centre_and_change``), (ii) its
``nexar_val`` crop endpoint >= the control's - ``V5_NEXAR_GUARD_MARGIN``, (iii) its
position-controlled (b) Δ is not shown below 0 (95 % high >= 0). Pass -> 3-seed confirmation.

**R7 fast pilot (``--gate r7``, PREREG_V5 §4.2):** (i) is the P-e rule (``r7_points``: hit - chance,
lift - lift_pre, hit - C14 change map, mass - centre, all 95 % lows > 0); (ii) is **not read**
without ``--nexar-dir`` (the fast pilot extracts no ``nexar_val`` grids) and is reported as such;
(iii) as v5.1. Outcome label MAPS-ONLY-PILOT / NULL-PILOT (§4.2 P-h).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import frame_auc, normalize_scores
from core.tools.final_readout import fused_clip_aucs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.v5_maps import dev_meta
from core.tools.v5_stage0 import BIN_NAMES
from core.tools.v5_stage2 import (
    bootstrap,
    check_same_clips,
    clip_aps,
    clip_delta,
    load_nexar_row,
    load_pb_row,
    pooled_minmax_ap,
    seed_name,
)

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/v5/PREREG_V5.md §3"
READOUT_JSON = "v51_pilot_readout.json"
READOUT_MD = "v51_pilot_readout.md"
R7_READOUT_JSON = "r7_pilot_readout.json"
EQUIVALENT, DIFFERENT, INCONCLUSIVE = "equivalent", "different", "inconclusive"
PASS, FAIL = "PILOT-PASS", "PILOT-FAIL"
GATE_V51, GATE_R7 = "v51", "r7"
R7_PASS, R7_NULL = "MAPS-ONLY-PILOT", "NULL-PILOT"
NOT_READ = "not read"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- position


def relative_position(n: int) -> np.ndarray:
    out: np.ndarray = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    return out


def fit_position(labels: dict[str, np.ndarray]) -> np.ndarray:
    """Least-squares cubic of the frame label on ``t / (n - 1)`` over every clip (D5)."""
    x = np.concatenate([relative_position(len(y)) for y in labels.values()])
    y = np.concatenate([np.asarray(v, dtype=np.float64) for v in labels.values()])
    coeffs: np.ndarray = np.polyfit(x, y, constants.V5_TILE_POS_DEGREE)
    return coeffs


def position_scores(labels: dict[str, np.ndarray], coeffs: np.ndarray) -> dict[str, np.ndarray]:
    return {v: np.polyval(coeffs, relative_position(len(y))) for v, y in labels.items()}


def _mann_whitney(pos: np.ndarray, neg: np.ndarray) -> float:
    """Sum over (pos, neg) pairs of [pos > neg] + 0.5 [pos == neg]."""
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):  # average ranks over ties
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float(ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)


def cross_clip_position_auc(
    scores: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    bins: int = constants.V5_TILE_POS_BINS,
) -> float | None:
    """Form (a): pairs of an anomalous and a normal frame from different clips, same position bin.

    Scores are min-max normalized per clip. Per bin the pooled Mann-Whitney count minus every
    clip's own count leaves exactly the cross-clip pairs; bins are pooled by pair count.
    """
    wins = pairs = 0.0
    per_bin: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(bins)]
    for v in sorted(scores):
        y = np.asarray(labels[v]).astype(bool)
        s = normalize_scores(scores[v], constants.SCORE_NORM_MINMAX)
        b = np.minimum((relative_position(len(y)) * bins).astype(int), bins - 1)
        for k in range(bins):
            sel = b == k
            per_bin[k].append((s[sel & y], s[sel & ~y]))
    for clips in per_bin:
        pos = np.concatenate([p for p, _ in clips])
        neg = np.concatenate([n for _, n in clips])
        own = sum(_mann_whitney(p, n) for p, n in clips)
        own_pairs = sum(len(p) * len(n) for p, n in clips)
        wins += _mann_whitney(pos, neg) - own
        pairs += len(pos) * len(neg) - own_pairs
    return None if pairs == 0 else wins / pairs


def fused_clips(
    curves: dict[str, dict[str, np.ndarray]], labels: dict[str, np.ndarray]
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Per-clip AUC of ``z(y) + 2 z(p)`` per row, and of ``p`` alone."""
    prior = position_scores(labels, fit_position(labels))
    fused = {
        r: fused_clip_aucs(c, prior, labels, constants.V5_TILE_POS_WEIGHT)
        for r, c in curves.items()
    }
    return fused, fused_clip_aucs(prior, prior, labels, None)


def fused_reads(
    curves: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    control: str,
) -> dict[str, Any]:
    """Form (b): macro of ``z(y) + 2 z(p)`` per row, ``p`` alone, Δ vs the control (95 %)."""
    fused, alone = fused_clips(curves, labels)
    macro_p = float(np.mean(list(alone.values())))
    return {
        "weight": constants.V5_TILE_POS_WEIGHT,
        "position_alone_macro": macro_p,
        "macro": {r: float(np.mean(list(a.values()))) for r, a in fused.items()},
        "gain_beyond_position": {
            r: float(np.mean(list(a.values()))) - macro_p for r, a in fused.items()
        },
        "delta_vs_control": {
            r: bootstrap(clip_delta(fused[r], fused[control]), constants.V5_BOOTSTRAP_LEVEL)
            for r in fused
            if r != control
        },
    }


# --------------------------------------------------------------------------- alignment


def shift_scores(score: np.ndarray, rows: int, stride: int) -> np.ndarray:
    """``s'[t] = s[t - rows * stride]``, edge values repeated."""
    n = len(score)
    index = np.clip(np.arange(n) - rows * stride, 0, n - 1)
    out: np.ndarray = np.asarray(score)[index]
    return out


def macro_auc(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> float:
    aucs = [
        frame_auc(np.asarray(scores[v], dtype=np.float64), np.asarray(labels[v]))
        for v in sorted(scores)
        if 0 < int(np.asarray(labels[v]).sum()) < len(labels[v])
    ]
    return float(np.mean(aucs))


def shift_test(
    scores: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    stride: int,
    offsets: tuple[int, ...] = constants.V5_TILE_SHIFT_ROWS,
) -> dict[str, Any]:
    curve = {
        o: macro_auc({v: shift_scores(s, o, stride) for v, s in scores.items()}, labels)
        for o in offsets
    }
    peak = max(curve, key=lambda o: curve[o])
    return {
        "macro_by_offset": {str(o): m for o, m in curve.items()},
        "peak_offset": peak,
        "pass": abs(peak) <= constants.V5_TILE_SHIFT_TOLERANCE,
    }


# --------------------------------------------------------------------------- statistics


def tost(values: dict[str, float]) -> dict[str, Any]:
    """Equivalence at ±margin (90 % interval) and difference (95 %) of a clip-level Δ."""
    narrow = bootstrap(values, constants.V5_TILE_TOST_LEVEL)
    wide = bootstrap(values, constants.V5_BOOTSTRAP_LEVEL)
    if narrow is None or wide is None:
        raise ValueError("no clips to read")
    m = constants.V5_TILE_TOST_MARGIN
    if wide["low"] > 0 or wide["high"] < 0:
        verdict = DIFFERENT
    elif -m < narrow["low"] and narrow["high"] < m:
        verdict = EQUIVALENT
    else:
        verdict = INCONCLUSIVE
    return {"ci95": wide, "ci90": narrow, "margin": m, "verdict": verdict}


def subset_reads(
    per_clip: dict[str, dict[str, float]], groups: dict[str, str | None], control: str
) -> dict[str, Any]:
    """Macro per row and Δ vs the control (95 %) inside each clip group (ego, box-share bin)."""
    out: dict[str, Any] = {}
    for name in sorted({g for g in groups.values() if g is not None}):
        clips = {v for v, g in groups.items() if g == name}
        sub = {r: {v: a for v, a in aucs.items() if v in clips} for r, aucs in per_clip.items()}
        if not sub[control]:
            continue
        out[name] = {
            "clips": len(sub[control]),
            "macro": {r: float(np.mean(list(a.values()))) for r, a in sub.items() if a},
            "delta_vs_control": {
                r: bootstrap(clip_delta(sub[r], sub[control]), constants.V5_BOOTSTRAP_LEVEL)
                for r in sub
                if r != control
            },
        }
    return out


def ap_summary(
    curves: dict[str, dict[str, np.ndarray]], labels: dict[str, np.ndarray], control: str
) -> dict[str, Any]:
    """Proposal §4.4 AP beside AUC: clip-macro AP (Δ vs control, 95 %) and pooled min-max AP."""
    aps = {r: clip_aps(c, labels) for r, c in curves.items()}
    return {
        "clip_macro": {r: float(np.mean(list(a.values()))) for r, a in aps.items()},
        "clip_delta_vs_control": {
            r: bootstrap(clip_delta(aps[r], aps[control]), constants.V5_BOOTSTRAP_LEVEL)
            for r in aps
            if r != control
        },
        "pooled_minmax": {r: pooled_minmax_ap(c, labels) for r, c in curves.items()},
        "positive_frame_share": float(np.concatenate(list(labels.values())).mean()),
    }


MECHANICS_KEYS = ("mil", "tile", "tile_p_abnormal", "tile_p_normal", "v5_temp",
                  "v5_entropy_q0", "v5_logit_std_q0", "motion_share", "clip_share")


def mechanics(run_dir: Path) -> dict[str, Any] | None:
    """Last-step diagnostics of one run's ``metrics.jsonl`` (printed, never gating)."""
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    last = json.loads(lines[-1])
    out: dict[str, Any] = {"step": last.get("global_step")}
    out |= {k: float(last[k]) for k in MECHANICS_KEYS if k in last}
    out["finite"] = all(np.isfinite(v) for k, v in out.items() if k != "step" and v is not None)
    if "v5_entropy_q0" in out:
        out["entropy_over_log_cells"] = out["v5_entropy_q0"] / float(
            np.log(constants.V5_TILE_CELLS)
        )
    return out


def pilot_gate(
    row: str,
    maps: dict[str, Any] | None,
    nexar: dict[str, dict[str, Any]] | None,
    fused: dict[str, Any],
    control: str,
    gate: str = GATE_V51,
) -> dict[str, Any]:
    """§3 (i)-(iii) for one candidate row; ``gate=r7`` reads §4.2's map rule and may skip (ii)."""
    maps_key = "r7_points" if gate == GATE_R7 else "beats_centre_and_change"
    checks: dict[str, bool | str] = {"maps": bool(maps and maps.get(maps_key))}
    if nexar is None:
        if gate != GATE_R7:
            raise ValueError("the v5.1 gate needs the nexar_val guardrail (--nexar-dir)")
        checks["nexar_guard"] = NOT_READ
    else:
        checks["nexar_guard"] = bool(
            nexar[row]["crop_endpoint_mean"]
            >= nexar[control]["crop_endpoint_mean"] - constants.V5_NEXAR_GUARD_MARGIN
        )
    delta = fused["delta_vs_control"].get(row)
    checks["position_controlled_not_below"] = bool(delta is not None and delta["high"] >= 0)
    failed = [k for k, ok in checks.items() if ok is False]
    if gate == GATE_R7:
        outcome = R7_PASS if not failed else f"{R7_NULL} ({', '.join(failed)})"
        if checks["nexar_guard"] == NOT_READ:
            outcome += " -- nexar_val guardrail not read"
        return {"checks": checks, "outcome": outcome}
    return {"checks": checks, "outcome": PASS if not failed else f"{FAIL} ({', '.join(failed)})"}


# --------------------------------------------------------------------------- run


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = [seed_name(s) for s in args.seeds]
    rows = [args.control, *[r for r in args.rows if r != args.control]]
    pb_rows = {r: load_pb_row(args.pb_dir, r, seeds) for r in rows}
    labels = check_same_clips(pb_rows)
    curves = {r: d["scores"] for r, d in pb_rows.items()}
    nexar = (
        {r: load_nexar_row(args.nexar_dir, r, seeds) for r in rows}
        if args.nexar_dir is not None
        else None
    )
    maps_rows: dict[str, Any] = (
        _json(args.maps_readout)["rows"] if args.maps_readout is not None else {}
    )
    fused = fused_reads(curves, labels, args.control)
    fused_per_clip, _ = fused_clips(curves, labels)
    clip_aucs = {r: d["clip_aucs"] for r, d in pb_rows.items()}
    out: dict[str, Any] = {
        "rule": RULE if args.gate == GATE_V51 else "core/docs/v5/PREREG_V5.md §4 (R7 fast pilot)",
        "gate_kind": args.gate,
        "seeds": seeds,
        "preliminary": len(seeds) < len(constants.V5_RUN_SEEDS),
        "control": args.control,
        "clips": len(labels),
        "macro": {r: float(np.mean(list(d["clip_aucs"].values()))) for r, d in pb_rows.items()},
        "per_seed_macro": {r: d["readout"]["per_run_macro"] for r, d in pb_rows.items()},
        "clip_delta": {
            r: tost(clip_delta(pb_rows[r]["clip_aucs"], pb_rows[args.control]["clip_aucs"]))
            for r in rows
            if r != args.control
        },
        "position_cross_clip": {r: cross_clip_position_auc(c, labels) for r, c in curves.items()},
        "position_fused": fused,
        "shift_test": {r: shift_test(c, labels, args.stride) for r, c in curves.items()},
        "ap": ap_summary(curves, labels, args.control),
        "nexar_val": nexar,
        "maps": {r: maps_rows.get(r) for r in rows},
    }
    if args.annotations_dir is not None:
        _, _, ego, bins = dev_meta(
            {v: len(y) for v, y in labels.items()},
            args.metadata,
            args.split_file,
            args.annotations_dir,
        )
        ego_group: dict[str, str | None] = {
            v: ("ego_involved" if e else "ego_uninvolved") for v, e in ego.items()
        }
        out["subsets"] = {
            "auc_by_ego": subset_reads(clip_aucs, ego_group, args.control),
            "pos_b_by_ego": subset_reads(fused_per_clip, ego_group, args.control),
            "auc_by_box_share": subset_reads(clip_aucs, dict(bins), args.control),
            "pos_b_by_box_share": subset_reads(fused_per_clip, dict(bins), args.control),
            "bins": list(BIN_NAMES),
        }
    if args.runs_root is not None:
        out["mechanics"] = {
            f"{r}/{s}": mechanics(args.runs_root / r / s) for r in rows for s in seeds
        }
    if args.pb_rev_dir is not None:
        out["time_reversed_macro"] = {
            r: _json(args.pb_rev_dir / r / "protocol_b_readout.json")["macro"]
            for r in rows
            if (args.pb_rev_dir / r / "protocol_b_readout.json").is_file()
        }
        out["time_reversed_minus_forward"] = {
            r: m["mean"] - out["macro"][r]
            for r, m in out["time_reversed_macro"].items()
            if m is not None
        }
    out["gate"] = {
        r: pilot_gate(r, maps_rows.get(r), nexar, fused, args.control, args.gate)
        for r in rows
        if r != args.control
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = READOUT_JSON if args.gate == GATE_V51 else R7_READOUT_JSON
    write_json_atomic(args.out_dir / name, out)
    write_text_atomic(args.out_dir / name.replace(".json", ".md"), render_markdown(out))
    return out


def _ci(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def render_markdown(r: dict[str, Any]) -> str:
    control = r["control"]
    banner = (
        f"**PRELIMINARY (n = {len(r['seeds'])} seed)** -- a pilot read; it adopts nothing.\n\n"
        if r["preliminary"]
        else ""
    )
    fused = r["position_fused"]
    lines = [
        "# v5.1 tile-scored MQR -- pilot read-out"
        if r.get("gate_kind", GATE_V51) == GATE_V51
        else "# v5 R7 -- MQR on CLIP 14 x 14 patches, fast pilot read-out",
        "",
        f"Rule: {r['rule']}. Control {control}. Seeds {', '.join(r['seeds'])}. "
        f"DoTA-CAP-dev clips {r['clips']}.",
        "",
        banner,
        "## Detection",
        "",
        f"Position alone (cubic in t / (n - 1)): macro {fused['position_alone_macro']:.4f}.",
        "",
        "| row | macro | Δ vs control (95 %) | TOST ±0.01 | pos. (a) cross-clip | "
        "pos. (b) z(y)+2z(p) | gain beyond position | Δ (b) vs control | shift peak | "
        "nexar_val crop |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row, macro in r["macro"].items():
        cd = r["clip_delta"].get(row)
        st = r["shift_test"][row]
        lines.append(
            f"| {row} | {macro:.4f} | {_ci(cd['ci95']) if cd else '—'} | "
            f"{cd['verdict'] if cd else '—'} | {_num(r['position_cross_clip'][row])} | "
            f"{fused['macro'][row]:.4f} | {fused['gain_beyond_position'][row]:+.4f} | "
            f"{_ci(fused['delta_vs_control'].get(row))} | {st['peak_offset']:+d} "
            f"({'ok' if st['pass'] else 'off'}) | "
            + (
                f"{r['nexar_val'][row]['crop_endpoint_mean']:.4f} |"
                if r["nexar_val"]
                else "not read |"
            )
        )
    if "time_reversed_macro" in r:
        lines += ["", "Time-reversed control (printed; reversed - forward should be ≤ 0): "
                  + ", ".join(f"{k} {_ci(v)} ({r['time_reversed_minus_forward'].get(k, 0):+.4f})"
                              for k, v in r["time_reversed_macro"].items())]
    ap = r["ap"]
    lines += ["", f"AP (positive frame share {ap['positive_frame_share']:.3f}): "
              + ", ".join(f"{k} clip-macro {v:.4f} / pooled {ap['pooled_minmax'][k]:.4f}"
                          for k, v in ap["clip_macro"].items())
              + "; Δ clip-macro vs control "
              + ", ".join(f"{k} {_ci(v)}" for k, v in ap["clip_delta_vs_control"].items())]
    lines += render_subsets(r)
    lines += render_maps(r)
    lines += render_mechanics(r)
    r7 = r.get("gate_kind") == GATE_R7
    maps_head = "maps: R7 rule (§4.2 P-e)" if r7 else "maps beat centre + change"
    lines += ["", "## Pilot gate (§4.2)" if r7 else "## Pilot gate (§3)", "",
              f"| row | {maps_head} | nexar guard | pos. (b) not below | outcome |",
              "|---|---|---|---|---|"]

    def mark(v: bool | str) -> str:
        return v if isinstance(v, str) else ("yes" if v else "no")

    for row, g in r["gate"].items():
        c = g["checks"]
        lines.append(
            f"| {row} | {mark(c['maps'])} | {mark(c['nexar_guard'])} | "
            f"{mark(c['position_controlled_not_below'])} | **{g['outcome']}** |"
        )
    return "\n".join(lines) + "\n"


def render_subsets(r: dict[str, Any]) -> list[str]:
    if "subsets" not in r:
        return []
    control = r["control"]
    lines = ["", "## Subsets (proposal §4.4; printed, 95 % cluster CI of Δ vs control)", "",
             "| read | group | clips | " + " | ".join(r["macro"]) + " | "
             + " | ".join(f"Δ {k}" for k in r["macro"] if k != control) + " |",
             "|---|---|---|" + "---|" * (2 * len(r["macro"]) - 1)]
    for read, groups in r["subsets"].items():
        if read == "bins":
            continue
        for name, g in groups.items():
            lines.append(
                f"| {read} | {name} | {g['clips']} | "
                + " | ".join(_num(g["macro"].get(k)) for k in r["macro"])
                + " | "
                + " | ".join(_ci(g["delta_vs_control"].get(k)) for k in r["macro"] if k != control)
                + " |"
            )
    return lines


def render_maps(r: dict[str, Any]) -> list[str]:
    rows = {k: m for k, m in r["maps"].items() if m}
    if not rows:
        return []
    lines = ["", "## Maps on DoTA-CAP-dev boxes (proposal §4.4 H2)", "",
             "| row | hit - chance | hit - centre | hit - change | mass - chance | mass - change | "
             "STAUC | uniform | centre | change |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in rows.items():
        b = m.get("stauc_baselines", {})
        lines.append(
            f"| {k} | {_ci(m.get('hit_minus_chance'))} | {_ci(m.get('hit_minus_centre'))} | "
            f"{_ci(m.get('hit_minus_change'))} | {_ci(m.get('mass_minus_chance'))} | "
            f"{_ci(m.get('mass_minus_change'))} | {m['stauc']['stauc']:.4f} | "
            + " | ".join(_num(b[x]["stauc"]) if x in b else "—"
                         for x in ("uniform", "centre", "change"))
            + " |"
        )
    return lines


def render_mechanics(r: dict[str, Any]) -> list[str]:
    if "mechanics" not in r:
        return []
    lines = ["", "## Mechanics (last step; printed)", "",
             "| run | finite | mil | tile | p abnormal | p normal | T | entropy / log 196 | "
             "motion share | CLIP share (R7) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in r["mechanics"].items():
        if m is None:
            lines.append(f"| {k} | no metrics | | | | | | | | |")
            continue
        lines.append(
            f"| {k} | {'yes' if m['finite'] else '**NO**'} | {_num(m.get('mil'))} | "
            f"{_num(m.get('tile'))} | {_num(m.get('tile_p_abnormal'))} | "
            f"{_num(m.get('tile_p_normal'))} | {_num(m.get('v5_temp'))} | "
            f"{_num(m.get('entropy_over_log_cells'))} | {_num(m.get('motion_share'))} | "
            f"{_num(m.get('clip_share'))} |"
        )
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--pb-dir", type=Path, required=True, help="protocol_b_eval <row>/ dirs")
    parser.add_argument("--pb-rev-dir", type=Path, default=None, help="--reverse-time reads")
    parser.add_argument("--nexar-dir", type=Path, default=None,
                        help="nexar_eval <row>/ dirs (required by --gate v51)")
    parser.add_argument("--gate", choices=(GATE_V51, GATE_R7), default=GATE_V51)
    parser.add_argument("--maps-readout", type=Path, default=None, help="v5_maps read json")
    parser.add_argument("--rows", nargs="+", default=list(constants.V5_TILE_PILOT_ROWS))
    parser.add_argument("--control", default="R1c")
    parser.add_argument("--seeds", nargs="+", type=int, default=[constants.V5_TILE_SEED])
    parser.add_argument("--stride", type=int, default=constants.V5_TILE_DOTA_STRIDE)
    parser.add_argument("--runs-root", type=Path, default=None, help="<row>/s<seed>/metrics.jsonl")
    parser.add_argument("--annotations-dir", type=Path, default=None,
                        help="DoTA annotations/ (ego flag, boxes) for the subset reads")
    parser.add_argument("--metadata", type=Path, default=None, help="DoTA metadata_val.json")
    parser.add_argument("--split-file", type=Path, default=None, help="DoTA val ids")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = run(build_arg_parser().parse_args(argv))
    for row, g in out["gate"].items():
        LOGGER.info("%s: %s", row, g["outcome"])


if __name__ == "__main__":
    main()
