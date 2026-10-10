"""v2.1 E4-0: A3-ign vs A3 and the base rule (proposal v2.1 §11.4, fixed before any number).

Pure aggregation -- no model is loaded. A3-ign is A3 trained with ``dvs_anchor_mode = ignore``;
A3 is E3's five runs. Inputs, as laid out by ``colab/v2/e4_0_a3ign.ipynb``::

    <e3-dir>/REPORTS/pb/A3/dota_cap_dev/{protocol_b_readout.json, clip_scores.npz}   (E3, M3)
    <e3-dir>/REPORTS/position_prior/dota_cap_dev/position_prior.npz                  (E3, G6)
    <e4-dir>/REPORTS/pb/A3ign/dota_cap_dev/{protocol_b_readout.json, clip_scores.npz}
    <e4-dir>/<A3|A3ign>/s<seed>/diag/diag.json        v2_diagnostics, both arms re-diagnosed
                                                       with the same code (t2_val_reads)

The rule, mechanically (§11.4): ``B`` = A3-ign iff

1. the paired t95 of the per-seed DoTA-CAP-dev macro Δ (A3-ign - A3) lies above 0;
2. the ``f_2`` Δ point estimate is ≥ 0 (``f_2 = z(score) + 2·z(p_T2)``, seed-averaged scores);
3. the guardrails hold: seed-mean T2-val macro ≥ A3's - 0.01, seed-mean normal-window peak
   ≤ A3's + 0.02, O1' on every A3-ign seed;

otherwise ``B`` = A3. The mechanism reads (§7, §11.4) license the Gap S sentence only; they never
move ``B``. Everything else is printed.

CLI::

    python -m core.tools.e4_0_readout --e3-dir outputs/v2_e3 --e4-dir outputs/v2_e4_0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import span_reads, span_summary
from core.tools.e3_readout import (
    DIAG_DIR,
    PB_DIR,
    PRIOR_DIR,
    REPORTS,
    bootstrap,
    clip_delta,
    paired_interval,
    positive,
    seed_name,
)
from core.tools.final_readout import fused_clip_aucs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.position_prior import PRIOR_NPZ
from core.tools.posthoc_hold import tail_clips
from core.tools.protocol_b_eval import CLIP_SCORES_NPZ, clip_aucs, read_clip_scores
from core.tools.protocol_b_eval import READOUT_JSON as PB_READOUT_JSON
from core.tools.v2_diagnostics import DIAG_JSON
from core.train import METRICS_FILENAME

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "e4_0_readout.json"
READOUT_MD = "e4_0_readout.md"
CAP = constants.V2_SPLIT_DOTA_CAP_DEV
CONTROL = constants.V2_E4_0_CONTROL
ARM = constants.V2_E4_0_ARM
LOSS_KEYS = ("mil", "dvs_sup", "dvs_sup_mil", "mul_mil", "total")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def _json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"E4-0 input missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_pb(root: Path, arm: str, seeds: list[str]) -> dict[str, Any]:
    d = root / REPORTS / PB_DIR / arm / CAP
    readout = _json(d / PB_READOUT_JSON)
    if sorted(readout["runs"]) != seeds:
        raise ValueError(f"{arm}: protocol B runs {readout['runs']} are not the seeds {seeds}")
    scores, labels = read_clip_scores(d / CLIP_SCORES_NPZ)
    return {"readout": readout, "scores": scores, "labels": labels}


def load_diags(e4_dir: Path, arm: str, seeds: list[str], pb: dict[str, Any]) -> dict[str, Any]:
    """Per-seed diags; each must read the same checkpoint step as protocol B and carry the
    v2.1 ``t2_val_reads`` (an E3-era diag does not -- re-diagnose)."""
    diags = {s: _json(e4_dir / arm / s / DIAG_DIR / DIAG_JSON) for s in seeds}
    for s, d in diags.items():
        if pb["readout"]["global_steps"][s] != d["global_step"]:
            raise ValueError(
                f"{arm} {s}: protocol B read step {pb['readout']['global_steps'][s]}, "
                f"diagnostics step {d['global_step']} -- not the same checkpoint"
            )
        if "t2_val_reads" not in d:
            raise ValueError(f"{arm} {s}: diag has no t2_val_reads -- re-run v2_diagnostics")
    return diags


def final_epoch_losses(checkpoint: str) -> dict[str, float] | None:
    """Mean of each loss over the last epoch of the run's ``metrics.jsonl``; printed only."""
    path = Path(checkpoint).parent / METRICS_FILENAME
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    last = [r for r in rows if r["epoch"] == rows[-1]["epoch"]]
    return {k: float(np.mean([r[k] for r in last])) for k in LOSS_KEYS if k in last[0]}


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------
def _seed_mean(diags: dict[str, Any], get: Any) -> float | None:
    values = [get(d) for d in diags.values()]
    return None if any(v is None for v in values) else float(np.mean(values))


def _ge(a: float | None, b: float | None, offset: float) -> bool:
    """``a >= b + offset``; an undefined side fails."""
    return a is not None and b is not None and a >= b + offset


def guardrails(control: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    """§11.1 against A3: T2-val macro, normal-window peak (seed means), O1' every seed."""
    def macro(d: dict[str, Any]) -> float:
        return float(d["guardrails"]["macro"])

    def peak(d: dict[str, Any]) -> float | None:
        value = d["t2_val_reads"]["normal_window_peak"]
        return None if value is None else float(value)

    missing = [s for s, d in arm.items() if "o1_prime" not in d["guardrails"]]
    if missing:
        raise ValueError(f"{ARM}: no O1' in diags of {missing} -- run with --a0-diag")
    m = {CONTROL: _seed_mean(control, macro), ARM: _seed_mean(arm, macro)}
    p = {CONTROL: _seed_mean(control, peak), ARM: _seed_mean(arm, peak)}
    o1p = {s: bool(d["guardrails"]["o1_prime"]["pass"]) for s, d in arm.items()}
    reads: dict[str, Any] = {"t2_val_macro": m, "normal_window_peak": p, "o1_prime": o1p}
    if p[CONTROL] is None or p[ARM] is None:
        raise ValueError("a run has no all-normal T2-val window: the peak guardrail is undefined")
    checks = {
        "t2_val_macro": _ge(m[ARM], m[CONTROL], -constants.V2_E4_0_T2_MACRO_MARGIN),
        "normal_window_peak": _ge(p[CONTROL], p[ARM], -constants.V2_E4_0_PEAK_MARGIN),
        "o1_prime_every_seed": all(o1p.values()),
    }
    return {**reads, "checks": checks, "hold": all(checks.values())}


def mechanism(
    control: dict[str, np.ndarray],
    arm: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    """§7 / §11.4 on seed-averaged DoTA-CAP-dev scores: span reads and the tail-clip macro."""
    reads = {CONTROL: span_reads(control, labels), ARM: span_reads(arm, labels)}
    deltas = {
        key: bootstrap(
            {v: reads[ARM][v][key] - reads[CONTROL][v][key]
             for v in reads[ARM] if key in reads[ARM][v]},
            seed,
        )
        for key in ("argmax_in_span", "pre_mean", "in_mean")
    }
    tails = tail_clips(labels, constants.V2_E4_0_TAIL_FRAMES)
    if len(tails) != constants.V2_E4_0_TAIL_CLIPS:
        raise ValueError(
            f"{len(tails)} tail clips at w = {constants.V2_E4_0_TAIL_FRAMES}, H2(a) had "
            f"{constants.V2_E4_0_TAIL_CLIPS}: not the same clip set"
        )
    tail_delta = clip_delta(
        {v: a for v, a in clip_aucs(arm, labels).items() if v in tails},
        {v: a for v, a in clip_aucs(control, labels).items() if v in tails},
    )
    deltas["tail_macro"] = bootstrap(tail_delta, seed)

    def low_above(c: dict[str, float] | None) -> bool:
        return c is not None and c["low"] > 0.0

    def high_below(c: dict[str, float] | None) -> bool:
        return c is not None and c["high"] < 0.0

    conditions = {
        "pre_mean_falls": high_below(deltas["pre_mean"]),
        "in_mean_does_not_fall": not high_below(deltas["in_mean"]),
        "argmax_in_span_rises": low_above(deltas["argmax_in_span"]),
        "tail_macro_does_not_fall": not high_below(deltas["tail_macro"]),
    }
    return {
        "summary": {k: span_summary(r) for k, r in reads.items()},
        "delta": deltas,
        "tail_clips": len(tails),
        "conditions": conditions,
        "gap_s_claim": all(conditions.values()),
    }


def decide(
    interval: dict[str, Any], f2: dict[str, float] | None, guards: dict[str, Any]
) -> dict[str, Any]:
    """§11.4 base rule; no tie clause."""
    checks = {
        "macro_t95_above_0": positive(interval),
        "f2_point_ge_0": f2 is not None and f2["mean"] >= 0.0,
        "guardrails_hold": bool(guards["hold"]),
    }
    return {"checks": checks, "base": ARM if all(checks.values()) else CONTROL}


def claims(decision: dict[str, Any], f2: dict[str, float] | None, mech: dict[str, Any],
           interval: dict[str, Any]) -> list[str]:
    """§11.4 'what the thesis may claim', in the order the proposal lists them."""
    out: list[str] = []
    if decision["checks"]["macro_t95_above_0"]:
        out.append("Dropping the whole-anchor DVS label improves DoTA-CAP (n/1397) dev detection.")
        if f2 is not None and f2["low"] > 0.0:
            out.append("... beyond the T2 position prior (f_2 interval excludes 0).")
        else:
            out.append("The gain is not separable from the T2 position prior (f_2 interval ∋ 0).")
    elif interval["high"] < 0.0:
        out.append("Dropping the whole-anchor DVS label lowers DoTA-CAP (n/1397) dev detection.")
    else:
        out.append("A3-ign vs A3 on DoTA-CAP dev macro: not detectable at this size "
                   "(not evidence that C29 is harmless on T2).")
    out.append("The whole-anchor label causes (part of) Gap S." if mech["gap_s_claim"] else
               "No Gap S claim: not every §7 mechanism condition held.")
    if decision["base"] == CONTROL:
        out.append("E4 runs on span; the C29 interaction is a limitation of any SG-NM result.")
    else:
        out.append("A3-ign is the new best adoptable free arm F; later costly arms must beat it.")
    return out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = [seed_name(s) for s in args.seeds]
    pbs = {CONTROL: load_pb(args.e3_dir, CONTROL, seeds), ARM: load_pb(args.e4_dir, ARM, seeds)}
    labels = pbs[CONTROL]["labels"]
    if set(pbs[ARM]["labels"]) != set(labels):
        raise ValueError("A3 and A3-ign protocol B reads cover different clips")
    diags = {arm: load_diags(args.e4_dir, arm, seeds, pb) for arm, pb in pbs.items()}
    prior, _ = read_clip_scores(args.e3_dir / REPORTS / PRIOR_DIR / CAP / PRIOR_NPZ)

    per_seed = {arm: pb["readout"]["per_run_macro"] for arm, pb in pbs.items()}
    interval = paired_interval(per_seed[ARM], per_seed[CONTROL], constants.V2_E3_T95)
    scores = {arm: pb["scores"] for arm, pb in pbs.items()}
    weight = constants.V2_FINAL_FUSION_PRIMARY
    f2_aucs = {arm: fused_clip_aucs(s, prior, labels, weight) for arm, s in scores.items()}
    f2 = bootstrap(clip_delta(f2_aucs[ARM], f2_aucs[CONTROL]), args.seed)
    raw_aucs = {arm: clip_aucs(s, labels) for arm, s in scores.items()}
    prior_aucs = clip_aucs(prior, labels)
    guards = guardrails(diags[CONTROL], diags[ARM])
    mech = mechanism(scores[CONTROL], scores[ARM], labels, args.seed)
    decision = decide(interval, f2, guards)

    result: dict[str, Any] = {
        "rule": "core/docs/v2.1/KAT_VAD_PROPOSAL_v2.1.md §11.4 (fixed before any A3-ign number)",
        "set": "DoTA-CAP (n/1397) dev",
        "seeds": seeds,
        "macro": {arm: float(np.mean(list(a.values()))) for arm, a in raw_aucs.items()},
        "per_seed_macro": per_seed,
        "decision_interval": interval,
        "clip_delta_raw": bootstrap(clip_delta(raw_aucs[ARM], raw_aucs[CONTROL]), args.seed),
        "f2": {
            "weight": weight,
            "macro": {arm: float(np.mean(list(a.values()))) for arm, a in f2_aucs.items()},
            "delta": f2,
        },
        "p_t2_macro": float(np.mean(list(prior_aucs.values()))),
        "guardrails": guards,
        "mechanism": mech,
        "t2_val_span_profile": {
            arm: {s: d["t2_val_reads"]["span_profile"] for s, d in ds.items()}
            for arm, ds in diags.items()
        },
        "diagnostics": {
            arm: {
                "source_shortcut_v_t": _seed_mean(ds, lambda d: d["source_shortcut_auc"]["v_t"]),
                "position_r2_v_t": _seed_mean(ds, lambda d: d["position_r2"]["v_t"]),
                "final_epoch_losses": {s: final_epoch_losses(d["checkpoint"])
                                       for s, d in ds.items()},
            }
            for arm, ds in diags.items()
        },
        "decision": decision,
    }
    result["claims"] = claims(decision, f2, mech, interval)
    out = args.e4_dir / REPORTS
    write_json_atomic(out / READOUT_JSON, result)
    write_text_atomic(out / READOUT_MD, render_markdown(result))
    LOGGER.info("E4-0: base B = %s -> %s", decision["base"], out)
    return result


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _ci(c: dict[str, float] | None) -> str:
    return "n/a" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def render_markdown(r: dict[str, Any]) -> str:
    i, g, m = r["decision_interval"], r["guardrails"], r["mechanism"]
    lines = [
        f"# v2.1 E4-0 — {ARM} vs {CONTROL} on {r['set']}",
        "",
        f"Rule: {r['rule']}. **Base B = {r['decision']['base']}.**",
        "",
        "## Decision",
        "",
        "| check | result |",
        "|---|---|",
        *[f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in r["decision"]["checks"].items()],
        "",
        f"- macro {CONTROL} {_num(r['macro'][CONTROL])} · {ARM} {_num(r['macro'][ARM])} · "
        f"`p_T2` {_num(r['p_t2_macro'])}",
        f"- paired t95 ({ARM} - {CONTROL}): {i['mean']:+.4f} [{i['low']:+.4f}, {i['high']:+.4f}]"
        f" (n = {i['n']}); per seed "
        + ", ".join(f"{s} {d:+.4f}" for s, d in i["per_seed"].items()),
        f"- clip-level raw Δ (printed): {_ci(r['clip_delta_raw'])}",
        f"- `f_{r['f2']['weight']:g}` Δ: {_ci(r['f2']['delta'])}",
        "",
        "## Guardrails (T2-val, seed means)",
        "",
        f"| read | {CONTROL} | {ARM} | pass |",
        "|---|---|---|---|",
        f"| macro | {_num(g['t2_val_macro'][CONTROL])} | {_num(g['t2_val_macro'][ARM])} | "
        f"{g['checks']['t2_val_macro']} |",
        f"| normal-window peak | {_num(g['normal_window_peak'][CONTROL])} | "
        f"{_num(g['normal_window_peak'][ARM])} | {g['checks']['normal_window_peak']} |",
        f"| O1' every seed | — | {g['o1_prime']} | {g['checks']['o1_prime_every_seed']} |",
        "",
        "## Mechanism (DoTA-CAP-dev, seed-averaged, per-clip min-max; never moves B)",
        "",
        f"| read | {CONTROL} | {ARM} | Δ [95 % cluster CI] |",
        "|---|---|---|---|",
        *[
            f"| {k} | {_num(m['summary'][CONTROL][k])} | {_num(m['summary'][ARM][k])} | "
            f"{_ci(m['delta'][k])} |"
            for k in ("argmax_in_span", "pre_mean", "in_mean")
        ],
        f"| tail-clip macro Δ ({m['tail_clips']} clips) | | | {_ci(m['delta']['tail_macro'])} |",
        "",
        *[f"- {k}: {v}" for k, v in m["conditions"].items()],
        "",
        "## Claims",
        "",
        *[f"- {c}" for c in r["claims"]],
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--e3-dir", type=Path, required=True, help="E3's dir (A3's reads, p_T2)")
    parser.add_argument("--e4-dir", type=Path, required=True, help="E4-0's dir")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(constants.V2_E3_SEEDS))
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap seed")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
