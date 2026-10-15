"""v2.2 SC-2: A3-B vs A3-ign and the adoption rule (v2 addendum §25, fixed before any A3-B number).

Pure aggregation -- no model is loaded. A3-B is A3-ign trained with VideoMAE V2-B in place of V2-S;
A3-ign is E4-0's five runs (never retrained). Inputs, as laid out by ``colab/v2/sc2_a3b.ipynb``::

    <e3-dir>/REPORTS/position_prior/dota_cap_dev/position_prior.npz                  (E3, p_T2)
    <e4-dir>/REPORTS/pb/A3ign/dota_cap_dev/{protocol_b_readout.json, clip_scores.npz} (E4-0)
    <e4-dir>/A3ign/s<seed>/diag/diag.json                                             (E4-0)
    <sc2-dir>/REPORTS/pb/A3B/dota_cap_dev/{protocol_b_readout.json, clip_scores.npz}
    <sc2-dir>/A3B/s<seed>/diag/diag.json              v2_diagnostics with --a0-diag (O1')

The rule, mechanically (§25 Y8): adopt A3-B iff

1. Y5 (a) the paired t95 of the per-seed DoTA-CAP-dev macro Δ (A3-B - A3-ign) lies above 0,
   (b) the cluster-bootstrap interval of the per-clip Δ (seed-averaged scores) lies above 0,
   (c) the paired mean is ≥ :data:`constants.V22_SC2_MIN_GAIN`;
2. Y6 the ``f_2`` Δ point estimate is ≥ 0;
3. Y7 seed-mean T2-val macro ≥ A3-ign's - :data:`constants.V22_SC2_T2_MACRO_MARGIN`, and O1' holds
   on every A3-B seed;

otherwise A3-ign stays. Everything else is printed (Y9).

CLI::

    python -m core.tools.sc2_readout --e3-dir outputs/v2_e3 --e4-dir outputs/v2_e4_0 \\
        --sc2-dir outputs/v22_sc2 [--alignment .../dota_cap_alignment.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools.e3_readout import (
    PRIOR_DIR,
    REPORTS,
    bootstrap,
    clip_delta,
    paired_interval,
    positive,
    seed_name,
)
from core.tools.e4_0_readout import CAP, final_epoch_losses, load_diags, load_pb
from core.tools.final_readout import fused_clip_aucs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.position_prior import PRIOR_NPZ
from core.tools.protocol_b_eval import clip_aucs, read_clip_scores
from core.tools.v22_gates import rate_groups

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "sc2_readout.json"
READOUT_MD = "sc2_readout.md"
CONTROL = constants.V22_SC2_CONTROL
ARM = constants.V22_SC2_ARM
ADOPT_SENTENCE = "A larger frozen video encoder improves DoTA-CAP-dev detection."
KEEP_SENTENCE = ("A larger frozen video encoder does not improve DoTA-CAP-dev detection at this "
                 "size.")


def seed_mean(diags: dict[str, Any], get: Any) -> float | None:
    values = [get(d) for d in diags.values()]
    return None if any(v is None for v in values) else float(np.mean(values))


def guardrails(control: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    """Y7: T2-val macro (seed means) and O1' on every A3-B seed; the peak is printed only."""
    missing = [s for s, d in arm.items() if "o1_prime" not in d["guardrails"]]
    if missing:
        raise ValueError(f"{ARM}: no O1' in diags of {missing} -- run v2_diagnostics --a0-diag")
    macro = {name: seed_mean(ds, lambda d: float(d["guardrails"]["macro"]))
             for name, ds in ((CONTROL, control), (ARM, arm))}
    peak = {name: seed_mean(ds, lambda d: d["t2_val_reads"]["normal_window_peak"])
            for name, ds in ((CONTROL, control), (ARM, arm))}
    o1p = {s: bool(d["guardrails"]["o1_prime"]["pass"]) for s, d in arm.items()}
    a, c = macro[ARM], macro[CONTROL]
    checks = {
        "t2_val_macro": a is not None and c is not None
        and a >= c - constants.V22_SC2_T2_MACRO_MARGIN,
        "o1_prime_every_seed": all(o1p.values()),
    }
    return {"t2_val_macro": macro, "normal_window_peak": peak, "o1_prime": o1p,
            "checks": checks, "hold": all(checks.values())}


def decide(
    interval: dict[str, Any], clip: dict[str, float] | None, f2: dict[str, float] | None,
    guards: dict[str, Any],
) -> dict[str, Any]:
    """§25 Y8; no tie clause."""
    checks = {
        "y5a_paired_t95_above_0": positive(interval),
        "y5b_cluster_ci_above_0": clip is not None and clip["low"] > 0.0,
        "y5c_point_ge_min_gain": interval["mean"] >= constants.V22_SC2_MIN_GAIN,
        "y6_f2_point_ge_0": f2 is not None and f2["mean"] >= 0.0,
        "y7_guardrails_hold": bool(guards["hold"]),
    }
    adopt = all(checks.values())
    return {"checks": checks, "base": ARM if adopt else CONTROL,
            "sentence": ADOPT_SENTENCE if adopt else KEEP_SENTENCE}


def by_group(
    delta: dict[str, float], groups: dict[str, str], seed: int
) -> dict[str, dict[str, float] | None]:
    """Y9: the per-clip Δ split by DoTA-CAP native-fps group (printed only)."""
    return {g: bootstrap({v: d for v, d in delta.items() if groups[v] == g}, seed)
            for g in sorted(set(groups.values()))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = [seed_name(s) for s in args.seeds]
    pbs = {CONTROL: load_pb(args.e4_dir, CONTROL, seeds), ARM: load_pb(args.sc2_dir, ARM, seeds)}
    labels = pbs[CONTROL]["labels"]
    if set(pbs[ARM]["labels"]) != set(labels):  # Y11(e)
        raise ValueError(f"{CONTROL} and {ARM} protocol B reads cover different clips")
    for v, y in labels.items():
        if not np.array_equal(y, pbs[ARM]["labels"][v]):
            raise ValueError(f"{v}: {CONTROL} and {ARM} protocol B labels differ")
    diags = {CONTROL: load_diags(args.e4_dir, CONTROL, seeds, pbs[CONTROL]),
             ARM: load_diags(args.sc2_dir, ARM, seeds, pbs[ARM])}  # Y11(f)
    prior, _ = read_clip_scores(args.e3_dir / REPORTS / PRIOR_DIR / CAP / PRIOR_NPZ)

    per_seed = {name: pb["readout"]["per_run_macro"] for name, pb in pbs.items()}
    interval = paired_interval(per_seed[ARM], per_seed[CONTROL], constants.V2_E3_T95)
    scores = {name: pb["scores"] for name, pb in pbs.items()}
    raw = {name: clip_aucs(s, labels) for name, s in scores.items()}
    raw_delta = clip_delta(raw[ARM], raw[CONTROL])
    clip = bootstrap(raw_delta, args.seed)
    weight = constants.V2_FINAL_FUSION_PRIMARY
    f2_aucs = {name: fused_clip_aucs(s, prior, labels, weight) for name, s in scores.items()}
    f2 = bootstrap(clip_delta(f2_aucs[ARM], f2_aucs[CONTROL]), args.seed)
    guards = guardrails(diags[CONTROL], diags[ARM])
    decision = decide(interval, clip, f2, guards)

    groups = None
    if args.alignment is not None:
        alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
        groups = by_group(raw_delta, rate_groups(alignment, sorted(raw_delta)), args.seed)
    result: dict[str, Any] = {
        "rule": "core/docs/v2/PREREG_ADDENDUM.md §25 (fixed before any A3-B number)",
        "set": "DoTA-CAP (n/1397) dev",
        "seeds": seeds,
        "macro": {name: float(np.mean(list(a.values()))) for name, a in raw.items()},
        "p_t2_macro": float(np.mean(list(clip_aucs(prior, labels).values()))),
        "per_seed_macro": per_seed,
        "decision_interval": interval,
        "clip_delta": clip,
        "f2": {"weight": weight,
               "macro": {name: float(np.mean(list(a.values()))) for name, a in f2_aucs.items()},
               "delta": f2},
        "guardrails": guards,
        "rate_groups": groups,
        "t2_val_span_profile": {
            name: {s: d["t2_val_reads"]["span_profile"] for s, d in ds.items()}
            for name, ds in diags.items()
        },
        "diagnostics": {
            name: {
                "source_shortcut_v_t": seed_mean(ds, lambda d: d["source_shortcut_auc"]["v_t"]),
                "position_r2_v_t": seed_mean(ds, lambda d: d["position_r2"]["v_t"]),
                "final_epoch_losses": {s: final_epoch_losses(d["checkpoint"])
                                       for s, d in ds.items()},
            }
            for name, ds in diags.items()
        },
        "decision": decision,
    }
    out = args.sc2_dir / REPORTS
    write_json_atomic(out / READOUT_JSON, result)
    write_text_atomic(out / READOUT_MD, render_markdown(result))
    LOGGER.info("SC-2: base = %s -> %s", decision["base"], out)
    return result


def _ci(c: dict[str, float] | None) -> str:
    return "n/a" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def render_markdown(r: dict[str, Any]) -> str:
    i, g = r["decision_interval"], r["guardrails"]
    lines = [
        f"# v2.2 SC-2 — {ARM} vs {CONTROL} on {r['set']}", "",
        f"Rule: {r['rule']}. **Base = {r['decision']['base']}.**", "",
        f"> {r['decision']['sentence']}", "",
        "## Decision (Y8)", "",
        "| check | result |", "|---|---|",
        *[f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in r["decision"]["checks"].items()], "",
        f"- macro {CONTROL} {_num(r['macro'][CONTROL])} · {ARM} {_num(r['macro'][ARM])} · "
        f"`p_T2` {_num(r['p_t2_macro'])}",
        f"- paired t95 ({ARM} - {CONTROL}): {i['mean']:+.4f} [{i['low']:+.4f}, {i['high']:+.4f}]"
        f" (n = {i['n']}); per seed "
        + ", ".join(f"{s} {d:+.4f}" for s, d in i["per_seed"].items()),
        f"- cluster bootstrap, per-clip Δ: {_ci(r['clip_delta'])}",
        f"- `f_{r['f2']['weight']:g}` Δ: {_ci(r['f2']['delta'])} (holds for DoTA-CAP's placement "
        "only, pending (bn))", "",
        "## Guardrails (Y7, T2-val seed means)", "",
        f"| read | {CONTROL} | {ARM} | pass |", "|---|---|---|---|",
        f"| macro | {_num(g['t2_val_macro'][CONTROL])} | {_num(g['t2_val_macro'][ARM])} | "
        f"{g['checks']['t2_val_macro']} |",
        f"| normal-window peak (printed) | {_num(g['normal_window_peak'][CONTROL])} | "
        f"{_num(g['normal_window_peak'][ARM])} | — |",
        f"| O1' every seed | — | {g['o1_prime']} | {g['checks']['o1_prime_every_seed']} |", "",
        "## Printed only (Y9)", "",
    ]
    if r["rate_groups"] is not None:
        lines.append("- per-clip Δ by native-fps group: "
                     + "; ".join(f"{k} {_ci(c)}" for k, c in r["rate_groups"].items()))
    d = r["diagnostics"]
    lines += [
        "- `f_2` macros: " + ", ".join(f"{k} {_num(v)}" for k, v in r["f2"]["macro"].items()),
        "- source shortcut AUC (`v^t`): " + ", ".join(
            f"{k} {_num(v['source_shortcut_v_t'])}" for k, v in d.items()),
        "- position R² (`v^t`): " + ", ".join(
            f"{k} {_num(v['position_r2_v_t'])}" for k, v in d.items()),
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--e3-dir", type=Path, required=True, help="E3's dir (p_T2)")
    parser.add_argument("--e4-dir", type=Path, required=True, help="E4-0's dir (A3-ign)")
    parser.add_argument("--sc2-dir", type=Path, required=True, help="SC-2's dir (A3-B)")
    parser.add_argument("--alignment", type=Path, default=None,
                        help="dota_cap_alignment.json: prints Δ per native-fps group")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(constants.V2_E3_SEEDS))
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap seed")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
