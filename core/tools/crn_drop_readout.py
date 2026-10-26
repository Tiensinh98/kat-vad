"""v2 Amendment 20: can `B` drop CRN? A2-ign vs A3-ign, non-inferiority (addendum §31, fixed first).

Pure aggregation -- no model is loaded. A2-ign is A3-ign trained with ``v2.crn = none``; A3-ign is
E4-0's five runs (never retrained). Inputs, as laid out by ``colab/v2/crn_drop_a2ign.ipynb``::

    <e3-dir>/REPORTS/position_prior/dota_cap_dev/position_prior.npz                  (E3, p_T2)
    <e3-dir>/REPORTS/pb/A2/dota_cap_dev/{protocol_b_readout.json, clip_scores.npz}   (E3, printed)
    <e4-dir>/REPORTS/pb/A3ign/dota_cap_dev/{protocol_b_readout.json, clip_scores.npz} (E4-0)
    <e4-dir>/A3ign/s<seed>/diag/diag.json                                             (E4-0)
    <out-dir>/REPORTS/pb/A2ign/dota_cap_dev/{protocol_b_readout.json, clip_scores.npz}
    <out-dir>/A2ign/s<seed>/diag/diag.json            v2_diagnostics with --a0-diag (O1')

The rule, mechanically (§31 Z7): drop CRN (``B`` = A2-ign) iff, with margin
:data:`constants.V2_CRNDROP_MARGIN` = M,

1. Z4 (a) the paired t95 of the per-seed DoTA-CAP-dev macro Δ (A2-ign - A3-ign) has low > -M,
   (b) the cluster-bootstrap interval of the per-clip Δ (seed-averaged scores) has low > -M;
2. Z5 the cluster-bootstrap interval of the ``f_2`` Δ has low > -M;
3. Z6 seed-mean T2-val macro ≥ A3-ign's - M, and O1' holds on every A2-ign seed;

otherwise A3-ign stays. A failure never reads as "CRN helps". Everything else is printed (Z8).

CLI::

    python -m core.tools.crn_drop_readout --e3-dir outputs/v2_e3 --e4-dir outputs/v2_e4_0 \\
        --out-dir outputs/v2_crndrop [--alignment .../dota_cap_alignment.json]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools.e3_readout import PB_DIR, REPORTS, seed_name
from core.tools.e4_0_readout import CAP, load_pb
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import clip_aucs
from core.tools.sc2_readout import paired_reads

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "crn_drop_readout.json"
READOUT_MD = "crn_drop_readout.md"
CONTROL = constants.V2_CRNDROP_CONTROL
ARM = constants.V2_CRNDROP_ARM
E3_TWIN = "A2"  # E3's A2 (span): A2-ign without `ignore`, printed only (Z8)
MARGIN = constants.V2_CRNDROP_MARGIN
DROP_SENTENCE = ("Removing CRN does not cost DoTA-CAP-dev detection, in total or beyond position "
                 f"(non-inferiority margin {MARGIN:g}, n = 5).")
KEEP_SENTENCE = "Removing CRN is not shown to be harmless on DoTA-CAP-dev at n = 5."


def above(ci: dict[str, float] | None, bound: float) -> bool:
    """Non-inferiority reading: the interval's lower end strictly above ``bound``."""
    return ci is not None and ci["low"] > bound


def decide(
    interval: dict[str, Any], clip: dict[str, float] | None, f2: dict[str, float] | None,
    guards: dict[str, Any],
) -> dict[str, Any]:
    """§31 Z7; no tie clause."""
    checks = {
        "z4a_paired_t95_low_gt_minus_margin": above(interval, -MARGIN),
        "z4b_cluster_ci_low_gt_minus_margin": above(clip, -MARGIN),
        "z5_f2_ci_low_gt_minus_margin": above(f2, -MARGIN),
        "z6_guardrails_hold": bool(guards["hold"]),
    }
    drop = all(checks.values())
    return {"checks": checks, "failed": [k for k, v in checks.items() if not v],
            "base": ARM if drop else CONTROL, "drop_crn": drop,
            "sentence": DROP_SENTENCE if drop else KEEP_SENTENCE}


def e3_twin_macro(e3_dir: Path, seeds: list[str], labels: dict[str, np.ndarray]) -> float | None:
    """Z8: E3's A2 macro on the same clips, or None when its protocol B read is absent."""
    if not (e3_dir / REPORTS / PB_DIR / E3_TWIN / CAP).exists():
        return None
    pb = load_pb(e3_dir, E3_TWIN, seeds)
    if set(pb["labels"]) != set(labels):
        raise ValueError(f"E3 {E3_TWIN} protocol B covers other clips than {CONTROL}")
    return float(np.mean(list(clip_aucs(pb["scores"], labels).values())))


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = [seed_name(s) for s in args.seeds]
    reads = paired_reads(args.e4_dir, CONTROL, args.out_dir, ARM, args.e3_dir, seeds, args.seed,
                         args.alignment, MARGIN)
    labels = load_pb(args.e4_dir, CONTROL, seeds)["labels"]
    decision = decide(reads["decision_interval"], reads["clip_delta"], reads["f2"]["delta"],
                      reads["guardrails"])
    result: dict[str, Any] = {
        "rule": "core/docs/v2/PREREG_ADDENDUM.md §31 (fixed before any A2-ign number)",
        "margin": MARGIN,
        **reads,
        "e3_a2_span_macro": e3_twin_macro(args.e3_dir, seeds, labels),
        "decision": decision,
    }
    out = args.out_dir / REPORTS
    write_json_atomic(out / READOUT_JSON, result)
    write_text_atomic(out / READOUT_MD, render_markdown(result))
    LOGGER.info("Amendment 20: base = %s -> %s", decision["base"], out)
    return result


def _ci(c: dict[str, float] | None) -> str:
    return "n/a" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def render_markdown(r: dict[str, Any]) -> str:
    i, g, d = r["decision_interval"], r["guardrails"], r["diagnostics"]
    lines = [
        f"# v2 Amendment 20 — {ARM} vs {CONTROL} on {r['set']} (drop CRN?)", "",
        f"Rule: {r['rule']}. Margin {r['margin']:g}. **Base = {r['decision']['base']}.**", "",
        f"> {r['decision']['sentence']}", "",
        "## Decision (Z7)", "",
        "| check | result |", "|---|---|",
        *[f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in r["decision"]["checks"].items()], "",
        f"- macro {CONTROL} {_num(r['macro'][CONTROL])} · {ARM} {_num(r['macro'][ARM])} · "
        f"`p_T2` {_num(r['p_t2_macro'])} · E3 {E3_TWIN} (span, printed) "
        f"{_num(r['e3_a2_span_macro'])}",
        f"- paired t95 ({ARM} - {CONTROL}): {i['mean']:+.4f} [{i['low']:+.4f}, {i['high']:+.4f}]"
        f" (n = {i['n']}); per seed "
        + ", ".join(f"{s} {v:+.4f}" for s, v in i["per_seed"].items()),
        f"- cluster bootstrap, per-clip Δ: {_ci(r['clip_delta'])}",
        f"- `f_{r['f2']['weight']:g}` Δ: {_ci(r['f2']['delta'])} (holds for DoTA-CAP's placement "
        "only, pending (bn))", "",
        "## Guardrails (Z6, T2-val seed means)", "",
        f"| read | {CONTROL} | {ARM} | pass |", "|---|---|---|---|",
        f"| macro | {_num(g['t2_val_macro'][CONTROL])} | {_num(g['t2_val_macro'][ARM])} | "
        f"{g['checks']['t2_val_macro']} |",
        f"| normal-window peak (printed) | {_num(g['normal_window_peak'][CONTROL])} | "
        f"{_num(g['normal_window_peak'][ARM])} | — |",
        f"| O1' every seed | — | {g['o1_prime']} | {g['checks']['o1_prime_every_seed']} |", "",
        "## Printed only (Z8)", "",
    ]
    if r["rate_groups"] is not None:
        lines.append("- per-clip Δ by native-fps group: "
                     + "; ".join(f"{k} {_ci(c)}" for k, c in r["rate_groups"].items()))
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
    parser.add_argument("--e3-dir", type=Path, required=True, help="E3's dir (p_T2, A2 printed)")
    parser.add_argument("--e4-dir", type=Path, required=True, help="E4-0's dir (A3-ign)")
    parser.add_argument("--out-dir", type=Path, required=True, help="this amendment's dir (A2-ign)")
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
