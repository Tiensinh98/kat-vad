"""v2 second look: A3-ign on the opened sets (addendum §23, Amendment 12, S1-S8).

Pure aggregation -- no model is loaded. Every number it prints is a **second look at an opened
set** (§19 opened DoTA-CAP-eval and T2-test for A0-A3). Inputs::

    <e3-dir>/REPORTS/pb/A3/dota_cap_dev/                       E3 (A3 dev, optimism)
    <e4-dir>/REPORTS/pb/A3ign/dota_cap_dev/                    E4-0 (A3-ign dev, its steps)
    <final-dir>/REPORTS/pb/{A0,A3}/dota_cap_eval/              the Final (§19)
    <final-dir>/REPORTS/position_prior/dota_cap_eval/          p_T2 on DoTA-CAP-eval
    <final-dir>/t2_test/s<seed>/o1prime_t2test.json            the Final's T2-test rows (A0, A3)
    <sl-dir>/REPORTS/pb/A3ign/dota_cap_eval/                   protocol_b_eval --final, this step
    <sl-dir>/t2_test/s<seed>/o1prime_t2test.json               v2_guard_test (A0, A3ign), this step

S7's integrity gates run first and stop the read-out on any failure. Then S3 (paired t95 of the
per-seed DoTA-CAP-eval macro Δ, A3-ign - A3), S4 (``f_2`` Δ on seed-averaged scores), S5 (the
sentence the outcome licenses) and S6 (T2-test, O1' every seed). Nothing here re-decides (S2).

CLI::

    python -m core.tools.second_look_readout --e3-dir outputs/v2_e3 --e4-dir outputs/v2_e4_0 \\
        --final-dir outputs/v2_final --sl-dir outputs/v2_e4_0/second_look
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools.e3_readout import REPORTS, bootstrap, clip_delta, paired_interval, seed_name
from core.tools.final_readout import (
    A0,
    A3,
    CAP_DEV,
    CAP_EVAL,
    T2_TEST_DIR,
    fused_clip_aucs,
    load_pb,
    load_prior,
)
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import clip_aucs
from core.tools.v2_guard_test import GUARD_JSON

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "second_look_readout.json"
READOUT_MD = "second_look_readout.md"
ARM = constants.V2_E4_0_ARM
LABEL = "second look at an opened set"


# ---------------------------------------------------------------------------
# S7: integrity gates
# ---------------------------------------------------------------------------
def check_steps(got: dict[str, int], want: dict[str, int], label: str) -> None:
    """The read uses exactly the checkpoints it is paired with (J10, S7)."""
    if got != want:
        raise ValueError(f"S7 FAILED, {label}: steps {got} != {want}")


def check_clips(a: dict[str, np.ndarray], b: dict[str, np.ndarray], label: str) -> None:
    """Both arms were scored on the same clips with the same labels (S7)."""
    if set(a) != set(b) or any(not np.array_equal(a[v], b[v]) for v in a):
        raise ValueError(f"S7 FAILED, {label}: clip sets or labels differ")
    if len(a) != constants.V2_SL_CAP_EVAL_CLIPS:
        raise ValueError(f"S7 FAILED, {label}: {len(a)} clips, expected "
                         f"{constants.V2_SL_CAP_EVAL_CLIPS}")


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def t2_rows(final_dir: Path, sl_dir: Path, seeds: list[str], ign_steps: dict[str, int],
            a3_steps: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Per seed: A3-ign and A0 (this step), A3 and A0 (the Final), after the S7 gates."""
    out: dict[str, dict[str, Any]] = {}
    for s in seeds:
        mine = _json(sl_dir / T2_TEST_DIR / s / GUARD_JSON)
        final = _json(final_dir / T2_TEST_DIR / s / GUARD_JSON)
        check_steps({s: mine["runs"][ARM]["global_step"]}, {s: ign_steps[s]}, f"T2-test {ARM}")
        check_steps({s: final["runs"][A3]["global_step"]}, {s: a3_steps[s]}, f"T2-test {A3}")
        for name, r in (("this step", mine), ("Final", final)):
            n = {arm: r["runs"][arm]["windows"] for arm in r["runs"]}
            if set(n.values()) != {constants.V2_SL_T2_TEST_WINDOWS}:
                raise ValueError(f"S7 FAILED, T2-test {name} {s}: windows {n}")
        a0_here, a0_final = mine["t2_test"][A0]["macro"], final["t2_test"][A0]["macro"]
        if abs(a0_here - a0_final) > constants.V2_FINAL_DEV_TOL:
            raise ValueError(f"S7 FAILED, T2-test {s}: A0 macro {a0_here:.6f} here vs "
                             f"{a0_final:.6f} in the Final -- not the same read")
        out[s] = {
            ARM: mine["t2_test"][ARM],
            A3: final["t2_test"][A3],
            A0: final["t2_test"][A0],
            "o1_prime": bool(mine["arms"][ARM]["o1_prime_t2_test"]["pass"]),
        }
    return out


# ---------------------------------------------------------------------------
# S5: the licensed sentence
# ---------------------------------------------------------------------------
def sentences(interval: dict[str, Any], f2: dict[str, float] | None) -> list[str]:
    """§23 S5, mechanically."""
    if interval["low"] > 0.0:
        out = ["Dropping the whole-anchor DVS label improves DoTA-CAP (n/1397) detection, on dev "
               "and on a second look at the opened eval set."]
    elif interval["high"] < 0.0:
        out = ["On the second look, A3-ign scores below A3: the dev gain did not hold. A3-ign "
               "stays the recipe by §22 (no re-decision); both numbers are reported."]
    else:
        out = ["On the second look, A3-ign's gain over A3 is not detectable at this size "
               "(dev: +0.040)."]
    if f2 is not None and f2["low"] > 0.0:
        out.append("... beyond the position prior (second look).")
    else:
        out.append("No beyond-position sentence on the second look (f_2 interval not above 0).")
    return out


def _macro(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> float:
    return float(np.mean(list(clip_aucs(scores, labels).values())))


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = sorted(seed_name(s) for s in args.seeds)
    a3_dev = load_pb(args.e3_dir, A3, CAP_DEV)
    ign_dev = load_pb(args.e4_dir, ARM, CAP_DEV)
    a3 = load_pb(args.final_dir, A3, CAP_EVAL)
    a0 = load_pb(args.final_dir, A0, CAP_EVAL)
    ign = load_pb(args.sl_dir, ARM, CAP_EVAL)
    _, prior = load_prior(args.final_dir, CAP_EVAL)

    ign_steps = ign_dev["readout"]["global_steps"]
    check_steps(ign["readout"]["global_steps"], ign_steps, f"{ARM} eval vs E4-0 dev")
    check_steps(a3["readout"]["global_steps"], a3_dev["readout"]["global_steps"], "A3 eval vs E3")
    check_clips(ign["labels"], a3["labels"], f"{ARM} vs A3 on {CAP_EVAL}")
    if sorted(ign["readout"]["per_run_macro"]) != seeds:
        raise ValueError(f"S7 FAILED: {ARM} runs {sorted(ign['readout']['per_run_macro'])}")
    labels = a3["labels"]

    arms = ((ARM, ign), (A3, a3), (A0, a0))
    per_run = {name: pb["readout"]["per_run_macro"] for name, pb in arms}
    interval = paired_interval(per_run[ARM], per_run[A3], constants.V2_E3_T95)
    weight = constants.V2_FINAL_FUSION_PRIMARY
    f2 = {name: fused_clip_aucs(pb["scores"], prior, labels, weight)
          for name, pb in ((ARM, ign), (A3, a3))}
    f2_delta = bootstrap(clip_delta(f2[ARM], f2[A3]), args.seed)
    macro = {name: _macro(pb["scores"], labels) for name, pb in arms}
    dev_macro = {ARM: _macro(ign_dev["scores"], ign_dev["labels"]),
                 A3: _macro(a3_dev["scores"], a3_dev["labels"])}

    t2 = t2_rows(args.final_dir, args.sl_dir, seeds, ign_steps, a3_dev["readout"]["global_steps"])
    t2_means = {arm: {k: float(np.mean([t2[s][arm][k] for s in seeds]))
                      for k in ("macro", "micro", "clip_oracle_micro", "window_level_auc")}
                for arm in (ARM, A3, A0)}
    result: dict[str, Any] = {
        "rule": "core/docs/v2/PREREG_ADDENDUM.md §23 (Amendment 12, S1-S8)",
        "label": LABEL,
        "seeds": seeds,
        "dota_cap_eval": {
            "clips": len(labels),
            "macro_seed_averaged": macro,
            "p_t2": _macro(prior, labels),
            "per_run_macro": per_run,
            "delta_vs_a3": interval,
            "delta_vs_a0": paired_interval(per_run[ARM], per_run[A0], constants.V2_E3_T95),
            "clip_delta_vs_a3": bootstrap(
                clip_delta(clip_aucs(ign["scores"], labels), clip_aucs(a3["scores"], labels)),
                args.seed),
            "f2": {"weight": weight,
                   "macro": {k: float(np.mean(list(v.values()))) for k, v in f2.items()},
                   "delta": f2_delta},
            "optimism_dev_minus_eval": {k: dev_macro[k] - macro[k] for k in dev_macro},
            "dev_macro": dev_macro,
        },
        "t2_test": {
            "means": t2_means,
            "delta_macro_vs_a3": paired_interval(
                {s: float(t2[s][ARM]["macro"]) for s in seeds},
                {s: float(t2[s][A3]["macro"]) for s in seeds}, constants.V2_E3_T95),
            "o1_prime": {s: t2[s]["o1_prime"] for s in seeds},
            "o1_prime_every_seed": all(t2[s]["o1_prime"] for s in seeds),
        },
        "sentences": sentences(interval, f2_delta),
    }
    out = args.sl_dir / REPORTS
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / READOUT_JSON, result)
    write_text_atomic(out / READOUT_MD, render_markdown(result))
    LOGGER.info("second look: %s - A3 %+.4f [%+.4f, %+.4f] -> %s", ARM, interval["mean"],
                interval["low"], interval["high"], out)
    return result


def _ci(c: dict[str, Any] | None) -> str:
    return "n/a" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def render_markdown(r: dict[str, Any]) -> str:
    d, t = r["dota_cap_eval"], r["t2_test"]
    m, o = d["macro_seed_averaged"], d["optimism_dev_minus_eval"]
    i = d["delta_vs_a3"]
    o1 = "PASS" if t["o1_prime_every_seed"] else "**FAIL** (guardrail failed on the second look)"
    return "\n".join([
        f"# v2 second look — {ARM} on the opened sets ({LABEL})",
        "",
        f"Rule: {r['rule']}. Nothing re-decides (S2): `B` = {ARM} stands; §19's numbers stay.",
        "",
        f"## DoTA-CAP (n/1397) eval — {d['clips']} clips",
        "",
        "| arm | macro (seed-averaged) | dev macro | optimism (dev - eval) |",
        "|---|---|---|---|",
        f"| {ARM} | {m[ARM]:.4f} | {d['dev_macro'][ARM]:.4f} | {o[ARM]:+.4f} |",
        f"| A3 | {m[A3]:.4f} | {d['dev_macro'][A3]:.4f} | {o[A3]:+.4f} |",
        f"| A0 | {m[A0]:.4f} | — | — |",
        f"| `p_T2` | {d['p_t2']:.4f} | — | — |",
        "",
        f"- **{ARM} - A3, paired t95 (S3): {i['mean']:+.4f} [{i['low']:+.4f}, {i['high']:+.4f}]**"
        f" (n = {i['n']}); per seed "
        + ", ".join(f"{s} {v:+.4f}" for s, v in i["per_seed"].items()),
        f"- {ARM} - A0 (printed): {_ci(d['delta_vs_a0'])}; clip-level {ARM} - A3 (printed): "
        f"{_ci(d['clip_delta_vs_a3'])}",
        f"- `f_{d['f2']['weight']:g}` (S4): {ARM} {d['f2']['macro'][ARM]:.4f} · A3 "
        f"{d['f2']['macro'][A3]:.4f} · Δ {_ci(d['f2']['delta'])}",
        "",
        "## T2-test (S6) — seed means",
        "",
        "| arm | macro | micro | clip oracle | window AUC |",
        "|---|---|---|---|---|",
        *[f"| {a} | {v['macro']:.4f} | {v['micro']:.4f} | {v['clip_oracle_micro']:.4f} | "
          f"{v['window_level_auc']:.4f} |" for a, v in t["means"].items()],
        "",
        f"- {ARM} - A3 T2-test macro, paired t95 (printed): {_ci(t['delta_macro_vs_a3'])}",
        f"- O1' vs A0, every seed: {o1} {t['o1_prime']}",
        "",
        "## Sentences (S5)",
        "",
        *[f"- {s}" for s in r["sentences"]],
        "",
    ])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--e3-dir", type=Path, required=True, help="E3's dir (A3 dev)")
    parser.add_argument("--e4-dir", type=Path, required=True, help="E4-0's dir (A3-ign dev)")
    parser.add_argument("--final-dir", type=Path, required=True, help="the Final's dir (§19)")
    parser.add_argument("--sl-dir", type=Path, required=True, help="this step's dir")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(constants.V2_E3_SEEDS))
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap seed")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
