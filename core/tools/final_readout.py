"""v2 Final: the sealed-set read-out of the adopted model (addendum §19, Amendment 10, P1-P8).

Pure aggregation -- no model is loaded. It reads E3's dev outputs and the Final notebook's
sealed-set outputs, both laid out as ``e3_readout`` expects::

    <e3-dir>/REPORTS/e3_readout.json                                  E3's decision (P1)
    <e3-dir>/REPORTS/pb/<arm>/{dota_dev,dota_cap_dev}/                 protocol_b_eval
    <e3-dir>/REPORTS/position_prior/{dota_dev,dota_cap_dev}/          position_prior
    <final-dir>/REPORTS/pb/<arm>/{dota_eval,dota_cap_eval}/            protocol_b_eval --final
    <final-dir>/REPORTS/position_prior/{dota_eval,dota_cap_eval}/     position_prior --final
    <final-dir>/t2_test/s<seed>/o1prime_t2test.json                   v2_guard_test, per seed

What it does, in the order §19 fixes:

* **P7 gate** (:func:`dev_gate`, also its own ``--gate-only`` CLI, run *before* any sealed set
  is opened): the position-controlled read recomputed on DoTA-CAP-dev must reproduce
  ``RESULTS_E3.md`` §5's means to ``V2_FINAL_DEV_TOL``.
* **P1:** E3's decision must be the recorded one (A3, ``F`` = A0); nothing here re-decides.
* **P3:** seed-mean macro, the §10.3 contrasts (descriptive), clip-level Δ, share bins,
  optimism = dev - eval, ``p_T2`` and ``t/N``, DoTA micro/AP (per-clip min-max, lesson 12),
  T2-test micro / macro / clip oracle / window AUC and O1' per seed.
* **P4/P5:** ``f_w = z(ȳ) + w · z(p_T2)`` within each clip, its macro and paired Δs, and the
  sentences each outcome licenses at ``w = 2``.

Name every DoTA-CAP number **DoTA-CAP (n/1397)**; never put one beside full DoTA, 62.60 or a
phase-4 number (D14). Never "motion" (G3 / D6).

CLI::

    python -m core.tools.final_readout --e3-dir outputs/v2_e3 --gate-only
    python -m core.tools.final_readout --e3-dir outputs/v2_e3 --final-dir outputs/v2_final
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import frame_auc, pooled_metrics
from core.tools.e3_readout import (
    E3_READOUT_JSON,
    PB_DIR,
    PRIOR_DIR,
    REPORTS,
    bootstrap,
    by_share_bin,
    clip_delta,
    paired_interval,
    seed_name,
)
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.position_prior import PRIOR_NPZ
from core.tools.position_prior import READOUT_JSON as PRIOR_READOUT_JSON
from core.tools.protocol_b_eval import (
    CLIP_AUCS_JSON,
    CLIP_SCORES_NPZ,
    read_clip_scores,
)
from core.tools.protocol_b_eval import (
    READOUT_JSON as PB_READOUT_JSON,
)
from core.tools.v2_diagnostics import o1_prime
from core.tools.v2_guard_test import GUARD_JSON

LOGGER = logging.getLogger(__name__)

FINAL_READOUT_JSON = "final_readout.json"
FINAL_READOUT_MD = "final_readout.md"
GATE_JSON = "final_gate.json"
T2_TEST_DIR = "t2_test"

A0, A1, A2, A3 = constants.V2_E3_ARMS
DEV = constants.V2_SPLIT_DOTA_DEV
CAP_DEV = constants.V2_SPLIT_DOTA_CAP_DEV
EVAL = constants.V2_SPLIT_DOTA_EVAL
CAP_EVAL = constants.V2_SPLIT_DOTA_CAP_EVAL
DEV_OF = {EVAL: DEV, CAP_EVAL: CAP_DEV}
# P2: A2/A3 have V2-S features on DoTA-CAP only (D13)
EVAL_SPLITS_OF = {A0: (EVAL, CAP_EVAL), A1: (EVAL, CAP_EVAL), A2: (CAP_EVAL,), A3: (CAP_EVAL,)}
CONTRASTS = (
    ("A1 - A0", A1, A0, EVAL),
    ("A2 - A0", A2, A0, CAP_EVAL),
    ("A3 - A0", A3, A0, CAP_EVAL),
    ("A3 - A1", A3, A1, CAP_EVAL),
    ("A2 - A1", A2, A1, CAP_EVAL),
)
FUSION_PAIRS = ((A3, A0), (A2, A0), (A3, A1))  # P4: the Δs printed under position control


# ---------------------------------------------------------------------------
# P4: the position-controlled read
# ---------------------------------------------------------------------------
def zscore(values: np.ndarray) -> np.ndarray:
    """Standardize within one clip; a constant series maps to zeros."""
    x = np.asarray(values, dtype=np.float64)
    sd = float(x.std())
    return np.zeros_like(x) if sd == 0.0 else (x - x.mean()) / sd


def fused_clip_aucs(
    scores: dict[str, np.ndarray],
    prior: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    weight: float | None,
) -> dict[str, float]:
    """Per two-class clip AUC of ``z(score) + weight · z(p_T2)``; ``None`` = the score alone."""
    out: dict[str, float] = {}
    for v in sorted(scores):
        lab = np.asarray(labels[v])
        if not 0 < int(lab.sum()) < len(lab):
            continue
        if len(scores[v]) != len(prior[v]):
            raise ValueError(f"{v}: {len(scores[v])} scores vs {len(prior[v])} prior frames")
        fused = scores[v] if weight is None else zscore(scores[v]) + weight * zscore(prior[v])
        out[v] = frame_auc(np.asarray(fused, dtype=np.float64), lab)
    return out


def position_control(
    scores: dict[str, dict[str, np.ndarray]],
    prior: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    """P4 on one set: per arm and weight the fused macro; paired Δs; arm + prior vs prior."""
    weights: list[float | None] = [None, *constants.V2_FINAL_FUSION_WEIGHTS]
    aucs = {
        arm: {_wkey(w): fused_clip_aucs(s, prior, labels, w) for w in weights}
        for arm, s in scores.items()
    }
    prior_aucs = fused_clip_aucs(prior, prior, labels, None)
    macro = {arm: {k: float(np.mean(list(a.values()))) for k, a in per.items()}
             for arm, per in aucs.items()}
    deltas = {
        f"{x} - {y}": {k: bootstrap(clip_delta(aucs[x][k], aucs[y][k]), seed) for k in aucs[x]}
        for x, y in FUSION_PAIRS
        if x in aucs and y in aucs
    }
    primary = _wkey(constants.V2_FINAL_FUSION_PRIMARY)
    over_prior = {
        arm: bootstrap(clip_delta(aucs[arm][primary], prior_aucs), seed) for arm in aucs
    }
    return {
        "p_t2_macro": float(np.mean(list(prior_aucs.values()))),
        "macro": macro,
        "delta": deltas,
        "over_prior": over_prior,
        "two_class": len(prior_aucs),
    }


def _wkey(weight: float | None) -> str:
    return "raw" if weight is None else f"w={weight:g}"


def sentences(control: dict[str, Any]) -> list[str]:
    """P5 (a)-(c) at the primary weight; printed, never moves P1."""
    w = _wkey(constants.V2_FINAL_FUSION_PRIMARY)

    def above(c: dict[str, float] | None) -> bool:
        return c is not None and c["low"] > 0.0

    out = [
        "A3's gain over A0 survives the T2 position prior."
        if above(control["delta"]["A3 - A0"][w]) else
        f"A3's gain over A0 is not separable from the T2 position prior at {w}.",
        "The video stream's gain on top of CRN (A3 - A1) survives the T2 position prior."
        if above(control["delta"]["A3 - A1"][w]) else
        f"The video stream's gain on top of CRN (A3 - A1) is not separable from the T2 "
        f"position prior at {w}.",
        "A3 carries signal beyond position."
        if above(control["over_prior"][A3]) else
        "A3 adds nothing measurable to position.",
    ]
    return out


# ---------------------------------------------------------------------------
# P7: the dev regression gate
# ---------------------------------------------------------------------------
def _json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Final input missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_pb(root: Path, arm: str, split: str) -> dict[str, Any]:
    d = root / REPORTS / PB_DIR / arm / split
    scores, labels = read_clip_scores(d / CLIP_SCORES_NPZ)
    return {
        "readout": _json(d / PB_READOUT_JSON),
        "clip_aucs": _json(d / CLIP_AUCS_JSON),
        "scores": scores,
        "labels": labels,
    }


def load_prior(root: Path, split: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    d = root / REPORTS / PRIOR_DIR / split
    scores, _ = read_clip_scores(d / PRIOR_NPZ)
    return _json(d / PRIOR_READOUT_JSON), scores


def dev_gate(e3_dir: Path, seed: int) -> dict[str, Any]:
    """P7: recompute P4 on DoTA-CAP-dev and compare every mean to RESULTS_E3.md §5."""
    pbs = {arm: load_pb(e3_dir, arm, CAP_DEV) for arm in constants.V2_E3_ARMS}
    _, prior = load_prior(e3_dir, CAP_DEV)
    control = position_control(
        {arm: pb["scores"] for arm, pb in pbs.items()}, prior, pbs[A0]["labels"], seed
    )
    keys = [_wkey(w) for w in (None, *constants.V2_FINAL_FUSION_WEIGHTS)]
    checks: list[dict[str, Any]] = [
        _check("p_T2", control["p_t2_macro"], constants.V2_FINAL_DEV_P_T2)
    ]
    for arm, recorded in constants.V2_FINAL_DEV_ARM_MACRO.items():
        checks += [_check(f"{arm} {k}", control["macro"][arm][k], r)
                   for k, r in zip(keys, recorded, strict=True)]
    for (x, y), recorded in constants.V2_FINAL_DEV_DELTA.items():
        got = control["delta"][f"{x} - {y}"]
        checks += [_check(f"{x} - {y} {k}", _mean(got[k]), r)
                   for k, r in zip(keys, recorded, strict=True)]
    failed = [c for c in checks if not c["pass"]]
    return {"pass": not failed, "failed": failed, "checks": checks, "control": control}


def _mean(c: dict[str, float] | None) -> float:
    if c is None:
        raise ValueError("a recorded Δ has no clips")
    return float(c["mean"])


def _check(name: str, got: float, recorded: float) -> dict[str, Any]:
    return {"name": name, "got": got, "recorded": recorded,
            "pass": abs(got - recorded) <= constants.V2_FINAL_DEV_TOL}


# ---------------------------------------------------------------------------
# P1 / P3: the read-out
# ---------------------------------------------------------------------------
def check_decision(e3: dict[str, Any]) -> None:
    """P1: the Final reads the adoption E3 recorded; it never re-decides."""
    got = (e3["decision"]["adopt"], e3["decision"]["F"])
    want = (constants.V2_FINAL_ADOPTED, constants.V2_FINAL_F_ARM)
    if got != want:
        raise ValueError(f"E3 decision {got} is not the recorded one {want} (§19 P1)")


def check_same_checkpoints(dev: dict[str, Any], ev: dict[str, Any], label: str) -> None:
    """The eval read must be the very checkpoints E3 scored (M1, J10)."""
    if dev["readout"]["global_steps"] != ev["readout"]["global_steps"]:
        raise ValueError(
            f"{label}: eval steps {ev['readout']['global_steps']} != "
            f"dev steps {dev['readout']['global_steps']} -- not E3's checkpoints"
        )


def micro_ap(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> dict[str, float]:
    """DoTA micro AUC / AP under per-clip min-max (lesson 12), seed-averaged scores."""
    ids = sorted(scores)
    m = pooled_metrics([scores[v] for v in ids], [labels[v] for v in ids],
                       constants.SCORE_NORM_MINMAX)
    return {"micro": float(m["auc"]), "ap": float(m["ap"])}


def t2_test(
    final_dir: Path, seeds: list[str], dev_steps: dict[str, dict[str, int]]
) -> dict[str, Any]:
    """P3, T2-test: per seed micro / macro / oracle / window AUC; O1' vs A0 of that seed."""
    per_seed: dict[str, dict[str, Any]] = {}
    for s in seeds:
        r = _json(final_dir / T2_TEST_DIR / s / GUARD_JSON)
        rows = r["t2_test"]
        for arm in constants.V2_E3_ARMS:
            step = r["runs"][arm]["global_step"]
            if step != dev_steps[arm][s]:
                raise ValueError(f"T2-test {arm} {s}: step {step} != E3's {dev_steps[arm][s]}")
        per_seed[s] = {
            arm: {
                "micro": float(rows[arm]["micro"]),
                "macro": float(rows[arm]["macro"]),
                "clip_oracle": float(rows[arm]["clip_oracle_micro"]),
                "window_auc": float(rows[arm]["window_level_auc"]),
                "o1_registered": bool(rows[arm]["pass"]),
                **({} if arm == A0 else {"o1_prime": o1_prime(rows[arm], rows[A0])["pass"]}),
            }
            for arm in constants.V2_E3_ARMS
        }
    means = {
        arm: {k: float(np.mean([per_seed[s][arm][k] for s in seeds]))
              for k in ("micro", "macro", "clip_oracle", "window_auc")}
        for arm in constants.V2_E3_ARMS
    }
    o1p = {arm: sum(per_seed[s][arm]["o1_prime"] for s in seeds)
           for arm in constants.V2_E3_ARMS if arm != A0}
    macro_delta = {
        f"{arm} - A0": paired_interval(
            {s: per_seed[s][arm]["macro"] for s in seeds},
            {s: per_seed[s][A0]["macro"] for s in seeds}, constants.V2_E3_T95,
        )
        for arm in (A1, A2, A3)
    }
    return {"per_seed": per_seed, "means": means, "o1_prime_pass": o1p,
            "macro_delta_vs_a0": macro_delta}


def run(args: argparse.Namespace) -> dict[str, Any]:
    gate = dev_gate(args.e3_dir, args.seed)
    if not gate["pass"]:
        raise ValueError(f"P7 gate FAILS -- no sealed read is valid: {gate['failed']}")
    check_decision(_json(args.e3_dir / REPORTS / E3_READOUT_JSON))
    seeds = sorted(seed_name(s) for s in args.seeds)

    dev = {arm: {split: load_pb(args.e3_dir, arm, DEV_OF[split]) for split in splits}
           for arm, splits in EVAL_SPLITS_OF.items()}
    ev = {arm: {split: load_pb(args.final_dir, arm, split) for split in splits}
          for arm, splits in EVAL_SPLITS_OF.items()}
    for arm, splits in EVAL_SPLITS_OF.items():
        for split in splits:
            check_same_checkpoints(dev[arm][split], ev[arm][split], f"{arm}/{split}")
            if sorted(ev[arm][split]["readout"]["runs"]) != seeds:
                raise ValueError(f"{arm}/{split}: runs are not the E3 seeds {seeds}")

    def per_seed_macro(arm: str, split: str) -> dict[str, float]:
        return {s: float(m) for s, m in ev[arm][split]["readout"]["per_run_macro"].items()}

    contrasts = {
        name: {**paired_interval(per_seed_macro(x, split), per_seed_macro(y, split),
                                 constants.V2_E3_T95), "split": split}
        for name, x, y, split in CONTRASTS
    }
    clip_level = {
        name: bootstrap(clip_delta(ev[x][split]["clip_aucs"], ev[y][split]["clip_aucs"]),
                        args.seed)
        for name, x, y, split in CONTRASTS
    }
    share = {
        split: {v: float(np.mean(lab)) for v, lab in ev[A0][split]["labels"].items()}
        for split in (EVAL, CAP_EVAL)
    }
    priors = {split: load_prior(args.final_dir, split) for split in (EVAL, CAP_EVAL)}
    arms_out: dict[str, Any] = {}
    for arm, splits in EVAL_SPLITS_OF.items():
        arms_out[arm] = {}
        for split in splits:
            e, d = ev[arm][split]["readout"], dev[arm][split]["readout"]
            arms_out[arm][split] = {
                "clips": e["clips"],
                "split_clips": e.get("split_clips", e["clips"]),
                "excluded_featureless": e.get("excluded_featureless", []),
                "macro": e["macro"],
                "dev_macro": d["macro"],
                "optimism": float(d["macro"]["mean"] - e["macro"]["mean"]),
                "share_bins": by_share_bin(ev[arm][split]["clip_aucs"], share[split], args.seed),
                **micro_ap(ev[arm][split]["scores"], ev[arm][split]["labels"]),
            }
    cap_scores = {arm: ev[arm][CAP_EVAL]["scores"] for arm in constants.V2_E3_ARMS}
    control = position_control(
        cap_scores, priors[CAP_EVAL][1], ev[A0][CAP_EVAL]["labels"], args.seed
    )
    dev_steps = {arm: dev[arm][CAP_EVAL]["readout"]["global_steps"]
                 for arm in constants.V2_E3_ARMS}
    readout: dict[str, Any] = {
        "addendum": "PREREG_ADDENDUM.md §19 (Amendment 10, P1-P8); proposal §10.2",
        "seeds": seeds,
        "adopted": constants.V2_FINAL_ADOPTED,
        "f_arm": constants.V2_FINAL_F_ARM,
        "p7_gate": {"pass": gate["pass"], "checks": len(gate["checks"])},
        "arms": arms_out,
        "contrasts": contrasts,
        "clip_level_paired_delta": clip_level,
        "position": {
            split: {"p_t2": priors[split][0]["p_t2_macro"],
                    "monotone_ruler": priors[split][0]["monotone_ruler_macro"]}
            for split in (EVAL, CAP_EVAL)
        },
        "p_cap_dev": constants.V2_E3_P_CAP,
        "position_control": {CAP_EVAL: control, CAP_DEV: gate["control"]},
        "sentences": sentences(control),
        "t2_test": t2_test(args.final_dir, seeds, dev_steps),
    }
    out = args.final_dir / REPORTS
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / FINAL_READOUT_JSON, readout)
    write_text_atomic(out / FINAL_READOUT_MD, render_markdown(readout))
    LOGGER.info("Final read-out -> %s", out)
    return readout


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _ci(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _level(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _set(split: str) -> str:
    return {EVAL: "DoTA-eval", CAP_EVAL: "DoTA-CAP-eval (n/1397)",
            CAP_DEV: "DoTA-CAP-dev (n/1397)", DEV: "DoTA-dev"}[split]


def render_gate(gate: dict[str, Any]) -> str:
    lines = [f"# Final P7 gate: {'PASS' if gate['pass'] else 'FAIL'}", "",
             "| read | got | recorded (RESULTS_E3.md §5) | ok |", "|---|---|---|---|"]
    lines += [f"| {c['name']} | {c['got']:.5f} | {c['recorded']:.4f} | "
              f"{'✓' if c['pass'] else '✗'} |" for c in gate["checks"]]
    return "\n".join(lines) + "\n"


def render_markdown(r: dict[str, Any]) -> str:
    weights = [_wkey(w) for w in (None, *constants.V2_FINAL_FUSION_WEIGHTS)]
    lines = [
        "# KAT-VAD v2 — Final read-out (sealed sets, opened once)",
        "",
        f"Rules: {r['addendum']}. Seeds {', '.join(r['seeds'])}. Adopted **{r['adopted']}** "
        f"(`F` = {r['f_arm']}), closed at E3 (P1): nothing below re-decides. P7 gate: PASS "
        f"({r['p7_gate']['checks']} dev means reproduced).",
        "",
        "Never \"motion\" (G3 / D6). A2/A3: \"O1 FAIL as registered; not collapsed under O1'\" "
        "(§17 Q5). §18 provenance: rules frozen in `59ee17e`, signed after the E3 read-out.",
        "",
        "## Position control (P4/P5) — DoTA-CAP-eval (n/1397)",
        "",
        *(f"* **{s}**" for s in r["sentences"]),
        "",
        "| arm | " + " | ".join(weights) + " |",
        "|---|" + "---|" * len(weights),
    ]
    for split in (CAP_EVAL, CAP_DEV):
        c = r["position_control"][split]
        lines.append(f"| `p_T2` alone ({_set(split)}) | {c['p_t2_macro']:.4f} |"
                     + " |" * (len(weights) - 1))
        lines += [f"| {arm} ({_set(split)}) | "
                  + " | ".join(f"{c['macro'][arm][k]:.4f}" for k in weights) + " |"
                  for arm in c["macro"]]
    lines += ["", "| Δ (clip-level, cluster bootstrap) | set | " + " | ".join(weights) + " |",
              "|---|---|" + "---|" * len(weights)]
    for split in (CAP_EVAL, CAP_DEV):
        c = r["position_control"][split]
        lines += [f"| {name} | {_set(split)} | " + " | ".join(_ci(d[k]) for k in weights) + " |"
                  for name, d in c["delta"].items()]
        lines += [f"| {arm} + 2p - `p_T2` | {_set(split)} | {_ci(c['over_prior'][arm])} |"
                  + " |" * (len(weights) - 1) for arm in c["over_prior"]]
    lines += [
        "",
        "## Macro per arm (seed-averaged) and optimism",
        "",
        "| arm | set | eval macro | dev macro | optimism (dev - eval) | micro (min-max) | AP |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, per in r["arms"].items():
        lines += [f"| {arm} | {_set(split)} | {_level(a['macro'])} | {_level(a['dev_macro'])} | "
                  f"{a['optimism']:+.4f} | {a['micro']:.4f} | {a['ap']:.4f} |"
                  for split, a in per.items()]
    lines += ["", "DoTA-eval and DoTA-CAP-eval are different clip sets (D14): never compare them.",
              ""]
    for split in (EVAL, CAP_EVAL):
        a = r["arms"][A0][split]
        dropped = ", ".join(a["excluded_featureless"]) or "none"
        lines.append(f"* {_set(split)}: scored {a['clips']} / {a['split_clips']} clips; "
                     f"excluded without CLIP features (§20): {dropped}")
    for split, p in r["position"].items():
        lines.append(f"* {_set(split)}: `p_T2` {_level(p['p_t2'])}; `t/N` (monotone) "
                     f"{_level(p['monotone_ruler'])}")
    lines.append(f"* `p_CAP` (dev, in-domain ceiling): {r['p_cap_dev']:.3f}")
    lines += [
        "",
        "## Contrasts (descriptive: paired t95 over seeds; the decision was E3's)",
        "",
        "| contrast | set | mean Δ | t95 | clip-level Δ |",
        "|---|---|---|---|---|",
        *(f"| {name} | {_set(c['split'])} | {c['mean']:+.4f} | [{c['low']:+.4f}, {c['high']:+.4f}] "
          f"| {_ci(r['clip_level_paired_delta'][name])} |" for name, c in r["contrasts"].items()),
        "",
        "## Macro per accident-share bin",
        "",
        "| arm | set | " + " | ".join(constants.V2_SHARE_BIN_LABELS) + " |",
        "|---|---|" + "---|" * len(constants.V2_SHARE_BIN_LABELS),
    ]
    for arm, per in r["arms"].items():
        lines += [f"| {arm} | {_set(split)} | "
                  + " | ".join(_level(a["share_bins"][b]) for b in constants.V2_SHARE_BIN_LABELS)
                  + " |" for split, a in per.items()]
    t2 = r["t2_test"]
    lines += [
        "",
        "## T2-test (1,106 windows; key secondary, never an adoption)",
        "",
        "| arm | micro | macro | clip oracle | window AUC | O1' pass (of 5) "
        "| macro Δ vs A0 (t95) |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, m in t2["means"].items():
        o1p = "—" if arm == A0 else str(t2["o1_prime_pass"][arm])
        d = t2["macro_delta_vs_a0"].get(f"{arm} - A0")
        dl = "—" if d is None else f"{d['mean']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]"
        lines.append(f"| {arm} | {m['micro']:.4f} | {m['macro']:.4f} | {m['clip_oracle']:.4f} | "
                     f"{m['window_auc']:.4f} | {o1p} | {dl} |")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--e3-dir", type=Path, required=True, help="E3's dir (dev reads)")
    parser.add_argument("--final-dir", type=Path, default=None, help="the Final dir (sealed reads)")
    parser.add_argument("--gate-only", action="store_true",
                        help="P7 only: reproduce RESULTS_E3.md §5 on dev, open nothing")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(constants.V2_E3_SEEDS))
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap seed")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.gate_only:
        gate = dev_gate(args.e3_dir, args.seed)
        out = args.e3_dir / REPORTS
        write_json_atomic(out / GATE_JSON, {k: v for k, v in gate.items() if k != "control"})
        LOGGER.info("\n%s", render_gate(gate))
        if not gate["pass"]:
            raise SystemExit("P7 gate FAILS: do not open any sealed set (§19 P7)")
        return
    if args.final_dir is None:
        raise SystemExit("--final-dir is required unless --gate-only")
    run(args)


if __name__ == "__main__":
    main()
