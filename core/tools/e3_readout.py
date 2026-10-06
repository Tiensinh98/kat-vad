"""v2 E3: the 2 x 2 factorial read-out and adoption decision (proposal §10.1-§10.3, Amendment 9).

Pure aggregation -- no model is loaded. It reads, from one E3 directory laid out by
``colab/v2/e3_factorial.ipynb``::

    <e3-dir>/<arm>/s<seed>/diag/diag.json                      v2_diagnostics (M2)
    <e3-dir>/REPORTS/pb/<arm>/<split>/protocol_b_readout.json  protocol_b_eval (M3)
    <e3-dir>/REPORTS/pb/<arm>/<split>/clip_aucs.json
    <e3-dir>/REPORTS/pb/<arm>/<split>/clip_scores.npz
    <e3-dir>/REPORTS/position_prior/<split>/position_prior.npz position_prior (M6 e)

and applies the pre-registered rules mechanically:

* **Decision interval (M4):** per-seed macro Δ paired by seed, mean ± t95 · SD / √5.
  Adoption needs the lower bound > 0.
* **Sets (D13):** A1 - A0 and ``F`` on ``dota_dev``; every motion contrast on ``dota_cap_dev``.
* **Guardrails (M2 / Q6):** A1-A3 pass O1' on every seed; A0 is printed under O1.
* **Free rule (M5)** -> ``F``; then A3, A2, A1, A0 in that order (§10.3).
* **Printed, never decided on (M6, M7):** clip-level paired bootstrap Δs, share bins, mean
  diagnostics, the T2 position prior ``p_T2`` and per-arm score-position Spearman (G6),
  ``MDE_E3``. The wording it prints follows M8 (never "motion", D6).

CLI::

    python -m core.tools.e3_readout --e3-dir outputs/v2/e3
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
from core.data.v2_splits import dota_group, share_bin
from core.metrics import cluster_bootstrap_ci
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
from core.tools.v2_diagnostics import DIAG_JSON

LOGGER = logging.getLogger(__name__)

E3_READOUT_JSON = "e3_readout.json"
E3_READOUT_MD = "e3_readout.md"
REPORTS = "REPORTS"
PB_DIR = "pb"
PRIOR_DIR = "position_prior"
DIAG_DIR = "diag"

DEV = constants.V2_SPLIT_DOTA_DEV
CAP = constants.V2_SPLIT_DOTA_CAP_DEV
A0, A1, A2, A3 = constants.V2_E3_ARMS
MOTION_ARMS = (A2, A3)
# (name, X, Y, split) -- the paired contrasts of §10.3 on the sets of D13; F is resolved later
MAIN_CONTRASTS = (
    ("A1 - A0", A1, A0, DEV),
    ("A2 - A0", A2, A0, CAP),
    ("A3 - A0", A3, A0, CAP),
    ("A3 - A1", A3, A1, CAP),
)
# X - F with F = A1 needs A2 - A1 too (A3 - A1 is already main); MDE_E3 pools MAIN only (M7)
F_CONTRASTS = (("A2 - A1", A2, A1, CAP),)
SPLITS_OF = {A0: (DEV, CAP), A1: (DEV, CAP), A2: (CAP,), A3: (CAP,)}
G6_PAIRS = ((A2, A0), (A3, A1))


def seed_name(seed: int) -> str:
    return f"s{seed}"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def paired_interval(x: dict[str, float], y: dict[str, float], t95: float) -> dict[str, Any]:
    """§10.1 decision interval: mean ± t95 · SD / √n of the per-seed Δ, paired by seed."""
    if set(x) != set(y):
        raise ValueError(f"contrast arms have different seeds: {sorted(x)} vs {sorted(y)}")
    seeds = sorted(x)
    deltas = np.array([x[s] - y[s] for s in seeds], dtype=np.float64)
    if len(deltas) < 2:
        raise ValueError("a paired seed interval needs at least two seeds")
    sd = float(deltas.std(ddof=1))
    half = t95 * sd / math.sqrt(len(deltas))
    mean = float(deltas.mean())
    return {
        "mean": mean, "low": mean - half, "high": mean + half, "sd": sd, "n": len(deltas),
        "per_seed": dict(zip(seeds, deltas.tolist(), strict=True)),
    }


def positive(interval: dict[str, Any]) -> bool:
    """Adoption reading of "excludes 0" (M4): the whole interval above 0."""
    return bool(interval["low"] > 0.0)


def excludes_zero(interval: dict[str, Any]) -> bool:
    return bool(interval["low"] > 0.0 or interval["high"] < 0.0)


def mde_e3(intervals: list[dict[str, Any]], t95: float) -> float:
    """M7: t95 · SD_pooled / √n, SD pooled over the main contrasts' per-seed Δ."""
    pooled = math.sqrt(float(np.mean([i["sd"] ** 2 for i in intervals])))
    return t95 * pooled / math.sqrt(intervals[0]["n"])


def clip_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {v: a[v] - b[v] for v in sorted(set(a) & set(b))}


def bootstrap(values: dict[str, float], seed: int) -> dict[str, float] | None:
    groups = {v: dota_group(v) for v in values}
    return cluster_bootstrap_ci(
        values, groups, constants.V2_E1_BOOTSTRAP, seed, constants.V2_E1_CI
    )


def by_share_bin(
    values: dict[str, float], share: dict[str, float], seed: int
) -> dict[str, dict[str, float] | None]:
    return {
        label: bootstrap({v: x for v, x in values.items() if share_bin(share[v]) == label}, seed)
        for label in constants.V2_SHARE_BIN_LABELS
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    """0-based ranks, ties sharing their mean rank (Spearman's convention)."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return np.asarray((starts + (counts - 1) / 2.0)[inverse], dtype=np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rho = Pearson's r of the average ranks; both inputs non-constant."""
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def clip_spearman(
    scores: dict[str, np.ndarray], prior: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> tuple[dict[str, float], int]:
    """Per two-class clip Spearman(score, ``p_T2``); clips with a constant score are dropped."""
    out: dict[str, float] = {}
    dropped = 0
    for v in sorted(scores):
        lab = labels[v]
        if not 0 < int(lab.sum()) < len(lab):
            continue
        if len(scores[v]) != len(prior[v]):
            raise ValueError(f"{v}: {len(scores[v])} scores vs {len(prior[v])} prior frames")
        if np.ptp(scores[v]) == 0.0 or np.ptp(prior[v]) == 0.0:
            dropped += 1
            continue
        out[v] = spearman(scores[v], prior[v])
    return out, dropped


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"E3 input missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pb_dir(e3_dir: Path, arm: str, split: str) -> Path:
    return e3_dir / REPORTS / PB_DIR / arm / split


def load_arm(e3_dir: Path, arm: str, seeds: list[str]) -> dict[str, Any]:
    """Diags per seed and protocol-B read-outs per split; every step must agree (M1, M3)."""
    diags = {s: _json(e3_dir / arm / s / DIAG_DIR / DIAG_JSON) for s in seeds}
    pb: dict[str, Any] = {}
    for split in SPLITS_OF[arm]:
        d = pb_dir(e3_dir, arm, split)
        readout = _json(d / PB_READOUT_JSON)
        if sorted(readout["runs"]) != seeds:
            raise ValueError(f"{arm}/{split}: runs {readout['runs']} are not the seeds {seeds}")
        for s in seeds:
            if readout["global_steps"][s] != diags[s]["global_step"]:
                raise ValueError(
                    f"{arm} {s}: protocol B read step {readout['global_steps'][s]}, "
                    f"diagnostics step {diags[s]['global_step']} -- not the same checkpoint"
                )
        scores, labels = read_clip_scores(d / CLIP_SCORES_NPZ)
        pb[split] = {
            "readout": readout,
            "clip_aucs": _json(d / CLIP_AUCS_JSON),
            "scores": scores,
            "labels": labels,
        }
    return {"diags": diags, "pb": pb}


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------
def guardrails(arm: str, diags: dict[str, Any]) -> dict[str, Any]:
    """M2: A1-A3 hold iff O1' passes on every seed; A0 is printed under O1."""
    if arm == A0:
        per_seed = {s: bool(d["guardrails"]["pass"]) for s, d in diags.items()}
        return {"rule": "O1 (printed)", "per_seed": per_seed, "hold": all(per_seed.values())}
    missing = [s for s, d in diags.items() if "o1_prime" not in d["guardrails"]]
    if missing:
        raise ValueError(
            f"{arm}: no O1' in diags of {missing} -- run with --a0-diag of the same seed"
        )
    per_seed = {s: bool(d["guardrails"]["o1_prime"]["pass"]) for s, d in diags.items()}
    return {"rule": "O1' every seed (M2)", "per_seed": per_seed, "hold": all(per_seed.values()),
            "o1_registered": {s: bool(d["guardrails"]["pass"]) for s, d in diags.items()}}


def free_rule(
    contrast: dict[str, Any], t2_delta: float, bins: dict[str, dict[str, float] | None]
) -> dict[str, Any]:
    """M5: A1 is adoptable iff mean Δ ≥ 0 on DoTA-dev and T2-val, with no reversed share bin."""
    reversed_bins = [b for b, c in bins.items() if c is not None and c["high"] < 0.0]
    checks = {
        "dota_dev_mean_ge_0": contrast["mean"] >= 0.0,
        "t2_val_macro_mean_ge_0": t2_delta >= 0.0,
        "no_reversed_share_bin": not reversed_bins,
    }
    return {"checks": checks, "reversed_bins": reversed_bins, "pass": all(checks.values())}


def decide(
    contrasts: dict[str, dict[str, Any]],
    guards: dict[str, dict[str, Any]],
    free: dict[str, Any],
) -> dict[str, Any]:
    """§10.3 in order: A3, A2 (each vs A0 and vs ``F``, guardrails holding), A1, A0."""
    f_arm = A1 if free["pass"] else A0
    for arm in (A3, A2):
        vs_a0 = contrasts[f"{arm} - A0"]
        vs_f = vs_a0 if f_arm == A0 else contrasts[f"{arm} - A1"]
        if positive(vs_a0) and positive(vs_f) and guards[arm]["hold"]:
            return {"adopt": arm, "F": f_arm}
    if free["pass"]:
        return {"adopt": A1, "F": f_arm}
    return {"adopt": A0, "F": f_arm}


def claims(adopt: str, contrasts: dict[str, dict[str, Any]], mde: float) -> list[str]:
    """M8: the sentences the read-out licenses (never "motion", D6 / G3)."""
    out: list[str] = []
    if adopt == A3:
        out.append("v2 as a whole (CRN + V2-S) improves DoTA-CAP (n/1397) over A0.")
        out.append(
            "The video stream (V2-S) improves DoTA-CAP (n/1397) on top of CRN (A3 - A1)."
            if positive(contrasts["A3 - A1"]) else
            "The video stream's own contribution (A3 - A1) is not detectable at "
            f"MDE_E3 = {mde:.4f}."
        )
    elif adopt == A2:
        out.append("The video stream (V2-S) improves DoTA-CAP (n/1397) (A2 - A0).")
    elif adopt == A1:
        out.append(
            "CRN is adopted as the free arm (non-negative on DoTA-dev and T2-val, no reversed bin)."
        )
    else:
        out.append(f"v2 is a bounded null: no contrast clears the rules (MDE_E3 = {mde:.4f}).")
    out.append("Never write \"motion\" of the stream: D6 found no temporal-order effect (§15 G3).")
    return out


# ---------------------------------------------------------------------------
# read-out
# ---------------------------------------------------------------------------
def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def diag_means(diags: dict[str, Any]) -> dict[str, float | None]:
    def field(path: tuple[str, ...]) -> list[float | None]:
        out: list[float | None] = []
        for d in diags.values():
            node: Any = d
            for key in path:
                node = None if node is None else node.get(key)
            out.append(None if node is None else float(node))
        return out

    return {
        "t2_val_micro": _mean(field(("guardrails", "micro"))),
        "t2_val_macro": _mean(field(("guardrails", "macro"))),
        "window_auc": _mean(field(("guardrails", "window_level_auc"))),
        "clip_oracle": _mean(field(("guardrails", "clip_oracle_micro"))),
        "shortcut_v_t": _mean(field(("source_shortcut_auc", "v_t"))),
        "shortcut_y_bin": _mean(field(("source_shortcut_auc", "y_bin"))),
        "position_r2_v_t": _mean(field(("position_r2", "v_t"))),
        "rho_u_last": _mean(field(("motion_share", "motion_share", "last"))),
    }


def position_reads(
    arms: dict[str, Any],
    prior: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    contrasts: dict[str, dict[str, Any]], seed: int
) -> dict[str, Any]:
    """G6 (c) on ``dota_cap_dev`` + the sentence rule."""
    prior_scores, prior_labels = prior
    per_arm: dict[str, Any] = {}
    rho: dict[str, dict[str, float]] = {}
    for arm, data in arms.items():
        cap = data["pb"][CAP]
        rho[arm], dropped = clip_spearman(cap["scores"], prior_scores, prior_labels)
        per_arm[arm] = {"mean_spearman": _mean(list(rho[arm].values())), "dropped": dropped}
    pairs: dict[str, Any] = {}
    for x, y in G6_PAIRS:
        name = f"{x} - {y}"
        delta = bootstrap(clip_delta(rho[x], rho[y]), seed)
        co_moves = bool(
            positive(contrasts[name]) and delta is not None and delta["low"] > 0.0
        )
        pairs[name] = {"delta_spearman": delta, "co_moves_with_position_prior": co_moves}
    return {"per_arm": per_arm, "pairs": pairs}


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = sorted(seed_name(s) for s in args.seeds)
    arms = {arm: load_arm(args.e3_dir, arm, seeds) for arm in constants.V2_E3_ARMS}
    t95 = constants.V2_E3_T95

    def per_seed_macro(arm: str, split: str) -> dict[str, float]:
        return {s: float(m) for s, m in arms[arm]["pb"][split]["readout"]["per_run_macro"].items()}

    contrasts = {
        name: {**paired_interval(per_seed_macro(x, split), per_seed_macro(y, split), t95),
               "split": split}
        for name, x, y, split in (*MAIN_CONTRASTS, *F_CONTRASTS)
    }
    a3_a2 = paired_interval(per_seed_macro(A3, CAP), per_seed_macro(A2, CAP), t95)
    a1_a0_cap = paired_interval(per_seed_macro(A1, CAP), per_seed_macro(A0, CAP), t95)
    interaction = {
        "mean": a3_a2["mean"] - a1_a0_cap["mean"],
        "note": "(A3 - A2) - (A1 - A0) on dota_cap_dev; descriptive only (§10.3)",
    }
    mde = mde_e3([contrasts[name] for name, *_ in MAIN_CONTRASTS], t95)

    guards = {arm: guardrails(arm, arms[arm]["diags"]) for arm in constants.V2_E3_ARMS}
    t2_macro = {
        arm: {s: float(d["guardrails"]["macro"]) for s, d in arms[arm]["diags"].items()}
        for arm in constants.V2_E3_ARMS
    }
    t2_a1_a0 = float(np.mean([t2_macro[A1][s] - t2_macro[A0][s] for s in seeds]))

    dev_labels = arms[A0]["pb"][DEV]["labels"]
    dev_share = {v: float(np.mean(lab)) for v, lab in dev_labels.items()}
    cap_labels = arms[A0]["pb"][CAP]["labels"]
    cap_share = {v: float(np.mean(lab)) for v, lab in cap_labels.items()}
    a1_bins = by_share_bin(
        clip_delta(arms[A1]["pb"][DEV]["clip_aucs"], arms[A0]["pb"][DEV]["clip_aucs"]),
        dev_share, args.seed,
    )
    free = free_rule(contrasts["A1 - A0"], t2_a1_a0, a1_bins)
    decision = decide(contrasts, guards, free)

    clip_level = {
        name: bootstrap(
            clip_delta(arms[x]["pb"][split]["clip_aucs"], arms[y]["pb"][split]["clip_aucs"]),
            args.seed,
        )
        for name, x, y, split in (*MAIN_CONTRASTS, *F_CONTRASTS)
    }
    share_bins = {
        arm: {split: by_share_bin(arms[arm]["pb"][split]["clip_aucs"],
                                  dev_share if split == DEV else cap_share, args.seed)
              for split in SPLITS_OF[arm]}
        for arm in constants.V2_E3_ARMS
    }
    priors = {
        split: _json(args.e3_dir / REPORTS / PRIOR_DIR / split / PRIOR_READOUT_JSON)
        for split in (DEV, CAP)
    }
    prior_cap = read_clip_scores(args.e3_dir / REPORTS / PRIOR_DIR / CAP / PRIOR_NPZ)
    readout: dict[str, Any] = {
        "addendum": "PREREG_ADDENDUM.md §18 (Amendment 9, M1-M10); proposal §10.1-§10.3",
        "seeds": seeds,
        "t95": t95,
        "contrasts": contrasts,
        "interaction": interaction,
        "mde_e3": mde,
        "guardrails": guards,
        "t2_val_macro_a1_minus_a0": t2_a1_a0,
        "free_rule": {**free, "share_bins_a1_minus_a0": a1_bins},
        "decision": decision,
        "claims": claims(decision["adopt"], contrasts, mde),
        "printed": {
            "clip_level_paired_delta": clip_level,
            "macro": {arm: {split: arms[arm]["pb"][split]["readout"]["macro"]
                            for split in SPLITS_OF[arm]} for arm in constants.V2_E3_ARMS},
            "monotone_ruler": {split: arms[A0]["pb"][split]["readout"]["position_ruler"]
                               for split in (DEV, CAP)},
            "share_bins": share_bins,
            "diag_means": {arm: diag_means(arms[arm]["diags"]) for arm in constants.V2_E3_ARMS},
            "p_t2": {split: priors[split]["p_t2_macro"] for split in (DEV, CAP)},
            "p_cap": constants.V2_E3_P_CAP,
            "g6": position_reads(arms, prior_cap, contrasts, args.seed),
        },
    }
    out = args.e3_dir / REPORTS
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / E3_READOUT_JSON, readout)
    write_text_atomic(out / E3_READOUT_MD, render_markdown(readout))
    LOGGER.info("E3: adopt %s (F = %s) -> %s", decision["adopt"], decision["F"], out)
    return readout


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _ci(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _level(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _num(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def _set_name(split: str) -> str:
    return "DoTA-CAP-dev (n/1397)" if split == CAP else "DoTA-dev"


def render_markdown(r: dict[str, Any]) -> str:
    mde = r["mde_e3"]
    lines = [
        "# KAT-VAD v2 — E3 read-out (2 x 2 x 5 seeds)",
        "",
        f"Rules: {r['addendum']}. Seeds {', '.join(r['seeds'])}; decision interval = paired "
        f"t95 (t = {r['t95']}). **MDE_E3 = {mde:.4f}** (E3-internal, not E0b; M7).",
        "",
        f"## Decision: **adopt {r['decision']['adopt']}** (F = {r['decision']['F']})",
        "",
        *(f"* {c}" for c in r["claims"]),
        "",
        "## Contrasts (decision interval, per-seed macro Δ)",
        "",
        "| contrast | set | mean Δ | t95 | per seed | clip-level Δ (printed) | reading |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, c in r["contrasts"].items():
        seeds = " / ".join(f"{d:+.4f}" for d in c["per_seed"].values())
        reading = ("excludes 0" if excludes_zero(c) else f"not detectable at MDE_E3 = {mde:.4f}")
        lines.append(
            f"| {name} | {_set_name(c['split'])} | {c['mean']:+.4f} | "
            f"[{c['low']:+.4f}, {c['high']:+.4f}] | {seeds} | "
            f"{_ci(r['printed']['clip_level_paired_delta'][name])} | {reading} |"
        )
    lines += [
        "",
        f"Interaction {r['interaction']['note']}: {r['interaction']['mean']:+.4f}.",
        "",
        "## Guardrails (T2-val, per seed)",
        "",
        "| arm | rule | per seed | hold |",
        "|---|---|---|---|",
        *(
            f"| {arm} | {g['rule']} | "
            f"{' '.join('✓' if v else '✗' for v in g['per_seed'].values())} | "
            f"{'yes' if g['hold'] else '**no**'} |"
            for arm, g in r["guardrails"].items()
        ),
        "",
        "## Free rule for A1 (M5)",
        "",
        *(f"* {k}: {'PASS' if v else 'FAIL'}" for k, v in r["free_rule"]["checks"].items()),
        f"* T2-val macro A1 - A0 (seed mean): {r['t2_val_macro_a1_minus_a0']:+.4f}",
        "* DoTA-dev clip-level Δ(A1 - A0) per share bin: " + ", ".join(
            f"{b} {_ci(c)}" for b, c in r["free_rule"]["share_bins_a1_minus_a0"].items()
        ),
        "",
        "## Printed, never decided on (M6)",
        "",
        "| arm | DoTA-dev macro | DoTA-CAP-dev macro | T2-val micro | T2-val macro | window AUC | "
        "shortcut V^t | shortcut y^bin | pos R² V^t | rho_u last | Spearman(y, p_T2) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    p = r["printed"]
    for arm in constants.V2_E3_ARMS:
        m, d = p["macro"][arm], p["diag_means"][arm]
        lines.append(
            f"| {arm} | {_level(m.get(DEV))} | {_level(m.get(CAP))} | {_num(d['t2_val_micro'])} | "
            f"{_num(d['t2_val_macro'])} | {_num(d['window_auc'])} | {_num(d['shortcut_v_t'])} | "
            f"{_num(d['shortcut_y_bin'])} | {_num(d['position_r2_v_t'])} | "
            f"{_num(d['rho_u_last'])} | "
            f"{_num(p['g6']['per_arm'][arm]['mean_spearman'])} |"
        )
    lines += [
        "",
        "DoTA-dev and DoTA-CAP-dev columns are different clip sets (D14): never compare them.",
        "",
        f"Position: `p_T2` DoTA-dev {_level(p['p_t2'][DEV])}, "
        f"DoTA-CAP-dev {_level(p['p_t2'][CAP])}; "
        f"`p_CAP` (in-domain) {p['p_cap']:.3f}; `t/N` (monotone) DoTA-dev "
        f"{_level(p['monotone_ruler'][DEV])}, DoTA-CAP-dev {_level(p['monotone_ruler'][CAP])}.",
        "",
    ]
    for name, g in p["g6"]["pairs"].items():
        note = (" — **this gain co-moves with the position prior**"
                if g["co_moves_with_position_prior"] else "")
        lines.append(f"* G6 Δ Spearman({name}) on DoTA-CAP-dev: {_ci(g['delta_spearman'])}{note}")
    lines += ["", "### Macro per accident-share bin", "",
              "| arm | set | " + " | ".join(constants.V2_SHARE_BIN_LABELS) + " |",
              "|---|---|" + "---|" * len(constants.V2_SHARE_BIN_LABELS)]
    for arm, per_split in p["share_bins"].items():
        for split, bins in per_split.items():
            cells = " | ".join(_level(bins[b]) for b in constants.V2_SHARE_BIN_LABELS)
            lines.append(f"| {arm} | {_set_name(split)} | {cells} |")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--e3-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(constants.V2_E3_SEEDS))
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap seed")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
