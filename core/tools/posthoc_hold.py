"""H2(a): does a causal post-hoc hold on ``y^bin`` recover long-span accidents? (exploratory).

**Not an amendment.** The rule is pre-registered in ``.project/plans/katvad-v2-h2a-hold.md``
(fixed 2026-10-09, before the first number); it decides only whether H2(a) is kept as a free
post-processing step, never anything about E3 or Final.

Pure aggregation -- no model is loaded. It reads E3's DoTA-CAP-dev outputs, laid out as
:mod:`core.tools.final_readout` expects, and the T2 window ``meta.json`` (hold length only).
DoTA-CAP-eval and DoTA-eval are never read.

* **G0 (hard):** :func:`core.tools.final_readout.dev_gate` must pass on the same ``--e3-dir``.
* **Hold length** ``w`` (plan §3): median annotated abnormal span of the T2-train sources (minus
  T2-val), in seconds, times ``DOTA_FPS``. No DoTA number enters it.
* **Variants** (plan §4): ``max_hold`` (decides) = causal max over the last ``w`` frames;
  ``ema`` (printed) = span-``w`` EMA, ``alpha = 2 / (w + 1)``. Both see only the past.
* **Rule** (plan §5, A3 only, held vs raw, per-clip Δ AUC, cluster bootstrap over source videos):
  ``GO`` iff G1 (share >= 0.5 clips) CI low > 0, G2 (``f_2 = z(score) + 2 z(p_T2)``) CI low > 0
  and G3 (all two-class clips) mean >= 0; else ``KILL``.
* **Printed, never decided on:** every share bin, ``ema``, arms A0-A2, ``p_T2``, tail clips (a
  normal tail of >= ``w`` frames after the last abnormal frame).

Known limit: E3 keeps seed-averaged curves only, so the hold is applied to the seed mean.

CLI::

    python -m core.tools.posthoc_hold --e3-dir outputs/v2/v2_s3 \\
        --t2-meta outputs/EDA/DADA2000_orig_T2_w20s8/meta.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.v2_splits import load_split, share_bin
from core.tools.e3_readout import REPORTS, bootstrap, by_share_bin, clip_delta
from core.tools.final_readout import CAP_DEV, dev_gate, fused_clip_aucs, load_pb, load_prior
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import clip_aucs

LOGGER = logging.getLogger(__name__)

OUT_DIR = "h2a_hold"
READOUT_JSON = "h2a_readout.json"
READOUT_MD = "h2a_readout.md"
PLAN = ".project/plans/katvad-v2-h2a-hold.md"


# ---------------------------------------------------------------------------
# hold length (plan §3)
# ---------------------------------------------------------------------------
def hold_length(
    meta: dict[str, dict[str, Any]], val_sources: set[str]
) -> dict[str, Any]:
    """``w`` in DoTA native frames from the T2-train sources' median abnormal span."""
    spans: dict[str, float] = {}
    for window in meta.values():
        source = str(window["source"])
        if window["split"] != constants.H2A_T2_SPLIT or source in val_sources:
            continue
        start, end = window["annotation_span_frames"]
        seconds = (int(end) - int(start)) / constants.DADA_ASSUMED_FPS
        if spans.setdefault(source, seconds) != seconds:
            raise ValueError(f"{source}: windows disagree on the annotated span")
    if not spans:
        raise ValueError("no T2-train source left after removing T2-val")
    median_s = float(np.median(list(spans.values())))
    return {
        "frames": max(1, round(median_s * constants.DOTA_FPS)),
        "median_s": median_s,
        "sources": len(spans),
    }


# ---------------------------------------------------------------------------
# the two causal holds (plan §4)
# ---------------------------------------------------------------------------
def max_hold(scores: np.ndarray, w: int) -> np.ndarray:
    """``s'_t = max(s_{t-w+1} .. s_t)`` over the available past."""
    x = np.asarray(scores, dtype=np.float64)
    padded = np.concatenate([np.full(w - 1, -np.inf), x])
    out: np.ndarray = np.lib.stride_tricks.sliding_window_view(padded, w).max(axis=1)
    return out


def ema(scores: np.ndarray, w: int) -> np.ndarray:
    """Span-``w`` EMA: ``s'_t = a s_t + (1 - a) s'_{t-1}``, ``a = 2 / (w + 1)``, ``s'_0 = s_0``."""
    x = np.asarray(scores, dtype=np.float64)
    alpha = 2.0 / (w + 1)
    out = np.empty_like(x)
    acc = x[0] if len(x) else 0.0
    for t, value in enumerate(x):
        acc = alpha * value + (1.0 - alpha) * acc
        out[t] = acc
    return out


HOLDS = {"max_hold": max_hold, "ema": ema}


def held(scores: dict[str, np.ndarray], variant: str, w: int) -> dict[str, np.ndarray]:
    fn = HOLDS[variant]
    return {v: fn(s, w) for v, s in scores.items()}


def tail_clips(labels: dict[str, np.ndarray], w: int) -> set[str]:
    """Two-class clips whose last abnormal frame is followed by >= ``w`` normal frames."""
    out: set[str] = set()
    for v, lab in labels.items():
        lab = np.asarray(lab)
        if not 0 < int(lab.sum()) < len(lab):
            continue
        last = int(np.flatnonzero(lab)[-1])
        if len(lab) - 1 - last >= w:
            out.add(v)
    return out


# ---------------------------------------------------------------------------
# read-out
# ---------------------------------------------------------------------------
def arm_read(
    scores: dict[str, np.ndarray],
    hold: dict[str, np.ndarray],
    prior: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    w: int,
    seed: int,
) -> dict[str, Any]:
    """Raw vs held for one arm: macro, Δ (all / long / f_2 / tail) and every share bin."""
    raw_auc = clip_aucs(scores, labels)
    held_auc = clip_aucs(hold, labels)
    delta = clip_delta(held_auc, raw_auc)
    weight = constants.H2A_FUSION_WEIGHT
    f2 = clip_delta(
        fused_clip_aucs(hold, prior, labels, weight), fused_clip_aucs(scores, prior, labels, weight)
    )
    share = {v: float(np.mean(labels[v])) for v in delta}
    long_ = {v: d for v, d in delta.items() if share_bin(share[v]) in constants.H2A_LONG_SHARE_BINS}
    tails = tail_clips(labels, w)
    return {
        "macro_raw": float(np.mean(list(raw_auc.values()))),
        "macro_held": float(np.mean(list(held_auc.values()))),
        "delta_all": bootstrap(delta, seed),
        "delta_long": bootstrap(long_, seed),
        "delta_f2": bootstrap(f2, seed),
        "delta_tail": bootstrap({v: d for v, d in delta.items() if v in tails}, seed),
        "bins": by_share_bin(delta, share, seed),
        "n_long": len(long_),
        "n_tail": len(tails & set(delta)),
    }


def decide(read: dict[str, Any]) -> dict[str, Any]:
    """Plan §5 on A3 / ``max_hold``."""
    def low_above_zero(c: dict[str, float] | None) -> bool:
        return c is not None and c["low"] > 0.0

    g1 = low_above_zero(read["delta_long"])
    g2 = low_above_zero(read["delta_f2"])
    g3 = read["delta_all"] is not None and read["delta_all"]["mean"] >= 0.0
    return {
        "verdict": "GO" if g1 and g2 and g3 else "KILL",
        "g1_long_ci_low_gt_0": g1,
        "g2_f2_ci_low_gt_0": g2,
        "g3_all_mean_ge_0": g3,
        "arm": constants.H2A_ARM,
        "variant": constants.H2A_PRIMARY,
    }


def readout(
    scores: dict[str, dict[str, np.ndarray]],
    prior: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    w: int,
    seed: int,
) -> dict[str, Any]:
    variants = {
        variant: {
            arm: arm_read(s, held(s, variant, w), prior, labels, w, seed)
            for arm, s in scores.items()
        }
        for variant in constants.H2A_VARIANTS
    }
    prior_auc = clip_aucs(prior, labels)
    return {
        "p_t2_macro": float(np.mean(list(prior_auc.values()))),
        "two_class": len(prior_auc),
        "variants": variants,
        "decision": decide(variants[constants.H2A_PRIMARY][constants.H2A_ARM]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    gate = dev_gate(args.e3_dir, args.seed)
    if not gate["pass"]:
        moved = len(gate["failed"])
        raise SystemExit(f"G0 FAILS: {moved} recorded E3 dev means moved; read nothing")
    meta = json.loads(args.t2_meta.read_text(encoding="utf-8"))
    w = hold_length(meta, set(load_split(constants.V2_SPLIT_T2_VAL, args.split_dir)))
    pbs = {arm: load_pb(args.e3_dir, arm, CAP_DEV) for arm in constants.V2_E3_ARMS}
    labels = pbs[constants.H2A_ARM]["labels"]
    for arm, pb in pbs.items():
        if set(pb["labels"]) != set(labels):
            raise ValueError(f"{arm}: clips differ from {constants.H2A_ARM}'s")
    _, prior = load_prior(args.e3_dir, CAP_DEV)
    result = readout({arm: pb["scores"] for arm, pb in pbs.items()}, prior, labels,
                     w["frames"], args.seed)
    result.update({
        "status": "exploratory (not an amendment); rule pre-registered before the first number",
        "plan": PLAN,
        "e3_dir": str(args.e3_dir),
        "t2_meta": str(args.t2_meta),
        "set": "DoTA-CAP (n/1397) dev",
        "g0_dev_gate": gate["pass"],
        "hold": w,
        "seed": args.seed,
    })
    out = args.out_dir or args.e3_dir / REPORTS / OUT_DIR
    write_json_atomic(out / READOUT_JSON, result)
    write_text_atomic(out / READOUT_MD, render_markdown(result))
    LOGGER.info("H2(a): %s (w = %d frames) -> %s", result["decision"]["verdict"], w["frames"], out)
    return result


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _ci(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def render_markdown(r: dict[str, Any]) -> str:
    d, w = r["decision"], r["hold"]
    lines = [
        "# H2(a) — causal post-hoc hold on y^bin (exploratory)",
        "",
        f"Plan `{r['plan']}` (rule fixed before this number). Set: **{r['set']}**, "
        f"{r['two_class']} two-class clips, E3 seed-averaged scores. G0 dev gate: "
        f"{'PASS' if r['g0_dev_gate'] else 'FAIL'}.",
        f"Hold length w = **{w['frames']}** DoTA frames (median T2-train span "
        f"{w['median_s']:.2f} s over {w['sources']} sources, x {constants.DOTA_FPS} fps).",
        "",
        f"## Verdict: **{d['verdict']}** ({d['arm']}, {d['variant']})",
        "",
        f"* G1 long-span (share ≥ 0.5) CI low > 0: {d['g1_long_ci_low_gt_0']}",
        f"* G2 beyond position (f_{constants.H2A_FUSION_WEIGHT:g}) CI low > 0: "
        f"{d['g2_f2_ci_low_gt_0']}",
        f"* G3 all-clip mean ≥ 0: {d['g3_all_mean_ge_0']}",
        "",
        f"`p_T2` macro (printed): {r['p_t2_macro']:.4f}",
    ]
    for variant, arms in r["variants"].items():
        tag = "decides (A3)" if variant == constants.H2A_PRIMARY else "printed only"
        lines += [
            "",
            f"## {variant} — {tag}",
            "",
            "| arm | macro raw | macro held | Δ all | Δ long | Δ f_2 | Δ tail (n) |",
            "|---|---|---|---|---|---|---|",
        ]
        for arm, a in arms.items():
            lines.append(
                f"| {arm} | {a['macro_raw']:.4f} | {a['macro_held']:.4f} | {_ci(a['delta_all'])} | "
                f"{_ci(a['delta_long'])} | {_ci(a['delta_f2'])} | "
                f"{_ci(a['delta_tail'])} ({a['n_tail']}) |"
            )
        lines += ["", "| arm | " + " | ".join(constants.V2_SHARE_BIN_LABELS) + " |",
                  "|---|" + "---|" * len(constants.V2_SHARE_BIN_LABELS)]
        for arm, a in arms.items():
            lines.append(f"| {arm} | " + " | ".join(
                _ci(a["bins"][b]) for b in constants.V2_SHARE_BIN_LABELS) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--e3-dir", type=Path, required=True, help="E3's dir (dev reads only)")
    parser.add_argument("--t2-meta", type=Path, required=True,
                        help="T2 window meta.json (DADA2000_orig_T2_w20s8), for w only")
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"default <e3-dir>/{REPORTS}/{OUT_DIR}")
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap seed")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
