"""v2 TAD zero-shot generalization probe (addendum §26, fixed before any v2 arm is scored on TAD).

Pure aggregation -- no model is loaded. Every run is a ``core.evaluate --save-scores`` output on
TAD's 100-video test protocol at stride 8 (Gate T0's protocol, T2), laid out by
``colab/v2/tad_zeroshot.ipynb``::

    <root>/<A0|A3|A3ign>/s<seed>/{results.json, scores/<video>.npz}
    <root>/T0/{results.json, scores/<video>.npz}          LaGoVAD's released checkpoint

The sentence, mechanically (§26 T5): "v2's components transfer to fixed-camera TAD beyond CLIP"
iff the paired t95 of the per-seed macro Δ (A3-ign - A0) **and** the cluster bootstrap of the
per-clip Δ on seed-averaged scores both lie above 0; "they cost on TAD" iff both lie below 0;
otherwise "no transfer gain is shown at n = 5". The LaGoVAD sentence (T6) reads the per-clip
bootstrap of A3-ign - T0 only. Nothing here moves ``B``. Micro AUC is printed beside the clip
oracle and the length ruler and is never put next to the published 89.56 (C8b, C14).

CLI::

    python -m core.tools.tadz_readout --root outputs/v2_tadz [--t2-v2-dir data/DADA2000_orig_v2]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools.e3_readout import REPORTS, clip_delta, excludes_zero, paired_interval, seed_name
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.position_prior import prior_scores, training_frames

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "tadz_readout.json"
READOUT_MD = "tadz_readout.md"
SCORES_DIRNAME = "scores"
RESULTS_FILENAME = "results.json"

Curves = dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_run(run_dir: Path) -> tuple[Curves, Curves]:
    """``({video: score}, {video: gt})`` from one ``core.evaluate --save-scores`` output."""
    files = sorted((run_dir / SCORES_DIRNAME).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no saved scores in {run_dir / SCORES_DIRNAME}")
    scores: Curves = {}
    labels: Curves = {}
    for path in files:
        with np.load(path) as data:
            scores[path.stem] = np.asarray(data["score"], dtype=np.float64)
            labels[path.stem] = np.asarray(data["gt"], dtype=np.int64)
    return scores, labels


def check_labels(runs: dict[str, tuple[Curves, Curves]]) -> Curves:
    """T9(d): every run scored the same videos with identical labels; T9(b) the protocol size."""
    names = sorted(runs)
    reference = runs[names[0]][1]
    for name in names[1:]:
        labels = runs[name][1]
        if sorted(labels) != sorted(reference):
            raise ValueError(f"{name} scored other videos than {names[0]}")
        bad = [v for v in reference if not np.array_equal(labels[v], reference[v])]
        if bad:
            raise ValueError(f"{name}: labels differ from {names[0]} on {bad[:5]}")
        short = [v for v in reference if len(runs[name][0][v]) != len(reference[v])]
        if short:
            raise ValueError(f"{name}: score length != label length on {short[:5]}")
    abnormal = sum(int(g.max() > 0) for g in reference.values())
    if (len(reference), abnormal) != (constants.V2_TADZ_VIDEOS, constants.V2_TADZ_ABNORMAL):
        raise ValueError(
            f"TAD protocol is {constants.V2_TADZ_VIDEOS} videos / {constants.V2_TADZ_ABNORMAL} "
            f"abnormal, got {len(reference)} / {abnormal}"
        )
    return reference


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def clip_aucs(scores: Curves, labels: Curves) -> dict[str, float]:
    """Per-video frame AUC over the two-class videos (the macro's items)."""
    return {
        v: frame_auc(scores[v], labels[v])
        for v in sorted(labels)
        if 0 < int(labels[v].sum()) < len(labels[v])
    }


def micro(scores: Curves, labels: Curves) -> float:
    ids = sorted(labels)
    return frame_auc(np.concatenate([scores[v] for v in ids]),
                     np.concatenate([labels[v] for v in ids]))


def video_auc(scores: Curves, labels: Curves) -> float:
    """Clip-level AUC: a video's max score vs its video label (C14's other half)."""
    ids = sorted(labels)
    return frame_auc(np.array([scores[v].max() for v in ids]),
                     np.array([int(labels[v].max() > 0) for v in ids]))


def seed_average(per_seed: list[Curves]) -> Curves:
    return {v: np.mean([s[v] for s in per_seed], axis=0) for v in per_seed[0]}


def bootstrap(values: dict[str, float], seed: int) -> dict[str, float] | None:
    """TAD videos are distinct sources: each is its own cluster."""
    return cluster_bootstrap_ci(
        values, {v: v for v in values}, constants.V2_E1_BOOTSTRAP, seed, constants.V2_E1_CI
    )


def rulers(labels: Curves, prior: Curves | None, seed: int) -> dict[str, Any]:
    """What no pixel scores: the clip oracle and length ruler (micro), position (macro)."""
    oracle = {v: np.full(len(g), float(g.max() > 0)) for v, g in labels.items()}
    # shorter video -> higher score: TAD_SETUP §8.1's orientation (0.8968 on this protocol)
    length = {v: np.full(len(g), -float(len(g))) for v, g in labels.items()}
    monotone = {v: np.arange(len(g)) / len(g) for v, g in labels.items()}
    middle = {v: -np.abs(np.arange(len(g)) / len(g) - 0.5) for v, g in labels.items()}
    return {
        "clip_oracle_micro": micro(oracle, labels),
        "length_ruler_micro": micro(length, labels),
        "monotone_ruler_macro": bootstrap(clip_aucs(monotone, labels), seed),
        "middle_ruler_macro": bootstrap(clip_aucs(middle, labels), seed),
        "p_t2_macro": None if prior is None else bootstrap(clip_aucs(prior, labels), seed),
    }


def arm_reads(per_seed: dict[str, Curves], labels: Curves, seed: int) -> dict[str, Any]:
    averaged = seed_average([per_seed[s] for s in sorted(per_seed)])
    return {
        "per_seed_macro": {
            s: float(np.mean(list(clip_aucs(c, labels).values()))) for s, c in per_seed.items()
        },
        "per_seed_micro": {s: micro(c, labels) for s, c in per_seed.items()},
        "per_seed_video_auc": {s: video_auc(c, labels) for s, c in per_seed.items()},
        "macro": bootstrap(clip_aucs(averaged, labels), seed),
        "micro": micro(averaged, labels),
        "video_auc": video_auc(averaged, labels),
        "clip_aucs": clip_aucs(averaged, labels),
    }


def contrast(a: dict[str, Any], b: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "paired_t95": paired_interval(a["per_seed_macro"], b["per_seed_macro"],
                                      constants.V2_E3_T95),
        "clip_bootstrap": bootstrap(clip_delta(a["clip_aucs"], b["clip_aucs"]), seed),
    }


def transfer_sentence(c: dict[str, Any]) -> str:
    """T5, mechanical: both intervals above 0, both below 0, or neither."""
    t95, boot = c["paired_t95"], c["clip_bootstrap"]
    if t95["low"] > 0 and boot is not None and boot["low"] > 0:
        return "v2's components transfer to fixed-camera TAD beyond CLIP"
    if t95["high"] < 0 and boot is not None and boot["high"] < 0:
        return "v2's components cost detection on fixed-camera TAD"
    return "no transfer gain of v2's components is shown on TAD at n = 5"


def reference_sentence(boot: dict[str, float] | None) -> str:
    """T6: the per-clip bootstrap of A3-ign - T0 alone (T0 is one checkpoint)."""
    if boot is None:
        return "not computable"
    if boot["low"] > 0:
        return "A3-ign localizes above LaGoVAD's released checkpoint on TAD"
    if boot["high"] < 0:
        return "A3-ign localizes below LaGoVAD's released checkpoint on TAD"
    return "A3-ign and LaGoVAD's released checkpoint are not separated on TAD"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = [seed_name(s) for s in args.seeds]
    runs: dict[str, tuple[Curves, Curves]] = {
        constants.V2_TADZ_REFERENCE: load_run(args.root / constants.V2_TADZ_REFERENCE)
    }
    for arm in constants.V2_TADZ_ARMS:
        for s in seeds:
            runs[f"{arm}/{s}"] = load_run(args.root / arm / s)
    labels = check_labels(runs)

    t0_scores = runs[constants.V2_TADZ_REFERENCE][0]
    t0_micro = micro(t0_scores, labels)
    if abs(t0_micro - constants.V2_TADZ_T0_MICRO) > constants.V2_TADZ_T0_TOLERANCE:
        raise ValueError(
            f"T9(a): T0 re-scored at micro {t0_micro:.4f}, not {constants.V2_TADZ_T0_MICRO} -- "
            "the TAD labels or TAD_ncc cache changed since Gate T0; stop"
        )

    arms = {
        arm: arm_reads({s: runs[f"{arm}/{s}"][0] for s in seeds}, labels, args.seed)
        for arm in constants.V2_TADZ_ARMS
    }
    t0_aucs = clip_aucs(t0_scores, labels)
    ref = {
        "macro": bootstrap(t0_aucs, args.seed),
        "micro": t0_micro,
        "video_auc": video_auc(t0_scores, labels),
    }
    prior = None
    if args.t2_v2_dir is not None:
        matrix, target, _ = training_frames(args.t2_v2_dir)
        prior = prior_scores(matrix, target, {v: len(g) for v, g in labels.items()}, args.seed)

    arm, control, printed = (constants.V2_TADZ_ARM, constants.V2_TADZ_CONTROL,
                             constants.V2_TADZ_PRINTED_ARM)
    endpoint = contrast(arms[arm], arms[control], args.seed)
    vs_t0 = bootstrap(clip_delta(arms[arm]["clip_aucs"], t0_aucs), args.seed)
    readout: dict[str, Any] = {
        "addendum": "PREREG_ADDENDUM.md §26 (Amendment 15, T1-T10)",
        "protocol": f"TAD test, stride {constants.V2_TADZ_STRIDE}, whole video, raw scores",
        "videos": len(labels),
        "two_class_videos": len(t0_aucs),
        "seeds": seeds,
        "arms": {a: {k: v for k, v in r.items() if k != "clip_aucs"} for a, r in arms.items()},
        "reference_t0": ref,
        "rulers": rulers(labels, prior, args.seed),
        "endpoint": endpoint,
        "endpoint_sentence": transfer_sentence(endpoint),
        "printed_contrasts": {
            f"{printed}-{control}": contrast(arms[printed], arms[control], args.seed),
            f"{arm}-{printed}": contrast(arms[arm], arms[printed], args.seed),
        },
        "vs_t0_clip_bootstrap": vs_t0,
        "vs_t0_sentence": reference_sentence(vs_t0),
        "collapse_print": {
            a: {
                "video_auc_minus_a0": arms[a]["video_auc"] - arms[control]["video_auc"],
                "macro_minus_a0": arms[a]["macro"]["mean"] - arms[control]["macro"]["mean"],
            }
            for a in (printed, arm)
        },
    }
    out = args.root / REPORTS
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / READOUT_JSON, readout)
    write_text_atomic(out / READOUT_MD, render_markdown(readout))
    LOGGER.info("TAD endpoint %s -> %s", readout["endpoint_sentence"], out)
    return readout


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _ci(c: dict[str, float] | None) -> str:
    if not c:
        return "—"
    return f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _level(c: dict[str, float] | None) -> str:
    if not c:
        return "—"
    return f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _contrast_row(name: str, c: dict[str, Any]) -> str:
    t = c["paired_t95"]
    mark = " (excludes 0)" if excludes_zero(t) else ""
    return (f"| {name} | {t['mean']:+.4f} [{t['low']:+.4f}, {t['high']:+.4f}]{mark} | "
            f"{_ci(c['clip_bootstrap'])} |")


def render_markdown(r: dict[str, Any]) -> str:
    rl = r["rulers"]
    lines = [
        f"# TAD zero-shot — {r['videos']} videos ({r['two_class_videos']} two-class), "
        f"{r['protocol']}",
        "",
        f"Rule: {r['addendum']}. Seeds {r['seeds']}. `B` = A3-ign is not re-decided here.",
        "",
        f"**Endpoint (T5): {r['endpoint_sentence']}.**",
        "",
        f"**LaGoVAD (T6): {r['vs_t0_sentence']}.**",
        "",
        "## Contrasts (macro)",
        "",
        "| Δ | paired t95 over seeds | per-clip bootstrap (seed-averaged) |",
        "|---|---|---|",
        _contrast_row(f"**{constants.V2_TADZ_ARM} - {constants.V2_TADZ_CONTROL}** (endpoint)",
                      r["endpoint"]),
        *[_contrast_row(f"{k} (printed)", c) for k, c in r["printed_contrasts"].items()],
        f"| {constants.V2_TADZ_ARM} - T0 (printed, T6) | — | {_ci(r['vs_t0_clip_bootstrap'])} |",
        "",
        "## Levels (seed-averaged scores)",
        "",
        "| | macro | clip-level AUC | micro (raw) | per-seed macro |",
        "|---|---|---:|---:|---|",
    ]
    for arm, a in r["arms"].items():
        per_seed = ", ".join(f"{v:.4f}" for _, v in sorted(a["per_seed_macro"].items()))
        lines.append(f"| {arm} | {_level(a['macro'])} | {a['video_auc']:.4f} | "
                     f"{a['micro']:.4f} | {per_seed} |")
    t0 = r["reference_t0"]
    lines += [
        f"| T0 (LaGoVAD released) | {_level(t0['macro'])} | {t0['video_auc']:.4f} | "
        f"{t0['micro']:.4f} | — |",
        "",
        "## Rulers — what no pixel scores (printed beside every number)",
        "",
        f"* clip oracle micro **{rl['clip_oracle_micro']:.4f}** · length ruler micro "
        f"**{rl['length_ruler_micro']:.4f}** — micro on TAD is clip classification (C14); "
        "never put it beside 89.56",
        f"* `p_T2` macro {_level(rl['p_t2_macro'])} · monotone `t/N` "
        f"{_level(rl['monotone_ruler_macro'])} · middle {_level(rl['middle_ruler_macro'])}",
        "",
        "## Collapse print (O1' shape: clip-level up while macro down), vs A0",
        "",
    ]
    for arm, c in r["collapse_print"].items():
        lines.append(f"* {arm}: clip-level AUC {c['video_auc_minus_a0']:+.4f}, "
                     f"macro {c['macro_minus_a0']:+.4f}")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--root", type=Path, required=True, help="the probe's output root")
    parser.add_argument("--t2-v2-dir", type=Path, default=None,
                        help="v2 dataset dir (p_T2's training windows); omit to skip p_T2")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(constants.V2_E3_SEEDS))
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap seed")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
