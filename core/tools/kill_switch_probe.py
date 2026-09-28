"""Kill-switch K on T2 -- does frozen VideoMAE V2 carry a frame-level signal?

Pre-registered in ``core/docs/v2/PREREG_ADDENDUM.md`` §6.1 (Amendment 1, D7):
DoTA has no pixels left, so K runs on T2 alone. Two subcommands:

``pick``
    ~``V2_K_SOURCES`` T2-**train** sources, none from T2-val, drawn per accident
    ``type`` with :func:`core.tools.subset_train.draw_subset` (the same drawer and
    pool as the frozen T2-val). Writes the id list the extractors take and a
    manifest with its sha1. Reads no score.

``run``
    Frame linear probe (the ``core.eda`` logistic probe) on **whole source
    videos** at stride 8 -- labels from the annotation span, Gate D0's convention
    -- for three feature sets: CLIP ``x``, VideoMAE ``u`` and ``[x ; u]``:

    * (i') source-grouped 5-fold CV (in-domain);
    * (ii') type-grouped 5-fold CV: test folds hold accident types absent from
      their train fold. **Not a domain transfer** -- the stricter of the two
      generalization reads left without DoTA pixels.

    Metric = macro, the mean per-source frame AUC over two-class sources. Every
    Δ is paired per source; CIs are percentile bootstraps over **sources**.
    Verdict = the pre-registered rule, applied mechanically:
    KILL iff both Δ(``xu`` - ``x``) have upper < ``V2_K_KILL_UPPER`` (amended rule);
    the positive control (``u`` under (i'), CI lower > 0.5) failing overrides KILL
    with ``SUSPECT_PIPELINE``; otherwise GO.

CLI::

    python -m core.tools.kill_switch_probe pick --t2-meta <T2>/meta.json \\
        --out-dir outputs/REPORTS/v2_K
    python -m core.tools.kill_switch_probe run --ids-file outputs/REPORTS/v2_K/k_sources.txt \\
        --annotation <DADA>/dada标注.xlsx --census '<counts>/*.json' \\
        --clip-dir cache/clip/DADA2000_orig \\
        --video-dir cache/video/vit_b_k710_dl_from_giant/DADA2000_orig_s8_squash \\
        --out-dir outputs/REPORTS/v2_K --commit <sha>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.dada import sampled_frame_labels
from core.data.dada_origin import load_counts, make_record, parse_annotation
from core.data.v2_splits import lines_sha1, load_split
from core.eda.features import _probe_fit_predict
from core.metrics import frame_auc
from core.tools.feature_cache import read_ids_file
from core.tools.freeze_splits import t2_pool_from_meta
from core.tools.subset_train import draw_subset, validate_subset

LOGGER = logging.getLogger(__name__)

FEATURE_SETS = ("x", "u", "xu")
PROBES = {"i_source": "source-grouped CV (in-domain)",
          "ii_type": "type-grouped CV (held-out accident types; NOT a domain transfer)"}
CONTROL_PROBE = "i_source"
DELTA = ("xu", "x")
SOURCES_FILENAME = "k_sources.txt"
PICK_MANIFEST_FILENAME = "k_sources_manifest.json"
READOUT_JSON = "k_readout.json"
READOUT_MD = "k_readout.md"
VERDICT_KILL = "KILL"
VERDICT_GO = "GO"
VERDICT_SUSPECT = "SUSPECT_PIPELINE"
PERCENT = 100.0


# ---------------------------------------------------------------------------
# pick
# ---------------------------------------------------------------------------
def pick_sources(
    meta: dict[str, dict[str, Any]],
    val_sources: set[str],
    seed: int = constants.V2_SPLIT_SEED,
    target: int = constants.V2_K_SOURCES,
) -> tuple[list[str], dict[str, Any]]:
    """K's T2-train sources (T2-val excluded) and their manifest block."""
    pool = t2_pool_from_meta(meta)
    total = len(pool.windows_by_source)
    allowed = set(pool.windows_by_source) - val_sources
    fraction = target / total
    plan = draw_subset(pool, fraction, seed, allowed)
    validate_subset(pool, plan, allowed)
    leaked = plan.kept_sources & val_sources
    if leaked:
        raise ValueError(f"{len(leaked)} K sources are T2-val sources: {sorted(leaked)[:5]}")
    ids = sorted(plan.kept_sources)
    return ids, {
        "target": target,
        "fraction_of_train_sources": fraction,
        "train_sources_total": total,
        "val_sources_excluded": len(val_sources),
        "picked": len(ids),
        "stratum": "type",
        "seed": seed,
        "sha1": lines_sha1(ids),
        "kept_by_type": plan.kept_by_type,
        "types_without_k_source": sorted(plan.dropped_types, key=int),
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def source_labels(
    annotation: Path, census: dict[str, int], ids: list[str], stride: int
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Per-source stride-``stride`` frame labels and accident type (Gate D0 rule)."""
    rows = {row.video_id: row for row in parse_annotation(annotation)}
    missing = sorted(v for v in ids if v not in rows or v not in census)
    if missing:
        raise ValueError(
            f"{len(missing)} K sources lack an annotation row or census: {missing[:5]}"
        )
    labels: dict[str, np.ndarray] = {}
    types: dict[str, int] = {}
    for video_id in ids:
        record = make_record(rows[video_id], census[video_id])
        if record is None:
            raise ValueError(f"{video_id}: anomaly starts past the on-disk frames")
        labels[video_id] = np.asarray(sampled_frame_labels(record, stride), dtype=np.int64)
        types[video_id] = rows[video_id].type_id
    return labels, types


def load_features(cache_dir: Path, ids: list[str]) -> dict[str, np.ndarray]:
    """``{id: (L, D)}`` from a per-video ``.npy`` cache; a missing file raises."""
    missing = [v for v in ids if not (cache_dir / f"{v}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ids have no feature in {cache_dir}: {missing[:5]}")
    return {v: np.load(cache_dir / f"{v}.npy").astype(np.float64) for v in ids}


def check_alignment(
    labels: dict[str, np.ndarray], x: dict[str, np.ndarray], u: dict[str, np.ndarray]
) -> None:
    """Every source must give one label, one CLIP row and one ``u`` row per step."""
    bad = [
        f"{v}: labels {len(labels[v])} x {len(x[v])} u {len(u[v])}"
        for v in sorted(labels)
        if not len(labels[v]) == len(x[v]) == len(u[v])
    ]
    if bad:
        raise ValueError(f"{len(bad)} sources are misaligned: {bad[:5]}")


def feature_sets(
    x: dict[str, np.ndarray], u: dict[str, np.ndarray]
) -> dict[str, dict[str, np.ndarray]]:
    """The three probed representations, keyed as in ``FEATURE_SETS``."""
    return {"x": x, "u": u, "xu": {v: np.concatenate([x[v], u[v]], axis=1) for v in x}}


def per_source_auc(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    group_of: dict[str, int],
    ids: list[str],
    folds: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold per-source frame AUC, in ``ids`` order (all two-class)."""
    matrix = np.concatenate([features[v] for v in ids], axis=0)
    target = np.concatenate([labels[v] for v in ids])
    groups = np.concatenate([np.full(len(labels[v]), group_of[v]) for v in ids])
    usable = min(folds, len(np.unique(groups)))
    scores = _probe_fit_predict(matrix, target, groups, usable, seed)
    aucs: list[float] = []
    offset = 0
    for video_id in ids:
        length = len(labels[video_id])
        aucs.append(frame_auc(scores[offset:offset + length], target[offset:offset + length]))
        offset += length
    return np.asarray(aucs)


def bootstrap_ci(
    values: np.ndarray, resamples: int, seed: int, level: float = constants.V2_K_CI
) -> dict[str, float]:
    """Mean and percentile CI of ``values`` resampled over sources."""
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    tail = (1.0 - level) / 2.0 * PERCENT
    low, high = np.percentile(draws, [tail, PERCENT - tail])
    return {"mean": float(values.mean()), "low": float(low), "high": float(high)}


def decide(deltas: dict[str, dict[str, float]], control: dict[str, float]) -> str:
    """The addendum §6.1 rule (as amended 2026-09-28, option A), applied mechanically.

    KILL on the upper bound alone: "no gain >= V2_K_KILL_UPPER is compatible with
    the data". The P0 rule also required a point estimate <= 0, which made a
    redundant ``u`` (Δ ~ ±1e-4) a sign coin-flip (pending lesson (ae)).
    """
    if control["low"] <= constants.V2_K_CONTROL_FLOOR:
        return VERDICT_SUSPECT
    kill = all(d["high"] < constants.V2_K_KILL_UPPER for d in deltas.values())
    return VERDICT_KILL if kill else VERDICT_GO


def run_k(
    labels: dict[str, np.ndarray],
    types: dict[str, int],
    x: dict[str, np.ndarray],
    u: dict[str, np.ndarray],
    folds: int = constants.EDA_PROBE_FOLDS,
    seed: int = constants.V2_SPLIT_SEED,
    resamples: int = constants.V2_K_BOOTSTRAP,
) -> dict[str, Any]:
    """All probes, intervals and the verdict (no I/O)."""
    check_alignment(labels, x, u)
    ids = sorted(v for v in labels if 0 < int(labels[v].sum()) < len(labels[v]))
    if len(ids) < constants.EDA_PROBE_MIN_CLIPS:
        raise ValueError(f"only {len(ids)} two-class sources; K needs a real subset")
    groupings = {
        "i_source": {v: i for i, v in enumerate(ids)},
        "ii_type": {v: types[v] for v in ids},
    }
    sets = feature_sets(x, u)
    arms: dict[str, dict[str, Any]] = {}
    deltas: dict[str, dict[str, float]] = {}
    for probe, group_of in groupings.items():
        aucs = {
            name: per_source_auc(sets[name], labels, group_of, ids, folds, seed)
            for name in FEATURE_SETS
        }
        arms[probe] = {name: bootstrap_ci(a, resamples, seed) for name, a in aucs.items()}
        deltas[probe] = bootstrap_ci(aucs[DELTA[0]] - aucs[DELTA[1]], resamples, seed)
    verdict = decide(deltas, arms[CONTROL_PROBE]["u"])
    return {
        "sources_two_class": len(ids),
        "sources_dropped_single_class": len(labels) - len(ids),
        "frames": int(sum(len(labels[v]) for v in ids)),
        "dim_x": int(next(iter(x.values())).shape[1]),
        "dim_u": int(next(iter(u.values())).shape[1]),
        "folds": folds,
        "seed": seed,
        "bootstrap": resamples,
        "ci": constants.V2_K_CI,
        "probes": PROBES,
        "arms": arms,
        "delta_xu_minus_x": deltas,
        "rule": {
            "kill_upper": constants.V2_K_KILL_UPPER,
            "control_floor": constants.V2_K_CONTROL_FLOOR,
            "control": f"u under {CONTROL_PROBE}",
        },
        "gate_d0_macro_reference": constants.V2_K_GATE_D0_MACRO,
        "verdict": verdict,
    }


def _fmt(ci: dict[str, float]) -> str:
    return f"{ci['mean']:+.4f} [{ci['low']:+.4f}, {ci['high']:+.4f}]"


def render_markdown(readout: dict[str, Any]) -> str:
    """Human read-out; the JSON beside it is the record."""
    lines = [
        "# Kill-switch K (T2 only) — read-out",
        "",
        f"Pre-registration: `core/docs/v2/PREREG_ADDENDUM.md` §6.1 · commit `{readout['commit']}`",
        f"Sources: {readout['sources_two_class']} two-class ({readout['frames']} frames), "
        f"ids sha1 `{readout['ids_sha1']}` · dims x {readout['dim_x']} / u {readout['dim_u']}",
        f"Encoder cache manifest: `{json.dumps(readout['video_manifest'], sort_keys=True)}`",
        "",
        "| probe | x | u | [x;u] | Δ([x;u] - x) |",
        "|---|---|---|---|---|",
    ]
    for probe, title in PROBES.items():
        arm = readout["arms"][probe]
        lines.append(
            f"| {title} | {_fmt(arm['x'])} | {_fmt(arm['u'])} | {_fmt(arm['xu'])} | "
            f"{_fmt(readout['delta_xu_minus_x'][probe])} |"
        )
    lines += [
        "",
        f"Macro = mean per-source frame AUC; 95 % percentile bootstrap over sources "
        f"(B = {readout['bootstrap']}).",
        f"x-only (i') beside Gate D0 {readout['gate_d0_macro_reference']:.4f} "
        "(different subset, same label convention) — printed, not gated.",
        "",
        f"## Verdict: **{readout['verdict']}**",
        "",
        f"KILL iff both Δ have upper < +{readout['rule']['kill_upper']}; "
        f"positive control = {readout['rule']['control']} with CI lower > "
        f"{readout['rule']['control_floor']} (fails → SUSPECT_PIPELINE, no KILL).",
    ]
    return "\n".join(lines) + "\n"


def write_json_atomic(target: Path, payload: object) -> None:
    """Stage then rename (C11c)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + constants.CACHE_PART_SUFFIX)
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staged.replace(target)


def write_text_atomic(target: Path, text: str) -> None:
    """Stage then rename (C11c)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + constants.CACHE_PART_SUFFIX)
    staged.write_text(text, encoding="utf-8")
    staged.replace(target)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("pick", help="draw the K source list (no scores read)")
    pick.add_argument("--t2-meta", type=Path, required=True)
    pick.add_argument("--out-dir", type=Path, required=True)
    pick.add_argument("--seed", type=int, default=constants.V2_SPLIT_SEED)
    pick.add_argument("--target", type=int, default=constants.V2_K_SOURCES)
    run = sub.add_parser("run", help="probe and apply the pre-registered rule")
    run.add_argument("--ids-file", type=Path, required=True)
    run.add_argument("--annotation", type=Path, required=True)
    run.add_argument("--census", nargs="+", required=True,
                     help="census JSON globs ({video_id: frames on disk})")
    run.add_argument("--clip-dir", type=Path, required=True)
    run.add_argument("--video-dir", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    run.add_argument("--commit", required=True, help="git commit of the tree that ran K")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "pick":
        meta = json.loads(args.t2_meta.read_text(encoding="utf-8"))
        val = set(load_split(constants.V2_SPLIT_T2_VAL))
        ids, block = pick_sources(meta, val, args.seed, args.target)
        write_text_atomic(args.out_dir / SOURCES_FILENAME, "\n".join(ids) + "\n")
        write_json_atomic(args.out_dir / PICK_MANIFEST_FILENAME, block)
        LOGGER.info("Picked %d K sources (sha1 %s) -> %s", len(ids), block["sha1"], args.out_dir)
        return
    ids = sorted(read_ids_file(args.ids_file))
    manifest_path = args.video_dir / constants.VIDEO_MANIFEST_FILENAME
    video_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # before compute
    labels, types = source_labels(args.annotation, load_counts(args.census), ids, args.stride)
    readout = run_k(
        labels, types, load_features(args.clip_dir, ids), load_features(args.video_dir, ids)
    )
    readout.update({
        "commit": args.commit,
        "ids_sha1": lines_sha1(ids),
        "ids_requested": len(ids),
        "clip_dir": str(args.clip_dir),
        "video_dir": str(args.video_dir),
        "video_manifest": video_manifest,
    })
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(readout))
    LOGGER.info("K verdict: %s -> %s", readout["verdict"], args.out_dir)


if __name__ == "__main__":
    main()
