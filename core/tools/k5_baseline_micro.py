"""K5's between-video baseline: micro AUC under per-video transforms (``KIM_RUN.md`` §21).

On DoTA-CAP-dev and DADA-2000 every video holds both normal and abnormal frames, so the micro AUC
(all frames pooled) is below the macro AUC (mean of per-video AUCs) whenever the *level* of one
video's normal frames sits near the abnormal frames of another. Pure aggregation over saved
``clip_scores.npz`` files (protocol-B DoTA, whole-video DADA); no model is loaded and nothing is
trained. Every transform below is increasing within a video, so the macro cannot move (gate G2);
only the order *between* videos changes.

Rows (all computed on the raw saved curves):

* label-free: ``raw``, per-video ``min-max``, ``z-score``, ``minus median``, ``minus q20``,
  per-video ``rank``;
* oracle (use the ground truth, a ceiling and never a method): ``oracle offset`` (subtract the
  median of the video's normal frames) and ``oracle normal-z`` (z-score on the normal frames).

Also printed: micro split into within-video and cross-video frame pairs, and the spread of the
videos' normal level. The recorded read-out beside each curve file gates the numbers (G1). CLI::

    python -m core.tools.k5_baseline_micro --out-dir outputs/kim_baseline/REPORTS \\
        --set DoTA-CAP-dev/K5 D/clip_scores.npz minmax dota D/protocol_b_readout.json \\
        --set DADA-2000/K5 A/clip_scores.npz raw item A/dada_readout.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from core import constants
from core.data.v2_splits import dota_group
from core.metrics import frame_ap, frame_auc, normalize_scores
from core.tools import tadz_readout as td
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import read_clip_scores
from core.tools.tad_clip_gate import micro_bootstrap_pairs
from core.tools.zs_lagovad_readout import versus_sentence

LOGGER = logging.getLogger(__name__)

Curves = dict[str, np.ndarray]
SEED = constants.V5_BOOTSTRAP_SEED
LABEL = "PRELIMINARY (n = 1 seed)"

RAW, MINMAX, ZSCORE = "raw", "min-max", "z-score"
MINUS_MEDIAN, RANK = "minus median", "rank"
MINUS_LOW = f"minus q{round(100 * constants.BASEMICRO_LOW_QUANTILE)}"
ORACLE_OFFSET, ORACLE_Z = "oracle offset", "oracle normal-z"
LABEL_FREE = (RAW, MINMAX, ZSCORE, MINUS_MEDIAN, MINUS_LOW, RANK)
ORACLE = (ORACLE_OFFSET, ORACLE_Z)
ROW_ORDER = LABEL_FREE + ORACLE
AS_READ_CHOICES = {"raw": RAW, "minmax": MINMAX}
CLUSTER_CHOICES = ("item", "dota")
SHOWN, NOT_SHOWN = "shown", "not shown"


def _normal_frames(scores: np.ndarray, gt: np.ndarray) -> np.ndarray:
    normal = scores[gt == 0]
    if normal.size == 0:
        raise ValueError("an oracle row needs a normal frame in every video")
    return normal


def _oracle_offset(scores: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return scores - float(np.median(_normal_frames(scores, gt)))


def _oracle_z(scores: np.ndarray, gt: np.ndarray) -> np.ndarray:
    normal = _normal_frames(scores, gt)
    return (scores - normal.mean()) / (normal.std() + constants.SCORE_NORM_EPS)


TRANSFORMS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    RAW: lambda s, gt: s,
    MINMAX: lambda s, gt: normalize_scores(s, constants.SCORE_NORM_MINMAX),
    ZSCORE: lambda s, gt: normalize_scores(s, constants.SCORE_NORM_ZSCORE),
    MINUS_MEDIAN: lambda s, gt: s - float(np.median(s)),
    MINUS_LOW: lambda s, gt: s - float(np.quantile(s, constants.BASEMICRO_LOW_QUANTILE)),
    RANK: lambda s, gt: np.asarray(rankdata(s, method="average") / len(s), dtype=np.float64),
    ORACLE_OFFSET: _oracle_offset,
    ORACLE_Z: _oracle_z,
}


def transform_curves(scores: Curves, labels: Curves, name: str) -> Curves:
    """Apply one named per-video transform to every curve."""
    if name not in TRANSFORMS:
        raise ValueError(f"unknown transform {name!r}; choose from {list(TRANSFORMS)}")
    return {v: np.asarray(TRANSFORMS[name](scores[v], labels[v]), dtype=np.float64)
            for v in sorted(scores)}


def two_class(labels: Curves) -> list[str]:
    """Videos holding both classes (the macro's items, and the only ones the oracles can read)."""
    return [v for v in sorted(labels) if 0 < int(labels[v].sum()) < len(labels[v])]


def within_video(scores: Curves, labels: Curves) -> dict[str, float]:
    """Frame pairs inside one video: their number and their pair-weighted AUC.

    A transform that is increasing inside every video leaves this unchanged, so it is computed once.
    """
    ids = sorted(labels)
    n_abn = np.array([int(labels[v].sum()) for v in ids], dtype=np.float64)
    n_norm = np.array([len(labels[v]) for v in ids], dtype=np.float64) - n_abn
    pairs = n_abn * n_norm
    concordant = sum(
        frame_auc(scores[v], labels[v]) * p for v, p in zip(ids, pairs, strict=True) if p > 0)
    total = float(n_abn.sum() * n_norm.sum())
    return {"pairs_within": float(pairs.sum()), "pairs_total": total,
            "concordant_within": float(concordant)}


def cross_video_auc(micro: float, within: dict[str, float]) -> float | None:
    """AUC over frame pairs drawn from two different videos (``None`` if there are none)."""
    cross = within["pairs_total"] - within["pairs_within"]
    if cross <= 0:
        return None
    return (micro * within["pairs_total"] - within["concordant_within"]) / cross


def baseline_spread(scores: Curves, labels: Curves) -> dict[str, float]:
    """Spread of the videos' normal level against the normal-to-abnormal gap, on ``scores``."""
    ids = two_class(labels)
    normal = np.array([np.median(scores[v][labels[v] == 0]) for v in ids])
    abnormal = np.array([np.median(scores[v][labels[v] > 0]) for v in ids])
    gap = float(np.mean(abnormal - normal))
    spread = float(np.std(normal))
    return {"normal_level_std": spread, "mean_gap": gap,
            "ratio": spread / gap if gap > 0 else float("nan")}


def macro(scores: Curves, labels: Curves) -> float:
    return float(np.mean(list(td.clip_aucs(scores, labels).values())))


def recorded_value(readout: dict[str, Any], key: str) -> float | None:
    """A recorded value: a float, or the ``mean`` of a bootstrap dict; ``None`` if absent."""
    value = readout.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    return None if value is None else float(value)


def clusters_of(kind: str, ids: list[str]) -> dict[str, str] | None:
    """Bootstrap clusters: each video alone, or a DoTA clip's source video."""
    if kind == "item":
        return None
    if kind == "dota":
        return {v: dota_group(v) for v in ids}
    raise ValueError(f"cluster must be one of {CLUSTER_CHOICES}, not {kind!r}")


def read_set(label: str, npz: Path, as_read: str, cluster: str, readout: Path) -> dict[str, Any]:
    """Every row of one saved-score set, gated on its recorded read (G1) and on the macro (G2)."""
    if as_read not in AS_READ_CHOICES:
        raise ValueError(f"as-read must be one of {list(AS_READ_CHOICES)}, not {as_read!r}")
    scores, labels = read_clip_scores(npz)
    ids = two_class(labels)
    excluded = len(labels) - len(ids)
    scores = {v: scores[v] for v in ids}
    labels = {v: labels[v] for v in ids}
    recorded = json.loads(readout.read_text(encoding="utf-8"))
    base = AS_READ_CHOICES[as_read]
    curves = {name: transform_curves(scores, labels, name) for name in ROW_ORDER}

    macro_raw = macro(scores, labels)
    want = recorded_value(recorded, "macro")
    if want is None or abs(macro_raw - want) > constants.BASEMICRO_MACRO_TOL:
        raise ValueError(f"{label} G1: macro {macro_raw:.8f} from the scores != {want} recorded "
                         f"in {readout}: not the scores of that read")
    micro_read = td.micro(curves[base], labels)
    want_micro = recorded_value(recorded, "micro")
    if want_micro is not None and abs(micro_read - want_micro) > constants.BASEMICRO_MICRO_TOL:
        raise ValueError(f"{label} G1: as-read ({base}) micro {micro_read:.6f} != {want_micro:.6f} "
                         f"recorded in {readout}")
    for name in ROW_ORDER:
        drift = abs(macro(curves[name], labels) - macro_raw)
        if drift > constants.BASEMICRO_INVARIANCE_TOL:
            raise ValueError(f"{label} G2: macro moved by {drift:.2e} under {name!r}")

    groups = clusters_of(cluster, ids)
    boots = micro_bootstrap_pairs(
        curves, {name: (name, base) for name in ROW_ORDER if name != base}, labels, SEED,
        groups=groups, draws=constants.BASEMICRO_BOOTSTRAP_DRAWS)
    within = within_video(scores, labels)
    pooled_y = np.concatenate([labels[v] for v in ids])
    rows: dict[str, dict[str, Any]] = {}
    for name in ROW_ORDER:
        value = td.micro(curves[name], labels)
        rows[name] = {
            "micro": value,
            "ap": frame_ap(np.concatenate([curves[name][v] for v in ids]), pooled_y),
            "macro": macro(curves[name], labels),
            "cross_video_auc": cross_video_auc(value, within),
            "delta_vs_as_read": boots.get(name),
        }
        if name in LABEL_FREE and name in boots:
            rows[name]["sentence"] = versus_sentence(boots[name])
    ceiling = rows[ORACLE_Z]["micro"] - rows[base]["micro"]
    for name in ROW_ORDER:
        share = None
        if ceiling > constants.BASEMICRO_HEADROOM_MARGIN and name != base:
            share = (rows[name]["micro"] - rows[base]["micro"]) / ceiling
        rows[name]["share_of_headroom"] = share
    headroom = boots[ORACLE_Z]
    return {
        "label": label, "as_read": base, "cluster": cluster, "readout": str(readout),
        "videos": len(ids), "excluded_single_class": excluded,
        "clusters": len(set((groups or {v: v for v in ids}).values())),
        "base_rate": float(pooled_y.mean()),
        "pairs": {"within": within["pairs_within"], "total": within["pairs_total"],
                  "within_auc": within["concordant_within"] / within["pairs_within"]},
        "baseline_spread_as_read": baseline_spread(curves[base], labels),
        "rows": rows,
        "headroom_sentence": (SHOWN if headroom["low"] > constants.BASEMICRO_HEADROOM_MARGIN
                              else NOT_SHOWN),
    }


def _ci(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _share(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def _set_md(entry: dict[str, Any]) -> list[str]:
    spread = entry["baseline_spread_as_read"]
    pairs = entry["pairs"]
    base = entry["as_read"]
    lines = [
        f"## {entry['label']}", "",
        f"{entry['videos']} two-class videos ({entry['excluded_single_class']} single-class left "
        f"out), {entry['clusters']} bootstrap clusters; frame base rate {entry['base_rate']:.4f}. "
        f"As read: **{base}**.", "",
        f"* within-video frame pairs: {pairs['within'] / pairs['total']:.3f} of all abnormal x "
        f"normal pairs, pair-weighted AUC {pairs['within_auc']:.4f}; the rest cross two videos.",
        f"* normal-level spread on the as-read curves: std of the videos' normal median "
        f"{spread['normal_level_std']:.4f} against a mean normal-to-abnormal gap "
        f"{spread['mean_gap']:.4f} (ratio {spread['ratio']:.2f}).",
        f"* calibration headroom (`{ORACLE_Z}` minus as-read, lower bound > "
        f"{constants.BASEMICRO_HEADROOM_MARGIN}): **{entry['headroom_sentence']}**.", "",
        "| row | micro | AP | macro | cross-video AUC | Δ micro vs as-read [95 %] | sentence "
        "| share of headroom |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for name in ROW_ORDER:
        r = entry["rows"][name]
        cross = r["cross_video_auc"]
        tag = " (as read)" if name == base else (" (oracle)" if name in ORACLE else "")
        lines.append(
            f"| {name}{tag} | {r['micro']:.4f} | {r['ap']:.4f} | {r['macro']:.4f} "
            f"| {'—' if cross is None else f'{cross:.4f}'} | {_ci(r['delta_vs_as_read'])} "
            f"| {r.get('sentence', '—')} | {_share(r['share_of_headroom'])} |")
    return [*lines, ""]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# K5 between-video baseline: micro under per-video transforms (`KIM_RUN.md` §21)", "",
        f"{LABEL}. Saved scores only; nothing is trained. Macro is unchanged by every row (G2). "
        f"Bootstrap: {report['draws']} cluster draws, seed {report['seed']}. Oracle rows use the "
        "ground truth: ceilings, never methods. Every other transform needs the whole video "
        "(offline) and is a **protocol change, printed only**: it is not a model improvement and "
        "never sits beside a published number.", "",
    ]
    for entry in report["sets"].values():
        lines += _set_md(entry)
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    sets: dict[str, Any] = {}
    for label, npz, as_read, cluster, readout in args.set:
        if label in sets:
            raise SystemExit(f"--set {label!r} given twice")
        sets[label] = read_set(label, Path(npz), as_read, cluster, Path(readout))
        LOGGER.info("%s: as-read micro %.4f | headroom %s", label,
                    sets[label]["rows"][sets[label]["as_read"]]["micro"],
                    sets[label]["headroom_sentence"])
    report = {"sets": sets, "seed": SEED, "draws": constants.BASEMICRO_BOOTSTRAP_DRAWS}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / constants.BASEMICRO_REPORT_STEM
    write_json_atomic(stem.with_suffix(".json"), report)
    write_text_atomic(stem.with_suffix(".md"), render_markdown(report))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--set", nargs=5, action="append", required=True,
                        metavar=("LABEL", "CLIP_SCORES_NPZ", "AS_READ", "CLUSTER", "READOUT_JSON"),
                        help="one saved-score set; AS_READ in {raw, minmax} is the pooling of its "
                             "recorded micro, CLUSTER in {item, dota} the bootstrap unit")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
