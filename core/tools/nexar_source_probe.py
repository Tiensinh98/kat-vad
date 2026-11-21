"""Can frozen CLIP tell which Nexar video a normal window came from? (``KIM_RUN.md`` §38 S0).

K5fno, the arm this gates, keeps K5f's positive windows and draws its normal windows only
from Nexar's **negative** videos. In K5f a normal window comes from the same video as the
positive one, so scene, weather and camera are shared by both classes. In K5fno they are not.
If the two normal pools are separable on frozen CLIP, K5fno's label can be read from the
video and not from the event, and its TAD numbers say nothing about where normals come from
(lesson C38: every imported normal pool so far had shortcut AUC 1.000).

Read on one built corpus, ``Nexar_window_neg`` (§17), whose train split holds every window
the question needs, each tagged in ``meta.json`` with ``source_is_abnormal`` and
``positive_frames``:

* **A**: normal windows inside positive videos (K5f's normal pool);
* **B**: windows of negative videos (K5fno's normal pool);
* **P**: positive windows (the same in both arms).

Probes (window = mean of its CLIP rows; logistic, standardized, class-balanced; GroupKFold by
source video, so no video is scored by a fold that saw it):

* **S1 (decides)**: A vs B. ``>= NEXAR_SOURCE_PROBE_SHORTCUT_AUC`` = shortcut, K5fno is not built.
* **S2 (printed)**: S1 with B subsampled to A's scene x weather shares (strata in both only).
* **S3 (printed)**: scene x weather one-hot alone, A vs B: what the metadata explains, no pixels.
* **S4 (printed)**: P vs A (the within-video contrast K5f learns) and P vs B (K5fno's).

Usage::

    python -m core.tools.nexar_source_probe --data-dir data/Nexar_window_neg \\
        --clip-dir cache/clip/Nexar_s8_ncc --out-dir outputs/v_final/kim_source_probe/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from core import constants
from core.data.windows import FeatureSlicer, load_windows
from core.eda.features import transfer_scores
from core.metrics import frame_auc
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

POOL_A = "A"  # normal windows inside positive videos
POOL_B = "B"  # windows of negative videos
POOL_P = "P"  # positive windows
UNKNOWN_STRATUM = "unknown"


def split_pools(meta: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Train window ids by pool; raises if a pool is empty (wrong corpus)."""
    pools: dict[str, list[str]] = {POOL_A: [], POOL_B: [], POOL_P: []}
    for wid, row in sorted(meta.items()):
        if row.get("split") != "train":
            continue
        if int(row["positive_frames"]) > 0:
            pools[POOL_P].append(wid)
        elif row["source_is_abnormal"]:
            pools[POOL_A].append(wid)
        else:
            pools[POOL_B].append(wid)
    empty = [name for name, ids in pools.items() if not ids]
    if empty:
        raise ValueError(
            f"pools {empty} are empty: the corpus must be Nexar_window_neg (§17), whose train "
            "split holds positive windows, in-video normals and negative-video windows")
    return pools


def stratum_of(row: dict[str, Any]) -> str:
    """``scene|weather`` of a window's source video."""
    return "|".join(str(row.get(k) or UNKNOWN_STRATUM)
                    for k in constants.NEXAR_SOURCE_PROBE_STRATA_KEYS)


def window_means(clip_dir: Path, ids: list[str], slicer: FeatureSlicer, row_stride: int = 1
                 ) -> np.ndarray:
    """``(len(ids), D)`` mean row of each window, each source file read once.

    ``row_stride`` maps a cache to the corpus rows: ``s1[::8]`` = Nexar's s8 training rows.
    """
    by_source: dict[str, list[int]] = {}
    for index, wid in enumerate(ids):
        by_source.setdefault(slicer.source_of(wid), []).append(index)
    out: np.ndarray | None = None
    for source, indices in by_source.items():
        rows = np.load(clip_dir / f"{source}.npy")[::row_stride]
        for index in indices:
            window = slicer.window_of(ids[index])
            if window is None or window.end > len(rows):
                raise ValueError(f"{ids[index]} does not fit {source}.npy ({len(rows)} rows): "
                                 "cache and windows were built at different strides (C2)")
            mean = rows[window.start:window.end].mean(axis=0)
            if out is None:
                out = np.zeros((len(ids), mean.shape[0]), dtype=np.float64)
            out[index] = mean
    if out is None:
        raise ValueError("no window to load")
    return out


def grouped_auc(matrix: np.ndarray, target: np.ndarray, groups: list[str], folds: int,
                seed: int) -> float:
    """Out-of-fold AUC of the EDA logistic probe, folds grouped by source video."""
    codes = {g: k for k, g in enumerate(sorted(set(groups)))}
    group = np.array([codes[g] for g in groups])
    scores = np.zeros(len(target), dtype=np.float64)
    for train, test in GroupKFold(n_splits=folds).split(matrix, target, group):
        scores[test] = transfer_scores(matrix[train], target[train], matrix[test], seed)
    return frame_auc(scores, target)


def stratified_match(ids_a: list[str], ids_b: list[str], strata: dict[str, str],
                     seed: int) -> tuple[list[str], list[str]]:
    """A restricted to strata B has, and B subsampled to A's shares over those strata."""
    count_b = Counter(strata[w] for w in ids_b)
    kept_a = [w for w in ids_a if count_b[strata[w]] > 0]
    share_a = Counter(strata[w] for w in kept_a)
    total = min(count_b[s] * len(kept_a) // n for s, n in share_a.items())
    rng = np.random.default_rng(seed)
    kept_b: list[str] = []
    for stratum, n in sorted(share_a.items()):
        pool = sorted(w for w in ids_b if strata[w] == stratum)
        take = n * total // len(kept_a)
        kept_b.extend(rng.choice(pool, size=take, replace=False).tolist())
    return kept_a, sorted(kept_b)


def one_hot(ids: list[str], strata: dict[str, str]) -> np.ndarray:
    names = sorted({strata[w] for w in ids})
    index = {s: k for k, s in enumerate(names)}
    matrix = np.zeros((len(ids), len(names)), dtype=np.float64)
    for row, wid in enumerate(ids):
        matrix[row, index[strata[wid]]] = 1.0
    return matrix


def contrast(features: dict[str, np.ndarray], positive: list[str], negative: list[str],
             sources: dict[str, str], folds: int, seed: int) -> float:
    ids = [*positive, *negative]
    matrix = np.stack([features[w] for w in ids])
    target = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))]).astype(np.int64)
    return grouped_auc(matrix, target, [sources[w] for w in ids], folds, seed)


def run(data_dir: Path, clip_dir: Path, folds: int, seed: int, row_stride: int = 1
        ) -> dict[str, Any]:
    meta = json.loads((data_dir / constants.META_FILENAME).read_text(encoding="utf-8"))
    windows = load_windows(data_dir)
    if windows is None:
        raise ValueError(f"{data_dir} has no windows.json: not a windowed corpus")
    slicer = FeatureSlicer(windows)
    pools = split_pools(meta)
    ids = [w for name in (POOL_A, POOL_B, POOL_P) for w in pools[name]]
    features = dict(zip(ids, window_means(clip_dir, ids, slicer, row_stride), strict=True))
    sources = {w: str(meta[w]["source"]) for w in ids}
    strata = {w: stratum_of(meta[w]) for w in ids}
    a, b, p = pools[POOL_A], pools[POOL_B], pools[POOL_P]
    a_matched, b_matched = stratified_match(a, b, strata, seed)
    meta_ids = [*a, *b]
    meta_target = np.concatenate([np.ones(len(a)), np.zeros(len(b))]).astype(np.int64)
    s1 = contrast(features, a, b, sources, folds, seed)
    report: dict[str, Any] = {
        "rule": "KIM_RUN.md §38 S0: S1 >= threshold -> shortcut, K5fno not built",
        "threshold": constants.NEXAR_SOURCE_PROBE_SHORTCUT_AUC,
        "folds": folds,
        "seed": seed,
        "clip_dir": str(clip_dir),
        "row_stride": row_stride,
        "feature_dim": int(next(iter(features.values())).shape[0]),
        "windows": {name: len(pool) for name, pool in pools.items()},
        "videos": {name: len({sources[w] for w in pool}) for name, pool in pools.items()},
        "strata": {name: dict(Counter(strata[w] for w in pool).most_common())
                   for name, pool in pools.items()},
        "S1_A_vs_B": s1,
        "S2_A_vs_B_matched": contrast(features, a_matched, b_matched, sources, folds, seed),
        "S2_windows": {POOL_A: len(a_matched), POOL_B: len(b_matched)},
        "S3_metadata_only": grouped_auc(one_hot(meta_ids, strata), meta_target,
                                        [sources[w] for w in meta_ids], folds, seed),
        "S4_P_vs_A": contrast(features, p, a, sources, folds, seed),
        "S4_P_vs_B": contrast(features, p, b, sources, folds, seed),
        "shortcut": bool(s1 >= constants.NEXAR_SOURCE_PROBE_SHORTCUT_AUC),
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    verdict = ("**SHORTCUT: K5fno is not built**" if report["shortcut"]
               else "**no shortcut at the gate: K5fno may be built (stratified B printed)**")
    lines = [
        "# Nexar source-shortcut probe (`KIM_RUN.md` §38 S0)",
        "",
        f"Frozen CLIP, mean over each window, logistic probe, {report['folds']}-fold GroupKFold "
        f"by source video, seed {report['seed']}. Gate: S1 >= {report['threshold']:.2f}.",
        "",
        f"Input: `{report['clip_dir']}` rows `[::{report['row_stride']}]`, "
        f"{report['feature_dim']} dims.",
        "",
        "| pool | windows | videos |",
        "| --- | --- | --- |",
        *(f"| {name} | {report['windows'][name]} | {report['videos'][name]} |"
          for name in (POOL_A, POOL_B, POOL_P)),
        "",
        "A = normal windows inside positive videos (K5f's normals); B = negative-video windows "
        "(K5fno's normals); P = positive windows.",
        "",
        "| probe | AUC | role |",
        "| --- | --- | --- |",
        f"| S1 A vs B | {report['S1_A_vs_B']:.4f} | decides |",
        f"| S2 A vs B, B matched to A's scene x weather "
        f"({report['S2_windows'][POOL_A]} / {report['S2_windows'][POOL_B]}) | "
        f"{report['S2_A_vs_B_matched']:.4f} | printed |",
        f"| S3 scene x weather one-hot only | {report['S3_metadata_only']:.4f} | printed |",
        f"| S4 P vs A (K5f's contrast) | {report['S4_P_vs_A']:.4f} | printed |",
        f"| S4 P vs B (K5fno's contrast) | {report['S4_P_vs_B']:.4f} | printed |",
        "",
        verdict,
        "",
        "S4: if P vs B sits far above P vs A, K5fno's task is easier than K5f's by a margin that "
        "is not the event. Printed, never gating.",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="built Nexar_window_neg corpus (meta.json + windows.json)")
    parser.add_argument("--clip-dir", type=Path, required=True,
                        help="CLIP cache keyed by source video at the corpus stride (s8)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=constants.NEXAR_SOURCE_PROBE_FOLDS)
    parser.add_argument("--seed", type=int, default=constants.SEED)
    parser.add_argument("--row-stride", type=int, default=1,
                        help="rows[::k] of each cache file (8 for Nexar's s1 CLIP cache)")
    parser.add_argument("--name", default="", help="report suffix: <stem>_<name>.{json,md}")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    report = run(args.data_dir, args.clip_dir, args.folds, args.seed, args.row_stride)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.name}" if args.name else ""
    stem = args.out_dir / f"{constants.NEXAR_SOURCE_PROBE_STEM}{suffix}"
    write_json_atomic(stem.with_suffix(".json"), report)
    stem.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    LOGGER.info("S1 %.4f (gate %.2f) -> %s", report["S1_A_vs_B"], report["threshold"],
                stem.with_suffix(".md"))


if __name__ == "__main__":
    main()
