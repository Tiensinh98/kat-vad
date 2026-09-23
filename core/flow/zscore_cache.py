"""Rebuild a flow cache with a z-scored target (Option A) -- no frames, no RAFT.

``L_KIP_rec`` is a bare MSE against ``e_O = s @ M``, where ``s`` is 23 raw
frame-global flow statistics in pixel units. On T2 that made the term ~32x the
rest of the objective and let it capture the shared trunk (lesson **C37**;
``.project/plans/katvad-kip-loss-scale-diagnosis.md`` Appendix A). This tool
standardizes ``s`` with **train-split** moments and re-applies the **same**
projection ``M``. Everything it needs is already in the v1 cache:

    {src_root}/flow_projection.npz          M, loaded via load_projection (fails loud)
    {src_root}/{DATASET}/{id}.stats.npy      raw s -> input; copied byte-identical
    {src_root}/{DATASET}/{id}.npy            v1 e_O -> gate G0 only

and it writes a new cache version (lesson **C2**: never overwrite ``flow/v1``):

    {dst_root}/flow_projection.npz           the same M, copied
    {dst_root}/{DATASET}/{id}.npy            ((s - mu) / sigma) @ M, float32
    {dst_root}/{DATASET}/{id}.stats.npy      raw s, unchanged (KNN motion key)
    {dst_root}/{DATASET}/zscore_stats.npz    mu, sigma, and the split they bind to
    {dst_root}/{DATASET}/zscore_manifest.json gates, target baselines, lambda_rec

``mu, sigma`` are fitted over the **train windows** through
:class:`~core.data.windows.FeatureSlicer` -- the exact rows ``L_KIP_rec``
averages over, and the population D1's ``V`` was measured on. Every source file
is transformed (the transform is per-frame), so the file set matches v1's.

After writing, the tool scores the plan's pre-registered build bars (§6.1) with
:func:`core.eda.features.flow_stats` on the new cache and derives
``lambda_rec = 1 / V``. It is **derived, never swept** (lesson 14), and it is not
v1's 0.0316: reusing that value on this cache switches ``L_KIP_rec`` off.

CLI::

    python -m core.flow.zscore_cache --data-dir data/DADA2000_orig
        --dataset DADA2000_orig [--src-root cache/flow/v1]
        [--dst-root cache/flow/v2_zscore] [--force]
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.windows import FeatureSlicer
from core.eda.corpus import load_dataset_files
from core.eda.features import flow_stats
from core.flow.raft_extract import load_projection
from core.flow.zscore import (
    MomentAccumulator,
    ZScoreStats,
    fit_zscore,
    ids_digest,
    load_zscore_stats,
    save_zscore_stats,
    standardize,
)
from core.tools.feature_cache import Progress, is_complete, part_path, pending_items, save_array

LOGGER = logging.getLogger(__name__)


def source_files(src_dir: Path) -> dict[str, Path]:
    """``{id: raw-stats path}`` for every source; raise if its ``e_O`` is missing."""
    suffix = constants.FLOW_STATS_SUFFIX
    sources = {
        path.name[: -len(suffix)]: path for path in sorted(src_dir.glob(f"*{suffix}"))
    }
    if not sources:
        raise FileNotFoundError(f"No {suffix} files under {src_dir}")
    orphans = [sid for sid in sources if not (src_dir / f"{sid}.npy").is_file()]
    if orphans:
        raise FileNotFoundError(
            f"{len(orphans)} sources have raw stats but no e_O in {src_dir} "
            f"(coverage {len(sources) - len(orphans)}/{len(sources)}, lesson C10): "
            f"{orphans[:5]}"
        )
    return sources


def check_projection(
    raw: np.ndarray, embeddings: np.ndarray, projection: np.ndarray, item_id: str
) -> float:
    """Gate G0: a cached ``e_O`` must equal its own raw stats @ projection.

    Returns the max absolute error; raises when it is out of tolerance, because
    then v1's ``.npy`` and ``.stats.npy`` (or the projection) disagree and nothing
    rebuilt from the stats would describe the target v1 trained on.
    """
    if raw.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"G0 {item_id}: {raw.shape[0]} stats rows vs {embeddings.shape[0]} e_O rows"
        )
    expected = (raw.astype(np.float32) @ projection).astype(np.float32)
    error = float(np.max(np.abs(expected - embeddings))) if expected.size else 0.0
    if not np.allclose(
        embeddings,
        expected,
        rtol=constants.FLOW_ZSCORE_G0_RTOL,
        atol=constants.FLOW_ZSCORE_G0_ATOL,
    ):
        raise ValueError(
            f"G0 failed for {item_id}: cached e_O differs from stats @ projection "
            f"by up to {error:.4g} -- v1 cache and projection disagree; HARD STOP"
        )
    return error


def fit_train_zscore(
    src_dir: Path, train_ids: list[str], slicer: FeatureSlicer, src_version: str
) -> ZScoreStats:
    """Moments of the raw stats over the train windows, exactly as the loss sees them."""
    accumulator = MomentAccumulator(constants.FLOW_STATS_DIM)
    for item_id in train_ids:
        source = src_dir / f"{slicer.source_of(item_id)}{constants.FLOW_STATS_SUFFIX}"
        if not source.is_file():
            raise FileNotFoundError(
                f"Train item {item_id} has no flow stats at {source}; a KIP-on run "
                "would crash on it too"
            )
        rows = np.asarray(
            slicer.load(src_dir, item_id, constants.FLOW_STATS_SUFFIX), dtype=np.float64
        )
        if not np.isfinite(rows).all():
            raise ValueError(f"Non-finite raw flow stats in train item {item_id}")
        accumulator.add(rows)
    zscore = fit_zscore(accumulator, ids_digest(train_ids), src_version)
    LOGGER.info(
        "Fitted z-score on %d train items / %d rows (split sha1 %s)",
        zscore.n_items, zscore.n_rows, zscore.train_ids_sha1[:12],
    )
    return zscore


def _pin_zscore(dst_dir: Path, zscore: ZScoreStats, force: bool) -> None:
    """Write the statistics, or refuse to mix two normalizations in one directory."""
    path = dst_dir / constants.FLOW_ZSCORE_STATS_FILENAME
    if path.is_file() and not force:
        existing = load_zscore_stats(path)
        if not existing.matches(zscore):
            raise ValueError(
                f"{path} was fitted on a different split or different source stats "
                f"(sha1 {existing.train_ids_sha1[:12]} vs {zscore.train_ids_sha1[:12]}); "
                "arrays built under it would mix two normalizations. Use a new "
                "--dst-root, or --force to rebuild every file"
            )
        return
    save_zscore_stats(path, zscore)


def _pin_projection(src_root: Path, dst_root: Path) -> np.ndarray:
    """Load v1's projection and make ``dst_root`` carry the identical matrix."""
    src_path = src_root / constants.FLOW_PROJECTION_FILENAME
    projection = load_projection(src_path)
    dst_path = dst_root / constants.FLOW_PROJECTION_FILENAME
    if dst_path.is_file():
        if not np.array_equal(load_projection(dst_path), projection):
            raise ValueError(f"{dst_path} differs from {src_path}; refusing to mix")
    else:
        dst_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, dst_path)
    return projection


def _copy_atomic(source: Path, target: Path) -> None:
    staged = part_path(target)
    shutil.copyfile(source, staged)
    staged.replace(target)


def rebuild_item(
    src_dir: Path,
    dst_dir: Path,
    item_id: str,
    projection: np.ndarray,
    zscore: ZScoreStats,
) -> float:
    """G0-check one source, then write its raw stats and z-scored ``e_O``.

    ``e_O`` is written last: it is what resume keys on, so a crash in between
    redoes both (same order as :mod:`core.flow.raft_extract`).
    """
    raw_path = src_dir / f"{item_id}{constants.FLOW_STATS_SUFFIX}"
    raw = np.load(raw_path)
    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite raw flow stats in {raw_path}")
    error = check_projection(raw, np.load(src_dir / f"{item_id}.npy"), projection, item_id)
    embeddings = (standardize(raw, zscore) @ projection).astype(np.float32)
    _copy_atomic(raw_path, dst_dir / f"{item_id}{constants.FLOW_STATS_SUFFIX}")
    save_array(dst_dir / f"{item_id}.npy", embeddings)
    return error


def _in_band(value: float, band: tuple[float, float]) -> bool:
    return band[0] <= value <= band[1]


def evaluate_gates(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Score the plan's §6.1 G1/G2 bars on a :func:`flow_stats` report of the new cache."""
    standardized = report["standardized"]
    target = report["target"]
    means = np.abs(np.asarray(standardized["per_dim_mean"]))
    stds = np.asarray(standardized["per_dim_std"])
    zero = float(target["mse_zero_predictor"])
    variance = float(target["mse_global_mean_predictor"])
    predicted = float(target["predicted_second_moment_from_raw"])
    centred = zero / variance if variance > 0 else float("inf")
    roundtrip = predicted / zero if zero > 0 else float("inf")
    std_band = constants.FLOW_ZSCORE_G1_STD_BAND
    return {
        "G1_mean": {
            "value": float(means.max()),
            "bar": f"<= {constants.FLOW_ZSCORE_G1_MEAN_TOL}",
            "passed": bool(means.max() <= constants.FLOW_ZSCORE_G1_MEAN_TOL),
            "hard": True,
        },
        "G1_std": {
            "value": [float(stds.min()), float(stds.max())],
            "bar": f"in {list(std_band)}",
            "passed": bool(stds.min() >= std_band[0] and stds.max() <= std_band[1]),
            "hard": True,
        },
        "G2a_V": {
            "value": variance,
            "bar": f"in {list(constants.FLOW_ZSCORE_G2A_V_BAND)} (predicted ~1)",
            "passed": _in_band(variance, constants.FLOW_ZSCORE_G2A_V_BAND),
            "hard": False,
        },
        "G2b_centred": {
            "value": centred,
            "bar": f"in {list(constants.FLOW_ZSCORE_G2B_CENTRED_BAND)}",
            "passed": _in_band(centred, constants.FLOW_ZSCORE_G2B_CENTRED_BAND),
            "hard": True,
        },
        "G2c_roundtrip": {
            "value": roundtrip,
            "bar": f"in {list(constants.FLOW_ZSCORE_G2C_ROUNDTRIP_BAND)}",
            "passed": _in_band(roundtrip, constants.FLOW_ZSCORE_G2C_ROUNDTRIP_BAND),
            "hard": True,
        },
        "G2d_between_item_share": {
            "value": float(target["between_item_share"]),
            "bar": "reported; v1 = 0.488, trip-wire 0.5",
            "passed": True,
            "hard": False,
        },
        "G2e_item_oracle_ratio": {
            "value": float(target["mse_item_mean_predictor"]) / variance if variance > 0 else 0.0,
            "bar": "reported (W / V)",
            "passed": True,
            "hard": False,
        },
    }


def build_zscore_cache(
    data_dir: Path,
    dataset: str,
    src_root: Path = constants.FLOW_CACHE_DIR,
    dst_root: Path = constants.FLOW_ZSCORE_CACHE_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Build ``{dst_root}/{dataset}``, score its gates, write and return the manifest.

    Raises after writing the manifest when a HARD gate fails, so the evidence of
    the failure survives the non-zero exit.
    """
    if src_root.resolve() == dst_root.resolve():
        raise ValueError(f"--dst-root must differ from --src-root ({src_root}); C2")
    files = load_dataset_files(data_dir, dataset)
    slicer = files.slicer
    train_ids = files.train_ids
    src_dir = src_root / dataset
    dst_dir = dst_root / dataset
    sources = source_files(src_dir)

    projection = _pin_projection(src_root, dst_root)
    zscore = fit_train_zscore(src_dir, train_ids, slicer, src_root.name)
    dst_dir.mkdir(parents=True, exist_ok=True)
    _pin_zscore(dst_dir, zscore, force)

    pending = pending_items(sources, dst_dir, force)
    progress = Progress(len(pending))
    g0_max = 0.0
    for item_id, _ in pending:
        g0_max = max(g0_max, rebuild_item(src_dir, dst_dir, item_id, projection, zscore))
        LOGGER.debug("Rebuilt %s -- %s", item_id, progress.step())
    complete = sum(is_complete(dst_dir / f"{sid}.npy") for sid in sources)
    if complete != len(sources):
        raise RuntimeError(
            f"Coverage {complete}/{len(sources)} in {dst_dir} after the rebuild (C10)"
        )

    report = flow_stats(dst_dir, train_ids, slicer=slicer)
    gates = evaluate_gates(report)
    passed = all(g["passed"] for g in gates.values() if g["hard"])
    variance = float(report["target"]["mse_global_mean_predictor"])
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "created": datetime.date.today().isoformat(),
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "coverage": {"sources": len(sources), "written_this_run": len(pending)},
        "g0": {"checked_this_run": len(pending), "max_abs_err": g0_max},
        "zscore": {
            "n_items": zscore.n_items,
            "n_rows": zscore.n_rows,
            "train_ids_sha1": zscore.train_ids_sha1,
            "stat_names": list(constants.FLOW_STAT_NAMES),
            "mean": [float(x) for x in zscore.mean],
            "std": [float(x) for x in zscore.std],
            # equal weight per stat: the angle histogram's share of the target
            "histogram_block_share": constants.FLOW_ANGLE_BINS / constants.FLOW_STATS_DIM,
        },
        "target": report["target"],
        "gates": gates,
        "passed": passed,
        "lambda_rec": round(1.0 / variance, constants.FLOW_ZSCORE_LAMBDA_DECIMALS)
        if variance > 0
        else None,
        "note": "lambda_rec = 1/V on THIS cache; never v1's 0.0316 (plan §6.1)",
    }
    manifest_path = dst_dir / constants.FLOW_ZSCORE_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, gate in gates.items():
        LOGGER.info(
            "%-24s %-5s value=%s bar=%s", name,
            "PASS" if gate["passed"] else ("FAIL" if gate["hard"] else "NOTE"),
            gate["value"], gate["bar"],
        )
    if not passed:
        failed = [n for n, g in gates.items() if g["hard"] and not g["passed"]]
        raise RuntimeError(f"HARD gate(s) failed: {failed}; see {manifest_path}")
    LOGGER.info("All hard gates pass. lambda_rec = %s (1/V, V=%.4f)",
                manifest["lambda_rec"], variance)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild a flow cache with a z-scored e_O target (Option A)"
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="dataset dir holding labels_train.json (+ windows.json)")
    parser.add_argument("--dataset", required=True, help="cache subdirectory name")
    parser.add_argument("--src-root", type=Path, default=constants.FLOW_CACHE_DIR)
    parser.add_argument("--dst-root", type=Path, default=constants.FLOW_ZSCORE_CACHE_DIR)
    parser.add_argument("--force", action="store_true",
                        help="refit and rewrite every file, replacing an existing z-score")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    build_zscore_cache(args.data_dir, args.dataset, args.src_root, args.dst_root, args.force)


if __name__ == "__main__":
    main()
