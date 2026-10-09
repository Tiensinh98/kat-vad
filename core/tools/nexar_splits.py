"""Freeze the Nexar v2 splits (nexar_train / nexar_val / nexar_test) from the N0 census alone.

Plan ``.project/plans/katvad-v2-nexar-feasibility.md`` N0. The split reads **metadata only** --
labels, durations and annotated event times from ``census.json`` -- so it cannot be tuned
against a result. ``nexar_test`` is sealed (``constants.V2_SEALED_SPLITS``) until Nexar's final
read-out; the frozen ids go to ``core/splits/v2/`` with their own manifest
(``NEXAR_MANIFEST.json``), which :func:`core.data.v2_splits.frozen_splits` merges.

Rule (fixed before the census was read):

* one video = one group (no two Nexar ids are known to share a recording);
* stratum = label x duration half (<= / > the pooled median duration) and, for positives,
  the tercile of ``t_event / duration`` (edges = pooled tercile edges over positives with a
  clean annotation; a positive without one goes to its own ``pna`` stratum);
* ``nexar_test`` = ``NEXAR_TEST_FRACTION`` of all videos; ``nexar_val`` =
  ``NEXAR_VAL_FRACTION`` of all videos, drawn from what the test draw left; the rest is
  ``nexar_train``. Both draws use :func:`core.data.v2_splits.grouped_split` with the v2 seed.

Every video with a census row is placed, including ones with annotation issues: which of
those a corpus keeps is N1's decision, made per split afterwards.

CLI::

    python -m core.tools.nexar_splits --census outputs/v2/REPORTS/nexar_n0/census.json
    python -m core.tools.nexar_splits --census ... --check
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.v2_splits import grouped_split, lines_sha1
from core.tools.freeze_splits import check_splits, file_sha1, write_splits

LOGGER = logging.getLogger(__name__)

STRATUM_NO_POSITION = "pna"


def position_edges(rows: dict[str, dict[str, Any]]) -> list[float]:
    """Pooled tercile edges of ``t_event / duration`` over clean positives."""
    rel = [
        r["time_of_event"] / r["duration_s"]
        for r in rows.values()
        if r["label"] == 1 and not r["issues"] and r.get("duration_s")
    ]
    if not rel:
        return []
    qs = np.linspace(0.0, 1.0, constants.NEXAR_POSITION_STRATA + 1)[1:-1]
    return [float(q) for q in np.quantile(np.asarray(rel), qs)]


def stratum_of(row: dict[str, Any], median_duration: float, edges: list[float]) -> str:
    """``neg|short`` / ``pos|long|p2`` / ``pos|short|pna`` ..."""
    duration = row.get("duration_s")
    half = "long" if duration is not None and duration > median_duration else "short"
    if row["label"] == 0:
        return f"neg|{half}"
    if row["issues"] or not duration:
        return f"pos|{half}|{STRATUM_NO_POSITION}"
    rel = row["time_of_event"] / duration
    return f"pos|{half}|p{int(np.searchsorted(edges, rel, side='right'))}"


def freeze_nexar(census: dict[str, Any], seed: int) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """The three Nexar splits and their manifest block; deterministic in census + seed."""
    rows: dict[str, dict[str, Any]] = census["videos"]
    if len(rows) != constants.NEXAR_VIDEOS:
        raise ValueError(
            f"census has {len(rows)} videos, expected {constants.NEXAR_VIDEOS}: "
            "freeze only on the complete release"
        )
    durations = [r["duration_s"] for r in rows.values() if r.get("duration_s")]
    median_duration = float(statistics.median(durations))
    edges = position_edges(rows)
    strata = {vid: stratum_of(r, median_duration, edges) for vid, r in rows.items()}
    groups = {vid: [vid] for vid in rows}

    test, rest = grouped_split(groups, strata, constants.NEXAR_TEST_FRACTION, seed)
    val_fraction = constants.NEXAR_VAL_FRACTION / (1.0 - constants.NEXAR_TEST_FRACTION)
    rest_groups = {vid: [vid] for vid in sorted(rest)}
    val, train = grouped_split(rest_groups, {v: strata[v] for v in rest_groups}, val_fraction, seed)
    splits = {
        constants.V2_SPLIT_NEXAR_TRAIN: sorted(train),
        constants.V2_SPLIT_NEXAR_VAL: sorted(val),
        constants.V2_SPLIT_NEXAR_TEST: sorted(test),
    }

    per_stratum: dict[str, dict[str, int]] = {}
    for name, ids in splits.items():
        for stratum, n in sorted(Counter(strata[v] for v in ids).items()):
            per_stratum.setdefault(stratum, {})[name] = n
    manifest = {
        "plan": ".project/plans/katvad-v2-nexar-feasibility.md N0",
        "rule": (
            "one video = one group; stratum = label x duration half (pooled median) x, for "
            "positives, tercile of t_event/duration (pooled over clean positives; else 'pna'); "
            f"test = {constants.NEXAR_TEST_FRACTION} of all, val = {constants.NEXAR_VAL_FRACTION} "
            "of all drawn from the rest (core.data.v2_splits.grouped_split)"
        ),
        "seed": seed,
        "sealed": sorted(constants.V2_SEALED_SPLITS & splits.keys()),
        "median_duration_s": round(median_duration, 4),
        "position_edges": [round(e, 6) for e in edges],
        "per_label": {
            name: dict(sorted(Counter(str(rows[v]["label"]) for v in ids).items()))
            for name, ids in splits.items()
        },
        "per_stratum": dict(sorted(per_stratum.items())),
        "with_issues": {
            name: sum(bool(rows[v]["issues"]) for v in ids) for name, ids in splits.items()
        },
        "splits": {
            name: {"count": len(ids), "sha1": lines_sha1(ids)} for name, ids in splits.items()
        },
    }
    return splits, manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--census", type=Path, required=True, help="census.json from nexar_census")
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=constants.V2_SPLIT_SEED)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="recompute and verify the committed Nexar splits; write nothing",
    )
    mode.add_argument(
        "--force", action="store_true", help="overwrite (only before any Nexar number is read)"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    census: dict[str, Any] = json.loads(args.census.read_text(encoding="utf-8"))
    splits, manifest = freeze_nexar(census, args.seed)
    manifest["inputs"] = {
        "census": {"name": args.census.name, "sha1": file_sha1(args.census)},
        "metadata": census["inputs"],
    }
    LOGGER.info("Nexar splits %s, per label %s", manifest["splits"], manifest["per_label"])
    if args.check:
        check_splits(splits, manifest, args.split_dir, constants.V2_NEXAR_MANIFEST_FILENAME)
        LOGGER.info("Frozen Nexar splits match a fresh recomputation")
        return
    write_splits(splits, manifest, args.split_dir, args.force, constants.V2_NEXAR_MANIFEST_FILENAME)
    LOGGER.info("Froze Nexar splits -> %s", args.split_dir)


if __name__ == "__main__":
    main()
