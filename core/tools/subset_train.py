"""Train-source subset of a dataset dir: the input to a data learning curve.

A dataset's training set is exactly the ids in ``labels_train.json``
(:class:`core.data.dataset.VADDataset`), so a subset is a copy of the dataset
dir whose ``labels_train.json`` keeps only the windows of a fraction of the
**source videos**. Every other file is copied unchanged: the test split, the
definitions and ``windows.json`` stay the parent's, so an arm trained on the
subset is scored on exactly the same items as the full-data arm.

Three rules make the subset a clean contrast (plan
``.project/plans/katvad-t2-learning-curve.md`` §3-§5):

* **By source, never by window.** A T2 source carries ~3.8 train windows; drawing
  windows would keep fragments of every accident, and "less data" would mean
  "less of each video" instead of "fewer videos".
* **Stratified by accident type**, like ``core.data.dada_origin.split_by_type``:
  ``round(n_type * fraction)`` sources per type. A type that rounds to zero is
  dropped and reported in the manifest, never forced in.
* **Nested** with ``--nest-in``: the draw is restricted to a larger subset's
  sources, so 25 % is a subset of 50 % for the same seed and the paired contrast
  between fractions compares nested training sets.

Refuses (raises) rather than writes when the result could not train or would
leak: a subset without both classes, a kept source that also has test windows,
or a ``--nest-in`` parent that is not a subset of this dataset.

The KNN cache is **not** copied: it is built from ``labels_train.json`` and must
be rebuilt on the subset dir (``python -m core.data.knn_cache``), or DVS splices
in normals the arm never trains on.

CLI::

    python -m core.tools.subset_train \\
      --data-dir "$DATA/DADA2000_orig" \\
      --out-dir  "$DATA/DADA2000_orig_f050_s2024" \\
      --fraction 0.5 --seed 2024
    python -m core.tools.subset_train \\
      --data-dir "$DATA/DADA2000_orig" \\
      --out-dir  "$DATA/DADA2000_orig_f025_s2024" \\
      --fraction 0.25 --seed 2024 --nest-in "$DATA/DADA2000_orig_f050_s2024"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import constants
from core.data.dataset_files import NORMAL_CLASS, TRAIN_IDS_FILENAME
from core.data.windows import load_windows
from core.tools.feature_cache import part_path

LOGGER = logging.getLogger(__name__)

# Files rewritten here rather than copied from the parent.
_REWRITTEN = frozenset(
    {constants.LABELS_TRAIN_FILENAME, constants.SUBSET_MANIFEST_FILENAME, TRAIN_IDS_FILENAME}
)
# Recorded in the manifest: the parent's KNN cache is keyed to its train split.
_KNN_HINT = "rebuild with: python -m core.data.knn_cache --data-dir <out-dir>"


@dataclass(frozen=True)
class TrainPool:
    """The parent dataset's train split, grouped for drawing."""

    labels: dict[str, int]
    windows_by_source: dict[str, list[str]]
    sources_by_type: dict[str, list[str]]
    test_sources: frozenset[str]


@dataclass
class SubsetPlan:
    """What a draw kept, and why it dropped what it dropped."""

    kept_sources: set[str]
    kept_by_type: dict[str, int] = field(default_factory=dict)
    total_by_type: dict[str, int] = field(default_factory=dict)
    dropped_types: list[str] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json_atomic(target: Path, payload: object) -> None:
    """Stage to a ``.part`` sibling, then rename (lesson C11c)."""
    staged = part_path(target)
    with staged.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    staged.replace(target)


def load_train_pool(data_dir: Path) -> TrainPool:
    """Group ``labels_train.json`` by source video and sources by accident type.

    Without ``windows.json`` every id is its own source. The type of a source is
    the ``class_name`` of its windows in ``meta.json``; a corpus whose meta carries
    none falls into one stratum, ``SUBSET_UNTYPED_GROUP``.
    """
    labels: dict[str, int] = _read_json(data_dir / constants.LABELS_TRAIN_FILENAME)
    windows = load_windows(data_dir)

    def source_of(item_id: str) -> str:
        if windows is None:
            return item_id
        if item_id not in windows:
            raise ValueError(f"{item_id} is in labels but not in {constants.WINDOWS_FILENAME}")
        return windows[item_id].source

    windows_by_source: dict[str, list[str]] = {}
    for item_id in sorted(labels):
        windows_by_source.setdefault(source_of(item_id), []).append(item_id)

    meta_path = data_dir / constants.META_FILENAME
    meta: dict[str, dict[str, Any]] = _read_json(meta_path) if meta_path.exists() else {}
    sources_by_type: dict[str, list[str]] = {}
    for source, item_ids in windows_by_source.items():
        names = {str(n) for i in item_ids if (n := meta.get(i, {}).get("class_name")) is not None}
        # T2 windows inherit their source's class name; should a corpus label its
        # negative windows Normal instead, the accident type is still the stratum.
        types = sorted(names - {NORMAL_CLASS}) or sorted(names)
        stratum = str(types[0]) if types else constants.SUBSET_UNTYPED_GROUP
        sources_by_type.setdefault(stratum, []).append(source)

    test_ids: dict[str, Any] = _read_json(data_dir / constants.FRAME_LABELS_TEST_FILENAME)
    test_sources = frozenset(source_of_test(i, windows) for i in test_ids)
    return TrainPool(labels, windows_by_source, sources_by_type, test_sources)


def source_of_test(item_id: str, windows: dict[str, Any] | None) -> str:
    """Source of a test id; unwindowed corpora use the id itself."""
    if windows is None or item_id not in windows:
        return item_id
    return str(windows[item_id].source)


def draw_subset(
    pool: TrainPool,
    fraction: float,
    seed: int,
    allowed: set[str] | None = None,
) -> SubsetPlan:
    """Keep ``round(n_type * fraction)`` sources per type, drawn with ``seed``.

    ``n_type`` is always the FULL train count of the type, so a nested draw
    (``allowed`` = a larger subset's sources) keeps the same per-type quota it
    would have kept from the whole pool, capped by what the parent holds.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    # deterministic subset draw, not security-sensitive
    rng = random.Random(f"{seed}:{fraction}")  # nosec B311
    plan = SubsetPlan(kept_sources=set())
    for stratum in sorted(pool.sources_by_type):
        sources = sorted(pool.sources_by_type[stratum])
        candidates = [s for s in sources if allowed is None or s in allowed]
        quota = min(round(len(sources) * fraction), len(candidates))
        rng.shuffle(candidates)
        plan.total_by_type[stratum] = len(sources)
        plan.kept_by_type[stratum] = quota
        plan.kept_sources.update(candidates[:quota])
        if quota == 0:
            plan.dropped_types.append(stratum)
    return plan


def validate_subset(pool: TrainPool, plan: SubsetPlan, allowed: set[str] | None) -> None:
    """Gates G-S1..G-S3 of the plan; raise instead of writing a bad subset."""
    leaked = plan.kept_sources & pool.test_sources
    if leaked:
        raise ValueError(f"{len(leaked)} kept sources also have test items: {sorted(leaked)[:3]}")
    if allowed is not None and not plan.kept_sources <= allowed:
        raise ValueError("nested draw kept sources outside its --nest-in parent")
    kept_labels = [pool.labels[i] for s in plan.kept_sources for i in pool.windows_by_source[s]]
    if 1 not in kept_labels or 0 not in kept_labels:
        raise ValueError(
            f"subset keeps {kept_labels.count(1)} abnormal / {kept_labels.count(0)} normal "
            "train items; training needs both classes"
        )


def parent_sources(parent_dir: Path, pool: TrainPool) -> set[str]:
    """Sources of a ``--nest-in`` subset, checked against this dataset's pool."""
    parent_labels: dict[str, int] = _read_json(parent_dir / constants.LABELS_TRAIN_FILENAME)
    index = {i: s for s, ids in pool.windows_by_source.items() for i in ids}
    unknown = sorted(set(parent_labels) - set(index))
    if unknown:
        raise ValueError(f"--nest-in has {len(unknown)} ids not in this dataset: {unknown[:3]}")
    return {index[i] for i in parent_labels}


def build_manifest(
    pool: TrainPool,
    plan: SubsetPlan,
    fraction: float,
    seed: int,
    data_dir: Path,
    nest_in: Path | None,
) -> dict[str, Any]:
    """Every number the plan's build gates read, plus provenance (C17)."""
    kept_ids = sorted(i for s in plan.kept_sources for i in pool.windows_by_source[s])
    n_abnormal = sum(pool.labels[i] for i in kept_ids)
    digest = hashlib.sha1(  # an identity fingerprint of the kept sources
        "\n".join(sorted(plan.kept_sources)).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return {
        "fraction": fraction,
        "seed": seed,
        "parent_data_dir": str(data_dir),
        "nest_in": str(nest_in) if nest_in is not None else None,
        "sources_kept": len(plan.kept_sources),
        "sources_total": len(pool.windows_by_source),
        "items_kept": len(kept_ids),
        "items_total": len(pool.labels),
        "item_fraction": len(kept_ids) / len(pool.labels),
        "abnormal_kept": n_abnormal,
        "normal_kept": len(kept_ids) - n_abnormal,
        "kept_by_type": plan.kept_by_type,
        "total_by_type": plan.total_by_type,
        "dropped_types": plan.dropped_types,
        "kept_sources_sha1": digest,
        "knn_cache": _KNN_HINT,
    }


def write_subset(
    data_dir: Path,
    out_dir: Path,
    pool: TrainPool,
    plan: SubsetPlan,
    manifest: dict[str, Any],
) -> Path:
    """Copy the parent's files, then write the filtered train split + manifest."""
    if out_dir.resolve() == data_dir.resolve():
        raise ValueError("--out-dir must differ from --data-dir; the parent is never edited")
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            LOGGER.warning("Not copying sub-directory %s", item.name)
            continue
        if item.name in _REWRITTEN or item.name.endswith(constants.CACHE_PART_SUFFIX):
            continue
        shutil.copy2(item, out_dir / item.name)

    kept_ids = {i for s in plan.kept_sources for i in pool.windows_by_source[s]}
    labels = {i: pool.labels[i] for i in sorted(kept_ids)}
    _write_json_atomic(out_dir / constants.LABELS_TRAIN_FILENAME, labels)

    train_ids_file = data_dir / TRAIN_IDS_FILENAME
    if train_ids_file.exists():
        keep = plan.kept_sources | kept_ids
        lines = train_ids_file.read_text(encoding="utf-8").splitlines()
        staged = part_path(out_dir / TRAIN_IDS_FILENAME)
        staged.write_text("".join(f"{x}\n" for x in lines if x.strip() in keep), encoding="utf-8")
        staged.replace(out_dir / TRAIN_IDS_FILENAME)

    target = out_dir / constants.SUBSET_MANIFEST_FILENAME
    _write_json_atomic(target, manifest)
    LOGGER.info(
        "Subset %.2f seed %d: %d/%d sources, %d/%d items (%d abnormal) -> %s",
        manifest["fraction"], manifest["seed"], manifest["sources_kept"],
        manifest["sources_total"], manifest["items_kept"], manifest["items_total"],
        manifest["abnormal_kept"], out_dir,
    )
    if plan.dropped_types:
        LOGGER.warning("Types with no source at this fraction: %s", plan.dropped_types)
    return target


def make_subset(
    data_dir: Path,
    out_dir: Path,
    fraction: float,
    seed: int,
    nest_in: Path | None = None,
) -> dict[str, Any]:
    """Draw, validate and write one subset; returns its manifest."""
    pool = load_train_pool(data_dir)
    allowed = parent_sources(nest_in, pool) if nest_in is not None else None
    plan = draw_subset(pool, fraction, seed, allowed)
    validate_subset(pool, plan, allowed)
    manifest = build_manifest(pool, plan, fraction, seed, data_dir, nest_in)
    write_subset(data_dir, out_dir, pool, plan, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="the FULL dataset dir (labels_train.json, windows.json, ...)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="where the subset dataset dir is written")
    parser.add_argument("--fraction", type=float, required=True,
                        help="fraction of train SOURCE videos kept, per accident type")
    parser.add_argument("--seed", type=int, default=constants.SEED)
    parser.add_argument("--nest-in", type=Path, default=None,
                        help="a larger subset dir of the same dataset; draw only from it")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    make_subset(args.data_dir, args.out_dir, args.fraction, args.seed, args.nest_in)


if __name__ == "__main__":
    main()
