"""The v2 training dataset dir: T2-train minus T2-val, evaluated on T2-val's windows.

Proposal §7.2: "All arms train on T2-train minus T2-val", and T2-val is the in-domain
decision set (pilot read-out, addendum §4; E3's key secondary). The T2 dataset dir trains on
every T2-train window -- T2-val included -- and evaluates on T2-test. This tool writes a copy
whose

* ``labels_train.json`` keeps only the windows of T2-train sources **not** in the frozen
  T2-val split (``core/splits/v2/t2_val_sources.txt``, read through ``load_split``);
* ``frame_labels_test.json`` holds **T2-val's windows** instead of T2-test's, labelled by the
  same arithmetic that labelled T2-test (``sampled_frame_labels`` over the source, sliced to
  the window), so ``core.evaluate`` on this dir scores T2-val and T2-test stays unread.

Every other file is copied unchanged (``windows.json`` keeps both sides' slices). The KNN
cache is keyed to the train split, so it must be rebuilt on the output dir
(``python -m core.data.knn_cache --data-dir <out-dir>``) or DVS splices in T2-val normals.
A ``v2_dataset_manifest.json`` records the split sha1 and the counts. Phase 4 trained on the
parent dir, so a phase-4 checkpoint scored on this dir's T2-val is scored **in-sample**.

CLI::

    python -m core.tools.v2_dataset --data-dir data/DADA2000_orig \\
        --out-dir data/DADA2000_orig_v2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.data.dada import DadaRecord, sampled_frame_labels
from core.data.v2_splits import lines_sha1, load_split
from core.tools.kill_switch_probe import write_json_atomic
from core.tools.subset_train import (
    SubsetPlan,
    TrainPool,
    load_train_pool,
    validate_subset,
    write_subset,
)

LOGGER = logging.getLogger(__name__)

MANIFEST_FILENAME = "v2_dataset_manifest.json"


def window_frame_labels(entry: dict[str, Any], window_id: str) -> list[int]:
    """A window's s8 frame labels from its meta entry: the source's labels, sliced."""
    span = entry.get("normalized_span")
    record = DadaRecord(
        video_id=str(entry["source"]),
        folder_name=str(entry["source"]),
        class_name=str(entry["class_name"]),
        fault_label=str(entry.get("fault_label", "")),
        total_frames=int(entry["total_frames"]),
        span=None if span is None else (float(span[0]), float(span[1])),
        accident_frac=entry.get("accident_frac"),
    )
    source = sampled_frame_labels(record, constants.FRAME_STRIDE)
    start, end = int(entry["start"]), int(entry["end"])
    if end > len(source) or end - start != int(entry["sampled_frames"]):
        raise ValueError(
            f"{window_id}: window {start}:{end} does not fit its source's {len(source)} rows "
            f"or its {entry['sampled_frames']} sampled frames"
        )
    return source[start:end]


def check_label_arithmetic(
    meta: dict[str, dict[str, Any]], parent_frame_labels: dict[str, list[int]]
) -> int:
    """Rebuild the parent's own test windows' labels; every one must match (or raise).

    The T2-val labels written below come from the same function, so this proves the
    arithmetic on windows whose labels the T2 build wrote itself.
    """
    bad = [w for w, labels in parent_frame_labels.items()
           if window_frame_labels(meta[w], w) != list(labels)]
    if bad:
        raise ValueError(
            f"{len(bad)} of the parent's test windows are not reproduced from meta.json: "
            f"{bad[:5]}; T2-val labels built the same way cannot be trusted"
        )
    return len(parent_frame_labels)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def plan_v2(pool: TrainPool, val_sources: list[str]) -> SubsetPlan:
    """Keep every T2-train source except T2-val's; T2-val must be T2-train sources."""
    train_sources = set(pool.windows_by_source)
    missing = sorted(set(val_sources) - train_sources)
    if missing:
        raise ValueError(f"{len(missing)} T2-val sources are not T2-train sources: {missing[:5]}")
    kept = train_sources - set(val_sources)
    kept_by_type = {t: sum(s in kept for s in srcs) for t, srcs in pool.sources_by_type.items()}
    total_by_type = {t: len(srcs) for t, srcs in pool.sources_by_type.items()}
    plan = SubsetPlan(kept, kept_by_type, total_by_type, [])
    validate_subset(pool, plan, None)
    return plan


def make_v2_dataset(data_dir: Path, out_dir: Path, split_dir: Path) -> dict[str, Any]:
    """Write the v2 dataset dir; returns its manifest."""
    pool = load_train_pool(data_dir)
    val_sources = load_split(constants.V2_SPLIT_T2_VAL, split_dir)
    plan = plan_v2(pool, val_sources)
    meta: dict[str, dict[str, Any]] = _read_json(data_dir / constants.META_FILENAME)
    checked = check_label_arithmetic(
        meta, _read_json(data_dir / constants.FRAME_LABELS_TEST_FILENAME)
    )
    val_windows = sorted(w for s in val_sources for w in pool.windows_by_source[s])
    frame_labels = {w: window_frame_labels(meta[w], w) for w in val_windows}
    disagree = sorted(w for w in val_windows if int(any(frame_labels[w])) != pool.labels[w])
    if disagree:
        raise ValueError(
            f"{len(disagree)} T2-val windows' frame labels disagree with labels_train.json: "
            f"{disagree[:5]}"
        )
    train_ids = sorted(w for s in plan.kept_sources for w in pool.windows_by_source[s])
    manifest = {
        "rule": "proposal §7.2: train on T2-train minus T2-val; evaluate on T2-val windows",
        "parent": str(data_dir),
        "t2_val_split_sha1": lines_sha1(val_sources),
        "t2_val_sources": len(val_sources),
        "t2_val_windows": len(val_windows),
        "t2_val_windows_abnormal": sum(pool.labels[w] for w in val_windows),
        "train_sources": len(plan.kept_sources),
        "train_windows": len(train_ids),
        "train_windows_abnormal": sum(pool.labels[w] for w in train_ids),
        "train_ids_sha1": lines_sha1(sorted(plan.kept_sources)),
        "frame_stride": constants.FRAME_STRIDE,
        "label_arithmetic_checked_on_parent_test_windows": checked,
        "knn_cache": "rebuild: python -m core.data.knn_cache --data-dir <this dir>",
        "in_sample_warning": "a checkpoint trained on the parent dir has seen T2-val's windows",
        # write_subset logs these three keys
        "fraction": len(plan.kept_sources) / len(pool.windows_by_source),
        "seed": None,
        "sources_kept": len(plan.kept_sources),
        "sources_total": len(pool.windows_by_source),
        "items_kept": len(train_ids),
        "items_total": len(pool.labels),
        "abnormal_kept": sum(pool.labels[w] for w in train_ids),
    }
    write_subset(data_dir, out_dir, pool, plan, manifest)
    (out_dir / constants.SUBSET_MANIFEST_FILENAME).replace(out_dir / MANIFEST_FILENAME)
    write_json_atomic(out_dir / constants.FRAME_LABELS_TEST_FILENAME, frame_labels)
    LOGGER.info("v2 dataset: train %d sources / %d windows; eval T2-val %d sources / %d windows",
                manifest["train_sources"], manifest["train_windows"],
                manifest["t2_val_sources"], manifest["t2_val_windows"])
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="the T2 dataset dir")
    parser.add_argument("--out-dir", type=Path, required=True, help="the v2 dataset dir")
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    make_v2_dataset(args.data_dir, args.out_dir, args.split_dir)


if __name__ == "__main__":
    main()
