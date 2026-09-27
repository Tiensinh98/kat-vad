"""Freeze the KAT-VAD v2 decision splits: T2-val and DoTA-dev / DoTA-eval.

Run once, commit the output, and record the commit hash in every v2 read-out
(proposal ``core/docs/v2/KAT_VAD_PROPOSAL_v2.md`` §7.2 and §10.2 step 1; plan
``.project/plans/katvad-v2-e0-e2.md`` P0).

* **T2-val** = ``V2_T2_VAL_FRACTION`` of the T2 **train source videos**, drawn
  per accident ``type`` (1..52) with :func:`core.tools.subset_train.draw_subset`.
  The stratum is ``type``, not ``class_name``: every T2 window's ``class_name``
  is ``CarAccident``, so a ``class_name`` stratum is a single group. Input is the
  T2 dataset's ``meta.json``; a window is abnormal iff ``positive_frames > 0``,
  which reproduces the T2 run manifest's 3,242 / 1,159 train counts.
* **DoTA-dev / DoTA-eval** = ``V2_DOTA_DEV_FRACTION`` of the val clips, grouped
  by source YouTube video (:func:`core.data.v2_splits.dota_group`) and
  stratified by the accident-share bin of each video's median clip.

Nothing here reads a score, so freezing cannot be tuned against a result.
Refuses to overwrite a frozen split without ``--force``; ``--check`` recomputes
everything and raises if the committed files differ.

CLI::

    python -m core.tools.freeze_splits \\
      --t2-meta outputs/EDA/DADA2000_orig_T2_w20s8/meta.json \\
      --dota-metadata data/DoTA/metadata_val.json \\
      --dota-split data/DoTA/val_split.txt
    python -m core.tools.freeze_splits ... --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
from pathlib import Path
from typing import Any

from core import constants
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_splits import (
    dota_group,
    grouped_split,
    lines_sha1,
    share_bin,
    split_path,
)
from core.tools.feature_cache import part_path
from core.tools.subset_train import TrainPool, draw_subset, validate_subset

LOGGER = logging.getLogger(__name__)

_T2_TRAIN = "train"
_T2_TEST = "test"
_DEV_SIDE = "dev"
_EVAL_SIDE = "eval"


def file_sha1(path: Path) -> str:
    """Content fingerprint of an input file (provenance, C17)."""
    return hashlib.sha1(path.read_bytes(), usedforsecurity=False).hexdigest()


def t2_pool_from_meta(meta: dict[str, dict[str, Any]]) -> TrainPool:
    """The T2 train split, grouped by source and stratified by accident ``type``."""
    labels: dict[str, int] = {}
    windows_by_source: dict[str, list[str]] = {}
    type_of_source: dict[str, str] = {}
    test_sources: set[str] = set()
    for window_id in sorted(meta):
        entry = meta[window_id]
        source = str(entry["source"])
        if entry["split"] == _T2_TEST:
            test_sources.add(source)
            continue
        if entry["split"] != _T2_TRAIN:
            raise ValueError(f"{window_id}: unknown split {entry['split']!r}")
        labels[window_id] = int(entry["positive_frames"] > 0)
        windows_by_source.setdefault(source, []).append(window_id)
        stratum = str(entry["type"])
        if type_of_source.setdefault(source, stratum) != stratum:
            raise ValueError(f"source {source} has windows of two types")
    sources_by_type: dict[str, list[str]] = {}
    for source, stratum in sorted(type_of_source.items()):
        sources_by_type.setdefault(stratum, []).append(source)
    return TrainPool(labels, windows_by_source, sources_by_type, frozenset(test_sources))


def freeze_t2_val(
    meta: dict[str, dict[str, Any]], seed: int
) -> tuple[list[str], dict[str, Any]]:
    """T2-val source ids and their manifest block."""
    pool = t2_pool_from_meta(meta)
    plan = draw_subset(pool, constants.V2_T2_VAL_FRACTION, seed)
    validate_subset(pool, plan, allowed=None)
    val_windows = [w for s in plan.kept_sources for w in pool.windows_by_source[s]]
    n_abnormal = sum(pool.labels[w] for w in val_windows)
    block = {
        "fraction": constants.V2_T2_VAL_FRACTION,
        "stratum": "type",
        "label_rule": "abnormal iff positive_frames > 0",
        "train_sources_total": len(pool.windows_by_source),
        "train_windows_total": len(pool.labels),
        "val_sources": len(plan.kept_sources),
        "val_windows": len(val_windows),
        "val_windows_abnormal": n_abnormal,
        "val_windows_normal": len(val_windows) - n_abnormal,
        "kept_by_type": plan.kept_by_type,
        "types_without_val_source": sorted(plan.dropped_types, key=int),
    }
    return sorted(plan.kept_sources), block


def freeze_dota(
    metadata_path: Path, split_path_: Path, seed: int
) -> tuple[list[str], list[str], dict[str, Any]]:
    """DoTA-dev and DoTA-eval clip ids and their manifest block."""
    records = parse_metadata(metadata_path, read_split_ids(split_path_))
    share = {r.video_id: r.span[1] - r.span[0] for r in records}
    clips_by_group: dict[str, list[str]] = {}
    for clip_id in sorted(share):
        clips_by_group.setdefault(dota_group(clip_id), []).append(clip_id)
    stratum_of_group = {
        group: share_bin(statistics.median(share[c] for c in clips))
        for group, clips in clips_by_group.items()
    }
    dev, rest = grouped_split(
        clips_by_group, stratum_of_group, constants.V2_DOTA_DEV_FRACTION, seed
    )

    per_bin: dict[str, dict[str, int]] = {
        label: {_DEV_SIDE: 0, _EVAL_SIDE: 0} for label in constants.V2_SHARE_BIN_LABELS
    }
    for clip_id, value in share.items():
        per_bin[share_bin(value)][_DEV_SIDE if clip_id in dev else _EVAL_SIDE] += 1
    block = {
        "fraction": constants.V2_DOTA_DEV_FRACTION,
        "group": "source YouTube video (clip id minus its trailing _frame)",
        "stratum": "share bin of the source video's median clip share",
        "clips_total": len(share),
        "clips_dev": len(dev),
        "clips_eval": len(rest),
        "groups_total": len(clips_by_group),
        "groups_dev": len({dota_group(c) for c in dev}),
        "groups_eval": len({dota_group(c) for c in rest}),
        "clips_per_share_bin": per_bin,
        "all_normal_clips": {
            c: (_DEV_SIDE if c in dev else _EVAL_SIDE) for c in sorted(share) if share[c] == 0.0
        },
    }
    return sorted(dev), sorted(rest), block


def build_splits(
    t2_meta: Path, dota_metadata: Path, dota_split: Path, seed: int
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Every split's ids plus the manifest; deterministic in its inputs and seed."""
    with t2_meta.open("r", encoding="utf-8") as fh:
        meta: dict[str, dict[str, Any]] = json.load(fh)
    t2_val, t2_block = freeze_t2_val(meta, seed)
    dev, rest, dota_block = freeze_dota(dota_metadata, dota_split, seed)
    splits = {
        constants.V2_SPLIT_T2_VAL: t2_val,
        constants.V2_SPLIT_DOTA_DEV: dev,
        constants.V2_SPLIT_DOTA_EVAL: rest,
    }
    manifest = {
        "seed": seed,
        "proposal": "core/docs/v2/KAT_VAD_PROPOSAL_v2.md §7.2",
        "plan": ".project/plans/katvad-v2-e0-e2.md P0",
        "sealed": sorted(constants.V2_SEALED_SPLITS),
        "inputs": {
            "t2_meta": {"name": t2_meta.name, "sha1": file_sha1(t2_meta)},
            "dota_metadata": {"name": dota_metadata.name, "sha1": file_sha1(dota_metadata)},
            "dota_split": {"name": dota_split.name, "sha1": file_sha1(dota_split)},
        },
        "splits": {
            name: {"count": len(ids), "sha1": lines_sha1(ids)} for name, ids in splits.items()
        },
        "t2_val": t2_block,
        "dota": dota_block,
    }
    return splits, manifest


def _write_atomic(target: Path, text: str) -> None:
    """Stage to a ``.part`` sibling, then rename (lesson C11c)."""
    staged = part_path(target)
    staged.write_text(text, encoding="utf-8")
    staged.replace(target)


def _manifest_text(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def write_splits(
    splits: dict[str, list[str]], manifest: dict[str, Any], out_dir: Path, force: bool
) -> None:
    """Write the id files and the manifest; refuse to overwrite without ``force``."""
    manifest_path = out_dir / constants.V2_SPLITS_MANIFEST_FILENAME
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"{manifest_path} exists: the v2 splits are frozen. Use --check to verify, "
            "or --force only if no v2 number has been read on them yet"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        _write_atomic(split_path(name, out_dir), "".join(f"{i}\n" for i in ids))
    _write_atomic(manifest_path, _manifest_text(manifest))


def check_splits(
    splits: dict[str, list[str]], manifest: dict[str, Any], out_dir: Path
) -> None:
    """Raise unless the committed files equal a fresh recomputation."""
    manifest_path = out_dir / constants.V2_SPLITS_MANIFEST_FILENAME
    if manifest_path.read_text(encoding="utf-8") != _manifest_text(manifest):
        raise ValueError(f"{manifest_path} differs from a fresh freeze of the same inputs")
    for name, ids in splits.items():
        path = split_path(name, out_dir)
        if path.read_text(encoding="utf-8") != "".join(f"{i}\n" for i in ids):
            raise ValueError(f"{path} differs from a fresh freeze of the same inputs")


def log_summary(manifest: dict[str, Any]) -> None:
    t2 = manifest["t2_val"]
    dota = manifest["dota"]
    LOGGER.info(
        "T2-val: %d / %d train sources, %d windows (%d abnormal / %d normal)",
        t2["val_sources"], t2["train_sources_total"], t2["val_windows"],
        t2["val_windows_abnormal"], t2["val_windows_normal"],
    )
    LOGGER.info(
        "DoTA: dev %d clips / %d videos, eval %d clips / %d videos; per share bin %s",
        dota["clips_dev"], dota["groups_dev"], dota["clips_eval"], dota["groups_eval"],
        dota["clips_per_share_bin"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--t2-meta", type=Path, required=True,
                        help="meta.json of the T2 dataset (DADA2000_orig, W=20 hop 8)")
    parser.add_argument("--dota-metadata", type=Path,
                        default=constants.DATA_ROOT / constants.DOTA_DATASET / "metadata_val.json")
    parser.add_argument("--dota-split", type=Path,
                        default=constants.DATA_ROOT / constants.DOTA_DATASET / "val_split.txt")
    parser.add_argument("--out-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=constants.V2_SPLIT_SEED)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="recompute and verify the committed splits; write nothing")
    mode.add_argument("--force", action="store_true",
                      help="overwrite frozen splits (only before any v2 number is read)")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    splits, manifest = build_splits(args.t2_meta, args.dota_metadata, args.dota_split, args.seed)
    log_summary(manifest)
    if args.check:
        check_splits(splits, manifest, args.out_dir)
        LOGGER.info("Frozen splits in %s match a fresh recomputation", args.out_dir)
        return
    write_splits(splits, manifest, args.out_dir, args.force)
    LOGGER.info("Wrote v2 splits to %s", args.out_dir)


if __name__ == "__main__":
    main()
