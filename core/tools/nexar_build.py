"""Nexar N4: the two training corpora (``whole`` / ``window``) and the A0 stride-8 cache.

Addendum §21. Reads the N0 census, the frozen Nexar splits and the N2 CLIP ``s1`` cache; never
reads ``nexar_test`` (it is sealed) and never reads a score.

Span rule (D-N1): a positive's native frame ``f`` is abnormal iff ``t_alert <= t_f <=
t_event + NEXAR_POST_EVENT_S`` with ``t_f = f * duration / F``; at stride ``k`` the same span is
applied as a normalized fraction of the sampled rows (:func:`core.data.dada.sampled_frame_labels`,
T2's arithmetic). ``F`` is the cache's row count -- the frames the features were computed on.

Constructions (both train at ``s1[::8]``, both labelled by the rule above):

``whole`` -> ``data/Nexar_whole/``
    ``labels_train`` = every ``nexar_train`` video with its video label (750-video negative pool
    included -- the arm exists to measure what that does). ``frame_labels_test`` = every
    ``nexar_val`` video at stride 8.
``window`` -> ``data/Nexar_window/``
    T2's construction: windows of 20 rows, hop 8, cut from the **positive** videos only (pre- and
    post-event windows are the negatives), labelled by their own rows
    (:func:`core.data.dada.write_windowed`). No per-video cap (constants).

``subsample`` writes ``cache/clip/Nexar_s8_ncc`` = ``Nexar_s1_ncc[::8]`` for A0 and the KNN key.

The ``plan`` block of ``build_report.json`` gives each construction's epochs for the shared step
budget (``NEXAR_STEP_BUDGET``): ``steps/epoch = ceil(2 * abnormal items / batch)`` (A9).

CLI::

    python -m core.tools.nexar_build corpus --census .../nexar_n0/census.json \\
        --clip-s1-dir cache/clip/Nexar_s1_ncc --out-root data \\
        --report .../nexar_n4/build_report.json
    python -m core.tools.nexar_build subsample --clip-s1-dir cache/clip/Nexar_s1_ncc \\
        --ids-file ids.txt --stride 8 --out-dir cache/clip/Nexar_s8_ncc
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.dada import DadaRecord, sampled_frame_labels, write_windowed
from core.data.dataset_files import (
    NORMAL_CLASS,
    class_name_list,
    write_dataset_files,
    write_test_ids,
    write_train_ids,
)
from core.data.v2_splits import lines_sha1, load_split
from core.metrics import frame_auc
from core.tools.feature_cache import save_array
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

FAULT_LABEL = "nexar_ego"  # every Nexar positive is an ego collision / near-collision
META_KEYS = ("scene", "weather", "light_conditions", "fps", "duration_s")
SUBCOMMAND_CORPUS = "corpus"
SUBCOMMAND_SUBSAMPLE = "subsample"


def span_seconds(row: dict[str, Any], post_event_s: float = constants.NEXAR_POST_EVENT_S) -> tuple[
    float, float
]:
    """``(t_alert, t_event + post)`` in seconds, clipped to the video (D-N1)."""
    end = min(float(row["duration_s"]), float(row["time_of_event"]) + post_event_s)
    return float(row["time_of_alert"]), end


def make_record(video_id: str, row: dict[str, Any], frames: int) -> DadaRecord:
    """A :class:`DadaRecord` so T2's window and label code applies unchanged."""
    if row["label"] == 0:
        return DadaRecord(video_id, video_id, NORMAL_CLASS, FAULT_LABEL, frames, None, None)
    duration = float(row["duration_s"])
    start, end = span_seconds(row)
    return DadaRecord(
        video_id=video_id,
        folder_name=video_id,
        class_name=constants.DADA_CLASS_NAME,
        fault_label=FAULT_LABEL,
        total_frames=frames,
        span=(start / duration, end / duration),
        accident_frac=float(row["time_of_event"]) / duration,
    )


def native_labels(row: dict[str, Any], frames: int) -> np.ndarray:
    """Per native frame 0/1 (all zero for a negative), ``t_f = f * duration / frames``."""
    if row["label"] == 0:
        return np.zeros(frames, dtype=np.int64)
    start, end = span_seconds(row)
    times = np.arange(frames) * float(row["duration_s"]) / frames
    return ((times >= start) & (times <= end)).astype(np.int64)


def cache_frames(clip_s1_dir: Path, ids: list[str]) -> dict[str, int]:
    """Rows per id in the s1 cache (header read only); a missing id raises (C10)."""
    missing = [v for v in ids if not (clip_s1_dir / f"{v}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ids have no s1 CLIP rows: {missing[:5]}")
    return {v: int(np.load(clip_s1_dir / f"{v}.npy", mmap_mode="r").shape[0]) for v in ids}


def video_meta(row: dict[str, Any], split: str, frames: int) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "class_name": constants.DADA_CLASS_NAME if row["label"] else NORMAL_CLASS,
        "split": "train" if split == constants.V2_SPLIT_NEXAR_TRAIN else "test",
        "nexar_split": split,
        "label": row["label"],
        "total_frames": frames,
        "time_of_alert": row.get("time_of_alert"),
        "time_of_event": row.get("time_of_event"),
    }
    for key in META_KEYS:
        meta[key] = row.get(key, row.get(f"meta_{key}"))
    return meta


def build_whole(
    out_dir: Path, rows: dict[str, dict[str, Any]], train: list[str], val: list[str],
    frames: dict[str, int],
) -> dict[str, Any]:
    """``whole``: video-level train labels, val videos at stride 8 as the eval set."""
    stride = constants.NEXAR_TRAIN_STRIDE
    labels_train = {v: int(rows[v]["label"]) for v in train}
    frame_labels = {
        v: sampled_frame_labels(make_record(v, rows[v], frames[v]), stride) for v in val
    }
    meta = {v: video_meta(rows[v], constants.V2_SPLIT_NEXAR_TRAIN, frames[v]) for v in train}
    meta.update({v: video_meta(rows[v], constants.V2_SPLIT_NEXAR_VAL, frames[v]) for v in val})
    write_dataset_files(
        out_dir, labels_train, frame_labels, class_name_list({constants.DADA_CLASS_NAME}), meta
    )
    write_train_ids(out_dir, sorted(train))
    write_test_ids(out_dir, sorted(val))
    vanished = sorted(v for v in val if rows[v]["label"] and not any(frame_labels[v]))
    return {
        "train_items": len(labels_train),
        "train_abnormal": sum(labels_train.values()),
        "val_videos": len(val),
        "val_positive_without_positive_row": vanished,
    }


def build_window(
    out_dir: Path, rows: dict[str, dict[str, Any]], train: list[str], val: list[str],
    frames: dict[str, int],
) -> dict[str, Any]:
    """``window``: T2's windows over the positive videos of each split."""
    train_pos = [make_record(v, rows[v], frames[v]) for v in train if rows[v]["label"] == 1]
    val_pos = [make_record(v, rows[v], frames[v]) for v in val if rows[v]["label"] == 1]
    write_windowed(
        out_dir,
        train_pos,
        val_pos,
        constants.NEXAR_TRAIN_STRIDE,
        {constants.DADA_CLASS_NAME},
        constants.NEXAR_WINDOW_LENGTH,
        constants.NEXAR_WINDOW_STRIDE,
        constants.WINDOW_MIN_POSITIVE,
        "drop",
        constants.NEXAR_WINDOW_MAX_PER_CLIP,
        {r.video_id: {k: rows[r.video_id].get(f"meta_{k}") for k in ("scene", "weather")}
         for r in train_pos + val_pos},
    )
    labels = json.loads((out_dir / constants.LABELS_TRAIN_FILENAME).read_text(encoding="utf-8"))
    test = json.loads((out_dir / constants.FRAME_LABELS_TEST_FILENAME).read_text(encoding="utf-8"))
    return {
        "train_items": len(labels),
        "train_abnormal": sum(labels.values()),
        "train_sources": len(train_pos),
        "val_windows": len(test),
        "val_two_class_windows": sum(1 for lab in test.values() if 0 < sum(lab) < len(lab)),
        "val_abnormal_windows": sum(1 for lab in test.values() if sum(lab) > 0),
    }


def epoch_plan(train_abnormal: int, batch_size: int = constants.BATCH_SIZE) -> dict[str, int]:
    """Epochs that spend ``NEXAR_STEP_BUDGET`` optimizer steps (DVS len = 2 x abnormal)."""
    steps_per_epoch = math.ceil(2 * train_abnormal / batch_size)
    epochs = max(1, round(constants.NEXAR_STEP_BUDGET / steps_per_epoch))
    return {
        "items_per_epoch": 2 * train_abnormal,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "steps": epochs * steps_per_epoch,
    }


def middle_ruler_macro(rows: dict[str, dict[str, Any]], ids: list[str], frames: dict[str, int]
                       ) -> float:
    """Macro AUC of ``-|t/T - 0.5|`` over positive videos: what position alone scores (printed)."""
    aucs = []
    for v in ids:
        if rows[v]["label"] != 1:
            continue
        labels = native_labels(rows[v], frames[v])
        if 0 < labels.sum() < len(labels):
            ruler = -np.abs(np.arange(frames[v]) / frames[v] - 0.5)
            aucs.append(frame_auc(ruler, labels))
    return float(np.mean(aucs))


def run_corpus(args: argparse.Namespace) -> dict[str, Any]:
    census = json.loads(args.census.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = census["videos"]
    train = load_split(constants.V2_SPLIT_NEXAR_TRAIN, args.split_dir)
    val = load_split(constants.V2_SPLIT_NEXAR_VAL, args.split_dir)
    frames = cache_frames(args.clip_s1_dir, train + val)
    mismatch = sorted(v for v in frames if frames[v] != rows[v].get("frames_header"))
    issues = sorted(v for v in train + val if rows[v]["issues"])
    if issues:
        raise ValueError(f"{len(issues)} train/val videos carry annotation issues: {issues[:5]}")
    report: dict[str, Any] = {
        "rule": "addendum §21 D-N1: abnormal = [t_alert, t_event + "
                f"{constants.NEXAR_POST_EVENT_S} s]",
        "splits": {"nexar_train": lines_sha1(train), "nexar_val": lines_sha1(val)},
        "frames_vs_census_header_mismatch": mismatch,
        "fps_below_29": sorted(v for v in train + val if float(rows[v]["fps"]) < 29.0),
        "val_middle_ruler_macro": round(middle_ruler_macro(rows, val, frames), 4),
        "constructions": {},
        "plan": {},
    }
    builders = {
        constants.NEXAR_CONSTRUCTION_WHOLE: build_whole,
        constants.NEXAR_CONSTRUCTION_WINDOW: build_window,
    }
    for name in args.constructions:
        out_dir = args.out_root / f"{constants.NEXAR_DATASET}_{name}"
        stats = builders[name](out_dir, rows, train, val, frames)
        report["constructions"][name] = {"dir": str(out_dir), **stats}
        report["plan"][name] = epoch_plan(stats["train_abnormal"])
        LOGGER.info("%s: %s, plan %s", name, stats, report["plan"][name])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.report, report)
    LOGGER.info("val middle-ruler macro %.4f; header mismatches %d -> %s",
                report["val_middle_ruler_macro"], len(mismatch), args.report)
    return report


def run_subsample(args: argparse.Namespace) -> int:
    """``out/{id}.npy = s1/{id}.npy[::stride]`` (resumable, atomic)."""
    ids = sorted(line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                 if line.strip())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for v in ids:
        target = args.out_dir / f"{v}.npy"
        if target.is_file():
            continue
        rows = np.load(args.clip_s1_dir / f"{v}.npy")[:: args.stride]
        save_array(target, np.ascontiguousarray(rows))
        written += 1
    LOGGER.info("subsampled %d / %d ids at stride %d -> %s", written, len(ids), args.stride,
                args.out_dir)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    corpus = sub.add_parser(SUBCOMMAND_CORPUS, help="build data/Nexar_{whole,window}")
    corpus.add_argument("--census", type=Path, required=True)
    corpus.add_argument("--clip-s1-dir", type=Path, required=True)
    corpus.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    corpus.add_argument("--out-root", type=Path, default=constants.DATA_ROOT)
    corpus.add_argument("--constructions", nargs="+", default=list(constants.NEXAR_CONSTRUCTIONS),
                        choices=constants.NEXAR_CONSTRUCTIONS)
    corpus.add_argument("--report", type=Path, required=True)
    sub_s = sub.add_parser(SUBCOMMAND_SUBSAMPLE, help="derive an s<k> CLIP cache from s1")
    sub_s.add_argument("--clip-s1-dir", type=Path, required=True)
    sub_s.add_argument("--ids-file", type=Path, required=True)
    sub_s.add_argument("--stride", type=int, default=constants.NEXAR_TRAIN_STRIDE)
    sub_s.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == SUBCOMMAND_CORPUS:
        run_corpus(args)
    else:
        run_subsample(args)


if __name__ == "__main__":
    main()
