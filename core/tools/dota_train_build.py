"""Recovered DoTA train clips (``KIM_RUN.md`` §55): the windowed training corpus.

Every clip is an accident clip; its normal windows are cut from its own frames outside the
annotated span ``[anomaly_start, anomaly_end)`` (DoTA frames, half-open). Row ``j`` at stride 3 is
DoTA frame ``3j`` and is abnormal iff that frame is in the span. Windows: 12 rows (3.6 s), hop 6;
a window is abnormal iff it holds ``WINDOW_MIN_POSITIVE`` abnormal rows. No test split is written
(the arms are read by ``protocol_b_eval``), so ``frame_labels_test.json`` is empty. CLI::

    python -m core.tools.dota_train_build --annotations DoTA_annotations.zip \\
        --clip-dir cache/clip/DoTA_train_s3_ncc --out-dir data/DoTA_train_window \\
        --report outputs/kim_dota_train/REPORTS/build_report.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.data.dada import DadaRecord, plan_record_windows, sampled_frame_labels, window_label
from core.data.dataset_files import (
    class_name_list,
    num_sampled_frames,
    write_dataset_files,
    write_train_ids,
    write_windows,
)
from core.data.v2_splits import lines_sha1
from core.tools.dota_train_extract import train_clips
from core.tools.dota_train_match import DotaClip, read_dota
from core.tools.kill_switch_probe import write_json_atomic
from core.tools.nexar_build import cache_frames, epoch_plan

LOGGER = logging.getLogger(__name__)

WEAK_MODE = "drop"  # every clip is annotated; nothing is weak


def make_record(clip: DotaClip) -> DadaRecord:
    """The span is placed on rows, so row ``j`` is abnormal iff ``start <= 3j < end`` exactly."""
    stride = constants.DOTA_TRAIN_STRIDE
    rows = num_sampled_frames(clip.frames, stride)
    start, end = (num_sampled_frames(min(f, clip.frames), stride) for f in (clip.start, clip.end))
    if not 0 <= clip.start < clip.end:
        raise ValueError(f"{clip.dota_id}: annotated span [{clip.start}, {clip.end})")
    return DadaRecord(
        video_id=clip.dota_id,
        folder_name=clip.dota_id,
        class_name=constants.DADA_CLASS_NAME,
        fault_label=constants.DOTA_TRAIN_FAULT_LABEL,
        total_frames=clip.frames,
        span=(start / rows, end / rows),
        accident_frac=start / rows,
    )


def build(out_dir: Path, clips: list[DotaClip]) -> dict[str, Any]:
    """Write the corpus files and return the counts."""
    records = [make_record(c) for c in clips]
    windows, frame_labels, by_window = plan_record_windows(
        records, constants.DOTA_TRAIN_STRIDE, constants.DOTA_TRAIN_WINDOW_LENGTH,
        constants.DOTA_TRAIN_WINDOW_HOP, constants.WINDOW_MIN_POSITIVE, WEAK_MODE,
        constants.NEXAR_WINDOW_MAX_PER_CLIP,
    )
    labels = {w: window_label(frame_labels[w], by_window[w], constants.WINDOW_MIN_POSITIVE)
              for w in windows}
    meta = {
        w: {
            "class_name": by_window[w].class_name, "fault_label": by_window[w].fault_label,
            "split": "train", "source": window.source, "start": window.start, "end": window.end,
            "total_frames": by_window[w].total_frames, "sampled_frames": len(frame_labels[w]),
            "positive_frames": sum(frame_labels[w]),
        }
        for w, window in sorted(windows.items())
    }
    write_dataset_files(out_dir, labels, {}, class_name_list({constants.DADA_CLASS_NAME}), meta)
    write_windows(out_dir, windows)
    write_train_ids(out_dir, sorted({w.source for w in windows.values()}))
    positive = {windows[w].source for w, y in labels.items() if y}
    normal = {windows[w].source for w, y in labels.items() if not y}
    row_labels = [sampled_frame_labels(r, constants.DOTA_TRAIN_STRIDE) for r in records]
    return {
        "clips": len(clips),
        "clips_with_windows": len(positive | normal),
        "clips_with_both_kinds": len(positive & normal),
        "clips_giving_normals": len(normal),
        "train_items": len(labels),
        "train_abnormal": sum(labels.values()),
        "train_normal": len(labels) - sum(labels.values()),
        "row_abnormal_share": round(sum(map(sum, row_labels)) / sum(map(len, row_labels)), 4),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    wanted = train_clips(args.map)
    annotated = read_dota(args.annotations)
    clips = [annotated[k] for k in wanted]
    wrong = [c.dota_id for c in clips if c.frames != int(wanted[c.dota_id]["dota_frames"])]
    if wrong:
        raise ValueError(f"{len(wanted)} clips: {len(wrong)} disagree with the map: {wrong[:5]}")
    rows = cache_frames(args.clip_dir, list(wanted))
    short = [c.dota_id for c in clips
             if rows[c.dota_id] != num_sampled_frames(c.frames, constants.DOTA_TRAIN_STRIDE)]
    if short:
        raise ValueError(f"{len(short)} clips: cached rows do not match the frames: {short[:5]}")
    stats = build(args.out_dir, clips)
    report = {
        "rule": "KIM_RUN.md §55",
        "excluded": list(constants.DOTA_TRAIN_EXCLUDED),
        "clip_ids_sha1": lines_sha1(sorted(wanted)),
        "geometry": {"stride": constants.DOTA_TRAIN_STRIDE,
                     "window_length": constants.DOTA_TRAIN_WINDOW_LENGTH,
                     "hop": constants.DOTA_TRAIN_WINDOW_HOP,
                     "min_positive": constants.WINDOW_MIN_POSITIVE},
        "construction": stats,
        "plan": epoch_plan(stats["train_abnormal"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.report, report)
    LOGGER.info("DoTA_train_window: %s, plan %s", stats, report["plan"])
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--annotations", type=Path, required=True, help="DoTA_annotations.zip")
    parser.add_argument("--map", type=Path, default=(
        constants.DOTA_TRAIN_SPLITS_DIR / constants.DOTA_TRAIN_MAP_FILENAME))
    parser.add_argument("--clip-dir", type=Path, required=True, help="CLIP rows at stride 3")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
