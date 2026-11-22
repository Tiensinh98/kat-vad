"""ACCIDENT (``KIM_RUN.md`` §39): the frozen split and the windowed training corpus.

ACCIDENT ships 2,027 real CCTV accident clips (``metadata-real.csv``: one ``accident_time`` per
clip, no end time, no normal video). Reads metadata and the CLIP ``s1`` cache's row counts only;
never a score.

``split`` -> ``core/splits/accident/{accident_train,accident_val}.txt`` + manifest
    Grouped by YouTube source (the id minus its ``_NN`` suffix), stratified by ``type``, 90 / 10,
    seeded (§39 S). ACCIDENT's own IID / geo benchmark splits are not used.
``corpus`` -> ``data/ACCIDENT_window/``
    Every clip is cut at ``t_impact + ACCIDENT_POST_IMPACT_S`` (the aftermath is unlabeled) and
    labelled by time: a 30 fps CFR frame ``f`` is abnormal iff ``t_impact <= f / 30``. Windows are
    Nexar's (20 rows at s8, hop 8, no cap), through T2's :func:`core.data.dada.write_windowed`;
    sources listed in ``--exclude-ids`` (the TAD dedup, §39 D) are dropped from both splits.

CLI::

    python -m core.tools.accident_build split --metadata data/ACCIDENT/metadata-real.csv
    python -m core.tools.accident_build corpus --metadata .../metadata-real.csv \\
        --clip-s1-dir cache/clip/ACCIDENT_s1_ncc --exclude-ids .../accident_dedup_exclude_ids.txt \\
        --out-dir data/ACCIDENT_window --report .../build_report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.dada import DadaRecord, write_windowed
from core.data.v2_splits import lines_sha1
from core.tools.kill_switch_probe import write_json_atomic
from core.tools.nexar_build import cache_frames, epoch_plan

LOGGER = logging.getLogger(__name__)

SUBCOMMAND_SPLIT = "split"
SUBCOMMAND_CORPUS = "corpus"
META_KEYS = ("type", "scene_layout", "weather", "day_time", "quality", "region")
SPLIT_RULE = (
    "grouped by YouTube source (id minus its _NN suffix), stratified by the source's first "
    f"clip type, val = round({constants.ACCIDENT_VAL_FRACTION} x sources per type), "
    f"seed {constants.ACCIDENT_SPLIT_SEED}"
)


def video_id(row: dict[str, str]) -> str:
    """``real_videos/<id>.mp4`` -> ``<id>``."""
    return Path(row["path"]).stem


def source_of(clip_id: str) -> str:
    """The upload a clip was cut from: ``<upload>[_<segment>]_<NN>`` -> ``<upload>``.

    257 ids carry a segment index of one upload (``3Xd6PxEvbNk_3_00``, ``kaggle-..._1_00``); it is
    stripped too, so every segment of an upload lands in one split. Over-grouping (a YouTube id
    that itself ends in ``_<digits>``) only merges sources, which can never leak across splits.
    """
    sep = constants.ACCIDENT_SOURCE_SEPARATOR
    head = clip_id.rpartition(sep)[0] or clip_id
    base, found, tail = head.rpartition(sep)
    return base if found and base and tail.isdigit() else head


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    """``{id: csv row}``; a duplicate id raises."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = {video_id(r): r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"{path}: {len(rows) - len(out)} duplicate clip ids")
    return out


def file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes(), usedforsecurity=False).hexdigest()


def split_sources(
    rows: dict[str, dict[str, str]],
    val_fraction: float = constants.ACCIDENT_VAL_FRACTION,
    seed: int = constants.ACCIDENT_SPLIT_SEED,
) -> dict[str, list[str]]:
    """``{split: sorted clip ids}``: whole sources go to one split, per-type val share."""
    clips: dict[str, list[str]] = defaultdict(list)
    for clip_id in sorted(rows):
        clips[source_of(clip_id)].append(clip_id)
    by_type: dict[str, list[str]] = defaultdict(list)
    for source in sorted(clips):
        by_type[rows[clips[source][0]]["type"]].append(source)
    rng = np.random.default_rng(seed)
    val_sources: set[str] = set()
    for kind in sorted(by_type):
        sources = by_type[kind]
        order = rng.permutation(len(sources))
        val_sources.update(sources[i] for i in order[: round(val_fraction * len(sources))])
    train, val = constants.ACCIDENT_SPLIT_TRAIN, constants.ACCIDENT_SPLIT_VAL
    split: dict[str, list[str]] = {train: [], val: []}
    for source, ids in clips.items():
        split[val if source in val_sources else train].extend(ids)
    return {name: sorted(ids) for name, ids in split.items()}


def write_splits(metadata: Path, split_dir: Path) -> dict[str, Any]:
    """Freeze the split files and their manifest (ids sha1 + metadata sha1)."""
    rows = read_metadata(metadata)
    split = split_sources(rows)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in split.items():
        (split_dir / f"{name}{constants.V2_SPLIT_FILE_SUFFIX}").write_text(
            "".join(f"{i}\n" for i in ids), encoding="utf-8")
    manifest = {
        "rule": SPLIT_RULE,
        "metadata": {"name": metadata.name, "sha1": file_sha1(metadata), "clips": len(rows)},
        "splits": {
            name: {
                "clips": len(ids),
                "sources": len({source_of(i) for i in ids}),
                "sha1": lines_sha1(ids),
                "per_type": dict(sorted(
                    {k: sum(1 for i in ids if rows[i]["type"] == k)
                     for k in {r["type"] for r in rows.values()}}.items())),
            }
            for name, ids in split.items()
        },
    }
    write_json_atomic(split_dir / constants.ACCIDENT_SPLIT_MANIFEST_FILENAME, manifest)
    return manifest


def load_accident_split(name: str, split_dir: Path = constants.ACCIDENT_SPLITS_DIR) -> list[str]:
    """Ids of a frozen ACCIDENT split, verified against its manifest's sha1."""
    manifest = json.loads(
        (split_dir / constants.ACCIDENT_SPLIT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    path = split_dir / f"{name}{constants.V2_SPLIT_FILE_SUFFIX}"
    ids = sorted(line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip())
    if lines_sha1(ids) != manifest["splits"][name]["sha1"]:
        raise ValueError(f"{path} does not match {constants.ACCIDENT_SPLIT_MANIFEST_FILENAME}")
    return ids


def cut_frames(row: dict[str, str], frames: int,
               post_impact_s: float = constants.ACCIDENT_POST_IMPACT_S,
               fps: int = constants.ACCIDENT_FPS) -> int:
    """CFR frames kept: every frame with ``f / fps <= t_impact + post_impact_s``."""
    cut_s = float(row["accident_time"]) + post_impact_s
    end = math.floor(cut_s * fps + constants.ACCIDENT_CFR_TIME_EPS)
    return max(0, min(frames, end + 1))


def windowable(row: dict[str, str], fps: int = constants.ACCIDENT_FPS) -> bool:
    """Can this clip's cut give one window? Metadata duration, with a margin (no decode).

    ``accident_extract`` skips the others (they give no window, train or val) and the corpus
    leaves them out, so both see the same clips.
    """
    frames = math.floor(float(row["duration"]) * fps + constants.ACCIDENT_CFR_TIME_EPS) + 1
    rows = math.ceil(cut_frames(row, frames, fps=fps) / constants.FRAME_STRIDE)
    return rows >= constants.NEXAR_WINDOW_LENGTH - constants.ACCIDENT_SKIP_MARGIN_ROWS


def make_record(clip_id: str, row: dict[str, str], frames: int,
                post_impact_s: float = constants.ACCIDENT_POST_IMPACT_S,
                fps: int = constants.ACCIDENT_FPS) -> DadaRecord:
    """A :class:`DadaRecord` over the cut clip; the span runs from the impact to the cut."""
    kept = cut_frames(row, frames, post_impact_s, fps)
    if kept == 0:
        raise ValueError(f"{clip_id}: no CFR frame before t_impact + {post_impact_s} s")
    impact = min(1.0, float(row["accident_time"]) * fps / kept)
    return DadaRecord(
        video_id=clip_id,
        folder_name=clip_id,
        class_name=constants.DADA_CLASS_NAME,
        fault_label=constants.ACCIDENT_FAULT_LABEL,
        total_frames=kept,
        span=(impact, 1.0),
        accident_frac=impact,
    )


def read_id_lines(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def run_corpus(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_metadata(args.metadata)
    train = load_accident_split(constants.ACCIDENT_SPLIT_TRAIN, args.split_dir)
    val = load_accident_split(constants.ACCIDENT_SPLIT_VAL, args.split_dir)
    excluded = read_id_lines(args.exclude_ids)
    unknown = sorted(excluded - set(rows))
    if unknown:
        raise ValueError(f"{len(unknown)} excluded ids are not ACCIDENT clips: {unknown[:5]}")
    short = sorted(v for v in train + val if not windowable(rows[v]))
    train_kept = [v for v in train if v not in excluded and windowable(rows[v])]
    val_kept = [v for v in val if v not in excluded and windowable(rows[v])]
    frames = cache_frames(args.clip_s1_dir, train_kept + val_kept)
    records = {v: make_record(v, rows[v], frames[v], args.post_impact_s)
               for v in train_kept + val_kept}
    extra: dict[str, dict[str, object]] = {
        v: {**{k: rows[v].get(k) for k in META_KEYS}, "youtube_source": source_of(v)}
        for v in records}
    write_windowed(
        args.out_dir,
        [records[v] for v in train_kept],
        [records[v] for v in val_kept],
        constants.FRAME_STRIDE,
        {constants.DADA_CLASS_NAME},
        constants.NEXAR_WINDOW_LENGTH,
        constants.NEXAR_WINDOW_STRIDE,
        constants.WINDOW_MIN_POSITIVE,
        "drop",
        constants.NEXAR_WINDOW_MAX_PER_CLIP,
        extra,
    )
    def read(name: str) -> Any:
        return json.loads((args.out_dir / name).read_text(encoding="utf-8"))

    labels = read(constants.LABELS_TRAIN_FILENAME)
    windows = read(constants.WINDOWS_FILENAME)
    test = read(constants.FRAME_LABELS_TEST_FILENAME)
    train_sources = {windows[w]["source"] for w in labels}
    leaked = sorted((train_sources & set(val)) | (train_sources & excluded))
    if leaked:
        raise ValueError(f"{len(leaked)} val / excluded clips have train windows: {leaked[:5]}")
    stats = {
        "train_clips": len(train), "val_clips": len(val),
        "excluded_clips": len(excluded),
        "train_clips_after_dedup": len(train_kept), "val_clips_after_dedup": len(val_kept),
        "train_items": len(labels), "train_abnormal": sum(labels.values()),
        "train_sources_with_windows": len(train_sources),
        "val_windows": len(test),
        "val_two_class_windows": sum(1 for lab in test.values() if 0 < sum(lab) < len(lab)),
        "cut_frames_median": int(np.median([r.total_frames for r in records.values()])),
    }
    report = {
        "rule": f"KIM_RUN.md §39 A3: abnormal = [t_impact, t_impact + {args.post_impact_s} s], "
                "later frames cut; 30 fps CFR; windows 20 rows at s8, hop 8, no cap",
        "metadata_sha1": file_sha1(args.metadata),
        "splits": {"train": lines_sha1(train), "val": lines_sha1(val)},
        "excluded_ids_sha1": lines_sha1(sorted(excluded)),
        "skipped_short_clips": len(short),
        "construction": {"dir": str(args.out_dir), **stats},
        "plan": epoch_plan(stats["train_abnormal"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.report, report)
    LOGGER.info("ACCIDENT_window: %s, plan %s -> %s", stats, report["plan"], args.report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser(SUBCOMMAND_SPLIT, help="freeze core/splits/accident")
    split.add_argument("--metadata", type=Path, required=True)
    split.add_argument("--split-dir", type=Path, default=constants.ACCIDENT_SPLITS_DIR)
    corpus = sub.add_parser(SUBCOMMAND_CORPUS, help="build data/ACCIDENT_window")
    corpus.add_argument("--metadata", type=Path, required=True)
    corpus.add_argument("--clip-s1-dir", type=Path, required=True)
    corpus.add_argument("--split-dir", type=Path, default=constants.ACCIDENT_SPLITS_DIR)
    corpus.add_argument("--exclude-ids", type=Path, default=None, help="dedup exclusions (§39 D)")
    corpus.add_argument("--post-impact-s", type=float, default=constants.ACCIDENT_POST_IMPACT_S)
    corpus.add_argument("--out-dir", type=Path, required=True)
    corpus.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == SUBCOMMAND_SPLIT:
        LOGGER.info("splits: %s", write_splits(args.metadata, args.split_dir)["splits"])
    else:
        run_corpus(args)


if __name__ == "__main__":
    main()
