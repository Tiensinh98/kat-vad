"""MSAD preprocessor (spec §7.2) — traffic slice or the entire MSAD benchmark.

Parses the MSAD annotation table

    name    scenario    total_frames    anomaly_start_frame    anomaly_end_frame

(tab-, comma- or multi-space-separated, optional header) and emits the
standardized label files of the data-layout contract:

    data/MSAD/labels_train.json           {video_id: 0|1}
    data/MSAD/frame_labels_test.json      {video_id: [0,1,...]}  per SAMPLED frame
    data/MSAD/defs.json                   global class-name list
    data/MSAD/meta.json                   per-video scenario/class/split/window

Weak supervision is preserved: anomaly windows of *train* videos are written to
``meta.json`` for diagnostics only, never into ``labels_train.json``.

Split defaults follow MSAD protocol ii proportions (abnormal 120/240 = 0.5 test,
normal 120/480 = 0.25 test), seeded and stratified by (label, scenario); an
explicit ``--split-file`` (JSON list or one-id-per-line text) overrides it.

Full-MSAD mode: the official benchmark ships temporal windows for the test
split only — abnormal *train* videos carry a video-level label in their name
(``Fighting_012``). ``--infer-abnormal-from-name`` marks such window-less rows
abnormal by class-name prefix (window stays unknown; weak supervision).
Window-less abnormal videos are refused in the test split (frame-level eval
would be silently wrong) — pin them to train via ``--split-file``.

CLI::

    python -m core.data.msad --annotation anno.tsv [--out-dir data/MSAD]
        [--scenarios traffic] [--split-file test_ids.txt] [--one-indexed]
        [--end-exclusive] [--stride 8] [--seed 2024]
        [--infer-abnormal-from-name]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from core import constants
from core.data.definitions import DATASET_CLS_DEFS

LOGGER = logging.getLogger(__name__)

# Spec §7.2 traffic slice for the traffic-focused experiments.
TRAFFIC_SCENARIOS = frozenset(
    {
        "highway",  #
        "road",
        "street highview",  #
        "parking lot",
        "parkinglot",  #
        "pedestrian street",
        "sidewalk",  #
        "shop",  #
        "restaurant",  #
        "train",  #
        "frontdoor",  #
    }
)

# Known MSAD anomaly classes (definitions.py); used for name-prefix inference.
MSAD_CLASS_NAMES = tuple(DATASET_CLS_DEFS["msad"].keys())

NORMAL_CLASS = "Normal"
ABNORMAL_CLASS = "Abnormal"  # fallback when the class is not inferable

TEST_RATIO_ABNORMAL = 0.5  # protocol ii: 120 of 240 abnormal in test
TEST_RATIO_NORMAL = 0.25  # protocol ii: 120 of 480 normal in test

_HEADER_HINTS = ("name", "scenario", "frame")


@dataclass
class VideoRecord:
    """One row of the MSAD annotation table, index-adjusted to 0-based frames.

    ``anomaly_start``/``anomaly_end`` are inclusive 0-based raw-frame indices;
    both are ``None`` for normal videos.
    """

    video_id: str
    scenario: str
    total_frames: int
    anomaly_start: int | None
    anomaly_end: int | None
    class_name: str

    @property
    def is_abnormal(self) -> bool:
        # class_name covers window-less abnormal rows (--infer-abnormal-from-name)
        return self.anomaly_start is not None or self.class_name != NORMAL_CLASS


def _split_row(line: str) -> list[str]:
    """Split one annotation row: tab, then comma, then 2+ spaces.

    Single-space splitting is refused because scenario names contain spaces
    ("parking lot"); such files must be re-delimited by the user.
    """
    if "\t" in line:
        return [c.strip() for c in line.split("\t") if c.strip()]
    if "," in line:
        return [c.strip() for c in line.split(",") if c.strip()]
    cells = [c.strip() for c in re.split(r"\s{2,}", line) if c.strip()]
    if len(cells) < 3:
        raise ValueError(
            f"Cannot split annotation row {line!r}: use tab, comma or 2+ spaces "
            "between columns (scenario names may contain single spaces)"
        )
    return cells


def _parse_frame_field(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _name_class(video_id: str) -> str:
    """Longest MSAD class whose normalized name prefixes ``video_id`` ('' if none)."""
    normalized = re.sub(r"[\s_-]+", "", video_id).lower()
    best_cls, best_len = "", 0
    for cls in MSAD_CLASS_NAMES:
        key = re.sub(r"[\s_-]+", "", cls).lower()
        if normalized.startswith(key) and len(key) > best_len:
            best_cls, best_len = cls, len(key)
    return best_cls


def infer_class_name(video_id: str, is_abnormal: bool) -> str:
    """Infer an MSAD class from the video name prefix; fall back to Normal/Abnormal."""
    best = _name_class(video_id)
    if best and (best != NORMAL_CLASS) == is_abnormal:
        return best
    return ABNORMAL_CLASS if is_abnormal else NORMAL_CLASS


def parse_annotation_file(
    path: Path,
    one_indexed: bool = False,
    end_exclusive: bool = False,
    infer_abnormal_from_name: bool = False,
) -> list[VideoRecord]:
    """Parse the 5-column MSAD annotation table into :class:`VideoRecord` rows.

    A row is *normal* when the window fields are missing, non-numeric or
    negative, or when the (index-adjusted) window is empty — unless
    ``infer_abnormal_from_name`` is set and the video name prefix matches a
    known MSAD anomaly class, in which case the row is abnormal with an
    unknown window (full-MSAD train side: video-level labels only).
    """
    records: list[VideoRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cells = _split_row(line)
            if len(cells) < 3:
                raise ValueError(
                    f"{path}:{line_no}: expected >=3 columns, got {cells!r}"
                )
            if _parse_frame_field(cells[2]) is None:
                if line_no == 1 and any(h in line.lower() for h in _HEADER_HINTS):
                    continue  # header row
                raise ValueError(
                    f"{path}:{line_no}: non-numeric total_frames {cells[2]!r}"
                )

            video_id = Path(cells[0]).stem
            scenario = cells[1].lower()
            total_frames = int(cells[2])
            if total_frames <= 0:
                raise ValueError(f"{path}:{line_no}: total_frames must be > 0")

            start = _parse_frame_field(cells[3]) if len(cells) > 3 else None
            end = _parse_frame_field(cells[4]) if len(cells) > 4 else None
            if start is not None and end is not None and start >= 0 and end >= 0:
                if one_indexed:
                    start, end = start - 1, end - 1
                if end_exclusive:
                    end -= 1
                end = min(end, total_frames - 1)
                if start < 0 or end < start:
                    start, end = None, None  # empty window -> normal
            else:
                start, end = None, None

            class_name = infer_class_name(video_id, start is not None)
            if infer_abnormal_from_name and start is None:
                name_cls = _name_class(video_id)
                if name_cls and name_cls != NORMAL_CLASS:
                    class_name = name_cls
            records.append(
                VideoRecord(
                    video_id=video_id,
                    scenario=scenario,
                    total_frames=total_frames,
                    anomaly_start=start,
                    anomaly_end=end,
                    class_name=class_name,
                )
            )
    if not records:
        raise ValueError(f"No annotation rows parsed from {path}")
    LOGGER.info("Parsed %d annotation rows from %s", len(records), path)
    return records


def filter_scenarios(
    records: list[VideoRecord], scenarios: frozenset[str] | None
) -> list[VideoRecord]:
    """Keep only records whose scenario is in ``scenarios`` (None = keep all)."""
    if scenarios is None:
        return records
    kept = [r for r in records if r.scenario in scenarios]
    LOGGER.info("Scenario filter kept %d/%d videos", len(kept), len(records))
    if not kept:
        raise ValueError(f"Scenario filter removed every video: {sorted(scenarios)}")
    return kept


def num_sampled_frames(total_frames: int, stride: int) -> int:
    """Number of frames produced by ``range(0, total_frames, stride)`` sampling."""
    return (total_frames + stride - 1) // stride


def sampled_frame_labels(record: VideoRecord, stride: int) -> list[int]:
    """Per-sampled-frame 0/1 labels: frame ``i`` covers raw index ``i*stride``."""
    length = num_sampled_frames(record.total_frames, stride)
    start, end = record.anomaly_start, record.anomaly_end
    if start is None or end is None:
        return [0] * length
    return [1 if start <= i * stride <= end else 0 for i in range(length)]


def split_records(
    records: list[VideoRecord],
    seed: int = constants.SEED,
    test_ratio_abnormal: float = TEST_RATIO_ABNORMAL,
    test_ratio_normal: float = TEST_RATIO_NORMAL,
    split_file: Path | None = None,
) -> tuple[list[VideoRecord], list[VideoRecord]]:
    """Deterministic train/test split.

    With ``split_file`` the listed ids form the test set. Otherwise a seeded
    shuffle within each (label, scenario) group sends ``test_ratio_*`` of the
    group to test.
    """
    if split_file is not None:
        test_ids = _read_split_file(split_file)
        unknown = test_ids - {r.video_id for r in records}
        if unknown:
            raise ValueError(
                f"Split file lists unknown video ids: {sorted(unknown)[:5]}"
            )
        train = [r for r in records if r.video_id not in test_ids]
        test = [r for r in records if r.video_id in test_ids]
    else:
        # deterministic split shuffling, not security-sensitive
        rng = random.Random(seed)  # nosec B311
        groups: dict[tuple[bool, str], list[VideoRecord]] = {}
        for rec in records:
            groups.setdefault((rec.is_abnormal, rec.scenario), []).append(rec)
        train, test = [], []
        for (is_abn, _scenario), group in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1])
        ):
            group = sorted(group, key=lambda r: r.video_id)
            rng.shuffle(group)
            ratio = test_ratio_abnormal if is_abn else test_ratio_normal
            n_test = round(len(group) * ratio)
            test.extend(group[:n_test])
            train.extend(group[n_test:])
    if not train or not any(r.is_abnormal for r in train):
        raise ValueError("Train split has no abnormal videos; adjust ratios/split file")
    LOGGER.info(
        "Split: train=%d (%d abnormal) test=%d (%d abnormal)",
        len(train),
        sum(r.is_abnormal for r in train),
        len(test),
        sum(r.is_abnormal for r in test),
    )
    return train, test


def _read_split_file(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        ids = json.loads(text)
        if not isinstance(ids, list):
            raise ValueError(f"Split file {path} must be a JSON list of video ids")
        return {Path(str(i)).stem for i in ids}
    return {Path(line.strip()).stem for line in text.splitlines() if line.strip()}


def class_name_list(records: list[VideoRecord]) -> list[str]:
    """Global class-name list for defs.json: Normal first, then sorted rest."""
    names = {r.class_name for r in records}
    names.discard(NORMAL_CLASS)
    return [NORMAL_CLASS, *sorted(names)]


def write_standard_files(
    train: list[VideoRecord],
    test: list[VideoRecord],
    out_dir: Path,
    stride: int = constants.FRAME_STRIDE,
) -> None:
    """Write labels_train / frame_labels_test / defs / meta into ``out_dir``."""
    windowless = [r.video_id for r in test if r.is_abnormal and r.anomaly_start is None]
    if windowless:
        raise ValueError(
            "Abnormal test videos without an anomaly window (frame-level eval "
            f"would be silently wrong): {sorted(windowless)[:5]}... Pin them to "
            "the train split via --split-file or add temporal annotations."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_train = {r.video_id: int(r.is_abnormal) for r in train}
    frame_labels_test = {r.video_id: sampled_frame_labels(r, stride) for r in test}
    defs = class_name_list(train + test)
    meta = {
        r.video_id: {**asdict(r), "split": split}
        for split, recs in (("train", train), ("test", test))
        for r in recs
    }

    for filename, payload in (
        (constants.LABELS_TRAIN_FILENAME, labels_train),
        (constants.FRAME_LABELS_TEST_FILENAME, frame_labels_test),
        (constants.DEFS_FILENAME, defs),
        (constants.META_FILENAME, meta),
    ):
        target = out_dir / filename
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        LOGGER.info("Wrote %s", target)


def preprocess(
    annotation: Path,
    out_dir: Path,
    scenarios: frozenset[str] | None = None,
    stride: int = constants.FRAME_STRIDE,
    seed: int = constants.SEED,
    split_file: Path | None = None,
    one_indexed: bool = False,
    end_exclusive: bool = False,
    test_ratio_abnormal: float = TEST_RATIO_ABNORMAL,
    test_ratio_normal: float = TEST_RATIO_NORMAL,
    infer_abnormal_from_name: bool = False,
) -> tuple[list[VideoRecord], list[VideoRecord]]:
    """Full preprocessing pipeline; returns the (train, test) records."""
    records = parse_annotation_file(
        annotation, one_indexed, end_exclusive, infer_abnormal_from_name
    )
    records = filter_scenarios(records, scenarios)
    train, test = split_records(
        records, seed, test_ratio_abnormal, test_ratio_normal, split_file
    )
    write_standard_files(train, test, out_dir, stride)
    return train, test


def _parse_scenarios(value: str) -> frozenset[str] | None:
    if value == "all":
        return None
    if value == "traffic":
        return TRAFFIC_SCENARIOS
    return frozenset(s.strip().lower() for s in value.split(",") if s.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--annotation", type=Path, required=True, help="annotation table"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=constants.DATA_ROOT / constants.MSAD_DATASET,
        help="output directory (default: data/MSAD)",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help="'all', 'traffic' (spec §7.2 slice) or comma-separated scenario names",
    )
    parser.add_argument("--split-file", type=Path, default=None, help="test video ids")
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--seed", type=int, default=constants.SEED)
    parser.add_argument(
        "--one-indexed",
        action="store_true",
        help="annotation frame indices start at 1 (default: 0)",
    )
    parser.add_argument(
        "--end-exclusive",
        action="store_true",
        help="anomaly end frame is exclusive (default: inclusive)",
    )
    parser.add_argument(
        "--infer-abnormal-from-name",
        action="store_true",
        help="window-less rows are abnormal when the name prefix matches a "
        "known MSAD anomaly class (full MSAD: train side has video-level "
        "labels only)",
    )
    parser.add_argument(
        "--test-ratio-abnormal", type=float, default=TEST_RATIO_ABNORMAL
    )
    parser.add_argument("--test-ratio-normal", type=float, default=TEST_RATIO_NORMAL)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_arg_parser().parse_args(argv)
    preprocess(
        annotation=args.annotation,
        out_dir=args.out_dir,
        scenarios=_parse_scenarios(args.scenarios),
        stride=args.stride,
        seed=args.seed,
        split_file=args.split_file,
        one_indexed=args.one_indexed,
        end_exclusive=args.end_exclusive,
        test_ratio_abnormal=args.test_ratio_abnormal,
        test_ratio_normal=args.test_ratio_normal,
        infer_abnormal_from_name=args.infer_abnormal_from_name,
    )


if __name__ == "__main__":
    main()
