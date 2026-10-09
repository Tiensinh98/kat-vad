"""Nexar collision prediction: annotation parsing (plan ``katvad-v2-nexar-feasibility.md`` N0).

The HF release (``nexar-ai/nexar_collision_prediction``, gated) ships 1,500 training videos as
``train/{positive,negative}/{id}.mp4`` with one ``metadata.csv`` per class folder (the
``videofolder`` layout, so the video column is ``file_name``). The label is the **folder**, not
a column. Positives carry ``time_of_alert`` and ``time_of_event`` in seconds; negatives leave
them empty. There is **no frame span and no end-of-anomaly time** -- the frame-level label is a
rule fixed in N1, not something this module invents.

Nothing here reads a pixel. Malformed annotations are kept and *counted*
(:func:`annotation_issues`), never silently dropped (lessons C10, C18): which of them a corpus
keeps is a pre-registered decision, not a parser default.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

from core import constants

FILE_NAME_COLUMN = "file_name"
EVENT_COLUMN = "time_of_event"
ALERT_COLUMN = "time_of_alert"
_MISSING_TOKENS = frozenset({"", "none", "nan", "null"})

ISSUE_MISSING_TIMES = "positive_missing_times"
ISSUE_NEGATIVE_HAS_TIMES = "negative_has_times"
ISSUE_ALERT_AFTER_EVENT = "alert_after_event"
ISSUE_NEGATIVE_TIME = "negative_time"


@dataclass(frozen=True)
class NexarRecord:
    """One training video's annotation row."""

    video_id: str
    label: int
    class_dir: str
    time_of_event: float | None
    time_of_alert: float | None
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def relative_path(self) -> str:
        """Path of the mp4 inside the dataset root."""
        return f"{constants.NEXAR_TRAIN_DIR}/{self.class_dir}/{self.video_id}.mp4"


def parse_time(raw: str | None) -> float | None:
    """A seconds value, or ``None`` for the empty / ``None`` / ``nan`` cells negatives carry."""
    if raw is None or raw.strip().lower() in _MISSING_TOKENS:
        return None
    value = float(raw)
    if math.isnan(value):
        return None
    return value


def video_id_from_file_name(file_name: str) -> str:
    """``00042.mp4`` -> ``00042``; anything that is not a bare mp4 name raises."""
    path = Path(file_name)
    if path.suffix.lower() != ".mp4" or path.parent != Path(".") or not path.stem:
        raise ValueError(f"not a bare .mp4 file name: {file_name!r}")
    return path.stem


def parse_metadata(path: Path, class_dir: str) -> list[NexarRecord]:
    """Rows of one class folder's ``metadata.csv``; the label comes from ``class_dir``."""
    if class_dir not in constants.NEXAR_CLASS_DIRS:
        raise ValueError(
            f"unknown class folder {class_dir!r}; expected {constants.NEXAR_CLASS_DIRS}"
        )
    label = constants.NEXAR_CLASS_DIRS[class_dir]
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or FILE_NAME_COLUMN not in reader.fieldnames:
            raise ValueError(f"{path}: no {FILE_NAME_COLUMN!r} column in {reader.fieldnames}")
        records = []
        for row in reader:
            extras = {
                k: (v or "")
                for k, v in row.items()
                if k not in (FILE_NAME_COLUMN, EVENT_COLUMN, ALERT_COLUMN) and k is not None
            }
            records.append(
                NexarRecord(
                    video_id=video_id_from_file_name(row[FILE_NAME_COLUMN]),
                    label=label,
                    class_dir=class_dir,
                    time_of_event=parse_time(row.get(EVENT_COLUMN)),
                    time_of_alert=parse_time(row.get(ALERT_COLUMN)),
                    extras=extras,
                )
            )
    return records


def load_train_records(root: Path) -> list[NexarRecord]:
    """Both class folders of ``root/train``; a video id in two rows (or folders) raises."""
    records: list[NexarRecord] = []
    for class_dir in sorted(constants.NEXAR_CLASS_DIRS):
        path = root / constants.NEXAR_TRAIN_DIR / class_dir / constants.NEXAR_METADATA_FILENAME
        records.extend(parse_metadata(path, class_dir))
    seen: dict[str, str] = {}
    for record in records:
        if record.video_id in seen:
            raise ValueError(
                f"video id {record.video_id} appears twice ({seen[record.video_id]}, "
                f"{record.class_dir})"
            )
        seen[record.video_id] = record.class_dir
    return sorted(records, key=lambda r: r.video_id)


def annotation_issues(record: NexarRecord) -> list[str]:
    """Every rule the row breaks; empty for a clean row. Counted by the census, never fixed here."""
    issues: list[str] = []
    times = (record.time_of_event, record.time_of_alert)
    if record.label == 1:
        if record.time_of_event is None or record.time_of_alert is None:
            issues.append(ISSUE_MISSING_TIMES)
        elif record.time_of_alert > record.time_of_event:
            issues.append(ISSUE_ALERT_AFTER_EVENT)
    elif any(t is not None for t in times):
        issues.append(ISSUE_NEGATIVE_HAS_TIMES)
    if any(t is not None and t < 0 for t in times):
        issues.append(ISSUE_NEGATIVE_TIME)
    return issues
