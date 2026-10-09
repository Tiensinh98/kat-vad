"""Nexar N0 census: what is on disk, measured before any feature or score exists.

Plan ``.project/plans/katvad-v2-nexar-feasibility.md`` N0. Reads the two ``metadata.csv`` files
and the **container headers** of every training mp4 (PyAV, no decode), and writes
``census.json`` (one row per video + a summary) and ``census.md`` beside it.

What the summary prints, and why:

* **Length by class** + the length ruler AUC (duration alone predicting the label, lesson C28).
  The README says length depends on the dashcam model; if the models are not balanced across
  classes, whole-video length carries the label.
* **fps / resolution / codec by class.** The v2 geometry (VideoMAE 16 x 3 frames = 1.5 s, s3 =
  0.1 s) assumes 30 fps; a camera model off 30 fps changes it.
* **Header frames vs duration x fps**, and ``--verify-decode K`` videos decoded in full: a
  header count is a claim, not a measurement (C10).
* **Positives:** ``t_event / duration`` (the position prior the v2 report warns about),
  ``t_alert``, the alert -> event gap, and the abnormal frame share under
  ``[t_alert, t_event + d]`` for each ``d`` in ``NEXAR_POST_EVENT_PROBES_S`` -- informational
  input to N1's span rule, never a choice.
* **Annotation issues**, counted per rule (:func:`core.data.nexar.annotation_issues`) plus
  ``event_after_end`` (needs the probe).

No label is combined with any model output here; everything is metadata.

CLI::

    python -m core.tools.nexar_census --root /content/drive/MyDrive/Thesis/data/Nexar/raw \\
        --out-dir outputs/v2/REPORTS/nexar_n0 [--workers 8] [--verify-decode 20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import av
import numpy as np
from av.error import FFmpegError

from core import constants
from core.data.nexar import NexarRecord, annotation_issues, load_train_records
from core.metrics import frame_auc
from core.tools.feature_cache import part_path

LOGGER = logging.getLogger(__name__)

CENSUS_JSON = "census.json"
CENSUS_MD = "census.md"
ISSUE_EVENT_AFTER_END = "event_after_end"
ISSUE_PROBE_FAILED = "probe_failed"
DEFAULT_WORKERS = 8
DEFAULT_VERIFY_DECODE = 20
HEADER_FRAME_TOLERANCE = 1  # frames; |header - duration * fps| above this is flagged
MAX_CATEGORY_LEVELS = 20  # an extra column with more distinct values is not tabulated


# --------------------------------------------------------------------------- probing


def probe_video(path: Path) -> dict[str, Any]:
    """Container-header facts of one mp4 (no decode)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        else:
            duration = None
        rate = stream.average_rate
        return {
            "frames_header": int(stream.frames),
            "fps": float(rate) if rate else None,
            "duration_s": duration,
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "codec": str(stream.codec_context.name),
            "size_bytes": path.stat().st_size,
        }


def count_decoded_frames(path: Path) -> int:
    """Frames a full decode actually yields."""
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def safe_probe(path: Path) -> dict[str, Any]:
    """:func:`probe_video`, with a missing or unreadable file recorded instead of raised."""
    try:
        return probe_video(path)
    except (FFmpegError, OSError, IndexError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def verify_ids(records: list[NexarRecord], k: int) -> list[str]:
    """``k`` ids to decode in full: evenly spaced over each class's sorted ids, half per class."""
    if k <= 0:
        return []
    chosen: list[str] = []
    per_class = max(1, k // len(constants.NEXAR_CLASS_DIRS))
    for label in sorted(set(constants.NEXAR_CLASS_DIRS.values())):
        ids = sorted(r.video_id for r in records if r.label == label)
        if not ids:
            continue
        idx = np.unique(np.linspace(0, len(ids) - 1, min(per_class, len(ids))).round().astype(int))
        chosen.extend(ids[i] for i in idx)
    return chosen


# --------------------------------------------------------------------------- summary


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in constants.EDA_PERCENTILES}


def _histogram(values: list[float], edges: np.ndarray) -> dict[str, int]:
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=edges)
    return {
        f"[{lo:g},{hi:g})": int(c) for lo, hi, c in zip(edges[:-1], edges[1:], counts, strict=True)
    }


def abnormal_share(row: dict[str, Any], post_event_s: float) -> float | None:
    """Share of the video inside ``[t_alert, min(t_event + post_event_s, duration)]``."""
    duration = row.get("duration_s")
    alert, event = row.get("time_of_alert"), row.get("time_of_event")
    if not duration or alert is None or event is None or alert > event:
        return None
    end = min(event + post_event_s, duration)
    return float(max(0.0, end - max(alert, 0.0)) / duration)


def build_rows(
    records: list[NexarRecord], probes: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """One census row per video: annotation + probe + issues."""
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        probe = probes.get(record.video_id, {"error": "not probed"})
        issues = annotation_issues(record)
        if "error" in probe:
            issues.append(ISSUE_PROBE_FAILED)
        duration = probe.get("duration_s")
        if record.time_of_event is not None and duration and record.time_of_event > duration:
            issues.append(ISSUE_EVENT_AFTER_END)
        rows[record.video_id] = {
            "label": record.label,
            "path": record.relative_path,
            "time_of_event": record.time_of_event,
            "time_of_alert": record.time_of_alert,
            **{f"meta_{k}": v for k, v in sorted(record.extras.items())},
            **probe,
            "issues": issues,
        }
    return rows


def _by_label(rows: dict[str, dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = {}
    for row in rows.values():
        value = row.get(key)
        if value is None:
            continue
        table.setdefault(str(value), Counter())[str(row["label"])] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(table.items())}


def summarize(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The census summary (§ docstring); pure function of the rows."""
    labels = {vid: int(r["label"]) for vid, r in rows.items()}
    probed = {vid: r for vid, r in rows.items() if r.get("duration_s")}
    summary: dict[str, Any] = {
        "videos": len(rows),
        "per_label": dict(sorted(Counter(str(v) for v in labels.values()).items())),
        "probed": len(probed),
        "issues": dict(sorted(Counter(i for r in rows.values() for i in r["issues"]).items())),
    }

    durations = {
        lab: [r["duration_s"] for r in probed.values() if r["label"] == lab] for lab in (0, 1)
    }
    summary["duration_s"] = {str(lab): _percentiles(v) for lab, v in durations.items()}
    all_durations = durations[0] + durations[1]
    if all_durations:
        top = float(np.ceil(max(all_durations) / constants.NEXAR_DURATION_BIN_S))
        edges = np.arange(
            0.0, (top + 1) * constants.NEXAR_DURATION_BIN_S, constants.NEXAR_DURATION_BIN_S
        )
        summary["duration_hist"] = {str(lab): _histogram(v, edges) for lab, v in durations.items()}
    if durations[0] and durations[1]:
        y = np.asarray([r["label"] for r in probed.values()])
        x = np.asarray([r["duration_s"] for r in probed.values()])
        auc = frame_auc(x, y)
        summary["length_ruler_auc"] = {
            "auc": round(auc, 4),
            "separability": round(max(auc, 1 - auc), 4),
        }

    fps_rows = {
        vid: {**r, "fps_rounded": round(r["fps"], constants.NEXAR_FPS_ROUND)}
        for vid, r in probed.items()
        if r.get("fps")
    }
    summary["fps"] = _by_label(fps_rows, "fps_rounded")
    res_rows = {
        vid: {**r, "resolution": f"{r['width']}x{r['height']}"} for vid, r in probed.items()
    }
    summary["resolution"] = _by_label(res_rows, "resolution")
    summary["codec"] = _by_label(probed, "codec")
    mismatched = [
        vid
        for vid, r in probed.items()
        if r.get("fps")
        and r["frames_header"] > 0
        and abs(r["frames_header"] - r["duration_s"] * r["fps"]) > HEADER_FRAME_TOLERANCE
    ]
    summary["header_frames_vs_duration_fps"] = {
        "mismatched": len(mismatched),
        "ids": mismatched[:20],
    }
    summary["frames_header_total"] = int(sum(r["frames_header"] for r in probed.values()))

    meta_keys = sorted({k for r in rows.values() for k in r if k.startswith("meta_")})
    summary["categories"] = {
        key: _by_label(rows, key)
        for key in meta_keys
        if len({str(r.get(key)) for r in rows.values()}) <= MAX_CATEGORY_LEVELS
    }

    positives = [r for r in probed.values() if r["label"] == 1 and not r["issues"]]
    rel = [r["time_of_event"] / r["duration_s"] for r in positives]
    summary["positives_clean"] = len(positives)
    summary["event_relative_position"] = _percentiles(rel)
    if rel:
        summary["event_relative_hist"] = _histogram(
            rel, np.linspace(0.0, 1.0, constants.NEXAR_HIST_BINS + 1)
        )
    summary["time_of_event_s"] = _percentiles([r["time_of_event"] for r in positives])
    summary["time_of_alert_s"] = _percentiles([r["time_of_alert"] for r in positives])
    summary["alert_to_event_s"] = _percentiles(
        [r["time_of_event"] - r["time_of_alert"] for r in positives]
    )
    total_s = sum(all_durations)
    share: dict[str, Any] = {}
    for post in constants.NEXAR_POST_EVENT_PROBES_S:
        pairs = [(r, s) for r in positives if (s := abnormal_share(r, post)) is not None]
        per_video = [s for _, s in pairs]
        abnormal_s = sum(s * r["duration_s"] for r, s in pairs)
        share[f"{post:g}"] = {
            "per_positive_video": _percentiles(per_video),
            "corpus_frame_share": round(abnormal_s / total_s, 4) if total_s else None,
        }
    summary["abnormal_share_by_post_event_s"] = share
    return summary


# --------------------------------------------------------------------------- report


def _table(title: str, table: dict[str, dict[str, int]]) -> list[str]:
    lines = [f"**{title}**", "", "| value | label 0 | label 1 |", "|---|---:|---:|"]
    lines += [f"| {k} | {v.get('0', 0)} | {v.get('1', 0)} |" for k, v in table.items()]
    return [*lines, ""]


def render_markdown(census: dict[str, Any]) -> str:
    """Human read-out of :func:`summarize` (the JSON is the record)."""
    s = census["summary"]
    lines = [
        "# Nexar N0 census",
        "",
        f"- videos {s['videos']} (per label {s['per_label']}), probed {s['probed']}, "
        f"clean positives {s['positives_clean']}",
        f"- metadata columns: {census['columns']}",
        f"- issues: {s['issues'] or 'none'}",
        f"- header frames total {s['frames_header_total']}; header vs duration x fps mismatched "
        f"{s['header_frames_vs_duration_fps']['mismatched']}",
        f"- decode check: {census['decode_check']}",
        f"- length ruler (duration -> label): {s.get('length_ruler_auc')}",
        "",
        "| quantity | label | " + " | ".join(f"p{p}" for p in constants.EDA_PERCENTILES) + " |",
        "|---|---|" + "---:|" * len(constants.EDA_PERCENTILES),
    ]

    def row(name: str, label: str, pct: dict[str, float] | None) -> str:
        cells = [f"{pct[f'p{p}']:.3f}" if pct else "-" for p in constants.EDA_PERCENTILES]
        return f"| {name} | {label} | " + " | ".join(cells) + " |"

    for lab in ("0", "1"):
        lines.append(row("duration_s", lab, s["duration_s"].get(lab)))
    lines += [
        row("t_event / duration", "1", s["event_relative_position"]),
        row("time_of_event_s", "1", s["time_of_event_s"]),
        row("time_of_alert_s", "1", s["time_of_alert_s"]),
        row("alert -> event (s)", "1", s["alert_to_event_s"]),
    ]
    for post, block in s["abnormal_share_by_post_event_s"].items():
        lines.append(row(f"share [alert, event+{post}s]", "1", block["per_positive_video"]))
    lines.append("")
    lines.append(
        "Corpus abnormal frame share by post-event extension: "
        + ", ".join(
            f"+{k}s {v['corpus_frame_share']}"
            for k, v in s["abnormal_share_by_post_event_s"].items()
        )
    )
    lines.append("")
    if "event_relative_hist" in s:
        lines += ["**t_event / duration histogram (positives)**", "", "| bin | n |", "|---|---:|"]
        lines += [f"| {k} | {v} |" for k, v in s["event_relative_hist"].items()]
        lines.append("")
    lines += (
        _table("fps", s["fps"])
        + _table("resolution", s["resolution"])
        + _table("codec", s["codec"])
    )
    for key, table in s["categories"].items():
        lines += _table(key, table)
    return "\n".join(lines) + "\n"


def _write_atomic(target: Path, text: str) -> None:
    staged = part_path(target)
    staged.write_text(text, encoding="utf-8")
    staged.replace(target)


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes(), usedforsecurity=False).hexdigest()


def run_census(root: Path, workers: int, verify_decode: int) -> dict[str, Any]:
    """Probe every training video under ``root`` and build the census record."""
    records = load_train_records(root)
    if len(records) != constants.NEXAR_VIDEOS:
        LOGGER.warning(
            "metadata lists %d videos, README says %d", len(records), constants.NEXAR_VIDEOS
        )
    paths = {r.video_id: root / r.relative_path for r in records}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        probes = dict(zip(paths, pool.map(safe_probe, paths.values()), strict=True))
    LOGGER.info(
        "probed %d videos (%d failed)", len(probes), sum("error" in p for p in probes.values())
    )

    rows = build_rows(records, probes)
    decode: dict[str, Any] = {}
    for vid in verify_ids(records, verify_decode):
        if "error" in probes[vid]:
            continue
        decoded = count_decoded_frames(paths[vid])
        decode[vid] = {"header": probes[vid]["frames_header"], "decoded": decoded}
        rows[vid]["frames_decoded"] = decoded
    decode_check = {
        "checked": len(decode),
        "max_abs_diff": max(
            (abs(d["header"] - d["decoded"]) for d in decode.values()), default=None
        ),
    }

    metadata_files = {
        class_dir: root / constants.NEXAR_TRAIN_DIR / class_dir / constants.NEXAR_METADATA_FILENAME
        for class_dir in sorted(constants.NEXAR_CLASS_DIRS)
    }
    return {
        "plan": ".project/plans/katvad-v2-nexar-feasibility.md N0",
        "hf_repo": constants.NEXAR_HF_REPO,
        "inputs": {
            k: {"name": f"{k}/{p.name}", "sha1": _sha1(p)} for k, p in metadata_files.items()
        },
        "columns": sorted(
            {k for r in records for k in r.extras} | {"file_name", "time_of_event", "time_of_alert"}
        ),
        "decode_check": decode_check,
        "summary": summarize(rows),
        "videos": rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="dataset root holding train/{positive,negative}/ (HF layout)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="parallel header probes (I/O bound on Drive)",
    )
    parser.add_argument(
        "--verify-decode",
        type=int,
        default=DEFAULT_VERIFY_DECODE,
        help="videos decoded in full to check the header frame count",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    census = run_census(args.root, args.workers, args.verify_decode)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(args.out_dir / CENSUS_JSON, json.dumps(census, indent=1, sort_keys=True) + "\n")
    _write_atomic(args.out_dir / CENSUS_MD, render_markdown(census))
    LOGGER.info("wrote %s", args.out_dir)


if __name__ == "__main__":
    main()
