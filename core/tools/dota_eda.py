"""EDA of the cut DoTA train corpus, ``DoTA_train_window`` (``KIM_RUN.md`` §57).

The corpus is rebuilt **in memory** from DoTA's annotation zip and the frozen train map with the
same planner that wrote it (:func:`core.tools.dota_train_build.plan`), so the report needs no
feature cache. With ``--data-dir`` the rebuilt windows are **gated** against the corpus on disk
(ids, labels, start / end, positive rows); a mismatch raises.

Every number is a dataset property, never a model result, and nothing here gates an arm. Read:

* **clips** (the 3,020 train clips): length, span length, where the span sits (start / end /
  centre as a share of the clip), how many clips' span reaches the clip end or starts at row 0,
  ego / night / accident class;
* **time profile**: the share of abnormal rows per relative-position bin and per row counted back
  from the clip end, plus the no-pixel position rulers (late = ``t / N``, hump);
* **windows**: counts, abnormal rows per positive window, where they sit, the gap from each normal
  window to its span, the window-centre rulers, the windows whose V2-S input is padded (the first
  rows of a clip) and those touching the clip end, positive rate by ego / night / class, MIL k.

    python -m core.tools.dota_eda --annotations DoTA_annotations.zip --output-dir <dir> \\
        [--data-dir <data/DoTA_train_window>] --plots
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from core import constants
from core.data.dada import sampled_frame_labels
from core.eda.corpus import describe
from core.metrics import frame_auc
from core.tools.dota_train_build import make_record, plan
from core.tools.dota_train_extract import train_clips
from core.tools.dota_train_match import DotaClip, read_dota
from core.tools.kill_switch_probe import write_json_atomic
from core.tools.nexar_eda import (
    _auc,
    _desc,
    _fmt,
    _new,
    _plot_center,
    _plot_normal_gap,
    _save,
    _table,
    _within_source_auc,
    covariate_table,
    normal_geometry,
    row_profile,
)

LOGGER = logging.getLogger(__name__)


def youtube_id(dota_id: str) -> str:
    """DoTA ids are ``<youtube id>_<start frame>``; YouTube ids may hold ``_`` themselves."""
    return dota_id.rsplit("_", 1)[0]


def first_full_row() -> int:
    """First stride-3 row whose causal V2-S clip needs no clamp to frame 0."""
    return math.ceil(constants.DOTA_VIDEOMAE_FRAME_STEP * (constants.VIDEOMAE_CLIP_FRAMES - 1)
                     / constants.DOTA_TRAIN_STRIDE)


def read_extras(annotations: Path) -> dict[str, dict[str, Any]]:
    """``ego_involve`` and ``accident_name`` per clip (the fields :func:`read_dota` drops)."""
    out: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(annotations) as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                d = json.loads(zf.read(name))
                out[d["video_name"]] = {"ego": bool(d["ego_involve"]),
                                        "accident": str(d["accident_name"])}
    return out


def clip_row(clip: DotaClip, extra: dict[str, Any]) -> dict[str, Any]:
    labels = np.asarray(sampled_frame_labels(make_record(clip), constants.DOTA_TRAIN_STRIDE))
    n, span = int(labels.size), np.flatnonzero(labels)
    start = int(span[0]) if span.size else None
    end = int(span[-1]) + 1 if span.size else None
    return {
        "id": clip.dota_id, "youtube": youtube_id(clip.dota_id), "rows": n,
        "seconds": clip.frames / constants.DOTA_FPS, "span_rows": int(span.size),
        "span_start": start, "span_end": end,
        "start_rel": None if start is None else start / n,
        "end_rel": None if end is None else end / n,
        "center_rel": None if start is None or end is None else (start + end) / 2 / n,
        "pre_rows": start, "post_rows": None if end is None else n - end,
        "night": "night" if clip.night else "day", "ego": "ego" if extra["ego"] else "other",
        "accident": extra["accident"], "labels": labels,
    }


def _shares(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    counts = Counter(r[key] for r in rows)
    return {k: v / len(rows) for k, v in counts.most_common()}


def clip_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spanned = [r for r in rows if r["span_rows"]]
    tol = constants.DOTA_EDA_END_TOLERANCE_ROWS
    return {
        "clips": len(rows), "clips_without_abnormal_rows": len(rows) - len(spanned),
        "rows": describe([r["rows"] for r in rows], "rows / clip (s3)"),
        "seconds": describe([r["seconds"] for r in rows], "seconds / clip"),
        "span_rows": describe([r["span_rows"] for r in spanned], "abnormal rows / clip"),
        "span_share": describe([r["span_rows"] / r["rows"] for r in spanned],
                               "abnormal share of the clip"),
        "start_rel": describe([r["start_rel"] for r in spanned], "span start / clip rows"),
        "end_rel": describe([r["end_rel"] for r in spanned], "span end / clip rows"),
        "center_rel": describe([r["center_rel"] for r in spanned], "span centre / clip rows"),
        "pre_rows": describe([r["pre_rows"] for r in spanned], "normal rows before the span"),
        "post_rows": describe([r["post_rows"] for r in spanned], "normal rows after the span"),
        "share_span_reaches_end": float(np.mean([r["post_rows"] <= tol for r in spanned])),
        "share_span_starts_at_0": float(np.mean([r["pre_rows"] == 0 for r in spanned])),
        "share_pre_below_window": float(np.mean(
            [r["pre_rows"] < constants.DOTA_TRAIN_WINDOW_LENGTH for r in spanned])),
        "ego": _shares(rows, "ego"), "night": _shares(rows, "night"),
        "accident": _shares(rows, "accident"),
        "youtube_ids": len({r["youtube"] for r in rows}),
    }


def time_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Abnormal share per relative-position bin, per row back from the end and from the start."""
    bins, tail = constants.DOTA_EDA_POSITION_BINS, constants.DOTA_EDA_TAIL_ROWS
    hits, seen = np.zeros(bins), np.zeros(bins)
    end_hits, end_seen = np.zeros(tail), np.zeros(tail)
    head_hits, head_seen = np.zeros(tail), np.zeros(tail)
    for r in rows:
        y, n = r["labels"], r["rows"]
        b = np.minimum((np.arange(n) * bins) // n, bins - 1)
        np.add.at(hits, b, y)
        np.add.at(seen, b, 1)
        k = min(tail, n)
        end_hits[:k] += y[::-1][:k]
        end_seen[:k] += 1
        head_hits[:k] += y[:k]
        head_seen[:k] += 1
    return {
        "by_relative_bin": (hits / np.maximum(seen, 1)).tolist(),
        "by_rows_from_end": (end_hits / np.maximum(end_seen, 1)).tolist(),
        "by_rows_from_start": (head_hits / np.maximum(head_seen, 1)).tolist(),
        "row_share": float(hits.sum() / seen.sum()),
    }


def position_rulers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """No-pixel row-level rulers: pooled (micro) and mean within-clip (macro) AUC."""
    late: list[float] = []
    hump: list[float] = []
    labels: list[int] = []
    macro_late: list[float] = []
    macro_hump: list[float] = []
    for r in rows:
        y, n = r["labels"], r["rows"]
        t = (np.arange(n) + 0.5) / n
        h = -np.abs(t - constants.NEXAR_EDA_HUMP_CENTER)
        late += t.tolist()
        hump += h.tolist()
        labels += y.tolist()
        if 0 < y.sum() < n:
            macro_late.append(frame_auc(t, y))
            macro_hump.append(frame_auc(h, y))
    return {
        "micro_late": _auc(late, labels), "micro_hump": _auc(hump, labels),
        "macro_late": float(np.mean(macro_late)), "macro_hump": float(np.mean(macro_hump)),
        "two_class_clips": len(macro_late),
        "definition": "late = t / N; hump = -|t / N - 0.5|; macro = mean over two-class clips",
    }


def window_rows(clips: list[DotaClip], extras: dict[str, dict[str, Any]]
                ) -> list[dict[str, Any]]:
    windows, frame_labels, by_window, labels = plan(clips)
    spans = {c.dota_id: clip_row(c, extras[c.dota_id]) for c in clips}
    padded_until = first_full_row()
    out = []
    for wid, w in sorted(windows.items()):
        c = spans[w.source]
        profile = list(frame_labels[wid])
        out.append({
            "id": wid, "source": w.source, "label": int(labels[wid]), "start": w.start,
            "end": w.end, "positive": int(sum(profile)), "profile": profile,
            "source_rows": c["rows"], "center": (w.start + w.end) / 2 / c["rows"],
            "span_start": c["span_start"], "span_end": c["span_end"],
            "padded_rows": max(0, padded_until - w.start), "touches_end": w.end == c["rows"],
            "ego": c["ego"], "night": c["night"], "accident": c["accident"],
            "record_frames": by_window[wid].total_frames,
        })
    return out


def gate(rows: list[dict[str, Any]], data_dir: Path) -> dict[str, Any]:
    """The rebuilt windows must equal the corpus on disk, window by window."""
    disk = json.loads((data_dir / constants.LABELS_TRAIN_FILENAME).read_text(encoding="utf-8"))
    meta = json.loads((data_dir / constants.META_FILENAME).read_text(encoding="utf-8"))
    mine = {r["id"]: r for r in rows}
    if set(disk) != set(mine):
        raise ValueError(f"window ids differ: {len(set(disk) ^ set(mine))} not shared "
                         f"(disk {len(disk)}, rebuilt {len(mine)})")
    bad = [w for w, y in disk.items()
           if (int(y), meta[w]["start"], meta[w]["end"], meta[w]["positive_frames"])
           != (mine[w]["label"], mine[w]["start"], mine[w]["end"], mine[w]["positive"])]
    if bad:
        raise ValueError(f"{len(bad)} windows differ from {data_dir}: {bad[:3]}")
    return {"data_dir": str(data_dir), "windows_checked": len(disk), "equal": True}


def window_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [r for r in rows if r["label"] == 1]
    counts = [r["positive"] for r in positive]
    length = constants.DOTA_TRAIN_WINDOW_LENGTH
    per_source: dict[str, list[int]] = {}
    for r in rows:
        per_source.setdefault(r["source"], [0, 0])[r["label"]] += 1
    labels = [r["label"] for r in rows]
    for r in rows:
        r["hump"] = -abs(r["center"] - constants.NEXAR_EDA_HUMP_CENTER)
    by_class = {name: [r for r in rows if r["label"] == y] for y, name in ((0, "normal"),
                                                                          (1, "positive"))}
    return {
        "counts": {
            "windows": len(rows), "positive": len(positive), "normal": len(rows) - len(positive),
            "sources": len(per_source),
            "sources_without_a_normal_window": sum(1 for n, _ in per_source.values() if n == 0),
            "normal_per_source": describe([n for n, _ in per_source.values()], "normal / source"),
            "positive_per_source": describe([p for _, p in per_source.values()],
                                            "positive / source"),
        },
        "positive_windows": {
            "abnormal_rows": describe(counts, "abnormal rows / positive window"),
            "share_at_most": {str(k): float(np.mean([c <= k for c in counts]))
                              for k in constants.DOTA_EDA_ROW_THRESHOLDS},
            "share_whole_window_abnormal": float(np.mean([c == length for c in counts])),
            "row_profile": row_profile([r["profile"] for r in positive]),
        },
        "normal_windows": normal_geometry(rows),
        "rulers": {
            "center_raw": _auc([r["center"] for r in rows], labels),
            "center_hump": _auc([r["hump"] for r in rows], labels),
            "center_raw_within_source": _within_source_auc(rows, "center"),
            "center_hump_within_source": _within_source_auc(rows, "hump"),
            "padded_rows": _auc([r["padded_rows"] for r in rows], labels),
            "touches_end": _auc([float(r["touches_end"]) for r in rows], labels),
        },
        "padding": {
            "first_full_row": first_full_row(),
            **{f"share_{k}_padded": float(np.mean([r["padded_rows"] > 0 for r in v]))
               for k, v in by_class.items()},
            **{f"share_{k}_touching_end": float(np.mean([r["touches_end"] for r in v]))
               for k, v in by_class.items()},
        },
        "ego": covariate_table(rows, "ego"), "night": covariate_table(rows, "night"),
        "accident": covariate_table(rows, "accident")[: constants.DOTA_EDA_TOP_CLASSES],
        "mil_k": {"mil": max(1, length // constants.DADA_ORIGIN_MIL_TOPK_PCT),
                  "mul_mil": max(1, length // constants.BOTTOMK_MIL_TOPK_PCT),
                  "bottomk_25_arms": max(1, length // constants.KIM_BOTTOMK_TOPK_PCT)},
    }


def build_report(annotations: Path, data_dir: Path | None) -> tuple[dict[str, Any], dict]:
    """``(report, rows)``; rows = ``{"clips": clip rows, "windows": window rows}``."""
    annotated, extras = read_dota(annotations), read_extras(annotations)
    train_ids = sorted(train_clips(constants.DOTA_TRAIN_SPLITS_DIR
                                   / constants.DOTA_TRAIN_MAP_FILENAME))
    clips = [annotated[v] for v in train_ids]
    rows = {"clips": [clip_row(c, extras[c.dota_id]) for c in clips],
            "windows": window_rows(clips, extras)}
    report = {
        "annotations": str(annotations), "stride": constants.DOTA_TRAIN_STRIDE,
        "window": {"length": constants.DOTA_TRAIN_WINDOW_LENGTH,
                   "hop": constants.DOTA_TRAIN_WINDOW_HOP},
        "gate": gate(rows["windows"], data_dir) if data_dir else None,
        "clips": clip_summary(rows["clips"]),
        "time_profile": time_profile(rows["clips"]),
        "position_rulers": position_rulers(rows["clips"]),
        "windows": window_summary(rows["windows"]),
    }
    return report, rows


def _cov_rows(table: list[dict[str, Any]], key: str) -> list[list[Any]]:
    return [[t[key], t["normal"], t["positive"], t["sources"], t["positive_rate"]]
            for t in table]


def render_markdown(report: dict[str, Any]) -> str:
    c, tp, w = report["clips"], report["time_profile"], report["windows"]
    pr, ru, pad = report["position_rulers"], w["rulers"], w["padding"]
    g = report["gate"]
    bins = constants.DOTA_EDA_POSITION_BINS
    out = [
        "# DoTA train corpus EDA (`DoTA_train_window`, KIM_RUN.md §57)", "",
        f"Stride {report['stride']} (0.3 s / row) · windows {report['window']['length']} rows, "
        f"hop {report['window']['hop']} · dataset properties only, no model, nothing gates.", "",
        ("Gate: rebuilt windows == corpus on disk, "
         f"{g['windows_checked']} windows checked." if g else
         "Gate: not run (no `--data-dir`); windows rebuilt from the annotations only."), "",
        "## 1. Clips", "",
        *_table(["", "value"], [
            ["clips", c["clips"]],
            ["clips with no abnormal row", c["clips_without_abnormal_rows"]],
            ["YouTube ids", c["youtube_ids"]],
            ["span reaches the clip end (<= 1 row)", c["share_span_reaches_end"]],
            ["span starts at row 0", c["share_span_starts_at_0"]],
            ["normal rows before span < one window (12)", c["share_pre_below_window"]],
            ["ego-involved", c["ego"].get("ego")],
            ["night", c["night"].get("night")],
        ]),
        *[f"- {k}: {_desc(c[k])}" for k in
          ("seconds", "rows", "span_rows", "span_share", "start_rel", "end_rel", "center_rel",
           "pre_rows", "post_rows")],
        "",
        "Accident class (share of clips):", "",
        *_table(["class", "share"],
                [list(kv) for kv in list(c["accident"].items())[: constants.DOTA_EDA_TOP_CLASSES]]),
        "## 2. Time profile of the abnormal rows", "",
        f"Row abnormal share: {_fmt(tp['row_share'])}.", "",
        *_table(["relative bin", "abnormal share"],
                [[f"{i / bins:.1f}-{(i + 1) / bins:.1f}", a]
                 for i, a in enumerate(tp["by_relative_bin"])]),
        *_table(["row (1 = last / first)", "from the clip end", "from the clip start"],
                [[i + 1, a, b] for i, (a, b) in enumerate(zip(
                    tp["by_rows_from_end"], tp["by_rows_from_start"], strict=True))]),
        "No-pixel row rulers (AUC; 0.5 = nothing):", "",
        *_table(["micro late", "micro hump", "macro late", "macro hump", "two-class clips"],
                [[pr["micro_late"], pr["micro_hump"], pr["macro_late"], pr["macro_hump"],
                  pr["two_class_clips"]]]),
        "## 3. Windows", "",
        f"{w['counts']['windows']} windows = {w['counts']['positive']} positive + "
        f"{w['counts']['normal']} normal from {w['counts']['sources']} clips; clips without a "
        f"normal window {w['counts']['sources_without_a_normal_window']}.", "",
        f"- normal / clip: {_desc(w['counts']['normal_per_source'])}",
        f"- positive / clip: {_desc(w['counts']['positive_per_source'])}",
        f"- abnormal rows / positive window: {_desc(w['positive_windows']['abnormal_rows'])}; "
        + ", ".join(f"<= {k}: {_fmt(v)}" for k, v in
                    w["positive_windows"]["share_at_most"].items())
        + f"; whole window: {_fmt(w['positive_windows']['share_whole_window_abnormal'])}",
        "- row profile inside a positive window: "
        + " ".join(_fmt(v) for v in w["positive_windows"]["row_profile"]),
        f"- normal windows: {w['normal_windows']['before_span']} before the span, "
        f"{w['normal_windows']['after_span']} after; gap rows "
        f"{_desc(w['normal_windows']['gap_rows'])}; within {constants.NEXAR_EDA_NEAR_ROWS} rows "
        f"{_fmt(w['normal_windows']['share_within_near_rows'])}", "",
        f"V2-S padding: rows < {pad['first_full_row']} of a clip see a clamped (repeated) first "
        "frame.", "",
        *_table(["", "normal windows", "positive windows"], [
            ["share holding padded rows", pad["share_normal_padded"],
             pad["share_positive_padded"]],
            ["share touching the clip end", pad["share_normal_touching_end"],
             pad["share_positive_touching_end"]],
        ]),
        "No-pixel window rulers (AUC of the window label):", "",
        *_table(["centre raw", "centre hump", "raw within clip", "hump within clip",
                 "padded rows", "touches end"],
                [[ru["center_raw"], ru["center_hump"],
                  ru["center_raw_within_source"]["mean_auc"],
                  ru["center_hump_within_source"]["mean_auc"], ru["padded_rows"],
                  ru["touches_end"]]]),
        "Positive rate by stratum (windows):", "",
        *_table(["ego", "normal", "positive", "clips", "positive rate"],
                _cov_rows(w["ego"], "ego")),
        *_table(["night", "normal", "positive", "clips", "positive rate"],
                _cov_rows(w["night"], "night")),
        *_table(["accident", "normal", "positive", "clips", "positive rate"],
                _cov_rows(w["accident"], "accident")),
        f"MIL k on a 12-row window: `L_MIL` k = {w['mil_k']['mil']} (pct 5), multi-class k = "
        f"{w['mil_k']['mul_mil']} (pct 16), bottom-k arms (§56) k = "
        f"{w['mil_k']['bottomk_25_arms']}.", "",
    ]
    return "\n".join(out) + "\n"


def _wide() -> Figure:
    """An empty two-panel-wide figure (the caller adds its own axes)."""
    return Figure(figsize=(2 * constants.NEXAR_EDA_FIGURE_WIDTH, constants.NEXAR_EDA_FIGURE_HEIGHT))


def _plot_span_position(clips: list[dict[str, Any]], out_dir: Path) -> Path:
    edges = np.linspace(0.0, 1.0, constants.NEXAR_EDA_CENTER_BINS + 1)
    figure, axis = _new()
    for key, name, colour in (("start_rel", "span start", constants.NEXAR_EDA_COLOUR_TRAIN),
                              ("end_rel", "span end", constants.NEXAR_EDA_COLOUR_MARK)):
        values = [r[key] for r in clips if r[key] is not None]
        axis.hist(values, bins=edges, density=True, histtype="step", linewidth=2,
                  color=colour, label=f"{name} ({len(values)} clips)")
    axis.set_xlabel("position / clip rows")
    axis.set_ylabel("density")
    axis.set_title("Where the annotated span starts and ends in its clip")
    axis.legend()
    return _save(figure, out_dir / "span_position.png")


def _plot_time_profile(report: dict[str, Any], out_dir: Path) -> Path:
    figure = _wide()
    left, right = figure.subplots(1, 2)
    bins = constants.DOTA_EDA_POSITION_BINS
    tp = report["time_profile"]
    left.plot((np.arange(bins) + 0.5) / bins, tp["by_relative_bin"], marker="o",
              color=constants.NEXAR_EDA_COLOUR_TRAIN)
    right.plot(range(1, len(tp["by_rows_from_end"]) + 1), tp["by_rows_from_end"], marker="o",
               color=constants.NEXAR_EDA_COLOUR_MARK, label="from the clip end")
    right.plot(range(1, len(tp["by_rows_from_start"]) + 1), tp["by_rows_from_start"],
               marker="x", linestyle="--", color=constants.NEXAR_EDA_COLOUR_TRAIN,
               label="from the clip start")
    left.set_xlabel("relative position in the clip")
    right.set_xlabel("rows from the clip end / start (0.3 s each)")
    right.legend()
    for axis in (left, right):
        axis.set_ylabel("share of rows abnormal")
        axis.set_ylim(0, 1)
        axis.grid(alpha=constants.NEXAR_EDA_GRID_ALPHA)
    figure.suptitle("Abnormal share by position (a corpus prior, not a model result)")
    return _save(figure, out_dir / "time_profile.png")


def _plot_abnormal_rows(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    length = constants.DOTA_TRAIN_WINDOW_LENGTH
    values = [r["positive"] for r in rows if r["label"] == 1]
    figure, axis = _new()
    axis.hist(values, bins=np.arange(0.5, length + 1.5),
              weights=np.full(len(values), 1.0 / len(values)),
              color=constants.NEXAR_EDA_COLOUR_TRAIN)
    axis.set_xlabel(f"abnormal rows in a positive window (of {length})")
    axis.set_ylabel("share of positive windows")
    axis.set_title(f"How much of a positive bag is abnormal (n = {len(values)})")
    return _save(figure, out_dir / "abnormal_rows_per_positive_window.png")


def _plot_strata(report: dict[str, Any], out_dir: Path) -> Path:
    figure = _wide()
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})
    for axis, key in ((left, "ego"), (right, "accident")):
        table = report["windows"][key]
        names = [t[key] for t in table]
        x = np.arange(len(names))
        axis.bar(x - 0.2, [t["normal"] for t in table], width=0.4,
                 color=constants.NEXAR_EDA_COLOUR_NORMAL, label="normal windows")
        axis.bar(x + 0.2, [t["positive"] for t in table], width=0.4,
                 color=constants.NEXAR_EDA_COLOUR_MARK, label="positive windows")
        axis.set_xticks(x, names, rotation=45, ha="right")
        axis.set_ylabel("windows")
        axis.legend()
    figure.suptitle("Windows by ego involvement and accident class")
    return _save(figure, out_dir / "strata.png")


def write_plots(report: dict[str, Any], rows: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [_plot_span_position(rows["clips"], out_dir), _plot_time_profile(report, out_dir),
            _plot_abnormal_rows(rows["windows"], out_dir),
            _plot_normal_gap(rows["windows"], report["windows"], out_dir),
            _plot_center(rows["windows"], report["windows"], out_dir),
            _plot_strata(report, out_dir)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--annotations", type=Path, required=True, help="DoTA_annotations.zip")
    parser.add_argument("--data-dir", type=Path, help="DoTA_train_window on disk: gate against it")
    parser.add_argument("--output-dir", type=Path, required=True)
    figures = constants.NEXAR_EDA_FIGURES_DIRNAME
    parser.add_argument("--plots", action="store_true",
                        help=f"also write PNGs to <output-dir>/{figures}")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    report, rows = build_report(args.annotations, args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plots:
        figures = write_plots(report, rows, args.output_dir / constants.NEXAR_EDA_FIGURES_DIRNAME)
        report["figures"] = [p.name for p in figures]
    stem = constants.DOTA_EDA_REPORT_STEM
    write_json_atomic(args.output_dir / f"{stem}.json", report)
    (args.output_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    LOGGER.info("DoTA EDA -> %s", args.output_dir)


if __name__ == "__main__":
    main()
