"""EDA of the Nexar *training* corpus K5f reads: the cut windows (``KIM_RUN.md`` §40).

``core.tools.eda`` characterizes a corpus from its test-side frame labels; a windowed corpus's
**train** windows carry one weak label each, so it cannot see the geometry the MIL loss trains
on. ``meta.json`` of a windowed corpus keeps ``source`` / ``start`` / ``end`` /
``positive_frames`` / ``normalized_span`` for every window, so the frame labels of each source
are rebuilt with :func:`core.data.dada.sampled_frame_labels` and **gated**: every window's
rebuilt positive count must equal ``meta.json``'s, or the report refuses (a position read
on labels that are not the corpus's would be fiction).

Every number is a dataset property measured from files, never a model result, and nothing here
gates. Read: counts · abnormal rows per positive window and where they sit in the window · how far
the normal windows lie from the span · windows per source · scene / weather per class · what
position and source length alone score as a window-label predictor · the same geometry on the
val windows (the side every Nexar number is read on). ``--plots`` also writes six PNG figures
(:func:`write_plots`) under ``<output-dir>/figures``.

    python -m core.tools.nexar_eda --data-dir <Thesis-V2/data/Nexar_window> --output-dir <dir> \\
        --plots
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from core import constants
from core.data.dada import DadaRecord, sampled_frame_labels
from core.data.dataset_files import num_sampled_frames
from core.eda.corpus import describe
from core.eda.protocol import clip_constant_oracle, frame_share
from core.metrics import frame_auc
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)
TRAIN_SPLIT = "train"
NO_SCENE = "unknown"


def load_corpus(
    data_dir: Path,
) -> tuple[dict[str, int], dict[str, dict[str, Any]], dict[str, list[int]]]:
    """``(labels_train, meta, frame_labels_test)`` of one built windowed corpus."""
    def read(name: str) -> Any:
        return json.loads((data_dir / name).read_text(encoding="utf-8"))

    return (read(constants.LABELS_TRAIN_FILENAME), read(constants.META_FILENAME),
            read(constants.FRAME_LABELS_TEST_FILENAME))


def source_labels(meta_row: dict[str, Any], stride: int) -> np.ndarray:
    """The source video's stride-``stride`` frame labels, rebuilt exactly as the corpus was."""
    span = meta_row["normalized_span"]
    record = DadaRecord(
        str(meta_row["source"]), str(meta_row["source"]), str(meta_row["class_name"]),
        str(meta_row["fault_label"]), int(meta_row["total_frames"]),
        None if span is None else (float(span[0]), float(span[1])), meta_row["accident_frac"],
    )
    return np.asarray(sampled_frame_labels(record, stride), dtype=np.int64)


def window_rows(
    labels: dict[str, int], meta: dict[str, dict[str, Any]], stride: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """One row per train window, plus the ids whose rebuilt positive count disagrees."""
    cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    mismatched: list[str] = []
    for wid, label in sorted(labels.items()):
        row = meta[wid]
        if row["split"] != TRAIN_SPLIT:
            raise ValueError(f"{wid} is in labels_train.json but meta.json says {row['split']!r}")
        source = str(row["source"])
        if source not in cache:
            cache[source] = source_labels(row, stride)
        full = cache[source]
        start, end = int(row["start"]), int(row["end"])
        window = full[start:end]
        if int(window.sum()) != int(row["positive_frames"]):
            mismatched.append(wid)
        hits = np.flatnonzero(window)
        span = np.flatnonzero(full)
        length = num_sampled_frames(int(row["total_frames"]), stride)
        rows.append({
            "id": wid, "source": source, "label": int(label), "start": start, "end": end,
            "positive": int(window.sum()), "source_rows": length,
            "center": (start + end) / 2 / length,
            "first": int(hits[0]) if hits.size else None,
            "last": int(hits[-1]) if hits.size else None,
            "span_start": int(span[0]) if span.size else None,
            "span_end": int(span[-1]) + 1 if span.size else None,
            "in_abnormal_source": bool(row["source_is_abnormal"]),
            "scene": str(row.get("scene") or NO_SCENE),
            "weather": str(row.get("weather") or NO_SCENE),
            "profile": window.tolist(),
        })
    return rows, mismatched


def _share(count: int, total: int) -> float | None:
    return count / total if total else None


def corpus_counts(rows: list[dict[str, Any]], test: dict[str, list[int]]) -> dict[str, Any]:
    positive = [r for r in rows if r["label"] == 1]
    per_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        per_source[r["source"]][r["label"]] += 1
    return {
        "train_windows": len(rows), "train_positive": len(positive),
        "train_normal": len(rows) - len(positive),
        "train_sources": len(per_source),
        "sources_without_a_positive_window": sum(1 for n, p in per_source.values() if p == 0),
        "sources_without_a_normal_window": sum(1 for n, p in per_source.values() if n == 0),
        "normal_per_source": describe([n for n, _ in per_source.values()],
                                      "normal windows / source"),
        "positive_per_source": describe([p for _, p in per_source.values()],
                                        "positive windows / source"),
        "normal_windows_from_abnormal_sources": sum(
            1 for r in rows if r["label"] == 0 and r["in_abnormal_source"]),
        "window_rows": describe([len(r["profile"]) for r in rows], "rows / window"),
        "val_windows": len(test),
        "val_two_class": sum(1 for v in test.values() if 0 < sum(v) < len(v)),
        "val_all_normal": sum(1 for v in test.values() if sum(v) == 0),
        "val_all_abnormal": sum(1 for v in test.values() if sum(v) == len(v)),
    }


def row_profile(profiles: list[list[int]]) -> list[float]:
    """Share of windows whose abnormal label is on at each row index (windows of one length)."""
    return np.asarray(profiles, dtype=np.float64).mean(axis=0).tolist() if profiles else []


def positive_geometry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [r for r in rows if r["label"] == 1]
    counts = [r["positive"] for r in positive]
    length = len(positive[0]["profile"]) if positive else 0
    return {
        "abnormal_rows": describe(counts, "abnormal rows / positive window"),
        "share_at_most": {str(k): _share(sum(1 for c in counts if c <= k), len(counts))
                          for k in constants.NEXAR_EDA_ROW_THRESHOLDS},
        "share_whole_window_abnormal": _share(sum(1 for c in counts if c == length), len(counts)),
        "first_abnormal_row": describe([r["first"] for r in positive], "first abnormal row"),
        "last_abnormal_row": describe([r["last"] for r in positive], "last abnormal row"),
        "row_profile": row_profile([r["profile"] for r in positive]),
    }


def normal_geometry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Distance (rows) from each normal window to its own source's span."""
    gaps: list[int] = []
    pre = post = 0
    for r in rows:
        if r["label"] == 1 or r["span_start"] is None:
            continue
        if r["end"] <= r["span_start"]:
            pre += 1
            gaps.append(r["span_start"] - r["end"])
        else:
            post += 1
            gaps.append(r["start"] - r["span_end"])
    return {
        "normal_windows_with_a_span": len(gaps), "before_span": pre, "after_span": post,
        "gap_rows": describe(gaps, "rows between a normal window and its span"),
        "share_within_near_rows": _share(
            sum(1 for g in gaps if g <= constants.NEXAR_EDA_NEAR_ROWS), len(gaps)),
    }


def _auc(scores: list[float], labels: list[int]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return frame_auc(np.asarray(scores, dtype=np.float64), np.asarray(labels))


def _within_source_auc(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Mean over sources holding both classes of the AUC of ``key`` (position inside a video)."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    values = []
    for items in by_source.values():
        auc = _auc([it[key] for it in items], [it["label"] for it in items])
        if auc is not None:
            values.append(auc)
    return {"sources": len(values), "mean_auc": float(np.mean(values)) if values else None}


def shortcut_rulers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What no-pixel window properties score as a window-label predictor (0.5 = nothing)."""
    labels = [r["label"] for r in rows]
    for r in rows:
        r["hump"] = -abs(r["center"] - constants.NEXAR_EDA_HUMP_CENTER)
    return {
        "source_rows": _auc([r["source_rows"] for r in rows], labels),
        "center_raw": _auc([r["center"] for r in rows], labels),
        "center_hump": _auc([r["hump"] for r in rows], labels),
        "center_hump_within_source": _within_source_auc(rows, "hump"),
        "definition": "AUC over train windows; hump = -|window center / source rows - 0.5|",
    }


def covariate_table(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    sources: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        counts[r[key]][r["label"]] += 1
        sources[r[key]].add(r["source"])
    total = [sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values())]
    return [{
        key: name, "normal": c[0], "positive": c[1], "sources": len(sources[name]),
        "share_of_normal": _share(c[0], total[0]), "share_of_positive": _share(c[1], total[1]),
        "positive_rate": _share(c[1], c[0] + c[1]),
    } for name, c in sorted(counts.items(), key=lambda kv: -sum(kv[1]))]


def val_geometry(test: dict[str, list[int]]) -> dict[str, Any]:
    arrays = [np.asarray(v, dtype=np.int64) for v in test.values()]
    positive = [a for a in arrays if a.sum() > 0]
    return {
        "frame_share": frame_share(arrays),
        "clip_oracle": clip_constant_oracle(arrays),
        "abnormal_rows": describe([int(a.sum()) for a in positive],
                                 "abnormal rows / positive window"),
        "row_profile": row_profile([a.tolist() for a in positive if a.size == arrays[0].size]),
    }


def gated_rows(
    data_dir: Path, stride: int
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """The train window rows and the val frame labels, after the rebuild gate."""
    labels, meta, test = load_corpus(data_dir)
    rows, mismatched = window_rows(labels, meta, stride)
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} windows' rebuilt positive rows differ from meta.json "
            f"({mismatched[:3]}): wrong --stride, or a corpus built by other code")
    minimum = constants.WINDOW_MIN_POSITIVE
    wrong_label = [r["id"] for r in rows if r["label"] != int(r["positive"] >= minimum)]
    if wrong_label:
        raise ValueError(f"{len(wrong_label)} windows' label != (positive rows >= "
                         f"{constants.WINDOW_MIN_POSITIVE}): {wrong_label[:3]}")
    return rows, test


def summarize(
    rows: list[dict[str, Any]], test: dict[str, list[int]], data_dir: Path, stride: int
) -> dict[str, Any]:
    """The report dict of gated rows (:func:`gated_rows`)."""
    return {
        "data_dir": str(data_dir), "stride": stride,
        "gate": {"windows_checked": len(rows), "positive_rows_rebuilt_equal_meta": True},
        "corpus": corpus_counts(rows, test),
        "positive_windows": positive_geometry(rows),
        "normal_windows": normal_geometry(rows),
        "rulers": shortcut_rulers(rows),
        "scene": covariate_table(rows, "scene"),
        "weather": covariate_table(rows, "weather"),
        "val": val_geometry(test),
    }


def build_report(data_dir: Path, stride: int) -> dict[str, Any]:
    return summarize(*gated_rows(data_dir, stride), data_dir, stride)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _desc(d: dict[str, Any]) -> str:
    p = d["percentiles"]
    if not p:
        return "-"
    return (f"mean {_fmt(d['mean'])}; p0 {_fmt(p['p0'])} / p25 {_fmt(p['p25'])} / "
            f"**p50 {_fmt(p['p50'])}** / p75 {_fmt(p['p75'])} / p100 {_fmt(p['p100'])}")


def _table(header: list[str], body: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in body]
    return [*lines, ""]


def render_markdown(report: dict[str, Any]) -> str:
    c, p, n = report["corpus"], report["positive_windows"], report["normal_windows"]
    ru, v = report["rulers"], report["val"]
    out = [
        "# Nexar training corpus EDA", "",
        f"`{report['data_dir']}` · stride {report['stride']} · dataset properties only, no model.",
        f"Gate: the positive rows rebuilt from `normalized_span` equal `meta.json` on all "
        f"{report['gate']['windows_checked']} train windows.", "",
        "## 1. Counts", "",
        f"- train: **{c['train_windows']}** windows = {c['train_positive']} positive + "
        f"{c['train_normal']} normal, from {c['train_sources']} sources "
        f"({c['normal_windows_from_abnormal_sources']} normal windows come from abnormal sources)",
        f"- sources without a positive window: {c['sources_without_a_positive_window']}; "
        f"without a normal window: {c['sources_without_a_normal_window']}",
        f"- rows per window: {_desc(c['window_rows'])}",
        f"- normal windows / source: {_desc(c['normal_per_source'])}",
        f"- positive windows / source: {_desc(c['positive_per_source'])}",
        f"- val: {c['val_windows']} windows = {c['val_two_class']} two-class + "
        f"{c['val_all_normal']} all-normal + {c['val_all_abnormal']} all-abnormal", "",
        "## 2. Positive windows", "",
        f"- abnormal rows / positive window: {_desc(p['abnormal_rows'])}",
        "- share with at most k abnormal rows: " + ", ".join(
            f"k={k}: {_fmt(s)}" for k, s in p["share_at_most"].items()),
        f"- whole window abnormal: {_fmt(p['share_whole_window_abnormal'])}",
        f"- first abnormal row: {_desc(p['first_abnormal_row'])}",
        f"- last abnormal row: {_desc(p['last_abnormal_row'])}", "",
        "Share of positive windows abnormal at each row index (train | val):", "",
    ]
    out += _table(["row"] + [str(i) for i in range(len(p["row_profile"]))],
                  [["train"] + [round(x, 2) for x in p["row_profile"]],
                   ["val"] + [round(x, 2) for x in v["row_profile"]]])
    out += [
        "## 3. Normal windows against their own span", "",
        f"- with a span in their source: {n['normal_windows_with_a_span']} "
        f"(before the span {n['before_span']}, after {n['after_span']})",
        f"- rows between the window and the span: {_desc(n['gap_rows'])}",
        f"- within {constants.NEXAR_EDA_NEAR_ROWS} rows of the span: "
        f"{_fmt(n['share_within_near_rows'])}", "",
        "## 4. What no-pixel properties score (AUC over train windows; 0.5 = nothing)", "",
        f"- source length (rows): {_fmt(ru['source_rows'])}",
        f"- window center, raw: {_fmt(ru['center_raw'])}; "
        f"hump `-|c - 0.5|`: {_fmt(ru['center_hump'])}",
        f"- hump inside one source (mean over {ru['center_hump_within_source']['sources']} sources "
        f"with both classes): {_fmt(ru['center_hump_within_source']['mean_auc'])}", "",
        "## 5. Scene and weather", "",
    ]
    for key in ("scene", "weather"):
        out += _table([key, "normal", "positive", "sources", "share of normal", "share of positive",
                       "positive rate"],
                      [[r[key], r["normal"], r["positive"], r["sources"], r["share_of_normal"],
                        r["share_of_positive"], r["positive_rate"]] for r in report[key]])
    fs, oracle = v["frame_share"], v["clip_oracle"]
    out += [
        "## 6. Val windows (where every Nexar number is read)", "",
        f"- frames: {fs['total_frames']}; clips by type {fs['clips']}",
        f"- constant-score-per-window oracle: micro AUC {_fmt(oracle['auc_micro'])}, "
        f"AP {_fmt(oracle['ap_micro'])} (macro 0.5 by construction)",
        f"- abnormal rows / positive window: {_desc(v['abnormal_rows'])}", "",
    ]
    return "\n".join(out)


def _save(figure: Figure, target: Path) -> Path:
    figure.tight_layout()
    figure.savefig(target, dpi=constants.NEXAR_EDA_FIGURE_DPI)
    return target


def _new(width: float = constants.NEXAR_EDA_FIGURE_WIDTH) -> tuple[Figure, Any]:
    figure = Figure(figsize=(width, constants.NEXAR_EDA_FIGURE_HEIGHT))
    return figure, figure.subplots()


def _plot_abnormal_rows(rows: list[dict[str, Any]], test: dict[str, list[int]],
                        out_dir: Path) -> Path:
    train = [r["positive"] for r in rows if r["label"] == 1]
    val = [int(sum(v)) for v in test.values() if sum(v) > 0]
    edges = np.arange(0.5, constants.NEXAR_WINDOW_LENGTH + 1.5)
    figure, axis = _new()
    for values, name, colour in ((train, "train", constants.NEXAR_EDA_COLOUR_TRAIN),
                                 (val, "val", constants.NEXAR_EDA_COLOUR_VAL)):
        if values:
            weights = np.full(len(values), 1.0 / len(values))
            axis.hist(values, bins=edges, weights=weights, histtype="step", linewidth=2,
                      color=colour, label=f"{name} (n = {len(values)}, p50 "
                                          f"{float(np.median(values)):.0f})")
    axis.set_xlabel(f"abnormal rows in a positive window (of {constants.NEXAR_WINDOW_LENGTH})")
    axis.set_ylabel("share of positive windows")
    axis.set_title("How much of a positive bag is abnormal")
    axis.legend()
    return _save(figure, out_dir / "abnormal_rows_per_positive_window.png")


def _plot_row_profile(report: dict[str, Any], out_dir: Path) -> Path:
    figure, axis = _new()
    for key, name, colour in (("positive_windows", "train", constants.NEXAR_EDA_COLOUR_TRAIN),
                              ("val", "val", constants.NEXAR_EDA_COLOUR_VAL)):
        profile = report[key]["row_profile"]
        axis.plot(range(len(profile)), profile, marker="o", color=colour, label=name)
    axis.set_ylim(0, 1)
    axis.set_xlabel("row index inside the window")
    axis.set_ylabel("share of positive windows abnormal at this row")
    axis.set_title("Where the abnormal rows sit inside a positive window (flat = no position cue)")
    axis.grid(alpha=constants.NEXAR_EDA_GRID_ALPHA)
    axis.legend()
    return _save(figure, out_dir / "row_profile.png")


def _plot_normal_gap(rows: list[dict[str, Any]], report: dict[str, Any], out_dir: Path) -> Path:
    before = [r["span_start"] - r["end"] for r in rows
              if r["label"] == 0 and r["span_start"] is not None and r["end"] <= r["span_start"]]
    after = [r["start"] - r["span_end"] for r in rows
             if r["label"] == 0 and r["span_start"] is not None and r["end"] > r["span_start"]]
    top = max([*before, *after, 1])
    width = constants.NEXAR_EDA_NEAR_ROWS + 1  # one bin = the "near the span" rows exactly
    edges = np.arange(-0.5, top + width, width)
    figure, axis = _new()
    axis.hist([before, after], bins=edges, stacked=True,
              color=[constants.NEXAR_EDA_COLOUR_TRAIN, constants.NEXAR_EDA_COLOUR_VAL],
              label=[f"before the span ({len(before)})", f"after the span ({len(after)})"])
    near = constants.NEXAR_EDA_NEAR_ROWS
    share = report["normal_windows"]["share_within_near_rows"]
    axis.axvline(near + 0.5, color=constants.NEXAR_EDA_COLOUR_MARK, linestyle="--",
                 label=f"<= {near} rows: {_fmt(share)} of normal windows")
    axis.set_xlabel("rows between a normal window and its own video's span")
    axis.set_ylabel("normal windows")
    axis.set_title("How hard the in-video negatives are (near the span = hard)")
    axis.legend()
    return _save(figure, out_dir / "normal_gap_to_span.png")


def _plot_center(rows: list[dict[str, Any]], report: dict[str, Any], out_dir: Path) -> Path:
    edges = np.linspace(0.0, 1.0, constants.NEXAR_EDA_CENTER_BINS + 1)
    figure, axis = _new()
    for label, name, colour in ((0, "normal", constants.NEXAR_EDA_COLOUR_NORMAL),
                                (1, "positive", constants.NEXAR_EDA_COLOUR_MARK)):
        values = [r["center"] for r in rows if r["label"] == label]
        if values:
            axis.hist(values, bins=edges, density=True, histtype="step", linewidth=2,
                      color=colour, label=f"{name} windows ({len(values)})")
    hump = report["rulers"]["center_hump"]
    axis.set_xlabel("window centre / source length")
    axis.set_ylabel("density")
    axis.set_title(f"Window position in its video (hump AUC {_fmt(hump)}: a corpus prior, "
                   "not a model result)")
    axis.legend()
    return _save(figure, out_dir / "window_center_by_class.png")


def _plot_covariates(report: dict[str, Any], out_dir: Path) -> Path:
    figure = Figure(figsize=(2 * constants.NEXAR_EDA_FIGURE_WIDTH,
                             constants.NEXAR_EDA_FIGURE_HEIGHT))
    axes = figure.subplots(1, 2)
    total = report["corpus"]["train_positive"] / max(report["corpus"]["train_windows"], 1)
    for axis, key in zip(axes, ("scene", "weather"), strict=True):
        table = report[key]
        names = [str(r[key]) for r in table]
        rates = [r["positive_rate"] or 0.0 for r in table]
        bars = axis.bar(names, rates, color=constants.NEXAR_EDA_COLOUR_TRAIN)
        for bar, r in zip(bars, table, strict=True):
            axis.annotate(f"n={r['normal'] + r['positive']}", (bar.get_x() + bar.get_width() / 2,
                          bar.get_height()), ha="center", va="bottom", fontsize="small")
        axis.axhline(total, color=constants.NEXAR_EDA_COLOUR_MARK, linestyle="--",
                     label=f"corpus {total:.3f}")
        axis.set_ylim(0, 1)
        axis.set_ylabel("positive rate (positive / all windows)")
        axis.set_title(f"Positive rate by {key}")
        axis.tick_params(axis="x", labelrotation=30)
        axis.legend()
    return _save(figure, out_dir / "scene_weather_positive_rate.png")


def _plot_val_share(test: dict[str, list[int]], report: dict[str, Any], out_dir: Path) -> Path:
    arrays = [np.asarray(v, dtype=np.int64) for v in test.values()]
    all_normal = sum(a.size for a in arrays if a.sum() == 0)
    mixed = [a for a in arrays if 0 < a.sum() < a.size]
    mixed_pos = int(sum(a.sum() for a in mixed))
    mixed_neg = int(sum(a.size for a in mixed)) - mixed_pos
    all_pos = sum(a.size for a in arrays if a.size and a.sum() == a.size)
    figure, axis = _new()
    names = ["all-normal windows", "mixed: normal rows", "mixed: abnormal rows",
             "all-abnormal windows"]
    bars = axis.bar(names, [all_normal, mixed_neg, mixed_pos, all_pos],
                    color=[constants.NEXAR_EDA_COLOUR_NORMAL, constants.NEXAR_EDA_COLOUR_VAL,
                           constants.NEXAR_EDA_COLOUR_MARK, constants.NEXAR_EDA_COLOUR_MARK])
    for bar in bars:
        axis.annotate(str(int(bar.get_height())), (bar.get_x() + bar.get_width() / 2,
                      bar.get_height()), ha="center", va="bottom", fontsize="small")
    oracle = report["val"]["clip_oracle"]["auc_micro"]
    axis.set_ylabel("val frames (rows)")
    axis.set_title(f"Val frames by window kind: constant-per-window oracle micro AUC "
                   f"{_fmt(oracle)}")
    axis.tick_params(axis="x", labelrotation=15)
    return _save(figure, out_dir / "val_frame_share.png")


def write_plots(rows: list[dict[str, Any]], test: dict[str, list[int]], report: dict[str, Any],
                out_dir: Path) -> list[Path]:
    """Six figures of the gated rows; every one is a dataset property, never a model result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _plot_abnormal_rows(rows, test, out_dir),
        _plot_row_profile(report, out_dir),
        _plot_normal_gap(rows, report, out_dir),
        _plot_center(rows, report, out_dir),
        _plot_covariates(report, out_dir),
        _plot_val_share(test, report, out_dir),
    ]
    LOGGER.info("Wrote %d figures to %s", len(written), out_dir)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="a built windowed Nexar corpus dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=constants.NEXAR_TRAIN_STRIDE)
    parser.add_argument("--plots", action="store_true",
                        help=f"also write PNG figures to <output-dir>/"
                             f"{constants.NEXAR_EDA_FIGURES_DIRNAME}")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    rows, test = gated_rows(args.data_dir, args.stride)
    report = summarize(rows, test, args.data_dir, args.stride)
    if args.plots:
        write_plots(rows, test, report, args.output_dir / constants.NEXAR_EDA_FIGURES_DIRNAME)
    stem = constants.NEXAR_EDA_REPORT_STEM
    write_json_atomic(args.output_dir / f"{stem}.json", report)
    (args.output_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    LOGGER.info("Nexar EDA written to %s", args.output_dir / f"{stem}.md")


if __name__ == "__main__":
    main()
