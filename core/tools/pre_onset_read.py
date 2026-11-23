"""Pre-onset rise on ``nexar_val``: does a row's score climb before the alert (``KIM_RUN.md`` §43)?

Pure aggregation over saved ``nexar_eval --save-scores`` reads (whole videos, native frames); no
model is loaded and nothing gates. Onset = the first ``gt = 1`` frame (= ``time_of_alert``);
time = frames / ``NEXAR_NOMINAL_FPS``. Per row:

* **event-locked curve**: mean raw score per ``PRE_ONSET_BIN_S`` bin relative to onset, with a
  per-bin interval over videos, beside the row's negative-video mean; and the same curve with
  each video rescaled to ``(s - far) / (span - far)`` (0 = its far level, 1 = its span level);
* **zone AUC**: frames of a pre-onset zone (and of ``far`` and the span) against every frame of
  the negative videos, on the row's own ranking (0.5 = the zone scores like no accident);
* **rise** per positive video: ``(mean[-PRE_ONSET_RISE_S, 0) - far) / (span - far)``, far = the
  mean before ``-PRE_ONSET_FAR_S``; and paired per video, ``rise(row) - rise(control)``.

CLI::

    python -m core.tools.pre_onset_read --control K5f \\
        --row K5f=outputs/v_final/kim_fixed/K5f/eval/nexar/K5f \\
        --row K5df=outputs/v_final/kim_lneg_cap_fixed/K5df/eval/nexar/K5df \\
        --out-dir outputs/v_final/kim_fixed/REPORTS/pre_onset --plots
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from core import constants
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools import tadz_readout
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

Curves = dict[str, np.ndarray]
FAR, SPAN = "far", "span"
FIGURE_NAME = "pre_onset_curves.png"


def parse_row(text: str) -> tuple[str, Path]:
    """``NAME=DIR`` (DIR holds ``scores/*.npz``)."""
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--row wants NAME=DIR, got {text!r}")
    return name, Path(path)


def load_rows(rows: list[tuple[str, Path]]) -> tuple[dict[str, Curves], Curves]:
    """Every row's scores and the labels they share (same videos, same ``gt``, same lengths)."""
    names = [n for n, _ in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"row names repeat: {names}")
    runs = {name: tadz_readout.load_run(path) for name, path in rows}
    labels = tadz_readout.check_labels(runs, expect=None)
    for name, (scores, _) in runs.items():
        bad = [v for v in labels if len(scores[v]) != len(labels[v])]
        if bad:
            raise ValueError(f"{name}: score length != label length on {bad[:5]}")
    if not any(g.any() for g in labels.values()) or all(g.any() for g in labels.values()):
        raise ValueError("pre-onset read needs positive and negative videos")
    return {name: run[0] for name, run in runs.items()}, labels


def onset_seconds(gt: np.ndarray, fps: float) -> np.ndarray:
    """Seconds of every frame relative to the first positive frame."""
    onset = int(np.argmax(gt > 0))
    return (np.arange(len(gt)) - onset) / fps


def _ci(values: dict[str, float]) -> dict[str, float] | None:
    return cluster_bootstrap_ci(values, {v: v for v in values}, constants.PRE_ONSET_BOOTSTRAP,
                                constants.PRE_ONSET_BOOTSTRAP_SEED, constants.PRE_ONSET_LEVEL)


def _mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def video_reads(score: np.ndarray, gt: np.ndarray, fps: float) -> dict[str, Any]:
    """One positive video: far / pre / span means and its rise (``None`` where undefined)."""
    t = onset_seconds(gt, fps)
    far = _mean(score[t < -constants.PRE_ONSET_FAR_S])
    pre = _mean(score[(t >= -constants.PRE_ONSET_RISE_S) & (t < 0)])
    span = float(score[gt > 0].mean())
    rise = None
    if far is not None and pre is not None and span - far > constants.PRE_ONSET_MIN_CONTRAST:
        rise = (pre - far) / (span - far)
    return {"far": far, "pre": pre, "span": span, "rise": rise}


def bin_edges() -> np.ndarray:
    low, high = constants.PRE_ONSET_WINDOW_S
    step = constants.PRE_ONSET_BIN_S
    return np.arange(low, high + step / 2, step)


def event_locked(scores: Curves, labels: Curves, fps: float,
                 reads: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Mean score per bin over positive videos; ``reads`` given = each video rescaled to
    ``(s - far) / (span - far)`` (videos without a rise are left out)."""
    edges = bin_edges()
    per_bin: list[dict[str, float]] = [{} for _ in range(len(edges) - 1)]
    for v, gt in labels.items():
        if not gt.any():
            continue
        score = scores[v]
        if reads is not None:
            r = reads[v]
            if r["rise"] is None:
                continue
            score = (score - r["far"]) / (r["span"] - r["far"])
        t = onset_seconds(gt, fps)
        which = np.digitize(t, edges) - 1
        for b in range(len(per_bin)):
            hit = score[which == b]
            if hit.size:
                per_bin[b][v] = float(hit.mean())
    return [{"from_s": float(edges[b]), "to_s": float(edges[b + 1]), "ci": _ci(values)}
            for b, values in enumerate(per_bin)]


def zone_aucs(scores: Curves, labels: Curves, fps: float) -> dict[str, float | None]:
    """Frames of each zone vs every negative-video frame, on the row's own ranking."""
    negatives = np.concatenate([scores[v] for v, g in labels.items() if not g.any()])
    zones: dict[str, list[np.ndarray]] = {FAR: [], SPAN: []}
    for low, high in constants.PRE_ONSET_ZONES_S:
        zones[_zone_key(low, high)] = []
    for v, gt in labels.items():
        if not gt.any():
            continue
        t, score = onset_seconds(gt, fps), scores[v]
        zones[FAR].append(score[t < -constants.PRE_ONSET_FAR_S])
        zones[SPAN].append(score[gt > 0])
        for low, high in constants.PRE_ONSET_ZONES_S:
            zones[_zone_key(low, high)].append(score[(t >= low) & (t < high) & (gt == 0)])
    out: dict[str, float | None] = {}
    for key, parts in zones.items():
        zone = np.concatenate(parts) if parts else np.empty(0)
        if not zone.size:
            out[key] = None
            continue
        values = np.concatenate([zone, negatives])
        target = np.concatenate([np.ones(zone.size), np.zeros(negatives.size)])
        out[key] = frame_auc(values, target)
    return out


def _zone_key(low: float, high: float) -> str:
    return f"[{low:g}, {high:g})"


def row_read(scores: Curves, labels: Curves, fps: float) -> dict[str, Any]:
    reads = {v: video_reads(scores[v], g, fps) for v, g in labels.items() if g.any()}
    negative = {v: float(scores[v].mean()) for v, g in labels.items() if not g.any()}

    def pick(key: str) -> dict[str, float]:
        return {v: r[key] for v, r in reads.items() if r[key] is not None}

    return {
        "videos": {"positive": len(reads), "negative": len(negative),
                   "with_rise": len(pick("rise"))},
        "negative_mean": _ci(negative), "far_mean": _ci(pick("far")),
        "pre_mean": _ci(pick("pre")), "span_mean": _ci(pick("span")),
        "rise": _ci(pick("rise")), "zone_auc": zone_aucs(scores, labels, fps),
        "curve_raw": event_locked(scores, labels, fps),
        "curve_scaled": event_locked(scores, labels, fps, reads),
        "per_video": reads,
    }


def paired_rise(rows: dict[str, dict[str, Any]], control: str) -> dict[str, Any]:
    """``rise(row) - rise(control)`` per video, on the videos both define."""
    base = rows[control]["per_video"]
    out: dict[str, Any] = {}
    for name, read in rows.items():
        if name == control:
            continue
        diff = {v: r["rise"] - base[v]["rise"] for v, r in read["per_video"].items()
                if r["rise"] is not None and base[v]["rise"] is not None}
        out[name] = _ci(diff)
    return out


def build(rows: list[tuple[str, Path]], control: str, fps: float) -> dict[str, Any]:
    if control not in {n for n, _ in rows}:
        raise ValueError(f"control {control!r} is not among the rows {[n for n, _ in rows]}")
    scores, labels = load_rows(rows)
    reads = {name: row_read(scores[name], labels, fps) for name, _ in rows}
    return {
        "rule": "KIM_RUN.md §43 (printed only, decides nothing)", "control": control,
        "fps": fps, "sources": {name: str(path) for name, path in rows},
        "videos": len(labels), "rows": reads, "paired_rise": paired_rise(reads, control),
    }


def _fmt(c: dict[str, float] | float | None, digits: int = 3) -> str:
    if c is None:
        return "-"
    if isinstance(c, dict):
        return f"{c['mean']:.{digits}f} [{c['low']:.{digits}f}, {c['high']:.{digits}f}]"
    return f"{c:.{digits}f}"


def render(report: dict[str, Any]) -> str:
    control, rows = report["control"], report["rows"]
    names = list(rows)
    zones = list(next(iter(rows.values()))["zone_auc"])
    rise_s, far_s = constants.PRE_ONSET_RISE_S, constants.PRE_ONSET_FAR_S
    out = [
        "# Pre-onset rise on nexar_val (KIM_RUN.md §43)", "",
        f"{report['videos']} videos; onset = first `gt = 1` frame (= `time_of_alert`); "
        f"{report['fps']:g} rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** "
        f"Control {control}.", "",
        "## 1. Levels and rise (raw scores; intervals over videos)", "",
        f"rise = (mean[-{rise_s:g} s, 0) - far) / (span - far), far = mean before -{far_s:g} s: "
        "0 = still at the far level 2 s before the alert, 1 = already at the span level.", "",
        "| row | negative videos | far | last 2 s | span | rise | rise - " + control + " |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in names:
        r = rows[n]
        paired = "-" if n == control else _fmt(report["paired_rise"][n])
        out.append(f"| {n} | {_fmt(r['negative_mean'])} | {_fmt(r['far_mean'])} | "
                   f"{_fmt(r['pre_mean'])} | {_fmt(r['span_mean'])} | {_fmt(r['rise'])} | "
                   f"{paired} |")
    out += ["", "## 2. Zone AUC against every negative-video frame (0.5 = like no accident)", "",
            "| row | " + " | ".join(f"{z} s" if z.startswith("[") else z for z in zones) + " |",
            "|---|" + "---|" * len(zones)]
    for n in names:
        out.append(f"| {n} | " + " | ".join(_fmt(rows[n]["zone_auc"][z]) for z in zones) + " |")
    out += ["", "## 3. Event-locked mean raw score (bin start, s)", "",
            "| bin | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for b, cell in enumerate(rows[names[0]]["curve_raw"]):
        values = [rows[n]["curve_raw"][b]["ci"] for n in names]
        out.append(f"| {cell['from_s']:+.1f} | " + " | ".join(
            "-" if c is None else f"{c['mean']:.3f}" for c in values) + " |")
    out += ["", "## How to read this", "",
            "* A rise above 0 says the score leaves its far level before the alert; it is "
            "unlabeled positive score, not foresight (Nexar's alert is itself an anticipation "
            "mark). Never \"anticipates accidents\".",
            "* Zone AUC uses the row's own ranking, so rows with different calibration compare; "
            "the rise is scale-free per video. Raw levels in §1/§3 are not comparable across "
            "rows.",
            "* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s "
            "are not its smear (the encoder's attention can still carry them).",
            "* A row contrast changes more than one training factor for K5fno and LaGoVAD; no "
            "cause follows from n = 1.", ""]
    return "\n".join(out)


def write_plot(report: dict[str, Any], target: Path) -> Path:
    """Two panels: the raw event-locked mean (negative-video mean dashed) and the rescaled one."""
    figure = Figure(figsize=constants.PRE_ONSET_FIGURE_SIZE)
    raw_axis, scaled_axis = figure.subplots(1, 2)
    colours = constants.PRE_ONSET_ROW_COLOURS
    for i, (name, read) in enumerate(report["rows"].items()):
        colour = colours[i % len(colours)]
        for axis, key in ((raw_axis, "curve_raw"), (scaled_axis, "curve_scaled")):
            cells = [c for c in read[key] if c["ci"] is not None]
            mids = [(c["from_s"] + c["to_s"]) / 2 for c in cells]
            axis.plot(mids, [c["ci"]["mean"] for c in cells], color=colour, marker=".",
                      linewidth=2 if name == report["control"] else 1.3, label=name)
            axis.fill_between(mids, [c["ci"]["low"] for c in cells],
                              [c["ci"]["high"] for c in cells], color=colour, alpha=0.12)
        neg = read["negative_mean"]
        if neg is not None:
            raw_axis.axhline(neg["mean"], color=colour, linestyle=":", linewidth=1)
    for axis in (raw_axis, scaled_axis):
        axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1)
        axis.axvspan(-constants.PRE_ONSET_RISE_S, 0.0, color="#cccccc", alpha=0.3)
        axis.set_xlabel("seconds from the alert (first gt = 1 frame)")
        axis.grid(alpha=0.3)
    raw_axis.set_ylabel("mean raw score over positive videos")
    raw_axis.set_title("Raw score (dotted = the row's negative-video mean)")
    scaled_axis.axhline(0.0, color="#555555", linewidth=0.8)
    scaled_axis.axhline(1.0, color="#555555", linewidth=0.8)
    scaled_axis.set_ylabel("(s - far) / (span - far), per video")
    scaled_axis.set_ylim(*constants.PRE_ONSET_SCALED_YLIM)
    scaled_axis.set_title("Rescaled: 0 = far level, 1 = span level (unstable when span ~ far)")
    scaled_axis.legend(fontsize="small")
    figure.suptitle("nexar_val: score around the alert (grey = the 2 s rise zone; printed only)")
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=constants.PRE_ONSET_FIGURE_DPI)
    return target


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--row", type=parse_row, action="append", required=True,
                        help="NAME=DIR with DIR/scores/*.npz; repeat per row")
    parser.add_argument("--control", default=constants.PRE_ONSET_CONTROL)
    parser.add_argument("--fps", type=float, default=constants.NEXAR_NOMINAL_FPS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true", help=f"also write {FIGURE_NAME}")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    report = build(args.row, args.control, args.fps)
    stem = constants.PRE_ONSET_REPORT_STEM
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    if args.plots:
        write_plot(report, args.out_dir / FIGURE_NAME)
    LOGGER.info("pre-onset read written to %s", args.out_dir / f"{stem}.md")


if __name__ == "__main__":
    main()
