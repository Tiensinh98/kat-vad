"""Window-geometry read: the DoTA-host CLIP rows scored on 12-row windows (``KIM_RUN.md`` §67).

§64.2: L5dfiDoTA (KIM on the CLIP-only trunk, trained on 12-row DoTA windows) reads below chance on
whole clips. This tool scores DoTA-CAP-dev twice per row with the same checkpoint, clips and labels:

* **whole** -- protocol B exactly (``rate_matched_eval.score_steps``, whole clip, s1[::3]);
* **windowed** -- the same s3 rows cut into ``DOTA_TRAIN_WINDOW_LENGTH`` windows at hop
  ``DOTA_TRAIN_WINDOW_HOP`` (+ a tail window), each scored alone, per-row mean over the windows
  covering it.

Both are interpolated to native frames; macro = mean per-clip AUC over two-class clips. Gate
(WG4): every row's whole macro equals its saved protocol-B read-out. Contrasts (WG5, paired per
clip, ``dota_group`` cluster bootstrap): G = L5 windowed - L5 whole, R = L5 windowed - L0 windowed
decide (WG6); the rest is printed (WG7: window-level AUC, in-window position profile, start
spike). Inference only, n = 1, in-domain. CLI::

    python -m core.tools.window_geometry_read \\
        --row L5 L5dfiDoTA .../kim_clip_host/L5dfiDoTA --row L0 L0dfiDoTArr ... \\
        --row L5cap L5dficapDoTA ... --s1-dir cache/clip/DoTA_s1_ncc \\
        --metadata data/DoTA/metadata_val.json --split-file data/DoTA/val_split.txt \\
        --data-dir data/DoTA/labels_s8 --out-dir .../REPORTS/window_geometry
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core import constants
from core.data.definitions import dataset_abbr, item_verbalizer
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_inputs import check_input_manifest
from core.data.v2_splits import dota_group, load_split
from core.device import resolve_device
from core.inference import make_class_feats_fn, sliding_window_scores
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools import k5_dvs_ignore_report as dvs
from core.tools import kim_ablation_report as ka
from core.tools import protocol_b_eval as pb
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.rate_matched_eval import (
    Protocol,
    Run,
    load_finished_model,
    native_labels,
    overlap_average,
    score_steps,
    to_native,
    window_starts,
)
from core.train import load_class_names

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/KIM_RUN.md §67"
LABEL = "PRELIMINARY (n = 1 seed, in-domain); decides only §64.2's explanation, K5f stays"
L5, L0, L5CAP = constants.KIM_WG_ROWS
WHOLE, WINDOWED = "whole", "windowed"
GAIN, COST = constants.KIM_DVS_IGNORE_GAIN, constants.KIM_DVS_IGNORE_COST
STRIDE = constants.DOTA_TRAIN_STRIDE
WINDOW, HOP = constants.DOTA_TRAIN_WINDOW_LENGTH, constants.DOTA_TRAIN_WINDOW_HOP
DECIDING = ("G", "R")
# WG5: contrast -> ((row, read), sign) terms
TERMS: dict[str, tuple[tuple[str, str, int], ...]] = {
    "G": ((L5, WINDOWED, 1), (L5, WHOLE, -1)),
    "R": ((L5, WINDOWED, 1), (L0, WINDOWED, -1)),
    "L0 windowed - whole": ((L0, WINDOWED, 1), (L0, WHOLE, -1)),
    "L5cap windowed - whole": ((L5CAP, WINDOWED, 1), (L5CAP, WHOLE, -1)),
    "L5cap windowed - L0 windowed": ((L5CAP, WINDOWED, 1), (L0, WINDOWED, -1)),
}
TERM_TEXT = {"G": "G = L5 windowed - L5 whole", "R": "R = L5 windowed - L0 windowed"}


# --------------------------------------------------------------------------- pure pieces
def topk_mean(scores: np.ndarray, pct: int) -> float:
    """Window score as ``L_MIL`` reads it: mean of the top ``max(1, n // pct)`` steps."""
    k = max(1, len(scores) // pct)
    return float(np.sort(scores)[-k:].mean())


def window_records(parts: list[tuple[int, np.ndarray]], row_labels: np.ndarray, pct: int
                   ) -> list[tuple[float, int]]:
    """``(top-k score, label)`` per window; label = >= WINDOW_MIN_POSITIVE abnormal rows."""
    out = []
    for start, scores in parts:
        positive = int(row_labels[start : start + len(scores)].sum())
        out.append((topk_mean(scores, pct), int(positive >= constants.WINDOW_MIN_POSITIVE)))
    return out


def position_profile(parts: list[tuple[int, np.ndarray]], window: int) -> list[float]:
    """Mean raw score at each position inside a full-length window (WG7)."""
    full = np.stack([s for _, s in parts if len(s) == window]) if parts else np.empty((0, window))
    return [float(x) for x in full.mean(axis=0)] if len(full) else []


def spike_ratio(steps: np.ndarray, rows: tuple[int, ...] = constants.KIM_WG_SPIKE_ROWS) -> float:
    """Mean score at ``rows`` over the clip's median step score (WG7)."""
    picked = steps[[r for r in rows if r < len(steps)]]
    median = float(np.median(steps))
    return float(picked.mean() / median) if len(picked) and median > 0 else float("nan")


def verdict(labels: dict[str, str]) -> str:
    """WG6 from the labels of G and R."""
    if labels["G"] != GAIN:
        return constants.KIM_WG_NOT
    return constants.KIM_WG_PART if labels["R"] == COST else constants.KIM_WG_GEOMETRY


def paired(aucs: dict[tuple[str, str], dict[str, float]], terms: tuple[tuple[str, str, int], ...]
           ) -> dict[str, float]:
    """Per-clip signed sum over the clips every term scored."""
    common = sorted(set.intersection(*(set(aucs[(r, m)]) for r, m, _ in terms)))
    return {v: sum(sign * aucs[(r, m)][v] for r, m, sign in terms) for v in common}


# --------------------------------------------------------------------------- scoring
def windowed_steps(model: Any, feats: torch.Tensor, class_feats_fn: Any
                   ) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """Each window scored alone; per-row mean over the windows covering it (WG3)."""
    parts = [(start, sliding_window_scores(model, feats[start : start + WINDOW],
                                           class_feats_fn)[0].numpy())
             for start in window_starts(len(feats), WINDOW, HOP)]
    return overlap_average(parts, len(feats)), parts


def score_row(name: str, root: Path, seed: int, ids: list[str], frames: dict[str, int],
              labels: dict[str, np.ndarray], args: argparse.Namespace, device: torch.device
              ) -> dict[str, Any]:
    """Whole and windowed native scores of one row, plus its WG7 prints."""
    run_dir = root / constants.KIM_MAIN_ARM / f"s{seed}"
    model, step, text_encode_fn, cfg = load_finished_model(
        Run(name, run_dir / "checkpoint_last.pt"), device)
    check_input_manifest(args.s1_dir, cfg.v2.crn, cfg.v2.motion)  # plain rows only
    pct = int(ka._load_config(run_dir)["loss.mil_topk_pct"])
    class_names = load_class_names(args.data_dir)
    whole: dict[str, np.ndarray] = {}
    windowed: dict[str, np.ndarray] = {}
    windows: list[tuple[float, int]] = []
    parts_all: list[tuple[int, np.ndarray]] = []
    spikes: list[float] = []
    with torch.no_grad():
        for v in ids:
            rows = torch.from_numpy(pb.input_rows(args.s1_dir, v, frames[v], STRIDE, False))
            steps = score_steps(model, rows, text_encode_fn, class_names, v,
                                Protocol(STRIDE, None))
            fn = make_class_feats_fn(text_encode_fn, class_names,
                                     item_verbalizer(dataset_abbr(constants.DOTA_DATASET), v))
            wsteps, parts = windowed_steps(model, rows, fn)
            whole[v] = to_native(steps, STRIDE, frames[v])
            windowed[v] = to_native(wsteps, STRIDE, frames[v])
            windows += window_records(parts, labels[v][::STRIDE], pct)
            parts_all += parts
            spikes.append(spike_ratio(steps))
    LOGGER.info("%s (step %d): scored %d clips whole and windowed", name, step, len(ids))
    w_scores, w_labels = (np.array(x) for x in zip(*windows, strict=True))
    return {"step": step, "mil_topk_pct": pct, "scores": {WHOLE: whole, WINDOWED: windowed},
            "window_auc": frame_auc(w_scores, w_labels), "windows": len(windows),
            "positive_windows": int(w_labels.sum()),
            "position_profile": position_profile(parts_all, WINDOW),
            "start_spike_whole": float(np.nanmean(spikes))}


# --------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = {key: (name, Path(root)) for key, name, root in args.row}
    if set(rows) != set(constants.KIM_WG_ROWS):
        raise ValueError(f"need exactly the rows {constants.KIM_WG_ROWS}, got {sorted(rows)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    ids = load_split(args.split)
    records = {r.video_id: r for r in parse_metadata(args.metadata,
                                                     read_split_ids(args.split_file))}
    frames = {v: len(np.load(args.s1_dir / f"{v}.npy", mmap_mode="r")) for v in ids}
    labels = {v: native_labels(records[v], frames[v]) for v in ids}
    groups = {v: dota_group(v) for v in ids}
    per_row: dict[str, Any] = {}
    aucs: dict[tuple[str, str], dict[str, float]] = {}
    for key, (name, root) in rows.items():
        got = score_row(name, root, args.seed, ids, frames, labels, args, device)
        for read in (WHOLE, WINDOWED):
            aucs[(key, read)] = pb.clip_aucs(got["scores"][read], labels)
            pb.write_clip_scores(_npz(args.out_dir, key, read), got["scores"][read], labels)
        saved = json.loads((root / "eval" / "dota" / constants.KIM_MAIN_ARM
                            / pb.READOUT_JSON).read_text(encoding="utf-8"))["macro"]["mean"]
        whole_macro = float(np.mean(list(aucs[(key, WHOLE)].values())))
        if abs(whole_macro - saved) > constants.KIM_WG_GATE_TOL:  # WG4
            raise ValueError(f"{name}: whole macro {whole_macro:.6f} != saved read-out "
                             f"{saved:.6f}; the scorer is not protocol B")
        per_row[key] = {k: v for k, v in got.items() if k != "scores"} | {
            "name": name, "macro": {r: _ci(aucs[(key, r)], groups)
                                    for r in (WHOLE, WINDOWED)}}
    contrasts = {t: _ci(paired(aucs, terms), groups) for t, terms in TERMS.items()}
    labels_out = {t: dvs.label({"delta": c}) for t, c in contrasts.items()}
    report = {"rule": RULE, "label": LABEL, "split": args.split, "clips": len(ids),
              "two_class": len(aucs[(L5, WHOLE)]), "window": WINDOW, "hop": HOP,
              "stride": STRIDE, "rows": per_row, "contrasts": contrasts, "labels": labels_out,
              "verdict": verdict(labels_out)}
    write_json_atomic(args.out_dir / f"{constants.KIM_WG_REPORT_STEM}.json", report)
    write_text_atomic(args.out_dir / f"{constants.KIM_WG_REPORT_STEM}.md", render(report))
    return report


def _npz(out_dir: Path, key: str, read: str) -> Path:
    path = out_dir / key / f"clip_scores_{read}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ci(values: dict[str, float], groups: dict[str, str]) -> dict[str, float] | None:
    return cluster_bootstrap_ci(values, {v: groups[v] for v in values}, constants.V2_E1_BOOTSTRAP,
                                constants.V5_BOOTSTRAP_SEED, constants.V2_E1_CI)


# --------------------------------------------------------------------------- render
def _fmt(c: dict[str, float] | None, signed: bool = False) -> str:
    if c is None:
        return "—"
    f = "+.4f" if signed else ".4f"
    return f"{c['mean']:{f}} [{c['low']:{f}}, {c['high']:{f}}]"


def render(r: dict[str, Any]) -> str:
    rows = r["rows"]
    lines = [f"# §67 Window-geometry read, DoTA-CAP-dev — **{r['label']}**", "",
             f"{r['clips']} clips ({r['two_class']} two-class), s1[::{r['stride']}]; windowed = "
             f"{r['window']} rows, hop {r['hop']}, each window scored alone. Gate WG4 passed "
             "(whole = saved protocol B).", "",
             f"## Verdict (WG6): **{r['verdict']}**", "",
             "| contrast | Δ macro [95 % CI] | label |", "|---|---|---|"]
    for t, c in r["contrasts"].items():
        tag = "" if t in DECIDING else " *(printed)*"
        lines.append(f"| {TERM_TEXT.get(t, t)}{tag} | {_fmt(c, True)} | {r['labels'][t]} |")
    lines += ["", "## Rows", "",
              "| row | whole macro | windowed macro | window AUC (top-k vs any-abnormal) | "
              "start spike, whole (rows 1-2 / median) |", "|---|---|---|---|---|"]
    for key, row in rows.items():
        lines.append(f"| {key} = {row['name']} | {_fmt(row['macro'][WHOLE])} | "
                     f"{_fmt(row['macro'][WINDOWED])} | {row['window_auc']:.4f} "
                     f"({row['positive_windows']}/{row['windows']} positive) | "
                     f"{row['start_spike_whole']:.2f} |")
    lines += ["", "## Mean raw score by position inside a 12-row window (printed)", "",
              "| row | " + " | ".join(str(i) for i in range(r["window"])) + " |",
              "|---|" + "---|" * r["window"]]
    for key, row in rows.items():
        lines.append(f"| {key} | " + " | ".join(f"{x:.3f}" for x in row["position_profile"])
                     + " |")
    lines += ["", "Never: \"KIM works on DoTA\" from GEOMETRY (the deployable read is the whole "
              "clip); \"the start zeros are the cause\" (printed only); any zero-shot sentence; "
              "a seed claim."]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--row", nargs=3, action="append", required=True,
                        metavar=("KEY", "NAME", "ROOT"),
                        help=f"KEY in {constants.KIM_WG_ROWS}; ROOT holds K5/s<seed>/ and eval/")
    parser.add_argument("--s1-dir", type=Path, required=True, help="plain DoTA_s1_ncc")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="class definitions")
    parser.add_argument("--split", default=constants.V2_SPLIT_DOTA_CAP_DEV,
                        choices=[constants.V2_SPLIT_DOTA_CAP_DEV])
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(build_arg_parser().parse_args(argv))
    LOGGER.info("verdict: %s", report["verdict"])


__all__ = ["paired", "position_profile", "render", "run", "spike_ratio", "topk_mean",
           "verdict", "window_records", "windowed_steps"]


if __name__ == "__main__":
    main()
