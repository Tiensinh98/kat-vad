"""KIM form probe: does SA²'s two-sided, signed second difference read better than KIM's? (no train)

Rule ``core/docs/final/KIM_RUN.md`` §14, written before any number. Inference only on the pilot's
own host (K0 s2024), its reads and its inputs; nothing is extracted. For ``vt`` and ``clip``, per
step (``suffix`` = the row name after ``<feature>2``):

* none -- KIM's form, ``kim_pilot.change_scores`` order 2 (the reference);
* ``s`` -- signed one-sided, ``mean_c |z_t - 2 z_{t-1} + z_{t-2}|``;
* ``c`` -- signed two-sided, ``mean_c |2 z_t - z_{t-1} - z_{t+1}|``: ``s`` one step earlier;
* ``m`` -- SA²'s own ``Mo`` (``core.data.stp_patches.motion_magnitude``) on one cell.

Every read is within one video (macro AUC, ranks, peak), so only the reference carries ``rho``.
A candidate passes iff it beats the same feature's reference by ``KIM_PILOT_INPUT_MARGIN`` on both
DoTA-CAP-dev and DADA. Before anything is written, the recomputed ``vt2``, ``clip2`` and ``host``
macros must equal the recorded pilot's (``--baseline``), or the run raises.

CLI::

    python -m core.tools.kim_form_probe --name K0 --ckpt outputs/kim/K0/s2024/checkpoint_last.pt \\
        --baseline outputs/kim/REPORTS/pilot_edges/kim_pilot.json \\
        --set dota_cap_dev native <K0 DoTA read> <A2 DoTA bake> \\
        --set dada rows <K0 DADA read> <A2 DADA bake> \\
        --set tad rows <K0 TAD read> <A2 TAD bake> \\
        --metadata data/DoTA/metadata_val.json --split-file data/DoTA/val_split.txt \\
        --out-dir outputs/kim/REPORTS/form_probe
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
from core.data.dota import parse_metadata, read_split_ids
from core.data.stp_patches import motion_magnitude
from core.data.v2_inputs import check_input_manifest
from core.device import resolve_device
from core.models.kat_vad import KATVAD
from core.tools import kim_pilot
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.rate_matched_eval import Run, load_finished_model, to_native
from core.tools.stp_readout import ci as dota_ci
from core.tools.zs_readout import video_ci

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "kim_form_probe"
RULE = "core/docs/final/KIM_RUN.md §14"
FEATURES = (kim_pilot.KIM_INPUT, "clip")
REFERENCE, SIGNED, CENTRAL, SA2_MO = "", "s", "c", "m"
FORMS = (REFERENCE, SIGNED, CENTRAL, SA2_MO)
ENDPOINTS = (kim_pilot.DOTA_SET, kim_pilot.DADA_SET)
HOST_ROW = kim_pilot.HOST_ROW
GO, STOP = "go", "stop"

Curves = dict[str, np.ndarray]


def row_name(feature: str, form: str) -> str:
    return f"{kim_pilot.candidate_name(feature, 2)}{form}"


# ---------------------------------------------------------------------------
# pure pieces (tested without a model)
# ---------------------------------------------------------------------------
def form_scores(z: np.ndarray) -> dict[str, np.ndarray]:
    """``{form: (L,)}`` second-order change per step of one ``(L, D)`` sequence."""
    z = np.asarray(z, dtype=np.float64)
    magnitude = np.abs(z[2:] - 2.0 * z[1:-1] + z[:-2]).mean(axis=1) if len(z) > 2 else np.empty(0)
    signed, central = np.zeros(len(z)), np.zeros(len(z))
    signed[2:] = magnitude
    central[1:-1] = magnitude
    mo = motion_magnitude(z[:, None, :])[:, 0].astype(np.float64)
    return {REFERENCE: kim_pilot.change_scores(z)[2], SIGNED: signed, CENTRAL: central, SA2_MO: mo}


def macro_ci(name: str, per_video: dict[str, float]) -> dict[str, float] | None:
    """DoTA clusters clips by source; every other set by video (as ``kim_pilot``)."""
    ci = dota_ci if name in kim_pilot.EGO_SETS else video_ci
    return ci(per_video)


def summarize_set(
    name: str,
    curves: dict[str, Curves],
    host_scores: Curves,
    labels: Curves,
    ego: dict[str, bool] | None,
    seconds: float | None,
) -> dict[str, Any]:
    """Every number of one set (pure: curves in, numbers out)."""
    aucs = {c: kim_pilot.two_class_aucs(s, labels) for c, s in curves.items()}
    aucs[HOST_ROW] = kim_pilot.two_class_aucs(host_scores, labels)
    rows: dict[str, Any] = {}
    for cand, per_video in aucs.items():
        row: dict[str, Any] = {"macro": macro_ci(name, per_video)}
        if ego is not None:
            for split, flag in (("ego", True), ("non_ego", False)):
                row[split] = macro_ci(name, {v: a for v, a in per_video.items()
                                             if ego.get(v) is flag})
        rows[cand] = row
    deltas: dict[str, Any] = {}
    for feature in FEATURES:
        base = aucs[row_name(feature, REFERENCE)]
        for form in FORMS[1:]:
            cand = row_name(feature, form)
            deltas[cand] = macro_ci(name, {v: aucs[cand][v] - base[v]
                                           for v in sorted(set(aucs[cand]) & set(base))})
    rank: dict[str, Any] = {}
    lags: dict[str, Any] = {}
    for form in FORMS:
        cand = row_name(kim_pilot.KIM_INPUT, form)
        blended = {v: 0.5 * (kim_pilot.rank01(host_scores[v]) + kim_pilot.rank01(curves[cand][v]))
                   for v in labels}
        rank[cand] = macro_ci(name, kim_pilot.two_class_aucs(blended, labels))
        if seconds is not None:
            found = [kim_pilot.onset_lag(curves[cand][v], labels[v], seconds) for v in labels]
            lags[cand] = kim_pilot.lag_summary([lag for lag in found if lag is not None])
    return {"videos": len(labels), "two_class": len(aucs[HOST_ROW]), "rows": rows,
            "delta_vs_reference": deltas, "rank_with_host": rank, "lags": lags}


def check_baseline(
    host: str, step: int, sets: dict[str, dict[str, Any]], baseline: dict[str, Any]
) -> None:
    """Raise unless this run reproduces the recorded pilot's host, step and macros (§14)."""
    if baseline["host"] != host or baseline["global_step"] != step:
        raise ValueError(f"the pilot read {baseline['host']} step {baseline['global_step']}, "
                         f"this run reads {host} step {step}")
    for name, s in sets.items():
        recorded = baseline["sets"].get(name)
        if recorded is None:
            raise ValueError(f"the recorded pilot has no set {name}")
        for cand in (*(row_name(f, REFERENCE) for f in FEATURES), HOST_ROW):
            got, want = s["rows"][cand]["macro"]["mean"], recorded["rows"][cand]["macro"]["mean"]
            if abs(got - want) > constants.KIM_FORM_BASELINE_TOL:
                raise ValueError(f"{name} {cand}: macro {got:.6f}, the pilot recorded {want:.6f}")


def decide(sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """§14: a candidate passes iff it beats its reference by the margin on both endpoints."""
    missing = [s for s in ENDPOINTS if s not in sets]
    if missing:
        raise ValueError(f"§14 needs both endpoints, missing {missing}")
    margins: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        for form in FORMS[1:]:
            cand = row_name(feature, form)
            margins[cand] = {
                s: float(sets[s]["rows"][cand]["macro"]["mean"]
                         - sets[s]["rows"][row_name(feature, REFERENCE)]["macro"]["mean"])
                for s in ENDPOINTS
            }
    passes = [c for c, m in margins.items()
              if all(v >= constants.KIM_PILOT_INPUT_MARGIN for v in m.values())]
    return {"margin": constants.KIM_PILOT_INPUT_MARGIN, "margins": margins, "passes": passes,
            "verdict": GO if passes else STOP,
            "consequence": ("Claude may propose one arm with its own dated note; the user decides"
                            if passes else "the form is closed; nothing is trained")}


# ---------------------------------------------------------------------------
# run / render
# ---------------------------------------------------------------------------
def score_set(
    kind: str, input_dir: Path, labels: Curves, model: KATVAD, device: torch.device, stride: int
) -> dict[str, Curves]:
    """Candidate curves on the label grid."""
    curves: dict[str, Curves] = {row_name(f, form): {} for f in FEATURES for form in FORMS}
    for video_id in sorted(labels):
        rows = np.load(input_dir / f"{video_id}.npy")
        n = len(labels[video_id])
        expected = len(range(0, n, stride)) if kind == kim_pilot.KIND_NATIVE else n
        if len(rows) != expected:
            raise ValueError(f"{video_id}: {len(rows)} input rows, the labels need {expected}")
        seqs = kim_pilot.feature_sequences(model, rows, device)
        for feature in FEATURES:
            for form, steps in form_scores(seqs[feature]).items():
                curve = to_native(steps, stride, n) if kind == kim_pilot.KIND_NATIVE else steps
                curves[row_name(feature, form)][video_id] = curve
    return curves


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    model, step, _, cfg = load_finished_model(Run(args.name, args.ckpt), device)
    kim_pilot.check_model(model)
    baseline = json.loads(Path(args.baseline).read_text())
    ego = None
    if args.metadata is not None and args.split_file is not None:
        records = parse_metadata(args.metadata, read_split_ids(args.split_file))
        ego = {r.video_id: r.is_ego for r in records}
    sets: dict[str, dict[str, Any]] = {}
    for name, kind, eval_dir, input_dir in args.set:
        if kind not in kim_pilot.KINDS:
            raise ValueError(f"--set {name}: kind must be one of {kim_pilot.KINDS}, got {kind!r}")
        check_input_manifest(Path(input_dir), cfg.v2.crn, cfg.v2.motion)
        host_scores, labels = kim_pilot.read_labels(Path(eval_dir))
        stride = args.native_stride if kind == kim_pilot.KIND_NATIVE else 1
        curves = score_set(kind, Path(input_dir), labels, model, device, stride)
        sets[name] = summarize_set(name, curves, host_scores, labels,
                                   ego if name in kim_pilot.EGO_SETS else None,
                                   kim_pilot.DEFAULT_STEP_SECONDS.get(name))
        LOGGER.info("%s: %d videos scored", name, len(labels))
    check_baseline(args.name, step, sets, baseline)
    result = {"rule": RULE, "host": args.name, "checkpoint": str(args.ckpt), "global_step": step,
              "disclosure": constants.KIM_FORM_DISCLOSURE, "sets": sets, "decision": decide(sets)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{READOUT_NAME}.json", result)
    write_text_atomic(args.out_dir / f"{READOUT_NAME}.md", render_markdown(result))
    LOGGER.info("form probe -> %s", args.out_dir / f"{READOUT_NAME}.md")
    return result


def render_markdown(r: dict[str, Any]) -> str:
    num, delta = kim_pilot._num, kim_pilot._delta
    d = r["decision"]
    lines = [f"# KIM form probe (host {r['host']} s2024, step {r['global_step']}, no training) "
             "-- PRELIMINARY (n = 1)", "", f"Rule: `{r['rule']}`. **{r['disclosure']}**", "",
             f"**Verdict: {d['verdict']}.** Passes: {d['passes'] or 'none'}. {d['consequence']}.",
             "", f"Margin vs the same feature's reference, bar {d['margin']:+.2f} on both sets:",
             "", "| candidate | " + " | ".join(ENDPOINTS) + " |",
             "| --- | " + " | ".join("---" for _ in ENDPOINTS) + " |"]
    for cand, margin in d["margins"].items():
        lines.append(f"| {cand} | " + " | ".join(f"{margin[s]:+.4f}" for s in ENDPOINTS) + " |")
    lines.append("")
    for name, s in r["sets"].items():
        lines += [f"## {name} ({s['two_class']} two-class of {s['videos']})", "",
                  "| row | macro | row - reference | host + row (rank avg) | median abs lag (s) |",
                  "| --- | --- | --- | --- | --- |"]
        for cand, row in s["rows"].items():
            lag = s["lags"].get(cand)
            lines.append(f"| {cand} | {num(row['macro'])} "
                         f"| {delta(s['delta_vs_reference'].get(cand))} "
                         f"| {num(s['rank_with_host'].get(cand))} "
                         f"| {'—' if lag is None else format(lag['median_abs_s'], '.2f')} |")
        if any("ego" in row for row in s["rows"].values()):
            lines += ["", "| row | ego | non-ego |", "| --- | --- | --- |"]
            for cand, row in s["rows"].items():
                lines.append(f"| {cand} | {num(row.get('ego'))} | {num(row.get('non_ego'))} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--ckpt", type=Path, required=True, help="K0's checkpoint_last.pt")
    parser.add_argument("--name", required=True, help="the host arm; must equal the pilot's")
    parser.add_argument("--baseline", type=Path, required=True, help="the recorded kim_pilot.json")
    parser.add_argument(
        "--set", nargs=4, action="append", required=True,
        metavar=("NAME", "KIND", "EVAL_DIR", "INPUT_DIR"),
        help="KIND native (DoTA protocol B) | rows; EVAL_DIR = this host's read of the set",
    )
    parser.add_argument("--native-stride", type=int, default=constants.V2_E1_STRIDE_BC)
    parser.add_argument("--metadata", type=Path, default=None, help="DoTA metadata_val.json")
    parser.add_argument("--split-file", type=Path, default=None, help="DoTA val_split.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
