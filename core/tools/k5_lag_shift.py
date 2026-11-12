"""K5 lag shift: advance the score curve by a lag fitted on ``nexar_val``, read zero-shot once.

``KIM_RUN.md`` §23 (rule written 2026-11-12, before any number). §22.3 found K5's in-domain
response peaking about +0.45 s after ``t_event``. This tool fits one lag δ on the ``nexar_val``
positives (whole-video macro, LS3), then reads DoTA-CAP-dev and DADA-2000 at that δ against
δ = 0: macro paired per video, min-max micro (§21.1 BM7(i)), cluster bootstrap. Saved scores
only. The shift uses future frames, so every row is an **offline protocol change**, never a
model result.

Inputs::

    --nexar-run     <kim_gate>/eval/nexar/K5/   (scores/<video>.npz {score, gt}, tadz format)
    --nexar-readout <kim_gate>/eval/nexar/K5/nexar_readout.json   (G1)
    --census        <v2>/REPORTS/nexar_n0/census.json      duration_s per video (LS1)
    --dota          <kim>/eval/dota/K5/  (clip_scores.npz + protocol_b_readout.json)
    --dada          <kim>/eval/dada/K5/  (clip_scores.npz + dada_readout.json)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.v2_splits import dota_group
from core.metrics import cluster_bootstrap_ci, normalize_scores
from core.tools import tadz_readout as td
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import CLIP_SCORES_NPZ, READOUT_JSON, read_clip_scores
from core.tools.tad_clip_gate import micro_bootstrap_pairs
from core.tools.zs_lagovad_readout import versus_sentence

LOGGER = logging.getLogger(__name__)

Curves = dict[str, np.ndarray]
SEED = constants.V5_BOOTSTRAP_SEED
DRAWS = constants.BASEMICRO_BOOTSTRAP_DRAWS
DADA_READOUT_JSON = "dada_readout.json"
DOTA, DADA = "DoTA-CAP-dev", "DADA-2000"
STEPS_PER_S = {DOTA: float(constants.DOTA_FPS),
               DADA: constants.DADA_ASSUMED_FPS / constants.FRAME_STRIDE}
ZERO, FITTED = "delta 0", "delta*"


def grid() -> list[float]:
    """§23 LS2: 0.00 ... LAGSHIFT_GRID_MAX_S in LAGSHIFT_GRID_STEP_S steps (rounded, exact)."""
    n = round(constants.LAGSHIFT_GRID_MAX_S / constants.LAGSHIFT_GRID_STEP_S)
    return [round(i * constants.LAGSHIFT_GRID_STEP_S, 10) for i in range(n + 1)]


def shift_curve(scores: np.ndarray, steps: float) -> np.ndarray:
    """§23 LS1: ``s(i + steps)`` by linear interpolation, clamped to the last step."""
    if steps < 0:
        raise ValueError(f"the shift advances the curve; steps must be >= 0, not {steps}")
    x = np.asarray(scores, dtype=np.float64)
    if steps == 0:
        return x.copy()
    index = np.arange(len(x), dtype=np.float64) + steps
    return np.interp(index, np.arange(len(x), dtype=np.float64), x)


def shift_all(scores: Curves, rate: dict[str, float] | float, delta: float) -> Curves:
    """Every curve advanced by ``delta`` seconds; ``rate`` = steps per second (per video or one)."""
    out: Curves = {}
    for v, s in scores.items():
        r = rate[v] if isinstance(rate, dict) else rate
        out[v] = shift_curve(s, delta * r)
        if len(out[v]) != len(s):
            raise AssertionError(f"G2: {v} changed length under the shift")
    return out


def macro(scores: Curves, labels: Curves) -> float:
    return float(np.mean(list(td.clip_aucs(scores, labels).values())))


def minmax(scores: Curves) -> Curves:
    return {v: normalize_scores(s, constants.SCORE_NORM_MINMAX) for v, s in scores.items()}


def load_nexar(run_dir: Path, census_path: Path) -> tuple[Curves, Curves, dict[str, float]]:
    """``nexar_val`` curves (``run_dir/scores``) and steps per second (``len / duration_s``, G3)."""
    scores, labels = td.load_run(run_dir)
    videos = json.loads(census_path.read_text(encoding="utf-8"))["videos"]
    missing = sorted(v for v in labels if v not in videos)
    if missing:
        raise ValueError(f"G3: {len(missing)} nexar_val videos have no census row: {missing[:5]}")
    rate = {v: len(scores[v]) / float(videos[v]["duration_s"]) for v in scores}
    return scores, labels, rate


def fit_delta(scores: Curves, labels: Curves, rate: dict[str, float]) -> dict[str, Any]:
    """§23 LS3: the grid's argmax of whole-video macro on the positives; ties -> smallest δ."""
    curve = {d: macro(shift_all(scores, rate, d), labels) for d in grid()}
    best = max(curve.values())
    chosen = min(d for d, m in curve.items() if best - m <= constants.LAGSHIFT_TIE_TOL)
    return {"delta": chosen, "macro_by_delta": curve,
            "pooled_raw_micro": {ZERO: td.micro(scores, labels),
                                 FITTED: td.micro(shift_all(scores, rate, chosen), labels)}}


def check_close(name: str, got: float, want: float | None, tol: float) -> None:
    if want is None or abs(got - want) > tol:
        raise ValueError(f"G1: {name} {got:.8f} at delta 0 != {want} recorded (tol {tol})")


def _recorded(readout: Path, key: str) -> float | None:
    value = json.loads(readout.read_text(encoding="utf-8")).get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    return None if value is None else float(value)


def read_zero_shot(name: str, run_dir: Path, delta: float) -> dict[str, Any]:
    """One zero-shot set at δ* against δ = 0 (LS4), gated at δ = 0 (G1) and printed on the grid."""
    readout = run_dir / (READOUT_JSON if name == DOTA else DADA_READOUT_JSON)
    scores, labels = read_clip_scores(run_dir / CLIP_SCORES_NPZ)
    check_close(f"{name} macro", macro(scores, labels), _recorded(readout, "macro"),
                constants.BASEMICRO_MACRO_TOL)
    if name == DADA:
        check_close(f"{name} raw micro", td.micro(scores, labels), _recorded(readout, "micro"),
                    constants.BASEMICRO_MICRO_TOL)
    rate = STEPS_PER_S[name]
    curves = {ZERO: scores, FITTED: shift_all(scores, rate, delta)}
    pooled = {k: minmax(c) for k, c in curves.items()}
    groups = {v: dota_group(v) for v in labels} if name == DOTA else {v: v for v in labels}
    per = {k: td.clip_aucs(c, labels) for k, c in curves.items()}
    diff = {v: per[FITTED][v] - per[ZERO][v] for v in per[ZERO]}
    macro_ci = cluster_bootstrap_ci(diff, groups, DRAWS, SEED, constants.V2_E1_CI)
    micro_ci = micro_bootstrap_pairs(pooled, {"micro": (FITTED, ZERO)}, labels, SEED,
                                     groups=None if name == DADA else groups,
                                     draws=DRAWS)["micro"]
    grid_rows = {}
    for d in grid():
        shifted = shift_all(scores, rate, d)
        grid_rows[d] = {"macro": macro(shifted, labels), "micro": td.micro(minmax(shifted), labels)}
    return {
        "set": name, "videos": len(labels), "clusters": len(set(groups.values())),
        "steps_per_s": rate, "shift_steps": delta * rate,
        "rows": {k: {"macro": float(np.mean(list(per[k].values()))),
                     "micro_minmax": td.micro(pooled[k], labels)} for k in curves},
        "delta_macro": macro_ci, "macro_sentence": versus_sentence(macro_ci),
        "delta_micro": micro_ci, "micro_sentence": versus_sentence(micro_ci),
        "grid_printed": grid_rows,
    }


def follow_up(sets: dict[str, dict[str, Any]]) -> str:
    """§23 LS7: the only sentence the rule allows about what comes next."""
    above = [n for n, s in sets.items() if s["macro_sentence"] == "above"]
    if len(above) == len(sets):
        return "macro above on both zero-shot sets: Claude may propose §22.5 (b); the user decides"
    if above:
        return f"macro above on {above[0]} only: reported, nothing proposed"
    return "macro not above on either set: §22.5 (b) is not supported by this read"


def run(args: argparse.Namespace) -> dict[str, Any]:
    scores, labels, rate = load_nexar(args.nexar_run, args.census)
    whole = json.loads(args.nexar_readout.read_text(encoding="utf-8"))["whole"]
    check_close("nexar_val whole macro", macro(scores, labels), float(whole["macro"]["mean"]),
                constants.LAGSHIFT_NEXAR_MACRO_TOL)
    fit = fit_delta(scores, labels, rate)
    LOGGER.info("delta* = %.2f s (nexar_val macro %.4f -> %.4f)", fit["delta"],
                fit["macro_by_delta"][0.0], fit["macro_by_delta"][fit["delta"]])
    sets = {DOTA: read_zero_shot(DOTA, args.dota, fit["delta"]),
            DADA: read_zero_shot(DADA, args.dada, fit["delta"])}
    report = {"rule": "KIM_RUN.md §23", "seed": SEED, "draws": DRAWS,
              "nexar_positives": len(td.clip_aucs(scores, labels)), "fit": fit, "sets": sets,
              "follow_up": follow_up(sets)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / constants.LAGSHIFT_REPORT_STEM
    write_json_atomic(stem.with_suffix(".json"), report)
    write_text_atomic(stem.with_suffix(".md"), render_markdown(report))
    return report


def _ci(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def render_markdown(report: dict[str, Any]) -> str:
    fit = report["fit"]
    d = fit["delta"]
    lines = [
        "# K5 offline lag shift (`KIM_RUN.md` §23)", "",
        "PRELIMINARY (n = 1, K5 s2024). Saved scores only. The shift uses future frames: an "
        "**offline protocol change**, never a model result, never beside a published number.", "",
        f"## Fit on `nexar_val` ({report['nexar_positives']} positives, whole-video macro)", "",
        f"**δ\\* = {d:.2f} s**; macro {fit['macro_by_delta'][0.0]:.4f} at δ = 0 → "
        f"{fit['macro_by_delta'][d]:.4f}. Pooled raw micro {fit['pooled_raw_micro'][ZERO]:.4f} → "
        f"{fit['pooled_raw_micro'][FITTED]:.4f} (printed).", "",
        "| δ (s) | " + " | ".join(f"{x:.2f}" for x in fit["macro_by_delta"]) + " |",
        "|---|" + "---:|" * len(fit["macro_by_delta"]),
        "| macro | " + " | ".join(f"{m:.4f}" for m in fit["macro_by_delta"].values()) + " |", "",
        f"## Zero-shot read at δ\\* = {d:.2f} s (bootstrap {report['draws']} draws, seed "
        f"{report['seed']})", "",
        "| set | shift (steps) | macro δ=0 → δ* | Δ macro [95 %] | sentence | min-max micro "
        "δ=0 → δ* | Δ micro [95 %] | sentence |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for name, s in report["sets"].items():
        r = s["rows"]
        lines.append(
            f"| {name} ({s['videos']} videos, {s['clusters']} clusters) | {s['shift_steps']:.2f} | "
            f"{r[ZERO]['macro']:.4f} → {r[FITTED]['macro']:.4f} | {_ci(s['delta_macro'])} | "
            f"{s['macro_sentence']} | {r[ZERO]['micro_minmax']:.4f} → "
            f"{r[FITTED]['micro_minmax']:.4f} | {_ci(s['delta_micro'])} | {s['micro_sentence']} |")
    lines += ["", f"**LS7:** {report['follow_up']}.", "",
              "## Printed, never choosing δ (post-hoc): the grid on the zero-shot sets", ""]
    for name, s in report["sets"].items():
        g = s["grid_printed"]
        best = max(g, key=lambda x: g[x]["macro"])
        lines += [f"**{name}**: best macro on the grid at δ = {best:.2f} s "
                  f"({g[best]['macro']:.4f}); δ* gives {g[d]['macro']:.4f}.", "",
                  "| δ (s) | " + " | ".join(f"{x:.2f}" for x in g) + " |",
                  "|---|" + "---:|" * len(g),
                  "| macro | " + " | ".join(f"{v['macro']:.4f}" for v in g.values()) + " |",
                  "| micro | " + " | ".join(f"{v['micro']:.4f}" for v in g.values()) + " |", ""]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument("--nexar-run", type=Path, required=True,
                        help="nexar_val run dir holding scores/ (nexar_eval --save-scores)")
    parser.add_argument("--nexar-readout", type=Path, required=True, help="nexar_readout.json")
    parser.add_argument("--census", type=Path, required=True, help="nexar_n0/census.json")
    parser.add_argument("--dota", type=Path, required=True, help="K5 protocol-B read dir")
    parser.add_argument("--dada", type=Path, required=True, help="K5 DADA whole read dir")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
