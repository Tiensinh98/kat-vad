"""Zero-shot TAD / DADA-2000 read-out: SA² (R4) vs its host (W0), Nexar-trained, s2024.

Rule: ``core/docs/final/ZS_TAD_DADA.md`` §3 (written before any number). Pure aggregation --
no model is loaded. Inputs are ordinary protocol outputs laid out by
``colab/final/zs_tad_dada_r4.ipynb``::

    <root>/eval/tad/<arm>/s<seed>/{results.json, scores/<video>.npz}   core.evaluate (§26)
    <root>/eval/dada/<arm>/{dada_readout.json, clip_aucs.json}         dada_whole_eval

Per set, the one sentence: per-video Δ (R4 - W0) of the frame AUC over the two-class videos,
mean with a 95 % cluster bootstrap (cluster = the video), "gain shown" iff the low bound > 0
and the mean ≥ +0.01, "cost shown" iff the high bound < 0, otherwise "no difference shown";
with one seed it carries **PRELIMINARY (n = 1 seed)** and adopts nothing.

Printed, never deciding: v2 Nexar A2 (W0's function) at s2024 (the reproduction check: W0 must
equal it to ``ZS_REPRO_TOL``) and its 5-seed per-frame average (v2's table; seed averaging
smooths the curves, so R4 - A2-mean is biased toward A2), micro, clip-level AUC, the position
rulers, and the training diagnostics of both runs.

CLI::

    python -m core.tools.zs_readout --root outputs/zs --v2-root .../v2_nexar_window \\
        --runs W0 .../wu/W0/s2024 --runs R4 .../stp/R4/s2024
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.metrics import cluster_bootstrap_ci
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_transfer_readout import ArmRead, read_dada, read_tad
from core.tools.stp_readout import _ci, _diag, _num, paired, run_diagnostics, sentence

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "zs_readout"
RULE = "core/docs/final/ZS_TAD_DADA.md §3"
EVAL_DIR = "eval"
TAD, DADA = "tad", "dada"
SETS = (TAD, DADA)
SET_TITLE = {
    TAD: "TAD test (100 videos, s8, §26's protocol, raw scores)",
    DADA: "DADA-2000 whole videos (370 sources of T2-test, s8)",
}
V2_ARM = "A2"
V2_SEEDS = constants.NEXAR_RUN_SEEDS


def video_ci(values: dict[str, float]) -> dict[str, float] | None:
    """Mean and 95 % bootstrap over videos (each video its own cluster)."""
    return cluster_bootstrap_ci(values, {v: v for v in values}, constants.V2_E1_BOOTSTRAP,
                                constants.V5_BOOTSTRAP_SEED, constants.V2_E1_CI)


def read_arm(set_name: str, arm_dir: Path, seeds: list[int]) -> ArmRead:
    if set_name == TAD:
        return read_tad(arm_dir, seeds, constants.V5_BOOTSTRAP_SEED)
    return read_dada(arm_dir)


def delta(a: ArmRead, b: ArmRead) -> dict[str, Any]:
    d = paired(a["items"], b["items"])
    c = video_ci(d)
    return {"delta": c, "sentence": sentence(c)}


def row(r: ArmRead) -> dict[str, Any]:
    return {"macro": r["macro"], "per_seed": r["per_seed"], "micro": r["micro"],
            "clip_auc": r["clip_auc"], "extra": r["extra"]}


def read_set(set_name: str, root: Path, v2_root: Path, seed: int) -> dict[str, Any]:
    base = root / EVAL_DIR / set_name
    arms = {a: read_arm(set_name, base / a, [seed]) for a in constants.ZS_ARMS}
    v2_dir = v2_root / EVAL_DIR / set_name / V2_ARM
    a2_mean = read_arm(set_name, v2_dir, list(V2_SEEDS)) if v2_dir.is_dir() else None
    a2_seed = (read_arm(set_name, v2_dir, [seed])
               if v2_dir.is_dir() and set_name == TAD else None)  # DADA keeps no per-seed curves
    control, sa2 = arms[constants.ZS_CONTROL_ARM], arms[constants.ZS_SA2_ARM]
    out: dict[str, Any] = {
        "title": SET_TITLE[set_name],
        "rows": {name: row(r) for name, r in arms.items()},
        "primary": delta(sa2, control),
        "a2_v2_mean": row(a2_mean) if a2_mean is not None else None,
        "sa2_minus_a2_mean": delta(sa2, a2_mean) if a2_mean is not None else None,
        "reproduction": None,
    }
    if a2_mean is not None:
        name = f"s{seed}"
        ref = a2_mean["per_seed"].get(name)
        got = control["per_seed"].get(name)
        if ref is not None and got is not None:
            out["reproduction"] = {"w0": got, "a2_v2": ref, "gap": got - ref,
                                   "flag": abs(got - ref) > constants.ZS_REPRO_TOL}
    if a2_seed is not None:
        out["w0_minus_a2_same_seed"] = delta(control, a2_seed)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed = args.seed
    result: dict[str, Any] = {
        "rule": RULE,
        "control": constants.ZS_CONTROL_ARM,
        "arm": constants.ZS_SA2_ARM,
        "seed": seed,
        "label": "PRELIMINARY (n = 1 seed)",
        "sets": {s: read_set(s, args.root, args.v2_root, seed) for s in SETS
                 if (args.root / EVAL_DIR / s).is_dir()},
        "diagnostics": {},
    }
    if not result["sets"]:
        raise FileNotFoundError(f"no {EVAL_DIR}/{{tad,dada}} under {args.root}")
    for name, run_dir in args.runs or []:
        diag = run_diagnostics(Path(run_dir))
        if diag is not None:
            result["diagnostics"][name] = diag
    out_dir = args.out_dir or args.root / "REPORTS"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / f"{READOUT_NAME}.json", result)
    write_text_atomic(out_dir / f"{READOUT_NAME}.md", render_markdown(result))
    LOGGER.info("read-out -> %s", out_dir / f"{READOUT_NAME}.md")
    return result


def _f(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def _ruler(extra: dict[str, Any]) -> str:
    parts = []
    for key in ("middle_ruler", "monotone_ruler"):
        if extra.get(key) is not None:
            parts.append(f"{key.replace('_', ' ')} {_num(extra[key])}")
    for key in ("clip_oracle_micro", "length_ruler_micro"):
        if extra.get(key) is not None:
            parts.append(f"{key.replace('_', ' ')} {extra[key]:.4f}")
    return " · ".join(parts)


def render_markdown(r: dict[str, Any]) -> str:
    label = r["label"]
    lines = [
        f"# Zero-shot TAD / DADA-2000 — {r['arm']} (SA²) vs {r['control']} (host) — **{label}**",
        "",
        f"Rule: `{r['rule']}`. Both Nexar-trained, seed {r['seed']}, "
        "prefix OFF (released prompts).",
        "",
    ]
    for s in r["sets"].values():
        lines += [
            f"## {s['title']}",
            "",
            "| row | macro [95 % CI over videos] | micro | clip AUC |",
            "|---|---|---|---|",
        ]
        for name, x in s["rows"].items():
            lines.append(f"| {name} s{r['seed']} | {_num(x['macro'])} | {_f(x['micro'])} | "
                         f"{_f(x['clip_auc'])} |")
        if s["a2_v2_mean"] is not None:
            a = s["a2_v2_mean"]
            lines.append(f"| v2 A2, 5-seed per-frame mean (printed) | {_num(a['macro'])} | "
                         f"{_f(a['micro'])} | {_f(a['clip_auc'])} |")
        p = s["primary"]
        lines += [
            "",
            f"* **{r['arm']} - {r['control']} (paired, per video): {_ci(p['delta'])} → "
            f"{p['sentence']} — {label}**",
        ]
        if s["sa2_minus_a2_mean"] is not None:
            lines.append(f"* printed: {r['arm']} - v2 A2 5-seed mean: "
                         f"{_ci(s['sa2_minus_a2_mean']['delta'])} (biased toward A2: averaging)")
        rep = s["reproduction"]
        if rep is not None:
            flag = " **FLAG: W0 ≠ A2 at the same seed**" if rep["flag"] else ""
            lines.append(f"* reproduction: W0 {rep['w0']:.4f} vs v2 A2 s{r['seed']} "
                         f"{rep['a2_v2']:.4f} (gap {rep['gap']:+.5f}){flag}")
        lines += [f"* rulers (no pixels): {_ruler(s['rows'][r['control']]['extra'])}", ""]
    if r["diagnostics"]:
        lines += [
            "Training diagnostics (first → last logged batch):",
            "",
            "| run | step | SA² entropy / log K | ‖W_q‖ | prompt norm | rho_u | ‖W_u‖ | mil |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, d in r["diagnostics"].items():
            lines.append(
                f"| {name} | {d['last_step']} | {_diag(d, 'stp_entropy')} | "
                f"{_diag(d, 'stp_wq_norm')} | {_diag(d, 'prompt_norm')} | "
                f"{_diag(d, 'motion_share')} | {_diag(d, 'w_u_norm')} | {_diag(d, 'mil')} |"
            )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--root", type=Path, required=True, help="outputs/zs (eval/ under it)")
    parser.add_argument("--v2-root", type=Path, required=True,
                        help="v2's outputs/v2_nexar_window (A2 printed beside)")
    parser.add_argument("--runs", nargs=2, metavar=("NAME", "DIR"), action="append",
                        help="a run dir with metrics.jsonl (training diagnostics)")
    parser.add_argument("--seed", type=int, default=constants.ZS_SEED)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <root>/REPORTS")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
