"""Normal-text probe: does K5 care that its "Normal" definition says *surveillance*? (no train)

Rule ``core/docs/final/KIM_RUN.md`` §15, written before any number. Inference only on K5's
finished checkpoints; nothing is trained, extracted or baked. Three sub-commands:

* ``alpha`` -- step 1: read ``bin_head.adp_alpha`` from each checkpoint and print the pre-fusion
  weight ``w = sigmoid(alpha * ADAPTIVE_FUSE_SCALE)`` of ``H_bin`` (the branch that never sees
  text gets ``w``, the fused branch ``1 - w``).
* ``score`` -- step 2: run ``protocol_b_eval`` (DoTA-CAP-dev) or ``dada_whole_eval`` (DADA)
  *in this process*, with the DoTA/DADA "Normal" definitions either left as they are
  (``--normal original``, the gate) or swapped for ``NORMAL_PROBE_DASHCAM_DEFS``
  (``--normal dashcam``). The swap keeps the list length, so the per-item verbalizer draws the
  same indices and only the Normal sentence changes.
* ``report`` -- the gate (in-process original == recorded K5 read), the paired deltas and §15's
  verdict.

CLI::

    python -m core.tools.normal_text_probe alpha --run s2024 <ckpt> --run s2025 <ckpt> \\
        --out outputs/kim/REPORTS/normal_text_probe/alpha.json
    python -m core.tools.normal_text_probe score --tool dota --normal dashcam -- \\
        <every protocol_b_eval argument>
    python -m core.tools.normal_text_probe report --alpha <alpha.json> \\
        --read dota s2024 <recorded> <original> <dashcam> ... --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch

from core import constants
from core.data.definitions import DATASET_CLS_DEFS
from core.tools import dada_whole_eval, protocol_b_eval
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.stp_readout import ci as dota_ci
from core.tools.zs_readout import video_ci

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "normal_text_probe"
RULE = "core/docs/final/KIM_RUN.md §15"
ORIGINAL, DASHCAM = "original", "dashcam"
NORMAL_MODES = (ORIGINAL, DASHCAM)
DOTA, DADA = "dota", "dada"
SETS = (DOTA, DADA)
TOOLS: dict[str, Callable[[list[str] | None], None]] = {
    DOTA: protocol_b_eval.main,
    DADA: dada_whole_eval.main,
}
READOUT_FILES = {DOTA: protocol_b_eval.READOUT_JSON, DADA: "dada_readout.json"}
CLIP_AUCS = protocol_b_eval.CLIP_AUCS_JSON
NULL, GAIN, COST, MIXED = "null", "gain", "cost", "mixed"
CONSEQUENCES = {
    NULL: "the Normal wording does not matter at inference; nothing is trained",
    GAIN: "Claude may propose one retrain arm with dashcam Normal text, with its own dated note; "
          "the user decides",
    COST: "uninterpretable: K5 was trained with the original text, so a swap at inference is a "
          "train/test mismatch; nothing is concluded about training and nothing is trained",
    MIXED: "no claim; nothing is trained",
}


# ---------------------------------------------------------------------------
# pure pieces (tested without a model)
# ---------------------------------------------------------------------------
def prefusion_weight(alpha: float, scale: float = constants.ADAPTIVE_FUSE_SCALE) -> float:
    """``H_bin``'s weight on the branch that never sees text: ``sigmoid(alpha * scale)``."""
    return 1.0 / (1.0 + math.exp(-alpha * scale))


def normal_override(mode: str) -> dict[str, list[str]]:
    """The ``{Normal: sentences}`` patch for ``mode`` (empty for the original text)."""
    if mode not in NORMAL_MODES:
        raise ValueError(f"--normal must be one of {NORMAL_MODES}, got {mode!r}")
    if mode == ORIGINAL:
        return {}
    original = DATASET_CLS_DEFS[constants.NORMAL_PROBE_VERBALIZER_KEY][constants.NORMAL_PROBE_CLASS]
    swapped = list(constants.NORMAL_PROBE_DASHCAM_DEFS)
    if len(swapped) != len(original):
        raise ValueError(f"the swap has {len(swapped)} Normal sentences, the original "
                         f"{len(original)}: the verbalizer would draw different indices")
    return {constants.NORMAL_PROBE_CLASS: swapped}


def macro_ci(set_name: str, per_clip: dict[str, float]) -> dict[str, float] | None:
    """DoTA clusters clips by source video; DADA by video (as every K5 read)."""
    return (dota_ci if set_name == DOTA else video_ci)(per_clip)


def paired_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """Per-clip ``a - b`` over the clips both reads scored."""
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("the two reads share no scored clip")
    return {v: a[v] - b[v] for v in common}


def check_gate(set_name: str, seed: str, recorded: float, original: float) -> None:
    """§15 gate: the in-process original-text macro must equal the recorded K5 read."""
    if abs(recorded - original) > constants.NORMAL_PROBE_REPRO_TOL:
        raise ValueError(f"{set_name} {seed}: in-process original macro {original:.6f}, the "
                         f"recorded K5 read {recorded:.6f} "
                         f"(tol {constants.NORMAL_PROBE_REPRO_TOL})")


def decide(deltas: dict[str, dict[str, float]], pooled: dict[str, dict[str, float] | None]) -> str:
    """§15 verdict from per-seed mean deltas ``{set: {seed: d}}`` and pooled CIs ``{set: ci}``."""
    margin = constants.NORMAL_PROBE_MARGIN
    missing = [s for s in SETS if s not in deltas or not deltas[s]]
    if missing:
        raise ValueError(f"§15 needs both sets, missing {missing}")
    if any(d <= -margin for s in SETS for d in deltas[s].values()):
        return COST
    pooled_means = {s: float(np.mean(list(deltas[s].values()))) for s in SETS}
    if all(abs(m) < margin for m in pooled_means.values()):
        return NULL
    gain = all(d >= margin for s in SETS for d in deltas[s].values()) and all(
        (c := pooled[s]) is not None and c["low"] > 0 for s in SETS)
    return GAIN if gain else MIXED


# ---------------------------------------------------------------------------
# sub-commands
# ---------------------------------------------------------------------------
def run_alpha(runs: list[list[str]], out: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, ckpt in runs:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)  # nosec B614 - own ckpt
        state = payload["model"]
        if constants.NORMAL_PROBE_ADP_ALPHA_KEY not in state:
            raise KeyError(f"{ckpt} has no {constants.NORMAL_PROBE_ADP_ALPHA_KEY}: not an "
                           "adaptive H_bin")
        alpha = float(state[constants.NORMAL_PROBE_ADP_ALPHA_KEY].reshape(-1)[0])
        w = prefusion_weight(alpha)
        rows[name] = {"ckpt": str(ckpt), "global_step": int(payload.get("global_step", -1)),
                      "alpha": alpha, "w_prefusion": w, "w_text_branch": 1.0 - w}
        LOGGER.info("%s: alpha %.4f -> pre-fusion w %.4f, text branch %.4f", name, alpha, w, 1 - w)
    init_w = prefusion_weight(constants.ADAPTIVE_FUSE_ALPHA0)
    result = {"rule": RULE, "init_alpha": constants.ADAPTIVE_FUSE_ALPHA0,
              "init_w_prefusion": init_w, "minor_bar": constants.NORMAL_PROBE_TEXT_MINOR_W,
              "runs": rows,
              "text_branch_minor": all(r["w_prefusion"] >= constants.NORMAL_PROBE_TEXT_MINOR_W
                                       for r in rows.values())}
    write_json_atomic(out, result)
    return result


def run_score(tool: str, mode: str, tool_argv: list[str]) -> None:
    """Run one eval tool in this process with the Normal definitions patched for ``mode``."""
    if tool not in TOOLS:
        raise ValueError(f"--tool must be one of {sorted(TOOLS)}, got {tool!r}")
    target = DATASET_CLS_DEFS[constants.NORMAL_PROBE_VERBALIZER_KEY]
    with mock.patch.dict(target, normal_override(mode)):
        LOGGER.info("%s, Normal = %s: %s", tool, mode, target[constants.NORMAL_PROBE_CLASS])
        TOOLS[tool](tool_argv)


def read_dir(set_name: str, d: Path) -> tuple[float, dict[str, float]]:
    macro = json.loads((d / READOUT_FILES[set_name]).read_text())["macro"]["mean"]
    return float(macro), {k: float(v) for k, v in json.loads((d / CLIP_AUCS).read_text()).items()}


def run_report(alpha_path: Path, reads: list[list[str]], out_dir: Path) -> dict[str, Any]:
    alpha = json.loads(alpha_path.read_text())
    per_set: dict[str, Any] = {s: {} for s in SETS}
    clip_deltas: dict[str, dict[str, dict[str, float]]] = {s: {} for s in SETS}
    for set_name, seed, recorded_dir, original_dir, dashcam_dir in reads:
        if set_name not in SETS:
            raise ValueError(f"--read set must be one of {SETS}, got {set_name!r}")
        rec_macro, _ = read_dir(set_name, Path(recorded_dir))
        orig_macro, orig_aucs = read_dir(set_name, Path(original_dir))
        check_gate(set_name, seed, rec_macro, orig_macro)
        dash_macro, dash_aucs = read_dir(set_name, Path(dashcam_dir))
        d = paired_delta(dash_aucs, orig_aucs)
        clip_deltas[set_name][seed] = d
        per_set[set_name][seed] = {"recorded": rec_macro, "original": orig_macro,
                                   "dashcam": dash_macro, "delta": macro_ci(set_name, d)}
    pooled: dict[str, dict[str, float] | None] = {}
    for s in SETS:
        seeds = clip_deltas[s]
        common = sorted(set.intersection(*(set(d) for d in seeds.values()))) if seeds else []
        pooled[s] = macro_ci(s, {v: float(np.mean([seeds[k][v] for k in seeds])) for v in common})
    means = {s: {seed: float(r["delta"]["mean"]) for seed, r in per_set[s].items() if r["delta"]}
             for s in SETS}
    verdict = decide(means, pooled)
    result = {"rule": RULE, "alpha": alpha, "sets": per_set, "pooled": pooled,
              "dashcam_defs": list(constants.NORMAL_PROBE_DASHCAM_DEFS),
              "decision": {"margin": constants.NORMAL_PROBE_MARGIN, "verdict": verdict,
                           "consequence": CONSEQUENCES[verdict]}}
    write_json_atomic(out_dir / f"{READOUT_NAME}.json", result)
    write_text_atomic(out_dir / f"{READOUT_NAME}.md", render_markdown(result))
    return result


def _fmt(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def render_markdown(r: dict[str, Any]) -> str:
    a = r["alpha"]
    lines = [f"# Normal-text probe (K5, inference only) — {r['rule']}", "",
             "## Step 1: H_bin's blend", "",
             f"Init: alpha {a['init_alpha']}, pre-fusion w {a['init_w_prefusion']:.4f}. "
             f"Minor-text bar: w ≥ {a['minor_bar']}.", "",
             "| run | step | alpha | pre-fusion w | text branch 1 - w |", "|---|---|---|---|---|"]
    for name, row in a["runs"].items():
        lines.append(f"| {name} | {row['global_step']} | {row['alpha']:+.4f} | "
                     f"{row['w_prefusion']:.4f} | {row['w_text_branch']:.4f} |")
    lines += ["", f"text branch minor at every seed: **{a['text_branch_minor']}**", "",
              "## Step 2: dashcam - original Normal text (paired per clip, 95 % CI)", "",
              "| set | seed | recorded | original (gate) | dashcam | Δ |",
              "|---|---|---|---|---|---|"]
    for s, seeds in r["sets"].items():
        for seed, row in seeds.items():
            lines.append(f"| {s} | {seed} | {row['recorded']:.4f} | {row['original']:.4f} | "
                         f"{row['dashcam']:.4f} | {_fmt(row['delta'])} |")
        lines.append(f"| {s} | pooled | | | | {_fmt(r['pooled'][s])} |")
    d = r["decision"]
    lines += ["", f"**Verdict: {d['verdict']}** (margin {d['margin']}) — {d['consequence']}.", "",
              "Dashcam Normal sentences: " + " / ".join(f"`{t}`" for t in r["dashcam_defs"]), ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K5 Normal-text probe (KIM_RUN.md §15)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("alpha", help="step 1: H_bin's learned pre-/post-fusion blend")
    p.add_argument("--run", nargs=2, action="append", required=True, metavar=("NAME", "CKPT"))
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score", help="step 2: one eval tool with the Normal text patched")
    p.add_argument("--tool", choices=sorted(TOOLS), required=True)
    p.add_argument("--normal", choices=NORMAL_MODES, required=True)
    p.add_argument("tool_args", nargs=argparse.REMAINDER, help="after --: the tool's own arguments")
    p = sub.add_parser("report", help="gate, paired deltas, §15 verdict")
    p.add_argument("--alpha", type=Path, required=True)
    p.add_argument("--read", nargs=5, action="append", required=True,
                   metavar=("SET", "SEED", "RECORDED", "ORIGINAL", "DASHCAM"))
    p.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.cmd == "alpha":
        run_alpha(args.run, args.out)
    elif args.cmd == "score":
        tool_args = args.tool_args[1:] if args.tool_args[:1] == ["--"] else args.tool_args
        run_score(args.tool, args.normal, tool_args)
    else:
        result = run_report(args.alpha, args.read, args.out_dir)
        LOGGER.info("verdict %s", result["decision"]["verdict"])


if __name__ == "__main__":
    main()
