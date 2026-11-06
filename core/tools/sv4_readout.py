"""SV4 read-out: does x^AS help when it enters the host through a zero-init matrix ``W_p``?

Rule ``core/docs/final/SA2V2_RUN.md`` §11, written before any SV4 number. Rows W0D (the paired
control), SV3 (scalar gate), SV4 (``W_p``); W0 printed. Primary per set: ``SV4 - W0D`` with §3's
sentence. ``W_p`` reads "used" iff the tail mean of ``||W_p x^AS|| / ||c||`` is at least
``SV4_USED_SHARE``. The verdict sentence (§11.3) and the stop rule (§11.4) are fixed before the
numbers exist.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.sa2v2_readout import PRELIMINARY, _json
from core.tools.stp_readout import COST, GAIN
from core.tools.stp_readout import ci as dota_ci
from core.tools.sv3_readout import _ci, _num
from core.tools.w0d_readout import dada_items, dota_items, paired_contrasts
from core.tools.zs_readout import video_ci

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "sv4_readout"
RULE = "core/docs/final/SA2V2_RUN.md §11"
ROWS = ("W0", "W0D", "SV3", "SV4")
PAIRS = (("SV4", "W0D", "W_p module on the dropout host (primary)"),
         ("SV4", "SV3", "matrix vs gate (printed; other scorer draws, D13)"),
         ("SV4", "W0", "printed"))
SHARE_KEY = "sv2_patch_share"
NORM_KEY = "sv2_wp_norm"
PAIR_KEYS = ("mil", "dvs_sup")
NEXT_SEEDS = "write its own rule, then seeds 2025-2026 of SV4 and W0D, and v5_maps read on SV4"
NEXT_CLOSE = "the token line closes: no further arm, gate, LR or fusion variant on tokens"


def read_metrics(path: Path) -> list[dict[str, Any]]:
    """Every logged step of a run."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tail_mean(records: list[dict[str, Any]], key: str) -> float:
    """Mean of ``key`` over its last ``SV4_TAIL_STEPS`` logged values."""
    seen = [float(r[key]) for r in records if key in r]
    if not seen:
        raise ValueError(f"no {key} in this run's metrics.jsonl")
    return float(np.mean(seen[-constants.SV4_TAIL_STEPS :]))


def mechanics(sv4: list[dict[str, Any]], w0d: list[dict[str, Any]]) -> dict[str, Any]:
    """§11.2: is ``W_p`` used, the training fit against W0D, and pairing at step 1."""
    if not any(SHARE_KEY in r for r in sv4):
        raise ValueError(f"the SV4 run logs no {SHARE_KEY}: is this an SV4 run (stp.sv2_proj)?")
    share = tail_mean(sv4, SHARE_KEY)
    step1 = {k: {"sv4": float(sv4[0][k]), "w0d": float(w0d[0][k])} for k in PAIR_KEYS}
    return {
        "share_tail": share,
        "wp_norm_tail": tail_mean(sv4, NORM_KEY),
        "used": share >= constants.SV4_USED_SHARE,
        "mil_tail": {"sv4": tail_mean(sv4, "mil"), "w0d": tail_mean(w0d, "mil")},
        "step1": step1,
        "paired": all(
            abs(v["sv4"] - v["w0d"]) <= constants.SV4_PAIRED_TOL for v in step1.values()
        ),
    }


def verdict(used: bool, primary: str) -> str:
    """§11.3: ``W_p``'s use and SV4 - W0D together."""
    if not used:
        if primary == COST:
            return "cost with W_p unused (not through x^AS: read the mechanics)"
        return "W_p stays small: the host does not take x^AS under this objective"
    if primary == GAIN:
        return "used and helps"
    if primary == COST:
        return "used and costs"
    return "used without a shown effect"


def next_step(dada: str, dota: str) -> str:
    """§11.4: seeds only on a DADA gain that DoTA-CAP-dev does not contradict."""
    return NEXT_SEEDS if dada == GAIN and dota != COST else NEXT_CLOSE


def read_set(
    dirs: dict[str, Path], reader: Any, ci_fn: Any, mech: dict[str, Any]
) -> dict[str, Any]:
    head, items = reader(dirs)
    out = {**head, **paired_contrasts(items, ci_fn, PAIRS)}
    primary = out["SV4-W0D"]["sentence"]
    out["verdict"] = verdict(mech["used"], primary)
    out["overfit_tell"] = primary == COST and mech["mil_tail"]["sv4"] < mech["mil_tail"]["w0d"]
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    mech = mechanics(read_metrics(args.sv4_run / "metrics.jsonl"),
                     read_metrics(args.w0d_run / "metrics.jsonl"))
    dada = {"W0": args.w0_dada, "W0D": args.w0d_dada, "SV3": args.sv3_dada, "SV4": args.sv4_dada}
    dota = {"W0": args.w0_dota, "W0D": args.w0d_dota, "SV3": args.sv3_dota, "SV4": args.sv4_dota}
    out: dict[str, Any] = {"rule": RULE, "mechanics": mech,
                           "dada": read_set(dada, dada_items, video_ci, mech),
                           "dota": read_set(dota, dota_items, dota_ci, mech)}
    readout = _json(args.sv4_dada / "dada_readout.json")
    out["dada"]["middle_ruler"] = readout.get("middle_ruler_macro")
    out["next"] = next_step(out["dada"]["SV4-W0D"]["sentence"], out["dota"]["SV4-W0D"]["sentence"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{READOUT_NAME}.json", out)
    write_text_atomic(args.out_dir / f"{READOUT_NAME}.md", render_markdown(out))
    LOGGER.info("share %.4f; DADA: %s; next: %s", mech["share_tail"], out["dada"]["verdict"],
                out["next"])
    return out


def _set_lines(title: str, r: dict[str, Any], unit: str) -> list[str]:
    lines = [f"## {title}", "", "| Row | Macro |", "| --- | --- |",
             *[f"| {n} | {_num(r['macro'][n])} |" for n in ROWS], "",
             f"| Contrast (per {unit}) | What | Δ | Sentence |", "| --- | --- | --- | --- |"]
    for a, b, _ in PAIRS:
        c = r[f"{a}-{b}"]
        lines.append(f"| {a} - {b} | {c['what']} | {_ci(c['delta'])} | {c['sentence']} |")
    tell = "yes" if r["overfit_tell"] else "no"
    line = f"**Verdict (§11.3): {r['verdict']}, {PRELIMINARY}.** Overfit tell: {tell}."
    return [*lines, "", line, ""]


def render_markdown(r: dict[str, Any]) -> str:
    m = r["mechanics"]
    state = "used" if m["used"] else "unused"
    step1 = " · ".join(f"{k} {v['sv4']:.5f} vs {v['w0d']:.5f}" for k, v in m["step1"].items())
    lines = [
        "# SV4 read-out: SA²-v2 through W_p, Nexar-trained, seed 2024", "",
        f"Rule: `{RULE}`, written before any SV4 number. {PRELIMINARY}: adopts nothing.", "",
        f"W_p: share ‖W_p x^AS‖ / ‖c‖ over the last {constants.SV4_TAIL_STEPS} steps "
        f"**{m['share_tail']:.4f}** → **{state}** (bar {constants.SV4_USED_SHARE}); "
        f"‖W_p‖ {m['wp_norm_tail']:.4f}.", "",
        f"`mil`, last {constants.SV4_TAIL_STEPS} steps: SV4 {m['mil_tail']['sv4']:.4f}, "
        f"W0D {m['mil_tail']['w0d']:.4f}.", "",
        f"Paired with W0D at step 1 (|Δ| ≤ {constants.SV4_PAIRED_TOL}): "
        f"**{'yes' if m['paired'] else 'NO'}** ({step1}).", "",
        *_set_lines("DADA-2000 (370 sources, whole video, s8)", r["dada"], "video"),
        f"Printed: SV4's middle ruler (no pixels) {_num(r['dada']['middle_ruler'])}.", "",
        *_set_lines("DoTA-CAP-dev (protocol B)", r["dota"], "clip"),
        f"**Next (§11.4): {r['next']}.**",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    for row in ("w0", "w0d", "sv3", "sv4"):
        parser.add_argument(f"--{row}-dada", type=Path, required=True)
        parser.add_argument(f"--{row}-dota", type=Path, required=True)
    parser.add_argument("--sv4-run", type=Path, required=True, help="SV4's run (metrics.jsonl)")
    parser.add_argument("--w0d-run", type=Path, required=True, help="W0D's run (metrics.jsonl)")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
