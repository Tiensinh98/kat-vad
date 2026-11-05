"""SV3 read-out: does SA²-v2's x^AS help once the host may refuse it (gate g = 0 at init)?

Rule ``core/docs/final/SA2V2_RUN.md`` §10, written before any SV3 number. Rows W0D (host with
SV2's dropout = SV3 at step 0), SV2 (ungated), SV3 (gated); W0 printed. Primary per set:
``SV3 - W0D`` with §3's sentence; ``SV3 - SV2`` is what the gate changed. The gate's last
``tanh(g)`` from ``metrics.jsonl`` reads "opened" iff ``|tanh g| >= SV3_GATE_OPEN``; the verdict
sentence combines the two (§10.3), fixed before the numbers exist.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.sa2v2_readout import PRELIMINARY
from core.tools.stp_readout import COST, GAIN
from core.tools.stp_readout import ci as dota_ci
from core.tools.w0d_readout import dada_items, dota_items, paired_contrasts
from core.tools.zs_readout import video_ci

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "sv3_readout"
RULE = "core/docs/final/SA2V2_RUN.md §10"
ROWS = ("W0", "W0D", "SV2", "SV3")
PAIRS = (("SV3", "W0D", "gated module on the dropout host (primary)"),
         ("SV3", "SV2", "what the gate changed"),
         ("SV3", "W0", "printed"))
GATE_KEY = "sv2_gate"


def gate_trace(metrics: Path) -> dict[str, Any]:
    """First, max |.|, last ``tanh(g)`` logged in a run's ``metrics.jsonl``."""
    seen = []
    with metrics.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if GATE_KEY in record:
                    seen.append(float(record[GATE_KEY]))
    if not seen:
        raise ValueError(f"{metrics} logs no {GATE_KEY}: is this an SV3 run (stp.sv2_gate)?")
    last = seen[-1]
    return {"first": seen[0], "last": last, "max_abs": max(abs(v) for v in seen),
            "opened": abs(last) >= constants.SV3_GATE_OPEN}


def verdict(opened: bool, primary: str) -> str:
    """§10.3: the gate and SV3 - W0D together."""
    if not opened:
        if primary == COST:
            return "cost with a shut gate (not through x^AS: read the mechanics)"
        return "the module stays unused when the host may refuse it"
    if primary == GAIN:
        return "opened and helps"
    if primary == COST:
        return "opened and costs: training wants x^AS, transfer does not"
    return "opened without a shown effect"


def read_set(
    dirs: dict[str, Path], reader: Any, ci_fn: Any, gate: dict[str, Any]
) -> dict[str, Any]:
    head, items = reader(dirs)
    out = {**head, **paired_contrasts(items, ci_fn, PAIRS)}
    out["verdict"] = verdict(gate["opened"], out["SV3-W0D"]["sentence"])
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    gate = gate_trace(args.sv3_run / "metrics.jsonl")
    dada = {"W0": args.w0_dada, "W0D": args.w0d_dada, "SV2": args.sv2_dada, "SV3": args.sv3_dada}
    out: dict[str, Any] = {"rule": RULE, "gate": gate,
                           "dada": read_set(dada, dada_items, video_ci, gate)}
    dota = {"W0": args.w0_dota, "W0D": args.w0d_dota, "SV2": args.sv2_dota, "SV3": args.sv3_dota}
    have = all(d is not None and (d / "clip_aucs.json").is_file() for d in dota.values())
    out["dota"] = read_set(dota, dota_items, dota_ci, gate) if have else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{READOUT_NAME}.json", out)
    write_text_atomic(args.out_dir / f"{READOUT_NAME}.md", render_markdown(out))
    LOGGER.info("gate %+.4f; DADA: %s", gate["last"], out["dada"]["verdict"])
    return out


def _num(c: dict[str, float] | None) -> str:
    return "—" if not c else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _ci(c: dict[str, float] | None) -> str:
    return "—" if not c else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _set_lines(title: str, r: dict[str, Any], unit: str) -> list[str]:
    lines = [f"## {title}", "", "| Row | Macro |", "| --- | --- |",
             *[f"| {n} | {_num(r['macro'][n])} |" for n in ROWS], "",
             f"| Contrast (per {unit}) | What | Δ | Sentence |", "| --- | --- | --- | --- |"]
    for a, b, _ in PAIRS:
        c = r[f"{a}-{b}"]
        lines.append(f"| {a} - {b} | {c['what']} | {_ci(c['delta'])} | {c['sentence']} |")
    return [*lines, "", f"**Verdict (§10.3): {r['verdict']}, {PRELIMINARY}.**", ""]


def render_markdown(r: dict[str, Any]) -> str:
    g = r["gate"]
    state = "opened" if g["opened"] else "stayed shut"
    lines = [
        "# SV3 read-out: gated SA²-v2, Nexar-trained, seed 2024", "",
        f"Rule: `{RULE}`, written before any SV3 number. {PRELIMINARY}: adopts nothing.", "",
        f"Gate tanh(g): first {g['first']:+.4f} → last **{g['last']:+.4f}** (max |·| "
        f"{g['max_abs']:.4f}) → **{state}** (bar |tanh g| ≥ {constants.SV3_GATE_OPEN}).", "",
        *_set_lines("DADA-2000 (370 sources, whole video, s8)", r["dada"], "video"),
    ]
    if r["dota"] is None:
        lines.append("## DoTA-CAP-dev\n\npending.")
    else:
        lines += _set_lines("DoTA-CAP-dev (protocol B)", r["dota"], "clip")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    for row in ("w0", "w0d", "sv2", "sv3"):
        parser.add_argument(f"--{row}-dada", type=Path, required=True)
        parser.add_argument(f"--{row}-dota", type=Path, default=None)
    parser.add_argument("--sv3-run", type=Path, required=True, help="SV3's run (metrics.jsonl)")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
