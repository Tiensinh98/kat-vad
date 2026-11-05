"""W0D read-out: does SV2's cost come from motion dropout or from the spatial module?

Rule ``core/docs/final/SA2V2_RUN.md`` §8, written before any W0D number. Three rows, one seed:
W0 (host), W0D (host + SV2's motion dropout), SV2 (W0D + SA²-v2). On each set, two paired
contrasts with the sentence of §3: ``W0D - W0`` (dropout alone) and ``SV2 - W0D`` (spatial module
on a dropout host). The attribution sentence is fixed by §8.2 before the numbers exist.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_transfer_readout import read_dada
from core.tools.sa2v2_readout import PRELIMINARY, _json
from core.tools.stp_readout import COST, paired, sentence
from core.tools.stp_readout import ci as dota_ci
from core.tools.zs_readout import video_ci

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "w0d_readout"
RULE = "core/docs/final/SA2V2_RUN.md §8"
ROWS = ("W0", "W0D", "SV2")
CONTRASTS = (("W0D", "W0", "dropout alone"), ("SV2", "W0D", "spatial module on W0D"),
             ("SV2", "W0", "both (printed, = §7)"))


def attribution(drop: str, spatial: str) -> str:
    """§8.2: which change carries the cost, from the two sentences."""
    if drop == COST and spatial == COST:
        return "both carry a cost"
    if drop == COST:
        return "the cost is motion dropout's; the spatial module shows none on top"
    if spatial == COST:
        return "the cost is the spatial module's; dropout alone shows none"
    return "neither change alone shows a cost"


def contrasts(items: dict[str, dict[str, float]], ci_fn: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a, b, what in CONTRASTS:
        delta = ci_fn(paired(items[a], items[b]))
        out[f"{a}-{b}"] = {"what": what, "delta": delta, "sentence": sentence(delta)}
    out["attribution"] = attribution(out["W0D-W0"]["sentence"], out["SV2-W0D"]["sentence"])
    return out


def read_dada_set(dirs: dict[str, Path]) -> dict[str, Any]:
    reads = {name: read_dada(d) for name, d in dirs.items()}
    return {"macro": {n: r["macro"] for n, r in reads.items()},
            "micro": {n: r["micro"] for n, r in reads.items()},
            **contrasts({n: r["items"] for n, r in reads.items()}, video_ci)}


def read_dota_set(dirs: dict[str, Path]) -> dict[str, Any]:
    macro = {n: _json(d / "protocol_b_readout.json").get("macro") for n, d in dirs.items()}
    items = {n: _json(d / "clip_aucs.json") for n, d in dirs.items()}
    return {"macro": macro, **contrasts(items, dota_ci)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"rule": RULE, "dada": read_dada_set(
        {"W0": args.w0_dada, "W0D": args.w0d_dada, "SV2": args.sv2_dada})}
    dota = {"W0": args.w0_dota, "W0D": args.w0d_dota, "SV2": args.sv2_dota}
    have = all(d is not None and (d / "clip_aucs.json").is_file() for d in dota.values())
    out["dota"] = read_dota_set(dota) if have else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{READOUT_NAME}.json", out)
    write_text_atomic(args.out_dir / f"{READOUT_NAME}.md", render_markdown(out))
    LOGGER.info("DADA: %s", out["dada"]["attribution"])
    return out


def _num(c: dict[str, float] | None) -> str:
    return "—" if not c else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _ci(c: dict[str, float] | None) -> str:
    return "—" if not c else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _set_lines(title: str, r: dict[str, Any], unit: str) -> list[str]:
    lines = [f"## {title}", "", "| Row | Macro |", "| --- | --- |",
             *[f"| {n} | {_num(r['macro'][n])} |" for n in ROWS], "",
             f"| Contrast (per {unit}) | What | Δ | Sentence |", "| --- | --- | --- | --- |"]
    for a, b, _ in CONTRASTS:
        c = r[f"{a}-{b}"]
        lines.append(f"| {a} - {b} | {c['what']} | {_ci(c['delta'])} | {c['sentence']} |")
    return [*lines, "", f"**Attribution (§8.2): {r['attribution']}, {PRELIMINARY}.**", ""]


def render_markdown(r: dict[str, Any]) -> str:
    lines = ["# W0D read-out: motion dropout vs spatial module, Nexar-trained, seed 2024", "",
             f"Rule: `{RULE}`, written before any W0D number. {PRELIMINARY}: adopts nothing.", "",
             *_set_lines("DADA-2000 (370 sources, whole video, s8)", r["dada"], "video")]
    if r["dota"] is None:
        lines.append("## DoTA-CAP-dev\n\npending.")
    else:
        lines += _set_lines("DoTA-CAP-dev (protocol B)", r["dota"], "clip")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    for row in ("w0", "w0d", "sv2"):
        parser.add_argument(f"--{row}-dada", type=Path, required=True)
        parser.add_argument(f"--{row}-dota", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
