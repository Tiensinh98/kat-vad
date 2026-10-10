"""Window trigger (addendum §21, D-N10): does the pilot ``whole`` read call for the ``window`` arm?

Reads three ``nexar_eval`` read-outs on ``nexar_val`` -- the pilot ``whole``/A3, the pilot
``whole``/A0, and E3's T2-trained A3 checkpoints scored zero-shot -- and fires if any of:

* **W1 position** -- crop macro at the centre placement exceeds the crop macro at either edge
  placement by more than ``NEXAR_TRIGGER_MAX_EDGE_DROP`` (the model learned "mid-video").
* **W2 no gain** -- the ``whole``/A3 crop endpoint is not above the zero-shot A3 endpoint
  (training on Nexar bought nothing over not training on it).
* **W3 clip classifier** -- ``whole``/A3 clip-level AUC >= ``NEXAR_TRIGGER_CLIP_AUC`` **and** its
  crop endpoint is below ``whole``/A0's (C14: it learned which video, not which frame).

Point estimates (seed-averaged means) are compared; the zero-shot edge drop is printed beside W1,
never gated. The verdict decides only whether ``window`` is trained; it is not a result.

CLI::

    python -m core.tools.nexar_trigger --whole-a3 REPORTS/nexar_val/pilot_whole_A3 \\
        --whole-a0 REPORTS/nexar_val/pilot_whole_A0 --zero-shot REPORTS/nexar_val/zeroshot_e3_A3 \\
        --out-dir REPORTS/nexar_val/trigger_pilot
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_eval import READOUT_JSON

LOGGER = logging.getLogger(__name__)

TRIGGER_JSON = "window_trigger.json"
TRIGGER_MD = "window_trigger.md"


def crop_mean(readout: dict[str, Any], placement: float | None = None) -> float:
    """Seed-averaged crop macro at one placement, or the endpoint (mean over placements)."""
    key = "mean_over_placements" if placement is None else str(placement)
    return float(readout["crops"][key]["macro"]["mean"])


def edge_drop(readout: dict[str, Any]) -> float:
    """Largest (centre - edge) crop macro over the edge placements."""
    centre = crop_mean(readout, constants.NEXAR_TRIGGER_CENTRE)
    return max(centre - crop_mean(readout, p) for p in constants.NEXAR_TRIGGER_EDGES)


def window_trigger(
    whole_a3: dict[str, Any], whole_a0: dict[str, Any], zero_shot: dict[str, Any]
) -> dict[str, Any]:
    """W1-W3 on the three read-outs; ``build_window`` iff any fires."""
    end_a3, end_a0, end_zs = crop_mean(whole_a3), crop_mean(whole_a0), crop_mean(zero_shot)
    drop = edge_drop(whole_a3)
    clip_auc = whole_a3["whole"]["clip_auc_max"]
    checks = {
        "W1_position": {
            "fires": drop > constants.NEXAR_TRIGGER_MAX_EDGE_DROP,
            "edge_drop": drop,
            "threshold": constants.NEXAR_TRIGGER_MAX_EDGE_DROP,
            "zero_shot_edge_drop": edge_drop(zero_shot),
        },
        "W2_no_gain": {
            "fires": end_a3 <= end_zs,
            "whole_A3_endpoint": end_a3,
            "zero_shot_A3_endpoint": end_zs,
        },
        "W3_clip_classifier": {
            "fires": clip_auc is not None and clip_auc >= constants.NEXAR_TRIGGER_CLIP_AUC
            and end_a3 < end_a0,
            "clip_auc_max": clip_auc,
            "threshold": constants.NEXAR_TRIGGER_CLIP_AUC,
            "whole_A3_endpoint": end_a3,
            "whole_A0_endpoint": end_a0,
        },
    }
    return {
        "checks": checks,
        "fired": [name for name, c in checks.items() if c["fires"]],
        "build_window": any(c["fires"] for c in checks.values()),
    }


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def render_markdown(result: dict[str, Any], sources: dict[str, str]) -> str:
    c = result["checks"]
    w1, w2, w3 = c["W1_position"], c["W2_no_gain"], c["W3_clip_classifier"]
    verdict = ("**BUILD `window`** — fired: " + ", ".join(result["fired"])
               if result["build_window"] else "**`whole` passes** — no trigger fired")
    return "\n".join([
        "# Nexar window trigger (§21 D-N10)",
        "",
        verdict,
        "",
        "| check | fires | read | bar |",
        "|---|---|---|---|",
        f"| W1 position | {w1['fires']} | centre - edge crop macro {_num(w1['edge_drop'])} "
        f"(zero-shot, printed: {_num(w1['zero_shot_edge_drop'])}) | > {w1['threshold']} |",
        f"| W2 no gain | {w2['fires']} | whole/A3 {_num(w2['whole_A3_endpoint'])} vs zero-shot A3 "
        f"{_num(w2['zero_shot_A3_endpoint'])} | ≤ |",
        f"| W3 clip classifier | {w3['fires']} | clip AUC {_num(w3['clip_auc_max'])}; whole/A3 "
        f"{_num(w3['whole_A3_endpoint'])} vs whole/A0 {_num(w3['whole_A0_endpoint'])} | "
        f"≥ {w3['threshold']} and A3 < A0 |",
        "",
        "Sources: " + ", ".join(f"{k} `{v}`" for k, v in sources.items()),
        "",
    ])


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = {"whole_A3": args.whole_a3, "whole_A0": args.whole_a0, "zero_shot_A3": args.zero_shot}
    readouts = {
        k: json.loads((d / READOUT_JSON).read_text(encoding="utf-8")) for k, d in dirs.items()
    }
    for name, r in readouts.items():
        if r["split"] != constants.V2_SPLIT_NEXAR_VAL:
            raise SystemExit(f"{name} was read on {r['split']}; the trigger reads nexar_val only")
    result = window_trigger(readouts["whole_A3"], readouts["whole_A0"], readouts["zero_shot_A3"])
    sources = {k: str(d) for k, d in dirs.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / TRIGGER_JSON, {**result, "sources": sources})
    write_text_atomic(args.out_dir / TRIGGER_MD, render_markdown(result, sources))
    LOGGER.info("window trigger: build_window=%s fired=%s", result["build_window"], result["fired"])
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--whole-a3", type=Path, required=True, help="nexar_eval out dir")
    parser.add_argument("--whole-a0", type=Path, required=True, help="nexar_eval out dir")
    parser.add_argument("--zero-shot", type=Path, required=True,
                        help="nexar_eval out dir of E3's T2-trained A3 checkpoints")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
