"""Zero-shot comparison table: LaGoVAD / W0 / R4 x macro / micro / AP on the four sets.

The read-out (``zs_lagovad_readout``) owns macro and micro; it never reported AP. This tool reads
macro and micro from its JSON, computes AP from the saved scores with **the same pooling as that
set's micro**, and refuses to write when a recomputed micro disagrees with the read-out (so an AP
cannot come from a different score file than the numbers beside it). A row whose score file is
absent gets a blank AP cell, never a substitute. Pure aggregation, no model is loaded::

    python -m core.tools.zs_compare_table --lagovad-root outputs/v_final/zs_lagovad \\
        --zs-root outputs/v_final/zs --dota W0 outputs/v_final/wu/eval/dota/dota_cap_dev/W0 \\
        --dota R4 outputs/v_final/stp/eval/dota/dota_cap_dev/R4 \\
        --out core/docs/final/ZS_COMPARISON.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import frame_ap, frame_auc, normalize_scores
from core.tools import protocol_b_eval, tadz_readout
from core.tools.kill_switch_probe import write_text_atomic

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "REPORTS/zs_lagovad_readout.json"
EVAL_DIR = "eval"
DOTA, DADA, TAD = "dota", "dada", "tad"
DAD = constants.DAD_OUTPUT_SET
SET_ORDER = (DOTA, DADA, TAD, DAD)
SET_NAME = {DOTA: "DoTA-CAP-dev", DADA: "DADA-2000", TAD: "TAD", DAD: "DAD"}
ROWS = (constants.ZS_LAGOVAD_ARM, constants.ZS_CONTROL_ARM, constants.ZS_SA2_ARM)
ROW_NAME = {
    constants.ZS_LAGOVAD_ARM: "LaGoVAD (public ckpt)",
    constants.ZS_CONTROL_ARM: "LaGoVAD + W0",
    constants.ZS_SA2_ARM: "LaGoVAD + R4",
}
MICRO_TOLERANCE = 1e-6
BLANK = "—"
NOTES = """
## Protocol per set

| Set | Items | Rows and pooling of micro / AP |
|---|---|---|
| DoTA-CAP-dev | 569 two-class clips | protocol B (`s1[::3]`, native frames); per-clip min-max |
| DADA-2000 | 370 two-class T2-test sources | stride 8, whole video; raw scores |
| TAD | 100 test videos, macro over the 60 two-class | stride 8; raw scores |
| DAD | 466 test videos, macro over the 165 positives | stride 7 at 25 fps, `toa = 90`; raw scores |

## How to read this table

* **PRELIMINARY (n = 1 seed, s2024).** Nothing here adopts anything.
* **Not a matched comparison.** The public checkpoint trained on PreVAD, W0 and R4 on Nexar, and
  its zero-shot status on these sets is unverified (`ZS_TAD_DADA.md` §7). Never write "beats
  LaGoVAD". Every LaGoVAD cell is measured here; no paper number is used, and none may sit beside
  these (the published numbers use other protocols: lesson C8).
* **The bracket is the 95 % cluster bootstrap of the macro.** R4 - W0 per set: DoTA-CAP-dev no
  difference, DADA-2000 no difference, TAD gain (fragile, `ZS_TAD_DADA.md` §6), DAD no difference.
* **DAD macro is not localization (`ZS_DAD.md` §8).** Every positive is abnormal on the same last
  two rows, so a no-pixel `t/N` ruler scores 1.000.
* **Micro on TAD and DAD measures clip classification.** A constant-score-per-clip oracle reaches
  0.9226 on TAD and 0.8390 on DAD.
* **Read AP against the base rate in its row**: the share of positive frames, which is what a
  random scorer gets.
"""

Curves = dict[str, np.ndarray]


def _sources(args: argparse.Namespace, set_name: str, row: str) -> Path:
    """The saved scores of one row: a ``clip_scores.npz`` or a ``core.evaluate`` run dir."""
    lagovad = row == constants.ZS_LAGOVAD_ARM
    run = constants.ZS_LAGOVAD_RUN if lagovad else f"s{args.seed}"
    eval_root: Path = args.lagovad_root / EVAL_DIR
    zs_eval: Path = args.zs_root / EVAL_DIR
    if set_name == DOTA:
        arms = dict(args.dota or [])
        base = eval_root / DOTA / row / "B" if lagovad else Path(str(arms[row]))
        return base / protocol_b_eval.CLIP_SCORES_NPZ
    if set_name == DADA:
        base = eval_root / DADA / row if lagovad else zs_eval / DADA / row
        return base / protocol_b_eval.CLIP_SCORES_NPZ
    if set_name == TAD:
        return eval_root / TAD / row / run if lagovad else zs_eval / TAD / row / run
    return eval_root / DAD / row / run


def load_scores(source: Path) -> tuple[Curves, Curves] | None:
    """``(scores, labels)`` or ``None`` when this row's score file is not on this disk."""
    try:
        if source.suffix == ".npz":
            if not source.is_file():
                raise FileNotFoundError(source)
            return protocol_b_eval.read_clip_scores(source)
        return tadz_readout.load_run(source)
    except FileNotFoundError:
        LOGGER.warning("no scores at %s: AP stays blank", source)
        return None


def pooled_reads(scores: Curves, labels: Curves, minmax: bool) -> dict[str, float]:
    """Pooled micro, AP and base rate under one set's pooling."""
    ids = sorted(labels)
    pooled = np.concatenate([
        normalize_scores(scores[v], constants.SCORE_NORM_MINMAX) if minmax else scores[v]
        for v in ids
    ])
    truth = np.concatenate([labels[v] for v in ids])
    return {"micro": frame_auc(pooled, truth), "ap": frame_ap(pooled, truth),
            "base_rate": float(np.mean(truth > 0))}


def build_table(args: argparse.Namespace, readout: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{set: {row: {macro, micro, ap}, "base_rate": float | None}}``, gated on the read-out."""
    table: dict[str, dict[str, Any]] = {}
    for set_name in SET_ORDER:
        reported = readout["sets"][set_name]["rows"]
        table[set_name] = {"base_rate": None}
        for row in ROWS:
            cell = {"macro": reported[row]["macro"], "micro": reported[row]["micro"], "ap": None}
            loaded = load_scores(_sources(args, set_name, row))
            if loaded is not None:
                reads = pooled_reads(*loaded, minmax=set_name == DOTA)
                if abs(reads["micro"] - cell["micro"]) > MICRO_TOLERANCE:
                    raise ValueError(
                        f"{set_name}/{row}: micro {reads['micro']:.6f} from the score file != "
                        f"{cell['micro']:.6f} in the read-out: not the same scores"
                    )
                cell["ap"] = reads["ap"]
                table[set_name]["base_rate"] = reads["base_rate"]
            table[set_name][row] = cell
    return table


def _macro(c: dict[str, float]) -> str:
    return f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _plain(value: float | None) -> str:
    return BLANK if value is None else f"{value:.4f}"


def render(table: dict[str, dict[str, Any]], seed: int) -> str:
    """The comparison as one markdown document."""
    header = " | ".join(ROW_NAME[r] for r in ROWS)
    lines = [
        "# Zero-shot comparison: LaGoVAD, LaGoVAD + W0, LaGoVAD + R4",
        "",
        f"**PRELIMINARY (n = 1 seed, s{seed}).** Generated by `core.tools.zs_compare_table` from "
        "`zs_lagovad_readout.json` and the saved scores. W0 = no SA², R4 = SA², both trained on "
        "Nexar. A `—` is a score file not available when this was generated.",
        "",
        f"| Dataset | Metric | {header} | Base rate |",
        "|---|---|---|---|---|---|",
    ]
    for set_name in SET_ORDER:
        rows = table[set_name]
        macro = " | ".join(_macro(rows[r]["macro"]) for r in ROWS)
        micro = " | ".join(_plain(rows[r]["micro"]) for r in ROWS)
        ap = " | ".join(_plain(rows[r]["ap"]) for r in ROWS)
        lines += [
            f"| **{SET_NAME[set_name]}** | AUC macro | {macro} | |",
            f"| | AUC micro | {micro} | |",
            f"| | AP | {ap} | {_plain(rows['base_rate'])} |",
        ]
    return "\n".join(lines) + "\n" + NOTES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--lagovad-root", type=Path, required=True,
                        help="outputs/zs_lagovad (REPORTS/ and eval/ under it)")
    parser.add_argument("--zs-root", type=Path, required=True,
                        help="outputs/zs (W0 / R4 TAD + DADA reads)")
    parser.add_argument("--dota", nargs=2, metavar=("ARM", "DIR"), action="append",
                        help="an arm's DoTA-CAP-dev protocol-B read dir (W0 and R4)")
    parser.add_argument("--seed", type=int, default=constants.ZS_SEED)
    parser.add_argument("--out", type=Path, required=True, help="the markdown file to write")
    return parser


def run(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    readout = json.loads((args.lagovad_root / READOUT_JSON).read_text(encoding="utf-8"))
    table = build_table(args, readout)
    write_text_atomic(args.out, render(table, args.seed))
    LOGGER.info("comparison table -> %s", args.out)
    return table


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
