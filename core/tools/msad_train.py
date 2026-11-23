"""K5f / K5dfi / L0f / L0dfi trained on MSAD-full, kernel 3 and 9 (``KIM_RUN.md`` §41).

Pure pieces (no model is loaded, no data is read except saved ``core.evaluate --save-scores``
runs): the arm recipes as ``--set`` lists, the step plan, the config-diff guard the runbook
applies after training, and the read-out table.

A *row* is ``<arm>M<kernel>`` (``K5fM3``, ``L0dfiM9``). The recipes are the Nexar rows' own
(``k5_fixed.ipynb``, ``k5_dvs_ignore.ipynb``, ``l0dfi_lagovad_nexar.ipynb``); only the corpus
(``MSAD-full``, whole videos), ``loss.mil_topk_pct`` (5 -> 16),
``model.score_head_kernel`` (3 and 9) and ``train.num_epochs`` (125) change (§41 R1 / R2).
Both kernels are rows and **neither is picked on the MSAD test set**: the report prints every
row and never a best-of.

Layout read by ``report``::

    <root>/eval/<row>/s<seed>/        core.evaluate on the MSAD test set (trained rows AND the
                                      Nexar-trained controls K5f / K5dfi / L0dfi, zero-shot)
    <root>/eval_tad/<row>/s<seed>/    optional: the same rows on TAD (raw scores)

CLI::

    python -m core.tools.msad_train plan --positives 120
    python -m core.tools.msad_train report --root outputs/v_final/kim_msad_train --seed 2024
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools import k5_msad_report as kmr
from core.tools import tadz_readout as td
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.stp_readout import paired
from core.tools.zs_lagovad_readout import versus_sentence

LOGGER = logging.getLogger(__name__)

Curves = dict[str, np.ndarray]

# Nexar-trained controls, read zero-shot on the MSAD test set (their own checkpoints, §41 R4)
CONTROLS = ("K5f", "K5dfi", "L0dfi")
TAD_SET = "tad"
LABEL = "PRELIMINARY (n = 1)"

# --- the recipes: exactly the Nexar rows' own switches (config diffs in KIM_RUN.md §41 R1) ---
_KIM_ON = ("kim.enabled=true", "kim.prototypes=true", "kim.edge_mask=true")
_DVS_OFF = ("dvs.theta=1.0", "dvs.theta_ego=1.0", "loss.dvs_anchor_mode=span")
_DVS_IGNORE = ("dvs.theta=0.7", "dvs.theta_ego=0.85", "loss.dvs_anchor_mode=ignore")
ARM_SETS: dict[str, tuple[str, ...]] = {
    "K5f": (*_KIM_ON, *_DVS_OFF),
    "K5dfi": (*_KIM_ON, "kim.seam_mask=true", *_DVS_IGNORE),
    "L0f": ("kim.enabled=false", *_DVS_OFF),
    "L0dfi": ("kim.enabled=false", *_DVS_IGNORE),
}
MOTION_ARMS = ("K5f", "K5dfi")  # V2-S motion stream + KIM; the L0 arms read plain CLIP
# config keys a row may differ from its Nexar parent on (everything else must be equal)
CORPUS_KEYS = frozenset({"data.dataset", "train.num_epochs", "loss.mil_topk_pct",
                         "model.score_head_kernel"})
PARENT = {"K5f": "K5f", "K5dfi": "K5dfi", "L0f": "K5f", "L0dfi": "L0dfi"}
# L0f has no Nexar run: its parent is K5f with KIM and motion switched off
L0F_EXTRA_KEYS = frozenset({"kim.enabled", "kim.prototypes", "kim.edge_mask", "model.motion_dim",
                            "v2.motion"})


def row_name(arm: str, kernel: int) -> str:
    if arm not in ARM_SETS:
        raise ValueError(f"unknown arm {arm!r}; one of {tuple(ARM_SETS)}")
    return f"{arm}{constants.KIM_MSAD_TRAIN_ROW_INFIX}{kernel}"


def parse_row(row: str) -> tuple[str, int]:
    """``K5dfiM9`` -> ``("K5dfi", 9)``."""
    arm, sep, kernel = row.rpartition(constants.KIM_MSAD_TRAIN_ROW_INFIX)
    if not sep or arm not in ARM_SETS or not kernel.isdigit():
        raise ValueError(f"{row!r} is not a §41 row "
                         f"(<arm>{constants.KIM_MSAD_TRAIN_ROW_INFIX}<kernel>)")
    return arm, int(kernel)


def all_rows(kernels: tuple[int, ...] = constants.KIM_MSAD_TRAIN_KERNELS) -> list[str]:
    return [row_name(a, k) for k in kernels for a in constants.KIM_MSAD_TRAIN_ARMS]


def epoch_plan(positives: int, epochs: int = constants.KIM_MSAD_TRAIN_EPOCHS,
               batch: int = constants.BATCH_SIZE) -> dict[str, int]:
    """Balanced epoch = 2 x positives; ``steps = ceil(2 x positives / batch) x epochs``."""
    per_epoch = math.ceil(constants.KIM_MSAD_TRAIN_BALANCED_FACTOR * positives / batch)
    return {"positives": positives, "epochs": epochs, "steps_per_epoch": per_epoch,
            "steps": per_epoch * epochs}


def train_sets(arm: str, kernel: int, seed: int = constants.KIM_SEED,
               epochs: int = constants.KIM_MSAD_TRAIN_EPOCHS) -> list[str]:
    """Every ``--set`` value of ``core.train`` for one row (stage 2, AMP, KIP off, MSAD-full)."""
    if arm not in ARM_SETS:
        raise ValueError(f"unknown arm {arm!r}; one of {tuple(ARM_SETS)}")
    motion = (f"v2.motion={constants.V5_ENCODER}", f"model.motion_dim={constants.V5_GRID_DIM}") \
        if arm in MOTION_ARMS else (f"v2.motion={constants.V2_OFF}", "model.motion_dim=0")
    return [
        "train.stage=2", f"train.seed={seed}", f"train.num_epochs={epochs}", "train.amp=true",
        f"data.dataset={constants.MSAD_FULL_DATASET}",
        f"loss.mil_topk_pct={constants.KIM_MSAD_TRAIN_MIL_TOPK_PCT}",
        "kip.enabled=false", "data.is_egocentric=false",
        f"model.score_head_kernel={kernel}", *ARM_SETS[arm], *motion,
    ]


def config_diff(base: dict[str, Any], got: dict[str, Any],
                defaults: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Keys that differ; a key added after ``base`` trained may sit at its default."""
    out: dict[str, tuple[Any, Any]] = {}
    for k in sorted(set(base) | set(got)):
        if base.get(k) == got.get(k):
            continue
        if k not in base and got.get(k) == defaults.get(k):
            continue
        if k not in got and base.get(k) == defaults.get(k):
            continue
        out[k] = (base.get(k), got.get(k))
    return out


def check_config(row: str, parent: dict[str, Any], got: dict[str, Any],
                 defaults: dict[str, Any], seed: int = constants.KIM_SEED) -> dict[str, Any]:
    """Raise unless ``got`` (the run's flattened config) is its Nexar parent's, changed only in
    the §41 keys and with the §41 values. ``parent`` is K5f's / K5dfi's / L0dfi's config; for
    L0f it is K5f's and KIM + motion are expected to differ."""
    arm, kernel = parse_row(row)
    allowed = CORPUS_KEYS | (L0F_EXTRA_KEYS if arm == "L0f" else frozenset())
    diff = config_diff(parent, got, defaults)
    stray = sorted(set(diff) - allowed)
    if stray:
        raise ValueError(f"{row}: config differs from its Nexar parent on {stray}: "
                         f"{ {k: diff[k] for k in stray} }")
    want = {"data.dataset": constants.MSAD_FULL_DATASET,
            "train.num_epochs": constants.KIM_MSAD_TRAIN_EPOCHS,
            "loss.mil_topk_pct": constants.KIM_MSAD_TRAIN_MIL_TOPK_PCT,
            "model.score_head_kernel": kernel, "train.seed": seed}
    wrong = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    if wrong:
        raise ValueError(f"{row}: config values (got, want) {wrong}")
    return diff


# --- the read-out ---------------------------------------------------------------------------------
def contrast_pairs(rows: set[str]) -> list[tuple[str, str, str]]:
    """(label, a, b) for a - b, only where both rows were read. §41 R5; none of them decides."""
    pairs: list[tuple[str, str, str]] = []
    for k in constants.KIM_MSAD_TRAIN_KERNELS:
        f, d = row_name("K5f", k), row_name("K5dfi", k)
        lf, ld = row_name("L0f", k), row_name("L0dfi", k)
        pairs += [(f"motion + KIM, DVS off (k{k})", f, lf),
                  (f"motion + KIM, DVS ignore (k{k})", d, ld),
                  (f"DVS ignore on KIM (k{k})", d, f),
                  (f"DVS ignore on plain CLIP (k{k})", ld, lf)]
        pairs += [(f"in-domain vs zero-shot (k{k})", row_name(a, k), c)
                  for a, c in (("K5f", "K5f"), ("K5dfi", "K5dfi"), ("L0dfi", "L0dfi"))]
    for arm in constants.KIM_MSAD_TRAIN_ARMS:
        lo, hi = (row_name(arm, k) for k in constants.KIM_MSAD_TRAIN_KERNELS)
        pairs.append((f"kernel 9 - 3, {arm} (printed, never a selection)", hi, lo))
    return [p for p in pairs if p[1] in rows and p[2] in rows]


def load_set(root: Path, folder: str, seed: int) -> dict[str, tuple[Curves, Curves]]:
    base = root / folder
    runs = {d.name: td.load_run(d / f"s{seed}") for d in sorted(base.iterdir())
            if (d / f"s{seed}" / "results.json").is_file()} if base.is_dir() else {}
    return runs


def read_set(runs: dict[str, tuple[Curves, Curves]], set_name: str) -> dict[str, Any]:
    """Per-row reads + the contrasts of one set (``msad_full`` / ``msad_traffic`` / ``tad``)."""
    labels_all = td.check_labels(runs, None)
    ids = labels_all if set_name == TAD_SET else kmr.set_ids(list(labels_all), set_name)
    labels = {v: labels_all[v] for v in sorted(ids)}
    reads = {r: kmr.row_read({v: runs[r][0][v] for v in labels}, labels) for r in runs}
    rulers = td.rulers(labels, None, kmr.SEED)
    items = {r: reads[r]["items"] for r in reads}
    contrasts = []
    for label, a, b in contrast_pairs(set(runs)):
        delta = td.bootstrap(paired(items[a], items[b]), kmr.SEED)
        contrasts.append({"label": label, "a": a, "b": b, "delta": delta,
                          "sentence": versus_sentence(delta)})
    base_rate = float(np.concatenate([labels[v] for v in labels]).mean())
    return {
        "videos": len(labels), "two_class": len(next(iter(items.values()))),
        "base_rate": base_rate,
        "rows": {r: {k: v for k, v in reads[r].items() if k != "items"} for r in reads},
        "rulers": {k: rulers[k] for k in ("clip_oracle_micro", "length_ruler_micro",
                                           "monotone_ruler_macro")},
        "contrasts": contrasts,
    }


def _ci(c: dict[str, float] | None, signed: bool = False) -> str:
    if c is None:
        return "n/a"
    f = "{:+.4f}" if signed else "{:.4f}"
    return f"{f.format(c['mean'])} [{f.format(c['low'])}, {f.format(c['high'])}]"


def render(report: dict[str, Any]) -> str:
    lines = [
        "# K5f / K5dfi / L0f / L0dfi trained on MSAD-full — KIM_RUN.md §41", "",
        f"**{LABEL}.** Seed {report['seed']}; one seed per row. Both kernels (3, 9) are printed; "
        "nothing here picks a kernel or a row (§41 R2 / R6). Controls K5f / K5dfi / L0dfi are the "
        "Nexar-trained rows read zero-shot.", "",
    ]
    for name, s in report["sets"].items():
        lines += [f"## {name} ({s['videos']} videos, macro over {s['two_class']}, base rate "
                  f"{s['base_rate']:.4f})", "",
                  "| row | AUC macro | micro | micro, equal video | AP | clip AUC |",
                  "|---|---|---|---|---|---|"]
        for r, v in s["rows"].items():
            lines.append(f"| {r} | {_ci(v['macro'])} | {v['micro']:.4f} | "
                         f"{v['micro_equal_video']:.4f} | {v['ap']:.4f} | {v['clip_auc']:.4f} |")
        ru = s["rulers"]
        lines += ["", f"Rulers (no pixel): clip oracle micro {ru['clip_oracle_micro']:.4f}, length "
                  f"ruler micro {ru['length_ruler_micro']:.4f}, `t/N` macro "
                  f"{_ci(ru['monotone_ruler_macro'])}.", "",
                  "| contrast (a - b) | a | b | delta macro [95 %] | |", "|---|---|---|---|---|"]
        lines += [f"| {c['label']} | {c['a']} | {c['b']} | {_ci(c['delta'], True)} | "
                  f"{c['sentence']} |" for c in s["contrasts"]]
        lines.append("")
    lines += [
        "## How to read this", "",
        "* Micro on MSAD measures clip classification (oracle in the rulers); read macro first.",
        "* Never write \"the better kernel\": it would be chosen on the set it is read on.",
        "* \"In-domain vs zero-shot\" mixes the corpus with the recipe's MSAD geometry "
        "(top-k 16, 125 epochs); it is not a pure data effect.",
        "* TAD micro is clip classification (oracle 0.9226); TAD macro is the localization read.",
    ]
    return "\n".join(lines) + "\n"


def run_report(root: Path, seed: int, out_dir: Path | None) -> dict[str, Any]:
    msad = load_set(root, "eval", seed)
    if not msad:
        raise FileNotFoundError(f"no <row>/s{seed}/results.json under {root / 'eval'}")
    sets = {kmr.SET_NAME[s]: read_set(msad, s) for s in kmr.SET_ORDER}
    tad = load_set(root, "eval_tad", seed)
    if tad:
        sets["TAD"] = read_set(tad, TAD_SET)
    report = {"seed": seed, "rows": sorted(msad), "sets": sets}
    out = out_dir or root / "REPORTS"
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / f"{constants.KIM_MSAD_TRAIN_REPORT_STEM}.json", report)
    write_text_atomic(out / f"{constants.KIM_MSAD_TRAIN_REPORT_STEM}.md", render(report))
    LOGGER.info("wrote %s", out / f"{constants.KIM_MSAD_TRAIN_REPORT_STEM}.md")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="print the step plan and every row's --set list")
    p.add_argument("--positives", type=int, default=constants.MSAD_TRAIN_ABNORMAL)
    p.add_argument("--seed", type=int, default=constants.KIM_SEED)
    r = sub.add_parser("report", help="the §41 table from saved core.evaluate runs")
    r.add_argument("--root", type=Path, required=True, help="OUTPUT_ROOT/kim_msad_train")
    r.add_argument("--seed", type=int, default=constants.KIM_SEED)
    r.add_argument("--out-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "plan":
        plan = epoch_plan(args.positives)
        rows = {r: train_sets(*parse_row(r), seed=args.seed) for r in all_rows()}
        LOGGER.info("plan %s", json.dumps(plan))
        LOGGER.info("rows %s", json.dumps(rows, indent=1))
        return
    run_report(args.root, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
