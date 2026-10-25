"""Amendment 18 read-out (addendum §29): Nexar-trained arms in-domain and zero-shot, one table.

Pure aggregation -- no model is loaded. Every input is an ordinary protocol output, laid out by
``colab/v2/nexar_zeroshot.ipynb`` under ``<root>/eval/``::

    nexar_test/<nexar|t2>/<arm>/{nexar_readout.json, item_aucs.json}   nexar_eval --final
    tad/<arm>/s<seed>/{results.json, scores/<video>.npz}               core.evaluate (§26)
    dada/<arm>/{dada_readout.json, clip_aucs.json, clip_scores.npz}    dada_whole_eval
    dota/<arm>/<dota_cap_dev|dota_cap_eval>/{protocol_b_readout.json, clip_aucs.json,
                                             clip_scores.npz}          protocol_b_eval

The one sentence (D-N18), mechanically: on ``nexar_test``, the per-seed crop endpoint of
Nexar-trained A3-ign minus A0, paired t95 over seeds 2024-2028 -- "A3-ign helps on Nexar" iff the
interval lies above 0, "A3-ign hurts on Nexar" iff below 0, otherwise "no difference shown at
n = 5". Everything else (the other arms, TAD / DADA / DoTA, the T2-trained rows) is printed and
decides nothing (D-N19). A set that was not run prints "not run"; only the primary is required.

CLI::

    python -m core.tools.nexar_transfer_readout --root outputs/v2_nexar_window
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
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools import dada_whole_eval, nexar_eval, protocol_b_eval, tadz_readout
from core.tools.e3_readout import REPORTS, clip_delta, paired_interval, seed_name
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "transfer_readout.json"
READOUT_MD = "transfer_readout.md"
EVAL_DIR = "eval"
NEXAR_TEST, TAD, DADA, DOTA = "nexar_test", "tad", "dada", "dota"
SETS = (NEXAR_TEST, TAD, DADA, DOTA)
TRAINED_ON = {"nexar": "Nexar (window)", "t2": "T2 (E3 / E4-0)"}
DOTA_SPLITS = (constants.V2_SPLIT_DOTA_CAP_DEV, constants.V2_SPLIT_DOTA_CAP_EVAL)
SET_TITLE = {
    NEXAR_TEST: "Nexar test (in-domain; endpoint = shifted-crop mean, D-N2)",
    TAD: "TAD test (100 videos, s8, §26's protocol)",
    DADA: "DADA-2000 whole videos (sources of T2-test, s8)",
    DOTA: "DoTA whole clips (DoTA-CAP dev + eval, protocol B)",
}

ArmRead = dict[str, Any]


# ---------------------------------------------------------------------------
# one adapter per protocol -> {per_seed, items, groups, macro, micro, clip_auc, extra}
# ---------------------------------------------------------------------------
def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_groups(items: dict[str, float]) -> dict[str, str]:
    return {v: v for v in items}


def read_nexar(arm_dir: Path) -> ArmRead:
    r = _json(arm_dir / nexar_eval.READOUT_JSON)
    if r["split"] != constants.V2_SPLIT_NEXAR_TEST:
        raise ValueError(f"{arm_dir}: read on {r['split']}, not {constants.V2_SPLIT_NEXAR_TEST}")
    items = nexar_eval.crop_video_means(_json(arm_dir / nexar_eval.ITEM_AUCS_JSON))
    w = r["whole"]
    return {
        "per_seed": r["per_run_crop_endpoint"],
        "items": items,
        "groups": _self_groups(items),
        "macro": r["crops"]["mean_over_placements"]["macro"],
        "micro": w["micro"],
        "clip_auc": w["clip_auc_max"],
        "extra": {
            "whole_macro": w["macro"],
            "crop_macro": {str(p): r["crops"][str(p)]["macro"]
                           for p in constants.NEXAR_CROP_PLACEMENTS},
            "middle_ruler": {str(p): r["crops"][str(p)]["middle_ruler_macro"]
                             for p in constants.NEXAR_CROP_PLACEMENTS},
            "clip_oracle_micro": w["clip_oracle_micro"],
        },
    }


def read_tad(arm_dir: Path, seeds: list[int], seed: int) -> ArmRead:
    runs = {seed_name(s): tadz_readout.load_run(arm_dir / seed_name(s)) for s in seeds}
    labels = tadz_readout.check_labels(runs)
    reads = tadz_readout.arm_reads({n: c for n, (c, _) in runs.items()}, labels, seed)
    rulers = tadz_readout.rulers(labels, None, seed)
    return {
        "per_seed": reads["per_seed_macro"],
        "items": reads["clip_aucs"],
        "groups": _self_groups(reads["clip_aucs"]),
        "macro": reads["macro"],
        "micro": reads["micro"],
        "clip_auc": reads["video_auc"],
        "extra": {"clip_oracle_micro": rulers["clip_oracle_micro"],
                  "length_ruler_micro": rulers["length_ruler_micro"],
                  "middle_ruler": rulers["middle_ruler_macro"]},
    }


def read_dada(arm_dir: Path) -> ArmRead:
    r = _json(arm_dir / dada_whole_eval.READOUT_JSON)
    items = _json(arm_dir / dada_whole_eval.CLIP_AUCS_JSON)
    return {
        "per_seed": r["per_run_macro"],
        "items": items,
        "groups": _self_groups(items),
        "macro": r["macro"],
        "micro": r["micro"],
        "clip_auc": r["video_auc"],
        "extra": {"monotone_ruler": r["monotone_ruler_macro"],
                  "middle_ruler": r["middle_ruler_macro"]},
    }


def pooled_per_seed(readouts: list[dict[str, Any]]) -> dict[str, float]:
    """Per-run macro over the union of the splits' two-class clips (weighted by their counts)."""
    runs = set(readouts[0]["per_run_macro"])
    if any(set(r["per_run_macro"]) != runs for r in readouts):
        raise ValueError("the DoTA splits were scored with different runs")
    total = sum(r["two_class"] for r in readouts)
    return {n: sum(r["per_run_macro"][n] * r["two_class"] for r in readouts) / total
            for n in sorted(runs)}


def read_dota(arm_dir: Path, seed: int) -> ArmRead:
    readouts, items = [], {}
    scores: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split in DOTA_SPLITS:
        d = arm_dir / split
        readouts.append(_json(d / protocol_b_eval.READOUT_JSON))
        items.update(_json(d / protocol_b_eval.CLIP_AUCS_JSON))
        s, g = protocol_b_eval.read_clip_scores(d / protocol_b_eval.CLIP_SCORES_NPZ)
        scores.update(s)
        labels.update(g)
    groups = {v: dota_group(v) for v in items}
    ids = sorted(labels)
    return {
        "per_seed": pooled_per_seed(readouts),
        "items": items,
        "groups": groups,
        "macro": cluster_bootstrap_ci(items, groups, constants.V2_E1_BOOTSTRAP, seed,
                                      constants.V2_E1_CI),
        "micro": frame_auc(np.concatenate([scores[v] for v in ids]),
                           np.concatenate([labels[v] for v in ids])),
        "clip_auc": None,  # every DoTA-CAP clip contains its accident
        "extra": {"clips": len(ids), "per_split_macro": {
            r["split"]: r["macro"] for r in readouts}},
    }


def read_set(root: Path, name: str, source: str, arm: str, seeds: list[int],
             seed: int) -> ArmRead | None:
    """One arm on one set, or ``None`` when its output is not on disk."""
    base = root / EVAL_DIR / name
    arm_dir = base / source / arm if name == NEXAR_TEST else base / arm
    if not arm_dir.is_dir():
        return None
    if name == NEXAR_TEST:
        return read_nexar(arm_dir)
    if name == TAD:
        return read_tad(arm_dir, seeds, seed)
    if name == DADA:
        return read_dada(arm_dir)
    return read_dota(arm_dir, seed)


# ---------------------------------------------------------------------------
# contrasts + the sentence
# ---------------------------------------------------------------------------
def contrast(a: ArmRead, b: ArmRead, seed: int, paired_seeds: bool = True) -> dict[str, Any]:
    """Paired t95 over seeds (same-corpus arms only) and a cluster bootstrap over items."""
    delta = clip_delta(a["items"], b["items"])
    return {
        "paired_t95": paired_interval(a["per_seed"], b["per_seed"], constants.V2_E3_T95)
        if paired_seeds else None,
        "item_bootstrap": cluster_bootstrap_ci(delta, {v: a["groups"][v] for v in delta},
                                               constants.V2_E1_BOOTSTRAP, seed,
                                               constants.V2_E1_CI),
    }


def primary_sentence(interval: dict[str, Any]) -> str:
    if interval["low"] > 0.0:
        return "A3-ign helps on Nexar"
    if interval["high"] < 0.0:
        return "A3-ign hurts on Nexar"
    return "no difference shown at n = 5"


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = list(args.seeds)
    arms = list(constants.NEXAR_FULL_ARMS)
    reads: dict[str, dict[str, dict[str, ArmRead | None]]] = {
        name: {src: {a: read_set(args.root, name, src, a, seeds, args.seed) for a in arms}
               for src in (("nexar", "t2") if name == NEXAR_TEST else ("nexar",))}
        for name in SETS
    }
    x, y = constants.NEXAR_FULL_PRIMARY
    primary_reads = reads[NEXAR_TEST]["nexar"]
    if primary_reads[x] is None or primary_reads[y] is None:
        raise FileNotFoundError(f"the primary needs nexar_test reads of Nexar-trained {x} and {y}")
    want = {seed_name(s) for s in seeds}
    for a in (x, y):
        got = set(primary_reads[a]["per_seed"])  # type: ignore[index]
        if got != want:
            raise ValueError(f"primary arm {a} was read with runs {sorted(got)}, "
                             f"need {sorted(want)}")
    primary = contrast(primary_reads[x], primary_reads[y], args.seed)  # type: ignore[arg-type]
    primary["sentence"] = primary_sentence(primary["paired_t95"])

    contrasts: dict[str, dict[str, Any]] = {}
    for name in SETS:
        nexar = reads[name]["nexar"]
        rows = {}
        for a, b in ((x, y), *constants.NEXAR_FULL_CONTRASTS):
            if nexar[a] is not None and nexar[b] is not None:
                rows[f"{a} - {b}"] = contrast(nexar[a], nexar[b], args.seed)  # type: ignore[arg-type]
        contrasts[name] = rows
    corpus = {}
    for a in arms:  # same arm, Nexar-trained - T2-trained (different runs: item bootstrap only)
        n, t = reads[NEXAR_TEST]["nexar"][a], reads[NEXAR_TEST]["t2"][a]
        if n is not None and t is not None:
            corpus[a] = contrast(n, t, args.seed, paired_seeds=False)

    readout = {
        "addendum": "core/docs/v2/PREREG_ADDENDUM.md §29 (Amendment 18, D-N16-D-N22)",
        "seeds": seeds,
        "primary": {"contrast": f"{x} - {y}", "set": constants.V2_SPLIT_NEXAR_TEST, **primary},
        "reads": {name: {src: {a: None if r is None else {k: v for k, v in r.items()
                                                           if k not in ("items", "groups")}
                               for a, r in by_arm.items()}
                         for src, by_arm in by_src.items()}
                  for name, by_src in reads.items()},
        "contrasts": contrasts,
        "nexar_minus_t2_on_nexar_test": corpus,
    }
    out = args.root / REPORTS
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / READOUT_JSON, readout)
    write_text_atomic(out / READOUT_MD, render_markdown(readout))
    LOGGER.info("§29: %s -> %s", primary["sentence"], out)
    return readout


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _ci(c: dict[str, float] | None) -> str:
    return "—" if not c else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def _arm_rows(by_arm: dict[str, dict[str, Any] | None], label: str) -> list[str]:
    rows = []
    for a, r in by_arm.items():
        if r is None:
            rows.append(f"| {label} | {a} | not run | | | |")
            continue
        sd = float(np.std(list(r["per_seed"].values()), ddof=1)) if len(r["per_seed"]) > 1 else 0
        rows.append(f"| {label} | {a} | **{_ci(r['macro'])}** | {_num(r['micro'])} | "
                    f"{_num(r['clip_auc'])} | {sd:.4f} |")
    return rows


def _contrast_rows(rows: dict[str, dict[str, Any]]) -> list[str]:
    out = ["| contrast | paired t95 over seeds | item bootstrap |", "|---|---|---|"]
    for name, c in rows.items():
        out.append(f"| {name} | {_ci(c['paired_t95'])} | {_ci(c['item_bootstrap'])} |")
    return out


def render_markdown(r: dict[str, Any]) -> str:
    p = r["primary"]
    lines = [
        "# Nexar full read-out — Amendment 18 (§29)",
        "",
        f"Seeds {r['seeds']}. Nexar arms trained on `window` only (A0-A3 `span`, A3-ign `ignore`).",
        "",
        f"## Primary (D-N18): {p['contrast']} on {p['set']}, per-seed crop endpoint",
        "",
        f"* paired t95: {_ci(p['paired_t95'])}",
        f"* video bootstrap (printed): {_ci(p['item_bootstrap'])}",
        f"* **Sentence: {p['sentence']}**",
        "",
        "Everything below is printed only (D-N19): it decides nothing and never moves `B`.",
        "",
    ]
    for name in SETS:
        lines += [f"## {SET_TITLE[name]}", "",
                  "| trained on | arm | macro [95 % CI] | micro | clip AUC | seed SD |",
                  "|---|---|---|---|---|---|"]
        for src, by_arm in r["reads"][name].items():
            lines += _arm_rows(by_arm, TRAINED_ON[src])
        lines += ["", *_contrast_rows(r["contrasts"][name]), ""]
        first = next((v for v in r["reads"][name]["nexar"].values() if v is not None), None)
        if first is not None:
            lines += [f"Rulers / extras (no pixels or set facts): "
                      f"{json.dumps(_round(first['extra']), ensure_ascii=False)}", ""]
    if r["nexar_minus_t2_on_nexar_test"]:
        lines += ["## Same arm, Nexar-trained - T2-trained, on Nexar test (video bootstrap)", "",
                  "| arm | Δ crop endpoint |", "|---|---|"]
        for a, c in r["nexar_minus_t2_on_nexar_test"].items():
            lines.append(f"| {a} | {_ci(c['item_bootstrap'])} |")
        lines.append("")
    return "\n".join(lines)


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--root", type=Path, required=True, help="outputs/v2_nexar_window")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(constants.NEXAR_RUN_SEEDS))
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap seed")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
