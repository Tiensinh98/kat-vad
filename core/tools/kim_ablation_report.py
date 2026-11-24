"""Which arm benefits from KIM: KIM-on against KIM-off, micro AUC first, macro AUC second.

``KIM_RUN.md`` §53. Pure aggregation over saved reads (``kim_ablation.ipynb``), PRELIMINARY
(n = 1). Per arm and per set the contrast is on - off:

* **Micro** (primary): pooled AUC. DoTA-CAP-dev and DADA after per-clip min-max, Nexar val (whole
  video), TAD and DAD raw; cluster bootstrap with the same resamples for every row.
* **Macro** (secondary): the read-outs' endpoints and bootstrap (Nexar = per-video crop means).
* **Difference of differences** (KA7): (on - off) of an arm minus (on - off) of the first arm.

Guards (raise): last-epoch ``mul_mil`` above the §31 floor; on -> off config diff = exactly the
on row's true ``kim.*`` flags; one checkpoint per row, none shared; same items and labels; every
score file's micro equals its read-out's. The start gate (KA3) does not raise: where a re-run of
the on row exists under ``--root``, it replaces the saved on row; unequal step-1 ``mil`` turns the
arm into UNMATCHED START. CLI::

    python -m core.tools.kim_ablation_report --outputs-root outputs --out-dir .../REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core import constants
from core.config import Config
from core.metrics import cluster_bootstrap_ci, frame_auc, normalize_scores
from core.tools import k5_dvs_ignore_report as dvs
from core.tools import k5_edge_report as edge
from core.tools import k5_whole_report as kw
from core.tools import k5_zs_report as zs_reads
from core.tools import nexar_eval, zs_compare_table
from core.tools.k5_bottomk_report import read_env
from core.tools.k5_lneg_cap_report import check_fixed_runs
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.pre_onset_read import load_rows
from core.tools.stp_readout import paired
from core.tools.v5_stage1 import _flat

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/KIM_RUN.md §53"
NEXAR, TAD = kw.NEXAR, zs_reads.TAD
ALL_SETS = (NEXAR, *kw.ZS_SETS)
DECISION_SETS = dvs.DECISION_SETS
GAIN, COST = constants.KIM_DVS_IGNORE_GAIN, constants.KIM_DVS_IGNORE_COST
BENEFITS, HURT, MIXED = "BENEFITS", "HURT", "MIXED"
NO_DIFFERENCE = constants.KIM_DVS_IGNORE_NO_DIFFERENCE
UNMATCHED = "UNMATCHED START (not decided)"
CONFIG_FILENAME = "config.yaml"
ON, OFF = "on", "off"
MICRO, MACRO = "micro", "macro"
MICRO_TOL = 1e-6  # a score file's pooled micro vs its read-out's
LABEL = "PRELIMINARY (n = 1 seed); decides nothing, K5f stays the reference"

Curves = dict[str, np.ndarray]
Interval = dict[str, float]


@dataclass(frozen=True)
class Arm:
    """One arm's two rows: ``<root>/{K5/s<seed>, eval}``, Nexar reads under ``eval/nexar/<row>``."""

    name: str
    on_row: str
    on_root: Path
    off_row: str
    off_root: Path


def arm_paths(name: str, outputs_root: Path) -> Arm:
    """KA3: the re-run of the on row replaces the saved one wherever it was trained."""
    off_row, on_dirname = constants.KIM_ABLATION_ARMS[name]
    root = outputs_root / constants.KIM_ABLATION_OUTPUT_DIRNAME
    on_row, on_root = name, outputs_root / on_dirname / name
    rerun = name + constants.KIM_ABLATION_RERUN_SUFFIX
    if (root / rerun).is_dir():
        on_row, on_root = rerun, root / rerun
    return Arm(name, on_row, on_root, off_row, root / off_row)


def _load_config(run_dir: Path) -> dict[str, Any]:
    return _flat(yaml.safe_load((run_dir / CONFIG_FILENAME).read_text(encoding="utf-8")))


def config_diff(base: dict[str, Any], got: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Keys that differ; a key one run predates may sit at its default in the other."""
    defaults = _flat(Config().to_dict())
    out: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(base) | set(got)):
        a, b = base.get(key, defaults.get(key)), got.get(key, defaults.get(key))
        if a != b:
            out[key] = (a, b)
    return out


def check_kim_only(on_cfg: dict[str, Any], off_cfg: dict[str, Any], where: str) -> list[str]:
    """KA2: on -> off turns off exactly the on row's true KIM flags. Returns those flags."""
    want = {k: (True, False) for k in constants.KIM_ABLATION_FLAGS if on_cfg[k]}
    if "kim.enabled" not in want:
        raise ValueError(f"{where}: the on row has kim.enabled=false")
    diff = config_diff(on_cfg, off_cfg)
    if diff != want:
        raise ValueError(f"{where}: config diff {diff}, §53 allows exactly {want}")
    return sorted(want)


def metric_verdict(labels: dict[str, str]) -> str:
    """KA6 for one metric, from its labels on the deciding sets."""
    gain = any(labels[s] == GAIN for s in DECISION_SETS)
    cost = any(labels[s] == COST for s in DECISION_SETS)
    if gain and cost:
        return MIXED
    if gain:
        return BENEFITS
    return HURT if cost else NO_DIFFERENCE


def headline(micro: str, macro: str) -> str:
    """Micro verdict, with "micro only" when macro loses where micro gains (KA6)."""
    if micro == BENEFITS and macro in (HURT, MIXED):
        return f"{BENEFITS} (micro only; macro {macro})"
    return f"{micro} (macro: {macro})"


class PooledAuc:
    """Pooled AUC of one curve set under cluster weights: one sort, then O(frames) per draw.

    ``auc(counts)`` is the AUC of the frames with every cluster repeated ``counts`` times, ties at
    one half -- what ``frame_auc`` returns on the concatenated resample.
    """

    def __init__(self, scores: Curves, labels: Curves, cluster: dict[str, int]) -> None:
        ids = sorted(labels)
        s = np.concatenate([np.asarray(scores[v], dtype=np.float64) for v in ids])
        y = np.concatenate([labels[v] > 0 for v in ids])
        c = np.concatenate([np.full(len(labels[v]), cluster[v]) for v in ids])
        order = np.argsort(s, kind="stable")
        s = s[order]
        self._positive = y[order]
        self._cluster = c[order]
        self._starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])

    def auc(self, counts: np.ndarray) -> float:
        """``counts[k]`` = times cluster ``k`` was drawn; ``nan`` when one class is empty."""
        w = counts[self._cluster].astype(np.float64)
        pos = np.add.reduceat(np.where(self._positive, w, 0.0), self._starts)
        neg = np.add.reduceat(np.where(self._positive, 0.0, w), self._starts)
        total = pos.sum() * neg.sum()
        if total == 0:
            return float("nan")
        below = np.cumsum(neg) - neg
        return float((pos * (below + 0.5 * neg)).sum() / total)


def micro_contrasts(curves: dict[str, Curves], labels: Curves, groups: dict[str, str],
                    terms: dict[str, dict[str, int]], draws: int, seed: int
                    ) -> dict[str, Interval]:
    """Point and percentile interval of ``sum(sign * micro(row))`` per term, shared resamples.

    The resamples are ``tad_clip_gate.micro_bootstrap_pairs``'s: same seed, same cluster order.
    """
    names = sorted({groups[v] for v in labels})
    index = {g: i for i, g in enumerate(names)}
    cluster = {v: index[groups[v]] for v in labels}
    pooled = {row: PooledAuc(c, labels, cluster) for row, c in curves.items()}
    one = np.ones(len(names))
    point = {row: p.auc(one) for row, p in pooled.items()}
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(names), size=(draws, len(names)))
    boot = {row: np.empty(draws) for row in pooled}
    for i, draw in enumerate(resamples):
        counts = np.bincount(draw, minlength=len(names))
        for row, p in pooled.items():
            boot[row][i] = p.auc(counts)
    alpha = (1.0 - constants.V2_E1_CI) / 2.0
    out: dict[str, Interval] = {}
    for term, signs in terms.items():
        low, high = np.nanquantile(
            np.sum([sign * boot[row] for row, sign in signs.items()], axis=0),
            [alpha, 1.0 - alpha])
        out[term] = {"mean": float(sum(sign * point[row] for row, sign in signs.items())),
                     "low": float(low), "high": float(high)}
    return out


def _key(arm: str, side: str) -> str:
    return f"{arm}:{side}"


def contrast_terms(arms: list[str]) -> dict[str, dict[str, int]]:
    """``{arm: on - off}`` and ``{arm-vs-first: difference of differences}`` (KA4, KA7)."""
    terms = {a: {_key(a, ON): 1, _key(a, OFF): -1} for a in arms}
    ref = arms[0]
    for a in arms[1:]:
        terms[f"{a} vs {ref}"] = {**terms[a], _key(ref, ON): -1, _key(ref, OFF): 1}
    return terms


def _macro_ci(d: dict[str, float], groups: dict[str, str]) -> Interval | None:
    return cluster_bootstrap_ci(d, groups, constants.V2_E1_BOOTSTRAP, constants.V5_BOOTSTRAP_SEED,
                                constants.V2_E1_CI)


def macro_contrasts(items: dict[str, dict[str, float]], groups: dict[str, str],
                    arms: list[str]) -> dict[str, Interval | None]:
    """Per-item on - off, and its difference from the first arm's, with the read-outs' bootstrap."""
    deltas = {a: paired(items[_key(a, ON)], items[_key(a, OFF)]) for a in arms}
    out = {a: _macro_ci(d, groups) for a, d in deltas.items()}
    ref = arms[0]
    for a in arms[1:]:
        out[f"{a} vs {ref}"] = _macro_ci(paired(deltas[a], deltas[ref]), groups)
    return out


def _nexar_dir(root: Path, row: str) -> Path:
    return root / zs_compare_table.EVAL_DIR / NEXAR / row


def read_set(set_name: str, rows: dict[str, tuple[Path, str]], seed: int) -> dict[str, Any]:
    """Curves (pooled as KA4), labels, clusters, per-item macro values and cells of every row."""
    if set_name == NEXAR:
        dirs = {k: _nexar_dir(root, row) for k, (root, row) in rows.items()}
        curves, labels = load_rows(list(dirs.items()))
        items, cells = {}, {}
        oracle = None
        for k, d in dirs.items():
            readout, items[k] = kw.nexar_read(d)
            whole, oracle = readout["whole"]["micro"], readout["whole"]["clip_oracle_micro"]
            got = _pooled(curves[k], labels)
            if abs(got - whole) > MICRO_TOL:
                raise ValueError(f"nexar/{k}: micro {got} != read-out {whole}")
            crops = readout["crops"]["mean_over_placements"]["macro"]
            cells[k] = {MICRO: got, MACRO: crops["mean"]}
        return {"curves": curves, "labels": labels, "groups": {v: v for v in labels},
                "items": items, "item_groups": {v: v for d in items.values() for v in d},
                "cells": cells, "oracle": oracle}
    views = {k: kw._view(root, seed) for k, (root, _) in rows.items()}
    curves, labels = load_rows(
        [(k, zs_reads.score_source(v, set_name, kw.K5)) for k, v in views.items()])
    if set_name in zs_compare_table.TABLE_MINMAX_SETS:
        curves = {k: {v: normalize_scores(s, constants.SCORE_NORM_MINMAX) for v, s in c.items()}
                  for k, c in curves.items()}
    items, cells = {}, {}
    groups: dict[str, str] = {}
    oracle = None
    for k, view in views.items():
        read, pooled = kw.zero_shot_row(view, set_name)  # gates the read-out's micro
        items[k] = read["items"]
        groups = read["groups"]
        oracle = read.get("extra", {}).get("clip_oracle_micro")
        got = _pooled(curves[k], labels)
        if abs(got - pooled["micro"]) > MICRO_TOL:
            raise ValueError(f"{set_name}/{k}: micro {got} != table {pooled['micro']}")
        cells[k] = {MICRO: got, MACRO: read["macro"]["mean"]}
    return {"curves": curves, "labels": labels,
            "groups": {v: groups.get(v, v) for v in labels},
            "items": items, "item_groups": groups, "cells": cells, "oracle": oracle}


def _pooled(scores: Curves, labels: Curves) -> float:
    ids = sorted(labels)
    return frame_auc(np.concatenate([scores[v] for v in ids]),
                     np.concatenate([labels[v] for v in ids]))


def check_items(items: dict[str, dict[str, float]], set_name: str) -> None:
    first = next(iter(items))
    for k, d in items.items():
        if set(d) != set(items[first]):
            raise ValueError(f"{set_name}: {k} scored other macro items than {first}")


def check_checkpoints(rows: dict[str, tuple[Path, str]], seed: int) -> None:
    """KA2: one checkpoint per row over its five reads, and no two rows share one."""
    owner: dict[str, str] = {}
    for k, (root, row) in rows.items():
        seen = zs_reads.k5_checkpoints(kw._view(root, seed), list(kw.ZS_SETS))
        path = _nexar_dir(root, row) / nexar_eval.READOUT_JSON
        seen[NEXAR] = str(json.loads(path.read_text(encoding="utf-8"))["checkpoints"][f"s{seed}"])
        zs_reads.check_one_checkpoint(seen)
        ckpt = seen[NEXAR]
        if ckpt in owner:
            raise ValueError(f"{k} and {owner[ckpt]} read the same checkpoint {ckpt}")
        owner[ckpt] = k


def check_edge_read(starts: dict[str, dict[str, Any]]) -> None:
    """KA2: K0fepre's step-1 ``mil`` must differ from K0f's, else the edge corpus was not read."""
    pre, ref = constants.KIM_EDGE_PRE_ROW, constants.KIM_FIXED_ARM
    if not {pre, ref} <= set(starts):
        return
    if abs(starts[pre]["new"] - starts[ref]["new"]) <= constants.KIM_START_TOL:
        raise ValueError("K0fepre starts like K0f: forced_normals.json was not read")


def run(args: argparse.Namespace) -> dict[str, Any]:
    arms = [a for a in constants.KIM_ABLATION_ARMS if a in args.arms]
    paths = {a: arm_paths(a, args.outputs_root) for a in arms}
    rows = {_key(a, side): (root, row) for a, p in paths.items()
            for side, root, row in ((ON, p.on_root, p.on_row), (OFF, p.off_root, p.off_row))}
    run_dirs = {k: root / kw.K5 / f"s{args.seed}" for k, (root, _) in rows.items()}
    mul_mil = check_fixed_runs(run_dirs)
    flags = {a: check_kim_only(_load_config(run_dirs[_key(a, ON)]),
                               _load_config(run_dirs[_key(a, OFF)]), a) for a in arms}
    check_checkpoints(rows, args.seed)
    starts = {a: edge.start_check(run_dirs[_key(a, ON)], run_dirs[_key(a, OFF)]) for a in arms}
    check_edge_read(starts)
    terms = contrast_terms(arms)
    sets: dict[str, Any] = {}
    for set_name in ALL_SETS:
        data = read_set(set_name, rows, args.seed)
        check_items(data["items"], set_name)
        sets[set_name] = {
            "cells": data["cells"], "clip_oracle_micro": data["oracle"],
            "videos": len(data["labels"]),
            MICRO: micro_contrasts(data["curves"], data["labels"], data["groups"], terms,
                                   args.draws, constants.V5_BOOTSTRAP_SEED),
            MACRO: macro_contrasts(data["items"], data["item_groups"], arms),
        }
        LOGGER.info("%s: %d rows, %d draws", set_name, len(rows), args.draws)
    report: dict[str, Any] = {
        "rule": RULE, "seed": args.seed, "draws": args.draws, "label": LABEL,
        "arms": {a: {"on": p.on_row, "off": p.off_row, "kim_flags_off": flags[a],
                     "start": starts[a], "env": {s: read_env(run_dirs[_key(a, s)])
                                                 for s in (ON, OFF)}} for a, p in paths.items()},
        "fixed_mul_mil": mul_mil, "sets": sets, "reference": arms[0],
    }
    for a in arms:
        labels = {m: {s: dvs.label({"delta": sets[s][m][a]}) for s in ALL_SETS}
                  for m in (MICRO, MACRO)}
        verdicts = {m: metric_verdict(labels[m]) for m in (MICRO, MACRO)}
        matched = starts[a]["same_draws"]
        report["arms"][a].update({
            "labels": labels, "verdicts": verdicts,
            "verdict": headline(verdicts[MICRO], verdicts[MACRO]) if matched else UNMATCHED,
            "mean_micro_delta": float(np.mean([sets[s][MICRO][a]["mean"] for s in DECISION_SETS])),
        })
    report["ranking"] = sorted((a for a in arms if starts[a]["same_draws"]),
                               key=lambda a: -report["arms"][a]["mean_micro_delta"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = constants.KIM_ABLATION_REPORT_STEM
    write_json_atomic(args.out_dir / f"{stem}.json", report)
    write_text_atomic(args.out_dir / f"{stem}.md", render(report))
    return report


def _signed(c: Interval | None, label: str = "") -> str:
    if c is None:
        return "n/a"
    text = f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"
    return f"**{text}** {label}" if label in (GAIN, COST) else text


def render(r: dict[str, Any]) -> str:
    arms = list(r["arms"])
    lines = [f"# KIM-on - KIM-off on {len(arms)} arms, s{r['seed']} — **{r['label']}**", "",
             f"Rule: `{r['rule']}`. Micro is primary, macro secondary. Deciding sets: "
             f"{', '.join(DECISION_SETS)} (TAD printed). GAIN / COST: |Δ| ≥ "
             f"{constants.V5_MIN_GAIN} and the interval off 0. {r['draws']} micro draws.", "",
             "## Verdict per arm", "",
             "| arm | on / off row | step-1 mil on / off | verdict (micro first) | mean micro Δ |",
             "| --- | --- | --- | --- | --- |"]
    for a, info in r["arms"].items():
        st = info["start"]
        lines.append(f"| {a} | {info['on']} / {info['off']} | {st['control']:.6f} / "
                     f"{st['new']:.6f} | {info['verdict']} | {info['mean_micro_delta']:+.4f} |")
    lines += ["", "Ranking by mean micro Δ on the deciding sets (matched starts only; printed, "
                  "decides nothing): "
                  f"{' > '.join(r['ranking'])}", ""]
    header = "| arm | " + " | ".join(r["sets"]) + " |"
    rule = "| --- |" + " --- |" * len(r["sets"])
    for metric in (MICRO, MACRO):
        lines += [f"## Δ {metric} (on - off)", "", header, rule]
        for a, info in r["arms"].items():
            lines.append(f"| {a} | " + " | ".join(
                _signed(t[metric][a], info["labels"][metric][s]) for s, t in r["sets"].items())
                + " |")
        lines += ["", f"## Δ {metric}: difference of differences vs {r['reference']} (KA7)", "",
                  header, rule]
        lines += [f"| {a} | " + " | ".join(
            _signed(t[metric][f"{a} vs {r['reference']}"]) for t in r["sets"].values()) + " |"
            for a in arms[1:]] + [""]
    lines += ["## Cells (micro / macro)", "", "| arm | side | " + " | ".join(r["sets"]) + " |",
              "| --- | --- |" + " --- |" * len(r["sets"])]
    for a in arms:
        for side in (ON, OFF):
            cells = [t["cells"][_key(a, side)] for t in r["sets"].values()]
            lines.append(f"| {a} | {side} | " + " | ".join(
                f"{c[MICRO]:.4f} / {c[MACRO]:.4f}" for c in cells) + " |")
    oracles = [f"{s} {t['clip_oracle_micro']:.4f}" for s, t in r["sets"].items()
               if t["clip_oracle_micro"] is not None]
    lines += ["", f"Clip oracle micro (a constant score per video): {', '.join(oracles)}.",
              "Nexar macro = per-video crop means; Nexar micro = whole video.", "",
              "## Runs", ""]
    for a, info in r["arms"].items():
        lines.append(f"* {a}: KIM flags off {info['kim_flags_off']}; env on {info['env'][ON]}, "
                     f"off {info['env'][OFF]}")
    lines += [f"* last-epoch mul_mil: "
              f"{ {k: round(v, 4) for k, v in r['fixed_mul_mil'].items()} }",
              "", "KA8: 48 labels at n = 1: about two leave 0 by chance. A single-set label is a "
              "lead, not a finding. Nothing is adopted."]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KIM-on vs KIM-off on six arms (KIM_RUN §53).")
    parser.add_argument("--outputs-root", type=Path, required=True,
                        help="dir that holds kim_fixed/, kim_dvs_ignore/, ..., kim_ablation/")
    parser.add_argument("--arms", nargs="+", default=list(constants.KIM_ABLATION_ARMS),
                        choices=list(constants.KIM_ABLATION_ARMS))
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--draws", type=int, default=constants.BASEMICRO_BOOTSTRAP_DRAWS)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()

__all__ = ["PooledAuc", "check_kim_only", "config_diff", "metric_verdict", "micro_contrasts",
           "render", "run"]
