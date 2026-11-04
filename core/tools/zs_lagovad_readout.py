"""Zero-shot summary of branch ``v_final``: LaGoVAD (released ckpt) vs W0 (no SA²) vs R4 (SA²).

Rule: ``core/docs/final/ZS_TAD_DADA.md`` §7 (written before the released checkpoint was read on
DoTA-CAP-dev or DADA-2000). Pure aggregation -- no model is loaded. Inputs::

    <lagovad-root>/eval/dota/LaGoVAD/B/   protocol_b_eval --baseline, s1[::3] (protocol B)
    <lagovad-root>/eval/dota/LaGoVAD/A/   protocol_b_eval --baseline --stride 8 (printed)
    <lagovad-root>/eval/tad/LaGoVAD/released/   core.evaluate --baseline-ckpt --save-scores
    <lagovad-root>/eval/dada/LaGoVAD/     dada_whole_eval --baseline
    <lagovad-root>/eval/dad/{LaGoVAD/released, W0/s2024, R4/s2024}/   core.evaluate (optional,
                                          ZS_DAD.md; the set is read when eval/dad/ exists)
    --dota W0 <dir> --dota R4 <dir>       the arms' DoTA-CAP-dev protocol-B reads (wu / stp)
    <zs-root>/eval/{tad,dada}/{W0,R4}/    the arms' zero-shot reads (zs_tad_dada_r4.ipynb)

Per set, every row's macro (95 % cluster bootstrap: DoTA = source video, TAD / DADA = the video),
micro and clip-level AUC, and three paired per-video contrasts:

* ``W0 - LaGoVAD`` and ``R4 - LaGoVAD``: "above" iff low > 0, "below" iff high < 0, else "not
  separated" (§26 T6's wording). **Not a matched comparison** -- LaGoVAD trained on PreVAD, the
  arms on Nexar -- and the released checkpoint's zero-shot status on DADA-2000 is unverified.
* ``R4 - W0``: ``stp_readout.sentence`` ("gain / cost / no difference shown"), as in §3.

The arms have one seed, so every sentence carries **PRELIMINARY (n = 1 seed)** and adopts
nothing. Integrity (§7.3), checked here: the TAD micro of the released checkpoint must equal
Gate T0 (0.7912) within ``ZS_LAGOVAD_T0_TOL``; every row of a set must cover the same items.

CLI::

    python -m core.tools.zs_lagovad_readout --lagovad-root outputs/zs_lagovad \\
        --zs-root outputs/zs --dota W0 outputs/wu/eval/dota/dota_cap_dev/W0 \\
        --dota R4 outputs/stp/eval/dota/dota_cap_dev/R4

``check-inputs`` (used by the notebook before any number): the plain CLIP rows LaGoVAD reads
equal the CLIP columns of the arms' bake, row for row (C2, C13)::

    python -m core.tools.zs_lagovad_readout check-inputs --plain cache/clip/DADA2000_orig \\
        --bake cache/v2/A2_S/DADA2000_orig_test_sources_nexar --ids-file ids.txt --plain-stride 1
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
from core.metrics import cluster_bootstrap_ci, frame_auc, normalize_scores
from core.tools import protocol_b_eval
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_transfer_readout import ArmRead, read_dada, read_tad_runs
from core.tools.stp_readout import _ci, _num, paired, sentence

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "zs_lagovad_readout"
RULE = "core/docs/final/ZS_TAD_DADA.md §7"
LABEL = "PRELIMINARY (n = 1 seed)"
EVAL_DIR = "eval"
DOTA, TAD, DADA = "dota", "tad", "dada"
DAD = constants.DAD_OUTPUT_SET
SETS = (DOTA, TAD, DADA)
OPTIONAL_SETS = (DAD,)  # read when <lagovad-root>/eval/<set>/ exists (ZS_DAD.md)
PROTOCOL_B, PROTOCOL_A = "B", "A"
SET_TITLE = {
    DOTA: "DoTA-CAP-dev (n/1397) · protocol B (s1[::3], whole clip, native frames)",
    TAD: "TAD test · 100 videos, s8, raw scores (§26)",
    DADA: "DADA-2000 · 370 T2-test sources, whole video, s8 (D-N20)",
    DAD: (f"DAD · testing split, whole video, s{constants.DAD_STRIDE} at "
          f"{constants.DAD_ASSUMED_FPS} fps (ZS_DAD.md)"),
}
ABOVE, BELOW, NOT_SEPARATED = "above", "below", "not separated"
MICRO_NOTE = {
    DOTA: "per-clip min-max, pooled (lesson 12: DoTA-CAP is all-abnormal)",
    TAD: "raw, pooled (clip oracle 0.9226: measures clip classification)",
    DADA: "raw, pooled",
    DAD: "raw, pooled (equal-length videos: the length ruler is uninformative)",
}


# ---------------------------------------------------------------------------
# pure pieces (tested without a model)
# ---------------------------------------------------------------------------
def versus_sentence(c: dict[str, float] | None) -> str:
    """§26 T6's three-way sentence for an arm minus the released checkpoint."""
    if c is None:
        return NOT_SEPARATED
    if c["low"] > 0:
        return ABOVE
    return BELOW if c["high"] < 0 else NOT_SEPARATED


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def minmax_micro(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> float:
    """Pooled frame AUC after per-clip min-max (the DoTA pooling, lesson 12)."""
    ids = sorted(labels)
    return frame_auc(
        np.concatenate([normalize_scores(scores[v], constants.SCORE_NORM_MINMAX) for v in ids]),
        np.concatenate([labels[v] for v in ids]),
    )


def read_dota(arm_dir: Path) -> ArmRead:
    """One protocol-B read of DoTA-CAP-dev (``protocol_b_eval`` output dir)."""
    r = _json(arm_dir / protocol_b_eval.READOUT_JSON)
    items = _json(arm_dir / protocol_b_eval.CLIP_AUCS_JSON)
    scores, labels = protocol_b_eval.read_clip_scores(arm_dir / protocol_b_eval.CLIP_SCORES_NPZ)
    groups = {v: dota_group(v) for v in items}
    return {
        "per_seed": r["per_run_macro"],
        "items": items,
        "groups": groups,
        "macro": cluster_bootstrap_ci(items, groups, constants.V2_E1_BOOTSTRAP,
                                      constants.V5_BOOTSTRAP_SEED, constants.V2_E1_CI),
        "micro": minmax_micro(scores, labels),
        "clip_auc": None,  # every DoTA-CAP clip contains its accident
        "extra": {"position_ruler": r.get("position_ruler"), "stride": r.get("protocol")},
    }


def delta_ci(a: ArmRead, b: ArmRead) -> dict[str, float] | None:
    """Per-item ``a - b`` with ``a``'s clusters (DoTA: source video; else the video itself)."""
    d = paired(a["items"], b["items"])
    groups = {v: a["groups"][v] for v in d}
    return cluster_bootstrap_ci(d, groups, constants.V2_E1_BOOTSTRAP,
                                constants.V5_BOOTSTRAP_SEED, constants.V2_E1_CI)


def check_same_items(rows: dict[str, ArmRead], set_name: str) -> None:
    """Every row of a set scored the same two-class items, or the contrasts are not paired."""
    ref_name = next(iter(rows))
    ref = set(rows[ref_name]["items"])
    for name, r in rows.items():
        if set(r["items"]) != ref:
            raise ValueError(
                f"{set_name}: {name} scored {len(r['items'])} items, {ref_name} {len(ref)} "
                f"(symmetric difference {len(ref ^ set(r['items']))})"
            )


def t0_check(lagovad_tad: ArmRead) -> dict[str, Any]:
    """§7.3: the released checkpoint must reproduce Gate T0's TAD micro."""
    gap = float(lagovad_tad["micro"]) - constants.ZS_LAGOVAD_T0_MICRO
    return {"micro": lagovad_tad["micro"], "t0": constants.ZS_LAGOVAD_T0_MICRO, "gap": gap,
            "pass": abs(gap) <= constants.ZS_LAGOVAD_T0_TOL}


def input_mismatches(
    plain: np.ndarray, bake: np.ndarray, plain_stride: int, atol: float
) -> str | None:
    """Why LaGoVAD's plain rows are not the bake's CLIP columns, or ``None`` if they are."""
    rows = plain[::plain_stride]
    if len(rows) != len(bake):
        return f"rows {len(rows)} (plain[::{plain_stride}]) != bake {len(bake)}"
    dim = constants.ZS_LAGOVAD_INPUT_DIM
    if rows.shape[1] != dim or bake.shape[1] < dim:
        return f"dims plain {rows.shape[1]} bake {bake.shape[1]} (need {dim} / >= {dim})"
    worst = float(np.max(np.abs(rows.astype(np.float64) - bake[:, :dim].astype(np.float64))))
    return None if worst <= atol else f"max |plain - bake[:, :{dim}]| = {worst:.3g} > {atol}"


def check_inputs(
    plain_dir: Path, bake_dir: Path, ids: list[str], plain_stride: int,
    atol: float = constants.ZS_LAGOVAD_INPUT_ATOL,
) -> dict[str, str]:
    """Every id whose plain CLIP rows differ from the bake's CLIP columns (empty = pass)."""
    bad: dict[str, str] = {}
    for v in ids:
        why = input_mismatches(np.load(plain_dir / f"{v}.npy"), np.load(bake_dir / f"{v}.npy"),
                               plain_stride, atol)
        if why is not None:
            bad[v] = why
    return bad


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def read_rows(args: argparse.Namespace, set_name: str) -> tuple[dict[str, ArmRead], Any]:
    """``({row: read}, printed)`` of one set: LaGoVAD first, then the arms."""
    base = args.lagovad_root / EVAL_DIR / set_name / constants.ZS_LAGOVAD_ARM
    rows: dict[str, ArmRead] = {}
    printed: Any = None
    if set_name == DOTA:
        rows[constants.ZS_LAGOVAD_ARM] = read_dota(base / PROTOCOL_B)
        if (base / PROTOCOL_A).is_dir():
            printed = read_dota(base / PROTOCOL_A)
        for name, arm_dir in args.dota:
            rows[name] = read_dota(Path(arm_dir))
        return rows, printed
    if set_name == TAD:
        rows[constants.ZS_LAGOVAD_ARM] = read_tad_runs(
            base, [constants.ZS_LAGOVAD_RUN], constants.V5_BOOTSTRAP_SEED)
        for arm in constants.ZS_ARMS:
            rows[arm] = read_tad_runs(args.zs_root / EVAL_DIR / TAD / arm,
                                      [f"s{args.seed}"], constants.V5_BOOTSTRAP_SEED)
        return rows, t0_check(rows[constants.ZS_LAGOVAD_ARM])
    if set_name == DAD:  # every row is a core.evaluate output under eval/dad/<row>/<run>/
        rows[constants.ZS_LAGOVAD_ARM] = read_tad_runs(
            base, [constants.ZS_LAGOVAD_RUN], constants.V5_BOOTSTRAP_SEED, None)
        for arm in constants.ZS_ARMS:
            rows[arm] = read_tad_runs(base.parent / arm, [f"s{args.seed}"],
                                      constants.V5_BOOTSTRAP_SEED, None)
        return rows, None
    rows[constants.ZS_LAGOVAD_ARM] = read_dada(base)
    for arm in constants.ZS_ARMS:
        rows[arm] = read_dada(args.zs_root / EVAL_DIR / DADA / arm)
    return rows, None


def summarize_set(set_name: str, rows: dict[str, ArmRead], printed: Any) -> dict[str, Any]:
    check_same_items(rows, set_name)
    base = rows[constants.ZS_LAGOVAD_ARM]
    out: dict[str, Any] = {
        "title": SET_TITLE[set_name],
        "micro_note": MICRO_NOTE[set_name],
        "rows": {n: {k: r[k] for k in ("macro", "per_seed", "micro", "clip_auc", "extra")}
                 for n, r in rows.items()},
        "versus_lagovad": {},
        "sa2_minus_host": None,
    }
    for arm in constants.ZS_ARMS:
        if arm in rows:
            c = delta_ci(rows[arm], base)
            out["versus_lagovad"][arm] = {"delta": c, "sentence": versus_sentence(c)}
    if all(a in rows for a in constants.ZS_ARMS):
        c = delta_ci(rows[constants.ZS_SA2_ARM], rows[constants.ZS_CONTROL_ARM])
        out["sa2_minus_host"] = {"delta": c, "sentence": sentence(c)}
    if set_name == DOTA and printed is not None:
        out["lagovad_protocol_a"] = {"macro": printed["macro"], "micro": printed["micro"]}
    if set_name == TAD:
        out["t0_check"] = printed
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dota or {n for n, _ in args.dota} != set(constants.ZS_ARMS):
        raise ValueError(f"--dota must name each of {constants.ZS_ARMS} once")
    present = [*SETS, *(s for s in OPTIONAL_SETS
                        if (args.lagovad_root / EVAL_DIR / s).is_dir())]
    sets = {s: summarize_set(s, *read_rows(args, s)) for s in present}
    result: dict[str, Any] = {
        "rule": RULE, "label": LABEL, "seed": args.seed,
        "baseline": constants.ZS_LAGOVAD_ARM, "control": constants.ZS_CONTROL_ARM,
        "arm": constants.ZS_SA2_ARM, "sets": sets,
    }
    check = sets[TAD]["t0_check"]
    if not check["pass"]:
        raise ValueError(f"released checkpoint does not reproduce Gate T0 on TAD: {check} "
                         "(wrong checkpoint, config or TAD_ncc) -- no number of this read is valid")
    out_dir = args.out_dir or args.lagovad_root / "REPORTS"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / f"{READOUT_NAME}.json", result)
    write_text_atomic(out_dir / f"{READOUT_NAME}.md", render_markdown(result))
    LOGGER.info("read-out -> %s", out_dir / f"{READOUT_NAME}.md")
    return result


def _f(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


ROW_TITLE = {
    constants.ZS_LAGOVAD_ARM: "LaGoVAD (released ckpt, PreVAD-trained)",
    constants.ZS_CONTROL_ARM: "W0 (no SA², Nexar, s2024)",
    constants.ZS_SA2_ARM: "R4 (SA², Nexar, s2024)",
}


def render_markdown(r: dict[str, Any]) -> str:
    lines = [
        f"# Zero-shot summary — LaGoVAD vs W0 vs R4 — **{r['label']}**",
        "",
        f"Rule: `{r['rule']}`. Arms: Nexar-trained, seed {r['seed']}, prefix OFF. "
        "LaGoVAD = the released `best.ckpt`, KIP off, plain CLIP `_ncc` rows (the arms' own CLIP "
        "columns, checked row for row). **Not a matched comparison**: LaGoVAD trained on PreVAD.",
        "",
        "## Summary (macro AUC [95 % cluster CI])",
        "",
        "| row | " + " | ".join(r["sets"][s]["title"].split(" · ")[0] for s in r["sets"]) + " |",
        "|---|" + "---|" * len(r["sets"]),
    ]
    for name in (constants.ZS_LAGOVAD_ARM, *constants.ZS_ARMS):
        cells = [_num(r["sets"][s]["rows"][name]["macro"]) for s in r["sets"]]
        lines.append(f"| {ROW_TITLE[name]} | " + " | ".join(cells) + " |")
    lines.append("")
    for s in r["sets"].values():
        lines += [
            f"## {s['title']}",
            "",
            f"| row | macro | micro ({s['micro_note']}) | clip AUC |",
            "|---|---|---|---|",
        ]
        for name, x in s["rows"].items():
            lines.append(f"| {ROW_TITLE[name]} | {_num(x['macro'])} | {_f(x['micro'])} | "
                         f"{_f(x['clip_auc'])} |")
        lines.append("")
        for arm, v in s["versus_lagovad"].items():
            lines.append(f"* {arm} - LaGoVAD (paired, per item): {_ci(v['delta'])} → "
                         f"**{v['sentence']}** — {r['label']}, not matched")
        if s["sa2_minus_host"] is not None:
            p = s["sa2_minus_host"]
            lines.append(f"* **R4 - W0 (paired, per item): {_ci(p['delta'])} → {p['sentence']} "
                         f"— {r['label']}**")
        if "lagovad_protocol_a" in s:
            a = s["lagovad_protocol_a"]
            lines.append(f"* printed: LaGoVAD at protocol A (s1[::8], its training rate): macro "
                         f"{_num(a['macro'])}, micro {_f(a['micro'])}")
        if "t0_check" in s:
            t = s["t0_check"]
            lines.append(f"* integrity: LaGoVAD TAD micro {t['micro']:.4f} vs Gate T0 "
                         f"{t['t0']:.4f} (gap {t['gap']:+.5f}) → {'PASS' if t['pass'] else 'FAIL'}")
        extra = s["rows"][constants.ZS_CONTROL_ARM]["extra"]
        rulers = [f"{k.replace('_', ' ')} {_num(v) if isinstance(v, dict) else _f(v)}"
                  for k, v in extra.items() if k.endswith(("ruler", "micro")) and v is not None]
        if rulers:
            lines.append(f"* rulers (no pixels): {' · '.join(rulers)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command")
    chk = sub.add_parser("check-inputs", help="plain CLIP rows == the bake's CLIP columns")
    chk.add_argument("--plain", type=Path, required=True, help="plain CLIP cache (LaGoVAD's)")
    chk.add_argument("--bake", type=Path, required=True, help="the arms' baked cache")
    chk.add_argument("--ids-file", type=Path, required=True)
    chk.add_argument("--plain-stride", type=int, default=1,
                     help="rows = plain[::stride] (DoTA s1 -> 3; DADA / TAD s8 caches -> 1)")
    parser.add_argument("--lagovad-root", type=Path, help="outputs/zs_lagovad (eval/ under it)")
    parser.add_argument("--zs-root", type=Path, help="outputs/zs (W0 / R4 TAD + DADA reads)")
    parser.add_argument("--dota", nargs=2, metavar=("ARM", "DIR"), action="append",
                        help="an arm's DoTA-CAP-dev protocol-B read dir (W0 and R4)")
    parser.add_argument("--seed", type=int, default=constants.ZS_SEED)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <lagovad-root>/REPORTS")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "check-inputs":
        ids = [v for v in args.ids_file.read_text(encoding="utf-8").split() if v]
        bad = check_inputs(args.plain, args.bake, ids, args.plain_stride)
        if bad:
            raise SystemExit(f"{len(bad)}/{len(ids)} inputs differ: {dict(list(bad.items())[:5])}")
        LOGGER.info("inputs OK: %d ids, plain[::%d] == bake[:, :%d]", len(ids), args.plain_stride,
                    constants.ZS_LAGOVAD_INPUT_DIM)
        return
    if args.lagovad_root is None or args.zs_root is None:
        raise SystemExit("--lagovad-root and --zs-root are required for the read-out")
    run(args)


if __name__ == "__main__":
    main()
