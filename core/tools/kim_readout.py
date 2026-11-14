"""KIM first-pass read-out: K0k9 / K2 / K5 against K0, W0 printed (no model is loaded).

Rule ``core/docs/final/KIM_RUN.md`` §3.1 and §6, written before any K number. Inputs are the
ordinary protocol outputs of each row:

* ``--dota NAME DIR`` -- ``protocol_b_eval`` out dir (``clip_aucs.json``, DoTA-CAP-dev);
* ``--dada NAME DIR`` -- ``dada_whole_eval`` out dir (370 sources, whole video, s8);
* ``--tad NAME DIR``  -- the row's TAD dir holding ``s<seed>/`` (``core.evaluate --save-scores``).

Per set and contrast: per-video paired Δ of the frame AUC over the two-class videos both rows
scored, mean with a 95 % cluster bootstrap (DoTA: source video; DADA / TAD: the video), and the
sentence **gain** (low > 0 and mean ≥ +0.01) / **cost** (high < 0) / **no difference shown**,
always ``PRELIMINARY (n = 1)``. Endpoints: DoTA-CAP-dev and DADA; TAD is printed, never decides.

Decisions fixed before the numbers: K0k9 - K0 gain on **both** endpoints → kernel 9 hosts every K
row (§3.1), else kernel 3; K5 - control cost on **both** endpoints → KIM line closed (§6), else
continue. §6.1's stricter stop is read from the numbers. At the replicate seed the ``replication``
line replaces ``stop_rule``: K5 - control gain on DADA → reproduced (§6.2). K5d - W0 gain on DADA
→ arm K5d goes on, else the KIM line closes (``dvs_host``, §12). K5ln - K5 gain on DADA with no
cost on DoTA-CAP-dev → s2025 next, else the LayerNorm line closes (``ln_rule``, §25); at the
replicate seed a DADA gain alone reproduces it. Printed beside K5ln: the step-1 start check
(``mil`` within ``KIM_START_TOL`` of K5's) and, with ``--motion-cache``, each corpus' per-step
norm of the baked motion columns (what the LayerNorm erases). Printed: K0 - W0 (the
cost of DVS off alone), K5 - W0, Kcap - W0, K5 - K0k9, ego / non-ego splits, the position rulers,
the training diagnostics of every run (first → last, plus the last-epoch mean ``mil``), and with
``--pilot`` the training-free rank average of the host with ``vt2`` (§5.2).

CLI::

    python -m core.tools.kim_readout --dota K0 .../eval/dota/K0 --dota K5 .../eval/dota/K5 \\
        --dada K0 .../eval/dada/K0 --dada K5 .../eval/dada/K5 --tad K0 .../eval/tad/K0 \\
        --run K0 .../K0/s2024 --run K5 .../K5/s2024 --metadata data/DoTA/metadata_val.json \\
        --split-file data/DoTA/val_split.txt --out-dir outputs/kim/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.dota import parse_metadata, read_split_ids
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_transfer_readout import read_dada, read_tad
from core.tools.protocol_b_eval import CLIP_AUCS_JSON, READOUT_JSON
from core.tools.stp_readout import COST, GAIN, paired, sentence
from core.tools.stp_readout import ci as dota_ci
from core.tools.zs_readout import video_ci

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "kim_readout"
RULE = "core/docs/final/KIM_RUN.md §3.1, §6, §6.1, §6.2, §6.3, §12, §25"
LABEL = "PRELIMINARY (n = 1)"
DOTA, DADA, TAD = "dota", "dada", "tad"
ENDPOINTS = (DOTA, DADA)
SET_TITLE = {
    DOTA: "DoTA-CAP-dev (protocol B, per-clip min-max)",
    DADA: "DADA-2000 (370 sources, whole video, s8, raw)",
    TAD: "TAD test (100 videos, s8, raw) — printed, never decides",
}
K0K9, K2, K5 = "K0k9", "K2", "K5"
K5D = constants.KIM_DVS_ARM
K5LN = constants.KIM_LN_ARM
DIAG_KEYS = ("mil", "dvs_sup", "kim_mil", "kim_nkc", *constants.KIM_DIAG_KEYS,
             *constants.KIM_SEAM_DIAG_KEYS,
             "motion_share", "w_u_norm")
CONTINUE, CLOSED, PENDING = "continue to §3's second group", "KIM line closed (§6)", "pending"
KERNEL_9, KERNEL_3 = "host = kernel 9 for every K row", "host keeps kernel 3"
REPRODUCED = "reproduced at n = 2: the line goes on (§6.2)"
NOT_REPRODUCED = "not reproduced: no Kcap or K2 run, no variants (§6.2)"
DVS_GOES_ON = "gain on DADA vs the DVS-on host: s2025 and a DVS-on Kcap next, nothing adopted (§12)"
DVS_CLOSED = "no gain on DADA vs the DVS-on host: the KIM line closes (§12)"
LN_GOES_ON = "gain on DADA, no cost on DoTA-CAP-dev: K5ln s2025 next, nothing adopted (§25)"
LN_CLOSED = "no gain on DADA, or a cost on DoTA-CAP-dev: the LayerNorm line closes (§25)"
LN_REPRODUCED = "K5ln - K5 gain on DADA at n = 2: reproduced, still nothing adopted (§25)"
LN_NOT_REPRODUCED = "not reproduced at the replicate seed: the LayerNorm line closes (§25)"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_dota_row(d: Path) -> dict[str, Any]:
    readout = _json(d / READOUT_JSON)
    return {"items": _json(d / CLIP_AUCS_JSON), "macro": readout.get("macro"),
            "ruler": readout.get("position_ruler"), "steps": readout.get("global_steps")}


def read_rows(set_name: str, dirs: dict[str, Path], seed: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name, d in dirs.items():
        if set_name == DOTA:
            rows[name] = read_dota_row(d)
            continue
        r = read_dada(d) if set_name == DADA else read_tad(d, [seed], constants.V5_BOOTSTRAP_SEED)
        rows[name] = {"items": r["items"], "macro": r["macro"], "micro": r["micro"],
                      "clip_auc": r["clip_auc"], "ruler": r["extra"].get("monotone_ruler"),
                      "middle": r["extra"].get("middle_ruler")}
    return rows


def contrast_pairs(names: list[str], control: str) -> list[tuple[str, str, str]]:
    """``(a, b, what)`` for every contrast whose two rows are present."""
    pairs = []
    if control in names and constants.KIM_W0 in names:
        what = "DVS off alone" if control == constants.KIM_CONTROL else "DVS off + host change"
        pairs.append((control, constants.KIM_W0, f"{what} (printed)"))
    for arm, what in ((K0K9, "host kernel 9 (§3.1)"), (K2, "KIM, no prototypes"),
                      (K5, "full KIM"),
                      (K5D, "full KIM + seam mask, DVS on, vs DVS off (printed)"),
                      (constants.KIM_CAP_ARM, "capacity + losses, no differences")):
        if arm in names and control in names and arm != control:
            pairs.append((arm, control, what))
    if K5 in names and K2 in names:
        pairs.append((K5, K2, "prototypes + NKC"))
    if K5 in names and constants.KIM_CAP_ARM in names:
        pairs.append((K5, constants.KIM_CAP_ARM, "the differences (§6.3)"))
    if K5 in names and constants.KIM_W0 in names:
        pairs.append((K5, constants.KIM_W0, "full KIM, DVS off, vs the DVS-on host (printed)"))
    if constants.KIM_CAP_ARM in names and constants.KIM_W0 in names:
        pairs.append((constants.KIM_CAP_ARM, constants.KIM_W0,
                      "branch + losses, DVS off, vs the DVS-on host (printed)"))
    if K5 in names and K0K9 in names:
        pairs.append((K5, K0K9, "full KIM vs host kernel 9, no KIM (printed)"))
    if K5D in names and constants.KIM_W0 in names:
        pairs.append((K5D, constants.KIM_W0, "full KIM + seam mask, both DVS on (§12)"))
    if K5D in names and K5 in names:
        pairs.append((K5D, K5, "DVS on vs off, both full KIM (printed)"))
    if K5LN in names and K5 in names:
        pairs.append((K5LN, K5, "per-step LayerNorm on the motion columns (§25)"))
    if K5LN in names and constants.KIM_W0 in names:
        pairs.append((K5LN, constants.KIM_W0, "K5ln vs the DVS-on host (printed)"))
    return pairs


def contrast(
    a: dict[str, float], b: dict[str, float], set_name: str, ego: dict[str, bool] | None
) -> dict[str, Any]:
    ci = dota_ci if set_name == DOTA else video_ci
    delta = paired(a, b)
    c = ci(delta)
    out: dict[str, Any] = {"delta": c, "sentence": sentence(c)}
    if set_name == DOTA and ego is not None:
        for split, flag in (("ego", True), ("non_ego", False)):
            out[split] = ci({v: d for v, d in delta.items() if ego.get(v) is flag})
    return out


def read_set(
    set_name: str, dirs: dict[str, Path], control: str, seed: int, ego: dict[str, bool] | None
) -> dict[str, Any]:
    rows = read_rows(set_name, dirs, seed)
    contrasts = {
        f"{a} - {b}": {"what": what, **contrast(rows[a]["items"], rows[b]["items"], set_name, ego)}
        for a, b, what in contrast_pairs(list(rows), control)
    }
    return {"title": SET_TITLE[set_name],
            "rows": {n: {k: v for k, v in r.items() if k != "items"} for n, r in rows.items()},
            "contrasts": contrasts}


def _endpoint_sentences(sets: dict[str, Any], key: str) -> list[str] | None:
    """The contrast's sentence on each endpoint; ``None`` if an endpoint lacks it."""
    got = [sets[s]["contrasts"].get(key) for s in ENDPOINTS if s in sets]
    if len(got) != len(ENDPOINTS) or any(g is None for g in got):
        return None
    return [g["sentence"] for g in got]


def decide(
    sets: dict[str, Any], control: str, seed: int = constants.KIM_SEED
) -> dict[str, str]:
    """§3.1 and §6 at the first seed; at the replicate seed §6.2 alone (DADA decides)."""
    out: dict[str, str] = {}
    ln = _ln_rule(sets, seed)
    if ln is not None:
        return ln  # §25 is the rule in force once K5ln is read
    if control == constants.KIM_CONTROL:
        k9 = _endpoint_sentences(sets, f"{K0K9} - {control}")
        if k9 is not None:
            out["kernel"] = KERNEL_9 if all(s == GAIN for s in k9) else KERNEL_3
    if seed == constants.KIM_REPLICATE_SEED:
        dada = sets.get(DADA, {}).get("contrasts", {}).get(f"{K5} - {control}")
        out["replication"] = PENDING if dada is None else (
            REPRODUCED if dada["sentence"] == GAIN else NOT_REPRODUCED)
        return out
    dvs = sets.get(DADA, {}).get("contrasts", {}).get(f"{K5D} - {constants.KIM_W0}")
    if dvs is not None:
        out["dvs_host"] = DVS_GOES_ON if dvs["sentence"] == GAIN else DVS_CLOSED
        return out  # §12 is the rule in force once K5d is read: no §6 verdict beside it
    k5 = _endpoint_sentences(sets, f"{K5} - {control}")
    out["stop_rule"] = PENDING if k5 is None else (
        CLOSED if all(s == COST for s in k5) else CONTINUE)
    return out


def _ln_rule(sets: dict[str, Any], seed: int) -> dict[str, str] | None:
    """§25 on K5ln - K5; ``None`` when that contrast was not read on DADA."""
    key = f"{K5LN} - {K5}"
    dada = sets.get(DADA, {}).get("contrasts", {}).get(key)
    if dada is None:
        return None
    if seed == constants.KIM_REPLICATE_SEED:
        return {"ln_rule": LN_REPRODUCED if dada["sentence"] == GAIN else LN_NOT_REPRODUCED}
    dota = sets.get(DOTA, {}).get("contrasts", {}).get(key)
    if dota is None:
        return {"ln_rule": PENDING}
    ok = dada["sentence"] == GAIN and dota["sentence"] != COST
    return {"ln_rule": LN_GOES_ON if ok else LN_CLOSED}


def start_check(diagnostics: dict[str, Any], arm: str, control: str) -> dict[str, Any] | None:
    """Step-1 ``mil`` of ``arm`` vs ``control`` (printed, never decides; §12, §25)."""
    a, b = diagnostics.get(arm, {}).get("mil"), diagnostics.get(control, {}).get("mil")
    if a is None or b is None:
        return None
    gap = abs(float(a["first"]) - float(b["first"]))
    return {"arm": arm, "control": control, "arm_mil": a["first"], "control_mil": b["first"],
            "gap": gap, "matched": gap <= constants.KIM_START_TOL}


def motion_profile(cache_dir: Path, hidden_dim: int = constants.HIDDEN_DIM) -> dict[str, Any]:
    """Per-step L2 norm of the baked motion columns over every ``.npy`` row file in a cache."""
    norms = []
    for path in sorted(cache_dir.glob("*.npy")):
        rows = np.load(path, mmap_mode="r")
        if rows.ndim != 2 or rows.shape[1] <= hidden_dim:
            raise ValueError(f"{path} has shape {rows.shape}: not a motion-baked (A2) row file")
        norms.append(np.linalg.norm(np.asarray(rows[:, hidden_dim:], dtype=np.float64), axis=1))
    if not norms:
        raise ValueError(f"no .npy row files in {cache_dir}")
    flat = np.concatenate(norms)
    q10, q50, q90 = np.quantile(flat, [0.1, 0.5, 0.9])
    return {"videos": len(norms), "steps": int(flat.size), "mean": float(flat.mean()),
            "q10": float(q10), "median": float(q50), "q90": float(q90)}


def run_diagnostics(run_dir: Path) -> dict[str, Any] | None:
    """First / last of each diagnostic, and the mean ``mil`` over the last epoch."""
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return None
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return None
    out: dict[str, Any] = {"last_step": lines[-1].get("global_step")}
    for key in DIAG_KEYS:
        seen = [r[key] for r in lines if key in r]
        if seen:
            out[key] = {"first": seen[0], "last": seen[-1]}
    last_epoch = lines[-1].get("epoch")
    tail = [r["mil"] for r in lines if r.get("epoch") == last_epoch and "mil" in r]
    out["mil_last_epoch_mean"] = sum(tail) / len(tail) if tail else None
    for key in constants.KIM_SEAM_DIAG_KEYS:  # §12 reads run-level shares, not one batch
        seen = [r[key] for r in lines if key in r]
        if seen:
            out[f"{key}_mean"] = sum(seen) / len(seen)
    return out


def free_baseline(pilot_json: Path) -> dict[str, Any]:
    """The pilot's training-free rank average, per read-out set (printed, never decides)."""
    pilot = _json(pilot_json)
    out: dict[str, Any] = {"host": pilot["host"], "sets": {}}
    for name, pilot_set in ((DOTA, constants.V2_SPLIT_DOTA_CAP_DEV), (DADA, "dada"), (TAD, "tad")):
        s = pilot["sets"].get(pilot_set)
        if s is not None:
            out["sets"][name] = {"host": s["rows"]["host"]["macro"],
                                 "rank": s["rows"]["host+vt2 (rank avg)"]["macro"],
                                 "delta": s["kp2_delta"]}
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    ego = None
    if args.metadata is not None and args.split_file is not None:
        ego = {r.video_id: r.is_ego
               for r in parse_metadata(args.metadata, read_split_ids(args.split_file))}
    given = {DOTA: args.dota, DADA: args.dada, TAD: args.tad}
    sets = {s: read_set(s, {n: Path(d) for n, d in pairs}, args.control, args.seed, ego)
            for s, pairs in given.items() if pairs}
    if not sets:
        raise ValueError("give at least one --dota / --dada / --tad row")
    result: dict[str, Any] = {
        "rule": RULE, "label": LABEL, "control": args.control, "seed": args.seed,
        "disclosure": constants.KIM_PILOT_DISCLOSURE, "p_T2_cap_dev": constants.V5_P_T2_CAP_DEV,
        "p_cap_dev": constants.V2_E3_P_CAP,
        "free_baseline": free_baseline(args.pilot) if args.pilot is not None else None,
        "sets": sets, "decisions": decide(sets, args.control, args.seed),
        "diagnostics": {n: d for n, rd in args.run or []
                        if (d := run_diagnostics(Path(rd))) is not None},
    }
    result["start_check"] = start_check(result["diagnostics"], K5LN, K5)
    result["motion_profile"] = {n: motion_profile(Path(d)) for n, d in args.motion_cache or []}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{READOUT_NAME}.json", result)
    write_text_atomic(args.out_dir / f"{READOUT_NAME}.md", render_markdown(result))
    LOGGER.info("decisions: %s", result["decisions"])
    return result


def _num(c: Any) -> str:
    if c is None:
        return "—"
    if isinstance(c, (int, float)):
        return f"{c:.4f}"
    return f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _delta(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _diag(d: dict[str, Any], key: str) -> str:
    v = d.get(key)
    return "—" if v is None else f"{v['first']:.4f} → {v['last']:.4f}"


def _control_note(control: str) -> str:
    return " (W0 + DVS off)" if control == constants.KIM_CONTROL else ""


def render_markdown(r: dict[str, Any]) -> str:
    lines = [f"# KIM first-pass read-out, Nexar-trained, seed {r['seed']} — **{r['label']}**", "",
             f"Rule: `{r['rule']}`. Control = **{r['control']}**{_control_note(r['control'])}; "
             "W0 printed. "
             f"{r['disclosure']}", ""]
    for s in r["sets"].values():
        lines += [f"## {s['title']}", "", "| row | macro | micro | t/N ruler | middle ruler |",
                  "| --- | --- | --- | --- | --- |"]
        for name, row in s["rows"].items():
            lines.append(f"| {name} | {_num(row['macro'])} | {_num(row.get('micro'))} | "
                         f"{_num(row.get('ruler'))} | {_num(row.get('middle'))} |")
        lines += ["", "| contrast | what | Δ | sentence | Δ ego | Δ non-ego |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, c in s["contrasts"].items():
            lines.append(f"| {name} | {c['what']} | {_delta(c['delta'])} | "
                         f"{c['sentence']} — {r['label']} | {_delta(c.get('ego'))} | "
                         f"{_delta(c.get('non_ego'))} |")
        lines.append("")
    lines += [f"Printed: v2's cubic position prior `p_T2` on CAP-dev = {r['p_T2_cap_dev']:.4f}, "
              f"in-domain cubic `p` = {r['p_cap_dev']:.4f}.", ""]
    free = r.get("free_baseline")
    if free:
        lines += [f"## Training-free baseline: {free['host']} + `vt2` rank average "
                  "(§5.2, printed, never decides)", "",
                  "| set | host macro | rank avg macro | Δ (paired) |", "| --- | --- | --- | --- |"]
        lines += [f"| {name} | {_num(s['host'])} | {_num(s['rank'])} | {_delta(s['delta'])} |"
                  for name, s in free["sets"].items()]
        lines.append("")
    lines += ["## Decisions (fixed before the numbers)", ""]
    lines += [f"- **{k}**: {v}" for k, v in r["decisions"].items()]
    start = r.get("start_check")
    if start:
        verdict = "matched" if start["matched"] else "NOT matched: the contrast carries this flag"
        lines += ["", f"Start check (printed, never decides): step-1 `mil` {start['arm']} "
                  f"{start['arm_mil']:.6f} vs {start['control']} {start['control_mil']:.6f}, "
                  f"gap {start['gap']:.2e} (tol {constants.KIM_START_TOL:.0e}) — {verdict}."]
    if r.get("motion_profile"):
        lines += ["", "## Per-step norm of the baked motion columns (what LN erases; printed)", "",
                  "| cache | videos | steps | mean | q10 | median | q90 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {n} | {m['videos']} | {m['steps']} | {m['mean']:.4f} | {m['q10']:.4f} | "
                  f"{m['median']:.4f} | {m['q90']:.4f} |" for n, m in r["motion_profile"].items()]
    if r["diagnostics"]:
        keys = ("mil", "kim_mil", "kim_nkc", "kim_w_out_norm", "kim_vel_gate", "kim_acc_gate",
                "kim_anchor_share", "kim_proto_cos")
        lines += ["", "## Training diagnostics (first → last logged batch)", "",
                  "| run | step | mil last-epoch mean | " + " | ".join(keys) + " |",
                  "| --- | --- | --- | " + " | ".join("---" for _ in keys) + " |"]
        for name, d in r["diagnostics"].items():
            mil = d.get("mil_last_epoch_mean")
            lines.append(f"| {name} | {d['last_step']} | {_num(mil)} | "
                         + " | ".join(_diag(d, k) for k in keys) + " |")
        seam_keys = [f"{k}_mean" for k in constants.KIM_SEAM_DIAG_KEYS]
        seam = {n: d for n, d in r["diagnostics"].items() if seam_keys[0] in d}
        if seam:
            lines += ["", "## Seam check (§12, mean over logged batches, never decides)",
                      "", "| run | near a join, all valid steps | near a join, anchors |",
                      "| --- | --- | --- |"]
            lines += [f"| {n} | {_num(d.get(seam_keys[0]))} | {_num(d.get(seam_keys[1]))} |"
                      for n, d in seam.items()]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    for flag, what in (("--dota", "protocol_b_eval out dir"), ("--dada", "dada_whole_eval out dir"),
                       ("--tad", "TAD dir holding s<seed>/ (core.evaluate --save-scores)")):
        parser.add_argument(flag, nargs=2, action="append", metavar=("NAME", "DIR"), help=what)
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "RUN_DIR"),
                        help="training dir with metrics.jsonl (diagnostics)")
    parser.add_argument("--motion-cache", nargs=2, action="append", metavar=("NAME", "DIR"),
                        help="A2 baked cache: per-step motion norm profile (§25, printed)")
    parser.add_argument("--control", default=constants.KIM_CONTROL,
                        help="K0, or K0k9 once §3.1 moved the host to kernel 9")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--pilot", type=Path, default=None, help="kim_pilot.json (§5.2, printed)")
    parser.add_argument("--metadata", type=Path, default=None, help="DoTA metadata_val.json")
    parser.add_argument("--split-file", type=Path, default=None, help="DoTA val_split.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
