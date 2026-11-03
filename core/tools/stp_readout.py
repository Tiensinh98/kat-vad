"""LaGoVAD-STP read-out on DoTA-CAP-dev (protocol B): every arm against R1, and prefix ON - OFF.

Rule (``core/docs/final/STP_BUILD.md`` §3, written before any STP number):

* **Endpoint** per arm: DoTA-CAP-dev protocol-B macro AUC, the per-clip Δ (arm - R1) over the
  two-class clips both rows scored, mean with a 95 % source-video cluster bootstrap.
* **Prefix contrast**: R6 - R4 the same way (the only difference is the text prefix).
* **Sentence**: "gain shown" iff the bootstrap low > 0 **and** the mean ≥ ``V5_MIN_GAIN``;
  "cost shown" iff the high < 0; else "no difference shown".
* With fewer seeds than ``STP_SEEDS`` every sentence is labelled **PRELIMINARY (n = k seeds)**:
  quotable only with that label, it adopts nothing.
* Printed, never deciding: Δ by ego-involved / ego-uninvolved clip (proposal H1), the t/N
  position ruler, v2's cubic position prior ``p_T2`` on CAP-dev, and the training diagnostics
  from ``metrics.jsonl`` (SA² entropy / log K at the first and last step, ``||W_q||``, prompt
  norm, ``rho_u``). An entropy that never leaves 1.0 means the attention stayed at
  "selection + average" -- the arm still runs, it is reported as such (§3.2.4).

CLI::

    python -m core.tools.stp_readout --control R1 outputs/v5/v5_stage2/eval/dota/dota_cap_dev/R1 \\
        --arm R4 outputs/stp/eval/dota/R4 --arm R6 outputs/stp/eval/dota/R6 \\
        --runs-root outputs/stp --seeds 2024 --metadata data/DoTA/metadata_val.json \\
        --split-file data/DoTA/val_split.txt --out-dir outputs/stp/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_splits import dota_group
from core.metrics import cluster_bootstrap_ci
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import CLIP_AUCS_JSON, READOUT_JSON

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "stp_readout"
DEFAULT_TITLE = "LaGoVAD-STP read-out"
DEFAULT_RULE = "core/docs/final/STP_BUILD.md §3"
DEFAULT_CONTROL_NOTE = "(v5 Stage 1 motion-stream host)"
PREFIX_OFF, PREFIX_ON = "R4", "R6"
DIAG_KEYS = ("stp_entropy", "stp_wq_norm", "prompt_norm", "motion_share", "w_u_norm", "mil")
GAIN, COST, NONE = "gain shown", "cost shown", "no difference shown"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ci(values: dict[str, float]) -> dict[str, float] | None:
    groups = {v: dota_group(v) for v in values}
    return cluster_bootstrap_ci(
        values, groups, constants.V2_E1_BOOTSTRAP, constants.V5_BOOTSTRAP_SEED, constants.V2_E1_CI
    )


def sentence(c: dict[str, float] | None) -> str:
    if c is None:
        return NONE
    if c["low"] > 0 and c["mean"] >= constants.V5_MIN_GAIN:
        return GAIN
    return COST if c["high"] < 0 else NONE


def paired(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """Per-clip ``a - b`` over the clips both rows scored."""
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("the two rows share no scored clip")
    return {v: a[v] - b[v] for v in common}


def contrast(
    a: dict[str, float], b: dict[str, float], ego: dict[str, bool]
) -> dict[str, Any]:
    delta = paired(a, b)
    out: dict[str, Any] = {"all": ci(delta), "sentence": sentence(ci(delta))}
    for name, flag in (("ego", True), ("non_ego", False)):
        sub = {v: d for v, d in delta.items() if ego.get(v) is flag}
        out[name] = ci(sub)
    return out


def run_diagnostics(run_dir: Path) -> dict[str, Any] | None:
    """First and last logged value of each diagnostic in ``metrics.jsonl``."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return None
    out: dict[str, Any] = {"last_step": lines[-1].get("global_step")}
    for key in DIAG_KEYS:
        seen = [r[key] for r in lines if key in r]
        if seen:
            out[key] = {"first": seen[0], "last": seen[-1]}
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    control_name, control_dir = args.control
    rows = {control_name: Path(control_dir), **{n: Path(d) for n, d in args.arm}}
    aucs = {name: _json(d / CLIP_AUCS_JSON) for name, d in rows.items()}
    readouts = {name: _json(d / READOUT_JSON) for name, d in rows.items()}
    records = parse_metadata(args.metadata, read_split_ids(args.split_file))
    ego = {r.video_id: r.is_ego for r in records}
    n_seeds = len(args.seeds)
    preliminary = n_seeds < len(constants.STP_SEEDS)

    result: dict[str, Any] = {
        "rule": args.rule,
        "title": args.title,
        "control_note": args.control_note,
        "set": "DoTA-CAP (n/1397) dev, protocol B",
        "control": control_name,
        "seeds": args.seeds,
        "preliminary": preliminary,
        "label": f"PRELIMINARY (n = {n_seeds} seed{'s' if n_seeds > 1 else ''})"
        if preliminary else f"n = {n_seeds} seeds",
        "rows": {
            name: {
                "macro": ci(aucs[name]),
                "per_run_macro": readouts[name].get("per_run_macro"),
                "position_ruler": readouts[name].get("position_ruler"),
                "arm": readouts[name].get("arm"),
                "global_steps": readouts[name].get("global_steps"),
            }
            for name in rows
        },
        "contrasts": {},
        "p_T2_cap_dev": constants.V5_P_T2_CAP_DEV,
        "diagnostics": {},
    }
    for name in rows:
        if name == control_name:
            continue
        result["contrasts"][f"{name} - {control_name}"] = contrast(
            aucs[name], aucs[control_name], ego
        )
    if PREFIX_OFF in aucs and PREFIX_ON in aucs:
        result["contrasts"][f"{PREFIX_ON} - {PREFIX_OFF} (prefix ON - OFF)"] = contrast(
            aucs[PREFIX_ON], aucs[PREFIX_OFF], ego
        )
    if args.runs_root is not None:
        for name in rows:
            for seed in args.seeds:
                diag = run_diagnostics(args.runs_root / name / f"s{seed}")
                if diag is not None:
                    result["diagnostics"][f"{name}/s{seed}"] = diag
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{args.name}.json", result)
    write_text_atomic(args.out_dir / f"{args.name}.md", render_markdown(result))
    return result


def _ci(c: dict[str, float] | None) -> str:
    if c is None:
        return "—"
    return f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}] (n={c['clips']})"


def _num(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _diag(d: dict[str, Any], key: str) -> str:
    v = d.get(key)
    return "—" if v is None else f"{v['first']:.4f} → {v['last']:.4f}"


def render_markdown(r: dict[str, Any]) -> str:
    lines = [
        f"# {r.get('title', DEFAULT_TITLE)} — {r['set']} — **{r['label']}**",
        "",
        f"Rule: `{r['rule']}`. Control = **{r['control']}** "
        f"{r.get('control_note', DEFAULT_CONTROL_NOTE)}. "
        f"Seeds: {', '.join(map(str, r['seeds']))}.",
        "",
        "| row | macro [95 % cluster CI] | per-run macro | t/N ruler | arm |",
        "|---|---|---|---|---|",
    ]
    for name, row in r["rows"].items():
        lines.append(
            f"| {name} | {_num(row['macro'])} | {row['per_run_macro']} | "
            f"{_num(row['position_ruler'])} | `{row['arm']}` |"
        )
    lines += [
        "",
        f"Printed: v2's cubic position prior `p_T2` on CAP-dev = {r['p_T2_cap_dev']:.4f}.",
        "",
        "| contrast | Δ all | sentence | Δ ego | Δ non-ego |",
        "|---|---|---|---|---|",
    ]
    for name, c in r["contrasts"].items():
        tag = f"{c['sentence']} — {r['label']}" if r["preliminary"] else c["sentence"]
        lines.append(f"| {name} | {_ci(c['all'])} | {tag} | {_ci(c['ego'])} | "
                     f"{_ci(c['non_ego'])} |")
    if r["diagnostics"]:
        lines += [
            "",
            "Training diagnostics (first → last logged batch):",
            "",
            "| run | step | SA² entropy / log K | ‖W_q‖ | prompt norm | rho_u | ‖W_u‖ | mil |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, d in r["diagnostics"].items():
            lines.append(
                f"| {name} | {d['last_step']} | {_diag(d, 'stp_entropy')} | "
                f"{_diag(d, 'stp_wq_norm')} | {_diag(d, 'prompt_norm')} | "
                f"{_diag(d, 'motion_share')} | {_diag(d, 'w_u_norm')} | {_diag(d, 'mil')} |"
            )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--control", nargs=2, metavar=("NAME", "DIR"), required=True,
                        help="the control's protocol_b_eval out dir (R1)")
    parser.add_argument("--arm", nargs=2, metavar=("NAME", "DIR"), action="append",
                        required=True, help="an STP arm's protocol_b_eval out dir")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--runs-root", type=Path, default=None,
                        help="<arm>/s<seed>/metrics.jsonl for the training diagnostics")
    parser.add_argument("--metadata", type=Path, required=True, help="DoTA metadata_val.json")
    parser.add_argument("--split-file", type=Path, required=True, help="DoTA val_split.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default=READOUT_NAME, help="output file stem")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--rule", default=DEFAULT_RULE, help="the document that holds the rule")
    parser.add_argument("--control-note", default=DEFAULT_CONTROL_NOTE)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run(build_arg_parser().parse_args(argv))
    LOGGER.info("%s", {k: v["sentence"] for k, v in result["contrasts"].items()})


if __name__ == "__main__":
    main()
