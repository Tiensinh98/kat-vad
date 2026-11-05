"""SA²-v2 read-out: SV2 vs W0 on DADA-2000, controls C1-C4, then DoTA-CAP-dev if it has been read.

Rule ``core/docs/final/SA2V2_RUN.md`` §3-§4, written before any number. Pure aggregation, no model.
Run it after each stage: what has not been run yet is written as "pending".

CLI::

    python -m core.tools.sa2v2_readout --w0-dada outputs/zs/eval/dada/W0 \\
        --sv2-dada outputs/sv2/eval/dada/SV2 --dump outputs/sv2/eval/dada/maps_SV2.npz \\
        --run-dir outputs/sv2/SV2/s2024 --out-dir outputs/sv2/REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools import dada_whole_eval
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_transfer_readout import read_dada
from core.tools.protocol_b_eval import clip_aucs, read_clip_scores
from core.tools.sa2v2_eval import BRANCH_KEY, MEAN_KEY, TOP_KEY
from core.tools.stp_readout import ci as dota_ci
from core.tools.stp_readout import paired, sentence
from core.tools.v5_maps import SCORE_KEY
from core.tools.zs_readout import video_ci

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "sa2v2_readout"
RULE = "core/docs/final/SA2V2_RUN.md §3-§4"
PRELIMINARY = "PRELIMINARY (n = 1 seed)"
PENDING = "pending"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_series(path: Path, prefix: str) -> dict[str, np.ndarray]:
    """``{video: array}`` of one key family of a ``sa2v2_eval`` dump."""
    with np.load(path) as data:
        return {k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix)}


def c1_read(path: Path, labels: dict[str, np.ndarray]) -> dict[str, Any]:
    """C1: the trained head on ``x^AS`` vs the same head on the cell mean, per-video AUC."""
    on_pool = clip_aucs(dump_series(path, BRANCH_KEY), labels)
    on_mean = clip_aucs(dump_series(path, MEAN_KEY), labels)
    delta = video_ci(paired(on_pool, on_mean))
    return {"head_x_as": video_ci(on_pool), "head_mean": video_ci(on_mean), "delta": delta,
            "sentence": sentence(delta)}


def c3_read(path: Path, labels: dict[str, np.ndarray]) -> dict[str, Any]:
    """C3: top-3 ``q`` on abnormal minus normal rows, inside each two-class video."""
    top = dump_series(path, TOP_KEY)
    delta: dict[str, float] = {}
    abnormal, normal = [], []
    for video_id, q in top.items():
        y = labels.get(video_id)
        if y is None or len(y) != len(q) or y.min() == y.max():
            continue
        high, low = float(q[y > 0].mean()), float(q[y == 0].mean())
        abnormal.append(high)
        normal.append(low)
        delta[video_id] = high - low
    return {
        "videos": len(delta),
        "abnormal": float(np.mean(abnormal)) if abnormal else None,
        "normal": float(np.mean(normal)) if normal else None,
        "delta": video_ci(delta) if delta else None,
    }


def curve_gap(path: Path, scores: dict[str, np.ndarray]) -> float:
    """Dump's step scores vs the frame read's: the dump must describe the model that was read."""
    dumped = dump_series(path, SCORE_KEY)
    gaps = [float(np.abs(dumped[v] - s).max()) for v, s in scores.items() if v in dumped]
    if not gaps:
        raise ValueError("the dump and the frame read share no video")
    return max(gaps)


def diagnostics(run_dir: Path | None) -> dict[str, Any] | None:
    """First and last logged value of each SA²-v2 diagnostic; ``None`` before the run exists."""
    path = None if run_dir is None else run_dir / "metrics.jsonl"
    if path is None or not path.is_file():
        return None
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    out: dict[str, Any] = {}
    for key in constants.SV2_DIAG_KEYS:
        seen = [r[key] for r in lines if key in r]
        if seen:
            out[key] = {"first": seen[0], "last": seen[-1],
                        "tail": float(np.mean(seen[-len(seen) // 10 or -1 :]))}
    return out


def beyond_control(real: dict[str, Any] | None, control: dict[str, Any] | None) -> dict[str, Any]:
    """C2 / C4: did each diagnostic move further from its start than in the shuffled run?"""
    if not real or not control:
        return {}
    out = {}
    for key, r in real.items():
        c = control.get(key)
        if c is None:
            continue
        moved, moved_c = abs(r["tail"] - r["first"]), abs(c["tail"] - c["first"])
        out[key] = {"real": moved, "shuffled": moved_c, "exceeds": moved > moved_c}
    return out


def read_dada_set(args: argparse.Namespace) -> dict[str, Any]:
    w0, sv2 = read_dada(args.w0_dada), read_dada(args.sv2_dada)
    delta = video_ci(paired(sv2["items"], w0["items"]))
    scores, labels = read_clip_scores(args.sv2_dada / dada_whole_eval.CLIP_SCORES_NPZ)
    out: dict[str, Any] = {
        "w0": {k: w0[k] for k in ("macro", "micro", "clip_auc", "extra")},
        "sv2": {k: sv2[k] for k in ("macro", "micro", "clip_auc", "extra")},
        "delta": delta,
        "sentence": sentence(delta),
        "c1": c1_read(args.dump, labels),
        "c3": c3_read(args.dump, labels),
        "dump_vs_frame_read_max_gap": curve_gap(args.dump, scores),
    }
    if args.r4_dada is not None and args.r4_dada.is_dir():
        r4 = read_dada(args.r4_dada)
        out["r4_printed"] = {"macro": r4["macro"], "sv2_minus_r4": video_ci(
            paired(sv2["items"], r4["items"]))}
    return out


def read_dota_set(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.sv2_dota is None or not (args.sv2_dota / "clip_aucs.json").is_file():
        return None
    readout = _json(args.sv2_dota / "protocol_b_readout.json")
    control = _json(args.w0_dota / "protocol_b_readout.json")
    delta = dota_ci(
        paired(_json(args.sv2_dota / "clip_aucs.json"), _json(args.w0_dota / "clip_aucs.json"))
    )
    out = {"sv2_macro": readout.get("macro"), "w0_macro": control.get("macro"),
           "delta": delta, "sentence": sentence(delta)}
    if args.maps_md is not None and args.maps_md.is_file():
        out["maps_md"] = args.maps_md.read_text(encoding="utf-8")
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    real = diagnostics(args.run_dir)
    control = diagnostics(args.shuffled_run_dir)
    readout: dict[str, Any] = {
        "rule": RULE,
        "dada": read_dada_set(args),
        "diagnostics": real,
        "diagnostics_shuffled": control,
        "c2_c4": beyond_control(real, control),
        "dota": read_dota_set(args),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{READOUT_NAME}.json", readout)
    write_text_atomic(args.out_dir / f"{READOUT_NAME}.md", render_markdown(readout))
    LOGGER.info("DADA: %s -> %s", readout["dada"]["sentence"], args.out_dir)
    return readout


def _ci(c: dict[str, float] | None) -> str:
    return "—" if not c else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _num(c: dict[str, float] | None) -> str:
    return "—" if not c else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _f(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def render_markdown(r: dict[str, Any]) -> str:
    d = r["dada"]
    lines = [
        "# SA²-v2 read-out: SV2 vs W0, Nexar-trained, seed 2024",
        "",
        f"Rule: `{RULE}`, written before any number. {PRELIMINARY}: adopts nothing.",
        "",
        "## DADA-2000 (370 sources, whole video, s8)",
        "",
        "| Row | Macro | Micro | Clip AUC | Middle ruler |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in ("w0", "sv2"):
        row = d[name]
        lines.append(f"| {name.upper()} | {_num(row['macro'])} | {_f(row['micro'])} | "
                     f"{_f(row['clip_auc'])} | {_f(row['extra'].get('middle_ruler'))} |")
    verdict = f"{_ci(d['delta'])} → {d['sentence']}, {PRELIMINARY}"
    lines += ["", f"**SV2 - W0 (per video): {verdict}.**", ""]
    if "r4_printed" in d:
        lines += [f"Printed: first SA² macro {_num(d['r4_printed']['macro'])}; SV2 - R4 "
                  f"{_ci(d['r4_printed']['sv2_minus_r4'])}.", ""]
    c1, c3 = d["c1"], d["c3"]
    lines += [
        f"Check: dump vs frame read, max |Δ score| = {d['dump_vs_frame_read_max_gap']:.2e}.",
        "",
        "## Controls",
        "",
        f"* **C1** head on `x^AS` {_num(c1['head_x_as'])} vs same head on the cell mean "
        f"{_num(c1['head_mean'])}; Δ {_ci(c1['delta'])} → {c1['sentence']} (on DADA, rule D5).",
        f"* **C3** top-3 q on abnormal {_f(c3['abnormal'])} vs normal rows {_f(c3['normal'])}; "
        f"Δ {_ci(c3['delta'])} over {c3['videos']} videos.",
        "",
        "### C2 / C4: training diagnostics (first → mean of the last tenth)",
        "",
        "| Diagnostic | SV2 | Shuffled | Moved further than shuffled? |",
        "| --- | --- | --- | --- |",
    ]
    real, control = r["diagnostics"] or {}, r["diagnostics_shuffled"] or {}
    for key in constants.SV2_DIAG_KEYS:
        a, b = real.get(key), control.get(key)
        cell = "—" if a is None else f"{a['first']:.4f} → {a['tail']:.4f}"
        shuffled = PENDING if not control else (
            "—" if b is None else f"{b['first']:.4f} → {b['tail']:.4f}")
        moved = r["c2_c4"].get(key, {}).get("exceeds", PENDING if not control else "—")
        lines.append(f"| {key} | {cell} | {shuffled} | {moved} |")
    lines += ["", "## DoTA-CAP-dev", ""]
    dota = r["dota"]
    if dota is None:
        lines.append(f"{PENDING}: runs after the DADA read.")
    else:
        lines += [
            f"Macro: SV2 {_num(dota['sv2_macro'])}, W0 {_num(dota['w0_macro'])}.",
            "",
            f"**SV2 - W0 (per clip): {_ci(dota['delta'])} → {dota['sentence']}, {PRELIMINARY}.**",
            "",
            dota.get("maps_md", f"Maps: {PENDING}."),
        ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--w0-dada", type=Path, required=True)
    parser.add_argument("--sv2-dada", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True, help="sa2v2_eval dump on DADA")
    parser.add_argument("--run-dir", type=Path, required=True, help="SV2's run (metrics.jsonl)")
    parser.add_argument("--shuffled-run-dir", type=Path, default=None, help="C2's run, once it ran")
    parser.add_argument("--r4-dada", type=Path, default=None, help="printed only")
    parser.add_argument("--w0-dota", type=Path, default=None)
    parser.add_argument("--sv2-dota", type=Path, default=None)
    parser.add_argument("--maps-md", type=Path, default=None, help="v5_maps read's maps_readout.md")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
