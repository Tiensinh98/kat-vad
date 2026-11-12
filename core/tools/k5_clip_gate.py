"""K5's micro headroom and clip gates on K5 (``KIM_RUN.md`` §19 MSAD gate, §20 Nexar gate).

Pure aggregation over saved per-video curves (``core.evaluate --save-scores`` or
``nexar_eval --save-scores``) and ``tad_clip_gate score`` files; no model is loaded.

* **headroom** (§19 step 0): what a per-video offset could buy. Micro of K5's raw scores, of
  ``s + K·1[abnormal video]`` (oracle gate, ``K`` = score spread + 1, so every normal-video frame
  ranks below every abnormal-video frame while each video keeps its own order), of per-video
  min-max, and of offsets (same ``K``) by K5's own peak / mean; plus the clip-level AUC of K5's
  video statistics.
* **gates**: each gate ``c(v)`` (logistic on mean raw CLIP, ``tad_clip_gate``) fused as
  ``c(v) · minmax_v(s)``. Rows K5 raw, min-max only, gate only (``c(v)`` on every frame) and the
  fusion; macro of every fusion must equal K5's (``V2_TADG_MACRO_TOL``) or the tool raises.
  Sentence per (set, gate): micro(fusion) - micro(K5) "above / below / not separated", unless the
  gate was fitted on that set's corpus (``--in-domain GATE:SET``: printed, no sentence).
* **strata** (§20 NX8): clip AUC of each gate and of K5's peak within each value of each key of
  ``--strata SET JSON`` (``{video: {key: value}}``), values with both classes only.

Every row is printed beside the clip oracle and the length ruler: on these sets micro is mostly
clip classification (C12/C14), so a micro gain is a clip-level gain, not localization.

CLI::

    python -m core.tools.k5_clip_gate \\
        --set nexar_val <kim_gate>/eval/nexar/K5 NG=<c_nexar_val_NG.json> \\
            MS=<c_nexar_val_MS.json> \\
        --set TAD outputs/v_final/kim/eval/tad/K5/s2024 NG=<...> MS=<...> \\
        --in-domain MS:MSAD --strata nexar_val strata.json --out-dir <kim_gate>/REPORTS
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
from core.tools import tadz_readout as td
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.tad_clip_gate import fuse, micro_bootstrap_pairs, minmax
from core.tools.zs_lagovad_readout import versus_sentence

LOGGER = logging.getLogger(__name__)

Curves = dict[str, np.ndarray]
SEED = constants.V5_BOOTSTRAP_SEED
FUSED = "K5 x gate"


# ---------------------------------------------------------------------------
# pure pieces
# ---------------------------------------------------------------------------
def video_labels(labels: Curves) -> dict[str, int]:
    return {v: int(g.max() > 0) for v, g in labels.items()}


def offset(scores: Curves, values: dict[str, float]) -> Curves:
    """``s + K·value(v)``, ``K`` = score spread + 1: for a 0/1 value every 1-video frame ranks
    above every 0-video frame (the oracle); for a continuous value it is a strong offset, not a
    strict lexicographic rank."""
    spread = max(float(s.max()) for s in scores.values()) - min(
        float(s.min()) for s in scores.values())
    k = spread + 1.0
    return {v: s + k * values[v] for v, s in scores.items()}


def macro(scores: Curves, labels: Curves) -> float:
    return float(np.mean(list(td.clip_aucs(scores, labels).values())))


def headroom(scores: Curves, labels: Curves) -> dict[str, Any]:
    """Step 0: what per-video offsets do to micro; macro is unchanged by every one of them."""
    y = video_labels(labels)
    ids = sorted(labels)
    target = np.array([y[v] for v in ids])
    stats = {
        "peak": {v: float(scores[v].max()) for v in ids},
        "mean": {v: float(scores[v].mean()) for v in ids},
        "range": {v: float(np.ptp(scores[v])) for v in ids},
    }
    rulers = td.rulers(labels, None, SEED)
    return {
        "videos": len(ids),
        "normal_videos": int(len(ids) - target.sum()),
        "macro": macro(scores, labels),
        "micro_raw": td.micro(scores, labels),
        "micro_oracle_gate": td.micro(offset(scores, {v: float(y[v]) for v in ids}), labels),
        "micro_minmax": td.micro(
            {v: normalize_scores(scores[v], constants.SCORE_NORM_MINMAX) for v in ids}, labels),
        "micro_offset_peak": td.micro(offset(scores, stats["peak"]), labels),
        "micro_offset_mean": td.micro(offset(scores, stats["mean"]), labels),
        "clip_auc_of": {k: frame_auc(np.array([s[v] for v in ids]), target)
                        for k, s in stats.items()},
        "clip_oracle_micro": rulers["clip_oracle_micro"],
        "length_ruler_micro": rulers["length_ruler_micro"],
    }


def _row(scores: Curves, labels: Curves) -> dict[str, float]:
    ids = sorted(labels)
    pooled = np.concatenate([scores[v] for v in ids])
    lab = np.concatenate([labels[v] for v in ids])
    return {"micro": td.micro(scores, labels), "ap": frame_ap(pooled, lab),
            "clip_auc": td.video_auc(scores, labels)}


def fused_curves(scores: Curves, labels: Curves, gate: dict[str, float]) -> Curves:
    """``c · minmax(s)``; raises if a video has no gate score or if macro moves (R3 / G3)."""
    missing = sorted(set(labels) - set(gate))
    if missing:
        raise ValueError(f"{len(missing)} videos have no gate score, e.g. {missing[:3]}")
    fused = fuse(scores, gate)
    drift = abs(macro(fused, labels) - macro(scores, labels))
    if drift > constants.V2_TADG_MACRO_TOL:
        raise ValueError(f"macro moved by {drift:.2e} under a per-video positive rescale")
    return fused


def set_reads(scores: Curves, labels: Curves, gates: dict[str, dict[str, float]]
              ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    """Per-gate reads (K5 raw / min-max / gate only / ``c · minmax(s)``) and the cross-gate deltas.

    Every bootstrap of the set runs on one set of resamples (``micro_bootstrap_pairs``):
    each distinct curve is scored once per draw. Cross-gate keys are ``a - b`` over sorted names.
    """
    minmaxed = {v: minmax(s) for v, s in scores.items()}
    curves: dict[str, Curves] = {"K5 raw": scores, "min-max only": minmaxed}
    pairs: dict[str, tuple[str, str]] = {}
    for name, gate in gates.items():
        curves[f"fused {name}"] = fused_curves(scores, labels, gate)
        curves[f"gate only {name}"] = {v: np.full(len(s), gate[v]) for v, s in scores.items()}
        pairs[f"{name} raw"] = (f"fused {name}", "K5 raw")
        pairs[f"{name} minmax"] = (f"fused {name}", "min-max only")
        pairs[f"{name} gate only"] = (f"fused {name}", f"gate only {name}")
    names = sorted(gates)
    cross = {f"{a} - {b}": (f"fused {a}", f"fused {b}")
             for i, a in enumerate(names) for b in names[i + 1:]}
    boots = micro_bootstrap_pairs(curves, {**pairs, **{f"cross {k}": v for k, v in cross.items()}},
                                  labels, SEED)
    reads: dict[str, dict[str, Any]] = {}
    for name in gates:
        rows = {"K5 raw": scores, "min-max only": minmaxed,
                "gate only": curves[f"gate only {name}"], FUSED: curves[f"fused {name}"]}
        delta = boots[f"{name} raw"]
        reads[name] = {
            "macro": macro(scores, labels),
            "macro_drift": abs(macro(rows[FUSED], labels) - macro(scores, labels)),
            "rows": {k: _row(x, labels) for k, x in rows.items()},
            "delta_vs_raw": delta,
            "sentence": versus_sentence(delta),
            "delta_vs_minmax": boots[f"{name} minmax"],
            "delta_vs_gate_only": boots[f"{name} gate only"],
        }
    return reads, {k: boots[f"cross {k}"] for k in cross}


def gate_read(scores: Curves, labels: Curves, gate: dict[str, float]) -> dict[str, Any]:
    """``set_reads`` for one gate."""
    return set_reads(scores, labels, {"gate": gate})[0]["gate"]


def strata_read(scores: Curves, labels: Curves, gates: dict[str, dict[str, float]],
                strata: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Clip AUC of each gate and of K5's peak within each stratum value with both classes."""
    y = video_labels(labels)
    keys = sorted({k for v in labels if v in strata for k in strata[v]})
    out: dict[str, Any] = {}
    for key in keys:
        groups: dict[str, list[str]] = {}
        for v in sorted(labels):
            if v in strata and key in strata[v]:
                groups.setdefault(str(strata[v][key]), []).append(v)
        out[key] = {}
        for value, ids in sorted(groups.items()):
            target = np.array([y[v] for v in ids])
            entry: dict[str, Any] = {"videos": len(ids), "abnormal": int(target.sum())}
            if 0 < target.sum() < len(ids):
                entry["K5 peak"] = frame_auc(np.array([scores[v].max() for v in ids]), target)
                for name, g in gates.items():
                    entry[name] = frame_auc(np.array([g[v] for v in ids]), target)
            out[key][value] = entry
    return out


def read_gate_file(path: Path, scores: Curves) -> dict[str, float]:
    """``tad_clip_gate score`` output; its row count must match the scored curve (R2)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    bad = [v for v in scores if v in raw and int(raw[v]["rows"]) != len(scores[v])]
    if bad:
        raise ValueError(f"{path.name}: {len(bad)} videos with gate rows != scored rows, "
                         f"e.g. {bad[:3]}")
    return {v: float(raw[v]["c"]) for v in scores if v in raw}


def parse_set(spec: list[str]) -> tuple[str, Path, dict[str, Path]]:
    """``NAME RUN_DIR [GATE=JSON ...]``."""
    if len(spec) < 2:
        raise ValueError("--set takes NAME RUN_DIR [GATE=JSON ...]")
    gates: dict[str, Path] = {}
    for item in spec[2:]:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"gate spec {item!r} is not GATE=JSON")
        gates[name] = Path(path)
    return spec[0], Path(spec[1]), gates


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _ci(c: dict[str, float]) -> str:
    return f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _headroom_md(h: dict[str, Any]) -> list[str]:
    return [f"{h['videos']} videos ({h['normal_videos']} normal); macro {h['macro']:.4f}; "
            f"clip oracle {h['clip_oracle_micro']:.4f}; "
            f"length ruler {h['length_ruler_micro']:.4f}", "",
            "| step 0: micro under | value |", "|---|---:|",
            f"| K5 raw | {h['micro_raw']:.4f} |",
            f"| oracle per-video gate | {h['micro_oracle_gate']:.4f} |",
            f"| per-video min-max | {h['micro_minmax']:.4f} |",
            f"| offset by K5's peak | {h['micro_offset_peak']:.4f} |",
            f"| offset by K5's mean | {h['micro_offset_mean']:.4f} |", "",
            "Clip AUC of K5's own video statistics: "
            + ", ".join(f"{k} {v:.3f}" for k, v in h["clip_auc_of"].items()), ""]


def _gate_md(name: str, g: dict[str, Any], in_domain: bool) -> list[str]:
    verdict = ("in-domain for this gate: printed, no sentence" if in_domain
               else f"**{g['sentence']}**")
    lines = [f"### gate {name} — {verdict}", "",
             "| row | micro | AP | clip AUC |", "|---|---:|---:|---:|"]
    lines += [f"| {k} | {x['micro']:.4f} | {x['ap']:.4f} | {x['clip_auc']:.4f} |"
              for k, x in g["rows"].items()]
    return [*lines, "", f"- micro(K5 x gate) - micro(K5 raw): {_ci(g['delta_vs_raw'])}",
            f"- micro(K5 x gate) - micro(min-max only): {_ci(g['delta_vs_minmax'])}",
            f"- micro(K5 x gate) - micro(gate only): {_ci(g['delta_vs_gate_only'])}",
            f"- macro drift {g['macro_drift']:.1e}", ""]


def _strata_md(strata: dict[str, Any]) -> list[str]:
    lines = ["### clip AUC within strata (printed, never gating)", ""]
    for key, values in strata.items():
        for value, e in values.items():
            cells = ", ".join(f"{k} {v:.3f}" for k, v in e.items()
                              if k not in ("videos", "abnormal"))
            lines.append(f"- {key} = {value} ({e['abnormal']} / {e['videos']} abnormal): "
                         f"{cells or 'one class only'}")
    return [*lines, ""]


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# K5 micro headroom and clip gates (`KIM_RUN.md` §19 / §20)", "",
             "PRELIMINARY (n = 1: K5 s2024, one deterministic fit per gate). "
             "Macro is unchanged by construction.", ""]
    for name, r in report["sets"].items():
        lines += [f"## {name}", "", *_headroom_md(r["headroom"])]
        for gname, g in r["gates"].items():
            lines += _gate_md(gname, g, gname in r["in_domain_gates"])
        for pair, c in r.get("cross_gate", {}).items():
            lines += [f"- micro(K5 x {pair.split(' - ')[0]}) - micro(K5 x "
                      f"{pair.split(' - ')[1]}): {_ci(c)} (printed)", ""]
        if "strata" in r:
            lines += _strata_md(r["strata"])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    in_domain: dict[str, set[str]] = {}
    for item in args.in_domain or []:
        gate, sep, set_name = item.partition(":")
        if not sep:
            raise SystemExit(f"--in-domain {item!r} is not GATE:SET")
        in_domain.setdefault(set_name, set()).add(gate)
    strata_files = {s: Path(p) for s, p in (args.strata or [])}
    sets: dict[str, Any] = {}
    for spec in args.set:
        name, run_dir, gate_files = parse_set(spec)
        scores, labels = td.load_run(run_dir)
        gates = {g: read_gate_file(p, scores) for g, p in gate_files.items()}
        reads, cross = set_reads(scores, labels, gates)
        entry: dict[str, Any] = {
            "run": str(run_dir), "gate_files": {g: str(p) for g, p in gate_files.items()},
            "in_domain_gates": sorted(in_domain.get(name, set())),
            "headroom": headroom(scores, labels),
            "gates": reads,
        }
        if len(gates) > 1:
            entry["cross_gate"] = cross
        if name in strata_files:
            strata = json.loads(strata_files[name].read_text(encoding="utf-8"))
            entry["strata"] = strata_read(scores, labels, gates, strata)
        sets[name] = entry
        LOGGER.info("%s: micro raw %.4f | %s", name, entry["headroom"]["micro_raw"],
                    {g: round(x["rows"][FUSED]["micro"], 4) for g, x in entry["gates"].items()})
    report = {"sets": sets, "seed": SEED}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / constants.K5G_REPORT_STEM
    write_json_atomic(stem.with_suffix(".json"), report)
    write_text_atomic(stem.with_suffix(".md"), render_markdown(report))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--set", nargs="+", action="append", required=True,
                        metavar="ITEM", help="NAME RUN_DIR [GATE=JSON ...]")
    parser.add_argument("--in-domain", nargs="*", default=None, metavar="GATE:SET",
                        help="a gate fitted on that set's corpus (printed, no sentence)")
    parser.add_argument("--strata", nargs=2, action="append", default=None,
                        metavar=("SET", "JSON"), help="{video: {key: value}} for NX8's print")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
