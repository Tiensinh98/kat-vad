"""v4 probe gates (PREREG §3). Stage A: ``g1a`` (T2-train → T2-val). G1b / G1c come with Stage B.

G1a fits the ``core.eda`` logistic probe on every step of the T2-train sources (frame labels
from the annotation span at s8, K's convention: the probe measures feature content, it is
not the model) and reads the per-source macro on T2-val. Features are CRN'd per source
(I6); ``p`` is K-pos's cubic position ruler.

* Decision sets (G1a.1): ``[ẽ;p]`` (selected heads), ``[ẽ_full;p]``, the 20 ``[ẽ_rand;p]``,
  ``[ũ;p]``, ``p``.
* **PASS iff** (G1a.2) S10's stability holds, macro ``[ẽ;p]`` > macro ``[ẽ_full;p]`` and >
  the median draw's macro (point estimates). Paired Δ with cluster CIs are printed and a
  margin below the CI half-width is flagged, not failed (G1a.3).
* Printed only (G1a.4): the raw selection, ``[x̃;ũ;ẽ;p]`` - ``[x̃;ũ;p]``, each head's
  ``[ẽ_k;p]`` - ``p``.

The selection must be committed first (S14): ``--selection-sha256`` must equal the file's.

CLI::

    python -m core.tools.v4_gate g1a --t2-dir data/DADA2000_orig \\
        --annotation <DADA>/dada标注.xlsx --census 'data/DADA2000_orig/counts/*.json' \\
        --clip-dir cache/clip/DADA2000_orig \\
        --video-dir cache/video/vit_s_k710_dl_from_giant/DADA2000_orig_s8_squash \\
        --heads-dir cache/v4/heads/vit_s_k710_dl_from_giant/DADA2000_orig_s8_squash \\
        --selection outputs/v4/REPORTS/stage_a/head_selection.json --selection-sha256 <sha> \\
        --out-dir outputs/v4/REPORTS/stage_a
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.crn.reference import reference
from core.data.dada_origin import load_counts
from core.data.v2_splits import lines_sha1
from core.metrics import cluster_bootstrap_ci
from core.tools.crn_select import t2_sources
from core.tools.e2d_probe import transfer_aucs
from core.tools.extract_head_features import load_head_rows
from core.tools.kill_switch_probe import (
    load_features,
    position_features,
    source_labels,
    write_json_atomic,
    write_text_atomic,
)
from core.tools.select_heads import crn_rows, head_name

LOGGER = logging.getLogger(__name__)

SUBCOMMAND_G1A = "g1a"
SEL, FULL, U, P, RAW = "sel_p", "full_p", "u_p", "p", "raw_sel_p"
BASE_XU, WITH_XUE = "xu_p", "xue_p"
DRAW = "rand_p_{:02d}"


@dataclass(frozen=True)
class Streams:
    """CRN'd rows per source: CLIP ``x``, pooled ``u``, all heads ``(T, H, d)``, labels."""

    x: dict[str, np.ndarray]
    u: dict[str, np.ndarray]
    heads: dict[str, np.ndarray]
    y: dict[str, np.ndarray]

    def two_class(self, ids: list[str]) -> list[str]:
        return [v for v in ids if 0 < int(self.y[v].sum()) < len(self.y[v])]


Builder = Callable[[Streams, str], np.ndarray]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_selection(path: Path, expected_sha256: str, train: list[str]) -> dict[str, Any]:
    """S14: the committed file, for these T2-train sources; otherwise refuse."""
    found = sha256_file(path)
    if found != expected_sha256:
        raise SystemExit(f"{path} sha256 {found} != --selection-sha256 {expected_sha256} (S14: "
                         "commit the selection, then paste its sha256)")
    selection: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if selection["train_ids_sha1"] != lines_sha1(train):
        raise SystemExit("the selection was made on different T2-train sources")
    return selection


def crn(rows: np.ndarray) -> np.ndarray:
    centred: np.ndarray = rows - reference(rows, constants.V4_CRN)
    return centred


def heads_of(streams: Streams, v: str, heads: list[int]) -> np.ndarray:
    return streams.heads[v][:, heads, :].reshape(len(streams.heads[v]), -1)


def feature_sets(selection: dict[str, Any]) -> dict[str, Builder]:
    """Every probed representation, as a builder of one source's rows."""
    sel, full, raw = selection["selected"], selection["full_heads"], selection["raw"]["top"]

    def with_p(*parts: Builder) -> Builder:
        def build(s: Streams, v: str) -> np.ndarray:
            blocks = [part(s, v) for part in parts] + [position_features(len(s.x[v]))]
            return np.concatenate(blocks, axis=1)
        return build

    def h(ids: list[int]) -> Builder:
        return lambda s, v: heads_of(s, v, ids)

    def x_(s: Streams, v: str) -> np.ndarray:
        return s.x[v]

    def u_(s: Streams, v: str) -> np.ndarray:
        return s.u[v]

    sets: dict[str, Builder] = {
        SEL: with_p(h(sel)),
        FULL: with_p(h(full)),
        U: with_p(u_),
        P: with_p(),
        RAW: with_p(h(raw)),
        BASE_XU: with_p(x_, u_),
        WITH_XUE: with_p(x_, u_, h(sel)),
    }
    for j, draw in enumerate(selection["g1a_draws"]):
        sets[DRAW.format(j)] = with_p(h(draw))
    for k in sel:
        sets[f"head_{k:02d}_p"] = with_p(h([k]))
    return sets


def probe(
    build: Builder,
    streams: Streams,
    train: list[str],
    val: list[str],
    seed: int,
) -> dict[str, float]:
    """Per-source T2-val AUC of the probe fitted on every T2-train step (G0)."""
    return transfer_aucs(
        {v: build(streams, v) for v in train}, streams.y,
        {v: build(streams, v) for v in val}, streams.y, val, seed,
    )


def macro(values: dict[str, float]) -> float:
    return float(np.mean(list(values.values())))


def ci(values: dict[str, float], resamples: int, seed: int) -> dict[str, float] | None:
    return cluster_bootstrap_ci(values, {v: v for v in values}, resamples, seed, constants.V4_CI)


def paired(a: dict[str, float], b: dict[str, float], resamples: int, seed: int) -> dict[str, Any]:
    block = ci({v: a[v] - b[v] for v in a}, resamples, seed)
    if block is None:
        raise ValueError("no paired sources")
    return block


def median_draw(draw_macros: dict[str, float]) -> str:
    """The draw whose macro is the (lower) median of the 20 (G1a.2)."""
    ordered = sorted(draw_macros, key=lambda k: (draw_macros[k], k))
    return ordered[(len(ordered) - 1) // 2]


def decide_g1a(
    stability_verdict: str, sel: float, full: float, draws: dict[str, float]
) -> dict[str, Any]:
    """G1a.2, mechanically: stability, beats the full block, beats the median draw."""
    median_name = median_draw(draws)
    checks = {
        "stability": stability_verdict == constants.V4_PASS,
        "beats_full_block": sel > full,
        "beats_median_draw": sel > draws[median_name],
    }
    return {
        "checks": checks,
        "median_draw": median_name,
        "median_draw_macro": draws[median_name],
        "verdict": constants.V4_PASS if all(checks.values()) else constants.V4_FAIL,
    }


def flag(delta: dict[str, Any]) -> bool:
    """G1a.3: the margin is smaller than the paired CI's half-width."""
    return bool(abs(delta["mean"]) < (delta["high"] - delta["low"]) / 2)


def load_streams(args: argparse.Namespace, ids: list[str]) -> Streams:
    census = load_counts([str(p) for p in args.census])
    y, _types = source_labels(args.annotation, census, ids, constants.FRAME_STRIDE)
    x = load_features(args.clip_dir, ids)
    u = load_features(args.video_dir, ids)
    heads = load_head_rows(args.heads_dir, ids)
    bad = [v for v in ids if not len(y[v]) == len(x[v]) == len(u[v]) == len(heads[v])]
    if bad:
        raise ValueError(f"{len(bad)} sources misaligned (labels / CLIP / u / heads): {bad[:5]}")
    return Streams(
        {v: crn(x[v]) for v in ids},
        {v: crn(u[v]) for v in ids},
        {v: crn_rows(heads[v]).astype(np.float32) for v in ids},
        y,
    )


def run_g1a(args: argparse.Namespace) -> dict[str, Any]:
    meta = json.loads((args.t2_dir / constants.META_FILENAME).read_text(encoding="utf-8"))
    val, train = t2_sources(meta)
    selection = check_selection(args.selection, args.selection_sha256, train)
    streams = load_streams(args, sorted(set(train) | set(val)))
    val_ids = streams.two_class(val)
    aucs = {
        name: probe(build, streams, train, val_ids, args.seed)
        for name, build in feature_sets(selection).items()
    }
    for name, values in aucs.items():
        LOGGER.info("G1a %-12s macro %.4f", name, macro(values))
    draws = {k: macro(v) for k, v in aucs.items() if k.startswith("rand_p_")}
    decision = decide_g1a(
        selection["stability"]["verdict"], macro(aucs[SEL]), macro(aucs[FULL]), draws
    )
    r, s = args.resamples, args.seed
    vs_full = paired(aucs[SEL], aucs[FULL], r, s)
    vs_draw = paired(aucs[SEL], aucs[decision["median_draw"]], r, s)
    readout = {
        "prereg": "core/docs/v4/PREREG.md G1a.1-G1a.5",
        "selection_sha256": args.selection_sha256,
        "selected": selection["selected"],
        "selected_names": selection["selected_names"],
        "stability": selection["stability"],
        "val_sources": len(val_ids),
        "train_sources": len(train),
        "macro": {name: ci(values, r, s) for name, values in aucs.items()},
        "decision": decision,
        "delta_vs_full": {**vs_full, "flag": flag(vs_full)},
        "delta_vs_median_draw": {**vs_draw, "flag": flag(vs_draw)},
        "printed": {
            "raw_selection_minus_selected": paired(aucs[RAW], aucs[SEL], r, s),
            "xue_minus_xu": paired(aucs[WITH_XUE], aucs[BASE_XU], r, s),
            "per_head_minus_p": {
                head_name(k, int(selection["heads_per_block"])):
                    paired(aucs[f"head_{k:02d}_p"], aucs[P], r, s)
                for k in selection["selected"]
            },
        },
        "seed": s,
        "commit": args.commit,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / constants.V4_G1A_JSON, readout)
    write_text_atomic(args.out_dir / constants.V4_G1A_MD, render_g1a(readout))
    return readout


def _fmt(block: dict[str, Any] | None) -> str:
    if block is None:
        return "-"
    return f"{block['mean']:+.4f} [{block['low']:+.4f}, {block['high']:+.4f}]"


def _m(block: dict[str, Any] | None) -> str:
    if block is None:
        return "-"
    return f"{block['mean']:.4f} [{block['low']:.4f}, {block['high']:.4f}]"


def render_g1a(r: dict[str, Any]) -> str:
    d = r["decision"]
    m = r["macro"]
    rows = [(SEL, "`[ẽ;p]` selected"), (FULL, "`[ẽ_full;p]`"), (U, "`[ũ;p]`"), (P, "`p` alone"),
            (d["median_draw"], "`[ẽ_rand;p]` median draw"), (RAW, "`[ẽ_raw;p]` (printed)"),
            (BASE_XU, "`[x̃;ũ;p]` (printed)"), (WITH_XUE, "`[x̃;ũ;ẽ;p]` (printed)")]
    lines = [
        "# v4 G1a: T2-train → T2-val (PREREG G1a)",
        "",
        f"Selection sha256 `{r['selection_sha256']}`; selected {r['selected']} = "
        f"{', '.join(r['selected_names'])}; {r['train_sources']} train → {r['val_sources']} "
        f"two-class T2-val sources; commit `{r['commit']}`.",
        "",
        f"## Verdict: **G1a {d['verdict']}**",
        "",
        "| check (G1a.2) | result |",
        "|---|---|",
        *[f"| {k} | {'yes' if v else '**no**'} |" for k, v in d["checks"].items()],
        "",
        "| set | macro [95 % cluster CI] |",
        "|---|---|",
        *[f"| {label} | {_m(m[name])} |" for name, label in rows],
        "",
        "| paired Δ (G1a.3) | Δ [CI] | flagged (margin < half-width) |",
        "|---|---|---|",
        f"| selected - full block | {_fmt(r['delta_vs_full'])} | {r['delta_vs_full']['flag']} |",
        f"| selected - median draw | {_fmt(r['delta_vs_median_draw'])} | "
        f"{r['delta_vs_median_draw']['flag']} |",
        "",
        "**Printed only (G1a.4):**",
        "",
        f"- raw selection - selected: {_fmt(r['printed']['raw_selection_minus_selected'])}",
        f"- `[x̃;ũ;ẽ;p]` - `[x̃;ũ;p]`: {_fmt(r['printed']['xue_minus_xu'])}",
        *[f"- `[ẽ_{k};p]` - `p`: {_fmt(v)}" for k, v in r["printed"]["per_head_minus_p"].items()],
        "",
        "PASS → Stage B (CAP-dev + DF8 windows, G1b, G1c). FAIL → stop V4 at Stage A (G1a.5).",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g1a = sub.add_parser(SUBCOMMAND_G1A, help="Stage A probe gate on T2")
    g1a.add_argument("--t2-dir", type=Path, required=True)
    g1a.add_argument("--annotation", type=Path, required=True)
    g1a.add_argument("--census", nargs="+", required=True)
    g1a.add_argument("--clip-dir", type=Path, required=True)
    g1a.add_argument("--video-dir", type=Path, required=True)
    g1a.add_argument("--heads-dir", type=Path, required=True)
    g1a.add_argument("--selection", type=Path, required=True)
    g1a.add_argument("--selection-sha256", required=True)
    g1a.add_argument("--out-dir", type=Path, required=True)
    g1a.add_argument("--seed", type=int, default=constants.V4_SEED)
    g1a.add_argument("--resamples", type=int, default=constants.V4_BOOTSTRAP)
    g1a.add_argument("--commit", default="UNKNOWN")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    readout = run_g1a(args)
    LOGGER.info("G1a %s", readout["decision"]["verdict"])


if __name__ == "__main__":
    main()
