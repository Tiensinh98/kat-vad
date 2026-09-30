"""MM-AU / CAP-DATA Phase 0: which CAP videos are DoTA clips or DADA-2000 sources?

Plan ``.project/plans/katvad-mmau-phase0.md`` (§3 matching rule, §4 D9 branch rule, both
fixed before any number). CAP-DATA ships no provenance column, so provenance is read from
content: every query (a DoTA clip, a DADA-2000 source) is looked up in the CAP CLIP cache.

- **Retrieval:** clip descriptor = mean of L2-normalized frames, re-normalized; the
  ``MMAU_MATCH_TOP_K`` CAP clips with the highest descriptor cosine are candidates.
- **Containment** ``kappa(q, c)`` = mean over ``q``'s frames of ``max_j cos(q_i, c_j)``:
  how much of ``q`` is found in ``c``, independent of order, frame rate and of ``c`` being
  longer than ``q``. Best match = the candidate with the highest ``kappa``; the hard null =
  ``kappa`` against the K-th ranked candidate.
- **Grade:** ``exact`` (``kappa >= MMAU_MATCH_EXACT``), ``near`` (``>= MMAU_MATCH_NEAR``),
  else ``none``. For exact/near pairs ``j = rate * i + offset`` is fitted on the frames whose
  max-cos clears ``MMAU_MATCH_NEAR`` (``rate`` = CAP frames per query frame).
  **Amendment P0b-1** (fixed after the feasibility-only group ``1-10`` read-out, before P0c):
  a ``near`` pair must also align with ``rate >= MMAU_NEAR_MIN_RATE``, else it is ``none``
  (counted as ``near_rejected``). Group ``1-10`` had 9 near pairs at kappa 0.951-0.968 with
  ``rate`` <= 0.71 (dashcam look-alikes, one CAP clip "matched" by 3 YouTube videos) and 22 at
  0.98-0.99 with ``rate`` ~1 or ~3. ``exact`` and the null flag are unchanged.
- **Branch** (only when the whole CAP cache is present): ``P`` (exact coverage of all DoTA
  and of DoTA-dev both >= ``MMAU_COVERAGE_BAR``), ``P_PRIME`` (exact or near does), ``C``
  otherwise; ``UNRELIABLE`` if more than ``MMAU_NULL_FLAG_SHARE`` of hard nulls reach
  ``MMAU_MATCH_NEAR``. A partial cache reads ``FEASIBILITY_ONLY``.

DoTA-eval stays sealed: only DoTA-dev is read through ``load_split``; coverage is printed
for all DoTA and for DoTA-dev. No label or score is read.

CLI::

    python -m core.tools.mmau_match \\
        --cap-clip-dir cache/clip/MMAU_CAP_s1_ncc --dota-clip-dir cache/clip/DoTA_s1_ncc \\
        [--dada-clip-dir cache/clip/DADA2000_orig] --out-dir outputs/v2/REPORTS/mmau_p0/all
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.v2_splits import load_split
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "mmau_p0_readout.json"
READOUT_MD = "mmau_p0_readout.md"
MATCHES_JSON = "mmau_p0_matches.json"
GRADE_EXACT, GRADE_NEAR, GRADE_NONE = "exact", "near", "none"
BRANCH_P, BRANCH_P_PRIME, BRANCH_C = "P", "P_PRIME", "C"
BRANCH_UNRELIABLE, BRANCH_PARTIAL = "UNRELIABLE", "FEASIBILITY_ONLY"
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass(frozen=True)
class Match:
    """One query's best CAP clip, its containment grade and time alignment."""

    query: str
    cap_id: str
    kappa: float
    null_kappa: float
    grade: str
    frames: int
    rate: float | None = None
    offset: float | None = None
    hits: list[str] = field(default_factory=list)  # every candidate graded exact/near
    near_rejected: bool = False  # kappa in the near band but no forward alignment (P0b-1)


def normalize_rows(rows: np.ndarray) -> np.ndarray:
    """L2-normalize each frame embedding (float32)."""
    x = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit: np.ndarray = x / np.maximum(norms, constants.V2_NORM_EPS)
    return unit


def load_normalized(cache_dir: Path, ids: list[str] | None = None) -> dict[str, np.ndarray]:
    """``{id: (L, D) unit rows}`` for every ``.npy`` (or ``ids``); empty clips are skipped."""
    paths = (
        sorted(cache_dir.glob("*.npy"))
        if ids is None
        else [cache_dir / f"{v}.npy" for v in ids]
    )
    out: dict[str, np.ndarray] = {}
    for path in paths:
        rows = np.load(path)
        if rows.ndim != 2 or len(rows) == 0:
            LOGGER.warning("Skipping %s: shape %s", path.name, rows.shape)
            continue
        out[path.stem] = normalize_rows(rows)
    if not out:
        raise ValueError(f"no usable feature files in {cache_dir}")
    LOGGER.info("Loaded %d clips from %s", len(out), cache_dir)
    return out


def descriptor(rows: np.ndarray) -> np.ndarray:
    """Clip descriptor: the re-normalized mean of unit frame rows."""
    unit: np.ndarray = normalize_rows(rows.mean(axis=0, keepdims=True))[0]
    return unit


def containment(query: np.ndarray, cap: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """``kappa``, plus each query frame's best cosine and the CAP frame it hit."""
    sims = query @ cap.T
    best = sims.max(axis=1)
    return float(best.mean()), best, sims.argmax(axis=1)


def fit_alignment(best: np.ndarray, hit: np.ndarray) -> tuple[float, float] | None:
    """``(rate, offset)`` of ``hit ~ rate * i + offset`` over frames with best >= NEAR."""
    frames = np.nonzero(best >= constants.MMAU_MATCH_NEAR)[0]
    if len(frames) < constants.MMAU_ALIGN_MIN_POINTS or np.ptp(frames) == 0:
        return None
    rate, offset = np.polyfit(frames.astype(np.float64), hit[frames].astype(np.float64), 1)
    return float(rate), float(offset)


def grade(kappa: float) -> str:
    if kappa >= constants.MMAU_MATCH_EXACT:
        return GRADE_EXACT
    return GRADE_NEAR if kappa >= constants.MMAU_MATCH_NEAR else GRADE_NONE


def grade_pair(kappa: float, best: np.ndarray, hit: np.ndarray) -> tuple[str, bool]:
    """Grade of one (query, CAP) pair and whether Amendment P0b-1 demoted it from near."""
    g = grade(kappa)
    if g != GRADE_NEAR:
        return g, False
    align = fit_alignment(best, hit)
    if align is None or align[0] < constants.MMAU_NEAR_MIN_RATE:
        return GRADE_NONE, True
    return g, False


def match_queries(
    queries: dict[str, np.ndarray],
    cap: dict[str, np.ndarray],
    top_k: int = constants.MMAU_MATCH_TOP_K,
) -> list[Match]:
    """Best CAP clip for every query (plan §3)."""
    cap_ids = sorted(cap)
    bank = np.stack([descriptor(cap[c]) for c in cap_ids])
    k = min(top_k, len(cap_ids))
    matches: list[Match] = []
    for name in sorted(queries):
        rows = queries[name]
        scores: np.ndarray = bank @ descriptor(rows)
        ranked = np.argsort(-scores, kind="stable")[:k]
        scored = [(cap_ids[i], *containment(rows, cap[cap_ids[i]])) for i in ranked]
        best_id, kappa, best, hit = max(scored, key=lambda s: s[1])
        g, rejected = grade_pair(kappa, best, hit)
        align = fit_alignment(best, hit) if g != GRADE_NONE else None
        matches.append(
            Match(
                query=name,
                cap_id=best_id,
                kappa=kappa,
                null_kappa=scored[-1][1] if k > 1 else float("nan"),
                grade=g,
                frames=len(rows),
                rate=None if align is None else align[0],
                offset=None if align is None else align[1],
                hits=sorted(
                    c for c, kap, b, h in scored if grade_pair(kap, b, h)[0] != GRADE_NONE
                ),
                near_rejected=rejected,
            )
        )
    return matches


def _quantiles(values: list[float]) -> dict[str, float] | None:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return None
    return {f"q{int(q * 100):02d}": float(np.quantile(finite, q)) for q in QUANTILES}


def summarize(matches: list[Match], ids: set[str] | None = None) -> dict[str, Any]:
    """Grade counts and coverage over ``matches`` (restricted to ``ids`` if given)."""
    rows = [m for m in matches if ids is None or m.query in ids]
    n = len(rows)
    counts = {g: sum(m.grade == g for m in rows) for g in (GRADE_EXACT, GRADE_NEAR, GRADE_NONE)}
    nulls = [m.null_kappa for m in rows if np.isfinite(m.null_kappa)]
    return {
        "n": n,
        "missing": 0 if ids is None else len(ids - {m.query for m in rows}),
        "counts": counts,
        "near_rejected": sum(m.near_rejected for m in rows),
        "exact_coverage": counts[GRADE_EXACT] / n if n else 0.0,
        "near_or_exact_coverage": (counts[GRADE_EXACT] + counts[GRADE_NEAR]) / n if n else 0.0,
        "kappa": _quantiles([m.kappa for m in rows]),
        "null_kappa": _quantiles(nulls),
        "null_at_near_share": (
            sum(v >= constants.MMAU_MATCH_NEAR for v in nulls) / len(nulls) if nulls else 0.0
        ),
        "rate": _quantiles([m.rate for m in rows if m.rate is not None]),
        "distinct_cap_clips": len({m.cap_id for m in rows if m.grade != GRADE_NONE}),
    }


def decide_branch(dota_all: dict[str, Any], dota_dev: dict[str, Any], partial: bool) -> str:
    """Plan §4: the D9 branch this read-out points to."""
    if partial:
        return BRANCH_PARTIAL
    if max(dota_all["null_at_near_share"], dota_dev["null_at_near_share"]) > (
        constants.MMAU_NULL_FLAG_SHARE
    ):
        return BRANCH_UNRELIABLE
    bar = constants.MMAU_COVERAGE_BAR
    if min(dota_all["exact_coverage"], dota_dev["exact_coverage"]) >= bar:
        return BRANCH_P
    if min(dota_all["near_or_exact_coverage"], dota_dev["near_or_exact_coverage"]) >= bar:
        return BRANCH_P_PRIME
    return BRANCH_C


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    cap = load_normalized(args.cap_clip_dir)
    dota = load_normalized(args.dota_clip_dir)
    dev = set(load_split(constants.V2_SPLIT_DOTA_DEV, args.split_dir))
    dota_matches = match_queries(dota, cap, args.top_k)
    dota_all, dota_dev = summarize(dota_matches), summarize(dota_matches, dev)
    matched_cap = {h for m in dota_matches for h in m.hits}
    per_corpus = {"dota": [asdict(m) for m in dota_matches]}
    dada_block: dict[str, Any] | None = None
    dada_cap: set[str] = set()
    if args.dada_clip_dir is not None:
        dada_matches = match_queries(load_normalized(args.dada_clip_dir), cap, args.top_k)
        dada_block = summarize(dada_matches)
        dada_cap = {h for m in dada_matches for h in m.hits}
        per_corpus["dada"] = [asdict(m) for m in dada_matches]
    partial = len(cap) < args.expected_cap
    readout = {
        "plan": ".project/plans/katvad-mmau-phase0.md §3-§4",
        "thresholds": {
            "top_k": args.top_k,
            "exact": constants.MMAU_MATCH_EXACT,
            "near": constants.MMAU_MATCH_NEAR,
            "null_flag_share": constants.MMAU_NULL_FLAG_SHARE,
            "coverage_bar": constants.MMAU_COVERAGE_BAR,
            "near_min_rate": constants.MMAU_NEAR_MIN_RATE,
        },
        "cap": {
            "clips": len(cap),
            "expected": args.expected_cap,
            "partial": partial,
            "hit_by_dota": len(matched_cap),
            "hit_by_dada": len(dada_cap),
            "clean": len(set(cap) - matched_cap - dada_cap),
            "clean_frames": int(sum(len(cap[c]) for c in set(cap) - matched_cap - dada_cap)),
        },
        "dota_all": dota_all,
        "dota_dev": dota_dev,
        "dada": dada_block,
        "branch": decide_branch(dota_all, dota_dev, partial),
    }
    return readout, per_corpus


def _fmt(q: dict[str, float] | None) -> str:
    return "—" if q is None else " / ".join(f"{v:.3f}" for v in q.values())


def render_markdown(readout: dict[str, Any]) -> str:
    th, cap = readout["thresholds"], readout["cap"]
    lines = [
        "# MM-AU / CAP-DATA Phase 0 — provenance read-out",
        "",
        f"Rule: {readout['plan']}. exact κ ≥ {th['exact']}, near κ ≥ {th['near']}, "
        f"top-K {th['top_k']} (hard null = K-th), null flag > {th['null_flag_share']:.0%}, "
        f"coverage bar {th['coverage_bar']:.0%}, near needs rate ≥ {th['near_min_rate']} "
        "(Amendment P0b-1).",
        "",
        f"CAP clips cached: **{cap['clips']} / {cap['expected']}**"
        + (" (partial → no branch)" if cap["partial"] else ""),
        "",
        "| query set | n | exact | near | near rej. | none | exact cov. | near+exact cov. "
        "| κ q05/q25/q50/q75/q95 | null κ q05…q95 | null ≥ near | rate q05…q95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    blocks = [("DoTA (all)", readout["dota_all"]), ("DoTA-dev", readout["dota_dev"])]
    if readout["dada"] is not None:
        blocks.append(("DADA-2000 sources", readout["dada"]))
    for name, b in blocks:
        c = b["counts"]
        lines.append(
            f"| {name} | {b['n']} | {c['exact']} | {c['near']} | {b['near_rejected']} "
            f"| {c['none']} "
            f"| {b['exact_coverage']:.3f} | {b['near_or_exact_coverage']:.3f} "
            f"| {_fmt(b['kappa'])} | {_fmt(b['null_kappa'])} | {b['null_at_near_share']:.3f} "
            f"| {_fmt(b['rate'])} |"
        )
    lines += [
        "",
        f"CAP clips hit by DoTA: {cap['hit_by_dota']} · by DADA: {cap['hit_by_dada']} · "
        f"**clean: {cap['clean']}** ({cap['clean_frames']} frames).",
        "",
        f"## Branch (plan §4): **{readout['branch']}**",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--cap-clip-dir", type=Path, required=True)
    parser.add_argument("--dota-clip-dir", type=Path, required=True, help="DoTA_s1_ncc")
    parser.add_argument("--dada-clip-dir", type=Path, default=None, help="DADA2000_orig sources")
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--top-k", type=int, default=constants.MMAU_MATCH_TOP_K)
    parser.add_argument("--expected-cap", type=int, default=constants.MMAU_CAP_VIDEOS)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    readout, per_corpus = run(args)
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_json_atomic(args.out_dir / MATCHES_JSON, per_corpus)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(readout))
    LOGGER.info("MM-AU P0 branch: %s -> %s", readout["branch"], args.out_dir)


if __name__ == "__main__":
    main()
