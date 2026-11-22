"""ACCIDENT vs TAD dedup (``KIM_RUN.md`` §39 D): which ACCIDENT clips may show a TAD test camera.

TAD's file names carry no source id and ACCIDENT's "TAD overlap" list refers to another TAD
(pending (dq)), so overlap is measured by content. Score(a, b) = the **max cosine over CLIP
frames** of clip ``a`` and video ``b`` (ACCIDENT ``s1[::8][::step]``, TAD ``s8[::step]``): a static
CCTV camera seen twice gives near-identical frames somewhere.

The threshold is calibrated, not guessed: **tau = the ``ACCIDENT_DEDUP_CONTROL_PERCENTILE``-th
percentile of the same-source control** (per upload with >= 2 clips, its best clip pair: the
same-camera positive control). Printed beside it: the cross-source nearest neighbour of a
seeded sample (what "a similar kind of scene" scores) and every clip's TAD top-1. A clip whose
TAD top-1 >= tau is flagged and its **whole source** is excluded. If more than
``ACCIDENT_DEDUP_MAX_EXCLUDED_SHARE`` of the clips would go, the status is STOP: tau then does
not separate a camera from a kind of scene.

Writes ``<stem>.json`` / ``<stem>.md``, ``<stem>_exclude_ids.txt`` and ``<stem>_pairs.json``
(flagged + just-below-tau pairs with the frame of each side, for the contact sheet). CLI::

    python -m core.tools.accident_dedup --acc-clip-s1-dir cache/clip/ACCIDENT_s1_ncc \\
        --acc-ids-file ids.txt --tad-clip-dir cache/clip/TAD_ncc --tad-data-dir data/TAD \\
        --out-dir .../REPORTS
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools.accident_build import source_of
from core.tools.feature_cache import read_ids_file
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
SHEET_COLUMNS = ("acc_id", "acc_frame", "tad_id", "tad_frame", "score")


def unit_rows(rows: np.ndarray) -> np.ndarray:
    """L2-normalized float32 rows."""
    rows = rows.astype(np.float32)
    out: np.ndarray = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    return out


def load_frames(cache_dir: Path, ids: list[str], stride: int, step: int) -> dict[str, np.ndarray]:
    """``{id: unit rows[::stride][::step]}``; a missing id raises (C10)."""
    missing = [v for v in ids if not (cache_dir / f"{v}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ids have no rows in {cache_dir}: {missing[:5]}")
    return {v: unit_rows(np.load(cache_dir / f"{v}.npy")[::stride][::step]) for v in ids}


class Bank:
    """Frames of many videos stacked once, with the owner of each row (for per-video maxima)."""

    def __init__(self, frames: dict[str, np.ndarray]) -> None:
        self.ids = sorted(frames)
        self.rows = np.concatenate([frames[v] for v in self.ids])
        lengths = [len(frames[v]) for v in self.ids]
        self.starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)

    def best(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per bank video: max cosine with ``query``, and the (query row, bank row) achieving it."""
        sims = query @ self.rows.T  # (q, N)
        col_arg = sims.argmax(axis=0)
        col_max = sims[col_arg, np.arange(sims.shape[1])]
        per_video = np.maximum.reduceat(col_max, self.starts)
        bank_row = np.empty(len(self.ids), dtype=np.int64)
        query_row = np.empty(len(self.ids), dtype=np.int64)
        ends = [*self.starts[1:], len(self.rows)]
        for i, (lo, hi) in enumerate(zip(self.starts, ends, strict=True)):
            j = lo + int(col_max[lo:hi].argmax())
            bank_row[i], query_row[i] = j - lo, int(col_arg[j])
        return per_video, query_row, bank_row


def same_source_scores(frames: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Per source with >= 2 clips: its best clip pair and that pair's max frame cosine.

    The best pair, not every pair: a large source is often a compilation of several cameras, and
    its cross-camera pairs would drag a same-camera control down (one value per source).
    """
    by_source: dict[str, list[str]] = defaultdict(list)
    for v in sorted(frames):
        by_source[source_of(v)].append(v)
    best = []
    for source, ids in sorted(by_source.items()):
        pairs = [(float((frames[a] @ frames[b].T).max()), a, b)
                 for i, a in enumerate(ids) for b in ids[i + 1:]]
        if pairs:
            score, a, b = max(pairs)
            best.append({"source": source, "a": a, "b": b, "score": score})
    return best


def cross_source_nn(frames: dict[str, np.ndarray], sample: int, seed: int) -> list[float]:
    """For a seeded sample of clips: the best score against a clip of ANOTHER source."""
    bank = Bank(frames)
    sources = np.array([source_of(v) for v in bank.ids])
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(bank.ids), size=min(sample, len(bank.ids)), replace=False)
    out = []
    for i in sorted(picks):
        scores, _, _ = bank.best(frames[bank.ids[i]])
        scores[sources == sources[i]] = -np.inf
        out.append(float(scores.max()))
    return out


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {f"p{q}": round(float(np.percentile(values, q)), 4) for q in QUANTILES}


def dedup(
    acc: dict[str, np.ndarray], tad: dict[str, np.ndarray],
    percentile: float = constants.ACCIDENT_DEDUP_CONTROL_PERCENTILE,
    max_share: float = constants.ACCIDENT_DEDUP_MAX_EXCLUDED_SHARE,
    cross_sample: int = constants.ACCIDENT_DEDUP_CROSS_SAMPLE,
    topk: int = constants.ACCIDENT_DEDUP_TOPK,
    near: int = constants.ACCIDENT_DEDUP_NEAR_LISTED,
    seed: int = constants.ACCIDENT_DEDUP_SEED,
) -> dict[str, Any]:
    """The §39 D read: tau from the same-source control, flagged clips, excluded sources."""
    control = same_source_scores(acc)
    if not control:
        raise ValueError("no source holds two clips: the same-camera control is empty")
    tau = float(np.percentile([p["score"] for p in control], percentile))
    bank = Bank(tad)
    matches: dict[str, list[dict[str, Any]]] = {}
    for v in sorted(acc):
        scores, q_row, t_row = bank.best(acc[v])
        order = np.argsort(-scores)[:topk]
        matches[v] = [{"tad_id": bank.ids[i], "score": round(float(scores[i]), 6),
                       "acc_row": int(q_row[i]), "tad_row": int(t_row[i])} for i in order]
    top1 = {v: m[0]["score"] for v, m in matches.items()}
    flagged = sorted(v for v, s in top1.items() if s >= tau)
    flagged_sources = {source_of(v) for v in flagged}
    excluded = sorted(v for v in acc if source_of(v) in flagged_sources)
    share = len(excluded) / len(acc)
    below = sorted((v for v in acc if top1[v] < tau), key=lambda v: -top1[v])[:near]
    return {
        "tau": round(tau, 6),
        "percentile": percentile,
        "status": (constants.ACCIDENT_DEDUP_OK if share <= max_share
                   else constants.ACCIDENT_DEDUP_STOP),
        "clips": len(acc),
        "tad_videos": len(tad),
        "control_sources": len(control),
        "control_quantiles": quantiles([p["score"] for p in control]),
        "cross_source_nn_quantiles": quantiles(cross_source_nn(acc, cross_sample, seed)),
        "tad_top1_quantiles": quantiles(list(top1.values())),
        "flagged_clips": flagged,
        "excluded_sources": sorted(flagged_sources),
        "excluded_clips": excluded,
        "excluded_share": round(share, 4),
        "max_excluded_share": max_share,
        "near_below_tau": below,
        "matches": matches,
    }


def pair_record(result: dict[str, Any], clip_id: str, kind: str, row_step: int,
                stride: int) -> dict[str, Any]:
    """A clip's best TAD match as native frame indices (ACCIDENT: CFR frame; TAD: image)."""
    m = result["matches"][clip_id][0]
    return {"kind": kind, "acc_id": clip_id, "acc_frame": m["acc_row"] * row_step * stride,
            "tad_id": m["tad_id"], "tad_frame": m["tad_row"] * row_step * stride,
            "score": m["score"]}


def sheet_pairs(result: dict[str, Any], row_step: int, stride: int) -> list[dict[str, Any]]:
    """Flagged and just-below pairs as native frame indices (ACCIDENT: CFR frame; TAD: image)."""
    return [pair_record(result, v, kind, row_step, stride)
            for kind, ids in (("flagged", result["flagged_clips"]),
                              ("near", result["near_below_tau"]))
            for v in ids]


def high_score_pairs(result: dict[str, Any], min_score: float, row_step: int,
                     stride: int) -> list[dict[str, Any]]:
    """Every clip whose best TAD match scores >= ``min_score``, highest score first.

    Independent of tau (a STOP run still has ``matches`` for every clip): the contact sheet of
    the pairs most likely to be one camera, for choosing a threshold by eye (``KIM_RUN.md`` §39).
    """
    ids = [v for v, m in result["matches"].items() if m[0]["score"] >= min_score]
    ids.sort(key=lambda v: (-result["matches"][v][0]["score"], v))
    return [pair_record(result, v, "high", row_step, stride) for v in ids]


def table_row(result: dict[str, Any], clip_id: str) -> str:
    best = result["matches"][clip_id][0]
    return f"| {clip_id} | {best['tad_id']} | {best['score']:.4f} |"


def markdown(result: dict[str, Any]) -> str:
    q = ("control_quantiles", "cross_source_nn_quantiles", "tad_top1_quantiles")
    names = ("best same-source pair (control)", "cross-source nearest neighbour",
             "TAD top-1 per ACCIDENT clip")
    keys = list(result["control_quantiles"])
    r = result
    lines = [
        "# ACCIDENT vs TAD dedup (`KIM_RUN.md` §39 D)", "",
        f"**Status: {r['status']}.** tau = p{r['percentile']:g} of the best same-source pair of "
        f"{r['control_sources']} sources = **{r['tau']:.4f}**. "
        f"{len(r['flagged_clips'])} clips flagged "
        f"(TAD top-1 >= tau) -> {len(r['excluded_sources'])} sources, "
        f"**{len(r['excluded_clips'])} / {r['clips']} clips excluded "
        f"({100 * r['excluded_share']:.1f} %, STOP above {100 * r['max_excluded_share']:.0f} %)**. "
        f"TAD videos: {r['tad_videos']}.", "",
        "Max CLIP frame cosine. Quantiles:", "",
        "| distribution | " + " | ".join(keys) + " |",
        "|---|" + "---|" * len(keys),
    ]
    for key, name in zip(q, names, strict=True):
        lines.append(f"| {name} | " + " | ".join(f"{result[key].get(k, float('nan')):.4f}"
                                                for k in keys) + " |")
    lines += ["", "## Flagged clips (excluded with their source)", "",
              "| ACCIDENT clip | TAD video | score |", "|---|---|---|"]
    lines += [table_row(r, v) for v in r["flagged_clips"]]
    lines += ["", f"## Just below tau (kept; top {len(result['near_below_tau'])})", "",
              "| ACCIDENT clip | TAD video | score |", "|---|---|---|"]
    lines += [table_row(r, v) for v in r["near_below_tau"]]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    acc_ids = sorted(read_ids_file(args.acc_ids_file))
    tad_ids = sorted(json.loads(
        (args.tad_data_dir / constants.FRAME_LABELS_TEST_FILENAME).read_text(encoding="utf-8")))
    acc = load_frames(args.acc_clip_s1_dir, acc_ids, constants.FRAME_STRIDE, args.row_step)
    tad = load_frames(args.tad_clip_dir, tad_ids, 1, args.row_step)  # TAD_ncc is already s8
    result = dedup(acc, tad, args.percentile)
    stem = args.out_dir / constants.ACCIDENT_DEDUP_STEM
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(stem.with_suffix(".json"), result)
    write_text_atomic(stem.with_suffix(".md"), markdown(result))
    write_text_atomic(args.out_dir / f"{constants.ACCIDENT_DEDUP_STEM}_exclude_ids.txt",
                      "".join(f"{v}\n" for v in result["excluded_clips"]))
    write_json_atomic(args.out_dir / f"{constants.ACCIDENT_DEDUP_STEM}_pairs.json",
                      sheet_pairs(result, args.row_step, constants.FRAME_STRIDE))
    LOGGER.info("dedup %s: tau %.4f, %d / %d clips excluded", result["status"], result["tau"],
                len(result["excluded_clips"]), result["clips"])
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--acc-clip-s1-dir", type=Path, required=True)
    parser.add_argument("--acc-ids-file", type=Path, required=True)
    parser.add_argument("--tad-clip-dir", type=Path, required=True, help="plain TAD_ncc (s8)")
    parser.add_argument("--tad-data-dir", type=Path, required=True, help="holds frame_labels_test")
    parser.add_argument("--row-step", type=int, default=constants.ACCIDENT_DEDUP_ROW_STEP)
    parser.add_argument("--percentile", type=float,
                        default=constants.ACCIDENT_DEDUP_CONTROL_PERCENTILE)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
