"""v4 Stage A: robust expert-head selection on T2-train (PREREG S1-S14).

HeadHunt-VAD's ranking with folds of source videos in place of prompts, scored on
**weak** step labels and on features **detrended** within each source, so a head that
only encodes "late in the video" or "aftermath" cannot win (proposal §7.1):

1. Step labels (S1-S2): a step is labelled only if every T2-train window covering it
   agrees (window abnormal iff ``positive_frames > 0``); the rest are not scored.
2. Representation (S3): per source and channel, the residual of a cubic in
   ``tau = (t + 0.5) / T``. The raw R2-CRN rows (S4) are ranked too, printed only.
3. Five disjoint folds of sources (S5); per fold a class-balanced sample, the same steps
   for every head (S6); four measures per head (S7): LDA separability, symmetrised
   Gaussian KL, MMD² and K-means NMI; min-max across heads, averaged (S8).
4. ``RSS = mean - 0.5 * std`` over folds, top 5 (S9); stability floor on the folds' top-5
   Jaccard (S10, gating); best block (S11); the A4-rand draw and G1a's 20 draws (S12).

Reads T2-train only (T2-val and T2-test never). Refuses to run unless every source's
I4 row check passed. Writes ``head_selection.{json,md}``; commit the JSON before G1a.

CLI::

    python -m core.tools.select_heads --t2-dir data/DADA2000_orig \\
        --heads-dir cache/v4/heads/vit_s_k710_dl_from_giant/DADA2000_orig_s8_squash \\
        --out-dir outputs/v4/REPORTS/stage_a
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import normalized_mutual_info_score

from core import constants
from core.crn.reference import reference
from core.data.v2_splits import lines_sha1
from core.tools.crn_select import t2_sources
from core.tools.e3_readout import spearman
from core.tools.extract_head_features import load_head_rows, ucos_failures
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

MEASURES = ("lda", "kl", "mmd", "nmi")
UNLABELLED = -1
DETRENDED = "detrended"
RAW = "raw"
SPLIT_TRAIN = "train"


# ---------------------------------------------------------------------------
# S1-S2: weak step labels
# ---------------------------------------------------------------------------
def train_windows(
    meta: dict[str, dict[str, Any]], sources: set[str]
) -> dict[str, list[tuple[int, int, int]]]:
    """``{source: [(start, end, label), ...]}`` of the T2-train windows of ``sources`` (S1)."""
    out: dict[str, list[tuple[int, int, int]]] = {}
    for entry in meta.values():
        source = str(entry["source"])
        if entry["split"] != SPLIT_TRAIN or source not in sources:
            continue
        label = int(int(entry["positive_frames"]) > 0)
        out.setdefault(source, []).append((int(entry["start"]), int(entry["end"]), label))
    return out


def step_labels(windows: list[tuple[int, int, int]], length: int) -> np.ndarray:
    """Per step 0/1 if every covering window agrees, else :data:`UNLABELLED` (S2)."""
    seen = np.zeros((length, 2), dtype=bool)
    for start, end, label in windows:
        if not 0 <= start < end <= length:
            raise ValueError(f"window [{start}, {end}) outside a {length}-step source")
        seen[start:end, label] = True
    labels = np.full(length, UNLABELLED, dtype=np.int64)
    labels[seen[:, 0] & ~seen[:, 1]] = 0
    labels[seen[:, 1] & ~seen[:, 0]] = 1
    return labels


# ---------------------------------------------------------------------------
# S3-S4: representations
# ---------------------------------------------------------------------------
def crn_rows(rows: np.ndarray) -> np.ndarray:
    """``(T, H, d)`` minus the source's R2 reference, per channel (S4, I6)."""
    flat = rows.reshape(len(rows), -1).astype(np.float64)
    centred: np.ndarray = (flat - reference(flat, constants.V4_CRN)).reshape(rows.shape)
    return centred


def detrend_rows(rows: np.ndarray, degree: int = constants.V4_DETREND_DEGREE) -> np.ndarray:
    """Residual of a per-channel least-squares polynomial in ``tau = (t + 0.5) / T`` (S3)."""
    length = len(rows)
    flat = rows.reshape(length, -1).astype(np.float64)
    tau = (np.arange(length, dtype=np.float64) + 0.5) / length
    design = np.stack([tau ** k for k in range(degree + 1)], axis=1)
    coef, *_ = np.linalg.lstsq(design, flat, rcond=None)
    residual: np.ndarray = (flat - design @ coef).reshape(rows.shape)
    return residual


# ---------------------------------------------------------------------------
# S7: the four measures, one head at a time
# ---------------------------------------------------------------------------
def _cov(x: np.ndarray, centred: bool = False) -> np.ndarray:
    cov: np.ndarray = LedoitWolf(assume_centered=centred).fit(x).covariance_
    return cov


def lda_separability(a: np.ndarray, b: np.ndarray) -> float:
    """``Δμᵀ S_w⁻¹ Δμ`` with a Ledoit-Wolf pooled within-class covariance (S7a)."""
    delta = b.mean(axis=0) - a.mean(axis=0)
    within = _cov(np.concatenate([a - a.mean(axis=0), b - b.mean(axis=0)]), centred=True)
    return float(delta @ np.linalg.solve(within, delta))


def sym_gaussian_kl(a: np.ndarray, b: np.ndarray) -> float:
    """``KL(N_a‖N_b) + KL(N_b‖N_a)``, Ledoit-Wolf covariances; the log-dets cancel (S7b)."""
    cov_a, cov_b = _cov(a), _cov(b)
    delta = b.mean(axis=0) - a.mean(axis=0)
    traces = np.trace(np.linalg.solve(cov_b, cov_a)) + np.trace(np.linalg.solve(cov_a, cov_b))
    mahal = delta @ (np.linalg.solve(cov_a, delta) + np.linalg.solve(cov_b, delta))
    return float(0.5 * (traces + mahal - 2 * a.shape[1]))


def mmd_rbf(a: np.ndarray, b: np.ndarray) -> float:
    """Biased MMD² with ``exp(-d²/median d²)`` over the pooled sample's pairs (S7c)."""
    pooled = np.concatenate([a, b])
    norms = (pooled ** 2).sum(axis=1)
    sq = np.maximum(norms[:, None] + norms[None, :] - 2.0 * pooled @ pooled.T, 0.0)
    upper = sq[np.triu_indices(len(pooled), k=1)]
    scale = float(np.median(upper)) if upper.size else 0.0
    kernel = np.exp(-sq / (scale if scale > 0 else 1.0))
    n = len(a)
    return float(kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2.0 * kernel[:n, n:].mean())


def kmeans_nmi(a: np.ndarray, b: np.ndarray, seed: int) -> float:
    """NMI between a 2-means partition of the pooled sample and the labels (S7d)."""
    pooled = np.concatenate([a, b])
    target = np.concatenate([np.zeros(len(a), dtype=np.int64), np.ones(len(b), dtype=np.int64)])
    clusters = KMeans(
        n_clusters=2,
        n_init=constants.V4_KMEANS_N_INIT,  # pyright: ignore[reportArgumentType]  (stub says str)
        random_state=seed,
    ).fit_predict(pooled)
    return float(normalized_mutual_info_score(target, clusters))


def head_measures(
    sample: np.ndarray, labels: np.ndarray, mmd_index: np.ndarray, seed: int
) -> np.ndarray:
    """``(heads, 4)`` raw measures, columns in :data:`MEASURES` order."""
    out = np.zeros((sample.shape[1], len(MEASURES)))
    for head in range(sample.shape[1]):
        x = sample[:, head, :].astype(np.float64)
        a, b = x[labels == 0], x[labels == 1]
        sub = x[mmd_index]
        sub_y = labels[mmd_index]
        out[head] = (
            lda_separability(a, b),
            sym_gaussian_kl(a, b),
            mmd_rbf(sub[sub_y == 0], sub[sub_y == 1]),
            kmeans_nmi(a, b, seed),
        )
    return out


# ---------------------------------------------------------------------------
# S5-S10: folds, scores, ranking, stability
# ---------------------------------------------------------------------------
def make_folds(sources: list[str], folds: int, seed: int) -> list[list[str]]:
    """Sorted sources, one seeded permutation, ``array_split`` into disjoint folds (S5)."""
    order = np.random.default_rng(seed).permutation(sorted(sources))
    return [sorted(str(v) for v in part) for part in np.array_split(order, folds)]


def fold_sample(
    rows: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    sources: list[str],
    rng: np.random.Generator,
    cap: int = constants.V4_SELECT_MAX_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced labelled steps of ``sources``, capped per class (S6)."""
    x = np.concatenate([rows[v][labels[v] != UNLABELLED] for v in sources])
    y = np.concatenate([labels[v][labels[v] != UNLABELLED] for v in sources])
    per_class = min(int((y == 0).sum()), int((y == 1).sum()), cap)
    if per_class < 2:
        raise ValueError(f"a fold has {per_class} labelled steps in its smaller class")
    keep = np.concatenate([
        np.sort(rng.choice(np.flatnonzero(y == c), size=per_class, replace=False)) for c in (0, 1)
    ])
    return x[keep], y[keep]


def mmd_subsample(labels: np.ndarray, rng: np.random.Generator,
                  cap: int = constants.V4_MMD_MAX_PER_CLASS) -> np.ndarray:
    """Seeded index of at most ``cap`` steps per class, shared by every head (S7c)."""
    return np.concatenate([
        np.sort(rng.permutation(np.flatnonzero(labels == c))[:cap]) for c in (0, 1)
    ])


def normalise(measures: np.ndarray) -> np.ndarray:
    """Min-max each measure across heads; a constant column becomes 0 (S8)."""
    low, high = measures.min(axis=0), measures.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    out: np.ndarray = np.where(high > low, (measures - low) / span, 0.0)
    return out


def robust_score(scores: np.ndarray, lam: float = constants.V4_RSS_LAMBDA) -> np.ndarray:
    """``RSS = mean - lam * std`` over folds (rows) for every head (columns) (S9)."""
    out: np.ndarray = scores.mean(axis=0) - lam * scores.std(axis=0)
    return out


def top_k(values: np.ndarray, k: int) -> list[int]:
    """Indices of the ``k`` largest values; ties broken by the lower index (S9)."""
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
    return sorted(order[:k])


def jaccard(a: list[int], b: list[int]) -> float:
    union = set(a) | set(b)
    return len(set(a) & set(b)) / len(union) if union else 1.0


def mean_pairwise_jaccard(sets: list[list[int]]) -> float:
    pairs = list(combinations(sets, 2))
    return float(np.mean([jaccard(a, b) for a, b in pairs])) if pairs else 1.0


def chance_jaccard(pool: int, k: int) -> float:
    """Expected Jaccard of two independent uniform ``k``-subsets of ``pool`` (hypergeometric)."""
    total = comb(pool, k)
    return sum(
        comb(k, i) * comb(pool - k, k - i) / total * (i / (2 * k - i)) for i in range(k + 1)
    )


def best_block(rss: np.ndarray, heads_per_block: int) -> int:
    """The block with the highest mean RSS over its heads (S11)."""
    return int(np.argmax(rss.reshape(-1, heads_per_block).mean(axis=1)))


def draw_heads(pool: list[int], k: int, draws: int, seed: int) -> list[list[int]]:
    """``draws`` seeded draws of ``k`` heads from ``pool`` (S12)."""
    rng = np.random.default_rng(seed)
    return [sorted(int(h) for h in rng.choice(pool, size=k, replace=False)) for _ in range(draws)]


def rank(
    rows: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    folds: list[list[str]],
    seed: int,
) -> dict[str, Any]:
    """Per-fold scores ``S`` (folds x heads), RSS, top-5 and the folds' top-5 sets."""
    scores = []
    for index, fold in enumerate(folds):
        rng = np.random.default_rng(seed + index)
        sample, y = fold_sample(rows, labels, fold, rng)
        raw = head_measures(sample, y, mmd_subsample(y, rng), seed)
        scores.append(normalise(raw).mean(axis=1))
        LOGGER.info("fold %d: %d sources, %d steps per class", index, len(fold), len(y) // 2)
    matrix = np.stack(scores)
    rss = robust_score(matrix)
    fold_top = [top_k(row, constants.V4_HEADS_K) for row in matrix]
    return {
        "scores": matrix,
        "rss": rss,
        "top": top_k(rss, constants.V4_HEADS_K),
        "fold_top": fold_top,
        "stability": mean_pairwise_jaccard(fold_top),
    }


def head_name(index: int, heads_per_block: int) -> str:
    return f"b{index // heads_per_block}h{index % heads_per_block}"


def assemble(
    detrended: dict[str, Any], raw: dict[str, Any], heads_per_block: int, seed: int
) -> dict[str, Any]:
    """Everything S9-S13 commit, from the two rankings."""
    pool = detrended["rss"].size
    selected = detrended["top"]
    others = [h for h in range(pool) if h not in selected]
    block = best_block(detrended["rss"], heads_per_block)
    stability = {
        "mean_fold_jaccard": detrended["stability"],
        "floor": constants.V4_STABILITY_MIN_JACCARD,
        "chance": chance_jaccard(pool, constants.V4_HEADS_K),
        "verdict": constants.V4_PASS
        if detrended["stability"] >= constants.V4_STABILITY_MIN_JACCARD else constants.V4_FAIL,
    }
    return {
        "selected": selected,
        "selected_names": [head_name(h, heads_per_block) for h in selected],
        "heads_per_block": heads_per_block,
        "stability": stability,
        "best_block": block,
        "full_heads": list(range(block * heads_per_block, (block + 1) * heads_per_block)),
        "a4_rand": draw_heads(others, constants.V4_HEADS_K, 1,
                              seed + constants.V4_A4_RAND_SEED_OFFSET)[0],
        "g1a_draws": draw_heads(others, constants.V4_HEADS_K, constants.V4_RAND_DRAWS,
                                seed + constants.V4_DRAWS_SEED_OFFSET),
        "fold_top": detrended["fold_top"],
        "rss": [float(v) for v in detrended["rss"]],
        "fold_scores": detrended["scores"].round(6).tolist(),
        "raw": {
            "top": raw["top"],
            "rss": [float(v) for v in raw["rss"]],
            "fold_top": raw["fold_top"],
            "stability": raw["stability"],
            "shared_with_selected": sorted(set(raw["top"]) & set(selected)),
            "jaccard_with_selected": jaccard(raw["top"], selected),
            "spearman_rss": spearman(detrended["rss"], raw["rss"]),
        },
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    meta = json.loads((args.t2_dir / constants.META_FILENAME).read_text(encoding="utf-8"))
    _val, train = t2_sources(meta)
    bad = ucos_failures(args.heads_dir, train)
    if bad:
        raise SystemExit(f"I4 row check failed or missing for {len(bad)} sources: "
                         f"{dict(list(bad.items())[:5])}")
    manifest = json.loads((args.heads_dir / constants.VIDEO_MANIFEST_FILENAME).read_text("utf-8"))
    heads_per_block = int(manifest["heads"])
    windows = train_windows(meta, set(train))
    rows = load_head_rows(args.heads_dir, train)
    labels = {v: step_labels(windows.get(v, []), len(rows[v])) for v in train}
    counts = np.bincount(np.concatenate(list(labels.values())) + 1, minlength=3)
    folds = make_folds(train, constants.V4_SELECT_FOLDS, args.seed)
    rankings = {}
    for mode, transform in ((DETRENDED, detrend_rows), (RAW, crn_rows)):
        reps = {v: transform(rows[v]).astype(np.float32) for v in train}
        rankings[mode] = rank(reps, labels, folds, args.seed)
        LOGGER.info("%s: top-5 %s, fold Jaccard %.3f", mode, rankings[mode]["top"],
                    rankings[mode]["stability"])
        del reps
    readout = assemble(rankings[DETRENDED], rankings[RAW], heads_per_block, args.seed)
    readout.update({
        "prereg": "core/docs/v4/PREREG.md S1-S14",
        "encoder": manifest["encoder"],
        "heads_manifest": {
            k: manifest[k] for k in ("feature", "layout", "depth", "heads", "head_dim")
        },
        "train_sources": len(train),
        "train_ids_sha1": lines_sha1(train),
        "steps": {
            "unlabelled": int(counts[0]), "normal": int(counts[1]), "abnormal": int(counts[2])
        },
        "folds": [len(f) for f in folds],
        "seed": args.seed,
        "commit": args.commit,
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / constants.V4_SELECTION_FILENAME, readout)
    write_text_atomic(args.out_dir / constants.V4_SELECTION_MD, render_markdown(readout))
    return readout


def render_markdown(r: dict[str, Any]) -> str:
    st = r["stability"]
    raw = r["raw"]
    lines = [
        "# v4 Stage A: head selection (PREREG S1-S14)",
        "",
        f"Encoder `{r['encoder']}`, {r['train_sources']} T2-train sources "
        f"(sha1 `{r['train_ids_sha1']}`), folds {r['folds']}, seed {r['seed']}, "
        f"commit `{r['commit']}`.",
        f"Steps: {r['steps']['normal']} normal / {r['steps']['abnormal']} abnormal labelled, "
        f"{r['steps']['unlabelled']} not scored (S2).",
        "",
        f"**Selected (detrended, S9):** {r['selected']} = {', '.join(r['selected_names'])}",
        "",
        f"**S10 stability:** mean fold Jaccard **{st['mean_fold_jaccard']:.3f}** vs floor "
        f"{st['floor']} (chance {st['chance']:.4f}) → **{st['verdict']}**",
        "",
        f"Best block (S11): {r['best_block']} → `ẽ_full` heads {r['full_heads']}. "
        f"A4-rand (S12): {r['a4_rand']}.",
        "",
        "| fold | top-5 (detrended) | top-5 (raw) |",
        "|---|---|---|",
        *[f"| {i} | {d} | {w} |"
          for i, (d, w) in enumerate(zip(r["fold_top"], raw["fold_top"], strict=True))],
        "",
        "**Printed only (S13):** raw top-5 "
        f"{raw['top']}; shared with selected {raw['shared_with_selected']} "
        f"(Jaccard {raw['jaccard_with_selected']:.2f}); Spearman rho of RSS "
        f"{raw['spearman_rss']:.3f}; "
        f"raw fold Jaccard {raw['stability']:.3f}.",
        "",
        "Next: **commit `head_selection.json`**, then run G1a with its sha256 (S14).",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--t2-dir", type=Path, required=True, help="T2 corpus dir with meta.json")
    parser.add_argument("--heads-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=constants.V4_SEED)
    parser.add_argument("--commit", default="UNKNOWN")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    readout = run(build_arg_parser().parse_args(argv))
    LOGGER.info("selected %s; S10 %s", readout["selected"], readout["stability"]["verdict"])


if __name__ == "__main__":
    main()
