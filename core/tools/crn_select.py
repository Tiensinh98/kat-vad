"""v2 P2 -- E0 rate audit (record only), E2(a) share histograms, E2(b) CRN reference
choice, E2(c) transfer-probe veto. Numpy on the stride-8 CLIP caches; no training.

Authority: proposal ``core/docs/v2/KAT_VAD_PROPOSAL_v2.md`` §4.2 (the choice rule,
steps 1-6) and §10.2; addendum ``PREREG_ADDENDUM.md`` D2 (stride 3 dropped by D8),
D3 (DoTA intervals resample source videos), §1 (the ``>70`` bin is flagged, not
silently accepted).

Corpora
    * **DoTA-dev** -- the frozen 50 % (``core/splits/v2``), one clip = one unit. Either the
      s8 cache + s8 labels (P2), or ``--dota-s1-dir --dota-stride N``: ``s1[::N]`` with
      labels rounded from the annotation onto the s1 length (addendum D11, after E1 = B).
    * **T2-val** -- the frozen 219 T2 **source videos**, whole, s8, labels from the
      annotation span (Gate D0's convention). The unit is the source because CRN's
      reference is computed over the source video in training (proposal §4.2).
    * **T2-train** -- the remaining T2-train sources; only the E2(c) probe is fitted on them.

E2(b), per reference R1-R4 (``core.crn.reference``):
    ``d_t = ||x_t - mu_t^ref||``; ``f`` = cubic in ``t/T`` fitted on the **normal**
    steps of T2-val, held at its value outside the 5th-95th percentile of their
    ``t/T``; ``r_t = d_t - f(t/T)``. Read: macro AUC of ``r`` (and raw ``d``, reported)
    overall and per share bin on both corpora, plus the position-stratified AUC.
    The position ruler ``t/T`` is printed beside every row.
Decision (step 4-5): best DoTA-dev overall of the decision metric (``r`` macro, or the
stratified AUC under the coverage fallback) among references that stay >= 0.5 in both
``>50 %`` bins and whose stratified AUC is >= 0.5; none -> ``CRN_DROPPED``.
Veto (step 6): transfer probe T2-train -> DoTA-dev on ``x - mu^ref`` vs raw ``x``;
the paired Δ's interval entirely below 0 drops CRN.

CLI::

    python -m core.tools.crn_select \\
        --dota-labels-dir data/DoTA/labels_s8 --dota-clip-dir cache/clip/DoTA_s8_ncc \\
        --t2-meta outputs/EDA/DADA2000_orig_T2_w20s8/meta.json \\
        --annotation data/DADA/dada标注.xlsx \\
        --census 'outputs/EDA/DADA2000_orig_T2/counts/*.json' \\
        --dada-clip-dir cache/clip/DADA2000_orig --out-dir outputs/v2/REPORTS/v2_E2abc
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.crn.reference import deviation, reference
from core.data.dada_origin import load_counts
from core.data.dota import parse_metadata, read_split_ids, resized_frame_labels
from core.data.v2_splits import dota_group, load_split, share_bin
from core.eda.features import transfer_scores
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools.freeze_splits import t2_pool_from_meta
from core.tools.kill_switch_probe import source_labels, write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "e2_readout.json"
READOUT_MD = "e2_readout.md"
DOTA_META = "meta.json"
DOTA_FRAME_LABELS = "frame_labels_test.json"
LAST_FIFTH = 0.8
HIGH_BINS = ("50-70", ">70")
FLAG_BIN = ">70"
METRIC_R = "r"
METRIC_STRAT = "stratified"
VERDICT_DROPPED = "CRN_DROPPED"
VERDICT_VETOED = "CRN_VETOED"


@dataclass(frozen=True)
class Corpus:
    """Per-clip features, s8 labels, accident share and bootstrap cluster."""

    name: str
    x: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    share: dict[str, float]
    group: dict[str, str]

    @property
    def ids(self) -> list[str]:
        return sorted(self.y)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _load_rows(
    cache_dir: Path, ids: list[str], labels: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Cached ``(T, D)`` rows; a missing file or a length mismatch raises (C2)."""
    missing = [v for v in ids if not (cache_dir / f"{v}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ids have no feature in {cache_dir}: {missing[:5]}")
    rows = {v: np.load(cache_dir / f"{v}.npy").astype(np.float64) for v in ids}
    bad = [
        f"{v}: {len(rows[v])} rows vs {len(labels[v])} labels"
        for v in ids
        if len(rows[v]) != len(labels[v])
    ]
    if bad:
        raise ValueError(f"{len(bad)} clips misaligned with their labels: {bad[:5]}")
    return rows


def load_dota_dev(labels_dir: Path, clip_dir: Path) -> tuple[Corpus, int]:
    """DoTA-dev at s8; share = native annotation span (the freeze's stratum)."""
    frame_labels = json.loads((labels_dir / DOTA_FRAME_LABELS).read_text(encoding="utf-8"))
    meta = json.loads((labels_dir / DOTA_META).read_text(encoding="utf-8"))
    dev = load_split(constants.V2_SPLIT_DOTA_DEV)
    ids = sorted(v for v in dev if v in frame_labels)
    y = {v: np.asarray(frame_labels[v], dtype=np.int64) for v in ids}
    share = {v: float(meta[v]["anomaly_span"][1] - meta[v]["anomaly_span"][0]) for v in ids}
    corpus = Corpus(
        "DoTA-dev", _load_rows(clip_dir, ids, y), y, share, {v: dota_group(v) for v in ids}
    )
    return corpus, len(dev) - len(ids)


def load_dota_dev_s1(
    s1_dir: Path, metadata: Path, split_file: Path, stride: int
) -> tuple[Corpus, int]:
    """DoTA-dev from a stride-1 cache at ``stride`` (D11); labels from the annotation (J1)."""
    dev = load_split(constants.V2_SPLIT_DOTA_DEV)
    records = {r.video_id: r for r in parse_metadata(metadata, read_split_ids(split_file))}
    missing = [v for v in dev if not (s1_dir / f"{v}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} DoTA-dev ids have no s1 feature: {missing[:5]}")
    x: dict[str, np.ndarray] = {}
    y: dict[str, np.ndarray] = {}
    for v in sorted(dev):
        s1 = np.load(s1_dir / f"{v}.npy")
        x[v] = np.asarray(s1[::stride], dtype=np.float64)
        y[v] = np.asarray(resized_frame_labels(records[v], len(s1), stride), dtype=np.int64)
        if len(y[v]) != len(x[v]):
            raise ValueError(f"{v}: {len(x[v])} rows vs {len(y[v])} labels at stride {stride}")
    share = {v: records[v].span[1] - records[v].span[0] for v in y}
    corpus = Corpus(f"DoTA-dev-s{stride}", x, y, share, {v: dota_group(v) for v in y})
    return corpus, 0


def load_t2(
    name: str, ids: list[str], annotation: Path, census: dict[str, int], clip_dir: Path
) -> Corpus:
    """Whole T2 source videos at s8; share = s8 label mean; each source its own cluster."""
    y, _types = source_labels(annotation, census, ids, constants.FRAME_STRIDE)
    share = {v: float(y[v].mean()) for v in ids}
    return Corpus(name, _load_rows(clip_dir, ids, y), y, share, {v: v for v in ids})


def t2_sources(meta: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """(T2-val, T2-train minus T2-val) source ids, both sorted."""
    val = sorted(load_split(constants.V2_SPLIT_T2_VAL))
    pool = set(t2_pool_from_meta(meta).windows_by_source)
    missing = sorted(set(val) - pool)
    if missing:
        raise ValueError(f"{len(missing)} T2-val sources are not T2-train sources: {missing[:5]}")
    return val, sorted(pool - set(val))


# ---------------------------------------------------------------------------
# E0 + E2(a)
# ---------------------------------------------------------------------------
def rate_audit(corpora: dict[str, tuple[Corpus, int, int]]) -> dict[str, Any]:
    """E0 (record only, D8): seconds per step and clip length in seconds, per ``(fps, stride)``."""
    out: dict[str, Any] = {}
    for name, (corpus, fps, stride) in corpora.items():
        steps = np.asarray([len(corpus.y[v]) for v in corpus.ids], dtype=np.float64)
        per_step = stride / fps
        out[name] = {
            "fps": fps,
            "stride": stride,
            "seconds_per_step": per_step,
            "steps_median": float(np.median(steps)),
            "clip_seconds_median": float(np.median(steps) * per_step),
        }
    names = list(out)
    out["step_ratio"] = out[names[0]]["seconds_per_step"] / out[names[1]]["seconds_per_step"]
    return out


def share_histogram(corpus: Corpus) -> dict[str, Any]:
    """E2(a): clips per share bin, and how many start normal."""
    bins: dict[str, int] = dict.fromkeys(constants.V2_SHARE_BIN_LABELS, 0)
    for v in corpus.ids:
        bins[share_bin(corpus.share[v])] += 1
    shares = np.asarray([corpus.share[v] for v in corpus.ids])
    return {
        "clips": len(corpus.ids),
        "clips_per_bin": bins,
        "share_median": float(np.median(shares)),
        "starts_normal": float(np.mean([corpus.y[v][0] == 0 for v in corpus.ids])),
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def rel_position(length: int) -> np.ndarray:
    """``t / T`` for ``t = 0 .. T-1`` (the proposal's position ruler)."""
    return np.arange(length, dtype=np.float64) / length


def two_class(corpus: Corpus) -> list[str]:
    return [v for v in corpus.ids if 0 < int(corpus.y[v].sum()) < len(corpus.y[v])]


def clip_aucs(scores: dict[str, np.ndarray], corpus: Corpus) -> dict[str, float]:
    """Per-clip frame AUC over two-class clips."""
    return {v: frame_auc(scores[v], corpus.y[v]) for v in two_class(corpus)}


def cluster_ci(
    values: dict[str, float], corpus: Corpus, resamples: int, seed: int
) -> dict[str, float] | None:
    """Mean of ``values`` and a percentile CI resampling ``corpus.group`` clusters (D3)."""
    return cluster_bootstrap_ci(values, corpus.group, resamples, seed, constants.V2_E2_CI)


def macro_by_bin(
    values: dict[str, float], corpus: Corpus, resamples: int, seed: int
) -> dict[str, Any]:
    """Overall and per-share-bin macro, each with its cluster CI."""
    by_bin = {
        label: {v: a for v, a in values.items() if share_bin(corpus.share[v]) == label}
        for label in constants.V2_SHARE_BIN_LABELS
    }
    return {
        "overall": cluster_ci(values, corpus, resamples, seed),
        "bins": {label: cluster_ci(sub, corpus, resamples, seed) for label, sub in by_bin.items()},
    }


def stratified_auc(scores: dict[str, np.ndarray], corpus: Corpus, ids: list[str]) -> float | None:
    """Model-free cross-check (§4.2 step 3): per-clip min-max, then pooled AUC inside
    equal ``t/T`` bins, averaged with weights = normal x accident pairs."""
    if not ids:
        return None
    norm, pos, lab = [], [], []
    for v in ids:
        s = scores[v]
        span = s.max() - s.min()
        norm.append((s - s.min()) / span if span > 0 else np.zeros_like(s))
        pos.append(rel_position(len(s)))
        lab.append(corpus.y[v])
    score, where, label = np.concatenate(norm), np.concatenate(pos), np.concatenate(lab)
    edges = np.linspace(0.0, 1.0, constants.V2_E2_STRAT_BINS + 1)
    which = np.clip(np.digitize(where, edges[1:-1]), 0, constants.V2_E2_STRAT_BINS - 1)
    total, weight = 0.0, 0.0
    for b in range(constants.V2_E2_STRAT_BINS):
        mask = which == b
        n1 = int(label[mask].sum())
        n0 = int(mask.sum()) - n1
        if n0 and n1:
            total += n0 * n1 * frame_auc(score[mask], label[mask])
            weight += n0 * n1
    return total / weight if weight else None


@dataclass(frozen=True)
class Trend:
    """``f``: cubic in ``t/T`` on T2-val normal steps, clamped to [lo, hi] of their ``t/T``."""

    coeffs: np.ndarray
    lo: float
    hi: float

    def __call__(self, tau: np.ndarray) -> np.ndarray:
        return np.polyval(self.coeffs, np.clip(tau, self.lo, self.hi))


def fit_trend(d: dict[str, np.ndarray], corpus: Corpus) -> Trend:
    """Least-squares cubic of ``d`` on ``t/T`` over label-0 steps only."""
    tau = np.concatenate([rel_position(len(corpus.y[v]))[corpus.y[v] == 0] for v in corpus.ids])
    dev = np.concatenate([d[v][corpus.y[v] == 0] for v in corpus.ids])
    lo, hi = np.percentile(tau, constants.V2_E2_TREND_CLAMP_PCT)
    held = np.clip(tau, lo, hi)
    return Trend(np.polyfit(held, dev, constants.V2_E2_TREND_DEGREE), float(lo), float(hi))


def normal_coverage(corpus: Corpus) -> float:
    """Share of label-0 steps in the last fifth of ``t/T`` (the fallback trigger)."""
    tau = np.concatenate([rel_position(len(corpus.y[v]))[corpus.y[v] == 0] for v in corpus.ids])
    return float(np.mean(tau >= LAST_FIFTH))


# ---------------------------------------------------------------------------
# E2(b)
# ---------------------------------------------------------------------------
def _read_scores(
    scores: dict[str, np.ndarray], corpus: Corpus, resamples: int, seed: int
) -> dict[str, Any]:
    ids = two_class(corpus)
    per_bin = {
        label: [v for v in ids if share_bin(corpus.share[v]) == label]
        for label in constants.V2_SHARE_BIN_LABELS
    }
    return {
        "macro": macro_by_bin(clip_aucs(scores, corpus), corpus, resamples, seed),
        "stratified": {
            "overall": stratified_auc(scores, corpus, ids),
            "bins": {b: stratified_auc(scores, corpus, sub) for b, sub in per_bin.items()},
        },
    }


def position_ruler(corpus: Corpus, resamples: int, seed: int) -> dict[str, Any]:
    return _read_scores(
        {v: rel_position(len(corpus.y[v])) for v in corpus.ids}, corpus, resamples, seed
    )


def evaluate_reference(
    kind: str, t2val: Corpus, dota: Corpus, resamples: int, seed: int
) -> dict[str, Any]:
    """Raw ``d`` and residualized ``r`` for one reference on both corpora."""
    d_val = {v: deviation(t2val.x[v], kind) for v in t2val.ids}
    trend = fit_trend(d_val, t2val)
    out: dict[str, Any] = {
        "trend": {"coeffs": trend.coeffs.tolist(), "clamp": [trend.lo, trend.hi]}
    }
    for corpus, d in ((t2val, d_val), (dota, {v: deviation(dota.x[v], kind) for v in dota.ids})):
        r = {v: d[v] - trend(rel_position(len(d[v]))) for v in d}
        minus_f = {v: -trend(rel_position(len(d[v]))) for v in d}
        out[corpus.name] = {
            "d": _read_scores(d, corpus, resamples, seed),
            "r": _read_scores(r, corpus, resamples, seed),
            # printed, not decided on: what -f alone (a pure position score) scores
            "minus_f": _read_scores(minus_f, corpus, resamples, seed),
        }
    return out


def _decision_value(block: dict[str, Any], metric: str, where: str) -> float | None:
    """The decision metric's point value, overall or in one share bin."""
    if metric == METRIC_R:
        ci = block["r"]["macro"]["overall" if where == "overall" else "bins"]
        ci = ci if where == "overall" else ci[where]
        return None if ci is None else float(ci["mean"])
    strat = block["r"]["stratified"]
    value = strat["overall"] if where == "overall" else strat["bins"][where]
    return None if value is None else float(value)


def choose(references: dict[str, dict[str, Any]], dota_name: str, metric: str) -> dict[str, Any]:
    """§4.2 steps 4-5, applied mechanically on DoTA-dev."""
    floor = constants.V2_E2_REVERSAL_FLOOR
    rows: dict[str, Any] = {}
    for kind, result in references.items():
        block = result[dota_name]
        highs = {b: _decision_value(block, metric, b) for b in HIGH_BINS}
        strat = block["d"]["stratified"]["overall"]  # §4.2 step 3: the cross-check is on d
        no_reversal = all(v is not None and v >= floor for v in highs.values())
        strat_ok = strat is not None and strat >= floor
        rows[kind] = {
            "overall": _decision_value(block, metric, "overall"),
            "high_bins": highs,
            "stratified": strat,
            "eligible": no_reversal and strat_ok,
            "decided_by_gt70": (
                highs["50-70"] is not None
                and highs["50-70"] >= floor
                and strat_ok
                and not no_reversal
            ),
        }
    eligible = {k: r for k, r in rows.items() if r["eligible"] and r["overall"] is not None}
    chosen = max(eligible, key=lambda k: eligible[k]["overall"]) if eligible else None
    return {"metric": metric, "rows": rows, "chosen": chosen}


# ---------------------------------------------------------------------------
# E2(c)
# ---------------------------------------------------------------------------
def _stack(corpus: Corpus, kind: str | None) -> tuple[np.ndarray, np.ndarray]:
    feats = [
        corpus.x[v] - reference(corpus.x[v], kind) if kind else corpus.x[v] for v in corpus.ids
    ]
    return np.concatenate(feats), np.concatenate([corpus.y[v] for v in corpus.ids])


def transfer_probe(
    kinds: list[str], train: Corpus, dota: Corpus, resamples: int, seed: int
) -> dict[str, Any]:
    """Probe fitted on T2-train, scored on DoTA-dev: raw ``x`` vs ``x - mu^ref``."""
    per_kind: dict[str, dict[str, float]] = {}
    for kind in [None, *kinds]:
        train_x, train_y = _stack(train, kind)
        test_x, _ = _stack(dota, kind)
        flat = transfer_scores(train_x, train_y, test_x, seed)
        scores, offset = {}, 0
        for v in dota.ids:
            scores[v] = flat[offset : offset + len(dota.y[v])]
            offset += len(dota.y[v])
        per_kind[kind or "raw"] = clip_aucs(scores, dota)
        LOGGER.info(
            "E2(c) %s: DoTA-dev transfer macro %.4f",
            kind or "raw",
            float(np.mean(list(per_kind[kind or "raw"].values()))),
        )
    raw = per_kind["raw"]
    out: dict[str, Any] = {"raw": cluster_ci(raw, dota, resamples, seed)}
    for kind in kinds:
        delta = {v: per_kind[kind][v] - raw[v] for v in raw}
        out[kind] = {
            "macro": cluster_ci(per_kind[kind], dota, resamples, seed),
            "delta_vs_raw": cluster_ci(delta, dota, resamples, seed),
        }
    return out


def verdict(choice: dict[str, Any], transfer: dict[str, Any]) -> str:
    chosen = choice["chosen"]
    if chosen is None:
        return VERDICT_DROPPED
    if transfer[chosen]["delta_vs_raw"]["high"] < 0:
        return VERDICT_VETOED
    return str(chosen)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _ci(ci: dict[str, float] | None) -> str:
    if ci is None:
        return "—"
    return f"{ci['mean']:.4f} [{ci['low']:.4f}, {ci['high']:.4f}] (n={ci['clips']})"


def _pt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_markdown(readout: dict[str, Any]) -> str:
    """Human read-out; the JSON beside it is the record."""
    bins = constants.V2_SHARE_BIN_LABELS
    rate, hist = readout["e0_rate"], readout["e2a_share"]
    lines = [
        "# v2 P2 — E0 / E2(a) / E2(b) / E2(c) read-out",
        "",
        "Rule: proposal §4.2 steps 1-6; addendum D3 (DoTA CI over source videos), "
        f"D11 (DoTA-dev at stride {readout.get('dota_stride', constants.FRAME_STRIDE)}).",
        f"DoTA-dev clips without labels: {readout['dota_dev_unlabelled']}.",
        "",
        "## E0 — rate audit (record only, D8)",
        "",
        "| corpus | fps | s/step | median steps | median clip s |",
        "|---|---|---|---|---|",
    ]
    for name, row in rate.items():
        if isinstance(row, dict):
            lines.append(
                f"| {name} | {row['fps']} | {row['seconds_per_step']:.3f} | "
                f"{row['steps_median']:.0f} | {row['clip_seconds_median']:.1f} |"
            )
    lines += [
        "",
        f"Step-length ratio: **{rate['step_ratio']:.2f}x**.",
        "",
        "## E2(a) — accident share",
        "",
        "| corpus | clips | " + " | ".join(bins) + " | median share | starts normal |",
        "|---|---|" + "---|" * len(bins) + "---|---|",
    ]
    for name, h in hist.items():
        lines.append(
            f"| {name} | {h['clips']} | "
            + " | ".join(str(h["clips_per_bin"][b]) for b in bins)
            + f" | {h['share_median']:.3f} | {h['starts_normal']:.3f} |"
        )
    lines += [
        "",
        f"Normal-step coverage of the last fifth of t/T (T2-val): "
        f"**{readout['coverage']:.3f}** (fallback below {constants.V2_E2_COVERAGE_MIN}) → "
        f"decision metric **{readout['choice']['metric']}**.",
        "",
        "## E2(b) — macro AUC per share bin",
        "",
    ]
    for corpus_name in readout["corpora"]:
        lines += [
            f"### {corpus_name}",
            "",
            "| score | overall | " + " | ".join(bins) + " | stratified |",
            "|---|---|" + "---|" * len(bins) + "---|",
        ]
        rows = [("position t/T", readout["position"][corpus_name])]
        for kind, result in readout["references"].items():
            rows += [
                (f"{kind} d (raw)", result[corpus_name]["d"]),
                (f"{kind} r (residual)", result[corpus_name]["r"]),
                (f"{kind} -f (trend only)", result[corpus_name]["minus_f"]),
            ]
        for title, block in rows:
            m = block["macro"]
            lines.append(
                f"| {title} | {_ci(m['overall'])} | "
                + " | ".join(_ci(m["bins"][b]) for b in bins)
                + f" | {_pt(block['stratified']['overall'])} |"
            )
        lines.append("")
    choice = readout["choice"]
    lines += [
        "## Choice (DoTA-dev, §4.2 steps 4-5)",
        "",
        "| ref | overall | 50-70 | >70 | stratified | eligible | decided by >70 |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, row in choice["rows"].items():
        lines.append(
            f"| {kind} | {_pt(row['overall'])} | {_pt(row['high_bins']['50-70'])} | "
            f"{_pt(row['high_bins']['>70'])} | {_pt(row['stratified'])} | "
            f"{row['eligible']} | {row['decided_by_gt70']} |"
        )
    transfer = readout["transfer"]
    lines += [
        "",
        "## E2(c) — transfer probe T2-train → DoTA-dev (veto)",
        "",
        f"raw x: {_ci(transfer['raw'])}",
        "",
        "| ref | macro | Δ vs raw |",
        "|---|---|---|",
    ]
    for kind in readout["references"]:
        lines.append(
            f"| {kind} | {_ci(transfer[kind]['macro'])} | {_ci(transfer[kind]['delta_vs_raw'])} |"
        )
    lines += ["", f"## Verdict: **{readout['verdict']}**", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dota_s1_dir is not None:
        dota, unlabelled = load_dota_dev_s1(
            args.dota_s1_dir, args.dota_metadata, args.dota_split_file, args.dota_stride
        )
        dota_stride = args.dota_stride
    else:
        dota, unlabelled = load_dota_dev(args.dota_labels_dir, args.dota_clip_dir)
        dota_stride = constants.FRAME_STRIDE
    meta = json.loads(args.t2_meta.read_text(encoding="utf-8"))
    val_ids, train_ids = t2_sources(meta)
    census = load_counts(args.census)
    t2val = load_t2("T2-val", val_ids, args.annotation, census, args.dada_clip_dir)
    t2train = load_t2("T2-train", train_ids, args.annotation, census, args.dada_clip_dir)
    LOGGER.info(
        "DoTA-dev %d clips, T2-val %d sources, T2-train %d sources",
        len(dota.ids),
        len(t2val.ids),
        len(t2train.ids),
    )
    coverage = normal_coverage(t2val)
    metric = METRIC_STRAT if coverage < constants.V2_E2_COVERAGE_MIN else METRIC_R
    kinds: list[str] = list(constants.V2_CRN_REFERENCES)
    references = {k: evaluate_reference(k, t2val, dota, args.bootstrap, args.seed) for k in kinds}
    choice = choose(references, dota.name, metric)
    transfer = transfer_probe(kinds, t2train, dota, args.bootstrap, args.seed)
    return {
        "corpora": [t2val.name, dota.name],
        "dota_stride": dota_stride,
        "dota_dev_unlabelled": unlabelled,
        "e0_rate": rate_audit(
            {
                dota.name: (dota, constants.DOTA_FPS, dota_stride),
                t2val.name: (t2val, constants.DADA_ASSUMED_FPS, constants.FRAME_STRIDE),
            }
        ),
        "e2a_share": {c.name: share_histogram(c) for c in (dota, t2val)},
        "coverage": coverage,
        "position": {c.name: position_ruler(c, args.bootstrap, args.seed) for c in (t2val, dota)},
        "references": references,
        "choice": choice,
        "transfer": transfer,
        "verdict": verdict(choice, transfer),
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "t2_train_sources": len(t2train.ids),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--dota-labels-dir", type=Path, default=None, help="s8 path: labels_s8")
    parser.add_argument("--dota-clip-dir", type=Path, default=None, help="s8 path: DoTA_s8_ncc")
    parser.add_argument("--dota-s1-dir", type=Path, default=None, help="D11 path: DoTA_s1_ncc")
    parser.add_argument("--dota-stride", type=int, default=constants.V2_E1_STRIDE_BC)
    parser.add_argument("--dota-metadata", type=Path, default=None, help="metadata_val.json")
    parser.add_argument("--dota-split-file", type=Path, default=None, help="val_split.txt")
    parser.add_argument("--t2-meta", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument(
        "--census", nargs="+", required=True, help="census JSON globs ({video_id: frames on disk})"
    )
    parser.add_argument("--dada-clip-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=constants.V2_E2_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=constants.V2_SPLIT_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    s1_path = args.dota_s1_dir is not None
    if s1_path and (args.dota_metadata is None or args.dota_split_file is None):
        parser.error("--dota-s1-dir needs --dota-metadata and --dota-split-file")
    if not s1_path and (args.dota_labels_dir is None or args.dota_clip_dir is None):
        parser.error("give --dota-labels-dir + --dota-clip-dir (s8) or --dota-s1-dir (D11)")
    if s1_path and (args.dota_labels_dir is not None or args.dota_clip_dir is not None):
        parser.error("the s8 flags and --dota-s1-dir are exclusive")
    readout = run(args)
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(readout))
    LOGGER.info("E2 verdict: %s -> %s", readout["verdict"], args.out_dir)


if __name__ == "__main__":
    main()
