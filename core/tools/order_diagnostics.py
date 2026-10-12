"""v2.2 order diagnostics S1-S3: does temporal order between steps carry detection signal?

Pre-registered in ``core/docs/v2.2/KAT_VAD_PROPOSAL_v2.2.md`` §3-§6 (review round 4) and
``.project/plans/katvad-v2.2-order-diagnostics.md`` (O1-O15). No trunk is trained. Every gate uses
one bar (§6.1): cluster-bootstrap 95 % lower bound **> 0** and mean **>= +0.01**, on DoTA-CAP-dev at
protocol B (stride 3, whole clip). T2-val is read per source and reported, never decides.

* ``s1`` -- parameter-free, CLIP only.
  **S1a** per-step score ``m5`` (1.5 s causal max of the rate-normalised change ‖Δx̄‖): AUC - 0.5 and
  AUC - AUC(twin) must both pass, else the TDP branch (S3, S5) closes; **S2 runs either way**.
  **S1b** Δu exit echo: Δu is admitted to κ only if the CI of (steps +5..+6 - steps +10..+15
  after the span end) lies inside ±0.1 SD of the baseline (equivalence).
  **S1c** clip start: any of five summary channels with |SMD| > 0.1 withdraws the early-onset
  prediction (never stops).
* ``s2`` -- the temporal probe on ``F0 = [x̃ ; ũ ; p]``, true order vs the adjacency-free block
  shuffle. Gate on ``F0``; the same contrast on ``[x̃ ; p]`` (no VideoMAE bag) is printed.
* ``s3`` -- refuses to run unless S1a and S2 passed. ``F0`` vs ``F0 + K`` (VALUE) and
  ``F0 + K`` vs ``F0 + K^π`` (ORDER); both must pass for S5 to be eligible.

Printed beside every Δ: the ``f_2`` Δ (``z(score) + 2 z(p_T2)`` within clip) and per-seed Δs.
DoTA-eval and DoTA-CAP-eval are never loaded.

CLI (shared arguments, then the step)::

    python -m core.tools.order_diagnostics s1 \\
        --data-dir data/DADA2000_orig_v2 --annotation <DADA>/dada标注.xlsx \\
        --census data/DADA2000_orig/counts/resolved_census.json \\
        --t2-clip-dir cache/clip/DADA2000_orig --video-root cache/video \\
        --dota-s1-dir cache/clip/DoTA_s1_ncc --metadata data/DoTA/metadata_val.json \\
        --split-file data/DoTA/val_split.txt --out-dir outputs/v2/REPORTS/v22_order
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from core import constants
from core.crn.reference import reference
from core.data.dada_origin import load_counts
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_splits import load_split
from core.eda.features import transfer_scores
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools import dota_cap, e2d_probe, extract_video_features
from core.tools import order_features as of
from core.tools import order_probe as op
from core.tools.final_readout import fused_clip_aucs
from core.tools.kill_switch_probe import (
    load_features,
    source_labels,
    write_json_atomic,
    write_text_atomic,
)
from core.tools.position_prior import prior_scores, training_frames
from core.tools.v2_dataset import window_frame_labels

LOGGER = logging.getLogger(__name__)

STEPS = ("s1", "s2", "s3")
TRUE, SHUFFLED = "true", "shuffled"
F0, CLIP_ONLY, F0_K, F0_KPI = "F0", "clip_only", "F0+K", "F0+K_pi"
DOTA, T2_VAL = "dota_cap_dev", "t2_val"
SHORT = 12  # sequences under this length cannot reach the twin gap of 6 (arch §4)


def readout_json(step: str) -> str:
    return f"{step}_readout.json"


def readout_md(step: str) -> str:
    return f"{step}_readout.md"


@dataclass(frozen=True)
class Corpus:
    """Raw per-sequence CLIP rows, V2-S rows, step labels and bootstrap clusters."""

    x: dict[str, np.ndarray]
    u: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    group: dict[str, str]
    ego: dict[str, bool]

    def two_class(self) -> list[str]:
        return sorted(v for v in self.y if 0 < int(self.y[v].sum()) < len(self.y[v]))


@dataclass(frozen=True)
class Data:
    train: Corpus  # T2-train - T2-val sources
    windows: list[tuple[str, int, int]]  # (source, start, end) of every training window
    val: Corpus  # T2-val sources (reported)
    dota: Corpus  # DoTA-CAP-dev clips (decides)
    prior: dict[str, np.ndarray]  # p_T2 at every DoTA step (O8)


# ---------------------------------------------------------------------------
# loading (O1-O3, O8, O15)
# ---------------------------------------------------------------------------
def _t2_corpus(ids: list[str], args: argparse.Namespace, census: dict[str, int]) -> Corpus:
    y, _types = source_labels(args.annotation, census, ids, constants.FRAME_STRIDE)
    x = load_features(args.t2_clip_dir, ids)
    u = load_features(e2d_probe.t2_video_dir(args.video_root, constants.VIDEOMAE_ENCODER_S), ids)
    bad = [v for v in ids if not len(y[v]) == len(x[v]) == len(u[v])]
    if bad:
        raise ValueError(f"{len(bad)} T2 sources misaligned (labels / CLIP / V2-S): {bad[:5]}")
    return Corpus(x, u, y, {v: v for v in ids}, {})


def training_windows(data_dir: Path, y: dict[str, np.ndarray]) -> list[tuple[str, int, int]]:
    """Every T2-train - T2-val window; its labels must equal its source's, sliced (O2)."""
    train: dict[str, int] = json.loads((data_dir / constants.LABELS_TRAIN_FILENAME).read_text())
    meta: dict[str, dict[str, Any]] = json.loads((data_dir / constants.META_FILENAME).read_text())
    out: list[tuple[str, int, int]] = []
    for window_id in sorted(train):
        entry = meta[window_id]
        source, start, end = str(entry["source"]), int(entry["start"]), int(entry["end"])
        expected = np.asarray(window_frame_labels(entry, window_id), dtype=np.int64)
        if source not in y or not np.array_equal(y[source][start:end], expected):
            raise ValueError(f"{window_id}: window labels differ from its source's slice (O2)")
        out.append((source, start, end))
    return out


def load_data(args: argparse.Namespace) -> Data:
    census = load_counts([str(p) for p in args.census])
    meta: dict[str, dict[str, Any]] = json.loads(
        (args.data_dir / constants.META_FILENAME).read_text()
    )
    train_ids = json.loads((args.data_dir / constants.LABELS_TRAIN_FILENAME).read_text())
    sources = sorted({str(meta[w]["source"]) for w in train_ids})
    val_ids = load_split(constants.V2_SPLIT_T2_VAL, args.split_dir)
    if set(sources) & set(val_ids):
        raise ValueError("T2-val sources appear among the training windows: wrong --data-dir")
    train = _t2_corpus(sources, args, census)
    windows = training_windows(args.data_dir, train.y)
    val = _t2_corpus(val_ids, args, census)

    enc = constants.VIDEOMAE_ENCODER_S
    alignment = e2d_probe.dota_cap_alignment_sha256(args.split_dir)
    dota_dir = e2d_probe.dota_video_dir(args.video_root, enc)
    e2d_probe.check_manifest(dota_dir, dota_cap.video_manifest(enc, alignment))
    e2d_probe.check_manifest(
        e2d_probe.t2_video_dir(args.video_root, enc),
        extract_video_features.build_manifest(
            enc, constants.FRAME_STRIDE, constants.VIDEOMAE_WEIGHTS_SHA256[enc]
        ),
    )
    parsed = parse_metadata(args.metadata, read_split_ids(args.split_file))
    records = {r.video_id: r for r in parsed}
    cap_ids = load_split(constants.V2_SPLIT_DOTA_CAP_DEV, args.split_dir)
    probe = e2d_probe.load_dota(
        cap_ids, records, args.dota_s1_dir, {enc: dota_dir}, constants.V22_DOTA_STRIDE
    )
    dota = Corpus(probe.x, probe.u[enc], probe.y, probe.group,
                  {v: records[v].is_ego for v in cap_ids})
    matrix, target, _ = training_frames(args.data_dir)
    prior = prior_scores(matrix, target, {v: len(dota.y[v]) for v in cap_ids}, args.seed)
    LOGGER.info("loaded: %d train sources (%d windows), %d T2-val sources, %d DoTA-CAP-dev clips",
                len(sources), len(windows), len(val_ids), len(cap_ids))
    return Data(train, windows, val, dota, prior)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def ci(values: dict[str, float], groups: dict[str, str], args: argparse.Namespace) -> Any:
    return cluster_bootstrap_ci(values, groups, args.resamples, args.seed, constants.V22_CI)


def passes(c: dict[str, float] | None) -> bool:
    """Proposal §6.1: CI low > 0 and mean >= +0.01."""
    return c is not None and c["low"] > 0.0 and c["mean"] >= constants.V22_BAR_MIN


def clip_aucs(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> dict[str, float]:
    return {v: frame_auc(scores[v], labels[v]) for v in sorted(scores)
            if 0 < int(labels[v].sum()) < len(labels[v])}


def diff(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {v: a[v] - b[v] for v in a if v in b}


def contrast(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], corpus: Corpus,
    prior: dict[str, np.ndarray] | None, args: argparse.Namespace,
) -> dict[str, Any]:
    """Macro of both, the paired Δ a - b, and (DoTA only) the f_2 Δ."""
    auc_a, auc_b = clip_aucs(a, corpus.y), clip_aucs(b, corpus.y)
    out = {"a": ci(auc_a, corpus.group, args), "b": ci(auc_b, corpus.group, args),
           "delta": ci(diff(auc_a, auc_b), corpus.group, args)}
    if prior is not None:
        w = constants.V22_FUSION_WEIGHT
        out["f2_delta"] = ci(diff(fused_clip_aucs(a, prior, corpus.y, w),
                                  fused_clip_aucs(b, prior, corpus.y, w)), corpus.group, args)
    return out


def _seq_lengths(corpus: Corpus) -> dict[str, int]:
    return {v: len(y) for v, y in corpus.y.items()}


def short_share(lengths: dict[str, int], gaps: dict[str, int]) -> dict[str, Any]:
    """Share of sequences under 12 steps and the twin gaps reached (arch §4)."""
    values, counts = np.unique(list(gaps.values()) or [0], return_counts=True)
    return {
        "sequences": len(lengths),
        "under_12": int(sum(n < SHORT for n in lengths.values())),
        "under_3_excluded": int(sum(n < constants.V22_MIN_STEPS for n in lengths.values())),
        "gap_counts": {int(g): int(c) for g, c in zip(values, counts, strict=True)} if gaps else {},
    }


# ---------------------------------------------------------------------------
# S1
# ---------------------------------------------------------------------------
def _scores_and_twin(
    x: np.ndarray, stats: of.OrderStats, tag: str, name: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Columns ``[m5, s]`` in true order and their quantile-matched twin."""

    def columns(z: np.ndarray) -> np.ndarray:
        parts = of.change_parts(z, stats)
        return np.concatenate([parts["m5"], parts["s"]], axis=1)

    true = columns(x)
    twin, gap = of.twin_rows(x, true, columns, of.sequence_rng(tag, name))
    return true, twin, gap


def s1a(data: Data, stats: of.OrderStats, args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, corpus in ((DOTA, data.dota), (T2_VAL, data.val)):
        scores: dict[str, dict[str, np.ndarray]] = {k: {} for k in ("m5", "m5_pi", "s", "s_pi")}
        gaps: dict[str, int] = {}
        for v, x in corpus.x.items():
            if len(x) < constants.V22_MIN_STEPS:
                continue
            true, twin, gaps[v] = _scores_and_twin(x, stats, "s1-twin", v)
            scores["m5"][v], scores["s"][v] = true[:, 0], true[:, 1]
            scores["m5_pi"][v], scores["s_pi"][v] = twin[:, 0], twin[:, 1]
        aucs = {k: clip_aucs(s, corpus.y) for k, s in scores.items()}
        block: dict[str, Any] = {
            "macro": {k: ci(a, corpus.group, args) for k, a in aucs.items()},
            "m5_minus_half": ci({v: a - 0.5 for v, a in aucs["m5"].items()}, corpus.group, args),
            "m5_minus_twin": ci(diff(aucs["m5"], aucs["m5_pi"]), corpus.group, args),
            "s_minus_half": ci({v: a - 0.5 for v, a in aucs["s"].items()}, corpus.group, args),
            "s_minus_twin": ci(diff(aucs["s"], aucs["s_pi"]), corpus.group, args),
            "sequences": short_share(_seq_lengths(corpus), gaps),
        }
        if label == DOTA:
            prior_auc = clip_aucs(data.prior, corpus.y)
            ids = sorted(aucs["m5"])
            block["corr_with_p_t2"] = float(np.corrcoef(
                [aucs["m5"][v] for v in ids], [prior_auc[v] for v in ids])[0, 1])
            for side, keep in (("ego", True), ("non_ego", False)):
                sub = {v: a for v, a in aucs["m5"].items() if corpus.ego[v] == keep}
                block[f"m5_{side}"] = ci(sub, corpus.group, args)
                block[f"m5_minus_twin_{side}"] = ci(
                    {v: d for v, d in diff(aucs["m5"], aucs["m5_pi"]).items() if v in sub},
                    corpus.group, args)
            top = []
            for v in ids:
                order = np.argsort(-scores["m5"][v], kind="stable")[: constants.V22_TOP_STEPS]
                top.append(float(corpus.y[v][order].mean()))
            block["top5_in_span_share"] = float(np.mean(top))
        out[label] = block
    decides = out[DOTA]
    out["verdict"] = (constants.V22_GATE_OK
                      if passes(decides["m5_minus_half"]) and passes(decides["m5_minus_twin"])
                      else constants.V22_GATE_NO)
    return out


def _echo_series(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    s = np.linalg.norm(of.rate_normalised_differences(x, sigma), axis=1)
    return np.concatenate([s[:1], s])


def s1b(data: Data, args: argparse.Namespace) -> dict[str, Any]:
    """Exit echo of Δu (decides admission) beside the same read on Δx (printed)."""
    peak, base = constants.V22_ECHO_PEAK, constants.V22_ECHO_BASE
    out: dict[str, Any] = {}
    for stream, rows, train in (("u", data.dota.u, data.train.u), ("x", data.dota.x, data.train.x)):
        sigma = of.fit_stats(long_rows(train), dim=1).sigma_delta
        deltas: dict[str, float] = {}
        baseline: list[np.ndarray] = []
        autocorr: dict[int, list[float]] = {lag: [] for lag in constants.V22_ECHO_LAGS}
        for v, z in rows.items():
            if len(z) < constants.V22_MIN_STEPS:
                continue
            s = _echo_series(z, sigma)
            for lag in autocorr:
                if len(s) > lag + 2:
                    autocorr[lag].append(float(np.corrcoef(s[:-lag], s[lag:])[0, 1]))
            abnormal = np.flatnonzero(data.dota.y[v])
            if len(abnormal) == 0 or abnormal[-1] + base[1] >= len(s):
                continue
            end = int(abnormal[-1])
            base_rows = s[end + base[0] : end + base[1] + 1]
            deltas[v] = float(s[end + peak[0] : end + peak[1] + 1].mean() - base_rows.mean())
            baseline.append(base_rows)
        sd = float(np.concatenate(baseline).std()) if baseline else float("nan")
        c = ci(deltas, data.dota.group, args)
        margin = constants.V22_ECHO_MARGIN_SD * sd
        out[stream] = {
            "clips": len(deltas),
            "baseline_sd": sd,
            "delta": c,
            "margin": margin,
            "inside_margin": c is not None and -margin <= c["low"] and c["high"] <= margin,
            "autocorr": {lag: float(np.nanmean(vals)) if vals else None
                         for lag, vals in autocorr.items()},
        }
    out["admit_du"] = bool(out["u"]["inside_margin"])
    return out


def _smd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pooled = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2.0)
    pooled[pooled == 0.0] = 1.0
    return np.asarray((a.mean(axis=0) - b.mean(axis=0)) / pooled, dtype=np.float64)


def s1c(data: Data, stats: of.OrderStats) -> dict[str, Any]:
    head: list[np.ndarray] = []
    rest: list[np.ndarray] = []
    head_k: list[np.ndarray] = []
    rest_k: list[np.ndarray] = []
    k = constants.V22_START_STEPS
    for v, x in data.dota.x.items():
        y = data.dota.y[v]
        if len(x) <= k or y[:k].any():
            continue
        parts = of.change_parts(x, stats)
        summary, kap = of.summary_channels(parts), of.kappa(parts)
        normal = np.flatnonzero(y[k:] == 0) + k
        head += [summary[:k]]
        rest += [summary[normal]]
        head_k += [kap[:k]]
        rest_k += [kap[normal]]
    smd = _smd(np.concatenate(head), np.concatenate(rest))
    smd_all = _smd(np.concatenate(head_k), np.concatenate(rest_k))
    summary_smd = dict(zip(of.SUMMARY_CHANNELS, (float(s) for s in smd), strict=True))
    return {
        "clips": len(head),
        "summary_smd": summary_smd,
        "all_channels_abs_max": float(np.abs(smd_all).max()),
        "all_channels_over_0.1": int((np.abs(smd_all) > constants.V22_SMD_MAX).sum()),
        "h6_kept": bool(all(abs(s) <= constants.V22_SMD_MAX for s in summary_smd.values())),
    }


def long_rows(rows: dict[str, np.ndarray]) -> list[np.ndarray]:
    """T2-train sequences long enough for κ (O10): the rows every statistic is fitted on."""
    return [z for z in rows.values() if len(z) >= constants.V22_MIN_STEPS]


def run_s1(data: Data, args: argparse.Namespace) -> dict[str, Any]:
    train = long_rows(data.train.x)
    stats = of.fit_kappa_stats(of.fit_stats(train), train)
    return {
        "step": "S1",
        "bar": constants.V22_BAR_MIN,
        "s1a": s1a(data, stats, args),
        "s1b": s1b(data, args),
        "s1c": s1c(data, stats),
    }


# ---------------------------------------------------------------------------
# S2 / S3: the temporal probe
# ---------------------------------------------------------------------------
def crn(rows: np.ndarray) -> np.ndarray:
    return np.asarray(rows - reference(rows, constants.V22_CRN), dtype=np.float64)


def _sequences(
    base: dict[str, np.ndarray], y: dict[str, np.ndarray], shuffle: bool, tag: str
) -> tuple[dict[str, op.Sequence], dict[str, np.ndarray]]:
    seqs: dict[str, op.Sequence] = {}
    perms: dict[str, np.ndarray] = {}
    for v, rows in base.items():
        n = len(rows)
        perm = of.block_permutation(n, of.sequence_rng(tag, v)) if shuffle else np.arange(n)
        seqs[v], perms[v] = op.Sequence(rows[perm], perm, y[v][perm]), perm
    return seqs, perms


def probe_scores(
    data: Data, bases: dict[str, dict[str, np.ndarray]], shuffle: bool, position: bool,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]], int]:
    """Seed-averaged scores per test sequence (DoTA + T2-val), mapped back to original order.

    ``bases`` holds per-corpus feature rows keyed ``train`` / ``val`` / ``dota``.
    Returns the averaged scores, the per-seed scores and the number of unshuffled short sequences.
    """
    labels = {"train": data.train.y, "val": data.val.y, "dota": data.dota.y}
    seqs: dict[str, dict[str, op.Sequence]] = {}
    perms: dict[str, np.ndarray] = {}
    for part, base in bases.items():
        seqs[part], p = _sequences(base, labels[part], shuffle, "s2-shuffle")
        perms |= p
    train = [(seqs["train"][s], a, b) for s, a, b in data.windows if s in seqs["train"]]
    test = seqs["val"] | seqs["dota"]
    averaged, per_seed = op.seed_averaged_scores(
        train, test, position, tuple(args.seeds), torch.device(args.device)
    )

    def back(slot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for v, s in slot.items():
            restored = np.empty_like(s)
            restored[perms[v]] = s
            out[v] = restored
        return out

    unshuffled = sum(len(p) < constants.V22_MIN_SHUFFLE_BLOCK for p in perms.values())
    return back(averaged), {seed: back(s) for seed, s in per_seed.items()}, unshuffled


def read_pair(
    data: Data, a: tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]],
    b: tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]], args: argparse.Namespace,
) -> dict[str, Any]:
    """Contrast a - b on DoTA-CAP-dev (with f_2) and T2-val, plus per-seed DoTA Δs."""
    def part(scores: dict[str, np.ndarray], ids: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {v: scores[v] for v in ids if v in scores}

    out: dict[str, Any] = {
        DOTA: contrast(part(a[0], data.dota.y), part(b[0], data.dota.y), data.dota,
                       data.prior, args),
        T2_VAL: contrast(part(a[0], data.val.y), part(b[0], data.val.y), data.val, None, args),
    }
    out["per_seed_dota_delta"] = {
        int(seed): float(np.mean(list(diff(
            clip_aucs(part(a[1][seed], data.dota.y), data.dota.y),
            clip_aucs(part(b[1][seed], data.dota.y), data.dota.y)).values())))
        for seed in a[1]
    }
    return out


def f0_bases(data: Data, clip_only: bool) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for part, corpus in (("train", data.train), ("val", data.val), ("dota", data.dota)):
        out[part] = {v: crn(x) if clip_only else np.concatenate([crn(x), crn(corpus.u[v])], 1)
                     for v, x in corpus.x.items()}
    return out


def run_s2(data: Data, args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"step": "S2", "bar": constants.V22_BAR_MIN}
    for name, clip_only in ((F0, False), (CLIP_ONLY, True)):
        bases = f0_bases(data, clip_only)
        true = probe_scores(data, bases, False, True, args)
        shuffled = probe_scores(data, bases, True, True, args)
        out[name] = read_pair(data, true[:2], shuffled[:2], args)
        out[name]["unshuffled_short_sequences"] = shuffled[2]
        LOGGER.info("S2 %s: Δ(true - shuffled) DoTA %s", name, out[name][DOTA]["delta"])
    ok = passes(out[F0][DOTA]["delta"])
    out["verdict"] = constants.V22_GATE_OK if ok else constants.V22_GATE_NO
    return out


def _read_verdict(out_dir: Path, step: str, key: tuple[str, ...]) -> str:
    path = out_dir / readout_json(step)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run {step} first")
    node: Any = json.loads(path.read_text(encoding="utf-8"))
    for k in key:
        node = node[k]
    return str(node)


def kappa_rows(
    data: Data, admit_du: bool
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """Standardised κ and its quantile-matched twin for every sequence with >= 3 steps."""
    streams = [("x", {p: c.x for p, c in _corpora(data).items()}, long_rows(data.train.x))]
    if admit_du:
        streams.append(("u", {p: c.u for p, c in _corpora(data).items()}, long_rows(data.train.u)))
    true: dict[str, dict[str, list[np.ndarray]]] = {p: {} for p in _corpora(data)}
    twin: dict[str, dict[str, list[np.ndarray]]] = {p: {} for p in _corpora(data)}
    gaps: dict[str, int] = {}
    for stream, rows_by_part, train_rows in streams:
        stats = of.fit_kappa_stats(of.fit_stats(train_rows), train_rows)

        def k_of(z: np.ndarray, st: of.OrderStats = stats) -> np.ndarray:
            return of.kappa(of.change_parts(z, st))

        for part, rows in rows_by_part.items():
            for v, z in rows.items():
                if len(z) < constants.V22_MIN_STEPS:
                    continue
                k = k_of(z)
                k_pi, gaps[v] = of.twin_rows(z, k, k_of, of.sequence_rng(f"s3-twin-{stream}", v))
                true[part].setdefault(v, []).append(of.standardise(k, stats))
                twin[part].setdefault(v, []).append(of.standardise(k_pi, stats))
    join = {p: {v: np.concatenate(ks, 1) for v, ks in d.items()} for p, d in true.items()}
    join_pi = {p: {v: np.concatenate(ks, 1) for v, ks in d.items()} for p, d in twin.items()}
    return join, join_pi, gaps


def _corpora(data: Data) -> dict[str, Corpus]:
    return {"train": data.train, "val": data.val, "dota": data.dota}


def shortcut_auc(k: dict[str, dict[str, np.ndarray]], seed: int) -> float:
    """Printed (O14): grouped-CV DADA-vs-DoTA AUC on κ rows."""
    ids = [(v, 0) for p in ("train", "val") for v in k[p]] + [(v, 1) for v in k["dota"]]
    rows = {v: k[p][v] for p in k for v in k[p]}
    matrix = np.concatenate([rows[v] for v, _ in ids])
    target = np.concatenate([np.full(len(rows[v]), c) for v, c in ids])
    groups = np.concatenate([np.full(len(rows[v]), i) for i, (v, _) in enumerate(ids)])
    scores = np.zeros(len(target))
    folds = GroupKFold(n_splits=constants.V22_SHORTCUT_FOLDS)
    for fit, held in folds.split(matrix, target, groups):
        scores[held] = transfer_scores(matrix[fit], target[fit], matrix[held], seed)
    return frame_auc(scores, target)


def run_s3(data: Data, args: argparse.Namespace) -> dict[str, Any]:
    s1 = _read_verdict(args.out_dir, "s1", ("s1a", "verdict"))
    s2 = _read_verdict(args.out_dir, "s2", ("verdict",))
    if s1 != constants.V22_GATE_OK or s2 != constants.V22_GATE_OK:
        raise RuntimeError(f"S3 is not run: S1a {s1}, S2 {s2} (proposal §5; both must PASS)")
    admit_du = _read_verdict(args.out_dir, "s1", ("s1b", "admit_du")) == "True"
    k, k_pi, gaps = kappa_rows(data, admit_du)
    base = f0_bases(data, clip_only=False)
    kept = {p: {v: base[p][v] for v in k[p]} for p in base}

    def plus(extra: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
        return {p: {v: np.concatenate([kept[p][v], extra[p][v]], 1) for v in kept[p]}
                for p in kept}

    scores = {name: probe_scores(data, b, False, True, args)[:2]
              for name, b in ((F0, kept), (F0_K, plus(k)), (F0_KPI, plus(k_pi)))}
    out: dict[str, Any] = {
        "step": "S3",
        "bar": constants.V22_BAR_MIN,
        "admit_du": admit_du,
        "kappa_dim": int(next(iter(k["train"].values())).shape[1]),
        "value": read_pair(data, scores[F0_K], scores[F0], args),
        "order": read_pair(data, scores[F0_K], scores[F0_KPI], args),
        "sequences": {p: short_share(_seq_lengths(c), {v: g for v, g in gaps.items() if v in c.y})
                      for p, c in _corpora(data).items()},
        "shortcut_auc_kappa": shortcut_auc(k, args.seed),
    }
    value_ok, order_ok = passes(out["value"][DOTA]["delta"]), passes(out["order"][DOTA]["delta"])
    out["verdict"] = {"value": constants.V22_GATE_OK if value_ok else constants.V22_GATE_NO,
                      "order": constants.V22_GATE_OK if order_ok else constants.V22_GATE_NO,
                      "s5_eligible": value_ok and order_ok}
    return out


# ---------------------------------------------------------------------------
# read-outs
# ---------------------------------------------------------------------------
def _fmt(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _macro(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _pair_lines(title: str, block: dict[str, Any]) -> list[str]:
    d, t = block[DOTA], block[T2_VAL]
    seeds = ", ".join(f"{s}: {v:+.4f}" for s, v in block["per_seed_dota_delta"].items())
    return [
        f"| {title} | {_macro(d['a'])} | {_macro(d['b'])} | **{_fmt(d['delta'])}** | "
        f"{_fmt(d.get('f2_delta'))} | {_fmt(t['delta'])} |",
        f"| ↳ per-seed Δ (DoTA, stability only) | {seeds} | | | | |",
    ]


PAIR_HEADER = [
    "| contrast | a (DoTA-CAP-dev macro) | b | **Δ a - b** (decides) | `f_2` Δ "
    "| T2-val Δ (reported) |",
    "|---|---|---|---|---|---|",
]


def _bar(r: dict[str, Any]) -> str:
    return (f"Bar (§6.1): cluster-bootstrap 95 % low > 0 **and** mean ≥ +{r['bar']:.2f}, "
            "DoTA-CAP-dev, protocol B. DoTA-CAP (n/1397).")


def _s1_row(name: str, key: str, d: dict[str, Any], t: dict[str, Any], bold: bool) -> str:
    cell = f"**{_fmt(d[key])}**" if bold else _fmt(d[key])
    return f"| {name} | {cell} | {_fmt(t[key])} |"


def render_s1(r: dict[str, Any]) -> str:
    a, b, c = r["s1a"], r["s1b"], r["s1c"]
    d, t = a[DOTA], a[T2_VAL]
    admitted = "ADMITTED" if b["admit_du"] else "NOT admitted (CLIP only)"
    lines = [
        "# v2.2 S1 — free diagnostics (no training)", "", _bar(r), "",
        f"## S1a — change score `m5`: **{a['verdict']}** "
        "(gates the TDP branch only; S2 runs regardless)",
        "",
        "| read | DoTA-CAP-dev | T2-val (reported) |", "|---|---|---|",
        f"| AUC(`m5`) | {_macro(d['macro']['m5'])} | {_macro(t['macro']['m5'])} |",
        _s1_row("**AUC(`m5`) - 0.5**", "m5_minus_half", d, t, bold=True),
        _s1_row("**AUC(`m5`) - AUC(`m5`^π)**", "m5_minus_twin", d, t, bold=True),
        _s1_row("AUC(`s`) - 0.5 (secondary)", "s_minus_half", d, t, bold=False),
        _s1_row("AUC(`s`) - AUC(`s`^π) (secondary)", "s_minus_twin", d, t, bold=False),
        "",
        f"Printed: per-clip corr(AUC(`m5`), AUC(`p_T2`)) = {d['corr_with_p_t2']:.3f}; "
        f"ego {_macro(d['m5_ego'])}, non-ego {_macro(d['m5_non_ego'])}; "
        f"Δ vs twin ego {_fmt(d['m5_minus_twin_ego'])}, "
        f"non-ego {_fmt(d['m5_minus_twin_non_ego'])}; share of each clip's top-5 `m5` steps "
        f"inside the span {d['top5_in_span_share']:.3f}.",
        f"Sequences: DoTA {d['sequences']}; T2-val {t['sequences']}.",
        "",
        f"## S1b — Δu exit echo: Δu **{admitted}**",
        "",
        "| stream | clips | Δ (+5..+6 - +10..+15) | ±0.1 SD margin | inside? "
        "| autocorr lags 3-7 |",
        "|---|---|---|---|---|---|",
    ]
    for stream in ("u", "x"):
        e = b[stream]
        ac = ", ".join("—" if val is None else f"{val:.2f}" for val in e["autocorr"].values())
        role = " (decides)" if stream == "u" else " (printed)"
        lines.append(f"| {stream}{role} | {e['clips']} | {_fmt(e['delta'])} | "
                     f"±{e['margin']:.4f} | {e['inside_margin']} | {ac} |")
    smd = ", ".join(f"`{k}` {v:+.3f}" for k, v in c["summary_smd"].items())
    kept = "kept" if c["h6_kept"] else "WITHDRAWN"
    lines += [
        "",
        f"## S1c — clip start: early-onset prediction **{kept}**",
        "",
        f"{c['clips']} clips with normal steps 0-2. SMD (steps 0-2 vs normal steps ≥ 3): "
        f"{smd}. Printed: max |SMD| over all κ channels {c['all_channels_abs_max']:.3f}, "
        f"{c['all_channels_over_0.1']} channels above 0.1.",
    ]
    return "\n".join(lines) + "\n"


def render_s2(r: dict[str, Any]) -> str:
    lines = ["# v2.2 S2 — D6-step probe (true order vs adjacency-free block shuffle)", "",
             _bar(r), "", f"**Verdict (on `F0`): {r['verdict']}**", "", *PAIR_HEADER]
    lines += _pair_lines("`F0 = [x̃;ũ;p]`: true - shuffled (decides)", r[F0])
    lines += _pair_lines("`[x̃;p]` CLIP only: true - shuffled (printed)", r[CLIP_ONLY])
    lines += ["", f"Unshuffled short sequences (n ≤ 3): {r[F0]['unshuffled_short_sequences']}. "
              "Estimand: order finer than ≈ 1.5 s, beyond coarse position and beyond "
              "VideoMAE's 1.5 s bag; lag-2 neighbours remain."]
    return "\n".join(lines) + "\n"


def render_s3(r: dict[str, Any]) -> str:
    v = r["verdict"]
    du = "admitted" if r["admit_du"] else "not admitted"
    lines = ["# v2.2 S3 — κ-probe", "", _bar(r), "",
             f"κ width {r['kappa_dim']} (Δu {du}). **VALUE {v['value']} · ORDER {v['order']} "
             f"· S5 eligible: {v['s5_eligible']}**", "", *PAIR_HEADER]
    lines += _pair_lines("VALUE: `F0+K` - `F0`", r["value"])
    lines += _pair_lines("ORDER: `F0+K` - `F0+K^π`", r["order"])
    lines += ["", f"Printed: DADA↔DoTA shortcut AUC on κ {r['shortcut_auc_kappa']:.3f}. "
              f"Sequences: {r['sequences']}."]
    return "\n".join(lines) + "\n"


RENDERERS = {"s1": render_s1, "s2": render_s2, "s3": render_s3}


def render(step: str, r: dict[str, Any]) -> str:
    return RENDERERS[step](r)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
RUNNERS = {"s1": run_s1, "s2": run_s2, "s3": run_s3}


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = load_data(args)
    readout = RUNNERS[args.step](data, args)
    readout["plan"] = ".project/plans/katvad-v2.2-order-diagnostics.md"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / readout_json(args.step), readout)
    write_text_atomic(args.out_dir / readout_md(args.step), render(args.step, readout))
    LOGGER.info("%s written to %s", args.step, args.out_dir)
    return readout


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="the v2 dataset dir (T2-train - T2-val windows)")
    parser.add_argument("--annotation", type=Path, required=True, help="dada标注.xlsx")
    parser.add_argument("--census", type=Path, nargs="+", required=True,
                        help="T2 census JSON(s) (frame counts per source)")
    parser.add_argument("--t2-clip-dir", type=Path, required=True,
                        help="cache/clip/DADA2000_orig (s8, per source)")
    parser.add_argument("--video-root", type=Path, default=constants.VIDEO_CACHE_DIR)
    parser.add_argument("--dota-s1-dir", type=Path, required=True, help="cache/clip/DoTA_s1_ncc")
    parser.add_argument("--metadata", type=Path, required=True, help="DoTA metadata_val.json")
    parser.add_argument("--split-file", type=Path, required=True, help="DoTA val_split.txt")
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(constants.V22_PROBE_SEEDS))
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap / prior seed")
    parser.add_argument("--resamples", type=int, default=constants.V22_BOOTSTRAP)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
