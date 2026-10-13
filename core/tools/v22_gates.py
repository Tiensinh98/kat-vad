"""v2.2 round 7, batch 1: the R0 root-cause diagnostics and the K1 global-flow kill-switch.

Pre-registered in ``core/docs/v2.2/KAT_VAD_PROPOSAL_v2.2.md`` §R7 (D1-D13), §2.7, §5 and §8, and
``.project/plans/katvad-v2.2-two-tracks.md`` (G1-G9). Nothing is trained except probes; nothing here
adopts a model. The only rules applied are the ΔT demotion rules of §5.3 (eligibility, futility).

**ΔT** (onset-transition macro AUC, §5.2) is always read on the **step grid** (D12): per clip, the
positives are the steps of the first 2 s of the span, the negatives the normal steps in
``[t_s - 3 s, t_s - 0.5 s)``; a clip is eligible with >= 2 positives and >= 3 negatives. Step
times assume DoTA = 10 fps (A1') and DADA = 30 fps (A1); both are unverified (D1).

* ``r0-saved`` -- local, saved protocol-B scores only (no features): ΔT of the E3 / E4-0 arms and
  ``p_T2`` on DoTA-CAP-dev, eligibility, the A3-ign half of the futility rule, per CAP rate group
  (D7). Saved native scores are read at ``native[::3]``, which are exactly the step scores
  (``rate_matched_eval.to_native`` interpolates through them).
* ``r0-probe`` -- Colab: ΔT of ``m5`` / ``s`` (R0-a), of the S3 probes ``F0`` and ``F0 + κ`` (R0-b,
  the probe half of the futility rule), the ``p``-only probe (R0-e) and the precursor share of
  ``m5``'s top steps outside the span (R0-d). R0-c moved to the M0-R batch (D11); R0-f closed (D2).
* ``k1`` -- Colab: the S2 probe on ``F0`` vs ``F0 + stats23`` (flow/v1 raw RAFT statistics, shifted
  one step later so row ``t`` is the pair ``(t-1, t)``: D6), trained on T2-train - T2-val and read
  on T2-val only (span and ΔT). Soft stop (D10): both Δ points < +0.005.

CLI::

    python -m core.tools.v22_gates r0-saved \\
        --scores A3ign=outputs/v2/v2_e4_0/REPORTS/pb/A3ign/dota_cap_dev/clip_scores.npz \\
                 A3=outputs/v2/v2_e3/REPORTS/pb/A3/dota_cap_dev/clip_scores.npz ... \\
        --prior outputs/v2/v2_e3/REPORTS/position_prior/dota_cap_dev/position_prior.npz \\
        --alignment outputs/v2/REPORTS/dota_cap/dota_cap_alignment.json \\
        --out-dir outputs/v2/REPORTS/v22_gates
    python -m core.tools.v22_gates r0-probe <order_diagnostics arguments> --out-dir ...
    python -m core.tools.v22_gates k1 <order_diagnostics arguments> \\
        --flow-dir cache/flow/v1/DADA2000_orig
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core import constants
from core.data.v2_splits import dota_group, load_split
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools import order_diagnostics as od
from core.tools import order_features as of
from core.tools import order_probe as op
from core.tools.final_readout import zscore
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import read_clip_scores

LOGGER = logging.getLogger(__name__)

R0_SAVED, R0_PROBE, K1 = "r0-saved", "r0-probe", "k1"
STEPS = (R0_SAVED, R0_PROBE, K1)
A3IGN, PRIOR, F0, F0_K, P_ONLY, F0_FLOW = "A3ign", "p_T2", "F0", "F0+K", "p_only", "F0+stats23"
OTHER_GROUP = "other"
DESCRIPTION = "v2.2 round 7, batch 1: R0 diagnostics and the K1 global-flow kill-switch"


def readout_json(step: str) -> str:
    return f"{step.replace('-', '_')}_readout.json"


def readout_md(step: str) -> str:
    return f"{step.replace('-', '_')}_readout.md"


# ---------------------------------------------------------------------------
# ΔT (pure)
# ---------------------------------------------------------------------------
def onset_sets(labels: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Positive and negative step indices of one sequence (§5.2), or ``None`` if ineligible.

    ``t_s`` is the first abnormal step's time, ``t_e`` the end of the last abnormal step.
    """
    lab = np.asarray(labels).astype(bool)
    abnormal = np.flatnonzero(lab)
    if len(abnormal) == 0 or len(abnormal) == len(lab):
        return None
    tau = np.arange(len(lab)) * dt
    t_s, t_e = abnormal[0] * dt, (abnormal[-1] + 1) * dt
    far, near = constants.V22_ONSET_NEG_SECONDS
    pos = np.flatnonzero((tau >= t_s) & (tau < min(t_s + constants.V22_ONSET_POS_SECONDS, t_e)))
    neg = np.flatnonzero(~lab & (tau >= t_s - far) & (tau < t_s - near))
    if len(pos) < constants.V22_ONSET_MIN_POS or len(neg) < constants.V22_ONSET_MIN_NEG:
        return None
    return pos, neg


def onset_aucs(
    scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], dt: float
) -> dict[str, float]:
    """Per eligible sequence: AUC of the scores on its ΔT steps."""
    out: dict[str, float] = {}
    for v in sorted(scores):
        sets = onset_sets(labels[v], dt)
        if sets is None:
            continue
        idx = np.concatenate(sets)
        target = np.concatenate([np.ones(len(sets[0])), np.zeros(len(sets[1]))])
        out[v] = frame_auc(np.asarray(scores[v], dtype=np.float64)[idx], target)
    return out


def detrended_onset_aucs(
    scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], dt: float
) -> dict[str, float]:
    """ΔT on the residual of a line fitted to the negatives' scores over time (G10, reported).

    R0-saved found ΔT(``p_T2``) = 0.970: positives always follow negatives in time, so any score
    rising through the onset wins ΔT. Removing the pre-onset linear trend leaves only the excess
    at the onset over where the score was already heading; a linear ramp scores 0.5.
    """
    out: dict[str, float] = {}
    for v in sorted(scores):
        sets = onset_sets(labels[v], dt)
        if sets is None:
            continue
        pos, neg = sets
        y = np.asarray(scores[v], dtype=np.float64)
        slope, intercept = np.polyfit(neg * dt, y[neg], 1)
        idx = np.concatenate([pos, neg])
        resid = y[idx] - (intercept + slope * idx * dt)
        resid = np.round(resid, constants.V22_DETREND_DECIMALS)  # a ramp's float noise is a tie
        target = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        out[v] = frame_auc(resid, target)
    return out


def fused_onset_aucs(
    scores: dict[str, np.ndarray], prior: dict[str, np.ndarray], labels: dict[str, np.ndarray],
    dt: float,
) -> dict[str, float]:
    """``f_2``-ΔT (G3): ``z(score) + 2 z(p_T2)``, z over the whole clip, AUC on its ΔT steps."""
    w = constants.V22_FUSION_WEIGHT
    fused = {v: zscore(scores[v]) + w * zscore(prior[v]) for v in scores if v in prior}
    return onset_aucs(fused, labels, dt)


def step_grid(native: dict[str, np.ndarray], stride: int) -> dict[str, np.ndarray]:
    """Protocol-B native curves back on the step grid (D12)."""
    return {v: np.asarray(a)[::stride] for v, a in native.items()}


def rate_groups(alignment: dict[str, Any], ids: list[str]) -> dict[str, str]:
    """CAP alignment rate group of each clip: ``r1`` / ``r3``, else ``other`` (D7)."""
    out: dict[str, str] = {}
    for v in ids:
        rate = round(float(alignment["clips"][v]["rate"]))
        out[v] = f"r{rate}" if rate in constants.V22_CAP_RATE_GROUPS else OTHER_GROUP
    return out


def demotion(eligible: int, onset_a3ign: float | None, onset_f0: float | None) -> dict[str, Any]:
    """§5.3: ΔT is demoted to "reported" on too few eligible clips or on futility."""
    reasons = []
    if eligible < constants.V22_ONSET_MIN_ELIGIBLE:
        reasons.append(f"eligible {eligible} < {constants.V22_ONSET_MIN_ELIGIBLE}")
    for name, value in ((A3IGN, onset_a3ign), (F0, onset_f0)):
        if value is not None and value >= constants.V22_ONSET_FUTILITY:
            reasons.append(f"ΔT({name}) {value:.4f} >= {constants.V22_ONSET_FUTILITY}")
    complete = onset_a3ign is not None and onset_f0 is not None
    return {"demoted": bool(reasons), "reasons": reasons, "complete": complete}


def _ci(values: dict[str, float], groups: dict[str, str], args: argparse.Namespace) -> Any:
    return cluster_bootstrap_ci(values, groups, args.resamples, args.seed, constants.V22_CI)


def _sub(values: dict[str, float], keep: set[str]) -> dict[str, float]:
    return {v: a for v, a in values.items() if v in keep}


def onset_block(
    scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], prior: dict[str, np.ndarray],
    groups: dict[str, str], rates: dict[str, str] | None, dt: float, args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, float]]:
    """ΔT, ``f_2``-ΔT, span macro, and (DoTA) ΔT per CAP rate group."""
    onset = onset_aucs(scores, labels, dt)
    span = od.clip_aucs(scores, labels)
    out: dict[str, Any] = {
        "onset": _ci(onset, groups, args),
        "span": _ci(span, groups, args),
        "f2_onset": (_ci(fused_onset_aucs(scores, prior, labels, dt), groups, args)
                     if prior else None),
        "onset_detrended": _ci(detrended_onset_aucs(scores, labels, dt), groups, args),
    }
    if rates is not None:
        out["onset_by_rate"] = {g: _ci(_sub(onset, {v for v, r in rates.items() if r == g}),
                                       groups, args) for g in sorted(set(rates.values()))}
    return out, onset


# ---------------------------------------------------------------------------
# r0-saved (local)
# ---------------------------------------------------------------------------
def parse_scores(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--scores expects ARM=path, got {item!r}")
        out[name] = Path(path)
    if A3IGN not in out:
        raise ValueError(f"--scores must include {A3IGN} (the futility rule reads it)")
    return out


def run_r0_saved(args: argparse.Namespace) -> dict[str, Any]:
    ids = load_split(constants.V2_SPLIT_DOTA_CAP_DEV, args.split_dir)
    stride = constants.V22_DOTA_STRIDE
    prior_native, prior_labels = read_clip_scores(args.prior)
    labels = step_grid(prior_labels, stride)
    prior = step_grid(prior_native, stride)
    arms: dict[str, dict[str, np.ndarray]] = {}
    for name, path in parse_scores(args.scores).items():
        native, native_labels = read_clip_scores(path)
        if set(native) != set(ids):
            raise ValueError(f"{name}: {path} does not cover DoTA-CAP-dev exactly")
        bad = [v for v in ids if not np.array_equal(native_labels[v], prior_labels[v])]
        if bad:
            raise ValueError(f"{name}: labels differ from the prior's on {bad[:3]}")
        arms[name] = step_grid(native, stride)
    groups = {v: dota_group(v) for v in ids}
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    rates = rate_groups(alignment, ids)
    dt = constants.V22_DOTA_STEP_SECONDS
    reads: dict[str, Any] = {}
    onsets: dict[str, dict[str, float]] = {}
    for name, scores in {**arms, PRIOR: prior}.items():
        reads[name], onsets[name] = onset_block(
            scores, labels, {} if name == PRIOR else prior, groups, rates, dt, args)
    contrasts = {
        f"{a} - {b}": _ci(od.diff(onsets[a], onsets[b]), groups, args)
        for a, b in (("A3ign", "A3"), ("A3", "A0"), ("A3ign", PRIOR), ("A3", "A1"))
        if a in onsets and b in onsets
    }
    eligible = len(onsets[A3IGN])
    a3ign = reads[A3IGN]["onset"]
    return {
        "step": "R0-saved",
        "set": "DoTA-CAP-dev (n/1397), protocol B, step grid native[::3]",
        "step_seconds": dt,
        "rate_group_counts": {g: sum(r == g for r in rates.values())
                              for g in sorted(set(rates.values()))},
        "eligible": eligible,
        "eligible_by_rate": {g: len(_sub(onsets[A3IGN], {v for v, r in rates.items() if r == g}))
                             for g in sorted(set(rates.values()))},
        "reads": reads,
        "onset_contrasts": contrasts,
        "demotion_partial": demotion(eligible, None if a3ign is None else a3ign["mean"], None),
    }


# ---------------------------------------------------------------------------
# r0-probe and k1 (Colab)
# ---------------------------------------------------------------------------
def _dota_scores(scores: dict[str, np.ndarray], data: od.Data) -> dict[str, np.ndarray]:
    return {v: scores[v] for v in data.dota.y if v in scores}


def precursor_shares(
    m5: dict[str, np.ndarray], labels: dict[str, np.ndarray], dt: float
) -> dict[str, Any]:
    """R0-d: where a clip's top-5 ``m5`` steps fall when they are outside the span."""
    counts = {"precursor": 0, "earlier": 0, "after": 0}
    for v, s in m5.items():
        lab = labels[v].astype(bool)
        abnormal = np.flatnonzero(lab)
        if len(abnormal) == 0:
            continue
        start, end = abnormal[0], abnormal[-1]
        for t in np.argsort(-s, kind="stable")[: constants.V22_TOP_STEPS]:
            if lab[t]:
                continue
            if t > end:
                counts["after"] += 1
            elif (start - t) * dt <= constants.V22_PRECURSOR_SECONDS:
                counts["precursor"] += 1
            else:
                counts["earlier"] += 1
    total = sum(counts.values())
    return {"outside_steps": total,
            "share": {k: (c / total if total else None) for k, c in counts.items()}}


def run_r0_probe(data: od.Data, args: argparse.Namespace) -> dict[str, Any]:
    dt = constants.V22_DOTA_STEP_SECONDS
    rates = rate_groups(json.loads(args.alignment.read_text(encoding="utf-8")), sorted(data.dota.y))
    train = od.long_rows(data.train.x)
    stats = of.fit_kappa_stats(of.fit_stats(train), train)
    m5: dict[str, np.ndarray] = {}
    s: dict[str, np.ndarray] = {}
    for v, x in data.dota.x.items():
        parts = of.change_parts(x, stats)
        m5[v], s[v] = parts["m5"][:, 0], parts["s"][:, 0]
    k, _, _ = od.kappa_rows(data, admit_du=False)
    base = od.f0_bases(data, clip_only=False)
    kept = {p: {v: base[p][v] for v in k[p]} for p in base}
    plus_k = {p: {v: np.concatenate([kept[p][v], k[p][v]], 1) for v in kept[p]} for p in kept}
    empty = {p: {v: np.zeros((len(r), 0)) for v, r in kept[p].items()} for p in kept}
    probes = {name: od.probe_scores(data, b, False, True, args)[0]
              for name, b in ((F0, kept), (F0_K, plus_k), (P_ONLY, empty))}
    reads: dict[str, Any] = {}
    onsets: dict[str, dict[str, float]] = {}
    named = {"m5": m5, "s": s, **{n: _dota_scores(p, data) for n, p in probes.items()}}
    for name, scores in named.items():
        reads[name], onsets[name] = onset_block(
            scores, data.dota.y, data.prior, data.dota.group, rates, dt, args)
    t2 = {name: onset_block({v: p[v] for v in data.val.y}, data.val.y, {}, data.val.group,
                            None, constants.V22_T2_STEP_SECONDS, args)[0]
          for name, p in probes.items() if name != P_ONLY}
    saved = _saved_onset(args.out_dir)
    f0 = reads[F0]["onset"]
    return {
        "step": "R0-probe",
        "reads_dota": reads,
        "reads_t2_val": t2,
        "r0b_kappa_onset": _ci(od.diff(onsets[F0_K], onsets[F0]), data.dota.group, args),
        "r0d_precursor": precursor_shares(m5, data.dota.y, dt),
        "eligible": len(onsets[F0]),
        "demotion": demotion(len(onsets[F0]), saved, None if f0 is None else f0["mean"]),
    }


def _saved_onset(out_dir: Path) -> float | None:
    path = out_dir / readout_json(R0_SAVED)
    if not path.exists():
        LOGGER.warning("%s missing: the futility rule is completed without A3-ign", path)
        return None
    read = json.loads(path.read_text(encoding="utf-8"))["reads"][A3IGN]["onset"]
    return None if read is None else float(read["mean"])


def causal_stats(stats: np.ndarray) -> np.ndarray:
    """D6: flow/v1 row ``i`` is the pair ``(i, i+1)``; shift so row ``t`` is ``(t-1, t)``."""
    return np.concatenate([stats[:1], stats[:-1]]) if len(stats) > 1 else stats


def load_stats(flow_dir: Path, ids: list[str], lengths: dict[str, int]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for v in ids:
        path = flow_dir / f"{v}{constants.FLOW_STATS_SUFFIX}"
        if not path.exists():
            raise FileNotFoundError(f"no flow/v1 statistics for {v}: {path}")
        raw = np.load(path).astype(np.float64)
        if raw.shape != (lengths[v], constants.FLOW_STATS_DIM):
            raise ValueError(f"{v}: stats {raw.shape} vs {lengths[v]} CLIP rows")
        out[v] = causal_stats(raw)
    return out


def _val_scores(
    data: od.Data, bases: dict[str, dict[str, np.ndarray]], args: argparse.Namespace
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    seqs = {p: {v: op.Sequence(r, np.arange(len(r)), y[v]) for v, r in bases[p].items()}
            for p, y in (("train", data.train.y), ("val", data.val.y))}
    train = [(seqs["train"][s], a, b) for s, a, b in data.windows if s in seqs["train"]]
    return op.seed_averaged_scores(train, seqs["val"], True, tuple(args.seeds),
                                   torch.device(args.device))


def run_k1(data: od.Data, args: argparse.Namespace) -> dict[str, Any]:
    dt = constants.V22_T2_STEP_SECONDS
    base = od.f0_bases(data, clip_only=False)
    stats = {p: load_stats(args.flow_dir, sorted(c.x), {v: len(x) for v, x in c.x.items()})
             for p, c in (("train", data.train), ("val", data.val))}
    bases = {p: base[p] for p in ("train", "val")}
    flow = {p: {v: np.concatenate([bases[p][v], stats[p][v]], 1) for v in bases[p]} for p in bases}
    a, a_seed = _val_scores(data, flow, args)
    b, b_seed = _val_scores(data, bases, args)
    y, g = data.val.y, data.val.group
    out: dict[str, Any] = {"step": "K1", "set": "T2-val (per source)", "step_seconds": dt}
    for name, reader in (("span", od.clip_aucs),
                         ("onset", lambda sc, lab: onset_aucs(sc, lab, dt))):
        out[name] = {
            "a": _ci(reader(a, y), g, args), "b": _ci(reader(b, y), g, args),
            "delta": _ci(od.diff(reader(a, y), reader(b, y)), g, args),
            "per_seed_delta": {int(sd): float(np.mean(list(od.diff(
                reader(a_seed[sd], y), reader(b_seed[sd], y)).values()))) for sd in a_seed},
        }
    points = [out[n]["delta"]["mean"] for n in ("span", "onset") if out[n]["delta"] is not None]
    soft = bool(points) and all(p < constants.V22_K1_SOFT_STOP for p in points)
    out["soft_stop"] = soft
    return out


# ---------------------------------------------------------------------------
# read-outs
# ---------------------------------------------------------------------------
def _f(c: dict[str, float] | None, signed: bool = False) -> str:
    if c is None:
        return "—"
    fmt = "{:+.4f}" if signed else "{:.4f}"
    return f"{fmt.format(c['mean'])} [{fmt.format(c['low'])}, {fmt.format(c['high'])}]"


def render_r0_saved(r: dict[str, Any]) -> str:
    groups = list(r["rate_group_counts"])
    lines = [
        "# v2.2 R0-saved — ΔT on saved protocol-B scores (DoTA-CAP-dev, descriptive)", "",
        f"{r['set']}; step {r['step_seconds']:.2f} s (A1'). Eligible clips: **{r['eligible']}** "
        f"(by rate group {r['eligible_by_rate']}; clips per group {r['rate_group_counts']}).", "",
        "| score | span macro | **ΔT** | ΔT detrended | `f_2`-ΔT | "
        + " | ".join(f"ΔT {g}" for g in groups) + " |",
        "|---|---|---|---|---|" + "---|" * len(groups),
    ]
    for name, read in r["reads"].items():
        by = " | ".join(_f(read["onset_by_rate"].get(g)) for g in groups)
        lines.append(f"| {name} | {_f(read['span'])} | **{_f(read['onset'])}** | "
                     f"{_f(read['onset_detrended'])} | {_f(read['f2_onset'])} | {by} |")
    lines += ["", "ΔT contrasts (paired by clip, descriptive):", ""]
    lines += [f"- {k}: {_f(c, signed=True)}" for k, c in r["onset_contrasts"].items()]
    d = r["demotion_partial"]
    verdict = "DEMOTED" if d["demoted"] else "not demoted by this half"
    lines += ["", f"**Demotion rule (§5.3), A3-ign half: {verdict}** {d['reasons']}. "
              "The `F0`-probe half is read by `r0-probe`."]
    return "\n".join(lines) + "\n"


def render_r0_probe(r: dict[str, Any]) -> str:
    lines = ["# v2.2 R0-probe — ΔT of change scores and probes (DoTA-CAP-dev, descriptive)", "",
             f"Eligible clips: {r['eligible']}.", "",
             "| score | span macro | **ΔT** | ΔT detrended | `f_2`-ΔT |", "|---|---|---|---|---|"]
    for name, read in r["reads_dota"].items():
        lines.append(f"| {name} | {_f(read['span'])} | **{_f(read['onset'])}** | "
                     f"{_f(read['onset_detrended'])} | {_f(read['f2_onset'])} |")
    lines += ["", f"R0-b ΔT(`F0+κ`) - ΔT(`F0`): {_f(r['r0b_kappa_onset'], signed=True)}",
              f"R0-d top-5 `m5` steps outside the span: {r['r0d_precursor']}", "",
              "T2-val (per source, A1 step 0.267 s): " + "; ".join(
                  f"{n} span {_f(t['span'])}, ΔT {_f(t['onset'])}"
                  for n, t in r["reads_t2_val"].items())]
    d = r["demotion"]
    state = "DEMOTED" if d["demoted"] else "NOT demoted"
    lines += ["", f"**ΔT demotion rule (§5.3): {state}** {d['reasons']} "
              f"(both halves read: {d['complete']})."]
    return "\n".join(lines) + "\n"


def render_k1(r: dict[str, Any]) -> str:
    lines = ["# v2.2 K1 — global flow (flow/v1 stats23, causal) on T2-val", "",
             "Probe `F0 + stats23` (a) vs `F0` (b), trained on T2-train - T2-val, read per T2-val "
             "source. stats23 pairs are 0.27 s apart (A1), shifted causal (D6).", "",
             "| read | a | b | **Δ a - b** | per-seed Δ |", "|---|---|---|---|---|"]
    for name in ("span", "onset"):
        x = r[name]
        seeds = ", ".join(f"{s}: {v:+.4f}" for s, v in x["per_seed_delta"].items())
        lines.append(f"| {name} | {_f(x['a'])} | {_f(x['b'])} | "
                     f"**{_f(x['delta'], True)}** | {seeds} |")
    stop = ("SOFT STOP: full M0 needs a written advisor go-ahead (weak evidence, D10)"
            if r["soft_stop"] else "no soft stop")
    return "\n".join([*lines, "", f"**{stop}.**"]) + "\n"


RENDER = {R0_SAVED: render_r0_saved, R0_PROBE: render_r0_probe, K1: render_k1}


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.step == R0_SAVED:
        result = run_r0_saved(args)
    else:
        data = od.load_data(args)
        result = run_r0_probe(data, args) if args.step == R0_PROBE else run_k1(data, args)
    write_json_atomic(args.out_dir / readout_json(args.step), result)
    write_text_atomic(args.out_dir / readout_md(args.step), RENDER[args.step](result))
    LOGGER.info("%s written to %s", args.step, args.out_dir)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap / prior seed")
    parser.add_argument("--resamples", type=int, default=constants.V22_BOOTSTRAP)
    parser.add_argument("--alignment", type=Path,
                        help="dota_cap_alignment.json (r0-saved, r0-probe)")
    saved = parser.add_argument_group("r0-saved")
    saved.add_argument("--scores", nargs="+", default=[], help="ARM=clip_scores.npz (DoTA-CAP-dev)")
    saved.add_argument("--prior", type=Path, help="position_prior.npz (DoTA-CAP-dev)")
    probe = parser.add_argument_group("r0-probe / k1 (order_diagnostics inputs)")
    probe.add_argument("--data-dir", type=Path)
    probe.add_argument("--annotation", type=Path)
    probe.add_argument("--census", type=Path, nargs="+")
    probe.add_argument("--t2-clip-dir", type=Path)
    probe.add_argument("--video-root", type=Path, default=constants.VIDEO_CACHE_DIR)
    probe.add_argument("--dota-s1-dir", type=Path)
    probe.add_argument("--metadata", type=Path)
    probe.add_argument("--split-file", type=Path)
    probe.add_argument("--seeds", type=int, nargs="+", default=list(constants.V22_PROBE_SEEDS))
    probe.add_argument("--device", default="cpu")
    probe.add_argument("--flow-dir", type=Path, help="k1: cache/flow/v1/DADA2000_orig")
    return parser


REQUIRED = {
    R0_SAVED: ("scores", "prior", "alignment"),
    R0_PROBE: ("data_dir", "annotation", "census", "t2_clip_dir", "dota_s1_dir", "metadata",
               "split_file", "alignment"),
    K1: ("data_dir", "annotation", "census", "t2_clip_dir", "dota_s1_dir", "metadata",
         "split_file", "flow_dir"),
}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    missing = [k for k in REQUIRED[args.step] if not getattr(args, k)]
    if missing:
        parser.error(f"{args.step} needs --{', --'.join(m.replace('_', '-') for m in missing)}")
    run(args)


if __name__ == "__main__":
    main()
