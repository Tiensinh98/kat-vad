"""v7 Step 0 -- does a normal-dynamics predictor in V2-S space beat fixed rules? (no detector).

Rules: ``core/docs/v7/PREREG_V7.md`` §1 (signed before any number). Plan
``.project/plans/katvad-v7-lsp.md`` §4.

``d1``    cos(u_t, u_{t+k}), k = 1..40 lattice steps, on the 404 training negatives (pre-training).
``train`` LSP stage 1 for every K in ``V7_CONTEXTS`` on the 404 training negatives, best epoch by
          held-out L_rec on the 45 frozen held-out negatives; K chosen once; ŝ statistics.
``diag``  D2 (held-out L_rec vs fixed predictors), D3 (``nexar_val`` video AUC; DoTA-CAP-dev
          per-clip frame AUC), D4 (source classifier), D5 (lead-in maxima, corr) -> GO / NO-GO.

Inputs are caches that already exist (plan §3): V2-S ``fc_norm(mean)`` at every native frame,
``Nexar_s1_squash`` (30 fps, lattice step 3) and ``DoTA_CAP_s1_squash`` (10 fps, step 1).
``u = (v - b) / gamma`` with V2-S's ``fc_norm`` read from the pinned encoder weights (X5).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from core import constants
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_splits import dota_group, lines_sha1, load_split
from core.device import resolve_device
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.models.lsp import (
    LatentSurprisePredictor,
    SurpriseSeries,
    clip_unit_vectors,
    first_target,
    parameter_count,
    row_loss,
    surprise_series,
    unit_vectors,
)
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.rate_matched_eval import native_labels
from core.tools.v5_stage0 import load_boxes
from core.tools.v5_tile_read import fit_position, position_scores

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/v7/PREREG_V7.md §1"
D1_JSON, D1_MD = "d1.json", "d1.md"
TRAIN_JSON = "train.json"
SELECTED_JSON = "lsp_selected.json"
READOUT_JSON, READOUT_MD = "step0_readout.json", "step0_readout.md"
FIXED = ("copy_last", "const_velocity", "context_mean")
D4_MIN_STEPS = 8  # a DoTA lead-in enters D4 only with at least this many valid lattice steps
D4_FEATURES = ("mean_s", "std_s", "max_s", "mean_rho", "max_rho")


def rule_of(stream: str) -> str:
    """The section that governs a stream's read: §1 (V2-S) or §1.1 (CLIP, Amendment 1)."""
    return RULE if stream == constants.V7_STREAM_V2S else f"{RULE}.1 (Amendment 1)"


def step0_dir(stream: str = constants.V7_STREAM_V2S) -> Path:
    root = constants.OUTPUT_ROOT / constants.V7_OUTPUT_DIRNAME / constants.V7_LSP_DIRNAME
    return root / constants.V7_STREAM_OUT[stream]


def default_cache_dirs(stream: str) -> tuple[Path, Path]:
    """``(Nexar s1 dir, DoTA s1 dir)`` of a stream under ``CACHE_ROOT``."""
    if stream == constants.V7_STREAM_CLIP:
        clip = constants.CLIP_CACHE_DIR
        return clip / constants.NEXAR_CLIP_S1_DATASET, clip / f"{constants.DOTA_DATASET}_s1_ncc"
    v2s = constants.VIDEO_CACHE_DIR / constants.V7_ENCODER
    return (
        v2s / f"{constants.NEXAR_VIDEO_S1_DATASET}_squash",
        v2s / f"{constants.DOTA_CAP_DATASET}_s1_squash",
    )


def model_path(models_dir: Path, context: int) -> Path:
    return models_dir / f"lsp_K{context}.pt"


# --------------------------------------------------------------------------- inputs


def fc_norm_params(encoder: str, weights: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """``(gamma, b)`` of V2-S's ``fc_norm`` from the pinned weights (strict load, C5)."""
    from core.models.videomae_v2 import load_pretrained  # heavy; only the CLI needs it

    model = load_pretrained(encoder, torch.device("cpu"), weights)
    return (
        model.fc_norm.weight.detach().numpy().astype(np.float32),
        model.fc_norm.bias.detach().numpy().astype(np.float32),
    )


def lattices(u_native: np.ndarray, step: int) -> list[np.ndarray]:
    """Every phase of the 10 fps lattice of one video: ``u[phase::step]``."""
    return [np.ascontiguousarray(u_native[phase::step]) for phase in range(step)]


@dataclass
class NexarSets:
    train: list[str]  # training negatives (404)
    holdout: list[str]  # frozen held-out negatives (45)
    val: list[str]  # nexar_val, both labels
    labels: dict[str, int]


def nexar_sets(census: Path, split_dir: Path, holdout_file: Path) -> NexarSets:
    """Training / held-out negatives and ``nexar_val``; held-out = the frozen file."""
    rows: dict[str, dict[str, Any]] = json.loads(census.read_text(encoding="utf-8"))["videos"]
    train = load_split(constants.V2_SPLIT_NEXAR_TRAIN, split_dir)
    val = load_split(constants.V2_SPLIT_NEXAR_VAL, split_dir)
    neg = sorted(v for v in train if int(rows[v]["label"]) == 0)
    n_hold = round(constants.V7_HOLDOUT_FRACTION * len(neg))
    drawn = sorted(np.random.default_rng(constants.V7_SEED).permutation(neg)[:n_hold].tolist())
    frozen = [x.strip() for x in holdout_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if drawn != sorted(frozen):
        raise ValueError(f"{holdout_file} differs from the seeded draw over nexar_train negatives")
    hold = set(drawn)
    return NexarSets(
        train=[v for v in neg if v not in hold],
        holdout=drawn,
        val=val,
        labels={v: int(rows[v]["label"]) for v in train + val},
    )


Loader = Callable[[str], np.ndarray]
Unit = Callable[[np.ndarray], np.ndarray]


def stream_unit(args: argparse.Namespace) -> Unit:
    """Cached rows -> ``u``: V2-S inverts ``fc_norm`` (L0-2), CLIP takes ``LN_no-affine`` (§1.1)."""
    if args.stream == constants.V7_STREAM_CLIP:
        return clip_unit_vectors
    gamma, bias = fc_norm_params(args.encoder, args.weights)
    return lambda cached: unit_vectors(cached, gamma, bias)


def cache_loader(video_dir: Path, unit: Unit) -> Loader:
    def load(video_id: str) -> np.ndarray:
        return unit(np.load(video_dir / f"{video_id}.npy"))

    return load


def nexar_sequences(ids: list[str], load: Loader) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for v in ids:
        out.extend(lattices(load(v), constants.V7_NEXAR_LATTICE_STEP))
    return out


# --------------------------------------------------------------------------- D1


def lag_cosines(seqs: list[np.ndarray], max_lag: int) -> dict[int, float]:
    """Mean cos(u_t, u_{t+k}) pooled over every pair of every sequence."""
    sums = np.zeros(max_lag + 1)
    counts = np.zeros(max_lag + 1)
    for u in seqs:
        unit = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), constants.V7_EPS)
        for k in range(1, min(max_lag, len(u) - 1) + 1):
            c = np.sum(unit[:-k] * unit[k:], axis=1)
            sums[k] += c.sum()
            counts[k] += len(c)
    return {k: float(sums[k] / counts[k]) for k in range(1, max_lag + 1) if counts[k]}


def d1_verdict(curve: dict[int, float]) -> dict[str, Any]:
    at_h, at_1 = curve[constants.V7_HORIZON], curve[1]
    failed = at_h >= at_1 - constants.V7_D1_FAIL_DELTA
    return {"cos_k1": at_1, "cos_horizon": at_h, "failed": bool(failed)}


def run_d1(args: argparse.Namespace) -> dict[str, Any]:
    sets = nexar_sets(args.census, args.split_dir, args.holdout_file)
    load = cache_loader(args.video_dir, stream_unit(args))
    seqs = [lattices(load(v), constants.V7_NEXAR_LATTICE_STEP)[0] for v in sets.train]
    curve = lag_cosines(seqs, constants.V7_D1_MAX_LAG)
    out = {
        "rule": rule_of(args.stream),
        "stream": args.stream,
        "videos": len(seqs),
        "curve": curve,
        **d1_verdict(curve),
    }
    reports = args.out_dir / "REPORTS"
    write_json_atomic(reports / D1_JSON, out)
    lines = [
        f"# v7 D1 — cos(u_t, u_{{t+k}}) on nexar_train negatives (phase 0), stream {args.stream}",
        "",
        f"Videos {out['videos']}; Δ = {constants.V7_HORIZON}. cos k=1 {out['cos_k1']:.4f}, "
        f"k=Δ {out['cos_horizon']:.4f} → **{'FAIL' if out['failed'] else 'pass'}**",
        "",
        "| k | cos |",
        "|---|---|",
        *[f"| {k} | {c:.4f} |" for k, c in curve.items()],
    ]
    write_text_atomic(reports / D1_MD, "\n".join(lines) + "\n")
    LOGGER.info("D1: %s", {k: out[k] for k in ("cos_k1", "cos_horizon", "failed")})
    return out


# --------------------------------------------------------------------------- stage 1


class Packed:
    """Sequences concatenated into one tensor; a target is addressed by its global row."""

    def __init__(self, seqs: list[np.ndarray], device: torch.device) -> None:
        lengths = [len(u) for u in seqs]
        self.offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.u = torch.as_tensor(np.concatenate(seqs), dtype=torch.float32, device=device)

    def targets(self, first: int) -> np.ndarray:
        """Global rows of every target ``t >= first`` of every sequence."""
        parts = [
            o + np.arange(first, n)
            for o, n in zip(self.offsets, self.lengths, strict=True)
            if n > first
        ]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    def sample(self, first: int, per_seq: int, rng: np.random.Generator) -> np.ndarray:
        """``per_seq`` random targets (with replacement) from each sequence that has one."""
        keep = self.lengths > first
        lo = self.offsets[keep] + first
        hi = self.offsets[keep] + self.lengths[keep]
        draws = rng.integers(lo[:, None], hi[:, None], size=(int(keep.sum()), per_seq))
        out: np.ndarray = draws.reshape(-1)
        return out

    def batch(self, rows: np.ndarray, context: int, horizon: int) -> tuple[Tensor, Tensor]:
        """Context ``(B, K, D)`` and target ``(B, D)`` for global target rows."""
        idx = torch.as_tensor(rows, device=self.u.device)
        offs = torch.arange(context, device=self.u.device) - (horizon + context - 1)
        return self.u[idx[:, None] + offs], self.u[idx]


def fixed_predictions(
    ctx: Tensor, packed: Packed, rows: np.ndarray, horizon: int
) -> dict[str, Tensor]:
    idx = torch.as_tensor(rows, device=packed.u.device)
    last = ctx[:, -1]
    return {
        "copy_last": last,
        "const_velocity": 2 * last - packed.u[idx - 2 * horizon],
        "context_mean": ctx.mean(1),
    }


@torch.no_grad()
def heldout_losses(
    model: LatentSurprisePredictor,
    packed: Packed,
    first: int = constants.V7_HELDOUT_FIRST_TARGET,
    horizon: int = constants.V7_HORIZON,
    with_fixed: bool = False,
    batch: int = constants.V7_BATCH * 4,
) -> dict[str, float]:
    """Mean L_rec over every held-out target ``t >= first`` (the same targets for every K)."""
    if first < first_target(model.context, horizon) or first < 2 * horizon:
        raise ValueError(f"first target {first} is too early for K={model.context}, Δ={horizon}")
    rows = packed.targets(first)
    names = ("lsp", *FIXED) if with_fixed else ("lsp",)
    sums = dict.fromkeys(names, 0.0)
    model.eval()
    for lo in range(0, len(rows), batch):
        r = rows[lo : lo + batch]
        ctx, target = packed.batch(r, model.context, horizon)
        preds = {"lsp": model.predict(ctx)}
        if with_fixed:
            preds |= fixed_predictions(ctx, packed, r, horizon)
        for name in names:
            sums[name] += float(row_loss(preds[name], target).sum())
    return {name: sums[name] / max(len(rows), 1) for name in names}


@dataclass
class Fit:
    model: LatentSurprisePredictor
    best_epoch: int
    best_heldout: float
    history: list[dict[str, float]]


def train_lsp(
    train: Packed,
    heldout: Packed,
    context: int,
    epochs: int = constants.V7_EPOCHS,
    seed: int = constants.V7_SEED,
    horizon: int = constants.V7_HORIZON,
    heads: int = constants.V7_HEADS,
) -> Fit:
    """Stage 1: Smooth-L1 to stop-gradient targets, AdamW + cosine, best epoch on held-out L_rec."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dim = int(train.u.shape[1])
    model = LatentSurprisePredictor(dim=dim, context=context, heads=heads).to(train.u.device)
    opt = AdamW(model.parameters(), lr=constants.V7_LR, weight_decay=constants.V7_WEIGHT_DECAY)
    first = first_target(context, horizon)
    per_epoch = math.ceil(
        int((train.lengths > first).sum()) * constants.V7_SAMPLES_PER_SEQUENCE / constants.V7_BATCH
    )
    sched = CosineAnnealingLR(opt, T_max=max(per_epoch * epochs, 1))
    best = Fit(copy.deepcopy(model), 0, heldout_losses(model, heldout)["lsp"], [])
    for epoch in range(1, epochs + 1):
        model.train()
        rows = rng.permutation(train.sample(first, constants.V7_SAMPLES_PER_SEQUENCE, rng))
        total = 0.0
        for lo in range(0, len(rows), constants.V7_BATCH):
            ctx, target = train.batch(rows[lo : lo + constants.V7_BATCH], context, horizon)
            loss = row_loss(model.predict(ctx), target.detach()).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(target)
        held = heldout_losses(model, heldout)["lsp"]
        best.history.append({"epoch": epoch, "train": total / max(len(rows), 1), "heldout": held})
        if held < best.best_heldout:
            best.model, best.best_epoch, best.best_heldout = copy.deepcopy(model), epoch, held
        LOGGER.info(
            "K=%d epoch %d train %.5f heldout %.5f", context, epoch, best.history[-1]["train"], held
        )
    best.model.eval()
    return best


def choose_context(heldout: dict[int, float]) -> int:
    """Lowest held-out L_rec; relative gap < ``V7_CONTEXT_TIE_REL`` -> smaller K."""
    ks = sorted(heldout)
    best = min(ks, key=lambda k: heldout[k])
    small = ks[0]
    if (
        best != small
        and (heldout[small] - heldout[best]) / heldout[small] < constants.V7_CONTEXT_TIE_REL
    ):
        return small
    return best


def surprise_stats(model: LatentSurprisePredictor, seqs: list[np.ndarray]) -> tuple[float, float]:
    """Mean / std of ``s`` over every valid target of the held-out negatives (ŝ standardization)."""
    vals = [ser.s[ser.valid] for ser in (surprise_series(model, u) for u in seqs)]
    s = np.concatenate([v for v in vals if len(v)])
    return float(s.mean()), float(s.std())


def save_model(path: Path, model: LatentSurprisePredictor, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    meta = {"context": model.context, "dim": model.dim, "heads": model.heads}
    torch.save({"state_dict": model.state_dict(), **meta, **extra}, tmp)
    tmp.replace(path)


def load_model(path: Path, device: torch.device) -> LatentSurprisePredictor:
    data = torch.load(path, map_location=device, weights_only=True)
    model = LatentSurprisePredictor(
        dim=int(data.get("dim", constants.V7_DIM)),  # §1 checkpoints predate the field
        context=int(data["context"]),
        heads=int(data.get("heads", constants.V7_HEADS)),
    ).to(device)
    model.load_state_dict(data["state_dict"])  # strict (C5)
    model.eval()
    return model


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    sets = nexar_sets(args.census, args.split_dir, args.holdout_file)
    load = cache_loader(args.video_dir, stream_unit(args))
    train = Packed(nexar_sequences(sets.train, load), device)
    hold_seqs = nexar_sequences(sets.holdout, load)
    heldout = Packed(hold_seqs, device)
    models_dir = args.out_dir / "models"
    fits: dict[int, dict[str, Any]] = {}
    for k in constants.V7_CONTEXTS:
        fit = train_lsp(
            train, heldout, k, epochs=args.epochs, heads=constants.V7_STREAM_HEADS[args.stream]
        )
        save_model(model_path(models_dir, k), fit.model, {"best_epoch": fit.best_epoch})
        fits[k] = {
            "best_epoch": fit.best_epoch,
            "heldout": fit.best_heldout,
            "history": fit.history,
            "params": parameter_count(fit.model),
        }
    chosen = choose_context({k: f["heldout"] for k, f in fits.items()})
    model = load_model(model_path(models_dir, chosen), device)
    s_mean, s_std = surprise_stats(model, hold_seqs)
    selected = {
        "stream": args.stream,
        "context": chosen,
        "horizon": constants.V7_HORIZON,
        "s_mean": s_mean,
        "s_std": s_std,
        "train_ids_sha1": lines_sha1(sets.train),
        "holdout_ids_sha1": lines_sha1(sets.holdout),
        "encoder": args.encoder,
        "video_dir": str(args.video_dir),
    }
    write_json_atomic(models_dir / SELECTED_JSON, selected)
    report = {
        "rule": rule_of(args.stream),
        "selected": selected,
        "fits": fits,
        "train_videos": len(sets.train),
        "holdout_videos": len(sets.holdout),
    }
    write_json_atomic(args.out_dir / "REPORTS" / TRAIN_JSON, report)
    LOGGER.info(
        "stage 1: K=%d chosen, held-out %s",
        chosen,
        {k: round(f["heldout"], 5) for k, f in fits.items()},
    )
    return report


# --------------------------------------------------------------------------- D2-D5


def d2(model: LatentSurprisePredictor, heldout: Packed) -> dict[str, Any]:
    losses = heldout_losses(model, heldout, with_fixed=True)
    best_fixed = min(FIXED, key=lambda n: losses[n])
    gain = 1.0 - losses["lsp"] / losses[best_fixed]
    return {
        "losses": losses,
        "best_fixed": best_fixed,
        "rel_gain": gain,
        "passed": bool(gain >= constants.V7_D2_MIN_REL_GAIN),
    }


def video_max(series: SurpriseSeries, key: str) -> float:
    vals = getattr(series, key)[series.valid]
    return float(vals.max()) if len(vals) else float("nan")


def paired_auc_diff(
    a: np.ndarray, b: np.ndarray, y: np.ndarray, resamples: int, seed: int, level: float
) -> dict[str, float]:
    """Video bootstrap of AUC(a) - AUC(b) on the same videos."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(resamples):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            diffs.append(frame_auc(a[i], y[i]) - frame_auc(b[i], y[i]))
    tail = (1.0 - level) / 2.0
    return {
        "mean": frame_auc(a, y) - frame_auc(b, y),
        "low": float(np.quantile(diffs, tail)),
        "high": float(np.quantile(diffs, 1.0 - tail)),
    }


def d3_nexar(series: dict[str, SurpriseSeries], labels: dict[str, int]) -> dict[str, Any]:
    ids = [v for v in sorted(series) if series[v].valid.any()]
    y = np.asarray([labels[v] for v in ids])
    s = np.asarray([video_max(series[v], "s") for v in ids])
    p = np.asarray([video_max(series[v], "p") for v in ids])
    rho = np.asarray([video_max(series[v], "rho") for v in ids])
    return {
        "videos": len(ids),
        "auc_s": frame_auc(s, y),
        "auc_p": frame_auc(p, y),
        "auc_rho": frame_auc(rho, y),
        "s_minus_p": paired_auc_diff(
            s,
            p,
            y,
            constants.V7_BOOTSTRAP,
            constants.V7_BOOTSTRAP_SEED,
            constants.V7_BOOTSTRAP_LEVEL,
        ),
    }


@dataclass
class DotaClip:
    series: SurpriseSeries
    labels: np.ndarray  # native = lattice (10 fps)
    ego: bool | None


def ci(values: dict[str, float]) -> dict[str, float] | None:
    return cluster_bootstrap_ci(
        values,
        {v: dota_group(v) for v in values},
        constants.V7_BOOTSTRAP,
        constants.V7_BOOTSTRAP_SEED,
        constants.V7_BOOTSTRAP_LEVEL,
    )


def d3_dota(clips: dict[str, DotaClip]) -> dict[str, Any]:
    coeffs = fit_position({v: c.labels for v, c in clips.items()})
    pos = position_scores({v: c.labels for v, c in clips.items()}, coeffs)
    per: dict[str, dict[str, float]] = {}
    for v, c in clips.items():
        m, y = c.series.valid, c.labels
        if not 0 < int(y[m].sum()) < int(m.sum()):
            continue
        per[v] = {
            "s": frame_auc(c.series.s[m], y[m]),
            "p": frame_auc(c.series.p[m], y[m]),
            "rho": frame_auc(c.series.rho[m], y[m]),
            "pos": frame_auc(pos[v][m], y[m]),
        }

    def subset(keep: Callable[[str], bool]) -> dict[str, Any]:
        ids = [v for v in per if keep(v)]
        return {name: ci({v: per[v][name] for v in ids}) for name in ("s", "p", "rho", "pos")} | {
            "s_minus_p": ci({v: per[v]["s"] - per[v]["p"] for v in ids}),
            "clips": len(ids),
        }

    return {
        "clips_total": len(clips),
        "clips_read": len(per),
        "all": subset(lambda v: True),
        "ego": subset(lambda v: clips[v].ego is True),
        "non_ego": subset(lambda v: clips[v].ego is False),
    }


def video_features(series: SurpriseSeries, keep: np.ndarray) -> np.ndarray | None:
    m = series.valid & keep
    if int(m.sum()) < D4_MIN_STEPS:
        return None
    s, rho = series.s[m], series.rho[m]
    return np.asarray([s.mean(), s.std(), s.max(), rho.mean(), rho.max()], dtype=np.float64)


def d4(
    nexar: dict[str, SurpriseSeries],
    labels: dict[str, int],
    clips: dict[str, DotaClip],
    anomaly_auc: float,
) -> dict[str, Any]:
    """Source classifier: ``nexar_val`` normals (0) vs DoTA-CAP-dev lead-ins (1)."""
    feats, src = [], []
    for v, ser in nexar.items():
        f = video_features(ser, np.ones_like(ser.valid)) if labels[v] == 0 else None
        if f is not None:
            feats.append(f)
            src.append(0)
    for c in clips.values():
        onset = int(np.argmax(c.labels)) if c.labels.any() else len(c.labels)
        f = video_features(c.series, np.arange(len(c.labels)) < onset)
        if f is not None:
            feats.append(f)
            src.append(1)
    x, y = np.asarray(feats), np.asarray(src)
    if min(int(y.sum()), int(len(y) - y.sum())) < constants.V7_D4_FOLDS:
        return {
            "normals": int(len(y) - y.sum()),
            "lead_ins": int(y.sum()),
            "source_auc": None,
            "gap": None,
            "flag": None,
        }
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    folds = StratifiedKFold(constants.V7_D4_FOLDS, shuffle=True, random_state=constants.V7_SEED)
    proba = np.asarray(cross_val_predict(clf, x, y, cv=folds, method="predict_proba"))[:, 1]
    auc = frame_auc(proba, y)
    return {
        "normals": int(len(y) - y.sum()),
        "lead_ins": int(y.sum()),
        "features": D4_FEATURES,
        "source_auc": auc,
        "gap": auc - anomaly_auc,
        "flag": bool(auc - anomaly_auc > constants.V7_D4_FLAG_GAP),
    }


def d5(
    nexar: dict[str, SurpriseSeries], labels: dict[str, int], clips: dict[str, DotaClip]
) -> dict[str, Any]:
    lead: dict[str, list[bool]] = {"s": [], "p": []}
    for c in clips.values():
        m = c.series.valid
        onset = int(np.argmax(c.labels)) if c.labels.any() else len(c.labels)
        if not (m & (c.labels > 0)).any() or not (m & (np.arange(len(m)) < onset)).any():
            continue
        for key in lead:
            score = np.where(m, getattr(c.series, key), -np.inf)
            lead[key].append(int(np.argmax(score)) < onset)
    s = np.concatenate([ser.s[ser.valid] for v, ser in nexar.items() if labels[v] == 0])
    p = np.concatenate([ser.p[ser.valid] for v, ser in nexar.items() if labels[v] == 0])
    return {
        "lead_in_max_share": {k: float(np.mean(x)) if x else None for k, x in lead.items()},
        "clips": len(lead["s"]),
        "corr_s_p_normals": float(np.corrcoef(s, p)[0, 1]) if len(s) > 1 else None,
    }


def verdict(
    d1_out: dict[str, Any] | None, d2_out: dict[str, Any], d3n: dict[str, Any], d3d: dict[str, Any]
) -> tuple[str, dict[str, bool]]:
    diff = d3d["all"]["s_minus_p"]
    checks = {
        "d1_not_failed": d1_out is not None and not d1_out["failed"],
        "d2": d2_out["passed"],
        "d3_nexar": d3n["auc_s"] > d3n["auc_p"],
        "d3_dota": diff is not None and diff["mean"] > 0,
    }
    return (constants.V7_VERDICT_GO if all(checks.values()) else constants.V7_VERDICT_NO_GO), checks


def load_dota_clips(
    args: argparse.Namespace, unit: Unit, model: LatentSurprisePredictor
) -> tuple[dict[str, DotaClip], list[str]]:
    dev = sorted(load_split(constants.V2_SPLIT_DOTA_CAP_DEV, args.split_dir))
    records = {
        r.video_id: r for r in parse_metadata(args.metadata, read_split_ids(args.split_file))
    }
    clips: dict[str, DotaClip] = {}
    missing: list[str] = []
    for v in dev:
        path = args.dota_video_dir / f"{v}.npy"
        if not path.exists() or v not in records:
            missing.append(v)
            continue
        u = lattices(unit(np.load(path)), constants.V7_DOTA_LATTICE_STEP)[0]
        ann = args.annotations_dir / f"{v}.json"
        clips[v] = DotaClip(
            surprise_series(model, u),
            native_labels(records[v], len(u)),
            load_boxes(ann)[1] if ann.exists() else None,
        )
    return clips, missing


def run_diag(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    models_dir = args.out_dir / "models"
    selected = json.loads((models_dir / SELECTED_JSON).read_text(encoding="utf-8"))
    model = load_model(model_path(models_dir, int(selected["context"])), device)
    sets = nexar_sets(args.census, args.split_dir, args.holdout_file)
    unit = stream_unit(args)
    load = cache_loader(args.video_dir, unit)
    d1_path = args.out_dir / "REPORTS" / D1_JSON
    d1_out = json.loads(d1_path.read_text(encoding="utf-8")) if d1_path.exists() else None
    d2_out = d2(model, Packed(nexar_sequences(sets.holdout, load), device))
    nexar = {
        v: surprise_series(model, lattices(load(v), constants.V7_NEXAR_LATTICE_STEP)[0])
        for v in sets.val
    }
    d3n = d3_nexar(nexar, sets.labels)
    clips, missing = load_dota_clips(args, unit, model)
    d3d = d3_dota(clips)
    decision, checks = verdict(d1_out, d2_out, d3n, d3d)
    readout = {
        "rule": rule_of(args.stream),
        "stream": args.stream,
        "decision": decision,
        "checks": checks,
        "selected": selected,
        "d1": d1_out and {k: d1_out[k] for k in ("cos_k1", "cos_horizon", "failed")},
        "d2": d2_out,
        "d3_nexar": d3n,
        "d3_dota": d3d,
        "d4": d4(nexar, sets.labels, clips, d3n["auc_s"]),
        "d5": d5(nexar, sets.labels, clips),
        "dota_missing": missing,
    }
    write_json_atomic(args.out_dir / "REPORTS" / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / "REPORTS" / READOUT_MD, render(readout))
    LOGGER.info("Step 0: %s %s", decision, checks)
    return readout


# --------------------------------------------------------------------------- render


def _f(x: float | None, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def _ci(stat: dict[str, float] | None, sign: bool = False) -> str:
    if stat is None:
        return "—"
    n = f" (n = {stat['clips']})" if "clips" in stat else ""
    return f"{_f(stat['mean'], sign)} [{_f(stat['low'], sign)}, {_f(stat['high'], sign)}]{n}"


def render(r: dict[str, Any]) -> str:
    sel, d2o, d3n, d3d, d4o, d5o = (
        r["selected"],
        r["d2"],
        r["d3_nexar"],
        r["d3_dota"],
        r["d4"],
        r["d5"],
    )
    lines = [
        f"# v7 Step 0 — Latent Surprise Predictor, stream {r.get('stream', 'v2s')} (no detector)",
        "",
        f"Rule: {r['rule']}. **Decision: {r['decision']}** — checks {r['checks']}",
        "",
        f"LSP: K = {sel['context']}, Δ = {sel['horizon']} (10 fps lattice); ŝ stats mean "
        f"{sel['s_mean']:.5f}, std {sel['s_std']:.5f}.",
        "",
        f"D1: {r['d1']}",
        "",
        "## D2 — held-out L_rec (45 negatives, t ≥ Δ + max K - 1)",
        "",
        "| predictor | L_rec |",
        "|---|---|",
        *[f"| {k} | {v:.5f} |" for k, v in d2o["losses"].items()],
        "",
        f"Best fixed = {d2o['best_fixed']}; relative gain {d2o['rel_gain']:+.4f} "
        f"(bar ≥ {constants.V7_D2_MIN_REL_GAIN}) → **{'pass' if d2o['passed'] else 'FAIL'}**",
        "",
        "## D3 — standalone surprise",
        "",
        f"nexar_val video AUC of max (n = {d3n['videos']}): "
        f"ŝ {_f(d3n['auc_s'])}, p {_f(d3n['auc_p'])}, "
        f"rho {_f(d3n['auc_rho'])}; ŝ - p {_ci(d3n['s_minus_p'], True)}",
        "",
        f"DoTA-CAP-dev per-clip frame AUC on valid frames (read {d3d['clips_read']} of "
        f"{d3d['clips_total']}; missing caches {len(r['dota_missing'])}):",
        "",
        "| subset | ŝ | p | rho | cubic position | ŝ - p |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("all", "ego", "non_ego"):
        sub = d3d[name]
        lines.append(
            f"| {name} | {_ci(sub['s'])} | {_ci(sub['p'])} | {_ci(sub['rho'])} | "
            f"{_ci(sub['pos'])} | {_ci(sub['s_minus_p'], True)} |"
        )
    lines += [
        "",
        "## D4 — source classifier (printed)",
        "",
        f"normals {d4o['normals']}, lead-ins {d4o['lead_ins']}: "
        f"source AUC {_f(d4o['source_auc'])}, "
        f"gap vs D3 {_f(d4o['gap'], True)}, flag {d4o['flag']}",
        "",
        "## D5 — printed",
        "",
        f"lead-in max share (n = {d5o['clips']}): ŝ {_f(d5o['lead_in_max_share']['s'])}, "
        f"p {_f(d5o['lead_in_max_share']['p'])}; corr(s, p) on nexar_val normals "
        f"{_f(d5o['corr_s_p_normals'])}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--stream",
        choices=constants.V7_STREAMS,
        default=constants.V7_STREAM_V2S,
        help="v2s = §1 (V2-S fc_norm vectors); clip = §1.1 (CLIP frame embeddings)",
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Nexar s1 cache of the stream (default: under CACHE_ROOT)",
    )
    p.add_argument("--census", type=Path, required=True, help="Nexar N0 census.json")
    p.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    p.add_argument(
        "--holdout-file", type=Path, default=constants.V7_SPLITS_DIR / constants.V7_HOLDOUT_FILENAME
    )
    p.add_argument("--encoder", default=constants.V7_ENCODER)
    p.add_argument("--weights", type=Path, default=None, help="local V2-S weights (fc_norm only)")
    p.add_argument("--out-dir", type=Path, default=None, help="default: step0 / step0_clip")
    p.add_argument("--device", default="auto")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    add_common(sub.add_parser("d1", help="cos(u_t, u_{t+k}) on training negatives"))
    tr = sub.add_parser("train", help="LSP stage 1 for every K, K chosen once")
    add_common(tr)
    tr.add_argument("--epochs", type=int, default=constants.V7_EPOCHS)
    dg = sub.add_parser("diag", help="D2-D5 + GO / NO-GO")
    add_common(dg)
    dg.add_argument(
        "--dota-video-dir",
        type=Path,
        default=None,
        help="DoTA s1 cache of the stream (default: under CACHE_ROOT)",
    )
    dg.add_argument(
        "--annotations-dir",
        type=Path,
        default=constants.DATA_ROOT / "DoTA" / "raw_annotations" / "annotations",
    )
    dg.add_argument(
        "--metadata", type=Path, default=constants.DATA_ROOT / "DoTA" / "metadata_val.json"
    )
    dg.add_argument(
        "--split-file", type=Path, default=constants.DATA_ROOT / "DoTA" / "val_split.txt"
    )
    return parser


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill the stream-dependent path defaults (cache dirs, output dir)."""
    nexar_dir, dota_dir = default_cache_dirs(args.stream)
    if args.video_dir is None:
        args.video_dir = nexar_dir
    if getattr(args, "dota_video_dir", "absent") is None:
        args.dota_video_dir = dota_dir
    if args.out_dir is None:
        args.out_dir = step0_dir(args.stream)
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = resolve_defaults(build_arg_parser().parse_args(argv))
    {"d1": run_d1, "train": run_train, "diag": run_diag}[args.command](args)


if __name__ == "__main__":
    main()
