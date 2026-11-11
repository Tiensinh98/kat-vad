"""Pos/neg feature gap of one checkpoint on Nexar val (``core/docs/final/POSNEG_GAP.md``).

Inference only. For every positive video of ``nexar_val`` (whole video, protocol N-B rows) it
pools the model's ``vis_feats`` twice with ``L_neg``'s operator (an attention pool whose weights
are ``softmax(+-logit / tau)``): once over all frames for ``pos`` (weights ``+logit``) and once
over a frame set for ``neg`` (weights ``-logit``). Three frame sets are compared:

* **N-label**: the ground-truth normal frames (D-N1). Measurement only, no loss may use it.
* **N-thr**: frames with ``sigmoid(logit) < 0.5``.
* **N-soft**: every frame.

``gap = 1 - cos(pos, neg)``; ``floor`` is the same two operators on the GT-normal frames only
(half A with ``+logit``, half B with ``-logit``), i.e. how far apart they land by chance.
Gated: STOP iff ``mean(gap_label) < POSNEG_GAP_RATIO * mean(floor)``. Everything else is printed.

CLI::

    python -m core.tools.pos_neg_gap --name K5 \\
        --ckpt outputs/v_final/kim_whole/K5/s2024/checkpoint_last.pt \\
        --census .../nexar_n0/census.json --clip-s1-dir cache/clip/Nexar_s1_ncc \\
        --video-s1-dir cache/video/<enc>/Nexar_s1_squash --stats-dir cache/v2/<arm>/Nexar_s8 \\
        --data-dir data/Nexar_window --out-dir outputs/v_final/kim_whole/REPORTS/pos_neg_gap
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
from core.data.definitions import dataset_abbr, item_verbalizer
from core.data.v2_inputs import load_stats
from core.data.v2_splits import load_split
from core.device import resolve_device
from core.inference import make_class_feats_fn
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_build import native_labels
from core.tools.nexar_eval import item_rows
from core.tools.rate_matched_eval import Run, load_finished_model
from core.train import load_class_names

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/POSNEG_GAP.md"
STOP, CONTINUE = "STOP", "CONTINUE"
METRICS = (
    "gap_label", "gap_thr", "gap_soft", "floor", "thr_precision", "thr_recall",
    "soft_mass_normal", "pos_mass_abnormal",
)
SKIP_NO_NORMAL = "too_few_normal_frames"
SKIP_NO_ABNORMAL = "no_abnormal_frame"


# ---------------------------------------------------------------------------
# pure math (float64, numpy)
# ---------------------------------------------------------------------------
def attention_weights(logits: np.ndarray, sign: float, mask: np.ndarray, tau: float) -> np.ndarray:
    """``softmax(sign * logit / tau)`` over the ``mask`` frames, 0 elsewhere (sums to 1)."""
    if not mask.any():
        raise ValueError("attention over an empty frame set")
    z = sign * logits.astype(np.float64) / tau
    z = np.where(mask, z, -np.inf)
    z = z - z[mask].max()
    w = np.exp(z)
    out: np.ndarray = w / w.sum()
    return out


def pool(feats: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pooled: np.ndarray = weights @ feats.astype(np.float64)
    return pooled


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """``1 - cos(a, b)``; an all-zero vector has no direction and raises."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine distance of a zero vector")
    return 1.0 - float(a @ b) / (na * nb)


def floor_distance(
    feats: np.ndarray, logits: np.ndarray, normal: np.ndarray, tau: float,
    draws: int, rng: np.random.Generator,
) -> float:
    """Mean over random half splits of the normal frames: ``+logit`` pool of A, ``-logit`` of B."""
    idx = np.flatnonzero(normal)
    half = len(idx) // 2
    total = 0.0
    for _ in range(draws):
        perm = rng.permutation(idx)
        a_mask = np.zeros(len(normal), dtype=bool)
        b_mask = np.zeros(len(normal), dtype=bool)
        a_mask[perm[:half]] = True
        b_mask[perm[half : 2 * half]] = True
        a = pool(feats, attention_weights(logits, 1.0, a_mask, tau))
        b = pool(feats, attention_weights(logits, -1.0, b_mask, tau))
        total += cosine_distance(a, b)
    return total / draws


def video_metrics(
    feats: np.ndarray, logits: np.ndarray, labels: np.ndarray, tau: float,
    draws: int, rng: np.random.Generator,
    sigmoid_threshold: float = constants.POSNEG_SIGMOID_THRESHOLD,
) -> dict[str, float | None] | str:
    """The per-video record, or the skip reason (a string). ``labels`` is 1 on abnormal frames."""
    if not (len(feats) == len(logits) == len(labels)):
        raise ValueError(f"lengths differ: {len(feats)} feats, {len(logits)} logits, "
                         f"{len(labels)} labels")
    abnormal = labels == 1
    normal = labels == 0
    if not abnormal.any():
        return SKIP_NO_ABNORMAL
    if normal.sum() < constants.POSNEG_MIN_NORMAL_FRAMES:
        return SKIP_NO_NORMAL
    everything = np.ones(len(labels), dtype=bool)
    w_pos = attention_weights(logits, 1.0, everything, tau)
    pos = pool(feats, w_pos)
    w_label = attention_weights(logits, -1.0, normal, tau)
    w_soft = attention_weights(logits, -1.0, everything, tau)
    thr = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))) < sigmoid_threshold
    record: dict[str, float | None] = {
        "gap_label": cosine_distance(pos, pool(feats, w_label)),
        "gap_soft": cosine_distance(pos, pool(feats, w_soft)),
        "floor": floor_distance(feats, logits, normal, tau, draws, rng),
        "soft_mass_normal": float(w_soft[normal].sum()),
        "pos_mass_abnormal": float(w_pos[abnormal].sum()),
        "gap_thr": None, "thr_precision": None, "thr_recall": None,
    }
    if thr.any():  # printed, never gated: an empty N-thr is counted by the caller
        w_thr = attention_weights(logits, -1.0, thr, tau)
        record["gap_thr"] = cosine_distance(pos, pool(feats, w_thr))
        record["thr_precision"] = float(normal[thr].mean())
        record["thr_recall"] = float(thr[normal].mean())
    return record


def bootstrap_ci(values: np.ndarray, draws: int, seed: int) -> list[float]:
    """Percentile 95 % interval of the mean over videos."""
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(values, size=len(values), replace=True).mean())
             for _ in range(draws)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(
    records: dict[str, dict[str, float | None]], skipped: dict[str, str], seed: int,
) -> dict[str, Any]:
    """Per-metric mean / median / interval over the videos that define it, and the decision."""
    stats: dict[str, Any] = {}
    for name in METRICS:
        values = np.array([r[name] for r in records.values() if r[name] is not None], float)
        stats[name] = None if values.size == 0 else {
            "n": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)),
            "ci95": bootstrap_ci(values, constants.POSNEG_BOOTSTRAP_DRAWS, seed),
        }
    gate = stats["gap_label"] is not None and stats["floor"] is not None
    ratio = stats["gap_label"]["mean"] / stats["floor"]["mean"] if gate else None
    decision = None
    if ratio is not None:
        decision = STOP if ratio < constants.POSNEG_GAP_RATIO else CONTINUE
    return {
        "videos_scored": len(records),
        "videos_skipped": skipped,
        "n_thr_empty": sum(r["gap_thr"] is None for r in records.values()),
        "metrics": stats,
        "ratio_gap_label_over_floor": ratio,
        "ratio_bar": constants.POSNEG_GAP_RATIO,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# model side
# ---------------------------------------------------------------------------
@torch.no_grad()
def video_outputs(
    model: Any, rows: np.ndarray, class_feats_fn: Any, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """``vis_feats (T, D)`` and pre-sigmoid ``cls_bin_logits (T,)``, windowed as the eval scorer."""
    feats = torch.from_numpy(rows)
    vis, logits = [], []
    for start in range(0, len(feats), constants.MAX_VIS_LEN):
        window = feats[start : start + constants.MAX_VIS_LEN].unsqueeze(0).to(device)
        lengths = torch.tensor([window.shape[1]], device=device)
        out = model(window, lengths, class_feats=class_feats_fn())
        vis.append(out["vis_feats"][0].float().cpu())
        logits.append(out["cls_bin_logits"][0].float().cpu())
    return torch.cat(vis).numpy(), torch.cat(logits).numpy()


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    model, step, text_encode_fn, cfg = load_finished_model(Run(args.name, args.ckpt), device)
    stats = load_stats(args.stats_dir) if args.stats_dir is not None else None
    expected = (constants.V2_OFF, constants.V2_OFF) if stats is None else (stats.crn, stats.motion)
    if (cfg.v2.crn, cfg.v2.motion) != expected:
        raise SystemExit(f"{args.name} is crn={cfg.v2.crn} motion={cfg.v2.motion}, "
                         f"the stats are {expected} (K4)")
    if stats is not None and stats.has_motion and args.video_s1_dir is None:
        raise SystemExit("a motion arm needs --video-s1-dir")
    census = json.loads(args.census.read_text(encoding="utf-8"))
    rows_meta: dict[str, dict[str, Any]] = census["videos"]
    ids = [v for v in load_split(constants.V2_SPLIT_NEXAR_VAL, args.split_dir, final=False)
           if rows_meta[v]["label"] == 1]
    class_names = load_class_names(args.data_dir)
    rng = np.random.default_rng(args.seed)
    stride = constants.NEXAR_EVAL_STRIDE
    records: dict[str, dict[str, float | None]] = {}
    skipped: dict[str, str] = {}
    for video in ids:
        x1 = np.load(args.clip_s1_dir / f"{video}.npy")
        u1 = None if stats is None or not stats.has_motion else np.load(
            args.video_s1_dir / f"{video}.npy")
        frames = len(x1)
        rows = item_rows(x1, u1, stats, 0, frames)
        labels = native_labels(rows_meta[video], frames)[::stride]
        if len(labels) != len(rows):
            raise ValueError(f"{video}: {len(rows)} rows but {len(labels)} strided labels")
        verbalizer = item_verbalizer(dataset_abbr(constants.DOTA_DATASET), video)
        class_feats_fn = make_class_feats_fn(text_encode_fn, class_names, verbalizer)
        feats, logits = video_outputs(model, rows, class_feats_fn, device)
        result = video_metrics(feats, logits, labels, constants.CONTRASTIVE_TEMP,
                               constants.POSNEG_FLOOR_DRAWS, rng)
        if isinstance(result, str):
            skipped[video] = result
        else:
            records[video] = result
    summary = summarize(records, skipped, args.seed)
    readout = {
        "rule": RULE, "name": args.name, "checkpoint": str(args.ckpt), "global_step": step,
        "split": constants.V2_SPLIT_NEXAR_VAL, "tau": constants.CONTRASTIVE_TEMP,
        "kim_on": model.kim is not None, "seed": args.seed, **summary,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{constants.POSNEG_REPORT_STEM}.json", readout)
    write_json_atomic(args.out_dir / f"{constants.POSNEG_REPORT_STEM}_per_video.json", records)
    write_text_atomic(args.out_dir / f"{constants.POSNEG_REPORT_STEM}.md", render_markdown(readout))
    LOGGER.info("decision %s (ratio %s) -> %s", readout["decision"],
                readout["ratio_gap_label_over_floor"], args.out_dir)
    return readout


def _cell(stat: dict[str, Any] | None) -> str:
    if stat is None:
        return "n/a"
    lo, hi = stat["ci95"]
    return f"{stat['mean']:.4f} [{lo:.4f}, {hi:.4f}] (median {stat['median']:.4f}, n={stat['n']})"


def render_markdown(r: dict[str, Any]) -> str:
    m = r["metrics"]
    ratio = r["ratio_gap_label_over_floor"]
    lines = [
        f"# Pos/neg feature gap -- {r['name']} (step {r['global_step']}), `{r['split']}`", "",
        f"Rule: `{r['rule']}`. n = 1 checkpoint: the decision reads this checkpoint only.", "",
        f"**Decision: {r['decision']}** -- gap(N-label) / floor = "
        f"{'n/a' if ratio is None else f'{ratio:.3f}'} against the bar {r['ratio_bar']} "
        f"(STOP iff below).", "",
        "| Quantity | mean [95 % bootstrap over videos] |", "|---|---|",
        f"| gap, N-label (gated) | {_cell(m['gap_label'])} |",
        f"| floor (gated) | {_cell(m['floor'])} |",
        f"| gap, N-thr (printed) | {_cell(m['gap_thr'])} |",
        f"| gap, N-soft (printed) | {_cell(m['gap_soft'])} |",
        f"| N-thr precision (GT-normal share of its frames) | {_cell(m['thr_precision'])} |",
        f"| N-thr recall (GT-normal frames inside it) | {_cell(m['thr_recall'])} |",
        f"| N-soft attention mass on GT-normal | {_cell(m['soft_mass_normal'])} |",
        f"| pos attention mass on GT-abnormal | {_cell(m['pos_mass_abnormal'])} |", "",
        f"Videos scored {r['videos_scored']}; skipped {len(r['videos_skipped'])} "
        f"({sorted(set(r['videos_skipped'].values()))}); N-thr empty on {r['n_thr_empty']}.",
        f"KIM on: {r['kim_on']}. tau = {r['tau']}. Ground-truth frame labels are used to measure "
        "only; training has video-level labels.", "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--name", required=True, help="the arm, recorded in the read-out")
    parser.add_argument("--ckpt", type=Path, required=True, help="a finished checkpoint (J10)")
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--clip-s1-dir", type=Path, required=True)
    parser.add_argument("--video-s1-dir", type=Path, default=None)
    parser.add_argument("--stats-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, required=True, help="training dir (defs.json)")
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=constants.SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
