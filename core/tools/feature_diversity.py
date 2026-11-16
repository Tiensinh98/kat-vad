"""Within-video feature diversity of two checkpoints on Nexar val (``KIM_RUN.md`` §28.2, P2).

Inference only. For every positive ``nexar_val`` video (whole video, protocol N-B rows) both
checkpoints are run on the same rows; their ``vis_feats`` rows are L2-normalized and three
numbers are taken per video (FD4):

* ``row_cos``: mean off-diagonal cosine between rows (higher = rows more alike);
* ``eff_rank``: exp(entropy of the normalized singular values of the centred rows), printed;
* ``sep``: ``|mu_abnormal - mu_normal|^2 / mean |z - mu_own|^2``, a scale-free Fisher ratio.

The contrast is paired per video (``new - base``) with a percentile bootstrap over videos; the
verdict is FD6 (SUPPORTED iff ``sep`` falls and ``row_cos`` rises, both intervals clear of 0).

CLI::

    python -m core.tools.feature_diversity \\
        --base-name K5 --base-ckpt outputs/v_final/kim/K5/s2024/checkpoint_last.pt \\
        --new-name K5Lneg --new-ckpt outputs/v_final/kim_lneg/K5/s2024/checkpoint_last.pt \\
        --census .../nexar_n0/census.json --clip-s1-dir cache/clip/Nexar_s1_ncc \\
        --video-s1-dir cache/video/<enc>/Nexar_s1_squash --stats-dir cache/v2/A2_S/Nexar_s8 \\
        --data-dir data/Nexar_window --out-dir outputs/v_final/kim_lneg/REPORTS/feature_diversity
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.definitions import dataset_abbr, item_verbalizer
from core.data.v2_inputs import load_stats
from core.data.v2_splits import load_split
from core.device import resolve_device
from core.inference import make_class_feats_fn
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_build import native_labels
from core.tools.nexar_eval import item_rows
from core.tools.pos_neg_gap import bootstrap_ci, video_outputs
from core.tools.rate_matched_eval import Run, load_finished_model
from core.train import load_class_names

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/KIM_RUN.md §28.2"
SKIP_FEW_ROWS = "too_few_abnormal_or_normal_rows"


# ---------------------------------------------------------------------------
# pure math (float64, numpy)
# ---------------------------------------------------------------------------
def normalize_rows(feats: np.ndarray) -> np.ndarray:
    """Each row to unit L2 norm; a zero row has no direction and raises."""
    x = feats.astype(np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0.0).any():
        raise ValueError("a vis_feats row is all zero")
    out: np.ndarray = x / norms
    return out


def row_cos(z: np.ndarray) -> float:
    """Mean off-diagonal cosine of unit rows: ``(|sum z|^2 - T) / (T (T - 1))``."""
    t = len(z)
    if t < 2:
        raise ValueError("row_cos needs at least two rows")
    total = z.sum(axis=0)
    return float((total @ total - t) / (t * (t - 1)))


def eff_rank(z: np.ndarray) -> float:
    """exp(entropy) of the normalized singular values of the centred rows (Roy & Vetterli)."""
    s = np.linalg.svd(z - z.mean(axis=0), compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return 1.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def separation(z: np.ndarray, abnormal: np.ndarray) -> float:
    """``|mu_a - mu_n|^2 / mean_rows |z - mu_own|^2`` over unit rows ``z``."""
    normal = ~abnormal
    mu_a, mu_n = z[abnormal].mean(axis=0), z[normal].mean(axis=0)
    within = np.concatenate([((z[abnormal] - mu_a) ** 2).sum(axis=1),
                             ((z[normal] - mu_n) ** 2).sum(axis=1)]).mean()
    between = float(((mu_a - mu_n) ** 2).sum())
    if within == 0.0:
        raise ValueError("every row equals its class mean: sep is undefined")
    return between / float(within)


def video_metrics(feats: np.ndarray, labels: np.ndarray) -> dict[str, float] | str:
    """FD4 for one video and one checkpoint, or the skip reason. ``labels`` is 1 on abnormal."""
    if len(feats) != len(labels):
        raise ValueError(f"lengths differ: {len(feats)} rows, {len(labels)} labels")
    abnormal = labels == 1
    if min(int(abnormal.sum()), int((~abnormal).sum())) < constants.FEATDIV_MIN_ROWS:
        return SKIP_FEW_ROWS
    z = normalize_rows(feats)
    return {"row_cos": row_cos(z), "eff_rank": eff_rank(z), "sep": separation(z, abnormal)}


def paired_summary(base: dict[str, dict[str, float]], new: dict[str, dict[str, float]],
                   seed: int) -> dict[str, Any]:
    """FD5 + FD6: per-metric means, paired ``new - base`` with its interval, and the verdict."""
    if sorted(base) != sorted(new):
        raise ValueError("the two checkpoints were scored on different videos")
    videos = sorted(base)
    if not videos:
        raise ValueError("no video was scored")
    out: dict[str, Any] = {}
    for name in constants.FEATDIV_METRICS:
        b = np.array([base[v][name] for v in videos])
        n = np.array([new[v][name] for v in videos])
        d = n - b
        out[name] = {"base": float(b.mean()), "new": float(n.mean()),
                     "delta": float(d.mean()), "delta_median": float(np.median(d)),
                     "delta_ci95": bootstrap_ci(d, constants.POSNEG_BOOTSTRAP_DRAWS, seed),
                     "share_new_below": float((d < 0).mean())}
    sep_lo, sep_hi = out["sep"]["delta_ci95"]
    cos_lo, _ = out["row_cos"]["delta_ci95"]
    if sep_hi < 0 and cos_lo > 0:
        verdict = constants.FEATDIV_SUPPORTED
    elif sep_lo > 0:
        verdict = constants.FEATDIV_OPPOSITE
    else:
        verdict = constants.FEATDIV_NOT_SUPPORTED
    return {"videos": len(videos), "metrics": out, "verdict": verdict}


# ---------------------------------------------------------------------------
# model side
# ---------------------------------------------------------------------------
def score_checkpoint(name: str, ckpt: Path, args: argparse.Namespace,
                     videos: dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray]],
                     ) -> tuple[dict[str, dict[str, float]], dict[str, str], int]:
    """FD4 on every video for one checkpoint; returns records, skips and the checkpoint step."""
    device = resolve_device(args.device)
    model, step, text_encode_fn, cfg = load_finished_model(Run(name, ckpt), device)
    stats = load_stats(args.stats_dir) if args.stats_dir is not None else None
    expected = (constants.V2_OFF, constants.V2_OFF) if stats is None else (stats.crn, stats.motion)
    if (cfg.v2.crn, cfg.v2.motion) != expected:
        raise SystemExit(f"{name} is crn={cfg.v2.crn} motion={cfg.v2.motion}, "
                         f"the stats are {expected} (K4)")
    if model.kim is None:
        raise SystemExit(f"{name} was built without KIM: not a K5-family checkpoint")
    class_names = load_class_names(args.data_dir)
    records: dict[str, dict[str, float]] = {}
    skipped: dict[str, str] = {}
    for video, (x1, u1, labels) in videos.items():
        rows = item_rows(x1, u1, stats, 0, len(x1))
        if len(labels) != len(rows):
            raise ValueError(f"{video}: {len(rows)} rows but {len(labels)} strided labels")
        verbalizer = item_verbalizer(dataset_abbr(constants.DOTA_DATASET), video)
        class_feats_fn = make_class_feats_fn(text_encode_fn, class_names, verbalizer)
        feats, _ = video_outputs(model, rows, class_feats_fn, device)
        result = video_metrics(feats, labels)
        if isinstance(result, str):
            skipped[video] = result
        else:
            records[video] = result
    LOGGER.info("%s (step %d): %d videos scored, %d skipped", name, step, len(records),
                len(skipped))
    return records, skipped, step


def load_videos(args: argparse.Namespace
                ) -> dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """FD2: CLIP rows, motion rows and strided D-N1 labels of every positive ``nexar_val`` video."""
    census: dict[str, dict[str, Any]] = json.loads(args.census.read_text(encoding="utf-8"))[
        "videos"]
    ids = [v for v in load_split(constants.V2_SPLIT_NEXAR_VAL, args.split_dir, final=False)
           if census[v]["label"] == 1]
    out: dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray]] = {}
    for video in ids:
        x1 = np.load(args.clip_s1_dir / f"{video}.npy")
        u1 = None if args.video_s1_dir is None else np.load(args.video_s1_dir / f"{video}.npy")
        labels = native_labels(census[video], len(x1))[:: constants.NEXAR_EVAL_STRIDE]
        out[video] = (x1, u1, labels)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    videos = load_videos(args)
    base, base_skip, base_step = score_checkpoint(args.base_name, args.base_ckpt, args, videos)
    new, new_skip, new_step = score_checkpoint(args.new_name, args.new_ckpt, args, videos)
    if base_skip != new_skip:
        raise ValueError("the two checkpoints skipped different videos (labels are shared)")
    readout = {
        "rule": RULE, "split": constants.V2_SPLIT_NEXAR_VAL, "seed": args.seed,
        "base": {"name": args.base_name, "checkpoint": str(args.base_ckpt), "step": base_step},
        "new": {"name": args.new_name, "checkpoint": str(args.new_ckpt), "step": new_step},
        "positives": len(videos), "skipped": base_skip,
        **paired_summary(base, new, args.seed),
    }
    stem = constants.FEATDIV_REPORT_STEM
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{stem}.json", readout)
    write_json_atomic(args.out_dir / f"{stem}_per_video.json", {"base": base, "new": new})
    write_text_atomic(args.out_dir / f"{stem}.md", render_markdown(readout))
    LOGGER.info("verdict %s -> %s", readout["verdict"], args.out_dir)
    return readout


def render_markdown(r: dict[str, Any]) -> str:
    base, new = r["base"]["name"], r["new"]["name"]
    lines = [
        f"# Within-video feature diversity -- {new} vs {base}, `{r['split']}`", "",
        f"Rule: `{r['rule']}`. n = 1 checkpoint per arm ({base} step {r['base']['step']}, "
        f"{new} step {r['new']['step']}). Inference only.", "",
        f"**Verdict (FD6): {r['verdict']}** -- SUPPORTED iff Δ sep CI < 0 and Δ row_cos CI > 0; "
        "OPPOSITE iff Δ sep CI > 0.", "",
        f"| metric (per video, unit rows) | {base} | {new} | Δ {new} - {base} [95 %] | "
        f"share of videos {new} lower |", "|---|---|---|---|---|",
    ]
    labels = {"row_cos": "row_cos (decides)", "sep": "sep (decides)",
              "eff_rank": "eff_rank (printed)"}
    for name in ("sep", "row_cos", "eff_rank"):
        m = r["metrics"][name]
        lo, hi = m["delta_ci95"]
        lines.append(f"| {labels[name]} | {m['base']:.4f} | {m['new']:.4f} | "
                     f"{m['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {m['share_new_below']:.3f} |")
    lines += ["", f"Videos scored {r['videos']} of {r['positives']} positives; skipped "
              f"{len(r['skipped'])} ({sorted(set(r['skipped'].values()))}). Frame labels "
              "(D-N1) measure only.", ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    for role in ("base", "new"):
        parser.add_argument(f"--{role}-name", required=True, help=f"the {role} arm's name")
        parser.add_argument(f"--{role}-ckpt", type=Path, required=True,
                            help=f"the {role} arm's finished checkpoint (J10)")
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
