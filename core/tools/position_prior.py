"""v2 E3: the T2 position prior ``p_T2`` scored on DoTA (Amendment 6 G6 (a), Amendment 9 M6).

What could a model score on DoTA by having learned **only T2's position prior**? This tool
answers it without pixels: ``core.eda``'s logistic probe on the cubic relative-position
features (``kill_switch_probe.position_features``: ``τ, τ², τ³``) is fitted on every
**training window** of the v2 dataset dir (T2-train minus T2-val) with that window's frame
labels (``v2_dataset.window_frame_labels``, the arithmetic that labelled T2-test), then scored
at each DoTA clip's **native** frames. Relative position makes a 20-step T2 window and a
100-frame DoTA clip comparable; nothing else is shared.

Writes ``position_prior.npz`` (native ``p_T2`` scores + labels per clip, the
``protocol_b_eval.write_clip_scores`` layout, read by ``e3_readout``) and a read-out with the
macro AUC over two-class clips (cluster bootstrap by source video, D3) beside the monotone
``t/N`` ruler (G7). Printed, never decided on. Name a ``dota_cap_dev`` number
**DoTA-CAP (n/1397)** (D14).

CLI::

    python -m core.tools.position_prior --data-dir data/DADA2000_orig_v2 \\
        --s1-dir cache/clip/DoTA_s1_ncc --metadata data/DoTA/metadata_val.json \\
        --split-file data/DoTA/val_split.txt --split dota_cap_dev \\
        --out-dir outputs/v2/e3/REPORTS/position_prior/dota_cap_dev
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_splits import dota_group, load_split
from core.eda.features import transfer_scores
from core.metrics import cluster_bootstrap_ci
from core.tools.kill_switch_probe import position_features, write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import OPEN_SPLITS, clip_aucs, write_clip_scores
from core.tools.rate_matched_eval import native_labels
from core.tools.v2_dataset import window_frame_labels

LOGGER = logging.getLogger(__name__)

PRIOR_NPZ = "position_prior.npz"
READOUT_JSON = "position_prior.json"
READOUT_MD = "position_prior.md"


def training_frames(data_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Cubic position features and frame labels of every training window of ``data_dir``."""
    train: dict[str, int] = json.loads(
        (data_dir / constants.LABELS_TRAIN_FILENAME).read_text(encoding="utf-8")
    )
    meta: dict[str, dict[str, Any]] = json.loads(
        (data_dir / constants.META_FILENAME).read_text(encoding="utf-8")
    )
    feats: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for window_id in sorted(train):
        labels = np.asarray(window_frame_labels(meta[window_id], window_id), dtype=np.int64)
        if int(labels.any()) != int(train[window_id]):
            raise ValueError(f"{window_id}: frame labels disagree with its bag label")
        feats.append(position_features(len(labels)))
        targets.append(labels)
    return np.concatenate(feats), np.concatenate(targets), len(train)


def prior_scores(
    matrix: np.ndarray, target: np.ndarray, native_frames: dict[str, int], seed: int
) -> dict[str, np.ndarray]:
    """``p_T2`` at every native frame of every clip: one fit, scored per clip length."""
    lengths = sorted(set(native_frames.values()))
    stacked = np.concatenate([position_features(n) for n in lengths])
    scored = transfer_scores(matrix, target, stacked, seed)
    by_length: dict[int, np.ndarray] = {}
    start = 0
    for n in lengths:
        by_length[n] = scored[start:start + n]
        start += n
    return {v: by_length[n] for v, n in native_frames.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.split not in OPEN_SPLITS:
        raise ValueError(f"position_prior reads {OPEN_SPLITS} only, not {args.split!r}")
    matrix, target, windows = training_frames(args.data_dir)
    ids = load_split(args.split, args.split_dir)
    parsed = parse_metadata(args.metadata, read_split_ids(args.split_file))
    records = {r.video_id: r for r in parsed}
    native_frames = {v: len(np.load(args.s1_dir / f"{v}.npy", mmap_mode="r")) for v in ids}
    labels = {v: native_labels(records[v], native_frames[v]) for v in ids}
    scores = prior_scores(matrix, target, native_frames, args.seed)
    groups = {v: dota_group(v) for v in ids}

    def ci(values: dict[str, float]) -> dict[str, float] | None:
        return cluster_bootstrap_ci(
            values, groups, constants.V2_E1_BOOTSTRAP, args.seed, constants.V2_E1_CI
        )

    ruler = {v: np.arange(n) / n for v, n in native_frames.items()}
    readout: dict[str, Any] = {
        "addendum": "PREREG_ADDENDUM.md §15 G6 (a), §18 M6 (printed, never decided on)",
        "data_dir": str(args.data_dir),
        "train_windows": windows,
        "train_frames": len(target),
        "train_abnormal_share": float(target.mean()),
        "split": args.split,
        "clips": len(ids),
        "p_t2_macro": ci(clip_aucs(scores, labels)),
        "monotone_ruler_macro": ci(clip_aucs(ruler, labels)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_clip_scores(args.out_dir / PRIOR_NPZ, scores, labels)
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(readout))
    LOGGER.info("p_T2 on %s: %s -> %s", args.split, readout["p_t2_macro"], args.out_dir)
    return readout


def _fmt(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def render_markdown(r: dict[str, Any]) -> str:
    name = "DoTA-CAP (n/1397)" if r["split"] == constants.V2_SPLIT_DOTA_CAP_DEV else "DoTA"
    return "\n".join([
        f"# T2 position prior `p_T2` on `{r['split']}` — {name}",
        "",
        f"Cubic position probe fitted on {r['train_windows']} T2 training windows "
        f"({r['train_frames']} frames, abnormal share {r['train_abnormal_share']:.3f}); "
        f"scored at native frames of {r['clips']} clips. Printed, never decided on (G6).",
        "",
        f"* `p_T2` macro: **{_fmt(r['p_t2_macro'])}**",
        f"* `t/N` ruler (monotone): {_fmt(r['monotone_ruler_macro'])}",
        f"* `p_CAP` (E2(d), in-domain cubic on `dota_cap_dev`): {constants.V2_E3_P_CAP:.3f}",
    ]) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="the v2 dataset dir")
    parser.add_argument("--s1-dir", type=Path, required=True,
                        help="cache/clip/DoTA_s1_ncc: native frame count per clip")
    parser.add_argument("--metadata", type=Path, required=True, help="metadata_val.json")
    parser.add_argument("--split-file", type=Path, required=True, help="val_split.txt")
    parser.add_argument("--split", choices=OPEN_SPLITS, default=constants.V2_SPLIT_DOTA_CAP_DEV)
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=constants.SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
