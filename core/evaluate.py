"""KAT-VAD evaluation CLI (spec §10) — sliding-window full-length eval.

Scores every video in ``frame_labels_test.json`` through the inference path
(windows of ``max_vis_len``), computes micro frame-level **AUC** and **AP**
over all videos' concatenated sampled frames (the baseline's torchmetrics
accumulation semantics; AUC/AP are invariant to the uniform per-stride A5
expansion, so metrics are computed at sampled-frame resolution), and writes
``results.json`` plus one per-video score ``.npz`` consumable by
``core/tools/visualize.py``.

MCC family / AUC_A / mAP@IoU are Phase-7 scope and intentionally stubbed.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from core import constants
from core.config import load_config
from core.data.dataset import FeatureEvalDataset
from core.data.definitions import DatasetSpecVerbalizer, dataset_abbr
from core.device import resolve_device
from core.inference import (
    load_model_for_scoring,
    make_class_feats_fn,
    score_to_npz,
    sliding_window_scores,
)
from core.models.text_encoding import (
    TEXT_ENCODER_CHOICES,
    TEXT_ENCODER_CLIP,
    make_text_encoder,
)
from core.train import load_class_names

LOGGER = logging.getLogger(__name__)

RESULTS_FILENAME = "results.json"
SCORES_DIRNAME = "scores"


def frame_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Micro frame-level ROC AUC (labels in {0,1}); tested against torchmetrics."""
    return float(roc_auc_score(labels.astype(np.int64), scores))


def frame_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Micro frame-level average precision; tested against torchmetrics."""
    return float(average_precision_score(labels.astype(np.int64), scores))


def abnormal_only_auc(*_args: np.ndarray) -> float:
    """AUC_A (abnormal-videos-only AUC) — Phase 7."""
    raise NotImplementedError("AUC_A is Phase-7 scope (spec §10)")


def mcc_family(*_args: np.ndarray) -> dict[str, float]:
    """MCC / AUC_MCC / MCC@0.5 — Phase 7."""
    raise NotImplementedError("MCC metrics are Phase-7 scope (spec §10)")


def map_at_iou(*_args: np.ndarray) -> dict[str, float]:
    """mAP@IoU{0.1..0.5} temporal localization — Phase 7."""
    raise NotImplementedError("mAP@IoU is Phase-7 scope (spec §10)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate KAT-VAD (spec §10)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    parser.add_argument("--ckpt", type=Path, default=None, help="KAT-VAD checkpoint")
    parser.add_argument("--baseline-ckpt", type=Path, default=None,
                        help="LaGoVAD best.ckpt (compat loader; reproduction gate a)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="dataset dir with frame_labels_test.json + defs.json")
    parser.add_argument("--clip-dir", type=Path, default=None,
                        help="CLIP feature cache dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-encoder", choices=TEXT_ENCODER_CHOICES,
                        default=TEXT_ENCODER_CLIP)
    parser.add_argument("--no-verbalize", action="store_true",
                        help="encode raw class names instead of sampled definitions")
    parser.add_argument("--save-scores", action="store_true",
                        help="write per-video score .npz files for visualization")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    device = resolve_device(cfg.train.device)

    dataset_name = cfg.data.dataset
    data_dir = args.data_dir if args.data_dir else constants.DATA_ROOT / dataset_name
    clip_dir = args.clip_dir if args.clip_dir else constants.CLIP_CACHE_DIR / dataset_name

    dataset = FeatureEvalDataset(data_dir, clip_dir)
    class_names = load_class_names(data_dir)
    model = load_model_for_scoring(cfg, device, args.ckpt, args.baseline_ckpt, args.text_encoder)
    text_encode_fn = make_text_encoder(model, args.text_encoder, device, dim=cfg.model.hidden_dim)
    verbalizer = None
    if not args.no_verbalize:
        # Seeded: definition sampling is per-window, and an unseeded verbalizer
        # makes eval non-reproducible (measured: ±0.003 AUC, ±0.3 per-video
        # max_score across identical runs on the 28-video MSAD slice).
        verbalizer = DatasetSpecVerbalizer(
            dataset_abbr(dataset_name),
            rng=random.Random(constants.SEED),  # nosec B311 - not security-sensitive
        )
    class_feats_fn = make_class_feats_fn(text_encode_fn, class_names, verbalizer)

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    per_video: dict[str, dict[str, float]] = {}
    scores_dir = args.output_dir / SCORES_DIRNAME
    for i in range(len(dataset)):
        item = dataset[i]
        video_id: str = item["video_id"]
        score, sim = sliding_window_scores(
            model, item["v_feat"], class_feats_fn, cfg.data.max_vis_len
        )
        gt = item["frame_label"].numpy()
        scores_np = score.numpy()
        all_scores.append(scores_np)
        all_labels.append(gt)
        per_video[video_id] = {
            "num_frames": float(len(gt)),
            "max_score": float(scores_np.max()),
            "abnormal": float(gt.max() > 0),
        }
        if args.save_scores:
            score_to_npz(scores_dir, video_id, score, sim, class_names, gt=gt)
        LOGGER.info("scored %s (%d sampled frames)", video_id, len(gt))

    scores_cat = np.concatenate(all_scores)
    labels_cat = np.concatenate(all_labels)
    if labels_cat.min() == labels_cat.max():
        raise ValueError("Test set has a single class; AUC/AP undefined")
    results = {
        "dataset": dataset_name,
        "num_videos": len(dataset),
        "num_frames": len(labels_cat),
        "auc": frame_auc(scores_cat, labels_cat),
        "ap": frame_ap(scores_cat, labels_cat),
        "checkpoint": str(args.ckpt or args.baseline_ckpt),
        "per_video": per_video,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / RESULTS_FILENAME
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    LOGGER.info(
        "AUC %.4f | AP %.4f | %d videos -> %s",
        results["auc"], results["ap"], len(dataset), results_path,
    )


if __name__ == "__main__":
    main()
