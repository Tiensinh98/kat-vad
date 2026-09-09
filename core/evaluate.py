"""KAT-VAD evaluation CLI (spec §10) — sliding-window full-length eval.

Scores every video in ``frame_labels_test.json`` through the inference path
(windows of ``max_vis_len``), computes micro frame-level **AUC** and **AP**
over all videos' concatenated sampled frames (the baseline's torchmetrics
accumulation semantics; AUC/AP are invariant to the uniform per-stride A5
expansion, so metrics are computed at sampled-frame resolution), and writes
``results.json`` plus one per-video score ``.npz`` consumable by
``core/tools/visualize.py``.

``--score-norm`` selects how per-video score curves are pooled before the micro
metric (lesson C12). The default ``auto`` uses min-max per video when every
eval video is abnormal -- pooling raw scores over an all-abnormal set measures
between-clip confidence rather than localization, and cost LaGoVAD's released
checkpoint 11 AUC points on DoTA. ``auc_raw`` is always reported alongside so
the two protocols stay comparable. To recompute these from an existing run's
``scores/`` directory without re-running inference, use
``python -m core.tools.rescore``.

``--equalize-length N`` is the lesson **C28** control: it crops every clip to
exactly ``N`` sampled frames and drops the shorter ones, so clip length carries
no label information and the resulting AUC is what the model earns from the
pixels alone. On the reconstructed DADA-2000 a detector reading only the frame
count scores micro AUC 0.8654, so the uncontrolled number is not interpretable
(``core/docs/DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md`` §2, §4.1).

MCC family / AUC_A / mAP@IoU are Phase-7 scope and intentionally stubbed.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

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
from core.metrics import pooled_metrics
from core.models.text_encoding import (
    TEXT_ENCODER_CHOICES,
    TEXT_ENCODER_CLIP,
    make_text_encoder,
)
from core.train import load_class_names

LOGGER = logging.getLogger(__name__)

RESULTS_FILENAME = "results.json"
SCORES_DIRNAME = "scores"


def abnormal_only_auc(*_args: np.ndarray) -> float:
    """AUC_A (abnormal-videos-only AUC) — Phase 7."""
    raise NotImplementedError("AUC_A is Phase-7 scope (spec §10)")


def mcc_family(*_args: np.ndarray) -> dict[str, float]:
    """MCC / AUC_MCC / MCC@0.5 — Phase 7."""
    raise NotImplementedError("MCC metrics are Phase-7 scope (spec §10)")


def map_at_iou(*_args: np.ndarray) -> dict[str, float]:
    """mAP@IoU{0.1..0.5} temporal localization — Phase 7."""
    raise NotImplementedError("mAP@IoU is Phase-7 scope (spec §10)")


def equalize_window(length: int, target: int, anchor: str) -> tuple[int, int]:
    """``[start, end)`` of the ``target``-frame window kept from a ``length`` clip.

    Lesson **C28**: with every scored clip the same length, clip length carries
    no label information, so whatever AUC survives was earned from the features.
    Clips shorter than ``target`` cannot be cropped and are dropped by the
    caller — padding them would fabricate frames and interact with the score
    head's ``padding_mode="replicate"``.
    """
    if anchor not in constants.EQUALIZE_ANCHOR_CHOICES:
        raise ValueError(
            f"anchor must be one of {constants.EQUALIZE_ANCHOR_CHOICES}, got {anchor!r}"
        )
    if target <= 0:
        raise ValueError(f"--equalize-length must be positive, got {target}")
    if length < target:
        raise ValueError(f"Clip of {length} frames is shorter than target {target}")
    if anchor == constants.EQUALIZE_ANCHOR_START:
        start = 0
    elif anchor == constants.EQUALIZE_ANCHOR_END:
        start = length - target
    else:
        start = (length - target) // 2
    return start, start + target


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
    parser.add_argument(
        "--equalize-length", type=int, default=None,
        help="Lesson C28 control: crop every clip to exactly this many sampled "
             "frames and drop shorter ones, so clip length cannot carry the label",
    )
    parser.add_argument(
        "--equalize-anchor", choices=constants.EQUALIZE_ANCHOR_CHOICES,
        default=constants.EQUALIZE_ANCHOR_CENTER,
        help="Which window to keep under --equalize-length (default: center). "
             "Use 'end' where the event sits at the clip end, as on DADA-2000",
    )
    parser.add_argument("--score-norm", choices=constants.SCORE_NORM_CHOICES,
                        default=constants.SCORE_NORM_AUTO,
                        help="per-video score pooling before the micro metric "
                             "(auto: minmax if every eval video is abnormal)")
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
    dropped_short: list[str] = []
    positives_before = 0
    positives_after = 0
    for i in range(len(dataset)):
        item = dataset[i]
        video_id: str = item["video_id"]
        feats, frame_label = item["v_feat"], item["frame_label"]
        if args.equalize_length is not None:
            positives_before += int(frame_label.sum())
            if len(frame_label) < args.equalize_length:
                dropped_short.append(video_id)
                continue
            start, end = equalize_window(
                len(frame_label), args.equalize_length, args.equalize_anchor
            )
            feats, frame_label = feats[start:end], frame_label[start:end]
            positives_after += int(frame_label.sum())
        score, sim = sliding_window_scores(
            model, feats, class_feats_fn, cfg.data.max_vis_len
        )
        gt = frame_label.numpy()
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

    if args.equalize_length is not None:
        LOGGER.warning(
            "C28 length control: kept %d/%d clips at exactly %d frames "
            "(anchor=%s); dropped %d shorter clips; positives retained %d/%d "
            "(%.1f%%)",
            len(all_scores), len(dataset), args.equalize_length,
            args.equalize_anchor, len(dropped_short),
            positives_after, positives_before,
            100.0 * positives_after / positives_before if positives_before else 0.0,
        )
        if not all_scores:
            raise ValueError(
                f"--equalize-length {args.equalize_length} dropped every clip; "
                "pick a length at or below the corpus median"
            )

    metrics = pooled_metrics(all_scores, all_labels, args.score_norm)
    results: dict[str, object] = {
        "dataset": dataset_name,
        "num_videos": len(all_scores),
        "num_videos_total": len(dataset),
        **metrics,
        "checkpoint": str(args.ckpt or args.baseline_ckpt),
        "per_video": per_video,
    }
    if args.equalize_length is not None:
        results["equalize"] = {
            "length": args.equalize_length,
            "anchor": args.equalize_anchor,
            "clips_kept": len(all_scores),
            "clips_dropped": len(dropped_short),
            "dropped_ids": sorted(dropped_short),
            "positive_frames_before": positives_before,
            "positive_frames_after": positives_after,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / RESULTS_FILENAME
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    LOGGER.info(
        "AUC %.4f | AP %.4f | norm=%s (raw AUC %.4f) | macro AUC %.4f over %d/%d "
        "videos -> %s",
        results["auc"], results["ap"], results["score_norm"], results["auc_raw"],
        results["auc_macro"], results["auc_macro_videos"], len(all_scores),
        results_path,
    )


if __name__ == "__main__":
    main()
