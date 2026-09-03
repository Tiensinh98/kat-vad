"""KAT-VAD inference CLI (spec §9) — RGB + text only, no RAFT, no DVS.

Scores single feature files, directories of cached features, or (with CLIP
weights) a raw video via the extraction path. Full-length videos are scored in
sliding windows of ``max_vis_len`` (baseline ``full_length_eval`` convention)
and the per-window curves are concatenated.

Outputs one ``<video_id>.npz`` per video into ``--output-dir`` with:
``score (L,)`` sigmoid anomaly curve, ``sim (L, C)`` per-frame category
sigmoids, and ``class_names``. ``core/evaluate.py`` and
``core/tools/visualize.py`` consume the same format.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from core import constants
from core.config import Config, load_config
from core.data.definitions import (
    DatasetSpecVerbalizer,
    dataset_abbr,
    verbalize_class_name,
)
from core.device import resolve_device
from core.kip import gate_shift
from core.models.ckpt_compat import load_baseline_checkpoint, load_kip_state_dict
from core.models.kat_vad import KATVAD, KIP_DIAG_PREFIX
from core.models.text_encoding import (
    TEXT_ENCODER_CHOICES,
    TEXT_ENCODER_CLIP,
    TextEncodeFn,
    make_text_encoder,
)

LOGGER = logging.getLogger(__name__)

ClassFeatsFn = Callable[[], Tensor]

# KIP gate diagnostics are stored as kip_<name> keys inside the existing .npz.
KIP_DIAG_NPZ_PREFIX = "kip_"


def load_model_for_scoring(
    cfg: Config,
    device: torch.device,
    ckpt: Path | None,
    baseline_ckpt: Path | None,
    text_encoder: str,
) -> KATVAD:
    """Build a KATVAD in eval mode from ours or a LaGoVAD checkpoint."""
    if (ckpt is None) == (baseline_ckpt is None):
        raise ValueError("Provide exactly one of --ckpt / --baseline-ckpt")
    needs_clip = text_encoder == TEXT_ENCODER_CLIP
    model = KATVAD.from_config(cfg, load_clip=needs_clip, training=False)
    if ckpt is not None:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)  # nosec B614 - own ckpt
        state = payload.get("model", payload)
        report = load_kip_state_dict(model, state)
        LOGGER.info(
            "Loaded KAT-VAD checkpoint %s (%d tensors, %d train-only dropped)",
            ckpt,
            len(report.loaded),
            len(report.skipped_train_only),
        )
    else:
        assert baseline_ckpt is not None  # nosec B101 - narrowing for type-checkers
        report = load_baseline_checkpoint(model, baseline_ckpt)
        LOGGER.info(
            "Loaded LaGoVAD checkpoint %s (%d tensors)", baseline_ckpt, len(report.loaded)
        )
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sliding_window_scores(
    model: KATVAD,
    features: Tensor,
    class_feats_fn: ClassFeatsFn,
    max_vis_len: int = constants.MAX_VIS_LEN,
    kip_diagnostics: bool = False,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Score one full-length video: ``(L,) sigmoid curve, (L, C) sim sigmoids``.

    ``class_feats_fn`` is called once per window (baseline re-samples the
    verbalized definitions each window).

    With ``kip_diagnostics``, the third return value carries the per-position
    gate internals concatenated across windows (empty dict otherwise). Note the
    windowing: ``s_t`` comes from a *within-window* rank, so a video longer than
    ``max_vis_len`` is ranked per window, not globally — which is exactly what
    the scored model does, and is why these arrays are the right ones to read.
    """
    device = next(model.parameters()).device
    length = len(features)
    bin_parts: list[Tensor] = []
    sim_parts: list[Tensor] = []
    diag_parts: dict[str, list[Tensor]] = {}
    for start in range(0, length, max_vis_len):
        window = features[start : start + max_vis_len].unsqueeze(0).to(device)
        lengths = torch.tensor([window.shape[1]], device=device)
        outputs = model(
            window,
            lengths,
            class_feats=class_feats_fn(),
            return_kip_diagnostics=kip_diagnostics,
        )
        bin_parts.append(outputs["cls_bin_logits"][0].sigmoid().cpu())
        sim_parts.append(outputs["cls_sim_mat"][0].sigmoid().cpu())
        for key, value in outputs.items():
            if key.startswith(KIP_DIAG_PREFIX):
                diag_parts.setdefault(key[len(KIP_DIAG_PREFIX) :], []).append(value[0].cpu())
    diagnostics = {k: torch.cat(v, dim=0)[:length] for k, v in diag_parts.items()}
    return torch.cat(bin_parts, dim=0)[:length], torch.cat(sim_parts, dim=0)[:length], diagnostics


def make_class_feats_fn(
    text_encode_fn: TextEncodeFn,
    class_names: list[str],
    verbalizer: DatasetSpecVerbalizer | None,
) -> ClassFeatsFn:
    """Per-window class-feature provider (verbalized when a verbalizer is given)."""

    def provide() -> Tensor:
        if verbalizer is None:
            return text_encode_fn(class_names)
        return text_encode_fn(
            [verbalize_class_name(verbalizer, name) for name in class_names]
        )

    return provide


def expand_to_frames(scores: np.ndarray, stride: int) -> np.ndarray:
    """A5 clip→frame expansion: repeat each sampled-frame score ``stride`` times."""
    return np.repeat(scores, stride, axis=0)


def resolve_class_names(
    defs_path: Path | None, class_names_arg: str | None
) -> list[str]:
    if (defs_path is None) == (class_names_arg is None):
        raise ValueError("Provide exactly one of --defs / --class-names")
    if class_names_arg is not None:
        names = [n.strip() for n in class_names_arg.split(",") if n.strip()]
        if not names:
            raise ValueError("--class-names must contain at least one name")
        return names
    assert defs_path is not None  # nosec B101 - narrowing for type-checkers
    with defs_path.open("r", encoding="utf-8") as fh:
        defs = json.load(fh)
    if not isinstance(defs, list) or not defs:
        raise ValueError(f"{defs_path} must hold a non-empty class-name list")
    return list(defs)


def iter_feature_files(features: Path) -> Iterator[Path]:
    if features.is_dir():
        yield from sorted(features.glob("*.npy"))
    else:
        yield features


def extract_video_features(video: Path, device: torch.device) -> Tensor:
    """Raw-video path: frozen CLIP tower over stride-8 frames (needs HF weights)."""
    from core.tools.extract_clip_features import encode_video, load_pretrained_encoder

    encoder = load_pretrained_encoder(device)
    return torch.from_numpy(encode_video(video, encoder, device))


def score_to_npz(
    out_dir: Path,
    video_id: str,
    score: Tensor,
    sim: Tensor,
    class_names: list[str],
    gt: np.ndarray | None = None,
    kip_diagnostics: dict[str, Tensor] | None = None,
) -> Path:
    """Write one clip's scores, and optionally its KIP gate diagnostics.

    Diagnostics land as ``kip_<name>`` keys alongside ``score``/``sim``: no new
    artifact format and no new directory layout, so every existing reader keeps
    working and ``rescore``/``evaluate`` are unaffected. ``s`` is written as
    int16 (it is bounded by ``D/K = 128``), the rest as float32.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{video_id}.npz"
    payload: dict[str, np.ndarray] = {
        "score": score.numpy().astype(np.float32),
        "sim": sim.numpy().astype(np.float32),
        "class_names": np.array(class_names),
    }
    if gt is not None:
        payload["gt"] = gt.astype(np.float32)
    for name, value in (kip_diagnostics or {}).items():
        array = value.numpy()
        payload[f"{KIP_DIAG_NPZ_PREFIX}{name}"] = (
            array.astype(np.int16) if name == "s" else array.astype(np.float32)
        )
    np.savez(path, **payload)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KAT-VAD inference (spec §9)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    parser.add_argument("--ckpt", type=Path, default=None, help="KAT-VAD checkpoint")
    parser.add_argument("--baseline-ckpt", type=Path, default=None,
                        help="LaGoVAD best.ckpt (compat loader)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features", type=Path,
                        help="cached CLIP feature .npy file or directory")
    source.add_argument("--video", type=Path,
                        help="raw video (extraction path; requires CLIP weights)")
    parser.add_argument("--defs", type=Path, default=None,
                        help="defs.json with the class-name list")
    parser.add_argument("--class-names", type=str, default=None,
                        help="comma-separated class names (Normal first)")
    parser.add_argument("--no-verbalize", action="store_true",
                        help="encode raw class names instead of sampled definitions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-encoder", choices=TEXT_ENCODER_CHOICES,
                        default=TEXT_ENCODER_CLIP)
    parser.add_argument("--gate-type", choices=gate_shift.GATE_TYPES, default=None,
                        help="override kip.gate_type; must match what trained the "
                             "checkpoint (a mismatch raises)")
    parser.add_argument("--dump-kip-diag", dest="dump_kip_diag",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="write KIP gate diagnostics (s, gate_ratio, m, mu_norm, "
                             "eo_norm) into each score .npz. On by default here; the "
                             "training path never collects them.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.gate_type is not None:
        cfg.kip.gate_type = args.gate_type
        if args.gate_type not in gate_shift.MLP_GATE_TYPES:
            # rank/constant consume no intensity signal; keeping a stale one
            # would fail validation for a key the CLI just made irrelevant.
            cfg.kip.gate_signal = None
        cfg.kip.validate()
        LOGGER.info("Gate type overridden from CLI: %s", args.gate_type)
    device = resolve_device(cfg.train.device)

    class_names = resolve_class_names(args.defs, args.class_names)
    model = load_model_for_scoring(cfg, device, args.ckpt, args.baseline_ckpt, args.text_encoder)
    text_encode_fn = make_text_encoder(model, args.text_encoder, device, dim=cfg.model.hidden_dim)
    verbalizer = None
    if not args.no_verbalize:
        # Seeded for reproducible scoring (see core/evaluate.py).
        verbalizer = DatasetSpecVerbalizer(
            dataset_abbr(cfg.data.dataset),
            rng=random.Random(constants.SEED),  # nosec B311 - not security-sensitive
        )
    class_feats_fn = make_class_feats_fn(text_encode_fn, class_names, verbalizer)

    if args.video is not None:
        features_by_id = {args.video.stem: extract_video_features(args.video, device)}
    else:
        features_by_id = {
            path.stem: torch.from_numpy(np.load(path).astype(np.float32))
            for path in iter_feature_files(args.features)
        }
    if not features_by_id:
        raise FileNotFoundError(f"No feature files found under {args.features}")

    dump_diag = args.dump_kip_diag and cfg.kip.enabled
    if args.dump_kip_diag and not cfg.kip.enabled:
        LOGGER.info("--dump-kip-diag requested but KIP is disabled; nothing to log")
    for video_id, feats in features_by_id.items():
        score, sim, diagnostics = sliding_window_scores(
            model, feats, class_feats_fn, cfg.data.max_vis_len, kip_diagnostics=dump_diag
        )
        path = score_to_npz(
            args.output_dir,
            video_id,
            score,
            sim,
            class_names,
            kip_diagnostics=diagnostics or None,
        )
        LOGGER.info("%s: %d frames scored -> %s", video_id, len(score), path)


if __name__ == "__main__":
    main()
