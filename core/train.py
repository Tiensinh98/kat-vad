"""KAT-VAD training CLI (plain torch, spec §8).

Stages: ``1`` = KIP-only warm-up (``L_KIP_rec + L_KIP_align``, everything else
frozen), ``2`` = full objective (baseline losses + weighted KIP losses).

Resumability: batches are iterated manually from a per-epoch seeded
permutation (no DataLoader workers), and checkpoints carry model/optimizer/
scheduler state plus *all* RNG states (python, numpy, torch, dataset) and the
number of batches consumed in the current epoch — so ``--resume`` replays the
exact remaining RNG/data stream, mid-epoch included (weights match a straight
run within FP tolerance; see ``core/docs/TRAINING.md``). AMP and gradient
accumulation are flags.

Deviation from the baseline (documented in ``core/docs/TRAINING.md``): the
``L_dvs`` pair is applied only to normal or synthesized rows, because our DVS
yields an all-zero ``y^p`` for un-synthesized abnormal clips (unknown window
under weak supervision) and BCE toward all-zero would fight ``L_MIL``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR

from core import constants
from core.config import Config, load_config
from core.data.collate import collate_variable_length
from core.data.dataset import DVSFeatureDataset
from core.data.definitions import (
    NORMAL_QUERY_CAPTION,
    DatasetSpecVerbalizer,
    dataset_abbr,
    verbalize_class_name,
)
from core.data.knn_cache import load_knn_cache
from core.device import resolve_device
from core.kip.losses import (
    kinematic_loss,
    kip_alignment_loss,
    kip_reconstruction_loss,
)
from core.losses.contrastive import CapContrastLoss
from core.losses.dvs import pseudo_sup_mil_loss, supervised_loss
from core.losses.mil import mil_loss, multi_class_mil_loss
from core.models.kat_vad import KATVAD
from core.models.text_encoding import (
    TEXT_ENCODER_CHOICES,
    TEXT_ENCODER_CLIP,
    TextEncodeFn,
    make_text_encoder,
)

LOGGER = logging.getLogger(__name__)

CHECKPOINT_LAST = "checkpoint_last.pt"
METRICS_FILENAME = "metrics.jsonl"
STAGE_KIP_WARMUP = 1
STAGE_FULL = 2
# Only the frozen CLIP text tower may be absent from a warm-start source:
# stage 1 builds the model without it (load_clip=False).
TEXT_TOWER_PREFIX = "clip_text_model."


def set_global_seed(seed: int) -> None:
    """Seed python / numpy / torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def warm_start_model(model: torch.nn.Module, path: Path) -> None:
    """Load ONLY the model weights from a checkpoint (spec §8 stage 1 → 2).

    Unlike ``Trainer.load_checkpoint`` (same-run resume: restores optimizer,
    LR schedule, epoch/step counters and RNG streams), this seeds a *new* run
    with trained weights and leaves everything else fresh. Fail-loud: every
    key must match except the frozen CLIP text tower, which the stage-1
    model legitimately lacks (built with ``load_clip=False``) and which keeps
    its pinned pretrained weights in the target.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)  # nosec B614 - own ckpt
    state: dict[str, Tensor] = payload["model"]
    own_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    unexpected = sorted(ckpt_keys - own_keys)
    missing = sorted(
        k for k in own_keys - ckpt_keys if not k.startswith(TEXT_TOWER_PREFIX)
    )
    if unexpected or missing:
        raise ValueError(
            f"--init-weights checkpoint does not match the model: "
            f"missing={missing[:5]} unexpected={unexpected[:5]} "
            f"(only '{TEXT_TOWER_PREFIX}*' may be absent from the source; "
            f"e.g. a KIP-off run cannot warm-start from a KIP-on checkpoint)"
        )
    model.load_state_dict(state, strict=False)
    LOGGER.info(
        "Warm-started model weights from %s (source epoch=%s step=%s); "
        "optimizer/schedule/counters start fresh",
        path, payload.get("epoch"), payload.get("global_step"),
    )


def cosine_warmup_lambda(warmup_steps: int, total_steps: int) -> Any:
    """LR multiplier: linear 0→1 over ``warmup_steps``, cosine 1→0 afterwards."""

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / remaining)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule


def epoch_permutation(seed: int, epoch: int, n: int) -> list[int]:
    """Deterministic sample order for one epoch."""
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(n, generator=generator).tolist()


def class_index_tensor(
    cls_labels: list[str], class_names: list[str], device: torch.device
) -> Tensor:
    """Map per-video class-name strings to indices in ``class_names`` (Normal=0)."""
    indices = []
    for name in cls_labels:
        if name not in class_names:
            raise ValueError(
                f"Class {name!r} not in the dataset class list {class_names}; "
                "regenerate defs.json or fix the annotation"
            )
        indices.append(class_names.index(name))
    return torch.tensor(indices, dtype=torch.long, device=device)


class Trainer:
    """Plain-torch training loop with full-state checkpoint/resume."""

    def __init__(
        self,
        cfg: Config,
        model: KATVAD,
        dataset: DVSFeatureDataset,
        class_names: list[str],
        text_encode_fn: TextEncodeFn,
        output_dir: Path,
        device: torch.device | None = None,
        log_every: int = 10,
    ) -> None:
        self.cfg = cfg
        self.device = device if device is not None else resolve_device(cfg.train.device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.class_names = class_names
        self.text_encode_fn = text_encode_fn
        self.output_dir = output_dir
        self.log_every = log_every
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.verbalizer = DatasetSpecVerbalizer(
            dataset_abbr(cfg.data.dataset), rng=random.Random(cfg.train.seed)  # nosec B311
        )
        self.cap_contrast = CapContrastLoss(
            cfg.loss.contrastive_neg_mining, cfg.loss.contrastive_temp
        )

        self.stage = cfg.train.stage
        if self.stage not in (STAGE_KIP_WARMUP, STAGE_FULL):
            raise ValueError(f"Unknown training stage: {self.stage}")
        if self.stage == STAGE_KIP_WARMUP:
            if self.model.kip is None:
                raise ValueError("Stage 1 (KIP warm-up) requires kip.enabled=true")
            for name, param in self.model.named_parameters():
                param.requires_grad = name.startswith("kip.")

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
        )
        steps_per_epoch = math.ceil(
            math.ceil(len(dataset) / cfg.train.batch_size) / cfg.train.grad_accum_steps
        )
        total_steps = max(1, steps_per_epoch * cfg.train.num_epochs)
        self.scheduler = LambdaLR(
            self.optimizer, cosine_warmup_lambda(cfg.train.warmup_steps, total_steps)
        )
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.epoch = 0  # completed epochs
        self.global_step = 0  # optimizer steps taken
        self._resume_batches_done = 0

    # ------------------------------------------------------------------ losses
    def _kip_loss_flags(self) -> tuple[bool, bool]:
        """(rec+align active, kin active) for the current config/stage."""
        kip = self.cfg.kip
        if not kip.enabled or self.model.kip is None:
            return False, False
        if self.stage == STAGE_KIP_WARMUP:
            return True, False
        kin = kip.use_lkin and not kip.pmg_only
        return True, kin

    def compute_losses(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Forward + full loss dict for one collated batch (spec §8 stage 2)."""
        loss_cfg = self.cfg.loss
        v_feat = batch["v_feat"].to(self.device)
        e_o = batch["e_o"].to(self.device)
        lengths = batch["v_feat_l"].to(self.device)
        mask = batch["mask"].to(self.device)
        pseudo = batch["pseudo_frame_label"].to(self.device)
        labels = batch["label"].to(self.device).float()
        is_synth = batch["is_synthesized"].to(self.device)
        cls_labels: list[str] = batch["cls_label"]

        rec_align_on, kin_on = self._kip_loss_flags()

        losses: dict[str, Tensor] = {}
        if self.stage == STAGE_KIP_WARMUP:
            # visual path only: no text, no heads
            outputs = self.model(v_feat, lengths)
            flow_proj, rgb_proj = self.model.kip.project_for_align(  # type: ignore[union-attr]
                outputs["eo_hat"], outputs["vt"]
            )
            losses["kip_rec"] = kip_reconstruction_loss(outputs["eo_hat"], e_o, mask)
            losses["kip_align"] = kip_alignment_loss(
                flow_proj,
                rgb_proj,
                mask,
                tau=loss_cfg.tau_align,
                subsample=loss_cfg.align_subsample,
                exclude_window=loss_cfg.align_exclude_window,
            )
            losses["total"] = (
                loss_cfg.lambda_rec * losses["kip_rec"]
                + loss_cfg.lambda_align * losses["kip_align"]
            )
            return losses

        # ---------------- stage 2: full objective ----------------
        cls_idx = class_index_tensor(cls_labels, self.class_names, self.device)
        class_feats = self.text_encode_fn(
            [verbalize_class_name(self.verbalizer, name) for name in self.class_names]
        )

        captions: list[str] | None = None
        cap_labels: Tensor | None = None
        if loss_cfg.captions_from_definitions:
            abnormal_caps = [
                verbalize_class_name(self.verbalizer, name)
                for name, idx in zip(cls_labels, cls_idx.tolist(), strict=True)
                if idx != 0
            ]
            if abnormal_caps:
                captions = [NORMAL_QUERY_CAPTION, *abnormal_caps]
                cap_ids, ano_cnt = [], 1
                for idx in cls_idx.tolist():
                    if idx == 0:
                        cap_ids.append(0)
                    else:
                        cap_ids.append(ano_cnt)
                        ano_cnt += 1
                cap_labels = torch.tensor(cap_ids, dtype=torch.long, device=self.device)
        caption_feats = self.text_encode_fn(captions) if captions else None

        outputs = self.model(v_feat, lengths, class_feats, caption_feats)

        # L_MIL (binary top-k)
        loss_bin = mil_loss(
            outputs["cls_bin_logits"], labels, lengths, topk_pct=loss_cfg.mil_topk_pct
        )
        if "cap_bin_logits" in outputs:
            loss_bin = loss_bin + mil_loss(
                outputs["cap_bin_logits"], labels, lengths, topk_pct=loss_cfg.mil_topk_pct
            )
        losses["mil"] = loss_bin

        # L_dvs pair — only rows whose y^p is meaningful (normal or synthesized)
        dvs_rows = ((labels < 0.5) | is_synth.bool()).nonzero(as_tuple=True)[0]
        if dvs_rows.numel() > 0:
            sub = (
                outputs["cls_bin_logits"][dvs_rows],
                pseudo[dvs_rows],
                lengths[dvs_rows],
            )
            losses["dvs_sup"] = supervised_loss(*sub)
            losses["dvs_sup_mil"] = pseudo_sup_mil_loss(
                *sub, topk_pct=loss_cfg.sup_mil_topk_pct
            )
            if "cap_bin_logits" in outputs:
                cap_sub = outputs["cap_bin_logits"][dvs_rows]
                losses["dvs_sup"] = losses["dvs_sup"] + supervised_loss(
                    cap_sub, pseudo[dvs_rows], lengths[dvs_rows]
                )
                losses["dvs_sup_mil"] = losses["dvs_sup_mil"] + pseudo_sup_mil_loss(
                    cap_sub,
                    pseudo[dvs_rows],
                    lengths[dvs_rows],
                    topk_pct=loss_cfg.sup_mil_topk_pct,
                )

        # L_MIL-align (multi-class)
        loss_mul = multi_class_mil_loss(
            outputs["cls_sim_mat"], cls_idx, lengths, topk_pct=loss_cfg.mul_mil_topk_pct
        )
        if "cap_sim_mat" in outputs and cap_labels is not None:
            loss_mul = loss_mul + multi_class_mil_loss(
                outputs["cap_sim_mat"], cap_labels, lengths,
                topk_pct=loss_cfg.mul_mil_topk_pct,
            )
        losses["mul_mil"] = loss_mul

        # L_neg (caption contrastive with hard-negative mining)
        if (
            caption_feats is not None
            and loss_cfg.cap_contrastive_weight > 0.0
            and int((cls_idx != 0).sum()) > 0
        ):
            losses["cap_contrastive"] = self.cap_contrast(
                outputs["cls_bin_logits"],
                lengths,
                outputs["vis_feats"],
                caption_feats[1:],
                cls_idx,
                pseudo,
            )

        # KIP losses
        if rec_align_on:
            flow_proj, rgb_proj = self.model.kip.project_for_align(  # type: ignore[union-attr]
                outputs["eo_hat"], outputs["vt"]
            )
            losses["kip_rec"] = kip_reconstruction_loss(outputs["eo_hat"], e_o, mask)
            losses["kip_align"] = kip_alignment_loss(
                flow_proj,
                rgb_proj,
                mask,
                tau=loss_cfg.tau_align,
                subsample=loss_cfg.align_subsample,
                exclude_window=loss_cfg.align_exclude_window,
            )
        if kin_on:
            losses["kin"] = kinematic_loss(
                outputs["motion_scores"],
                outputs["cls_bin_logits"].detach().sigmoid(),
                labels,
                mask=mask,
                pseudo_labels=pseudo,
                is_synthesized=is_synth,
                use_yp_anchor=loss_cfg.use_yp_anchor,
                beta=loss_cfg.beta_cons,
                topk_pct=loss_cfg.mil_topk_pct,
            )

        total = losses["mil"] + loss_cfg.mul_weight * losses["mul_mil"]
        if "dvs_sup" in losses:
            total = total + loss_cfg.pseudo_sup_weight * losses["dvs_sup"]
            total = total + loss_cfg.pseudo_sup_mil_weight * losses["dvs_sup_mil"]
        if "cap_contrastive" in losses:
            total = total + loss_cfg.cap_contrastive_weight * losses["cap_contrastive"]
        if "kip_rec" in losses:
            total = total + loss_cfg.lambda_rec * losses["kip_rec"]
            total = total + loss_cfg.lambda_align * losses["kip_align"]
        if "kin" in losses:
            total = total + loss_cfg.gamma_kin * losses["kin"]
        losses["total"] = total
        return losses

    # ------------------------------------------------------------ checkpoints
    def _rng_payload(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            "dataset": self.dataset._rng.getstate(),
            "verbalizer": self.verbalizer._rng.getstate(),
        }

    def _restore_rng(self, payload: dict[str, Any]) -> None:
        random.setstate(payload["python"])
        np.random.set_state(payload["numpy"])
        torch.set_rng_state(torch.as_tensor(payload["torch"], dtype=torch.uint8))
        if payload.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda"])
        if payload.get("mps") is not None and torch.backends.mps.is_available():
            torch.mps.set_rng_state(payload["mps"])
        self.dataset._rng.setstate(payload["dataset"])
        self.verbalizer._rng.setstate(payload["verbalizer"])

    def save_checkpoint(self, path: Path, batches_done: int = 0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "batches_done": batches_done,
                "config": self.cfg.to_dict(),
                "class_names": self.class_names,
                "rng": self._rng_payload(),
            },
            path,
        )
        LOGGER.info("Checkpoint saved: %s (epoch=%d step=%d)", path, self.epoch, self.global_step)

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)  # nosec B614 - own ckpt
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.epoch = payload["epoch"]
        self.global_step = payload["global_step"]
        self._resume_batches_done = payload.get("batches_done", 0)
        self._restore_rng(payload["rng"])
        LOGGER.info(
            "Resumed from %s (epoch=%d step=%d batches_done=%d)",
            path, self.epoch, self.global_step, self._resume_batches_done,
        )

    # ------------------------------------------------------------------ train
    def _log_metrics(self, record: dict[str, Any]) -> None:
        with (self.output_dir / METRICS_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def train(self, stop_after_epochs: int | None = None) -> None:
        """Run to ``train.num_epochs``; ``stop_after_epochs`` caps how many
        epochs *this invocation* completes (time-boxed Colab sessions / the
        kill-and-resume test) without touching the LR-schedule horizon."""
        cfg_t = self.cfg.train
        batch_size = cfg_t.batch_size
        num_batches = math.ceil(len(self.dataset) / batch_size)
        self.model.train()

        for epochs_this_run, epoch in enumerate(
            range(self.epoch, cfg_t.num_epochs), start=1
        ):
            order = epoch_permutation(cfg_t.seed, epoch, len(self.dataset))
            start_batch = self._resume_batches_done
            self._resume_batches_done = 0
            self.optimizer.zero_grad(set_to_none=True)
            epoch_losses: list[float] = []

            for batch_idx in range(start_batch, num_batches):
                indices = order[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                samples = [self.dataset[i] for i in indices]
                batch = collate_variable_length(samples)

                # plain context off-AMP: autocast rejects some device types
                # (e.g. mps on torch 2.4) even when disabled
                amp_ctx: Any = (
                    torch.autocast(self.device.type, enabled=True)
                    if self.use_amp
                    else contextlib.nullcontext()
                )
                with amp_ctx:
                    losses = self.compute_losses(batch)
                total = losses["total"] / cfg_t.grad_accum_steps
                self.scaler.scale(total).backward()

                is_boundary = (
                    (batch_idx + 1) % cfg_t.grad_accum_steps == 0
                    or batch_idx == num_batches - 1
                )
                if is_boundary:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1

                epoch_losses.append(float(losses["total"].detach()))
                record = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": self.global_step,
                    "lr": self.scheduler.get_last_lr()[0],
                    **{k: float(v.detach()) for k, v in losses.items()},
                }
                self._log_metrics(record)
                if batch_idx % self.log_every == 0:
                    LOGGER.info(
                        "epoch %d batch %d/%d loss %.4f",
                        epoch, batch_idx, num_batches, record["total"],
                    )

                if (
                    cfg_t.checkpoint_every_steps > 0
                    and is_boundary
                    and self.global_step % cfg_t.checkpoint_every_steps == 0
                ):
                    self.save_checkpoint(
                        self.output_dir / f"checkpoint_step_{self.global_step}.pt",
                        batches_done=batch_idx + 1,
                    )
                    self.save_checkpoint(
                        self.output_dir / CHECKPOINT_LAST, batches_done=batch_idx + 1
                    )

            self.epoch = epoch + 1
            self.save_checkpoint(self.output_dir / CHECKPOINT_LAST)
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            LOGGER.info("epoch %d done: mean loss %.4f", epoch, mean_loss)
            if stop_after_epochs is not None and epochs_this_run >= stop_after_epochs:
                LOGGER.info("Stopping after %d epoch(s) this run (resume to continue)",
                            epochs_this_run)
                return

        LOGGER.info("Training complete: %d epochs, %d steps", self.epoch, self.global_step)


def load_class_names(data_dir: Path) -> list[str]:
    """Read the global class-name list from ``defs.json`` (Normal first)."""
    with (data_dir / constants.DEFS_FILENAME).open("r", encoding="utf-8") as fh:
        defs = json.load(fh)
    if not isinstance(defs, list) or not defs:
        raise ValueError(
            f"{constants.DEFS_FILENAME} must be a non-empty class-name list, got {type(defs)}"
        )
    return list(defs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train KAT-VAD (spec §8)")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="SECTION.KEY=VALUE", help="config override (repeatable)",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="dataset dir with labels/defs (default: data root + dataset)")
    parser.add_argument("--clip-dir", type=Path, default=None,
                        help="CLIP feature cache dir (default: cache/clip/<dataset>)")
    parser.add_argument("--flow-dir", type=Path, default=None,
                        help="flow cache dir (default: cache/flow/v1/<dataset>)")
    parser.add_argument("--knn-cache", type=Path, default=None,
                        help="DVS KNN cache .npz (optional)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="checkpoints + metrics destination (Drive on Colab)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="checkpoint to resume from (e.g. <output>/checkpoint_last.pt)")
    parser.add_argument("--init-weights", type=Path, default=None,
                        help="checkpoint whose MODEL WEIGHTS seed this run "
                             "(stage 1 → stage 2 warm-start); optimizer, LR "
                             "schedule and counters start fresh")
    parser.add_argument("--stop-after-epochs", type=int, default=None,
                        help="cap epochs completed by THIS invocation (time-boxed "
                             "sessions); schedule horizon stays train.num_epochs")
    parser.add_argument("--text-encoder", choices=TEXT_ENCODER_CHOICES,
                        default=TEXT_ENCODER_CLIP,
                        help="'clip' needs HF weights; 'stub' is the data-free smoke mode")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.resume is not None and args.init_weights is not None:
        parser.error("--resume and --init-weights are mutually exclusive: "
                     "resume continues a run, init-weights starts a new one")
    cfg = load_config(args.config, args.overrides)
    set_global_seed(cfg.train.seed)

    dataset_name = cfg.data.dataset
    data_dir = args.data_dir if args.data_dir else constants.DATA_ROOT / dataset_name
    clip_dir = args.clip_dir if args.clip_dir else constants.CLIP_CACHE_DIR / dataset_name
    flow_dir = args.flow_dir if args.flow_dir else constants.FLOW_CACHE_DIR / dataset_name

    knn_cache = load_knn_cache(args.knn_cache) if args.knn_cache else None
    require_flow = cfg.kip.enabled and not cfg.kip.disable_pmg
    dataset = DVSFeatureDataset(
        data_dir=data_dir,
        clip_dir=clip_dir,
        flow_dir=flow_dir if require_flow else None,
        dvs=cfg.dvs,
        vis_max_len=cfg.data.max_vis_len,
        is_egocentric=cfg.data.is_egocentric,
        require_flow=require_flow,
        knn_cache=knn_cache,
        seed=cfg.train.seed,
    )
    class_names = load_class_names(data_dir)

    device = resolve_device(cfg.train.device)
    needs_clip = args.text_encoder == TEXT_ENCODER_CLIP and cfg.train.stage == STAGE_FULL
    model = KATVAD.from_config(cfg, load_clip=needs_clip)
    text_encode_fn = make_text_encoder(
        model, args.text_encoder, device, dim=cfg.model.hidden_dim
    )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        dataset=dataset,
        class_names=class_names,
        text_encode_fn=text_encode_fn,
        output_dir=args.output_dir,
        device=device,
        log_every=args.log_every,
    )
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    elif args.init_weights is not None:
        warm_start_model(trainer.model, args.init_weights)
    cfg.save_yaml(args.output_dir / "config.yaml")
    trainer.train(stop_after_epochs=args.stop_after_epochs)


if __name__ == "__main__":
    main()
