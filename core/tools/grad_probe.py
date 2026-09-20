"""Gradient attribution probe — who is actually steering the shared trunk.

``L_KIP_rec`` is a bare MSE against unnormalized RAFT statistics while every
other term in the objective is a BCE or an InfoNCE, so the two live on different
scales by construction. A loss *value* cannot settle what that costs: the
question is how much of the gradient arriving at the **temporal encoder** — the
one module ``PMG`` and the anomaly heads share — comes from each term.

This probe answers it directly. For one batch it computes the full stage-2 loss
once, then takes a separate ``autograd.grad`` per weighted term against each
parameter group, and reports:

- the **L2 norm** each term contributes to each group, and the ratio
  ``|g_KIP| / |g_task|`` at the trunk;
- the **cosine** between the task gradient and each KIP gradient at the trunk.
  A negative cosine means the two are pulling the encoder apart — the terms are
  in direct conflict, not merely unequal. An orthogonal one (``~0``) means KIP
  is spending trunk capacity rather than fighting for it.

Nothing is optimized: no ``optimizer.step``, no checkpoint written. The probe is
read-only with respect to the run it inspects. ``--checkpoint`` therefore loads
**model weights only** (via :func:`core.train.warm_start_model`, not the strict
same-run resume path), which is also what lets a *stage-1* checkpoint be probed
under the stage-2 objective: stage 1 builds the model with ``load_clip=False``,
so it carries no CLIP text tower, and that state is precisely what stage 2
started from.

**Precision.** The probe runs in fp32 even where training used AMP. The headline
quantities are *ratios and cosines* between gradients of one graph, and a global
loss-scale factor cancels in both.

CLI::

    python -m core.tools.grad_probe \\
      --config outputs/.../stage2_kip_on/config.yaml \\
      --data-dir "$DATA/DADA2000_orig" \\
      --clip-dir "$CACHE/clip/DADA2000_orig" \\
      --flow-dir "$CACHE/flow/v1/DADA2000_orig" \\
      --checkpoint runs/s2024/stage2_kip_on/checkpoint_last.pt \\
      --output-dir outputs/DIAG/grad_probe/s2024_end \\
      --num-batches 8 --text-encoder stub
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from core.config import Config, load_config
from core.data.collate import collate_variable_length
from core.models.text_encoding import TEXT_ENCODER_CHOICES, TEXT_ENCODER_CLIP
from core.train import (
    STAGE_FULL,
    Trainer,
    build_trainer,
    epoch_permutation,
    set_global_seed,
    warm_start_model,
)

LOGGER = logging.getLogger(__name__)

# Parameter groups, by the prefix of their qualified name in KATVAD. The trunk is
# the only module both KIP and the anomaly heads write gradients into, so it is
# the group the whole diagnostic exists to measure.
PARAM_GROUPS: dict[str, tuple[str, ...]] = {
    "temporal_encoder": ("temporal_encoder.",),
    "kip": ("kip.",),
    "fusion": ("fusion.",),
    "heads": ("bin_head.", "sim_head."),
}
TRUNK_GROUP = "temporal_encoder"
KIP_TERMS = ("kip_rec", "kip_align", "kin")
# compute_losses assembles `total` itself; this table only mirrors the weights so
# the probe can split it. TOTAL_MISMATCH_TOL guards the mirror: if train.py's
# assembly changes and this table does not, the check fires instead of silently
# reporting an attribution of the wrong sum.
TOTAL_MISMATCH_TOL = 1e-4
REPORT_JSON = "grad_probe.json"
REPORT_MARKDOWN = "grad_probe.md"


def term_weights(cfg: Config) -> dict[str, float]:
    """Weight each loss key carries in ``Trainer.compute_losses``'s ``total``."""
    loss = cfg.loss
    return {
        "mil": 1.0,
        "mul_mil": loss.mul_weight,
        "bottomk": loss.bottomk_weight,
        "dvs_sup": loss.pseudo_sup_weight,
        "dvs_sup_mil": loss.pseudo_sup_mil_weight,
        "cap_contrastive": loss.cap_contrastive_weight,
        "kip_rec": loss.lambda_rec,
        "kip_align": loss.lambda_align,
        "kin": loss.gamma_kin,
    }


def collect_groups(model: torch.nn.Module) -> dict[str, list[Tensor]]:
    """Trainable parameters of each :data:`PARAM_GROUPS` entry, in a stable order."""
    groups: dict[str, list[Tensor]] = {name: [] for name in PARAM_GROUPS}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        for group, prefixes in PARAM_GROUPS.items():
            if name.startswith(prefixes):
                groups[group].append(param)
                break
    return {name: params for name, params in groups.items() if params}


def flat_gradient(term: Tensor, params: list[Tensor]) -> Tensor:
    """``d term / d params`` flattened into one vector; unused params read zero.

    ``allow_unused`` is required and not a shortcut: ``L_KIP_rec`` genuinely does
    not reach the fusion block or the heads, and a term that touches nothing in a
    group must report 0, not raise.
    """
    grads = torch.autograd.grad(
        term, params, retain_graph=True, allow_unused=True, materialize_grads=True
    )
    return torch.cat([g.detach().reshape(-1).float() for g in grads])


def probe_batch(trainer: Trainer, batch: dict[str, Any]) -> dict[str, Any]:
    """Per-term gradient norms and trunk cosines for one collated batch."""
    weights = term_weights(trainer.cfg)
    losses = trainer.compute_losses(batch)
    present = {k: v for k, v in losses.items() if k != "total"}
    unknown = sorted(set(present) - set(weights))
    if unknown:
        raise ValueError(
            f"Loss keys {unknown} have no entry in term_weights(); the probe would "
            "attribute a sum that is not the one being optimized"
        )
    terms = {k: weights[k] * v for k, v in present.items()}
    rebuilt = torch.stack(list(terms.values())).sum()
    drift = float((rebuilt - losses["total"]).abs())
    if drift > TOTAL_MISMATCH_TOL:
        raise ValueError(
            f"Reassembled total {float(rebuilt):.6f} != compute_losses total "
            f"{float(losses['total']):.6f} (drift {drift:.2e}). term_weights() has "
            "fallen out of step with Trainer.compute_losses — fix the table, do not "
            "raise the tolerance."
        )

    task_terms = [v for k, v in terms.items() if k not in KIP_TERMS]
    if not task_terms:
        raise ValueError(
            "No task loss in this batch: the probe needs a stage-2 objective to "
            "attribute against (train.stage=2)"
        )
    task_total = torch.stack(task_terms).sum()

    groups = collect_groups(trainer.model)
    vectors: dict[str, dict[str, Tensor]] = {}
    for group, params in groups.items():
        vectors[group] = {"task": flat_gradient(task_total, params)}
        for key in KIP_TERMS:
            if key in terms:
                vectors[group][key] = flat_gradient(terms[key], params)

    record: dict[str, Any] = {
        "losses": {k: float(v.detach()) for k, v in losses.items()},
        "weighted": {k: float(v.detach()) for k, v in terms.items()},
        "grad_norm": {
            group: {k: float(v.norm()) for k, v in per_term.items()}
            for group, per_term in vectors.items()
        },
    }
    trunk = vectors.get(TRUNK_GROUP)
    if trunk is not None:
        task_vec = trunk["task"]
        task_norm = float(task_vec.norm())
        record["trunk"] = {
            "task_norm": task_norm,
            "kip_over_task": {
                key: (float(vec.norm()) / task_norm if task_norm > 0 else math.inf)
                for key, vec in trunk.items()
                if key != "task"
            },
            "cosine_with_task": {
                key: float(F.cosine_similarity(vec, task_vec, dim=0))
                for key, vec in trunk.items()
                if key != "task"
            },
        }
    return record


def run_probe(
    trainer: Trainer, num_batches: int, epoch: int, batch_size: int | None = None
) -> dict[str, Any]:
    """Probe the first ``num_batches`` batches of ``epoch``'s real training order."""
    if trainer.stage != STAGE_FULL:
        raise ValueError(
            f"grad_probe needs train.stage={STAGE_FULL}: stage 1 freezes everything "
            "outside kip.* , so there is no shared trunk gradient to attribute"
        )
    if trainer.model.kip is None:
        raise ValueError(
            "kip.enabled=false: there is no KIP gradient to attribute. Probe the "
            "kip_on arm's config, not the kip_off one."
        )
    size = batch_size if batch_size else trainer.cfg.train.batch_size
    order = epoch_permutation(trainer.cfg.train.seed, epoch, len(trainer.dataset))
    trainer.model.train()
    records: list[dict[str, Any]] = []
    for index in range(num_batches):
        indices = order[index * size : (index + 1) * size]
        if not indices:
            LOGGER.warning("Epoch %d has only %d batches at size %d", epoch, index, size)
            break
        batch = collate_variable_length([trainer.dataset[i] for i in indices])
        records.append(probe_batch(trainer, batch))
        LOGGER.info("probed batch %d/%d", index + 1, num_batches)
    if not records:
        raise ValueError("No batch was probed; check --num-batches and the dataset size")
    return {
        "dataset": trainer.cfg.data.dataset,
        "stage": trainer.stage,
        "seed": trainer.cfg.train.seed,
        "epoch": epoch,
        "batch_size": size,
        "batches": len(records),
        "lambda_rec": trainer.cfg.loss.lambda_rec,
        "lambda_align": trainer.cfg.loss.lambda_align,
        "gamma_kin": trainer.cfg.loss.gamma_kin,
        "device": str(trainer.device),
        "records": records,
        "summary": summarize(records),
    }


def _mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else math.nan


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of each per-batch quantity — the numbers the verdict is read off."""
    trunk = [r["trunk"] for r in records if "trunk" in r]
    keys = sorted({k for r in trunk for k in r["kip_over_task"]})
    summary: dict[str, Any] = {
        "trunk_task_norm": _mean([r["task_norm"] for r in trunk]),
        "trunk_kip_over_task": {
            key: _mean([r["kip_over_task"][key] for r in trunk if key in r["kip_over_task"]])
            for key in keys
        },
        "trunk_cosine_with_task": {
            key: _mean(
                [r["cosine_with_task"][key] for r in trunk if key in r["cosine_with_task"]]
            )
            for key in keys
        },
    }
    summary["trunk_kip_total_over_task"] = sum(
        v for v in summary["trunk_kip_over_task"].values() if math.isfinite(v)
    )
    loss_keys = sorted({k for r in records for k in r["weighted"]})
    summary["weighted_loss"] = {
        key: _mean([r["weighted"][key] for r in records if key in r["weighted"]])
        for key in loss_keys
    }
    return summary


def render_markdown(payload: dict[str, Any]) -> str:
    """One page: the loss split, the trunk gradient split, and the cosines."""
    summary = payload["summary"]
    lines = [
        f"# Gradient attribution — {payload['dataset']} "
        f"(seed {payload['seed']}, epoch {payload['epoch']})",
        "",
        f"{payload['batches']} batches of {payload['batch_size']} on "
        f"`{payload['device']}`, fp32, no optimizer step. "
        f"`lambda_rec={payload['lambda_rec']}`, "
        f"`lambda_align={payload['lambda_align']}`, "
        f"`gamma_kin={payload['gamma_kin']}`.",
        "",
        "## 1. Weighted loss, as it enters `total`",
        "",
        "| term | weighted value | share of total |",
        "|---|---:|---:|",
    ]
    weighted = summary["weighted_loss"]
    total = sum(v for v in weighted.values() if math.isfinite(v))
    for key, value in sorted(weighted.items(), key=lambda kv: -kv[1]):
        share = value / total if total else math.nan
        lines.append(f"| `{key}` | {value:.4f} | {share:.1%} |")
    lines += [
        "",
        "## 2. Gradient arriving at the shared temporal encoder",
        "",
        f"Task gradient L2 norm: **{summary['trunk_task_norm']:.4f}** "
        "(`mil` + `mul_mil` + `dvs_*` + `cap_contrastive`, weighted).",
        "",
        "| KIP term | `|g| / |g_task|` | cosine with task gradient |",
        "|---|---:|---:|",
    ]
    for key in sorted(summary["trunk_kip_over_task"]):
        ratio = summary["trunk_kip_over_task"][key]
        cosine = summary["trunk_cosine_with_task"][key]
        lines.append(f"| `{key}` | {ratio:.3f} | {cosine:+.3f} |")
    lines += [
        f"| **all KIP terms** | **{summary['trunk_kip_total_over_task']:.3f}** | — |",
        "",
        "## 3. How to read it",
        "",
        "- A ratio **> 1** means KIP moves the trunk further per step than the "
        "anomaly objective does. That is a redirected trunk, whatever the AUC says.",
        "- A **negative cosine** means the two gradients disagree about where the "
        "encoder should go: direct conflict, and lowering `lambda_rec` trades one "
        "for the other rather than buying both.",
        "- A cosine **near 0** means KIP is orthogonal — it spends trunk capacity "
        "without fighting the task. Then the cost is representational, not a tug of "
        "war, and the fix is capacity or normalization, not a weight.",
        "",
        "Per-group norms for every batch are in `grad_probe.json`.",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="SECTION.KEY=VALUE", help="config override (repeatable)",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--clip-dir", type=Path, default=None)
    parser.add_argument("--flow-dir", type=Path, default=None)
    parser.add_argument("--knn-cache", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="model weights to probe (weights only); omitted = the run's initialization",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-batches", type=int, default=8,
        help="batches drawn from the real training order (default 8)",
    )
    parser.add_argument(
        "--epoch", type=int, default=0,
        help="which epoch's permutation to draw from (default 0)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="override train.batch_size for the probe only",
    )
    parser.add_argument(
        "--text-encoder", choices=TEXT_ENCODER_CHOICES, default=TEXT_ENCODER_CLIP,
        help="'clip' needs HF weights; 'stub' is the data-free smoke mode",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    set_global_seed(cfg.train.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(
        cfg=cfg,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        clip_dir=args.clip_dir,
        flow_dir=args.flow_dir,
        knn_cache_path=args.knn_cache,
        text_encoder=args.text_encoder,
    )
    if args.checkpoint is not None:
        # Weights only, through the stage-1 -> stage-2 loader. `load_checkpoint`
        # is the same-run *resume* path: it is strict, and a stage-1 checkpoint
        # legitimately has no `clip_text_model.*` because stage 1 builds the
        # model with `load_clip=False` (`build_trainer`). Probing that
        # checkpoint under the stage-2 objective is exactly what this tool is
        # for, so a strict load would refuse the arm the plan pre-registered.
        # It would also restore an optimizer/scheduler/RNG state that a
        # read-only probe must not touch, over parameter groups a stage-1
        # checkpoint does not even match.
        warm_start_model(trainer.model, args.checkpoint, flag="--checkpoint")
    payload = run_probe(
        trainer,
        num_batches=args.num_batches,
        epoch=args.epoch,
        batch_size=args.batch_size,
    )
    payload["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
    json_path = args.output_dir / REPORT_JSON
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    markdown_path = args.output_dir / REPORT_MARKDOWN
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    LOGGER.info("Wrote %s and %s", json_path, markdown_path)
    summary = payload["summary"]
    LOGGER.info(
        "trunk |g_KIP|/|g_task| = %.3f, cos(kip_rec, task) = %+.3f",
        summary["trunk_kip_total_over_task"],
        summary["trunk_cosine_with_task"].get("kip_rec", math.nan),
    )


__all__ = [
    "PARAM_GROUPS",
    "build_arg_parser",
    "collect_groups",
    "flat_gradient",
    "main",
    "probe_batch",
    "render_markdown",
    "run_probe",
    "summarize",
    "term_weights",
]


if __name__ == "__main__":
    main()
