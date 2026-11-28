"""KIM gradient probe: how hard does each KIM loss pull, next to the task (``KIM_RUN.md`` §66 NK1).

A loss *value* does not say how much a term steers the model: ``kim_nkc`` is an InfoNCE at
temperature 0.1, so its gradient carries a 1/τ factor the value hides (lesson C37: weigh terms by
gradient). For a trained checkpoint this probe draws batches from the run's own order, computes the
objective once in fp32 with no optimizer step, and takes one ``autograd.grad`` per weighted term
(task = every term of ``total`` that is not KIM's; ``kim_mil``; ``kim_nkc``) against each parameter
group. It prints, per group, the L2 norms; at the trunk, each KIM term over the task and its cosine
with it; in KIM, ``|g_nkc| / |g_kim_mil|``, their cosine and ``lambda_eq`` = the NKC weight at which
the two KIM terms would pull equally (a scale reference, never a weight to adopt; lesson 14).

Read-only: ``global_step`` is set to ``kim.warmup_steps`` so NKC is computed against the
checkpoint's prototypes, and the prototype buffers are restored after every batch. Illustration,
n = 1 checkpoint: decides nothing. CLI::

    python -m core.tools.kim_grad_probe --label L5dfi \\
        --config .../L5dfi/K5/s2024/config.yaml --checkpoint .../checkpoint_last.pt \\
        --data-dir .../Nexar_window --clip-dir .../Nexar_s8_ncc --knn-cache .../knn.npz \\
        --output-dir .../kim_nkc/REPORTS/grad_probe/L5dfi
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from core import constants
from core.config import load_config
from core.data.collate import collate_variable_length
from core.models.text_encoding import TEXT_ENCODER_CHOICES, TEXT_ENCODER_CLIP
from core.tools.grad_probe import flat_gradient, term_weights
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.train import Trainer, build_trainer, epoch_permutation, set_global_seed, warm_start_model

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/KIM_RUN.md §66 NK1"
LABEL = "illustration (n = 1 checkpoint); decides nothing"
TASK = "task"
MIL, NKC = constants.KIM_GRAD_PROBE_TERMS
TRUNK, BRANCH = constants.KIM_GRAD_PROBE_TRUNK, constants.KIM_GRAD_PROBE_BRANCH


def kim_term_weights(trainer: Trainer) -> dict[str, float]:
    """Weight of every key ``Trainer.compute_losses`` sums into ``total``, KIM's included."""
    kim = trainer.cfg.kim
    return {**term_weights(trainer.cfg), MIL: kim.mil_weight, NKC: kim.nkc_weight}


def collect_groups(model: torch.nn.Module) -> dict[str, list[Tensor]]:
    """Trainable parameters per ``KIM_GRAD_PROBE_GROUPS`` group (empty groups dropped)."""
    groups: dict[str, list[Tensor]] = {g: [] for g in constants.KIM_GRAD_PROBE_GROUPS}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        for group, prefixes in constants.KIM_GRAD_PROBE_GROUPS.items():
            if name.startswith(prefixes):
                groups[group].append(param)
                break
    return {g: p for g, p in groups.items() if p}


def check_probe_ready(trainer: Trainer) -> None:
    """Refuse a model NK1 cannot read: no KIM, no NKC weight, or prototypes never built."""
    module = trainer.model.kim
    if module is None:
        raise ValueError("kim.enabled=false: there is no KIM term to probe")
    if trainer.cfg.kim.nkc_weight <= 0:
        raise ValueError("kim.nkc_weight is 0: kim_nkc carries no gradient to probe")
    if not module.prototypes_ready:
        raise ValueError("the checkpoint's prototypes are not built: kim_nkc is not computed "
                         "(probe a checkpoint past kim.warmup_steps)")


def split_terms(trainer: Trainer, losses: dict[str, Tensor]) -> dict[str, Tensor]:
    """``{task, kim_mil, kim_nkc}`` weighted, after checking they rebuild ``total``."""
    weights = kim_term_weights(trainer)
    weighted = {k: weights[k] * v for k, v in losses.items()
                if k in weights and weights[k] != 0 and v.requires_grad}
    rebuilt = torch.stack(list(weighted.values())).sum()
    drift = float((rebuilt - losses["total"]).abs())
    if drift > constants.KIM_GRAD_PROBE_TOTAL_TOL:
        raise ValueError(f"reassembled total {float(rebuilt):.6f} != compute_losses total "
                         f"{float(losses['total']):.6f} (drift {drift:.2e}): the weight table "
                         "is out of step with Trainer.compute_losses")
    missing = [k for k in (MIL, NKC) if k not in weighted]
    if missing:
        raise ValueError(f"no {missing} in this batch's objective")
    task = [v for k, v in weighted.items() if k not in constants.KIM_GRAD_PROBE_TERMS]
    if not task:
        raise ValueError("no task term in the objective")
    return {TASK: torch.stack(task).sum(), MIL: weighted[MIL], NKC: weighted[NKC]}


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def _cos(a: Tensor, b: Tensor) -> float:
    if float(a.norm()) == 0 or float(b.norm()) == 0:
        return math.nan
    return float(F.cosine_similarity(a, b, dim=0))


def probe_batch(trainer: Trainer, batch: dict[str, Any]) -> dict[str, Any]:
    """Norms per group and term, trunk and branch ratios / cosines, for one collated batch."""
    module = trainer.model.kim
    assert module is not None  # nosec B101 - check_probe_ready
    saved = (module.prototypes.clone(), module.proto_init.clone())
    try:
        losses = trainer.compute_losses(batch)
        terms = split_terms(trainer, losses)
        vectors = {g: {t: flat_gradient(v, params) for t, v in terms.items()}
                   for g, params in collect_groups(trainer.model).items()}
    finally:  # NK1: read-only, the training-mode prototype update is undone
        with torch.no_grad():
            module.prototypes.copy_(saved[0])
            module.proto_init.copy_(saved[1])
    norms = {g: {t: float(v.norm()) for t, v in per.items()} for g, per in vectors.items()}
    record: dict[str, Any] = {"weighted": {t: float(v.detach()) for t, v in terms.items()},
                              "grad_norm": norms}
    if TRUNK in vectors:
        trunk = vectors[TRUNK]
        record["trunk"] = {f"{t}_over_task": _ratio(norms[TRUNK][t], norms[TRUNK][TASK])
                           for t in (MIL, NKC)}
        record["trunk"] |= {f"cos_{t}_task": _cos(trunk[t], trunk[TASK]) for t in (MIL, NKC)}
    if BRANCH in vectors:
        branch, nb = vectors[BRANCH], norms[BRANCH]
        record["branch"] = {
            "nkc_over_mil": _ratio(nb[NKC], nb[MIL]),
            "cos_nkc_mil": _cos(branch[NKC], branch[MIL]),
            "task_over_mil": _ratio(nb[TASK], nb[MIL]),
            "lambda_eq": trainer.cfg.kim.nkc_weight * _ratio(nb[MIL], nb[NKC]),
        }
    return record


def _mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else math.nan


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean over batches of every per-batch quantity."""
    out: dict[str, Any] = {"weighted": {}, "grad_norm": {}}
    for key in ("weighted",):
        for t in records[0][key]:
            out[key][t] = _mean([r[key][t] for r in records])
    for g in records[0]["grad_norm"]:
        out["grad_norm"][g] = {t: _mean([r["grad_norm"][g][t] for r in records])
                               for t in records[0]["grad_norm"][g]}
    for part in ("trunk", "branch"):
        if part in records[0]:
            out[part] = {k: _mean([r[part][k] for r in records]) for k in records[0][part]}
    return out


def run_probe(trainer: Trainer, num_batches: int, epoch: int) -> dict[str, Any]:
    """Probe the first ``num_batches`` batches of ``epoch``'s training order (NK1)."""
    trainer.global_step = max(trainer.global_step, trainer.cfg.kim.warmup_steps)
    check_probe_ready(trainer)
    size = trainer.cfg.train.batch_size
    order = epoch_permutation(trainer.cfg.train.seed, epoch, len(trainer.dataset))
    trainer.model.train()
    records = []
    for index in range(num_batches):
        indices = order[index * size : (index + 1) * size]
        if not indices:
            LOGGER.warning("epoch %d has only %d batches of %d", epoch, index, size)
            break
        batch = collate_variable_length([trainer.dataset[i] for i in indices])
        records.append(probe_batch(trainer, batch))
        LOGGER.info("probed batch %d/%d", index + 1, num_batches)
    if not records:
        raise ValueError("no batch was probed; check --num-batches and the dataset size")
    return {"rule": RULE, "label": LABEL, "dataset": trainer.cfg.data.dataset,
            "seed": trainer.cfg.train.seed, "epoch": epoch, "batch_size": size,
            "batches": len(records), "nkc_weight": trainer.cfg.kim.nkc_weight,
            "mil_weight": trainer.cfg.kim.mil_weight, "device": str(trainer.device),
            "records": records, "summary": summarize(records)}


def render(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [f"# §66 NK1 KIM gradient probe — {payload.get('row', '?')} — **{payload['label']}**",
             "",
             f"{payload['batches']} batches of {payload['batch_size']} from epoch "
             f"{payload['epoch']}'s order ({payload['dataset']}, s{payload['seed']}), fp32, "
             f"train mode, no optimizer step, prototypes restored. `kim.mil_weight` "
             f"{payload['mil_weight']}, `kim.nkc_weight` {payload['nkc_weight']}.", "",
             "## Weighted loss values (as in `total`)", "", "| term | value |", "|---|---:|"]
    lines += [f"| `{t}` | {v:.4f} |" for t, v in s["weighted"].items()]
    lines += ["", "## Gradient L2 norm per group (weighted terms)", "",
              f"| group | {TASK} | {MIL} | {NKC} |", "|---|---:|---:|---:|"]
    for g, per in s["grad_norm"].items():
        lines.append(f"| `{g}` | {per[TASK]:.4g} | {per[MIL]:.4g} | {per[NKC]:.4g} |")
    if "trunk" in s:
        t = s["trunk"]
        head = "| term | \\|g\\| / \\|g_task\\| | cos with task |"
        lines += ["", f"## Trunk (`{TRUNK}`)", "", head,
                  "|---|---:|---:|",
                  f"| `{MIL}` | {t[f'{MIL}_over_task']:.3f} | {t[f'cos_{MIL}_task']:+.3f} |",
                  f"| `{NKC}` | {t[f'{NKC}_over_task']:.3f} | {t[f'cos_{NKC}_task']:+.3f} |"]
    if "branch" in s:
        b = s["branch"]
        lines += ["", f"## KIM (`{BRANCH}`)", "",
                  f"- \\|g_nkc\\| / \\|g_kim_mil\\| = **{b['nkc_over_mil']:.3f}**, "
                  f"cos = {b['cos_nkc_mil']:+.3f}",
                  f"- \\|g_task\\| / \\|g_kim_mil\\| = {b['task_over_mil']:.3f}",
                  f"- lambda_eq = **{b['lambda_eq']:.3g}** (the NKC weight at which the two KIM "
                  "terms would pull equally here; a scale reference, never a weight to adopt)"]
    lines += ["", "Never: \"the weight is too small / right\" from this probe alone (a gradient "
              "size is not a contribution, §66)."]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--label", required=True, help="row name printed in the report")
    parser.add_argument("--config", type=Path, required=True, help="the run's config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--knn-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-batches", type=int, default=constants.KIM_GRAD_PROBE_BATCHES)
    parser.add_argument("--epoch", type=int, default=constants.KIM_GRAD_PROBE_EPOCH)
    parser.add_argument("--device", default=None, help="override train.device (default: run's)")
    parser.add_argument("--text-encoder", choices=TEXT_ENCODER_CHOICES, default=TEXT_ENCODER_CLIP)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    overrides = ["train.amp=false"] + ([f"train.device={args.device}"] if args.device else [])
    cfg = load_config(args.config, overrides)
    set_global_seed(cfg.train.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(cfg=cfg, output_dir=args.output_dir, data_dir=args.data_dir,
                            clip_dir=args.clip_dir, knn_cache_path=args.knn_cache,
                            text_encoder=args.text_encoder)
    warm_start_model(trainer.model, args.checkpoint, flag="--checkpoint")
    trainer.model.to(trainer.device)
    payload = run_probe(trainer, args.num_batches, args.epoch)
    payload |= {"row": args.label, "checkpoint": str(args.checkpoint)}
    stem = constants.KIM_GRAD_PROBE_STEM
    write_json_atomic(args.output_dir / f"{stem}.json", payload)
    write_text_atomic(args.output_dir / f"{stem}.md", render(payload))
    LOGGER.info("wrote %s.{json,md} in %s", stem, args.output_dir)
    return payload


__all__ = ["check_probe_ready", "collect_groups", "kim_term_weights", "probe_batch", "render",
           "run_probe", "split_terms", "summarize"]


if __name__ == "__main__":
    main()
