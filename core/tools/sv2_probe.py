"""SA²-v2 learning probe: why did the scorer's priors and text term not move? (read-only).

Reading rule: ``core/docs/final/SA2V2_RUN.md`` §9, written before any probe number. Printed only:
nothing here changes a verdict of §7; it decides what a next arm may change.

``adam`` (CPU): the run's own AdamW state in ``checkpoint_last.pt``. One row per scalar coordinate
(``beta[0..3]``, ``gamma``, ``log_temperature``, ``bias``) and one per module (``w``, ``w_cell``,
``w_prior``, ``pos``, ``layer``, ``branch_head``, ``W_u``): displacement from init over the LR
budget ``sum_s lr_s`` (the most a coordinate can move under Adam), gradient RMS ``sqrt(v_hat)``
against ``eps``, and ``SNR = mean|m_hat| / mean sqrt(v_hat)`` beside the zero-mean-noise SNR.
Same arithmetic as ``core.tools.stp_diag prompt`` (D-1), so the two reads compare.

``live`` (GPU): the real loss on the first batches of epoch 0's order, at init (no
``--checkpoint``) or at a checkpoint. Per batch, the gradient of each weighted term (``host`` =
``total - patch - branch``; ``patch`` = ``lambda_p L_patch``; ``branch`` = ``lambda_b L_branch``)
on every scalar coordinate and on ``w``; then its mean, std and coherence ``|mean| / rms`` over
batches, and, per term pair, the cosine on ``w``. It also splits the cell logits
``l = beta.priors + gamma z(cos) + w^T H``: the per-frame std across cells of each part says which
part ranks the cells.
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
from core.config import Config, load_config
from core.data.collate import collate_variable_length
from core.models.kat_vad import KATVAD
from core.models.sa2v2 import SA2V2
from core.models.text_encoding import TEXT_ENCODER_CHOICES, TEXT_ENCODER_CLIP
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.stp_diag import lr_budget, optimizer_state, param_stats
from core.train import (
    Trainer,
    build_trainer,
    epoch_permutation,
    optimizer_param_groups,
    set_global_seed,
    warm_start_model,
)

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/final/SA2V2_RUN.md §9"
PREFIX = "sa2v2."
ADAM_JSON, ADAM_MD = "sv2_probe_adam.json", "sv2_probe_adam.md"
LIVE_JSON, LIVE_MD = "sv2_probe_live.json", "sv2_probe_live.md"
TERMS = ("host", "patch", "branch", "total")
HOST_MOTION = ("w_u", "motion_residual.")


# --------------------------------------------------------------------------- adam


def scalar_rows(names: list[str]) -> list[tuple[str, str, int | None]]:
    """``(label, key, index)`` of every scalar coordinate, then every module, the probe reads."""
    rows: list[tuple[str, str, int | None]] = []
    for name in constants.SV2_PROBE_SCALARS:
        key = PREFIX + name
        if key not in names:
            raise ValueError(f"{key} is not a trainable parameter of this model")
        size = len(constants.SV2_BETA_INIT) if name == "beta" else 1
        rows += [(f"{name}[{i}]" if size > 1 else name, key, i if size > 1 else None)
                 for i in range(size)]
    return rows


def module_keys(names: list[str], prefix: str) -> list[str]:
    return [n for n in names if n.startswith(prefix)]


def coordinate_stats(
    key: str, index: int | None, final: dict[str, Tensor], init: dict[str, Tensor],
    opt: dict[str, dict[str, Any]], budget: float,
) -> dict[str, Any]:
    """``param_stats`` of one coordinate (``index``) or one whole tensor (``None``)."""
    state = dict(opt[key])

    def part(t: Tensor) -> Tensor:
        flat = t.flatten()
        return flat if index is None else flat[index : index + 1]

    for name in ("exp_avg", "exp_avg_sq"):
        if name in state:
            state[name] = part(state[name])
    return param_stats(part(final[key]), part(init[key]), state, budget)


def merged_module_stats(
    keys: list[str], final: dict[str, Tensor], init: dict[str, Tensor],
    opt: dict[str, dict[str, Any]], budget: float,
) -> dict[str, Any] | None:
    """One row for a module: its tensors concatenated (one AdamW group)."""
    if not keys:
        return None
    states = [opt[k] for k in keys]
    merged: dict[str, Any] = {"group": states[0]["group"]}
    if all("exp_avg_sq" in s for s in states):
        merged |= {
            "step": states[0]["step"],
            "exp_avg": torch.cat([s["exp_avg"].flatten() for s in states]),
            "exp_avg_sq": torch.cat([s["exp_avg_sq"].flatten() for s in states]),
        }
    cat_f = torch.cat([final[k].flatten() for k in keys])
    cat_i = torch.cat([init[k].flatten() for k in keys])
    return {"keys": keys, **param_stats(cat_f, cat_i, merged, budget)}


def optimizer_names(model: KATVAD, cfg: Config) -> list[str]:
    """Trainable names in the order AdamW holds them (its groups, as ``core.train`` builds them).

    With one group this is ``named_parameters`` order; with a ``W_u`` or gate group the ids are
    concatenated group by group, and a plain ``named_parameters`` zip would mislabel every row.
    """
    groups = optimizer_param_groups(
        model, cfg.train.learning_rate, cfg.train.motion_lr_mult,
        cfg.stp.gate_lr_mult if cfg.stp.sv2_gate else 1.0,
    )
    name_of = {id(p): n for n, p in model.named_parameters()}
    if groups and isinstance(groups[0], dict):
        return [name_of[id(p)] for g in groups for p in g["params"]]
    return [name_of[id(p)] for p in groups]


def rebuild_init(cfg: Config, text_encoder: str) -> KATVAD:
    """The model as ``core.train.build_trainer`` holds it before step 0 (seed, then build).

    Scalars and ``w`` start at constants, so their displacement is exact; the matrices' init is
    the rebuilt draw (``stp_diag`` D-1 made the same assumption and checked it on the prompt).
    """
    set_global_seed(cfg.train.seed)
    return KATVAD.from_config(cfg, load_clip=text_encoder == TEXT_ENCODER_CLIP)


def reading(row: dict[str, Any]) -> str:
    """§9 reading of one row (printed, never gating); thresholds of ``stp_diag`` D-1."""
    if not row.get("adam_state"):
        return "no Adam state"
    if row["share_below_10eps"] >= constants.STP_DIAG_DEAD_SHARE:
        return "dead: gradient below Adam's eps"
    if row["snr"] is not None and row["snr"] < constants.STP_DIAG_NOISE_FACTOR * row["noise_snr"]:
        return "noise: gradient alive, sign-incoherent"
    if row["budget_used"] is not None and row["budget_used"] < constants.STP_DIAG_MOVED:
        return "lr: coherent gradient, budget unused"
    return "moved"


def run_adam(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    if not cfg.stp.sa2v2:
        raise ValueError(f"{run_dir} is not an SA²-v2 run (stp.sa2v2=false)")
    payload = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu",
                         weights_only=False)  # nosec B614 - own checkpoint
    init_model = rebuild_init(cfg, args.text_encoder)
    names = optimizer_names(init_model, cfg)
    init_sd = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
    del init_model
    final_sd, opt = payload["model"], optimizer_state(payload, names)
    steps = int(payload["global_step"])
    budget = lr_budget(cfg.train.learning_rate, cfg.train.warmup_steps, steps, steps)
    rows: dict[str, dict[str, Any]] = {}
    for label, key, index in scalar_rows(names):
        rows[label] = {"keys": [key], "index": index,
                       **coordinate_stats(key, index, final_sd, init_sd, opt, budget)}
    gate = PREFIX + "gate"
    if gate in names:  # SV3: its own group, so its own budget (lr x gate_lr_mult)
        mult = cfg.stp.gate_lr_mult
        rows["gate"] = {"keys": [gate], "index": None, "lr_mult": mult,
                        **coordinate_stats(gate, None, final_sd, init_sd, opt, budget * mult)}
    for name in constants.SV2_PROBE_MODULES:
        row = merged_module_stats(module_keys(names, f"{PREFIX}{name}."), final_sd, init_sd, opt,
                                  budget)
        if row is not None:
            rows[name] = row
    host = merged_module_stats(module_keys(names, HOST_MOTION[1]), final_sd, init_sd, opt, budget)
    if host is not None:
        rows[HOST_MOTION[0]] = host
    for row in rows.values():
        row["reading"] = reading(row)
    out = {"rule": RULE, "run_dir": str(run_dir), "global_step": steps,
           "learning_rate": cfg.train.learning_rate, "lr_budget": budget, "rows": rows}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / ADAM_JSON, out)
    write_text_atomic(args.out_dir / ADAM_MD, render_adam(out))
    return out


def _num(x: Any, spec: str = ".4g") -> str:
    return "—" if x is None else format(x, spec)


def render_adam(out: dict[str, Any]) -> str:
    lines = [
        f"# SA²-v2 probe, AdamW state (rule `{out['rule']}`)", "",
        f"`{out['run_dir']}` · step {out['global_step']} · lr {out['learning_rate']:.3g} · "
        f"Σ lr = **{out['lr_budget']:.4g}** (the most one coordinate can move)", "",
        "| param | numel | θ₀ (or ‖θ₀‖) | θ (or ‖θ‖) | RMS Δ / Σlr | grad RMS | "
        "share √v̂ < 10 eps | SNR (noise ≈) | reading |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, r in out["rows"].items():
        snr = f"{_num(r.get('snr'), '.3f')} ({_num(r.get('noise_snr'), '.3f')})"
        lines.append(
            f"| {label} | {r['numel']} | {_num(r['norm_init'])} | {_num(r['norm_final'])} | "
            f"{_num(r['budget_used'], '.3f')} | {_num(r.get('grad_rms'))} | "
            f"{_num(r.get('share_below_10eps'), '.3f')} | {snr} | {r['reading']} |")
    lines += ["", "Scalars print |θ|: a signed value is in the JSON as the cell logits use it.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- live


def probe_params(module: SA2V2) -> dict[str, Tensor]:
    return {**{name: getattr(module, name) for name in constants.SV2_PROBE_SCALARS},
            "w": module.w.weight}


def split_terms(trainer: Trainer, losses: dict[str, Tensor]) -> dict[str, Tensor]:
    """``total`` = ``host + patch + branch`` exactly as ``Trainer._sv2_losses`` adds them."""
    stp = trainer.cfg.stp
    patch = stp.patch_weight * losses["sv2_patch"]
    branch = stp.branch_weight * losses["sv2_branch"]
    return {"host": losses["total"] - patch - branch, "patch": patch, "branch": branch,
            "total": losses["total"]}


class LogitSplit:
    """Captures ``beta.priors``, ``w^T H`` and the total ``l`` of one forward (read-only)."""

    def __init__(self, module: SA2V2) -> None:
        self.module = module
        self.parts: dict[str, Tensor] = {}
        self._orig = module._logits
        self._hook = module.w.register_forward_hook(self._on_w)

    def _on_w(self, _m: Any, _i: Any, out: Tensor) -> None:
        self.parts["residual"] = out.detach().squeeze(-1).float()

    def __enter__(self) -> LogitSplit:
        def wrapped(*a: Any, **k: Any) -> tuple[Tensor, Tensor]:
            logits, priors = self._orig(*a, **k)
            self.parts["total"] = logits.detach()
            self.parts["prior"] = (priors @ self.module.beta).detach()
            return logits, priors

        self.module._logits = wrapped  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        self.module._logits = self._orig  # type: ignore[method-assign]
        self._hook.remove()

    def spreads(self) -> dict[str, float]:
        p = self.parts
        text = p["total"] - p["prior"] - p["residual"]
        named = {"prior": p["prior"], "text": text, "residual": p["residual"], "l": p["total"]}
        return {k: float(v.std(dim=-1, unbiased=False).mean()) for k, v in named.items()}


def probe_batch(trainer: Trainer, batch: dict[str, Any]) -> dict[str, Any]:
    module = trainer.model.sa2v2
    assert module is not None  # nosec B101 - run_live checks
    params = probe_params(module)
    with LogitSplit(module) as split:
        losses = trainer.compute_losses(batch)
    terms = split_terms(trainer, losses)
    grads: dict[str, dict[str, list[float]]] = {}
    w_vec: dict[str, Tensor] = {}
    for term, value in terms.items():
        got = torch.autograd.grad(value, list(params.values()), retain_graph=True,
                                  allow_unused=True, materialize_grads=True)
        for (name, _), g in zip(params.items(), got, strict=True):
            if name == "w":
                w_vec[term] = g.detach().flatten().float()
            else:
                grads.setdefault(name, {})[term] = g.detach().flatten().float().tolist()
    cos = {f"{a}|{b}": float(F.cosine_similarity(w_vec[a], w_vec[b], dim=0))
           for i, a in enumerate(TERMS[:3]) for b in TERMS[i + 1 : 3]}
    return {
        "losses": {k: float(v.detach()) for k, v in terms.items()},
        "scalar_grads": grads,
        "w_grad_norm": {t: float(v.norm()) for t, v in w_vec.items()},
        "w_cos": cos,
        "logit_spread": split.spreads(),
    }


def _stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    rms = math.sqrt(sum(v * v for v in values) / n)
    std = math.sqrt(max(sum((v - mean) ** 2 for v in values) / n, 0.0))
    return {"mean": mean, "std": std, "coherence": abs(mean) / rms if rms > 0 else 0.0}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    scalars: dict[str, dict[str, dict[str, float]]] = {}
    for name in records[0]["scalar_grads"]:
        size = len(records[0]["scalar_grads"][name]["total"])
        for i in range(size):
            label = f"{name}[{i}]" if size > 1 else name
            scalars[label] = {t: _stats([r["scalar_grads"][name][t][i] for r in records])
                              for t in TERMS}

    def mean_of(field: str) -> dict[str, float]:
        keys = records[0][field]
        return {k: sum(r[field][k] for r in records) / len(records) for k in keys}

    return {"scalars": scalars, "w_grad_norm": mean_of("w_grad_norm"),
            "w_cos": mean_of("w_cos"), "logit_spread": mean_of("logit_spread"),
            "losses": mean_of("losses")}


def build_probe_trainer(args: argparse.Namespace, cfg: Config) -> Trainer:
    set_global_seed(cfg.train.seed)
    trainer = build_trainer(cfg=cfg, output_dir=args.out_dir, data_dir=args.data_dir,
                            clip_dir=args.clip_dir, knn_cache_path=args.knn_cache,
                            patch_dir=args.patch_dir, text_encoder=args.text_encoder)
    if args.checkpoint is not None:
        warm_start_model(trainer.model, args.checkpoint, flag="--checkpoint")
    return trainer


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if not cfg.stp.sa2v2:
        raise ValueError(f"{args.config} is not an SA²-v2 config (stp.sa2v2=false)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_probe_trainer(args, cfg)
    size = cfg.train.batch_size
    order = epoch_permutation(cfg.train.seed, 0, len(trainer.dataset))
    trainer.model.train()
    records = []
    for b in range(args.num_batches):
        indices = order[b * size : (b + 1) * size]
        if not indices:
            break
        records.append(probe_batch(trainer, collate_variable_length(
            [trainer.dataset[i] for i in indices])))
        LOGGER.info("probed batch %d/%d", b + 1, args.num_batches)
    if not records:
        raise ValueError("no batch was probed")
    out = {"rule": RULE, "name": args.name,
           "checkpoint": str(args.checkpoint) if args.checkpoint else None,
           "batches": len(records), "batch_size": size, "device": str(trainer.device),
           "records": records, "summary": summarize(records)}
    write_json_atomic(args.out_dir / LIVE_JSON, out)
    write_text_atomic(args.out_dir / LIVE_MD, render_live(out))
    return out


def render_live(out: dict[str, Any]) -> str:
    s = out["summary"]
    where = out["checkpoint"] or "init (rebuilt as core.train builds it)"
    lines = [
        f"# SA²-v2 probe, live gradients — {out['name']} (rule `{RULE}`)", "",
        f"At {where}; {out['batches']} batches of {out['batch_size']}, epoch-0 order, "
        f"train mode, fp32, no optimizer step, on `{out['device']}`.", "",
        "## Gradient on each scalar, per term (mean ± std over batches; coherence |mean|/rms)", "",
        "| param | " + " | ".join(TERMS) + " |", "|---|" + "---|" * len(TERMS),
    ]
    for label, per in s["scalars"].items():
        cells = [f"{per[t]['mean']:+.2e} ± {per[t]['std']:.1e} ({per[t]['coherence']:.2f})"
                 for t in TERMS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## `w` (the residual scorer)", "",
              "| term | ‖∇w‖ |", "|---|---|",
              *[f"| {t} | {v:.3e} |" for t, v in s["w_grad_norm"].items()], "",
              "| pair | cos on ∇w |", "|---|---|",
              *[f"| {p} | {v:+.3f} |" for p, v in s["w_cos"].items()], "",
              "## Which part of `l` ranks the cells (mean per-frame std across 49 cells)", "",
              "| part | spread |", "|---|---|",
              *[f"| {k} | {v:.4f} |" for k, v in s["logit_spread"].items()], "",
              "Weighted loss (mean): " + ", ".join(f"{k} {v:.4f}" for k, v in s["losses"].items()),
              ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    adam = sub.add_parser("adam", help="AdamW state of the run's checkpoint (CPU)")
    adam.add_argument("--run-dir", type=Path, required=True,
                      help="config.yaml + checkpoint_last.pt of an SA²-v2 run")
    live = sub.add_parser("live", help="per-term gradients on real batches (GPU)")
    live.add_argument("--config", type=Path, required=True, help="the run's config.yaml")
    live.add_argument("--name", required=True, help="label in the report, e.g. init / final")
    live.add_argument("--checkpoint", type=Path, default=None, help="weights; omitted = init")
    live.add_argument("--data-dir", type=Path, required=True)
    live.add_argument("--clip-dir", type=Path, required=True, help="the A2 bake the run used")
    live.add_argument("--patch-dir", type=Path, required=True, help="the run's cell cache")
    live.add_argument("--knn-cache", type=Path, default=None)
    live.add_argument("--num-batches", type=int, default=constants.SV2_PROBE_BATCHES)
    for p in (adam, live):
        p.add_argument("--out-dir", type=Path, required=True)
        p.add_argument("--text-encoder", choices=TEXT_ENCODER_CHOICES, default=TEXT_ENCODER_CLIP,
                       help="as the run trained ('stub' = data-free tests only)")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    args = build_arg_parser().parse_args(argv)
    if args.command == "adam":
        LOGGER.info("\n%s", render_adam(run_adam(args)))
    else:
        LOGGER.info("\n%s", render_live(run_live(args)))


if __name__ == "__main__":
    main()
