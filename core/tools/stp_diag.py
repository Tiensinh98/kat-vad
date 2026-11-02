"""LaGoVAD-STP first-run diagnostics -- why R4 / R6 read "no difference" (inference only).

Reading rules: ``core/docs/final/STP_BUILD.md`` §5, written before any diagnostic number.
Printed only: neither command changes a verdict of §3; they decide what the next arm may be.

``prompt`` (Colab, CPU is enough): did the text prompt train? For each checkpoint the init is
rebuilt the way ``core.train`` builds it (``set_global_seed`` then ``KATVAD.from_config``), and
AdamW's own state is read for the prompt, ``W_q`` (SA²) and ``W_u`` (motion residual):

* displacement ``||theta - theta_0||`` and its per-coordinate RMS over the LR budget
  ``sum_s lr_s`` (1 = every step moved every coordinate the full Adam step the same way);
* gradient RMS ``sqrt(v_hat)`` (vs AdamW's ``eps``) and ``SNR = mean|m_hat| / mean sqrt(v_hat)``;
  zero-mean noise gives ``SNR ~ sqrt((1 - b1) / (1 + b1)) * sqrt(2 / pi)`` (printed beside).

``pointing`` (local or Colab, CPU): do the motion-selected patches point at the anomaly? On the
DoTA-CAP-dev patch cache (``idx/``, ``mo/``) against the DoTA boxes, per clip: Mo's own pointing
(v5 Stage 0's reads, ``core.tools.v5_stage0.clip_read``), the share of the top-K tiles that touch
a box minus chance, the same before the anomaly with the first anomaly mask (box growth / centre
control, Amendment 1's form), and where the top-K sit (border ring, bottom row, centre 3 x 3).
With ``--windows-dir`` the ego-compensated Mo (``stp.motion=ego``, arm R7) is read the same way.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core import constants
from core.config import Config, load_config
from core.data.dota import parse_metadata, read_split_ids
from core.data.stp_patches import motion_magnitude, select_top_k
from core.data.v2_splits import load_split
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.rate_matched_eval import native_labels
from core.tools.stp_patches import patch_cells
from core.tools.v5_stage0 import ci, clip_read, load_boxes, summarize_rep, tile_weights
from core.train import cosine_warmup_lambda, set_global_seed

LOGGER = logging.getLogger(__name__)

PROMPT_JSON = "stp_diag_prompt.json"
PROMPT_MD = "stp_diag_prompt.md"
POINTING_JSON = "stp_diag_pointing.json"
POINTING_MD = "stp_diag_pointing.md"
RULE = "core/docs/final/STP_BUILD.md §5"
PROMPT_KEY = "clip_text_model.prompt_embedding.weight"
WATCHED = {  # report name -> state-dict key suffix (first match)
    "prompt": PROMPT_KEY,
    "sa2": "sa2.",
    "w_u": "motion_residual.",
}
GRID = constants.STP_GRID
BORDER = np.array([r in (0, GRID - 1) or c in (0, GRID - 1)
                   for r in range(GRID) for c in range(GRID)])
BOTTOM = np.array([r == GRID - 1 for r in range(GRID) for _ in range(GRID)])
CENTRE = np.array([abs(r - GRID // 2) <= 1 and abs(c - GRID // 2) <= 1
                   for r in range(GRID) for c in range(GRID)])
REGIONS = {"border": BORDER, "bottom_row": BOTTOM, "centre_3x3": CENTRE}


# --------------------------------------------------------------------------- prompt (D-1)


def noise_snr(beta1: float) -> float:
    """``mean|m| / mean sqrt(v)`` of Adam's EMAs when the gradient is zero-mean Gaussian noise."""
    return math.sqrt((1.0 - beta1) / (1.0 + beta1)) * math.sqrt(2.0 / math.pi)


def lr_budget(base_lr: float, warmup_steps: int, total_steps: int, steps_taken: int) -> float:
    """``sum_s lr_s`` over the steps taken: the most any coordinate can move under Adam."""
    schedule = cosine_warmup_lambda(warmup_steps, total_steps)
    return float(sum(base_lr * schedule(s) for s in range(steps_taken)))


def trainable_names(model: torch.nn.Module) -> list[str]:
    """Parameter names in the order ``core.train.Trainer`` hands them to AdamW."""
    return [n for n, p in model.named_parameters() if p.requires_grad]


def optimizer_state(payload: dict[str, Any], names: list[str]) -> dict[str, dict[str, Any]]:
    """AdamW state keyed by parameter name; refuses a checkpoint whose groups do not match."""
    groups = payload["optimizer"]["param_groups"]
    ids = [i for g in groups for i in g["params"]]
    if len(ids) != len(names):
        raise ValueError(f"optimizer has {len(ids)} params, the rebuilt model {len(names)} "
                         "trainable ones: the architecture does not match the checkpoint")
    group_of = {i: g for g in groups for i in g["params"]}
    state = payload["optimizer"]["state"]
    return {n: {**state.get(i, {}), "group": group_of[i]} for n, i in zip(names, ids, strict=True)}


def param_stats(
    final: torch.Tensor, init: torch.Tensor, state: dict[str, Any], budget: float
) -> dict[str, Any]:
    """Displacement, gradient RMS and SNR of one parameter tensor."""
    group = state["group"]
    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    f, i = final.detach().double().flatten(), init.detach().double().flatten()
    delta = f - i
    out: dict[str, Any] = {
        "numel": int(f.numel()),
        "norm_final": float(f.norm()),
        "norm_init": float(i.norm()),
        "displacement": float(delta.norm()),
        "rms_step_per_coord": float(delta.pow(2).mean().sqrt()),
        "budget_used": float(delta.pow(2).mean().sqrt()) / budget if budget > 0 else None,
        "cos_final_init": (float(torch.nn.functional.cosine_similarity(f, i, dim=0))
                           if float(i.norm()) > 0 else None),
        "eps": eps,
    }
    if "exp_avg_sq" not in state:
        return {**out, "adam_state": False}
    step = float(state["step"])
    m_hat = state["exp_avg"].double().flatten() / (1.0 - beta1**step)
    v_hat = state["exp_avg_sq"].double().flatten() / (1.0 - beta2**step)
    root = v_hat.sqrt()
    return {
        **out,
        "adam_state": True,
        "adam_step": step,
        "grad_rms": float(v_hat.mean().sqrt()),
        "grad_rms_median": float(root.median()),
        "share_below_10eps": float((root < 10 * eps).double().mean()),
        "snr": float(m_hat.abs().mean() / root.mean()) if float(root.mean()) > 0 else None,
        "noise_snr": noise_snr(beta1),
    }


def pick(state_dict: dict[str, torch.Tensor], names: list[str], suffix: str) -> list[str]:
    """Trainable keys a watched name covers (exact key, or every key under a module prefix)."""
    if suffix in names:
        return [suffix]
    return [n for n in names if n.startswith(suffix) and n in state_dict]


def merged_stats(
    keys: list[str],
    final: dict[str, torch.Tensor],
    init: dict[str, torch.Tensor],
    opt: dict[str, dict[str, Any]],
    budget: float,
) -> dict[str, Any] | None:
    """One row for a module: its tensors concatenated (same group, so one eps / beta)."""
    if not keys:
        return None
    states = [opt[k] for k in keys]
    has_state = all("exp_avg_sq" in s for s in states)
    merged: dict[str, Any] = {"group": states[0]["group"]}
    if has_state:
        merged |= {
            "step": states[0]["step"],
            "exp_avg": torch.cat([s["exp_avg"].flatten() for s in states]),
            "exp_avg_sq": torch.cat([s["exp_avg_sq"].flatten() for s in states]),
        }
    cat_f = torch.cat([final[k].flatten() for k in keys])
    cat_i = torch.cat([init[k].flatten() for k in keys])
    return {"keys": keys, **param_stats(cat_f, cat_i, merged, budget)}


def rebuild_init(cfg: Config) -> torch.nn.Module:
    """The model exactly as ``core.train.main`` holds it before step 0 (seed, then build)."""
    from core.models.kat_vad import KATVAD

    set_global_seed(cfg.train.seed)
    return KATVAD.from_config(cfg, load_clip=True)


def logged_init_norm(metrics: Path) -> float | None:
    """The first ``prompt_norm`` of a run: logged on batch 0, before any optimizer step."""
    if not metrics.is_file():
        return None
    with metrics.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "prompt_norm" in record:
                return float(record["prompt_norm"])
    return None


def init_reproduced(rebuilt_norm: float, logged: float | None) -> bool | None:
    """Fingerprint check of the rebuilt init: its norm must equal the run's step-0 log."""
    if logged is None or logged <= 0:
        return None
    return abs(rebuilt_norm - logged) / logged < constants.STP_DIAG_INIT_NORM_RTOL


def classify(row: dict[str, Any] | None) -> str:
    """§5 D-1 reading of the prompt row (printed, never gating)."""
    if row is None or not row.get("adam_state"):
        return "no prompt / no Adam state"
    if row.get("init_reproduced") is False:
        return "INIT NOT REPRODUCED (displacement unreadable; read the Adam columns only)"
    if row["share_below_10eps"] >= constants.STP_DIAG_DEAD_SHARE:
        return "D1-dead: gradient below Adam's eps -- the gradient path is the defect"
    if row["snr"] is not None and row["snr"] < constants.STP_DIAG_NOISE_FACTOR * row["noise_snr"]:
        return "D1-noise: gradient alive but sign-incoherent -- an LR multiplier would random-walk"
    if row["budget_used"] is not None and row["budget_used"] < constants.STP_DIAG_MOVED:
        return "D1-lr: coherent gradient, LR budget too small -- LR multiplier is the fix"
    return "D1-moved: the prompt did train; prompt_norm was a misleading monitor"


def run_prompt(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"rule": RULE, "runs": {}}
    for name, run_dir in args.run:
        run_dir = Path(run_dir)
        cfg = load_config(run_dir / "config.yaml")
        payload = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu",
                             weights_only=False)  # nosec B614 - own checkpoint
        init_model = rebuild_init(cfg)
        names = trainable_names(init_model)
        init_sd = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
        del init_model
        final_sd = payload["model"]
        opt = optimizer_state(payload, names)
        steps = int(payload["global_step"])
        budget = lr_budget(cfg.train.learning_rate, cfg.train.warmup_steps, steps, steps)
        rows = {
            label: merged_stats(pick(final_sd, names, suffix), final_sd, init_sd, opt, budget)
            for label, suffix in WATCHED.items()
        }
        if rows["prompt"] is not None:
            logged = logged_init_norm(run_dir / "metrics.jsonl")
            rows["prompt"]["logged_norm_step0"] = logged
            rows["prompt"]["init_reproduced"] = init_reproduced(rows["prompt"]["norm_init"], logged)
        out["runs"][name] = {
            "run_dir": str(run_dir),
            "text_prompt": cfg.stp.text_prompt,
            "global_step": steps,
            "lr_budget": budget,
            "rows": rows,
            "reading": classify(rows["prompt"]),
        }
        LOGGER.info("%s: %s", name, out["runs"][name]["reading"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / PROMPT_JSON, out)
    write_text_atomic(args.out_dir / PROMPT_MD, render_prompt(out))
    return out


def _num(x: Any, spec: str = ".4g") -> str:
    return "—" if x is None else format(x, spec)


def render_prompt(out: dict[str, Any]) -> str:
    lines = [f"# LaGoVAD-STP diagnostic D-1 — did the prompt train? (rule: `{out['rule']}`)", "",
             "| run | param | numel | ‖θ₀‖ | ‖θ‖ | ‖θ - θ₀‖ | cos(θ, θ₀) | RMS Δ / Σlr | "
             "grad RMS | median √v̂ | share √v̂ < 10 eps | SNR (noise ≈) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, run in out["runs"].items():
        for label, row in run["rows"].items():
            if row is None:
                continue
            snr = f"{_num(row.get('snr'), '.3f')} ({_num(row.get('noise_snr'), '.3f')})"
            lines.append(
                f"| {name} | {label} | {row['numel']} | {_num(row['norm_init'])} | "
                f"{_num(row['norm_final'])} | {_num(row['displacement'])} | "
                f"{_num(row['cos_final_init'], '.5f')} | {_num(row['budget_used'], '.3f')} | "
                f"{_num(row.get('grad_rms'))} | {_num(row.get('grad_rms_median'))} | "
                f"{_num(row.get('share_below_10eps'), '.3f')} | {snr} |")
    lines += ["", "| run | prompt layout | steps | Σ lr | ‖θ₀‖ rebuilt / logged step 0 | "
              "reading (§5 D-1) |", "|---|---|---|---|---|---|"]
    for name, run in out["runs"].items():
        p = run["rows"].get("prompt") or {}
        check = (f"{_num(p.get('norm_init'), '.6f')} / {_num(p.get('logged_norm_step0'), '.6f')}"
                 f" ({'ok' if p.get('init_reproduced') else 'NOT reproduced'})")
        lines.append(f"| {name} | {run['text_prompt']} | {run['global_step']} | "
                     f"{run['lr_budget']:.4g} | {check} | {run['reading']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- pointing (D-2)


@dataclass
class SelectionRead:
    """Per-clip reads of one selected set (top-K tiles per row)."""

    lift: float | None  # share of top-K tiles touching a box - chance, anomaly rows
    lift_vs_pre: float | None  # the same minus its pre-anomaly value under the first mask
    regions: dict[str, float]  # share of top-K in each region - region share, every row


def _touch(index: np.ndarray, masks: list[np.ndarray]) -> float:
    return float(np.mean([(w[idx] > 0).mean() - (w > 0).mean() for idx, w in zip(index, masks,
                                                                                 strict=True)]))


def selection_read(
    index: np.ndarray, steps: np.ndarray, labels: np.ndarray, boxes: dict[int, list[list[float]]]
) -> SelectionRead:
    """``index (L, K)`` grid indices of one clip -> lift over chance, pre control and regions."""
    first_pos = int(np.argmax(labels)) if labels.any() else len(labels)
    anom = [k for k in range(len(steps)) if labels[steps[k]] and boxes.get(int(steps[k]))]
    masks = {k: tile_weights(boxes[int(steps[k])], GRID) for k in anom}
    anom = [k for k in anom if 0 < masks[k].sum() < GRID * GRID]
    pre = [k for k in range(len(steps)) if steps[k] < first_pos]
    lift = vs_pre = None
    if len(anom) >= constants.V5_MIN_STEPS:
        lift = _touch(index[anom], [masks[k] for k in anom])
        if len(pre) >= constants.V5_MIN_STEPS:
            fixed = masks[anom[0]]
            vs_pre = lift - _touch(index[pre], [fixed] * len(pre))
    regions = {name: float(region[index].mean() - region.mean())
               for name, region in REGIONS.items()}
    return SelectionRead(lift, vs_pre, regions)


@dataclass
class ClipInputs:
    labels: np.ndarray
    steps: np.ndarray
    boxes: dict[int, list[list[float]]]
    ego: bool


def load_clip_inputs(args: argparse.Namespace, ids: list[str]) -> dict[str, ClipInputs]:
    records = {r.video_id: r for r in parse_metadata(args.metadata,
                                                    read_split_ids(args.split_file))}
    out: dict[str, ClipInputs] = {}
    for v in ids:
        n = int(np.load(args.s1_dir / f"{v}.npy", mmap_mode="r").shape[0])
        rows = int(np.load(args.patch_dir / constants.STP_MAGNITUDE_SUBDIR / f"{v}.npy",
                           mmap_mode="r").shape[0])
        steps = np.arange(rows) * constants.STP_DOTA_STRIDE
        if rows != math.ceil(n / constants.STP_DOTA_STRIDE):
            raise ValueError(f"{v}: {rows} patch rows for {n} native frames at stride "
                             f"{constants.STP_DOTA_STRIDE} (lesson C13)")
        boxes, ego = load_boxes(args.annotations_dir / f"{v}.json")
        out[v] = ClipInputs(native_labels(records[v], n), steps, boxes, ego)
    return out


def variant_reads(
    clips: dict[str, ClipInputs], mo: dict[str, np.ndarray], index: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Mo pointing (v5 Stage 0 reads) + selected-set reads, all / ego / non-ego, with CIs."""
    reads = {v: clip_read(mo[v], mo[v].mean(1), c.steps, c.labels, c.boxes, constants.STP_TOP_K)
             for v, c in clips.items()}
    ego = {v: c.ego for v, c in clips.items()}
    bins: dict[str, str | None] = dict.fromkeys(clips)
    sel = {v: selection_read(index[v], c.steps, c.labels, c.boxes) for v, c in clips.items()}

    def by(fn: Any) -> dict[str, Any]:
        out = {}
        for label, keep in (("all", lambda v: True), ("ego", lambda v: ego[v]),
                            ("non_ego", lambda v: not ego[v])):
            vals = {v: x for v, s in sel.items() if keep(v) and (x := fn(s)) is not None}
            out[label] = ci(vals)
        return out

    return {
        "mo": summarize_rep(reads, ego, bins),
        "topk_lift": by(lambda s: s.lift),
        "topk_lift_vs_pre": by(lambda s: s.lift_vs_pre),
        "regions": {name: by(lambda s, n=name: s.regions[n]) for name in REGIONS},
        "clips": len(clips),
    }


def ego_compensated(windows_dir: Path, v: str) -> tuple[np.ndarray, np.ndarray]:
    rows = np.load(windows_dir / f"{v}.npy").astype(np.float32)
    patches = patch_cells(rows.reshape(len(rows), 1 + constants.STP_PATCHES, -1))
    mo = motion_magnitude(patches, constants.STP_MOTION_EGO)
    _, index = select_top_k(patches, mo, constants.STP_TOP_K)
    return mo, index


def run_pointing(args: argparse.Namespace) -> dict[str, Any]:
    ids = sorted(load_split(constants.V2_SPLIT_DOTA_CAP_DEV))
    clips = load_clip_inputs(args, ids)
    mo = {v: np.load(args.patch_dir / constants.STP_MAGNITUDE_SUBDIR / f"{v}.npy")
          .astype(np.float64) for v in ids}
    index = {v: np.load(args.patch_dir / constants.STP_INDEX_SUBDIR / f"{v}.npy")
             .astype(np.int64) for v in ids}
    out: dict[str, Any] = {
        "rule": RULE,
        "patch_dir": str(args.patch_dir),
        "ego_count": {"ego": sum(c.ego for c in clips.values()),
                      "non_ego": sum(not c.ego for c in clips.values())},
        "variants": {constants.STP_MOTION_RAW: variant_reads(clips, mo, index)},
    }
    if args.windows_dir is not None:
        pairs = {v: ego_compensated(args.windows_dir, v) for v in ids}
        out["variants"][constants.STP_MOTION_EGO] = variant_reads(
            clips, {v: p[0] for v, p in pairs.items()}, {v: p[1] for v, p in pairs.items()})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / POINTING_JSON, out)
    write_text_atomic(args.out_dir / POINTING_MD, render_pointing(out))
    return out


def _ci(stat: dict[str, float] | None) -> str:
    if stat is None:
        return "—"
    return f"{stat['mean']:+.4f} [{stat['low']:+.4f}, {stat['high']:+.4f}]"


def render_pointing(out: dict[str, Any]) -> str:
    n = out["ego_count"]
    lines = [f"# LaGoVAD-STP diagnostic D-2 — do the top-{constants.STP_TOP_K} motion patches "
             f"point at the anomaly? (rule: `{out['rule']}`)", "",
             f"DoTA-CAP-dev, ego-involved {n['ego']} / not {n['non_ego']}. Every value is "
             "per-clip, mean [95 % source-video cluster CI].", "",
             "| Mo | subset | top-K lift over chance | lift - pre-anomaly | border - 24/49 | "
             "bottom row - 7/49 | centre 3x3 - 9/49 |", "|---|---|---|---|---|---|---|"]
    for motion, var in out["variants"].items():
        for subset in ("all", "ego", "non_ego"):
            lines.append(
                f"| {motion} | {subset} | {_ci(var['topk_lift'][subset])} | "
                f"{_ci(var['topk_lift_vs_pre'][subset])} | "
                f"{_ci(var['regions']['border'][subset])} | "
                f"{_ci(var['regions']['bottom_row'][subset])} | "
                f"{_ci(var['regions']['centre_3x3'][subset])} |")
    lines += ["", "Mo's own pointing (v5 Stage 0 reads on the 49-tile Mo map; training-free AUCs):",
              "", "| Mo | rho | top-1 hit - chance | hit - pre | macro global | macro top-K mean |",
              "|---|---|---|---|---|---|"]
    for motion, var in out["variants"].items():
        m = var["mo"]
        lines.append(f"| {motion} | {_ci(m['rho'])} | {_ci(m['hit_minus_chance'])} | "
                     f"{_ci(m['hit_minus_pre'])} | {_ci(m['macro']['global'])} | "
                     f"{_ci(m['macro']['topk'])} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prompt = sub.add_parser("prompt", help="D-1: displacement + Adam state of the prompt")
    prompt.add_argument("--run", nargs=2, action="append", required=True,
                        metavar=("NAME", "RUN_DIR"),
                        help="run dir with config.yaml + checkpoint_last.pt (repeatable)")
    point = sub.add_parser("pointing", help="D-2: top-K patch pointing on DoTA-CAP-dev")
    point.add_argument("--patch-dir", type=Path, required=True, help="DoTA_CAP_s3_k12_raw")
    point.add_argument("--windows-dir", type=Path, default=None,
                       help="clip_windows cache (grids 1 7): adds the ego-compensated Mo")
    point.add_argument("--s1-dir", type=Path, required=True, help="DoTA_s1_ncc (frame counts)")
    point.add_argument("--annotations-dir", type=Path, required=True,
                       help="DoTA raw_annotations (boxes, ego_involve)")
    point.add_argument("--metadata", type=Path, required=True)
    point.add_argument("--split-file", type=Path, required=True)
    for p in (prompt, point):
        p.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    args = build_arg_parser().parse_args(argv)
    if args.command == "prompt":
        out = run_prompt(args)
        LOGGER.info("\n%s", render_prompt(out))
    else:
        out = run_pointing(args)
        LOGGER.info("\n%s", render_pointing(out))


if __name__ == "__main__":
    main()
