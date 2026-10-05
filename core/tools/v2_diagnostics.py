"""v2 P6 pilot diagnostics for one checkpoint (addendum §4, §14 O1-O7; proposal §10.1).

One forward pass per item gives the trunk output ``V^t`` (``vis_feats``; KIP is off in v2)
and ``y^bin``. From them, for the arm the checkpoint was trained as:

* **O1 guardrails on T2-val** (the v2 dataset dir's evaluation file, ``core.tools.v2_dataset``):
  micro (``score_norm=auto``) < the constant-per-clip oracle; macro >= micro; micro >= A0's micro
  - ``V2_GUARD_A0_MARGIN`` when ``--a0-diag`` is given.
* **O2 collapse signature (C14):** window-level AUC of the max score vs macro.
* **O1' (Amendment 8, §17 Q2)** when ``--a0-diag`` is given: the arm is *collapsed* iff its
  window-level AUC is above A0's **and** its macro is below A0's - ``V2_GUARD_A0_MARGIN``;
  it passes iff not collapsed and O1's A0-margin leg holds. Printed beside O1, which stays.
* **O3 motion share:** ``motion_share`` (``rho_u``) and ``w_u_norm`` from ``metrics.jsonl``.
* **O4 position probe on V^t:** standardized ridge predicting ``t/T`` within each T2-val
  window, out-of-fold R² (folds grouped by source video). Printed beside the input's own R².
* **O5 source-shortcut AUC:** how well ``V^t`` (logistic probe, grouped folds) and ``y^bin``
  (the score alone, direction-free) separate T2-val steps from DoTA-dev steps -- the corpus
  separability lesson C38 measured on inputs. DoTA-dev is read **without labels**: no DoTA score
  or metric is computed, so D4 ("DoTA-dev is not printed") holds.

``--dota-split dota_cap_dev`` (Amendment 6, batch 2) reads O5 on the DoTA-CAP part only: a
motion arm has DoTA rows only there, and every arm of the batch is re-read on the same clips.

DoTA-dev rows follow protocol B through the arm's own input (``protocol_b_eval.input_rows``):
the plain ``DoTA_s1_ncc`` for A0, a baked ``apply --stride 3`` cache otherwise. The run's
``metrics.jsonl`` must end at the checkpoint's step (J10).

CLI::

    python -m core.tools.v2_diagnostics --run A1_s2099 runs/A1_s2099/checkpoint_last.pt \\
        --data-dir data/DADA2000_orig_v2 --t2-input-dir cache/v2/A1_R2/DADA2000_orig \\
        --dota-input-dir cache/v2/A1_R2/DoTA_s3_from_s1 --dota-s1-dir cache/clip/DoTA_s1_ncc \\
        --dota-data-dir data/DoTA/labels_s8 --a0-diag outputs/v2/pilot/A0_s2099/diag.json \\
        --out-dir outputs/v2/pilot/A1_s2099
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from core import constants
from core.data.dataset import FeatureEvalDataset
from core.data.definitions import dataset_abbr, item_verbalizer
from core.data.v2_inputs import check_input_manifest
from core.data.v2_splits import dota_group, load_split
from core.device import resolve_device
from core.eda.features import transfer_scores
from core.eda.protocol import clip_constant_oracle
from core.inference import make_class_feats_fn
from core.metrics import frame_auc, pooled_metrics
from core.models.kat_vad import KATVAD
from core.models.text_encoding import TextEncodeFn
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import check_baked_stride, input_rows
from core.tools.rate_matched_eval import Run, load_finished_model
from core.train import METRICS_FILENAME, load_class_names

LOGGER = logging.getLogger(__name__)

DIAG_JSON = "diag.json"
DIAG_MD = "diag.md"
T2 = 1  # corpus label of a T2-val step in the source-shortcut probe
DOTA = 0
SHARE_KEYS = ("motion_share", "w_u_norm")


# ---------------------------------------------------------------------------
# one forward pass
# ---------------------------------------------------------------------------
@torch.inference_mode()
def forward_item(
    model: KATVAD, rows: np.ndarray, class_feats: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """``(V^t (L, D), y^bin (L,))`` for one item scored whole (no sliding window)."""
    if len(rows) > constants.MAX_VIS_LEN:
        raise ValueError(f"{len(rows)} steps > MAX_VIS_LEN {constants.MAX_VIS_LEN}")
    device = next(model.parameters()).device
    feats = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32))[None].to(
        device
    )
    lengths = torch.tensor([len(rows)], device=device)
    out = model(feats, lengths, class_feats=class_feats.to(device))
    return (
        out["vis_feats"][0].float().cpu().numpy(),
        out["cls_bin_logits"][0].sigmoid().float().cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# reads (pure, tested without a model)
# ---------------------------------------------------------------------------
def guardrails(
    scores: list[np.ndarray], labels: list[np.ndarray], a0_micro: float | None
) -> dict[str, Any]:
    """O1 + O2 on T2-val windows."""
    metrics = pooled_metrics(scores, labels, constants.SCORE_NORM_AUTO)
    micro, macro = float(metrics["auc"]), float(metrics["auc_macro"])
    oracle = float(clip_constant_oracle(labels)["auc_micro"])
    window_label = np.asarray([int(a.max() > 0) for a in labels])
    window_auc = frame_auc(np.asarray([s.max() for s in scores]), window_label)
    checks = {
        "micro_below_clip_oracle": micro < oracle,
        "macro_at_least_micro": macro >= micro,
    }
    if a0_micro is not None:
        checks["micro_within_a0_margin"] = (
            micro >= a0_micro - constants.V2_GUARD_A0_MARGIN
        )
    return {
        "micro": micro,
        "macro": macro,
        "score_norm": metrics["score_norm"],
        "clip_oracle_micro": oracle,
        "window_level_auc": window_auc,
        "a0_micro": a0_micro,
        "checks": checks,
        "pass": all(checks.values()),
    }


def o1_prime(arm: dict[str, Any], a0: dict[str, Any]) -> dict[str, Any]:
    """Amendment 8 Q2 on two ``guardrails`` read-outs of the same set (arm, A0 of the same seed)."""
    margin = constants.V2_GUARD_A0_MARGIN
    window_up = arm["window_level_auc"] > a0["window_level_auc"]
    macro_down = arm["macro"] < a0["macro"] - margin
    collapsed = bool(window_up and macro_down)
    checks = {
        "not_collapsed": not collapsed,
        "micro_within_a0_margin": arm["micro"] >= a0["micro"] - margin,
    }
    return {
        "window_auc_delta": arm["window_level_auc"] - a0["window_level_auc"],
        "macro_delta": arm["macro"] - a0["macro"],
        "micro_delta": arm["micro"] - a0["micro"],
        "collapsed": collapsed,
        "checks": checks,
        "pass": all(checks.values()),
    }


def position_r2(
    features: dict[str, np.ndarray], groups: dict[str, str], folds: int
) -> float:
    """O4: out-of-fold R² of a standardized ridge predicting ``t/T`` from each step."""
    ids = sorted(features)
    matrix = np.concatenate([features[i] for i in ids]).astype(np.float64)
    target = np.concatenate(
        [np.arange(len(features[i])) / len(features[i]) for i in ids]
    )
    names = {g: k for k, g in enumerate(sorted({groups[i] for i in ids}))}
    group = np.concatenate([np.full(len(features[i]), names[groups[i]]) for i in ids])
    predicted = np.zeros_like(target)
    for train, test in GroupKFold(n_splits=min(folds, len(names))).split(
        matrix, target, group
    ):
        scaler = StandardScaler().fit(matrix[train])
        model = Ridge(alpha=constants.V2_DIAG_RIDGE_ALPHA).fit(
            scaler.transform(matrix[train]), target[train]
        )
        predicted[test] = model.predict(scaler.transform(matrix[test]))
    residual = float(((target - predicted) ** 2).sum())
    total = float(((target - target.mean()) ** 2).sum())
    return 1.0 - residual / total


def source_shortcut(
    t2: dict[str, np.ndarray],
    dota: dict[str, np.ndarray],
    groups: dict[str, str],
    folds: int,
    seed: int,
) -> float:
    """O5 on a representation: out-of-fold AUC of a logistic probe, T2-val vs DoTA-dev."""
    ids = [*sorted(t2), *sorted(dota)]
    rows = {**t2, **dota}
    matrix = np.concatenate([rows[i] for i in ids]).astype(np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    target = np.concatenate(
        [np.full(len(rows[i]), T2 if i in t2 else DOTA) for i in ids]
    )
    names = {g: k for k, g in enumerate(sorted({groups[i] for i in ids}))}
    group = np.concatenate([np.full(len(rows[i]), names[groups[i]]) for i in ids])
    scores = np.zeros(len(target))
    for train, test in GroupKFold(n_splits=folds).split(matrix, target, group):
        scores[test] = transfer_scores(matrix[train], target[train], matrix[test], seed)
    return frame_auc(scores, target)


def score_shortcut(t2: dict[str, np.ndarray], dota: dict[str, np.ndarray]) -> float:
    """O5 on ``y^bin``: the score alone as a corpus classifier, direction-free."""
    scores = np.concatenate(
        [*(t2[i] for i in sorted(t2)), *(dota[i] for i in sorted(dota))]
    )
    target = np.concatenate(
        [
            *(np.full(len(t2[i]), T2) for i in sorted(t2)),
            *(np.full(len(dota[i]), DOTA) for i in sorted(dota)),
        ]
    )
    auc = frame_auc(scores, target)
    return max(auc, 1.0 - auc)


def motion_share(metrics_path: Path) -> dict[str, Any] | None:
    """O3: first / last / max of ``rho_u`` and ``‖W_u‖`` (``None`` for a no-motion arm)."""
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out: dict[str, Any] = {}
    for key in SHARE_KEYS:
        values = [float(r[key]) for r in records if key in r]
        if values:
            out[key] = {
                "first": values[0],
                "last": values[-1],
                "max": max(values),
                "records": len(values),
            }
    return out or None


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def score_t2_windows(
    model: KATVAD,
    text_encode_fn: TextEncodeFn,
    data_dir: Path,
    input_dir: Path,
    t2_dataset: str,
) -> tuple[FeatureEvalDataset, dict[str, dict[str, Any]]]:
    """Every evaluation window of ``data_dir`` through the arm's input cache, scored whole.

    Returns the dataset and, per window id, ``vt`` (``V^t``), ``x`` (input rows), ``y``
    (``y^bin``), ``label`` (frame labels) and ``group`` (source video).
    """
    dataset = FeatureEvalDataset(data_dir, input_dir)
    names = load_class_names(data_dir)
    abbr = dataset_abbr(t2_dataset)
    out: dict[str, dict[str, Any]] = {}
    for i in range(len(dataset)):
        item = dataset[i]
        wid = item["video_id"]
        rows = item["v_feat"].numpy()
        class_feats = make_class_feats_fn(
            text_encode_fn, names, item_verbalizer(abbr, wid)
        )()
        vt, y = forward_item(model, rows, class_feats)
        out[wid] = {
            "vt": vt,
            "x": rows,
            "y": y,
            "label": item["frame_label"].numpy().astype(np.int64),
            "group": dataset.slicer.source_of(wid),
        }
    return dataset, out


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    name, ckpt = args.run
    model, step, text_encode_fn, cfg = load_finished_model(
        Run(name, Path(ckpt)), device
    )
    check_input_manifest(args.t2_input_dir, cfg.v2.crn, cfg.v2.motion)
    check_input_manifest(args.dota_input_dir, cfg.v2.crn, cfg.v2.motion)
    baked = check_baked_stride(args.dota_input_dir, args.stride)

    _, windows = score_t2_windows(
        model, text_encode_fn, args.data_dir, args.t2_input_dir, args.t2_dataset
    )
    vt_t2 = {w: r["vt"] for w, r in windows.items()}
    x_t2 = {w: r["x"] for w, r in windows.items()}
    y_t2 = {w: r["y"] for w, r in windows.items()}
    labels = {w: r["label"] for w, r in windows.items()}
    groups: dict[str, str] = {w: r["group"] for w, r in windows.items()}

    dota_ids = load_split(args.dota_split, args.split_dir)
    dota_names = load_class_names(args.dota_data_dir)
    dota_abbr = dataset_abbr(constants.DOTA_DATASET)
    vt_dota: dict[str, np.ndarray] = {}
    y_dota: dict[str, np.ndarray] = {}
    for vid in dota_ids:
        frames = len(np.load(args.dota_s1_dir / f"{vid}.npy", mmap_mode="r"))
        rows = input_rows(args.dota_input_dir, vid, frames, args.stride, baked)
        class_feats = make_class_feats_fn(
            text_encode_fn, dota_names, item_verbalizer(dota_abbr, vid)
        )()
        vt_dota[vid], y_dota[vid] = forward_item(model, rows, class_feats)
        groups[vid] = f"dota:{dota_group(vid)}"

    a0 = json.loads(args.a0_diag.read_text(encoding="utf-8")) if args.a0_diag else None
    window_ids = sorted(labels)
    guard = guardrails(
        [y_t2[w] for w in window_ids],
        [labels[w] for w in window_ids],
        None if a0 is None else a0["guardrails"]["micro"],
    )
    if a0 is not None:
        guard["o1_prime"] = o1_prime(guard, a0["guardrails"])
    readout: dict[str, Any] = {
        "addendum": "core/docs/v2/PREREG_ADDENDUM.md §4, §14 (O1-O7)",
        "run": name,
        "checkpoint": str(ckpt),
        "global_step": step,
        "arm": {
            "crn": cfg.v2.crn,
            "motion": cfg.v2.motion,
            "motion_dim": cfg.model.motion_dim,
        },
        "t2_val_windows": len(window_ids),
        "dota_split": args.dota_split,
        "dota_dev_clips_unlabelled": len(dota_ids),
        "guardrails": guard,
        "motion_share": motion_share(Path(ckpt).parent / METRICS_FILENAME),
        "position_r2": {
            "v_t": position_r2(vt_t2, groups, args.folds),
            "input": position_r2(x_t2, groups, args.folds),
            "a0_v_t": None if a0 is None else a0["position_r2"]["v_t"],
        },
        "source_shortcut_auc": {
            "v_t": source_shortcut(vt_t2, vt_dota, groups, args.folds, args.seed),
            "y_bin": score_shortcut(y_t2, y_dota),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / DIAG_JSON, readout)
    write_text_atomic(args.out_dir / DIAG_MD, render_markdown(readout))
    LOGGER.info(
        "%s: guardrails %s -> %s",
        name,
        "PASS" if guard["pass"] else "FAIL",
        args.out_dir,
    )
    return readout


def d5(a0: dict[str, Any], references: list[dict[str, Any]]) -> dict[str, Any]:
    """O6 / D5: A0's T2-val micro and macro each within ``V2_D5_MARGIN`` of the references' mean."""
    out: dict[str, Any] = {
        "margin": constants.V2_D5_MARGIN,
        "references": len(references),
    }
    passed = True
    for key in ("micro", "macro"):
        ref = float(np.mean([r["guardrails"][key] for r in references]))
        delta = float(a0["guardrails"][key]) - ref
        out[key] = {"a0": a0["guardrails"][key], "reference_mean": ref, "delta": delta}
        passed &= abs(delta) <= constants.V2_D5_MARGIN
    out["pass"] = passed
    return out


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _yes(flag: bool) -> str:
    return "PASS" if flag else "**FAIL**"


def render_markdown(r: dict[str, Any]) -> str:
    g = r["guardrails"]
    lines = [
        f"# v2 pilot diagnostics — {r['run']}",
        "",
        f"Arm crn={r['arm']['crn']} motion={r['arm']['motion']} · step {r['global_step']} · "
        f"T2-val windows {r['t2_val_windows']} · `{r.get('dota_split', 'dota_dev')}` clips read "
        f"unlabelled {r['dota_dev_clips_unlabelled']} (no DoTA metric computed, D4).",
        "",
        "| T2-val | value |",
        "|---|---|",
        f"| micro (norm {g['score_norm']}) | {g['micro']:.4f} |",
        f"| macro | {g['macro']:.4f} |",
        f"| clip oracle micro | {g['clip_oracle_micro']:.4f} |",
        f"| window-level AUC (C14 signature) | {g['window_level_auc']:.4f} |",
        f"| A0 micro | {_num(g['a0_micro'])} |",
        "",
        "Guardrails: "
        + ", ".join(f"{k} {_yes(v)}" for k, v in g["checks"].items())
        + f" → **{'PASS' if g['pass'] else 'FAIL'}**",
    ]
    if "o1_prime" in g:
        lines += ["", render_o1_prime(g["o1_prime"])]
    lines += [
        "",
        f"Position R² on V^t {r['position_r2']['v_t']:.4f} (input {r['position_r2']['input']:.4f}"
        + (
            ""
            if r["position_r2"]["a0_v_t"] is None
            else f", A0 {r['position_r2']['a0_v_t']:.4f}"
        )
        + ")",
        "",
        f"Source-shortcut AUC (T2-val vs {r.get('dota_split', 'dota_dev')}): "
        f"V^t {r['source_shortcut_auc']['v_t']:.4f}, "
        f"y^bin {r['source_shortcut_auc']['y_bin']:.4f}",
    ]
    if r["motion_share"]:
        lines += ["", "| motion | first | last | max |", "|---|---|---|---|"]
        for key, v in r["motion_share"].items():
            lines.append(
                f"| {key} | {v['first']:.4g} | {v['last']:.4g} | {v['max']:.4g} |"
            )
    return "\n".join(lines) + "\n"


def render_o1_prime(p: dict[str, Any]) -> str:
    """One line for O1' (Amendment 8 Q2), deltas against A0 of the same seed and set."""
    return (
        f"O1' (Amendment 8): Δ window AUC {p['window_auc_delta']:+.4f}, "
        f"Δ macro {p['macro_delta']:+.4f}, Δ micro {p['micro_delta']:+.4f}; "
        + ", ".join(f"{k} {_yes(v)}" for k, v in p["checks"].items())
        + f" → **{'PASS' if p['pass'] else 'FAIL'}**"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--run", nargs=2, required=True, metavar=("NAME", "CKPT"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="v2 dataset dir (core.tools.v2_dataset): T2-val windows",
    )
    parser.add_argument(
        "--t2-input-dir",
        type=Path,
        required=True,
        help="the arm's T2 cache (plain CLIP for A0, a `fit` cache otherwise)",
    )
    parser.add_argument(
        "--dota-input-dir",
        type=Path,
        required=True,
        help="DoTA_s1_ncc (A0) or the arm's `apply --stride 3` cache",
    )
    parser.add_argument(
        "--dota-s1-dir", type=Path, required=True, help="cache/clip/DoTA_s1_ncc"
    )
    parser.add_argument(
        "--dota-data-dir",
        type=Path,
        required=True,
        help="DoTA labels dir (defs.json only; no label is read)",
    )
    parser.add_argument("--t2-dataset", default=constants.DADA_ORIGIN_DATASET)
    parser.add_argument(
        "--a0-diag", type=Path, default=None, help="A0's diag.json (O1, O4)"
    )
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument(
        "--dota-split",
        default=constants.V2_SPLIT_DOTA_DEV,
        choices=(constants.V2_SPLIT_DOTA_DEV, constants.V2_SPLIT_DOTA_CAP_DEV),
        help="O5's DoTA side: dota_dev (batch 1) or dota_cap_dev (batch 2, Amendment 6)",
    )
    parser.add_argument("--stride", type=int, default=constants.V2_E1_STRIDE_BC)
    parser.add_argument("--folds", type=int, default=constants.V2_DIAG_FOLDS)
    parser.add_argument("--seed", type=int, default=constants.SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
