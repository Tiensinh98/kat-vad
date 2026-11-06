"""KIM pilot KP1-KP4: does the change of the host's features already rise at anomalies? (no train)

Rule ``core/docs/final/KIM_RUN.md`` §5, written before the pilot runs. Inference only, on the
decided control's s2024 checkpoint (K0 or K0k9, after the kernel read-out and before K2 / K5:
dated note in §4, W0's checkpoint is gone). For every video of a set, the candidate step scores
are the per-step change magnitudes of four feature sequences (no learned part):

* ``vt`` -- the RoFormer output (KIM's input), ``x`` -- the RoFormer input ``c + W_u m``,
  ``clip`` -- the CLIP columns, ``motion`` -- the baked VideoMAE columns;
* order 1: ``mean_c |z_t - z_{t-1}| / rho``, order 2: ``mean_c |d1_t - d1_{t-1}| / rho`` with
  ``d1`` the per-channel first difference, ``rho`` the video's median order-1 change (D6).

Labels and the host's own scores are read from **the host's existing eval outputs** of the same
set (``clip_scores.npz`` of ``protocol_b_eval`` / ``dada_whole_eval``, or ``scores/*.npz`` of
``core.evaluate``), so the pilot scores exactly the frames and labels the host's read-out scored.
The input rows come from the baked cache that eval read. Kinds:

* ``native`` (DoTA-CAP-dev, protocol B): rows are ``s1[::stride]`` steps, curves interpolated to
  the native frames of the saved labels (J3);
* ``rows`` (DADA-2000 whole video, TAD): one label per cached row.

Checks (each with the consequence fixed in the rule): KP1 macro of every candidate (DoTA also by
ego flag); KP2 rank-average of the host's score with ``vt`` order 2 against the host alone
(paired, per video); KP3 share of the top-5 % ``vt`` order-1 steps on the first or last difference
vs chance; KP4 lag of the ``vt`` order-2 peak behind the label onset (seconds, sets with a known
step). The pilot reads these sets before K2 / K5 train: every KIM read-out says so.

CLI::

    python -m core.tools.kim_pilot --name K0 --ckpt outputs/kim/K0/s2024/checkpoint_last.pt \\
        --set dota_cap_dev native <K0 DoTA read> <A2 DoTA bake> \\
        --set dada rows <K0 DADA read> <A2 DADA bake> \\
        --set tad rows <K0 TAD read>/s2024 <A2 TAD bake> \\
        --data-dir data/Nexar_window --metadata data/DoTA/metadata_val.json \\
        --split-file data/DoTA/val_split.txt --out-dir outputs/kim/REPORTS/pilot
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core import constants
from core.data.collate import padding_mask
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_inputs import check_input_manifest
from core.device import resolve_device
from core.metrics import frame_auc
from core.models.kat_vad import KATVAD
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import CLIP_SCORES_NPZ, read_clip_scores
from core.tools.rate_matched_eval import Run, load_finished_model, to_native
from core.tools.stp_readout import ci as dota_ci
from core.tools.tadz_readout import load_run
from core.tools.zs_readout import video_ci

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "kim_pilot"
RULE = "core/docs/final/KIM_RUN.md §5"
KIND_NATIVE, KIND_ROWS = "native", "rows"
KINDS = (KIND_NATIVE, KIND_ROWS)
FEATURES = ("vt", "x", "clip", "motion")
ORDERS = (1, 2)
KIM_INPUT = "vt"
HOST_ROW, RANK_NAME = "host", "host+vt2 (rank avg)"
DOTA_SET, DADA_SET = constants.V2_SPLIT_DOTA_CAP_DEV, "dada"
EGO_SETS = (DOTA_SET,)
DEFAULT_STEP_SECONDS = {  # seconds per curve index (native frame for DoTA, row otherwise)
    DOTA_SET: 1.0 / constants.DOTA_FPS,
    DADA_SET: constants.NEXAR_ZS_DADA_STRIDE / constants.DADA_ASSUMED_FPS,
}
PASS, FAIL = "pass", "fail"

Curves = dict[str, np.ndarray]


def candidate_name(feature: str, order: int) -> str:
    return f"{feature}{order}"


# ---------------------------------------------------------------------------
# pure pieces (tested without a model)
# ---------------------------------------------------------------------------
def change_scores(z: np.ndarray) -> dict[int, np.ndarray]:
    """``{1: order-1, 2: order-2}`` change per step of one ``(L, D)`` sequence (row 0 = 0)."""
    z = np.asarray(z, dtype=np.float64)
    d1 = np.zeros_like(z)
    d1[1:] = np.abs(z[1:] - z[:-1])
    d2 = np.zeros_like(z)
    d2[2:] = np.abs(d1[2:] - d1[1:-1])
    per_step = d1.mean(axis=1)
    rho = float(np.median(per_step[1:])) if len(z) > 1 else 1.0
    rho = rho if rho > 0 else 1.0
    return {1: per_step / (rho + constants.KIM_EPS), 2: d2.mean(axis=1) / (rho + constants.KIM_EPS)}


def rank01(x: np.ndarray) -> np.ndarray:
    """Ranks scaled to ``[0, 1]`` (ties by order; a constant curve maps to 0.5)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2 or np.ptp(x) == 0:
        return np.full(len(x), 0.5)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    return ranks / (len(x) - 1)


def two_class_aucs(scores: Curves, labels: Curves) -> dict[str, float]:
    return {v: frame_auc(scores[v], labels[v]) for v in sorted(scores)
            if 0 < int(labels[v].sum()) < len(labels[v])}


def edge_share(step_curve: np.ndarray) -> tuple[float, float] | None:
    """(share of the top-5 % differences on the first or last one, chance) for one video.

    Differences live on steps ``1..L-1``; the edges are steps 1 and ``L-1``. ``None`` below 4
    steps (no interior).
    """
    diffs = np.asarray(step_curve, dtype=np.float64)[1:]
    if len(diffs) < 3:
        return None
    top = max(1, math.ceil(constants.KIM_PILOT_TOP_SHARE * len(diffs)))
    picked = np.argsort(-diffs, kind="stable")[:top]
    at_edge = np.isin(picked, (0, len(diffs) - 1)).sum()
    return float(at_edge) / top, 2.0 / len(diffs)


def onset_lag(curve: np.ndarray, labels: np.ndarray, seconds: float) -> float | None:
    """Seconds from the first positive frame to the curve's peak (``None`` without a positive)."""
    positives = np.flatnonzero(np.asarray(labels) > 0)
    if not len(positives):
        return None
    return float(int(np.argmax(curve)) - int(positives[0])) * seconds


def lag_summary(lags: list[float]) -> dict[str, float] | None:
    if not lags:
        return None
    arr = np.abs(np.asarray(lags))
    return {"clips": len(lags), "median_abs_s": float(np.median(arr)),
            "share_within": float((arr <= constants.KIM_PILOT_ONSET_S).mean())}


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
@torch.no_grad()
def feature_sequences(
    model: KATVAD, rows: np.ndarray, device: torch.device
) -> dict[str, np.ndarray]:
    """``vt`` / ``x`` / ``clip`` / ``motion`` of one video, windowed as the eval scorer does."""
    hidden = model.hidden_dim
    motion = model.motion_residual
    if motion is None:
        raise ValueError("the pilot reads a motion-stream host: motion_dim > 0")
    feats = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32))
    vt_parts, x_parts = [], []
    for start in range(0, len(feats), constants.MAX_VIS_LEN):
        window = feats[start : start + constants.MAX_VIS_LEN].unsqueeze(0).to(device)
        lengths = torch.tensor([window.shape[1]], device=device)
        vt_parts.append(model(window, lengths)["vis_feats"][0].float().cpu())
        x, _ = motion(window, padding_mask(lengths, window.shape[1]).to(device))
        x_parts.append(x[0].float().cpu())
    base = hidden + motion.motion_dim
    return {"vt": torch.cat(vt_parts).numpy(), "x": torch.cat(x_parts).numpy(),
            "clip": rows[:, :hidden], "motion": rows[:, hidden:base]}


def read_labels(eval_dir: Path) -> tuple[Curves, Curves]:
    """``(host scores, labels)`` from the host's eval output of this set."""
    if (eval_dir / CLIP_SCORES_NPZ).is_file():
        return read_clip_scores(eval_dir / CLIP_SCORES_NPZ)
    return load_run(eval_dir)


def check_model(model: KATVAD) -> None:
    if model.kim is not None or model.kip is not None:
        raise ValueError("the pilot reads the bare host: the checkpoint must have KIM and KIP off")


# ---------------------------------------------------------------------------
# one set
# ---------------------------------------------------------------------------
def score_set(
    kind: str,
    input_dir: Path,
    labels: Curves,
    model: KATVAD,
    device: torch.device,
    stride: int,
) -> tuple[dict[str, Curves], Curves]:
    """Candidate curves on the label grid, and the ``vt`` order-1 step curves (for KP3)."""
    curves: dict[str, Curves] = {candidate_name(f, o): {} for f in FEATURES for o in ORDERS}
    steps_vt1: Curves = {}
    for video_id in sorted(labels):
        rows = np.load(input_dir / f"{video_id}.npy")
        n = len(labels[video_id])
        expected = len(range(0, n, stride)) if kind == KIND_NATIVE else n
        if len(rows) != expected:
            raise ValueError(f"{video_id}: {len(rows)} input rows, the labels need {expected}")
        seqs = feature_sequences(model, rows, device)
        for feature, z in seqs.items():
            for order, steps in change_scores(z).items():
                curve = to_native(steps, stride, n) if kind == KIND_NATIVE else steps
                curves[candidate_name(feature, order)][video_id] = curve
                if feature == KIM_INPUT and order == 1:
                    steps_vt1[video_id] = steps
    return curves, steps_vt1


def summarize_set(
    name: str,
    curves: dict[str, Curves],
    steps_vt1: Curves,
    host_scores: Curves,
    labels: Curves,
    ego: dict[str, bool] | None,
    seconds: float | None,
) -> dict[str, Any]:
    """Every pilot number of one set (pure: curves in, numbers out)."""
    ci = dota_ci if name in EGO_SETS else video_ci
    rank = {v: 0.5 * (rank01(host_scores[v]) + rank01(curves[candidate_name(KIM_INPUT, 2)][v]))
            for v in labels}
    all_curves = {**curves, HOST_ROW: host_scores, RANK_NAME: rank}
    aucs = {c: two_class_aucs(s, labels) for c, s in all_curves.items()}
    rows: dict[str, Any] = {}
    for cand, per_video in aucs.items():
        row: dict[str, Any] = {"macro": ci(per_video)}
        if ego is not None:
            for split, flag in (("ego", True), ("non_ego", False)):
                row[split] = ci({v: a for v, a in per_video.items() if ego.get(v) is flag})
        rows[cand] = row
    common = sorted(set(aucs[RANK_NAME]) & set(aucs[HOST_ROW]))
    kp2 = ci({v: aucs[RANK_NAME][v] - aucs[HOST_ROW][v] for v in common})
    edges = [e for e in (edge_share(c) for c in steps_vt1.values()) if e is not None]
    kp3 = None if not edges else {
        "share": float(np.mean([e[0] for e in edges])),
        "chance": float(np.mean([e[1] for e in edges])),
        "videos": len(edges),
    }
    kp4 = None
    if seconds is not None:
        vt2 = curves[candidate_name(KIM_INPUT, 2)]
        lags = [lag for v in labels if (lag := onset_lag(vt2[v], labels[v], seconds)) is not None]
        kp4 = lag_summary(lags)
    monotone = {v: np.arange(len(g)) / len(g) for v, g in labels.items()}
    return {"videos": len(labels), "two_class": len(aucs[HOST_ROW]), "rows": rows,
            "kp2_delta": kp2, "kp3": kp3, "kp4": kp4, "seconds_per_index": seconds,
            "t_over_n": ci(two_class_aucs(monotone, labels))}


# ---------------------------------------------------------------------------
# decisions (KIM_RUN.md §5, fixed before the pilot)
# ---------------------------------------------------------------------------
def _mean(c: dict[str, float] | None) -> float | None:
    return None if c is None else float(c["mean"])


def decide(sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    dota, dada = sets.get(DOTA_SET), sets.get(DADA_SET)
    vt2 = candidate_name(KIM_INPUT, 2)
    if dota is not None:
        ego = _mean(dota["rows"][vt2].get("ego"))
        out["KP1"] = {"vt2_ego_macro": ego,
                      "verdict": PASS if ego is not None and ego >= constants.KIM_PILOT_KP1_BAR
                      else FAIL}
        k2 = dota["kp2_delta"]
        out["KP2"] = {"delta": k2, "verdict": PASS if k2 is not None and k2["low"] > 0 else FAIL}
    if dota is not None and dada is not None:
        base = {s: _mean(sets[s]["rows"][vt2]["macro"]) or 0.0 for s in (DOTA_SET, DADA_SET)}
        beats = []
        for feature in FEATURES:
            if feature == KIM_INPUT:
                continue
            cand = candidate_name(feature, 2)
            margin = {s: (_mean(sets[s]["rows"][cand]["macro"]) or 0.0) - base[s]
                      for s in (DOTA_SET, DADA_SET)}
            if all(m >= constants.KIM_PILOT_INPUT_MARGIN for m in margin.values()):
                beats.append((min(margin.values()), feature))
        out["input"] = max(beats)[1] if beats else KIM_INPUT
    kp3 = {s: r["kp3"] for s, r in sets.items() if r["kp3"] is not None}
    if kp3:
        ok = all(v["share"] <= constants.KIM_PILOT_EDGE_FACTOR * v["chance"] for v in kp3.values())
        out["KP3"] = {"verdict": PASS if ok else FAIL,
                      "consequence": "none" if ok else "mask the last valid pair (KIM_EDGE_MASK)"}
    kp4 = {s: sets[s]["kp4"] for s in (DOTA_SET, DADA_SET) if s in sets and sets[s]["kp4"]}
    if kp4:
        ok = all(v["share_within"] >= constants.KIM_PILOT_ONSET_SHARE for v in kp4.values())
        out["KP4"] = {"verdict": PASS if ok else FAIL,
                      "consequence": "none" if ok else "H6 dropped from the claims"}
    return out


# ---------------------------------------------------------------------------
# run / render
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    model, step, _, cfg = load_finished_model(Run(args.name, args.ckpt), device)
    check_model(model)
    ego = None
    if args.metadata is not None and args.split_file is not None:
        records = parse_metadata(args.metadata, read_split_ids(args.split_file))
        ego = {r.video_id: r.is_ego for r in records}
    seconds = {**DEFAULT_STEP_SECONDS, **{n: float(s) for n, s in args.step_seconds or []}}
    sets: dict[str, dict[str, Any]] = {}
    for name, kind, eval_dir, input_dir in args.set:
        if kind not in KINDS:
            raise ValueError(f"--set {name}: kind must be one of {KINDS}, got {kind!r}")
        check_input_manifest(Path(input_dir), cfg.v2.crn, cfg.v2.motion)
        host_scores, labels = read_labels(Path(eval_dir))
        stride = args.native_stride if kind == KIND_NATIVE else 1
        curves, steps_vt1 = score_set(kind, Path(input_dir), labels, model, device, stride)
        sets[name] = summarize_set(name, curves, steps_vt1, host_scores, labels,
                                   ego if name in EGO_SETS else None, seconds.get(name))
        LOGGER.info("%s: %d videos scored", name, len(labels))
    result = {"rule": RULE, "host": args.name, "checkpoint": str(args.ckpt), "global_step": step,
              "disclosure": constants.KIM_PILOT_DISCLOSURE, "sets": sets,
              "decisions": decide(sets)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{READOUT_NAME}.json", result)
    write_text_atomic(args.out_dir / f"{READOUT_NAME}.md", render_markdown(result))
    LOGGER.info("pilot -> %s", args.out_dir / f"{READOUT_NAME}.md")
    return result


def _num(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _delta(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def render_markdown(r: dict[str, Any]) -> str:
    lines = [f"# KIM pilot KP1-KP4 (host {r['host']} s2024, step {r['global_step']}, no training)",
             "",
             f"Rule: `{r['rule']}`. **{r['disclosure']}**", ""]
    for name, s in r["sets"].items():
        has_ego = any("ego" in row for row in s["rows"].values())
        head = "| candidate | macro | ego | non-ego |" if has_ego else "| candidate | macro |"
        lines += [f"## {name} ({s['two_class']} two-class of {s['videos']}; t/N ruler "
                  f"{_num(s['t_over_n'])})", "", head,
                  "| --- | --- | --- | --- |" if has_ego else "| --- | --- |"]
        for cand, row in s["rows"].items():
            cells = [_num(row["macro"])]
            if has_ego:
                cells += [_num(row.get("ego")), _num(row.get("non_ego"))]
            lines.append(f"| {cand} | " + " | ".join(cells) + " |")
        lines += ["", f"KP2 (rank avg - host, per video): {_delta(s['kp2_delta'])}"]
        if s["kp3"]:
            k = s["kp3"]
            lines.append(f"KP3 edge share of top-5 % vt1 steps: {k['share']:.3f} "
                         f"(chance {k['chance']:.3f}, {k['videos']} videos)")
        if s["kp4"]:
            k = s["kp4"]
            lines.append(f"KP4 vt2 peak - onset: median |lag| {k['median_abs_s']:.2f} s, "
                         f"{k['share_within']:.2f} of {k['clips']} clips within "
                         f"±{constants.KIM_PILOT_ONSET_S} s")
        lines.append("")
    lines += ["## Decisions (fixed in §5 before the pilot)", ""]
    for key, value in r["decisions"].items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--ckpt", type=Path, required=True, help="the host's checkpoint_last.pt")
    parser.add_argument("--name", required=True, help="the host arm, recorded in the read-out")
    parser.add_argument(
        "--set", nargs=4, action="append", required=True,
        metavar=("NAME", "KIND", "EVAL_DIR", "INPUT_DIR"),
        help="KIND native (DoTA protocol B) | rows; EVAL_DIR = this host's read of the set",
    )
    parser.add_argument("--native-stride", type=int, default=constants.V2_E1_STRIDE_BC)
    parser.add_argument("--step-seconds", nargs=2, action="append", metavar=("NAME", "SECONDS"))
    parser.add_argument("--metadata", type=Path, default=None, help="DoTA metadata_val.json")
    parser.add_argument("--split-file", type=Path, default=None, help="DoTA val_split.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
