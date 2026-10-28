"""v5 maps: dump a row's spatial maps, read them against DoTA boxes, port STAUC (PREREG_V5 S2-8).

Map per step (S2-8): R5 / R5Δ ``E_t(j) = sum_m s_{t,m} a_{t,m,j}``; R3 ``a_{t,1,j}`` (no sMIL head).
R1 has no map (its readout is uniform; hit - chance = 0 by construction).

``dump`` (Colab): one finished checkpoint on DoTA-CAP-dev at protocol B (its A2 rows + the clip's
V7 grid, Stage 0's cache) -> per clip ``E (K, 49)``, the step score and the native frame count.
``--split dota_cap_eval --final`` dumps B13's decision set (§2.4 B13); the sealed split needs both.

``read`` -- S2-8: per seed the Stage 0 pointing read (:func:`core.tools.v5_stage0.clip_read`, map
in place of ``Δ``), hit rates averaged over seeds per clip; then D3 hit - chance, Amendment 1's
lift - lift_pre, and hit(R) - hit(centre prior), each a 95 % cluster bootstrap over DoTA source
videos. **"R's maps point at the agent"** iff all three lows > 0. Printed: by ego and box-share
bin, STAUC (seed-averaged map and per-clip min-max score, §2.4 B3).

``g5`` -- **S1-G5**: (a) the read fed Stage 0's stored V7 change maps reproduces
``RESULTS_V5_STAGE0.md`` §3's V7 D3 row (|Δ| <= 1e-4); (b) random maps fail, planted in-box maps
pass (lesson (bt)); (c) the STAUC port matches DoTA's official ``stauc.py`` on a fixed synthetic
input (|Δ| <= 1e-6).

STAUC (DoTA, ``MoonBlvd/Detection-of-Traffic-Anomaly/stauc.py``, ``method_type='frame'``): each
abnormal frame's true-positive weight is its TARR -- the share of the top-K map mass inside the
boxes, K = the boxes' pixel count at map resolution -- and ``STAUC = AUC(FPR, sum TARR-weighted
TP / #positives)``. Ported verbatim (torch ``topk`` included, so ties break identically).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from core import constants
from core.data.definitions import dataset_abbr, item_verbalizer
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_splits import load_split
from core.data.v5_grids import eval_grid, v5_arm
from core.device import resolve_device
from core.inference import make_class_feats_fn
from core.tools import v5_stage0 as s0
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import check_baked_stride, check_split, input_rows
from core.tools.rate_matched_eval import Run, load_finished_model, native_labels
from core.train import load_class_names

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/v5/PREREG_V5.md S2-8 + S1-G5"
MAP_KEY, SCORE_KEY, FRAMES_KEY = "E_", "s_", "n_"
TOPK = constants.V5_TOPK[constants.V5_GRID_COARSE]
TARR_EPS = 1e-6  # stauc.py: values.sum() + 1e-6


# --------------------------------------------------------------------------- STAUC (port)


def tarr_frame(
    difference_map: np.ndarray,
    label: int,
    bboxes: list[list[float]],
    image_size: tuple[int, int] = constants.V5_BOX_FRAME_SIZE,
) -> float:
    """``stauc.get_tarr(..., method_type='frame')``: TARR of one frame's ``(H, W)`` map."""
    if label == 0:
        return 0.0
    if len(bboxes) == 0:
        return 1.0
    dmap = torch.as_tensor(np.asarray(difference_map), dtype=torch.float32)
    height, width = dmap.shape
    boxes = torch.tensor(copy.deepcopy(bboxes), dtype=torch.float32)
    if boxes.max() > 1:
        boxes[:, [0, 2]] /= image_size[0]
        boxes[:, [1, 3]] /= image_size[1]
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    boxes = boxes.type(torch.long)
    mask = torch.zeros([height, width])
    for box in boxes:
        mask[box[1] : box[3], box[0] : box[2]] = 1
    k = int(mask.sum())
    values, indices = torch.topk(dmap.view(-1), k=k)
    inside = mask[indices // width, indices % width] == 1
    return float(values[inside].sum() / (values.sum() + TARR_EPS))


def _trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """``sklearn.metrics.auc`` for an increasing ``x``."""
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def stauc(labels: np.ndarray, scores: np.ndarray, tarrs: np.ndarray) -> dict[str, float]:
    """``STAUCMetrics.get_stauc``: spatio-temporal AUC and the plain AUC on the same curve."""
    y_true = np.asarray(labels) == 1
    y_score = np.asarray(scores, dtype=np.float64)
    weight = np.asarray(tarrs, dtype=np.float64)
    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_score, y_true, weight = y_score[order], y_true[order], weight[order]
    distinct = np.where(np.diff(y_score))[0]
    thresholds = np.r_[distinct, y_true.size - 1]
    tps = np.cumsum(y_true * weight, dtype=np.float64)[thresholds]
    positives = np.cumsum(y_true, dtype=np.float64)[thresholds]
    fps = np.cumsum(1 - y_true, dtype=np.float64)[thresholds]
    if len(fps) > 2:
        keep = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps, tps, positives = fps[keep], tps[keep], positives[keep]
    tps, fps, positives = np.r_[0, tps], np.r_[0, fps], np.r_[0, positives]
    fpr = fps / fps[-1]
    return {
        "stauc": _trapezoid(fpr, tps / positives[-1]),
        "auc": _trapezoid(fpr, positives / positives[-1]),
    }


def upsample(grid_map: np.ndarray, size: int = constants.V5_STAUC_MAP_SIZE) -> np.ndarray:
    """``(49,)`` tile map -> ``(size, size)`` bilinear (§2.4 B3)."""
    side = math.isqrt(grid_map.shape[-1])
    tile = torch.as_tensor(grid_map, dtype=torch.float32).view(1, 1, side, side)
    out: np.ndarray = F.interpolate(tile, size=(size, size), mode="bilinear", align_corners=False)[
        0, 0
    ].numpy()
    return out


# --------------------------------------------------------------------------- dump


def step_maps(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """``(B, L, 49)`` map: ``sum_m s a`` with an sMIL head, else the single query's attention."""
    attn = outputs["v5_attn"]
    if "v5_smil" in outputs:
        return torch.einsum("blm,blmt->blt", outputs["v5_smil"], attn)
    if attn.shape[2] != 1:
        raise ValueError("a multi-query readout without sMIL scores has no defined map")
    return attn[:, :, 0]


@torch.no_grad()
def clip_maps(
    model: Any, rows: torch.Tensor, class_feats_fn: Any, max_len: int = constants.MAX_VIS_LEN
) -> tuple[np.ndarray, np.ndarray]:
    """One clip's ``(K, 49)`` maps and ``(K,)`` sigmoid step scores (windows of ``max_len``)."""
    device = next(model.parameters()).device
    maps, scores = [], []
    for start in range(0, len(rows), max_len):
        window = rows[start : start + max_len].unsqueeze(0).to(device)
        out = model(
            window, torch.tensor([window.shape[1]], device=device), class_feats=class_feats_fn()
        )
        maps.append(step_maps(out)[0].float().cpu())
        scores.append(out["cls_bin_logits"][0].sigmoid().float().cpu())
    return torch.cat(maps).numpy(), torch.cat(scores).numpy()


def run_dump(args: argparse.Namespace) -> dict[str, Any]:
    check_split(args.split, args.final, "v5_maps dump")
    run = Run(args.name, args.ckpt)
    device = resolve_device(args.device)
    model, step, text_encode_fn, cfg = load_finished_model(run, device)
    if cfg.v5.readout != constants.V5_READOUT_MQR:
        raise ValueError(f"{run.name}: v5.readout={cfg.v5.readout!r} has no map (R1 is uniform)")
    grid = eval_grid(cfg.v5.readout, cfg.v5.keys, args.grid_dir, args.stride)
    assert grid is not None  # nosec B101 - mqr
    baked = check_baked_stride(args.input_dir, args.stride)
    class_names = load_class_names(args.data_dir)
    ids = sorted(load_split(args.split, args.split_dir, final=args.final))
    arrays: dict[str, np.ndarray] = {}
    for video_id in ids:
        frames = len(np.load(args.s1_dir / f"{video_id}.npy", mmap_mode="r"))
        rows = grid.append(
            input_rows(args.input_dir, video_id, frames, args.stride, baked), video_id, 0, frames
        )
        feats_fn = make_class_feats_fn(
            text_encode_fn,
            class_names,
            item_verbalizer(dataset_abbr(constants.DOTA_DATASET), video_id),
        )
        maps, scores = clip_maps(model, torch.from_numpy(rows), feats_fn)
        arrays |= {
            f"{MAP_KEY}{video_id}": maps.astype(np.float32),
            f"{SCORE_KEY}{video_id}": scores.astype(np.float32),
            f"{FRAMES_KEY}{video_id}": np.array(frames),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(args.out)
    meta = {
        "run": run.name,
        "ckpt": str(run.ckpt),
        "global_step": step,
        "clips": len(ids),
        "split": args.split,
        "stride": args.stride,
        **v5_arm(cfg.v5),
    }
    write_json_atomic(args.out.with_suffix(".json"), meta)
    LOGGER.info("%s: maps of %d clips -> %s", run.name, len(ids), args.out)
    return meta


def load_dump(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """``{clip: {"E", "s", "n"}}`` of one dump file."""
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path) as data:
        for key in data.files:
            if key.startswith(MAP_KEY):
                v = key[len(MAP_KEY) :]
                out[v] = {
                    "E": data[key],
                    "s": data[f"{SCORE_KEY}{v}"],
                    "n": data[f"{FRAMES_KEY}{v}"],
                }
    return out


# --------------------------------------------------------------------------- read


def centre_prior(grid: int = constants.V5_GRID_COARSE) -> np.ndarray:
    """Isotropic Gaussian at the frame centre, sigma = 1/4 of the grid width (S2-8), ``(g*g,)``."""
    sigma = constants.V5_CENTRE_SIGMA_FRACTION * grid
    centre = (grid - 1) / 2.0
    yy, xx = np.mgrid[0:grid, 0:grid]
    prior: np.ndarray = np.exp(-((xx - centre) ** 2 + (yy - centre) ** 2) / (2 * sigma**2)).reshape(
        -1
    )
    return prior


def clip_point_reads(
    maps: list[np.ndarray],
    steps: np.ndarray,
    labels: np.ndarray,
    boxes: dict[int, list[list[float]]],
) -> dict[str, float] | None:
    """One clip: seed-averaged hit / hit_pre, chance / chance_pre, the centre prior's hit."""
    reads = [s0.clip_read(m, np.zeros(len(m)), steps, labels, boxes, TOPK) for m in maps]
    first = reads[0]
    if first.hit is None or first.chance is None:
        return None
    centre = s0.clip_read(
        np.tile(centre_prior(), (len(steps), 1)), np.zeros(len(steps)), steps, labels, boxes, TOPK
    )
    out = {
        "hit": float(np.mean([r.hit for r in reads if r.hit is not None])),
        "chance": first.chance,
        "hit_centre": float(centre.hit or 0.0),
    }
    if first.hit_pre is not None and first.chance_pre is not None:
        out |= {
            "hit_pre": float(np.mean([r.hit_pre for r in reads if r.hit_pre is not None])),
            "chance_pre": first.chance_pre,
        }
    return out


def point_summary(
    per_clip: dict[str, dict[str, float]], ego: dict[str, bool], bins: dict[str, str | None]
) -> dict[str, Any]:
    """S2-8's three intervals, the verdict and the printed splits."""
    d3 = {v: r["hit"] - r["chance"] for v, r in per_clip.items()}
    am1 = {
        v: (r["hit"] - r["chance"]) - (r["hit_pre"] - r["chance_pre"])
        for v, r in per_clip.items()
        if "hit_pre" in r
    }
    centre = {v: r["hit"] - r["hit_centre"] for v, r in per_clip.items()}
    stats = {
        "hit_minus_chance": s0.ci(d3),
        "hit_minus_pre": s0.ci(am1),
        "hit_minus_centre": s0.ci(centre),
    }
    points = all(st is not None and st["low"] > 0 for st in stats.values())
    return {
        **stats,
        "points": points,
        "by_ego": {
            lab: s0.ci({v: x for v, x in d3.items() if ego.get(v) == flag})
            for flag, lab in ((True, "ego"), (False, "non_ego"))
        },
        "by_bin": {
            b: s0.ci({v: x for v, x in d3.items() if bins.get(v) == b}) for b in s0.BIN_NAMES
        },
    }


def dev_meta(
    clips: dict[str, int], metadata: Path, split_file: Path, annotations_dir: Path
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[int, list[list[float]]]],
    dict[str, bool],
    dict[str, str | None],
]:
    """Labels, boxes, ego flag and box-share bin of each clip (``clips`` = native frame counts)."""
    records = {r.video_id: r for r in parse_metadata(metadata, read_split_ids(split_file))}
    labels, boxes, ego, bins = {}, {}, {}, {}
    for v, frames in clips.items():
        path = annotations_dir / f"{v}.json"
        if not path.is_file():
            continue
        labels[v] = native_labels(records[v], frames)
        boxes[v], ego[v] = s0.load_boxes(path)
        bins[v] = s0.share_bin(s0.box_share(boxes[v], labels[v]))
    return labels, boxes, ego, bins


def minmax(x: np.ndarray) -> np.ndarray:
    span = float(x.max() - x.min())
    out: np.ndarray = (x - x.min()) / span if span > 0 else np.zeros_like(x, dtype=np.float64)
    return out


def stauc_of(
    maps: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    boxes: dict[str, dict[int, list[list[float]]]],
    stride: int,
) -> dict[str, float]:
    """STAUC over every step of the boxed clips (per-clip min-max step scores)."""
    lab, sc, tr = [], [], []
    for v in sorted(maps):
        if v not in labels:
            continue
        s = minmax(scores[v])
        for k, frame in enumerate(range(0, len(labels[v]), stride)):
            y = int(labels[v][frame])
            lab.append(y)
            sc.append(float(s[k]))
            tr.append(tarr_frame(upsample(maps[v][k]), y, boxes[v].get(frame, [])))
    return stauc(np.array(lab), np.array(sc), np.array(tr))


def run_read(args: argparse.Namespace) -> dict[str, Any]:
    by_row: dict[str, dict[int, Path]] = {}
    for row, seed, path in args.run:
        by_row.setdefault(row, {})[int(seed)] = Path(path)
    out: dict[str, Any] = {"rule": RULE, "rows": {}}
    for row, seeds in sorted(by_row.items()):
        dumps = {s: load_dump(p) for s, p in sorted(seeds.items())}
        clips = {v: int(d["n"]) for v, d in next(iter(dumps.values())).items()}
        labels, boxes, ego, bins = dev_meta(
            clips, args.metadata, args.split_file, args.annotations_dir
        )
        per_clip: dict[str, dict[str, float]] = {}
        for v in labels:
            steps = np.arange(0, clips[v], args.stride)
            read = clip_point_reads([d[v]["E"] for d in dumps.values()], steps, labels[v], boxes[v])
            if read is not None:
                per_clip[v] = read
        mean_map = {v: np.mean([d[v]["E"] for d in dumps.values()], axis=0) for v in labels}
        mean_score = {v: np.mean([d[v]["s"] for d in dumps.values()], axis=0) for v in labels}
        out["rows"][row] = {
            "seeds": sorted(seeds),
            "clips_read": len(per_clip),
            **point_summary(per_clip, ego, bins),
            "stauc": stauc_of(mean_map, mean_score, labels, boxes, args.stride),
        }
    out["printed_beside"] = {
        "stage0_v7_change_lag6_hit_minus_chance": 0.076,
        "uniform_map_r1_hit_minus_chance": 0.0,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / "maps_readout.json", out)
    write_text_atomic(args.out_dir / "maps_readout.md", render_read(out))
    return out


def _f(st: dict[str, float] | None) -> str:
    return (
        "—"
        if st is None
        else f"{st['mean']:+.3f} [{st['low']:+.3f}, {st['high']:+.3f}] ({st['clips']})"
    )


def render_read(out: dict[str, Any]) -> str:
    lines = [
        "# v5 S2-8 maps read-out",
        "",
        f"Rule: {out['rule']}",
        "",
        "| row | D3 hit - chance | lift - lift_pre (Am. 1) | hit - centre prior | points? "
        "| STAUC | AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for row, r in out["rows"].items():
        lines.append(
            f"| {row} | {_f(r['hit_minus_chance'])} | {_f(r['hit_minus_pre'])} | "
            f"{_f(r['hit_minus_centre'])} | {'**yes**' if r['points'] else 'no'} | "
            f"{r['stauc']['stauc']:.4f} | {r['stauc']['auc']:.4f} |"
        )
    lines += [
        "",
        "Printed beside: Stage 0 V7 change at lag 6 hit - chance +0.076; uniform map (R1) = 0.",
        "",
    ]
    for row, r in out["rows"].items():
        lines.append(
            f"- {row} by ego: "
            + ", ".join(f"{k} {_f(v)}" for k, v in r["by_ego"].items())
            + "; by bin: "
            + ", ".join(f"{k} {_f(v)}" for k, v in r["by_bin"].items())
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- S1-G5


def g5a(inputs: s0.DevInputs, reference: dict[str, Any]) -> dict[str, Any]:
    """The read on Stage 0's stored V7 change maps vs the recorded V7 D3 row."""
    per_clip: dict[str, dict[str, float]] = {}
    for v, m in inputs.maps.items():
        read = clip_point_reads([m["d7"]], m["steps"], inputs.labels[v], inputs.boxes[v])
        if read is not None:
            per_clip[v] = read
    got = point_summary(per_clip, inputs.ego, inputs.bins)
    ref = reference["summary"]["V7"]
    diffs = {k: abs(got[k]["mean"] - ref[k]["mean"]) for k in ("hit_minus_chance", "hit_minus_pre")}
    return {
        "diffs": diffs,
        "got": {k: got[k] for k in diffs},
        "pass": max(diffs.values()) <= constants.V5_G5_MAX_ABS,
    }


def g5b(inputs: s0.DevInputs, seed: int = constants.V5_BOOTSTRAP_SEED) -> dict[str, Any]:
    """Random maps must not point; maps with a planted in-box signal must (lesson (bt))."""
    rng = np.random.default_rng(seed)
    grid = constants.V5_GRID_COARSE
    reads: dict[str, dict[str, dict[str, float]]] = {"random": {}, "planted": {}}
    for v, m in inputs.maps.items():
        steps = m["steps"]
        noise = rng.random((len(steps), grid * grid))
        planted = noise.copy()
        for k, frame in enumerate(steps):
            if inputs.labels[v][frame] and inputs.boxes[v].get(int(frame)):
                planted[k] += (s0.tile_weights(inputs.boxes[v][int(frame)], grid) > 0).astype(float)
        for name, mp in (("random", noise), ("planted", planted)):
            read = clip_point_reads([mp], steps, inputs.labels[v], inputs.boxes[v])
            if read is not None:
                reads[name][v] = read
    random_points = point_summary(reads["random"], inputs.ego, inputs.bins)["points"]
    planted_points = point_summary(reads["planted"], inputs.ego, inputs.bins)["points"]
    return {
        "random_points": random_points,
        "planted_points": planted_points,
        "pass": (not random_points) and planted_points,
    }


def load_official(path: Path) -> ModuleType:
    """DoTA's ``stauc.py`` as a module (needs torch + scikit-learn)."""
    spec = importlib.util.spec_from_file_location("dota_official_stauc", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_case(seed: int = constants.V5_BOOTSTRAP_SEED, frames: int = 80) -> dict[str, Any]:
    """Fixed input for S1-G5(c): labels, boxes (some frames empty), maps, tied scores."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(frames) < 0.4).astype(int)
    labels[:2] = [0, 1]
    boxes: list[list[list[float]]] = []
    for y in labels:
        count = int(rng.integers(0, 3)) if y else 0
        frame_boxes = []
        for _ in range(count):
            x1, y1 = rng.uniform(0, 1100), rng.uniform(0, 600)
            frame_boxes.append([x1, y1, x1 + rng.uniform(20, 180), y1 + rng.uniform(20, 120)])
        boxes.append(frame_boxes)
    maps = rng.random((frames, constants.V5_STAUC_MAP_SIZE, constants.V5_STAUC_MAP_SIZE)).astype(
        np.float32
    )
    scores = np.round(rng.random(frames), 2)
    return {"labels": labels, "boxes": boxes, "maps": maps, "scores": scores}


def g5c(official_path: Path) -> dict[str, Any]:
    """The port vs the official code on :func:`synthetic_case`."""
    case = synthetic_case()
    official = load_official(official_path)
    ref_tarrs = [
        float(official.get_tarr(case["maps"][i], int(y), case["boxes"][i], method_type="frame")[0])
        for i, y in enumerate(case["labels"])
    ]
    metrics = official.STAUCMetrics()
    metrics.labels, metrics.scores = np.array(case["labels"]), np.array(case["scores"])
    metrics.tarrs = np.array(ref_tarrs)
    ref_stauc, ref_auc, _ = metrics.get_stauc()
    tarrs = [
        tarr_frame(case["maps"][i], int(y), case["boxes"][i]) for i, y in enumerate(case["labels"])
    ]
    port = stauc(case["labels"], case["scores"], np.array(tarrs))
    diffs = {
        "tarr": float(np.max(np.abs(np.array(tarrs) - np.array(ref_tarrs)))),
        "stauc": abs(port["stauc"] - float(ref_stauc)),
        "auc": abs(port["auc"] - float(ref_auc)),
    }
    return {
        "official": {"stauc": float(ref_stauc), "auc": float(ref_auc)},
        "port": port,
        "diffs": diffs,
        "pass": max(diffs.values()) <= constants.V5_G5C_MAX_ABS,
    }


def run_g5(args: argparse.Namespace) -> dict[str, Any]:
    inputs = s0.load_dev(args)
    reference = json.loads(args.stage0_readout.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "rule": "S1-G5",
        "a": g5a(inputs, reference),
        "b": g5b(inputs),
        "c": g5c(args.official),
    }
    payload["pass"] = all(payload[k]["pass"] for k in "abc")
    md = (
        f"# S1-G5 map tools: **{'PASS' if payload['pass'] else 'FAIL'}**\n\n"
        f"- (a) Stage 0 V7 D3 row reproduced: max |Δ| {payload['a']['diffs']} "
        f"(≤ {constants.V5_G5_MAX_ABS})\n"
        f"- (b) random maps point: {payload['b']['random_points']} (must be False); "
        "planted maps point: "
        f"{payload['b']['planted_points']} (must be True)\n"
        f"- (c) STAUC port vs official: {payload['c']['diffs']} (≤ {constants.V5_G5C_MAX_ABS}); "
        f"official {payload['c']['official']}\n"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / "g5_maps.json", payload)
    write_text_atomic(args.out_dir / "g5_maps.md", md)
    return payload


# --------------------------------------------------------------------------- CLI


def _dota_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--metadata", type=Path, required=True, help="data/DoTA/metadata_val.json")
    p.add_argument("--split-file", type=Path, required=True, help="data/DoTA/val_split.txt")
    p.add_argument(
        "--annotations-dir", type=Path, required=True, help="data/DoTA/raw_annotations/annotations"
    )
    p.add_argument("--out-dir", type=Path, required=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    dump = sub.add_parser("dump", help="one checkpoint's maps on DoTA-CAP-dev (GPU)")
    dump.add_argument("--name", required=True)
    dump.add_argument("--ckpt", type=Path, required=True)
    dump.add_argument("--input-dir", type=Path, required=True, help="A2 bake of DoTA-CAP-dev at s3")
    dump.add_argument("--grid-dir", type=Path, required=True, help="Stage 0's DoTA-CAP V7 cache")
    dump.add_argument("--s1-dir", type=Path, required=True, help="cache/clip/DoTA_s1_ncc")
    dump.add_argument("--data-dir", type=Path, required=True, help="DoTA labels dir (defs.json)")
    dump.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    dump.add_argument("--stride", type=int, default=constants.V5_STAGE0_STRIDE)
    dump.add_argument(
        "--split",
        choices=[constants.V2_SPLIT_DOTA_CAP_DEV, constants.V2_SPLIT_DOTA_CAP_EVAL],
        default=constants.V2_SPLIT_DOTA_CAP_DEV,
    )
    dump.add_argument("--final", action="store_true", help="required for dota_cap_eval (B13)")
    dump.add_argument("--out", type=Path, required=True, help="<...>/maps_<row>_s<seed>.npz")
    dump.add_argument("--device", default="auto")

    read = sub.add_parser("read", help="S2-8 pointing reads + STAUC")
    read.add_argument(
        "--run", nargs=3, action="append", required=True, metavar=("ROW", "SEED", "NPZ")
    )
    read.add_argument("--stride", type=int, default=constants.V5_STAGE0_STRIDE)
    _dota_args(read)

    g5 = sub.add_parser("g5", help="S1-G5 (a)(b)(c)")
    g5.add_argument("--reports", type=Path, nargs="+", required=True, help="Stage 0 extract_*.json")
    g5.add_argument("--maps-dir", type=Path, required=True, help="Stage 0 maps/")
    g5.add_argument(
        "--stage0-readout", type=Path, required=True, help="REPORTS/stage0_readout.json"
    )
    g5.add_argument("--official", type=Path, required=True, help="DoTA's stauc.py")
    _dota_args(g5)
    return parser


COMMANDS = {"dump": run_dump, "read": run_read, "g5": run_g5}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
