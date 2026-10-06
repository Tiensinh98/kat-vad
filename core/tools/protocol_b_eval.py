"""v2: score any arm's checkpoints on DoTA at protocol B (E1), native frames, per clip.

E1 (``core.tools.rate_matched_eval``) fixed the v2 DoTA protocol: ``s1[::3]``, whole clip,
step ``t`` at native frame ``3t``, scores linearly interpolated to every native frame (J3),
seed-averaged per frame (J4), macro over two-class clips (J5) with a source-video cluster
bootstrap (D3). That tool reads the raw CLIP cache only. This one scores **the input the
checkpoint was trained on**:

* **A0** -- ``--input-dir`` is the plain stride-1 CLIP cache (``DoTA_s1_ncc``); rows are
  ``[::stride]``-subsampled here.
* **A1-A3** -- ``--input-dir`` is a baked DoTA cache from ``build_v2_inputs apply --stride 3``
  (CRN per clip over the stride-3 rows, motion scaled with the T2-train statistics). Its
  manifest must name the checkpoint's ``v2.crn`` / ``v2.motion`` (K4) and the same stride, and
  every clip must have exactly ``len(s1[::stride])`` rows.

The clip set is a frozen v2 split: ``dota_dev`` (CLIP-only arms, CRN and ``F``) or
``dota_cap_dev`` (every motion contrast, D13). Sealed splits are refused by ``load_split``.
Writes ``clip_aucs.json`` (seed-averaged per-clip AUC -- the input of paired Δs and of D15)
and ``clip_scores.npz`` (seed-averaged native scores + labels, E3's position reads, Amendment 9
M3/M6) beside a read-out. Name a ``dota_cap_dev`` number **DoTA-CAP (n/1397)** (D14).

CLI::

    python -m core.tools.protocol_b_eval \\
        --run s2099 runs/A1_s2099/checkpoint_last.pt \\
        --input-dir cache/v2/A1_R2/DoTA_s3_from_s1 --s1-dir cache/clip/DoTA_s1_ncc \\
        --metadata data/DoTA/metadata_val.json --split-file data/DoTA/val_split.txt \\
        --data-dir data/DoTA/labels_s8 --split dota_dev --out-dir outputs/v2/REPORTS/pb_A1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core import constants
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_inputs import check_input_manifest, read_input_manifest
from core.data.v2_splits import dota_group, load_split
from core.device import resolve_device
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.rate_matched_eval import (
    Protocol,
    Run,
    load_finished_model,
    native_labels,
    score_steps,
    seed_average,
    to_native,
)
from core.train import load_class_names

LOGGER = logging.getLogger(__name__)

CLIP_AUCS_JSON = "clip_aucs.json"
CLIP_SCORES_NPZ = "clip_scores.npz"
_SCORE_KEY, _LABEL_KEY = "score/", "label/"
READOUT_JSON = "protocol_b_readout.json"
READOUT_MD = "protocol_b_readout.md"
OPEN_SPLITS = (constants.V2_SPLIT_DOTA_DEV, constants.V2_SPLIT_DOTA_CAP_DEV)


def input_rows(
    input_dir: Path, video_id: str, native_frames: int, stride: int, baked: bool
) -> np.ndarray:
    """The ``(ceil(N / stride), d)`` rows the model reads for one clip.

    A plain cache is the stride-1 CLIP cache, subsampled here; a baked cache already holds
    the stride-``stride`` rows. Either way the row count must be ``len(range(0, N, stride))``.
    """
    rows = np.load(input_dir / f"{video_id}.npy")
    expected = len(range(0, native_frames, stride))
    if not baked:
        if len(rows) != native_frames:
            raise ValueError(f"{video_id}: plain cache has {len(rows)} rows, s1 {native_frames}")
        rows = rows[::stride]
    if len(rows) != expected:
        raise ValueError(f"{video_id}: {len(rows)} input rows, protocol needs {expected}")
    return np.ascontiguousarray(rows, dtype=np.float32)


def check_baked_stride(input_dir: Path, stride: int) -> bool:
    """True for a baked cache built at ``stride``; False for a plain CLIP cache."""
    manifest = read_input_manifest(input_dir)
    if manifest is None:
        return False
    built = manifest.get("stride_over_clip_dir")
    if built != stride:
        raise ValueError(
            f"{input_dir} was baked at stride {built!r}; protocol B here is stride {stride} "
            "(bake it with build_v2_inputs apply --stride)"
        )
    return True


def write_clip_scores(
    path: Path, scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> None:
    """Seed-averaged native scores and labels of every clip, one ``.npz`` (atomic)."""
    if set(scores) != set(labels):
        raise ValueError("scores and labels must cover the same clips")
    arrays: dict[str, np.ndarray] = {
        f"{_SCORE_KEY}{v}": np.asarray(scores[v], dtype=np.float64) for v in scores
    }
    arrays.update({f"{_LABEL_KEY}{v}": np.asarray(labels[v], dtype=np.int64) for v in labels})
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def read_clip_scores(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Inverse of :func:`write_clip_scores`: ``(scores, labels)`` keyed by clip id."""
    with np.load(path) as data:
        scores = {k[len(_SCORE_KEY):]: data[k] for k in data.files if k.startswith(_SCORE_KEY)}
        labels = {k[len(_LABEL_KEY):]: data[k] for k in data.files if k.startswith(_LABEL_KEY)}
    if set(scores) != set(labels):
        raise ValueError(f"{path}: scores and labels cover different clips")
    return scores, labels


def clip_aucs(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-clip native-frame AUC over the two-class clips (J5)."""
    return {
        v: frame_auc(scores[v], labels[v])
        for v in sorted(scores)
        if 0 < int(labels[v].sum()) < len(labels[v])
    }


def score_checkpoint(
    run: Run,
    ids: list[str],
    native_frames: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], int, dict[str, str]]:
    """Native-frame scores of one checkpoint on ``ids``, its step and its v2 arm."""
    model, step, text_encode_fn, cfg = load_finished_model(run, device)
    check_input_manifest(args.input_dir, cfg.v2.crn, cfg.v2.motion)
    baked = check_baked_stride(args.input_dir, args.stride)
    class_names = load_class_names(args.data_dir)
    protocol = Protocol(args.stride, None)
    native: dict[str, np.ndarray] = {}
    for video_id in ids:
        rows = input_rows(args.input_dir, video_id, native_frames[video_id], args.stride, baked)
        steps = score_steps(
            model, torch.from_numpy(rows), text_encode_fn, class_names, video_id, protocol
        )
        native[video_id] = to_native(steps, args.stride, native_frames[video_id])
    LOGGER.info("%s (step %d, crn=%s motion=%s): scored %d clips",
                run.name, step, cfg.v2.crn, cfg.v2.motion, len(ids))
    return native, step, {"crn": cfg.v2.crn, "motion": cfg.v2.motion}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.split not in OPEN_SPLITS:
        raise ValueError(f"protocol_b_eval scores {OPEN_SPLITS} only, not {args.split!r}")
    runs = [Run(name, Path(ckpt)) for name, ckpt in args.run]
    device = resolve_device(args.device)
    ids = load_split(args.split, args.split_dir)
    parsed = parse_metadata(args.metadata, read_split_ids(args.split_file))
    records = {r.video_id: r for r in parsed}
    native_frames = {
        v: len(np.load(args.s1_dir / f"{v}.npy", mmap_mode="r")) for v in ids
    }
    labels = {v: native_labels(records[v], native_frames[v]) for v in ids}

    per_run: dict[str, dict[str, np.ndarray]] = {}
    steps: dict[str, int] = {}
    arms: dict[str, dict[str, str]] = {}
    for r in runs:
        per_run[r.name], steps[r.name], arms[r.name] = score_checkpoint(
            r, ids, native_frames, args, device
        )
    if len({json.dumps(a, sort_keys=True) for a in arms.values()}) != 1:
        raise ValueError(f"checkpoints of different arms cannot be seed-averaged: {arms}")

    averaged = seed_average([per_run[name] for name in sorted(per_run)])
    aucs = clip_aucs(averaged, labels)
    groups = {v: dota_group(v) for v in ids}

    def ci(values: dict[str, float]) -> dict[str, float] | None:
        return cluster_bootstrap_ci(
            values, groups, constants.V2_E1_BOOTSTRAP, args.seed, constants.V2_E1_CI
        )

    ruler = {v: np.arange(n) / n for v, n in native_frames.items()}
    readout: dict[str, Any] = {
        "protocol": f"B: s1[::{args.stride}], whole clip, native frames (addendum §8.1 J1-J5)",
        "split": args.split,
        "clips": len(ids),
        "two_class": len(aucs),
        "input_dir": str(args.input_dir),
        "arm": next(iter(arms.values())),
        "runs": sorted(per_run),
        "checkpoints": {r.name: str(r.ckpt) for r in runs},
        "global_steps": steps,
        "macro": ci(aucs),
        "per_run_macro": {
            name: float(np.mean(list(clip_aucs(scores, labels).values())))
            for name, scores in per_run.items()
        },
        "position_ruler": ci(clip_aucs(ruler, labels)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / CLIP_AUCS_JSON, aucs)
    write_clip_scores(args.out_dir / CLIP_SCORES_NPZ, averaged, labels)
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(readout))
    LOGGER.info("protocol B macro on %s: %s -> %s", args.split, readout["macro"], args.out_dir)
    return readout


def _fmt(c: dict[str, float] | None) -> str:
    return "—" if c is None else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def render_markdown(readout: dict[str, Any]) -> str:
    name = "DoTA-CAP (n/1397)" if readout["split"] == constants.V2_SPLIT_DOTA_CAP_DEV else "DoTA"
    lines = [
        f"# Protocol B on `{readout['split']}` — {name}",
        "",
        f"{readout['protocol']}. Arm: crn={readout['arm']['crn']} "
        f"motion={readout['arm']['motion']}. Clips {readout['clips']} "
        f"({readout['two_class']} two-class). Input `{readout['input_dir']}`.",
        "",
        f"**Macro (seed-averaged, 95 % cluster CI): {_fmt(readout['macro'])}**",
        "",
        f"Position ruler `t/N` (no pixels): {_fmt(readout['position_ruler'])}",
        "",
        "| run | global_step | macro |",
        "|---|---|---|",
        *(
            f"| {r} | {readout['global_steps'][r]} | {readout['per_run_macro'][r]:.4f} |"
            for r in readout["runs"]
        ),
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--run", nargs=2, action="append", required=True, metavar=("NAME", "CKPT"),
        help="finished checkpoint, metrics.jsonl beside it (repeat per seed; one arm only)",
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="DoTA_s1_ncc (A0) or a baked `apply --stride` DoTA cache (A1-A3)")
    parser.add_argument("--s1-dir", type=Path, required=True,
                        help="cache/clip/DoTA_s1_ncc: native frame count per clip")
    parser.add_argument("--metadata", type=Path, required=True, help="metadata_val.json")
    parser.add_argument("--split-file", type=Path, required=True, help="val_split.txt")
    parser.add_argument("--data-dir", type=Path, required=True, help="DoTA labels dir (defs.json)")
    parser.add_argument("--split", choices=OPEN_SPLITS, default=constants.V2_SPLIT_DOTA_DEV)
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--stride", type=int, default=constants.V2_E1_STRIDE_BC)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=constants.SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
