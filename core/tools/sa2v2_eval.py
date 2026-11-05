"""Dump one SA²-v2 checkpoint's per-step reads for the controls and the map read-out.

Rule ``core/docs/final/SA2V2_RUN.md``. Per video, on the same rows and verbalizer as the frame
read: the map ``q`` (``E_``), the step score (``s_``), the head on ``x^AS`` (``b_``) and on the
49-cell mean (``a_``), the top-3 ``q`` (``t_``), ``alpha``'s entropy (``h_``) and Mo (``M_``).
``E_ / s_ / n_`` are ``core.tools.v5_maps``'s dump format, so ``v5_maps read`` reads the file. With
``--baseline-maps`` the Mo, uniform and centre maps are written the same way, with the same scores.

CLI::

    python -m core.tools.sa2v2_eval --name SV2 --ckpt .../checkpoint_last.pt --input-dir bake \\
        --cell-dir cells --data-dir nexar_defs --ids-file ids.txt --stride 8 --out-dir eval/dada
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core import constants
from core.data.definitions import dataset_abbr, item_verbalizer
from core.data.stp_patches import EvalPatches, stp_eval_inputs
from core.device import resolve_device
from core.inference import make_class_feats_fn
from core.models.sa2 import attention_entropy
from core.tools.feature_cache import read_ids_file
from core.tools.kill_switch_probe import write_json_atomic
from core.tools.rate_matched_eval import Run, load_finished_model
from core.tools.v5_maps import FRAMES_KEY, MAP_KEY, SCORE_KEY, centre_prior
from core.train import load_class_names

LOGGER = logging.getLogger(__name__)

BRANCH_KEY, MEAN_KEY, TOP_KEY, ENTROPY_KEY, MO_KEY = "b_", "a_", "t_", "h_", "M_"
BASELINE_MAPS = ("Mo", "uniform", "centre")


@torch.no_grad()
def clip_reads(
    model: Any, rows: torch.Tensor, class_feats_fn: Any, max_len: int = constants.MAX_VIS_LEN
) -> dict[str, np.ndarray]:
    """One clip's reads, arrays of ``(K, ...)``, in windows of ``max_len`` as the frame read."""
    module = model.sa2v2
    device = next(model.parameters()).device
    parts: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(rows), max_len):
        window = rows[start : start + max_len].unsqueeze(0).to(device)
        out = model(
            window, torch.tensor([window.shape[1]], device=device), class_feats=class_feats_fn()
        )
        columns = window[0, :, -module.width :]
        cells = columns[:, : constants.SV2_CELL_WIDTH].reshape(
            -1, constants.SV2_CELLS, constants.SV2_DIM
        )
        change = columns[:, constants.SV2_CELL_WIDTH :].reshape(
            -1, constants.SV2_CELLS, constants.SV2_DIM
        )
        q = out["sv2_cell_logits"][0].sigmoid()
        reads = {
            MAP_KEY: q,
            SCORE_KEY: out["cls_bin_logits"][0].sigmoid(),
            BRANCH_KEY: module.branch_scores(out["sv2_x_as"][0]),
            MEAN_KEY: module.branch_scores(cells.mean(dim=1)),
            TOP_KEY: q.topk(constants.SV2_TOP_CELLS, dim=-1).values.mean(dim=-1),
            ENTROPY_KEY: torch.stack(
                [attention_entropy(a[None]) for a in out["sv2_attn"][0]]
            ),
            MO_KEY: change.norm(dim=-1),
        }
        for key, value in reads.items():
            parts.setdefault(key, []).append(value.float().cpu())
    return {key: torch.cat(value).numpy() for key, value in parts.items()}


def baseline_maps(reads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The maps a learned ``q`` is read beside: Mo, uniform, centre (no training)."""
    steps = len(reads[MO_KEY])
    return {
        "Mo": reads[MO_KEY],
        "uniform": np.full((steps, constants.SV2_CELLS), 1.0 / constants.SV2_CELLS, np.float32),
        "centre": np.tile(centre_prior().astype(np.float32), (steps, 1)),
    }


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def input_rows(
    input_dir: Path, video_id: str, stride: int, s1_dir: Path | None, patches: EvalPatches
) -> tuple[np.ndarray, int]:
    """A baked cache's rows + the video's columns, and its frame count (``n_`` of the dump)."""
    rows = np.ascontiguousarray(np.load(input_dir / f"{video_id}.npy"), dtype=np.float32)
    frames = len(rows)
    if s1_dir is not None:
        frames = int(np.load(s1_dir / f"{video_id}.npy", mmap_mode="r").shape[0])
        if len(rows) != len(range(0, frames, stride)):
            raise ValueError(f"{video_id}: {len(rows)} rows, s{stride} of {frames} frames")
    return patches.append(rows, video_id), frames


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    model, step, text_encode_fn, cfg = load_finished_model(Run(args.name, args.ckpt), device)
    if not cfg.stp.sa2v2:
        raise ValueError(f"{args.ckpt} has stp.sa2v2=false: it has no map to dump")
    patches = stp_eval_inputs(cfg.stp, args.cell_dir, args.stride)
    assert patches is not None  # nosec B101 - stp_eval_inputs refused a missing cache
    class_names = load_class_names(args.data_dir)
    ids = sorted(read_ids_file(args.ids_file))
    arrays: dict[str, np.ndarray] = {}
    extra: dict[str, dict[str, np.ndarray]] = {name: {} for name in BASELINE_MAPS}
    for video_id in ids:
        rows, frames = input_rows(args.input_dir, video_id, args.stride, args.s1_dir, patches)
        feats_fn = make_class_feats_fn(
            text_encode_fn,
            class_names,
            item_verbalizer(dataset_abbr(constants.DOTA_DATASET), video_id),
        )
        reads = clip_reads(model, torch.from_numpy(rows), feats_fn)
        arrays |= {f"{key}{video_id}": value for key, value in reads.items()}
        arrays[f"{FRAMES_KEY}{video_id}"] = np.array(frames)
        if args.baseline_maps:
            for name, grid in baseline_maps(reads).items():
                extra[name] |= {
                    f"{MAP_KEY}{video_id}": grid,
                    f"{SCORE_KEY}{video_id}": reads[SCORE_KEY],
                    f"{FRAMES_KEY}{video_id}": np.array(frames),
                }
    save_npz(args.out_dir / f"maps_{args.name}.npz", arrays)
    if args.baseline_maps:
        for name, data in extra.items():
            save_npz(args.out_dir / f"maps_{name}.npz", data)
    meta = {"name": args.name, "ckpt": str(args.ckpt), "global_step": step, "clips": len(ids),
            "stride": args.stride}
    write_json_atomic(args.out_dir / f"maps_{args.name}.json", meta)
    LOGGER.info("%s: %d videos -> %s", args.name, len(ids), args.out_dir)
    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True, help="baked host rows")
    parser.add_argument("--cell-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="training dir (defs.json)")
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--s1-dir", type=Path, default=None,
                        help="stride-1 cache: frame counts when rows are not one per frame")
    parser.add_argument("--baseline-maps", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
