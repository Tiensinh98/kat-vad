"""v2 P4 -- D6 / N11: temporal-shuffle control for the encoder E2(d) picked (printed only).

Addendum §2 D6 and §12 N11. The picked encoder is re-extracted on ``dota_cap_dev`` with the 16
frames of every window permuted by one fixed permutation (``dota_cap extract --shuffle-seed``),
so each window keeps its appearance content and loses its temporal order. The in-domain probe
of E2(d) (N3, same folds, same seed) is re-read on both caches:

* ``Δ(ordered - shuffled)`` for ``u`` alone, ``[x;u]`` (A2) / its CRN form (A3), and the same with
  position ``p`` appended. Near zero ⇒ the stream adds a second appearance encoder, not motion;
  this limits what the thesis may call motion. It never moves the E2(d) pick.
* ``Δ(shuffled - CLIP-only)``: what a shuffled stream still adds over ``x``.

DoTA-eval is sealed: only ``dota_cap_dev`` is loaded.

CLI::

    python -m core.tools.e2d_shuffle \\
        --dota-s1-dir cache/clip/DoTA_s1_ncc --metadata data/DoTA/metadata_val.json \\
        --split-file data/DoTA/val_split.txt --out-dir outputs/v2/REPORTS/v2_E2d_shuffle \\
        [--encoder vit_s_k710_dl_from_giant] [--shuffle-seed 2024]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_splits import load_split
from core.tools import dota_cap
from core.tools.e2d_probe import (
    ARMS,
    _fmt,
    _row,
    arm_sets,
    check_manifest,
    ci,
    diff,
    dota_cap_alignment_sha256,
    in_domain_aucs,
    load_dota,
)
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "shuffle_readout.json"
READOUT_MD = "shuffle_readout.md"
ORDERED = "ordered"
SHUFFLED = "shuffled"
STREAM_SETS = ("u", "with", "with_p")  # the sets that contain u


def shuffle_dirs(root: Path, encoder: str, seed: int) -> dict[str, Path]:
    """The ordered DoTA-CAP cache and its shuffle-control twin under ``root``."""
    ordered = root / encoder / f"{constants.DOTA_CAP_DATASET}_s1_squash"
    return {ORDERED: ordered, SHUFFLED: ordered.with_name(
        f"{ordered.name}_{constants.V2_D6_SHUFFLE_TAG}{seed}"
    )}


def run(args: argparse.Namespace) -> dict[str, Any]:
    parsed = parse_metadata(args.metadata, read_split_ids(args.split_file))
    records = {r.video_id: r for r in parsed}
    cap_ids = load_split(constants.V2_SPLIT_DOTA_CAP_DEV, args.split_dir)
    alignment = dota_cap_alignment_sha256(args.split_dir)
    dirs = shuffle_dirs(args.video_root, args.encoder, args.shuffle_seed)
    check_manifest(dirs[ORDERED], dota_cap.video_manifest(args.encoder, alignment))
    shuffled_manifest = dota_cap.video_manifest(args.encoder, alignment, args.shuffle_seed)
    check_manifest(dirs[SHUFFLED], shuffled_manifest)
    LOGGER.info("D6 gates passed: %d DoTA-CAP-dev clips, %s, frame order %s",
                len(cap_ids), args.encoder, shuffled_manifest["frame_order"])

    corpus = load_dota(cap_ids, records, args.dota_s1_dir, dirs, constants.V2_E2D_STRIDE)
    ids = corpus.two_class()
    groups = corpus.group
    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        sets = {side: arm_sets(corpus, side, arm) for side in (ORDERED, SHUFFLED)}
        base = in_domain_aucs(sets[ORDERED]["base"], corpus, ids, args.folds, args.seed)
        aucs = {
            side: {
                name: in_domain_aucs(sets[side][name], corpus, ids, args.folds, args.seed)
                for name in STREAM_SETS
            }
            for side in (ORDERED, SHUFFLED)
        }
        for side, block in aucs.items():
            for name, values in block.items():
                LOGGER.info("D6 %s %s %s: macro %.4f", arm, side, name,
                            float(np.mean(list(values.values()))))
        per_arm[arm] = {
            "base": ci(base, groups, args.resamples, args.seed),
            "macro": {
                side: {n: ci(a, groups, args.resamples, args.seed) for n, a in block.items()}
                for side, block in aucs.items()
            },
            "ordered_minus_shuffled": {
                n: ci(diff(aucs[ORDERED][n], aucs[SHUFFLED][n]), groups, args.resamples, args.seed)
                for n in STREAM_SETS
            },
            "shuffled_with_minus_base": ci(
                diff(aucs[SHUFFLED]["with"], base), groups, args.resamples, args.seed
            ),
        }
    readout = {
        "addendum": "core/docs/v2/PREREG_ADDENDUM.md §2 D6, §12 N11 (printed, never decided on)",
        "encoder": args.encoder,
        "shuffle_seed": args.shuffle_seed,
        "frame_order": shuffled_manifest["frame_order"],
        "stride": constants.V2_E2D_STRIDE,
        "dota_cap_dev": {"clips": len(cap_ids), "two_class": len(ids)},
        "folds": args.folds,
        "seed": args.seed,
        "bootstrap": args.resamples,
        "per_arm": per_arm,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(readout))
    LOGGER.info("D6 read-out -> %s", args.out_dir)
    return readout


def render_markdown(readout: dict[str, Any]) -> str:
    """Human read-out; the JSON is the record."""
    lines = [
        "# v2 D6: temporal-shuffle control on DoTA-CAP-dev (printed, never decided on)",
        "",
        f"Addendum §2 D6 / §12 N11. Encoder `{readout['encoder']}`, in-domain probe (N3), "
        f"`dota_cap_dev` {readout['dota_cap_dev']['clips']} clips "
        f"({readout['dota_cap_dev']['two_class']} two-class), stride {readout['stride']}. "
        f"Every window's 16 frames permuted by seed {readout['shuffle_seed']}: "
        f"`{readout['frame_order']}`. Cluster bootstrap over source videos, "
        f"B = {readout['bootstrap']}. Name it **DoTA-CAP (n/1397)** (D14).",
        "",
        _row(["Arm", "Set", "ordered", "shuffled", "**ordered - shuffled**"]),
        "|---|---|---|---|---|",
    ]
    for arm, block in readout["per_arm"].items():
        for name in STREAM_SETS:
            lines.append(_row([
                arm, name, _fmt(block["macro"][ORDERED][name]),
                _fmt(block["macro"][SHUFFLED][name]),
                _fmt(block["ordered_minus_shuffled"][name]),
            ]))
    lines += ["", _row(["Arm", "CLIP-only", "shuffled `with` - CLIP-only"]), "|---|---|---|"]
    for arm, block in readout["per_arm"].items():
        lines.append(_row([arm, _fmt(block["base"]), _fmt(block["shuffled_with_minus_base"])]))
    lines += [
        "",
        "Reading (fixed in Amendment 6 before this number): ordered - shuffled ≈ 0 on `with_p` "
        "⇒ the stream's gain is appearance, and the thesis may not call it motion.",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--dota-s1-dir", type=Path, required=True, help="cache/clip/DoTA_s1_ncc")
    parser.add_argument("--metadata", type=Path, required=True, help="DoTA metadata_val.json")
    parser.add_argument("--split-file", type=Path, required=True, help="DoTA val_split.txt")
    parser.add_argument("--video-root", type=Path, default=constants.VIDEO_CACHE_DIR)
    parser.add_argument("--encoder", default=constants.VIDEOMAE_ENCODER_S,
                        choices=sorted(constants.VIDEOMAE_ARCH))
    parser.add_argument("--shuffle-seed", type=int, default=constants.V2_D6_SHUFFLE_SEED)
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--folds", type=int, default=constants.EDA_PROBE_FOLDS)
    parser.add_argument("--seed", type=int, default=constants.V2_SPLIT_SEED)
    parser.add_argument("--resamples", type=int, default=constants.V2_E2D_BOOTSTRAP)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
