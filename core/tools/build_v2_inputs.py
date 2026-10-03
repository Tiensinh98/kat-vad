"""Bake a v2 input cache: CRN and/or the fixed motion scaling (architecture §4-§5, §11).

Two subcommands, both writing one ``{id}.npy`` of float32 rows per source video or
clip, plus :data:`constants.V2_INPUT_MANIFEST_FILENAME` and the fitted statistics:

``fit``
    T2 (DADA-2000 original, windowed). The statistics ``s``, ``c``, ``m_u``,
    ``sigma_u`` are fitted on **T2-train minus T2-val** source videos (the frozen
    v2 split); every source the corpus references (train, T2-val and test) is
    then baked with its own CRN reference. Windows stay slices of the baked source
    file, so a T2 window's reference is its source video in training *and* in eval.

``apply``
    Another corpus (DoTA-dev clips) baked with the statistics of a ``fit`` cache;
    each clip is its own CRN reference. Nothing is refitted.

Arms (``--crn``, ``--motion``): A1 = CRN only, A2 = motion only, A3 = both. A0 is
the plain CLIP cache and is never baked.

CLI::

    python -m core.tools.build_v2_inputs fit \\
        --t2-dir data/DADA2000_orig --clip-dir cache/clip/DADA2000_orig \\
        [--video-dir cache/video/vit_b_k710_dl_from_giant/DADA2000_orig_s8_squash] \\
        --crn R2 --motion none --out-dir cache/v2/A1_R2/DADA2000_orig

    python -m core.tools.build_v2_inputs apply \\
        --stats-dir cache/v2/A1_R2/DADA2000_orig --clip-dir cache/clip/DoTA_s1_ncc --stride 3 \\
        --ids-file core/splits/v2/dota_dev.txt --out-dir cache/v2/A1_R2/DoTA_s3_from_s1

``--stride N`` subsamples each clip's rows ``[::N]`` before baking (DoTA at the E1 protocol
B = ``s1[::3]``); the CRN reference is then taken over the subsampled clip, as the model sees it.
A motion arm's ``--video-dir`` must then be a stride-1 cache row-aligned with the CLIP one
(DoTA-CAP's ``DoTA_CAP_s1_squash``); its rows are subsampled the same way.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.v2_inputs import (
    STATS_FILENAME,
    V2Stats,
    bake_rows,
    fit_stats,
    load_stats,
    save_stats,
)
from core.data.v2_splits import lines_sha1
from core.tools.crn_select import t2_sources
from core.tools.feature_cache import save_array
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

SUBCOMMAND_FIT = "fit"
SUBCOMMAND_APPLY = "apply"
FITTED_ON = "T2-train minus T2-val (core/splits/v2/t2_val_sources.txt)"


def arm_name(crn: str, motion: str) -> str:
    """A1 / A2 / A3 from the two switches; both off is A0, which is never baked."""
    has_crn, has_motion = crn != constants.V2_OFF, motion != constants.V2_OFF
    if not (has_crn or has_motion):
        raise ValueError("crn and motion are both 'none': that is A0, the plain CLIP cache")
    return "A3" if has_crn and has_motion else ("A1" if has_crn else "A2")


def load_rows(cache_dir: Path, ids: list[str]) -> dict[str, np.ndarray]:
    """``{id: (T, d)}`` from a per-id ``.npy`` cache; a missing id raises (C10)."""
    missing = [v for v in ids if not (cache_dir / f"{v}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ids have no feature in {cache_dir}: {missing[:5]}")
    return {v: np.load(cache_dir / f"{v}.npy") for v in ids}


def check_video_stride(video_dir: Path, stride: int) -> None:
    """``--stride`` subsamples the motion rows too, so they must be stride-1 like the CLIP rows.

    A stride-1 VideoMAE cache (DoTA-CAP, ``core.tools.dota_cap``) is row-aligned with
    ``DoTA_s1_ncc``; a cache built at another stride would be subsampled twice (C2/C13).
    """
    if stride == 1:
        return
    path = video_dir / constants.VIDEO_MANIFEST_FILENAME
    if not path.is_file():
        raise SystemExit(f"{path} missing: cannot verify the motion cache's stride")
    built = json.loads(path.read_text(encoding="utf-8")).get("stride")
    if built != 1:
        raise SystemExit(
            f"--stride {stride} needs a stride-1 motion cache; {video_dir} was built at {built!r}"
        )


def t2_all_sources(t2_dir: Path) -> list[str]:
    """Every source video a T2 corpus references (train, T2-val and test windows)."""
    meta = json.loads((t2_dir / constants.META_FILENAME).read_text(encoding="utf-8"))
    return sorted({str(entry["source"]) for entry in meta.values()})


def bake(
    ids: list[str],
    clips: dict[str, np.ndarray],
    motions: dict[str, np.ndarray] | None,
    stats: V2Stats,
    out_dir: Path,
) -> int:
    """Write every id's baked rows atomically; returns the row width."""
    out_dir.mkdir(parents=True, exist_ok=True)
    width = 0
    for video_id in ids:
        rows = bake_rows(clips[video_id], None if motions is None else motions[video_id], stats)
        width = rows.shape[1]
        save_array(out_dir / f"{video_id}.npy", rows)
    LOGGER.info("baked %d files (width %d) -> %s", len(ids), width, out_dir)
    return width


def _manifest(stats: V2Stats, width: int, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm": arm_name(stats.crn, stats.motion),
        "crn": stats.crn,
        "motion": stats.motion,
        "crn_warmup": constants.V2_CRN_WARMUP_STEPS,
        "width": width,
        "s": stats.s,
        "c": stats.c,
        "stats_file": STATS_FILENAME,
        "architecture": "core/docs/v2/KAT_VAD_v2_ARCHITECTURE.md §4-§5",
        **extra,
    }


def run_fit(args: argparse.Namespace) -> dict[str, Any]:
    meta = json.loads((args.t2_dir / constants.META_FILENAME).read_text(encoding="utf-8"))
    val, train = t2_sources(meta)
    everything = t2_all_sources(args.t2_dir)
    has_motion = args.motion != constants.V2_OFF
    clips = load_rows(args.clip_dir, everything)
    motions = load_rows(args.video_dir, everything) if has_motion else None
    stats = fit_stats(
        {v: clips[v] for v in train},
        None if motions is None else {v: motions[v] for v in train},
        args.crn,
        args.motion,
    )
    width = bake(everything, clips, motions, stats, args.out_dir)
    save_stats(args.out_dir, stats)
    manifest = _manifest(
        stats,
        width,
        {
            "mode": SUBCOMMAND_FIT,
            "fitted_on": FITTED_ON,
            "train_sources": len(train),
            "train_ids_sha1": lines_sha1(train),
            "t2_val_sources": len(val),
            "t2_val_sha1": lines_sha1(val),
            "baked_ids": len(everything),
            "reference_unit": "source video (train and eval windows are slices of it)",
            "t2_dir": str(args.t2_dir),
            "clip_dir": str(args.clip_dir),
            "video_dir": str(args.video_dir) if has_motion else None,
        },
    )
    write_json_atomic(args.out_dir / constants.V2_INPUT_MANIFEST_FILENAME, manifest)
    LOGGER.info(
        "fit %s: s=%.4f c=%.4f on %d T2-train sources",
        manifest["arm"], stats.s, stats.c, len(train),
    )
    return manifest


def run_apply(args: argparse.Namespace) -> dict[str, Any]:
    stats = load_stats(args.stats_dir)
    fitted = json.loads(
        (args.stats_dir / constants.V2_INPUT_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    if args.ids_file is not None:
        ids = sorted(
            line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    else:
        ids = sorted(p.stem for p in args.clip_dir.glob("*.npy"))
    clips = {v: rows[:: args.stride] for v, rows in load_rows(args.clip_dir, ids).items()}
    motions = None
    if stats.has_motion:
        if args.video_dir is None:
            raise SystemExit("this arm has a motion stream: pass --video-dir")
        check_video_stride(args.video_dir, args.stride)
        motions = {v: rows[:: args.stride] for v, rows in load_rows(args.video_dir, ids).items()}
    width = bake(ids, clips, motions, stats, args.out_dir)
    save_stats(args.out_dir, stats)
    manifest = _manifest(
        stats,
        width,
        {
            "mode": SUBCOMMAND_APPLY,
            "fitted_on": fitted["fitted_on"],
            "train_ids_sha1": fitted["train_ids_sha1"],
            "stats_from": str(args.stats_dir),
            "baked_ids": len(ids),
            "baked_ids_sha1": lines_sha1(ids),
            "reference_unit": "clip",
            "clip_dir": str(args.clip_dir),
            "stride_over_clip_dir": args.stride,
            "video_dir": str(args.video_dir) if stats.has_motion else None,
        },
    )
    write_json_atomic(args.out_dir / constants.V2_INPUT_MANIFEST_FILENAME, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser(SUBCOMMAND_FIT, help="fit on T2-train minus T2-val, bake all T2 sources")
    fit.add_argument("--t2-dir", type=Path, required=True, help="T2 dataset dir (meta.json)")
    fit.add_argument("--clip-dir", type=Path, required=True, help="per-source CLIP cache")
    fit.add_argument("--video-dir", type=Path, default=None, help="per-source VideoMAE cache")
    fit.add_argument("--crn", choices=constants.V2_CRN_CHOICES, required=True)
    fit.add_argument("--motion", choices=constants.V2_MOTION_CHOICES, required=True)
    fit.add_argument("--out-dir", type=Path, required=True)
    apply = sub.add_parser(SUBCOMMAND_APPLY, help="bake another corpus with fitted stats")
    apply.add_argument("--stats-dir", type=Path, required=True, help="a `fit` output dir")
    apply.add_argument("--clip-dir", type=Path, required=True, help="per-clip CLIP cache")
    apply.add_argument("--video-dir", type=Path, default=None, help="per-clip VideoMAE cache")
    apply.add_argument("--ids-file", type=Path, default=None, help="ids to bake (default: all)")
    apply.add_argument(
        "--stride", type=int, default=1, help="subsample clip rows [::N] first (DoTA s1 -> 3)"
    )
    apply.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == SUBCOMMAND_FIT:
        if args.motion != constants.V2_OFF and args.video_dir is None:
            raise SystemExit("--motion needs --video-dir")
        arm_name(args.crn, args.motion)
        run_fit(args)
    else:
        run_apply(args)


if __name__ == "__main__":
    main()
