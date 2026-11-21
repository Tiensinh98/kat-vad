"""Whole DoTA-CAP from one row's two protocol-B reads: ``dota_cap_dev`` + ``dota_cap_eval``.

Rule: ``core/docs/final/KIM_RUN.md`` §35 / §36. The thesis reports DoTA-CAP as one set (dev +
eval = 1,129 clips). Every row is read twice by ``core.tools.protocol_b_eval`` (once per split);
this tool concatenates the two reads **per clip, without re-scoring** into a directory that
``core.tools.zs_arm_table`` reads like any other protocol-B read (``clip_scores.npz``,
``clip_aucs.json``, ``protocol_b_readout.json``).

It refuses two reads that are not the same model under the same protocol: the arm, the runs,
their global steps, the checkpoint files, the released-baseline flag and the protocol string
must be equal, the splits must be ``dota_cap_dev`` and ``dota_cap_eval``, the clip ids
disjoint, and (by default) the clip counts the frozen manifest's 569 / 560.

CLI::

    python -m core.tools.dota_cap_pool --dev <row>/eval/dota_cap_dev \\
        --eval <row>/eval/dota_cap_eval --out-dir <row>/eval/dota_cap_all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools import protocol_b_eval as pb
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

SAME_KEYS = ("arm", "runs", "global_steps", "checkpoints", "protocol")
POOLED_NOTE = "concatenation of the dev read and the eval read; no re-scoring"


def _readout(read_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((read_dir / pb.READOUT_JSON).read_text(encoding="utf-8"))
    return out


def check_pair(dev: dict[str, Any], ev: dict[str, Any]) -> None:
    """Raise unless ``dev`` and ``ev`` are one model's reads of the two DoTA-CAP splits."""
    if (dev.get("split"), ev.get("split")) != (constants.V2_SPLIT_DOTA_CAP_DEV,
                                               constants.V2_SPLIT_DOTA_CAP_EVAL):
        want = (constants.V2_SPLIT_DOTA_CAP_DEV, constants.V2_SPLIT_DOTA_CAP_EVAL)
        raise ValueError(f"splits {dev.get('split')!r} / {ev.get('split')!r}: want {want}")
    for key in SAME_KEYS:
        if dev.get(key) != ev.get(key):
            raise ValueError(f"{key} differs: dev {dev.get(key)!r} vs eval {ev.get(key)!r}")
    if bool(dev.get("released_baseline")) != bool(ev.get("released_baseline")):
        raise ValueError("one read is the released baseline and the other is not")


def pool(dev_dir: Path, eval_dir: Path, expect: tuple[int, int] | None) -> tuple[
        dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    """``(scores, labels, clip_aucs, readout)`` of the concatenated read (pure: writes nothing)."""
    dev, ev = _readout(dev_dir), _readout(eval_dir)
    check_pair(dev, ev)
    sd, ld = pb.read_clip_scores(dev_dir / pb.CLIP_SCORES_NPZ)
    se, le = pb.read_clip_scores(eval_dir / pb.CLIP_SCORES_NPZ)
    overlap = sorted(set(sd) & set(se))
    if overlap:
        raise ValueError(f"{len(overlap)} clips in both reads: {overlap[:5]}")
    if expect is not None and (len(sd), len(se)) != expect:
        raise ValueError(f"clips dev {len(sd)} / eval {len(se)}, want {expect[0]} / {expect[1]}")
    aucs_d = json.loads((dev_dir / pb.CLIP_AUCS_JSON).read_text(encoding="utf-8"))
    aucs_e = json.loads((eval_dir / pb.CLIP_AUCS_JSON).read_text(encoding="utf-8"))
    for name, aucs, ids in (("dev", aucs_d, sd), ("eval", aucs_e, se)):
        if not set(aucs) <= set(ids):
            raise ValueError(f"{name}: clip_aucs.json names clips absent from clip_scores.npz")
    aucs = {**aucs_d, **aucs_e}
    scores, labels = {**sd, **se}, {**ld, **le}
    readout = {
        "protocol": dev["protocol"],
        "split": f"{constants.V2_SPLIT_DOTA_CAP_DEV} + {constants.V2_SPLIT_DOTA_CAP_EVAL}",
        "clips": len(scores),
        "two_class": len(aucs),
        "arm": dev["arm"],
        "runs": dev["runs"],
        "released_baseline": bool(dev.get("released_baseline")),
        "global_steps": dev["global_steps"],
        "checkpoints": dev["checkpoints"],
        "per_run_macro": {"pooled": float(np.mean(list(aucs.values())))},
        "position_ruler": None,
        "parts": {"dev": str(dev_dir), "eval": str(eval_dir)},
        "note": POOLED_NOTE,
    }
    return scores, labels, aucs, readout


def write_pooled(dev_dir: Path, eval_dir: Path, out_dir: Path,
                 expect: tuple[int, int] | None) -> dict[str, Any]:
    """Pool and write the three files ``zs_arm_table`` reads; return the read-out."""
    scores, labels, aucs, readout = pool(dev_dir, eval_dir, expect)
    out_dir.mkdir(parents=True, exist_ok=True)
    pb.write_clip_scores(out_dir / pb.CLIP_SCORES_NPZ, scores, labels)
    write_json_atomic(out_dir / pb.CLIP_AUCS_JSON, aucs)
    write_json_atomic(out_dir / pb.READOUT_JSON, readout)
    LOGGER.info("pooled %d clips (%d two-class), mean per-clip AUC %.4f -> %s",
                readout["clips"], readout["two_class"], readout["per_run_macro"]["pooled"], out_dir)
    return readout


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--dev", type=Path, required=True, help="protocol_b_eval dir (dev)")
    parser.add_argument("--eval", type=Path, required=True, help="protocol_b_eval dir (eval)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--any-size", action="store_true",
                        help="skip the 569 / 560 clip-count check (tests, partial reads)")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    counts = (constants.DOTA_CAP_DEV_CLIPS, constants.DOTA_CAP_EVAL_CLIPS)
    expect = None if args.any_size else counts
    write_pooled(args.dev, args.eval, args.out_dir, expect)


if __name__ == "__main__":
    main()
