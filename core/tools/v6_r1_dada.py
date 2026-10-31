"""v6 post-hoc read: v5 R1 s2024 zero-shot on the official DADA-2000 test set (PREREG_V6 §4).

The set is SimpleTAD's (CVPR 2025) DADA-2000 evaluation list, ``DADA2K_my_split/validation.txt``
from its ``D2K.zip``: every clip of the original release that is **not** in LOTVS's own
training list (``simple-tad/data_tools/dada/prepare_anno_dada2000.py``). Lines are
``{type}/{video:03d}``; this project names the same clip ``t{type:02d}_v{video:03d}``
(:func:`core.data.dada_origin.video_id`).

R1 was trained with ``v5.readout = mean``: its motion term is
``c (fc_norm(mean G) - m_u) / sigma_u`` read from a 7x7 V2-S grid by a readout with
**no trainable parameter** (frozen buffers only).
A2's baked input carries that same term as ``u`` (PREREG_V5 S1-G2(b), ≤ 1.1e-6), and no DADA
grid exists. ``strip`` therefore writes the same weights with ``v5.readout = none`` (the
readout's buffers dropped), so R1 reads A2's baked DADA rows; gate **G-S** checks that this
stripped R1 reproduces R1's own protocol-B scores on DoTA-CAP-dev before any DADA number.

CLI::

    python -m core.tools.v6_r1_dada ids --official validation.txt --annotation dada标注.xlsx \\
        --census resolved_census.json --clip-dir cache/clip/DADA2000_orig \\
        --video-dir cache/video/.../DADA2000_orig_s8_squash --out-dir REPORTS/dada_official
    python -m core.tools.v6_r1_dada strip --ckpt R1/s2024/checkpoint_last.pt \\
        --out-dir R1_stripped/s2024
    python -m core.tools.v6_r1_dada gate --reference R1/clip_scores.npz \\
        --candidate R1_stripped_pb/clip_scores.npz --out REPORTS/g_s.json
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from core import constants
from core.data.dada_origin import load_counts, make_record, parse_annotation, video_id
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import read_clip_scores
from core.tools.tadz_readout import clip_aucs
from core.train import CHECKPOINT_LAST

LOGGER = logging.getLogger(__name__)

IDS_TXT = "dada_official_ids.txt"
IDS_JSON = "dada_official_ids.json"
STRIP_JSON = "strip_report.json"
READOUT_PREFIX = "grid_readout."
# GridReadout(mean)'s state: frozen buffers only (core/models/mqr.py);
# anything else = a trained readout
MEAN_READOUT_BUFFERS = frozenset(
    {"fc_weight", "fc_bias", "c", "m_u", "sigma_u", "head_loaded"}
)
RUN_FILES = ("metrics.jsonl",)
CONFIG_YAML = "config.yaml"


# ---------------------------------------------------------------------------
# ids: SimpleTAD's list -> this project's source ids, with every exclusion named
# ---------------------------------------------------------------------------
def parse_official(path: Path) -> list[str]:
    """``{type}/{video}`` lines -> ``t{type:02d}_v{video:03d}`` (sorted, unique)."""
    ids: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("/")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError(f"{path}:{n}: expected 'type/video', got {line!r}")
        ids.add(video_id(int(parts[0]), int(parts[1])))
    if not ids:
        raise ValueError(f"{path} lists no clip")
    return sorted(ids)


def filter_sources(
    official: list[str],
    annotated: dict[str, Any],
    census: dict[str, int],
    feature_dirs: dict[str, Path],
) -> tuple[list[str], dict[str, list[str]]]:
    """Kept ids and ``{reason: ids}`` for every official clip this harness cannot read."""
    dropped: dict[str, list[str]] = {}
    kept: list[str] = []
    for v in official:
        if v not in annotated:
            reason = "no_annotation_row"
        elif v not in census:
            reason = "no_frames_on_disk"
        elif make_record(annotated[v], census[v]) is None:
            reason = "anomaly_past_frames"
        else:
            reason = next(
                (f"no_{name}_cache" for name, d in feature_dirs.items()
                 if not (d / f"{v}.npy").is_file()),
                "",
            )
        if reason:
            dropped.setdefault(reason, []).append(v)
        else:
            kept.append(v)
    return kept, dropped


def run_ids(args: argparse.Namespace) -> dict[str, Any]:
    official = parse_official(args.official)
    annotated = {row.video_id: row for row in parse_annotation(args.annotation)}
    census = load_counts(args.census)
    kept, dropped = filter_sources(
        official, annotated, census, {"clip": args.clip_dir, "video": args.video_dir}
    )
    coverage = len(kept) / len(official)
    report = {
        "official": len(official),
        "kept": len(kept),
        "coverage": coverage,
        "min_coverage": constants.V6_DADA_MIN_COVERAGE,
        "passes": coverage >= constants.V6_DADA_MIN_COVERAGE,
        "dropped": {k: sorted(v) for k, v in sorted(dropped.items())},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(args.out_dir / IDS_TXT, "".join(f"{v}\n" for v in kept))
    write_json_atomic(args.out_dir / IDS_JSON, report)
    LOGGER.info("official DADA list: %d clips, kept %d (%.3f), dropped %s", len(official),
                len(kept), coverage, {k: len(v) for k, v in dropped.items()})
    if not report["passes"]:
        raise SystemExit(f"coverage {coverage:.3f} < {constants.V6_DADA_MIN_COVERAGE}: "
                         f"extract the missing features first ({sorted(dropped)})")
    return report


# ---------------------------------------------------------------------------
# strip: R1 (mean readout) -> the same weights reading A2's baked u
# ---------------------------------------------------------------------------
def strip_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The checkpoint with ``v5.readout = none`` and the mean readout's buffers removed."""
    config = payload.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("v5"), dict):
        raise ValueError("checkpoint carries no v5 config")
    v5 = config["v5"]
    if v5.get("readout") != constants.V5_READOUT_MEAN or int(v5.get("queries", 1)) != 1:
        raise ValueError(f"only a 1-query mean readout is parameter-free; got {v5}")
    if float(v5.get("smil_weight", 0.0)) != 0.0:
        raise ValueError("an sMIL head is trained: this checkpoint is not R1")
    state = payload["model"]
    dropped = sorted(k for k in state if k.startswith(READOUT_PREFIX))
    unknown = {k[len(READOUT_PREFIX):] for k in dropped} - MEAN_READOUT_BUFFERS
    if unknown:
        raise ValueError(f"the readout holds non-buffer state {sorted(unknown)}")
    if not bool(state.get(f"{READOUT_PREFIX}head_loaded", torch.zeros(()))):
        raise ValueError(
            "the readout's motion head was never loaded: R1 did not read A2's statistics"
        )
    out = dict(payload)
    out["model"] = {k: t for k, t in state.items() if not k.startswith(READOUT_PREFIX)}
    out["config"] = {**config, "v5": {**v5, "readout": constants.V5_READOUT_OFF}}
    return out, dropped


def run_strip(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # nosec B614 - own ckpt
    stripped, dropped = strip_payload(payload)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.out_dir / (CHECKPOINT_LAST + ".partial")
    torch.save(stripped, tmp)
    tmp.replace(args.out_dir / CHECKPOINT_LAST)
    for name in RUN_FILES:
        shutil.copy2(args.ckpt.parent / name, args.out_dir / name)
    config_yaml = args.ckpt.parent / CONFIG_YAML
    if config_yaml.is_file():
        cfg = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
        cfg["v5"]["readout"] = constants.V5_READOUT_OFF
        write_text_atomic(args.out_dir / CONFIG_YAML, yaml.safe_dump(cfg, sort_keys=False))
    report = {"source": str(args.ckpt), "global_step": int(payload.get("global_step", -1)),
              "dropped_keys": dropped}
    write_json_atomic(args.out_dir / STRIP_JSON, report)
    LOGGER.info("stripped %s -> %s (dropped %d buffers)", args.ckpt, args.out_dir, len(dropped))
    return report


# ---------------------------------------------------------------------------
# gate G-S: the stripped R1 reproduces R1 on DoTA-CAP-dev
# ---------------------------------------------------------------------------
def compare_scores(
    reference: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    candidate: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Macro of each, |Δ macro| and max per-frame |Δ| over the same clips and labels."""
    (ref_s, ref_l), (cand_s, cand_l) = reference, candidate
    if set(ref_s) != set(cand_s):
        raise ValueError(f"clip sets differ: {len(ref_s)} vs {len(cand_s)}")
    for v in ref_l:
        if not np.array_equal(ref_l[v], cand_l[v]):
            raise ValueError(f"{v}: labels differ")
    max_abs = max(float(np.max(np.abs(ref_s[v] - cand_s[v]))) for v in ref_s)
    macro_ref = float(np.mean(list(clip_aucs(ref_s, ref_l).values())))
    macro_cand = float(np.mean(list(clip_aucs(cand_s, ref_l).values())))
    passes = (abs(macro_cand - macro_ref) <= constants.V6_GS_MACRO_TOL
              and max_abs <= constants.V6_GS_SCORE_TOL)
    return {"clips": len(ref_s), "macro_reference": macro_ref, "macro_candidate": macro_cand,
            "macro_abs_delta": abs(macro_cand - macro_ref), "max_abs_score_delta": max_abs,
            "macro_tol": constants.V6_GS_MACRO_TOL, "score_tol": constants.V6_GS_SCORE_TOL,
            "passes": passes}


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    report = compare_scores(read_clip_scores(args.reference), read_clip_scores(args.candidate))
    write_json_atomic(args.out, report)
    LOGGER.info("G-S %s: %s", "PASS" if report["passes"] else "FAIL", report)
    if not report["passes"]:
        raise SystemExit("G-S FAIL: the stripped R1 is not R1 -- no DADA number is read")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ids = sub.add_parser("ids", help="SimpleTAD's DADA list -> readable source ids")
    ids.add_argument("--official", type=Path, required=True, help="DADA2K_my_split/validation.txt")
    ids.add_argument("--annotation", type=Path, required=True, help="dada标注.xlsx")
    ids.add_argument("--census", nargs="+", required=True, help="resolved_census.json")
    ids.add_argument("--clip-dir", type=Path, required=True, help="CLIP s8 cache of DADA2000_orig")
    ids.add_argument("--video-dir", type=Path, required=True, help="V2-S s8 cache of DADA2000_orig")
    ids.add_argument("--out-dir", type=Path, required=True)
    strip = sub.add_parser("strip", help="R1 (mean readout) -> readout none on A2's baked u")
    strip.add_argument("--ckpt", type=Path, required=True)
    strip.add_argument("--out-dir", type=Path, required=True)
    gate = sub.add_parser("gate", help="G-S: stripped R1 vs R1 protocol-B scores")
    gate.add_argument("--reference", type=Path, required=True, help="R1's clip_scores.npz")
    gate.add_argument("--candidate", type=Path, required=True, help="stripped R1's clip_scores.npz")
    gate.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    {"ids": run_ids, "strip": run_strip, "gate": run_gate}[args.command](args)


if __name__ == "__main__":
    main()
