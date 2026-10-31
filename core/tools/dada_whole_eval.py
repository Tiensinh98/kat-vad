"""Read one arm's checkpoints on whole DADA-2000 original source videos (addendum §29, D-N20).

The DADA-2000 test set of Amendment 18 is the **source videos** behind T2-test's windows,
scored whole: every row of the per-source stride-8 cache in one forward, one curve per video,
labels from the original annotation's span at stride 8 (``kill_switch_probe.source_labels``,
the rule Gate D0 and K used). A model trained on another corpus (Nexar) has never seen any
DADA video, so this is a zero-shot read; a T2-trained model has seen the *train* sources'
windows, never these.

Macro over the two-class sources (cluster = the video itself), micro over all frames,
seed-averaged per frame; per-run macro / micro for the paired seed interval. DADA-original
has no all-normal video, so a clip-level AUC exists only if one turns up (printed when it
does). The ``t/N`` and middle rulers show what position alone scores on the same frames.

CLI::

    python -m core.tools.dada_whole_eval --run s2024 runs/A3/s2024/checkpoint_last.pt ... \\
        --t2-dir data/DADA2000_orig --annotation .../dada标注.xlsx \\
        --census data/DADA2000_orig/counts/resolved_census.json \\
        --input-dir cache/v2/A3_R2_S/DADA2000_orig_test_nexar --data-dir data/Nexar_window \\
        --out-dir outputs/v2_nexar_window/eval/dada/A3
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
from core.data.dada_origin import load_counts
from core.data.v2_inputs import check_input_manifest
from core.device import resolve_device
from core.tools.kill_switch_probe import source_labels, write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import write_clip_scores
from core.tools.rate_matched_eval import Protocol, Run, load_finished_model, score_steps
from core.tools.tadz_readout import bootstrap, clip_aucs, micro, seed_average, video_auc
from core.train import load_class_names

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "dada_readout.json"
READOUT_MD = "dada_readout.md"
CLIP_AUCS_JSON = "clip_aucs.json"
CLIP_SCORES_NPZ = "clip_scores.npz"
TEST_SPLIT = "test"  # the T2 meta's split tag of every T2-test window

Curves = dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# pure pieces (tested without a model)
# ---------------------------------------------------------------------------
def test_sources(t2_dir: Path) -> list[str]:
    """The source videos behind T2-test's windows (T2's ``meta.json``)."""
    meta = json.loads((t2_dir / constants.META_FILENAME).read_text(encoding="utf-8"))
    sources = sorted({str(e["source"]) for e in meta.values() if e.get("split") == TEST_SPLIT})
    if not sources:
        raise ValueError(f"{t2_dir / constants.META_FILENAME} names no test window")
    return sources


def read_ids(path: Path) -> list[str]:
    """Source ids of ``--ids-file``: one ``t{type:02d}_v{video:03d}`` per line."""
    ids = sorted({line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()})
    if not ids:
        raise ValueError(f"{path} names no source")
    return ids


def check_rows(labels: Curves, rows: dict[str, int]) -> None:
    """One label per cached row on every source (C10, C13): a mismatch is a stride error."""
    missing = sorted(set(labels) - set(rows))
    if missing:
        raise FileNotFoundError(f"{len(missing)} sources have no input rows: {missing[:5]}")
    bad = [f"{v}: labels {len(labels[v])} rows {rows[v]}" for v in sorted(labels)
           if len(labels[v]) != rows[v]]
    if bad:
        raise ValueError(f"{len(bad)} sources: label length != cached rows: {bad[:5]}")


def has_both_classes(labels: Curves) -> bool:
    video = [int(g.max() > 0) for g in labels.values()]
    return 0 < sum(video) < len(video)


def summarize(per_run: dict[str, Curves], labels: Curves, seed: int) -> dict[str, Any]:
    """Every number of the read-out from per-run step curves (seed-averaged per frame)."""
    averaged = seed_average([per_run[n] for n in sorted(per_run)])
    monotone = {v: np.arange(len(g)) / len(g) for v, g in labels.items()}
    middle = {v: -np.abs(np.arange(len(g)) / len(g) - 0.5) for v, g in labels.items()}
    both = has_both_classes(labels)
    aucs = clip_aucs(averaged, labels)
    return {
        "videos": len(labels),
        "two_class": len(aucs),
        "abnormal_videos": sum(int(g.max() > 0) for g in labels.values()),
        "macro": bootstrap(aucs, seed),
        "micro": micro(averaged, labels),
        "video_auc": video_auc(averaged, labels) if both else None,
        "per_run_macro": {
            n: float(np.mean(list(clip_aucs(c, labels).values()))) for n, c in per_run.items()
        },
        "per_run_micro": {n: micro(c, labels) for n, c in per_run.items()},
        "monotone_ruler_macro": bootstrap(clip_aucs(monotone, labels), seed),
        "middle_ruler_macro": bootstrap(clip_aucs(middle, labels), seed),
        "clip_aucs": aucs,
        "averaged": averaged,
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score_run(
    run: Run, ids: list[str], input_dir: Path, data_dir: Path, device: torch.device
) -> tuple[Curves, int, dict[str, str]]:
    """Step curves of every source for one checkpoint, after the K4 input check."""
    model, step, text_encode_fn, cfg = load_finished_model(run, device)
    check_input_manifest(input_dir, cfg.v2.crn, cfg.v2.motion)
    class_names = load_class_names(data_dir)
    whole = Protocol(1, None)  # every cached (stride-8) row, whole video
    curves: Curves = {}
    for video_id in ids:
        feats = torch.from_numpy(np.ascontiguousarray(np.load(input_dir / f"{video_id}.npy"),
                                                      dtype=np.float32))
        curves[video_id] = score_steps(model, feats, text_encode_fn, class_names, video_id, whole)
    LOGGER.info("%s (step %d): scored %d DADA sources", run.name, step, len(curves))
    return curves, step, {"crn": cfg.v2.crn, "motion": cfg.v2.motion}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (args.ids_file is None) == (args.t2_dir is None):
        raise ValueError("pass exactly one of --t2-dir (T2-test's sources) / --ids-file")
    ids = read_ids(args.ids_file) if args.ids_file is not None else test_sources(args.t2_dir)
    labels = {
        v: np.asarray(g, dtype=np.int64)
        for v, g in source_labels(args.annotation, load_counts(args.census), ids,
                                  args.stride)[0].items()
    }
    rows = {v: int(np.load(args.input_dir / f"{v}.npy", mmap_mode="r").shape[0])
            for v in ids if (args.input_dir / f"{v}.npy").is_file()}
    check_rows(labels, rows)

    device = resolve_device(args.device)
    per_run: dict[str, Curves] = {}
    steps: dict[str, int] = {}
    arms: dict[str, dict[str, str]] = {}
    for name, ckpt in args.run:
        per_run[name], steps[name], arms[name] = score_run(
            Run(name, Path(ckpt)), ids, args.input_dir, args.data_dir, device
        )
    if len({json.dumps(a, sort_keys=True) for a in arms.values()}) != 1:
        raise ValueError(f"checkpoints of different arms cannot be seed-averaged: {arms}")
    summary = summarize(per_run, labels, args.seed)
    readout: dict[str, Any] = {
        "protocol": f"whole source video, every s{args.stride} row, labels = annotation span",
        "set": args.set_name or "DADA-2000 original: the sources of T2-test's windows",
        "input_dir": str(args.input_dir),
        "arm": next(iter(arms.values())),
        "checkpoints": dict(args.run),
        "global_steps": steps,
        **{k: v for k, v in summary.items() if k not in ("clip_aucs", "averaged")},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / CLIP_AUCS_JSON, summary["clip_aucs"])
    write_clip_scores(args.out_dir / CLIP_SCORES_NPZ, summary["averaged"], labels)
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(readout))
    LOGGER.info("DADA whole macro %s -> %s", readout["macro"], args.out_dir)
    return readout


def _fmt(c: dict[str, float] | None) -> str:
    return "—" if not c else f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def render_markdown(r: dict[str, Any]) -> str:
    video = "—" if r["video_auc"] is None else f"{r['video_auc']:.4f}"
    lines = [
        f"# {r['set']} — whole videos, {r['videos']} sources ({r['abnormal_videos']} abnormal, "
        f"{r['two_class']} two-class)",
        "",
        f"{r['protocol']}. Arm `{r['arm']}`, input `{r['input_dir']}`, steps {r['global_steps']}.",
        "",
        f"* **macro (seed-averaged, 95 % bootstrap over videos): {_fmt(r['macro'])}**",
        f"* micro: {r['micro']:.4f} · clip-level AUC: {video}",
        f"* rulers (no pixels): `t/N` {_fmt(r['monotone_ruler_macro'])} · "
        f"middle {_fmt(r['middle_ruler_macro'])}",
        f"* per-run macro: { {k: round(v, 4) for k, v in r['per_run_macro'].items()} }",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--run", nargs=2, action="append", required=True,
                        metavar=("NAME", "CKPT"), help="one finished checkpoint (repeat per seed)")
    parser.add_argument("--t2-dir", type=Path, default=None,
                        help="the PARENT T2 dir (meta.json names T2-test's sources)")
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="instead of --t2-dir: the sources to read, one id per line")
    parser.add_argument("--set-name", default=None, help="set label printed in the read-out")
    parser.add_argument("--annotation", type=Path, required=True, help="dada标注.xlsx")
    parser.add_argument("--census", nargs="+", required=True, help="resolved_census.json")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="per-source s8 rows: plain CLIP (A0) or the arm's baked cache")
    parser.add_argument("--data-dir", type=Path, required=True, help="training dir (defs.json)")
    parser.add_argument("--stride", type=int, default=constants.NEXAR_ZS_DADA_STRIDE)
    parser.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap seed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
