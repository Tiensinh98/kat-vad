"""TAD read by anomaly category, with clip AUC (``KIM_RUN.md`` §38.2 N4; printed, never gating).

TAD-test's 60 abnormal videos all carry ``class_name: "Car Accident"``, but only 23 are
``01_Accident``; the rest are illegal turns, pedestrians on the road, spills and others. A model
trained on accidents can only be expected to move the 23. This reads the per-video score files
``core.evaluate --save-scores`` writes (``scores/<video>.npz`` with ``score`` and ``gt``) and
prints, per category:

* **macro**: mean frame AUC over the category's two-class videos (TAD's ``auc_macro`` per class);
* **clip AUC**: peak score of the category's videos vs every ``Normal`` video's peak;
* **micro**: raw frames of the category's videos pooled with every ``Normal`` video's frames.

Usage::

    python -m core.tools.tad_category_read --read K5f .../eval/tad/K5/s2024 \\
        --read K5fno .../eval/tad/K5/s2024 --out-dir REPORTS --name k5fno_s2024
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import frame_auc
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

NORMAL_CATEGORY = "Normal"
ALL_ABNORMAL = "all abnormal"
SCORES_DIRNAME = "scores"
_SUFFIX = re.compile(r"_\d+$")


def category_of(video_id: str) -> str:
    """``01_Accident_004`` -> ``01_Accident``; ``Normal_128`` -> ``Normal``."""
    return _SUFFIX.sub("", video_id)


def load_scores(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{video: (score, gt)}`` from ``run_dir/scores/*.npz``."""
    files = sorted((run_dir / SCORES_DIRNAME).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no score files in {run_dir / SCORES_DIRNAME}")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in files:
        with np.load(path) as z:
            score, gt = np.asarray(z["score"], np.float64), np.asarray(z["gt"], np.int64)
        if score.shape != gt.shape:
            raise ValueError(f"{path.name}: score {score.shape} vs gt {gt.shape}")
        out[path.stem] = (score, gt)
    return out


def _cell(videos: dict[str, tuple[np.ndarray, np.ndarray]], ids: list[str],
          normals: list[str]) -> dict[str, Any]:
    two_class = [v for v in ids if 0 < videos[v][1].sum() < len(videos[v][1])]
    macro = [frame_auc(*videos[v]) for v in two_class]
    peaks = np.array([videos[v][0].max() for v in (*ids, *normals)])
    is_abn = np.array([1] * len(ids) + [0] * len(normals))
    pooled = [*ids, *normals]
    return {
        "videos": len(ids),
        "two_class": len(two_class),
        "macro": float(np.mean(macro)) if macro else None,
        "clip_auc": frame_auc(peaks, is_abn) if normals else None,
        "micro": frame_auc(np.concatenate([videos[v][0] for v in pooled]),
                           np.concatenate([videos[v][1] for v in pooled])),
    }


def read_categories(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-category cells, plus ``all abnormal`` (every non-Normal video together)."""
    videos = load_scores(run_dir)
    normals = sorted(v for v in videos if category_of(v) == NORMAL_CATEGORY)
    by_cat: dict[str, list[str]] = {}
    for v in sorted(videos):
        if category_of(v) != NORMAL_CATEGORY:
            by_cat.setdefault(category_of(v), []).append(v)
    cells = {cat: _cell(videos, ids, normals) for cat, ids in sorted(by_cat.items())}
    cells[ALL_ABNORMAL] = _cell(videos, [v for ids in by_cat.values() for v in ids], normals)
    return cells


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_markdown(reads: dict[str, dict[str, dict[str, Any]]]) -> str:
    rows = list(reads)
    cats = list(next(iter(reads.values())))
    lines = [
        "# TAD by category (`KIM_RUN.md` §38.2 N4) — printed, decides nothing",
        "",
        "Macro = mean frame AUC over the category's two-class videos. Clip AUC = peak score of the "
        "category's videos vs every Normal video's peak. Micro = the category's frames pooled with "
        "every Normal video's frames (raw). n = 1 seed; per-category counts are small.",
        "",
        "| category | videos | " + " | ".join(f"{r} macro / clip AUC / micro" for r in rows) + " |",
        "| --- | --- | " + " | ".join("---" for _ in rows) + " |",
    ]
    for cat in cats:
        first = reads[rows[0]][cat]
        cells = [f"{_num(reads[r][cat]['macro'])} / {_num(reads[r][cat]['clip_auc'])} / "
                 f"{_num(reads[r][cat]['micro'])}" for r in rows]
        lines.append(f"| {cat} | {first['videos']} ({first['two_class']} two-class) | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--read", nargs=2, action="append", metavar=("ROW", "RUN_DIR"),
                        required=True, help="a core.evaluate TAD output dir (holds scores/)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    reads = {row: read_categories(Path(d)) for row, d in args.read}
    sets = {row: sorted(cells) for row, cells in reads.items()}
    if len({tuple(v) for v in sets.values()}) != 1:
        raise ValueError(f"rows read different category sets: {sets}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"{constants.TAD_CATEGORY_STEM}_{args.name}"
    write_json_atomic(stem.with_suffix(".json"), reads)
    stem.with_suffix(".md").write_text(render_markdown(reads), encoding="utf-8")
    LOGGER.info("TAD by category -> %s", stem.with_suffix(".md"))


if __name__ == "__main__":
    main()
