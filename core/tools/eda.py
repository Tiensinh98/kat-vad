"""EDA CLI — characterize a corpus before trusting a number measured on it.

``report`` analyses one dataset; ``compare`` puts two or more finished reports
side by side. Both write Markdown for reading and JSON for scripting.

Typical use, on Colab after ``DADA_SETUP.md`` sections 5-9::

    python -m core.tools.eda report \\
      --dataset DADA2000 \\
      --data-dir  "$KATVAD_DATA_ROOT/DADA2000" \\
      --clip-dir  "$KATVAD_CACHE_ROOT/clip/DADA2000" \\
      --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/DADA2000" \\
      --output-dir "$KATVAD_OUTPUT_ROOT/eda/DADA2000" --plots

    python -m core.tools.eda compare \\
      "$KATVAD_OUTPUT_ROOT/eda/DADA2000/eda_report.json" \\
      "$KATVAD_OUTPUT_ROOT/eda/DoTA/eda_report.json" \\
      --output "$KATVAD_OUTPUT_ROOT/eda/compare.md"

Add ``--scores-dir <eval_dir>/scores`` to fold a trained arm's score curves into
§3.4 (the flatness diagnostic of ``RESULTS_DADA.md`` §4/§7.1).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from core import constants
from core.eda import corpus, report

LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("report", help="analyse one dataset")
    run.add_argument("--dataset", required=True,
                     help="dataset name, e.g. DADA2000 or DoTA (labels the report)")
    run.add_argument("--data-dir", type=Path, required=True,
                     help="dir holding labels_train.json / frame_labels_test.json / meta.json")
    run.add_argument("--clip-dir", type=Path, default=None,
                     help="CLIP feature cache; required by the 'features' section")
    run.add_argument("--flow-dir", type=Path, default=None,
                     help="RAFT cache dir (flow/v1/{DATASET}); optional")
    run.add_argument("--scores-dir", type=Path, default=None,
                     help="an eval run's scores/ dir, for the curve-flatness block")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--sections", default=",".join(constants.EDA_SECTIONS),
                     help=f"comma-separated subset of {','.join(constants.EDA_SECTIONS)}")
    run.add_argument("--score-head-kernel", type=int, default=constants.SCORE_HEAD_KERNEL,
                     help="kernel the coverage table is computed against (lesson C27)")
    run.add_argument("--mil-topk-pct", type=int, default=constants.MIL_TOPK_PCT)
    run.add_argument("--max-clips", type=int, default=0,
                     help="cap clips read in the features section (0 = all)")
    run.add_argument("--probe", action=argparse.BooleanOptionalAction, default=True,
                     help="run the supervised linear probes (RESULTS_DADA.md §10-B)")
    run.add_argument("--plots", action="store_true", help="also write PNG figures")
    run.add_argument("--seed", type=int, default=constants.SEED)

    cmp_ = sub.add_parser("compare", help="side-by-side table from finished reports")
    cmp_.add_argument("reports", type=Path, nargs="+", help="eda_report.json paths")
    cmp_.add_argument("--output", type=Path, required=True, help="Markdown destination")
    return parser


def _resolve_sections(raw: str) -> tuple[str, ...]:
    requested = tuple(s.strip() for s in raw.split(",") if s.strip())
    unknown = [s for s in requested if s not in constants.EDA_SECTIONS]
    if unknown:
        raise ValueError(
            f"Unknown section(s) {unknown}; choose from {list(constants.EDA_SECTIONS)}"
        )
    return requested


def _run_report(args: argparse.Namespace) -> None:
    sections = _resolve_sections(args.sections)
    if constants.EDA_SECTION_SCORES in sections and args.scores_dir is None:
        LOGGER.info("Section 'scores' requested without --scores-dir; nothing to fold in")
    payload = report.build_report(
        dataset=args.dataset,
        data_dir=args.data_dir,
        clip_dir=args.clip_dir,
        flow_dir=args.flow_dir,
        scores_dir=args.scores_dir,
        sections=sections,
        kernel=args.score_head_kernel,
        topk_pct=args.mil_topk_pct,
        max_clips=args.max_clips,
        probe=args.probe,
        seed=args.seed,
    )
    json_path, md_path = report.write_report(payload, args.output_dir)
    if args.plots:
        files = corpus.load_dataset_files(args.data_dir, args.dataset)
        report.write_plots(payload, files, args.output_dir / constants.EDA_PLOTS_DIRNAME)
    for verdict in payload["verdicts"]:
        LOGGER.info("[%s] %s -- %s", verdict["level"], verdict["title"], verdict["detail"])
    LOGGER.info("EDA written to %s and %s", md_path, json_path)


def _run_compare(args: argparse.Namespace) -> None:
    payloads = []
    for path in args.reports:
        with path.open(encoding="utf-8") as fh:
            payloads.append(json.load(fh))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.compare_reports(payloads), encoding="utf-8")
    LOGGER.info("Comparison of %d reports written to %s", len(payloads), args.output)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "report":
        _run_report(args)
    else:
        _run_compare(args)


if __name__ == "__main__":
    main()
