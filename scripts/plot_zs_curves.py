"""Draw the zero-shot score curve of EVERY video of one or more sets: LaGoVAD+KIM vs LaGoVAD.

A thin wrapper over ``core.tools.zs_curves --all`` so nobody has to remember the paths.
LaGoVAD+KIM is the K5 main model (s2024); LaGoVAD is the released checkpoint (DoTA under
protocol B, as K5). Illustration only (n = 1): decides nothing.

Output per set (``<tag>`` defaults to ``lagovad_kim_vs_lagovad_s2024``)::

    outputs/v_final/kim/REPORTS/figures/<set>/<tag>/videos/<set>/<rank>_<video>.png
        rank 1 = LaGoVAD+KIM's worst per-video AUC; all-normal videos last
    outputs/v_final/kim/REPORTS/figures/<set>/<tag>/onset_aligned.png
    outputs/v_final/kim/REPORTS/figures/<set>/<tag>/zs_curves_<tag>.json

Usage::

    source .venv/bin/activate
    python scripts/plot_zs_curves.py --set dota
    python scripts/plot_zs_curves.py --set dada tad dad --tag my_tag
    python scripts/plot_zs_curves.py --set tad --main <run dir> --lagovad <run dir>
"""

from __future__ import annotations

import argparse
import logging
import shlex
from pathlib import Path

from core import constants
from core.tools import zs_curves

LOGGER = logging.getLogger(__name__)

MAIN = constants.ZS_CURVES_MAIN_LABEL
LAGOVAD = constants.ZS_LAGOVAD_ARM
SOURCES = constants.ZS_CURVES_DEFAULT_SOURCES
FIGURES = (constants.ZS_CURVES_FINAL_ROOT / constants.KIM_OUTPUT_DIRNAME / "REPORTS"
           / constants.ZS_CURVES_FIGURES_DIRNAME)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score curve of every video of a set.")
    parser.add_argument("--set", dest="sets", nargs="+", required=True, choices=list(SOURCES),
                        help="one or more zero-shot sets, each drawn into figures/<set>/<tag>/")
    parser.add_argument("--tag", default=constants.ZS_CURVES_DEFAULT_TAG,
                        help="output folder name under figures/<set>/ and the json name")
    parser.add_argument("--main", type=Path, default=None,
                        help=f"{MAIN} scores (one --set only); default per set")
    parser.add_argument("--lagovad", type=Path, default=None,
                        help=f"{LAGOVAD} scores (one --set only); default per set")
    parser.add_argument("--seed-label", default=f"s{constants.KIM_SEED}, n = 1")
    parser.add_argument("--dry-run", action="store_true", help="print the zs_curves calls only")
    return parser


def sources(set_name: str, args: argparse.Namespace) -> tuple[Path, Path]:
    """(main, LaGoVAD) score paths of a set: the CLI override, else the default."""
    main_rel, lagovad_rel = SOURCES[set_name]
    root = constants.ZS_CURVES_FINAL_ROOT
    return (args.main or root / main_rel, args.lagovad or root / lagovad_rel)


def tool_argv(set_name: str, args: argparse.Namespace) -> list[str]:
    """The ``core.tools.zs_curves`` arguments this wrapper stands for, for one set."""
    main_src, lagovad_src = sources(set_name, args)
    return ["--all", "--rank-by", MAIN, "--name", args.tag,
            "--out-dir", str(FIGURES / set_name / args.tag), "--seed-label", args.seed_label,
            "--read", set_name, MAIN, str(main_src),
            "--read", set_name, LAGOVAD, str(lagovad_src)]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if len(args.sets) > 1 and (args.main or args.lagovad):
        parser.error("--main / --lagovad take one path, so they need exactly one --set")
    missing = [str(p) for s in args.sets for p in sources(s, args) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"score source(s) not found: {missing}")
    for set_name in args.sets:
        call = tool_argv(set_name, args)
        LOGGER.info("python -m core.tools.zs_curves %s", shlex.join(call))
        if not args.dry_run:
            zs_curves.run(zs_curves.build_arg_parser().parse_args(call))


if __name__ == "__main__":
    main()
