"""Draw the zero-shot score curve of EVERY video of one or more sets: LaGoVAD+KIM vs LaGoVAD.

A thin wrapper over ``core.tools.zs_curves --all`` so nobody has to remember the paths.
LaGoVAD+KIM is the K5 model (s2024) trained on ``--train`` (Nexar, or T2 shown as DADA2000);
LaGoVAD is the released checkpoint (DoTA under protocol B, as K5). Illustration only (n = 1):
decides nothing.

Output per set (``<tag>`` defaults to ``lagovad_kim_vs_lagovad_s2024``)::

    outputs/v_final/kim/REPORTS/figures/train_<train>/<set>/<tag>/videos/<set>/<rank>_<video>.png
        rank 1 = LaGoVAD+KIM's worst per-video AUC; all-normal videos last
    outputs/v_final/kim/REPORTS/figures/train_<train>/<set>/<tag>/onset_aligned.png
    outputs/v_final/kim/REPORTS/figures/train_<train>/<set>/<tag>/zs_curves_<tag>.json

Usage::

    source .venv/bin/activate
    python scripts/plot_zs_curves.py --set dota                       # trained on Nexar
    python scripts/plot_zs_curves.py --train t2 --set dota tad dad    # trained on T2 (DADA2000)
    python scripts/plot_zs_curves.py --set tad --main <run dir> --lagovad <run dir>
    python scripts/plot_zs_curves.py --set nexar --main-only          # in-domain, K5 alone

A set that is the training corpus's own split (Nexar val for ``--train nexar``) is titled
"in-domain" instead of "zero-shot".
"""

from __future__ import annotations

import argparse
import logging
import shlex
from pathlib import Path

from core import constants
from core.tools import zs_curves

LOGGER = logging.getLogger(__name__)

LAGOVAD = constants.ZS_LAGOVAD_ARM
MAIN_SOURCES = constants.ZS_CURVES_MAIN_SOURCES
LAGOVAD_SOURCES = constants.ZS_CURVES_LAGOVAD_SOURCES
FIGURES = (constants.ZS_CURVES_FINAL_ROOT / constants.KIM_OUTPUT_DIRNAME / "REPORTS"
           / constants.ZS_CURVES_FIGURES_DIRNAME)


def main_label(train: str) -> str:
    """K5's name on the figures, with the corpus it was trained on."""
    return (f"{constants.ZS_CURVES_MAIN_LABEL} "
            f"(trained on {constants.ZS_CURVES_TRAIN_LABELS[train]})")


def out_dir(train: str, set_name: str, tag: str) -> Path:
    return FIGURES / f"train_{train}" / set_name / tag


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score curve of every video of a set.")
    parser.add_argument("--train", default=constants.ZS_CURVES_DEFAULT_TRAIN,
                        choices=list(MAIN_SOURCES), help="corpus LaGoVAD+KIM was trained on")
    parser.add_argument("--set", dest="sets", nargs="+", required=True,
                        choices=list(LAGOVAD_SOURCES),
                        help="one or more zero-shot sets, each drawn into its own folder")
    parser.add_argument("--tag", default=None,
                        help="output folder name under train_<train>/<set>/ and the json name "
                             f"(default {constants.ZS_CURVES_DEFAULT_TAG}, or "
                             f"{constants.ZS_CURVES_MAIN_ONLY_TAG} with --main-only)")
    parser.add_argument("--main", type=Path, default=None,
                        help="LaGoVAD+KIM scores (one --set only); default per train and set")
    parser.add_argument("--lagovad", type=Path, default=None,
                        help=f"{LAGOVAD} scores (one --set only); default per set")
    parser.add_argument("--seed-label", default=f"s{constants.KIM_SEED}, n = 1")
    parser.add_argument("--main-only", action="store_true",
                        help=f"draw LaGoVAD+KIM alone (no {LAGOVAD} read, e.g. not synced yet)")
    parser.add_argument("--dry-run", action="store_true", help="print the zs_curves calls only")
    return parser


def sources(set_name: str, args: argparse.Namespace) -> tuple[Path, ...]:
    """(LaGoVAD+KIM[, LaGoVAD]) score paths of a set: the CLI override, else the default."""
    root = constants.ZS_CURVES_FINAL_ROOT
    main_src = args.main or root / MAIN_SOURCES[args.train][set_name]
    if args.main_only:
        return (main_src,)
    return main_src, args.lagovad or root / LAGOVAD_SOURCES[set_name]


def domain(train: str, set_name: str) -> str:
    """Figure-title domain: the training corpus's own split is in-domain, the rest zero-shot."""
    if (train, set_name) in constants.ZS_CURVES_IN_DOMAIN_PAIRS:
        return constants.ZS_CURVES_DOMAIN_IN
    return constants.ZS_CURVES_DOMAIN_ZERO_SHOT


def tag_of(args: argparse.Namespace) -> str:
    """``--tag``, else the default for a two-model or a ``--main-only`` figure."""
    if args.tag:
        return str(args.tag)
    return constants.ZS_CURVES_MAIN_ONLY_TAG if args.main_only else constants.ZS_CURVES_DEFAULT_TAG


def tool_argv(set_name: str, args: argparse.Namespace) -> list[str]:
    """The ``core.tools.zs_curves`` arguments this wrapper stands for, for one set."""
    paths = sources(set_name, args)
    label = main_label(args.train)
    tag = tag_of(args)
    argv = ["--all", "--rank-by", label, "--name", tag,
            "--out-dir", str(out_dir(args.train, set_name, tag)),
            "--seed-label", args.seed_label, "--domain", domain(args.train, set_name),
            "--read", set_name, label, str(paths[0])]
    if len(paths) > 1:
        argv += ["--read", set_name, LAGOVAD, str(paths[1])]
    return argv


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if len(args.sets) > 1 and (args.main or args.lagovad):
        parser.error("--main / --lagovad take one path, so they need exactly one --set")
    if args.main_only and args.lagovad:
        parser.error("--main-only draws no LaGoVAD read: drop --lagovad")
    no_read = [s for s in args.sets if s not in MAIN_SOURCES[args.train] and not args.main]
    if no_read:
        parser.error(f"--train {args.train} has no zero-shot read on {no_read} "
                     f"(available: {sorted(MAIN_SOURCES[args.train])})")
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
