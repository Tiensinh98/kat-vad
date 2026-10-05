"""Amendment 8 Q4-Q5: the pilot arms' O1 / O1' on T2-test and the per-arm verdict.

Each checkpoint is scored on the **parent** T2 dir's evaluation file (T2-test, unseen by every
v2 arm) through its own input cache, with the exact forward and reads of ``v2_diagnostics``
(``score_t2_windows`` + ``guardrails``). O1' (``v2_diagnostics.o1_prime``) is read against A0
of the same seed on the same set, on T2-test here and on T2-val from each arm's batch
``diag.json``.

Q5, per arm: a survivor if it passed O1 on T2-val as registered, **or** it passes O1' on
T2-val **and** on T2-test. A0 is the reference and must have passed O1 on T2-val. When every
motion arm fails, the Motion Stream is dropped. Nothing here reads DoTA, and no adoption reads
T2-test (§17 Q4).

Hard gate (Q4): every T2-test source must have a row file in each arm's input cache, else
the tool stops -- the cache is re-applied, never refit.

CLI::

    python -m core.tools.v2_guard_test --data-dir data/DADA2000_orig \\
        --arm A0 runs/A0/checkpoint_last.pt cache/clip/DADA2000_orig \\
        --arm A2 runs/A2/checkpoint_last.pt cache/v2/A2/DADA2000_orig \\
        --val-diag A0 runs/A0/diag/diag.json --val-diag A2 runs/A2/diag_cap/diag.json \\
        --out-dir outputs/v2_pilot/o1prime_t2test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from core import constants
from core.data.dataset import FeatureEvalDataset
from core.data.v2_inputs import check_input_manifest
from core.device import resolve_device
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.rate_matched_eval import Run, load_finished_model
from core.tools.v2_diagnostics import (
    guardrails,
    o1_prime,
    render_o1_prime,
    score_t2_windows,
)

LOGGER = logging.getLogger(__name__)

GUARD_JSON = "o1prime_t2test.json"
GUARD_MD = "o1prime_t2test.md"
MISSING_SHOWN = 5  # missing sources printed in the hard-gate error


def check_sources(dataset: FeatureEvalDataset, input_dir: Path) -> None:
    """Q4 hard gate: every window's source has ``{source}.npy`` in ``input_dir``."""
    sources = sorted({dataset.slicer.source_of(w) for w in dataset.video_ids})
    missing = [s for s in sources if not (input_dir / f"{s}.npy").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(sources)} evaluation sources have no rows in {input_dir} "
            f"(first: {missing[:MISSING_SHOWN]}); re-apply the cache with its fitted "
            "statistics -- never refit it"
        )


def score_arm(
    name: str,
    ckpt: Path,
    input_dir: Path,
    data_dir: Path,
    t2_dataset: str,
    device: torch.device,
) -> dict[str, Any]:
    """O1's reads (no A0 margin yet) for one checkpoint on ``data_dir``'s evaluation windows."""
    check_sources(FeatureEvalDataset(data_dir, input_dir), input_dir)
    model, step, text_encode_fn, cfg = load_finished_model(Run(name, ckpt), device)
    check_input_manifest(input_dir, cfg.v2.crn, cfg.v2.motion)
    _, windows = score_t2_windows(model, text_encode_fn, data_dir, input_dir, t2_dataset)
    ids = sorted(windows)
    return {
        "checkpoint": str(ckpt),
        "global_step": step,
        "input_dir": str(input_dir),
        "motion": cfg.v2.motion != constants.V2_OFF,
        "arm": {"crn": cfg.v2.crn, "motion": cfg.v2.motion},
        "windows": len(ids),
        "scores": [windows[w]["y"] for w in ids],
        "labels": [windows[w]["label"] for w in ids],
    }


def verdict(
    test: dict[str, dict[str, Any]],
    val: dict[str, dict[str, Any]],
    a0: str,
    motion: dict[str, bool],
) -> dict[str, Any]:
    """Q5 from ``guardrails`` read-outs on T2-test (``test``) and T2-val (``val``)."""
    arms: dict[str, Any] = {}
    for name in test:
        if name == a0:
            arms[name] = {"reference": True, "survivor": bool(val[name]["pass"])}
            continue
        val_prime = o1_prime(val[name], val[a0])
        test_prime = o1_prime(test[name], test[a0])
        o1_val = bool(val[name]["pass"])
        rescued = val_prime["pass"] and test_prime["pass"]
        arms[name] = {
            "reference": False,
            "o1_t2_val": o1_val,
            "o1_t2_test": bool(test[name]["pass"]),
            "o1_prime_t2_val": val_prime,
            "o1_prime_t2_test": test_prime,
            "survivor": o1_val or rescued,
            "by": "O1" if o1_val else ("O1' (Amendment 8)" if rescued else None),
        }
    motion_arms = [n for n in arms if motion.get(n)]
    dropped = bool(motion_arms) and not any(arms[n]["survivor"] for n in motion_arms)
    return {"a0": a0, "arms": arms, "motion_stream_dropped": dropped}


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    names = [name for name, _, _ in args.arm]
    if args.a0 not in names:
        raise ValueError(f"--a0 {args.a0!r} is not one of the --arm names {names}")
    val_paths = {name: Path(path) for name, path in args.val_diag}
    if set(val_paths) != set(names):
        raise ValueError(f"--val-diag names {sorted(val_paths)} != --arm names {sorted(names)}")
    val = {
        n: json.loads(p.read_text(encoding="utf-8"))["guardrails"]
        for n, p in val_paths.items()
    }

    scored = {
        name: score_arm(name, Path(ckpt), Path(inp), args.data_dir, args.t2_dataset, device)
        for name, ckpt, inp in args.arm
    }
    a0_test = guardrails(scored[args.a0]["scores"], scored[args.a0]["labels"], None)
    test = {
        name: a0_test
        if name == args.a0
        else guardrails(s["scores"], s["labels"], a0_test["micro"])
        for name, s in scored.items()
    }
    out = verdict(test, val, args.a0, {n: s["motion"] for n, s in scored.items()})
    readout: dict[str, Any] = {
        "addendum": "core/docs/v2/PREREG_ADDENDUM.md §17 (Q2, Q4, Q5)",
        "data_dir": str(args.data_dir),
        "runs": {
            n: {k: v for k, v in s.items() if k not in ("scores", "labels")}
            for n, s in scored.items()
        },
        "t2_test": test,
        "t2_val": val,
        **out,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / GUARD_JSON, readout)
    write_text_atomic(args.out_dir / GUARD_MD, render_markdown(readout))
    LOGGER.info("Q5 written to %s", args.out_dir)
    return readout


def _row(name: str, g: dict[str, Any]) -> str:
    return (
        f"| {name} | {g['micro']:.4f} | {g['macro']:.4f} | {g['clip_oracle_micro']:.4f} | "
        f"{g['window_level_auc']:.4f} | {'PASS' if g['pass'] else '**FAIL**'} |"
    )


def render_markdown(r: dict[str, Any]) -> str:
    header = [
        "| arm | micro | macro | clip oracle | window AUC | O1 (registered) |",
        "|---|---|---|---|---|---|",
    ]
    lines = [
        "# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)",
        "",
        f"Evaluation windows of `{r['data_dir']}` (T2-test, unseen by every arm). "
        f"A0 = `{r['a0']}`. One seed: every Δ is printed, **not a result**.",
        "",
        "## T2-test",
        "",
        *header,
        *(_row(n, g) for n, g in r["t2_test"].items()),
        "",
        "## T2-val (batch diag.json)",
        "",
        *header,
        *(_row(n, g) for n, g in r["t2_val"].items()),
        "",
        "## Q5 verdict",
        "",
    ]
    for name, a in r["arms"].items():
        if a["reference"]:
            lines.append(
                f"* **{name}** (reference): O1 on T2-val "
                f"{'PASS' if a['survivor'] else '**FAIL**'}"
            )
            continue
        lines += [
            f"* **{name}**: O1 T2-val {'PASS' if a['o1_t2_val'] else 'FAIL'} (registered, kept)"
            f" · O1 T2-test {'PASS' if a['o1_t2_test'] else 'FAIL'} (printed)",
            f"  * T2-val  {render_o1_prime(a['o1_prime_t2_val'])}",
            f"  * T2-test {render_o1_prime(a['o1_prime_t2_test'])}",
            "  * → **"
            + (f"survivor, by {a['by']}" if a["survivor"] else "COLLAPSED: does not enter E3")
            + "**",
        ]
    lines += [
        "",
        "**Motion Stream: "
        + ("DROPPED — E3 = the surviving non-motion arms" if r["motion_stream_dropped"] else "kept")
        + "**",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="the PARENT T2 dir: its evaluation file is T2-test",
    )
    parser.add_argument(
        "--arm",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "CKPT", "INPUT_DIR"),
        help="one checkpoint and its T2 input cache (plain CLIP for A0); repeat per arm",
    )
    parser.add_argument(
        "--val-diag",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "DIAG_JSON"),
        help="the arm's batch diag.json (T2-val guardrails); one per --arm",
    )
    parser.add_argument("--a0", default="A0", help="the --arm name of the reference")
    parser.add_argument("--t2-dataset", default=constants.DADA_ORIGIN_DATASET)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
