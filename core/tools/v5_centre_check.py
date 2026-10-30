"""v5 check C: does a stored change map point at the agent beyond a centre prior? (PREREG_V5 §3.7).

Training-free and CPU-only. Reads Stage 0's stored DoTA-CAP-dev change maps (``maps/<id>.npz``:
``dc14`` CLIP 14 x 14 patches, ``d7`` / ``d14`` VideoMAE V2-S) and, per clip, scores the same
pointing read as S2-8 (:func:`core.tools.v5_stage0.clip_read`: the most-changed tile hits a box)
for the map **and** for the centre prior on the **same grid and the same defined steps**:

* primary -- ``C14`` (native 14 x 14): ``hit - hit(centre prior)``, 95 % cluster bootstrap over DoTA
  source videos. **CLIP-BEATS-CENTRE** iff the low > 0, else **CENTRE-WINS**.
* printed -- ``C14`` 2 x 2-mean pooled to 7 x 7 (the grid of the v5.1 pilot's centre number),
  ``V7`` and ``V14``; hit - chance of the map and of the centre prior; mass-in-box minus the
  centre prior's; ``hit - centre`` by ego and by box-share bin.

Validity gate **G-C0**: ``C14`` hit - chance reproduces Stage 0's recorded value (``stage0_readout
.json``) within ``V5_G5_MAX_ABS``. On a 14 x 14 grid the centre prior's four middle tiles tie and
``argmax`` takes the first, tile (6, 6) -- disclosed; the pooled 7 x 7 read has one centre tile.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools import v5_stage0 as s0
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.v5_maps import centre_prior, clip_mass

LOGGER = logging.getLogger(__name__)

RULE = "core/docs/v5/PREREG_V5.md §3.7 (check C)"
READOUT_JSON = "check_c_readout.json"
READOUT_MD = "check_c_readout.md"
PRIMARY = "C14"
STAGE0_C14_KEY = "dc14"  # Stage 0 maps: the CLIP 14 x 14 lag-1 change map
REPS = (  # name -> (stored key, pooling factor); the primary first
    (PRIMARY, STAGE0_C14_KEY, 1),
    ("C14->7", STAGE0_C14_KEY, constants.V5_CHECK_C_POOL),
    ("V7", "d7", 1),
    ("V14", "d14", 1),
)
BEATS = "CLIP-BEATS-CENTRE"
LOSES = "CENTRE-WINS"
INVALID = "INVALID"


def pool_map(delta: np.ndarray, factor: int) -> np.ndarray:
    """``(K, g*g)`` -> ``(K, (g/f)^2)``: mean over ``f x f`` tile blocks; NaN rows stay NaN."""
    if factor == 1:
        return delta
    grid = math.isqrt(delta.shape[1])
    if grid % factor:
        raise ValueError(f"grid {grid} is not divisible by {factor}")
    g = grid // factor
    out: np.ndarray = delta.reshape(len(delta), g, factor, g, factor).mean((2, 4)).reshape(
        len(delta), -1
    )
    return out


def matched_prior(delta: np.ndarray) -> np.ndarray:
    """The centre prior tiled over ``delta``'s steps, NaN where ``delta`` is undefined (paired)."""
    prior = np.tile(centre_prior(math.isqrt(delta.shape[1])), (len(delta), 1))
    prior[~np.isfinite(delta).all(1)] = np.nan
    return prior


def clip_reads(
    delta: np.ndarray,
    steps: np.ndarray,
    labels: np.ndarray,
    boxes: dict[int, list[list[float]]],
) -> dict[str, float] | None:
    """One clip: the map's hit / chance / mass and the centre prior's on the same steps."""
    grid = math.isqrt(delta.shape[1])
    topk = constants.V5_TOPK.get(grid, constants.V5_TOPK[constants.V5_GRID_COARSE])
    zeros = np.zeros(len(delta))
    read = s0.clip_read(delta, zeros, steps, labels, boxes, topk)
    if read.hit is None or read.chance is None:
        return None
    prior = matched_prior(delta)
    centre = s0.clip_read(prior, zeros, steps, labels, boxes, topk)
    out = {"hit": read.hit, "chance": read.chance, "hit_centre": float(centre.hit or 0.0)}
    mass = clip_mass(delta, steps, labels, boxes)
    centre_mass = clip_mass(prior, steps, labels, boxes)
    if mass is not None and centre_mass is not None:
        out |= {"mass": mass[0], "mass_chance": mass[1], "mass_centre": centre_mass[0]}
    return out


def summarize(
    per_clip: dict[str, dict[str, float]], ego: dict[str, bool], bins: dict[str, str | None]
) -> dict[str, Any]:
    """Every interval of one representation (cluster bootstrap, S0-9)."""
    vs_centre = {v: r["hit"] - r["hit_centre"] for v, r in per_clip.items()}
    massed = {v: r for v, r in per_clip.items() if "mass" in r}
    return {
        "clips": len(per_clip),
        "hit_minus_centre": s0.ci(vs_centre),
        "hit_minus_chance": s0.ci({v: r["hit"] - r["chance"] for v, r in per_clip.items()}),
        "centre_minus_chance": s0.ci(
            {v: r["hit_centre"] - r["chance"] for v, r in per_clip.items()}
        ),
        "mass_minus_centre": s0.ci({v: r["mass"] - r["mass_centre"] for v, r in massed.items()}),
        "mass_minus_chance": s0.ci({v: r["mass"] - r["mass_chance"] for v, r in massed.items()}),
        "by_ego": {
            lab: s0.ci({v: x for v, x in vs_centre.items() if ego.get(v) == flag})
            for flag, lab in ((True, "ego"), (False, "non_ego"))
        },
        "by_bin": {
            b: s0.ci({v: x for v, x in vs_centre.items() if bins.get(v) == b})
            for b in s0.BIN_NAMES
        },
    }


def g_c0(
    summary: dict[str, Any], reference: dict[str, Any], tol: float = constants.V5_G5_MAX_ABS
) -> dict[str, Any]:
    """G-C0: the primary map's hit - chance reproduces Stage 0's recorded C14 row (``tol``:
    1e-4 on Stage 0's own maps; ``V5_R7_REPRO_TOL`` on an R7 cache, PREREG_V5 §4.5)."""
    got = summary["hit_minus_chance"]
    ref = reference["summary"][PRIMARY]["hit_minus_chance"]
    diff = math.inf if got is None else abs(got["mean"] - ref["mean"])
    return {"got": got, "ref": ref["mean"], "diff": diff, "tol": tol, "pass": diff <= tol}


def use_r7_grids(inputs: s0.DevInputs, grid_dir: Path) -> None:
    """R7 G-R7c (PREREG_V5 §4.5): replace each clip's C14 map by the lag-1 change map of the
    stored R7 CLIP grid (same steps), so check C runs on the R7 cache."""
    for v, m in inputs.maps.items():
        grid = np.load(grid_dir / f"{v}.npy").astype(np.float32)
        if len(grid) != len(m["steps"]):
            raise ValueError(f"{v}: R7 grid has {len(grid)} rows, Stage 0 read {len(m['steps'])}")
        m[STAGE0_C14_KEY] = s0.change_map(grid)


def decide(primary: dict[str, Any], valid: bool) -> str:
    if not valid:
        return INVALID
    stat = primary["hit_minus_centre"]
    return BEATS if stat is not None and stat["low"] > 0 else LOSES


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    inputs = s0.load_dev(args)
    r7_dir: Path | None = getattr(args, "r7_grid_dir", None)
    r7 = r7_dir is not None
    if r7_dir is not None:
        use_r7_grids(inputs, r7_dir)
    summary: dict[str, Any] = {}
    for name, key, factor in REPS:
        per_clip: dict[str, dict[str, float]] = {}
        for v, m in inputs.maps.items():
            delta = pool_map(m[key], factor)
            read = clip_reads(delta, m["steps"], inputs.labels[v], inputs.boxes[v])
            if read is not None:
                per_clip[v] = read
        summary[name] = summarize(per_clip, inputs.ego, inputs.bins)
    reference = json.loads(args.stage0_readout.read_text(encoding="utf-8"))
    tol = constants.V5_R7_REPRO_TOL if r7 else constants.V5_G5_MAX_ABS
    gate = g_c0(summary[PRIMARY], reference, tol)
    valid = gate["pass"] and not inputs.failed and not inputs.missing
    out = {
        "rule": RULE if not r7 else f"{RULE}; run on the R7 cache {r7_dir} (§4.5)",
        "r7_grid_dir": str(r7_dir) if r7 else None,
        "decision": decide(summary[PRIMARY], valid),
        "g_c0": gate,
        "validity": inputs.validity(),
        "summary": summary,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / READOUT_JSON, out)
    write_text_atomic(args.out_dir / READOUT_MD, render(out))
    return out


def _f(stat: dict[str, float] | None) -> str:
    if stat is None:
        return "—"
    return f"{stat['mean']:+.4f} [{stat['low']:+.4f}, {stat['high']:+.4f}]"


def render(out: dict[str, Any]) -> str:
    valid, gate = out["validity"], out["g_c0"]
    lines = [
        "# v5 check C -- change maps vs the centre prior on DoTA-CAP-dev boxes",
        "",
        f"Rule: {out['rule']}. Training-free; the centre prior is read on each map's own grid "
        "and steps.",
        "",
        f"**Decision: {out['decision']}** (primary: {PRIMARY} hit - centre, 95 % low > 0).",
        "",
        f"Validity: dev maps {valid['dev_maps']}, failed {len(valid['failed'])}, missing "
        f"{len(valid['missing_dev'])}; G-C0 {PRIMARY} hit - chance {_f(gate['got'])} vs Stage 0 "
        f"{gate['ref']:+.4f} (|Δ| {gate['diff']:.1e}, "
        f"tol {gate.get('tol', constants.V5_G5_MAX_ABS):g}) -> "
        f"{'PASS' if gate['pass'] else 'FAIL'}",
        "",
        "| map | clips | hit - centre | hit - chance | centre - chance | mass - centre "
        "| mass - chance |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, st in out["summary"].items():
        lines.append(
            f"| {name}{' (primary)' if name == PRIMARY else ''} | {st['clips']} "
            f"| {_f(st['hit_minus_centre'])} | {_f(st['hit_minus_chance'])} "
            f"| {_f(st['centre_minus_chance'])} | {_f(st['mass_minus_centre'])} "
            f"| {_f(st['mass_minus_chance'])} |"
        )
    lines += ["", "hit - centre by ego / box-share bin (printed):", ""]
    for name, st in out["summary"].items():
        ego = ", ".join(f"{k} {_f(v)}" for k, v in st["by_ego"].items())
        bins = ", ".join(f"{k} {_f(v)}" for k, v in st["by_bin"].items())
        lines.append(f"- {name}: {ego}; {bins}")
    lines += [
        "",
        f"{PRIMARY} (14 x 14): the centre prior's four middle tiles tie; argmax takes tile (6, 6).",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    s0.add_read_args(parser)
    parser.add_argument(
        "--stage0-readout", type=Path, required=True, help="Stage 0's stage0_readout.json (G-C0)"
    )
    parser.add_argument(
        "--r7-grid-dir", type=Path, default=None,
        help="R7 G-R7c (PREREG_V5 §4.5): read C14 from this CLIP 14x14 cache instead of dc14",
    )
    parser.set_defaults(
        out_dir=constants.OUTPUT_ROOT
        / constants.V5_OUTPUT_DIRNAME
        / constants.V5_CHECK_C_DIRNAME
        / "REPORTS"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = run_check(build_arg_parser().parse_args(argv))
    LOGGER.info("check C decision: %s", out["decision"])


if __name__ == "__main__":
    main()
