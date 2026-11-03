"""LaGoVAD-OSTP Step 1 read-out: does a patch-only local branch (L1) add what a second host seed
(L0b) does not? DoTA-CAP-dev, protocol B. Rule: ``core/docs/final/LOCAL_STEP1.md`` §4-§5,
written before any L1 number.

* **Fusion** (fixed): per clip ``LOCAL_FUSE_WEIGHT * minmax(a) + (1 - w) * minmax(b)`` on the
  native-frame scores ``protocol_b_eval`` saved (``clip_scores.npz``). L0+L1 and L0+L0b use the
  same rule, so their difference is what L1 adds beyond a free ensemble.
* **G1** L1 alone: macro CI low > ``LOCAL_G1_FLOOR`` and mean > ``LOCAL_G1_MO_REF`` (the
  training-free raw-Mo top-K read of D-2).
* **G2** (L0+L1) - (L0+L0b), per clip, cluster bootstrap: "gain shown" (low > 0 and mean >=
  ``V5_MIN_GAIN``).
* **G3** oracle: per clip ``max(L0, L1) - max(L0, L0b)``; CI low > 0.
* Printed, never deciding: L1 - L0, (L0+L1) - L0, (L0+L0b) - L0, L0b - L0, each by ego flag;
  D2 (Spearman of per-clip AUCs and the share of clips each row wins by >= ``LOCAL_D2_MARGIN``);
  the mean per-clip Spearman of each row's scores with the cubic position prior ``p_T2`` (if
  given); training diagnostics from ``metrics.jsonl`` (``mil`` beside W0's 0.1760).
* One seed per row => every sentence is **PRELIMINARY (n = 1 seed)** and adopts nothing.

CLI::

    python -m core.tools.local_readout --l0 outputs/wu/eval/dota/dota_cap_dev/W0 \\
        --l0b outputs/local/eval/dota/dota_cap_dev/L0b \\
        --l1 outputs/local/eval/dota/dota_cap_dev/L1 \\
        --l0-run outputs/wu/W0/s2024 --l0b-run outputs/local/L0b/s2025 \\
        --l1-run outputs/local/L1/s2024 --position position_prior.npz \\
        --metadata data/DoTA/metadata_val.json --split-file data/DoTA/val_split.txt \\
        --out-dir outputs/local/REPORTS
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.dota import parse_metadata, read_split_ids
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import CLIP_SCORES_NPZ, clip_aucs, read_clip_scores
from core.tools.stp_readout import GAIN, ci, contrast, paired, run_diagnostics, sentence

LOGGER = logging.getLogger(__name__)

READOUT_NAME = "local_readout"
L0, L0B, L1 = constants.LOCAL_CONTROL_ARM, constants.LOCAL_ENSEMBLE_ARM, constants.LOCAL_ARM
FUSED_ENS, FUSED_LOC = f"{L0}+{L0B}", f"{L0}+{L1}"
LABEL = "PRELIMINARY (n = 1 seed)"
VERDICT_PASS = "STEP 1 PASS"
VERDICT_REDUNDANT = "G1 pass, G2/G3 fail: L1 learns, but adds no more than a second host seed"
VERDICT_OVERFIT = "G1 FAIL with the overfit tell: L1's mil ends below W0's"
VERDICT_WEAK = "G1 FAIL without the overfit tell: the patch input itself is weak"


def minmax(x: np.ndarray) -> np.ndarray:
    """Per-clip min-max to [0, 1]; a constant clip maps to zeros (its AUC is 0.5 either way)."""
    x = np.asarray(x, dtype=np.float64)
    span = float(x.max() - x.min()) if x.size else 0.0
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def fuse(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], weight: float = constants.LOCAL_FUSE_WEIGHT
) -> dict[str, np.ndarray]:
    """``weight * minmax(a) + (1 - weight) * minmax(b)`` per clip, over the clips both scored."""
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("the two rows share no scored clip")
    out = {}
    for v in common:
        if len(a[v]) != len(b[v]):
            raise ValueError(f"{v}: {len(a[v])} vs {len(b[v])} native frames")
        out[v] = weight * minmax(a[v]) + (1.0 - weight) * minmax(b[v])
    return out


def oracle_gain(
    base: dict[str, float], local: dict[str, float], ensemble: dict[str, float]
) -> dict[str, float]:
    """G3 per clip: ``max(base, local) - max(base, ensemble)`` over the clips all three scored."""
    common = sorted(set(base) & set(local) & set(ensemble))
    return {v: max(base[v], local[v]) - max(base[v], ensemble[v]) for v in common}


def average_rank(x: np.ndarray) -> np.ndarray:
    """Ranks 0..n-1 with ties given their mean rank."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    mean_rank: np.ndarray = (sums / counts)[inverse]
    return mean_rank


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman's rho; ``None`` when either side is constant."""
    ra, rb = average_rank(a), average_rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def d2_reads(row: dict[str, float], base: dict[str, float]) -> dict[str, Any]:
    """D2: per-clip AUC Spearman with ``base`` and the share of clips each side wins clearly."""
    common = sorted(set(row) & set(base))
    a = np.array([row[v] for v in common])
    b = np.array([base[v] for v in common])
    margin = constants.LOCAL_D2_MARGIN
    return {
        "clips": len(common),
        "spearman": spearman(a, b),
        "row_wins": float(np.mean(a - b >= margin)) if common else None,
        "base_wins": float(np.mean(b - a >= margin)) if common else None,
    }


def position_rho(
    scores: dict[str, np.ndarray], prior: dict[str, np.ndarray]
) -> dict[str, Any] | None:
    """Mean per-clip Spearman of a row's native scores with ``p_T2`` (printed only)."""
    rhos = []
    for v in sorted(set(scores) & set(prior)):
        if len(scores[v]) != len(prior[v]):
            raise ValueError(f"{v}: {len(scores[v])} scores vs {len(prior[v])} prior frames")
        rho = spearman(scores[v], prior[v])
        if rho is not None:
            rhos.append(rho)
    if not rhos:
        return None
    return {"mean": float(np.mean(rhos)), "clips": len(rhos)}


def verdict(g1: bool, g2: bool, g3: bool, l1_mil_last: float | None) -> str:
    """LOCAL_STEP1.md §5's decision table."""
    if g1 and g2 and g3:
        return VERDICT_PASS
    if g1:
        return VERDICT_REDUNDANT
    if l1_mil_last is not None and l1_mil_last < constants.LOCAL_W0_MIL_LAST:
        return VERDICT_OVERFIT
    return VERDICT_WEAK


def _last(diag: dict[str, Any] | None, key: str) -> float | None:
    if diag is None or key not in diag:
        return None
    return float(diag[key]["last"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = {L0: args.l0, L0B: args.l0b, L1: args.l1}
    scores: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, np.ndarray] | None = None
    for name, d in dirs.items():
        scores[name], lab = read_clip_scores(Path(d) / CLIP_SCORES_NPZ)
        if labels is None:
            labels = lab
        elif set(lab) != set(labels) or any(not np.array_equal(lab[v], labels[v]) for v in lab):
            raise ValueError(f"{name}: labels differ from {L0}'s -- not the same clip set")
    assert labels is not None  # nosec B101 - three rows were read
    scores[FUSED_ENS] = fuse(scores[L0], scores[L0B])
    scores[FUSED_LOC] = fuse(scores[L0], scores[L1])
    aucs = {name: clip_aucs(s, {v: labels[v] for v in s}) for name, s in scores.items()}

    records = parse_metadata(args.metadata, read_split_ids(args.split_file))
    ego = {r.video_id: r.is_ego for r in records}

    l1_macro = ci(aucs[L1])
    g1 = bool(
        l1_macro is not None
        and l1_macro["low"] > constants.LOCAL_G1_FLOOR
        and l1_macro["mean"] > constants.LOCAL_G1_MO_REF
    )
    g2_ci = ci(paired(aucs[FUSED_LOC], aucs[FUSED_ENS]))
    g2 = sentence(g2_ci) == GAIN
    g3_ci = ci(oracle_gain(aucs[L0], aucs[L1], aucs[L0B]))
    g3 = bool(g3_ci is not None and g3_ci["low"] > 0)

    runs = {L0: args.l0_run, L0B: args.l0b_run, L1: args.l1_run}
    diagnostics = {
        name: run_diagnostics(Path(d)) for name, d in runs.items() if d is not None
    }
    prior = read_clip_scores(args.position)[0] if args.position is not None else None

    result: dict[str, Any] = {
        "rule": "core/docs/final/LOCAL_STEP1.md §4-§5",
        "set": "DoTA-CAP (n/1397) dev, protocol B",
        "label": LABEL,
        "fusion": f"{constants.LOCAL_FUSE_WEIGHT} minmax(a) + "
        f"{1 - constants.LOCAL_FUSE_WEIGHT} minmax(b), per clip, native frames",
        "rows": {name: ci(a) for name, a in aucs.items()},
        "gates": {
            "G1": {"ci": l1_macro, "floor": constants.LOCAL_G1_FLOOR,
                   "mo_ref": constants.LOCAL_G1_MO_REF, "pass": g1},
            "G2": {"ci": g2_ci, "sentence": sentence(g2_ci), "pass": g2},
            "G3": {"ci": g3_ci, "pass": g3},
        },
        "contrasts": {
            f"{FUSED_LOC} - {FUSED_ENS} (G2)": contrast(aucs[FUSED_LOC], aucs[FUSED_ENS], ego),
            f"{L1} - {L0}": contrast(aucs[L1], aucs[L0], ego),
            f"{FUSED_LOC} - {L0}": contrast(aucs[FUSED_LOC], aucs[L0], ego),
            f"{FUSED_ENS} - {L0}": contrast(aucs[FUSED_ENS], aucs[L0], ego),
            f"{L0B} - {L0} (seed spread)": contrast(aucs[L0B], aucs[L0], ego),
        },
        "d2": {L1: d2_reads(aucs[L1], aucs[L0]), L0B: d2_reads(aucs[L0B], aucs[L0])},
        "position_rho": (
            {name: position_rho(scores[name], prior) for name in (L0, L0B, L1)}
            if prior is not None else None
        ),
        "p_T2_cap_dev": constants.V5_P_T2_CAP_DEV,
        "diagnostics": diagnostics,
        "w0_mil_last": constants.LOCAL_W0_MIL_LAST,
        "verdict": verdict(g1, g2, g3, _last(diagnostics.get(L1), "mil")),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / f"{READOUT_NAME}.json", result)
    write_text_atomic(args.out_dir / f"{READOUT_NAME}.md", render_markdown(result))
    return result


def _ci(c: dict[str, float] | None, signed: bool = True) -> str:
    if c is None:
        return "—"
    f = "+.4f" if signed else ".4f"
    return f"{c['mean']:{f}} [{c['low']:{f}}, {c['high']:{f}}] (n={c['clips']})"


def _opt(x: float | None, f: str = ".3f") -> str:
    return "—" if x is None else f"{x:{f}}"


def _diag(d: dict[str, Any] | None, key: str) -> str:
    if d is None or key not in d:
        return "—"
    return f"{d[key]['first']:.4f} → {d[key]['last']:.4f}"


def render_markdown(r: dict[str, Any]) -> str:
    g = r["gates"]
    lines = [
        f"# LaGoVAD-OSTP Step 1 read-out — {r['set']} — **{r['label']}**",
        "",
        f"Rule: `{r['rule']}`. Fusion: {r['fusion']}.",
        "",
        f"## Verdict: **{r['verdict']}** — {r['label']}",
        "",
        "| gate | read | bar | pass |",
        "|---|---|---|---|",
        f"| G1 L1 alone | {_ci(g['G1']['ci'], signed=False)} | low > {g['G1']['floor']}, "
        f"mean > {g['G1']['mo_ref']} | {g['G1']['pass']} |",
        f"| G2 ({FUSED_LOC}) - ({FUSED_ENS}) | {_ci(g['G2']['ci'])} | low > 0, mean ≥ "
        f"{constants.V5_MIN_GAIN} | {g['G2']['pass']} |",
        f"| G3 oracle max({L0},{L1}) - max({L0},{L0B}) | {_ci(g['G3']['ci'])} | low > 0 | "
        f"{g['G3']['pass']} |",
        "",
        "| row | macro [95 % cluster CI] |",
        "|---|---|",
        *(f"| {name} | {_ci(c, signed=False)} |" for name, c in r["rows"].items()),
        "",
        f"Printed: `p_T2` (cubic position prior, no pixels) on CAP-dev = {r['p_T2_cap_dev']:.4f}.",
        "",
        "| contrast (printed) | Δ all | sentence | Δ ego | Δ non-ego |",
        "|---|---|---|---|---|",
        *(
            f"| {name} | {_ci(c['all'])} | {c['sentence']} — {r['label']} | {_ci(c['ego'])} | "
            f"{_ci(c['non_ego'])} |"
            for name, c in r["contrasts"].items()
        ),
        "",
        f"D2 (printed): per-clip AUC vs {L0}; a clip is won at a lead ≥ "
        f"{constants.LOCAL_D2_MARGIN}.",
        "",
        f"| row | clips | Spearman with {L0} | row wins | {L0} wins |",
        "|---|---|---|---|---|",
        *(
            f"| {name} | {d['clips']} | {_opt(d['spearman'])} | {_opt(d['row_wins'])} | "
            f"{_opt(d['base_wins'])} |"
            for name, d in r["d2"].items()
        ),
    ]
    if r["position_rho"] is not None:
        lines += [
            "",
            "Mean per-clip Spearman of the native scores with `p_T2` (printed; high = the row "
            "tracks position):",
            "",
            "| row | rho | clips |",
            "|---|---|---|",
            *(
                f"| {name} | {_opt(p['mean'] if p else None)} | {p['clips'] if p else '—'} |"
                for name, p in r["position_rho"].items()
            ),
        ]
    if r["diagnostics"]:
        lines += [
            "",
            f"Training diagnostics (first → last logged batch; W0 s2024 mil last = "
            f"{r['w0_mil_last']}):",
            "",
            "| run | step | mil | SA² entropy / log K | ‖W_q‖ | rho_u |",
            "|---|---|---|---|---|---|",
            *(
                f"| {name} | {d['last_step'] if d else '—'} | {_diag(d, 'mil')} | "
                f"{_diag(d, 'stp_entropy')} | {_diag(d, 'stp_wq_norm')} | "
                f"{_diag(d, 'motion_share')} |"
                for name, d in r["diagnostics"].items()
            ),
        ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--l0", type=Path, required=True, help=f"{L0}'s protocol_b_eval out dir")
    parser.add_argument("--l0b", type=Path, required=True, help=f"{L0B}'s protocol_b_eval out dir")
    parser.add_argument("--l1", type=Path, required=True, help=f"{L1}'s protocol_b_eval out dir")
    parser.add_argument("--l0-run", type=Path, default=None, help=f"{L0}'s run dir (metrics.jsonl)")
    parser.add_argument("--l0b-run", type=Path, default=None, help=f"{L0B}'s run dir")
    parser.add_argument("--l1-run", type=Path, default=None, help=f"{L1}'s run dir")
    parser.add_argument("--position", type=Path, default=None,
                        help="position_prior.npz of p_T2 on the same split (printed only)")
    parser.add_argument("--metadata", type=Path, required=True, help="DoTA metadata_val.json")
    parser.add_argument("--split-file", type=Path, required=True, help="DoTA val_split.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run(build_arg_parser().parse_args(argv))
    LOGGER.info("verdict: %s", result["verdict"])


if __name__ == "__main__":
    main()
