"""Is the CRN arms' cost on Nexar the CRN reference unit? (addendum §30, Amendment 19).

A diagnostic over §29's Nexar-trained arms; it decides no arm and never moves ``B``.

* ``share`` (C7, printed only) -- per-clip CRN cost (``A3 - A2``, ``A1 - A0``, ``A3ign - A2``)
  against the clip's event share, i.e. how much of the clip's own CRN reference is the event.
  Reads the saved §29 scores: DoTA-CAP dev / eval and DADA whole (``clip_scores.npz``), TAD
  (per-seed score files, seed-averaged), Nexar val / test (``item_aucs.json``; whole vs crops).
* ``ref`` (C4-C6) -- ``nexar_eval --crn-reference source`` against §29's ``item`` reads of
  ``nexar_val``: the parity gate, the paired Δ_ref of A3's crop endpoint, the fraction ``f`` of
  the A2 - A3 gap it closes, and C6's sentence.

CLI::

    python -m core.tools.crn_cost_diag share --root outputs/v2_nexar_window \\
        --census .../nexar_n0/census.json --out-dir outputs/v2_nexar_window/REPORTS/crn_diag
    python -m core.tools.crn_cost_diag ref --item-dir outputs/v2_nexar_window/val \\
        --source-dir outputs/v2_nexar_window/val_crnref --out-dir .../REPORTS/crn_diag
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.metrics import frame_auc
from core.tools.e3_readout import paired_interval
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_build import native_labels
from core.tools.nexar_eval import CROP_SEP, ITEM_AUCS_JSON, READOUT_JSON, crop_bounds

LOGGER = logging.getLogger(__name__)

SHARE_JSON = "crn_share.json"
SHARE_MD = "crn_share.md"
REF_JSON = "crn_ref.json"
REF_MD = "crn_ref.md"
SHARE_CONTRASTS = (constants.CRN_DIAG_PRIMARY, *constants.CRN_DIAG_PRINTED)
DOTA_SPLITS = ("dota_cap_dev", "dota_cap_eval")
CLIP_SCORES = "clip_scores.npz"

ClipRead = dict[str, tuple[float, float]]  # clip -> (AUC, event share)


# ---------------------------------------------------------------------------
# share (C7)
# ---------------------------------------------------------------------------
def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties sharing their mean rank."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    for value in np.unique(x):
        tied = x == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def two_class(score: np.ndarray, label: np.ndarray) -> tuple[float, float] | None:
    if not 0 < int(label.sum()) < len(label):
        return None
    return frame_auc(score, label), float(label.mean())


def read_npz(path: Path) -> ClipRead:
    """``clip_scores.npz`` (``score/<clip>``, ``label/<clip>``) -> per two-class clip AUC, share."""
    out: ClipRead = {}
    with np.load(path) as data:
        for key in data.files:
            if key.startswith("score/"):
                clip = key.split("/", 1)[1]
                read = two_class(data[key], data[f"label/{clip}"])
                if read is not None:
                    out[clip] = read
    return out


def read_tad(arm_dir: Path, seeds: tuple[int, ...]) -> ClipRead:
    """Seed-averaged TAD scores (``s<seed>/scores/<video>.npz``: ``score``, ``gt``)."""
    out: ClipRead = {}
    first = arm_dir / f"s{seeds[0]}" / "scores"
    for path in sorted(first.glob("*.npz")):
        with np.load(path) as data:
            gt = data["gt"]
        score = np.mean([np.load(arm_dir / f"s{s}" / "scores" / path.name)["score"]
                         for s in seeds], axis=0)
        read = two_class(score, gt)
        if read is not None:
            out[path.stem] = read
    return out


def nexar_shares(items: list[str], rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Event share of each Nexar item (whole video or ``<video>@<placement>`` crop).

    Frames = the census's decoded count, else its header count (the census decodes only a probe
    sample; header and decode agree to a frame there, and the share is printed only).
    """
    out: dict[str, float] = {}
    for item in items:
        video, _, placement = item.partition(CROP_SEP)
        row = rows[video]
        frames = int(row.get("frames_decoded", row["frames_header"]))
        labels = native_labels(row, frames)
        if placement:
            bounds = crop_bounds(row, frames, float(placement))
            if bounds is None:
                raise ValueError(f"{item}: the crop does not fit its video")
            labels = labels[bounds[0] : bounds[1]]
        out[item] = float(labels.mean())
    return out


def read_nexar(path: Path, rows: dict[str, dict[str, Any]]) -> ClipRead:
    aucs: dict[str, float | None] = json.loads(path.read_text(encoding="utf-8"))
    kept = {k: float(v) for k, v in aucs.items() if v is not None}
    shares = nexar_shares(sorted(kept), rows)
    return {k: (kept[k], shares[k]) for k in kept}


def cost_table(a: ClipRead, b: ClipRead) -> dict[str, Any]:
    """Per-clip ``a - b`` against the event share: mean, Spearman, share-quartile means."""
    clips = sorted(set(a) & set(b))
    if not clips:
        raise ValueError("the two arms share no two-class clip")
    delta = np.array([a[c][0] - b[c][0] for c in clips])
    share = np.array([a[c][1] for c in clips])
    edges = np.quantile(share, np.linspace(0, 1, constants.CRN_DIAG_SHARE_QUARTILES + 1))
    bins = np.clip(np.searchsorted(edges[1:-1], share, side="right"), 0,
                   constants.CRN_DIAG_SHARE_QUARTILES - 1)
    return {
        "clips": len(clips),
        "mean_delta": float(delta.mean()),
        "median_share": float(np.median(share)),
        "spearman_share_delta": spearman(share, delta),
        "share_edges": [float(e) for e in edges],
        "delta_by_share_bin": [
            {"clips": int((bins == i).sum()),
             "mean_delta": float(delta[bins == i].mean()) if (bins == i).any() else None}
            for i in range(constants.CRN_DIAG_SHARE_QUARTILES)
        ],
    }


def split_nexar(read: ClipRead) -> dict[str, ClipRead]:
    return {
        "whole": {k: v for k, v in read.items() if CROP_SEP not in k},
        "crops": {k: v for k, v in read.items() if CROP_SEP in k},
    }


def run_share(args: argparse.Namespace) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = json.loads(
        args.census.read_text(encoding="utf-8"))["videos"]
    root: Path = args.root
    seeds = tuple(args.seeds)
    arms = sorted({a for pair in SHARE_CONTRASTS for a in pair})
    sets: dict[str, dict[str, ClipRead]] = {}
    for split in DOTA_SPLITS:
        sets[split] = {a: read_npz(root / "eval" / "dota" / a / split / CLIP_SCORES) for a in arms}
    sets["dada_whole"] = {a: read_npz(root / "eval" / "dada" / a / CLIP_SCORES) for a in arms}
    sets["tad"] = {a: read_tad(root / "eval" / "tad" / a, seeds) for a in arms}
    nexar = {
        "nexar_val": {a: read_nexar(root / "val" / a / ITEM_AUCS_JSON, rows) for a in arms},
        "nexar_test": {a: read_nexar(root / "eval" / "nexar_test" / "nexar" / a / ITEM_AUCS_JSON,
                                     rows) for a in arms},
    }
    for name, by_arm in nexar.items():
        for part in ("whole", "crops"):
            sets[f"{name}_{part}"] = {a: split_nexar(r)[part] for a, r in by_arm.items()}
    tables = {
        name: {f"{x} - {y}": cost_table(by_arm[x], by_arm[y]) for x, y in SHARE_CONTRASTS}
        for name, by_arm in sets.items()
    }
    out = {"amendment": "§30 C7 (printed only)", "root": str(root), "seeds": list(seeds),
           "tables": tables}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / SHARE_JSON, out)
    write_text_atomic(args.out_dir / SHARE_MD, render_share(out))
    LOGGER.info("share table -> %s", args.out_dir)
    return out


def render_share(r: dict[str, Any]) -> str:
    lines = [
        "# CRN cost vs event share — addendum §30 C7 (printed only)",
        "",
        "Per two-class clip: Δ = AUC(arm with CRN) - AUC(its no-CRN twin), Nexar-trained (§29). "
        "Share = the clip's event share, i.e. how much of the clip's own CRN reference is the "
        "event. H1 predicts the cost grows with the share "
        "(Spearman < 0, the high bin most negative).",
        "",
        "| set | contrast | clips | mean Δ | median share | Spearman(share, Δ) "
        "| Δ by share quartile |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for name, tables in r["tables"].items():
        for contrast, t in tables.items():
            rho = "—" if t["spearman_share_delta"] is None else f"{t['spearman_share_delta']:+.3f}"
            bins = " / ".join("—" if b["mean_delta"] is None else f"{b['mean_delta']:+.3f}"
                              for b in t["delta_by_share_bin"])
            lines.append(f"| {name} | {contrast} | {t['clips']} | {t['mean_delta']:+.4f} | "
                         f"{t['median_share']:.3f} | {rho} | {bins} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# ref (C4-C6)
# ---------------------------------------------------------------------------
def load_read(arm_dir: Path) -> tuple[dict[str, Any], dict[str, float | None]]:
    readout = json.loads((arm_dir / READOUT_JSON).read_text(encoding="utf-8"))
    aucs = json.loads((arm_dir / ITEM_AUCS_JSON).read_text(encoding="utf-8"))
    return readout, aucs


def parity_gate(
    item: tuple[dict[str, Any], dict[str, float | None]],
    source: tuple[dict[str, Any], dict[str, float | None]],
) -> list[str]:
    """C4: failures (empty = pass). Steps, the item set, and whole items equal on both paths."""
    failures: list[str] = []
    (r_item, a_item), (r_src, a_src) = item, source
    if r_src.get("crn_reference") != constants.CRN_REFERENCE_SOURCE:
        failures.append(f"source read has crn_reference={r_src.get('crn_reference')!r}")
    if r_src["split"] != constants.V2_SPLIT_NEXAR_VAL or r_item["split"] != r_src["split"]:
        failures.append(f"splits {r_item['split']} / {r_src['split']} (C2: nexar_val only)")
    # C4(a): the same finished checkpoints (J10 itself is asserted by the notebook from the plan)
    if r_item["global_steps"] != r_src["global_steps"]:
        failures.append(f"global steps {r_item['global_steps']} vs {r_src['global_steps']}")
    if r_item["checkpoints"] != r_src["checkpoints"]:
        failures.append("the two reads scored different checkpoints")
    if set(a_item) != set(a_src):
        failures.append(f"item sets differ ({len(a_item)} vs {len(a_src)})")
    for name in sorted(n for n in a_item if CROP_SEP not in n):
        x, y = a_item[name], a_src.get(name)
        if (x is None) != (y is None) or (
                x is not None and y is not None and abs(x - y) > constants.CRN_DIAG_PARITY_TOL):
            failures.append(f"whole item {name}: item {x} vs source {y}")
    return failures


def seed_mean(readout: dict[str, Any]) -> float:
    return float(np.mean(list(readout["per_run_crop_endpoint"].values())))


def reference_effect(
    item: dict[str, Any], source: dict[str, Any], twin: dict[str, Any]
) -> dict[str, Any]:
    """C5: paired Δ_ref of the per-seed crop endpoint and the fraction of the twin gap it closes."""
    interval = paired_interval(source["per_run_crop_endpoint"], item["per_run_crop_endpoint"],
                               constants.V2_E3_T95)
    gap = seed_mean(twin) - seed_mean(item)
    return {
        "endpoint_item": seed_mean(item),
        "endpoint_source": seed_mean(source),
        "endpoint_twin": seed_mean(twin),
        "gap_twin_minus_item": gap,
        "delta_ref": interval,
        "fraction_closed": None if gap <= 0 else interval["mean"] / gap,
        "per_placement": {
            str(p): {
                path: r["crops"][str(p)]["macro"]["mean"]
                for path, r in (("item", item), ("source", source), ("twin", twin))
            }
            for p in constants.NEXAR_CROP_PLACEMENTS
        },
    }


def ref_sentence(effect: dict[str, Any]) -> str:
    """C6, checked in order; no tie clause."""
    low, high = effect["delta_ref"]["low"], effect["delta_ref"]["high"]
    f = effect["fraction_closed"]
    if low > 0 and f is not None and f >= constants.CRN_DIAG_MOST:
        return "the reference unit explains most of CRN's crop cost on Nexar"
    if low > 0:
        return "the reference unit explains part of it"
    if high < 0:
        return ("the training reference costs more on crops; "
                "the reference unit does not explain the cost")
    return "the reference unit does not explain CRN's crop cost (no change shown at n = 5)"


def run_ref(args: argparse.Namespace) -> dict[str, Any]:
    pairs = (constants.CRN_DIAG_PRIMARY, *constants.CRN_DIAG_PRINTED)
    gate: dict[str, list[str]] = {}
    effects: dict[str, dict[str, Any]] = {}
    for arm, twin in pairs:
        item, source = load_read(args.item_dir / arm), load_read(args.source_dir / arm)
        gate[arm] = parity_gate(item, source)
        if not gate[arm]:
            twin_read, _ = load_read(args.item_dir / twin)
            effects[f"{arm} (twin {twin})"] = reference_effect(item[0], source[0], twin_read)
    failed = {a: f for a, f in gate.items() if f}
    if failed:
        raise SystemExit(f"§30 C4 gate failed, nothing is printed: {failed}")
    primary_arm, primary_twin = constants.CRN_DIAG_PRIMARY
    primary = effects[f"{primary_arm} (twin {primary_twin})"]
    out = {
        "amendment": "§30 (C4-C7)",
        "item_dir": str(args.item_dir),
        "source_dir": str(args.source_dir),
        "gate": "pass",
        "primary": primary_arm,
        "sentence": ref_sentence(primary),
        "effects": effects,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / REF_JSON, out)
    write_text_atomic(args.out_dir / REF_MD, render_ref(out))
    LOGGER.info("§30: %s -> %s", out["sentence"], args.out_dir)
    return out


def _iv(c: dict[str, Any]) -> str:
    return f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def render_ref(r: dict[str, Any]) -> str:
    lines = [
        "# CRN reference unit on Nexar val — addendum §30",
        "",
        f"Gate C4: **{r['gate']}**. Primary arm **{r['primary']}**.",
        "",
        f"**Sentence (C6): {r['sentence']}**",
        "",
        "| arm | endpoint item | endpoint source | twin | Δ_ref paired t95 "
        "| f (share of twin gap) |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for name, e in r["effects"].items():
        f = "—" if e["fraction_closed"] is None else f"{e['fraction_closed']:.2f}"
        lines.append(f"| {name} | {e['endpoint_item']:.4f} | {e['endpoint_source']:.4f} | "
                     f"{e['endpoint_twin']:.4f} | {_iv(e['delta_ref'])} | {f} |")
    header = " | ".join(str(p) for p in constants.NEXAR_CROP_PLACEMENTS)
    lines += ["", "Printed (C7): crop macro per placement, seed-averaged scores.", "",
              f"| arm | path | {header} |",
              "|---|---|" + "---:|" * len(constants.NEXAR_CROP_PLACEMENTS)]
    for name, e in r["effects"].items():
        for path in ("item", "source", "twin"):
            vals = " | ".join(f"{e['per_placement'][str(p)][path]:.4f}"
                              for p in constants.NEXAR_CROP_PLACEMENTS)
            lines.append(f"| {name} | {path} | {vals} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    share = sub.add_parser("share", help="C7: per-clip CRN cost vs event share (printed)")
    share.add_argument("--root", type=Path, required=True, help="§29's NEXAR_FULL_RUNS_DIR")
    share.add_argument("--census", type=Path, required=True)
    share.add_argument("--seeds", type=int, nargs="+", default=list(constants.NEXAR_RUN_SEEDS))
    share.add_argument("--out-dir", type=Path, required=True)
    share.set_defaults(func=run_share)
    ref = sub.add_parser("ref", help="C4-C6: source vs item CRN reference on nexar_val")
    ref.add_argument("--item-dir", type=Path, required=True, help="§29's val/ (<arm>/ reads)")
    ref.add_argument("--source-dir", type=Path, required=True,
                     help="the --crn-reference source reads")
    ref.add_argument("--out-dir", type=Path, required=True)
    ref.set_defaults(func=run_ref)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
