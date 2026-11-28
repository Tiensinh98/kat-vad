"""Where KIM's min-max micro goes: within-clip vs cross-clip pairs (``KIM_RUN.md`` §62).

On an all-abnormal set, the reports pool micro after per-clip min-max. Over all
(positive frame, negative frame) pairs this micro splits exactly into the pairs inside one
clip (AUC_W, weight w_W) and the pairs across clips (AUC_X); AUC_X is split again by the
negative frame's place in its clip (before the first positive = pre, after the last = post,
between positives = gap). An on - off pair of rows is read part by part with a paired cluster
bootstrap. Saved scores only; printed, decides nothing.

Usage::

    python -m core.tools.micro_pair_split \\
        --pair 'K5f|dota|ON_DIR|OFF_DIR' --pair 'K5dfiT2|dota|ON1+ON2|OFF1+OFF2' ...

``ON`` / ``OFF`` are read directories holding ``clip_scores.npz``; several joined by ``+`` are
pooled (e.g. DoTA-CAP dev + eval). The set (``dota`` / ``dada``) picks the bootstrap cluster.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.v2_splits import dota_group
from core.metrics import frame_auc
from core.tools import k5_fail_tail as ft
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import CLIP_SCORES_NPZ, read_clip_scores

LOGGER = logging.getLogger(__name__)

Curves = dict[str, np.ndarray]
PARTS = ("pre", "post", "gap")
SETS = ("dota", "dada")
POS, PRE, POST, GAP = 0, 1, 2, 3
KIND_OF_PART = {"pre": PRE, "post": POST, "gap": GAP}
TIE = 0.5


def minmax(x: np.ndarray) -> np.ndarray:
    span = float(x.max() - x.min())
    return np.asarray((x - x.min()) / span if span > 0 else np.zeros_like(x, dtype=np.float64))


def frame_kinds(labels: np.ndarray) -> np.ndarray:
    """Per frame: POS, or the negative's place (PRE / POST / GAP) relative to the positives."""
    pos = np.flatnonzero(labels > 0)
    idx = np.arange(len(labels))
    kinds = np.full(len(labels), GAP, dtype=np.int64)
    kinds[idx < pos[0]] = PRE
    kinds[idx > pos[-1]] = POST
    kinds[labels > 0] = POS
    return kinds


def load_pooled(dirs: list[Path]) -> tuple[Curves, Curves]:
    scores: Curves = {}
    labels: Curves = {}
    for d in dirs:
        s, y = read_clip_scores(d / CLIP_SCORES_NPZ)
        if set(s) & set(scores):
            raise ValueError(f"{d}: clip ids repeat across pooled read dirs")
        scores.update(s)
        labels.update(y)
    return scores, labels


class Row:
    """One row's min-max frames, flattened, with each clip's own (within) pair counts and AUCs."""

    def __init__(self, scores: Curves, labels: Curves, ids: list[str]) -> None:
        for v in ids:
            if not 0 < int((labels[v] > 0).sum()) < len(labels[v]):
                raise ValueError(f"{v}: §62 MP2 needs every clip two-class")
        mm = [minmax(scores[v].astype(np.float64)) for v in ids]
        kinds = [frame_kinds(labels[v]) for v in ids]
        self.ids = ids
        self.s = np.concatenate(mm)
        self.kind = np.concatenate(kinds)
        self.clip = np.concatenate([np.full(len(m), i) for i, m in enumerate(mm)])
        order = np.argsort(self.s, kind="stable")
        _, self.rank = np.unique(self.s[order], return_inverse=True)
        self.order = order
        n_clips = len(ids)
        self.p = np.array([(k == POS).sum() for k in kinds], dtype=np.float64)
        self.n = {part: np.array([(k == KIND_OF_PART[part]).sum() for k in kinds], dtype=np.float64)
                  for part in PARTS}
        self.n_all = np.sum(np.stack([self.n[part] for part in PARTS]), axis=0)
        self.auc = {part: np.zeros(n_clips) for part in PARTS}
        self.auc_all = np.zeros(n_clips)
        for i, (m, k) in enumerate(zip(mm, kinds, strict=True)):
            self.auc_all[i] = frame_auc(m, (k == POS).astype(np.int64))
            for part in PARTS:
                sel = (k == POS) | (k == KIND_OF_PART[part])
                if self.n[part][i] > 0:
                    self.auc[part][i] = frame_auc(m[sel], (k[sel] == POS).astype(np.int64))
        self.level = {name: float(self.s[self.kind == code].mean()) if (self.kind == code).any()
                      else float("nan")
                      for name, code in (("pos", POS), ("pre", PRE), ("post", POST), ("gap", GAP))}

    def _pooled_wins(self, weight: np.ndarray, neg_kinds: tuple[int, ...]) -> float:
        """Σ over (pos, neg) frame pairs of w_pos · w_neg · [pos > neg] (ties ½)."""
        w = weight[self.clip][self.order]
        kind = self.kind[self.order]
        is_neg = np.isin(kind, neg_kinds)
        n_vals = int(self.rank.max()) + 1
        neg_at = np.bincount(self.rank, weights=w * is_neg, minlength=n_vals)
        pos_at = np.bincount(self.rank, weights=w * (kind == POS), minlength=n_vals)
        below = np.concatenate([[0.0], np.cumsum(neg_at)[:-1]])
        return float((pos_at * (below + TIE * neg_at)).sum())

    def parts(self, weight: np.ndarray) -> dict[str, float]:
        """§62 MP1 for clip multiplicities ``weight``; copies of one clip count as one clip."""
        out: dict[str, float] = {}
        w2 = weight ** 2
        groups: dict[str, tuple[tuple[int, ...], np.ndarray, np.ndarray]] = {
            "all": ((PRE, POST, GAP), self.n_all, self.auc_all)}
        for part in PARTS:
            groups[part] = ((KIND_OF_PART[part],), self.n[part], self.auc[part])
        for name, (neg_kinds, n_c, auc_c) in groups.items():
            total_pairs = float((weight * self.p).sum() * (weight * n_c).sum())
            wins = self._pooled_wins(weight, neg_kinds)
            within_pairs = float((w2 * self.p * n_c).sum())
            within_wins = float((w2 * self.p * n_c * auc_c).sum())
            cross_pairs = total_pairs - within_pairs
            out[f"x_{name}"] = ((wins - within_wins) / cross_pairs if cross_pairs > 0
                                else float("nan"))
            if name == "all":
                out["micro"] = wins / total_pairs
                out["w_within"] = within_pairs / total_pairs
                own = float((weight * self.p * n_c).sum())
                out["within"] = float((weight * self.p * n_c * auc_c).sum()) / own
                out["macro"] = float((weight * auc_c).sum() / weight.sum())
                recon = (out["w_within"] * (within_wins / within_pairs)
                         + (1 - out["w_within"]) * out["x_all"])
                if abs(recon - out["micro"]) > constants.MICRO_SPLIT_GATE_ATOL:
                    raise ValueError(f"§62 MP2 gate: parts {recon:.8f} != micro {out['micro']:.8f}")
        return out


def cluster_of(set_name: str, ids: list[str]) -> dict[str, str]:
    return {v: dota_group(v) for v in ids} if set_name == "dota" else {v: v for v in ids}


def read_pair(label: str, set_name: str, on_dirs: list[Path],
              off_dirs: list[Path]) -> dict[str, Any]:
    s_on, y_on = load_pooled(on_dirs)
    s_off, y_off = load_pooled(off_dirs)
    if set(y_on) != set(y_off):
        raise ValueError(f"{label}: on and off score different clips")
    ids = sorted(y_on)
    for v in ids:
        if not np.array_equal(y_on[v] > 0, y_off[v] > 0):
            raise ValueError(f"{label}: {v} labels differ between on and off")
    on, off = Row(s_on, y_on, ids), Row(s_off, y_off, ids)
    ones = np.ones(len(ids))
    point = {"on": on.parts(ones), "off": off.parts(ones)}
    draws = ft.draws_of(ids, cluster_of(set_name, ids))
    keys = list(point["on"])
    deltas: dict[str, list[float]] = {k: [] for k in keys}
    for idx in draws:
        w = np.bincount(idx, minlength=len(ids)).astype(np.float64)
        a, b = on.parts(w), off.parts(w)
        for k in keys:
            deltas[k].append(a[k] - b[k])
    delta = {}
    for k in keys:
        lo, hi = interval(deltas[k])
        delta[k] = {"mean": point["on"][k] - point["off"][k], "low": lo, "high": hi}
    neg_share = {part: float(on.n[part].sum() / on.n_all.sum()) for part in PARTS}
    return {"label": label, "set": set_name, "items": len(ids), "point": point, "delta": delta,
            "level": {"on": on.level, "off": off.level}, "neg_share": neg_share,
            "reading": reading(delta)}


def interval(values: list[float]) -> tuple[float, float]:
    """95 % percentile interval of the finite draws; NaN when a part is empty (no gap frames)."""
    arr = np.asarray([x for x in values if np.isfinite(x)])
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, ft.ALPHA)), float(np.quantile(arr, 1.0 - ft.ALPHA))


def _below(d: dict[str, float]) -> bool:
    return d["high"] < 0


def reading(delta: dict[str, dict[str, float]]) -> str:
    """§62 MP5, declared before any number."""
    if _below(delta["within"]):
        return "within-clip"
    if _below(delta["x_all"]):
        where = [p for p in PARTS
                 if np.isfinite(delta[f"x_{p}"]["mean"]) and _below(delta[f"x_{p}"])]
        return "cross-clip" + (f" ({', '.join(where)})" if where else " (place not located)")
    return "not located"


def parse_pair(text: str) -> tuple[str, str, list[Path], list[Path]]:
    parts = text.split("|")
    if len(parts) != 4 or parts[1] not in SETS:
        raise ValueError(f"expected 'LABEL|dota-or-dada|ON[+ON]|OFF[+OFF]', got {text!r}")
    label, set_name, on, off = parts
    return label, set_name, [Path(p) for p in on.split("+")], [Path(p) for p in off.split("+")]


def _ci(d: dict[str, float]) -> str:
    return f"{d['mean']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Min-max micro split, within vs cross clip ({report['rule']}) — printed, "
             f"decides nothing, n = 1", "",
             f"Paired cluster bootstrap, {report['draws']} draws, seed {report['seed']}. "
             "x_pre / x_post = cross-clip pairs whose negative frame sits before / after the "
             "clip's positives.", "",
             "## Δ on - off", "",
             "| pair | set | clips | Δ micro | Δ macro | Δ within (AUC_W) | Δ cross (AUC_X) "
             "| Δ x_pre | Δ x_post | reading |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in report["pairs"]:
        d = r["delta"]
        lines.append(f"| {r['label']} | {r['set']} | {r['items']} | {_ci(d['micro'])} | "
                     f"{_ci(d['macro'])} | {_ci(d['within'])} | {_ci(d['x_all'])} | "
                     f"{_ci(d['x_pre'])} | {_ci(d['x_post'])} | {r['reading']} |")
    lines += ["", "## Parts and levels (off → on)", "",
              "| pair | set | w_W | micro | within | cross | x_pre | x_post | neg share pre / post "
              "/ gap | level pos | level pre | level post |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in report["pairs"]:
        a, b = r["point"]["off"], r["point"]["on"]
        la, lb = r["level"]["off"], r["level"]["on"]
        ns = r["neg_share"]

        def arrow(k: str, x: dict[str, float] = a, y: dict[str, float] = b) -> str:
            return f"{x[k]:.4f} → {y[k]:.4f}"

        lines.append(f"| {r['label']} | {r['set']} | {a['w_within']:.4f} | {arrow('micro')} | "
                     f"{arrow('within')} | {arrow('x_all')} | {arrow('x_pre')} | "
                     f"{arrow('x_post')} | "
                     f"{ns['pre']:.3f} / {ns['post']:.3f} / {ns['gap']:.3f} | "
                     f"{arrow('pos', la, lb)} | {arrow('pre', la, lb)} | {arrow('post', la, lb)} |")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    pairs = [read_pair(*parse_pair(p)) for p in args.pair]
    report = {"rule": "KIM_RUN.md §62", "seed": ft.SEED, "draws": ft.DRAWS, "pairs": pairs}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / constants.MICRO_SPLIT_REPORT_STEM
    write_json_atomic(stem.with_suffix(".json"), report)
    write_text_atomic(stem.with_suffix(".md"), render_markdown(report))
    LOGGER.info("wrote %s.{json,md}", stem)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--pair", action="append", required=True,
                        help="LABEL|dota-or-dada|ON[+ON]|OFF[+OFF] (repeatable)")
    parser.add_argument("--out-dir", type=Path,
                        default=constants.OUTPUT_ROOT / constants.MICRO_SPLIT_REPORT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
