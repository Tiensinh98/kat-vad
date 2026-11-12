"""K5's failing tail: are the items it ranks wrongly concentrated in a recorded attribute?

``KIM_RUN.md`` §24 (rule written 2026-11-12, before any number). An item **fails** when its own
frame AUC is below ``FAILTAIL_FAIL_AUC``. The tool reads the fail share per stratum of the
datasets' own annotations, the separation of failing items by span geometry, where failing
items peak, and whether LaGoVAD fails on the same items. Saved scores and annotations only.

Inputs::

    --dota-k5 / --dota-lagovad   protocol-B read dirs (clip_scores.npz)
    --dota-annotations           DoTA raw_annotations/annotations/ (one <clip>.json each)
    --dada-k5 / --dada-lagovad   DADA whole read dirs (clip_scores.npz)
    --dada-workbook              dada标注.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.dada_origin import parse_annotation
from core.data.v2_splits import dota_group
from core.metrics import frame_auc
from core.tools import tadz_readout as td
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.protocol_b_eval import CLIP_SCORES_NPZ, read_clip_scores

LOGGER = logging.getLogger(__name__)

Curves = dict[str, np.ndarray]
SEED = constants.V5_BOOTSTRAP_SEED
DRAWS = constants.BASEMICRO_BOOTSTRAP_DRAWS
ALPHA = (1.0 - constants.V2_E1_CI) / 2.0
DOTA, DADA = "DoTA-CAP-dev", "DADA-2000"
STEPS_PER_S = {DOTA: float(constants.DOTA_FPS),
               DADA: constants.DADA_ASSUMED_FPS / constants.FRAME_STRIDE}
OTHER, SMALL = "other", "too small"
PEAK_PLACES = ("before span", "inside span", "after span")


def item_aucs(run_dir: Path) -> tuple[dict[str, float], Curves, Curves]:
    scores, labels = read_clip_scores(run_dir / CLIP_SCORES_NPZ)
    return td.clip_aucs(scores, labels), scores, labels


def dota_strata(ids: list[str], annotation_dir: Path) -> dict[str, dict[str, str]]:
    """§24 FT2: ``ego_involve``, ``night``, ``accident_name`` per clip; a missing file raises."""
    out: dict[str, dict[str, str]] = {"ego_involve": {}, "night": {}, "accident_name": {}}
    for v in ids:
        path = annotation_dir / f"{v}.json"
        if not path.exists():
            raise FileNotFoundError(f"no DoTA annotation for {v} in {annotation_dir}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        for key in out:
            out[key][v] = str(doc[key])
    return out


def dada_actor(text: str) -> str:
    """§24 FT2: the first of ``FAILTAIL_DADA_ACTORS`` named in the clip's text, else other."""
    words = re.findall(r"[a-z]+", text.lower())
    found = [a for a in constants.FAILTAIL_DADA_ACTORS if a in words]
    return min(found, key=words.index) if found else OTHER


def dada_strata(ids: list[str], workbook: Path) -> tuple[dict[str, dict[str, str]],
                                                         dict[str, float]]:
    """§24 FT2 strata and the risk-phase length in seconds (start -> accident, A1)."""
    rows = {r.video_id: r for r in parse_annotation(workbook)}
    missing = [v for v in ids if v not in rows]
    if missing:
        raise ValueError(f"{len(missing)} DADA videos have no workbook row: {missing[:5]}")
    keys = ("light", "weather", "scenes", "linear")
    out: dict[str, dict[str, str]] = {k: {v: rows[v].attributes[k] for v in ids} for k in keys}
    out["ego"] = {v: str(constants.FAILTAIL_DADA_EGO_TOKEN in rows[v].attributes["texts"].lower())
                  for v in ids}
    out["actor"] = {v: dada_actor(rows[v].attributes["texts"]) for v in ids}
    risk = {v: (rows[v].accident - rows[v].start) / constants.DADA_ASSUMED_FPS for v in ids}
    return out, risk


def pool_small(values: dict[str, str]) -> dict[str, str]:
    """Strata under ``FAILTAIL_MIN_STRATUM`` items are pooled into one; still small -> marked."""
    counts: dict[str, int] = {}
    for x in values.values():
        counts[x] = counts.get(x, 0) + 1
    pooled = {v: x if counts[x] >= constants.FAILTAIL_MIN_STRATUM else OTHER
              for v, x in values.items()}
    n_other = sum(1 for x in pooled.values() if x == OTHER)
    if 0 < n_other < constants.FAILTAIL_MIN_STRATUM:
        pooled = {v: SMALL if x == OTHER else x for v, x in pooled.items()}
    return pooled


def covariates(labels: Curves, rate: float) -> dict[str, dict[str, float]]:
    """§24 FT3 from the labels alone: length, span share, centre, span seconds."""
    out: dict[str, dict[str, float]] = {k: {} for k in
                                        ("length", "span share", "span centre", "span seconds")}
    for v, g in labels.items():
        idx = np.flatnonzero(g > 0)
        out["length"][v] = float(len(g))
        out["span share"][v] = float(len(idx) / len(g))
        out["span centre"][v] = float((idx[0] + idx[-1]) / 2 / len(g))
        out["span seconds"][v] = float(len(idx) / rate)
    return out


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties sharing their mean rank (what a Spearman correlation needs)."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    return np.asarray((upper - (counts - 1) / 2.0)[inverse], dtype=np.float64)


def draws_of(ids: list[str], groups: dict[str, str]) -> list[np.ndarray]:
    """Cluster bootstrap: each draw is the item indices of clusters drawn with replacement."""
    members: dict[str, list[int]] = {}
    for i, v in enumerate(ids):
        members.setdefault(groups[v], []).append(i)
    units = [np.array(members[g]) for g in sorted(members)]
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(units), size=(DRAWS, len(units)))
    return [np.concatenate([units[j] for j in row]) for row in picks]


def _interval(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([x for x in values if np.isfinite(x)])
    return float(np.quantile(arr, ALPHA)), float(np.quantile(arr, 1.0 - ALPHA))


def stratum_reads(fail: np.ndarray, auc: np.ndarray, column: np.ndarray,
                  draws: list[np.ndarray]) -> dict[str, Any]:
    """§24 FT4 for one attribute: per stratum n, fail share, mean AUC, Δ share vs rest."""
    overall = float(fail.mean())
    out: dict[str, Any] = {}
    for name in sorted(set(column.tolist())):
        m = column == name
        if m.sum() == 0 or (~m).sum() == 0:
            continue
        delta = float(fail[m].mean() - fail[~m].mean())
        boot = [float(fail[d][m[d]].mean() - fail[d][~m[d]].mean())
                if m[d].any() and (~m[d]).any() else float("nan") for d in draws]
        low, high = _interval(boot)
        share = float(fail[m].mean())
        lead = (name != SMALL and int(m.sum()) >= constants.FAILTAIL_MIN_STRATUM
                and share >= constants.FAILTAIL_LEAD_RATIO * overall and low > 0)
        out[name] = {"n": int(m.sum()), "fail_share": share, "mean_auc": float(auc[m].mean()),
                     "delta_share": {"mean": delta, "low": low, "high": high}, "lead": lead}
    return out


def covariate_read(x: np.ndarray, fail: np.ndarray, draws: list[np.ndarray]) -> dict[str, Any]:
    """AUC of a covariate separating failing items (FT4) and whether it is a lead (FT5)."""
    value = frame_auc(x, fail.astype(np.int64))
    boot = [frame_auc(x[d], fail[d].astype(np.int64)) if 0 < fail[d].sum() < len(d)
            else float("nan") for d in draws]
    low, high = _interval(boot)
    lead = (low > 0.5 or high < 0.5) and abs(value - 0.5) >= constants.FAILTAIL_COVARIATE_MARGIN
    return {"auc": value, "low": low, "high": high, "lead": lead,
            "mean_fail": float(x[fail].mean()), "mean_rest": float(x[~fail].mean())}


def peak_places(scores: Curves, labels: Curves, failing: list[str]) -> dict[str, float]:
    counts = dict.fromkeys(PEAK_PLACES, 0)
    for v in failing:
        idx = np.flatnonzero(labels[v] > 0)
        p = int(np.argmax(scores[v]))
        place = PEAK_PLACES[0] if p < idx[0] else PEAK_PLACES[2] if p > idx[-1] else PEAK_PLACES[1]
        counts[place] += 1
    return {k: c / len(failing) if failing else float("nan") for k, c in counts.items()}


def read_set(name: str, k5_dir: Path, lag_dir: Path, strata: dict[str, dict[str, str]],
             extra: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Every FT4 read of one set."""
    k5, scores, labels = item_aucs(k5_dir)
    lag, _, lag_labels = item_aucs(lag_dir)
    if set(k5) != set(lag) or any(not np.array_equal(labels[v], lag_labels[v]) for v in k5):
        raise ValueError(f"{name}: K5 and LaGoVAD did not score the same items and labels")
    ids = sorted(k5)
    auc = np.array([k5[v] for v in ids])
    fail = auc < constants.FAILTAIL_FAIL_AUC
    groups = {v: dota_group(v) for v in ids} if name == DOTA else {v: v for v in ids}
    draws = draws_of(ids, groups)
    attrs: dict[str, Any] = {}
    for key, values in strata.items():
        pooled = pool_small(values)
        attrs[key] = stratum_reads(fail, auc, np.array([pooled[v] for v in ids]), draws)
    covs = {**covariates({v: labels[v] for v in ids}, STEPS_PER_S[name]), **extra}
    cov_reads = {k: covariate_read(np.array([c[v] for v in ids]), fail, draws)
                 for k, c in covs.items()}
    lag_auc = np.array([lag[v] for v in ids])
    lag_fail = lag_auc < constants.FAILTAIL_FAIL_AUC
    rho = float(np.corrcoef(average_ranks(auc), average_ranks(lag_auc))[0, 1])  # Spearman
    failing = [v for v, f in zip(ids, fail, strict=True) if f]
    return {
        "set": name, "items": len(ids), "clusters": len(set(groups.values())),
        "fail_share": float(fail.mean()), "failing": int(fail.sum()),
        "strata": attrs, "covariates": cov_reads,
        "peak_of_failing": peak_places(scores, labels, failing),
        "shared": {"spearman_k5_lagovad": rho, "lagovad_fail_share": float(lag_fail.mean()),
                   "lagovad_fail_share_among_k5_fails":
                       float(lag_fail[fail].mean()) if fail.any() else float("nan")},
    }


def leads(report_sets: dict[str, Any]) -> list[str]:
    out = []
    for name, s in report_sets.items():
        for key, strata in s["strata"].items():
            out += [f"{name} {key} = {st}" for st, r in strata.items() if r["lead"]]
        out += [f"{name} {k}" for k, r in s["covariates"].items() if r["lead"]]
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    dota_ids = sorted(item_aucs(args.dota_k5)[0])
    dada_ids = sorted(item_aucs(args.dada_k5)[0])
    d_strata, risk = dada_strata(dada_ids, args.dada_workbook)
    sets = {
        DOTA: read_set(DOTA, args.dota_k5, args.dota_lagovad,
                       dota_strata(dota_ids, args.dota_annotations), {}),
        DADA: read_set(DADA, args.dada_k5, args.dada_lagovad, d_strata,
                       {"risk-phase seconds": risk}),
    }
    found = leads(sets)
    follow = ("leads found: Claude describes them and may propose one targeted check; the user "
              "decides" if found else "no lead: K5's failures are not concentrated in any "
              "recorded attribute; Claude recommends closing the micro line and writing it up")
    report = {"rule": "KIM_RUN.md §24", "seed": SEED, "draws": DRAWS, "sets": sets,
              "leads": found, "follow_up": follow}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / constants.FAILTAIL_REPORT_STEM
    write_json_atomic(stem.with_suffix(".json"), report)
    write_text_atomic(stem.with_suffix(".md"), render_markdown(report))
    return report


def _ci(c: dict[str, float]) -> str:
    return f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}]"


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# K5's failing tail (`KIM_RUN.md` §24)", "",
             f"PRELIMINARY (n = 1, K5 s2024). Fail = own frame AUC < "
             f"{constants.FAILTAIL_FAIL_AUC}. Cluster bootstrap {report['draws']} draws, seed "
             f"{report['seed']}. Lead rules FT5; about one false lead is expected by chance.", ""]
    for name, s in report["sets"].items():
        sh = s["shared"]
        lines += [f"## {name}: {s['failing']} / {s['items']} fail ({s['fail_share']:.3f}); "
                  f"{s['clusters']} clusters", "",
                  "| attribute | stratum | n | fail share | mean AUC | Δ share vs rest [95 %] "
                  "| lead |",
                  "|---|---|---:|---:|---:|---|---|"]
        for key, strata in s["strata"].items():
            for st, r in strata.items():
                lines.append(f"| {key} | {st} | {r['n']} | {r['fail_share']:.3f} | "
                             f"{r['mean_auc']:.3f} | {_ci(r['delta_share'])} | "
                             f"{'**lead**' if r['lead'] else ''} |")
        lines += ["", "| covariate | mean (fail / rest) | AUC fail vs rest [95 %] | lead |",
                  "|---|---|---|---|"]
        for k, r in s["covariates"].items():
            lines.append(f"| {k} | {r['mean_fail']:.3f} / {r['mean_rest']:.3f} | {r['auc']:.3f} "
                         f"[{r['low']:.3f}, {r['high']:.3f}] | {'**lead**' if r['lead'] else ''} |")
        peaks = ", ".join(f"{k} {x:.3f}" for k, x in s["peak_of_failing"].items())
        lines += ["", f"Peak of the failing items: {peaks}.", "",
                  f"Shared difficulty: Spearman rho(K5, LaGoVAD per-item AUC) = "
                  f"{sh['spearman_k5_lagovad']:.3f}; LaGoVAD fails on "
                  f"{sh['lagovad_fail_share_among_k5_fails']:.3f} of K5's failing items against "
                  f"{sh['lagovad_fail_share']:.3f} overall.", ""]
    lead_text = "; ".join(report["leads"]) if report["leads"] else "none"
    lines += [f"**Leads (FT5):** {lead_text}.", "", f"**FT6:** {report['follow_up']}."]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    for flag in ("--dota-k5", "--dota-lagovad", "--dota-annotations", "--dada-k5",
                 "--dada-lagovad", "--dada-workbook", "--out-dir"):
        parser.add_argument(flag, type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
