"""K5nc2 (negative-video windows capped at 2 per video) beside K5 and K5neg (``KIM_RUN.md`` §27).

Pure aggregation over saved reads. The K5 - K5nc2 contrast reuses ``k5_whole_report``'s gates
(same items, micro equal to its read, one checkpoint per row, rows on different checkpoints,
``nexar_val`` only) and §17 NG4 (K5's ``nexar_val`` re-read reproduces §13). Inputs::

    <kim-root>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/       K5
    <neg-root>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/       K5neg (§17)
    <neg-root>/eval/nexar/{K5,K5neg}/                                 §17's nexar_val reads
    <nc-root>/eval/{dota,dada}/K5/, eval/{tad,dad}/K5/s<seed>/        K5nc2
    <nc-root>/eval/nexar/K5nc2/ (+ scores/)                           K5nc2 on nexar_val
    <k5-reference-dir>/                                               §13's K5 read on nexar_val
    <k5-scores-dir>/scores/, <neg-scores-dir>/scores/                 per-frame nexar_val curves

NC5 decides on macro; NC6 is printed: false alarms (peak > ``NORMAL_VIDEO_PEAK``), detection at
matched false-alarm rates (cut = the ``1 - rate`` linear quantile of a row's own negative peaks,
§17.1's arithmetic), 5 s chunk rates and the last 5 s before onset, K5nc2 - K5neg.

CLI::

    python -m core.tools.k5_nc2_report --kim-root outputs/v_final/kim \\
        --neg-root outputs/v_final/kim_neg --nc-root outputs/v_final/kim_negcap \\
        --k5-reference-dir outputs/v_final/kim_t2/eval/nexar/K5 \\
        --k5-scores-dir outputs/v_final/kim_gate/eval/nexar/K5 \\
        --neg-scores-dir outputs/v_final/kim_neg/eval/nexar_curves/K5neg \\
        --out-dir outputs/v_final/kim_negcap/REPORTS
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.tools import k5_neg_report as kn
from core.tools import k5_whole_report as kw
from core.tools import k5_zs_report as zs
from core.tools import tadz_readout
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.zs_lagovad_readout import ABOVE, BELOW

LOGGER = logging.getLogger(__name__)

REPORT_STEM = "k5_nc2_report"
K5, NEG, NC, NEXAR = kw.K5, constants.KIM_NEG_ROW, constants.KIM_NEGCAP_ROW, kw.NEXAR
ROWS = (K5, NEG, NC)
VERDICT_STAYS, VERDICT_CANDIDATE, VERDICT_NONE = "K5 stays", "candidate", "no claim"
NOTES = """## How to read this

* **PRELIMINARY (n = 1 seed, s2024)** (`KIM_RUN.md` §27). One factor changes against K5: the
  normal pool gains at most 2 evenly spaced windows per negative video (~14 % of the pool; K5neg
  ~55 %). K5 stays the main model unless NC7 reads "candidate" and the user decides.
* **Matched false-alarm detection has one point per rate and no interval** (112 / 113 videos);
  the cut is each row's own, so it reads ranking, not calibration.
* **Nexar val is in-domain**; crops are the endpoint, whole-video macro is a position test.
* **Micro on TAD and DAD measures clip classification. DAD macro is not localization**.
"""


def matched_detection(stats: dict[str, dict[str, Any]], rates: tuple[float, ...]
                      ) -> dict[str, dict[str, float]]:
    """Detection of positives at each false-alarm rate, the cut taken on the row's own negatives."""
    neg = np.array([s["peak"] for s in stats.values() if s["label"] == 0], dtype=float)
    pos = np.array([s["peak"] for s in stats.values() if s["label"] == 1], dtype=float)
    if not len(neg) or not len(pos):
        raise ValueError("nexar: matched detection needs positive and negative videos")
    out = {}
    for rate in rates:
        cut = float(np.quantile(neg, 1.0 - rate))
        out[f"{rate:g}"] = {"cut": cut, "fa": float(np.mean(neg > cut)),
                            "detect": float(np.mean(pos > cut))}
    return out


def _chunk_peaks(score: np.ndarray, rows: int) -> list[float]:
    return [float(score[i * rows:(i + 1) * rows].max()) for i in range(len(score) // rows)]


def chunk_rates(scores_dir: Path) -> dict[str, Any]:
    """NC6 (c)(d): 5 s chunks with max > the peak cut, far before onset vs negatives; last 5 s."""
    scores, labels = tadz_readout.load_run(scores_dir)
    chunk = round(constants.KIM_NEGCAP_CHUNK_S * constants.NEXAR_NOMINAL_FPS)
    zone = round(constants.KIM_NEGCAP_PRE_ONSET_S * constants.NEXAR_NOMINAL_FPS)
    cut = constants.NORMAL_VIDEO_PEAK
    far: list[float] = []
    neg: list[float] = []
    last: list[float] = []
    for v, score in scores.items():
        gt = labels[v]
        if len(gt) != len(score):
            raise ValueError(f"{scores_dir}: {v} has {len(score)} scores for {len(gt)} labels")
        if not gt.any():
            neg += _chunk_peaks(score, chunk)
            continue
        onset = int(np.argmax(gt > 0))
        far += _chunk_peaks(score[:max(0, onset - zone)], chunk)
        near = score[max(0, onset - zone):onset]
        if len(near):
            last.append(float(near.max()))
    return {"videos": len(scores),
            "far_pre_onset": {"chunks": len(far), "share": float(np.mean(np.array(far) > cut))},
            "negatives": {"chunks": len(neg), "share": float(np.mean(np.array(neg) > cut))},
            "last_pre_onset": {"videos": len(last),
                               "share": float(np.mean(np.array(last) > cut))}}


def follow_up(report: dict[str, Any]) -> dict[str, Any]:
    """NC7: candidate iff no set below and (a set above or matched detection gains the margin)."""
    sentences = {name: s["sentence"] for name, s in report["sets"].items()}
    below = [n for n, s in sentences.items() if s == BELOW]
    above = [n for n, s in sentences.items() if s == ABOVE]
    key = f"{constants.KIM_NEGCAP_DECIDE_FA:g}"
    m = report["matched"]
    gain = m[NC][key]["detect"] - m[K5][key]["detect"]
    if below:
        verdict = VERDICT_STAYS
    elif above or gain >= constants.KIM_NEGCAP_DETECT_MARGIN:
        verdict = VERDICT_CANDIDATE
    else:
        verdict = VERDICT_NONE
    return {"below": below, "above": above, "detect_gain": gain, "verdict": verdict}


def _contrast_lines(title: str, sets: dict[str, Any]) -> list[str]:
    lines = ["", f"## {title}", "", "| Set | Δ [95 % CI] | Sentence |", "|---|---|---|"]
    for name, s in sets.items():
        what = "crops" if name == NEXAR else "macro"
        lines.append(f"| {kw.SET_NAME[name]} ({what}) | {kw._signed(s['delta'])} | "
                     f"{s['sentence']} |")
    return lines


def render(report: dict[str, Any]) -> str:
    """The comparison as one markdown document."""
    n = report["sets"][NEXAR]
    lines = [
        f"# K5nc2 (negative-video windows capped at 2) vs K5 and K5neg — **{kw.LABEL}**",
        "",
        f"Seed {report['seed']}. Rule `KIM_RUN.md` §27, written before these numbers. Generated by "
        "`core.tools.k5_nc2_report`. `nexar_test` is not read. NG4: K5's re-read "
        f"{report['repro']['reread']:.4f} vs §13 {report['repro']['reference']:.4f} "
        f"(gap {report['repro']['gap']:+.1e}).",
        "",
        f"## Nexar val ({n['videos']} videos, in-domain)",
        "",
        "| Row | crops (endpoint) | whole macro | micro | AP | clip AUC |",
        "|---|---|---|---|---|---|",
    ]
    cells = {K5: n["rows"][K5], NC: n["rows"][NC], NEG: report["neg_sets"][NEXAR]["rows"][NEG]}
    for row in ROWS:
        c = cells[row]
        lines.append(f"| {row} | {kw._macro(c['macro_crops'])} | {kw._macro(c['macro_whole'])} | "
                     f"{kn._num(c['micro'])} | {kn._num(c['ap'])} | {kn._num(c['clip_auc'])} |")
    rates = list(report["matched"][K5])
    fa = report["false_alarms"]
    lines += [
        "",
        f"## False alarms and detection on Nexar val (NC6, printed; peak cut "
        f"{constants.NORMAL_VIDEO_PEAK})",
        "",
        "| Row | negatives peak > cut | positives peak > cut | "
        + " | ".join(f"detect @ FA {r}" for r in rates)
        + " | far pre-onset 5 s chunks | negative 5 s chunks | last 5 s before onset |",
        "|---|---|---|" + "---|" * len(rates) + "---|---|---|",
    ]
    for row in ROWS:
        m, c = report["matched"][row], report["chunks"][row]
        lines.append(
            f"| {row} | {fa[row]['share']:.4f} | {report['detected'][row]:.4f} | "
            + " | ".join(f"{m[r]['detect']:.4f} (FA {m[r]['fa']:.3f})" for r in rates)
            + f" | {c['far_pre_onset']['share']:.4f} | {c['negatives']['share']:.4f} | "
            f"{c['last_pre_onset']['share']:.4f} |")
    scenes = (*kn.SCENES, kn.OTHER_SCENE)
    lines += ["", "Negatives with peak > cut, by scene:", "",
              "| Row | " + " | ".join(
                  f"{s} (n = {fa[K5]['by_scene'][s]['videos']})" for s in scenes) + " |",
              "|---|" + "---|" * len(scenes)]
    for row in ROWS:
        lines.append(f"| {row} | " + " | ".join(
            kn._num(fa[row]["by_scene"][s]["share"]) for s in scenes) + " |")
    lines += ["", "## Zero-shot sets (macro)", "", "| Set | K5 | K5neg | K5nc2 |",
              "|---|---|---|---|"]
    for name in kw.ZS_SETS:
        r, rn = report["sets"][name]["rows"], report["neg_sets"][name]["rows"]
        lines.append(f"| {kw.SET_NAME[name]} | {kw._macro(r[K5]['macro'])} | "
                     f"{kw._macro(rn[NEG]['macro'])} | {kw._macro(r[NC]['macro'])} |")
    lines += _contrast_lines("K5nc2 - K5, paired per item (§27 NC5, decides)", report["sets"])
    lines += _contrast_lines("K5nc2 - K5neg, paired per item (§27 NC6 f, printed)",
                             report["vs_neg"])
    f = report["follow_up"]
    lines += [
        "",
        f"**§27 NC7:** below = {f['below'] or 'none'}; above = {f['above'] or 'none'}; detection "
        f"at matched FA {constants.KIM_NEGCAP_DECIDE_FA:g}, K5nc2 - K5 {f['detect_gain']:+.4f} "
        f"(margin {constants.KIM_NEGCAP_DETECT_MARGIN}). Verdict: **{f['verdict']}**.",
        "",
    ]
    return "\n".join(lines) + "\n" + NOTES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--kim-root", type=Path, required=True, help="outputs/kim (K5)")
    parser.add_argument("--neg-root", type=Path, required=True, help="outputs/kim_neg (§17)")
    parser.add_argument("--nc-root", type=Path, required=True, help="outputs/kim_negcap")
    parser.add_argument("--k5-reference-dir", type=Path, required=True,
                        help="§13's K5 nexar_val read (kim_t2/eval/nexar/K5)")
    parser.add_argument("--k5-scores-dir", type=Path, required=True,
                        help="K5's nexar_val per-frame scores (kim_gate/eval/nexar/K5)")
    parser.add_argument("--neg-scores-dir", type=Path, required=True,
                        help="K5neg's nexar_val per-frame scores (kim_neg/eval/nexar_curves/K5neg)")
    parser.add_argument("--seed", type=int, default=constants.KIM_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    views = {K5: kw._view(args.kim_root, args.seed), NEG: kw._view(args.neg_root, args.seed),
             NC: kw._view(args.nc_root, args.seed)}
    neg_nexar = args.neg_root / zs.EVAL_DIR / NEXAR
    dirs = {K5: neg_nexar / K5, NEG: neg_nexar / NEG, NC: args.nc_root / zs.EVAL_DIR / NEXAR / NC}
    repro = kn.reproduce_gate(dirs[K5], args.k5_reference_dir)
    for other in (NEG, NC):
        kw.check_checkpoints({K5: views[K5], other: views[other]},
                             {K5: dirs[K5], other: dirs[other]}, args.seed, new=other)
    kw.check_checkpoints({K5: views[NEG], NC: views[NC]}, {K5: dirs[NEG], NC: dirs[NC]},
                         args.seed, new=NC)

    def contrast(control: str, new: str) -> dict[str, Any]:
        return {NEXAR: kw.nexar_set({K5: dirs[control], new: dirs[new]}, new=new),
                **{s: kw.zero_shot_set({K5: views[control], new: views[new]}, s, new=new)
                   for s in kw.ZS_SETS}}

    stats = {row: kn.load_video_stats(dirs[row]) for row in ROWS}
    if len({frozenset(s) for s in stats.values()}) != 1:
        raise ValueError("nexar: the rows' video_stats cover different videos")
    score_dirs = {K5: args.k5_scores_dir, NEG: args.neg_scores_dir, NC: dirs[NC]}
    report: dict[str, Any] = {
        "label": kw.LABEL, "seed": args.seed, "repro": repro,
        "sets": contrast(K5, NC), "neg_sets": contrast(K5, NEG), "vs_neg": contrast(NEG, NC),
        "false_alarms": {row: kn.false_alarms(stats[row]) for row in ROWS},
        "detected": {row: float(np.mean([s["peak"] > constants.NORMAL_VIDEO_PEAK
                                         for s in stats[row].values() if s["label"] == 1]))
                     for row in ROWS},
        "matched": {row: matched_detection(stats[row], constants.KIM_NEGCAP_MATCHED_FA)
                    for row in ROWS},
        "chunks": {row: chunk_rates(score_dirs[row]) for row in ROWS},
    }
    if len({c["videos"] for c in report["chunks"].values()} | {len(stats[K5])}) != 1:
        raise ValueError("nexar: the per-frame score dirs and video_stats differ in size")
    report["follow_up"] = follow_up(report)
    write_json_atomic(args.out_dir / f"{REPORT_STEM}.json", report)
    write_text_atomic(args.out_dir / f"{REPORT_STEM}.md", render(report))
    LOGGER.info("report -> %s", args.out_dir / f"{REPORT_STEM}.md")
    return report


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
