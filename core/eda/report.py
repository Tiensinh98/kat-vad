"""Assemble the EDA sections, render Markdown, and diff two corpora.

The Markdown is the deliverable: every table carries the threshold that makes it
actionable, and :func:`_verdicts` turns the numbers into the specific warnings
``core/docs/v3/RESULTS_DADA.md`` had to be written to record — so the next corpus
raises them *before* a campaign, not after.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from core import constants
from core.eda import corpus, features, labels, protocol

# Colab and CI both run headless; force the non-interactive backend at import.
matplotlib.use("Agg", force=True)

LOGGER = logging.getLogger(__name__)

FULLY_COVERED_WARN_FRACTION = 0.25  # lesson C27 fires above this
K1_WARN_FRACTION = 0.5  # MIL degenerates to a plain max on this share of clips
ORACLE_WARN_AUC = 0.75  # a clip classifier already gets this much of micro
LENGTH_LEAK_WARN_AUC = 0.65  # clip length alone predicts the label above this (C28)
BETWEEN_WITHIN_WARN = 5.0  # feature/score variance dominated by scene identity
PROBE_SIGNAL_AUC = 0.60  # a macro AUC above this counts as real frame-level signal
# Every task loss in this objective is a BCE or an InfoNCE and sits at O(1). A
# constant-predictor MSE above this means lambda_rec = 1.0 buys L_KIP_rec a
# multiple of the whole anomaly objective, not a share of it.
FLOW_TARGET_SCALE_WARN_MSE = 4.0
FLOW_TARGET_BETWEEN_ITEM_WARN = 0.5  # half the target reachable from item identity alone


def build_report(
    dataset: str,
    data_dir: Path,
    clip_dir: Path | None,
    flow_dir: Path | None,
    scores_dir: Path | None,
    sections: tuple[str, ...],
    kernel: int,
    topk_pct: int,
    max_clips: int,
    probe: bool,
    seed: int,
) -> dict[str, Any]:
    """Run the requested sections and return one JSON-serializable report."""
    files = corpus.load_dataset_files(data_dir, dataset)
    report: dict[str, Any] = {
        "dataset": dataset,
        "data_dir": str(data_dir),
        "frame_stride_assumed": constants.FRAME_STRIDE,
        "score_head_kernel": kernel,
        "mil_topk_pct": topk_pct,
        "sections": list(sections),
    }
    if constants.EDA_SECTION_CORPUS in sections:
        report["corpus"] = corpus.corpus_report(files, kernel, topk_pct)
    if constants.EDA_SECTION_LABELS in sections:
        report["labels"] = labels.label_report(files)
    if constants.EDA_SECTION_PROTOCOL in sections:
        report["protocol"] = protocol.protocol_report(
            files, scores_dir if constants.EDA_SECTION_SCORES in sections else None
        )
    if constants.EDA_SECTION_FEATURES in sections:
        if clip_dir is None:
            LOGGER.warning("Section 'features' requested without --clip-dir; skipping")
        else:
            report["features"] = features.feature_report(
                files, clip_dir, flow_dir, max_clips, probe, seed
            )
    report["verdicts"] = _verdicts(report)
    return report


def _verdicts(report: dict[str, Any]) -> list[dict[str, str]]:
    """Turn the measurements into named, actionable warnings."""
    out: list[dict[str, str]] = []

    def add(level: str, title: str, detail: str, action: str) -> None:
        out.append({"level": level, "title": title, "detail": detail, "action": action})

    cov = report.get("corpus", {}).get("kernel_coverage_test", {})
    if cov.get("fraction_fully_covered", 0.0) >= FULLY_COVERED_WARN_FRACTION:
        add(
            "CRITICAL", "Score head spans the clip (lesson C27)",
            f"{cov['fraction_fully_covered']:.1%} of test clips are <= "
            f"kernel_size {cov['kernel']} (median T = {cov['median_length']:.0f}); "
            f"{cov['fraction_frames_fully_covered']:.1%} of frames live in them. "
            "Every output timestep there sees the whole clip.",
            f"Lower model.score_head_kernel (try 3) or lower frame_stride so "
            f"median T >> {cov['kernel']}. Both fire lesson C2 on existing caches.",
        )
    mil = report.get("corpus", {}).get("mil_topk_floor_test", {})
    if mil.get("fraction_at_k1", 0.0) >= K1_WARN_FRACTION:
        add(
            "HIGH", "MIL top-k degenerates to a plain max",
            f"{mil['fraction_at_k1']:.1%} of clips get k = 1 at "
            f"mil_topk_pct = {mil['topk_pct']}: one supervised frame per clip per step.",
            "Lower loss.mil_topk_pct, or accept that supervision is clip-level here "
            "and stop reading frame-level claims into it.",
        )
    van = report.get("labels", {}).get("vanished_windows", {})
    if van.get("count", 0):
        unit = van.get("unit", "clip")
        add(
            "HIGH", f"Abnormal {unit}s with an all-zero label vector",
            f"{van['count']} {unit}s declared abnormal in meta.json have no positive "
            "sampled frame anywhere; every metric counts them as normal.",
            "Exclude them at scoring time (core/docs/DADA_SETUP.md §5.1). Do NOT "
            "'fix' this with --strict: that flag raises, it does not repair.",
        )
    proto = report.get("protocol", {})
    oracle = proto.get("clip_constant_oracle", {})
    pairs = proto.get("pair_decomposition", {})
    if oracle.get("auc_micro", 0.0) >= ORACLE_WARN_AUC:
        add(
            "CRITICAL", "Micro AUC here is mostly clip classification (lesson C12)",
            f"A constant-score-per-clip oracle scores micro AUC "
            f"{oracle['auc_micro']:.4f} with zero localization; "
            f"{pairs.get('cross_clip_fraction', float('nan')):.1%} of the "
            "positive/negative pairs span two clips.",
            "Report auc_macro as the headline and print this oracle beside any "
            "micro number. Never place a micro number from this corpus next to a "
            "published frame-level AUC.",
        )
    leak = proto.get("clip_length_leak", {})
    if leak.get("auc_clip_level", 0.0) >= LENGTH_LEAK_WARN_AUC:
        add(
            "CRITICAL", "Clip length alone predicts the label (lesson C28)",
            f"A constant-score-per-clip detector reading only the clip's frame "
            f"count scores clip-level AUC {leak['auc_clip_level']:.4f} and micro "
            f"AUC {leak['auc_micro']:.4f} — {leak['direction']} clips are the "
            f"abnormal ones. {_fmt(leak['disjoint_normal_clips'])} normal clips "
            f"({_fmt(leak['disjoint_normal_frames'])} frames) fall outside the "
            "abnormal length range entirely.",
            "Rebuild the corpus into fixed-length windows with the anomaly at a "
            "random offset. Until then, print this baseline beside every micro "
            "AUC and treat any arm that fails to beat it as unmeasured.",
        )
    flat = proto.get("scored_run", {}).get("flatness", {})
    if flat.get("between_over_within") and flat["between_over_within"] >= BETWEEN_WITHIN_WARN:
        add(
            "HIGH", "The scored run's curves are flat",
            f"Between-clip / within-clip score variance = "
            f"{flat['between_over_within']:.1f}; "
            f"{flat.get('constant_curve_clips', 0)} clips have a literally constant curve.",
            "This model has learned clip separation, not localization. Expect it to "
            "collapse under a per-clip min-max protocol (lesson C8).",
        )
    feat = report.get("features", {})
    var = feat.get("variance_decomposition", {})
    if var.get("between_over_within") and var["between_over_within"] >= BETWEEN_WITHIN_WARN:
        add(
            "INFO", "Frozen features are dominated by scene identity",
            f"Between-clip / within-clip feature variance = "
            f"{var['between_over_within']:.1f}.",
            "Temporal dynamics are a small part of this embedding; a head reading "
            "them is working against the representation's own scale.",
        )
    fprobe = feat.get("frame_linear_probe", {})
    if fprobe.get("ran"):
        macro = fprobe.get("auc_macro")
        cprobe = feat.get("clip_linear_probe", {})
        if macro is not None and macro >= PROBE_SIGNAL_AUC:
            add(
                "INFO", "The features DO carry a frame-level signal",
                f"Supervised linear probe reaches auc_macro {macro:.4f} on the same "
                "cached features the trained arms saw.",
                "The deficit is SUPERVISION, not representation. A better head or "
                "denser labels can close it; a different backbone is not required.",
            )
        elif macro is not None:
            add(
                "CRITICAL", "The features carry no frame-level signal",
                f"Even a supervised linear probe reaches only auc_macro {macro:.4f}"
                + (
                    f", while the clip-level probe reaches {cprobe['auc']:.4f}"
                    if cprobe.get("ran") else ""
                )
                + ".",
                "No head on these features can localize. Frame-level work on this "
                "corpus needs a different backbone or a finer stride -- not another "
                "KIP variant (RESULTS_DADA.md §6, §10-C).",
            )
    target = feat.get("flow", {}).get("target", {})
    if target:
        baseline = target["mse_global_mean_predictor"]
        if baseline >= FLOW_TARGET_SCALE_WARN_MSE:
            top = feat["flow"].get("raw_energy_share", [{}])[0]
            add(
                "HIGH", "The KIP reconstruction target is unnormalized",
                f"A single global-mean vector already scores MSE {baseline:.2f} on "
                f"e_O, and `{top.get('name', '?')}` carries "
                f"{top.get('energy_share', float('nan')):.1%} of E[s²]. Every task "
                "loss in this objective is a BCE or an InfoNCE, i.e. O(1), so at "
                f"lambda_rec = 1.0 L_KIP_rec enters the sum ~{baseline:.0f}x larger "
                "and its gradient reaches the shared temporal encoder.",
                "Measure the gradient split with `python -m core.tools.grad_probe` "
                "before changing anything. Then either z-score e_O (fires lesson C2: "
                "new cache version, every KIP-on number re-measured) or set "
                f"loss.lambda_rec ~ {1.0 / baseline:.3f}. Do not read a raw kip_rec "
                "as 'the head fits badly' -- divide it by this baseline first.",
            )
        if target["between_item_share"] >= FLOW_TARGET_BETWEEN_ITEM_WARN:
            add(
                "HIGH", "The KIP target is mostly item identity, not dynamics",
                f"{target['between_item_share']:.1%} of e_O's variance is between "
                f"items: an oracle knowing only each item's own mean scores MSE "
                f"{target['mse_item_mean_predictor']:.2f} against the global-mean "
                f"baseline's {baseline:.2f}.",
                "Most of L_KIP_rec is reachable by predicting which clip this is. "
                "Report R^2 against BOTH baselines; only the margin below "
                "mse_item_mean_predictor is evidence PMG learned motion at all.",
            )
    if not out:
        add("OK", "No structural red flag found", "All thresholds passed.", "Proceed.")
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _describe_row(label: str, block: dict[str, Any]) -> str:
    pct = block.get("percentiles", {})
    return (
        f"| {label} | {_fmt(block.get('n'))} | {_fmt(block.get('mean'), 2)} | "
        f"{_fmt(pct.get('p5'), 1)} | {_fmt(pct.get('p25'), 1)} | {_fmt(pct.get('p50'), 1)} | "
        f"{_fmt(pct.get('p75'), 1)} | {_fmt(pct.get('p95'), 1)} | "
        f"{_fmt(pct.get('p0'), 1)} | {_fmt(pct.get('p100'), 1)} |"
    )


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable report. The JSON stays the machine-readable source."""
    lines: list[str] = [
        f"# EDA — {report['dataset']}",
        "",
        f"Data dir: `{report['data_dir']}` · score head kernel "
        f"**{report['score_head_kernel']}** · MIL top-k pct **{report['mil_topk_pct']}** · "
        f"sections: {', '.join(report['sections'])}",
        "",
        "Every number here is a property of the data on disk, not of a model.",
        "",
        "## 0. Verdicts",
        "",
        "| Level | Finding | Measurement | What to do |",
        "|---|---|---|---|",
    ]
    for v in report["verdicts"]:
        lines.append(f"| **{v['level']}** | {v['title']} | {v['detail']} | {v['action']} |")

    if "corpus" in report:
        c = report["corpus"]
        s = c["splits"]
        lines += [
            "", "## 1. Corpus shape", "",
            "| | clips | abnormal | normal | frames |",
            "|---|---:|---:|---:|---:|",
            f"| train | {_fmt(s['train_clips'])} | {_fmt(s['train_abnormal'])} | "
            f"{_fmt(s['train_normal'])} | — |",
            f"| test | {_fmt(s['test_clips'])} | {_fmt(s['test_abnormal'])} | "
            f"{_fmt(s['test_normal'])} | {_fmt(s['test_sampled_frames'])} |",
            "",
            f"Test positive frames: **{_fmt(s['test_positive_frames'])}** "
            f"({s['test_positive_frames'] / s['test_sampled_frames']:.2%} of sampled frames)"
            if s["test_sampled_frames"] else "",
            f"DVS dataset length (2 x abnormal train): **{_fmt(s['dvs_dataset_len'])}** → "
            f"**{_fmt(s['steps_per_epoch_at_batch_64'])} steps/epoch** at batch 64.",
            "",
        ]
        if s["split_leak_ids"]:
            lines.append(
                f"> **SPLIT LEAK — {len(s['split_leak_ids'])} ids in both splits.** "
                f"Every number measured on this build is void.\n"
            )
        lines += [
            "### 1.1 Clip-length distribution (sampled frames)", "",
            "| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            _describe_row("test sampled", c["lengths"]["test_sampled"]),
            _describe_row("train sampled", c["lengths"]["train_sampled"]),
            _describe_row("raw (pre-stride)", c["lengths"]["raw_total_frames"]),
            "",
        ]
        cov = c["kernel_coverage_test"]
        lines += [
            f"### 1.2 Score-head receptive field (kernel {cov['kernel']}) — lesson C27", "",
            f"- median clip length **{_fmt(cov['median_length'], 0)}** sampled frames",
            f"- median fraction of a clip inside one output timestep: "
            f"**{cov['median_coverage']:.1%}**",
            f"- clips entirely inside the kernel: **{_fmt(cov['clips_fully_covered'])}** "
            f"(**{cov['fraction_fully_covered']:.1%}**), holding "
            f"**{cov['fraction_frames_fully_covered']:.1%}** of all test frames",
            "- short-clip counts: "
            + ", ".join(f"`{k}` → {v}" for k, v in cov["clips_at_or_below"].items()),
            "",
        ]
        mil = c["mil_topk_floor_test"]
        lines += [
            f"### 1.3 MIL top-k floor (mil_topk_pct = {mil['topk_pct']})", "",
            f"- clips at `k = 1` (loss is a plain max): **{_fmt(mil['clips_at_k1'])}** "
            f"(**{mil['fraction_at_k1']:.1%}**)",
            f"- median k: **{_fmt(mil['median_k'], 1)}**",
            f"- k distribution: {mil['k_distribution']}",
            "",
        ]
        for key, table in c["subgroups"].items():
            if not table.get("present") or table["num_groups"] < 2:
                continue
            lines += [
                f"### 1.4 Subgroups by `{key}` ({table['num_groups']} groups)", "",
                "| group | clips | train | test | test frames | positive frames | pos. frac |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for group, row in table["groups"].items():
                frac = row["test_positive_fraction"]
                lines.append(
                    f"| {group} | {_fmt(row['clips'])} | {_fmt(row['train_clips'])} | "
                    f"{_fmt(row['test_clips'])} | {_fmt(row['test_frames'])} | "
                    f"{_fmt(row['test_positive_frames'])} | "
                    f"{f'{frac:.2%}' if frac is not None else '—'} |"
                )
            lines.append("")

    if "labels" in report:
        lab = report["labels"]
        pos = lab["positives"]
        lines += [
            "## 2. Label geometry", "",
            "| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            _describe_row("positives / test clip", pos["all_clips"]),
            _describe_row("positives / abnormal clip", pos["abnormal_clips"]),
            _describe_row(
                "positive frac. within abnormal",
                pos["positive_fraction_within_abnormal"],
            ),
            _describe_row("spans / abnormal clip", lab["spans"]["spans_per_abnormal_clip"]),
            _describe_row("span length", lab["spans"]["span_length"]),
            "",
            f"- abnormal clips with **exactly one** positive frame: "
            f"**{_fmt(pos['abnormal_clips_with_one_positive'])}**",
            f"- abnormal clips with <= 2 positive frames: "
            f"**{_fmt(pos['abnormal_clips_with_two_or_fewer'])}**",
            f"- multi-span abnormal clips: **{_fmt(lab['spans']['multi_span_clips'])}** "
            "(lesson C18 applies if > 0)",
            f"- **abnormal {lab['vanished_windows'].get('unit', 'clip')}s whose "
            f"window vanished: {_fmt(lab['vanished_windows']['count'])}**",
        ]
        negatives = lab["vanished_windows"].get("negative_windows_of_abnormal_clips", 0)
        if negatives:
            lines.append(
                f"- windows of abnormal clips holding no positive frame: "
                f"**{_fmt(negatives)}** — these are *correct negatives*, not vanished "
                "windows; producing them is the point of re-sharding"
            )
        lines.append("")
        if lab["vanished_windows"]["video_ids"]:
            lines += [
                "<details><summary>Vanished-window ids (exclude these at scoring)</summary>",
                "", "```", *lab["vanished_windows"]["video_ids"], "```", "</details>", "",
            ]

    if "protocol" in report:
        p = report["protocol"]
        fs, oracle, pairs = p["frame_share"], p["clip_constant_oracle"], p["pair_decomposition"]
        leak = p.get("clip_length_leak", {})
        lines += [
            "## 3. What the metric measures", "",
            "### 3.1 Where the frames live", "",
            "| clip kind | clips | frames | share of frames |",
            "|---|---:|---:|---:|",
        ]
        for key in ("all_normal", "mixed", "all_positive"):
            share = fs["frame_fraction"][key]
            lines.append(
                f"| {key} | {_fmt(fs['clips'][key])} | {_fmt(fs['frames'][key])} | "
                f"{f'{share:.2%}' if share is not None else '—'} |"
            )
        lines += [
            "",
            "### 3.2 The clip-level oracle (lesson C12)", "",
            "A model emitting **one constant score per clip**, ranking clips perfectly "
            "and localizing nothing, scores:",
            "",
            f"- micro AUC **{_fmt(oracle['auc_micro'])}**, micro AP "
            f"**{_fmt(oracle['ap_micro'])}**, macro AUC **0.5000** by construction",
            f"- {pairs['cross_clip_fraction']:.2%} of positive/negative frame pairs span "
            f"two clips ({_fmt(pairs['cross_clip_pairs'])} of "
            f"{_fmt(pairs['total_pairs'])}); only "
            f"{pairs['within_clip_fraction']:.2%} can be won by localization",
            "",
            "**Print this oracle beside every micro AUC measured on this corpus.**",
            "",
            "### 3.3 The clip-length leak (lesson C28)", "",
        ]
        if "auc_clip_level" in leak:
            lines += [
                "A constant-score-per-clip detector whose **only** input is the "
                "clip's frame count — no pixels, no model — scores:",
                "",
                f"- clip-level AUC **{_fmt(leak['auc_clip_level'])}** "
                f"(**{leak['direction']}** clips are the abnormal ones); "
                f"micro AUC **{_fmt(leak['auc_micro'])}**, micro AP "
                f"**{_fmt(leak['ap_micro'])}**, macro AUC **0.5000** by construction",
                f"- abnormal clip length T: median "
                f"**{_fmt(leak['abnormal_length']['percentiles']['p50'], 1)}**, "
                f"min {_fmt(leak['abnormal_length']['percentiles']['p0'], 1)}, "
                f"max **{_fmt(leak['abnormal_length']['percentiles']['p100'], 1)}**",
                f"- normal clip length T: median "
                f"**{_fmt(leak['normal_length']['percentiles']['p50'], 1)}**, "
                f"min {_fmt(leak['normal_length']['percentiles']['p0'], 1)}, "
                f"max {_fmt(leak['normal_length']['percentiles']['p100'], 1)}",
                f"- normal clips outside the abnormal length range entirely: "
                f"**{_fmt(leak['disjoint_normal_clips'])}** "
                f"({_fmt(leak['disjoint_normal_frames'])} frames)",
                "",
                "**Any arm that does not beat this baseline is unmeasured.**",
                "",
            ]
        else:
            lines += [
                f"Not computed: {leak.get('skipped', 'unavailable')}.", "",
            ]
        lines += [
            "### 3.4 Per-clip AUC resolution", "",
            f"- two-class clips (the only ones `auc_macro` averages): "
            f"**{_fmt(p['macro_resolution']['two_class_clips'])}**; single-class: "
            f"{_fmt(p['macro_resolution']['single_class_clips'])}",
            "",
            "A clip with `p` positives and `n` negatives has `p*n` orderable pairs, so "
            "its AUC only takes values on a `1/(p*n)` grid. A coarse grid makes "
            "`auc_macro` honest but low-resolution — quote it with these counts.",
            "",
            "| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            _describe_row("AUC grid step 1/(pos*neg)", p["macro_resolution"]["auc_grid_step"]),
            "",
            f"- `--score-norm auto` resolves to **{p['score_norm_auto']['resolves_to']}** "
            f"(normal-clip fraction {p['score_norm_auto']['normal_clip_fraction']:.3f} vs "
            f"threshold {p['score_norm_auto']['threshold']})",
            "",
        ]
        if "scored_run" in p:
            run = p["scored_run"]
            fl = run["flatness"]
            lines += [
                "### 3.5 A scored run's curves", "",
                f"`{run['scores_dir']}` — {_fmt(run['clips'])} clips, "
                f"auc_macro **{_fmt(run['auc_macro'])}** over {run['auc_macro_videos']} clips",
                "",
                f"- between-clip / within-clip score variance: "
                f"**{_fmt(fl['between_over_within'], 2)}**",
                f"- median within-clip score range: "
                f"**{_fmt(fl['within_clip_range']['percentiles'].get('p50'), 3)}**",
                f"- literally constant curves: **{_fmt(fl['constant_curve_clips'])}** clips",
                "",
            ]

    if "features" in report:
        f = report["features"]
        lines += [
            "## 4. The cached frozen-CLIP features", "",
            f"`{f['clip_dir']}` — {_fmt(f['coverage']['found'])} of "
            f"{_fmt(f['coverage']['requested'])} test clips found"
            + (f", **{f['coverage']['missing']} missing**" if f["coverage"]["missing"] else ""),
            "",
        ]
        st = f["stats"]
        if st.get("clips"):
            lines += [
                f"- {_fmt(st['frames'])} frames x {st['dim']} dims; "
                f"mean L2 norm {_fmt(st['l2_norm']['mean'], 2)}",
                f"- dimensions holding 90 % of the variance: "
                f"**{st['effective_dims_90pct_variance']}** of {st['dim']}",
                "",
            ]
        ac = f["temporal_autocorrelation"]["cosine_by_lag"]
        lines += [
            "### 4.1 Temporal autocorrelation (cosine between frame t and t+lag)", "",
            "| lag (sampled frames) | " + " | ".join(ac) + " |",
            "|---|" + "---|" * len(ac),
            "| mean cosine | " + " | ".join(_fmt(v["mean"], 3) for v in ac.values()) + " |",
            "",
            "A value near 1.0 at lag 1 means consecutive sampled frames are nearly "
            "identical, i.e. the stride is finer than the content changes.",
            "",
        ]
        vd = f["variance_decomposition"]
        if vd.get("clips"):
            lines += [
                f"- between-clip / within-clip **feature** variance: "
                f"**{_fmt(vd['between_over_within'], 2)}** "
                "(high = the embedding encodes scene identity more than dynamics)",
                "",
            ]
        fp = f.get("frame_linear_probe", {})
        cp = f.get("clip_linear_probe", {})
        lines += ["### 4.2 Supervised linear probe — the representation ceiling", ""]
        if fp.get("ran"):
            lines += [
                "| probe | AUC | macro AUC | AP | AP baseline | folds | units |",
                "|---|---:|---:|---:|---:|---:|---:|",
                f"| frame-level | {_fmt(fp['auc_micro'])} | {_fmt(fp['auc_macro'])} | "
                f"{_fmt(fp['ap_micro'])} | {_fmt(fp['ap_baseline'])} | {fp['folds']} | "
                f"{_fmt(fp['frames'])} frames |",
            ]
            if cp.get("ran"):
                lines.append(
                    f"| clip-level (mean-pooled) | {_fmt(cp['auc'])} | — | {_fmt(cp['ap'])} | "
                    f"{_fmt(cp['ap_baseline'])} | {cp['folds']} | {_fmt(cp['clips'])} clips |"
                )
            lines += [
                "",
                "Grouped cross-validation by clip, so no clip's frames score themselves. "
                "This is the *supervised ceiling* of these features: a trained arm cannot "
                "be expected to beat it, and a large gap below it is a supervision "
                "problem, not a representation problem.",
                "",
            ]
        else:
            lines += [f"Probe did not run: {fp.get('reason', 'not requested')}.", ""]
        if "flow" in f and f["flow"].get("clips"):
            fl = f["flow"]
            lines += [
                "### 4.3 RAFT flow descriptors (train split only)", "",
                f"- {_fmt(fl['clips'])} clips, {_fmt(fl['frames'])} frames, "
                f"**{fl['raw_dims']} raw dims**; dead dims: {fl['dead_dims']}; "
                f"non-finite frames: {fl['nonfinite_frames']}",
                f"- {fl['note']}. Flow is train-time only and the train split carries no "
                "frame labels, so no flow-versus-label correlation is computable.",
                "",
            ]
            top = fl.get("raw_energy_share", [])[:5]
            if top:
                lines += [
                    "Where the target's scale comes from — `e_O = s @ M` with "
                    "`M ~ N(0, 1/23)`, so every one of the 256 dimensions is scaled by "
                    "`mean_j E[s_j^2]`:",
                    "",
                    "| raw stat | mean | std | share of E[s²] |",
                    "|---|---:|---:|---:|",
                    *[
                        f"| `{row['name']}` | {row['mean']:.3f} | {row['std']:.3f} | "
                        f"{row['energy_share']:.1%} |"
                        for row in top
                    ],
                    "",
                ]
            tgt = fl.get("target")
            if tgt:
                zero_mse = tgt["mse_zero_predictor"]
                ratio = (
                    tgt["predicted_second_moment_from_raw"] / zero_mse
                    if zero_mse > 0
                    else float("nan")
                )
                lines += [
                    "#### 4.3.1 `L_KIP_rec` baselines — what a measured `kip_rec` means",
                    "",
                    "| predictor | MSE it scores | reads |",
                    "|---|---:|---|",
                    f"| all-zeros | {tgt['mse_zero_predictor']:.3f} | "
                    "the loss at step 1 of an untrained PMG head |",
                    f"| one global mean vector | {tgt['mse_global_mean_predictor']:.3f} | "
                    "**the baseline any measured `kip_rec` must beat** |",
                    f"| each item's own mean | {tgt['mse_item_mean_predictor']:.3f} | "
                    "an oracle that knows item identity and nothing else |",
                    "",
                    f"- **R^2 of a measured `kip_rec` = 1 - kip_rec / "
                    f"{tgt['mse_global_mean_predictor']:.3f}.**",
                    f"- {tgt['between_item_share']:.1%} of the target's variance is "
                    "*between* items: that share is reachable by predicting item "
                    "identity, with no within-item dynamics at all.",
                    f"- Projection check: predicted "
                    f"`{tgt['predicted_second_moment_from_raw']:.3f}` vs measured "
                    f"`{zero_mse:.3f}` (ratio {ratio:.3f} — away from 1.0 means the "
                    "cache and the seeded projection disagree).",
                    "",
                ]

    return "\n".join(line for line in lines if line is not None) + "\n"


def write_plots(report: dict[str, Any], files: corpus.DatasetFiles, out_dir: Path) -> list[Path]:
    """Four figures: length histogram, positives histogram, autocorrelation, frame share."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    kernel = report["score_head_kernel"]
    lengths = [len(v) for v in files.frame_labels_test.values()]

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(lengths, bins=min(60, max(10, len(set(lengths)))), color="#4477aa")
    axis.axvline(kernel, color="#cc3311", linestyle="--", label=f"score head kernel = {kernel}")
    axis.set_xlabel("sampled frames per test clip")
    axis.set_ylabel("clips")
    axis.set_title(f"{report['dataset']} — clip length vs receptive field")
    axis.legend()
    written.append(_save(figure, out_dir / "clip_length_hist.png"))

    positives = [sum(v) for v in files.frame_labels_test.values() if any(v)]
    if positives:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(positives, bins=min(40, max(5, len(set(positives)))), color="#228833")
        axis.set_xlabel("positive frames per abnormal test clip")
        axis.set_ylabel("clips")
        axis.set_title(f"{report['dataset']} — positives per abnormal clip")
        written.append(_save(figure, out_dir / "positives_per_clip_hist.png"))

    ac = report.get("features", {}).get("temporal_autocorrelation", {}).get("cosine_by_lag", {})
    means = [v["mean"] for v in ac.values() if v["mean"] is not None]
    if means:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(range(1, len(means) + 1), means, marker="o", color="#ee7733")
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("lag (sampled frames)")
        axis.set_ylabel("mean cosine similarity")
        axis.set_title(f"{report['dataset']} — CLIP feature temporal autocorrelation")
        axis.grid(alpha=0.3)
        written.append(_save(figure, out_dir / "temporal_autocorrelation.png"))

    share = report.get("protocol", {}).get("frame_share")
    oracle = report.get("protocol", {}).get("clip_constant_oracle")
    if share and oracle:
        figure, axis = plt.subplots(figsize=(7, 4))
        keys = ["all_normal", "mixed", "all_positive"]
        axis.bar(keys, [share["frames"][k] for k in keys],
                 color=["#bbbbbb", "#4477aa", "#cc3311"])
        axis.set_ylabel("sampled frames")
        axis.set_title(
            f"{report['dataset']} — frames by clip kind "
            f"(clip oracle micro AUC = {oracle['auc_micro']:.4f})"
        )
        written.append(_save(figure, out_dir / "frame_share.png"))

    LOGGER.info("Wrote %d plots to %s", len(written), out_dir)
    return written


def _save(figure: Figure, target: Path) -> Path:
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def compare_reports(reports: list[dict[str, Any]]) -> str:
    """Side-by-side Markdown for two or more corpora — the cross-corpus table."""
    names = [r["dataset"] for r in reports]

    def row(label: str, getter: Any, digits: int = 4) -> str:
        cells = []
        for r in reports:
            try:
                cells.append(_fmt(getter(r), digits))
            except (KeyError, TypeError, ZeroDivisionError):
                cells.append("—")
        return f"| {label} | " + " | ".join(cells) + " |"

    lines = [
        "# EDA — cross-corpus comparison",
        "",
        "| property | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
        row("test clips", lambda r: r["corpus"]["splits"]["test_clips"]),
        row("test sampled frames", lambda r: r["corpus"]["splits"]["test_sampled_frames"]),
        row("test positive frames", lambda r: r["corpus"]["splits"]["test_positive_frames"]),
        row("all-normal test clips",
            lambda r: r["protocol"]["frame_share"]["clips"]["all_normal"]),
        row("frame share in all-normal clips",
            lambda r: r["protocol"]["frame_share"]["frame_fraction"]["all_normal"], 4),
        row("**median clip length T**",
            lambda r: r["corpus"]["kernel_coverage_test"]["median_length"], 1),
        row("clips with T <= kernel",
            lambda r: r["corpus"]["kernel_coverage_test"]["fraction_fully_covered"], 4),
        row("median kernel coverage",
            lambda r: r["corpus"]["kernel_coverage_test"]["median_coverage"], 4),
        row("clips at MIL k = 1",
            lambda r: r["corpus"]["mil_topk_floor_test"]["fraction_at_k1"], 4),
        row("median positives / abnormal clip",
            lambda r: r["labels"]["positives"]["abnormal_clips"]["percentiles"]["p50"], 1),
        row("vanished windows", lambda r: r["labels"]["vanished_windows"]["count"]),
        row("**clip-oracle micro AUC**",
            lambda r: r["protocol"]["clip_constant_oracle"]["auc_micro"]),
        row("**length-only micro AUC**",
            lambda r: r["protocol"]["clip_length_leak"]["auc_micro"]),
        row("length-only clip AUC",
            lambda r: r["protocol"]["clip_length_leak"]["auc_clip_level"]),
        row("cross-clip pair fraction",
            lambda r: r["protocol"]["pair_decomposition"]["cross_clip_fraction"], 4),
        row("score-norm auto resolves to",
            lambda r: r["protocol"]["score_norm_auto"]["resolves_to"]),
        row("feature between/within variance",
            lambda r: r["features"]["variance_decomposition"]["between_over_within"], 2),
        row("CLIP cosine at lag 1",
            lambda r: r["features"]["temporal_autocorrelation"]["cosine_by_lag"]["lag1"]["mean"],
            3),
        row("**frame probe macro AUC**",
            lambda r: r["features"]["frame_linear_probe"]["auc_macro"]),
        row("clip probe AUC", lambda r: r["features"]["clip_linear_probe"]["auc"]),
        "",
        "A corpus whose clip-oracle micro AUC is high and whose median T is at or "
        "below the score-head kernel cannot support a frame-level claim, whatever "
        "its micro AUC says (lessons C12, C27). A corpus whose length-only micro "
        "AUC is high does not support one at all: the label is readable without "
        "the pixels (lesson C28).",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write ``eda_report.json`` and ``eda_report.md``; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / constants.EDA_REPORT_JSON_FILENAME
    md_path = out_dir / constants.EDA_REPORT_MD_FILENAME
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=_json_default)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    LOGGER.info("Wrote %s and %s", json_path, md_path)
    return json_path, md_path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON-serializable: {type(value)!r}")


__all__ = [
    "build_report",
    "compare_reports",
    "render_markdown",
    "write_plots",
    "write_report",
]
