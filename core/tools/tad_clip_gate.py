"""v2 clip-level gate for TAD: ``s(v, t) = c(v) * minmax_v(y(v, t))`` (addendum §27, Amendment 16).

§26 found the deployed A3-ign blind at clip level on TAD: normal videos peak as high as accident
videos. ``c(v)`` is a length-blind logistic gate on the mean raw CLIP feature of a video, fitted on
MSAD-train (both classes from the same cameras, the same ``*_ncc`` extractor as ``TAD_ncc``) and
never on TAD. Because ``c > 0`` rescales a whole video, every within-video ranking -- hence macro
AUC -- is unchanged (G3); only the between-video interleaving that micro AUC reads can move.

Subcommands::

    fit      MSAD-train features + labels -> gate.npz (scaler + logistic weights, no pickle)
    score    gate.npz + a feature cache   -> {video: {"c": p, "rows": L}}
    readout  §26's score files + gate scores on TAD-train (dev) and TAD-test -> tadg_readout.*

The sentence, mechanically (G6): "a clip-level gate trained on MSAD restores clip-level
discrimination on TAD" iff the paired t95 over seeds **and** the video bootstrap of
micro(s) - micro(A3-ign) both lie above 0; "costs" iff both lie below; otherwise "no clip-level
gain is shown". If the dev gate (G5) fails, the test is not read.

CLI::

    python -m core.tools.tad_clip_gate fit --clip-dir cache/clip/MSAD_ncc \\
        --labels data/MSAD/labels_train.json --out gate.npz
    python -m core.tools.tad_clip_gate score --gate gate.npz --clip-dir cache/clip/TAD_ncc \\
        --ids-file ids.txt --out c.json
    python -m core.tools.tad_clip_gate readout --tadz-root outputs/v2_tadz --dev c_dev.json \\
        --dev-labels dev.json --test c_test.json [--msad-test c.json --msad-test-labels m.json] \\
        --out-dir outputs/v2_tadg
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from core import constants
from core.metrics import frame_auc
from core.tools.e3_readout import REPORTS, excludes_zero, paired_interval, seed_name
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.tadz_readout import (
    Curves,
    check_labels,
    clip_aucs,
    load_run,
    micro,
    seed_average,
    video_auc,
)

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "tadg_readout.json"
READOUT_MD = "tadg_readout.md"
GateScores = dict[str, dict[str, float]]


# ---------------------------------------------------------------------------
# gate: fit / score
# ---------------------------------------------------------------------------
def mean_features(clip_dir: Path, ids: list[str]) -> tuple[np.ndarray, list[int]]:
    """``(len(ids), 512)`` mean raw CLIP feature per video, and each video's row count."""
    pooled, rows = [], []
    for vid in ids:
        path = clip_dir / f"{vid}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"G9: no feature file for {vid} in {clip_dir}")
        x = np.asarray(np.load(path), dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != constants.CLIP_FEATURE_DIM or len(x) == 0:
            raise ValueError(
                f"G9(d): {path} has shape {x.shape}, expected (L, {constants.CLIP_FEATURE_DIM})"
            )
        pooled.append(x.mean(axis=0))
        rows.append(len(x))
    return np.stack(pooled), rows


def fit_gate(features: np.ndarray, target: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """G2: standardized, class-balanced logistic regression; returned as plain arrays."""
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("the gate needs both classes")
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=constants.V2_TADG_C,
        max_iter=constants.V2_TADG_MAX_ITER,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(scaler.transform(features), target)
    return {
        "mean": np.asarray(scaler.mean_),
        "scale": np.asarray(scaler.scale_),
        "coef": np.asarray(model.coef_).ravel(),
        "intercept": np.asarray(model.intercept_),
    }


def gate_probability(gate: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    logit = ((features - gate["mean"]) / gate["scale"]) @ gate["coef"] + gate["intercept"][0]
    return np.asarray(1.0 / (1.0 + np.exp(-logit)), dtype=np.float64)


def read_labels(path: Path) -> dict[str, int]:
    """``{video: 0|1}``; a list value (frame labels) is reduced to its max."""
    raw = json.loads(path.read_text())
    return {v: int(max(y) if isinstance(y, list) else y) for v, y in raw.items()}


def cmd_fit(args: argparse.Namespace) -> dict[str, Any]:
    labels = read_labels(args.labels)
    ids = sorted(labels)
    target = np.array([labels[v] for v in ids], dtype=np.int64)
    counts = (int(target.sum()), int(len(target) - target.sum()))
    if args.expect_counts and counts != tuple(args.expect_counts):
        raise ValueError(
            f"G9(b): train split is {counts} abnormal/normal, expected {tuple(args.expect_counts)}"
        )
    features, _ = mean_features(args.clip_dir, ids)
    gate = fit_gate(features, target, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **gate)
    train_auc = frame_auc(gate_probability(gate, features), target)
    summary = {
        "videos": len(ids),
        "abnormal_normal": counts,
        "train_clip_auc": train_auc,
        "clip_dir": str(args.clip_dir),
    }
    write_json_atomic(args.out.with_suffix(".json"), summary)
    LOGGER.info(
        "gate fitted on %d videos (%s), train clip AUC %.4f -> %s",
        len(ids),
        counts,
        train_auc,
        args.out,
    )
    return summary


def cmd_score(args: argparse.Namespace) -> GateScores:
    with np.load(args.gate) as data:
        gate = {k: np.asarray(data[k]) for k in data.files}
    ids = [line.strip() for line in args.ids_file.read_text().splitlines() if line.strip()]
    features, rows = mean_features(args.clip_dir, ids)
    probs = gate_probability(gate, features)
    out = {v: {"c": float(p), "rows": float(n)} for v, p, n in zip(ids, probs, rows, strict=True)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out, out)
    LOGGER.info("gate scored %d videos -> %s", len(ids), args.out)
    return out


# ---------------------------------------------------------------------------
# fusion + statistics
# ---------------------------------------------------------------------------
def minmax(x: np.ndarray) -> np.ndarray:
    low, high = x.min(), x.max()
    return np.asarray((x - low) / (high - low + constants.SCORE_NORM_EPS), dtype=np.float64)


def fuse(scores: Curves, gate: dict[str, float]) -> Curves:
    """G3: ``c(v) · minmax_v(y)``."""
    return {v: gate[v] * minmax(y) for v, y in scores.items()}


def clip_level(values: dict[str, float], labels: dict[str, int]) -> float:
    ids = sorted(labels)
    return frame_auc(np.array([values[v] for v in ids]), np.array([labels[v] for v in ids]))


def micro_bootstrap(a: Curves, b: Curves, labels: Curves, seed: int) -> dict[str, float]:
    """Percentile CI of micro(a) - micro(b), resampling whole videos (each its own cluster)."""
    ids = sorted(labels)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(constants.V2_E1_BOOTSTRAP, len(ids)))
    deltas = np.empty(len(draws))
    for i, draw in enumerate(draws):
        pick = [ids[j] for j in draw]
        y = np.concatenate([labels[v] for v in pick])
        if y.min() == y.max():
            deltas[i] = np.nan
            continue
        deltas[i] = frame_auc(np.concatenate([a[v] for v in pick]), y) - frame_auc(
            np.concatenate([b[v] for v in pick]), y
        )
    alpha = (1.0 - constants.V2_E1_CI) / 2.0
    low, high = np.nanquantile(deltas, [alpha, 1.0 - alpha])
    return {"mean": micro(a, labels) - micro(b, labels), "low": float(low), "high": float(high)}


def gate_sentence(paired: dict[str, Any], boot: dict[str, float]) -> str:
    if paired["low"] > 0 and boot["low"] > 0:
        return "a clip-level gate trained on MSAD restores clip-level discrimination on TAD"
    if paired["high"] < 0 and boot["high"] < 0:
        return "the MSAD gate costs on TAD"
    return "no clip-level gain of the MSAD gate is shown on TAD"


def length_ruler(rows: dict[str, float], labels: dict[str, int]) -> float:
    """Clip-level AUC of 'shorter video = accident' (TAD_SETUP §8.1's orientation)."""
    return clip_level({v: -rows[v] for v in labels}, labels)


def _gate_values(gate: GateScores) -> dict[str, float]:
    return {v: g["c"] for v, g in gate.items()}


def dev_read(dev: GateScores, dev_labels: dict[str, int], test_ids: set[str]) -> dict[str, Any]:
    """G5 + G9(c): the go/no-go on TAD-train, read before any test number."""
    overlap = sorted(set(dev_labels) & test_ids)
    if overlap:
        raise ValueError(f"G9(c): TAD-train and TAD-test share {overlap[:5]}")
    if set(dev) != set(dev_labels):
        raise ValueError("G9(c): dev gate scores and dev labels name different videos")
    n_abn = sum(dev_labels.values())
    if len(dev_labels) < constants.V2_TADG_DEV_MIN_VIDEOS or not 0 < n_abn < len(dev_labels):
        raise ValueError(f"G9(c): dev split has {len(dev_labels)} videos / {n_abn} abnormal")
    auc = clip_level(_gate_values(dev), dev_labels)
    return {
        "videos": len(dev_labels),
        "abnormal": n_abn,
        "gate_clip_auc": auc,
        "length_ruler_clip_auc": length_ruler({v: g["rows"] for v, g in dev.items()}, dev_labels),
        "bar": constants.V2_TADG_DEV_MIN_AUC,
        "passed": auc >= constants.V2_TADG_DEV_MIN_AUC,
    }


def cmd_readout(args: argparse.Namespace) -> dict[str, Any]:
    seeds = [seed_name(s) for s in args.seeds]
    arm = constants.V2_TADG_ARM
    runs = {constants.V2_TADZ_REFERENCE: load_run(args.tadz_root / constants.V2_TADZ_REFERENCE)}
    for s in seeds:
        runs[f"{arm}/{s}"] = load_run(args.tadz_root / arm / s)
    labels = check_labels(runs)  # G9(a): §26's T9(b)(d)
    t0 = runs[constants.V2_TADZ_REFERENCE][0]
    if abs(micro(t0, labels) - constants.V2_TADZ_T0_MICRO) > constants.V2_TADZ_T0_TOLERANCE:
        raise ValueError("G9(a): T0 no longer re-scores at Gate T0's micro; stop")
    video_labels = {v: int(g.max() > 0) for v, g in labels.items()}
    name_labels = {v: int(not v.startswith(constants.V2_TADG_NORMAL_PREFIX)) for v in labels}
    if name_labels != video_labels:
        raise ValueError("G9(c): id-derived labels do not reproduce the test labels")

    dev = json.loads(args.dev.read_text())
    readout: dict[str, Any] = {
        "addendum": "PREREG_ADDENDUM.md §27 (Amendment 16, G1-G10)",
        "seeds": seeds,
        "dev": dev_read(dev, read_labels(args.dev_labels), set(labels)),
    }
    if args.msad_test and args.msad_test_labels:
        msad = json.loads(args.msad_test.read_text())
        readout["msad_test_gate_clip_auc"] = clip_level(
            _gate_values(msad), read_labels(args.msad_test_labels)
        )
    out = args.out_dir / REPORTS
    out.mkdir(parents=True, exist_ok=True)
    if not readout["dev"]["passed"]:
        readout["sentence"] = "the MSAD gate does not reach TAD (G5 failed); TAD-test not read"
        write_json_atomic(out / READOUT_JSON, readout)
        write_text_atomic(out / READOUT_MD, render_markdown(readout))
        LOGGER.info("G5 failed -> %s", out)
        return readout

    test = json.loads(args.test.read_text())
    if set(test) != set(labels):
        raise ValueError("G9: test gate scores name other videos than §26's runs")
    c = _gate_values(test)
    per_seed = {s: runs[f"{arm}/{s}"][0] for s in seeds}
    fused = {s: fuse(y, c) for s, y in per_seed.items()}
    for s in seeds:  # G3: the gate cannot move a within-video ranking
        a, b = clip_aucs(fused[s], labels), clip_aucs(per_seed[s], labels)
        worst = max(abs(a[v] - b[v]) for v in a)
        if worst > constants.V2_TADG_MACRO_TOL:
            raise ValueError(f"G9(e): {s} macro moved by {worst:.2e} under the gate")
    paired = paired_interval(
        {s: micro(fused[s], labels) for s in seeds},
        {s: micro(per_seed[s], labels) for s in seeds},
        constants.V2_E3_T95,
    )
    avg = seed_average(list(per_seed.values()))
    fused_avg = fuse(avg, c)
    boot = micro_bootstrap(fused_avg, avg, labels, args.seed)
    t0_gate = {v: float(y.mean()) for v, y in t0.items()}
    lago_fused = fuse(avg, t0_gate)
    readout.update(
        {
            "endpoint": {"paired_t95": paired, "video_bootstrap": boot},
            "sentence": gate_sentence(paired, boot),
            "levels": {
                arm: {"micro": micro(avg, labels), "clip_auc": video_auc(avg, labels)},
                f"c x minmax({arm})": {
                    "micro": micro(fused_avg, labels),
                    "clip_auc": video_auc(fused_avg, labels),
                    "per_seed_micro": {s: micro(fused[s], labels) for s in seeds},
                },
                "T0 (LaGoVAD released)": {
                    "micro": micro(t0, labels),
                    "clip_auc": video_auc(t0, labels),
                },
                f"mean(T0) x minmax({arm}) (form seen on this set)": {
                    "micro": micro(lago_fused, labels),
                    "clip_auc": video_auc(lago_fused, labels),
                },
            },
            "macro": float(np.mean(list(clip_aucs(avg, labels).values()))),
            "gate_test_clip_auc": clip_level(c, video_labels),
            "test_length_ruler_clip_auc": length_ruler(
                {v: float(len(g)) for v, g in labels.items()}, video_labels
            ),
            "vs_t0_micro_bootstrap": micro_bootstrap(fused_avg, t0, labels, args.seed),
        }
    )
    write_json_atomic(out / READOUT_JSON, readout)
    write_text_atomic(out / READOUT_MD, render_markdown(readout))
    LOGGER.info("TAD gate: %s -> %s", readout["sentence"], out)
    return readout


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _ci(c: dict[str, float]) -> str:
    return f"{c['mean']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def render_markdown(r: dict[str, Any]) -> str:
    d = r["dev"]
    verdict = "PASS" if d["passed"] else "FAIL"
    lines = [
        "# TAD clip-level gate — `s = c(v) · minmax_v(y_A3ign)`, `c` fitted on MSAD-train",
        "",
        f"Rule: {r['addendum']}. `B` = A3-ign is not re-decided; "
        "macro is unchanged by construction (G3).",
        "",
        f"**Sentence (G6): {r['sentence']}.**",
        "",
        "## Dev gate (G5) — TAD-train, never fitted on",
        "",
        f"* {d['videos']} videos ({d['abnormal']} abnormal): gate clip-level AUC "
        f"**{d['gate_clip_auc']:.4f}** (bar {d['bar']:.2f} → {verdict}) · length ruler "
        f"{d['length_ruler_clip_auc']:.4f}",
    ]
    if "msad_test_gate_clip_auc" in r:
        lines.append(
            f"* MSAD-test (in-domain sanity): gate clip-level AUC "
            f"{r['msad_test_gate_clip_auc']:.4f}"
        )
    if "endpoint" not in r:
        lines.append("")
        return "\n".join(lines)
    e = r["endpoint"]
    t = e["paired_t95"]
    mark = " (excludes 0)" if excludes_zero(t) else ""
    lines += [
        "",
        "## Endpoint (G6) — TAD-test micro, Δ = micro(s) - micro(A3-ign)",
        "",
        "| paired t95 over seeds | video bootstrap (seed-averaged) |",
        "|---|---|",
        f"| {_ci(t)}{mark} | {_ci(e['video_bootstrap'])} |",
        "",
        f"## Levels (seed-averaged; macro {r['macro']:.4f} for every row built on A3-ign)",
        "",
        "| | micro | clip-level AUC |",
        "|---|---:|---:|",
    ]
    for name, lv in r["levels"].items():
        lines.append(f"| {name} | {lv['micro']:.4f} | {lv['clip_auc']:.4f} |")
    lines += [
        "",
        "## Printed (G7)",
        "",
        f"* gate alone, clip-level AUC on TAD-test {r['gate_test_clip_auc']:.4f} · length ruler "
        f"{r['test_length_ruler_clip_auc']:.4f}",
        f"* micro(s) - micro(T0), video bootstrap {_ci(r['vs_t0_micro_bootstrap'])} "
        "— not matched (T6)",
        "* micro rulers (§26): clip oracle 0.9226 · length ruler 0.8968; never beside 89.56",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit the gate on a labelled feature cache (G2, G4)")
    fit.add_argument("--clip-dir", type=Path, required=True)
    fit.add_argument("--labels", type=Path, required=True, help="{video: 0|1} (labels_train.json)")
    fit.add_argument("--out", type=Path, required=True, help="gate .npz")
    fit.add_argument(
        "--expect-counts",
        type=int,
        nargs=2,
        default=list(constants.V2_TADG_MSAD_TRAIN),
        metavar=("ABNORMAL", "NORMAL"),
        help="G9(b) guard; pass 0 0 to skip",
    )
    fit.add_argument("--seed", type=int, default=constants.SEED)
    score = sub.add_parser("score", help="score videos with a fitted gate")
    score.add_argument("--gate", type=Path, required=True)
    score.add_argument("--clip-dir", type=Path, required=True)
    score.add_argument("--ids-file", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    read = sub.add_parser("readout", help="dev gate, then the TAD-test endpoint (G5-G7)")
    read.add_argument("--tadz-root", type=Path, required=True, help="§26's output root")
    read.add_argument("--dev", type=Path, required=True, help="gate scores on TAD-train")
    read.add_argument("--dev-labels", type=Path, required=True)
    read.add_argument("--test", type=Path, required=True, help="gate scores on TAD-test")
    read.add_argument("--msad-test", type=Path, default=None)
    read.add_argument("--msad-test-labels", type=Path, default=None)
    read.add_argument("--out-dir", type=Path, required=True)
    read.add_argument("--seeds", type=int, nargs="+", default=list(constants.V2_E3_SEEDS))
    read.add_argument("--seed", type=int, default=constants.SEED, help="bootstrap seed")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "fit" and args.expect_counts == [0, 0]:
        args.expect_counts = None
    {"fit": cmd_fit, "score": cmd_score, "readout": cmd_readout}[args.command](args)


if __name__ == "__main__":
    main()
