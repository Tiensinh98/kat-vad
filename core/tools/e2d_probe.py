"""v2 P4 -- E2(d): VideoMAE V2-B vs V2-S on DoTA-CAP-dev (frozen-feature probes, no training).

Pre-registered in ``core/docs/v2/PREREG_ADDENDUM.md`` §12 (N1-N12), on the endpoint fixed by
Amendment 4 (D13: DoTA-CAP). Proposal §10.2 E2(d), applied as written:

* **Rows (N1):** ``dota_cap_dev`` at protocol B -- CLIP ``DoTA_s1_ncc[::3]`` and VideoMAE
  ``DoTA_CAP_s1_squash[::3]``, whole clip, labels = the D11 loader's stride-3 labels.
* **Representations (N2):** A2 ``[x ; u]`` vs ``x``; A3 ``[x - R2(x) ; u - R2(u)]`` vs
  ``x - R2(x)`` (per-clip median). ``s``, ``c``, ``m_u`` and ``sigma_u`` are per-channel affine maps
  that the probe's standardization absorbs, so they are not applied.
* **Probes:** in-domain (N3) = the ``core.eda`` logistic frame probe, 5-fold CV grouped by source
  YouTube video; transfer (N4) = the same probe fitted on the 300 frozen K sources (T2-train,
  whole, s8) and scored on ``dota_cap_dev``.
* **Δ (N5):** per-clip paired AUC(with ``u``) - AUC(CLIP-only), cluster bootstrap over source
  videos. **Eligible (N6)** iff in-domain mean Δ >= +0.10 or transfer mean Δ >= +0.03, for A2 or
  A3. **Pick (N7):** largest A3 transfer mean Δ; within 0.02 the cheaper encoder; none eligible
  -> ``MOTION_DROPPED``.
* **Printed only (N9, N10):** ``u`` alone, position ``p`` alone, ``[base;p]`` vs ``[with;p]``,
  Δ per share bin, the T2-side source-grouped read, and D15 representativeness (CLIP-only probe
  on all of ``dota_dev`` vs its DoTA-CAP part; kept vs dropped clip length, share and category;
  A0's per-clip macro when ``--a0-clip-aucs`` is given).
* **Gates (N12)** run before any probe: split sha1, row counts, cache manifests, K-source sha1.

DoTA-eval is sealed: only ``dota_dev`` and ``dota_cap_dev`` are loaded.

CLI::

    python -m core.tools.e2d_probe \\
        --dota-s1-dir cache/clip/DoTA_s1_ncc --metadata data/DoTA/metadata_val.json \\
        --split-file data/DoTA/val_split.txt \\
        --k-ids-file outputs/REPORTS/v2_K/k_sources.txt \\
        --k-manifest outputs/REPORTS/v2_K/k_sources_manifest.json \\
        --annotation data/DADA2000Origin/DADA2000/dada标注.xlsx \\
        --census data/DADA2000_orig/counts/resolved_census.json \\
        --t2-clip-dir cache/clip/DADA2000_orig --out-dir outputs/v2/REPORTS/v2_E2d \\
        [--a0-clip-aucs outputs/v2/REPORTS/v2_A0_dev/clip_aucs.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.crn.reference import reference
from core.data.dada_origin import load_counts
from core.data.dota import DotaRecord, parse_metadata, read_split_ids, resized_frame_labels
from core.data.v2_splits import dota_group, lines_sha1, load_split, share_bin
from core.eda.features import transfer_scores
from core.metrics import cluster_bootstrap_ci, frame_auc
from core.tools import dota_cap, extract_video_features
from core.tools.kill_switch_probe import (
    load_features,
    per_source_auc,
    position_features,
    source_labels,
    write_json_atomic,
    write_text_atomic,
)

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "e2d_readout.json"
READOUT_MD = "e2d_readout.md"
ARMS = ("A2", "A3")
PICK_ARM = "A3"  # N7
PROBES = ("in_domain", "transfer")
DECISION_SETS = ("with", "base")  # Δ = with - base
PRINTED_SETS = ("u", "p", "base_p", "with_p")
MOTION_DROPPED = "MOTION_DROPPED"
DEV = "dev"
CAP = "cap"
DROPPED = "dropped"


@dataclass(frozen=True)
class ProbeCorpus:
    """Per-clip CLIP rows, VideoMAE rows per encoder, labels and bootstrap cluster."""

    x: dict[str, np.ndarray]
    u: dict[str, dict[str, np.ndarray]]
    y: dict[str, np.ndarray]
    group: dict[str, str]

    def two_class(self) -> list[str]:
        return sorted(v for v in self.y if 0 < int(self.y[v].sum()) < len(self.y[v]))


# ---------------------------------------------------------------------------
# gates (N12) and loading (N1, N4)
# ---------------------------------------------------------------------------
def check_manifest(cache_dir: Path, expected: dict[str, Any]) -> None:
    """The cache's ``video_manifest.json`` must carry ``expected``'s values (C2)."""
    path = cache_dir / constants.VIDEO_MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: a cache without a manifest is unverifiable")
    found = json.loads(path.read_text(encoding="utf-8"))
    diff = sorted(k for k in expected if found.get(k) != expected[k])
    if diff:
        raise ValueError(f"{cache_dir} was built with different {diff} than E2(d) expects")


def dota_video_dir(root: Path, encoder: str) -> Path:
    return root / encoder / f"{constants.DOTA_CAP_DATASET}_s1_squash"


def t2_video_dir(root: Path, encoder: str) -> Path:
    stride = constants.FRAME_STRIDE
    return root / encoder / f"{constants.DADA_ORIGIN_DATASET}_s{stride}_squash"


def dota_cap_alignment_sha256(split_dir: Path) -> str:
    manifest = split_dir / constants.V2_DOTA_CAP_MANIFEST_FILENAME
    return str(json.loads(manifest.read_text(encoding="utf-8"))["source"]["alignment_sha256"])


def load_dota(
    ids: list[str],
    records: dict[str, DotaRecord],
    s1_dir: Path,
    video_dirs: dict[str, Path],
    stride: int,
) -> ProbeCorpus:
    """Protocol-B rows of ``ids``; a VideoMAE file whose rows differ from ``DoTA_s1_ncc`` raises."""
    x: dict[str, np.ndarray] = {}
    y: dict[str, np.ndarray] = {}
    u: dict[str, dict[str, np.ndarray]] = {enc: {} for enc in video_dirs}
    for video_id in ids:
        s1 = np.load(s1_dir / f"{video_id}.npy").astype(np.float64)
        x[video_id] = s1[::stride]
        y[video_id] = np.asarray(
            resized_frame_labels(records[video_id], len(s1), stride), dtype=np.int64
        )
        if len(y[video_id]) != len(x[video_id]):
            raise ValueError(f"{video_id}: {len(x[video_id])} rows vs {len(y[video_id])} labels")
        for enc, cache in video_dirs.items():
            rows = np.load(cache / f"{video_id}.npy").astype(np.float64)
            if len(rows) != len(s1):
                raise ValueError(
                    f"{video_id}: {enc} has {len(rows)} rows, DoTA_s1_ncc {len(s1)} (L6)"
                )
            u[enc][video_id] = rows[::stride]
    return ProbeCorpus(x, u, y, {v: dota_group(v) for v in ids})


def load_t2_k(
    ids: list[str],
    annotation: Path,
    census: dict[str, int],
    clip_dir: Path,
    video_dirs: dict[str, Path],
) -> ProbeCorpus:
    """The K sources, whole, at s8 (K's convention); each source its own cluster."""
    y, _types = source_labels(annotation, census, ids, constants.FRAME_STRIDE)
    x = load_features(clip_dir, ids)
    u = {enc: load_features(cache, ids) for enc, cache in video_dirs.items()}
    bad = [
        v for v in ids
        if any(len(rows) != len(y[v]) for rows in (x[v], *(u[e][v] for e in u)))
    ]
    if bad:
        raise ValueError(f"{len(bad)} K sources misaligned (labels / CLIP / VideoMAE): {bad[:5]}")
    return ProbeCorpus(x, u, y, {v: v for v in ids})


# ---------------------------------------------------------------------------
# representations (N2) and probes (N3-N5)
# ---------------------------------------------------------------------------
def _cat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts, axis=1)


def arm_sets(corpus: ProbeCorpus, encoder: str, arm: str) -> dict[str, dict[str, np.ndarray]]:
    """Every probed representation of ``arm`` under ``encoder``, keyed by set name."""
    sets: dict[str, dict[str, np.ndarray]] = {name: {} for name in (*DECISION_SETS, *PRINTED_SETS)}
    for v, x in corpus.x.items():
        u = corpus.u[encoder][v]
        if arm == "A3":
            x = x - reference(x, constants.V2_E2D_CRN)
            u = u - reference(u, constants.V2_E2D_CRN)
        p = position_features(len(x))
        sets["base"][v] = x
        sets["with"][v] = _cat(x, u)
        sets["u"][v] = u
        sets["p"][v] = p
        sets["base_p"][v] = _cat(x, p)
        sets["with_p"][v] = _cat(x, u, p)
    return sets


def in_domain_aucs(
    features: dict[str, np.ndarray], corpus: ProbeCorpus, ids: list[str], folds: int, seed: int
) -> dict[str, float]:
    """Out-of-fold per-clip AUC, CV grouped by the corpus's cluster (N3)."""
    index = {g: i for i, g in enumerate(sorted({corpus.group[v] for v in ids}))}
    group_of = {v: index[corpus.group[v]] for v in ids}
    aucs = per_source_auc(features, corpus.y, group_of, ids, folds, seed)
    return dict(zip(ids, (float(a) for a in aucs), strict=True))


def transfer_aucs(
    train: dict[str, np.ndarray],
    train_y: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    test_y: dict[str, np.ndarray],
    ids: list[str],
    seed: int,
) -> dict[str, float]:
    """Probe fitted on every train frame, per-clip AUC on ``ids`` (N4)."""
    order = sorted(train)
    flat = transfer_scores(
        np.concatenate([train[v] for v in order]),
        np.concatenate([train_y[v] for v in order]),
        np.concatenate([test[v] for v in ids]),
        seed,
    )
    out: dict[str, float] = {}
    offset = 0
    for v in ids:
        n = len(test_y[v])
        out[v] = frame_auc(flat[offset : offset + n], test_y[v])
        offset += n
    return out


def ci(values: dict[str, float], groups: dict[str, str], resamples: int, seed: int) -> Any:
    return cluster_bootstrap_ci(values, groups, resamples, seed, constants.V2_E2D_CI)


def diff(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {v: a[v] - b[v] for v in a}


def probe_arm(
    dota: ProbeCorpus,
    t2: ProbeCorpus,
    encoder: str,
    arm: str,
    folds: int,
    seed: int,
    resamples: int,
    share: dict[str, float],
) -> dict[str, Any]:
    """Every set under both probes, the decision Δ and the printed Δ for one arm and encoder."""
    ids = dota.two_class()
    dota_sets = arm_sets(dota, encoder, arm)
    t2_sets = arm_sets(t2, encoder, arm)
    out: dict[str, Any] = {}
    for probe in PROBES:
        aucs: dict[str, dict[str, float]] = {}
        for name in (*DECISION_SETS, *PRINTED_SETS):
            if probe == "in_domain":
                aucs[name] = in_domain_aucs(dota_sets[name], dota, ids, folds, seed)
            else:
                aucs[name] = transfer_aucs(
                    t2_sets[name], t2.y, dota_sets[name], dota.y, ids, seed
                )
            LOGGER.info("E2(d) %s %s %s %s: macro %.4f", encoder, arm, probe, name,
                        float(np.mean(list(aucs[name].values()))))
        delta = diff(aucs["with"], aucs["base"])
        out[probe] = {
            "macro": {name: ci(a, dota.group, resamples, seed) for name, a in aucs.items()},
            "delta": ci(delta, dota.group, resamples, seed),
            "delta_beyond_position": ci(
                diff(aucs["with_p"], aucs["base_p"]), dota.group, resamples, seed
            ),
            "delta_by_share_bin": {
                label: ci(
                    {v: d for v, d in delta.items() if share_bin(share[v]) == label},
                    dota.group, resamples, seed,
                )
                for label in constants.V2_SHARE_BIN_LABELS
            },
        }
    return out


# ---------------------------------------------------------------------------
# rule (N6, N7)
# ---------------------------------------------------------------------------
def eligible_arms(arms: dict[str, dict[str, Any]]) -> list[str]:
    """Arms whose in-domain or transfer mean Δ reaches the proposal's threshold (N6)."""
    return [
        arm for arm, probes in arms.items()
        if probes["in_domain"]["delta"]["mean"] >= constants.V2_E2D_INDOMAIN_MIN
        or probes["transfer"]["delta"]["mean"] >= constants.V2_E2D_TRANSFER_MIN
    ]


def cost(encoder: str) -> int:
    """Embedding width as the cost proxy: S (384) is cheaper than B (768)."""
    return constants.VIDEOMAE_ARCH[encoder][0]


def decide(per_encoder: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """N6 eligibility per encoder, then N7's pick on the A3 transfer Δ."""
    eligible = {enc: eligible_arms(arms) for enc, arms in per_encoder.items()}
    candidates = {
        enc: per_encoder[enc][PICK_ARM]["transfer"]["delta"]["mean"]
        for enc, arms in eligible.items() if arms
    }
    if not candidates:
        return {"eligible": eligible, "pick": MOTION_DROPPED, "tie": False}
    best = max(candidates, key=lambda enc: candidates[enc])
    close = [enc for enc, d in candidates.items()
             if candidates[best] - d < constants.V2_E2D_TIE_MARGIN]
    pick = min(close, key=cost)
    return {
        "eligible": eligible,
        "a3_transfer_delta": candidates,
        "pick": pick,
        "tie": len(close) > 1,
    }


# ---------------------------------------------------------------------------
# printed extras: T2-side read (N9 d) and D15 (N10)
# ---------------------------------------------------------------------------
def t2_side(t2: ProbeCorpus, encoders: list[str], folds: int, seed: int, resamples: int) -> Any:
    """Source-grouped CV on the K sources: ``[x;u]`` - ``x`` per encoder (K's (i'))."""
    ids = t2.two_class()
    out = {}
    for enc in encoders:
        base = in_domain_aucs(t2.x, t2, ids, folds, seed)
        both = in_domain_aucs(
            {v: _cat(t2.x[v], t2.u[enc][v]) for v in ids}, t2, ids, folds, seed
        )
        out[enc] = {
            "x": ci(base, t2.group, resamples, seed),
            "xu": ci(both, t2.group, resamples, seed),
            "delta": ci(diff(both, base), t2.group, resamples, seed),
        }
    return out


def split_macro(
    values: dict[str, float], cap: set[str], groups: dict[str, str], resamples: int, seed: int
) -> dict[str, Any]:
    """Macro over all of ``values``, its DoTA-CAP part and the dropped part."""
    return {
        DEV: ci(values, groups, resamples, seed),
        CAP: ci({v: a for v, a in values.items() if v in cap}, groups, resamples, seed),
        DROPPED: ci({v: a for v, a in values.items() if v not in cap}, groups, resamples, seed),
    }


def representativeness(
    dev: ProbeCorpus,
    cap: set[str],
    records: dict[str, DotaRecord],
    lengths: dict[str, int],
    a0_clip_aucs: dict[str, float] | None,
    folds: int,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """D15 (N10): probe and A0 macro on dev vs its DoTA-CAP part; kept vs dropped composition."""
    ids = dev.two_class()
    probe = {
        "x": in_domain_aucs(dev.x, dev, ids, folds, seed),
        "x_crn": in_domain_aucs(
            {v: dev.x[v] - reference(dev.x[v], constants.V2_E2D_CRN) for v in ids},
            dev, ids, folds, seed,
        ),
    }
    out: dict[str, Any] = {
        "clip_only_probe": {
            name: split_macro(a, cap, dev.group, resamples, seed) for name, a in probe.items()
        },
    }
    if a0_clip_aucs is not None:
        out["a0_protocol_b"] = split_macro(
            {v: a for v, a in a0_clip_aucs.items() if v in dev.y}, cap, dev.group,
            resamples, seed,
        )
    composition: dict[str, Any] = {}
    for side, members in ((CAP, sorted(cap)), (DROPPED, sorted(set(dev.y) - cap))):
        shares = [records[v].span[1] - records[v].span[0] for v in members]
        composition[side] = {
            "clips": len(members),
            "native_frames_median": float(np.median([lengths[v] for v in members])),
            "accident_share_median": float(np.median(shares)),
            "clips_per_share_bin": dict(Counter(share_bin(s) for s in shares)),
            "category": dict(Counter(records[v].anomaly_class for v in members).most_common()),
        }
    out["composition"] = composition
    return out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    encoders = list(args.encoders)
    parsed = parse_metadata(args.metadata, read_split_ids(args.split_file))
    records = {r.video_id: r for r in parsed}
    dev_ids = load_split(constants.V2_SPLIT_DOTA_DEV, args.split_dir)
    cap_ids = load_split(constants.V2_SPLIT_DOTA_CAP_DEV, args.split_dir)
    k_ids = sorted(
        line.strip() for line in args.k_ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    k_sha1 = json.loads(args.k_manifest.read_text(encoding="utf-8"))["sha1"]
    if lines_sha1(k_ids) != k_sha1:
        raise ValueError(f"K sources sha1 {lines_sha1(k_ids)} != its manifest's {k_sha1} (N12)")

    alignment = dota_cap_alignment_sha256(args.split_dir)
    dota_dirs = {enc: dota_video_dir(args.video_root, enc) for enc in encoders}
    t2_dirs = {enc: t2_video_dir(args.video_root, enc) for enc in encoders}
    for enc in encoders:
        weights = constants.VIDEOMAE_WEIGHTS_SHA256[enc]
        check_manifest(dota_dirs[enc], dota_cap.video_manifest(enc, alignment))
        check_manifest(
            t2_dirs[enc],
            extract_video_features.build_manifest(enc, constants.FRAME_STRIDE, weights),
        )
    LOGGER.info("N12 gates passed: %d DoTA-CAP-dev clips, %d K sources, encoders %s",
                len(cap_ids), len(k_ids), encoders)

    stride = constants.V2_E2D_STRIDE
    dota = load_dota(cap_ids, records, args.dota_s1_dir, dota_dirs, stride)
    census = load_counts([str(p) for p in args.census])
    t2 = load_t2_k(k_ids, args.annotation, census, args.t2_clip_dir, t2_dirs)
    share = {v: records[v].span[1] - records[v].span[0] for v in cap_ids}

    per_encoder = {
        enc: {
            arm: probe_arm(dota, t2, enc, arm, args.folds, args.seed, args.resamples, share)
            for arm in ARMS
        }
        for enc in encoders
    }
    dev = load_dota(dev_ids, records, args.dota_s1_dir, {}, stride)
    lengths = {v: len(np.load(args.dota_s1_dir / f"{v}.npy", mmap_mode="r")) for v in dev_ids}
    a0 = (
        json.loads(args.a0_clip_aucs.read_text(encoding="utf-8"))
        if args.a0_clip_aucs is not None else None
    )
    readout = {
        "addendum": "core/docs/v2/PREREG_ADDENDUM.md §12 (N1-N12)",
        "stride": stride,
        "crn": constants.V2_E2D_CRN,
        "encoders": encoders,
        "dota_cap_dev": {"clips": len(cap_ids), "two_class": len(dota.two_class())},
        "k_sources": {"sources": len(k_ids), "sha1": k_sha1},
        "folds": args.folds,
        "seed": args.seed,
        "bootstrap": args.resamples,
        "rule": {
            "in_domain_min": constants.V2_E2D_INDOMAIN_MIN,
            "transfer_min": constants.V2_E2D_TRANSFER_MIN,
            "tie_margin": constants.V2_E2D_TIE_MARGIN,
            "pick_arm": PICK_ARM,
        },
        "per_encoder": per_encoder,
        "decision": decide(per_encoder),
        "printed": {
            "t2_side": t2_side(t2, encoders, args.folds, args.seed, args.resamples),
            "d15": representativeness(
                dev, set(cap_ids), records, lengths, a0, args.folds, args.seed, args.resamples
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(readout))
    LOGGER.info("E2(d) decision: %s -> %s", readout["decision"], args.out_dir)
    return readout


# ---------------------------------------------------------------------------
# read-out
# ---------------------------------------------------------------------------
def _fmt(c: dict[str, float] | None) -> str:
    if c is None:
        return "—"
    return f"{c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_markdown(readout: dict[str, Any]) -> str:
    """Human read-out; the JSON is the record."""
    decision = readout["decision"]
    lines = [
        "# v2 E2(d): encoder choice on DoTA-CAP-dev",
        "",
        f"Addendum §12. `dota_cap_dev` {readout['dota_cap_dev']['clips']} clips "
        f"({readout['dota_cap_dev']['two_class']} two-class), protocol B (stride "
        f"{readout['stride']}); transfer fitted on {readout['k_sources']['sources']} K sources "
        f"(sha1 `{readout['k_sources']['sha1'][:12]}`). Cluster bootstrap over source videos, "
        f"B = {readout['bootstrap']}. Name it **DoTA-CAP (n/1397)** (D14).",
        "",
        _row(["Encoder", "Arm", "Probe", "CLIP-only", "with u", "**Δ** (decides)",
              "Δ beyond position (printed)"]),
        "|---|---|---|---|---|---|---|",
    ]
    for enc, arms in readout["per_encoder"].items():
        for arm, probes in arms.items():
            for probe, block in probes.items():
                lines.append(_row([
                    enc, arm, probe, _fmt(block["macro"]["base"]), _fmt(block["macro"]["with"]),
                    _fmt(block["delta"]), _fmt(block["delta_beyond_position"]),
                ]))
    rule = readout["rule"]
    lines += [
        "",
        f"Rule (N6/N7): eligible iff in-domain Δ ≥ {rule['in_domain_min']} or transfer Δ ≥ "
        f"{rule['transfer_min']} (point estimate) for A2 or A3; pick = largest {rule['pick_arm']} "
        f"transfer Δ, within {rule['tie_margin']} the cheaper encoder.",
        "",
        f"Eligible arms per encoder: {decision['eligible']}",
        "",
        f"**Decision: {decision['pick']}**"
        + (" (tie → cheaper encoder)" if decision["tie"] else ""),
        "",
        "## Printed, never decided on",
        "",
        "`u` alone and position `p` alone (macro):",
        "",
        _row(["Encoder", "Arm", "Probe", "u", "p (position only)"]),
        "|---|---|---|---|---|",
    ]
    for enc, arms in readout["per_encoder"].items():
        for arm, probes in arms.items():
            for probe, block in probes.items():
                lines.append(_row([enc, arm, probe, _fmt(block["macro"]["u"]),
                                   _fmt(block["macro"]["p"])]))
    lines += ["", "Δ per share bin:", "",
              _row(["Encoder", "Arm", "Probe", *constants.V2_SHARE_BIN_LABELS]),
              "|---|---|---|" + "---|" * len(constants.V2_SHARE_BIN_LABELS)]
    for enc, arms in readout["per_encoder"].items():
        for arm, probes in arms.items():
            for probe, block in probes.items():
                bins = block["delta_by_share_bin"]
                lines.append(_row([enc, arm, probe,
                                   *(_fmt(bins[b]) for b in constants.V2_SHARE_BIN_LABELS)]))
    lines += ["", "T2 side (K sources, source-grouped CV):", "",
              _row(["Encoder", "x", "[x;u]", "Δ"]), "|---|---|---|---|"]
    for enc, block in readout["printed"]["t2_side"].items():
        lines.append(_row([enc, _fmt(block["x"]), _fmt(block["xu"]), _fmt(block["delta"])]))
    d15 = readout["printed"]["d15"]
    lines += ["", "D15 representativeness (all `dota_dev` vs its DoTA-CAP part vs dropped):", "",
              _row(["Read", DEV, CAP, DROPPED]), "|---|---|---|---|"]
    for name, block in d15["clip_only_probe"].items():
        lines.append(_row([f"CLIP-only probe `{name}`",
                           *(_fmt(block[k]) for k in (DEV, CAP, DROPPED))]))
    if "a0_protocol_b" in d15:
        block = d15["a0_protocol_b"]
        lines.append(_row(["A0 (E1 ckpts, protocol B)",
                           *(_fmt(block[k]) for k in (DEV, CAP, DROPPED))]))
    lines += ["", _row(["Side", "clips", "native frames q50", "share q50", "top categories"]),
              "|---|---|---|---|---|"]
    for side, comp in d15["composition"].items():
        top = ", ".join(f"{k} {v}" for k, v in list(comp["category"].items())[:4])
        lines.append(_row([side, str(comp["clips"]), f"{comp['native_frames_median']:.0f}",
                           f"{comp['accident_share_median']:.3f}", top]))
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--dota-s1-dir", type=Path, required=True, help="cache/clip/DoTA_s1_ncc")
    parser.add_argument("--metadata", type=Path, required=True, help="DoTA metadata_val.json")
    parser.add_argument("--split-file", type=Path, required=True, help="DoTA val_split.txt")
    parser.add_argument("--k-ids-file", type=Path, required=True, help="v2_K/k_sources.txt")
    parser.add_argument("--k-manifest", type=Path, required=True,
                        help="v2_K/k_sources_manifest.json")
    parser.add_argument("--annotation", type=Path, required=True, help="dada标注.xlsx")
    parser.add_argument("--census", type=Path, nargs="+", required=True,
                        help="T2 census JSON(s) (frame counts per source)")
    parser.add_argument("--t2-clip-dir", type=Path, required=True,
                        help="cache/clip/DADA2000_orig (s8)")
    parser.add_argument("--video-root", type=Path, default=constants.VIDEO_CACHE_DIR)
    parser.add_argument("--encoders", nargs="+",
                        default=[constants.VIDEOMAE_ENCODER_B, constants.VIDEOMAE_ENCODER_S],
                        choices=sorted(constants.VIDEOMAE_ARCH))
    parser.add_argument("--a0-clip-aucs", type=Path, default=None,
                        help="per-clip A0 AUCs on dota_dev (rate_matched_eval score), D15")
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--folds", type=int, default=constants.EDA_PROBE_FOLDS)
    parser.add_argument("--seed", type=int, default=constants.V2_SPLIT_SEED)
    parser.add_argument("--resamples", type=int, default=constants.V2_E2D_BOOTSTRAP)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
