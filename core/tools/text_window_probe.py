"""Probe: does text-guided multi-scale CLIP window pooling add to A3's inputs? (pending (ba)).

**Exploratory, not an amendment.** Its rule is fixed here, in code, before its first number;
it decides only whether the idea earns a pre-registered build, never anything about v2.

The idea (AnyAnomaly's WinCLIP-based attention + LaGoVAD's definition set ``Z``): pool the CLIP
embeddings of the multi-scale windows of :mod:`core.tools.clip_windows` with weights given by how
much each window looks like ``Z``'s anomaly text rather than its normal text, so a small or distant
collision is not averaged away by the whole frame.

**Text.** ``z_abn`` = mean of the unit CLIP text embeddings of the ``CarAccident`` definitions of
DoTA/DADA's ``Z`` (``DATASET_CLS_DEFS["dota"]``), ``z_norm`` = the same for ``Normal``; frozen CLIP
text tower, pinned revision, no soft prompt. Window margin ``s_w = cos(e_w, z_abn) - cos(e_w,
z_norm)``.

**Pooled streams** (whole-frame cell excluded; each grid's cells pooled, then the grids averaged
-- AnyAnomaly's average over scales):

* ``text``  -- ``softmax(s_w / TW_TEMPERATURE)`` within each grid (the idea);
* ``mean``  -- uniform weights (multi-scale only: the control for "text helps");
* ``prior`` -- one fixed per-cell weight map = the mean ``text`` weights over every K-source frame
  (T2 only, no DoTA pixel or label): the control for "it is only *where* in the frame";
* ``score`` -- printed only: per grid, the max margin (WinCLIP's zero-shot evidence, 3 numbers).

**Probes** (E2(d)'s, N3/N4): every stream in the A3 form (per-clip R2 median removed), position
``p`` appended, base = ``[x; u; p]`` with ``u`` = V2-S. Transfer = logistic probe fitted on the
295 frozen K sources (T2-train, whole, stride 8), scored on ``dota_cap_dev`` at protocol B;
in-domain = 5-fold CV grouped by source video (printed). Δ = per-clip paired AUC, cluster bootstrap
over source videos. Every Δ is also split by DoTA's ego / non-ego classes.

**Rule (fixed before the first number):** ``GO`` iff the transfer Δ(``+text`` - base) over all
two-class clips has mean ≥ ``TW_GAIN_MIN`` (+0.01) **and** a CI lower bound > 0; else ``KILL``.
Printed sentences (never move the verdict): "text guidance adds beyond multi-scale pooling" iff
Δ(``+text`` - ``+mean``) has CI low > 0; "text guidance adds beyond the T2 spatial prior" iff
Δ(``+text`` - ``+prior``) has CI low > 0. Also printed: the zero-shot macro of the max margin, and
the normalized entropy of the text weights (1 = flat: the text map then pools like ``mean``).

DoTA-eval is sealed: only ``dota_cap_dev`` is loaded.

CLI::

    python -m core.tools.text_window_probe --dota-s1-dir cache/clip/DoTA_s1_ncc \\
        --metadata data/DoTA/metadata_val.json --split-file data/DoTA/val_split.txt \\
        --k-ids-file .../v2_K/k_sources.txt --k-manifest .../v2_K/k_sources_manifest.json \\
        --annotation .../dada标注.xlsx --census .../resolved_census.json \\
        --t2-clip-dir cache/clip/DADA2000_orig --video-root cache/video \\
        --window-root cache/clip_windows --out-dir outputs/v2/REPORTS/tw_probe
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.crn.reference import reference
from core.data.dada_origin import load_counts
from core.data.definitions import DATASET_CLS_DEFS
from core.data.dota import parse_metadata, read_split_ids
from core.data.v2_splits import dota_group, lines_sha1, load_split
from core.metrics import frame_auc
from core.tools import clip_windows, dota_cap, extract_video_features
from core.tools.e2d_probe import (
    ProbeCorpus,
    _fmt,
    _row,
    check_manifest,
    ci,
    diff,
    dota_cap_alignment_sha256,
    dota_video_dir,
    in_domain_aucs,
    load_dota,
    load_t2_k,
    t2_video_dir,
    transfer_aucs,
)
from core.tools.kill_switch_probe import position_features, write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

READOUT_JSON = "tw_readout.json"
READOUT_MD = "tw_readout.md"
TEXT_NPZ = "tw_text.npz"
ABNORMAL_CLASS = "CarAccident"
NORMAL_CLASS = "Normal"
DEFS_KEY = "dota"
ENCODER = constants.VIDEOMAE_ENCODER_S  # V2-S, A3's stream (E2(d) pick)
BASE = "base"
POOLS = ("mean", "text", "prior")
SETS = (BASE, *POOLS, "score")
CONTRASTS = {  # name: (with, without); the first one decides
    "text - base": ("text", BASE),
    "mean - base": ("mean", BASE),
    "prior - base": ("prior", BASE),
    "text - mean": ("text", "mean"),
    "text - prior": ("text", "prior"),
    "score - base": ("score", BASE),
}
DECISION = "text - base"
SUBSETS = ("all", "ego", "non-ego")
GO = "GO"
KILL = "KILL"
SENTENCES = {
    "text - mean": "text guidance adds beyond multi-scale pooling",
    "text - prior": "text guidance adds beyond the T2 spatial prior",
}


# ---------------------------------------------------------------------------
# text anchors
# ---------------------------------------------------------------------------
def definition_texts() -> dict[str, list[str]]:
    defs = DATASET_CLS_DEFS[DEFS_KEY]
    return {ABNORMAL_CLASS: list(defs[ABNORMAL_CLASS]), NORMAL_CLASS: list(defs[NORMAL_CLASS])}


def encode_texts(texts: list[str]) -> np.ndarray:
    """Frozen CLIP ViT-B/16 text tower (pinned revision, C4), ``(len(texts), 512)`` float32."""
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        constants.CLIP_MODEL_NAME, revision=constants.CLIP_MODEL_REVISION
    )
    model = CLIPTextModelWithProjection.from_pretrained(
        constants.CLIP_MODEL_NAME, revision=constants.CLIP_MODEL_REVISION, use_safetensors=True
    ).eval()
    with torch.no_grad():
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return model(**tokens).text_embeds.float().numpy()  # type: ignore[no-any-return]


def unit(rows: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(rows, axis=-1, keepdims=True)
    out: np.ndarray = rows / np.maximum(norm, np.finfo(np.float32).tiny)
    return out


def class_anchor(embeddings: np.ndarray) -> np.ndarray:
    """Prompt-ensemble anchor: the unit mean of unit embeddings."""
    out: np.ndarray = unit(unit(embeddings).mean(axis=0))
    return out


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------
def local_slices(grids: tuple[int, ...] = constants.TW_GRIDS) -> list[slice]:
    """Index range of each non-global grid inside a frame's windows."""
    out, offset = [], 0
    for g in grids:
        if g != constants.TW_GLOBAL_GRID:
            out.append(slice(offset, offset + g * g))
        offset += g * g
    return out


def margins(windows: np.ndarray, z_abn: np.ndarray, z_norm: np.ndarray) -> np.ndarray:
    """``(L, W)`` cos(window, z_abn) - cos(window, z_norm)."""
    w = unit(windows)
    out: np.ndarray = w @ z_abn - w @ z_norm
    return out


def text_weights(
    margin: np.ndarray, slices: list[slice], tau: float = constants.TW_TEMPERATURE
) -> np.ndarray:
    """Softmax of ``margin / tau`` within each grid; zero on the global cell."""
    weights = np.zeros_like(margin)
    for s in slices:
        logits = margin[:, s] / tau
        logits = logits - logits.max(axis=1, keepdims=True)
        expd = np.exp(logits)
        weights[:, s] = expd / expd.sum(axis=1, keepdims=True)
    return weights


def uniform_weights(rows: int, width: int, slices: list[slice]) -> np.ndarray:
    weights = np.zeros((rows, width))
    for s in slices:
        weights[:, s] = 1.0 / (s.stop - s.start)
    return weights


def pool(windows: np.ndarray, weights: np.ndarray, slices: list[slice]) -> np.ndarray:
    """Weighted cell sum per grid, then the mean over grids: ``(L, D)``."""
    per_grid = [np.einsum("lw,lwd->ld", weights[:, s], windows[:, s]) for s in slices]
    out: np.ndarray = np.mean(per_grid, axis=0)
    return out


def max_margin(margin: np.ndarray, slices: list[slice]) -> np.ndarray:
    """``(L, n_grids)``: the largest margin of each grid (WinCLIP-style evidence)."""
    return np.stack([margin[:, s].max(axis=1) for s in slices], axis=1)


def weight_entropy(weights: np.ndarray, slices: list[slice]) -> float:
    """Mean normalized entropy of the text weights over frames and grids (1 = uniform)."""
    values = []
    for s in slices:
        p = np.clip(weights[:, s], np.finfo(np.float64).tiny, None)
        values.append((-(p * np.log(p)).sum(axis=1) / np.log(s.stop - s.start)).mean())
    return float(np.mean(values))


def window_streams(
    windows: dict[str, np.ndarray],
    z_abn: np.ndarray,
    z_norm: np.ndarray,
    prior: np.ndarray | None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Per clip: the pooled streams and the text weights (``prior`` None -> no prior stream)."""
    slices = local_slices()
    streams: dict[str, dict[str, np.ndarray]] = {name: {} for name in (*POOLS, "score")}
    weights_of: dict[str, np.ndarray] = {}
    for v, win in windows.items():
        margin = margins(win, z_abn, z_norm)
        weights = text_weights(margin, slices)
        weights_of[v] = weights
        streams["text"][v] = pool(win, weights, slices)
        streams["mean"][v] = pool(win, uniform_weights(len(win), win.shape[1], slices), slices)
        streams["score"][v] = max_margin(margin, slices)
        if prior is not None:
            streams["prior"][v] = pool(win, np.broadcast_to(prior, weights.shape), slices)
    return streams, weights_of


def spatial_prior(weights_of: dict[str, np.ndarray]) -> np.ndarray:
    """Mean text weight per cell over every frame given (T2 K sources only)."""
    out: np.ndarray = np.concatenate(list(weights_of.values())).mean(axis=0)
    return out


def probe_sets(
    corpus: ProbeCorpus, streams: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, np.ndarray]]:
    """A3 form: every stream minus its per-clip R2 reference; position appended."""

    def crn(rows: np.ndarray) -> np.ndarray:
        out: np.ndarray = rows - reference(rows, constants.TW_CRN)
        return out

    sets: dict[str, dict[str, np.ndarray]] = {name: {} for name in SETS}
    for v, x in corpus.x.items():
        base = [crn(x), crn(corpus.u[ENCODER][v])]
        p = position_features(len(x))
        sets[BASE][v] = np.concatenate([*base, p], axis=1)
        for name in (*POOLS, "score"):
            sets[name][v] = np.concatenate([*base, crn(streams[name][v]), p], axis=1)
    return sets


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
def subset_ids(ids: list[str], ego: dict[str, bool]) -> dict[str, list[str]]:
    return {
        "all": ids,
        "ego": [v for v in ids if ego[v]],
        "non-ego": [v for v in ids if not ego[v]],
    }


def summarize(
    aucs: dict[str, dict[str, float]],
    subsets: dict[str, list[str]],
    groups: dict[str, str],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Macro per set and every contrast, per subset."""
    out: dict[str, Any] = {"macro": {}, "delta": {}}
    for sub, ids in subsets.items():
        out["macro"][sub] = {
            name: ci({v: a[v] for v in ids}, groups, resamples, seed) for name, a in aucs.items()
        }
        out["delta"][sub] = {
            name: ci({v: d for v, d in diff(aucs[w], aucs[b]).items() if v in set(ids)},
                     groups, resamples, seed)
            for name, (w, b) in CONTRASTS.items()
        }
    return out


def verdict(transfer: dict[str, Any]) -> dict[str, Any]:
    decision = transfer["delta"]["all"][DECISION]
    go = decision["mean"] >= constants.TW_GAIN_MIN and decision["low"] > 0.0
    sentences = {
        name: text if transfer["delta"]["all"][name]["low"] > 0.0 else f"NOT: {text}"
        for name, text in SENTENCES.items()
    }
    return {"verdict": GO if go else KILL, "decision_delta": decision, "sentences": sentences}


def zero_shot_macro(
    scores: dict[str, np.ndarray], corpus: ProbeCorpus, ids: list[str], resamples: int, seed: int
) -> Any:
    """Macro AUC of the max margin over all local windows, no probe."""
    aucs = {v: frame_auc(scores[v].max(axis=1), corpus.y[v]) for v in ids}
    return ci(aucs, corpus.group, resamples, seed)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def load_windows(window_dir: Path, ids: list[str], rows: dict[str, int]) -> dict[str, np.ndarray]:
    """``{id: (L, W, D)}``; a missing file or a row count off the CLIP rows raises."""
    out: dict[str, np.ndarray] = {}
    bad: list[str] = []
    for v in ids:
        path = window_dir / f"{v}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{v}: no window features in {window_dir}")
        out[v] = clip_windows.unflatten(np.load(path))
        if len(out[v]) != rows[v]:
            bad.append(v)
    if bad:
        raise ValueError(f"{len(bad)} clips' window rows != their CLIP rows: {bad[:5]}")
    return out


def text_anchors(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    texts = definition_texts()
    if args.text_npz is not None:
        stored = np.load(args.text_npz)
        abn, norm = stored["abnormal"], stored["normal"]
    else:
        abn, norm = encode_texts(texts[ABNORMAL_CLASS]), encode_texts(texts[NORMAL_CLASS])
        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(args.out_dir / TEXT_NPZ, abnormal=abn, normal=norm)
    z_abn, z_norm = class_anchor(abn), class_anchor(norm)
    return z_abn, z_norm, {"texts": texts, "anchor_cos": float(z_abn @ z_norm)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    parsed = parse_metadata(args.metadata, read_split_ids(args.split_file))
    records = {r.video_id: r for r in parsed}
    cap_ids = load_split(constants.V2_SPLIT_DOTA_CAP_DEV, args.split_dir)
    k_ids = sorted(
        line.strip() for line in args.k_ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    k_sha1 = json.loads(args.k_manifest.read_text(encoding="utf-8"))["sha1"]
    if lines_sha1(k_ids) != k_sha1:
        raise ValueError(f"K sources sha1 {lines_sha1(k_ids)} != its manifest's {k_sha1}")

    alignment = dota_cap_alignment_sha256(args.split_dir)
    dota_u, t2_u = dota_video_dir(args.video_root, ENCODER), t2_video_dir(args.video_root, ENCODER)
    check_manifest(dota_u, dota_cap.video_manifest(ENCODER, alignment))
    check_manifest(t2_u, extract_video_features.build_manifest(
        ENCODER, constants.FRAME_STRIDE, constants.VIDEOMAE_WEIGHTS_SHA256[ENCODER]))
    t2_win = args.window_root / clip_windows.cache_dir(
        constants.DADA_ORIGIN_DATASET, constants.FRAME_STRIDE).name
    dota_win = args.window_root / clip_windows.cache_dir(
        constants.DOTA_CAP_DATASET, constants.TW_DOTA_STRIDE).name
    for d in (t2_win, dota_win):
        found = json.loads((d / constants.VIDEO_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        if found.get("grids") != list(constants.TW_GRIDS):
            raise ValueError(f"{d} was built with grids {found.get('grids')}")

    dota = load_dota(
        cap_ids, records, args.dota_s1_dir, {ENCODER: dota_u}, constants.TW_DOTA_STRIDE
    )
    census = load_counts([str(p) for p in args.census])
    t2 = load_t2_k(k_ids, args.annotation, census, args.t2_clip_dir, {ENCODER: t2_u})
    ids = dota.two_class()
    LOGGER.info("gates passed: %d DoTA-CAP-dev clips (%d two-class), %d K sources",
                len(cap_ids), len(ids), len(k_ids))

    z_abn, z_norm, text_info = text_anchors(args)
    t2_windows = load_windows(t2_win, k_ids, {v: len(t2.x[v]) for v in k_ids})
    dota_windows = load_windows(dota_win, cap_ids, {v: len(dota.x[v]) for v in cap_ids})
    t2_streams, t2_weights = window_streams(t2_windows, z_abn, z_norm, None)
    prior = spatial_prior(t2_weights)
    t2_streams, _ = window_streams(t2_windows, z_abn, z_norm, prior)
    dota_streams, dota_weights = window_streams(dota_windows, z_abn, z_norm, prior)

    t2_sets, dota_sets = probe_sets(t2, t2_streams), probe_sets(dota, dota_streams)
    transfer = {
        name: transfer_aucs(t2_sets[name], t2.y, dota_sets[name], dota.y, ids, args.seed)
        for name in SETS
    }
    in_domain = {
        name: in_domain_aucs(dota_sets[name], dota, ids, args.folds, args.seed) for name in SETS
    }
    for name in SETS:
        LOGGER.info("%s: transfer %.4f, in-domain %.4f", name,
                    np.mean(list(transfer[name].values())), np.mean(list(in_domain[name].values())))
    subsets = subset_ids(ids, {v: records[v].is_ego for v in ids})
    groups = {v: dota_group(v) for v in ids}
    slices = local_slices()
    transfer_block = summarize(transfer, subsets, groups, args.resamples, args.seed)
    readout = {
        "status": "exploratory probe (pending (ba)); rule fixed in code before the first number",
        "grids": list(constants.TW_GRIDS),
        "temperature": constants.TW_TEMPERATURE,
        "encoder_u": ENCODER,
        "crn": constants.TW_CRN,
        "stride": constants.TW_DOTA_STRIDE,
        "dota_cap_dev": {"clips": len(cap_ids), "two_class": len(ids),
                         **{k: len(v) for k, v in subsets.items()}},
        "k_sources": {"sources": len(k_ids), "sha1": k_sha1},
        "folds": args.folds, "seed": args.seed, "bootstrap": args.resamples,
        "rule": {"gain_min": constants.TW_GAIN_MIN, "decision": DECISION},
        "text": text_info,
        "transfer": transfer_block,
        "in_domain": summarize(in_domain, subsets, groups, args.resamples, args.seed),
        "printed": {
            "zero_shot_max_margin": zero_shot_macro(
                dota_streams["score"], dota, ids, args.resamples, args.seed),
            "text_weight_entropy": {
                "t2": weight_entropy(np.concatenate(list(t2_weights.values())), slices),
                "dota_cap_dev": weight_entropy(np.concatenate(list(dota_weights.values())), slices),
            },
            "spatial_prior": [prior[s].round(4).tolist() for s in slices],
            "position_only": ci(
                transfer_aucs({v: position_features(len(t2.x[v])) for v in k_ids}, t2.y,
                              {v: position_features(len(dota.x[v])) for v in ids}, dota.y,
                              ids, args.seed),
                groups, args.resamples, args.seed),
        },
        "decision": verdict(transfer_block),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / READOUT_JSON, readout)
    write_text_atomic(args.out_dir / READOUT_MD, render_markdown(readout))
    LOGGER.info("text-window probe: %s -> %s", readout["decision"]["verdict"], args.out_dir)
    return readout


def render_markdown(readout: dict[str, Any]) -> str:
    """Human read-out; the JSON is the record."""
    decision = readout["decision"]
    n = readout["dota_cap_dev"]
    lines = [
        "# Text-guided multi-scale CLIP window probe (exploratory, pending (ba))",
        "",
        f"Grids {readout['grids']} (1x1 = row gate only), softmax temperature "
        f"{readout['temperature']}, every stream R2-centred per clip, position `p` appended, "
        f"`u` = {readout['encoder_u']}. Transfer: fitted on {readout['k_sources']['sources']} K "
        f"sources, scored on `dota_cap_dev` ({n['two_class']} two-class clips: {n['ego']} ego, "
        f"{n['non-ego']} non-ego), protocol B. Cluster bootstrap over source videos, "
        f"B = {readout['bootstrap']}. Name it **DoTA-CAP (n/1397)**.",
        "",
        f"**Verdict: {decision['verdict']}** — rule: transfer Δ(text - base) mean ≥ "
        f"{readout['rule']['gain_min']} and CI low > 0; Δ = {_fmt(decision['decision_delta'])}.",
        "",
        *(f"* {s}" for s in decision["sentences"].values()),
        "",
    ]
    for probe in ("transfer", "in_domain"):
        block = readout[probe]
        lines += [f"## {probe} — macro", "", _row(["set", *SUBSETS]),
                  "|---|" + "---|" * len(SUBSETS)]
        for name in SETS:
            lines.append(_row([name, *(_fmt(block["macro"][s][name]) for s in SUBSETS)]))
        lines += ["", f"## {probe} — Δ", "", _row(["contrast", *SUBSETS]),
                  "|---|" + "---|" * len(SUBSETS)]
        for name in CONTRASTS:
            label = f"**{name}**" if name == DECISION and probe == "transfer" else name
            lines.append(_row([label, *(_fmt(block["delta"][s][name]) for s in SUBSETS)]))
        lines.append("")
    printed = readout["printed"]
    entropy = printed["text_weight_entropy"]
    lines += [
        "## Printed, never decided on",
        "",
        f"* Position only (cubic `p`, transfer): {_fmt(printed['position_only'])}",
        f"* Zero-shot max window margin (no probe): {_fmt(printed['zero_shot_max_margin'])}",
        f"* Text-weight normalized entropy (1 = flat): T2 {entropy['t2']:.4f}, "
        f"DoTA-CAP-dev {entropy['dota_cap_dev']:.4f}",
        f"* cos(z_abn, z_norm) = {readout['text']['anchor_cos']:.4f}",
        f"* T2 spatial prior per grid (row-major): {printed['spatial_prior']}",
    ]
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
    parser.add_argument("--census", type=Path, nargs="+", required=True)
    parser.add_argument("--t2-clip-dir", type=Path, required=True,
                        help="cache/clip/DADA2000_orig (s8)")
    parser.add_argument("--video-root", type=Path, default=constants.VIDEO_CACHE_DIR)
    parser.add_argument("--window-root", type=Path, default=constants.TW_CACHE_DIR)
    parser.add_argument("--text-npz", type=Path, default=None,
                        help="pre-encoded text (abnormal, normal); default: encode with CLIP")
    parser.add_argument("--split-dir", type=Path, default=constants.V2_SPLITS_DIR)
    parser.add_argument("--folds", type=int, default=constants.EDA_PROBE_FOLDS)
    parser.add_argument("--seed", type=int, default=constants.V2_SPLIT_SEED)
    parser.add_argument("--resamples", type=int, default=constants.TW_BOOTSTRAP)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
