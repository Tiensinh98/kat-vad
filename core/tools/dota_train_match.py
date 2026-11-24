"""Locate DoTA's **train** clips inside CAP-DATA (``KIM_RUN.md`` §52), without any pixel.

MM-AU Phase 0 matched only DoTA's 1,397 val clips, by CLIP content. The train clips have no
surviving frame or feature to compare, but the 1,248 exact val pairs show how CAP was built:
inside three CAP id ranges the DoTA clips appear in **file-name order as Windows sorts it**, at
10 fps with exactly DoTA's frame count, or at 30 fps with ``3n - 2`` / ``3n - 3`` frames. So the
whole DoTA name list (train + val) is aligned to CAP's id list by a monotone matching per range,
anchored on the exact val pairs.

``match`` (CPU, local, then commit the map)
    Aligns, reports the precision on held-out val pairs (each fold's anchors are the other
    fold), and freezes ``DOTA_TRAIN_CAP.json``. A CAP video that matched any val clip (exact or
    near) is never given to a train clip. Night flag and span end are not used by the alignment
    and are printed as an independent check.
``expand``
    The frozen map -> an alignment json in ``dota_cap_frames``' format (frame maps included), so
    ``dota_cap_frames pack`` streams the train clips unchanged.
``finalize``
    Drops the clips whose streamed CAP video did not have the annotated frame count; every other
    missing clip raises.

CLI::

    python -m core.tools.dota_train_match match --cap-csv data/MMAU/cap_annotations_min.csv \\
        --annotations data/DoTA/raw_annotations/DoTA_annotations.zip \\
        --val-split data/DoTA/val_split.txt \\
        --matches outputs/v2/REPORTS/mmau_p0/all/mmau_p0_matches.json
    python -m core.tools.dota_train_match expand --out .../dota_train_alignment.json
    python -m core.tools.dota_train_match finalize --alignment .../dota_train_alignment.json \\
        --archive-dir .../DoTA_CAP_train_frames --reports .../pack_*.json --out .../final.json

``--cap-csv`` holds CAP's ``cap_text_annotations.xls`` columns ``video,type,light,start,end,total``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import constants
from core.data.v2_splits import dota_group, lines_sha1
from core.tools.dota_cap import OK, file_sha256
from core.tools.dota_cap_frames import CAP_FRAMES_MISMATCH, archive_frames, archive_path
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

EXACT, NEAR = "exact", "near"
SORT_LOWEST = "\x01"  # Windows sorts "_" before every digit and letter
END_TOLERANCE_FRAMES = 15  # printed independent check only


@dataclass(frozen=True)
class CapVideo:
    cap_id: int
    cap_type: int
    light: int
    start: int
    end: int
    total: int


@dataclass(frozen=True)
class DotaClip:
    dota_id: str
    frames: int
    start: int
    end: int
    night: bool


def windows_key(name: str) -> tuple[str, str]:
    """Windows Explorer's order of DoTA ids: case-blind, ``-`` ignored, ``_`` before digits."""
    return name.lower().replace("-", "").replace("_", SORT_LOWEST), name


def read_cap_csv(path: Path) -> list[CapVideo]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [
            CapVideo(int(r["video"]), int(r["type"]), int(r["light"]), int(r["start"]),
                     int(r["end"]), int(r["total"]))
            for r in csv.DictReader(fh)
        ]
    if len(rows) != constants.MMAU_CAP_VIDEOS or len({r.cap_id for r in rows}) != len(rows):
        raise ValueError(f"{path}: {len(rows)} rows, expected {constants.MMAU_CAP_VIDEOS} unique")
    return sorted(rows, key=lambda r: r.cap_id)


def read_dota(annotations: Path) -> dict[str, DotaClip]:
    clips: dict[str, DotaClip] = {}
    with zipfile.ZipFile(annotations) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            d = json.loads(zf.read(name))
            clips[d["video_name"]] = DotaClip(
                d["video_name"], int(d["num_frames"]), int(d["anomaly_start"]),
                int(d["anomaly_end"]), bool(d["night"]),
            )
    return clips


def read_val_pairs(matches: Path) -> tuple[dict[str, int], dict[str, set[int]]]:
    """``(exact val pairs, every CAP id a val clip hit)`` of the MM-AU P0 read."""
    raw = json.loads(matches.read_text(encoding="utf-8"))["dota"]
    entries = list(raw.values()) if isinstance(raw, dict) else raw
    exact = {m["query"]: int(m["cap_id"]) for m in entries if m["grade"] == EXACT}
    hits = {
        m["query"]: {int(m["cap_id"]), *(int(h) for h in m["hits"])}
        for m in entries if m["grade"] in (EXACT, NEAR)
    }
    return exact, hits


def monotone_match(weight: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-weight strictly increasing matching of rows to columns (0 = not allowed)."""
    rows, cols = weight.shape
    best = np.zeros((rows + 1, cols + 1), dtype=np.float32)
    for i in range(rows):
        take = np.where(weight[i] > 0, best[i, :-1] + weight[i], 0.0)
        best[i + 1, 1:] = np.maximum.accumulate(np.maximum(best[i, 1:], take))
    pairs: list[tuple[int, int]] = []
    i, j = rows, cols
    while i > 0 and j > 0:
        if best[i, j] == best[i, j - 1]:
            j -= 1
        elif best[i, j] == best[i - 1, j]:
            i -= 1
        else:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
    return pairs[::-1]


def block_weights(
    cap: list[CapVideo], dota: list[DotaClip], rate: int, anchors: dict[str, int]
) -> np.ndarray:
    """Allowed pairs (frame count) weighted by how close the two annotated starts are."""
    totals = np.array([c.total for c in cap])[:, None]
    frames = np.array([d.frames for d in dota])[None, :]
    allowed = np.isin(totals - rate * frames, constants.DOTA_TRAIN_LENGTH_DIFF[rate])
    starts = np.array([c.start for c in cap], dtype=np.float64)[:, None] / rate
    gap = np.abs(starts - np.array([d.start for d in dota], dtype=np.float64)[None, :])
    weight = np.where(allowed, 1.0 + np.exp(-gap / constants.DOTA_TRAIN_START_SCALE), 0.0)
    row_of = {c.cap_id: k for k, c in enumerate(cap)}
    col_of = {d.dota_id: k for k, d in enumerate(dota)}
    for dota_id, cap_id in anchors.items():
        if cap_id in row_of and dota_id in col_of:
            weight[:, col_of[dota_id]] = 0.0
            weight[row_of[cap_id], :] = 0.0
    for dota_id, cap_id in anchors.items():
        if cap_id in row_of and dota_id in col_of:
            weight[row_of[cap_id], col_of[dota_id]] = constants.DOTA_TRAIN_ANCHOR_WEIGHT
    return weight


def align(
    cap: list[CapVideo], dota: dict[str, DotaClip], anchors: dict[str, int]
) -> dict[str, dict[str, int]]:
    """``{dota_id: {block: cap_id}}`` of every block's monotone matching."""
    ordered = [dota[k] for k in sorted(dota, key=windows_key)]
    out: dict[str, dict[str, int]] = {}
    for block, (low, high, rate) in constants.DOTA_TRAIN_BLOCKS.items():
        videos = [c for c in cap if low <= c.cap_id < high]
        if not videos:
            continue
        for i, j in monotone_match(block_weights(videos, ordered, rate, anchors)):
            out.setdefault(ordered[j].dota_id, {})[block] = videos[i].cap_id
    return out


def pick(found: dict[str, int], forbidden: set[int]) -> tuple[str, int] | None:
    for block in constants.DOTA_TRAIN_BLOCK_PREFERENCE:
        if block in found and found[block] not in forbidden:
            return block, found[block]
    return None


def cross_validate(
    cap: list[CapVideo], dota: dict[str, DotaClip], exact: dict[str, int],
    hits: dict[str, set[int]],
) -> dict[str, Any]:
    """Precision / recall on val pairs hidden from the anchors."""
    ids = sorted(exact)
    random.Random(constants.DOTA_TRAIN_CV_SEED).shuffle(ids)
    counts: Counter[tuple[str, bool]] = Counter()
    for fold in range(constants.DOTA_TRAIN_CV_FOLDS):
        held = set(ids[fold :: constants.DOTA_TRAIN_CV_FOLDS])
        found = align(cap, dota, {q: c for q, c in exact.items() if q not in held})
        for q in held:
            chosen = pick(found.get(q, {}), set())
            if chosen is not None:
                counts[chosen[0], chosen[1] in hits[q]] += 1
    right = sum(n for (_, good), n in counts.items() if good)
    total = sum(counts.values())
    return {
        "pairs": len(ids), "right": right, "wrong": total - right,
        "precision": round(right / total, 4) if total else None,
        "recall": round(right / len(ids), 4) if ids else None,
        "per_block": {
            b: {"right": counts[b, True], "wrong": counts[b, False]}
            for b in constants.DOTA_TRAIN_BLOCKS
        },
    }


def independent_checks(
    picks: dict[str, tuple[str, int]], cap: list[CapVideo], dota: dict[str, DotaClip]
) -> dict[str, Any]:
    """Agreement of two fields the alignment never saw, per block."""
    by_id = {c.cap_id: c for c in cap}
    out: dict[str, Any] = {}
    for block, (_, _, rate) in constants.DOTA_TRAIN_BLOCKS.items():
        pairs = [(dota[k], by_id[c]) for k, (b, c) in picks.items() if b == block]
        if not pairs:
            continue
        night = [d.night == (c.light == constants.CAP_LIGHT_NIGHT) for d, c in pairs]
        end = [abs(c.end / rate - d.end) <= END_TOLERANCE_FRAMES for d, c in pairs]
        out[block] = {"clips": len(pairs), "night_agree": round(float(np.mean(night)), 4),
                      "end_within_15_frames": round(float(np.mean(end)), 4)}
    return out


def build_map(
    cap: list[CapVideo], dota: dict[str, DotaClip], val_ids: set[str],
    exact: dict[str, int], hits: dict[str, set[int]],
) -> dict[str, Any]:
    if not val_ids <= dota.keys():
        raise ValueError(f"{len(val_ids - dota.keys())} val ids are not annotated")
    found = align(cap, dota, exact)
    val_caps = set().union(*hits.values()) if hits else set()
    picks: dict[str, tuple[str, int]] = {}
    for dota_id in sorted(dota.keys() - val_ids):
        chosen = pick(found.get(dota_id, {}), val_caps)
        if chosen is not None:
            picks[dota_id] = chosen
    used = [c for _, c in picks.values()]
    if len(set(used)) != len(used):
        raise ValueError("one CAP video was given to two train clips")
    if set(used) & val_caps:
        raise ValueError("a train clip was given a val clip's CAP video")
    by_id = {c.cap_id: c for c in cap}
    clips = {
        k: {"block": b, "rate": constants.DOTA_TRAIN_BLOCKS[b][2], "cap_id": f"{c:06d}",
            "cap_type": by_id[c].cap_type, "cap_frames": by_id[c].total,
            "dota_frames": dota[k].frames}
        for k, (b, c) in picks.items()
    }
    train = dota.keys() - val_ids
    return {
        "rule": "core/docs/final/KIM_RUN.md §52 DT1-DT5",
        "train_clips": len(train), "matched": len(clips),
        "coverage": round(len(clips) / len(train), 4),
        "per_block": dict(Counter(c["block"] for c in clips.values())),
        "youtube_videos": len({dota_group(k) for k in clips}),
        "cross_validation": cross_validate(cap, dota, exact, hits),
        "independent_checks": independent_checks(picks, cap, dota),
        "ids_sha1": lines_sha1(list(clips)),
        "clips": clips,
    }


def frame_map(dota_frames: int, cap_frames: int, rate: int) -> list[int]:
    """CAP frame of every DoTA frame: one fixed clock per rate, clamped to the CAP video."""
    raw = rate * np.arange(dota_frames) + constants.DOTA_TRAIN_FRAME_PHASE[rate]
    return [int(j) for j in np.clip(raw, 0, cap_frames - 1)]


def expand(frozen: dict[str, Any]) -> dict[str, Any]:
    """The frozen map in ``dota_cap_frames``' alignment format."""
    clips = {
        k: {**c, "dota_id": k, "reason": OK,
            "frame_map": frame_map(c["dota_frames"], c["cap_frames"], c["rate"])}
        for k, c in frozen["clips"].items()
    }
    return {"clips": clips, "ids_sha1": frozen["ids_sha1"]}


def load_map(path: Path) -> dict[str, Any]:
    frozen: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if lines_sha1(list(frozen["clips"])) != frozen["ids_sha1"]:
        raise ValueError(f"{path}: ids do not match its ids_sha1")
    return frozen


def finalize(
    alignment: dict[str, Any], reports: list[dict[str, Any]], archive_dir: Path
) -> dict[str, Any]:
    """Keep the packed clips; only a frame-count mismatch may drop one."""
    mismatched = {k for r in reports for k, v in r.items() if v["reason"] == CAP_FRAMES_MISMATCH}
    kept: dict[str, Any] = {}
    unresolved: list[str] = []
    for dota_id, clip in alignment["clips"].items():
        if archive_frames(archive_path(archive_dir, dota_id)) == clip["dota_frames"]:
            kept[dota_id] = clip
        elif dota_id not in mismatched:
            unresolved.append(dota_id)
    if unresolved:
        raise ValueError(f"{len(unresolved)} clips not packed yet: {sorted(unresolved)[:5]}")
    dropped = sorted(alignment["clips"].keys() - kept.keys())
    LOGGER.info("Coverage %d/%d (%d dropped: CAP frame count)", len(kept),
                len(alignment["clips"]), len(dropped))
    return {"clips": kept, "ids_sha1": lines_sha1(list(kept)), "dropped": dropped}


# --------------------------------------------------------------------------- CLI


def run_match(args: argparse.Namespace) -> None:
    if args.out.exists() and not args.force:
        raise FileExistsError(f"{args.out} is frozen; --force to overwrite")
    val_ids = {v for v in args.val_split.read_text(encoding="utf-8").split() if v}
    exact, hits = read_val_pairs(args.matches)
    payload = build_map(read_cap_csv(args.cap_csv), read_dota(args.annotations), val_ids,
                        exact, hits)
    payload["sources"] = {
        "cap_csv_sha256": file_sha256(args.cap_csv),
        "annotations_sha256": file_sha256(args.annotations),
        "matches_sha256": file_sha256(args.matches),
    }
    write_json_atomic(args.out, payload)
    LOGGER.info("%s", {k: v for k, v in payload.items() if k != "clips"})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    default_map = constants.DOTA_TRAIN_SPLITS_DIR / constants.DOTA_TRAIN_MAP_FILENAME
    m = sub.add_parser("match", help="align DoTA names to CAP ids and freeze the map")
    m.add_argument("--cap-csv", type=Path, required=True)
    m.add_argument("--annotations", type=Path, required=True, help="DoTA_annotations.zip")
    m.add_argument("--val-split", type=Path, required=True)
    m.add_argument("--matches", type=Path, required=True, help="mmau_p0_matches.json")
    m.add_argument("--out", type=Path, default=default_map)
    m.add_argument("--force", action="store_true")
    e = sub.add_parser("expand", help="frozen map -> dota_cap_frames alignment json")
    e.add_argument("--map", type=Path, default=default_map)
    e.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("finalize", help="alignment of the clips actually packed")
    f.add_argument("--alignment", type=Path, required=True)
    f.add_argument("--archive-dir", type=Path, required=True)
    f.add_argument("--reports", type=Path, nargs="+", required=True)
    f.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "match":
        run_match(args)
    elif args.command == "expand":
        write_json_atomic(args.out, expand(load_map(args.map)))
    else:
        reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.reports]
        alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
        write_json_atomic(args.out, finalize(alignment, reports, args.archive_dir))


if __name__ == "__main__":
    main()
