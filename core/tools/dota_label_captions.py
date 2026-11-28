"""Label-template captions for the DoTA train clips (``KIM_RUN.md`` §65).

One sentence per clip, built only from DoTA's own annotation -- the accident type, the ego flag and
the object categories boxed in the accident frames -- so it is right about who collides and how by
construction and costs no GPU. It is the upper bound of what a caption carrying the label can teach
the caption path; a Qwen caption adds detail but also errors (§63.1 / §63.3).

    python -m core.tools.dota_label_captions build \\
        --annotations DoTA_annotations.zip --out <dir>/dota_label_captions.json

The manifest is ``{"train_split": "dota_train", "train_ids_sha1": ..., "captions": {id: text}}``,
the format ``core.data.captions.load_caption_manifest`` reads.
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from core import constants
from core.data.captions import MANIFEST_CAPTIONS_KEY, MANIFEST_SHA_KEY, dota_train_ids
from core.data.v2_splits import lines_sha1
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

NORMAL_FRAME = "normal"
TOP_CAPTIONS = 15  # most frequent captions printed in the build summary


def parties(record: dict[str, Any]) -> list[str]:
    """Nouns of the road users boxed in the accident frames, most frequent first.

    One noun per object track. DoTA boxes a rider and its vehicle separately: riders are dropped
    when a motorcycle or bicycle is boxed, and named "rider" only when none is.
    """
    tracks: dict[Any, str] = {}
    for frame in record["labels"]:
        if frame["accident_name"] == NORMAL_FRAME:
            continue
        for obj in frame["objects"]:
            tracks.setdefault(obj["obj_track_id"], obj["category"])
    cats = Counter(tracks.values())
    if any(cats[c] for c in constants.DOTA_LABEL_CAPTION_RIDDEN):
        cats["rider"] = 0
    order = list(constants.DOTA_LABEL_CAPTION_NOUNS)
    out: list[str] = []
    for cat in sorted((c for c in cats if cats[c] > 0), key=lambda c: (-cats[c], order.index(c))):
        out += [constants.DOTA_LABEL_CAPTION_NOUNS[cat]] * cats[cat]
    return out


def _phrase(nouns: list[str]) -> str:
    """``["car", "car", "truck"]`` -> ``"two cars and a truck"``."""
    counts = Counter(nouns)
    parts = []
    for noun in dict.fromkeys(nouns):
        n = counts[noun]
        if n == 1:
            parts.append(f"{'an' if noun[0] in 'aeiou' else 'a'} {noun}")
        else:
            parts.append(f"{constants.DOTA_LABEL_CAPTION_NUMBERS.get(n, 'several')} {noun}s")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def label_caption(record: dict[str, Any]) -> str:
    """The clip's caption from ``accident_name``, ``ego_involve`` and :func:`parties`."""
    kind = record["accident_name"]
    if kind not in constants.DOTA_LABEL_CAPTION_EGO:
        raise ValueError(f"{record.get('video_name')}: no template for accident type {kind!r}")
    nouns = parties(record)
    if record["ego_involve"]:
        main = [n for n in nouns if kind != "pedestrian" or n == "pedestrian"][:1]
        if kind == "pedestrian" and not main:
            main = ["pedestrian"]
        p = _phrase(main) if main else constants.DOTA_LABEL_CAPTION_DEFAULT_OTHER
        return constants.DOTA_LABEL_CAPTION_EGO[kind].format(p=p)
    if kind == "pedestrian":
        nouns = [n for n in nouns if n != "pedestrian"]
    if kind in constants.DOTA_LABEL_CAPTION_SINGLE:
        p = _phrase(nouns[:1]) if nouns else constants.DOTA_LABEL_CAPTION_DEFAULT_ONE
    else:
        p = _phrase(nouns) if len(nouns) > 1 else (
            f"{_phrase(nouns)} and {constants.DOTA_LABEL_CAPTION_DEFAULT_OTHER}" if nouns
            else f"two {constants.DOTA_LABEL_CAPTION_DEFAULT_OTHER.split()[-1]}s")
    text = constants.DOTA_LABEL_CAPTION_OTHER[kind].format(p=p)
    return text[0].upper() + text[1:] + constants.DOTA_LABEL_CAPTION_NOT_EGO_TAIL


def read_records(annotations: Path, ids: list[str]) -> dict[str, dict[str, Any]]:
    """DoTA's per-clip annotation JSONs from the official zip or an extracted folder."""
    if annotations.is_dir():
        return {v: json.loads((annotations / f"{v}.json").read_text(encoding="utf-8"))
                for v in ids}
    with zipfile.ZipFile(annotations) as zf:
        return {v: json.loads(zf.read(f"annotations/{v}.json")) for v in ids}


def build_manifest(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The §65 manifest over exactly the §55 train clips (another id set raises)."""
    want = dota_train_ids()
    if sorted(records) != want:
        raise ValueError(f"{len(records)} records, the §55 train set has {len(want)} clips")
    return {constants.CAPTION_MANIFEST_SPLIT_KEY: constants.CAPTION_SPLIT_DOTA,
            MANIFEST_SHA_KEY: lines_sha1(want),
            MANIFEST_CAPTIONS_KEY: {v: label_caption(records[v]) for v in want}}


def summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Distinct captions, the most frequent ones and word counts (printed, never a gate)."""
    caps = list(manifest[MANIFEST_CAPTIONS_KEY].values())
    counts = Counter(caps)
    words = sorted(len(c.split()) for c in caps)
    return {"clips": len(caps), "distinct": len(counts), "largest_share": max(counts.values())
            / len(caps), "words_median": words[len(words) // 2], "words_max": words[-1],
            "top": counts.most_common(TOP_CAPTIONS)}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="write the label-template caption manifest")
    build.add_argument("--annotations", type=Path, required=True,
                       help="DoTA_annotations.zip or its extracted annotations/ folder")
    build.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(read_records(args.annotations, dota_train_ids()))
    write_json_atomic(args.out, manifest)
    LOGGER.info("%s", json.dumps(summary(manifest), indent=1))
    LOGGER.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
