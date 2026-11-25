"""Pixel gate of the recovered DoTA train clips (``KIM_RUN.md`` §52 DT7): DoTA's boxes by eye.

A train clip has no reference feature, so the map of ``dota_train_match`` is checked on pixels:
DoTA annotates a box per accident participant and frame; drawn on the frames recovered from CAP
they must sit on the vehicle. A wrong clip, or a shifted clock, puts them elsewhere.

``sheet`` draws numbered rows (first / middle / last boxed frame of the annotated span), shuffled
and unlabelled: real train clips per frame-rate group, val clips whose pixels CLIP already
verified (controls) and **decoys** (a neighbouring clip of the same YouTube video under this
clip's boxes -- the error an off-by-one match would make). ``key.json`` holds the truth; do not
open it before judging. ``score`` takes the row numbers the judge calls wrong and stores them.
``--exclude-key`` keeps every clip of an earlier sheet out of a redraw, so it is judged blind.

CLI::

    python -m core.tools.dota_train_sheet sheet --alignment .../dota_train_alignment.json \\
        --archive-dir .../DoTA_CAP_train_frames --val-archive-dir .../DoTA_CAP_frames \\
        --annotations .../DoTA_annotations.zip --out-dir .../REPORTS/dota_train/box_sheet
    python -m core.tools.dota_train_sheet score --key .../key.json --flagged 3 17 \\
        --out .../gate.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageDraw

from core import constants
from core.data.v2_splits import dota_group
from core.tools.dota_cap_frames import archive_frames, archive_path
from core.tools.dota_train_match import windows_key
from core.tools.kill_switch_probe import write_json_atomic

LOGGER = logging.getLogger(__name__)

REAL, CONTROL, DECOY = "real", "control", "decoy"
GROUP_OF_RATE = {1: "10fps", 3: "30fps"}
KEY_JSON = "key.json"
PAGE_NAME = "page_{:02d}.png"


@dataclass(frozen=True)
class Row:
    kind: str
    group: str
    dota_id: str  # whose boxes are drawn
    frames_id: str  # whose frames are shown (differs for a decoy)
    frames: list[int]


def read_boxes(zf: zipfile.ZipFile, dota_id: str) -> dict[int, list[list[float]]]:
    """``{frame: boxes}`` of the annotated span; frames without a box are absent."""
    d = json.loads(zf.read(f"annotations/{dota_id}.json"))
    return {
        k: [o["bbox"] for o in label["objects"]]
        for k, label in enumerate(d["labels"])
        if label["objects"] and d["anomaly_start"] <= k < d["anomaly_end"]
    }


def spread(frames: list[int], count: int) -> list[int]:
    """``count`` frames from first to last (fewer if there are not that many)."""
    if len(frames) <= count:
        return frames
    last = len(frames) - 1
    return [frames[round(i * last / (count - 1))] for i in range(count)]


def neighbour(dota_id: str, pool: list[str]) -> list[str]:
    """Clips of the same YouTube video, nearest in name order first."""
    ordered = sorted(
        (k for k in pool if dota_group(k) == dota_group(dota_id)), key=windows_key
    )
    at = ordered.index(dota_id)
    near = sorted(range(len(ordered)), key=lambda i: (abs(i - at), i))
    return [ordered[i] for i in near if i != at]


def choose_rows(
    clips: dict[str, dict[str, Any]], zf: zipfile.ZipFile, archive_dir: Path,
    val_archive_dir: Path, seed: int, n_frames: int = constants.DOTA_SHEET_FRAMES,
    seen: frozenset[str] = frozenset(),
) -> list[Row]:
    rng = random.Random(seed)
    counts = {k: archive_frames(archive_path(archive_dir, k)) for k in sorted(clips)}
    packed = [k for k, n in counts.items() if n and k not in seen]
    rng.shuffle(packed)
    rows: list[Row] = []
    used: set[str] = set()

    def boxed(dota_id: str, limit: int) -> list[int]:
        return spread([f for f in sorted(read_boxes(zf, dota_id)) if f < limit], n_frames)

    for group, want in constants.DOTA_SHEET_PER_GROUP.items():
        taken = 0
        for k in packed:
            if taken == want:
                break
            if k in used or GROUP_OF_RATE[clips[k]["rate"]] != group:
                continue
            frames = boxed(k, counts[k] or 0)
            if len(frames) == n_frames:
                rows.append(Row(REAL, group, k, k, frames))
                used.add(k)
                taken += 1
        if taken < want:
            raise ValueError(f"only {taken} boxed clips in group {group}, need {want}")
    decoys = 0
    for k in packed:
        if decoys == constants.DOTA_SHEET_DECOYS:
            break
        if k in used:
            continue
        for other in neighbour(k, packed):
            frames = boxed(k, min(counts[k] or 0, counts[other] or 0))
            if len(frames) == n_frames:
                rows.append(Row(DECOY, GROUP_OF_RATE[clips[k]["rate"]], k, other, frames))
                used.add(k)
                decoys += 1
                break
    if decoys < constants.DOTA_SHEET_DECOYS:
        raise ValueError(f"only {decoys} decoys, need {constants.DOTA_SHEET_DECOYS}")
    controls = sorted(p.stem for p in val_archive_dir.glob("*.zip") if p.stem not in seen)
    rng.shuffle(controls)
    n_controls = 0
    for k in controls:
        if n_controls == constants.DOTA_SHEET_VAL_CONTROLS:
            break
        frames = boxed(k, archive_frames(archive_path(val_archive_dir, k)) or 0)
        if len(frames) == n_frames:
            rows.append(Row(CONTROL, CONTROL, k, k, frames))
            n_controls += 1
    if n_controls < constants.DOTA_SHEET_VAL_CONTROLS:
        raise ValueError(f"only {n_controls} val controls in {val_archive_dir}")
    rng.shuffle(rows)
    return rows


def load_frames(path: Path, indices: list[int]) -> list[Image.Image]:
    images = []
    with zipfile.ZipFile(path) as zf:
        by_index = {int(PurePosixPath(n).stem): n for n in zf.namelist()}
        for i in indices:
            images.append(Image.open(io.BytesIO(zf.read(by_index[i]))).convert("RGB"))
    return images


def draw_boxes(image: Image.Image, boxes: list[list[float]]) -> Image.Image:
    """One tile: boxes (on DoTA's 1280 x 720 grid) scaled onto the frame."""
    width = constants.DOTA_SHEET_TILE_WIDTH
    height = round(width * constants.DOTA_BOX_HEIGHT / constants.DOTA_BOX_WIDTH)
    tile = image.resize((width, height))
    sx, sy = width / constants.DOTA_BOX_WIDTH, height / constants.DOTA_BOX_HEIGHT
    pen = ImageDraw.Draw(tile)
    for x1, y1, x2, y2 in boxes:
        pen.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                      outline=constants.DOTA_SHEET_BOX_COLOUR, width=constants.DOTA_SHEET_BOX_LINE)
    return tile


def draw_page(numbered: list[tuple[int, list[Image.Image]]]) -> Image.Image:
    tile_w, tile_h = numbered[0][1][0].size
    pad, label = constants.DOTA_SHEET_MARGIN, constants.DOTA_SHEET_LABEL_WIDTH
    columns = len(numbered[0][1])
    page = Image.new("RGB", (label + columns * (tile_w + pad),
                             len(numbered) * (tile_h + pad)), "white")
    pen = ImageDraw.Draw(page)
    for r, (number, tiles) in enumerate(numbered):
        top = r * (tile_h + pad)
        pen.text((pad, top + tile_h // 2), f"#{number}", fill="black", font_size=label // 3)
        for c, tile in enumerate(tiles):
            page.paste(tile, (label + c * (tile_w + pad), top))
    return page


def seen_clips(keys: list[Path]) -> frozenset[str]:
    """Every clip an earlier sheet showed, as boxes or as pixels."""
    rows = [r for p in keys
            for r in json.loads(p.read_text(encoding="utf-8"))["rows"].values()]
    return frozenset(r[field] for r in rows for field in ("dota_id", "frames_id"))


def run_sheet(args: argparse.Namespace) -> None:
    clips = json.loads(args.alignment.read_text(encoding="utf-8"))["clips"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen = seen_clips(args.exclude_key)
    with zipfile.ZipFile(args.annotations) as zf:
        rows = choose_rows(clips, zf, args.archive_dir, args.val_archive_dir, args.seed,
                           args.frames, seen)
        drawn: list[tuple[int, list[Image.Image]]] = []
        sizes: dict[str, int] = {}
        for number, row in enumerate(rows, start=1):
            source = args.val_archive_dir if row.kind == CONTROL else args.archive_dir
            boxes = read_boxes(zf, row.dota_id)
            images = load_frames(archive_path(source, row.frames_id), row.frames)
            size = "x".join(str(v) for v in images[0].size)
            sizes[size] = sizes.get(size, 0) + 1
            drawn.append((number, [draw_boxes(im, boxes[f])
                                   for im, f in zip(images, row.frames, strict=True)]))
    per_page = constants.DOTA_SHEET_ROWS_PER_PAGE
    for page, start in enumerate(range(0, len(drawn), per_page), start=1):
        draw_page(drawn[start : start + per_page]).save(args.out_dir / PAGE_NAME.format(page))
    key = {str(n): asdict(row) for n, row in enumerate(rows, start=1)}
    write_json_atomic(args.out_dir / KEY_JSON, {"seed": args.seed, "frames_per_row": args.frames,
                                                "excluded": sorted(seen), "frame_sizes": sizes,
                                                "rows": key})
    LOGGER.info("%d rows on %d pages, %d earlier clips excluded -> %s", len(rows),
                -(-len(rows) // per_page), len(seen), args.out_dir)


def score(key: dict[str, Any], flagged: set[int]) -> dict[str, Any]:
    """DT7's verdict: is the judge sensitive, and which rate groups are usable."""
    rows = {int(n): r for n, r in key["rows"].items()}
    unknown = flagged - rows.keys()
    if unknown:
        raise ValueError(f"no such rows: {sorted(unknown)}")
    missed = sorted(n for n, r in rows.items() if r["kind"] == DECOY and n not in flagged)
    caught = sum(1 for n in flagged if rows[n]["kind"] == DECOY)
    controls = sum(1 for n in flagged if rows[n]["kind"] == CONTROL)
    sensitive = caught >= constants.DOTA_SHEET_MIN_DECOYS_CAUGHT
    fair = controls <= constants.DOTA_SHEET_MAX_CONTROLS_FLAGGED
    groups: dict[str, Any] = {}
    for group in constants.DOTA_SHEET_PER_GROUP:
        wrong = sorted(n for n in flagged
                       if rows[n]["kind"] == REAL and rows[n]["group"] == group)
        groups[group] = {
            "clips": sum(1 for r in rows.values() if r["kind"] == REAL and r["group"] == group),
            "flagged": len(wrong), "flagged_ids": [rows[n]["dota_id"] for n in wrong],
            "pass": len(wrong) <= constants.DOTA_SHEET_MAX_FLAGGED,
        }
    valid = sensitive and fair
    return {"flagged": sorted(flagged), "decoys_caught": caught,
            "decoys_missed": {str(n): rows[n]["dota_id"] for n in missed},
            "controls_flagged": controls, "gate_valid": valid,
            "groups": groups, "usable_groups": [g for g, v in groups.items() if v["pass"]]
            if valid else []}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sheet", help="draw the shuffled, unlabelled rows")
    s.add_argument("--alignment", type=Path, required=True)
    s.add_argument("--archive-dir", type=Path, required=True)
    s.add_argument("--val-archive-dir", type=Path, required=True)
    s.add_argument("--annotations", type=Path, required=True, help="DoTA_annotations.zip")
    s.add_argument("--seed", type=int, default=constants.DOTA_SHEET_SEED)
    s.add_argument("--frames", type=int, default=constants.DOTA_SHEET_FRAMES,
                   help="boxed frames per row")
    s.add_argument("--exclude-key", type=Path, nargs="*", default=[],
                   help="key.json of earlier sheets: their clips are not drawn again")
    s.add_argument("--out-dir", type=Path, required=True)
    g = sub.add_parser("score", help="verdict from the row numbers judged wrong")
    g.add_argument("--key", type=Path, required=True)
    g.add_argument("--flagged", type=int, nargs="*", default=[])
    g.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "sheet":
        run_sheet(args)
        return
    result = score(json.loads(args.key.read_text(encoding="utf-8")), set(args.flagged))
    write_json_atomic(args.out, result)
    LOGGER.info("%s", result)


if __name__ == "__main__":
    main()
