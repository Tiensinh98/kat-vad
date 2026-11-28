"""One caption per recovered DoTA train clip with Qwen2.5-VL (``core/docs/final/KIM_RUN.md`` §63).

A data artifact; nothing here trains. The clips are the §52 map minus ``DOTA_TRAIN_EXCLUDED``
(3,020), read from their ``{dota_id}.zip`` frame archives (DoTA-native 10 fps, entry
``{d:06d}.jpg`` = DoTA frame ``d``). Per clip, ``DOTA_CAPTION_FRAMES`` frames are taken at the
bin centres of ``[anomaly_start - PRE, min(anomaly_start + POST, anomaly_end)]`` (widened to the
clip when shorter than the frame count), resized to ``DOTA_CAPTION_SIZE`` and passed as one
video. Unlike Nexar (§29.3), DoTA annotates the accident type and whether the ego vehicle takes
part, so the prompt carries both as a hint (``DOTA_CAPTION_PROMPT``), the way LaGoVAD's prompt
carries the video label. The captioner, the cleaning, the record identity check and the
Tab. B QC scoring are ``core.tools.nexar_caption``'s, unchanged.

Subcommands::

    python -m core.tools.dota_caption pilot --archive-dir data/DoTA_CAP_train_frames \\
        --annotations DoTA_annotations.zip --revision <sha> --out-dir outputs/dota_captions
    python -m core.tools.dota_caption qc-score --pilot-dir outputs/dota_captions/pilot/<tag>

``pilot`` captions the ``DOTA_CAPTION_PILOT_SAMPLE`` seeded clips (one per accident type, the
rest drawn at random) into ``pilot/<prompt tag>_<model>_<frames>f/`` and writes one contact
sheet per clip plus ``qc/qc_ratings.csv`` (empty ``score``; optional ``type_ok`` = the caption
agrees with the annotated type, ``roles_ok`` = the road users and their roles are right).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import random
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from core import constants
from core.device import resolve_device
from core.tools.dota_train_extract import train_clips
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic
from core.tools.nexar_caption import (
    CAPTIONS_DIRNAME,
    QC_CRITERIA,
    QC_DIRNAME,
    QC_FLAG_COLUMNS,
    QC_RATINGS_FILENAME,
    QC_SCORE_COLUMN,
    QC_SHEET_FILENAME,
    REFUSED_REVISIONS,
    QwenCaptioner,
    clean_caption,
    contact_sheet,
    expected_grid,
    forbidden_terms,
    load_record,
    prompt_tag,
    score_qc,
)

LOGGER = logging.getLogger(__name__)

PILOT_DIRNAME = "pilot"
PILOT_SUMMARY_FILENAME = "pilot_summary.json"
ANNOTATION_KEYS = ("num_frames", "anomaly_start", "anomaly_end", "accident_name", "ego_involve")
EGO_PHRASE = "ego vehicle"


# --------------------------------------------------------------------------- pure pieces
def load_annotations(annotations: Path, ids: list[str]) -> dict[str, dict[str, Any]]:
    """DoTA's per-clip fields for ``ids`` from the official zip; an unknown type raises."""
    out: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(annotations) as zf:
        for vid in ids:
            record = json.loads(zf.read(f"annotations/{vid}.json"))
            out[vid] = {k: record[k] for k in ANNOTATION_KEYS}
    unknown = sorted({a["accident_name"] for a in out.values()}
                     - set(constants.DOTA_CAPTION_TYPE_TEXT))
    if unknown:
        raise ValueError(f"accident types without a prompt sentence: {unknown}")
    return out


def caption_window(ann: dict[str, Any], n: int = constants.DOTA_CAPTION_FRAMES
                   ) -> tuple[list[int], tuple[int, int]]:
    """``n`` distinct DoTA frame indices at the bin centres of the window ``[start, end)``.

    The window opens ``PRE`` s before ``anomaly_start`` and closes ``POST`` s after it or at
    ``anomaly_end``, whichever is sooner; one shorter than ``n`` frames is widened inside the clip.
    """
    frames = int(ann["num_frames"])
    if frames < n:
        raise ValueError(f"clip holds {frames} frames < {n}")
    onset = int(ann["anomaly_start"])
    start = max(0, onset - round(constants.DOTA_CAPTION_PRE_ONSET_S * constants.DOTA_FPS))
    close = onset + round(constants.DOTA_CAPTION_POST_ONSET_S * constants.DOTA_FPS)
    end = min(frames, close, max(int(ann["anomaly_end"]), onset + 1))
    if end - start < n:
        end = min(frames, start + n)
        start = max(0, end - n)
    width = (end - start) / n  # >= 1, so the floored bin centres are distinct
    indices = [int(start + width * (i + 0.5)) for i in range(n)]
    if len(set(indices)) != n:
        raise ValueError(f"window [{start}, {end}) gave repeated frames {indices}")
    return indices, (start, end)


def prompt_for(ann: dict[str, Any]) -> str:
    """The template filled with the clip's annotated type and ego involvement."""
    return constants.DOTA_CAPTION_PROMPT.format(
        type_text=constants.DOTA_CAPTION_TYPE_TEXT[ann["accident_name"]],
        ego_text=constants.DOTA_CAPTION_EGO_TEXT[bool(ann["ego_involve"])])


def decode_frames(path: Path, indices: list[int], frames: int,
                  size: tuple[int, int] = constants.DOTA_CAPTION_SIZE) -> np.ndarray:
    """``(n, H, W, 3)`` uint8 RGB of ``indices`` from one clip archive; a wrong count raises."""
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as zf:  # one sequential Drive read
        names = sorted(zf.namelist())
        if len(names) != frames:
            raise ValueError(f"{path}: {len(names)} frames, the annotation says {frames}")
        out = []
        for i in indices:
            with Image.open(io.BytesIO(zf.read(names[i]))) as image:
                resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
                out.append(np.asarray(resized))
    return np.stack(out)


def pilot_sample(anns: dict[str, dict[str, Any]], n: int = constants.DOTA_CAPTION_PILOT_SAMPLE,
                 seed: int = constants.DOTA_CAPTION_PILOT_SEED) -> list[str]:
    """One clip per accident type (seeded), then the rest drawn at random from what is left."""
    rng = random.Random(seed)  # nosec B311 - a fixed, documented draw
    by_type: dict[str, list[str]] = {}
    for vid in sorted(anns):
        by_type.setdefault(anns[vid]["accident_name"], []).append(vid)
    chosen = [rng.choice(by_type[t]) for t in sorted(by_type)]
    if len(chosen) > n:
        raise ValueError(f"{len(chosen)} accident types > a pilot of {n}")
    rest = sorted(set(anns) - set(chosen))
    return sorted(chosen + rng.sample(rest, n - len(chosen)))


def identity(model: str, revision: str, quant: str) -> dict[str, Any]:
    """The keys ``nexar_caption.load_record`` compares; a record that differs is refused."""
    if revision.strip().lower() in REFUSED_REVISIONS:
        raise ValueError(f"pin a model revision (a commit sha), not {revision!r} (C4)")
    if quant not in constants.NEXAR_CAPTION_QUANT_CHOICES:
        raise ValueError(f"quant must be one of {constants.NEXAR_CAPTION_QUANT_CHOICES}")
    digest = hashlib.sha1(constants.DOTA_CAPTION_PROMPT.encode("utf-8"), usedforsecurity=False)
    return {
        "model": model, "revision": revision, "quant": quant, "prompt_sha1": digest.hexdigest(),
        "frames": constants.DOTA_CAPTION_FRAMES, "size": list(constants.DOTA_CAPTION_SIZE),
        "max_new_tokens": constants.NEXAR_CAPTION_MAX_NEW_TOKENS,
        "pre_event_s": constants.DOTA_CAPTION_PRE_ONSET_S,
        "post_event_s": constants.DOTA_CAPTION_POST_ONSET_S,
    }


def pilot_dirname(model: str) -> str:
    """One folder per prompt template, model and frame count."""
    tag = prompt_tag(constants.DOTA_CAPTION_PROMPT)
    return f"{tag}_{model.split('/')[-1]}_{constants.DOTA_CAPTION_FRAMES}f"


def summarize(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Printed beside the QC, never deciding: lengths, scene words, ego wording on non-ego clips."""
    words = [r["words"] for r in records.values()]
    non_ego = [r for r in records.values() if not r["ego_involve"]]
    return {
        "clips": len(records),
        "words_min": min(words), "words_median": float(np.median(words)), "words_max": max(words),
        "over_max_words": sum(w > constants.NEXAR_CAPTION_MAX_WORDS for w in words),
        "distinct": len({r["caption"] for r in records.values()}),
        "scene_word_share": float(np.mean([bool(r["forbidden"]) for r in records.values()])),
        "non_ego_clips": len(non_ego),
        "non_ego_naming_ego": sum(EGO_PHRASE in r["caption"].lower() for r in non_ego),
        "types": {t: sum(r["accident_name"] == t for r in records.values())
                  for t in sorted({r["accident_name"] for r in records.values()})},
    }


# --------------------------------------------------------------------------- model
class DotaCaptioner(QwenCaptioner):
    """``nexar_caption.QwenCaptioner`` with the prompt given per clip."""

    def __init__(self, model: str, revision: str, device: torch.device, quant: str) -> None:
        super().__init__(model, revision, constants.DOTA_CAPTION_PROMPT, device, quant)

    @torch.no_grad()
    def caption(self, frames: np.ndarray, fps: float, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "video"},
                                                 {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[frames], fps=fps, padding=True,
                                return_tensors="pt").to(self.device)
        grid = tuple(int(x) for x in inputs["video_grid_thw"][0])
        want = expected_grid(len(frames), (frames.shape[2], frames.shape[1]))
        if grid != want:
            raise ValueError(f"video grid {grid} != {want}: the frames were resized")
        out = self.model.generate(**inputs, max_new_tokens=constants.NEXAR_CAPTION_MAX_NEW_TOKENS,
                                  do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        answer: str = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0]
        return answer


# --------------------------------------------------------------------------- run
def caption_clips(archive_dir: Path, anns: dict[str, dict[str, Any]], ids: list[str],
                  out_dir: Path, ident: dict[str, Any], captioner: Any) -> dict[str, Any]:
    """Caption ``ids``; finished clips are skipped (atomic per-clip records, C11c)."""
    records: dict[str, dict[str, Any]] = {}
    t0, made = time.time(), 0
    for n, vid in enumerate(ids, 1):
        path = out_dir / CAPTIONS_DIRNAME / f"{vid}.json"
        record = load_record(path, ident)
        if record is None:
            ann = anns[vid]
            indices, (start, end) = caption_window(ann)
            frames = decode_frames(archive_dir / f"{vid}.zip", indices, ann["num_frames"])
            prompt = prompt_for(ann)
            fps = len(indices) * constants.DOTA_FPS / (end - start)
            answer = clean_caption(captioner.caption(frames, fps, prompt))
            record = {**ident, "video_id": vid, "caption": answer, "prompt": prompt,
                      "accident_name": ann["accident_name"],
                      "ego_involve": bool(ann["ego_involve"]),
                      "anomaly": [ann["anomaly_start"], ann["anomaly_end"]],
                      "window_frames": [start, end], "frame_indices": indices,
                      "words": len(answer.split()), "forbidden": forbidden_terms(answer)}
            write_json_atomic(path, record)
            made += 1
            LOGGER.info("%d/%d %s (%.1f s/clip): %s", n, len(ids), vid,
                        (time.time() - t0) / made, answer)
        records[vid] = record
    return records


def write_sheet(archive_dir: Path, records: dict[str, dict[str, Any]], sheet_dir: Path,
                anns: dict[str, dict[str, Any]], title: str) -> None:
    """Contact sheets + ratings csv + a markdown sheet for the user."""
    sheet_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "",
             f"Give each caption an integer `{QC_SCORE_COLUMN}` from "
             f"{constants.NEXAR_CAPTION_QC_SCORE_MIN} to {constants.NEXAR_CAPTION_QC_SCORE_MAX} in "
             f"`{QC_RATINGS_FILENAME}` (LaGoVAD Tab. B), judged against the frames. Optional "
             "y / n: `type_ok` = the caption matches the annotated type, `roles_ok` = road users "
             "and who does what are right. Printed only; nothing is decided here.", "",
             "| score | criteria |", "|---|---|",
             *(f"| {s} | {QC_CRITERIA[s]} |" for s in sorted(QC_CRITERIA, reverse=True)), "",
             "| # | clip | type | ego | window | caption |", "|---|---|---|---|---|---|"]
    with (sheet_dir / QC_RATINGS_FILENAME).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "video_id", "event_type", "ego_involve", "caption",
                         QC_SCORE_COLUMN, *QC_FLAG_COLUMNS])
        for rank, vid in enumerate(sorted(records), 1):
            r = records[vid]
            frames = decode_frames(archive_dir / f"{vid}.zip", r["frame_indices"],
                                   anns[vid]["num_frames"])
            contact_sheet(frames).save(sheet_dir / f"{rank:02d}_{vid}.png")
            writer.writerow([rank, vid, r["accident_name"], r["ego_involve"], r["caption"], "",
                             *[""] * len(QC_FLAG_COLUMNS)])
            start, end = r["window_frames"]
            lines.append(f"| {rank} | {vid} | {r['accident_name']} | {r['ego_involve']} | "
                         f"{start}-{end} (onset {r['anomaly'][0]}) | {r['caption']} |")
    write_text_atomic(sheet_dir / QC_SHEET_FILENAME, "\n".join(lines) + "\n")


def run_pilot(args: argparse.Namespace) -> Path:
    clips = train_clips(constants.DOTA_TRAIN_SPLITS_DIR / constants.DOTA_TRAIN_MAP_FILENAME)
    anns = load_annotations(args.annotations, sorted(clips))
    ids = pilot_sample(anns)
    missing = [v for v in ids if not (args.archive_dir / f"{v}.zip").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} pilot archives missing: {missing[:5]}")
    ident = identity(args.model, args.revision, args.quant)
    pilot_dir: Path = args.out_dir / PILOT_DIRNAME / pilot_dirname(args.model)
    write_text_atomic(pilot_dir / "prompt_template.txt", constants.DOTA_CAPTION_PROMPT + "\n")
    captioner = DotaCaptioner(args.model, args.revision, resolve_device(args.device), args.quant)
    records = caption_clips(args.archive_dir, anns, ids, pilot_dir, ident, captioner)
    write_json_atomic(pilot_dir / PILOT_SUMMARY_FILENAME,
                      {"prompt_tag": prompt_tag(constants.DOTA_CAPTION_PROMPT), **ident,
                       "ids": ids, **summarize(records)})
    write_sheet(args.archive_dir, records, pilot_dir / QC_DIRNAME, anns,
                f"DoTA train caption pilot, prompt {prompt_tag(constants.DOTA_CAPTION_PROMPT)} "
                "(§63)")
    return pilot_dir


# --------------------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--archive-dir", type=Path, required=True, help="DoTA_CAP_train_frames")
    pilot.add_argument("--annotations", type=Path, required=True, help="DoTA_annotations.zip")
    pilot.add_argument("--out-dir", type=Path, required=True)
    pilot.add_argument("--revision", required=True, help="model commit sha (never main; C4)")
    pilot.add_argument("--model", default=constants.NEXAR_CAPTION_MODEL)
    pilot.add_argument("--quant", choices=constants.NEXAR_CAPTION_QUANT_CHOICES,
                       default=constants.NEXAR_CAPTION_QUANT)
    pilot.add_argument("--device", default="auto")
    score = sub.add_parser("qc-score")
    score.add_argument("--pilot-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "qc-score":
        LOGGER.info("QC: %s", score_qc(args.pilot_dir))
        return
    LOGGER.info("pilot sheet: %s", run_pilot(args) / QC_DIRNAME / QC_SHEET_FILENAME)


if __name__ == "__main__":
    main()
