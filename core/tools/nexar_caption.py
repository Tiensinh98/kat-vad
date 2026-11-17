"""One caption per Nexar event with Qwen2.5-VL (``core/docs/final/KIM_RUN.md`` §29, §29.3).

A data artifact for option C; nothing here trains. For each positive ``nexar_train`` video the
window ``[t_event - NEXAR_CAPTION_PRE_EVENT_S, t_event + NEXAR_POST_EVENT_S]`` (§29.2 CP2'; not
the D-N1 label span, which starts at ``t_alert`` and can leave one frame before contact)
is sampled at ``NEXAR_CAPTION_FRAMES`` evenly spaced frames of the original mp4, each resized to
``NEXAR_CAPTION_SIZE`` and passed to the model as one video. Greedy decoding; one JSON per
video, written atomically, so a Colab disconnect costs at most the video in flight (C11c).

Subcommands::

    python -m core.tools.nexar_caption run --raw-root data/Nexar/raw \\
        --census outputs/v2/REPORTS/nexar_n0/census.json --revision <sha> \\
        --out-dir outputs/nexar_captions_v2 [--subset dev | --limit 3] [--quant nf4]
    python -m core.tools.nexar_caption qc --raw-root data/Nexar/raw \\
        --census outputs/v2/REPORTS/nexar_n0/census.json --out-dir outputs/nexar_captions
    python -m core.tools.nexar_caption qc-score --out-dir outputs/nexar_captions

``run --subset dev`` captions only the ``NEXAR_CAPTION_DEV_SAMPLE`` dev videos into
``dev/<prompt sha1[:8]>/`` with contact sheets: the only place the prompt may be tuned (§29.3).
``qc`` draws ``NEXAR_CAPTION_QC_SAMPLE`` videos (seed ``NEXAR_CAPTION_QC_SEED``) from the
positives outside the dev set and outside every earlier QC draw, writes one contact sheet per
video and ``qc/qc_ratings.csv`` with an empty ``score`` column (and optional ``type_ok`` /
``roles_ok``, y or n) for the user; ``qc-score`` applies CP6' (LaGoVAD Tab. B 1-5 scale, pass iff
the mean is at least ``NEXAR_CAPTION_QC_MIN_MEAN``) and prints the two shares.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from core import constants
from core.data.v2_splits import lines_sha1, load_split
from core.device import resolve_device
from core.tools.kill_switch_probe import write_json_atomic, write_text_atomic

LOGGER = logging.getLogger(__name__)

CAPTIONS_DIRNAME = "captions"
MANIFEST_FILENAME = "captions.json"
QC_DIRNAME = "qc"
QC_RATINGS_FILENAME = "qc_ratings.csv"
QC_SUMMARY_FILENAME = "qc_summary.json"
QC_SHEET_FILENAME = "qc_sheet.md"
QC_SCORE_COLUMN = "score"
QC_FLAG_COLUMNS = ("type_ok", "roles_ok")  # §29.3: optional y / n, printed, never decide
QC_FLAG_VALUES = {"y": True, "n": False}
DEV_DIRNAME = "dev"
SUBSET_ALL, SUBSET_DEV = "all", "dev"
PROMPT_TAG_CHARS = 8
# LaGoVAD supp. Tab. B, verbatim: the scale the user rates on (CP6')
QC_CRITERIA = {
    5: "The text fully matches the video content, providing an accurate, complete, and natural "
       "description of the abnormal event.",
    4: "The text is generally consistent with the video, with only minor deviations that do not "
       "affect overall understanding.",
    3: "The text captures the main content but contains notable omissions or partial "
       "inaccuracies.",
    2: "The text largely fails to match the video, capturing only limited relevant information "
       "or showing clear misinterpretations.",
    1: "The text does not match the video content at all and fails to reflect the actual video.",
}
REFUSED_REVISIONS = ("", "main", "master")
IDENTITY_KEYS = ("model", "revision", "quant", "prompt_sha1", "frames", "size",
                 "max_new_tokens", "pre_event_s", "post_event_s")
WORD = re.compile(r"[a-z]+")
CONTACT = re.compile(r"\bcontact\s*:\s*(yes|no)\b\s*[.;,:-]?\s*", re.IGNORECASE)
CLOSEST = re.compile(r"^\s*closest\s*:\s*", re.IGNORECASE)

Captioner = Callable[[np.ndarray, float], str]


# --------------------------------------------------------------------------- pure pieces
def train_positives(census: dict[str, Any]) -> list[str]:
    """CP1: the positive videos of ``nexar_train``; a census row with an issue raises."""
    rows = census["videos"]
    ids = sorted(load_split(constants.V2_SPLIT_NEXAR_TRAIN))
    missing = [v for v in ids if v not in rows]
    if missing:
        raise KeyError(f"{len(missing)} nexar_train ids are not in the census: {missing[:5]}")
    positives = [v for v in ids if rows[v]["label"] == 1]
    flagged = [v for v in positives if rows[v].get("issues")]
    if flagged:
        raise ValueError(f"{len(flagged)} positives carry census issues: {flagged[:5]}")
    return positives


def event_frame_indices(row: dict[str, Any], n: int = constants.NEXAR_CAPTION_FRAMES
                        ) -> tuple[list[int], tuple[float, float]]:
    """CP2': ``n`` distinct frame indices at the centres of ``n`` equal bins of the window
    ``[t_event - PRE, t_event + POST]``, clipped to the video."""
    event = float(row["time_of_event"])
    start = max(0.0, event - constants.NEXAR_CAPTION_PRE_EVENT_S)
    end = min(float(row["duration_s"]), event + constants.NEXAR_POST_EVENT_S)
    if end <= start:
        raise ValueError(f"empty span [{start}, {end}]")
    fps, frames = float(row["fps"]), int(row["frames_header"])
    times = [start + (end - start) * (i + 0.5) / n for i in range(n)]
    indices = [min(frames - 1, round(t * fps)) for t in times]
    if len(set(indices)) != n:
        raise ValueError(f"span [{start:.2f}, {end:.2f}] s at {fps} fps holds < {n} frames")
    return indices, (start, end)


def decode_frames(path: Path, indices: list[int],
                  size: tuple[int, int] = constants.NEXAR_CAPTION_SIZE) -> np.ndarray:
    """``(n, H, W, 3)`` uint8 RGB of ``indices`` (ascending), each resized to ``size`` = (W, H)."""
    wanted = {i: k for k, i in enumerate(indices)}
    out: list[np.ndarray | None] = [None] * len(indices)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                image = frame.to_image().resize(size, Image.Resampling.BILINEAR)
                out[wanted[i]] = np.asarray(image.convert("RGB"))
            if i >= indices[-1]:
                break
    absent = [indices[k] for k, f in enumerate(out) if f is None]
    if absent:
        raise ValueError(f"{path}: frames {absent} were not decoded")
    return np.stack([f for f in out if f is not None])


def expected_grid(n: int = constants.NEXAR_CAPTION_FRAMES,
                  size: tuple[int, int] = constants.NEXAR_CAPTION_SIZE) -> tuple[int, int, int]:
    """The processor's ``video_grid_thw`` for ``n`` frames of ``size`` (§29.3: (8, 28, 50))."""
    width, height = size
    return (n // constants.QWEN_VL_TEMPORAL_PATCH, height // constants.QWEN_VL_PATCH,
            width // constants.QWEN_VL_PATCH)


def clean_caption(text: str) -> str:
    """First non-empty line, quotes and surrounding whitespace removed; empty raises.

    A bare ``Contact: yes.`` first line (§29.3 v3) is joined with the line after it.
    """
    lines = [line.strip().strip('"').strip("'").strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("the model returned an empty caption")
    if len(lines) > 1 and CONTACT.search(lines[0]) and CONTACT.split(lines[0])[-1] == "":
        return f"{lines[0]} {lines[1]}"
    return lines[0]


def split_contact(answer: str) -> tuple[str, str]:
    """§29.3: ``(contact, caption)``; the caption is what follows ``Contact: yes|no.``, and an
    optional leading ``Closest: ...`` clause (v5) is dropped with it."""
    match = CONTACT.search(answer)
    if match is None:
        return constants.NEXAR_CAPTION_EVENT_UNKNOWN, answer.strip()
    return match.group(1).lower(), answer[match.end():].strip()


def closest_clause(answer: str) -> str:
    """§29.3 v5: the ``Closest: ...`` text before the contact clause, ``""`` if absent."""
    match = CONTACT.search(answer)
    head = answer[: match.start()] if match else ""
    return CLOSEST.sub("", head).strip().rstrip(".").strip() if CLOSEST.match(head) else ""


def contact_consistent(contact: str, kind: str) -> bool:
    """Contact yes pairs with accident / collision, no with near-collision / dangerous maneuver."""
    return kind in constants.NEXAR_CAPTION_CONTACT_TYPES.get(contact, ())


def event_type(caption: str) -> str:
    """§29.3: the type the caption opens with (``Accident: ...``); anything else is unknown."""
    head = caption.split(":", 1)[0].strip().lower() if ":" in caption else ""
    known = head in constants.NEXAR_CAPTION_EVENT_TYPES
    return head if known else constants.NEXAR_CAPTION_EVENT_UNKNOWN


def dev_type_ok(vid: str, caption: str) -> str:
    """§29.3 round 4: ``y`` / ``n`` against the frozen dev reference, ``""`` off the dev set."""
    truth = constants.NEXAR_CAPTION_DEV_TRUTH.get(vid)
    if truth is None:
        return ""
    return "y" if event_type(caption) in truth else "n"


def forbidden_terms(caption: str) -> list[str]:
    """CP6: the scene words a caption still names."""
    words = set(WORD.findall(caption.lower()))
    return sorted(words & set(constants.NEXAR_CAPTION_FORBIDDEN))


def identity(model: str, revision: str, prompt: str,
             quant: str = constants.NEXAR_CAPTION_QUANT) -> dict[str, Any]:
    """What makes two captions comparable; a record that differs is refused (CP5)."""
    if revision.strip().lower() in REFUSED_REVISIONS:
        raise ValueError(f"pin a model revision (a commit sha), not {revision!r} (C4)")
    if quant not in constants.NEXAR_CAPTION_QUANT_CHOICES:
        raise ValueError(f"quant must be one of {constants.NEXAR_CAPTION_QUANT_CHOICES}, "
                         f"got {quant!r}")
    return {
        "model": model,
        "revision": revision,
        "quant": quant,
        "prompt_sha1": hashlib.sha1(prompt.encode("utf-8"), usedforsecurity=False).hexdigest(),
        "frames": constants.NEXAR_CAPTION_FRAMES,
        "size": list(constants.NEXAR_CAPTION_SIZE),
        "max_new_tokens": constants.NEXAR_CAPTION_MAX_NEW_TOKENS,
        "pre_event_s": constants.NEXAR_CAPTION_PRE_EVENT_S,
        "post_event_s": constants.NEXAR_POST_EVENT_S,
    }


def load_record(path: Path, ident: dict[str, Any]) -> dict[str, Any] | None:
    """A finished record, ``None`` if absent; one made under another identity raises."""
    if not path.exists():
        return None
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    differs = {k: (record.get(k), ident[k]) for k in IDENTITY_KEYS if record.get(k) != ident[k]}
    if differs:
        raise ValueError(f"{path} was made under another setting {differs}; refusing to mix")
    return record


# --------------------------------------------------------------------------- model
class QwenCaptioner:
    """Qwen2.5-VL-Instruct, bf16 or 4-bit NF4 (bf16 compute), greedy; ``(frames, fps) -> text``."""

    def __init__(self, model: str, revision: str, prompt: str, device: torch.device,
                 quant: str = constants.NEXAR_CAPTION_QUANT) -> None:
        self.device = device
        self.prompt = prompt
        self.processor = AutoProcessor.from_pretrained(model, revision=revision)
        quant_config = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16,
                                           bnb_4bit_use_double_quant=True)
                        if quant == constants.NEXAR_CAPTION_QUANT_NF4 else None)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, revision=revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            use_safetensors=True, quantization_config=quant_config, device_map={"": str(device)},
        ).eval()
        messages = [{"role": "user", "content": [{"type": "video"},
                                                 {"type": "text", "text": prompt}]}]
        self.text = self.processor.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True)
        if self.text.count(self.processor.video_token) != 1:
            raise ValueError("the chat template did not place exactly one video token")

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, fps: float) -> str:
        inputs = self.processor(text=[self.text], videos=[frames], fps=fps, padding=True,
                                return_tensors="pt").to(self.device)
        grid = tuple(int(x) for x in inputs["video_grid_thw"][0])
        if grid != expected_grid(len(frames), (frames.shape[2], frames.shape[1])):
            raise ValueError(f"video grid {grid} != {expected_grid()}: the frames were resized")
        out = self.model.generate(**inputs, max_new_tokens=constants.NEXAR_CAPTION_MAX_NEW_TOKENS,
                                  do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        text: str = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)[0]
        return text


# --------------------------------------------------------------------------- run
def caption_videos(raw_root: Path, census: dict[str, Any], out_dir: Path,
                   ident: dict[str, Any], captioner: Captioner,
                   ids: list[str] | None = None) -> dict[str, Any]:
    """Caption ``ids`` (default every train positive); finished videos are skipped."""
    rows = census["videos"]
    todo = train_positives(census) if ids is None else ids
    records: dict[str, dict[str, Any]] = {}
    t0, made = time.time(), 0
    for n, vid in enumerate(todo, 1):
        path = out_dir / CAPTIONS_DIRNAME / f"{vid}.json"
        record = load_record(path, ident)
        if record is None:
            indices, (start, end) = event_frame_indices(rows[vid])
            frames = decode_frames(raw_root / rows[vid]["path"], indices)
            answer = clean_caption(captioner(frames, len(indices) / (end - start)))
            contact, caption = split_contact(answer)
            kind = event_type(caption)
            record = {**ident, "video_id": vid, "caption": caption, "answer": answer,
                      "closest": closest_clause(answer), "contact": contact, "event_type": kind,
                      "contact_consistent": contact_consistent(contact, kind),
                      "frame_indices": indices, "span_s": [start, end],
                      "words": len(caption.split()), "forbidden": forbidden_terms(caption)}
            write_json_atomic(path, record)
            made += 1
            LOGGER.info("%d/%d %s (%.1f s/video): %s", n, len(todo), vid,
                        (time.time() - t0) / made, caption)
        records[vid] = record
    return records


def write_manifest(out_dir: Path, census: dict[str, Any], ident: dict[str, Any],
                   records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """CP5: identity + every caption, bound to the train split's ids."""
    positives = train_positives(census)
    if sorted(records) != positives:
        raise ValueError(f"{len(records)} captions for {len(positives)} train positives")
    manifest = {
        **ident, "prompt": constants.NEXAR_CAPTION_PROMPT, "split": constants.V2_SPLIT_NEXAR_TRAIN,
        "train_ids_sha1": lines_sha1(sorted(load_split(constants.V2_SPLIT_NEXAR_TRAIN))),
        "videos": len(records), "captions": {v: r["caption"] for v, r in sorted(records.items())},
        "summary": summarize(records),
    }
    write_json_atomic(out_dir / MANIFEST_FILENAME, manifest)
    return manifest


def summarize(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """CP6 prints: forbidden-term share, word-count range, distinct captions."""
    words = [r["words"] for r in records.values()]
    return {
        "videos": len(records),
        "forbidden_share": float(np.mean([bool(r["forbidden"]) for r in records.values()])),
        "forbidden_terms": sorted({t for r in records.values() for t in r["forbidden"]}),
        "words_min": min(words), "words_median": float(np.median(words)), "words_max": max(words),
        "over_max_words": sum(w > constants.NEXAR_CAPTION_MAX_WORDS for w in words),
        "distinct": len({r["caption"] for r in records.values()}),
        "event_types": {t: sum(event_type(r["caption"]) == t for r in records.values())
                        for t in (*constants.NEXAR_CAPTION_EVENT_TYPES,
                                  constants.NEXAR_CAPTION_EVENT_UNKNOWN)},
        "contact": {c: sum(r.get("contact") == c for r in records.values())
                    for c in (*constants.NEXAR_CAPTION_CONTACT_TYPES,
                              constants.NEXAR_CAPTION_EVENT_UNKNOWN)},
        "contact_inconsistent": sum(not r.get("contact_consistent", True)
                                    for r in records.values()),
    }


# --------------------------------------------------------------------------- QC
def _draw(ids: list[str], seed: int, n: int) -> list[str]:
    rng = random.Random(seed)  # nosec B311 - a fixed, documented draw
    return sorted(rng.sample(sorted(ids), n))


def dev_sample(ids: list[str]) -> list[str]:
    """§29.3 CP4'': the videos the prompt may be tuned on; never rated in QC."""
    return _draw(ids, constants.NEXAR_CAPTION_DEV_SEED, constants.NEXAR_CAPTION_DEV_SAMPLE)


def qc_sample(ids: list[str]) -> list[str]:
    """CP6'': the seeded draw the user rates, outside the dev set and every earlier QC draw."""
    seen = set(dev_sample(ids))
    for seed in constants.NEXAR_CAPTION_QC_SEEN_SEEDS:
        seen |= set(_draw(ids, seed, constants.NEXAR_CAPTION_QC_SAMPLE))
    return _draw([v for v in ids if v not in seen], constants.NEXAR_CAPTION_QC_SEED,
                 constants.NEXAR_CAPTION_QC_SAMPLE)


def prompt_tag(prompt: str) -> str:
    """The dev folder of one prompt version: its sha1 prefix."""
    digest = hashlib.sha1(prompt.encode("utf-8"), usedforsecurity=False).hexdigest()
    return digest[:PROMPT_TAG_CHARS]


def contact_sheet(frames: np.ndarray, columns: int = 4) -> Image.Image:
    """Frames in reading order on a ``columns``-wide grid."""
    n, height, width, _ = frames.shape
    rows = -(-n // columns)
    sheet = Image.new("RGB", (columns * width, rows * height))
    for k, frame in enumerate(frames):
        sheet.paste(Image.fromarray(frame), ((k % columns) * width, (k // columns) * height))
    return sheet


def write_sheet(raw_root: Path, census: dict[str, Any], captions_dir: Path, sheet_dir: Path,
                sample: list[str], title: str) -> None:
    """Contact sheets + ratings csv + a markdown sheet of ``sample`` for the user."""
    rows = census["videos"]
    sheet_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "",
             f"Give each caption an integer `{QC_SCORE_COLUMN}` from "
             f"{constants.NEXAR_CAPTION_QC_SCORE_MIN} to {constants.NEXAR_CAPTION_QC_SCORE_MAX} in "
             f"`{QC_RATINGS_FILENAME}` (LaGoVAD Tab. B). Pass iff the mean over the "
             f"{len(sample)} rows is >= {constants.NEXAR_CAPTION_QC_MIN_MEAN}. Optional: "
             f"{' / '.join(f'`{c}`' for c in QC_FLAG_COLUMNS)} = y or n (event type right? roles "
             "right?), printed only.", "",
             "| score | criteria |", "|---|---|",
             *(f"| {s} | {QC_CRITERIA[s]} |" for s in sorted(QC_CRITERIA, reverse=True)), "",
             "| # | video | span (s) | contact | type | caption |", "|---|---|---|---|---|---|"]
    with (sheet_dir / QC_RATINGS_FILENAME).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "video_id", "event_type", "caption", QC_SCORE_COLUMN,
                         *QC_FLAG_COLUMNS])
        for rank, vid in enumerate(sample, 1):
            record = json.loads((captions_dir / f"{vid}.json").read_text(encoding="utf-8"))
            frames = decode_frames(raw_root / rows[vid]["path"], record["frame_indices"])
            contact_sheet(frames).save(sheet_dir / f"{rank:02d}_{vid}.png")
            kind = event_type(record["caption"])
            flags = [dev_type_ok(vid, record["caption"]), ""]  # QC_FLAG_COLUMNS order
            writer.writerow([rank, vid, kind, record["caption"], "", *flags])
            start, end = record["span_s"]
            lines.append(f"| {rank} | {vid} | {start:.2f}-{end:.2f} | "
                         f"{record.get('contact', '')} | {kind} | {record['caption']} |")
    write_text_atomic(sheet_dir / QC_SHEET_FILENAME, "\n".join(lines) + "\n")


def write_qc(raw_root: Path, census: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Contact sheets + ratings template for the user; refuses an incomplete caption set."""
    manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    sample = qc_sample(list(manifest["captions"]))
    write_sheet(raw_root, census, out_dir / CAPTIONS_DIRNAME, out_dir / QC_DIRNAME, sample,
                "Nexar caption QC (§29.3 CP6'')")
    return {"sample": sample, "summary": manifest["summary"]}


def score_qc(out_dir: Path) -> dict[str, Any]:
    """CP6': pass iff the mean Tab. B score is >= ``NEXAR_CAPTION_QC_MIN_MEAN``; blanks raise."""
    with (out_dir / QC_DIRNAME / QC_RATINGS_FILENAME).open(encoding="utf-8", newline="") as fh:
        marks = {r["video_id"]: r[QC_SCORE_COLUMN].strip() for r in csv.DictReader(fh)}
    allowed = {str(s) for s in range(constants.NEXAR_CAPTION_QC_SCORE_MIN,
                                     constants.NEXAR_CAPTION_QC_SCORE_MAX + 1)}
    bad = {v: m for v, m in marks.items() if m not in allowed}
    if bad:
        raise ValueError(f"{len(bad)} rows are not scored {constants.NEXAR_CAPTION_QC_SCORE_MIN}-"
                         f"{constants.NEXAR_CAPTION_QC_SCORE_MAX}: {sorted(bad)[:5]}")
    scores = {v: int(m) for v, m in marks.items()}
    mean = float(np.mean(list(scores.values())))
    with (out_dir / QC_DIRNAME / QC_RATINGS_FILENAME).open(encoding="utf-8", newline="") as fh:
        flags = list(csv.DictReader(fh))
    shares: dict[str, float | None] = {}
    for column in QC_FLAG_COLUMNS:
        marked = [QC_FLAG_VALUES[r[column].strip().lower()] for r in flags
                  if r.get(column, "").strip().lower() in QC_FLAG_VALUES]
        shares[f"{column}_share"] = float(np.mean(marked)) if marked else None
    result = {"rated": len(scores), "mean": mean, "pass_bar": constants.NEXAR_CAPTION_QC_MIN_MEAN,
              "verdict": "pass" if mean >= constants.NEXAR_CAPTION_QC_MIN_MEAN else "fail",
              "histogram": {s: sum(x == s for x in scores.values()) for s in sorted(QC_CRITERIA)},
              "below_bar": sorted(v for v, s in scores.items()
                                  if s < constants.NEXAR_CAPTION_QC_MIN_MEAN),
              **shares}
    write_json_atomic(out_dir / QC_DIRNAME / QC_SUMMARY_FILENAME, result)
    return result


# --------------------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "qc"):
        p = sub.add_parser(name)
        p.add_argument("--raw-root", type=Path, required=True, help="Nexar mirror (holds train/)")
        p.add_argument("--census", type=Path, required=True, help="N0 census.json")
        p.add_argument("--out-dir", type=Path, required=True)
    run = sub.choices["run"]
    run.add_argument("--revision", required=True, help="model commit sha (never main; C4)")
    run.add_argument("--model", default=constants.NEXAR_CAPTION_MODEL)
    run.add_argument("--limit", type=int, default=None, help="first N videos only (a smoke run)")
    run.add_argument("--subset", choices=(SUBSET_ALL, SUBSET_DEV), default=SUBSET_ALL,
                     help="dev = the prompt-tuning videos only, into dev/<prompt sha1[:8]>/")
    run.add_argument("--quant", choices=constants.NEXAR_CAPTION_QUANT_CHOICES,
                     default=constants.NEXAR_CAPTION_QUANT)
    run.add_argument("--device", default="auto")
    score = sub.add_parser("qc-score")
    score.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "qc-score":
        LOGGER.info("QC: %s", score_qc(args.out_dir))
        return
    census = json.loads(args.census.read_text(encoding="utf-8"))
    if args.command == "qc":
        LOGGER.info("QC sheets written: %s", write_qc(args.raw_root, census, args.out_dir))
        return
    prompt = constants.NEXAR_CAPTION_PROMPT
    ident = identity(args.model, args.revision, prompt, args.quant)
    captioner = QwenCaptioner(args.model, args.revision, prompt, resolve_device(args.device),
                              args.quant)
    ids = train_positives(census)
    if args.subset == SUBSET_DEV:
        dev_dir = args.out_dir / DEV_DIRNAME / prompt_tag(prompt)
        write_text_atomic(dev_dir / "prompt.txt", prompt + "\n")
        dev = dev_sample(ids)
        if sorted(constants.NEXAR_CAPTION_DEV_TRUTH) != dev:
            raise ValueError(f"dev reference covers {sorted(constants.NEXAR_CAPTION_DEV_TRUTH)}, "
                             f"the dev draw is {dev}")
        records = caption_videos(args.raw_root, census, dev_dir, ident, captioner, dev)
        type_ok = sum(dev_type_ok(v, r["caption"]) == "y" for v, r in records.items())
        write_json_atomic(dev_dir / "dev_summary.json", {
            "prompt_tag": prompt_tag(prompt), "type_ok": type_ok, "videos": len(records),
            "freeze_type_ok": constants.NEXAR_CAPTION_DEV_FREEZE_TYPE_OK,
            "freeze_mean": constants.NEXAR_CAPTION_DEV_FREEZE_MEAN, **summarize(records)})
        LOGGER.info("dev type_ok %d/%d (freeze needs >= %d and mean score >= %.1f)", type_ok,
                    len(records), constants.NEXAR_CAPTION_DEV_FREEZE_TYPE_OK,
                    constants.NEXAR_CAPTION_DEV_FREEZE_MEAN)
        write_sheet(args.raw_root, census, dev_dir / CAPTIONS_DIRNAME, dev_dir, dev,
                    f"Nexar caption dev set, prompt {prompt_tag(prompt)} (§29.3 CP4'')")
        LOGGER.info("dev sheet: %s", dev_dir / QC_SHEET_FILENAME)
        return
    records = caption_videos(args.raw_root, census, args.out_dir, ident, captioner,
                             ids[: args.limit] if args.limit else None)
    if args.limit is None:
        manifest = write_manifest(args.out_dir, census, ident, records)
        LOGGER.info("manifest: %s", manifest["summary"])


if __name__ == "__main__":
    main()
