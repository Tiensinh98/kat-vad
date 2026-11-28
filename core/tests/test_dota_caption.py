"""``core.tools.dota_caption`` (``KIM_RUN.md`` §63) with a fake captioner: no model download."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from core import constants
from core.tools import dota_caption as dc
from core.tools import nexar_caption as nc

FRAMES = 60
SIZE = (64, 36)


def _ann(start: int = 30, end: int = 45, frames: int = FRAMES, kind: str = "turning",
         ego: bool = True) -> dict[str, Any]:
    return {"num_frames": frames, "anomaly_start": start, "anomaly_end": end,
            "accident_name": kind, "ego_involve": ego}


def _archive(path: Path, frames: int = FRAMES) -> None:
    """Frame ``d`` is a flat JPEG of value ``d``: the decoded mean says which frame it is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for d in range(frames):
            buf = io.BytesIO()
            Image.fromarray(np.full((36, 64, 3), d, dtype=np.uint8)).save(buf, "JPEG", quality=95)
            zf.writestr(f"{d:06d}.jpg", buf.getvalue())


class FakeCaptioner:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def caption(self, frames: np.ndarray, fps: float, prompt: str) -> str:
        width, height = constants.DOTA_CAPTION_SIZE
        assert frames.shape == (constants.DOTA_CAPTION_FRAMES, height, width, 3)
        assert fps > 0
        self.prompts.append(prompt)
        return f'"A car turns across the ego vehicle\'s path ({len(self.prompts)})."\n'


# --------------------------------------------------------------------------- window
def test_window_spans_pre_and_post_onset() -> None:
    indices, (start, end) = dc.caption_window(_ann(start=20, end=58))
    pre = round(constants.DOTA_CAPTION_PRE_ONSET_S * constants.DOTA_FPS)
    post = round(constants.DOTA_CAPTION_POST_ONSET_S * constants.DOTA_FPS)
    assert (start, end) == (20 - pre, 20 + post)
    assert len(set(indices)) == constants.DOTA_CAPTION_FRAMES == len(indices)
    assert indices == sorted(indices) and start <= indices[0] and indices[-1] < end


def test_window_closes_at_anomaly_end_when_sooner() -> None:
    _, (start, end) = dc.caption_window(_ann(start=30, end=40))
    pre = round(constants.DOTA_CAPTION_PRE_ONSET_S * constants.DOTA_FPS)
    assert end == 40 and start == 30 - pre


def test_short_window_is_widened_inside_the_clip() -> None:
    n = constants.DOTA_CAPTION_FRAMES
    indices, (start, end) = dc.caption_window(_ann(start=0, end=1))
    assert (start, end) == (0, n) and indices == list(range(n))
    indices, (start, end) = dc.caption_window(_ann(start=FRAMES - 1, end=FRAMES))
    assert end == FRAMES and len(set(indices)) == n


def test_clip_shorter_than_the_frame_count_raises() -> None:
    with pytest.raises(ValueError, match="frames"):
        dc.caption_window(_ann(start=2, end=5, frames=constants.DOTA_CAPTION_FRAMES - 1))


# --------------------------------------------------------------------------- prompt / sample
def test_prompt_carries_type_and_ego() -> None:
    ego = dc.prompt_for(_ann(kind="oncoming", ego=True), "v1")
    other = dc.prompt_for(_ann(kind="oncoming", ego=False), "v1")
    assert constants.DOTA_CAPTION_TYPE_TEXT["oncoming"] in ego
    assert constants.DOTA_CAPTION_EGO_TEXT[True] in ego
    assert constants.DOTA_CAPTION_EGO_TEXT[False] in other and "{" not in other


def test_v1_keeps_the_pilots_identity_and_folder() -> None:
    """§63.1's 32B records (tag 7eb9f067) must still load: v1 hashes exactly as before."""
    assert dc.pilot_dirname("Qwen/Qwen2.5-VL-32B-Instruct", "v1") == \
        "7eb9f067_Qwen2.5-VL-32B-Instruct_16f"
    v1 = dc.identity(constants.NEXAR_CAPTION_MODEL, "abc", constants.NEXAR_CAPTION_QUANT, "v1")
    v2 = dc.identity(constants.NEXAR_CAPTION_MODEL, "abc", constants.NEXAR_CAPTION_QUANT, "v2")
    assert v1["prompt_sha1"] != v2["prompt_sha1"]
    assert dc.pilot_dirname("m/x", "v1") != dc.pilot_dirname("m/x", "v2")


def test_v2_has_one_template_per_ego_flag() -> None:
    ego = dc.prompt_for(_ann(kind="lateral", ego=True), "v2")
    other = dc.prompt_for(_ann(kind="lateral", ego=False), "v2")
    for text in (ego, other):
        assert constants.DOTA_CAPTION_TYPE_TEXT["lateral"] in text and "{" not in text
        assert "do not add that the vehicles stop" in text and "exact class" in text
    assert "NOT involved" in other and "moment of impact" in other
    assert "NOT involved" not in ego
    with pytest.raises(ValueError, match="prompt version"):
        dc.prompt_for(_ann(), "v9")


def test_every_type_has_a_sentence() -> None:
    kinds = ("turning", "lateral", "moving_ahead_or_waiting", "oncoming", "leave_to_left",
             "leave_to_right", "start_stop_or_stationary", "pedestrian", "obstacle", "unknown")
    assert set(constants.DOTA_CAPTION_TYPE_TEXT) == set(kinds)


def test_pilot_sample_covers_every_type_and_is_seeded() -> None:
    kinds = sorted(constants.DOTA_CAPTION_TYPE_TEXT)
    anns = {f"v{i:03d}": _ann(kind=kinds[i % len(kinds)]) for i in range(200)}
    first, again = dc.pilot_sample(anns), dc.pilot_sample(anns)
    assert first == again and len(first) == constants.DOTA_CAPTION_PILOT_SAMPLE
    assert len(set(first)) == len(first)
    assert {anns[v]["accident_name"] for v in first} == set(kinds)


def test_identity_refuses_main() -> None:
    with pytest.raises(ValueError, match="revision"):
        dc.identity(constants.NEXAR_CAPTION_MODEL, "main", constants.NEXAR_CAPTION_QUANT)


def test_load_annotations_rejects_an_unknown_type(tmp_path: Path) -> None:
    path = tmp_path / "ann.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("annotations/a.json", json.dumps({**_ann(kind="flying"), "labels": []}))
    with pytest.raises(ValueError, match="flying"):
        dc.load_annotations(path, ["a"])


# --------------------------------------------------------------------------- run + QC
def test_decode_reads_the_requested_frames(tmp_path: Path) -> None:
    _archive(tmp_path / "a.zip")
    frames = dc.decode_frames(tmp_path / "a.zip", [3, 40], FRAMES, SIZE)
    assert frames.shape == (2, SIZE[1], SIZE[0], 3)
    assert abs(frames[0].mean() - 3) < 2 and abs(frames[1].mean() - 40) < 2
    with pytest.raises(ValueError, match="annotation"):
        dc.decode_frames(tmp_path / "a.zip", [0], FRAMES + 1, SIZE)


def test_caption_sheet_and_score_round_trip(tmp_path: Path) -> None:
    anns = {"a": _ann(ego=True), "b": _ann(kind="pedestrian", ego=False)}
    for vid in anns:
        _archive(tmp_path / "frames" / f"{vid}.zip")
    ident = dc.identity(constants.NEXAR_CAPTION_MODEL, "abc123", constants.NEXAR_CAPTION_QUANT)
    out = tmp_path / "pilot"
    fake = FakeCaptioner()
    records = dc.caption_clips(tmp_path / "frames", anns, sorted(anns), out, ident, fake)
    assert len(fake.prompts) == 2 and records["a"]["caption"].startswith("A car turns")
    assert not records["b"]["ego_involve"]
    again = dc.caption_clips(tmp_path / "frames", anns, sorted(anns), out, ident, fake)
    assert len(fake.prompts) == 2 and again == records  # resume: nothing re-captioned
    other = {**ident, "revision": "zzz"}
    with pytest.raises(ValueError, match="another setting"):
        dc.caption_clips(tmp_path / "frames", anns, sorted(anns), out, other, fake)
    summary = dc.summarize(records)
    assert summary["non_ego_clips"] == 1 and summary["non_ego_naming_ego"] == 1

    dc.write_sheet(tmp_path / "frames", records, out / nc.QC_DIRNAME, anns, "test")
    ratings = out / nc.QC_DIRNAME / nc.QC_RATINGS_FILENAME
    assert len(list((out / nc.QC_DIRNAME).glob("*.png"))) == 2
    with ratings.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row, score in zip(rows, ("5", "3"), strict=True):
        row["score"], row["type_ok"] = score, "y"
    with ratings.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = nc.score_qc(out)
    assert result["mean"] == 4.0 and result["verdict"] == "pass" and result["type_ok_share"] == 1.0

    dc.write_sheet(tmp_path / "frames", records, out / nc.QC_DIRNAME, anns, "again")
    with ratings.open(encoding="utf-8", newline="") as fh:  # a re-run never blanks the scores
        assert [r["score"] for r in csv.DictReader(fh)] == ["5", "3"]

    v1 = dc.identity(constants.NEXAR_CAPTION_MODEL, "abc123", constants.NEXAR_CAPTION_QUANT, "v1")
    out_v1 = tmp_path / "pilot_v1"
    dc.caption_clips(tmp_path / "frames", anns, sorted(anns), out_v1, v1, fake, "v1")
    assert constants.DOTA_CAPTION_EGO_TEXT[False] in fake.prompts[-1]
    dc.write_sheet(tmp_path / "frames", dc.caption_clips(
        tmp_path / "frames", anns, sorted(anns), out_v1, v1, fake, "v1"),
        out_v1 / nc.QC_DIRNAME, anns, "v1")
    text = dc.compare({"v1": out_v1, "v2": out}, tmp_path / "cmp.md")
    assert "| v1 | v2 |" in text and "**[5]**" in text and text.count("\n| ") == 3

