"""Tests for ``core.tools.dota_eda`` (KIM_RUN.md §57): synthetic clips, no data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import dota_eda as eda
from core.tools import dota_train_build as dtb
from core.tools.dota_train_match import DotaClip

EXTRA = {"ego": True, "accident": "turning"}


def _clip(frames: int, start: int, end: int, dota_id: str) -> DotaClip:
    return DotaClip(dota_id, frames, start, end, False)


def _clips() -> list[DotaClip]:
    # 30 frames at s3 = 10 rows; long clips so every one gives windows of 12 rows
    return [_clip(120, 45, 75, "a_b_000001"), _clip(90, 30, 90, "c_000002"),
            _clip(150, 3, 60, "d_000003")]


def test_youtube_id_keeps_underscores_inside_the_id() -> None:
    assert eda.youtube_id("r6_ZhT7rmhM_004291") == "r6_ZhT7rmhM"


def test_first_full_row_is_five_at_stride_3() -> None:
    assert eda.first_full_row() == 5
    clamp = constants.DOTA_VIDEOMAE_FRAME_STEP * (constants.VIDEOMAE_CLIP_FRAMES - 1)
    assert 5 * constants.DOTA_TRAIN_STRIDE >= clamp > 4 * constants.DOTA_TRAIN_STRIDE


def test_clip_row_span_geometry() -> None:
    r = eda.clip_row(_clip(90, 30, 90, "c_000002"), EXTRA)
    assert (r["rows"], r["span_start"], r["span_end"], r["post_rows"]) == (30, 10, 30, 0)
    assert r["start_rel"] == pytest.approx(1 / 3) and r["end_rel"] == 1.0
    assert r["labels"].sum() == r["span_rows"] == 20


def test_time_profile_counts_rows_from_both_ends() -> None:
    rows = [eda.clip_row(_clip(90, 30, 90, "c_000002"), EXTRA)]  # span runs to the end
    tp = eda.time_profile(rows)
    assert tp["by_rows_from_end"][0] == 1.0 and tp["by_rows_from_start"][0] == 0.0
    assert tp["by_relative_bin"][-1] == 1.0 and tp["by_relative_bin"][0] == 0.0
    assert tp["row_share"] == pytest.approx(20 / 30)


def test_position_rulers_late_is_one_when_the_span_ends_the_clip() -> None:
    rows = [eda.clip_row(_clip(90, 30, 90, "c_000002"), EXTRA)]
    pr = eda.position_rulers(rows)
    assert pr["macro_late"] == 1.0 and pr["micro_late"] == 1.0
    assert pr["two_class_clips"] == 1


def test_window_rows_match_the_corpus_and_gate(tmp_path: Path) -> None:
    clips = _clips()
    extras = {c.dota_id: EXTRA for c in clips}
    rows = eda.window_rows(clips, extras)
    dtb.build(tmp_path, clips)
    g = eda.gate(rows, tmp_path)
    assert g["equal"] and g["windows_checked"] == len(rows)
    assert all(r["label"] == int(r["positive"] >= constants.WINDOW_MIN_POSITIVE) for r in rows)
    padded = [r for r in rows if r["start"] == 0]
    assert padded and all(r["padded_rows"] == eda.first_full_row() for r in padded)


def test_gate_refuses_a_changed_corpus(tmp_path: Path) -> None:
    clips = _clips()
    rows = eda.window_rows(clips, {c.dota_id: EXTRA for c in clips})
    dtb.build(tmp_path, clips)
    path = tmp_path / constants.LABELS_TRAIN_FILENAME
    labels = json.loads(path.read_text())
    first = sorted(labels)[0]
    labels[first] = 1 - labels[first]
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="differ"):
        eda.gate(rows, tmp_path)


def test_window_summary_and_markdown() -> None:
    clips = _clips()
    rows = eda.window_rows(clips, {c.dota_id: EXTRA for c in clips})
    w = eda.window_summary(rows)
    c = w["counts"]
    assert c["positive"] + c["normal"] == c["windows"] == len(rows)
    assert w["mil_k"] == {"mil": 2, "mul_mil": 1, "bottomk_25_arms": 2}
    assert set(w["padding"]) >= {"first_full_row", "share_normal_padded",
                                 "share_positive_padded"}
    clip_rows = [eda.clip_row(x, EXTRA) for x in clips]
    report = {
        "stride": 3, "window": {"length": 12, "hop": 6}, "gate": None,
        "clips": eda.clip_summary(clip_rows), "time_profile": eda.time_profile(clip_rows),
        "position_rulers": eda.position_rulers(clip_rows), "windows": w,
    }
    text = eda.render_markdown(report)
    assert "## 3. Windows" in text and "MIL k on a 12-row window" in text
    assert "dev" not in text
    assert np.isfinite(report["position_rulers"]["macro_hump"])
