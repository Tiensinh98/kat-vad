"""Tests for the ACCIDENT tools (``KIM_RUN.md`` §39): CFR resampling, split, corpus, TAD dedup.

The failures that matter are silent: a CFR grid that shifts the impact by a frame or reads a
future frame, a source split across train and val, a window that reaches past
``t_impact + 2 s`` (unlabeled aftermath read as normal), an excluded clip that still trains, and
a dedup threshold that flags everything (or nothing) without saying so.
"""

from __future__ import annotations

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from core import constants
from core.data.dada import sampled_frame_labels
from core.data.definitions import dataset_abbr
from core.tools import accident_build, accident_dedup, accident_extract
from core.tools.accident_build import windowable
from core.tools.accident_extract import cfr_index, decode_resized_with_times

FPS = constants.ACCIDENT_FPS
TYPES = ("single", "t-bone", "rear-end")


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(clip_id: str, kind: str, impact: float, duration: float) -> dict[str, str]:
    return {"path": f"real_videos/{clip_id}.mp4", "type": kind, "accident_time": str(impact),
            "duration": str(duration), "scene_layout": "intersection", "weather": "normal",
            "day_time": "day", "quality": "Good", "region": "World"}


def _corpus_inputs(tmp_path: Path, n_sources: int = 30) -> tuple[Path, Path, dict[str, int]]:
    """Metadata + a fake CLIP s1 cache (rows = 30 fps CFR frames) + frozen splits."""
    rows, frames = [], {}
    clip_dir = tmp_path / "clip_s1"
    clip_dir.mkdir()
    for s in range(n_sources):
        for k in range(2 if s % 5 == 0 else 1):
            clip_id = f"src{s:03d}_{k:02d}"
            impact, duration = 8.0 + (s % 4), 25.0
            rows.append(_row(clip_id, TYPES[s % len(TYPES)], impact, duration))
            frames[clip_id] = int(duration * FPS)
            np.save(clip_dir / f"{clip_id}.npy", np.zeros((frames[clip_id], 4), np.float32))
    metadata = _write_metadata(tmp_path / "metadata-real.csv", rows)
    accident_build.write_splits(metadata, tmp_path / "splits")
    return metadata, clip_dir, frames


# ---------------------------------------------------------------------------
# CFR resampling (A2)
# ---------------------------------------------------------------------------
def test_cfr_index_repeats_frames_of_a_15fps_clip() -> None:
    times = np.arange(10) / 15.0
    index = cfr_index(times)
    assert len(index) == 19  # 0 .. 0.6 s at 30 fps
    assert index.tolist() == [k // 2 for k in range(19)]


def test_cfr_index_never_reads_a_future_frame_and_ignores_start_offset() -> None:
    rng = np.random.default_rng(0)
    times = 3.0 + np.cumsum(rng.uniform(0.01, 0.2, size=200))
    index = cfr_index(times)
    rel = times - times[0]
    grid = np.arange(len(index)) / FPS
    assert np.all(rel[index] <= grid + constants.ACCIDENT_CFR_TIME_EPS)
    nxt = np.minimum(index + 1, len(times) - 1)
    later = index + 1 < len(times)
    assert np.all(rel[nxt][later] > grid[later])  # the last frame at or before each grid time


def _write_mp4(path: Path, native: int, size: int = 32) -> Path:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=15)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 15)
        for i in range(native):
            img = np.full((size, size, 3), 20 * (i % 12), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_decoding_to_the_cut_gives_the_full_decode_cut(tmp_path: Path) -> None:
    path = _write_mp4(tmp_path / "x_00.mp4", 40)                  # 15 fps, 2.6 s
    row = _row("x_00", "single", impact=0.5, duration=40 / 15)    # cut at 2.5 s = CFR 75
    full_decoded, full_times = decode_resized_with_times(path)
    full = cfr_index(full_times)[: accident_build.cut_frames(row, len(cfr_index(full_times)))]
    decoded, times = decode_resized_with_times(path, 0.5 + constants.ACCIDENT_POST_IMPACT_S)
    index = cfr_index(times)
    index = index[: accident_build.cut_frames(row, len(index))]
    assert len(times) < len(full_times) and len(index) == len(full) == 76
    assert np.array_equal(index, full)
    assert torch.equal(decoded[index], full_decoded[full])


def test_windowable_skips_only_clips_too_short_for_a_window() -> None:
    need = constants.NEXAR_WINDOW_LENGTH * constants.FRAME_STRIDE / FPS     # 5.33 s of cut
    assert windowable(_row("a_00", "single", impact=need, duration=60.0))
    assert not windowable(_row("b_00", "single", impact=1.0, duration=60.0))   # cut at 3 s
    assert not windowable(_row("c_00", "single", impact=1.0, duration=3.0))
    margin = constants.ACCIDENT_SKIP_MARGIN_ROWS * constants.FRAME_STRIDE / FPS
    assert windowable(_row("d_00", "single", impact=need - margin - 2.0, duration=60.0))


class _FakeClip:
    def __call__(self, pixel_values: torch.Tensor) -> object:
        return argparse.Namespace(image_embeds=pixel_values.flatten(1)[:, :512])


class _FakeVideo(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0, :, 0, 0]


def test_extract_keeps_the_cut_runs_motion_at_s8_and_skips_short_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos = tmp_path / "real_videos"
    videos.mkdir()
    _write_mp4(videos / "long_00.mp4", 120)                       # 8 s at 15 fps
    _write_mp4(videos / "short_00.mp4", 60)
    meta = _write_metadata(tmp_path / "m.csv", [_row("long_00", "single", 4.0, 8.0),
                                                 _row("short_00", "single", 0.5, 4.0)])
    ids = tmp_path / "ids.txt"
    ids.write_text("long_00\nshort_00\n")
    monkeypatch.setattr(accident_extract, "load_pretrained_encoder", lambda _d: _FakeClip())
    monkeypatch.setattr(accident_extract, "load_pretrained", lambda *_a: _FakeVideo())
    argv = ["--videos-dir", str(videos), "--metadata", str(meta), "--ids-file", str(ids),
            "--clip-out", str(tmp_path / "clip"), "--video-out", str(tmp_path / "video"),
            "--device", "cpu", "--video-batch", "8"]
    report = accident_extract.extract(accident_extract.build_arg_parser().parse_args(argv))
    assert report["skipped_short"] == ["short_00"] and report["summary"]["extracted"] == 1
    cut = int((4.0 + constants.ACCIDENT_POST_IMPACT_S) * FPS) + 1  # 181 CFR frames, not 211
    x, u = np.load(tmp_path / "clip" / "long_00.npy"), np.load(tmp_path / "video" / "long_00.npy")
    assert x.shape == (cut, 512) and len(u) == len(x[:: constants.FRAME_STRIDE]) == 23
    assert not (tmp_path / "clip" / "short_00.npy").exists()
    manifest = json.loads((tmp_path / "video" / constants.VIDEO_MANIFEST_FILENAME).read_text())
    assert manifest["stride"] == constants.FRAME_STRIDE
    again = accident_extract.extract(accident_extract.build_arg_parser().parse_args(argv))
    assert again["summary"]["extracted"] == 0


def test_a_15fps_mp4_decodes_to_twice_its_frames_in_order(tmp_path: Path) -> None:
    path, native, size = tmp_path / "x_00.mp4", 12, 32
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=15)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 15)
        for i in range(native):
            img = np.full((size, size, 3), 20 * i, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    decoded, times = decode_resized_with_times(path)
    index = cfr_index(times)
    assert decoded.shape[0] == native and len(index) == 2 * native - 1
    brightness = decoded[index].mean(dim=(1, 2, 3)).numpy()
    assert np.all(np.diff(brightness) >= -1e-3)  # never steps back to an earlier frame


def test_cfr_index_refuses_decreasing_times() -> None:
    with pytest.raises(ValueError, match="decrease"):
        cfr_index(np.array([0.0, 0.1, 0.05]))


# ---------------------------------------------------------------------------
# split (S)
# ---------------------------------------------------------------------------
def test_split_keeps_sources_whole_and_is_frozen(tmp_path: Path) -> None:
    metadata, _, _ = _corpus_inputs(tmp_path)
    train = accident_build.load_accident_split(constants.ACCIDENT_SPLIT_TRAIN, tmp_path / "splits")
    val = accident_build.load_accident_split(constants.ACCIDENT_SPLIT_VAL, tmp_path / "splits")
    assert val and not {accident_build.source_of(v) for v in train} & {
        accident_build.source_of(v) for v in val}
    assert len(train) + len(val) == len(accident_build.read_metadata(metadata))
    again = accident_build.split_sources(accident_build.read_metadata(metadata))
    assert again[constants.ACCIDENT_SPLIT_VAL] == val  # seeded: same split twice
    path = tmp_path / "splits" / f"{constants.ACCIDENT_SPLIT_VAL}.txt"
    path.write_text(path.read_text() + "intruder_00\n")
    with pytest.raises(ValueError, match="does not match"):
        accident_build.load_accident_split(constants.ACCIDENT_SPLIT_VAL, tmp_path / "splits")


def test_source_of_strips_the_clip_suffix_and_segment() -> None:
    assert accident_build.source_of("-2UPLUV7JLg_00") == "-2UPLUV7JLg"
    assert accident_build.source_of("3Xd6PxEvbNk_11_00") == "3Xd6PxEvbNk"  # segment of one upload
    assert accident_build.source_of("kaggle-x-cctv_1_00") == "kaggle-x-cctv"
    assert accident_build.source_of("russia12_00") == "russia12"


def test_frozen_repo_split_matches_its_manifest() -> None:
    ids = [accident_build.load_accident_split(n) for n in constants.ACCIDENT_SPLITS]
    assert sum(map(len, ids)) == constants.ACCIDENT_REAL_VIDEOS
    assert not set(ids[0]) & set(ids[1])


# ---------------------------------------------------------------------------
# labels and corpus (A3)
# ---------------------------------------------------------------------------
def test_record_cuts_at_impact_plus_post_and_labels_by_time() -> None:
    row = _row("x_00", "single", impact=10.0, duration=30.0)
    record = accident_build.make_record("x_00", row, frames=900)
    assert record.total_frames == int(12.0 * FPS) + 1  # frames 0 .. 360
    labels = sampled_frame_labels(record, constants.FRAME_STRIDE)
    first = labels.index(1)
    assert abs(first * constants.FRAME_STRIDE / FPS - 10.0) <= constants.FRAME_STRIDE / FPS
    assert all(labels[first:])  # abnormal from the impact to the cut, nothing after the cut


def test_record_keeps_a_short_clip_whole() -> None:
    row = _row("x_00", "single", impact=5.0, duration=6.0)
    assert accident_build.make_record("x_00", row, frames=180).total_frames == 180


def test_corpus_excludes_ids_and_never_reaches_past_the_cut(tmp_path: Path) -> None:
    metadata, clip_dir, _ = _corpus_inputs(tmp_path)
    train = accident_build.load_accident_split(constants.ACCIDENT_SPLIT_TRAIN, tmp_path / "splits")
    excluded = [v for v in train if v.startswith("src000")]
    exclude_file = tmp_path / "exclude.txt"
    exclude_file.write_text("".join(f"{v}\n" for v in excluded))
    out = tmp_path / "ACCIDENT_window"
    args = argparse.Namespace(
        metadata=metadata, clip_s1_dir=clip_dir, split_dir=tmp_path / "splits",
        exclude_ids=exclude_file, post_impact_s=constants.ACCIDENT_POST_IMPACT_S,
        out_dir=out, report=tmp_path / "build_report.json")
    report = accident_build.run_corpus(args)
    windows = json.loads((out / constants.WINDOWS_FILENAME).read_text())
    labels = json.loads((out / constants.LABELS_TRAIN_FILENAME).read_text())
    meta = json.loads((out / constants.META_FILENAME).read_text())
    rows = accident_build.read_metadata(metadata)
    assert not {windows[w]["source"] for w in labels} & set(excluded)
    for w in labels:
        src = windows[w]["source"]
        cut_rows = -(-accident_build.cut_frames(rows[src], 750) // constants.FRAME_STRIDE)
        assert windows[w]["end"] <= cut_rows
    assert 0 < sum(labels.values()) < len(labels)
    assert {meta[w]["class_name"] for w, y in labels.items() if y} == {constants.DADA_CLASS_NAME}
    assert report["construction"]["excluded_clips"] == len(excluded)
    assert report["plan"]["steps_per_epoch"] == -(-2 * sum(labels.values()) // constants.BATCH_SIZE)


def test_unknown_exclusion_raises(tmp_path: Path) -> None:
    metadata, clip_dir, _ = _corpus_inputs(tmp_path)
    exclude_file = tmp_path / "exclude.txt"
    exclude_file.write_text("not_a_clip_00\n")
    args = argparse.Namespace(
        metadata=metadata, clip_s1_dir=clip_dir, split_dir=tmp_path / "splits",
        exclude_ids=exclude_file, post_impact_s=2.0,
        out_dir=tmp_path / "out", report=tmp_path / "r.json")
    with pytest.raises(ValueError, match="not ACCIDENT clips"):
        accident_build.run_corpus(args)


def test_accident_uses_the_nexar_definition_set() -> None:
    assert dataset_abbr(constants.ACCIDENT_DATASET) == dataset_abbr(constants.NEXAR_DATASET)


# ---------------------------------------------------------------------------
# dedup (D)
# ---------------------------------------------------------------------------
def _random_unit(rng: np.random.Generator, n: int, d: int = 32) -> np.ndarray:
    return accident_dedup.unit_rows(rng.normal(size=(n, d)))


def test_dedup_flags_a_shared_camera_and_drops_its_source() -> None:
    rng = np.random.default_rng(1)
    acc: dict[str, np.ndarray] = {}
    for s in range(40):
        cam = _random_unit(rng, 1)
        for k in range(2 if s < 10 else 1):  # sources 0-9: two clips of one camera
            acc[f"s{s:02d}_{k:02d}"] = accident_dedup.unit_rows(
                cam + 0.05 * rng.normal(size=(6, 32)))
    tad = {f"tad{i}": _random_unit(rng, 6) for i in range(5)}
    tad["tad_dup"] = accident_dedup.unit_rows(acc["s20_00"][:3] + 0.01 * rng.normal(size=(3, 32)))
    result = accident_dedup.dedup(acc, tad, cross_sample=10)
    assert result["flagged_clips"] == ["s20_00"]
    assert result["excluded_clips"] == ["s20_00"]
    assert result["matches"]["s20_00"][0]["tad_id"] == "tad_dup"
    assert result["status"] == constants.ACCIDENT_DEDUP_OK
    assert result["control_sources"] == 10
    assert result["control_quantiles"]["p50"] > result["cross_source_nn_quantiles"]["p50"]


def test_dedup_flags_the_whole_source_and_stops_when_too_many_go() -> None:
    rng = np.random.default_rng(2)
    cam = _random_unit(rng, 1)
    acc = {f"s{s:02d}_{k:02d}": accident_dedup.unit_rows(cam + 0.05 * rng.normal(size=(4, 32)))
           for s in range(10) for k in range(2)}
    tad = {"tad": accident_dedup.unit_rows(np.repeat(cam, 4, axis=0))}
    result = accident_dedup.dedup(acc, tad, cross_sample=5)
    assert result["status"] == constants.ACCIDENT_DEDUP_STOP
    flagged_sources = {accident_build.source_of(v) for v in result["flagged_clips"]}
    expected = {v for v in acc if accident_build.source_of(v) in flagged_sources}
    assert set(result["excluded_clips"]) == expected


def test_bank_best_returns_the_rows_that_achieve_the_max() -> None:
    rng = np.random.default_rng(3)
    bank_frames = {"a": _random_unit(rng, 5), "b": _random_unit(rng, 7)}
    query = _random_unit(rng, 4)
    scores, q_row, b_row = accident_dedup.Bank(bank_frames).best(query)
    for i, v in enumerate(sorted(bank_frames)):
        sims = query @ bank_frames[v].T
        assert scores[i] == pytest.approx(sims.max())
        assert sims[q_row[i], b_row[i]] == pytest.approx(sims.max())


def test_dedup_needs_a_same_source_control() -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="control is empty"):
        accident_dedup.dedup({"a_00": _random_unit(rng, 3), "b_00": _random_unit(rng, 3)},
                             {"t": _random_unit(rng, 3)}, cross_sample=2)
