"""Tests for the Nexar N2-N5 tools: extraction, the two corpora, fit-ids baking, the read-out.

The failures that matter are silent: a fused resize that drifts from the CLIP / squash transforms
(every cached number changes, C2), a motion row that sees a future frame, a span that lands one
step off, a window corpus whose negatives come from another pool, a fit that leaks val/test into
the statistics, a crop whose event is not where the placement says, and a read-out that opens
``nexar_test``.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
import torch
from torch import Tensor, nn

from core import constants
from core.data.dada import sampled_frame_labels
from core.data.v2_inputs import fit_stats, load_stats
from core.tools import build_v2_inputs, nexar_build, nexar_eval, nexar_extract, nexar_trigger
from core.tools.extract_clip_features import preprocess_frames
from core.tools.extract_video_features import causal_clip_indices, squash_frames

FPS = 30
FRAME_SIZE = 48
SEED = 11


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_mp4(path: Path, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width = stream.height = FRAME_SIZE
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, FPS)
        for i in range(frames):
            img = np.full((FRAME_SIZE, FRAME_SIZE, 3), (i * 9) % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class _FakeClip:
    """CLIP-shaped: ``.image_embeds`` = the first 512 normalized pixel values."""

    def __call__(self, pixel_values: Tensor) -> Any:
        class _Out:
            image_embeds = pixel_values.flatten(1)[:, :512]

        return _Out()


class _FakeVideo(nn.Module):
    """``(B, 3, 16, H, W)`` -> ``(B, 16)``: each slot's top-left red value, so order is visible."""

    def forward(self, x: Tensor) -> Tensor:
        return x[:, 0, :, 0, 0]


def _row(label: int, duration: float, alert: float | None = None,
         event: float | None = None) -> dict[str, Any]:
    return {
        "label": label, "duration_s": duration, "time_of_alert": alert, "time_of_event": event,
        "issues": [], "fps": 30.0, "meta_scene": "Urban", "meta_weather": "Clear",
        "frames_header": round(duration * FPS),
    }


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def test_fused_resize_matches_both_reference_transforms() -> None:
    frames = np.random.default_rng(SEED).integers(0, 256, (5, 90, 160, 3), dtype=np.uint8)
    resized = nexar_extract.resize_frames(frames)
    clip = nexar_extract.normalize(resized, constants.CLIP_IMAGE_MEAN, constants.CLIP_IMAGE_STD)
    squash = nexar_extract.normalize(
        resized, constants.VIDEOMAE_IMAGE_MEAN, constants.VIDEOMAE_IMAGE_STD
    )
    assert torch.equal(clip, preprocess_frames(frames, center_crop=False))
    assert torch.equal(squash, squash_frames(frames))


def test_motion_rows_are_causal_stride1_clips() -> None:
    frames = 50
    resized = torch.arange(frames, dtype=torch.float32).view(frames, 1, 1, 1).expand(
        frames, 3, 2, 2
    ).contiguous()
    rows = nexar_extract.motion_rows(resized, _FakeVideo(), torch.device("cpu"), batch_size=7)
    mean, std = constants.VIDEOMAE_IMAGE_MEAN[0], constants.VIDEOMAE_IMAGE_STD[0]
    decoded = rows * std + mean  # back to the frame index each slot read
    assert rows.shape == (frames, constants.VIDEOMAE_CLIP_FRAMES)
    for end in (0, 10, 44, 49):
        np.testing.assert_allclose(decoded[end], causal_clip_indices(end), atol=1e-4)
        assert decoded[end].max() <= end + 1e-4  # never a future frame


def test_extract_writes_both_caches_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "raw"
    _write_mp4(root / "train/positive/00001.mp4", 40)
    _write_mp4(root / "train/negative/00002.mp4", 25)
    census = {"inputs": {}, "videos": {
        "00001": {**_row(1, 40 / FPS, 0.5, 1.0), "path": "train/positive/00001.mp4"},
        "00002": {**_row(0, 25 / FPS), "path": "train/negative/00002.mp4"},
    }}
    (tmp_path / "census.json").write_text(json.dumps(census))
    calls = {"clip": 0}

    def fake_clip(_device: torch.device) -> _FakeClip:
        calls["clip"] += 1
        return _FakeClip()

    monkeypatch.setattr(nexar_extract, "load_pretrained_encoder", fake_clip)
    monkeypatch.setattr(nexar_extract, "load_pretrained", lambda *_a: _FakeVideo())
    argv = ["--root", str(root), "--census", str(tmp_path / "census.json"),
            "--clip-out", str(tmp_path / "clip"), "--video-out", str(tmp_path / "video"),
            "--device", "cpu", "--report", str(tmp_path / "rep.json"), "--video-batch", "8"]
    report = nexar_extract.extract(nexar_extract.build_arg_parser().parse_args(argv))
    assert report["summary"]["extracted"] == 2 and report["summary"]["header_mismatch"] == []
    for vid, n in (("00001", 40), ("00002", 25)):
        assert np.load(tmp_path / "clip" / f"{vid}.npy").shape == (n, 512)
        assert np.load(tmp_path / "video" / f"{vid}.npy").shape == (n, 16)
    manifest = json.loads((tmp_path / "video" / constants.VIDEO_MANIFEST_FILENAME).read_text())
    assert manifest["stride"] == 1
    again = nexar_extract.extract(nexar_extract.build_arg_parser().parse_args(argv))
    assert again["summary"]["extracted"] == 0


# ---------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------
def test_span_rule_and_native_labels() -> None:
    row = _row(1, 10.0, alert=4.0, event=5.0)
    assert nexar_build.span_seconds(row) == (4.0, 5.0 + constants.NEXAR_POST_EVENT_S)
    labels = nexar_build.native_labels(row, 300)
    times = np.arange(300) / 30
    assert labels[times < 4.0].sum() == 0 and labels[(times >= 4.0) & (times <= 6.4)].all()
    assert labels[times > 6.4 + 1e-9].sum() == 0
    late = _row(1, 10.0, alert=9.0, event=9.5)
    assert nexar_build.span_seconds(late)[1] == 10.0  # clipped to the video
    assert nexar_build.native_labels(_row(0, 10.0), 300).sum() == 0


def test_sampled_labels_agree_with_native_labels() -> None:
    row = _row(1, 40.0, alert=18.0, event=19.5)
    frames = 1200
    record = nexar_build.make_record("v", row, frames)
    sampled = np.array(sampled_frame_labels(record, 8))
    native = nexar_build.native_labels(row, frames)
    assert abs(int(sampled.sum()) - int(native[::8].sum())) <= 1


def _rows() -> tuple[dict[str, dict[str, Any]], list[str], list[str], dict[str, int]]:
    rows = {
        "p1": _row(1, 40.0, 18.0, 19.5), "p2": _row(1, 40.0, 20.0, 21.0),
        "n1": _row(0, 40.0), "n2": _row(0, 40.0),
        "pv": _row(1, 40.0, 19.0, 20.0), "nv": _row(0, 40.0),
    }
    return rows, ["n1", "n2", "p1", "p2"], ["nv", "pv"], dict.fromkeys(rows, 1200)


def test_build_whole_keeps_negative_videos(tmp_path: Path) -> None:
    rows, train, val, frames = _rows()
    stats = nexar_build.build_whole(tmp_path, rows, train, val, frames)
    labels = json.loads((tmp_path / constants.LABELS_TRAIN_FILENAME).read_text())
    assert labels == {"n1": 0, "n2": 0, "p1": 1, "p2": 1}
    test = json.loads((tmp_path / constants.FRAME_LABELS_TEST_FILENAME).read_text())
    assert set(test) == {"nv", "pv"} and sum(test["nv"]) == 0 and sum(test["pv"]) > 0
    assert len(test["pv"]) == 150  # ceil(1200 / 8)
    assert not (tmp_path / constants.WINDOWS_FILENAME).exists()
    assert stats["val_positive_without_positive_row"] == []


def test_build_window_uses_positive_videos_only(tmp_path: Path) -> None:
    rows, train, val, frames = _rows()
    stats = nexar_build.build_window(tmp_path, rows, train, val, frames)
    windows = json.loads((tmp_path / constants.WINDOWS_FILENAME).read_text())
    assert {w["source"] for w in windows.values()} == {"p1", "p2", "pv"}
    assert {w["end"] - w["start"] for w in windows.values()} == {constants.NEXAR_WINDOW_LENGTH}
    labels = json.loads((tmp_path / constants.LABELS_TRAIN_FILENAME).read_text())
    assert 0 < sum(labels.values()) < len(labels)  # in-video negatives exist
    per_source = (150 - constants.NEXAR_WINDOW_LENGTH) // constants.NEXAR_WINDOW_STRIDE + 1
    assert stats["train_items"] == 2 * per_source  # no cap
    assert stats["val_two_class_windows"] > 0


def test_epoch_plan_spends_the_step_budget() -> None:
    plan = nexar_build.epoch_plan(451)
    assert plan["steps_per_epoch"] == 15
    assert abs(plan["steps"] - constants.NEXAR_STEP_BUDGET) <= plan["steps_per_epoch"] / 2


def test_subsample(tmp_path: Path) -> None:
    (tmp_path / "s1").mkdir()
    np.save(tmp_path / "s1" / "a.npy", np.arange(40, dtype=np.float32).reshape(20, 2))
    (tmp_path / "ids.txt").write_text("a\n")
    args = argparse.Namespace(clip_s1_dir=tmp_path / "s1", ids_file=tmp_path / "ids.txt",
                              stride=8, out_dir=tmp_path / "s8")
    assert nexar_build.run_subsample(args) == 1
    np.testing.assert_array_equal(np.load(tmp_path / "s8" / "a.npy")[:, 0], [0, 16, 32])
    assert nexar_build.run_subsample(args) == 0


# ---------------------------------------------------------------------------
# fit-ids
# ---------------------------------------------------------------------------
def test_fit_ids_fits_on_train_only_and_bakes_at_stride(tmp_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    clip, video = tmp_path / "clip", tmp_path / "video"
    clip.mkdir()
    video.mkdir()
    (video / constants.VIDEO_MANIFEST_FILENAME).write_text(json.dumps({"stride": 1}))
    data = {v: (rng.normal(size=(64, 512)), rng.normal(size=(64, 16))) for v in "abcd"}
    data["d"] = (data["d"][0] * 50, data["d"][1] * 50)  # a val outlier must not move the stats
    for v, (x, u) in data.items():
        np.save(clip / f"{v}.npy", x.astype(np.float32))
        np.save(video / f"{v}.npy", u.astype(np.float32))
    (tmp_path / "train.txt").write_text("a\nb\n")
    (tmp_path / "all.txt").write_text("a\nb\nc\nd\n")
    out = tmp_path / "A3"
    build_v2_inputs.main([
        "fit-ids", "--train-ids-file", str(tmp_path / "train.txt"),
        "--ids-file", str(tmp_path / "all.txt"), "--clip-dir", str(clip),
        "--video-dir", str(video), "--stride", "8", "--crn", "R2",
        "--motion", constants.VIDEOMAE_ENCODER_S, "--out-dir", str(out),
    ])
    manifest = json.loads((out / constants.V2_INPUT_MANIFEST_FILENAME).read_text())
    assert manifest["arm"] == "A3" and manifest["train_sources"] == 2
    assert manifest["stride_over_clip_dir"] == 8 and manifest["baked_ids"] == 4
    expected = fit_stats(
        {v: data[v][0].astype(np.float32)[::8] for v in "ab"},
        {v: data[v][1].astype(np.float32)[::8] for v in "ab"},
        "R2", constants.VIDEOMAE_ENCODER_S,
    )
    assert load_stats(out).s == pytest.approx(expected.s)
    assert np.load(out / "c.npy").shape == (8, 512 + 16)


def test_fit_ids_refuses_train_ids_outside_the_bake(tmp_path: Path) -> None:
    (tmp_path / "train.txt").write_text("a\nz\n")
    (tmp_path / "all.txt").write_text("a\n")
    with pytest.raises(SystemExit, match="not baked"):
        build_v2_inputs.main([
            "fit-ids", "--train-ids-file", str(tmp_path / "train.txt"),
            "--ids-file", str(tmp_path / "all.txt"), "--clip-dir", str(tmp_path),
            "--crn", "R2", "--motion", "none", "--out-dir", str(tmp_path / "o"),
        ])


# ---------------------------------------------------------------------------
# read-out
# ---------------------------------------------------------------------------
def test_crop_puts_the_event_centre_at_the_placement() -> None:
    row = _row(1, 40.0, alert=18.0, event=19.6)  # span 18.0 -> 21.0, centre 19.5 s
    for p in constants.NEXAR_CROP_PLACEMENTS:
        bounds = nexar_eval.crop_bounds(row, 1200, p)
        assert bounds is not None
        start, end = bounds
        assert end - start == round(constants.NEXAR_CROP_S * FPS)
        assert (19.5 * FPS - start) / (end - start) == pytest.approx(p, abs=1 / (end - start))
    early = _row(1, 40.0, alert=1.0, event=2.0)
    assert nexar_eval.crop_bounds(early, 1200, 0.9) is None


def test_items_of_skips_crops_that_do_not_fit() -> None:
    rows = {"p": _row(1, 40.0, 18.0, 19.5), "e": _row(1, 40.0, 1.0, 2.0), "n": _row(0, 40.0)}
    items, skipped = nexar_eval.items_of(sorted(rows), rows, dict.fromkeys(rows, 1200))
    assert {"e", "n", "p"} <= set(items)
    assert sum(1 for i in items if i.startswith("p@")) == len(constants.NEXAR_CROP_PLACEMENTS)
    assert not any(i.startswith("n@") for i in items)
    assert "e" in skipped["0.9"]


def test_item_rows_rereference_each_crop() -> None:
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=(300, 512)).astype(np.float32)
    stats = fit_stats({"a": x1[::8]}, None, "R2", constants.V2_OFF)
    rows = nexar_eval.item_rows(x1, None, stats, 30, 150)
    np.testing.assert_allclose(np.median(rows, axis=0), 0.0, atol=1e-5)
    assert len(rows) == len(range(30, 150, constants.NEXAR_EVAL_STRIDE))
    np.testing.assert_array_equal(nexar_eval.item_rows(x1, None, None, 0, 9), x1[0:9:3])


def test_summarize_reads_perfect_scores_and_the_middle_ruler() -> None:
    rows = {"p": _row(1, 40.0, 18.0, 19.6), "n": _row(0, 40.0)}
    frames = dict.fromkeys(rows, 1200)
    items, _ = nexar_eval.items_of(sorted(rows), rows, frames)
    labels = {i: nexar_build.native_labels(rows[v], frames[v])[s:e]
              for i, (v, s, e) in items.items()}
    scores = {i: lab.astype(float) for i, lab in labels.items()}
    out = nexar_eval.summarize(scores, labels, sorted(rows), rows, items, SEED)
    assert out["whole"]["macro"]["mean"] == 1.0 and out["whole"]["clip_auc_max"] == 1.0
    assert out["crops"]["mean_over_placements"]["macro"]["mean"] == 1.0
    ruler = {p: out["crops"][str(p)]["middle_ruler_macro"]["mean"]
             for p in constants.NEXAR_CROP_PLACEMENTS}
    assert ruler[0.5] > 0.9 and ruler[0.1] < 0.5 and ruler[0.9] < 0.5


def test_readout_refuses_the_sealed_split(tmp_path: Path) -> None:
    args = nexar_eval.build_arg_parser().parse_args([
        "--run", "s", str(tmp_path / "c.pt"), "--census", str(tmp_path / "c.json"),
        "--clip-s1-dir", str(tmp_path), "--data-dir", str(tmp_path),
        "--split", constants.V2_SPLIT_NEXAR_TEST, "--out-dir", str(tmp_path / "o"),
    ])
    with pytest.raises(SystemExit, match="sealed"):
        nexar_eval.run(args)


def _readout(crops: dict[float, float], clip_auc: float = 0.7) -> dict[str, Any]:
    """A minimal nexar_eval read-out: crop macro means per placement, endpoint = their mean."""
    def ci(m: float) -> dict[str, float]:
        return {"mean": m, "low": m, "high": m}

    out: dict[str, Any] = {str(p): {"macro": ci(m)} for p, m in crops.items()}
    out["mean_over_placements"] = {"macro": ci(float(np.mean(list(crops.values()))))}
    whole = {"clip_auc_max": clip_auc}
    return {"split": constants.V2_SPLIT_NEXAR_VAL, "crops": out, "whole": whole}


FLAT = dict.fromkeys(constants.NEXAR_CROP_PLACEMENTS, 0.70)


def test_trigger_passes_a_flat_model_above_zero_shot_and_a0() -> None:
    result = nexar_trigger.window_trigger(_readout(FLAT), _readout(dict.fromkeys(FLAT, 0.6)),
                                          _readout(dict.fromkeys(FLAT, 0.6)))
    assert not result["build_window"] and result["fired"] == []


def test_trigger_w1_fires_on_a_centre_peaked_model() -> None:
    peaked = {**FLAT, 0.5: 0.80, 0.9: 0.70 - 1e-3}       # centre - edge = 0.101 > 0.05
    result = nexar_trigger.window_trigger(_readout(peaked), _readout(dict.fromkeys(FLAT, 0.6)),
                                          _readout(dict.fromkeys(FLAT, 0.6)))
    assert result["fired"] == ["W1_position"]
    edge = {**FLAT, 0.5: 0.70 + constants.NEXAR_TRIGGER_MAX_EDGE_DROP / 2}
    assert not nexar_trigger.window_trigger(_readout(edge), _readout(FLAT), _readout(FLAT))[
        "checks"]["W1_position"]["fires"]


def test_trigger_w2_fires_when_training_does_not_beat_zero_shot() -> None:
    result = nexar_trigger.window_trigger(_readout(FLAT), _readout(dict.fromkeys(FLAT, 0.6)),
                                          _readout(FLAT))                  # tie counts as no gain
    assert result["fired"] == ["W2_no_gain"]


def test_trigger_w3_needs_both_a_high_clip_auc_and_a_loss_to_a0() -> None:
    zs, a0 = _readout(dict.fromkeys(FLAT, 0.6)), _readout(dict.fromkeys(FLAT, 0.75))
    high = _readout(FLAT, clip_auc=constants.NEXAR_TRIGGER_CLIP_AUC)
    assert nexar_trigger.window_trigger(high, a0, zs)["fired"] == ["W3_clip_classifier"]
    assert nexar_trigger.window_trigger(_readout(FLAT, clip_auc=0.9), a0, zs)["fired"] == []
    assert nexar_trigger.window_trigger(high, _readout(dict.fromkeys(FLAT, 0.6)), zs)["fired"] == []


def test_trigger_cli_refuses_a_non_val_readout(tmp_path: Path) -> None:
    dirs = {}
    for name, split in (("a3", constants.V2_SPLIT_NEXAR_VAL), ("a0", constants.V2_SPLIT_NEXAR_VAL),
                        ("zs", constants.V2_SPLIT_NEXAR_TEST)):
        dirs[name] = tmp_path / name
        dirs[name].mkdir()
        payload = {**_readout(FLAT), "split": split}
        (dirs[name] / nexar_eval.READOUT_JSON).write_text(json.dumps(payload))
    argv = ["--whole-a3", str(dirs["a3"]), "--whole-a0", str(dirs["a0"]),
            "--zero-shot", str(dirs["zs"]), "--out-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="nexar_val only"):
        nexar_trigger.run(nexar_trigger.build_arg_parser().parse_args(argv))
    zero_shot = _readout(dict.fromkeys(FLAT, 0.6))
    (dirs["zs"] / nexar_eval.READOUT_JSON).write_text(json.dumps(zero_shot))
    nexar_trigger.main(argv)
    written = json.loads((tmp_path / "out" / nexar_trigger.TRIGGER_JSON).read_text())
    assert written["build_window"] is False
    assert "passes" in (tmp_path / "out" / nexar_trigger.TRIGGER_MD).read_text()
