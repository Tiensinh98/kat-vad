"""``core.tools.nexar_caption`` (``KIM_RUN.md`` §29) with a fake captioner: no model download."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

from core import constants
from core.data.v2_splits import load_split
from core.tools import nexar_caption as nc

FPS = 30
FRAMES = 150  # 5 s
ALERT, EVENT = 1.0, 2.0  # span [1.0, 3.4] s


def _mp4(path: Path, frames: int = FRAMES, fps: int = FPS) -> None:
    """Frame ``i`` is a flat image of value ``i``: the decoded mean says which frame it is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(frames):
            img = np.full((48, 64, 3), i, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def _row(vid: str, label: int = 1) -> dict[str, Any]:
    pos = label == 1
    return {"label": label, "path": f"train/{'positive' if pos else 'negative'}/{vid}.mp4",
            "fps": float(FPS), "frames_header": FRAMES, "duration_s": FRAMES / FPS,
            "time_of_alert": ALERT if pos else None, "time_of_event": EVENT if pos else None,
            "issues": []}


def _census(positives: list[str]) -> dict[str, Any]:
    ids = sorted(load_split(constants.V2_SPLIT_NEXAR_TRAIN))
    return {"videos": {v: _row(v, int(v in positives)) for v in ids}}


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, dict[str, Any], list[str]]:
    positives = sorted(load_split(constants.V2_SPLIT_NEXAR_TRAIN))[:3]
    census = _census(positives)
    for v in positives:
        _mp4(tmp_path / "raw" / census["videos"][v]["path"])
    return tmp_path, census, positives


class FakeCaptioner:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frames: np.ndarray, fps: float) -> str:
        self.calls += 1
        assert frames.shape == (constants.NEXAR_CAPTION_FRAMES, 252, 448, 3)
        return f'"The camera car brakes behind a truck ({fps:.2f} fps)."\nextra line'


IDENT = nc.identity(constants.NEXAR_CAPTION_MODEL, "abc123", constants.NEXAR_CAPTION_PROMPT)


class TestFrames:
    def test_indices_cover_the_span(self) -> None:
        indices, (start, end) = nc.event_frame_indices(_row("x"))
        assert (start, end) == (ALERT, pytest.approx(EVENT + constants.NEXAR_POST_EVENT_S))
        assert len(set(indices)) == constants.NEXAR_CAPTION_FRAMES == len(indices)
        assert indices == sorted(indices)
        assert start * FPS <= indices[0] and indices[-1] <= end * FPS

    def test_span_is_clipped_at_the_video_end(self) -> None:
        row = _row("x") | {"time_of_event": FRAMES / FPS - 0.5}
        _, (_, end) = nc.event_frame_indices(row)
        assert end == pytest.approx(FRAMES / FPS)

    def test_too_few_frames_raises(self) -> None:
        with pytest.raises(ValueError, match="holds <"):
            nc.event_frame_indices(_row("x") | {"fps": 2.0})

    def test_decode_returns_the_requested_frames_resized(self, tmp_path: Path) -> None:
        _mp4(tmp_path / "v.mp4")
        frames = nc.decode_frames(tmp_path / "v.mp4", [10, 40, 90])
        assert frames.shape == (3, 252, 448, 3) and frames.dtype == np.uint8
        np.testing.assert_allclose(frames.reshape(3, -1).mean(1), [10, 40, 90], atol=4)

    def test_decode_past_the_end_raises(self, tmp_path: Path) -> None:
        _mp4(tmp_path / "v.mp4")
        with pytest.raises(ValueError, match="not decoded"):
            nc.decode_frames(tmp_path / "v.mp4", [10, FRAMES + 5])

    def test_qwen_video_grid_is_576_tokens(self) -> None:
        """CP2 against the real processor class: 448 x 252 is kept, grid (4, 18, 32)."""
        assert nc.expected_grid() == (4, 18, 32)
        frames = np.zeros((constants.NEXAR_CAPTION_FRAMES, 252, 448, 3), dtype=np.uint8)
        out = Qwen2VLVideoProcessor()(videos=[frames], min_pixels=3136, max_pixels=12845056,
                                       return_tensors="pt")
        assert tuple(out["video_grid_thw"][0].tolist()) == nc.expected_grid()
        assert out["pixel_values_videos"].shape[0] // 4 == 576  # merge 2 x 2


class TestText:
    def test_clean_caption(self) -> None:
        assert nc.clean_caption('\n  "The car swerves."  \nmore') == "The car swerves."
        with pytest.raises(ValueError, match="empty"):
            nc.clean_caption("  \n ")

    def test_forbidden_terms(self) -> None:
        assert nc.forbidden_terms("The white car hits a truck on the highway.") == [
            "highway", "white"]
        assert nc.forbidden_terms("The camera car rear-ends a truck.") == []

    def test_main_revision_is_refused(self) -> None:
        with pytest.raises(ValueError, match="pin a model revision"):
            nc.identity(constants.NEXAR_CAPTION_MODEL, "main", constants.NEXAR_CAPTION_PROMPT)


class TestRun:
    def test_captions_are_written_then_skipped(self, setup: tuple[Path, dict[str, Any], list[str]]
                                               ) -> None:
        root, census, positives = setup
        fake = FakeCaptioner()
        records = nc.caption_videos(root / "raw", census, root / "out", IDENT, fake)
        assert sorted(records) == positives and fake.calls == 3
        rec = json.loads((root / "out" / "captions" / f"{positives[0]}.json").read_text())
        assert rec["caption"].startswith("The camera car") and rec["revision"] == "abc123"
        assert rec["forbidden"] == [] and len(rec["frame_indices"]) == 8
        nc.caption_videos(root / "raw", census, root / "out", IDENT, fake)
        assert fake.calls == 3  # resume: nothing recomputed

    def test_a_record_from_another_revision_is_refused(
            self, setup: tuple[Path, dict[str, Any], list[str]]) -> None:
        root, census, _ = setup
        nc.caption_videos(root / "raw", census, root / "out", IDENT, FakeCaptioner())
        other = IDENT | {"revision": "def456"}
        with pytest.raises(ValueError, match="another setting"):
            nc.caption_videos(root / "raw", census, root / "out", other, FakeCaptioner())

    def test_manifest_binds_the_split(self, setup: tuple[Path, dict[str, Any], list[str]]) -> None:
        root, census, positives = setup
        records = nc.caption_videos(root / "raw", census, root / "out", IDENT, FakeCaptioner())
        manifest = nc.write_manifest(root / "out", census, IDENT, records)
        assert manifest["videos"] == 3 and manifest["summary"]["distinct"] == 1
        assert manifest["train_ids_sha1"] == json.loads(
            (root / "out" / nc.MANIFEST_FILENAME).read_text())["train_ids_sha1"]
        with pytest.raises(ValueError, match="captions for"):
            nc.write_manifest(root / "out", census, IDENT, {positives[0]: records[positives[0]]})

    def test_census_issue_raises(self) -> None:
        census = _census(sorted(load_split(constants.V2_SPLIT_NEXAR_TRAIN))[:1])
        vid = next(v for v, r in census["videos"].items() if r["label"] == 1)
        census["videos"][vid]["issues"] = ["alert_after_event"]
        with pytest.raises(ValueError, match="census issues"):
            nc.train_positives(census)


class TestQc:
    def test_sheets_and_score(self, setup: tuple[Path, dict[str, Any], list[str]],
                              monkeypatch: pytest.MonkeyPatch) -> None:
        root, census, _ = setup
        monkeypatch.setattr(constants, "NEXAR_CAPTION_QC_SAMPLE", 2)
        monkeypatch.setattr(constants, "NEXAR_CAPTION_QC_PASS", 2)
        records = nc.caption_videos(root / "raw", census, root / "out", IDENT, FakeCaptioner())
        nc.write_manifest(root / "out", census, IDENT, records)
        qc = nc.write_qc(root / "raw", census, root / "out")
        assert len(qc["sample"]) == 2 and qc["sample"] == nc.qc_sample(list(records))
        assert len(list((root / "out" / "qc").glob("*.png"))) == 2
        ratings = root / "out" / "qc" / nc.QC_RATINGS_FILENAME
        with pytest.raises(ValueError, match="not rated"):
            nc.score_qc(root / "out")
        rows = list(csv.DictReader(ratings.open()))
        rows[0]["right"], rows[1]["right"] = "y", "n"
        with ratings.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = nc.score_qc(root / "out")
        assert (result["right"], result["verdict"]) == (1, "fail")


def test_run_requires_a_revision() -> None:
    with pytest.raises(SystemExit):
        nc.build_arg_parser().parse_args(["run", "--raw-root", "r", "--census", "c",
                                          "--out-dir", "o"])
