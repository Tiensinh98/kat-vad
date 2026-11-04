"""DAD zero-shot set (``core/docs/final/ZS_DAD.md``): census, labels, decode, the VideoMAE frame
step flag, and the DAD column of the zero-shot read-out."""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest

from core import constants
from core.data.definitions import dataset_abbr
from core.tests.test_zs_lagovad import _layout
from core.tests.test_zs_tad_dada import _tad_run
from core.tools import dad, zs_lagovad_readout
from core.tools import extract_video_features as evf
from core.tools.nexar_transfer_readout import read_tad_runs

ANN = "\n".join([
    "000001\t1\tmotorbike\t640\t257\t694\t332\t0",
    "000001\t9\tcar\t318\t230\t426\t286\t0",
    *[f"{f:06d}\t1\tmotorbike\t640\t257\t694\t332\t{int(f >= 91)}" for f in range(2, 101)],
])
ANN_INVOLVED = "\n".join(f"{f:06d}\t1\tcar\t1\t2\t3\t4\t1" for f in range(1, 101))
ANN_UNFLAGGED = "\n".join(f"{f:06d}\t2\tcar\t1\t2\t3\t4\t0" for f in range(1, 101))


def _mp4(path: Path, frames: int, fps: int = constants.DAD_ASSUMED_FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(frames):
            img = np.full((48, 64, 3), (i * 2) % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def _dad_root(tmp_path: Path, fps: int = constants.DAD_ASSUMED_FPS,
             pos_frames: int = 100) -> Path:
    """pos 000001 (flagged from frame 1, as in DAD), neg 000001 (shared stem), neg 000002."""
    root = tmp_path / "DAD"
    _mp4(root / "videos" / "testing" / "positive" / "000001.mp4", pos_frames, fps)
    for stem in ("000001", "000002"):
        _mp4(root / "videos" / "testing" / "negative" / f"{stem}.mp4", 100, fps)
    ann = root / constants.DAD_ANNOTATION_DIRNAME
    ann.mkdir(parents=True)
    (ann / "000001.txt").write_text(ANN_INVOLVED)
    (ann / "000002.txt").write_text(ANN_UNFLAGGED)
    return root


class TestPieces:
    def test_parse_annotation(self) -> None:
        flagged, last = dad.parse_annotation(ANN)
        assert flagged == set(range(91, 101)) and last == 100
        with pytest.raises(ValueError, match="columns"):
            dad.parse_annotation("000001\t1\tcar\t1\t2\t3")
        with pytest.raises(ValueError, match="out of range"):
            dad.parse_annotation("000001\t1\tcar\t1\t2\t3\t4\t2")

    def test_labels_follow_toa_not_the_flag(self) -> None:
        native = dad.native_labels("positive", 100)
        assert native[: constants.DAD_TOA_FRAME].sum() == 0 and native[90:].sum() == 10
        assert dad.native_labels("negative", 100).sum() == 0
        rows = dad.row_labels(native, constants.DAD_STRIDE)  # frames 0, 7, ..., 98
        assert len(rows) == 15 and rows[-2:] == [1, 1] and sum(rows) == 2
        with pytest.raises(ValueError, match="toa"):
            dad.native_labels("positive", 90)

    def test_entry_reasons(self) -> None:
        r = dad.entry_reason
        assert r("positive", 25.0, 100) == dad.OK
        assert r("negative", 25.0, 100) == dad.OK
        assert r("positive", 20.0, 100) == dad.FPS_MISMATCH
        assert r("positive", 25.0, 90) == dad.TOO_SHORT

    def test_dad_uses_the_traffic_definitions(self) -> None:
        assert dataset_abbr(constants.DAD_DATASET) == dataset_abbr(constants.NEXAR_DATASET)

    def test_protocol_geometry(self) -> None:
        step = constants.DAD_STRIDE / constants.DAD_ASSUMED_FPS
        assert step == pytest.approx(0.28)
        span = (constants.VIDEOMAE_CLIP_FRAMES - 1) * constants.DAD_VIDEOMAE_FRAME_STEP
        assert span / constants.DAD_ASSUMED_FPS == pytest.approx(1.8)

    def test_videomae_manifest_records_the_frame_step(self) -> None:
        sha = constants.VIDEOMAE_WEIGHTS_SHA256[constants.VIDEOMAE_ENCODER_S]
        base = evf.build_manifest(constants.VIDEOMAE_ENCODER_S, 7, sha)
        dad_m = evf.build_manifest(constants.VIDEOMAE_ENCODER_S, 7, sha, clip_frame_step=2,
                                   assumed_fps=constants.DAD_ASSUMED_FPS)
        assert base["clip_frame_step"] == constants.VIDEOMAE_CLIP_FRAME_STEP
        assert {k for k in base if base[k] != dad_m[k]} == {"clip_frame_step", "assumed_fps"}
        args = evf.build_arg_parser().parse_args(
            ["--frames-dir", "f", "--dataset", "DAD", "--clip-frame-step", "2"])
        assert args.clip_frame_step == 2


class TestCensusPrepareDecode:
    def test_end_to_end(self, tmp_path: Path) -> None:
        root = _dad_root(tmp_path)
        out = tmp_path / "census.json"
        dad.main(["census", "--root", str(root), "--out", str(out)])
        doc = json.loads(out.read_text())
        assert sorted(doc["videos"]) == ["neg_000001", "neg_000002", "pos_000001"]
        flags = doc["summary"]["printed_flag_stats"]
        assert flags["first_flag_frame_median"] == 1 and flags["positives_flagged"] == 1
        assert doc["summary"]["shared_stems"] == 1 and doc["summary"]["pass"] is True

        defs = tmp_path / "defs.json"
        defs.write_text(json.dumps(["Normal", "Abnormal"]))
        data = tmp_path / "data"
        dad.main(["prepare", "--census", str(out), "--defs", str(defs), "--out-dir", str(data)])
        labels = json.loads((data / constants.FRAME_LABELS_TEST_FILENAME).read_text())
        assert labels["pos_000001"] == [0] * 13 + [1, 1]  # toa, though the flag covers all 100
        assert sum(labels["neg_000001"]) == 0
        man = json.loads((data / constants.DAD_MANIFEST_FILENAME).read_text())
        assert (man["videos"], man["abnormal"], man["two_class_at_stride"]) == (3, 1, 1)

        frames = tmp_path / "frames"
        dad.main(["decode", "--root", str(root), "--census", str(out), "--out-dir", str(frames)])
        assert len(list((frames / "pos_000001").glob("*.jpg"))) == 100
        assert dad.decode_video(root / "videos/testing/positive/000001.mp4",
                                frames / "pos_000001", 100) is False  # resumed

    def test_census_fails_on_a_wrong_fps(self, tmp_path: Path) -> None:
        root = _dad_root(tmp_path, fps=20)
        with pytest.raises(SystemExit, match="FAILED"):
            dad.main(["census", "--root", str(root), "--out", str(tmp_path / "c.json")])
        doc = json.loads((tmp_path / "c.json").read_text())
        assert doc["videos"]["pos_000001"]["reason"] == dad.FPS_MISMATCH
        with pytest.raises(ValueError, match="did not pass"):
            dad.prepare(doc, tmp_path / "c.json", 7, tmp_path / "d")

    def test_census_fails_on_a_short_positive(self, tmp_path: Path) -> None:
        root = _dad_root(tmp_path, pos_frames=80)
        with pytest.raises(SystemExit, match="FAILED"):
            dad.main(["census", "--root", str(root), "--out", str(tmp_path / "c.json")])


class TestReadoutDad:
    def test_dad_column_is_added_when_present(self, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
        argv = _layout(tmp_path)
        t0 = read_tad_runs(tmp_path / "lag" / "eval" / "tad" / "LaGoVAD", ["released"], 0)
        monkeypatch.setattr(constants, "ZS_LAGOVAD_T0_MICRO", t0["micro"])
        rng = np.random.default_rng(3)
        base = tmp_path / "lag" / "eval" / "dad"
        _tad_run(base / "LaGoVAD" / "released", 0.3, rng)
        _tad_run(base / "W0" / "s2024", 0.3, rng)
        _tad_run(base / "R4" / "s2024", 3.0, rng)
        r = zs_lagovad_readout.run(zs_lagovad_readout.build_arg_parser().parse_args(argv))
        assert list(r["sets"]) == ["dota", "tad", "dada", "dad"]
        assert r["sets"]["dad"]["sa2_minus_host"]["sentence"] == "gain shown"
        md = (tmp_path / "lag" / "REPORTS" / "zs_lagovad_readout.md").read_text()
        assert "| row | DoTA-CAP-dev (n/1397) | TAD test | DADA-2000 | DAD |" in md
