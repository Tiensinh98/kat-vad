"""``dota_train_match`` / ``dota_train_sheet``: DoTA train clips in CAP-DATA (KIM_RUN.md §52)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from core import constants
from core.tools import dota_cap_frames as dcf
from core.tools import dota_train_match as dtm
from core.tools import dota_train_sheet as dts

BLOCK_B = constants.DOTA_TRAIN_BLOCKS["B"][0]
BLOCK_C = constants.DOTA_TRAIN_BLOCKS["C"][0]


def _clip(name: str, frames: int, start: int = 10) -> dtm.DotaClip:
    return dtm.DotaClip(name, frames, start, start + 20, False)


def _cap(cap_id: int, total: int, start: int = 10) -> dtm.CapVideo:
    return dtm.CapVideo(cap_id, 11, 1, start, start + 20, total)


def test_windows_key_ignores_hyphen_and_puts_underscore_first() -> None:
    names = ["t4Cv_000345", "t_bC_004475", "J-j5_000194", "JHfj_000551", "0q_1"]
    assert sorted(names, key=dtm.windows_key) == [
        "0q_1", "JHfj_000551", "J-j5_000194", "t_bC_004475", "t4Cv_000345"
    ]


def test_monotone_match_maximises_weight_and_stays_increasing() -> None:
    w = np.zeros((3, 4))
    w[0, 2] = w[1, 0] = w[2, 1] = 1.0  # taking (0, 2) would block both others
    assert dtm.monotone_match(w) == [(1, 0), (2, 1)]


def test_match_gives_train_clips_the_rows_between_val_anchors() -> None:
    lengths = [60, 71, 82, 93, 104]
    ids = [f"vid{i}_00000{i}" for i in range(5)]
    dota = {k: _clip(k, n) for k, n in zip(ids, lengths, strict=True)}
    cap = [_cap(BLOCK_B + 2 * i, n) for i, n in enumerate(lengths)]
    val = {ids[0], ids[3]}
    exact = {ids[0]: BLOCK_B, ids[3]: BLOCK_B + 6}
    out = dtm.build_map(cap, dota, val, exact, {q: {c} for q, c in exact.items()})
    got = {k: int(c["cap_id"]) for k, c in out["clips"].items()}
    assert got == {ids[1]: BLOCK_B + 2, ids[2]: BLOCK_B + 4, ids[4]: BLOCK_B + 8}
    assert out["cross_validation"]["right"] == 2 and out["cross_validation"]["wrong"] == 0


def test_a_val_clips_cap_video_is_never_given_to_a_train_clip() -> None:
    dota = {k: _clip(k, 60) for k in ("a_000001", "b_000001")}
    cap = [_cap(BLOCK_B, 60), _cap(BLOCK_B + 1, 60)]
    hits = {"a_000001": {BLOCK_B, BLOCK_B + 1}}  # both copies are the val clip's
    out = dtm.build_map(cap, dota, {"a_000001"}, {"a_000001": BLOCK_B}, hits)
    assert out["clips"] == {}


def test_ten_fps_copy_is_preferred_and_30fps_needs_the_measured_length() -> None:
    dota = {"a_000001": _clip("a_000001", 60), "b_000001": _clip("b_000001", 70)}
    cap = [_cap(BLOCK_B, 60), _cap(BLOCK_C, 178, 30), _cap(BLOCK_C + 1, 208, 30)]
    found = dtm.align(cap, dota, {})
    assert dtm.pick(found["a_000001"], set()) == ("B", BLOCK_B)
    assert dtm.pick(found["b_000001"], set()) == ("C", BLOCK_C + 1)
    assert "a_000001" not in dtm.align([_cap(BLOCK_C, 181, 30)], dota, {})


@pytest.mark.parametrize("rate, total", [(1, 5), (3, 13), (3, 12)])
def test_frame_map_stays_inside_the_cap_video(rate: int, total: int) -> None:
    m = dtm.frame_map(5, total, rate)
    assert len(m) == 5 and m[0] == 0 and max(m) < total and m == sorted(set(m))
    assert m == ([0, 1, 2, 3, 4] if rate == 1 else [0, 2, 5, 8, 11])


def test_frozen_map_is_consistent() -> None:
    frozen = dtm.load_map(constants.DOTA_TRAIN_SPLITS_DIR / constants.DOTA_TRAIN_MAP_FILENAME)
    clips = dtm.expand(frozen)["clips"]
    assert len(clips) == frozen["matched"] == 3023
    assert len({c["cap_id"] for c in clips.values()}) == len(clips)
    assert all(len(c["frame_map"]) == c["dota_frames"] for c in clips.values())
    assert all(c["frame_map"][-1] < c["cap_frames"] for c in clips.values())
    val = {p for name in (constants.V2_SPLIT_DOTA_CAP_DEV, constants.V2_SPLIT_DOTA_CAP_EVAL)
           for p in (constants.V2_SPLITS_DIR / f"{name}.txt").read_text().split()}
    assert not val & clips.keys()


def test_load_map_refuses_edited_ids(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"clips": {"a_1": {}}, "ids_sha1": "nope"}))
    with pytest.raises(ValueError, match="ids_sha1"):
        dtm.load_map(path)


def _archive(archive_dir: Path, dota_id: str, frames: int) -> None:
    src = archive_dir / "src"
    src.mkdir(parents=True, exist_ok=True)
    path = src / "f.jpg"
    Image.new("RGB", (1280, 720), "grey").save(path)
    dcf.write_archive(dcf.archive_path(archive_dir, dota_id), [path], [0] * frames)


def test_finalize_drops_only_a_frame_count_mismatch(tmp_path: Path) -> None:
    _archive(tmp_path, "a_000001", 3)
    alignment: dict[str, Any] = {"clips": {"a_000001": {"dota_frames": 3},
                                           "b_000001": {"dota_frames": 4}}}
    with pytest.raises(ValueError, match="not packed yet"):
        dtm.finalize(alignment, [{}], tmp_path)
    out = dtm.finalize(alignment, [{"b_000001": {"reason": dcf.CAP_FRAMES_MISMATCH}}], tmp_path)
    assert list(out["clips"]) == ["a_000001"] and out["dropped"] == ["b_000001"]


def test_verify_without_an_s1_cache(tmp_path: Path) -> None:
    _archive(tmp_path / "arc", "a_000001", 3)
    alignment = tmp_path / "align.json"
    alignment.write_text(json.dumps({"clips": {"a_000001": {
        "reason": "ok", "dota_frames": 3, "frame_map": [0, 0, 0]}}}))
    dcf.main(["verify", "--alignment", str(alignment), "--archive-dir", str(tmp_path / "arc"),
              "--report", str(tmp_path / "v.json")])
    manifest = json.loads((tmp_path / "arc" / constants.DOTA_CAP_FRAMES_MANIFEST).read_text())
    assert manifest["pass"] and "annotation" in manifest["rows"]


@pytest.fixture
def small_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(constants, "DOTA_SHEET_PER_GROUP", {"10fps": 2, "30fps": 1})
    monkeypatch.setattr(constants, "DOTA_SHEET_DECOYS", 1)
    monkeypatch.setattr(constants, "DOTA_SHEET_VAL_CONTROLS", 1)


def test_sheet_draws_shuffled_rows_and_hides_the_truth(
    tmp_path: Path, small_gate: None
) -> None:
    train, val = tmp_path / "train", tmp_path / "val"
    ids = [f"vid_00000{i}" for i in range(5)]
    clips = {k: {"rate": 3 if k == ids[4] else 1} for k in ids}
    for k in ids:
        _archive(train, k, 12)
    _archive(val, "val_000001", 12)
    ann = tmp_path / "ann.zip"
    with zipfile.ZipFile(ann, "w") as zf:
        for k in [*ids, "val_000001"]:
            labels = [{"objects": [{"bbox": [100, 100, 300, 200]}]} for _ in range(12)]
            zf.writestr(f"annotations/{k}.json", json.dumps(
                {"anomaly_start": 2, "anomaly_end": 9, "labels": labels}))
    alignment = tmp_path / "a.json"
    alignment.write_text(json.dumps({"clips": clips}))
    out = tmp_path / "out"
    dts.main(["sheet", "--alignment", str(alignment), "--archive-dir", str(train),
              "--val-archive-dir", str(val), "--annotations", str(ann), "--out-dir", str(out)])
    key = json.loads((out / dts.KEY_JSON).read_text())
    kinds = sorted(r["kind"] for r in key["rows"].values())
    assert kinds == ["control", "decoy", "real", "real", "real"]
    assert all(r["frames"] == [2, 5, 8] for r in key["rows"].values())
    decoy = next(r for r in key["rows"].values() if r["kind"] == dts.DECOY)
    assert decoy["frames_id"] != decoy["dota_id"]
    assert (out / dts.PAGE_NAME.format(1)).is_file() and key["frame_sizes"] == {"1280x720": 5}


def test_boxes_are_scaled_onto_the_tile() -> None:
    tile = dts.draw_boxes(Image.new("RGB", (640, 360), "black"), [[0, 0, 1280, 720]])
    assert tile.getpixel((1, tile.size[1] // 2)) == (255, 32, 32)
    assert tile.getpixel((tile.size[0] // 2, tile.size[1] // 2)) == (0, 0, 0)


def _key() -> dict[str, Any]:
    rows = [{"kind": dts.REAL, "group": g, "dota_id": f"{g}{i}"}
            for g in ("10fps", "30fps") for i in range(30)]
    rows += [{"kind": dts.DECOY, "group": "10fps", "dota_id": f"d{i}"} for i in range(10)]
    rows += [{"kind": dts.CONTROL, "group": dts.CONTROL, "dota_id": f"c{i}"} for i in range(6)]
    return {"rows": {str(n): r for n, r in enumerate(rows, start=1)}}


def test_score_needs_caught_decoys_and_quiet_controls() -> None:
    decoys = set(range(61, 71))
    good = dts.score(_key(), decoys)
    assert good["gate_valid"] and good["usable_groups"] == ["10fps", "30fps"]
    assert not dts.score(_key(), set(range(61, 68)))["gate_valid"]  # 7 of 10 caught
    assert not dts.score(_key(), decoys | {71, 72})["gate_valid"]  # 2 controls flagged
    dropped = dts.score(_key(), decoys | {31, 32, 33})  # 3 real clips of the 30 fps group
    assert dropped["gate_valid"] and dropped["usable_groups"] == ["10fps"]
    with pytest.raises(ValueError, match="no such rows"):
        dts.score(_key(), {999})
