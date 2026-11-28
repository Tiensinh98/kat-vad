"""Label-template captions for DoTA train (``KIM_RUN.md`` §65) and their manifest binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import constants
from core.data.captions import dota_train_ids, load_caption_manifest
from core.data.v2_splits import lines_sha1, load_split
from core.tools import dota_label_captions as dl


def _record(kind: str, ego: bool, cats: list[str], vid: str = "x_000001") -> dict[str, Any]:
    objects = [{"obj_track_id": k, "category": c, "bbox": [0, 0, 1, 1]} for k, c in enumerate(cats)]
    return {"video_name": vid, "accident_name": kind, "ego_involve": ego,
            "labels": [{"accident_name": "normal", "objects": [{"obj_track_id": 99,
                                                                 "category": "bus"}]},
                       {"accident_name": kind, "objects": objects},
                       {"accident_name": kind, "objects": objects}]}


@pytest.mark.parametrize(("kind", "ego", "cats", "want"), [
    ("turning", True, ["car"], "The camera car collides with a car that turns into or crosses "
                               "the road."),
    ("turning", False, ["car", "truck"], "A car and a truck collide while one turns into or "
                                         "crosses the road. The camera car is not involved."),
    ("oncoming", False, ["car", "car"], "Two cars coming from opposite directions collide. "
                                        "The camera car is not involved."),
    ("lateral", False, ["car"], "A car and another vehicle moving sideways in the same direction "
                                "collide. The camera car is not involved."),
    ("pedestrian", True, ["person"], "The camera car hits a pedestrian."),
    ("pedestrian", False, ["car", "person"], "A car hits a pedestrian. The camera car is not "
                                             "involved."),
    ("obstacle", True, [], "The camera car hits an obstacle on the road."),
    ("leave_to_left", False, [], "A vehicle loses control and leaves the road to the left. "
                                 "The camera car is not involved."),
    ("moving_ahead_or_waiting", True, [], "The camera car hits another vehicle that is moving "
                                          "ahead or waiting in the same lane."),
    ("turning", True, ["motor", "rider"], "The camera car collides with a motorcycle that turns "
                                          "into or crosses the road."),
])
def test_caption_from_type_ego_and_boxed_parties(kind: str, ego: bool, cats: list[str],
                                                  want: str) -> None:
    assert dl.label_caption(_record(kind, ego, cats)) == want


def test_parties_ignore_normal_frames_and_count_tracks_once() -> None:
    assert dl.parties(_record("turning", False, ["truck", "car", "car"])) == ["car", "car", "truck"]
    assert dl.parties(_record("turning", False, ["rider"])) == ["rider"]


def test_every_type_has_both_templates() -> None:
    assert set(constants.DOTA_LABEL_CAPTION_EGO) == set(constants.DOTA_LABEL_CAPTION_OTHER) \
        == set(constants.DOTA_CAPTION_TYPE_TEXT)
    with pytest.raises(ValueError, match="no template"):
        dl.label_caption(_record("rollover", True, []))


def _manifest(tmp_path: Path) -> Path:
    ids = dota_train_ids()
    records = {v: _record("turning", k % 2 == 0, ["car"], v) for k, v in enumerate(ids)}
    path = tmp_path / constants.DOTA_LABEL_CAPTION_FILENAME
    path.write_text(json.dumps(dl.build_manifest(records)))
    return path


def test_manifest_covers_the_section55_clips_and_loads_for_dota_only(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    caps = load_caption_manifest(path, constants.DOTA_TRAIN_DATASET)
    assert sorted(caps) == dota_train_ids() and len(caps) == 3020
    assert not set(caps) & set(constants.DOTA_TRAIN_EXCLUDED)
    with pytest.raises(ValueError, match="cannot train"):
        load_caption_manifest(path, constants.NEXAR_DATASET)
    summary = dl.summary(json.loads(path.read_text()))
    assert summary["clips"] == 3020 and summary["distinct"] == 2


def test_manifest_of_another_id_set_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="§55 train set"):
        dl.build_manifest({"a_000001": _record("turning", True, ["car"])})
    path = _manifest(tmp_path)
    data = json.loads(path.read_text())
    data[constants.CAPTION_MANIFEST_SPLIT_KEY] = constants.CAPTION_SPLIT_NEXAR
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="bound to train split"):
        load_caption_manifest(path)


def test_a_nexar_manifest_without_a_split_key_still_binds_nexar(tmp_path: Path) -> None:
    ids = sorted(load_split(constants.V2_SPLIT_NEXAR_TRAIN))
    path = tmp_path / "captions.json"
    path.write_text(json.dumps({"captions": {ids[0]: "a car hits the ego vehicle"},
                                "train_ids_sha1": lines_sha1(ids)}))
    assert load_caption_manifest(path, constants.NEXAR_DATASET) == {
        ids[0]: "a car hits the ego vehicle"}
    with pytest.raises(ValueError, match="cannot train"):
        load_caption_manifest(path, constants.DOTA_TRAIN_DATASET)


def test_read_records_from_folder(tmp_path: Path) -> None:
    rec = _record("obstacle", True, [], "a_000001")
    (tmp_path / "a_000001.json").write_text(json.dumps(rec))
    assert dl.read_records(tmp_path, ["a_000001"]) == {"a_000001": rec}
