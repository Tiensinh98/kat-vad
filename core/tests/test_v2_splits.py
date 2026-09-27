"""Tests for the frozen KAT-VAD v2 decision splits (plan katvad-v2-e0-e2.md P0).

The failures that matter are silent: a DoTA video on both sides of the split,
a T2-val source that also has test windows, a sealed DoTA-eval list read by a
harness before the final report, or a split file edited after freezing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import constants
from core.data.v2_splits import (
    SealedSplitError,
    dota_group,
    grouped_split,
    lines_sha1,
    load_split,
    read_manifest,
    share_bin,
    split_path,
)
from core.tools import freeze_splits

N_TYPES = 3
TRAIN_SOURCES_PER_TYPE = 20
TEST_SOURCES_PER_TYPE = 4
N_DOTA_VIDEOS = 40
DOTA_CLIP_FRAMES = 100
SEED = 7


def _t2_meta() -> dict[str, dict[str, object]]:
    """T2-shaped meta: per source 2 abnormal + 1 normal window; test by source."""
    meta: dict[str, dict[str, object]] = {}
    for type_id in range(1, N_TYPES + 1):
        for video in range(TRAIN_SOURCES_PER_TYPE + TEST_SOURCES_PER_TYPE):
            source = f"t{type_id:02d}_v{video:03d}"
            split = "train" if video < TRAIN_SOURCES_PER_TYPE else "test"
            for w, positive in enumerate((5, 3, 0)):
                meta[f"{source}__w{w:03d}"] = {
                    "source": source, "split": split, "type": type_id,
                    "positive_frames": positive, "class_name": "CarAccident",
                }
    return meta


def _write_dota(root: Path) -> tuple[Path, Path]:
    """DoTA-shaped metadata: video i has 1 + i % 5 clips, shares across all bins."""
    metadata: dict[str, dict[str, object]] = {}
    for video in range(N_DOTA_VIDEOS):
        for clip in range(1 + video % 5):
            span = 10 + (video * 7 + clip * 13) % 85
            metadata[f"vid_{video:02d}_{clip * 100:06d}"] = {
                "anomaly_start": 5, "anomaly_end": 5 + span,
                "num_frames": DOTA_CLIP_FRAMES, "anomaly_class": "ego: turning",
            }
    metadata_path = root / "metadata_val.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    split = root / "val_split.txt"
    split.write_text("".join(f"{k}\n" for k in metadata), encoding="utf-8")
    return metadata_path, split


@pytest.fixture
def frozen(tmp_path: Path) -> Path:
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps(_t2_meta()), encoding="utf-8")
    metadata, split = _write_dota(tmp_path)
    out = tmp_path / "splits"
    freeze_splits.main([
        "--t2-meta", str(meta), "--dota-metadata", str(metadata),
        "--dota-split", str(split), "--out-dir", str(out), "--seed", str(SEED),
    ])
    return out


class TestHelpers:
    @pytest.mark.parametrize(("share", "label"), [
        (0.0, "<30"), (0.2999, "<30"), (0.3, "30-50"), (0.5, "50-70"),
        (0.6999, "50-70"), (0.7, ">70"), (1.0, ">70"),
    ])
    def test_share_bin_edges(self, share: float, label: str) -> None:
        assert share_bin(share) == label

    def test_share_bin_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="accident share"):
            share_bin(1.2)

    def test_dota_group_keeps_underscores_in_video_id(self) -> None:
        assert dota_group("J-j5e_Ig8I0_001514") == "J-j5e_Ig8I0"

    @pytest.mark.parametrize("bad", ["nounderscore", "_000123", "vid_abc"])
    def test_dota_group_rejects_non_clip_ids(self, bad: str) -> None:
        with pytest.raises(ValueError, match="DoTA clip id"):
            dota_group(bad)

    def test_lines_sha1_is_order_free(self) -> None:
        assert lines_sha1(["b", "a"]) == lines_sha1(["a", "b"])


class TestGroupedSplit:
    def _groups(self) -> tuple[dict[str, list[str]], dict[str, str]]:
        items = {f"g{g:02d}": [f"g{g:02d}_{i}" for i in range(1 + g % 4)] for g in range(30)}
        strata = {g: ("a" if int(g[1:]) % 2 else "b") for g in items}
        return items, strata

    def test_groups_never_straddle_and_cover_everything(self) -> None:
        items, strata = self._groups()
        selected, rest = grouped_split(items, strata, 0.5, SEED)
        assert not selected & rest
        assert selected | rest == {i for v in items.values() for i in v}
        for members in items.values():
            assert set(members) <= selected or set(members) <= rest

    def test_hits_the_item_fraction_and_is_deterministic(self) -> None:
        items, strata = self._groups()
        selected, _ = grouped_split(items, strata, 0.5, SEED)
        total = sum(len(v) for v in items.values())
        assert abs(len(selected) / total - 0.5) < 0.1
        assert grouped_split(items, strata, 0.5, SEED)[0] == selected

    def test_rejects_a_group_without_stratum(self) -> None:
        items, strata = self._groups()
        strata.pop("g00")
        with pytest.raises(ValueError, match="no stratum"):
            grouped_split(items, strata, 0.5, SEED)


class TestFreeze:
    def test_t2_val_is_train_only_and_type_stratified(self, frozen: Path) -> None:
        val = load_split(constants.V2_SPLIT_T2_VAL, frozen)
        meta = _t2_meta()
        train_sources = {m["source"] for m in meta.values() if m["split"] == "train"}
        assert set(val) <= train_sources
        per_type = {int(s[1:3]) for s in val}
        assert per_type == set(range(1, N_TYPES + 1))
        expected = round(TRAIN_SOURCES_PER_TYPE * constants.V2_T2_VAL_FRACTION) * N_TYPES
        assert len(val) == expected

    def test_dota_dev_and_eval_are_video_disjoint(self, frozen: Path) -> None:
        dev = load_split(constants.V2_SPLIT_DOTA_DEV, frozen)
        rest = load_split(constants.V2_SPLIT_DOTA_EVAL, frozen, final=True)
        assert not set(dev) & set(rest)
        assert not {dota_group(c) for c in dev} & {dota_group(c) for c in rest}

    def test_refuses_to_overwrite_a_frozen_split(self, frozen: Path, tmp_path: Path) -> None:
        with pytest.raises(FileExistsError, match="frozen"):
            freeze_splits.main([
                "--t2-meta", str(tmp_path / "meta.json"),
                "--dota-metadata", str(tmp_path / "metadata_val.json"),
                "--dota-split", str(tmp_path / "val_split.txt"),
                "--out-dir", str(frozen), "--seed", str(SEED),
            ])

    def test_check_passes_then_fails_after_a_different_seed(
        self, frozen: Path, tmp_path: Path
    ) -> None:
        args = [
            "--t2-meta", str(tmp_path / "meta.json"),
            "--dota-metadata", str(tmp_path / "metadata_val.json"),
            "--dota-split", str(tmp_path / "val_split.txt"),
            "--out-dir", str(frozen), "--check",
        ]
        freeze_splits.main([*args, "--seed", str(SEED)])
        with pytest.raises(ValueError, match="differs"):
            freeze_splits.main([*args, "--seed", str(SEED + 1)])


class TestLoadGuard:
    def test_dota_eval_is_sealed_by_default(self, frozen: Path) -> None:
        with pytest.raises(SealedSplitError, match="sealed"):
            load_split(constants.V2_SPLIT_DOTA_EVAL, frozen)

    def test_an_edited_split_file_is_refused(self, frozen: Path) -> None:
        path = split_path(constants.V2_SPLIT_DOTA_DEV, frozen)
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("".join(f"{x}\n" for x in lines[1:]), encoding="utf-8")
        with pytest.raises(ValueError, match="edited after freezing"):
            load_split(constants.V2_SPLIT_DOTA_DEV, frozen)

    def test_unknown_split_name(self, frozen: Path) -> None:
        with pytest.raises(KeyError, match="unknown v2 split"):
            load_split("dota_test", frozen)


class TestCommittedSplits:
    """The frozen files in the repo: intact, disjoint, and the sizes the plan records."""

    def test_committed_splits_are_intact_and_disjoint(self) -> None:
        manifest = read_manifest()
        dev = load_split(constants.V2_SPLIT_DOTA_DEV)
        rest = load_split(constants.V2_SPLIT_DOTA_EVAL, final=True)
        assert len(dev) + len(rest) == manifest["dota"]["clips_total"]
        assert not {dota_group(c) for c in dev} & {dota_group(c) for c in rest}
        assert len(load_split(constants.V2_SPLIT_T2_VAL)) == manifest["t2_val"]["val_sources"]
