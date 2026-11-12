"""Tests for ``core.tools.k5_baseline_micro`` (``KIM_RUN.md`` §21)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import k5_baseline_micro as bm
from core.tools import tad_clip_gate as tg
from core.tools import tadz_readout as td
from core.tools.protocol_b_eval import write_clip_scores

N_VIDEOS = 12
N_FRAMES = 20


def _toy(offsets: bool = True) -> tuple[bm.Curves, bm.Curves]:
    """Every video localizes perfectly, but each sits at its own level: micro < macro = 1."""
    rng = np.random.default_rng(0)
    scores: bm.Curves = {}
    labels: bm.Curves = {}
    for i in range(N_VIDEOS):
        gt = np.zeros(N_FRAMES, dtype=np.int64)
        gt[8:14] = 1
        level = 0.3 * i if offsets else 0.0
        scores[f"v{i:02d}"] = level + 0.05 * rng.random(N_FRAMES) + 0.5 * gt
        labels[f"v{i:02d}"] = gt
    return scores, labels


def _write(tmp: Path, scores: bm.Curves, labels: bm.Curves, **recorded: object
           ) -> tuple[Path, Path]:
    npz = tmp / "clip_scores.npz"
    write_clip_scores(npz, scores, labels)
    readout = tmp / "readout.json"
    readout.write_text(json.dumps({"macro": {"mean": bm.macro(scores, labels)}, **recorded}))
    return npz, readout


class TestTransforms:
    def test_every_transform_keeps_the_macro(self) -> None:
        scores, labels = _toy()
        want = bm.macro(scores, labels)
        for name in bm.ROW_ORDER:
            assert bm.macro(bm.transform_curves(scores, labels, name), labels) == pytest.approx(
                want, abs=constants.BASEMICRO_INVARIANCE_TOL)

    def test_per_video_calibration_repairs_a_pure_offset(self) -> None:
        scores, labels = _toy()
        assert td.micro(scores, labels) < 0.8
        for name in (bm.MINMAX, bm.MINUS_MEDIAN, bm.RANK, bm.ORACLE_OFFSET, bm.ORACLE_Z):
            out = bm.transform_curves(scores, labels, name)
            assert td.micro(out, labels) > 0.99, name

    def test_oracle_needs_a_normal_frame(self) -> None:
        scores = {"a": np.array([0.1, 0.9])}
        with pytest.raises(ValueError, match="normal frame"):
            bm.transform_curves(scores, {"a": np.array([1, 1])}, bm.ORACLE_OFFSET)

    def test_unknown_transform_raises(self) -> None:
        scores, labels = _toy()
        with pytest.raises(ValueError, match="unknown transform"):
            bm.transform_curves(scores, labels, "nope")


class TestDecomposition:
    def test_micro_is_the_pair_weighted_mix(self) -> None:
        scores, labels = _toy()
        within = bm.within_video(scores, labels)
        micro = td.micro(scores, labels)
        cross = bm.cross_video_auc(micro, within)
        assert cross is not None
        mix = (within["concordant_within"]
               + cross * (within["pairs_total"] - within["pairs_within"])) / within["pairs_total"]
        assert mix == pytest.approx(micro, abs=1e-12)
        assert within["concordant_within"] / within["pairs_within"] == pytest.approx(1.0)
        assert cross < 0.7  # offsets of 0.3 per video against a 0.5 gap: well below within = 1

    def test_a_transform_changes_only_the_cross_video_part(self) -> None:
        scores, labels = _toy()
        within = bm.within_video(scores, labels)
        ranked = bm.transform_curves(scores, labels, bm.RANK)
        assert bm.within_video(ranked, labels)["concordant_within"] == pytest.approx(
            within["concordant_within"])

    def test_baseline_spread_reads_the_level_of_normal_frames(self) -> None:
        scores, labels = _toy()
        spread = bm.baseline_spread(scores, labels)
        assert spread["normal_level_std"] > spread["mean_gap"] * 0.5
        flat_scores, flat_labels = _toy(offsets=False)
        assert bm.baseline_spread(flat_scores, flat_labels)["ratio"] < 0.2


class TestBootstrapClusters:
    def test_each_video_its_own_cluster_equals_no_groups(self, monkeypatch: pytest.MonkeyPatch
                                                         ) -> None:
        monkeypatch.setattr(constants, "V2_E1_BOOTSTRAP", 100)
        scores, labels = _toy()
        other = bm.transform_curves(scores, labels, bm.RANK)
        curves = {"a": scores, "b": other}
        pairs = {"d": ("b", "a")}
        plain = tg.micro_bootstrap_pairs(curves, pairs, labels, 5)
        grouped = tg.micro_bootstrap_pairs(curves, pairs, labels, 5,
                                           groups={v: v for v in labels})
        assert plain == grouped

    def test_clusters_move_together_and_draws_are_honoured(self) -> None:
        scores, labels = _toy()
        other = bm.transform_curves(scores, labels, bm.RANK)
        groups = {v: f"g{int(v[1:]) // 4}" for v in labels}  # 3 clusters of 4
        out = tg.micro_bootstrap_pairs({"a": scores, "b": other}, {"d": ("b", "a")}, labels, 5,
                                       groups=groups, draws=50)
        assert out["d"]["mean"] > 0.1
        assert out["d"]["low"] <= out["d"]["high"]


class TestReadSet:
    def _read(self, tmp_path: Path, as_read: str = "raw", cluster: str = "item", **recorded: object
              ) -> dict[str, object]:
        scores, labels = _toy()
        npz, readout = _write(tmp_path, scores, labels, **recorded)
        return bm.read_set("toy", npz, as_read, cluster, readout)

    def test_report_rows_and_headroom(self, monkeypatch: pytest.MonkeyPatch,
                                      tmp_path: Path) -> None:
        monkeypatch.setattr(constants, "BASEMICRO_BOOTSTRAP_DRAWS", 100)
        entry = self._read(tmp_path)
        rows = entry["rows"]  # type: ignore[index]
        assert set(rows) == set(bm.ROW_ORDER)
        assert rows[bm.RAW]["delta_vs_as_read"] is None
        assert rows[bm.MINMAX]["sentence"] == "above"
        assert entry["headroom_sentence"] == "shown"
        assert rows[bm.ORACLE_Z]["share_of_headroom"] == pytest.approx(1.0)

    def test_dota_clusters_are_the_source_videos(self, monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
        monkeypatch.setattr(constants, "BASEMICRO_BOOTSTRAP_DRAWS", 50)
        scores, labels = _toy()
        renamed = {f"src{int(v[1:]) // 4}_{int(v[1:]):06d}": s for v, s in scores.items()}
        relabel = {f"src{int(v[1:]) // 4}_{int(v[1:]):06d}": g for v, g in labels.items()}
        npz, readout = _write(tmp_path, renamed, relabel)
        entry = bm.read_set("toy", npz, "minmax", "dota", readout)
        assert entry["clusters"] == 3

    def test_g1_refuses_scores_that_are_not_the_recorded_read(self, tmp_path: Path) -> None:
        scores, labels = _toy()
        npz, readout = _write(tmp_path, scores, labels)
        readout.write_text(json.dumps({"macro": {"mean": 0.5}}))
        with pytest.raises(ValueError, match="G1"):
            bm.read_set("toy", npz, "raw", "item", readout)

    def test_g1_checks_the_recorded_micro_when_there_is_one(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="G1"):
            self._read(tmp_path, micro=0.123)

    def test_single_class_videos_are_left_out_and_counted(self, monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
        monkeypatch.setattr(constants, "BASEMICRO_BOOTSTRAP_DRAWS", 50)
        scores, labels = _toy()
        scores["extra"] = np.linspace(0, 1, N_FRAMES)
        labels["extra"] = np.ones(N_FRAMES, dtype=np.int64)
        npz = tmp_path / "clip_scores.npz"
        write_clip_scores(npz, scores, labels)
        readout = tmp_path / "readout.json"
        readout.write_text(json.dumps({"macro": {"mean": bm.macro(scores, labels)}}))
        entry = bm.read_set("toy", npz, "raw", "item", readout)
        assert entry["excluded_single_class"] == 1 and entry["videos"] == N_VIDEOS


def test_cli_writes_the_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(constants, "BASEMICRO_BOOTSTRAP_DRAWS", 50)
    scores, labels = _toy()
    npz, readout = _write(tmp_path, scores, labels)
    bm.main(["--set", "TOY/K5", str(npz), "raw", "item", str(readout),
             "--set", "TOY2", str(npz), "minmax", "item", str(readout),
             "--out-dir", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out" / "k5_baseline_micro.json").read_text())
    assert set(report["sets"]) == {"TOY/K5", "TOY2"} and report["draws"] == 50
    md = (tmp_path / "out" / "k5_baseline_micro.md").read_text()
    assert "protocol change, printed only" in md and "cross-video AUC" in md
