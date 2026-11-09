"""Tests for ``core.tools.normal_text_probe`` (KIM_RUN.md §15); no model, no data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest import mock

import pytest
import torch

from core import constants
from core.data.definitions import DATASET_CLS_DEFS, item_verbalizer, verbalize_class_name
from core.tools import normal_text_probe as ntp

CLASSES = [constants.NORMAL_PROBE_CLASS, "CarAccident"]


def _defs() -> dict[str, list[str]]:
    return DATASET_CLS_DEFS[constants.NORMAL_PROBE_VERBALIZER_KEY]


class TestPrefusionWeight:
    def test_init_is_about_092(self) -> None:
        w = ntp.prefusion_weight(constants.ADAPTIVE_FUSE_ALPHA0)
        assert w == pytest.approx(1 / (1 + math.exp(-2.5)))
        assert 0.92 < w < 0.93

    def test_zero_alpha_is_half(self) -> None:
        assert ntp.prefusion_weight(0.0) == pytest.approx(0.5)


class TestSwap:
    @staticmethod
    def _draws(item: str) -> list[list[str]]:
        """Three windows' class texts of one item, as the eval tools draw them."""
        v = item_verbalizer("dota", item)
        return [[verbalize_class_name(v, c) for c in CLASSES] for _ in range(3)]

    def test_dota_and_dada_share_the_dict(self) -> None:
        assert DATASET_CLS_DEFS["dota"] is DATASET_CLS_DEFS["dada"]

    def test_original_is_no_patch(self) -> None:
        assert ntp.normal_override(ntp.ORIGINAL) == {}

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            ntp.normal_override("highway")

    def test_same_length_as_original(self) -> None:
        patch = ntp.normal_override(ntp.DASHCAM)[constants.NORMAL_PROBE_CLASS]
        assert len(patch) == len(_defs()[constants.NORMAL_PROBE_CLASS])
        assert not any("surveillance" in s.lower() for s in patch)

    def test_only_the_normal_sentence_changes(self) -> None:
        """Same item -> same indices: CarAccident identical, Normal at the same position."""
        original_normal = list(_defs()[constants.NORMAL_PROBE_CLASS])
        dashcam = list(constants.NORMAL_PROBE_DASHCAM_DEFS)
        for item in (f"clip_{i:03d}" for i in range(40)):
            before = self._draws(item)
            with mock.patch.dict(_defs(), ntp.normal_override(ntp.DASHCAM)):
                after = self._draws(item)
            for b, a in zip(before, after, strict=True):
                assert a[1] == b[1]
                assert dashcam.index(a[0]) == original_normal.index(b[0])

    def test_score_patches_then_restores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        def fake(argv: list[str] | None) -> None:
            seen["normal"] = list(_defs()[constants.NORMAL_PROBE_CLASS])
            seen["argv"] = argv

        original = list(_defs()[constants.NORMAL_PROBE_CLASS])
        monkeypatch.setitem(ntp.TOOLS, ntp.DOTA, fake)
        ntp.run_score(ntp.DOTA, ntp.DASHCAM, ["--out-dir", "x"])
        assert seen["normal"] == list(constants.NORMAL_PROBE_DASHCAM_DEFS)
        assert seen["argv"] == ["--out-dir", "x"]
        assert _defs()[constants.NORMAL_PROBE_CLASS] == original

    def test_score_restores_after_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(argv: list[str] | None) -> None:
            raise RuntimeError("tool failed")

        original = list(_defs()[constants.NORMAL_PROBE_CLASS])
        monkeypatch.setitem(ntp.TOOLS, ntp.DADA, boom)
        with pytest.raises(RuntimeError):
            ntp.run_score(ntp.DADA, ntp.DASHCAM, [])
        assert _defs()[constants.NORMAL_PROBE_CLASS] == original

    def test_main_strips_double_dash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        got: dict[str, list[str] | None] = {}
        monkeypatch.setitem(ntp.TOOLS, ntp.DOTA, lambda argv: got.update(argv=argv))
        ntp.main(["score", "--tool", "dota", "--normal", "original", "--", "--split", "dev"])
        assert got["argv"] == ["--split", "dev"]


class TestDecide:
    @staticmethod
    def _ci(low: float) -> dict[str, float]:
        return {"mean": low + 0.01, "low": low, "high": low + 0.02}

    def test_null(self) -> None:
        d = {"dota": {"s2024": 0.002, "s2025": -0.004}, "dada": {"s2024": 0.0, "s2025": 0.006}}
        assert ntp.decide(d, {"dota": None, "dada": None}) == ntp.NULL

    def test_cost_wins(self) -> None:
        d = {"dota": {"s2024": 0.03, "s2025": 0.03}, "dada": {"s2024": -0.01, "s2025": 0.02}}
        assert ntp.decide(d, {"dota": self._ci(0.01), "dada": self._ci(0.01)}) == ntp.COST

    def test_gain_needs_every_seed_and_ci(self) -> None:
        d = {"dota": {"s2024": 0.02, "s2025": 0.015}, "dada": {"s2024": 0.011, "s2025": 0.03}}
        assert ntp.decide(d, {"dota": self._ci(0.005), "dada": self._ci(0.001)}) == ntp.GAIN
        assert ntp.decide(d, {"dota": self._ci(0.005), "dada": self._ci(-0.001)}) == ntp.MIXED
        d["dada"]["s2025"] = 0.005
        assert ntp.decide(d, {"dota": self._ci(0.005), "dada": self._ci(0.001)}) == ntp.MIXED

    def test_missing_set_raises(self) -> None:
        with pytest.raises(ValueError):
            ntp.decide({"dota": {"s2024": 0.0}}, {"dota": None})


class TestGate:
    def test_within_tol_passes(self) -> None:
        ntp.check_gate("dota", "s2024", 0.8195, 0.8195 + constants.NORMAL_PROBE_REPRO_TOL / 2)

    def test_outside_tol_raises(self) -> None:
        with pytest.raises(ValueError, match="recorded K5 read"):
            ntp.check_gate("dada", "s2024", 0.7740, 0.7750)


def _write_read(d: Path, set_name: str, aucs: dict[str, float]) -> Path:
    d.mkdir(parents=True)
    macro = sum(aucs.values()) / len(aucs)
    (d / ntp.READOUT_FILES[set_name]).write_text(json.dumps({"macro": {"mean": macro}}))
    (d / ntp.CLIP_AUCS).write_text(json.dumps(aucs))
    return d


class TestEndToEnd:
    def test_alpha(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt.pt"
        torch.save({"global_step": 1749,
                    "model": {constants.NORMAL_PROBE_ADP_ALPHA_KEY: torch.tensor([0.3])}}, ckpt)
        r = ntp.run_alpha([["s2024", str(ckpt)]], tmp_path / "alpha.json")
        assert r["runs"]["s2024"]["w_prefusion"] == pytest.approx(ntp.prefusion_weight(0.3))
        assert r["text_branch_minor"] is True
        saved = json.loads((tmp_path / "alpha.json").read_text())
        assert saved["runs"]["s2024"]["global_step"] == 1749

    def test_alpha_refuses_non_adaptive(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt.pt"
        torch.save({"model": {}}, ckpt)
        with pytest.raises(KeyError):
            ntp.run_alpha([["s2024", str(ckpt)]], tmp_path / "alpha.json")

    def test_report(self, tmp_path: Path) -> None:
        alpha = tmp_path / "alpha.json"
        alpha.write_text(json.dumps({"init_alpha": 0.25, "init_w_prefusion": 0.92, "minor_bar": 0.9,
                                     "runs": {}, "text_branch_minor": True}))
        dota_ids = [f"vid{i:02d}_{j:06d}" for i in range(6) for j in range(3)]
        dada_ids = [f"t01_v{i:03d}" for i in range(20)]
        reads = []
        for set_name, ids in (("dota", dota_ids), ("dada", dada_ids)):
            base = {v: 0.6 + 0.01 * (k % 7) for k, v in enumerate(ids)}
            dash = {v: a + 0.002 for v, a in base.items()}
            for seed in ("s2024", "s2025"):
                root = tmp_path / set_name / seed
                reads.append([set_name, seed, str(_write_read(root / "rec", set_name, base)),
                              str(_write_read(root / "orig", set_name, base)),
                              str(_write_read(root / "dash", set_name, dash))])
        r = ntp.run_report(alpha, reads, tmp_path / "out")
        assert r["decision"]["verdict"] == ntp.NULL
        assert r["sets"]["dada"]["s2024"]["delta"]["mean"] == pytest.approx(0.002)
        assert (tmp_path / "out" / f"{ntp.READOUT_NAME}.md").read_text().count("pooled") == 2

    def test_report_gate_raises(self, tmp_path: Path) -> None:
        alpha = tmp_path / "alpha.json"
        alpha.write_text("{}")
        ids = {f"t01_v{i:03d}": 0.7 for i in range(5)}
        moved = dict.fromkeys(ids, 0.8)
        read = ["dada", "s2024", str(_write_read(tmp_path / "rec", "dada", ids)),
                str(_write_read(tmp_path / "orig", "dada", moved)),
                str(_write_read(tmp_path / "dash", "dada", ids))]
        with pytest.raises(ValueError, match="recorded K5 read"):
            ntp.run_report(alpha, [read], tmp_path / "out")
