"""K5f / K5dfi / L0f / L0dfi on MSAD-full (``KIM_RUN.md`` §41; data-free)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from core import constants
from core.config import Config, load_config
from core.tools import msad_train as mt

REPO = Path(__file__).resolve().parents[2]
NEXAR_PARENTS = {
    "K5f": REPO / "outputs/v_final/kim_fixed/K5f/K5/s2024/config.yaml",
    "K5dfi": REPO / "outputs/v_final/kim_dvs_ignore/K5dfi/K5/s2024/config.yaml",
    "L0dfi": REPO / "outputs/v_final/kim_lagovad_nexar/L0dfi/K5/s2024/config.yaml",
}


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _row_config(arm: str, kernel: int) -> dict[str, Any]:
    cfg = load_config(None, mt.train_sets(arm, kernel))
    return _flat(cfg.to_dict())


class TestRows:
    def test_roundtrip_and_order(self) -> None:
        assert mt.parse_row(mt.row_name("K5dfi", 9)) == ("K5dfi", 9)
        rows = mt.all_rows()
        assert len(rows) == 8 and rows[0] == "K5fM3" and rows[-1] == "L0dfiM9"

    @pytest.mark.parametrize("bad", ["K5f", "K5fMx", "X5fM3", "M3"])
    def test_parse_refuses(self, bad: str) -> None:
        with pytest.raises(ValueError):
            mt.parse_row(bad)

    def test_unknown_arm_refused(self) -> None:
        with pytest.raises(ValueError):
            mt.row_name("K5", 3)
        with pytest.raises(ValueError):
            mt.train_sets("K5", 3)


class TestPlan:
    def test_msad_is_v1_budget(self) -> None:
        plan = mt.epoch_plan(constants.MSAD_TRAIN_ABNORMAL)
        assert (plan["steps_per_epoch"], plan["steps"]) == (4, 500)  # RESULTS_MSAD.md: 125 ep = 500

    def test_nexar_k5f_plan_matches_its_record(self) -> None:
        assert mt.epoch_plan(1689, epochs=33)["steps"] == 1749  # K5's J10 step, k5_fixed.ipynb


class TestRecipes:
    def test_no_duplicate_keys(self) -> None:
        for row in mt.all_rows():
            keys = [s.split("=")[0] for s in mt.train_sets(*mt.parse_row(row))]
            assert len(keys) == len(set(keys)), row

    @pytest.mark.parametrize("kernel", constants.KIM_MSAD_TRAIN_KERNELS)
    def test_every_row_is_a_valid_config(self, kernel: int) -> None:
        for arm in constants.KIM_MSAD_TRAIN_ARMS:
            c = _row_config(arm, kernel)
            assert c["data.dataset"] == constants.MSAD_FULL_DATASET
            assert c["loss.mil_topk_pct"] == 16 and c["train.num_epochs"] == 125
            assert c["model.score_head_kernel"] == kernel
            assert c["kip.enabled"] is False and c["data.is_egocentric"] is False
            assert c["kim.enabled"] is (arm in mt.MOTION_ARMS)
            assert (c["v2.motion"] == constants.V5_ENCODER) is (arm in mt.MOTION_ARMS)
            assert (c["dvs.theta"] == 0.7) is arm.endswith("dfi")
            assert c["loss.dvs_anchor_mode"] == ("ignore" if arm.endswith("dfi") else "span")
        assert _row_config("K5dfi", kernel)["kim.seam_mask"] is True
        assert _row_config("K5f", kernel)["kim.seam_mask"] is False

    @pytest.mark.skipif(not all(p.is_file() for p in NEXAR_PARENTS.values()),
                        reason="the Nexar rows' saved configs are not in this checkout")
    @pytest.mark.parametrize("arm", constants.KIM_MSAD_TRAIN_ARMS)
    @pytest.mark.parametrize("kernel", constants.KIM_MSAD_TRAIN_KERNELS)
    def test_rows_differ_from_the_nexar_rows_only_in_the_corpus_keys(self, arm: str,
                                                                      kernel: int) -> None:
        parent = _flat(yaml.safe_load(NEXAR_PARENTS[mt.PARENT[arm]].read_text()))
        defaults = _flat(Config().to_dict())
        diff = mt.check_config(mt.row_name(arm, kernel), parent, _row_config(arm, kernel), defaults)
        extra = mt.L0F_EXTRA_KEYS if arm == "L0f" else frozenset()
        assert set(diff) <= mt.CORPUS_KEYS | extra

    def test_check_config_refuses_a_stray_key_and_a_wrong_value(self) -> None:
        defaults = _flat(Config().to_dict())
        got = _row_config("K5f", 3)
        nexar = {"loss.mil_topk_pct": 5, "train.num_epochs": 33, "data.dataset": "Nexar"}
        parent = dict(got, **nexar)
        assert set(mt.check_config("K5fM3", parent, got, defaults)) == {
            "loss.mil_topk_pct", "train.num_epochs", "data.dataset"}
        with pytest.raises(ValueError, match="differs from its Nexar parent"):
            mt.check_config("K5fM3", dict(parent, **{"dvs.theta": 0.7}), got, defaults)
        with pytest.raises(ValueError, match="config values"):
            mt.check_config("K5fM9", parent, got, defaults)  # trained with kernel 3, not 9

    def test_config_diff_tolerates_a_key_added_after_the_parent_trained(self) -> None:
        defaults = {"a": 1, "b": 2}
        assert mt.config_diff({"a": 1}, {"a": 1, "b": 2}, defaults) == {}
        assert mt.config_diff({"a": 1}, {"a": 1, "b": 3}, defaults) == {"b": (None, 3)}


IDS = {"Traffic_accident_1": 12, "Traffic_accident_2": 10, "Fire_1": 14,
       "MSAD_normal_testing_1": 30, "MSAD_normal_testing_2": 20}


def _labels(vid: str, n: int) -> np.ndarray:
    g = np.zeros(n, dtype=np.int64)
    if not vid.startswith(constants.MSAD_ZS_NORMAL_PREFIX):
        g[n // 3: 2 * n // 3] = 1
    return g


def _write(d: Path, shift: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    (d / "scores").mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"auc": 0.5}))
    for vid, n in IDS.items():
        g = _labels(vid, n)
        np.savez(d / "scores" / f"{vid}.npz", score=rng.random(n) + shift * g, gt=g)


class TestReport:
    def _root(self, tmp_path: Path, rows: dict[str, float]) -> Path:
        for i, (row, shift) in enumerate(rows.items()):
            _write(tmp_path / "eval" / row / "s2024", shift, seed=i)
        return tmp_path

    @pytest.fixture(autouse=True)
    def _small(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mt.kmr, "TWO_CLASS", {mt.kmr.TRAFFIC: 2, mt.kmr.FULL: 3})

    def test_contrast_pairs_need_both_rows(self) -> None:
        assert mt.contrast_pairs({"K5fM3"}) == []
        labels = {p[0] for p in mt.contrast_pairs({"K5fM3", "L0fM3", "K5fM9"})}
        assert labels == {"motion + KIM, DVS off (k3)",
                          "kernel 9 - 3, K5f (printed, never a selection)"}

    def test_report_prints_every_row_and_no_selection(self, tmp_path: Path) -> None:
        root = self._root(tmp_path, {"K5fM3": 0.9, "L0fM3": 0.0, "K5fM9": 0.5, "K5f": 0.2})
        report = mt.run_report(root, 2024, None)
        full = report["sets"]["MSAD-full"]
        assert set(full["rows"]) == {"K5fM3", "L0fM3", "K5fM9", "K5f"}
        assert full["videos"] == 5 and full["two_class"] == 3
        by = {c["label"]: c for c in full["contrasts"]}
        motion = by["motion + KIM, DVS off (k3)"]
        assert motion["a"] == "K5fM3" and motion["delta"]["mean"] > 0
        text = (root / "REPORTS" / f"{constants.KIM_MSAD_TRAIN_REPORT_STEM}.md").read_text()
        assert "PRELIMINARY (n = 1)" in text and "never a selection" in text
        assert "best" not in text.lower().replace("never write \"the better kernel\"", "")
        assert "TAD" not in report["sets"]

    def test_tad_set_is_read_when_present(self, tmp_path: Path) -> None:
        root = self._root(tmp_path, {"K5fM3": 0.9, "L0fM3": 0.0})
        for i, row in enumerate(("K5fM3", "L0fM3")):
            _write(root / "eval_tad" / row / "s2024", 0.5 * i, seed=10 + i)
        report = mt.run_report(root, 2024, tmp_path / "out")
        assert report["sets"]["TAD"]["videos"] == 5
        assert (tmp_path / "out" / f"{constants.KIM_MSAD_TRAIN_REPORT_STEM}.json").is_file()

    def test_missing_reads_raise(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            mt.run_report(tmp_path, 2024, None)
