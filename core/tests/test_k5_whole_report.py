"""The K5-whole vs K5-window table (``KIM_RUN.md`` §16; data-free)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from core import constants
from core.tests.test_k5_zs_report import CKPT, _layout
from core.tools import k5_whole_report, nexar_eval, protocol_b_eval

K5, WHOLE = k5_whole_report.K5, k5_whole_report.WHOLE
PLACEMENTS = constants.NEXAR_CROP_PLACEMENTS
WHOLE_CKPT = "/drive/kim_whole/K5/s2024/checkpoint_last.pt"
CI = {"mean": 0.7, "low": 0.6, "high": 0.8}


def _nexar_row(d: Path, auc: float, ckpt: str, clip_auc: float = 0.7,
               split: str = constants.V2_SPLIT_NEXAR_VAL) -> None:
    d.mkdir(parents=True)
    (d / nexar_eval.ITEM_AUCS_JSON).write_text(json.dumps(
        {f"v{i}{nexar_eval.CROP_SEP}{p}": auc + 0.01 * i for i in range(20) for p in PLACEMENTS}))
    crops = {str(p): {"macro": {**CI, "mean": auc - abs(p - 0.5) * 0.2}} for p in PLACEMENTS}
    (d / nexar_eval.READOUT_JSON).write_text(json.dumps({
        "split": split, "videos": 225, "checkpoints": {"s2024": ckpt},
        "crops": {**crops, "mean_over_placements": {"macro": {**CI, "mean": auc}}},
        "whole": {"macro": CI, "micro": 0.7, "micro_ap": 0.2, "clip_auc_max": clip_auc,
                  "positive_frame_share": 0.03, "middle_ruler_macro": CI},
    }))


def _rewrite(run_root: Path, old: str, new: str) -> None:
    for name in ("results.json", protocol_b_eval.READOUT_JSON, "dada_readout.json"):
        for path in run_root.rglob(name):
            path.write_text(path.read_text().replace(old, new))


def _roots(root: Path, whole_crops: float = 0.6, whole_clip_auc: float = 0.7) -> argparse.Namespace:
    _layout(root)
    kim, whole = root / "kim", root / "kim_whole"
    shutil.copytree(kim, whole)
    _rewrite(whole, CKPT, WHOLE_CKPT)
    _nexar_row(root / "k5_nexar", 0.9, CKPT)
    _nexar_row(whole / "eval" / "nexar" / WHOLE, whole_crops, WHOLE_CKPT, whole_clip_auc)
    return k5_whole_report.build_arg_parser().parse_args(
        ["--kim-root", str(kim), "--whole-root", str(whole),
         "--k5-nexar-dir", str(root / "k5_nexar"), "--out-dir", str(root / "out")])


class TestReport:
    def test_five_sets_two_rows(self, tmp_path: Path) -> None:
        report = k5_whole_report.run(_roots(tmp_path))
        assert list(report["sets"]) == ["nexar", "dota", "dada", "tad", "dad"]
        for s in report["sets"].values():
            assert set(s["rows"]) == {K5, WHOLE}
        nexar = report["sets"]["nexar"]
        assert nexar["delta"]["mean"] == pytest.approx(-0.3)
        assert nexar["sentence"] == "below"
        # same saved scores on both sides: no difference on the zero-shot sets
        assert report["sets"]["dota"]["delta"]["mean"] == pytest.approx(0.0)
        assert report["follow_up"]["may_propose_s2025"] is False
        text = (tmp_path / "out" / "k5_whole_report.md").read_text()
        assert "decides nothing" in text and "| **DADA-2000** | AUC macro |" in text
        assert "Claude may propose s2025: **no**" in text and "W0" not in text

    def test_edge_drop_and_clip_flag(self, tmp_path: Path) -> None:
        report = k5_whole_report.run(_roots(tmp_path, whole_clip_auc=0.97))
        rows = report["sets"]["nexar"]["rows"]
        assert rows[WHOLE]["edge_drop"] == pytest.approx(0.08)
        assert rows[WHOLE]["clip_classifier_flag"] is True
        assert rows[K5]["clip_classifier_flag"] is False

    def test_pilot_is_printed_beside(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        _nexar_row(tmp_path / "pilot", 0.58, "/drive/v2/pilot.pt")
        args.pilot_dir = tmp_path / "pilot"
        report = k5_whole_report.run(args)
        assert report["pilot"]["macro_crops"]["mean"] == pytest.approx(0.58)
        assert "other host" in (tmp_path / "out" / "k5_whole_report.md").read_text()

    def test_follow_up_needs_crops_and_three_sets(self) -> None:
        above, ns = {"sentence": "above"}, {"sentence": "not separated"}
        sets = {"nexar": above, "dota": above, "dada": above, "tad": ns, "dad": ns}
        assert k5_whole_report.follow_up({"sets": sets})["may_propose_s2025"] is False
        sets["tad"] = above
        assert k5_whole_report.follow_up({"sets": sets})["may_propose_s2025"] is True
        sets["nexar"] = ns
        assert k5_whole_report.follow_up({"sets": sets})["may_propose_s2025"] is False


class TestRefusals:
    def test_shared_checkpoint(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        _rewrite(tmp_path / "kim_whole", WHOLE_CKPT, CKPT)
        path = tmp_path / "kim_whole" / "eval" / "nexar" / WHOLE / nexar_eval.READOUT_JSON
        path.write_text(path.read_text().replace(WHOLE_CKPT, CKPT))
        with pytest.raises(ValueError, match="share a checkpoint"):
            k5_whole_report.run(args)
        assert not (tmp_path / "out").exists()

    def test_two_checkpoints_in_one_row(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        path = tmp_path / "kim_whole" / "eval" / "nexar" / WHOLE / nexar_eval.READOUT_JSON
        path.write_text(path.read_text().replace(WHOLE_CKPT, "/drive/other.pt"))
        with pytest.raises(ValueError, match="different checkpoints"):
            k5_whole_report.run(args)

    def test_nexar_test_is_refused(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        d = tmp_path / "kim_whole" / "eval" / "nexar" / WHOLE
        shutil.rmtree(d)
        _nexar_row(d, 0.6, WHOLE_CKPT, split=constants.V2_SPLIT_NEXAR_TEST)
        with pytest.raises(ValueError, match="nexar_val"):
            k5_whole_report.run(args)

    def test_different_crops_are_refused(self, tmp_path: Path) -> None:
        args = _roots(tmp_path)
        path = tmp_path / "kim_whole" / "eval" / "nexar" / WHOLE / nexar_eval.ITEM_AUCS_JSON
        items = json.loads(path.read_text())
        path.write_text(json.dumps({k: v for k, v in items.items() if not k.startswith("v0")}))
        with pytest.raises(ValueError, match="different crops"):
            k5_whole_report.run(args)
