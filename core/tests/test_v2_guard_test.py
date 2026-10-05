"""Amendment 8 Q4-Q5: O1' on T2-test and the per-arm verdict (model-free except the smoke)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.models.kat_vad import KATVAD
from core.tests.test_v2_dataset import _build
from core.tools import v2_guard_test as gt

SMALL = 32


def _g(micro: float, macro: float, window: float, passed: bool) -> dict:
    return {"micro": micro, "macro": macro, "window_level_auc": window, "pass": passed,
            "clip_oracle_micro": 0.6986}


# The pilot's T2-val read-outs (batch1/2_readout.md).
VAL = {
    "A0": _g(0.6586, 0.6737, 0.6863, True),
    "A1": _g(0.6683, 0.6705, 0.6828, True),
    "A2": _g(0.7120, 0.6882, 0.7767, False),
    "A3": _g(0.7161, 0.7052, 0.7637, False),
}
MOTION = {"A0": False, "A1": False, "A2": True, "A3": True}


class TestVerdict:
    def test_o1_survivors_and_o1_prime_rescues(self) -> None:
        test = {"A0": _g(0.63, 0.64, 0.66, True), "A1": _g(0.64, 0.64, 0.66, True),
                "A2": _g(0.68, 0.65, 0.74, False), "A3": _g(0.69, 0.67, 0.73, False)}
        out = gt.verdict(test, VAL, "A0", MOTION)
        assert out["arms"]["A0"] == {"reference": True, "survivor": True}
        assert out["arms"]["A1"]["by"] == "O1"
        assert out["arms"]["A2"]["survivor"] and out["arms"]["A2"]["by"].startswith("O1'")
        assert not out["motion_stream_dropped"]

    def test_collapse_on_t2_test_alone_drops_the_arm(self) -> None:
        test = {"A0": _g(0.63, 0.64, 0.66, True), "A1": _g(0.64, 0.64, 0.66, True),
                "A2": _g(0.68, 0.60, 0.80, False), "A3": _g(0.69, 0.67, 0.73, False)}
        out = gt.verdict(test, VAL, "A0", MOTION)
        assert out["arms"]["A2"]["o1_prime_t2_val"]["pass"]
        assert out["arms"]["A2"]["o1_prime_t2_test"]["collapsed"]
        assert not out["arms"]["A2"]["survivor"]
        assert out["arms"]["A3"]["survivor"] and not out["motion_stream_dropped"]

    def test_every_motion_arm_failing_drops_the_stream(self) -> None:
        test = {"A0": _g(0.63, 0.64, 0.66, True), "A1": _g(0.64, 0.64, 0.66, True),
                "A2": _g(0.68, 0.60, 0.80, False), "A3": _g(0.60, 0.67, 0.73, False)}
        out = gt.verdict(test, VAL, "A0", MOTION)
        assert not out["arms"]["A3"]["survivor"]  # micro 0.60 < A0 0.63 - 0.01
        assert out["motion_stream_dropped"]
        md = gt.render_markdown({"data_dir": "T2", "t2_test": test, "t2_val": VAL, **out})
        assert "DROPPED" in md and "COLLAPSED" in md


class TestSourceGate:
    def test_a_missing_source_stops(self, tmp_path: Path) -> None:
        np.save(tmp_path / "s1.npy", np.zeros((3, 2)))
        windows = {"s1__w0": "s1", "s2__w0": "s2"}
        dataset = SimpleNamespace(video_ids=sorted(windows),
                                  slicer=SimpleNamespace(source_of=windows.__getitem__))
        with pytest.raises(FileNotFoundError, match=r"1/2 .*never refit"):
            gt.check_sources(dataset, tmp_path)  # type: ignore[arg-type]
        np.save(tmp_path / "s2.npy", np.zeros((3, 2)))
        gt.check_sources(dataset, tmp_path)  # type: ignore[arg-type]


class TestRunSmoke:
    """CLI wiring: parent T2 dir (T2-test), two plain-input arms, T2-val diags, read-out."""

    def _setup(self, tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Path]]:
        data_dir, _ = _build(tmp_path)
        rng = np.random.default_rng(7)
        clip = tmp_path / "clip"
        clip.mkdir()
        windows = json.loads((data_dir / "windows.json").read_text())
        for s in {w["source"] for w in windows.values()}:
            np.save(clip / f"{s}.npy", rng.normal(size=(100, SMALL)).astype(np.float32))
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        diags = {}
        for name in ("A0", "A2"):
            diags[name] = tmp_path / f"{name}_diag.json"
            diags[name].write_text(json.dumps({"guardrails": VAL[name]}))
        return data_dir, clip, run_dir, diags

    def test_run_writes_the_verdict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        data_dir, clip, run_dir, diags = self._setup(tmp_path)

        def fake_loader(_run: object, _device: object) -> tuple:
            torch.manual_seed(0)
            model = KATVAD(hidden_dim=SMALL, temporal_heads=4, fusion_heads=4).eval()
            cfg = SimpleNamespace(v2=SimpleNamespace(crn="none", motion="none"))
            return model, 10, lambda batch: torch.randn(len(batch), SMALL), cfg

        monkeypatch.setattr(gt, "load_finished_model", fake_loader)
        out = tmp_path / "out"
        ckpt = str(run_dir / "checkpoint_last.pt")
        gt.main([
            "--data-dir", str(data_dir), "--device", "cpu", "--out-dir", str(out),
            "--arm", "A0", ckpt, str(clip), "--arm", "A2", ckpt, str(clip),
            "--val-diag", "A0", str(diags["A0"]), "--val-diag", "A2", str(diags["A2"]),
        ])
        readout = json.loads((out / gt.GUARD_JSON).read_text())
        assert readout["runs"]["A0"]["windows"] == readout["runs"]["A2"]["windows"] > 0
        prime = readout["arms"]["A2"]["o1_prime_t2_test"]
        assert prime["window_auc_delta"] == pytest.approx(0.0)  # same model, same windows
        assert "Q5 verdict" in (out / gt.GUARD_MD).read_text()

    def test_names_must_match(self, tmp_path: Path) -> None:
        data_dir, clip, run_dir, diags = self._setup(tmp_path)
        ckpt = str(run_dir / "checkpoint_last.pt")
        with pytest.raises(ValueError, match="--val-diag names"):
            gt.main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path / "o"),
                     "--arm", "A0", ckpt, str(clip), "--val-diag", "A2", str(diags["A2"])])
        with pytest.raises(ValueError, match="--a0"):
            gt.main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path / "o"),
                     "--arm", "A2", ckpt, str(clip), "--val-diag", "A2", str(diags["A2"])])
