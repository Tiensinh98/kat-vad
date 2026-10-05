"""P6 pilot diagnostics (addendum §4, §14 O1-O7), model-free except one forward test."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core import constants
from core.data.v2_splits import lines_sha1
from core.models.kat_vad import KATVAD
from core.tests.test_v2_dataset import _build
from core.tools import v2_dataset
from core.tools import v2_diagnostics as diag

SMALL = 32
SEED = 2024


def _windows(n: int, steps: int, seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    labels, scores = [], []
    for i in range(n):
        y = np.zeros(steps, dtype=np.int64)
        if i % 2:
            y[steps // 2 :] = 1
        labels.append(y)
        scores.append(0.5 * y + 0.1 * rng.random(steps))
    return scores, labels


class TestGuardrails:
    def test_a_localizing_model_passes(self) -> None:
        scores, labels = _windows(20, 20, 0)
        out = diag.guardrails(scores, labels, a0_micro=None)
        assert out["macro"] == pytest.approx(1.0)
        assert out["checks"]["macro_at_least_micro"]
        assert "micro_within_a0_margin" not in out["checks"]

    def test_a_clip_classifier_fails_the_oracle_and_macro_checks(self) -> None:
        _, labels = _windows(20, 20, 1)
        flat = [np.full(len(y), float(y.max())) + 1e-3 * np.arange(len(y)) for y in labels]
        out = diag.guardrails(flat, labels, a0_micro=None)
        assert out["window_level_auc"] == pytest.approx(1.0)
        assert not out["pass"]

    def test_a0_margin(self) -> None:
        scores, labels = _windows(20, 20, 2)
        micro = diag.guardrails(scores, labels, None)["micro"]
        assert diag.guardrails(scores, labels, micro + 0.009)["checks"]["micro_within_a0_margin"]
        assert not diag.guardrails(scores, labels, micro + 0.011)["checks"][
            "micro_within_a0_margin"
        ]


def _g(micro: float, macro: float, window: float, passed: bool = False) -> dict:
    return {"micro": micro, "macro": macro, "window_level_auc": window, "pass": passed}


class TestO1Prime:
    """Amendment 8 Q2: collapse = window AUC up AND macro below A0 - margin."""

    A0 = _g(0.6586, 0.6737, 0.6863, True)

    def test_an_improving_arm_is_not_collapsed(self) -> None:
        out = diag.o1_prime(_g(0.7161, 0.7052, 0.7637), self.A0)  # pilot A3 on T2-val
        assert not out["collapsed"] and out["pass"]
        assert out["window_auc_delta"] == pytest.approx(0.0774)

    def test_the_c14_signature_is_collapsed(self) -> None:
        out = diag.o1_prime(_g(0.70, 0.62, 0.95), self.A0)  # window up, macro -0.054
        assert out["collapsed"] and not out["pass"]

    def test_a_macro_drop_without_a_window_gain_is_not_collapse(self) -> None:
        out = diag.o1_prime(_g(0.66, 0.60, 0.60), self.A0)
        assert not out["collapsed"]

    def test_the_macro_margin_is_o1s_margin(self) -> None:
        inside = self.A0["macro"] - constants.V2_GUARD_A0_MARGIN + 1e-6
        outside = self.A0["macro"] - constants.V2_GUARD_A0_MARGIN - 1e-6
        assert not diag.o1_prime(_g(0.70, inside, 0.9), self.A0)["collapsed"]
        assert diag.o1_prime(_g(0.70, outside, 0.9), self.A0)["collapsed"]

    def test_the_micro_leg_still_applies(self) -> None:
        out = diag.o1_prime(_g(0.64, 0.70, 0.70), self.A0)
        assert not out["collapsed"] and not out["pass"]
        assert not out["checks"]["micro_within_a0_margin"]

    def test_render_names_every_check(self) -> None:
        line = diag.render_o1_prime(diag.o1_prime(_g(0.7161, 0.7052, 0.7637), self.A0))
        assert "not_collapsed PASS" in line and "**PASS**" in line


class TestProbes:
    def test_position_r2_high_when_encoded_and_low_when_not(self) -> None:
        rng = np.random.default_rng(3)
        encoded, noise, groups = {}, {}, {}
        for i in range(20):
            steps = 20
            tau = np.arange(steps) / steps
            encoded[f"w{i}"] = np.c_[tau, rng.normal(size=(steps, 3))]
            noise[f"w{i}"] = rng.normal(size=(steps, 4))
            groups[f"w{i}"] = f"s{i // 2}"
        assert diag.position_r2(encoded, groups, 5) > 0.95
        assert diag.position_r2(noise, groups, 5) < 0.05

    def test_source_shortcut_separable_vs_identical(self) -> None:
        rng = np.random.default_rng(4)
        t2 = {f"t{i}": rng.normal(0, 1, (15, 4)) for i in range(20)}
        far = {f"d{i}": rng.normal(3, 1, (15, 4)) for i in range(20)}
        same = {f"d{i}": rng.normal(0, 1, (15, 4)) for i in range(20)}
        groups = {k: k for k in [*t2, *far]}
        assert diag.source_shortcut(t2, far, groups, 5, SEED) > 0.99
        assert abs(diag.source_shortcut(t2, same, groups, 5, SEED) - 0.5) < 0.1

    def test_score_shortcut_is_direction_free(self) -> None:
        low = {"t": np.full(10, 0.1)}
        high = {"d": np.full(10, 0.9)}
        assert diag.score_shortcut(low, high) == pytest.approx(1.0)
        assert diag.score_shortcut(high, low) == pytest.approx(1.0)


class TestMotionShareAndD5:
    def test_motion_share_summary_and_no_motion_arm(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        rows = [{"step": s, "motion_share": 0.01 * s, "w_u_norm": 0.1 * s} for s in range(4)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = diag.motion_share(path)
        assert out is not None
        assert out["motion_share"]["first"] == 0.0 and out["w_u_norm"]["max"] == pytest.approx(0.3)
        path.write_text(json.dumps({"step": 1, "total": 1.0}) + "\n")
        assert diag.motion_share(path) is None

    def test_d5_uses_the_reference_mean_and_both_metrics(self) -> None:
        def g(micro: float, macro: float) -> dict:
            return {"guardrails": {"micro": micro, "macro": macro}}

        refs = [g(0.62, 0.63), g(0.60, 0.61)]
        assert diag.d5(g(0.61, 0.62), refs)["pass"]
        assert not diag.d5(g(0.61, 0.59), refs)["pass"]


class TestForward:
    def test_forward_returns_trunk_and_scores_per_step(self) -> None:
        torch.manual_seed(0)
        model = KATVAD(hidden_dim=SMALL, temporal_heads=4, fusion_heads=4).eval()
        vt, ybin = diag.forward_item(model, np.random.rand(9, SMALL), torch.randn(2, SMALL))
        assert vt.shape == (9, SMALL) and ybin.shape == (9,)
        assert ((ybin > 0) & (ybin < 1)).all()

    def test_too_long_an_item_is_refused(self) -> None:
        model = KATVAD(hidden_dim=SMALL, temporal_heads=4, fusion_heads=4).eval()
        with pytest.raises(ValueError, match="MAX_VIS_LEN"):
            diag.forward_item(model, np.zeros((513, SMALL)), torch.randn(2, SMALL))


class TestRunSmoke:
    """CLI wiring end to end: v2 dataset dir, plain A0 inputs, unlabelled DoTA-dev, read-out."""

    @pytest.mark.parametrize("split", ["dota_dev", "dota_cap_dev"])
    def test_run_writes_diag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str
    ) -> None:
        data_dir, _ = _build(tmp_path)
        v2_dir = tmp_path / "v2"
        v2_dataset.make_v2_dataset(data_dir, v2_dir, tmp_path / "splits")
        rng = np.random.default_rng(5)
        clip = tmp_path / "clip"
        clip.mkdir()
        sources = {json.loads((data_dir / "windows.json").read_text())[w]["source"]
                   for w in json.loads((data_dir / "windows.json").read_text())}
        for s in sources:
            np.save(clip / f"{s}.npy", rng.normal(size=(100, SMALL)).astype(np.float32))
        dev = [f"vid{g:02d}_{i:06d}" for g in range(6) for i in range(2)]
        split_dir, s1 = tmp_path / "dsplits", tmp_path / "s1"
        split_dir.mkdir()
        s1.mkdir()
        cap = dev[::2]  # Amendment 6: batch 2 reads O5 on the DoTA-CAP part only
        (split_dir / "dota_dev.txt").write_text("".join(f"{v}\n" for v in dev))
        (split_dir / "dota_cap_dev.txt").write_text("".join(f"{v}\n" for v in cap))
        (split_dir / constants.V2_SPLITS_MANIFEST_FILENAME).write_text(json.dumps(
            {"splits": {"dota_dev": {"count": len(dev), "sha1": lines_sha1(dev)}}}))
        (split_dir / constants.V2_DOTA_CAP_MANIFEST_FILENAME).write_text(json.dumps(
            {"splits": {"dota_cap_dev": {"count": len(cap), "sha1": lines_sha1(cap)}}}))
        for v in dev:
            np.save(s1 / f"{v}.npy", rng.normal(2.0, 1.0, (31, SMALL)).astype(np.float32))
        labels_dir = tmp_path / "dota_labels"
        labels_dir.mkdir()
        (labels_dir / constants.DEFS_FILENAME).write_text(json.dumps(["Normal", "CarAccident"]))
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").write_text(json.dumps({"global_step": 10, "epoch": 0}) + "\n")

        def fake_loader(_run: object, _device: object) -> tuple:
            torch.manual_seed(0)
            model = KATVAD(hidden_dim=SMALL, temporal_heads=4, fusion_heads=4).eval()
            cfg = SimpleNamespace(v2=SimpleNamespace(crn="none", motion="none"),
                                  model=SimpleNamespace(motion_dim=0))
            return model, 10, lambda batch: torch.randn(len(batch), SMALL), cfg

        monkeypatch.setattr(diag, "load_finished_model", fake_loader)
        out = tmp_path / "out"
        diag.main([
            "--run", "A0_s2099", str(run_dir / "checkpoint_last.pt"), "--data-dir", str(v2_dir),
            "--t2-input-dir", str(clip), "--dota-input-dir", str(s1), "--dota-s1-dir", str(s1),
            "--dota-data-dir", str(labels_dir), "--split-dir", str(split_dir), "--folds", "3",
            "--dota-split", split, "--device", "cpu", "--out-dir", str(out),
        ])
        readout = json.loads((out / diag.DIAG_JSON).read_text())
        assert readout["dota_split"] == split
        assert readout["dota_dev_clips_unlabelled"] == len(dev if split == "dota_dev" else cap)
        assert readout["t2_val_windows"] > 0 and readout["motion_share"] is None
        assert readout["source_shortcut_auc"]["v_t"] > 0.9  # the DoTA rows were shifted by +2
        assert "Guardrails" in (out / diag.DIAG_MD).read_text()
        assert "o1_prime" not in readout["guardrails"]  # no --a0-diag: A0 has no reference

        a1_out = tmp_path / "out_a1"
        diag.main([
            "--run", "A1_s2099", str(run_dir / "checkpoint_last.pt"), "--data-dir", str(v2_dir),
            "--t2-input-dir", str(clip), "--dota-input-dir", str(s1), "--dota-s1-dir", str(s1),
            "--dota-data-dir", str(labels_dir), "--split-dir", str(split_dir), "--folds", "3",
            "--dota-split", split, "--device", "cpu", "--out-dir", str(a1_out),
            "--a0-diag", str(out / diag.DIAG_JSON),
        ])
        prime = json.loads((a1_out / diag.DIAG_JSON).read_text())["guardrails"]["o1_prime"]
        assert prime["window_auc_delta"] == pytest.approx(0.0)  # same model, same windows
        assert prime["pass"]
        assert "O1' (Amendment 8)" in (a1_out / diag.DIAG_MD).read_text()
