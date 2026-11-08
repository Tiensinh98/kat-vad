"""KIM form probe (``core/docs/final/KIM_RUN.md`` §14)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from core import constants
from core.config import Config
from core.data.stp_patches import patch_change
from core.models.kat_vad import KATVAD
from core.tools import kim_form_probe as probe
from core.tools import kim_pilot
from core.tools.protocol_b_eval import write_clip_scores

MOTION = 384
BASE = constants.HIDDEN_DIM + MOTION
CPU = torch.device("cpu")
PEAK = 3  # step of the reversal in _reversal()
K0_STEP = 1749  # K0 s2024, the recorded pilot


def _reversal() -> np.ndarray:
    """One channel moving at equal speed up and then down: its speed never changes at the peak."""
    return np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])[:, None]


def _host() -> KATVAD:
    cfg = Config()
    cfg.kip.enabled = False
    cfg.model.motion_dim = MOTION
    cfg.v2.motion = constants.V5_ENCODER
    cfg.model.score_head_kernel = 3
    torch.manual_seed(0)
    return KATVAD.from_config(cfg).eval()


def _labels(n: int, videos: int = 6) -> dict[str, np.ndarray]:
    grid = np.zeros(n)
    grid[n // 3 : 2 * n // 3] = 1.0
    return {f"src{i % 3}_{i:06d}": grid.copy() for i in range(videos)}


def _macro(mean: float) -> dict[str, Any]:
    return {"macro": {"mean": mean, "low": mean - 0.01, "high": mean + 0.01}}


def _set(reference: float, candidate: float) -> dict[str, Any]:
    rows = {probe.HOST_ROW: _macro(0.8)}
    for feature in probe.FEATURES:
        for form in probe.FORMS:
            rows[probe.row_name(feature, form)] = _macro(reference)
    rows[probe.row_name("vt", probe.SIGNED)] = _macro(candidate)
    return {"rows": rows}


class TestForms:
    def test_reference_is_the_pilots_order_2(self) -> None:
        z = np.random.default_rng(0).standard_normal((12, 4))
        np.testing.assert_array_equal(probe.form_scores(z)[probe.REFERENCE],
                                      kim_pilot.change_scores(z)[2])

    def test_kim_form_misses_a_reversal_and_signed_forms_do_not(self) -> None:
        s = probe.form_scores(_reversal())
        assert s[probe.REFERENCE][PEAK + 1] == 0.0
        assert s[probe.SIGNED][PEAK + 1] == pytest.approx(2.0)
        assert s[probe.CENTRAL][PEAK] == pytest.approx(2.0)

    def test_central_is_signed_one_step_earlier(self) -> None:
        z = np.random.default_rng(1).standard_normal((15, 8))
        s = probe.form_scores(z)
        np.testing.assert_allclose(s[probe.CENTRAL][1:-1], s[probe.SIGNED][2:])
        assert s[probe.CENTRAL][0] == s[probe.CENTRAL][-1] == 0.0
        assert s[probe.SIGNED][0] == s[probe.SIGNED][1] == 0.0

    def test_mo_is_sa2s_own_change_on_one_cell(self) -> None:
        z = np.random.default_rng(2).standard_normal((9, 16)).astype(np.float32)
        want = np.linalg.norm(patch_change(z[:, None, :]), axis=-1)[:, 0]
        np.testing.assert_allclose(probe.form_scores(z)[probe.SA2_MO], want, rtol=1e-5)

    @pytest.mark.parametrize("length", [1, 2, 3])
    def test_short_sequences_keep_their_length(self, length: int) -> None:
        for curve in probe.form_scores(np.ones((length, 4))).values():
            assert curve.shape == (length,)


class TestDecide:
    def test_needs_both_endpoints(self) -> None:
        first, second = probe.ENDPOINTS
        go = probe.decide({first: _set(0.70, 0.73), second: _set(0.70, 0.72)})
        assert go["verdict"] == probe.GO and go["passes"] == ["vt2s"]
        stop = probe.decide({first: _set(0.70, 0.73), second: _set(0.70, 0.71)})
        assert stop["verdict"] == probe.STOP and stop["passes"] == []

    def test_refuses_a_missing_endpoint(self) -> None:
        with pytest.raises(ValueError, match="both endpoints"):
            probe.decide({probe.ENDPOINTS[0]: _set(0.7, 0.7), "tad": _set(0.7, 0.7)})


_SETS = {"dada": _set(0.7, 0.7)}


class TestBaseline:
    def _recorded(self, vt2: float = 0.7) -> dict[str, Any]:
        recorded = {"dada": _set(0.7, 0.7)}
        recorded["dada"]["rows"]["vt2"] = _macro(vt2)
        return {"host": "K0", "global_step": 1749, "sets": recorded}

    def test_match_passes(self) -> None:
        probe.check_baseline("K0", 1749, _SETS, self._recorded())

    def test_a_moved_macro_raises(self) -> None:
        with pytest.raises(ValueError, match="vt2"):
            probe.check_baseline("K0", 1749, _SETS, self._recorded(0.7 + 2e-4))

    @pytest.mark.parametrize(("host", "step"), [("K5", 1749), ("K0", 1000)])
    def test_another_checkpoint_raises(self, host: str, step: int) -> None:
        with pytest.raises(ValueError, match="the pilot read"):
            probe.check_baseline(host, step, _SETS, self._recorded())

    def test_a_set_the_pilot_did_not_read_raises(self) -> None:
        with pytest.raises(ValueError, match="no set tad"):
            probe.check_baseline("K0", 1749, {"tad": _set(0.7, 0.7)}, self._recorded())


class TestRun:
    def _inputs(self, tmp_path: Path, name: str, length: int) -> tuple[Path, Path, dict[str, Any]]:
        labels = _labels(length)
        rng = np.random.default_rng(3)
        eval_dir, bake = tmp_path / name / "read", tmp_path / name / "bake"
        eval_dir.mkdir(parents=True)
        bake.mkdir()
        write_clip_scores(eval_dir / kim_pilot.CLIP_SCORES_NPZ,
                          {v: rng.random(length) for v in labels}, labels)
        for v in labels:
            np.save(bake / f"{v}.npy", rng.standard_normal((length, BASE)).astype(np.float32))
        return eval_dir, bake, labels

    def _argv(self, tmp_path: Path, baseline: Path) -> list[str]:
        argv = ["--name", "K0", "--ckpt", "K0.pt", "--baseline", str(baseline),
                "--out-dir", str(tmp_path / "out"), "--device", "cpu"]
        for name in probe.ENDPOINTS:
            eval_dir, bake, _ = self._inputs(tmp_path, name, 12)
            argv += ["--set", name, "rows", str(eval_dir), str(bake)]
        return argv

    def test_end_to_end_gates_on_the_recorded_pilot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(probe, "load_finished_model",
                            lambda run, device: (_host(), 1749, None, Config()))
        monkeypatch.setattr(probe, "check_input_manifest", lambda *a: None)
        baseline = tmp_path / "pilot.json"
        baseline.write_text(json.dumps({"host": "K0", "global_step": 1749, "sets": {}}))
        argv = self._argv(tmp_path, baseline)
        with pytest.raises(ValueError, match="no set"):
            probe.run(probe.build_arg_parser().parse_args(argv))
        assert not (tmp_path / "out").exists()  # nothing is written before the gate

        calls: list[dict[str, Any]] = []
        real = probe.check_baseline
        monkeypatch.setattr(probe, "check_baseline",
                            lambda host, step, sets, base: calls.append(sets))
        probe.run(probe.build_arg_parser().parse_args(argv))
        recorded = {name: {"rows": {c: {"macro": {"mean": s["rows"][c]["macro"]["mean"]}}
                                    for c in ("vt2", "clip2", probe.HOST_ROW)}}
                    for name, s in calls[0].items()}
        real("K0", 1749, calls[0], {"host": "K0", "global_step": 1749, "sets": recorded})

        saved = json.loads((tmp_path / "out" / "kim_form_probe.json").read_text())
        assert saved["disclosure"] == constants.KIM_FORM_DISCLOSURE
        assert saved["decision"]["verdict"] in (probe.GO, probe.STOP)
        assert set(saved["decision"]["margins"]) == {"vt2s", "vt2c", "vt2m",
                                                     "clip2s", "clip2c", "clip2m"}
        md = (tmp_path / "out" / "kim_form_probe.md").read_text()
        assert "PRELIMINARY (n = 1)" in md and "Verdict:" in md and "never K5's" in md
        assert "vt2c" in md and "| host |" in md

    def test_refuses_a_kim_checkpoint(self) -> None:
        cfg = Config()
        cfg.kip.enabled = False
        cfg.model.motion_dim = MOTION
        cfg.v2.motion = constants.V5_ENCODER
        cfg.kim.enabled = True
        cfg.dvs.theta = cfg.dvs.theta_ego = 1.0
        with pytest.raises(ValueError, match="bare host"):
            kim_pilot.check_model(KATVAD.from_config(cfg))

    def test_native_sets_land_on_the_label_grid(self, tmp_path: Path) -> None:
        labels = _labels(13, videos=2)
        for v in labels:
            rows = np.random.default_rng(4).standard_normal((5, BASE)).astype(np.float32)
            np.save(tmp_path / f"{v}.npy", rows)
        curves = probe.score_set(kim_pilot.KIND_NATIVE, tmp_path, labels, _host(), CPU, 3)
        names = {probe.row_name(f, form) for f in probe.FEATURES for form in probe.FORMS}
        assert set(curves) == names
        assert all(len(c[v]) == 13 for c in curves.values() for v in labels)

    def test_a_wrong_row_count_raises(self, tmp_path: Path) -> None:
        labels = _labels(12, videos=1)
        np.save(tmp_path / f"{next(iter(labels))}.npy", np.zeros((11, BASE), dtype=np.float32))
        with pytest.raises(ValueError, match="input rows"):
            probe.score_set(kim_pilot.KIND_ROWS, tmp_path, labels, _host(), CPU, 1)

    def test_recorded_pilot_has_the_rows_the_gate_reads(self) -> None:
        recorded = constants.REPO_ROOT / "outputs/v_final/kim/REPORTS/pilot_edges/kim_pilot.json"
        if not recorded.is_file():
            pytest.skip("the recorded pilot is not in this checkout")
        saved = json.loads(recorded.read_text())
        assert saved["host"] == "K0" and saved["global_step"] == K0_STEP
        for s in saved["sets"].values():
            for cand in ("vt2", "clip2", probe.HOST_ROW):
                assert isinstance(s["rows"][cand]["macro"]["mean"], float)
