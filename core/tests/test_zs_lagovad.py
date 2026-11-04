"""LaGoVAD's released checkpoint on v_final's zero-shot sets (``ZS_TAD_DADA.md`` §7): the
``--baseline`` path of ``protocol_b_eval`` / ``dada_whole_eval`` and the three-row read-out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from core import constants
from core.tests.test_zs_tad_dada import _dada_run, _tad_run
from core.tools import dada_whole_eval, protocol_b_eval, rate_matched_eval, zs_lagovad_readout
from core.tools.nexar_transfer_readout import read_tad, read_tad_runs
from core.tools.rate_matched_eval import Run, runs_from_args

# --------------------------------------------------------------------------- run selection


class TestRunsFromArgs:
    def test_run_or_baseline_never_both_or_neither(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            runs_from_args([["s1", "a.pt"]], [["released", "best.ckpt"]])
        with pytest.raises(ValueError, match="not both"):
            runs_from_args(None, None)

    def test_kinds(self) -> None:
        ours = runs_from_args([["s2024", "a.pt"], ["s2025", "b.pt"]], None)
        assert [r.baseline for r in ours] == [False, False] and ours[1].ckpt == Path("b.pt")
        (base,) = runs_from_args(None, [["released", "best.ckpt"]])
        assert base == Run("released", Path("best.ckpt"), baseline=True)

    @pytest.mark.parametrize("tool", [protocol_b_eval, dada_whole_eval])
    def test_both_clis_take_baseline(self, tool: Any) -> None:
        argv = ["--baseline", "released", "best.ckpt", "--out-dir", "o", "--input-dir", "i",
                "--data-dir", "d"]
        if tool is protocol_b_eval:
            argv += ["--s1-dir", "s", "--metadata", "m", "--split-file", "f"]
        else:
            argv += ["--t2-dir", "t", "--annotation", "a", "--census", "c"]
        args = tool.build_arg_parser().parse_args(argv)
        assert args.run is None and args.baseline == [["released", "best.ckpt"]]


class TestLoadReleased:
    def test_baseline_skips_j10_and_loads_kip_off(self, monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
        seen: dict[str, Any] = {}

        def fake_load(cfg: Any, device: Any, ckpt: Any, baseline: Any, enc: Any,
                      overrides: Any) -> torch.nn.Module:
            seen.update(cfg=cfg, ckpt=ckpt, baseline=baseline, overrides=overrides)
            return torch.nn.Linear(1, 1)

        def no_j10(*_: Any) -> None:
            raise AssertionError("J10 must not run on the released checkpoint")

        monkeypatch.setattr(rate_matched_eval, "load_model_for_scoring", fake_load)
        monkeypatch.setattr(rate_matched_eval, "make_text_encoder", lambda *a, **k: "enc")
        monkeypatch.setattr(rate_matched_eval, "check_finished", no_j10)
        ckpt = tmp_path / "best.ckpt"  # never opened: the compat loader is mocked
        _, step, enc, cfg = rate_matched_eval.load_finished_model(
            Run("released", ckpt, baseline=True), torch.device("cpu"))
        assert step == constants.RELEASED_CKPT_STEP and enc == "enc"
        assert seen["ckpt"] is None and seen["baseline"] == ckpt
        assert cfg.kip.enabled is False and "kip.enabled=false" in seen["overrides"]
        assert (cfg.v2.crn, cfg.v2.motion) == (constants.V2_OFF, constants.V2_OFF)
        assert cfg.stp.sa2 is False


# --------------------------------------------------------------------------- inputs


class TestCheckInputs:
    def test_plain_rows_equal_bake_clip_columns(self, tmp_path: Path) -> None:
        d = constants.ZS_LAGOVAD_INPUT_DIM
        rng = np.random.default_rng(0)
        s1 = rng.standard_normal((10, d)).astype(np.float32)
        bake = np.concatenate([s1[::3], rng.standard_normal((4, 384))], axis=1).astype(np.float32)
        plain_dir, bake_dir = tmp_path / "plain", tmp_path / "bake"
        plain_dir.mkdir()
        bake_dir.mkdir()
        np.save(plain_dir / "a_000001.npy", s1)
        np.save(bake_dir / "a_000001.npy", bake)
        assert zs_lagovad_readout.check_inputs(plain_dir, bake_dir, ["a_000001"], 3) == {}
        bad = zs_lagovad_readout.check_inputs(plain_dir, bake_dir, ["a_000001"], 1)
        assert "rows 10" in bad["a_000001"]
        bake[0, 0] += 1.0
        np.save(bake_dir / "a_000001.npy", bake)
        assert "max |plain" in zs_lagovad_readout.check_inputs(
            plain_dir, bake_dir, ["a_000001"], 3)["a_000001"]


# --------------------------------------------------------------------------- read-out


class TestPieces:
    def test_versus_sentence(self) -> None:
        s = zs_lagovad_readout.versus_sentence
        assert s({"mean": 0.02, "low": 0.001, "high": 0.04}) == "above"
        assert s({"mean": -0.02, "low": -0.04, "high": -0.001}) == "below"
        assert s({"mean": 0.02, "low": -0.001, "high": 0.04}) == "not separated"
        assert s(None) == "not separated"

    def test_minmax_micro_ignores_per_clip_offsets(self) -> None:
        gt = np.array([0, 0, 1, 1, 0])
        scores = {"a": np.array([0.0, 0.1, 0.8, 0.9, 0.2]),
                  "b": 100.0 + np.array([0.0, 0.1, 0.8, 0.9, 0.2])}
        assert zs_lagovad_readout.minmax_micro(scores, {"a": gt, "b": gt}) == 1.0

    def test_read_tad_is_read_tad_runs(self, tmp_path: Path) -> None:
        _tad_run(tmp_path / "s2024", 1.0, np.random.default_rng(1))
        a = read_tad(tmp_path, [2024], 7)
        b = read_tad_runs(tmp_path, ["s2024"], 7)
        assert a["macro"] == b["macro"] and a["micro"] == b["micro"]


def _dota_read(out: Path, shift: float, rng: np.random.Generator, ids: list[str]) -> None:
    out.mkdir(parents=True)
    scores, labels = {}, {}
    for v in ids:
        g = np.zeros(20, dtype=np.int64)
        g[8:14] = 1
        labels[v] = g
        scores[v] = rng.random(20) + shift * g
    aucs = protocol_b_eval.clip_aucs(scores, labels)
    protocol_b_eval.write_clip_scores(out / protocol_b_eval.CLIP_SCORES_NPZ, scores, labels)
    (out / protocol_b_eval.CLIP_AUCS_JSON).write_text(json.dumps(aucs))
    (out / protocol_b_eval.READOUT_JSON).write_text(json.dumps({
        "per_run_macro": {"s": float(np.mean(list(aucs.values())))},
        "position_ruler": None, "protocol": "B"}))


def _layout(tmp_path: Path) -> list[str]:
    rng = np.random.default_rng(0)
    lag, zs = tmp_path / "lag" / "eval", tmp_path / "zs" / "eval"
    dota_ids = [f"src{i // 3}_{i:06d}" for i in range(30)]
    _dota_read(lag / "dota" / "LaGoVAD" / "B", 0.2, rng, dota_ids)
    _dota_read(lag / "dota" / "LaGoVAD" / "A", 0.2, rng, dota_ids)
    _dota_read(tmp_path / "w0_dota", 1.0, rng, dota_ids)
    _dota_read(tmp_path / "r4_dota", 1.0, rng, dota_ids)
    _tad_run(lag / "tad" / "LaGoVAD" / "released", 0.5, rng)
    _tad_run(zs / "tad" / "W0" / "s2024", 0.3, rng)
    _tad_run(zs / "tad" / "R4" / "s2024", 3.0, rng)
    dada_ids = [f"d{i}" for i in range(40)]
    _dada_run(lag / "dada" / "LaGoVAD", dict.fromkeys(dada_ids, 0.6), {"released": 0.6})
    _dada_run(zs / "dada" / "W0", dict.fromkeys(dada_ids, 0.7), {"s2024": 0.7})
    _dada_run(zs / "dada" / "R4", dict.fromkeys(dada_ids, 0.7), {"s2024": 0.7})
    return ["--lagovad-root", str(tmp_path / "lag"), "--zs-root", str(tmp_path / "zs"),
            "--dota", "W0", str(tmp_path / "w0_dota"), "--dota", "R4", str(tmp_path / "r4_dota")]


class TestReadout:
    def test_end_to_end(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        argv = _layout(tmp_path)
        t0 = read_tad_runs(tmp_path / "lag" / "eval" / "tad" / "LaGoVAD", ["released"], 0)
        monkeypatch.setattr(constants, "ZS_LAGOVAD_T0_MICRO", t0["micro"])
        r = zs_lagovad_readout.run(zs_lagovad_readout.build_arg_parser().parse_args(argv))
        dota, tad, dada = (r["sets"][s] for s in ("dota", "tad", "dada"))
        assert dota["versus_lagovad"]["W0"]["sentence"] == "above"
        assert "lagovad_protocol_a" in dota and dota["rows"]["LaGoVAD"]["clip_auc"] is None
        assert tad["t0_check"]["pass"] is True
        assert tad["sa2_minus_host"]["sentence"] == "gain shown"
        assert dada["versus_lagovad"]["R4"]["delta"]["mean"] == pytest.approx(0.1)
        assert dada["sa2_minus_host"]["sentence"] == "no difference shown"
        md = (tmp_path / "lag" / "REPORTS" / "zs_lagovad_readout.md").read_text()
        assert "PRELIMINARY (n = 1 seed)" in md and "not matched" in md and "Gate T0" in md

    def test_t0_mismatch_stops_the_read(self, tmp_path: Path) -> None:
        argv = _layout(tmp_path)  # synthetic TAD micro != 0.7912
        with pytest.raises(ValueError, match="Gate T0"):
            zs_lagovad_readout.run(zs_lagovad_readout.build_arg_parser().parse_args(argv))
        assert not (tmp_path / "lag" / "REPORTS").exists()

    def test_rows_must_cover_the_same_items(self, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
        argv = _layout(tmp_path)
        t0 = read_tad_runs(tmp_path / "lag" / "eval" / "tad" / "LaGoVAD", ["released"], 0)
        monkeypatch.setattr(constants, "ZS_LAGOVAD_T0_MICRO", t0["micro"])
        bad = tmp_path / "lag" / "eval" / "dada" / "LaGoVAD" / dada_whole_eval.CLIP_AUCS_JSON
        bad.write_text(json.dumps({"d0": 0.6}))
        with pytest.raises(ValueError, match="dada"):
            zs_lagovad_readout.run(zs_lagovad_readout.build_arg_parser().parse_args(argv))

    def test_dota_needs_both_arms(self, tmp_path: Path) -> None:
        args = zs_lagovad_readout.build_arg_parser().parse_args(
            ["--lagovad-root", str(tmp_path), "--zs-root", str(tmp_path),
             "--dota", "W0", str(tmp_path)])
        with pytest.raises(ValueError, match="--dota"):
            zs_lagovad_readout.run(args)
