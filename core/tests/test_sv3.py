"""SV3: SA²-v2 with a zero-init gate on x^AS (``core/docs/final/SA2V2_RUN.md`` §10)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants
from core.models.kat_vad import KATVAD
from core.models.sa2v2 import SA2V2
from core.tests.test_sa2v2 import LEVEL, _cfg, _fixture, _inputs
from core.tools import dada_whole_eval, sa2v2_cells, sv2_probe, sv3_readout
from core.tools.protocol_b_eval import write_clip_scores
from core.tools.stp_readout import COST, GAIN, NONE
from core.train import build_trainer, optimizer_param_groups, set_global_seed


def _gated(module: SA2V2 | None = None) -> SA2V2:
    module = module or SA2V2(gate=True)
    module.set_level(*LEVEL)
    return module


class TestGate:
    def test_ungated_sv2_is_unchanged(self) -> None:
        model = KATVAD.from_config(_cfg())
        assert model.sa2v2 is not None and model.sa2v2.gate is None
        assert model.sa2v2.gate_value() is None
        assert "sa2v2.gate" not in model.state_dict()  # every SV2 checkpoint loads strictly

    def test_gate_starts_shut_and_only_g_gets_the_host_gradient(self) -> None:
        module = _gated()
        cls, columns, text = _inputs()
        out = module(cls, columns, text)
        torch.testing.assert_close(out.features, cls)  # tanh(0) x^AS = 0: SV3 == host at step 0
        assert float(out.x_as.abs().sum()) > 0  # L_branch still sees the ungated x^AS
        out.features.square().sum().backward()
        assert module.gate is not None and float(module.gate.grad.abs()) > 0
        assert float(module.w.weight.grad.abs().sum()) == 0.0  # no host loss reaches the scorer

    def test_open_gate_scales_x_as(self) -> None:
        module = _gated()
        assert module.gate is not None
        with torch.no_grad():
            module.gate.fill_(0.5)
        cls, columns, text = _inputs()
        out = module(cls, columns, text)
        torch.testing.assert_close(out.features, cls + torch.tanh(torch.tensor(0.5)) * out.x_as)

    def test_config_refuses(self) -> None:
        cfg = _cfg()
        cfg.stp.sa2v2, cfg.stp.sv2_gate = False, True
        with pytest.raises(ValueError, match=r"needs stp\.sa2v2"):
            KATVAD.from_config(cfg)
        cfg = _cfg()
        cfg.stp.sv2_gate, cfg.stp.gate_lr_mult = True, 0.0
        with pytest.raises(ValueError, match="gate_lr_mult"):
            KATVAD.from_config(cfg)


class TestParamGroups:
    def test_default_and_sv2_stay_one_flat_group(self) -> None:
        model = KATVAD.from_config(_cfg())
        groups = optimizer_param_groups(model, 5e-5, 1.0, constants.SV3_GATE_LR_MULT)
        assert not isinstance(groups[0], dict)  # no gate -> the multiplier is inert

    def test_gate_gets_its_own_group(self) -> None:
        cfg = _cfg()
        cfg.stp.sv2_gate = True
        model = KATVAD.from_config(cfg)
        assert model.sa2v2 is not None
        groups = optimizer_param_groups(model, 5e-5, 1.0, 25.0)
        assert len(groups) == 2 and groups[1]["params"] == [model.sa2v2.gate]
        assert groups[1]["lr"] == pytest.approx(5e-5 * 25) and groups[0]["lr"] == 5e-5
        both = optimizer_param_groups(model, 5e-5, 10.0, 25.0)
        assert [g["lr"] for g in both] == pytest.approx([5e-5, 5e-4, 1.25e-3])
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert sum(len(g["params"]) for g in both) == trainable


class TestTrainAndProbe:
    def test_trains_logs_the_gate_and_the_probe_reads_it(self, tmp_path: Path) -> None:
        data_dir, host, cells = _fixture(tmp_path)
        level = dict(zip(sa2v2_cells.LEVEL_KEYS, LEVEL, strict=True))
        (cells / constants.SV2_LEVEL_FILENAME).write_text(json.dumps(level))
        cfg = _cfg()
        cfg.stp.sv2_gate = True
        cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 1, "cpu"
        out = tmp_path / "run"
        set_global_seed(cfg.train.seed)
        trainer = build_trainer(cfg, out, data_dir, host, text_encoder="stub", patch_dir=cells)
        assert len(trainer.optimizer.param_groups) == 2
        cfg.save_yaml(out / "config.yaml")
        trainer.train()
        gate = sv3_readout.gate_trace(out / "metrics.jsonl")
        assert gate["first"] == pytest.approx(0.0, abs=1e-3) and not gate["opened"]
        adam = sv2_probe.run_adam(argparse.Namespace(
            run_dir=out, out_dir=tmp_path / "adam", text_encoder="stub"))
        assert adam["rows"]["gate"]["lr_mult"] == 25.0 and adam["rows"]["gate"]["adam_state"]
        assert adam["rows"]["beta[0]"]["norm_init"] == pytest.approx(1.0)  # names map right


class TestReadout:
    @pytest.mark.parametrize(("opened", "primary", "words"), [
        (False, NONE, "stays unused"), (False, COST, "shut gate"), (True, GAIN, "helps"),
        (True, NONE, "without a shown effect"), (True, COST, "opened and costs"),
    ])
    def test_verdict_table(self, opened: bool, primary: str, words: str) -> None:
        assert words in sv3_readout.verdict(opened, primary)

    def test_gate_trace_refuses_an_ungated_run(self, tmp_path: Path) -> None:
        (tmp_path / "metrics.jsonl").write_text(json.dumps({"global_step": 1}) + "\n")
        with pytest.raises(ValueError, match="sv2_gate"):
            sv3_readout.gate_trace(tmp_path / "metrics.jsonl")

    def test_end_to_end_on_dada(self, tmp_path: Path) -> None:
        labels = {f"v{i}": np.array([0, 0, 1, 1, 0]) for i in range(30)}
        dirs = {}
        for arm, auc in (("W0", 0.76), ("W0D", 0.75), ("SV2", 0.71), ("SV3", 0.75)):
            d = tmp_path / arm
            d.mkdir()
            rng = np.random.default_rng(len(arm) + int(auc * 100))
            (d / dada_whole_eval.CLIP_AUCS_JSON).write_text(
                json.dumps({v: auc + 0.01 * rng.standard_normal() for v in labels}))
            ci = {"mean": auc, "low": auc - 0.02, "high": auc + 0.02}
            (d / dada_whole_eval.READOUT_JSON).write_text(json.dumps({
                "per_run_macro": {"s2024": auc}, "macro": ci, "micro": auc, "video_auc": None,
                "monotone_ruler_macro": ci, "middle_ruler_macro": ci}))
            write_clip_scores(d / "clip_scores.npz", {v: rng.random(5) for v in labels}, labels)
            dirs[arm] = d
        run = tmp_path / "sv3_run"
        run.mkdir()
        (run / "metrics.jsonl").write_text("\n".join(
            json.dumps({"global_step": i, "sv2_gate": 0.001 * i}) for i in range(20)))
        args = argparse.Namespace(
            w0_dada=dirs["W0"], w0d_dada=dirs["W0D"], sv2_dada=dirs["SV2"], sv3_dada=dirs["SV3"],
            w0_dota=None, w0d_dota=None, sv2_dota=None, sv3_dota=None, sv3_run=run,
            out_dir=tmp_path / "rep")
        r = sv3_readout.run(args)
        assert not r["gate"]["opened"] and r["dota"] is None
        assert r["dada"]["SV3-SV2"]["sentence"] == GAIN
        assert r["dada"]["verdict"] == sv3_readout.verdict(False, r["dada"]["SV3-W0D"]["sentence"])
        md = (tmp_path / "rep" / "sv3_readout.md").read_text()
        assert "stayed shut" in md and "| SV3 - W0D |" in md and "PRELIMINARY" in md
