"""SV4: SA²-v2 with a zero-init matrix ``W_p`` on x^AS (``core/docs/final/SA2V2_RUN.md`` §11)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from core import constants
from core.config import Config
from core.data.stp_patches import EvalCells
from core.inference import load_model_for_scoring, sliding_window_scores
from core.models.kat_vad import KATVAD
from core.models.sa2v2 import SA2V2
from core.tests.test_sa2v2 import LEVEL, _cfg, _fixture, _inputs
from core.tools import dada_whole_eval, sa2v2_cells, sv2_probe, sv4_readout
from core.tools.protocol_b_eval import write_clip_scores
from core.tools.stp_readout import COST, GAIN, NONE
from core.train import build_trainer, optimizer_param_groups, set_global_seed

CPU = torch.device("cpu")


def _sv4_cfg() -> Config:
    cfg = _cfg()
    cfg.stp.sv2_proj = True
    return cfg


def _w0d_cfg() -> Config:
    cfg = _cfg()
    cfg.stp.sa2v2 = False
    cfg.model.motion_dropout = constants.W0D_MOTION_DROPOUT
    return cfg


def _module() -> SA2V2:
    module = SA2V2(proj=True)
    module.set_level(*LEVEL)
    return module


def _grad_sum(param: torch.Tensor) -> float:
    assert param.grad is not None
    return float(param.grad.abs().sum())


class TestModule:
    def test_sv2_and_sv3_carry_no_w_p(self) -> None:
        model = KATVAD.from_config(_cfg())
        assert model.sa2v2 is not None and model.sa2v2.w_p is None
        assert not [k for k in model.state_dict() if "w_p." in k]  # old checkpoints load strictly

    def test_starts_as_the_host_and_only_w_p_gets_the_host_gradient(self) -> None:
        module = _module()
        cls, columns, text = _inputs()
        out = module(cls, columns, text)
        torch.testing.assert_close(out.features, cls)
        assert float(out.x_as.abs().sum()) > 0  # L_branch still reads x^AS
        out.features.square().sum().backward()
        assert module.w_p is not None
        assert _grad_sum(module.w_p.weight) > 0 and _grad_sum(module.w_p.bias) > 0
        assert _grad_sum(module.w.weight) == 0.0

    def test_nonzero_w_p_is_applied_and_padding_stays_the_host(self) -> None:
        module = _module()
        assert module.w_p is not None
        with torch.no_grad():
            module.w_p.weight.fill_(0.01)
            module.w_p.bias.fill_(1.0)
        cls, columns, text = _inputs()
        mask = torch.ones(cls.shape[:2])
        mask[:, -2:] = 0
        out = module(cls, columns, text, mask)
        torch.testing.assert_close(out.features[:, -2:], cls[:, -2:])  # no bias leak
        torch.testing.assert_close(out.features[:, 0], cls[:, 0] + module.w_p(out.x_as[:, 0]))

    def test_refuses_gate_and_w_p_together(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            SA2V2(gate=True, proj=True)

    def test_config_refuses(self) -> None:
        cfg = _sv4_cfg()
        cfg.stp.sv2_gate = True
        with pytest.raises(ValueError, match="pick one"):
            KATVAD.from_config(cfg)
        cfg = _sv4_cfg()
        cfg.stp.sa2v2 = False
        with pytest.raises(ValueError, match=r"needs stp\.sa2v2"):
            KATVAD.from_config(cfg)

    def test_w_p_stays_in_the_one_group(self) -> None:
        cfg = _sv4_cfg()
        model = KATVAD.from_config(cfg)
        assert model.sa2v2 is not None and model.sa2v2.w_p is not None
        groups = optimizer_param_groups(model, 5e-5, 1.0, 1.0)
        assert not isinstance(groups[0], dict)
        assert any(p is model.sa2v2.w_p.weight for p in groups)


class TestPairedWithW0D:
    """D13: the module's init must not move the host's init or the global RNG stream."""

    def test_host_init_and_rng_stream_equal_w0d(self) -> None:
        set_global_seed(7)
        w0d = KATVAD.from_config(_w0d_cfg())
        after_w0d = torch.get_rng_state()
        set_global_seed(7)
        sv4 = KATVAD.from_config(_sv4_cfg())
        assert torch.equal(after_w0d, torch.get_rng_state())
        reference = w0d.state_dict()
        host = {k: v for k, v in sv4.state_dict().items() if not k.startswith("sa2v2.")}
        assert host.keys() == reference.keys()
        for key, value in host.items():
            assert torch.equal(value, reference[key]), key

    def test_sv3_keeps_its_build(self) -> None:
        cfg = _cfg()
        cfg.stp.sv2_gate = True
        set_global_seed(7)
        KATVAD.from_config(_w0d_cfg())
        after_w0d = torch.get_rng_state()
        set_global_seed(7)
        KATVAD.from_config(cfg)
        assert not torch.equal(after_w0d, torch.get_rng_state())  # unpaired, as §10.4 records


def _train(cfg: Config, out: Path, data_dir: Path, host: Path, cells: Path | None) -> None:
    cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 1, "cpu"
    set_global_seed(cfg.train.seed)
    trainer = build_trainer(cfg, out, data_dir, host, text_encoder="stub", patch_dir=cells)
    cfg.save_yaml(out / "config.yaml")
    trainer.train()


class TestTrainAndProbe:
    def test_step_one_equals_w0d_and_the_probe_reads_w_p(self, tmp_path: Path) -> None:
        data_dir, host, cells = _fixture(tmp_path)
        level = dict(zip(sa2v2_cells.LEVEL_KEYS, LEVEL, strict=True))
        (cells / constants.SV2_LEVEL_FILENAME).write_text(json.dumps(level))
        sv4_dir, w0d_dir = tmp_path / "sv4", tmp_path / "w0d"
        _train(_sv4_cfg(), sv4_dir, data_dir, host, cells)
        _train(_w0d_cfg(), w0d_dir, data_dir, host, None)
        mech = sv4_readout.mechanics(sv4_readout.read_metrics(sv4_dir / "metrics.jsonl"),
                                     sv4_readout.read_metrics(w0d_dir / "metrics.jsonl"))
        assert mech["paired"], mech["step1"]
        assert mech["share_tail"] > 0 and mech["wp_norm_tail"] > 0
        adam = sv2_probe.run_adam(argparse.Namespace(
            run_dir=sv4_dir, out_dir=tmp_path / "adam", text_encoder="stub"))
        assert adam["rows"]["w_p"]["adam_state"]
        assert adam["rows"]["w_p"]["numel"] == constants.SV2_DIM**2 + constants.SV2_DIM

        model = load_model_for_scoring(Config(), CPU, sv4_dir / "checkpoint_last.pt", None, "stub")
        assert model.sa2v2 is not None and model.sa2v2.w_p is not None
        assert float(model.sa2v2.w_p.weight.abs().sum()) > 0  # the trained W_p, not a fresh zero
        video = next(p.stem for p in cells.glob("*.npy"))
        rows = EvalCells(cells).append(np.load(host / f"{video}.npy"), video)
        text = torch.randn(2, constants.SV2_DIM)
        curve, _ = sliding_window_scores(model, torch.from_numpy(rows), lambda: text)
        assert curve.shape == (len(rows),) and torch.isfinite(curve).all()


def _metrics(path: Path, share: float, mil: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.jsonl").write_text("\n".join(json.dumps({
        "global_step": i + 1, "mil": 0.7 if i == 0 else mil, "dvs_sup": 0.5,
        sv4_readout.SHARE_KEY: share, sv4_readout.NORM_KEY: 1.0,
    }) for i in range(5)))


class TestReadout:
    @pytest.mark.parametrize(("used", "primary", "words"), [
        (False, NONE, "stays small"), (False, GAIN, "stays small"), (False, COST, "unused"),
        (True, GAIN, "helps"), (True, NONE, "without a shown effect"),
        (True, COST, "used and costs"),
    ])
    def test_verdict_table(self, used: bool, primary: str, words: str) -> None:
        assert words in sv4_readout.verdict(used, primary)

    @pytest.mark.parametrize(("dada", "dota", "seeds"), [
        (GAIN, GAIN, True), (GAIN, NONE, True), (GAIN, COST, False),
        (NONE, GAIN, False), (COST, GAIN, False),
    ])
    def test_stop_rule(self, dada: str, dota: str, seeds: bool) -> None:
        assert (sv4_readout.next_step(dada, dota) == sv4_readout.NEXT_SEEDS) is seeds

    def test_mechanics_refuses_a_run_without_w_p(self) -> None:
        with pytest.raises(ValueError, match="sv2_patch_share"):
            sv4_readout.mechanics([{"mil": 0.1, "dvs_sup": 0.1}], [{"mil": 0.1, "dvs_sup": 0.1}])

    def test_tail_uses_the_last_steps_only(self) -> None:
        records = [{"mil": 9.0}] * 5 + [{"mil": 1.0}] * constants.SV4_TAIL_STEPS
        assert sv4_readout.tail_mean(records, "mil") == pytest.approx(1.0)

    def test_end_to_end(self, tmp_path: Path) -> None:
        labels = {f"v{i}_{i:06d}": np.array([0, 0, 1, 1, 0]) for i in range(30)}
        dirs: dict[str, Any] = {}
        for arm, auc in (("W0", 0.76), ("W0D", 0.75), ("SV3", 0.73), ("SV4", 0.60)):
            rng = np.random.default_rng(int(auc * 100))
            aucs = {v: auc + 0.01 * rng.standard_normal() for v in labels}
            ci = {"mean": auc, "low": auc - 0.02, "high": auc + 0.02}
            for name, read_file, read in (
                ("dada", dada_whole_eval.READOUT_JSON, {
                    "per_run_macro": {"s2024": auc}, "macro": ci, "micro": auc, "video_auc": None,
                    "monotone_ruler_macro": ci, "middle_ruler_macro": ci}),
                ("dota", "protocol_b_readout.json", {"macro": ci}),
            ):
                d = tmp_path / name / arm
                d.mkdir(parents=True)
                (d / dada_whole_eval.CLIP_AUCS_JSON).write_text(json.dumps(aucs))
                (d / read_file).write_text(json.dumps(read))
                write_clip_scores(d / "clip_scores.npz", {v: rng.random(5) for v in labels}, labels)
                dirs[f"{arm.lower()}_{name}"] = d
        _metrics(tmp_path / "sv4_run", 0.2, 0.2)
        _metrics(tmp_path / "w0d_run", 0.0, 0.3)
        out_dir = tmp_path / "rep"
        r = sv4_readout.run(argparse.Namespace(
            **dirs, sv4_run=tmp_path / "sv4_run", w0d_run=tmp_path / "w0d_run", out_dir=out_dir))
        assert r["mechanics"]["used"] and r["mechanics"]["paired"]
        assert r["dada"]["SV4-W0D"]["sentence"] == COST and r["dada"]["overfit_tell"]
        assert r["next"] == sv4_readout.NEXT_CLOSE
        md = (out_dir / "sv4_readout.md").read_text()
        assert "| SV4 - W0D |" in md and "used and costs" in md and "PRELIMINARY" in md
        assert "Overfit tell: yes" in md and "token line closes" in md and "middle ruler" in md
