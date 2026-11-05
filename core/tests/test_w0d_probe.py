"""W0D (host motion dropout) and the SA²-v2 probe (``core/docs/final/SA2V2_RUN.md`` §8-§9)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants
from core.config import Config, load_config
from core.data.collate import collate_variable_length
from core.models.kat_vad import KATVAD
from core.tests.test_sa2v2 import LEVEL, _cfg, _fixture
from core.tools import dada_whole_eval, sa2v2_cells, sv2_probe, w0d_readout
from core.tools.protocol_b_eval import write_clip_scores
from core.tools.stp_readout import COST, GAIN, NONE
from core.train import build_trainer, set_global_seed

MOTION = 384
BASE = constants.HIDDEN_DIM + MOTION


def _host_cfg(dropout: float = 0.0) -> Config:
    cfg = _cfg()
    cfg.stp.sa2v2 = False
    cfg.model.motion_dropout = dropout
    return cfg


class TestW0DConfig:
    def test_default_host_is_unchanged(self) -> None:
        model = KATVAD.from_config(_host_cfg())
        assert model.sa2v2 is None and model.motion_dropout == 0.0
        mask = torch.ones(2, 3)
        model.train()
        assert model._motion_mask(mask) is mask  # same object: the default graph is untouched
        assert "motion_dropout" not in "".join(model.state_dict())

    def test_host_dropout_zeroes_the_whole_term_in_training_only(self) -> None:
        model = KATVAD.from_config(_host_cfg(0.999))
        assert model.sa2v2 is None and model.motion_dropout == pytest.approx(0.999)
        assert model.motion_residual is not None
        torch.nn.init.ones_(model.motion_residual.proj.bias)
        rows, mask = torch.randn(2, 3, BASE), torch.ones(2, 3)
        model.train()
        torch.manual_seed(0)
        out, _ = model.motion_residual(rows, model._motion_mask(mask))
        torch.testing.assert_close(out, rows[..., : constants.HIDDEN_DIM])
        model.eval()
        assert model._motion_mask(mask) is mask

    def test_sv2_keeps_its_own_knob(self) -> None:
        cfg = _cfg()
        assert KATVAD.from_config(cfg).motion_dropout == cfg.stp.motion_dropout

    @pytest.mark.parametrize(("change", "match"), [
        ({"model.motion_dropout": 1.0}, r"\[0, 1\)"),
        ({"model.motion_dropout": -0.1}, r"\[0, 1\)"),
        ({"model.motion_dropout": 0.3, "stp.sa2v2": True}, "own motion dropout"),
        ({"model.motion_dropout": 0.3, "model.motion_dim": 0, "v2.motion": constants.V2_OFF},
         "motion stream"),
    ])
    def test_config_refuses(self, change: dict[str, object], match: str) -> None:
        cfg = _host_cfg()
        for key, value in change.items():
            section, name = key.split(".")
            setattr(getattr(cfg, section), name, value)
        with pytest.raises(ValueError, match=match):
            KATVAD.from_config(cfg)

    def test_old_yaml_without_the_key_loads_as_off(self, tmp_path: Path) -> None:
        cfg = _host_cfg()
        cfg.save_yaml(tmp_path / "config.yaml")
        text = (tmp_path / "config.yaml").read_text()
        kept = [line for line in text.splitlines() if "motion_dropout: 0.0" not in line]
        assert len(kept) == len(text.splitlines()) - 1  # only the model key was dropped
        (tmp_path / "config.yaml").write_text("\n".join(kept))
        assert load_config(tmp_path / "config.yaml").model.motion_dropout == 0.0


class TestW0DReadout:
    @pytest.mark.parametrize(("drop", "spatial", "words"), [
        (COST, NONE, "motion dropout's"),
        (NONE, COST, "spatial module's"),
        (COST, COST, "both"),
        (NONE, GAIN, "neither"),
    ])
    def test_attribution_table(self, drop: str, spatial: str, words: str) -> None:
        assert words in w0d_readout.attribution(drop, spatial)

    def test_end_to_end_on_dada(self, tmp_path: Path) -> None:
        labels = {f"v{i}": np.array([0, 0, 1, 1, 0]) for i in range(30)}
        dirs = {}
        for arm, auc in (("W0", 0.80), ("W0D", 0.70), ("SV2", 0.70)):
            d = tmp_path / arm
            d.mkdir()
            rng = np.random.default_rng(len(arm))
            aucs = {v: auc + 0.01 * rng.standard_normal() for v in labels}
            (d / dada_whole_eval.CLIP_AUCS_JSON).write_text(
                json.dumps(aucs))
            ci = {"mean": auc, "low": auc - 0.02, "high": auc + 0.02}
            (d / dada_whole_eval.READOUT_JSON).write_text(
                json.dumps({"per_run_macro": {"s2024": auc}, "macro": ci, "micro": auc,
                            "video_auc": None, "monotone_ruler_macro": ci,
                            "middle_ruler_macro": ci}))
            write_clip_scores(d / "clip_scores.npz", {v: rng.random(5) for v in labels}, labels)
            dirs[arm] = d
        args = argparse.Namespace(
            w0_dada=dirs["W0"], w0d_dada=dirs["W0D"], sv2_dada=dirs["SV2"], w0_dota=None,
            w0d_dota=None, sv2_dota=None, out_dir=tmp_path / "rep")
        r = w0d_readout.run(args)
        assert r["dada"]["W0D-W0"]["sentence"] == COST
        assert r["dada"]["SV2-W0D"]["sentence"] == NONE
        assert "motion dropout's" in r["dada"]["attribution"] and r["dota"] is None
        md = (tmp_path / "rep" / "w0d_readout.md").read_text()
        assert "PRELIMINARY" in md and "| W0D - W0 |" in md and "pending" in md


class TestProbe:
    def _run(self, tmp_path: Path) -> tuple[Path, Config, Path, Path, Path]:
        data_dir, host, cells = _fixture(tmp_path)
        level = dict(zip(sa2v2_cells.LEVEL_KEYS, LEVEL, strict=True))
        (cells / constants.SV2_LEVEL_FILENAME).write_text(json.dumps(level))
        cfg = _cfg()
        cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 1, "cpu"
        out = tmp_path / "run"
        set_global_seed(cfg.train.seed)
        trainer = build_trainer(cfg, out, data_dir, host, text_encoder="stub", patch_dir=cells)
        cfg.save_yaml(out / "config.yaml")
        trainer.train()
        return out, cfg, data_dir, host, cells

    def test_adam_and_live_end_to_end(self, tmp_path: Path) -> None:
        run_dir, _cfg_used, data_dir, host, cells = self._run(tmp_path)
        adam = sv2_probe.run_adam(argparse.Namespace(
            run_dir=run_dir, out_dir=tmp_path / "adam", text_encoder="stub"))
        rows = adam["rows"]
        assert {"beta[0]", "beta[3]", "gamma", "log_temperature", "bias", "w", "w_u"} <= set(rows)
        assert rows["beta[0]"]["numel"] == 1 and rows["beta[0]"]["norm_init"] == pytest.approx(1.0)
        assert rows["gamma"]["norm_init"] == 0.0 and rows["w"]["norm_init"] == 0.0
        assert adam["lr_budget"] > 0 and all("reading" in r for r in rows.values())
        assert "Σ lr" in (tmp_path / "adam" / sv2_probe.ADAM_MD).read_text()

        for name, ckpt in (("init", None), ("final", run_dir / "checkpoint_last.pt")):
            live = sv2_probe.run_live(argparse.Namespace(
                config=run_dir / "config.yaml", name=name, checkpoint=ckpt, data_dir=data_dir,
                clip_dir=host, patch_dir=cells, knn_cache=None, num_batches=2,
                out_dir=tmp_path / name, text_encoder="stub"))
            s = live["summary"]
            assert {"beta[0]", "gamma", "bias"} <= set(s["scalars"])
            per = s["scalars"]["beta[0]"]
            assert set(per) == set(sv2_probe.TERMS)
            # host + patch + branch == total, gradient by gradient
            assert per["total"]["mean"] == pytest.approx(
                per["host"]["mean"] + per["patch"]["mean"] + per["branch"]["mean"], abs=1e-6)
            spread = s["logit_spread"]
            assert set(spread) == {"prior", "text", "residual", "l"}
        # w starts at 0, so at init the residual ranks nothing
        init = json.loads((tmp_path / "init" / sv2_probe.LIVE_JSON).read_text())
        assert init["summary"]["logit_spread"]["residual"] == pytest.approx(0.0, abs=1e-7)

    def test_split_terms_sums_to_total(self, tmp_path: Path) -> None:
        _run_dir, cfg, data_dir, host, cells = self._run(tmp_path)
        trainer = sv2_probe.build_probe_trainer(argparse.Namespace(
            out_dir=tmp_path / "p", data_dir=data_dir, clip_dir=host, knn_cache=None,
            patch_dir=cells, checkpoint=None, text_encoder="stub"), cfg)
        losses = trainer.compute_losses(collate_variable_length(
            [trainer.dataset[i] for i in range(4)]))
        terms = sv2_probe.split_terms(trainer, losses)
        torch.testing.assert_close(terms["host"] + terms["patch"] + terms["branch"],
                                   losses["total"])

    def test_refuses_a_host_run(self, tmp_path: Path) -> None:
        cfg = _host_cfg()
        cfg.save_yaml(tmp_path / "config.yaml")
        with pytest.raises(ValueError, match="not an SA²-v2"):
            sv2_probe.run_adam(argparse.Namespace(run_dir=tmp_path, out_dir=tmp_path / "o",
                                                  text_encoder="stub"))
