"""core.tools.kim_grad_probe (KIM_RUN §66 NK1): refuses unprobeable models, is read-only, CPU."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from core import constants
from core.config import Config
from core.tests.fixtures import build_fixture
from core.tools import kim_grad_probe as gp
from core.train import Trainer, build_trainer, set_global_seed, warm_start_model


def _l5(nkc: float = constants.KIM_NKC_WEIGHT) -> Config:
    cfg = Config()
    cfg.kip.enabled = False
    cfg.model.motion_dim = 0
    cfg.v2.motion = constants.V2_OFF
    cfg.model.score_head_kernel = constants.DADA_ORIGIN_SCORE_HEAD_KERNEL
    cfg.loss.mil_topk_pct = constants.DADA_ORIGIN_MIL_TOPK_PCT
    cfg.dvs.theta = cfg.dvs.theta_ego = 1.0
    cfg.kim.enabled = cfg.kim.prototypes = cfg.kim.edge_mask = True
    cfg.kim.nkc_weight = nkc
    cfg.kim.warmup_steps = 1
    cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 2, "cpu"
    return cfg


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Config, Path, Path, Path]:
    root = tmp_path_factory.mktemp("kgp")
    fixture = build_fixture(root / "fx")
    cfg = _l5()
    set_global_seed(cfg.train.seed)
    trainer = build_trainer(cfg, root / "run", fixture.data_dir, fixture.clip_dir,
                            text_encoder="stub")
    trainer.train()
    return cfg, fixture.data_dir, fixture.clip_dir, root / "run" / "checkpoint_last.pt"


def _trainer(cfg: Config, data: Path, clip: Path, out: Path, ckpt: Path | None) -> Trainer:
    set_global_seed(cfg.train.seed)
    trainer = build_trainer(cfg, out, data, clip, text_encoder="stub")
    if ckpt is not None:
        warm_start_model(trainer.model, ckpt, flag="--checkpoint")
    return trainer


def test_probe_reports_both_kim_terms_and_is_read_only(
        trained: tuple[Config, Path, Path, Path], tmp_path: Path) -> None:
    cfg, data, clip, ckpt = trained
    trainer = _trainer(cfg, data, clip, tmp_path, ckpt)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    payload = gp.run_probe(trainer, num_batches=2, epoch=0)
    after = trainer.model.state_dict()
    for key, value in before.items():
        torch.testing.assert_close(after[key], value, msg=key)
    s = payload["summary"]
    assert payload["batches"] == 2 and set(s["weighted"]) == {gp.TASK, gp.MIL, gp.NKC}
    assert {"temporal_encoder", "kim", "fusion", "heads"} <= set(s["grad_norm"])
    assert s["grad_norm"]["kim"][gp.NKC] > 0 and s["grad_norm"]["temporal_encoder"][gp.NKC] > 0
    b = s["branch"]
    assert b["lambda_eq"] == pytest.approx(cfg.kim.nkc_weight / b["nkc_over_mil"], rel=0.5)
    assert -1.0 <= s["trunk"][f"cos_{gp.NKC}_task"] <= 1.0
    assert "lambda_eq" in gp.render({**payload, "row": "L5"})


def test_probe_refuses_unbuilt_prototypes(trained: tuple[Config, Path, Path, Path],
                                          tmp_path: Path) -> None:
    cfg, data, clip, _ = trained
    trainer = _trainer(cfg, data, clip, tmp_path, None)
    with pytest.raises(ValueError, match="prototypes are not built"):
        gp.run_probe(trainer, num_batches=1, epoch=0)


def test_probe_refuses_zero_weight(trained: tuple[Config, Path, Path, Path],
                                   tmp_path: Path) -> None:
    _, data, clip, ckpt = trained
    trainer = _trainer(_l5(nkc=0.0), data, clip, tmp_path, ckpt)
    with pytest.raises(ValueError, match="nkc_weight is 0"):
        gp.run_probe(trainer, num_batches=1, epoch=0)


def test_probe_refuses_a_model_without_kim(trained: tuple[Config, Path, Path, Path],
                                           tmp_path: Path) -> None:
    _, data, clip, _ = trained
    cfg = _l5()
    cfg.kim.enabled = cfg.kim.prototypes = cfg.kim.edge_mask = False
    trainer = _trainer(cfg, data, clip, tmp_path, None)
    with pytest.raises(ValueError, match="no KIM term"):
        gp.run_probe(trainer, num_batches=1, epoch=0)


def test_weight_table_covers_kim_terms() -> None:
    cfg = _l5()
    weights = gp.kim_term_weights(cast(Trainer, SimpleNamespace(cfg=cfg)))
    assert weights[gp.MIL] == cfg.kim.mil_weight and weights[gp.NKC] == cfg.kim.nkc_weight
