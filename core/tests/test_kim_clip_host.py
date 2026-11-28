"""KIM on the CLIP-only host (``KIM_RUN.md`` §64): L5 / L5cap start as L0 and train to score."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from core import constants
from core.config import Config, validate_kim
from core.inference import load_model_for_scoring
from core.kim.module import KIM, KIMCapacity
from core.models.kat_vad import KATVAD
from core.tests.fixtures import build_fixture
from core.train import build_trainer, set_global_seed

CPU = torch.device("cpu")
D = constants.HIDDEN_DIM


def _l0() -> Config:
    """L0-like host: plain CLIP rows, no motion stream, KIP off, DVS off."""
    cfg = Config()
    cfg.kip.enabled = False
    cfg.model.motion_dim = 0
    cfg.v2.motion = constants.V2_OFF
    cfg.model.score_head_kernel = constants.DADA_ORIGIN_SCORE_HEAD_KERNEL
    cfg.loss.mil_topk_pct = constants.DADA_ORIGIN_MIL_TOPK_PCT
    cfg.dvs.theta = cfg.dvs.theta_ego = 1.0
    return cfg


def _l5(cap: bool = False) -> Config:
    cfg = _l0()
    cfg.kim.enabled = cfg.kim.prototypes = True
    if cap:
        cfg.kim.capacity_control = True
    else:
        cfg.kim.edge_mask = True
    return cfg


def _rows(lengths: tuple[int, ...] = (9, 5)) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(len(lengths), max(lengths), D), torch.tensor(lengths)


@pytest.mark.parametrize("cap", [False, True])
def test_config_is_valid_and_builds_the_module(cap: bool) -> None:
    cfg = _l5(cap)
    validate_kim(cfg)
    model = KATVAD.from_config(cfg)
    assert isinstance(model.kim, KIMCapacity if cap else KIM)


@pytest.mark.parametrize("cap", [False, True])
def test_host_init_and_first_scores_equal_l0(cap: bool) -> None:
    """Private RNG (D13) and zero-init return: L5 / L5cap start draw for draw as L0."""
    torch.manual_seed(0)
    host = KATVAD.from_config(_l0())
    after_host = torch.rand(3)
    torch.manual_seed(0)
    kim = KATVAD.from_config(_l5(cap))
    torch.testing.assert_close(torch.rand(3), after_host)
    on = kim.state_dict()
    for key, value in host.state_dict().items():
        torch.testing.assert_close(on[key], value, msg=key)
    rows, lengths = _rows()
    text = torch.randn(3, D)
    torch.testing.assert_close(kim.eval()(rows, lengths, text)["cls_bin_logits"],
                               host.eval()(rows, lengths, text)["cls_bin_logits"])


def _train(cfg: Config, data_dir: Path, clip_dir: Path, out: Path) -> list[dict[str, float]]:
    cfg.kim.warmup_steps = 1
    cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 2, "cpu"
    set_global_seed(cfg.train.seed)
    trainer = build_trainer(cfg, out, data_dir, clip_dir, text_encoder="stub", log_every=1)
    cfg.save_yaml(out / "config.yaml")
    trainer.train()
    return [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if x]


@pytest.mark.parametrize("cap", [False, True])
def test_trains_to_score_with_the_same_start_as_l0(tmp_path: Path, cap: bool) -> None:
    fixture = build_fixture(tmp_path / "fx")
    off = _train(_l0(), fixture.data_dir, fixture.clip_dir, tmp_path / "l0")
    on = _train(_l5(cap), fixture.data_dir, fixture.clip_dir, tmp_path / "l5")
    assert on[0]["mil"] == pytest.approx(off[0]["mil"], abs=constants.KIM_START_TOL)
    assert "kim_mil" in on[0] and "kim_mil" not in off[0]
    model = load_model_for_scoring(Config(), CPU, tmp_path / "l5" / "checkpoint_last.pt", None,
                                   "stub")
    assert isinstance(model.kim, KIMCapacity if cap else KIM)
    saved = yaml.safe_load((tmp_path / "l5" / "config.yaml").read_text(encoding="utf-8"))
    assert saved["model"]["motion_dim"] == 0 and saved["kim"]["enabled"]
    rows, lengths = _rows()
    assert torch.isfinite(model(rows, lengths, torch.randn(2, D))["cls_bin_logits"]).all()
