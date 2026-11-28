"""KIM edge fill (``KIM_RUN.md`` §68): no constant at the sequence edges; default unchanged. CPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from core import constants
from core.config import Config, validate_kim
from core.inference import load_model_for_scoring
from core.kim.module import KIM, fill_from_nearest
from core.models.kat_vad import KATVAD
from core.tests.fixtures import build_fixture
from core.train import build_trainer, set_global_seed

CPU = torch.device("cpu")
D = constants.HIDDEN_DIM


def test_fill_takes_the_next_real_row_else_the_previous() -> None:
    x = torch.arange(6, dtype=torch.float32).view(1, 6, 1) * 10
    have = torch.tensor([[0, 0, 1, 0, 1, 0]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 0]])
    out = fill_from_nearest(x, have, valid)[0, :, 0].tolist()
    assert out == [20.0, 20.0, 20.0, 40.0, 40.0, 50.0]  # last row invalid: left as is


def test_fill_falls_back_to_previous_and_leaves_rows_without_any_value() -> None:
    x = torch.tensor([[[1.0], [2.0], [3.0]], [[7.0], [8.0], [9.0]]])
    have = torch.tensor([[1, 0, 0], [0, 0, 0]])
    valid = torch.ones(2, 3)
    out = fill_from_nearest(x, have, valid)[..., 0].tolist()
    assert out == [[1.0, 1.0, 1.0], [7.0, 8.0, 9.0]]


def _kim(fill: bool) -> KIM:
    torch.manual_seed(0)
    return KIM(D, prototypes=True, edge_mask=True, edge_fill=fill)


def test_fill_changes_no_parameter_and_no_init_draw() -> None:
    off, on = _kim(False).state_dict(), _kim(True).state_dict()
    assert off.keys() == on.keys()
    for key, value in off.items():
        torch.testing.assert_close(on[key], value, msg=key)
    assert _kim(True).vel_gate.padding_mode == constants.KIM_GATE_PADDING_FILL
    assert _kim(False).vel_gate.padding_mode == "zeros"


def test_no_structural_zero_at_the_start_with_fill() -> None:
    torch.manual_seed(1)
    vt, mask = torch.randn(2, 12, D), torch.ones(2, 12)
    width = constants.KIM_WIDTH
    off = _kim(False).eval()(vt, mask).kin_feat
    on = _kim(True).eval()(vt, mask).kin_feat
    vel_off, acc_off = off[..., width:2 * width], off[..., 2 * width:]
    vel_on, acc_on = on[..., width:2 * width], on[..., 2 * width:]
    assert vel_off[:, :2].abs().sum() == 0 and acc_off[:, :3].abs().sum() == 0
    assert (vel_on.abs().sum(-1) > 0).all() and (acc_on.abs().sum(-1) > 0).all()
    torch.testing.assert_close(on[..., :width], off[..., :width])  # the state is untouched


def test_fill_keeps_padded_rows_zero() -> None:
    torch.manual_seed(2)
    vt, mask = torch.randn(1, 8, D), torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    feat = _kim(True).eval()(vt, mask).kin_feat
    assert feat[0, 5:].abs().sum() == 0


def _l5(fill: bool = True) -> Config:
    cfg = Config()
    cfg.kip.enabled = False
    cfg.model.motion_dim = 0
    cfg.v2.motion = constants.V2_OFF
    cfg.model.score_head_kernel = constants.DADA_ORIGIN_SCORE_HEAD_KERNEL
    cfg.loss.mil_topk_pct = constants.DADA_ORIGIN_MIL_TOPK_PCT
    cfg.dvs.theta = cfg.dvs.theta_ego = 1.0
    cfg.kim.enabled = cfg.kim.prototypes = cfg.kim.edge_mask = True
    cfg.kim.edge_fill = fill
    return cfg


@pytest.mark.parametrize(("change", "match"), [
    ({"enabled": False, "prototypes": False, "edge_mask": False}, "edge_fill needs"),
    ({"capacity_control": True, "edge_mask": False}, "edge_fill needs"),
])
def test_config_refuses_fill_without_pairs(change: dict[str, bool], match: str) -> None:
    cfg = _l5()
    for key, value in change.items():
        setattr(cfg.kim, key, value)
    with pytest.raises(ValueError, match=match):
        validate_kim(cfg)


def test_host_starts_equal_with_fill() -> None:
    torch.manual_seed(0)
    plain = KATVAD.from_config(_l5(False))
    torch.manual_seed(0)
    filled = KATVAD.from_config(_l5(True))
    torch.manual_seed(3)
    rows, lengths, text = torch.randn(2, 9, D), torch.tensor([9, 6]), torch.randn(3, D)
    torch.testing.assert_close(filled.eval()(rows, lengths, text)["cls_bin_logits"],
                               plain.eval()(rows, lengths, text)["cls_bin_logits"])


def test_trains_and_scores_with_fill_from_a_default_config(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path / "fx")
    cfg = _l5()
    cfg.kim.warmup_steps = 1
    cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 2, "cpu"
    set_global_seed(cfg.train.seed)
    out = tmp_path / "run"
    trainer = build_trainer(cfg, out, fixture.data_dir, fixture.clip_dir, text_encoder="stub")
    cfg.save_yaml(out / "config.yaml")
    trainer.train()
    saved = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert saved["kim"]["edge_fill"]
    model = load_model_for_scoring(Config(), CPU, out / "checkpoint_last.pt", None, "stub")
    assert isinstance(model.kim, KIM) and model.kim.edge_fill
    scores = model(torch.randn(1, 12, D), torch.tensor([12]), torch.randn(2, D))["cls_bin_logits"]
    assert torch.isfinite(scores).all()
