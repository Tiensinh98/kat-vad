"""W_u arms (core/docs/final/WU_ARMS.md): non-zero init and a learning-rate multiplier on W_u.

CPU-only, data-free. Every knob must leave the shipped path untouched at its default, and the
init must not move any other module's random stream (the arms differ from W0 in W_u only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR

from core import constants, train
from core.config import Config, load_config, validate_v2
from core.models.kat_vad import KATVAD
from core.models.motion_residual import MotionResidual
from core.tests.fixtures import FixtureLayout
from core.tests.test_v2_model import (  # noqa: F401  (module-scoped pytest fixtures)
    D_S,
    ENCODER_S,
    _args,
    a3_cache,
    fixture,
)
from core.tools import stp_readout

torch.set_num_threads(1)
SMALL = 32
MOTION_DIM = 6


def _model(init_std: float = 0.0, seed: int = 0) -> KATVAD:
    torch.manual_seed(0)
    return KATVAD(
        hidden_dim=SMALL, temporal_heads=4, fusion_heads=4, motion_dim=MOTION_DIM,
        motion_init_std=init_std, motion_init_seed=seed,
    ).eval()


def _motion_cfg(*overrides: str) -> Config:
    base = ["kip.enabled=false", f"v2.motion={ENCODER_S}", f"model.motion_dim={D_S}"]
    return load_config(None, [*base, *overrides])


class TestConfig:
    def test_defaults_are_the_shipped_path(self) -> None:
        cfg = Config()
        assert cfg.model.motion_init_std == 0.0 and cfg.train.motion_lr_mult == 1.0

    @pytest.mark.parametrize("arm", sorted(constants.WU_ARMS))
    def test_every_arm_validates_and_applies(self, arm: str) -> None:
        sets = [f"{k}={v}" for k, v in constants.WU_ARMS[arm].items()]
        cfg = _motion_cfg(*sets)
        validate_v2(cfg)
        assert cfg.model.motion_init_std == constants.WU_ARMS[arm].get("model.motion_init_std", 0.0)
        assert cfg.train.motion_lr_mult == constants.WU_ARMS[arm].get("train.motion_lr_mult", 1.0)

    def test_the_control_is_the_shipped_host(self) -> None:
        assert constants.WU_ARMS[constants.WU_CONTROL_ARM] == {}

    @pytest.mark.parametrize(
        ("override", "match"),
        [
            ("model.motion_init_std=-0.1", "motion_init_std must be >= 0"),
            ("train.motion_lr_mult=0", "motion_lr_mult must be > 0"),
        ],
    )
    def test_out_of_range_raises(self, override: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_v2(_motion_cfg(override))

    @pytest.mark.parametrize("override", ["model.motion_init_std=0.003", "train.motion_lr_mult=10"])
    def test_a_knob_without_a_motion_stream_raises(self, override: str) -> None:
        cfg = load_config(None, ["kip.enabled=false", override])
        with pytest.raises(ValueError, match="need a motion stream"):
            validate_v2(cfg)


class TestInit:
    def test_zero_is_the_default(self) -> None:
        block = MotionResidual(16, 8)
        assert float(block.proj.weight.abs().max()) == 0.0
        assert float(block.proj.bias.abs().max()) == 0.0

    def test_std_sets_the_weight_scale_and_leaves_the_bias_zero(self) -> None:
        block = MotionResidual(64, 64, init_std=constants.WU_INIT_STD, init_seed=1)
        std = float(block.proj.weight.std())
        assert std == pytest.approx(constants.WU_INIT_STD, rel=0.1)
        assert float(block.proj.bias.abs().max()) == 0.0

    def test_same_seed_same_weight_other_seed_other_weight(self) -> None:
        a = MotionResidual(16, 8, init_std=0.01, init_seed=3).proj.weight
        b = MotionResidual(16, 8, init_std=0.01, init_seed=3).proj.weight
        c = MotionResidual(16, 8, init_std=0.01, init_seed=4).proj.weight
        torch.testing.assert_close(a, b)
        assert not torch.equal(a, c)

    def test_negative_std_raises(self) -> None:
        with pytest.raises(ValueError, match="init_std"):
            MotionResidual(16, 8, init_std=-1.0)

    def test_the_init_draws_nothing_from_the_global_rng(self) -> None:
        """nn.Linear's own default init always draws; the W_u draw must add nothing to it."""
        torch.manual_seed(5)
        MotionResidual(16, 8)
        zero_init_after = torch.randn(4)
        torch.manual_seed(5)
        MotionResidual(16, 8, init_std=0.01, init_seed=0)
        torch.testing.assert_close(torch.randn(4), zero_init_after)

    def test_only_w_u_differs_from_the_zero_init_model(self) -> None:
        """Every other parameter of WI == W0 (same seed): the arms differ in W_u alone."""
        w0, wi = _model(0.0).state_dict(), _model(constants.WU_INIT_STD).state_dict()
        assert w0.keys() == wi.keys()
        differing = [k for k in w0 if not torch.equal(w0[k], wi[k])]
        assert differing == ["motion_residual.proj.weight"]

    def test_a_non_zero_w_u_makes_the_motion_term_live_at_step_zero(self) -> None:
        model = _model(0.05)
        x = torch.randn(2, 7, SMALL)
        rows = torch.cat([x, torch.randn(2, 7, MOTION_DIM)], dim=-1)
        out = model(rows, torch.tensor([7, 5]), class_feats=torch.randn(3, SMALL))
        assert float(out["motion_share"]) > 0.0


class TestParamGroups:
    def test_multiplier_one_is_the_flat_list(self) -> None:
        model = _model()
        got = train.optimizer_param_groups(model, 5e-5, 1.0)
        assert all(isinstance(p, torch.Tensor) for p in got)
        assert [id(p) for p in got] == [id(p) for p in model.parameters() if p.requires_grad]

    def test_w_u_gets_its_own_group_at_the_multiplied_rate(self) -> None:
        model = _model()
        groups = train.optimizer_param_groups(model, 5e-5, 10.0)
        assert [g["lr"] for g in groups] == [pytest.approx(5e-5), pytest.approx(5e-4)]
        motion = {id(p) for p in model.motion_residual.parameters()}  # type: ignore[union-attr]
        assert {id(p) for p in groups[1]["params"]} == motion
        every = [id(p) for g in groups for p in g["params"]]
        assert sorted(every) == sorted(id(p) for p in model.parameters() if p.requires_grad)
        assert len(every) == len(set(every))

    def test_no_motion_stream_raises(self) -> None:
        plain = KATVAD(hidden_dim=SMALL, temporal_heads=4, fusion_heads=4)
        with pytest.raises(ValueError, match="needs a motion stream"):
            train.optimizer_param_groups(plain, 5e-5, 10.0)

    def test_the_schedule_scales_both_groups_alike(self) -> None:
        model = _model()
        opt = AdamW(train.optimizer_param_groups(model, 5e-5, 10.0), lr=5e-5, weight_decay=0.01)
        sched = LambdaLR(opt, train.cosine_warmup_lambda(2, 20))
        for _ in range(6):
            opt.step()
            sched.step()
        base, motion = (g["lr"] for g in opt.param_groups)
        assert motion / base == pytest.approx(10.0)
        assert base < 5e-5  # the factor is applied, not frozen at the start


class TestEndToEnd:
    def test_wil_trains_with_a_live_w_u_and_two_lr_groups(
        self, fixture: FixtureLayout, a3_cache: Path, tmp_path: Path  # noqa: F811
    ) -> None:
        out = tmp_path / "wil"
        extra = [f"{k}={v}" for k, v in constants.WU_ARMS["WIL"].items()]
        argv = _args(fixture, a3_cache, out, v2=True)
        for item in extra:
            argv += ["--set", item]
        train.main(argv)
        records = [json.loads(x) for x in (out / train.METRICS_FILENAME).read_text().splitlines()]
        assert records[0]["w_u_norm"] > 0.0, "W_u should start non-zero in arm WI/WIL"
        payload = torch.load(out / train.CHECKPOINT_LAST, map_location="cpu", weights_only=False)
        lrs = [g["initial_lr"] for g in payload["optimizer"]["param_groups"]]
        assert lrs[1] / lrs[0] == pytest.approx(constants.WU_LR_MULT)

    def test_the_default_still_starts_at_zero(
        self, fixture: FixtureLayout, a3_cache: Path, tmp_path: Path  # noqa: F811
    ) -> None:
        out = tmp_path / "w0"
        train.main(_args(fixture, a3_cache, out, v2=True))
        first = json.loads((out / train.METRICS_FILENAME).read_text().splitlines()[0])
        assert first["w_u_norm"] == 0.0
        payload = torch.load(out / train.CHECKPOINT_LAST, map_location="cpu", weights_only=False)
        assert len(payload["optimizer"]["param_groups"]) == 1


class TestReadoutNaming:
    def _parse(self, *extra: str) -> argparse.Namespace:
        return stp_readout.build_arg_parser().parse_args(
            ["--control", "W0", "d0", "--arm", "WI", "d1", "--seeds", "2024",
             "--metadata", "m", "--split-file", "s", "--out-dir", "o", *extra]
        )

    def test_the_stp_defaults_are_unchanged(self) -> None:
        args = self._parse()
        assert (args.name, args.title, args.rule) == (
            "stp_readout", "LaGoVAD-STP read-out", "core/docs/final/STP_BUILD.md §3")

    def test_the_w_u_read_out_can_name_itself(self) -> None:
        args = self._parse("--name", "wu_readout", "--title", "W_u arms read-out")
        assert (args.name, args.title) == ("wu_readout", "W_u arms read-out")
