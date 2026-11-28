"""§59: KIM reads its own temporal encoder (``kim.input_encoder``), rule ``KIM_RUN.md`` §59."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from core import constants
from core.config import Config, validate_kim
from core.data.collate import padding_mask
from core.inference import load_model_for_scoring
from core.kim.module import KIM, PrivateRNG
from core.models.kat_vad import KATVAD
from core.models.temporal_encoder import TemporalEncoder
from core.tests.test_kim import CPU, D, _fixture, _kim, _rows
from core.train import build_trainer, set_global_seed

CLIP = constants.KIM_INPUT_ENCODER_CLIP
FUSED = constants.KIM_INPUT_ENCODER_FUSED


def _cfg(source: str = CLIP) -> Config:
    cfg = _kim(prototypes=True)
    cfg.kim.edge_mask = True
    cfg.kim.input_encoder = source
    return cfg


def _build(cfg: Config, seed: int = 0) -> KATVAD:
    torch.manual_seed(seed)
    return KATVAD.from_config(cfg)


class TestConfig:
    @pytest.mark.parametrize("source", [CLIP, FUSED])
    def test_accepts_both_sources(self, source: str) -> None:
        validate_kim(_cfg(source))

    def test_refuses_what_no_arm_defines(self) -> None:
        bad_value = _cfg("raw")
        off = _cfg()
        off.kim.enabled = off.kim.prototypes = off.kim.edge_mask = False
        cap = _cfg()
        cap.kim.edge_mask, cap.kim.capacity_control = False, True
        no_motion = _cfg()
        no_motion.model.motion_dim = 0
        for cfg, match in ((bad_value, "must be one of"), (off, "needs kim.enabled"),
                           (cap, "no capacity_control"), (no_motion, "motion stream")):
            with pytest.raises(ValueError, match=match):
                validate_kim(cfg)

    def test_default_is_none_and_builds_no_encoder(self) -> None:
        cfg = _kim(prototypes=True)
        assert cfg.kim.input_encoder == constants.KIM_INPUT_ENCODER_NONE
        model = KATVAD.from_config(cfg)
        assert model.kim is not None and model.kim.encoder is None
        assert not any(k.startswith("kim.encoder.") for k in model.state_dict())


class TestInit:
    @pytest.mark.parametrize("source", [CLIP, FUSED])
    def test_host_and_kim_init_equal_the_arm_without_it(self, source: str) -> None:
        """CE2: the host's and KIM's draws are K5dfi's; the only new keys are the encoder's."""
        plain = _build(_kim(prototypes=True)).state_dict()
        cfg = _kim(prototypes=True)
        cfg.kim.input_encoder = source
        with_enc = _build(cfg).state_dict()
        for key, value in plain.items():
            torch.testing.assert_close(with_enc[key], value, msg=key)
        extra = set(with_enc) - set(plain)
        assert extra and all(k.startswith("kim.encoder.") for k in extra)
        torch.manual_seed(0)
        KATVAD.from_config(_kim(prototypes=True))
        after_plain = torch.rand(3)
        torch.manual_seed(0)
        KATVAD.from_config(cfg)
        torch.testing.assert_close(torch.rand(3), after_plain)  # data order unchanged

    def test_encoder_has_the_hosts_configuration(self) -> None:
        model = _build(_cfg())
        assert model.kim is not None and isinstance(model.kim.encoder, TemporalEncoder)
        host = {k: v.shape for k, v in model.temporal_encoder.state_dict().items()}
        enc = {k: v.shape for k, v in model.kim.encoder.state_dict().items()}
        assert enc == host
        assert model.kim.encoder.window_size == model.temporal_encoder.window_size

    @pytest.mark.parametrize("train", [False, True])
    def test_step_one_forward_equals_the_arm_without_it(self, train: bool) -> None:
        """W_out = 0 and the private dropout stream: the host draws exactly what K5dfi draws."""
        plain, with_enc = _build(_kim(prototypes=True)), _build(_cfg())
        plain.train(train)
        with_enc.train(train)
        rows, lengths = _rows()
        text = torch.randn(3, D)
        torch.manual_seed(7)
        a = plain(rows, lengths, text)["cls_bin_logits"]
        after_a = torch.rand(3)
        torch.manual_seed(7)
        b = with_enc(rows, lengths, text)["cls_bin_logits"]
        after_b = torch.rand(3)
        torch.testing.assert_close(b, a)
        torch.testing.assert_close(after_b, after_a)


class TestSource:
    def _kin(self, model: KATVAD, rows: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        feat: torch.Tensor = model.eval()(rows, lengths, torch.randn(3, D))["kim_feat"]
        return feat

    def test_clip_reads_the_clip_columns_only_and_fused_reads_w_u(self) -> None:
        clip, fused = _build(_cfg(CLIP)), _build(_cfg(FUSED))
        for model in (clip, fused):
            assert model.motion_residual is not None
            torch.nn.init.normal_(model.motion_residual.proj.weight)  # W_u away from zero
        rows, lengths = _rows()
        moved = rows.clone()
        moved[..., D:] += torch.randn_like(moved[..., D:])  # only the motion columns change
        torch.testing.assert_close(self._kin(clip, moved, lengths), self._kin(clip, rows, lengths))
        assert not torch.allclose(self._kin(fused, moved, lengths),
                                  self._kin(fused, rows, lengths))

    def test_return_path_stays_v_t(self) -> None:
        model = _build(_cfg())
        assert model.kim is not None
        torch.nn.init.normal_(model.kim.w_out.weight)
        rows, lengths = _rows()
        out = model.eval()(rows, lengths, torch.randn(3, D))
        added = out["vis_feats"] - out["vt"]
        torch.testing.assert_close(added, model.kim.w_out(out["kim_feat"]))

    def test_module_refuses_a_missing_or_stray_source(self) -> None:
        plain = KIM(D)
        mask = padding_mask(torch.tensor([5]), 5)
        with pytest.raises(ValueError, match="without an input encoder"):
            plain(torch.randn(1, 5, D), mask, source=torch.randn(1, 5, D))
        plain.attach_encoder(TemporalEncoder(hidden_size=D), dropout_seed=1)
        with pytest.raises(ValueError, match="needs source rows"):
            plain(torch.randn(1, 5, D), mask)


class TestPrivateRNG:
    def test_stream_advances_and_the_host_is_restored(self) -> None:
        stream = PrivateRNG(3)
        torch.manual_seed(0)
        with stream.active(CPU):
            first = torch.rand(4)
        host_draw = torch.rand(4)
        with stream.active(CPU):
            second = torch.rand(4)
        torch.manual_seed(0)
        torch.testing.assert_close(torch.rand(4), host_draw)  # as if the stream never ran
        assert not torch.equal(first, second)  # the stream does not repeat its masks
        generator = torch.Generator().manual_seed(3)
        torch.testing.assert_close(first, torch.rand(4, generator=generator))


class TestTraining:
    @pytest.mark.parametrize("source", [CLIP, FUSED])
    def test_trains_logs_and_scores(self, tmp_path: Path, source: str) -> None:
        data_dir, host = _fixture(tmp_path)
        cfg = _cfg(source)
        cfg.kim.warmup_steps = 0
        cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 2, "cpu"
        set_global_seed(cfg.train.seed)
        out = tmp_path / "run"
        trainer = build_trainer(cfg, out, data_dir, host, text_encoder="stub", log_every=1)
        cfg.save_yaml(out / "config.yaml")
        trainer.train()
        logged = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if x]
        stepped = [r for r in logged if "kim_enc_grad_norm" in r]
        assert stepped and all(r["kim_enc_grad_norm"] > 0.0 for r in stepped)
        assert stepped[-1]["kim_enc_drift"] > stepped[0]["kim_enc_drift"] > 0.0
        model = load_model_for_scoring(Config(), CPU, out / "checkpoint_last.pt", None, "stub")
        assert model.kim is not None and isinstance(model.kim.encoder, TemporalEncoder)
        assert model.kim_input == source
        trained = trainer.model.kim
        assert trained is not None and trained.encoder is not None
        for key, value in trained.encoder.state_dict().items():
            torch.testing.assert_close(model.kim.encoder.state_dict()[key], value, msg=key)
        rows, lengths = _rows()
        assert torch.isfinite(model(rows, lengths, torch.randn(2, D))["cls_bin_logits"]).all()

    def test_plain_kim_logs_no_encoder_key(self, tmp_path: Path) -> None:
        data_dir, host = _fixture(tmp_path)
        cfg = _kim(prototypes=False)
        cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 1, "cpu"
        set_global_seed(cfg.train.seed)
        out = tmp_path / "run"
        build_trainer(cfg, out, data_dir, host, text_encoder="stub", log_every=1).train()
        logged = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if x]
        assert not any("kim_enc_grad_norm" in r or "kim_enc_drift" in r for r in logged)

