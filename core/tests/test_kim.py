"""LaGoVAD-KIM (rule ``core/docs/final/KIM_RUN.md`` §7): module, losses, wiring, trainer."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants
from core.config import Config, validate_kim
from core.data.collate import collate_variable_length, padding_mask
from core.data.dataset import DVSFeatureDataset
from core.inference import load_model_for_scoring
from core.kim.losses import kim_anchors, kim_mil_loss, kim_nkc_loss, normal_steps
from core.kim.module import KIM, first_difference, kim_scale, pair_mask
from core.models.kat_vad import KATVAD
from core.models.text_encoding import make_text_encoder
from core.tests.fixtures import build_fixture
from core.train import Trainer, build_trainer, load_class_names, set_global_seed

CPU = torch.device("cpu")
D = constants.HIDDEN_DIM
MOTION = 384
BASE = D + MOTION


def _host() -> Config:
    """W0-like host: CLIP + motion columns, KIP off, DVS off (KIM_RUN.md D1-D2)."""
    cfg = Config()
    cfg.kip.enabled = False
    cfg.model.motion_dim = MOTION
    cfg.v2.motion = constants.V5_ENCODER
    cfg.model.score_head_kernel = 3
    cfg.loss.mil_topk_pct = 5
    cfg.dvs.theta = cfg.dvs.theta_ego = 1.0
    return cfg


def _kim(prototypes: bool = True) -> Config:
    cfg = _host()
    cfg.kim.enabled, cfg.kim.prototypes = True, prototypes
    return cfg


def _rows(lengths: tuple[int, ...] = (9, 5), seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return torch.randn(len(lengths), max(lengths), BASE), torch.tensor(lengths)


class TestFeatures:
    def test_pair_mask_needs_both_steps_valid(self) -> None:
        mask = padding_mask(torch.tensor([4, 2]), 4)
        expected = torch.tensor([[0.0, 1, 1, 1], [0, 1, 0, 0]])
        torch.testing.assert_close(pair_mask(mask), expected)

    def test_first_difference_and_scale(self) -> None:
        x = torch.tensor([[[0.0], [1.0], [3.0], [100.0]]])  # last step is padding below
        d1 = first_difference(x)
        torch.testing.assert_close(d1[0, :, 0], torch.tensor([0.0, 1, 2, 97]))
        pair = pair_mask(padding_mask(torch.tensor([3]), 4))
        assert float(kim_scale(d1, pair)[0]) == pytest.approx(1.0)  # median of {1, 2}
        empty = pair_mask(padding_mask(torch.tensor([1]), 4))
        assert float(kim_scale(d1, empty)[0]) == 1.0

    def test_padding_never_reaches_the_features(self) -> None:
        module = KIM(D)
        vt = torch.randn(2, 6, D)
        mask = padding_mask(torch.tensor([6, 3]), 6)
        out = module(vt, mask)
        assert out.kin_feat[1, 3:].abs().sum() == 0
        changed = vt.clone()
        changed[1, 3:] = 1e3  # padded steps may hold anything
        again = module(changed, mask)
        torch.testing.assert_close(again.kin_feat[1, :3], out.kin_feat[1, :3])
        torch.testing.assert_close(again.score[1, :3], out.score[1, :3])

    def test_first_step_has_no_velocity_and_scale_is_detached(self) -> None:
        module = KIM(D)
        vt = torch.randn(1, 5, D, requires_grad=True)
        out = module(vt, torch.ones(1, 5))
        width = module.width
        assert out.kin_feat[0, 0, width:].abs().sum() == 0  # vel and acc are 0 at t = 0
        assert out.kin_feat[0, 1, 2 * width :].abs().sum() == 0  # acc needs two pairs
        assert not out.scale.requires_grad

    def test_zero_init_return_path(self) -> None:
        module = KIM(D)
        vt = torch.randn(2, 4, D)
        out = module(vt, torch.ones(2, 4))
        torch.testing.assert_close(out.features, vt)
        assert float(module.w_out_norm()) == 0.0
        out.features.sum().backward()
        assert module.w_out.weight.grad is not None and module.w_out.weight.grad.abs().sum() > 0

    def test_gates_start_at_one_half(self) -> None:
        module = KIM(D)
        for conv in (module.vel_gate, module.acc_gate):
            assert conv.bias is not None and float(conv.bias.abs().sum()) == 0.0

    def test_prototype_weighting_only_when_both_exist(self) -> None:
        module = KIM(D, prototypes=True)
        vt, mask = torch.randn(1, 4, D), torch.ones(1, 4)
        plain = module(vt, mask)
        torch.testing.assert_close(plain.score, torch.softmax(plain.logits, -1)[..., 1])
        feats = plain.kin_feat
        module.update_prototypes(feats, torch.ones(1, 4), torch.zeros(1, 4), 0.1)
        assert not module.prototypes_ready  # mu_1 still missing
        module.update_prototypes(feats, torch.zeros(1, 4), torch.ones(1, 4), 0.1)
        assert module.prototypes_ready
        weighted = module(vt, mask)
        assert not torch.allclose(weighted.score, plain.score)

    def test_prototype_momentum_and_no_grad(self) -> None:
        module = KIM(D, prototypes=True)
        a = torch.ones(1, 2, 3 * module.width, requires_grad=True)
        both = torch.ones(1, 2)
        module.update_prototypes(a, both, both, 0.1)
        module.update_prototypes(a * 3, both, torch.zeros(1, 2), 0.1)
        torch.testing.assert_close(module.prototypes[0], torch.full((3 * module.width,), 1.2))
        torch.testing.assert_close(module.prototypes[1], torch.ones(3 * module.width))
        assert not module.prototypes.requires_grad
        assert "prototypes" in module.state_dict() and "proto_init" in module.state_dict()


class TestLosses:
    def test_kim_mil_averages_top_k_probabilities(self) -> None:
        score = torch.tensor([[0.9, 0.1, 0.8, 0.2, 0.0], [0.3, 0.6, 0.0, 0.0, 0.0]])
        labels, lengths = torch.tensor([1.0, 0.0]), torch.tensor([5, 2])
        loss = kim_mil_loss(score, labels, lengths, topk_pct=2)  # k = 2 and 1
        p = torch.tensor([0.85, 0.6])
        expected = -(labels * p.log() + (1 - labels) * (1 - p).log()).mean()
        assert float(loss) == pytest.approx(float(expected), rel=1e-5)

    def test_anchors_are_top_k_of_abnormal_bags_above_threshold(self) -> None:
        score = torch.tensor([[0.9, 0.4, 0.8, 0.7], [0.99, 0.99, 0.0, 0.0]])
        anchors = kim_anchors(score, torch.tensor([1.0, 0.0]), torch.tensor([4, 2]), 2, 0.5)
        expected = torch.tensor([[True, False, True, False], [False, False, False, False]])
        assert torch.equal(anchors, expected)
        low = kim_anchors(score * 0.1, torch.tensor([1.0, 0.0]), torch.tensor([4, 2]), 2, 0.5)
        assert not low.any()

    def test_normal_steps_are_valid_steps_of_label_zero_bags(self) -> None:
        mask = padding_mask(torch.tensor([3, 2]), 3)
        got = normal_steps(torch.tensor([1.0, 0.0]), mask)
        assert torch.equal(got, torch.tensor([[False] * 3, [True, True, False]]))

    def test_nkc_by_hand(self) -> None:
        feats = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [-1.0, 0.0]]])
        anchors = torch.tensor([[True, False], [False, False]])
        negatives = torch.tensor([[False, False], [True, True]])
        protos = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # mu_0, mu_1
        tau = 0.5
        loss = kim_nkc_loss(feats, anchors, negatives, protos, num_negatives=1, temperature=tau)
        pos, neg0, hard = 1.0, 0.0, 2 ** -0.5  # cos to mu_1, mu_0, most similar normal step
        logits = torch.tensor([pos, neg0, hard]) / tau
        expected = torch.logsumexp(logits, 0) - logits[0]
        assert float(loss) == pytest.approx(float(expected), rel=1e-5)

    def test_nkc_without_anchors_is_zero_and_connected(self) -> None:
        feats = torch.randn(1, 3, 4, requires_grad=True)
        none = torch.zeros(1, 3, dtype=torch.bool)
        loss = kim_nkc_loss(feats, none, none, torch.randn(2, 4), 16, 0.1)
        assert float(loss) == 0.0
        loss.backward()
        assert feats.grad is not None


class TestConfigAndModel:
    def test_default_model_is_unchanged(self) -> None:
        cfg = _host()
        model = KATVAD.from_config(cfg)
        assert model.kim is None
        assert not any(k.startswith("kim.") for k in model.state_dict())
        out = model(*_rows(), torch.randn(3, D))
        assert not any(k.startswith("kim_") for k in out)

    def test_kim_off_and_on_share_the_host_init(self) -> None:
        """KIM is built on a private RNG (D13): the host weights equal K0's draw for draw."""
        torch.manual_seed(0)
        off = KATVAD.from_config(_host()).state_dict()
        torch.manual_seed(0)
        on = KATVAD.from_config(_kim()).state_dict()
        for key, value in off.items():
            torch.testing.assert_close(on[key], value, msg=key)
        torch.manual_seed(0)
        KATVAD.from_config(_host())
        after_off = torch.rand(3)
        torch.manual_seed(0)
        KATVAD.from_config(_kim())
        torch.testing.assert_close(torch.rand(3), after_off)

    def test_zero_init_kim_scores_equal_the_host_at_step_zero(self) -> None:
        torch.manual_seed(0)
        host = KATVAD.from_config(_host()).eval()
        torch.manual_seed(0)
        kim = KATVAD.from_config(_kim()).eval()
        rows, lengths = _rows()
        text = torch.randn(3, D)
        torch.testing.assert_close(
            kim(rows, lengths, text)["cls_bin_logits"], host(rows, lengths, text)["cls_bin_logits"]
        )

    @pytest.mark.parametrize(("change", "match"), [
        ({"kip.enabled": True}, "KIP's slot"),
        ({"stp.sa2v2": True}, "stp.sa2"),
        ({"kim.nkc_temp": 0.0}, "nkc_temp"),
        ({"kim.anchor_thresh": 1.0}, "anchor_thresh"),
        ({"kim.enabled": False}, "needs kim.enabled"),
    ])
    def test_config_refuses(self, change: dict[str, object], match: str) -> None:
        cfg = _kim()
        for key, value in change.items():
            section, name = key.split(".")
            setattr(getattr(cfg, section), name, value)
        with pytest.raises(ValueError, match=match):
            validate_kim(cfg)

    def test_yaml_round_trip_and_old_configs(self, tmp_path: Path) -> None:
        from core.config import load_config

        cfg = _kim()
        cfg.save_yaml(tmp_path / "c.yaml")
        assert load_config(tmp_path / "c.yaml").kim == cfg.kim
        text = (tmp_path / "c.yaml").read_text()
        (tmp_path / "old.yaml").write_text(text.split("kim:")[0])
        assert load_config(tmp_path / "old.yaml").kim == dataclasses.replace(cfg.kim,
                                                                             enabled=False,
                                                                             prototypes=False)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    fixture = build_fixture(tmp_path / "fx")
    rng = np.random.default_rng(1)
    host = tmp_path / "host"
    host.mkdir()
    for path in fixture.clip_dir.glob("*.npy"):
        x = np.load(path)
        np.save(host / path.name, np.concatenate(
            [x, rng.standard_normal((len(x), MOTION))], 1).astype(np.float32))
    (host / constants.V2_INPUT_MANIFEST_FILENAME).write_text(
        json.dumps({"crn": constants.V2_OFF, "motion": constants.V5_ENCODER})
    )
    return fixture.data_dir, host


class TestTraining:
    def test_dvs_off_means_no_splice(self, tmp_path: Path) -> None:
        data_dir, host = _fixture(tmp_path)
        cfg = _host()
        dataset = DVSFeatureDataset(
            data_dir=data_dir, clip_dir=host, dvs=cfg.dvs, require_flow=False, seed=0
        )
        assert not any(bool(dataset[i]["is_synthesized"]) for i in range(len(dataset)))

    @pytest.mark.parametrize("prototypes", [False, True])
    def test_losses_and_diagnostics(self, tmp_path: Path, prototypes: bool) -> None:
        data_dir, host = _fixture(tmp_path)
        cfg = _kim(prototypes)
        cfg.kim.warmup_steps = 0
        cfg.kim.anchor_thresh = 0.0  # every top-k slot of an abnormal bag is an anchor
        set_global_seed(cfg.train.seed)
        dataset = DVSFeatureDataset(
            data_dir=data_dir, clip_dir=host, dvs=cfg.dvs, require_flow=False, seed=0
        )
        model = KATVAD.from_config(cfg)
        trainer = Trainer(
            cfg=cfg, model=model, dataset=dataset, class_names=load_class_names(data_dir),
            text_encode_fn=make_text_encoder(model, "stub", CPU, dim=D),
            output_dir=tmp_path / "out", device=CPU,
        )
        order = sorted(range(len(dataset)), key=lambda i: float(dataset[i]["label"]))
        batch = collate_variable_length([dataset[i] for i in order[:2] + order[-2:]])
        assert 0 < float(batch["label"].sum()) < 4  # both classes in the batch
        losses = trainer.compute_losses(batch)
        assert "kin" not in losses  # KIP v1's L_kin never runs here
        assert "dvs_sup" in losses  # D2: with no splice L_dvs keeps its normal-bag part
        assert torch.isfinite(losses["total"]) and "kim_mil" in losses
        wanted = set(constants.KIM_DIAG_KEYS)
        if not prototypes:
            wanted -= {"kim_proto_cos", "kim_proto_ready"}
        assert wanted <= set(losses)
        assert ("kim_nkc" in losses) is False  # prototypes are built by this first step
        if prototypes:
            assert model.kim is not None and model.kim.prototypes_ready
            second = trainer.compute_losses(batch)
            assert "kim_nkc" in second and torch.isfinite(second["kim_nkc"])
            second["total"].backward()
        else:
            losses["total"].backward()
        assert model.kim is not None
        assert model.kim.state_proj.weight.grad is not None
        grad = model.kim.w_out.weight.grad
        assert grad is not None and grad.abs().sum() > 0

    def test_warmup_keeps_nkc_and_prototypes_off(self, tmp_path: Path) -> None:
        data_dir, host = _fixture(tmp_path)
        cfg = _kim(True)
        dataset = DVSFeatureDataset(
            data_dir=data_dir, clip_dir=host, dvs=cfg.dvs, require_flow=False, seed=0
        )
        model = KATVAD.from_config(cfg)
        trainer = Trainer(
            cfg=cfg, model=model, dataset=dataset, class_names=load_class_names(data_dir),
            text_encode_fn=make_text_encoder(model, "stub", CPU, dim=D),
            output_dir=tmp_path / "out", device=CPU,
        )
        losses = trainer.compute_losses(collate_variable_length([dataset[i] for i in range(4)]))
        assert "kim_nkc" not in losses and float(losses["kim_proto_ready"]) == 0.0
        assert model.kim is not None and not bool(model.kim.proto_init.any())

    def test_train_then_score_through_the_real_wiring(self, tmp_path: Path) -> None:
        data_dir, host = _fixture(tmp_path)
        cfg = _kim(True)
        cfg.kim.warmup_steps = 1
        cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 2, "cpu"
        set_global_seed(cfg.train.seed)
        out = tmp_path / "run"
        trainer = build_trainer(cfg, out, data_dir, host, text_encoder="stub")
        cfg.save_yaml(out / "config.yaml")
        trainer.train()
        metrics = (out / "metrics.jsonl").read_text()
        assert "kim_mil" in metrics and "kim_w_out_norm" in metrics
        model = load_model_for_scoring(Config(), CPU, out / "checkpoint_last.pt", None, "stub")
        assert model.kim is not None and model.kim.use_prototypes and not model.training
        trained = trainer.model.kim
        assert trained is not None
        torch.testing.assert_close(model.kim.prototypes, trained.prototypes)
        rows, lengths = _rows()
        assert torch.isfinite(model(rows, lengths, torch.randn(2, D))["cls_bin_logits"]).all()
