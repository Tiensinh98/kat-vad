"""Unit tests for the Phase-5 building blocks (no dataset, CPU-only).

Covers: the in-house cosine-with-warmup schedule, deterministic epoch
permutations, class-index mapping, the stub text encoder, sliding-window
scoring vs a single forward, A5 expansion, metric parity with torchmetrics
(R11), and the stage/ablation loss gating on a wired Trainer.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torchmetrics

from core import constants
from core.config import Config
from core.data.definitions import DatasetSpecVerbalizer, dataset_abbr, verbalize_class_name
from core.inference import expand_to_frames, sliding_window_scores
from core.metrics import frame_ap, frame_auc
from core.models.kat_vad import KATVAD
from core.models.text_encoding import make_text_encoder, stub_text_features
from core.tests.fixtures import build_fixture
from core.train import (
    Trainer,
    class_index_tensor,
    cosine_warmup_lambda,
    epoch_permutation,
    load_class_names,
    set_global_seed,
    warm_start_model,
)

CPU = torch.device("cpu")


class TestSchedule:
    def test_warmup_then_cosine_to_zero(self) -> None:
        fn = cosine_warmup_lambda(warmup_steps=10, total_steps=110)
        assert fn(0) == pytest.approx(0.1)
        assert fn(9) == pytest.approx(1.0)
        assert fn(10) == pytest.approx(1.0)  # cos(0)
        assert fn(60) == pytest.approx(0.5)  # halfway through cosine
        assert fn(110) == pytest.approx(0.0)
        assert fn(500) == pytest.approx(0.0)  # clamped past the end

    def test_no_warmup(self) -> None:
        fn = cosine_warmup_lambda(warmup_steps=0, total_steps=100)
        assert fn(0) == pytest.approx(1.0)
        assert fn(50) == pytest.approx(0.5, abs=1e-3)

    def test_monotone_warmup(self) -> None:
        fn = cosine_warmup_lambda(warmup_steps=20, total_steps=200)
        values = [fn(s) for s in range(20)]
        assert values == sorted(values)


class TestEpochPermutation:
    def test_deterministic_and_complete(self) -> None:
        a = epoch_permutation(2024, 3, 17)
        b = epoch_permutation(2024, 3, 17)
        assert a == b
        assert sorted(a) == list(range(17))

    def test_differs_across_epochs(self) -> None:
        assert epoch_permutation(2024, 0, 32) != epoch_permutation(2024, 1, 32)


class TestClassIndex:
    def test_maps_names(self) -> None:
        names = ["Normal", "Traffic_accident"]
        idx = class_index_tensor(
            ["Normal", "Traffic_accident", "Normal"], names, CPU
        )
        assert idx.tolist() == [0, 1, 0]

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not in the dataset class list"):
            class_index_tensor(["Ghost"], ["Normal"], CPU)


class TestTextStub:
    def test_deterministic_per_string(self) -> None:
        a = stub_text_features(["car crash", "normal road"])
        b = stub_text_features(["car crash", "normal road"])
        torch.testing.assert_close(a, b)
        assert a.shape == (2, 512)
        assert not torch.allclose(a[0], a[1])

    def test_make_text_encoder_stub_dim(self) -> None:
        cfg = Config()
        model = KATVAD.from_config(cfg)
        encode = make_text_encoder(model, "stub", CPU, dim=cfg.model.hidden_dim)
        assert encode(["x"]).shape == (1, cfg.model.hidden_dim)

    def test_clip_mode_without_weights_raises(self) -> None:
        model = KATVAD.from_config(Config())
        encode = make_text_encoder(model, "clip", CPU)
        with pytest.raises(RuntimeError, match="requires clip_text_model"):
            encode(["x"])


class TestVerbalize:
    def test_known_and_fallback(self) -> None:
        verbalizer = DatasetSpecVerbalizer(dataset_abbr("MSAD"))
        text = verbalize_class_name(verbalizer, "Traffic_accident")
        assert isinstance(text, str) and text
        assert verbalize_class_name(verbalizer, "Abnormal") == "Abnormal"

    def test_dataset_abbr_unknown(self) -> None:
        with pytest.raises(KeyError, match="Unknown dataset"):
            dataset_abbr("NotADataset")

    def test_msad_full_shares_msad_definitions(self) -> None:
        assert dataset_abbr("MSAD-full") == dataset_abbr("MSAD") == "msad"


class TestMetricsParity:
    def test_auc_ap_match_torchmetrics(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.random(500).astype(np.float32)
        labels = (rng.random(500) > 0.7).astype(np.int64)
        auroc = torchmetrics.AUROC(task="binary")
        ap = torchmetrics.AveragePrecision(task="binary")
        auroc.update(torch.from_numpy(scores), torch.from_numpy(labels))
        ap.update(torch.from_numpy(scores), torch.from_numpy(labels))
        # torchmetrics stubs type compute() as None; it returns a Tensor
        ref_auc = float(auroc.compute())  # type: ignore[arg-type,func-returns-value]
        ref_ap = float(ap.compute())  # type: ignore[arg-type,func-returns-value]
        assert frame_auc(scores, labels) == pytest.approx(ref_auc, abs=1e-6)
        assert frame_ap(scores, labels) == pytest.approx(ref_ap, abs=1e-6)

    def test_metrics_invariant_to_uniform_expansion(self) -> None:
        rng = np.random.default_rng(1)
        scores = rng.random(80).astype(np.float32)
        labels = (rng.random(80) > 0.5).astype(np.int64)
        expanded_s = expand_to_frames(scores, 8)
        expanded_l = expand_to_frames(labels, 8)
        assert expanded_s.shape == (640,)
        assert frame_auc(scores, labels) == pytest.approx(
            frame_auc(expanded_s, expanded_l), abs=1e-9
        )


class TestSlidingWindow:
    def test_matches_single_forward_when_short(self) -> None:
        cfg = Config()
        torch.manual_seed(0)
        model = KATVAD.from_config(cfg).eval()
        feats = torch.randn(40, cfg.model.hidden_dim)
        class_feats = torch.randn(3, cfg.model.hidden_dim)
        score, sim = sliding_window_scores(model, feats, lambda: class_feats, 512)
        with torch.no_grad():
            ref = model(feats[None], torch.tensor([40]), class_feats=class_feats)
        torch.testing.assert_close(score, ref["cls_bin_logits"][0].sigmoid())
        torch.testing.assert_close(sim, ref["cls_sim_mat"][0].sigmoid())
        assert score.shape == (40,)
        assert sim.shape == (40, 3)

    def test_windows_concatenate_to_full_length(self) -> None:
        cfg = Config()
        torch.manual_seed(0)
        model = KATVAD.from_config(cfg).eval()
        feats = torch.randn(37, cfg.model.hidden_dim)
        class_feats = torch.randn(2, cfg.model.hidden_dim)
        score, sim = sliding_window_scores(model, feats, lambda: class_feats, 16)
        assert score.shape == (37,)
        assert sim.shape == (37, 2)
        assert torch.isfinite(score).all()


def _make_trainer(tmp_path: Path, **cfg_updates: object) -> Trainer:
    from core.data.dataset import DVSFeatureDataset

    fixture = build_fixture(tmp_path / "fx")
    cfg = Config()
    cfg.train.batch_size = 4
    cfg.train.num_epochs = 1
    for dotted, value in cfg_updates.items():
        section, key = dotted.split("__")
        setattr(getattr(cfg, section), key, value)
    set_global_seed(cfg.train.seed)
    dataset = DVSFeatureDataset(
        data_dir=fixture.data_dir,
        clip_dir=fixture.clip_dir,
        flow_dir=fixture.flow_dir if cfg.kip.enabled else None,
        dvs=cfg.dvs,
        require_flow=cfg.kip.enabled,
        seed=cfg.train.seed,
    )
    model = KATVAD.from_config(cfg)
    encode = make_text_encoder(model, "stub", CPU, dim=cfg.model.hidden_dim)
    return Trainer(
        cfg=cfg,
        model=model,
        dataset=dataset,
        class_names=load_class_names(fixture.data_dir),
        text_encode_fn=encode,
        output_dir=tmp_path / "out",
        device=CPU,
    )


class TestLossGating:
    def _batch(self, trainer: Trainer) -> dict[str, object]:
        from core.data.collate import collate_variable_length

        samples = [trainer.dataset[i] for i in range(4)]
        return collate_variable_length(samples)

    def test_stage2_kip_on_has_all_terms(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, loss__captions_from_definitions=True)
        losses = trainer.compute_losses(self._batch(trainer))
        for key in ("mil", "mul_mil", "kip_rec", "kip_align", "kin", "total"):
            assert key in losses, key
            assert torch.isfinite(losses[key]), key
        assert math.isfinite(float(losses["total"]))

    def test_stage2_kip_off_has_no_kip_terms(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, kip__enabled=False)
        losses = trainer.compute_losses(self._batch(trainer))
        assert "kip_rec" not in losses
        assert "kin" not in losses
        assert torch.isfinite(losses["total"])

    def test_pmg_only_drops_kin(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, kip__pmg_only=True)
        losses = trainer.compute_losses(self._batch(trainer))
        assert "kip_rec" in losses
        assert "kin" not in losses

    def test_no_captions_by_default(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        losses = trainer.compute_losses(self._batch(trainer))
        assert "cap_contrastive" not in losses

    # --- Phase 1 arms (DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md §6) --------
    def test_phase1_arms_are_off_by_default(self, tmp_path: Path) -> None:
        """A new term must not appear until someone asks for it."""
        trainer = _make_trainer(tmp_path)
        assert trainer.cfg.loss.dvs_anchor_mode == constants.DVS_ANCHOR_MODE_SPAN
        assert trainer.cfg.loss.bottomk_weight == 0.0
        assert "bottomk" not in trainer.compute_losses(self._batch(trainer))

    def test_bottomk_arm_adds_a_finite_term(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, loss__bottomk_weight=0.5)
        losses = trainer.compute_losses(self._batch(trainer))
        assert "bottomk" in losses and torch.isfinite(losses["bottomk"])
        assert torch.isfinite(losses["total"])

    def test_dvs_ignore_arm_changes_dvs_sup(self, tmp_path: Path) -> None:
        """The arm must actually move the loss it targets, not silently no-op.

        theta = 0 forces every abnormal anchor to be spliced, so ``y^p`` is
        guaranteed to carry positives -- exactly the rows lesson **C29** is
        about. Same weights, same batch: any difference is the arm itself.
        """
        values: dict[str, float] = {}
        batch = None
        for mode in constants.DVS_ANCHOR_MODE_CHOICES:
            trainer = _make_trainer(
                tmp_path / mode,
                loss__dvs_anchor_mode=mode,
                dvs__theta=0.0,
                dvs__theta_ego=0.0,
            )
            if batch is None:
                batch = self._batch(trainer)
                pseudo = batch["pseudo_frame_label"]
                assert isinstance(pseudo, torch.Tensor)
                assert float(pseudo.sum()) > 0, (
                    "fixture drew no synthesized abnormal row; test is vacuous"
                )
            losses = trainer.compute_losses(batch)
            assert torch.isfinite(losses["dvs_sup"])
            values[mode] = float(losses["dvs_sup"])
        assert values[constants.DVS_ANCHOR_MODE_SPAN] != pytest.approx(
            values[constants.DVS_ANCHOR_MODE_IGNORE]
        ), "loss.dvs_anchor_mode=ignore did not reach supervised_loss"

    def test_invalid_dvs_anchor_mode_raises(self, tmp_path: Path) -> None:
        """A typo must fail loudly, not leave the arm quietly off."""
        trainer = _make_trainer(tmp_path, loss__dvs_anchor_mode="middle")
        with pytest.raises(ValueError, match="dvs_anchor_mode"):
            trainer.compute_losses(self._batch(trainer))

    def test_stage1_trains_kip_only(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, train__stage=1)
        losses = trainer.compute_losses(self._batch(trainer))
        assert set(losses) == {"kip_rec", "kip_align", "total"}
        losses["total"].backward()
        for name, param in trainer.model.named_parameters():
            if name.startswith("kip."):
                assert param.requires_grad, name
            else:
                assert not param.requires_grad, name

    def test_stage2_gradients_reach_model(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        losses = trainer.compute_losses(self._batch(trainer))
        losses["total"].backward()
        grads = [
            p.grad for p in trainer.model.parameters() if p.requires_grad and p.grad is not None
        ]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)


class TestWarmStart:
    def test_weights_transfer(self, tmp_path: Path) -> None:
        cfg = Config()
        donor = KATVAD.from_config(cfg)
        target = KATVAD.from_config(cfg)
        ckpt = tmp_path / "stage1.pt"
        torch.save({"model": donor.state_dict(), "epoch": 3, "global_step": 42}, ckpt)
        warm_start_model(target, ckpt)
        for key, tensor in donor.state_dict().items():
            assert torch.equal(target.state_dict()[key], tensor), key

    def test_missing_non_text_key_fails_loud(self, tmp_path: Path) -> None:
        cfg = Config()
        state = KATVAD.from_config(cfg).state_dict()
        dropped = next(k for k in state if k.startswith("kip."))
        del state[dropped]
        ckpt = tmp_path / "bad.pt"
        torch.save({"model": state}, ckpt)
        with pytest.raises(ValueError, match="init-weights"):
            warm_start_model(KATVAD.from_config(cfg), ckpt)

    def test_unexpected_key_fails_loud(self, tmp_path: Path) -> None:
        cfg = Config()
        state = KATVAD.from_config(cfg).state_dict()
        state["not_a_real.key"] = torch.zeros(1)
        ckpt = tmp_path / "extra.pt"
        torch.save({"model": state}, ckpt)
        with pytest.raises(ValueError, match="unexpected"):
            warm_start_model(KATVAD.from_config(cfg), ckpt)
