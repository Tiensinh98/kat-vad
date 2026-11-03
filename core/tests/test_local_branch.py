"""LaGoVAD-OSTP Step 1 (core/docs/final/LOCAL_STEP1.md): the patch-only local branch L1 and its
read-out against a free ensemble. CPU-only, data-free (the MSAD fixture for the trainer)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants
from core.config import Config, load_config, validate_stp
from core.data.collate import collate_variable_length
from core.data.dataset import DVSFeatureDataset
from core.data.stp_patches import stp_arm
from core.inference import adopt_checkpoint_architecture, sliding_window_scores
from core.models.kat_vad import KATVAD
from core.models.sa2 import SpatialAttentionAggregation
from core.models.text_encoding import make_text_encoder
from core.tests.test_stp import BASE, MOTION, _stp_fixture
from core.tools import local_readout
from core.tools.protocol_b_eval import CLIP_SCORES_NPZ, write_clip_scores
from core.train import Trainer, load_class_names, set_global_seed, stp_patch_dir

torch.set_num_threads(1)
CPU = torch.device("cpu")
K, D = constants.STP_TOP_K, constants.STP_PATCH_DIM
HOST_SETS = [
    "kip.enabled=false",
    f"model.motion_dim={MOTION}",
    f"v2.motion={constants.V5_ENCODER}",
]


def _cfg(arm: str = constants.LOCAL_ARM, *extra: str) -> Config:
    sets = [f"{k}={str(v).lower()}" for k, v in constants.LOCAL_ARMS[arm].items()]
    return load_config(None, [*HOST_SETS, *sets, *extra])


def _rows(steps: int = 6, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, steps, constants.HIDDEN_DIM, generator=g),
        torch.randn(1, steps, MOTION, generator=g),
        torch.randn(1, steps, K * D, generator=g),
    )


class TestConfig:
    def test_arms_validate(self) -> None:
        for arm in constants.LOCAL_ARMS:
            validate_stp(_cfg(arm))
        assert _cfg().stp.branch == constants.STP_BRANCH_LOCAL
        assert Config().stp.branch == constants.STP_BRANCH_HOST

    @pytest.mark.parametrize(
        ("override", "match"),
        [
            (["stp.branch=global"], "stp.branch"),
            (["stp.branch=local"], "stp.sa2"),
            (["stp.sa2=true", "stp.branch=local", "stp.text_prompt=prefix"], "prefix"),
            (["stp.sa2=true", "stp.branch=local", "loss.captions_from_definitions=true"],
             "captions"),
        ],
    )
    def test_bad_configs_raise(self, override: list[str], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_stp(load_config(None, ["kip.enabled=false", *override]))

    def test_provenance_keys(self) -> None:
        assert "stp_branch" not in stp_arm(load_config(None, ["stp.sa2=true"]).stp)
        assert stp_arm(_cfg().stp)["stp_branch"] == constants.STP_BRANCH_LOCAL

    def test_eval_adopts_the_branch(self) -> None:
        cfg = Config()
        adopt_checkpoint_architecture(cfg, {"config": _cfg().to_dict()}, [])
        assert cfg.stp.branch == constants.STP_BRANCH_LOCAL


class TestModel:
    def test_local_has_no_global_or_text_module(self) -> None:
        model = KATVAD.from_config(_cfg())
        assert model.local_branch and model.sa2 is not None
        assert model.motion_residual is None and model.fusion is None and model.sim_head is None
        assert model.bin_head.bin_head_type == "vanilla"
        names = {n.split(".")[0] for n, _ in model.named_parameters()}
        assert names == {"sa2", "temporal_encoder", "bin_head"}

    def test_needs_sa2(self) -> None:
        with pytest.raises(ValueError, match="SA²"):
            KATVAD(local_branch=True)

    def test_output_is_patch_mean_through_own_encoder(self) -> None:
        model = KATVAD.from_config(_cfg()).eval()
        c, u, p = _rows()
        lengths = torch.tensor([6])
        out = model(torch.cat([c, u, p], -1), lengths, class_feats=torch.randn(3, D))
        assert out["cls_bin_logits"].shape == (1, 6)
        assert "cls_sim_mat" not in out and "motion_share" not in out
        ref = model.temporal_encoder(p.view(1, 6, K, D).mean(-2), lengths)  # W_q = 0: average
        torch.testing.assert_close(out["vis_feats"], ref)

    def test_reads_neither_u_nor_c_content(self) -> None:
        model = KATVAD.from_config(_cfg()).eval()
        c, u, p = _rows()
        c2, u2, _ = _rows(seed=1)
        lengths, text = torch.tensor([6]), torch.randn(3, D)
        a = model(torch.cat([c, u, p], -1), lengths, class_feats=text)["cls_bin_logits"]
        b = model(torch.cat([c2, u2, p], -1), lengths, class_feats=torch.randn(5, D))
        torch.testing.assert_close(a, b["cls_bin_logits"])  # W_q = 0: c is a (flat) query only

    def test_no_logits_without_class_feats(self) -> None:
        model = KATVAD.from_config(_cfg()).eval()
        out = model(torch.cat(_rows(), -1), torch.tensor([6]))
        assert "cls_bin_logits" not in out

    def test_missing_patch_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="--patch-dir"):
            KATVAD.from_config(_cfg())(torch.randn(1, 4, BASE), torch.tensor([4]))

    def test_host_sa2_path_is_unchanged(self) -> None:
        sa2 = SpatialAttentionAggregation()
        c, _, p = _rows()
        full = sa2(c, p)
        alone = sa2(c, p, add_cls=False)
        torch.testing.assert_close(full.features, c + alone.features)

    def test_scorer_returns_empty_sim(self) -> None:
        model = KATVAD.from_config(_cfg()).eval()
        feats = torch.cat(_rows(steps=20), -1)[0]
        score, sim = sliding_window_scores(model, feats, lambda: torch.randn(3, D), 8)
        assert score.shape == (20,) and sim.shape == (20, 0)


class TestTraining:
    def test_losses(self, tmp_path: Path) -> None:
        data_dir, a2_dir, patch_dir = _stp_fixture(tmp_path)
        cfg = _cfg()
        cfg.train.batch_size, cfg.train.num_epochs = 4, 1
        set_global_seed(cfg.train.seed)
        dataset = DVSFeatureDataset(
            data_dir=data_dir, clip_dir=a2_dir, dvs=cfg.dvs, require_flow=False,
            seed=cfg.train.seed, patch_dir=stp_patch_dir(cfg, patch_dir),
        )
        model = KATVAD.from_config(cfg)
        trainer = Trainer(
            cfg=cfg, model=model, dataset=dataset, class_names=load_class_names(data_dir),
            text_encode_fn=make_text_encoder(model, "stub", CPU, dim=constants.HIDDEN_DIM),
            output_dir=tmp_path / "out", device=CPU,
        )
        losses = trainer.compute_losses(collate_variable_length([dataset[i] for i in range(4)]))
        assert "mul_mil" not in losses and "prompt_norm" not in losses
        assert "motion_share" not in losses
        assert torch.isfinite(losses["total"])
        assert float(losses["stp_entropy"]) == pytest.approx(1.0, abs=1e-5)
        losses["total"].backward()
        assert model.sa2 is not None and model.sa2.w_q.weight.grad is not None
        assert model.sa2.w_k.weight.grad is not None


def _write_row(root: Path, name: str, scores: dict[str, np.ndarray],
               labels: dict[str, np.ndarray]) -> Path:
    out = root / name
    out.mkdir(parents=True)
    write_clip_scores(out / CLIP_SCORES_NPZ, scores, labels)
    return out


def _metadata(path: Path, ids: list[str]) -> tuple[Path, Path]:
    meta = {v: {"anomaly_start": 2, "anomaly_end": 5, "num_frames": 10,
                "anomaly_class": ("ego: " if i % 2 == 0 else "other: ") + "turning"}
            for i, v in enumerate(ids)}
    (path / "meta.json").write_text(json.dumps(meta))
    (path / "split.txt").write_text("".join(f"{v}\n" for v in ids))
    return path / "meta.json", path / "split.txt"


class TestReadout:
    def test_minmax_and_fuse(self) -> None:
        np.testing.assert_allclose(local_readout.minmax(np.array([2.0, 4.0, 3.0])), [0, 1, 0.5])
        assert not local_readout.minmax(np.ones(4)).any()
        fused = local_readout.fuse({"a": np.array([0.0, 1.0])}, {"a": np.array([5.0, 3.0])})
        np.testing.assert_allclose(fused["a"], [0.5, 0.5])
        with pytest.raises(ValueError, match="native frames"):
            local_readout.fuse({"a": np.zeros(2)}, {"a": np.zeros(3)})

    def test_oracle_and_spearman(self) -> None:
        gain = local_readout.oracle_gain({"a": 0.6, "b": 0.9}, {"a": 0.8, "b": 0.5},
                                         {"a": 0.7, "b": 0.95})
        assert gain == pytest.approx({"a": 0.1, "b": -0.05})
        assert local_readout.spearman(np.array([1, 2, 3.0]), np.array([3, 5, 9.0])) == 1.0
        assert local_readout.spearman(np.ones(3), np.arange(3.0)) is None
        np.testing.assert_allclose(local_readout.average_rank(np.array([1.0, 1.0, 2.0])),
                                   [0.5, 0.5, 2.0])

    @pytest.mark.parametrize(
        ("gates", "mil", "want"),
        [
            ((True, True, True), None, local_readout.VERDICT_PASS),
            ((True, False, True), None, local_readout.VERDICT_REDUNDANT),
            ((False, True, True), 0.05, local_readout.VERDICT_OVERFIT),
            ((False, False, False), 0.30, local_readout.VERDICT_WEAK),
            ((False, False, False), None, local_readout.VERDICT_WEAK),
        ],
    )
    def test_verdict(self, gates: tuple[bool, bool, bool], mil: float | None,
                     want: str) -> None:
        assert local_readout.verdict(*gates, mil) == want

    def test_end_to_end(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        ids = [f"vid{i:03d}_{j:03d}" for i in range(20) for j in range(2)]
        labels = {v: (np.arange(30) >= 15).astype(np.int64) for v in ids}
        noisy = {v: labels[v] + rng.normal(0, 1.5, 30) for v in ids}
        good = {v: labels[v] + rng.normal(0, 0.3, 30) for v in ids}  # L1 sees the event
        l0 = _write_row(tmp_path, "L0", noisy, labels)
        l0b = _write_row(tmp_path, "L0b", {v: s + rng.normal(0, 0.2, 30) for v, s in noisy.items()},
                         labels)
        l1 = _write_row(tmp_path, "L1", good, labels)
        meta, split = _metadata(tmp_path, ids)
        run_dir = tmp_path / "run_l1"
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").write_text(
            json.dumps({"global_step": 1, "mil": 0.7}) + "\n"
            + json.dumps({"global_step": 2, "mil": 0.3}) + "\n"
        )
        args = argparse.Namespace(
            l0=l0, l0b=l0b, l1=l1, l0_run=None, l0b_run=None, l1_run=run_dir,
            position=None, metadata=meta, split_file=split, out_dir=tmp_path / "rep",
        )
        result = local_readout.run(args)
        assert result["gates"]["G1"]["pass"] and result["gates"]["G2"]["pass"]
        assert result["gates"]["G3"]["pass"]
        assert result["verdict"] == local_readout.VERDICT_PASS
        md = (tmp_path / "rep" / "local_readout.md").read_text()
        assert "PRELIMINARY (n = 1 seed)" in md and "STEP 1 PASS" in md

    def test_labels_must_match(self, tmp_path: Path) -> None:
        lab = {"v_000": np.array([0, 1, 1])}
        s = {"v_000": np.array([0.1, 0.5, 0.9])}
        a = _write_row(tmp_path, "a", s, lab)
        b = _write_row(tmp_path, "b", s, {"v_000": np.array([1, 1, 0])})
        meta, split = _metadata(tmp_path, ["v_000"])
        args = argparse.Namespace(
            l0=a, l0b=b, l1=a, l0_run=None, l0b_run=None, l1_run=None, position=None,
            metadata=meta, split_file=split, out_dir=tmp_path / "rep",
        )
        with pytest.raises(ValueError, match="labels differ"):
            local_readout.run(args)
