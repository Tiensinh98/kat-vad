"""SA²-v2: cells, priors, scorer, losses, trainer, eval (``core/docs/final/SA2V2_RUN.md``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from core import constants
from core.config import Config, validate_stp
from core.data.collate import collate_variable_length
from core.data.dataset import DVSFeatureDataset
from core.data.stp_patches import (
    EvalCells,
    cell_columns,
    cell_manifest,
    check_cell_cache,
    motion_magnitude,
    patch_change,
    pool_token_grid,
    stp_arm,
    stp_eval_inputs,
    stp_rows,
)
from core.inference import load_model_for_scoring, sliding_window_scores
from core.losses.mil import patch_mil_loss
from core.models.kat_vad import KATVAD
from core.models.sa2v2 import SA2V2, cell_priors, zscore_cells
from core.models.text_encoding import make_text_encoder
from core.tests.fixtures import build_fixture
from core.tools import sa2v2_cells, sa2v2_eval, sa2v2_readout
from core.tools.protocol_b_eval import write_clip_scores
from core.train import Trainer, build_trainer, load_class_names, set_global_seed, stp_patch_dir

CPU = torch.device("cpu")
CELLS, D = constants.SV2_CELLS, constants.SV2_DIM
MOTION = 384
BASE = constants.HIDDEN_DIM + MOTION
LEVEL = (1.0, 0.5, 1.0, 0.5)


def _cells(rows: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, CELLS * D)).astype(np.float32)


def _module() -> SA2V2:
    module = SA2V2()
    module.set_level(*LEVEL)
    return module


def _inputs(rows: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    cls = torch.randn(2, rows, D)
    columns = torch.from_numpy(cell_columns(_cells(2 * rows))).reshape(2, rows, -1)
    return cls, columns, torch.randn(3, D)


def _cfg() -> Config:
    cfg = Config()
    cfg.stp.sa2v2 = True
    cfg.model.motion_dim = MOTION
    cfg.v2.motion = constants.V5_ENCODER
    cfg.kip.enabled = False
    return cfg


class TestCells:
    def test_pool_is_the_2x2_mean_row_major(self) -> None:
        grid = np.zeros((1, 14, 14, D), dtype=np.float16)
        grid[0, 2:4, 4:6] = 2.0  # one 2 x 2 block -> one cell
        grid[0, 2, 4] = 6.0
        cells = pool_token_grid(grid)
        assert cells.shape == (1, CELLS, D)
        assert cells[0, 1 * 7 + 2, 0] == pytest.approx(3.0)
        assert cells[0, 0, 0] == 0.0

    def test_window_change_equals_the_whole_source(self) -> None:
        stored = _cells(9)
        whole = cell_columns(stored)
        for start, stop in ((0, 3), (3, 7), (6, 9)):
            np.testing.assert_allclose(cell_columns(stored, start, stop), whole[start:stop])

    def test_no_seam_motion_across_a_splice(self) -> None:
        """Two sources concatenated after D: D at the seam is each source's own, not the jump."""
        a, b = _cells(4, 1), _cells(4, 2)
        spliced = np.concatenate([cell_columns(a), cell_columns(b)])
        np.testing.assert_allclose(spliced[3], cell_columns(a)[3])
        np.testing.assert_allclose(spliced[4], cell_columns(b)[0])
        jump = cell_columns(np.concatenate([a, b]))
        assert not np.allclose(spliced[3], jump[3])

    def test_refactor_keeps_motion_magnitude(self) -> None:
        p = _cells(5).reshape(5, CELLS, D)
        np.testing.assert_allclose(motion_magnitude(p), np.linalg.norm(patch_change(p), axis=-1))

    def test_cache_check_refuses(self, tmp_path: Path) -> None:
        sa2v2_cells.extract_video_features.check_or_write_manifest(
            tmp_path, cell_manifest("x", 8, "test"), False
        )
        assert check_cell_cache(tmp_path, 8)["grid"] == constants.SV2_GRID
        with pytest.raises(ValueError, match="this run needs"):
            check_cell_cache(tmp_path, 3)


class TestPriors:
    def test_rho_marks_the_outlier_cell(self) -> None:
        base = torch.nn.functional.normalize(torch.ones(1, CELLS, D), dim=-1)
        cells = base + 0.001 * torch.randn(1, CELLS, D)
        cells[0, 7] = -cells[0, 7]
        mo, rho = cell_priors(cells, torch.zeros(1, CELLS, D))
        assert int(rho.argmax()) == 7
        assert mo.abs().max() == 0

    def test_zscore_of_a_constant_is_zero(self) -> None:
        assert zscore_cells(torch.ones(2, CELLS)).abs().max() == 0


class TestScorer:
    def test_parameters_match_the_proposal(self) -> None:
        module = SA2V2()
        head = sum(p.numel() for p in module.branch_head.parameters())
        total = sum(p.numel() for p in module.parameters()) - head
        assert head == 513
        assert 0.85e6 < total < 0.95e6

    def test_start_is_the_motion_prior(self) -> None:
        cls, columns, text = _inputs()
        module = _module()
        out = module(cls, columns, text)
        change = columns[..., constants.SV2_CELL_WIDTH :].reshape(2, 6, CELLS, D)
        expected = zscore_cells(change.norm(dim=-1)).softmax(dim=-1)
        torch.testing.assert_close(out.attn, expected, atol=1e-5, rtol=1e-4)
        assert out.cell_logits.mean() < 0  # b = -2
        p = columns[..., : constants.SV2_CELL_WIDTH].reshape(2, 6, CELLS, D)
        pooled = (expected[..., None] * p).sum(-2)
        torch.testing.assert_close(out.x_as, pooled, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(out.features, cls + out.x_as)

    def test_mixed_precision_keeps_the_scorer_in_float32(self) -> None:
        cls, columns, text = _inputs()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = _module()(cls, columns, text)
        assert out.cell_logits.dtype == torch.float32 and out.attn.dtype == torch.float32
        assert torch.isfinite(out.features).all()

    def test_padding_stays_the_host_row(self) -> None:
        cls, columns, text = _inputs()
        mask = torch.ones(2, 6)
        mask[1, 4:] = 0
        out = _module()(cls, columns, text, mask)
        torch.testing.assert_close(out.features[1, 4:], cls[1, 4:])
        assert out.cell_logits[1, 4:].abs().max() == 0

    def test_every_term_learns(self) -> None:
        cls, columns, text = _inputs()
        module = _module()
        out = module(cls, columns, text)
        (out.x_as.sum() + out.cell_logits.sum()).backward()
        for name in ("beta", "gamma", "bias", "log_temperature"):
            assert getattr(module, name).grad is not None, name
        assert module.w.weight.grad.abs().sum() > 0
        assert module.gamma.grad.abs() > 0
        # w = 0 holds the encoder's gradient at the first step; one step on w releases it
        torch.optim.SGD(module.parameters(), lr=1e-3).step()
        module.zero_grad()
        out = module(cls, columns, text)
        (out.x_as.sum() + out.cell_logits.sum()).backward()
        assert module.w_cell.weight.grad.abs().sum() > 0

    def test_refuses_unset_level_constants(self) -> None:
        cls, columns, text = _inputs()
        with pytest.raises(ValueError, match="level constants"):
            SA2V2()(cls, columns, text)

    def test_text_delta_uses_the_projection(self) -> None:
        module = _module()
        text = torch.randn(3, D)
        before = module.text_delta(text).clone()
        module.set_text_projection(2.0 * torch.linalg.qr(torch.randn(D, D)).Q)
        assert not torch.allclose(before, module.text_delta(text))
        with pytest.raises(ValueError, match="needs Normal"):
            module.text_delta(text[:1])


class TestPatchLoss:
    def test_selection_is_the_same_in_both_bag_kinds(self) -> None:
        cells = torch.zeros(2, 5, CELLS)
        cells[:, 2, :3] = 4.0  # the chosen step's top cells
        frames = torch.zeros(2, 5)
        frames[:, 2] = 9.0
        labels, lengths = torch.tensor([1.0, 0.0]), torch.tensor([5, 5])
        loss, means = patch_mil_loss(cells, frames, labels, lengths, topk_pct=5)
        q = float(torch.sigmoid(torch.tensor(4.0)))
        torch.testing.assert_close(means, torch.tensor([q, q]))
        expected = (-np.log(q) - np.log(1 - q)) / 2
        assert float(loss) == pytest.approx(expected, rel=1e-5)

    def test_no_gradient_through_the_host_choice(self) -> None:
        cells = torch.zeros(1, 5, CELLS, requires_grad=True)
        frames = torch.randn(1, 5, requires_grad=True)
        loss, _ = patch_mil_loss(cells, frames, torch.ones(1), torch.tensor([5]), topk_pct=5)
        loss.backward()
        assert frames.grad is None and cells.grad is not None


class TestModel:
    def test_old_arms_get_nothing_new(self) -> None:
        cfg = _cfg()
        cfg.stp.sa2v2 = False
        model = KATVAD.from_config(cfg)
        assert model.sa2v2 is None and model.motion_dropout == 0.0
        assert not [k for k in model.state_dict() if "sa2v2" in k]
        mask = torch.ones(2, 3)
        model.train()
        assert model._motion_mask(mask) is mask

    def test_config_refuses(self) -> None:
        cfg = _cfg()
        validate_stp(cfg)
        cfg.stp.sa2 = True
        with pytest.raises(ValueError, match="pick one"):
            validate_stp(cfg)
        cfg = _cfg()
        cfg.model.motion_dim = 0
        with pytest.raises(ValueError, match="motion stream"):
            validate_stp(cfg)

    def test_motion_dropout_zeroes_the_whole_term_in_training_only(self) -> None:
        cfg = _cfg()
        cfg.stp.motion_dropout = 0.999
        model = KATVAD.from_config(cfg)
        assert model.motion_residual is not None
        torch.nn.init.ones_(model.motion_residual.proj.bias)
        rows = torch.randn(2, 3, BASE)
        mask = torch.ones(2, 3)
        model.train()
        torch.manual_seed(0)
        out, _ = model.motion_residual(rows, model._motion_mask(mask))
        torch.testing.assert_close(out, rows[..., : constants.HIDDEN_DIM])
        model.eval()
        out, _ = model.motion_residual(rows, model._motion_mask(mask))
        assert not torch.allclose(out, rows[..., : constants.HIDDEN_DIM])

    def test_forward_refuses_missing_text_and_wrong_width(self) -> None:
        model = KATVAD.from_config(_cfg())
        model.sa2v2.set_level(*LEVEL)
        rows = torch.randn(1, 3, BASE + constants.SV2_ROW_WIDTH)
        with pytest.raises(ValueError, match="class_feats"):
            model(rows, torch.tensor([3]))
        with pytest.raises(ValueError, match="cell cache"):
            model(rows[..., 1:], torch.tensor([3]), class_feats=torch.randn(2, D))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    fixture = build_fixture(tmp_path / "fx")
    rng = np.random.default_rng(1)
    host, cells = tmp_path / "host", tmp_path / "cells"
    host.mkdir()
    sa2v2_cells.extract_video_features.check_or_write_manifest(
        cells, cell_manifest("x", constants.FRAME_STRIDE, "test"), False
    )
    for path in fixture.clip_dir.glob("*.npy"):
        x = np.load(path)
        np.save(host / path.name, np.concatenate(
            [x, rng.standard_normal((len(x), MOTION))], 1).astype(np.float32))
        np.save(cells / path.name, rng.standard_normal((len(x), CELLS * D)).astype(np.float16))
    (host / constants.V2_INPUT_MANIFEST_FILENAME).write_text(
        json.dumps({"crn": constants.V2_OFF, "motion": constants.V5_ENCODER})
    )
    return fixture.data_dir, host, cells


class TestTraining:
    def test_losses_diagnostics_and_checkpoint(self, tmp_path: Path) -> None:
        data_dir, host, cells = _fixture(tmp_path)
        cfg = _cfg()
        cfg.train.batch_size, cfg.train.num_epochs = 4, 1
        set_global_seed(cfg.train.seed)
        dataset = DVSFeatureDataset(
            data_dir=data_dir, clip_dir=host, dvs=cfg.dvs, require_flow=False,
            seed=cfg.train.seed, cell_dir=stp_patch_dir(cfg, cells),
        )
        assert dataset[0]["v_feat"].shape[-1] == BASE + constants.SV2_ROW_WIDTH
        model = KATVAD.from_config(cfg)
        model.sa2v2.set_level(*LEVEL)
        trainer = Trainer(
            cfg=cfg, model=model, dataset=dataset, class_names=load_class_names(data_dir),
            text_encode_fn=make_text_encoder(model, "stub", CPU, dim=constants.HIDDEN_DIM),
            output_dir=tmp_path / "out", device=CPU,
        )
        losses = trainer.compute_losses(collate_variable_length([dataset[i] for i in range(4)]))
        assert torch.isfinite(losses["total"])
        wanted = {k for k in constants.SV2_DIAG_KEYS if not k.startswith("sv2_top3q")}
        assert wanted <= set(losses)
        host_total = sum(
            v for k, v in losses.items() if k == "sv2_patch"
        ) * cfg.stp.patch_weight + losses["sv2_branch"] * cfg.stp.branch_weight
        assert float(losses["total"]) > float(host_total)
        losses["total"].backward()
        assert model.sa2v2.branch_head.weight.grad is not None
        assert model.sa2v2.w.weight.grad.abs().sum() > 0
        rebuilt = KATVAD.from_config(cfg)
        rebuilt.load_state_dict(model.state_dict())
        torch.testing.assert_close(rebuilt.sa2v2.level, model.sa2v2.level)

    def test_train_then_score_through_the_real_wiring(self, tmp_path: Path) -> None:
        """build_trainer -> checkpoint -> load_model_for_scoring -> the frame scorer."""
        data_dir, host, cells = _fixture(tmp_path)
        level = dict(zip(sa2v2_cells.LEVEL_KEYS, LEVEL, strict=True))
        (cells / constants.SV2_LEVEL_FILENAME).write_text(json.dumps(level))
        cfg = _cfg()
        cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 1, "cpu"
        set_global_seed(cfg.train.seed)
        out = tmp_path / "run"
        trainer = build_trainer(cfg, out, data_dir, host, text_encoder="stub", patch_dir=cells)
        assert trainer.model.sa2v2 is not None
        torch.testing.assert_close(trainer.model.sa2v2.level, torch.tensor(LEVEL))
        cfg.save_yaml(out / "config.yaml")
        trainer.train()
        metrics = (out / "metrics.jsonl").read_text()
        assert "sv2_gamma" in metrics and "sv2_entropy" in metrics

        model = load_model_for_scoring(Config(), CPU, out / "checkpoint_last.pt", None, "stub")
        assert model.sa2v2 is not None and not model.training
        torch.testing.assert_close(model.sa2v2.level, torch.tensor(LEVEL))
        video = next(p.stem for p in cells.glob("*.npy"))
        rows = EvalCells(cells).append(np.load(host / f"{video}.npy"), video)
        text = torch.randn(2, D)
        curve, _ = sliding_window_scores(model, torch.from_numpy(rows), lambda: text)
        assert curve.shape == (len(rows),) and torch.isfinite(curve).all()

    def test_patch_dir_rules(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="needs --patch-dir"):
            stp_patch_dir(_cfg(), None)
        cfg = _cfg()
        with pytest.raises(FileNotFoundError):
            stp_patch_dir(cfg, tmp_path)

    def test_eval_inputs_and_arm(self, tmp_path: Path) -> None:
        sa2v2_cells.extract_video_features.check_or_write_manifest(
            tmp_path, cell_manifest("x", 8, "test"), False
        )
        stp = _cfg().stp
        inputs = stp_eval_inputs(stp, tmp_path, 8)
        assert isinstance(inputs, EvalCells)
        np.save(tmp_path / "v.npy", _cells(3).astype(np.float16))
        assert inputs.append(np.zeros((3, BASE)), "v").shape == (3, BASE + constants.SV2_ROW_WIDTH)
        with pytest.raises(ValueError, match="cell cache"):
            stp_eval_inputs(stp, None, 8)
        assert stp_arm(stp)["stp_sa2v2"] is True
        assert stp_arm(Config().stp) == {}

    def test_rows_must_align(self) -> None:
        with pytest.raises(ValueError, match="C13"):
            stp_rows(np.zeros((3, BASE)), cell_columns(_cells(4)))


def _tiny_clip() -> CLIPVisionModelWithProjection:
    torch.manual_seed(0)
    return CLIPVisionModelWithProjection(CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        image_size=224, patch_size=16, projection_dim=D,
    )).eval()


class TestTools:
    def test_clip_cells_shapes_and_g1(self) -> None:
        frames = torch.rand(2, 3, 60, 100)
        embeds, cells, gap = sa2v2_cells.clip_cells(frames, _tiny_clip())
        assert embeds.shape == (2, D) and cells.shape == (2, CELLS, D)
        assert gap <= constants.SV2_G1_MAX_ABS

    def test_gate_video(self) -> None:
        rows = np.random.default_rng(0).standard_normal((4, D)).astype(np.float32)
        assert sa2v2_cells.gate_video(rows, rows, 0.0)["reason"] == sa2v2_cells.OK
        assert sa2v2_cells.gate_video(rows, rows, 1.0)["reason"] == sa2v2_cells.G1_FAILED
        assert sa2v2_cells.gate_video(rows, -rows, 0.0)["reason"] != sa2v2_cells.OK
        assert sa2v2_cells.gate_video(rows, rows[:3], 0.0)["reason"] != sa2v2_cells.OK

    def test_pool_checks_rows(self, tmp_path: Path) -> None:
        grid, host, out = tmp_path / "grid", tmp_path / "host", tmp_path / "out"
        grid.mkdir()
        host.mkdir()
        np.save(grid / "a.npy", np.zeros((3, 14, 14, D), dtype=np.float16))
        np.save(host / "a.npy", np.zeros((3, BASE), dtype=np.float32))
        ids = tmp_path / "ids.txt"
        ids.write_text("a\n")
        args = argparse.Namespace(ids_file=ids, out_dir=out, dataset="x", stride=8, grid_dir=grid,
                                  input_dir=host, force=False)
        assert sa2v2_cells.run_pool(args)["done"] == 1
        assert np.load(out / "a.npy").shape == (3, CELLS * D)
        np.save(host / "a.npy", np.zeros((4, BASE), dtype=np.float32))
        with pytest.raises(ValueError, match="failed"):
            sa2v2_cells.run_pool(args)

    def test_level_stats_match_a_direct_computation(self, tmp_path: Path) -> None:
        stored = _cells(5)
        np.save(tmp_path / "a.npy", stored.astype(np.float16))
        stats = sa2v2_cells.level_stats(tmp_path, ["a"])
        columns = torch.from_numpy(cell_columns(np.load(tmp_path / "a.npy")))
        flat = columns.reshape(-1, 2, CELLS, D)
        mo, rho = cell_priors(flat[:, 0], flat[:, 1])
        assert stats["mo_mean"] == pytest.approx(float(mo.mean()), rel=1e-5)
        assert stats["rho_std"] == pytest.approx(float(rho.std(unbiased=False)), rel=1e-3)
        assert stats["cells"] == 5 * CELLS

    def test_shuffle_keeps_counts_and_pairs_label_with_class(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        ids = [f"v{i:02d}" for i in range(40)]
        labels = {v: int(i < 10) for i, v in enumerate(ids)}
        meta = {v: {"class_name": "CarAccident" if labels[v] else "Normal"} for v in ids}
        (data / constants.LABELS_TRAIN_FILENAME).write_text(json.dumps(labels))
        (data / constants.META_FILENAME).write_text(json.dumps(meta))
        out = tmp_path / "out"
        report = sa2v2_cells.run_shuffle(argparse.Namespace(data_dir=data, out_dir=out, seed=1))
        got = json.loads((out / constants.LABELS_TRAIN_FILENAME).read_text())
        new_meta = json.loads((out / constants.META_FILENAME).read_text())
        assert sum(got.values()) == 10 and report["bags"] == 40
        assert got != labels
        assert all((new_meta[v]["class_name"] == "CarAccident") == bool(got[v]) for v in ids)

    def test_clip_reads_and_baselines(self) -> None:
        cfg = _cfg()
        model = KATVAD.from_config(cfg).eval()
        model.sa2v2.set_level(*LEVEL)
        rows = torch.from_numpy(stp_rows(np.random.default_rng(0).standard_normal((7, BASE)),
                                         cell_columns(_cells(7))))
        reads = sa2v2_eval.clip_reads(model, rows, lambda: torch.randn(2, D), max_len=4)
        assert reads[sa2v2_eval.MAP_KEY].shape == (7, CELLS)
        assert reads[sa2v2_eval.MO_KEY].shape == (7, CELLS)
        assert reads[sa2v2_eval.TOP_KEY].shape == (7,)
        assert (reads[sa2v2_eval.ENTROPY_KEY] <= 1.0 + 1e-5).all()
        maps = sa2v2_eval.baseline_maps(reads)
        shapes = {k: v.shape for k, v in maps.items()}
        assert shapes == dict.fromkeys(sa2v2_eval.BASELINE_MAPS, (7, CELLS))

    def test_readout_end_to_end(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        labels = {f"v{i}": np.array([0, 0, 1, 1, 0]) for i in range(30)}
        curves = {v: rng.random(5) for v in labels}
        for arm, shift in (("W0", 0.0), ("SV2", 1.0)):
            d = tmp_path / arm
            d.mkdir()
            scores = {v: c + shift * labels[v] for v, c in curves.items()}
            aucs = dict.fromkeys(labels, 0.5 + 0.4 * shift)
            (d / sa2v2_readout.dada_whole_eval.CLIP_AUCS_JSON).write_text(json.dumps(aucs))
            (d / sa2v2_readout.dada_whole_eval.READOUT_JSON).write_text(json.dumps({
                "per_run_macro": {"s2024": 0.7}, "macro": {"mean": 0.7, "low": 0.6, "high": 0.8},
                "micro": 0.7, "video_auc": 0.5, "monotone_ruler_macro": 0.5,
                "middle_ruler_macro": 0.7,
            }))
            write_clip_scores(d / sa2v2_readout.dada_whole_eval.CLIP_SCORES_NPZ, scores, labels)
        dump = tmp_path / "dump.npz"
        scores, _ = sa2v2_readout.read_clip_scores(tmp_path / "SV2" / "clip_scores.npz")
        arrays = {}
        for v, y in labels.items():
            arrays |= {f"b_{v}": y + 0.1 * rng.random(5), f"a_{v}": rng.random(5),
                       f"t_{v}": 0.3 + 0.2 * y, f"s_{v}": scores[v]}
        np.savez(dump, **arrays)
        runs = {}
        for name, gamma in (("real", 0.5), ("shuf", 0.01)):
            run = tmp_path / name
            run.mkdir()
            lines = [{"global_step": i, "sv2_gamma": gamma * i / 20} for i in range(21)]
            (run / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
            runs[name] = run
        args = argparse.Namespace(
            w0_dada=tmp_path / "W0", sv2_dada=tmp_path / "SV2", dump=dump, run_dir=runs["real"],
            shuffled_run_dir=runs["shuf"], r4_dada=None, w0_dota=None, sv2_dota=None, maps_md=None,
            out_dir=tmp_path / "rep",
        )
        r = sa2v2_readout.run(args)
        assert r["dada"]["sentence"] == "gain shown"
        assert r["dada"]["dump_vs_frame_read_max_gap"] < 1e-6
        assert r["dada"]["c1"]["sentence"] == "gain shown"
        assert r["dada"]["c3"]["abnormal"] > r["dada"]["c3"]["normal"]
        assert r["c2_c4"]["sv2_gamma"]["exceeds"] is True
        md = (tmp_path / "rep" / "sa2v2_readout.md").read_text()
        assert "PRELIMINARY" in md and "runs after the DADA read" in md
