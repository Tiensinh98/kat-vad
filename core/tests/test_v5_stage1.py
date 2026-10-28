"""v5 Stage 1: grid rows, the MQR readout, L_sMIL, the trainer / eval wiring and the gates
(``core/docs/v5/PREREG_V5.md`` §2)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from core import constants
from core.config import Config, load_config, validate_v5
from core.data.collate import collate_variable_length
from core.data.dataset import DVSFeatureDataset
from core.data.v5_grids import (
    GridGeometry,
    change_lag,
    eval_grid,
    grid_change,
    item_grid,
    read_grid_geometry,
    row_width,
    save_head,
    v5_rows,
)
from core.inference import adopt_checkpoint_architecture
from core.losses.smil import spatial_mil_loss
from core.models.kat_vad import KATVAD
from core.models.mqr import GridReadout, query_entropy
from core.models.text_encoding import make_text_encoder
from core.tests.fixtures import FixtureLayout, build_fixture
from core.tools import v5_maps, v5_stage1
from core.train import Trainer, load_class_names, set_global_seed, v5_grid_lag

CPU = torch.device("cpu")
TILES, DIM, WIDTH = constants.V5_GRID_TILES, constants.V5_GRID_DIM, constants.V5_GRID_WIDTH
BASE = constants.HIDDEN_DIM + DIM  # A2 row: [x ; c u~ / sigma_u]
OFFICIAL_STAUC = constants.REPO_ROOT / "data" / "DoTA" / "raw_annotations" / "stauc.py"


def _head(path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    return save_head(
        path,
        1 + 0.1 * rng.standard_normal(DIM),
        0.1 * rng.standard_normal(DIM),
        constants.VIDEOMAE_LN_EPS,
        0.4,
        0.1 * rng.standard_normal(DIM),
        0.5 + rng.random(DIM),
    )


def _grid_manifest(grid_dir: Path, stride: int, frame_step: int) -> None:
    grid_dir.mkdir(parents=True, exist_ok=True)
    (grid_dir / constants.V5_GRID_MANIFEST_FILENAME).write_text(
        json.dumps({"stride": stride, "frame_step": frame_step}), encoding="utf-8"
    )


class TestGridRows:
    def test_change_lag_matches_the_prereg(self) -> None:
        nexar = constants.VIDEOMAE_CLIP_FRAME_STEP
        assert change_lag(8, nexar) == 6  # 46 frames at s8 (S2-4)
        assert change_lag(3, nexar) == 16  # 48 frames at s3
        assert change_lag(3, constants.DOTA_VIDEOMAE_FRAME_STEP) == 6  # DoTA: check L's lag 6

    def test_change_rows_and_width(self) -> None:
        grid = np.random.default_rng(0).standard_normal((9, 7, 7, DIM)).astype(np.float16)
        flat = grid.reshape(9, -1).astype(np.float32)
        delta = grid_change(flat, 3)
        assert not delta[:3].any()
        np.testing.assert_array_equal(delta[3:], flat[3:] - flat[:-3])
        rows = v5_rows(np.ones((9, BASE)), grid, 3)
        assert (
            rows.shape == (9, row_width(BASE, constants.V5_KEYS_CHANGE))
            and rows.dtype == np.float32
        )
        assert v5_rows(np.ones((9, BASE)), grid, None).shape == (9, BASE + WIDTH)
        with pytest.raises(ValueError, match="C13"):
            v5_rows(np.ones((8, BASE)), grid, None)

    def test_item_grid_maps_native_frames_to_cache_rows(self) -> None:
        grid = np.arange(30)[:, None]
        np.testing.assert_array_equal(
            item_grid(grid, GridGeometry(1, 3), 4, 13, 3)[:, 0], [4, 7, 10]
        )
        np.testing.assert_array_equal(
            item_grid(grid, GridGeometry(3, 1), 0, 10, 3)[:, 0], [0, 1, 2, 3]
        )
        with pytest.raises(ValueError, match="not on the cache"):
            item_grid(grid, GridGeometry(3, 1), 1, 10, 3)

    def test_geometry_reads_stage0_manifest(self, tmp_path: Path) -> None:
        (tmp_path / constants.V5_GRID_MANIFEST_FILENAME).write_text(
            json.dumps({"rows": f"s1[::{constants.V5_STAGE0_STRIDE}] (protocol B)"})
        )
        assert read_grid_geometry(tmp_path) == GridGeometry(3, constants.DOTA_VIDEOMAE_FRAME_STEP)

    def test_eval_grid_refuses_a_mismatch(self, tmp_path: Path) -> None:
        _grid_manifest(tmp_path, 1, 3)
        assert eval_grid(constants.V5_READOUT_OFF, constants.V5_KEYS_CONTENT, None, 3) is None
        with pytest.raises(ValueError, match=r"v5\.readout=none"):
            eval_grid(constants.V5_READOUT_OFF, constants.V5_KEYS_CONTENT, tmp_path, 3)
        with pytest.raises(ValueError, match="pass its grid cache"):
            eval_grid(constants.V5_READOUT_MQR, constants.V5_KEYS_CONTENT, None, 3)
        got = eval_grid(constants.V5_READOUT_MQR, constants.V5_KEYS_CHANGE, tmp_path, 3)
        assert got is not None and got.lag == 16


class TestReadout:
    @pytest.mark.parametrize("queries", [1, constants.V5_QUERIES])
    @pytest.mark.parametrize("keys", constants.V5_KEYS)
    def test_init_is_the_tile_mean(self, queries: int, keys: str) -> None:  # S1-G2(a)
        grid = torch.randn(2, 4, TILES, DIM)
        change = torch.randn(grid.shape) if keys == constants.V5_KEYS_CHANGE else None
        r, attn, z = GridReadout(constants.V5_READOUT_MQR, queries, keys).read(grid, change)
        assert (r - grid.mean(-2)).abs().max() <= constants.V5_G2_UNIT_MAX_ABS
        assert attn is not None and attn.shape == (2, 4, queries, TILES)
        assert z is not None and z.shape == (2, 4, queries, DIM // queries)

    def test_motion_is_a2s_formula(self, tmp_path: Path) -> None:
        readout = GridReadout(constants.V5_READOUT_MEAN)
        readout.load_head(_head(tmp_path / "h.npz"))
        grid = torch.randn(1, 3, TILES, DIM)
        u = torch.nn.functional.layer_norm(
            grid.mean(-2), (DIM,), readout.fc_weight, readout.fc_bias, constants.VIDEOMAE_LN_EPS
        )
        want = readout.c * (u - readout.m_u) / readout.sigma_u
        torch.testing.assert_close(readout(grid).motion, want)
        assert bool(readout.head_loaded)

    def test_change_keys_need_the_change_block(self) -> None:
        readout = GridReadout(constants.V5_READOUT_MQR, 8, constants.V5_KEYS_CHANGE)
        with pytest.raises(ValueError, match="ΔG absent"):
            readout(torch.randn(1, 2, TILES, DIM))

    def test_queries_learn_away_from_the_mean(self) -> None:
        readout = GridReadout(constants.V5_READOUT_MQR, 8)
        grid = torch.randn(1, 3, TILES, DIM)
        (readout(grid).motion * torch.randn(1, 3, DIM)).sum().backward()  # sum(LN) is 0
        assert readout.query is not None and readout.query.grad is not None
        assert readout.query.grad.abs().sum() > 0

    def test_entropy_of_uniform_attention_is_log_tiles(self) -> None:
        attn = torch.full((2, 3, 8, TILES), 1.0 / TILES)
        ent = query_entropy(attn, torch.ones(2, 3))
        torch.testing.assert_close(ent, torch.full((8,), math.log(TILES)))


class TestConfig:
    def test_rows_validate(self) -> None:
        for row in constants.V5_ROWS:
            validate_v5(v5_stage1.row_config(row))

    @pytest.mark.parametrize(
        "override, match",
        [
            (["v5.readout=mqr"], "reads A2's inputs"),
            ([*v5_stage1.A2_OVERRIDES, "v5.readout=mqr", "v5.queries=7"], "must divide"),
            ([*v5_stage1.A2_OVERRIDES, "v5.readout=mean", "v5.queries=8"], "R1"),
            ([*v5_stage1.A2_OVERRIDES, "v5.readout=bogus"], "v5.readout"),
        ],
    )
    def test_bad_rows_raise(self, override: list[str], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_v5(load_config(None, override))

    def test_v2_checkpoints_build_without_a_readout(self) -> None:
        model = KATVAD.from_config(load_config(None, v5_stage1.A2_OVERRIDES))
        assert model.grid_readout is None and model.smil_head is None

    def test_eval_adopts_the_v5_section(self) -> None:
        cfg = load_config(None)
        stored = v5_stage1.row_config("R5d").to_dict()
        adopt_checkpoint_architecture(cfg, {"config": stored}, [])
        assert (cfg.v5.readout, cfg.v5.queries, cfg.v5.keys) == ("mqr", 8, "change")


class TestModel:
    def test_r1_at_init_feeds_the_baked_motion_term(self, tmp_path: Path) -> None:  # S1-G2(b) shape
        model = KATVAD.from_config(v5_stage1.row_config("R1")).eval()
        assert model.grid_readout is not None
        model.grid_readout.load_head(_head(tmp_path / "h.npz"))
        grid = torch.randn(1, 5, TILES, DIM)
        baked = model.grid_readout(grid).motion
        rows = torch.cat(
            [torch.randn(1, 5, constants.HIDDEN_DIM), baked, grid.reshape(1, 5, -1)], -1
        )
        out = model._grid_rows(rows, {})
        torch.testing.assert_close(
            out[..., constants.HIDDEN_DIM :], rows[..., constants.HIDDEN_DIM : BASE]
        )

    def test_forward_emits_maps_and_smil(self) -> None:
        model = KATVAD.from_config(v5_stage1.row_config("R5d"))
        rows = torch.randn(2, 6, BASE + 2 * WIDTH)
        out = model(rows, torch.tensor([6, 4]), class_feats=torch.randn(2, constants.HIDDEN_DIM))
        assert out["v5_attn"].shape == (2, 6, 8, TILES)
        assert out["v5_smil"].shape == (2, 6, 8)
        assert v5_maps.step_maps(out).shape == (2, 6, TILES)

    def test_wrong_width_raises(self) -> None:
        model = KATVAD.from_config(v5_stage1.row_config("R5"))
        with pytest.raises(ValueError, match="is the grid appended"):
            model(torch.randn(1, 3, BASE), torch.tensor([3]))


class TestSmilLoss:
    def test_hand_computed(self) -> None:
        probs = torch.tensor([[[0.1, 0.9], [0.2, 0.3], [0.5, 0.4]]])  # S_t = 0.9, 0.3, 0.5
        loss = spatial_mil_loss(probs, torch.tensor([1.0]), torch.tensor([3]), topk_pct=2)
        torch.testing.assert_close(loss, -torch.log(torch.tensor(0.9)))  # k = 1 -> top step 0.9

    def test_padding_is_ignored(self) -> None:
        probs = torch.tensor([[[0.3], [0.99]]])
        loss = spatial_mil_loss(probs, torch.tensor([0.0]), torch.tensor([1]))
        torch.testing.assert_close(loss, -torch.log(torch.tensor(0.7)))


def _v5_fixture(tmp_path: Path) -> tuple[FixtureLayout, Path, Path]:
    """The MSAD fixture with A2-wide rows and a V7 grid cache at the fixture stride."""
    fixture = build_fixture(tmp_path / "fx")
    rng = np.random.default_rng(1)
    a2_dir, grid_dir = tmp_path / "a2", tmp_path / "grid"
    a2_dir.mkdir()
    _grid_manifest(grid_dir, constants.FRAME_STRIDE, constants.VIDEOMAE_CLIP_FRAME_STEP)
    for path in fixture.clip_dir.glob("*.npy"):
        x = np.load(path)
        np.save(
            a2_dir / path.name,
            np.concatenate([x, rng.standard_normal((len(x), DIM))], 1).astype(np.float32),
        )
        np.save(grid_dir / path.name, rng.standard_normal((len(x), 7, 7, DIM)).astype(np.float16))
    (a2_dir / constants.V2_INPUT_MANIFEST_FILENAME).write_text(
        json.dumps({"crn": constants.V2_OFF, "motion": constants.V5_ENCODER})
    )
    _head(grid_dir / constants.V5_HEAD_FILENAME)
    return fixture, a2_dir, grid_dir


class TestTraining:
    @pytest.mark.parametrize("row", ["R1", "R5", "R5d"])
    def test_losses(self, tmp_path: Path, row: str) -> None:
        fixture, a2_dir, grid_dir = _v5_fixture(tmp_path)
        cfg = v5_stage1.row_config(row)
        cfg.train.batch_size, cfg.train.num_epochs = 4, 1
        set_global_seed(cfg.train.seed)
        lag = v5_grid_lag(cfg, grid_dir)
        assert lag == (6 if row == "R5d" else None)
        dataset = DVSFeatureDataset(
            data_dir=fixture.data_dir,
            clip_dir=a2_dir,
            dvs=cfg.dvs,
            require_flow=False,
            seed=cfg.train.seed,
            grid_dir=grid_dir,
            grid_lag=lag,
        )
        model = KATVAD.from_config(cfg)
        assert model.grid_readout is not None
        model.grid_readout.load_head(grid_dir / constants.V5_HEAD_FILENAME)
        trainer = Trainer(
            cfg=cfg,
            model=model,
            dataset=dataset,
            class_names=load_class_names(fixture.data_dir),
            text_encode_fn=make_text_encoder(model, "stub", CPU, dim=constants.HIDDEN_DIM),
            output_dir=tmp_path / "out",
            device=CPU,
        )
        losses = trainer.compute_losses(collate_variable_length([dataset[i] for i in range(4)]))
        assert torch.isfinite(losses["total"])
        assert ("smil" in losses) == (row != "R1")
        assert (f"v5_entropy_q{constants.V5_QUERIES - 1}" in losses) == (row != "R1")
        if "smil" in losses:
            losses["total"].backward()
            assert (
                model.grid_readout.query is not None and model.grid_readout.query.grad is not None
            )

    def test_dataset_change_is_per_item(self, tmp_path: Path) -> None:  # S2-4, no splice leak
        fixture, a2_dir, grid_dir = _v5_fixture(tmp_path)
        dataset = DVSFeatureDataset(
            data_dir=fixture.data_dir,
            clip_dir=a2_dir,
            require_flow=False,
            grid_dir=grid_dir,
            grid_lag=2,
        )
        vid = dataset.anomaly_ids[0]
        feats = dataset._load_features(vid)
        grid = np.load(grid_dir / f"{vid}.npy").reshape(len(feats), -1).astype(np.float32)
        np.testing.assert_allclose(feats[:, BASE + WIDTH :].numpy(), grid_change(grid, 2))

    def test_grid_dir_without_readout_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="would be ignored"):
            v5_grid_lag(Config(), tmp_path)

    def test_stride_mismatch_raises(self, tmp_path: Path) -> None:
        _grid_manifest(tmp_path, 3, 3)
        with pytest.raises(ValueError, match="C2"):
            v5_grid_lag(v5_stage1.row_config("R5"), tmp_path)


class TestStauc:
    def test_all_mass_in_the_box(self) -> None:
        dmap = np.zeros((10, 10))
        dmap[:5, :5] = 1.0
        assert v5_maps.tarr_frame(dmap, 1, [[0.0, 0.0, 640.0, 360.0]]) == pytest.approx(
            1.0, abs=1e-5
        )
        assert v5_maps.tarr_frame(dmap, 0, [[0.0, 0.0, 640.0, 360.0]]) == 0.0
        assert v5_maps.tarr_frame(dmap, 1, []) == 1.0

    def test_full_tarr_is_the_plain_auc(self) -> None:
        labels = np.array([0, 1, 0, 1, 1, 0])
        scores = np.array([0.1, 0.8, 0.4, 0.35, 0.9, 0.2])
        got = v5_maps.stauc(labels, scores, np.where(labels == 1, 1.0, 0.0))
        assert got["stauc"] == pytest.approx(got["auc"]) and got["auc"] == pytest.approx(8 / 9)

    @pytest.mark.skipif(not OFFICIAL_STAUC.is_file(), reason="DoTA's stauc.py not on disk")
    def test_port_matches_official(self) -> None:  # S1-G5(c)
        pytest.importorskip("sklearn")
        result = v5_maps.g5c(OFFICIAL_STAUC)
        assert result["pass"], result


class TestPointRead:
    @staticmethod
    def _clips(
        n: int = 60,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, dict[int, list[list[float]]]]]:
        out = {}
        for i in range(n):
            labels = np.zeros(60, dtype=np.int64)
            labels[30:] = 1
            boxes = {f: [[100.0 + 10 * (i % 7), 100.0, 400.0, 300.0]] for f in range(30, 60)}
            out[f"vid{i % 12}_{i:06d}"] = (np.arange(0, 60, 3), labels, boxes)
        return out

    def test_random_fails_planted_passes(self) -> None:  # S1-G5(b) shape
        rng = np.random.default_rng(0)
        reads: dict[str, dict[str, dict[str, float]]] = {"random": {}, "planted": {}}
        for v, (steps, labels, boxes) in self._clips().items():
            noise = rng.random((len(steps), TILES))
            planted = noise.copy()
            for k, f in enumerate(steps):
                if f in boxes:
                    planted[k] += v5_maps.s0.tile_weights(boxes[int(f)], 7) > 0
            for name, mp in (("random", noise), ("planted", planted)):
                read = v5_maps.clip_point_reads([mp], steps, labels, boxes)
                assert read is not None
                reads[name][v] = read
        ego = dict.fromkeys(reads["random"], False)
        bins = dict.fromkeys(reads["random"])
        assert not v5_maps.point_summary(reads["random"], ego, bins)["points"]
        assert v5_maps.point_summary(reads["planted"], ego, bins)["points"]

    def test_centre_prior_peaks_in_the_middle(self) -> None:
        prior = v5_maps.centre_prior()
        assert int(np.argmax(prior)) == TILES // 2


class TestGates:
    def test_g2_synthetic_passes(self) -> None:
        got = v5_stage1.g2_synthetic()
        assert got["readout_pass"] and got["splice_keeps_item_change"], got

    def test_provenance_flags_a_non_v5_diff(self, tmp_path: Path) -> None:
        a2, run = tmp_path / "a2", tmp_path / "R5"
        for d, cfg in (
            (a2, load_config(None, v5_stage1.A2_OVERRIDES)),
            (run, v5_stage1.row_config("R5")),
        ):
            d.mkdir()
            cfg.save_yaml(d / "config.yaml")
            torch.save({"global_step": 7}, d / "checkpoint_last.pt")
            (d / "metrics.jsonl").write_text(json.dumps({"global_step": 7}) + "\n")
        assert v5_stage1.provenance(run, "R5", a2, 7)["pass"]
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        cfg["loss"]["mil_topk_pct"] = 99
        (run / "config.yaml").write_text(yaml.safe_dump(cfg))
        got = v5_stage1.provenance(run, "R5", a2, 7)
        assert got["config_diff_vs_a2"] == ["loss.mil_topk_pct"] and not got["pass"]

    def test_collapse_flags_a_uniform_run(self, tmp_path: Path) -> None:
        ent = {f"v5_entropy_q{m}": math.log(TILES) for m in range(8)}
        (tmp_path / "metrics.jsonl").write_text(
            json.dumps({"global_step": 5, "mil": 0.3, "smil": 0.6, **ent})
        )
        got = v5_stage1.collapse(tmp_path)
        assert got["pass"] and got["did_not_leave_mean"]


class TestExtract:
    def test_decoded_ahead_keeps_order_and_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(path: Path) -> torch.Tensor:
            return torch.full((2,), float(path.stem))

        monkeypatch.setattr(v5_stage1, "decode_resized", fake)
        ids = ["3", "1", "2"]
        got = list(v5_stage1.decoded_ahead({v: Path(f"{v}.mp4") for v in ids}, ids))
        assert [v for v, _ in got] == ids
        assert [float(x[0]) for _, x in got] == [3.0, 1.0, 2.0]
        assert list(v5_stage1.decoded_ahead({}, [])) == []

    def test_exact_fp32_turns_tf32_off(self) -> None:
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cudnn.allow_tf32 = True
            v5_stage1.exact_fp32()
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before
