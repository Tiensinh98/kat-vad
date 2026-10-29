"""v5.1 tile-scored MQR: centred 4 x 7 x 7 grids, the tile scorer and its readout, R3c, L_tile,
the trainer / map wiring (``core/docs/v5/PREREG_V5.md`` §3, proposal 2026-10-29 §3.2)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants
from core.data.collate import collate_variable_length
from core.data.dataset import DVSFeatureDataset
from core.data.v5_grids import (
    GridGeometry,
    centred_reference,
    flat_grid,
    item_grid,
    read_grid_geometry,
    save_head,
    tile_boundary_rows,
)
from core.losses.smil import boundary_steps, tile_mil_loss
from core.models.kat_vad import KATVAD
from core.models.mqr import (
    ChangeTopKReadout,
    GridReadout,
    TileReadout,
    change_magnitude,
    tile_features,
    tile_residuals,
)
from core.models.text_encoding import make_text_encoder
from core.tests.fixtures import FixtureLayout, build_fixture
from core.tools import v5_maps, v5_stage1
from core.tools.extract_video_features import causal_clip_indices, centred_clip_indices
from core.tools.v5_stage0 import tile_grid
from core.train import (
    Trainer,
    load_class_names,
    optimizer_param_groups,
    set_global_seed,
    v5_grid_lag,
)

CPU = torch.device("cpu")
DIM, CELLS = constants.V5_GRID_DIM, constants.V5_TILE_CELLS
WIDTH = constants.V5_TILE_GRID_WIDTH
BASE = constants.HIDDEN_DIM + DIM  # A2 row: [x ; c u~ / sigma_u]
STEP = constants.VIDEOMAE_CLIP_FRAME_STEP


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


def _tile_manifest(grid_dir: Path, stride: int, frame_step: int = STEP) -> None:
    grid_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stride": stride,
        "frame_step": frame_step,
        "slots": constants.V5_SLOTS_TILE,
        "window_kind": constants.V5_WINDOW_CENTRED,
    }
    (grid_dir / constants.V5_GRID_MANIFEST_FILENAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class TestWindow:
    def test_centred_is_the_shifted_causal_clip_inside_the_video(self) -> None:
        lag = (constants.VIDEOMAE_CLIP_FRAMES - 1 - constants.V5_CENTRE_LEAD) * STEP
        t = 60
        assert centred_clip_indices(t, 500, step=STEP) == causal_clip_indices(t + lag, step=STEP)
        assert centred_clip_indices(t, 500, step=STEP)[constants.V5_CENTRE_LEAD] == t

    def test_edges_repeat(self) -> None:
        clip = centred_clip_indices(2, 10, step=STEP)
        assert clip[0] == 0 and clip[-1] == 9 and len(clip) == constants.VIDEOMAE_CLIP_FRAMES
        assert clip == sorted(clip)

    def test_reference_rows(self) -> None:
        keep, index = centred_reference(np.array([0, 24, 30, 90]), STEP, 100)
        np.testing.assert_array_equal(keep, [False, True, True, False])  # 0 - 24 < 0; 90 + 21 > 99
        np.testing.assert_array_equal(index, [45, 51])

    def test_boundary_rows(self) -> None:
        assert tile_boundary_rows(GridGeometry(8, 3), 8) == 3  # Nexar: 24 frames at s8
        assert tile_boundary_rows(GridGeometry(3, 1), 3) == 3  # DoTA: 8 frames at s3


class TestGrid:
    def test_tile_grid_keeps_slot_pairs_and_the_token_mean(self) -> None:
        tokens = torch.randn(2, 8 * 14 * 14, 16)
        grid = tile_grid(tokens)
        assert grid.shape == (2, 4, 7, 7, 16)
        torch.testing.assert_close(grid.flatten(1, 3).mean(1), tokens.mean(1))
        slots = tokens.reshape(2, 8, 14, 14, 16)
        torch.testing.assert_close(grid[:, 1, 0, 0], slots[:, 2:4, :2, :2].mean((1, 2, 3)))

    def test_flat_width_accepts_both_grids(self) -> None:
        assert flat_grid(np.zeros((3, 4, 7, 7, DIM))).shape == (3, WIDTH)
        assert flat_grid(np.zeros((3, 7, 7, DIM))).shape == (3, constants.V5_GRID_WIDTH)
        with pytest.raises(ValueError, match="expected one of"):
            flat_grid(np.zeros((3, 2, 7, 7, DIM)))

    def test_centred_cache_snaps_crops_to_phase_zero(self) -> None:
        grid = np.arange(30)[:, None]
        centred = GridGeometry(3, STEP, constants.V5_SLOTS_TILE, constants.V5_WINDOW_CENTRED)
        np.testing.assert_array_equal(item_grid(grid, centred, 4, 13, 3)[:, 0], [1, 2, 3])
        with pytest.raises(ValueError, match="not on the cache"):  # Stages 0-2 caches unchanged
            item_grid(grid, GridGeometry(3, 1), 4, 13, 3)

    def test_geometry_reads_slots_and_window(self, tmp_path: Path) -> None:
        _tile_manifest(tmp_path, 8)
        geometry = read_grid_geometry(tmp_path)
        assert (geometry.slots, geometry.window) == (4, constants.V5_WINDOW_CENTRED)


class TestFeatures:
    def test_frame_wide_component_is_removed(self) -> None:
        grid = torch.randn(2, 3, CELLS, 8)
        shift = torch.randn(2, 3, 4, 1, 8).expand(-1, -1, -1, 49, -1).flatten(2, 3)
        r0, d0 = tile_residuals(grid)
        r1, d1 = tile_residuals(grid + shift)
        torch.testing.assert_close(r0, r1)
        torch.testing.assert_close(d0, d1, rtol=1e-5, atol=1e-5)

    def test_change_repeats_the_last_slot(self) -> None:
        grid = torch.randn(1, 1, CELLS, 8)
        _, d = tile_residuals(grid)
        d = d.unflatten(-2, (4, 49))
        torch.testing.assert_close(d[..., 3, :, :], d[..., 2, :, :])

    def test_features_are_two_layer_normed_halves(self) -> None:
        feats = tile_features(torch.randn(1, 2, CELLS, 8))
        assert feats.shape == (1, 2, CELLS, 16)
        torch.testing.assert_close(
            feats[..., :8].mean(-1), torch.zeros(1, 2, CELLS), atol=1e-5, rtol=0
        )


class TestReadout:
    def test_tile_readout_starts_as_r1c(self, tmp_path: Path) -> None:
        tile, mean = TileReadout(), GridReadout(constants.V5_READOUT_MEAN, tiles=CELLS)
        tile.load_head(_head(tmp_path / "h.npz"))
        mean.load_head(tmp_path / "h.npz")
        grid = torch.randn(2, 5, CELLS, DIM)
        out = tile(grid)
        torch.testing.assert_close(out.motion, mean(grid).motion, atol=1e-5, rtol=1e-5)
        assert out.tile_logits is not None and float(out.tile_logits.abs().max()) == 0.0
        assert out.attn is not None and out.attn.shape == (2, 5, 1, CELLS)
        torch.testing.assert_close(out.attn, torch.full_like(out.attn, 1 / CELLS))

    def test_scorer_gets_a_gradient_at_init(self) -> None:
        tile = TileReadout()
        logits = tile(torch.randn(1, 4, CELLS, DIM)).tile_logits
        assert logits is not None
        logits.sum().backward()
        assert tile.scorer.out.weight.grad is not None
        assert float(tile.scorer.out.weight.grad.abs().sum()) > 0

    def test_temperature_is_clamped(self) -> None:
        tile = TileReadout()
        with torch.no_grad():
            tile.log_temp.fill_(-10.0)
        assert float(tile.temperature()) == pytest.approx(constants.V5_TILE_TEMP_MIN)

    def test_change_topk_picks_the_moving_cell(self) -> None:
        grid = torch.zeros(1, 1, CELLS, DIM)
        grid[0, 0, 49 + 10] = 5.0  # slot 1, tile 10 departs from its frame
        assert int(change_magnitude(grid)[0, 0].argmax()) in (10, 49 + 10)
        readout = ChangeTopKReadout(k=1)
        out = readout(grid)
        assert out.attn is not None
        assert int(out.attn[0, 0, 0].argmax()) in (10, 49 + 10)


class TestTileLoss:
    def _call(self, logits: torch.Tensor, label: float, dense: bool, **kw: object) -> torch.Tensor:
        n = logits.shape[1]
        out = tile_mil_loss(
            logits,
            torch.tensor([label]),
            torch.tensor([n]),
            kw.pop("pseudo", torch.zeros(1, n)),  # type: ignore[arg-type]
            kw.pop("synth", torch.tensor([False])),  # type: ignore[arg-type]
            dense_negatives=dense,
            topk_cells=2,
            beta=0.0,
            **kw,  # type: ignore[arg-type]
        )
        return out["tile"]

    def test_dense_negatives_push_down_every_cell(self) -> None:
        logits = torch.zeros(1, 4, 6, requires_grad=True)
        self._call(logits, 0.0, dense=True).backward()
        assert logits.grad is not None and bool((logits.grad > 0).all())

    def test_bag_negatives_touch_only_the_selected_cells(self) -> None:
        logits = torch.randn(1, 4, 6).requires_grad_()
        self._call(logits, 0.0, dense=False).backward()
        assert logits.grad is not None
        assert int((logits.grad != 0).sum()) == 2  # k = 1 step x K = 2 cells

    def test_abnormal_hand_computed(self) -> None:
        logits = torch.full((1, 2, 3), -10.0)
        logits[0, 1, :2] = math.log(0.8 / 0.2)  # step 1 holds two cells at p = 0.8
        loss = self._call(logits, 1.0, dense=True)
        torch.testing.assert_close(loss, -torch.log(torch.tensor(0.8)), atol=1e-5, rtol=1e-5)

    def test_synthetic_negatives_skip_the_boundary(self) -> None:
        pseudo = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        near = boundary_steps(pseudo[0], 1)
        assert near.tolist() == [False, False, False, True, True, True, True, False, False, False]
        logits = torch.zeros(1, 10, 3)
        logits[0, 4:6] = 2.0  # the MIL term picks the inserted anomaly
        logits.requires_grad_()
        self._call(
            logits, 1.0, dense=True, pseudo=pseudo, synth=torch.tensor([True]), boundary_rows=1
        ).backward()
        assert logits.grad is not None
        assert bool((logits.grad[0, [0, 1, 2, 7, 8, 9]] > 0).all())
        assert float(logits.grad[0, 3].abs().sum()) == 0.0  # boundary, not a negative

    def test_padding_is_ignored(self) -> None:
        logits = torch.zeros(1, 5, 3, requires_grad=True)
        out = tile_mil_loss(
            logits, torch.tensor([0.0]), torch.tensor([2]), torch.zeros(1, 5),
            torch.tensor([False]), dense_negatives=True,
        )
        out["tile"].backward()
        assert logits.grad is not None and float(logits.grad[0, 2:].abs().sum()) == 0.0


class TestConfig:
    @pytest.mark.parametrize("row", list(constants.V5_TILE_ROWS))
    def test_rows_validate(self, row: str) -> None:
        cfg = v5_stage1.row_config(row)
        assert (cfg.v5.grid_slots, cfg.v5.window) == (4, constants.V5_WINDOW_CENTRED)
        KATVAD.from_config(cfg)

    @pytest.mark.parametrize(
        ("override", "match"),
        [
            (["v5.readout=tile", "v5.smil_weight=0.5"], "grid_slots=4"),
            (["v5.readout=tile", "v5.grid_slots=4"], "smil_weight > 0"),
            (["v5.readout=mqr", "v5.grid_slots=4"], "grid_slots=1"),
            (["v5.window=sideways"], "v5.window"),
        ],
    )
    def test_bad_rows_raise(self, override: list[str], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            KATVAD.from_config(v5_stage1.row_config("R1", override))

    def test_temperature_has_no_weight_decay(self) -> None:
        cfg = v5_stage1.row_config("R6c")
        model = KATVAD.from_config(cfg)
        groups = optimizer_param_groups(model, cfg)
        assert len(groups) == 2 and groups[1]["weight_decay"] == 0.0
        assert isinstance(model.grid_readout, TileReadout)
        assert groups[1]["params"][0] is model.grid_readout.log_temp

    def test_r1c_keeps_the_single_group(self) -> None:
        cfg = v5_stage1.row_config("R1c")
        assert len(optimizer_param_groups(KATVAD.from_config(cfg), cfg)) == 1


def _tile_fixture(tmp_path: Path) -> tuple[FixtureLayout, Path, Path]:
    """The MSAD fixture with A2-wide rows and a centred 4 x 7 x 7 grid cache."""
    fixture = build_fixture(tmp_path / "fx")
    rng = np.random.default_rng(1)
    a2_dir, grid_dir = tmp_path / "a2", tmp_path / "grid"
    a2_dir.mkdir()
    _tile_manifest(grid_dir, constants.FRAME_STRIDE)
    for path in fixture.clip_dir.glob("*.npy"):
        x = np.load(path)
        np.save(
            a2_dir / path.name,
            np.concatenate([x, rng.standard_normal((len(x), DIM))], 1).astype(np.float32),
        )
        np.save(
            grid_dir / path.name, rng.standard_normal((len(x), 4, 7, 7, DIM)).astype(np.float16)
        )
    (a2_dir / constants.V2_INPUT_MANIFEST_FILENAME).write_text(
        json.dumps({"crn": constants.V2_OFF, "motion": constants.V5_ENCODER})
    )
    _head(grid_dir / constants.V5_HEAD_FILENAME)
    return fixture, a2_dir, grid_dir


class TestTraining:
    @pytest.mark.parametrize("row", list(constants.V5_TILE_ROWS))
    def test_losses(self, tmp_path: Path, row: str) -> None:
        fixture, a2_dir, grid_dir = _tile_fixture(tmp_path)
        cfg = v5_stage1.row_config(row)
        cfg.train.batch_size, cfg.train.num_epochs = 4, 1
        set_global_seed(cfg.train.seed)
        assert v5_grid_lag(cfg, grid_dir) is None
        dataset = DVSFeatureDataset(
            data_dir=fixture.data_dir, clip_dir=a2_dir, dvs=cfg.dvs, require_flow=False,
            seed=cfg.train.seed, grid_dir=grid_dir,
        )
        model = KATVAD.from_config(cfg)
        assert model.grid_readout is not None
        model.grid_readout.load_head(grid_dir / constants.V5_HEAD_FILENAME)
        trainer = Trainer(
            cfg=cfg, model=model, dataset=dataset,
            class_names=load_class_names(fixture.data_dir),
            text_encode_fn=make_text_encoder(model, "stub", CPU, dim=constants.HIDDEN_DIM),
            output_dir=tmp_path / "out", device=CPU, tile_boundary_rows=1,
        )
        losses = trainer.compute_losses(collate_variable_length([dataset[i] for i in range(4)]))
        assert torch.isfinite(losses["total"])
        has_tile = constants.V5_TILE_ROWS[row]["readout"] == constants.V5_READOUT_TILE
        assert ("tile" in losses) == has_tile
        if has_tile:
            losses["total"].backward()
            assert isinstance(model.grid_readout, TileReadout)
            grad = model.grid_readout.scorer.out.weight.grad
            assert grad is not None and float(grad.abs().sum()) > 0

    def test_causal_cache_is_refused(self, tmp_path: Path) -> None:
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / constants.V5_GRID_MANIFEST_FILENAME).write_text(
            json.dumps({"stride": constants.FRAME_STRIDE, "frame_step": STEP})
        )
        with pytest.raises(ValueError, match="PREREG_V5 §3"):
            v5_grid_lag(v5_stage1.row_config("R6c"), tmp_path)


class TestMaps:
    def test_tile_map_is_the_slot_max(self) -> None:
        model = KATVAD.from_config(v5_stage1.row_config("R6c"))
        rows = torch.randn(1, 3, BASE + WIDTH)
        out = model(rows, torch.tensor([3]), class_feats=torch.randn(2, constants.HIDDEN_DIM))
        assert out["v5_tile_logits"].shape == (1, 3, CELLS)
        maps = v5_maps.step_maps(out)
        assert maps.shape == (1, 3, constants.V5_GRID_TILES)
        torch.testing.assert_close(maps, torch.full_like(maps, 0.5))  # zero logits at init

    def test_r3c_map_is_the_change_map(self) -> None:
        model = KATVAD.from_config(v5_stage1.row_config("R3c")).eval()
        rows = torch.randn(4, BASE + WIDTH)
        maps, scores, change = v5_maps.clip_maps(
            model, rows, lambda: torch.randn(2, constants.HIDDEN_DIM), tile_grid=True
        )
        assert change is not None and maps.shape == (4, constants.V5_GRID_TILES)
        np.testing.assert_array_equal(maps, change)
        assert scores.shape == (4,)

    def test_mass_in_box(self) -> None:
        labels = np.ones(6, dtype=bool)
        quarter = [[0.0, 0.0, 640.0, 360.0]]  # top-left quarter of 1280 x 720
        boxes = {0: quarter, 3: quarter}
        planted = np.zeros((2, 49))
        planted[:, 0] = 1.0  # all mass on the top-left tile
        read = v5_maps.clip_mass(planted, np.array([0, 3]), labels, boxes)
        assert read is not None and read[0] == pytest.approx(1.0)
        uniform = v5_maps.clip_mass(np.ones((2, 49)), np.array([0, 3]), labels, boxes)
        assert uniform is not None and uniform[0] == pytest.approx(uniform[1])


class TestRead:
    def test_position_alone_reads_half_on_cross_clip_pairs(self) -> None:
        from core.tools import v5_tile_read as rd

        labels, scores = {}, {}
        hump = np.repeat(np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=float), 4)  # bin-constant
        for i in range(12):
            y = np.zeros(40, dtype=bool)
            y[14 + i : 22 + i] = True  # mid-clip accidents at varying places
            labels[f"c{i}"] = y
            scores[f"c{i}"] = hump
        assert rd.macro_auc(scores, labels) > 0.75  # position wins inside each clip
        cross = rd.cross_clip_position_auc(scores, labels)
        assert cross == pytest.approx(0.5)  # but not across clips at the same position

    def test_cross_clip_counts_only_other_clips(self) -> None:
        from core.tools import v5_tile_read as rd

        labels = {"a": np.array([0, 1], dtype=bool), "b": np.array([0, 1], dtype=bool)}
        scores = {"a": np.array([0.0, 1.0]), "b": np.array([1.0, 0.0])}  # b inverted
        assert rd.cross_clip_position_auc(scores, labels, bins=1) == pytest.approx(0.5)

    def test_shift_peak_finds_a_lag(self) -> None:
        from core.tools import v5_tile_read as rd

        labels, scores = {}, {}
        for i in range(10):
            y = np.zeros(60, dtype=bool)
            y[20 + i : 30 + i] = True
            labels[str(i)] = y
            scores[str(i)] = np.roll(y.astype(float), -6)  # evidence 6 frames early
        st = rd.shift_test(scores, labels, 3)
        assert st["peak_offset"] == 2 and not st["pass"]

    def test_position_fit_is_a_cubic(self) -> None:
        from core.tools import v5_tile_read as rd

        labels = {str(i): np.r_[np.zeros(10), np.ones(10), np.zeros(10)] for i in range(5)}
        prior = rd.position_scores(labels, rd.fit_position(labels))
        assert int(np.argmax(prior["0"])) in range(10, 20)

    def test_run_writes_every_proposal_read(self, tmp_path: Path) -> None:
        from core.tools import protocol_b_eval as pb
        from core.tools import v5_tile_read as rd

        rng = np.random.default_rng(1)
        labels: dict[str, np.ndarray] = {}
        for i in range(24):  # DoTA-style ids: two clips per source video
            y = np.zeros(60, dtype=bool)
            y[20 + i % 7 : 34 + i % 7] = True
            labels[f"src{i // 2:03d}_{i % 2:06d}"] = y
        signal = {"R1c": 0.6, "R3c": 0.6, "R6c": 1.2}
        for row, gain in signal.items():
            d = tmp_path / "pb" / row
            d.mkdir(parents=True)
            scores = {v: gain * y + rng.standard_normal(len(y)) for v, y in labels.items()}
            pb.write_clip_scores(d / pb.CLIP_SCORES_NPZ, scores, labels)
            aucs = {v: rd.macro_auc({v: scores[v]}, {v: y}) for v, y in labels.items()}
            (d / pb.CLIP_AUCS_JSON).write_text(json.dumps(aucs))
            (d / pb.READOUT_JSON).write_text(json.dumps(
                {"per_run_macro": {"s2024": float(np.mean(list(aucs.values())))}}))
            n = tmp_path / "nexar" / row
            n.mkdir(parents=True)
            (n / "nexar_readout.json").write_text(json.dumps(
                {"per_run_crop_endpoint": {"s2024": 0.70}, "whole": {"macro": None}}))
            run = tmp_path / "runs" / row / "s2024"
            run.mkdir(parents=True)
            (run / "metrics.jsonl").write_text(json.dumps(
                {"global_step": 9, "mil": 0.5, "tile": 0.3, "tile_p_abnormal": 0.2,
                 "tile_p_normal": 0.05, "v5_temp": 1.0, "v5_entropy_q0": 5.0}) + "\n")
        hit = {"mean": 0.1, "low": 0.05, "high": 0.15, "clips": 24}
        maps = {"rows": {r: {"beats_centre_and_change": True, "hit_minus_chance": hit,
                             "hit_minus_centre": hit, "hit_minus_change": hit,
                             "stauc": {"stauc": 0.6, "auc": 0.7},
                             "stauc_baselines": {"uniform": {"stauc": 0.5}}}
                         for r in ("R3c", "R6c")}}
        (tmp_path / "maps.json").write_text(json.dumps(maps))
        args = rd.build_arg_parser().parse_args([
            "--pb-dir", str(tmp_path / "pb"), "--nexar-dir", str(tmp_path / "nexar"),
            "--maps-readout", str(tmp_path / "maps.json"), "--runs-root", str(tmp_path / "runs"),
            "--out-dir", str(tmp_path / "out"),
        ])
        out = rd.run(args)
        assert out["preliminary"]
        assert out["gate"]["R6c"]["outcome"] == rd.PASS
        assert out["clip_delta"]["R6c"]["verdict"] == rd.DIFFERENT
        assert out["position_fused"]["delta_vs_control"]["R6c"]["low"] > 0
        assert out["mechanics"]["R6c/s2024"]["finite"]
        assert set(out["ap"]["clip_macro"]) == {"R1c", "R3c", "R6c"}
        md = (tmp_path / "out" / rd.READOUT_MD).read_text()
        assert "PRELIMINARY" in md and "Maps on DoTA-CAP-dev" in md and "Mechanics" in md


class TestReadSupport:
    def test_uniform_tarr_is_the_box_share(self) -> None:
        from core.tools import v5_maps

        box = [[0.0, 0.0, 640.0, 360.0]]  # a quarter of the 1280 x 720 frame
        assert v5_maps.uniform_tarr(1, box) == pytest.approx(0.25, abs=0.02)
        assert v5_maps.uniform_tarr(0, box) == 0.0

    def test_change_map_baseline_beats_uniform_when_it_covers_the_box(self) -> None:
        from core.tools import v5_maps

        n, stride = 12, 1
        labels = {"a": np.r_[np.zeros(6), np.ones(6)].astype(bool)}
        boxes = {"a": {f: [[0.0, 0.0, 640.0, 360.0]] for f in range(6, 12)}}
        scores = {"a": np.r_[np.zeros(6), np.ones(6)]}
        top_left = np.zeros((n, 49))
        top_left[:, [0, 1, 2, 7, 8, 9, 14, 15, 16]] = 1.0
        b = v5_maps.baseline_staucs(scores, {"a": top_left}, labels, boxes, stride)
        assert b["change"]["stauc"] > b["uniform"]["stauc"]

    def test_stage0_todo_skips_finished_tile_grids(self, tmp_path: Path) -> None:
        from core.tools import v5_stage0

        np.save(tmp_path / "c.npy", np.zeros((3, 4, 7, 7, 2), dtype=np.float16))
        clips = {"c": {"dota_id": "c"}}
        report = {"c": {"reason": "ok"}}
        work = v5_stage0.todo(
            {"clips": clips}, ["c"], tmp_path, tmp_path, set(), report, frozenset(), maps=False
        )
        assert work == []
