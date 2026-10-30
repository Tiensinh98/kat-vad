"""v5 R7 Phase 1: CLIP 14 x 14 x 512 patch grids for Nexar and DoTA-CAP (PREREG_V5 §4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from core import constants
from core.data.v5_grids import read_grid_geometry
from core.tests.test_v5_stage0 import _frames, _write_tar, tiny, tiny_clip  # noqa: F401
from core.tools import extract_clip_features, nexar_extract, v5_stage0, v5_stage1
from core.tools import extract_video_features as evf

CPU = torch.device("cpu")
SUBDIR = "images"
GRID = constants.V5_CLIP_GRID
PROJ = 24  # tiny_clip's projection_dim


def _no_videomae(*_: Any) -> Any:
    raise AssertionError("the CLIP grid path must not load VideoMAE")


class TestPatchTokens:
    def test_shared_core_matches_stage0_and_keeps_cls_identity(
        self,
        tmp_path: Path,
        tiny_clip: extract_clip_features.ImageEncoder,  # noqa: F811
    ) -> None:
        paths = []
        for i, data in enumerate(_frames(5, 3)):
            paths.append(tmp_path / f"{i:06d}.jpg")
            paths[-1].write_bytes(data)
        embeds, grid, worst = v5_stage0.clip_patches(paths, [0, 1, 2], tiny_clip, CPU, 2)
        pixels = extract_clip_features.preprocess_frames(
            v5_stage0.read_images(paths), center_crop=False
        )
        e2, g2, w2 = v5_stage0.clip_patch_tokens(pixels, tiny_clip)
        assert grid.shape == (3, GRID, GRID, PROJ)
        np.testing.assert_allclose(grid, g2.numpy(), atol=1e-5)
        np.testing.assert_allclose(embeds, e2.numpy(), atol=1e-5)
        assert max(worst, w2) <= constants.V5_G1_MAX_ABS


class TestGap:
    def test_dc14_gap_reads_the_stored_map(self, tmp_path: Path) -> None:
        grid = np.random.default_rng(0).normal(size=(4, GRID, GRID, 8)).astype(np.float16)
        path = tmp_path / "m.npz"
        assert v5_stage0.dc14_gap(grid, [0, 3, 6, 9], path) is None
        np.savez(
            path, steps=np.array([0, 3, 6, 9]), dc14=v5_stage0.change_map(grid.astype(np.float32))
        )
        assert v5_stage0.dc14_gap(grid, [0, 3, 6, 9], path) == 0.0
        assert v5_stage0.dc14_gap(grid, [0, 3, 6, 12], path) == float("inf")


class TestDotaCap:
    def _setup(
        self,
        tmp_path: Path,
        tiny_clip: extract_clip_features.ImageEncoder,  # noqa: F811
        tiny: Any,  # noqa: F811
    ) -> dict[str, Any]:
        n = 24
        cap = {"000010": _frames(1, n), "000012": _frames(3, n)}
        ids = {"aaa_bbb_000001": "000010", "eee_fff_000001": "000012"}
        ref_dir, v2_dir, s1_dir = tmp_path / "ref", tmp_path / "v2", tmp_path / "s1"
        v2_dir.mkdir()
        s1_dir.mkdir()
        for dota_id, cap_id in ids.items():
            folder = ref_dir / dota_id / SUBDIR
            folder.mkdir(parents=True)
            for i, data in enumerate(cap[cap_id]):
                (folder / f"{i:06d}.jpg").write_bytes(data)
            np.save(
                v2_dir / f"{dota_id}.npy",
                evf.encode_frame_dir(
                    ref_dir / dota_id, tiny, CPU, stride=1, subdir=SUBDIR, clip_step=1
                ),
            )
            np.save(
                s1_dir / f"{dota_id}.npy",
                extract_clip_features.encode_frame_dir(
                    ref_dir / dota_id, tiny_clip, CPU, stride=1, subdir=SUBDIR, center_crop=False
                ),
            )
        alignment = {
            "clips": {
                d: {
                    "dota_id": d,
                    "cap_id": c,
                    "cap_frames": n,
                    "frame_map": list(range(n)),
                    "reason": "ok",
                }
                for d, c in ids.items()
            }
        }
        (tmp_path / "align.json").write_text(json.dumps(alignment))
        (tmp_path / "final.json").write_text(json.dumps({"ids": sorted(ids)}))
        parts = _write_tar(tmp_path / "parts" / "11.part_aa", cap)
        return {"n": n, "v2": v2_dir, "s1": s1_dir, "parts": parts}

    def _cmd(
        self, tmp_path: Path, env: dict[str, Any], out: Path, report: Path, *extra: str
    ) -> list[str]:
        return [
            "extract",
            "--alignment",
            str(tmp_path / "align.json"),
            "--final-readout",
            str(tmp_path / "final.json"),
            "--parts",
            *map(str, env["parts"]),
            "--work-dir",
            str(tmp_path / "work"),
            "--dota-clip-dir",
            str(env["s1"]),
            "--report",
            str(report),
            "--out-dir",
            str(out),
            "--batch-size",
            "4",
            "--device",
            "cpu",
            *extra,
        ]

    def test_every_clip_stored_and_dev_reproduces_stage0(
        self,
        tmp_path: Path,
        tiny: Any,  # noqa: F811
        tiny_clip: extract_clip_features.ImageEncoder,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = self._setup(tmp_path, tiny_clip, tiny)
        monkeypatch.setattr(
            v5_stage0, "load_split", lambda name: ["aaa_bbb_000001"]
        )  # one dev clip
        monkeypatch.setattr(constants, "CACHE_ROOT", tmp_path / "cache")
        monkeypatch.setattr(v5_stage0, "load_pretrained", lambda n_, d, w: tiny)
        monkeypatch.setattr(extract_clip_features, "load_pretrained_encoder", lambda d: tiny_clip)
        stage0_out = tmp_path / "s0"
        v5_stage0.main(
            self._cmd(
                tmp_path, env, stage0_out, stage0_out / "r.json", "--v2-cache", str(env["v2"])
            )
        )
        assert (stage0_out / "maps" / "aaa_bbb_000001.npz").exists()

        monkeypatch.setattr(v5_stage0, "load_pretrained", _no_videomae)
        out, report = tmp_path / "r7", tmp_path / "r7" / "extract_cgrid_11.json"
        v5_stage0.main(
            self._cmd(
                tmp_path,
                env,
                out,
                report,
                "--clip-grid",
                "--stage0-maps-dir",
                str(stage0_out / "maps"),
            )
        )
        got = json.loads(report.read_text())
        assert {v: e["reason"] for v, e in got.items()} == {
            "aaa_bbb_000001": "ok",
            "eee_fff_000001": "ok",
        }
        assert got["aaa_bbb_000001"]["dc14_max_abs"] <= constants.V5_R7_DC14_MAX_ABS
        assert "dc14_max_abs" not in got["eee_fff_000001"]  # eval clip: stored, no Stage 0 map
        grid_dir = v5_stage0.grid_out_dir("", clip_grid=True)
        assert grid_dir.name == f"{constants.DOTA_CAP_DATASET}_s3_{constants.V5_CLIP_GRID_TAG}"
        grid = np.load(grid_dir / "eee_fff_000001.npy")
        assert grid.shape == (env["n"] // 3, GRID, GRID, PROJ) and grid.dtype == np.float16
        geometry = read_grid_geometry(grid_dir)
        assert (geometry.stride, geometry.slots, geometry.window) == (
            constants.V5_R7_DOTA_STRIDE,
            constants.V5_SLOTS_FRAME,
            constants.V5_WINDOW_FRAME,
        )

    def test_a_dc14_mismatch_stores_nothing(
        self,
        tmp_path: Path,
        tiny: Any,  # noqa: F811
        tiny_clip: extract_clip_features.ImageEncoder,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = self._setup(tmp_path, tiny_clip, tiny)
        monkeypatch.setattr(v5_stage0, "load_split", lambda name: ["aaa_bbb_000001"])
        monkeypatch.setattr(constants, "CACHE_ROOT", tmp_path / "cache")
        monkeypatch.setattr(v5_stage0, "load_pretrained", _no_videomae)
        monkeypatch.setattr(extract_clip_features, "load_pretrained_encoder", lambda d: tiny_clip)
        maps = tmp_path / "maps"
        maps.mkdir()
        rows = env["n"] // 3
        steps = evf.step_end_frames(env["n"], 3)
        np.savez(
            maps / "aaa_bbb_000001.npz", steps=np.asarray(steps), dc14=np.ones((rows, GRID * GRID))
        )
        out, report = tmp_path / "r7", tmp_path / "r7" / "rep.json"
        v5_stage0.main(
            self._cmd(tmp_path, env, out, report, "--clip-grid", "--stage0-maps-dir", str(maps))
        )
        got = json.loads(report.read_text())
        assert got["aaa_bbb_000001"]["reason"] == "g_r7c"
        assert not (v5_stage0.grid_out_dir("", clip_grid=True) / "aaa_bbb_000001.npy").exists()

    def test_tile_and_clip_grid_are_exclusive(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text(json.dumps({"clips": {}}))
        (tmp_path / "f.json").write_text(json.dumps({"ids": []}))
        with pytest.raises(ValueError, match="--clip-grid"):
            v5_stage0.main(
                [
                    "extract",
                    "--alignment",
                    str(tmp_path / "a.json"),
                    "--final-readout",
                    str(tmp_path / "f.json"),
                    "--parts",
                    str(tmp_path / "p"),
                    "--work-dir",
                    str(tmp_path / "w"),
                    "--report",
                    str(tmp_path / "r.json"),
                    "--out-dir",
                    str(tmp_path),
                    "--tile",
                    "--clip-grid",
                ]
            )


class TestNexar:
    def _run(
        self,
        tmp_path: Path,
        tiny_clip: extract_clip_features.ImageEncoder,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
        frames: dict[str, torch.Tensor],
        reference: dict[str, np.ndarray],
    ) -> tuple[dict[str, Any], Path]:
        census = {"videos": {v: {"label": 1} for v in frames}}
        (tmp_path / "census.json").write_text(json.dumps(census))
        (tmp_path / "ids.txt").write_text("".join(f"{v}\n" for v in frames))
        s1 = tmp_path / "s1"
        s1.mkdir()
        for v, x in reference.items():
            np.save(s1 / f"{v}.npy", x)
        monkeypatch.setattr(
            v5_stage1, "video_paths", lambda root, c, ids: {v: Path(v) for v in ids}
        )
        monkeypatch.setattr(v5_stage1, "decode_resized", lambda path: frames[str(path)])
        monkeypatch.setattr(v5_stage1, "load_pretrained_encoder", lambda d: tiny_clip)
        monkeypatch.setattr(v5_stage1, "load_pretrained", _no_videomae)
        grid_dir, out = tmp_path / "grids", tmp_path / "out"
        v5_stage1.main(
            [
                "extract",
                "--clip-grid",
                "--root",
                str(tmp_path),
                "--census",
                str(tmp_path / "census.json"),
                "--ids-file",
                str(tmp_path / "ids.txt"),
                "--clip-s1-dir",
                str(s1),
                "--grid-dir",
                str(grid_dir),
                "--stride",
                "8",
                "--report",
                str(out / "extract_cgrid_train.json"),
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--out-dir",
                str(out),
            ]
        )
        return json.loads((out / "g1_extract_cgrid_s8.json").read_text()), grid_dir

    def test_grids_match_n2_rows_and_failures_store_nothing(
        self,
        tmp_path: Path,
        tiny_clip: extract_clip_features.ImageEncoder,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        gen = torch.Generator().manual_seed(0)
        frames = {
            v: torch.rand(20, 3, constants.CROP_SIZE, constants.CROP_SIZE, generator=gen)
            for v in ("good", "short", "moved")
        }
        ref = {v: nexar_extract.clip_rows(x, tiny_clip, CPU, 7) for v, x in frames.items()}
        ref["short"] = ref["short"][:-1]
        ref["moved"] = -ref["moved"]  # rows of some other content
        summary, grid_dir = self._run(tmp_path, tiny_clip, monkeypatch, frames, ref)
        assert (
            summary["failed"] == {"short": "g1_length", "moved": "g1_cos"} and not summary["pass"]
        )
        report = json.loads((tmp_path / "out" / "extract_cgrid_train.json").read_text())
        assert summary["rows"] == 3 and report["good"]["g1_min_cos"] >= constants.V5_R7_G1_MIN_COS
        grid = np.load(grid_dir / "good.npy")
        assert grid.shape == (3, GRID, GRID, PROJ) and grid.dtype == np.float16
        assert not (grid_dir / "short.npy").exists() and not (grid_dir / "moved.npy").exists()
        geometry = read_grid_geometry(grid_dir)
        assert (geometry.stride, geometry.frame_step, geometry.window) == (
            8,
            constants.V5_CLIP_FRAME_STEP,
            constants.V5_WINDOW_FRAME,
        )
        manifest = json.loads((grid_dir / constants.V5_GRID_MANIFEST_FILENAME).read_text())
        assert (
            manifest["revision"] == constants.CLIP_MODEL_REVISION and manifest["dtype"] == "float16"
        )

    def test_resume_skips_finished_videos(
        self,
        tmp_path: Path,
        tiny_clip: extract_clip_features.ImageEncoder,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        frames = {"good": torch.rand(9, 3, constants.CROP_SIZE, constants.CROP_SIZE)}
        ref = {"good": nexar_extract.clip_rows(frames["good"], tiny_clip, CPU, 4)}
        summary, grid_dir = self._run(tmp_path, tiny_clip, monkeypatch, frames, ref)
        assert summary["pass"] and summary["rows"] == 2
        monkeypatch.setattr(v5_stage1, "decode_resized", _no_videomae)  # nothing left to decode
        stamp = (grid_dir / "good.npy").stat().st_mtime_ns
        v5_stage1.main(
            [
                "extract",
                "--clip-grid",
                "--root",
                str(tmp_path),
                "--census",
                str(tmp_path / "census.json"),
                "--ids-file",
                str(tmp_path / "ids.txt"),
                "--clip-s1-dir",
                str(tmp_path / "s1"),
                "--grid-dir",
                str(grid_dir),
                "--stride",
                "8",
                "--report",
                str(tmp_path / "out" / "extract_cgrid_train.json"),
                "--device",
                "cpu",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        assert (grid_dir / "good.npy").stat().st_mtime_ns == stamp


# --------------------------------------------------------------------------- Phase 2 (§4.2)

R7_DIM, R7_CELLS = constants.V5_CLIP_GRID_DIM, constants.V5_R7_CELLS


def _frame_fixture(tmp_path: Path) -> tuple[Any, Path, Path]:
    """The MSAD fixture with A2-wide rows and a per-frame CLIP 14 x 14 x 512 cache."""
    from core.tests.fixtures import build_fixture

    fixture = build_fixture(tmp_path / "fx")
    rng = np.random.default_rng(2)
    a2_dir, grid_dir = tmp_path / "a2", tmp_path / "cgrid"
    a2_dir.mkdir()
    grid_dir.mkdir()
    manifest = v5_grids_manifest(constants.FRAME_STRIDE)
    (grid_dir / constants.V5_GRID_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    for path in fixture.clip_dir.glob("*.npy"):
        x = np.load(path)
        u = rng.standard_normal((len(x), constants.V5_GRID_DIM))
        np.save(a2_dir / path.name, np.concatenate([x, u], 1).astype(np.float32))
        np.save(
            grid_dir / path.name,
            rng.standard_normal((len(x), GRID, GRID, R7_DIM)).astype(np.float16),
        )
    (a2_dir / constants.V2_INPUT_MANIFEST_FILENAME).write_text(
        json.dumps({"crn": constants.V2_OFF, "motion": constants.V5_ENCODER})
    )
    return fixture, a2_dir, grid_dir


def v5_grids_manifest(stride: int) -> dict[str, Any]:
    from core.data.v5_grids import clip_grid_manifest

    return clip_grid_manifest(stride, "test rows", "test transform")


class TestR7Config:
    @pytest.mark.parametrize("row", list(constants.V5_R7_ROWS))
    def test_rows_validate(self, row: str) -> None:
        from core.config import validate_v5

        validate_v5(v5_stage1.row_config(row))

    @pytest.mark.parametrize(
        ("override", "match"),
        [
            (["v5.readout=change_topk"], "mean or tile"),
            (["v5.smil_weight=0"], "iff readout=tile"),
            (["v5.grid_slots=4"], "1 slot"),
            (["v5.keys=change"], "no queries or change keys"),
        ],
    )
    def test_bad_rows_raise(self, override: list[str], match: str) -> None:
        from core.config import validate_v5

        with pytest.raises(ValueError, match=match):
            validate_v5(v5_stage1.row_config("R7t", override))


class TestR7Model:
    def test_r7t_starts_as_r7m_and_both_as_r0(self) -> None:
        from core.data.v5_grids import v5_rows
        from core.models.kat_vad import KATVAD

        rng = np.random.default_rng(0)
        base = rng.standard_normal((12, constants.HIDDEN_DIM + constants.V5_GRID_DIM))
        grid = rng.standard_normal((12, GRID, GRID, R7_DIM)).astype(np.float16)
        cells = torch.from_numpy(grid.astype(np.float32)).reshape(1, 12, R7_CELLS, R7_DIM)
        change = torch.from_numpy(
            v5_rows(base.astype(np.float32), grid, constants.V5_R7_CHANGE_ROWS)[
                :, -cells[0, 0].numel() :
            ]
        ).reshape(1, 12, R7_CELLS, R7_DIM)
        motion = {}
        for row in constants.V5_R7_ROWS:
            model = KATVAD.from_config(v5_stage1.row_config(row)).eval()
            assert model.grid_readout is not None and model.motion_residual is not None
            assert model.motion_residual.motion_dim == R7_DIM  # W_c reads the 512-d readout
            rows = torch.from_numpy(v5_rows(base.astype(np.float32), grid, lag_of(row)))[None]
            with torch.no_grad():
                out = model(rows, torch.tensor([12]))
                uses = model.grid_readout.uses_change
                motion[row] = model.grid_readout(cells, change if uses else None).motion
            assert float(out["motion_share"]) == 0.0  # W_c = 0: R0's function at init
        torch.testing.assert_close(motion["R7t"], motion["R7m"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(  # identity head: LN of the cell mean, no fc_norm
            motion["R7m"], torch.nn.functional.layer_norm(cells.mean(-2), (R7_DIM,), eps=1e-6)
        )

    def test_tile_readout_refuses_a_missing_change_block(self) -> None:
        from core.models.mqr import TileReadout

        with pytest.raises(ValueError, match="ΔG"):
            TileReadout(R7_DIM, R7_CELLS, frame=True)(torch.randn(1, 3, R7_CELLS, R7_DIM))

    def test_frame_features_remove_the_frame_median(self) -> None:
        from core.models.mqr import frame_tile_features

        grid = torch.randn(2, 3, R7_CELLS, 8) + 5.0  # a frame-wide offset
        feats = frame_tile_features(grid, torch.zeros_like(grid))
        torch.testing.assert_close(feats, frame_tile_features(grid - 5.0, torch.zeros_like(grid)))
        assert feats.shape == (2, 3, R7_CELLS, 16)


def lag_of(row: str) -> int | None:
    from core.data.v5_grids import frame_lag

    return frame_lag(str(constants.V5_R7_ROWS[row]["readout"]))


class TestR7Training:
    @pytest.mark.parametrize("row", list(constants.V5_R7_ROWS))
    def test_losses_and_lag(self, tmp_path: Path, row: str) -> None:
        from core.data.collate import collate_variable_length
        from core.data.dataset import DVSFeatureDataset
        from core.models.kat_vad import KATVAD
        from core.models.mqr import TileReadout
        from core.models.text_encoding import make_text_encoder
        from core.train import Trainer, load_class_names, set_global_seed, v5_grid_lag

        fixture, a2_dir, grid_dir = _frame_fixture(tmp_path)
        cfg = v5_stage1.row_config(row)
        cfg.train.batch_size, cfg.train.num_epochs = 4, 1
        set_global_seed(cfg.train.seed)
        lag = v5_grid_lag(cfg, grid_dir)
        assert lag == lag_of(row)
        dataset = DVSFeatureDataset(
            data_dir=fixture.data_dir, clip_dir=a2_dir, dvs=cfg.dvs, require_flow=False,
            seed=cfg.train.seed, grid_dir=grid_dir, grid_lag=lag,
        )  # fmt: skip
        model = KATVAD.from_config(cfg)
        trainer = Trainer(
            cfg=cfg, model=model, dataset=dataset,
            class_names=load_class_names(fixture.data_dir),
            text_encode_fn=make_text_encoder(model, "stub", CPU, dim=constants.HIDDEN_DIM),
            output_dir=tmp_path / "out", device=CPU,
        )  # fmt: skip
        losses = trainer.compute_losses(collate_variable_length([dataset[i] for i in range(4)]))
        assert torch.isfinite(losses["total"])
        assert ("tile" in losses) == (row == "R7t")
        losses["total"].backward()
        if row == "R7t":
            assert isinstance(model.grid_readout, TileReadout)
            grad = model.grid_readout.scorer.out.weight.grad
            assert grad is not None and float(grad.abs().sum()) > 0

    def test_videomae_cache_is_refused(self, tmp_path: Path) -> None:
        from core.train import v5_grid_lag

        (tmp_path / constants.V5_GRID_MANIFEST_FILENAME).write_text(
            json.dumps({"stride": constants.FRAME_STRIDE, "frame_step": 3})
        )
        with pytest.raises(ValueError, match="1-slot causal"):
            v5_grid_lag(v5_stage1.row_config("R7t"), tmp_path)


class TestR7Maps:
    def test_baseline_is_stage0_c14(self) -> None:
        from core.data.v5_grids import v5_rows
        from core.tools import v5_maps

        rng = np.random.default_rng(3)
        grid = rng.standard_normal((6, GRID, GRID, R7_DIM)).astype(np.float16)
        base = rng.standard_normal((6, 896)).astype(np.float32)
        rows = torch.from_numpy(v5_rows(base, grid, constants.V5_R7_CHANGE_ROWS))
        got = v5_maps.frame_change_map(rows)
        want = v5_stage0.change_map(grid.astype(np.float32))
        assert np.isnan(got[0]).all() and np.isnan(want[0]).all()
        np.testing.assert_allclose(got[1:], want[1:], atol=1e-6)

    def test_rule_uses_mass_minus_centre_on_the_map_grid(self) -> None:
        from core.tools import v5_maps

        steps = np.arange(0, 30, 3)
        labels = np.zeros(30, dtype=bool)
        labels[9:] = True
        box = [[0.0, 0.0, 200.0, 120.0]]  # top-left corner, away from the centre
        boxes = {int(t): box for t in steps if labels[t]}
        w = v5_stage0.tile_weights(box, GRID)
        good = np.tile(w, (len(steps), 1)) + 1e-3  # a map that sits on the box
        read = v5_maps.clip_point_reads([good], steps, labels, boxes, change=np.ones_like(good))
        assert read is not None
        assert read["hit"] == 1.0 and read["hit_centre"] == 0.0
        assert read["mass"] > read["mass_centre"]
        per_clip = {f"s{i:03d}_000001": read for i in range(8)}
        summary = v5_maps.point_summary(per_clip, {}, {})
        assert summary["mass_minus_centre"]["low"] > 0
        assert "r7_points" in summary


class TestR7Read:
    def test_gate_without_nexar_says_not_read(self) -> None:
        from core.tools import v5_tile_read as rd

        fused = {"delta_vs_control": {"R7t": {"mean": 0.0, "low": -0.01, "high": 0.01}}}
        g = rd.pilot_gate("R7t", {"r7_points": True}, None, fused, "R7m", rd.GATE_R7)
        assert g["outcome"].startswith(rd.R7_PASS) and "not read" in g["outcome"]
        g = rd.pilot_gate("R7t", {"r7_points": False}, None, fused, "R7m", rd.GATE_R7)
        assert g["outcome"].startswith(rd.R7_NULL)
        with pytest.raises(ValueError, match="nexar"):
            rd.pilot_gate("R7t", {}, None, fused, "R7m", rd.GATE_V51)


class TestR7Eval:
    def test_eval_grid_appends_grid_and_change_at_the_cache_stride(self, tmp_path: Path) -> None:
        from core.data.v5_grids import eval_grid

        (tmp_path / constants.V5_GRID_MANIFEST_FILENAME).write_text(
            json.dumps(v5_grids_manifest(constants.V5_R7_DOTA_STRIDE))
        )
        grid = np.random.default_rng(4).normal(size=(5, GRID, GRID, R7_DIM)).astype(np.float16)
        np.save(tmp_path / "clip.npy", grid)
        base = np.zeros((5, 896), dtype=np.float32)
        tile = eval_grid(constants.V5_READOUT_TILE, constants.V5_KEYS_CONTENT, tmp_path, 3)
        mean = eval_grid(constants.V5_READOUT_MEAN, constants.V5_KEYS_CONTENT, tmp_path, 3)
        assert tile is not None and mean is not None
        assert tile.lag == constants.V5_R7_CHANGE_ROWS and mean.lag is None
        rows = tile.append(base, "clip", 0, 15)
        assert rows.shape == (5, 896 + 2 * constants.V5_R7_GRID_WIDTH)
        assert mean.append(base, "clip", 0, 15).shape == (5, 896 + constants.V5_R7_GRID_WIDTH)
        with pytest.raises(ValueError, match="s1\\[::3\\]"):
            eval_grid(constants.V5_READOUT_TILE, constants.V5_KEYS_CONTENT, tmp_path, 1)
