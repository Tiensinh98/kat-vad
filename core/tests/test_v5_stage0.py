"""v5 Stage 0: token grids, change maps, box reads and the S0-11 decision (PREREG_V5.md §1)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torchvision.io import encode_jpeg

from core import constants
from core.models import videomae_v2
from core.tools import extract_clip_features, v5_stage0
from core.tools import extract_video_features as evf

CPU = torch.device("cpu")
SUBDIR = "images"
TINY_VIDEOMAE = (32, 1, 2)
GRID7 = constants.V5_GRID_COARSE


@pytest.fixture(scope="module")
def tiny() -> videomae_v2.VideoMAEv2Encoder:
    torch.manual_seed(1)
    return videomae_v2.VideoMAEv2Encoder(*TINY_VIDEOMAE).eval()


@pytest.fixture(scope="module")
def tiny_clip() -> extract_clip_features.ImageEncoder:
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=constants.CROP_SIZE,
        patch_size=16,
        projection_dim=24,
    )
    torch.manual_seed(0)
    model: extract_clip_features.ImageEncoder = CLIPVisionModelWithProjection(config).eval()
    return model


class TestTokens:
    def test_forward_is_fc_norm_of_the_token_mean(
        self, tiny: videomae_v2.VideoMAEv2Encoder
    ) -> None:
        x = torch.randn(
            2, 3, constants.VIDEOMAE_CLIP_FRAMES, constants.CROP_SIZE, constants.CROP_SIZE
        )
        with torch.no_grad():
            tokens = tiny.forward_tokens(x)
            torch.testing.assert_close(tiny(x), tiny.fc_norm(tokens.mean(1)))
        assert tokens.shape == (2, 8 * 14 * 14, TINY_VIDEOMAE[0])

    def test_grids_keep_the_token_mean(self) -> None:
        tokens = torch.randn(3, 8 * 14 * 14, 5)
        v14, v7 = v5_stage0.token_grids(tokens)
        assert v14.shape == (3, 14, 14, 5) and v7.shape == (3, GRID7, GRID7, 5)
        mean = tokens.mean(1)
        torch.testing.assert_close(v14.mean((1, 2)), mean)
        torch.testing.assert_close(v7.mean((1, 2)), mean)

    def test_grid_tile_is_the_matching_patch_block(self) -> None:
        tokens = torch.zeros(1, 8, 14, 14, 1)
        tokens[0, :, 2:4, 6:8] = 1.0  # one 2x2 block -> V7 tile (1, 3)
        _, v7 = v5_stage0.token_grids(tokens.reshape(1, -1, 1))
        assert v7[0, 1, 3, 0] == 1.0 and float(v7.sum()) == 1.0

    def test_token_count_must_factor(self) -> None:
        with pytest.raises(ValueError, match="not T'"):
            v5_stage0.token_grids(torch.zeros(1, 100, 4))


class TestChange:
    def test_static_grid_has_zero_change_and_nan_first_row(self) -> None:
        grid = np.repeat(np.random.default_rng(0).normal(size=(1, 7, 7, 4)), 3, axis=0)
        delta = v5_stage0.change_map(grid)
        assert delta.shape == (3, 49) and np.isnan(delta[0]).all()
        np.testing.assert_allclose(delta[1:], 0.0, atol=1e-12)

    def test_change_is_scale_free(self) -> None:
        grid = np.random.default_rng(1).normal(size=(4, 7, 7, 6))
        np.testing.assert_allclose(v5_stage0.change_map(grid * 9.0), v5_stage0.change_map(grid))

    def test_global_change(self) -> None:
        vec = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        out = v5_stage0.global_change(vec)
        assert np.isnan(out[0]) and out[1] == pytest.approx(1.0) and out[2] == pytest.approx(0.0)

    def test_lagged_change_skips_lag_rows(self) -> None:
        grid = np.random.default_rng(3).normal(size=(5, 7, 7, 4))
        delta = v5_stage0.change_map(grid, lag=3)
        assert np.isnan(delta[:3]).all() and np.isfinite(delta[3:]).all()
        np.testing.assert_allclose(delta[3], v5_stage0.change_map(grid[[0, 3]])[1])
        np.testing.assert_allclose(delta[4], v5_stage0.change_map(grid[[1, 4]])[1])
        assert np.isnan(v5_stage0.global_change(grid.reshape(5, -1)[:2], lag=3)).all()


class TestTiles:
    def test_full_frame_box_covers_every_tile(self) -> None:
        np.testing.assert_allclose(v5_stage0.tile_weights([[0, 0, 1280, 720]], GRID7), 1.0)

    def test_left_half_box(self) -> None:
        w = v5_stage0.tile_weights([[0, 0, 640, 720]], 14).reshape(14, 14)
        np.testing.assert_allclose(w[:, :7], 1.0)
        np.testing.assert_allclose(w[:, 7:], 0.0)

    def test_union_not_sum(self) -> None:
        w = v5_stage0.tile_weights([[0, 0, 640, 720], [0, 0, 640, 720]], GRID7)
        assert w.max() == 1.0

    def test_small_box_is_a_fraction_of_one_tile(self) -> None:
        w = v5_stage0.tile_weights([[0, 0, 40, 30]], GRID7)  # ~7x9 px at 224 inside tile 0
        assert w[0] == pytest.approx(7 * 10 / 32**2, rel=0.2) and w[1:].sum() == 0


def _boxes(frames: range, box: tuple[float, ...]) -> dict[int, list[list[float]]]:
    return {f: [list(box)] for f in frames}


class TestClipRead:
    LEFT = (0.0, 0.0, 640.0, 720.0)  # left half of the frame

    def _delta(self, steps: int, hot: set[int], left_extra: float) -> np.ndarray:
        rng = np.random.default_rng(2)
        delta = rng.uniform(0.1, 0.2, size=(steps, 196))
        grid = delta.reshape(steps, 14, 14)
        for k in hot:
            grid[k, :, :7] += left_extra
        return grid.reshape(steps, 196)

    def test_evidence_only_inside_the_box_during_the_anomaly(self) -> None:
        steps = np.arange(0, 60, 3)
        labels = np.zeros(60, dtype=np.int64)
        labels[30:] = 1
        anom = {k for k, t in enumerate(steps) if t >= 30}
        read = v5_stage0.clip_read(
            self._delta(len(steps), anom, 1.0),
            np.linspace(0, 1, 20),
            steps,
            labels,
            _boxes(range(30, 60), self.LEFT),
            20,
        )
        assert read.effect is not None and read.effect > 1.0
        assert read.hit == 1.0 and read.chance == pytest.approx(0.5)
        assert read.hit_pre is not None and read.hit_pre < 1.0
        assert read.auc["topk"] > 0.9

    def test_placement_prior_alone_gives_no_effect(self) -> None:
        steps = np.arange(0, 60, 3)
        labels = np.zeros(60, dtype=np.int64)
        labels[30:] = 1
        always = set(range(len(steps)))
        read = v5_stage0.clip_read(
            self._delta(len(steps), always, 1.0),
            np.zeros(20),
            steps,
            labels,
            _boxes(range(30, 60), self.LEFT),
            20,
        )
        assert read.effect is not None and abs(read.effect) < 0.1
        assert read.hit == read.hit_pre == 1.0

    def test_no_pre_steps_means_no_effect(self) -> None:
        steps = np.arange(0, 30, 3)
        labels = np.ones(30, dtype=np.int64)
        read = v5_stage0.clip_read(
            self._delta(len(steps), set(), 0.0),
            np.zeros(10),
            steps,
            labels,
            _boxes(range(30), self.LEFT),
            20,
        )
        assert read.effect is None and read.rho is not None and read.auc == {}


class TestDecide:
    @staticmethod
    def _g(a: bool, b: bool, c: bool) -> dict[str, bool]:
        return {"G-A": a, "G-B": b, "G-C": c}

    @pytest.mark.parametrize(
        ("v7", "v14", "want"),
        [
            ((1, 1, 1), (0, 0, 0), "GO-7"),
            ((1, 0, 1), (1, 1, 1), "GO-14"),
            ((1, 0, 1), (0, 0, 0), "PARTIAL"),
            ((0, 1, 1), (1, 0, 0), "PARTIAL"),
            ((0, 1, 1), (0, 1, 1), "KILL"),
        ],
    )
    def test_table(self, v7: tuple[int, ...], v14: tuple[int, ...], want: str) -> None:
        by_rep = {"V7": self._g(*map(bool, v7)), "V14": self._g(*map(bool, v14))}
        assert v5_stage0.decide(by_rep, True) == want

    def test_invalid_overrides(self) -> None:
        g = self._g(True, True, True)
        assert v5_stage0.decide({"V7": g, "V14": g}, False) == "INVALID"

    def test_gate_a_needs_a_positive_non_ego_point(self) -> None:
        ok = {"mean": 0.2, "low": 0.1, "high": 0.3}
        rep = {
            "effect": ok,
            "effect_non_ego": {**ok, "mean": -0.01},
            "hit_minus_chance": ok,
            "hit_minus_pre": ok,
            "d2_topk_minus_global_non_ego": ok,
        }
        assert v5_stage0.gates(rep) == {"G-A": False, "G-B": True, "G-C": True}


class TestClipPatches:
    def test_projected_cls_is_image_embeds(
        self, tmp_path: Path, tiny_clip: extract_clip_features.ImageEncoder
    ) -> None:
        paths = []
        for i, data in enumerate(_frames(3, 4)):
            paths.append(tmp_path / f"{i:06d}.jpg")
            paths[-1].write_bytes(data)
        embeds, grid, worst = v5_stage0.clip_patches(paths, [0, 2], tiny_clip, CPU, 2)
        assert embeds.shape == (2, 24) and grid.shape == (2, 14, 14, 24)
        assert worst <= constants.V5_G1_MAX_ABS


# --------------------------------------------------------------------------- end to end


def _frames(seed: int, n: int, w: int = 64, h: int = 36) -> list[bytes]:
    gen = torch.Generator().manual_seed(seed)
    return [
        bytes(encode_jpeg((torch.rand(3, h, w, generator=gen) * 255).to(torch.uint8)))
        for _ in range(n)
    ]


def _write_tar(path: Path, videos: dict[str, list[bytes]]) -> list[Path]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for vid, frames in videos.items():
            for i, data in enumerate(frames):
                info = tarfile.TarInfo(f"11/11/{vid}/{SUBDIR}/{i:06d}.jpg")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return [path]


class TestEndToEnd:
    def test_extract_then_readout(
        self,
        tmp_path: Path,
        tiny: videomae_v2.VideoMAEv2Encoder,
        tiny_clip: extract_clip_features.ImageEncoder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        n = 24
        cap = {"000010": _frames(1, n), "000011": _frames(2, n, w=40, h=40)}
        ids = {"aaa_bbb_000001": "000010", "ccc_ddd_000001": "000011"}
        ref_dir = tmp_path / "ref"
        v2_dir, s1_dir = tmp_path / "v2", tmp_path / "s1"
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
        dev = {"aaa_bbb_000001", "ccc_ddd_000001"}
        monkeypatch.setattr(v5_stage0, "load_split", lambda name: sorted(dev))
        monkeypatch.setattr(constants, "CACHE_ROOT", tmp_path / "cache")
        monkeypatch.setattr(v5_stage0, "load_pretrained", lambda n_, d, w: tiny)
        monkeypatch.setattr(extract_clip_features, "load_pretrained_encoder", lambda d: tiny_clip)
        out = tmp_path / "out"
        report = out / "extract_11.json"
        v5_stage0.main(
            [
                "extract",
                "--alignment",
                str(tmp_path / "align.json"),
                "--final-readout",
                str(tmp_path / "final.json"),
                "--parts",
                *map(str, _write_tar(tmp_path / "parts" / "11.part_aa", cap)),
                "--work-dir",
                str(tmp_path / "work"),
                "--dota-clip-dir",
                str(s1_dir),
                "--v2-cache",
                str(v2_dir),
                "--report",
                str(report),
                "--out-dir",
                str(out),
                "--batch-size",
                "4",
                "--device",
                "cpu",
            ]
        )
        got = json.loads(report.read_text())
        assert got["aaa_bbb_000001"]["reason"] == "ok"
        assert got["aaa_bbb_000001"]["g0_min_cos"] >= constants.V5_G0_MIN_COS
        assert got["ccc_ddd_000001"]["reason"] == "g2_aspect"  # square frames: boxes unmappable
        grid = np.load(v5_stage0.grid_out_dir(constants.V5_STAGE0_ENCODER) / "aaa_bbb_000001.npy")
        assert grid.shape == (n // 3, GRID7, GRID7, TINY_VIDEOMAE[0]) and grid.dtype == np.float16
        with np.load(out / "maps" / "aaa_bbb_000001.npz") as maps:
            assert maps["d7"].shape == (n // 3, 49) and maps["dc14"].shape == (n // 3, 196)
            assert maps["clip_s1"].shape[0] == n
        assert not (out / "maps" / "ccc_ddd_000001.npz").exists()

        ann = tmp_path / "ann"
        ann.mkdir()
        meta = {}
        for dota_id in ids:
            labels = [
                {"frame_id": f, "objects": [{"bbox": [0, 0, 640, 720]}] if f >= 12 else []}
                for f in range(n)
            ]
            (ann / f"{dota_id}.json").write_text(
                json.dumps({"ego_involve": dota_id.startswith("aaa"), "labels": labels})
            )
            meta[dota_id] = {
                "video_start": 0,
                "video_end": n - 1,
                "anomaly_start": 12,
                "anomaly_end": n,
                "anomaly_class": "x",
                "num_frames": n,
                "subset": "val",
            }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        (tmp_path / "split.txt").write_text("".join(f"{d}\n" for d in ids))
        readout = v5_stage0.run_readout(
            v5_stage0.build_arg_parser().parse_args(
                [
                    "readout",
                    "--reports",
                    str(report),
                    "--maps-dir",
                    str(out / "maps"),
                    "--annotations-dir",
                    str(ann),
                    "--metadata",
                    str(tmp_path / "meta.json"),
                    "--split-file",
                    str(tmp_path / "split.txt"),
                    "--out-dir",
                    str(out / "REPORTS"),
                ]
            )
        )
        assert readout["decision"] == "INVALID"  # one clip failed G2
        assert readout["validity"]["failed"] == {"ccc_ddd_000001": "g2_aspect"}
        assert readout["validity"]["dev_maps"] == 1
        assert (out / "REPORTS" / v5_stage0.READOUT_MD).read_text().startswith("# v5 Stage 0")
        lagged = v5_stage0.run_lag(
            v5_stage0.build_arg_parser().parse_args(
                [
                    "lag",
                    "--reports",
                    str(report),
                    "--maps-dir",
                    str(out / "maps"),
                    "--annotations-dir",
                    str(ann),
                    "--metadata",
                    str(tmp_path / "meta.json"),
                    "--split-file",
                    str(tmp_path / "split.txt"),
                    "--out-dir",
                    str(out / "REPORTS"),
                    "--grid-dir",
                    str(v5_stage0.grid_out_dir(constants.V5_STAGE0_ENCODER)),
                    "--lags",
                    "1",
                    "2",
                    "--shifts",
                    "0",
                    "2",
                ]
            )
        )
        assert lagged["lag1_vs_stored_d7_max_abs"] < 1e-2  # fp16 grid vs fp32 maps
        assert set(lagged["cells"]) == {"lag1_shift0", "lag1_shift2", "lag2_shift0", "lag2_shift2"}
        assert lagged["l3_window"] is None  # shift 3 not run: no L-3 verdict
        assert (out / "REPORTS" / v5_stage0.LAG_MD).read_text().startswith("# v5 Stage 0 — check L")


class TestAmendment2:
    def test_row_gate_ignores_the_clip_mean(self) -> None:
        assert v5_stage0.passes_row_gate(np.array([0.96, 0.985, 0.99]))  # mean < 0.99: still ok
        assert not v5_stage0.passes_row_gate(np.array([0.999, 0.94]))

    def test_retry_redoes_only_the_named_reason(self, tmp_path: Path) -> None:
        clips = {v: {"dota_id": v} for v in ("a", "b", "c")}
        report = {"a": {"reason": "g1_pixel"}, "b": {"reason": "g2_aspect"}, "c": {"reason": "ok"}}
        ids, dev = ["a", "b", "c"], set[str]()

        def todo(retry: frozenset[str]) -> list[dict[str, Any]]:
            return v5_stage0.todo({"clips": clips}, ids, tmp_path, tmp_path, dev, report, retry)

        assert [c["dota_id"] for c in todo(frozenset())] == ["c"]  # c: ok but no grid file
        got = todo(frozenset({"g1_pixel"}))
        assert [c["dota_id"] for c in got] == ["a", "c"]


class TestCheckL:
    def test_lag_reads_skip_undefined_rows(self) -> None:
        steps = np.arange(0, 60, 3)
        labels = np.zeros(60, dtype=np.int64)
        labels[30:] = 1
        rng = np.random.default_rng(4)
        grid = rng.normal(size=(1, 7, 7, 8)) + 0.05 * rng.normal(size=(len(steps), 7, 7, 8))
        grid[10:, :, :3] = rng.normal(size=(10, 7, 3, 8))  # left tiles re-drawn from t=30
        boxes = _boxes(range(30, 60), (0.0, 0.0, 548.0, 720.0))  # ~3 of 7 columns
        for lag in (1, 6):
            read = v5_stage0.lag_read(grid, steps, labels, boxes, lag)
            assert read.effect is not None and read.effect > 0
            assert set(read.auc) == {"global", "max", "topk"}


class TestCentreShift:
    def test_shift_rows_moves_up_and_leaves_the_tail_undefined(self) -> None:
        out = v5_stage0.shift_rows(np.arange(5.0), 2)
        np.testing.assert_array_equal(out[:3], [2.0, 3.0, 4.0])
        assert np.isnan(out[3:]).all()

    def test_shift_recovers_a_late_window(self) -> None:
        """Evidence two rows after the box (a causal window catching up) is read at shift 2."""
        steps = np.arange(0, 90, 3)
        labels = np.zeros(90, dtype=np.int64)
        labels[45:60] = 1  # rows 15-19
        rng = np.random.default_rng(6)
        grid = rng.normal(size=(1, 7, 7, 8)) + 0.05 * rng.normal(size=(len(steps), 7, 7, 8))
        grid[17:22, :, :3] = rng.normal(size=(5, 7, 3, 8))  # left tiles move at rows 17-21
        boxes = _boxes(range(45, 60), (0.0, 0.0, 548.0, 720.0))
        late = v5_stage0.lag_read(grid, steps, labels, boxes, 1, 0)
        centred = v5_stage0.lag_read(grid, steps, labels, boxes, 1, 2)
        assert late.hit is not None and centred.hit is not None and centred.hit > late.hit
        assert centred.hit == 1.0

    def test_verdict_needs_both_shifts(self) -> None:
        def cells(gb: dict[int, bool]) -> dict[str, dict[str, Any]]:
            return {v5_stage0.cell_key(1, s): {"gates": {"G-B": ok}} for s, ok in gb.items()}

        assert (
            v5_stage0.centre_verdict(cells({0: False, 2: True, 3: True}), [1]) == "centred (lag 1)"
        )
        assert v5_stage0.centre_verdict(cells({0: False, 2: True, 3: False}), [1]) == "causal"
        assert v5_stage0.centre_verdict(cells({0: True, 2: True, 3: True}), [1]) == "causal"


class TestAmendment1:
    def test_growing_boxes_do_not_fake_a_pointing_lift(self) -> None:
        """Pure-noise change maps, boxes growing through the anomaly (S0-7')."""
        rng = np.random.default_rng(5)
        steps = np.arange(0, 90, 3)
        labels = np.zeros(90, dtype=np.int64)
        labels[30:] = 1
        boxes = {f: [[0.0, 0.0, 100.0 + 25.0 * (f - 30), 720.0]] for f in range(30, 90)}
        lifts: list[float] = []
        raw: list[float] = []
        for _ in range(200):
            read = v5_stage0.clip_read(
                rng.random((len(steps), 49)), rng.random(len(steps)), steps, labels, boxes, 5
            )
            lift = v5_stage0._vs_pre(read)
            assert lift is not None
            lifts.append(lift)
            assert read.hit is not None and read.hit_pre is not None
            raw.append(read.hit - read.hit_pre)
        assert abs(float(np.mean(lifts))) < 0.03
        assert float(np.mean(raw)) > 0.1  # the defect Amendment 1 removed
