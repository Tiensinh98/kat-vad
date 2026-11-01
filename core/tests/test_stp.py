"""LaGoVAD-STP: motion selection, SA², the text prefix, the trainer / eval wiring, the extractors
(proposal ``core/docs/final/`` §3.2, build notes ``core/docs/final/STP_BUILD.md``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel

from core import constants
from core.config import Config, load_config, validate_stp
from core.data.collate import collate_variable_length
from core.data.dataset import DVSFeatureDataset
from core.data.stp_patches import (
    build_manifest,
    check_patch_cache,
    eval_patches,
    motion_magnitude,
    select_top_k,
    select_video,
    stp_rows,
)
from core.inference import adopt_checkpoint_architecture
from core.models.clip_text import SoftPromptCLIPTextModel
from core.models.kat_vad import KATVAD
from core.models.sa2 import SpatialAttentionAggregation, attention_entropy
from core.models.text_encoding import make_text_encoder
from core.tests.fixtures import build_fixture
from core.tools import clip_windows, stp_patches, stp_readout
from core.train import Trainer, load_class_names, set_global_seed, stp_patch_dir

CPU = torch.device("cpu")
K, D = constants.STP_TOP_K, constants.STP_PATCH_DIM
MOTION = 384  # A2's motion columns
BASE = constants.HIDDEN_DIM + MOTION


def _arm(name: str, motion_dim: int = MOTION) -> Config:
    overrides = [
        "kip.enabled=false",
        f"model.motion_dim={motion_dim}",
        f"v2.motion={constants.V5_ENCODER if motion_dim else constants.V2_OFF}",
    ]
    overrides += [f"stp.{k}={v}" for k, v in constants.STP_ARMS[name].items()]
    return load_config(None, overrides)


def _patch_manifest(patch_dir: Path, stride: int = constants.FRAME_STRIDE) -> None:
    patch_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest("fx", stride, constants.STP_MOTION_RAW, K,
                              constants.STP_CELLS_RESIZED, "test")
    (patch_dir / constants.STP_PATCH_MANIFEST_FILENAME).write_text(json.dumps(manifest))


class TestSelection:
    def test_moving_patch_is_selected(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.standard_normal((1, constants.STP_PATCHES, 16)).astype(np.float32)
        patches = np.repeat(base, 5, axis=0)
        patches[2, 30] = rng.standard_normal(16)  # patch 30 changes at step 2 only
        mo = motion_magnitude(patches)
        assert mo.shape == (5, constants.STP_PATCHES)
        assert int(mo[2].argmax()) == 30
        assert mo[0].max() == pytest.approx(0.0, abs=1e-6)  # step 0 repeats its neighbour
        _, index = select_top_k(patches, mo, 3)
        assert index[2, 0] == 30

    def test_edges_repeat_their_neighbour(self) -> None:
        rng = np.random.default_rng(1)
        patches = rng.standard_normal((4, constants.STP_PATCHES, 8)).astype(np.float32)
        unit = patches / np.linalg.norm(patches, axis=-1, keepdims=True)
        mo = motion_magnitude(patches)
        np.testing.assert_allclose(mo[0], np.linalg.norm(unit[0] - unit[1], axis=-1), rtol=1e-5)
        np.testing.assert_allclose(mo[-1], np.linalg.norm(unit[-1] - unit[-2], axis=-1), rtol=1e-5)

    def test_ego_removes_a_frame_wide_change(self) -> None:
        rng = np.random.default_rng(2)
        patches = np.repeat(rng.standard_normal((1, constants.STP_PATCHES, 8)), 3, axis=0)
        # every patch moves the same way at step 1 (pure ego-motion, in unit space)
        unit = patches / np.linalg.norm(patches, axis=-1, keepdims=True)
        shift = np.zeros_like(unit)
        shift[1] = 0.1
        moved = unit + shift
        raw, ego = motion_magnitude(moved), motion_magnitude(moved, constants.STP_MOTION_EGO)
        assert ego[1].mean() < 0.5 * raw[1].mean()  # most of the shared change is removed

    def test_select_video_shapes_and_order(self) -> None:
        rng = np.random.default_rng(3)
        patches = rng.standard_normal((6, constants.STP_PATCHES, D)).astype(np.float32)
        out = select_video(patches, constants.STP_MOTION_RAW)
        assert out["features"].shape == (6, K * D) and out["features"].dtype == np.float16
        assert out["index"].shape == (6, K) and out["magnitude"].shape == (6, constants.STP_PATCHES)
        mo = out["magnitude"].astype(np.float32)
        picked = np.take_along_axis(mo, out["index"].astype(np.int64), axis=1)
        assert (np.diff(picked, axis=1) <= 1e-3).all()  # decreasing Mo
        first = out["features"][:, :D].astype(np.float32)
        np.testing.assert_allclose(
            first, patches[np.arange(6), out["index"][:, 0]], rtol=1e-2, atol=1e-2
        )


class TestSA2:
    def test_init_is_selection_plus_average(self) -> None:
        sa2 = SpatialAttentionAggregation(D, K)
        cls = torch.randn(2, 5, D)
        patches = torch.randn(2, 5, K * D)
        out = sa2(cls, patches)
        mean = patches.view(2, 5, K, D).mean(-2)
        torch.testing.assert_close(out.features, cls + mean)
        torch.testing.assert_close(out.attn, torch.full((2, 5, K), 1 / K))
        assert float(attention_entropy(out.attn)) == pytest.approx(1.0)
        assert float(sa2.query_norm()) == 0.0

    def test_w_q_gets_a_gradient_at_init(self) -> None:
        sa2 = SpatialAttentionAggregation(D, K)
        out = sa2(torch.randn(1, 4, D), torch.randn(1, 4, K * D))
        (out.features ** 2).sum().backward()
        assert sa2.w_q.weight.grad is not None and float(sa2.w_q.weight.grad.norm()) > 0

    def test_padding_stays_the_host_row(self) -> None:
        sa2 = SpatialAttentionAggregation(D, K)
        cls = torch.randn(1, 3, D)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        out = sa2(cls, torch.randn(1, 3, K * D), mask)
        torch.testing.assert_close(out.features[0, 2], cls[0, 2])

    def test_wrong_width_raises(self) -> None:
        with pytest.raises(ValueError, match="patch columns"):
            SpatialAttentionAggregation(D, K)(torch.randn(1, 2, D), torch.randn(1, 2, D))


class TestConfigAndModel:
    def test_arms_validate(self) -> None:
        for name in constants.STP_ARMS:
            validate_stp(_arm(name))

    @pytest.mark.parametrize(
        ("override", "match"),
        [
            (["stp.sa2=true", "v2.crn=R2"], "crn"),
            (["stp.text_prompt=coop"], "text_prompt"),
            (["stp.sa2=true", "stp.top_k=50"], "top_k"),
        ],
    )
    def test_bad_configs_raise(self, override: list[str], match: str) -> None:
        cfg = load_config(None, ["kip.enabled=false", *override])
        with pytest.raises(ValueError, match=match):
            validate_stp(cfg)

    def test_default_model_has_no_sa2(self) -> None:
        assert KATVAD.from_config(load_config(None, ["kip.enabled=false"])).sa2 is None

    def test_forward_at_init_is_host_plus_patch_mean(self) -> None:
        model = KATVAD.from_config(_arm("R4")).eval()
        assert model.sa2 is not None and model.motion_residual is not None
        x, u = torch.randn(1, 6, constants.HIDDEN_DIM), torch.randn(1, 6, MOTION)
        p = torch.randn(1, 6, K * D)
        lengths = torch.tensor([6])
        out = model(torch.cat([x, u, p], -1), lengths)
        host = KATVAD.from_config(load_config(None, [
            "kip.enabled=false", f"model.motion_dim={MOTION}", f"v2.motion={constants.V5_ENCODER}",
        ])).eval()
        host.load_state_dict(model.state_dict(), strict=False)
        ref = host(torch.cat([x + p.view(1, 6, K, D).mean(-2), u], -1), lengths)
        torch.testing.assert_close(out["vis_feats"], ref["vis_feats"])
        assert out["stp_attn"].shape == (1, 6, K)

    def test_missing_patch_columns_raise(self) -> None:
        model = KATVAD.from_config(_arm("R6"))
        with pytest.raises(ValueError, match="--patch-dir"):
            model(torch.randn(1, 4, BASE), torch.tensor([4]))

    def test_eval_adopts_the_stp_section(self) -> None:
        trained = _arm("R6")
        cfg = Config()
        adopt_checkpoint_architecture(cfg, {"config": trained.to_dict()}, [])
        assert cfg.stp.sa2 and cfg.stp.text_prompt == constants.STP_PROMPT_PREFIX


class TestPrefix:
    VOCAB, EOS_ID, BOS_ID = 100, 99, 98

    def _tiny(self, layout: str, n: int) -> SoftPromptCLIPTextModel:
        config = CLIPTextConfig(
            vocab_size=self.VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, max_position_embeddings=77, eos_token_id=2,
            bos_token_id=self.BOS_ID,
        )
        torch.manual_seed(0)
        return SoftPromptCLIPTextModel("unused", n, clip_model=CLIPTextModel(config), layout=layout)

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros(2, 10, dtype=torch.long)
        ids[0, :6] = torch.tensor([self.BOS_ID, 5, 9, 23, 42, self.EOS_ID])
        ids[1, :4] = torch.tensor([self.BOS_ID, 11, 3, self.EOS_ID])
        mask = (ids > 0).long()
        return ids, mask

    def test_prefix_layout_puts_every_prompt_before_the_text(self) -> None:
        model = self._tiny(constants.STP_PROMPT_PREFIX, constants.STP_PREFIX_LEN)
        ids, mask = self._inputs()
        hidden, expanded, new_mask = model._insert_soft_prompts(ids, mask)
        n = constants.STP_PREFIX_LEN
        assert (expanded[:, 1 : 1 + n] == -1).all()
        assert (expanded[:, 1 + n :] != -1).all()
        assert model.prompt_embedding is not None
        torch.testing.assert_close(hidden[0, 1 : 1 + n], model.prompt_embedding.weight)
        assert int(new_mask[0].sum()) == 6 + n and int(new_mask[1].sum()) == 4 + n
        assert int(expanded[0].argmax()) == 5 + n  # the text's own EOS is pooled

    def test_prefix_init_is_small(self) -> None:
        model = self._tiny(constants.STP_PROMPT_PREFIX, 512)
        assert model.prompt_embedding is not None
        std = float(model.prompt_embedding.weight.std())
        assert std == pytest.approx(constants.STP_PREFIX_INIT_STD, rel=0.1)

    def test_prefix_learns(self) -> None:
        model = self._tiny(constants.STP_PROMPT_PREFIX, 4)
        ids, mask = self._inputs()
        model(ids, mask).pooler_output.sum().backward()
        assert model.prompt_embedding is not None and model.prompt_embedding.weight.grad is not None

    def test_released_layout_is_unchanged(self) -> None:
        model = self._tiny(constants.STP_PROMPT_RELEASED, 4)
        ids, mask = self._inputs()
        _, expanded, _ = model._insert_soft_prompts(ids, mask)
        assert int((expanded[0] == -1).sum()) == 4 and (expanded[0, 1:3] == -1).all()

    def test_unknown_layout_raises(self) -> None:
        with pytest.raises(ValueError, match="layout"):
            self._tiny("coop", 4)


def _stp_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """The MSAD fixture with A2-wide rows and a patch cache at the fixture stride."""
    fixture = build_fixture(tmp_path / "fx")
    rng = np.random.default_rng(1)
    a2_dir, patch_dir = tmp_path / "a2", tmp_path / "patches"
    a2_dir.mkdir()
    _patch_manifest(patch_dir)
    for path in fixture.clip_dir.glob("*.npy"):
        x = np.load(path)
        np.save(a2_dir / path.name,
                np.concatenate([x, rng.standard_normal((len(x), MOTION))], 1).astype(np.float32))
        np.save(patch_dir / path.name, rng.standard_normal((len(x), K * D)).astype(np.float16))
    (a2_dir / constants.V2_INPUT_MANIFEST_FILENAME).write_text(
        json.dumps({"crn": constants.V2_OFF, "motion": constants.V5_ENCODER})
    )
    return fixture.data_dir, a2_dir, patch_dir


class TestTraining:
    @pytest.mark.parametrize("arm", ["R4", "R6"])
    def test_losses_and_diagnostics(self, tmp_path: Path, arm: str) -> None:
        data_dir, a2_dir, patch_dir = _stp_fixture(tmp_path)
        cfg = _arm(arm)
        cfg.train.batch_size, cfg.train.num_epochs = 4, 1
        set_global_seed(cfg.train.seed)
        dataset = DVSFeatureDataset(
            data_dir=data_dir, clip_dir=a2_dir, dvs=cfg.dvs, require_flow=False,
            seed=cfg.train.seed, patch_dir=stp_patch_dir(cfg, patch_dir),
        )
        assert dataset[0]["v_feat"].shape[-1] == BASE + K * D
        model = KATVAD.from_config(cfg)
        trainer = Trainer(
            cfg=cfg, model=model, dataset=dataset, class_names=load_class_names(data_dir),
            text_encode_fn=make_text_encoder(model, "stub", CPU, dim=constants.HIDDEN_DIM),
            output_dir=tmp_path / "out", device=CPU,
        )
        losses = trainer.compute_losses(collate_variable_length([dataset[i] for i in range(4)]))
        assert torch.isfinite(losses["total"])
        assert float(losses["stp_entropy"]) == pytest.approx(1.0, abs=1e-5)
        assert float(losses["stp_wq_norm"]) == 0.0
        losses["total"].backward()
        assert model.sa2 is not None and model.sa2.w_q.weight.grad is not None

    def test_patch_dir_checks(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="would be ignored"):
            stp_patch_dir(Config(), tmp_path)
        with pytest.raises(ValueError, match="needs --patch-dir"):
            stp_patch_dir(_arm("R4"), None)
        _patch_manifest(tmp_path / "s3", stride=3)
        with pytest.raises(ValueError, match="this run needs"):
            stp_patch_dir(_arm("R4"), tmp_path / "s3")
        assert stp_patch_dir(_arm("R5"), None) is None  # prefix-only arm has no patches

    def test_rows_must_align(self) -> None:
        with pytest.raises(ValueError, match="C13"):
            stp_rows(np.zeros((3, BASE)), np.zeros((4, K * D)))

    def test_eval_patches_refuse_mismatch(self, tmp_path: Path) -> None:
        assert eval_patches(False, "raw", K, None, 3) is None
        with pytest.raises(ValueError, match=r"stp\.sa2=false"):
            eval_patches(False, "raw", K, tmp_path, 3)
        with pytest.raises(ValueError, match="--patch-dir"):
            eval_patches(True, "raw", K, None, 3)
        _patch_manifest(tmp_path, stride=3)
        assert eval_patches(True, "raw", K, tmp_path, 3) is not None
        with pytest.raises(ValueError, match="this run needs"):
            check_patch_cache(tmp_path, 3, constants.STP_MOTION_EGO, K)


class FakeEncoder(torch.nn.Module):
    """CLIP stand-in: the per-channel mean of each normalized cell, padded to 512."""

    def forward(self, pixel_values: torch.Tensor) -> object:
        feats = pixel_values.mean(dim=(-1, -2))
        embeds = torch.nn.functional.pad(feats, (0, D - feats.shape[-1]))

        class Out:
            image_embeds = embeds

        return Out()


class TestExtractors:
    def test_resized_cells_global_window_is_the_cls_input(self) -> None:
        torch.manual_seed(0)
        frames = torch.rand(2, 3, 90, 160)
        windows = clip_windows.encode_frame_windows(
            frames, FakeEncoder(), constants.STP_WINDOW_GRIDS, constants.STP_CELLS_RESIZED
        )
        assert windows.shape == (2, 1 + constants.STP_PATCHES, D)
        squashed = clip_windows.squash(frames)
        mean = torch.tensor(constants.CLIP_IMAGE_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(constants.CLIP_IMAGE_STD).view(1, 3, 1, 1)
        cls = ((squashed - mean) / std).mean(dim=(-1, -2)).numpy()
        np.testing.assert_allclose(windows[:, 0, :3], cls, rtol=1e-5, atol=1e-5)
        boxes = clip_windows.window_boxes(constants.CROP_SIZE, constants.CROP_SIZE, (7,))
        assert all(b - t == 32 and r - left == 32 for t, b, left, r in boxes)

    def test_probe_manifest_and_dir_are_unchanged(self) -> None:
        manifest = clip_windows.build_manifest("DADA", 8, "src")
        assert "cells" not in manifest and manifest["grids"] == list(constants.TW_GRIDS)
        assert clip_windows.cache_dir("DADA", 8).name == "DADA_s8_g1235_ncc"
        assert clip_windows.cache_dir(
            "DoTA_CAP", 3, constants.STP_WINDOW_GRIDS, constants.STP_CELLS_RESIZED
        ).name == "DoTA_CAP_s3_g17_resized224_ncc"

    def test_decode_strided_matches_full_decode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rng = np.random.default_rng(0)
        video = rng.integers(0, 255, (21, 40, 64, 3), dtype=np.uint8)
        monkeypatch.setattr(stp_patches, "iter_frame_chunks",
                            lambda _p: iter([video[:8], video[8:16], video[16:]]))
        got = stp_patches.decode_strided(tmp_path / "v.mp4", 8)
        want = stp_patches.resize_frames(video)[::8]
        torch.testing.assert_close(got, want)

    def test_cls_gate(self) -> None:
        rng = np.random.default_rng(0)
        windows = rng.standard_normal((4, 50, D)).astype(np.float32)
        assert stp_patches.cls_gate(windows, windows[:, 0])["reason"] == stp_patches.OK
        assert stp_patches.cls_gate(windows, windows[:3, 0])["reason"] == stp_patches.ROWS_MISMATCH
        assert stp_patches.cls_gate(windows, -windows[:, 0])["reason"] == stp_patches.GATE_FAILED

    def test_select_subcommand(self, tmp_path: Path) -> None:
        win = tmp_path / "win"
        win.mkdir()
        manifest = clip_windows.build_manifest(
            "DoTA_CAP", 3, "src", constants.STP_WINDOW_GRIDS, constants.STP_CELLS_RESIZED
        )
        (win / constants.VIDEO_MANIFEST_FILENAME).write_text(json.dumps(manifest))
        rng = np.random.default_rng(0)
        np.save(win / "abc_000001.npy", rng.standard_normal((5, 50 * D)).astype(np.float16))
        out = tmp_path / "sel"
        stp_patches.main(["select", "--windows-dir", str(win), "--out-dir", str(out)])
        assert np.load(out / "abc_000001.npy").shape == (5, K * D)
        assert np.load(out / constants.STP_INDEX_SUBDIR / "abc_000001.npy").shape == (5, K)
        check_patch_cache(out, 3, constants.STP_MOTION_RAW, K)


class TestReadout:
    def test_sentences(self) -> None:
        assert stp_readout.sentence({"mean": 0.02, "low": 0.001, "high": 0.04}) == stp_readout.GAIN
        assert stp_readout.sentence({"mean": 0.005, "low": 0.001, "high": 0.01}) == stp_readout.NONE
        cost = {"mean": -0.02, "low": -0.04, "high": -0.001}
        assert stp_readout.sentence(cost) == stp_readout.COST
        assert stp_readout.sentence(None) == stp_readout.NONE

    def test_paired_uses_common_clips(self) -> None:
        delta = stp_readout.paired({"a_1": 0.7, "b_2": 0.6}, {"a_1": 0.5, "c_3": 0.1})
        assert delta == {"a_1": pytest.approx(0.2)}

    def test_entropy_of_uniform_is_one(self) -> None:
        attn = torch.full((1, 3, K), 1 / K)
        assert float(attention_entropy(attn)) == pytest.approx(1.0)
        peaked = torch.zeros(1, 3, K)
        peaked[..., 0] = 1
        assert float(attention_entropy(peaked)) < 1e-4 * math.log(K) + 1e-4


class TestNexarExtractEndToEnd:
    def test_nexar_subcommand(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        video = rng.integers(0, 255, (20, 36, 64, 3), dtype=np.uint8)
        stride = 4
        resized = stp_patches.resize_frames(video)
        encoder = FakeEncoder()
        cls = clip_windows.encode_frame_windows(
            resized, encoder, (1,), constants.STP_CELLS_RESIZED
        )[:, 0]
        clip_s1 = tmp_path / "clip_s1"
        clip_s1.mkdir()
        np.save(clip_s1 / "v1.npy", np.repeat(cls, 1, axis=0))  # every native frame
        census = tmp_path / "census.json"
        census.write_text(json.dumps({"videos": {"v1": {"path": "v1.mp4"}}}))
        monkeypatch.setattr(stp_patches, "video_paths", lambda *_: {"v1": tmp_path / "v1.mp4"})
        monkeypatch.setattr(stp_patches, "iter_frame_chunks", lambda _p: iter([video]))
        monkeypatch.setattr(stp_patches, "load_pretrained_encoder", lambda _d: encoder)
        out, report = tmp_path / "patches", tmp_path / "report.json"
        ids = tmp_path / "ids.txt"
        ids.write_text("v1\n")
        stp_patches.main([
            "nexar", "--root", str(tmp_path), "--census", str(census), "--ids-file", str(ids),
            "--clip-s1-dir", str(clip_s1), "--stride", str(stride), "--out-dir", str(out),
            "--report", str(report), "--device", "cpu",
        ])
        rows = len(range(0, 20, stride))
        assert np.load(out / "v1.npy").shape == (rows, K * D)
        assert np.load(out / constants.STP_INDEX_SUBDIR / "v1.npy").shape == (rows, K)
        summary = json.loads(report.read_text())["summary"]
        assert summary["pass"] and summary["min_cls_cos"] > 0.9999
        check_patch_cache(out, stride, constants.STP_MOTION_RAW, K)


def _jpegs(folder: Path, n: int, seed: int = 0) -> list[Path]:
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        path = folder / f"{i:06d}.jpg"
        Image.fromarray(rng.integers(0, 255, (24, 40, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


class TestFrameArchive:
    def _clip(self, frame_map: list[int], cap_frames: int) -> dict[str, object]:
        return {"dota_id": "abc_000010", "cap_id": "000001", "reason": "ok",
                "frame_map": frame_map, "dota_frames": len(frame_map), "cap_frames": cap_frames}

    def test_pack_keeps_original_bytes_in_dota_order(self, tmp_path: Path) -> None:
        from core.tools import dota_cap_frames as dcf

        cap = _jpegs(tmp_path / "cap" / "000001" / "images", 9)
        frame_map = [0, 3, 6, 8]
        packer = dcf.Packer({}, tmp_path / "arc", "images", {}, lambda: None)
        entry = packer.one_clip(self._clip(frame_map, 9), cap)
        assert entry["reason"] == "ok" and entry["frames"] == 4
        zpath = dcf.archive_path(tmp_path / "arc", "abc_000010")
        assert dcf.archive_frames(zpath) == 4
        folder = dcf.unpack(zpath, tmp_path / "out")
        got = sorted(folder.iterdir())
        assert [p.name for p in got] == [dcf.entry_name(d) for d in range(4)]
        pairs = zip(got, frame_map, strict=True)
        assert all(g.read_bytes() == cap[j].read_bytes() for g, j in pairs)
        assert packer.one_clip(self._clip(frame_map, 10), cap)["reason"] == dcf.CAP_FRAMES_MISMATCH

    def test_torn_archive_is_not_done(self, tmp_path: Path) -> None:
        from core.tools import dota_cap_frames as dcf

        bad = tmp_path / "x.zip"
        bad.write_bytes(b"PK\x03\x04 torn")
        assert dcf.archive_frames(bad) is None
        assert dcf.archive_frames(tmp_path / "absent.zip") is None

    def test_verify_and_archive_windows(self, tmp_path: Path) -> None:
        from core.tools import dota_cap_frames as dcf

        cap = _jpegs(tmp_path / "cap", 6)
        clip = self._clip([0, 1, 2, 3, 4, 5], 6)
        dcf.Packer({}, tmp_path / "arc", "images", {}, lambda: None).one_clip(clip, cap)
        alignment = tmp_path / "align.json"
        alignment.write_text(json.dumps({"clips": {"abc_000010": clip}}))
        s1 = tmp_path / "s1"
        s1.mkdir()
        # the CLS rows the fake encoder gives on the squashed frames (the 1 x 1 cell)
        from core.data.video_io import read_images

        frames = torch.from_numpy(read_images(cap)).permute(0, 3, 1, 2).float() / 255.0
        cls = clip_windows.encode_frame_windows(frames, FakeEncoder(), (1,), "resized224")[:, 0]
        np.save(s1 / "abc_000010.npy", cls)
        dcf.main(["verify", "--alignment", str(alignment), "--archive-dir", str(tmp_path / "arc"),
                  "--dota-clip-dir", str(s1), "--report", str(tmp_path / "verify.json")])
        assert (tmp_path / "arc" / constants.DOTA_CAP_FRAMES_MANIFEST).is_file()
        ids = tmp_path / "ids.txt"
        ids.write_text("abc_000010\n")
        args = clip_windows.build_arg_parser().parse_args([
            "archive", "--archive-dir", str(tmp_path / "arc"), "--work-dir", str(tmp_path / "w"),
            "--dota-clip-dir", str(s1), "--dota-ids-file", str(ids),
            "--report", str(tmp_path / "rep.json"), "--grids", "1", "7",
            "--cells", "resized224", "--output-dir", str(tmp_path / "win"), "--device", "cpu",
        ])
        clip_windows.run_archive(args, FakeEncoder())
        rep = json.loads((tmp_path / "rep.json").read_text())["abc_000010"]
        assert rep["reason"] == "ok", rep
        assert np.load(tmp_path / "win" / "abc_000010.npy").shape == (2, 50 * D)  # 6 frames, s3
        assert not (tmp_path / "w" / "abc_000010").exists()

    def test_verify_flags_a_missing_clip(self, tmp_path: Path) -> None:
        from core.tools import dota_cap_frames as dcf

        alignment = tmp_path / "align.json"
        alignment.write_text(json.dumps({"clips": {"abc_000010": self._clip([0], 1)}}))
        with pytest.raises(SystemExit):
            dcf.main(["verify", "--alignment", str(alignment), "--archive-dir", str(tmp_path),
                      "--dota-clip-dir", str(tmp_path), "--report", str(tmp_path / "v.json")])
        assert not (tmp_path / constants.DOTA_CAP_FRAMES_MANIFEST).exists()
