"""v2 model inputs (plan P5): config, MotionResidual, baked CRN/motion rows, manifests.

CPU-only, data-free. The end-to-end class trains A3 through the real CLI on the
synthetic fixture with a baked input cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants, evaluate, train
from core.config import Config, load_config, validate_v2
from core.data import v2_inputs
from core.inference import adopt_checkpoint_architecture
from core.models.kat_vad import KATVAD
from core.models.motion_residual import MotionResidual
from core.tests.fixtures import FixtureLayout, build_fixture
from core.tools import build_v2_inputs

torch.set_num_threads(1)
ENCODER_S = constants.VIDEOMAE_ENCODER_S
D_S = constants.VIDEOMAE_ARCH[ENCODER_S][0]  # 384
SMALL = 32  # hidden width for the unit-level models


def _v2_config(crn: str = constants.V2_OFF, motion: str = constants.V2_OFF) -> Config:
    overrides = ["kip.enabled=false", f"v2.crn={crn}", f"v2.motion={motion}"]
    if motion != constants.V2_OFF:
        overrides.append(f"model.motion_dim={constants.VIDEOMAE_ARCH[motion][0]}")
    return load_config(None, overrides)


class TestConfig:
    def test_defaults_are_v1(self) -> None:
        cfg = Config()
        assert (cfg.v2.crn, cfg.v2.motion, cfg.model.motion_dim) == ("none", "none", 0)
        validate_v2(cfg)  # a v1 KIP-on default config stays valid

    @pytest.mark.parametrize(
        ("crn", "motion"), [("R1", "none"), ("none", ENCODER_S), ("R4", ENCODER_S)]
    )
    def test_every_arm_validates(self, crn: str, motion: str) -> None:
        validate_v2(_v2_config(crn, motion))

    def test_motion_without_dim_raises(self) -> None:
        cfg = load_config(None, ["kip.enabled=false", f"v2.motion={ENCODER_S}"])
        with pytest.raises(ValueError, match="motion_dim > 0"):
            validate_v2(cfg)

    def test_dim_must_match_encoder(self) -> None:
        cfg = _v2_config(motion=ENCODER_S)
        cfg.model.motion_dim = 768
        with pytest.raises(ValueError, match="emits 384"):
            validate_v2(cfg)

    def test_kip_with_v2_raises(self) -> None:
        cfg = load_config(None, ["v2.crn=R1"])
        with pytest.raises(ValueError, match=r"kip\.enabled=false"):
            validate_v2(cfg)

    def test_unknown_reference_raises(self) -> None:
        with pytest.raises(ValueError, match=r"v2\.crn"):
            validate_v2(load_config(None, ["kip.enabled=false", "v2.crn=R9"]))


class TestMotionResidual:
    def test_zero_init_is_identity(self) -> None:
        block = MotionResidual(SMALL, 6)
        rows = torch.randn(2, 5, SMALL + 6)
        h, rho = block(rows)
        assert torch.equal(h, rows[..., :SMALL])
        assert float(rho) == 0.0

    def test_wrong_width_raises(self) -> None:
        with pytest.raises(ValueError, match="v2 input cache"):
            MotionResidual(SMALL, 6)(torch.randn(1, 3, SMALL))

    def test_padded_rows_get_no_motion_and_rho_is_share(self) -> None:
        block = MotionResidual(SMALL, 6)
        torch.nn.init.normal_(block.proj.weight)
        torch.nn.init.normal_(block.proj.bias)
        rows = torch.randn(1, 4, SMALL + 6)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # padding_mask convention: 1 = valid
        h, rho = block(rows, mask)
        assert torch.equal(h[0, 2:], rows[0, 2:, :SMALL])
        motion = block.proj(rows[0, :2, SMALL:])
        expected = motion.norm() / rows[0, :2, :SMALL].norm()
        assert float(rho) == pytest.approx(float(expected), rel=1e-5)
        assert float(block.weight_norm()) > 0


def _small_model(motion_dim: int) -> KATVAD:
    torch.manual_seed(0)
    return KATVAD(hidden_dim=SMALL, temporal_heads=4, fusion_heads=4, motion_dim=motion_dim).eval()


class TestModel:
    def test_a0_has_no_motion_block(self) -> None:
        assert _small_model(0).motion_residual is None

    def test_zero_w_u_gives_the_no_motion_model(self) -> None:
        """W_u = 0 at init => A2/A3 output == the no-motion model on the same x (arch. §5)."""
        a0, a2 = _small_model(0), _small_model(6)
        missing, unexpected = a2.load_state_dict(a0.state_dict(), strict=False)
        assert unexpected == [] and all(k.startswith("motion_residual.") for k in missing)
        x = torch.randn(2, 7, SMALL)
        rows = torch.cat([x, torch.randn(2, 7, 6)], dim=-1)
        lengths = torch.tensor([7, 5])
        text = torch.randn(3, SMALL)
        with torch.no_grad():
            out0, out2 = a0(x, lengths, class_feats=text), a2(rows, lengths, class_feats=text)
        torch.testing.assert_close(out2["cls_bin_logits"], out0["cls_bin_logits"])
        torch.testing.assert_close(out2["cls_sim_mat"], out0["cls_sim_mat"])
        assert "motion_share" in out2 and "motion_share" not in out0

    def test_a0_state_into_motion_model_strict_raises(self) -> None:
        """A2/A3 missing W_u must fail loud (lesson C5)."""
        with pytest.raises(RuntimeError, match="motion_residual"):
            _small_model(6).load_state_dict(_small_model(0).state_dict())

    def test_old_checkpoint_config_adopts_as_a0(self) -> None:
        cfg = _v2_config()
        old = Config().to_dict()
        del old["v2"]
        del old["model"]["motion_dim"]
        adopt_checkpoint_architecture(cfg, {"config": old}, [])
        assert (cfg.model.motion_dim, cfg.v2.crn, cfg.v2.motion) == (0, "none", "none")

    def test_v2_checkpoint_config_is_adopted(self) -> None:
        cfg = Config()
        stored = _v2_config("R1", ENCODER_S).to_dict()
        adopt_checkpoint_architecture(cfg, {"config": stored}, [])
        assert (cfg.model.motion_dim, cfg.v2.crn, cfg.v2.motion) == (D_S, "R1", ENCODER_S)


def _sources(rng: np.random.Generator, n: int = 6, dim: int = 8, d_u: int = 5) -> tuple[dict, dict]:
    clips = {f"v{i}": rng.normal(2.0, 1.0, (int(rng.integers(6, 15)), dim)) for i in range(n)}
    motions = {v: rng.normal(-1.0, 3.0, (len(x), d_u)) for v, x in clips.items()}
    return clips, motions


class TestStats:
    def test_s_preserves_mean_norm_on_train(self) -> None:
        clips, _ = _sources(np.random.default_rng(0))
        stats = v2_inputs.fit_stats(clips, None, "R1", constants.V2_OFF)
        baked = np.concatenate([v2_inputs.bake_rows(x, None, stats) for x in clips.values()])
        raw = np.concatenate(list(clips.values()))
        assert np.linalg.norm(baked, axis=1).mean() == pytest.approx(
            np.linalg.norm(raw, axis=1).mean(), rel=1e-5
        )
        assert stats.c == pytest.approx(np.linalg.norm(raw, axis=1).mean() / np.sqrt(8))

    @pytest.mark.parametrize("crn", ["none", "R1", "R4"])
    def test_motion_columns_have_rms_c_per_channel(self, crn: str) -> None:
        clips, motions = _sources(np.random.default_rng(1))
        stats = v2_inputs.fit_stats(clips, motions, crn, ENCODER_S)
        baked = np.concatenate(
            [v2_inputs.bake_rows(clips[v], motions[v], stats) for v in sorted(clips)]
        )
        motion_cols = baked[:, 8:]
        np.testing.assert_allclose(motion_cols.std(axis=0), stats.c, rtol=1e-4)
        if crn == "none":  # A2 centres on the train channel mean
            np.testing.assert_allclose(motion_cols.mean(axis=0), 0.0, atol=1e-4)

    def test_r4_rows_never_see_the_future(self) -> None:
        clips, motions = _sources(np.random.default_rng(2))
        stats = v2_inputs.fit_stats(clips, motions, "R4", ENCODER_S)
        x, u = clips["v0"], motions["v0"]
        full = v2_inputs.bake_rows(x, u, stats)
        cut = constants.V2_CRN_WARMUP_STEPS + 2
        prefix = v2_inputs.bake_rows(x[:cut], u[:cut], stats)
        np.testing.assert_allclose(full[:cut], prefix, rtol=1e-6)

    def test_length_mismatch_raises(self) -> None:
        clips, motions = _sources(np.random.default_rng(3))
        stats = v2_inputs.fit_stats(clips, motions, "R1", ENCODER_S)
        with pytest.raises(ValueError, match="C13"):
            v2_inputs.bake_rows(clips["v0"], motions["v0"][:-1], stats)


class TestManifest:
    def test_plain_cache_passes_only_for_a0(self, tmp_path: Path) -> None:
        v2_inputs.check_input_manifest(tmp_path, "none", "none")
        with pytest.raises(ValueError, match="plain CLIP cache"):
            v2_inputs.check_input_manifest(tmp_path, "R1", "none")

    def test_mismatch_and_a0_on_v2_cache_raise(self, tmp_path: Path) -> None:
        (tmp_path / constants.V2_INPUT_MANIFEST_FILENAME).write_text(
            json.dumps({"crn": "R1", "motion": "none"})
        )
        v2_inputs.check_input_manifest(tmp_path, "R1", "none")
        with pytest.raises(ValueError, match="baked for crn='R1'"):
            v2_inputs.check_input_manifest(tmp_path, "R2", "none")
        with pytest.raises(ValueError, match="baked for"):
            v2_inputs.check_input_manifest(tmp_path, "none", "none")

    def test_arm_names(self) -> None:
        assert build_v2_inputs.arm_name("R1", "none") == "A1"
        assert build_v2_inputs.arm_name("none", ENCODER_S) == "A2"
        assert build_v2_inputs.arm_name("R1", ENCODER_S) == "A3"
        with pytest.raises(ValueError, match="A0"):
            build_v2_inputs.arm_name("none", "none")


# ---------------------------------------------------------------------------
# end to end: bake an A3 cache on the synthetic fixture, train, evaluate
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> FixtureLayout:
    return build_fixture(tmp_path_factory.mktemp("v2e2e"))


@pytest.fixture(scope="module")
def a3_cache(fixture: FixtureLayout, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Bake A3 (R1 + VideoMAE-S-shaped random motion) for every fixture video."""
    rng = np.random.default_rng(7)
    ids = sorted(p.stem for p in fixture.clip_dir.glob("*.npy"))
    clips = {v: np.load(fixture.clip_dir / f"{v}.npy") for v in ids}
    motions = {v: rng.normal(0.0, 1.0, (len(x), D_S)).astype(np.float32) for v, x in clips.items()}
    stats = v2_inputs.fit_stats({v: clips[v] for v in fixture.train_ids}, motions, "R1", ENCODER_S)
    out = tmp_path_factory.mktemp("a3_cache")
    width = build_v2_inputs.bake(ids, clips, motions, stats, out)
    v2_inputs.save_stats(out, stats)
    manifest = {"arm": "A3", "crn": "R1", "motion": ENCODER_S, "width": width}
    (out / constants.V2_INPUT_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return out


def _args(fixture: FixtureLayout, clip_dir: Path, out: Path, v2: bool) -> list[str]:
    args = [
        "--data-dir", str(fixture.data_dir), "--clip-dir", str(clip_dir),
        "--knn-cache", str(fixture.knn_cache_path), "--output-dir", str(out),
        "--text-encoder", "stub",
        "--set", "train.num_epochs=2", "--set", "train.batch_size=4",
        "--set", "train.device=cpu", "--set", "kip.enabled=false",
    ]
    if v2:
        args += ["--set", "v2.crn=R1", "--set", f"v2.motion={ENCODER_S}",
                 "--set", f"model.motion_dim={D_S}"]
    return args


class TestEndToEnd:
    def test_a3_trains_logs_motion_and_evaluates(
        self, fixture: FixtureLayout, a3_cache: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "a3"
        train.main(_args(fixture, a3_cache, out, v2=True))
        lines = (out / train.METRICS_FILENAME).read_text().splitlines()
        records = [json.loads(line) for line in lines]
        assert all("motion_share" in r and "w_u_norm" in r for r in records)
        assert records[-1]["w_u_norm"] > 0.0, "W_u never left zero"
        assert all(np.isfinite(r["total"]) for r in records)
        payload = torch.load(out / train.CHECKPOINT_LAST, map_location="cpu", weights_only=False)
        assert "motion_residual.proj.weight" in payload["model"]

        # evaluate adopts v2 from the checkpoint: the baked cache scores, the plain one is refused
        eval_args = [
            "--ckpt", str(out / train.CHECKPOINT_LAST), "--data-dir", str(fixture.data_dir),
            "--text-encoder", "stub", "--set", "train.device=cpu",
        ]
        evaluate.main(
            [*eval_args, "--clip-dir", str(a3_cache), "--output-dir", str(tmp_path / "ev")]
        )
        assert (tmp_path / "ev" / evaluate.RESULTS_FILENAME).exists()
        with pytest.raises(ValueError, match="plain CLIP cache"):
            evaluate.main([*eval_args, "--clip-dir", str(fixture.clip_dir),
                           "--output-dir", str(tmp_path / "ev_bad")])

    def test_v2_config_on_plain_cache_refuses_to_train(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="plain CLIP cache"):
            train.main(_args(fixture, fixture.clip_dir, tmp_path / "bad", v2=True))

    def test_a0_config_on_v2_cache_refuses_to_train(
        self, fixture: FixtureLayout, a3_cache: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="baked for"):
            train.main(_args(fixture, a3_cache, tmp_path / "bad0", v2=False))


class TestApplyStride:
    def test_apply_subsamples_then_references_the_subsampled_clip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(11)
        fit_dir, clip_dir, out = tmp_path / "fit", tmp_path / "s1", tmp_path / "out"
        fit_dir.mkdir()
        clip_dir.mkdir()
        stats = v2_inputs.fit_stats({"t": rng.normal(size=(30, 8))}, None, "R1", constants.V2_OFF)
        v2_inputs.save_stats(fit_dir, stats)
        (fit_dir / constants.V2_INPUT_MANIFEST_FILENAME).write_text(json.dumps(
            {"crn": "R1", "motion": "none", "fitted_on": "T2", "train_ids_sha1": "x"}
        ))
        clip = rng.normal(size=(25, 8)).astype(np.float32)
        np.save(clip_dir / "c1.npy", clip)
        build_v2_inputs.main(["apply", "--stats-dir", str(fit_dir), "--clip-dir", str(clip_dir),
                              "--stride", "3", "--out-dir", str(out)])
        baked = np.load(out / "c1.npy")
        sub = clip[::3].astype(np.float64)
        np.testing.assert_allclose(baked, stats.s * (sub - sub.mean(axis=0)), rtol=1e-5, atol=1e-6)
        manifest = json.loads((out / constants.V2_INPUT_MANIFEST_FILENAME).read_text())
        assert manifest["stride_over_clip_dir"] == 3 and manifest["reference_unit"] == "clip"
        v2_inputs.check_input_manifest(out, "R1", "none")
