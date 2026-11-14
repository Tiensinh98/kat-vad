"""Arm K5ln (core/docs/final/KIM_RUN.md §25): per-step LayerNorm on the motion columns.

CPU-only, data-free. The default must leave every earlier arm's build, state dict and output
untouched; the LN arm must start where K5 starts (W_u = 0), refuse an unset scale, and carry
the cache's ``c`` through training into its checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants, train
from core.config import load_config, validate_v2
from core.data import v2_inputs
from core.models.kat_vad import KATVAD
from core.models.motion_residual import MotionResidual
from core.tests.fixtures import FixtureLayout
from core.tests.test_v2_model import (  # noqa: F401  (module-scoped pytest fixtures)
    D_S,
    ENCODER_S,
    fixture,
)
from core.tools import build_v2_inputs

torch.set_num_threads(1)
SMALL = 32
MOTION_DIM = 6
LN = constants.MOTION_NORM_LAYERNORM
C_VALUE = 0.37


def _model(norm: str = constants.MOTION_NORM_OFF) -> KATVAD:
    torch.manual_seed(0)
    return KATVAD(
        hidden_dim=SMALL, temporal_heads=4, fusion_heads=4, motion_dim=MOTION_DIM,
        motion_norm=norm,
    ).eval()


def _rows(seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(2, 7, SMALL + MOTION_DIM, generator=gen)


def _motion_cfg(*overrides: str) -> list[str]:
    return ["kip.enabled=false", f"v2.motion={ENCODER_S}", f"model.motion_dim={D_S}", *overrides]


class TestConfig:
    def test_default_is_off(self) -> None:
        assert load_config(None, []).model.motion_norm == constants.MOTION_NORM_OFF

    def test_layernorm_validates_on_a_motion_host(self) -> None:
        validate_v2(load_config(None, _motion_cfg(f"model.motion_norm={LN}")))

    def test_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError, match="motion_norm must be one of"):
            validate_v2(load_config(None, _motion_cfg("model.motion_norm=batchnorm")))

    def test_without_a_motion_stream_raises(self) -> None:
        cfg = load_config(None, ["kip.enabled=false", f"model.motion_norm={LN}"])
        with pytest.raises(ValueError, match="motion_norm needs a motion stream"):
            validate_v2(cfg)


class TestBlock:
    def test_off_has_no_new_state(self) -> None:
        """Every earlier checkpoint (norm off) has exactly the keys it had before (C5)."""
        assert set(MotionResidual(16, 8).state_dict()) == {"proj.weight", "proj.bias"}

    def test_layernorm_adds_only_the_scale_buffer(self) -> None:
        keys = set(MotionResidual(16, 8, norm=LN).state_dict())
        assert keys == {"proj.weight", "proj.bias", "norm_scale"}

    def test_off_passes_u_through(self) -> None:
        u = torch.randn(2, 5, 8)
        assert torch.equal(MotionResidual(16, 8).normalize(u), u)

    def test_unset_scale_refuses_to_run(self) -> None:
        block = MotionResidual(16, 8, norm=LN)
        with pytest.raises(RuntimeError, match="norm_scale was never set"):
            block(torch.randn(1, 3, 24))

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_scale_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            MotionResidual(16, 8, norm=LN).set_norm_scale(bad)

    def test_set_scale_on_the_off_block_raises(self) -> None:
        with pytest.raises(ValueError, match="only defined"):
            MotionResidual(16, 8).set_norm_scale(1.0)

    def test_each_step_is_centred_with_rms_c(self) -> None:
        block = MotionResidual(16, 64, norm=LN)
        block.set_norm_scale(C_VALUE)
        out = block.normalize(torch.randn(3, 9, 64) * 5.0 + 2.0)
        torch.testing.assert_close(out.mean(-1), torch.zeros(3, 9), atol=1e-5, rtol=0)
        rms = out.pow(2).mean(-1).sqrt()
        torch.testing.assert_close(rms, torch.full((3, 9), C_VALUE), atol=1e-4, rtol=0)

    def test_the_step_magnitude_is_erased(self) -> None:
        """The hypothesis under test: LN removes ||u_t||, so a scaled step maps to the same row."""
        block = MotionResidual(16, 8, norm=LN)
        block.set_norm_scale(1.0)
        u = torch.randn(1, 1, 8)
        # equal up to LayerNorm's eps, which weighs on the smaller copy only
        torch.testing.assert_close(block.normalize(u), block.normalize(4.0 * u), atol=1e-4, rtol=0)

    def test_padding_rows_stay_zero(self) -> None:
        block = MotionResidual(4, 6, norm=LN)
        block.set_norm_scale(C_VALUE)
        with torch.no_grad():
            block.proj.weight.normal_()
            block.proj.bias.normal_()
        rows = torch.randn(1, 4, 10)
        rows[0, 2:] = 0.0
        h, _ = block(rows, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
        torch.testing.assert_close(h[0, 2:], rows[0, 2:, :4])


class TestModel:
    def test_off_is_bit_identical_to_the_shipped_build(self) -> None:
        """Default build == a build that never heard of motion_norm: same state and output."""
        torch.manual_seed(0)
        legacy = KATVAD(
            hidden_dim=SMALL, temporal_heads=4, fusion_heads=4, motion_dim=MOTION_DIM
        ).eval()
        off = _model()
        assert legacy.state_dict().keys() == off.state_dict().keys()
        for key, value in legacy.state_dict().items():
            assert torch.equal(value, off.state_dict()[key]), key

    def test_the_ln_arm_starts_where_k5_starts(self) -> None:
        """W_u = 0 at init, so step 1's forward is identical with LN on or off (§25 start check)."""
        off, ln = _model(), _model(LN)
        assert ln.motion_residual is not None
        ln.motion_residual.set_norm_scale(C_VALUE)
        shared = {k: v for k, v in ln.state_dict().items() if k != "motion_residual.norm_scale"}
        assert shared.keys() == off.state_dict().keys()
        for key, value in shared.items():
            assert torch.equal(value, off.state_dict()[key]), key
        rows, lengths, feats = _rows(), torch.tensor([7, 5]), torch.randn(3, SMALL)
        a = off(rows, lengths, class_feats=feats)
        b = ln(rows, lengths, class_feats=feats)
        torch.testing.assert_close(a["cls_bin_logits"], b["cls_bin_logits"])

    def test_a_live_w_u_sees_a_different_input(self) -> None:
        off, ln = _model(), _model(LN)
        assert off.motion_residual is not None and ln.motion_residual is not None
        ln.motion_residual.set_norm_scale(C_VALUE)
        with torch.no_grad():
            for block in (off.motion_residual, ln.motion_residual):
                torch.manual_seed(3)
                block.proj.weight.normal_()
        rows, lengths, feats = _rows(), torch.tensor([7, 5]), torch.randn(3, SMALL)
        a = off(rows, lengths, class_feats=feats)["cls_bin_logits"]
        b = ln(rows, lengths, class_feats=feats)["cls_bin_logits"]
        assert not torch.allclose(a, b)

    def test_from_config_builds_the_ln_block(self) -> None:
        cfg = load_config(None, _motion_cfg(f"model.motion_norm={LN}"))
        model = KATVAD.from_config(cfg)
        assert model.motion_residual is not None and model.motion_residual.norm == LN


@pytest.fixture(scope="module")
def a2_cache(fixture: FixtureLayout, tmp_path_factory: pytest.TempPathFactory) -> Path:  # noqa: F811
    """Bake A2 (CRN off + VideoMAE-S-shaped motion) with the real manifest writer, so it has c."""
    rng = np.random.default_rng(11)
    ids = sorted(p.stem for p in fixture.clip_dir.glob("*.npy"))
    clips = {v: np.load(fixture.clip_dir / f"{v}.npy") for v in ids}
    motions = {v: rng.normal(0.0, 1.0, (len(x), D_S)).astype(np.float32) for v, x in clips.items()}
    stats = v2_inputs.fit_stats(
        {v: clips[v] for v in fixture.train_ids}, motions, constants.V2_OFF, ENCODER_S
    )
    out = tmp_path_factory.mktemp("a2_cache")
    width = build_v2_inputs.bake(ids, clips, motions, stats, out)
    v2_inputs.save_stats(out, stats)
    manifest = {"arm": "A2", "crn": constants.V2_OFF, "motion": ENCODER_S, "width": width,
                "c": stats.c}
    (out / constants.V2_INPUT_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return out


def _train_args(fixture: FixtureLayout, clip_dir: Path, out: Path, *sets: str) -> list[str]:  # noqa: F811
    args = [
        "--data-dir", str(fixture.data_dir), "--clip-dir", str(clip_dir),
        "--knn-cache", str(fixture.knn_cache_path), "--output-dir", str(out),
        "--text-encoder", "stub",
        "--set", "train.num_epochs=2", "--set", "train.batch_size=4",
        "--set", "train.device=cpu",
    ]
    for item in _motion_cfg(*sets):
        args += ["--set", item]
    return args


class TestEndToEnd:
    def test_ln_arm_trains_and_its_checkpoint_carries_c(
        self, fixture: FixtureLayout, a2_cache: Path, tmp_path: Path  # noqa: F811
    ) -> None:
        out = tmp_path / "k5ln"
        train.main(_train_args(fixture, a2_cache, out, f"model.motion_norm={LN}"))
        payload = torch.load(out / train.CHECKPOINT_LAST, map_location="cpu", weights_only=False)
        expected = json.loads((a2_cache / constants.V2_INPUT_MANIFEST_FILENAME).read_text())["c"]
        got = float(payload["model"]["motion_residual.norm_scale"])
        assert got == pytest.approx(expected, rel=1e-6)

    def test_step_one_losses_match_the_off_arm(
        self, fixture: FixtureLayout, a2_cache: Path, tmp_path: Path  # noqa: F811
    ) -> None:
        records = {}
        for name, sets in (("off", ()), ("ln", (f"model.motion_norm={LN}",))):
            out = tmp_path / name
            train.main(_train_args(fixture, a2_cache, out, *sets))
            first = (out / train.METRICS_FILENAME).read_text().splitlines()[0]
            records[name] = json.loads(first)
        tol = constants.KIM_START_TOL
        assert records["ln"]["mil"] == pytest.approx(records["off"]["mil"], abs=tol)

    def test_a_manifest_without_c_raises(
        self, fixture: FixtureLayout, a2_cache: Path, tmp_path: Path  # noqa: F811
    ) -> None:
        stripped = tmp_path / "no_c"
        stripped.mkdir()
        for path in a2_cache.iterdir():
            (stripped / path.name).symlink_to(path)
        manifest = json.loads((a2_cache / constants.V2_INPUT_MANIFEST_FILENAME).read_text())
        manifest.pop("c")
        (stripped / constants.V2_INPUT_MANIFEST_FILENAME).unlink()
        (stripped / constants.V2_INPUT_MANIFEST_FILENAME).write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="needs the cache's c"):
            train.main(_train_args(fixture, stripped, tmp_path / "out", f"model.motion_norm={LN}"))
