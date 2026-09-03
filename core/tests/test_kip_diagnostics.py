"""Phase 4 tests: gate diagnostics from module to .npz (plan T4.5).

These make ablation-ladder rows 1-2 readable off existing checkpoints with no
training: `s_t`, the gate ratio, the ECMR residual and the prototype norm are
carried out of the forward pass and written beside the scores.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core import constants
from core.config import Config, KIPConfig
from core.inference import (
    KIP_DIAG_NPZ_PREFIX,
    score_to_npz,
    sliding_window_scores,
)
from core.kip import (
    GATE_TYPE_CONSTANT,
    GATE_TYPE_MLP_FROZEN,
    GATE_TYPE_RANK,
    KIP,
    KIPOutput,
)
from core.models.kat_vad import KATVAD, KIP_DIAG_PREFIX

BATCH, LENGTH = 2, 24
DIAG_KEYS = {"s", "gate_ratio", "m", "mu_norm", "eo_norm"}
MAX_SHIFT = constants.HIDDEN_DIM // constants.FOLDING_FACTOR


def make_cfg(gate_type: str = GATE_TYPE_RANK) -> Config:
    cfg = Config()
    cfg.kip = KIPConfig(gate_type=gate_type)
    return cfg


def make_mask() -> torch.Tensor:
    mask = torch.zeros(BATCH, LENGTH)
    mask[0, :LENGTH] = 1.0
    mask[1, : LENGTH - 9] = 1.0
    return mask


class TestKIPOutput:
    def test_is_a_named_tuple_with_four_fields(self):
        out = KIP()(torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM))
        assert isinstance(out, KIPOutput)
        assert len(out) == 4
        assert out.vk.shape == (BATCH, LENGTH, constants.HIDDEN_DIM)
        assert out.eo_hat.shape == (BATCH, LENGTH, constants.FLOW_DIM)

    def test_diagnostics_absent_by_default(self):
        """The training path must pay nothing for a logging feature."""
        assert KIP()(torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM)).diagnostics is None

    def test_diagnostics_present_when_requested(self):
        vt = torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM)
        out = KIP()(vt, make_mask(), return_diagnostics=True)
        assert out.diagnostics is not None
        assert set(out.diagnostics) == DIAG_KEYS

    def test_requesting_diagnostics_does_not_change_outputs(self):
        vt = torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM)
        mask = make_mask()
        torch.manual_seed(3)
        kip = KIP()
        plain = kip(vt, mask)
        with_diag = kip(vt, mask, return_diagnostics=True)
        assert torch.equal(plain.vk, with_diag.vk)
        assert torch.equal(plain.eo_hat, with_diag.eo_hat)

    def test_diagnostics_none_when_shift_disabled(self):
        kip = KIP(use_gate_shift=False)
        out = kip(torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM), return_diagnostics=True)
        assert out.diagnostics is None


class TestDiagnosticContent:
    @pytest.mark.parametrize(
        "gate_type", [GATE_TYPE_RANK, GATE_TYPE_MLP_FROZEN, GATE_TYPE_CONSTANT]
    )
    def test_shapes_dtypes_and_detachment(self, gate_type: str):
        vt = torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM, requires_grad=True)
        out = KIP(gate_type=gate_type)(vt, make_mask(), return_diagnostics=True)
        assert out.diagnostics is not None
        for key, value in out.diagnostics.items():
            assert value.shape == (BATCH, LENGTH), key
            assert not value.requires_grad, key
        assert out.diagnostics["s"].dtype == torch.long

    @pytest.mark.parametrize(
        "gate_type", [GATE_TYPE_RANK, GATE_TYPE_MLP_FROZEN, GATE_TYPE_CONSTANT]
    )
    def test_padded_positions_are_zero(self, gate_type: str):
        vt = torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM)
        out = KIP(gate_type=gate_type)(vt, make_mask(), return_diagnostics=True)
        assert out.diagnostics is not None
        assert torch.all(out.diagnostics["s"][1, LENGTH - 9 :] == 0), gate_type

    def test_reported_s_is_the_s_that_was_applied(self):
        """A diagnostic that disagrees with the forward pass is worse than none."""
        vt = torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM)
        mask = make_mask()
        kip = KIP(gate_type=GATE_TYPE_RANK)
        out = kip(vt, mask, return_diagnostics=True)
        assert out.diagnostics is not None
        assert kip.shift is not None
        masked_vt = vt * mask.unsqueeze(-1)
        expected = kip.shift.compute_shift_counts(masked_vt, out.eo_hat, mask)
        assert torch.equal(out.diagnostics["s"], expected)

    def test_rank_gate_spans_the_full_range(self):
        vt = torch.randn(1, 128, constants.HIDDEN_DIM)
        out = KIP(gate_type=GATE_TYPE_RANK)(vt, return_diagnostics=True)
        assert out.diagnostics is not None
        s = out.diagnostics["s"]
        assert int(s.min()) == 0
        assert int(s.max()) == MAX_SHIFT

    def test_frozen_gate_is_near_constant(self):
        """Records the H4' measurement as a regression guard, not a claim."""
        vt = torch.randn(1, 128, constants.HIDDEN_DIM)
        torch.manual_seed(2024)
        out = KIP(gate_type=GATE_TYPE_MLP_FROZEN)(vt, return_diagnostics=True)
        assert out.diagnostics is not None
        s = out.diagnostics["s"]
        assert int(s.max() - s.min()) <= 8

    def test_mu_norm_is_zero_for_non_ecmr_gates(self):
        """Only the rank gate builds a prototype; others report a zero column."""
        vt = torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM)
        out = KIP(gate_type=GATE_TYPE_MLP_FROZEN)(vt, make_mask(), return_diagnostics=True)
        assert out.diagnostics is not None
        assert torch.all(out.diagnostics["mu_norm"] == 0)


class TestModelSurface:
    def test_katvad_forward_exposes_prefixed_keys(self):
        model = KATVAD.from_config(make_cfg(), training=False).eval()
        feats = torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM)
        lengths = torch.tensor([LENGTH, LENGTH - 9])
        text = torch.randn(3, constants.HIDDEN_DIM)
        with torch.no_grad():
            out = model(feats, lengths, class_feats=text, return_kip_diagnostics=True)
        assert {k[len(KIP_DIAG_PREFIX) :] for k in out if k.startswith(KIP_DIAG_PREFIX)} == (
            DIAG_KEYS
        )

    def test_off_by_default(self):
        model = KATVAD.from_config(make_cfg(), training=False).eval()
        with torch.no_grad():
            out = model(
                torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM),
                torch.tensor([LENGTH, LENGTH]),
                class_feats=torch.randn(3, constants.HIDDEN_DIM),
            )
        assert not any(k.startswith(KIP_DIAG_PREFIX) for k in out)

    def test_scores_unchanged_when_diagnostics_requested(self):
        model = KATVAD.from_config(make_cfg(), training=False).eval()
        feats = torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM)
        lengths = torch.tensor([LENGTH, LENGTH])
        text = torch.randn(3, constants.HIDDEN_DIM)
        with torch.no_grad():
            plain = model(feats, lengths, class_feats=text)
            diag = model(feats, lengths, class_feats=text, return_kip_diagnostics=True)
        assert torch.equal(plain["cls_bin_logits"], diag["cls_bin_logits"])


class TestSlidingWindowAndNpz:
    def _model(self) -> KATVAD:
        return KATVAD.from_config(make_cfg(), training=False).eval()

    def test_returns_empty_dict_when_not_requested(self):
        feats = torch.randn(40, constants.HIDDEN_DIM)
        text = torch.randn(3, constants.HIDDEN_DIM)
        _, _, diag = sliding_window_scores(self._model(), feats, lambda: text, 16)
        assert diag == {}

    def test_diagnostics_span_the_whole_video_across_windows(self):
        length = 40
        feats = torch.randn(length, constants.HIDDEN_DIM)
        text = torch.randn(3, constants.HIDDEN_DIM)
        score, _, diag = sliding_window_scores(
            self._model(), feats, lambda: text, 16, kip_diagnostics=True
        )
        assert set(diag) == DIAG_KEYS
        for key, value in diag.items():
            assert value.shape == (length,), key
        assert len(score) == length

    def test_npz_round_trip(self, tmp_path):
        length = 20
        feats = torch.randn(length, constants.HIDDEN_DIM)
        text = torch.randn(3, constants.HIDDEN_DIM)
        score, sim, diag = sliding_window_scores(
            self._model(), feats, lambda: text, 32, kip_diagnostics=True
        )
        path = score_to_npz(tmp_path, "clip0", score, sim, ["Normal", "A", "B"],
                            kip_diagnostics=diag)
        loaded = np.load(path, allow_pickle=False)
        for key in DIAG_KEYS:
            assert f"{KIP_DIAG_NPZ_PREFIX}{key}" in loaded.files
            assert loaded[f"{KIP_DIAG_NPZ_PREFIX}{key}"].shape == (length,)
        assert loaded[f"{KIP_DIAG_NPZ_PREFIX}s"].dtype == np.int16
        assert loaded[f"{KIP_DIAG_NPZ_PREFIX}m"].dtype == np.float32
        # existing readers must be unaffected
        assert loaded["score"].shape == (length,)
        assert loaded["sim"].shape == (length, 3)

    def test_npz_without_diagnostics_is_unchanged(self, tmp_path):
        score = torch.rand(10)
        sim = torch.rand(10, 3)
        path = score_to_npz(tmp_path, "clip1", score, sim, ["Normal", "A", "B"])
        loaded = np.load(path, allow_pickle=False)
        assert not any(k.startswith(KIP_DIAG_NPZ_PREFIX) for k in loaded.files)
        assert set(loaded.files) == {"score", "sim", "class_names"}


class TestCLIDefaults:
    def test_inference_defaults_to_on(self):
        from core.inference import build_arg_parser

        args = build_arg_parser().parse_args(["--features", "x.npy", "--output-dir", "o"])
        assert args.dump_kip_diag is True

    def test_inference_can_be_turned_off(self):
        from core.inference import build_arg_parser

        args = build_arg_parser().parse_args(
            ["--features", "x.npy", "--output-dir", "o", "--no-dump-kip-diag"]
        )
        assert args.dump_kip_diag is False

    def test_evaluate_defaults_to_on(self):
        from core.evaluate import build_arg_parser

        args = build_arg_parser().parse_args(["--output-dir", "o"])
        assert args.dump_kip_diag is True

    def test_training_never_collects_diagnostics(self):
        """The decision was on for inference, off for training — assert the latter."""
        import inspect

        from core import train

        source = inspect.getsource(train)
        assert "return_kip_diagnostics" not in source
        assert "dump_kip_diag" not in source
