"""Phase 2 tests: the four gate types and the config surface (plan T2.6).

The bit-identity fixture below was captured from the pre-refactor code and is
the guard on lesson C14: every number in ``core/docs/RESULTS_*.md`` was measured
with ``mlp_frozen``, so if that path ever moves, those numbers stop describing
any model we can build.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
import yaml

from core import constants
from core.config import KIPConfig, load_config
from core.kip import (
    GATE_SIGNAL_FEAT_VAR,
    GATE_TYPE_CONSTANT,
    GATE_TYPE_MLP_FROZEN,
    GATE_TYPE_MLP_STE,
    GATE_TYPE_RANK,
    KIP,
    KinematicShift,
)

LOGGER = logging.getLogger(__name__)

MAX_SHIFT = constants.HIDDEN_DIM // constants.FOLDING_FACTOR  # 128

# Captured from the pre-refactor KinematicShift under torch.manual_seed(1234).
GOLDEN_SEED = 1234
GOLDEN_COUNTS = [
    [65, 65, 65, 66, 65, 65, 65, 65, 66, 65, 65, 65],
    [65, 65, 65, 66, 65, 65, 66, 0, 0, 0, 0, 0],
]
GOLDEN_COUNTS_FEAT_VAR = [
    [65, 65, 66, 65, 65, 65, 65, 65, 65, 65, 65, 65],
    [65, 65, 65, 65, 66, 65, 65, 0, 0, 0, 0, 0],
]
GOLDEN_VK_SUM = -54.4203834534


def golden_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly the tensors used to capture the fixture — order matters."""
    torch.manual_seed(GOLDEN_SEED)
    module_seeded = KinematicShift(gate_type=GATE_TYPE_MLP_FROZEN)
    vt = torch.randn(2, 12, constants.HIDDEN_DIM)
    eo = torch.randn(2, 12, constants.FLOW_DIM)
    mask = torch.zeros(2, 12)
    mask[0, :12] = 1.0
    mask[1, :7] = 1.0
    del module_seeded  # constructed only to advance the RNG identically
    return vt, eo, mask


def seeded_gate(gate_type: str, **kwargs: object) -> KinematicShift:
    torch.manual_seed(GOLDEN_SEED)
    return KinematicShift(gate_type=gate_type, **kwargs)  # type: ignore[arg-type]


class TestBitIdentity:
    """mlp_frozen must reproduce v1 exactly — this is the C14 guard."""

    def test_counts_match_pre_refactor_fixture(self):
        vt, eo, mask = golden_inputs()
        gate = seeded_gate(GATE_TYPE_MLP_FROZEN)
        assert gate.compute_shift_counts(vt, eo, mask).tolist() == GOLDEN_COUNTS

    def test_output_matches_pre_refactor_fixture(self):
        vt, eo, mask = golden_inputs()
        gate = seeded_gate(GATE_TYPE_MLP_FROZEN)
        assert float(gate(vt, eo, mask).sum()) == pytest.approx(GOLDEN_VK_SUM, abs=1e-4)

    def test_feat_var_matches_pre_refactor_fixture(self):
        vt, eo, mask = golden_inputs()
        gate = seeded_gate(GATE_TYPE_MLP_FROZEN, gate_signal=GATE_SIGNAL_FEAT_VAR)
        assert gate.compute_shift_counts(vt, eo, mask).tolist() == GOLDEN_COUNTS_FEAT_VAR


class TestSTE:
    def test_forward_is_bit_identical_to_frozen(self):
        """The whole point of the variant: a pure gradient intervention."""
        vt, eo, mask = golden_inputs()
        frozen = seeded_gate(GATE_TYPE_MLP_FROZEN)
        ste = seeded_gate(GATE_TYPE_MLP_STE)
        assert torch.equal(ste(vt, eo, mask), frozen(vt, eo, mask))

    def test_gradient_reaches_the_gate_mlp(self):
        vt, eo, mask = golden_inputs()
        ste = seeded_gate(GATE_TYPE_MLP_STE)
        eo = eo.clone().requires_grad_(True)
        ste(vt, eo, mask).sum().backward()
        assert ste.mlp is not None
        grads = [p.grad for p in ste.mlp.parameters()]
        assert all(g is not None for g in grads)
        assert any(float(g.abs().sum()) > 0 for g in grads if g is not None)

    def test_gradient_reaches_the_flow_embedding(self):
        vt, eo, mask = golden_inputs()
        ste = seeded_gate(GATE_TYPE_MLP_STE)
        eo = eo.clone().requires_grad_(True)
        ste(vt, eo, mask).sum().backward()
        assert eo.grad is not None
        assert float(eo.grad.abs().sum()) > 0


class TestGradientTopologyUnchanged:
    """rank removes 321 dead params; it must not add or delete a gradient edge."""

    @pytest.mark.parametrize(
        "gate_type", [GATE_TYPE_RANK, GATE_TYPE_MLP_FROZEN, GATE_TYPE_CONSTANT]
    )
    def test_no_gradient_reaches_eo(self, gate_type: str):
        _, _, mask = golden_inputs()
        gate = seeded_gate(gate_type)
        vt = torch.randn(2, 12, constants.HIDDEN_DIM, requires_grad=True)
        eo = torch.randn(2, 12, constants.FLOW_DIM, requires_grad=True)
        gate(vt, eo, mask).sum().backward()
        assert eo.grad is None, f"{gate_type} unexpectedly opened a gradient path to e_O"
        assert vt.grad is not None, "the shift must still pass gradient to v^t"

    def test_frozen_mlp_still_receives_nothing(self):
        _, _, mask = golden_inputs()
        gate = seeded_gate(GATE_TYPE_MLP_FROZEN)
        vt = torch.randn(2, 12, constants.HIDDEN_DIM, requires_grad=True)
        eo = torch.randn(2, 12, constants.FLOW_DIM)
        gate(vt, eo, mask).sum().backward()
        assert gate.mlp is not None
        assert all(p.grad is None for p in gate.mlp.parameters())


class TestParameterCounts:
    @pytest.mark.parametrize(
        ("gate_type", "expected"),
        [
            (GATE_TYPE_RANK, 0),
            (GATE_TYPE_CONSTANT, 0),
            (GATE_TYPE_MLP_FROZEN, 321),
            (GATE_TYPE_MLP_STE, 321),
        ],
    )
    def test_score_path_parameter_count(self, gate_type: str, expected: int):
        gate = KinematicShift(gate_type=gate_type)
        assert sum(p.numel() for p in gate.parameters()) == expected

    def test_rank_gate_builds_no_mlp(self):
        assert KinematicShift(gate_type=GATE_TYPE_RANK).mlp is None
        assert KinematicShift(gate_type=GATE_TYPE_CONSTANT).mlp is None

    def test_kip_inference_path_under_rank_gate(self):
        """v3 §9: +311,808 params, zero on the score path."""
        kip = KIP(gate_type=GATE_TYPE_RANK)
        assert kip.shift is not None
        assert sum(p.numel() for p in kip.pmg.parameters()) == 311_808
        assert sum(p.numel() for p in kip.shift.parameters()) == 0
        assert sum(p.numel() for p in kip.parameters()) == 443_393


class TestGateBehaviour:
    def test_rank_spans_full_range_frozen_does_not(self):
        """The measured difference between the v1 and v3 gates."""
        vt = torch.randn(1, 200, constants.HIDDEN_DIM)
        eo = torch.randn(1, 200, constants.FLOW_DIM)
        mask = torch.ones(1, 200)
        rank = seeded_gate(GATE_TYPE_RANK).compute_shift_counts(vt, eo, mask)
        frozen = seeded_gate(GATE_TYPE_MLP_FROZEN).compute_shift_counts(vt, eo, mask)
        assert int(rank.max() - rank.min()) == MAX_SHIFT
        assert int(frozen.max() - frozen.min()) <= 8  # measured: 0-4 across 8 seeds

    def test_constant_gate_is_constant_and_ignores_flow(self):
        vt = torch.randn(1, 40, constants.HIDDEN_DIM)
        mask = torch.ones(1, 40)
        gate = KinematicShift(gate_type=GATE_TYPE_CONSTANT, const_shift_ratio=0.25)
        a = gate.compute_shift_counts(vt, torch.randn(1, 40, constants.FLOW_DIM), mask)
        b = gate.compute_shift_counts(vt, torch.randn(1, 40, constants.FLOW_DIM) * 99, mask)
        assert torch.equal(a, b)
        assert int(a.min()) == int(a.max()) == int(0.25 * MAX_SHIFT)

    def test_all_gate_types_preserve_shape_and_padding(self):
        vt = torch.randn(2, 20, constants.HIDDEN_DIM)
        eo = torch.randn(2, 20, constants.FLOW_DIM)
        mask = torch.zeros(2, 20)
        mask[0, :20] = 1.0
        mask[1, :11] = 1.0
        for gate_type in (
            GATE_TYPE_RANK,
            GATE_TYPE_MLP_FROZEN,
            GATE_TYPE_MLP_STE,
            GATE_TYPE_CONSTANT,
        ):
            vk = seeded_gate(gate_type)(vt, eo, mask)
            assert vk.shape == vt.shape, gate_type
            assert torch.all(vk[1, 11:] == 0), gate_type

    def test_diagnostics_shapes_and_keys(self):
        vt = torch.randn(2, 15, constants.HIDDEN_DIM)
        eo = torch.randn(2, 15, constants.FLOW_DIM)
        mask = torch.ones(2, 15)
        for gate_type in (GATE_TYPE_RANK, GATE_TYPE_MLP_FROZEN, GATE_TYPE_CONSTANT):
            diag = seeded_gate(gate_type).diagnostics(vt, eo, mask)
            assert set(diag) == {"s", "gate_ratio", "m", "mu_norm", "eo_norm"}
            for key, value in diag.items():
                assert value.shape == (2, 15), (gate_type, key)
                assert not value.requires_grad, (gate_type, key)


class TestGateConstruction:
    def test_unknown_gate_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown gate_type"):
            KinematicShift(gate_type="bogus")

    def test_gate_signal_rejected_for_parameter_free_gates(self):
        """Silently ignoring it would let a config lie about what ran."""
        for gate_type in (GATE_TYPE_RANK, GATE_TYPE_CONSTANT):
            with pytest.raises(ValueError, match="meaningless"):
                KinematicShift(gate_type=gate_type, gate_signal="flow_norm")

    def test_unknown_gate_signal_rejected_for_mlp_gates(self):
        with pytest.raises(ValueError, match="Unknown gate_signal"):
            KinematicShift(gate_type=GATE_TYPE_MLP_FROZEN, gate_signal="bogus")

    def test_const_shift_ratio_bounds(self):
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError, match="const_shift_ratio"):
                KinematicShift(gate_type=GATE_TYPE_CONSTANT, const_shift_ratio=bad)


class TestKIPConfigSurface:
    def test_fresh_config_defaults_to_rank(self):
        assert KIPConfig().gate_type == GATE_TYPE_RANK
        assert KIPConfig().gate_signal is None

    def test_post_init_validates(self):
        with pytest.raises(ValueError, match=r"Unknown kip\.gate_type"):
            KIPConfig(gate_type="bogus")
        with pytest.raises(ValueError, match="meaningless"):
            KIPConfig(gate_type=GATE_TYPE_RANK, gate_signal="flow_norm")
        with pytest.raises(ValueError, match="ecmr_lambda"):
            KIPConfig(ecmr_lambda=1.0)

    def test_from_config_threads_every_gate_field(self):
        cfg = KIPConfig(gate_type=GATE_TYPE_CONSTANT, const_shift_ratio=0.75, ecmr_lambda=0.5)
        shift = KIP.from_config(cfg).shift
        assert shift is not None
        assert shift.gate_type == GATE_TYPE_CONSTANT
        assert shift.const_shift_ratio == 0.75
        assert shift.ecmr_lambda == 0.5


class TestAmbiguousConfigGuard:
    """T2.4 — a pre-v3 run config must not resolve to a gate it never ran."""

    def _write(self, tmp_path: Path, kip_section: dict | None) -> Path:
        payload: dict = {"model": {"hidden_dim": constants.HIDDEN_DIM}}
        if kip_section is not None:
            payload["kip"] = kip_section
        path = tmp_path / "config.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh)
        return path

    def test_pre_v3_config_raises(self, tmp_path):
        """Shape copied from outputs/MSAD_ncc/stage2_kip_on/config.yaml."""
        path = self._write(
            tmp_path,
            {
                "enabled": True,
                "pmg_only": False,
                "use_gate_shift": True,
                "use_lkin": True,
                "gate_signal": "flow_norm",
                "on_raw_features": False,
            },
        )
        with pytest.raises(KeyError, match="predates the v3 gate"):
            load_config(path)

    def test_explicit_gate_type_is_accepted(self, tmp_path):
        path = self._write(
            tmp_path, {"enabled": True, "gate_type": "mlp_frozen", "gate_signal": "flow_norm"}
        )
        assert load_config(path).kip.gate_type == GATE_TYPE_MLP_FROZEN

    def test_kip_disabled_config_is_not_ambiguous(self, tmp_path):
        """No gate exists, so there is nothing to be ambiguous about."""
        path = self._write(tmp_path, {"enabled": False, "use_lkin": False})
        assert load_config(path).kip.enabled is False

    def test_config_without_a_kip_section_gets_the_v3_default(self, tmp_path):
        path = self._write(tmp_path, None)
        assert load_config(path).kip.gate_type == GATE_TYPE_RANK

    def test_invalid_gate_type_in_file_raises(self, tmp_path):
        path = self._write(tmp_path, {"enabled": True, "gate_type": "bogus"})
        with pytest.raises(ValueError, match=r"Unknown kip\.gate_type"):
            load_config(path)

    def test_override_path_is_validated_too(self, tmp_path):
        """_apply_section and the override loop both bypass __post_init__."""
        with pytest.raises(ValueError, match=r"Unknown kip\.gate_type"):
            load_config(None, overrides=["kip.gate_type=bogus"])

    def test_real_run_configs_on_disk_all_raise(self):
        """Every archived run config predates the key; none may load silently."""
        configs = sorted(Path("outputs").glob("*/*/config.yaml"))
        if not configs:
            pytest.skip("no archived run configs on this machine")
        for path in configs:
            with pytest.raises((KeyError, ValueError)):
                load_config(path)


class TestRequireFlow:
    def test_disable_pmg_drops_the_flow_requirement(self):
        """T2.5: the plain-TSM control has no PMG head, so it needs no targets."""
        cfg = KIPConfig(gate_type=GATE_TYPE_CONSTANT, disable_pmg=True)
        assert (cfg.enabled and not cfg.disable_pmg) is False
        assert (KIPConfig().enabled and not KIPConfig().disable_pmg) is True
