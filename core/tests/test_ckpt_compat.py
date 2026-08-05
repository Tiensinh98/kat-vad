"""Checkpoint-compat loader tests on a synthetic baseline-keyed state dict.

Builds a fake LaGoVAD ``best.ckpt`` payload with the baseline's exact key
layout (derived via the pinned inverse mapping from a randomly initialized
KATVAD) and asserts the compat loader restores every mapped tensor exactly,
skips the frozen CLIP body, leaves KIP untouched, and fails loudly on
unknown / missing / mis-shaped keys. The real-ckpt check is Phase 6 gate (a).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.config import Config
from core.models.ckpt_compat import (
    baseline_key_for,
    load_baseline_checkpoint,
    load_baseline_state_dict,
    map_baseline_key,
)
from core.models.kat_vad import KATVAD


def _model(kip: bool = True, gate: bool = False) -> KATVAD:
    cfg = Config()
    cfg.kip.enabled = kip
    cfg.model.temp_gate = gate
    torch.manual_seed(0)
    return KATVAD.from_config(cfg)


def _baseline_state_dict(model: KATVAD) -> dict[str, torch.Tensor]:
    """Synthetic baseline ckpt content: our tensors under baseline key names."""
    state = {}
    for key, value in model.state_dict().items():
        baseline_key = baseline_key_for(key)
        if baseline_key is not None:
            state[baseline_key] = value.clone()
    return state


class TestKeyMapping:
    def test_round_trip_all_keys(self) -> None:
        model = _model(kip=True, gate=True)
        for our_key in model.state_dict():
            baseline_key = baseline_key_for(our_key)
            if our_key.startswith("kip."):
                assert baseline_key is None
            else:
                assert baseline_key is not None
                assert map_baseline_key(baseline_key) == our_key

    def test_temporal_and_gate_mapping(self) -> None:
        assert (
            map_baseline_key("temporal_encoder.layer.0.attention.self.query.weight")
            == "temporal_encoder.encoder.layer.0.attention.self.query.weight"
        )
        assert map_baseline_key("gate_alpha") == "temporal_encoder.gate_alpha"
        assert map_baseline_key("clip_text_model.model.encoder.x") is None

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError, match="Unrecognized baseline"):
            map_baseline_key("mystery_module.weight")


class TestLoad:
    def test_exact_restore_and_kip_untouched(self) -> None:
        donor = _model(kip=False)
        state = _baseline_state_dict(donor)

        target = _model(kip=True)
        torch.manual_seed(123)  # re-randomize so the load has to do the work
        for param in target.parameters():
            param.data.add_(torch.randn_like(param) * 0.1)
        kip_before = {
            k: v.clone() for k, v in target.state_dict().items() if k.startswith("kip.")
        }

        report = load_baseline_state_dict(target, state)
        assert len(report.loaded) == len(state)
        assert not report.missing_ours

        restored = target.state_dict()
        for baseline_key, tensor in state.items():
            our_key = map_baseline_key(baseline_key)
            assert our_key is not None
            torch.testing.assert_close(restored[our_key], tensor)
        for key, tensor in kip_before.items():
            torch.testing.assert_close(restored[key], tensor)

    def test_clip_body_keys_skipped(self) -> None:
        donor = _model(kip=False)
        state = _baseline_state_dict(donor)
        state["clip_text_model.model.embeddings.token_embedding.weight"] = torch.zeros(3, 3)
        report = load_baseline_state_dict(_model(kip=True), state)
        assert report.skipped_clip_body == [
            "clip_text_model.model.embeddings.token_embedding.weight"
        ]

    def test_missing_key_raises(self) -> None:
        state = _baseline_state_dict(_model(kip=False))
        state.pop(next(k for k in state if k.startswith("bin_head.")))
        with pytest.raises(KeyError, match="missing weights"):
            load_baseline_state_dict(_model(kip=True), state)

    def test_shape_mismatch_raises(self) -> None:
        state = _baseline_state_dict(_model(kip=False))
        key = next(iter(state))
        state[key] = torch.zeros(1)
        with pytest.raises(ValueError, match="Shape mismatch"):
            load_baseline_state_dict(_model(kip=True), state)

    def test_lightning_file_round_trip(self, tmp_path: Path) -> None:
        donor = _model(kip=False)
        ckpt_path = tmp_path / "best.ckpt"
        torch.save({"state_dict": _baseline_state_dict(donor)}, ckpt_path)
        target = _model(kip=True)
        report = load_baseline_checkpoint(target, ckpt_path)
        assert report.loaded
        fusion_key = next(k for k in donor.state_dict() if k.startswith("fusion."))
        torch.testing.assert_close(
            target.state_dict()[fusion_key], donor.state_dict()[fusion_key]
        )
