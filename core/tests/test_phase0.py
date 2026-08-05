"""Phase 0 tests: config system, device resolution, data-layout constants."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core import constants
from core.config import Config, load_config
from core.device import resolve_device


class TestConfig:
    def test_defaults_match_baseline(self):
        cfg = Config()
        assert cfg.model.hidden_dim == 512
        assert cfg.kip.d_flow == 256
        assert cfg.kip.folding_factor == 4
        assert cfg.loss.lambda_rec == 1.0
        assert cfg.loss.lambda_align == 0.1
        assert cfg.loss.gamma_kin == 0.2
        assert cfg.loss.beta_cons == 0.5
        assert cfg.loss.tau_align == 0.07
        assert cfg.dvs.theta == 0.7
        assert cfg.dvs.theta_ego == 0.85
        assert cfg.train.learning_rate == 5e-5
        assert cfg.train.batch_size == 64
        assert cfg.train.seed == 2024

    def test_yaml_roundtrip(self, tmp_path: Path):
        cfg = Config()
        cfg.kip.enabled = False
        cfg.train.stage = 1
        path = tmp_path / "cfg.yaml"
        cfg.save_yaml(path)
        loaded = load_config(path)
        assert loaded.kip.enabled is False
        assert loaded.train.stage == 1
        assert loaded.to_dict() == cfg.to_dict()

    def test_overrides(self):
        cfg = load_config(
            overrides=["kip.enabled=false", "train.batch_size=8", "loss.lambda_align=0.5"]
        )
        assert cfg.kip.enabled is False
        assert cfg.train.batch_size == 8
        assert cfg.loss.lambda_align == 0.5

    def test_unknown_key_rejected(self):
        with pytest.raises(KeyError):
            load_config(overrides=["kip.bogus=1"])
        with pytest.raises(KeyError):
            load_config(overrides=["bogus.key=1"])

    def test_yaml_unknown_section_rejected(self, tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text("nonsense:\n  a: 1\n", encoding="utf-8")
        with pytest.raises(KeyError):
            load_config(path)


class TestDevice:
    def test_cpu_explicit(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_auto_returns_device(self):
        dev = resolve_device("auto")
        assert dev.type in ("cuda", "mps", "cpu")

    def test_invalid_spec(self):
        with pytest.raises(ValueError):
            resolve_device("tpu")


class TestLayoutContract:
    def test_cache_paths_versioned(self):
        assert constants.FLOW_CACHE_DIR.name == constants.FLOW_CACHE_VERSION
        assert constants.CLIP_CACHE_DIR.parent == constants.CACHE_ROOT

    def test_label_filenames(self):
        assert constants.LABELS_TRAIN_FILENAME.endswith(".json")
        assert constants.FRAME_LABELS_TEST_FILENAME.endswith(".json")
        assert constants.DEFS_FILENAME.endswith(".json")
