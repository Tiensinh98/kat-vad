"""Typed configuration for KAT-VAD.

Dataclass-backed config with YAML loading and ``key.subkey=value`` CLI overrides.
Every spec §10 ablation toggle and §11 flag lives here; defaults come from
:mod:`core.constants` (baseline values adopted verbatim for reproduction).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core import constants

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Backbone + baseline architecture."""

    backbone: str = "clip_vitb16"  # | "clip_vitb16+hn" | "openclip_vitl" | "alertclip"
    hidden_dim: int = constants.HIDDEN_DIM
    temporal_layers: int = constants.TEMPORAL_LAYERS
    temporal_heads: int = constants.TEMPORAL_HEADS
    temporal_window: int = constants.TEMPORAL_WINDOW
    temporal_max_positions: int = constants.TEMPORAL_MAX_POSITIONS
    temp_gate: bool = False  # baseline default.yaml ships with the gate disabled
    temp_gate_weight: float = constants.TEMP_GATE_WEIGHT
    temp_gate_init: float = constants.TEMP_GATE_INIT
    num_soft_prompts: int = constants.NUM_SOFT_PROMPTS
    fusion_num_layers: int = constants.FUSION_NUM_LAYERS
    fusion_heads: int = constants.FUSION_HEADS
    score_head_layers: int = constants.SCORE_HEAD_LAYERS
    score_head_kernel: int = constants.SCORE_HEAD_KERNEL
    bin_head_type: str = "adaptive"  # "vanilla" | "fused_vanilla" | "adaptive"
    adaptive_fuse_alpha0: float = constants.ADAPTIVE_FUSE_ALPHA0
    adaptive_fuse_scale: float = constants.ADAPTIVE_FUSE_SCALE
    multiclass_temp: float = constants.MULTICLASS_TEMP


@dataclass
class KIPConfig:
    """KIP module + spec §10 ablation toggles."""

    enabled: bool = True  # ablation 1: False = pure LaGoVAD baseline
    pmg_only: bool = False  # ablation 2: v^k = v^t, only L_KIP_rec + L_KIP_align active
    use_gate_shift: bool = True  # ablation 3 pairs with use_lkin=False
    use_lkin: bool = True  # ablation 4
    gate_signal: str = "flow_norm"  # ablation 5: "flow_norm" (ours) | "feat_var" (RefineVAD)
    on_raw_features: bool = False  # ablation 6: splice KIP directly on F (skip temporal enc)
    folding_factor: int = constants.FOLDING_FACTOR
    d_flow: int = constants.FLOW_DIM
    pmg_latent_dim: int = constants.PMG_LATENT_DIM
    align_proj_dim: int = constants.ALIGN_PROJ_DIM


@dataclass
class LossConfig:
    """Loss weights and A11 mitigation flags for L_KIP_align."""

    lambda_rec: float = constants.LAMBDA_REC
    lambda_align: float = constants.LAMBDA_ALIGN
    gamma_kin: float = constants.GAMMA_KIN
    beta_cons: float = constants.BETA_CONS
    tau_align: float = constants.TAU_ALIGN
    mil_topk_pct: int = constants.MIL_TOPK_PCT
    sup_mil_topk_pct: int = constants.SUP_MIL_TOPK_PCT
    mul_mil_topk_pct: int = constants.MUL_MIL_TOPK_PCT
    mul_weight: float = 1.0
    pseudo_sup_weight: float = 1.0
    pseudo_sup_mil_weight: float = 1.0
    cap_contrastive_weight: float = 1.0
    contrastive_neg_mining: str = "n3"  # "vanilla" | "n3"
    contrastive_temp: float = constants.CONTRASTIVE_TEMP
    use_yp_anchor: bool = True
    # --- Phase 1 arms (DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md §6) ---------
    # Both default to the baseline; flipping either is an experiment.
    # 1.1 (C29): "span" = the whole spliced anchor is a dense positive
    # (baseline); "ignore" = drop the anchor interior from the dense BCE.
    dvs_anchor_mode: str = constants.DVS_ANCHOR_MODE_SPAN
    # 1.2: weight of the bottom-k MIL term on abnormal clips (0 = off).
    bottomk_weight: float = constants.BOTTOMK_WEIGHT
    bottomk_topk_pct: int = constants.BOTTOMK_MIL_TOPK_PCT
    # Derive per-video captions from the class-definition verbalizer when the
    # dataset ships no descriptions (MSAD). Off = baseline-faithful (caption
    # branch + L_neg inactive). Caveat: same-class captions become InfoNCE
    # false negatives, hence not the default.
    captions_from_definitions: bool = False
    # A11 mitigations (default off = spec-as-written)
    align_subsample: int = 0  # >0: subsample this many positions per video
    align_exclude_window: int = 0  # >0: exclude ±w temporal neighbors as negatives


@dataclass
class DVSConfig:
    """Dynamic video synthesis (spec §6.1). theta = NO-synthesis probability."""

    theta: float = constants.DVS_THETA
    theta_ego: float = constants.DVS_THETA_EGO
    delta_m: int = constants.DVS_DELTA_M
    delta_m_ego: int = constants.DVS_DELTA_M_EGO
    syn_max_num_clips: int = 5
    knn_filler_ratio: float = constants.DVS_KNN_FILLER_RATIO
    motion_aware_knn_key: bool = False  # append e_O descriptor to KNN key (ego-centric)


@dataclass
class DataConfig:
    """Dataset identity + sampling."""

    dataset: str = constants.MSAD_DATASET  # user has MSAD-traffic on disk (2026-07-08)
    is_egocentric: bool = False
    frame_stride: int = constants.FRAME_STRIDE
    crop_size: int = constants.CROP_SIZE
    max_vis_len: int = constants.MAX_VIS_LEN
    num_workers: int = 4


@dataclass
class TrainConfig:
    """Optimizer / schedule / run control (baseline hyperparameters)."""

    stage: int = 2  # 1 = KIP-only warm-up, 2 = full objective
    learning_rate: float = constants.LEARNING_RATE
    batch_size: int = constants.BATCH_SIZE
    num_epochs: int = constants.NUM_EPOCHS
    warmup_steps: int = constants.WARMUP_STEPS
    weight_decay: float = 0.01
    seed: int = constants.SEED
    amp: bool = False
    grad_accum_steps: int = 1
    device: str = "auto"
    stage0_warmup: bool = False
    stage05_hntune: bool = False


@dataclass
class Config:
    """Top-level KAT-VAD configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    kip: KIPConfig = field(default_factory=KIPConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dvs: DVSConfig = field(default_factory=DVSConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
        LOGGER.info("Config saved to %s", path)


def _coerce(value: str, current: Any) -> Any:
    """Coerce a CLI string override to the type of the current value."""
    if isinstance(current, bool):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot parse boolean from {value!r}")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply_section(section: Any, data: dict[str, Any], prefix: str) -> None:
    valid = {f.name for f in dataclasses.fields(section)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"Unknown config key: {prefix}{key}")
        setattr(section, key, value)


def load_config(path: Path | None = None, overrides: list[str] | None = None) -> Config:
    """Build a Config from an optional YAML file plus ``section.key=value`` overrides."""
    cfg = Config()

    if path is not None:
        with Path(path).open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        valid_sections = {f.name for f in dataclasses.fields(cfg)}
        for section_name, section_data in raw.items():
            if section_name not in valid_sections:
                raise KeyError(f"Unknown config section: {section_name}")
            if section_data is not None:
                _apply_section(getattr(cfg, section_name), section_data, f"{section_name}.")
        LOGGER.info("Config loaded from %s", path)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        parts = dotted.split(".")
        if len(parts) != 2:
            raise ValueError(f"Override must be section.key=value, got {item!r}")
        section_name, key = parts
        section = getattr(cfg, section_name, None)
        if section is None or not dataclasses.is_dataclass(section):
            raise KeyError(f"Unknown config section: {section_name}")
        if key not in {f.name for f in dataclasses.fields(section)}:
            raise KeyError(f"Unknown config key: {dotted}")
        setattr(section, key, _coerce(value, getattr(section, key)))
        LOGGER.debug("Config override applied: %s", item)

    return cfg
