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
    # v2 motion stream (architecture §5): width d_v of the pre-scaled motion columns
    # appended to each input row. 0 = no motion stream (A0/A1, every v1 checkpoint).
    motion_dim: int = 0


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
class V2Config:
    """Which v2 input the run was fed (architecture §11). Provenance, not architecture.

    Both streams are baked into the input cache by ``core.tools.build_v2_inputs``;
    these fields name what the cache must contain, and the cache's manifest is
    checked against them (:func:`core.data.v2_inputs.check_input_manifest`).
    """

    crn: str = constants.V2_OFF  # none | R1 | R2 | R3 | R4
    motion: str = constants.V2_OFF  # none | VideoMAE V2 encoder name


@dataclass
class V5Config:
    """v5 grid readout (``core/docs/v5/PREREG_V5.md`` §2, S2-3 ... S2-6).

    ``readout = none`` is every v1 / v2 model. Otherwise each input row also carries the
    flattened V7 grid (and its change ``G - G_{t-l}`` for ``keys = change``), and the motion
    term of v2's A2 is computed from a readout of that grid instead of the baked ``u``.
    """

    readout: str = constants.V5_READOUT_OFF  # none | mean (R1) | mqr (R3 / R5 / R5Δ)
    queries: int = 1  # M; 1 for R1 / R3, V5_QUERIES for R5 / R5Δ
    keys: str = constants.V5_KEYS_CONTENT  # content | change (R5Δ)
    smil_weight: float = 0.0  # lambda of L_sMIL; 0 = term absent (R1 / R3)
    query_lr_mult: float = 1.0  # §2.4 B13: AdamW LR multiplier (and no weight decay) of the queries


@dataclass
class Config:
    """Top-level KAT-VAD configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    kip: KIPConfig = field(default_factory=KIPConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dvs: DVSConfig = field(default_factory=DVSConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    v2: V2Config = field(default_factory=V2Config)
    v5: V5Config = field(default_factory=V5Config)

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


def validate_v2(cfg: Config) -> None:
    """Fail loud on a v2 config that cannot be the arm it names (architecture §11).

    * ``v2.crn`` / ``v2.motion`` must be known values;
    * a motion stream needs ``model.motion_dim`` > 0 and vice versa;
    * v2 retires KIP: either v2 stream with ``kip.enabled`` is a mixed model no arm defines.
    """
    v2 = cfg.v2
    if v2.crn not in constants.V2_CRN_CHOICES:
        raise ValueError(f"v2.crn must be one of {constants.V2_CRN_CHOICES}, got {v2.crn!r}")
    if v2.motion not in constants.V2_MOTION_CHOICES:
        raise ValueError(
            f"v2.motion must be one of {constants.V2_MOTION_CHOICES}, got {v2.motion!r}"
        )
    has_motion = v2.motion != constants.V2_OFF
    if has_motion != (cfg.model.motion_dim > 0):
        raise ValueError(
            f"v2.motion={v2.motion!r} needs model.motion_dim > 0 (and only then); "
            f"got model.motion_dim={cfg.model.motion_dim}"
        )
    if has_motion:
        expected = constants.VIDEOMAE_ARCH[v2.motion][0]
        if cfg.model.motion_dim != expected:
            raise ValueError(
                f"model.motion_dim={cfg.model.motion_dim} but {v2.motion} emits {expected}"
            )
    if cfg.kip.enabled and (has_motion or v2.crn != constants.V2_OFF):
        raise ValueError("v2 inputs (v2.crn / v2.motion) require kip.enabled=false")


def validate_v5(cfg: Config) -> None:
    """Fail loud on a v5 config that is not one of PREREG_V5 §2's rows.

    The grid readout replaces A2's baked motion stream, so it needs exactly A2's inputs
    (V2-S, no CRN, KIP off; S2-1); the query count must split the grid channels evenly
    (``z_m`` is the ``m``-th ``D/M`` slice, S2-4); change keys and the sMIL term need queries.
    """
    v5 = cfg.v5
    if v5.readout not in constants.V5_READOUTS:
        raise ValueError(f"v5.readout must be one of {constants.V5_READOUTS}, got {v5.readout!r}")
    if v5.keys not in constants.V5_KEYS:
        raise ValueError(f"v5.keys must be one of {constants.V5_KEYS}, got {v5.keys!r}")
    if v5.smil_weight < 0:
        raise ValueError(f"v5.smil_weight must be >= 0, got {v5.smil_weight}")
    if v5.query_lr_mult <= 0:
        raise ValueError(f"v5.query_lr_mult must be > 0, got {v5.query_lr_mult}")
    if v5.query_lr_mult != 1 and v5.readout != constants.V5_READOUT_MQR:
        raise ValueError("v5.query_lr_mult needs v5.readout=mqr (only MQR has queries)")
    if v5.readout == constants.V5_READOUT_OFF:
        return
    if (cfg.v2.crn, cfg.v2.motion) != (constants.V2_OFF, constants.V5_ENCODER):
        raise ValueError(
            f"v5.readout={v5.readout!r} reads A2's inputs (v2.crn=none, "
            f"v2.motion={constants.V5_ENCODER}); got crn={cfg.v2.crn!r} motion={cfg.v2.motion!r}"
        )
    if v5.queries < 1 or constants.V5_GRID_DIM % v5.queries:
        raise ValueError(
            f"v5.queries={v5.queries} must divide the {constants.V5_GRID_DIM} grid channels"
        )
    if v5.readout == constants.V5_READOUT_MEAN and (
        v5.queries != 1 or v5.keys != constants.V5_KEYS_CONTENT or v5.smil_weight > 0
    ):
        raise ValueError("v5.readout=mean (R1) has no queries, change keys or sMIL term")
