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
    # W_u init (core/docs/final/WU_ARMS.md): 0 = zero-init (every shipped arm); > 0 = N(0, std^2)
    motion_init_std: float = 0.0
    # host motion dropout (SA2V2_RUN.md §8, arm W0D): share of training bags whose W_u term is
    # zero, without SA²-v2. 0 = off (every other arm); SV2 uses stp.motion_dropout instead
    motion_dropout: float = 0.0


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
    # learning-rate multiplier of W_u alone (core/docs/final/WU_ARMS.md); 1 = one group, as shipped
    motion_lr_mult: float = 1.0


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


@dataclass
class STPConfig:
    """LaGoVAD-STP's novel component (proposal ``core/docs/final/`` §3.2).

    ``sa2`` adds Spatial Attention Aggregation to the CLIP branch: each input row then also
    carries the ``top_k`` motion-selected patch features of its frame (selected offline by
    ``core.tools.stp_patches`` with the ``motion`` prior), and ``c~_t = c_t + x^AS_t`` replaces
    ``c_t``. ``text_prompt`` picks the learnable text context: LaGoVAD's released split soft
    prompts or the CoOp prefix of ``prefix_len`` vectors. Defaults = every pre-STP model.
    """

    sa2: bool = False
    top_k: int = constants.STP_TOP_K
    motion: str = constants.STP_MOTION_RAW  # raw | ego (R7); must match the patch cache
    text_prompt: str = constants.STP_PROMPT_RELEASED  # released | prefix
    prefix_len: int = constants.STP_PREFIX_LEN
    # SA²-v2 (rule core/docs/final/SA2V2_RUN.md): rows carry 49 pooled tokens + their change
    sa2v2: bool = False
    patch_weight: float = constants.SV2_PATCH_WEIGHT
    branch_weight: float = constants.SV2_BRANCH_WEIGHT
    motion_dropout: float = constants.SV2_MOTION_DROPOUT


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
    stp: STPConfig = field(default_factory=STPConfig)

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
    if cfg.model.motion_init_std < 0:
        raise ValueError(f"model.motion_init_std must be >= 0, got {cfg.model.motion_init_std}")
    if cfg.train.motion_lr_mult <= 0:
        raise ValueError(f"train.motion_lr_mult must be > 0, got {cfg.train.motion_lr_mult}")
    if not has_motion and (cfg.model.motion_init_std > 0 or cfg.train.motion_lr_mult != 1.0):
        raise ValueError("model.motion_init_std / train.motion_lr_mult need a motion stream (W_u)")
    if not 0.0 <= cfg.model.motion_dropout < 1.0:
        raise ValueError(f"model.motion_dropout must be in [0, 1), got {cfg.model.motion_dropout}")
    if cfg.model.motion_dropout > 0 and not has_motion:
        raise ValueError("model.motion_dropout needs a motion stream (W_u)")
    if cfg.model.motion_dropout > 0 and cfg.stp.sa2v2:
        raise ValueError("stp.sa2v2 carries its own motion dropout (stp.motion_dropout): "
                         "leave model.motion_dropout at 0")


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


def validate_stp(cfg: Config) -> None:
    """Fail loud on an STP config no arm defines (proposal §3.2, ``STP_BUILD.md``).

    SA² reads the CLIP columns of an input row, so it cannot sit on a CRN-normalised row (the
    element-wise sum must stay in CLIP's embedding space) or beside the v5 grid readout.
    """
    stp = cfg.stp
    if stp.text_prompt not in constants.STP_PROMPTS:
        raise ValueError(
            f"stp.text_prompt must be one of {constants.STP_PROMPTS}, got {stp.text_prompt!r}"
        )
    if stp.motion not in constants.STP_MOTIONS:
        raise ValueError(f"stp.motion must be one of {constants.STP_MOTIONS}, got {stp.motion!r}")
    if stp.prefix_len < 1:
        raise ValueError(f"stp.prefix_len must be >= 1, got {stp.prefix_len}")
    if stp.sa2 and stp.sa2v2:
        raise ValueError("stp.sa2 and stp.sa2v2 are two versions of one module: pick one")
    if not stp.sa2 and not stp.sa2v2:
        return
    if not 1 <= stp.top_k <= constants.STP_PATCHES:
        raise ValueError(f"stp.top_k must be in [1, {constants.STP_PATCHES}], got {stp.top_k}")
    if not 0.0 <= stp.motion_dropout < 1.0:
        raise ValueError(f"stp.motion_dropout must be in [0, 1), got {stp.motion_dropout}")
    if stp.sa2v2 and stp.motion_dropout > 0 and cfg.model.motion_dim == 0:
        raise ValueError("stp.motion_dropout needs a motion stream (model.motion_dim > 0)")
    if cfg.v2.crn != constants.V2_OFF:
        raise ValueError("stp.sa2 adds CLIP patch features to c_t: it needs v2.crn=none")
    if cfg.v5.readout != constants.V5_READOUT_OFF:
        raise ValueError("stp.sa2 and the v5 grid readout cannot be combined")
    if cfg.kip.enabled:
        raise ValueError("stp.sa2 requires kip.enabled=false")
