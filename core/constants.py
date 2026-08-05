"""Project-wide constants for KAT-VAD.

Single source of truth for model dimensions, baseline hyperparameters and the
on-disk data-layout contract shared by download CLIs, extractors and dataloaders.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots (overridable via environment so Colab can point everything at Drive)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("KATVAD_DATA_ROOT", str(REPO_ROOT / "data")))
CACHE_ROOT = Path(os.environ.get("KATVAD_CACHE_ROOT", str(REPO_ROOT / "cache")))
CKPT_ROOT = Path(os.environ.get("KATVAD_CKPT_ROOT", str(REPO_ROOT / "ckpts")))
OUTPUT_ROOT = Path(os.environ.get("KATVAD_OUTPUT_ROOT", str(REPO_ROOT / "outputs")))

# ---------------------------------------------------------------------------
# Data-layout contract (spec §7; plan Phase 0)
#
#   data/{DATASET}/videos/...                     raw videos
#   data/{DATASET}/annotations/...                shipped annotation files
#   data/{DATASET}/labels_train.json              {video_id: 0|1}
#   data/{DATASET}/frame_labels_test.json         {video_id: [0,1,...]} per sampled frame
#   data/{DATASET}/defs.json                      {video_id: [desc,...]} or class-name list
#   cache/clip/{DATASET}/{video_id}.npy           (L, 512) float32 CLIP features
#   cache/flow/v1/{DATASET}/{video_id}.npy        (L, 256) float32 RAFT flow embeddings
#   cache/flow/v1/flow_projection.npz             seeded fixed linear map (A10)
#   cache/knn/{DATASET}/knn_cache.npz             KNN filler cache for DVS
#   ckpts/...                                     checkpoints (ours + LaGoVAD best.ckpt)
# ---------------------------------------------------------------------------
VIDEOS_DIRNAME = "videos"
ANNOTATIONS_DIRNAME = "annotations"
LABELS_TRAIN_FILENAME = "labels_train.json"
FRAME_LABELS_TEST_FILENAME = "frame_labels_test.json"
DEFS_FILENAME = "defs.json"
META_FILENAME = "meta.json"  # per-video scenario/class/split/window (diagnostics only)

CLIP_CACHE_DIR = CACHE_ROOT / "clip"
CACHE_PART_SUFFIX = ".part"  # in-flight write; renamed onto the target when complete
FLOW_CACHE_VERSION = "v1"
FLOW_CACHE_DIR = CACHE_ROOT / "flow" / FLOW_CACHE_VERSION
FLOW_PROJECTION_FILENAME = "flow_projection.npz"
KNN_CACHE_DIR = CACHE_ROOT / "knn"
KNN_CACHE_FILENAME = "knn_cache.npz"

TAD_DATASET = "TAD"
PREVAD_DATASET = "PreVAD"
DOTA_DATASET = "DoTA"
DADA_DATASET = "DADA2000"
MSAD_DATASET = "MSAD"  # user's traffic slice (frozen split, gates a/b/c)
MSAD_FULL_DATASET = "MSAD-full"  # entire MSAD benchmark (paper-comparable runs)
UCF_CRIME_DATASET = "UCF-Crime"

LAGOVAD_BEST_CKPT_FILENAME = "lagovad_best.ckpt"
LAGOVAD_BEST_CKPT_GDRIVE_ID = "161T7EkR64Px1cveP7_dG1xupT8IEICbM"

# ---------------------------------------------------------------------------
# Feature extraction (spec §7)
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
# Pinned HF revision (supply-chain safety). main (57c21647...) ships only
# pytorch_model.bin, which transformers 4.56 refuses to torch.load on
# torch < 2.6 (CVE-2025-32434). This sha is the SFconvertbot safetensors
# conversion (refs/pr/17, identical weights; resolved 2026-07-11).
CLIP_MODEL_REVISION = "5ef227a78de3f75873f373246dac80def63b0003"
FRAME_STRIDE = 8  # sample every 8 frames (DoTA/DADA: native fps instead)
CROP_SIZE = 224
CLIP_FEATURE_DIM = 512
# OpenAI CLIP normalization (transformers CLIPImageProcessor defaults)
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# ---------------------------------------------------------------------------
# RAFT flow extraction + A10 statistics pooling (spec §1)
# ---------------------------------------------------------------------------
RAFT_WEIGHTS_NAME = "C_T_SKHT_V2"  # torchvision Raft_Large_Weights variant
FLOW_RAFT_HEIGHT = 240  # RAFT input size (must be divisible by 8)
FLOW_RAFT_WIDTH = 320
FLOW_ANGLE_BINS = 16  # magnitude-weighted angle histogram bins
# stats vector: mag mean/std/max + u mean/std + v mean/std + angle histogram
FLOW_STATS_DIM = 7 + FLOW_ANGLE_BINS
FLOW_PROJECTION_SEED = 2024  # seeded fixed linear map stats -> FLOW_DIM (A10)
FLOW_STATS_SUFFIX = ".stats.npy"  # per-video raw stats cached next to e_O

# ---------------------------------------------------------------------------
# DVS KNN filler cache (A4)
# ---------------------------------------------------------------------------
KNN_TOP_K = 10  # neighbors stored per abnormal video

# ---------------------------------------------------------------------------
# Model dimensions (spec §2-5, baseline hidden size)
# ---------------------------------------------------------------------------
HIDDEN_DIM = 512
FLOW_DIM = 256  # d_O, RAFT flow-embedding size
PMG_LATENT_DIM = 128
ALIGN_PROJ_DIM = 128  # d_c, common dim for L_KIP_align projections
FOLDING_FACTOR = 4  # K; max shiftable channels per direction = HIDDEN_DIM // K
GATE_MLP_HIDDEN_DIM = 16
MOTION_HEAD_HIDDEN_DIM = 128

# Temporal encoder (baseline: RoFormer 2 layers / 4 heads / window 25)
TEMPORAL_LAYERS = 2
TEMPORAL_HEADS = 4
TEMPORAL_WINDOW = 25
TEMPORAL_MAX_POSITIONS = 1536  # baseline max_position_embeddings
TEMPORAL_DROPOUT = 0.1  # RoFormerConfig hidden/attention dropout default
LAYER_NORM_EPS = 1e-12  # RoFormerConfig layer_norm_eps default
TEMP_GATE_WEIGHT = 10.0  # gated-residual scale (baseline temp_gate_weight)
TEMP_GATE_INIT = 0.0  # gate alpha init (baseline temp_gate_init)

# Text encoder soft prompts (baseline)
NUM_SOFT_PROMPTS = 32
CLIP_MAX_TOKENS = 77

# Fusion (baseline: co-attention, 2 layers, 8 heads, FFN = 4x hidden)
FUSION_NUM_LAYERS = 2
FUSION_HEADS = 8
FUSION_DROPOUT = 0.1

# Heads (baseline)
SCORE_HEAD_KERNEL = 9
SCORE_HEAD_LAYERS = 1  # baseline head_num_layers (single Conv1d 512 -> 1)
ADAPTIVE_FUSE_ALPHA0 = 0.25
ADAPTIVE_FUSE_SCALE = 10.0
MULTICLASS_TEMP = 0.2

# ---------------------------------------------------------------------------
# Losses (spec §5.2, §8; baseline)
# ---------------------------------------------------------------------------
LAMBDA_REC = 1.0
LAMBDA_ALIGN = 0.1
GAMMA_KIN = 0.2
BETA_CONS = 0.5
TAU_ALIGN = 0.07
MIL_TOPK_PCT = 16  # k = max(1, L // MIL_TOPK_PCT)
SUP_MIL_TOPK_PCT = 4  # baseline sup_mil_topk_pct (pseudo-supervised MIL, L_dvs)
MUL_MIL_TOPK_PCT = 16  # baseline mul_mil_topk_pct (multi-class MIL)
CONTRASTIVE_TEMP = 0.02
CONTRASTIVE_LABEL_SMOOTHING = 0.1  # asymmetric InfoNCE label smoothing
N3_MIN_SCORE_RANGE = 0.2  # n3 mining: skip videos whose prob range < this
PSEUDO_LABEL_IGNORE = -100  # additive mask pushing non-annotated frames out of top-k
NEG_INF_MASK_VALUE = -1e4
LOGIT_EPS = 1e-6  # torch.logit clamp: BCE-on-probs as autocast-safe BCE-with-logits

# ---------------------------------------------------------------------------
# DVS — dynamic video synthesis (spec §6.1; θ = no-synthesis probability)
# ---------------------------------------------------------------------------
DVS_THETA = 0.7
DVS_THETA_EGO = 0.85
DVS_DELTA_M = 5
DVS_DELTA_M_EGO = 2
DVS_KNN_FILLER_RATIO = 0.5  # 50% KNN / 50% random normal filler

# ---------------------------------------------------------------------------
# Training (baseline hyperparameters, adopted verbatim for reproduction)
# ---------------------------------------------------------------------------
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
NUM_EPOCHS = 40
WARMUP_STEPS = 20
SEED = 2024
MAX_VIS_LEN = 512  # sliding-window length for full-video eval

# ---------------------------------------------------------------------------
# Reproduction gates (plan §1; active context 2026-07-07 amendment)
# ---------------------------------------------------------------------------
TAD_ZERO_SHOT_AUC = 89.56
GATE_A_TOLERANCE = 0.5
