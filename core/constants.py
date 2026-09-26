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
# Optional fifth dataset file (Phase 2a): {window_id: {source, start, end}}.
# Present only for a corpus rebuilt into fixed-length windows; when it is absent
# every loader behaves exactly as before. See core/docs/DATA_LAYOUT.md.
WINDOWS_FILENAME = "windows.json"
WINDOW_ID_SEPARATOR = "__w"  # "{source}__w{index:03d}"
# core.tools.subset_train: a train-source subset of a dataset dir (learning curve,
# plan katvad-t2-learning-curve.md). Written beside the filtered labels_train.json.
SUBSET_MANIFEST_FILENAME = "subset_manifest.json"
SUBSET_UNTYPED_GROUP = "__all__"  # stratum used when meta.json carries no class_name

CLIP_CACHE_DIR = CACHE_ROOT / "clip"
CACHE_PART_SUFFIX = ".part"  # in-flight write; renamed onto the target when complete
FLOW_CACHE_VERSION = "v1"
FLOW_CACHE_DIR = CACHE_ROOT / "flow" / FLOW_CACHE_VERSION
FLOW_PROJECTION_FILENAME = "flow_projection.npz"
# Option A (plan katvad-flow-zscore-option-a.md): e_O rebuilt from the v1 raw
# stats standardized with train-split moments, then the SAME projection. Bound to
# one dataset's train split -- never the default flow dir; pass --flow-dir.
FLOW_ZSCORE_CACHE_VERSION = "v2_zscore"
FLOW_ZSCORE_CACHE_DIR = CACHE_ROOT / "flow" / FLOW_ZSCORE_CACHE_VERSION
FLOW_ZSCORE_STATS_FILENAME = "zscore_stats.npz"  # per dataset dir: mean, std, provenance
FLOW_ZSCORE_MANIFEST_FILENAME = "zscore_manifest.json"
KNN_CACHE_DIR = CACHE_ROOT / "knn"
KNN_CACHE_FILENAME = "knn_cache.npz"

TAD_DATASET = "TAD"
# TAD ships extracted frames split by a top-level directory, and exactly one
# anomaly class -- the name the official test annotation and _TAD_CLS_DEFS both
# use. The train split has no annotation file, so the directory IS the label.
TAD_ABNORMAL_DIRNAME = "abnormal"
TAD_NORMAL_DIRNAME = "normal"
TAD_ABNORMAL_CLASS = "Car Accident"
PREVAD_DATASET = "PreVAD"
DOTA_DATASET = "DoTA"
DADA_DATASET = "DADA2000"
# DADA-2000 ships extracted frames split by fault-attribution directory, plus
# Cleaned_Metadata.csv (frame-level accident windows for the two fault dirs
# only -- 0_Normal_Driving carries no CSV rows, the directory IS the label).
DADA_METADATA_FILENAME = "Cleaned_Metadata.csv"
DADA_NON_EGO_FAULT_DIRNAME = "0_Non_Ego_Fault"
DADA_NORMAL_DIRNAME = "0_Normal_Driving"
DADA_EGO_FAULT_DIRNAME = "1_Ego_Fault"
DADA_CLASS_NAME = "CarAccident"  # definition key shared with DoTA (spec §7.5)
# type<N>_vid<N> folder names repeat across the three fault-attribution dirs
# (confirmed on the real archive) -- video_id is prefixed by dirname to stay
# globally unique; this separator must never appear inside a dirname itself.
DADA_ID_SEPARATOR = "__"

# --- DADA-2000, ORIGINAL release (Phase 2 / T2) ----------------------------
# A DIFFERENT corpus from DADA_DATASET above, not a variant of it. The archive
# this project trained on until 2026-09-15 trims both classes unequally; the
# original release is untrimmed (aggregate frame delta +0.30 %) and ships **no
# normal videos at all**, so T2 cuts its negatives from inside the accident
# videos. Separate dataset name => separate feature cache, always (lesson C2).
DADA_ORIGIN_DATASET = "DADA2000_orig"
# The ORIGINAL annotation: 1,962 rows, no Fault_Label column. Its sheets are
# named the opposite of their contents -- "text" holds the 1-38 type taxonomy
# and "Sheet1" holds the per-clip table -- so the sheet is DETECTED by its
# columns, never addressed by name.
DADA_ORIGIN_ANNOTATION_FILENAME = "dada标注.xlsx"
# Measured layout (DADA_ORIGIN_PHASE0.md §3.1):
#   {frames_dir}/DADA2000/{type}/{video:03d}/images/{frame:04d}.png
DADA_ORIGIN_ROOT_DIRNAME = "DADA2000"
DADA_ORIGIN_IMAGES_SUBDIR = "images"
# DadaRecord.fault_label is not optional and the original release has no such
# column; this sentinel keeps the id scheme honest about that.
DADA_ORIGIN_FAULT_SENTINEL = "origin"

# T2 geometry (.project/plans/katvad-dada-original-phase2-t2.md §4). Deliberately
# NOT the WINDOW_* defaults below: those are the trimmed archive's Phase 2a
# geometry, and a 32-frame window keeps only 72.8 % of this corpus's abnormal
# sources (C32).
#
# MEASURED, not predicted (2026-09-16, Gate W on the real archive census, 1,945
# clips). The plan's W=16 was sized from a simulation over the annotation alone
# and it FAILS: clip oracle 0.7529 against the 0.75 bar, because all-normal
# windows hold 6,528 test frames against 6,377 negative frames inside abnormal
# windows -- C33's closed form needs F_norm <= X, and it misses by 151 frames.
# The window length is the only flag that moves it (the per-clip cap spans
# 0.7527-0.7529 over a 3x range; see lessons-learned C35):
#
#     W  hop | oracle  retention  two-class  ratio | Gate W
#     16   8 | 0.7529      0.983        787   2.06 | FAIL
#     16   4 | 0.7336      0.983        956   2.42 | pass
#     20   8 | 0.7037      0.957        798   2.80 | PASS  <- adopted
#     24   8 | 0.6557      0.899        780   3.87 | FAIL (retention, ratio)
#     32  16 | 0.6127      0.728        388   5.00 | FAIL
#
# W=20 also retires the EDA's CRITICAL "micro AUC is mostly clip classification"
# verdict, which W=16 fires. Full record:
# outputs/EDA/DADA2000_orig_T2_w20s8/ (passing) beside DADA2000_orig_T2/ (failing).
DADA_ORIGIN_WINDOW_LENGTH = 20
DADA_ORIGIN_WINDOW_STRIDE = 8
# Score-head kernel for a T2 window: kernel 9 spans 45 % of 20 sampled frames and
# a head whose kernel spans the clip is a clip classifier (C27). NOT the package
# default SCORE_HEAD_KERNEL below, which stays 9 for every other corpus, so every
# T2 arm must pass model.score_head_kernel explicitly.
DADA_ORIGIN_SCORE_HEAD_KERNEL = 3
# MIL top-k on a T2 window. The default MIL_TOPK_PCT = 16 gives
# k = max(1, 20 // 16) = 1 for EVERY window, i.e. L_MIL degenerates to a plain
# max with ONE supervised frame per bag per step (EDA verdict, HIGH). The bag
# shrank, not the formula: MSAD's median clip is 86 sampled frames, where the
# same pct gives k = 5. 5 keeps k = 4, which holds the NUMBER of supervised
# frames per bag comparable to the MSAD campaign rather than its fraction.
# A declared deviation, pre-registered before the arms (plan §6), never tuned.
DADA_ORIGIN_MIL_TOPK_PCT = 5
# Test fraction of SOURCE VIDEOS (never of windows -- splitting on windows puts
# the same accident in both splits; T2 yields ~3.8 windows per video).
DADA_ORIGIN_TEST_RATIO = 0.2
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
# Names in the order flow_statistics() emits them. Derived from FLOW_ANGLE_BINS so
# the two cannot drift; the first seven are raw pixel units and unnormalized, the
# histogram is L1-normalized and therefore bounded by 1.
FLOW_STAT_NAMES = (
    "mag_mean",
    "mag_std",
    "mag_max",
    "u_mean",
    "u_std",
    "v_mean",
    "v_std",
    *(f"angle_hist_{i:02d}" for i in range(FLOW_ANGLE_BINS)),
)
FLOW_PROJECTION_SEED = 2024  # seeded fixed linear map stats -> FLOW_DIM (A10)
FLOW_STATS_SUFFIX = ".stats.npy"  # per-video raw stats cached next to e_O
# A raw stat whose train-split std falls below this is dead; standardizing it
# would divide by ~0. D1 measured 0 dead dims on T2, so one appearing is a stop.
FLOW_ZSCORE_MIN_STD = 1e-6
# Gate G0: a cached e_O must equal its own raw stats @ projection. Both are
# float32; the tolerance covers BLAS reordering, not a different projection.
FLOW_ZSCORE_G0_RTOL = 1e-4
FLOW_ZSCORE_G0_ATOL = 1e-3
# Pre-registered build bars (plan §6.1), checked on the train windows of the new
# cache. HARD bars stop the run; G2-a is reported but never re-weights anything.
FLOW_ZSCORE_G1_MEAN_TOL = 1e-3  # |mean_j| of the standardized train stats
FLOW_ZSCORE_G1_STD_BAND = (0.999, 1.001)  # std_j of the standardized train stats
FLOW_ZSCORE_G2A_V_BAND = (0.80, 1.25)  # global-mean MSE of e_O; predicted ~1
FLOW_ZSCORE_G2B_CENTRED_BAND = (0.99, 1.01)  # zero-predictor / global-mean MSE
FLOW_ZSCORE_G2C_ROUNDTRIP_BAND = (0.9, 1.1)  # mean_j E[z_j^2] / mean_d E[e_d^2]
FLOW_ZSCORE_LAMBDA_DECIMALS = 4  # lambda_rec = round(1 / V, 4), derived not swept

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

# Phase 1 arms (DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md §6). Both default to the
# baseline behavior; enabling either is an experiment, never a silent change.
# --- 1.1 DVS anchor semantics (lesson C29) -------------------------------
# "span"   = baseline: the whole spliced anchor clip is a dense positive.
# "ignore" = the anchor interior is dropped from the dense BCE; filler frames
#            stay hard negatives and `pseudo_sup_mil_loss` supplies the
#            positive pressure via its in-span top-k. Use on any corpus whose
#            abnormal clips are not ~entirely anomalous.
DVS_ANCHOR_MODE_SPAN = "span"
DVS_ANCHOR_MODE_IGNORE = "ignore"
DVS_ANCHOR_MODE_CHOICES = (DVS_ANCHOR_MODE_SPAN, DVS_ANCHOR_MODE_IGNORE)
# --- 1.2 bottom-k MIL: downward pressure inside abnormal clips -----------
BOTTOMK_MIL_TOPK_PCT = 16  # k = max(1, L // this), the lowest-k frames
BOTTOMK_WEIGHT = 0.0  # OFF by default: a new term is an arm, not a default
# --- 1.3 length-controlled evaluation (lesson C28) -----------------------
# Crop every scored clip to a common length so clip length carries no label
# information. The anchor decides which window survives; DADA-2000's accident
# sits at the end of the clip, so "start" would drop most positives.
EQUALIZE_ANCHOR_START = "start"
EQUALIZE_ANCHOR_CENTER = "center"
EQUALIZE_ANCHOR_END = "end"
EQUALIZE_ANCHOR_CHOICES = (
    EQUALIZE_ANCHOR_CENTER,
    EQUALIZE_ANCHOR_START,
    EQUALIZE_ANCHOR_END,
)

# --- 2.1 fixed-length windows (lesson C28) -------------------------------
# A corpus whose clip length predicts its label is unmeasurable, however good
# the model is: on the reconstructed DADA-2000 a detector reading only the frame
# count scores micro AUC 0.8654. Re-sharding every clip into equal-length
# windows removes the channel at the source, where --equalize-length can only
# remove it at scoring time. Defaults are DADA-2000's Phase 2a geometry
# (.project/plans/katvad-dada-phase2-corpus-rebuild.md §4.2).
WINDOW_LENGTH = 32  # sampled frames per window; T becomes constant
WINDOW_STRIDE = 16  # hop between consecutive windows (50 % overlap at the default)
WINDOW_MIN_POSITIVE = 1  # a window is abnormal iff it holds >= this many positive frames
# Cap on windows from ONE source clip, evenly spaced. A fixed hop alone gives
# each clip windows in proportion to its length; DADA-2000's normal clips are
# ~3x longer than its accident clips (raw median 139 vs 49), so uncapped
# windowing manufactured a 10:1 class imbalance the corpus never had (C32).
WINDOW_MAX_PER_CLIP = 4

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
# Evaluation score pooling (lesson C12)
#
# Micro AUC concatenates every video's frames into one ranking, so the
# *between-video* score scale enters the metric. That is signal only when the
# test set contains normal videos (MSAD). On an all-abnormal set (DoTA) the
# task is purely within-clip localization and the between-video scale is noise
# that swamps it -- LaGoVAD's own offline_dota_eval.py min-max normalizes each
# clip before accumulating for exactly this reason.
# ---------------------------------------------------------------------------
SCORE_NORM_AUTO = "auto"  # minmax when the test set is effectively all-abnormal
SCORE_NORM_NONE = "none"
SCORE_NORM_MINMAX = "minmax"  # LaGoVAD offline_dota_eval.py convention
SCORE_NORM_ZSCORE = "zscore"
SCORE_NORM_CHOICES = (
    SCORE_NORM_AUTO,
    SCORE_NORM_NONE,
    SCORE_NORM_MINMAX,
    SCORE_NORM_ZSCORE,
)
SCORE_NORM_EPS = 1e-12  # guards constant-score clips (flat model output)
# Below this share of normal videos, the between-video scale has too little to
# calibrate against and `auto` switches to per-video normalization. Measured
# split: MSAD test is 50.4 % normal, DoTA val 0.21 % (3 of 1,397 -- clips whose
# anomaly window rounds away at stride 8). Requiring *zero* normal videos would
# let those 3 silently restore the raw protocol on an all-abnormal benchmark.
SCORE_NORM_AUTO_NORMAL_FRACTION = 0.05

# ---------------------------------------------------------------------------
# EDA (core/eda, core/tools/eda.py) — dataset characterization
# ---------------------------------------------------------------------------
EDA_REPORT_JSON_FILENAME = "eda_report.json"
EDA_REPORT_MD_FILENAME = "eda_report.md"
EDA_PLOTS_DIRNAME = "plots"
# Percentiles reported for every length/count distribution.
EDA_PERCENTILES = (0, 5, 25, 50, 75, 95, 100)
# Clip-length thresholds the kernel-coverage table reports (lesson C27).
EDA_SHORT_CLIP_THRESHOLDS = (3, 5, 9, 13, 17)
# Cosine-autocorrelation lags, in sampled frames, for the CLIP feature cache.
EDA_AUTOCORR_MAX_LAG = 8
# Frame-level linear probe (RESULTS_DADA.md §10-B).
EDA_PROBE_FOLDS = 5
EDA_PROBE_MAX_ITER = 2000
EDA_PROBE_C = 1.0
EDA_PROBE_MIN_CLIPS = 10  # below this a grouped CV split is meaningless
EDA_SECTION_CORPUS = "corpus"
EDA_SECTION_LABELS = "labels"
EDA_SECTION_PROTOCOL = "protocol"
EDA_SECTION_FEATURES = "features"
EDA_SECTION_SCORES = "scores"
EDA_SECTIONS = (
    EDA_SECTION_CORPUS,
    EDA_SECTION_LABELS,
    EDA_SECTION_PROTOCOL,
    EDA_SECTION_FEATURES,
    EDA_SECTION_SCORES,
)

# ---------------------------------------------------------------------------
# Reproduction gates (plan §1; active context 2026-07-07 amendment)
# ---------------------------------------------------------------------------
TAD_ZERO_SHOT_AUC = 89.56
GATE_A_TOLERANCE = 0.5
