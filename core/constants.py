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
FLOW_ZSCORE_STATS_FILENAME = (
    "zscore_stats.npz"  # per dataset dir: mean, std, provenance
)
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
WINDOW_MIN_POSITIVE = (
    1  # a window is abnormal iff it holds >= this many positive frames
)
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
# KAT-VAD v2 decision splits (proposal core/docs/v2/KAT_VAD_PROPOSAL_v2.md §7.2,
# plan .project/plans/katvad-v2-e0-e2.md P0). Frozen once by
# `python -m core.tools.freeze_splits` and committed; every v2 decision reads
# T2-val / DoTA-dev only. DoTA-eval is sealed until the final report.
# ---------------------------------------------------------------------------
V2_SPLITS_DIR = Path(__file__).resolve().parent / "splits" / "v2"
V2_SPLITS_MANIFEST_FILENAME = "SPLITS_MANIFEST.json"
V2_SPLIT_FILE_SUFFIX = ".txt"
V2_SPLIT_T2_VAL = "t2_val_sources"
V2_SPLIT_DOTA_DEV = "dota_dev"
V2_SPLIT_DOTA_EVAL = "dota_eval"
# DoTA-CAP (addendum §11, L7): DoTA restricted to the clips whose pixels CAP-DATA gives back.
# Frozen by `python -m core.tools.dota_cap freeze` into its own manifest beside the base one;
# its eval side is DoTA-CAP minus DoTA-dev, so freezing it never reads DoTA-eval's ids.
V2_SPLIT_DOTA_CAP_DEV = "dota_cap_dev"
V2_SPLIT_DOTA_CAP_EVAL = "dota_cap_eval"
V2_DOTA_CAP_MANIFEST_FILENAME = "DOTA_CAP_MANIFEST.json"
# Nexar (plan .project/plans/katvad-v2-nexar-feasibility.md N0): frozen from the census alone by
# `python -m core.tools.nexar_splits freeze` into its own manifest; nexar_test is sealed.
V2_SPLIT_NEXAR_TRAIN = "nexar_train"
V2_SPLIT_NEXAR_VAL = "nexar_val"
V2_SPLIT_NEXAR_TEST = "nexar_test"
V2_NEXAR_MANIFEST_FILENAME = "NEXAR_MANIFEST.json"
V2_DERIVED_MANIFEST_FILENAMES = (V2_DOTA_CAP_MANIFEST_FILENAME, V2_NEXAR_MANIFEST_FILENAME)
V2_SEALED_SPLITS = frozenset({V2_SPLIT_DOTA_EVAL, V2_SPLIT_DOTA_CAP_EVAL, V2_SPLIT_NEXAR_TEST})
V2_T2_VAL_FRACTION = 0.15  # of T2-train SOURCE videos, per accident type
V2_DOTA_DEV_FRACTION = 0.5  # of DoTA val clips, grouped by source YouTube video
V2_SPLIT_SEED = SEED
# Accident-share bins (proposal §4.2 step 2): share = anomaly frames / clip frames.
V2_SHARE_BIN_EDGES = (0.3, 0.5, 0.7)
V2_SHARE_BIN_LABELS = ("<30", "30-50", "50-70", ">70")

# ---------------------------------------------------------------------------
# KAT-VAD v2 motion stream: frozen VideoMAE V2 distilled (proposal §4.1,
# addendum §3/§6.1). Not in `transformers` -- the model is vendored in
# core/models/videomae_v2.py and the weights are pinned by HF commit AND
# sha256 (lesson C4); they load with torch.load(weights_only=True).
# ---------------------------------------------------------------------------
VIDEOMAE_REPO_ID = "OpenGVLab/VideoMAE2"
VIDEOMAE_REVISION = "706cc172d65ebd4dedbee3f9c0183a93df9fa125"
VIDEOMAE_ENCODER_B = "vit_b_k710_dl_from_giant"
VIDEOMAE_ENCODER_S = "vit_s_k710_dl_from_giant"
VIDEOMAE_WEIGHTS_SUBDIR = "distill"
VIDEOMAE_WEIGHTS_SHA256 = {
    VIDEOMAE_ENCODER_B: "8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec",
    VIDEOMAE_ENCODER_S: "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172",
}
# (embed_dim, depth, num_heads) of the upstream vit_{base,small}_patch16_224.
VIDEOMAE_ARCH = {
    VIDEOMAE_ENCODER_B: (768, 12, 12),
    VIDEOMAE_ENCODER_S: (384, 12, 6),
}
VIDEOMAE_PATCH_SIZE = 16
VIDEOMAE_TUBELET_SIZE = 2
VIDEOMAE_MLP_RATIO = 4
VIDEOMAE_LN_EPS = 1e-6
VIDEOMAE_CLIP_FRAMES = 16  # frames per clip the checkpoint was trained on
# ImageNet statistics: VideoMAE V2's fine-tuning pipeline normalizes with these
# (its inference example), not with the (0.5, 0.5, 0.5) of the timm `_cfg` stub.
VIDEOMAE_IMAGE_MEAN = (0.485, 0.456, 0.406)
VIDEOMAE_IMAGE_STD = (0.229, 0.224, 0.225)
# DADA-2000 = 30 fps is an ASSUMPTION from the literature (plan A1): the release
# ships PNGs. Every 3rd frame -> 10 fps, 16 frames -> a causal 1.5 s clip.
DADA_ASSUMED_FPS = 30
VIDEOMAE_CLIP_FRAME_STEP = 3
VIDEO_CACHE_DIR = CACHE_ROOT / "video"
VIDEO_MANIFEST_FILENAME = "video_manifest.json"
# Kill-switch K on T2 (addendum §6.1): subset size, bootstrap, verdict bars.
V2_K_SOURCES = 300
V2_K_BOOTSTRAP = 2000
V2_K_KILL_UPPER = 0.03  # KILL needs every Δ upper bound below this
V2_K_CONTROL_FLOOR = 0.5  # positive control: u-only CI lower bound above this
V2_K_CI = 0.95
V2_K_GATE_D0_MACRO = 0.6518  # printed beside the x-only probe, not gated
V2_K_POSITION_DEGREE = 3  # K-pos (addendum §6.2): cubic in relative position, as E2(b)

# --- v2 CRN + E0/E2(a-c) (proposal §4.2, §10.2; plan P2) ---
DOTA_FPS = 10  # DoTA ships frames at 10 fps (MoonBlvd/Detection-of-Traffic-Anomaly README)
V2_CRN_REFERENCES = ("R1", "R2", "R3", "R4")  # mean / median / robust mean / past-only
V2_CRN_WARMUP_STEPS = 8  # R4's N_w (proposal §4.2)
V2_CRN_ROBUST_KEEP = 0.5  # R3: mean of the half of steps closest (l2) to the median
V2_E2_TREND_DEGREE = 3  # f: cubic in t/T fitted on T2-val normal steps
V2_E2_TREND_CLAMP_PCT = (5.0, 95.0)  # f held at its boundary outside these percentiles of t/T
V2_E2_STRAT_BINS = 5  # position-stratified AUC: equal bins of t/T
V2_E2_COVERAGE_MIN = 0.05  # last fifth of t/T must hold >= this share of normal steps
V2_E2_BOOTSTRAP = 2000
V2_E2_CI = 0.95
V2_E2_REVERSAL_FLOOR = 0.5  # r_t AUC in both >50 % bins and stratified AUC must stay >= this

# --- v2 E1: rate-matched DoTA evaluation (proposal §7.3/§10.1; addendum §8, J1-J9) ---
V2_E1_STRIDE_A = 8  # arm A: today's protocol, whole clip
V2_E1_STRIDE_BC = 3  # arms B and C: 0.30 s/step at 10 fps, matching T2's 0.27 s
V2_E1_WINDOW = 20  # arm C: T2's training window, in steps
V2_E1_HOP = 4  # arm C: window hop, in steps
V2_E1_ARMS = ("A", "B", "C")
V2_E1_BOOTSTRAP = 10000  # §10.1: 10,000 resamples (clusters = source video, D3)
V2_E1_CI = 0.95
V2_E1_TIE_MARGIN = 0.01  # J7: both eligible and means closer than this -> B
V2_E1_REGRESSION_ATOL = 1e-4  # J9: step-level A vs phase-4 max_score

# --- v2 model inputs (architecture §4-§5, §11; plan P5) ---
# CRN and the motion scaling are parameter-free, so they are baked offline into a
# "v2 input cache" whose rows are [x_or_x~ ; c*u~/sigma_u]; the model only adds W_u.
V2_OFF = "none"  # v2.crn / v2.motion value that switches a stream off
V2_CRN_CHOICES = (V2_OFF, *V2_CRN_REFERENCES)
V2_MOTION_CHOICES = (V2_OFF, VIDEOMAE_ENCODER_B, VIDEOMAE_ENCODER_S)
V2_INPUT_MANIFEST_FILENAME = "v2_input_manifest.json"
V2_MIN_SIGMA = 1e-6  # a constant motion channel is left unscaled rather than divided by ~0
V2_NORM_EPS = 1e-12  # guards rho_u = ||W_u u|| / ||x|| against an all-zero x

# --- MM-AU / CAP-DATA Phase 0: provenance probe (.project/plans/katvad-mmau-phase0.md §3-§4) ---
MMAU_CAP_DATASET = "MMAU_CAP"
MMAU_HF_REPO = "JeffreyChou/MM-AU"
MMAU_CAP_VIDEOS = 9768  # rows of cap_text_annotations.xls; fewer cached -> partial (no branch)
MMAU_MATCH_TOP_K = 5  # descriptor retrieval candidates; the K-th is the hard null
MMAU_MATCH_EXACT = 0.99  # containment >= this: same frames, re-encoded at most
MMAU_MATCH_NEAR = 0.95  # containment >= this: same video, processed differently
MMAU_NULL_FLAG_SHARE = 0.01  # > this share of hard nulls at >= NEAR -> threshold unreliable
MMAU_COVERAGE_BAR = 0.95  # branch P needs exact coverage >= this on all DoTA and on DoTA-dev
MMAU_ALIGN_MIN_POINTS = 2  # a rate/offset fit needs at least this many matched frames
# Amendment P0b-1 (2026-10-01; after the feasibility-only group 1-10 read-out, before P0c): a near
# pair must align forward at no fewer CAP frames than query frames; a look-alike scene does not.
MMAU_NEAR_MIN_RATE = 0.9
MMAU_STREAM_BATCH_GB = 8.0  # frames on VM disk before a streamed CAP batch is encoded and deleted
BYTES_PER_GB = 1e9

# --- DoTA-CAP: DoTA pixels recovered from CAP, the motion endpoint (addendum §11, Amendment 4) ---
DOTA_CAP_DATASET = "DoTA_CAP"
# L1-L3 alignment gates on CLIP rows (DoTA_s1_ncc vs MMAU_CAP_s1_ncc), fixed before any score
DOTA_CAP_MEAN_COS = MMAU_MATCH_EXACT  # mean cosine of the aligned frames
DOTA_CAP_MIN_COS = MMAU_MATCH_NEAR  # every aligned frame
DOTA_CAP_STEP_TOL = 1  # a CAP step |dj - median dj| > this is irregular
DOTA_CAP_MAX_IRREGULAR = 0.05  # share of irregular steps a clip may have
# DoTA is native 10 fps: 16 consecutive frames = the same causal 1.5 s as DADA's every 3rd at 30 fps
DOTA_VIDEOMAE_FRAME_STEP = VIDEOMAE_CLIP_FRAME_STEP * DOTA_FPS // DADA_ASSUMED_FPS

# --- Nexar collision prediction: N0 census + frozen split (plan katvad-v2-nexar-feasibility.md) ---
NEXAR_DATASET = "Nexar"
NEXAR_HF_REPO = "nexar-ai/nexar_collision_prediction"
NEXAR_TRAIN_DIR = "train"  # train/{positive,negative}/{id}.mp4 + metadata.csv per class folder
NEXAR_CLASS_DIRS = {"positive": 1, "negative": 0}  # folder -> video label
NEXAR_METADATA_FILENAME = "metadata.csv"
NEXAR_VIDEOS = 1500  # README: 750 positive + 750 negative; fewer -> the census refuses
NEXAR_TEST_FRACTION = 0.25  # of all train videos, sealed
NEXAR_VAL_FRACTION = 0.15  # of all train videos; taken from what the test draw leaves
NEXAR_POSITION_STRATA = 3  # positives: terciles of t_event / duration, pooled over positives
NEXAR_HIST_BINS = 10  # relative-position histogram in the census
NEXAR_DURATION_BIN_S = 5.0  # duration histogram bin width
NEXAR_POST_EVENT_PROBES_S = (0.0, 1.0, 2.0)  # informational abnormal share for N1's Δ_post
NEXAR_FPS_ROUND = 1  # decimals when tabulating frame rates
# --- Nexar N1/N2/N5: span rule, caches, the two training constructions (addendum §21) ---
# D-N1: abnormal = [t_alert, t_event + this]; = median (accident -> abnormal end) of DADA's
# 1,945 annotated rows at 30 fps (1.40 s), read from the annotation, never from a Nexar score.
NEXAR_POST_EVENT_S = 1.4
NEXAR_CLIP_S1_DATASET = "Nexar_s1_ncc"  # cache/clip/<this>/{id}.npy, every native frame
NEXAR_VIDEO_S1_DATASET = "Nexar_s1"  # cache/video/<encoder>/<this>_squash/, row-aligned
NEXAR_EXTRACT_CHUNK = 64  # frames decoded + resized per step (C9)
NEXAR_TRAIN_STRIDE = FRAME_STRIDE  # training rows = s1[::8], as T2
NEXAR_EVAL_STRIDE = 3  # protocol N-B rows = s1[::3], as E1's protocol B
NEXAR_CONSTRUCTION_WHOLE = "whole"  # every nexar_train video, video-level label
NEXAR_CONSTRUCTION_WINDOW = "window"  # T2-style windows from positive videos only
NEXAR_CONSTRUCTIONS = (NEXAR_CONSTRUCTION_WHOLE, NEXAR_CONSTRUCTION_WINDOW)
NEXAR_WINDOW_LENGTH = DADA_ORIGIN_WINDOW_LENGTH  # 20 rows at s8, T2's geometry
NEXAR_WINDOW_STRIDE = DADA_ORIGIN_WINDOW_STRIDE  # hop 8
# No per-video cap: T2's cap of 4 evenly spaced windows would skip the ~11-row event of a
# 150-row Nexar video. Balance comes from DVSFeatureDataset (2 x abnormal, A9), not the cap.
NEXAR_WINDOW_MAX_PER_CLIP = 10_000
# Optimizer steps of every Nexar run = the E3 recipe's (20 epochs x 87 steps, v2_pilot metrics):
# the two constructions differ in items per epoch, so epochs = budget / steps per epoch.
NEXAR_STEP_BUDGET = 1740
# Whole videos (~150 rows at s8) under DVS's 5-clip splice reach ~750 rows and truncate_sample
# cuts at 512, which can drop the anchor; 3 clips keep a 40 s video's splice under 512.
NEXAR_WHOLE_SYN_MAX_CLIPS = 3
NEXAR_CROP_S = 8.0  # D-N2: shifted-crop length (739 / 750 positives fit all five placements)
NEXAR_CROP_PLACEMENTS = (0.1, 0.3, 0.5, 0.7, 0.9)  # event centre's relative position in a crop
NEXAR_RUN_SEEDS = (2024, 2025, 2026, 2027, 2028)
NEXAR_PILOT_SEED = 2099
# D-N10 (addendum §21): `whole` is trained first; the `window` arm is trained iff the pilot
# whole/A3 val read fires any of W1-W3 (thresholds fixed before any Nexar score existed).
NEXAR_TRIGGER_CENTRE = 0.5  # W1: reference crop placement
NEXAR_TRIGGER_EDGES = (0.1, 0.9)  # W1: placements compared to the centre
NEXAR_TRIGGER_MAX_EDGE_DROP = 0.05  # W1: centre - edge crop macro above this fires
NEXAR_TRIGGER_CLIP_AUC = 0.95  # W3: clip-level AUC at/above this (with crops < A0) fires

# --- E2(d): encoder choice on DoTA-CAP-dev (addendum §12, N1-N12; proposal §10.2) ---
V2_E2D_STRIDE = V2_E1_STRIDE_BC  # N1: protocol B rows of the stride-1 caches
V2_E2D_CRN = "R2"  # N2: A3's reference (D11)
V2_E2D_INDOMAIN_MIN = 0.10  # N6: eligible if the in-domain mean Δ reaches this ...
V2_E2D_TRANSFER_MIN = 0.03  # ... or the transfer mean Δ reaches this
V2_E2D_TIE_MARGIN = 0.02  # N7: A3 transfer Δ closer than this -> the cheaper encoder
V2_E2D_BOOTSTRAP = V2_E1_BOOTSTRAP  # N5
V2_E2D_CI = V2_E1_CI
# N11 / D6: one fixed permutation of a window's 16 frames, the same for every window
V2_D6_SHUFFLE_SEED = SEED
V2_D6_SHUFFLE_TAG = "shuf"  # cache dir suffix: <dataset>_s1_squash_shuf<seed>

# --- P6 pilot diagnostics (addendum §4, §14 O1-O7; proposal §10.1) ---
V2_GUARD_A0_MARGIN = 0.01  # O1: T2-val micro may fall at most this far below A0's
V2_D5_MARGIN = 0.02  # O6 / D5: A0 on v2 code within this of the reference on T2-val
V2_DIAG_RIDGE_ALPHA = 1.0  # O4: position probe on V^t (standardized ridge)
V2_DIAG_FOLDS = 5  # O4/O5: grouped CV folds (the core.eda probe default)

# --- E3 factorial (proposal §10.1-§10.3; addendum §18 Amendment 9, M1-M10) ---
V2_E3_ARMS = ("A0", "A1", "A2", "A3")
V2_E3_SEEDS = (2024, 2025, 2026, 2027, 2028)  # §10.1: n = 5, paired by seed (M1)
V2_E3_T95 = 2.776  # §10.1: t(0.975, df = 4), the decision interval (M4)
V2_E3_FREE_ARM = "A1"  # CRN: adoptable on a non-negative result (§10.3, M5)
V2_E3_BASE_ARM = "A0"
V2_E3_P_CAP = 0.852  # G6 (b): E2(d)'s in-domain cubic position probe on dota_cap_dev

# --- Final step (addendum §19 Amendment 10, P1-P8) ---
V2_FINAL_ADOPTED = "A3"  # P1: E3's adoption (RESULTS_E3.md), closed before any sealed read
V2_FINAL_F_ARM = "A0"  # P1: E3's best adoptable free arm
V2_FINAL_FUSION_WEIGHTS = (1.0, 2.0)  # P4: f_w = z(y) + w * z(p_T2), within clip
V2_FINAL_FUSION_PRIMARY = 2.0  # P4/P5: the weight the sentences are read at
# Amendment 11 (§20) R1: dota_eval clips with no CLIP features anywhere (lost to a FUSE unzip in
# v1, REPORT_KIP_MSAD_DOTA_PREVAD.md coverage note; DoTA pixels are gone). The only droppable clips.
V2_FINAL_DOTA_EVAL_NO_FEATURES = (
    "TNZv-NBcV5U_002389",
    "TNZv-NBcV5U_002660",
    "W6YrlYyWguc_005597",
    "W6YrlYyWguc_005927",
    "nADqn-DZ-Dc_000075",
)
V2_FINAL_DEV_TOL = 1e-4  # P7: the tool must reproduce RESULTS_E3.md §5's dev means this closely
# P7: RESULTS_E3.md §5 (DoTA-CAP-dev, seed-averaged), per weight (raw = no prior, then w = 1, 2)
V2_FINAL_DEV_P_T2 = 0.8220
V2_FINAL_DEV_ARM_MACRO = {
    "A0": (0.6768, 0.7915, 0.8194),
    "A1": (0.6577, 0.7977, 0.8245),
    "A2": (0.7536, 0.8168, 0.8264),
    "A3": (0.7576, 0.8365, 0.8424),
}
V2_FINAL_DEV_DELTA = {
    ("A3", "A0"): (0.0808, 0.0451, 0.0230),
    ("A2", "A0"): (0.0768, 0.0253, 0.0069),
    ("A3", "A1"): (0.0999, 0.0388, 0.0179),
}

# --- Exploratory probe: text-guided multi-scale CLIP windows (pending (ba); not an amendment) ---
# AnyAnomaly's WinCLIP-style windows on the `_ncc` field of view: non-overlapping g x g grids.
# Grid 1 is the whole frame, i.e. the existing CLIP row `x` -- extracted only as a row gate.
TW_GRIDS = (1, 2, 3, 5)
TW_GLOBAL_GRID = 1
TW_CACHE_DIR = CACHE_ROOT / "clip_windows"
TW_FRAMES_PER_BATCH = 8  # frames per encoder call (x 39 windows)
TW_STORE_DTYPE = "float16"
TW_TEMPERATURE = 0.01  # softmax temperature over windows: CLIP's logit scale (1/100)
TW_DOTA_STRIDE = V2_E2D_STRIDE  # protocol B rows (s1[::3]), as E2(d)
TW_CRN = V2_E2D_CRN  # every stream in the A3 form: per-clip median removed
TW_GAIN_MIN = 0.01  # GO iff the transfer gain of the text-pooled stream beyond [x;u;p] reaches this
TW_BOOTSTRAP = V2_E2D_BOOTSTRAP
TW_CI = V2_E2D_CI

# --- Exploratory: H2(a) causal post-hoc hold on y^bin (.project/plans/katvad-v2-h2a-hold.md) ---
H2A_ARM = "A3"  # §5: the rule reads the adopted arm only
H2A_PRIMARY = "max_hold"  # §4: decides; "ema" is printed only
H2A_VARIANTS = ("max_hold", "ema")
H2A_LONG_SHARE_BINS = ("50-70", ">70")  # G1: accident share >= 0.5
H2A_FUSION_WEIGHT = V2_FINAL_FUSION_PRIMARY  # G2: f_2 = z(score) + 2 z(p_T2), Final's P4
H2A_T2_SPLIT = "train"  # §3: T2 meta split the hold length is measured on (minus T2-val sources)

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
