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
# Optional corpus file (KIM_RUN.md §49): normal window ids drawn exactly once per epoch, outside
# labels_train.json's random normal pool. Absent = the sampler every earlier corpus used
FORCED_NORMALS_FILENAME = "forced_normals.json"
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
# L_neg form (KIM_RUN.md §28 NL2). "baseline" = LaGoVAD's caption path (captions_from_definitions);
# "class" = ClassContrastLoss: class-text rows, same-class columns positive, no caption branch.
CAP_CONTRAST_MODE_BASELINE = "baseline"
CAP_CONTRAST_MODE_CLASS = "class"
# KIM_RUN.md §28.3: the class form with one fixed, meaningless unit target per source video
CAP_CONTRAST_MODE_UNIQ = "uniq"
CAP_CONTRAST_MODE_CAPTION = "caption"  # §30: one row per source video's §29.3 caption
# §32.1: LaGoVAD's caption path: [Normal caption, per-source captions] feed the caption branch
# (cap_bin MIL + DVS sup + L_MIL-align, normal -> row 0) AND L_neg; captions w/o soft prompts
CAP_CONTRAST_MODE_CAPTION_LAGOVAD = "caption_lagovad"
CAP_CONTRAST_MODE_CAPTIONS = (CAP_CONTRAST_MODE_CAPTION, CAP_CONTRAST_MODE_CAPTION_LAGOVAD)
CAP_CONTRAST_MODE_CHOICES = (CAP_CONTRAST_MODE_BASELINE, CAP_CONTRAST_MODE_CLASS,
                             CAP_CONTRAST_MODE_UNIQ, CAP_CONTRAST_MODE_CAPTION,
                             CAP_CONTRAST_MODE_CAPTION_LAGOVAD)
LNEG_CAPTION_TEMP = 0.02  # §30 C2: LaGoVAD's tau (supp. C.2), user 2026-11-17
LNEG_CAPTION_DUP_COS = 0.95  # §30.1: cross-source caption cosine at or above this is masked
LNEG_CAPTION_MASK_WARN_SHARE = 0.20  # §30.1: report to the user before training above this
LNEG_UNIQ_TARGET_SEED = 2024  # §28.3: targets are data, fixed apart from train.seed
LNEG_CLASS_TEMP = 0.07  # cosine temperature of the class form; CLIP's initial value, not tuned
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
# per-step LayerNorm on the baked motion columns before W_u (KIM_RUN.md §25, arm K5ln)
MOTION_NORM_OFF = V2_OFF
MOTION_NORM_LAYERNORM = "layernorm"
MOTION_NORM_CHOICES = (MOTION_NORM_OFF, MOTION_NORM_LAYERNORM)
MOTION_NORM_EPS = 1e-5  # torch LayerNorm's default eps

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
# Nexar training-corpus EDA (core/tools/nexar_eda.py, KIM_RUN.md §40): dataset properties only.
NEXAR_EDA_OUTPUT_DIRNAME = "nexar_eda"
NEXAR_EDA_REPORT_STEM = "nexar_eda_report"
NEXAR_EDA_NEAR_ROWS = 2  # a normal window this many rows (~0.5 s at s8) or fewer from the span
NEXAR_EDA_ROW_THRESHOLDS = (1, 3, 5, 10)  # share of positive windows with <= k abnormal rows
NEXAR_EDA_FIGURES_DIRNAME = "figures"  # nexar_eda --plots writes its PNGs here
NEXAR_EDA_FIGURE_DPI = 150
NEXAR_EDA_FIGURE_WIDTH = 7.5  # inches; one panel
NEXAR_EDA_FIGURE_HEIGHT = 4.5
NEXAR_EDA_COLOUR_TRAIN = "#4477aa"
NEXAR_EDA_COLOUR_VAL = "#ee7733"
NEXAR_EDA_COLOUR_MARK = "#cc3311"
NEXAR_EDA_COLOUR_NORMAL = "#999999"
NEXAR_EDA_GRID_ALPHA = 0.3
NEXAR_EDA_CENTER_BINS = 20  # window-centre histogram bins over [0, 1]
NEXAR_EDA_HUMP_CENTER = 0.5  # accidents sit mid-video: position ruler = -|center - 0.5| (a hump)
# Event captions for option C (core/docs/final/KIM_RUN.md §29; core.tools.nexar_caption).
# §29.3 CP3'': 4-bit NF4 (bitsandbytes). Rounds 2-6: Qwen2.5-VL-32B on a 40 GB A100; round 7
# (user, an 80 GB GPU): Qwen2.5-VL-72B (~40 GB in NF4; same class and patch size as 32B)
NEXAR_CAPTION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
NEXAR_CAPTION_QUANT_NF4 = "nf4"
NEXAR_CAPTION_QUANT_NONE = "none"
NEXAR_CAPTION_QUANT_CHOICES = (NEXAR_CAPTION_QUANT_NF4, NEXAR_CAPTION_QUANT_NONE)
NEXAR_CAPTION_QUANT = NEXAR_CAPTION_QUANT_NF4
# CP2' (§29.2): frames evenly spaced over [t_event - PRE, t_event + NEXAR_POST_EVENT_S] (3.4 s);
# Qwen merges 2 frames per time step, so 16 frames = 8 steps ~0.21 s apart, ~9 before contact
NEXAR_CAPTION_FRAMES = 32  # §29.3 round 7: 16 -> 32 (~9.4 fps over the 3.4 s window)
NEXAR_CAPTION_PRE_EVENT_S = 2.0  # a round number, not tuned; earliest nexar_train t_event = 3.03 s
# CP2: every frame is resized to this (W, H) before the processor; both are multiples of 28, so
# smart_resize keeps it and the grid is (FRAMES / 2, H / 14, W / 14). §29.3 dev round 3 (user):
# 448 x 252 -> 700 x 392 (672 x 378 is not a multiple of 28): grid (8, 28, 50) = 2,800 tokens
NEXAR_CAPTION_SIZE = (700, 392)
QWEN_VL_PATCH = 14  # Qwen2-VL video processor: patch_size; merge_size 2, temporal_patch_size 2
QWEN_VL_TEMPORAL_PATCH = 2
# CP4' (§29.1): one "detailed" sentence; PreVAD's mean is 22.9 words, 96 tokens leaves headroom
NEXAR_CAPTION_MAX_NEW_TOKENS = 128  # §29.3: room for a type + 40 words
NEXAR_CAPTION_MAX_WORDS = 40  # printed count of captions longer than this (no longer a prompt cap)
NEXAR_CAPTION_QC_SAMPLE = 30  # CP6: videos the user rates
# CP6' (§29.1): LaGoVAD Tab. B 1-5 consistency scale; pass iff the mean over the sample >= this
NEXAR_CAPTION_QC_SCORE_MIN = 1
NEXAR_CAPTION_QC_SCORE_MAX = 5
NEXAR_CAPTION_QC_MIN_MEAN = 4.0
# §29.3 CP6'': a fresh QC draw; the §29.1 draw (seed 2024) was read, so it is excluded, and so
# is the dev set the prompt is tuned on
NEXAR_CAPTION_QC_SEED = 2026
NEXAR_CAPTION_QC_SEEN_SEEDS = (2024,)
NEXAR_CAPTION_DEV_SAMPLE = 10  # §29.3 CP4'': the only videos the prompt may be tuned on
NEXAR_CAPTION_DEV_SEED = 2027
NEXAR_CAPTION_OUTPUT_DIRNAME = "nexar_captions_v2"  # §29.3; §29.1's records stay in nexar_captions
# §29.3 CP4'': dashcam prompt. Nexar positives are collisions or near-collisions (no column tells
# which), so the caption opens with one event type; the rest is the description L_neg reads
NEXAR_CAPTION_EVENT_TYPES = ("accident", "collision", "no-collision")
NEXAR_CAPTION_EVENT_UNKNOWN = "unknown"
# v6 (dev round 6, user, 2026-11-17): three types. Accident = a severe, clear crash; Collision =
# contact, or contact more likely than not; No-collision only when at least 70 % sure of no
# contact (near-collisions and swerves). The type carries the contact call; a "Contact:" clause
# is no longer asked for (a reply that still has one is parsed and dropped)
NEXAR_CAPTION_CONTACT_TYPES = {"yes": ("accident", "collision"), "no": ("no-collision",)}
NEXAR_CAPTION_NO_COLLISION_MIN_CONFIDENCE = 70  # %, written into the prompt (user)
NEXAR_CAPTION_PROMPT = (
    "This video was recorded by a dashcam mounted behind the windshield of the ego vehicle, "
    "facing forward. It contains a safety-critical traffic event. Decide whether any physical "
    "contact happens between road users, including the ego vehicle, at its front or at its "
    "side. Signs of contact: another road user fills the view and stops moving relative to the "
    "camera, the camera jolts or shakes, a door or mirror is hit, a vehicle beside the ego "
    "vehicle scrapes it, vehicles stop while touching, or debris appears. Then choose exactly "
    "one type. Accident: a severe, clearly visible crash with a hard impact. Collision: contact "
    "happens, or contact is more likely than not, but it is not a severe crash. No-collision: "
    "you are at least "
    f"{NEXAR_CAPTION_NO_COLLISION_MIN_CONFIDENCE}% sure that no contact happens, for example "
    "after hard braking, a last-moment stop or a swerve. If you are not at least "
    f"{NEXAR_CAPTION_NO_COLLISION_MIN_CONFIDENCE}% sure that there was no contact, answer "
    "Collision or Accident. Start your answer with exactly one of 'Accident:', 'Collision:' or "
    "'No-collision:', then describe the event in one sentence of at most 40 words: which road "
    "users take part (the ego vehicle, cars, trucks, buses, motorcycles, cyclists, pedestrians), "
    "what each one does, and from which direction. Name only road users that take part in the "
    "event; mention a pedestrian only if one moves into a vehicle's path. Call the camera car "
    "'the ego vehicle'. Describe only what is visible; do not guess. Output only the answer."
)
# §29.3 dev: the user's frozen reference (before round 4), mapped to the three v6 types
# (near-collision and dangerous maneuver -> no-collision); a type is right iff it is in the set
NEXAR_CAPTION_DEV_TRUTH = {
    "00013": ("collision", "no-collision"),
    "00100": ("no-collision",),
    "00125": ("collision",),
    "00362": ("accident",),
    "00474": ("no-collision",),
    "00546": ("accident",),
    "00715": ("no-collision",),
    "00765": ("collision",),
    "00921": ("no-collision",),
    "00938": ("no-collision",),
}
NEXAR_CAPTION_DEV_FREEZE_MEAN = 4.0  # freeze iff dev mean >= this and type_ok >= the count below
NEXAR_CAPTION_DEV_FREEZE_TYPE_OK = 8
# CP6: printed share of captions that name a scene attribute (the NG5 shortcut; printed only,
# the prompt no longer forbids them since §29.1)
NEXAR_CAPTION_FORBIDDEN = (
    "rain", "rainy", "snow", "snowy", "fog", "foggy", "wet", "sunny", "cloudy", "night",
    "nighttime", "dark", "daytime", "dusk", "dawn", "highway", "freeway", "urban", "city",
    "suburban", "rural", "red", "blue", "white", "black", "silver", "gray", "grey", "green",
    "yellow", "orange",
)
NEXAR_CLIP_S1_DATASET = "Nexar_s1_ncc"  # cache/clip/<this>/{id}.npy, every native frame
NEXAR_VIDEO_S1_DATASET = "Nexar_s1"  # cache/video/<encoder>/<this>_squash/, row-aligned
NEXAR_EXTRACT_CHUNK = 64  # frames decoded + resized per step (C9)
NEXAR_TRAIN_STRIDE = FRAME_STRIDE  # training rows = s1[::8], as T2
NEXAR_EVAL_STRIDE = 3  # protocol N-B rows = s1[::3], as E1's protocol B
NEXAR_CONSTRUCTION_WHOLE = "whole"  # every nexar_train video, video-level label
NEXAR_CONSTRUCTION_WINDOW = "window"  # T2-style windows from positive videos only
NEXAR_CONSTRUCTIONS = (NEXAR_CONSTRUCTION_WHOLE, NEXAR_CONSTRUCTION_WINDOW)
# KIM_RUN.md §17: `window` + windows cut from the negative videos too; opt-in, not a default build
NEXAR_CONSTRUCTION_WINDOW_NEG = "window_neg"
# KIM_RUN.md §27: `window_neg` with at most NEXAR_NEG_WINDOW_CAP windows per NEGATIVE video
# (evenly spaced); positive videos keep NEXAR_WINDOW_MAX_PER_CLIP. Opt-in, not a default build
NEXAR_CONSTRUCTION_WINDOW_NEGCAP = "window_negcap"
# §27 NC2: <= 2 x 449 negative windows vs 5,622 in-video normals = ~14 % of the normal pool
# (window_neg: ~55 %), every negative video kept. Chosen from the share, never from a score
NEXAR_NEG_WINDOW_CAP = 2
# KIM_RUN.md §38.2: `window_neg` minus every in-video normal window, negative videos capped so the
# normal pool matches the in-video normals it replaces. Opt-in, not a default build
NEXAR_CONSTRUCTION_WINDOW_NEGONLY = "window_negonly"
# KIM_RUN.md §49: `window` plus one normal window right before / right after each positive train
# video's span, drawn once per epoch (forced_normals.json). Opt-in, not a default build
NEXAR_EDGE_SIDE_PRE = "pre"
NEXAR_EDGE_SIDE_POST = "post"
NEXAR_EDGE_SIDES = (NEXAR_EDGE_SIDE_PRE, NEXAR_EDGE_SIDE_POST)
NEXAR_CONSTRUCTION_WINDOW_EDGEPRE = "window_edgepre"
NEXAR_CONSTRUCTION_WINDOW_EDGEPOST = "window_edgepost"
NEXAR_EDGE_WINDOW_ID = "{source}__edge_{side}"
NEXAR_CORPUS_CHOICES = (*NEXAR_CONSTRUCTIONS, NEXAR_CONSTRUCTION_WINDOW_NEG,
                        NEXAR_CONSTRUCTION_WINDOW_NEGCAP, NEXAR_CONSTRUCTION_WINDOW_NEGONLY,
                        NEXAR_CONSTRUCTION_WINDOW_EDGEPRE, NEXAR_CONSTRUCTION_WINDOW_EDGEPOST)
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
# Amendment 17 (addendum §28): both Nexar arms train with B's DVS anchor (A0-ign, A3-ign);
# W2's zero-shot bar is E4-0's T2-trained A3-ign. is_egocentric stays false (T2's recipe).
NEXAR_DVS_ANCHOR_MODE = DVS_ANCHOR_MODE_IGNORE
NEXAR_RUNS_DIR = "v2_nexar_ign"  # under OUTPUT_ROOT: <construction>/<arm>/s<seed>/
# Amendment 18 (addendum §29): `window` only, T2's five arms (E3's A0-A3 with `span` + E4-0's
# A3-ign), 5 seeds; read on nexar_test once and zero-shot on TAD / DADA / DoTA whole videos.
NEXAR_FULL_RUNS_DIR = "v2_nexar_window"  # under OUTPUT_ROOT: <arm>/s<seed>/ + eval/ + REPORTS/
NEXAR_FULL_ARMS = ("A0", "A1", "A2", "A3", "A3ign")
NEXAR_FULL_ANCHOR = {  # D-N16: each arm keeps its T2 recipe's DVS anchor
    "A0": DVS_ANCHOR_MODE_SPAN, "A1": DVS_ANCHOR_MODE_SPAN, "A2": DVS_ANCHOR_MODE_SPAN,
    "A3": DVS_ANCHOR_MODE_SPAN, "A3ign": DVS_ANCHOR_MODE_IGNORE,
}
NEXAR_FULL_PRIMARY = ("A3ign", "A0")  # D-N18: the one contrast that writes a sentence
NEXAR_FULL_CONTRASTS = (  # D-N19: printed beside, never decided on
    ("A1", "A0"), ("A2", "A0"), ("A3", "A0"), ("A3", "A1"), ("A3", "A2"), ("A3ign", "A3"),
)
NEXAR_TEST_MARKER = "NEXAR_TEST_OPENED.json"  # written once, before the first --final read
NEXAR_ZS_DADA_STRIDE = FRAME_STRIDE  # D-N20: DADA whole sources at s8 = T2's and the training rows
# Amendment 19 (addendum §30): is the CRN arms' crop cost the reference unit? (nexar_val only)
CRN_REFERENCE_ITEM = "item"  # §29: each scored item is its own CRN reference
CRN_REFERENCE_SOURCE = "source"  # C3: the whole source video at the eval stride (training's unit)
CRN_REFERENCES = (CRN_REFERENCE_ITEM, CRN_REFERENCE_SOURCE)
CRN_DIAG_PRIMARY = ("A3", "A2")  # C5: the arm whose Δ_ref decides, and the no-CRN arm of its gap
CRN_DIAG_PRINTED = (("A1", "A0"), ("A3ign", "A2"))  # C7: same numbers, printed only
CRN_DIAG_MOST = 0.5  # C6: fraction of the CRN crop cost closed for "explains most"
CRN_DIAG_PARITY_TOL = 1e-3  # C4(b): whole-item AUC, source vs item path
CRN_DIAG_SHARE_QUARTILES = 4  # C7: event-share bins of the per-clip cost table

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

# --- v2 E4-0: A3-ign and the base rule (core/docs/v2/PREREG_ADDENDUM.md §22) ---
V2_E4_0_CONTROL = "A3"  # trained with dvs_anchor_mode = span (E3's runs)
V2_E4_0_ARM = "A3ign"  # A3 with dvs_anchor_mode = ignore, nothing else
V2_E4_0_ANCHOR_MODE = DVS_ANCHOR_MODE_IGNORE
V2_E4_0_T2_MACRO_MARGIN = 0.01  # §11.1 guardrail: T2-val macro at most this far below A3's
V2_E4_0_PEAK_MARGIN = 0.02  # §11.1 guardrail: normal-window peak rises at most this much
V2_E4_0_TAIL_FRAMES = 30  # H2(a)'s w: T2-train median span 3.02 s at DoTA's 10 fps
V2_E4_0_TAIL_CLIPS = 277  # H2(a)'s tail-clip count on DoTA-CAP-dev at that w (regression gate)

# --- v2 second look of A3-ign on the opened sets (addendum §23, Amendment 12) ---
V2_SL_DIR = "second_look"  # under the E4-0 output dir
V2_SL_T2_TEST_WINDOWS = 1106  # S7: every T2-test read has exactly these windows
V2_SL_CAP_EVAL_CLIPS = 560  # S1: DoTA-CAP-eval

# --- v2.2 order diagnostics S1-S3 (.project/plans/katvad-v2.2-order-diagnostics.md) ---
V22_BAR_MIN = 0.01  # proposal §6.1: gate = CI low > 0 AND mean >= this (H3's bar)
V22_BOOTSTRAP = V2_E1_BOOTSTRAP  # cluster bootstrap over source videos
V22_CI = V2_E1_CI
V22_CRN = V2_E2D_CRN  # x~ = x - R2(x), u~ = u - R2(u), per source / clip (O4)
V22_DOTA_STRIDE = V2_E2D_STRIDE  # protocol B rows of DoTA-CAP
V22_BLOCK = 5  # arch §2: shuffle block, ~1.5 s
V22_MIN_SHUFFLE_BLOCK = 4  # arch §2: no adjacency-free permutation of 2 or 3 steps
V22_TWIN_GAP = 6  # arch §4: neighbours >= 6 original steps apart in the order-free twin
V22_TWIN_RESTARTS = 200  # randomised construction attempts per gap before lowering it
V22_PCA_DIM = 64  # arch §3 row 5: d_q
V22_JITTER = 3  # arch §3 rows 6-7: causal mean over t-2..t
V22_MAX_WINDOW = 5  # arch §3 row 4: m5 = causal max over t-4..t (~1.5 s)
V22_TRACE_HALF_LIFE = 3.5  # arch §3 row 4: e_t half-life in steps (~1 s)
V22_MIN_STEPS = 3  # O10: acceleration needs three steps
V22_PROBE_CHANNELS = 64  # arch §5: temporal probe width
V22_PROBE_KERNEL = 5  # arch §5: two valid convs -> receptive field 9
V22_PROBE_CONTEXT = 4  # arch §5: real context steps on each side of a training window
V22_PROBE_LR = 1e-3
V22_PROBE_WEIGHT_DECAY = 1e-2
V22_PROBE_BATCH = 64
V22_PROBE_EPOCHS = 20
V22_PROBE_SEEDS = V2_E3_SEEDS  # O7: scores averaged over these before any metric
V22_FUSION_WEIGHT = V2_FINAL_FUSION_PRIMARY  # f_2 = z(score) + 2 z(p_T2)
V22_ECHO_PEAK = (5, 6)  # S1b: steps after the span end where the exit echo would sit
V22_ECHO_BASE = (10, 15)  # S1b: post-span baseline steps
V22_ECHO_MARGIN_SD = 0.1  # S1b: admit du iff the CI lies inside +-0.1 SD of the baseline
V22_ECHO_LAGS = (3, 4, 5, 6, 7)  # S1b: printed autocorrelation lags
V22_START_STEPS = 3  # S1c: steps 0-2 vs the rest
V22_SMD_MAX = 0.1  # S1c: any summary |SMD| above this withdraws the early-onset prediction
V22_TOP_STEPS = 5  # S1a printed: share of a clip's top-5 m5 steps inside the span
V22_SHORTCUT_FOLDS = 5  # O14: corpus-shortcut probe on kappa (printed)
V22_GATE_OK = "PASS"
V22_GATE_NO = "FAIL"
# v2.2 round 7, batch 1 (proposal §R7 D10-D13; plan katvad-v2.2-two-tracks.md G1-G9)
V22_DOTA_STEP_SECONDS = V22_DOTA_STRIDE / DOTA_FPS  # 0.30 s, conditional on A1' (DoTA = 10 fps)
V22_T2_STEP_SECONDS = FRAME_STRIDE / DADA_ASSUMED_FPS  # 0.267 s, conditional on A1 (DADA = 30 fps)
V22_ONSET_POS_SECONDS = 2.0  # ΔT positives: first 2 s of the span (proposal §5.2)
V22_ONSET_NEG_SECONDS = (3.0, 0.5)  # ΔT negatives: normal steps in [t_s - 3 s, t_s - 0.5 s)
V22_ONSET_MIN_POS = 2  # ΔT eligibility, in steps
V22_ONSET_MIN_NEG = 3
V22_ONSET_MIN_ELIGIBLE = 200  # §5.3: fewer eligible CAP-dev clips -> ΔT demoted to "reported"
V22_ONSET_FUTILITY = 0.90  # §5.3: ΔT(A3-ign) or ΔT(F0 probe) at or above -> ΔT demoted
V22_PRECURSOR_SECONDS = 2.0  # R0-d: the window before onset counted as "precursor"
V22_K1_SOFT_STOP = 0.005  # K1: both T2-val Δ points below -> advisor go-ahead before M0 (D10)
V22_DETREND_DECIMALS = 9  # G10: residuals rounded so a linear ramp ties exactly (AUC 0.5)
V22_CAP_RATE_GROUPS = (1, 3)  # D7: DoTA-CAP alignment rate groups that are reported and gated
# v2.2 batch 2 = SC-1, the DF8 position swap (v2 addendum §24, Amendment 13, X1-X10)
V22_DF8_K = 3  # X2: frames kept on the near side of the span in the shifted arms
V22_DF8_MIN_CONTEXT = 10  # X2: min(pre, post) in native frames (1 s at 10 fps) for eligibility
V22_DF8_EXPECTED_CLIPS = 491  # X2/X10(d): measured on the frozen CAP-dev labels before any score
V22_DF8_FULL, V22_DF8_CONTROL, V22_DF8_EARLY, V22_DF8_LATE = "full", "control", "early", "late"
V22_DF8_ARMS = (V22_DF8_FULL, V22_DF8_CONTROL, V22_DF8_EARLY, V22_DF8_LATE)
V22_DF8_SHIFTED = (V22_DF8_CONTROL, V22_DF8_EARLY, V22_DF8_LATE)  # arms that may start past frame 0
V22_DF8_NONINFERIOR = 0.01  # X7: moved - control lower bound must exceed -this
V22_DF8_HEAD_GATE_COS = 0.999  # X10(c): re-extracted unclamped window vs the cache row
V22_DF8_SPAN_BINS_SECONDS = (2.0, 4.0)  # X9: span-length bins for the printed R1 split

# v2.2 batch 3 = SC-2, A3-B vs A3-ign (v2 addendum §25, Amendment 14, Y1-Y11)
V22_SC2_CONTROL = "A3ign"  # Y2: E4-0's runs, never retrained
V22_SC2_ARM = "A3B"  # Y1: A3-ign with VideoMAE V2-B, nothing else
V22_SC2_ENCODER = VIDEOMAE_ENCODER_B
V22_SC2_MIN_GAIN = 0.01  # Y5(c): the paired mean must reach this (proposal §7.1 trained-arm bar)
V22_SC2_T2_MACRO_MARGIN = 0.01  # Y7: T2-val macro at most this far below A3-ign's

# Amendment 20 (addendum §31): can `B` drop CRN? A2-ign vs A3-ign on T2, non-inferiority (Z1-Z10)
V2_CRNDROP_CONTROL = V2_E4_0_ARM  # Z2: E4-0's A3-ign runs, never retrained
V2_CRNDROP_ARM = "A2ign"  # Z1: A3-ign with v2.crn = none, nothing else
V2_CRNDROP_CACHE_TAG = "A2_S"  # Z1: E3's A2 input caches (T2 + DoTA-CAP), reused as baked
V2_CRNDROP_MARGIN = 0.01  # Z4-Z6: non-inferiority margin (proposal §7.1 bar, as Y5(c) / Y7)

# v2 TAD zero-shot generalization probe (v2 addendum §26, Amendment 15, T1-T10)
V2_TADZ_STRIDE = 8  # T2: TAD read at s8 = Gate T0's protocol and T2's seconds/step (assumed 30 fps)
V2_TADZ_ARM = "A3ign"  # T3: the deployed B
V2_TADZ_CONTROL = "A0"  # T3: CLIP only, same T2 recipe (E3)
V2_TADZ_PRINTED_ARM = "A3"  # T3: E3's A3 -- splits the A3-ign - A0 Δ into factorial and `ignore`
V2_TADZ_ARMS = (V2_TADZ_CONTROL, V2_TADZ_PRINTED_ARM, V2_TADZ_ARM)
V2_TADZ_REFERENCE = "T0"  # T6: LaGoVAD's released checkpoint, Gate T0
V2_TADZ_T0_MICRO = 0.7912  # T9(a): TAD_SETUP.md §8.1, re-scored here before any v2 number
V2_TADZ_T0_TOLERANCE = 0.001
V2_TADZ_VIDEOS = 100  # T9(b): tad_test_anno.json, 60 Car Accident + 40 Normal
V2_TADZ_ABNORMAL = 60
V2_TADZ_FRAME_GATE_COS = 0.999  # T9(c): Kaggle-frame CLIP re-extraction vs Drive TAD_ncc rows
V2_TADZ_FRAME_GATE_IDS = 5  # T9(c): test videos re-extracted for that gate

# v2 TAD clip-level gate s = c(v) * minmax_v(y) (v2 addendum §27, Amendment 16, G1-G10)
V2_TADG_C = 1.0  # G2: logistic regression inverse regularization, fixed
V2_TADG_MAX_ITER = 2000  # G2
V2_TADG_ARM = V2_TADZ_ARM  # G3: the scores the gate multiplies (A3-ign, §26's files)
V2_TADG_DEV_MIN_AUC = 0.60  # G5: the gate's clip-level AUC on TAD-train must reach this
V2_TADG_DEV_MIN_VIDEOS = 300  # G9(c)
V2_TADG_MSAD_TRAIN = (120, 360)  # G9(b): abnormal / normal MSAD-train videos
V2_TADG_MACRO_TOL = 1e-9  # G3: macro(s) == macro(A3-ign) per seed
V2_TADG_NORMAL_PREFIX = "Normal"  # G5: TAD ids of the normal class (Normal_012, ...)

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

# ---------------------------------------------------------------------------
# KAT-VAD v5 Stage 0 (core/docs/v5/PREREG_V5.md §1, S0-1 ... S0-12)
# ---------------------------------------------------------------------------
V5_OUTPUT_DIRNAME = "v5"
V5_STAGE0_DIRNAME = "v5_stage0"
V5_STAGE0_ENCODER = VIDEOMAE_ENCODER_S  # S0-3: the encoder v2 adopted (E2(d))
V5_STAGE0_STRIDE = V2_E1_STRIDE_BC  # S0-3: rows s1[::3], protocol B
V5_GRID_FINE = 14  # S0-4: V14 = temporal mean of the 14 x 14 token grid
V5_GRID_COARSE = 7  # S0-4: V7 = 2 x 2 average of V14 (proposal §3.2)
V5_GRID_CACHE_DIRNAME = "video_grid"
V5_GRID_DATASET_TAG = f"{DOTA_CAP_DATASET}_s{V5_STAGE0_STRIDE}_grid{V5_GRID_COARSE}"
V5_TOPK = {V5_GRID_COARSE: 5, V5_GRID_FINE: 20}  # S0-8: top-k tiles per grid size
V5_MIN_STEPS = 2  # S0-6: |A_c| and |P_c| needed for e_c
V5_BOX_FRAME_SIZE = (1280, 720)  # S0-2: DoTA annotation geometry (stauc.py image_size)
V5_ASPECT_TOLERANCE = 0.02  # S0-10 G2: CAP frame aspect vs 16:9
V5_G0_MIN_COS = 0.9999  # S0-10 G0: fc_norm(mean tokens) vs DoTA_CAP_s1_squash
V5_G1_MAX_ABS = 1e-5  # S0-10 G1: projected CLS vs image_embeds
V5_BOOTSTRAP = V2_E1_BOOTSTRAP  # S0-9
V5_BOOTSTRAP_SEED = 2026  # S0-9
V5_BOOTSTRAP_LEVEL = 0.95  # S0-9
V5_BOX_SHARE_BINS = (0.005, 0.02, 0.08)  # S0-12: < 0.5 %, 0.5-2 %, 2-8 %, > 8 %
V5_P_T2_CAP_DEV = 0.8220  # S0-12: v2's cubic position prior on CAP-dev (RESULTS_E3.md), printed
V5_D0_LAG = 8  # S0-12: CLIP frames apart for D0
V5_LAG_ROWS = (1, 3, 6)  # PREREG_V5 §1.3 L-1: row lags; 6 rows = 18 frames, no shared frame
V5_SHIFT_ROWS = (0, 2, 3)  # PREREG_V5 §1.3 L-3: grid row k + s read against step k's box (centre shift)

# ---------------------------------------------------------------------------
# KAT-VAD v5 Stages 1-2 (core/docs/v5/PREREG_V5.md §2, S1-G1 ... S1-G6, S2-1 ... S2-14)
# ---------------------------------------------------------------------------
V5_STAGE1_DIRNAME = "v5_stage1"  # under OUTPUT_ROOT/V5_OUTPUT_DIRNAME: <row>/s<seed>/ + REPORTS/
V5_ENCODER = V5_STAGE0_ENCODER  # S2-2: grids of V2-S, the encoder of v2's A2 rows
V5_GRID_TILES = V5_GRID_COARSE * V5_GRID_COARSE  # 49 tiles of V7
V5_GRID_DIM = VIDEOMAE_ARCH[V5_ENCODER][0]  # 384 channels per tile (last layer, before fc_norm)
V5_GRID_WIDTH = V5_GRID_TILES * V5_GRID_DIM  # columns one flattened grid adds to an input row
V5_SPAN_GAPS = VIDEOMAE_CLIP_FRAMES - 1  # S2-4: a window spans V5_SPAN_GAPS * step + 1 frames
V5_READOUT_OFF = "none"  # v5.readout: no grid (every v1 / v2 checkpoint)
V5_READOUT_MEAN = "mean"  # R1: the tile mean, nothing trainable in the readout
V5_READOUT_MQR = "mqr"  # R3 / R5 / R5Δ: query attention over the 49 tiles (S2-4)
V5_READOUTS = (V5_READOUT_OFF, V5_READOUT_MEAN, V5_READOUT_MQR)
V5_KEYS_CONTENT = "content"  # keys LN(G)
V5_KEYS_CHANGE = "change"  # keys LN([G ; G - G_{t-l}]) (R5Δ)
V5_KEYS = (V5_KEYS_CONTENT, V5_KEYS_CHANGE)
V5_QUERIES = 8  # S2-3 / S2-13 D2: M fixed, never searched
V5_SMIL_WEIGHT = 0.5  # S2-6 / S2-13 D2: lambda fixed
V5_SMIL_TOPK_PCT = 16  # S2-6: k = max(floor(L / 16), 1) (proposal §3.2)
V5_SMIL_EPS = 1e-6  # clamp of s-hat before BCE on probabilities
V5_HEAD_FILENAME = "v5_head.npz"  # fc_norm + A2's c / m_u / sigma_u, beside the train grids
V5_GRID_MANIFEST_FILENAME = VIDEO_MANIFEST_FILENAME  # check_or_write_manifest (Stage 0 too)
V5_NEXAR_TRAIN_STRIDE = 8  # S2-2: training rows s1[::8] (v2 Nexar `window`)
V5_NEXAR_VAL_STRIDE = 1  # S2-2 + §2.4 B1: nexar_val grids at every frame (crops start on any phase)
V5_ROWS = {  # S2-3: row -> config overrides beyond v2 Nexar A2's (S1-G4: only v5 keys differ)
    "R1": {"readout": V5_READOUT_MEAN, "queries": 1, "keys": V5_KEYS_CONTENT, "smil_weight": 0.0},
    "R3": {"readout": V5_READOUT_MQR, "queries": 1, "keys": V5_KEYS_CONTENT, "smil_weight": 0.0},
    "R5": {"readout": V5_READOUT_MQR, "queries": V5_QUERIES, "keys": V5_KEYS_CONTENT,
           "smil_weight": V5_SMIL_WEIGHT},
    "R5d": {"readout": V5_READOUT_MQR, "queries": V5_QUERIES, "keys": V5_KEYS_CHANGE,
            "smil_weight": V5_SMIL_WEIGHT},
}
V5_RUN_SEEDS = NEXAR_RUN_SEEDS[:3]  # S2-1: 2024-2026, paired with v2's Nexar runs
V5_G2_WINDOWS = 100  # S1-G2(b): real Nexar windows compared with the baked A2 rows
V5_G2_MAX_ABS = 1e-3  # S1-G2(b) tolerance (fp16 grids)
V5_G2_UNIT_MAX_ABS = 1e-6  # S1-G2(a) tolerance (fp32)
V5_G5_MAX_ABS = 1e-4  # S1-G5(a): pointing read on Stage 0's maps vs RESULTS_V5_STAGE0 §3
V5_G5C_MAX_ABS = 1e-6  # S1-G5(c): STAUC port vs the official code
V5_G6_ENTROPY_FRACTION = 0.999  # S1-G6: every query >= 0.999 log 49 = "did not leave the mean"
V5_STAUC_MAP_SIZE = CROP_SIZE  # §2.4 B3: E_t bilinearly upsampled to 224 x 224 before get_tarr
V5_CENTRE_SIGMA_FRACTION = 0.25  # S2-8: centre prior sigma = 1/4 grid width
V5_PRIMARY_LEVEL = 0.975  # S2-7: Bonferroni over P1, P2
V5_STAGE2_DIRNAME = "v5_stage2"  # under OUTPUT_ROOT/V5_OUTPUT_DIRNAME: <row>/s<seed>/ + REPORTS/
V5_CONTROL_ROW = "R1"  # S2-7: every contrast is R - R1 (mean pooling)
V5_PRIMARY_ROWS = ("R5", "R5d")  # S2-7: P1 = R5 - R1, P2 = R5Δ - R1
V5_STAGE2_ROWS = ("R3", "R5", "R5d")  # S2-3: rows Stage 2 trains (R1 was trained in Stage 1)
V5_MIN_GAIN = 0.01  # S2-7 (iii): point >= +0.01
V5_T95_N3 = 4.303  # S2-7: t(0.975, df = 2), the paired seed t (printed, never gates)
V5_NEXAR_GUARD_MARGIN = 0.01  # S2-9: nexar_val crop endpoint R - R1 >= -0.01 (seed mean)
V5_F2_WEIGHT = V2_FINAL_FUSION_PRIMARY  # S2-14: v2's beyond-position f_2 = z(y) + 2 z(p_T2)

# ---------------------------------------------------------------------------
# LaGoVAD-STP (core/docs/final/ proposal §3.2, build notes core/docs/final/STP_BUILD.md)
# ---------------------------------------------------------------------------
STP_GRID = 7  # §3.2.2: 7 x 7 grid of 32 x 32 windows on the 224 x 224 resize
STP_PATCHES = STP_GRID * STP_GRID  # 49 patch features per sampled frame
STP_TOP_K = 12  # §3.2.2 / §3.3: STPrompt's K, fixed (K-search is ablation R10)
STP_ATTN_DIM = 128  # §3.2.2: W_q, W_k in R^{128 x 512}
STP_MOTION_RAW = "raw"  # Mo = ||2P_t - P_{t-1} - P_{t+1}|| on L2-normalised patches (Eq. 1)
STP_MOTION_EGO = "ego"  # ablation R7: the frame-mean change subtracted first
STP_MOTIONS = (STP_MOTION_RAW, STP_MOTION_EGO)
STP_CELLS_RESIZED = "resized224"  # §3.2.2 default: cells cut from the 224^2 squash of the frame
STP_CELLS_NATIVE = "native"  # side ablation "crop resolution": cells cut from the native frame
STP_CELL_SOURCES = (STP_CELLS_RESIZED, STP_CELLS_NATIVE)
STP_WINDOW_GRIDS = (1, STP_GRID)  # 1 x 1 = the CLS row itself (gated against the CLIP cache)
STP_PATCH_DIM = CLIP_FEATURE_DIM  # patch features live in CLIP's image-embedding space
STP_PATCH_WIDTH = STP_TOP_K * STP_PATCH_DIM  # columns the selected patches add to an input row
STP_STORE_DTYPE = "float16"
STP_PATCH_CACHE_DIRNAME = "clip_patches"  # under CACHE_ROOT: <dataset>_s<stride>_k<K>_<motion>
STP_PATCH_MANIFEST_FILENAME = VIDEO_MANIFEST_FILENAME  # check_or_write_manifest
STP_INDEX_SUBDIR = "idx"  # per video (L, K) int16 selected-patch indices (evidence maps)
STP_MAGNITUDE_SUBDIR = "mo"  # per video (L, 49) float16 motion magnitudes
STP_NORM_EPS = 1e-6  # L2 normalisation of patch features before Mo
STP_PROMPT_RELEASED = "released"  # LaGoVAD's 32 soft prompts, 16 before + 16 after the text
STP_PROMPT_PREFIX = "prefix"  # §3.2.3: l learnable context vectors before the text (CoOp)
STP_PROMPTS = (STP_PROMPT_RELEASED, STP_PROMPT_PREFIX)
STP_PREFIX_LEN = 8  # §3.2.3: l = 8 (l-search is a side ablation)
STP_PREFIX_INIT_STD = 0.02  # §3.2.4: e_i ~ N(0, 0.02^2)
STP_NEXAR_STRIDE = V5_NEXAR_TRAIN_STRIDE  # training rows s1[::8], the rows of v2 Nexar A2's bake
STP_DOTA_STRIDE = V2_E1_STRIDE_BC  # protocol B rows s1[::3] on DoTA-CAP
STP_FRAMES_PER_BATCH = 16  # frames per CLIP call (x 50 windows)
STP_OUTPUT_DIRNAME = "stp"  # under OUTPUT_ROOT: <arm>/s<seed>/ + eval/ + REPORTS/
STP_CONTROL_ARM = V5_CONTROL_ROW  # every contrast is arm - R1 (v5 Stage 1's motion-stream host)
STP_ARMS = {  # arm -> stp.* overrides beyond the host (v2 Nexar A2's recipe, W_u motion)
    "R4": {"sa2": True, "text_prompt": STP_PROMPT_RELEASED},  # SA² only (prefix OFF)
    "R5": {"sa2": False, "text_prompt": STP_PROMPT_PREFIX},  # prefix only (no patches)
    "R6": {"sa2": True, "text_prompt": STP_PROMPT_PREFIX},  # full STP (prefix ON)
}
STP_SEEDS = V5_RUN_SEEDS  # 2024-2026; the first run is s2024 only (preliminary)
# STP_BUILD.md §5 -- first-run diagnostics (core.tools.stp_diag), printed, never gating
STP_DIAG_INIT_NORM_RTOL = 1e-4  # D-1: rebuilt ||theta_0|| vs metrics.jsonl's first prompt_norm
STP_DIAG_DEAD_SHARE = 0.5  # D-1: >= half the prompt coordinates with sqrt(v_hat) < 10 eps = dead
STP_DIAG_NOISE_FACTOR = 2.0  # D-1: SNR below 2x the zero-mean-noise SNR = sign-incoherent
STP_DIAG_MOVED = 0.1  # D-1: RMS displacement / sum(lr) below this = the LR budget was not used
# core/docs/final/WU_ARMS.md -- the motion-fusion matrix W_u: non-zero init / its own learning rate
WU_OUTPUT_DIRNAME = "wu"  # under OUTPUT_ROOT: <arm>/s<seed>/ + eval/ + REPORTS/
WU_INIT_STD = 0.003  # learned ||W_u||_F 1.41 (STP R4/R6 s2024) / sqrt(512 * 384) = 0.0032
WU_LR_MULT = 10.0  # one tenth of 1: the budget rule of the Nov 2 proposal §3.5.1 (5e-5 -> 5e-4)
WU_CONTROL_ARM = "W0"  # same code path, W_u zero-init, LR x1: every contrast is arm - W0
WU_ARMS = {  # arm -> config overrides on top of the host (v2 Nexar A2's recipe, stp off)
    "W0": {},  # control: the shipped host
    "WI": {"model.motion_init_std": WU_INIT_STD},  # W_u ~ N(0, 0.003^2), same LR
    "WL": {"train.motion_lr_mult": WU_LR_MULT},  # W_u zero-init, LR x10 on W_u only
    "WIL": {"model.motion_init_std": WU_INIT_STD, "train.motion_lr_mult": WU_LR_MULT},  # optional
}
WU_SEEDS = V5_RUN_SEEDS  # 2024-2026; the first run is s2024 only (preliminary)
DOTA_CAP_FRAMES_DIRNAME = "DoTA_CAP_frames"  # under DATA_ROOT: {dota_id}.zip, DoTA-native frames
DOTA_CAP_FRAMES_MANIFEST = "frames_manifest.json"  # written by dota_cap_frames verify on PASS
# core/docs/final/ZS_TAD_DADA.md -- zero-shot TAD / DADA-2000: Nexar-trained host vs SA² (s2024)
ZS_OUTPUT_DIRNAME = "zs"  # under OUTPUT_ROOT: eval/<tad|dada>/<arm>/ + REPORTS/
ZS_FRAMES_DIRNAME = "ZS_frames"  # <drive data>/<this>/<dataset>/{id}.zip: the s8 row frames only
ZS_FRAMES_MANIFEST = DOTA_CAP_FRAMES_MANIFEST  # written by zs_frames verify on PASS
ZS_STRIDE = FRAME_STRIDE  # TAD (§26) and DADA whole (D-N20) are read at s8
ZS_CONTROL_ARM = WU_CONTROL_ARM  # W0 = v2 Nexar A2 s2024 bit for bit (DoTA-CAP-dev 0.8110072)
ZS_SA2_ARM = "R4"  # SA² + released prompts (prefix OFF)
ZS_ARMS = (ZS_CONTROL_ARM, ZS_SA2_ARM)
ZS_SEED = STP_SEEDS[0]  # 2024 only -> every sentence is PRELIMINARY (n = 1 seed)
ZS_REPRO_TOL = 1e-3  # |W0 - v2 A2 s2024| macro above this is flagged (same function, same seed)

# --- LaGoVAD's released checkpoint read on v_final's zero-shot sets (ZS_TAD_DADA.md §7) ---
RELEASED_CKPT_OVERRIDES = ("kip.enabled=false",)  # LaGoVAD has no KIP (Gate T0's recipe)
RELEASED_CKPT_STEP = -1  # best.ckpt stores no global_step; J10 does not apply to it
ZS_LAGOVAD_ARM = "LaGoVAD"  # the released-checkpoint row
ZS_LAGOVAD_RUN = "released"  # its run name (it has no seed)
ZS_LAGOVAD_OUTPUT_DIRNAME = "zs_lagovad"  # under OUTPUT_ROOT: eval/<dota|tad|dada>/ + REPORTS/
ZS_LAGOVAD_T0_MICRO = 0.7912  # TAD Gate T0: the released ckpt's micro on TAD_ncc (TAD_SETUP §8.1)
ZS_LAGOVAD_T0_TOL = 1e-3  # |micro - T0| above this: the checkpoint or TAD_ncc drifted -> stop
ZS_LAGOVAD_INPUT_DIM = 512  # LaGoVAD reads the plain CLIP row = the first 512 columns of a bake
ZS_LAGOVAD_INPUT_ATOL = 1e-3  # plain CLIP row vs the bake's CLIP columns (fp16 storage slack)

# SA²-v2 on pooled CLIP patch tokens (proposal LaGoVAD-STP-V2 §3.2, rule docs/final/SA2V2_RUN.md)
SV2_ARM = "SV2"  # proposal row R7: scorer + text term + L_patch + head + motion dropout
SV2_SHUFFLED_ARM = "SV2_shuf"  # control C2: labels and class names permuted across bags
SV2_SEED = ZS_SEED  # 2024 only -> every sentence is PRELIMINARY (n = 1 seed)
SV2_SHUFFLE_SEED = 20261105  # the permutation of C2
SV2_GRID = STP_GRID  # 7 x 7 cells
SV2_CELLS = STP_PATCHES
SV2_SOURCE_GRID = 14  # CLIP ViT-B/16 patch tokens at 224^2
SV2_POOL = SV2_SOURCE_GRID // SV2_GRID  # each cell = the mean of 2 x 2 tokens
SV2_DIM = STP_PATCH_DIM
SV2_CELL_WIDTH = SV2_CELLS * SV2_DIM  # columns of P, and of D
SV2_ROW_WIDTH = 2 * SV2_CELL_WIDTH  # [P ; D] appended to the host row
SV2_CELL_INPUTS = 3  # [P ; P - c ; D]
SV2_TOKEN_DIM = 256
SV2_HEADS = 2
SV2_FFN_DIM = 512
SV2_PRIORS = 4  # z(Mo), z(rho), L(Mo), L(rho)
SV2_RHO_K = 5
SV2_BETA_INIT = (1.0, 0.0, 0.0, 0.0)  # starts as the per-frame motion prior
SV2_BIAS_INIT = -2.0
SV2_POS_INIT_STD = 0.02
SV2_Z_EPS = 1e-6
SV2_MIN_CLASSES = 2  # Normal + at least one category
SV2_PATCH_WEIGHT = 0.2  # lambda_p
SV2_BRANCH_WEIGHT = 0.3  # lambda_b
SV2_MOTION_DROPOUT = 0.3  # share of training bags whose W_u term is zero
SV2_TOP_CELLS = 3  # m: cells per selected step in L_patch
SV2_CACHE_DIRNAME = "sv2_cells"  # under CACHE_ROOT: <dataset>_s<stride>_g7
SV2_LEVEL_FILENAME = "level_stats.json"  # m, s of Mo and rho, Nexar training sources
SV2_OUTPUT_DIRNAME = "sv2"  # under OUTPUT_ROOT: <arm>/s<seed>/ + eval/ + REPORTS/
SV2_STORE_DTYPE = "float16"
SV2_FRAMES_PER_BATCH = 64  # frames per CLIP call: one pass per frame
SV2_G1_MAX_ABS = V5_G1_MAX_ABS  # projected CLS vs image_embeds
W0D_ARM = "W0D"  # SA2V2_RUN.md §8: W0 + SV2's motion dropout, no spatial module
W0D_MOTION_DROPOUT = SV2_MOTION_DROPOUT  # the same share SV2 trained with (rule D6)
SV3_ARM = "SV3"  # SA2V2_RUN.md §10: SV2 with c~ = c + tanh(g) x^AS, g = 0 at init (= W0D at step 0)
# Gate LR: Σ lr at the base rate is 0.04378 (§9.1), so tanh(g) could never pass 0.044. A
# coherent gradient must be able to reach g = 1 (tanh 0.76): mult >= 1 / 0.04378 = 22.8 -> 25.
# A sign-incoherent one random-walks ~ 25 * 5e-5 * sqrt(1749) = 0.05 (§10.2).
SV3_GATE_LR_MULT = 25.0
SV3_GATE_OPEN = 0.2  # |tanh g| at the last step >= this = "opened" (4x the random-walk reach)
SV4_ARM = "SV4"  # SA2V2_RUN.md §11: SV2 with c~ = c + W_p x^AS, W_p = 0 at init (paired with W0D)
SV4_USED_SHARE = 0.05  # ||W_p x^AS|| / ||c||, tail mean >= this = "used" (~ 1/3 of W0's rho_u)
SV4_TAIL_STEPS = 100  # steps averaged for the tail reads: last-batch values cannot rank arms
SV4_PAIRED_TOL = 1e-3  # |step-1 loss - W0D's| at or below this = paired with W0D (§11.2)
SV2_PROBE_DIRNAME = "probe"  # under OUTPUT_ROOT/sv2/: sv2_probe reads (§9)
SV2_PROBE_BATCHES = 8  # batches of the real epoch-0 order per live read
SV2_PROBE_SCALARS = ("beta", "gamma", "log_temperature", "bias")  # one row per coordinate
SV2_PROBE_MODULES = ("w", "w_cell", "w_prior", "pos", "layer", "branch_head", "w_p")
SV2_DIAG_KEYS = (  # logged per batch, read by the controls C2-C4, never in the total
    "sv2_beta1", "sv2_beta2", "sv2_beta3", "sv2_beta4", "sv2_gamma", "sv2_temperature",
    "sv2_bias", "sv2_w_norm", "sv2_entropy", "sv2_spread", "sv2_xas_ratio",
    "sv2_top3q_abnormal", "sv2_top3q_normal", "sv2_patch", "sv2_branch", "sv2_gate",
    "sv2_patch_share", "sv2_wp_norm",
)
SV2_ARM_ONLY_DIAG_KEYS = ("sv2_gate", "sv2_patch_share", "sv2_wp_norm")  # logged by SV3 / SV4 only

# --- DAD (Dashcam Accident Dataset) zero-shot: LaGoVAD / W0 / R4 (core/docs/final/ZS_DAD.md) ---
DAD_DATASET = "DAD"
DAD_SPLIT_DIRNAME = "testing"  # <data>/DAD/videos/testing/{positive,negative}/*.mp4
DAD_CLASSES = ("positive", "negative")
DAD_ID_PREFIX = {"positive": "pos", "negative": "neg"}  # stems may repeat across the two dirs
DAD_ANNOTATION_DIRNAME = "annotation"  # <data>/DAD/annotation/<stem>.txt
DAD_ANNOTATION_COLUMNS = 8  # frame, track id, class, x1, y1, x2, y2, accident flag
DAD_ASSUMED_FPS = 25  # measured by the census on all 466 test videos (2026-11-04; 20 was wrong)
DAD_FPS_TOL = 0.5  # |measured - 25| above this fails the census
DAD_TEST_VIDEOS = 466  # the testing split (census 2026-11-04): 165 positive, 301 negative
DAD_STRIDE = 7  # 0.28 s/step at 25 fps, the nearest to the arms' training step (Nexar s8, 0.267 s)
DAD_VIDEOMAE_FRAME_STEP = 3  # 16 x 3 frames at 25 fps = 1.8 s (T2 / Nexar step; user 2026-11-04)
DAD_TOA_FRAME = 90  # accident = the last 10 frames (1-based 91-100, 0-based 90-99), the DAD
# convention (DSA / UString); the annotation flag marks involved objects, not the accident time
DAD_JPEG_QUALITY = 95  # decode mp4 -> one JPEG folder; CLIP, V2-S and the windows all read it
DAD_CENSUS_FILENAME = "dad_census.json"
DAD_MANIFEST_FILENAME = "dad_manifest.json"
DAD_OUTPUT_SET = "dad"  # <zs_lagovad root>/eval/dad/<row>/<run>/ (core.evaluate outputs)

# --- LaGoVAD-KIM (rule core/docs/final/KIM_RUN.md) ---------------------------------------------
# Proposal symbols are NOT code names here (KIM_RUN.md §1): L_kin -> kim_mil (the KIP v1 "kin"
# loss keeps its name), p_t -> state, u_t -> vel, a_t -> acc, k_t -> kin_feat, W_p -> state_proj.
KIM_WIDTH = 128  # width of each of state / vel / acc (proposal §3.2.2: three 128-d features)
KIM_SCORER_HIDDEN = 128  # W_1: 384 -> 128, W_2: 128 -> 2 (proposal §3.2.5 parameter count)
KIM_GATE_KERNEL = 3  # gate convolution kernel, zero padding (IAM)
KIM_EPS = 1e-6  # added to kim_scale before dividing (D6)
KIM_MIL_WEIGHT = 1.0  # lambda_k (D12)
KIM_NKC_WEIGHT = 0.1  # lambda_c (D12)
KIM_NKC_TEMP = 0.1  # NKC temperature (D12)
KIM_NKC_NEGATIVES = 16  # M (D12)
KIM_ANCHOR_THRESH = 0.5  # alpha (D12)
KIM_PROTO_MOMENTUM = 0.1  # beta (D12)
KIM_WARMUP_STEPS = 219  # round(5 / 40 * 1,749): NKC + prototype updates start here (D11)
KIM_CLASSES = 2  # binary normal / abnormal (proposal §3.2.1)
KIM_ARMS = ("K0", "K0k9", "K2", "K5")  # first pass (KIM_RUN.md §3); K0k9 = host kernel 9
KIM_START_TOL = 1e-4  # step-1 mil within this of the control = same start (KIM_RUN.md §12)
KIM_DVS_ARM = "K5d"  # K5, DVS on, seam mask (KIM_RUN.md §12); not in KIM_ARMS
KIM_MAIN_ARM = "K5"  # the user's choice of main model (KIM_RUN.md §12.3), not a rule's outcome
KIM_LN_ARM = "K5ln"  # K5 + model.motion_norm=layernorm (KIM_RUN.md §25); not in KIM_ARMS
KIM_TOPK_ARM = "K5k2"  # K5 + loss.mil_topk_pct=KIM_TOPK_MIL_PCT (KIM_RUN.md §26); not in KIM_ARMS
# §26: k = 20 // 10 = 2 rows per 20-row window (mil, kim_mil, KIM anchors). Chosen from span
# statistics, never from a score: positive windows with fewer abnormal rows than k fall from
# 20.5 % (k = 4, pct 5) to 7.0 % (k = 2); k = 1 (pct 20) is a plain max (T2 EDA, HIGH).
KIM_TOPK_MIL_PCT = 10
ZS_ARM_TABLE_STEM = "zs_arm_table"  # REPORTS/<stem>_<name>.{md,json}: macro + micro, 4 sets (§25.2)
KIM_T2_OUTPUT_DIRNAME = "kim_t2"  # OUTPUT_ROOT/kim_t2: K5/s<seed>/, eval/, REPORTS/ (KIM_RUN.md §13)
KIM_T2_ROW = "K5T2"  # K5 trained on T2, its row under eval/nexar/ (KIM_RUN.md §13)
KIM_T2_EPOCHS = 20  # E3's recipe: 20 x 87 = NEXAR_STEP_BUDGET steps
# OUTPUT_ROOT/kim_whole: K5/s<seed>/, eval/, REPORTS/ (KIM_RUN.md §16)
KIM_WHOLE_OUTPUT_DIRNAME = "kim_whole"
KIM_WHOLE_ROW = "K5whole"  # K5 trained on Nexar_whole, row under eval/nexar/ (KIM_RUN.md §16)
# §16 WH1: k = n // 16 = 9 rows on a 151-row video fits the ~11-row span
# (at 5, k = 30 exceeds the span on 97.8 % of nexar_train positives)
KIM_WHOLE_MIL_TOPK_PCT = MIL_TOPK_PCT
# OUTPUT_ROOT/kim_neg: K5/s<seed>/, eval/, REPORTS/ (KIM_RUN.md §17)
KIM_NEG_OUTPUT_DIRNAME = "kim_neg"
KIM_NEG_ROW = "K5neg"  # K5 + windows from the negative videos, row under eval/nexar/ (§17)
KIM_NEG_FA_MARGIN = 0.05  # §17 NG7: false-alarm share on nexar_val negatives must fall this much
KIM_NEG_REPRO_TOL = 1e-4  # §17 NG4: K5 re-read on nexar_val vs its §13 crop endpoint
# OUTPUT_ROOT/kim_negcap: K5/s<seed>/, eval/, REPORTS/ (KIM_RUN.md §27)
KIM_NEGCAP_OUTPUT_DIRNAME = "kim_negcap"
KIM_NEGCAP_ROW = "K5nc2"  # K5 + <= 2 windows per negative video, row under eval/nexar/ (§27)
KIM_NEGCAP_DETECT_MARGIN = 0.05  # §27 NC7: detection at matched FA 0.20 must rise this much
KIM_NEGCAP_MATCHED_FA = (0.20, 0.283)  # §27 NC6(b): false-alarm rates read, from §17.1
KIM_NEGCAP_DECIDE_FA = 0.20  # §27 NC7: the matched false-alarm rate whose detection may decide
# §27 NC6(c)(d): chunk length and the pre-onset zone (seconds; Nexar val rows at NEXAR_NOMINAL_FPS)
KIM_NEGCAP_CHUNK_S = 5.0
KIM_NEGCAP_PRE_ONSET_S = 5.0
# OUTPUT_ROOT/kim_lneg: K5/s<seed>/, eval/, REPORTS/ (KIM_RUN.md §28); NL5-NL7 reuse §27's read
KIM_LNEG_OUTPUT_DIRNAME = "kim_lneg"
KIM_LNEG_ROW = "K5Lneg"  # K5 + loss.cap_contrast_mode=class, row under eval/nexar/ (§28)
KIM_LNEG_REPORT_STEM = "k5_lneg_report"
KIM_LNEG_UNIQ_OUTPUT_DIRNAME = "kim_lneg_uniq"
KIM_LNEG_UNIQ_ROW = "K5Lnegu"  # K5 + loss.cap_contrast_mode=uniq, row under eval/nexar/ (§28.3)
KIM_LNEG_UNIQ_REPORT_STEM = "k5_lneg_uniq_report"
# §28.3 U2: the sets K5Lneg fell on; TAD is printed only
KIM_LNEG_UNIQ_SETS = ("nexar", "dota", "dada", "dad")
KIM_LNEG_UNIQ_B_MIN_BELOW = 3  # B: K5Lnegu - K5 "below" on at least this many of the sets
KIM_LNEG_UNIQ_B_MAX_ABOVE = 1  # B: and K5Lnegu - K5Lneg "above" on at most this many
KIM_LNEG_UNIQ_A, KIM_LNEG_UNIQ_B, KIM_LNEG_UNIQ_MIXED = "A", "B", "MIXED"
# §30: per-video caption L_neg x DVS, 2 x 2 on K5 (K5d = KIM_DVS_ARM, read only)
KIM_LNEG_CAP_OUTPUT_DIRNAME = "kim_lneg_cap"  # OUTPUT_ROOT/<this>/<row>/{K5/s<seed>, eval}, REPORTS
KIM_LNEG_CAP_ROW = "K5c"  # K5 + loss.cap_contrast_mode=caption
KIM_LNEG_CAP_DVS_ROW = "K5dc"  # K5d + loss.cap_contrast_mode=caption
KIM_LNEG_CAP_REPORT_STEM = "k5_lneg_cap_report"
KIM_LNEG_CAP_SETS = ("nexar", "dota", "dada", "dad")  # C6 / C7 deciding sets; TAD printed
KIM_LNEG_CAP_MIN_SETS = 2  # C6: interaction above (or below) on at least this many sets
KIM_LNEG_CAP_COMPLEMENTARY = "COMPLEMENTARY"
KIM_LNEG_CAP_ANTAGONISTIC = "ANTAGONISTIC"
KIM_LNEG_CAP_NO_CLAIM = "NO CLAIM"
# §31: K5 retrained with label-0 items always Normal (lesson (dn)); same config as K5
KIM_FIXED_ARM = "K5f"
KIM_FIXED_OUTPUT_DIRNAME = "kim_fixed"  # OUTPUT_ROOT/<this>/K5f/{K5/s<seed>, eval}, REPORTS
KIM_FIXED_REPORT_NAME = "k5f_s2024"  # zs_arm_table report = zs_arm_table_<this>.{md,json}
KIM_FIXED_MUL_MIL_FLOOR = 0.01  # §31 F3: K5 ended at 0.0015 (constant target); K5f must stay above
# §31.1 (user): K5f REPLACES K5 as the reference of every new arm. KIM_MAIN_ARM stays "K5" only
# because it names the eval sub-folders (eval/<set>/K5); never use it as the control of a new arm.
KIM_REFERENCE_ARM = KIM_FIXED_ARM
# §32: the §30 2 x 2 re-run on the fixed label path, against K5f
# OUTPUT_ROOT/<this>/<row>/{K5/s<seed>, eval}, REPORTS
KIM_LNEG_CAP_FIXED_OUTPUT_DIRNAME = "kim_lneg_cap_fixed"
KIM_FIXED_DVS_ROW = "K5df"  # K5d's config on the fixed code
KIM_FIXED_CAP_ROW = "K5cf"  # K5c's config on the fixed code
KIM_FIXED_CAP_DVS_ROW = "K5dcf"  # K5dc's config on the fixed code
KIM_LNEG_CAP_FIXED_REPORT_STEM = "k5f_lneg_cap_report"
NORMAL_VIDEO_PEAK = 0.5  # a normal video whose peak score passes this counts as a false alarm
# OUTPUT_ROOT/kim_msad: eval/<row>/<run>/ + REPORTS/ (KIM_RUN.md §18, zero-shot MSAD)
MSAD_ZS_OUTPUT_DIRNAME = "kim_msad"
MSAD_ZS_VIDEOS_DIRNAME = "MSAD Videos blurred"  # the user's shortcut under Thesis-Final/data/
# only the official folders are read: the shortcut also holds VALU_MSAD_subset_2026, a modified copy
# of the same test videos (watermarks masked, some cut into `_clip_` files; §18 M2 note 2026-11-10)
MSAD_ZS_SOURCE_DIRS = ("MSAD_anomaly", "MSAD_normal_testing")
MSAD_ZS_TEST_VIDEOS = 240  # frame_labels_test.json: 120 abnormal-named + 120 normal
MSAD_ZS_TWO_CLASS = 96  # two-class at s8 (MSAD-full macro items; Explosion_3 has no positive row)
MSAD_ZS_TRAFFIC_CLASS = "Traffic_accident"  # MSAD-traffic = this class + every normal (§18 MS5)
MSAD_ZS_TRAFFIC_TWO_CLASS = 19
MSAD_ZS_NORMAL_PREFIX = "MSAD_normal_testing"
MSAD_ZS_GATE_A_MICRO = 0.8949  # §18 M0: full_gate_a, released ckpt on MSAD_ncc (0.894897)
MSAD_ZS_GATE_A_TOL = 1e-3
MSAD_ZS_PAPER_MICRO = 0.9041  # LaGoVAD's printed MSAD AUC: beside LaGoVAD's re-read only (C8b)
MSAD_ZS_ASSUMED_FPS = 30  # Nexar's V2-S geometry: 16 x 3 frames = 1.6 s at 30 fps (§18 MS3)
MSAD_ZS_IDENTITY_ATOL = 1e-3  # §18 M1: bit-identical bar on max abs; printed, not gated
MSAD_ZS_IDENTITY_MIN_COS = 0.999  # §18 M1': per-row cosine vs MSAD_ncc; set after the first read
MSAD_ZS_CENSUS_FILENAME = "msad_census.json"
MSAD_ZS_IDENTITY_FILENAME = "msad_identity.json"
MSAD_ZS_FULL_SET = "msad_full"
MSAD_ZS_TRAFFIC_SET = "msad_traffic"
KIM_WHOLE_PROPOSE_MIN_SETS = 3  # §16 WH8: above K5 on Nexar crops and >= 3 of 4 zero-shot sets
KIM_HOST_KERNEL_K9 = 9  # K0k9's model.score_head_kernel (KIM_RUN.md §3.1)
KIM_CAP_ARM = "Kcap"  # K5 without differences (KIM_RUN.md §3, §6.3); not in KIM_ARMS (runbooks unpack it)
KIM_CAP_HIDDEN = 183  # 164,535 MLP params vs 164,224 in W_state + both gates, at KIM_WIDTH (+0.2 %)
KIM_DIAG_KEYS = (  # logged per batch, never in the total
    "kim_w_out_norm", "kim_w_state_norm", "kim_vel_gate", "kim_acc_gate", "kim_scale",
    "kim_anchor_share", "kim_proto_cos", "kim_proto_ready",
)
KIM_SEAM_DIAG_KEYS = ("kim_seam_share", "kim_anchor_seam_share")  # arm K5d only (KIM_RUN.md §12)
KIM_OUTPUT_DIRNAME = "kim"  # OUTPUT_ROOT/kim: <arm>/s<seed>/, eval/{dota,dada,tad}/<arm>, REPORTS/
KIM_SEED = 2024  # first pass (KIM_RUN.md §3)
KIM_REPLICATE_SEED = 2025  # the first replication of K0 / K5 (KIM_RUN.md §6.2)
KIM_CONTROL = "K0"
KIM_W0 = "W0"  # printed beside, never the control (D3)
# pilot KP1-KP4 (KIM_RUN.md §5): every number fixed before the pilot runs
KIM_PILOT_TOP_SHARE = 0.05  # KP3: top 5 % of a video's vt order-1 differences
KIM_PILOT_EDGE_FACTOR = 2.0  # KP3 pass: edge share <= 2 x chance on every set
KIM_PILOT_KP1_BAR = 0.60  # KP1 pass: vt order-2 macro on DoTA-CAP-dev ego-involved
KIM_PILOT_INPUT_MARGIN = 0.02  # KP1: another input replaces vt iff it beats it by this on both sets
KIM_PILOT_ONSET_S = 0.5  # KP4: |peak - onset| within this many seconds ...
KIM_PILOT_ONSET_SHARE = 0.5  # ... on at least this share of abnormal clips, DoTA-CAP-dev and DADA
KIM_PILOT_DISCLOSURE = (
    "The KP pilot reads DoTA-CAP-dev / DADA / TAD with the decided control (K0 or K0k9), after the "
    "kernel read-out and before K2 / K5 train (KIM_RUN.md §5.1)."
)
# form probe (KIM_RUN.md §14): inference only, the decision margin is KIM_PILOT_INPUT_MARGIN
KIM_FORM_BASELINE_TOL = 1e-4  # recomputed vt2 / clip2 / host macros must match the recorded pilot
KIM_FORM_REPORT_DIRNAME = "form_probe"  # OUTPUT_ROOT/kim/REPORTS/form_probe
KIM_FORM_DISCLOSURE = (
    "Inference on K0 s2024, the pilot's host: K0's vt, never K5's. DoTA-CAP-dev / DADA / TAD and "
    "K5's scores on them were read before this probe. A step score's AUC is not a trained arm's "
    "gain (KIM_RUN.md §14)."
)
# Normal-text probe (KIM_RUN.md §15): K5's "Normal" definition swapped at inference, no training.
# Exactly as many sentences as the original Normal list, in the same shape (bare word, two short,
# one long), so the per-item verbalizer draws the same indices: CarAccident's text is unchanged.
NORMAL_PROBE_DASHCAM_DEFS: tuple[str, ...] = (
    "Normal driving",
    "Normal dashcam footage of a vehicle driving on the road.",
    "Normal driving video recorded by an in-car camera.",
    "Normal driving recorded by a dashcam that does not involve any collision or unusual event.",
)
NORMAL_PROBE_CLASS = "Normal"
NORMAL_PROBE_VERBALIZER_KEY = "dota"  # protocol_b_eval and dada_whole_eval both verbalize as DoTA
NORMAL_PROBE_MARGIN = 0.01  # §15: |delta| below this on both sets reads "wording does not matter"
NORMAL_PROBE_TEXT_MINOR_W = 0.9  # §15 step 1: pre-fusion w at or above this = text branch minor
NORMAL_PROBE_REPRO_TOL = 1e-4  # §15 gate: in-process original-text macro vs the recorded K5 read
NORMAL_PROBE_REPORT_DIRNAME = "normal_text_probe"  # OUTPUT_ROOT/kim/REPORTS/normal_text_probe
# KIM_RUN.md §33 (§32.1 A7): row 0 of the caption_lagovad caption set. "lagovad" = LaGoVAD's
# literal NORMAL_QUERY_CAPTION (default, K5cf); "dashcam" = §15's long dashcam sentence (K5cfn)
NORMAL_CAPTION_LAGOVAD = "lagovad"
NORMAL_CAPTION_DASHCAM = "dashcam"
NORMAL_CAPTION_CHOICES = (NORMAL_CAPTION_LAGOVAD, NORMAL_CAPTION_DASHCAM)
NORMAL_CAPTION_DASHCAM_TEXT = NORMAL_PROBE_DASHCAM_DEFS[-1]
KIM_FIXED_CAP_DASHCAM_ROW = "K5cfn"  # K5cf + loss.normal_caption=dashcam
KIM_A7_REPORT_STEM = "k5cfn_report"  # REPORTS/<this>.{md,json}
# KIM_RUN.md §34: K5df with loss.dvs_anchor_mode=ignore (K5dfi), and the same with
# data.is_egocentric=true (K5dfie). OUTPUT_ROOT/<this>/<row>/{K5/s<seed>, eval}, REPORTS
KIM_DVS_IGNORE_OUTPUT_DIRNAME = "kim_dvs_ignore"
KIM_DVS_IGNORE_ROW = "K5dfi"  # K5df + loss.dvs_anchor_mode=ignore
KIM_DVS_IGNORE_EGO_ROW = "K5dfie"  # K5dfi + data.is_egocentric=true (theta_ego, delta_m_ego)
KIM_DVS_IGNORE_REPORT_STEM = "k5dfi_report"  # REPORTS/<this>.{md,json}
KIM_DVS_IGNORE_GAIN = "GAIN"
KIM_DVS_IGNORE_COST = "COST"
KIM_DVS_IGNORE_NO_DIFFERENCE = "NO DIFFERENCE"
KIM_DVS_IGNORE_GAIN_SETS = ("dota", "dada")  # §34 R4: a GAIN must show on one of these
# KIM_RUN.md §36: the LaGoVAD trunk (CLIP only: KIM off, motion off) on K5dfi's recipe and corpus.
# OUTPUT_ROOT/<this>/<row>/{K5/s<seed>, eval/<set>}, REPORTS/
KIM_LAGOVAD_NEXAR_OUTPUT_DIRNAME = "kim_lagovad_nexar"
KIM_LAGOVAD_NEXAR_ROW = "L0dfi"  # K5dfi with kim.enabled=false, v2.motion=none, model.motion_dim=0
DOTA_CAP_ALL_DIRNAME = "dota_cap_all"  # eval/<this>: a row's dev + eval reads, concatenated
DOTA_CAP_DEV_CLIPS = 569  # the frozen DOTA_CAP_MANIFEST counts (dota_cap_dev / dota_cap_eval)
DOTA_CAP_EVAL_CLIPS = 560
# KIM_RUN.md §37: K5f and K5dfi retrained on T2 (K5T2's command, §31 code), read on whole DoTA-CAP.
KIM_T2_FIXED_OUTPUT_DIRNAME = "kim_t2_fixed"  # OUTPUT_ROOT/<this>/<row>/{K5/s<seed>, eval}, REPORTS
KIM_T2_FIXED_ROW = "K5fT2"  # K5T2's config on the §31 code (window labels fixed)
KIM_T2_FIXED_DVS_IGNORE_ROW = "K5dfiT2"  # K5fT2 + K5dfi's DVS keys (theta, theta_ego, seam, ignore)
NORMAL_PROBE_ADP_ALPHA_KEY = "bin_head.adp_alpha"  # BinaryHead's learned blend parameter

# --- pos/neg feature gap probe (core/docs/final/POSNEG_GAP.md, rule dated 2026-11-11) ---
POSNEG_GAP_RATIO = 1.5  # STOP iff mean gap(pos, N-label) < this x mean floor(pos, neg of normal frames)
POSNEG_SIGMOID_THRESHOLD = 0.5  # N-thr: frames whose sigmoid(logit) is below this are "normal"
POSNEG_FLOOR_DRAWS = 20  # random half-splits of a video's normal frames per video (floor)
POSNEG_BOOTSTRAP_DRAWS = 2000  # video-level bootstrap for the printed intervals
POSNEG_MIN_NORMAL_FRAMES = 2  # a floor needs two halves of >= 1 normal frame each
POSNEG_REPORT_STEM = "pos_neg_gap"
# --- within-video feature diversity, K5 vs K5Lneg (KIM_RUN.md §28.2, rule dated 2026-11-16) ---
FEATDIV_MIN_ROWS = 2  # FD4: a video needs this many abnormal and this many normal rows
# kim_lneg/REPORTS/<stem>/<stem>{.md,.json,_per_video.json}
FEATDIV_REPORT_STEM = "feature_diversity"
FEATDIV_METRICS = ("row_cos", "eff_rank", "sep")  # FD4 (a), (b), (c)
FEATDIV_SUPPORTED, FEATDIV_OPPOSITE = "SUPPORTED", "OPPOSITE"
FEATDIV_NOT_SUPPORTED = "NOT SUPPORTED"

# --- K5 micro headroom + MSAD clip gate (KIM_RUN.md §19, read 2026-11-11) ---
K5G_REPORT_STEM = "k5_clip_gate"  # OUTPUT_ROOT/kim/REPORTS/k5_clip_gate.{json,md}
KIM_GATE_OUTPUT_DIRNAME = "kim_gate"  # OUTPUT_ROOT/kim_gate: gates/, eval/nexar/K5/, REPORTS/ (§20)
KIM_GATE_NEXAR_TRAIN_COUNTS = (451, 449)  # §20 NX5 R4: nexar_train abnormal / normal videos
KIM_GATE_REPRO_TOL = 1e-4  # §20 NX5 R1: K5's nexar_val re-read vs §13's recorded read

# --- K5 between-video baseline, micro under per-video transforms (KIM_RUN.md §21, 2026-11-12) ---
BASEMICRO_REPORT_STEM = "k5_baseline_micro"  # REPORTS/k5_baseline_micro.{json,md}
BASEMICRO_OUTPUT_DIRNAME = "kim_baseline"  # OUTPUT_ROOT/kim_baseline: REPORTS/ only
BASEMICRO_LOW_QUANTILE = 0.2  # label-free offset: this quantile of the video's own scores
BASEMICRO_BOOTSTRAP_DRAWS = 2000  # cluster bootstrap draws (§21 BM3); POSNEG precedent
BASEMICRO_HEADROOM_MARGIN = 0.02  # §21 BM5: headroom shown iff the oracle's lower bound > this
BASEMICRO_MACRO_TOL = 1e-6  # §21 BM4 G1: macro from clip_scores.npz vs its recorded read
BASEMICRO_MICRO_TOL = 1e-4  # §21 BM4 G1: as-read micro vs the recorded micro (if recorded)
BASEMICRO_INVARIANCE_TOL = 1e-9  # §21 BM4 G2: macro under every transform == the raw macro
LAGSHIFT_REPORT_STEM = "k5_lag_shift"  # KIM_RUN §23: REPORTS/k5_lag_shift.{json,md}
LAGSHIFT_OUTPUT_DIRNAME = "kim_lag"  # OUTPUT_ROOT/kim_lag: REPORTS/ only
LAGSHIFT_GRID_STEP_S = 0.05  # §23 LS2: grid step of the lag delta, seconds
LAGSHIFT_GRID_MAX_S = 1.5  # §23 LS2: largest delta, seconds (grid 0.00 ... 1.50, 31 values)
LAGSHIFT_TIE_TOL = 1e-9  # §23 LS3: macros this close tie; the smallest delta wins
LAGSHIFT_NEXAR_MACRO_TOL = 1e-4  # §23 LS5 G1: Nexar whole macro at delta 0 vs its read-out
FAILTAIL_REPORT_STEM = "k5_fail_tail"  # KIM_RUN §24: REPORTS/k5_fail_tail.{json,md}
FAILTAIL_OUTPUT_DIRNAME = "kim_tail"  # OUTPUT_ROOT/kim_tail: REPORTS/ only
FAILTAIL_FAIL_AUC = 0.5  # §24 FT1: an item fails when its own frame AUC is below this
FAILTAIL_MIN_STRATUM = 20  # §24 FT2: a stratum is read only with at least this many items
FAILTAIL_LEAD_RATIO = 2.0  # §24 FT5: a categorical lead's fail share is >= this x the overall
FAILTAIL_COVARIATE_MARGIN = 0.10  # §24 FT5: a covariate lead's |AUC - 0.5| is at least this
FAILTAIL_DADA_ACTORS = ("pedestrian", "cyclist", "motorbike", "vehicle")  # §24 FT2, in order
FAILTAIL_DADA_EGO_TOKEN = "ego-car"  # §24 FT2: DADA ego flag = the clip's texts name this

# --- Zero-shot per-video score curves + onset-aligned mean (tools/zs_curves.py, 2026-11-15) ---
ZS_CURVES_STEM = "zs_curves"  # <out-dir>/<stem>_<name>.json + grid_<set>.png + onset_aligned.png
ZS_CURVES_PICK_WITHIN = "within"  # --pick-auc: a video's own rows (rank only)
ZS_CURVES_PICK_GLOBAL = "global"  # --pick-auc: its abnormal rows vs every normal row of the set
ZS_CURVES_PICK_AUCS = (ZS_CURVES_PICK_WITHIN, ZS_CURVES_PICK_GLOBAL)
ZS_CURVES_N_PICK = 6  # videos per set grid, spread evenly over the rank model's per-video AUC
ZS_CURVES_GRID_COLS = 3  # grid columns (rows = ceil(n / cols))
ZS_CURVES_ONSET_WINDOW_S = 5.0  # onset-aligned mean over [-w, +w] seconds around the GT onset
ZS_CURVES_ONSET_BIN_ROWS = 2  # onset bin width = this many score rows of the set (no empty bins)
ZS_CURVES_MIN_VIDEOS_PER_BIN = 10  # onset bins averaged over fewer videos are not drawn
ZS_CURVES_DPI = 130  # PNG resolution
ZS_CURVES_RAW_ALPHA = 0.3  # raw curve under the min-max curve on DoTA / DADA
ZS_CURVES_GT_ALPHA = 0.22  # ground-truth shading
ZS_CURVES_VIDEO_SUBDIR = "videos"  # --all: <out-dir>/videos/<rank>_<video>.png, one per video
ZS_CURVES_VIDEO_FIGSIZE = (8.0, 3.2)  # --all: inches per single-video figure
ZS_CURVES_ONSET_PANEL_W = 4.5  # onset_aligned.png: inches per set panel
ZS_CURVES_ONSET_MIN_W = 6.5  # ... and at least this wide, so a one-set title is not cut off
ZS_CURVES_ONSET_H = 3.8  # ... figure height
ZS_CURVES_GT_SHARE_COLOR = "dimgray"  # onset: share of rows still inside the GT (right axis)
ZS_CURVES_GT_SHARE_ALPHA = 0.08  # ... its fill
# scripts/plot_zs_curves.py: every video of a set, K5 (drawn as LaGoVAD+KIM) vs LaGoVAD
ZS_CURVES_FINAL_ROOT = OUTPUT_ROOT / "v_final"
ZS_CURVES_MAIN_LABEL = "LaGoVAD+KIM"  # K5's name on the figures (legend, title, rank)
ZS_CURVES_DEFAULT_TAG = "lagovad_kim_vs_lagovad_s2024"
ZS_CURVES_MAIN_ONLY_TAG = "lagovad_kim_s2024"  # plot_zs_curves --main-only: no LaGoVAD read
ZS_CURVES_FIGURES_DIRNAME = "figures"  # <final>/kim/REPORTS/figures/train_<train>/<set>/<tag>/
ZS_CURVES_DEFAULT_TRAIN = "nexar"
# training corpus -> its name on the figures: "LaGoVAD+KIM (trained on <name>)". T2 is cut
# from the DADA-2000 original release (KIM_RUN.md §13), so it is shown as DADA2000.
ZS_CURVES_TRAIN_LABELS = {"nexar": "Nexar", "t2": "DADA2000"}
# Nexar val as a curve set (ZS_CURVES.md "Nexar val"): nexar_eval --save-scores whole-video
# native-frame curves, raw pooling. The event sits mid-video (middle ruler 0.9735, KIM_RUN.md §13),
# so these curves show shape only. nexar_test is never drawn.
ZS_CURVES_NEXAR_SET = "nexar"
ZS_CURVES_NEXAR_DIRNAME = "nexar_curves"  # <kim_t2 root>/eval/nexar_curves/<row>/scores/*.npz
# rows per second on nexar_val's native frames: measured 29.0-30.6 (123 / 225 videos at 30.0),
# so a fixed 30 puts a 40 s video within ~2 % of its true time axis
NEXAR_NOMINAL_FPS = 30.0
# training corpus -> set -> K5 scores, relative to ZS_CURVES_FINAL_ROOT: a clip_scores.npz on
# DoTA / DADA, a run dir on TAD / DAD. T2 has no DADA read: it is trained on DADA-2000.
ZS_CURVES_MAIN_SOURCES = {
    "nexar": {
        "dota": "kim/eval/dota/K5/clip_scores.npz",
        "dada": "kim/eval/dada/K5/clip_scores.npz",
        "tad": "kim/eval/tad/K5/s2024",
        DAD_OUTPUT_SET: "kim/eval/dad/K5/s2024",
        # in-domain: K5 trains on Nexar_window (20 rows at s8, hop 8), read on nexar_val whole
        ZS_CURVES_NEXAR_SET: "kim_gate/eval/nexar/K5",
    },
    "t2": {
        "dota": "kim_t2/eval/dota/K5/clip_scores.npz",
        "tad": "kim_t2/eval/tad/K5/s2024",
        DAD_OUTPUT_SET: "kim_t2/eval/dad/K5/s2024",
        ZS_CURVES_NEXAR_SET: f"{KIM_T2_OUTPUT_DIRNAME}/eval/{ZS_CURVES_NEXAR_DIRNAME}/{KIM_T2_ROW}",
    },
}
# set -> LaGoVAD released-checkpoint scores (DoTA = protocol B, as K5), shared by every corpus
ZS_CURVES_LAGOVAD_SOURCES = {
    "dota": "zs_lagovad/eval/dota/LaGoVAD/B/clip_scores.npz",
    "dada": "zs_lagovad/eval/dada/LaGoVAD/clip_scores.npz",
    "tad": "zs_lagovad/eval/tad/LaGoVAD/released",
    DAD_OUTPUT_SET: "zs_lagovad/eval/dad/LaGoVAD/released",
    ZS_CURVES_NEXAR_SET: f"{KIM_T2_OUTPUT_DIRNAME}/eval/{ZS_CURVES_NEXAR_DIRNAME}/{ZS_LAGOVAD_ARM}",
}
# seconds per score row on each zero-shot set: DoTA protocol B is interpolated to native frames;
# DADA whole and TAD at s8 (30 fps assumed, A1); DAD at s7 of 25 fps (census 2026-11-04)
ZS_CURVES_STEP_SECONDS = {
    "dota": 1.0 / DOTA_FPS,
    "dada": ZS_STRIDE / DADA_ASSUMED_FPS,
    "tad": ZS_STRIDE / DADA_ASSUMED_FPS,
    DAD_OUTPUT_SET: DAD_STRIDE / DAD_ASSUMED_FPS,
    ZS_CURVES_NEXAR_SET: 1.0 / NEXAR_NOMINAL_FPS,
}
# onset bin width in rows where ZS_CURVES_ONSET_BIN_ROWS is too fine: Nexar's native rows are
# 1/30 s, so 8 rows = 0.27 s, the s8 bin width of DADA / TAD
ZS_CURVES_ONSET_BIN_ROWS_BY_SET = {ZS_CURVES_NEXAR_SET: 8}
ZS_CURVES_DOMAIN_ZERO_SHOT = "zero-shot"  # figure titles: the set is unseen in training ...
ZS_CURVES_DOMAIN_IN = "in-domain"  # ... or it is the training corpus's own split
ZS_CURVES_DOMAIN_MIXED = "zero-shot / in-domain"  # one model trained on this set, one not
ZS_CURVES_DOMAINS = (ZS_CURVES_DOMAIN_ZERO_SHOT, ZS_CURVES_DOMAIN_IN, ZS_CURVES_DOMAIN_MIXED)
ZS_CURVES_IN_DOMAIN_PAIRS = {("nexar", ZS_CURVES_NEXAR_SET)}  # (--train, set) read in-domain
ZS_CURVES_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:purple")  # one per --read model

# core.tools.loss_curves: training-loss curves of one or more runs from their metrics.jsonl
LOSS_CURVES_STEM = "loss_curves"  # <out-dir>/<stem>_<name>.{png,json}
LOSS_CURVES_METRICS_FILENAME = "metrics.jsonl"  # = core.train.METRICS_FILENAME (no torch here)
LOSS_CURVES_CONFIG_FILENAME = "config.yaml"  # core.train writes it beside the metrics
LOSS_CURVES_KEYS = ("total", "mil", "mul_mil", "dvs_sup", "dvs_sup_mil", "kim_mil", "kim_nkc", "lr")
# loss term -> its config weight in `total` (core/train.py: total = mil + sum w * term); None = 1
LOSS_CURVES_WEIGHTS = {
    "mil": None,
    "mul_mil": "loss.mul_weight",
    "dvs_sup": "loss.pseudo_sup_weight",
    "dvs_sup_mil": "loss.pseudo_sup_mil_weight",
    "bottomk": "loss.bottomk_weight",
    "kim_mil": "kim.mil_weight",
    "kim_nkc": "kim.nkc_weight",
}
LOSS_CURVES_SMOOTH_STEPS = 25  # centred rolling mean / std window, in logged steps
LOSS_CURVES_EDGE_STEPS = 50  # JSON: mean over the first / last this many steps a key is logged
LOSS_CURVES_COLS = 4  # panels per row
LOSS_CURVES_PANEL_W = 4.2  # inches per panel
LOSS_CURVES_PANEL_H = 3.0
LOSS_CURVES_BAND_ALPHA = 0.2  # +-1 rolling std band around the rolling mean
LOSS_CURVES_DIRNAME = "losses"  # <final>/kim/REPORTS/figures/train_<a>_vs_<b>/losses/<tag>/

# ---------------------------------------------------------------------------
# KIM_RUN.md §38 S0: Nexar source-shortcut probe (before any K5fno build)
# ---------------------------------------------------------------------------
# A logistic probe on mean-pooled CLIP windows, grouped by source video: in-video normal windows
# of positive videos (A) vs windows of negative videos (B). AUC >= this = a source shortcut
# (C38's claim gate), and K5fno (normals only from negative videos) is not built.
NEXAR_SOURCE_PROBE_SHORTCUT_AUC = 0.90
NEXAR_SOURCE_PROBE_FOLDS = EDA_PROBE_FOLDS
NEXAR_SOURCE_PROBE_STRATA_KEYS = ("scene", "weather")  # windowed meta keys (nexar_build)
NEXAR_SOURCE_PROBE_STEM = "nexar_source_probe"  # REPORTS/<stem>.{json,md}
KIM_SOURCE_PROBE_OUTPUT_DIRNAME = "kim_source_probe"  # OUTPUT_ROOT/<this>/REPORTS (§38 S0)
KIM_NEGONLY_ROW = "K5fno"  # K5f with normals only from negative videos (§38.2 N1)
KIM_NEGONLY_OUTPUT_DIRNAME = "kim_negonly"  # OUTPUT_ROOT/<this>/K5fno/s<seed>, eval/, REPORTS/
KIM_NEGONLY_REPORT_NAME = "k5fno_s2024"  # REPORTS/zs_arm_table_<this>.md (control K5f)
TAD_CATEGORY_STEM = "tad_category"  # REPORTS/<stem>_<name>.{md,json} (core.tools.tad_category_read)

# ---------------------------------------------------------------------------
# KIM_RUN.md §39: ACCIDENT (real CCTV accident clips, Kaggle picekl/accident) as the training
# corpus of K5dfi's / L0dfi's recipes, read zero-shot on TAD (H-dom)
# ---------------------------------------------------------------------------
ACCIDENT_DATASET = "ACCIDENT"  # data.dataset; same traffic definition set as Nexar (definitions)
ACCIDENT_KAGGLE_HANDLE = "picekl/accident"
ACCIDENT_METADATA_FILENAME = "metadata-real.csv"
ACCIDENT_VIDEOS_DIRNAME = "real_videos"  # <download>/real_videos/<id>.mp4 (synthetic not used)
ACCIDENT_REAL_VIDEOS = 2027
ACCIDENT_FPS = 30  # every clip resampled to 30 fps CFR by timestamp (A2): Nexar's VideoMAE geometry
ACCIDENT_CFR_TIME_EPS = 1e-6  # s; a frame stamped at exactly k/fps is CFR frame k
ACCIDENT_POST_IMPACT_S = 2.0  # A3: abnormal = [t_impact, t_impact + 2 s]; later frames are cut
ACCIDENT_FAULT_LABEL = "accident_cctv"  # DadaRecord.fault_label (not ego: is_egocentric=false)
ACCIDENT_SOURCE_SEPARATOR = "_"  # <upload>[_<segment>]_<NN>: the source drops both (accident_build)
ACCIDENT_SPLITS_DIR = Path(__file__).resolve().parent / "splits" / "accident"
ACCIDENT_SPLIT_TRAIN = "accident_train"
ACCIDENT_SPLIT_VAL = "accident_val"
ACCIDENT_SPLITS = (ACCIDENT_SPLIT_TRAIN, ACCIDENT_SPLIT_VAL)
ACCIDENT_SPLIT_MANIFEST_FILENAME = "ACCIDENT_MANIFEST.json"
ACCIDENT_VAL_FRACTION = 0.10  # S: grouped by source, stratified by type
ACCIDENT_SPLIT_SEED = 2026
ACCIDENT_CLIP_S1_DATASET = "ACCIDENT_s1_ncc"  # cache/clip/<this>/{id}.npy, CFR frames to the cut
ACCIDENT_VIDEO_S8_DATASET = "ACCIDENT_s8"  # cache/video/<encoder>/<this>_squash/, = CLIP [::8]
# Extract only what training reads: frames to t_impact + post (later ones are cut anyway), V2-S
# only at the s8 rows, and no clip whose cut is shorter than one window (it gives no window).
# The margin keeps clips whose metadata duration is within 2 rows (0.53 s) of one window.
ACCIDENT_SKIP_MARGIN_ROWS = 2
ACCIDENT_CLIP_S8_DATASET = "ACCIDENT_s8_ncc"  # plain CLIP s1[::8]: L0 rows' input + KNN key
ACCIDENT_CONSTRUCTION_WINDOW = "window"  # data/ACCIDENT_window/
ACCIDENT_BUILD_REPORT_FILENAME = "build_report.json"
# D: dedup against the TAD test videos (max CLIP cosine over frames)
ACCIDENT_DEDUP_ROW_STEP = 4  # every 4th s8 row (~1.07 s) on both sides
ACCIDENT_DEDUP_CONTROL_PERCENTILE = 10.0  # tau = this percentile of same-source pair scores
ACCIDENT_DEDUP_MAX_EXCLUDED_SHARE = 0.10  # more clips than this excluded -> STOP
ACCIDENT_DEDUP_CROSS_SAMPLE = 300  # clips whose cross-source nearest neighbour is printed
ACCIDENT_DEDUP_TOPK = 3  # TAD matches kept per clip
ACCIDENT_DEDUP_NEAR_LISTED = 20  # pairs just below tau listed for the contact sheet
ACCIDENT_DEDUP_SEED = 2026
ACCIDENT_DEDUP_SHEET_MIN_SCORE = 0.87  # high-score sheet: every clip with TAD top-1 >= this (§39)
ACCIDENT_DEDUP_SHEET_ROWS_PER_PAGE = 8  # pairs per PNG, so each page stays readable
ACCIDENT_DEDUP_SHEET_STEM = "accident_dedup_high"  # REPORTS/<stem>_pairs.json, <stem>_p<NN>.png
ACCIDENT_DEDUP_STEM = "accident_dedup"  # REPORTS/<stem>.{json,md}, <stem>_exclude_ids.txt
ACCIDENT_DEDUP_OK = "OK"
ACCIDENT_DEDUP_STOP = "STOP"
KIM_ACCIDENT_OUTPUT_DIRNAME = "kim_accident"  # OUTPUT_ROOT/<this>/<row>/{K5/s<seed>, eval}, REPORTS
KIM_ACCIDENT_ROW = "K5dfiACC"  # K5dfi's command on ACCIDENT_window (§39 A1)
KIM_ACCIDENT_L0_ROW = "L0dfiACC"  # L0dfi's command on ACCIDENT_window (§39 A1)
KIM_ACCIDENT_REPORT_STEM = "k5dfiacc_report"  # REPORTS/<this>.md: build + val + the two tables
KIM_ACCIDENT_TABLE_NAME = "k5dfiacc_s2024"  # zs_arm_table control K5dfi (rows K5dfiACC, K5f)
KIM_ACCIDENT_L0_TABLE_NAME = "l0dfiacc_s2024"  # zs_arm_table control L0dfi (row L0dfiACC)
ACCIDENT_BAKE_DIRNAME = "ACCIDENT_s8_nexar"  # cache/v2/A2_S/<this>: Nexar's A2 statistics (§39 B)
# KIM_RUN.md §42: K5dfiACC with the A2 statistics FITTED ON ACCIDENT train (§39 B used Nexar's)
KIM_ACCIDENT_STATS_ROW = "K5dfiACCs"  # K5dfiACC's command; only the input cache's statistics differ
KIM_ACCIDENT_STATS_OUTPUT_DIRNAME = "kim_accident_stats"  # OUTPUT_ROOT/<this>/<row>/K5/s<seed>
KIM_ACCIDENT_STATS_REPORT_STEM = "k5dfiaccs_report"  # REPORTS/<this>.md
KIM_ACCIDENT_STATS_TABLE_NAME = "k5dfiaccs_s2024"  # zs_arm_table control K5dfiACC (row K5dfiACCs)
ACCIDENT_SELF_BAKE_DIRNAME = "ACCIDENT_s8_acc"  # cache/v2/A2_S/<this>: fit-ids on ACCIDENT train
NEXAR_STATS_BAKE_SUFFIX = "_nexar"  # a zero-shot A2 bake with Nexar's statistics ends with this
ACCIDENT_STATS_BAKE_SUFFIX = "_acc"  # ... and its restat to ACCIDENT's statistics ends with this
V2_RESTAT_GATE_ATOL = 1e-4  # §42 R: restat(Nexar-stats bake) vs a direct bake, max |diff|
# TAD (Lv 2021, WSAL release) on Kaggle: TAD/frames/{abnormal,normal}/<id>.mp4/<k>.jpg
# Only the §39 dedup contact sheet reads it (single frames); TAD scores read cache/clip/TAD_ncc.
TAD_KAGGLE_HANDLE = "nikanvasei/traffic-anomaly-dataset-tad"

# --- KIM_RUN.md §41: K5f / K5dfi / L0f / L0dfi trained on MSAD-full, kernel 3 and 9 ---
# OUTPUT_ROOT/<this>/<row>/{K5/s<seed>, eval}, REPORTS
KIM_MSAD_TRAIN_OUTPUT_DIRNAME = "kim_msad_train"
KIM_MSAD_TRAIN_ARMS = ("K5f", "K5dfi", "L0f", "L0dfi")  # recipes (§41 R1); a row is <arm>M<kernel>
KIM_MSAD_TRAIN_ROW_INFIX = "M"
KIM_MSAD_TRAIN_KERNELS = (3, 9)  # §41 R2: both are rows; neither is picked on the MSAD test set
KIM_MSAD_TRAIN_EPOCHS = 125  # v1's MSAD budget (RESULTS_MSAD.md): 4 balanced steps/epoch = 500
KIM_MSAD_TRAIN_MIL_TOPK_PCT = MIL_TOPK_PCT  # 16: k = n // 16, LaGoVAD's and v1's MSAD value
KIM_MSAD_TRAIN_BALANCED_FACTOR = 2  # a balanced epoch draws 2 x the positive videos
KIM_MSAD_TRAIN_REPORT_STEM = "k5msad_train_report"  # REPORTS/<this>.{md,json}
# the official folders only (never VALU's modified copy)
MSAD_TRAIN_SOURCE_DIRS = ("MSAD_anomaly", "MSAD_normal_training")
MSAD_ANOMALY_DIR = "MSAD_anomaly"  # a train id's label must agree with the folder it is found in
MSAD_VIDEO_SUFFIXES = (".mp4", ".avi")  # one normal-train video is an .avi (perdestrian_street7)
# the release spells 26 files "perdestrian_street<n>"; labels_train.json and MSAD_ncc say
# "pedestrian_street<n>"
MSAD_FILE_STEM_ALIASES = {"perdestrian_street": "pedestrian_street"}
MSAD_TRAIN_VIDEOS = 480  # labels_train.json: 120 abnormal + 360 normal (V2_TADG_MSAD_TRAIN)
MSAD_TRAIN_ABNORMAL = 120
MSAD_DECODE_MAX_WORKERS = 8  # default cap on parallel decode threads (one frame each)
MSAD_TRAIN_CENSUS_FILENAME = "msad_train_census.json"
# cache/v2/A2_S/<this>: the statistics fitted on MSAD train (§41 B) + the 480 train videos baked
MSAD_TRAIN_BAKE_DIRNAME = "MSAD-full_train_s8_msad"
MSAD_TEST_BAKE_DIRNAME = "MSAD-full_test_s8_msad"  # the 240 test videos, MSAD's statistics
MSAD_TAD_BAKE_DIRNAME = "TAD_s8_msad"  # TAD (100 videos), MSAD's statistics, for the TAD reads
# --- pre-onset rise on nexar_val (KIM_RUN.md §43; printed only) ---
PRE_ONSET_BIN_S = 0.5  # event-locked curve bin width (s)
PRE_ONSET_WINDOW_S = (-10.0, 3.0)  # event-locked curve range relative to the alert (s)
PRE_ONSET_FAR_S = KIM_NEGCAP_PRE_ONSET_S  # "far" = more than this before the alert (§27)
PRE_ONSET_ZONES_S = ((-5.0, -2.0), (-2.0, -1.0), (-1.0, 0.0))  # zones read against negatives
PRE_ONSET_RISE_S = 2.0  # rise = (mean[-this, 0) - far) / (span - far)
PRE_ONSET_MIN_CONTRAST = 1e-3  # a video enters the rise read only if span - far exceeds this
PRE_ONSET_BOOTSTRAP = V2_E1_BOOTSTRAP
PRE_ONSET_BOOTSTRAP_SEED = V5_BOOTSTRAP_SEED
PRE_ONSET_LEVEL = V5_BOOTSTRAP_LEVEL
PRE_ONSET_CONTROL = "K5f"
PRE_ONSET_REPORT_STEM = "pre_onset_read"
PRE_ONSET_DIRNAME = "pre_onset"  # <kim_fixed>/REPORTS/<this>/
PRE_ONSET_FIGURE_DPI = 150
PRE_ONSET_FIGURE_SIZE = (13.0, 4.8)  # inches, two panels
PRE_ONSET_SCALED_YLIM = (-1.0, 2.0)  # display only: low-contrast videos explode the ratio
PRE_ONSET_ROW_COLOURS = ("#222222", "#cc3311", "#4477aa", "#228833", "#aa3377", "#999999")
# §43 P2: row -> its saved nexar_val read, relative to OUTPUT_ROOT (Thesis-Final/outputs)
PRE_ONSET_ROWS = {
    "K5f": "kim_fixed/K5f/eval/nexar/K5f",
    "K5df": "kim_lneg_cap_fixed/K5df/eval/nexar/K5df",
    "K5dfi": "kim_dvs_ignore/K5dfi/eval/nexar/K5dfi",
    "K5fno": "kim_negonly/K5fno/eval/nexar/K5fno",
    "LaGoVAD": "kim_t2/eval/nexar_curves/LaGoVAD",
}
PRE_ONSET_REPORT_DIR = "kim_fixed/REPORTS/pre_onset"  # relative to OUTPUT_ROOT
# §43.1 (post-hoc, printed): the same pre-onset reads on DoTA-CAP (all 1,129) and DADA whole,
# on a fixed composition (videos with >= PREROLL s before onset), rescaled per video
PRE_ONSET_FIXED_PREROLL_S = 5.0  # pre-roll a video needs to enter the fixed reads
PRE_ONSET_FIXED_FAR_S = 4.0  # far = mean before -this (the fixed reads; >= 1 s of far by the above)
PRE_ONSET_FIXED_ZONES_S = ((-5.0, -4.0), (-4.0, -3.0), (-3.0, -2.0), (-2.0, -1.0), (-1.0, 0.0))
PRE_ONSET_RESCALE_MIN_CONTRAST = 0.05  # span - far below this: the video is left out of a ratio
PRE_ONSET_COST_ZONE_S = (-2.0, 0.0)  # macro recomputed without the normal frames in this zone
PRE_ONSET_PREROLL_STRATA_S = (2.0, 4.0)  # min-max level of the last 1 s by pre-roll: <2, 2-4, >=4
PRE_ONSET_ZS_ROWS = {  # set -> row -> its saved read, relative to OUTPUT_ROOT
    "dota": {
        "K5f": "kim_cap_eval/K5f/eval/dota_cap_all/K5/clip_scores.npz",
        "K5dfi": "kim_cap_eval/K5dfi/eval/dota_cap_all/K5/clip_scores.npz",
        "L0dfi": "kim_lagovad_nexar/L0dfi/eval/dota_cap_all/K5/clip_scores.npz",
        "LaGoVAD": "kim_cap_eval/LaGoVAD/eval/dota_cap_all/B/clip_scores.npz",
    },
    "dada": {
        "K5f": "kim_fixed/K5f/eval/dada/K5/clip_scores.npz",
        "K5dfi": "kim_dvs_ignore/K5dfi/eval/dada/K5/clip_scores.npz",
        "L0dfi": "kim_lagovad_nexar/L0dfi/eval/dada/K5/clip_scores.npz",
        "LaGoVAD": "zs_lagovad/eval/dada/LaGoVAD/clip_scores.npz",
    },
}

# --- KIM_RUN §44.2: where raw micro is lost on DoTA-CAP / DADA (micro_level_read) ---
MICRO_LEVEL_LOW = 0.5  # L2: a clip is low when its accident frames rank below the set's median normal
MICRO_LEVEL_GATE_ATOL = 1e-6  # L2: positive-weighted mean of A_c must equal the pooled raw micro
MICRO_LEVEL_REPORT_STEM = "micro_level_read"
MICRO_LEVEL_REPORT_DIR = "kim_fixed/REPORTS/micro_level"  # relative to OUTPUT_ROOT
MICRO_LEVEL_ROWS = ("K5f", "K5dfi")  # L1, the first is the overlap's reference

# --- KIM_RUN §46: does the ego gap in level shrink on T2-trained rows? (ego_gap_read) ---
EGO_GAP_PAIRS = (("K5f", "K5fT2"), ("K5dfi", "K5dfiT2"))  # EG1 (Nexar row, T2 row), first = primary
EGO_GAP_REFERENCE_LOW = {  # EG2 gate: §44.3 low share (ego-uninvolved, ego-involved)
    "K5f": (0.434, 0.091),
    "K5dfi": (0.359, 0.056),
}
EGO_GAP_GATE_ATOL = 1e-3  # EG2
EGO_GAP_THIRD = 0.114  # EG4: a third of K5f's §44.3 gap (0.343)
EGO_GAP_REPORT_STEM = "ego_gap_read"

# --- KIM_RUN §47: does box size explain the ego gap in level? (box_size_read) ---
BOX_SIZE_FRAME_AREA = 1280 * 720  # BS2: DoTA frame in px, the boxes' coordinate frame
BOX_SIZE_EDGES = (0.01, 0.02, 0.04, 0.08, 0.16)  # BS4: fixed size bins
BOX_SIZE_THIRD = 1.0 / 3.0  # BS5: share of G_raw the size mix must carry
BOX_SIZE_RHO_FLOOR = 0.2  # BS5: ρ upper bound below this (and including 0) = no relation
BOX_SIZE_ROWS = ("K5f", "K5dfi", "K5fT2", "K5dfiT2")  # BS1, first = primary
BOX_SIZE_REPORT_STEM = "box_size_read"

# --- KIM_RUN §49: edge normal windows and bottom-k MIL on K5f (k5_edge_report) ---
KIM_EDGE_PRE_ROW = "K5fepre"  # K5f + pre-span edge windows, once per epoch (ED2, ED3)
KIM_EDGE_POST_ROW = "K5fepost"  # K5f + post-span edge windows, once per epoch
KIM_BOTTOMK_ROW = "K5fbk"  # K5f + loss.bottomk_weight / bottomk_topk_pct (ED5)
KIM_EDGE_ROWS = (KIM_EDGE_PRE_ROW, KIM_EDGE_POST_ROW, KIM_BOTTOMK_ROW)
KIM_EDGE_OUTPUT_DIRNAME = "kim_edge"  # OUTPUT_ROOT/<this>/<row>/{K5/s<seed>, eval}, REPORTS/
KIM_EDGE_REPORT_STEM = "k5_edge_report"  # REPORTS/<this>.{md,json}
KIM_EDGE_TABLE_NAME = "k5_edge_s2024"  # REPORTS/zs_arm_table_<this>.md (control K5f)
KIM_BOTTOMK_WEIGHT = 1.0  # ED5: symmetry with L_MIL, not tuned
KIM_BOTTOMK_TOPK_PCT = DADA_ORIGIN_MIL_TOPK_PCT  # ED5: a divisor; k = 20 // 5 = 4 = L_MIL's k
EDGE_TARGET_ZONE_S = (-1.0, 0.0)  # ED8 T1: zone AUC against negative-video frames
EDGE_ONSET_S = 2.0  # ED8 T2: span vs the normal frames within this many seconds of its edge

# --- KIM_RUN §50: bottom-k weight 0.25 against K5f re-trained in the same notebook ---
KIM_BOTTOMK25_CONTROL = "K5frr"  # BQ1: K5f's exact command, re-trained in the session (control)
KIM_BOTTOMK25_ROW = "K5fbk25"  # BQ1: K5frr + the two bottom-k keys
KIM_BOTTOMK25_WEIGHT = 0.25  # BQ1: one value, not derived and not swept
KIM_BOTTOMK25_OUTPUT_DIRNAME = "kim_bottomk25"  # OUTPUT_ROOT/<this>/<row>/{K5/s<seed>, eval}
KIM_BOTTOMK25_REPORT_STEM = "k5_bottomk_report"  # REPORTS/<this>.{md,json}
KIM_BOTTOMK25_TABLE_NAME = "k5_bottomk25_s2024"  # REPORTS/zs_arm_table_<this>.md
KIM_RUN_ENV_FILENAME = "env.json"  # beside a run: GPU name, torch and CUDA version
SPAN_THIRDS = 3  # BQ5: the span's abnormal rows split in this many parts; the first is read
BOTTOMK25_STRICT_LOW = -0.01  # BQ5: printed non-inferiority bound beside target A

# --- KIM_RUN §51: bottom-k weight 0.25 on K5dfi, K5dfi re-trained in the same notebook ---
KIM_DFI_BOTTOMK25_CONTROL = "K5dfirr"  # BD1: K5dfi's exact command, re-trained (control)
KIM_DFI_BOTTOMK25_ROW = "K5dfibk25"  # BD1: K5dfirr + the two bottom-k keys
KIM_DFI_BOTTOMK25_OUTPUT_DIRNAME = "kim_dfi_bottomk25"  # OUTPUT_ROOT/<this>/<row>/
KIM_DFI_BOTTOMK25_REPORT_STEM = "k5dfi_bottomk_report"  # REPORTS/<this>.{md,json}
KIM_DFI_BOTTOMK25_TABLE_NAME = "k5dfi_bottomk25_s2024"  # REPORTS/zs_arm_table_<this>.md

# --- KIM_RUN §59: KIM reads its own temporal encoder (kim.input_encoder) ---
KIM_INPUT_ENCODER_NONE = "none"  # KIM reads v^t (every arm before §59)
KIM_INPUT_ENCODER_CLIP = "clip"  # second encoder over the CLIP columns (K5dfice)
KIM_INPUT_ENCODER_FUSED = "fused"  # second encoder over x + W_u u (capacity control K5dficx)
KIM_INPUT_ENCODERS = (KIM_INPUT_ENCODER_NONE, KIM_INPUT_ENCODER_CLIP, KIM_INPUT_ENCODER_FUSED)
KIM_ENCODER_RNG_OFFSET = 1  # CE2: the second encoder's dropout stream is seeded train.seed + this
KIM_ENC_MIN_DRIFT = 1e-3  # CE5: final ||theta - theta_0|| / ||theta_0|| below this = DID NOT TRAIN
KIM_CLIP_ENC_ROW = "K5dfice"  # CE1: K5dfi + kim.input_encoder=clip
KIM_CLIP_ENC_CAP_ROW = "K5dficx"  # CE1: K5dfi + kim.input_encoder=fused
KIM_CLIP_ENC_CONTROL = KIM_DFI_BOTTOMK25_CONTROL  # CE3: K5dfirr, reused on a matching GPU
KIM_CLIP_ENC_CONTROL_GPU = "Tesla T4"  # CE3: the GPU K5dfirr trained on (§51.1)
KIM_CLIP_ENC_OUTPUT_DIRNAME = "kim_dfi_clip_encoder"  # OUTPUT_ROOT/<this>/<row>/
KIM_CLIP_ENC_REPORT_STEM = "k5dfi_clip_encoder_report"  # REPORTS/<this>.{md,json}
KIM_CLIP_ENC_TABLE_NAME = "k5dfi_clip_encoder_s2024"  # REPORTS/zs_arm_table_<this>.md

# --- KIM_RUN §53: KIM-on vs KIM-off on six arms (kim_ablation_report) ---
# KA1: KIM-on arm -> (its KIM-off row, OUTPUT_ROOT dir that holds the on row); first = KA7 reference
KIM_ABLATION_ARMS = {
    KIM_FIXED_ARM: ("K0f", KIM_FIXED_OUTPUT_DIRNAME),
    KIM_DVS_IGNORE_ROW: ("K0dfi", KIM_DVS_IGNORE_OUTPUT_DIRNAME),
    KIM_DVS_IGNORE_EGO_ROW: ("K0dfie", KIM_DVS_IGNORE_OUTPUT_DIRNAME),
    KIM_BOTTOMK25_ROW: ("K0fbk25", KIM_BOTTOMK25_OUTPUT_DIRNAME),
    KIM_DFI_BOTTOMK25_ROW: ("K0dfibk25", KIM_DFI_BOTTOMK25_OUTPUT_DIRNAME),
    KIM_EDGE_PRE_ROW: ("K0fepre", KIM_EDGE_OUTPUT_DIRNAME),
}
KIM_ABLATION_RERUN_SUFFIX = "rr"  # KA3: the on row re-trained when the starts do not match
KIM_ABLATION_FLAGS = ("kim.enabled", "kim.prototypes", "kim.edge_mask", "kim.seam_mask")  # KA1
KIM_ABLATION_OUTPUT_DIRNAME = "kim_ablation"  # OUTPUT_ROOT/<this>/<row>/{K5/s<seed>, eval}
KIM_ABLATION_REPORT_STEM = "kim_ablation_report"  # REPORTS/<this>.{md,json}

# --- KIM_RUN §52: DoTA train pixels located in CAP-DATA (construction only, no model) ---
DOTA_TRAIN_SPLITS_DIR = Path(__file__).resolve().parent / "splits" / "dota_train"
DOTA_TRAIN_MAP_FILENAME = "DOTA_TRAIN_CAP.json"  # frozen dota_id -> CAP video map
DOTA_TRAIN_FRAMES_DIRNAME = "DoTA_CAP_train_frames"  # <drive data>/<this>/{dota_id}.zip
DOTA_TRAIN_OUTPUT_DIRNAME = "dota_train"  # OUTPUT_ROOT/REPORTS/<this>/
# DT2: CAP id ranges [low, high) holding DoTA clips in name order, with CAP frames per DoTA frame;
# read off the 1,248 exact val pairs of MM-AU P0
DOTA_TRAIN_BLOCKS = {"A": (800, 1509, 1), "B": (1509, 6450, 1), "C": (9000, 11800, 3)}
DOTA_TRAIN_BLOCK_PREFERENCE = ("B", "A", "C")  # a clip CAP stores twice: 10 fps copy first
# DT3: CAP total frames - rate * DoTA frames that counts as the same clip (the val pairs' values)
DOTA_TRAIN_LENGTH_DIFF = {1: (0,), 3: (-2, -3)}
DOTA_TRAIN_START_SCALE = 8.0  # DoTA frames; weight 1 + exp(-|start diff| / this), never a filter
DOTA_TRAIN_ANCHOR_WEIGHT = 50.0  # an exact val pair outweighs any chain of unanchored matches
DOTA_TRAIN_CV_FOLDS = 2
DOTA_TRAIN_CV_SEED = 0
DOTA_TRAIN_FRAME_PHASE = {1: 0, 3: -1}  # DT5: CAP frame = rate * d + this, clamped (val maps)
# KIM_RUN §55: K5dfi's recipe trained on the recovered DoTA train clips (user's override of DT7)
DOTA_TRAIN_EXCLUDED = (  # 3 clips whose box rows were wrong when re-read unblinded (§52.2)
    "V3--0ubJkNE_005461", "gZbGLa253Ak_002191", "nvVk7L3GrJQ_002650",
)
DOTA_TRAIN_DATASET = "DoTA_train"  # data.dataset; same traffic definition set (definitions)
DOTA_TRAIN_STRIDE = V2_E1_STRIDE_BC  # DoTA is 10 fps: one row = 0.3 s, protocol B's row
DOTA_TRAIN_WINDOW_LENGTH = 12  # rows = 3.6 s
DOTA_TRAIN_WINDOW_HOP = 6
DOTA_TRAIN_FAULT_LABEL = "dota_train"  # DadaRecord.fault_label (not ego: is_egocentric=false)
DOTA_TRAIN_CLIP_DATASET = "DoTA_train_s3_ncc"  # cache/clip/<this>/{id}.npy
DOTA_TRAIN_VIDEO_DATASET = "DoTA_train_s3"  # cache/video/<encoder>/<this>_squash/
DOTA_TRAIN_CORPUS_DIRNAME = "DoTA_train_window"  # data/<this>/
DOTA_TRAIN_BAKE_DIRNAME = "DoTA_train_s3_dota"  # cache/v2/A2_S/<this>: fit-ids on DoTA train
DOTA_TRAIN_STATS_BAKE_SUFFIX = "_dota"  # a zero-shot bake restated to those statistics ends so
DOTA_TRAIN_BUILD_REPORT = "build_report.json"
DOTA_TRAIN_DECODE_CHUNK = 4  # jpgs decoded and resized at a time: small = 720p floats fit the cache
DOTA_TRAIN_IO_THREADS = 4  # decode threads beyond one per CPU core, to cover Drive read waits
KIM_DOTA_TRAIN_ROW = "K5dfiDoTA"  # in-domain on DoTA, never a zero-shot row
KIM_DOTA_TRAIN_OUTPUT_DIRNAME = "kim_dota_train"  # OUTPUT_ROOT/<this>/<row>/{K5/s<seed>, eval}
KIM_DOTA_TRAIN_REPORT_STEM = "k5dfidota_report"  # REPORTS/<this>.md
KIM_DOTA_TRAIN_TABLE_NAME = "k5dfidota_s2024"  # zs_arm_table control K5dfi (row K5f printed)
# §56: three arms on the §55 corpus, control K5dfiDoTA; each = K5dfiDoTA's command + these sets
KIM_DOTA_TRAIN_ARMS = {
    "K5dfieDoTA": ("data.is_egocentric=true",),
    "K5dfibk25DoTA": (f"loss.bottomk_weight={KIM_BOTTOMK25_WEIGHT}",
                      f"loss.bottomk_topk_pct={KIM_BOTTOMK_TOPK_PCT}"),
    "K5dfiebk25DoTA": ("data.is_egocentric=true", f"loss.bottomk_weight={KIM_BOTTOMK25_WEIGHT}",
                       f"loss.bottomk_topk_pct={KIM_BOTTOMK_TOPK_PCT}"),
}
KIM_DOTA_TRAIN_ARMS_REPORT_STEM = "k5dfidota_arms_report"  # REPORTS/<this>.md
# §57: EDA of the cut DoTA train corpus (core/tools/dota_eda.py); figures reuse NEXAR_EDA_*
DOTA_EDA_OUTPUT_DIRNAME = "dota_eda"  # OUTPUT_ROOT/<this>/REPORTS
DOTA_EDA_REPORT_STEM = "dota_eda_report"
DOTA_EDA_ROW_THRESHOLDS = (1, 2, 3, 6)  # share of positive windows with <= k abnormal rows (of 12)
DOTA_EDA_POSITION_BINS = 10  # relative clip position bins for the row-level time profile
DOTA_EDA_TAIL_ROWS = 10  # rows counted back from the clip end (3 s at s3)
DOTA_EDA_END_TOLERANCE_ROWS = 1  # a span "reaches the clip end" if it stops this close to it
DOTA_EDA_TOP_CLASSES = 12  # accident_name rows shown in the table / figure
KIM_DOTA_TRAIN_ARMS_TABLE_NAME = "k5dfidota_arms_s2024"  # zs_arm_table control K5dfiDoTA
# §58: ablation on the §55 corpus; KIM = K5dfiDoTA - K0dfiDoTA, VideoMAE = K0dfiDoTA - L0dfiDoTA
KIM_DOTA_TRAIN_OFF_ROW = "K0dfiDoTA"  # K5dfiDoTA with KIM_ABLATION_FLAGS false (§53 KA1)
KIM_DOTA_TRAIN_L0_ROW = "L0dfiDoTA"  # + v2.motion=none, model.motion_dim=0, plain CLIP (§36 R1)
KIM_DOTA_TRAIN_ABL_REPORT_STEM = "k5dfidota_ablation_report"  # REPORTS/<this>.md
KIM_DOTA_TRAIN_ABL_TABLES = {"kim": "k5dfidota_abl_vs_k0_s2024",  # control K0dfiDoTA
                             "motion": "k5dfidota_abl_vs_l0_s2024"}  # control L0dfiDoTA
KIM_DOTA_TRAIN_POSITION_RULER = (0.8545, 0.8386, 0.8695)  # §55: cubic position, DoTA-CAP-dev
KIM_DADA_MIDDLE_RULER = (0.7342, 0.7076, 0.7594)  # §36.1: position-only middle ruler, DADA
CAP_LIGHT_NIGHT = 2  # CAP's light column: 1 day, 2 night
# DT7 pixel gate: DoTA boxes drawn on the recovered frames, judged by eye
DOTA_BOX_WIDTH = 1280  # DoTA's boxes live on a 1280 x 720 frame
DOTA_BOX_HEIGHT = 720
DOTA_SHEET_SEED = 2024
DOTA_SHEET_PER_GROUP = {"10fps": 30, "30fps": 30}  # real train clips per rate group
DOTA_SHEET_VAL_CONTROLS = 6  # val clips whose pixels CLIP already verified
DOTA_SHEET_DECOYS = 10  # wrong on purpose: frames of a neighbouring clip of the same video
DOTA_SHEET_FRAMES = 3  # first, middle and last boxed frame of the annotated span
DOTA_SHEET_REDRAW_SEED = 2025  # DT7: the one redraw after a gate that is not valid (§52.1)
DOTA_SHEET_REDRAW_FRAMES = 5
DOTA_SHEET_ROWS_PER_PAGE = 8
DOTA_SHEET_TILE_WIDTH = 480
DOTA_SHEET_MARGIN = 6
DOTA_SHEET_LABEL_WIDTH = 60
DOTA_SHEET_BOX_COLOUR = "#ff2020"
DOTA_SHEET_BOX_LINE = 3
DOTA_SHEET_MIN_DECOYS_CAUGHT = 8  # of DOTA_SHEET_DECOYS, else the gate cannot see errors
DOTA_SHEET_MAX_FLAGGED = 2  # real clips per rate group; more drops the group
DOTA_SHEET_MAX_CONTROLS_FLAGGED = 1  # val controls; more means the judging is too strict
