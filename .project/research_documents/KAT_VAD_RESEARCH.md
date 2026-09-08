# KAT-VAD Research Report — Pre-Implementation Findings

**Date:** 2026-07-07
**Purpose:** feed the follow-up planning session that turns this into an implementation plan for `core/`.
**Inputs:** `core/docs/KAT_VAD_PROPOSAL.md`, `core/docs/KAT_VAD_IMPLEMENTATION_SPEC.md`, full audit of `LaGoVAD-PreVAD/` (read-only baseline), repo environment (`pyproject.toml`, `.python-version`).

All paths relative to project root: `/Users/sinhpham/Public/cs/Internship-2/Video Anomaly Detection papers/code/`.

---

## 0. Executive summary

- The two spec docs are coherent with each other; the real friction is **spec vs. baseline reality**. Several names the spec uses (`L_dvs`, `θ=0.7`, "frozen text encoder") do not map 1:1 to the LaGoVAD code, and two things the spec assumes exist are **missing from the released baseline**: the DVS KNN retrieval-cache **builder** and any optical-flow code.
- The reusable LaGoVAD components (temporal encoder, fusion, heads, MIL/contrastive losses) are clean `nn.Module`s / plain functions — easy to re-implement in `core/`. The **hard port is the DVS dataloader** (`PreVADDatasetOnline`), which is entangled, print-laden, and depends on a precomputed retrieval cache whose builder was never released.
- KIP itself is fully specified with exact shapes and has zero baseline dependency — it can be built and unit-tested immediately, in parallel with the port.
- The spec's build order is directionally right but skips a mandatory milestone: **baseline reproduction in `core/` before splicing KIP** (otherwise KIP gains cannot be attributed) and an explicit **step 0 for environment + data acquisition** (root `pyproject.toml` currently has zero runtime dependencies; no datasets or features are on disk).
- Biggest external risks: PreVAD raw-video acquisition (needed for RAFT flow targets — features alone are not enough), MSAD access request lead time, and the Python-version conflict (repo pins 3.10; baseline env is 3.11).

---

## 1. RQ1 — Spec comprehension

### 1.1 Module inventory (target `core/` model)

| # | Module | Source | Input → Output | Key params |
|---|--------|--------|----------------|-----------|
| 1 | Frozen CLIP image encoder (offline feature extraction) | reuse (LaGoVAD) | frames `(L,3,224,224)` → `F (L,512)`, cached `.npy` | ViT-B/16, stride 8 (DoTA at 10 fps) |
| 2 | Frozen CLIP text encoder + soft prompts | reuse | definition `Z` (C strings) → `z^t (C,512)` | HF `openai/clip-vit-base-patch16`; 32 learnable soft prompts (see §1.6-A7) |
| 3 | Temporal encoder | reuse | `F (B,L,512)` → `v^t (B,L,512)` | RoFormer 2 layers, 4 heads, RoPE, local window 25, gated residual |
| 4 | **KIP — PMGFlowHead** | **new** (spec §2) | `v^t (B,L,512)` → `ê_O (B,L,256)` | Conv1d 512→128 k3 → GELU → Linear 128→128 → GELU → Conv1d 128→256 k3 |
| 5 | **KIP — KinematicShift** | **new** (spec §3) | `v^t`, `ê_O` → `v^k (B,L,512)` | `m_t=‖ê_O,t‖₂`, per-video min-max norm, MLP 1→16→16→1 + σ, `s_t=⌊r_t·D/K⌋`, K=4, bidirectional channel shift, zero-fill boundaries, respect mask |
| 6 | **KIP — MotionScoreHead** | **new** (spec §4) | `ê_O (B,L,256)` → `ŷ_O (B,L)` | Linear 256→128, GELU, Linear 128→1, σ |
| 7 | **KIP wrapper** | **new** (spec §5.1) | `v^t`, mask → `(v^k, ê_O, ŷ_O)` | + projections `proj_flow 256→128`, `proj_rgb 512→128` for L_KIP_align |
| 8 | Co-attention fusion `U` | reuse | `(v^k (B,L,512), z^t (B,C,512))` → `(v^u, z^u)` | 2 × CoAttnFusionLayer (bidir cross-attn + FFN) |
| 9 | Detection head `H_bin` | reuse | pre=`v^k`, post=`v^u` → `y^bin (B,L)` | 2 × ConvScoreHead (Conv1d k=9, replicate pad), fused by learnable scalar `σ(α·10)`, α₀=0.25 |
| 10 | Classification head `H_mul` | reuse | `(v^u, z^u)` → `y^mul (B,L,C)` | cosine sim of L2-normed feats / learnable temperature (init 0.2) |
| 11 | RAFT flow-target extractor (offline, train only) | **new** (spec §1) | video → `e_O (L,256)` cached `cache/flow/{id}.npy` | RAFT on adjacent sampled-frame pairs; deterministic pooling to 256-d |
| 12 | ATS sampler + MLLM report (Stage 3, optional, off-path) | new, build last | `y^bin` + frames → text report | inverse-CDF sampling; LoRA r=64/α=128 |

The **only splice** into the baseline flow: KIP sits between temporal encoder and fusion; `v^k` (not `v^t`) feeds both fusion `U` and `H_bin`'s pre-fusion path; `L_neg` also consumes `v^k`.

### 1.2 Loss inventory

| Loss | Spec meaning | Baseline reality (see §2) | Stage-2 weight |
|------|--------------|---------------------------|----------------|
| `L_MIL` | top-k MIL BCE on `y^bin` | `mil_loss` (`losses.py:40`), k = `length//mil_topk_pct` (pct=16) | 1.0 |
| `L_MIL-align` | MIL on `y^mul` | `multi_class_mil_loss` / `_v2` (`losses.py:68/96`), CE, pct=16, `mul_weight=1.0` | 1.0 |
| `L_dvs` | DVS pseudo-label loss | **no function of that name** — it is `supervised_loss` (frame BCE, `losses.py:127`) + `pseudo_sup_mil_loss` (top-k within synthesized span, pct=4, `losses.py:144`) driven by `pseudo_frame_label` | 1.0 + 1.0 |
| `L_neg` | hard-negative contrastive | `CapContrastLoss` + `asymmetric_infonce_loss` (`losses.py:197/176`), `neg_mining='n3'`, temp=0.02 | 1.0 |
| `L_KIP_rec` | masked MSE(`ê_O`, cached `e_O`) | **new** | λ_rec = 1.0 (deliberately unweighted) |
| `L_KIP_align` | bidirectional per-video InfoNCE(`ê_O`, `v^t`), τ=0.07, proj to 128-d, mask padded | **new** | λ_al = 0.1 |
| `L_kin` | `BCE(topk_mean(ŷ_O), ŷ)` + β·smoothL1 on union of top-k indices of `ŷ_O` and `y^bin.detach()` (abnormal videos only; `y^p` anchor on synthesized samples) | **new** | γ = 0.2, β = 0.5 |

### 1.3 Training stages

| Stage | Status | Trains | Loss | Notes |
|-------|--------|--------|------|-------|
| 0 | optional | temporal encoder + KIP.pmg | masked-token MSE + `MSE(pmg(v^t), e_O)` on 75%-masked positions | BDD100K / PreVAD-normal; discard recon decoder |
| 0.5 | optional (skip on stock-CLIP default) | thin projection on frozen CLIP | `L_neg` contrastive on same-scene + semantic text hard negatives | weights-free Alert-CLIP substitute |
| 1 | **required** | KIP only (CLIP frozen) | `L_KIP_rec + L_KIP_align` | ~5 epochs until `L_KIP_rec` plateaus; Pi-VAD warm-up pattern |
| 2 | **required** | temporal enc + KIP + fusion + heads + projections (CLIP frozen; RAFT never in graph) | LaGoVAD full + KIP terms (§1.2) | AdamW, LR 5e-5, batch 64, ~40 epochs, 1 GPU; tune weights on PreVAD-val (only split with frame labels) |
| 3 | optional, last | ATS sampler + LoRA adapter | instruction tuning | detector fully frozen |

### 1.4 Datasets

| Role | Dataset | Notes |
|------|---------|-------|
| WS train (primary) | PreVAD (35,279 videos) | LaGoVAD native; descriptions + 7/35 category names; frame labels on val only |
| WS train (fixed-cam traffic) | MSAD Protocol ii | 360N+120A train / 120+120 test; **access request required**; traffic scenario filter |
| WS train (bridge) | UCF-Crime | comparability only |
| Eval ego-centric | DoTA (headline), DADA-2000, A3D (opt) | DoTA at 10 fps; frame labels from temporal windows |
| Eval fixed-cam | TAD, MSAD traffic slice, Street Scene (opt) | |
| Reasoning eval (Stage 3) | CUVA traffic subset | |
| Stage-0 warm-up | BDD100K normal (+ PIE opt) | normal-only, no labels |

Standardized label files: `labels_train.json` (id→0/1), `frame_labels_test.json` (id→[0/1 per sampled frame]), `defs.json`. Caches: `cache/clip/{id}.npy`, `cache/flow/{id}.npy` (train only).

### 1.5 Evaluation protocol

Frame-level AUC (all; DoTA delta over 62.60 is the headline), AP (PreVAD-val, MSAD), AUC_A/AnoAUC, MCC/AUCMCC/MCC@0.5 (DoTA/DADA), mAP@IoU 0.1–0.5+AVG (MSAD, UCF), cross-dataset zero-shot (train PreVAD/MSAD/UCF → DoTA/DADA/TAD), drift@5 (LaGoVAD Protocol 2), MSAD per-scenario slices (highway stress), GFLOPs/FPS with reasoning latency reported separately. Seven required ablations toggled by config flags (spec §10), including flow-norm vs feature-variance gate.

### 1.6 Ambiguities, contradictions, underspecified parts

**A1 — `θ=0.7` does not exist in the baseline.** Both docs cite DVS `θ=0.7`; the spec (§6.1) interprets it as a "no-synthesis probability" (raise to 0.85 for ego-centric). No such parameter exists in the code. Closest analogues: `enhance_single_clip_factor=0.3` (probability of using a single clip) and `syn_max_num_clips=5` (= `δ_m`). The KNN retrieval threshold interpretation is also impossible to verify since the retrieval cache is precomputed offline and its builder is not in the repo. **Decision needed:** in our re-implementation define `θ` explicitly as the no-synthesis probability (spec §6.1 semantics) and document the deviation from `enhance_single_clip_factor` semantics.

**A2 — `L_dvs` is not a single loss.** In the baseline the DVS signal is realized as `supervised_loss` + `pseudo_sup_mil_loss` on `pseudo_frame_label`. The spec's `L_dvs(y^bin, y^p)` should be implemented as this pair (two weights, both 1.0), not as one new function.

**A3 — `L_MIL-align` naming.** The spec maps it to "MIL-align on `y^mul`"; the baseline's function is `multi_class_mil_loss` (CE over class logits, top-k pct=16; v2 adds bottom-k on normals). Adopt the baseline math under the spec's name.

**A4 — DVS KNN cache builder is missing (also affects the motion-aware-KNN novelty).** `PreVADDatasetOnline` loads `v6/train_search_cache.json` and uses it 50% of the time (50% fully random normal). The faiss index / cache-building script was never released. We must write our own builder — which is fine, because spec §6.1's motion-aware KNN key (append ego-motion descriptor from cached `e_O`) requires a custom builder anyway. Underspecified: similarity threshold, K, and the exact CLIP "central-frame feature" key construction.

**A5 — "per-frame" vs per-snippet scores.** Features are per sampled clip (interval 8); the baseline maps clip scores to frame scores via a per-dataset `frame_time` list (`[8,8,8,8,2,1,8,8]`) at eval. The spec's `y^bin ∈ ℝ^L` is really per-snippet. Keep the baseline's expansion convention so AUC numbers are comparable.

**A6 — CLIP checkpoint inconsistency in baseline.** Config default is `patch32`, actual configs use `patch16`; image features come from the bundled OpenAI `clip` (jit) while the text side uses HF `transformers`. **Recommendation:** unify on HF `openai/clip-vit-base-patch16` for both image and text in `core/` (one dependency, one weight source); verify feature parity on a few videos against the OpenAI-clip extractor before committing (numerics differ slightly between the two implementations — this can shift a reproduction).

**A7 — "Frozen CLIP text encoder" is not fully frozen.** The baseline's `SoftPromptCLIPTextModel` adds **32 trainable soft-prompt embeddings** (body frozen). Neither doc mentions soft prompts. Keep them (baseline fidelity; they're part of what makes definition conditioning work), and state so in the plan.

**A8 — Temporal encoder extras the spec omits.** The baseline adds a local windowed attention mask (`temp_window_size=25`) and a gated residual (`v_feat + temp_out·tanh(α)·w`). These live in `lagovad.py`, not in the encoder module, and must be ported explicitly.

**A9 — Batch balance contradiction.** Proposal §7.3 prescribes balanced 32+32 minibatches (RefineVAD convention); the baseline instead balances implicitly (`__len__ = 2×num_anomaly`, first half abnormal-synthesis, second half normal-synthesis; `BinaryBalancedBatchSampler` exists but is unused). Keep the baseline mechanism for reproduction; treat 32+32 as an optional ablation.

**A10 — Flow pooling underspecified (spec §1 step 3).** Two alternatives offered (resize-flatten-random/PCA-project vs flow statistics + fixed linear map). The only hard requirement is determinism across epochs. **Recommendation:** flow statistics (mean/std/max magnitude + angle histogram) + fixed seeded linear map to 256-d — cheaper, resolution-independent, and interpretable for the `m_t = ‖ê_O‖` gate. Persist the projection matrix with the cache.

**A11 — `L_KIP_align` with long videos.** Per-video InfoNCE over up to L=512 positions with `targets=arange(L)`: temporally adjacent frames are near-duplicates, so most "negatives" are false negatives. Pi-VAD works at snippet level with far fewer positions. Mitigations to decide at implementation: subsample positions per video, or exclude a ±w temporal neighborhood from negatives. Flag for the plan; start with the spec as written, watch the loss.

**A12 — `L_kin` top-k.** "use LaGoVAD's k" = `length // mil_topk_pct` (pct=16), i.e. dynamic per video, not a fixed k.

**A13 — Spec's `KinematicShift` reference code is O(B·L) Python loops** and `.item()` host syncs. Acceptable for correctness-first per the spec's own note, but plan a vectorized gather before Stage-2 scale training (batch 64 × L 512 × 40 epochs makes the loop a real cost).

**A14 — Framework mismatch.** Baseline is PyTorch Lightning 2.3 + `LightningCLI` (YAML, jsonargparse, no argparse); CLAUDE.md requires argparse CLIs for every entry point. Decision needed (see Open Questions).

---

## 2. RQ2 — Baseline code audit (`LaGoVAD-PreVAD/`, read-only)

Apache-2.0 code; dataset CC-BY-NC-4.0. Stack: Python 3.11.9, PyTorch 2.4.0 (CUDA 12.4), Lightning 2.3.3, transformers 4.56.0, torchmetrics 1.4.0, faiss-cpu 1.9.0, decord, einops.

### 2.1 Entry points

| Script | Path | Interface | Notes |
|--------|------|-----------|-------|
| Train | `LaGoVAD-PreVAD/src/main.py` | `LightningCLI` + `src/configs/default.yaml` | `train.sh` hardcodes `CUDA_VISIBLE_DEVICES=3` and a local proxy |
| Detection eval | `LaGoVAD-PreVAD/src/full_length_eval.py` | argparse: `--config --ckpt --dataset --data_root --cache_dir` | sliding window (`max_vis_len=512`), AUROC+AP |
| Classification eval | `LaGoVAD-PreVAD/src/offline_evals/offline_{ucf,xd,dota}_eval.py` | none — **hardcoded paths** (`ckpts/pred_results.pkl`, `/data/datasets/PreVAD/...`) | Accuracy + macro-F1 |
| End-to-end inference | `LaGoVAD-PreVAD/src/end2end_inference.py` | argparse: `--config --ckpt --cache_dir -vp` | ffmpeg frames → HF CLIP feats → model → viz PNG; class defs hardcoded |
| Feature extraction | `LaGoVAD-PreVAD/tools/extract_feat_clip.py` | none — **hardcoded** dirs, `device=cuda:1` | bundled OpenAI clip ViT-B/16 jit, interval 8, batch 256 |
| Data scraping | `LaGoVAD-PreVAD/data_download/` | assorted scrapers + Flask annotator | raw videos gated; not a one-shot download script |

### 2.2 Reuse map (baseline file → target `core/` module)

| Component | Baseline location | Target in `core/` | Port effort |
|-----------|------------------|-------------------|-------------|
| CLIP text encoder (+soft prompts) | `src/models/LaGoVAD/modeling_clip.py:12` `SoftPromptCLIPTextModel` | `core/models/clip_text.py` | **Medium** — in-place tensor ops + per-batch Python loop in prompt insertion; freezing done externally (`lagovad.py:87-88`); Chinese comments |
| CLIP image feature extraction | `tools/extract_feat_clip.py` (OpenAI clip) / `end2end_inference.py:110-136` (HF) | `core/tools/extract_clip_features.py` (argparse; HF path) | **Easy** — unify the two divergent paths on HF |
| Temporal encoder (RoPE) | `src/models/LaGoVAD/modeling_roformer.py:428` `RoFormerEncoder` (RoPE at `:213`) + gate/local-mask logic in `lagovad.py:275-300,384-412` | `core/models/temporal_encoder.py` | **Easy/Medium** — encoder itself is clean vendored HF RoFormer; must extract window-mask + gated-residual glue from the LightningModule |
| Co-attention fusion `U` | `src/models/LaGoVAD/fusion_encoders.py:8` `CoAttnFusionLayer`, `:170` `FusionV1` | `core/models/fusion.py` | **Easy** — clean, no globals/prints |
| `H_bin` | `src/models/LaGoVAD/heads.py:84` `BinaryHead`, `:9` `ConvScoreHead` | `core/models/heads.py` | **Easy** — extract the string-driven dispatch from `lagovad._fuse_and_head` (`:436-480`) |
| `H_mul` | `src/models/LaGoVAD/heads.py:134` `MultiClassHead`, `:52` `SimScoreHead` | `core/models/heads.py` | **Easy** |
| `L_MIL` | `src/models/LaGoVAD/losses.py:40` `mil_loss` | `core/losses/mil.py` | **Easy** — per-sample Python loop; vectorize later |
| `L_MIL-align` | `losses.py:68/96` `multi_class_mil_loss(_v2)` | `core/losses/mil.py` | **Easy** |
| `L_dvs` (pair) | `losses.py:127` `supervised_loss` + `:144` `pseudo_sup_mil_loss` | `core/losses/dvs.py` | **Medium** — semantics tied to DVS dataloader outputs |
| `L_neg` | `losses.py:197` `CapContrastLoss` + `:176` `asymmetric_infonce_loss` | `core/losses/contrastive.py` | **Easy** — but fix: re-instantiated every step in `training_step` (`lagovad.py:667`); move to `__init__`; magic constants 0.2 / −1e4 → named |
| DVS dataloader | `src/datasets/PreVAD.py:428` `PreVADDatasetOnline` (`__getitem__ :594-718`, KNN filler `:490-590`) | `core/data/prevad.py` + `core/data/synthesis.py` | **Hard** — entangled, `print()`s, hardcoded category maps, depends on missing `train_search_cache.json` builder |
| Pad/mask utils, collate | `src/datasets/utils.py:8,53`, `src/models/LaGoVAD/utils.py:24` | `core/data/collate.py` | **Easy** |
| Verbalizers (definitions Z) | `src/models/LaGoVAD/verbalizer.py:7,508,539` | `core/data/definitions.py` | **Easy/Medium** — large per-dataset prompt tables to carry over |
| Orchestration / training loop | `src/models/LaGoVAD/lagovad.py:40` `LaGoVADLightModel` (1,022 lines) | `core/models/kat_vad.py` + `core/train.py` | **Medium/Hard** — monolith with commented-out code and debug prints; re-organize, don't copy |
| Eval | `src/full_length_eval.py`, in-training val `lagovad.py:693-790` | `core/evaluate.py` | **Medium** — add MCC-family, AUC_A, mAP@IoU (not in baseline) |

### 2.3 Key hyperparameters confirmed (`src/configs/default.yaml`)

hidden 512; soft prompts 32; RoFormer 2 layers / 4 heads / window 25 / max_pos 1536; fusion co_attn ×2; head kernel 9, adaptive (α₀=0.25, scale 10); AdamW lr 5e-5, wd 0.0, cosine, 20 warmup steps; batch 64; 40 epochs; seed 2024; `vis_max_len=512`; `mil_topk_pct=16`, `sup_mil_topk_pct=4`, `mul_mil_topk_pct=16`; `contrastive_temp=0.02`, neg_mining `n3`; `syn_max_num_clips=5`; `enhance_single_clip_factor=0.3`; retrieval cache `v6/train_search_cache.json`.

### 2.4 Things the baseline does NOT contain (explicit findings)

1. **No RAFT / optical-flow / motion code anywhere.** KIP and `flow/raft_extract.py` are 100% greenfield (confirms spec).
2. **No KNN retrieval-cache builder** (faiss-cpu is a dep, but no script builds `train_search_cache.json`).
3. **No `ckpts/`** — the released LaGoVAD checkpoint (`best.ckpt`) must be downloaded from the Google Drive link in the README (id `161T7EkR64Px1cveP7_dG1xupT8IEICbM`). Needed for a sanity cross-check of our reproduction.
4. **Broken finetuned configs** — `src/configs/finetuned/*.yaml` reference `models.CLIP4VAD.*` classes that don't exist in the repo. Not on our critical path, but do not model `core/` configs on them.
5. **No frame-level metrics beyond AUROC/AP** — MCC family, AUC_A, mAP@IoU must be written new for spec §10.
6. Only test annotations for external datasets are shipped (`data/other_datasets/*_test_anno.json` for ucf/xd/msad/dota/ubnormal/tad/lad) — a useful head start for the eval preprocessors; raw features are not shipped.

---

## 3. RQ3 — Novel-component (KIP) feasibility

### 3.1 External dependencies

| Dependency | Purpose | Options | Recommendation |
|------------|---------|---------|----------------|
| RAFT | offline flow targets `e_O` | (a) `torchvision.models.optical_flow.raft_large/raft_small` with bundled pretrained weights (torchvision ≥ 0.12; 0.19.0 already matches the baseline env); (b) official `princeton-vl/RAFT` repo (BSD-3) with `raft-things.pth` | **(a) torchvision** — no vendored third-party code, weights auto-download, pinned by the existing torchvision version. Verify exact API against current docs at implementation time (context7). Keep the extractor behind an interface so (b) is a drop-in swap |
| CLIP | frame + text features | HF `transformers` `openai/clip-vit-base-patch16` (auto-download) | unify image+text on HF (see A6) |
| open_clip | backbone-ladder option 3 (ViT-L) | optional | defer; config-gated |
| Alert-CLIP | backbone-ladder option 4 | weights **not public** | config-gated stub only; never a blocker (per proposal) |
| PyTorch stack | everything | baseline env: torch 2.4.0 + torchvision 0.19.0 + CUDA 12.4 | pin the same majors in `pyproject.toml` via `uv` |
| faiss (cpu) | our own DVS KNN cache builder | faiss-cpu 1.9.0 (baseline parity) | needed because the builder is missing upstream |
| Lightning / torchmetrics | training loop + metrics | if we keep Lightning (see OQ2) | torchmetrics useful either way |

Pretrained weights to download: HF CLIP ViT-B/16 (~600 MB), torchvision RAFT-Large weights (~20 MB), LaGoVAD `best.ckpt` (Google Drive, for reproduction cross-check), optionally OpenAI clip ViT-B/16 jit (only if verifying feature parity with the baseline extractor).

### 3.2 Compute / caching for offline flow extraction

- Per video: RAFT on `L−1` adjacent sampled-frame pairs (stride 8 → L ≈ 100–500 for typical clips), last flow duplicated → `e_O (L,256)` float32.
- **Storage is trivial:** `L×256×4B` ≈ 0.5 MB per 500-frame video → well under ~20 GB for all of PreVAD + MSAD + UCF training splits. CLIP feature caches are the same order.
- **GPU time is the real cost:** at roughly 25–50 ms/pair (RAFT-Large, ~440×~1024-ish inputs, batched pairs) on a single high-end GPU, PreVAD's 35,279 videos × ~200 pairs ≈ 50–100 GPU-hours. One-off and resumable; run on a subset first (build-order step 1). RAFT-Small or reduced input resolution can cut this 3–5× if needed — an acceptable deviation since `e_O` is pooled to statistics anyway.
- **Hard prerequisite:** RAFT needs **raw videos** (or extracted frames), not the released CLIP features. Raw-video availability/disk (PreVAD is web-scale; plausibly hundreds of GB to TBs) is a Stage-0 acquisition task and the single largest logistics risk (see risk register).
- Determinism: persist the pooling projection matrix (seeded) alongside the cache; version the cache (`cache/flow/v1/...`) so a pooling change invalidates cleanly.

### 3.3 KIP module itself

Pure `nn.Module`s over `(B,L,512)` tensors — no data dependency, unit-testable on random `v^t (2,50,512)` day one. Parameter count is tiny (<1M). Only implementation caution: vectorize `KinematicShift` before full training (A13) and honor the padding mask in every loss (spec is explicit about this).

---

## 4. RQ4 — Gaps and risks

### 4.1 Environment gaps (blocking, cheap to fix)

| Gap | Evidence | Fix |
|-----|----------|-----|
| Root `pyproject.toml` has **zero runtime dependencies** (only dev linters) | `pyproject.toml` lines 7, 9–16 | `uv add` torch/torchvision/transformers/lightning(?)/torchmetrics/faiss-cpu/einops/decord/opencv/numpy/scikit-learn |
| Python pinned to **3.10** vs baseline env **3.11.9** | `.python-version` vs `LaGoVAD-PreVAD/environment.yaml` | torch 2.4 supports both; either bump `.python-version` to 3.11 or validate on 3.10 — decide (OQ1) |
| Dev machine is **macOS (darwin)** — no CUDA | env | code must be device-agnostic (`cuda`/`mps`/`cpu` flag); unit tests CPU-only; training + RAFT extraction assume a remote CUDA GPU (proposal: single RTX 4090). Where is it? (OQ4) |
| No datasets or feature caches on disk anywhere in the repo | audit | Stage-0 acquisition plan needed (OQ3) |

### 4.2 Risk register

| # | Risk | Sev | Likelihood | Mitigation |
|---|------|-----|------------|-----------|
| R1 | **PreVAD raw videos unavailable/huge** — RAFT needs pixels, not features; scraping scripts (bilibili/youtube) imply link-rot | High | Medium | Check LaGoVAD release page/HF for packaged videos or frames first; fall back to extracting flow only on obtainable subsets (MSAD/UCF/DoTA-train) and report KIP there |
| R2 | **MSAD access request lead time** (gated) | Medium | High | File the request now; MSAD is not needed until build step 7 |
| R3 | **DVS KNN cache builder missing** → our rebuilt cache ≠ LaGoVAD's → reproduction drift | Medium | High | Reproduce first with the shipped `train_search_cache.json` if PreVAD's cache file is distributed with the dataset; else accept drift and document; the 50%-random path bounds the impact |
| R4 | **Baseline reproduction misses paper numbers** (feature-extractor numerics A6, cache R3, seeds) | High | Medium | Insert explicit reproduction milestone (§5); cross-check with downloaded `best.ckpt` via `full_length_eval.py` before training our own |
| R5 | `L_KIP_align` false negatives on long, static videos (A11) | Medium | Medium | Position subsampling / neighborhood exclusion as ablation-ready flags |
| R6 | `KinematicShift` Python loop too slow at batch 64 × L 512 (A13) | Medium | High | Vectorized gather before Stage 2; loop version kept for the unit test oracle |
| R7 | Flow-extraction GPU budget (50–100 h on PreVAD) contends with training | Medium | Medium | RAFT-Small / lower res; subset-first; resumable per-video caching |
| R8 | Lightning-vs-plain-torch divergence from CLAUDE.md argparse rule (A14) | Low | High | OQ2 — recommend plain torch loop with argparse (cleaner `core/`), reusing baseline hyperparameters exactly |
| R9 | Alert-CLIP weights never ship | Low | High | Already handled by backbone ladder — config-gated, non-blocking |
| R10 | Licensing: code Apache-2.0 (fine to re-implement); PreVAD data CC-BY-NC-4.0 | Low | — | Non-commercial research use only; note in docs |
| R11 | Metrics not in baseline (MCC family, AUC_A, mAP@IoU) written from scratch — silent metric bugs flatter results | Medium | Medium | Unit-test metrics against sklearn/torchmetrics references and published numbers on `best.ckpt` predictions |

### 4.3 Things the spec assumes exist but don't

1. `θ` in the baseline (A1) — must be defined by us.
2. KNN cache builder (A4/R3) — must be written by us (needed for motion-aware KNN anyway).
3. A "LaGoVAD forward to patch" — the spec's §6 wording implies editing LaGoVAD; the CLAUDE.md boundary makes `LaGoVAD-PreVAD/` read-only, so §6 really means "the `core/` re-implementation's forward". No conflict in substance, but the plan must schedule the **full port** as a first-class work item, not a "patch".
4. Runtime environment (§4.1).
5. `frame_labels_test.json` builders for DoTA/DADA/TAD — partially covered by the shipped `*_test_anno.json` files (§2.4.6); alignment to stride-8/10-fps sampling still to write.

---

## 5. RQ5 — Build-order validation (refined)

The spec's 8-step order is right about KIP-first-in-isolation and training-stages-last, but it (a) has no environment/data step, (b) hides the biggest work item ("patch LaGoVAD forward" is actually "port the whole baseline into `core/`"), and (c) lacks a baseline-reproduction gate, without which the headline claim (KIP's DoTA delta over 62.60) is unattributable.

**Refined order** (spec step numbers in parentheses):

| Step | Work | Gate / exit criterion |
|------|------|----------------------|
| 0 | Environment: `uv` deps, Python-version decision, device-agnostic scaffolding, `core/` package skeleton + `constants.py`/`config.py`; download CLIP weights + LaGoVAD `best.ckpt`; kick off dataset acquisition (PreVAD, MSAD request, DoTA) | `uv sync` green; quality gates run |
| 1 | **KIP submodules** (spec 2): `core/kip/{pmg,gate_shift,motion_head,kip_module}.py` + shape unit tests on random `v^t (2,50,512)` with masks | all shapes match spec §2–5; mask honored |
| 2 | **KIP losses** (spec 3): `core/kip/losses.py` + finite-scalar unit tests; include A11/A12 flags | each loss finite, differentiable, mask-correct |
| 3 | **Baseline port** (bulk of spec 4): CLIP text encoder, temporal encoder (+gate/window glue), fusion, heads, LaGoVAD losses, pad/mask/collate, verbalizers → `core/models/`, `core/losses/`, `core/data/`; argparse `extract_clip_features.py` | unit tests per module; parity spot-checks vs baseline modules on identical random inputs |
| 4 | **`flow/raft_extract.py`** (spec 1) on a small obtainable subset; deterministic pooling persisted | `e_O (L,256)` verified; re-run bit-identical |
| 5 | **Dataloader + DVS** (rest of spec 4): `PreVADDatasetOnline` re-implementation incl. our KNN cache builder (motion-aware key optional flag), `y^p`, `e_O`, mask passthrough; `θ` defined per A1 | batch dict contains `clip_feats, e_O, ŷ, Z, y^p, mask` with documented shapes |
| 6 | **Baseline reproduction milestone (NEW)**: train `core/` model with KIP off on PreVAD; compare to paper numbers and to downloaded `best.ckpt` evaluated through our eval code | PreVAD-val AUC within a tolerance band of paper/ckpt; eval code cross-validated (R11) |
| 7 | **Stage 1 KIP warm-up** (spec 5) | `L_KIP_rec` decreases and plateaus |
| 8 | **Stage 2 full training** (spec 6) on PreVAD; sanity AUC on PreVAD-val; ablation flags wired (spec §10) | KIP-on ≥ KIP-off on PreVAD-val; DoTA zero-shot delta measured |
| 9 | **Dataset preprocessors + full eval** (spec 7): DoTA/DADA/TAD/MSAD/UCF one at a time; new metrics (MCC family, AUC_A, mAP@IoU) unit-tested | eval table for spec §10 |
| 10 | Optional stages (spec 8): Stage 0 warm-up, Stage 0.5 HN tuning, Stage 3 ATS/MLLM | ablation deltas reported |

Notes: steps 1–2 and 3 are independent and parallelizable; step 4 needs only raw videos of a subset, not the full acquisition; step 6 is the critical new gate; RAFT extraction at PreVAD scale (R7) can run in the background between steps 5 and 8.

---

## 6. Consolidated dependency list

Runtime: `torch==2.4.*`, `torchvision==0.19.*` (RAFT + transforms), `transformers==4.56.*` (CLIP), `torchmetrics`, `numpy<2` (baseline used 1.26.4), `einops`, `opencv-python`, `decord` (video decode; check macOS wheel — fall back to PyAV/ffmpeg), `faiss-cpu`, `scikit-learn`, `ffmpeg` (system), optional: `lightning==2.3.*` (pending OQ2), `open_clip_torch` (ladder opt. 3), `peft` (Stage 3 LoRA), `wandb`.
Dev (already present): ruff, mypy, bandit, pycycle, pyright.
Weights: HF CLIP ViT-B/16; torchvision RAFT-Large (or princeton-vl `raft-things.pth`); LaGoVAD `best.ckpt` (GDrive `161T7EkR64Px1cveP7_dG1xupT8IEICbM`).
Data: PreVAD (train/val + annos v6 + retrieval cache if distributed), MSAD (request), UCF-Crime, DoTA, DADA-2000, TAD, optional A3D/DAD/CCD/BDD100K/CUVA.

---

## 7. Open questions — RESOLVED (user answers 2026-07-07)

1. **OQ1 — Python version:** → **Stay on 3.10 for now**; user will upgrade to 3.11 later. Pin deps compatible with both (torch 2.4 supports both); avoid 3.11-only syntax.
2. **OQ2 — Training framework:** → **Plain PyTorch loop + argparse CLIs.** Copy LaGoVAD's optimizer/scheduler/loop semantics exactly (AdamW 5e-5, cosine, 20 warmup steps, batch 64, 40 epochs, seed 2024); keep torchmetrics. No Lightning dependency.
3. **OQ3 — Data:** → **No PreVAD or DoTA on disk yet; MSAD access request in progress.** Dataset acquisition (PreVAD first) is a step-0 work item; all code must be testable on synthetic/random tensors and tiny fixtures before data lands.
4. **OQ4 — Compute:** → **Google Colab with A100 (40 GB)** (no local CUDA; dev machine is macOS). Plan implications: (a) every long job — RAFT flow extraction, Stage-1/2 training — must be **resumable and checkpointed** (per-video flow cache already is; add per-epoch + per-N-step training checkpoints) because Colab sessions still time out; (b) caches/checkpoints persisted to Google Drive, not ephemeral Colab disk; (c) A100 40 GB comfortably fits batch 64 × L 512 — keep gradient-accumulation + AMP as config flags anyway (AMP for speed; accumulation as fallback if a session lands on a smaller GPU); (d) provide a thin Colab bootstrap (notebook or shell snippet) that clones the repo, installs deps from `pyproject.toml`, mounts Drive, and invokes the argparse CLIs.
5. **OQ5 — `θ` semantics (delegated to Claude):** **Decision: `θ` = no-synthesis probability** (spec §6.1 reading), default 0.7, raised to 0.85 for ego-centric data. Documented as a deviation from the baseline's `enhance_single_clip_factor` semantics (A1).
6. **OQ6 — Reproduction tolerance (delegated to Claude):** **Decision: two-part gate.** (a) Eval-code validation: the downloaded `best.ckpt` scored through *our* eval pipeline must match published numbers within **±0.5 AUC** (catches metric bugs, R11). (b) Our KIP-off retrain on PreVAD must land **within 1.5 AUC of the paper's PreVAD-val number** (absorbs feature-extractor numerics A6 + rebuilt KNN cache R3). Miss either → investigate before any KIP training.
7. **OQ7 — v1 dataset scope (amended 2026-07-07):** → **Train on TAD first** (user decision; supersedes the earlier PreVAD-first reading). TAD (fixed-camera traffic, ~400 weakly-labeled train videos + 100 test with frame labels, 7 anomaly categories) is small enough that acquisition, RAFT flow extraction (hours, not 50–100 h), and full train runs are cheap on the A100 — a good first end-to-end vehicle. PreVAD (the baseline's native training set) moves to phase 2, DoTA after.

   **Consequences for the plan:**
   - **Reproduction gate redefined (amends OQ6):** there is no published LaGoVAD *trained-on-TAD* number — the paper's TAD 89.56 AUC is zero-shot from PreVAD training. Gate becomes: (a) unchanged in spirit — downloaded `best.ckpt` scored on TAD-test through *our* eval pipeline must reproduce ≈89.56 within ±0.5 AUC (validates eval code; needs only TAD test videos + our CLIP feature extractor); (b) our KIP-off model *trained on TAD-train* should meet or beat that 89.56 zero-shot reference on TAD-test (a trained-on-domain model underperforming zero-shot transfer signals a training-pipeline bug); (c) KIP-on ≥ KIP-off on TAD-test.
   - **DVS on TAD:** the shipped `train_search_cache.json` is PreVAD-specific and useless here — our own KNN cache builder (already a work item, A4) is now needed in v1, built over TAD-train. Small dataset → cache build is trivial.
   - **Definitions Z for TAD:** the baseline ships `tad_test_anno.json` and verbalizer prompt tables; port/adapt the TAD category definitions for training-time conditioning, not just eval.
   - **Caveats to carry:** TAD is fixed-camera — KIP's headline motivation (ego-motion-dominated DoTA) is not exercised; TAD results validate the pipeline, not the thesis. ~400 training videos also means higher overfitting risk than 35k-video PreVAD (watch val curves; the DVS synthesis path matters more here). PreVAD remains the target for the headline claim.

---

## 8. Pointers (for the planning session)

- Specs: `core/docs/KAT_VAD_PROPOSAL.md`, `core/docs/KAT_VAD_IMPLEMENTATION_SPEC.md`
- Baseline anchors: `LaGoVAD-PreVAD/src/models/LaGoVAD/lagovad.py` (orchestration, 1,022 lines), `losses.py`, `heads.py`, `fusion_encoders.py`, `modeling_roformer.py`, `modeling_clip.py`, `src/datasets/PreVAD.py` (DVS), `src/configs/default.yaml`, `src/full_length_eval.py`, `tools/extract_feat_clip.py`, `data/other_datasets/*_test_anno.json`
- Environment: `pyproject.toml` (no runtime deps yet), `.python-version` (3.10), `LaGoVAD-PreVAD/environment.yaml` (reference env)
