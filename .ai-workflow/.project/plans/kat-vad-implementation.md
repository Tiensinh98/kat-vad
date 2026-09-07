# KAT-VAD Implementation Plan

**Date:** 2026-07-07 (rev. 2 — code-first: all code, incl. download/train/infer CLIs, is written and smoke-tested on synthetic data first; the user runs downloads, training, and benchmarks afterwards)
**Input:** `.project/research_documents/KAT_VAD_RESEARCH.md` (all open questions resolved)
**Specs:** `core/docs/KAT_VAD_PROPOSAL.md`, `core/docs/KAT_VAD_IMPLEMENTATION_SPEC.md`
**Baseline (read-only):** `LaGoVAD-PreVAD/`

---

## 1. Summary

- Build **KAT-VAD** in `core/`: a full re-implementation of the LaGoVAD baseline (plain PyTorch + argparse, no Lightning) plus the novel **KIP** module (PMGFlowHead, KinematicShift, MotionScoreHead) spliced between the temporal encoder and fusion.
- **Code-first:** Phases 0–5 produce *all* code — models, losses, data pipeline, download CLIs, train/inference/eval/viz CLIs — validated end-to-end on **synthetic data only** (no dataset, checkpoint, or GPU required). No phase blocks on external downloads.
- **Phase 6 is the user-run execution phase:** download TAD + LaGoVAD `best.ckpt`, extract features/flow, pass the reproduction gates, run Stage 1–2 KIP training on the Colab A100.
- v1 targets **TAD**; PreVAD training and the headline DoTA zero-shot delta are project phase 2 (Phase 7).
- The **reproduction gates** (unchanged in substance, now all in Phase 6): (a) our eval code reproduces `best.ckpt` zero-shot TAD ≈ 89.56 AUC ±0.5; (b) our TAD-trained KIP-off model ≥ 89.56; (c) KIP-on ≥ KIP-off.

## 2. Assumptions & Constraints

- Python **3.10** (upgrade to 3.11 later; avoid 3.11-only syntax). Deps pinned to baseline majors: torch 2.4.*, torchvision 0.19.*, transformers 4.56.*, numpy<2.
- Compute: **Google Colab A100 40 GB** for Phase-6 runs; dev machine is macOS (CPU/MPS) — all code and tests must run data-free on CPU. All long jobs resumable + Drive-persisted; AMP and grad-accumulation as config flags.
- **No pretrained weights during development** where avoidable: unit/parity tests use randomly initialized modules; anything needing real weights (CLIP, RAFT, `best.ckpt`) is behind a download CLI and exercised only in Phase 6. (Exception: if a CLIP/RAFT weight auto-download is trivial locally, smoke tests MAY use it, but must not require it.)
- `LaGoVAD-PreVAD/` is **never modified**; ideas are re-implemented in `core/` only.
- Baseline hyperparameters adopted verbatim for reproduction (report §2.3): hidden 512, RoFormer 2L/4H/window 25, AdamW 5e-5 cosine + 20 warmup steps, batch 64, 40 epochs, seed 2024, `mil_topk_pct=16`, `contrastive_temp=0.02`, etc.
- Design decisions locked in the report: `θ` = no-synthesis probability (0.7; 0.85 ego-centric); `L_dvs` = `supervised_loss` + `pseudo_sup_mil_loss` pair; keep 32 soft prompts; HF CLIP for both image and text; torchvision RAFT; flow pooling = statistics + seeded fixed linear map (A10).
- CLAUDE.md code standards apply: argparse CLI per entry point, type hints, logging (no print), constants in `constants.py`, quality gates before commit.

## 3. Plan Metadata

- **Plan type:** New model implementation (baseline re-implementation + novel module), greenfield `core/` package, code-first delivery
- **Size / scope:** Large
- **Estimated duration:** 3–5 weeks of code (Phases 0–5); Phase 6 execution depends on user download/GPU time; Phase 7 open-ended
- **Storage path:** `@.project/plans/kat-vad-implementation.md`

## 4. Phases Overview

| Phase | Name | Goal | Est. | Depends on |
|-------|------|------|------|------------|
| 0 | Environment & Skeleton | Runnable repo skeleton, deps installed, config system, quality gates green | 1–2 days | — |
| 1 | KIP Modules (data-free) | Spec §2–5 modules unit-tested on random tensors | 3–4 days | 0 |
| 2 | KIP Losses (data-free) | `L_KIP_rec`, `L_KIP_align`, `L_kin` unit-tested | 2–3 days | 1 |
| 3 | Baseline Port (data-free) | All reuse components re-implemented with parity checks on random inputs | 1–1.5 weeks | 0; parallel with 1–2 |
| 4 | Data Pipeline Code + Download CLIs | TAD preprocessor, feature/flow extraction CLIs, KNN builder, DVS dataloader, dataset/checkpoint download CLIs — all tested on synthetic fixtures | 5–7 days | 3 |
| 5 | Train/Infer/Eval Code + Synthetic E2E | Train, inference, eval, viz CLIs; ckpt-compat loader; full pipeline smoke-run on a synthetic mini-dataset | ~1 week | 3, 4; KIP from 1–2 |
| 6 | Execution & Benchmarks (USER-RUN) | Downloads, feature/flow extraction, gates (a)(b)(c), Stage 1–2 KIP training on TAD | user-paced | 5 |
| 7 | Scale-out (project phase 2) | PreVAD training, DoTA headline eval, remaining datasets, optional stages | open | 6 |

Parallelism: Phases 1–2 and 3 are independent. Nothing in 0–5 waits on any download.

## 5. Detailed Tasks by Phase

### Phase 0 – Environment & Skeleton

- **Goal:** Everything needed to write and test code exists; zero external downloads.
- **Tasks:**
  - [x] Add runtime deps via `uv` (torch 2.4.1, torchvision 0.19.1, transformers 4.56.2, torchmetrics, numpy 1.26.4, einops, opencv-python, **PyAV** (decord has no macOS arm64 wheel), faiss-cpu, scikit-learn, gdown, matplotlib, pyyaml); `uv sync` green on Python 3.10.6.
  - [x] Create `core/` package skeleton: `core/{models,kip,losses,data,flow,tools}/`, `core/constants.py`, `core/config.py` (dataclass/YAML config with every spec §10 ablation flag), `core/tests/`. Project renamed `kat-vad`, hatchling build → `core` importable.
  - [x] Device-resolution utility (`core/device.py`, `auto|cuda|mps|cpu`, explicit specs fail loudly).
  - [x] Define on-disk data layout contract as constants (`core/constants.py`) + doc (`core/docs/DATA_LAYOUT.md`).
  - [x] Verify quality gates run clean (ruff/mypy/bandit/pyright configs exclude `LaGoVAD-PreVAD/`; pycycle runs as `cd core && pycycle --here`). All 5 green; 10 unit tests pass.
- **Deliverables:** green `uv sync`; importable `core/` skeleton; documented data-layout contract; quality gates pass.

### Phase 1 – KIP Modules (spec §2–5, greenfield)

- **Goal:** All KIP `nn.Module`s correct on random `v^t (2,50,512)` with masks.
- **Tasks:**
  - [x] `core/kip/pmg.py` — PMGFlowHead: Conv1d 512→128 k3 → GELU → Linear → GELU → Conv1d 128→256 k3; `(B,L,512)→(B,L,256)`.
  - [x] `core/kip/gate_shift.py` — KinematicShift: `m_t=‖ê_O‖₂`, per-video min-max norm (mask-aware), MLP 1→16→16→1+σ, shift `s_t=⌊r_t·D/K⌋` K=4, bidirectional channel shift, zero-fill boundaries. Implement **loop version (oracle) + vectorized gather**; test equivalence (A13/R6). *(Vectorized = broadcast channel-index masks + nested `torch.where`; also carries `gate_signal="feat_var"` for ablation 5.)*
  - [x] `core/kip/motion_head.py` — MotionScoreHead 256→128→1+σ (padded scores forced to 0 so top-k never selects them).
  - [x] `core/kip/kip_module.py` — KIP wrapper returning `(v^k, ê_O, ŷ_O)`; includes `proj_flow` 256→128 and `proj_rgb` 512→128; `KIP.from_config(KIPConfig)` maps ablation flags (`pmg_only` → shift off).
  - [x] Shape/mask/gradient unit tests for each module (CPU-only, no data) — 26 tests in `core/tests/test_kip_modules.py`; vectorized shift benchmarked ~8× faster than loop oracle (2×400×512 CPU). **Finding (pending lesson P3):** spec-as-written hard `floor()` gives the gate MLP zero gradient — untrained at init; asserted by test, revisit at Stage 1/2 training.
- **Deliverables:** `core/kip/` modules + passing tests; vectorized shift benchmarked vs loop.

### Phase 2 – KIP Losses (spec §5.2)

- **Goal:** Each loss finite, differentiable, mask-correct on random inputs.
- **Tasks:**
  - [x] `L_KIP_rec` — masked MSE(`ê_O`, cached `e_O`), λ_rec=1.0. (`kip_reconstruction_loss`)
  - [x] `L_KIP_align` — bidirectional per-video InfoNCE τ=0.07 on 128-d projections; **flags for A11 mitigations** (position subsampling, ±w neighborhood exclusion), default = spec-as-written, λ_al=0.1. (`kip_alignment_loss`; takes pre-projected pairs from `KIP.project_for_align`)
  - [x] `L_kin` — BCE(topk_mean(ŷ_O), ŷ) + β·smoothL1 on union of top-k indices (k = `length//16`, dynamic per A12); abnormal-only masking; `y^p` anchor path for synthesized samples (`is_synthesized` flag per video); γ=0.2, β=0.5. (`kinematic_loss`; detaches `y^bin` internally)
  - [x] Unit tests: finite scalar, gradient flow, padded-position invariance, degenerate cases (all-normal batch, L<k) — 18 tests in `core/tests/test_kip_losses.py`; also: manual-computation parity, detach verified (no grad to `y^bin`), β linearity, yp-anchor on/off paths, single-valid-frame InfoNCE = 0.
- **Deliverables:** `core/kip/losses.py` + tests.

### Phase 3 – Baseline Port (report §2.2 reuse map, data-free)

- **Goal:** Every reuse component re-implemented in `core/`, spot-checked for parity against the baseline module on identical **random inputs with randomly initialized weights** (run baseline modules read-only in a test harness; no pretrained weights needed for architectural parity).
- **Tasks:**
  - [x] `core/models/clip_text.py` — SoftPromptCLIPTextModel re-implementation (32 soft prompts, frozen body); CLIP model injectable for weight-free tests; HF revision pinned; `return_dict` dropped for transformers 4.56.
  - [x] `core/models/temporal_encoder.py` — RoFormer 2L/4H RoPE **including** the window-25 attention mask and gated residual (A8); encoder-only re-implementation with baseline-identical module names (state dicts round-trip). Note: shipped `default.yaml` has `temp_gate:` empty → gate **off** in best.ckpt; ours defaults off, flag available.
  - [x] `core/models/fusion.py` — CoAttnFusionLayer ×2 (`CoAttentionFusion`; only co_attn ported — the variant best.ckpt uses).
  - [x] `core/models/heads.py` — H_bin (ConvScoreHead k9 **1 layer** per shipped config, adaptive fuse α₀=0.25/scale 10) + H_mul (cosine/temperature 0.2, `sim` type); dispatch extracted from `_fuse_and_head`.
  - [x] `core/losses/{mil,dvs,contrastive}.py` — `mil_loss`, `multi_class_mil_loss(_v2)`, `supervised_loss` + `pseudo_sup_mil_loss` (= `L_dvs` pair, A2), `CapContrastLoss` + `asymmetric_infonce_loss` (module instantiated once; 0.2/−1e4/−100/0.1 constants named). n3 mining's literal `pseudo_frame_label != 0` selection kept verbatim (baseline comment says "normal portion" but code selects labeled frames).
  - [x] `core/data/collate.py` — pad/mask utils and collate fn (baseline's multi-dim `np.pad` bug fixed: pads time axis only).
  - [x] `core/data/definitions.py` — DatasetSpecVerbalizer port (prevad/ucf/msad/dota/dada/tad; injectable RNG); TAD definitions available at training time; spec §7.4 traffic set as `TRAFFIC_DEFINITIONS`.
  - [x] `core/models/kat_vad.py` — assembled model (plain `nn.Module`): CLIP feats → temporal encoder → **KIP (config-gated on/off)** → fusion → heads; `v^k` feeds fusion, H_bin-pre, and `L_neg` (`vis_feats` output) when KIP on; `v^t` when off; text encoding decoupled (`encode_text`) so visual path tests need no CLIP weights.
  - [x] Per-module unit tests + parity tests — 59 new tests (`test_baseline_parity/models/losses.py`, `test_data_utils.py`), 113 total green; parity via read-only `lagovad_ref` loader (skips baseline `__init__` → no lightning dep) with 1:1 state-dict copy on random weights; CLIP parity on tiny random CLIPTextConfig.
- **Deliverables:** `core/models/`, `core/losses/`, `core/data/` with tests; parity report vs baseline modules. **DONE 2026-07-08** (all 5 quality gates green).

### Phase 4 – Data Pipeline Code + Download CLIs (all synthetic-tested)

- **Goal:** Every data-facing tool exists and is tested on tiny synthetic fixtures (generated frames / random features); nothing requires a real dataset yet.
- **Pivot (2026-07-08, user):** the user already has **MSAD-traffic on disk** (20–30s clips, annotation table `name/scenario/total_frames/anomaly_start/anomaly_end`) → the dataset preprocessor targets **MSAD instead of TAD** (`core/data/msad.py`; spec §7.2). TAD later = thin adapter (only annotation parsing is dataset-specific). TAD download subcommand dropped from `download.py` (dataset already local; Phase-6 gates to be restated against MSAD when Phase 6 is planned). `DataConfig.dataset` default switched to MSAD.
- **Tasks:**
  - [x] `core/tools/download.py` — argparse CLI: `ckpt` (LaGoVAD `best.ckpt`, GDrive via gdown, resume + optional `--ckpt-sha256`), `clip` (HF snapshot prefetch, pinned revision), `raft` (torchvision Raft_Large C_T_SKHT_V2 prefetch), `all`; skip-if-exists + `--dry-run`. Tested with monkeypatched transfers only.
  - [x] `core/data/msad.py` — MSAD preprocessor (replaces planned tad.py): tolerant 5-column parser (tab/comma/2+-space, header optional, `--one-indexed`/`--end-exclusive` flags), scenario filter (`traffic` preset = spec §7.2 slice), class inference from name prefix, protocol-ii-ratio seeded stratified split or `--split-file`, writes `labels_train/frame_labels_test/defs/meta.json` (train windows → meta.json only; weak supervision preserved). Frame labels stride-8 aligned (`i*stride ∈ [start,end]`).
  - [x] `core/tools/extract_clip_features.py` — argparse CLI, HF CLIP vision tower (injectable for tests), stride-8 PyAV sampling (`core/data/video_io.py`, shared with RAFT), deterministic resize/centercrop/normalize, resumable per-video `.npy`. Tested on cv2-generated videos + tiny random CLIPVisionConfig; A6 parity deferred to Phase 6.
  - [x] `core/flow/raft_extract.py` — argparse CLI: torchvision RAFT on adjacent sampled pairs, pooling = 23-d statistics (mag mean/std/max, u/v mean/std, 16-bin magnitude-weighted angle histogram) + seeded fixed linear map to 256-d (A10), projection persisted in versioned `cache/flow/v1/` (loaders fail loudly), raw stats cached per video (`.stats.npy`) for the motion-aware KNN key, last-frame duplication, resumable. Note: RAFT needs inputs ≥128px and /8 (validated). Tested with random-weight raft_small.
  - [x] `core/data/knn_cache.py` — KNN cache builder (A4): faiss IndexFlatIP over L2-normalized central-frame CLIP features, abnormal→top-K(10) normal mapping, `--motion-key` appends a 4-d descriptor from cached flow stats (spec §6.1; off for fixed-camera MSAD). **faiss forced single-threaded — torch+faiss libomp clash segfaults on macOS arm64 (lesson #3).**
  - [x] `core/data/synthesis.py` + `core/data/dataset.py` — DVS generalized from `PreVADDatasetOnline`: θ = no-synthesis probability, synthesis branch splices ≥1 filler (50% KNN / 50% random normal), `__len__=2×num_anomaly` (A9), yields `v_feat/e_o/y^p/is_synthesized/cls_label`, variable-length + collate padding. **`y^p` is all-zero for un-synthesized abnormal clips (window unknown under weak supervision → L_kin falls back to `y^bin.detach()`).** `require_flow=False` yields zero `e_O` for KIP-off runs. Plus `FeatureEvalDataset` (frame-labeled test videos, full length).
  - [x] Synthetic fixture generator (`core/tests/fixtures.py`): deterministic MSAD-style mini-dataset (annotation TSV → real preprocessor → labels; random CLIP/flow caches + projection; real KNN builder; optional cv2 mp4s) reused by extractor tests and Phase-5 E2E.
- **Deliverables:** DONE 2026-07-08 — download/feature/flow CLIs; KNN builder; DVS dataloader; fixture generator; 54 new tests (167 total green), all 5 quality gates green, no network/data dependency.

### Phase 5 – Train/Infer/Eval Code + Synthetic End-to-End

- **Goal:** "Ready to run" is proven: the full train → checkpoint → resume → inference → eval → visualize loop completes on the synthetic mini-dataset on CPU.
- **Tasks:**
  - [x] `core/train.py` — argparse plain-torch loop: AdamW 5e-5/cosine/20-warmup/batch 64/40 epochs/seed 2024 (all overridable); stage selection (1 = KIP-only warm-up, 2 = full); AMP + grad-accum flags; per-epoch + per-N-step checkpointing with full resume (model/optim/sched/epoch/RNG incl. dataset/verbalizer/mps states) + `--stop-after-epochs` for time-boxed sessions. In-house cosine-warmup LambdaLR (P4 HF-drift avoidance). **L_dvs pair row-gated to normal|synthesized** (our y^p is all-zero on un-synthesized abnormal — deviation documented in `core/docs/TRAINING.md`); `loss.captions_from_definitions` flag activates the caption branch/L_neg on description-less datasets (default off = baseline-faithful).
  - [x] `core/inference.py` — argparse: feature-file/dir or raw-video (extraction path) scoring; sliding-window `sliding_window_scores` shared with evaluate; per-video `.npz` (score/sim/class_names); `--text-encoder {clip,stub}` (stub = deterministic sha256-seeded embeddings, zero downloads).
  - [x] `core/evaluate.py` — argparse: sliding-window full-length eval (`max_vis_len=512`), per-window verbalized definitions (baseline convention, `--no-verbalize` off-switch); micro AUC + AP (sklearn, parity-tested vs torchmetrics — invariant to A5 uniform expansion, helper `expand_to_frames`); MCC/AUC_A/mAP@IoU stubbed for Phase 7; `results.json` + `--save-scores` npz.
  - [x] Checkpoint-compat loader — `core/models/ckpt_compat.py`: temporal_encoder.*→temporal_encoder.encoder.*, gate_alpha→temporal_encoder.gate_alpha, CLIP body skipped, kip.* stays init, fail-loud on unknown/missing/mis-shaped keys; validated on synthetic baseline-keyed state dicts (`test_ckpt_compat.py`) + exercised through evaluate CLI.
  - [x] `core/tools/visualize.py` — argparse: anomaly-curve PNG per video (gt shading, optional threshold line), Agg backend.
  - [x] Colab bootstrap — `core/docs/COLAB.md` (clone/install/Drive mount/env roots + full Phase-6 command sequence; locally dry-runnable with `--text-encoder stub`).
  - [x] **Synthetic E2E test** (`core/tests/test_e2e_synthetic.py`, 8 tests): stage-2 KIP-on 2 epochs (all loss terms present + finite), KIP-off, stage-1 warm-up (L_KIP_rec **and** L_KIP_align decrease), kill→resume (epoch-boundary and mid-epoch step-ckpt; weights match straight run within FP tolerance — bitwise impossible on Accelerate BLAS, pending lesson P7), inference/eval/viz CLIs produce artifacts, baseline-keyed ckpt through evaluate (gate-a path). **E2E pins train.device=cpu: torch 2.4 MPS training diverges (pending lesson P6 [HIGH]).**
- **Deliverables:** DONE 2026-07-09 — train/inference/eval/viz CLIs; ckpt-compat loader; Colab bootstrap; synthetic E2E green (205 tests total, all 5 quality gates green) — **code-complete milestone**.

### Phase 6 – Execution & Benchmarks (USER-RUN)

- **Goal:** Real data through the finished code: gates passed, KIP trained on TAD.
- **Tasks (user executes; Claude assists/debugs):**
  - [ ] Run `download.py`: TAD (train+test), `best.ckpt`, CLIP + RAFT weights → Drive.
  - [ ] Run TAD preprocessing + CLIP feature extraction + RAFT flow extraction on Colab A100 (hours); build KNN cache.
  - [ ] A6 parity spot-check: HF CLIP features vs bundled OpenAI-clip on a few real videos.
  - [ ] **Gate (a):** score `best.ckpt` on TAD-test through our eval → ≈89.56 ±0.5 AUC (validates eval code + ckpt loader).
  - [ ] **Gate (b):** train KIP-off on TAD-train → TAD-test AUC ≥ 89.56; watch overfitting (~400 videos).
  - [ ] Stage 1 KIP warm-up (~5 epochs): `L_KIP_rec` decreases and plateaus; watch `L_KIP_align` (A11), enable mitigation flags if needed.
  - [ ] Stage 2 full training → **gate (c):** KIP-on ≥ KIP-off on TAD-test.
  - [ ] Smoke-test ablation flags (spec §10); document results + deviations in `core/docs/`; add lessons learned (CLAUDE.md §7).
- **Deliverables:** three checkpoints (KIP-off, Stage-1, Stage-2) + results table; go/no-go for scale-out.

### Phase 7 – Scale-out (project phase 2, separate detailed plan later)

- **Goal:** The thesis: PreVAD-trained KAT-VAD beats LaGoVAD's DoTA 62.60 zero-shot.
- **Tasks (outline only):**
  - [ ] Acquire PreVAD (raw videos — R1, top external risk) and DoTA; MSAD when access granted.
  - [ ] PreVAD flow extraction at scale (~50–100 A100-hours, resumable, background) + KNN cache.
  - [ ] PreVAD reproduction vs paper (OQ6 original tolerances) → Stage 2 on PreVAD → DoTA/DADA/TAD zero-shot evals.
  - [ ] Full metric suite (MCC family, AUC_A, mAP@IoU), seven required ablations, remaining datasets.
  - [ ] Optional: Stage 0 warm-up, Stage 0.5 hard-negative tuning, Stage 3 ATS+MLLM reasoning head.
- **Deliverables:** headline results table; ablation table; final docs.

## 6. Risks & Mitigations

- **Download CLI untested against real endpoints until Phase 6** (code-first tradeoff) → keep the CLI thin (gdown/requests + checksums); expect a fix-up iteration when the user first runs it; TAD mirror URL verified during Phase 4 authoring (reading pages costs nothing).
- **Synthetic E2E can't catch data-distribution bugs** (label alignment, frame_time expansion on real annotations) → gate (a) exists precisely for this; keep the fabricated mini-annotation in TAD's exact schema, copied from `tad_test_anno.json` structure.
- **Gate (a) misses ±0.5** (feature-extractor numerics, A6) → parity-check HF vs OpenAI-clip first; if HF drifts, fall back to bundled OpenAI-clip extraction for reproduction, keep HF for new training.
- **Gate (b) misses 89.56** → training-pipeline bug hunt (DVS labels, loss masks, LR schedule) before touching KIP.
- **Overfitting on ~400 TAD videos** → DVS synthesis, early stopping on TAD-test AUC trend, report best + last.
- **`L_KIP_align` false negatives on long videos (A11/R5)** → mitigation flags built in Phase 2, toggled on Stage-1 evidence.
- **Colab session death mid-run** → resumability asserted by the synthetic E2E (kill-and-resume) before any real run.
- **KinematicShift loop too slow (R6)** → vectorized version + oracle equivalence test in Phase 1.
- **Silent metric bugs flatter results (R11)** → metrics unit-tested against sklearn/torchmetrics + gate (a) end-to-end check.
- **TAD is fixed-camera** → Phase 6 validates the pipeline, not the ego-motion thesis; DoTA (Phase 7) carries the claim.

## 7. Next Steps for the User

1. Confirm this revised plan — implementation of Phases 0–5 then starts immediately, no downloads needed from you until Phase 6.
2. (During Phase 4, quick check) confirm the TAD source URL I put in `download.py` matches where you intend to get it.
3. When Phase 5's synthetic E2E is green, run Phase 6 step by step (download → extract → gates → train) on your Colab A100; I debug anything that breaks.
4. Keep the MSAD request alive — needed only in Phase 7.
