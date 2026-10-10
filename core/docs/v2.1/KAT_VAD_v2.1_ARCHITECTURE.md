# KAT-VAD v2.1 — Model Architecture (with tensor shapes)

**Scope:** the network only. It covers what each block is, what goes in and comes out, and what is frozen, trained or used only in training. The reasons, the evidence and the experiment plan are in `KAT_VAD_PROPOSAL_v2.1.md`.

**What v2.1 is:** the adopted v2 model (arm **A3**), unchanged, plus one **training-only SG-NM branch** (§9) that adds one loss term on the detector. **The deployed network is A3's, with or without SG-NM.**

**The model:** LaGoVAD's trunk, unchanged, fed by
- a frozen CLIP ViT-B/16 frame stream;
- a frozen **VideoMAE V2-S video stream**, scaled by a fixed per-channel `σ_u` and one fixed scalar `c`, and added through a zero-initialized residual;
- with both streams passed through **Clip-Referenced Normalization (CRN)** with the per-dimension median as the reference (R2).

During training of step E4 only, an **SG-NM branch** reads the detached temporal features and adds one loss on `y^bin`. Nothing is added to the trunk or the heads.

**Naming.** v2 called the VideoMAE input the "motion stream". The frame-order control (D6, results report §3) showed that shuffling the 16 frames of each clip changes the result by ±0.0004, so the stream is a second, video-trained appearance encoder. It is called the **video stream** here.

**Conventions:**
- Shapes are written `batch × time × channels`.
- Trunk sizes and loss settings are those of the A3 E3 runs' `config.yaml`, which keeps the LaGoVAD baseline's values except the T2 deviations (score kernel 3, MIL `topk_pct` 5).
- `sg(·)` is stop-gradient.

---

## 0. Symbols

| Symbol | Meaning | Value |
|---|---|---|
| `B` | sequences per training batch | 64, drawn from a per-epoch permutation of a balanced item list (≈ 32 abnormal + 32 normal on average, not fixed) |
| `L` | steps per T2 window | 20 |
| `L'` | steps per training sequence, padded | 20 for a single window; up to δ_m · L = 100 when DVS splices windows (hard cap 512) |
| `T` | steps in a test clip | variable (DoTA median ≈ 33 at stride 3) |
| `r` | step rate | 3.75 Hz for DADA (30 fps, stride 8) · 3.33 Hz for DoTA (10 fps, stride 3) |
| `D` | trunk width | 512 |
| `d_v` | video feature size | 384 (VideoMAE V2-S) |
| `C` | categories in the definition `Z` | 2: `Normal` and `CarAccident` (every T2 window is `CarAccident`; definitions shared with DoTA), unchanged since phase 4 |
| `k` | MIL top-k | max(1, ⌊L'/5⌋): 4 on a 20-step window, up to 20 on a 100-step DVS sequence |
| `K` | SG-NM prototypes per sequence | 2 |
| `M` | SG-NM candidate normal steps per window segment | 20 (normal segment) · 8 (abnormal segment) |
| `λ_n` | weight of `L_consist` | `min(1, 0.2 / ρ_pilot)`, fixed once in a pilot (§9) |

The training tables write `L` for the time axis. With DVS on, the time axis is `L'` and every per-step tensor carries a valid-step mask.

---

## 1. Overview

```
video ─► steps @ r ─┬─ frame @ t ─────────────────► CLIP ViT-B/16 image (frozen) ─► x_t ∈ ℝ^512 ─► CRN (R2) ─► x̃_t ──────────────────┐
                    │                                                                                                            (+) ─► h_t ∈ ℝ^512
                    └─ causal clip 16 f @ 10 fps ─► VideoMAE V2-S (frozen) ───────► u_t ∈ ℝ^384 ─► CRN (R2) ─► c·(⊘σ_u) ─► W_u (0-init) ┘

h ∈ B×L×512 ─► temporal encoder (2 layers, RoPE, outer residual) ─► V^t ∈ B×L×512 ─┬─► H_bin, language-agnostic path ──────────────┐
                                                                                   ├─► CoAttn(V^t, z) ─► V^u ─► H_bin, language-guided path ─┴─► y^bin ∈ B×L
                                                                                   │                      └─► H_mul(V^u, Z^u) ─► y^mul ∈ B×L×C
                                                                                   └─► sg ─► SG-NM branch (training only) ─► S̃ ∈ B×L ─► L_consist(σ(y^bin), sg(S̃))
Z ─► CLIP text (frozen) + soft prompts ─► z ∈ C×512 ─► CoAttn
```

---

## 2. Input sampling

| Item | Value |
|---|---|
| Step rate | DADA: 30 fps, stride 8 (3.75 Hz, 0.27 s per step) · DoTA: 10 fps, stride 3 (3.33 Hz, 0.30 s per step) |
| Frame input (per step) | the frame at the step; full frame resized to 224 × 224 (squash), no centre crop |
| Clip input (per step) | 16 frames at 10 fps **ending at** the step (causal, 1.5 s). DADA: every 3rd native frame; DoTA: native frames. Full frame squashed to 224 × 224, as for the frame input. Normalized with VideoMAE's mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225). |
| Training item | a T2 window of `L` = 20 consecutive steps of one source video; with DVS, a sequence spliced from up to δ_m = 5 windows |
| Per-batch raw input | frames `B × L × 3 × 224 × 224`; clips `B × L × 3 × 16 × 224 × 224` |
| DoTA test set for this model | **DoTA-CAP**: the 1,129 of 1,397 DoTA clips whose pixels were recovered from CAP-DATA, split dev 569 / eval 560 |

Both encoders are frozen and run **offline**; their outputs are cached. Training starts from the cached features in §3.

---

## 3. Frozen encoders

### 3.1 CLIP ViT-B/16, image tower (unchanged)

| Stage | Input | Output |
|---|---|---|
| Patch embed (Conv2d 16 × 16, stride 16) | 3 × 224 × 224 | 196 × 768 |
| + [CLS] + positional embedding | 196 × 768 | 197 × 768 |
| 12 Transformer blocks (width 768, 12 heads, MLP 3072) | 197 × 768 | 197 × 768 |
| LN([CLS]) → projection 768 → 512 | 768 | **`x_t ∈ ℝ^512`** |

Cached: **`X ∈ B × L × 512`**. Mean ‖x‖ on T2 ≈ 9.87, i.e. ≈ 0.44 per channel.

### 3.2 VideoMAE V2-S, video tower (adopted in v2)

The public distilled checkpoint: ViT-S, K710 post-trained, distilled from ViT-g.

| Stage | Input | Output |
|---|---|---|
| Tubelet embed (Conv3d 2 × 16 × 16, stride 2 × 16 × 16) | 3 × 16 × 224 × 224 | 8 × 14 × 14 = 1,568 tokens × 384 |
| + fixed sin-cos positional embedding | 1,568 × 384 | 1,568 × 384 |
| 12 blocks, joint space-time attention (width 384, 6 heads, MLP 1536) | 1,568 × 384 | 1,568 × 384 |
| Mean over tokens → final LayerNorm; the classifier head is removed | 1,568 × 384 | **`u_t ∈ ℝ^384`** |

Cached: **`U ∈ B × L × 384`**. Cost ≈ 57 GFLOPs per step.

### 3.3 CLIP ViT-B/16, text tower (unchanged), with LaGoVAD soft prompts (trainable)

| Stage | Input | Output |
|---|---|---|
| Sample one definition string per class (LaGoVAD's verbalizer), tokenize | C strings | C × ≤ 45 tokens |
| Token embedding, with 32 learnable soft-prompt vectors inserted: 16 after the start token, 16 after the content | C × ≤ 45 | C × ≤ 77 × 512 |
| 12 blocks (width 512, 8 heads, causal mask) | C × 77 × 512 | C × 77 × 512 |
| LN(EOT) → projection 512 → 512 | C × 512 | **`z ∈ ℝ^{C×512}`** |

`z` is **not** cached. In training it is recomputed at every step, because the soft prompts train and a new definition is sampled per step. At inference it is recomputed per forward pass with a freshly sampled definition.

---

## 4. Clip-Referenced Normalization (CRN), R2 — parameter-free

**Reference unit:** the window's **source video** during training; the **whole clip** at test time.

**Reference:** the per-dimension median over all steps of that unit (R2). v2 also evaluated the mean, a robust mean and a past-only mean; R2 was chosen (results report §3). R2 needs the whole clip, so the pipeline is not streamable.

| Operation | Input | Output |
|---|---|---|
| Reference per unit: `μ^x = median_t x_t`, `μ^u = median_t u_t` (per dimension) | all steps of the unit | `B × 1 × 512`, `B × 1 × 384` |
| CLIP stream: `x̃ = s · (X − μ^x)`, where `s` is one scalar fixed on T2-train so that the mean ‖x̃‖ equals the mean ‖x‖ | `B×L×512` and `μ^x` | **`X̃ ∈ B × L × 512`** |
| Video stream: `ũ = U − μ^u` | `B×L×384` and `μ^u` | **`Ũ ∈ B × L × 384`** |

- **No learned parameters.**
- CRN, `s`, `c` and `σ_u` are applied **offline**: `build_v2_inputs` bakes one row `[x̃ ; c·ũ⊘σ_u]` (512 + 384) per step into a v2 input cache with a manifest. Training and evaluation refuse a cache whose manifest does not name the run's `v2.crn` / `v2.motion`.
- References are computed per source video (training) or per clip (DoTA, over the stride-3 rows), so a T2 window only slices its source's baked rows.
- With DVS on, each spliced segment therefore carries its **own source's** reference; nothing is re-centred after splicing.

---

## 5. Video-stream fusion — the only new trainable block in the deployed network

| Operation | Input | Output |
|---|---|---|
| Fixed scaling: `c · ũ ⊘ σ_u`, where `σ_u ∈ ℝ^384` is the per-channel std of `ũ` over T2-train steps and `c` is the CLIP input's per-channel RMS on T2-train (E‖x‖ / √512 ≈ 0.44). Both are computed once and frozen. | `Ũ ∈ B×L×384` | `B × L × 384` |
| Linear `W_u ∈ ℝ^{384×512}` (+ bias), **initialized to zero** (≈ 0.20 M parameters) | `B × L × 384` | `B × L × 512` |
| Residual add onto the CLIP stream: `h_t = x̃_t + W_u·(c · ũ_t ⊘ σ_u)` | `X̃`, the projected video term | **`H ∈ B × L × 512`** |

**Initialization.** At initialization `H = X̃` exactly, so training starts from the CRN-only function. It does not stay there by construction: under AdamW the first updates are ≈ lr · sign(g), so the video term can reach the CLIP channel size within ≈ 30 steps. The **video-stream share** `ρ_u = ‖W_u(c·ũ⊘σ_u)‖ / ‖x̃‖`, computed as a Frobenius ratio over the batch's valid steps (the "motion share" in v2), is logged at **every** training step in `metrics.jsonl` (`motion_share`, beside `w_u_norm` = `‖W_u‖_F`) as the evidence that the stream is used. Padded steps get no video term, so the bias does not leak into padding.

**No LayerNorm here, on purpose.**
- A per-token LayerNorm divides each step by its own norm. After CRN that norm is the step's deviation from the clip reference, so LayerNorm would make a barely-deviating step and a strongly-deviating one look the same.
- `σ_u` is one dataset-level constant per channel, so relative magnitudes between steps survive **into `V^t`**.
- The temporal encoder's layers are post-LN, but the encoder wraps them in an outer residual, `V^t = H + Enc(H)`. `H` therefore reaches `V^t` un-normalized; only the `Enc(H)` branch is re-normalized.

**Where the magnitude goes after `V^t`:**
- **Its size relative to the normalized branch.** `Enc(H)` ends in a LayerNorm, so its per-token norm is ≈ √512 ≈ 22.6 at gain ≈ 1 (the gain is trained). ‖H‖ ≈ 9.87, so the un-normalized part enters at ≈ 0.44× the other. The heads can weight it, but it does not dominate.
- **The co-attention re-normalizes it.** Each co-attention layer is post-LN, and the layers are stacked with no outer skip, so `V^u` is re-normalized.
- **So it reaches one of the two `H_bin` paths.** The language-agnostic path reads `V^t` and keeps the magnitude. The language-guided path and `H_mul` read `V^u` and see only its direction and context.
- **SG-NM also reads `V^t`** (§9), detached.

---

## 6. Temporal encoder (LaGoVAD, unchanged)

| Stage | Input | Output |
|---|---|---|
| 2 RoFormer layers, RoPE over time (max 1,536 positions), width 512, 4 heads, FFN 2048 (GELU), dropout 0.1, post-LN; **local attention band of 25 steps** (each step attends to ±12 valid steps); outer residual `V^t = H + Enc(H)` (the optional gate on `Enc(H)` is off) | `H ∈ B×L×512` | **`V^t ∈ B × L × 512`** |

On an unspliced 20-step window the band covers the whole window. On a DVS sequence (≤ 100 steps) or a DoTA clip (median ≈ 33 steps at stride 3), each step sees only its ±12 neighbours.

---

## 7. Co-attention fusion `CoAttn` (LaGoVAD, unchanged)

| Stage | Input | Output |
|---|---|---|
| Broadcast the definition embedding | `z ∈ C×512` | `B × C × 512` |
| 2 co-attention layers, 8 heads, FFN 2048 (GELU), dropout 0.1. Each layer: text attends to vision (padding masked) and vision attends to text, each as `LN(x + attn)`, then `LN(x + FFN(x))` per stream. No skip around the stack. | `V^t ∈ B×L×512`, `B×C×512` | **`V^u ∈ B × L × 512`**, **`Z^u ∈ B × C × 512`** |

---

## 8. Heads (LaGoVAD, unchanged)

**`H_bin`, the detection head:**

| Stage | Input | Output |
|---|---|---|
| Language-agnostic path: one Conv1d over time on `V^t`, 512 → 1, kernel 3, replicate padding | `B × 512 × L` | `a ∈ B × L` |
| Language-guided path: same structure on `V^u` | `B × 512 × L` | `g ∈ B × L` |
| Learnable scalar blend: `y^bin = w·a + (1 − w)·g`, `w = σ(10·α)`, `α` initialized to 0.25, so `w` ≈ 0.92 at start | `a`, `g` | **`y^bin ∈ B × L`** (logits); `s = σ(y^bin)` |

The language-agnostic path, which reads `V^t` and keeps CRN's deviation size, starts with ≈ 92 % of the weight.

**`H_mul`, the classification head:**

| Stage | Input | Output |
|---|---|---|
| Cosine similarity between `V^u` and `Z^u` ÷ a learnable temperature (initialized to 0.2); no projection layers | `V^u`, `Z^u` | **`y^mul ∈ B × L × C`** |
| Video level (in `L_MIL-align`): per class, the mean of the top-k similarities over time, k = max(1, ⌊L'/16⌋) (1 on a 20-step window), then cross-entropy against the class index (Normal = 0) | `B × L × C` | `B × C` |

---

## 9. SG-NM branch — training only (new in v2.1; step E4)

Self-Guided Normality Modeling, from DSANet, sized for 20-step windows and made safe for this trunk. **Every input is detached**, so the branch's own losses reach only the branch. The one term that reaches the detector is `L_consist`, through `y^bin`.

**Inputs**

| Tensor | Shape | Source |
|---|---|---|
| `V̄ = sg(V^t)` | `B × L × 512` | temporal encoder (§6), detached |
| `s̄ = sg(σ(y^bin))` | `B × L` | `H_bin` (§8), detached; used only to choose candidates |
| segment labels | one per window segment | 0 = normal, 1 = abnormal. An unspliced sequence is one segment. A DVS-spliced sequence has one segment per window: the anchor window and its normal filler windows. |
| valid-step mask | `B × L` | padding mask |

**Stages**

| # | Stage | Input | Output |
|---|---|---|---|
| 1 | **Candidate normal steps, per window segment.** Normal segment: every valid step. Abnormal segment: the ⌈0.4 · 20⌉ = 8 valid steps with the lowest `s̄` inside that segment. | `s̄`, segment labels, mask | `m_n ∈ {0,1}^{B×L}` |
| 2 | **Prototype extraction.** Learned queries `Q ∈ ℝ^{2×512}`; one multi-head cross-attention layer (8 heads; queries `Q`, keys = values = `V̄`, key mask `m_n`), then LN | `Q` (broadcast to `B×2×512`), `V̄`, `m_n` | **`P ∈ B × 2 × 512`** |
| 3 | **Compactness loss.** For each candidate step, the cosine distance to its nearest prototype; mean over candidates | `V̄`, `P`, `m_n` | `B × L × 2` → **`L_compact`** (scalar) |
| 4 | **Decoder queries.** `MLP(V̄)`: 512 → 512 → 512, GELU | `V̄` | `G ∈ B × L × 512` |
| 5 | **Decoder layer 1.** `R1 = LN(MHA(G, P, P))`, with **no residual from `G`**; then `R1 = LN(R1 + FFN(R1))`, FFN 512 → 1024 → 512 | `G`, `P` | `R1 ∈ B × L × 512` |
| 6 | **Decoder layer 2.** `R2 = LN(R1 + MHA(R1, P, P))`; then `R = LN(R2 + FFN(R2))` | `R1`, `P` | **`R ∈ B × L × 512`** |
| 7 | **Reconstruction error.** `e = 1 − cos(R, V̄)`, in [0, 2] | `R`, `V̄` | `e ∈ B × L` |
| 8 | **Reconstruction loss.** Mean of `e` over candidate steps: the decoder learns to rebuild normal steps | `e`, `m_n` | **`L_rec`** (scalar) |
| 9 | **Normality score, dataset-level.** `S̃ = clip((e − μ_e) / (3·σ_e), 0, 1)`. `μ_e` and `σ_e` are running means (EMA 0.99) of the mean and std of `e` over valid steps of normal segments, detached. **No per-window min-max.** | `e` | **`S̃ ∈ B × L`** |
| 10 | **Consistency loss** (one-way). `L_consist = mean over valid steps of (σ(y^bin) − sg(S̃))²` | `y^bin` (not detached), `S̃` | **`L_consist`** (scalar) |

**Why the reconstruction error stays informative.**
- Stage 5 has no residual, so its output depends on a step only through its attention weights over the 2 prototypes: one number per head, 8 in total.
- Stages 5–6 after that are per-step functions of those numbers.
- Each sequence's reconstructions therefore lie on an at-most-8-dimensional family inside the 512-d space. Steps far from the sequence's normal prototypes keep a high error.

**Schedule:**
- **Stages 1–9 run from step 0.**
- **Stage 10 (`L_consist`) starts at step 174**, the end of epoch 2 of 20 (87 steps per epoch, 1,740 in total).
- **`λ_n`** is fixed once in a pilot on seed 2024, with `λ_n` = 1, run to step 300. `ρ = ‖g_consist‖ / ‖g_task‖` is measured on the detector parameters `L_consist` reaches (`W_u`, temporal encoder, co-attention, soft prompts, `H_bin`) at steps 200, 250 and 300. `g_task` is the gradient of `L_MIL + L_MIL-align + L_dvs`. Then `λ_n = min(1, 0.2 / median ρ)`. The pilot run is discarded.
- **ρ is logged every 50 steps** in every E4 run. A run with ρ > 0.3 at any logged step cannot be adopted.

**Diagnostics logged:**
- ρ;
- the macro AUC of `S̃` alone;
- the mean top-k score on T2-val normal windows ("normal-window peak");
- prototype usage: the share of candidates nearest each prototype. More than 95 % on one prototype means a collapse to K = 1;
- `L_dvs` and `L_MIL` trajectories against A3's.

**At inference:** the branch is not run.

**For the E4a gate only** (proposal §11.2): the branch is trained on frozen `V^t` and `y^bin` from A3's five E3 checkpoints, with the same schedule and training stream. It is then run on whole DoTA-CAP-dev clips (`T` steps, protocol B). Candidates are chosen without labels: the 40 % of the clip's steps with the lowest `y^bin`. `S̃` uses the `μ_e`, `σ_e` frozen at the end of branch training.

---

## 10. Losses

| Loss | Arms | Input | Shape used | Reaches |
|---|---|---|---|---|
| `L_MIL`: BCE on the mean of the top-k **logits** of `y^bin` per sequence, k = max(1, ⌊L'/5⌋), against the sequence label | all | `y^bin` | `B × L'` → `B` | `W_u`, temporal encoder, co-attention, soft prompts, `H_bin` |
| `L_MIL-align`: per-class top-k mean over time (k = max(1, ⌊L'/16⌋)), cross-entropy against the class index | all | `y^mul` | `B × L' × C` → `B × C` | `W_u`, temporal encoder, co-attention, soft prompts, `H_mul` |
| `L_dvs` = `L_dvs-sup` + `L_dvs-mil`, on every normal sequence and every DVS-spliced sequence (θ = 0.7 = probability of **no** splice; δ_m = 5), masked. `L_dvs-sup`: per-step BCE vs `y^p`. `L_dvs-mil`: top-k (k = ⌊n_pos/4⌋) inside the `y^p` = 1 steps, or over all steps of a normal sequence. With `dvs_anchor_mode = span` (A3), `y^p` = 1 on the **whole** anchor window of a spliced abnormal sequence (lesson C29). | all | `y^bin` | `B × L'` | as `L_MIL` |
| `L_neg` | — | — | **off**: the T2 training configuration provides no caption features, as in phase 4 and v2 (activation recipe: proposal DF3) | — |
| `λ_n · L_consist`, from step 174 | **E4 only** | `y^bin`, `sg(S̃)` | `B × L` | as `L_MIL` (through `y^bin`) |
| `L_compact + L_rec` | **E4 only** | `sg(V^t)` | `B × L × 512` | **SG-NM branch only** |

---

## 11. Inference (one clip of `T` steps; protocol B)

| Step | Input | Output |
|---|---|---|
| Encode every step (§3) | `T` steps | `X ∈ T × 512`, `U ∈ T × 384`; `z ∈ C × 512` (re-encoded per pass with a sampled definition, §3.3) |
| CRN with R2, the per-dimension median over all `T` steps | `T × d` | `X̃ ∈ T × 512`, `Ũ ∈ T × 384` |
| Fusion (§5) | `X̃`, `Ũ` | `H ∈ T × 512` |
| The whole clip in one pass through the trunk and heads (§6–§8); clips longer than 512 steps would be split into 512-step chunks (DoTA clips never are) | `1 × T × 896` baked rows | **`y^bin ∈ T`**, **`y^mul ∈ T × C`** |
| `σ(y^bin)` interpolated linearly to native frames (step `t` at frame `3t`); per-clip min-max (benchmark micro) or a threshold (deployment). Benchmark reads average the seeds per frame. | `T` | frame-level anomaly curve |

- **SG-NM is not run.**
- R2 and the whole-clip pass both need the full clip, so the pipeline is not streamable.
- The ATS → MLLM report (Holmes-VAU) is optional and asynchronous. It reads `y^bin` and the frames. It is designed only; nothing in `core/` implements it.

---

## 12. Arms (same network; switches only)

| Arm | CRN (R2) | Video stream | SG-NM (training) | `H` fed to the temporal encoder | New parameters |
|---|:-:|:-:|:-:|---|---|
| A0 (phase-4 KIP-off; reference) | – | – | – | `X` | 0 |
| **A3 (v2, adopted; the E4 control)** | ✓ | ✓ | – | `s·(X − μ^x) + W_u·(c·(U − μ^u) ⊘ σ_u)` | `W_u` (≈ 0.20 M) |
| **A3-ign** (E4-0) | ✓ | ✓ | – | as A3; trained with `dvs_anchor_mode = ignore` (the anchor's steps leave the dense `L_dvs` BCE) | as A3 |
| **A3 + SG-NM** (E4) | ✓ | ✓ | ✓ | as A3 | as A3, plus the training-only branch (≈ 5.8 M) |

- **Base `B`.** E4 adds SG-NM to the base `B` ∈ {A3, A3-ign} chosen by the proposal's base rule (§11.4) and keeps `B`'s `dvs_anchor_mode`. Wherever §9 says "A3" for the E4a checkpoints or the E4 control, read `B`.

- A1 (CRN only) and A2 (video stream only) were v2's factorial arms. Their results are in the results report §4.
- E4 runs only if the E4a gate passes. It is adopted under the auxiliary rule (proposal §11.3).

---

## 13. Master shape table (training, one batch, A3 + SG-NM)

| # | Component | Input | Output | Params |
|---|---|---|---|---|
| 1 | CLIP ViT-B/16 image | `B×L×3×224×224` | `X: B×L×512` | frozen |
| 2 | VideoMAE V2-S | `B×L×3×16×224×224` | `U: B×L×384` | frozen |
| 3 | CLIP text + soft prompts | `C×77` | `z: C×512` | tower frozen; prompts trained |
| 4 | CRN, R2 (both streams) | `X`, `U` + references | `X̃: B×L×512`, `Ũ: B×L×384` | none |
| 5 | Video fusion (fixed `c·(·)⊘σ_u` → `W_u`, zero-init → add) | `X̃`, `Ũ` | `H: B×L×512` | **≈ 0.20 M**; `σ_u`, `c` fixed |
| 6 | Temporal encoder | `H` | `V^t: B×L×512` | trained (unchanged) |
| 7 | Co-attention `CoAttn` | `V^t`, `z` | `V^u: B×L×512`, `Z^u: B×C×512` | trained (unchanged) |
| 8 | `H_bin` | `V^t`, `V^u` | `y^bin: B×L` | trained (unchanged) |
| 9 | `H_mul` | `V^u`, `Z^u` | `y^mul: B×L×C` → `B×C` | trained (unchanged) |
| 10 | SG-NM: candidates | `sg(σ(y^bin))`, segment labels | `m_n: B×L` | none |
| 11 | SG-NM: prototypes | `sg(V^t)`, `m_n` | `P: B×2×512` | **≈ 1.05 M, training only** |
| 12 | SG-NM: decoder (MLP + 2 layers) | `sg(V^t)`, `P` | `R: B×L×512` | **≈ 4.7 M, training only** |
| 13 | SG-NM: score | `R`, `sg(V^t)` | `e`, `S̃: B×L` | none (EMA statistics) |
| 14 | Losses (§10) | `y^bin`, `y^mul`, `S̃`, `e`, `P` | scalars | — |

SG-NM parameters: queries 2 × 512; prototype attention ≈ 4 · 512² ≈ 1.05 M; MLP ≈ 2 · 512² ≈ 0.52 M; each decoder layer ≈ 1.05 M (attention) + 1.05 M (FFN 512 → 1024 → 512), × 2. Total ≈ **5.8 M**, all training-only.

---

## 14. Frozen vs trained, and what was removed

| Frozen | Trained, deployed | Trained, training only (E4) | Removed from v1 |
|---|---|---|---|
| CLIP image and text towers; VideoMAE V2-S | soft prompts; `W_u` (`σ_u`, `c` and `s` are fixed statistics); temporal encoder; co-attention; `H_bin`; `H_mul` | SG-NM branch: queries `Q`, prototype attention, MLP, decoder (≈ 5.8 M) | KIP (PMG flow head, integer-cast gate, 50 % shift, motion head, `L_KIP-rec/align`, `L_kin`), the stage-1 warm-up, RAFT |

- **Deployed network:** two frozen encoders, plus LaGoVAD's trunk, plus ≈ 0.20 M new parameters (`W_u`). Identical with or without SG-NM.
- **Cost per step:** CLIP ≈ 17.5 + VideoMAE V2-S ≈ 57 ≈ 75 GFLOPs; ≈ 250–280 GFLOPs/s at 3.33–3.75 steps/s. The trunk and heads are negligible.
- **Inference is RGB only.**
