# KAT-VAD v2 — Model Architecture (with tensor shapes)

**Scope:** the network only. It covers what each block is, what goes in and comes out, and what is frozen or trained. The reasons, evidence and experiment plan are in `KAT_VAD_PROPOSAL_v2.md`.

**The model:** LaGoVAD's trunk, unchanged, fed by
- a frozen CLIP ViT-B/16 frame stream (as before),
- a new frozen **VideoMAE V2 motion stream**, scaled by a fixed per-channel `σ_u` and one fixed scalar `c`, and added through a zero-initialized residual,
- with both streams passed through **Clip-Referenced Normalization (CRN)**.

There are no new losses, and nothing is added to the trunk or heads.

**Conventions:**
- Shapes are written `batch × time × channels`.
- Values marked **‹code›** are unchanged LaGoVAD / KIP-off settings: take them from the current implementation.

---

## 0. Symbols

| Symbol | Meaning | Value |
|---|---|---|
| `B` | windows per training batch | 64 (32 abnormal + 32 normal) |
| `L` | steps per window | 20 |
| `T` | steps in a full test clip | variable (DoTA median ≈ 35 at stride 3) |
| `r` | step rate | 3.75 Hz (30 fps sources, stride 8) · 3.33 Hz (10 fps sources, stride 3) |
| `D` | trunk width | 512 |
| `d_v` | VideoMAE V2 feature size | 768 (ViT-B) · 384 (ViT-S) |
| `C` | categories in the definition `Z` | ‹code› |
| `k` | MIL top-k | 4 |

---

## 1. Overview

```
video ─► steps @ r ─┬─ frame @ t ─────────────► CLIP ViT-B/16 image (frozen) ─► x_t ∈ ℝ^512 ─► CRN ─► x̃_t ∈ ℝ^512 ───────────┐
                    │                                                                                                      (+) ─► h_t ∈ ℝ^512
                    └─ causal clip 16f@10fps ─► VideoMAE V2 (frozen) ─────────► u_t ∈ ℝ^dv ─► CRN ─► c·(⊘σ_u) ─► W_u (0-init) ┘
                                                                                     ▼
h ∈ B×L×512 ─► temporal encoder (2 layers, RoPE)  ─► V^t ∈ B×L×512
Z ─► CLIP text (frozen) + soft prompts ─► z ∈ C×512 ─► co-attention CoAttn(V^t, z) ─► V^u ∈ B×L×512, Z^u ∈ B×C×512
H_bin(V^t, V^u) ─► y^bin ∈ B×L          H_mul(V^u, Z^u) ─► y^mul ∈ B×L×C
```

---

## 2. Input sampling

| Item | Value |
|---|---|
| Step rate | 3.75 Hz for DADA (30 fps, stride 8) · 3.33 Hz for DoTA (10 fps, stride 3, if E1 adopts it; otherwise stride 8) |
| Frame input (per step) | the frame at the step, full frame resized to 224 × 224, no centre crop |
| Clip input (per step) | 16 frames at 10 fps **ending at** the step (causal, 1.5 s). DADA: every 3rd native frame; DoTA: native frames. Full frame squashed to 224 × 224, as the CLIP stream (no geometry probe; proposal §4.1). VideoMAE mean/std. |
| Training item | a T2 window of `L` = 20 consecutive steps of one source video |
| Per-batch raw input | frames `B × L × 3 × 224 × 224`; clips `B × L × 3 × 16 × 224 × 224` |

Both encoders are frozen and run **offline**; their outputs are cached. Training starts from the cached features in §3.

---

## 3. Frozen encoders

### 3.1 CLIP ViT-B/16, image tower (unchanged from KIP-off)

| Stage | Input | Output |
|---|---|---|
| Patch embed (Conv2d 16 × 16, stride 16) | 3 × 224 × 224 | 196 × 768 |
| + [CLS] + positional embedding | 196 × 768 | 197 × 768 |
| 12 Transformer blocks (width 768, 12 heads, MLP 3072) | 197 × 768 | 197 × 768 |
| LN([CLS]) → projection 768 → 512 | 768 | **`x_t ∈ ℝ^512`** |

Cached: **`X ∈ B × L × 512`**.

### 3.2 VideoMAE V2, video tower (new; distilled K710 checkpoint)

| Stage | Input | Output |
|---|---|---|
| Tubelet embed (Conv3d 2 × 16 × 16, stride 2 × 16 × 16) | 3 × 16 × 224 × 224 | 8 × 14 × 14 = 1,568 tokens × 768 |
| + fixed sin-cos positional embedding | 1,568 × 768 | 1,568 × 768 |
| 12 blocks, joint space-time attention (width 768, 12 heads, MLP 3072) | 1,568 × 768 | 1,568 × 768 |
| Mean over tokens → fc_norm (LN); the classifier head is removed | 1,568 × 768 | **`u_t ∈ ℝ^768`** |

ViT-S variant: width 384, 6 heads, MLP 1536, so `u_t ∈ ℝ^384`. E2 picks B or S.

Cached: **`U ∈ B × L × d_v`**.

### 3.3 CLIP ViT-B/16, text tower (unchanged), with LaGoVAD soft prompts (trainable)

| Stage | Input | Output |
|---|---|---|
| Tokenize `C` definition strings | C strings | C × 77 |
| Token embedding, with ‹code› learnable soft-prompt vectors inserted | C × 77 | C × 77 × 512 |
| 12 blocks (width 512, 8 heads, causal mask) | C × 77 × 512 | C × 77 × 512 |
| LN(EOT) → projection 512 → 512 | C × 512 | **`z ∈ ℝ^{C×512}`** (computed once per definition set) |

---

## 4. Clip-Referenced Normalization (CRN) — parameter-free

**Reference unit:** the window's **source video** during training; the **whole clip** at test time.

**Reference form:** one of four, fixed by the E2 check before training.

| Form | Computed over | Shape per window |
|---|---|---|
| R1 mean | all steps of the source video / clip | `B × 1 × d` |
| R2 median (per dimension) | all steps | `B × 1 × d` |
| R3 robust mean | the 50 % of steps closest (ℓ2) to R2 | `B × 1 × d` |
| R4 past-only | mean of the strictly past steps τ < t for t ≥ `N_w`; the mean of the first `N_w` steps for t < `N_w` (`N_w` = 8; streamable after the warm-up) | `B × L × d` (time-varying) |

| Operation | Input | Output |
|---|---|---|
| CLIP stream: `x̃ = s · (X − μ^x)`, where `s` is one scalar fixed on T2-train so that the mean ‖x̃‖ equals the mean ‖x‖ | `B×L×512` and the reference | **`X̃ ∈ B × L × 512`** |
| Motion stream: `ũ = U − μ^u` (arm A2, no CRN: `ũ = U − m_u`, with `m_u` the T2-train channel mean) | `B×L×d_v` and the reference | **`Ũ ∈ B × L × d_v`** |

- **No learned parameters.**
- References are precomputed per source video (training) or per clip (test), so a window only slices them.

---

## 5. Motion-stream fusion — the only new trainable block

| Operation | Input | Output |
|---|---|---|
| Fixed scaling: `c · ũ ⊘ σ_u`, where `σ_u ∈ ℝ^{d_v}` is the per-channel std of `ũ` over T2-train steps and `c` is the CLIP input's per-channel RMS on T2-train (E‖x‖/√512 ≈ 0.44); both computed once and frozen (not trained) | `Ũ ∈ B×L×d_v` | `B × L × d_v` |
| Linear `W_u ∈ ℝ^{d_v×512}` (+ bias), **initialized to zero** | `B × L × d_v` | `B × L × 512` |
| Residual add onto the CLIP stream: `h_t = x̃_t + W_u·(c · ũ_t ⊘ σ_u)` | `X̃`, the projected motion term | **`H ∈ B × L × 512`** |

At initialization `H = X̃` exactly, so the model starts from the KIP-off function (plus CRN). It does not stay there by construction: under AdamW the first updates are ≈ lr·sign(g), so the motion term can reach the CLIP channel size within ≈ 30 steps. The **motion share** `ρ_u = ‖W_u(c·ũ⊘σ_u)‖ / ‖x̃‖` is logged every 50 steps as the evidence that the stream is used.

**No LayerNorm here, on purpose.** A per-token LayerNorm divides each step by its own norm. After CRN that norm is the step's deviation from the clip reference, so LayerNorm would make a barely-deviating step and a strongly-deviating one look the same. `σ_u` is one dataset-level constant per channel, so relative magnitudes between steps survive **into `V^t`**. The temporal encoder's layers are post-LN (`core/models/temporal_encoder.py:118`, `:161`), but the encoder wraps them in an outer residual, `V^t = H + Enc(H)` (`temporal_encoder.py:287`). `H` therefore reaches `V^t` un-normalized; only the `Enc(H)` branch is re-normalized.

**Where the magnitude goes after `V^t`:**
- **Its size relative to the normalized branch.** `Enc(H)` ends in a LayerNorm, so its per-token norm is ≈ √512 ≈ 22.6 while the gain is ≈ 1 (its initial value; it is trained). ‖H‖ ≈ 9.87, so the un-normalized part is present at ≈ 0.44× the other. The heads can weight it, but it does not dominate.
- **The co-attention re-normalizes it.** `CoAttnFusionLayer` is post-LN (`vis_norm1`, `vis_norm2`; `core/models/fusion.py:56–58`) and `CoAttentionFusion` stacks the layers with no outer skip (`fusion.py:84–86`), so `V^u` is re-normalized.
- **So it reaches one of the two `H_bin` paths.** `H_bin`'s language-agnostic path reads `V^t` (`before_fused`, `core/models/kat_vad.py:154`) and keeps the magnitude. The language-guided path and `H_mul` read `V^u` and see only its direction and context.

---

## 6. Temporal encoder (LaGoVAD, unchanged)

| Stage | Input | Output |
|---|---|---|
| 2 Transformer encoder layers, RoPE over time, width 512; heads and FFN ‹code› | `H ∈ B×L×512` | **`V^t ∈ B × L × 512`** |

---

## 7. Co-attention fusion `CoAttn` (LaGoVAD, unchanged)

| Stage | Input | Output |
|---|---|---|
| Broadcast the definition embedding | `z ∈ C×512` | `B × C × 512` |
| 2 co-attention layers: vision attends to text, text attends to vision, then FFN (heads / FFN ‹code›) | `V^t ∈ B×L×512`, `B×C×512` | **`V^u ∈ B × L × 512`**, **`Z^u ∈ B × C × 512`** |

---

## 8. Heads (LaGoVAD, unchanged)

**`H_bin`, the detection head:**

| Stage | Input | Output |
|---|---|---|
| Language-agnostic path: Conv1d over time on `V^t` (kernel 3, replicate padding; layers ‹code›) | `B × 512 × L` | `a ∈ B × L` |
| Language-guided path: same structure on `V^u` | `B × 512 × L` | `g ∈ B × L` |
| Learnable scalar blend | `a`, `g` | **`y^bin ∈ B × L`** (logits); `s = σ(y^bin)` |

**`H_mul`, the classification head:**

| Stage | Input | Output |
|---|---|---|
| Linear projections (512 → 512) and cosine similarity ÷ temperature | `V^u`, `Z^u` | **`y^mul ∈ B × L × C`** |
| Video level: min over time for the normal class, max over time for accident classes, then softmax | `B × L × C` | `B × C` |

---

## 9. Losses (KIP-off set, unchanged; no new terms)

| Loss | Input | Shape used |
|---|---|---|
| `L_MIL`: top-k (k = 4) mean of `s` per window, BCE against the window label | `y^bin` | `B × L` → `B` |
| `L_MIL-align`: video-level classification | `y^mul` | `B × L × C` → `B × C` |
| `L_dvs` (only if the KIP-off config enables it): spliced sequences of length `L' ≤ δ_m·L`, masked. CRN is applied per segment **before** splicing. | `y^bin` | `B × L'` |
| `L_neg` | — | **off**, as in the KIP-off T2 config |

---

## 10. Inference (one clip of `T` steps)

| Step | Input | Output |
|---|---|---|
| Encode every step (§3) | T steps | `X ∈ T × 512`, `U ∈ T × d_v`; `z ∈ C × 512` (cached) |
| CRN with the chosen reference over the clip (R4: strictly past, streamable after the `N_w`-step warm-up) | `T × d` | `X̃ ∈ T × 512`, `Ũ ∈ T × d_v` |
| Fusion (§5) | `X̃`, `Ũ` | `H ∈ T × 512` |
| Sliding windows: W = 20, hop 4, last window aligned to the clip end; `n_w = max(1, ⌈(T − 20)/4⌉ + 1)` | `T × 512` | `n_w × 20 × 512` (DoTA median: T ≈ 35 → n_w = 5) |
| Trunk and heads (§6–§8) | `n_w × 20 × 512` | `y^bin ∈ n_w × 20`, `y^mul ∈ n_w × 20 × C` |
| Overlap-average onto the clip timeline | `n_w × 20` | **`y^bin ∈ T`**, **`y^mul ∈ T × C`** |
| Interpolate to native frames; per-clip min-max (benchmark) or a threshold (deployment) | `T` | frame-level anomaly curve |

- Clips with `T ≤ 20` run as a single window.
- If E1 does not adopt rate matching, DoTA is read at stride 8 and runs as a whole clip, as now.
- The ATS → MLLM report is optional and asynchronous. It reads `y^bin` and the frames.

---

## 11. The four experimental arms (same network; switches only)

| Arm | CRN | Motion stream | `H` fed to the temporal encoder | New parameters |
|---|:-:|:-:|---|---|
| A0 (KIP-off) | – | – | `X` | 0 |
| A1 | ✓ | – | `s·(X − μ^x)` | 0 |
| A2 | – | ✓ | `X + W_u·(c·(U − m_u) ⊘ σ_u)` | `W_u` |
| A3 (full v2) | ✓ | ✓ | `s·(X − μ^x) + W_u·(c·(U − μ^u) ⊘ σ_u)` | `W_u` |

`σ_u` is computed per arm on the representation that arm feeds (`U − m_u` for A2, `U − μ^u` for A3). Model selection between arms follows the proposal §10.3: a costly arm (A2 or A3) needs its contrast against A0 **and** against the best adoptable free arm `F` (A1 if A1 passes its rule, A0 otherwise) to exclude 0.

---

## 12. Master shape table (training, one batch, full v2)

| # | Component | Input | Output | Params |
|---|---|---|---|---|
| 1 | CLIP ViT-B/16 image | `B×L×3×224×224` | `X: B×L×512` | frozen |
| 2 | VideoMAE V2 (B or S) | `B×L×3×16×224×224` | `U: B×L×d_v` | frozen |
| 3 | CLIP text + soft prompts | `C×77` | `z: C×512` | tower frozen; prompts trained |
| 4 | CRN (both streams) | `X`, `U` + references | `X̃: B×L×512`, `Ũ: B×L×d_v` | none |
| 5 | Motion fusion (fixed `c·(·)⊘σ_u` → `W_u`, zero-init → add) | `X̃`, `Ũ` | `H: B×L×512` | **new: ≈ 0.39 M (B) · 0.20 M (S)**; `σ_u` fixed |
| 6 | Temporal encoder | `H` | `V^t: B×L×512` | trained (unchanged) |
| 7 | Co-attention `CoAttn` | `V^t`, `z` | `V^u: B×L×512`, `Z^u: B×C×512` | trained (unchanged) |
| 8 | `H_bin` | `V^t`, `V^u` | `y^bin: B×L` | trained (unchanged) |
| 9 | `H_mul` | `V^u`, `Z^u` | `y^mul: B×L×C` → `B×C` | trained (unchanged) |
| 10 | Losses (§9) | `y^bin`, `y^mul` | scalar | — |

---

## 13. Frozen vs trained, and what was removed

| Frozen | Trained | Removed from v1 |
|---|---|---|
| CLIP image and text towers; VideoMAE V2 | soft prompts; motion `W_u` (new; `σ_u` and `c` are fixed statistics); temporal encoder; co-attention; `H_bin`; `H_mul` | KIP (PMG flow head, integer-cast gate, 50 % shift, motion head, `L_KIP-rec/align`, `L_kin`), the stage-1 warm-up, RAFT |

- **Deployed network:** two frozen encoders, plus LaGoVAD's trunk, plus ≈ 0.2–0.4 M new parameters.
- **Inference is RGB only.**
