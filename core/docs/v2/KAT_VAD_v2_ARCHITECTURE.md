# KAT-VAD v2 — Model Architecture (component-by-component, with tensor shapes)

**Scope:** the network only. What each block is, what goes in, what comes out, and what is frozen or trained. The reasons for each choice, the evidence and the experiment plan are in `KAT_VAD_PROPOSAL_v2.md`.

**Two encoder options.** v2 has two ways to feed the network. The frozen-feature probe (step E2 of the proposal) chooses between them, and ties go to Option A. Everything from §4 onward is identical in both.
- **Option A — single text-aligned stream:** one frozen video-text encoder (ViCLIP / InternVideo2-CLIP). It provides both the per-step video feature and the text tower for the definitions.
- **Option B — two visual streams:** frozen CLIP ViT-B/16 on frames, plus frozen VideoMAE V2 on clips. CLIP's text tower is used for the definitions.

**Conventions:**
- Shapes are written `batch × time × channels`. The batch dimension is dropped where a tensor is per-video or shared.
- Values marked **‹code›** are unchanged LaGoVAD / KIP-off settings: take them from the current implementation.
- Encoder internals marked **(public)** come from the released model definitions.

---

## 0. Symbols

| Symbol | Meaning | Value |
|---|---|---|
| `B` | training windows per batch | 64 (32 abnormal + 32 normal, same-source paired) |
| `L` | steps per window | 20 |
| `T` | steps in a full test clip | variable (DoTA median ≈ 35 at stride 3) |
| `r` | step rate | 3.75 Hz (30 fps sources, stride 8) · 3.33 Hz (10 fps sources, stride 3) |
| `D` | trunk hidden size | 512 |
| `d_f` | CLIP ViT-B/16 image embedding | 512 |
| `d_v` | VideoMAE embedding (Option B) | 768 (ViT-B) · 384 (ViT-S) |
| `d_e` | video-text embedding (Option A) | 512 (ViCLIP-B/16, InternVideo2-CLIP-L14) · 768 (ViCLIP-L/14) |
| `C` | categories in the definition `Z` | ‹code› (normal + the accident classes or descriptions used) |
| `N_tok` | text context length | 77 |
| `n_ctx` | LaGoVAD soft-prompt tokens | ‹code› |
| `k` | MIL top-k | 4 |
| `K` | SG-KN prototypes | 2 |
| `M` | SG-KN candidate normals per window | ⌈0.4·L⌉ = 8, plus ONM known negatives (≤ L) |
| `η, τ` | `L_neg` temperatures | 0.02, 0.02 |

---

## 1. Overview

```
                                   OPTION B                                               OPTION A
video ─► steps @ r ─┬─ frame @ τ_t ─► CLIP-img ViT-B/16 ─► x_t (512)       video ─► steps @ r ─► causal clip 8f@5fps ─► E_vt-video ─► e_t (d_e)
                    └─ clip 16f@10fps ─► VideoMAE-v2 ──► u_t (d_v)
definition Z ─► CLIP-text (+soft prompts) ─► z (C×512)                     definition Z ─► E_vt-text (+soft prompts) ─► z (C×d_e) ─► [W_z if d_e≠512]
────────────────────────────────────────────────────────── shared from here ──────────────────────────────────────────────────────────
per-step features ─► CRN (subtract clip mean, ÷σ_w) ─► input projection(s) + LN ─► H ∈ B×L×512
H ─► Temporal encoder (2-layer Transformer, RoPE) ─► V^t ∈ B×L×512
clip means ─► W_ctx + LN ─► [CTX] ∈ B×1×512
Co-attention U( [CTX ; V^t] , z ) ─► V^u ∈ B×L×512 , Z^u ∈ B×C×512
H_bin( V^t , V^u ) ─► y^bin ∈ B×L          H_mul( V^u , Z^u ) ─► y^mul ∈ B×L×C
training only: ONM-MIL (y^bin) · MIL-align (y^mul) · L_neg with same-source rows (V^t, text) · SG-KN (V^t, y^bin)
```

---

## 2. Input sampling

| Item | Option A | Option B |
|---|---|---|
| Step rate `r` | 3.75 Hz (DADA: 30 fps, stride 8) · 3.33 Hz (DoTA: 10 fps, stride 3) | same |
| Per-step visual input | **one causal clip**: 8 frames @ 5 fps = 1.5 s ending at the step. DADA: every 6th native frame; DoTA: every 2nd. | **one frame** at the step (→ CLIP), **plus one causal clip**: 16 frames @ 10 fps = 1.5 s. DADA: every 3rd native frame; DoTA: native frames. |
| Spatial preprocessing | full frame resized to 224 × 224, no centre crop; each encoder's own mean/std | same |
| Training item | T2 window: `L` = 20 consecutive steps of one source video | same |
| Test item | whole clip of `T` steps, read as sliding windows of 20 (§11) | same |
| Per-window tensor | clips: `B × L × 3 × 8 × 224 × 224` | frames: `B × L × 3 × 224 × 224`; clips: `B × L × 3 × 16 × 224 × 224` |

All encoders are **frozen and run offline**. Their per-step outputs are cached, so the trainable network starts from the cached features in §3.

---

## 3. Frozen encoders (offline, cached)

### 3.1 Option B

**CLIP ViT-B/16, image tower** (public; unchanged from v1)

| Stage | Input | Output |
|---|---|---|
| Patch embed (Conv2d 16×16, stride 16) | 3 × 224 × 224 | 768 × 14 × 14 → 196 × 768 |
| + [CLS] + positional embedding | 196 × 768 | 197 × 768 |
| 12 Transformer blocks (width 768, 12 heads, MLP 3072) | 197 × 768 | 197 × 768 |
| LN([CLS]) → projection 768 → 512 | 768 | **x_t ∈ ℝ^512** |

Cached per window: **`X_f ∈ B × L × 512`**.

**VideoMAE V2 ViT-B** (distilled from ViT-g, K710; public)

| Stage | Input | Output |
|---|---|---|
| Tubelet embed (Conv3d 2×16×16, stride 2×16×16) | 3 × 16 × 224 × 224 | 768 × 8 × 14 × 14 → 1,568 × 768 |
| + fixed sin-cos positional embedding | 1,568 × 768 | 1,568 × 768 |
| 12 Transformer blocks (width 768, 12 heads, MLP 3072; joint space-time attention) | 1,568 × 768 | 1,568 × 768 |
| Mean over tokens → fc_norm (LN); the classifier head is removed | 1,568 × 768 | **u_t ∈ ℝ^768** |

ViT-S variant: width 384, 6 heads, MLP 1536, giving `u_t ∈ ℝ^384`.

Cached per window: **`U ∈ B × L × d_v`**.

**CLIP ViT-B/16, text tower `G`** (public; frozen), with LaGoVAD soft prompts (trainable)

| Stage | Input | Output |
|---|---|---|
| Tokenize `C` definition strings | C strings | C × 77 token ids |
| Token embedding; `n_ctx` learnable soft-prompt vectors inserted | C × 77 | C × 77 × 512 |
| 12 Transformer blocks (width 512, 8 heads), causal mask | C × 77 × 512 | C × 77 × 512 |
| LN(EOT token) → projection 512 → 512 | C × 512 | **z ∈ ℝ^{C×512}** |

Per-window text for `L_neg`: each abnormal window's own class name or description gives **`Z̃ ∈ B2 × 512`** (B2 = number of abnormal windows = 32).

### 3.2 Option A

**ViCLIP-B/16, video tower** (public)

| Stage | Input | Output |
|---|---|---|
| Patch embed (Conv3d 1×16×16, i.e. per-frame patches) | 3 × 8 × 224 × 224 | 768 × 8 × 14 × 14 → 1,568 × 768 |
| + [CLS] + spatial and temporal positional embedding | 1,568 × 768 | 1,569 × 768 |
| 12 blocks, **joint spatio-temporal attention** (width 768, 12 heads) | 1,569 × 768 | 1,569 × 768 |
| LN([CLS]) → projection 768 → 512 | 768 | **e_t ∈ ℝ^512** |

**ViCLIP-L/14** (public): patch 14, giving 8 × 16 × 16 = 2,048 (+1) tokens; width 1024, 24 blocks, 16 heads; projection 1024 → 768. So `e_t ∈ ℝ^768`.

**InternVideo2-CLIP-L14** (distilled; public): patch 14, 8 frames, 24 blocks, width 1024; attention-pooled and projected to **`e_t ∈ ℝ^512`**.

Cached per window: **`E ∈ B × L × d_e`**.

**Video-text encoder, text tower** (frozen), with LaGoVAD soft prompts (trainable)

| Encoder | Text tower | Output |
|---|---|---|
| ViCLIP-B/16 | CLIP-B text transformer (12 layers, width 512, 77 tokens) | z ∈ ℝ^{C×512} |
| ViCLIP-L/14 | CLIP-L text transformer (12 layers, width 768, 77 tokens) | z ∈ ℝ^{C×768} → **W_z (768 → 512, trainable)** → C × 512 |
| InternVideo2-CLIP-L14 | MobileCLIP-B text (12 blocks, 77 tokens) | z ∈ ℝ^{C×512} |

The soft prompts are inserted exactly as in the CLIP tower (§3.1). `Z̃ ∈ B2 × 512` is built the same way, after `W_z` when `d_e = 768`.

---

## 4. Clip-Referenced Normalization (CRN) — parameter-free

| Step | Input | Output |
|---|---|---|
| Reference mean. Training: over **all steps of the window's source video**. Test: over the whole clip. Streaming: causal EMA, `μ_t = (1−α)μ_{t−1} + α·x_t`, `α = 1/(r·10 s)`. | source video: S_v × d | **μ ∈ ℝ^d**, one per stream per video → batch `B × 1 × d` |
| Centre and scale | `B × L × d` and `B × 1 × d` | **`(X − μ)/σ_w ∈ B × L × d`** |

- `σ_w` is **one scalar per stream**: the global within-clip standard deviation, computed once on T2-train.
- Option B applies CRN to both streams: `X̃_f ∈ B×L×512` and `Ũ ∈ B×L×d_v`.
- Option A applies it to one stream: `Ẽ ∈ B×L×d_e`.
- The means are kept for the context token (§7): `μ^f ∈ B×512` and `μ^u ∈ B×d_v`, or `μ^e ∈ B×d_e`.

---

## 5. Input projection and fusion — trainable

| Option | Operation | Input | Output |
|---|---|---|---|
| B | `H = LN(X̃_f W_f) + LN(Ũ W_u)`; `W_f ∈ ℝ^{512×512}`, `W_u ∈ ℝ^{d_v×512}` | `B×L×512`, `B×L×d_v` | **`H ∈ B × L × 512`** |
| A | `H = LN(Ẽ W_e)`; `W_e ∈ ℝ^{d_e×512}` | `B×L×d_e` | **`H ∈ B × L × 512`** |

---

## 6. Temporal encoder — trainable (LaGoVAD, unchanged)

| Stage | Input | Output |
|---|---|---|
| 2 Transformer encoder layers, **RoPE** on the time axis, width 512; heads and FFN width ‹code› | `B × L × 512` | **`V^t ∈ B × L × 512`** |

`V^t` feeds the language-agnostic detection path, the co-attention, `L_neg` and SG-KN.

---

## 7. Context token [CTX] — trainable (new)

| Option | Operation | Input | Output |
|---|---|---|---|
| B | `c = LN( [μ^f ; μ^u] W_ctx )`; `W_ctx ∈ ℝ^{(512+d_v)×512}` | `B × (512 + d_v)` | **`c ∈ B × 1 × 512`** |
| A | `c = LN( μ^e W_ctx )`; `W_ctx ∈ ℝ^{d_e×512}` | `B × d_e` | **`c ∈ B × 1 × 512`** |

The token is used **only** inside the co-attention (§8). It never reaches the language-agnostic path.

---

## 8. Co-attention fusion `U` — trainable (LaGoVAD, unchanged except the prepended token)

| Stage | Input | Output |
|---|---|---|
| Build the visual sequence `[c ; V^t]` | `B×1×512`, `B×L×512` | `B × (L+1) × 512` |
| Broadcast the definition embedding | `C × 512` | `B × C × 512` |
| 2 co-attention layers: vision attends to text, text attends to vision, then FFN on both (heads / FFN ‹code›) | `B×(L+1)×512`, `B×C×512` | `B×(L+1)×512`, `B×C×512` |
| Drop the [CTX] position | `B × (L+1) × 512` | **`V^u ∈ B × L × 512`**, **`Z^u ∈ B × C × 512`** |

---

## 9. Heads — trainable (LaGoVAD, unchanged)

**`H_bin`, the detection head (two paths plus a learnable blend)**

| Stage | Input | Output |
|---|---|---|
| Language-agnostic path: Conv1d stack over time on `V^t` (kernel **3**, replicate padding; layers ‹code›) | `B × 512 × L` | `a ∈ B × L` |
| Language-guided path: same structure on `V^u` | `B × 512 × L` | `g ∈ B × L` |
| Learnable scalar blend of the two paths | `a, g` | **`y^bin ∈ B × L`** (logits); scores `s = σ(y^bin)` |

**`H_mul`, the classification head**

| Stage | Input | Output |
|---|---|---|
| Linear projections `W_mv`, `W_mz` (512 → 512) | `V^u`, `Z^u` | `B×L×512`, `B×C×512` |
| Cosine similarity ÷ temperature | — | **`y^mul ∈ B × L × C`** |
| Video level: **min** over time for the normal class, **max** over time for accident classes, then softmax | `B × L × C` | `B × C` |

---

## 10. Training-only modules (discarded at inference)

### 10.1 Overlap-negative MIL (`L_MIL^ONM`) — parameter-free

| Item | Shape |
|---|---|
| Known-negative mask `m`: 1 where a step is covered by an all-normal window of the same source video | `B × L` (precomputed) |
| Abnormal windows: top-`k` of `y^bin` over `{m = 0}` → mean → BCE with target 1 | `B_a` |
| Normal windows: top-`k` of `y^bin` → mean → BCE with target 0 | `B_n` |
| Dense known-negative term: BCE with target 0 on every step with `m = 1` | `Σ m` steps |

### 10.2 MIL-align (`L_MIL-align`) — LaGoVAD, unchanged

Uses the video-level `B × C` output of `H_mul`, with a cross-entropy against the window's category.

### 10.3 Contrastive loss with hard negatives (`L_neg`), plus same-source rows

| Stage | Input | Output |
|---|---|---|
| Foreground and background weights: `softmax(±y^bin / η)` over time | `B × L` | `B × L` (two maps) |
| Foregrounds: `ṽ_pos = Σ_t w⁺_t V^t_t` | `V^t`, `w⁺` | `B1 × 512` (B1 = 64) |
| Own-window backgrounds, abnormal windows only: `ṽ_neg = Σ_t w⁻_t V^t_t` | `V^t`, `w⁻` | `B2 × 512` (B2 = 32) |
| **Same-source rows (new):** mean over time of `V^t` for each abnormal window's paired normal window | `V^t` | `B2' × 512` (B2' ≤ 32) |
| Stack | — | `Ṽ ∈ (B1 + B2 + B2') × 512`, up to 128 × 512 |
| Text: one embedding per abnormal window (pre-fusion) | — | `Z̃ ∈ B2 × 512` |
| Similarity `S = norm(Ṽ) · norm(Z̃)ᵀ` | — | `(B1 + B2 + B2') × B2`, up to 128 × 32 |
| Loss: `L_{t→v}` (softmax over all rows, including both kinds of hard negative) + `L_{v→t}` (foreground rows only), `τ` = 0.02 | `S` | scalar |

### 10.4 SG-KN, Self-Guided Kinematic Normality — trainable (new)

| Stage | Input | Output |
|---|---|---|
| Candidate normals `Ω_n`: known negatives ∪ bottom-8 steps of `s`; gathered and padded, with a validity mask | `V^t ∈ B×L×512`, `s`, `m` | `F_n ∈ B × M_max × 512` (M_max ≤ 20) |
| Prototype queries `Q` (learnable) | — | `2 × 512` → broadcast to `B × 2 × 512` |
| Prototype extraction: 1 cross-attention layer (Q = `Q`, K = V = `F_n`, masked; 8 heads) | `B×2×512`, `B×M_max×512` | **`P ∈ B × 2 × 512`** |
| `L_compact`: cosine distance of each candidate to its nearest prototype, averaged over valid candidates | `F_n`, `P` | `B × M_max × 2` → scalar |
| Decoder queries: `MLP(V^t)` (512 → 512 → 512, GELU) | `B × L × 512` | `B × L × 512` |
| Decoder layer 1: cross-attention to `P` **with no residual**, then FFN (512 → 1024 → 512) | queries, `P` | `B × L × 512` |
| Decoder layer 2: cross-attention to `P` with residual, then FFN | `B × L × 512`, `P` | **`R ∈ B × L × 512`** |
| Reconstruction error `1 − cos(R_t, V^t_t)`, min-max over the window | `R`, `V^t` | **`S_rec ∈ B × L`** |
| `L_consist = mean_t (s_t − S_rec,t)²` | `s`, `S_rec` | scalar |

`L_SGKN = L_compact + L_consist`, weighted by `λ_n`, the value giving `ρ = ‖g_SGKN‖ / ‖g_task‖ ≈ 0.2` at step 200 (ceiling 0.3).

### 10.5 Total training loss

```
L_total = L_MIL^ONM + L_MIL-align + L_neg^src (+ L_dvs, iff the KIP-off baseline uses it) + λ_n · L_SGKN
```

If `L_dvs` is on, its spliced sequences have a variable length `L' ≤ δ_m · L`, padded with a mask. CRN is applied to each source segment **before** splicing.

---

## 11. Inference (per clip of `T` steps)

| Step | Input | Output |
|---|---|---|
| Encode every step (§3) | T steps | B: `T × 512` and `T × d_v`; A: `T × d_e`. Text `z`: `C × 512`, computed once per definition. |
| CRN with the **whole-clip** mean (offline) or the EMA (streaming) | `T × d` | `T × d` (centred); context means `d` |
| Cut sliding windows: `W` = 20, hop 4, last window aligned to the clip end; `n_w = max(1, ⌈(T − 20)/4⌉ + 1)` | `T × d` | `n_w × 20 × d` (DoTA median: T ≈ 35 → n_w = 5) |
| §5 → §9 on the window batch | `n_w × 20 × d` | `y^bin ∈ n_w × 20`, `y^mul ∈ n_w × 20 × C` |
| Overlap-average back to the clip timeline | `n_w × 20` | **`y^bin ∈ T`**, **`y^mul ∈ T × C`** |
| Interpolate to native frames | `T` | `F_native` scores |
| Benchmark: per-clip min-max. Deployment: a threshold. | `F_native` | frame-level anomaly curve |
| Optional ATS → MLLM (off-path) | `y^bin`, frames | incident report |

Clips with `T ≤ 20` run as a single window of length `T`. In streaming mode, only the newest window is run (the last 20 steps), with the EMA reference.

---

## 12. Master shape table (training, one batch)

| # | Component | Option | Input | Output | Params |
|---|---|---|---|---|---|
| 1 | CLIP ViT-B/16 image | B | `B×L×3×224×224` | `X_f: B×L×512` | frozen |
| 2 | VideoMAE V2 (B / S) | B | `B×L×3×16×224×224` | `U: B×L×d_v` | frozen |
| 2′ | Video-text encoder (video tower) | A | `B×L×3×8×224×224` | `E: B×L×d_e` | frozen |
| 3 | Text tower + soft prompts | A/B | `C×77` | `z: C×512` (A with `d_e` = 768: via `W_z`) | tower frozen; prompts trained |
| 4 | CRN | A/B | `B×L×d` (+ source means) | `B×L×d`; means `B×d` | none |
| 5 | Input projection(s) + LN | A/B | `B×L×d` | `H: B×L×512` | trained |
| 6 | Temporal encoder (2 layers, RoPE) | A/B | `B×L×512` | `V^t: B×L×512` | trained |
| 7 | [CTX] token | A/B | `B×d_means` | `c: B×1×512` | trained |
| 8 | Co-attention `U` (2 layers) | A/B | `B×(L+1)×512`, `B×C×512` | `V^u: B×L×512`, `Z^u: B×C×512` | trained |
| 9 | `H_bin` (2 paths + blend) | A/B | `V^t`, `V^u` | `y^bin: B×L` | trained |
| 10 | `H_mul` | A/B | `V^u`, `Z^u` | `y^mul: B×L×C` → `B×C` | trained |
| 11 | ONM-MIL | A/B | `y^bin`, `m` | scalar | none |
| 12 | `L_neg` + same-source rows | A/B | `V^t`, `y^bin`, `Z̃` | `S: ≤128×32` → scalar | none |
| 13 | SG-KN | A/B | `V^t`, `s`, `m` | `P: B×2×512`, `S_rec: B×L` → scalar | trained (training-only) |

---

## 13. Trainable parameters added by v2 (new modules only; approximate)

| Module | Option B | Option A (`d_e` = 512) | Kept at inference? |
|---|---:|---:|:-:|
| Input projections `W_f`, `W_u` / `W_e` (+ LN) | ≈ 0.26 M + 0.39 M (`d_v` = 768) | ≈ 0.26 M | yes |
| `W_ctx` (+ LN) | ≈ 0.66 M | ≈ 0.26 M | yes |
| `W_z` (only ViCLIP-L/14, `d_e` = 768) | — | ≈ 0.39 M | yes |
| SG-KN (queries, 1 extraction layer, MLP, 2 decoder layers with FFN 1024) | ≈ 5.8 M | ≈ 5.8 M | **no** |

- The LaGoVAD modules (temporal encoder, co-attention, heads, soft prompts) keep their ‹code› sizes.
- KIP's modules are removed: the PMG head, gate, shift and motion head.
- **Deployed network:** the frozen encoder(s), plus LaGoVAD's trunk, plus the new parameters: ≈ 1.3 M (Option B), ≈ 0.5 M (Option A with `d_e` = 512), or ≈ 0.9 M (ViCLIP-L/14).

---

## 14. Frozen vs trained, at a glance

| Frozen (never updated) | Trained | Training-only (dropped at inference) |
|---|---|---|
| Option B: CLIP image and text towers, VideoMAE. Option A: the video-text encoder's video and text towers. | Soft prompts; input projections; [CTX] projection; temporal encoder; co-attention `U`; `H_bin`; `H_mul`; `W_z` (if any) | SG-KN; the ONM mask; same-source rows in `L_neg`; all losses |
