# KAT-VAD v2.2 — Architecture (rev. 2026-10-13b, review round 6, with tensor shapes)

**Scope:** the deployed network; the score-track arm **A3-B**; **EMS-v2.1** (ego-compensated, epipolar motion stream) and its frame-difference control; the kill-switches K1 and K2; the redundancy regressor, the probes, the trained arms and the use tests; the ΔT estimand with its demotion rules; and, for the record, κ (TDP, rejected), EMS-v1 and EMS-v2 (superseded). Reasons, gates and rules are in `KAT_VAD_PROPOSAL_v2.2.md`.

**The deployed network is A3-ign, unchanged** (§1). It changes only if SC-2 adopts **A3-B** or M2 adopts **M**, each on span only (proposal §8).

**Round 7 (2026-10-13c):** proposal §R7 D1–D13 override this file where they conflict (fps assumed A1/A1′, pairs in frames; `r = 3` pairs on raw CAP; width-matched δ (224); motion base fixed = A3-ign; K1 stats shifted causal; ΔT on the step grid).

**Conventions:** shapes `batch × time × channels`; images `channels × height × width`; **‹code›** = unchanged LaGoVAD / A3-ign settings; "per sequence" = per source video in training, per clip at test. **Displacements are in frame-width units (both axes) per second.**

---

## 0. Symbols

| Symbol | Meaning | Value |
|---|---|---|
| `B` | sequences per training batch | 64 (32 abnormal + 32 normal) |
| `L` | steps per training window | 20 |
| `T` | steps in a test clip | variable (DoTA median ≈ 33 at stride 3) |
| `S` | steps in a source video | variable |
| `fps` | native frame rate, **measured** per sequence | DoTA-CAP: 10 (817 clips) or 30 (310); DADA: measured at K2 (literature: 30, unverified) |
| `Δt_step` | step duration, from measured fps | 0.27 s at 30 fps stride 8 (DADA, if 30 fps) · 0.30 s (DoTA) |
| `k`, `Δt_f` | flow pair offset and interval | `k = round(0.1 s · fps)`, `Δt_f = k / fps` |
| `W_n`, `H_n` | native frame size | DADA 1584 × 660 · DoTA 1280 × 720 |
| working size | aspect-preserving, width 448, padded to a multiple of 8 | DADA 448 × 187 → 448 × 192 · DoTA 448 × 252 → 448 × 256 |
| `G` | flow grid, equal **frame-fraction** cells | 56 × 28 points |
| `C_g` | descriptor grid | 4 × 4 cells (14 × 7 grid points each) |
| `c_k` | ego reference points | quadrant centres (0.25 \| 0.75 of width, 0.25 \| 0.75 of height) |
| `n^P_min` | off-plane support for F usability | 8 F-inliers that are H-outliers |
| `d_ψ` | EMS descriptor | **224** (ego 24, local 200) |
| `d_δ` | frame-difference control | 36 |
| `D` | trunk width | 512 |
| `k_MIL` | MIL top-k | 4 |

---

## 1. Deployed network: A3-ign (unchanged)

```
frame ─► CLIP ViT-B/16 (frozen) ─► x_t ∈ ℝ^512 ─► CRN (R2) ─► x̃_t ───────────────────────────────┐
causal 16 f, 1.5 s ─► VideoMAE V2-S (frozen) ─► u_t ∈ ℝ^384 ─► CRN (R2) ─► c·(⊘σ_u) ─► W_u (0-init) ─(+)─► h_t ∈ ℝ^512
h ─► temporal encoder (2 layers, RoPE, V^t = H + Enc(H)) ─► V^t ─► CoAttn(V^t, z) ─► V^u, Z^u
H_bin(V^t, V^u) ─► y^bin ∈ B×L        H_mul(V^u, Z^u) ─► y^mul ∈ B×L×C
```

| # | Component | Input | Output | Params |
|---|---|---|---|---|
| 1 | CLIP ViT-B/16 image | `B×L×3×224×224` | `X: B×L×512` | frozen |
| 2 | VideoMAE V2-S | `B×L×3×16×224×224` | `U: B×L×384` | frozen |
| 3 | CLIP text + soft prompts | `C×77` | `z: C×512` | tower frozen; prompts trained |
| 4 | CRN R2 | `X`, `U` | `X̃`, `Ũ` | none |
| 5 | Video fusion `x̃ + W_u(c·ũ ⊘ σ_u)`, `W_u` zero-init | `X̃`, `Ũ` | `B×L×512` | 197,120 |
| 6 | Temporal encoder | `H` | `V^t` | trained |
| 7 | Co-attention | `V^t`, `z` | `V^u`, `Z^u` | trained |
| 8 | `H_bin` | | `y^bin: B×L` | trained |
| 9 | `H_mul` | `V^u`, `Z^u` | `y^mul: B×L×C` | trained |

**Losses:** `L_MIL + L_MIL-align + L_dvs` (anchor `ignore`); `L_neg` off. **Inference:** DoTA stride 3, whole clip, interpolated to native frames.

### 1.1 Score-track arm A3-B (SC-2)

Identical to A3-ign except row 2 and row 5:

| # | Component | Input | Output | Params |
|---|---|---|---|---|
| 2′ | **VideoMAE V2-B** (frozen) | `B×L×3×16×224×224` | `U: B×L×768` | frozen |
| 4′ | CRN R2 on `U` (own references) | | `Ũ: B×L×768` | none |
| 5′ | Video fusion, `W_u ∈ ℝ^{512×768}` zero-init, own `σ_u`, `c` | | `B×L×512` | **393,728** (*derived*: 768·512 + 512) |

Requires V2-B extraction for full T2 (exists for DoTA-CAP and the 295 K sources). 5 seeds, current recipe. Adopted on span only.

---

## 2. EMS-v2.1 — offline descriptor (per sequence, before windowing and DVS splicing)

### 2.1 Per-sequence set-up

| # | Operation | Output |
|---|---|---|
| 1 | Measure fps: container metadata **and** frame count ÷ duration; flag the sequence if they differ by > 2 % | `fps`, flag |
| 2 | `k = round(0.1 s · fps)`; `Δt_f = k / fps`; step times `τ_t` from `fps` | scalars |
| 3 | Resize: width 448, aspect preserved; pad bottom to a multiple of 8; padding added to the mask | working frames |
| 4 | Coordinates of grid point `p`: `(x / W_n, y / W_n)` — **isotropic width units** | grid `G` |

### 2.2 Flow, mask, two-view geometry (per step)

| # | Operation | Input | Output |
|---|---|---|---|
| 1 | Frame pair `(I_{t−k}, I_t)` at working size | 2 native frames | `2 × 3 × h × 448` |
| 2 | **RAFT-small**, FlyingChairs + FlyingThings weights, frozen; ‹iterations: library default› | | `2 × h × 448` (working px) |
| 3 | Area-pool to `G` over equal frame-fraction cells (padding excluded); convert to width units; ÷ `Δt_f` | | `f_t ∈ 2×28×56` (widths / s) |
| 4 | **Static mask** (v1 rule) ∪ padding; empty below the ego-stationary floor ‹floor fixed at M0› | `f_{1..S}` | `M ∈ {0,1}^{28×56}` |
| 5 | **Homography** `H_t`: `cv2.findHomography`, RANSAC, threshold 0.005 (width units of displacement, i.e. 0.5 % of width), 500 iterations, fixed seed, on unmasked `(p, p + f_t Δt_f)`. Failure (< 8 points / degenerate): translation = median unmasked flow; **H-fit failure flag** | `f_t`, `M` | `H_t`, inlier set `I^H_t`, ratio `π^H_t` |
| 6 | **Fundamental matrix** `F_t`: `cv2.findFundamentalMat`, `FM_RANSAC`, threshold 0.005, confidence 0.999, fixed seed, same correspondences | | `F_t`, inlier set `I^F_t` |
| 7 | **Off-plane support** `n^P_t = |I^F_t \ I^H_t|`; **F usable** `v_t = 1` iff `|I^F_t| ≥ 15` and `n^P_t ≥ 8` | | `v_t ∈ {0,1}`, `n^P_t` |
| 8 | **Epipolar independence** `e^F_t(p)` = Sampson distance of `(p, p + f_tΔt_f)` w.r.t. `F_t`, ÷ `Δt_f`, if `v_t`; else **hold** `e^F_{t′}(p)` from the last usable step `t′`; leading unusable steps back-filled from the first usable one; no usable step in the sequence → 0 and flagged | | `e^F_t ∈ 28×56` |
| 9 | Ego flow `f^H_t(p) = (H_t(p) − p) / Δt_f`; residual `r_t = f_t − f^H_t` (masked points excluded) | | `r_t ∈ 2×28×56` |
| 10 | `|r_t|`; **looming** `ℓ_t = ∂r_x/∂x + ∂r_y/∂y` (central differences; spacing `Δx = 1/56`, `Δy = (H_n/W_n)/28` width units); **inward lateral** `λ_t = −sign(p_x − 0.5)·r_{t,x}`; **vertical** `ν_t = r_{t,y}` | `r_t` | four `28×56` maps |
| 11 | **Ego 4-point**: `ego_t = [(H_t(c_k) − c_k) / Δt_f]_{k=1..4}` | `H_t` | `ℝ^8` |

### 2.3 Per-step descriptor `φ_t ∈ ℝ^110`

| Block | Channels | Dims |
|---|---|---|
| **Ego** | `ego_t` (8); mean `|f^H_t|` (1); `π^H_t` (1); `v_t` (1); `n^P_t / #unmasked` (1) | 12 |
| **Cells** (16 × 6) | mean `e^F`; p90 `e^F`; p90 `|r|`; mean `ℓ`; mean `λ`; mean `ν` | 96 |
| **Global** | `ρ^F_t` = max over cells of p90 `e^F`; `ρ^H_t` = max over cells of p90 `|r|` | 2 |

### 2.4 Temporal features and normalisation

| # | Operation | Output |
|---|---|---|
| 1 | IAM acceleration `a_t = |φ_t − φ_{t−1}| / Δt_step` for the 109 channels other than `v_t`; for the 32 `e^F` cell channels and `ρ^F`, `a_t := 0` unless `v_t v_{t−1} = 1`; `ā_t = mean(a_{t−2..t})` | `S×109` |
| 2 | Ego jerk `j_t = ‖ego_t − ego_{t−1}‖ / Δt_step` | `S×1` (row 3 only) |
| 3 | Summaries: 1.5 s causal max of `ρ^F`, `ρ^H`, `j`; 1 s traces `λ_tr = 2^{−Δt_step/1 s}` of `ρ^F`, `ρ^H` | `S×5` |
| 4 | `ψ_t = [φ_t ; ā_t ; summaries]` | **`Ψ ∈ S×224`** |
| 5 | Replicate padding at the ψ level (`t ≤ 1` for `ā`; `t = 0` for `j`) | complete rows |
| 6 | `ψ̃_t = (ψ_t − median_τ ψ_τ) ⊘ σ_ψ`; `σ_ψ` per channel on T2-train, fixed. **Corpus-gate fallback** (only if the M0 gate fails): `⊘ MAD_τ(ψ)` per clip | **`Ψ̃ ∈ S×224`** |

**Channel subsets.** **ego** = 12 `φ` + 11 `ā` + max `j` = **24**; **local** = 98 `φ` + 98 `ā` + 4 summaries (`ρ^F`, `ρ^H` maxes and traces) = **200**. **Direction-type** (reported): `ego_t`, cell `λ`, cell `ν` and their `ā` = **80**; all other channels are magnitude-type or geometry quality.

Windows and DVS segments slice precomputed rows; nothing reads across a splice junction or window start.

---

## 3. Frame-difference control δ

| # | Operation | Output |
|---|---|---|
| 1 | Grayscale `I_{t−k}, I_t` at the **same working size and padding mask** as §2; `D_t = |I_t − I_{t−k}|` | `h × 448` |
| 2 | Mean per 4×4 **frame-fraction** cell (16) + frame mean (1), padding excluded | `δ_t ∈ ℝ^17` |
| 3 | Acceleration `ā^δ_t` (as §2.4 row 1) | `ℝ^17` |
| 4 | 1.5 s causal max and 1 s trace of the max cell | `ℝ^2` |
| 5 | Concatenate, pad, reference, own `σ_δ` | **`Δ̃ ∈ S×36`** |

---

## 4. Kill-switches and gates on cached features

### 4.1 K1 — global-flow probe (free; T2-val only)

Unchanged S2 probe (§4.4) on `F0` (897) vs `F0 + stats23` (the cached flow/v1 `.stats.npy`, 23 per-step RAFT statistics incl. 16 direction bins, aligned with T2 steps; T2-train-standardised). `d_in` 920, **294,785** parameters. Trained on T2-train − T2-val, read on **T2-val** span and ΔT, 5 seeds. **Soft stop** if both Δ points < +0.005.

### 4.2 K2 — quality-only pilot (no labels, no probe)

Sample: 100 T2-**train** sources + 100 CAP-dev clips (50 native 10 fps, 50 native 30 fps).

| Metric | Gate |
|---|---|
| fps agreement (metadata vs count/duration) | flagged sequences printed; DADA fps recorded |
| H-fit failure share | ≤ 20 % of steps, per corpus and per fps group |
| **F-usable share** `mean(v_t)` | **≥ 50 %** of steps, per corpus and per fps group |
| **Speed coupling** on all pilot steps (labels not read): per clip, Spearman ρ between each cell's p90 `e^F` (usable steps) and mean `|f^H|`; same for p90 `|r|`; median over cells and clips | **ρ(`e^F`) ≤ ρ(`|r|`) − 0.10** |
| Clip-start SMD (steps 0–2 vs later steps) on `ψ̃` | reported (κ reference: 61/131 > 0.1) |
| Static-mask coverage; RAFT-small FLOPs/step | reported |

Any gate failure → descriptor redesign before full extraction.

### 4.3 Redundancy regressor (M0-R; R0-c on κ)

The S2 probe body with a vector output: Conv1d `897 → 64`, k 5, valid, ReLU → Conv1d `64 → d`, k 5, valid; input `F0 = [x̃ ; ũ ; p]` in true order, windows of 20 + 4 context steps each side (`B×28×897 → B×20×d`); targets T2-train-standardised; MSE; AdamW lr 1e-3, wd 1e-2, batch 64, 20 epochs; 5 seeds; R² per channel on T2-val, averaged per block.

| Target | `d` | Params (*derived*: 897·64·5 + 64 + 64·d·5 + d) | Role |
|---|---|---|---|
| κ | 131 | 329,155 | positive control (expected R² ≥ 0.5) |
| ψ̃ (all; blocks read from one run) | 224 | 359,008 | — |
| ↳ **local block** | 200 | (in the 224 run) | **gate**: mean R² ≤ 0.5 |
| ↳ ego block | 24 | (in the 224 run) | reported; **expected > 0.5** |
| ↳ direction-type / magnitude-type | 80 / 144 | (in the 224 run) | reported |
| δ̃ | 36 | 298,660 | reported |

### 4.4 Probes (M1; R0-e; P1 optional)

Unchanged S2 probe: Conv1d `d_in → 64`, k 5 → ReLU → Conv1d `64 → 1`, k 5; RF 9; per-step BCE; same optimiser; 5 seeds, seed-averaged. Read on ΔT and on span; CAP-dev results also per native-fps group.

| Probe | Input | `d_in` | Params (`d_in·320 + 385`) |
|---|---|---|---|
| reference | `F0` | 897 | 287,425 |
| **M1 STRUCTURE (primary)** | `F0 + ψ̃` vs `F0 + δ̃` | 1,121 vs 933 | 359,105 vs 298,945 |
| **M1 VALUE (prerequisite)** | `F0 + ψ̃` vs `F0` | 1,121 vs 897 | — |
| split (reported) | `F0 + ψ̃_ego` · `F0 + ψ̃_local` | 921 · 1,097 | 295,105 · 351,425 |
| K1 | `F0 + stats23` | 920 | 294,785 |
| R0-e | `p` only | 1 | 705 |
| P1(a)/(b) (optional) | `F0c` / causal `F0` | 897 | 287,425 |

---

## 5. Estimand computation

### 5.1 ΔT (onset transition)

```
for each clip with span [t_s, t_e] (native-time onset/offset; step times τ_t from MEASURED fps):
  P = { t : t_s ≤ τ_t < min(t_s + 2.0 s, t_e) }
  N = { t : label_t = normal  and  t_s − 3.0 s ≤ τ_t < t_s − 0.5 s }
  eligible iff |P| ≥ 2 and |N| ≥ 3
  AUC_clip = AUC(scores[P ∪ N], labels)            (step scores; no interpolation)
ΔT = mean over eligible clips; cluster bootstrap over source videos (B = 10,000)
printed beside: n eligible; p_T2 on the same steps; f_2-ΔT = AUC of z(ȳ) + 2·z(p_T2) on the same steps;
                the same per native-fps group
```

**Demotion rules (evaluated at R0, fixed before it runs):** n eligible on CAP-dev < 200 → ΔT reported only; **ΔT(A3-ign) ≥ 0.90 or ΔT(saved `F0` probe) ≥ 0.90 → ΔT reported only**. If demoted, M1/M2 motion rules read span (labelled).

T2-val: the same rule on windows that contain an onset with ≥ 3 normal steps before it.

### 5.2 Span macro and DF8 swap

Span macro: unchanged (protocol B, per-clip AUC on native frames). DF8 swap: the 491-clip same-length early/late swap of CAP-dev (plan `katvad-v2-df8-position-shift.md`); CRN and ψ̃ medians recomputed per crop; forward passes only. Used by SC-1 (A3-ign, A3, `p_T2` frozen at its T2 fit), SC-2 (A3-B) and U (base, M, Mδ).

---

## 6. Trained arms (M2) and use tests (U)

### 6.1 EMS fusion block (arm M)

| # | Stage | Input | Output | Params |
|---|---|---|---|---|
| 1 | `g_t = GELU(W_1 ψ̃_t + b_1)`, `W_1 ∈ ℝ^{128×224}`, separate seeded `torch.Generator` | `B×L×224` | `B×L×128` | 28,800 |
| 2 | `W_m ∈ ℝ^{512×128}` (+ bias), **zero-init** | `B×L×128` | `B×L×512` | 66,048 |
| 3 | `h_t = x̃_t + W_u(c·ũ_t ⊘ σ_u) + W_m g_t` | | `H: B×L×512` | — |

**Total 94,848.** No LayerNorm; no temporal operation; `ρ_m = ‖W_m g‖ / ‖x̃‖` logged every 50 steps. δ block (Mδ): `W_1 ∈ ℝ^{128×36}`, **70,784**.

### 6.2 Arms

The **base** is A3-ign, or A3-B if SC-2 adopted it before M2 starts (frozen at that moment).

| Arm | Added at `h` | New params | Role |
|---|---|---|---|
| **base (re-run)** | — | 0 | control |
| **M** | `W_m g(ψ̃)` | 94,848 | candidate (adopted on span only) |
| **Mδ** | `W_m g(δ̃)` | 70,784 | identifying control (never adopted) |

**Unit test before any run:** frozen EMS block with `W_m = 0` reproduces the re-run's first 100 losses at the same seed.

### 6.3 Use tests (inference on M's 5 checkpoints; no training)

```
U0 (primary): for each clip of T steps, for seed s = 1..5:
    o_s ~ Uniform{⌈T/3⌉, …, ⌊2T/3⌋}           (fixed seed per clip)
    Ψ̃_shift[t] = Ψ̃[(t − o_s) mod T]            (all 224 channels)
    score_s = M(X̃, Ũ, Ψ̃_shift)
  score_U0 = mean_s score_s ;  ΔU0 = AUC(M) − AUC(score_U0)   on ΔT and span
```

| Test | Manipulation of `Ψ̃` (`T×224`) before fusion | Status |
|---|---|---|
| **U0** | within-clip circular shift (above) | **primary** use test |
| U0-perm | within-clip random row permutation, 5 seeds | reported |
| U0-zero | `Ψ̃ := 0` | reported; OOD-confounded (constant `W_m GELU(b_1)` enters LayerNorm/attention) |
| **U1** | `Ψ̃_t := Ψ̃_{t−s}`, `s = round(1.0 s / Δt_step)` (3 on DoTA), first `s` rows replicate row 0 | timing use |
| **U2-ego** / **U2-local** | circular shift (U0 offsets) of the 24 ego / 200 local channels only; zeroing reported | which half is used |

### 6.4 Master shape table (training, one batch, arm M)

| # | Component | Input | Output | Params |
|---|---|---|---|---|
| 1 | CLIP ViT-B/16 | `B×L×3×224×224` | `X: B×L×512` | frozen |
| 2 | VideoMAE V2-S (or V2-B if base = A3-B) | `B×L×3×16×224×224` | `U: B×L×384` (768) | frozen |
| 3 | CLIP text + prompts | `C×77` | `z: C×512` | prompts trained |
| 4 | CRN R2 | `X`, `U` | `X̃`, `Ũ` | none |
| 5 | Video fusion | `Ũ` | `B×L×512` | 197,120 (393,728) |
| 6 | **EMS-v2.1 descriptor (offline, §2)** | cached | `Ψ̃: B×L×224` | RAFT-small frozen; fixed statistics |
| 7 | **EMS fusion (§6.1)** | `Ψ̃` | `B×L×512` | **94,848** |
| 8 | Residual sum | 4, 5, 7 | `H: B×L×512` | — |
| 9–12 | Temporal encoder, co-attention, `H_bin`, `H_mul` | | `y^bin`, `y^mul` | trained |
| 13 | Losses (unchanged) | | scalar | — |

---

## 7. Inference (if M is adopted; one clip of `T` steps)

| Step | Input | Output |
|---|---|---|
| CLIP, VideoMAE per step | `T` steps | `X`, `U` |
| CRN R2 | `X`, `U` | `X̃`, `Ũ` |
| **EMS-v2.1:** measure fps → `k`; aspect-preserving resize → RAFT-small per pair → grid (width units / s) → mask → `H_t`, `F_t`, off-plane support, `v_t` → `e^F` (hold-last), `|r|`, `ℓ`, `λ`, `ν`, ego 4-point → `φ`, gated `ā`, summaries → median reference | `T` frame pairs | `Ψ̃ ∈ T×224` |
| Fusion; trunk; heads | `1×T×512` | `y^bin ∈ T`, `y^mul ∈ T×C` |
| Interpolate; per-clip min-max (benchmark) or threshold (deployment) | | anomaly curve |

RGB-only input; RAFT-small once per step; geometry on CPU (two RANSAC fits on ≤ 1,568 points per step). Flow, geometry and hold-last are causal; the static mask and the medians need the whole clip (not streamable, as v2).

---

## 8. Records

**κ (S1–S3, TDP rejected).** Offline, CLIP only; `κ ∈ ℝ^131`. S3(a) −0.0007, S3(b) +0.0080: both fail. Kept as a diagnostic and as the M0-R positive control.

**EMS-v1 (2026-10-12, superseded, never extracted).** Homography residual as independent motion; raw homography parameters. Replaced: parallax under forward motion.

**EMS-v2 (2026-10-13 round 5, superseded, never extracted).** 185-d. One channel `e` = Sampson distance if `π^H < η_H`, else `|r|`; 448 × 224 anisotropic squash; DADA pairs at native `n−3` assuming 30 fps; U0 = `ψ̃ := 0`. Replaced in round 6: channel switching at `η_H` (A1), anisotropic corpus-dependent scaling (A4), unverified fps (A5), U0 not an ablation (A2).

---

## 9. Frozen, fixed, trained

| Frozen | Fixed statistics / settings | Trained (deployed) | Trained (experiments only) |
|---|---|---|---|
| CLIP towers; VideoMAE V2-S (V2-B for A3-B); **RAFT-small** | CRN references, `s`, `σ_u`, `c`; **measured fps, `k`**; static mask; **RANSAC settings for H and F; `n^P_min` = 8; F-inlier minimum 15**; `σ_ψ`, `σ_δ`; mask floor; ΔT windows and demotion thresholds; U0 offsets (seeded) | prompts, `W_u`, encoder, co-attention, `H_bin`, `H_mul`; **`W_1`, `W_m` only if M is adopted** | regressor (M0-R); probes (K1, M1, R0-e, P1); A3-B (until adopted); δ block (Mδ); S4 arm |

**Not added (proposal §10):** IAM prototypes, `w_int`, `L_cst`; ACC; local Transformer; similarity GCN; LLM attribute text; TDP and its variants; `p_T2` as an input.
