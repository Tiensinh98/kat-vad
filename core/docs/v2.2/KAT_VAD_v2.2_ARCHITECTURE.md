# KAT-VAD v2.2 — Architecture (with tensor shapes)

**Scope:** the deployed network, the operators and models the v2.2 experiments use (the block shuffle, the change features, the temporal probe, the trunk-level order arm) and the conditional TDP. Reasons, gates and the plan are in `KAT_VAD_PROPOSAL_v2.2.md`.

**The deployed network is A3-ign, unchanged** (§1). It changes only if step S5 adopts TDP (§7).

**Conventions:**
- Shapes are written `batch × time × channels`.
- Values marked **‹code›** are unchanged LaGoVAD / A3-ign settings: take them from the current implementation.
- "Per sequence" means per source video during training and per clip at test.

---

## 0. Symbols

| Symbol | Meaning | Value |
|---|---|---|
| `B` | sequences per training batch | 64 (32 abnormal + 32 normal) |
| `L` | steps per training window | 20 |
| `T` | steps in a test clip | variable (DoTA median ≈ 33 at stride 3) |
| `S` | steps in a source video | variable |
| `D` | trunk width | 512 |
| `d_v` | VideoMAE V2-S feature size | 384 |
| `b` | shuffle block size | 5 steps (≈ 1.35 s DADA, ≈ 1.5 s DoTA) |
| `g` | minimum original distance between neighbours in the twin ordering | 6 steps |
| `d_q` | PCA width for κ | 64 |
| `d_κ` | κ width | 131 (CLIP only); 262 if S1b admits Δu |
| `p` | position feature: `[τ, τ², τ³]`, `τ = (t + 0.5)/n` of the step's original index (`kill_switch_probe.position_features`) | 3 channels |
| `k` | MIL top-k | 4 |

---

## 1. Deployed network: A3-ign (unchanged)

```
frame ─► CLIP ViT-B/16 (frozen) ─► x_t ∈ ℝ^512 ─► CRN (R2) ─► x̃_t ───────────────────────────┐
causal 16 f, 1.5 s ─► VideoMAE V2-S (frozen) ─► u_t ∈ ℝ^384 ─► CRN (R2) ─► c·(⊘σ_u) ─► W_u (0-init) ─(+)─► h_t ∈ ℝ^512
h ─► temporal encoder (2 layers, RoPE, outer residual V^t = H + Enc(H)) ─► V^t ─► CoAttn(V^t, z) ─► V^u, Z^u
H_bin(V^t, V^u) ─► y^bin ∈ B×L        H_mul(V^u, Z^u) ─► y^mul ∈ B×L×C
```

| # | Component | Input | Output | Params |
|---|---|---|---|---|
| 1 | CLIP ViT-B/16 image | `B×L×3×224×224` | `X: B×L×512` | frozen |
| 2 | VideoMAE V2-S | `B×L×3×16×224×224` | `U: B×L×384` | frozen |
| 3 | CLIP text + soft prompts | `C×77` | `z: C×512` | tower frozen; prompts trained |
| 4 | CRN, R2 (per-sequence median; per DVS segment before splicing) | `X`, `U` | `X̃: B×L×512`, `Ũ: B×L×384` | none |
| 5 | Video fusion: `h = x̃ + W_u(c·ũ ⊘ σ_u)`, `W_u` zero-init | `X̃`, `Ũ` | `H: B×L×512` | ≈ 0.20 M |
| 6 | Temporal encoder (LaGoVAD) | `H` | `V^t: B×L×512` | trained |
| 7 | Co-attention (LaGoVAD) | `V^t`, `z` | `V^u`, `Z^u` | trained |
| 8 | `H_bin`: Conv1d kernel 3 over time on `V^t` and on `V^u`, learnable blend | `V^t`, `V^u` | `y^bin: B×L` | trained |
| 9 | `H_mul` | `V^u`, `Z^u` | `y^mul: B×L×C` | trained |

**Losses:** `L_MIL + L_MIL-align + L_dvs` (anchor mode `ignore`); `L_neg` off. **Inference:** stride 3 on DoTA, whole clip in one pass, scores interpolated to native frames.

**Note on order access (corrected from the first draft).** `H` reaches `V^t` un-normalized through the outer residual, and `H_bin`'s kernel-3 conv over `V^t` can compute signed differences `W(h_t − h_{t−1})` linearly. What the trunk cannot form linearly is a **magnitude** of change or a second difference.

---

## 2. Block-shuffle operator `Π_b` (S2, S4)

Reorders the steps of one sequence; everything attached to a step (features, step labels, DVS pseudo-labels and masks, original position `p`) moves with it.

| Stage | Input | Output |
|---|---|---|
| Partition into blocks: `q = ⌊n / 5⌋` blocks; the first `q − 1` have 5 steps, the last has `5 + (n mod 5)` (5–9). `n = 4`: one block. `n ≤ 3`: identity (counted) | `n` steps | blocks |
| Per block, left to right, sample uniformly a permutation with **no originally adjacent steps in neighbouring slots**, in either direction, and whose first slot is not originally adjacent to the previous block's last slot (rejection sampling, seed fixed per sequence) | block of size `m` | permuted block |
| Concatenate | blocks | permutation `π` of `0…n−1`, displacement ≤ 4 (≤ 8 in a 6–9 tail) |

**Valid permutation counts** (enumerated): `m` = 4: 2 · 5: 14 · 6: 90 · 7: 646 · 8: 5,242 · 9: 47,622. None exist for `m` = 2 or 3. In the 14 valid 5-permutations, an average of 2.29 of 4 neighbouring slots hold steps that were originally 2 apart (the stated residual).

**Where it is applied:**

| Use | Unit | Then |
|---|---|---|
| S2 probe | each source video (training) or clip (test), once, on its own timeline | context-padded probe windows (§4) are cut from the shuffled sequence |
| S4 trunk, training | each **segment** separately: an unspliced window; in a DVS sequence the anchor and each filler, **before splicing**. CRN (§1 row 4) is order-invariant, so it is unaffected | the segment keeps its step set and window label |
| S4 trunk, test | each clip | whole clip through the trunk |
| All | — | scores are mapped back with `π⁻¹` before any metric |

---

## 3. Change features (S1; the offline part of TDP)

Computed **per sequence, offline**, from the existing caches. CLIP only unless S1b admits Δu (then the same pipeline runs on `u` and is concatenated).

| # | Operation | Input | Output |
|---|---|---|---|
| 1 | `Δx_t = x_t − x_{t−1}` for `t ≥ 1` (raw CLIP; CRN cancels up to the scalar `s`) | `x: S×512` | `Δx: S×512` (row 0 missing) |
| 2 | `m_x = median_{t≥1} ‖Δx_t‖` | `Δx` | scalar per sequence |
| 3 | `Δx̄_t = (Δx_t / m_x) ⊘ σ_Δ`; `σ_Δ ∈ ℝ^512`: per-channel std on T2-train, fixed | `Δx`, `m_x` | `S×512` |
| 4 | `s_t = ‖Δx̄_t‖`; `m5_t = max(s_{t−4..t})` (available steps); `e_t = λ e_{t−1} + (1 − λ) s_t`, `λ = 2^{−1/3.5}` (half-life ≈ 3.5 steps ≈ 1 s) | `Δx̄` | `S×3` |
| 5 | `v_t = |Q Δx̄_t|`; `Q ∈ ℝ^{64×512}`: top-64 PCA basis of `Δx̄` on T2-train, fixed | `Δx̄` | `S×64` |
| 6 | `v̄_t = mean(v_{t−2..t})` (available steps): fixed causal jitter filter | `v` | `S×64` |
| 7 | `a_t = |v̄_t − v̄_{t−1}|` for `t ≥ 2`; `ā_t = mean(a_{t−2..t})` (available steps) | `v̄` | `S×64` |
| 8 | **Replicate padding at the κ level:** velocity-type channels at `t = 0` copy `t = 1`; acceleration channels at `t ≤ 1` copy `t = 2` | rows 4–7 | complete rows |
| 9 | `κ_t = [v̄_t ; ā_t ; s_t ; m5_t ; e_t]` | rows 4–8 | `S×131` |
| 10 | **Fixed per-channel standardisation:** `κ̂_t = (κ_t − μ_κ) ⊘ σ_κ`; `μ_κ, σ_κ ∈ ℝ^131` from true-order κ over T2-train steps. Stored in the cache; read by the probe (S3) and the model (S5) alike | `κ` | **`K ∈ S×131`** |

- **S1a** uses `m5` (primary) and `s` (secondary), row 4, directly as per-step scores (standardisation does not change within-clip ranks).
- **S1c** compares steps 0–2 vs steps ≥ 3 of DoTA-CAP-dev normal steps (standardized mean difference) on five summary channels: `s`, `m5`, `e`, channel-mean of `v̄`, channel-mean of `ā`. All 131 are printed, not gated.
- **S1b** (echo check) runs rows 1–4 on `u` and reads `s^u` around span ends (equivalence rule, proposal §5).

---

## 4. Order-free twin `κ^π` (S1, S3b, S5's Tπ)

| # | Stage | Input | Output |
|---|---|---|---|
| 1 | Draw an ordering `σ` of the sequence's steps in which consecutive elements are **≥ 6 steps apart** in the original order (randomised construction with restarts; seed fixed per sequence and arm). Feasible for `S ≥ 12`, infeasible for 10–11 (exhaustive search); for `S < 12` the largest feasible gap is used and recorded. Per source video / clip, so DVS segment length is irrelevant; the share of sequences under 12 steps is reported per set | `S` steps | `σ` |
| 2 | Run §3 rows 1–9 **in the order `σ`** (so `Δ`, the filters, the max and the trace all read far-away steps), including its own `m_x` | `x` reordered by `σ` | `S×131` |
| 3 | Write each row back to its step's original position | — | `K^σ ∈ S×131` |
| 4 | **Quantile-match per sequence and channel** to the true-order `K` (after §3 row 10; the affine standardisation commutes with the matching): for channel `j`, sort `K[:, j]` and assign the sorted values to steps in the rank order of `K^σ[:, j]` | `K`, `K^σ` | **`K^π ∈ S×131`** |

`K^π` holds exactly `K`'s values per sequence and channel, each still tied to its own step's content, with no adjacency in its construction.

---

## 5. Temporal probe (S2, S3)

A fixed-capacity model on frozen features; the same architecture, hyper-parameters and seeds for every input set.

| Item | Value |
|---|---|
| Input sets | `F0 = [x̃ ; ũ ; p]` (899 channels; `p` = 3-channel cubic basis of the step's **original** index, moving with the step) · `F0 + K` (1,030) · `F0 + K^π` (1,030) · secondary, S2 only: `[x̃ ; p]` (515; CLIP only, no VideoMAE bag) |
| Normalisation | per-channel standardisation on T2-train, fixed (`K` and `K^π` arrive already standardised, §3 row 10) |
| Layer 1 | Conv1d `d_in → 64`, kernel 5, **valid**; ReLU |
| Layer 2 | Conv1d `64 → 1`, kernel 5, **valid** → per-step logit |
| Receptive field | 9 steps (±4 ≈ ±1.1–1.2 s) |
| Training windows | the 20 T2 window steps **plus 4 real context steps on each side** from the source video: input `B × 28 × d_in`, output `B × 20` (valid convolutions consume exactly the context; no padding inside the scored region). Missing context at a source's true ends: replicate padding |
| Test | whole clip, replicate-padded by 4 at the true ends, valid convolutions → `T` scores |
| Loss and labels | per-step BCE against the step labels the earlier probes (K, E2(d), D6) used ‹harness› |
| Optimiser (fixed, not tuned) | AdamW, lr 1e-3, weight decay 1e-2, batch 64 windows, 20 epochs |
| Seeds | 5; scores **seed-averaged** before metrics; per-seed Δs reported as a stability check |
| Params | `F0`: 899·64·5 + 64 + 64·5 + 1 = **288,065** · `F0 + K` / `F0 + K^π`: **329,985** · `[x̃ ; p]`: **165,185** (*derived*) |
| Metrics | per-clip macro AUC on DoTA-CAP-dev (protocol B; decides) and T2-val (reported); cluster bootstrap over source videos, B = 10,000; `f_2` Δ beside |

- **S2:** `F0` in true order vs `F0` under `Π_b` (§2; applied per sequence, then windows cut). Gated on `F0`; the same contrast on `[x̃ ; p]` is reported beside it (`ũ_t` carries the past 1.5 s as a bag through the shuffle; `x̃` alone does not).
- **S3:** `F0` vs `F0 + K` (VALUE) and `F0 + K` vs `F0 + K^π` (ORDER), all in true order.

---

## 6. Trunk-level order arm (S4)

The deployed network of §1, unchanged, trained and tested on block-shuffled sequences. Run only if S2 passes and the advisor approves (proposal §5).

| Arm | Data path | Network |
|---|---|---|
| **A3-ign (re-run)** | as §1, current commit, seeds 2024–2028 | §1 |
| **A3-ign-shuffled** | CRN per segment → **`Π_b` per segment** (anchor and each filler separately) → DVS splice → trunk; test clips: CRN → `Π_b` per clip → trunk; scores un-permuted | §1 |

Shapes are those of §1. The re-run A3-ign is also S5's control.

---

## 7. Conditional TDP (S5; built only if S1a, S2, S3 (a) and (b) pass, and the advisor approves)

**Offline (per sequence):** the standardised `K ∈ S×131` from §3 (row 10), or `K^π` for arm Tπ (§4); `κ̂_t` below is a row of it. Windows and DVS segments slice the precomputed rows, so nothing is computed across a splice junction or a window start.

**In the model (pointwise, no temporal operation):**

| # | Stage | Input | Output | Params |
|---|---|---|---|---|
| 1 | `g_t = GELU(W_1 κ̂_t + b_1)`, `W_1 ∈ ℝ^{128×131}`, default init, from a **separate seeded `torch.Generator`** | `B × L × 131` | `B × L × 128` | 16,896 |
| 2 | `W_k ∈ ℝ^{512×128}` (+ bias), **initialised to zero** (same generator) | `B × L × 128` | `B × L × 512` | 66,048 |
| 3 | Residual add: `h_t = x̃_t + W_u(c·ũ_t ⊘ σ_u) + W_k g_t` | §1 row 5, row 2 | `H: B × L × 512` | — |

**Total: 82,944 ≈ 0.08 M** (*derived*). No LayerNorm. Logged every 50 steps: `ρ_k = ‖W_k g‖ / ‖x̃‖`.

**Unit test (before any S5 run):** with TDP frozen (`requires_grad = False`) and `W_k = 0`, the first 100 training losses equal the A3-ign re-run's at the same seed (to float tolerance). This checks both the zero effect at initialisation and that TDP's construction leaves the trunk's RNG stream untouched.

**Arms:**

| Arm | κ | New parameters vs A3-ign |
|---|---|---|
| A3-ign (re-run; shared with S4) | — | 0 |
| **T** | `K` (true order) | ≈ 0.08 M |
| **Tπ** | `K^π` (§4) | ≈ 0.08 M |

**Inference (if T is adopted):** §1, plus §3 over the clip and rows 1–3 above. Cost: negligible beside the encoders' ≈ 75 GFLOPs per step. §3 is causal except for ≤ 2 replicated steps at the clip start, and the per-clip median `m_x` (like CRN's R2) needs the whole clip, so the pipeline remains non-streaming.

---

## 8. Frozen, fixed, trained

| Frozen | Fixed statistics | Trained (deployed) | Trained (experiments only) |
|---|---|---|---|
| CLIP image and text towers; VideoMAE V2-S | CRN references, `s`, `σ_u`, `c`; **`σ_Δ`, `Q`, `μ_κ`, `σ_κ` (T2-train); `m_x` per sequence** | soft prompts, `W_u`, temporal encoder, co-attention, `H_bin`, `H_mul`; **`W_1`, `W_k` only if S5 adopts T** | the temporal probe (S2, S3) |

**Not added (proposal §8):** IAM prototypes, `w_int`, `L_cst`; ACC; a local Transformer; a similarity GCN; LLM attribute text.
