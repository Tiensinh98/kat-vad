# LaGoVAD-OSTP Step 1 — patch-only local branch (L1) vs a free ensemble (L0b)

Branch `v_final-local` (off `v_final@8d69179`). Rule written **before any L1 or L0b number**; agreed with the user
2026-11-03 (L0b retrained with this branch's code; G1 floor 0.60). Proposal: `Thesis proposal … .md` (Nov 3
revision, two branches), §4.3 Step 1 (D1–D3). Runbook `colab/final/local_step1.ipynb`; read-out
`python -m core.tools.local_readout`.

## 1. Question

Before any YOLO, OPF or track attention is built: **does a patch-only local branch, trained alone with its own loss,
add to the host (L0) what a second seed of the host does not?** If not, the object branch has no footing and Step 2 is
not built.

Why an ensemble control: late-fusing *any* two models trained apart averages out per-clip noise and usually raises
macro a little. (L0+L1) − L0 cannot separate "L1 carries new evidence" from "two models beat one". (L0+L1) − (L0+L0b)
can: L0b is L0's exact recipe on another seed, fused by the same rule.

## 2. Rows (all Nexar `window`, v2 Nexar A2's recipe, 1,749 steps, LR 5e-5 ×1, this branch's code)

| row | what | seed | trained |
|---|---|---|---|
| **L0** | host = **W0** of `WU_ARMS.md` (W_u zero-init), released prompts, SA² off | 2024 | reused: `outputs/wu/W0/s2024` (0.8110) — same code path, W_u arms are byte-identical at ×1 |
| **L0b** | W0's exact command | **2025** | here |
| **L1** | `stp.sa2=true`, `stp.branch=local` | 2024 | here |

**L1 (the local branch), as built** (`KATVAD(local_branch=True)`, `_local_forward`):

* Input rows are the same caches as R4: `[c ; u ; P_sel]` (A2 bake + `clip_patches/Nexar_s8_k12_raw`). **`u` is never
  read** (no `MotionResidual` is built) and **`c_t` is only SA²'s query**: `x^AS_t = Σ_k a_{t,k} P_{t,k}`, `W_q` zero
  at init (= top-12 average), `add_cls=False`.
* `x^AS_t` → its own RoFormer (host config: 2 layers, 4 heads, window 25) → **one** `ConvScoreHead` (kernel 3 as the
  host's `model.score_head_kernel` for Nexar) → `z^L_t`.
* **No text**: no co-attention, no `H_mul`, no captions, no prompt (validated: `stp.branch=local` refuses a prefix and
  `captions_from_definitions`). The logit is emitted under the host's key `cls_bin_logits` so the trainer and the
  protocol-B scorer run unchanged; the class features passed in are ignored.
* Losses: top-k MIL BCE + the DVS pair (`dvs_sup`, `dvs_sup_mil`) on `z^L`, exactly the host's terms on its binary
  logit. `L_MIL-align` is absent (no `H_mul`).

Deviation from the plan sentence of 2026-11-03 ("loss = top-k MIL + DVS pseudo-labels"): none. Deviation from the
proposal §3.3.8: the local branch here has **no objects, no relation token, no track attention**, and the
patches are the existing `resized224` top-12 cache, not native-resolution windows — Step 1 tests the split, not the
resolution or the objects.

## 3. Fusion (fixed before any number)

Per clip, on `protocol_b_eval`'s seed-averaged native-frame scores (`clip_scores.npz`):
`s = 0.5·minmax(s_a) + 0.5·minmax(s_b)` (`LOCAL_FUSE_WEIGHT`). Protocol B is per-clip rank-based, so min-max leaves
each row's own AUC unchanged and only removes scale differences between the two rows. Same rule for L0+L1 and
L0+L0b. No weight is tuned.

## 4. Gates — DoTA-CAP (n/1397) dev, protocol B, 95 % source-video cluster bootstrap

| gate | read | pass iff |
|---|---|---|
| **G1** | L1 alone, macro | CI low > **0.60** **and** mean > **0.7132** (raw-Mo top-K read with no training, `stp_diag_pointing.md`) |
| **G2** | (L0+L1) − (L0+L0b), per clip | "gain shown": CI low > 0 **and** mean ≥ +0.01 (`V5_MIN_GAIN`) |
| **G3** | oracle, per clip `max(L0, L1) − max(L0, L0b)` | CI low > 0 |

**Printed, never deciding:** L1 − L0, (L0+L1) − L0, (L0+L0b) − L0, L0b − L0 (seed spread), each by ego flag; D2 =
Spearman of per-clip AUCs vs L0 and the share of clips each row wins by ≥ 0.05, for L1 and for L0b; the mean
per-clip Spearman of each row's native scores with v2's cubic position prior `p_T2` (0.822 on CAP-dev; a local branch
with a RoFormer and a conv head can learn position too); `mil`, SA² entropy / log K and ‖W_q‖ first → last
(L1's `mil` beside W0's 0.1760 — WL's overfit tell was `mil` 0.061 with Δ −0.042).

One seed per row ⇒ every sentence carries **PRELIMINARY (n = 1 seed)** and adopts nothing.

## 5. Decision table

| outcome | reading | next |
|---|---|---|
| G1, G2, G3 pass | the separate branch carries evidence the host lacks, beyond a free ensemble | confirm on L1 s2025/s2026 + L0b on two more seeds (own rule first), then Step 2 (YOLO + OPF-mini) |
| G1 pass, G2 or G3 fails | L1 learns, but is redundant with L0 or ensemble-sized | record the negative; Step 2 needs a new reason before any GPU is spent |
| G1 fails, L1 `mil` last < 0.1760 | overfit tell (as WL, v6) | stop the separate-branch line on Nexar's 1,500 videos |
| G1 fails, `mil` not below | the `resized224` patch input is weak | native-resolution windows for L1 alone (re-extraction) before any verdict on the split |
| L1's `p_T2` Spearman ≥ L0's | whatever L1 adds may be position | no gain is attributed to patches |

**Not licensed by any outcome:** any statement about objects, OPF, YOLO or native resolution; any claim beyond
DoTA-CAP-dev at n = 1; tuning the fusion weight; reading DoTA-CAP-eval.

## 6. Code

`stp.branch` (`host` | `local`, `validate_stp`), `KATVAD(local_branch)` (no `MotionResidual` / fusion / `H_mul`,
vanilla head; `_patch_base`, `_local_forward`), `SpatialAttentionAggregation.forward(add_cls=)`, trainer skips
`mul_mil` when there is no `cls_sim_mat`, `sliding_window_scores` returns an `(L, 0)` sim block for it, `stp_arm`
adds `stp_branch` only for non-host arms (every earlier read-out keeps its provenance keys), constants `LOCAL_*` /
`STP_BRANCH*`, `core/tools/local_readout.py`, tests `core/tests/test_local_branch.py`. Host paths (R4 / R6 / W0) are
unchanged: `_stp_rows` now reads the motion width from `row_motion_dim`, which equals `MotionResidual.motion_dim`
whenever the latter exists.

## 7. Read-out (2026-11-03) — **G1 FAIL without the overfit tell — PRELIMINARY (n = 1 seed)**

Raw: `Thesis-Final/outputs/local/REPORTS/local_readout.md` (untracked). DoTA-CAP (n/1397) dev, protocol B, 569 clips.

| row | macro [95 % cluster CI] |
|---|---|
| L0 (W0 s2024) | 0.8110 [0.7948, 0.8269] |
| L0b (W0 recipe s2025) | 0.8058 [0.7895, 0.8221] |
| **L1** | **0.6122** [0.5976, 0.6268] |
| L0+L0b | 0.8127 [0.7968, 0.8283] |
| L0+L1 | 0.7634 [0.7480, 0.7785] |

| gate | read | pass |
|---|---|---|
| G1 | 0.6122, low 0.5976 (bar 0.60) and mean 0.10 below the training-free Mo read 0.7132 | **no** |
| G2 (L0+L1) − (L0+L0b) | −0.0493 [−0.0584, −0.0402] (cost shown) | **no** |
| G3 oracle | +0.0003 [−0.0064, +0.0071] | **no** |

Printed: L1 − L0 −0.199 [−0.219, −0.179]; free ensemble (L0+L0b) − L0 +0.0017 [−0.002, +0.006]; seed spread
L0b − L0 −0.005. D2: Spearman of per-clip AUC with L0 — L1 0.259, L0b 0.887; clips won by ≥ 0.05 — L1 12.0 % vs L0
76.8 %, L0b 16.7 % vs 20.2 %. `p_T2` per-clip ρ — L0 0.561, L0b 0.557, **L1 0.152**. Training: `mil` L0 0.694 →
0.176, L0b 0.710 → 0.179, **L1 0.717 → 0.591** (underfit, not the WL overfit); SA² entropy 1.0 → 0.947, ‖W_q‖ 0 → 1.73.

**Reading.** L1 is different from the host (ρ 0.26) and does not track position (ρ 0.15), but is weak: it cannot fit
Nexar (mil 0.59) and loses to L0 on 77 % of clips; even a per-clip oracle gains nothing over a second host seed.
**Not licensed:** "patches / objects are useless", "two branches are wrong", anything about YOLO / OPF / native
resolution. **Post-hoc (printed, not a verdict):** L1's token carries the *content* of the 12 Mo-selected patches but
not the Mo values themselves, while the G1 reference 0.7132 *is* those values; Step 1B (`LOCAL_STEP1B.md`) tests
this one omission before any re-extraction.
