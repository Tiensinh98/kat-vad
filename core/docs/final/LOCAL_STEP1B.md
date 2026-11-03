# LaGoVAD-OSTP Step 1B — L1mo: the local branch with its motion magnitudes

Branch `v_final-local`. Rule written **after** Step 1's read-out (`LOCAL_STEP1.md` §7) and **before any L1mo number**;
user-approved 2026-11-03. Runbook `colab/final/local_step1b.ipynb`; read-out `python -m core.tools.local_readout
--local-name L1mo --ref L1 …`.

## 1. Question

Step 1's L1 failed G1 (0.612 < 0.7132) while underfitting Nexar (mil 0.59). Its token was the *content* of the 12
patches chosen by Mo; the Mo values — how strongly those patches changed — were used to choose and then dropped, yet
the G1 reference 0.7132 is exactly those values used as a score. **Does L1 fail only because it never sees Mo?**

## 2. The one change (L1mo = L1 + `stp.mo_token=true`, same seed 2024, same recipe)

Notation: N = 49 cells, K = 12, D = 512.

* `m_t = [Mo_{t,(1)} ≥ … ≥ Mo_{t,(K)} ; mean_j Mo_{t,j} ; std_j Mo_{t,j}] ∈ ℝ^{K+2}` — the K largest are exactly the
  Mo of the K patches in `x^AS_t` (`core.data.stp_patches.mo_summary`, from the cache's existing `mo/` arrays; no
  extraction).
* `token_t = x^AS_t + W_m · LN(m_t)`, `LN` over K+2 (γ, β ∈ ℝ^{K+2}), `W_m ∈ ℝ^{D×(K+2)}` + `b_m ∈ ℝ^D`, PyTorch
  default init (the branch has no trained host to protect); zeroed on padded steps.
* New parameters: `D(K+3) + 2(K+2)` = 7,708. Everything else (selection, SA², RoFormer, head, losses, data, steps,
  LR) is L1's. L1mo − L1 therefore isolates `m_t`.
* `‖W_m‖_F` is logged as `mo_proj_norm`.

## 3. Rows and gates — unchanged from `LOCAL_STEP1.md` §3–§4

L0 = W0 s2024 and L0b = W0 recipe s2025 are **reused** (their reads exist); only L1mo is trained. Same fusion
(0.5 minmax + 0.5 minmax), same G1 (low > 0.60 and mean > 0.7132 — now a like-for-like bar, since L1mo sees the
values the reference scores), same G2, same G3. Printed beside them: L1's row and L1mo − L1 (`--ref`), D2, `p_T2` ρ,
`mil`, SA² entropy, `mo_proj_norm`. One seed ⇒ **PRELIMINARY (n = 1 seed)**.

## 4. Decision table (stop rule fixed now)

| outcome | reading | next |
|---|---|---|
| G1 fails | even with Mo in the token, training does not beat Mo alone: the `resized224` patch path adds nothing learnable | **stop the two-branch line**; write the negative-results chapter; no native re-extraction |
| G1 passes, G2 or G3 fails | the local branch stands alone but repeats what the host (with VideoMAE) knows | **stop**; record the negative |
| G1, G2, G3 pass | the branch carries evidence the host lacks beyond a free ensemble | confirm on 2 more seeds (own rule), then native windows / YOLO |

**Not licensed by any outcome:** any claim about objects, OPF, YOLO or native resolution; tuning the fusion weight;
reading DoTA-CAP-eval; claiming beyond DoTA-CAP-dev at n = 1.
