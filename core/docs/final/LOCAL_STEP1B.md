# LaGoVAD-OSTP Step 1B — L1mo: the local branch with its motion magnitudes

Status: **read out, stopped (§5).** Branch `v_final-local`. Rule written **after** Step 1's read-out (`LOCAL_STEP1.md` §7) and **before any L1mo number**;
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

## 5. Read-out — **G1 FAIL → the two-branch line stops — PRELIMINARY (n = 1 seed)**

Raw: `outputs/v_final/local/REPORTS/L1mo/local_readout.md`. DoTA-CAP (n/1397) dev, protocol B, 569 clips; J10 step 1749.
This copy of the doc sits on `v_final-lagovad-zs` beside the committed results; the code and notebooks stay on
`v_final-local`.

| row | macro [95 % cluster CI] |
|---|---|
| L0 (W0 s2024) | 0.8110 [0.7948, 0.8269] |
| L0b (W0 recipe s2025) | 0.8058 [0.7895, 0.8221] |
| **L1mo** | **0.7041** [0.6841, 0.7230] |
| L1 (printed) | 0.6122 [0.5976, 0.6268] |
| L0+L0b | 0.8127 [0.7968, 0.8283] |
| L0+L1mo | 0.8006 [0.7825, 0.8182] |

| gate | read | pass |
|---|---|---|
| G1 | low 0.6841 > 0.60, but mean 0.7041 < 0.7132 | **no** |
| G2 (L0+L1mo) − (L0+L0b) | −0.0121 [−0.0212, −0.0031] (cost shown) | **no** |
| G3 oracle | +0.0133 [+0.0060, +0.0208] | yes |

Printed: L1mo − L1 **+0.0919 [+0.0737, +0.1096]** (ego +0.121, non-ego +0.054); L1mo − L0 −0.1069 [−0.1252, −0.0893];
G2 by ego flag: ego −0.0034 [−0.0147, +0.0075], non-ego −0.0233 [−0.0368, −0.0094]. D2: Spearman of per-clip AUC with L0
0.368 (L0b 0.887); clips won by ≥ 0.05: L1mo 20.0 % vs L0 57.3 %. `p_T2` per-clip ρ 0.300 (L0 0.561); `p_T2` 0.8220 above
every row. Training: `mil` 0.744 → 0.688 on the last batch, ≈ 0.63 mean over the last epoch (L1 ≈ 0.62, L0b ≈ 0.19), so
last-batch values cannot rank L1mo against L1; SA² entropy 1.0 → 0.904, ‖W_q‖ 0 → 2.00, `mo_proj_norm` last 13.06.

**Outcome (§4, row 1, fixed before the number): stop the two-branch line; write the negative-results chapter; no native
re-extraction.** The read-out tool's verdict string ("the patch input itself is weak") is `LOCAL_STEP1.md` §5's row and
does not reopen re-extraction here.

**Reading.** Adding `m_t` lifts the branch by +0.09, yet it still only reaches about what Mo scores with no training
(0.7132 [0.6956, 0.7294], CIs overlap; no paired L1mo − Mo read exists, so write "did not beat", not "worse than").
It still cannot fit Nexar. G3 passing means some clips differ between the rows, but an oracle picks with the labels, and
the fixed fusion costs; G3 does not override G1.
**Not licensed:** "patches / objects are useless", "two branches are wrong", anything about YOLO, OPF or native resolution,
or a tuned fusion weight read from the ego split.
