# KAT-VAD v2.2 — results of the order diagnostics S1–S3

**Date:** 2026-10-13 (run on Colab, read-outs dated 2026-10-13). **Branch:** `v2.2`.
**Design:** `KAT_VAD_PROPOSAL_v2.2.md` §3–§6, §9–§10; `KAT_VAD_v2.2_ARCHITECTURE.md` §2–§5.
**Plan:** `.project/plans/katvad-v2.2-order-diagnostics.md` (choices O1–O15, App. A).
**Code:** `core/tools/order_{features,probe,diagnostics}.py` (`b4e18ae`, `2dae545`); runbook `colab/v2/v22_order_s1_s3.ipynb`.
**Raw read-outs:** `outputs/v2/REPORTS/v22_order/s{1,2,3}_readout.{md,json}`.
**Sets:** decisions on DoTA-CAP-dev (569 clips, 91 source clusters, protocol B, n/1397); T2-val (219 sources) reported.
No sealed set was loaded (O15).

---

## 1. Verdict

| Step | Question | Verdict | Consequence |
|---|---|---|---|
| S1a | Does step-to-step change magnitude rank anomalous steps? | **PASS** | TDP branch open |
| S1b | Is the VideoMAE change Δu free of echo? | **Δu not admitted** | κ = CLIP only (131 channels) |
| S1c | Are the first steps of a clip unbiased? | **Early-onset prediction withdrawn** | affects S5 only |
| **S2** | **Does step order carry signal?** | **PASS** | order finer than ≈ 1.5 s is readable |
| S3(a) | Do explicit velocity/acceleration features (κ) add value? | **FAIL** | TDP not built |
| S3(b) | Does κ's gain depend on order? | **FAIL** | — |

**Claim-ladder row (proposal §9): "S2 passes, S3(a) fails".** Permitted wording:

> Step order finer than ≈ 1.5 s carries signal beyond coarse position, but explicit velocity/acceleration features add
> nothing a small temporal model does not learn itself.

Every order statement carries its estimand: *finer than ≈ 1.5 s, beyond coarse position, beyond VideoMAE's 1.5 s bag,
lag-2 neighbours remain*. Forbidden: "TDP" as a contribution; any motion / kinematic wording.

**Outcome row (proposal §10, row 2):** analysis chapter; S4 (10 runs) only if the advisor approves. **S5 is not run.
The deployed network stays A3-ign.**

Bar everywhere (§6.1): cluster-bootstrap 95 % lower bound > 0 **and** point estimate ≥ +0.01, B = 10,000, on
seed-averaged scores (probes: 5 seeds 2024–2028, per-seed Δs = stability check only).

---

## 2. S1 — free diagnostics (no training)

### S1a — change score `m5` (CLIP, rate-normalised, 1.5 s causal max)

| Read | DoTA-CAP-dev | T2-val (reported) |
|---|---|---|
| AUC(`m5`) | **0.7503** [0.7330, 0.7669] | 0.6943 [0.6657, 0.7208] |
| AUC(`m5^π`), quantile-matched twin | 0.4790 [0.4708, 0.4873] | 0.4966 |
| **AUC(`m5`) − 0.5** | **+0.2503 [+0.2330, +0.2669]** | +0.1943 [+0.1657, +0.2208] |
| **AUC(`m5`) − AUC(`m5^π`)** | **+0.2713 [+0.2537, +0.2891]** | +0.1977 [+0.1686, +0.2259] |
| AUC(`s`) − 0.5 (secondary, no causal max) | +0.2161 [+0.1996, +0.2317] | +0.1663 |
| AUC(`s`) − AUC(`s^π`) (secondary) | +0.2404 [+0.2255, +0.2547] | +0.1568 |

Printed: per-clip corr(AUC(`m5`), AUC(`p_T2`)) = **−0.053**; ego 0.785 (321 clips) vs non-ego 0.706 (247); Δ vs twin
ego +0.296, non-ego +0.239; share of each clip's top-5 `m5` steps inside the span **0.593**. All 569 sequences have
≥ 12 steps (gap 6 everywhere); none excluded.

### S1b — Δu echo (equivalence margin ±0.1 SD)

| Stream | Clips | Δ (+5…+6 − +10…+15) | Margin | Inside? | Autocorr lags 3–7 |
|---|---|---|---|---|---|
| `u` (decides) | 120 | +2.877 [+1.418, +4.362] (≈ 0.45 SD) | ±0.638 | **no** | 0.14, 0.09, 0.05, −0.03, −0.06 |
| `x` (printed) | 120 | +1.286 [+0.188, +2.372] (≈ 0.19 SD) | ±0.684 | no | 0.36, 0.25, 0.16, 0.07, 0.01 |

### S1c — clip start (560 clips, normal steps 0–2 vs ≥ 3)

SMD: `s` **+0.377**, `m5` **−0.346**, `e` +0.196, `v̄` mean **+0.395**, `ā` mean +0.080. Printed: max |SMD| over all 131
κ channels 0.390; 61 channels above 0.1.

---

## 3. S2 — D6-step probe (true order vs adjacency-free block shuffle)

Probe: 2 × Conv1d(kernel 5, 64 ch), RF 9 steps, trained on T2-train − T2-val, context-padded windows.

| Input | True (DoTA-CAP-dev) | Shuffled | **Δ** | `f_2` Δ | T2-val Δ | Per-seed Δ |
|---|---|---|---|---|---|---|
| **`F0 = [x̃; ũ; p]`** (decides) | 0.8532 [0.8416, 0.8651] | 0.8250 [0.8140, 0.8358] | **+0.0283 [+0.0212, +0.0356]** | +0.0138 [+0.0103, +0.0174] | +0.0165 [+0.0060, +0.0267] | +0.027, +0.030, +0.030, +0.026, +0.027 |
| `[x̃; p]` CLIP only (printed) | 0.7260 [0.7084, 0.7433] | 0.7140 [0.6969, 0.7313] | +0.0119 [+0.0015, +0.0230] | +0.0087 [+0.0035, +0.0138] | +0.0252 [+0.0103, +0.0401] | +0.009, +0.013, +0.027, +0.002, +0.010 |

T2-val absolute (F0): true 0.842, shuffled 0.825. Unshuffled short sequences: 0.

---

## 4. S3 — κ-probe (κ = 131 CLIP channels, T2-train standardised)

| Contrast | a | b | **Δ** | `f_2` Δ | T2-val Δ | Per-seed Δ |
|---|---|---|---|---|---|---|
| **VALUE** `F0+K − F0` | 0.8531 | 0.8538 | **−0.0007 [−0.0063, +0.0047]** | +0.0006 [−0.0026, +0.0038] | −0.0004 [−0.0108, +0.0101] | +0.005, +0.013, −0.006, −0.005, −0.010 |
| **ORDER** `F0+K − F0+K^π` | 0.8531 | 0.8451 | **+0.0080 [+0.0021, +0.0140]** | +0.0019 [−0.0014, +0.0053] | **−0.0034** [−0.0123, +0.0055] | +0.036, +0.012, −0.001, +0.004, −0.005 |

Printed: DADA↔DoTA shortcut AUC on κ **0.905**. Sequences: T2-train 1,272, T2-val 219, DoTA 569; all ≥ 12 steps.
`F0` reproduces across runs: 0.8532 (S2) vs 0.8538 (S3).

---

## 5. Interpretation

1. **Change information exists, and the temporal probe already has it.** A parameter-free change score reaches 0.750 on
   DoTA-CAP-dev (A3 dev: 0.758) and is nearly orthogonal to position (r = −0.05). But adding the full κ to `F0` changes
   nothing (−0.0007, interval inside ±0.01). A 9-step convolutional probe forms the same differences itself, as
   correction R1.2 predicted (`V^t = H + Enc(H)` can form signed differences).
2. **Step order carries signal, and more with the video stream than without.** S2's Δ on `F0` (+0.028) is about 2.4×
   the CLIP-only Δ (+0.012). The proposal expected the opposite (§3: VideoMAE's bag should absorb change, so `F0`
   "leans toward null"). True-order alignment of `x̃` with `ũ` is itself informative. The pass is clean: tight
   interval, 5/5 seeds within 0.026–0.030, T2-val same sign, `f_2` excludes 0.
3. **κ's order contrast is not a near-pass.** +0.008 misses the point bar, T2-val has the opposite sign, seeds split
   3/2, and `f_2` contains 0. κ is also corpus-specific (shortcut 0.905).
4. **ego > non-ego again** (0.785 vs 0.706 for `m5`), the same direction as A3 (results §6). Global change favours
   ego-involved accidents.

---

## 6. Caveats (recorded; the follow-up checks were NOT run, by the user's decision 2026-10-13)

| # | Caveat | Effect on the claim |
|---|---|---|
| C-a | **The quantile-matched twin scores below chance** (`m5^π` 0.479, `s^π` 0.476). §3 expected it to be high, because it keeps each step's atypicality. | "Beyond atypicality" for S1a is in practice "beyond chance". The control removed nothing measurable, so do not present it as a strong control. Verdict unchanged. |
| C-b | **Absolute level of S2.** True-order `F0` 0.853 ≈ cubic `p` alone in E2(d) (0.847 [0.831, 0.863] transfer, `RESULTS_E2D.md:31`; a logistic step probe, not this conv probe). Shuffled `F0` (0.825) sits below it. | The registered contrast stands. But "order helps beyond position" and "a shuffled context hurts transfer" are not separated in absolute terms. `f_2` (+0.014) is the only beyond-position evidence. The same-architecture `p`-only reference was not built. |
| C-c | `m5` is low at clip starts (SMD −0.35). Normal steps 0–2 therefore rank low, a small position-like effect. | Probably small (3 of ≈ 33 steps; r with `p_T2` −0.05). Not measured. |
| C-d | κ-level replicate padding did **not** remove the clip-start spike (S1c: 61/131 channels > 0.1). | A design defect for any future TDP; moot now. |
| C-e | S1b cannot separate echo from genuine post-accident change (`x` also fails). | Write "Δu was not shown echo-free", never "VideoMAE echoes". |
| C-f | The read-outs omit `p_T2` beside each macro, which §6.1 requires. Reference values: `p_T2` on DoTA-CAP-dev 0.822; A3-ign 0.791. | Reporting gap only. |
| C-g | **0.853 is not a better detector than A3-ign (0.791).** The probe receives `p`, which encodes DoTA's mid-clip accident bias. | Never quote it as a model result. |

---

## 7. What follows

- **S5 / TDP:** closed (S3(a) FAIL). κ code stays as a diagnostic tool.
- **Advisor check (proposal §10, row 2):** (i) is an analysis chapter acceptable in place of a second component — the
  chain *KIP fails → the video stream is appearance → step order carries signal, but the trunk can learn it itself*;
  (ii) approve or decline S4 (10 runs).
- **S4 assessment.** A pass credits the LaGoVAD trunk. A null says "the trunk does not use the order the probe finds",
  but its remedy (S5) is closed. So S4 informs the chapter, not the model. Cheaper preview: K0 (permute steps at
  inference on the existing A3-ign checkpoints; descriptive, decides nothing).
- **The open problem is unchanged:** `p_T2` 0.8256 > A3-ign 0.8120 on CAP-eval. The candidates that address it are
  DF8/H1 (deferred) and H4 (V2-B), not S4.
