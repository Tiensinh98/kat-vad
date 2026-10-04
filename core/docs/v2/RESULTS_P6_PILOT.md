# KAT-VAD v2 — P6 pilot results

Pre-registration: `PREREG_ADDENDUM.md` §4 (read-out), §13 D16 (two batches), §14 O1–O7 (how it is computed),
§16 Amendment 7 (H1–H4, how D5 is explained). Raw outputs: `outputs/v2/v2_pilot/` (gitignored; this file is the
durable record). Runbook: `colab/v2/p6_pilot_batch1.ipynb`. Colab read-outs carry `commit: UNKNOWN` by design.

**The pilot reads mechanics on T2-val only. DoTA-dev is not printed (D4).** It decides no adoption.

## Batch 1 — A0 and A1, seed 2099 (2026-10-04)

Both arms: v2 dataset dir (T2-train minus T2-val: 219 sources / 645 windows held out), phase 4's stage-2 recipe,
20 epochs → **step 1740** (phase 4: 2040 on the full T2-train). J10 holds on Drive for both. Config diff: A0 ==
phase 4 except the seed and the v2-only keys at "off"; A1 == A0 except `v2.crn=R2`.

| arm | micro | macro | clip oracle | window AUC (O2) | O1 guardrails | pos R² `V^t` (input) | shortcut `V^t` | shortcut `y^bin` |
|---|---|---|---|---|---|---|---|---|
| A0 (raw CLIP) | 0.6586 | 0.6737 | 0.6986 | 0.6863 | **PASS** | −0.201 (−0.214) | 0.9999 | 0.5375 |
| A1 (CRN = R2) | 0.6683 | 0.6705 | 0.6986 | 0.6828 | **PASS** (+ A0 margin) | −0.030 (−0.021) | **0.5507** | 0.5201 |

Read-outs:

* **No collapse (C14 signature absent).** Window-level AUC ≈ macro for both arms, and micro < oracle, macro ≥ micro.
* **CRN removes corpus separability from the trunk.** The `V^t` source-shortcut AUC (T2-val vs DoTA-dev, unlabelled)
  drops **0.9999 → 0.5507**. This is the mechanism CRN was built for and the first time in the project that T2 and
  DoTA are not linearly separable after the encoder (C38 measured 1.000 on every foreign corpus). It is a
  mechanics read, **not** a DoTA performance claim.
* A1 − A0: micro +0.0097, macro −0.0032 — one seed, no interval, not a result.
* Position R² on `V^t` is negative for both (a ridge cannot predict `t/T` inside 20-step T2-val windows), as K-pos
  predicted for windows (0.575 ruler). O4 caveat (R² above A0's) not triggered in any meaningful sense.
* No motion arm in batch 1 → O3 (`ρ_u`, `‖W_u‖`) not applicable.

### D5 (O6) — **FAIL**, as O6 predicted; explanation pre-registered as Amendment 7

| checkpoint | step | T2-val micro | macro | window AUC | O1 |
|---|---|---|---|---|---|
| A0 v2 (s2099) | 1740 | 0.6586 | 0.6737 | 0.686 | PASS |
| phase-4 KIP-off s2024 | 2040 | 0.7714 | 0.6918 | 0.972 | **FAIL** |
| phase-4 KIP-off s2025 (retrained, D12) | 2040 | 0.7743 | 0.6988 | 0.964 | **FAIL** |
| phase-4 KIP-off s2026 (retrained, D12) | 2040 | 0.7567 | 0.6524 | 0.961 | **FAIL** |
| **Δ A0 − ref mean** | | **−0.1089** | **−0.0073** | | |

Bar: each |Δ| ≤ 0.02 → micro fails, macro passes. **The verdict stays FAIL.** Every reference trained on T2-val's
windows (§13 record) and every one exceeds the clip oracle with window AUC ≈ 0.97 — the in-sample signature, and
it inflates exactly the clip-classification columns (micro, window AUC), not the within-window ranking (macro).
That is an argument, not a measurement: Amendment 7 (§16) re-reads D5 on **T2-test**, unseen by all four
checkpoints, with a code-identity gate (H1: s2024 must reproduce its phase-4 T2-test 0.6230 / 0.6334).
Runbook `colab/v2/p6_d5_t2test.ipynb`. Read below: explained.

### D5 explanation (Amendment 7) — **EXPLAINED** (2026-10-04)

`colab/v2/p6_d5_t2test.ipynb`, `core.evaluate` on the parent dir's 1,106 T2-test windows (unseen by all four).

| checkpoint | T2-test micro | macro | phase-4 printed | T2-val micro | T2-val − T2-test |
|---|---|---|---|---|---|
| A0 v2 (s2099) | 0.6320 | 0.6389 | — | 0.6586 | +0.027 |
| s2024 | 0.6230 | 0.6334 | 0.6230 / 0.6334 | 0.7714 | **+0.148** |
| s2025 (retrained) | 0.6182 | 0.6347 | 0.6164 / 0.6352 (lost ckpt) | 0.7743 | **+0.156** |
| s2026 (retrained) | 0.6153 | 0.6078 | 0.6152 / 0.6057 (lost ckpt) | 0.7567 | **+0.141** |

* **H1 PASS** — s2024 re-scored by the v2 tree reproduces phase 4 to Δ micro 9e-9, Δ macro 0: the v2 changes left
  v1's forward and scoring path bit-identical for a KIP-off checkpoint.
* **H2 PASS** — A0 vs ref mean 0.6188 / 0.6253: Δ micro **+0.0132**, Δ macro **+0.0136**, both within 0.02.
* **H3: D5 = FAIL on T2-val, EXPLAINED — in-sample bias of the references. D5 no longer blocks P7.**
* H4 (printed): the in-sample inflation is +0.14…+0.16 micro for the references vs +0.027 for A0 (A0's own
  T2-val − T2-test gap is ordinary set-to-set difference). A0 reads *above* the references on T2-test with 15 %
  fewer training windows — one seed, inside the margin, **not a result**. The retrained s2025/s2026 land within
  0.002 of the lost snapshots' printed numbers.

## Batch 2 — A2 and A3 (V2-S) — *not run*

`colab/v2/p6_pilot_batch2.ipynb`, after the full-T2 V2-S cache (`p4_s_full_t2_d6.ipynb` step 2). O5 on
`dota_cap_dev` for all four arms (G5).
