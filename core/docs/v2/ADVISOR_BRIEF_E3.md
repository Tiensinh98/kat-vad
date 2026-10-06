# KAT-VAD v2 — E3 result and two findings that change the claim

**Prepared for:** advisor review, before the Final (sealed-set) step. **Date:** 2026-10-06. **Code:** branch `v2`.
**Author:** sinhpham. Full record: `core/docs/v2/RESULTS_E3.md`; rules: `core/docs/v2/PREREG_ADDENDUM.md` §17–§19.

## 1. What was run

The pre-registered 2 × 2 factorial of the v2 proposal (§10): **A0** baseline (LaGoVAD trunk, frozen CLIP) · **A1** + Clip-Referenced
Normalization (CRN) · **A2** + a frozen VideoMAE V2-S stream (zero-init residual) · **A3** both. 5 seeds each (20 runs),
trained on DADA-2000 T2, zero-shot on DoTA-dev (stride-3 protocol B). DoTA-eval and T2-test are still sealed.

## 2. Result: the pre-registered rule adopts A3

| contrast (macro AUC, paired t95, n = 5) | Δ | interval |
|---|---|---|
| A3 − A0 (full v2) | **+0.086** | [+0.057, +0.114], 5/5 seeds |
| A3 − A1 (video stream on top of CRN) | **+0.098** | [+0.086, +0.109] |
| A1 − A0 (CRN alone, full DoTA-dev) | −0.015 | [−0.037, +0.007], not detectable (MDE 0.028) |

The guardrails hold on every seed (no in-domain collapse). CRN does remove corpus separability (T2 vs DoTA on the trunk:
1.00 → 0.57), but on its own it does not improve DoTA.

## 3. Two findings you should know before the Final step

**(a) The stream is not motion.** In the shuffle control (D6), permuting the 16 frames inside every VideoMAE window
does not change the probe (Δ ±0.0004, interval contains 0). The thesis will describe the stream as "a second (video)
appearance encoder", not "motion" or "kinematics-aware".

**(b) Position explains much of the DoTA gain.** A cubic probe on *relative frame position only* (no pixels), fitted on the
T2 training windows, scores **0.822** on the same DoTA clips, above every model (A3 0.758). DoTA accidents sit
mid-clip, so per-clip macro AUC rewards position. We ran a post-hoc check on dev: we added each model's score to the
position prior inside each clip and recomputed the AUC.
* The baseline adds nothing beyond position (0.819 vs 0.822).
* A3 adds +0.020 over position.
* After this position control, A3's gain over A0 shrinks from +0.081 to **+0.023 [+0.013, +0.032]**, still positive.
* A2's gain (stream without CRN) falls to +0.007 (interval contains 0).

## 4. What I propose (Amendment 10, written before any sealed set is opened)

* The adoption of A3 stays closed.
* The Final step opens DoTA-CAP-eval, DoTA-eval (A0/A1) and T2-test **once**.
* It repeats the position-controlled read above. The weight and the sentence each outcome licenses are fixed in
  advance, so the dev check becomes a test on held-out data.
* Every DoTA number in the thesis is printed with the position prior beside it.

Procedural note: the E3 rules were committed before any E3 number, but I formally signed them only after the
read-out. No rule was changed after the number was seen. The thesis will say this.

## 5. Questions for you

1. Is "video appearance stream + CRN, gain ≈ +0.02–0.05 beyond position" an acceptable headline for the thesis contribution?
2. Should the Final step also report a benchmark or protocol where accident position is not informative?
   (Option: score windows centred at random offsets.) Adding this now would be a new amendment.
3. Any objection to opening the sealed sets with the Amendment 10 rules as written?
