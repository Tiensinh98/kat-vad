# KAT-VAD v2.2 batch 2 / SC-1 — DF8 position swap: results

**Date:** 2026-10-15 (Colab run, read locally). **Branch:** `v2`.
**Rule:** v2 `PREREG_ADDENDUM.md` §24 (Amendment 13, X1–X10), frozen 2026-10-14 before any crop was scored.
**Plan:** `.project/plans/katvad-v2-df8-position-shift.md` (App. A). Batch context: `.project/plans/katvad-v2.2-two-tracks.md`.
**Code:** `core/tools/position_shift.py` (`plan`, `heads`, `score`, `prior`, `readout`; `same-frame` added post-hoc);
runbook `colab/v2/df8_position_shift.ipynb`.
**Raw read-outs:** `outputs/v2/REPORTS/v22_df8/df8_{readout,same_frame,crops}.{md,json}`.
**Set:** DoTA-CAP-dev shift subset, n = 491 clips, 86 source clusters (78 of 569 dropped for < 1 s context on one
side). Protocol B per crop. No training, nothing adopted: v2's `B` = A3-ign stands.

---

## 1. Integrity (X10): all pass

| Gate | Result |
|---|---|
| (a) J10 | all 15 checkpoints (A3-ign / A3 / A0 × 5) at `global_step` 1740 |
| (b) `full` arm == saved protocol-B scores | max abs diff A3-ign 3.5e-07, A3 / A0 / `p_T2` 0 |
| (c) re-extracted V2-S heads | 491 / 491 clips ok, 7,524 windows, min cosine **0.99935** (bar 0.999: passes, thin margin) |
| (d) clip count | 491 |

Crop centres (p10 / p50 / p90): control 0.40 / 0.51 / 0.59; early 0.20 / 0.30 / 0.41; late 0.59 / 0.70 / 0.80.

## 2. Verdict (X8): **R1 > 0, X7 fails**

> The prior's advantage does not survive a position swap, but the detector also depends on where the accident sits.
> (DoTA-CAP-dev shift subset, n = 491; accident centre 0.30–0.70)

| Read | Δ (cluster bootstrap 95 %) |
|---|---|
| **R1** shifted(A3-ign) − shifted(`p_T2`) | **+0.1201 [+0.0974, +0.1413]** |
| same Δ on the full clips | −0.0351 [−0.0613, −0.0098] |
| R2 early − control | −0.1251 [−0.1405, −0.1096] |
| R2 late − control | +0.0744 [+0.0573, +0.0912] |
| R2 early − late | −0.1994 [−0.2254, −0.1732] |
| X7 (both moved − control lower bounds > −0.01) | **fails** |
| R3 full − control (context cost) | +0.0264 [+0.0142, +0.0383] |

Macro per arm (seed-averaged):

| model | full | control | early | late |
|---|---|---|---|---|
| A3-ign | 0.7934 | 0.7670 | 0.6419 | 0.8414 |
| A3 | 0.7588 | 0.7091 | 0.6234 | 0.8135 |
| A0 | 0.6731 | 0.6436 | 0.6018 | 0.7086 |
| `p_T2` | 0.8285 | 0.8742 | **0.3760** | 0.8672 |

## 3. What the registered numbers mean

- **R1 > 0 is close to guaranteed and carries little information.** `p_T2` is a cubic in `t/L`. The early arm puts
  the accident where the cubic is low (0.376, below chance). On control and late the prior still beats A3-ign, and on
  the full clips it does too (−0.035). The whole R1 margin comes from the prior's collapse on early.
- **The informative read is R2.** A3-ign loses 0.20 from late to early, about twice A0's gap (≈ −0.107, unpaired).
  R2 does not say *why*. Early and late differ in content as well as position: early's negatives are mostly
  **aftermath**, late's are **pre-accident driving**. CRN R2 also takes its median over the crop's own rows, so its
  reference moves with that content (plan §5 risks 1–2).
- **X9, printed only:**
  - `f_2` collapses on early for every model (A3-ign 0.476, A3 0.455, A0 0.447) and peaks on late (0.913). See §5.
  - R1 by span: < 2 s +0.201, 2–4 s +0.138, ≥ 4 s −0.015 [−0.045, +0.015]. This is geometry: a long span leaves
    little room to move the centre.
  - D2 (the leaky cache slice) gives R1 +0.133, the same direction.
  - Kept vs dropped full-clip macros overlap (A3-ign 0.793 / 0.777).
  - Per-seed spread is small (A3-ign early 0.625–0.649, late 0.824–0.834).

## 4. Post-hoc: same frames, different position (printed only; decides nothing)

The question the user asked after reading §2: *is A3-ign learning position?* `position_shift same-frame` answers it
from the saved scores, with no new run. For each shifted crop it compares two AUCs on the **identical frames and
labels**:

- the crop's own scores, against
- the full clip's scores on those frames.

The difference ("shift") is what moving the frames does (position plus crop context), with content fixed. `p_T2`
is the pure-position ruler.

| model | control | early | late |
|---|---|---|---|
| `p_T2` (position only) | +0.148 [+0.132, +0.164] | **−0.328** [−0.364, −0.293] | **−0.053** [−0.070, −0.036] |
| **A3-ign** | +0.056 [+0.045, +0.068] | −0.077 [−0.089, −0.065] | **+0.038** [+0.027, +0.050] |
| A3 | +0.028 [+0.016, +0.041] | −0.045 [−0.062, −0.030] | +0.027 [+0.015, +0.040] |
| A0 | +0.008 [−0.002, +0.017] | −0.017 [−0.030, −0.004] | +0.011 [+0.002, +0.021] |

**Per-clip tracking** of the ruler's shift (Pearson r, OLS slope). A3-ign: control r −0.03; early r +0.21,
slope +0.09; late r +0.23, slope +0.18. A0: r −0.13…−0.05.

**R2's early − late, decomposed:**

| model | total | frame set (full-clip scores) | move |
|---|---|---|---|
| A3-ign | −0.199 | −0.084 [−0.110, −0.059] | −0.115 [−0.131, −0.100] |
| A3 | −0.190 | −0.117 | −0.073 |
| A0 | −0.107 | −0.079 | −0.028 |
| `p_T2` | −0.491 | −0.216 | −0.275 |

Reading:

1. **A3-ign does not behave like a position prior.** Moved over the same frames, it shifts by a quarter (early) to
   a third (control) of the ruler. On late it shifts in the **opposite direction** to the ruler.
2. **A small position-like component remains.** Per-clip r ≈ 0.2 on early and late, slope ≤ 0.18. The architecture
   allows it: the encoder has RoPE only (no absolute position embedding), but its attention window (25 steps) covers
   a whole crop (median ≈ 19 steps at stride 3), so every token sees the crop edges. **Never write "position-invariant"
   or "does not use position".**
3. **A0 has the same encoder and heads and barely moves** (≤ 0.017). The extra crop sensitivity of A3 / A3-ign
   therefore enters through CRN (median over the crop) and V2-S (causal window clamped at the crop start). Both are
   content and context paths, not position encodings. This attribution is inferred, not measured. The direct test
   would score A3-ign with the CRN reference taken from the full clip; it was not run (user, 2026-10-15).
4. **The frame set alone explains ≈ 40 % of A3-ign's early − late gap** (−0.084 of −0.199). It is present before
   any crop: the early frames' negatives are aftermath.

**Licensed sentence (post-hoc, exploratory):**

> On identical frames moved within the clip, A3-ign's AUC shifts by at most a quarter to a third of what a pure
> position prior shifts, with the opposite sign on late placement; a weak position-like component remains (per-clip
> r ≤ 0.23). The detector is context-sensitive, not a position prior.

## 5. Caveat for every earlier `f_2` number

`f_2 = z(score) + 2 z(p_T2)` inherits the prior's position distribution. On the early arm it falls to 0.45–0.48 for
every model. E3, E4-0, Final and S2 printed `f_2` as the "beyond position" read. That wording holds **only for
DoTA-CAP's own placement of accidents (centre ≈ 0.5–0.7)**. Quote every `f_2` with that qualifier. Pending (bn).

## 6. What follows

- The §24 verdict is final. Nothing re-decides; A3-ign stays.
- Optional, not scheduled: one scoring-only notebook with the CRN reference taken from the full clip, to test §4
  point 3. Only if the attribution is challenged.
- Next batch: SC-2 A3-B (two-tracks plan, batch 3).
