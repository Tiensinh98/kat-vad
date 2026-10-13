# KAT-VAD v2.2 — two tracks after S1–S3 (round 7 plan)

**Branch:** `v2.2`. **Design:** `core/docs/v2.2/KAT_VAD_PROPOSAL_v2.2.md` (rev. 2026-10-13c) — §R7 D1–D14 override
the body where they conflict. Predecessor plan (S1–S3): `katvad-v2.2-order-diagnostics.md`.
**Authority:** the user, 2026-10-13: "decide what you can and run the experiments" (not advisor-reviewed).

## 0. Status

| Batch | Track | Step | What | Where | Status |
|---|---|---|---|---|---|
| 1 | motion | **R0-saved** | ΔT / detrended ΔT / span of A0–A3, A3-ign, `p_T2` on saved CAP-dev scores; eligibility; per rate group | local | **RUN 2026-10-13 → D14: ΔT retired** (App. A.1) |
| 1 | motion | R0-probe | ΔT of `m5`, `s`, probes `F0`, `F0+κ`, `p`-only (R0-e), R0-d precursor share | Colab `colab/v2/v22_r0_k1.ipynb` | **RUN 2026-10-14** (App. A.2): `p`-only probe 0.833 → C-b answered |
| 1 | motion | K1 | `F0` vs `F0 + stats23` (flow/v1, causal), T2-val only | same notebook | **RUN 2026-10-14** (App. A.2): span +0.0103 [+0.0027, +0.0181], no soft stop; recorded, does not re-open the track |
| 2 | score | SC-1 | DF8 swap (plan `katvad-v2-df8-position-shift.md`, D1–D6 as recommended, + D8's R2 condition) | Colab | **next to build** (batch 1 closed 2026-10-14) |
| 3 | score | SC-2 | A3-B: V2-B extraction for full T2 + 5 runs, adopt on span only (re-opens N7, D9) | Colab | queued |
| 4 | motion | EMS-v2.1 + K2 | descriptor code, quality-only pilot | Colab | **parked (D14)** |

## 1. Files (batch 1)

| File | Role |
|---|---|
| `core/tools/v22_gates.py` | CLI `r0-saved` / `r0-probe` / `k1`; pure ΔT helpers (`onset_sets`, `onset_aucs`, `detrended_onset_aucs`, `fused_onset_aucs`), `rate_groups`, `demotion`, `causal_stats` |
| `core/constants.py` | `V22_*` round-7 block (step seconds under A1/A1′, ΔT windows, eligibility, futility, K1 soft stop, rate groups, detrend rounding) |
| `core/tests/test_v22_gates.py` | 23 tests: windows, ramp/step behaviour, rate groups, demotion, r0-saved e2e, r0-probe and k1 on the synthetic S1–S3 corpus |
| `colab/v2/v22_r0_k1.ipynb` | runbook for R0-probe + K1 |
| `outputs/v2/REPORTS/v22_gates/r0_saved_readout.{json,md}` | the local R0-saved read-out |

## 2. Choices fixed in code (G-numbers)

| # | Choice | Why |
|---|---|---|
| G1 | ΔT on the **step grid** only: saved native scores read at `native[::3]` (exact step scores: `to_native` interpolates through them); labels likewise | D12 |
| G2 | `t_s` = first abnormal step's time, `t_e` = end of the last abnormal step; positives `[t_s, min(t_s + 2 s, t_e))`; negatives = normal steps in `[t_s − 3 s, t_s − 0.5 s)`; eligible with ≥ 2 / ≥ 3 steps | proposal §5.2 |
| G3 | `f_2`-ΔT: `z(score) + 2 z(p_T2)`, z over the **whole clip**, AUC on the ΔT steps | `f_2` is a within-clip fusion everywhere else |
| G4 | DoTA step 0.30 s (A1′ = 10 fps), T2 step 8/30 s (A1 = 30 fps); both unverified (D1) | D1 |
| G5 | CAP rate group = `round(rate)` from `dota_cap_alignment.json`: `r1`, `r3`, else `other` (3 dev clips; reported, never gated) | D7 |
| G6 | Clusters: DoTA = YouTube source (`dota_group`); T2-val = source | as S1–S3 |
| G7 | R0-probe κ: CLIP only (S1b did not admit Δu), stats fitted on T2-train as S1/S3; probes = S3's `F0` / `F0+κ` plus a `p`-only probe (zero feature columns + the cubic `p`) | R0-b, R0-e |
| G8 | R0-d: each clip's top-5 `m5` steps that are normal, classed **precursor** (≤ 2 s before `t_s`), **earlier**, or **after** the span | proposal §2.7 |
| G9 | K1: flow/v1 `.stats.npy` (23 raw stats; rows = CLIP rows, else raise) shifted one step later (D6); appended to `F0` raw — the probe standardises on its training windows; trained on T2-train − T2-val windows, read per T2-val source (span, ΔT at 0.267 s); soft stop iff both Δ points < +0.005 | D6, D10 |
| G10 | Detrended ΔT (reported): per clip, OLS line of score on time over the **negatives**; AUC of the residuals on positives vs negatives (rounded to 1e-9 so a ramp ties at 0.5) | the validity check that retired ΔT (D14) |

## Appendix A — read-outs

### A.1 R0-saved, 2026-10-13 (local; `outputs/v2/REPORTS/v22_gates/r0_saved_readout.md`)

DoTA-CAP-dev, protocol B, step grid. Eligible **557 / 569** (`r1` 386 / 395, `r3` 168 / 171, other 3 / 3).

| score | span macro | ΔT | ΔT detrended | `f_2`-ΔT | ΔT `r1` | ΔT `r3` |
|---|---|---|---|---|---|---|
| A3-ign | 0.7936 | 0.8505 [0.8293, 0.8701] | 0.6307 | 0.9599 | 0.8295 | 0.9004 |
| A3 | 0.7610 | 0.8458 | 0.3216 | 0.9648 | 0.8263 | 0.8918 |
| A2 | 0.7559 | 0.8749 | 0.6032 | 0.9755 | 0.8530 | 0.9251 |
| A1 | 0.6592 | 0.7195 | 0.5200 | 0.9524 | 0.7036 | 0.7568 |
| A0 | 0.6752 | 0.7250 | 0.5890 | 0.9473 | 0.7047 | 0.7690 |
| **`p_T2`** | 0.8244 | **0.9700** [0.9577, 0.9807] | **0.0000** | — | 0.9809 | 0.9446 |

ΔT contrasts (paired, descriptive): A3-ign − A3 +0.0047 [−0.0089, +0.0172]; A3 − A0 +0.1208 [+0.0969, +0.1441];
A3 − A1 +0.1264; A3-ign − `p_T2` −0.1195 [−0.1433, −0.0961].

**Reading.** ΔT's premise (R0-a: "ΔT(`p_T2`) well below 0.822") is falsified: a no-pixel position prior wins ΔT at
0.970, because positives always follow negatives in time. Detrending swaps one artefact for another (each score's
pre-onset slope). → **D14: ΔT retired as a decision estimand; the §5.3 fallback (span + `f_2` + DF8) applies; the
motion track is parked after batch 1.** Side notes, descriptive: A3-ign's span gain over A3 is **not** at the onset
(ΔT +0.005, ∋ 0), consistent with E4-0's "floor suppression, not coverage"; `r3` clips read higher than `r1` for every
score including `p_T2`, i.e. a composition difference, not flow quality. The futility half on A3-ign (0.8505 < 0.90)
did not fire; moot.

### A.2 R0-probe + K1, 2026-10-14 (Colab; `outputs/v2/REPORTS/v22_gates/{r0_probe,k1}_readout.md`)

**R0-probe**: DoTA-CAP-dev, probes = S3's 5-seed setup; span on 568 clips, ΔT on 545 eligible (R0-saved: 557; the
probe reads the step labels of the order cache, not `native[::3]`, so the eligibility differs by 12 clips, and this is not material).

| score | span macro | ΔT | ΔT detrended | `f_2`-ΔT |
|---|---|---|---|---|
| `m5` | 0.7503 | 0.7415 [0.7133, 0.7693] | 0.5303 [0.4969, 0.5637] | 0.9618 |
| `s` | 0.7161 | 0.6765 | 0.6028 | 0.9523 |
| `F0` | **0.8529** [0.8406, 0.8650] | 0.8768 | 0.5060 | 0.9694 |
| `F0+κ` | 0.8539 | 0.8747 | 0.4989 | 0.9687 |
| **`p`-only probe (R0-e)** | **0.8328** [0.8154, 0.8506] | 0.9536 | 0.0568 | 0.9626 |

R0-b ΔT(`F0+κ`) − ΔT(`F0`) −0.0021 [−0.0109, +0.0065]. R0-d (top-5 `m5` steps outside the span, 1,156 steps):
precursor 0.270, earlier 0.176, **after 0.554**. T2-val: `F0` span 0.8411, `F0+κ` 0.8458. Demotion rule printed
"incomplete" (R0-saved's JSON was not on Drive), which is moot under D14.

**Reading.**
1. **C-b (results §6) is answered descriptively.** Same architecture, `p` only: 0.833. True-order `F0` 0.853 (+0.020),
   shuffled `F0` 0.825 (−0.008). So most of S2's +0.028 sits **above** a position-only probe; the shuffle's own damage
   is about −0.008. These are unpaired reads (per-clip probe scores not saved), so the numbers are descriptive and not gated.
2. ΔT ranks `p`-only first again (0.954), which confirms D14. Detrended, every score is ≈ 0.5 (`m5` 0.530, CI ∋ 0.5):
   **no score carries an onset signal beyond its own trend.**
3. `m5` fires mostly **after** the span (55 %), not before it (27 %). The change score is post-event activity, not
   anticipation. Do not write "precursor".
4. κ adds nothing at the onset either (R0-b ∋ 0), the same as S3 VALUE on the span.

**K1** (T2-val per source, 219 sources, 5 seeds): `F0 + stats23` − `F0` span **+0.0103 [+0.0027, +0.0181]**, 5/5 seeds
(+0.009…+0.022); onset +0.0100 [−0.0040, +0.0243]. No soft stop.

**Reading.** Global RAFT flow statistics add about one AUC point **in-domain** beyond CLIP + VideoMAE + `p`. The
point estimate clears +0.01 by 0.0003. **It does not re-open the motion track**, for four reasons:
(a) there is no DoTA endpoint, because DoTA pixels are unrecoverable, so the result cannot reach a decision set;
(b) as an input, `stats23` needs RAFT at inference, which breaks the RGB-only contract;
(c) distilling this target from RGB is what KIP-v1 did, and it was neutral on T2 (Option A, +0.0076 ∋ 0);
(d) D10 pre-declared K1 a weak read.
Permitted wording: "frame-global flow statistics carry ≈ 0.01 of in-domain signal that the RGB streams do not; the
transfer value is untestable here." **Batch 1 closed; motion track stays parked (D14).**
