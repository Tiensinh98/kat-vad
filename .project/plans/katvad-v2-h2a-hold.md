# H2(a) — causal post-hoc hold on `y^bin` (pre-registration)

**Status:** rule fixed 2026-10-09, approved by the user before any number was read. Exploratory, **not an
amendment**: it decides only whether H2(a) is kept as a free post-processing step; it never re-decides E3 or
Final. Source: `core/docs/v2/REPORT_V2_RESULTS.md` §6.3 (onset fade) and §8 (H2).

## 1. Question

A3 peaks ~1–2 s after onset and fades (§6.3; argmax inside the span on 61 % of clips), and A3 − `p_T2` worsens
with the accident share (−0.001 at <30 % → −0.196 at >70 %). Does a **causal** hold of the score — no training,
no DoTA-fitted parameter — recover the long-span clips, and does that gain survive the T2 position prior?

## 2. Inputs (no new model run)

* E3's DoTA-CAP-dev outputs (`<e3-dir>/REPORTS/pb/<arm>/dota_cap_dev/clip_scores.npz`, seed-averaged native
  scores + labels) and `position_prior/dota_cap_dev` (`p_T2`). DoTA-CAP-eval / DoTA-eval are **not read**.
* T2 window meta (`meta.json` of `DADA2000_orig_T2_w20s8`) — only for the hold length.
* **Gate G0 (hard):** `final_readout.dev_gate` must pass on the same `<e3-dir>` (reproduces `RESULTS_E3.md` §5
  to 1e-4), so the scores read are E3's.
* Known limit: only seed-averaged curves exist locally, so the hold is applied to the seed mean and every CI is
  the cluster bootstrap over source videos (as Final's `f_2`), not a seed-paired interval.

## 3. The hold length — fixed by T2 alone

`w` = median over **T2-train sources** (meta `split == train`, source ∉ `t2_val_sources.txt`, one value per
source) of the annotated abnormal span `(end − start) / DADA_ASSUMED_FPS` seconds, converted to DoTA native
frames: `w = max(1, round(median_s × DOTA_FPS))`. No DoTA number enters `w`. Never swept.

## 4. Variants

* **`max_hold` (primary, decides):** `s'_t = max(s_{t−w+1} … s_t)` over the available past.
* **`ema` (printed, never decides):** `s'_t = α s_t + (1 − α) s'_{t−1}`, `α = 2 / (w + 1)` (span-`w` EMA).

## 5. Rule (decides, A3 only, DoTA-CAP-dev two-class clips)

`GO` iff **all three** hold for `max_hold` vs A3 raw:

* **G1** — per-clip Δ AUC over clips with accident share ≥ 0.5 (bins `50-70` ∪ `>70`): cluster-bootstrap CI low > 0.
* **G2** — beyond position: per-clip Δ AUC of `f_2 = z(score) + 2·z(p_T2)` (within clip, Final's P4 at the primary
  weight), held vs raw, over all two-class clips: CI low > 0.
* **G3** — no net harm: per-clip Δ AUC over all two-class clips, mean ≥ 0.

Else `KILL`.

## 6. Printed, never decided on

Every share bin's Δ; the same tables for `ema`; raw → hold Δ for A0, A1, A2; `p_T2` macro; Δ on **tail clips**
(the last abnormal frame is followed by ≥ `w` normal frames — where a hold can only add false positives).

## 7. What each outcome buys

* **GO:** H2(a) is kept as a free, causal post-processing of A3, a dev result. Any number on an eval set is
  labeled "second look at an opened set" (REPORT §8 global constraint). Next: design H2(b) — note that
  `k = max(1, L // MIL_TOPK_PCT)` already scales with `L`, so the knob is the divisor, not "a fraction".
* **KILL:** drop H2(a). `ema` cannot rescue it (printed only). H2(b) needs its own case.

## Appendix A — read-out (2026-10-09, `outputs/v2/REPORTS/h2a_hold/`, copy of `<e3-dir>/REPORTS/h2a_hold/`)

**Provenance.** This file was written 21:10, the tool 21:12, the read-out 21:16 (local mtimes); the plan was not
edited after the run. The rule was **not in git** before the number (committed after, with the read-out) — lesson
(aw)'s failure mode, recorded rather than hidden. The verdict is KILL, so no edit had a motive.

**G0 PASS.** `w` = **30** DoTA frames (median T2-train span 3.02 s over 1,272 sources). 569 two-class clips.

**Verdict: KILL** (A3, `max_hold`) — G1 Δ long +0.0366 [−0.0066, +0.0787] ✗; G2 Δ `f_2` **−0.0222**
[−0.0292, −0.0149] ✗; G3 Δ all **−0.0714** [−0.0881, −0.0543] ✗ (macro 0.7576 → 0.6863).

Printed: A3 share bins <30 −0.108 · 30–50 −0.066 · 50–70 +0.029 (∋0) · >70 +0.075 [+0.003, +0.149]; tail clips
(277/569) **−0.125**. A0/A1 Δ all ≈ +0.001 with Δ long +0.10/+0.11; A2 −0.115, A3 −0.071. `f_2` negative for every
arm (−0.016 … −0.037). `ema` no better (A3 Δ all −0.074, `f_2` −0.033).

**Reading (exploratory, not gated).** (1) Half the clips carry ≥ 3 s of normal after the span: the §6.3 "fade" is
partly the model correctly tracking the span's end, which a hold cannot tell apart from forgetting. (2) The hold costs
most where the curve is sharpest (A2/A3, the video-stream arms) and is neutral on A0/A1: it trades A3's temporal
precision for coverage. (3) Under position control the hold always hurts — the late mass it adds is what `p_T2`
already holds. So the share-bin slope that motivated H2 (A3 − `p_T2` −0.001 → −0.196) is largely position: at >70 %
the few normal frames sit at the clip start. **H2(b) loses its motivation; not proposed.** Do not re-run with
another `w` (lesson 14). Next lever on this question: H1.
