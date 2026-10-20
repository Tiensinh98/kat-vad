# KAT-VAD v4 (ReSH) — Stage A failed: three ways forward

**Prepared for:** advisor review. **Date:** 2026-10-18. **Code:** branch `v4` (from `v2@5469bfd`), commit `0fbd0e5`.
**Author:** sinhpham. Rules: `core/docs/v4/PREREG.md` (frozen 2026-10-15, before any v4 number). Read-out:
`core/docs/v4/head_selection.md`. Design: `core/docs/v4/KAT VAD v4 architecture proposal.md` rev 3.

## 1. What v4 is, in one paragraph

v2 closed on **A3-ign** (frozen CLIP + CRN + frozen VideoMAE V2-S stream, `dvs_anchor_mode=ignore`; DoTA-CAP-eval
macro 0.8120, still below the no-pixel position prior `p_T2` 0.8256). v4 (ReSH) proposed to add **5 of V2-S's 72
attention heads**, picked on T2-train from weak labels, and to steer them with each clip's CRN reference. The plan put
the cheap kill points first: Stage A (head selection + probe gate G1a, no training) → Stage B (transfer probe G1b +
DF8 position probe G1c, no training) → only then 10–15 training runs.

## 2. Result: Stage A fails the stability floor

| | value | rule |
|---|---|---|
| Selected heads (detrended, S9) | [14, 17, 19, 22, 23] (blocks 2–3) | top-5 by RSS |
| **Mean fold Jaccard (S10)** | **0.180** | floor **≥ 0.25**; chance 0.0397 |
| Verdict | **S10 FAIL → G1a FAIL by construction → V4 stops at Stage A (G1a.5)** | — |

1,272 T2-train sources, 5 folds, seed 2024; 10,892 normal / 33,324 abnormal labelled steps. Per-fold top-5:

| fold | detrended top-5 | raw top-5 |
|---|---|---|
| 0 | [4, 11, 17, 19, 20] | [9, 14, 17, 21, 22] |
| 1 | [1, 10, 14, 17, 22] | [14, 17, 22, 23, 28] |
| 2 | [14, 17, 19, 21, 22] | [14, 17, 20, 23, 28] |
| 3 | [14, 15, 19, 22, 23] | [9, 10, 17, 20, 22] |
| 4 | [4, 10, 13, 33, 70] | [14, 16, 21, 22, 23] |

**Sentence the thesis may use:** *"No stable head selection is detectable on V2-S from weak labels after detrending."*
Not "heads carry no signal": the selection is 4.5× chance, it is just not stable enough to bet training runs on.

### Numbers printed for context only (they cannot change the verdict)

- Raw (not detrended) selection: fold Jaccard **0.349**, top-5 [14, 17, 21, 22, 23], shares 4 of 5 heads with the
  detrended set, Spearman ρ of the two rankings 0.816.
- Fold 4 is the outlier (it shares 0 heads with folds 2 and 3). The mean over folds 0–3 alone is **0.263** (computed
  from the table above for this brief).

I will **not** use either number to rescue Stage A. Detrending was pre-registered *because* raw window labels select
position and aftermath heads (review B2); switching to raw after it scored higher, or dropping a fold after seeing it,
is fitting the gate (lesson 14). Both numbers say the same thing: the result is near the floor and fragile, and that
is exactly when a pre-registered floor should hold.

## 3. Three options

The PREREG (§3, last paragraph) leaves one decision to you: **if G1 fails, whether the 5 fallback runs go to A4-S, go to
SC-2, or nowhere.** Each option would be written as a numbered Amendment before any run.

| | **(1) Close V4** | **(2) A4-S fallback** | **(3) SC-2: A3-B** |
|---|---|---|---|
| What | Report Stage A as a bounded negative; v4 ends | ReSH's controller on V2's existing stream: `W_u((c·ũ⊘σ_u) ⊙ a_u(r̂))`, one 384-d direction, 2,753 params. Does **not** use the head selection | A3-ign with V2-B instead of V2-S (diff = `{v2.motion, model.motion_dim}`); rule = v2 addendum §25 (Y1–Y11) |
| Question it answers | — | "Does clip-reference steering of the video stream help?" | "Does the larger encoder help as a trained arm?" |
| Prior evidence | — | **None.** No probe has ever tested steering | E2(d): V2-B ahead of V2-S beyond position in transfer (+0.037 vs +0.018); V2-S was picked by the **cost** tie rule. Expected Δ ≈ 0 to +0.02 (judgement) vs a +0.01 bar |
| GPU | 0 | CPU re-bake with `r̂` columns + **5 runs** + DF8 forward | **Largest job of the project:** V2-B on all T2 sources (1–2 Colab sessions) + **5 runs** + diag |
| Code | done | **Not built** (ReSH module, re-bake columns, DF8 per-crop `r̂`) — ~1–2 days | **Built and committed** on `v2` (`5469bfd`), never run |
| Main risk | none | DF8 position cost (X7). The gate reads the reference, the channel DF8 suspects (review B3); every adopted v2 step cost ≈ 0.04 on X7 (A3 − A0 −0.041, A3-ign − A3 −0.041), so a pass on CAP-dev may still fail X7 | Bar is strict (t95 **and** cluster lower > 0, mean ≥ +0.01); likely null. §25 runs DF8 only if adopted and has **no X7 bar** |
| Gap in the rules | — | PREREG says "one contrast vs A3-ign under §8.1" but does not say whether D3 (X7 non-inferiority) binds. The Amendment must fix it | Whether SC-2, now run as v4's fallback, inherits D3. The Amendment must fix it |

## 4. What I recommend

**(1) Close V4**, and if the budget allows one more GPU job, spend it on **(3) SC-2**, not on A4-S.

- A4-S tests a new mechanism with no prior evidence, its code does not exist, and its most likely failure (X7) is
  the one every adopted v2 step already fails. It is the weakest bet for 5 runs.
- SC-2 is already built, answers a cleaner question (one input swap), and has the only positive prior of the two.
  It is still likely to be null, and it is the most expensive run left.
- The thesis does not need either. v2 is closed with Final read-outs; v2.2 (order test) and v4 Stage A are two
  reported, pre-registered negatives.

If SC-2 is run, I propose the Amendment adds D3 (X7 lower bounds > −0.01 vs A3-ign) to §25, so that a V2-B adoption
carries the same position guardrail v4 would have had.

## 5. Questions for you

1. Close V4 now: yes / no?
2. If one more GPU job is allowed: SC-2, A4-S, or neither?
3. If SC-2 (or A4-S) runs: should D3 (the DF8 X7 non-inferiority bar) bind? I suggest yes.
4. G1a's printed-only probe numbers (e.g. `[ẽ; p]` vs `[ũ; p]` on T2-val) can still be produced after
   `head_selection.json` is committed. Do you want them before deciding, or is the S10 verdict enough?
5. Chapter scope: is "Stage A of a pre-registered head-selection extension failed its stability floor" acceptable as
   a short section, or should v4 go to an appendix?
