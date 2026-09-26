# BDD-A as a negative-bag pool for T2 (DADA-2000 original) — EDA results (branch `main`)

**Measured and read out 2026-09-26.** Plan: `.project/plans/katvad-bdda-normal-bag-eda.md`
(gates pre-registered in §4 before the run). Runbook: `colab/BDDA/eda_normal_bags.ipynb`.
Raw artefacts: `outputs/EDA/BDDA/{eda_bdda.json, eda_bdda.md, probes.json, inventory.csv,
montage.png, autocorr_seconds.png}`. Template and bars: `core/docs/D2CITY_EDA.md`.

**Verdict: NO-GO.** Used as the *only* negatives, BDD-A clips fail G-X by a wider margin
than D2City did (Δ −0.127 against −0.090). Used as *extra* negatives they pass G-M, but only
because they cost nothing measurable. They add nothing either, and they still hand the
probe a source shortcut (0.999). T2 stays the training corpus. This is the project's
**fourth** failure of a normal pool taken from a different source, after `0_Normal_Driving`
(C28/C32), TAD (C14) and D2City. Lesson **C38** (extended).

---

## 0. The question

Can BDD-A clips be **added to T2 as negative bags**? T2 is the set of W=20 hop 8 windows
cut from the DADA-2000 original accident videos, and its negatives come from inside those
same videos. The question is the one C38 asks. It is not whether the two sources look
alike. It is **whether BDD-A negatives teach a linear reader of frozen CLIP the accident,
or the source.**

**BDD-A = Berkeley DeepDrive *Attention*** (Xia et al., 2018), **not** BDD100K: US
footage from San Francisco, collected around **braking events**. That makes it a pool
enriched in near-misses, not in ordinary driving. The gate is therefore read on the
**`calm`** arm: clips with known GPS whose minimum acceleration is ≥ −3.0 m/s². The `all`
arm is reported for description only.

The protocol is identical to the D2City EDA: frozen-feature logistic probes
(standardized, `C = 1.0`, balanced classes), 5 folds grouped by source clip, the **same
DADA folds and seed** as the D2City run (so R0 must reproduce), paired deltas, and t95
with 4 df. No model was trained, and nothing in `core/` changed.

## 1. Mechanics gates: all pass, so the probe numbers can be read

| gate | measured | bar | |
|---|---|---|---|
| **G-I** inventory | 925/926 decodable (`922.mp4` has no moov atom) · 925 step-matched (per-clip stride 8 @ 30 fps / 16 @ 60 fps) · 925 ≥ one T2 window · GPS-known 88.7 % · **calm = 508** | ≥ 99 / 99 / 99 / 80 % · calm ≥ 300 | PASS |
| **G-L** length leak (L-T2) | clip-level length AUC at mix 1.0 **0.5000** (BDD-A cut into 1,588 W=20 hop 8 windows; 0 clips shorter than W) | [0.45, 0.55] | PASS |
| R0 sanity | `auc_macro` **0.676227** over 1,860 DADA videos | [0.62, 0.68] | PASS |
| R0 reproduces D2City | 0.676227 vs 0.676258 (\|Δ\| 3.1e-5) | < 1e-4 | PASS |
| T2 mix-0 oracle | 0.7037 | ≈ 0.7037 | PASS |
| DADA labels | 1,861 sources · 0 missing features · 0 row mismatches · 1,106 T2 test windows, 0 label mismatches | ≤ 1 % | clean |

Inventory: 1280×720 on all clips. fps 89 % ≈ 30 (29.97 / 30), 11 % ≈ 60, and one clip at 120.
Durations run p5–p75 = 10.0 s, p95 14 s, max 32 s, so the sampled length has p50 38 and p95 53
(DADA sources: p50 42). GPS min acceleration: p50 −2.59 m/s², p5 −5.49. Mean speed p50
5.5 m/s (urban and slow).

**Length lever and oracle cost.** L-T2 closes the length leak by construction. It
still **raises the T2 clip oracle**, because every BDD-A window is all-normal and C33's
frame-share formula applies:

| mix (BDD-A bags per T2 test window) | 0 | 0.25 | 0.5 | 1.0 |
|---|---:|---:|---:|---:|
| L-T2 clip oracle | 0.7037 | 0.7842 | 0.8305 | **0.8813** |
| L-raw length AUC (contrast) | — | 0.578 | 0.575 | 0.575 |

Even if BDD-A had passed, adding it at mix 1.0 would push the T2 micro AUC back toward
clip classification (0.70 → 0.88).

## 2. The probes

| quantity | all | **calm (gate)** | bar |
|---|---:|---:|---|
| R0 `auc_macro` (in-video negatives) | 0.6762 | 0.6762 | reference |
| **X** `auc_macro` (BDD-A-only negatives) | 0.5375 | **0.5492** | ≥ 0.60 (FAIL < 0.55) |
| **Δ(X − R0)**, t95 | −0.1387 [−0.1535, −0.1240] | **−0.1270 [−0.1532, −0.1008]** | ≥ −0.03 (FAIL < −0.06) |
| M `auc_macro` (in-video + BDD-A) | 0.6714 | 0.6733 | — |
| **Δ(M − R0)**, t95 | −0.0048 [−0.0089, −0.0007] | **−0.0029 [−0.0062, +0.0004]** | ≥ −0.01 |
| shortcut AUC R0 / X / M | 0.374 / 1.000 / 0.9995 | 0.374 / **1.000** / **0.999** | red flag > 0.90 |
| S (DADA pre-accident vs BDD-A) | 1.0000 | 1.0000 | claim gate |
| S-ref (DADA pre-accident vs DoTA) | 0.9999 | 0.9999 | — |
| **G-X / G-M** | FAIL / pass | **FAIL / pass** | |

Per-fold Δ(X − R0) on calm: −0.134, −0.139, −0.117, −0.150, −0.096. **All five folds are
beyond the −0.06 FAIL bar.** X also fails the absolute bar on its own (0.5492 < 0.55). Per-fold
Δ(M − R0) on calm: −0.0036, −0.0053, +0.0001, −0.0055, −0.0002. G-M passes, but only
because it is ≈ 0: **no fold shows BDD-A helping by more than 1e-4.**

(Shortcut AUC = P(a DADA normal frame scores above a BDD-A frame) under the probe's
anomaly score. R0's value is computed once on the BDD-A test clips and is shared by both
arms.)

## 3. Reading

1. **BDD-A negatives teach the source, not the accident, and they do it worse than
   D2City.** Swapping the in-video normal frames for BDD-A frames costs **12.7 AUC points**
   of within-video localization on the calm arm (D2City: 9.0). The margin is consistent
   across folds.
2. **The failure mode differs from D2City's at the start and is the same at the end.**
   R0, which never saw BDD-A, puts BDD-A frames **above** DADA's own normal frames in
   **63 %** of pairs (shortcut 0.374; D2City was the opposite, 0.820). Read literally, the
   in-video reader already finds braking-event footage *more* accident-like than a DADA
   pre-accident frame, which fits a pool built around hard braking. Once BDD-A is used as
   the negative class, the probe flips it to shortcut **1.000**: it separates the sources
   perfectly and loses 0.13 of localization doing so. So the start point does not
   matter, and a foreign pool that is separable ends in the same place.
3. **Filtering out hard braking barely helps.** `calm` beats `all` by +0.012 on X.
   Near-miss clips account for a small share of the gap. The source accounts for the rest.
4. **S against S-ref again carries no information**: 1.0000 vs 0.9999. On frozen CLIP
   every foreign dashcam corpus we have tried (DoTA, D2City, BDD-A) separates completely
   from DADA. No band-crop arm was run, by design (plan D2; it moved nothing on D2City).
5. **Montage.** San Francisco streets, the ego hood visible, **no overlay** (no timestamp or
   logo, unlike D2City). Frames from hard-braking clips show crosswalks, red lights and
   pedestrians. The shortcut therefore does not need a watermark: camera, ISP, geography
   and hood are enough.
6. **Time scale.** Per-second feature autocorrelation, on the same 0.267 s step: BDD-A
   (all and calm alike) sits **above** DADA pre-accident at every lag (lag 1: 0.977 vs
   0.967; lag 8 ≈ 2.1 s: 0.941 vs 0.936). That is the same ordering and roughly the same
   gap as D2City (0.975). One more source cue. A1 (DADA = 30 fps) is still unsettled.

The caveat, as for D2City: these are linear probes on frozen features, not a trained
model. A temporal encoder trained with MIL has more capacity to exploit a bag-level
shortcut, not less.

## 4. Decision (plan §4 verdict table)

| G-I | G-L | sanity | G-X (calm) | G-M (calm) | row |
|---|---|---|---|---|---|
| pass | pass | pass | **FAIL** | pass (≈ 0) | **NO-GO**: stay on T2; fourth separate-pool failure |

- G-X FAIL alone decides NO-GO. The verdict table reaches the "M-style" route only on a
  G-X **MARGINAL**, and the M arm would add no measured gain while its bags carry a 0.999
  shortcut and a +0.18 oracle rise at mix 1.0 (§1).
- **T2** stays the training corpus. **Option A** (`flow/v2_zscore`, phase 5) remains the
  live track.
- **Do not download more BDD data** (the BDD-A validation split, or `bdd100k_videos.zip`
  at 1.8 TB). The plan made further downloads conditional on a pass.
- `cache/clip/BDDA*` on Drive can be kept or deleted. Nothing depends on it.

## 5. Reproduction notes

- `probes.json` is resumable per probe. R0 is fitted once and shared by both GPS arms.
  It reproduced D2City's R0 to 3.1e-5, so the DADA side of both EDAs is the same
  pipeline.
- Per-clip stride = `round(fps × 8/30)`, with a step tolerance of 5 %. The CLIP model is
  `openai/clip-vit-base-patch16@5ef227a`, `no_center_crop`, 224. That matches the `_ncc`
  cache contract (C2).
- The notebook skips macOS `__MACOSX/` and `._*` archive members. Validation-split ids
  would be split-prefixed (C26), but the split was not used.
