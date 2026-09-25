# KAT-VAD — D2City as the normal-bag pool for DADA-2000 original (feasibility EDA)

**Branch:** `main` (KAT-VAD v1). **Status:** **CLOSED — NO-GO (measured 2026-09-25).** G-X FAIL
(Δ(X−R0) = −0.090, t95 [−0.101, −0.079]), G-M fail (−0.0123), shortcut AUC 1.000. T2 stays.
Results: Appendix A · `core/docs/D2CITY_EDA.md` · lesson C38.
**Author:** sinhpham · **Opened:** 2026-09-24
**Runbook:** `colab/D2City/eda_normal_bags.ipynb`
**Parent:** `.project/plans/katvad-dada-original-corpus.md` (T2). This plan does **not**
replace T2; it asks whether a *second* construction is admissible.

---

## 0. One paragraph

Proposed corpus: **positive bags = full-length DADA-2000 original accident videos**
(no W=20 windowing), **negative bags = D2City dashcam clips** (Didi, China; the
`training-video` parts in `data/D2City`). The user's prior — same country, dashcam,
similar fps — is necessary and not sufficient: `0_Normal_Driving` was also
same-country dashcam at the same fps and it still leaked the label through length
(C28/C32) and set the clip oracle to 0.9766. **The question this EDA answers is not
"do the two look alike" but "do D2City negatives teach a linear reader of the frozen
features the *accident*, or the *source*?"** Everything is measured on the frozen
CLIP ViT-B/16 features the model actually consumes. No `core/` change; no training.

---

## 1. Summary

- Measure D2City's container facts, pick the stride that matches DADA's sampled
  time step, and extract D2City CLIP features (two field-of-view arms).
- Quantify the three known ways a separate negative pool breaks a MIL corpus:
  **length leak** (C28), **clip oracle** (C12/C33), **source shortcut** (new here).
- Run one decisive gate — **G-X**: a probe trained with D2City as its *only* negatives,
  scored on *within-DADA-video* localization — against the in-video-negatives
  reference R0 (≈ Gate D0, 0.6518).
- Output: GO / MARGINAL / NO-GO, the length-matching recipe and the transform arm,
  written to `core/docs/D2CITY_EDA.md` after the run.

## 2. Assumptions & Constraints

| # | Assumption / constraint | Status |
|---|---|---|
| A1 | DADA-2000 is **30 fps**: 658,476 frames / 6.1 h (paper) = 29.98 fps; consistent with the xlsx median of 322 frames ≈ 10.7 s. The release ships PNGs, so it **cannot be measured here** | literature, unverified |
| A2 | D2City: **25.0 fps, 29.3–30.0 s (727–751 frames), 63 % 1920×1080 / 37 % 1280×720**, H.264 mp4 | **measured** on `0001/` (100 clips, local) |
| A3 | Local D2City = **7 zip parts × 100 clips = 700 clips, 8.4 GB**, plus 700 CVAT-style XML box tracks (12 classes) | measured |
| A4 | DADA original = **1584×660 (2.40:1)**, cache `cache/clip/DADA2000_orig/`, per **source** clip, stride 8, `no_center_crop` | measured (DADA_ORIGIN_PHASE0 §3.3, phase 2) |
| A5 | DoTA `labels_s8` + `DoTA_s8_ncc` on Drive — used as the cross-corpus **reference**, never trained on | exists (phase 5 paths) |
| C1 | **C2/C13:** the DADA cache and transform are frozen. Only D2City's side may be transformed | hard |
| C2 | `LaGoVAD-PreVAD/` read-only; no `core/` edit for an EDA (precedent: `build_d0_dataset.py`) | hard |
| C3 | DoTA stays held out (CLAUDE.md §14.5.1) | hard |

## 3. Plan Metadata

- **Plan type:** Data feasibility EDA (pre-registered gates, no training)
- **Size / scope:** Small–Medium (one notebook, ~1.5–2.5 h Colab GPU; upload is the long pole)
- **Estimated duration:** 1–2 days incl. upload and write-up
- **Storage path:** `@.project/plans/katvad-d2city-normal-bag-eda.md`

## 4. Debate — why feature probes, not eyeballs or a pilot training

| option | answers "compatible as normal bag"? | cost | verdict |
|---|---|---|---|
| **A. Descriptive only** (fps, resolution, length, montage) | **No.** `0_Normal_Driving` passed all of these and still failed (C28, C32) | minutes | necessary, kept as Phase 1–2 |
| **B. Frozen-feature probes** (source probe + cross-source transfer probe) | **Yes, at the representation the model reads**; a linear probe is the same ceiling Gate D0 used | ~1 h GPU extraction + ~30 min CPU | **chosen** |
| C. Pilot training on a mixed corpus | most faithful | full corpus build + RAFT for D2City + 3 seeds; and on TAD the model found the clip shortcut the probe predicts | premature — only after B passes |

**Risk of B:** a linear probe is not the model. A PASS licenses a training pilot, it is
not a result. A FAIL is stronger: if a linear reader learns source instead of accident,
a MIL model with top-k pooling has an *easier* shortcut still (TAD, C14).

## 5. Phases Overview

| phase | goal | est. | depends on |
|---|---|---|---|
| P0 Stage | D2City zips + XML on Drive, repo synced | user, 1–3 h upload | — |
| P1 Inventory (**G-I**) | every clip decodes; fps/res/length census; XML scene stats; montage | 10 min | P0 |
| P2 Temporal match | stride choice from A1/A2, checked against feature autocorrelation in seconds | derived | P1 |
| P3 Extraction | D2City CLIP features, stride 7, arms **V0** (full frame) / **V1** (2.40:1 band) | ~60–90 min T4 | P2 |
| P4 Protocol (**G-L**) | length-leak + clip-oracle **lever table** (raw / fixed / DADA-matched segments) | CPU, seconds | P3 |
| P5 Probes (**G-S**, **G-X**, **G-M**) | source separability; the decisive transfer gate | CPU, ~30 min | P3, P4 |
| P6 Verdict | JSON + md to Drive; `core/docs/D2CITY_EDA.md`; memory bank | 0.5 day | P5 |

## 6. Detailed Tasks by Phase

### P0 — Stage (user)
- [ ] Upload `data/D2City/training-video/000{1..7}.zip` and `training-annotation/` to
      `Drive/Thesis/data/D2City/` (same layout as local).
- [ ] Sync `core/` to `Drive/Thesis/kat-vad` (no new `core/` code is needed).
- **Deliverable:** notebook §0 prints every path `OK`.

### P1 — Inventory (Gate **G-I**, HARD)
- [ ] Unzip to VM-local disk; count **files**, not folders (C10). Probe each mp4 with PyAV:
      fps, W×H, frame count, duration, decodable.
- [ ] Bars: decodable ≥ **99 %**; fps == 25 on ≥ **99 %**; duration in [29, 31] s on ≥ 95 %.
      A clip failing decode is dropped **with a logged count**, never silently.
- [ ] XML (descriptive): objects/frame, share of `person`/two-wheelers/tricycles — the
      scene-density profile. DADA has no boxes, so this describes D2City only.
- [ ] Montage (human check): 8 D2City frames under V0 and V1 next to 8 DADA frames at the
      224×224 the encoder sees — look for timestamps, logos, hood, letterboxing.
- **Deliverables:** `inventory.csv`, `xml_stats.json`, `montage.png`.

### P2 — Temporal match (derived, no gate)
- DADA stride 8 @ 30 fps = **0.267 s**. D2City @ 25 fps: stride 6 = 0.240 s (−10 %),
  **stride 7 = 0.280 s (+5 %)**, stride 8 = 0.320 s (+20 %). **Choose 7.**
- [ ] Check A1 empirically: plot lag-k feature cosine vs **seconds** for DADA pre-accident
      frames and D2City. Curves that overlap within the spread say the time scales agree;
      a D2City curve far above (slower change) is a "static-ness" shortcut and is flagged.
- **Deliverable:** `autocorr_seconds.png` + numbers in the JSON.

### P3 — Extraction (two arms, one decode)
- [ ] Decode once per clip at stride 7 (`core.data.video_io.read_sampled_frames`), then:
  - **V0** — full frame → `preprocess_frames(center_crop=False)`: the project's `_ncc`
    transform. 16:9 is squashed 1.78× into 224², DADA's 2.40:1 is squashed 2.40×, so
    object geometry differs between sources.
  - **V1** — centre band at 2.40:1 (1920×800 / 1280×533) → same transform. Same squash
    as DADA; loses sky/hood rows.
- [ ] Atomic, resumable writes (`feature_cache.save_array` / `is_complete`, C11) straight
      to `cache/clip/D2City_s7_ncc/` and `cache/clip/D2City_s7_ncc_ar240/`. The two caches
      never share a directory (C2).
- **Deliverable:** 2 × ~700 `.npy`, `(L≈107, 512)`, plus an extraction manifest (C17).

### P4 — Protocol (Gate **G-L**, HARD on the chosen recipe)
Lever table (C35 — the lever is measured, not named):

| recipe | D2City bag | predicted length AUC | purpose |
|---|---|---|---|
| L-raw | whole 30 s clip (~107 sampled) | ≈ 1.0 (DADA median ≈ 40) | shows the leak exists |
| L-fixed | non-overlapping segments of DADA's median length | length AUC ≈ 0.5 only if DADA's spread is small | simple alternative |
| **L-match** | non-overlapping segments: draw lengths i.i.d. from DADA's empirical distribution **up front** (80 % of D2City capacity), then first-fit pack them into random clips with room | ≈ 0.5 by construction | candidate recipe |

- [ ] For each: `core.eda.protocol.clip_length_leak` and `clip_constant_oracle` on the
      label lists a test set would have (DADA frame labels + all-zero D2City bags).
- [ ] **G-L bar:** chosen recipe's clip-level length AUC in **[0.45, 0.55]**.
- [ ] Clip oracle is **descriptive** (its value is set by the test mix, C33:
      `(F_norm + 0.5X)/(F_norm + X)`); print it for D2City:DADA test ratios 0, 0.5, 1, 2.
- **Why pack, not draw-per-clip:** drawing per clip until a draw no longer fits rejects long
  draws more often and biases bags short — the dry run measured length AUC **0.578 'shorter'**
  that way vs **0.526** packed. The notebook reports `drawn/placed/unplaced` and
  `dada_longer_than_d2_clip` (DADA videos longer than any 30 s D2City clip can never be matched —
  a residual leak the corpus build must trim or exclude).
- **Bonus fact:** 700 clips × ~2.7 segments ≈ 1,900 bags ≈ DADA's ~1,900 videos → ~1:1.

### P5 — Probes (all grouped by **source clip**, 5 folds, standardized logistic
regression with `constants.EDA_PROBE_*`, same folds for every probe so deltas are paired)

| id | train positives | train negatives | scored on | role |
|---|---|---|---|---|
| **S** | DADA pre-accident frames | D2City frames | held-out fold, pooled AUC | source separability |
| S-ref | DADA pre-accident frames | DoTA pre-anomaly frames | same | "any other dashcam corpus" reference |
| **R0** | DADA in-span frames | DADA out-of-span frames (in-video) | held-out DADA videos, `auc_macro` | reference ≈ Gate D0 |
| **X** | DADA in-span frames | **D2City only** | held-out DADA videos, `auc_macro` | **the gate** |
| **M** | DADA in-span frames | DADA out-of-span + D2City | held-out DADA videos, `auc_macro` | does adding D2City hurt? |

Plus, for X and M: **shortcut AUC** = P(score(DADA normal frame) > score(D2City frame))
on held-out folds. ≈ 0.5 = the probe does not separate the two negative pools; → 1.0 =
D2City bags are trivially "more normal", i.e. a bag-level shortcut MIL can exploit.

**Pre-registered bars (C33 derivations inline):**

| gate | bar | derivation / reading |
|---|---|---|
| sanity | R0 `auc_macro` in **[0.62, 0.68]** | Gate D0 measured 0.6518 at n=400 on the same cache and labels; outside the band = pipeline error, **stop and debug**, do not read X |
| **G-X PASS** | X ≥ **0.60** **and** paired Δ(X − R0) ≥ **−0.03** | 0.60 is Gate D0's own bar; −0.03 ≈ the fold spread D0-class probes show — a larger loss means D2City negatives teach less accident than the video's own normal frames |
| G-X MARGINAL | otherwise (0.55 ≤ X < 0.60, or Δ in [−0.06, −0.03)) | pilot training only with in-video negatives kept (i.e. M, not X) |
| **G-X FAIL** | X < **0.55**, or Δ(X − R0) < **−0.06** | D2City teaches source, not accident → **NO-GO as negative pool** |
| **G-M** | paired Δ(M − R0) ≥ **−0.01** (t95 over 5 folds) | adding D2City must not *cost* localization; this decides "D2City **in addition to** in-video negatives" |
| G-S | not an admission gate; **claim gate** | read S **against S-ref**; S ≤ S-ref + 0.02 = "no more foreign than DoTA". Absolute S near 1.0 is expected (dataset bias) and is not by itself a NO-GO — G-X decides admission. **But** `systemPatterns.md` (2026-09-15) pre-registered *"source probe ≥ 0.90 AUC ⇒ no in-domain claim"* for any cross-dataset normal pool, and that rule stands: if S ≥ 0.90, a corpus built on D2City may be trained on, but no result from it is reported as in-domain DADA |
| shortcut | descriptive; red flag > **0.90** | no attainable-range derivation exists before the data — reported, not gated |

Each probe runs for **V0 and V1**. **Arm rule:** prefer V1 only if it raises X by ≥ 0.02
or lowers S by ≥ 0.02 with X not lower; otherwise V0 (no extra transform — KISS, and
flow extraction for D2City would have to replicate the crop, C13).

### P6 — Verdict & write-up
- [ ] Notebook writes `outputs/D2City_eda/{eda_d2city.json, eda_d2city.md, *.png}`.
- [ ] Decision table:

| G-I | G-L | sanity | G-X | G-M | verdict | next |
|---|---|---|---|---|---|---|
| pass | pass | pass | PASS | pass | **GO** | corpus-build plan: DADA full videos + D2City L-match bags; RAFT for D2City; re-derive `score_head_kernel`/`mil_topk_pct` for full-length bags (C27); new z-score cache (train split changes) |
| pass | pass | pass | MARGINAL | pass | **GO-with-in-video-negatives** | build M-style corpus only |
| pass | pass | pass | any | fail | **NO-GO (additive)** | stay on T2 |
| pass | pass | pass | FAIL | — | **NO-GO** | stay on T2; record as the third separate-pool failure |
| fail | — | — | — | — | fix ingest first | — |
| — | — | fail | — | — | pipeline bug | debug labels/cache alignment |

- [ ] `core/docs/D2CITY_EDA.md` (results), memory bank `activeContext.md` + `progress.md`,
      lesson candidate if something new broke (Section 7 of CLAUDE.md).

## 7. Risks & Mitigations

| risk | impact | mitigation |
|---|---|---|
| A1 wrong (DADA not 30 fps) | stride 7 mismatched in time | P2 autocorrelation-in-seconds check; stride is a cache key, so a wrong call costs one re-extraction (~1 h), no training |
| Source shortcut (aspect, ISP, compression, overlays) | model classifies bags by camera — TAD-style collapse | S vs S-ref, shortcut AUC, V1 arm, montage |
| "Boring normal": D2City lacks pre-accident risk context | probe X scores the *risky scene*, not the accident moment → within-video macro drops | that is exactly what G-X/Δ(X−R0) measures; M keeps in-video negatives |
| Length leak (30 s vs ~11 s) | C28 | L-match; G-L is HARD |
| Micro AUC inflated by all-normal D2City test bags | C12 mirror | oracle printed for every test mix; `auc_macro` stays the headline |
| Hidden anomalies inside D2City | label noise in negatives | montage + later spot check; D2City is curated normal driving, accept residual |
| Probe ≠ model | false GO | a GO only licenses a pilot; bars for the pilot are written in the next plan, not here |
| KIP needs `e_O` for D2City | adoption cost | out of EDA scope; RAFT on ~700 clips must use the chosen arm's field of view (C13) |
| Colab disk/RAM | 8.4 GB zips + unzipped ≈ 17 GB; probe rows ≈ 150k × 512 | VM-local NVMe; float32 features; unzip one part at a time if disk is short |
| Scene/geography differences (city mix, night share) | shift beyond camera | descriptive only here (XML density); not a gate |

## 8. Next Steps for the User

1. Upload the 7 zips + `training-annotation/` to `Drive/Thesis/data/D2City/` and sync `core/`.
2. Run `colab/D2City/eda_normal_bags.ipynb` top to bottom on a **GPU** runtime (T4 is enough).
3. Bring back `outputs/D2City_eda/eda_d2city.{json,md}` + `montage.png` + `autocorr_seconds.png`.
4. If you have the DADA-2000 source mp4s anywhere, confirm A1 (fps) — one `ffprobe` settles it.
5. We read the verdict against §6 P6 — **before** any corpus-build code.

---

## Appendix A — Results (measured 2026-09-25; write-up `core/docs/D2CITY_EDA.md`)

Raw: `outputs/EDA/D2City/`. All mechanics gates pass, so the probe rows are readable.

| quantity | V0 | V1 |
|---|---:|---:|
| G-I decodable / fps25 / duration | 1.000 / 1.000 / 0.996 — PASS | (same decode) |
| G-L length AUC (L-match) | **0.5009** — PASS (1,391 drawn = placed; L-raw 0.9949) | (same segments) |
| S / S-ref | 1.0000 / 0.9999 — both at ceiling, uninformative | 1.0000 / 0.9999 |
| R0 `auc_macro` | **0.6763** — sane | 0.6763 |
| X `auc_macro`, Δ(X−R0) t95 | **0.5864, −0.0899 [−0.1011, −0.0787]**, 5/5 folds < −0.06 | 0.5767, −0.0996 [−0.1121, −0.0870] |
| M `auc_macro`, Δ(M−R0) t95 | 0.6639, **−0.0123 [−0.0155, −0.0091]**, 5/5 folds < 0 | 0.6617, −0.0146 [−0.0189, −0.0102] |
| shortcut AUC (R0 / X / M) | 0.820 / **1.000** / **0.999** | 0.757 / 1.000 / 0.999 |
| verdict | **G-X FAIL, G-M fail → NO-GO** | FAIL / fail |

**Arm on record: V0** (V1 lowers X by 0.010, S unchanged). **Row taken: NO-GO — stay on
T2; third separate-pool failure** (lesson C38). No hood/logo-mask arm was added after
the read-out (lesson 14). The montage's DADA row was empty (frames not on Drive), so the
visual check against DADA was not performed. A1 is still unsettled: D2City's per-second
autocorrelation sits above DADA's at every lag.
