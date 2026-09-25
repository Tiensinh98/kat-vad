# D2City as a normal-bag pool for DADA-2000 original — EDA results (branch `main`)

**Measured 2026-09-25, read out the same day.** Plan:
`.project/plans/katvad-d2city-normal-bag-eda.md` (gates pre-registered in §6 P5/P6
before the run). Runbook: `colab/D2City/eda_normal_bags.ipynb`. Raw artefacts:
`outputs/EDA/D2City/{eda_d2city.json, eda_d2city.md, probe_V0.json, probe_V1.json,
inventory.csv, montage.png, autocorr_seconds.png}` — `outputs/` is gitignored, so
**this file is the durable record**.

**Verdict: NO-GO.** D2City clips fail as the *only* negatives (G-X FAIL) and also fail
as *extra* negatives next to the in-video ones (G-M fail). T2 stays the training corpus.
This is the project's **third** failure of a normal pool taken from a different source,
after `0_Normal_Driving` (C28/C32) and TAD's collapse (C14). Lesson **C38**.

---

## 0. The question

The proposed corpus was: positive bags = **full-length DADA-2000 original accident
videos**, negative bags = **D2City** dashcam clips (Didi, China; 25 fps, ~30 s). The
reasons given for it — same country, dashcam, similar fps — are necessary but not
sufficient. `0_Normal_Driving` matched on all three and still leaked. So the EDA did not
ask *how similar the two sources look*. It asked **whether D2City negatives teach a
linear reader of the frozen CLIP features the accident, or the source.**

Everything below uses frozen-feature logistic probes (standardized, `C = 1.0`, balanced
classes, 5 folds grouped by **source clip**, the same folds for every probe so deltas
are paired, t95 with 4 df). No model was trained, and nothing in `core/` changed.

## 1. Mechanics gates — all pass, so the probe numbers can be read

| gate | measured | bar | |
|---|---|---|---|
| **G-I** inventory | 700/700 decodable · 700/700 at 25 fps · 99.6 % in 29–31 s · 463 × 1080p / 237 × 720p | ≥ 99 / 99 / 95 % | PASS |
| **G-L** length leak (L-match) | clip-level length AUC **0.5009** (1,391 drawn = 1,391 placed, 0 unplaced) | [0.45, 0.55] | PASS |
| R0 sanity | `auc_macro` **0.6763** over 1,860 DADA videos | [0.62, 0.68] | PASS (Gate D0: 0.6518 at n = 400) |
| DADA labels | 1,861 sources · 0 missing features · 0 row mismatches | ≤ 1 % mismatch | clean |

Length lever table (mix = D2City bags per DADA video in a test set):

| recipe | bags | length AUC | clip oracle (mix 1.0) |
|---|---:|---:|---:|
| L-raw (whole 30 s clip) | 700 | **0.9949** ("shorter") | 0.7828 |
| L-fixed (DADA median) | 1,398 | 0.5105 | 0.7538 |
| **L-match** (chosen) | 1,391 | **0.5009** | 0.7556 |

L-raw shows why length matching is required at all: whole D2City clips (≈ 106 sampled
frames) against DADA videos (median 42) give away the label from length alone. Only
0.38 % of DADA videos are longer than every D2City clip.

R0 comes out identical on V0 and V1 (0.676258). It has to, because R0 never reads
D2City, so this doubles as a free consistency check.

## 2. The probes

| quantity | V0 (full frame) | V1 (2.40:1 band) | bar |
|---|---:|---:|---|
| R0 `auc_macro` (in-video negatives) | 0.6763 | 0.6763 | reference |
| **X** `auc_macro` (D2City-only negatives) | **0.5864** | 0.5767 | ≥ 0.60 |
| **Δ(X − R0)**, t95 | **−0.0899 [−0.1011, −0.0787]** | −0.0996 [−0.1121, −0.0870] | ≥ −0.03 (FAIL < −0.06) |
| M `auc_macro` (in-video + D2City) | 0.6639 | 0.6617 | — |
| **Δ(M − R0)**, t95 | **−0.0123 [−0.0155, −0.0091]** | −0.0146 [−0.0189, −0.0102] | ≥ −0.01 |
| shortcut AUC R0 / X / M | 0.820 / **1.000** / **0.999** | 0.757 / 1.000 / 0.999 | red flag > 0.90 |
| S (DADA pre-accident vs D2City) | 1.0000 | 1.0000 | claim gate |
| S-ref (DADA pre-accident vs DoTA) | 0.9999 | 0.9999 | — |
| **G-X / G-M** | **FAIL / fail** | FAIL / fail | |

Per-fold Δ(X − R0), V0: −0.088, −0.089, −0.090, −0.104, −0.078. **All five folds are
beyond the −0.06 FAIL bar.** Per-fold Δ(M − R0): −0.014, −0.010, −0.009, −0.016,
−0.012. All five are negative. The G-M point estimate misses its bar by 0.002 and the
upper end of the interval (−0.0091) grazes it, so G-M is a narrow fail. But **no fold
shows D2City helping**.

**Arm rule** (V1 only if it raises X by ≥ 0.02, or lowers S by ≥ 0.02 without lowering
X): V1 lowers X by 0.010 and leaves S unchanged, so **V0** is the arm on record. The
aspect ratio is not what the probe reads.

## 3. Reading

1. **D2City negatives teach the source, not the accident.** Swapping the in-video
   normal frames for D2City frames costs **9 AUC points** of within-video
   localization, consistently on every fold.
2. **Shortcut ≈ 1.0 is how it fails.** The X probe ranks every DADA normal frame
   above every D2City frame, so what it learned is "DADA or D2City?". Keeping the
   in-video negatives (M) does not stop this (0.999). The strongest evidence is that
   **R0, which never saw D2City during training, already puts D2City frames below
   DADA's own normal frames in 82 % of pairs**: D2City sits far from DADA in frozen
   CLIP space before any training happens. Under MIL a D2City bag would be a free
   "obviously normal" bag. That is the same bag-level shortcut that turned TAD
   training into clip classification (C14).
3. **S versus S-ref cannot tell the sources apart.** Both are at the ceiling
   (1.0000 vs 0.9999), so "no more foreign than DoTA" is technically true but carries
   no information. What the pair actually shows: **on frozen CLIP, every other dashcam
   corpus, DoTA included, separates from DADA completely.** A crop cannot fix this.
   The pre-registered claim rule (S ≥ 0.90 ⇒ no in-domain claim) would apply anyway.
4. **Visible source markers (montage).** Most D2City frames show the ego car's hood,
   a watermark logo bottom-left, a red timestamp top-left, and on some clips a strong
   cyan colour cast. The V1 band crop keeps the hood and the logo, so V1 tested the
   aspect-ratio hypothesis only. A hood/logo mask arm was **not** run, and adding one
   after this read-out would be tuning against Δ (lesson 14). S-ref shows it would not
   close the gap either. **The montage's DADA row is empty** because DADA frames are
   not on Drive, so the side-by-side visual check was never done. It is recorded as
   not done, and it does not affect the verdict.
5. **Time scale (A1).** Plotted per second, D2City's feature autocorrelation sits
   **above** DADA's pre-accident curve at every lag (lag 1: 0.975 vs 0.967; ≈ 2.2 s:
   0.947 vs 0.936), so D2City scenes change more slowly. This EDA cannot tell whether
   the cause is an fps mismatch (A1 = 30 fps is still literature-only) or content
   (pre-accident DADA is busier). Either way it is one more source cue.

Caveat, stated once: these are **linear probes on frozen features**, not a trained
model. A temporal encoder with MIL gets more capacity to exploit a bag-level shortcut,
not less, so the caveat cannot turn this NO-GO into a GO.

## 4. Decision (plan §6 P6)

| G-I | G-L | sanity | G-X | G-M | row |
|---|---|---|---|---|---|
| pass | pass | pass | **FAIL** | fail | **NO-GO** — stay on T2; record as the third separate-pool failure |

- **T2** (negatives cut from inside the accident videos, W = 20 hop 8) stays the
  training corpus. Option A (`flow/v2_zscore`, phase 5) stays the live track.
- The D2City caches `cache/clip/D2City_s7_ncc{,_ar240}/` can be kept or deleted.
  Nothing depends on them.
- The EDA settled nothing about A1 (DADA fps). One `ffprobe` on a DADA source mp4 is
  still the way to close it.

## 5. Reproduction notes

- `eda_d2city.json` has **no `extraction` section**: the final run restarted after
  §4 crashed repeatedly (suspected RAM; pending lesson (w)), and so skipped §2. The
  extraction manifest is written to `cache/clip/D2City_s7_ncc*/extraction_manifest.json`.
  The report's `data.d2_clips = 700` confirms full coverage.
- §4 writes each arm to `outputs/D2City_eda/probe_{V0,V1}.json` and skips any arm
  already on Drive when re-run (added 2026-09-25 with the RAM fix). The standardization
  now runs in place, which is bit-identical to the old version (max |Δ| = 0.0).
