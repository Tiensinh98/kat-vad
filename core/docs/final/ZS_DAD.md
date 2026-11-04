# Zero-shot DAD: LaGoVAD (released ckpt) vs W0 (no SA²) vs R4 (SA²)

Status: built 2026-11-04; census run 2026-11-04 → FAILED (25 fps, flag ≠ accident time) → **Amendment 1 (§7) replaces §1 labels, §2 stride / V2-S step and §3 gates**; **read out 2026-11-04 (§6), PRELIMINARY; Amendment 2 (§8) says what the macro does not measure.** Rule written before any DAD number. Runbook
`colab/final/zs_dad.ipynb` (GPU for extraction). Authorized by the user on 2026-11-04 ("setup
zero-shot trên DAD … LaGoVAD public checkpoint and SA² and no SA² as above … features stored in
cache of Thesis-Final"). Companion of `ZS_TAD_DADA.md` §7 (same rows, same read-out tool).

## 1. Data

`Thesis/data/DAD/videos/testing/{positive,negative}/*.mp4` (the DAD test split, read-only) and
`Thesis/data/DAD/annotation/<stem>.txt` (frame, track id, class, x1, y1, x2, y2, flag; frames
1-based). Ids are `pos_<stem>` / `neg_<stem>`; the annotation dir is flat, so a stem present in
both folders is attributed to the positive video.

**Labels.** A positive video's frame `f` is abnormal iff any annotated object carries flag 1 at
`f`; negatives are all-normal. Rows = `native[::5]`. No other label construction is tried after a
number is seen.

## 2. Protocol (chosen before any read)

| choice | value | why |
| --- | --- | --- |
| stride | **5** at 20 fps = 0.25 s/step | the arms trained at Nexar s8 ≈ 0.267 s/step; E1's rate-matching logic (DoTA got s3 at 10 fps) |
| V2-S window | 16 frames × step **2** = 1.5 s at 20 fps | the arms' causal 1.5 s (16 × 3 at 30 fps); the cache manifest records it |
| frames | mp4 → one JPEG folder per video (q 95, VM disk); CLIP, V2-S and the SA² windows all read it | one pixel source for every feature (C2) |
| CLIP | `extract_clip_features --no-center-crop --stride 5` → `Thesis-Final/cache/clip/DAD_s5_ncc` | the project's `_ncc` transform |
| arm input | `build_v2_inputs apply` with **Nexar A2's statistics** (`Thesis-V2/cache/v2/A2_S/Nexar_s8`) → `Thesis-Final/cache/v2/A2_S/DAD_s5_nexar` | W0 / R4 read every zero-shot set this way |
| R4 patches | `clip_windows frames --grids 1 7 --cells resized224` → `stp_patches select` (raw Mo, top-12) | as TAD / DADA |
| scoring | `core.evaluate --set data.dataset=DAD`, whole video, raw scores (`--score-norm auto`: DAD has negatives) | TAD's read |
| definitions | `Nexar_window/defs.json` (the arms' traffic definitions; `DAD` → the DADA / Nexar verbalizer) | as DADA |
| rows | LaGoVAD `best.ckpt` KIP off on plain `DAD_s5_ncc`; W0 / R4 s2024 on the bake (R4 + patches) | §7 of `ZS_TAD_DADA.md` |

## 3. Census gates (before any feature is extracted; `core.tools.dad census`)

1. Every video decodes and its measured fps is 20 ± 0.5.
2. Every positive has an annotation with ≥ 1 flagged frame; no negative (own stem) carries a flag;
   no annotated frame lies past the video's end.
3. **Flag timing:** the median first flagged frame over positives is ≥ 50. If the flag marks an
   object's involvement from its first appearance (median < 50), it is not an accident time and
   the run **stops** for the user's decision — no fallback label is picked here.

Printed: counts, frame lengths, first-flag min / median / max, flagged-frame counts, whether each
flag run is contiguous, shared stems.

## 4. Integrity (before any score is read)

* The plain CLIP rows LaGoVAD reads == the first 512 columns of the A2 bake on every video
  (`zs_lagovad_readout check-inputs`).
* V2-S manifest == Nexar's V2-S manifest on every key except `stride`, `clip_frame_step`,
  `assumed_fps`; V2-S rows == CLIP rows; patch rows == bake rows (C13).
* W0 / R4: checkpoint step == its `metrics.jsonl` (J10); `stp.sa2` false / true.
* Every row covers the same two-class videos (the read-out refuses otherwise).

## 5. Read-out and sentences

`zs_lagovad_readout` (the DAD column is added when `outputs/zs_lagovad/eval/dad/` exists): per
video Δ over the two-class videos, 95 % cluster bootstrap (each video its own cluster).
`arm − LaGoVAD` → above / below / not separated, **PRELIMINARY (n = 1 seed), not matched**;
`R4 − W0` → gain / cost / no difference shown. Printed: micro (raw pooled), clip-level AUC (DAD
has negatives), clip oracle, middle ruler. All DAD clips have about the same length, so the length
ruler should sit near 0.5 — a much higher value would mean the lengths carry the label.

**Never write:** "beats LaGoVAD" (PreVAD vs Nexar training); a DAD number beside a published DAD
number from the accident-anticipation literature (AP / TTA there, not frame AUC); that the
released checkpoint's DAD read is zero-shot as a fact (PreVAD not checked for DAD clips).

## 6. Read-out

**2026-11-04, PRELIMINARY (n = 1 seed, s2024), not matched.** Raw:
`outputs/v_final/zs_lagovad/REPORTS/zs_lagovad_readout.md` and `eval/dad/`. All 466 videos scored,
15 rows each; all 165 positives abnormal on exactly rows 13–14; V2-S manifest `clip_frame_step` 3,
`stride` 7, `assumed_fps` 25 (read by the user on Drive). AP = `core.tools.zs_compare_table`.

| row | macro [95 % CI] | micro | clip AUC | AP |
| --- | --- | --- | --- | --- |
| LaGoVAD (released ckpt) | 0.6462 [0.5881, 0.7016] | 0.5681 | 0.5808 | 0.0640 |
| W0 (no SA²) | 0.7317 [0.6897, 0.7716] | 0.6693 | 0.6451 | 0.1102 |
| R4 (SA²) | 0.7571 [0.7175, 0.7963] | 0.7020 | 0.6729 | 0.1385 |

Sentences (§5): W0 − LaGoVAD **+0.0855 [+0.0196, +0.1517] above**; R4 − LaGoVAD
**+0.1110 [+0.0466, +0.1758] above**; R4 − W0 **+0.0254 [−0.0084, +0.0592] no difference shown**.
Clip oracle micro 0.8390, base rate 0.0472. **Read §8 before quoting the macro.**

The four-set table is `ZS_COMPARISON.md`.

## 7. Amendment 1 (2026-11-04, after the census, before any feature or score)

**What the census measured** (`REPORTS/dad_census.json`, the only DAD output seen): 466 test videos
(165 positive / 301 negative), every one **100 frames at 25.00 fps** (not 20); first flagged frame
min 1 / **median 1** / max 86; flagged-frame counts 0…100 (most positives 100/100); flag runs not
contiguous; one positive (`pos_000488`) has no flag. The flag therefore marks the **objects
involved** in the accident from their first appearance, not the accident time. Under §1's rule most
positives would be abnormal on every row and leave the macro (two-class videos only).

**Authorized by the user on 2026-11-04** (choices offered with the census, before any feature
existed): labels = the DAD convention; stride / V2-S step re-derived for 25 fps.

| § | was | now | why |
| --- | --- | --- | --- |
| 1 labels | any object flag = 1 | **toa = 90**: positive 0-based frames 90–99 (1-based 91–100) abnormal; negatives normal; annotation parsed and printed only | the DAD convention (DSA, UString); the flag is involvement |
| 2 stride | 5 (0.25 s at 20 fps) | **7** = 0.28 s at 25 fps; rows frames 0, 7, …, 98 (15), 2 abnormal (frames 91, 98) | nearest to the arms' 0.267 s |
| 2 V2-S step | 2 (1.5 s at 20 fps) | **3** = 16 × 3 frames = 1.8 s at 25 fps | the T2 / Nexar step in frames (user's pick; step 2 would be 1.2 s) |
| 3 gates | fps 20; positives flagged; negatives unflagged; median first flag ≥ 50 | **fps 25 ± 0.5; every positive > 90 frames**; flag statistics printed, never gating | the flag no longer defines anything |

Caches and the data dir carry the stride in their name (`DAD_s7_*`), so nothing built under §2's
s5 can be read by mistake. Every other choice of §2–§5 stands. Never write that the DAD positives'
abnormal span was annotated per video: it is a fixed convention (the last 0.4 s).

## 8. Amendment 2 (2026-11-04, after the read-out): what the macro does not measure

**Authorized by the user on 2026-11-04.** It changes no number and no sentence of §6; it adds what
must be printed beside them. Everything below was computed **after seeing §6**, so it is printed,
not gated. Tool: `core.tools.dad_position_diag` (report `REPORTS/dad_position_diag.{md,json}`).

**The defect, in Amendment 1.** A fixed `toa = 90` makes the label a function of the row: every
positive is abnormal on the same last two rows, so a no-pixel `t/N` ruler scores macro **1.000**.
§5's rulers did not include it (the middle ruler printed 0.1538). The read-out now prints the
monotone ruler for TAD and DAD.

**Placebo: the same mask on the 301 negatives, which hold no accident.**

| row | macro | placebo, all rows | macro, rows 7+ | placebo, rows 7+ |
| --- | --- | --- | --- | --- |
| LaGoVAD | 0.6462 | 0.5134 [0.4709, 0.5564] | 0.6439 | 0.5260 [0.4817, 0.5692] |
| W0 | 0.7317 | **0.6260** [0.5933, 0.6586] | 0.5980 | 0.5042 [0.4648, 0.5432] |
| R4 | 0.7571 | **0.6477** [0.6141, 0.6797] | 0.6530 | 0.5349 [0.4939, 0.5742] |

W0 and R4 scores rise over the video without an accident; LaGoVAD's do not. On rows 7+ the placebo
is near chance and neither arm's within-video macro stays above LaGoVAD's. The "above" sentences
of §6 therefore come mainly from the opening rows.

**Unproven mechanism.** Row 7 is the first whose causal V2-S window holds no repeated frame (45
frames back at step 3, stride 7, `pad = clamp to frame 0`), and the negatives' mean curve flattens
near there. LaGoVAD, with no motion input, has no ramp. This is a coincidence of timing and an
absence, not a test: a pad-free read needs a re-extraction nobody has run, and a position trend
learned on Nexar is not excluded. On a 4 s video, 7 of 15 rows are padded.

**Same-position AUC: positive vs negative videos on the same rows** (position cannot help).

| row | accident rows 13–14 | rows 7–12 | difference |
| --- | --- | --- | --- |
| LaGoVAD | 0.5881 [0.5339, 0.6419] | 0.5762 [0.5220, 0.6308] | +0.0118 [+0.0025, +0.0212] |
| W0 | 0.6590 [0.6068, 0.7106] | 0.6363 [0.5823, 0.6884] | +0.0227 [+0.0058, +0.0396] |
| R4 | 0.6871 [0.6355, 0.7379] | 0.6632 [0.6113, 0.7144] | +0.0238 [+0.0056, +0.0418] |

Accident rows, paired: W0 − LaGoVAD +0.0709 [+0.0072, +0.1344]; R4 − LaGoVAD
+0.0990 [+0.0318, +0.1656]; R4 − W0 +0.0281 [+0.0001, +0.0557]. Most of the separation exists
before the accident rows: this separates accident videos from normal ones, it does not localize.
The R4 − W0 interval touches zero, is post-hoc, and **does not overturn §6's "no difference shown"**.

**Write:** the §6 table with the ruler beside it, and "on DAD the macro cannot separate a reaction
to the accident from a score that rises with time". **Never write:** that a DAD macro shows
accident detection or localization; a gain for R4 on DAD; the same-position AUC as the set's
primary metric (it was chosen after the numbers).
