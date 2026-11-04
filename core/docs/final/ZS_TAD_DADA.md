# Zero-shot TAD / DADA-2000: SA² (R4) vs its host (W0), Nexar-trained, seed 2024

Status: built 2026-11-03; §3 read out 2026-11-04 (§6); §7 (LaGoVAD released) read out 2026-11-04 (§7.5). Rule written before any number. Runbooks:
`colab/final/zs_pack_frames.ipynb` (CPU, once) → `colab/final/zs_tad_dada_r4.ipynb` (GPU).

## 1. Question

Does SA² on the CLIP branch (LaGoVAD-STP, `STP_BUILD.md`) change how the Nexar-trained model
transfers to two corpora it never saw: TAD (fixed traffic cameras) and DADA-2000 (dashcam)?
On DoTA-CAP-dev it read R4 − R1 +0.0009 [−0.009, +0.011], "no difference shown — PRELIMINARY".

## 2. Arms (both Nexar `window`, s2024, 1,749 steps, prefix OFF = LaGoVAD's released prompts)

| arm | what | checkpoint |
| --- | --- | --- |
| **W0** (control) | host: CLIP + V2-S through `W_u` (zero-init), no CRN, `span` | `Thesis-Final/outputs/wu/W0/s2024` (trained by `wu_arms.ipynb`; retrained by the notebook only if not verified) |
| **R4** | W0 + SA² on the top-12 motion-selected 7×7 CLIP patches | `Thesis-Final/outputs/stp/R4/s2024` |

W0 **is** v2 Nexar A2 s2024: same code path, same seed, DoTA-CAP-dev macro 0.8110072204788573 in
both reads. The notebook asserts both configs: `stp.text_prompt = released`; `stp.sa2` false / true.

## 3. Read-out rule

* **Sets.** TAD test: 100 videos (60 with an accident), stride 8, `core.evaluate`, raw scores
  (`--score-norm auto`), §26's protocol = v2's TAD table. DADA-2000: the 370 source videos of
  T2-test, whole video, every s8 row, annotation span, `dada_whole_eval` (D-N20) = v2's DADA column.
  Inputs = v2's A2 bakes with the **Nexar** statistics (`A2_S/TAD_s8_nexar`,
  `A2_S/DADA2000_orig_test_sources_nexar`), plus R4's patch cache.
* **Endpoint per set.** Per-video Δ (R4 − W0) of the frame AUC over the two-class videos, mean
  with a 95 % cluster bootstrap (each video its own cluster, 10,000 resamples, seed 2026).
* **Sentence.** "gain shown" iff the low bound > 0 **and** the mean ≥ +0.01; "cost shown" iff the
  high bound < 0; otherwise "no difference shown". One seed ⇒ every sentence carries
  **PRELIMINARY (n = 1 seed)** and adopts nothing.
* **Printed, never deciding.** v2 A2 s2024 beside W0 (reproduction: |W0 − A2 s2024| macro
  > 0.001 is flagged; v2 has per-seed curves for TAD only, per-seed macro for DADA); v2 A2's
  5-seed per-frame mean and R4 − that mean (biased toward A2: averaging smooths curves); micro;
  TAD clip-level AUC; rulers (TAD: middle 0.683, clip oracle 0.9226, length 0.8968; DADA: middle,
  `t/N`); training diagnostics (`stp_entropy`, `‖W_q‖`, `rho_u`, `‖W_u‖`).
* **Never write.** Not "SA² hurts / helps transfer" from one seed; not any TAD micro beside
  LaGoVAD's 89.56 (Gate T0 reproduces 0.7912 micro / 0.7578 macro, C8b); not "position-free"
  (TAD middle ruler 0.683 sits within ≈ 0.02 of W0's expected macro).

## 4. Pixels (why two notebooks)

SA² needs 1×1 + 7×7 window crops per row, cut from the 224² squash and re-encoded by CLIP — no
cache of that exists for TAD or DADA (the text-window probe's DADA cache has grids 2/3/5).

* **TAD**: frames only in the Kaggle copy (`kagglehub`), non-padded names → the numeric-order
  farm of `tad_zeroshot.ipynb`; rows = `frames[::8]` = `TAD_ncc` rows (`ceil(N/8)`).
* **DADA**: `Thesis/data/DADA2000Origin/DADA2000/DADA2000.zip` (6 volumes, 117 GiB; 370 sources
  ≈ 18 GiB of 1584×660 PNG). Only `frames[::8]` become rows, so `zs_frames dada-plan` reads the
  archive listing once and extracts only those (≈ 2.3 GiB), in lexicographic order inside each
  `images/` dir (= `list_frame_images`, the CLIP cache's order); the image count must equal the
  census (C10).

`zs_frames pack` keeps each video's row frames (original bytes) in `{id}.zip`
(`Thesis-Final/data/ZS_frames/{TAD,DADA2000_orig}/`); `verify` writes the manifest only when every
archive holds exactly its CLIP rows. The GPU notebook then runs `clip_windows packed --grids 1 7
--cells resized224` (gate: the 1×1 cell reproduces the CLIP row, per-video mean cos ≥ 0.99, every
row ≥ 0.95) and `stp_patches select` (Mo on ±1 row, top-12, raw).

Caveat (printed, not fixed): Mo's ±1 row is 8 frames — 0.27 s at Nexar's / DADA's 30 fps; TAD's
frame rate is not verified here.

## 5. Code

`core/tools/zs_frames.py` (`dada-plan`, `pack`, `verify`), `clip_windows packed`,
`core.evaluate --patch-dir` + `FeatureEvalDataset(patch_dir=)`, `dada_whole_eval --patch-dir`
(both refuse a mismatched cache through `eval_patches`, and an SA² checkpoint without one),
`core/tools/zs_readout.py`, constants `ZS_*`. Tests `core/tests/test_zs_tad_dada.py`.

## 6. Read-out (2026-11-04, `REPORTS/zs_readout.md`, PRELIMINARY n = 1 seed)

Reproduction: W0 = v2 A2 s2024 on both sets (gap +0.00001 / +0.00000).

| set | W0 macro | R4 macro | R4 − W0 (per video, 95 % cluster CI) | sentence |
| --- | --- | --- | --- | --- |
| TAD (60 two-class) | 0.6983 [0.622, 0.770] | 0.7220 [0.652, 0.789] | **+0.0237 [+0.0005, +0.0499]** | gain shown — PRELIMINARY |
| DADA (370) | 0.7615 [0.737, 0.786] | 0.7579 [0.733, 0.782] | −0.0036 [−0.0161, +0.0089] | no difference shown — PRELIMINARY |

Printed: TAD micro 0.603 → 0.717 and clip AUC 0.594 → 0.657 (most of R4's TAD change is
clip-level); TAD middle ruler 0.683, DADA middle 0.734. R4 SA² entropy 0.911 of log K (still ≈
selection + average), ‖W_q‖ 0 → 1.43. Caveats: the TAD low bound sits at +0.0005 and the CI
omits seed variance (n = 1); never write "SA² improves transfer".

## 7. LaGoVAD's released checkpoint on the same sets (rule written 2026-11-04, before any read)

Authorized by the user on 2026-11-04 ("từ public checkpoint của LaGoVAD hãy thực hiện zeroshot
trên DoTA-CAP-dev, DADA-2000"). LaGoVAD publishes no DADA-2000 number and no DoTA-CAP number;
the released `best.ckpt` read by this project becomes the reference row (lesson 8b).
Runbook `colab/final/zs_lagovad.ipynb` (CPU suffices); read-out `core.tools.zs_lagovad_readout`.

**Disclosure.** The released checkpoint's TAD numbers are already on record (Gate T0: micro
0.7912, macro 0.7579; clip AUC 0.769). It has **never** been read on DoTA-CAP-dev at protocol B
or on DADA-2000 whole sources.

### 7.1 What is read

| set | LaGoVAD command | input | identical to W0 / R4 in |
| --- | --- | --- | --- |
| DoTA-CAP-dev | `protocol_b_eval --baseline released best.ckpt --split dota_cap_dev` (stride 3) | plain `DoTA_s1_ncc`, `[::3]` here | clip set, labels, stride, native-frame interpolation, definitions (`DoTA/labels_s8` `defs.json`) |
| DADA-2000 | `dada_whole_eval --baseline released best.ckpt` | plain `cache/clip/DADA2000_orig` (s8) | 370 sources, labels, rows, definitions (`Nexar_window` `defs.json`) |
| TAD | `core.evaluate --baseline-ckpt best.ckpt --set kip.enabled=false --set data.dataset=TAD` (re-run; = Gate T0) | plain `TAD_ncc` | §3's TAD protocol |

* LaGoVAD = `best.ckpt`, **KIP off** (`RELEASED_CKPT_OVERRIDES`), no motion stream, no CRN, no SA²,
  released prompts. The compat loader raises on any missing / unknown key (C5).
* Printed, never deciding: LaGoVAD on DoTA-CAP-dev at **protocol A** (`--stride 8`), the rate its
  checkpoint was gated at; protocol B is the table's protocol for every row.

### 7.2 Contrasts and sentences

* Per set, per item (DoTA: clip, cluster = source video; TAD / DADA: the video), Δ over the
  two-class items all rows scored, 95 % cluster bootstrap (10,000 resamples, seed 2026).
* `W0 − LaGoVAD`, `R4 − LaGoVAD`: low > 0 → **"above"**; high < 0 → **"below"**; else **"not
  separated"** (§26 T6's wording). Each carries **"PRELIMINARY (n = 1 seed), not matched"**.
* `R4 − W0`: §3's sentence. On TAD and DADA it reproduces §6; on DoTA-CAP-dev it is new (§6
  of `STP_BUILD.md` contrasted R4 with R1, not W0).
* Micro: DoTA per-clip min-max pooled (lesson 12), TAD / DADA raw pooled; rulers beside.

### 7.3 Integrity (before any number is read)

1. The plain CLIP rows LaGoVAD reads equal the first 512 columns of the arms' A2 bakes on every
   item (`zs_lagovad_readout check-inputs`; DoTA `s1[::3]` vs `dota_cap_dev_s3_from_s1_nexar`,
   DADA s8 vs `DADA2000_orig_test_sources_nexar`; A2 has no CRN so its CLIP columns are raw, `s = 1`).
2. TAD micro re-read = Gate T0 0.7912 ± 0.001, or the read-out refuses to write (wrong
   checkpoint, config or cache).
3. Every row of a set covers the same two-class items, or the read-out refuses.

### 7.4 Never write

* "KAT-VAD / LaGoVAD-STP beats LaGoVAD" — not matched: LaGoVAD trained on PreVAD (large,
  multi-domain), the arms on Nexar; n = 1 seed.
* "LaGoVAD's DADA-2000 result" — write "LaGoVAD (released checkpoint, evaluated by us)".
* That LaGoVAD's DADA read is zero-shot as a fact: PreVAD is web-collected and has not been
  checked for DADA-2000 clips — write "zero-shot status not verified".
* A DoTA-CAP number beside LaGoVAD's 62.60 or full DoTA (D14); a TAD micro beside 89.56 (C8b).

Printed caveat (travels with the LaGoVAD row): the released checkpoint reads the arms' `_ncc`
rows, and `RESULTS_NCC.md` measured it **lower** under `no_center_crop` than under centre crop
on DoTA (macro −0.0170 [−0.0301, −0.0037]). Its row here may therefore sit slightly low; the
transform is kept because a second cache would break the row-for-row input identity (§7.3-1, C2).

### 7.5 Read-out (2026-11-04, `REPORTS/zs_lagovad_readout.md`, PRELIMINARY n = 1 seed, not matched)

Integrity: inputs identical row for row on all three sets; TAD micro 0.7912 vs T0 (gap +0.00003) PASS.

| set | LaGoVAD macro | W0 macro | R4 macro | W0 − LaGoVAD | R4 − LaGoVAD | R4 − W0 |
| --- | --- | --- | --- | --- | --- | --- |
| DoTA-CAP-dev (569) | 0.6418 [0.617, 0.667] | 0.8110 [0.795, 0.827] | 0.8111 [0.793, 0.829] | +0.169 [+0.144, +0.196] above | +0.169 [+0.140, +0.200] above | +0.0001 [−0.010, +0.010] no diff |
| TAD (60) | 0.7579 [0.696, 0.815] | 0.6983 [0.622, 0.770] | 0.7220 [0.652, 0.789] | −0.060 [−0.146, +0.026] not sep. | −0.036 [−0.118, +0.047] not sep. | +0.024 [+0.001, +0.050] gain |
| DADA-2000 (370) | 0.6202 [0.593, 0.648] | 0.7615 [0.737, 0.786] | 0.7579 [0.733, 0.782] | +0.141 [+0.109, +0.174] above | +0.138 [+0.106, +0.170] above | −0.004 [−0.016, +0.009] no diff |

Printed: micro LaGoVAD / W0 / R4 — DoTA (min-max) 0.611 / 0.726 / 0.729, TAD 0.791 / 0.603 / 0.717,
DADA 0.539 / 0.698 / 0.705; TAD clip AUC 0.769 / 0.594 / 0.657. LaGoVAD DoTA-CAP-dev protocol A
0.6185. Rulers: DoTA `t/N` 0.557 (cubic `p_T2` 0.822 on record, above every row); TAD middle
0.683; DADA middle **0.734 — above LaGoVAD (0.620)**, below the arms by 0.024–0.027 (R4 / W0).

Reading: the Nexar-trained arms sit far above the released checkpoint on both **dashcam** sets and
are not separated from it (point estimate below) on **fixed-camera** TAD, where LaGoVAD also keeps
a clear clip-level lead. The arms' training corpus is dashcam and LaGoVAD's is not, so the dashcam
gap cannot be credited to the architecture (not matched). SA² changes nothing on DoTA / DADA.
