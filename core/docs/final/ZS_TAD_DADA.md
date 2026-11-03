# Zero-shot TAD / DADA-2000: SA² (R4) vs its host (W0), Nexar-trained, seed 2024

Status: built 2026-11-03, not run. Rule written before any number. Runbooks:
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

## 6. Read-out

Not run.
