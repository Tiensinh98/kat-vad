# KAT-VAD v5 (LaGoVAD-MQR) — pre-registration

Proposal: `core/docs/final/Thesis proposal Deep learning-based traffic video anomaly detection.md` (2026-10-26).
Plan: `.project/plans/katvad-v5-mqr.md`. Each section is frozen **before** the number it governs is read; a defect
found later becomes a numbered amendment, never an edit (v2 convention, `core/docs/v2/PREREG_ADDENDUM.md`).

v5 = v2's trunk **without CRN**, with the V2-S stream read by the Motion Query Readout (MQR). The user chose
(2026-10-26): staged plan B, Nexar-only training, 3 seeds first, proposal left unedited until feasibility is known.

---

## §1 Stage 0 — does a VideoMAE token carry local anomaly evidence? (S0-1 … S0-12)

Signed 2026-10-26, before any token grid exists. Inference only; nothing is trained.

**S0-1 Set.** DoTA-CAP-dev (`dota_cap_dev`, 569 clips; 534 have at least one boxed frame in the anomaly window).
*Deviation from the proposal* (which names a 500-clip subset of DoTA-train): DoTA-train has no pixels in this
project (DoTA's own frames are gone; CAP alignment covers the 1,397 test clips only), and DoTA-CAP-dev is already the
design set of v2. `dota_cap_eval` is extracted (grids only, S0-4) and **no label of it is read** in Stage 0.

**S0-2 Boxes.** `DoTA_annotations.zip` from `MoonBlvd/Detection-of-Traffic-Anomaly` (`dataset/`), per-frame
`objects[].bbox = [x1, y1, x2, y2]` in 1280×720 pixels (the frame size `stauc.py` assumes). A step's mask is the union of
that frame's boxes mapped to the 224×224 squash (x·224/1280, y·224/720); tile weight `w_j` = covered fraction of tile
`j`. Classes without boxes (obstacle, leave_to_left/right, …) are absent from every box read, present in D2.

**S0-3 Steps.** Rows `t = 0, 3, 6, …` (protocol B's `s1[::3]`, E1). VideoMAE V2-S (`vit_s_k710_dl_from_giant`) on the
16-frame causal window ending at `t` with DoTA's native step (`DOTA_VIDEOMAE_FRAME_STEP`, 1.5 s) — the exact window of
v2's `DoTA_CAP_s1_squash` cache row `t`. CLIP ViT-B/16 on frame `t`, `no_center_crop`.
*Deviation from the proposal* (centred 1.6 s window): kept causal so that G0 can compare against v2's cache.

**S0-4 Representations.** Last-layer VideoMAE tokens, **before** `fc_norm`, `(8, 14, 14, 384)`:
`V14` = mean over the 8 temporal slots; `V7` = 2×2 average of `V14` (proposal §3.2's grid). `C14` = CLIP patch tokens,
`visual_projection(post_layernorm(h[:, 1:]))`. `V7` is stored fp16 for all 1,129 DoTA-CAP clips (Stage 2 reuses it);
change maps (S0-5) are stored for dev only. ClearCLIP patches: **not run** (proposal row-7 context, deferred).

**S0-5 Change.** For step `k ≥ 1` and tile `j`: `Δ_k(j) = 1 − cos(G_k(j), G_{k−1}(j))` (scale-free across encoders).
Global change `g_k = 1 − cos(mean_j G_k, mean_j G_{k−1})`. Step 0 is undefined and takes step 1's value in D2.

**S0-6 D1 — inside vs outside, against the same box placement before the anomaly.**
`in_k = Σ_j w_j Δ_k(j) / Σ_j w_j`, `out_k = Σ_j (1−w_j) Δ_k(j) / Σ_j (1−w_j)`.
Anomaly steps `A_c` = `k ≥ 1`, frame label 1, non-empty mask. Pre steps `P_c` = `k ≥ 1`, frame before `anomaly_start`.
`ρ_c = log(mean_{A_c} in / mean_{A_c} out)`; `ρ_c^pre` = the same over `P_c` with the **fixed** mask of the first step
in `A_c` (controls for where boxes sit: road ahead, frame centre, ego-motion expansion). Effect `e_c = ρ_c − ρ_c^pre`,
defined iff `|A_c| ≥ 2` and `|P_c| ≥ 2`.

**S0-7 D3 — pointing game.** Per anomaly step: hit = `w_{argmax Δ_k} > 0`; chance = share of tiles with `w > 0`.
Per clip: `h_c` (hit rate over `A_c`), `χ_c` (mean chance), `h_c^pre` (hit rate over `P_c` with the fixed mask).

**S0-8 D2 — training-free detection.** Per-step scores: global `g`, max `max_j Δ`, top-k mean (`k = 5` on 7×7,
`k = 20` on 14×14), linearly interpolated to native frames (`to_native`, stride 3), per-clip AUC on two-class clips.
`d_c = AUC_c(top-k) − AUC_c(global)`.

**S0-9 Statistics.** Every interval = cluster bootstrap over DoTA source videos (`dota_group`), B = 10,000,
level 0.95, seed 2026 (`core.metrics.cluster_bootstrap_ci`). Non-ego = `ego_involve == False`.

**S0-10 Validity gates (hard; any failure → INVALID, nothing is read).**
G0 every stored step: `cos(fc_norm(mean of the 1,568 tokens), DoTA_CAP_s1_squash[t]) ≥ 0.9999`.
G1 every CLIP step: `visual_projection(post_layernorm(h[:, 0]))` equals `image_embeds` (|Δ| ≤ 1e-5), and the frame
passes v2's pixel gate against `DoTA_s1_ncc[t]` (`dota_cap.passes_frame_gate`).
G2 every dev clip: CAP frame aspect within 2 % of 16:9 (box mapping assumes DoTA's 1280×720 geometry).

**S0-11 Decision (on `V7` first).**
- **G-A (premise):** `e_c` CI low > 0 over all eligible clips **and** point > 0 on non-ego clips.
- **G-B (pointing):** CI of `h_c − χ_c` low > 0 **and** CI of `h_c − h_c^pre` low > 0.
- **G-C (selection):** CI of `d_c` low > 0 on non-ego two-class clips (proposal §4.3: max-token beats global on
  ego-uninvolved clips; top-k is the gated form, max is printed).

| outcome | rule | next |
|---|---|---|
| **GO-7** | G-A, G-B, G-C pass on `V7` | Stage 1 with the 7×7 grid |
| **GO-14** | `V7` fails one of them, `V14` passes all three | Stage 1 with 14×14 (PCA, proposal side ablation) |
| **PARTIAL** | G-A passes on `V7` or `V14`, but G-B or G-C fails on that grid | report; the user decides |
| **KILL** | G-A fails on both `V7` and `V14` | MQR's premise not supported; Stage 1 not built |

**S0-12 Printed, never gating.** D1/D3/D2 for `C14` (proposal: a ratio near 1 confirms CLIP patches are the wrong
place to read); everything by box-share bin (< 0.5 %, 0.5–2 %, 2–8 %, > 8 % of the frame, median over the window) and by
ego; D2 `max`; the `t/N` ruler and v2's cubic `p_T2` on CAP-dev (0.8220, `RESULTS_E3.md`) beside every D2 macro;
D0 (CLIP cosine 8 frames apart within a clip vs random cross-clip pairs, raw and mean-centred). Disclosed before the
run: on CAP-dev, box share and ego are confounded (non-ego clips are 36 of 41 in the < 0.5 % bin, 32 of 208 in > 8 %),
so a bin effect and an ego effect cannot be separated here.

### §1.1 Amendment 1 (S0-7′) — pointing control against chance, written before any real grid

Found 2026-10-26 by a **null dry-run** of the read-out (random change maps on the real CAP-dev annotations): G-B's
second check `h_c − h_c^pre` read **+0.052 [+0.034, +0.070]** on pure noise. Cause: boxes grow during the anomaly
window (agents approach), so `χ_c` exceeds the chance of the fixed pre-anomaly mask, and a raw hit-rate difference
rewards box growth. **Replaced by** `(h_c − χ_c) − (h_c^pre − χ_c^pre)`, with `χ_c^pre` the tile share of the fixed
mask. Re-run on the same noise: +0.005 [−0.011, +0.021]. Planted-signal dry-run (Δ + 0.5 on boxed tiles during the
anomaly): every gate passes on V7 and V14 → GO-7; noise → KILL. Nothing else in §1 changes; no real Stage 0 number
existed when this was written.

### §1.2 Amendment 2 (S0-10 G1′) — the pixel gate on a row subset; written AFTER the first read-out

**Disclosed: written 2026-10-26 after the first read-out (`stage0_readout.md`, decision INVALID) was seen.**
Defect: `v5_stage0.py` applied v2's clip-level pixel gate (`dota_cap.passes_frame_gate`: mean cos ≥ 0.99 **and**
min ≥ 0.95) to the Stage 0 rows `s1[::3]` only. v2 applied it to **every** frame and passed all 569 dev clips. The 15
clips that failed (all `g1_pixel`) have v2 all-frame means 0.9901–0.9918 (`dota_cap_alignment.json`): a third of the
rows of a clip sitting on the 0.99 line falls under it by sampling, not because the pixels differ. G0 (min cos
0.99999999998) shows the frames are v2's. **Replaced by:** G1-pixel = v2's all-frame verdict for the clip (already
required: every finalized id has `reason == ok`) **and** min over the Stage 0 rows ≥ `DOTA_CAP_MIN_COS` (0.95; the
subset min cannot be below the all-frame min, so this only guards against a wrong frame map). The 15 clips are
re-extracted with the amended gate (their V7 grids were never written, and Stage 2 needs them), then the read-out is
re-run unchanged. What the first read-out already showed and this amendment cannot change: G-B fails on V7 and V14 with
CIs centred on 0 over 514 clips; 15 more clips cannot move it past 0, so the expected outcome is **PARTIAL**, and that
expectation is recorded here before the re-run.

### §1.3 Post-hoc check L — does window overlap hide VideoMAE's pointing? (printed, never gating)

Written 2026-10-26 after the first read-out, **before** any lagged number exists. Motive: consecutive Stage 0 rows are
3 frames apart, while each V2-S window spans 16 frames (`DOTA_VIDEOMAE_FRAME_STEP` = 1 at 10 fps), so two consecutive
windows share 13 of 16 frames. CLIP's change compares two single frames 3 apart. S0-5's VideoMAE change is therefore
diluted and lagged by construction.

**L-1 Read.** On the stored fp16 `V7` grids of DoTA-CAP-dev (no new extraction), S0-5 is computed at row lag
`ℓ ∈ {1, 3, 6}`: `Δ_k(j) = 1 − cos(G_k(j), G_{k−ℓ}(j))`, rows `k < ℓ` undefined, and the same for `g_k`.
`ℓ = 6` (18 frames) is the smallest lag with no shared frame; `ℓ = 1` must reproduce Stage 0's V7 (regression: max
|Δ| against the stored fp32 `d7` printed). D1, D3 (with Amendment 1), D2 and the three S0-11 gate booleans are printed
per `ℓ` with S0-9's bootstrap. Undefined leading rows take the first defined row's value in D2 (S0-5's rule).
**L-2 Interpretation (fixed now).** If at `ℓ = 6` both G-B intervals have low > 0, the Stage 0 pointing failure is
attributed to window overlap and the next design must compare non-overlapping windows; if they do not, VideoMAE token
change does not point at the agent whatever the overlap, and only C14 (CLIP patches) carried pointing on CAP-dev.
Either way nothing in §1's decision changes; C14 at matched lag needs pixels and is not run here.
**L-3 Centre shift (added 2026-10-27, before any lagged or shifted number exists).** The proposal centres the
VideoMAE window on `t`; S0-3 kept it causal (ends at `t`, centre `t − 7.5` frames), so a moving agent's box at `t` is
compared with tokens centred ~0.75 s earlier. A causal window ending at `t + 7.5` is the centred one, so on the stored
rows the box/label of step `k` is read against grid row `k + s`, `s ∈ {0, 2, 3}` (window centre `t − 7.5`, `t − 1.5`,
`t + 1.5` frames); the last `s` rows of a clip are undefined (no future frames stored) and are dropped from D1/D3,
and take the last defined row's value in D2. Every `(ℓ, s)` of `ℓ ∈ {1, 3, 6}` × `s ∈ {0, 2, 3}` is printed.
**Interpretation (fixed now):** if, for at least one `ℓ`, both G-B intervals have low > 0 at **both** `s = 2` and
`s = 3` while `s = 0` fails, the pointing failure is attributed to the causal offset and Stage 1 uses a centred window
(Nexar re-extracted centred; row 1 is then retrained, not borrowed from v2's A2). Otherwise Stage 1 keeps the causal
window (v2-comparable). Requiring both shifts guards against picking one of nine printed cells.

---

## §2 Stages 1–2 — build MQR, train on Nexar, read it on DoTA-CAP-dev (S1-G1 … S1-G6, S2-1 … S2-14)

**Status: SIGNED 2026-10-27 by the user** (before any Stage 1 code produced a number), with S2-14's open choices
answered: **(a) R7 is not run** (no CLIP-patch cache; R7's rows below stay as text only); **(b) λ = 0.5, M = 8, LR × 1
fixed**; **(c) DVS anchor `span`** (D9). Build notes §2.4 were fixed before any Stage 1 number. **Disclosed:** written after Stage 0's read-out (PARTIAL) and check L (`RESULTS_V5_STAGE0.md`
§3–§5). Row R5Δ exists because of check L; every other row is the proposal's ladder (§4.5). No Stage 1 or Stage 2
number exists when this is written.

### §2.1 Training and inputs

**S2-1 Corpus and recipe.** Nexar `window` (v2 §29: 20-row windows at `s1[::8]`, hop 8, positives only; 1,749 steps),
`nexar_train` statistics, v2 Nexar **A2's** recipe exactly: V2-S, no CRN (`v2.crn = none`), `loss.dvs_anchor_mode =
span`, `data.is_egocentric = false`, `L_neg` off. Seeds **2024, 2025, 2026** (the first three of `NEXAR_RUN_SEEDS`, so
every row is paired with v2's runs of the same seed). No hyperparameter is tuned (S2-13 D2).

**S2-2 Grids.** `G_t` = V7 of the V2-S window of row `t`: last-layer tokens before `fc_norm`, mean over the 8 temporal
slots, 2 × 2 spatial average → 7 × 7 × 384, fp16 (S0-3/S0-4; the window is the causal one v2's `Nexar_s1_squash` row `t`
used, L-3). Extracted: `nexar_train` positives at `s1[::8]` (training rows), `nexar_val` at `s1[::3]` (protocol N-B).
DoTA-CAP-dev at `s1[::3]` is Stage 0's stored V7 (`V5_GRID_DATASET_TAG`). `nexar_test` grids only at S2-11. TAD / DADA /
DoTA-CAP-eval grids are **not** extracted in Stage 2.

**S2-3 Rows** (one ingredient per row; proposal §4.5 numbering).

| row | readout of `G_t` | sMIL | source |
|---|---|---|---|
| R0 | none (CLIP only) | — | v2 Nexar A0, s2024–2026 (borrowed, no grid involved) |
| R1 | mean of the 49 tiles | — | ~~v2 Nexar A2, borrowed iff S1-G3 passes~~ → **retrained s2024–2026 through the v5 path** (§2.4 B9) |
| R3 | 1 query, content keys | — | new |
| R5 | **LaGoVAD-MQR**: 8 queries, content keys | λ = 0.5 | new |
| R5Δ | R5 with change keys (S2-4) | λ = 0.5 | new (Stage-0-motivated) |
| R7 | R5's readout on CLIP 7 × 7 patches | λ = 0.5 | new, **only if the user approves its cache** (S2-12) |

Not run in this stage: rows 2, 4, 6, every side ablation, warm start.

**S2-4 Readout.** Keys `K_{t,j}`: R3/R5 `LN(G_{t,j})`; R5Δ `LN([G_{t,j} ; ΔG_{t,j}])` with `ΔG_t = G_t − G_{t−ℓ}`.
`LN` has no affine parameters. `a_{t,m,j} = softmax_j(u_m^⊤ K_{t,j} / √d_K)`; `z_{t,m} = [W_V Σ_j a_{t,m,j} G_{t,j}]_m`
(the `m`-th `D/M` slice); `r_t = [z_{t,1}; …; z_{t,M}]`. Values are always `G` (content); only R5Δ's keys see change.
Init: `u_m = 0`, `W_V = I`, bias 0 → `r_t` = tile mean = mean of the 1,568 tokens, for every row.
**Lag `ℓ`** = the smallest row lag whose window shares no frame with row `t`'s: `ℓ = ⌈span / stride⌉`, `span =
15·step + 1` frames. Nexar (step 3): span 46 → `ℓ = 6` at s8, `ℓ = 16` at s3 (48 frames, 1.6 s both). DoTA (step 1):
span 16 → `ℓ = 6` at s3 (18 frames = check L's ℓ = 6). **`ΔG` is computed inside each dataset item** (training window,
eval crop, DoTA clip) **before** DVS splicing; its first `ℓ` rows are 0. Never computed on a spliced sequence: a splice
boundary would read as change exactly where DVS put the abnormal clip (a label leak).

**S2-5 Path into the trunk (deviation D1).** `u_t = fc_norm(r_t)` (frozen, checkpoint weights) → `c·(u_t − m_u)/σ_u`
with **Nexar A2's baked statistics** (`cache/v2/A2_S/Nexar_s8`) → v2's `MotionResidual` (`W_u` zero-init) →
`h_t = x_t + W_u(·)`. At init every row R1–R5Δ computes R1's function, and R1's equals R0's (`W_u = 0`).
R7: `h_t = x_t + W_c·LN(r_t^{CLIP})`, `W_c` zero-init (R0's function at init).

**S2-6 Spatial MIL loss.** Proposal §3.2 exactly: `s_{t,m} = σ(g(z_{t,m}))`, `g: ℝ^{D/M} → ℝ` shared, `S_t = max_m s_{t,m}`,
`ŝ = mean of the top-k valid S_t`, `k = max(⌊L/16⌋, 1)`, `L_sMIL = BCE(ŝ, y)`, total `L_LaGoVAD + 0.5·L_sMIL`. It
applies to every sequence the trunk's `L_MIL` sees (DVS-synthesized ones included), with the same label and padding
mask. Learning-rate multiplier 1.

### §2.2 Stage 1 gates (hard; any failure → stop, nothing is trained or read)

- **S1-G1 grid ↔ v2 cache.** Every stored Nexar row: `cos(fc_norm(mean_j G_t(j)), Nexar_s1_squash[t]) ≥ 0.9999`
  (`V5_G0_MIN_COS`).
- **S1-G2 same start.** (a) unit tests: zero queries and `W_V = I` give `r_t` = tile mean (|Δ| ≤ 1e-6, fp32), for M ∈
  {1, 8} and both key types; `ΔG` of a DVS-spliced sequence equals its items' `ΔG` (no boundary change); (b) on 100 real
  Nexar windows, the motion term at init equals the baked A2 rows to |Δ| ≤ 1e-3 (fp16 grids).
- ~~**S1-G3 R1 borrow.**~~ **Dropped by §2.4 B9 (R1 is retrained; nothing is borrowed).** R1 retrained once through the v5 path (readout fixed to the mean), seed 2024: `nexar_val` crop
  endpoint within ±0.01 of v2 A2 s2024 (**0.886**) **and** DoTA-CAP-dev macro within ±0.01 of v2 A2 s2024's.
  Pass → R1 = v2 A2 s2024–2026. Fail → R1 retrained on all three seeds through the v5 path (the borrowed runs are not used).
- **S1-G4 provenance.** Every checkpoint: `global_step` = 1,749 = last `metrics.jsonl` step (J10); `config.yaml` differs
  from v2 A2's only in v5 keys.
- **S1-G5 map tools.** (a) the pointing read fed Stage 0's stored V7 change maps reproduces `RESULTS_V5_STAGE0.md` §3's
  V7 D3 row (|Δ| ≤ 1e-4); (b) S2-8's read on random maps fails and on maps with a planted in-box signal passes (lesson (bt));
  (c) STAUC ported from DoTA's official evaluation (`MoonBlvd/Detection-of-Traffic-Anomaly`) matches it on a fixed
  synthetic input (|Δ| ≤ 1e-6).
- **S1-G6 no collapse (read before S2-7).** Per new run: last-step `mil` and `sMIL` finite; R3/R5/R5Δ attention entropy
  printed per query; a run whose every query has entropy ≥ 0.999·log 49 at the last step is reported as "did not leave the
  mean" (still read; not a failure).

### §2.4 Build notes (written with the Stage 1 code, 2026-10-27, before any Stage 1 number)

Where the code had to choose something §2.1–§2.2 did not fix. None changes a rule; B1 changes S2-2 and S2-12's storage.

- **B1 `nexar_val` grids at every frame (`s1`), not `s1[::3]`.** Protocol N-B's crops start on any native frame, so a
  crop's rows are `s1[start::3]` and only an `s1` cache serves every phase. Rows *read* are still at stride 3, and
  `ℓ = 16` is taken at the eval stride. Cost: ≈ 225 × 1,200 rows × 37.6 KB ≈ **10 GB** on Drive (S2-12 said ≈ 3.4 GB
  for this part). Training grids stay `s1[::8]`.
- **B2 Row layout.** A v5 input row is v2 A2's baked row `[x ; c·ũ/σ_u]` (896) followed by `G_t` flattened (18,816) and,
  for R5Δ, `ΔG_t` (18,816). The model reads `x` and recomputes the motion term from `G`; the baked `u` columns are kept
  only so the A2 input-manifest check still binds the run to A2's statistics and S1-G2(b) can compare against them.
  `ΔG` is built per item in the dataset (training) and per crop / clip (eval), never after DVS (S2-4).
- **B3 STAUC.** DoTA's official `get_tarr` (frame method) and `get_stauc`, ported; S1-G5(c) compares against
  `data/DoTA/raw_annotations/stauc.py`. The map fed to it is the 7 × 7 step map upsampled bilinearly to 224 × 224,
  boxes scaled from 1280 × 720; the step score is the per-clip min-max of the row's anomaly curve. STAUC is taken over
  protocol-B **steps** (`s1[::3]`, the label of the step's frame), not over native frames. Printed only (S2-8).
- **B4 sMIL numerics.** `BCE(ŝ, y)` is computed as `BCE-with-logits(logit(ŝ, 1e-6), y)` in fp32 with autocast off
  (identical value; plain BCE is banned under AMP in this repo). `k = max(⌊L/16⌋, 1)` exactly as S2-6.
- **B5 Naming.** R5Δ is written `R5d` in code, configs and paths.
- **B6 S1-G3 references.** v2 Nexar A2 s2024: `nexar_val` crop endpoint 0.885856, DoTA-CAP-dev macro 0.811007 (both
  from the v2 read-outs). Unused since B9.
- **B7 S1-G5(c) was run locally while building** (the official file is on disk): port = official to |Δ| = 0. It is a
  tool check on synthetic input, not a Stage 1 number; the notebook re-runs it.
- **B8 (written AFTER the first S1-G1 read-out, 2026-10-27, disclosed) — TF32 pinned off; failed videos retried.**
  First `nexar_train` extraction (A100): **FAIL**, 552 / 563, 11 videos `g1_cos` (min 0.99977), all in IDs 413–593
  (ID 400–600 median min cos 0.999976 vs 0.999999999 outside). Diagnosis on 3 videos (one failing): with TF32 off for
  matmul and cuDNN, cos = **1.00000000** on every row; fp16 storage moves it by ~2e-8. Cause = a tool defect: the
  extractor left cuDNN's default TF32 on, v2's cache was plain fp32. Fix: `extract` sets both TF32 flags off
  (`exact_fp32`), and gathers/normalizes windows on the GPU (elementwise fp32, speed only). **S1-G1 is unchanged**
  (per-row cos ≥ 0.9999); the 11 failed videos are re-extracted under the fix and the gate re-read on all 563. The 552
  that passed are kept as extracted (they passed the signed gate; 105 of them read < 0.99999, max deviation 1e-4).
- **B9 (decided by the user 2026-10-28, AFTER the `nexar_val` S1-G1 read-out, disclosed) — R1 retrained, not borrowed;
  S1-G1 printed only.** With TF32 off, `nexar_val` read **FAIL**: 181 / 225, 44 videos `g1_cos` (min 0.99946, median
  min 0.99996). Diagnosis: identical under every TF32 / normalization setting, decode deterministic today, deviations
  confined to a few isolated rows (e.g. 1 row in `00055`, 5 rows 798–813 in `01146`) while clean videos read
  1.00000000 → a few frames decode differently today than when v2 built `Nexar_s1_squash` (environment drift; not
  reproducible without v2's frames). The 0.9999 bound was carried over from Stage 0, where it compared two products
  of one pass; across environments it is too tight. Changes: (i) **R1 = retrained s2024–2026 through the v5 path**, so
  R1 and every other row read the same grids and no v2 run is borrowed (S1-G3 dropped; +2 training runs);
  (ii) **S1-G1** fails a video only if its decoded length differs from v2's (a real alignment fault) and stores every
  other grid; the per-row cosine against v2 is printed (extraction is shown correct where environments agree:
  1.00000000); (iii) **S1-G2(b)** compares every row's motion term at init with R1's on the same grid instead of the
  baked A2 columns. S2-7's P1 = R5 − R1, P2 = R5Δ − R1 are unchanged; R0 stays v2 Nexar A0 (no grid involved).

### §2.3 Stage 2 reads and decisions

**S2-7 Detection (H1) — primary.** Set: DoTA-CAP-dev, protocol B (`s1[::3]`, whole clip, per-clip min-max, scores
interpolated to native frames, macro over two-class clips), Nexar A2 statistics applied. Two primary contrasts:
**P1 = R5 − R1, P2 = R5Δ − R1.** Statistic: cluster bootstrap over DoTA source videos (`dota_group`) of the per-clip AUC
difference on seed-averaged curves, B = 10,000, seed 2026, **97.5 % intervals** (Bonferroni over P1, P2).
Sentence "R improves DoTA-CAP-dev detection over mean pooling" iff (i) interval low > 0, (ii) per-seed paired Δ > 0 on
3 / 3 seeds, (iii) point ≥ +0.01. Interval high < 0 → "R costs". Otherwise "no difference shown at n = 3".
Paired t over seeds printed (t₀.₉₇₅,₂ = 4.30: too wide to gate at n = 3).

**S2-8 Maps (H2) — co-primary.** Set: the 528 boxed CAP-dev clips (Stage 0's D3 set). Map per step: R5/R5Δ/R7
`E_t(j) = Σ_m s_{t,m}·a_{t,m,j}`; R3 `a_{t,1,j}`; per seed, then hit rates averaged over seeds per clip. Reads: D3 hit −
chance and Amendment 1's lift − lift_pre (Stage 0's code, the map in place of `Δ`), and hit(R) − hit(centre prior), the
centre prior being a fixed isotropic Gaussian at the frame centre, σ = ¼ of the grid width (no labels used). **"R's
maps point at the agent"** iff all three 95 % intervals have low > 0. Printed beside: Stage 0's V7 change at ℓ = 6
(+0.076), the uniform map (R1; hit − chance ≡ 0), STAUC per row (S1-G5c), by ego and box-share bin.

**S2-9 Guardrails** (a row that fails one is not adopted, whatever S2-7/S2-8 say). `nexar_val` crop endpoint (D-N2)
`R − R1 ≥ −0.01` (point, seed mean); S1-G4 passes.

**S2-10 Decision.** On R5 first; R5Δ replaces R5 as LaGoVAD-MQR **only** if R5 has neither sentence and R5Δ has one
(disclosed as Stage-0-motivated).

| detection (S2-7) | maps (S2-8) | outcome | next |
|---|---|---|---|
| improves | point | **GO-MQR** | seeds 2027–2028 for the adopted row (+ R1 if retrained); proposal edited; Stage 3 |
| improves | do not point | **DETECT-ONLY** | H2 dropped from the claim; proposal edited; Stage 3 on the user's word |
| no difference / costs | point | **MAPS-ONLY** | claim = weakly supervised maps that point, at no detection gain; no Stage 3 for H1 |
| no difference / costs | do not point | **NULL** | MQR closed; v5 written up as a measured negative with Stage 0 |

R7 (if run) is read with S2-7's statistic as R7 − R0 and R7 − R5, printed only, answering one sentence: "do CLIP
patches carry what motion tokens carry here?" It never changes S2-10.

**S2-11 `nexar_test`.** Opened once, after S2-10 is written, for R1 and the adopted row (none under NULL): crop endpoint,
paired over seeds, printed only.

**S2-12 Cost (estimate, printed before the run).** Grids: `nexar_train` positives ≈ 451 × 150 rows (s8) + `nexar_val`
225 × 400 rows (s3) ≈ 158 k rows × 37.6 KB ≈ **6 GB** (Drive); one PyAV decode of 676 mp4s + V2-S on 158 k windows
(a fraction of v2's N2). Training: R3, R5, R5Δ × 3 seeds + S1-G3 = **10 runs** of 1,749 steps; input I/O ≈ 50× v2's
per row. R7 adds **3 runs + a CLIP-patch cache** (CLIP re-run on the same rows + CAP-dev's 19 k rows ≈ 177 k ×
49 × 512 × 2 B ≈ **9 GB**).

**S2-13 Deviations from the proposal.** D1 v2's motion path (fixed `c/σ_u`, zero-init `W_u`) instead of
`tanh(α)·W·LN(r)`: R1 is then v2's measured A2. D2 no search over λ, M, LR multiplier (proposal: 18 configs on a held-out
15 %); fixed λ = 0.5, M = 8, ×1 (lesson 14). D3 Nexar-only, 3 seeds (user, 2026-10-26). D4 causal window (L-3). D5 no CRN
(v2 §29–§31). D6 rows 2/4/6 and side ablations not run. D7 R5Δ added (check L). D8 DoTA read on DoTA-CAP-dev (1,129-clip
CAP alignment; DoTA pixels are gone), never beside LaGoVAD's 62.60. D9 DVS anchor `span` (A2's recipe; v2's
`ignore` read −0.006 ns on Nexar in-domain).

**S2-14 Printed, never gating.** By ego and box-share bin (S0-12's confound holds); `p_T2` 0.822 and v2 Nexar A2's
0.8171 beside every DoTA-CAP-dev macro; v2's beyond-position `f_2` for P1/P2; `motion_share`, `‖W_u‖`, query
entropy and query-map overlap; `nexar_val` whole-video macro; R7 rows. **Open before signing (user):** (a) R7 yes/no;
(b) λ = 0.5, M = 8 fixed; (c) DVS `span` (D9).
