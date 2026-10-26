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
