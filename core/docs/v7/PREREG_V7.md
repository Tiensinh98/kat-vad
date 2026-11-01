# KAT-VAD v7 (LaGoVAD-LSP) — pre-registration

Proposal: `core/docs/final/Thesis proposal Deep learning-based traffic video anomaly detection.md` (2026-11-01, §3.2–§4.5).
Plan: `.project/plans/katvad-v7-lsp.md`. Each section is frozen **before** the number it governs is read. A defect
found later becomes a numbered amendment, never an edit (v2 convention, `core/docs/v2/PREREG_ADDENDUM.md`).

v7 = **v5** (LaGoVAD-MQR, `core/docs/v5/PREREG_V5.md`) with MQR replaced by the Latent Surprise Predictor. MQR was
measured as giving nothing (v5 Stage 2 / 2b / v5.1 / R7). v7 has no relation to v6. User decisions (2026-11-01):
Nexar-only training; causal window (a); a fixed 10 fps lattice for the predictor (b); `nexar_val` as the hold-out (c).
**Where proposal and code disagree, the code wins**, with one exception: the e_t fusion uses the proposal's tanh gate.

---

## §1 Step 0 — does a normal-dynamics predictor in V2-S space beat fixed rules? (L0-1 … L0-12)

Signed 2026-11-01, before any v7 number exists. No detector is trained.

**L0-1 Vectors.** V2-S (`vit_s_k710_dl_from_giant`) `fc_norm(mean tokens)`, causal 16-frame window ending at the row,
one row per native frame. The caches already exist: `Nexar_s1_squash` (all 1,500 videos, 30 fps) and
`DoTA_CAP_s1_squash` (10 fps), both under `Thesis-V2/cache/video/vit_s_k710_dl_from_giant/`. No extraction.
*Deviation (X4):* the proposal centres the window. It is kept causal because that is the window v5 R1, the host, read.

**L0-2 Unit vectors.** `u = (v − b) / γ`, where `γ, b` are V2-S's `fc_norm` from the pinned weights. This equals the
proposal's `LN_no-affine(mean tokens)` exactly (`core/models/lsp.py:unit_vectors`). *Deviation (X5):* none in effect.
The cache stores `fc_norm` applied, and the inversion is exact.

**L0-3 Lattice (X6).** The predictor reads a 10 fps lattice of one source video. Nexar: native frames `f, f − 3, …`, in
all three phases (`V7_NEXAR_LATTICE_STEP` = 3). DoTA: native frames (step 1). Horizon **Δ = 16 lattice steps (1.6 s)**.
A causal window spans 15 gaps, so the target clip shares no frame with the last context clip. K ∈ {16, 32}.

**L0-4 Data.** Predictor training = the 404 `nexar_train` negatives that remain after the hold-out. Hold-out = **45**
negatives (10 % of 449), drawn with `default_rng(2024).permutation(sorted negatives)[:45]` and frozen in
`core/splits/v7/lsp_holdout.txt` (sha1 `2d1c5a0d5e847f8956c18e8f140c102f80801946`). The tool refuses a file that differs
from the draw. Positives, `nexar_val`, `nexar_test` and every DoTA clip are never trained on. *Deviation (X1):* Nexar
only, no CCD.

**L0-5 Predictor.** As proposal §3.2.2: 2 pre-LN causal transformer layers, width 384, 6 heads, MLP 4×, GELU, no
dropout; learned positions over K (trunc-normal 0.02); final LN; linear readout of the last position,
**zero-initialised**, so `û_t = u_{t−Δ} + g(u_{t−Δ−K+1..t−Δ})` starts as copy-last. Measured size: **3.70 M** parameters
(the proposal says about 3.5 M; the gap is the readout and the positions).

**L0-6 Training.** Smooth-L1 (β = 1), mean over 384 dims, targets are constants. AdamW, lr 3e-4, weight decay 0.05,
cosine over all steps, 50 epochs, batch 256. One epoch = 32 random targets per (video, phase) sequence. Seed 2024.
The checkpoint kept is the epoch with the lowest held-out L_rec, scored on every target `t ≥ 47` of the 45 hold-out
negatives (`V7_HELDOUT_FIRST_TARGET` = Δ + max K − 1; the same targets for every K and every predictor).

**L0-7 K.** Both K are trained. The K with the lower held-out L_rec wins. If the relative gap is below 1 %, K = 16.
ŝ is standardised with the mean / std of `s` over every valid target of the 45 hold-out negatives, at the chosen K.

**L0-8 Evidence.** `s_t = mean ℓ(u_t − û_t)`, `p_t = mean ℓ(u_t − u_{t−Δ})`, `ρ_t = log((s_t + 1e-6)/(p_t + 1e-6))`.
A lattice step is valid iff `t ≥ Δ + K − 1` of its own source video. Invalid steps are not read.

**L0-9 D1 (run before training).** `cos(u_t, u_{t+k})` for k = 1…40, pooled over the 404 training negatives (phase 0).
**D1 fails iff cos at k = Δ ≥ cos at k = 1 − 0.01** (no decorrelation within 1.6 s, so copy-last cannot be beaten).
The curve is printed.

**L0-10 D2.** Held-out L_rec on the L0-6 targets for: the LSP (chosen K), copy-last `u_{t−Δ}`, constant velocity
`2u_{t−Δ} − u_{t−2Δ}`, and the context mean. **Pass iff LSP ≤ 0.90 × the best fixed predictor.**

**L0-11 D3.** (a) `nexar_val` (225 videos, both labels), phase 0: the video-level AUC of `max_t s_t` (identical ranking
to ŝ) against `max_t p_t`. Paired video bootstrap, 10,000 resamples, 95 %, seed 2026; ρ printed.
(b) DoTA-CAP-dev (569; `dota_cap_eval` not read): the per-clip frame AUC of `s` and of `p` on the valid frames of clips
with both classes among them. The paired per-clip difference gets a cluster bootstrap (source video), 10,000 resamples,
95 %. Printed beside it: the cubic position prior (fitted on these clips' labels, read on the same frames), ρ, and the
ego / non-ego split. **Pass iff AUC(s) > AUC(p) on `nexar_val` (point) and the mean DoTA difference > 0 (point).**
*Deviation (X10):* the proposal's DoTA train split has no V2-S vectors in this project.

**L0-12 D4, D5 (printed, never gating).** D4: a 5-fold logistic classifier (standardised) separates `nexar_val` normals
from DoTA-CAP-dev lead-ins (valid steps before the anomaly onset, at least 8 of them). Features: mean / std / max of `s`
and mean / max of ρ. Source AUC − D3(a)'s AUC(s) > 0.2 flags the median-subtracted ŝ variant for P2. *Deviation
(X11):* no CCD. D5: the share of DoTA-CAP-dev clips whose maximum `s` (and `p`) falls before the onset, and the Pearson
correlation of `s` and `p` over `nexar_val` normal steps.

**Decision.** **GO** iff D1 not failed ∧ D2 pass ∧ D3 pass. Otherwise **NO-GO**: record it and stop. The V-JEPA 2
control and any change of K, Δ or data become a separate, user-approved amendment.

**Tooling.** `python -m core.tools.v7_lsp {d1,train,diag}` (`core/tools/v7_lsp.py`), model `core/models/lsp.py`,
tests `core/tests/test_v7_lsp.py`, notebook `colab/v7/lsp_step0.ipynb`. Outputs:
`Thesis-V7/outputs/v7/v7_lsp/step0/{models,REPORTS}/`.

---

## §1.1 Amendment 1 — Step 0 re-run on CLIP frame vectors (L0C-1 … L0C-5)

Signed 2026-11-01. **Post-hoc and disclosed:** written after §1 read NO-GO on V2-S
(`core/docs/v7/RESULTS_V7_STEP0.md`) and after Claude printed one unregistered look at CLIP on DADA lead-ins
(s8 rows, no training): cos 0.967 → 0.943 at 1.6 s; context-mean 0.0450 vs copy-last 0.0539 at Δ = 1.6 s. That look was
on DADA, not Nexar, and chose nothing below. Authorized by the user ("ok. làm 1 tới 3 đi"); not advisor-reviewed. §1's
verdict on V2-S is unchanged by anything here.

**L0C-1 Vectors.** CLIP ViT-B/16 image embeddings, one per native frame, `no_center_crop`: `Nexar_s1_ncc`
(`Thesis-V2/cache/clip/`) and `DoTA_s1_ncc` (`Thesis/cache/clip/`, which covers DoTA-CAP-dev). These are the `c_t` the
host already reads. No extraction. `u = LN_no-affine(c)` (`core/models/lsp.py:clip_unit_vectors`), the L0-2 rule applied
to a vector that has no `fc_norm`.

**L0C-2 Everything else is §1's.** The lattice (L0-3: Nexar step 3, DoTA step 1, Δ = 16 = 1.6 s, K ∈ {16, 32}), the data
and the frozen hold-out (L0-4), the training recipe (L0-6), the K rule (L0-7), the evidence (L0-8), and every
diagnostic, bar and the decision (L0-9 … L0-12) are unchanged. Δ stays 1.6 s even though a CLIP vector has no window,
so the two streams answer the same question.

**L0C-3 Predictor width.** As L0-5, with width 512 and **8 heads** (head width 64, as V2-S's 384 / 6); about 6.6 M
parameters.

**L0C-4 Outputs.** `--stream clip` writes `Thesis-V7/outputs/v7/v7_lsp/step0_clip/`. §1's `step0/` is never touched.

**L0C-5 Reading.** GO on CLIP licenses only a P2 design on CLIP surprise. That P2 would need its own amendment, because
the host already reads `c_t` (proposal §3.2.4's objection is stronger there). NO-GO on CLIP is recorded beside §1's
NO-GO as one statement about frame/clip-global vectors on Nexar.
