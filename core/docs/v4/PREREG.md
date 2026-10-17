# KAT-VAD v4 (ReSH): pre-registration

**Status:** frozen 2026-10-15, **before** any v4 number exists (no head feature has been extracted).
**Proposal:** [`KAT VAD v4 architecture proposal.md`](KAT%20VAD%20v4%20architecture%20proposal.md) revision 3. Where the
two differ, this file wins. **Plan:** `.project/plans/katvad-v4-resh.md`. **Branch:** `v4` (from `v2@5469bfd`).

**Rules of this file.**
- It is committed before Stage A runs. A rule found defective after a number is read becomes a numbered
  **Amendment** at the end of this file, authorized by the user and marked "not advisor-reviewed"; it never edits a
  rule in place.
- A post-hoc read is "printed, not gated" and never changes a verdict.
- Sealed: DoTA-eval and DoTA-CAP-eval (`SealedSplitError`) until §6; T2-test until §6. Stage A reads T2-train and
  T2-val only.
- Every constant named here lives in `core/constants.py` under `V4_*`; the code reads it from there.

---

## 1. Fixed inputs (V4-I)

| ID | What | Value |
|---|---|---|
| I1 | Encoder | VideoMAE V2-S `vit_s_k710_dl_from_giant`, frozen, pinned sha256 (`constants.VIDEOMAE_WEIGHTS_SHA256`); 12 blocks × 6 heads × 64-d = **72 candidate heads** |
| I2 | Tap | input of each block's `attn.proj` (heads merged, channel = head·64 + d), mean over the 1,568 tokens; head index = block·6 + head |
| I3 | Geometry | v2's: causal 16 frames every 3rd raw frame (1.5 s at the assumed 30 fps), squash 224², ImageNet normalization; T2 at stride 8, row-aligned with `cache/clip/DADA2000_orig` |
| I4 | Row check | in the same pass, the pooled `u` vs the existing V2-S cache row: **per-row cosine ≥ 0.999** (`V4_U_COS_MIN`, as DF8 X10(c)). Any row below → selection refuses to run |
| I5 | Splits | T2-train = T2 train sources minus T2-val (`core/splits/v2/t2_val_sources.txt`; 1,491 − 219 = 1,272 sources); T2-val = 219 sources. T2-test is never read before §6 |
| I6 | CRN | R2 (per-dimension median) over the source video (T2) or the clip / crop (DoTA, DF8), as v2 |

## 2. Head selection (V4-S), Stage A, T2-train only

| ID | Rule |
|---|---|
| S1 | **Window label** = abnormal iff `positive_frames > 0` (the T2-val freeze's rule), T2-train windows only. |
| S2 | **Step label (weak).** A step is labelled only if every T2-train window covering it has the same label. Steps covered by windows of both labels, and steps outside every window, are **not scored** (they still enter S3's fit). |
| S3 | **Detrend.** Per source and channel, least-squares fit of a cubic in τ = (t + 0.5)/T over all steps of the source (intercept included); keep the residual. Degree `V4_DETREND_DEGREE = 3`. |
| S4 | **Raw (diagnostic) representation** = the source's R2-CRN rows (I6). |
| S5 | **Folds.** T2-train sources, sorted, permuted with seed `V4_SEED = 2024`, split into `V4_SELECT_FOLDS = 5` near-equal disjoint folds (`numpy.array_split`). |
| S6 | **Per-fold sample.** Labelled steps of the fold's sources; the majority class is subsampled to the minority count, capped at `V4_SELECT_MAX_PER_CLASS = 4000` per class; the same steps for all 72 heads. |
| S7 | **Measures per head** on its 64-d features: (a) LDA separability `Δμᵀ S_w⁻¹ Δμ`, `S_w` = Ledoit-Wolf covariance of the class-centred pooled sample; (b) symmetrised Gaussian KL, Ledoit-Wolf covariance per class; (c) biased MMD² with an RBF kernel `exp(−d²/median(d²))`, median over the pooled sample's off-diagonal pairs, on at most `V4_MMD_MAX_PER_CLASS = 1000` steps per class (a seeded subsample of S6's); (d) NMI between K-means (K = 2, `n_init = V4_KMEANS_N_INIT = 4`, seeded) and the labels. |
| S8 | **Score.** Each measure min-max normalised across the 72 heads within a fold (all-equal → 0); `S(k, m)` = their mean. |
| S9 | **Rank.** `RSS(k) = mean_m S(k, m) − 0.5 · std_m S(k, m)` (population std, `V4_RSS_LAMBDA = 0.5`). **Selected = top `V4_HEADS_K = 5`** by RSS on the detrended representation; ties by lower head index. |
| S10 | **Stability floor (gating).** Each fold's top-5 by `S(·, m)`; the mean Jaccard over the 10 fold pairs must be **≥ `V4_STABILITY_MIN_JACCARD = 0.25`**. Chance for two random 5-of-72 sets is **0.0397** (exact hypergeometric, printed). |
| S11 | **Best block** = the block with the highest mean RSS over its 6 heads; `ẽ_full` = its 6 heads (384-d). |
| S12 | **Random heads.** From the 67 non-selected heads, seeded: one **A4-rand** draw of 5 (`V4_SEED + 1`) and **20** G1a draws of 5 (`V4_SEED + 2`, `V4_RAND_DRAWS = 20`). |
| S13 | **Printed only:** the raw (S4) ranking, its top-5, shared heads and Jaccard with the selected set, Spearman ρ of the two RSS vectors, the per-fold top-5 sets. |
| S14 | **Commit.** `head_selection.json` (selected ids, A4-rand, the 20 draws, best block, stability, all RSS) is committed to git **before** any G1a probe is fitted; G1a refuses to run unless the file's sha256 is passed and matches. |

## 3. Probe gates (V4-G), no training

**Probe convention (G0).** The standardised, class-balanced logistic probe of `core.eda` (`transfer_scores`), fitted
on every step of the T2-train sources with **frame labels from the annotation span at s8** (K's convention — the probe
measures feature content, it is not the model). Features are CRN'd per source / clip / crop (I6). `p` = the cubic
position ruler of K-pos (`position_features`, degree 3). Metric = **macro**: the mean per-source (T2) or per-clip
(DoTA) frame AUC over two-class items. CIs = cluster bootstrap (T2: clusters = sources; DoTA: source videos), level
0.95, `V4_BOOTSTRAP = 10000` resamples, seed `V4_SEED`.

### G1a — Stage A, T2-train → T2-val

| ID | Rule |
|---|---|
| G1a.1 | Sets: `[ẽ; p]` (selected), `[ẽ_full; p]`, `[ẽ_rand,j; p]` for the 20 draws, `[ũ; p]`, `p`. |
| G1a.2 | **PASS iff all three:** S10 holds; macro`[ẽ; p]` > macro`[ẽ_full; p]`; macro`[ẽ; p]` > the median of the 20 draws' macros (point estimates). |
| G1a.3 | Printed beside the verdict: the paired per-source Δ with its cluster CI for (selected − full) and (selected − the median draw); a margin below the CI half-width is **flagged**, not failed. |
| G1a.4 | Printed only: `[ẽ_raw; p]` (raw selection); `[x̃; ũ; ẽ; p]` − `[x̃; ũ; p]`; each selected head's `[ẽ_k; p]` − `p`. |
| G1a.5 | FAIL → stop V4 at Stage A, report the bounded negative. |

### G1b — Stage B, T2-train → DoTA-CAP-dev (protocol B)

| ID | Rule |
|---|---|
| G1b.1 | Δ = macro`[x̃; ũ; ẽ; p]` − macro`[x̃; ũ; p]`, paired per clip, cluster bootstrap over source videos. |
| G1b.2 | **PASS iff** Δ mean ≥ **+0.01** (`V4_G1B_MIN_DELTA`, report-H3's kill threshold) **and** the CI lower bound > 0. |
| G1b.3 | Printed: the same Δ for `ẽ_full` and the median `ẽ_rand` draw; `p` alone. |
| G1b.4 | Prior odds (proposal §7.3): expected to fail more often than not. A fail costs no training. |

### G1c — Stage B, DF8 crops (early warning for X7)

| ID | Rule |
|---|---|
| G1c.1 | The two G1b probes **without `p`**: `[x̃; ũ; ẽ]` and `[x̃; ũ]`, fitted on T2-train. |
| G1c.2 | Scored on DF8's 491 CAP-dev clips (`core.tools.position_shift`, frozen `df8_crops.json`), `control` and `early` crops; CRN over each crop's own rows; crop-boundary windows re-extracted with the tap (I2). |
| G1c.3 | Probe X7 = macro(early) − macro(control); ΔX7_probe = X7(with heads) − X7(without), paired per clip, cluster bootstrap over DF8's 86 source videos. |
| G1c.4 | **FAIL iff** the CI upper bound of ΔX7_probe < 0. |

**Stage C starts only if G1a, G1b and G1c all pass.** If G1 fails, the optional fallback is A4-S (proposal §7.2);
whether its 5 runs go to A4-S or to SC-2 is the advisor's call, recorded as an Amendment.

## 4. Training arms (V4-T), Stage C–D

| ID | Rule |
|---|---|
| T1 | Recipe = E4-0's A3-ign (AdamW, lr 5e-5, batch 64, 1,740 steps, `loss.dvs_anchor_mode=ignore`), seeds 2024–2028 paired; no seed added after a result is seen. |
| T2 | Arms: A4-H (heads, no steering), A4 (ReSH), A4-rand (S12's draw, Stage D only). No A4-static (absorbed by `W_e`). |
| T3 | ReSH is built inside `torch.random.fork_rng` with its own seed. Asserts per seed: the trunk's step-0 `state_dict` bitwise equal to E4-0's; the first batches' item indices equal; with `resh.enabled=false`, CAP-dev scores reproduce E4-0 within GPU nondeterminism. If the third fails, A3-ign is retrained (+5 runs). |
| T4 | Frozen on T2-train before Stage C: `σ_e`, `m_r`, `s_r`, PCA-32 (whitened, explained variance logged); `r̂` baked as per-step columns. |

## 5. Decision (V4-D), on DoTA-CAP-dev

| ID | Rule |
|---|---|
| D1 | Primary metric: DoTA-CAP (n/1397) dev macro, protocol B. `f_2` is printed, always as "`f_2` (DoTA-CAP placement)". |
| D2 | **Bar** (= v2 addendum §25 Y5–Y7): (a) paired t95 lower > 0; (b) clip cluster-bootstrap lower > 0; (c) paired mean ≥ +0.01; (d) `f_2` point ≥ 0; (e) candidate's T2-val macro ≥ A3-ign − 0.01; (f) O1′ vs A0. |
| D3 | **X7 non-inferiority** on DF8 (491 clips): ΔX7 = [early − control](arm) − [early − control](A3-ign); paired-t95 **and** cluster lower bounds > **−0.01**. Calibration printed beside it (proposal §8.2: no adopted v2 step passes). |
| D4 | **Fixed sequence**, each at 5 %, each only if the earlier passed: (1) A4-H − A3-ign: D2(a)–(f) + D3 for A4-H; fail → V4 is a bounded null. (2) A4 − A4-H: D2(a)–(d) on this contrast + D2(e)(f) and D3 for A4 vs A3-ign; pass → **V4 = A4**, fail → **V4 = A4-H**. (3) A4 − A4-rand: D2(a)–(d); licenses the selection sentence only. |
| D5 | Context control (decides nothing): A4 scored with `r̂` permuted across clips (10 permutations per seed) and with `r̂ = 0`; own − permuted with paired t95 and cluster CI. Licenses "the steering uses the clip's own reference". |
| D6 | Claim checks printed for every arm: source-shortcut AUC and position-probe R² on `V^t` vs A3-ign's; `s`, `‖g‖` on T2 vs DoTA-CAP; `ρ_e`. |

## 6. Final read

One pre-registered read of the adopted arm paired with A3-ign: CAP-eval (**third look**), and DF8's eval counterpart
only if its crop list is built with the same tool and frozen before any V4 run. T2-test is read once, with the micro
floor 0.6128.

## 7. Wording

Proposal §6 is binding: never "motion", "kinematics-aware", "position-invariant", "tuning-free", "fixes non-ego
dilution"; never "beyond position" without "(DoTA-CAP placement)"; never "context matters" from A4 − A4-H alone; a
failed test is "not detectable", never "no effect".

---

## Amendments

*(none)*
