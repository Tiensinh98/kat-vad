# KAT-VAD v2.2 — order diagnostics S1–S3 (no trunk training)

**Branch:** `v2`. **Design:** `core/docs/v2.2/KAT_VAD_PROPOSAL_v2.2.md` §3–§6, §9–§10 and
`core/docs/v2.2/KAT_VAD_v2.2_ARCHITECTURE.md` §2–§5 (after review round 4, 2026-10-11).
**Status:** BUILT 2026-10-11; **RUN 2026-10-13: S1a PASS · S2 PASS · S3 VALUE FAIL · ORDER FAIL → TDP not built, S5 closed.**
Claim-ladder row "S2 passes, S3(a) fails". Results: `core/docs/v2.2/RESULTS_V22_ORDER.md`; App. A below.

## 0. Status

| Step | What | Training? | Gate (proposal §6.1 bar: 95 % low > 0 **and** mean ≥ +0.01) | Status |
|---|---|---|---|---|
| S1a | `m5` change score vs 0.5 and vs its quantile-matched twin, DoTA-CAP-dev | none | both Δs pass → TDP branch open; fail → S3/S5 closed, **S2 still runs** | **PASS** (+0.250 / +0.271) |
| S1b | Δu echo, equivalence ±0.1 SD | none | CI inside ±0.1 SD → Δu admitted to κ; else CLIP only | **not admitted** (CLIP only) |
| S1c | clip-start SMD on 5 summary channels | none | any \|SMD\| > 0.1 → H6 prediction withdrawn (never stops) | **withdrawn** |
| S2 | temporal probe, true order vs adjacency-free block shuffle, on `F0 = [x̃; ũ; p]` | probe only | bar on DoTA-CAP-dev; fail → stop (§9 row 1) | **PASS** (+0.0283) |
| S3 | probe on `F0` / `F0+K` / `F0+K^π` | probe only | VALUE and ORDER both pass → S5 eligible | **VALUE FAIL, ORDER FAIL** → S5 not eligible |
| advisor | outcome table §10 | — | before S4/S5 | **next** (§10 row 2: chapter + S4 yes/no) |

## 1. Files

| File | Role |
|---|---|
| `core/tools/order_features.py` | pure numpy: κ pipeline (arch §3), adjacency-free block shuffle (§2), far-neighbour ordering + quantile-matched twin (§4), fitted statistics (`σ_Δ`, `Q`, `μ_κ`, `σ_κ`) |
| `core/tools/order_probe.py` | torch: the fixed-capacity temporal probe (arch §5), window builder with 4 context steps each side, seed-averaged scoring |
| `core/tools/order_diagnostics.py` | CLI `s1` / `s2` / `s3`: loading, gates, rules, read-outs (`s{1,2,3}_readout.{json,md}`) |
| `core/constants.py` | `V22_*` block |
| `core/tests/test_order_features.py`, `test_order_probe.py`, `test_order_diagnostics.py` | unit + synthetic end-to-end |
| `colab/v2/v22_order_s1_s3.ipynb` | runbook (CPU is enough; GPU only speeds S2/S3) |

## 2. Choices made in code (fixed before any number; O-numbers)

| # | Choice | Why |
|---|---|---|
| O1 | Probe training rows = every window of `labels_train.json` in the v2 data dir (T2-train − T2-val), 20 steps + 4 real context steps each side from the source, replicate padding only at a source's true ends | arch §5 |
| O2 | Step labels = `kill_switch_probe.source_labels` (the K / E2(d) harness); every training window is checked against `v2_dataset.window_frame_labels` — a mismatch raises | one label arithmetic |
| O3 | T2-val read = **per source** (whole source, s8, two-class sources), cluster = source. Not T2-val windows: shuffling is per source, and a per-sequence AUC is invariant to where its steps sit | reported only |
| O4 | `x̃ = x − R2(x)`, `ũ = u − R2(u)` per source / clip; `s`, `c`, `σ_u` not applied (the probe's standardisation absorbs per-channel affine maps, E2(d) N2) | as E2(d) |
| O5 | `p = [τ, τ², τ³]` of the step's **original** index relative to the scored sequence: the T2 window during training (`τ = (i − start + 0.5)/20`, context steps fall outside [0, 1]), the clip / source at test. Under the shuffle `p` moves with its step | arch §0, §2 |
| O6 | Probe loss = BCE with `pos_weight = n_neg / n_pos` of the training steps (the earlier logistic probes are `class_weight="balanced"`) | parity with K / E2(d) |
| O7 | Probe seeds = `V2_E3_SEEDS`; scores averaged (logits) over seeds before every metric; per-seed Δs printed | proposal §6.1 |
| O8 | `f_2` at step level: `p_T2` = `position_prior`'s cubic fit on the v2 dir's training windows, scored on `position_features(n_steps)` per clip; `f_2 = z(score) + 2·z(p_T2)` within clip (`final_readout.zscore`) | same prior as E3/Final |
| O9 | Per-sequence seeds for the shuffle and the twin = sha1(arm tag, sequence id) → stable across machines (never Python `hash`) | reproducible |
| O10 | Sequences with < 3 steps get no κ: excluded from S1/S3 and counted | acceleration needs 3 |
| O11 | S1b: clips whose last abnormal step `e` has `e + 15 < n`; per-clip Δ = mean `s^u[e+5..e+6]` − mean `s^u[e+10..e+15]`; SD = pooled std of the baseline steps; admit iff the cluster CI ⊂ [−0.1 SD, +0.1 SD]. The same read on `s^x` is printed | proposal §5 |
| O12 | S1c: clips whose steps 0–2 are all normal; steps 0–2 vs normal steps ≥ 3 of the same clips, pooled; pooled-SD SMD | arch §3 |
| O13 | `σ_Δ`, `Q` (top-64 right singular vectors of `Δx̄`) and `μ_κ`, `σ_κ` fitted on every step of the T2-train − T2-val **sources** | arch §3, §8 |
| O14 | S3 corpus shortcut (printed): 5-fold grouped logistic DADA-vs-DoTA AUC on the standardised κ rows (rate-normalised; the un-normalised variant is not built) | proposal §7.3 |
| O15 | DoTA-CAP-dev only; `dota_eval` / `dota_cap_eval` never loaded | sealed |

## 3. Run order (notebook)

1. setup → preflight (splits sha1, v2 data dir manifest, CLIP + V2-S caches for every T2 source and every CAP-dev clip)
2. stage caches to VM disk
3. `order_diagnostics s1` → `s1_readout.md` (decides: TDP branch open?, Δu admitted?, H6 kept?)
4. `order_diagnostics s2` → `s2_readout.md` (decides: stop or continue)
5. `order_diagnostics s3` (only if S1a and S2 PASS; the tool refuses otherwise)
6. paste the three read-outs to Claude → App. A, then the advisor check (proposal §10)

## Appendix A — read-outs

### A.1 Run 2026-10-13 (Colab, `outputs/v2/REPORTS/v22_order/`)

Full tables and interpretation: `core/docs/v2.2/RESULTS_V22_ORDER.md`. DoTA-CAP-dev decides; T2-val reported.

| Step | Contrast | Δ [cluster 95 %] | `f_2` Δ | T2-val Δ | Seeds | Verdict |
|---|---|---|---|---|---|---|
| S1a | AUC(`m5`) − 0.5 | +0.2503 [+0.2330, +0.2669] | — | +0.194 | — | PASS |
| S1a | AUC(`m5`) − AUC(`m5^π`) | +0.2713 [+0.2537, +0.2891] | — | +0.198 | — | PASS |
| S1b | Δu echo (margin ±0.638) | +2.877 [+1.418, +4.362] | — | — | — | not admitted |
| S1c | max of 5 summary \|SMD\| | 0.395 (`v̄`) | — | — | — | H6 withdrawn |
| **S2** | `F0` true − shuffled | **+0.0283 [+0.0212, +0.0356]** | +0.0138 [+0.0103, +0.0174] | +0.0165 [+0.0060, +0.0267] | 5/5 + | **PASS** |
| S2 (printed) | `[x̃;p]` true − shuffled | +0.0119 [+0.0015, +0.0230] | +0.0087 | +0.0252 | 5/5 + | — |
| **S3(a)** | `F0+K − F0` | **−0.0007 [−0.0063, +0.0047]** | +0.0006 ∋ 0 | −0.0004 | 2+/3− | **FAIL** |
| S3(b) | `F0+K − F0+K^π` | +0.0080 [+0.0021, +0.0140] | +0.0019 ∋ 0 | −0.0034 | 3+/2− | FAIL (point < 0.01) |

Absolute: AUC(`m5`) 0.750 (twin 0.479); `F0` true 0.853 / shuffled 0.825; κ shortcut AUC 0.905; corr(`m5`, `p_T2`) −0.053.

**Caveats recorded, not acted on** (the user declined further notebooks, 2026-10-13): (a) the quantile-matched twin is
**below chance**, so the S1a "beyond atypicality" control removed nothing measurable; (b) true-order `F0` 0.853 ≈
E2(d)'s cubic `p` alone 0.847 (different probe class), and the same-architecture `p`-only reference was not built;
(c) `m5` is low at clip starts (SMD −0.35), a small position-like effect; (d) κ replicate padding did not remove the
clip-start spike (61/131 channels > 0.1); (e) S1b cannot separate echo from post-accident change; (f) read-outs omit
`p_T2` beside the macro (§6.1).

**Decision:** claim-ladder row "S2 passes, S3(a) fails". TDP not built; S5 closed; deployed network = A3-ign.
Next = advisor check (proposal §10 row 2): analysis chapter acceptable? S4 (10 runs) yes/no?
