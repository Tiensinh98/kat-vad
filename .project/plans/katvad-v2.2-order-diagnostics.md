# KAT-VAD v2.2 — order diagnostics S1–S3 (no trunk training)

**Branch:** `v2`. **Design:** `core/docs/v2.2/KAT_VAD_PROPOSAL_v2.2.md` §3–§6, §9–§10 and
`core/docs/v2.2/KAT_VAD_v2.2_ARCHITECTURE.md` §2–§5 (after review round 4, 2026-10-11).
**Status:** BUILT 2026-10-11, not run. Read-outs go to App. A.

## 0. Status

| Step | What | Training? | Gate (proposal §6.1 bar: 95 % low > 0 **and** mean ≥ +0.01) | Status |
|---|---|---|---|---|
| S1a | `m5` change score vs 0.5 and vs its quantile-matched twin, DoTA-CAP-dev | none | both Δs pass → TDP branch open; fail → S3/S5 closed, **S2 still runs** | built |
| S1b | Δu echo, equivalence ±0.1 SD | none | CI inside ±0.1 SD → Δu admitted to κ; else CLIP only | built |
| S1c | clip-start SMD on 5 summary channels | none | any \|SMD\| > 0.1 → H6 prediction withdrawn (never stops) | built |
| S2 | temporal probe, true order vs adjacency-free block shuffle, on `F0 = [x̃; ũ; p]` | probe only | bar on DoTA-CAP-dev; fail → stop (§9 row 1) | built |
| S3 | probe on `F0` / `F0+K` / `F0+K^π` | probe only | VALUE and ORDER both pass → S5 eligible | built; **refuses to run unless S1a and S2 PASS** |
| advisor | outcome table §10 | — | before S4/S5 | — |

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

*(empty)*
