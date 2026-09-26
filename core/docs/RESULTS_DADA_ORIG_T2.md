# KIP v1 on DADA-2000 original / T2: the paired A/B, its diagnosis and the repair (branch `main`)

**Read out 2026-09-26.** This is the durable record for three campaigns on the same corpus,
all measured by `main`'s own code (KAT-VAD **v1**). `outputs/` is gitignored, so this file
and `.project/memory-bank/progress.md` are what survives.

| step | date | plan / runbook | raw artefacts |
|---|---|---|---|
| Phase 4: KIP-off trunk + KIP A/B on `flow/v1` | 2026-09-16/18 | `.project/plans/katvad-dada-original-corpus.md` · `colab/DADA2000Origin/phase_4.ipynb` | `outputs/REPORTS/DADA2000_orig_phase4/` |
| D1/D2: loss-scale diagnosis | 2026-09-20/21 | `.project/plans/katvad-kip-loss-scale-diagnosis.md` (App. A) | `outputs/v1/DADA2000_orig_diag_kip_loss_scale/` |
| **Phase 5: Option A (`flow/v2_zscore`)** | 2026-09-26 | `.project/plans/katvad-flow-zscore-option-a.md` (App. A) · `colab/DADA2000Origin/phase_5_zscore.ipynb` | `outputs/REPORTS/DADA2000_orig_zscore/`, `outputs/v1/DADA2000_orig_zscore/` |

**Verdict (pre-registered decision table, plan §6.3): "cost removed; KIP-v1 neutral on T2."**
Z-scoring the flow target removed the trunk capture: `rho` fell from **3.105 to 0.179**.
With it went the phase-4 cost, and KIP-on(v2) beats KIP-on(v1) on T2 micro by **+0.020**
(t95 excludes 0). KIP-on(v2) against KIP-off is **+0.0076, t95 [−0.0002, +0.0154]**,
3/3 seeds positive, but the interval includes zero. That is a **bounded null**: any
in-domain effect of v1 KIP on T2 is below ≈ 1.5 AUC points at n = 3. **Stop spending seeds on
v1 KIP.**

---

## 1. Setup

- **Corpus:** T2 = W=20 hop 8 windows cut from the DADA-2000 *original* accident videos,
  with negatives taken from inside those same videos (leak 0.5000, clip oracle **0.7037**).
  `score_head_kernel = 3`, `mil_topk_pct = 5`. Stage 1 + stage 2, 2,040 steps each.
  Seeds 2024/2025/2026.
- **Pairing:**
  - KIP-off arms are the phase-4 checkpoints, reused rather than re-run. With KIP off,
    `kip_rec` is never computed and flow is never read (`TestKipOffIsInert`).
  - Measured config diff between v2 KIP-on and KIP-off: `kip.enabled` and `loss.lambda_rec`
    (0.9953 vs 1.0). The flow dir is a CLI path and is recorded in `run_manifest.json`.
- **C24 applies to every KIP-on arm:** on `main` the gate MLP gets no gradient, so each
  KIP-on run is a **fixed ~50 % channel shift**. Nothing below is "motion-gated".
- Environment (phase 5): torch 2.11.0+cu128, transformers 4.56.2, `core_sha256`
  `4b8e93cc996094d7`.

## 2. Why phase 4 lost 0.013, in one table (D1/D2, `flow/v1`)

| quantity | value | reading |
|---|---:|---|
| global-mean MSE `V` on `e_O` | **31.64** | the target is in raw pixel units; `mag_max` holds 83.0 % of `E[s²]` |
| measured `kip_rec` | 11.62 → `R²` 0.633 | a good fit that *looks* catastrophic |
| `kip_rec` share of `total` | 92.6–93.0 % | ~32× an O(1) objective at `lambda_rec = 1.0` |
| `rho` at stage-2 end (encoder) | **3.105** | KIP moves the trunk more than the task does |
| `cos(g_kip_rec, g_task)` | **−0.0010** | orthogonal: it spends capacity, it does not fight |

Decision row `capture + orthogonal` → **Option A**: z-score the 23 raw stats with
T2-train-window moments, re-project with the **same** `M`, and derive `lambda_rec = 1/V`.
Lesson **C37**.

## 3. Phase 5 build gates (§6.1): all pass

| gate | measured | bar |
|---|---|---|
| G0 v1 `.npy` == `stats @ M` | 1491/1491 files, max\|err\| **0.0** | ≤ atol, every file |
| G0-c coverage | 1491 written / 1491 sources | equal, 0 non-finite |
| G1 train moments | pass | \|mean\| ≤ 1e-3, std ∈ [0.999, 1.001] |
| G2-a / G2-c `V_v2` | **1.0047** (round-trip 0.995) | ∈ [0.909, 1.111] (operative) |
| G2-b centred | pass | Z/V ∈ [0.99, 1.01] |
| G2-d between-item share | **0.349** (v1 0.488) | trip-wire 0.5 |
| G2-e item-mean oracle | MSE 0.654 → `R²_item-oracle` **0.349** | reported |

`lambda_rec = round(1/V_v2, 4) = ` **0.9953**. The population was 4,401 train windows /
88,020 frames, `train_ids_sha1` `dd53964f…`.

## 4. Phase 5 read-out (§6.3)

### 4.1 Mechanism

| id | s2024 | s2025 | s2026 | mean | bar / reading |
|---|---:|---:|---:|---:|---|
| **R-1** `R²_v2 = 1 − kip_rec/V_v2` (stage-2 end) | 0.290 | 0.271 | 0.280 | **0.280** | > 0 ✔ |
| **R-1b** `R²` magnitude block (dims 0–6) | 0.365 | 0.351 | 0.356 | **0.357** | |
| **R-1b** `R²` histogram block (dims 7–22) | 0.073 | 0.068 | 0.071 | **0.071** | **risk 1 is live** |
| **R-2** `rho_all` at stage-2 end | 0.152 | 0.187 | 0.199 | **0.179** | **< 1.0 ✔ capture removed** (predicted ≈ 0.5) |
| `rho_kip_rec` / `cos(g_kip_rec, g_task)` | 0.054 / −0.051 | 0.074 / −0.056 | 0.083 / +0.003 | | small, still near-orthogonal |
| `rho_kin` / `cos(g_kin, g_task)` | 0.091 / **+0.612** | 0.105 / **+0.646** | 0.107 / **+0.723** | | see §5.3 |

At stage-1 end `rho_all` is 0.159 / 0.148 / 0.163, against **11.63** on v1.

### 4.2 AUC (paired, n = 3, t95 with 2 df)

| metric | KIP-off | KIP-on v1 | **KIP-on v2** | Δ v2 − off (per seed) | **Δ v2 − off, t95** | Δ v2 − v1, t95 |
|---|---:|---:|---:|---|---|---|
| **R-3** T2 micro (primary) | 0.6182 | 0.6054 | **0.6258** | +0.0065 / +0.0111 / +0.0051 | **+0.0076 [−0.0002, +0.0154]** | +0.0204 [+0.0010, +0.0398] |
| **R-4** T2 macro | 0.6248 | 0.6046 | 0.6305 | −0.0046 / −0.0043 / +0.0263 | +0.0058 [−0.0384, +0.0499] | +0.0259 [+0.0160, +0.0358] |
| R-5 DoTA 0-shot micro | 0.5856 | 0.5914 | 0.6077 | +0.0191 / +0.0105 / +0.0367 | +0.0221 [−0.0110, +0.0553] | +0.0163 [−0.0365, +0.0692] |
| R-5 DoTA 0-shot macro | 0.6113 | 0.6071 | 0.6376 | +0.0263 / +0.0081 / +0.0444 | +0.0263 [−0.0189, +0.0714] | +0.0305 [−0.0257, +0.0866] |

R-5 is **reported, never decided on** (pre-registered). The v2 − v1 column was **not** a
pre-registered bar. It is descriptive and is what attributes the phase-4 cost.

### 4.3 Decision

`R-2 < 1.0` × `R-3 t95 includes 0` → **"cost removed; KIP-v1 neutral on T2"** → write a
bounded null and stop spending seeds on v1 KIP.

## 5. What this does and does not license

1. **Licensed: the phase-4 cost was the loss scale (C37).** Removing it moves T2 micro
   **+0.020** and T2 macro **+0.026**, and both intervals exclude zero.
2. **Not licensed: "KIP helps on T2".** The primary interval's lower bound is −0.0002.
   **Do not add a fourth seed to push it across zero**, because that is optional stopping.
   The decision table requires a *fresh* seed triple for any positive claim.
3. **Not licensed: "KIP learns motion dynamics".**
   - `R²_v2` = 0.280 sits **below** the item-mean oracle (0.349). In MSE terms, `kip_rec`
     ≈ 0.72 > `W_v2` = 0.654, so **D1-b (`K < W`), which passed on v1, fails on v2**. The
     v1 claim that the PMG head fits within-item structure (`R²_item` 0.283) was carried by
     `mag_max`. On the equal-weighted target, the head does worse than a predictor that
     knows only which clip it is looking at.
   - The direction block (69.6 % of the target) is fitted at `R²` **0.07**.
   - C24 still makes the shift a fixed ~50 %.
   - No `main` arm separates shift from losses. That needs A2 on T2, which exists only on
     branch `v3`.
4. **The `cos(g_kin, g_task)` ≈ +0.6–0.7 is by construction, not a finding.** `L_kin` is
   BCE on the top-k mean of the motion curve against the **video label**
   (`core/kip/losses.py:kinematic_loss`), i.e. a second MIL head. Stage 1 does not optimize
   `L_kin` (its `metrics.jsonl` logs only `kip_rec` and `kip_align`), and at stage-1 end the
   cosine is ≈ 0. Do not cite it as motion evidence.
5. **DoTA:** every v2 − off DoTA Δ is positive on 3/3 seeds, but the intervals are 2–4×
   their Δ. This is the fourth DoTA-sign-without-interval result in the project. Don't
   quote it.

## 6. Numbers to carry forward

- T2 in-domain, 3 seeds: KIP-off **0.6182 / 0.6248** (micro / macro), KIP-on v2
  **0.6258 / 0.6305**. The clip oracle is 0.7037, so macro ≥ micro holds on both arms and
  there is no C14 collapse.
- `flow/v2_zscore/DADA2000_orig`: `V` 1.0047, `lambda_rec` 0.9953. It is bound to this
  train split (`train_ids_sha1`).
- Best zero-shot DoTA from a T2-trained model: KIP-on v2, micro **0.6077** / macro **0.6376**
  (n = 3, sd 0.013 / 0.017). The MSAD KIP-on bar is 0.6408 / 0.6529. It is below that bar on
  both, and the macro gap (−0.015) is within one sd.

## 7. Interpretation: what the z-scored flow contributes, and what it does not

**It removes a harm; it adds no capability.**

| | flow v1 (raw) | flow v2 (z-scored) |
|---|---|---|
| `kip_rec` share of `total` (stage 2) | ~93 % | ~48 % |
| `rho` at the temporal encoder | 3.105: KIP steers the trunk | 0.179: the task steers the trunk |
| T2 micro vs KIP-off | −0.013 (t95 excludes 0) | +0.0076 (t95 includes 0) |

1. **Repair (measured).** v2 − v1 = +0.020 micro and +0.026 macro, and both intervals
   exclude zero. This decomposes as ≈ 0.013 recovered from v1's cost plus ≈ 0.008
   residual over KIP-off.
2. **The residual +0.0076 is not attributed, and should not be.** On `main` three sources
   are confounded:
   - the fixed ~50 % shift (C24), i.e. temporal smoothing, which is the whole MSAD +0.09;
   - `L_kin`, a second MIL head on the video label (its gradient cosine with the task is
     +0.6–0.7 by construction);
   - noise at n = 3.

   Attributing an effect whose interval includes zero spends seeds on nothing.
3. **Diagnostic (measured).** The rescale falsified a phase-4 reading. "PMG fits 28 % of
   within-clip flow" (`R²_item` 0.283) was carried by `mag_max`. On the equal-weighted
   target, the head does worse than a predictor that knows only which clip it is in
   (0.723 > 0.654), and it fits flow *direction* at `R²` 0.07. From frozen CLIP features,
   the PMG head learns **how much a clip moves**, not **how motion evolves inside it**.
4. **Methodological (transferable, C37).** An auxiliary regression target in raw units can
   capture a shared trunk silently: no error is raised and the loss keeps falling. It is
   detected by `V` (the constant-predictor MSE) and `rho` (`core.tools.grad_probe`), and
   repaired by standardizing the target and setting the weight to `1/V`, derived rather
   than swept.
5. **Observation, not a claim.** The seed sd of T2 macro is 0.0165 for KIP-off, 0.0055 for
   KIP-on v1 and **0.0016** for KIP-on v2. That is consistent with the repaired auxiliary
   losses acting as a regularizer. But the KIP-off spread is one seed (s2026, 0.6057), and
   n = 3 cannot compare variances.

**Thesis framing.** KIP v1 first hurt, because a raw-unit flow loss captured the trunk.
That was diagnosed with `V`/`rho` and repaired by normalization, after which KIP is neutral.
The same repair shows that global flow statistics are not learnable as within-clip dynamics
from frozen CLIP. With 23 frame-global scalars and a gate that never trains, **"optical flow
teaches the model motion" is not supported by this design.** What remains open is the
design (a spatial or region-level flow target; a trainable gate, branch `v3`), not seeds
or data.
