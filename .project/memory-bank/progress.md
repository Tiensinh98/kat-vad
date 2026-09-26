# Progress

**Last updated:** 2026-09-27 — **KAT-VAD v2 architecture designed and signed off by the advisor**
(`core/docs/v2/`, four review rounds). Design only: KIP + RAFT removed; frozen VideoMAE V2
motion stream (zero-init residual, fixed `σ_u`, `c`) + Clip-Referenced Normalization;
rate-matched DoTA evaluation; losses unchanged; 2 × 2 factorial × 5 seeds. **No code, no
measurement.** Next: freeze splits, then E0 → E0b + E1 → E2 (see "What's left"). Counts
re-measured: **91 Python files (61 source + 30 test), 14,155 source LOC, 598 tests, 30 docs**.

**Earlier 2026-09-26 —** step B (T2 learning curve) read out INCONCLUSIVE; Option A read out
"cost removed; KIP-v1 neutral" (ledger rows below).

**Earlier 2026-09-26 —** **BDD-A EDA run and read out: NO-GO** (G-X FAIL on `calm`,
X 0.5492, Δ(X−R0) −0.127; shortcut 1.000; G-M ≈ 0). Fourth separate-pool failure; C38
extended. `core/docs/BDDA_EDA.md` added (docs 25 → 26). T2 + Option A remain; Option A unrun.

**Earlier 2026-09-25 —** **D2City EDA run and read out: NO-GO** (G-X FAIL, Δ(X−R0)
−0.090; G-M fail; shortcut AUC 1.000). T2 stays; track 0b closed. `core/docs/D2CITY_EDA.md`
added (docs 24 → 25); lesson C38. Option A (below) is still unrun.

**Earlier 2026-09-24 (second pass) —** D2City normal-bag feasibility EDA planned; notebook
written and dry-run. No `core/` change — counts unchanged at `ef9c3c3`
(89 files, 13,833 source LOC, 582 tests).

**Earlier 2026-09-24 —** **Option A is AUTHORIZED (2026-09-23, construction A1),
BUILT and COMMITTED (`ae4fded`), and NOT YET RUN.** `core/flow/zscore_cache.py` rebuilds
`e_O` from the v1 cache's raw stats (no frames, no RAFT) into `cache/flow/v2_zscore/`,
scores the pre-registered build gates G0–G2 and derives `lambda_rec = 1/V_v2` (expected
≈ 1.0, **not** v1's 0.0316). Runbook `colab/DADA2000Origin/phase_5_zscore.ipynb` (P0,
P2–P4). Plan `.project/plans/katvad-flow-zscore-option-a.md`. Counts: **89 Python files
(60 source + 29 test), 13,833 source LOC, 24 docs, 582 tests**.

**2026-09-21 —** the KIP A/B on T2 is NEGATIVE and now DIAGNOSED.
KIP-on costs **0.0129** T2 micro (3/3 seeds, t95 excludes 0); D1/D2 attribute it:
`L_KIP_rec` enters the sum **~32x** oversized because `e_O` is unnormalized, and the
resulting gradient at the shared trunk is **3.1x** the task gradient while being
**orthogonal** to it (cos = -0.001). Decision row `capture + orthogonal` -> repair
**Option A** (z-score `e_O`, new cache version, `lambda_rec = 1/V = 0.0316` derived),
**not yet authorized**. New lesson **C37**.
(Earlier: Gate W at W=20/hop 8, the KIP-off trunk on three seeds and **T2 is the first
corpus in this project where in-domain training did not collapse into a clip/window
classifier**; Gate D0 0.6518; Phase 0 P1/P2/P3; the TAD campaign. Counts re-measured on
this tree: **86 Python files (58 source + 28 test), 13,276 source LOC, 24 docs,
565 tests**.)

> **Branch:** `main`, tip `ae4fded` (2026-09-23) — the **v1** line. The v3 gate rebuild is on
> branch `v3` (tip `bb1516c`), which has its own memory bank. See
> [[activeContext]] for the full v1-vs-v3 table.
>
> **Everything under "Headline" and "Evolution of decisions" was measured with
> the `v3` branch's code and is kept here deliberately** — `outputs/` is
> gitignored and `core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md` does not exist on
> `main`, so for this branch these paragraphs *are* the record. They are history,
> not a description of this tree's code.

## What works — verified in this tree

| Phase | Scope | Status |
|---|---|---|
| 0 | Env, `core/` skeleton, config system, device resolution, data-layout contract | ✅ |
| 1 | KIP modules — PMGFlowHead, KinematicShift (loop oracle + vectorized), MotionScoreHead, KIP wrapper | ✅ 26 tests |
| 2 | KIP losses — `L_KIP_rec`, `L_KIP_align`, `L_kin` | ✅ 20 tests |
| 3 | Baseline port — clip_text, temporal encoder, fusion, heads, MIL/DVS/contrastive losses, collate, definitions, `KATVAD` assembly | ✅ parity-tested |
| 4 | Data pipeline — MSAD preprocessor, download CLI, CLIP extractor, RAFT extractor, KNN cache, DVS dataset, synthetic fixtures | ✅ |
| 5 | Train / inference / evaluate / visualize CLIs, ckpt-compat loader, Colab guide, synthetic E2E incl. kill-and-resume | ✅ 10 E2E tests |
| — | MSAD **full** benchmark support (11 classes, not just the traffic slice) | ✅ `b9978ff` |
| — | **PreVAD preprocessor** (`core/data/prevad.py`) + 36-class v6 definitions | ✅ 2026-08-24, 38 tests |
| — | ~~**v3 KIP gate rebuild**~~ — 4 selectable `gate_type`s, ECMR, inference graph without 3e/3f, gate diagnostics in every score `.npz` | ✅ 2026-08-30 **on branch `v3` only — NOT in this tree** |
| — | **TAD adapter** — `--with-train-split` (P1) + `raft_extract --frames-dir` (P2) | ✅ 2026-09-02 (its 467-test figure is `v3`'s) |
| — | **DADA-2000 adapter** (`core/data/dada.py`) — real seeded train/test split, `--flat-frames-dir` symlink farm | ✅ 2026-09-03/04 (its 497-test figure is `v3`'s) |
| — | **Option A tooling** — `core/flow/zscore.py` (numpy leaf: `MomentAccumulator`, `ZScoreStats`, `standardize`) + `core/flow/zscore_cache.py` CLI; `eda.flow_stats` reads `zscore_stats.npz` | ✅ 2026-09-23 `ae4fded`, 17 tests. **Never run on real data** — P2 is on Colab |
| — | **`core/eda/` pre-flight profiler** — 5 modules + `core/tools/eda.py` + `core/docs/EDA.md`; verdicts for C27 / C12 / vanished windows, §4.2 frame-level linear probe | ✅ 2026-09-06, **present on `main`**. Never run on real data — needs the Drive caches |

**Measured on `main`, 2026-09-24 (`ae4fded`): 89 Python files (60 source + 29 test),
13,833 source LOC, 24 docs under `core/docs/**` (21 top-level + 3 under `v3/`).
582 tests collected, 0 fail.** Data-free, CPU-only.
*(History: 565 on 2026-09-21, 563 on 2026-09-18, 546 on 2026-09-17, 514 on 2026-09-15 — verified three
times there, 514 dots, no F/E.)*
*Minor open item:* one invocation
(`pytest --tb=line -q >/dev/null`) returned exit 1 once and would not reproduce
across four subsequent full runs — **observed, not diagnosed**; do not treat the
suite as flaky without reproducing it first.
*(2026-09-09 Phase 1 added `loss.dvs_anchor_mode`, `loss.bottomk_weight`,
`--equalize-length`: 439 tests. 2026-09-12 Phase 2a added `core/data/windows.py`,
the `windows.json` contract, the `--window-*` flags and the C30 eval fix: +54.)*
`outputs/**` holds **87,213** per-clip `.npz` score files (on the user's disk;
gitignored) — up from 62,254 on 2026-09-08, the difference being the DADA Phase 1
/ windowed arms and the whole TAD campaign.

**The suite went green on 2026-09-08 by collapsing the gate matrix.** It had
been 425 collected → 413 pass, 12 fail:
`core/tests/test_dada.py::TestDadaTrainsUnderEveryGate` (5) and
`core/tests/test_tad.py::TestTadTrainsUnderEveryGate` (7) parametrized their arm
matrix over `kip.gate_type`, a **v3-only** config field, and `core/config.py:207`
raised `KeyError: 'Unknown config key: kip.gate_type'` by design. The classes are
now `TestDadaTrains` / `TestTadTrains`, running the one gate v1 ships
(`kip.gate_signal=flow_norm`). **7 parametrizations removed, no coverage lost** —
`test_kip_off_trains` (arm A0), `test_stage1_warmup_runs` and the
`config.yaml`-recording assertions all survive; the latter now assert
`gate_signal` instead of `gate_type`. This was deliberately **not** a partial
port of `gate_type` (which would also need `ecmr.py`, the STE shift and the
diagnostics — 12 loud failures would have become one silent wrong gate).

Test-count history (the later figures are the **`v3`** branch's): 221 at the
2026-07-31 init → 284 with the DoTA adapter, `core/metrics.py`, `rescore.py` and
`feature_cache.py` → 322 with the PreVAD adapter → *(v3 only)* 434 with the gate
rebuild → 467 with TAD → 497 with DADA-2000 → 537 with `core/eda/`.

## Current status

Code-complete through Phase 5 plus MSAD-full support. The user has run training
and evaluation on MSAD at this commit and reports **results matching the paper**.

~~**The numbers themselves are not recorded in this repo**~~ — closed. Measured
results now live in `core/docs/RESULTS_MSAD.md` (center-crop, 2026-08-01),
`core/docs/RESULTS_DOTA.md` (DoTA protocol bug, 2026-08-08, KIP verdict
superseded) and **`core/docs/RESULTS_NCC.md`** (the `no_center_crop` rebuild,
2026-08-12). `outputs/{MSAD,MSAD_ncc,DoTA,DoTA_ncc}/**` hold the score curves.

~~**Headline as of 2026-08-12**~~ — superseded twice. Later documents:
`core/docs/RESULTS_PHASE_A.md` (3-seed repeat, 2026-08-16) and
**`core/docs/RESULTS_ARM4_PROBE.md`** (arm 4 + trajectory probe, 2026-08-19).

~~**Headline as of 2026-08-25**~~ — **superseded 2026-09-01 by the attribution
result, and qualified again 2026-09-06 by DADA-2000.** The historical framing is
kept below because the controls it records are still valid; the *interpretation*
is not.

## Headline as of 2026-09-06 — the +0.09 is attributed, and it is not motion

**`core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md` (2026-09-01), MSAD-trained,
zero-shot DoTA.** **A2** — a fixed 50 % channel shift with no flow, no PMG head,
no KIP losses and no stage-1 warm-up — is **statistically indistinguishable from
full v1 KIP** (Δ = +0.0109, t95 [−0.0588, +0.0805], 6/6 metrics include zero),
while A2 − A0 = **+0.1025 ± 0.0350**, t95 [+0.0154, +0.1896]. The archived V1 arm
reproduces at +0.0916 ± 0.0087, so the pipeline was validated against a known
number before the new claim. **KIP's measured contribution is temporal
smoothing.** Worse for v3: the `rank` gate is the *worst* KIP arm
(A1 − A2 = −0.0683, CI [−0.0790, −0.0580]) and does not beat LaGoVAD's released
trunk. **No component of KIP has been attributed a positive contribution.**

**`core/docs/v3/RESULTS_DADA.md` (2026-09-06) — the second replication inverts
it.** Trained on DADA-2000, zero-shot DoTA: **A2 − A0 = −0.0918** (predicted
> 0) and **A1 − A2 = +0.0300** (predicted ≤ 0). Both signs flip; a second A2 seed
agrees. Mechanism: DADA's median clip is **9** stride-8 frames (MSAD 86, DoTA 13)
under a `Conv1d(kernel=9)` score head, so the shift collapses the score curve to
a per-clip constant — Spearman(flatness, DoTA AUC) = **−0.82** across 7 arms.
**A component whose sign flips with training clip length is a smoothing
hyperparameter, not a motion mechanism.**

**And the DADA in-domain number is a mirage.** A constant-score-per-clip oracle
scores **micro AUC 0.9086** on that test set (74 % of frames come from all-normal
clips); our best arm reached 0.8739 with `auc_macro` at chance (0.44–0.57 across
arms). **No arm has shown frame-level localization on DADA-2000.**

**Against SimpleTAD's DADA→DoTA 80.3:** 0.803 − 0.6423 (our best DoTA ever) =
**0.161 method ceiling**; 0.6423 − 0.6069 (best DADA-trained arm) = **0.035
corpus cost**. 82 % of the gap is the method class — fully-supervised fine-tuned
VideoMAE at 10 FPS vs weakly-supervised frozen CLIP at 3.75 FPS — not the run.

### Results ledger — every number, and which branch's code produced it

Kept verbatim so `main` never loses the campaign record. **Code column = the
branch whose `core/` produced the checkpoint.** `outputs/` is gitignored, so
these rows plus `core/docs/RESULTS_*.md` are all that survives.

| Campaign | Trained on | Headline number | Code |
|---|---|---|---|
| **MSAD reproduction** (`RESULTS_MSAD.md`, 2026-08-01) | MSAD-full, seed 2024, center-crop | KIP-off **AUC 0.9052 / AP 0.7249**; KIP-on **0.9064 / 0.7334**; released `best.ckpt` **0.8991 / 0.6811**; paper 0.9041 | **v1** |
| **DoTA protocol fix** (`RESULTS_DOTA.md`, 2026-08-08) | — (eval only) | `best.ckpt` **0.5055 raw vs 0.6142 per-clip min-max**, published 0.6260 → pooling is decided by the label distribution (C12) | **v1** |
| **`no_center_crop` rebuild** (`RESULTS_NCC.md`, 2026-08-12) | MSAD_ncc, seed 2024 | MSAD: off **0.8922**/AP 0.6587, on 0.8868/0.6574, `gate_a` 0.8949. DoTA_ncc: off **0.5607** (macro 0.5638), on **0.6519** (0.6746), `gate_a` 0.6012 | **v1** |
| **Phase A, 3 seeds** (`RESULTS_PHASE_A.md`, 2026-08-16) | MSAD_ncc ×3 | DoTA Δ(on−off) **+0.0915 ± 0.0088**, 9/9 CIs exclude zero; MSAD null in all three; gate (b) re-stated against `best.ckpt` and **passes** | **v1** |
| **Arm 4 + trajectory probe** (`RESULTS_ARM4_PROBE.md`, 2026-08-19) | MSAD_ncc ×3 | Δ vs warm-started KIP-off **+0.0988 ± 0.0148** (no shrinkage → warm-start confound closed); KIP-off flat 0.559–0.561 over a 33× `mil` range → **H3 rejected** | **v1** |
| **PreVAD** (`RESULTS_PREVAD.md`) | — | Gate P0 defined; **KIP-on is unbuildable** (features only, no pixels) | **v1** |
| **v3 gate attribution** (`RESULTS_V3_GATE_ATTRIBUTION.md`, 2026-09-01) — *doc absent on `main`* | MSAD-full, A2 ×3 seeds | **A2 (fixed 50 % shift, no flow/PMG/KIP-losses) ≡ full v1 KIP**: Δ = **+0.0109**, t95 **[−0.0588, +0.0805]**, 6/6 metrics include zero. **A2 − A0 = +0.1025 ± 0.0350**, t95 [+0.0154, +0.1896]. Archived V1 arm reproduces at **+0.0916 ± 0.0087**. **A1 (rank) − A2 = −0.0683**, CI [−0.0790, −0.0580]. **A2b − A2 = −0.0105** | **v3** |
| **DADA-2000** (`v3/RESULTS_DADA.md`, 2026-09-06) — *doc present on `main`* | DADA-2000, 7 arms | Zero-shot DoTA **A2 − A0 = −0.0918** (2nd seed −0.1089) and **A1 − A2 = +0.0300** — *both signs inverted vs MSAD*. In-domain best **0.8739 micro** vs a **0.9086** constant-per-clip oracle, `auc_macro` **0.44–0.57 (chance)**. Spearman(flatness, DoTA micro) = **−0.82** | **v3** |
| **Method-class gap** (2026-09-06) | — | SimpleTAD DADA→DoTA **80.3** vs our best-ever DoTA **0.6423**: **0.161 method ceiling + 0.035 corpus cost → 82 % is the method class** | — |
| **DADA Phase 1** (`RESULTS_DADA_PHASE1.md`, 2026-09-12) — *first campaign measured by `main`'s own code* | DADA-2000, 4 arms + 1 duplicate, seed 2024, **KIP-off** | **Both loss arms failed their pre-registered predictions.** Control `p1_ctrl` 0.7050 micro / **0.5190 macro** / DoTA macro **0.6254**; `ignore` 0.6896/0.5134/0.6132; `bottomk` 0.6902/0.5104/0.6053; both 0.6764/0.5097/0.5952 — monotone *down* on 7 metrics × 3 protocols. `d = gap/σ` **+0.164 → +0.039**: the losses shrink the score scale, they do not create contrast. Length-controlled (eq5): ruler → **0.5000 exactly**, but **macro below chance on every arm (0.4237–0.4333)**. n=1 seed | **v1** |

| **TAD, end to end** (`TAD_SETUP.md` §8.1/§15.1, 2026-09-15; `RESULTS_TAD.md` **not yet written**) | TAD, 5 arms + 3 reference gates, seed 2024, **KIP-off** | **Gate T0 = 0.7912 micro / 0.7578 macro vs published 89.56 — FAILS by 10.4, diagnosed not chased** (labels 100/100 exact vs LaGoVAD's own anno, stride 8=8, pooling raw=raw; **frame ordering exonerated by macro 0.7578**; residual = length weighting — equal-clip-weighted micro 0.6574, ruler 0.8968 — plus `_ncc`). **In-domain `m0` collapses to a clip classifier:** clip-AUC **0.9975**, macro **0.6174**, micro 0.9237 ≈ the corpus's clip oracle **0.9226**. **Loss ladder falsified:** `t1_bottomk` moved 4/4 columns *wrong* (macro 0.6016, DoTA macro 0.5331) though the term worked mechanically (range +37 %, gap +78 %) — variance without direction; the user's `bottomk_topk_pct=8` dose arm agrees. **`t2_warm` (PreVAD-trunk warm start) is the only arm to move all four columns right** — macro **0.6540**, d **+0.5482**, DoTA macro **0.5887**, recovering 26 % / 27 % / 59 % of the gap — but still misses the bar. **Headline: TAD training DESTROYS 0.104 of macro it was handed (0.7578 → 0.6540).** n=1 seed | **v1** |

| **DADA-2000 ORIGINAL — Gate W** (`outputs/EDA/DADA2000_orig_T2_w20s8/`, 2026-09-16) | — (corpus build) | **W=16 FAILED** (clip oracle **0.7529** vs a 0.75 bar — `F_norm` 6,528 > `X` 6,377 by **151 frames of 19,536**). **W=20 hop 8 PASSES all six**: leak **0.5000**, oracle **0.7037**, retention **0.9568** (1,861/1,945), ratio 2.80, kernel coverage 0.1500, **798** two-class windows. The per-clip cap is **inert** on the oracle (0.7527–0.7529 across caps 2–6) while the window length spans 0.61–0.75 → **lesson C35**. Frame linear probe on the windows: **0.5983** | **v1** |
| **DADA-2000 ORIGINAL — Phase 4, KIP-OFF trunk** (`outputs/REPORTS/DADA2000_orig_phase4/`, 2026-09-16) | T2 (W=20), 3 seeds, 2,040 steps, `score_head_kernel=3`, `mil_topk_pct=5` | **In-domain: `auc_macro` 0.6248 ±0.0165 — ABOVE the 0.5983 per-frame-linear probe — with micro 0.6182 BELOW the 0.7037 clip oracle.** Macro ≥ micro, i.e. **the C14/TAD collapse did NOT reproduce** (TAD: micro 0.9237 ≈ oracle 0.9226, macro fell 0.7578 → 0.6174). Zero-shot DoTA **micro 0.5856 ±0.0230 / macro 0.6113 ±0.0306**; like-for-like vs MSAD kip_off (0.5491/0.5453) **+0.0365 / +0.0660, 3/3 seeds same sign, but t95 ±0.046/±0.080 INCLUDES ZERO**; level with PreVAD kip_off (0.5867/0.6015). Below the MSAD **kip_on** bar (0.6408/0.6529) — a comparison that means nothing until T2's own KIP-on arm runs | **v1** |
| **DADA-2000 ORIGINAL — Phase 4, the KIP A/B** (`outputs/v1/DADA2000_orig_phase4/`, 2026-09-18) | T2 (W=20), 3 seeds, **paired** — configs differ in exactly one line (`kip.enabled`) | **KIP-on is DOWN in-domain.** T2 micro **0.6182 → 0.6054, Δ = −0.0129, t95 [−0.0249, −0.0008], 3/3 seeds same sign — the only interval that excludes zero.** T2 macro 0.6248 → 0.6046 (Δ −0.0201, CI includes 0). DoTA 0-shot macro 0.6113 → 0.6071 and micro 0.5856 → 0.5914, **both ~20× wider than their own Δ — do not quote them.** `kip_rec` is **92.6–93.0 % of `total`** at `lambda_rec = 1.0`; stage 1 (trunk frozen) plateaus at 20.2, stage 2 (trunk free) falls to 11.3, so **44 % of the reconstruction gain came from rewriting `v^t`**; every task loss ends **+24–32 %** higher than its paired `kip_off` run. **C24: every KIP-on arm here is a fixed ~50 % channel shift — not "motion-gated"** | **v1** |
| **D1/D2 — KIP loss-scale diagnosis** (`outputs/v1/DADA2000_orig_diag_kip_loss_scale/`, run 2026-09-20, read out 2026-09-21) | — (read-only probe of the Phase-4 checkpoints; no arm re-run, no weight changed) | **D1:** `Z` 71.146 · `V` 31.642 · `W` 16.206 · `K` 11.620 → **`R²_global` 0.6328 (bar ≥0.20 PASS)**, **`K < W` PASS** (`R²_item` **0.2830** — the PMG head fits 28.3 % of *within-item* flow variance, so the motion premise is **not** refuted), **`V` = 31.642 CONFIRMS overweight** (~32× an O(1) objective; equalizing weight **`1/V` = 0.0316**), projection round-trip **0.9239** ∈ [0.9,1.1], **`mag_max` carries 83.0 %** of `E[s²]`, **48.8 %** of target variance is between-item (trip-wire 0.50, missed by 0.012). **D2:** at the shared temporal encoder `rho = \|g_KIP\|/\|g_task\|` = **11.63 (stage 1) → 3.105 (stage-2 end)**, per-seed 3.925/3.027/2.361 → **CONFIRM** (bar ≥1.0); `cos(g_kip_rec, g_task)` = **−0.0010** → **ORTHOGONAL**. Decision row **`capture + orthogonal` → Option A**, unauthorized. Caveat: `rho` sd is 1.56–1.94 per batch at stage 2, and the 11.63→3.10 decay is partly `\|g_task\|` **growing** (4.91→10.07, 7.24→12.46, 6.21→23.61) | **v1** |
| **Option A — `flow/v2_zscore`, KIP A/B re-measured** (`RESULTS_DADA_ORIG_T2.md`, run + read out 2026-09-26) | T2 (W=20), 3 seeds, KIP-on stage 1+2 on v2; KIP-off **reused** from phase 4 (config diff = `kip.enabled` + `lambda_rec`) | **Verdict "cost removed; KIP-v1 neutral on T2"** (pre-registered table). Build gates all pass: G0 1491/1491 max\|err\| 0.0, **`V_v2` 1.0047 → `lambda_rec` 0.9953**, between-item share 0.349. **R-2 `rho` 3.105 → 0.179** (0.152/0.187/0.199) → capture removed. **R-3 T2 micro 0.6182 → 0.6258, Δ +0.0076, t95 [−0.0002, +0.0154]**, 3/3 seeds +, includes 0 → bounded null. R-4 macro +0.0058 [−0.038, +0.050]. R-5 DoTA micro/macro +0.022/+0.026, CIs include 0, not decided on. **v2 − v1 (descriptive): T2 micro +0.0204 [+0.0010, +0.0398], macro +0.0259 [+0.0160, +0.0358]** → the phase-4 cost was the loss scale. R-1 `R²_v2` 0.280; R-1b magnitude 0.357 / **histogram 0.071** (risk 1 live); **`K_v2` 0.72 > item oracle `W_v2` 0.654 → D1-b fails on v2**. `cos(g_kin, g_task)` +0.61–0.72 is by construction (pending (x)). C24 untouched | **v1** |
| **T2 learning curve — phase 6** (`RESULTS_DADA_ORIG_T2.md` §8, 2026-09-26) | T2 subsets by source (25 % ⊂ 50 % ⊂ 100 %), KIP-off, 3 seeds, fixed 2,040 steps; 100 % = phase-4 KIP-off reused (config diff = `num_epochs` only) | **Verdict INCONCLUSIVE → CCD parked → D.** T2 micro 0.5947 / 0.6083 / 0.6182: Δ25 **+0.0137 [+0.0017, +0.0256]**, **Δ50 +0.0099 [−0.0266, +0.0463]** (primary). T2 macro 0.6085 / 0.6180 / 0.6248. **DoTA flat over 4× data:** micro 0.5773 → 0.5856, macro 0.6029 → 0.6113, CIs include 0. **Power defect:** the plan predicted ±0.008 from phase 5's fixed-data contrast; measured ±0.036, so FLAT was unreachable (pending (z)). Build gates G-S1…S6 all pass, no C14 flag | **v1** |
| **D2City normal-bag EDA** (`D2CITY_EDA.md`, 2026-09-25) | — (frozen-CLIP logistic probes, 5 grouped folds, paired) | **NO-GO.** Mechanics pass (G-L length AUC **0.5009**, R0 sanity **0.6763**, n = 1,860). **X (D2City-only negatives) 0.5864, Δ(X−R0) −0.0899 t95 [−0.1011, −0.0787]** (5/5 folds < −0.06 FAIL bar); **M Δ −0.0123 [−0.0155, −0.0091]** (bar −0.01; 5/5 folds < 0). **Shortcut AUC X 1.000 / M 0.999**; R0 alone already 0.820. S 1.0000 vs S-ref DoTA 0.9999 (both ceiling). V1 band crop: X 0.5767, no help → V0. Third separate-pool failure (C38) | **v1** (no `core/` code; notebook) |
| **BDD-A normal-bag EDA** (`BDDA_EDA.md`, 2026-09-26) | — (same probes/folds as D2City; R0 reproduces to 3.1e-5) | **NO-GO.** Mechanics pass (G-I 925/926, calm 508; G-L L-T2 length AUC **0.5000**; R0 **0.676227**). **Calm (gate): X 0.5492, Δ(X−R0) −0.1270 t95 [−0.1532, −0.1008]** (5/5 folds < −0.06); M Δ −0.0029 [−0.0062, +0.0004] (G-M pass, no gain). `all`: X 0.5375, Δ −0.1387. **Shortcut X 1.000 / M 0.999**; R0 0.374 (BDD-A starts *above* DADA normals — reverse of D2City). S 1.0000 vs S-ref 0.9999. L-T2 oracle 0.7037 → 0.8813 at mix 1.0. Fourth separate-pool failure (C38) | **v1** (no `core/` code; notebook) |

**Cross-branch comparability, settled 2026-09-12:** for `kip.enabled=false` a
`main`-vs-`v3` Δ **is** a Δ. `p1_ctrl` reproduces v3's A0 to 4 dp, and a
mis-pointed eval of the v3 checkpoint produced **331/331 bitwise identical** score
curves (max |Δ| = 0). This does **not** extend to any KIP-on arm.

**The three sentences that outlive every number:** *KIP's measured contribution
is temporal smoothing*; *a component whose sign flips with training clip length
is a smoothing hyperparameter, not a motion mechanism*; and, new on 2026-09-15,
*on a corpus whose micro AUC is clip-dominated, weakly-supervised MIL optimizes
clip classification and destroys localization — measurably, even localization it
was handed.* On `main` this is not an
abstract caveat — v1 **has no gate other than the frozen MLP**, so every KIP-on
run on this branch is that fixed ~50 % shift by construction.

### Historical framing (2026-08-25, controls still valid, interpretation dead)

On `no_center_crop` features, KIP-on beat KIP-off on DoTA (zero-shot from MSAD)
by **+0.09 micro AUC**, surviving every control run at the time — but nothing had
attributed it to any part of KIP:

| comparison | Δ DoTA micro AUC | seeds | CIs excluding zero |
|---|---|---|---|
| vs KIP-off, cold start | +0.0915 ± 0.0088 | 2024/25/26 | 9 / 9 |
| vs KIP-off, **warm-started from the same stage 1** | **+0.0988 ± 0.0148** | 2024/25/26 | 9 / 9 |
| at **matched train `mil`** (trajectory probe, seed 2024) | ≈ +0.092 | 2024 | — |

> **Report the seed-level t-interval, not the clip bootstrap.** "18 CIs, 18
> exclusions of zero" is 3 seeds × 3 metrics × 2 controls: within a seed the
> three metrics come from the same score curves, and the KIP-on arm is shared
> across controls. Independent replications = **3**. Seed-level headline:
> **Δ micro vs cold [+0.070, +0.113]**, vs warm [+0.062, +0.136], macro
> [+0.095, +0.120]. All still clear the pre-registered +0.03 bar.
> (`core/docs/v2/KAT-VAD_experiment_audit.md` §2.)

- **H1 holds. H2 (seed luck) rejected. H3 (under-convergence) rejected. Warm
  start rejected as the mechanism.** Δ is attributable to the KIP module *as a
  whole*, at +1.82 % inference params (**+1.65 % once `mhead` is removed from the
  inference graph; only 321 params sit on the score path**), RGB-only at test time.
- **MSAD is a bounded null, not an absence.** Every seed's CI includes zero, so
  what was measured is *"any in-domain effect is smaller than ≈1 AUC point"* —
  at n=3 with SD 0.0031 the study cannot see a real ±0.005. The asymmetry is
  still the finding: the motion pathway pays where motion is the anomaly and is
  inert where it is not.
- The **MSAD reproduction gate passes as redefined** (our KIP-off is
  statistically indistinguishable from LaGoVAD's released `best.ckpt` under one
  identical protocol; 0.9041 is unreachable with the published artifacts —
  lesson 8b).
- **The mechanism is still unknown, and the attribution is now the bigger gap.**
  D4 (ego-kinematics) is refuted six for six; D5 (MSAD multi-class accuracy) did
  not replicate under `_ncc`. Two rival explanations were eliminated by
  subtraction; none was confirmed. **The three ablations that would attribute
  the gain to a sub-module were never run.**
- **New rival hypothesis H4 (temporal mixing), not refuted by anything measured.**
  The gate MLP is never trained (`core/kip/gate_shift.py:112` — hard `floor()`
  kills the gradient; verified). With a random MLP `sigmoid(·) ≈ 0.5`, so ≈64 of
  512 channels are pulled from each neighbour. If the *level* of mixing does the
  work and the *modulation* does not, KIP is a fixed temporal smoother — and
  +0.09 on a within-clip localization benchmark is what a smoother buys.

## What's left

### KAT-VAD v2 — E0–E2 gates (new 2026-09-27; design in `core/docs/v2/`, advisor sign-off)

- [ ] Write `.project/plans/katvad-v2-e0-e2.md` (harness, split files, read-out tables).
- [ ] **Freeze + commit splits and rules first:** T2-val = 15 % of T2-train sources (grouped);
      DoTA-dev / DoTA-eval = 50 / 50 by video. Record the commit hash in every read-out.
- [ ] Harness loads DoTA-dev IDs only (DoTA-eval never printed until the final report).
- [ ] **E0** rate audit from the caches (fps, stride, s/step). Gap not ≈ 3× → skip E1.
- [ ] **E0b** pooled `MDE_dev` (phase-4 v1−off, phase-5 v2−off, two learning-curve arms;
      ≤ 8 df), descriptive only. **E1** stride-3 / sliding re-score of the 3 KIP-off
      checkpoints on DoTA-dev (decided on the clip bootstrap). One harness for both.
- [ ] **E2** (a) accident-share histograms → (b) position ruler + CRN reference choice →
      (c) transfer-probe veto → (d) VideoMAE V2-B/S probe (start extraction in parallel).
- [ ] **E3** 2 × 2 factorial × 5 seeds (20 runs) — only after E2 fixes reference + encoder.
- [ ] Code to build (none exists): CRN references, motion-stream feature cache + fusion,
      sliding-window DoTA scorer at stride 3, per-arm switches. Load lessons first.

### D2City as the normal-bag pool — feasibility EDA (2026-09-24) — **CLOSED: NO-GO 2026-09-25**

Plan: `.project/plans/katvad-d2city-normal-bag-eda.md`. Runbook:
`colab/D2City/eda_normal_bags.ipynb`. Proposed corpus: full-length DADA-original
accident videos (positive bags) + D2City dashcam clips (negative bags). **It does not
replace T2 until G-X/G-M say so.**

- [x] Local inventory of `data/D2City/`: 7 zips × 100 clips, 25 fps, ~30 s,
      63 % 1080p / 37 % 720p, 700 XML box tracks (measured on `0001/`).
- [x] Plan with pre-registered gates G-I, G-L, R0 sanity, **G-X**, G-M, G-S (claim gate).
- [x] Notebook written; dry-run end-to-end locally (mechanics only). Fixed a
      short-biased L-match recipe (0.578 → 0.526) and a montage sampling bug.
- [x] **User:** ran on GPU (2026-09-25; §4 crashed on RAM first — patched, per-arm resume).
- [x] Plan Appendix A filled; `core/docs/D2CITY_EDA.md` written; **row taken: NO-GO**
      (G-X FAIL −0.090, G-M fail −0.0123, shortcut 1.000). **Track closed.**
- [~] ~~If GO: a corpus-build plan~~ — not applicable.
- [ ] Settle A1 (DADA 30 fps) with one `ffprobe` on a source mp4, if one exists.

### BDD-A as a T2 negative-bag pool — EDA (2026-09-25) — **CLOSED: NO-GO 2026-09-26**

Plan: `.project/plans/katvad-bdda-normal-bag-eda.md`. Runbook: `colab/BDDA/eda_normal_bags.ipynb`.
Write-up: `core/docs/BDDA_EDA.md`.

- [x] Local inventory of `data/BDDA/` (BDD-A training split, 926 mp4 + 1 Hz GPS); plan with
      pre-registered gates; notebook dry-run.
- [x] **User:** ran on GPU (2026-09-26); outputs in `outputs/EDA/BDDA/`.
- [x] Appendix A filled; **row taken: NO-GO** (G-X FAIL −0.127 on calm). **Track closed.**
- [~] ~~Download the validation split / BDD100K~~ — conditional on a pass; not applicable.

### DADA-2000 **original** corpus — the live track (new 2026-09-15)

Plan: `.project/plans/katvad-dada-original-corpus.md`. Runbook:
`core/docs/DADA_ORIGIN_PHASE0.md`. **Supersedes
`katvad-dada-phase2-corpus-rebuild.md`** — that plan re-sharded the *trimmed*
archive, which C32 and this session's reading of `outputs/EDA/DADA2000_w{32s2,24s1}`
both show is the worse corpus on every column.

- [x] **`data/DADA/dada标注.xlsx` identified** as the original DADA-2000
      annotation; `Cleaned_Metadata.csv` is a strict subset (975 ⊂ 1,962,
      **0/975** mismatches), adding only `Fault_Label`, which the original
      release **does not have**.
- [x] **`L_neg` from this file: ruled out**, with numbers — `texts` is a
      category label (83 unique over 1,962 videos), not a per-video description;
      G4 means nothing reads a caption field; `N3_MIN_SCORE_RANGE` gates the one
      branch that would matter. See [[activeContext]] §2.
- [x] **Corpus construction decided — T2**: negatives cut from *inside* the
      accident videos. ~~W=16, hop 8, oracle 0.6631, 98.1 % retained~~ — those
      were a **simulation over the annotation alone**. Measured, W=16 fails Gate W
      (oracle 0.7529) and the adopted geometry is **W=20 / hop 8** (oracle 0.7037,
      retention 0.9568). The 0.6631 prediction belongs to W≈24, which fails
      retention instead. Lesson **C35**.
- [x] **Phase 0 / P2 PASSES** — layout
      `DADA2000/{type}/{video:03d}/images/{frame:04d}.png`, 52/52 types match the
      xlsx, aggregate frame delta **+0.30 %** (vs −79 % for the trimmed archive).
      `images` = 94.01 GiB of 116.7 GiB; the other four subdirs are DADA's own
      driver-attention task and are never extracted.
- [x] **Phase 0 / P1 + P3 PASS** (2026-09-15) — 397/400 clips exact, mean signed
      delta −0.28; the release is **not** trimmed.
- [x] **Gate D0 PASSES — 0.6518** on 400 clips (bar ≥ 0.60), `d0_abs` 0.6492,
      |Δ| 0.0026 vs a 0.01 bar. DoTA 0.6708 · DADA-archive 0.5228 on the same
      frozen features → the archive's CRITICAL verdict is about that corpus.
- [x] **Phase 2 — T2 built** (`core/data/dada_origin.py`, `core/tests/test_dada_origin.py`,
      `colab/DADA2000Origin/phase_2.ipynb`). Sharded extraction, symlink farm at
      the `images` level (C26), absolute-index labels, type-stratified split.
- [x] **Phase 3 / Gate W PASSES at W=20 hop 8** — all six criteria; see the ledger.
      `core/constants.py:DADA_ORIGIN_WINDOW_LENGTH` changed 16 → 20 **after** the
      gate passed, never before.
- [x] **Phase 4 — KIP-OFF trunk trained and evaluated**, 3 seeds
      (`colab/DADA2000Origin/phase_4.ipynb`). **The C14/TAD collapse did not
      reproduce.** See the ledger row and [[activeContext]] 2026-09-17.
- [x] **Phase 4 — RAFT flow pass** done; `cache/flow/v1/DADA2000_orig` exists
      (4,401 train windows × 20 frames of 23-d stats in the D1 census).
- [x] **Phase 4 — the KIP A/B**, 3 seeds, paired. **NEGATIVE:** T2 micro
      −0.0129, t95 [−0.0249, −0.0008], 3/3 seeds. See the ledger and
      [[activeContext]] 2026-09-18. C24 still applies to every arm in it: on
      `main` there is no `kip.gate_type`, so each KIP-on run is a **fixed ~50 %
      channel shift** — do not call it "motion-gated".
- [x] **D1/D2 — the loss-scale diagnosis** run 2026-09-20, read out 2026-09-21.
      Decision row **`capture + orthogonal`**. `outputs/v1/DADA2000_orig_diag_kip_loss_scale/`.
- [x] **Option A authorized** 2026-09-23, construction **A1** (z-score the 23 raw
      stats with train-window moments, then the **same** `M`; not the 256-d `e_O`).
- [x] **Option A P1 — code, tests, docs** (`ae4fded`, 2026-09-23): `core/flow/zscore.py`,
      `core/flow/zscore_cache.py`, `flow_stats` z-score-aware, 17 tests, `TRAINING.md`
      + `COLAB.md` §4.3b. Runbook `colab/DADA2000Origin/phase_5_zscore.ipynb` written.
- [x] **Option A P0 + P2 — build `cache/flow/v2_zscore/DADA2000_orig`** (2026-09-26: all gates pass, `V_v2` 1.0047, `lambda_rec` 0.9953) on Colab; gates
      G0 (every v1 `e_O` == its `stats @ M`), G1, G2-b, G2-c HARD; read `lambda_rec`
      from `zscore_manifest.json`. **Never use 0.0316 on v2** — that is v1's `1/V`.
- [x] **Option A P3 — KIP-on stage 1 + 2 × 3 seeds on v2** (2026-09-26) (C2 re-measure). KIP-off
      arms are **reused** from phase 4 (inert to `lambda_rec` and flow;
      `TestKipOffIsInert`); the notebook's §4 asserts the config diff.
- [x] **Option A P4 — read-out** (2026-09-26): **"cost removed; KIP-v1 neutral on T2"**; `core/docs/RESULTS_DADA_ORIG_T2.md`. Against the pre-registered table (plan §6.3): R-1
      `R²_v2`, R-1b per-block R² via `ê_O @ pinv(M)`, R-2 `rho` < 1.0, R-3 paired Δ
      T2 micro. Fill the plan's Appendix A.
- [ ] ~~**If Option A does not move the A/B, the next suspect is C24**~~ — *not triggered*: R-3 was not negative. C24 stays open only if the thesis needs a motion-gating claim. That work is a `gate_type` + `ecmr.py` + STE + diagnostics port and
      belongs on branch **`v3`**, never piecemeal on `main`.
- [x] **Accept the bound** (2026-09-26): v1 KIP on T2 is a bounded null. **No fourth seed on R-3** (optional stopping). **Stop spending seeds on v1 KIP.** *Old note:* DoTA macro sd 0.0306 at n=3 cannot
      decide the T2-vs-MSAD gap (+0.066). Do not quote it as established.

### Branch hygiene on `main` (new 2026-09-08)
- [x] **Green the suite on `main`** — done 2026-09-08. Collapsed
      `TestDadaTrainsUnderEveryGate` / `TestTadTrainsUnderEveryGate` to the
      single v1 configuration and renamed them `TestDadaTrains` / `TestTadTrains`.
      12 failures → 0 (418 collected, 418 pass), no coverage lost for the
      adapters themselves.
- [ ] **Decide what `main` is allowed to grow into.** Current intent: `main`
      stays v1 and owns the v1 story; gate architecture happens on `v3`. If that
      changes, the port is `ecmr.py` + `gate_type` + the STE shift + diagnostics
      **together**, never piecemeal.
- [ ] **Cross-branch merge discipline.** `CLAUDE.md` and
      `.project/memory-bank/*` are tracked and have deliberately
      diverged; any `main`↔`v3` merge conflicts there must be resolved by branch
      identity, not by "take theirs".
- [x] ~~**The campaign notebooks are untracked on `main`.**~~ **Stale — corrected
      2026-09-24:** `git ls-files colab` lists every campaign notebook/script
      (`colab/{MSAD,DADA,DoTA*,DADA2000Origin}/…`), incl. `phase_5_zscore.ipynb`.

### Immediate
- [x] Record real MSAD metrics at `b9978ff` → **`core/docs/RESULTS_MSAD.md`**
      (2026-08-01). MSAD-full, 240 test videos, seed 2024, `checkpoint_last.pt`:
      KIP-off **AUC 0.9052 / AP 0.7249**, KIP-on **AUC 0.9064 / AP 0.7334**,
      released `best.ckpt` through our eval 0.8991 / 0.6811. LaGoVAD paper 0.9041
      → **baseline reproduction PASSES.**
- [~] Gate (c): KIP-on ≥ KIP-off on every aggregate metric (6/6) but **every
      95 % bootstrap CI includes zero** (ΔAUC +0.0012, CI [−0.0060, +0.0089]).
      Not defensible from one seed → needs 3 seeds (RESULTS_MSAD §6.1).
- [x] **`no_center_crop` rebuild** → `core/docs/RESULTS_NCC.md` (2026-08-12).
      4 runs (2 transforms × KIP on/off), seed 2024. MSAD_ncc: KIP-off
      0.8922 / AP 0.6587, KIP-on 0.8868 / 0.6574, `gate_a` 0.8949 / 0.6432.
      DoTA_ncc: KIP-off 0.5607 (macro 0.5638), KIP-on **0.6519 (0.6746)**,
      `gate_a` 0.6012 (0.6158). ~~**MSAD reproduction now FAILS** (0.8922 <
      0.9041).~~ **Wrong comparison — corrected 2026-08-16 (lesson 8b):** the
      released `best.ckpt` cannot reach 0.9041 either. Gate re-stated against
      the checkpoint and **passes**.
- [x] **Phase A — 3-seed on/off repeat** → `core/docs/RESULTS_PHASE_A.md`
      (2026-08-16). Δ +0.0915 ± 0.0088 on DoTA, nine CIs excluding zero, MSAD
      null in all three. **H2 rejected.** Also redefined and passed the MSAD
      reproduction gate against the released `best.ckpt` (lesson 8b).
- [x] **A4 — 4th arm: KIP-off warm-started from stage 1** →
      **`core/docs/RESULTS_ARM4_PROBE.md`** §2–3 (2026-08-19). 3 training runs,
      no code. Δ(on − off_warm) **+0.0988 ± 0.0148**, nine CIs excluding zero —
      no shrinkage. **Warm-start confound closed**; stage-1 trunk pretraining
      alone moves DoTA by ±0.02 with a seed-dependent sign.
- [x] **A5 — trajectory probe** → `RESULTS_ARM4_PROBE.md` §4 (2026-08-19). 10
      evals on cached features, no training. KIP-off flat at 0.559–0.561 across
      a 33× `mil` range; at matched convergence it reaches ≈ 0.560 vs KIP-on's
      0.6519. **H3 rejected.**
- [x] **PreVAD code prerequisites (G1/G2/G3)** — `core/data/prevad.py`, the
      36-class v6 taxonomy, 38 tests (2026-08-24). Nothing downloaded or run.
      → `core/docs/PREVAD_SETUP.md`, lessons 18/19.
- [x] **v3 gate rebuild** (2026-08-30) — four selectable `gate_type`s sharing one
      floor/clamp/mask step, parameter-free ECMR `rank` gate, `mlp_ste` STE
      applied to the selection weights (the spec's own formula is a no-op),
      train-only KIP submodules off the inference graph, gate diagnostics
      (`kip_s`, `kip_gate_ratio`, `kip_m`, …) in every score `.npz`.
      434 tests. → `.project/plans/katvad-v3-kip-gate-rebuild.md`
- [x] **v2 tier 1 T1.2 (constant-shift / H4), delivered as v3 arm A2** —
      **`core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md`** (2026-09-01). 16 evals,
      6 arms on MSAD-full, A2 at 3 seeds. **H4′ confirmed: the smoother is the
      whole effect.** Also killed the v3 rank gate as a design.
      Superseded the `RESULTS_OFFLINE_V2.md` / tier-0 plan — the controls answer
      the same question directly instead of by correlation.
- [x] **TAD replication prerequisites** (2026-09-02) — P1 `--with-train-split`,
      P2 `raft_extract --frames-dir`; frames-vs-video parity asserted (lesson
      C25). Runbooks `core/docs/TAD_SETUP.md`, `v3/setup/TAD_V3_SETUP.md`.
      **Nothing trained yet.**
- [x] **DADA-2000 replication, end to end** (2026-09-03 → 09-06) —
      `core/data/dada.py` (+ lesson C26 on id collisions), runbooks, 7 arms
      trained on Colab, both evals, analysed in
      **`core/docs/v3/RESULTS_DADA.md`**. **The pre-registered ordering inverts.**
- [ ] **v2 tier 2 — T2.3 shuffled flow targets.** Temporally permute `e_O`; if a
      gain survives a permuted target the motion claim is dead. **Largely moot
      now**: A2 uses no flow at all and reproduces the whole MSAD effect, so the
      motion claim is already unsupported. Keep only if a *positive* flow
      contribution is ever claimed again.
- [ ] **DADA follow-ups** (`RESULTS_DADA.md` §10), in order: **(A)** reporting
      fix — `auc_macro` + the 0.9086 clip-oracle row as the DADA headline;
      **(B) frame-level linear probe** on the cached frozen-CLIP DADA features
      against the real frame labels (~30 min, decisive: separates backbone
      capacity from weak supervision); **(C)** re-extract at stride 2–4 with
      `score_head_kernel=3` — only if B is positive, **fires lesson C2**;
      **(D)** root-cause the `mlp_ste` NaN.
- [x] **TAD Gate T0 — RUN 2026-09-14, result 0.7912, FAILS vs 89.56 and is
      diagnosed** (`TAD_SETUP.md` §8.1). Four of five suspects exonerated by
      measurement; **frame ordering cleared by `auc_macro = 0.7578`**. Residual is
      length weighting plus `_ncc`. **`0.7912` is now the reference for every TAD
      arm (C8b); 89.56 is retired from TAD tables** — it sits between this
      corpus's length ruler (0.8968) and its clip oracle (0.9226).
- [x] **TAD EDA** (2026-09-14, `outputs/EDA/TAD`) — **two CRITICAL verdicts**:
      clip oracle micro **0.9226** with 99.69 % of frame pairs cross-clip, and a
      frame-count-only ruler at clip-AUC **0.6940** / micro **0.8968**.
- [x] **TAD loss ladder (`t1_*`) — FALSIFIED** 2026-09-15, plan
      `.project/plans/katvad-tad-loss-ladder.md`, readout `TAD_SETUP.md` §15.1.
      `bottomk` moved all four columns the wrong way *while working mechanically*
      (loss 0.2294→0.0062, within-clip range +37 %, gap +78 %): **variance with no
      direction**. Refuted on a corpus where DADA's C27 confound is absent, plus
      the user's `bottomk_topk_pct=8` dose arm. **The objective-side branch is
      closed.**
- [x] **`t2_warm` (PreVAD-trunk warm start on TAD)** — 2026-09-15. First arm to
      move all four columns right (macro 0.6540, d +0.5482, DoTA macro 0.5887);
      recovers 26/27/59 % of the gap; still misses the pre-registered bar.
      **Buys the claim: TAD training destroys 0.104 of macro it was handed.**
- [ ] **TAD destruction curve — THE NEXT RUN.** `t2_warm` under
      `--stop-after-epochs 3 7 18 36` (geometric in steps, **C16**; LR horizon
      stays `num_epochs=72`, so each point is the same run stopped earlier). With
      `gate_t0` (E=0) and `t2_warm` (E=72) that is 6 points for ~950 steps.
      **If macro holds ≈0.75 early, early stopping yields a usable TAD model and
      the campaign unblocks for free; if it decays monotonically, that is the
      figure for the paper.**
- [ ] **`core/docs/RESULTS_TAD.md` does not exist yet.** Everything above lives
      only in `TAD_SETUP.md` §8.1/§15.1 and [[activeContext]]; `outputs/` is
      gitignored.
- [ ] **Confirm `t2_warm`'s init checkpoint, then patch C17.** `--init-weights`
      is recorded in no artifact — `t2_warm/stage2/config.yaml` is byte-identical
      to `m0`'s. Write a run manifest (`--init-weights`, resolved paths, git sha,
      `sys.argv`).
- [ ] **Gap G4 — wire real per-video captions into `L_neg`.** Audit 2026-09-15:
      **54 of 55 `config.yaml` never computed `L_neg`**; the sole exception,
      `outputs/v1/PreVAD/stage2_kip_off`, fabricated captions from class
      definitions. The `descriptions` field has **never** been used by any run.
      `L_neg` is the *directed* version of the pressure `bottomk` applied blindly
      (its video embedding is `softmax(logits/0.02) @ v_feats` — a near-argmax
      under the model's own curve, and `n3` mines an abnormal clip's lowest
      frames as negatives). Caveats before building: `N3_MIN_SCORE_RANGE = 0.2`
      leaves only ~14/60 TAD clips eligible at eval-time ranges, and
      AI-generated captions carry the label **and its timing** → train split
      only, test stays definition-only, and the method stops being comparable to
      LaGoVAD's weakly-supervised setting unless declared.
- [ ] **TAD KIP A/B (M1/M2/M3) — BLOCKED by C14.** `m0` is a clip classifier, so
      an arm stacked on it measures the collapse, not the smoother. Standing rule
      recorded in `TAD_SETUP.md` §15. TAD remains the intended tie-breaker
      between MSAD (+0.10) and DADA (−0.09), but only on a non-degenerate trunk.
- [ ] **TAD windowing — simulated and REJECTED at stride 8** (2026-09-15). No
      window keeps ≥ 90 % of abnormal clips without putting the kernel-9 head
      over ≥ 75 % of it (W=12: 93.3 % kept but kernel spans 75 % and the median
      span fills 92 %; W=24: 78.3 % kept). Stride 2 costs ~269 k frames / 3–5 h
      and **fires C2 on `gate_t0` and every TAD arm**, and buys resolution not
      balance — TAD's positive span is 33 % of its clip at every stride. Revisit
      only after the destruction curve, and only once DADA's own `w24s1`
      geometry has actually been trained.
- [ ] ~~**P0 — eval-time KIP diagnostics.**~~ Subsumed by v2 tiers 0–1, which
      answer the same question with controls instead of correlations.
- [ ] P1 — `L_KIP_align` runs 13–15 % below chance (3.66 vs E[ln n] = 4.27) for
      1,000 steps. Cause diagnosed as a granularity mis-transplant from Pi-VAD
      (snippet-level InfoNCE, not frame-level). **Pre-stated decision rule: if
      the respec shows nothing, delete the term and publish the negative**
      (−98,560 train params).
- [ ] P1 — `AUC_A` + MCC + mAP@IoU in `core/evaluate.py:58-71` (stubs exist).
- [ ] P2 — `_rng_payload` should store the numpy RNG state as raw bytes or omit
      it (lesson 15). Deferred: training code stays frozen while arms are
      compared.
- [~] **Phase B — validation split + checkpoint selection. Deferred on
      evidence.** It existed to answer H3; A5 did that for ~1 % of the cost, on
      the training set everything else was measured on. Its numbers would come
      from a 20 %-smaller training set and be incomparable to every result above
      (plan §3.3). Build it only when checkpoint selection is itself the
      deliverable. → `.project/plans/msad-ncc-seeds-and-selection.md` §3

### Phase 6 remainder (user-run, per `core/docs/COLAB.md`)
- [ ] Stage 1 KIP warm-up on MSAD-full — confirm `L_KIP_rec` decreases and
      plateaus; watch `L_KIP_align` (A11 mitigation flags exist if it collapses).
- [ ] Stage 2 full objective, KIP-on and KIP-off, matched seeds.
- [ ] Smoke-test the spec §10 ablation flags.

### Phase 7 (scale-out)
- [x] DoTA adapter + eval protocol — `core/data/dota.py`, `core/docs/DOTA_EVAL.md`
      (2026-08-01). Label arithmetic verified equal to the baseline's shipped
      `anomaly_span` on all 1,402 val clips. Also fixed `core/data/tad.py`,
      which did not import at `b9978ff` (missing `video_io` frame helpers).
- [x] **Run** DoTA zero-shot: 3 arms at stride 8, twice (center-crop 2026-08-08,
      `no_center_crop` 2026-08-12), then 3 seeds (2026-08-16) and a 4th warm arm
      + trajectory probe (2026-08-19). D1 passes, D2 descriptive, **D3 passes on
      `_ncc`** across every control, **D4 fails six for six** (gain is `other` >
      `ego`), **D5 fails** (MSAD multi-class effect did not replicate).
      → `core/docs/RESULTS_NCC.md`, `RESULTS_PHASE_A.md`, `RESULTS_ARM4_PROBE.md`
- [~] **PreVAD trunk transfer** — code prerequisites **done** (2026-08-24);
      next is Gate P0 (`best.ckpt` on PreVAD test through our eval). The released
      features make a KIP-**off** trunk trainable with no raw video and no flow;
      KIP-**on** on PreVAD stays out of reach (needs pixels the release does not
      ship). → `core/docs/PREVAD_SETUP.md`
- [ ] PreVAD *KIP-on* acquisition + flow extraction at scale (~50–100 A100-hours)
      — blocked on raw video the authors have not released
- [~] TAD / DADA adapters **shipped** (`core/data/{tad,dada,dada_origin}.py`, 2026-09-02/03/15)
      and both campaigns are measured. **UCF-Crime adapter: not built.**
- [ ] Full metric suite: MCC family, AUC_A, mAP@IoU (currently
      `NotImplementedError` in `core/evaluate.py`)
- [ ] Optional: Stage 0 warm-up, Stage 0.5 hard-negative tuning, Stage 3
      ATS + MLLM reasoning head

## Known issues / technical debt

| Issue | Where | Impact |
|---|---|---|
| ~~Extraction transform differs from the baseline's `no_center_crop`~~ | `core/tools/extract_clip_features.py` (`--no-center-crop`) | **Resolved 2026-08-12** — pipeline moved to `no_center_crop` for internal field-of-view consistency between the appearance and flow branches (lesson 13), *not* for baseline parity (refuted, pending P1). All current results are `_ncc`. |
| Checkpoints carry a pickled numpy RNG state | `core/train.py:389` (`_rng_payload`) | Artifacts stop loading when the runtime's numpy major version drifts (lesson 15). Worked around per `COLAB.md` §A4.0; fix deferred while arms are compared. |
| Step checkpoints are spaced uniformly in steps, not in loss | `train.checkpoint_every_steps` | A trajectory probe cannot sample the early, high-loss part of training (lesson 16). Cost the A5 probe its low-convergence segment. |
| ~~12 tests fail on `main`~~ | `core/tests/test_{dada,tad}.py` | **Resolved 2026-09-08.** The gate matrix was parametrized over `kip.gate_type`, a **v3-only** config field, so `core/config.py:207` raised `KeyError` (425 collected → 413 pass). Collapsed to the v1 gate and renamed `TestDadaTrains` / `TestTadTrains`: **418 collected, 418 pass**. `gate_type` was *not* ported — a partial port needs `ecmr.py` + the STE shift + diagnostics or the gate is silently wrong. |
| MPS training diverges on torch 2.4 | `core/train.py`, documented in `core/docs/TRAINING.md` | Local training must pin `train.device=cpu` |
| **Score head's kernel spans a short clip** | `core/models/heads.py:21` (`ConvScoreHead`, `kernel_size=9`) | On DADA-2000 (median T = 9) every output timestep sees the whole clip, so the detector is structurally a **clip classifier**: flat curves, `auc_macro` at chance, inflated micro AUC. MSAD (T = 86) is unaffected, DoTA (T = 13) partly. Check `score_head_kernel` against a corpus's median length before training on it (`RESULTS_DADA.md` §5). |
| **`mlp_ste` trains through NaNs under AMP** | `core/kip/gate_shift.py:99` (`shift_channels_straight_through`) | 33 of 500 steps NaN in `mil`/`mul_mil` on the DADA arm, over 17 of 20 epochs, while `kip_rec`/`kip_align` stay finite; no other arm at the same seed/batch/data. A4's DADA row is unreportable. Not root-caused (2026-09-06). |
| **Gate MLP is never trained** | `core/kip/gate_shift.py:112` | **Corrected 2026-08-25 — this was recorded as "zero gradient *at init*"; it is zero gradient *always*.** Hard `floor()` is non-differentiable, so all 321 params stay at random init for the whole run (verified: 6/6 tensors `grad is None`). Forces the "321 *frozen*" qualifier on every efficiency claim and opens **H4**. |
| **`L_KIP_rec` regresses an UNNORMALIZED target** | `core/flow/raft_extract.py:86-118` → `core/data/dataset.py:114`; weighted at `core/train.py` `lambda_rec=1.0` | **Measured 2026-09-20 (lesson C37).** A constant global-mean predictor scores MSE **31.64** on `e_O` and `mag_max` (raw pixel units, mean 27.19) carries **83.0 %** of `E[s²]`, so at `lambda_rec = 1.0` the term enters an otherwise-O(1) objective **~32× oversized**. It is not inert: `rho` = \|g_KIP\| / \|g_task\| at the shared temporal encoder is **3.1** at the stage-2 end (11.6 at stage 1), `cos` = **−0.001** (orthogonal), and every task loss ends **+24–32 %** higher with KIP on. Repair = **Option A**, **authorized 2026-09-23 and built (`ae4fded`)**: `cache/flow/v2_zscore/` + `lambda_rec = 1/V_v2` (≈ 1, derived on the NEW cache — 0.0316 was v1's). **Not yet measured.** |
| `Trainer.compute_losses` is not repeatable | `core/train.py:235` (source not traced — stub text encoder or DVS/align sampling suspected) | Same batch, `eval()`, `no_grad`, `torch.manual_seed(0)`: `total` 3.294 vs 3.268 (2026-09-23) — ~10⁴× P3's BLAS noise. Equality tests on the loss are flaky; test weight inertness by **NaN-poisoning** instead. May inflate `grad_probe`'s per-batch `rho` spread — **not measured**. `pending.md` candidate (iv). |
| Flow target is 23-d and non-spatial | `core/flow/raft_extract.py:52-81` | `ê_O` is 23 frame-global scalars projected to 256-d. No "localized/peripheral motion" story is expressible in it; restate as a global flow-statistic residual. |
| **Student/teacher aspect mismatch of exactly 4:3** | `extract_clip_features.py:71` (224×224) vs `raft_extract.py:135` (240×320) | Source-independent **bias**, not noise; warps `atan2(v,u)` — the direction channels. Same defect class as the crop that was worth +0.15. Fix = full flow re-extract (C2), scheduled behind the shuffled-target control. |
| Score saturation 86–89 % of frames > 0.99 | training protocol | Makes `L_kin`'s `y^bin` anchor a near-constant (term is near-vacuous) and **voids every threshold metric** until fixed. Tie rate ~0.2/clip, so ranking metrics are sound. |
| `L_dvs` row-gating deviates from the baseline | `core/train.py`, `core/docs/TRAINING.md` §1 | First knob to revisit if a reproduction gate misses |
| Resume is FP-tolerant, not bitwise | `core/tests/test_e2e_synthetic.py` | Accepted; Accelerate BLAS nondeterminism |
| No `AUC_A` / MCC / mAP@IoU | `core/evaluate.py` | Blocks the Phase-7 full metric table |

## Evolution of decisions

- **2026-07-07** — plan rev. 2 goes *code-first*: all Phases 0–5 code written and
  proven on synthetic data before any download, so no phase blocks on the user.
- **2026-07-08** — target dataset pivots **TAD → MSAD**: the user already had
  MSAD on disk. TAD demoted to a thin adapter; gates restated against MSAD.
- **2026-07-31** — MSAD scope widened from the traffic slice to the **full
  11-class benchmark** (`MSAD_FULL_DATASET`, `b9978ff`).
- **2026-07-31** — memory bank reset to this commit; prior experiment history
  deliberately not carried over.
- **2026-08-08** — DoTA pooling corrected to per-clip min-max on an
  effectively all-abnormal test set (lesson 12); `--score-norm auto` resolves it
  from the label distribution, not the dataset name.
- **2026-08-12** — pipeline moves to `no_center_crop`, on the field-of-view
  argument (lesson 13), not on baseline parity (refuted).
- **2026-08-16** — MSAD reproduction gate redefined: gate against the released
  `best.ckpt` under one identical protocol, not against the printed 0.9041
  (lesson 8b).
- **2026-08-19** — Phase B dropped from the critical path. Confounds are closed
  with control arms and existing step checkpoints (A4/A5), not with new
  selection machinery.
- **2026-08-22** — the v2 layer lands (`core/docs/v2/`, `.project/plans/katvad-v2-next-steps.md`).
  Ordering principle adopted from the audit: **attribution before architecture.**
  Reading the tree produced six code facts (F1–F6) that neither the report nor
  spec v2 contains; F1 (frozen gate) reframes the headline claim and makes two
  decisive controls nearly free.
- **2026-08-30** — **v3 gate rebuild lands.** H4′ confirmed by measurement: v1's
  gate moves `s_t` by ≤4 of 128 channels across its entire reachable input
  domain, so "motion-gated adaptive shift" was, in operation, a fixed ~50 % shift.
  Four selectable `gate_type`s make the decisive control a config flag.
- **2026-09-01** — **attribution closes, negatively.** The plain-TSM control (A2)
  reproduces the entire MSAD +0.09 with no flow, no PMG head and no KIP losses,
  and is statistically indistinguishable from full v1 KIP. The v3 `rank` gate is
  the worst KIP arm. **Ordering principle for everything after this: report what
  the ablation says, not what the proposal hoped.**
- **2026-09-02** — **TAD promoted from a deferred Phase-7 adapter to a
  replication corpus.** The question stops being "does KIP help" and becomes
  "does the smoother ordering survive a second training distribution".
- **2026-09-03** — **DADA-2000 added as a third corpus** and a second zero-shot
  transfer target. Project-invented seeded 20 % stratified split; no published
  DADA number is pinned, so nothing here is a reproduction gate.
- **2026-09-06** — **the smoother's sign is corpus-dependent.** DADA-2000 inverts
  the MSAD ordering, and the cause is mechanical (median clip length 9 under a
  kernel-9 score head), not statistical. Two consequences adopted: (i) `auc_macro`
  plus a constant-score-per-clip oracle is now mandatory on any corpus with
  all-normal test clips; (ii) `score_head_kernel` must be checked against a new
  corpus's median sequence length before training.
- **2026-08-24** — PreVAD pulled forward from Phase 7 as a *trunk-transfer* track:
  the released features train a KIP-off trunk with no video and no flow, so the
  A/B can run on MSAD where flow exists. Answers the 1.5 %-of-design-point
  external-validity threat without waiting for raw video.

- **2026-09-08** — **the branches are given separate identities.** `main` is the
  **v1** line (frozen-MLP gate; all dataset adapters; `core/eda/`), `v3` is the
  gate rebuild. The memory bank is tracked per branch and the two copies now
  differ on purpose. Consequence adopted: **a result is recorded with the branch
  whose code produced it** (see the results ledger above), because
  `RESULTS_V3_GATE_ATTRIBUTION.md` does not exist on `main` and the numbers would
  otherwise be lost to this branch entirely.
- **2026-09-23** — **Option A authorized, construction A1.** The flow target is
  standardized at the **23 raw stats** (train-window moments, same projection), not
  at the 256-d `e_O` — standardizing after the projection would fix the units but
  leave every dimension dominated by `mag_max`. Accepted cost, stated before any
  measurement: the angle histogram becomes 16/23 = **69.6 %** of the target (plan
  risk 1, read out by R-1b). KIP-off arms are reused, not re-run, because they are
  inert to both knobs that change. Built in `ae4fded`; not yet run.
