# Progress

**Last updated:** 2026-09-08 (branch identity recorded: this is `main` = KAT-VAD
**v1**; counts and test status re-measured on this tree)

> **Branch:** `main`, tip `6a5f648` — the **v1** line. The v3 gate rebuild is on
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
| — | **`core/eda/` pre-flight profiler** — 5 modules + `core/tools/eda.py` + `core/docs/EDA.md`; verdicts for C27 / C12 / vanished windows, §4.2 frame-level linear probe | ✅ 2026-09-06, **present on `main`**. Never run on real data — needs the Drive caches |

**Measured on `main`, 2026-09-13: 81 Python files (55 source + 26 test),
11,560 source LOC. 508 tests collected → 508 pass, 0 fail.** Data-free, CPU-only.
*(2026-09-09 Phase 1 added `loss.dvs_anchor_mode`, `loss.bottomk_weight`,
`--equalize-length`: 439 tests. 2026-09-12 Phase 2a added `core/data/windows.py`,
the `windows.json` contract, the `--window-*` flags and the C30 eval fix: +54.)*
`outputs/**` holds **62,254** per-clip `.npz` score files (on the user's disk;
gitignored).

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

**Cross-branch comparability, settled 2026-09-12:** for `kip.enabled=false` a
`main`-vs-`v3` Δ **is** a Δ. `p1_ctrl` reproduces v3's A0 to 4 dp, and a
mis-pointed eval of the v3 checkpoint produced **331/331 bitwise identical** score
curves (max |Δ| = 0). This does **not** extend to any KIP-on arm.

**The two sentences that outlive every number:** *KIP's measured contribution is
temporal smoothing*, and *a component whose sign flips with training clip length
is a smoothing hyperparameter, not a motion mechanism.* On `main` this is not an
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
- [ ] **The campaign notebooks are untracked on `main`.** `colab/` is gitignored
      after the `collab/` → `colab/` rename, so `colab/{MSAD,DADA}/v3/train.py`
      — the code that actually ran the arms — is not in git here. Decide whether
      to whitelist `colab/**/*.py` or accept Drive as their only home.

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
- [ ] **TAD Gate T0** — `best.ckpt` on TAD against `TAD_ZERO_SHOT_AUC = 89.56`,
      then A2 and A0 across 3 seeds, read `A2 − A0` on DoTA. **An in-domain TAD
      number is not comparable to 89.56** (`TAD_SETUP.md` §0.1). This is now the
      *third* corpus and the tie-breaker between MSAD (+0.10) and DADA (−0.09).
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
- [ ] TAD / DADA / UCF-Crime adapters
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
