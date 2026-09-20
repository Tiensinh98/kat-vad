# Project Brief — KAT-VAD

**Created:** 2026-07-31 (memory bank re-initialized from commit `b9978ff`)
**Last reviewed:** 2026-09-21 (full reconcile — gate (c) restated: on T2, the
project's first *within-corpus paired* KIP A/B, KIP-on is **negative**, and D1/D2
attribute it to an unnormalized reconstruction target capturing the shared trunk).
Earlier: 2026-09-15, the DADA-2000 **original** release adopted as the training
corpus and DoTA pinned as held-out; the same day, TAD run end to end.

> **Branch:** `main` (tip `702bd5b`) is the **KAT-VAD v1** line — KIP as
> originally specified: `PMGFlowHead` → `KinematicShift` with the frozen
> 321-parameter MLP gate → `MotionScoreHead`. The v3 gate rebuild (four
> `gate_type`s, ECMR, gate diagnostics) is on branch **`v3`** (tip `bb1516c`),
> which carries its own memory bank. They diverged at `fac71a3`.
> **The scope, gates and results below are the project's, not the branch's** —
> they are recorded here in full so `main` keeps the campaign record. Where a
> result was produced by v3 code, it says so.

## What this is

**KAT-VAD** = Kinematics-Aware, definition-conditioned Traffic Video Anomaly
Detection. It is the **LaGoVAD** weakly-supervised, language-definition-conditioned
baseline **plus** a novel **Kinematic Induction Pathway (KIP)** that induces
optical-flow motion evidence at *training* time so inference stays RGB-only.

The gap being closed: LaGoVAD has no motion modeling, and its weakest published
benchmark is the motion-dominated DoTA.

## Source of truth

| Artifact | Role |
|---|---|
| `core/docs/KAT_VAD_PROPOSAL.md` | The *why* — proposal, gap analysis, design justification |
| `core/docs/KAT_VAD_IMPLEMENTATION_SPEC.md` | The *what/how* — modules, tensor shapes, losses, config flags |
| `core/docs/REPORT_KIP_MSAD_DOTA_PREVAD.md` | The measured campaign write-up (renamed from `REPORT_KIP_MSAD_DOTA.md` on 2026-08-29) |
| `core/docs/v3/KAT-VAD-ARCHITECTURE.md` **(branch `v3` only)** | The architecture, phase by phase (P1–P7) |
| `core/docs/v3/KAT-VAD_spec_v3.md` **(branch `v3` only)** | **Current spec** for the v3 line — changes A–K, each tied to a measured defect. Supersedes v2 |
| `core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md` **(branch `v3` only — on `main` the numbers live in [[progress]]'s results ledger)** | **The attribution result** (2026-09-01) — read before claiming KIP does anything |
| `core/docs/v3/RESULTS_DADA.md` | **The DADA-2000 campaign** (2026-09-06) — read before quoting any DADA number |
| `.project/plans/katvad-v3-kip-gate-rebuild.md` | **The live plan** — v3 Phase-3 rebuild, phases 0–5, measured appendices A–E |
| ~~`core/docs/v2/*`, `.project/plans/katvad-v2-next-steps.md`~~ | **Do not exist in this tree.** v3 supersedes them; do not cite them |
| `core/docs/PREVAD_SETUP.md` | PreVAD download/setup/trunk-transfer runbook |
| `.project/plans/kat-vad-implementation.md` | Phase plan 0–7 with per-task checkboxes |
| `LaGoVAD-PreVAD/` | Baseline reference. **READ-ONLY, never modify** |
| `core/` | The only place code is written |

On conflict between proposal and spec: read the proposal, decide deliberately.
The spec's inline code is illustrative — implement rationally, don't transcribe.
**Between spec versions:** the newest spec wins where it cites a measurement, and
the live plan wins over the spec where it cites code — the plan read the tree,
the spec did not. **A `RESULTS_*.md` measurement outranks both.**

## Scope

**In scope on `main` (v1):** MSAD (traffic slice + full 11-class benchmark)
training and evaluation; the **v1** KIP on/off ablation (one gate — the frozen
MLP — so a KIP-on run here *is* the fixed ~50 % shift); the reproduction gates;
the DoTA / PreVAD / TAD / DADA-2000 adapters; `core/eda/`. **Out of scope on
`main`:** anything keyed on `kip.gate_type` — that is the `v3` branch. **DoTA zero-shot was
pulled forward from Phase 7** (2026-08-01) and is now the benchmark that carries
the result — it is where a motion pathway can show, and MSAD is where it cannot.

**Pulled forward (2026-08-24):** **PreVAD** is no longer purely Phase 7. Its
released `ViT-B-16-8p` features make a KIP-**off** trunk trainable without any
raw video, so the plan is trunk-transfer: pretrain on PreVAD, graft KIP, run the
A/B on MSAD where flow already exists (`core/docs/PREVAD_SETUP.md` §7). Code
prerequisites are done; nothing has been run.

**Promoted to replication corpora (2026-09-02 / 09-03):** **TAD** and
**DADA-2000** are no longer deferred adapters. After the 2026-09-01 attribution
result, the live question is *"does the smoother ordering survive a second and
third training distribution?"* — DADA-2000 **inverts** the ordering
(`core/docs/v3/RESULTS_DADA.md`).

**TAD is now RUN (2026-09-14/15) and it did not answer that question — it raised
a bigger one.** Gate T0 fails at **0.7912** (diagnosed, not a defect: see
[[activeContext]] §3), and the in-domain arm **collapses into a clip classifier**
(clip-AUC 0.9975, macro 0.6174 against the zero-shot trunk's 0.7578). The
intended `M2 − M0` tie-breaker is therefore **blocked by C14** — an arm stacked
on a collapsed trunk measures the collapse. TAD's live value has shifted from
*"third smoother datapoint"* to *"the corpus that proved WS-MIL destroys
localization when the bag task is trivially separable."*

**Scope change 2026-09-15 — the training corpus becomes the DADA-2000 ORIGINAL
release.** `.project/plans/katvad-dada-original-corpus.md`. The reconstructed
archive this project has trained on is not repairable in place: it trims both
classes unequally (abnormal raw median 56, normal 152, original **322**), leaking
its label through clip length at **0.8105**, and two fixed-window rebuilds left
the clip oracle at **0.9766 / 0.8965**. The original release is re-sharded into
**W=16 windows whose negatives come from inside the accident videos themselves**
(leak **0.5000**, oracle **0.6631**, 98.1 % of abnormal clips retained), using the
window machinery already on `main`. Gated: Phase 0 P1 (per-clip frame counts) then
**D0**, a supervised frame linear probe that must reach **≥ 0.60** or the track
stops and ships a negative result.

**DoTA is now explicitly a held-out benchmark and must stay one.** It carries
`train_clips: 0` and **3** normal clips, so it cannot host in-domain training
anyway; and every comparison this project owns — `RESULTS_{DOTA,NCC,PHASE_A}`, the
v3 attribution, every transfer column — is defined by DoTA being unseen. The bar
the new corpus must beat is **0.6408**, the MSAD-trained KIP-on mean.

**Deferred (Phase 7):** PreVAD KIP-**on** (needs pixels the release does not
ship), UCF-Crime, the full metric suite (MCC family, AUC_A, mAP@IoU), Stage 0 /
Stage 0.5 / Stage 3 (ATS + MLLM reasoning head), validation-split checkpoint
selection. **`L_neg` / per-video captions** join this list: DADA's `texts` column
is a 83-value category label, not a description, and gap **G4** means no code
path reads a caption field at all.

## Reproduction gates

| Gate | Statement | Status (latest reading) |
|---|---|---|
| (a) | Our eval code reproduces LaGoVAD `best.ckpt` on the target benchmark | **PASSES.** DoTA 0.6142 vs published 0.6260 under the baseline's own per-clip min-max protocol (`RESULTS_DOTA.md`) |
| (b) | Our KIP-off model trained on the benchmark ≥ gate (a) | **PASSES as redefined.** Our arms are statistically indistinguishable from `best.ckpt` on MSAD — eight paired bootstraps, all CIs include zero (`RESULTS_PHASE_A.md` §4) |
| (c) | KIP-on ≥ KIP-off | **PASSES on DoTA when trained on MSAD** — seed-level t-interval **[+0.070, +0.113]**, n=3 — but **the gain is not KIP's**. A plain 50 % channel shift with no flow, no PMG head and no KIP losses reproduces all of it (`RESULTS_V3_GATE_ATTRIBUTION.md`, 2026-09-01). **FAILS when trained on DADA-2000**: A2 − A0 = **−0.0918** (`RESULTS_DADA.md`, 2026-09-06). MSAD in-domain remains a **bounded null**. **FAILS on T2, and this is the only within-corpus PAIRED on/off A/B the project owns** (2026-09-18, 3 seeds, configs differing in one line): T2 micro **−0.0129**, t95 [−0.0249, −0.0008], 3/3 seeds. **Diagnosed 2026-09-20 (D1/D2, lesson C37):** `L_KIP_rec` enters the objective ~32× oversized against an unnormalized `e_O` and captures the shared trunk (`rho` = 3.1) **orthogonally** (`cos` = −0.001). The gate (c) verdict therefore stands *as measured*, but the arms it was measured on were weighted wrong — the repair (**Option A**) is specified and **not authorized** |

**Gate (b) is stated against the released checkpoint, not the printed 0.9041.**
LaGoVAD's own `best.ckpt` reaches only 0.8991 (crop) / 0.8949 (ncc) through our
eval, so 0.9041 is not reachable with the published artifacts. Quote it as the
paper's number, never as a target this tree failed to hit (lesson 8b).

**Numbers live in `core/docs/RESULTS_*.md`**, in measurement order:
`RESULTS_MSAD.md` (center-crop) → `RESULTS_DOTA.md` (DoTA protocol fix; KIP
verdict superseded) → `RESULTS_NCC.md` (`no_center_crop` rebuild) →
`RESULTS_PHASE_A.md` (3 seeds) → `RESULTS_ARM4_PROBE.md` (warm-start control +
trajectory probe) → `RESULTS_PREVAD.md` → **`v3/RESULTS_V3_GATE_ATTRIBUTION.md`**
(the attribution) → **`v3/RESULTS_DADA.md`** (the second replication).
**`outputs/` is gitignored** — the **87,213** score curves exist only on the user's
disk and Drive, so those documents *are* the durable record. For the T2 campaign
and the D1/D2 diagnosis there is **no `RESULTS_*.md` at all**, so
`.project/memory-bank/{activeContext,progress}.md` and
`.project/plans/katvad-kip-loss-scale-diagnosis.md` are the whole record. See [[progress]] for
the current headline.

## Non-negotiables

- `LaGoVAD-PreVAD/` is never modified.
- **Branch discipline.** `main` = v1, `v3` = the gate rebuild. Do not write v3
  gate code on `main`; do not assume a v3 doc or flag exists here. Check
  `git branch --show-current` and `grep gate_type core/config.py` before running
  any runbook command.
- KIP's only splice into the baseline forward pass: between temporal encoder and
  fusion; `v^k` (not `v^t`) feeds fusion `U` and `H_bin`.
- RAFT flow is train-time only — never on the inference/scoring path.
- Code standards + quality gates per `CLAUDE.md` §10–11.
- **Do not tune KIP against any measured delta** (lesson 14). The MSAD +0.09 is
  attributed to temporal smoothing and **does not replicate on DADA-2000** — it
  inverts. Tuning against either number is fitting a benchmark.
- **The v1 gate MLP is never trained** (`core/kip/gate_shift.py:112`, verified:
  all 6 tensors `grad is None`) **and is measurably near-constant** — sweeping its
  entire reachable input domain moves `s_t` by 0–4 of 128 channels. **H4′ is
  CONFIRMED**: never write "motion-gated" of `gate_type="mlp_frozen"` — and on
  `main` that frozen gate is the *only* gate that exists.
- **No component of KIP has been attributed a positive contribution** by any
  ablation. State that plainly; do not restate the +0.09 as a KIP result.
- **Check `score_head_kernel` against a corpus's median sequence length before
  training on it.** `ConvScoreHead(kernel_size=9)` over DADA's median-9 clips is a
  clip classifier, not a frame detector (`RESULTS_DADA.md` §5).
- **On any corpus with all-normal test clips, report `auc_macro` and the
  constant-score-per-clip oracle beside the micro AUC** — the mirror of lesson
  C12. DADA's oracle is 0.9086; **TAD's is 0.9226, above the published 89.56.**
- **A failed reproduction gate is diagnosed, never chased** (2026-09-15).
  TAD's T0 = 0.7912 vs a published 89.56 is *not* a pipeline defect:
  `auc_macro = 0.7578` exonerates frame ordering in one step (shuffled frames
  give ≈0.50), labels match LaGoVAD's own annotation 100/100, and the residual is
  length weighting. **Per C8b the measured 0.7912 is the reference; 89.56 never
  appears in a TAD table again.**
- **Weakly-supervised MIL optimizes whatever is cheapest to separate.** On TAD
  the two classes are two directories with different length distributions, so a
  per-clip constant satisfies `L_MIL` completely and localization is never
  learned. Warm-starting from a trunk that *could* localize (macro 0.7578) still
  ends at 0.6540 — **the training signal destroys 0.104 of macro it was handed.**
  Before training on a new corpus, compute its clip oracle and length ruler; if
  micro ≈ oracle, in-domain training will buy clip ranking and sell localization.
