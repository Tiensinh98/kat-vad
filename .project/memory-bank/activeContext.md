# Active Context

> ## ⚠️ BRANCH IDENTITY — READ BEFORE ANYTHING ELSE
>
> **This memory bank is the `main` branch's copy, and `main` is KAT-VAD **v1**.**
> The v3 KIP gate rebuild is **not** in this tree; it lives on branch **`v3`**,
> which carries its own copy of this file. The memory bank is git-tracked per
> branch and the two copies have **deliberately diverged** — never resolve a
> merge between them with "take theirs".
>
> | | `main` (this branch) | `v3` |
> |---|---|---|
> | tip | `6a5f648` | `bb1516c` |
> | diverged at | `fac71a3` ("docs: Update result for PreVAD") | same |
> | KIP | **v1 only** — `PMGFlowHead` → `KinematicShift` (frozen 321-param MLP gate) → `MotionScoreHead` | v1 **+** four selectable `gate_type`s, ECMR, gate diagnostics |
> | `kip.gate_type` | **does not exist** (`core/config.py` raises `KeyError`) | `rank` (default) / `mlp_frozen` / `mlp_ste` / `constant` |
> | `core/kip/ecmr.py` | absent | present |
> | `train_only_modules`, `--dump-kip-diag` | absent | present |
> | tests | **439 collected → 439 pass, 0 fail** (2026-09-09, after Phase 1) | 537 green |
>
> **Every measured result recorded below was produced by the `v3` branch's code.**
> They are kept here on purpose: `outputs/` is gitignored and
> `core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md` **is absent on `main`**, so for
> this branch *this file and `progress.md` are the only durable record of the
> attribution campaign*. Do not delete them; do not re-run those arms here.

**Last Memory Bank Update:** 2026-09-08 (branch split recorded: `main` = v1,
`v3` = v3; all counts re-measured on this tree)

## 2026-09-09 (latest) — Phase 1 shipped: three arms, all default-off

**Suite: 439 collected, 439 pass** (423 + 16). `ruff` / `mypy` / `pyright` /
`pycycle` clean; `bandit` 0 High. Files touched: `core/constants.py`,
`core/config.py`, `core/losses/{dvs,mil,__init__}.py`, `core/train.py`,
`core/evaluate.py`, `core/docs/{TRAINING,DIAGNOSIS_...}.md`, 3 test files.

### The EDA re-run confirmed the C28 check works

The user re-ran `core.tools.eda` on Colab. **DADA**: a 5th verdict now fires —
`[CRITICAL] Clip length alone predicts the label (C28)`, clip AUC **0.8105**,
micro **0.8654**, **107** normal clips (3,157 frames) outside the abnormal
length range. **DoTA**: §3.3 reads 0.5280 / 0.4993 / **0** disjoint clips and
**no C28 verdict**. The check is discriminative, not always-on — C28 is a defect
of the *reconstructed DADA build*, not of dashcam corpora.
*(Stale: `outputs/eda/DADA2000_A2_s2024/` is from Sep 8, pre-C28 code. Re-run
`EDA.md` §2.1 if the scored-run block is wanted with §3.3.)*

### What shipped

| Arm | Flag (default) | Effect |
|---|---|---|
| 1.1 (C29) | `loss.dvs_anchor_mode` = **`span`** \| `ignore` | `ignore` drops the anchor interior from the dense DVS BCE; fillers stay hard negatives, `pseudo_sup_mil_loss` supplies the positives |
| 1.2 | `loss.bottomk_weight` = **`0.0`** (+ `bottomk_topk_pct` = 16) | `abnormal_bottomk_loss`: the only term that lowers a frame inside a positive bag; normal clips excluded (`L_MIL` already bounds them) |
| 1.3 (C28) | `--equalize-length N`, `--equalize-anchor center\|start\|end` | Eval-only control: one length for every clip, shorter ones dropped, retention recorded |
| 1.4 | — | **Deferred to Phase 2** on purpose: `mil_topk_pct` does nothing at median T = 9 |

### The constraint that shaped the design

The blast-radius trace surfaced that `supervised_loss` / `mil_loss` are asserted
against the **vendored LaGoVAD reference** by `core/tests/test_baseline_parity.py`.
So every new parameter is **keyword-only with a baseline default**, and the
bottom-k term is *skipped* at weight 0 rather than computed and multiplied by
zero — the default graph and the logged loss keys stay byte-identical. A bad
`dvs_anchor_mode` **raises**; a typo must not leave an arm silently off.

Two numerical guards: `supervised_loss` divides by `mask.sum().clamp(min=1.0)`
(all-positive row under `ignore` → 0 *with* a grad_fn, not NaN; unreachable in
the default mode, so parity is exact), and `abnormal_bottomk_loss` returns
`logits.sum() * 0.0` on an all-normal batch.

### ⚠️ The codebase-memory graph under-reports callers

`trace_path` on `supervised_loss` returned **empty**; a Cypher `MATCH
(caller)-[:CALLS]->(f)` found only the **test** callers and **missed
`core/train.py` entirely** for `supervised_loss`, `pseudo_sup_mil_loss`,
`mil_loss` and `multi_class_mil_loss`. The real callers were found by reading.
**Do not treat `trace_path` output as a complete blast radius on this repo** —
confirm with `grep` before concluding a symbol is safe to change. Re-indexing
after this commit may or may not fix it; the gap was present on a "ready" index
with 2,656 nodes.

### The runbook is written and every command was parse-tested

`core/docs/DADA_SETUP.md` **§10.2** is the Phase 1 campaign, runnable on `main`:
**§10.2.0** the free eval-only C28 control on the existing A0 checkpoint;
**§10.2.1** four training arms (P0 control / P1 `dvs_anchor_mode=ignore` /
P2 `bottomk_weight=1.0` / P3 both) on a **KIP-off trunk** — no flow cache, no
stage 1; **§10.2.2** three evals per arm (DADA raw, DADA eq5, DoTA zero-shot);
**§10.2.3** the report-back table; **§10.2.4** pre-registered readings.

Every `--set` in §10.2 was run through `load_config` on this branch, both shell
blocks were expanded in bash, and `kip.gate_type` was confirmed to still raise.
**§10.1's A2 block is now marked ⛔ unrunnable on `main`** — it uses
`kip.gate_type`, `kip.const_shift_ratio` and `kip.disable_pmg`, none of which
exist here. `main`'s full KIP flag set is
`kip.{enabled, pmg_only, use_gate_shift, use_lkin, gate_signal, on_raw_features}`.

**P0 exists because A0 was trained on `v3`.** Diffing a `main` arm against
`RESULTS_DADA.md`'s 0.7050 / 0.5190 would be a cross-branch comparison. If P0
lands far from those, that gap is itself the finding.

**`bottomk_weight=1.0`** is a symmetry prior (same top-/bottom-k BCE as `L_MIL`,
opposite direction), **not tuned**; only a stability failure justifies 0.5.

### The eq5 control — measured geometry, so the run can be checked

`--equalize-length 5 --equalize-anchor end` on DADA: **331/383** clips kept
(52 dropped), 162 abnormal kept of which **156** still hold a positive,
**346/476** positives retained (72.7 %), **155** two-class clips for macro.
Length-only baseline becomes **0.5000 exactly**; the clip oracle falls
0.9086 → **0.8342**. So it removes **C28, not C12** — `auc_macro` stays the
honest metric. `N=7` was rejected: only 105 two-class clips left.
`anchor=end` because DADA's accident sits at the clip end.

### Pre-registered readings are in the doc, not here

`DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` §6 Phase 1 now carries the run commands
and a falsification table for each arm. **Judge all three on `auc_macro` and the
clip-mean-removed micro, never raw micro** — 1.1 and 1.2 are expected to *lower*
raw micro while improving the model. And expect both to under-deliver until
Phase 2: at median T = 9 under a 9-tap head and a global temporal window, no
loss can buy resolution the architecture does not have. Phase 1 isolates that
claim for free.

---

## 2026-09-08 — Phase 0 shipped: the reporting is fixed and two published numbers were wrong

**Code changed** (first code change on this branch since the test collapse):
`core/eda/protocol.py`, `core/eda/report.py`, `core/tests/test_eda.py`.
**Suite: 423 collected, 423 pass** (418 + 5 new). `ruff`, `mypy`, `pyright`
clean on the touched files.

### What shipped

* **`protocol.clip_length_leak()`** — the C28 check, wired into
  `protocol_report` (guarded: a single-clip-class corpus records `skipped`
  rather than raising, so DoTA-shaped corpora keep working), rendered as report
  **§3.3**, given a **CRITICAL** verdict above clip-level AUC **0.65**
  (`LENGTH_LEAK_WARN_AUC`), and added to the cross-corpus comparison table.
  Report sections renumbered: resolution 3.3 → **3.4**, scored run 3.4 → **3.5**
  (`core/docs/EDA.md` updated to match).
* **Blast radius traced first** (`trace_path`): `protocol_report` ←
  `build_report` ← `_run_report` ← `main`, all inside `core/eda/` +
  `core/tools/eda.py`. Nothing on the training, scoring or metrics path consumes
  it, and the change only *adds* a key. Real risk LOW; the tool's hop-distance
  labels say CRITICAL and are not meaningful here.
* **Docs:** `RESULTS_DADA.md` (correction banner, rebuilt §3 headline table with
  a `clip-mean removed` column and both baseline rows, new §3.1a / §4.1 / §4.2,
  §9 limitations, §10-A and §10-B marked done), `DADA_SETUP.md` §5.1,
  `DADA_V3_SETUP.md` pitfalls, `EDA.md`, `DIAGNOSIS_...md` Phase 0.

### Two published numbers were wrong

1. **The DADA clip oracle is 0.9086, not 0.9069.** `RESULTS_DADA.md` §4 used the
   3,880-frame `0_Normal_Driving` **subgroup** count instead of the
   all-normal-**clip** count (3,896). The 16-frame gap is **exactly** the four
   vanished-window clips — abnormal in `meta.json`, all-zero after stride-8
   rounding, therefore all-normal *clips* for every metric. **Phase 0.1 and
   Phase 0.2 were the same defect surfacing twice.** Corrected in 9 files
   (`protocol.py`, `EDA.md`, `RESULTS_DADA.md`, `DADA_V3_SETUP.md`,
   `activeContext`, `progress`, `projectbrief`, `systemPatterns`, lessons
   `index`/`detailed`/`meta-index`) and pinned by a renamed test.
2. **The length-only baseline was missing**, and it is the one that matters:
   0.8654 (383 clips) / **0.8681** (379). Best arm 0.8756. Margin **+0.0075**.

### The vanished-clip exclusion, computed (Phase 0.2 — no GPU, no re-eval)

Offline from the saved `.npz`. Micro moves **+0.001–0.002** per arm;
**`auc_macro` is unchanged to 4 dp on every arm** (single-class clips were
already skipped by `macro_video_auc`). Oracle 0.9086 → 0.9082; length-only
0.8654 → 0.8681 — the baseline rises *more than any arm does*. Full table in
`RESULTS_DADA.md` §3.1a. **Quote the 379-clip column from now on.**

### The leak is DADA-specific

Same function, same `.npz` files, DoTA test split: clip-level AUC **0.5280**,
micro **0.4993**, **0** clips separable by length. So C28 is a defect of the
*reconstructed* DADA build, not a property of dashcam corpora — and TAD must be
checked before it is trained, not after.

### Next

Phase 1 (code arms, no re-extraction): DVS anchor-*ignore* (C29), bottom-k MIL,
length-controlled eval. Phase 2 (corpus rebuild, fires **C2**). Phase 3 (probe →
the backbone decision). All in `DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` §6.

---

## 2026-09-08 — the DADA failure is attributed: a ruler scores 0.8654

**No code changed.** Diagnosis only, from the `core/eda/` reports in
`outputs/EDA/` plus a re-read of the 383 + 1,397 per-clip `.npz` curves already
on disk. Written up in **`core/docs/DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md`**;
two lessons added (**C28**, **C29**, both CRITICAL).

### The finding

The DADA arms are **clip classifiers, not frame detectors**, and three
independent defects each force that outcome:

1. **C28 — the corpus leaks its label through clip length.** Abnormal test clips
   are **max 17** stride-8 frames; **all 107 clips with T >= 18 are normal**. A
   constant-score-per-clip detector reading only `-T` scores **micro AUC
   0.8654 / AP 0.2630** — within **0.0085** of the best trained arm (0.8739) and
   ahead of four of the seven. This is a property of the **reconstructed**
   DADA-2000 build (accident videos trimmed, normals full length), visible in
   `eda_report.md` §1.4 as 7.0 vs 20.6 sampled frames per clip.
2. **C27, extended — three whole-clip mixers, not one.** `RESULTS_DADA.md` §5
   blames `ConvScoreHead` (kernel 9 >= median T 9). It is also
   `TEMPORAL_WINDOW = 25` -> `half_window = 12`, so **every token is already a
   function of every frame after transformer layer 1**. Nobody has changed
   `temporal_window`; it is the largest of the three spans.
3. **C29 — DVS labels the entire anchor clip positive.** `synthesis.py:83` sets
   `pseudo[anchor_span] = 1.0` over the whole anchor and `supervised_loss` eats
   it as a **dense** per-frame BCE. DADA abnormal clips are only **35.1 %**
   truly positive, so **64.9 %** of anchor frames are trained to 1 against a 0
   annotation. And **no term in the objective pushes any frame of an abnormal
   clip down** (checked all four wired terms, `train.py:285-336`).

### The numbers that settle it

| | |
|---|---|
| micro, best arm (A2 s2025) | 0.8739 |
| micro, **clip-mean removed** (A2 s2024) | **0.4943** — 100 % of micro is clip ranking |
| clip-level AUC (per-clip mean vs clip label) | **0.9260** |
| length-only detector | **0.8654** |
| clip oracle (perfect ranking, zero localization) | 0.9086 |
| pos / neg-in-abnormal / all-normal mean score | **0.4076 / 0.4078 / 0.0493** |
| argmax frame is a true positive | **28.3 %** vs 35.1 % base rate (6 of 7 arms below chance) |

`dvs_sup` converges 0.681 -> 0.027: the model fits the wrong pseudo-label
almost perfectly.

### The transfer story

The arm that fits DADA **worst** transfers **best**: A0 KIP-off, clip-AUC 0.767,
DoTA macro **0.6254**. A2 s2025, clip-AUC 0.934, DoTA macro **0.5086**. Textbook
shortcut learning. (Spearman(clip-AUC, DoTA macro) = −0.39 across 7 arms,
p = 0.38 — **suggestive only**, n is 7 single-seed arms.)

### The plan (doc §6)

Phase 0 reporting (zero GPU) -> Phase 1 code arms with no re-extraction
(DVS anchor-ignore, bottom-k MIL, length-controlled eval) -> Phase 2 corpus
rebuild: **fixed-length windows**, `frame_stride` 8->2, `score_head_kernel` 9->3,
`temporal_window` 25->9, `mil_topk_pct` 16->8, new cache path (**fires C2** —
invalidates every DADA number incl. all of `RESULTS_DADA.md`) -> Phase 3 re-run
the probe. Pre-registered readings are in the doc; per lesson 14 nothing here is
tuned against a delta.

### Backbone question — answered: not yet, and not as a swap

Asked directly by the user. Doc §7. On **DoTA** the frozen-CLIP frame probe
(**0.6708** macro) already **beats every trained arm** (best 0.6254) — the
deficit is supervision, not representation. On **DADA** the probe reads 0.5228,
but it was measured under C28+C27 and is not admissible; feature statistics are
indistinguishable from DoTA's (lag-1 cosine 0.963/0.965, B/W variance 2.72/3.19,
419/512 dims). Three blockers on a swap: it voids the LaGoVAD claim and the C8b
checkpoint gate; **VideoMAE has no text tower, so `H_mul` / `mul_mil` /
`L_neg` silently die** — only a CLIP-aligned video model (ViCLIP, X-CLIP,
InternVideo2-CLIP) preserves the definition conditioning; and it invalidates
every cache in the project (C2/C13) while being measured *through* C27–C29.
**Buy the answer instead:** a linear-probe A/B of video vs CLIP features on the
DoTA test split (~1-2 GPU-h, no training), with a pre-registered decision rule
(>= +0.10 macro -> adopt as a second stream; within ±0.03 -> keep CLIP). Free
first step: a `Δf_t = f_t − f_{t−1}` stream from the **existing** cache.

### `RESULTS_DADA.md` §10 status

**§10-B (frame-level probe) is now run** — 0.5228 DADA / 0.6708 DoTA. §10-A
(reporting fix) is specified but **not yet applied**. §10-C (stride 2 +
kernel 3) is now Phase 2 and is justified. §10-D (`mlp_ste` NaN) still open.

---

## 2026-09-08 — `main` is the v1 branch; the memory bank now says so

**No code changed.** This entry exists because the previous memory bank was the
**v3** memory bank sitting on `main` — it described `ecmr.py`, four gate types,
84 files and "537 tests green", none of which are in this tree.

### What this tree actually contains (measured 2026-09-08)

| Fact | `main` |
|---|---|
| Python files | **79** — 54 source + 25 test |
| Source LOC | **10,484** |
| Tests | **423 collected: 423 pass, 0 fail** (418 at the time of this entry; +5 later the same day, Phase 0) |
| KIP | v1: `pmg.py`, `gate_shift.py` (`KinematicShift`, frozen MLP gate), `motion_head.py`, `kip_module.py`, `losses.py` |
| Adapters | MSAD, DoTA, PreVAD, **TAD**, **DADA-2000** — all present |
| `core/eda/` | present (5 modules + `core/tools/eda.py` + `core/docs/EDA.md`) |
| `core/docs/**` | **20** markdown files |
| `core/docs/v3/` | **only** `RESULTS_DADA.md` + `setup/{DADA_V3_SETUP,TAD_V3_SETUP}.md` |

**Absent here, present on `v3`:** `core/kip/ecmr.py`; `kip.gate_type` and
`kip.const_shift_ratio` / `kip.gate_signal` in `core/config.py`;
`KIP.train_only_modules`; `--dump-kip-diag` and the `kip_s` / `kip_gate_ratio` /
`kip_m` columns in the score `.npz`; the gate-type guard in
`core/models/ckpt_compat.py`; and the docs
`v3/{KAT-VAD-ARCHITECTURE, KAT-VAD_spec_v3, RESULTS_V3_GATE_ATTRIBUTION,
KAT-VAD_audit_addendum_PreVAD}.md` plus `v3/setup/MSAD_DOTA_V3_SETUP.md`.

### The 12 failing tests were a branch artifact — collapsed, suite now green

```
core/tests/test_dada.py::TestDadaTrainsUnderEveryGate  — 5 failures
core/tests/test_tad.py::TestTadTrainsUnderEveryGate    — 7 failures
KeyError: 'Unknown config key: kip.gate_type'   (core/config.py:207)
```

Both test classes were written during the v3 rebuild and parametrized the arm
matrix over `kip.gate_type` (`v1_mlp_frozen`, `v1_mlp_ste`, `v3_rank`,
`v3_constant`). `core/config.py` on `main` has no such field and **raises by
design** — unknown keys are a hard error (a deliberate decision, see
[[systemPatterns]]). Nothing about the data adapters, the model or the metrics
was broken.

**Fixed 2026-09-08 by option 1 — collapse, not delete.** Both classes are now
`TestDadaTrains` / `TestTadTrains` and run the one gate v1 ships,
`V1_KIP_ARM = ["--set", "kip.gate_signal=flow_norm"]`. The v3-only
parametrizations are gone (**7 of them**, 425 → 418 collected); everything that
passed still passes: `test_kip_off_trains` (arm A0),
`test_stage1_warmup_runs` (TAD stage 1, the A1/A2b/A3/A4 precondition) and the
`config.yaml`-recording tests, which now assert `kip.gate_signal` instead of
`kip.gate_type`. `ruff` and `mypy` clean on both files.

The rejected alternative was **port the v3 gate to `main`** — only sane if `main`
is meant to grow past v1, which is *not* the current intent.

**Do not "fix" a future recurrence by adding `gate_type` to `core/config.py`
alone.** The v3 tests also expect `ecmr.py`, the STE shift and the diagnostics; a
partial port would turn 12 loud failures into a silently wrong gate.

### Runbooks on this branch that this branch cannot run

`core/docs/v3/setup/DADA_V3_SETUP.md` (24 references) and `TAD_V3_SETUP.md` (23)
prescribe the six-arm ladder A0–A4 keyed on `--set kip.gate_type=…`. **Every one
of those commands fails on `main` at config parse.** `core/docs/DADA_SETUP.md`
(6 references) and `TAD_SETUP.md` (1) have the same problem in their arm
sections; their *data-build* sections are fine and are what `main` is for.
**Run the arm ladder on `v3`. Build data on either.**

### Working-tree state (uncommitted, at the time of writing)

- `.gitignore`: `collab/` → `colab/`; the notebook directory was renamed on disk
  (`collab/MSAD/train.py` shows as deleted). `colab/` is now **gitignored**, so
  `colab/{MSAD,DADA}/v3/train.py` are untracked — the notebooks that ran the
  campaigns are **not** in git on this branch.
- The real, tracked memory bank is `.project/memory-bank/`. Adding the
  symlink to git is not useful; the files behind it are already tracked.

### What `main` is *for*, going forward

`main` = the **v1 result line**: the frozen-gate KIP as originally specified,
plus every dataset adapter and the EDA profiler. It is the branch that owns the
v1 story — including the honest one, that v1's "motion-gated adaptive shift" was
a fixed ~50 % smoother. Architecture work on the gate belongs on `v3`.

---

## 2026-09-06 — `core/eda/` shipped: measure the corpus before the campaign

**New code.** Package `core/eda/` (5 modules) + CLI `core/tools/eda.py` +
`core/tests/test_eda.py` (**40 tests**) + runbook `core/docs/EDA.md`.
**Tree: 84 Python files (55 source + 29 test), 11,157 source LOC, 537 tests
green**, ruff / mypy / pyright / pycycle clean, CPU, data-free.

**Why.** Every structural finding in `RESULTS_DADA.md` — the kernel-9 head over
9-frame clips (**C27**), the 0.9086 clip oracle (**C12**), the 4 vanished
windows — was computable from `frame_labels_test.json` in seconds, and was
instead discovered after a seven-arm campaign. The tool turns each into a
pre-flight check. Lesson **27** gained an *Enforcement* clause naming it, and
three trigger-map rows now point at `core/docs/EDA.md`.

**What it computes** (`python -m core.tools.eda report|compare`):

- **§0 Verdicts** — CRITICAL/HIGH/INFO findings with the measurement and the
  action. Thresholds are named constants at the top of `core/eda/report.py`.
- **§1** splits, class balance, clip-length percentiles, subgroups
  (`fault_label`/`anomaly_class`/`ego_involve`), DVS length and steps/epoch;
  **§1.2** score-head receptive-field coverage; **§1.3** the MIL `k=1` share.
- **§2** positives per clip, span counts/lengths, multi-span clips (**C18**),
  and the **vanished-window ids** (emitted for exclusion at scoring).
- **§3** frame share by clip kind, the **constant-score-per-clip oracle**, the
  cross-clip pair fraction (what share of micro AUC localization can even
  touch), per-clip AUC resolution, `--score-norm auto` resolution; **§3.4**
  between/within score variance for a trained arm via `--scores-dir`.
- **§4** CLIP feature norms, effective dims, **temporal autocorrelation** by lag,
  feature variance decomposition, RAFT descriptor distribution — and **§4.2 the
  frame-level linear probe**, i.e. `RESULTS_DADA.md` §10-B, the open decisive
  experiment. Grouped CV by clip; a clip-level probe runs beside it as contrast.
- `compare` renders a cross-corpus table from finished JSON reports.

**Reading §4.2 (the whole point):** frame-probe `auc_macro` >= 0.60 means the
frozen features DO carry frame-level signal and the deficit is **supervision**;
at ~0.50 the **representation** is the ceiling and frame-level work needs a
different backbone, not another KIP variant. It is an upper bound, never a
tuning target (lesson 14).

**Guardrails built in:** a feature array whose rows disagree with its label
vector **raises** (C2/C13); a zero-byte score `.npz` **raises** (C11b); a split
leak is logged as an error; `--strict`-style "gates" are not confused with
repairs. Verified end-to-end on synthetic DADA-shaped and DoTA-shaped corpora:
the DADA-shaped one fires C27 + C12 + vanished-windows, the DoTA-shaped one
correctly fires neither of the first two.

**Not yet run on real data** — it needs the Drive caches, so the first real
numbers come from Colab.


## 2026-09-06 (later) — `collab/DADA/v3/train.py` audited; both DADA runbooks rewritten

**Docs only — no code changed, tree still 76 files / 497 tests.**

**Audit of the notebook that ran the campaign.** Four real defects; the rest of
the suspicion list came back clean and is written down so it is not re-checked:

1. **`--strict` is a gate, not a repair** (`core/data/dada.py:417-432`). The 4
   vanished clips enter the *test* split with all-zero labels and read as
   genuine normals, inflating both the micro AUC and the 0.9086 clip oracle.
   `build_frame_labels` is test-only (`:557`) → **fixable with no retraining.**
2. **`clip/DADA2000` does not record its transform.** Extraction ran
   `--no-center-crop`; every other cache in the project carries `_ncc`, and
   `extract_clip_features.py` writes no manifest. Naked lesson C2 exposure.
3. **`--motion-key` vs `dvs.motion_aware_knn_key`.** DADA's KNN cache is
   motion-aware, MSAD's is not (uncontrolled cross-corpus difference), and the
   config field has **no consumer anywhere in `core/`** — so every arm's
   `config.yaml` says `false` while the cache it ate says otherwise. C17 again.
4. **No gradient clipping in `core/train.py`**, and `scaler.step()` without
   `scaler.unscale_()`. Leading hypothesis for the A4 NaNs: `mlp_ste` is the
   only arm whose gradient reaches the gate MLP / PMG head through the score
   path, so it is the only one exposed to an unclipped gradient through the
   sigmoid surrogate under fp16. Falsify with one `train.amp=false` re-run.

**Cleared** (do not re-audit): missing `--score-norm auto` on the rank/DoTA eval
(parser default *is* `auto`); missing `disable_pmg`/`is_egocentric` at eval
(train-time only, `train.py:593`/`:600`); A2b/A4 warm-starting from a sibling's
stage 1 (key sets match, `warm_start_model` is fail-loud); A2's PMG head
contaminating its score path (dead subgraph — `ê_O` reaches only `mhead` and the
align projections, all three losses zeroed); A2-cold vs A1-warm as a trunk
confound (stage 1 freezes all but `kip.*`); step-budget parity with MSAD
(both 500); RAFT/CLIP field-of-view parity; `--exclude-unannotated-abnormal`
(correctly unset — windowless clips are forced to train).

**Runbook rewrite.** `DADA_V3_SETUP.md` and `DADA_SETUP.md` prescribed
`clip/DADA2000_ncc`, `knn/DADA2000_ncc`, `$OUT/DADA_<arm>_s$S` — **paths that
never existed.** Both now use the as-run names, and carry:

- **§3.0 / §10.1:** `S` (train seed, sweep it) vs `core.data.dada --seed`
  (split seed, never touch it) called out explicitly — the trap a 2025/2026
  sweep walks into.
- **Six full copy-paste arm commands at `S=2024`** (`DADA_V3_SETUP.md`
  §3.1–§3.3) plus six eval blocks (§4.1), `E = 20` pinned, `--init-weights`
  sources named per arm, and "already run at seed X" on each.
- **§3.2c** now records that stage-1 convergence **FAILED** on the 2024 run →
  lesson 14 blocks A1/A2b/A3, not A0/A2.
- **§5.1** in `DADA_SETUP.md`: the vanished-window audit + the exclude-at-scoring
  repair (not `--stride 4`, which fires C2).
- **§1.2a / §7:** a `TRANSFORM.txt` marker cell for the cache, and pitfall rows
  for the split seed, the AMP/NaN arm, and the DADA micro-without-oracle trap.
- `DADA_SETUP.md` §10.1 duplicates **A0 and A2 only** (the flow-free arms that
  validate the data build) and names `DADA_V3_SETUP.md` §3–§4 as source of truth.

Three candidates now in `lessons-learned/pending.md`: the `mlp_ste` NaN (existing
entry, gained the no-grad-clipping mechanism), "a runbook path is not a fact
until checked against disk", and "`--strict` is a gate, not a repair".

**Note:** `.project/scripts/lesson_checks.sh` (CLAUDE.md §7) **does not exist**
in this tree — the CI check cannot be run.

## 2026-09-06 — DADA-2000: the ordering inverts, and the in-domain number is a mirage

Full analysis: **`core/docs/v3/RESULTS_DADA.md`**. Runbook:
`core/docs/v3/setup/DADA_V3_SETUP.md`; notebook `collab/DADA/v3/train.py`.
Artifacts: `outputs/v3/DADA2000/**` (7 arms × 2 evals + `gate_d0`).
**No code changed — analysis only, so the tree is still 497 tests / 76 files.**

### The two findings

**1. The pre-registered prediction failed.** `DADA_V3_SETUP.md` §0 registered
`A2 − A0 > 0` and `A1 − A2 <= 0` on zero-shot DoTA. Both **invert**:

| Δ, DoTA micro | MSAD-trained (n=3) | DADA-trained |
|---|---|---|
| **A2 − A0** | **+0.1025 ± 0.0350** | **−0.0918** (and −0.1089 with A2 s2025) |
| **A1 − A2** | −0.0683 | **+0.0300** |

A2's second seed (2025, DoTA 0.4980) agrees with 2024 (0.5151), so this is not
seed noise. More seeds would sharpen the interval, not the sign. This is the
runbook's "**Ordering inverts**" row.

**2. The in-domain DADA 0.86 is a clip-level number, not a frame-level one.**
74 % of the 5,244 test frames come from all-normal `0_Normal_Driving` clips, so a
model emitting **one constant score per clip** scores **micro AUC 0.9086**. Best
arm: A2 s2025 at **0.8739 — 96 % of that oracle** — with `auc_macro` **0.5716**.
Every arm's `auc_macro` is 0.44–0.57, i.e. **chance**; A2b is 0.4399, *below* it.
**No arm has demonstrated frame-level localization on DADA-2000.**

### Mechanism — one cause explains both

DADA's median clip is **9** stride-8 frames (MSAD 86, DoTA 13; 55 % of DADA clips
are ≤ 9). `ConvScoreHead` is a single `Conv1d(512→1, kernel_size=9,
padding_mode="replicate")` (`core/models/heads.py:21`), so **every output timestep
sees the whole clip**. Supervision is at the same floor: `_topk_k` is
`max(1, n // topk_pct)`, so `L_MIL` on a 9-frame clip is a plain max over 9.
Labels too: median **2** positive frames per abnormal clip, 53 clips with exactly
1, 4 clips whose window vanished at stride 8.

Measured consequence — between/within-clip score-variance ratio vs AUC across all
7 arms: **Spearman(flatness, DoTA micro) = −0.82**, **Spearman(flatness, DADA
micro) = +0.68**. The flatter the curve, the better it separates *clips* (which
DADA's micro metric rewards) and the worse it does anything DoTA's per-clip
min-max protocol can read. On MSAD (T ≈ 86) the 50 % shift is a mild blur and
helped; on DADA (T ≈ 9) it collapses the curve to a constant.

**Consequence for the thesis: a component whose sign flips with training clip
length is a smoothing hyperparameter, not a motion mechanism.** This weakens the
MSAD +0.09 further rather than replicating it.

### Why our DoTA transfer is 0.6x and SimpleTAD reports 0.80

Asked directly by the user. [SimpleTAD, ICCVW 2025](https://arxiv.org/abs/2507.09338)
reports **DADA-2000 → DoTA = 80.3**. It is a *different method class*: VideoMAE-B
**fully fine-tuned**, sliding 224×224×**16 frames @ 10 FPS**, **per-frame binary
labels**, ≈2.5 M sampled windows. We are frozen CLIP **image** features,
**video-level labels only**, stride 8 ≈ 3.75 FPS, **500 optimizer steps**.

Decomposition: 0.803 − **0.6423** (our best DoTA number ever, MSAD-trained A2)
= **0.161 method ceiling**; 0.6423 − 0.6069 (best DADA-trained arm, A0) =
**0.035 corpus cost**. **82 % of the gap is the method class, not the DADA run.**
LaGoVAD's own released `best.ckpt` reaches 0.6142. 0.80 is not reachable without
abandoning the frozen-CLIP WS-VAD baseline.

### Defects found

1. **`mlp_ste` (A4) trained through 33 NaN steps of 500**, over 17 of 20 epochs,
   always `mil` + `mul_mil` while `kip_rec`/`kip_align` stayed finite — so the NaN
   enters the score path *after* KIP. No other arm has one at the same seed,
   batch and data → specific to `shift_channels_straight_through`
   (`core/kip/gate_shift.py:99`) under `amp: true`. **A4's DADA row is not
   trustworthy.** Not root-caused.
2. **The PMG head is far from converged on DADA** — stage 1 moves `kip_rec`
   121.7 → 41.6 in 500 steps and is still falling; stage 2 ends at ≈22–30
   (MSAD opened stage 2 at 7.4–8.5). Lesson **C14** applies to A1/A2b/A3, **not**
   to A2, which uses no flow.
3. 4 abnormal test clips lost their window at stride 8 (`--strict` unset).
4. Lesson **C17** still open — `--init-weights` absent from `config.yaml`; warm
   start had to be inferred from `kip_rec` openings (A2b from A1's stage 1, A4
   from A3's — both bit-identical).

### Next on DADA (`RESULTS_DADA.md` §10)

**A. Reporting fix** (zero GPU) — make `auc_macro` + the 0.9086 clip-oracle row
the DADA headline everywhere. **B. Frame-level linear probe** on the cached
frozen-CLIP DADA features against the real frame labels — the decisive
experiment, ~30 min, separates "frozen CLIP cannot represent an accident frame"
from "weak supervision cannot find it". **C.** Re-extract at stride 2–4 with
`score_head_kernel=3` — only if B is positive; **fires lesson C2** and
invalidates every number above. **D.** Root-cause defect 1.

Two lesson candidates are queued in `lessons-learned/pending.md` (2026-09-06).

## 2026-09-04 — DADA-2000 folder-name collision fixed (still no numbers run)

Running `DADA_SETUP.md` §5 against the real archive crashed:
`ValueError: video id 'type10_vid001' appears under both '0_Non_Ego_Fault'
and '0_Normal_Driving' -- folder names are not globally unique`. Root cause:
`type<N>_vid<N>` folder names repeat across DADA-2000's three
fault-attribution directories on the **real** archive, not just
hypothetically — the accident-type taxonomy is shared across fault dirs, and
`0_Normal_Driving` reuses the same bucketing convention. `dada.py`'s
`_assert_no_id_collisions` was correctly catching this, but the fix could not
be "silence the assert": `extract_clip_features.py:194` and
`raft_extract.py:291` build `{video_id_from_path(p): p for p in
list_frame_folders(...)}` — a plain dict, **no collision check** — over the
same `--frames-dir`. Bypassing the assert without fixing the id scheme would
have let those two tools silently extract one colliding clip's pixels under
the other's id.

**Fix (new lesson `26`/`C26`):** `video_id` is now
`{fault_dirname}__{folder_name}` everywhere in `dada.py` — globally unique by
construction, not by luck. `dada.py --flat-frames-dir` materializes a symlink
farm named by that id, so `extract_clip_features.py`/`raft_extract.py
--frames-dir` need **zero** code changes — point them at the flat dir instead
of `/content/dada` directly (`DADA_SETUP.md` §5/§7/§8 updated). Scoped
entirely to `core/data/dada.py` + its tests/docs; `video_io.py` and the two
shared extraction tools are untouched, so TAD/DoTA are unaffected.

**497 tests green** (was 494; +3 for the flat-dir symlink behavior), ruff /
mypy / bandit / pyright clean on the changed files. `test_dada.py`'s former
`test_id_collision_across_fault_dirs_raises` is now
`test_colliding_bare_folder_names_are_disambiguated` — the archive's real
shape means this path is exercised, not just an edge case to reject.

Still nothing run against the real archive past this point — next is
`DADA_SETUP.md` §4's coverage/frame-order gates, then §5 with
`--flat-frames-dir`, then §6's sanity check.

## 2026-09-03 — DADA-2000 dataset support shipped (no runs yet)

New: `core/data/dada.py` (`core/tests/test_dada.py`, 27 tests — **494 total**,
was 467; ruff / mypy / bandit / pyright / pycycle clean). Docs:
`core/docs/DATA_SETUP.md`, `core/docs/v3/setup/DATA_V3_SETUP.md`.

DADA-2000 ships frame folders under three fault-attribution directories
(`0_Non_Ego_Fault/`, `0_Normal_Driving/`, `1_Ego_Fault/`) plus
`Cleaned_Metadata.csv` — one row per **accident** clip only (975 rows, 0
`0_Normal_Driving` rows, confirmed on the user's real CSV) carrying a raw-frame
accident window and a `(type, video)` index that joins to the on-disk folder
`type<N>_vid<N>` by **parsed int**, not a reconstructed padded string (the
archive's padding width is unverified). Unlike TAD/DoTA, DADA-2000 has
frame-level ground truth for nearly every abnormal clip, so `dada.py` builds a
**real train/test split** (MSAD-style, seeded + stratified by
`(is_abnormal, Fault_Label)`, default 0.2/0.2 ratios — a project choice, no
official WS-VAD split exists), not TAD's directory-only weak train. Three
record kinds: annotated abnormal (window known, either split), weak abnormal
(frame folder with no CSV row — forced into train, same treatment as a
windowless clip everywhere else in this project), normal (`0_Normal_Driving`,
no CSV rows at all, directory is the label like TAD). `dada.py` always builds
both splits — no eval-only mode.

`DADA_DATASET = "DADA2000"` and `dataset_abbr("DADA2000") -> "dada" ->
_DOTA_CLS_DEFS` already existed in the tree (spec §7.4/§7.5 anticipated this),
so `core/train.py`/`core/evaluate.py` needed **zero** code changes — the
dataset-name-keyed generic paths already worked. Added to `core/constants.py`:
`DADA_METADATA_FILENAME`, `DADA_NON_EGO_FAULT_DIRNAME`, `DADA_NORMAL_DIRNAME`,
`DADA_EGO_FAULT_DIRNAME`, `DADA_CLASS_NAME = "CarAccident"` (shared with DoTA).

**No published DADA-2000 number is pinned in this project** (no
`DADA_ZERO_SHOT_AUC` analog to `TAD_ZERO_SHOT_AUC`) — `DATA_SETUP.md` §6 is a
sanity check against the released checkpoint's own score, not a reproduction
gate. `DATA_V3_SETUP.md` mirrors `TAD_V3_SETUP.md`'s arm matrix (A0/A1/A2/A2b/
A3/A4) as a **second replication** of the plain-TSM-smoother finding
(`RESULTS_V3_GATE_ATTRIBUTION.md`), with DADA-2000 also usable as a **second
zero-shot transfer target** (alongside DoTA) for arms trained on TAD or MSAD.
Per lesson 14: do not tune anything on a DADA-2000 number, and do not treat any
in-domain DADA-2000 result as comparable to a paper number, because none
exists for this protocol.

**Nothing has been run against the real archive yet** — the user has only
`Cleaned_Metadata.csv` on disk (975 rows, `Counter({'0_Non_Ego_Fault': 567,
'1_Ego_Fault': 408})`, types 1-61), no frame folders, no zip. All 27 tests are
against synthetic fixtures. Next: unzip, run `DATA_SETUP.md` §4's ingest gates
(coverage + frame-order — **unverified for this archive**, unlike TAD where
the pitfall was already measured), then §6's sanity check before any training.

## 2026-09-02 — TAD is trainable: P1 + P2 shipped, docs written

Runbooks: **`core/docs/TAD_SETUP.md`** (data) and
**`core/docs/v3/setup/TAD_V3_SETUP.md`** (arms, training, eval).
**467 tests green** (was 434); ruff / mypy / bandit / pycycle / pyright clean.

### Why TAD at all

**Replication, not a headline.** The question is whether `A2 - A0 > 0` and
`A1 - A2 <= 0` hold on a *second* training corpus. Pre-registered 2026-09-02,
before any TAD arm is trained. Do not tune on a TAD number (lesson 14).

### The two blockers, now closed

- **P1** — `core/data/tad.py` was eval-only (`labels_train.json = {}`), so no TAD
  arm could train. `--with-train-split` builds the weakly-supervised split from
  the frame folders the annotation does not name. **The default is unchanged**:
  every pre-2026-09-02 TAD artifact is still reproducible bit-for-bit.
- **P2** — `core/flow/raft_extract.py` read videos only; TAD ships frames, so
  `e_O` was unbuildable and every KIP-on arm was dead. `--frames-dir` /
  `--frames-subdir` / `--ids-file` now mirror the CLIP extractor.
  `--videos-dir` is untouched, and the shared `_extract_sources` loop keeps the
  two from drifting.

### Two measurements worth keeping

- **The frames path was NOT bit-identical to the video path** — 2.67e-5 on
  identical pixels. Cause: `read_images` returns a non-contiguous permuted view
  and torch dispatches a different conv kernel on one. Fixed with
  `np.ascontiguousarray` in `read_sampled_frames_from_dir`; parity is now
  asserted. New lesson **C25**.
- **TAD's filename prefix is not its label.** The 60 abnormal test videos carry
  **seven** prefixes (`01_Accident` 23, `05_else` 14, `06_PedestrianOnRoad` 10,
  `07_RoadSpills` 4, `02_IllegalTurn`/`03_IllegalOccupation`/`04_Retrograde` 3
  each) and LaGoVAD labels all of them `Car Accident`. MSAD's
  `--infer-abnormal-from-name` precedent would have mislabelled **37 of 60**.
  The split *directory* is the label; `_label_from_directory` raises rather than
  defaulting. Candidate in `pending.md` (gate 1 fails — caught before it bit).

### Also measured on the real annotation

100 entries, 60/40, **10 multi-span videos, max 3 spans** (filled independently,
lesson C18). Narrowest normalized span **0.0260** (`03_IllegalOccupation_002`),
so a clip needs **>= 160 raw frames** for it to survive stride 8. The
preprocessor raises if one does not.

### Next on TAD

Data + **Gate T0** (`best.ckpt` on TAD, reference `TAD_ZERO_SHOT_AUC = 89.56`)
before any training. **An in-domain TAD number is not comparable to 89.56** —
that trap is §0.1 of `TAD_SETUP.md`. Then A2 and A0 across 3 seeds, read
`A2 - A0` on DoTA, and only then stage 1 + A1.

---

## 2026-09-01 — attribution is done. The smoother did it.

Full analysis: **`core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md`**.
Runbook: `core/docs/v3/setup/MSAD_DOTA_V3_SETUP.md`; notebook `collab/MSAD/v3/train.py`.
Artifacts: `outputs/v3/**` (16 evals) + archived A0/V1 arms in `outputs/*_ncc*`.

### The finding

**A2, a fixed 50 % channel shift with no flow, no PMG head, no KIP losses and no
stage-1 warm-up, is statistically indistinguishable from full v1 KIP on DoTA** —
Δ(A2 − V1) = +0.0109, t95 **[−0.0588, +0.0805]**, and 6/6 metrics include zero.
A2 − A0 = **+0.1025 ± 0.0350**, t95 [+0.0154, +0.1896] (n=3, seeds 2024/25/26).
The archived V1 arm reproduces at **+0.0916 ± 0.0087** — our pipeline was
validated against a known number before making a new claim.

**The pre-registered prediction (setup §2.2, written 2026-08-30 *before* the
eval) is confirmed. The motion-induction claim is not supported by any ablation
in this campaign.**

Worse for the v3 design: **the rank gate is the worst KIP arm.** A1 − A2 =
**−0.0683**, CI [−0.0790, −0.0580], P>0 = 0.000; 60 % of clips worse; A1 does not
even beat LaGoVAD's released trunk (−0.0272). The gate spans all 128 channels on
every clip (§6.8 verified) — it works exactly as designed, and that is what makes
the result damning rather than inconclusive. **The more the shift count varies
with motion evidence, the worse the score.**

The rest of the ladder, all seed 2024: **A3 − A2 = +0.0096, CI includes zero**
(the trained 321-param gate buys nothing over hard-coding its output);
**A2b − A2 = −0.0105, CI [−0.0207, −0.0011]** (adding real flow + all three
auxiliary losses *costs* AUC); A4 − A3 = +0.0029 (the gradient fix is real but
tiny). **No component of KIP has been attributed a positive contribution.**

MSAD stays a bounded null for the smoother too: A2 − A0 = +0.0036 ± 0.0076,
t95 [−0.0152, +0.0225]. D4 refuted **10/10 arm-seeds** — `other` > `ego`
everywhere, and A2's ego/other signature (+0.0725/+0.1425) matches V1's
(+0.0712/+0.1191), so that signature was never diagnostic of a motion mechanism.

### Two reproductions that license the analysis

- **A0 under v3 vs archived A0**, seed 2024: Δ +0.0002 MSAD / −0.0005 DoTA. This
  is what licenses borrowing archived A0 seeds 2025/2026 (v3 retrained A0 only at
  2024). Config parity verified — they differ only in `kip.*` keys inert under
  `kip.enabled=false`.
- **A3 (`mlp_frozen` retrained under v3) vs archived V1 KIP-on**, seed 2024:
  Δ −0.0004 DoTA. Setup §0's bit-identity claim survives a retrain.

### The honest restatement on the table

*A temporal shift module between LaGoVAD's temporal encoder and its fusion buys
≈ +0.09 zero-shot DoTA micro AUC, RGB-only, at 0 params on the score path, with
no in-domain penalty within ±0.02.* Smaller, cleaner, defensible — and it makes
`core/flow/`, the PMG head and `L_KIP_rec`/`L_KIP_align`/`L_kin` dead weight.
**Do not tune A1 to beat A2** (lesson 14): the +0.09 is now known not to be a
motion effect.

### Live queue

1. **Two more seeds for A1** (`rank`, 2025+2026; needs §5.1a stage 1 each). The
   only number that promotes the A1 ≪ A2 finding from n=1 to a result, and the
   last way the motion story could survive.
2. **Two more seeds for A3**, so the v1 bridge has a seed-level interval.
3. **Fix the two metric defects**: atomic writes in `evaluate --save-scores`
   (**C11b**, new) and float64 in `normalize_scores` (**C22**, still open), then
   re-`rescore --write` every run dir.
4. **The training run manifest** (C17) — still the only thing that can tell a
   `rank` arm from a `constant` arm after the fact.
5. **A2 at other `const_shift_ratio`** {0.125, 0.25, 0.75} — characterise the
   real mechanism instead of defending the abandoned one. Pre-register first.
6. Only then: decide whether to delete the flow/PMG infrastructure.

### Defects found

- **A 0-byte `.npz`** in A1's DoTA eval (`qzMjfBx1KI0_003085`) — `evaluate
  --save-scores` is not atomic. One clip dropped from *all* arms to keep the
  1,396-clip set identical. New lesson **C11b**.
- **`results.json` is float32-normalised**; reproduced exactly, float64 recompute
  differs by ΔAUC 0.0002 / ΔAP 0.0016. C22 confirmed on a second dataset; the
  method ("reproduce the tool's number first, then vary one factor") is **C22b**.
- `metrics.jsonl` for A3 stage 2 held 665 rows for 500 steps (restart append);
  **deduped 2026-09-01**, `.jsonl.bak` kept.
- **16 `eval_manifest.json` retro-filled** (setup §6.11), each asserting its §6.8
  gate span at write time. All 16 pass.

---

## 2026-08-30 — v3 gate rebuild shipped, and H4′ is confirmed

Plan: `.project/plans/katvad-v3-kip-gate-rebuild.md` (Phases 0–5, measured
appendices A–E). Branch `v3`. **73 source files, 434 tests green**, ruff / mypy /
pyright / bandit / pycycle clean.

### The finding that reorders the queue

**H4′ is confirmed: v1's KIP was a fixed ~50 % temporal smoother.** It cost zero
training to establish, because the gate MLP never trains — so random init *is*
the deployed function — and its input is min-max normalized, so `[0,1]` is the
*entire* reachable input domain. Sweeping that whole domain moves `s_t` by
**0–4 channels out of 128** across 8 seeds (seed 0: exactly 0; the three MSAD arm
seeds 2024/2025/2026: 1, 3, 3). `mlp_frozen` and an explicit `constant` gate at
`r = 0.5` agree to within half a channel on the mean.

So the +0.0915 ± 0.0088 DoTA gain was produced by a **constant temporal
smoother**, not by motion gating. The effect is real and replicated; the
*mechanism* attributed to it never existed.

**Consequences, in priority order:**

1. **The plain-TSM control (ablation 4) is now the top experiment**, not row 4.
   It is one config flag — `kip.gate_type=constant`, `kip.const_shift_ratio≈0.5`,
   `kip.disable_pmg=true` for the no-flow arm. **Pre-registered prediction:
   `r = 0.5` reproduces most of the +0.09.** If it does, the contribution is
   temporal smoothing and the thesis needs restating. This is the cheapest
   experiment in the program and it can falsify the central claim.
2. Never write "motion-gated" of `gate_type="mlp_frozen"` again. Lesson **24
   [CRITICAL]** now covers the general class.
3. `RESULTS_*.md` numbers are all `gate_type="mlp_frozen"` numbers. They were
   true for what ran; do **not** retro-edit them — add the qualifier at the point
   of next citation.

### What shipped

- `core/kip/ecmr.py` — Phases 3b/3c, **0 parameters**: causal EMA prototype,
  ego-compensated residual, within-clip rank map, shared floor step.
- Four selectable `kip.gate_type`s: `rank` (v3 default, span 128 on every clip),
  `mlp_frozen` (v1, **bit-identical**, pinned by a golden fixture captured before
  the refactor), `mlp_ste`, `constant`.
- **A config with a `kip:` section and no `gate_type` now raises.** All 22
  archived `outputs/*/*/config.yaml` predate the key; a silent default either way
  is the C14 defect. A test asserts every one of them raises.
- KIP's `mhead` (3e) and projections (3f) are off the inference graph:
  **311,808 params at inference, 0 on the score path**. `load_kip_state_dict`
  drops exactly those 8 tensors by allowlist — never `strict=False` — and refuses
  a gate-type mismatch in both directions.
- `--dump-kip-diag` (default **on** in `evaluate.py` with `--save-scores` and in
  `inference.py`; **never** in training) writes `kip_s`, `kip_gate_ratio`,
  `kip_m`, `kip_mu_norm`, `kip_eo_norm` into each score `.npz`.

### Two spec/doc errors found and corrected

- **The spec's STE formula is a no-op.** `s = u + (⌊u⌋−u).detach()` gives `s` a
  gradient, but `s` is read only inside `channel < s`, and a comparison passes
  none. Implemented on the selection *weights* instead — forward bit-identical
  (max diff 0.0), gradient live to both the MLP and `ê_O`.
- **The architecture doc's fixed-camera claim was wrong.** "On a fixed camera
  `μ_t ≈ 0` and the residual reduces to raw magnitude" — it reduces to **exactly
  zero** (`test_constant_flow_gives_zero_residual`). Corrected in place.

### Still true, still unimplemented

Gradient starvation of PMG is **pre-existing, not introduced by the rank gate**:
the rank gate removes 321 already-dead parameters and alters no gradient edge.
But it makes stage-1 convergence load-bearing, and spec v3 §8's
`assert stage1_final(L_KIP_rec) < τ_rec` is **not built** — it belongs to the
deferred training-pipeline plan, along with the run manifest (lesson 17), which
is also the only thing that can distinguish a `rank` arm from a `constant` arm
after the fact (their key layouts are identical).

**Alert-CLIP is deferred, not adopted** — no public checkpoint exists. Every
number in this repo is a **stock CLIP ViT-B/16** number.

**Blocked:** reading RefineVAD's MoTAR implementation for a straight-through
relaxation (no PDF extractor in the env, `pip` has no network). Unblock with
`brew install poppler` or `pip install pypdf`.

## Current state

**The memory bank was re-initialized from scratch at commit `b9978ff`
("feat: Include training/testing for MSAD full") on the user's instruction.**

Why this commit: the user trained and evaluated on MSAD with exactly this code
and **got results that match the paper**. It is therefore the trusted starting
point — treat this tree as the known-good baseline and change it deliberately.

Everything in this memory bank was derived by reading *this* tree on 2026-07-31.
Nothing was carried over from the previous memory bank.

## What happened to the previous memory bank

A sibling directory `.project/memory-bank/` still holds the old files —
including a `lessons-learned/` catalog and a large `activeContext.md` describing
later experiments (a "p17" feature-cache re-ablation, gate re-measurements).

**The user explicitly declined restoring it** and asked for a fresh init from
`b9978ff` instead. That old content is *not* authoritative for this tree and was
deliberately not merged in. It still exists on disk if a specific number is ever
needed — but treat anything from it as unverified against this commit.

## State of the tree, as verified

> Counts below are the 2026-07-31 snapshot and are **superseded** — as of
> 2026-08-25 the tree is **68 Python files / 7,723 source LOC / 322 tests
> green**, and `outputs/` is full (gitignored, 21,281 `.npz` score files). See
> [[techContext]] and [[progress]] for the live numbers; never quote this
> section.

- **Code:** Phases 0–5 complete. 59 Python files, ~9.0k LOC, **221 tests green**
  (`python -m pytest core/tests -q`, run 2026-07-31, zero failures).
- **Artifacts:** `outputs/` is empty. No `cache/` directory. `data/MSAD/` holds
  **annotations only** — no videos, no CLIP features, no flow. Every run
  artifact lives on the user's Colab/Drive side, not here.
- **Docs present:** `COLAB.md`, `DATA_LAYOUT.md`, `TRAINING.md`, proposal, spec.
  `TAD_EVAL.md` was present earlier in the session and is no longer in the tree —
  consistent with pruning the repo down to the MSAD story.
- **Plans present:** `.project/plans/kat-vad-implementation.md` only.

## 2026-08-01 — MSAD-full results landed and analysed

The user ran the full Colab sequence (`collab/MSAD/train.py`) and dropped the
artifacts into `outputs/MSAD/**`. Full analysis: **`core/docs/RESULTS_MSAD.md`**.

- Baseline reproduction **PASSES**: KIP-off AUC 0.9052 vs LaGoVAD paper 0.9041.
- KIP-on 0.9064 (+0.0012 AUC, +0.0086 AP) — wins on all 6 aggregate metrics but
  **every bootstrap CI includes zero**. Gate (c) is a point-estimate pass only.
- `Traffic_accident` (the target class) moves **−0.002 AUC**. Per-video ΔAUC is a
  coin flip (42 % win / 39 % loss) with ±0.4 tails.
- Strongest real signal: multi-class frame accuracy on anomalous frames
  0.475 → 0.512, concentrated on motion-textured classes (water, fire, fighting,
  traffic). Worth chasing (RESULTS §6.5).
- KIP inference cost +1.82 % params (345k), RGB-only — architectural claim holds.
- Two measurement defects found: eval reads `checkpoint_last` under train
  `mil` ≈ 0.001 (no val split, no model selection), and `L_KIP_align` sits at
  3.66 vs chance 4.27 (A11 near-duplicate-positive failure; both mitigation
  knobs off).

Next: 3-seed on/off repeat + validation split **before** any KIP ablation.

## 2026-08-01 — DoTA zero-shot eval wired up (not yet run)

The user has DoTA's val annotations (`data/DoTA/metadata_val.json`,
`val_split.txt`) locally and `DoTA_full.zip` on Colab. Protocol + decision
rules: **`core/docs/DOTA_EVAL.md`**.

- New code: `core/data/dota.py` (preprocessor), frame-folder readers in
  `core/data/video_io.py`, `--frames-dir/--frames-subdir/--ids-file` +
  streaming `encode_frame_dir` in `core/tools/extract_clip_features.py`,
  shared `TEST_IDS_FILENAME`/`write_test_ids`. **240 tests pass** (was 221);
  ruff/mypy/pyright/bandit clean.
- Found and fixed: `core/data/tad.py` did not import at `b9978ff` — it
  referenced three `video_io` helpers that do not exist in this tree. Adding
  them for DoTA fixed TAD as a side effect. `--ids-file` was likewise
  documented in `tad.py` but absent from the CLI.
- Protocol facts recovered from the baseline (lesson C8): labels are
  `round(normalized_span x feature_length)` half-open; features are
  `interval=8`; `full_length_eval.py` pools **raw** scores while
  `offline_dota_eval.py` min-max-normalizes per clip. Our derived spans equal
  the baseline's shipped `anomaly_span` on all 1,402 clips.
- Decided: run at **stride 8** (matches both LaGoVAD's DoTA features and the
  stride our MSAD checkpoints were trained at); extract once at stride 1 and
  slice `[::8]`, since `range(0,N,8)` makes that exact.
- **2026-08-02, first real unzip:** 1,403 folders on disk, 1,398 usable —
  **5 clips have an empty `images/`** (FUSE write failed after `mkdir`; all 5 are
  present in the archive). Folder counts lied about coverage, and the empty case
  was fatal with no escape hatch. Fixed: `resolve_frame_counts` classifies
  unreadable clips as absent vs. empty, `--allow-missing-frames` drops both and
  logs `Coverage N/M (P%)`, out-of-split folders are counted and ignored
  (lesson C10). 246 tests.
- **2026-08-02, extraction resume hardened.** The stride-1 DoTA pass is long
  enough to outlive a Colab session, and skip-if-exists was not a safe resume:
  `np.save` straight onto the target leaves a truncated `.npy` that `exists()`
  calls done. New `core/tools/feature_cache.py` (atomic `.part`→rename writes,
  header verification, `Resume: done/total` + per-item ETA at INFO) is used by
  **both** `extract_clip_features` and `raft_extract` (lesson C11). 258 tests.
- Three arms: released `best.ckpt`, `stage2_kip_off`, `stage2_kip_on`. The only
  scientifically meaningful number is Δ(on−off) with its bootstrap CI; the
  62.60 paper number is comparable to the `best.ckpt` arm **only**.

## 2026-08-08 — DoTA ran, the protocol was wrong, and the transform is changing

Full analysis: **`core/docs/RESULTS_DOTA.md`**.

The three DoTA arms came back at chance (0.505–0.524). **The models were fine;
the metric was broken.** `core/evaluate.py` pooled raw scores across clips,
which is correct on MSAD (50.4 % normal videos) and meaningless on DoTA
(0.21 % normal — every clip is abnormal, so the task is purely within-clip
localization and the between-clip scale is noise). Under per-clip min-max,
LaGoVAD's released `best.ckpt` scores **0.6142 vs its published 0.6260** — a
reproduction. Raw pooling had it at 0.5055.

Having `gate_a` in the run is what made this findable: a released checkpoint
cannot be at chance on its own benchmark.

- **Lesson C8 was wrong and caused it.** It recorded 62.60 as raw-pooled from a
  code read that was never checked against a reproduction. Corrected; **C12**
  (all-abnormal pooling) and **C13** (checkpoint↔transform binding) added.
- **New code:** `core/metrics.py` (pooling + macro AUC), `core/tools/rescore.py`
  (recompute metrics from saved `.npz`, no re-inference), `--score-norm` on
  `core.evaluate` (default `auto`, resolves from the label distribution, not the
  dataset name), `--no-center-crop` on `extract_clip_features`. **284 tests**
  (was 258); ruff/mypy/pyright clean.
- `outputs/DoTA/*/results.json` have been rewritten with the corrected metrics.

~~**KIP-on is worse than KIP-off**: Δ = −0.0329 per-clip AUC, 95 % CI
[−0.0513, −0.0138]~~ — **SUPERSEDED 2026-08-12, sign flipped to +0.0911 after
the crop was removed. See the 2026-08-12 entry below and `RESULTS_NCC.md`.**
Recording that signed delta while a known precondition defect was open is now
lesson **C14**. The rest of the 2026-08-08 entry (pooling, C12, the `gate_a`
reproduction) stands.

**Decision: the pipeline moves to `no_center_crop`.** Not only for baseline
parity — `preprocess_for_raft` is already full-frame, so KIP was being trained
to regress motion evidence center-cropped out of its own input, and 349 of
1,402 DoTA clips are `ego/other: lateral`. Rebuild sequence, with every artifact
going to new `*_ncc` paths and the flow cache reused unchanged:
**`core/docs/COLAB.md` § "no_center_crop rebuild"**.

## 2026-08-12 — the `no_center_crop` rebuild ran; KIP's DoTA sign flipped

Full analysis: **`core/docs/RESULTS_NCC.md`**. Plan for what comes next:
**`.project/plans/msad-ncc-seeds-and-selection.md`**.

The user ran the full rebuild (`collab/{MSAD_ncc,DoTA_ncc}`) and landed
`outputs/MSAD_ncc/**` + `outputs/DoTA_ncc/**`. Four training runs now exist:
2 transforms × KIP {on, off}, seed 2024, configs otherwise byte-identical.

| | MSAD-full (raw) | DoTA (per-clip min-max) |
|---|---|---|
| KIP-off | 0.8922 / AP 0.6587 | 0.5607 (macro 0.5638) |
| KIP-on | 0.8868 / AP 0.6574 | **0.6519 (macro 0.6746)** |
| Δ(on − off) | −0.0054, CI [−0.0155, +0.0033] — null | **+0.0911, CI [+0.0799, +0.1019]** |

- **The crop removal worked through KIP, not around it.** Crop→ncc moved KIP-on
  by **+0.1514** macro AUC (CI [+0.1354, +0.1683]), KIP-off by +0.0077 (CI
  includes 0), and `gate_a` by −0.0170 (negative). C13's mechanism is measured.
- KIP-on now also beats LaGoVAD's released `best.ckpt` (+0.0508 micro,
  CI [+0.0355, +0.0659]).
- **Cost:** MSAD KIP-on lost −0.0196 AUC / −0.0761 AP (both CIs exclude zero).
  The **MSAD reproduction gate no longer passes** — 0.8922 vs published 0.9041.
- **The transform-parity argument is refuted.** `gate_a` got *worse* under ncc on
  both benchmarks. Pending item P1 resolved as "not supported"; the
  field-of-view leg (C13) is what carried the result.
- **Both mechanism checks failed.** D4: gain is `other` (+0.1296) > `ego`
  (+0.0969) — do not claim ego-kinematics. D5: MSAD multi-class frame accuracy
  went +3.6 pp (crop) → −0.6 pp (ncc); the one live MSAD thread did **not**
  replicate, so the DoTA gain and the multi-class gain are not one mechanism.
- Saturation is extreme (89 % / 87 % of DoTA frames > 0.99) but float32 tie
  rates are ~0.2 frames/clip, so macro AUC is sound. Not a numerical artifact.
- New lesson **C14**; P1 resolved; `RESULTS_DOTA.md` carries a superseded banner.

**Three hypotheses remain open** and the result should not be claimed until they
are separated: H1 KIP genuinely helps under domain shift; H2 seed luck (n=1);
H3 under-convergence transfers (KIP-on's final train `mil` is 0.010 vs KIP-off's
0.0016 — worse in-domain / better out-of-domain is a regularizer's signature).
Plus an unaddressed confound: KIP-on warm-starts from stage 1, KIP-off does not.

> **Updated 2026-08-16:** H2 is **rejected** and H1 holds — the Δ replicates
> across three seeds (see the 2026-08-16 entry). H3 and the warm-start confound
> are still open. The MSAD "reproduction fails" claim above is **superseded**:
> the gate was measured against the wrong reference and now passes
> (`RESULTS_PHASE_A.md` §4).

## 2026-08-16 — Phase A ran: the DoTA gain replicates. H2 rejected.

Full analysis: **`core/docs/RESULTS_PHASE_A.md`**. Runbook for what comes next:
**`core/docs/COLAB.md` § "Arm 4 + trajectory probe"**.

The user ran seeds 2025 and 2026 (`collab/MSAD_ncc_s2025_2026/train.py`,
`collab/DoTA_ncc/evaluate_s2025_s2026.py`) into `outputs/{MSAD,DoTA}_ncc_s{2025,2026}`.
**Config parity verified:** all nine runs' `config.yaml` are byte-identical
across seeds once Drive paths and `train.seed` are normalised. Only the seed moved.

| seed | DoTA off | DoTA on | Δ micro | 95 % CI | MSAD Δ |
|---|---|---|---|---|---|
| 2024 | 0.5609 | 0.6520 | +0.0911 | [+0.0797, +0.1021] | −0.0054 (ns) |
| 2025 | 0.5587 | 0.6416 | +0.0829 | [+0.0725, +0.0945] | +0.0005 (ns) |
| 2026 | 0.5284 | 0.6289 | +0.1004 | [+0.0886, +0.1124] | −0.0008 (ns) |

- **H1 holds, H2 rejected.** Mean Δ **+0.0915 ± 0.0088**; min +0.0829 is ~2.8×
  plan §2.4's +0.03 bar. Nine CIs (micro/AP/macro × 3 seeds), nine exclusions of
  zero, all three micro CIs mutually overlapping.
- **MSAD is null in every seed** (micro AUC and AP CIs all include zero, sign not
  even stable). That asymmetry **is** the finding — the motion pathway pays where
  motion is the anomaly and is inert where it is not.
- Both DoTA arms drift down together across seeds; the *level* is seed-sensitive,
  the *gap* is not. Report the paired Δ, never either arm alone.
- **D4 refuted, three for three:** `other` (+0.130/+0.140/+0.141) > `ego`
  (+0.097/+0.073/+0.087) in all seeds. **Do not claim ego-kinematics.** At n=1
  this was a failed check; at n=3 it is a settled negative.
- Weak signal, stated with caveat: KIP-on's MSAD **macro** AUC is far more
  seed-stable (spread 0.006 vs KIP-off's 0.040). Only one of three Δ CIs excludes
  zero, and barely. An observation at n=3, not a result.

### The MSAD reproduction gate was the wrong comparison — now redefined

`RESULTS_NCC.md` recorded the gate as **failing** (0.8922 vs published 0.9041).
It was comparing against a number the released checkpoint cannot reach either:
`best.ckpt` through our eval scores **0.8991** (crop) / **0.8949** (ncc).

Paired bootstrap of every arm against `best.ckpt` on the identical protocol:
**all eight CIs include zero**, on both transforms, all three seeds; AP is
consistently *higher* for our arms. Our port is statistically indistinguishable
from the checkpoint the authors shipped.

**Gate redefined** (user's call, 2026-08-16, and the measurement backs it): the
gate is *"our KIP-off matches the released `best.ckpt` under one identical
protocol."* By that definition it **passes** everywhere. 0.9041 is not reachable
with the published artifacts — stop chasing it; quote it only as the paper's
printed number. New lesson **8b**.

### Defect found while auditing

`collab/DoTA_ncc/evaluate_s2025_s2026.py:113-153` — trailing cells reference
`$EXTRA`, unset in a fresh `%%bash` cell, which would evaluate a KIP-off
checkpoint with `kip.enabled=true`. **Did not corrupt the results**:
`core/inference.py:64` loads at `strict=True` and raises (lesson 5). The numbers
came from the correct loop at lines 100–110. Delete the dead cells.

## 2026-08-16 — every checkpoint on Drive stopped loading (environment drift)

First attempt at A4.0 died on seed 2024:

```
TypeError: _reconstruct: First argument must be a sub-type of ndarray
```

**Cause:** `save_checkpoint` (`core/train.py:408-425`) stores `rng`, and
`_rng_payload` (`core/train.py:389`) puts `np.random.get_state()` — a tuple
wrapping a 624-element uint32 numpy array — inside it. It is the payload's only
numpy object. A pickle is a single stream, so once the Colab VM's numpy major
version drifted from the one that trained, `torch.load(weights_only=False)`
aborts **before reaching any tensor**. The weights are intact; the artifact is
not portable.

**Blast radius is wider than A4.0.** Both remaining consumers use
`weights_only=False`: `warm_start_model` (`core/train.py:94`) and
`load_model_for_scoring` (`core/inference.py:62`). So A5's ~12
`checkpoint_step_*.pt` evals and any `--resume` fail identically.
`load_baseline_checkpoint` is unaffected (`core/models/ckpt_compat.py:137` uses
`weights_only=True`), which is why `full_gate_a` kept working.

**Resolution — recover, do not downgrade.** `COLAB.md` §A4.0/§A5.0 now load
through a `pickle.Unpickler` whose `find_class` refuses
`multiarray._reconstruct` (the RNG blob becomes a discardable placeholder) and
re-save slim `{"model", ...}` files. Lossless: tensors travel `persistent_load`
and never reach `find_class`. Both consumers read only `payload["model"]`.
Pinning `numpy<2` was rejected — Colab's torch wheel is built on the numpy 2 ABI.

**Backlog (not now):** `_rng_payload` should serialize the numpy state as raw
bytes, or omit it. Not touching it mid-experiment — the training code stays
fixed while arms are being compared. Recorded as lesson **15 / C15**.

## 2026-08-19 — Arm 4 + trajectory probe ran: both remaining confounds are closed

Full analysis: **`core/docs/RESULTS_ARM4_PROBE.md`**.

The user ran `COLAB.md` § "Arm 4 + trajectory probe": three `stage2_kip_off_warm`
runs (KIP-off warm-started from the same stage-1 checkpoint KIP-on used) with
MSAD + DoTA evals per seed, and 10 DoTA evals of seed 2024's saved
`checkpoint_step_{100..500}.pt` for both arms.

**A4 — the warm start is not the mechanism.** Δ(on − off_warm) on DoTA:

| seed | off_cold | off_warm | KIP-on | Δ(on − off_warm) | 95 % CI |
|---|---|---|---|---|---|
| 2024 | 0.5609 | 0.5482 | 0.6520 | **+0.1039** | [+0.0961, +0.1115] |
| 2025 | 0.5587 | 0.5313 | 0.6416 | **+0.1104** | [+0.1022, +0.1193] |
| 2026 | 0.5284 | 0.5466 | 0.6289 | **+0.0822** | [+0.0731, +0.0920] |

Mean **+0.0988 ± 0.0148**, nine CIs (micro/AP/macro × 3 seeds) all excluding
zero. It did **not** shrink against Phase A's cold Δ (+0.0915) — it is
marginally larger, and moves in both directions per seed. Decision-table row 1:
**the A/B is now "KIP vs no KIP" from a shared trunk.**

- **Stage-1 trunk pretraining alone transfers nothing stable.**
  Δ(off_warm − off_cold) on DoTA = −0.0128 / −0.0273 / **+0.0181**; every CI
  excludes zero **and the sign flips across seeds**. ±0.02 of seed-dependent
  level shift vs KIP's +0.099. The gain lives in the module at inference, not in
  what `L_KIP_rec`/`L_KIP_align` did to the temporal encoder.
- **MSAD stays null under arm 4** (Δ(on − off_warm) = +0.0001 / −0.0010 /
  −0.0030, all CIs include zero; Δ(off_warm − off_cold) null too).
- **D4 refuted six for six** — `other` > `ego` in all three warm comparisons
  (+0.1469/+0.1518/+0.1323 vs +0.1134/+0.1070/+0.0577). Per-clip win ~4:1.
- Warm start verified from the loss, not assumed: step-1 train `mil` for
  `off_warm` lands next to `on` and away from cold `off` in all three seeds
  (seed 2024: 0.70148 vs 0.70146 vs 0.86818). `config.yaml` does not record
  `--init-weights`.

**A5 — H3 is rejected.** Seed 2024, DoTA AUC vs train `mil` (20-step avg):

| arm | step 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| off `mil` | 0.0410 | 0.0031 | 0.0016 | 0.0012 | 0.0013 |
| off AUC | 0.5589 | 0.5604 | 0.5604 | 0.5608 | 0.5607 |
| on `mil` | 0.0834 | 0.0207 | 0.0103 | 0.0074 | **0.0075** |
| on AUC | 0.6097 | 0.6412 | 0.6482 | 0.6514 | 0.6519 |

- KIP-off crosses KIP-on's endpoint `mil` (0.0075) at **step ≈ 161**, bracketed
  by step 100 and 200 → interpolated **0.560 vs 0.6519**. The whole Δ survives
  at matched convergence.
- KIP-off's AUC spans **0.0019** while its `mil` falls **33×** — convergence
  level does not index transfer for that arm at all.
- KIP-on *gains* +0.042 as it fits harder — the opposite of H3's prediction.
- Probe step-500 reproduces the Phase A arms exactly (0.5607 / 0.6519). The
  `off` runs producing results at all proves `EXTRA` was set (lesson 5's
  `strict=True` would have raised) — the `$EXTRA` defect did not recur.
- Unambiguous → not extended to seeds 2025/2026.

**Net:** Δ ≈ +0.09 DoTA micro AUC is attributable to **the KIP module itself**,
+1.82 % inference params, RGB-only, no in-domain cost. Two rival explanations
eliminated; **none confirmed** — the mechanism is still unknown, and that is now
the only open scientific question.

**Sampling defect found (new lesson 16 / C16):** `checkpoint_every_steps=100`
over 500 steps left no checkpoint below step 100, where `mil` falls 0.63 →
0.041 — 94 % of the loss range sits inside the first, unprobed 20 % of steps.
The verdict holds (the endpoint is bracketed), but step-uniform checkpointing
undersamples the part of training a trajectory probe needs.

**Discipline:** A5 reads DoTA *test* scores along a training trajectory. It is a
diagnostic; the reported arm stays `checkpoint_last` and the probe curve must
never become a headline number.

## 2026-08-29 — SETTLED: a PreVAD KIP-on arm will never be run

**User's decision, stated directly: "I will not run the PreVAD KIP-on arm because
I cannot download the whole PreVAD dataset to extract the RAFT flow."** Treat
this as closed. Do not propose it, cost it, or reopen it.

**Why it is a hard constraint, not a budget one.** `L_KIP_rec` regresses `e_O`,
a cached RAFT embedding, and RAFT needs **pixels**. PreVAD ships CLIP features
only — `PreVAD/features/ViT-B-16-8p-features.zip` (35,279 `.npy`) and the ViT-L
sibling. There is no raw video anywhere in the release. Re-acquiring it is not a
workaround:

- `annotations/data_sources.csv` carries a platform URL for ~28,800 of 35,279
  rows (82 %); realistic yield after multi-year link rot, region blocks and dead
  stream archives is **50–70 %**.
- The 3,800 permanently unrecoverable rows are **China Expressway Camera** live
  captures — exactly the traffic/motion clips KIP is aimed at.
- A self-made download **cannot be paired with the released features**:
  different transcode, possibly different fps, `-Scene-NNN` ids imply an
  unpublished shot-detection pass. `core/data/dataset.py:112` raises on the
  length mismatch. Both features *and* flow would have to be re-extracted into a
  self-consistent cache, and a KIP-on arm on a 60 % subset is not comparable to
  a KIP-off arm on the full release — different training sets, uninterpretable Δ.

`require_flow = cfg.kip.enabled` makes this a **hard stop**: `kip.enabled=true`
with no flow cache raises `FileNotFoundError` (`core/data/dataset.py:107`).
**Never** work around it with `require_flow=False` — that zero-fills `e_O` and
trains the PMG head to predict zeros, a wrong run that still writes a checkpoint.

This was already documented in `PREVAD_SETUP.md` §2 (line 39) and **§7.4**, which
the 2026-08-28 analysis failed to read before listing a PreVAD KIP-on arm as a
next step. That listing is now struck from `RESULTS_PREVAD.md` §9 and the report
(new **§12.7**), and report limitation 2 is restated.

**What this does and does not block:**

- **Does NOT block the decisive run.** The §12.6 unblocking arm (PreVAD trunk +
  stage-1 KIP warm-up + `num_epochs=125`) trains KIP on **MSAD**, whose flow
  cache already exists at `cache/flow/v1/MSAD/MSAD-full`. PreVAD supplies only
  the trunk, which is already on disk and was trained KIP-off. **Limitation 14
  is closable with zero new flow extraction.** This is still the top priority.
- **Does block** report limitation 2 (one transfer benchmark) via this route.
  PreVAD's permanent role is **pretraining corpus + in-domain reproduction gate**
  (Gate P0), never a KIP A/B. Any second KIP benchmark must ship raw video —
  DoTA and MSAD do; TAD is the remaining candidate and is subject to the same
  pixels-for-RAFT requirement, unverified.

**Standing rule:** before proposing any KIP-on arm on a new dataset, check that
the dataset ships pixels. A feature-only release can host a KIP-**off** trunk, a
reproduction gate and an eval — never a KIP-on arm.

**Also 2026-08-29:** the user renamed
`core/docs/REPORT_KIP_MSAD_DOTA.md` → **`core/docs/REPORT_KIP_MSAD_DOTA_PREVAD.md`**.
References updated in `RESULTS_PREVAD.md`; `projectbrief.md`, `techContext.md`
and `CLAUDE.md` §14.1/§14.5 still carry the old name.

## 2026-08-28 — the PreVAD trunk campaign analysed: one strong positive, one void ablation

Full analysis: **`core/docs/RESULTS_PREVAD.md`** (new). Folded into
`REPORT_KIP_MSAD_DOTA_PREVAD.md` (renamed 2026-08-29) as **§12** (Trends/Conclusions/Limitations/Provenance
renumbered to §§13-16).

The user had run, without the memory bank recording it: a **PreVAD stage-2
KIP-off trunk** (32,673 clips, 13,360 steps, seed 2024), Gate P0, then an
**MSAD-full finetune × {KIP-on, KIP-off} × seeds {2024, 2025, 2026}** warm-started
from that trunk, evaluated on MSAD (in-domain) and DoTA (zero-shot).
Artifacts: `outputs/PreVAD/**`, `outputs/{MSAD,DoTA}_ncc_pv_s{2024,2025,2026}/**`.

**Three results.**

1. **Gate P0 passes, and the port reproduces on a third benchmark.** Released
   `best.ckpt` on PreVAD test = **0.9031** micro AUC / 0.6910 AP / 0.6721 macro
   (raw pooling — PreVAD is 49.9 % normal, `auto` resolved correctly with no
   human decision). Our trunk = 0.9007 / 0.6902 / 0.6709; paired Δ **−0.0025,
   CI [−0.0103, +0.0047]**, all three CIs include zero. C8b's gate now passes on
   MSAD *and* PreVAD.
2. **PreVAD pretraining transfers, and it is the campaign's real finding.**
   Against the cold MSAD-only arms, same seed, same arm type: MSAD **AP +0.0474**
   (t95 [+0.0318, +0.0630]), DoTA **AUC +0.0376** (all three clip CIs exclude
   zero, sign stable). Our KIP-off arm now **beats** the released checkpoint on
   MSAD: AUC +0.0039 (t95 [+0.0019, +0.0058]), AP +0.0604 (t95 [+0.0506,
   +0.0702]). Best MSAD numbers the project has: KIP-off mean 0.8988 / AP 0.7036.
   Also relieves the saturation limitation — DoTA frames > 0.99 fall 86 % → 38-48 %.
3. **The KIP A/B in this campaign is BLOCKED (lesson C14). Do not quote its
   sign.** Δ(on − off) is −0.0024 ± 0.0159 on DoTA (sign flips across seeds) and
   −0.0031 ± 0.0014 on MSAD — but **the KIP module was never trained**. The graft
   supplied `kip.*` at random init and the finetune ran 160 steps, so
   `L_KIP_rec` ends at 10.2/13.3/10.9 against the cold campaign's **4.91**,
   `L_KIP_align` never leaves 4.61 (**above** the 4.265 chance level; cold
   reached 3.70), `L_kin` 0.73 → 0.67 (cold 0.47). Three variables moved at once:
   trunk source, removal of the stage-1 KIP warm-up, and `num_epochs` 125 → 40.

**What it does license — a negative control on H4.** The gate+shift smoother is
fully present and active in these runs and delivers −0.002 ± 0.016. **The KIP
architecture with an untrained PMG head buys nothing**; only a trained flow head
produced +0.09. That is the first evidence against the strong form of H4
("KIP is a fixed temporal smoother whose content is irrelevant") — confounded by
the trunk change, so suggestive, not decisive.

**New limitation 14 on the report:** the +0.09 has been measured under exactly
one trunk. **One run closes it** — PreVAD trunk **+ the stage-1 KIP warm-up +
`num_epochs=125`**, one arm per seed. Until then §§6-11 stand as measured and the
trunk is an open confound.

**Also true, and it matters for the next planning pass:** broad-domain
pretraining is an independent lever on the *same* quantity KIP targets. A method
that is redundant with cheap pretraining is a different contribution from one
that composes with it, and **the composition is unmeasured**.

**Not on disk, despite how the campaign was described:** there is **no PreVAD
KIP-on arm** and **no per-seed PreVAD evaluation**. `outputs/PreVAD/` holds
`gate_p0`, `eval_trunk` and one `stage2_kip_off`. PreVAD was a pretraining
corpus, never an A/B benchmark — and **permanently cannot be one** (see the
2026-08-29 entry: PreVAD ships no pixels, so RAFT targets are unbuildable).

**Two new defects → lessons 22 and 23 (index + meta-index trigger map):**
- **C22** — `core.evaluate` min-max normalizes in **float32**, `core.tools.rescore`
  in float64; same `.npz`, different numbers (AP 0.369953 vs 0.366069, ΔAP
  0.0039 — the order of several reported deltas). `*_raw` agree exactly, so the
  divergence is created by the normalization, where float32 rounding manufactures
  ties among saturated frames. **Every number in the new docs uses the float32
  path** to match all existing `results.json`; deltas are paired so no conclusion
  moves. Fix in `core/metrics.py:normalize_scores`, then `rescore --write` over
  every run dir.
- **C23** — 90 of 2,606 PreVAD test ids contain `:`, written as `_` in
  `scores/*.npz`. A naive stem↔`video_id` join silently drops 3.5 % of the test
  set. It produced a wrong per-superclass table before the `?` bucket was noticed.
- **Lesson C17 upgraded to HIGH.** `--init-weights` is still unrecorded; this is
  now the *third* campaign whose arms had to be reconstructed from step-1 `mil`
  and from the Colab script. ~30 LOC.

**Process note:** Gate P0 was recorded *after* the runs it was meant to gate
(`PREVAD_SETUP.md` §10 step 8 gates steps 9-13). It passed, so nothing was
wasted — but the gate did no gating.

**Analysis provenance:** paired clip bootstrap, 2,000 draws, `default_rng(0)`,
both arms on the same resample, arm pairs checked for identical clip ids and
identical ground truth before differencing; seed-level two-sided 95 % t-intervals
(t = 4.303, n = 3). Scripts were scratch and are **not committed** — promoting
the paired bootstrap into `core/tools/` with tests is on the next-steps list.

## 2026-08-22 — the v2 layer: audit, mechanism-corrected spec, and six code facts

Three documents landed that the memory bank had not absorbed until this
reconcile. They are **the current planning frame**, above every `RESULTS_*.md`:

| Doc | What it is |
|---|---|
| `core/docs/REPORT_KIP_MSAD_DOTA.md` | the campaign write-up (committed in `7da7b06`) |
| `core/docs/v2/KAT-VAD_experiment_audit.md` | an audit of it — accepts the numbers, narrows the claims |
| `core/docs/v2/KAT-VAD_spec_v2.md` | mechanism-corrected spec: 10 changes, each tied to a measured defect |
| `.project/plans/katvad-v2-next-steps.md` | **the live plan** — tiers 0–5, facts F1–F6, 4 open decisions |

**The audit's two binding corrections to how results are stated:**

1. **"18 CIs excluding zero" over-counts.** It is 3 seeds × 3 metrics × 2
   controls; within a seed the metrics share score curves and the KIP-on arm is
   shared across controls. Independent replications = **3**. Headline becomes the
   seed-level t-interval **[+0.070, +0.113]** (still ~2.3× the +0.03 bar). Proof
   this matters: the stage-1-only effect had every clip-bootstrap CI excluding
   zero *while its sign flipped across seeds* — seed-level that is [−0.065, +0.050].
2. **The MSAD null is a bounded null.** "Costs nothing in-domain" was not
   measured; "any in-domain effect is < ≈1 AUC point at n=3" was.

**Six code facts from reading the tree — in neither the report nor spec v2:**

- **F1 — the gate MLP is never trained.** `(ratio * max_shift).floor().long()`
  (`core/kip/gate_shift.py:112`) is non-differentiable. **I verified this
  empirically: all 6 tensors get `grad is None`.** The 321 params stay at random
  init for the whole run. Two consequences: every "motion-gated" claim needs
  "**321 frozen, randomly-initialised** parameters", and **H4** is now open —
  KIP may be a fixed temporal smoother, which is exactly what buys AUC on a
  within-clip localization benchmark. Nothing measured refutes it.
  *(The memory bank previously said "zero gradient **at init**". That was wrong
  and is corrected in [[progress]] and [[systemPatterns]].)*
- **F2 — the "free" bypass eval is not free.** `use_gate_shift=false` sets
  `self.shift = None`, and `core/inference.py:64` loads at `strict=True`, so a
  KIP-on checkpoint raises on `kip.shift.mlp.*`. That is lesson C5 working —
  **do not weaken it.** Add `kip.bypass_gate_shift` inside `KIP.forward` (~15 LOC).
- **F3 — `ê_O` is 23-dimensional and non-spatial.** 23 frame-global scalars
  projected to 256-d. No localized/peripheral-motion story is expressible in the
  target; restate as a *global flow-statistic residual*.
- **F4 — student/teacher aspect mismatch is exactly 4:3**, source-independent, a
  bias not noise, and it warps the direction channels. Confirmed, not suspected.
- **F5 — 21,281 `.npz` score files are on disk**, so most of spec v2 §10 is
  computable today with zero GPU.
- **F6 — FP suppression is partly measurable now**: MSAD's test set has 20
  `Traffic_accident` + 120 normals — a real traffic detection slice with a
  negative pool, already scored for 3 arms × 3 seeds.

**Where the plan diverges from spec v2:** ECMR (Change 1) as specified inherits
F1's frozen MLP, so running it would test "a random 2→1 map of a new signal"
rather than the idea. Resolve F1 first, as an explicitly ablated change.

## 2026-08-24 — PreVAD prerequisites: G1, G2, G3 implemented (no run yet)

Runbook: **`core/docs/PREVAD_SETUP.md`** (status header updated). This session
wrote **code only** — nothing has been downloaded, unzipped, trained or scored.

The plan the runbook proposes: pretrain a KIP-**off** trunk on PreVAD's released
`ViT-B-16-8p` features (35,279 clips, no raw video shipped), graft KIP onto it,
and run the A/B on MSAD where the flow cache already exists. That answers the
one thing the +0.09 is missing — a Δ(on − off) measured on top of a properly
pretrained baseline instead of an MSAD-only one. §10 lists 13 steps; steps 5–6
(the blocking code) are now done, plus step 2.

**What landed**

- **`core/data/prevad.py`** (G1) — reads `train.csv`/`test.csv` into the layout
  contract, argparse CLI. **No `--stride`**: PreVAD ships no frames, the `.npy`
  *is* the interval-8 sequence, so lengths come from the array header and
  `--clip-dir` is required. §4.3's sanity checks are folded in and unskippable
  (width must be 512, non-empty, L2 norm sampled and warned on).
- **Multi-span labels** (G2) — every span filled independently, clamped to
  `[0, L]`. Lesson **18/C18**.
- **`_PREVAD_CLS_DEFS` rewritten** (G3) — the baseline's *live* 36-name
  `DEFAULT_CLASSES`, not the CamelCase block commented out above it. Lesson
  **19/C19**.
- **`core/tests/test_prevad.py`** — 38 tests. **322 green** (was 284).
  ruff / mypy / bandit / pyright / pycycle all clean.
- One pre-existing test updated: `test_data_utils.py::test_seeded_rng_is_deterministic`
  used the CamelCase `"CarAccident"` against the `prevad` dict. Intended
  consequence of G3, not a regression.

**What the data actually says** (measured from the real CSVs, not the doc)

| | |
|---|---|
| train | 32,673 rows — 22,000 normal / 10,673 abnormal, all with descriptions |
| test | 2,606 rows — 1,300 normal / 1,306 abnormal (**49.9 % normal → raw pooling**, the MSAD protocol, C12) |
| classes | **35** observed, all inside the baseline's 36; `Fire-related Accident` is a superclass with no rows |
| multi-span | **104** test clips, up to **4** windows |
| **defect** | **440 spans end past 1.0**, max **1.2104** |
| **defect** | **1 reversed span** (`RrUW8ITUqx0_aug3`, 0.9814 → 0.7110) |

Both defects are **absorbed, not repaired** — the baseline clamps them for free
by filling a 512-long buffer, so the released ground truth is defined by that
behaviour. All three quirk counts are logged on every run.

**Two claims in the runbook were wrong and are corrected in it**

1. G3 would **not** have raised. `class_index_tensor` compares against
   `defs.json`, which we generate from the same table; `verbalize_class_name`
   returns the bare class name for an unknown class *by design*. Composed, they
   are silent — the model would have trained on `"Store Robbery"` instead of a
   definition sentence, voiding the definition-conditioning claim with no error.
   `check_definition_coverage` in the preprocessor is now the real gate.
2. The taxonomy is **35 in the data, 36 in the lookup** — `defs.json` carries
   only observed classes so `H_mul` gets no permanently-negative column.

**Verified end-to-end on the real annotations** with a synthetic feature cache:
35 classes with `Normal` first, 2,606 clips labelled, label/feature lengths
aligned, and the real 4-span clip `qhly283_BV1Lf4y117wS` producing **4 disjoint
positive runs (40 rows)** — a hull collapse would have given 66.

**Next action: step 3 of §10** — verify the 2.39 GiB zip on Drive
(sha256 `52fc1579…`), unzip to **local VM disk** (`/content/cache/clip/PreVAD_rel`,
never Drive — 35k small random reads), then §4.3 counts, §5 label build, and
**Gate P0**: `best.ckpt` on PreVAD test through our eval. P0 is the real gate;
if it reads near chance the released features are not compatible and steps 9–13
are wasted.

**Still open, unchanged:** G4 (per-video descriptions → the caption branch) is
deferred by §8's own recommendation — option A (class-name verbalizer) first, B
as its own arm. Descriptions are carried into `meta.json` so they survive for it.
The trunk will be **n=1** (§7.1); seed variation covers stage 2 only. And per
lesson 14, none of this licenses tuning KIP on the +0.09.

## Open questions to settle before the next run

1. ~~No measured numbers recorded.~~ **Closed 2026-08-01** — see
   `core/docs/RESULTS_MSAD.md`. Remaining sub-question: confirm
   `ckpts/best.ckpt` (the `full_gate_a` source) really is LaGoVAD's released
   checkpoint, and note that our AUC is computed on 18,350 stride-8 sampled
   frames, not the 146,012 raw frames the paper number uses.
   **Closed 2026-08-16.** Both halves now matter less than they did: whatever
   `best.ckpt` is, our arms are statistically indistinguishable from it on MSAD
   under an identical protocol (eight paired-bootstrap CIs, all include zero),
   and it cannot reach 0.9041 either (0.8991 crop / 0.8949 ncc). The stride-8
   sampling difference is one plausible reason the paper's number is out of
   reach and is now recorded as such rather than as an open defect.
   `RESULTS_PHASE_A.md` §4; lesson 8b.
2. ~~**Feature-extraction transform discrepancy.**~~ **Decided 2026-08-08 —
   switching to `no_center_crop`**, on the KIP/RAFT field-of-view argument as
   much as baseline parity (see above). The switch is a full re-extract +
   retrain into `*_ncc` paths; the center-crop artifacts are kept, and the
   MSAD 0.9052 reproduction is **unverified until re-measured**. Cheap go/no-go
   first: `COLAB.md` §R0. Original text below for the record.
   **Closed 2026-08-12 — measured** (`RESULTS_NCC.md`). The switch was right for
   the field-of-view reason (C13), **not** for parity: `gate_a` scored *worse*
   under ncc on both benchmarks, so P1's generalization is refuted. The MSAD
   reproduction is re-measured and now **fails**: 0.8922 vs published 0.9041.

   `core/tools/extract_clip_features.py:41-64`
   resizes the shorter side to 224 then **center-crops** (the
   `CLIPImageProcessor` convention). The baseline's own extraction script uses
   `augmentation='no_center_crop'` (`LaGoVAD-PreVAD/tools/extract_feat_clip.py:39`),
   i.e. a plain anisotropic `Resize((224,224))`. These are different transforms
   and they produce incompatible feature caches. The user's paper-matching MSAD
   result was obtained with the **center-crop** version that is in this tree.
   → Do **not** "fix" this on theory alone. If it is ever changed, every cached
   feature and every metric measured on it is invalidated and must be redone.
3. ~~Gate (c) has no recorded result.~~ ~~**Measured, not settled** — passes on
   the point estimate, fails statistical defensibility at n=1 seed.~~
   **Closed 2026-08-16.** n=3 seeds, Δ +0.0915 ± 0.0088 on DoTA, nine CIs
   excluding zero. Statistically defensible against H2. ~~Still confounded by
   the warm start and H3.~~ **Both confounds closed 2026-08-19**
   (`RESULTS_ARM4_PROBE.md`): Δ(on − off_warm) +0.0988 ± 0.0148 and H3 rejected
   at matched convergence. The gate passes with no open confound; only the
   *mechanism* is unexplained.

4. ~~Does `--init-weights` accept a KIP-enabled stage-1 checkpoint into a
   `kip.enabled=false` model?~~ **Answered from the code, 2026-08-16: no, it
   raises.** `warm_start_model` (`core/train.py:98-108`) rejects *unexpected*
   keys and its error text names this exact case. A4 therefore needs a one-off
   prep step — strip `kip.*` into `checkpoint_last_nokip.pt` (cell A4.0 in
   `COLAB.md`). Do **not** weaken the guard with an "ignore unexpected" flag;
   it is lesson 5 working as designed.

## Immediate next steps

> **Superseded 2026-08-28 by the PreVAD analysis above.** The v2 track is
> **paused on the user's explicit instruction**. The live queue is now:
> **(1)** the one unblocking run — PreVAD trunk + stage-1 KIP warm-up +
> `num_epochs=125`, one arm per seed (report limitation 14);
> **(2)** fix C22 (float64 pooling) and re-`rescore --write` every run dir;
> **(3)** the run manifest for `--init-weights` (C17, now HIGH);
> **(4)** ~~a PreVAD KIP-on arm~~ — **ruled out permanently 2026-08-29**, PreVAD
> ships no pixels so RAFT targets cannot be built (see the 2026-08-29 entry);
> **(5)** promote the paired bootstrap into `core/tools/` with tests.
> The v2 tiers below are still valid work, just not next.

**Revised 2026-08-25.** Phase A (H2), A4 (warm start) and A5 (H3) are done and
no confound is open — but the audit reframes what is missing. It is no longer
just "the mechanism": **no ablation has attributed the +0.09 to any part of
KIP**, and F1 opened a rival hypothesis that two near-free runs could settle.

Ordering principle, adopted from the audit: **attribution before architecture.**

1. **v2 tier 0 — offline re-analysis. Zero GPU, local, ~1–2 days of code.**
   21,281 `.npz` files are already on disk. Delivers: MSAD traffic-slice
   breakdown (F6), seed-level t-intervals everywhere, the in-domain equivalence
   bound, the three metric stubs (`AUC_A`/MCC/mAP@IoU, `core/evaluate.py:58-71`),
   raw-vs-min-max side by side, per-class DoTA tables. → `RESULTS_OFFLINE_V2.md`.
   **Flag every threshold metric as void while saturation is 86–89 %.**
2. **v2 tier 1 — inference-time attribution. No training.** Prerequisite: the
   ~15 LOC `kip.bypass_gate_shift` (F2). Then **T1.2 constant-shift** (tests H4)
   and **T1.3 gate-reseed ×5** — the cheapest experiments in the program, and two
   of them can falsify the central claim. **Pre-register the readings first.**
3. **v2 tier 2 — T2.3 shuffled flow targets** (~5 LOC + a config flag so the
   manifest records it). If +0.09 survives a temporally permuted `e_O`, the
   motion claim is dead. Run this *before* the aspect re-extract (T3.1) — if the
   gain survives permutation, the expensive re-extract answers nothing.
4. **PreVAD track, in parallel** (no GPU contention with 1): step 3 of
   `PREVAD_SETUP.md` §10 — verify the zip on Drive, unzip to **local VM disk**,
   §4.3 counts, §5 label build, then **Gate P0**. P0 is the real gate.
5. **Before any new training arm:** the run manifest (lesson 17, ~30 LOC), so the
   next 9 runs are self-documenting.
6. **Backlog:** `_rng_payload` as raw bytes (lesson 15) — deferred while arms are
   compared. Phase B (val split) — still deferred on evidence, but note spec v2
   Change 6 wants it *mandatory* for calibration; revisit if saturation becomes
   the blocker.

**Four decisions the plan needs from the user** (`katvad-v2-next-steps.md` §10):
H4 controls now or after tier 0 is banked; commit now to deleting `L_KIP_align`
if its respec shows nothing; schedule the aspect re-extract or document it as a
limitation; and what story to tell if tier 1 says "regularizer, not forward-pass
mechanism".

Per lesson 14: still do **not** change KIP's architecture, losses, or
hyperparameters on the +0.09. Every tier 0–3 item repairs something *measured* or
attributes something *unmeasured*; tier 4 (ECMR) is the only architectural change
and it is gated on tiers 1–2.

## Working agreements

- Preference (`.project/preference.md`): English, direct, practical, plan in
  detail, no filler.
- `LaGoVAD-PreVAD/` is read-only. All code goes in `core/`.
- After any bug fix or feature: add a lesson (`lessons-learned/`, CLAUDE.md §7).

See [[progress]] for what is built vs. outstanding, [[systemPatterns]] for the
architecture, [[techContext]] for the stack.
