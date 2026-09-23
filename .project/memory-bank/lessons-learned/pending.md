# Pending lesson candidates

Observations that are real but not yet validated against the 5 gates
(`GATES.md`). Promote to `index.md` once they pass; delete if disproven.

---

## P1 — Extraction transform vs. the checkpoint author's script

**Observation:** `core/tools/extract_clip_features.py:41-64` resizes the shorter
side to 224 then center-crops. `LaGoVAD-PreVAD/tools/extract_feat_clip.py:39`
uses `augmentation='no_center_crop'` (anisotropic `Resize((224,224))`).

**Why it is not yet a lesson:** the user's MSAD result at `b9978ff` matched the
paper **with the center-crop version**. Whether the baseline's transform would
score higher, lower, or identically on MSAD is unmeasured. The generalized form
("a feature transform must come from the checkpoint author's extraction script")
is plausible but is exactly the kind of claim that needs a measurement, not a
deduction.

**To validate:** extract one MSAD cache each way, evaluate the same checkpoint on
both, report both numbers. Until then the settled part of this is lesson 2
(caches are transform-bound), which stands on its own.

**RESOLVED 2026-08-12 — and the generalization is refuted.** Both caches were
built and the same checkpoints scored on each (`core/docs/RESULTS_NCC.md`).
LaGoVAD's released `best.ckpt` scored **worse** under `no_center_crop` on both
benchmarks (DoTA macro −0.0170, CI [−0.0301, −0.0037], excludes zero; MSAD
−0.0042, ns). Our KIP-off arm dropped on MSAD too (0.9052 → 0.8922). So "a
feature transform must come from the checkpoint author's extraction script" is
**not** supported — either `best.ckpt` was not trained on `no_center_crop`
features, or our resize differs from the baseline's in some other way.

The switch to `no_center_crop` was still correct, but for the *other* reason:
internal field-of-view consistency between the appearance and flow branches
(lesson 13), which is worth +0.1514 macro AUC on the KIP arm. Do not promote P1
as written. The measured half is now lessons 13 and 14.

---

## P2 — Hard `floor()` kills the gate MLP's gradient

**Observation:** the spec's shift `s_t = ⌊r_t · D/K⌋` has zero gradient
everywhere, so the gate MLP that produces `r_t` receives no learning signal and
stays at its initialization. Asserted by a test in
`core/tests/test_kip_modules.py`.

**Why it is not yet a lesson:** whether this actually hurts is a training-time
question. An untrained gate is still a *fixed random* gate, which may be
adequate, and the fix (straight-through estimator or soft shift) is a design
change to the novel module, not a bug fix.

**To validate:** compare stage-2 MSAD with the gate as specified vs. a
straight-through variant, matched seeds.

---

## P3 — Resume is FP-tolerant, not bitwise

**Observation:** weights after kill-and-resume match a straight run within
floating-point tolerance but not bitwise. Apple Accelerate BLAS reduces
nondeterministically across threads regardless of `torch.set_num_threads`.

**Why it is not yet a lesson:** this is accepted behavior with a test that
encodes the accepted tolerance, not a defect. It becomes a lesson only if it
ever causes a real divergence on CUDA.

---

## P4 — `L_dvs` row-gating deviation

**Observation:** `train.py` applies the `L_dvs` pair only to rows that are normal
or synthesized, because our `y^p` is all-zero for un-synthesized abnormal clips
(true window unknown under weak supervision). The baseline marks the whole
abnormal anchor pseudo-positive.

**Why it is not yet a lesson:** it is a documented, deliberate deviation
(`core/docs/TRAINING.md` §1) flagged as "first knob to revisit if gate (b)
misses". It becomes a lesson only once a run tells us which side is right.

---

## P5 — Close a confound with a control arm, not with new machinery

**Observation:** `.project/plans/msad-ncc-seeds-and-selection.md` §3 planned a
`--val-ratio` split, a KNN rebuild, `core/tools/select_checkpoint.py` and a full
retraining cycle to answer H3 (does under-convergence explain KIP's DoTA gain?).
What actually answered it was **10 evals of checkpoints that already existed**
(`RESULTS_ARM4_PROBE.md` §4), on the training set every other number came from,
so nothing lost comparability. The warm-start confound fell to **one extra arm**
with no code at all.

**Why it is not yet a lesson:** n = 1. The generalization ("prefer the cheapest
arm that isolates the factor over building selection/eval machinery") is
plausible and matches how A4/A5 went, but one cycle is not evidence that it
holds when the confound is subtler. Gate 2 (recurrable) is likely; gate 1
(evidence) has a single instance; gate 4 (enforceable as a diff-checkable rule)
is the weak one — this is a planning rule, not a code rule.

**To validate:** the next time a confound appears, record whether a control arm
or existing artifacts could have answered it, and what the alternative would
have cost. Promote after a second instance.


---

## P6 — An ablation flag that only changes the *data* is not an ablation

**Observation:** `kip.disable_pmg=true` reads as "no PMG head". It does one
thing: `require_flow = cfg.kip.enabled and not cfg.kip.disable_pmg`
(`core/train.py:593`), so `DVSFeatureDataset` returns **zero** flow rows
(`core/data/dataset.py:103`). The 311,808-param PMG head is still built, still
forwarded, and `L_KIP_rec` still regresses `ê_O` toward those zeros — a
degenerate target, trained at full `lambda_rec`. A "plain-TSM control" written as
`gate_type=constant + disable_pmg=true` is therefore *not* a plain TSM: it is a
TSM plus an auxiliary head being fit to zeros. The clean control needs
`loss.lambda_rec=0`, `loss.lambda_align=0` and `kip.use_lkin=false` as well.
Found 2026-08-30 while writing `core/docs/v3/setup/MSAD_DOTA_V3_SETUP.md`;
verified by reading the code and by checking that `v^k` is bit-identical under
wildly different `ê_O` once `gate_type=constant`.

**Second instance, already settled:** `require_flow=False` was rejected as a
PreVAD KIP-on workaround for exactly this reason ("it trains the PMG head to
predict zeros", `PREVAD_SETUP.md` §7.4) — same root, different caller.

**Candidate rule:** An ablation flag must remove the module or zero its loss
weight, not merely starve its input; if it only starves the input, name it after
what it does (`zero_flow_targets`) and assert in a test that the ablated arm's
parameters receive no gradient.

**Why it is not yet a lesson:** gate 1 (evidence) has two instances of the same
root but only one of the flag itself; gate 4 (enforceable) is strong — a test
asserting `p.grad is None` for every `kip.pmg.*` parameter under the ablation
would catch it in a diff. Promote once an arm is actually run under it, or once
a third naming-vs-behaviour instance appears. Related: [[lesson-24]] (assert the
gradient for anything you call learned) — this is its mirror image: assert the
*absence* of gradient for anything you call ablated.

---

## Candidate — a key-match is not an identity check: two arms with the same key layout load into each other silently

**Observed 2026-09-01** while auditing §6 of `core/docs/v3/setup/MSAD_DOTA_V3_SETUP.md`.

`load_kip_state_dict` keys its gate-mismatch check on `kip.shift.mlp.*`
(`core/models/ckpt_compat.py:152-192`). That protects both directions between the
`mlp_*` family and the parameter-free family. It **cannot** protect `rank` from
`constant`: both have zero gate parameters and byte-identical key layouts, so an
A1 (`rank`) checkpoint scored under `--set kip.gate_type=constant` loads cleanly,
raises nothing, and returns an AUC for a model that was never trained that way.
`core/evaluate.py` writes no config, so the eval directory does not record which
gate produced it either.

The runbook had exactly one eval command, hardcoded to `GATE=constant`. Copying
it for the `rank` arm is a one-character edit that produces a silently wrong
number — the same shape as C14, one layer down.

**Candidate rule:** When two configurations share a state-dict key layout, the
checkpoint cannot discriminate them — write the discriminator into the *output*
(a manifest) and assert it from the artifact (here: `kip_s` span ≈128 for `rank`,
exactly 0 for `constant`) before reading any metric.

**Why it is not yet a lesson:** gate 1 (evidence) has **zero** measured
instances — the failure is demonstrated by reading the code, not by a run that
went wrong. Promote if a v3 arm is ever scored under the wrong parameter-free
gate, or fold into C5 if `evaluate.py` is changed to write a resolved config
(which would make it moot). Related: [[lesson-5]] (fail loud on key mismatch —
this is the case where the keys *match* and it is still wrong), [[lesson-17]]
(a run must record every flag that defines what it is), [[lesson-24]].

---

## The split directory is the label; the filename prefix is a trap (TAD, 2026-09-02)

Building TAD's weakly-supervised train split (P1) meant deriving a video-level
label for ~410 clips with no annotation file. Two rules were available, and the
repo already contains the wrong one: `core/data/msad.py` ships
`--infer-abnormal-from-name`, which marks a window-less row abnormal by class-name
prefix. TAD's ids look like they invite the same treatment (`01_Accident_106`).

Measured on the shipped `tad_test_anno.json`: the 60 abnormal test videos carry
**seven** filename prefixes — `01_Accident` (23), `05_else` (14),
`06_PedestrianOnRoad` (10), `07_RoadSpills` (4), `02_IllegalTurn` (3),
`03_IllegalOccupation` (3), `04_Retrograde` (3) — and LaGoVAD labels every one of
them `Car Accident` (`LaGoVAD-PreVAD/src/datasets/TAD.py` has a two-class
taxonomy). A prefix rule keyed on `Accident` would have mislabelled **37 of 60**
abnormal videos as normal, in the training supervision, silently.

**Candidate rule:** Derive a video-level label from the directory the dataset
ships the clip in, never from its filename, and raise on a clip that fits
neither bucket rather than defaulting it to normal.

**Why it is not yet a lesson:** gate 1 (evidence) fails — the mislabelling was
found by measuring the annotation *before* writing the rule, not by a run that
went wrong. `core/data/tad.py:_label_from_directory` takes the directory and
refuses ambiguity; `core/tests/test_tad.py::TestFailsLoud` pins both halves.
Promote if any dataset adapter is ever caught inferring a label from a filename.
Related: [[lesson-18]] (a shipped label format is defined by its widest case),
[[lesson-19]] (a graceful fallback hides a missing taxonomy).

---

## `mlp_ste` produces NaNs under AMP (DADA, 2026-09-06)

**Observation:** the A4 arm (`kip.gate_type=mlp_ste`) trained on DADA-2000 logged
`NaN` in **33 of 500 steps**, spread over 17 of 20 epochs. The NaN sets are
always `mil` + `mul_mil` (plus `kin` in 17 steps, `dvs_sup`/`dvs_sup_mil` in 16),
while `kip_rec` and `kip_align` stay finite in the same step — so the NaN enters
the **score path, after KIP**, not the flow branch. The first occurrence is
`global_step 1`. No other arm has a single NaN at the same seed (2024), batch
size (64), dataset and dataloader ordering, which localizes it to the one piece
of code only this arm runs: `shift_channels_straight_through`
(`core/kip/gate_shift.py:99`) under `train.amp=true`.

Ruled out: `_topk_k` returning 0 on a short clip — it is
`max(1, n_valid // topk_pct)` (`core/losses/mil.py:22`), so k >= 1 for every
length, and a length-driven fault would hit all seven arms identically.

**Why it is not yet a lesson:** gate 4 (ENFORCEABLE) fails — the mechanism is not
established, so the only rule statable today is "watch out for `mlp_ste`", which
is not checkable in a diff. Gate 1 passes (real, measured, A4's DADA row is
unreportable because of it).

**Mechanism candidate added 2026-09-06 (code read, not yet measured):**
`core/train.py` has **no `clip_grad_norm_` anywhere** and calls `scaler.step()`
(line 487) without `scaler.unscale_()`. `mlp_ste` is the only arm where a
gradient reaches the gate MLP and PMG head *through the score path* —
`mlp_frozen`'s is identically zero ([[lesson-24]]) and `rank`/`constant` are
parameter-free — so it is the only arm whose backward pass traverses the sigmoid
surrogate in `shift_channels_straight_through`. `GradScaler` skips inf/NaN
steps but does nothing about a large *finite* gradient, which is a sufficient
path to a weight blow-up and an fp16 overflow on the next forward. That predicts
the observed signature exactly (score path NaN, flow branch finite). If it
holds, the rule generalizes beyond the STE to "AMP without gradient clipping is
not a safe default on any newly-unblocked gradient path", which *is* checkable
in a diff.

**To validate:** rerun the arm with `train.amp=false`, separately with AMP on
but the STE weights forced to fp32, and separately with AMP on plus
`clip_grad_norm_(1.0)`, and see which stops the NaN. If it is fp16
overflow in the selection weights, the lesson generalizes to "a straight-through
estimator's surrogate weights must not inherit the autocast dtype" and can be
stated as a rule. Related: [[lesson-24]] (the STE exists because the spec's own
formula is a no-op), [[lesson-14]] (an arm measured under an open defect measures
the defect).



---

## A runbook path is not a fact until it is checked against disk (2026-09-06)

**Observation:** `DADA_V3_SETUP.md` and `DADA_SETUP.md` prescribed
`clip/DADA2000_ncc`, `knn/DADA2000_ncc` and `$OUT/DADA_<arm>_s$S` for the whole
DADA-2000 campaign. **No such directory was ever created.** The notebook that
actually ran it (`collab/DADA/v3/train.py`) used `clip/DADA2000`,
`knn/DADA2000` and `$OUT/DADA2000/<arm>_s$S`, and all seven arms plus 62k score
files live under those names. The divergence survived a full campaign, a
results document and an audit gate list, because every gate checked
*consistency across arms* and none checked *the doc against the disk*. Cost: the
runbook could not be used to launch seeds 2025/2026 without silently creating a
second, empty cache tree, and the `_ncc` suffix — the project's only carrier of
"this cache is `no_center_crop`", per [[lesson-2]]/[[lesson-13]] — was absent
from the path that really held the features.

**Why it is not yet a lesson:** gate 3 (RECURRING) is unproven — one instance so
far. Watch whether TAD's runbook has the same drift when its campaign runs; a
second instance promotes it.

**Candidate rule:** Every runbook that names a cache or output path ships a
preflight cell that `ls`-es each one and prints a count, and the campaign's
results document records the paths **as run**, not as prescribed. Related:
[[lesson-17]] (a run must record every flag that defines what it is) — this is
the same failure one level up: a *document* that does not record what was run.

---

## `--strict` is a gate, not a repair (2026-09-06)

**Observation:** `RESULTS_DADA.md` §8.3 recorded "4 abnormal test clips lost
their window at stride 8; `--strict` was not set". Read as a to-do, that invites
the fix "add `--strict`" — but `core/data/dada.py:429` **raises** on that
condition. Setting the flag does not clean the split, it aborts the build. The
actual defect is what the default does: the 4 clips enter the *test* split with
an all-zero label vector, indistinguishable from genuine `0_Normal_Driving`
clips, on a test set where 74 % of frames already come from all-normal clips —
so they inflate both the micro AUC and the constant-score clip oracle
([[lesson-12]]).

**Why it is not yet a lesson:** gate 3 (RECURRING) — DoTA's
`build_frame_labels` has the identical default-warn shape, so the *pattern*
exists twice, but only DADA has produced a corrupted number so far.

**Candidate rule:** A validation flag that raises is a gate; the corresponding
repair must be written separately and named (here: emit the vanished ids to a
file and exclude them at scoring). Never record "flag X was not set" as if
setting it were the fix.

---

## A fact about "the tree" is a fact about *a branch* (2026-09-08)

**Observation:** `main` and `v3` diverged at `fac71a3`. The memory bank, the
lesson catalog and `CLAUDE.md` were all committed to **both** branches with
identical content, so on `main` they described `core/kip/ecmr.py`, four
`gate_type`s, "84 Python files" and "537 tests green" — **none of which exist
there**. The tree measured 79 files, 10,484 source LOC, and **12 failing tests**
(`test_{dada,tad}.py::TestXTrainsUnderEveryGate`, `KeyError: 'Unknown config
key: kip.gate_type'`). Two shipped runbooks on `main`
(`v3/setup/{DADA,TAD}_V3_SETUP.md`) prescribe six arms that cannot parse on the
branch that carries them. Nothing warned: docs and tests travel with a merge,
config fields do not.

**Repaired 2026-09-08 (the shape of the repair matters).** The 12 failures were
resolved by **collapsing the v3-only parametrization, not deleting the class**:
`TestDadaTrainsUnderEveryGate` / `TestTadTrainsUnderEveryGate` became
`TestDadaTrains` / `TestTadTrains` over `V1_KIP_ARM =
["--set", "kip.gate_signal=flow_norm"]`. 7 parametrizations removed, 425 → 418
collected, **418 pass**. Deleting the classes would also have deleted
`test_kip_off_trains` (arm A0), `test_stage1_warmup_runs` and the
`config.yaml`-recording assertions — all v1-valid, and the only evidence that the
TAD and DADA adapters emit trainable files, neither corpus having been trained
yet. The other tempting repair — adding `gate_type` to `core/config.py` so the
v3 params parse — was rejected: the v3 tests also need `ecmr.py`, the STE shift
and the diagnostics, so it converts 12 loud failures into one silent wrong gate.

**Why it is not yet a lesson:** gate 3 (RECURRING) — one occurrence so far. It is
one branch split, though a costly one: the failure mode is a session planning
against a component that is not in its own tree. A second divergence (or a
merge that silently reunites the two memory banks) promotes it.

**Candidate rule:** State the branch beside any count, file list or config flag a
memory bank asserts, and re-measure them after a checkout rather than trusting
the committed text. Before running a documented command, grep the flag in
`core/config.py` **on the current branch**. **When a test fails only because it
exercises another branch's feature, narrow the parametrization to what this
branch can express — never delete the enclosing test, and never port the config
field alone to make it parse.** Related: [[lesson-17]] (a run must record every
flag that defines what it is) and the runbook-path candidate above — the same
failure at the branch level.


**Second instance — 2026-09-14, TAD (`main`).** The runbook-drift candidate above
said "watch whether TAD's runbook has the same drift". It did, one level up.
`core/docs/TAD_SETUP.md` on `main` was written against `v3`: it told the reader
`git checkout v3`, used the `Thesis-V3/kat-vad` repo path, asserted
`gate_shift.GATE_TYPES == ('rank','mlp_frozen','mlp_ste','constant')` as a
preflight, and delegated its entire experiment layer to
`v3/setup/TAD_V3_SETUP.md`, whose six arms raise
`KeyError: Unknown config key` on `kip.gate_type`, `kip.const_shift_ratio` and
`kip.disable_pmg` here. Both documents additionally pass
`train.checkpoint_every_steps`, removed from `TrainConfig` on 2026-09-13
(`cb7e2ac`) on **both** branches — so that one is a *time* drift, not a branch
drift, and it proves the same hole from the other side: a runbook is not
re-validated when the config it drives changes.

Two `main`-only constraints the v3 runbook cannot express, found only by reading
the tree: `require_flow = cfg.kip.enabled` (`core/train.py:715`) means **every**
KIP-on arm needs the RAFT cache — v3's cheap flow-free control does not exist
here; and `adopt_checkpoint_architecture` (`core/inference.py:83`) makes the
checkpoint authoritative for `model.*`/`kip.*` at eval, so v3's rule "repeat the
gate flags on every eval command" is inverted on `main` — a contradicting
`--set kip.*` **raises** ([[lesson-C34]]).

**Gate 3 (RECURRING) is now satisfied for both candidates.** Repaired by
rewriting `TAD_SETUP.md` as a self-contained `main` runbook with every `--set`
run through `load_config` on this branch and every shell block expanded in bash,
plus a §2.3 preflight that asserts the three v3 flags **raise**. Awaiting the
user's call on promotion to `index.md` (the `meta-index.md` trigger map already
carries two rows for this failure mode, pointing at [[activeContext]] rather
than at a numbered lesson).

**Candidate rule (merged, one sentence):** Before copying a command, a flag, a
file count or a test count out of any document, run `git branch --show-current`
and parse-test the flags against *that* branch's `core/config.py` — docs and
tests travel with a merge, config fields do not.

---

## Weakly-supervised MIL optimizes clip classification when the bag task is trivially separable (TAD, 2026-09-15)

**Observation.** Trained in-domain on TAD, the KIP-off arm `m0` reached
clip-level AUC **0.9975** and micro AUC **0.9237** — the latter landing on the
corpus's own constant-score-per-clip oracle of **0.9226** — while frame-level
`auc_macro` **fell** from the zero-shot trunk's 0.7578 to 0.6174, `d = gap/σ`
from +0.903 to +0.419, and DoTA transfer from 0.6158 to 0.5496. The signature is
four metrics moving together: **clip-AUC ↑, macro ↓, d ↓, transfer ↓.**

**Mechanism.** No term in the objective lowers a frame inside a positive bag
(`mil_loss` raises the top-k, `pseudo_sup_mil_loss` the in-span top-k,
`multi_class_mil_loss` the class top-k). TAD's abnormal clips are 33 % positive,
so 67 % of their frames get no downward pressure; and the two classes are two
directories with different length distributions (a frame-count-only ruler scores
clip-AUC 0.6940). A per-clip constant satisfies `L_MIL` completely, so
localization is never required and never learned.

**The strong form, measured.** Warm-starting from the PreVAD trunk (`t2_warm`)
starts the model at macro 0.7578 and ends it at **0.6540** after 504 steps.
**The training signal destroys 0.104 of localization it was handed** — not
merely "fails to learn it".

**Why it is not yet a lesson.** n = 1 seed, one corpus. DADA is consistent but
confounded by C27 (median clip 9 under kernel 9). The rule needs either a second
seed or a second corpus where the C27 confound is absent. **The destruction curve
(`--stop-after-epochs 3 7 18 36`) is the cheap validation** and is queued.

**Candidate rule.** *Before training on a new corpus, compute its clip oracle and
its length ruler. If micro ≈ oracle, in-domain training will buy clip ranking and
sell localization; gate on `auc_macro` and clip-level AUC, never on micro.*
Distinct from **C12** (a metric artifact) and **C28** (a corpus artifact): this is
a **training** disease caused by both.

→ `outputs/EDA/TAD`, `core/docs/TAD_SETUP.md` §8.1/§15.1, [[activeContext]] 2026-09-15

---

## Blind downward pressure creates variance without direction (TAD, 2026-09-15)

**Observation.** `loss.bottomk_weight=1.0` — the term written precisely to supply
the missing downward pressure above — **worked mechanically and made everything
worse**: its loss fell 0.2294 → 0.0062, `mil` rose 0.5707 → 0.6730 (real
tension), within-clip **range grew +37 %** and the raw gap **+78 %**, yet
`auc_macro` fell 0.6174 → **0.6016**, `d` fell +0.4189 → +0.3280 and DoTA macro
fell 0.5496 → **0.5331**. The user's `bottomk_topk_pct=8` dose arm was
indistinguishable, closing the "pressure too weak" escape.

**Contrast with DADA Phase 1**, where the same flag failed by *shrinking* the
score scale (C31, range −20 %). Here the scale **widened** and the ranking still
degraded. Different failure, same verdict.

**Candidate rule.** *A loss that only says "something here must be lower" adds
within-clip variance with no information about which frames. Report `auc_macro`
beside any range/gap statistic — a widening range with a falling macro means the
term is injecting noise.* The directed counterpart already exists and is unwired:
`L_neg` pools **under the model's own anomaly curve**
(`attn = softmax(logits/0.02)`) so a caption constrains *which* frames rise.

**Why it is not yet a lesson.** n = 1 seed; and the directed alternative has not
been run, so "direction is what was missing" is inference, not measurement.

→ `core/losses/mil.py:abnormal_bottomk_loss`, `core/losses/contrastive.py`,
  `.project/plans/katvad-tad-loss-ladder.md`

---

## C31 amendment — two standard readings of `gap/σ` order the arms oppositely (2026-09-15)

**Observation.** On the TAD ladder the two natural estimators disagree in sign:

| arm | per-clip standardized, then averaged | `gap / mean within-clip σ` (C31's literal wording) | **macro AUC** |
|---|---:|---:|---:|
| `m0` | +0.4189 | +0.1821 | 0.6174 |
| `t1_bottomk` | **+0.3280** (falls) | **+0.2414** (rises) | **0.6016** (falls) |

`auc_macro` is rank-based and immune to scale, and it agrees with the *per-clip*
estimator. C31 as written prescribes the estimator that disagrees.

**Candidate amendment.** *Standardize within each clip first, then average; and
let `auc_macro` arbitrate whenever the two disagree.* Keep C31's core claim
("a loss that lowers scores is not a loss that creates contrast") — only the
recipe changes.

**Why it is not yet promoted.** One corpus, one ladder. Re-check against DADA
Phase 1's saved `.npz` before editing C31 in `index.md` / `meta-index.md` —
if the per-clip estimator also flips DADA's conclusion, the amendment is bigger
than a wording fix.

→ `core/docs/RESULTS_DADA_PHASE1.md` §4, [[activeContext]] 2026-09-15

---

## A failed reproduction gate is diagnosed, not chased (TAD, 2026-09-15)

**Observation.** Gate T0 scored **0.7912** against a published **89.56**.
`TAD_SETUP.md` §8's table routed that to "pipeline defect, prime suspect §4.3
frame ordering" — and four of five suspects were then exonerated by measurement
in minutes: ids 100/100 against LaGoVAD's own `tad_test_anno.json`, frame labels
**0/100 mismatch** when rebuilt from that file's `anomaly_span`, stride 8 = the
baseline's `interval=8`, pooling raw = raw. **Frame ordering was cleared by
`auc_macro = 0.7578`** — shuffled frames would give ≈0.50.

**Candidate rules.** (a) *A macro AUC clear of chance exonerates frame ordering
in one step — read it before re-auditing an ingest.* (b) *Before blaming the
pipeline for a micro-AUC gap, recompute micro with every clip weighted equally;
on TAD that moves 0.7912 → 0.6574, so most of the "gap" was length weighting.*
(c) Reinforces **C8b**: the measured 0.7912 becomes the reference, and a published
number that sits *between* the corpus's length ruler (0.8968) and its clip oracle
(0.9226) should never have been a target.

**Why it is not yet a lesson.** (a) and (b) are one corpus each. (c) is already
C8b and needs no new entry — fold the diagnostics into C8b's remedy rather than
adding a lesson, if the next gate failure confirms the recipe.

→ `core/docs/TAD_SETUP.md` §8.1

---

## `TRAFFIC_DEFINITIONS` is exported but never consumed (2026-09-15)

**Observation.** `core/data/definitions.py:53` defines `TRAFFIC_DEFINITIONS`
(4 sentences) with the comment *"Spec §7.4: traffic definition set used for
DoTA/DADA-style zero-shot eval"*, and `core/data/__init__.py` re-exports it. No
other module references it: `_DOTA_CLS_DEFS` uses
`_UNIVERSAL_CLS_DEFS["CarAccident"]` instead.

**Why it is not a lesson.** It is a single dead symbol, severity **LOW**, and
gate 2 (generalizable rule) does not obviously pass. **Action instead of a
lesson:** decide whether the spec's §7.4 set should be wired for DoTA/DADA
zero-shot or the symbol deleted, and record the answer in `TRAINING.md`
§deviations. Leaving it as-is means a reader assumes spec §7.4 is implemented.

→ `core/data/definitions.py:53`, `core/data/__init__.py:13`

---

## A windowed corpus must be split by SOURCE VIDEO, not by window (2026-09-15)

**Observation.** `plan_record_windows` re-shards one clip into several windows
(`window_id` = `{source}__w{index:03d}`, up to `WINDOW_MAX_PER_CLIP = 4`). On the
planned DADA-original corpus (T2, `.project/plans/katvad-dada-original-corpus.md`)
this yields **~3.8 windows per source video** — 4,053 abnormal + 1,962 normal
windows from 1,908 videos, with **1,159 videos contributing both classes**.
`split_records` stratifies on `(is_abnormal, fault_label)` over *records*; if the
records handed to it are windows, two windows of the same accident — overlapping
by 50% at hop = W/2, i.e. sharing 8 of 16 sampled frames — can land on opposite
sides of the split. The test number then measures memorization of a clip the
model trained on.

**Candidate rule.** *When a corpus is re-sharded into windows, split on the
source id (`Window.source`), never on the window id; assert afterwards that no
source appears in both splits.*

**Why it is not yet a lesson.** Not observed in the wild — no windowed corpus has
been **trained** yet (`w32s2` and `w24s1` were built and profiled, never trained),
so this is predicted, not measured. Gate 1 (evidence of a real failure) does not
pass. Promote it the moment a windowed corpus is trained, or demote it to a
one-line assertion in `plan_record_windows` if the split code is fixed first.

**Cheap mitigation available now:** `split_records` already accepts `--split-file`
(test ids). Generating that file at *source* granularity sidesteps the issue
without touching the splitter.

→ `core/data/dada.py:plan_record_windows`, `core/data/dada.py:split_records`,
  `core/data/windows.py:window_id`

---

## A diagnostic's prose must implement its own decision table (2026-09-15)

**Observation.** `check_p1.py` (`core/docs/DADA_ORIGIN_PHASE0.md` §6) is the hard
gate for the DADA-2000 original corpus: if the clips are trimmed, frame labels
cannot be derived by fraction (`core/data/dada.py:296`) and the whole plan is
void. Its §6.1 decision table is pre-registered and quantitative — PASS at
**≥ 95 %** matched, STOP at **< 80 % or a strongly one-sided mean delta**.

The script implemented none of it. It printed the raw counts, then emitted

> *"A systematic one-sided delta means the clips were TRIMMED … the fraction
> mapping in core/data/dada.py:296 is then WRONG"*

**unconditionally, whenever `deltas` was non-empty** — no floor on the number of
mismatches, no relative scale, no verdict line. On the 2026-09-15 probe it fired
over **one** clip off by **3 frames on 345 (−0.87 %)**, at a matched rate of
**96.7 %** — i.e. the table's clearest PASS. The defect it names measures
**−79 %**. A reader who trusted the output and not the table would have stopped a
corpus migration that had just passed its gate.

**Candidate rule.** *A script that decides a pre-registered gate must print the
verdict, computed from the table's own thresholds as named constants. Any
diagnostic prose it emits must be gated on the same condition as the verdict it
implies — never on "something was non-zero". A verdict over n items prints n.*

**Why it is not yet a lesson (the 5 gates).** Gate 1 (a real failure) passes:
the wrong prose was printed on a real gate run, and it was read. Gate 2
(generalizable) passes. Gates 3–5 are the problem — **it is a one-off probe
script inside a runbook, not in `core/`**, so there is no file for a rule to be
enforced against and no test that could fail. It also has **n = 1 occurrence**:
the sibling case (a script whose prose contradicted its own table) has not been
observed elsewhere in this tree.

**What to look for before promoting.** Other pre-registered gates in this project
whose decision lives in prose while the script prints only raw numbers —
`core/tools/eda` verdict sections and the Gate D0 probe are the two candidates.
If either shows the same shape, promote with a **[HIGH]** severity: the failure
mode is a correct measurement read as its opposite, which is C14's family.

**Already fixed, so the candidate is about the pattern, not this script.**
§6's `check_p1.py` now carries `PASS_RATE` / `NOTE_RATE` / `MAX_MISSING` /
`TRIM_MIN_{N,AGREE,REL}` as constants, prints `VERDICT: …` in §6.1's four
branches, prints `matched EXACTLY`, and states `n` beside the verdict. Verified
on four fixtures: the real probe (PASS), a −79 % trim (STOP), an empty root
(BACK TO P2), and 83 % with two-sided ±5 deltas (PASS WITH A NOTE).

**Related:** **C33** (a pre-registered threshold must be *reachable*) is the same
family one step earlier — C33 is about writing a threshold nobody can hit, this
is about writing one nobody computes. **C14** is the consequence: a number read
under the wrong precondition.

→ `core/docs/DADA_ORIGIN_PHASE0.md` §6, §6.1, §8.1;
  `outputs/EDA/DADA2000Origin/phase0_report.md` §3.2

---

## A symlink farm must link at the IMAGE directory, not the clip directory (2026-09-15)

**Observation.** `core.data.dada.materialize_flat_dir` is the project's answer to
lesson **C26** (the extractors key their folder scan on bare `Path.name`): it
builds `flat/{video_id} -> real clip folder` so the stock tools see unique ids
"with no change to either tool", as its docstring says.

That works only when the frames sit *directly* in the clip folder, which is true
of the **trimmed** DADA archive (`type1_vid001/*.jpg`) and false of the
**original** release (`DADA2000/{type}/{video:03d}/images/*.png`).

`list_frame_folders` (`core/data/video_io.py:79`) walks `root.rglob(subdir)`, and
**`pathlib` refuses to recurse into a symlinked directory** — a cycle guard
present in every version from 3.10 (measured on this tree, 3.10.6) through 3.13.
So a clip-level farm plus `--frames-subdir images` finds **nothing**:

```
flat/{id} -> .../{video:03d}   + --frames-subdir images  ->  ValueError: No frame folders found
flat/{id} -> .../{video:03d}/images  + no subdir         ->  400 folders, 400 unique ids
```

A link at the *final* path component is matched (it is not recursed *through*),
and `list_frame_images` follows it via `iterdir`.

**Candidate rule.** *Point a symlink farm at the directory that directly holds
the files, and drop `--frames-subdir`. A farm one level up is invisible to any
`rglob`-based scan.*

**Why it is not yet a lesson.** **It fails loudly** — `ValueError: No frame
folders found under …` — so gate 1 (a real, costly failure) is weak: it costs
minutes, not a wrong number, which is what this catalog is for. It is also n = 1
so far, seen while writing `colab/DADA2000Origin/build_d0_dataset.py`.

**Promote it if** Phase 2's real adapter inherits the clip-level link and someone
"fixes" the resulting error by reaching into `core/data/video_io.py` — the caller
graph there is 5 CRITICAL hops (DoTA and TAD included), so *that* would be
expensive. Until then the guard is documentation:
`core/docs/DADA_ORIGIN_PHASE1.md` §4 and trap 3, plan §1.5, and the docstring of
`build_d0_dataset.materialize_flat`.

**Related:** **C26** (bare `Path.name` as an id), **C25** (a second source path
added to an existing cache), **C10** (a directory is not evidence of data).

→ `core/data/dada.py:401` (`materialize_flat_dir`),
  `core/data/video_io.py:79` (`list_frame_folders`),
  `colab/DADA2000Origin/build_d0_dataset.py:materialize_flat`

### Addendum 2026-09-15 (later) — a SECOND instance, so gate 1 now has n = 2

`pick_probe.py` (`core/docs/DADA_ORIGIN_PHASE0.md` §4) picked **one clip per type
and stopped**, so `--n` was silently capped at the number of strata (52). Phase 1
asked for **400** and got **52** — and `DADA_ORIGIN_PHASE1.md` §2 asserted the
script "fills", a behaviour it never had. The run completed cleanly: the builder
verified 52/52 ids, P1 98.1 %, labels within the pre-registered bar. Every check
passed **on the wrong sample size**, and the only symptom was one line reading
`picked 52` where the command said `--n 400`.

Same shape as the `check_p1.py` case above: **the document asserted a behaviour
the code did not have, and nothing in the output contradicted it loudly enough.**
Two independent instances in one day, in two different scripts, both in the
DADA-original runbooks.

**The rule the pair suggests, wider than the original:** *a script that takes a
requested quantity or decides a gate must print what it actually delivered next
to what was asked (`picked N of --n M`, `VERDICT: … (n = N)`), and any prose it
emits must be gated on the same condition as the decision it implies.*

**Why still not promoted.** Both instances are throwaway probe scripts in
runbooks, not `core/` code, so there is still no file to enforce a rule against
and no test that could fail. **Promote the moment a third instance appears inside
`core/`, or when the D0 builder graduates into `core/data/dada.py` in Phase 2 —
at that point the rule becomes enforceable and the severity is [HIGH]**, because
the failure mode is a correct measurement read at the wrong scale.

**Fixed:** the fill pass is in place, and the script now prints `picked N` plus
`types covered: X/52` and warns when the population is smaller than `--n`.
Verified old-vs-new at n = 5 / 30 / 52 (bit-identical, so Phase 0 reproduces) and
n = 100 / 400 / 5000 (fills, no duplicates, all 52 types retained).

---

## An editable install on the dev machine masks every import-path bug (2026-09-16)

**Observation.** `colab/DADA2000Origin/build_d0_dataset.py` imports `core`. It was
written, gated (ruff/mypy/bandit/pyright), and exercised on a 30-clip fixture and
a 3-shard flow — **all green locally**. On Colab it died on the first shard with
exit 1 and no visible message.

Cause: `python /content/build_d0_dataset.py` puts **the script's directory** on
`sys.path[0]`, never the working directory, so `cd "$REPO"` does nothing for the
import. Measured:

```
python /tmp/fakecontent/probe.py   (cwd=/tmp)   sys.path[0] = '/private/tmp/fakecontent'
  import core FAILS -> No module named 'core'
PYTHONPATH=<repo> python /tmp/fakecontent/probe.py
  import core OK
```

**Every local test passed because this venv has the project installed editable**
(`.venv/…/site-packages/_editable_impl_kat_vad.pth`), which puts `core` on the
path for *any* interpreter invocation from *any* directory. Colab has no such
install. Re-checked deliberately: running the script with `PYTHONPATH` stripped
still exits **0** on this machine.

**Candidate rule.** *A script that imports the project but ships outside the
package cannot be validated on a machine where the package is pip-installed. Pass
`PYTHONPATH=<repo>` explicitly at every call site, and test it from a directory
outside the repo with the install neutralised — or `cd` into the repo and use
`python -m`, which does put the cwd on `sys.path`.*

**Why it is not yet a lesson.** One occurrence, in a throwaway script, and it
fails loudly once the stderr is visible. **The generalization is real though**,
and it applies to `core/` the moment anyone runs a project entry point by file
path instead of `-m`: every runbook in this repo uses `python -m core.tools.…`,
which is immune. Promote if a `core/` entry point is ever invoked by path.

**Compounding defect, fixed with it.** The shard driver wrapped the child in
`subprocess.run(..., check=True)` without capturing output, so the failure
surfaced as a bare `CalledProcessError: returned non-zero exit status 1` with the
child's `ModuleNotFoundError` nowhere in the traceback — **one whole round trip
spent on a diagnosis the child had already printed**. The driver now runs the
short steps with `capture=True` and prints the child's stdout/stderr on failure.
Same family as the two candidates above: *the tooling hid what it already knew.*

→ `core/docs/DADA_ORIGIN_PHASE1.md` §4.0 (the warning), §4.2 (`run(..., capture=True)`),
  traps 14–15

### Addendum 2026-09-16 — fixing the producer was not enough; the CONSUMER must assert

`pick_probe.py` was given a fill pass on 2026-09-15. On 2026-09-16 the Phase 1
shard loop still processed **52 clips** (40 + 12) and reported
`shards done: 2 | cached clips: 52` as a success. The fixed script was verified
against the real `dada标注.xlsx` the same day — `--n 400` returns **400 rows,
52/52 types** — so the script was right and `/content/d0_clips.json` was simply a
stale 52-row file from the previous session.

**Nothing downstream checked.** The loop read the file, saw 52 items, and ran. A
producer fix does not survive a stale intermediate; **only the consumer can catch
that**, because only the consumer knows what it expected.

This is the fourth instance of the family (after `check_p1.py`'s prose,
`pick_probe`'s silent cap, and the swallowed `ModuleNotFoundError`), and it
sharpens the candidate rule:

> *Assert the size and shape of every intermediate you did not produce in the
> same cell. Print `got N, expected M` and stop when they differ — a pipeline
> stage that accepts whatever it is handed will eventually be handed the
> previous run's output.*

**Fixed:** `phase_1.ipynb` cell 7 re-reads `d0_clips.json` after writing it and
asserts `len(rows) == N_WANT` plus uniqueness, printing the file's mtime; cell 13
refuses to start unless the file holds exactly `N_WANT` clips, and says how to
recover. Both verified to fire at 52 and pass at 400.

---

## Candidate 2026-09-16 (a) — a caption field is only a caption if it is unique

**Raised by:** Phase 2 of the DADA-2000 original corpus, deciding whether
`L_neg` could finally be activated. Measured, not argued.

**Triggers:** L_neg, caption, contrastive, InfoNCE, cap_contrastive, descriptions, G4

**Problem:** a per-video text field that is really a *category label* makes
`asymmetric_infonce_loss` unsatisfiable, and it fails silently — the loss simply
plateaus at a positive floor.

`core/losses/contrastive.py:20-38` treats each caption's own video as the **only**
positive. Two clips sharing a string give identical text embeddings, so `ano_sim`
(S,S) carries two identical rows against a diagonal target: no parameter setting
satisfies it, and the gradient pushes two videos of the *same* accident type
apart — the opposite of definition-conditioning.

**Measured on `data/DADA/dada标注.xlsx`, 1,962 rows (2026-09-16):**

| caption source | unique | rows colliding | at `BATCH_SIZE` 64 |
|---|---:|---:|---:|
| `texts` alone | **81** (4.1 %) | 99.0 % | **78 % of a batch** |
| `texts+causes`+ 4 scene attrs | 1,124 (57.3 %) | 58.1 % | 10 % |

**Bad:**
```python
# "the dataset ships a text column, so L_neg can run"
captions = [row["texts"] for row in abnormal_rows]   # 81 distinct strings
```

**Good:**
```python
# count distinct values BEFORE wiring any contrastive loss to a text field
unique = len({normalize(row[col]) for row in rows})
assert unique / len(rows) > THRESHOLD, f"{col} is a category label, not a caption"
```

**Rule:** Measure a text field's distinct-value ratio before wiring it to a
contrastive loss; treat anything under ~50 % unique as a class label, and compose
extra annotation columns until the collision rate at the real batch size is
tolerable.

**Files:** `core/losses/contrastive.py:20-38`, `core/data/dada_origin.py:META_COLUMNS`,
`.project/plans/katvad-dada-original-phase2-t2.md` §3.4

**Correction this candidate carries:** `N3_MIN_SCORE_RANGE = 0.2`
(`core/losses/contrastive.py:91`) gates only the **mining** branch — when it
fires, `mined` is empty and `L_neg` falls back to vanilla contrastive, still
computed. `progress.md` and `activeContext.md` describe it as a blocker on
`L_neg` as a whole; it is not.

**Gate status:** not yet validated against `GATES.md`. Derived once, from one
corpus. Promote if a second dataset's "description" field turns out to be a
taxonomy.

---

## Candidate 2026-09-16 (b) — address a hand-maintained sheet by its columns, never its name

**Triggers:** xlsx, spreadsheet, sheet name, annotation, openpyxl, Sheet1

**Problem:** `dada标注.xlsx`'s sheets are named the **opposite** of their
contents — `name="text"` holds the 1–38 type taxonomy, `name="Sheet1"` holds the
1,962-row per-clip table. A reader keyed on the name (or on sheet order) parses
zero usable rows and the failure reads exactly like a corrupt file.

**Bad:**
```python
rows = load_workbook(path)["Sheet1"].iter_rows()   # the decoy
```

**Good:**
```python
sheet = find_sheet(path, REQUIRED_COLUMNS)   # first sheet carrying every column;
                                             # on failure prints every header
```

**Rule:** Detect a spreadsheet's sheet by the columns it carries, and on failure
print every sheet's header, so a re-export that renames a tab fails loudly
instead of reporting an empty corpus.

**Files:** `core/data/xlsx.py:find_sheet`, `core/data/dada_origin.py:REQUIRED_COLUMNS`

**Gate status:** not yet validated. The rule is now enforced by code, which may
make it a code-level invariant rather than a lesson.

---

## Candidate 2026-09-16 (c) — write the unrebuildable artifact to durable storage in the step that CREATES it

**Third instance of the same failure.** Phase 1 (2026-09-15) lost its per-clip
frame census to a recycled Colab runtime. `phase_2.ipynb` was written the next
day *with that lesson quoted in its own markdown* and still put
`counts_dir = P2 / 'counts'` on `/content`, copying to Drive only in §6. On
2026-09-16 §3 failed with `ValueError: No census file matched
['/content/p2/counts/*.json']` after the extraction had already run.

**Triggers:** colab, census, runtime recycled, /content, intermediate artifact, C17, resume

**Problem:** "copy the artifacts to durable storage at the end" is not a
policy — it is a race against the runtime. The steps between creation and the
copy are exactly the steps during which the artifact is irreplaceable.

**Bad:**
```python
counts_dir = VM_LOCAL / 'counts'      # ... and §6, much later, copies it to Drive
```

**Good:**
```python
counts_dir = DRIVE / 'DADA2000_orig' / 'counts'   # written where it is created
```

**Rule:** Decide for every intermediate whether re-creating it is cheap; write
the ones that are not to durable storage **in the step that creates them**, not
in a later "record the run" step — and give the expensive step a second,
independent way to rebuild that artifact.

**The second route, added here:** `7z l -slt` reads the archive's central
directory and yields an exact per-clip image count **without extracting a byte**
(`phase_2.ipynb` §2.0). A lost census now costs one minute, not a re-extraction
of 94 GiB. Section 3 merges archive-index counts with per-shard counts (shard
wins, being what the extractor actually read) and intersects with the CLIP cache,
so a clip can never enter the corpus with labels but no features.

**Files:** `colab/DADA2000Origin/phase_2.ipynb` §2.0/§2/§3

**Gate status:** three instances of the family, one of them a repeat *after* the
lesson was written down. Strong promote candidate — the existing C17 says
"record the run", which is demonstrably not enough to prevent this.

---

## Candidate 2026-09-16 (d) — every line of an `.ipynb` `source` list must carry its own `\n`

**Triggers:** ipynb, notebook, colab, source, generated notebook

**Problem:** `nbformat` concatenates the `source` list **verbatim**. A generator
that does `text.split("\n")` produces lines with no terminator; the file is valid
JSON, every cell still `compile()`s after a join, and Jupyter often renders it —
but **Colab shows the whole cell as one line**. The defect is invisible to every
check short of opening it in Colab.

**Bad:**
```python
cell["source"] = text.split("\n")
```

**Good:**
```python
cell["source"] = text.splitlines(keepends=True)
```

**Check:** `any(not s.endswith("\n") for s in cell["source"][:-1])` must be False
for every cell.

**Files:** `colab/DADA2000Origin/phase_2.ipynb`

**Gate status:** one instance, but zero-cost to enforce and it survived a syntax
check, a JSON check and a structural check before the user hit it.

---

## ~~Candidate 2026-09-16 (e)~~ — **PROMOTED to lesson C35** (2026-09-16)

The per-clip window cap is not the lever for the clip oracle. Promoted once Gate W
had run at **both** geometries, so the lesson cites a failing corpus (W=16,
oracle 0.7529) and a passing one (W=20, 0.7037) rather than a simulation.

→ `index.md` §35 · `detailed.md` §35 · `meta-index.md` (inline + trigger map)

---

## Candidate 2026-09-16 (f) — a frozen encoder that reads a library's INTERNAL layout fails late, in the middle of a run

**Triggers:** transformers, CLIPTextModel, text_model, colab, runtime drift, pin,
soft prompt, AttributeError, clip_text, C7, C21, preflight

**Problem:** `core/models/clip_text.py` reaches into three transformers internals —
`self.model.text_model` (line 115) plus the private
`_create_4d_causal_attention_mask` / `_prepare_4d_attention_mask`. Colab's
Python-3.13 runtime now ships **transformers v5**, where `CLIPTextModel` no longer
wraps a nested `.text_model`, and the run died with
`AttributeError: 'CLIPTextModel' object has no attribute 'text_model'`.

What makes it a lesson is **where** it died: not at import, not at config parse,
but inside `compute_losses` on the first batch — after the dataset was resolved
(5,507 windows), CLIP was downloaded, the device was picked, and `config.yaml`
had already been written to the run dir. The run directory looks *started*. Three
of the project's own layers had already passed: the notebook's command builder,
argparse, and `load_config`.

**Why it was not caught earlier.** `phase_1`/`phase_2` exercise only the CLIP
**vision** tower through the public API, so the same drifted stack extracted
1,945 clips without complaint. The text tower is the only module that depends on
internal layout, and nothing ran it until training started.

**Bad:** a `%%bash` cell that installs conveniences (`av`, `einops`, `faiss-cpu`)
and leaves the stack to whatever the runtime ships, while `techContext` pins
`transformers==4.56.*`.

**Good:** pin the library the internals belong to, **and** smoke-test the exact
path on a 1-layer random-weight model at the top of the notebook — one second, no
download, no Drive I/O:

```python
tiny = CLIPTextModel(CLIPTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                    num_hidden_layers=1, num_attention_heads=2,
                                    max_position_embeddings=77, eos_token_id=2))
SoftPromptCLIPTextModel(clip_model=tiny, num_soft_prompts=4)(
    input_ids=ids, attention_mask=mask).pooler_output
```

The module is already injectable (`clip_model=`) precisely so tests need no
download; the preflight just uses the seam that exists.

**Candidate rule.** *Every module that reads a library's internal layout gets a
data-free smoke test at the top of any notebook that will reach it, and the
library is pinned wherever that notebook runs. A dependency check belongs where
the failure is cheap, not where it is discovered.*

**Note on the remedy NOT taken.** Porting `clip_text.py` to v5 mid-campaign was
rejected: it sits on the score path, an untested port changes `z^t` silently, and
every arm would then measure the port (C14). The v5 port is a separate, tested
task — and `torch` is deliberately **not** pinned on Colab any more, because
torch 2.4 has no cp313 wheel and the feature caches were built on the runtime's
own torch.

**Files:** `core/models/clip_text.py:115`, `colab/DADA2000Origin/phase_4.ipynb` §0/§0.1,
`.project/memory-bank/techContext.md`, `core/docs/COLAB.md` Cell 0.2

**Gate status:** one instance, but it is the second environment-drift failure in
this project after C15 (pickled numpy state) and the same shape as C7. Promote if
it recurs on another module, or fold into C7 as an enforcement clause.

---

## Candidate 2026-09-16 (g) — a pre-registered bar must carry the metric it was measured in

**Triggers:** pre-registered, bar, headline, auc_macro, auc micro, min-max,
0.6408, comparison table, C12, C33, C35, phase 4

**Problem:** Phase 4's pre-registration named the headline as
*"DoTA zero-shot `auc_macro` > **0.6408**"*. **0.6408 is a MICRO min-max number** —
the parent plan's §7.2 table is headed *"micro (min-max)"* and 0.6408 is its
`ours, MSAD-trained kip_on` cell. The bar and the metric it was written against
came from the same table, one column apart, and the pre-registration silently
crossed them.

It very nearly produced a wrong verdict. Measured: T2 KIP-off DoTA macro
**0.6113**, micro **0.5856**. Against "macro > 0.6408" the corpus reads *refuted*.
Against the correct macro bar (**0.6529**, the same MSAD KIP-on runs) it is also
below — but against the **like-for-like** row that actually applies to a KIP-off
trunk (MSAD KIP-off, micro 0.5491 / macro 0.5453) T2 is **ahead by +0.037 micro
and +0.066 macro, all three seeds in the same direction**. Same numbers, opposite
conclusions, decided entirely by which cell of one table the bar was read from.

**Second half of the same defect:** the bar was a **KIP-on** arm and the run was a
**KIP-off** trunk. Comparing them refutes a corpus for not containing a component
it was never given. The T2 KIP-on arm is blocked on a RAFT pass.

**Bad:**
```markdown
| **headline** | DoTA zero-shot `auc_macro` | **> 0.6408** |
```

**Good:** carry the metric, the pooling and the arm in the bar itself, and state
the like-for-like row beside it:
```markdown
| headline        | DoTA `auc` micro, min-max   | > 0.6408 (MSAD **kip_on**)  |
| headline, macro | DoTA `auc_macro`, min-max   | > 0.6529 (MSAD **kip_on**)  |
| like-for-like   | either, vs MSAD **kip_off** | > 0.5491 / 0.5453           |
```

**Candidate rule.** *A pre-registered bar is a triple — number, metric, arm. Write
all three, and re-read them off the source table when the result arrives; a bar
that names only a number will be compared to whatever metric is nearest to hand.*

**Files:** `.project/plans/katvad-dada-original-phase2-t2.md` §6.2.3 (corrected in
place, with the correction visible), `.project/plans/katvad-dada-original-corpus.md`
§7.2, `colab/DADA2000Origin/phase_4.ipynb` §0/§7

**Gate status:** one instance, caught before anything was published, by re-reading
the source table rather than by any check. Sibling of **C35** (a threshold needs a
measured lever) and **C12** (a metric is defined by its pooling). Promote if it
recurs, or fold both into C33 as "a pre-registered quantity is number + metric +
arm + attainable range".


---

## (h) A sharded extraction loop must scope `--ids-file` to the SHARD, not the split (2026-09-17)

**Observation.** `phase_4.ipynb` §3 extracts RAFT targets 150 clips at a time —
7z one shard out of the 94 GiB archive, run RAFT, delete — and passed
`--ids-file train_ids.txt` (all **1,491** train ids) on every shard. The farm
holds 150 folders, so `core/tools/feature_cache.py:133` raised on the first one:

```
INFO core.data.video_io: Found 150 frame folders under /content/p4/farm_000
ValueError: 1341 requested ids have no frame folder under /content/p4/farm_000:
  ['t05_v109', 't05_v111', 't05_v112', 't05_v113', 't05_v115']
```

**The guard is correct; the call site was wrong.** `select_ids` exists so a
silently short cache fails at extraction rather than hours later inside training
(C10, C11). `--ids-file`'s own help string — *"restrict to these video ids… flow
is train-time only, so this is usually train_ids.txt"* — describes the **run's**
scope and reads perfectly natural at a call site whose real scope is one shard.

**Bad:**
```python
for k in range(0, len(shard_rows), SHARD):
    shard = shard_rows[k:k + SHARD]
    run(['7z', 'x', ..., *[f'{ROOT}/{r.type_id}/{r.video:03d}/images/*' for r in shard]])
    run([sys.executable, '-m', 'core.flow.raft_extract',
         '--frames-dir', farm, '--ids-file', T2 / 'train_ids.txt', ...])
```

**Good:** the file names this shard, so `select_ids` becomes the per-shard
completeness check it was written to be.
```python
    shard_ids = P4 / f'ids_{tag}.txt'
    shard_ids.write_text('\n'.join(r.video_id for r in shard) + '\n', encoding='utf-8')
    run([sys.executable, '-m', 'core.flow.raft_extract',
         '--frames-dir', farm, '--ids-file', shard_ids, ...])
```

**The obvious fix is the dangerous one.** Dropping `--ids-file` also clears the
error — the farm is already train-only by construction — and it silently deletes
the guarantee: a 7z that bungs 140 of 150 clips then produces a short cache with
no error at all, which is precisely C10's failure. **A flag that fires must be
re-scoped, not removed.** Two holes of the same family were closed with it: §3
wrote `census_{tag}.json` and **never read it** (`phase_2.ipynb` §2 does, and
raises on a clip the archive index says should be there), and `FLOW_READY` read
`len(have) >= len(train_ids)` — a **count**, which passes on an equal number of
*wrong* ids and then fails per item inside training.

**Candidate rule.** *When an extraction loop is sharded, derive `--ids-file` from
the shard being extracted, never from the split; and when a scope flag raises,
narrow the scope rather than dropping the flag.*

**Files:** `colab/DADA2000Origin/phase_4.ipynb` §3;
`core/tools/feature_cache.py:116-135` (`select_ids`);
`core/flow/raft_extract.py:290`, `core/tools/extract_clip_features.py:193`

**Gate status.** Gates 1/2/4/5 hold — real trace, the same `--ids-file` + shard
shape exists in both extractors and every future corpus, one-sentence rule,
nothing in `index.md` covers it. **Gate 3 is where it sits down: it fails loudly,
on the first shard, before any compute is spent** — same reason the 2026-09-16
import-path candidate is still here. The half that *is* silent is the wrong fix,
and that half has one instance. Promote if anyone drops a scope flag to clear an
error, or if a sharded loop ships a short cache. Closest relatives: **C10**
(a directory is not evidence of data) and **C17** (a run must record what it is);
fold into C10 if it recurs.

---

## ~~(i) 2026-09-18 — A loss's magnitude is a property of its target's units, not of the fit~~ — **PROMOTED to lesson C37** (2026-09-21)

> **Gate 3 resolved.** The candidate's own promotion condition was
> *"promote to [HIGH] if D2-a confirms `rho >= 1.0` at the trunk"*. D2-a measured
> **`rho` = 3.105** at the stage-2 end (11.63 at stage 1), `cos` = **−0.0010**
> (orthogonal), on three seeds × 8 batches — `outputs/v1/DADA2000_orig_diag_kip_loss_scale/`.
> D1 supplied the missing denominators: `V` = 31.64, `W` = 16.21, `K` = 11.62 →
> `R²_global` **0.633**, `R²_item` **0.283**, `mag_max` **83.0 %** of `E[s²]`.
> Severity **[HIGH]** is the one pre-registered here, kept deliberately: by the
> `GATES.md` letter a silently-invalid experimental result reads `[CRITICAL]`, but
> raising severity *after* seeing the result is the drift this catalog exists to
> prevent. Revisit only if a second instance appears.
> Text below is the candidate as written on 2026-09-18, kept for the audit trail.


**Trace.** Phase 4's KIP-on arms log `kip_rec` at **11.3–11.9** while the four
task losses together sum to **0.87–0.91**, i.e. `L_KIP_rec` is **93 % of
`total`** at `lambda_rec = 1.0`. The reading "the PMG head fits badly" is
unavailable: `L_KIP_rec` is a bare masked MSE
(`core/kip/losses.py:kip_reconstruction_loss`) against `e_O`, and `e_O` is 23
**unnormalized** frame-global flow scalars — `mag_mean/std/max`, `u/v mean/std`
in raw pixel units, plus an L1-normalized 16-bin histogram — lifted to 256-d by a
fixed Gaussian map scaled `1/sqrt(23)`
(`core/flow/raft_extract.py:flow_statistics`, `make_projection`). Nothing
normalizes them between the extractor and `core/data/dataset.py:114`. Every other
term in the objective is a BCE or an InfoNCE and sits at O(1) **by construction**,
so the two were never on a comparable scale and the 93 % says nothing about
either fit.

**Bad:** reading a raw MSE against an unnormalized target as a fit quality, or
setting its weight to 1.0 beside O(1) losses.
```yaml
loss:
  lambda_rec: 1.0        # against a target whose constant-predictor MSE is unmeasured
```

**Good:** measure what a constant predictor scores on the same target first, and
state R² against it.
```bash
python -m core.tools.eda report --sections features --no-probe \
  --flow-dir "$CACHE/flow/v1/$DATASET" ...   # prints Z / V / W baselines (§4.3.1)
```

**Candidate rule.** *Before weighting a regression loss beside classification
losses, measure the MSE a constant predictor scores on its target; report R²
against that baseline, and set the weight from the ratio rather than from 1.0.*

**Why it is more than bookkeeping here.** The scale is not inert. `PMGFlowHead`
reads `v^t` and stage 2 freezes nothing (`core/train.py:191-195` freezes only in
stage 1), so the gradient reaches the shared temporal encoder. Measured
consequences, all three seeds: stage 1 (trunk frozen) plateaus at `kip_rec`
**20.2**; stage 2 (trunk free) falls to **11.3** — 44 % of the gain came from
rewriting `v^t` — and every task loss ends **24–32 % higher** than the paired
`kip_off` run, with T2 micro AUC down **0.0129** (3/3 seeds, t95 excludes 0).

**Files:** `core/kip/losses.py:30-42`, `core/flow/raft_extract.py:86-118`,
`core/data/dataset.py:114`, `core/train.py:423-427`,
`outputs/v1/DADA2000_orig_phase4/s*/stage2_kip_*/metrics.jsonl`

**Gate status.** Gates 1/2/4/5 hold — measured trace, one-sentence rule, nothing
in `index.md` covers loss-scale commensurability, and the pattern generalizes to
any future auxiliary regression head. **Gate 3 is pending D1/D2**
(`.project/plans/katvad-kip-loss-scale-diagnosis.md`): the failure is currently
*silent* — nothing raises, the run completes, the number looks like a loss — which
is exactly the promotion argument, but the causal claim is not yet measured.
**Promote to [HIGH] if D2-a confirms `rho >= 1.0` at the trunk.** If D2-a refutes,
demote to [MEDIUM] and keep only the reporting half of the rule. Closest
relatives: **C14** (do not change the score path mid-campaign) and lesson **14**
(do not tune KIP on a delta) — the rule above deliberately derives the weight
instead of searching it.

---

## Candidate (iii) — 2026-09-21 — an openpyxl round-trip silently drops OOXML parts it does not model

**Severity proposed:** [LOW] (tooling, reporting artifacts only — no measured
number depends on it).

**Problem.** Editing `reports/Thesis_Report.xlsx` through
`openpyxl.load_workbook` → `save` rewrites the whole package. Parts openpyxl has
no model for are **not carried over and nothing warns**: this round-trip dropped
`xl/drawings/drawing{1,2,3}.xml`, `xl/persons/person.xml` and the two
`worksheets/_rels` that referenced them.

**Bad:** edit the shared workbook in place and assume a cell-value diff proves
nothing was lost.

**Good:** copy the file first, then diff the **zip member list** and the cell
values of every sheet you did not intend to touch.
```bash
unzip -l before.xlsx | awk '{print $4}' | sort > before.txt
unzip -l after.xlsx  | awk '{print $4}' | sort > after.txt
diff before.txt after.txt          # parts added/removed
```

**Candidate rule.** *After any openpyxl write to a workbook someone else
authored, diff the zip member list against a pre-edit copy and account for every
removed part before handing the file back.*

**Measured this time — the loss was harmless, and that is the point.** All three
`drawing*.xml` were 775-byte **empty** `<xdr:wsDr>` containers and `person.xml`
an **empty** `personList` (Excel/Sheets export residue); the four arXiv
hyperlinks on `Research Gap` and all 686 populated cells on the two untouched
sheets survived byte-identical. Had any of those drawings held a chart or an
image, the same silent path would have deleted it.

**Environment note that belongs with it.** `.venv` on this machine has **no
`pip`** and no `openpyxl`; `source .venv/bin/activate && pip install openpyxl`
reports "Requirement already satisfied" because `pip` resolves to the **system**
Python 3.10, whose site-packages does have it. The working interpreter for
spreadsheet work is
`/Library/Frameworks/Python.framework/Versions/3.10/bin/python3` — not `.venv`,
and not bare `python3`. A "requirement already satisfied" line is not evidence
that the *active* interpreter can import the module.

**Files:** `reports/Thesis_Report.xlsx`

**Gate status.** Gates 1/2/4 hold (measured trace, one-sentence rule, nothing in
`index.md` covers artifact round-trips). **Gate 3 is weak** — the failure is
silent but the blast radius is a report file, not a metric — and **Gate 5 is
weak**: this tree edits spreadsheets rarely. Keep in `pending.md`; promote only
if a second artifact round-trip loses something that mattered.

---

## Candidate (iv) — 2026-09-23 — `Trainer.compute_losses` is not repeatable; test inertness by poisoning, not by equality

**Severity proposed:** [LOW] (test design; no measured number depends on it —
but see the note on `grad_probe`).

**Problem.** Two calls of `trainer.compute_losses(batch)` on the same collated
batch, `model.eval()`, `torch.no_grad()`, even after `torch.manual_seed(0)`,
return different `mil` / `dvs_sup` / `mul_mil` / `total` on the stub fixture
(measured: `total` 3.2939 vs 3.2678, KIP off). The randomness comes from a source
the torch seed does not cover (not yet traced — stub text encoder or DVS/align
sampling are the suspects). **Not P3:** P3's BLAS reduction noise is ~1e-6;
this is 0.026 on the total, four orders larger — a sampling source, not
arithmetic. An "a knob does not change the loss" test written as
`total_a == total_b` fails for a reason unrelated to the knob.

**Bad:**
```python
base = trainer.compute_losses(batch)
trainer.cfg = replace(cfg, loss=replace(cfg.loss, lambda_rec=0.0316))
assert trainer.compute_losses(batch)["total"] == base["total"]   # flaky, and misleading when it fails
```
**Good:**
```python
trainer.cfg = replace(cfg, loss=replace(cfg.loss, lambda_rec=float("nan")))
losses = trainer.compute_losses(batch)
assert "kip_rec" not in losses and torch.isfinite(losses["total"])   # NaN reaches total iff the weight is used
```

**Candidate rule.** *Test that a loss weight is unused by setting it to NaN and
asserting the total stays finite — never by comparing two `compute_losses` calls.*

**Worth a follow-up:** `core.tools.grad_probe` draws 8 batches per point and
reports a large per-batch `rho` spread (D2: sd 1.56–9.54). Part of that spread may
be this non-repeatability rather than batch content. Not measured.

**Files:** `core/tests/test_flow_zscore.py::TestKipOffIsInert`, `core/train.py:235`

**Gate status.** Gates 1/2/4 hold. Gate 3 weak (a flaky test, not a wrong number),
Gate 5 unknown. Keep pending; promote if the source is traced to something that
also moves a measured metric.
