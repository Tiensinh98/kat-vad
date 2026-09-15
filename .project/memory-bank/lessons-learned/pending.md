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
