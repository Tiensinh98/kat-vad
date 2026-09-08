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

