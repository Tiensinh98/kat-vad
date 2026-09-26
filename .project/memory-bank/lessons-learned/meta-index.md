# Lessons — Tier 0 meta-index

**Re-initialized 2026-07-31 at commit `b9978ff`.** The previous catalog was
discarded on the user's instruction; every lesson below was re-derived by
reading this tree, with a file:line citation. Load this file before every
IMPLEMENTATION or DEBUG task (~400 tokens).

**Branch:** this is the `main` = **KAT-VAD v1** copy. Every lesson holds on both
branches; only C24's *remedy* differs (see its last line).

## Critical lessons (inline — read every time)

### C1 [CRITICAL] Env — faiss must be single-threaded after torch is imported
torch and faiss-cpu each bundle libomp on macOS arm64; multi-threaded faiss
segfaults inside `IndexFlat.search`. `faiss.omp_set_num_threads(1)` at import.
→ `core/data/knn_cache.py:34-37`

### C2 [CRITICAL] Data — a feature cache is only valid for the transform that built it
CLIP features, flow embeddings and the KNN cache are all keyed to one exact
preprocessing transform and one sampling stride. Changing either invalidates
every cached array **and every metric ever measured on it**. Never change a
transform to match a convention; change it only with a re-extraction plan and a
re-measured baseline.
→ `core/tools/extract_clip_features.py:41-64`, `core/constants.py:69`

### C3 [HIGH] Device — torch 2.4 MPS training diverges
Stage-1 `L_KIP_rec` climbs on MPS while CPU converges on identical seeds and
data. Pin `train.device=cpu` for local training; CUDA for real runs. Inference
on MPS is untested — assume the same until proven.
→ `core/docs/TRAINING.md` ("Device warning")

### C4 [HIGH] Supply chain — pin HF model revisions
`openai/clip-vit-base-patch16@main` ships only `pytorch_model.bin`, which
transformers 4.56 refuses to `torch.load` on torch < 2.6 (CVE-2025-32434). The
pinned sha is the safetensors conversion. Never resolve a model at `main`.
→ `core/constants.py:CLIP_MODEL_REVISION`

### C5 [HIGH] Checkpoints — fail loud on key mismatch
Unknown, missing or mis-shaped keys **raise**. A silent partial load produces a
plausible-looking model with randomly initialized layers and a meaningless score.
→ `core/models/ckpt_compat.py`

### C6 [MEDIUM] Porting — don't inherit the baseline's bugs
`np.pad` with a scalar pad-width pads *every* axis; the baseline's collate did
that. Ported code gets read, not transcribed.
→ `core/data/collate.py:1-28`

### C8 [HIGH] Comparability — a protocol is unverified until the released ckpt reproduces the number
A published AUC is defined by label construction + sampling stride + score
pooling, not by the dataset name. LaGoVAD's DoTA 62.60 = normalized spans
rounded to feature length, `interval=8` features, **per-clip min-max** pooling
(`offline_dota_eval.py`) — *not* the raw pooling of the generic
`full_length_eval.py` harness. This lesson asserted the opposite for a week on
the strength of a code read; the reproduction (0.5055 raw vs 0.6142 min-max,
published 0.6260) settled it. When two baseline scripts disagree, run the
released checkpoint and let the number decide.
→ `core/data/dota.py:180`, `LaGoVAD-PreVAD/src/offline_evals/offline_dota_eval.py`, `core/docs/RESULTS_DOTA.md`

### C8b [HIGH] Comparability — gate against the released checkpoint, not the printed number
C8 has no exit condition. When the released checkpoint cannot reach the published
number either, the gap is in the artifacts, not your tree. MSAD: we logged the
gate as failing (0.8922 vs 0.9041) while `best.ckpt` itself scores 0.8991/0.8949;
eight paired bootstraps of our arms vs `best.ckpt` all include zero. Re-gate
against the checkpoint and document the residual gap (ours = 18,350 stride-8
frames, not the paper's full count).
→ `core/docs/RESULTS_PHASE_A.md` §4

### C9 [MEDIUM] Extraction — stream frame batches, never a whole clip
A 284-frame 720p clip is ~3 GB once `preprocess_frames` casts to float32. Stride
the *path list*, then decode→preprocess→encode `batch_size` images at a time.
→ `core/tools/extract_clip_features.py:encode_frame_dir`

### C10 [MEDIUM] Ingest — an existing directory is not evidence of data
A FUSE/Drive unzip leaves clip folders created but empty; folder counts then lie
about coverage. Count *files*, split unreadable clips by cause (absent vs.
empty), and make dropping any of them an explicit flag that logs `Coverage N/M`.
→ `core/data/dota.py:resolve_frame_counts`

### C11 [HIGH] Caches — skip-if-exists is not a resume unless writes are atomic
A run killed mid-`np.save` leaves a truncated `.npy` that `exists()` calls done,
so it is never redone and crashes eval later. Write to a `.part` sibling and
rename; verify existing outputs by reading the header; log `Resume: done/total`.
→ `core/tools/feature_cache.py`

### C12 [CRITICAL] Metrics — micro AUC over an all-abnormal test set measures the wrong thing
Pooling raw scores across videos puts each clip's *absolute* score scale into
the ranking. That is signal only when the test set has normal videos. On DoTA
(1,394/1,397 abnormal) a confident clip's negatives outrank a hesitant clip's
positives: released `best.ckpt` scored 0.5055 raw vs 0.6142 min-max, published
0.6260. Decide pooling from the **label distribution**, not the dataset name
(`--score-norm auto`: min-max when normals < 5 %), and always report
`auc_macro` beside the micro number. **Mirror case (2026-09-06):** when
all-normal clips carry most of the frames, micro rewards pure clip
classification — DADA-2000 is 74 % such frames and a constant-score-per-clip
oracle scores **0.9086** there. Compute that oracle and print it beside micro.
→ `core/metrics.py:resolve_score_norm`, `core/evaluate.py`, `core/tools/rescore.py`

### C13 [HIGH] Transform — a checkpoint is bound to the preprocessing that trained it
Feeding a checkpoint features built with a different transform raises nothing;
it just returns a worse number that reads as a modelling result. Worse
internally: `preprocess_for_raft` is full-frame while `preprocess_frames`
center-crops, so KIP regressed motion evidence cropped out of its own input.
One cache dir per transform; score each checkpoint only on its own; keep the
appearance and flow branches on the same field of view.
→ `core/tools/extract_clip_features.py:preprocess_frames`, `core/flow/raft_extract.py:preprocess_for_raft`

### C14 [CRITICAL] Experiments — an ablation run under a known-open precondition defect measures the defect
KIP's DoTA A/B was recorded at Δ = −0.0329 (CI excluding zero) while C13 was
already known: the appearance branch was cropped, the flow target was not, so
KIP regressed evidence absent from its own input. The same A/B after removing
the crop gave **+0.0911** — sign flipped, magnitude tripled, and only the KIP arm
moved (+0.1514 macro vs +0.0077 for KIP-off). A signed number with a CI outlives
its prose caveats. Report the A/B as *blocked*, with no number, until the
precondition is closed; the reference arm alone still has diagnostic value.
→ `core/docs/RESULTS_NCC.md` §2

### C15 [HIGH] Checkpoints — a pickled library object expires with its environment
`_rng_payload` puts `np.random.get_state()` (a numpy array) into every
checkpoint. A pickle is one stream, so after a Colab runtime drifted to a
different numpy major, `torch.load(weights_only=False)` died with
`TypeError: _reconstruct: First argument must be a sub-type of ndarray` **before
reaching any tensor** — every stage-1 and step checkpoint on Drive unloadable,
the weights themselves fine. Don't pin `numpy<2` (breaks the torch ABI);
recover by unpickling with a `find_class` that refuses `multiarray._reconstruct`
and re-saving `{"model": ...}` slim. Never pickle a library's internal state
object into an artifact meant to outlive its environment.
→ `core/train.py:389`, `core/train.py:408-425`, `core/inference.py:62`, `core/docs/COLAB.md` §A4.0

### C16 [MEDIUM] Experiments — step-uniform checkpoints undersample the loss range
`checkpoint_every_steps=100` over 500 steps is not "five samples across
training": `mil` falls 0.63 → 0.041 before the first one, so all five sit in the
last 6 % of the loss range and a convergence-matched probe has nothing to read.
Space checkpoints geometrically in *loss*, not uniformly in steps.
→ `core/docs/RESULTS_ARM4_PROBE.md` §4.2

### C17 [HIGH] Experiments — a run must record every flag that defines what it is
`cfg.save_yaml()` writes the config tree only, so `--init-weights`,
`--data-dir`, `--clip-dir`, `--knn-cache` vanish. A warm-started control arm's
`config.yaml` is byte-identical to the cold arm's — the one fact the arm exists
to establish is absent from its own output. Write a run manifest (resolved
init-weights + step + hash, data/cache paths, git commit, `sys.argv`).
→ `core/train.py:563-628`, `core/docs/RESULTS_ARM4_PROBE.md` §1

### C18 [CRITICAL] Data — a shipped label format is defined by its widest case
`dota.py` fills exactly one `[start, end)` because DoTA has one. PreVAD ships the
same normalized-fraction format with **104 multi-span test clips (up to 4
windows)**, **440 spans ending past 1.0** (max 1.2104) and **one reversed span**.
Taking the hull turns 40 positive rows into 66 on a real clip — no exception, no
warning, a plausible AUC over the wrong ground truth. Fill every span
independently, clamp to `[0, L]` (what the baseline's 512-long buffer does by
accident), and *count* every span the clamp absorbs.
→ `core/data/prevad.py:sampled_frame_labels`, `core/data/dota.py:232`

### C19 [HIGH] Definitions — a graceful name-lookup fallback hides a missing taxonomy
`_PREVAD_CLS_DEFS` held the 22 CamelCase names from the block that is *commented
out* in the baseline; the live `DEFAULT_CLASSES` below it is 36 space-separated
names. 28 of 35 observed classes had no definition — and nothing raised:
`verbalize_class_name` returns the bare class name by design, so the model would
train on `"Store Robbery"` instead of a definition and the conditioning claim is
void with no error. Check coverage in the preprocessor; don't make the fallback
strict.
→ `core/data/prevad.py:check_definition_coverage`, `core/data/definitions.py:194`

### C24 [CRITICAL] Architecture — a non-differentiable op on the score path freezes everything upstream
`s_t = (sigmoid(MLP(m)) * max_shift).floor().long()` used as a slice index gave
the 321-param gate MLP — and the PMG head behind it — **zero gradient, ever**.
Nothing raised: params exist, losses fall, ruff/mypy/pyright are clean, the model
reaches a publishable number. Init was therefore the deployed function, and it
was degenerate: input is min-max normalized so `[0,1]` is the whole domain, and
sweeping it moves `s_t` by **0–4 of 128 channels** (8 seeds; seed 0 = exactly 0).
Three campaigns called a fixed ~50 % smoother "motion-gated adaptive mixing".
The obvious STE fix `s = u + (u.floor()-u).detach()` is a **no-op** — `s` is only
read inside `channel < s`, and a comparison passes no gradient; apply the
estimator to the selection *weights* instead.
**On `main` (v1) the fix is not applied and there is no `gate_type` to switch:**
the frozen gate is the only gate, so every KIP-on run on this branch is a fixed
~50 % shift. The four-gate rebuild lives on branch `v3`.
→ `core/kip/gate_shift.py`, plan Appendix C
  (`core/tests/test_kip_gate_types.py` is **branch `v3` only**)

### C28 [CRITICAL] Data — a corpus can leak its label through clip length
The reconstructed DADA-2000 trims accident videos around the accident and keeps
normal-driving videos full length: abnormal clips are **max 17** stride-8 frames,
all **107** test clips with T >= 18 are normal. A detector reading only the frame
count (constant score = `-T`) scores **micro AUC 0.8654** — within 0.009 of the
best trained arm. Compute a length-only baseline **and** the constant-score-per-clip
oracle for every new corpus and print both beside any micro AUC; rebuild into
fixed-length windows if length is predictive. Distinct from C12 (metric artifact)
and C27 (receptive field). **Shipped:** `core.tools.eda` §3.3 + CRITICAL verdict;
`core.evaluate --equalize-length N --equalize-anchor {center,start,end}` is the control.
→ `core/data/dada.py:403`, `core/eda/protocol.py:clip_length_leak`, `core/evaluate.py:equalize_window`

### C29 [CRITICAL] Losses — DVS marks the *whole* anchor clip positive
`compose_sequence` sets `pseudo[anchor_span] = 1.0` across the entire anchor and
`supervised_loss` consumes it as a dense per-frame BCE. Correct only if an
abnormal clip is ~entirely anomalous. On DADA the mean positive fraction is
**0.351**, so **64.9 %** of anchor frames are trained to 1 against a 0
annotation — and **no term in the objective pushes any frame of an abnormal clip
down**. Result: normal frames inside abnormal clips score 0.4078 vs positives
0.4076 (gap −0.0002), a +0.359 clip offset, `auc_macro` 0.53, argmax
localization below base rate. Measure the corpus's positive fraction before
enabling DVS; make the anchor interior an *ignore* target when it is not ≈1.
**Shipped 2026-09-09 as arms, both default-off:** `loss.dvs_anchor_mode=ignore`
and `loss.bottomk_weight>0` (`abnormal_bottomk_loss`). A bad mode string raises.
**Trained 2026-09-12 and both FAILED their pre-registered predictions** — the gap
shrank, `auc_macro` fell, the score scale fell with it. The lesson stands; it is
just not the bottleneck (C27 is). `RESULTS_DADA_PHASE1.md`.
→ `core/data/synthesis.py:83`, `core/losses/dvs.py:16`, `core/losses/mil.py:47`, `core/train.py:295-302`

### C30 [HIGH] Experiments — an eval-time sampler seeded once per run leaks across items
`core/evaluate.py:159-164` builds the verbalizer once above the scoring loop and
samples a definition **per window**, so every clip shares one RNG stream. Any
filter that skips an item (`--equalize-length`'s `continue`, a future `--strict`)
shifts the stream for every later clip: 32 of the 34 `T == 5` DADA clips — where
the eq5 crop is the identity and the input is byte-identical — returned different
scores, max |Δ| 0.0033, worth **±0.003 AUC**. Seed per item
(`random.Random(SEED + i)`), never per run. Within-protocol comparisons stay exact.
→ `core/evaluate.py:159-164`, `core/docs/RESULTS_DADA_PHASE1.md` §5

### C31 [MEDIUM] Metrics — a loss that lowers scores is not a loss that creates contrast
Phase 1's pre-registered criteria ("the gap turns positive", "the range widens")
were scale-dependent, and both losses lowered the scale: mean positive score
0.1160 → 0.0771/0.0953 → 0.0635, gap +0.0085 → +0.0013. Scale-invariantly,
`d = gap/σ` collapses **+0.164 → +0.039** — which agrees with `auc_macro` while
the raw gap is ambiguous. Report `gap / mean within-clip σ`, with the raw scale
printed beside it.
→ `core/docs/RESULTS_DADA_PHASE1.md` §4

### C32 [CRITICAL] Data — size a fixed-length re-shard against the SHORTEST class
Re-sharding into equal-length windows is C28's fix and it has two failure modes
that both pass the C28 check. DADA's accident clips are trimmed (raw median **49**
frames), so `--window-length 32` at stride 2 needed 64 raw frames and kept
**25.3 %** of abnormal clips — ~69 % of the abnormal supervision deleted, with no
error. And a fixed hop gives windows in proportion to clip length, so 3x-longer
normal clips produced **327 abnormal vs 3,244 normal** windows from a ~1:1 corpus.
The leak closed perfectly (length AUC 0.5000) while the clip oracle rose
0.9086 → **0.9766** and `auc_macro`'s population fell 190 → **57**. Size from the
shortest class's p5–p25, cap windows per clip, gate on retention.
→ `core/data/dada.py:plan_record_windows` (warns below 90 % retention),
  `core/data/windows.py:cap_windows`, `core/docs/DADA_SETUP.md` §10.3

### C33 [MEDIUM] Experiments — a pre-registered threshold must be reachable
Gate W's "clip oracle < 0.75" was unsatisfiable: the oracle is
`(F_norm + 0.5X)/(F_norm + X)` (reproduces 0.9086 and 0.9766 exactly), so < 0.75
needs abnormal clips to hold **≥ 62 %** of all test frames; balanced is 0.811.
Derive a metric's attainable range from the corpus's class mix *before* writing
the threshold, and record the derivation beside it.
→ `.project/plans/katvad-dada-phase2-corpus-rebuild.md` §1

### C35 [HIGH] Experiments — a pre-registered threshold needs a MEASURED lever
C33 fixed *whether* a threshold can be met; it says nothing about **which flag
moves it**. Phase 2's risk 3 named `--window-max-per-clip` as the knob for a
failing clip oracle and warned against `--window-length`. Measured: the cap spans
**0.7527-0.7765** across caps 2-6 (it thins both classes at once, so C33's
`(F_norm+0.5X)/(F_norm+X)` barely shifts); the window length spans **0.61-0.75**
and is the only flag that crosses the bar. W=16 FAILED at 0.7529, **W=20/hop 8
PASSES all six** (0.7037, retention 0.957, 798 two-class). The plan's predicted
0.6631 belongs to W~24, which fails retention instead. Publish a measured lever
table beside every threshold; a windowing rebuild costs seconds because the
feature cache is keyed by SOURCE clip and a window is a slice.
→ `.project/plans/katvad-dada-original-phase2-t2.md` §6.1, §7 risk 3;
  `core/constants.py:DADA_ORIGIN_WINDOW_LENGTH`

### C38 [CRITICAL] Data — admit an imported normal pool by transfer probe, not similarity
Separate-pool failures: 0_Normal_Driving C28/C32 · TAD C14 · D2City · BDD-A. D2City as the
only negatives: within-DADA `auc_macro` **0.5864** vs in-video **0.6763**, Δ **−0.090**
(5/5 folds), shortcut AUC **1.000**; added beside in-video negatives still −0.012. DADA vs
DoTA separates at 0.9999 on frozen CLIP — crops cannot remove a source cue. Gate = paired
probe with the pool as SOLE negatives, within 0.03 of R0; default = negatives cut from
inside the positive videos (T2). **Fourth failure 2026-09-26, BDD-A (US, calm arm):** X
0.5492, Δ **−0.127**, shortcut 1.000 — even though R0 had ranked BDD-A *above* DADA
normals (0.374). A separable pool ends at shortcut 1.0 whichever side it starts on.
→ `core/docs/D2CITY_EDA.md`, `core/docs/BDDA_EDA.md`, `colab/{D2City,BDDA}/eda_normal_bags.ipynb` §4

### C7 [MEDIUM] Deps — don't call library internals that drift
The LR schedule is an in-house `LambdaLR` rather than
`transformers.get_scheduler`, to avoid HF internal-API churn across versions.
→ `core/train.py`, `core/docs/TRAINING.md` §3

## Trigger map

| If you are about to… | Read |
|---|---|
| touch feature/flow extraction, stride, or any cache path | C2, C13, `core/docs/DATA_LAYOUT.md` |
| compare a metric to a published number, or add a dataset adapter | C8, **C8b**, C12, **C18**, **C19**, `core/docs/DOTA_EVAL.md` |
| build frame labels from an annotation's anomaly spans | **C18** |
| add or edit a dataset's class list / definition sentences | **C19** |
| declare a reproduction gate passed or failed | **C8b** |
| compute AUC/AP, or evaluate on a new benchmark | C12, **C22**, `core/docs/RESULTS_DOTA.md` |
| normalize/pool score curves, or compare `evaluate` against `rescore` | **C22** |
| recompute a metric offline that a tool already reported | **C22b** |
| join `scores/*.npz` to metadata by id | **C23** |
| score a checkpoint you did not train, or change an image transform | C13, **C34** |
| evaluate an ablation arm, or write an eval command for a ladder of arms | **C34** — the architecture comes from the checkpoint; a `temporal_window` mismatch does NOT raise |
| see `size mismatch for ...` from `load_state_dict` | **C34** — and check which sibling fields are mask-only, they failed silently |
| run, record, or act on an ablation / A/B delta | C14 |
| launch a training run with `--init-weights` or any non-default path | **C17** |
| set `checkpoint_every_steps`, or plot a metric against train loss | C16 |
| write an extractor that reads frame folders | C9, **C25** |
| add a second source path (frames vs videos) to an existing cache | **C25** |
| use a frame folder's bare name as `video_id`, or point `--frames-dir` at a multi-category root | **C26** |
| hand a numpy array straight to a torch model | **C25** |
| add or resume a long-running per-item cache | C11 |
| write **any** artifact file (scores, metrics, checkpoints), not just a resumable cache | **C11b**, **C11c** |
| call `torch.save` / `np.save` / `np.savez` onto a path a later run will read | **C11c** — stage to `.part`, `fsync`, `replace`; a writer is not fixed until a test kills it mid-write |
| join `scores/*.npz` counts against `results.json`, or compute a Δ across arms | **C11b**, **C23** |
| ingest an unzipped dataset, or drop items from an eval set | C10, **C20** |
| write a shell cell that touches every file in a dataset | **C20** |
| write `floor`/`round`/`argsort`/`argmax`/`topk`/`.long()` anywhere a gradient must pass | **C24** |
| add a module you will describe as *learned*, *adaptive*, or *gated* | **C24** |
| implement or review a straight-through estimator | **C24** |
| set or change `kip.gate_type`, or read `s_t` | **C24**, `core/docs/TRAINING.md` (the KIP gate) — and note `gate_type` **does not exist on `main`** |
| copy a command out of any `*_SETUP.md` runbook | check the flag exists in `core/config.py` **on this branch** first — the v3 arm ladders fail at parse on `main` (`KeyError: kip.gate_type`) |
| wonder why 12 tests fail, or plan to "fix" `core/config.py` | [[activeContext]] branch section — it is the v3 gate matrix on a v1 tree, not a regression |
| merge, rebase or cherry-pick between `main` and `v3` | [[activeContext]] — the memory bank and `CLAUDE.md` diverged **on purpose**; resolve by branch identity |
| train on a **new corpus**, or change `frame_stride` / `score_head_kernel` / `temporal_window` / MIL top-k | **C27**, C2, `core/docs/v3/RESULTS_DADA.md` — and run `core.tools.eda` first (`core/docs/EDA.md`) |
| plan a campaign on a corpus nobody has profiled, or wonder whether a benchmark can support a frame-level claim at all | `core/docs/EDA.md` §1, §3; **C27**, **C12** |
| ask whether a null result is the representation's fault or the supervision's | `core/docs/EDA.md` §3 (the linear probe), `RESULTS_DADA.md` §10-B |
| read a micro AUC as a localization result, or compare one to a published frame-level number | **C27**, C12, C8b |
| train on a **new corpus**, or accept a corpus someone else reconstructed/trimmed | **C28**, C27, C12 — compute the length-only baseline first |
| enable DVS (`theta`, `delta_m`) on a corpus whose abnormal clips are not trimmed to the anomaly | **C29** — and consider `loss.dvs_anchor_mode=ignore` |
| add a parameter to a loss that `test_baseline_parity` asserts | make it **keyword-only with a baseline default**, or the LaGoVAD parity assertion breaks (C6, C29) |
| wonder why `auc_macro` sits at chance while micro AUC looks strong | **C28**, **C29**, C27, C12, `core/docs/DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` |
| argue that the frozen-CLIP backbone is the bottleneck | `core/docs/DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` §7 — run the probe A/B before spending; a VideoMAE swap voids `H_mul` and every cache (C2, C13) |
| re-shard a corpus into fixed-length windows, or pick a `--window-length` | **C32** — measure the length distribution PER CLASS first |
| accept a rebuilt corpus, or read an EDA report after a corpus change | **C32** — check retention, two-class count and class ratio, not just the leak metric |
| write a pre-registered gate, exit criterion or falsification threshold | **C33** — derive the attainable range first; **C35** — and measure which flag moves it |
| write "if the gate fails, change X" in a plan, runbook or risk section | **C35** — sweep X against the real data first, or do not write the sentence |
| act on a failing Gate W criterion, or argue a miss is "only a bit over" | **C35** — the trade-off table is in the plan's §6.1; a passing geometry may be one row away |
| pick `--window-length` / `--window-max-per-clip` for the DADA-2000 ORIGINAL corpus | **C35**, C32 — W=20/hop 8 passes, W=16 does not; `core/constants.py:DADA_ORIGIN_WINDOW_LENGTH` |
| set `loss.mil_topk_pct` on a corpus of short windows | `k = max(1, L // pct)`; at L=20 the default 16 gives **k=1** and `L_MIL` is a plain max — `core/constants.py:DADA_ORIGIN_MIL_TOPK_PCT` |
| add any eval-time flag that skips, filters or subsets the scored items | **C30** — seed per item first, or the numbers are not comparable |
| compare a metric across two different scored subsets (raw vs `--equalize-length`, with/without an exclusion) | **C30** |
| read or report a positive/negative score gap, a score range, or "the curves widened" | **C31** — divide by the within-clip σ |
| pre-register a success criterion for a loss arm | **C31** (make it scale-invariant), C14 |
| judge the DADA Phase 1 arms, or wonder whether C29 was refuted | `core/docs/RESULTS_DADA_PHASE1.md` §7 — falsified, and C29 still stands |
| add normal/negative bags from ANOTHER dataset (D2City, BDD-A, BDD100K, `0_Normal_Driving`, …) | **C38** — run the G-X transfer probe first (pool as sole negatives vs in-video R0, paired folds); "same camera/country/fps" is not evidence. Default: T2 in-video negatives |
| read a source-separability probe (S) as an admission verdict | **C38** — it is a claim gate; frozen CLIP separates every dashcam corpus from DADA (DoTA 0.9999) |
| add or change a loss | C7, **C37**, `core/docs/TRAINING.md` (deviations) |
| set a loss WEIGHT, or read a raw loss value as "the head fits badly" | **C37** — measure the constant-predictor MSE on its target first; `R² = 1 − loss/V`, weight = `1/V`, never a sweep (lesson 14) |
| add an auxiliary REGRESSION head (flow, depth, pose, reconstruction) beside classification losses | **C37** — they are not on a comparable scale by default, and the mismatch reaches every parameter they share |
| wonder whether an auxiliary term is stealing the trunk | **C37** — `python -m core.tools.grad_probe`: `rho ≥ 1` = capture, `|cos| < 0.1` = orthogonal (fix = normalization), `cos ≤ −0.1` = conflict (fix = the detach control) |
| change `lambda_rec`, or read `kip_rec` in `metrics.jsonl` | **C37**, C24 — and note the D1/D2 record is `outputs/v1/DADA2000_orig_diag_kip_loss_scale/`, not a `RESULTS_*.md` |
| pass `--flow-dir`, or switch between `flow/v1` and `flow/v2_zscore` | **C2**, **C37** — `lambda_rec` belongs to the CACHE: v1 → 1.0, v2 → `1/V_v2` from `zscore_manifest.json` (≈1). Never carry v1's 0.0316 onto v2; `--flow-dir` is not in `config.yaml`, so record it in the run manifest |
| load or map a checkpoint | C5, C15 |
| load weights into a **tool that does not train** (probe, diagnostic, rescore, eval) | **C36** — `warm_start_model`, never `Trainer.load_checkpoint`; the resume path restores an optimizer/RNG a read-only tool must not touch |
| probe, score or warm-start from a **stage-1** checkpoint | **C36** — it has no `clip_text_model.*` by design (`load_clip=False`); a strict load refuses it |
| add a CLI flag that takes a checkpoint path | **C36** — write the test that actually passes one, or the only crashing branch stays uncovered |
| add a field to a saved checkpoint, or hit `torch.load` failing on old artifacts | **C15** |
| set up or debug a Colab/notebook environment, or pin a dependency | **C21** |
| run training locally on a Mac | C3 |
| use faiss / build the KNN cache | C1 |
| port anything from `LaGoVAD-PreVAD/` | C6 |
| add a new HF or torchvision weight | C4 |

## Category map

- **Experiment discipline:** C14, C16, C17, **C30**, **C33**, **C35**
- **Env / platform:** C1, C3, **C21**
- **Data & caches:** C2, C9, C10, C11, C13, **C18**, **C20**, **C23**, **C25**, **C26**, **C28**, **C32**, **C38**
- **Comparability / protocol:** C8, C8b, C12, C13
- **Metrics:** C12, **C22**, **C27**, **C28**, **C31**
- **Supply chain:** C4
- **Model loading:** C5, C15, **C36**
- **Porting discipline:** C6, C7, **C19**, **C29**
- **Loss scale / objective balance:** **C37**, C31, C24
- **Architecture / gradient flow:** **C24**, **C27**

Full catalog: `index.md`. Candidates awaiting validation: `pending.md`.
