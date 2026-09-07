# Lessons — Tier 0 meta-index

**Re-initialized 2026-07-31 at commit `b9978ff`.** The previous catalog was
discarded on the user's instruction; every lesson below was re-derived by
reading this tree, with a file:line citation. Load this file before every
IMPLEMENTATION or DEBUG task (~400 tokens).

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
oracle scores **0.9069** there. Compute that oracle and print it beside micro.
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
→ `core/kip/gate_shift.py`, `core/tests/test_kip_gate_types.py`, plan Appendix C

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
| score a checkpoint you did not train, or change an image transform | C13 |
| run, record, or act on an ablation / A/B delta | C14 |
| launch a training run with `--init-weights` or any non-default path | **C17** |
| set `checkpoint_every_steps`, or plot a metric against train loss | C16 |
| write an extractor that reads frame folders | C9, **C25** |
| add a second source path (frames vs videos) to an existing cache | **C25** |
| use a frame folder's bare name as `video_id`, or point `--frames-dir` at a multi-category root | **C26** |
| hand a numpy array straight to a torch model | **C25** |
| add or resume a long-running per-item cache | C11 |
| write **any** artifact file (scores, metrics, checkpoints), not just a resumable cache | **C11b** |
| join `scores/*.npz` counts against `results.json`, or compute a Δ across arms | **C11b**, **C23** |
| ingest an unzipped dataset, or drop items from an eval set | C10, **C20** |
| write a shell cell that touches every file in a dataset | **C20** |
| write `floor`/`round`/`argsort`/`argmax`/`topk`/`.long()` anywhere a gradient must pass | **C24** |
| add a module you will describe as *learned*, *adaptive*, or *gated* | **C24** |
| implement or review a straight-through estimator | **C24** |
| set or change `kip.gate_type`, or read `s_t` | **C24**, `core/docs/TRAINING.md` (the KIP gate) |
| train on a **new corpus**, or change `frame_stride` / `score_head_kernel` / `temporal_window` / MIL top-k | **C27**, C2, `core/docs/v3/RESULTS_DADA.md` — and run `core.tools.eda` first (`core/docs/EDA.md`) |
| plan a campaign on a corpus nobody has profiled, or wonder whether a benchmark can support a frame-level claim at all | `core/docs/EDA.md` §1, §3; **C27**, **C12** |
| ask whether a null result is the representation's fault or the supervision's | `core/docs/EDA.md` §3 (the linear probe), `RESULTS_DADA.md` §10-B |
| read a micro AUC as a localization result, or compare one to a published frame-level number | **C27**, C12, C8b |
| add or change a loss | C7, `core/docs/TRAINING.md` (deviations) |
| load or map a checkpoint | C5, C15 |
| add a field to a saved checkpoint, or hit `torch.load` failing on old artifacts | **C15** |
| set up or debug a Colab/notebook environment, or pin a dependency | **C21** |
| run training locally on a Mac | C3 |
| use faiss / build the KNN cache | C1 |
| port anything from `LaGoVAD-PreVAD/` | C6 |
| add a new HF or torchvision weight | C4 |

## Category map

- **Experiment discipline:** C14, C16, C17
- **Env / platform:** C1, C3, **C21**
- **Data & caches:** C2, C9, C10, C11, C13, **C18**, **C20**, **C23**, **C25**, **C26**
- **Comparability / protocol:** C8, C8b, C12, C13
- **Metrics:** C12, **C22**, **C27**
- **Supply chain:** C4
- **Model loading:** C5, C15
- **Porting discipline:** C6, C7, **C19**
- **Architecture / gradient flow:** **C24**, **C27**

Full catalog: `index.md`. Candidates awaiting validation: `pending.md`.
