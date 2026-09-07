# Lessons — Tier 2 detail

Root cause + full context per lesson. Numbering matches `index.md`.

---

## 1. faiss segfaults multi-threaded next to torch (macOS arm64)

**Symptom:** the KNN-cache builder dies with a bare segfault inside
`IndexFlat.search`, no Python traceback, exit code 139.

**Root cause:** `torch` and `faiss-cpu` each ship their own `libomp.dylib`. On
macOS arm64 both get loaded into the process; faiss's OpenMP thread pool then
runs against a runtime it did not initialize and crashes on the first parallel
search.

**Fix:** `faiss.omp_set_num_threads(1)` at module import, before any index
exists. Index sizes here are a few thousand 512-d vectors — single-threaded
search is not a bottleneck.

**Generalization:** any two wheels that each vendor an OpenMP runtime can do
this. Suspect it whenever a native library segfaults only *after* torch is
imported.

---

## 2. A feature cache is valid only for the transform and stride that built it

**Context:** the pipeline caches CLIP features (`cache/clip/`), RAFT flow
embeddings (`cache/flow/v1/`) and a KNN filler cache (`cache/knn/`). All three
are derived artifacts of one exact preprocessing choice: resize policy, crop,
normalization constants, and frame stride.

**Root cause of the danger:** nothing in the cache path encodes the transform,
so a changed transform reads back happily and silently shifts the feature
distribution. Downstream every metric moves, and the move looks like a modeling
result.

**Live example in this repo:** `preprocess_frames` resizes the shorter side to
224 and center-crops. The baseline's own extraction script instead passes
`augmentation='no_center_crop'`, i.e. an anisotropic `Resize((224,224))`
(`LaGoVAD-PreVAD/tools/extract_feat_clip.py:39`). These are genuinely different
transforms. **The MSAD result that matched the paper was produced with the
center-crop version in this tree** — so the discrepancy is recorded as an open
question, not silently "fixed". See `activeContext.md`.

**Resolution, 2026-08-08:** the project is moving to `no_center_crop`
everywhere — see lesson 13 for why (it is not only about baseline parity; the
flow branch was already full-frame, so the two branches disagreed). The switch
is a re-extract-and-re-measure of every artifact, into *separate* cache and
output paths, exactly as this lesson prescribes. `--no-center-crop` on
`core.tools.extract_clip_features` selects it; the default is unchanged so
existing caches stay valid.

**Rule in practice:** treat a transform change as a cache-version bump
(`cache/flow/v1/` already does this), re-extract, and re-measure every gate.

---

## 3. torch 2.4 MPS training diverges from CPU

**Symptom:** stage-1 KIP warm-up on Apple Silicon shows `L_KIP_rec` *increasing*
epoch over epoch; the identical seed and data on CPU converge normally.

**Root cause:** not isolated to a single op. Treated as a backend-level defect in
torch 2.4's MPS path for this graph (conv1d + gather-heavy shift + masked
reductions). Not worth bisecting — real training runs on CUDA.

**Fix:** `train.device=cpu` locally. `core/tests/test_e2e_synthetic.py` pins CPU
so the suite cannot accidentally validate on a broken backend.

---

## 4. Pin HF model revisions

**Symptom:** `from_pretrained` raises a `ValueError` about `torch.load` being
unsafe on torch < 2.6.

**Root cause:** transformers 4.56 refuses `.bin` weights on torch < 2.6 because
of CVE-2025-32434 (`torch.load` arbitrary code execution). The `main` revision of
`openai/clip-vit-base-patch16` ships only `pytorch_model.bin`. The pinned sha
`5ef227a7…` is the SFconvertbot safetensors conversion — identical weights.

**Second reason to pin:** reproducibility and supply-chain safety. An unpinned
`main` can change under you between two runs of the same experiment.

---

## 5. Fail loud on checkpoint key mismatch

**Context:** LaGoVAD's `best.ckpt` uses different key prefixes than `KATVAD`
(`temporal_encoder.*` → `temporal_encoder.encoder.*`, top-level `gate_alpha` →
`temporal_encoder.gate_alpha`), the frozen CLIP body is skipped (it comes from
HF), and `kip.*` legitimately stays at initialization.

**Root cause of the danger:** `strict=False` makes all four cases
indistinguishable. A typo'd prefix then leaves a whole submodule random, and the
model still emits scores in the right range — the failure surfaces as a bad
number, not as an error.

**Fix:** explicit mapping, and raise on unknown / missing / mis-shaped keys. The
only intentional exceptions are whitelisted by name.

---

## 6. Read ported code, don't transcribe it

`np.pad(array, 3)` pads *every* axis by 3 — on an `(L, 512)` feature array that
silently widens the channel dimension to 518. The baseline's collate did this.
The port pads axis 0 only and has a test asserting the channel dim is untouched.

**Generalization:** the reuse map in CLAUDE.md §14.4 says "organize, don't copy"
for exactly this reason. Every ported function needs a test that states what it
is supposed to do.

---

## 7. Don't depend on library internals that drift

`transformers.get_scheduler` is a convenience wrapper whose warmup/decay
semantics have changed across minor versions. A silently different LR curve is
one of the hardest training bugs to spot, because nothing errors and the loss
still goes down — just to a worse place.

The replacement is ~10 lines of `LambdaLR` closure with a unit test asserting the
curve's shape at boundaries (step 0, end of warmup, final step).

---

## 8. Reproduce the baseline's label arithmetic before quoting its number

Setting up the DoTA eval, three protocol facts had to be recovered from the
baseline's source before any number could be called comparable:

1. **Label construction.** `dota_test_anno.json` stores `anomaly_span` as
   normalized `[start/num_frames, end/num_frames]`;
   `LaGoVAD-PreVAD/src/datasets/base.py:57-69` rebuilds frame labels as
   `round(frac * feature_length)` filled half-open. Building labels from raw
   frame indices instead gives a *different* label vector at any stride > 1 —
   at stride 8 on DoTA val, 6,118 positives (their rule) vs 5,887 (raw
   indexing), and 3 vs 5 clips whose anomaly window rounds away entirely.
2. **Sampling stride.** `tools/extract_feat_clip.py:39` uses `interval=8`. The
   published number is measured on stride-8 features, so a "native fps" eval is
   a different experiment, not a better one.
3. **Pooling protocol.** `src/full_length_eval.py:216` updates the AUROC metric
   with **raw** sigmoid scores; `src/offline_evals/offline_dota_eval.py`
   min-max-normalizes each clip first. On a dataset where every clip contains
   both classes, the second is systematically higher. Two numbers, one dataset.

The check that closed it: derive the spans from `metadata_val.json` and assert
they equal the baseline's shipped `anomaly_span` for all 1,402 val clips. They
do, to floating-point equality — so the ground truth is provably the same one.

### Correction, 2026-08-08 — fact 3 was recorded backwards

This lesson originally concluded that the published 62.60 came from
`full_length_eval.py`'s **raw** pooling, and `core/evaluate.py` was written to
match. That was inference from reading code, never a reproduction, and it was
wrong. Re-scoring the three DoTA arms measured:

| Arm | raw | per-clip min-max | per-clip z-score | macro per-clip |
|---|---|---|---|---|
| `gate_a` (LaGoVAD released `best.ckpt`) | 0.5055 | **0.6142** | 0.6236 | 0.6328 |

Raw pooling puts the baseline's *own released checkpoint* at chance against its
published 0.6260. `offline_dota_eval.py` exists precisely because DoTA needs the
normalized protocol; `full_length_eval.py` is the generic multi-dataset harness.
The cost of the error: three eval arms reported at chance, and a KIP result that
looked like a modelling failure when the metric was the failure. See lesson 12
for why the all-abnormal label distribution — not the dataset name — is what
determines this.

**Generalization:** "we evaluate on dataset X" is not a protocol. Label
arithmetic, sampling rate and score pooling each move the metric by more than
most claimed improvements — and a protocol read off source code is a hypothesis
until the released checkpoint reproduces the published number under it.

---

## 9. Stream frame batches; never materialize a whole clip

`encode_video` reads a whole video, then preprocesses it, then encodes in
batches. That is fine for MSAD at stride 8 (~78 frames of already-downscaled
video). It does not survive a frame-folder dataset at native rate: a 284-frame
720p DoTA clip is 785 MB as uint8 and ~3 GB after `preprocess_frames` casts to
float32 and upsamples — before a single image reaches the GPU.

`encode_frame_dir` strides the *path list* first, then decodes → preprocesses →
encodes `batch_size` images per iteration, capping resident memory at the batch.
A test asserts batch size 2 and batch size 64 give identical features (atol
1e-5), so the streaming is a pure memory optimization and not a numerical change.

**Generalization:** any extractor whose input size is set by the dataset rather
than by a config value should stream. The failure mode is an OOM 40 minutes into
a Colab run, on the longest clip in the set.

---

## 12. Micro AUC over an all-abnormal test set measures the wrong thing

**Context:** `core/evaluate.py` concatenated every video's scores into one
array and called `roc_auc_score` once — the baseline's torchmetrics
accumulation semantics, and correct on MSAD.

**Root cause:** a micro AUC ranks *all* frames against each other, so each
clip's absolute score scale is part of the metric. That scale is meaningful
only when the test set contains normal videos, where "this whole clip is calm"
is a true statement the metric should reward. On DoTA every clip contains an
anomaly (1,394 of 1,397; ~33 % of frames positive), so the only question is
*when* within each clip — and the between-clip scale contributes pure noise.
Concretely, a clip the model is confident about contributes all of its frames,
positives **and negatives**, above a clip it is unsure about.

**How it was caught:** the reference arm. LaGoVAD's released `best.ckpt`
scored 0.5055 — chance — where the paper reports 62.60. A released checkpoint
cannot be at chance, so the fault had to be downstream of the model. Re-scoring
the saved `.npz` files under different pooling rules:

| Arm | raw | min-max | z-score | macro per-clip |
|---|---|---|---|---|
| `gate_a` (released `best.ckpt`) | 0.5055 | **0.6142** | 0.6236 | 0.6328 |
| `eval_kip_off` (MSAD-trained) | 0.5243 | 0.5539 | 0.5575 | 0.5561 |
| `eval_kip_on` (MSAD-trained) | 0.5126 | 0.5215 | 0.5215 | 0.5232 |

Eleven AUC points on the reference arm, from the pooling rule alone.

**The fix, and why it is label-driven:** `--score-norm auto` resolves from the
*label distribution*, not the dataset name — min-max when normal videos are
under 5 % of the test set. A dataset-name lookup would have to be remembered
for every new benchmark; the label distribution is already in front of the
metric. The 5 % threshold is not `== 0`: DoTA has 3 normal clips out of 1,397
(anomaly windows that round away at stride 8), and an all-or-nothing rule let
those 3 silently restore the broken protocol — which is exactly what the first
implementation did, and the rescore run caught.

**Report all three.** `results.json` carries `auc` (under the resolved rule),
`auc_raw`, and `auc_macro`. The macro number needs no normalization at all —
each video is ranked only against itself — so it is the honest localization
metric on an all-abnormal set, and a useful cross-check that a normalization
scheme is not doing something clever.

**Generalization:** before pooling scores across videos, ask what the between-
video scale means for *this* test set. If every video has the same label, it
means nothing, and pooling it in is a bug that reads as a null result.

---

## 13. A checkpoint is bound to the preprocessing that built its training features

**Context:** three DoTA arms were scored on one CLIP cache built with
center-crop. One of them, `gate_a`, is LaGoVAD's released checkpoint — trained
by them on `no_center_crop` features.

**Root cause:** a transform mismatch raises nothing. The features have the right
shape and dtype, the model runs, and a plausible-looking number comes out. The
only symptom is that the number is worse, which is indistinguishable from a
modelling result unless a reference arm exists to contradict it.

**The internal case is worse.** `core/flow/raft_extract.py:preprocess_for_raft`
resizes anisotropically to 240×320 — **full frame**. `preprocess_frames`
center-crops, discarding roughly the left and right quarters of a 16:9 frame.
So `L_KIP_rec` was training the PMG-flow head to regress motion evidence that
had been cropped out of its own input. On DoTA that is not a marginal concern:
349 of 1,402 clips are `ego: lateral` or `other: lateral`, where the kinematic
evidence lives exactly in the discarded strips. KIP may have been handicapped by
construction, independent of any baseline-parity argument.

**Rule in practice:**

- One cache directory per transform: `clip/DoTA_s8` vs `clip/DoTA_ncc_s8`.
- Score a checkpoint only on features built with the transform it was trained
  under — this is per-arm, not a global switch, and getting it "consistent" by
  switching everything can *introduce* mismatches on arms that were fine.
- Choose the appearance and flow transforms together; a model whose branches
  disagree about the field of view is inconsistent with itself.

**Generalization:** preprocessing is part of the model, not part of the data
loader. Any artifact derived from it inherits the binding.

## 14. An ablation run under a known-open precondition defect measures the defect

**What happened.** On 2026-08-08 the DoTA arms were scored and KIP-on came out
below KIP-off: Δ = −0.0329 mean per-clip AUC, 95 % CI [−0.0513, −0.0138]. The
CI excluded zero, so it was recorded in `RESULTS_DOTA.md` and `activeContext.md`
as "a real negative measurement" — hedged in prose, but with a signed number and
an interval attached.

At that same moment lesson 13 was **already written and already known**: the
appearance branch used `preprocess_frames` (shorter-side resize + center crop)
while the flow target used `preprocess_for_raft` (full frame). KIP's job is to
regress the flow embedding from the CLIP features. It was therefore being asked
to predict motion evidence that had been cropped out of its own input, on a
dashcam benchmark where 349 of 1,402 clips are labelled `lateral`.

**What the re-run showed.** Identical A/B, identical seed, hyperparameters, data
split and flow cache — only the image transform changed:

| arm | crop | no_center_crop | Δ macro AUC | 95 % CI |
|---|---|---|---|---|
| KIP-on | 0.5232 | 0.6746 | **+0.1514** | [+0.1354, +0.1683] |
| KIP-off | 0.5561 | 0.5638 | +0.0077 | [−0.0065, +0.0225] |
| `gate_a` | 0.6328 | 0.6158 | −0.0170 | [−0.0301, −0.0037] |

Δ(on − off) went from **−0.0329 to +0.0911**. Only the arm consuming motion
moved; the two baseline arms did not. The recorded negative was the crop.

**Why the hedging did not save it.** The prose caveat was correct and specific
("both checkpoints are `checkpoint_last`… the features cropped away the evidence
KIP exists to use"). It still did not stop −0.0329 from propagating into the
memory bank and the next planning cycle as the working belief about KIP. A
signed number with a confidence interval outlives its caveats; a stated *absence*
of a number does not.

**The asymmetry that makes this a rule.** Running the A/B early is nearly free
in compute and expensive in belief. Declining to record a number costs nothing
and can be reversed in an afternoon. When a precondition defect on the module's
own inputs is open, the reference arm still has diagnostic value — it is what
caught the pooling bug (lesson 12) — but the treatment-vs-control delta has
none.

**Rule.** Fix every open defect on a module's own inputs before running or
recording its ablation, and treat a signed delta measured under one as
unmeasured — report it as blocked, not as negative.

## 18. A shipped label format is defined by its widest case, not its common case

**Where it came from.** Writing `core/data/prevad.py` (2026-08-24). The obvious
move was to copy `core/data/dota.py:sampled_frame_labels`, which is correct,
tested, and reproduces the published DoTA number. It is also single-span,
because DoTA is: `metadata_val.json` gives one `anomaly_start`/`anomaly_end`
per clip, so `DotaRecord.span` is a single `tuple[float, float]`.

PreVAD stores the *same* normalized-fraction convention and reads as a drop-in.
It is not one:

| quirk | count in the shipped `test.csv` | worst case |
|---|---|---|
| clips with 2+ anomaly windows | **104** of 1,306 abnormal | 4 windows |
| spans ending past 1.0 | **440** | 1.2104 |
| spans with `end <= start` | **1** (`RrUW8ITUqx0_aug3`) | 0.9814 → 0.7110 |

**What the single-span assumption costs.** On the real 4-span clip
`qhly283_BV1Lf4y117wS` (L = 100 rows):

```
spans  [0.040, 0.126] [0.193, 0.285] [0.396, 0.466] [0.536, 0.690]
correct   4..13         19..28         40..47         54..69     = 40 positives
hull      4..............................................69      = 66 positives
```

26 rows that are normal in the ground truth become positive. No exception, no
length mismatch, no warning — `FeatureEvalDataset` accepts it, the AUC comes out
plausible, and it is measuring a different task.

**Why the bounds matter too.** The baseline gets clamping for free and by
accident: `LaGoVAD-PreVAD/src/datasets/base.py` fills into a `vis_max_len`-long
(512) zero buffer, so `frame_label[start:end] = 1.0` with `end > L` writes into
padding that evaluation later slices off, and a reversed span writes nothing at
all. Our label vectors are exactly `L` long, so the same behaviour has to be
written explicitly: `max(0, min(length, round(frac * length)))`.

That asymmetry is the trap. The released ground truth is *defined* by what the
baseline's buffer slicing produces. "Fixing" the 440 overflowing spans — by
rescaling, by dropping the clip, by refusing to parse — would diverge from the
labels the published number was measured against. Reproducing a protocol means
reproducing its accidents (lesson 8/8b).

**So the rule is two-sided:** absorb the quirk exactly as the baseline does,
and *count* it. `build_frame_labels` logs the overflow count with its max end,
the reversed count, and the clips whose windows round away entirely — so the
next person sees three warnings instead of discovering the format years later.

**Rule.** Derive a label builder's arity and bounds from the shipped
annotation's widest observed case, and log every span the clamp changes instead
of repairing it silently.

## 19. A graceful name-lookup fallback hides a missing class taxonomy

**Where it came from.** Gap G3 of `core/docs/PREVAD_SETUP.md`, resolved
2026-08-24. `_PREVAD_CLS_DEFS` in `core/data/definitions.py` held 22 CamelCase
class names — `CarAccident`, `FallDown`, `WarScene`, `Accident`. The release's
CSVs use `Car Accident`, `Fall to the Ground`, `War`, and 32 others. 28 of the
35 observed classes had no definition entry.

**Where the wrong list came from.** `LaGoVAD-PreVAD/src/datasets/PreVAD.py`
opens with a 22-name `DEFAULT_CLASSES` that is **entirely commented out**,
immediately followed by the live 36-name list. The port took the first block it
read. Both are named `DEFAULT_CLASSES`; only one is code.

**Why nothing catches it.** Two guards look like they would, and neither fires:

- `class_index_tensor` (`core/train.py:136-148`) raises on a class absent from
  `defs.json` — but `defs.json` is generated from the same CSV the class names
  come from, so it always agrees.
- `verbalize_class_name` (`core/data/definitions.py:342-346`) returns the bare
  class name when the class has no definitions. That is deliberate, documented
  ("Classes without definitions verbalize to themselves — unlike
  `verbalizer([...])`, this never changes list length"), and exactly right for
  the `Abnormal` fallback it was written for.

Composed, they are silent. The text branch gets `"Store Robbery"` where it
should get *"Someone robs a shop or a convenience store, threatening the clerk
over the counter…"*, `H_mul`'s class features become a bag of two-word labels,
and the definition-conditioning that is LaGoVAD's central claim quietly does not
happen. The run completes. The checkpoint loads. The AUC is a number.

`PREVAD_SETUP.md` §3 asserted this would "raise on the first batch". Reading the
fallback shows it cannot — and a silent version of this defect is strictly worse
than the predicted crash, because a crash is a gate and this is not.

**The fix is a positive check, not a better fallback.** Do not make
`verbalize_class_name` strict; its lenience is load-bearing elsewhere. Assert
coverage where the data and the taxonomy first meet — in the preprocessor:

```python
undefined = sorted({r.class_name for r in records} - set(DATASET_CLS_DEFS[key]))
if undefined:
    raise ValueError(f"{len(undefined)} classes have no definition sentences: {undefined} ...")
```

Classes defined but never observed go the other way: logged, and left out of
`defs.json`, so `H_mul` gets no permanently-negative column. PreVAD has exactly
one (`Fire-related Accident`, a superclass with no "Others" rows) — the lookup
carries 36 keys, `defs.json` carries the 35 that occur.

**Rule.** Validate that every class observed in the data has definition
sentences at preprocessing time, and never let a name-lookup fallback stand in
for a missing definition.

## 20. A dataset-sized glob is not an argument list

**What happened.** `PREVAD_SETUP.md` §4.3 told the user to inflate the released
PreVAD features to `/content` and then flatten the archive's own top-level
directory:

```bash
time unzip -n -q "$PREVAD_ZIP" -d "$PREVAD_CLIP"

if [ -d "$PREVAD_CLIP/ViT-B-16-8p-features" ]; then
  mv "$PREVAD_CLIP/ViT-B-16-8p-features/"* "$PREVAD_CLIP/"   # <-- 35,279 args
  rmdir "$PREVAD_CLIP/ViT-B-16-8p-features"
fi
```

First real run, 2026-08-25, on Colab:

```
real    0m14.491s
user    0m11.654s
bash: line 15: /usr/bin/mv: Argument list too long
```

The extraction itself was fine — 2.39 GiB in **14.5 s**, CPU-bound inflate. The
flatten is what died. `execve(2)` caps the combined size of `argv` + `envp`
(`ARG_MAX`, ~2 MB on Linux); 35,279 paths averaging ~60 bytes is well past it,
and the shell cannot even start `mv`.

**Why it survived review.** The identical idiom is correct in `COLAB.md` for
DoTA (1,403 clips) and for MSAD. It is not a wrong idiom — it is an idiom with a
size limit nobody states, and the limit sits between "the dataset I tested on"
and "the dataset the runbook exists for". `bash -n` does not catch it; a linter
cannot; the shell cell is not covered by the test suite. It fails **only** at the
scale where it matters.

**Why the failure mode is worse than a crash elsewhere.** `set -euo pipefail`
stopped the cell after the unzip succeeded, so the tree was left as
`$PREVAD_CLIP/ViT-B-16-8p-features/*.npy` — features present, one level too
deep, and every later cell (`§4.3`'s verifier, `§5`, `§6`) reports them
*missing*. That reads as a bad download, not as a failed `mv`.

**The fix is to not create the prefix.** `unzip -j` junks directory paths during
extraction, so there is nothing to flatten:

```bash
time unzip -n -j -q "$PREVAD_ZIP" -d "$PREVAD_CLIP"
```

`-j` is safe here for a reason worth stating: the contract is a flat
`{video_id}.npy`, and `video_id` equals the `.npy` stem for all 35,279 rows, so
no two entries can collide once the prefix is gone. Check that before reaching
for `-j` on some other archive.

Recovery from the half-done state runs in the same cell, and only **after** the
flat count is confirmed — never delete the nested copy on the assumption the
re-extraction worked:

```bash
N=$(find "$PREVAD_CLIP" -maxdepth 1 -name '*.npy' | wc -l)
[ "$N" -eq 35279 ] || { echo "EXPECTED 35279"; exit 1; }
if [ -d "$PREVAD_CLIP/ViT-B-16-8p-features" ]; then
  rm -rf "${PREVAD_CLIP:?}/ViT-B-16-8p-features"
fi
```

**When a bulk operation really is needed**, two forms have no argv limit:

- move the *containing directory*, which is one rename:
  `mv "$D/prefix" "$D.tmp" && rmdir "$D" && mv "$D.tmp" "$D"`
- stream the paths: `find "$D/prefix" -name '*.npy' -print0 | xargs -0 -I{} mv {} "$D/"`
  (slower — one `mv` per file — but bounded)

**Rule.** Never expand a dataset-sized glob into a command's argument list; make
the tool that writes the paths produce the layout you want, or operate on the
containing directory.

## 24. A non-differentiable op on the score path silently freezes everything upstream of it

### What was built

KIP's "kinematic gate" was specified as a learned function of motion intensity:

```python
m_hat = (m - m.min()) / (m.max() - m.min() + eps)      # per-clip min-max, in [0, 1]
ratio = torch.sigmoid(self.mlp(m_hat.unsqueeze(-1)))    # 1 -> 16 -> 16 -> 1
counts = (ratio * self.max_shift).floor().long()        # <-- gradient dies here
vk = torch.where(channel < counts, past, torch.where(channel < 2 * counts, fut, vt))
```

### Why nothing caught it

- **The module has parameters.** `sum(p.numel() ...)` returns 321; the optimizer
  accepts them; `state_dict()` saves them; checkpoints round-trip.
- **Every loss falls.** `L_MIL` reached 0.0012. The trunk trains fine — gradient
  still reaches `v^t` through the `torch.where` *values*, just never through the
  *condition*.
- **Static analysis is clean.** `ruff`, `mypy`, `pyright` have nothing to say
  about a tensor that happens to have no gradient path.
- **The result was good.** +0.0915 +/- 0.0088 DoTA micro AUC, three seeds, two
  controls, CI excluding zero. A real, replicated effect — attributed to a
  mechanism that did not exist.

### How degenerate it actually was

The gate never trains, so **random init is the deployed function**, and because
`m_hat` is min-max normalized, `[0, 1]` is the *entire reachable input domain*.
Sweeping that domain is therefore a complete characterization, not a sample:

| seed | span of `s_t` (of 128) | mean |
|---|---:|---:|
| 2024 | 1 | 63.6 |
| 2025 | 3 | 69.4 |
| 2026 | 3 | 64.4 |
| 0 | **0** | 68.0 |
| 42 | 1 | 58.4 |

`Linear(1,16) -> GELU -> Linear(16,16) -> GELU -> Linear(16,1)` at default init
has output magnitude ~1e-3 on a scalar input, so `sigmoid(.) ~ 0.5` and
`floor(0.5 * 128) = 64` almost everywhere. Side by side with an explicit
constant gate at `r = 0.5` (span 0, mean 64.0), the two are indistinguishable.

### The trap inside the fix

The natural STE repair, and the one the spec wrote, is:

```python
s = u + (u.floor() - u).detach()     # forward = floor(u), backward = identity
```

This **does not work here**. It gives `s` a gradient, but `s` is consumed only
inside `channel < s` — and a comparison emits a boolean, which no gradient flows
through. Shipped as written, the "gradient intervention" ablation would have been
bit-identical to the frozen gate in *both* passes and would have silently
measured nothing.

The estimator has to be applied to the **selection weights**, not the index:

```python
hard = (channel < u.detach().floor()).to(vt.dtype)
soft = torch.sigmoid(u - channel)
w = hard + (soft - soft.detach())     # forward exact, backward smooth
vk = w_past * past + (w_upto2 - w_past) * fut + (1 - w_upto2) * vt
```

Verified forward-identical (`torch.equal`, max diff 0.0) with a live gradient to
both the MLP and `e_O`.

### The cost

Three training campaigns, a headline result, and every write-up describing "321
parameters of motion-gated temporal mixing". The control that would have exposed
it in one run — a fixed-ratio temporal shift with no flow at all — was never run,
because the architecture on paper made it look like a strawman.

### The guard

Two tests, both cheap:

```python
def test_gradient_reaches_everything_we_call_learned():
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters())

def test_decision_variable_actually_varies():
    s = gate.compute_shift_counts(vt, eo, mask)
    assert int(s.max() - s.min()) > 0
```

Plus `--dump-kip-diag`, which writes `s_t` into every score `.npz` so the
distribution is one histogram away at any point in the future.


## 25. A second source for an existing cache must be proven bit-identical, not merely equivalent

### The setup
TAD ships extracted frames; MSAD ships videos. `core/flow/raft_extract.py` only
read videos, so TAD could not build `e_O` and no KIP-on arm could train on it.
The fix mirrors the CLIP extractor: keep `extract_directory` (videos) untouched,
add `extract_frame_directory` (frame folders), and factor the shared loop into
one private `_extract_sources` so the two cannot drift in cache layout, write
order or resume semantics.

### What went wrong
The parity test — dump one video's frames to lossless PNG, extract both ways,
compare — failed:

```
max abs diff 2.670288e-05   (flow stats, O(1)-O(10) values)
```

Three hypotheses were live: a stride mismatch, PNG round-trip loss, or
non-determinism in RAFT. All three were wrong, and guessing would have led to
"it's just float noise, relax the tolerance".

### The measurement that settled it
```
frame arrays equal:            True        # np.array_equal on the uint8 pixels
video arr contiguous:          True
frames arr contiguous:         False       # read_images returns a permuted VIEW
preprocess_for_raft identical: True        # max diff 0.0
video_flow_stats, same input twice: identical
video_flow_stats(np.ascontiguousarray(frames)) == video path: True, diff 0.0
```

So: identical values, identical preprocessing output, a self-reproducible
function — and a different answer. The only remaining variable was the numpy
array's **strides**. `read_images` builds
`torch.stack(images).permute(0, 2, 3, 1).numpy()`, which is a non-contiguous
view; torch's batched convolutions dispatch to a different kernel on one.

### Why it mattered more than 2.67e-5 suggests
The magnitude is far below anything that moves an AUC, and that is precisely the
danger. `cache/flow/v1/{DATASET}/` is supposed to mean the same thing whether
the dataset shipped videos or frames — that premise is what licenses comparing a
TAD-trained arm to an MSAD-trained one. A cache that silently depends on which
reader built it breaks that premise while passing every test that checks shape,
dtype, resume and "it ran". It is the same family as lesson 2 (a cache is bound
to something not recorded on disk), one layer lower.

### The fix
```python
# core/data/video_io.py
return np.ascontiguousarray(read_images(paths[::stride]))
```

One line, plus the test that would have caught it:

```python
def test_matches_the_video_path_exactly(self, video_fixture, tmp_path):
    """Same pixels in, bit-identical e_O out — whatever the source was."""
    ...
    assert np.array_equal(np.load(from_frames[0]), np.load(video_target))
    assert np.array_equal(np.load(frames_stats), np.load(video_stats))
```

`read_images` itself was left alone: it is the CLIP extractor's shared primitive
and changing it would perturb a path with a measured lineage for no gain here.

### Rule
Assert a new source path is bit-identical to the existing one on the same
content, and make any array you hand to a model C-contiguous. Equality of the
*inputs* is not evidence — assert equality of the *artifact*.

---

## 27. A score head whose kernel spans the clip is a clip classifier

### What happened
Seven arms were trained on DADA-2000 and scored in-domain and zero-shot on DoTA
(`core/docs/v3/RESULTS_DADA.md`, 2026-09-06). In-domain micro AUC came out at
**0.86** and looked like a strong frame-level result next to SimpleTAD's
published 85.6 on the same dataset family. It was not a frame-level result at
all: `auc_macro` across the seven arms ranged **0.44-0.57**, i.e. chance, and one
arm was measurably *below* it.

### The measurement that settled it
A model emitting **one constant score per clip** — no localization whatsoever —
scores micro AUC **0.9069** on this test split, because 3,880 of 5,244 test
frames (74%) come from all-normal `0_Normal_Driving` clips. The best arm reached
0.8739, i.e. **96% of that oracle**. There was nothing left over for
localization to explain.

The score curves say the same thing directly:

| quantity | value |
|---|---|
| median within-clip score range, [0,1] scale | **0.05-0.12** |
| between-clip / within-clip score variance | **3.8-48** |
| Spearman(that ratio, DADA micro AUC), n=7 arms | **+0.68** |
| Spearman(that ratio, DoTA micro AUC), n=7 arms | **-0.82** |

The flatter a model's curve, the better it separates *clips* and the worse it
does anything a within-clip metric can read.

### Root cause
`core/models/heads.py:21`, with `score_head_layers=1`:

```python
nn.Conv1d(512, 1, kernel_size=9, padding=4, padding_mode="replicate")
```

Receptive field 9. Median clip length at stride 8, measured from the saved score
files:

| corpus | median T | fraction of clip inside the kernel | clips with T <= 9 |
|---|---:|---:|---:|
| MSAD | 86 | 10% | 0% |
| DoTA | 13 | 69% | 12% |
| **DADA-2000** | **9** | **100%** | **55%** |

`kernel_size=9` was inherited from LaGoVAD, which trains on clips ~10x longer.
On DADA every output timestep sees every input frame, so the head is a clip
pooling operator on the majority of the corpus.

The supervision degenerates in the same direction. `core/losses/mil.py:22`:

```python
return max(1, n_valid // topk_pct)     # topk_pct = 16, n = 9  ->  k = 1
```

so `L_MIL` on the median clip is a plain max over 9 frames. And the labels are at
their floor too: median **2** positive frames per abnormal clip, 53 clips with
exactly 1, 4 clips whose window vanished entirely at stride 8.

### Why it is dangerous rather than merely wrong
Nothing raises. The model trains, and it trains *better* by the visible loss —
epoch-19 mean `mil` 0.255 for the collapsed arm vs 0.490 for the arm that stayed
temporally varied — because the in-domain metric rewards exactly the degenerate
solution. The number that comes out is high, is called "frame-level AUC", and is
directly comparable-looking to published frame-level numbers. It then propagates:
those arms transferred to DoTA as clip classifiers and landed at or below chance
once per-clip min-max deleted the between-clip scale they had learned.

### Rule
Check every temporal receptive field on the score path — `score_head_kernel`,
`temporal_window`, the MIL top-k divisor — against the new corpus's **median**
sequence length at the configured stride, before training. Gate the run on
`auc_macro`, not micro AUC. Related: [[lesson-12]] (the metric half of the same
failure), [[lesson-2]] (stride is part of the cache identity, so changing it to
fix this invalidates every cached feature and every metric measured on it).

