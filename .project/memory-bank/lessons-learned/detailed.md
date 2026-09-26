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
scores micro AUC **0.9086** on this test split, because 3,896 of 5,244 test
frames (74%) come from clips that are all-normal.
*(Corrected 2026-09-08 from 0.9069 / 3,880: the original used the
`0_Normal_Driving` subgroup count, which omits the 16 frames of the four
vanished-window clips — abnormal in `meta.json`, all-zero after stride-8
rounding, therefore all-normal clips for every metric. See [[lesson-28]].)* The best arm reached
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


---

## 28. A corpus can leak its label through clip length

### What happened
The DADA-2000 build the project trains on is a **reconstruction**, not the
original >100 GB release. While diagnosing why every arm's `auc_macro` sat at
chance (2026-09-08), the per-clip score curves were re-read from disk and the
clip *lengths* were tabulated against the clip labels. They are almost
separable.

### The measurement that settled it
On the 383-clip test split (383 `.npz` files, `constant_s2024`):

| | abnormal | normal |
|---|---:|---:|
| n | 191 | 192 |
| median T (stride-8 frames) | **7** | **19** |
| max / min T | **17** | 1 |
| clips with T >= 18 | **0** | **107** |

A detector whose only input is `T`, emitting the constant score `-T` for every
frame of the clip:

```
micro AUC = 0.8654    micro AP = 0.2630    macro AUC = 0.5000 by construction
```

Best trained arm: micro **0.8739**. The ruler is 0.0085 behind it, and ahead of
four of the seven arms. `outputs/EDA/DADA2000/eda_report.md` §1.4 shows it is a
corpus property, not a split artifact: 7.0 sampled frames per `CarAccident` clip
against 20.6 per `0_Normal_Driving` clip.

### Root cause
The reconstruction trims accident videos around the accident and leaves normal
driving videos at full length. The model has access to the leak: the temporal
encoder's band mask and RoPE positions are length-dependent
(`core/models/temporal_encoder.py:238`) and `core/inference.py:95` scores each
clip at its true length with no padding, so the shortcut is learnable at train
time and exploitable at test time.

### Why nothing caught it
`core.tools.eda` profiles clip length and label geometry **separately** — it
reports the median T and the clip-level oracle, but never crosses length against
the label. The corpus loads, the model trains, the loss falls, and the headline
micro AUC reads as a strong in-domain result.

### Why it is not lesson 12 or 27
C12 is a **metric** artifact (micro pooling over a skewed label distribution).
C27 is an **architecture** artifact (receptive field >= clip). C28 is a
**data** artifact: the corpus carries the label in a non-visual channel that
survives every model change. All three were live simultaneously on DADA, which
is why the failure was so hard to attribute.

### The rule
Compute a length-only constant-score baseline for every new corpus before
training on it, print it beside every micro AUC from that corpus, and treat any
arm that fails to beat it as unmeasured. Fix by rebuilding into fixed-length
windows with the anomaly at a random offset.

---

## 29. DVS marks the whole anchor clip positive

### What happened
The user's report was specific: "the model doesn't score normal frames with a
low score, it still scores them high as abnormal frames." Reading the loss
wiring end to end (2026-09-08) found the mechanism, and it is not a training
failure — it is what the objective asks for.

### The code
`core/data/synthesis.py:83`:

```python
pseudo = torch.zeros(len(features), dtype=torch.float32)
if anchor_is_abnormal and fillers:
    pseudo[start : start + len(anchor_features)] = 1.0   # the WHOLE anchor clip
```

consumed by a **dense per-frame BCE** at `pseudo_sup_weight = 1.0`
(`core/losses/dvs.py:16`, wired `core/train.py:302`). The docstring calls this
"the known-window supervision of spec §6.1" — which is true only when an
abnormal clip is approximately all anomaly.

### The measurement that settled it
DADA-2000 abnormal clips have a mean true positive fraction of **0.351**
(median 0.333), so the dense BCE trains **64.9 %** of the anchor's frames to 1
against a 0 annotation. And it is the *densest* gradient in the objective — every
frame counts, unlike `L_MIL`, which touches one.

Reading all four wired terms (`core/train.py:285-336`), **no term pushes any
frame of an abnormal clip toward 0**:

| Term | abnormal clip | normal clip |
|---|---|---|
| `mil_loss` | top-k (k=1 on 90 % of DADA clips) -> 1 | top-k -> 0 (suppresses the whole clip) |
| `supervised_loss` | *all* anchor frames -> 1 | all frames -> 0 |
| `pseudo_sup_mil_loss` | top-k inside the anchor span -> 1 | top-k -> 0 |
| `multi_class_mil_loss` | top-k -> `CarAccident` | top-k -> `Normal` |

The loss is perfectly satisfied by a per-clip constant, and the run converges on
exactly that: `dvs_sup` falls **0.681 -> 0.027** over 500 steps, and the trained
arm scores

```
positives in abnormal clips        0.4076
negatives in abnormal clips        0.4078   <- gap -0.0002
frames in all-normal clips         0.0493   <- clip offset +0.359
```

`auc_macro` 0.5292; the argmax frame is a true positive in 28.3 % of abnormal
clips against a 35.1 % base rate.

### What is correct here
The masking is right: `dvs_rows` (`core/train.py:295`) excludes un-synthesized
abnormal rows, whose `y^p` is legitimately all-zero. The defect is the
**semantics of the span**, not the row selection.

### The fix
Make the anchor interior an **ignore** target for the dense BCE — filler frames
stay hard negatives, and `pseudo_sup_mil_loss`'s in-span top-k supplies the
positive pressure it was already designed to supply. Ship it as a config arm
(`loss.dvs_anchor_mode = span | ignore`) defaulting to today's behavior, so the
MSAD and PreVAD numbers stay reproducible. Consider a bottom-k MIL term on
abnormal clips as the general antidote to "no downward pressure inside a
positive bag".

### The rule
Check the mean positive fraction of an abnormal clip before enabling DVS on a
new corpus, and never let a dense frame-level BCE consume a span label that is
known to be wider than the anomaly.

---

## 30. An eval-time sampler seeded once per run couples every item's result to which other items were scored

### What happened
Phase 1's C28 length control (`core.evaluate --equalize-length 5
--equalize-anchor end`) was expected to differ from the unfiltered eval only by
the crop. It also differs by the **text conditioning**.

`core/evaluate.py:159-164` constructs the verbalizer once, above the scoring
loop:

```python
verbalizer = DatasetSpecVerbalizer(
    dataset_abbr(dataset_name),
    rng=random.Random(constants.SEED),
)
class_feats_fn = make_class_feats_fn(text_encode_fn, class_names, verbalizer)
```

`make_class_feats_fn` samples a definition sentence **per scored window**, so
every clip advances a single shared RNG. The equalize branch added in `814c177`
sits *before* `sliding_window_scores` and `continue`s on a short clip:

```python
if len(frame_label) < args.equalize_length:
    dropped_short.append(video_id)
    continue          # <-- consumes no draws; every later clip shifts
```

52 of 383 DADA clips take that branch, and shorter windows draw fewer samples,
so clip *k*'s conditioning in the eq5 run is not clip *k*'s conditioning in the
raw run.

### The measurement that isolated it
34 test clips have `T == 5` exactly. For those, `equalize_window(5, 5, "end")`
returns `(0, 5)` — the crop is the identity and the model input is byte-identical.
**32 of 34 curves still differ**, max |Δ| = 0.0033 (p1_ctrl), 0.0025–0.0029 on
the other arms. Two clips match by coincidence. All four arms show it, including
`p1_ctrl`, whose *both* evals ran on post-`814c177` code — so it is not version
skew.

Corroborating datum from the other direction: `kipoff/eval_dada_eq5` and
`p1_ctrl/eval_dada_eq5` are **331/331 bitwise identical**. Two different
checkpoint paths, same code path, same drop set → same RNG stream → identical
output. The stream, not the weights, is what the filter perturbs.

### Why nothing caught it
The seed is *present*, and the comment above it explicitly claims
reproducibility — it was added after an earlier unseeded-verbalizer bug
(±0.003 AUC, ±0.3 per-video `max_score` on the MSAD slice). The fix made a run
reproducible **against itself**, which is what was tested. Nobody asked whether
item *k*'s result was independent of items *0..k-1*, and no linter or type
checker can see a stream dependency.

### Blast radius
Only comparisons **across item subsets** are affected: raw vs `--equalize-length`,
or any future subset flag. Comparisons *within* one protocol share the drop set
and are exact, so the Phase 1 arm ladder (`RESULTS_DADA_PHASE1.md` §3–§6) is
valid. The magnitude, ±0.003 AUC, is ~⅓ of the raw-vs-eq5 deltas and ~10× smaller
than the C28 leak it was built to control — so the control is still worth having,
it just cannot be read as a pure crop effect.

### The fix
Rebuild the sampler per item from a stable key, inside the loop, so the draw
depends on the item and nothing else:

```python
for i in range(len(dataset)):
    ...
    class_feats_fn = make_class_feats_fn(
        text_encode_fn, class_names,
        DatasetSpecVerbalizer(abbr, rng=random.Random(constants.SEED + i)),
    )
```

Pin it with a test that scores a set, scores a subset, and asserts the shared
curves are bit-identical.

### The rule
Seed any eval-time sampler per scored item, never once per run, so that skipping
an item cannot change another item's result.

---

## 31. A loss that lowers scores is not a loss that creates contrast

### What happened
Phase 1 shipped two losses whose purpose was to create **within-clip
separation** on DADA-2000, and pre-registered two success criteria for them
(`DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` §6):

* 1.1 `dvs_anchor_mode=ignore` — "the within-abnormal-clip gap turns positive"
* 1.2 `bottomk_weight>0` — "within-clip score range widens"

Both criteria are **scale-dependent**, and both losses change the scale.

### The measurement

| arm | mean score (pos) | mean within-clip range | mean σ | gap | **d = gap/σ** |
|---|---:|---:|---:|---:|---:|
| P0 control | 0.1160 | 0.1473 | 0.0521 | +0.0085 | **+0.164** |
| P1 `ignore` | 0.0771 | 0.1198 | 0.0425 | +0.0038 | +0.091 |
| P2 `bottomk` | 0.0953 | 0.1179 | 0.0414 | +0.0045 | +0.108 |
| P3 both | 0.0635 | 0.0925 | 0.0326 | +0.0013 | +0.039 |

Read as raw gaps this is ambiguous: gap and scale both shrank, so "worse
separation" and "same separation, quieter model" are indistinguishable.
Normalized it is decisive — `d` collapses 76 % on P3, and `auc_macro` agrees
(0.5190 → 0.5097), which is the point: `d` tracks the metric the arms were
supposed to move, and the raw gap does not.

### Why this was predictable from the gradients
Each term does exactly what it says and no more. `ignore` deletes the upward BCE
target on 64.9 % of anchor frames — fewer frames pushed up, lower scale.
`bottomk` pushes the lowest-k frames of an abnormal clip down with **nothing**
raising the rest harder — lower scale again. Shrinkage was the expected outcome;
the pre-registered readout simply could not tell it apart from success.

### Where else this shape appears
[[lesson-29]]'s headline is "positives 0.4076 vs negatives-in-abnormal 0.4078,
gap −0.0002" — a raw gap, with no scale beside it. It is still correct (the
conclusion there was "no separation", and `d ≈ 0` either way), but the number as
written cannot be compared to Phase 1's, because the two runs have different
score scales. Quote `d` when comparing across arms or campaigns.

### The rule
Report a within-clip positive/negative gap divided by the mean within-clip score
std, and print the raw score scale beside it, whenever comparing loss arms.

---

## 32. A fixed-length re-shard must be sized against the shortest class it has to preserve

### What happened
Phase 2a rebuilt DADA-2000 into fixed-length windows to close [[lesson-28]]. The
plan chose `--window-length 32 --stride 2` from this line of reasoning:

> median source clip is ~36 sampled frames at stride 2, so a 32-frame window
> gives 1-3 windows per clip

That median is the **corpus-wide** median, which on DADA is the normal class's
median. The class that binds is the abnormal one, and C28 had already said why:
accident clips are *trimmed around the accident*. Measured from
`data/DADA2000/meta.json` (n = 975 abnormal / 938 normal source clips):

| raw frames | p5 | p25 | **p50** | p75 |
|---|---:|---:|---:|---:|
| abnormal | 22 | 35 | **49** | 64 |
| normal | 29 | 79 | **139** | 209 |

A 32-frame window at stride 2 needs **64 raw frames** — the abnormal p75. So:

| window (raw frames) | abnormal kept | windows/abnormal clip | normal kept | windows/normal clip |
|---|---:|---:|---:|---:|
| 24 | **94.2 %** | 3.1 | 96.1 % | 11.9 |
| 32 | 81.8 % | 2.2 | 91.9 % | 8.9 |
| 48 | 51.7 % | 1.5 | 86.9 % | 5.8 |
| **64 (as built)** | **25.3 %** | 1.2 | 81.7 % | 4.2 |

### The two failure modes, and they compound
1. **Deletion.** 74.7 % of abnormal clips produced no window at all (the
   no-padding rule is correct — padding would fabricate frames, [[lesson-27]] —
   so a too-long window silently *is* a filter). Abnormal training windows: **253**,
   against ~800 abnormal clips before. `DVS dataset length 506 -> 8 steps/epoch`
   where the Phase 1 arms ran ~25.
2. **Manufactured imbalance.** With a fixed hop, windows-per-clip scales with clip
   length, and DADA's normal clips are 3x longer. Result: **327 abnormal vs 3,244
   normal** windows from a clip corpus that was ~1:1.

### Why the gate did not catch it
Because the gate checked the thing the rebuild was *for*. C28 was closed
perfectly — clip-length AUC **0.5000**, length-only micro **0.5000**, zero clips
separable by length, the CRITICAL verdict gone from the EDA report. Meanwhile:

| | stride-8 corpus | rebuilt corpus |
|---|---:|---:|
| clip oracle micro | 0.9086 | **0.9766** |
| frames in all-normal clips | 74 % | **92.5 %** |
| two-class test clips (`auc_macro`'s sample size) | 190 | **57** |

A rebuild can close the defect it targets and destroy the corpus in the same step.

### The fix
Two code changes, both now enforced:

* `core/data/windows.py:cap_windows` + `--window-max-per-clip` (default 4) — keeps
  at most N windows per source, **evenly spaced**. Evenly spaced, not the first N:
  DADA's accident sits at the clip end, so a head-biased cap would drop it.
* `core/data/dada.py:plan_record_windows` logs
  `Abnormal source retention: N/M (X%)` and warns below
  `WINDOW_ABNORMAL_RETENTION_WARN = 0.9`, naming this lesson.

And the geometry: **stride 1, window 24, hop 12, cap 4** — the same clips as
window 12 at stride 2 (94.2 % retention) but with twice the temporal resolution
inside each window, which matters because at 12 frames `score_head_kernel=9`,
`temporal_window=9` and `mil_topk_pct=8` are all degenerate again and there is
nothing left to ablate.

### The rule
Size a fixed-length window against the shortest class's length distribution, cap
windows per source clip, and gate the rebuild on class retention and the two-class
count rather than on the leak metric alone.

---

## 33. A pre-registered threshold is only rigorous if it is reachable

### What happened
Phase 2's Gate W pre-registered two criteria. E1 (length leak closed) was sound.
**E2 — "clip oracle micro < 0.75" — could not be satisfied by any corpus.**

It was picked the way such numbers usually are: the corpus measured 0.9086, and
0.75 looked like a decisive improvement. Nobody derived what the metric can do.

### The arithmetic that should have been done first
The constant-score-per-clip oracle gives every abnormal clip 1 and every normal
clip 0. Its positives all sit in abnormal clips; its negatives split between
normal clips (score 0, ranked correctly) and the normal frames *inside* abnormal
clips (score 1, tied with the positives). So with `F_norm` = frames in all-normal
clips and `X` = negative frames inside abnormal clips:

```
oracle = (F_norm + 0.5·X) / (F_norm + X)
```

Checked against both measured corpora, exactly:

* stride-8: `(3896 + 436)/(3896 + 872)` = **0.9086** ✓ (report: 0.9086)
* rebuilt:  `(22496 + 552.5)/(22496 + 1105)` = **0.9766** ✓ (report: 0.9766)

`oracle < 0.75` ⟺ `F_norm < X`. With a positive fraction of 0.39 inside abnormal
clips, that means abnormal clips must hold **≥ 62 % of every test frame**. A
perfectly balanced test set scores **0.811**; the realistic target for the
rebuilt corpus (1:1.3) scores **0.840**.

### Why it matters even though the gate "correctly" failed
E2 failed, and the gate *should* have failed — but for [[lesson-32]]'s reason, not
E2's. The failure was accidentally right, which is the worst kind: it would have
been read as evidence the criterion works. The symmetric error is the dangerous
one — a threshold that cannot fail waves a broken corpus through.

The replacement gates on what actually moves: **two-class window count >= 150**
(`auc_macro`'s sample size; 190 before, 57 in the failed build), abnormal-source
retention, and the class ratio. The oracle is still computed and printed beside
every micro number ([[lesson-12]]) — it is a caveat, not a gate.

### The rule
Derive a pre-registered metric's attainable range from the corpus's class mix
before setting the threshold, and record that derivation beside the number.

## 35. A pre-registered threshold needs a MEASURED lever, not an assumed one

**Where it came from.** Gate W (Phase 3 of the DADA-2000 original corpus) failed
on one criterion of six: clip oracle **0.7529** against a **0.75** bar. The plan's
risk 3 told the operator which flag to reach for, and it was the wrong one.

**The measurement.** Same 1,945-clip archive census, same build path
(`core.data.dada_origin`), one label rebuild per row (seconds each; the CLIP cache
is keyed by SOURCE clip and a window is a slice, so no re-extraction):

| W | hop | cap | clip oracle | retention | two-class | ratio | Gate W |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 8 | 2 | 0.7765 | 0.983 | 439 | 2.15 | FAIL |
| 16 | 8 | 3 | 0.7527 | 0.983 | 653 | 2.15 | FAIL |
| 16 | 8 | **4** | **0.7529** | 0.983 | 787 | 2.06 | **FAIL** |
| 16 | 8 | 6 | 0.7580 | 0.983 | 896 | — | FAIL |
| 16 | 4 | 4 | 0.7336 | 0.983 | 956 | 2.42 | pass |
| 18 | 9 | 4 | 0.7350 | 0.974 | 753 | 2.32 | pass |
| **20** | **8** | **4** | **0.7037** | **0.957** | **798** | **2.80** | **PASS** |
| 24 | 8 | 4 | 0.6557 | 0.899 | 780 | 3.87 | FAIL (retention, ratio) |
| 32 | 16 | 4 | 0.6127 | 0.728 | 388 | 5.00 | FAIL |

The cap — the flag the plan named — spans **0.025**. The window length, the flag
the plan warned against, spans **0.14** and is the only one that crosses the bar.

**Why the cap cannot work, from C33's closed form.** The oracle is
`(F_norm + 0.5·X)/(F_norm + X)`, with `F_norm` the frames in all-normal windows
and `X` the negative frames inside abnormal windows. `≤ 0.75` requires
`F_norm ≤ X`. At W=16 the corpus measured 6,528 vs 6,377 — short by **151 frames
of 19,536 (0.77 %)**, about ten windows. Capping windows per clip removes windows
of *both* kinds in proportion, so it moves `F_norm` and `X` together and the ratio
barely shifts. Lengthening the window converts whole all-normal windows into mixed
ones, which moves `F_norm` down and `X` up at the same time — two-sided, hence the
10x larger span. The mechanism was derivable; nobody derived it, and the remedy
was written from intuition instead.

**The sibling failure in the same plan.** §5.2's predicted oracle of **0.6631** at
W=16 was computed over the annotation alone. It is not wrong about the corpus — it
is the value at W≈24 — but W=24 fails retention (0.899 < 0.90) and class ratio
(3.87 > 3). A simulation that skips the build path can be right about a number and
wrong about which configuration produces it.

**What it cost, and what it would have cost.** Detected before any arm trained,
because the sweep was run when the gate failed rather than the prose followed:
three rebuild cycles saved. The real exposure was larger — the prose also invited
the *other* error, waving 0.0029 through as "only a bit over", which the sweep
showed to be unnecessary: a passing geometry existed two rows away.

**Not the same as C33.** C33 asks *can this threshold be met by any corpus*
(the attainable range). C35 asks *which flag moves it, and by how much* (the
lever). A threshold can be perfectly reachable, as this one was, and still be
unreachable in practice if the operator is pointed at an inert knob.

## 36. A resume API is not a load API

**Where it came from.** The first real run of the D2 gradient probe
(`core/tools/grad_probe.py`), 2026-09-20, on `s2024`'s stage-1 checkpoint. It
crashed in `Trainer.load_checkpoint` with 197 `Missing key(s) in state_dict`,
every one of them `clip_text_model.*` plus `clip_text_model.prompt_embedding.weight`.

**Why the keys are legitimately absent.** `build_trainer` decides the text tower
from the *stage*:

```python
needs_clip = text_encoder == TEXT_ENCODER_CLIP and cfg.train.stage == STAGE_FULL
model = KATVAD.from_config(cfg, load_clip=needs_clip)
```

Stage 1 freezes everything outside `kip.*`, so it never conditions on text and
never pays for the tower. Its checkpoint is therefore *correct* and *incomplete
by design* — and the project already knew this, which is why `warm_start_model`
exists and skips exactly the `clip_text_model.` prefix. Stage 2 warm-starts from
stage 1 through that function every time a campaign runs.

**The actual defect is a one-line API choice.** Two loaders sit next to each other
in `core/train.py`:

| | `Trainer.load_checkpoint` | `warm_start_model` |
|---|---|---|
| purpose | resume **this** run | seed a **new** run's weights |
| model load | strict | strict *except* the text tower |
| optimizer / scheduler / scaler | restored | untouched |
| RNG streams (python, numpy, torch, dataset, verbalizer) | restored | untouched |

A read-only probe wants the right-hand column on every row. It picked the left
because the method name matches the sentence "load the checkpoint". The strict
key check is only the first contract it could not satisfy: a stage-1 optimizer
state holds one parameter group (`kip.*`), so `optimizer.load_state_dict` would
have raised next; and `_restore_rng` unpickles `np.random.get_state()`, the exact
payload that expired across a numpy major in [[lesson-15]].

**Why the suite was green.** The probe shipped with 10 tests covering the term
re-sum guard, per-group reach, `lambda_rec` linearity and two refusals — and
**none** passed `--checkpoint`. The CLI test ran the probe at initialization,
where `args.checkpoint is None` and the whole branch is skipped. The one argument
that can fail was the one argument never exercised.

**The fix, and what it is not.** `grad_probe.main` now calls
`warm_start_model(trainer.model, args.checkpoint, flag="--checkpoint")`. This is
*not* [[lesson-5]] being relaxed: the load stays fail-loud, and a KIP-off
checkpoint probed under a KIP-on config still raises `ValueError` naming the
missing `kip.*` keys. It is the same strictness with the one documented exemption,
applied by the function that owns it.

**Cost.** One Colab session, and D2 — an instrument built specifically to settle
whether `lambda_rec` is the right lever — returned no data on the day it was run.
The measurement itself is unaffected: nothing was trained, nothing was written.

---

## 37 — A raw loss value measures its target's units, not the fit

**The trace.** Phase 4's paired KIP A/B on T2 (three seeds, configs differing in
exactly one line) came back **negative**: T2 micro 0.6182 → 0.6054, Δ **−0.0129**,
t95 [−0.0249, −0.0008], all three seeds agreeing. The obvious first reading of the
`metrics.jsonl` beside it was that the PMG head had failed to fit — `kip_rec` sat
at **11.3 / 12.0 / 11.6** while the four task losses together summed to
**0.87–0.91**, i.e. **93 % of `total`** at `lambda_rec = 1.0`.

That reading is unavailable, and the reason is the point of this lesson.

**What the numbers actually are.** `L_KIP_rec` is a bare masked MSE. Its target
`e_O` is 23 frame-global RAFT statistics — magnitude mean/std/**max**, `u`/`v`
mean/std in **raw pixel units**, plus an L1-normalized 16-bin angle histogram —
projected to 256-d by a fixed Gaussian map scaled `1/sqrt(23)`. Nothing
normalizes them anywhere between the extractor and the dataset. Measured on the
T2 train split (4,401 windows, 88,020 frames):

| predictor | MSE on `e_O` | what it is |
|---|---:|---|
| all-zeros | **71.15** | the loss at step 1 of an untrained head |
| one global-mean vector | **31.64** | **the baseline any `kip_rec` must beat** |
| each item's own mean | **16.21** | an oracle knowing item identity and nothing else |
| measured `kip_rec` | **11.62** | — |

So the "catastrophic" 11.62 is **R² = 1 − 11.62/31.64 = 0.633** against the
constant predictor, and **R²_item = 0.283** against the item-mean oracle: the head
fits real *within-item* structure. The 93 % share is a fact about **pixel units**,
not about the fit. `mag_max` alone (mean 27.19, sd 22.72) carries **83.0 %** of
`E[s²]`, and since `e_O = s @ M` with `M ~ N(0, 1/23)`, that one raw stat sets the
whole scale of the target.

**Why it is not merely cosmetic.** Every other term in the objective is a BCE or an
InfoNCE, i.e. **O(1) by construction**. At `lambda_rec = 1.0` the reconstruction
term therefore enters the sum ~**32×** oversized. `PMGFlowHead` reads `v^t`, and
stage 2 freezes nothing (`core/train.py:191-195` freezes only in stage 1), so that
gradient reaches the **shared temporal encoder**. The D2 probe, eight batches per
point, three seeds:

| point | `rho` = \|g_KIP\| / \|g_task\| | `cos(g_kip_rec, g_task)` |
|---|---:|---:|
| stage-1 end | **11.63** | ~+0.002 |
| stage-2 end | **3.105** (3.93 / 3.03 / 2.36) | **−0.0010** |

`rho > 1` means KIP moves the trunk further per step than the anomaly objective
does. `cos ≈ 0` means the two do **not** disagree about direction — KIP spends
trunk capacity **orthogonally**. That distinction decides the repair: under
conflict, lowering the weight trades one objective for the other and the clean
control is a `detach()`; under orthogonality, the fix is **normalization plus a
principled weight**. Corroboration from the same runs: stage 1 (trunk frozen)
plateaus at `kip_rec` **20.2**, stage 2 (trunk free) falls to **11.3** — **44 % of
the reconstruction gain came from rewriting `v^t`** — and every task loss ends
**24–32 %** higher than its paired `kip_off` run.

**Two traps inside the diagnosis itself.**

1. `rho` fell 11.63 → 3.10, which reads like KIP backing off. It is not: `|g_task|`
   **grew** (4.91 → 10.07, 7.24 → 12.46, 6.21 → 23.61) while `|g_kip_rec|` fell
   (51.9 → 39.2, 86.2 → 37.3, 69.4 → 55.2). Both terms moved; a ratio hides that.
2. The per-batch spread is large — `rho_kip_rec_sd` 9.54 at stage 1, 1.56–1.94 at
   stage 2. Report mean **and** spread; n=8 batches on one seed is a mechanism
   probe, not an effect size, and carries no t-interval.

**The rule, and why `1/V` rather than a sweep.** The equalizing weight is
`1/V = 0.0316`. Adopting a weight because it improved a Δ would be [[lesson-14]]
— fitting the benchmark. Deriving it from the target's own constant-predictor MSE
is a property of the **data**, measured before any arm runs, and it transfers to
the next corpus by re-running the same measurement rather than a new sweep.

**What this does not fix.** [[lesson-24]] is untouched: on `main` the gate MLP
still receives no gradient, so every KIP-on arm remains a fixed ~50 % channel
shift whatever `lambda_rec` becomes. And the pre-registered projection round-trip
came in at **0.9239** — inside its [0.9, 1.1] band but in the lower half, so any
new cache version must re-print it; outside the band every `kip_rec` measured on
that cache is uninterpretable.

**Outcome — the repair, measured (2026-09-26).** Option A (plan
`katvad-flow-zscore-option-a.md`, App. A) z-scored the 23 raw stats with T2-train
moments, re-projected them with the same `M`, and set `lambda_rec = 1/V_v2 = 0.9953`.
All build gates passed (`V_v2` 1.0047, G0 max\|err\| 0.0 over 1,491 files).

| | v1 (`lambda_rec` 1.0) | v2 (0.9953) |
|---|---:|---:|
| `rho` at stage-2 end | 3.105 | **0.179** (0.152 / 0.187 / 0.199) |
| `cos(g_kip_rec, g_task)` | −0.001 | −0.05 / −0.06 / +0.00 |
| `R²` vs global mean | 0.633 | 0.280 |
| `K` vs item-mean oracle `W` | 11.62 < 16.21 (**pass**) | 0.72 > 0.654 (**fail**) |
| T2 micro, KIP-on | 0.6054 | **0.6258** |

The capture is gone, and so is the cost: v2 − v1 is **+0.0204 [+0.0010, +0.0398]**
on T2 micro and **+0.0259 [+0.0160, +0.0358]** on macro. Against KIP-off, v2 is
**+0.0076 [−0.0002, +0.0154]**, which the pre-registered table reads as "neutral".

**A second trap, found only after the rescale.** On v1 the PMG head beat the
item-mean oracle, and that was the evidence that "the motion premise is not
refuted". On v2 it does not beat it. The v1 pass came from `mag_max`, the one stat
whose between-clip spread the head could fit. An oracle comparison made on an
unnormalized target inherits the dominant dimension's story. **Re-run every
oracle / R² comparison after any target rescale**, and do not carry a pass across
caches. The angle-histogram block, now 69.6 % of the target, is fitted at `R²`
0.07.

**Generalizes to:** any auxiliary regression head bolted onto a classification
objective — depth, flow, pose, reconstruction — whenever the two share a trunk.

## 38 — A normal pool imported from another source is admitted by a transfer probe, never by similarity

**The trace.** After T2 (negatives cut from inside the DADA accident videos) trained
without collapsing, the next proposal was to use full-length DADA videos as positives and
bring in normals from **D2City**. The argument was similarity: Chinese roads, dashcam,
25 fps against 30. `0_Normal_Driving` had carried the same argument and leaked, so the
plan pre-registered an admission gate (G-X) *before* any extraction was run.

**What the numbers were.** The mechanics were clean. The length leak was closed by
packing (0.5009), R0 reproduced Gate D0 (0.6763 vs 0.6518), and no labels mismatched.
Then:

| probe | negatives | `auc_macro` | shortcut AUC |
|---|---|---:|---:|
| R0 | DADA out-of-span (in-video) | 0.6763 | 0.820 |
| X | D2City only | **0.5864** | **1.000** |
| M | both | 0.6639 | 0.999 |

Δ(X−R0) = −0.0899, with every one of the 5 folds between −0.078 and −0.104. The crop arm
(DADA's 2.40:1 band) made it slightly worse. The source probe S was 1.0000. The reference
S-ref, which pits DADA against **DoTA**, was 0.9999.

**Why similarity cannot be the test.** Frozen CLIP embeds camera, ISP, codec, overlays and
geography along with the road. The montage showed an ego hood, a watermark, a timestamp
and a colour cast. Those are source identifiers a linear reader picks up at once. Given
two sources, a probe (and a MIL model) can separate them on those alone, and a bag from
the foreign source becomes *free* to call normal. Nothing in "looks alike" measures this.
The one thing that does is the question the gate asks: **do these negatives teach the
accident as well as the video's own normal frames do?**

**The rule that follows, and its default.** Admission is a paired transfer probe
(foreign pool as the only negatives, same folds as the in-video reference, bar: within
0.03). The source probe is a *claim* gate, never an admission gate: at 1.0000 vs 0.9999 it
separates everything, so it decides nothing. When no pool passes, the default is the T2
construction: negatives cut from inside the positive videos, which is leak-free by
construction (0.5000).

**Cost of the check.** About 1 GPU-hour of CLIP extraction plus about 20 min of probes. The
alternative is a RAFT extraction, a z-score cache, a corpus build and a 3-seed campaign,
all on a pool that would teach "which camera".

**Addendum 2026-09-26 — the fourth pool, BDD-A.** Same protocol, same DADA folds (R0
reproduced D2City's to 3.1e-5), gate read on the `calm` GPS arm because BDD-A is collected
around braking events. X = **0.5492**, Δ(X−R0) = **−0.1270** [−0.1532, −0.1008], 5/5 folds
past −0.06, shortcut **1.000**. That is worse than D2City, with **no overlay** in the
frames (no logo, no timestamp), so the shortcut does not need a watermark. The new
information is R0's starting side: before any training it ranked BDD-A frames *above* DADA
normals (shortcut 0.374), whereas D2City started below (0.820). Using BDD-A as the negatives
flipped it to 1.000 anyway. **The direction of the prior gap does not predict admission;
separability does, and on frozen CLIP every foreign dashcam corpus is separable.** G-M
passed only because Δ(M−R0) ≈ −0.003. It is not a gain. `core/docs/BDDA_EDA.md`.

