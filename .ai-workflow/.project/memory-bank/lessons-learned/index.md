# Lessons — Tier 1 catalog

**Re-initialized 2026-07-31 at `b9978ff`.** 7 lessons, all re-derived from this
tree. Add new ones per CLAUDE.md §7 (duplicate check → 5 gates → here +
`detailed.md` → update `meta-index.md` if CRITICAL/HIGH).

---

## 1. [CRITICAL] Env - faiss segfaults multi-threaded next to torch on macOS arm64
**Triggers:** faiss, knn, IndexFlat, segfault, libomp, omp
**Problem:** torch and faiss-cpu each bundle libomp; multi-threaded faiss search after torch is imported crashes the process.
**Bad:** `index = faiss.IndexFlatIP(d); index.search(q, k)  # default threads`
**Good:** `faiss.omp_set_num_threads(1)` at module import, before any index is built.
**Rule:** Call `faiss.omp_set_num_threads(1)` at import in any module that uses faiss alongside torch.
**Files:** core/data/knn_cache.py:34

## 2. [CRITICAL] Data - a feature cache is valid only for the transform and stride that built it
**Triggers:** clip features, extraction, transform, resize, center crop, stride, cache, re-extract
**Problem:** Changing the preprocessing transform or sampling stride silently invalidates every cached feature and every metric previously measured on it.
**Bad:** editing `preprocess_frames` to match another repo's convention, then reusing the existing cache.
**Good:** version the cache directory, re-extract, and re-measure the baseline before comparing anything.
**Rule:** Never change an extraction transform or stride without re-extracting the cache and re-measuring every gate that depended on it.
**Files:** core/tools/extract_clip_features.py:41, core/constants.py:69

## 3. [HIGH] Device - torch 2.4 MPS training diverges from CPU
**Triggers:** mps, apple silicon, device, diverge, loss increases, local training
**Problem:** On MPS, stage-1 `L_KIP_rec` climbs while CPU converges on identical seeds and data.
**Bad:** `train.device=auto` on a Mac → silently picks MPS → garbage run.
**Good:** `--set train.device=cpu` locally; CUDA for real runs.
**Rule:** Pin `train.device=cpu` for any local training on Apple Silicon under torch 2.4.
**Files:** core/docs/TRAINING.md

## 4. [HIGH] Supply chain - pin HF model revisions, never resolve at main
**Triggers:** huggingface, from_pretrained, revision, CVE, torch.load, safetensors
**Problem:** `main` of `openai/clip-vit-base-patch16` ships only `pytorch_model.bin`, which transformers 4.56 refuses to load on torch < 2.6 (CVE-2025-32434).
**Bad:** `CLIPModel.from_pretrained("openai/clip-vit-base-patch16")`
**Good:** pass an explicit `revision=` sha that is known to ship safetensors.
**Rule:** Always pass an explicit pinned `revision` to every `from_pretrained` call.
**Files:** core/constants.py:CLIP_MODEL_REVISION

## 5. [HIGH] Checkpoints - fail loud on unknown, missing or mis-shaped keys
**Triggers:** load_state_dict, checkpoint, strict, partial load, ckpt_compat
**Problem:** A tolerant load yields a model with randomly initialized layers that still produces plausible-looking scores.
**Bad:** `model.load_state_dict(sd, strict=False)`
**Good:** explicit key mapping; raise on anything unaccounted for.
**Rule:** Raise on any key the checkpoint mapper does not explicitly account for — never `strict=False`.
**Files:** core/models/ckpt_compat.py

## 6. [MEDIUM] Porting - read ported code, don't transcribe it
**Triggers:** port, lagovad, baseline, np.pad, collate, reimplement
**Problem:** The baseline's collate called `np.pad` with a scalar pad-width, which pads *every* axis, not just time.
**Bad:** `np.pad(feature, pad_len)`
**Good:** `np.pad(feature, [(0, pad_len)] + [(0, 0)] * (feature.ndim - 1))`
**Rule:** When porting from `LaGoVAD-PreVAD/`, verify each function's semantics against a test rather than copying it.
**Files:** core/data/collate.py:27

## 7. [MEDIUM] Deps - don't depend on library internals that drift across versions
**Triggers:** get_scheduler, transformers, lr schedule, internal api
**Problem:** `transformers.get_scheduler` is an internal-facing helper whose behavior changes between minor versions, silently altering the LR curve.
**Bad:** `get_scheduler("cosine", opt, num_warmup_steps=20, ...)`
**Good:** an in-house `LambdaLR` closure with its own unit test.
**Rule:** Implement small schedule/utility functions in-house with a unit test rather than importing an unstable library internal.
**Files:** core/train.py, core/docs/TRAINING.md

## 8. [HIGH] Comparability - reproduce the baseline's own label arithmetic before quoting its number
**Triggers:** cross-dataset, dota, tad, frame labels, anomaly span, auc comparable, published number, protocol
**Problem:** A published AUC is defined by a *label construction* and a *score-pooling protocol*, not just a dataset name; LaGoVAD's DoTA 62.60 uses normalized spans rounded to feature length AND per-clip min-max pooling (`offline_dota_eval.py`), not the raw pooling its generic `full_length_eval.py` harness applies.
**Bad:** reading the protocol off whichever eval script you found first, then quoting the paper number against it. (This lesson originally asserted 62.60 = raw pooling; re-scoring the released checkpoint measured 0.5055 raw vs 0.6142 min-max against a published 0.6260 — the *documented* protocol was wrong for a week and put three eval arms at chance.)
**Good:** derive labels with the baseline's exact arithmetic (`round(frac * feature_length)`, half-open), verify equality against its shipped annotation file, and pin the pooling protocol by **reproducing the published number with the released checkpoint** before trusting it.
**Rule:** Treat a protocol as unverified until the baseline's own released checkpoint reproduces its published number under it; when two of the baseline's scripts disagree, the reproduction decides which one the paper used.
**Files:** core/data/dota.py:180, LaGoVAD-PreVAD/src/datasets/base.py:57, LaGoVAD-PreVAD/src/offline_evals/offline_dota_eval.py, core/docs/RESULTS_DOTA.md

### 8b — amendment (2026-08-16): when the released checkpoint *cannot* reach the published number
**Triggers:** reproduction gate, gate a, published number, chasing the paper, best.ckpt, released checkpoint, baseline parity, unreachable
**Problem:** The rule above has no exit condition, and taken literally it declares a protocol "unverified" forever when the artifact simply does not contain the published result. MSAD: we recorded the reproduction gate as *failing* at 0.8922 vs a published 0.9041 — but LaGoVAD's released `best.ckpt`, through the same eval on the same features, scores only 0.8991 (center-crop) / 0.8949 (`no_center_crop`). It cannot reach 0.9041 either. Eight paired-bootstrap comparisons of our arms against `best.ckpt` (2 transforms × 3 seeds) all have CIs including zero, with our AP consistently higher — the port was never broken. We spent two documents treating a gap in the *published artifacts* as a defect in our tree.
**Bad:** `assert our_auc >= paper_auc` — gating on a printed number, with no check that the released weights achieve it, and recording "reproduction FAILS" when they do not.
**Good:** score the released checkpoint through your own eval first; if it plateaus below the published number across every protocol variant you can construct, re-gate against **the checkpoint** (paired bootstrap, same features, same stride) and record the residual gap as a property of the artifacts — e.g. our 18,350 stride-8 sampled frames vs the paper's full frame count.
**Rule:** Gate reproduction against the released checkpoint measured under your own protocol, never against the paper's printed number, and treat a gap the released checkpoint also cannot close as an artifact difference to document rather than a defect to chase.
**Files:** core/docs/RESULTS_PHASE_A.md §4, outputs/MSAD/full_gate_a/results.json, outputs/MSAD_ncc/full_gate_a/results.json

## 9. [MEDIUM] Extraction - stream frame batches; never materialize a whole clip
**Triggers:** frames dir, jpg, stride 1, oom, batch, extract, colab, memory
**Problem:** Decoding a whole clip before encoding blows up at native frame rate: a 284-frame 720p clip is ~785 MB as uint8 and ~3 GB once `preprocess_frames` casts to float32 — an OOM on a Colab T4.
**Bad:** `frames = read_all(folder)[::stride]; pixels = preprocess_frames(frames)`
**Good:** slice the *path* list by stride, then decode → preprocess → encode `batch_size` images at a time.
**Rule:** In any frame-folder extractor, stride the file list and stream fixed-size batches; never hold a full clip in memory.
**Files:** core/tools/extract_clip_features.py:encode_frame_dir

## 10. [MEDIUM] Ingest - an existing directory is not evidence of data
**Triggers:** unzip, frames dir, colab, drive, fuse, missing videos, coverage, partial dataset, empty folder
**Problem:** A dataset unzipped over a network/FUSE mount can leave a clip's directory created but empty; counting folders reports full coverage while the extractor has nothing to read, and dropping such clips silently changes the denominator a metric is quoted on.
**Bad:** `folders = {d.name for d in root.iterdir()}; missing = ids - folders` — then `raise` (or worse, skip) on whatever is left.
**Good:** count the *files* inside each folder, classify unreadable clips by cause (absent vs. empty), and require an explicit `--allow-missing-frames` to proceed — logging `Coverage N/M (P%)` when it does.
**Rule:** Validate ingest by file count, not directory existence, and make any drop in evaluation coverage an explicit opt-in that logs the surviving denominator.
**Files:** core/data/dota.py:resolve_frame_counts, core/docs/DOTA_EVAL.md §3.1.1

## 11. [HIGH] Caches - skip-if-exists is not a resume unless writes are atomic
**Triggers:** resume, extraction, colab, disconnect, npy cache, skip existing, partial write, eta, progress
**Problem:** A long extractor that skips existing outputs treats the file the process was killed mid-`np.save` on as finished; the truncated `.npy` is never redone and surfaces hours later as a crash (or a short feature array) inside eval — and with the skip logged at DEBUG, a resumed run looks identical to a hung one.
**Bad:** `if target.exists(): continue` … `np.save(target, features)`
**Good:** write to `target.npy.part` and `Path.replace` it into position; count an existing output as done only if `np.load(mmap_mode="r")` reads back a non-empty 2-D array; log `Resume: done/total` plus a per-item ETA.
**Rule:** Make every resumable cache write atomic and verify existing outputs by reading them, never by `exists()`, and log the resume state at INFO.
**Files:** core/tools/feature_cache.py, core/tools/extract_clip_features.py:extract_frame_directory, core/flow/raft_extract.py:extract_directory

### 11b — amendment (2026-09-01): the eval writer was never fixed, and it bit
`core/tools/feature_cache.py` gave the *extractors* atomic writes; `core.evaluate --save-scores` still does a plain `np.savez`. The v3 gate-attribution campaign found `outputs/v3/DoTA_rank_s2024/eval_dota/scores/qzMjfBx1KI0_003085.npz` at **0 bytes** — one clip of 1,397, in the arm the campaign existed to test. It is worse than a truncated cache because scores are *paired*: an arm missing one clip cannot be compared to any other arm until that clip is dropped from all of them, so a single silent 0-byte file perturbs every Δ in the study rather than one number. `results.json` was written from in-memory arrays and reports all 1,397, so the run looks complete from its own summary. **Extend the atomic-write rule to every artifact writer, not just the resumable ones**, and have any offline analysis assert the score-file count against `results.json:num_videos` before computing a delta.
**Files:** core/evaluate.py (`--save-scores`), core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md §1.2

## 12. [CRITICAL] Metrics - micro AUC over an all-abnormal test set measures the wrong thing
**Triggers:** dota, auc, pooling, normalize, micro, all-abnormal, localization, concatenate scores, chance level, zero-shot eval
**Problem:** Micro AUC concatenates every video's frames into one ranking, so each clip's absolute score scale enters the metric. When the test set contains normal videos that scale is signal; on an all-abnormal set (DoTA: 1,394/1,397 clips abnormal, ~33% positive frames) the task is purely *within-clip* localization, the between-clip scale is noise, and a confident clip's negatives outrank a hesitant clip's positives. Measured cost: LaGoVAD's released checkpoint scored 0.5055 (chance) raw vs 0.6142 min-max, against a published 0.6260 — the model was fine, the metric was not. **The mirror case is worse, added 2026-09-06:** when the test set is *dominated by all-normal clips* the between-clip scale is not merely signal, it is the whole metric. DADA-2000's test split is 74% frames from `0_Normal_Driving` clips, so a model emitting one **constant score per clip** — zero localization — scores micro AUC **0.9069**; our best arm reached 0.8739 with `auc_macro` at chance (0.44-0.57 across seven arms). A micro number that high reads as a frame-level result and is a video-classification result.
**Bad:** `scores_cat = np.concatenate(all_scores); roc_auc_score(labels_cat, scores_cat)` for every dataset.
**Good:** normalize each video's curve (min-max) before pooling when the test set is effectively all-abnormal; report `auc_macro` (mean per-video AUC, needs no normalization) and `auc_raw` alongside so the protocols stay auditable.
**Rule:** Decide score pooling from the *label distribution* of the test set, not the dataset name — normalize per video whenever normal videos fall below 5% of it, always report the macro per-video AUC beside the micro one, and where all-normal clips carry a large share of the frames also compute the constant-score-per-clip oracle and print it beside the micro number.
**Files:** core/metrics.py:resolve_score_norm, core/evaluate.py, core/tools/rescore.py, core/constants.py:SCORE_NORM_AUTO_NORMAL_FRACTION, core/docs/v3/RESULTS_DADA.md:96

## 13. [HIGH] Transform - a checkpoint is bound to the preprocessing it was trained under
**Triggers:** center crop, no_center_crop, transform, field of view, resize, feature cache, cross-model eval, released checkpoint, raft, kip
**Problem:** Feeding a checkpoint features built with a different image transform is a silent train/test mismatch — no error, just a degraded number that looks like a modelling result. Worse when it is *internal*: `preprocess_for_raft` is full-frame while `preprocess_frames` center-crops, so KIP was trained to regress motion evidence from the whole frame using CLIP features that only saw its middle 56% — on dashcam footage that is exactly where lateral motion lives.
**Bad:** one global CLIP cache reused for every checkpoint, and a flow branch whose field of view is chosen independently of the appearance branch.
**Good:** one cache directory per transform (`clip/DoTA_s8` vs `clip/DoTA_ncc_s8`), each checkpoint scored only on the transform it was trained with, and the flow/appearance transforms chosen together.
**Rule:** Pair every checkpoint with the transform that built its training features, give each transform its own cache path, and keep all branches of one model on the same field of view.
**Files:** core/tools/extract_clip_features.py:preprocess_frames, core/flow/raft_extract.py:preprocess_for_raft, LaGoVAD-PreVAD/src/utils/video_loader.py:79

## 14. [CRITICAL] Experiments - an ablation run under a known-open precondition defect measures the defect
**Triggers:** ablation, a/b, kip on off, negative result, confound, precondition, known defect, delta, verdict, blocked
**Problem:** KIP's DoTA A/B was run and recorded at Δ = −0.0329 with a CI excluding zero, while lesson 13 was already open and known: the appearance branch was center-cropped while the flow target was full-frame, so KIP was regressing motion evidence absent from its own input. Re-running the identical A/B after removing the crop gave Δ = +0.0911 — the sign flipped and the magnitude tripled. Crop→ncc moved KIP-on by +0.1514 macro AUC and KIP-off by +0.0077, so the recorded "negative result" was a measurement of the defect, not of the module.
**Bad:** record a signed, CI-bounded ablation verdict, caveat it in prose ("not a verdict on KIP"), and schedule the precondition fix for afterwards.
**Good:** close known precondition defects on the module under test first; until then run the reference arm only and report the A/B as blocked, with no number.
**Rule:** Fix every open defect on a module's own inputs before running or recording its ablation, and treat a signed delta measured under one as unmeasured.
**Files:** core/docs/RESULTS_NCC.md §2, core/docs/RESULTS_DOTA.md §5, .project/memory-bank/lessons-learned/index.md#13

## 15. [HIGH] Checkpoints - a pickled library object makes the artifact expire with its environment
**Triggers:** torch.load, _reconstruct, numpy version, rng state, checkpoint portability, colab, unpickle, weights_only, resume
**Problem:** `save_checkpoint` stores `rng["numpy"] = np.random.get_state()`, a tuple wrapping a 624-element uint32 **numpy array** — the only numpy object in the payload. A pickle is one stream, so when the loading VM's numpy major version differs from the saving one, `torch.load(..., weights_only=False)` aborts with `TypeError: _reconstruct: First argument must be a sub-type of ndarray` *before reaching a single tensor*. Observed 2026-08-16: every stage-1 and `checkpoint_step_*.pt` on Drive became unloadable after a Colab runtime drifted, blocking arm 4 and the trajectory probe on checkpoints that were themselves intact.
**Bad:** `torch.save({..., "rng": {"numpy": np.random.get_state(), ...}}, path)` — a resume convenience welded onto the only artifact that has to outlive its environment; and "fix" it by pinning `numpy<2`, which breaks a torch wheel built on the numpy 2 ABI.
**Good:** store version-fragile state as raw bytes (`np.random.get_state()[1].tobytes()`) or omit it, and keep the weights reachable on their own; to recover existing artifacts, load through a `pickle.Unpickler` whose `find_class` refuses `multiarray._reconstruct`, then re-save `{"model": ...}` slim — tensors travel `persistent_load` and never reach `find_class`, so the recovery is lossless.
**Rule:** Never pickle a third-party library's internal state object into a checkpoint; serialize it to bytes or leave it out, so the weights stay loadable under any version of that library.
**Files:** core/train.py:389 (`_rng_payload`), core/train.py:408-425 (`save_checkpoint`), core/train.py:94, core/inference.py:62, core/docs/COLAB.md §A4.0

## 16. [MEDIUM] Experiments - step-uniform checkpoints undersample the loss range a trajectory probe needs
**Triggers:** checkpoint_every_steps, trajectory probe, convergence, matched loss, under-convergence, regularizer, h3, step checkpoint, diagnostic
**Problem:** `checkpoint_every_steps=100` over a 500-step run reads like "five samples across training". It is not: under warmup + cosine decay the train `mil` falls 0.63 -> 0.041 inside the first 100 steps, so **94 % of the loss range lies before the first checkpoint** and the five saved models all sit in the last 6 %. When the A5 probe needed KIP-off scored at a *matched* convergence level, the low-`mil` end was bracketed by luck and the high-`mil` end was unmeasurable without retraining.
**Bad:** `--set train.checkpoint_every_steps=100` on a 500-step run, then trying to plot a metric against train loss.
**Good:** pick the interval from the *loss* trajectory, not the step count — dense early (every 10-25 steps through the first epochs), sparse once the loss plateaus; or save on loss-decade crossings. Five checkpoints spread over `mil` 0.6 / 0.2 / 0.05 / 0.01 / 0.002 answer questions that five spread over 0.041 / 0.003 / 0.0016 / 0.0012 / 0.0013 cannot.
**Rule:** Choose checkpoint spacing so the training loss is sampled roughly geometrically over the run, not the step count uniformly, whenever those checkpoints may have to answer a convergence-matched question.
**Files:** core/train.py (`train.checkpoint_every_steps`), core/docs/COLAB.md §A5.1, core/docs/RESULTS_ARM4_PROBE.md §4.2

## 17. [HIGH] Experiments - a run's output must record every flag that defines what the run is
**Triggers:** init-weights, warm start, provenance, config.yaml, run manifest, arm, control, audit, reproduce, argv
**Problem:** `core/train.py:628` writes `cfg.save_yaml(output_dir/"config.yaml")`, and `cfg` holds only the config tree. The CLI arguments that decide *what an arm is* — `--init-weights`, `--data-dir`, `--clip-dir`, `--flow-dir`, `--knn-cache`, `--resume` — are argparse-only and are never written anywhere. `stage2_kip_off_warm/config.yaml` is therefore **byte-identical** to `stage2_kip_off/config.yaml`, even though one is warm-started from stage 1 and the other is cold: the single fact that arm 4 exists to establish is absent from arm 4's own output directory. The warm start had to be inferred forensically from step-1 train `mil` (0.70148 next to KIP-on's 0.70146, against cold 0.86818). An arm whose defining flag is unrecorded is unauditable, and a mis-typed path would have produced a plausible run that silently answers a different question.
**Bad:** `cfg.save_yaml(args.output_dir / "config.yaml")` as the only provenance a run writes, with paths and warm-start sources living in the shell loop of a Colab cell.
**Good:** also write a `run_manifest.json` next to it holding the resolved `--init-weights` (with its resolved path, `global_step` and a weight hash), every data/cache directory, the git commit, and `sys.argv`; assert it exists when analysing an arm.
**Rule:** Write every CLI argument that changes what a run *is* into a manifest inside that run's own output directory, never only into the config tree or the launcher script.
**Files:** core/train.py:563-628, core/docs/RESULTS_ARM4_PROBE.md §1

## 18. [CRITICAL] Data - a shipped label format is defined by its widest case, not its common case
**Triggers:** anomaly span, multi-span, frame labels, dataset adapter, preprocessor, prevad, span list, clamp, ground truth
**Problem:** `core/data/dota.py:232 sampled_frame_labels` fills exactly one `[start, end)` window, because every DoTA clip has exactly one. PreVAD ships the *same* normalized-fraction format, and 104 of its 1,306 abnormal test clips carry **2-4 disjoint windows**; the shipped table also contains **440 spans that end past 1.0** (max 1.2104) and **one reversed span** (`RrUW8ITUqx0_aug3`, end < start). Porting the DoTA function unchanged would take `spans[0]` or the hull: on the real 4-span clip `qhly283_BV1Lf4y117wS` that turns 40 positive rows into 66, silently labelling the normal gaps between windows abnormal. Nothing raises — the AUC just measures the wrong ground truth.
**Bad:** `start, end = record.span` then one `for i in range(start, end)` — one window per clip, bounds trusted, arity never checked.
**Good:** loop `for start_frac, end_frac in record.spans`, clamp each to `[0, L]` (`max(0, min(length, round(frac * length)))`), fill half-open, and **count** every span the clamp absorbed — overflow, reversed, and rounded-away — in the run log.
**Rule:** Derive a label builder's arity and bounds from the shipped annotation's widest observed case, and log every span the clamp changes instead of repairing it silently.
**Files:** core/data/prevad.py:sampled_frame_labels, core/data/prevad.py:build_frame_labels, core/data/dota.py:232, core/tests/test_prevad.py

## 19. [HIGH] Definitions - a graceful name-lookup fallback hides a missing class taxonomy
**Triggers:** definitions, class names, taxonomy, verbalizer, defs.json, class_index_tensor, prevad, conditioning, DEFAULT_CLASSES
**Problem:** `_PREVAD_CLS_DEFS` held the **22 CamelCase** names from the block that is *commented out* in `LaGoVAD-PreVAD/src/datasets/PreVAD.py`; the live `DEFAULT_CLASSES` right below it is **36 space-separated** names, and those are what the release CSVs actually use. 28 of 35 observed classes had no definition. This does not raise: `verbalize_class_name` (`core/data/definitions.py:342-346`) returns the bare class name for an unknown class *by design*, so the model trains on `"Store Robbery"` instead of a definition sentence and LaGoVAD's whole definition-conditioning claim is void with no error anywhere. `PREVAD_SETUP.md` §3 predicted a crash; there is none, which is worse.
**Bad:** relying on `verbalize_class_name`'s fallback, or on `class_index_tensor` raising, to catch a class list that does not match the data — neither fires when `defs.json` is generated from the same table.
**Good:** `check_definition_coverage(records)` in the preprocessor: `sorted({r.class_name for r in records} - set(DATASET_CLS_DEFS[key]))` must be empty or it raises, naming the classes and the file to edit; classes defined but unobserved are logged and dropped from `defs.json`.
**Rule:** Validate that every class observed in the data has definition sentences at preprocessing time, and never let a name-lookup fallback stand in for a missing definition.
**Files:** core/data/prevad.py:check_definition_coverage, core/data/definitions.py:194, LaGoVAD-PreVAD/src/datasets/PreVAD.py:37

## 20. [HIGH] Runbook - a dataset-sized glob is not an argument list
**Triggers:** unzip, flatten, mv, cp, rm, glob, argument list too long, ARG_MAX, colab, runbook, shell cell, 35279, bulk file move
**Problem:** `PREVAD_SETUP.md` §4.3 flattened the archive's top-level directory with `mv "$PREVAD_CLIP/ViT-B-16-8p-features/"* "$PREVAD_CLIP/"`. The glob expands to **35,279 paths** and `execve` refuses the whole command: `bash: line 15: /usr/bin/mv: Argument list too long`. Under `set -euo pipefail` the cell died *after* a successful 14.5 s extraction, leaving the features one directory deeper than every later cell expects — a half-done setup that reads as a broken download. The same line is correct and untested on the 1,403-clip DoTA tree; it only fails once the dataset is big enough, which is exactly when the runbook is being used for real.
**Bad:** `mv "$DIR/prefix/"* "$DIR/"` — one process, one argument per file, capped by `ARG_MAX` (~2 MB of argv on Linux).
**Good:** never create the prefix in the first place — `unzip -n -j -q "$ZIP" -d "$DIR"` junks paths during extraction. Where a bulk operation is genuinely needed, move the *directory* (`mv dir tmp && mv tmp dir`) or pipe through `find … -print0 | xargs -0`, never a shell glob.
**Rule:** Never expand a dataset-sized glob into a command's argument list; make the tool that writes the paths produce the layout you want, or operate on the containing directory.
**Files:** core/docs/PREVAD_SETUP.md:§4.3

## 21. [HIGH] Env - a batch pip install is atomic, and a failed !pip does not stop the notebook
**Triggers:** pip install, colab, environment, pin, torch version, python 3.13, transformers, cp313, wheel, ModuleNotFoundError, AttributeError, dependency, notebook cell
**Problem:** `collab/PreVAD/evaluate.py` installed the full `pyproject.toml` pin set in one cell. `torch==2.4.*` has no cp313 wheels and Colab is on Python 3.13, so pip's resolver failed the whole set and installed **none** of the fourteen packages — not `transformers==4.56.*`, not `numpy<2`. `!pip` returns non-zero but does not raise, so the notebook ran on for three more cells. The symptom surfaced far away as `ModuleNotFoundError: No module named 'faiss'` and then `AttributeError: 'CLIPTextModel' object has no attribute 'text_model'` — an API break inside a frozen encoder, which reads as a code bug, not an install that never happened.
**Bad:** `!pip install -q "torch==2.4.*" "transformers==4.56.*" "numpy<2" ...` in one cell, then trusting that the pins are in effect because the cell "ran".
**Good:** install only what the runtime can satisfy (drop `torch`/`torchvision` on py3.13; drop `numpy<2` because Colab's torch is built on the numpy 2 ABI), then **assert** the versions in a following cell — `print(transformers.__version__)` plus a positive check of the API actually used (`'text_model' in CLIPTextModel.__init__.__code__.co_names`).
**Rule:** Verify an environment by importing and asserting versions after installing, and never batch an unsatisfiable pin with the pins you need.
**Files:** collab/PreVAD/evaluate.py, core/docs/PREVAD_SETUP.md:§4.0.1, pyproject.toml:7-22, core/models/clip_text.py:115

## 22. [MEDIUM] Metrics - a metric's dtype is part of its protocol
**Triggers:** float32, float64, dtype, precision, normalize_scores, minmax, rescore, ties, saturation, AUC, AP, pooling, results.json
**Problem:** `core.evaluate` min-max normalizes score curves in **float32** (the dtype `.npz` stores) while `core.tools.rescore` casts to float64 on load (`core/tools/rescore.py:57`), so the two tools return different numbers from the *same* files: `DoTA_ncc_pv_s2024/eval_kip_off` is AUC 0.586406 / **AP 0.369953** through `evaluate` and 0.585956 / **AP 0.366069** through `rescore`. `auc_raw`/`ap_raw` agree to the last digit — the divergence is created *by* the normalization, where float32 rounding manufactures ties among the 38-87 % of frames sitting above 0.99 and sklearn splits tie credit. ΔAP = 0.0039 is the same order as several reported deltas, so a table mixing a rescored arm with a non-rescored one is silently wrong.
**Bad:** `scaled = (scores - low) / (high - low + eps)` on whatever dtype arrived, with one tool loading float32 and another float64.
**Good:** cast to `np.float64` once at the entry of `normalize_scores` (and in `core/evaluate.py` before pooling), so every consumer of a score curve computes the identical number; record the dtype in `results.json` if it is ever allowed to vary.
**Rule:** Cast score curves to float64 before any pooling arithmetic, and make every tool that reads the same `.npz` produce the identical metric.
**Files:** core/metrics.py:normalize_scores, core/tools/rescore.py:57, core/evaluate.py, core/docs/RESULTS_PREVAD.md §10.1

### 22b — amendment (2026-09-01): reproduce the shipped number *exactly* before you trust a recompute
Second measured instance, and a method. The v3 campaign recomputed every arm offline and got 0.642312 / AP 0.421551 where `results.json` said 0.642111 / 0.419967 (A2 s2024) — a ΔAP of 0.0016. Three plausible causes were live at once: the float32/float64 split (C22), the `+SCORE_NORM_EPS` in `normalize_scores`' denominator, and one clip dropped for a truncated file. Guessing among them would have mislabelled the table. What settled it was **reproducing `evaluate`'s output bit-for-bit first** — float32 division including the eps gives 0.642111 / 0.419967, to the last digit — and only then changing one factor at a time. The eps turned out to be irrelevant (1e-12 rounds away against a float32 range of O(1)); pure float32 rounding is the whole effect, so C22's diagnosis is confirmed against a second dataset and arm. Also found: dropping the eps is not free — `W6YrlYyWguc_005367` has a **flat** score curve, so `(s-min)/(max-min)` is `0/0` and sklearn raises on the NaN. **When an offline recompute disagrees with the tool, reproduce the tool's exact number first, then vary one factor at a time** — and branch on `hi > lo` rather than relying on an eps you may have dropped.
**Files:** core/metrics.py:normalize_scores, core/constants.py:SCORE_NORM_EPS, core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md §1.1

## 23. [MEDIUM] Data - an id that survives a filesystem round-trip is not the same id
**Triggers:** video_id, filename, scores/*.npz, join, metadata, colon, path-hostile, prevad, per-class breakdown, stem
**Problem:** 90 of PreVAD's 2,606 test ids contain `:` (`2_KyluTt9Rk_00:13:32.467_00:13:42.733`). `--save-scores` writes them as `..._00_13_32.467_...npz`. Nothing is lost — every clip is scored, and micro/macro metrics are correct — but a per-class analysis that joins `scores/*.npz` stems against `test.csv:video_id` silently drops **3.5 % of the test set** into an unlabelled bucket, with no error and no warning. It produced a wrong per-superclass table here before the `?` bucket was noticed.
**Bad:** `meta[path.stem]` / `{r['video_id']: ...}[stem]` with a `.get(..., '?')` fallback that quietly absorbs the misses.
**Good:** key the metadata by the same transform the writer applied (`video_id.replace(":", "_")`), and **assert** every score-file stem resolves — `assert all(i in meta for i in ids)` — instead of defaulting.
**Rule:** Join score files to metadata through the writer's own id transform, and assert full coverage rather than defaulting unmatched ids.
**Files:** core/evaluate.py (score filename), PreVAD/test.csv, core/docs/RESULTS_PREVAD.md §10.2

## 24. [CRITICAL] Architecture - a non-differentiable op on the score path silently freezes everything upstream of it
**Triggers:** floor, round, argsort, argmax, topk, .long(), slice index, gate, shift, straight-through, STE, detach, frozen, adaptive, motion-gated, learned
**Problem:** KIP's shift count was `s_t = (sigmoid(MLP(m)) * max_shift).floor().long()`, used as a slice index. `floor` has zero gradient and an integer index has none at all, so the 321-parameter gate MLP — and, through it, the PMG head — **never received a single gradient**. Nothing raised: the module has parameters, the optimizer accepts them, every loss falls, `ruff`/`mypy`/`pyright` are clean, and the model trains to a publishable number. The gate stayed at random init, so **init was the deployed function**. Worse, it was measurably degenerate: because its input is min-max normalized, `[0,1]` is the whole reachable domain, and sweeping it moved `s_t` by **0-4 channels out of 128** across 8 seeds (seed 0: exactly 0). Three campaigns and a +0.0915 AUC result were reported as "motion-gated adaptive temporal mixing" when the module was, in operation, a **fixed ~50 % temporal smoother**. The correct control (a constant-ratio TSM) was never run because nobody knew that was what had been built.
**Bad:** `s = (ratio * max_shift).floor().long(); vk = torch.where(channel < s, past, vt)` behind a trainable `nn.Sequential`, with no test asserting the MLP learns anything. Also bad: the "obvious" STE fix `s = u + (u.floor() - u).detach()` — it gives `s` a gradient, but `s` is consumed only inside `channel < s`, and a comparison emits a boolean that no gradient flows through, so the fix is a **no-op that looks correct**.
**Good:** assert the gradient exists for anything you call learned - `loss.backward(); assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters())` - as a test, not a review note. When a hard decision genuinely belongs on the path, either make it parameter-free and say so (KIP's v3 rank gate: 0 params, deterministic, loggable) or apply the straight-through estimator to the **selection weights** that multiply the tensors, not to the index. And log the decision variable at eval: a gate whose output never moves is visible in one histogram.
**Rule:** Assert a non-zero gradient on every module you describe as learned, and log the output of any hard/rounded decision variable on the score path.
**Files:** core/kip/gate_shift.py (`shift_channels_straight_through`, `KinematicShift.gate_ratio`), core/tests/test_kip_gate_types.py::TestGradientTopologyUnchanged, core/docs/REPORT_KIP_MSAD_DOTA_PREVAD.md §12.5, .project/plans/katvad-v3-kip-gate-rebuild.md Appendix C


## 25. [MEDIUM] Extraction - a second source for an existing cache must be proven bit-identical, not merely equivalent
**Triggers:** frames-dir, videos-dir, second reader, extraction path, contiguous, ascontiguousarray, permute, np.array_equal, cache parity, flow cache, e_O, TAD, read_images, source-independent
**Problem:** TAD ships frames, MSAD ships videos, and `cache/flow/v1/{DATASET}/` is supposed to mean the same thing either way — that is the premise of every cross-corpus Δ. Adding a `--frames-dir` path to `raft_extract` produced `e_O` that differed from the video path by **2.67e-5 on bit-identical pixels**. The frames were not the problem: `np.array_equal(video_frames, folder_frames)` was `True`, and `preprocess_for_raft` output was bit-identical on both. The difference was **memory layout** — `read_images` returns a *permuted view* (`torch.stack(...).permute(0,2,3,1).numpy()`, non-contiguous) while `read_sampled_frames` returns a contiguous `np.stack`, and torch's batched convolutions dispatch to a different kernel on a non-contiguous input. Nothing raises, `ruff`/`mypy`/`pyright` are clean, and the magnitude is far below anything that moves an AUC — which is exactly why it would have survived: a flow cache that silently depends on which reader built it is not comparable to the cache every v1 number was measured against, and no test would ever have said so.
**Bad:** `return read_images(paths[::stride])` — a non-contiguous buffer handed to the model — plus a test that asserts only `shape`, `dtype` and "it ran".
**Good:** `return np.ascontiguousarray(read_images(paths[::stride]))`, and a test that dumps one video's frames to PNG, extracts both ways, and asserts `np.array_equal` on **both** `e_O` and the raw stats. Equality of the *inputs* is not evidence: assert equality of the *artifact*.
**Rule:** When adding a second source path to an existing cache, assert the new path is bit-identical to the old one on the same content, and make any array you hand to a model C-contiguous.
**Files:** core/data/video_io.py:read_sampled_frames_from_dir, core/flow/raft_extract.py:_extract_sources, core/tests/test_extractors.py::TestRaftFrameExtraction::test_matches_the_video_path_exactly


## 26. [HIGH] Data — a bare folder name is not a video id until proven unique across every directory a scan will traverse
**Triggers:** video_id_from_path, list_frame_folders, frames-dir, frame folder, category directory, fault-attribution, flat dir, symlink farm, id collision, dict comprehension, {video_id: path}
**Problem:** DADA-2000 ships `type<N>_vid<N>` frame folders under three category directories (`0_Non_Ego_Fault`, `1_Ego_Fault`, `0_Normal_Driving`); the same `type<N>_vid<N>` string recurs across all three (confirmed on the real archive: `type10_vid001` exists in both `0_Non_Ego_Fault` and `0_Normal_Driving`). `dada.py` used `video_id_from_path(folder)` — bare `Path.name` — as `video_id`, and `_assert_no_id_collisions` caught the first offender at ingest and raised. That assert was the *visible* half of the defect: `extract_clip_features.py:194` and `raft_extract.py:291` build `{video_id_from_path(p): p for p in list_frame_folders(frames_dir, subdir)}` — a **plain dict, no collision check** — over the same `--frames-dir`. Had the ingest-side assert not existed (or been bypassed), the extractors would have silently overwritten one colliding clip's folder with the other's in the id→path map and extracted the wrong pixels under a shared filename — no exception, no warning, a plausible-looking `.npy` trained under the wrong label.
**Bad:** `video_id = folder.name` (or `video_id_from_path(folder)`) trusted as globally unique for any dataset with more than one root directory feeding the same `--frames-dir` scan, with the id used unchecked as a dict key or output filename downstream.
**Good:** derive `video_id` from every dimension the archive actually uses to disambiguate clips (`{fault_dirname}__{folder_name}` here), not just the leaf folder name; where a shared, dataset-generic tool (`extract_clip_features.py`/`raft_extract.py`) cannot be taught the new id scheme without risking other datasets' caches, materialize a flat directory of symlinks named by the disambiguated id and point `--frames-dir` at that instead of the raw multi-category root.
**Rule:** Before trusting a frame-folder scan's bare-name id as globally unique, verify uniqueness across every subdirectory the scan will actually traverse; if it is not unique, disambiguate the id and route shared extraction tools through a flat, correctly-named directory rather than teaching them the dataset's internal layout.
**Files:** core/data/dada.py:_make_video_id, core/data/dada.py:materialize_flat_dir, core/tools/extract_clip_features.py:194, core/flow/raft_extract.py:291, core/tests/test_dada.py::TestFlatFramesDir

## 27. [HIGH] Architecture — a score head whose kernel spans the clip is a clip classifier, not a frame detector
**Triggers:** score_head_kernel, ConvScoreHead, Conv1d, receptive field, kernel_size, sequence length, short clips, frame_stride, auc_macro, flat score curve, new dataset, DADA, clip length
**Problem:** `H_bin` is a single `Conv1d(512 -> 1, kernel_size=9, padding=4, padding_mode="replicate")`. Its receptive field is 9 timesteps, chosen against MSAD, whose median clip is **86** stride-8 frames (kernel covers 10%). DADA-2000's median clip is **9** (55% of clips are <= 9), so on the median clip *every output timestep sees every input frame* and adjacent timesteps differ only by a fully-overlapping window slide. The head degenerates into clip pooling. Nothing raises: the model trains, the loss falls faster than the KIP-off arm's (epoch-19 mean `mil` 0.255 vs 0.490), and the reported micro AUC goes **up** (0.86) while `auc_macro` sits at chance (0.44-0.57) — measured as a between/within-clip score-variance ratio of 19-48 and a median within-clip score range of 0.05-0.12 on [0,1]. The supervision degenerates with it: `_topk_k` is `max(1, n // topk_pct)`, so `L_MIL` on a 9-frame clip is a plain max over 9. Downstream, the arm transfers as a clip classifier and collapses to chance under DoTA's per-clip min-max protocol: Spearman(flatness, DoTA micro AUC) = **-0.82** across seven arms.
**Bad:** porting `score_head_kernel=9` (or any fixed temporal kernel, window or top-k) unchanged onto a new corpus, then reading its micro AUC as a frame-level localization result.
**Good:** measure the corpus's **median** sequence length at the configured stride before training; keep the kernel a small fraction of it (or shrink the stride), and gate the run on `auc_macro` rather than micro AUC.
**Rule:** Check every temporal receptive field on the score path — `score_head_kernel`, `temporal_window`, MIL top-k — against the new corpus's median sequence length before training on it, and refuse to read a micro AUC as localization until `auc_macro` clears chance.
**Enforcement (added 2026-09-06):** run `python -m core.tools.eda report --dataset <NAME> --data-dir <DIR>` before the first arm on any corpus and paste its §0 verdict table into the campaign runbook. It computes the coverage fraction, the MIL `k = 1` share and the clip-level oracle from the label files alone, in seconds, with no GPU — the whole of this lesson was measurable before the campaign that discovered it. Thresholds live in `core/eda/report.py`; runbook in `core/docs/EDA.md`.
**Files:** core/models/heads.py:21, core/losses/mil.py:22, core/config.py:52, core/docs/v3/RESULTS_DADA.md:135, core/eda/corpus.py:kernel_coverage, core/eda/corpus.py:mil_topk_floor
