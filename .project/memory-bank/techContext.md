# Tech Context — stack, setup, constraints

**Created:** 2026-07-31 (re-init from `b9978ff`, read off `pyproject.toml`)
**Last reviewed:** 2026-09-15 (later — doc count 22→23 for
`DADA_ORIGIN_PHASE0.md`, and the DADA-original sizing constraints added; earlier
the same day, counts/test status re-measured and the text-branch + `L_neg`
constraints added)

> **Branch `main` = KAT-VAD v1.** The stack, pins and constraints below are
> branch-independent. Where a **path or flag** is v3-only it is marked.

## Stack

- **Python 3.10** (`.python-version` = 3.10.6; avoid 3.11-only syntax)
- **torch 2.4.\*** / **torchvision 0.19.\*** / **transformers 4.56.\*** / **numpy <2**
- faiss-cpu, scikit-learn, torchmetrics, einops, opencv-python, **PyAV** (decord
  has no macOS arm64 wheel), gdown, matplotlib, pyyaml, tqdm
- Package manager **uv**; build backend hatchling; `core` is the only wheel package
- Dev group: pytest, ruff, mypy, bandit, pycycle, pyright

## Compute

- **Dev:** macOS arm64, CPU. All code and all tests run data-free on CPU — on
  `main`, **514 collected, 514 pass** (measured 2026-09-15, three clean runs;
  green since 2026-09-08 when the v3-only `gate_type` matrix was collapsed to the
  v1 gate; see [[progress]]). `v3` is 537 green.
- **Training:** Google Colab **A100 40 GB**. Long jobs are resumable and
  Drive-persisted; AMP and grad-accumulation are config flags.

## Key file locations

| Path | Contents |
|---|---|
| `core/constants.py` | roots, data-layout contract, all baseline hyperparameters |
| `core/config.py` | `Config` dataclasses, YAML load, `--set section.key=value` |
| `core/train.py` (633 L) | training loop, stage selection, resume |
| `core/evaluate.py` / `core/inference.py` | sliding-window scoring, metrics |
| `core/models/` | temporal encoder, fusion, heads, clip_text, ckpt_compat, kat_vad |
| `core/kip/` | pmg, **gate_shift (v1 `KinematicShift`, frozen MLP gate — one gate type)**, motion_head, kip_module, losses. **`ecmr.py` and the 4 gate types are branch `v3` only** |
| `core/losses/` | mil, dvs, contrastive (baseline losses) |
| `core/metrics.py` | pooling rules, micro/macro AUC + AP; torch-free so `rescore` can import it |
| `core/data/` | msad, dota, prevad, **tad**, **dada**, dataset (DVS), synthesis, knn_cache, collate, definitions, video_io, dataset_files |
| `core/flow/raft_extract.py` | RAFT → 23-d stats → seeded 256-d projection |
| `core/tools/` | download, extract_clip_features, **feature_cache**, **rescore**, visualize, **eda** |
| `core/eda/` | corpus, labels, protocol, features, report — the pre-flight profiler (`python -m core.tools.eda report\|compare`), runbook `core/docs/EDA.md`. **Never run on real data yet** |
| `core/docs/` | **20 files on `main`**: COLAB, DATA_LAYOUT, DOTA_EVAL, **EDA**, PREVAD_SETUP, **TAD_SETUP**, **DADA_SETUP**, **DADA_ORIGIN_PHASE0** (new 2026-09-15), DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE, TRAINING, proposal, spec, 7 × RESULTS_\*.md, REPORT_KIP_MSAD_DOTA_PREVAD |
| `core/docs/v3/` | **On `main`: only `RESULTS_DADA.md` + `setup/{DADA_V3_SETUP,TAD_V3_SETUP}.md`** (3 files). ARCHITECTURE, spec_v3, audit_addendum_PreVAD, RESULTS_V3_GATE_ATTRIBUTION and `setup/MSAD_DOTA_V3_SETUP.md` are **branch `v3` only**. `core/docs/v2/` does not exist on either branch — do not cite it |

**Measured on `main`, 2026-09-15:** **81 Python files** (55 source + 26 test),
**11,756 LOC** source. **23 markdown docs** under `core/docs/**` (20 top-level + 3 under `v3/`). `outputs/**`
holds **87,213** `.npz` score files. (`v3` measures 84 files / 11,157 source LOC /
537 tests — different tree, different numbers.)

## Environment roots

`KATVAD_DATA_ROOT` (`./data`), `KATVAD_CACHE_ROOT` (`./cache`),
`KATVAD_CKPT_ROOT` (`./ckpts`), `KATVAD_OUTPUT_ROOT` (`./outputs`).
Colab points all four at Drive. Full contract: `core/docs/DATA_LAYOUT.md`.

## Pinned externals

- CLIP `openai/clip-vit-base-patch16`, **revision pinned** to
  `5ef227a78de3f75873f373246dac80def63b0003` — `main` ships only
  `pytorch_model.bin`, which transformers 4.56 refuses to `torch.load` on
  torch < 2.6 (CVE-2025-32434). The pinned sha is the safetensors conversion.
- RAFT `Raft_Large_Weights.C_T_SKHT_V2` (torchvision); inputs must be ≥128 px and
  divisible by 8 → extraction runs at 240×320.
- LaGoVAD `best.ckpt` via gdown, id in `constants.LAGOVAD_BEST_CKPT_GDRIVE_ID`.

## Constraints learned in-tree

- **PreVAD ships no pixels, so it can never host a KIP-on arm.** The release is
  CLIP features only (`PreVAD/features/ViT-B-16-8p-features.zip`, 35,279 `.npy`).
  `L_KIP_rec` regresses a cached **RAFT** embedding, and RAFT needs frames.
  Re-downloading from `annotations/data_sources.csv` yields 50–70 % after link
  rot, permanently loses the 3,800 highway-camera clips KIP most cares about,
  and cannot be paired with the released features (transcode/fps/length drift →
  `core/data/dataset.py:112` raises). `require_flow = cfg.kip.enabled` makes it a
  hard stop; **never** set `require_flow=False` to get past it — that zero-fills
  `e_O` and trains the PMG head to predict zeros. Settled by the user 2026-08-29.
  PreVAD's permanent role: **KIP-off pretraining trunk + in-domain gate (P0)**.
  Full reasoning: `PREVAD_SETUP.md` §7.4. **Generalises:** before proposing a
  KIP-on arm on any new dataset, confirm it ships raw video.
- **faiss must be single-threaded** — torch + faiss libomp clash segfaults on
  macOS arm64 (`core/data/knn_cache.py`).
- **torch 2.4 MPS training diverges** — stage-1 `L_KIP_rec` climbs on MPS while
  CPU converges on identical seeds. Pin `train.device=cpu` locally; CUDA on Colab.
- Resume matches a straight run **within FP tolerance**, not bitwise — Apple
  Accelerate BLAS reduces nondeterministically regardless of `set_num_threads`.
- **The local numpy pin and the Colab runtime disagree, by necessity.**
  `pyproject.toml` pins `numpy<2`; Colab's torch wheel is built on the numpy 2
  ABI, so pinning `numpy<2` there breaks the runtime (`_ARRAY_API not found`).
  That drift is what makes checkpoints expire — see below. Do not "fix" it by
  downgrading Colab.
- **Checkpoints are not portable across a numpy major version.**
  `save_checkpoint` pickles `np.random.get_state()`, and a pickle is one stream,
  so `torch.load(weights_only=False)` aborts with `TypeError: _reconstruct: ...`
  *before reaching any tensor*. Recovery: load through a `pickle.Unpickler`
  whose `find_class` refuses `multiarray._reconstruct`, re-save `{"model": ...}`
  slim (`COLAB.md` §A4.0). Lossless — tensors travel `persistent_load`. Lesson
  15; the real fix (serialize the RNG state as bytes, or omit it) is deferred
  while experiment arms are being compared.
- **A run's `config.yaml` does not record its CLI paths or `--init-weights`**
  (`core/train.py:628` saves the config tree only). Arm provenance must be
  reconstructed from the loss trace. Lesson 17. **This bit for real on
  2026-09-15:** `outputs/v1/TAD/2024/t2_warm/stage2/config.yaml` is *byte-identical*
  to the cold `m0`'s, so the one fact the arm exists to establish — that it was
  warm-started — survives nowhere. Worse, the loss trace is a weak witness here:
  step-1 `mil` was 0.8189 vs the cold arm's 0.7999, because `auc_macro` is
  rank-based while `mil` is BCE and calibration-sensitive — a trunk can rank well
  and still score BCE ≈ 0.8 on a new corpus's bag distribution. **Write a run
  manifest** (`--init-weights`, resolved data/cache paths, git sha, `sys.argv`).

- **Two different text inputs exist, and only one comes from the dataset.**
  (1) **Class definitions → `z^t`.** Hardcoded in `core/data/definitions.py`
  (`DATASET_CLS_DEFS`), *not* read from any annotation file; `DatasetSpecVerbalizer`
  samples one of N sentences per class per item, seeded from the item id (C30).
  TAD's coverage is 2/2 (`Normal`, `Car Accident` → 4 sentences each) so **no C19
  fallback fires** — but C = 2 means the conditioning surface is a single bit.
  (2) **`descriptions` → per-video captions → `L_neg`.** Only PreVAD's release
  ships the field (`core/data/prevad.py:172`); it reaches `meta.json` only, never
  `labels_train.json`, and **nothing reads it back** — gap **G4**, "deliberately
  not wired yet" (`prevad.py:441`). A `descriptions: null` in a TAD/DoTA/DADA/MSAD
  annotation is expected and harmless.
- **`L_neg` has effectively never been computed.** Audit of all 55 `config.yaml`
  under `outputs/` (2026-09-15): **54 carry `captions_from_definitions: false`**,
  so `captions = None` → `caption_feats = None` → `core/train.py:375` skips the
  term; `cap_contrastive_weight: 1.0` in all of them is decoration. The single
  exception is `outputs/v1/PreVAD/stage2_kip_off`, and even it **fabricated**
  captions from the class definitions. Mechanically `L_neg` is *not* a clip-level
  loss — `attn = softmax(logits / 0.02)`, `agg = attn @ v_feats` pools under the
  model's own anomaly curve, and `contrast_type='n3'` mines an abnormal clip's
  lowest-scoring frames as extra negatives. Before relying on the `n3` half note
  `N3_MIN_SCORE_RANGE = 0.2`: at measured eval-time ranges only **14/60** TAD
  abnormal clips qualify (`gate_t0` 18/60), so it is a general handbrake.
- **`TRAFFIC_DEFINITIONS` is exported but never consumed.** Spec §7.4 prescribes
  it for DoTA/DADA-style zero-shot; `_DOTA_CLS_DEFS` uses
  `_UNIVERSAL_CLS_DEFS["CarAccident"]` instead. Dead code or an unrecorded
  deviation — resolve it in `TRAINING.md` §deviations either way.
- **On `main` there is exactly one gate, and it is the frozen one.** `kip.gate_type`,
  `kip.gate_signal`, `kip.const_shift_ratio` and `core/kip/ecmr.py` are branch-`v3`
  additions. A KIP-on run on `main` is, in operation, a fixed ~50 % channel shift.
- **The gate MLP receives no gradient, permanently.** `(ratio * max_shift)
  .floor().long()` (`core/kip/gate_shift.py:112`) is not differentiable, so
  `KinematicShift.mlp`'s 321 parameters stay at random init for the whole run —
  verified empirically (all 6 tensors `grad is None`) and asserted by
  `test_gate_mlp_receives_no_gradient_spec_as_written`. Deliberate spec fidelity,
  but it means "motion-gated" describes the gate's *input*, not a learned map.
- **The RAFT flow target is 23-dimensional, not 256.** `flow_statistics`
  (`core/flow/raft_extract.py:52-81`) pools each field into 23 frame-global
  scalars (mag mean/std/max, u/v mean/std, 16-bin angle histogram), then a fixed
  seeded Gaussian projection lifts them to 256-d. `ê_O` therefore lives on a
  ≤23-d manifold with **no spatial content**.
- **A short corpus breaks the score head, silently.** `ConvScoreHead` is one
  `Conv1d(kernel_size=9)`; median clip length is MSAD 86, DoTA 13, **DADA-2000 9**.
  Where the kernel covers the clip the detector becomes a clip classifier — no
  error, just `auc_macro` at chance under an inflated micro AUC. **Check
  `score_head_kernel` against a new corpus's median sequence length before
  training.** `core/docs/v3/RESULTS_DADA.md` §5.
- **[v3 only] `gate_type=mlp_ste` produces NaNs under AMP.** 33 of 500 steps on the DADA
  arm, in `mil` / `mul_mil`, while `kip_rec` / `kip_align` stayed finite; no other
  gate type at the same seed, batch and data. Suspect
  `shift_channels_straight_through` (`core/kip/gate_shift.py:99`) in fp16. Not
  root-caused (2026-09-06) — treat any `mlp_ste` metric as provisional.
- **The two branches disagree on aspect ratio by exactly 4:3.** The student
  resizes to 224×224 (aspect 1.000, `extract_clip_features.py:71`); the teacher
  interpolates to 240×320 (aspect 1.333, `raft_extract.py:135`). Source-independent,
  a bias not noise, and it warps `atan2(v, u)` — the direction channels. Fixing it
  invalidates the whole flow cache (C2), so it is scheduled, not done.
- **The DADA-2000 original release does not fit on a Colab VM** (measured
  2026-09-15). It is a **spanned PKZIP archive**: `DADA2000.zip` + `.z01`–`.z05`,
  **116.7 GiB compressed over 6 volumes**. `unzip DADA2000.zip` *fails* — the
  `.zip` part is the **last** volume and holds the central directory; and
  `zip -s 0 … --out` needs **2×** the space. Use `7z` (`apt install p7zip-full`),
  which reads spanned archives natively and extracts selected entries.
  Of five per-clip subdirs only **`images` (94.01 GiB, 651,320 files)** is video;
  `maps` / `seg` / `semantic` / `fixation` are DADA's own driver-attention task
  and are never extracted. `/content` offers ~88 GB, so **Phase 2 shards**
  (~200 clips ≈ 9.6 GiB: extract → CLIP features → delete → next). Persisted
  output ≈ **167 MB**. Freeing Drive space does not help — the constraint is the
  VM overlay, not the Drive quota.
- **`dada标注.xlsx`'s sheets are named the opposite of their contents.**
  `name="text"` → `sheet1.xml` is the type 1–38 taxonomy; `name="Sheet1"` →
  `sheet2.xml` is the 1,962-row per-clip table. Detect the sheet by its columns;
  a hardcoded name parses zero rows and the failure reads like a corrupt file.
- **`preprocess_frames` takes a decoded array, not paths, and centre-crops by
  default.** `(frames: np.ndarray, crop_size=224, center_crop=True)` — pair it
  with `read_images(paths)` and pass **`center_crop=False`**, or you measure a
  transform this project does not use, silently (C2, C13).

## Commands

```bash
# tests  (pyproject sets addopts="-q", so the summary line is suppressed;
#          count with:  pytest --co -q | awk -F': ' '/^core/{s+=$2} END{print s}')
source .venv/bin/activate && python -m pytest core/tests -q
# on `main` (2026-09-15): 514 collected -> 514 pass, 0 fail.
# The gate matrix in test_{dada,tad}.py was collapsed to the single v1 gate;
# the `kip.gate_type` parametrization is branch-`v3`-only and stays there.

# quality gates (before every commit)
source .venv/bin/activate && ruff check * && mypy * && bandit * && pycycle * && pyright *
# note: pycycle runs as `cd core && pycycle --here`
```

Colab run sequence (downloads → preprocess → extract → gate (a) → stage 1 →
stage 2 → eval): `core/docs/COLAB.md`. Its later sections are the experiment
runbooks in order: § "no_center_crop rebuild" → § "Arm 4 + trajectory probe",
each carrying a RUN banner pointing at the `RESULTS_*.md` it produced.
**Per-corpus runbooks live in `core/docs/v3/setup/`** — `DADA_V3_SETUP.md` (run,
see `RESULTS_DADA.md`) and `TAD_V3_SETUP.md` (**unrun**) on `main`;
`MSAD_DOTA_V3_SETUP.md` is on `v3` only — over the data runbooks `TAD_SETUP.md` /
`DADA_SETUP.md`. `core/docs/PREVAD_SETUP.md` is still unrun.
**Warning:** every arm command in those runbooks passes `--set kip.gate_type=…`
and therefore **fails on `main` at config parse**; their data-build sections are
fine. Run the arm ladder on `v3`. Training notebooks are
`colab/{MSAD,DADA}/v3/train.py` — note `collab/` was renamed to `colab/` and
`colab/` is **gitignored**, so on `main` they are untracked.

**Analysis is offline.** Every reported number is recomputed from
`{run}/scores/*.npz` via `core/tools/rescore.py` or a scratchpad bootstrap
script — no re-inference, and `outputs/` is gitignored, so `core/docs/RESULTS_*.md`
is the only durable record of a measurement. **`outputs/**` currently holds
62,254 per-clip `.npz` files**, each with `score` and `gt` — enough to compute
most of spec v2 §10 (seed-level intervals, raw-vs-min-max, per-class DoTA
breakdowns, the MSAD traffic slice, AUC_A/MCC/mAP@IoU) locally with **zero GPU**.
That was v2 tier 0; the v3 control arms
(`RESULTS_V3_GATE_ATTRIBUTION.md`, `RESULTS_DADA.md`) answered the same questions
directly, so tier 0 is closed rather than pending.
