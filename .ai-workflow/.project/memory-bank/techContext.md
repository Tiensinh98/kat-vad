# Tech Context — stack, setup, constraints

**Created:** 2026-07-31 (re-init from `b9978ff`, read off `pyproject.toml`)
**Last reviewed:** 2026-09-08 (branch `main` = v1; counts and test status
re-measured on this tree)

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
  `main`, **425 collected: 413 pass, 12 fail** (the v3-only `gate_type` matrix,
  see [[progress]]). `v3` is 537 green.
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
| `core/docs/` | 17 files on `main`: COLAB, DATA_LAYOUT, DOTA_EVAL, **EDA**, PREVAD_SETUP, **TAD_SETUP**, **DADA_SETUP**, TRAINING, proposal, spec, 6 × RESULTS_\*.md, REPORT_KIP_MSAD_DOTA_PREVAD |
| `core/docs/v3/` | **On `main`: only `RESULTS_DADA.md` + `setup/{DADA_V3_SETUP,TAD_V3_SETUP}.md`** (3 files). ARCHITECTURE, spec_v3, audit_addendum_PreVAD, RESULTS_V3_GATE_ATTRIBUTION and `setup/MSAD_DOTA_V3_SETUP.md` are **branch `v3` only**. `core/docs/v2/` does not exist on either branch — do not cite it |

**Measured on `main`, 2026-09-08:** **79 Python files** (54 source + 25 test),
**10,484 LOC** source. **20 markdown docs** under `core/docs/**`. (`v3` measures
84 files / 11,157 source LOC / 537 tests — different tree, different numbers.)

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
  reconstructed from the loss trace. Lesson 17.
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

## Commands

```bash
# tests
source .venv/bin/activate && python -m pytest core/tests -q
# on `main` (2026-09-08): 425 collected -> 413 pass, 12 fail.
# The 12 are test_{dada,tad}.py::TestXTrainsUnderEveryGate, which parametrize
# over `kip.gate_type` -- a branch-`v3`-only config field. Not a regression.

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
