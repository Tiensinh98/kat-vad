# Colab run guide (Phase 6 execution)

> **Scope: the MSAD traffic slice** (`data/MSAD`, frozen 28-video test
> split, gates a/b/c). For training on the **entire MSAD benchmark**
> (paper-comparable, dataset name `MSAD-full`) see `COLAB_MSAD_ALL.md` —
> Steps 0–3 below are shared by both workflows.

Target: Colab **A100 40 GB** runtime, running everything with **Colab's own
Python** — no virtualenv. All data, caches, checkpoints and outputs live on
Google Drive so a killed session resumes with `--resume`.

Each block below is one notebook cell. Python cells are plain code; shell
commands are `!`-prefixed. Do **not** use `export` in `!` cells — Colab runs
each `!` line in a throwaway shell, so exported variables vanish. Environment
variables are set once via `os.environ` (Step 2) and inherited by every
subsequent `!python` subprocess.

Session-restart rule of thumb: after a runtime restart or reconnect, re-run
**Step 1 (mount) and Step 2 (env vars)** only. Everything else is
skip-if-exists / resumable.

---

## Step 0 — Runtime + install (once per VM)

Select the GPU runtime first: `Runtime → Change runtime type → A100 GPU`.

**Cell 0.1 — mount Drive** (needed before install only if you clone the repo
from Drive; always needed before Step 2):

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 0.2 — clone + install into Colab's Python:**

```bash
!git clone <YOUR_REPO_URL> /content/kat-vad
%cd /content/kat-vad
!pip install -q -e .
```

Notes:

- The repo goes to `/content` (local VM disk — fast). Only data/caches/ckpts
  go to Drive.
- `pip install -e .` installs into the system interpreter — no venv, per the
  Colab workflow. It **downgrades** Colab's preinstalled stack to our pins
  (`torch==2.4.*`, `torchvision==0.19.*`, `numpy<2`, `transformers==4.56.*`).
- **After the install finishes: `Runtime → Restart session`.** Colab has
  already imported the old numpy/torch into the kernel; the restart is
  mandatory to avoid a mixed ABI. The install itself survives the restart
  (same VM). Then re-run Cell 0.1 and Step 2 and continue.

**Cell 0.3 — sanity check (after the restart):**

```python
import torch, core
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

Expect `2.4.x True NVIDIA A100...`.

## Step 1 — Mount Drive (every session)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2 — Environment variables (every session)

All path resolution goes through `core/constants.py`, which reads these env
vars at import time (see `core/docs/DATA_LAYOUT.md`). Set them **before any
`!python -m core...` command**:

```python
import os

DRIVE = '/content/drive/MyDrive/kat-vad'
os.environ['KATVAD_DATA_ROOT']   = f'{DRIVE}/data'     # raw videos + label files
os.environ['KATVAD_CACHE_ROOT']  = f'{DRIVE}/cache'    # CLIP / flow / KNN caches
os.environ['KATVAD_CKPT_ROOT']   = f'{DRIVE}/ckpts'    # best.ckpt + our checkpoints
os.environ['KATVAD_OUTPUT_ROOT'] = f'{DRIVE}/outputs'  # eval results, plots, logs
for v in ('KATVAD_DATA_ROOT', 'KATVAD_CACHE_ROOT', 'KATVAD_CKPT_ROOT', 'KATVAD_OUTPUT_ROOT'):
    os.makedirs(os.environ[v], exist_ok=True)
```

`$KATVAD_DATA_ROOT` etc. in the `!` commands below expand because the shell
inherits the notebook process environment.

## Step 3 — Downloads (once, cached on Drive/VM)

```bash
!python -m core.tools.download all --dry-run   # inspect what would happen
!python -m core.tools.download all
```

What `all` fetches (each step skip-if-exists):

| Item | Source | Lands in |
|---|---|---|
| `ckpt` — LaGoVAD `best.ckpt` | Google Drive via gdown (resumable) | `$KATVAD_CKPT_ROOT/lagovad_best.ckpt` |
| `clip` — CLIP ViT-B/16 | HF `openai/clip-vit-base-patch16`, pinned revision | HF cache on the VM (re-fetched per new VM; small) |
| `raft` — RAFT-Large weights | torchvision (`C_T_SKHT_V2`) | torch hub cache on the VM |

Optional integrity check: `--ckpt-sha256 <hex>`.

## Step 4 — Preprocessing (once per dataset; MSAD shown)

### 4.0 Input format — what YOU must put on Drive first

Two things, and nothing else:

**(1) Videos** under `$KATVAD_DATA_ROOT/MSAD/videos/` — flat or nested
(subdirectories are searched recursively). Accepted extensions: `.mp4`,
`.avi`, `.mkv`, `.mov`, `.webm`. The **video id is the filename stem**
(`videos/foo/crash_001.mp4` → id `crash_001`) and must match the `name`
column of the annotation table exactly.

**(2) Annotation table** — one file (e.g.
`$KATVAD_DATA_ROOT/MSAD/annotations/anno.tsv`), one row per video, 5 columns:

```
name    scenario    total_frames    anomaly_start_frame    anomaly_end_frame
```

- Separator: tab, comma, or 2+ spaces. A header row is optional (auto-detected).
- `name`: video id without extension (must match a video file stem).
- `scenario`: MSAD scenario string (e.g. `road`, `sidewalk`, `parkinglot`);
  used by `--scenarios` filtering and the split stratification.
- Normal (non-anomalous) videos: leave `anomaly_start_frame` /
  `anomaly_end_frame` **empty or negative**.
- Frame-index conventions default to **0-based** and **end-inclusive**;
  override with `--one-indexed` and/or `--end-exclusive` if your file differs.
  **Our MSAD annotation (`anno.tsv`) is confirmed 1-indexed, end-inclusive**
  (starts of `1` appear, never `0`; anomaly ends frequently equal
  `total_frames`) → pass `--one-indexed`, do NOT pass `--end-exclusive`.

Example (tab-separated, with header — matches the real `anno.tsv` format):

```
name	scenario	total frames	starting frame of anomaly	ending frame of anomaly
Traffic_accident_3	road	719	35	250
Traffic_accident_4	restaurant	336	1	80
road2	road	607		
```

### 4.1 Build label files (CPU, seconds)

```bash
!python -m core.data.msad \
  --annotation "$KATVAD_DATA_ROOT/MSAD/annotations/anno.tsv" \
  --out-dir "$KATVAD_DATA_ROOT/MSAD" \
  --scenarios all \
  --one-indexed
```

- `--scenarios all` because the whole file **is** the slice: it contains only
  `Traffic_accident` anomalies plus scenario-matched normals (including
  shop/restaurant/train/frontdoor — where MSAD's crash-into-building/train
  subtypes occur). This prevents the model from learning a scenario shortcut.
- `--one-indexed` per the confirmed annotation convention above.
- The split is seeded and stratified by (label, scenario) — re-running with
  the same annotation and seed reproduces the same split, so gate numbers
  stay comparable. **Do not change `--seed` between gate runs.**

Writes into `$KATVAD_DATA_ROOT/MSAD/`:

- `labels_train.json` — `{video_id: 0|1}` video-level weak labels (train split)
- `frame_labels_test.json` — `{video_id: [0,1,...]}` per **sampled** frame
  (stride 8), test split
- `defs.json` — class-name list for the definition/verbalizer path
- `meta.json` — per-video scenario/class/split/window, diagnostics only

Split: seeded, stratified by (label, scenario), MSAD protocol-ii ratios
(abnormal 0.5 / normal 0.25 to test). Pass `--split-file <ids.txt>` to force
an explicit test-id list instead; `--seed` changes the draw.

### 4.2 Extract CLIP features (GPU, the long step)

```bash
!python -m core.tools.extract_clip_features \
  --videos-dir "$KATVAD_DATA_ROOT/MSAD/videos" \
  --dataset MSAD \
  --output-dir "$KATVAD_CACHE_ROOT/clip/MSAD" \
  --device cuda
```

Output: `cache/clip/MSAD/{video_id}.npy`, shape `(L, 512)` float32, stride-8
sampling. **Resumable** — one `.npy` per video; just re-run the cell after a
session death and it does only what is missing:

```
INFO core.tools.feature_cache: Resume: 412/1402 already cached in .../clip/DoTA_s1, 990 to extract
INFO core.tools.extract_clip_features: Saved 0qfbmt4G8Rw_000306.npy (137, 512) -- [1/990] elapsed 0m04s, eta 68m12s
```

The clip that was mid-write when the session died is redone, not skipped —
writes land on a `.part` sibling and are renamed into place, and an existing
`.npy` counts as done only if its header reads back (lesson C10). `--force`
re-extracts everything, which is what a **stride or transform change** needs;
nothing on disk records either, so resume cannot detect that for you (C2).

The same applies to `raft_extract` below.

### 4.3 Extract RAFT flow targets (GPU; train-time only)

```bash
!python -m core.flow.raft_extract \
  --videos-dir "$KATVAD_DATA_ROOT/MSAD/videos" \
  --dataset MSAD \
  --cache-root "$KATVAD_CACHE_ROOT/flow/v1" \
  --device cuda
```

Output under `cache/flow/v1/`:

- `MSAD/{video_id}.npy` — `(L, 256)` flow embeddings `e_O` (KIP targets)
- `MSAD/{video_id}.stats.npy` — `(L, 23)` raw pooling stats (motion-aware KNN key)
- `flow_projection.npz` — the seeded fixed projection; ships with this cache
  version, loaders fail loudly without it. Never overwrite `v1/` in place —
  a pooling change bumps to `v2/`.

Also resumable per video. Flow is never needed at inference.

### 4.4 Build the DVS KNN filler cache (CPU, fast)

```bash
!python -m core.data.knn_cache \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --dataset MSAD \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD" \
  --output "$KATVAD_CACHE_ROOT/knn/MSAD/knn_cache.npz"
```

Maps each abnormal train video to its top-10 normal fillers for dynamic video
synthesis. `--motion-key` (uses `--flow-dir`) appends the coarse motion
descriptor — intended for ego-centric sets (DoTA/DADA), leave it off for MSAD.

## Step 5 — Reproduction gate (a): best.ckpt through our eval

```bash
!python -m core.evaluate \
  --baseline-ckpt "$KATVAD_CKPT_ROOT/lagovad_best.ckpt" \
  --set kip.enabled=false \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD" \
  --output-dir "$KATVAD_OUTPUT_ROOT/gate_a" \
  --save-scores
```

Prints micro AUC/AP and writes `results.json` + per-video `.npz` scores.

Gate status on the frozen 28-video test slice (2026-07-12):

- **Gate (a′) — pipeline sanity: PASSED.** Zero-shot `best.ckpt` with the
  original (defs-v1) definitions scored **AUC 0.746 / AP 0.403** (positive
  frame base rate ≈ 0.25). The original ±0.5 reproduction tolerance was
  defined on TAD 89.56 and is untestable on a custom slice.
- **Definition-sensitivity ablation (2026-07-12, resolved):** extending the
  MSAD `Traffic_accident` definitions to cover single-vehicle obstacle/
  building/train collisions (defs-v2, per MSAD's published subtypes) changed
  nothing — AUC 0.746 (v1) vs 0.744 (v2) vs 0.743 (v1 re-run), all within
  run-to-run jitter. defs-v2 was reverted; v1 stands. Conclusion: the missed
  subtypes are appearance-ambiguous — richer text cannot re-weight visual
  evidence the frozen CLIP-frame features do not contain. That is the KIP
  motivation, measured.
- **Eval jitter (FIXED 2026-07-12):** the eval verbalizer used to be
  unseeded, resampling definitions per window per run. Across 3
  identical-config zero-shot runs: AUC spread ±0.003 (0.743–0.746) but
  per-video max_score swings up to ±0.3 (`_16`: 0.33/0.37/0.64). **Robust
  misses across all runs: `_31`, `_25`, `_20`** (the qualitative KIP test
  cases); `_11`, `_16`, `_39` are borderline/definition-sensitive. The
  verbalizer in `evaluate.py`/`inference.py` is now seeded with
  `constants.SEED` — runs made **after** this fix are reproducible to FP
  tolerance (±1e-5; bitwise equality is impossible on macOS, lesson P7) and
  are not comparable per-video against pre-fix runs.
- **Gate (b)** — Stage-2 KIP-off retrain AUC ≥ **0.744 ± 0.003** (zero-shot
  floor, mean of 3 runs).
- **Stage 1 milestone (build order §12 step 5): PASSED** — 100 epochs ×
  5 steps (batch 8), `kip_rec` mean 19.2 → 9.7 (median 17.2 → 7.9), plateau
  from ~step 200; `kip_align` 4.41 → 3.90. High per-step variance
  (3.2–22 in the last 50 steps) is expected: per-batch flow-target magnitude
  varies with clip motion content.
- **Gate (c)** — KIP-on ≥ KIP-off (same split, same defs).

## Step 6 — Training

> **Sync the repo first.** The 2026-07-12 changes are required here: the
> seeded eval verbalizer (reproducible gate numbers), the train CLI's
> `--init-weights` flag (stage 1 → stage 2 handoff), and the AMP-safe
> kinematic loss (any `train.amp=true` run with KIP-on crashes without it:
> "binary_cross_entropy ... unsafe to autocast"). Re-run Cell 0.2's
> `pip install -q -e .` only if dependencies changed; a `git pull` in
> `/content/kat-vad` is enough for code.
>
> **One experiment = one `--output-dir`.** `metrics.jsonl` appends across
> invocations of the same dir (that is what makes `--resume` seamless), so
> a re-run into the same dir mixes runs in one file.

### 6.1 Stage 1 — KIP-only warm-up

```bash
!python -m core.train \
  --set train.stage=1 \
  --set train.batch_size=8 --set train.num_epochs=100 \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD" \
  --flow-dir "$KATVAD_CACHE_ROOT/flow/v1/MSAD" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/stage1"
```

**Batch/epoch sizing on this slice matters.** The 84-video annotation yields
~40 train items per epoch (2×num_anomaly balancing), so the default
`batch_size=64` gives **one optimizer step per epoch** — 5 epochs is 5 steps,
all inside the 20-step LR warm-up (peak LR 5e-5 is never reached), and the
loss cannot move. `batch_size=8` → ~5 steps/epoch; 100 epochs ≈ 500 steps.
The cosine horizon auto-adjusts to `steps_per_epoch × num_epochs`.

Watch `kip_rec` in `$KATVAD_OUTPUT_ROOT/stage1/metrics.jsonl` — expect a
downward trend from ~step 30 (after warm-up). Missing flow files fail loudly
(`require_flow` is on whenever KIP is enabled), so a flat `kip_rec` at step
200+ means a real problem: inspect the `e_O` value range in a couple of
`cache/flow/v1/MSAD/*.npy` files and confirm `flow_projection.npz` is the
one produced by Step 4.3, before touching hyperparameters.

**Status 2026-07-12: done.** 500 steps, `kip_rec` mean 19.2 → 9.7 with a
plateau from ~step 200 — the resulting
`$KATVAD_OUTPUT_ROOT/stage1/checkpoint_last.pt` is the `--init-weights`
source for §6.2; no need to re-run stage 1.

### 6.2 Stage 2 — full objective, KIP-on (the gate (c) contender)

Stage 2 **warm-starts from the stage-1 weights** via `--init-weights`
(spec §8: stage 1 exists to make `ê_O` faithful *before* task training).
Do **not** use `--resume` for this — resume restores the stage-1 epoch
counter and its fully-decayed LR schedule, so training would do nothing.
`--init-weights` loads model weights only; optimizer, schedule and counters
start fresh.

```bash
!python -m core.train \
  --set train.stage=2 --set train.amp=true \
  --set train.batch_size=8 --set train.num_epochs=100 \
  --init-weights "$KATVAD_OUTPUT_ROOT/stage1/checkpoint_last.pt" \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD" \
  --flow-dir "$KATVAD_CACHE_ROOT/flow/v1/MSAD" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/stage2_kip_on"
```

- **Batch/epoch sizing:** same arithmetic as §6.1 — ~40 train items/epoch,
  so batch 8 → 5 steps/epoch, 100 epochs ≈ 500 steps (vs. 1 step/epoch at
  the default batch 64). Watch `mil`/`total` in `metrics.jsonl`.
- **Resume after a dead session:** rerun the same command but replace
  `--init-weights ...` with
  `--resume "$KATVAD_OUTPUT_ROOT/stage2_kip_on/checkpoint_last.pt"`
  (the two flags are mutually exclusive; resume carries the warm-started
  weights forward anyway). Resume replays the exact stream — checkpoints
  hold model/optim/sched/scaler plus all RNG states.
- **Time-boxed sessions:** add `--stop-after-epochs N` to cap what this
  invocation runs without shrinking the LR schedule horizon.
- **Mid-epoch checkpoints:** `--set train.checkpoint_every_steps=N`
  (recommended on Colab; Drive keeps the file when the VM dies).
- A100 knobs: `train.amp=true`; `--set train.grad_accum_steps=K` exists but
  is moot at batch 8.

### 6.3 Stage 2 — KIP-off (the gate (b) baseline)

Same command, three changes: `kip.enabled=false`, **no `--init-weights`**
(the KIP-off architecture has no `kip.*` weights, so a stage-1 warm-start
is meaningless — the loader rejects it loudly), fresh output dir:

```bash
!python -m core.train \
  --set train.stage=2 --set train.amp=true \
  --set kip.enabled=false \
  --set train.batch_size=8 --set train.num_epochs=100 \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/stage2_kip_off"
```

(`--flow-dir` is unnecessary here — KIP-off never reads flow.)

## Step 7 — Evaluation: decide gates (b) and (c)

Evaluation is deterministic since the 2026-07-12 verbalizer seeding — one
run per checkpoint is meaningful. Evaluate both stage-2 checkpoints on the
same frozen test slice:

```bash
!python -m core.evaluate \
  --ckpt "$KATVAD_OUTPUT_ROOT/stage2_kip_off/checkpoint_last.pt" \
  --set kip.enabled=false \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD" \
  --output-dir "$KATVAD_OUTPUT_ROOT/eval_kip_off" \
  --save-scores
```

```bash
!python -m core.evaluate \
  --ckpt "$KATVAD_OUTPUT_ROOT/stage2_kip_on/checkpoint_last.pt" \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD" \
  --output-dir "$KATVAD_OUTPUT_ROOT/eval_kip_on" \
  --save-scores
```

(The model is built from the CLI config, not the checkpoint — so the
KIP-off eval **must** repeat `--set kip.enabled=false` or the state dict
won't load.)

Read the gates off the two `results.json` files:

- **Gate (b):** `eval_kip_off` AUC ≥ **0.744** (the zero-shot floor from
  Step 5). Fails → the training recipe, not KIP, is the problem; revisit
  `core/docs/TRAINING.md` deviations (L_dvs row-gating,
  `loss.captions_from_definitions`) before touching KIP.
- **Gate (c):** `eval_kip_on` AUC ≥ `eval_kip_off` AUC. Also check the
  per-video scores of the robust zero-shot misses `_31`, `_25`, `_20` —
  KIP lifting exactly those is the thesis result.
- n=28 caveat: deltas under ~1 AUC point are inside the small-sample noise;
  for a defensible gate (c) margin, repeat both trainings with
  `--set train.seed=<s>` for 2–3 seeds and compare means.

Visualize either run:

```bash
!python -m core.tools.visualize \
  --scores "$KATVAD_OUTPUT_ROOT/eval_kip_on/scores" \
  --output-dir "$KATVAD_OUTPUT_ROOT/eval_kip_on/plots"
```

Single raw video (extraction path, needs the CLIP weights from Step 3):

```bash
!python -m core.inference \
  --video some_clip.mp4 \
  --ckpt "$KATVAD_OUTPUT_ROOT/stage2_kip_on/checkpoint_last.pt" \
  --defs "$KATVAD_DATA_ROOT/MSAD/defs.json" \
  --output-dir "$KATVAD_OUTPUT_ROOT/inference_demo"
```

---

## no_center_crop rebuild (2026-08-08)

**Why:** two independent reasons, detailed in `core/docs/RESULTS_DOTA.md` §4.

1. LaGoVAD trained everything on `no_center_crop` features (anisotropic resize
   to 224², full field of view). Our center-crop cache makes `gate_a` — the
   reproduction reference — a train/test mismatch.
2. `raft_extract` is already **full-frame** (240×320, no crop). Center-cropping
   the CLIP branch means KIP is trained to regress motion evidence removed from
   its own input; on dashcam footage that is exactly the lateral field of view
   where the kinematics live.

**Everything goes to new paths.** Nothing below overwrites a center-crop
artifact, so the existing MSAD 0.9052 reproduction stays on disk to compare
against (lesson C2/C13).

| | center-crop (existing) | `no_center_crop` (new) |
|---|---|---|
| MSAD CLIP | `cache/clip/MSAD` | `cache/clip/MSAD_ncc` |
| MSAD KNN | `cache/knn/MSAD` | `cache/knn/MSAD_ncc` |
| MSAD runs | `outputs/MSAD` | `outputs/MSAD_ncc` |
| DoTA CLIP | `cache/clip/DoTA_s1`, `_s8` | `cache/clip/DoTA_ncc_s1`, `_ncc_s8` |
| DoTA runs | `outputs/DoTA` | `outputs/DoTA_ncc` |
| **RAFT flow** | `cache/flow/v1/MSAD` | **reused unchanged** |

**The flow cache is not rebuilt.** `preprocess_for_raft` never used
`preprocess_frames`, so it is independent of this transform. That removes the
most expensive item from the rebuild.

### R0 — Cheap go/no-go first (~30 min, do this before anything else)

Extract ~200 DoTA clips with the new transform and re-score `gate_a` only. If
0.6142 moves toward the published 0.6260, the transform is confirmed and the
full rebuild is justified by evidence rather than by argument.

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
head -200 "$KATVAD_DATA_ROOT/DoTA/labels_s8/test_ids.txt" > /content/probe_ids.txt

python -m core.tools.extract_clip_features \
  --frames-dir "$KATVAD_DATA_ROOT/DoTA/frames" --frames-subdir images \
  --ids-file /content/probe_ids.txt \
  --dataset DoTA --stride 8 --batch-size 64 --device cuda \
  --no-center-crop \
  --output-dir "$KATVAD_CACHE_ROOT/clip/DoTA_probe_ncc_s8"
```

Then score `gate_a` on the probe subset under both caches and compare **the same
200 clips** on each — paired, so the comparison is clean. Proceed to R1 only if
the gap closes materially.

### R1 — MSAD CLIP features, new transform

```bash
!python -m core.tools.extract_clip_features \
  --videos-dir "$KATVAD_DATA_ROOT/MSAD/videos" \
  --dataset MSAD-full \
  --no-center-crop \
  --output-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --device cuda
```

Confirm the log line reads `Transform: anisotropic resize`. Resumable as usual.

### R2 — KNN cache (derived from CLIP, so it must be rebuilt)

```bash
!python -m core.data.knn_cache \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --dataset MSAD-full \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --output "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz"
```

### R3 — Retrain: stage 1, then stage 2 ×2

Same commands as §6.1–6.3 with `--clip-dir` → `MSAD_ncc`, `--knn-cache` →
`MSAD_ncc`, `--output-dir` → `$KATVAD_OUTPUT_ROOT/MSAD_ncc/...`. `--flow-dir`
is **unchanged** (`cache/flow/v1/MSAD`).

> **Fold the validation split + `checkpoint_best` work into this retrain.** It
> is already queued in `activeContext.md`, and the current DoTA arms are
> uninterpretable partly because eval reads `checkpoint_last` deep in the
> overfit regime (`RESULTS_DOTA.md` §3). Doing it separately means paying for
> two full training cycles.

### R4 — Re-measure MSAD

The 0.9052 vs paper 0.9041 reproduction was obtained on the center-crop cache.
It is **unverified** until re-measured here, and it may not land in the same
place — that is the accepted cost of the switch, not a surprise.

```bash
!python -m core.evaluate \
  --ckpt "$KATVAD_OUTPUT_ROOT/MSAD_ncc/stage2_kip_off/checkpoint_best.pt" \
  --set kip.enabled=false --set data.dataset=MSAD-full \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_ncc/eval_kip_off" --save-scores
```

`--score-norm` defaults to `auto`, which resolves to `none` on MSAD (50.4 %
normal videos). Confirm the log says so.

### R5 — DoTA, full re-extraction and the three arms

The long pass: 1,397 clips at stride 1. Resumable; expect it to outlive a Colab
session at least once.

```bash
!python -m core.tools.extract_clip_features \
  --frames-dir "$KATVAD_DATA_ROOT/DoTA/frames" --frames-subdir images \
  --ids-file "$KATVAD_DATA_ROOT/DoTA/labels_s8/test_ids.txt" \
  --dataset DoTA --stride 1 --batch-size 64 --device cuda \
  --no-center-crop \
  --output-dir "$KATVAD_CACHE_ROOT/clip/DoTA_ncc_s1"
```

Derive stride 8 (`range(0,N,8)` makes `[::8]` exact — see `DOTA_EVAL.md` §3.4),
then run the three arms as in `DOTA_EVAL.md` §3.5 with `--clip-dir` →
`DoTA_ncc_s8` and `--output-dir` → `$KATVAD_OUTPUT_ROOT/DoTA_ncc/...`.

`--score-norm auto` resolves to `minmax` on DoTA. Confirm the log line:

```
INFO core.metrics: score-norm auto -> minmax (1394/1397 videos abnormal; 0.21% normal, threshold 5%)
```

### R6 — Compare

```bash
for d in gate_a eval_kip_off eval_kip_on; do
  python -m core.tools.rescore --run-dir "$KATVAD_OUTPUT_ROOT/DoTA_ncc/$d"
done
```

Record against the center-crop baselines in `RESULTS_DOTA.md` §1. The number
that decides whether the rebuild was worth it is **`gate_a` min-max vs 0.6260**;
the number that matters scientifically is still **Δ(on − off) with its CI**.

---

## Arm 4 + trajectory probe (2026-08-16, post Phase A)

> **RUN 2026-08-19 — results in `core/docs/RESULTS_ARM4_PROBE.md`.** Both
> confounds closed: Δ(on − off_warm) = +0.0988 ± 0.0148 (nine CIs excluding
> zero), H3 rejected at matched convergence. §A5.0 was correctly skipped — a
> plain `torch.load` worked in that session. Kept here as the runbook of record.
> Next time, set `checkpoint_every_steps` denser early: the probe had no
> checkpoint below step 100, where 94 % of the loss range lives (lesson 16).

**Why:** Phase A settled H2 — the DoTA gain replicates across seeds 2024/2025/
2026 at Δ +0.0916 ± 0.0088, every CI excluding zero (`RESULTS_PHASE_A.md`). Two
confounds remain, and **neither needs the val-split machinery** originally
planned as Phase B:

| open item | what closes it | cost |
|---|---|---|
| **Warm-start asymmetry** — KIP-on inits from stage 1, KIP-off starts cold | Arm 4: KIP-off warm-started | 3 training runs, no code |
| **H3** — KIP-on is simply less fitted to MSAD (train `mil` 3.5–6.3× higher every seed) | Trajectory probe over saved step checkpoints | ~24 evals on cached features, no code, no training |

Do A4 first: it is the question a reviewer asks first, and if the Δ dies there
the probe is moot.

### A4 — the fourth arm: KIP-off, warm-started

Identical to the existing KIP-off arm in every respect except that it inits from
the *same* stage-1 checkpoint the KIP-on arm used. That isolates KIP itself from
the 125 epochs of trunk pretraining `L_KIP_rec`/`L_KIP_align` deliver through the
shared temporal encoder.

`--flow-dir` is **omitted** — with `kip.enabled=false` no flow target is
consumed, and passing it would only invite confusion about what the arm sees.

#### A4.0 — strip the KIP weights first (required; the run raises without it)

`warm_start_model` (`core/train.py:98-108`) is fail-loud on **unexpected** keys,
and its error message names this exact case: *"a KIP-off run cannot warm-start
from a KIP-on checkpoint."* A stage-1 checkpoint carries `kip.*` parameters that a
`kip.enabled=false` model does not have, so `--init-weights` rejects it outright.

That guard is correct and must not be weakened (lesson 5). Write a stripped copy
instead — explicit about exactly what is discarded, and it leaves the trainer's
contract untouched:

A plain `torch.load` on these checkpoints **fails on a Colab VM that has drifted
from the one that trained them** (observed 2026-08-16):

```
TypeError: _reconstruct: First argument must be a sub-type of ndarray
```

`save_checkpoint` (`core/train.py:408-425`) stores `rng`, and `rng["numpy"]` is
`np.random.get_state()` (`core/train.py:389`) — a tuple wrapping a 624-element
uint32 **numpy array**. That is the only numpy object in the payload, and it is
the one that fails to unpickle across a numpy major-version change. The tensors
are fine; the whole `torch.load` just aborts before reaching them, because a
pickle is one stream.

The cell below loads through an unpickler that refuses to reconstruct numpy
arrays at all — the RNG blob decodes to a discardable placeholder — then writes a
**slim** checkpoint carrying `model` plus provenance scalars and nothing else.
Both consumers only ever read `payload["model"]`: `warm_start_model`
(`core/train.py:94`) and `load_model_for_scoring` (`core/inference.py:62-63`).
Dropping `rng`/`optimizer`/`scheduler`/`scaler` costs nothing here — this is a
fresh run via `--init-weights`, not a resume — and makes the output immune to the
same failure later.

```python
import os, pickle, torch
from pathlib import Path


class _Discarded:
    """Stands in for a numpy array we deliberately refuse to reconstruct."""

    def __setstate__(self, state): pass


def _discard(*args, **kwargs): return _Discarded()


class _NumpyFreeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in ('_reconstruct', 'scalar') and module.endswith('multiarray'):
            return _discard
        return super().find_class(module, name)


class _shim:                       # duck-types the pickle module for torch.load
    Unpickler = _NumpyFreeUnpickler

    @staticmethod
    def load(f, **kw): return _NumpyFreeUnpickler(f, **kw).load()


OUT = Path(os.environ['KATVAD_OUTPUT_ROOT'])
for S in (2024, 2025, 2026):
    src = (OUT / 'MSAD_ncc' if S == 2024 else OUT / f'MSAD_ncc_s{S}') / 'stage1'
    payload = torch.load(src / 'checkpoint_last.pt', map_location='cpu',
                         weights_only=False, pickle_module=_shim)
    kept = {k: v for k, v in payload['model'].items() if not k.startswith('kip.')}
    dropped = len(payload['model']) - len(kept)
    slim = {'model': kept,
            'epoch': payload.get('epoch'),
            'global_step': payload.get('global_step'),
            'class_names': payload.get('class_names')}
    torch.save(slim, src / 'checkpoint_last_nokip.pt')
    print(f"seed {S}: dropped {dropped} kip.* tensors, kept {len(kept)}, "
          f"step={slim['global_step']}")
```

Two sanity checks before proceeding:

- `dropped` must be **> 0** for every seed. If it is 0 you pointed at the wrong
  checkpoint (a KIP-off stage-2 file, say) and the arm would be meaningless.
- `step` must match the stage-1 length in that seed's `config.yaml`. If it is
  `None`, the load silently returned something other than a trainer checkpoint.

The shim only intercepts numpy array reconstruction. Tensors travel the
`persistent_load` storage path and never reach `find_class`, so the weights are
byte-identical to a normal load — this is not a lossy recovery.

> **Do not "fix" this by downgrading numpy.** Colab's torch wheel is built
> against the numpy 2 ABI; pinning `numpy<2` to match the training VM tends to
> break the whole runtime (`_ARRAY_API not found`) for a problem that costs one
> cell to route around.

> The real defect is upstream: `save_checkpoint` pickles a library-versioned
> object (`np.random.get_state()`) into an artifact meant to outlive its
> environment. Storing it as raw bytes, or omitting it when RNG resume is not
> needed, would make checkpoints portable. Not changing that mid-experiment —
> the baseline of truth stays fixed — but it belongs in the backlog.

The trunk this transfers — temporal encoder, fusion, heads — is exactly the part
that received stage-1 gradients. Dropping `kip.*` is not a compromise: the KIP-off
arm has no KIP module to put them in. That is the whole point of the control.

> If this cell ever gets used a second time, promote it to
> `core/tools/strip_kip_weights.py` with a CLI and tests (§10/§11). One-off
> experiment scaffolding does not earn a module.

#### A4.1 — train

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad

for S in 2024 2025 2026; do
  SRC="$KATVAD_OUTPUT_ROOT/MSAD_ncc"; [ $S != 2024 ] && SRC="$KATVAD_OUTPUT_ROOT/MSAD_ncc_s$S"
  python -m core.train \
    --set train.stage=2 --set train.amp=true \
    --set kip.enabled=false \
    --set data.dataset=MSAD-full \
    --set train.num_epochs=125 \
    --set train.checkpoint_every_steps=100 \
    --set train.seed=$S \
    --init-weights "$SRC/stage1/checkpoint_last_nokip.pt" \
    --data-dir "$KATVAD_DATA_ROOT/MSAD" \
    --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
    --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
    --output-dir "$SRC/stage2_kip_off_warm"
done
```

> **Note the seed-2024 path asymmetry.** Seed 2024 lives in `MSAD_ncc/stage1`;
> 2025/2026 live in `MSAD_ncc_s$S/stage1`. The `SRC` line above handles it.
> Run seed 2024 alone first and confirm the log line
> `Warm-started model weights from ...` appears before queueing all three.

Then evaluate all three, both benchmarks:

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad

for S in 2024 2025 2026; do
  SRC="$KATVAD_OUTPUT_ROOT/MSAD_ncc"; DST="$KATVAD_OUTPUT_ROOT/DoTA_ncc"
  if [ $S != 2024 ]; then SRC="$KATVAD_OUTPUT_ROOT/MSAD_ncc_s$S"; DST="$KATVAD_OUTPUT_ROOT/DoTA_ncc_s$S"; fi

  python -m core.evaluate \
    --ckpt "$SRC/stage2_kip_off_warm/checkpoint_last.pt" \
    --set kip.enabled=false --set data.dataset=MSAD-full \
    --data-dir "$KATVAD_DATA_ROOT/MSAD" \
    --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
    --output-dir "$SRC/eval_kip_off_warm" --save-scores

  python -m core.evaluate \
    --ckpt "$SRC/stage2_kip_off_warm/checkpoint_last.pt" \
    --set kip.enabled=false --set data.dataset=DoTA \
    --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
    --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
    --output-dir "$DST/eval_kip_off_warm" --save-scores
done
```

**Decision rule** — compare KIP-on against `off_warm` instead of cold `off`:

| across 3 seeds | reading |
|---|---|
| Δ(on − off_warm) still ≥ +0.05, CIs exclude zero | Warm start is not the mechanism. **The A/B is now "KIP vs no KIP"** — the strongest version of the claim available |
| Δ shrinks materially but stays positive | Part of the gain was trunk pretraining. Report the *warm* Δ as the headline; the cold Δ overstates KIP |
| Δ collapses to ~0 | **The gain was the warm start, not KIP.** Say so. Stage-1 pretraining becomes the contribution, not the module |

### A5 — trajectory probe: does convergence level alone buy DoTA transfer?

H3's claim is that KIP-on transfers better *because* it is less fitted to MSAD.
That is directly falsifiable with checkpoints you already have — no val split, no
retraining, no smaller training set, so every number stays comparable to
Phase A's.

`checkpoint_every_steps=100` over 500 steps left 5 step checkpoints plus
`checkpoint_last` per stage-2 run. Score the whole trajectory of **both** arms on
DoTA, and read each checkpoint's train `mil` out of `metrics.jsonl`.

Start with seed 2024 only. Extend to 2025/2026 only if the answer is ambiguous.

#### A5.0 — sanitize the step checkpoints (same numpy failure as A4.0)

`core/inference.py:62` loads with `weights_only=False`, so **every**
`checkpoint_step_*.pt` hits the identical `_reconstruct` TypeError described in
A4.0. Rewrite each one slim first; `--ckpt` then points at the `_slim.pt` copy.
Reuse `_shim` from the A4.0 cell (run it in the same session, or paste the class
definitions again).

```python
S = 2024
SRC = OUT / 'MSAD_ncc'
for arm in ('on', 'off'):
    for ck in sorted((SRC / f'stage2_kip_{arm}').glob('checkpoint_step_*.pt')):
        if ck.stem.endswith('_slim'):
            continue
        payload = torch.load(ck, map_location='cpu',
                             weights_only=False, pickle_module=_shim)
        torch.save({'model': payload['model'],
                    'global_step': payload.get('global_step')},
                   ck.with_name(f'{ck.stem}_slim.pt'))
        print(f"{arm} {ck.stem}: step={payload.get('global_step')}, "
              f"{len(payload['model'])} tensors")
```

Check the printed `global_step` values are distinct and ascending. If two files
report the same step, you are about to plot the same model twice.

Skip A5.0 entirely if a plain `torch.load` on these files works in your session —
the failure is environment drift, not a property of the artifacts.

#### A5.1 — score the trajectory

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
S=2024
SRC="$KATVAD_OUTPUT_ROOT/MSAD_ncc"

for ARM in on off; do
  EXTRA=""; [ "$ARM" = off ] && EXTRA="--set kip.enabled=false"
  for CK in "$SRC/stage2_kip_$ARM"/checkpoint_step_*_slim.pt; do
    STEP=$(basename "$CK" _slim.pt)
    python -m core.evaluate \
      --ckpt "$CK" $EXTRA --set data.dataset=DoTA \
      --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
      --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
      --output-dir "$KATVAD_OUTPUT_ROOT/probe/DoTA_s$S/${ARM}_${STEP}"
  done
done
```

> `EXTRA` is set **inside** the loop on purpose. A `%%bash` cell starts with a
> clean environment, and an unset `EXTRA` silently turns a KIP-off eval into a
> KIP-on one — the exact defect found in
> `collab/DoTA_ncc/evaluate_s2025_s2026.py` (`RESULTS_PHASE_A.md` §8).

No `--save-scores`: the probe needs only `results.json`, and 12 × 1,397 score
files is a lot of Drive for a diagnostic.

Then plot DoTA AUC against that checkpoint's train `mil`, one curve per arm:

| observation | reading |
|---|---|
| A KIP-off checkpoint at `mil` ≈ 0.005–0.010 (matching KIP-on's endpoint) reaches DoTA ≈ 0.63 | **H3 holds.** KIP is acting as a regularizer. Still publishable — but not the proposal's claim, and say so plainly |
| KIP-off stays at 0.53–0.56 along its whole trajectory while KIP-on sits at 0.63 | **H3 is dead at matched convergence.** The Δ is KIP's |
| Curves overlap when plotted against `mil` rather than step | H3 holds — convergence level, not the module, indexes transfer |

**Discipline:** this reads DoTA *test* scores along a training trajectory. It is
a **diagnostic, not model selection**. The reported arm stays `checkpoint_last`,
and the probe curve never becomes a headline number — otherwise the result reads
as tuned on test. Write that sentence into whatever document reports it.

### Why the original Phase B is deferred

`.project/plans/msad-ncc-seeds-and-selection.md` §3 specified a `--val-ratio`
three-way split, a KNN rebuild, a new `core/tools/select_checkpoint.py`, and a
full retraining cycle — on a training set 20 % smaller (120 → 96 abnormal
videos), whose numbers are by its own §3.3 **not comparable** to anything already
measured or to LaGoVAD's.

A5 answers the same H3 question more directly, on the training set everything
else was measured on, for the cost of a dozen cached-feature evals. Build the
val-split machinery only if A5 comes back ambiguous.

---

## Local dry-run (no Colab, no downloads)

Skip the mount and env cells (roots default to the repo directory) and add
`--text-encoder stub` to train/evaluate/inference — deterministic stub
embeddings, zero network. The synthetic fixture
(`core/tests/fixtures.py::build_fixture`) drives the full pipeline on CPU.
**Pin `--set train.device=cpu` locally** — torch 2.4 MPS training diverges
(lesson P6).
