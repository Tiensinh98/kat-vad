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

## Local dry-run (no Colab, no downloads)

Skip the mount and env cells (roots default to the repo directory) and add
`--text-encoder stub` to train/evaluate/inference — deterministic stub
embeddings, zero network. The synthetic fixture
(`core/tests/fixtures.py::build_fixture`) drives the full pipeline on CPU.
**Pin `--set train.device=cpu` locally** — torch 2.4 MPS training diverges
(lesson P6).
