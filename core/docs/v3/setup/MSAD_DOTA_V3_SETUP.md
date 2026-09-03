# KAT-VAD v3 — MSAD + DoTA setup, training and evaluation runbook

**Written 2026-08-30, against branch `v3`** (73 source files, 434 tests green).
Supersedes, *for v3 runs only*, the run sequences in `core/docs/COLAB.md`
(§6–§7, §R1–R6) and `core/docs/DOTA_EVAL.md` (§3.2–§3.5). Those two documents
stay authoritative for everything this one does not repeat: the DoTA unzip and
coverage repair (`DOTA_EVAL.md` §3.1/§3.1.1), the analysis cell (§4), the
decision rules (§5), and the on-disk contract (`DATA_LAYOUT.md`).

> **What "again" means here.** Every number in `core/docs/RESULTS_*.md` was
> produced with `gate_type="mlp_frozen"` — a gate that, measured, is a **fixed
> ~50 % temporal smoother** (activeContext 2026-08-30, plan Appendix C, lesson
> **C24**). This runbook re-runs MSAD and DoTA under the v3 gate dispatch so the
> +0.09 DoTA effect can finally be **attributed** rather than re-observed.

> **DoTA is not trained on.** It ships no train split (`labels_train.json` is
> empty by construction) and it is the **zero-shot transfer** benchmark:
> checkpoints are trained on MSAD-full and scored on DoTA. Every "DoTA training"
> command you might expect below is deliberately absent.

---

## 0. What v3 changes for a run — five operational facts

1. **`kip.gate_type` is mandatory in any config file that enables KIP.** A YAML
   with a `kip:` section, `enabled` not false, and no `gate_type` **raises on
   load**. Every archived `outputs/*/*/config.yaml` from a KIP-on run predates
   the key and will now raise. That is intentional (lesson **C14**): resolving it
   silently either way makes two arms whose configs look identical and whose
   models are not. A KIP-**off** config still loads fine — it makes no claim
   about the gate.
2. **The checkpoint is bound to its gate type.** `load_kip_state_dict` refuses an
   `mlp_frozen` checkpoint under a `rank`/`constant` model and vice versa, keyed
   on `kip.shift.mlp.*`. **It cannot tell `rank` from `constant`** — both are
   parameter-free with an identical key layout. §5.4 is how you keep that
   straight by hand.
3. **KIP's train-only submodules are off the inference graph.** `kip.mhead.*`
   (3e) and `kip.proj_flow.*`/`kip.proj_rgb.*` (3f) are dropped by allowlist when
   scoring — never `strict=False`. Inference carries 311,808 KIP params and
   **0 on the score path** under `rank`/`constant`.
4. **Gate diagnostics ship in every score `.npz`.** `--dump-kip-diag` defaults
   **on** in `core/evaluate.py` (with `--save-scores`) and in `core/inference.py`;
   training never collects them. **§6.8 is the check that makes this runbook
   worth running** — and because `rank` and `constant` checkpoints load into
   each other silently, it is the only thing that proves which gate you scored.
5. **Stage-1 convergence is a load-bearing precondition, and it is not
   enforced.** No gradient reaches `ê_O` through the shift under `rank`,
   `mlp_frozen` or `constant` — `s_t` is a hard integer slice index. Spec v3 §8's
   `assert stage1_final(L_KIP_rec) < τ_rec` is **not implemented**. Check it by
   hand (§5.1) or you will train a stage-2 arm that ranks a badly-trained `ê_O`
   and still get a plausible number.

---

## 1. Preconditions

### 1.1 Code + install (Colab, once per VM)

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content/drive/MyDrive/Thesis-V3/kat-vad
!git checkout v3 && git pull
!pip install -q "torch==2.4.*" "torchvision==0.19.*" "transformers==4.56.*" \
  "numpy<2" "av>=12" "einops>=0.8" "faiss-cpu>=1.8" "gdown>=5" \
  "matplotlib>=3.9" "opencv-python==4.11.*" "pyyaml>=6" \
  "requests>=2.32" "scikit-learn>=1.5" "torchmetrics>=1.4" "tqdm>=4.66"
```

**Restart the runtime after the install** (`Runtime → Restart session`), then
re-run the mount and §1.2. Colab has already imported the old numpy/torch;
the restart avoids a mixed ABI.

Confirm you are on v3 code before anything else — a v2 tree accepts a config
this runbook writes and silently runs a different model:

```bash
!python -c "from core.kip import gate_shift; print(gate_shift.GATE_TYPES)"
# expect: ('rank', 'mlp_frozen', 'mlp_ste', 'constant')
```

### 1.2 Environment variables (every session)

```python
import os
DRIVE = '/content/drive/MyDrive/Thesis'
os.environ['KATVAD_DATA_ROOT']   = f'{DRIVE}/data'
os.environ['KATVAD_CACHE_ROOT']  = f'{DRIVE}/cache'
os.environ['KATVAD_CKPT_ROOT']   = f'{DRIVE}/ckpts'
os.environ['KATVAD_OUTPUT_ROOT'] = f'{DRIVE}/outputs'
for v in ('KATVAD_DATA_ROOT', 'KATVAD_CACHE_ROOT', 'KATVAD_CKPT_ROOT', 'KATVAD_OUTPUT_ROOT'):
    os.makedirs(os.environ[v], exist_ok=True)
```

Never `export` inside a `!` cell — Colab throws that shell away per line.

### 1.3 What is already on disk, and what v3 does *not* invalidate

The v3 change is **model code only**. It touches no transform, no stride, no
pooling — so **every cache below is reused as-is** (lesson **C2** does not fire).

| Artifact | Path | Rebuild for v3? |
|---|---|---|
| MSAD CLIP features (`no_center_crop`) | `$KATVAD_CACHE_ROOT/clip/MSAD_ncc` | **No** |
| MSAD RAFT flow targets | `$KATVAD_CACHE_ROOT/flow/v1/MSAD/MSAD-full` | **No** |
| MSAD DVS KNN cache | `$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz` | **No** |
| MSAD labels | `$KATVAD_DATA_ROOT/MSAD/{labels_train,frame_labels_test,defs,meta}.json` | **No** |
| DoTA labels (stride 8) | `$KATVAD_DATA_ROOT/DoTA/labels_s8/` | **No** |
| DoTA CLIP features | `$KATVAD_CACHE_ROOT/clip/DoTA_s1_ncc`, `clip/DoTA_s8_ncc` | **No** |
| LaGoVAD `best.ckpt` | `$KATVAD_CKPT_ROOT/best.ckpt` | **No** |
| **Checkpoints** | any archived `stage2_kip_on/` | **Yes** — those are `mlp_frozen` models, and their `config.yaml` predates `gate_type` so it now raises. v3 arms train into fresh dirs (§5.2) |

Verify before assuming, in one cell:

```bash
%%bash
for p in "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
         "$KATVAD_CACHE_ROOT/flow/v1/MSAD/MSAD-full" \
         "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc"; do
  echo "$(ls "$p" 2>/dev/null | wc -l) files  $p"
done
ls "$KATVAD_CACHE_ROOT/flow/v1/flow_projection.npz" && echo "flow projection OK"
```

If any of those come back empty, build them with §3. Otherwise skip to §4.

---

## 2. Pick the arm before you type anything

### 2.1 The four gate types

| `kip.gate_type` | Params | `s_t` behaviour | Flow needed at train time |
|---|---:|---|---|
| `rank` (**v3 default**) | **0** | ECMR residual → within-clip rank → spans `[0, 128]` on every clip | yes |
| `mlp_frozen` | 321 | **Near-constant `s ≈ 58–69`, span 0–4/128** (measured, 8 seeds) | yes |
| `mlp_ste` | 321 | Forward bit-identical to `mlp_frozen`; backward unblocked | yes |
| `constant` | 0 | Fixed `const_shift_ratio`; `ê_O` ignored entirely | only if `disable_pmg=false` |

`gate_signal` (`flow_norm` / `feat_var`) applies to `mlp_*` only. Passing it
beside `rank` or `constant` **raises** — it is not ignored. Left unset on an
`mlp_*` arm it resolves to `flow_norm` inside the module, but the saved
`config.yaml` records `null`; set it explicitly if the arm is going in a table.

### 2.2 The arm matrix

| Arm | Flags added to the stage-2 command | What it is for | Priority |
|---|---|---|---|
| **A0** | `--set kip.enabled=false` | Baseline trunk. Gate-independent — **the existing KIP-off checkpoints stay valid, do not re-run them** | reuse |
| **A1** | `--set kip.gate_type=rank` | The v3 model | 2nd |
| **A2** | `--set kip.gate_type=constant --set kip.const_shift_ratio=0.5 --set kip.disable_pmg=true --set loss.lambda_rec=0 --set loss.lambda_align=0 --set kip.use_lkin=false` | **Plain-TSM control.** A fixed 50 % channel shift and nothing else. **All six flags are required** — see the warning below | **1st** |
| **A2b** | `--set kip.gate_type=constant --set kip.const_shift_ratio=0.5` (needs `--flow-dir`) | Same smoother, PMG **genuinely trained** by `L_KIP_rec`/`L_KIP_align`/`L_kin` against real flow. Separates "the smoother did it" from "the auxiliary losses did it" | 3rd |
| **A3** | `--set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm` | Bit-identical v1 reproduction; the bridge to every `RESULTS_*.md` number | 4th |
| **A4** | `--set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm` | Pure gradient intervention, zero forward confound. Also opens `L_MIL` → PMG | 5th |

> **`disable_pmg=true` does not disable the PMG head.** Verified against the
> code, 2026-08-30: the flag only sets `require_flow=False`, so the dataset
> hands back **zero** flow rows (`core/train.py:593`, `core/data/dataset.py:103`)
> while `L_KIP_rec` and `L_KIP_align` stay in the objective — training the
> 311,808-param PMG head to predict zeros, which is the exact anti-pattern
> already settled for PreVAD. Zeroing `loss.lambda_rec` / `loss.lambda_align`
> and `kip.use_lkin` is what actually removes the pathway. What remains is
> provably a pure shift: under `gate_type=constant`, `v^k` is **bit-identical**
> for wildly different `ê_O` (checked directly). The PMG head is still
> instantiated and still random — it just has no consumer and no gradient.

**Run A2 first.** It is one config block, the cheapest run in the program, and it
carries a pre-registered prediction that can falsify the central claim:

> **Pre-registered (2026-08-30):** if `r = 0.5` reproduces most of the
> +0.09 DoTA gain, KIP's measured contribution is temporal smoothing and the
> thesis needs restating. Write this prediction into the run notes **before**
> the eval, not after.

The minimum decisive set is **A2 and A1 across 3 seeds**, compared against the
**existing** A0 arms. Everything else is follow-up.

### 2.3 What raises, so you recognise it

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: ...kip.enabled is true but kip.gate_type is missing` | `--config` points at a pre-v3 KIP-on YAML | add `gate_type:` to the file, or drop `--config` and use `--set` |
| `ValueError: kip.gate_signal=... is meaningless for kip.gate_type='rank'` | signal passed to a parameter-free gate | remove the flag |
| `KeyError: This checkpoint contains N kip.shift.mlp.* tensors ...` | scoring an `mlp_frozen` checkpoint under `rank` | add `--set kip.gate_type=mlp_frozen` (or `--gate-type` in `inference.py`) |
| `KeyError: The model was built with kip.gate_type='mlp_frozen' ... no kip.shift.mlp.*` | the reverse | build with the gate that trained it |
| `ValueError: Missing flow cache ...` | KIP enabled, `disable_pmg=false`, `--flow-dir` wrong | fix the path; do **not** reach for `require_flow=false` |

---

## 3. Data preparation (skip whatever §1.3 already found)

All commands run from `/content/drive/MyDrive/Thesis-V3/kat-vad`.

### 3.1 MSAD-full labels

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.data.msad \
  --annotation "$KATVAD_DATA_ROOT/MSAD/annotations/anno_train_full.tsv" \
  --out-dir    "$KATVAD_DATA_ROOT/MSAD" \
  --scenarios all --one-indexed --infer-abnormal-from-name \
  --split-file "$KATVAD_DATA_ROOT/MSAD/annotations/test_ids.txt"
```

`--split-file` pins the test ids, so the split is fixed across every seed and
every arm. **Do not re-run this between arms** — a moved split makes two arms
incomparable in a way no CI will show you.

### 3.2 MSAD CLIP features — `no_center_crop`

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.tools.extract_clip_features \
  --videos-dir "$KATVAD_DATA_ROOT/MSAD/videos" \
  --dataset MSAD-full --no-center-crop \
  --output-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --device cuda
```

Confirm the log reads `Transform: anisotropic resize`. Resumable per video
(atomic `.part` writes + header verification, lesson **C11**); re-running does
only what is missing. `--force` is for a stride/transform change only — nothing
on disk records either, so resume cannot detect it for you (**C2**).

### 3.3 MSAD RAFT flow targets (train-time only)

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.flow.raft_extract \
  --videos-dir "$KATVAD_DATA_ROOT/MSAD/videos" \
  --dataset MSAD-full \
  --cache-root "$KATVAD_CACHE_ROOT/flow/v1/MSAD" \
  --device cuda
```

Writes `flow/v1/MSAD/MSAD-full/{id}.npy` `(L, 256)`, `{id}.stats.npy` `(L, 23)`
and `flow/v1/flow_projection.npz`. **Never overwrite `v1/` in place.** The flow
cache is independent of the CLIP transform (`preprocess_for_raft` is full-frame),
which is why the `ncc` rebuild never touched it.

> The flow target has **23 effective dimensions** — frame-global scalars lifted
> to 256-d by a fixed seeded projection. Never claim KIP localizes anything
> spatially.

### 3.4 MSAD DVS KNN cache

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.data.knn_cache \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" --dataset MSAD-full \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --output   "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz"
```

Derived from the CLIP cache, so it belongs to `MSAD_ncc` and to no other
transform. Leave `--motion-key` off for MSAD.

### 3.5 DoTA labels (stride 8)

Unzip and coverage repair are unchanged — follow `DOTA_EVAL.md` §3.1/§3.1.1
first, then:

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.data.dota \
  --metadata   "$KATVAD_DATA_ROOT/DoTA/metadata_val.json" \
  --split-file "$KATVAD_DATA_ROOT/DoTA/val_split.txt" \
  --frames-dir "$KATVAD_DATA_ROOT/DoTA/frames" --frames-subdir images \
  --out-dir    "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --stride 8 --allow-missing-frames
```

Expect `1402 clips (805 ego / 597 other), 18478 sampled frames, 6118 positive`.
`meta.json` carries `ego_involve`, which the mechanism check in `DOTA_EVAL.md` §6
needs. **`--data-dir` for every DoTA eval is `labels_s8/`, not `DoTA/`.**

### 3.6 DoTA CLIP features — stride 1, then derive stride 8

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.tools.extract_clip_features \
  --frames-dir /content/dota/frames --frames-subdir images \
  --ids-file "$KATVAD_DATA_ROOT/DoTA/labels_s8/test_ids.txt" \
  --dataset DoTA --stride 1 --batch-size 64 --device cuda \
  --no-center-crop \
  --output-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s1_ncc"
```

```python
from pathlib import Path
import numpy as np, os
src = Path(os.environ['KATVAD_CACHE_ROOT']) / 'clip' / 'DoTA_s1_ncc'
dst = Path(os.environ['KATVAD_CACHE_ROOT']) / 'clip' / 'DoTA_s8_ncc'
dst.mkdir(parents=True, exist_ok=True)
for p in sorted(src.glob('*.npy')):
    t = dst / p.name
    if t.exists():
        continue
    np.save(t, np.load(p)[::8])
print(len(list(dst.glob('*.npy'))), 'files')
```

Exact, not approximate: `range(0, N, 8)` is what a stride-8 extraction samples.
Point `--frames-dir` at **VM-local** frames, not Drive — minutes vs hours.

---

## 4. Config: how the gate gets recorded

Two ways to specify an arm. Prefer **(a)** for anything going into a table.

**(a) A YAML per arm** — self-documenting, and the file is the record:

```yaml
# configs/v3_a2_constant.yaml — plain-TSM control (arm A2)
kip:
  enabled: true
  gate_type: constant       # MANDATORY; omitting it raises on load
  const_shift_ratio: 0.5
  disable_pmg: true         # dataset returns zero flow rows
  use_lkin: false           # no L_kin
loss:
  lambda_rec: 0.0           # without these two, L_KIP_rec regresses the zeros
  lambda_align: 0.0
train:
  stage: 2
  amp: true
  num_epochs: 125
data:
  dataset: MSAD-full
```

**(b) `--set` overrides** — what §5 uses, because the arm matrix is short.

Either way, `core/train.py` writes the **fully resolved** config to
`<output-dir>/config.yaml`, `gate_type` included. That file is the durable record
of what an arm was — the checkpoint alone cannot tell `rank` from `constant`.

`core/evaluate.py` writes **no** config. An eval directory therefore does not
record which gate scored it. §5.4 closes that by hand.

---

## 5. Training (MSAD-full only)

Sizing, unchanged from the seed campaign: default `batch_size=64` over MSAD-full
gives ~4 optimizer steps/epoch, so `num_epochs=125` ≈ **500 steps**. Warm-up is
20 steps; peak LR 5e-5. `checkpoint_every_steps=100` costs ~1.4 GB per stage-2
run on Drive — keep the step checkpoints until the analysis has used them, then
prune. **One experiment = one `--output-dir`** (`metrics.jsonl` appends).

Seeds: `2024 2025 2026`, matching the existing A0 arms.

### 5.1 Stage 1 — KIP warm-up (needed by A1, A2b, A3, A4; **not** by A0 or A2)

**What stage 1 actually trains** (`core/train.py:185-189`): it sets
`requires_grad = name.startswith("kip.")` — so **only KIP's parameters move.**
The temporal encoder, fusion and heads stay at initialization. Two consequences
that decide the whole arm design:

- **`kip.enabled=false` + `stage=1` raises**
  (`ValueError: Stage 1 (KIP warm-up) requires kip.enabled=true`). There is no
  such thing as a KIP-off stage 1 in this tree.
- **A warm-started stage 2 therefore carries no trunk advantage** — only trained
  KIP weights. So **cold A0 is a fair comparator**, and the "warm start is the
  mechanism" confound that `COLAB.md` §A4.1 chased does not exist here.
  (`COLAB.md`'s `checkpoint_last_nokip.pt` is **not produced by this tree** —
  ignore that path.)

**Only two stage-1 runs per seed are needed**, not four. Stage 1's loss is
`lambda_rec·L_KIP_rec + lambda_align·L_KIP_align`, neither of which reads `v^k`,
so **no gate type receives gradient in stage 1** — the run is gate-independent in
everything but its checkpoint key layout:

| Stage-1 run | Serves | Why |
|---|---|---|
| `gate_type=rank` | **A1, A2b** | parameter-free gates share a key layout, so `constant` loads it |
| `gate_type=mlp_frozen` | **A3, A4** | both own `kip.shift.mlp.*`; `mlp_ste` loads it |

#### 5.1a Stage 1 for the parameter-free family (serves A1 and A2b)

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026

python -m core.train \
  --set train.stage=1 --set data.dataset=MSAD-full \
  --set train.num_epochs=125 --set train.seed=$S \
  --set kip.gate_type=rank \
  --data-dir  "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/MSAD/MSAD-full" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_rank_s$S/stage1"
```

#### 5.1b Stage 1 for the MLP family (serves A3 and A4)

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026

python -m core.train \
  --set train.stage=1 --set data.dataset=MSAD-full \
  --set train.num_epochs=125 --set train.seed=$S \
  --set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm \
  --data-dir  "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/MSAD/MSAD-full" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_mlp_frozen_s$S/stage1"
```

**A2 skips stage 1 entirely** — its KIP losses are zeroed and the constant gate
ignores `ê_O`, so there is nothing a warm-up could train that anything reads.
**A0 cannot run stage 1 at all** (it raises, see above).

#### 5.1c Check the precondition before any stage 2 (spec v3 §8, unenforced)

```python
import json, pathlib, os
FAM = 'rank'          # or 'mlp_frozen'
S = 2024
p = (pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])
     / f'MSAD_{FAM}_s{S}' / 'stage1' / 'metrics.jsonl')
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
rec = [r['kip_rec'] for r in rows if 'kip_rec' in r]
first, last = sum(rec[:20])/20, sum(rec[-50:])/50
print(f'first 20 mean {first:.3f} | last 50 mean {last:.3f} | ratio {last/first:.2f}')
assert last < 0.6 * first, (
    f'STOP: L_KIP_rec did not roughly halve ({first:.3f} -> {last:.3f}). '
    'Stage 2 would rank a badly-trained e_O and still hand you a publishable '
    'number. This arm has no result until stage 1 converges.')
```

The v1 reference trajectory was mean **19.2 → 9.7** with a plateau from ~step
200. A last-50 mean that has not roughly halved is a **stop**, not a warning.

### 5.2 Stage 2 — one explicit block per arm

Every block below is **copy-paste complete**. Set `S` at the top, run, then
change `S` to 2025 and 2026 and run again. Nothing is parameterized by a
variable you have to remember to change twice.

**The output dirs are the contract with §6.** Train an arm somewhere else and
§6's `--ckpt` paths will not find it:

| Arm | Train dir (`$OUT` = `$KATVAD_OUTPUT_ROOT`) | Stage 1? | Flow? |
|---|---|---|---|
| **A0** KIP-off | `$OUT/MSAD_kipoff_s$S/stage2` | no (raises) | no |
| **A1** rank | `$OUT/MSAD_rank_s$S/stage2_kip_on` | §5.1a | yes |
| **A2** plain-TSM | `$OUT/MSAD_constant_s$S/stage2_kip_on` | **no** | **no** |
| **A2b** constant+PMG | `$OUT/MSAD_constant_pmg_s$S/stage2_kip_on` | §5.1a | yes |
| **A3** mlp_frozen | `$OUT/MSAD_mlp_frozen_s$S/stage2_kip_on` | §5.1b | yes |
| **A4** mlp_ste | `$OUT/MSAD_mlp_ste_s$S/stage2_kip_on` | §5.1b | yes |

#### 5.2-A2 — plain-TSM control ⭐ RUN THIS ONE FIRST

No stage 1, no flow, no KIP losses. All six `kip`/`loss` flags are required; drop
any one and you are not running a plain-TSM control (see §2.2's warning).

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=MSAD-full \
  --set train.num_epochs=125 --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.gate_type=constant --set kip.const_shift_ratio=0.5 \
  --set kip.disable_pmg=true \
  --set loss.lambda_rec=0 --set loss.lambda_align=0 --set kip.use_lkin=false \
  --data-dir  "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_constant_s$S/stage2_kip_on"
```

#### 5.2-A0 — KIP-off baseline (the arm every Δ subtracts from)

Cold by construction: stage 1 requires KIP, and `warm_start_model` refuses a
KIP-on checkpoint into a KIP-off model (`core/train.py:104-108`). That is fine —
stage 1 trains **only** `kip.*`, so the warm arms carry no trunk advantage
(§5.1). No gate flags: a KIP-off config makes no claim about the gate.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=MSAD-full \
  --set train.num_epochs=125 --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.enabled=false \
  --data-dir  "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_kipoff_s$S/stage2"
```

#### 5.2-A1 — the v3 model (rank gate)

Needs **§5.1a** for this seed, and §5.1c must have passed.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=MSAD-full \
  --set train.num_epochs=125 --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.gate_type=rank \
  --init-weights "$KATVAD_OUTPUT_ROOT/MSAD_rank_s$S/stage1/checkpoint_last.pt" \
  --data-dir  "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/MSAD/MSAD-full" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_rank_s$S/stage2_kip_on"
```

#### 5.2-A2b — constant gate, PMG genuinely trained

Separates "the smoother did it" from "the auxiliary losses did it". Identical to
A2's gate, but `L_KIP_rec` / `L_KIP_align` / `L_kin` stay on and are fed **real**
RAFT targets. Warm-starts from the **rank** stage 1 — parameter-free gates share
a key layout, so it loads without complaint.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=MSAD-full \
  --set train.num_epochs=125 --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.gate_type=constant --set kip.const_shift_ratio=0.5 \
  --init-weights "$KATVAD_OUTPUT_ROOT/MSAD_rank_s$S/stage1/checkpoint_last.pt" \
  --data-dir  "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/MSAD/MSAD-full" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_constant_pmg_s$S/stage2_kip_on"
```

#### 5.2-A3 — `mlp_frozen`, the bit-identical v1 bridge

The arm that connects every number in `core/docs/RESULTS_*.md` to the v3 tree.
**Never describe this arm as motion-gated** (lesson **C24**): its gate is a
near-constant smoother at `s ≈ 58–69`. Needs **§5.1b**.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=MSAD-full \
  --set train.num_epochs=125 --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm \
  --init-weights "$KATVAD_OUTPUT_ROOT/MSAD_mlp_frozen_s$S/stage1/checkpoint_last.pt" \
  --data-dir  "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/MSAD/MSAD-full" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_mlp_frozen_s$S/stage2_kip_on"
```

#### 5.2-A4 — `mlp_ste`, the pure gradient intervention

Forward pass bit-identical to A3; the backward pass is unblocked, so the 321-param
gate MLP and `ê_O` finally receive gradient from `L_MIL`. **Warm-starts from the
`mlp_frozen` stage 1 deliberately** — sharing stage 1 with A3 removes the one
difference that is not the gradient, which is the entire point of `A4 − A3`.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=MSAD-full \
  --set train.num_epochs=125 --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm \
  --init-weights "$KATVAD_OUTPUT_ROOT/MSAD_mlp_frozen_s$S/stage1/checkpoint_last.pt" \
  --data-dir  "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/MSAD/MSAD-full" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_mlp_ste_s$S/stage2_kip_on"
```

### 5.3 Resume, and time-boxed sessions

- Dead session: same command with `--init-weights ...` replaced by
  `--resume "<output-dir>/checkpoint_last.pt"`. The two flags are mutually
  exclusive; resume carries the warm-started weights forward anyway.
- `--stop-after-epochs N` caps one invocation without shrinking the LR horizon.
- Checkpoints store model/optim/sched/scaler + all RNG states, so resume replays
  the exact batch stream.
- **Lesson C15:** checkpoints pickle `np.random.get_state()`. A Colab runtime
  that drifts to a different numpy major makes every checkpoint on Drive
  unloadable *before reaching a tensor*. Do not pin `numpy<2` away; recover with
  the `find_class` unpickler in `COLAB.md` §A4.0.

### 5.4 Bookkeeping — the one manual step v3 requires

The run manifest (lesson **C17**) is not built. `config.yaml` records `gate_type`
but not `--init-weights`, `--data-dir`, `--clip-dir`, `--knn-cache`, the commit,
or `sys.argv`. Write it yourself, once per run, into the output dir:

```python
import json, os, subprocess, pathlib
run = pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])/'MSAD_constant_s2024'/'stage2_kip_on'
(run/'run_manifest.json').write_text(json.dumps({
    'arm': 'A2 plain-TSM control',
    'gate_type': 'constant', 'const_shift_ratio': 0.5, 'disable_pmg': True,
    'lambda_rec': 0.0, 'lambda_align': 0.0, 'use_lkin': False,
    'seed': 2024, 'init_weights': None,
    'clip_dir': f"{os.environ['KATVAD_CACHE_ROOT']}/clip/MSAD_ncc",
    'flow_dir': None,
    'knn_cache': f"{os.environ['KATVAD_CACHE_ROOT']}/knn/MSAD_ncc/knn_cache.npz",
    'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],
                                          cwd='/content/drive/MyDrive/Thesis-V3/kat-vad').decode().strip(),
    'prediction': 'r=0.5 reproduces most of the +0.09 DoTA gain',
}, indent=2))
```

Do the same in every **eval** output dir — `evaluate.py` writes no config, so
without this an eval directory does not record which gate produced it.

---

## 6. Evaluation

Deterministic since the verbalizer was seeded — one run per checkpoint is
meaningful. Scoring reads cached features: minutes, not hours.

**Two rules make this section long. Read them before copying any block.**

1. **The model is built from the CLI, not from the checkpoint.** The **gate**
   flags must repeat on *every* eval command; omitting them is a wrong number or
   a raise. The **loss** flags must not — `lambda_rec`, `lambda_align` and
   `use_lkin` shape training only, and `disable_pmg` only controls whether the
   dataset reads flow. Passing those at eval is harmless noise.
2. **`rank` and `constant` checkpoints are key-identical** (§0 fact 2). Scoring
   an A1 checkpoint under `--set kip.gate_type=constant` loads **cleanly**,
   raises nothing, and hands you a publishable number for a model that was never
   trained that way. `load_kip_state_dict` keys the mismatch check on
   `kip.shift.mlp.*`, and neither of those two gates has any. **§6.8 is the only
   thing between you and that number.** It is not optional.

### 6.0 The eval matrix — what has to run

Every trained arm gets **two** evals from the *same* `checkpoint_last.pt`: MSAD
in-domain and DoTA zero-shot. That is 2 evals × 3 seeds × 6 arms = **36 evals**,
plus the one-off reference arm in §6.9. Nothing below trains anything, and every
one reads cached features — minutes, not hours.

| Arm | `--ckpt` from | MSAD eval → | DoTA eval → | Gate flags at eval |
|---|---|---|---|---|
| **A0** | `MSAD_kipoff_s$S/stage2` | `MSAD_kipoff_s$S/eval_msad` | `DoTA_kipoff_s$S/eval_dota` | `--set kip.enabled=false` |
| **A1** | `MSAD_rank_s$S/stage2_kip_on` | `MSAD_rank_s$S/eval_msad` | `DoTA_rank_s$S/eval_dota` | `--set kip.gate_type=rank` |
| **A2** | `MSAD_constant_s$S/stage2_kip_on` | `MSAD_constant_s$S/eval_msad` | `DoTA_constant_s$S/eval_dota` | `--set kip.gate_type=constant --set kip.const_shift_ratio=0.5` |
| **A2b** | `MSAD_constant_pmg_s$S/stage2_kip_on` | `MSAD_constant_pmg_s$S/eval_msad` | `DoTA_constant_pmg_s$S/eval_dota` | same as A2 |
| **A3** | `MSAD_mlp_frozen_s$S/stage2_kip_on` | `MSAD_mlp_frozen_s$S/eval_msad` | `DoTA_mlp_frozen_s$S/eval_dota` | `--set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm` |
| **A4** | `MSAD_mlp_ste_s$S/stage2_kip_on` | `MSAD_mlp_ste_s$S/eval_msad` | `DoTA_mlp_ste_s$S/eval_dota` | `--set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm` |

**Which mistakes the code catches, and which it does not** — verified by reading
`core/models/ckpt_compat.py:169-192`, whose only test is presence or absence of
`kip.shift.mlp.*`:

| You score… | under… | Result |
|---|---|---|
| `rank` ckpt | `mlp_frozen` / `mlp_ste` | ✅ **raises** (`_GATE_MISSING_MSG`) |
| `mlp_*` ckpt | `rank` / `constant` | ✅ **raises** (`_GATE_MISMATCH_MSG`) |
| **`rank` ckpt** | **`constant`** | ❌ **loads clean, wrong number, no warning** |
| **`constant` ckpt** | **`rank`** | ❌ **loads clean, wrong number, no warning** |
| `mlp_frozen` ckpt | `mlp_ste` | ⚠️ loads clean — but **forward is bit-identical**, so the number is the same. Harmless |

The two ❌ rows are why **§6.8 is mandatory**. The ⚠️ row is genuinely safe: the
STE changes only the backward pass, and eval has no backward pass.

Two flag details the table cannot show:

- **`const_shift_ratio` is range-checked only** (`core/config.py:108`). Passed
  beside `rank` it is silently accepted and silently unused. Don't — `evaluate.py`
  writes no config, so the command line is the only record of the arm.
- **`gate_signal` beside `rank`/`constant` raises** (`core/config.py:100-104`).
  It is the one gate flag the config protects you from.

### 6.1 A2 — plain-TSM control ⭐ EVAL THIS ONE FIRST

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026
CKPT="$KATVAD_OUTPUT_ROOT/MSAD_constant_s$S/stage2_kip_on/checkpoint_last.pt"

# --- MSAD-full, in-domain ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=MSAD-full \
  --set kip.gate_type=constant --set kip.const_shift_ratio=0.5 \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_constant_s$S/eval_msad" \
  --save-scores

# --- DoTA, zero-shot ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=DoTA \
  --set kip.gate_type=constant --set kip.const_shift_ratio=0.5 \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_constant_s$S/eval_dota" \
  --save-scores
```

### 6.2 A0 — KIP-off baseline

No gate flags, deliberately. A0 carries **no** `kip_*` diagnostics in its
`.npz` (`core/evaluate.py:134` ANDs in `cfg.kip.enabled`) — an empty diagnostic
block here is correct, not a failure.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026
CKPT="$KATVAD_OUTPUT_ROOT/MSAD_kipoff_s$S/stage2/checkpoint_last.pt"

# --- MSAD-full, in-domain ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=MSAD-full --set kip.enabled=false \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_kipoff_s$S/eval_msad" \
  --save-scores

# --- DoTA, zero-shot ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=DoTA --set kip.enabled=false \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_kipoff_s$S/eval_dota" \
  --save-scores
```

### 6.3 A1 — the v3 model (rank gate)

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026
CKPT="$KATVAD_OUTPUT_ROOT/MSAD_rank_s$S/stage2_kip_on/checkpoint_last.pt"

# --- MSAD-full, in-domain ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=MSAD-full --set kip.gate_type=rank \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_rank_s$S/eval_msad" \
  --save-scores

# --- DoTA, zero-shot ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=DoTA --set kip.gate_type=rank \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_rank_s$S/eval_dota" \
  --save-scores
```

> **No `const_shift_ratio` here.** It would be accepted and ignored, and it would
> make the command read like a `constant` arm. §6.8 is what actually proves this
> was scored as `rank`.

### 6.4 A2b — constant gate, PMG trained

Byte-identical flags to A2 (§6.1). **Only `--ckpt` and `--output-dir` differ** —
get those wrong and the two arms merge with no error anywhere.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026
CKPT="$KATVAD_OUTPUT_ROOT/MSAD_constant_pmg_s$S/stage2_kip_on/checkpoint_last.pt"

# --- MSAD-full, in-domain ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=MSAD-full \
  --set kip.gate_type=constant --set kip.const_shift_ratio=0.5 \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_constant_pmg_s$S/eval_msad" \
  --save-scores

# --- DoTA, zero-shot ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=DoTA \
  --set kip.gate_type=constant --set kip.const_shift_ratio=0.5 \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_constant_pmg_s$S/eval_dota" \
  --save-scores
```

### 6.5 A3 — `mlp_frozen`, the v1 bridge

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026
CKPT="$KATVAD_OUTPUT_ROOT/MSAD_mlp_frozen_s$S/stage2_kip_on/checkpoint_last.pt"

# --- MSAD-full, in-domain ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=MSAD-full \
  --set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_mlp_frozen_s$S/eval_msad" \
  --save-scores

# --- DoTA, zero-shot ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=DoTA \
  --set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_mlp_frozen_s$S/eval_dota" \
  --save-scores
```

### 6.6 A4 — `mlp_ste`, the gradient intervention

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024        # re-run with 2025, then 2026
CKPT="$KATVAD_OUTPUT_ROOT/MSAD_mlp_ste_s$S/stage2_kip_on/checkpoint_last.pt"

# --- MSAD-full, in-domain ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=MSAD-full \
  --set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_mlp_ste_s$S/eval_msad" \
  --save-scores

# --- DoTA, zero-shot ---
python -m core.evaluate \
  --ckpt "$CKPT" \
  --set data.dataset=DoTA \
  --set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_mlp_ste_s$S/eval_dota" \
  --save-scores
```

### 6.7 Score norm — the same on every block above

**Leave `--score-norm` at `auto` and never pass it explicitly** (lesson **C12**).
Confirm the log line instead:

```
INFO core.metrics: score-norm auto -> none   (MSAD-full;  ≈50% normal videos)
INFO core.metrics: score-norm auto -> minmax (1394/1397 videos abnormal; 0.21% normal, threshold 5%)
```

**Never put a raw-pooled MSAD number and a min-max-pooled DoTA number in the same
column.** Always report `auc_macro` beside the micro AUC.

### 6.8 Verify the gate you actually scored — MANDATORY

Run this after **every** eval block above. Because `rank` and `constant`
checkpoints load into each other without complaint (§6.0), this cell is the only
thing that proves which gate produced a number.

```python
import numpy as np, pathlib, os

EXPECTED = {                      # gate_type -> (min span, max span) of 128
    'rank':       (100, 128),
    'mlp_frozen': (0, 8),
    'mlp_ste':    (0, 8),         # forward bit-identical to mlp_frozen
    'constant':   (0, 0),
}

def check(eval_dir: str, gate_type: str, n: int = 200) -> None:
    d = pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT']) / eval_dir / 'scores'
    files = sorted(d.glob('*.npz'))[:n]
    assert files, f'no scores in {d}'
    spans, means, ratios = [], [], []
    for p in files:
        z = np.load(p)
        assert 'kip_s' in z, f'{p.name}: no kip_s — KIP-off arm, or --dump-kip-diag was off'
        s = z['kip_s'].astype(np.int32)
        spans.append(s.max() - s.min()); means.append(s.mean())
        ratios.append(float(z['kip_gate_ratio'].mean()))
    lo, hi = EXPECTED[gate_type]
    span_mean = float(np.mean(spans))
    print(f'{eval_dir}: span mean {span_mean:.1f} (min {min(spans)}, max {max(spans)}) '
          f'of 128 | s mean {np.mean(means):.1f} | ratio mean {np.mean(ratios):.3f}')
    assert lo <= span_mean <= hi, (
        f'GATE MISMATCH: {eval_dir} was scored as {gate_type!r} but span '
        f'{span_mean:.1f} is outside [{lo}, {hi}]. This eval is VOID — do not '
        f'read its AUC.')

S = 2024
for gate, run in [('constant', 'constant'), ('rank', 'rank'),
                  ('constant', 'constant_pmg'), ('mlp_frozen', 'mlp_frozen'),
                  ('mlp_ste', 'mlp_ste')]:
    for bench, root in [('eval_msad', 'MSAD'), ('eval_dota', 'DoTA')]:
        d = f'{root}_{run}_s{S}/{bench}'
        if (pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])/d/'scores').is_dir():
            check(d, gate)
```

| Arm | Expected span (of 128) | `s` mean | If it comes back otherwise |
|---|---|---|---|
| `rank` (A1) | near **128 on every clip** | spread | the rank gate is not doing its job — stop, do not read the AUC |
| `mlp_frozen` (A3) | **0–4** | `≈ 58–69` | you are not running the gate you think you are |
| `mlp_ste` (A4) | **0–4** | `≈ 58–69` | forward is identical to A3 by construction; a difference means a flag did not take |
| `constant` (A2, A2b) | exactly **0** | `64` at `r = 0.5` | a flag did not take |
| `enabled=false` (A0) | *no `kip_*` keys at all* | — | a `kip_s` array in a KIP-off arm means the wrong checkpoint was scored |

**A span of 0 on an arm you believe is `rank` is the failure this section exists
for**: the `rank` checkpoint was scored under `gate_type=constant`, the load
succeeded, and the AUC is meaningless.

**Windowing caveat:** the rank is computed *within* a `data.max_vis_len` window,
so `s_t` resets at each window boundary. That is what the scored model does — it
is not a logging artifact, and it is a real property of the method to state in
any write-up.

### 6.9 The `gate_a` reference arm (once, gate-independent)

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.evaluate \
  --baseline-ckpt "$KATVAD_CKPT_ROOT/best.ckpt" \
  --set kip.enabled=false --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_gate_a/gate_a" --save-scores
```

LaGoVAD's released trunk, no KIP, reused across every v3 arm — run it once. Gate
against **the released checkpoint's own number**, not the paper's printed one
(lesson **C8b**).

### 6.10 Rescore — pooling as a variable, not an assumption

```bash
%%bash
for d in "$KATVAD_OUTPUT_ROOT"/MSAD_*_s*/eval_msad \
         "$KATVAD_OUTPUT_ROOT"/DoTA_*_s*/eval_dota \
         "$KATVAD_OUTPUT_ROOT"/DoTA_gate_a/gate_a; do
  [ -d "$d/scores" ] && python -m core.tools.rescore --run-dir "$d"
done
```

Recomputes the metrics from the saved `.npz` under every pooling rule. Cheap, and
it is the check that catches a silently wrong `auto` resolution.

### 6.11 The eval-side manifest

`core/evaluate.py` writes **no** config (§4), so an eval directory does not record
which gate produced it — and for `rank`-vs-`constant` the scores alone cannot tell
you either. Write the manifest right after §6.8 passes:

```python
import json, os, pathlib

S, ARM, GATE, RUN = 2024, 'A2 plain-TSM control', 'constant', 'constant'
OUT = os.environ['KATVAD_OUTPUT_ROOT']
for bench, root, ddir, cdir in [
    ('MSAD-full', 'MSAD', f"{os.environ['KATVAD_DATA_ROOT']}/MSAD",
     f"{os.environ['KATVAD_CACHE_ROOT']}/clip/MSAD_ncc"),
    ('DoTA', 'DoTA', f"{os.environ['KATVAD_DATA_ROOT']}/DoTA/labels_s8",
     f"{os.environ['KATVAD_CACHE_ROOT']}/clip/DoTA_s8_ncc"),
]:
    sub = 'eval_msad' if root == 'MSAD' else 'eval_dota'
    ev = pathlib.Path(OUT) / f'{root}_{RUN}_s{S}' / sub
    if not ev.is_dir():
        continue
    (ev / 'eval_manifest.json').write_text(json.dumps({
        'arm': ARM, 'gate_type': GATE, 'const_shift_ratio': 0.5,
        'benchmark': bench, 'seed': S,
        'ckpt': f'{OUT}/MSAD_{RUN}_s{S}/stage2_kip_on/checkpoint_last.pt',
        'data_dir': ddir, 'clip_dir': cdir,
        'gate_check_span_mean': None,   # paste §6.8's printed span here
    }, indent=2))
    print('wrote', ev / 'eval_manifest.json')
```

### 6.12 A2 before A1 — the stop point

Run §6.1 for **A2 only**, all three seeds, then §6.8, then §7's `A2 − A0` row.
**Stop and read it before training A1.** If the fixed 50 % smoother reproduces
most of the +0.09, A1 answers a different question than you think you are
asking, and three stage-1 runs are the wrong thing to spend next.

---

## 7. Analysis and decision rules

Use the analysis cell in `DOTA_EVAL.md` §4 unchanged — paired bootstrap over
clips, per-clip win/loss, the ego/non-ego mechanism split, per-class deltas.

Report Δ per seed, then the **seed-level t-interval over n = 3**. Do not report
"18 bootstrap CIs": 3 seeds × 3 metrics × 2 controls over-counts what are 3
independent replications.

| Comparison | What a positive Δ means |
|---|---|
| **A2 − A0** | **The decisive one.** A fixed 50 % smoother, alone, moves DoTA. If this reproduces most of +0.09, the contribution is temporal smoothing |
| | *A0 = `MSAD_kipoff_s$S` / `DoTA_kipoff_s$S` (§5.2-A0, §6.2), **cold**. There is no warm A0 in this tree — stage 1 requires KIP (`core/train.py:186`) and trains only `kip.*`, so the warm arms carry no trunk advantage and cold A0 is a fair comparator (§5.1)* |
| **A1 − A2** | What the *rank gate* adds on top of smoothing. This is the only number that supports a motion-gating claim |
| **A1 − A0** | The v3 headline, uninterpretable without the two rows above |
| **A2b − A2** | What the auxiliary KIP losses buy when the gate is fixed |
| **A3 − A0** | The v1 bridge — should land near the archived +0.09 |
| **A4 − A3** | Pure gradient effect: forward bit-identical, backward unblocked |

Standing rules:

- **MSAD is a bounded null.** The honest statement is "any in-domain effect is
  < ≈1 AUC point at n = 3", not "costs nothing". The MSAD-null / DoTA-positive
  asymmetry *is* the finding.
- **Do not tune anything on the +0.09** (lesson 14). The mechanism is exactly
  what is under test; tuning against it is fitting the benchmark.
- An A/B run under a known-open precondition defect measures the defect
  (lesson **C14**). If §5.1's stage-1 check failed, the arm has no number —
  report it as blocked, not as a result.

Record outcomes in a new `core/docs/RESULTS_V3_*.md`. **Do not retro-edit the
existing `RESULTS_*.md` numbers** — they were true for what ran (`mlp_frozen`);
add the qualifier at the point of next citation.

---

## 8. Order of work

```
[ ] 1.1–1.3  sync v3, env vars, verify caches exist          (10 min, CPU)
[ ] 3.x      build only what §1.3 found missing              (hours, GPU, resumable)

--- the decisive experiment, cheapest in the program -----------------------
[ ] 5.2-A2   plain-TSM control, S=2024 then 2025 then 2026   (3 × stage 2, no stage 1, no flow)
[ ] 5.2-A0   KIP-off baseline, same 3 seeds                  (3 × stage 2, cold)
[ ] 6.1      eval A2 — MSAD + DoTA, 3 seeds                  (6 evals, minutes)
[ ] 6.2      eval A0 — MSAD + DoTA, 3 seeds                  (6 evals)
[ ] 6.8      gate check on A2: span must be 0, s = 64        (seconds — assert, not eyeball)
[ ] 6.9      gate_a reference arm, once                      (minutes)
[ ] 6.11     eval manifests                                  (seconds)
[ ] 7        A2 − A0, all three seeds. STOP AND READ IT.     (§6.12)

--- only after reading the A2 − A0 row -------------------------------------
[ ] 5.1a     stage 1, parameter-free family, 3 seeds
[ ] 5.1c     stage-1 convergence assert — a fail means the arm has NO result
[ ] 5.2-A1   rank, 3 seeds
[ ] 6.3      eval A1 — MSAD + DoTA, 3 seeds
[ ] 6.8      gate check on A1: span must be ≈128 (a span of 0 = VOID eval)
[ ] 7        A1 − A2 — the only motion-gating evidence there is

--- follow-up, as the results demand ---------------------------------------
[ ] 5.2-A2b + 6.4    what the auxiliary KIP losses buy (needs 5.1a)
[ ] 5.1b + 5.2-A3 + 6.5   the v1 bridge; should land near the archived +0.09
[ ] 5.2-A4 + 6.6     pure gradient effect (reuses 5.1b's stage 1)

A2 before A1 is deliberate. If the smoother reproduces the gain, the A1 arm
answers a different question than you thought you were asking, and you want to
know that before spending three stage-1 runs on it.

---

## 9. Pitfalls, mapped to lessons

| Don't | Why | Lesson |
|---|---|---|
| Reuse a config.yaml from an archived KIP-on run | It raises — and that is the feature | C14, C24 |
| Say "motion-gated" of `mlp_frozen` | It is a fixed ~50 % smoother, measured | **C24** |
| Score a checkpoint under a different gate | Refused for `mlp_*`; **silently accepted** between `rank` and `constant` — §6.8 is the only detector | C5 |
| Read an AUC before running the §6.8 gate check | A `rank` arm scored as `constant` loads clean and returns a plausible number | C5, C24 |
| Mix pooling rules in one table | Min-max vs raw is a different metric, not a different run | C12 |
| Change the transform, stride, or split mid-campaign | Invalidates every cache *and every metric measured on it* | C2, C13 |
| Trust a stage-2 arm whose stage 1 did not converge | Nothing raises; you get a plausible number for an untrained `ê_O` | spec v3 §8 |
| `--set` your way past a missing flow cache | `require_flow=false` trains the PMG head to predict zeros | C14 |
| Treat `kip.disable_pmg=true` as "PMG off" | It only zeroes the flow *targets*; `L_KIP_rec` then regresses zeros. Zero the loss weights too | C14 |
| Report a Δ without its per-seed spread | n = 3; a single seed is not a result | — |
| Train KIP on PreVAD | It ships no pixels; RAFT targets are unbuildable. Settled 2026-08-29 | — |

---

## 10. Out of scope here

- **PreVAD** — `core/docs/PREVAD_SETUP.md`. KIP-off trunk + Gate P0 only.
- **Checkpoint selection / validation split** — deferred (`msad-ncc-seeds-and-selection.md`
  Phase B). Every arm above reads `checkpoint_last`, deep in the overfit regime;
  that confound is shared by all arms and cancels in the Δ, but it caps the
  absolute numbers.
- **The run manifest and the stage-1 assert** — deferred to the training-pipeline
  plan (v3 plan §7 item 1). §5.1 and §5.4 are the manual stand-ins.
- **Alert-CLIP** — no public checkpoint exists. Every number here is a **stock
  CLIP ViT-B/16** number.
- **Phase 7 ATS/MLLM reports** — off the critical path.
