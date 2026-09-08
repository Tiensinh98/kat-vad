# KAT-VAD v3 — DADA-2000 training + evaluation runbook

**Written 2026-09-03, against branch `v3`.** The experiment layer on top of
`core/docs/DATA_SETUP.md`, which owns everything about getting DADA-2000 onto
disk (unzip, frame-order gate, labels, CLIP, RAFT, KNN, sizing). **Do §1–§10
of that document first, including §6's sanity check.** Nothing here is
meaningful without it.

Sibling runbooks: `MSAD_DOTA_V3_SETUP.md`, `TAD_V3_SETUP.md`. This document
deliberately mirrors their structure and arm names, because the whole point of
running a third training corpus is that all three campaigns compare row for
row.

> `core/data/dada.py` builds **both** splits unconditionally (no TAD-style
> `--with-train-split` flag), and its output trains — covered by
> `core/tests/test_dada.py::TestDadaTrains`, so a DADA-2000 arm cannot fail for
> a *data* reason that was never exercised. **On `main` that test exercises the
> v1 gate only**; the `gate_type` matrix this runbook prescribes is covered on
> branch `v3`, which is also the only branch that can parse it.

---

## 0. What this campaign is for

Not a headline. A **second replication**, alongside TAD's.

`core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md` (2026-09-01) settled the
attribution on MSAD-full: a fixed 50 % channel shift with no flow, no PMG head
and no KIP losses (**A2**) is statistically indistinguishable from full v1 KIP
on zero-shot DoTA — Δ(A2 − V1) = +0.0109, t95 [−0.0588, +0.0805], 6/6 metrics
include zero — while A2 − A0 = **+0.1025 ± 0.0350**. The `rank` gate (**A1**)
is the *worst* KIP arm: A1 − A2 = −0.0683, CI [−0.0790, −0.0580]. TAD's job
(`TAD_V3_SETUP.md` §0) is to check whether that ordering survives training on
a corpus that is not MSAD. DADA-2000's job is the same question on a corpus
that is also not TAD — a third, independent training distribution, and (via
§4.4 below) a **second** zero-shot transfer target beyond DoTA.

**Pre-registered prediction (write this into the run notes before any arm is
trained):** `A2 − A0 > 0` and `A1 − A2 <= 0` on zero-shot DoTA when trained on
DADA-2000, reproducing the MSAD/TAD ordering.

| Result | Reading |
|---|---|
| Ordering replicates a third time | The smoother finding is corpus-independent across MSAD, TAD, DADA-2000 — as strong a claim as this project can make without a fourth corpus |
| Ordering inverts | Corpus dependence, and DADA-2000's dashcam/ego-motion character (closer to DoTA's own domain than MSAD or TAD) is the natural first hypothesis for why |
| Everything null | DADA-2000's train split may be too small or too homogeneous per category (61 fine-grained types, most with few clips) to move a transfer number — report as a bounded null, not a failure |

**Do not tune anything on a DADA-2000 number** (lesson 14).

---

## 1. Preconditions

### 1.1 Code, environment, data

`DATA_SETUP.md` §2 (install + env vars) and §3–§10 (ingest, labels, caches).
Then:

```bash
!python -c "from core.kip import gate_shift; print(gate_shift.GATE_TYPES)"
# expect: ('rank', 'mlp_frozen', 'mlp_ste', 'constant')
```

### 1.2 What must exist before §3

```bash
%%bash
for p in "$KATVAD_DATA_ROOT/DADA2000/labels_train.json" \
         "$KATVAD_DATA_ROOT/DADA2000/frame_labels_test.json" \
         "$KATVAD_DATA_ROOT/DADA2000/defs.json" \
         "$KATVAD_CACHE_ROOT/clip/DADA2000" \
         "$KATVAD_CACHE_ROOT/flow/v1/DADA2000" \
         "$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz" \
         "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
         "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
         "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
         "$KATVAD_DATA_ROOT/TAD"; do
  n=$(ls "$p" 2>/dev/null | wc -l); echo "$n  $p"
done
```

The DoTA and TAD rows are **zero-shot transfer** targets, reused unchanged
from their own campaigns — v3 changed model code only, so lesson **C2** does
not fire by reusing them here.

#### 1.2a Paths are as-run, and are *not* what this document said before 2026-09-06

Every command here now uses the paths the campaign **actually ran on**:

| | Path |
|---|---|
| CLIP features | `$KATVAD_CACHE_ROOT/clip/DADA2000` |
| DVS KNN cache | `$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz` |
| RAFT flow | `$KATVAD_CACHE_ROOT/flow/v1/DADA2000` |
| Every run | `$KATVAD_OUTPUT_ROOT/DADA2000/<arm>_s<S>/…` |

This document previously prescribed `clip/DADA2000_ncc`, `knn/DADA2000_ncc`
and `$OUT/DADA_<arm>_s$S`. **No such directory exists** — `collab/DADA/v3/train.py`
built and consumed the unsuffixed names, and all seven 2026-09-06 arms live
under them. Renaming now would orphan every 2024/2025 artifact and make the
seed sweep incomparable, so the doc moved to the code, not the other way round.

**The cost of that is real and is lesson C2's exact hazard:** the DADA cache
path does not record that it was built with `--no-center-crop`, unlike
`MSAD_ncc` / `DoTA_s8_ncc`, and `extract_clip_features.py` writes no manifest.
Close the gap without touching a byte of cache — run this once:

```bash
%%bash
printf 'transform=anisotropic_resize (--no-center-crop)\nstride=8\nbuilt=2026-09\n' \
  > "$KATVAD_CACHE_ROOT/clip/DADA2000/TRANSFORM.txt"
cat "$KATVAD_CACHE_ROOT/clip/DADA2000/TRANSFORM.txt"
```

A center-cropped re-extraction into this directory invalidates every number in
`RESULTS_DADA.md`. The marker is the only thing standing between you and doing
that silently.

### 1.3 Two facts that decide every command below

- **`data.dataset=DADA2000`** resolves to verbalizer key `dada`
  (`core/data/definitions.py:DATASET_NAME_TO_ABBR`), which shares
  `_DOTA_CLS_DEFS` with DoTA — taxonomy exactly `["Normal", "CarAccident"]`.
  `C = 2`, so `H_mul` is near-degenerate here too, the same caveat as TAD
  (`TAD_V3_SETUP.md` §1.3) — do not read anything into `mul` loss on this
  dataset.
- **`data.is_egocentric=true`** (the opposite default from TAD/MSAD). DADA-2000
  is dashcam footage — leaving this `false` silently reuses fixed-camera DVS
  tuning (`theta`/`delta_m` instead of `theta_ego`/`delta_m_ego`) on ego
  footage it was not chosen for.

### 1.4 Sizing — already computed: `E = 20`

`DATA_SETUP.md` §10 derives `num_epochs` from `2 x num_abnormal_train`. On the
2024 corpus build it printed **25 steps/epoch**, so `E = 20` gives the
project-standard **500 optimizer steps** (MSAD reached the same 500 as
4 x 125). Every block in §3 hardcodes it. **`E` is identical across arms and
both stages** — varying it puts a training-length difference inside every Δ.

Re-run this only if you rebuild the labels; if it prints anything but 20, the
split changed and §3's numbers no longer apply:

```python
import json, math, os
tr = json.load(open(f"{os.environ['KATVAD_DATA_ROOT']}/DADA2000/labels_train.json"))
A = sum(tr.values()); spe = math.ceil(2 * A / 64)
print(f'{A} abnormal -> len {2*A} -> {spe} steps/epoch -> E = {math.ceil(500 / spe)}')
```

---

## 2. The arm matrix on DADA-2000

Identical to `MSAD_DOTA_V3_SETUP.md` §2.2 and `TAD_V3_SETUP.md` §2 — same
names, same flags, so all three campaigns' rows line up.

| Arm | Flags added to the stage-2 command | Purpose | Needs |
|---|---|---|---|
| **A0** | `--set kip.enabled=false` | KIP-off trunk. The arm every Δ subtracts from | — |
| **A2** | `--set kip.gate_type=constant --set kip.const_shift_ratio=0.5 --set kip.disable_pmg=true --set loss.lambda_rec=0 --set loss.lambda_align=0 --set kip.use_lkin=false` | **Plain-TSM control.** All six flags required | — |
| **A1** | `--set kip.gate_type=rank` | The v3 model | stage 1 |
| **A2b** | `--set kip.gate_type=constant --set kip.const_shift_ratio=0.5` (+ `--flow-dir`) | Same smoother, PMG genuinely trained against real flow | stage 1 |
| **A3** | `--set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm` | The v1 bridge | stage 1 |
| **A4** | `--set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm` | Pure gradient intervention; forward bit-identical to A3 | stage 1 |

**A0 and A2 need neither a flow cache nor stage 1** — the cheapest,
first-run comparison. §6 exploits it.

> **`disable_pmg=true` does not disable the PMG head** — see
> `TAD_V3_SETUP.md` §2 for the full explanation. Zeroing `loss.lambda_rec` /
> `loss.lambda_align` / `kip.use_lkin` is what actually removes the pathway.

### 2.1 What raises, so you recognise it

Identical table to `TAD_V3_SETUP.md` §2.1, with `TAD` swapped for
`DADA2000` throughout (`ValueError: Training set needs both classes` still
means the labels were somehow not built — but `dada.py` builds
`labels_train.json` unconditionally, so seeing this means `--out-dir` points
somewhere stale, not a missing flag).

---

## 3. Training

Output dirs are the contract with §4.

| Arm | Train dir (`$OUT` = `$KATVAD_OUTPUT_ROOT`) | Stage 1? | Flow? |
|---|---|---|---|
| **A0** | `$OUT/DADA2000/kipoff_s$S/stage2` | no (raises) | no |
| **A2** | `$OUT/DADA2000/constant_s$S/stage2_kip_on` | **no** | **no** |
| **A1** | `$OUT/DADA2000/rank_s$S/stage2_kip_on` | §3.2a | yes |
| **A2b** | `$OUT/DADA2000/constant_pmg_s$S/stage2_kip_on` | §3.2a | yes |
| **A3** | `$OUT/DADA2000/mlp_frozen_s$S/stage2_kip_on` | §3.2b | yes |
| **A4** | `$OUT/DADA2000/mlp_ste_s$S/stage2_kip_on` | §3.2b | yes |

Seeds `2024 2025 2026`. **One experiment = one `--output-dir`**
(`metrics.jsonl` appends; a restart into the same dir silently duplicates
rows — lesson **17**).

### 3.0 The three constants every block below shares — set these, change nothing else

```
S=2024        # train seed. THIS is the one you change for 2025 / 2026.
E=20          # num_epochs -> 25 batches/epoch x 20 = 500 optimizer steps.
              # Verify with §1.4's cell; it printed 20 for the 2024 corpus build.
```

> **⚠ `core.data.dada --seed 2024` is a *different* seed — do not touch it.**
> That one draws the train/test **split**. Re-running the label build with
> `--seed 2025` re-draws the corpus, and every cross-seed Δ silently becomes a
> comparison between two different datasets. Sweep `train.seed` only; the label
> files under `$KATVAD_DATA_ROOT/DADA2000` are built **once**, at split seed
> 2024, and reused by all three train seeds (`DATA_SETUP.md` §5).

Everything below is copy-paste-ready at `S=2024`. Each block is one Colab cell.
The **Already run** line says which seeds exist on disk as of 2026-09-06
(`RESULTS_DADA.md` §1) — do not re-run those into the same `--output-dir`,
`metrics.jsonl` appends (lesson **17**).

### 3.1-A2 — plain-TSM control ⭐ RUN THIS FIRST

No stage 1, no flow cache. All six flags are required.
**Already run:** 2024, 2025.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024 ; E=20

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=DADA2000 \
  --set data.is_egocentric=true \
  --set train.num_epochs=$E --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.gate_type=constant --set kip.const_shift_ratio=0.5 \
  --set kip.disable_pmg=true \
  --set loss.lambda_rec=0 --set loss.lambda_align=0 --set kip.use_lkin=false \
  --data-dir  "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/constant_s$S/stage2_kip_on"
```

### 3.1-A0 — KIP-off baseline

The arm every Δ subtracts from. Cold by construction (stage 1 requires KIP), so
a warm A1/A2b/A3/A4 carries no trunk advantage cold A0 lacks — stage 1 freezes
everything except `kip.*` (`core/train.py:185-188`).
**Already run:** 2024.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024 ; E=20

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=DADA2000 \
  --set data.is_egocentric=true \
  --set train.num_epochs=$E --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.enabled=false \
  --data-dir  "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/kipoff_s$S/stage2"
```

### 3.2 Stage 1 — KIP warm-up (needed only by A1, A2b, A3, A4)

Stage 1's loss never reads `v^k`, so it is gate-independent except for
checkpoint key layout — **two** stage-1 runs per seed serve all four gated
arms. Note stage 1 sets no `train.amp`: it runs fp32, deliberately.

#### 3.2a Parameter-free family — serves A1 and A2b

**Already run:** 2024.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024 ; E=20

python -m core.train \
  --set train.stage=1 --set data.dataset=DADA2000 --set data.is_egocentric=true \
  --set train.num_epochs=$E --set train.seed=$S \
  --set kip.gate_type=rank \
  --data-dir  "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/DADA2000" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/rank_s$S/stage1"
```

#### 3.2b MLP family — serves A3 and A4

**Already run:** 2024.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024 ; E=20

python -m core.train \
  --set train.stage=1 --set data.dataset=DADA2000 --set data.is_egocentric=true \
  --set train.num_epochs=$E --set train.seed=$S \
  --set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm \
  --data-dir  "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/DADA2000" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/mlp_frozen_s$S/stage1"
```

#### 3.2c Convergence check — a stop, not a warning

Identical cell to `TAD_V3_SETUP.md` §3.2c, paths swapped to
`DADA2000/{rank,mlp_frozen}_s{S}/stage1`. No gradient reaches `ê_O` through the
shift under `rank`/`mlp_frozen`/`constant`, so an unconverged stage 1 hands you
a plausible but meaningless stage-2 number. Read the **ratio** against the first
window, not an absolute value — DADA-2000's flow statistics are their own
distribution.

> **This check FAILED on the 2024 run and the failure still stands.** Stage 1
> moved `kip_rec` 121.7 → 41.6 in 500 steps and was still falling; stage 2 ended
> at ≈22–30, against MSAD *opening* stage 2 at 7.4–8.5. Lesson **14** therefore
> applies to **A1, A2b and A3** on every seed run under this stage 1 — report
> them blocked, not as numbers. It does **not** apply to A2 or A0, which read no
> flow at all. (`RESULTS_DADA.md` §8.2.) Raising `E` for stage 1 alone would
> break the equal-training-length rule in §7; the honest fix is a separate,
> longer stage-1 budget applied to every gated arm at once.

### 3.3-A1 — the v3 model (`rank` gate)

Warm-starts from §3.2a. **Already run:** 2024.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024 ; E=20

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=DADA2000 \
  --set data.is_egocentric=true \
  --set train.num_epochs=$E --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.gate_type=rank \
  --init-weights "$KATVAD_OUTPUT_ROOT/DADA2000/rank_s$S/stage1/checkpoint_last.pt" \
  --data-dir  "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/DADA2000" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/rank_s$S/stage2_kip_on"
```

### 3.3-A2b — same smoother, PMG trained against real flow

Same gate as A2, but the KIP losses stay on and the flow cache is read.
Warm-starts from **§3.2a's `rank` stage 1** — legal because `rank` and
`constant` are both parameter-free, so the key sets match and `warm_start_model`
(fail-loud, lesson **5**) accepts it. **Already run:** 2024.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024 ; E=20

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=DADA2000 \
  --set data.is_egocentric=true \
  --set train.num_epochs=$E --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.gate_type=constant --set kip.const_shift_ratio=0.5 \
  --init-weights "$KATVAD_OUTPUT_ROOT/DADA2000/rank_s$S/stage1/checkpoint_last.pt" \
  --data-dir  "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/DADA2000" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/constant_pmg_s$S/stage2_kip_on"
```

### 3.3-A3 — the v1 bridge (`mlp_frozen`)

Warm-starts from §3.2b. **Already run:** 2024.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024 ; E=20

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=DADA2000 \
  --set data.is_egocentric=true \
  --set train.num_epochs=$E --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm \
  --init-weights "$KATVAD_OUTPUT_ROOT/DADA2000/mlp_frozen_s$S/stage1/checkpoint_last.pt" \
  --data-dir  "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/DADA2000" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/mlp_frozen_s$S/stage2_kip_on"
```

### 3.3-A4 — pure gradient intervention (`mlp_ste`)

Forward bit-identical to A3; only the backward pass differs. Warm-starts from
§3.2b's `mlp_frozen` stage 1 (both gates carry the same 321-param MLP, so the
keys match). **Already run:** 2024 — **and that run is not trustworthy**, see
the warning below.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024 ; E=20

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=DADA2000 \
  --set data.is_egocentric=true \
  --set train.num_epochs=$E --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm \
  --init-weights "$KATVAD_OUTPUT_ROOT/DADA2000/mlp_frozen_s$S/stage1/checkpoint_last.pt" \
  --data-dir  "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/DADA2000" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/mlp_ste_s$S/stage2_kip_on"
```

> **⚠ A4 trained through 33 NaN steps of 500 on the 2024 seed** — 17 of 20
> epochs, always `mil` + `mul_mil`, while `kip_rec`/`kip_align` stayed finite
> (`RESULTS_DADA.md` §8.1). It is the **only** arm where a gradient reaches the
> gate MLP and PMG head *through the score path*: `mlp_frozen`'s is identically
> zero (lesson **24**) and `rank`/`constant` are parameter-free. So it is the
> only arm exposed to the sigmoid surrogate in
> `shift_channels_straight_through` (`core/kip/gate_shift.py:98-136`) under
> `amp: true` — and `core/train.py` calls `scaler.step()` with **no
> `scaler.unscale_()` and no `clip_grad_norm_` anywhere**. `GradScaler` skips
> inf/NaN steps but does nothing about a large *finite* gradient, which is a
> sufficient path to a weight blow-up and an fp16 overflow on the next forward.
> **Before spending a seed on A4, falsify that:** re-run this cell once with
> `--set train.amp=false`, or add gradient clipping. If the NaNs vanish, the
> defect is closed and every A4 seed needs re-running under the fix. Until then
> A4's rows carry no claim except the §4.2 gate-range check.

### 3.4 Resume, and the manual manifest

Same as `TAD_V3_SETUP.md` §3.4 — dead session: `--resume` instead of
`--init-weights`; lesson **15** on checkpoint RNG pickling; lesson **17**'s
manifest is still not built by the training code, write it by hand per run
(swap `'corpus': 'TAD'` for `'corpus': 'DADA2000'` in the template there).

---

## 4. Evaluation

Every trained arm gets **two** evals from the same `checkpoint_last.pt`, and a
third once TAD's feature cache exists — DADA-2000 has its own genuine in-domain
test split *and* two zero-shot transfer targets. The 2026-09-06 campaign ran the
first two; `clip/TAD_ncc` was never built, so the TAD row below is aspirational
and §4.1 keeps it commented out:

| Eval | Benchmark | `--data-dir` | `--clip-dir` | `--score-norm auto` resolves to |
|---|---|---|---|---|
| **in-domain** | DADA-2000 test | `$KATVAD_DATA_ROOT/DADA2000` | `clip/DADA2000` | **raw** (`DATA_SETUP.md` §5) |
| **zero-shot → DoTA** | DoTA val (1,402 clips, ~all abnormal) | `$KATVAD_DATA_ROOT/DoTA/labels_s8` | `clip/DoTA_s8_ncc` | **min-max** |
| **zero-shot → TAD** | TAD test (100 clips, ~40 % normal) | `$KATVAD_DATA_ROOT/TAD` | `clip/TAD_ncc` | **raw** |

Never mix pooling rules in one column; always report `auc_macro` beside the
micro number (lesson **12**).

Two rules before you copy any block (identical to `TAD_V3_SETUP.md` §4):

1. The **gate** flags must repeat on every eval command; the **loss** flags
   (`lambda_rec`, `lambda_align`, `use_lkin`, `disable_pmg`) must not.
2. **`rank` and `constant` checkpoints are key-identical** — scoring one
   under the other's gate flag loads cleanly and hands you a number for a
   model never trained that way. §4.2 is the only detector.

### 4.1 Six eval blocks, one per arm — copy-paste at `S=2024`

Every block scores the arm's own `stage2*/checkpoint_last.pt` on DADA-2000
in-domain and on DoTA zero-shot. The **TAD** eval is appended as a third command
in the A2 and A0 blocks only, commented out: it needs
`$KATVAD_CACHE_ROOT/clip/TAD_ncc`, which the 2026-09-06 campaign did not have
(`TAD_V3_SETUP.md`, nothing trained or extracted yet). Uncomment it once that
cache exists, and add it to the other four blocks the same way.

Two rules before you edit any of them:

1. The **gate** flags repeat on every eval command; the **loss** flags
   (`lambda_rec`, `lambda_align`, `use_lkin`, `disable_pmg`) must **not** — they
   are train-time only (`core/train.py:593`), and so is `data.is_egocentric`
   (`core/train.py:600`). Neither changes the eval graph or the state dict.
2. **`rank` and `constant` checkpoints are key-identical** — scoring one under
   the other's gate flag loads cleanly and hands you a number for a model never
   trained that way. §4.2 is the only detector. Run it.

`--score-norm auto` is the parser default (`core/evaluate.py`), so writing it is
belt-and-braces; it resolves to **raw** on DADA-2000 and **min-max** on DoTA.

#### A2 — plain-TSM control

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024
CKPT="$KATVAD_OUTPUT_ROOT/DADA2000/constant_s$S/stage2_kip_on/checkpoint_last.pt"
GATE="--set kip.gate_type=constant --set kip.const_shift_ratio=0.5"

# --- DADA-2000, in-domain ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DADA2000 \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/constant_s$S/eval_dada"

# --- DoTA, zero-shot transfer ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/constant_s$S/eval_dota"

# --- TAD, zero-shot transfer (needs clip/TAD_ncc; not built as of 2026-09-06) ---
# python -m core.evaluate --ckpt "$CKPT" $GATE \
#   --set data.dataset=TAD \
#   --data-dir "$KATVAD_DATA_ROOT/TAD" \
#   --clip-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
#   --score-norm auto --save-scores \
#   --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/constant_s$S/eval_tad"
```

#### A0 — KIP-off baseline

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024
CKPT="$KATVAD_OUTPUT_ROOT/DADA2000/kipoff_s$S/stage2/checkpoint_last.pt"
GATE="--set kip.enabled=false"

# --- DADA-2000, in-domain ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DADA2000 \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/kipoff_s$S/eval_dada"

# --- DoTA, zero-shot transfer ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/kipoff_s$S/eval_dota"

# --- TAD, zero-shot transfer (needs clip/TAD_ncc; not built as of 2026-09-06) ---
# python -m core.evaluate --ckpt "$CKPT" $GATE \
#   --set data.dataset=TAD \
#   --data-dir "$KATVAD_DATA_ROOT/TAD" \
#   --clip-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
#   --score-norm auto --save-scores \
#   --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/kipoff_s$S/eval_tad"
```

#### A1 — `rank` (the v3 model)

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024
CKPT="$KATVAD_OUTPUT_ROOT/DADA2000/rank_s$S/stage2_kip_on/checkpoint_last.pt"
GATE="--set kip.gate_type=rank"

# --- DADA-2000, in-domain ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DADA2000 \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/rank_s$S/eval_dada"

# --- DoTA, zero-shot transfer ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/rank_s$S/eval_dota"
```

#### A2b — `constant` + PMG

Gate flags are **A2's**, not A1's, even though the checkpoint warm-started from
a `rank` stage 1. This is exactly the case rule 2 above warns about — §4.2 is
what proves you got it right.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024
CKPT="$KATVAD_OUTPUT_ROOT/DADA2000/constant_pmg_s$S/stage2_kip_on/checkpoint_last.pt"
GATE="--set kip.gate_type=constant --set kip.const_shift_ratio=0.5"

# --- DADA-2000, in-domain ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DADA2000 \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/constant_pmg_s$S/eval_dada"

# --- DoTA, zero-shot transfer ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/constant_pmg_s$S/eval_dota"
```

#### A3 — `mlp_frozen` (the v1 bridge)

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024
CKPT="$KATVAD_OUTPUT_ROOT/DADA2000/mlp_frozen_s$S/stage2_kip_on/checkpoint_last.pt"
GATE="--set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm"

# --- DADA-2000, in-domain ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DADA2000 \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/mlp_frozen_s$S/eval_dada"

# --- DoTA, zero-shot transfer ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/mlp_frozen_s$S/eval_dota"
```

#### A4 — `mlp_ste` (pure gradient intervention)

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024
CKPT="$KATVAD_OUTPUT_ROOT/DADA2000/mlp_ste_s$S/stage2_kip_on/checkpoint_last.pt"
GATE="--set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm"

# --- DADA-2000, in-domain ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DADA2000 \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/mlp_ste_s$S/eval_dada"

# --- DoTA, zero-shot transfer ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/mlp_ste_s$S/eval_dota"
```

Transfer evals live **inside the training arm's own directory**
(`DADA2000/<arm>_s$S/eval_dota`), not in a shared `DoTA_*` tree. That is what
keeps three training corpora from colliding on one benchmark's output dir
(lesson **17**) — MSAD's campaign owns `DoTA_constant_s$S`, and a DADA-trained
arm must never be written there.

### 4.2 Verify the gate you actually scored — MANDATORY

The one check that catches a checkpoint scored under the wrong gate. `rank` and
`constant` are key-identical, so nothing else will tell you. Run it over every
eval dir produced by §4.1 before reading a single AUC.

```python
import numpy as np, pathlib, os

EXPECTED = {'rank': (100, 128), 'mlp_frozen': (0, 8),
            'mlp_ste': (0, 8), 'constant': (0, 0)}

def check(eval_dir: str, gate_type: str, n: int = 200) -> None:
    d = pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT']) / eval_dir / 'scores'
    files = sorted(d.glob('*.npz'))[:n]
    assert files, f'no scores in {d}'
    empty = [p.name for p in files if p.stat().st_size == 0]
    assert not empty, f'0-byte score files (lesson 11b): {empty[:3]}'
    spans, means, ratios = [], [], []
    for p in files:
        z = np.load(p)
        assert 'kip_s' in z, f'{p.name}: no kip_s -- KIP-off arm, or diag was off'
        s = z['kip_s'].astype(np.int32)
        spans.append(s.max() - s.min()); means.append(s.mean())
        ratios.append(float(z['kip_gate_ratio'].mean()))
    lo, hi = EXPECTED[gate_type]; span_mean = float(np.mean(spans))
    print(f'{eval_dir}: span mean {span_mean:.1f} (min {min(spans)}, max {max(spans)}) '
          f'of 128 | s mean {np.mean(means):.1f} | ratio mean {np.mean(ratios):.3f}')
    assert lo <= span_mean <= hi, (
        f'GATE MISMATCH: {eval_dir} scored as {gate_type!r} but span {span_mean:.1f} '
        f'is outside [{lo}, {hi}]. This eval is VOID -- do not read its AUC.')

S = 2024
for gate, run in [('constant', 'constant'), ('rank', 'rank'),
                  ('constant', 'constant_pmg'), ('mlp_frozen', 'mlp_frozen'),
                  ('mlp_ste', 'mlp_ste')]:
    for d in (f'DADA2000/{run}_s{S}/eval_dada',
              f'DADA2000/{run}_s{S}/eval_dota',
              f'DADA2000/{run}_s{S}/eval_tad'):
        if (pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])/d/'scores').is_dir():
            check(d, gate)
```

A0 has no `kip_*` keys at all (KIP is not instantiated) — it is not in the loop
by design, and a `kip_s` appearing in a `kipoff_*` score file means the wrong
checkpoint was scored.

The expected-span table is identical to `TAD_V3_SETUP.md` §4.2 and
`MSAD_DOTA_V3_SETUP.md`: the property being checked — which gate the checkpoint
was actually built with — does not depend on the training corpus. On the
2026-09-06 DADA run this **passed on every arm**, and reproduced lesson **24** /
H4′ on a third corpus: `mlp_frozen` sat at `s = 61` on *every frame of every
clip* (ratio sd 0.001) and `mlp_ste` at 62, while `rank` spanned the full
0–128 (`RESULTS_DADA.md` §2, §7).

### 4.3 Per-arm index — what §4.1's blocks resolve to

| Arm | `--ckpt` | Eval dirs (both under the arm's own tree) | Gate flags at eval |
|---|---|---|---|
| **A0** | `DADA2000/kipoff_s$S/stage2` | `kipoff_s$S/eval_{dada,dota}` | `--set kip.enabled=false` |
| **A2** | `DADA2000/constant_s$S/stage2_kip_on` | `constant_s$S/eval_{dada,dota}` | `--set kip.gate_type=constant --set kip.const_shift_ratio=0.5` |
| **A1** | `DADA2000/rank_s$S/stage2_kip_on` | `rank_s$S/eval_{dada,dota}` | `--set kip.gate_type=rank` |
| **A2b** | `DADA2000/constant_pmg_s$S/stage2_kip_on` | `constant_pmg_s$S/eval_{dada,dota}` | same as A2 |
| **A3** | `DADA2000/mlp_frozen_s$S/stage2_kip_on` | `mlp_frozen_s$S/eval_{dada,dota}` | `--set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm` |
| **A4** | `DADA2000/mlp_ste_s$S/stage2_kip_on` | `mlp_ste_s$S/eval_{dada,dota}` | `--set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm` |

### 4.4 Reference arms

Three, all gate-independent, each run once:

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
# gate_d0 -- LaGoVAD's released trunk on DADA-2000 (DATA_SETUP.md §6; no published number to gate against)
python -m core.evaluate --baseline-ckpt "$KATVAD_CKPT_ROOT/best.ckpt" \
  --set kip.enabled=false --set data.dataset=DADA2000 \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" --clip-dir "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/gate_d0"

# gate_a / T0 -- the same trunk on DoTA / TAD; reuse existing campaigns' runs if they exist
python -m core.evaluate --baseline-ckpt "$KATVAD_CKPT_ROOT/best.ckpt" \
  --set kip.enabled=false --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_gate_a/gate_a"

python -m core.evaluate --baseline-ckpt "$KATVAD_CKPT_ROOT/best.ckpt" \
  --set kip.enabled=false --set data.dataset=TAD \
  --data-dir "$KATVAD_DATA_ROOT/TAD" --clip-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD_gate_t0/gate_t0"
```

Gate against **the released checkpoint's own number** on each benchmark
(lesson **8b**) — DADA-2000 has none published for this protocol at all, so
`gate_d0` *is* the reference, not a check against one.

### 4.5 Rescore and manifest

```bash
%%bash
for d in "$KATVAD_OUTPUT_ROOT"/DADA2000/*_s*/eval_dada \
         "$KATVAD_OUTPUT_ROOT"/DADA2000/*_s*/eval_dota \
         "$KATVAD_OUTPUT_ROOT"/DADA2000/*_s*/eval_tad \
         "$KATVAD_OUTPUT_ROOT"/DADA2000/gate_d0; do
  [ -d "$d/scores" ] && python -m core.tools.rescore --run-dir "$d"
done
```

Write an `eval_manifest.json` per eval dir (`MSAD_DOTA_V3_SETUP.md` §6.11's
cell, `DADA2000` paths, §4.2's span pasted in). **Before any Δ**, assert the
score-file count against `results.json:num_videos` in every arm (lesson
**11b**) — `evaluate --save-scores` is not atomic:

```python
import json, os, pathlib
OUT = pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])
for d in sorted(OUT.glob('DADA2000/*_s*/eval_*')):
    r = d / 'results.json'
    if r.exists():
        n_files = len(list((d/'scores').glob('*.npz'))) if (d/'scores').is_dir() else 0
        n_rep = json.load(open(r)).get('num_videos')
        flag = '' if n_files == n_rep else '   <-- MISMATCH'
        print(f'{d.relative_to(OUT)}: scores={n_files} results={n_rep}{flag}')
```

---

## 5. Analysis and decision rules

Use `DOTA_EVAL.md` §4's analysis cell unchanged for both transfer targets —
paired bootstrap over clips, per-clip win/loss, ego/non-ego split, per-class
deltas. Report Δ per seed, then the seed-level t-interval over n = 3, exactly
as MSAD's and TAD's campaigns do — never report per-metric clip-bootstrap CIs
as if they were independent replications (the audit's correction, still
binding: `activeContext.md` 2026-08-22).

| Comparison | What it answers |
|---|---|
| **A2 − A0** (DoTA, zero-shot) | Does the fixed-shift smoother move transfer when trained on DADA-2000, matching MSAD's +0.1025 and (if run) TAD's replication? |
| **A2 − A0** (TAD, zero-shot) | A second transfer target — the one TAD's own campaign cannot check against itself |
| **A1 − A2** (DoTA, TAD) | The only evidence that could support motion-gating over a fixed smoother. On MSAD this was −0.0683, CI excluding zero |
| **A2 − A0** (DADA-2000, in-domain) | On MSAD the in-domain effect was a bounded null (+0.0036 ± 0.0076). Does DADA-2000, whose test split is genuinely comparable in-domain (unlike TAD's), agree? |
| **A1 − A0** | The v3 headline, uninterpretable without the rows above |
| **A2b − A2** | What the auxiliary KIP losses buy with the gate fixed |
| **A3 − A0**, **A4 − A3** | The v1 bridge; pure gradient effect |

Standing rules, unchanged from `TAD_V3_SETUP.md` §5:

- **A small test set has wide intervals.** Read `DATA_SETUP.md` §5's printed
  test-split size before trusting any in-domain Δ at n = 3 seeds.
- **In-domain DADA-2000 is not comparable to any external number** — there is
  none pinned for this protocol (§0). Only `gate_d0` is a reference, and it is
  our own measurement, not a paper's.
- **An A/B run under a known-open precondition defect measures the defect**
  (lesson **14**). If §3.2c's convergence check failed, the arm has no
  number — report it blocked.
- **Do not tune on any Δ.**

Record outcomes in a new `core/docs/v3/RESULTS_V3_DADA.md`. Do not retro-edit
`RESULTS_V3_GATE_ATTRIBUTION.md` or a TAD results doc — each was true for what
ran on its own corpus; add cross-corpus qualifiers at the point of next
citation.

---

## 6. Order of work

```
--- data, no code changes needed -------------------------------------------
[ ] DATA_SETUP §2-§4   session, unzip, TWO ingest gates (esp. §4.3 frame order)
[ ] DATA_SETUP §5      build both splits; record the counts
[ ] DATA_SETUP §7      CLIP features, test ids first, then the full corpus
[ ] DATA_SETUP §6      sanity check with the released checkpoint

--- the rest of the caches --------------------------------------------------
[ ] DATA_SETUP §9      DVS KNN cache (--motion-key)
[ ] §1.4               compute E once

--- the decisive pair, no flow, no stage 1 ----------------------------------
[x] §3.1-A2   plain-TSM control          DONE s2024, s2025
[x] §3.1-A0   KIP-off baseline           DONE s2024
[x] §4.1      eval DADA + DoTA           DONE (TAD eval never run: no clip/TAD_ncc)
[x] §4.2      gate check                 DONE, PASS (RESULTS_DADA.md §2)
[x] §4.4      gate_d0 reference          DONE
[ ] §4.5      rescore, manifests, score-count assertion
[x] §5        A2 - A0 on DoTA            DONE = -0.0918. ORDERING INVERTED vs MSAD.

--- flow, and the gated arms ------------------------------------------------
[x] DATA_SETUP §8      flow targets, train ids only          DONE
[x] §3.2a + §3.2c      stage 1 (rank family)                 RUN, CONVERGENCE FAILED
[x] §3.3-A1 + §4       the v3 model                          DONE s2024 (blocked, lesson 14)
[x] §5                 A1 - A2 = +0.0300                     also inverted vs MSAD

--- follow-up, as the results demand ----------------------------------------
[x] A2b, A3, A4        DONE s2024 (A4 NaN-corrupted, §3.3-A4's warning)
```

### 6.1 What is actually left — the 2026-09-06 state

Read `RESULTS_DADA.md` §10 first; this is the same queue in run order.

```
[ ] Re-run the label build's window audit and drop the 4 clips that lost their
    anomaly window at stride 8 (DATA_SETUP.md §5.1). Test-split only, no
    retraining, and it moves the clip-oracle and every DADA micro number.
[ ] Frame-level linear probe on the cached DADA CLIP features against the real
    frame labels -- the decisive experiment, ~30 min, no training. Separates
    "frozen CLIP cannot represent an accident frame" from "weak supervision
    cannot find it". Everything below is conditional on it.
[ ] Falsify the A4 NaN hypothesis: one A2024 re-run at train.amp=false
    (§3.3-A4's warning). Cheap, and it either closes the defect or promotes it.
[ ] Seeds 2025 / 2026 for A0 (and 2026 for A2) -- the only pair whose Δ is not
    blocked by the stage-1 convergence failure. A1/A2b/A3 seeds buy nothing
    until stage 1 converges.
[ ] Only if the probe is positive: re-extract at stride 2-4 with
    score_head_kernel=3. FIRES LESSON C2 -- invalidates every number above.
```

A2 before A1, exactly as on MSAD and TAD — if the smoother reproduces the
gain on a third corpus, spending a stage-1 run and a full RAFT pass on A1
first would have answered a question you did not yet know you were asking.

---

## 7. Pitfalls specific to this campaign

| Don't | Why | Lesson |
|---|---|---|
| Compare an in-domain DADA-2000 number to a paper AUC | None is pinned for this protocol — there is nothing to compare against | §0 |
| Reuse `DoTA_constant_s$S` or `TAD_from_TAD_constant_s$S`-shaped dirs for a DADA-trained arm | Those belong to other campaigns; three corpora need three sets of dirs | **17** |
| Score a `rank` checkpoint under `constant` | Loads clean, raises nothing, wrong number. §4.2 is the only detector | **5**, **24** |
| Say "motion-gated" of `mlp_frozen` | Measured as a fixed ~50 % smoother over 8 seeds on MSAD; nothing about a new corpus changes that measurement | **24** |
| Vary `num_epochs` between arms | The Δ then contains a training-length difference | §1.4 |
| Change `core.data.dada --seed` when sweeping train seeds | That is the **split** seed. A new split re-draws the corpus and every cross-seed Δ compares two different datasets | §3.0 |
| Re-extract CLIP features into `clip/DADA2000` without checking `TRANSFORM.txt` | The dir name does not record `--no-center-crop`; a cropped re-extraction voids every number in `RESULTS_DADA.md` silently | **C2**, §1.2a |
| Read an `mlp_ste` row trained under `amp: true` | 33 of 500 steps carried NaN on the 2024 seed, and there is no gradient clipping in `core/train.py` | §3.3-A4 |
| Quote a DADA micro AUC without the 0.9069 clip-oracle beside it | 74 % of test frames are from all-normal clips; micro there measures clip classification | **C12**, `RESULTS_DADA.md` §4 |
| Set `data.is_egocentric=false` | DADA-2000 is dashcam footage; that flag selects the wrong DVS tuning | §1.3 |
| Read anything into DADA-2000's `mul` loss | `C = 2`; `H_mul` is near-degenerate on this taxonomy | §1.3 |
| Compute a Δ before asserting score-file counts | One 0-byte `.npz` perturbs every paired Δ in the study | **11b** |
| `--set kip.disable_pmg=true` to dodge a missing flow cache | Trains the PMG head to regress zeros | **14** |
| Restart training into an existing `--output-dir` | `metrics.jsonl` appends silently | **17** |
| Assume weak-abnormal (windowless) train clips can be a test candidate | `core/data/dada.py` forces them into train by construction; do not hand-edit `test_ids.txt` to add one | `DATA_SETUP.md` §0.2 |

---

## 8. Out of scope here

- **Data preparation** — `core/docs/DATA_SETUP.md`, all of it.
- **`core/data/dada.py` implementation** — `core/tests/test_dada.py`.
- **The MSAD campaign** — `MSAD_DOTA_V3_SETUP.md`,
  `RESULTS_V3_GATE_ATTRIBUTION.md`.
- **The TAD campaign** — `TAD_V3_SETUP.md`.
- **PreVAD** — `core/docs/PREVAD_SETUP.md`. KIP-off trunk + Gate P0 only; it
  ships no pixels, so no KIP-on arm exists there, ever.
- **Checkpoint selection.** DADA-2000 ships no official val split here
  either; every arm reads `checkpoint_last`.
