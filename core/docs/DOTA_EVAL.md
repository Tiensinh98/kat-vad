# DoTA zero-shot evaluation — setup, protocol, decision rules

**Written:** 2026-08-01 · **Companion to:** `core/docs/RESULTS_MSAD.md` (§6.6),
`core/docs/COLAB.md` (the MSAD run sequence this mirrors).

Scores three existing checkpoints on the DoTA val split. **No training.** One
feature-extraction job, three eval runs, one analysis cell.

---

## 0. What this experiment is — and what it is not

MSAD-full gave KIP +0.0012 AUC with a 95 % CI of [−0.0060, +0.0089]. That is a
null result on a fixed-camera benchmark whose KIP-off baseline already sits at
0.905 — near-saturated, and appearance-separable. DoTA is the opposite: 1,402
ego-centric driving clips where the anomaly *is* a kinematic event, and where
the baseline scores 62.60. **If the motion pathway does anything, it does it
here.**

| | |
|---|---|
| **Is** | A paired A/B of KIP-on vs KIP-off, same training data, same seed, same features, differing only by `kip.enabled`. |
| **Is** | A zero-shot cross-domain transfer test: MSAD-trained (fixed camera) → DoTA (ego camera). |
| **Is not** | A DoTA-trained result. Do not write "KAT-VAD on DoTA = X" in a paper from this. |
| **Is not** | Comparable to LaGoVAD's published **62.60**, which is PreVAD-trained. Only the `gate_a` arm (their released checkpoint) is. |

The one number that carries scientific weight is **Δ(KIP-on − KIP-off)** and its
confidence interval. Everything else is context.

---

## 1. The data, as it actually is

Verified against `data/DoTA/metadata_val.json` + `val_split.txt` (both already
in the repo — the 55 GB `DoTA_full.zip` is not):

| Item | Value |
|---|---|
| Clips | **1,402** (val split), every one abnormal |
| Ego-involved / other | **805 / 597** (from the `anomaly_class` prefix) |
| Raw frames | 142,747 at native **10 fps**; mean 101.8, median 99, range 26–284 |
| Anomaly window | mean 33.7 frames, median 29, **1 % below 7 frames** |
| Positive base rate | **33.1 %** at any stride |
| Anomaly classes | 20 fine-grained (`ego: turning` 279, `other: turning` 206, `ego: lateral` 153, …) |
| Frame layout after unzip | `frames/{video_id}/images/000000.jpg` (6-digit, 0-indexed) |

> **2026-08-08 — this protocol has been corrected and the experiment has been
> run.** The pooling rule documented below in its first version (raw pooled
> scores) was wrong, and put all three arms at chance. Results, the diagnosis
> and the corrected numbers: **`core/docs/RESULTS_DOTA.md`**. §2.1 below now
> carries the verified rule. The other change since: the pipeline is moving to
> `no_center_crop` features, so the runs described here are re-done into
> `*_ncc` paths (`core/docs/COLAB.md`).

**Label parity — verified, not assumed.** LaGoVAD's `dota_test_anno.json`
stores `anomaly_span` as normalized `[anomaly_start/num_frames,
anomaly_end/num_frames]`, and rebuilds frame labels as `round(frac ×
feature_length)` filled half-open (`LaGoVAD-PreVAD/src/datasets/base.py:57-69`).
`core/data/dota.py` reproduces that arithmetic exactly; the spans it derives
from `metadata_val.json` match all **1,402** of the baseline's entries to
floating-point equality. Our ground truth *is* their ground truth.

---

## 2. Stride — settled, not a judgement call

`LaGoVAD-PreVAD/tools/extract_feat_clip.py:39` extracts with `interval=8`.
So **stride 8 is the LaGoVAD-comparable setting**, and it is also the stride our
MSAD checkpoints were trained at — the temporal scale the encoder and KIP's
gate/shift learned. Both constraints point the same way.

| Stride | Sampled frames | Positive rate | Clips whose window rounds away |
|---:|---:|---:|---:|
| 1 | 142,747 | 0.3314 | 0 |
| 2 | 71,822 | 0.3295 | 2 |
| 4 | 36,235 | 0.3306 | 2 |
| **8** | **18,478** | **0.3311** | **3** |

(Measured by running `core.data.dota` at each stride, not estimated.)

Run the **primary** experiment at stride 8. The 3 clips that lose their window
(`90gyBengKDs_003658`, `bcpfYpmDcp0_000972`, `ezY7QrSWeWw_001685` — anomalies of
1–7 raw frames) contribute only negative frames; the baseline's own label code
does the same thing silently, so matching it is the correct choice.

**Extract once at stride 1 anyway.** Sampling is `range(0, N, stride)`, so
stride-8 features are *exactly* `stride1_features[::8]` — one extraction serves
every stride, and the finer strides become a free temporal-scale ablation (§7).

---

## 2.1 Score pooling — verified by reproduction, not by reading

**Every clip in DoTA val is abnormal** (1,394 of 1,397 usable; the 3 exceptions
are windows that round away at stride 8). So the task is purely *within-clip*
localization, and a micro AUC that concatenates all clips into one ranking puts
each clip's absolute score scale into the metric where it carries no label
information. A confident clip's negatives then outrank a hesitant clip's
positives.

LaGoVAD's own DoTA script min-max normalizes each clip before accumulating:

```python
# LaGoVAD-PreVAD/src/offline_evals/offline_dota_eval.py
pred_normed_score = (pred_score - pred_score.min()) / (pred_score.max() - pred_score.min())
metric.update(pred_normed_score, frame_label)
```

Its generic `full_length_eval.py` harness pools raw. **The two disagree, and the
reproduction decides which one the paper used:**

| `gate_a` = LaGoVAD released `best.ckpt` | AUC |
|---|---:|
| raw pooled | 0.5055 (chance) |
| **per-clip min-max** | **0.6142** |
| per-clip z-score | 0.6236 |
| published | 0.6260 |

So: **per-clip min-max**. `core/evaluate.py --score-norm auto` (the default)
resolves this from the label distribution — min-max when normal videos are under
5 % of the test set — so MSAD keeps raw pooling and DoTA gets normalized without
either being a dataset-name special case. `results.json` records `score_norm`,
`auc`, `auc_raw` and `auc_macro`.

`auc_macro` (mean per-clip AUC over clips with both classes) needs no
normalization at all and is the honest localization metric here. Quote it
alongside.

**This is a post-processing step.** Changing the pooling rule never needs
re-running inference — `python -m core.tools.rescore --run-dir <dir>` recomputes
every rule from the saved `.npz` curves.

## 3. Colab setup

Assumes `core/docs/COLAB.md`'s environment block is already run (Drive mounted,
`KATVAD_*` env vars set, repo at `$DRIVE/kat-vad`).

### 3.1 Unzip only the val clips

`DoTA_full.zip` holds all 4,677 clips (~55 GB). The val split is ~30 % of that,
so selective extraction saves both disk and time.

**Where to unzip.** `/content` (local NVMe, ephemeral) or Drive (persistent,
FUSE). The frames are an *intermediate* — the artifact worth persisting is
`clip/{video_id}.npy` (~300 MB total, §3.3), and once that exists the frames are
dead weight. The val split is ~280 k small jpgs, and Drive's FUSE mount charges
~50–100 ms per file open, so the cost of keeping them there is paid **twice**:

| | `/content` | Drive |
|---|---|---|
| unzip | ~10–20 min | slow; partial-extract failures observed (§3.1.1) |
| extraction reads 280 k files | ~20–35 min | hours, and rate-limit-prone |
| survives a disconnect | no | yes |
| re-do cost after disconnect | one unzip | none |

Recommended: **unzip to `/content`, extract, write features to Drive.** A
disconnect then costs one unzip, not a re-extraction. Unzipping to Drive is a
legitimate choice if you would rather never unzip again — just budget the
extraction pass accordingly, and read §3.1.1 if the count comes up short.

The cell below uses `/content`; swap the two paths to target Drive.

```bash
%%bash
set -e
mkdir -p /content/dota
cd /content/dota

# 1,402 patterns, one per val clip (~42 KB of argv — well under ARG_MAX)
python - <<'PY'
ids = [l.strip() for l in open('/content/drive/MyDrive/Thesis/data/DoTA/val_split.txt') if l.strip()]
open('/content/dota/patterns.txt','w').write('\n'.join(f'frames/{i}/*' for i in ids))
print(len(ids), 'patterns')
PY

df -h /content | tail -1   # need ~20 GB free
unzip -q "$KATVAD_DATA_ROOT/DoTA/DoTA_full.zip" $(tr '\n' ' ' < patterns.txt) -d /content/dota
ls /content/dota/frames | wc -l    # expect 1402
ls /content/dota/frames/$(ls /content/dota/frames | head -1)/images | head -3
```

If the archive's central directory makes selective extraction slow, fall back to
a full `unzip -q` (needs ~60 GB free) — the rest of the pipeline is unchanged.

**Sanity gate:** `ls frames | wc -l` must print **1402**. Anything less and the
label builder will refuse to run (§3.2 raises on a missing frame folder) — go to
§3.1.1 before doing anything else.

Two things that make this gate fail in practice:

- **Extracting into Drive.** Writing ~280 k small files through the FUSE mount
  is where truncated runs have actually been seen (1,120 of 1,397 folders
  landed, `unzip` still exiting 0). `unzip -n` is resumable, so re-running the
  same command fills the gap — but confirm with §3.1.1 that the gap is a write
  failure and not an archive that never had those clips.
- **A split file that isn't the one you think.** `data/DoTA/val_split.txt` in
  this repo has **1402** ids. If your Drive copy logs a different count
  (`Read N split ids`), you are evaluating a different denominator than §1
  documents. Reconcile the two files *first* — a number computed on 1,397 clips
  is not the number computed on 1,402.

### 3.1.1 If the unzip came up short

Symptom:

```
ValueError: 5/1402 annotated clips are unreadable under .../frames (subdir='images'):
0 with no frame folder [], 5 with an empty one ['TNZv-NBcV5U_002389', ...]
```

A clip fails for one of three reasons, and they need different fixes — an
existing directory is not evidence of data, so diagnose before repairing:

```python
import os, subprocess
ROOT = '/content/dota/frames'            # the --frames-dir you passed
ZIP  = os.environ['KATVAD_DATA_ROOT'] + '/DoTA/DoTA_full.zip'
ids  = [l.strip() for l in open(os.environ['KATVAD_DATA_ROOT'] + '/DoTA/val_split.txt') if l.strip()]

on_disk = set(os.listdir(ROOT)) if os.path.isdir(ROOT) else set()
usable  = {d for d in on_disk if os.path.isdir(f'{ROOT}/{d}/images') and os.listdir(f'{ROOT}/{d}/images')}
missing = [i for i in ids if i not in usable]
print(f'split={len(ids)}  folders={len(on_disk)}  usable={len(usable)}  missing={len(missing)}')

# which flavour of unreadable?
absent    = [i for i in missing if i not in on_disk]                       # no folder at all
no_subdir = [i for i in missing if i in on_disk and not os.path.isdir(f'{ROOT}/{i}/images')]
empty     = [i for i in missing if i not in absent and i not in no_subdir] # images/ exists, 0 files
print(f'  absent={len(absent)} {absent[:3]}\n  no images/ subdir={len(no_subdir)} {no_subdir[:3]}'
      f'\n  empty images/={len(empty)} {empty[:3]}')

# is the clip in the archive at all? (reads the central directory only, ~seconds)
names = subprocess.run(['unzip','-Z1',ZIP], capture_output=True, text=True).stdout.splitlines()
in_zip = {n.split('/')[1] for n in names if n.startswith('frames/') and n.count('/') >= 2}
print('clips in zip:', len(in_zip), '| missing AND absent from zip:',
      len([i for i in missing if i not in in_zip]))

open('/content/dota/missing.txt','w').write('\n'.join(missing))
```

| Reading | Cause | Fix |
|---|---|---|
| missing ids **are** in the zip (folder absent *or* `images/` empty) | unzip truncated — a FUSE write that failed after `mkdir` leaves the empty directory behind, which is why counting folders is not enough | re-unzip **only** those ids, below |
| missing ids **absent** from the zip | the archive is not the full release | get the rest, or accept partial coverage (§3.6) |
| folder exists, no `images/` inside at all | different layout in this archive | fix `--frames-subdir` (pass `--frames-subdir ""` if jpgs sit directly in the clip folder) |

`folders > split` is fine and expected — `DoTA_full.zip` holds all 4,990 clips,
and a pattern that over-matched simply leaves extra folders around. The
preprocessor logs how many it ignored and evaluates the split only.

Re-unzip just the gap (`-n` never overwrites what is already there, so this is
resumable and safe to run repeatedly):

```bash
%%bash
set -e
cd /content/dota
df -h /content | tail -1                     # need headroom before starting
sed 's|^|frames/|; s|$|/*|' missing.txt > missing_patterns.txt
unzip -qn "$KATVAD_DATA_ROOT/DoTA/DoTA_full.zip" $(tr '\n' ' ' < missing_patterns.txt) -d /content/dota
ls /content/dota/frames | wc -l
```

Then re-run §3.2. Repeat until the count matches the split.

### 3.2 Build the label files (stride 8, primary)

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad

python -m core.data.dota \
  --metadata   "$KATVAD_DATA_ROOT/DoTA/metadata_val.json" \
  --split-file "$KATVAD_DATA_ROOT/DoTA/val_split.txt" \
  --frames-dir "$KATVAD_DATA_ROOT/DoTA/frames" --frames-subdir images \
  --out-dir    "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --stride 8 --allow-missing-frames
```

Writes `frame_labels_test.json`, `defs.json` (`["Normal", "CarAccident"]` — the
definition set LaGoVAD's verbalizer uses for `dota`), an empty
`labels_train.json`, `meta.json` (carries the **`ego_involve`** flag, needed for
the mechanism check in §6), and `test_ids.txt`.

`--frames-dir` makes the on-disk image count authoritative over the annotation's
`num_frames`, so a truncated unzip shifts no labels — it warns loudly instead.

Expected log line:

```
DoTA val split: 1402 clips (805 ego / 597 other), 18478 sampled frames at stride 8, 6118 positive (0.3311)
WARNING: 3 clips lose their anomaly window at stride 8 ...
```

### 3.3 Extract CLIP features (once, at stride 1)

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad

python -m core.tools.extract_clip_features \
  --frames-dir /content/dota/frames --frames-subdir images \
  --ids-file "$KATVAD_DATA_ROOT/DoTA/labels_s8/test_ids.txt" \
  --dataset DoTA --stride 1 --batch-size 64 --device cuda \
  --output-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s1"
```

~143 k images; **20–35 min on a T4** (hours if `--frames-dir` points at Drive),
~292 MB of output.

**Resume is the same command.** Rerun the cell after a disconnect; it extracts
only the clips still missing and prints what it is doing:

```
INFO core.tools.feature_cache: Resume: 412/1402 already cached in .../clip/DoTA_s1, 990 to extract
INFO core.tools.extract_clip_features: Saved 0qfbmt4G8Rw_000306.npy (137, 512) -- [1/990] elapsed 0m04s, eta 68m12s
```

The ETA covers the pending clips only. Three things make that resume trustworthy
rather than merely convenient:

- **Atomic writes.** Features go to a `.part` sibling and are renamed into
  place, so a killed process leaves either a complete `.npy` or nothing.
- **Header verification.** A cached file counts as done only if it reads back as
  a non-empty 2-D array. The clip that was mid-write when Colab dropped is
  re-extracted and logged (`N cached files are unreadable (interrupted write)`)
  instead of being skipped forever and crashing eval hours later — the same
  failure shape as the empty `images/` folders in §3.1.1 (lesson C10).
- **`--force` is for the other kind of staleness.** Nothing on disk records the
  stride or the transform a cache was built with, so resume cannot detect a
  change to either. Changing one means re-extracting all of it (lesson C2).

> **Transform note (lesson C2).** This uses the repo's **center-crop**
> transform, the same one that built the MSAD cache the checkpoints were trained
> on. That is deliberate and non-negotiable here: the checkpoints must see the
> preprocessing they were trained under. It also means the `gate_a` arm pays the
> same ~0.5 pp handicap it paid on MSAD (the baseline extracted with
> `no_center_crop`). Do **not** "fix" the transform for this run.

### 3.4 Derive the stride-8 cache

```python
from pathlib import Path
import numpy as np, os

src = Path(os.environ['KATVAD_CACHE_ROOT']) / 'clip' / 'DoTA_s1'
dst = Path(os.environ['KATVAD_CACHE_ROOT']) / 'clip' / 'DoTA_s8'
dst.mkdir(parents=True, exist_ok=True)
for p in sorted(src.glob('*.npy')):
    target = dst / p.name
    if target.exists():          # resumable too: Drive writes are slow
        continue
    np.save(target, np.load(p)[::8])
print(len(list(dst.glob('*.npy'))), 'files')
```

Exact, not approximate: `range(0, N, 8)` is what a stride-8 extraction samples.

### 3.5 The three eval runs

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
DATA="$KATVAD_DATA_ROOT/DoTA/labels_s8"
CLIP="$KATVAD_CACHE_ROOT/clip/DoTA_s8"
OUT="$KATVAD_OUTPUT_ROOT/DoTA"

# Arm 1 — reference: LaGoVAD's released checkpoint (PreVAD-trained)
python -m core.evaluate \
  --baseline-ckpt "$KATVAD_CKPT_ROOT/best.ckpt" \
  --set kip.enabled=false --set data.dataset=DoTA \
  --data-dir "$DATA" --clip-dir "$CLIP" \
  --output-dir "$OUT/gate_a" --save-scores

# Arm 2 — our baseline (MSAD-trained, KIP off)
python -m core.evaluate \
  --ckpt "$KATVAD_OUTPUT_ROOT/MSAD/stage2_kip_off/checkpoint_last.pt" \
  --set kip.enabled=false --set data.dataset=DoTA \
  --data-dir "$DATA" --clip-dir "$CLIP" \
  --output-dir "$OUT/eval_kip_off" --save-scores

# Arm 3 — ours + KIP (MSAD-trained)
python -m core.evaluate \
  --ckpt "$KATVAD_OUTPUT_ROOT/MSAD/stage2_kip_on/checkpoint_last.pt" \
  --set data.dataset=DoTA \
  --data-dir "$DATA" --clip-dir "$CLIP" \
  --output-dir "$OUT/eval_kip_on" --save-scores
```

Each run is minutes (features are cached; only the heads and text encoder run).

**`--set kip.enabled=false` is mandatory on arms 1 and 2.** The model is built
from the CLI config, not from the checkpoint, and the state dict will not load
against the wrong architecture (lesson C5 — it fails loud, but read the error
rather than assuming a bad result).

### 3.6 Running on partial coverage (last resort)

When the gap is small, or the missing clips genuinely are not in the archive,
add `--allow-missing-frames` to §3.2. It drops every clip the extractor cannot
read — **no folder** and **empty folder** counted separately — and states the
coverage:

```
INFO    ...: 1 frame folders under .../frames are not in the split -- ignored
WARNING ...: 5/1402 annotated clips are unreadable under .../frames (subdir='images'):
             0 with no frame folder [], 5 with an empty one ['TNZv-NBcV5U_002389', ...]
             -- dropped from the evaluation set
WARNING ...: Coverage 1397/1402 clips (99.6%) -- report Delta(on-off) on this subset
             only; the absolute AUC is not the published protocol
INFO    ...: DoTA val split: 1397 clips (... ego / ... other), ... sampled frames ...
```

`test_ids.txt` and `frame_labels_test.json` then carry exactly the surviving
ids, so §3.3–§3.5 need no change.

What that costs you, precisely:

- **Δ(on − off) survives.** All three arms score the same clips, the bootstrap
  resamples that same set, D3/D4 are unaffected. This is the number the thesis
  claim rests on, so a partial run is still worth doing.
- **The absolute AUC stops being strictly comparable.** D1 and D2 compare
  against LaGoVAD's published 62.60, which is defined on the full 1,402-clip val
  split. On a subset it is a different quantity, so it never goes into a table
  as plain "DoTA AUC" — the coverage travels with it. How much that matters
  scales with the gap: **5 clips (99.6 %) is a rounding error** and D1/D2 stay
  usable with the caveat noted; 280 clips (80 %) is a different benchmark and
  D1/D2 are void.
- Ego/other subgroup balance shifts. Re-read the `ego_involve` counts from the
  §3.2 log rather than reusing §1's 805/597.

So: use it to get an early Δ reading while you sort out the archive; re-run at
full coverage before anything is recorded as a result.

---

## 4. Analysis cell

Paste after the three runs. Computes everything the decision rules need.

```python
import json, re, numpy as np, os
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

OUT  = Path(os.environ['KATVAD_OUTPUT_ROOT']) / 'DoTA'
META = json.load(open(Path(os.environ['KATVAD_DATA_ROOT']) / 'DoTA/labels_s8/meta.json'))
ARMS = ['gate_a', 'eval_kip_off', 'eval_kip_on']

runs = {a: {p.stem: dict(np.load(p)) for p in (OUT/a/'scores').glob('*.npz')} for a in ARMS}
vids = sorted(runs['eval_kip_on'])
ego  = [v for v in vids if META[v]['ego_involve']]
oth  = [v for v in vids if not META[v]['ego_involve']]

def pooled(arm, subset):
    y = np.concatenate([runs[arm][v]['gt']    for v in subset])
    s = np.concatenate([runs[arm][v]['score'] for v in subset])
    return roc_auc_score(y, s), average_precision_score(y, s)

def pooled_minmax(arm, subset):          # LaGoVAD offline_dota_eval.py protocol
    y, s = [], []
    for v in subset:
        sc = runs[arm][v]['score']; rng = sc.max() - sc.min()
        s.append((sc - sc.min()) / rng if rng > 0 else np.zeros_like(sc))
        y.append(runs[arm][v]['gt'])
    return roc_auc_score(np.concatenate(y), np.concatenate(s))

print(f"{'arm':14s} {'AUC':>7s} {'AP':>7s} {'AUC_ego':>8s} {'AUC_oth':>8s} {'AUC_mm':>7s}")
for a in ARMS:
    auc, ap = pooled(a, vids)
    print(f'{a:14s} {auc:7.4f} {ap:7.4f} {pooled(a,ego)[0]:8.4f} '
          f'{pooled(a,oth)[0]:8.4f} {pooled_minmax(a,vids):7.4f}')

# --- the decision number: paired bootstrap over clips ---
rng = np.random.default_rng(0)
gts = [runs['eval_kip_on'][v]['gt'] for v in vids]
son = [runs['eval_kip_on'][v]['score'] for v in vids]
sof = [runs['eval_kip_off'][v]['score'] for v in vids]
d_auc, d_ap = [], []
for _ in range(2000):
    pick = rng.choice(len(vids), len(vids), replace=True)
    y  = np.concatenate([gts[i] for i in pick])
    a  = np.concatenate([son[i] for i in pick])
    b  = np.concatenate([sof[i] for i in pick])
    d_auc.append(roc_auc_score(y,a) - roc_auc_score(y,b))
    d_ap.append(average_precision_score(y,a) - average_precision_score(y,b))
for name, d in (('dAUC', np.array(d_auc)), ('dAP', np.array(d_ap))):
    print(f'{name}: mean {d.mean():+.4f}  CI95 [{np.percentile(d,2.5):+.4f}, '
          f'{np.percentile(d,97.5):+.4f}]  P(>0)={np.mean(d>0):.3f}')

# --- per-clip win/loss and the ego-vs-other mechanism check ---
per = {}
for v in vids:
    y = runs['eval_kip_on'][v]['gt']
    if 0 < y.sum() < len(y):
        per[v] = (roc_auc_score(y, runs['eval_kip_on'][v]['score'])
                  - roc_auc_score(y, runs['eval_kip_off'][v]['score']))
d = np.array(list(per.values()))
print(f'per-clip dAUC: n={len(d)} mean {d.mean():+.4f} median {np.median(d):+.4f} '
      f'win {np.mean(d>0):.1%} loss {np.mean(d<0):.1%}')
de = np.array([per[v] for v in ego if v in per]); do = np.array([per[v] for v in oth if v in per])
print(f'  ego   n={len(de)} mean {de.mean():+.4f}')
print(f'  other n={len(do)} mean {do.mean():+.4f}')

# --- per anomaly-class delta ---
by = {}
for v in vids:
    by.setdefault(META[v]['anomaly_class'], []).append(v)
for cls, vs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    if len(vs) < 15: continue
    print(f'{cls:35s} n={len(vs):3d} dAUC {pooled("eval_kip_on",vs)[0]-pooled("eval_kip_off",vs)[0]:+.4f}')
```

Report **raw pooled AUC** as the headline: that is what
`LaGoVAD-PreVAD/src/full_length_eval.py:216` uses, and therefore what 62.60 is.
The `AUC_mm` column (per-clip min–max normalization, from their side script
`offline_dota_eval.py`) is a secondary number — it flatters every arm equally,
because every DoTA clip contains both classes. Never mix the two protocols in
one table.

---

## 5. Decision rules

Apply in order. **D1 is a stop-gate: if it fails, nothing below it means
anything.**

### D1 — Pipeline sanity (`gate_a`)

Read this against the **min-max** number (§2.1), not the raw one. Applying D1 to
raw scores is what produced a false "broken pipeline" reading on 2026-08-08 when
the pipeline was fine and the metric was not.

| `gate_a` AUC (min-max) | Verdict | Action |
|---|---|---|
| **≥ 0.58** | Data prep is sound (LaGoVAD 62.60 minus the center-crop handicap, ~0.5 pp on MSAD, plus transfer-free slack) | Proceed to D2 |
| 0.52 – 0.58 | Suspicious | Check §3.1 clip count, §3.2 warning lines, and that `--set kip.enabled=false` was passed. Then proceed, flagging it |
| **< 0.52** | Broken — near chance | **Stop.** Debug before interpreting anything: **first re-check the pooling rule** (`core.tools.rescore`), then frame ordering, label/feature length mismatch, or an incomplete unzip |

**Measured 2026-08-08:** 0.6142 min-max (0.5055 raw) — D1 **passes** on the
corrected protocol. It read as a hard fail on the raw one.

### D2 — Transfer floor (`eval_kip_off`)

Descriptive, no threshold. Expect it **below** `gate_a`: MSAD is fixed-camera,
120 abnormal training videos, and the domain shift to ego-centric video is
severe. If `eval_kip_off` beats `gate_a`, that is a genuinely interesting
finding about definition-conditioned transfer — record it, don't explain it away.

If `eval_kip_off` lands **below ~0.52**, the transfer has collapsed to noise and
the A/B has no headroom to measure. Report that honestly and move to
training on DoTA/PreVAD (§8) instead of iterating on KIP.

### D3 — The actual test: Δ(on − off)

| ΔAUC (bootstrap 95 % CI) | Verdict | Next action |
|---|---|---|
| **CI excludes 0 and mean ≥ +0.015** | **KIP works where motion matters.** This is the thesis result MSAD could not show | Confirm with 3 seeds (RESULTS_MSAD §6.1), then make DoTA the headline benchmark |
| CI excludes 0, mean +0.005 – 0.015 | Real but small | Worth reporting *with* the CI; still needs seeds before it goes in a table |
| **CI includes 0** | Same verdict as MSAD: **no measurable effect** | Stop iterating on KIP hyperparameters. Go to §8 |
| CI excludes 0, mean **negative** | KIP actively hurts under domain shift | Diagnose before anything else — likely a dead or mis-scaled gate (RESULTS_MSAD §6.4) |

### D4 — Mechanism check (the one that separates luck from a claim)

DoTA's `ego:` classes are the ones where the *camera's own* kinematics carry the
anomaly. If KIP helps for the reason the proposal claims, its gain must
concentrate there:

| Condition | Reading |
|---|---|
| `dAUC_ego` > `dAUC_other`, both positive | **Mechanism confirmed.** KIP's gain tracks kinematic content. Strongest possible version of the claim |
| `dAUC_ego` ≈ `dAUC_other` | KIP helps, but not demonstrably *because* of motion. Weaker claim — say so |
| `dAUC_ego` < `dAUC_other` | The gain is not coming from ego-motion. Do not claim a kinematic mechanism |

Report D4 whatever D3 says: a null D3 with a positive ego-only delta is still a
signal about where to look next.

### D5 — Cross-check against MSAD's one live thread

RESULTS_MSAD §3.5 found KIP's clearest MSAD effect in the **multi-class head**
(+3.6 pp on anomalous frames), concentrated on motion-textured classes. DoTA's
`sim` array is only 2-wide (Normal / CarAccident) so that metric does not
transfer — but if D3 is positive here **and** the multi-class effect replicates
across seeds on MSAD, the two findings share one mechanism and the thesis has a
coherent story. Note the coincidence; don't overclaim it.

---

## 6. Recording the result

Append a section to `core/docs/RESULTS_MSAD.md` (or start `RESULTS_DOTA.md` if
it grows past a page) with, per arm: checkpoint path, stride, **clip coverage
(N evaluated / N in split)**, AUC, AP, AUC_ego, AUC_other, ΔAUC + CI, per-clip
win/loss rate. Coverage is not optional bookkeeping — a number measured on a
partial unzip (§3.6) is a different quantity. Then update
`.project/memory-bank/progress.md` — Phase 7's first line item is exactly this.

**Do not** quote any number from this run against LaGoVAD's 62.60 except
`gate_a`.

---

## 7. Optional follow-ups (cheap, same features)

1. **Temporal-scale ablation.** Rebuild labels at stride 1/2/4
   (`core.data.dota --stride N --out-dir .../labels_sN`), slice the stride-1
   cache accordingly, rerun arms 2–3. If Δ is stride-dependent, KIP's shift is
   operating at a specific time constant — a real finding, and a hint for
   DoTA-native training. Off-protocol: report separately from the stride-8
   headline.
2. **Night / day split.** The per-clip annotations carry a `night` flag (188 of
   1,402 val clips). Not in `metadata_val.json` — needs
   `Detection-of-Traffic-Anomaly/dataset/DoTA_annotations.zip`.
3. **The 3 rounded-away clips.** Confirm they are excluded from any per-clip
   AUC statistic (they have no positive frames, so `roc_auc_score` on them is
   undefined — the analysis cell already skips them).

---

## 8. If D3 comes back null

Two live options, in order:

1. **Fix the measurement first, not the model.** RESULTS_MSAD §6.2 (validation
   split + `checkpoint_best`) and §6.3 (`L_KIP_align` runs at 3.66 vs chance
   4.27) are both unresolved. A null Δ measured at `checkpoint_last` with one
   inert loss term is weak evidence about KIP itself.
2. **Train on ego-centric data.** Zero-shot MSAD→DoTA asks the model to
   generalize across camera geometry *and* to exploit motion at the same time.
   Training on PreVAD (or DoTA's train split) separates those two questions.
   That is Phase 7 proper and it is where the proposal's headline claim actually
   lives.

---

## 9. Code added for this protocol

| File | What |
|---|---|
| `core/data/dota.py` | Preprocessor: `metadata_val.json` + `val_split.txt` → the four standard dataset files + `test_ids.txt`. Reproduces LaGoVAD's normalized-span label arithmetic; carries `ego_involve` into `meta.json`; `--allow-missing-frames` for partial-coverage runs (§3.6) |
| `core/data/video_io.py` | Frame-folder readers: `list_frame_folders`, `list_frame_images`, `read_images`, `read_sampled_frames_from_dir`, `video_id_from_path` (these were referenced by `core/data/tad.py` but missing — that module did not import before this change) |
| `core/tools/extract_clip_features.py` | `--frames-dir` / `--frames-subdir` / `--ids-file`, and `encode_frame_dir`, which streams batches so a 284-frame 720p clip at stride 1 does not materialize as a ~3 GB float32 array |
| `core/tools/feature_cache.py` | Resume for both extractors (CLIP and RAFT): atomic `.part`-then-rename writes, header verification of cached outputs, `Resume: done/total` + per-clip ETA logging |
| `core/data/dataset_files.py` | Shared `TEST_IDS_FILENAME` + `write_test_ids` (was duplicated in `tad.py`) |
| `core/tests/test_dota.py` | 26 tests: frame-folder IO, label arithmetic vs the baseline's rule, disk-count reconciliation, unreadable-clip handling (absent / empty, fatal / opt-in drop), out-of-split folders, streaming-batch invariance, resume after an interrupted write, and an end-to-end **MSAD-checkpoint-scores-DoTA** smoke test |

Gates at time of writing: **258 tests pass**, ruff / mypy / pyright / bandit clean.
