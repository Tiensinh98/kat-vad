# TAD — dataset setup on Colab (unzip → labels → CLIP → RAFT → KNN)

**Written 2026-09-02, against branch `v3`.** Dataset-level and version-agnostic:
everything here is about getting `data/TAD/` and the caches onto disk correctly.
The *experiment* on top of it — arms, seeds, ablations — lives in
`core/docs/v3/setup/TAD_V3_SETUP.md`.

Companion docs: `DATA_LAYOUT.md` (the on-disk contract this obeys),
`DOTA_EVAL.md` §3.1 (the unzip/coverage playbook this borrows), `COLAB.md`
(session mechanics, checkpoint recovery).

---

## 0. Read this before you plan anything on TAD

### 0.1 TAD has been a **zero-shot** benchmark in this project until now

`core/constants.py` carries `TAD_ZERO_SHOT_AUC = 89.56` with
`GATE_A_TOLERANCE = 0.5`. That is LaGoVAD's published number for a model
**trained elsewhere** (PreVAD) and scored on TAD's 100-video test split, never
having seen a TAD frame.

**The moment you train on TAD's train split, your TAD number stops being
comparable to 89.56.** In-domain beats zero-shot; a bigger number is not a
better method. Anyone who puts the two in one column has produced a table that
is wrong in the direction of their own claim.

Keep them apart, always, by name:

| Number | Trained on | Scored on | Comparable to 89.56? |
|---|---|---|---|
| **Gate T0** (§6) | LaGoVAD `best.ckpt`, PreVAD | TAD test | **yes** — this is the gate |
| **In-domain TAD** (§V3) | TAD train | TAD test | **no** |
| **Zero-shot DoTA** | TAD train | DoTA val | no — different benchmark, own baseline |

### 0.2 What training on TAD is actually *for*

As of 2026-09-01 the v3 campaign settled the attribution on MSAD: a fixed 50 %
channel shift (`gate_type=constant`, arm **A2**) is statistically
indistinguishable from full KIP on DoTA, and the `rank` gate (**A1**) is the
*worst* KIP arm. See `core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md`. The motion
story is not supported by any ablation we have run.

So the honest reason to add TAD as a **second training corpus** is
**replication of that attribution**, not a new headline. The question TAD
answers is:

> Does `A2 − A0 ≈ A1 − A0`, with `A1 − A2 ≈ 0`, hold on a training corpus that
> is not MSAD?

If it does, the smoother finding generalizes and the write-up gets stronger.
If it does *not*, we have learned something real about corpus dependence.
Either outcome is publishable. **Do not run TAD hoping for a bigger number**
(lesson 14).

### 0.3 The two code prerequisites — **shipped 2026-09-02**

TAD training needed two changes that no config flag could provide. Both landed
on branch `v3`; every command in this document runs against the tree as it is.

| # | What was missing | What landed |
|---|---|---|
| **P1** | `core/data/tad.py` was **eval-only** — it wrote `labels_train.json` as `{}` by construction, so `DVSFeatureDataset` raised `Training set needs both classes: 0 normal, 0 abnormal` (`core/data/dataset.py:71`) and no TAD run was possible | `--with-train-split` builds the weakly-supervised train split from the frame folders the annotation does not name. **Default is unchanged**: without the flag the module is still eval-only, so every pre-2026-09-02 TAD artifact is reproducible bit-for-bit |
| **P2** | `core/flow/raft_extract.py` accepted `--videos-dir` only and read through PyAV, so TAD — which ships **frames, not videos** — could not build `e_O` at all, killing every KIP-on arm | `--frames-dir` / `--frames-subdir` / `--ids-file`, mirroring `core.tools.extract_clip_features`. `--videos-dir` is untouched, and the two paths are asserted **bit-identical** on the same pixels |

```
# core.data.tad
--with-train-split            build labels_train.json from the frame folders the
                              annotation does not name (default: off, eval-only)
--abnormal-dirname abnormal   folder whose train clips get video-level label 1
--normal-dirname   normal     folder whose train clips get video-level label 0

# core.flow.raft_extract  (mirrors core.tools.extract_clip_features)
--frames-dir DIR              per-video extracted-frame folders (mutually
                              exclusive with --videos-dir, one required)
--frames-subdir NAME          image subfolder inside each frame folder
--ids-file PATH               restrict to these ids (flow is train-only)
```

**The one subtlety worth knowing.** `read_images` returns a *non-contiguous*
permuted view, and torch's batched convolutions take a different kernel path on
one — so flow-from-frames disagreed with flow-from-video by ~3e-5 on identical
pixels. `read_sampled_frames_from_dir` now returns a C-contiguous buffer and the
two are bit-identical, asserted in `core/tests/test_extractors.py`. That matters
because `cache/flow/v1/{DATASET}/` has to mean the same thing whether the dataset
shipped videos (MSAD) or frames (TAD); a cache that quietly depends on its source
is one you cannot compare across datasets.

---

## 1. What ships, and where it lands

Your archive, `$KATVAD_DATA_ROOT/TAD/archived.zip`:

```
frames/
  abnormal/
    01_Accident_001.mp4/     <- a DIRECTORY whose name ends .mp4
      0.jpg  1.jpg  2.jpg ...
    ...
  normal/
    Normal_001.mp4/
      0.jpg  1.jpg ...
    ...
```

Two properties of that layout the code already handles, and one that will bite:

- **`video_id_from_path` strips a video extension from a folder name**
  (`core/data/video_io.py`), so `01_Accident_001.mp4/` → id `01_Accident_001`.
  That matches the annotation's `video_path` stem exactly. Nothing to rename.
- **`list_frame_folders(root, subdir=None)`** accepts *any* directory at any
  depth that directly holds images, so the `abnormal/` / `normal/` split layer
  is transparent. Pass `--frames-dir .../frames` with **no** `--frames-subdir`
  (that flag is DoTA's `frames/{id}/images/` layout, not this one).
- ⚠️ **`0.jpg 1.jpg 10.jpg` is not zero-padded**, and
  `list_frame_images` sorts lexicographically. `10.jpg` sorts before `2.jpg`.
  **Lexicographic order is not temporal order for this archive.** §4.3 is the
  gate for that, and it is not optional — a mis-ordered clip produces features
  and flow that look perfectly healthy and are temporally scrambled.

Target layout (`DATA_LAYOUT.md`):

```
$KATVAD_DATA_ROOT/TAD/frames/{abnormal,normal}/{id}.mp4/*.jpg   (or /content, §3)
$KATVAD_DATA_ROOT/TAD/annotations/tad_test_anno.json            official test protocol
$KATVAD_DATA_ROOT/TAD/labels_train.json                         {id: 0|1}
$KATVAD_DATA_ROOT/TAD/frame_labels_test.json                    {id: [0,1,...]} per SAMPLED frame
$KATVAD_DATA_ROOT/TAD/defs.json                                 ["Normal", "Car Accident"]
$KATVAD_DATA_ROOT/TAD/meta.json                                 diagnostics only
$KATVAD_DATA_ROOT/TAD/test_ids.txt                              100 scored ids
$KATVAD_CACHE_ROOT/clip/TAD_ncc/{id}.npy                        (L,512) no_center_crop
$KATVAD_CACHE_ROOT/flow/v1/TAD/{id}.npy                         (L,256) e_O, train ids only
$KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz                    DVS filler cache
```

`_ncc` on the CLIP and KNN dirs, none on flow: the flow pipeline is full-frame
(`preprocess_for_raft`), so it is transform-independent. That asymmetry is
deliberate and documented in lesson **C13** — keep it.

### 1.1 The annotation file does not travel with the repo

`LaGoVAD-PreVAD/` is in `.gitignore`, so
`LaGoVAD-PreVAD/data/other_datasets/tad_test_anno.json` (100 entries, ~20 KB:
60 `Car Accident` + 40 `Normal`) is **not** in your Colab checkout. Upload it by
hand, once:

```
local:  LaGoVAD-PreVAD/data/other_datasets/tad_test_anno.json
drive:  MyDrive/Thesis/data/TAD/annotations/tad_test_anno.json
```

Verify after upload:

```python
import json, os
p = f"{os.environ['KATVAD_DATA_ROOT']}/TAD/annotations/tad_test_anno.json"
d = json.load(open(p))
from collections import Counter
print(len(d), Counter(e['class_name'] for e in d))   # expect: 100 Counter({'Car Accident': 60, 'Normal': 40})
```

Anything but `100 / 60 / 40` and you have the wrong file. Stop.

---

## 2. Colab session setup

### 2.1 Code + install (once per VM)

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

**Restart the runtime** (`Runtime → Restart session`), then re-mount and re-run
§2.2. A batch `pip install` is atomic and a failed `!pip` does not stop the
notebook (lesson 21) — check the cell actually succeeded before moving on.

```bash
!python -c "from core.kip import gate_shift; print(gate_shift.GATE_TYPES)"
# expect: ('rank', 'mlp_frozen', 'mlp_ste', 'constant')
```

### 2.2 Environment variables (every session)

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

---

## 3. Unzip

**Unzip to `/content` (VM-local NVMe), not to Drive.** The frames are an
*intermediate*: the artifacts worth persisting are `clip/TAD_ncc/{id}.npy` and
`flow/v1/TAD/{id}.npy`, and once those exist the jpgs are dead weight. Drive's
FUSE mount charges ~50–100 ms per file **open**, and you pay it twice — once
writing ~150 k small jpgs, once reading them all back during extraction.

| | `/content` | Drive |
|---|---|---|
| unzip | minutes | slow; truncated extractions observed on DoTA |
| extraction reads every jpg | fast | hours, rate-limit-prone |
| survives a disconnect | no | yes |
| re-do cost after a disconnect | one unzip | none |

```bash
%%bash
set -e
mkdir -p /content/tad
df -h /content | tail -1                 # check free space before, not after
unzip -q "$KATVAD_DATA_ROOT/TAD/archived.zip" -d /content/tad
ls /content/tad                          # expect: frames
ls /content/tad/frames                   # expect: abnormal  normal
```

If the archive nests everything one level deeper (`archived/frames/...`), move
the **directory**, never a glob of its contents — a dataset-sized glob overflows
`ARG_MAX` and dies *after* a successful extraction, leaving a half-done setup
that reads as a broken download (lesson **20**):

```bash
%%bash
# only if `ls /content/tad` printed something other than `frames`
mv /content/tad/*/frames /content/tad/frames_tmp && \
  rm -rf /content/tad/archived && mv /content/tad/frames_tmp /content/tad/frames
```

---

## 4. Ingest gates — run all three before extracting anything

An existing directory is not evidence of data (lesson **10**). Count *files*.

### 4.1 Coverage

```python
import os
ROOT = '/content/tad/frames'
counts = {}
for split in ('abnormal', 'normal'):
    d = f'{ROOT}/{split}'
    folders = sorted(os.listdir(d)) if os.path.isdir(d) else []
    empty = [f for f in folders if not os.listdir(f'{d}/{f}')]
    counts[split] = (len(folders), len(empty))
    print(f'{split:9s} folders={len(folders):4d}  empty={len(empty):3d}  {empty[:3]}')
total = sum(n for n, _ in counts.values())
print('total folders', total)
assert not any(e for _, e in counts.values()), 'empty folders = truncated unzip; re-run §3'
```

An **empty** folder means the unzip died after `mkdir`. `unzip -n` is resumable
— re-run §3's command, it will not overwrite what landed.

### 4.2 Every annotated test video has frames

```python
import json, os
anno = json.load(open(f"{os.environ['KATVAD_DATA_ROOT']}/TAD/annotations/tad_test_anno.json"))
ids  = {os.path.basename(e['video_path']).replace('.mp4', '') for e in anno}
have = set()
for split in ('abnormal', 'normal'):
    have |= {f.replace('.mp4', '') for f in os.listdir(f'/content/tad/frames/{split}')}
missing = sorted(ids - have)
print(f'annotated={len(ids)}  on disk={len(have)}  missing={len(missing)}  {missing[:5]}')
assert not missing, 'core.data.tad will raise on these; fix the unzip first'
```

`core/data/tad.py:resolve_records` raises on exactly this, by design. Better to
see it here than three cells later.

### 4.3 ⚠️ Frame ordering — the one that fails silently

`list_frame_images` sorts by filename. `0.jpg 1.jpg 10.jpg 2.jpg` is the
lexicographic order of this archive's names, and it is **not** temporal order.

```python
import os, re
d = '/content/tad/frames/abnormal'
sample = sorted(os.listdir(d))[0]
names = sorted(os.listdir(f'{d}/{sample}'))
print(sample, len(names), names[:5], '...', names[-3:])
lex   = [n for n in names]
num   = sorted(names, key=lambda n: int(re.sub(r'\D', '', n) or 0))
print('lexicographic == numeric order:', lex == num)
```

**If that prints `False`, stop.** Every downstream artifact — CLIP features,
RAFT flow, and therefore every metric — would be built on scrambled time. Two
honest fixes, in order of preference:

1. **Zero-pad the filenames once, in place, before extracting anything.** The
   frame extractors' contract is explicitly "zero-padded names, so lexicographic
   order is temporal order; non-padded names would sort wrong — callers own that
   contract" (`core/data/video_io.py:list_frame_images`). You are the caller.

   ```bash
   %%bash
   # rename 7.jpg -> 000007.jpg everywhere. The renamer goes in a FILE, not a
   # heredoc: `xargs` owns stdin, so a heredoc piped into it never reaches
   # python. `find -print0 | xargs -0` never builds a dataset-sized argv
   # (lesson 20). Idempotent — already-padded names are left alone.
   cat > /content/pad_frame_names.py <<'PY'
   import os, sys
   for path in sys.argv[1:]:
       folder, name = os.path.split(path)
       stem, ext = os.path.splitext(name)
       if stem.isdigit() and len(stem) < 6:
           target = os.path.join(folder, f'{int(stem):06d}{ext}')
           if not os.path.exists(target):
               os.rename(path, target)
   PY
   find /content/tad/frames -name '*.jpg' -print0 \
     | xargs -0 -n 200 python3 /content/pad_frame_names.py
   ```

   Then **re-run this cell** and confirm it prints `True`.

2. Change `list_frame_images` to sort numerically. **Do not do this
   casually** — it is a shared code path (DoTA, TAD, every extractor), it needs
   `trace_call_path` first, and it silently changes what every existing cache
   means (lesson **C2**). Renaming the files is the local, reversible fix.

Re-run §4.1 after any rename: the file count must be unchanged.

---

## 5. Build the label files

### 5.1 Test split only — works today

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.data.tad \
  --annotation "$KATVAD_DATA_ROOT/TAD/annotations/tad_test_anno.json" \
  --frames-dir /content/tad/frames \
  --out-dir    "$KATVAD_DATA_ROOT/TAD" \
  --stride 8 --dry-run
```

`--dry-run` parses, resolves against disk and logs the counts without writing.
Read the log line before dropping the flag:

```
TAD test split: 100 videos (60 abnormal), N sampled frames, M positive
Ignoring 410 frame folders absent from the annotation (TAD train split)
```

Then re-run without `--dry-run`. Writes `labels_train.json` (**empty** — that is what eval-only
means; §5.2 fills it), `frame_labels_test.json`, `defs.json` (`["Normal", "Car Accident"]`),
`meta.json`, `test_ids.txt`.

> **If it raises `N abnormal videos lose their anomaly window at stride 8`**:
> a span is shorter than one sampled frame and would silently turn an abnormal
> video into a negative-only one. This is a **stride** problem, not a label
> problem. Do not patch around it. Lower `--stride` — and understand that doing
> so invalidates every TAD cache and every number measured on one (lesson
> **C2**), so decide it once, here, before extracting features.
>
> **How close is that, really?** Measured on the shipped annotation
> (2026-09-02): the narrowest normalized span is **0.0260**, on
> `03_IllegalOccupation_002`. A span rounds to at least one sampled frame once
> the clip has **≥ 20 sampled frames — i.e. ≥ 160 raw frames** at stride 8. TAD
> clips are generally longer than that, so this should pass; it is not
> guaranteed to, and the preprocessor is what tells you.

The annotation also carries **10 multi-span videos, up to 3 windows each**.
`sampled_frame_labels` fills every span independently rather than taking the
hull (lesson **C18** — taking the hull is how PreVAD turned 40 positive rows
into 66 with no error).

### 5.2 Train + test

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.data.tad \
  --annotation "$KATVAD_DATA_ROOT/TAD/annotations/tad_test_anno.json" \
  --frames-dir /content/tad/frames \
  --out-dir    "$KATVAD_DATA_ROOT/TAD" \
  --stride 8 --with-train-split
```

The rule, and the only rule: **a frame folder the annotation does not name is a
train video; its video-level label is 1 if it sits under `abnormal/`, 0 under
`normal/`.** TAD has exactly one anomaly class, so abnormal train videos get
`class_name = "Car Accident"` — which is the annotation's own name for them and
the key in `_TAD_CLS_DEFS` (`core/data/definitions.py`), so definition coverage
is total (lesson **19** cannot fire here). A folder under neither directory
raises rather than defaulting to normal: an unlabelled anomaly in the normal
half poisons the MIL objective and nothing downstream would notice.

> ⚠️ **The directory is the label — the filename is not.** MSAD's preprocessor
> has an `--infer-abnormal-from-name` flag, and TAD's ids look like they invite
> the same treatment. They do not. Measured on the shipped annotation: the 60
> abnormal test videos carry **seven** different filename prefixes —
> `01_Accident` (23), `05_else` (14), `06_PedestrianOnRoad` (10), `07_RoadSpills`
> (4), `02_IllegalTurn` (3), `03_IllegalOccupation` (3), `04_Retrograde` (3) —
> and LaGoVAD labels **all** of them `Car Accident` (its `TAD.py` taxonomy has
> exactly two classes). A name-prefix rule keyed on `Accident` would mislabel
> **37 of 60** abnormal videos as normal, silently. Follow the baseline's
> collapse, and take the label from the directory.

Weak supervision is preserved by construction: TAD's train split has **no public
frame-level annotation**, so there is nothing to leak even by accident.

Verify, and record the counts — §10 needs them:

```python
import json, os
D = f"{os.environ['KATVAD_DATA_ROOT']}/TAD"
tr   = json.load(open(f'{D}/labels_train.json'))
te   = json.load(open(f'{D}/frame_labels_test.json'))
meta = json.load(open(f'{D}/meta.json'))
A = sum(tr.values()); N = len(tr) - A
print(f'train {len(tr)}  ({A} abnormal / {N} normal)')
print(f'test  {len(te)} ({sum(1 for v in te.values() if any(v))} with a positive frame)')
assert A and N, 'DVSFeatureDataset needs both classes'
assert not (set(tr) & set(te)), 'LEAK: an id is in both splits'
frac_normal = sum(1 for v in te.values() if not any(v)) / len(te)
print(f'test normal fraction {frac_normal:.3f}  -> score-norm auto resolves to '
      f'{"minmax" if frac_normal < 0.05 else "RAW"}')
```

The last line matters more than it looks. TAD test is ~40 % normal, so
`--score-norm auto` resolves to **raw** pooling — the MSAD protocol, *not*
DoTA's per-clip min-max. Pooling is decided by the label distribution, never by
the dataset name (lesson **12**). Never put a raw-pooled TAD number and a
min-max-pooled DoTA number in the same column.

---

## 6. Gate T0 — reproduce the zero-shot number before you train anything

**Run this before §7 and §8.** It costs one CLIP extraction pass and no
training, and it validates the entire ingest — unzip, frame order, label
arithmetic, stride, pooling — against a checkpoint whose TAD score is published.
If T0 fails, nothing downstream of it means anything.

Needs §7's `clip/TAD_ncc` cache first (it is the same cache the training runs
use, so this is not extra work).

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.evaluate \
  --baseline-ckpt "$KATVAD_CKPT_ROOT/best.ckpt" \
  --set kip.enabled=false --set data.dataset=TAD \
  --data-dir "$KATVAD_DATA_ROOT/TAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
  --score-norm auto \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD_gate_t0/gate_t0" --save-scores
```

Reference: `TAD_ZERO_SHOT_AUC = 89.56`, `GATE_A_TOLERANCE = 0.5`
(`core/constants.py`).

**How to read the result, honestly** (lessons **8** and **8b**):

| Outcome | Reading |
|---|---|
| micro AUC within ±0.5 of 89.56 | ingest verified. Proceed. |
| 85–89 | plausible-but-unverified. The most likely cause is **sampling rate**: LaGoVAD scored precomputed `tad_features/*.npy` whose extraction interval is not recorded anywhere in its repo, and we sample stride 8. Try `--stride 1` labels + features on the 100 test clips only (cheap) and compare before concluding anything about the model. |
| ≪ 85, or ≈ 50 | a pipeline defect, not a protocol difference. Prime suspect: §4.3 frame ordering. Check that first. |

**Gate against the released checkpoint's own number, not the printed one**
(lesson **8b**). Whatever `best.ckpt` scores here *is* the reference every later
TAD arm is compared to. Record it in the run notes with the date; do not carry
89.56 forward as if we had reproduced it.

```bash
%%bash
python -m core.tools.rescore --run-dir "$KATVAD_OUTPUT_ROOT/TAD_gate_t0/gate_t0"
```

`rescore` recomputes the metric under every pooling rule from the saved `.npz`.
It is seconds, and it is the check that catches a silently wrong `auto`.

---

## 7. CLIP features — `no_center_crop`, stride 8

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.tools.extract_clip_features \
  --frames-dir /content/tad/frames \
  --dataset TAD --stride 8 --batch-size 64 --device cuda \
  --no-center-crop \
  --output-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc"
```

- **No `--frames-subdir`.** TAD's jpgs sit directly in the clip folder; that flag
  is for DoTA's `frames/{id}/images/`.
- Confirm the log reads **`Transform: anisotropic resize`**. `no_center_crop` is
  LaGoVAD's transform and the one every current artifact in this project uses.
  A center-cropped cache is a *different, incompatible* cache (lesson **C2**) and
  would also break the appearance/flow field-of-view match (lesson **C13**).
- Resumable per video with atomic `.part` writes and header verification
  (lesson **11**); re-running does only what is missing and logs
  `Resume: done/total`.
- **`--force` is for a stride or transform change only.** Nothing on disk records
  either, so resume cannot detect it for you.
- To extract the 100 test clips first (for Gate T0, before committing GPU hours
  to all 510): add `--ids-file "$KATVAD_DATA_ROOT/TAD/test_ids.txt"`.

Coverage check:

```python
import os, json, numpy as np
C = f"{os.environ['KATVAD_CACHE_ROOT']}/clip/TAD_ncc"
D = f"{os.environ['KATVAD_DATA_ROOT']}/TAD"
have = {p[:-4] for p in os.listdir(C) if p.endswith('.npy')}
te = json.load(open(f'{D}/frame_labels_test.json'))
print(f'{len(have)} feature files')
bad = [(v, len(te[v]), len(np.load(f'{C}/{v}.npy'))) for v in te
       if v in have and len(np.load(f'{C}/{v}.npy')) != len(te[v])]
print('length mismatches vs frame labels:', len(bad), bad[:3])
assert not bad, 'feature rows must line up with label rows, one per sampled frame'
```

A mismatch here means the label arithmetic and the extractor disagree about how
many frames a clip has — i.e. the frame folder changed between the two runs, or
§4.3 bit you.

---

## 8. RAFT flow targets — train ids only

`e_O` is a **train-time** target for `L_KIP_rec` / `L_KIP_align`. It is never on
the inference path, and the test split never needs it. Extract for the ~410
train ids only.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
# train_ids.txt is written by §5.2, alongside test_ids.txt
python -m core.flow.raft_extract \
  --frames-dir /content/tad/frames \
  --ids-file  "$KATVAD_DATA_ROOT/TAD/train_ids.txt" \
  --dataset TAD \
  --cache-root "$KATVAD_CACHE_ROOT/flow/v1" \
  --stride 8 --batch-size 8 --device cuda
```

Writes `flow/v1/TAD/{id}.npy` `(L, 256)`, `{id}.stats.npy` `(L, 23)` and, if it
does not exist yet, `flow/v1/flow_projection.npz`.

- **The cache is source-independent.** `--frames-dir` and `--videos-dir` produce
  bit-identical `e_O` for the same pixels (asserted in
  `core/tests/test_extractors.py::TestRaftFrameExtraction`), so TAD's flow cache
  is directly comparable to MSAD's. That is a property worth keeping: check it
  again if either reader ever changes.
- **Never overwrite `v1/` in place.** The projection ships *with* the cache it
  produced; a loader fails loudly if it is missing for the version it reads.
  If you already have `flow_projection.npz` from the MSAD campaign, TAD reuses it
  — that is correct and required for cross-corpus comparability.
- **Same stride as the CLIP cache.** The dataset asserts
  `len(flow) == len(features)` per video and raises otherwise
  (`core/data/dataset.py:_load_flow`). That assert is the only thing standing
  between you and a silent misalignment.
- The flow target has **23 effective dimensions** — frame-global scalars lifted
  to 256-d by a fixed seeded projection. Never claim KIP localizes anything
  spatially.

---

## 9. DVS KNN filler cache

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.data.knn_cache \
  --data-dir "$KATVAD_DATA_ROOT/TAD" --dataset TAD \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
  --output   "$KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz"
```

Derived from the CLIP cache, so it belongs to `TAD_ncc` and to no other
transform. **Leave `--motion-key` off**: TAD is fixed-camera surveillance, not
ego-centric. For the same reason, keep `data.is_egocentric=false` in every TAD
training config (that flag selects `theta_ego` / `delta_m_ego` in DVS, tuned for
DoTA-style ego footage).

---

## 10. Sizing a TAD run — compute it, don't copy MSAD's

`DVSFeatureDataset.__len__` is `2 × num_abnormal_train`, **not** the number of
train videos. MSAD-full's `num_epochs=125 ≈ 500 steps` does not carry over.

```python
import json, math, os
tr = json.load(open(f"{os.environ['KATVAD_DATA_ROOT']}/TAD/labels_train.json"))
A  = sum(tr.values())
BATCH, TARGET_STEPS = 64, 500
steps_per_epoch = math.ceil(2 * A / BATCH)
print(f'{A} abnormal train videos -> dataset len {2*A} -> {steps_per_epoch} steps/epoch')
print(f'num_epochs for ~{TARGET_STEPS} steps: {math.ceil(TARGET_STEPS / steps_per_epoch)}')
```

Use that `num_epochs` for **both** stage 1 and stage 2, and for **every arm** —
an arm trained for a different number of steps is not a control. Warm-up stays
at 20 steps, peak LR 5e-5, unchanged from the baseline.

**Step checkpoints no longer exist** (removed 2026-09-13): a stage-2 run
writes only `checkpoint_last.pt`, so the ~1.4 GB per run this section used to
warn about is gone. Lesson **16** still stands for any future trajectory probe —
step-uniform checkpoints undersample the loss range — but such a probe now needs
the knob reintroduced deliberately rather than switched on.

---

## 11. Pitfalls, mapped to lessons

| Don't | Why | Lesson |
|---|---|---|
| Compare an in-domain TAD number to 89.56 | 89.56 is zero-shot. In-domain beats zero-shot; the comparison flatters the method for free | §0.1, **8b** |
| Trust `0.jpg 1.jpg 10.jpg` ordering | Lexicographic ≠ temporal; the whole pipeline is then built on scrambled time, with no error anywhere | §4.3, **10** |
| Extract before running §4's three gates | An existing directory is not evidence of data | **10** |
| Unzip to Drive | ~150 k FUSE opens, paid twice; truncated extractions observed on DoTA | §3 |
| `mv "$DIR/prefix/"* "$DIR/"` | Dataset-sized glob overflows `ARG_MAX` and dies *after* extracting | **20** |
| Center-crop the CLIP cache | Incompatible with every existing artifact, and breaks the appearance/flow FOV match | **C2**, **C13** |
| Change `--stride` after extracting | Invalidates every cache *and every metric measured on one* | **C2** |
| Min-max pool TAD scores | TAD test is ~40 % normal → raw pooling. Pooling follows the label distribution, not the dataset name | **12** |
| `--set kip.disable_pmg=true` to dodge a missing flow cache | It only zeroes the flow *targets*; `L_KIP_rec` then trains the PMG head to regress zeros | **14** |
| Pad or patch a label whose span vanished at stride 8 | It is a stride problem. Patching it hides a real sampling defect | §5.1 |
| Report a TAD Δ from one seed | 100 test clips is a small denominator; per-seed spread is not optional | — |
| Assume the score `.npz` set is complete | `evaluate --save-scores` is **not** atomic; a 0-byte `.npz` has been observed. Assert the file count against `results.json:num_videos` before any Δ | **11b** |

---

## 12. Out of scope here

- **The experiment itself** — arms, seeds, gate types, ablations, decision rules:
  `core/docs/v3/setup/TAD_V3_SETUP.md`.
- **P1 / P2 implementation.** Shipped (§0.3). Covered by `core/tests/test_tad.py`
  and `core/tests/test_extractors.py::TestRaftFrameExtraction`.
- **Checkpoint selection / validation split.** TAD ships no val split. Every arm
  reads `checkpoint_last`, deep in the overfit regime — a confound shared by all
  arms that cancels in a Δ but caps the absolute numbers.
- **Alert-CLIP.** No public checkpoint exists. Every number here is stock
  CLIP ViT-B/16.
