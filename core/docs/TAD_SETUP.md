# TAD — setup, training and evaluation on Colab (branch `main` = KAT-VAD v1)

**Written 2026-09-02 against branch `v3`. Rewritten 2026-09-14 for branch
`main`.** This document is now the *complete* TAD runbook for this branch: data
ingest (§1–§10), sizing (§11), and the experiment layer — arms, training,
evaluation, decision rules (§12–§16).

> ## ⚠️ Branch check — run this first
>
> ```bash
> git branch --show-current      # must print: main
> ```
>
> `core/docs/v3/setup/TAD_V3_SETUP.md` exists in this tree but **its arm matrix
> cannot run here.** Every arm there is keyed on `kip.gate_type`,
> `kip.const_shift_ratio` or `kip.disable_pmg`, and `core/config.py` on `main`
> raises `KeyError: Unknown config key` on all three (verified 2026-09-14). It
> also passes `train.checkpoint_every_steps`, which was removed from
> `TrainConfig` on 2026-09-13 and now raises as well. Read it for the
> *reasoning*; take the *commands* from here.
>
> The experiment layer below was written for `main` and **every `--set` in it
> was run through `load_config` on this branch.**

Companion docs: `DATA_LAYOUT.md` (the on-disk contract this obeys), `EDA.md`
(the pre-flight profiler §6 uses), `DOTA_EVAL.md` §3.1 (the unzip/coverage
playbook this borrows), `COLAB.md` (session mechanics, checkpoint recovery),
`TRAINING.md` (objective wiring and deviations).

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
| **Gate T0** (§8) | LaGoVAD `best.ckpt`, PreVAD | TAD test | **yes** — this is the gate |
| **In-domain TAD** (§14) | TAD train | TAD test | **no** |
| **Zero-shot DoTA** (§14) | TAD train | DoTA val | no — different benchmark, own baseline |

### 0.2 What training on TAD is actually *for*

Not a headline. A **replication**.

The v3 campaign settled the attribution on MSAD (2026-09-01, recorded in
`.project/memory-bank/progress.md` because `RESULTS_V3_GATE_ATTRIBUTION.md` does
not exist on this branch): a fixed 50 % channel shift with no flow, no PMG head
and no KIP losses is **statistically indistinguishable from full v1 KIP** on
zero-shot DoTA — Δ = +0.0109, t95 [−0.0588, +0.0805] — while the same shift
against a KIP-off trunk is worth **+0.1025 ± 0.0350**. KIP's measured
contribution is **temporal smoothing**. Then DADA-2000 (2026-09-06) *inverted*
the ordering: −0.0918 for the same comparison, because DADA's median clip is
9 sampled frames under a 9-tap score head. A component whose sign flips with the
training corpus's clip length is a smoothing hyperparameter, not a motion
mechanism.

So TAD is the **third training corpus**, and it answers exactly one question:

> Does the smoother's contribution keep the MSAD sign on a corpus that is
> neither MSAD nor DADA — fixed-camera, one anomaly class, ~410 train clips?

Three outcomes, all publishable, none of them "KIP works after all":

| Result | Reading |
|---|---|
| MSAD sign reproduces | Two corpora agree, one (DADA) is explained by its clip geometry. The smoothing claim gets stronger. |
| DADA sign reproduces | The MSAD result is the outlier. Check TAD's clip-length distribution against the head kernel (§6) before saying anything else. |
| Everything is null | TAD's train split (~410 clips, `C = 2`) may be too small or too homogeneous to move a transfer number. A **bounded** null, reported as one. |

**Do not run TAD hoping for a bigger number** (lesson **C14**), and do not tune
anything on a TAD delta.

### 0.3 The two code prerequisites — shipped 2026-09-02, present on `main`

| # | What was missing | What landed |
|---|---|---|
| **P1** | `core/data/tad.py` was **eval-only** — it wrote `labels_train.json` as `{}` by construction, so `DVSFeatureDataset` raised `Training set needs both classes: 0 normal, 0 abnormal` (`core/data/dataset.py:80`) and no TAD run was possible | `--with-train-split` builds the weakly-supervised train split from the frame folders the annotation does not name. **Default is unchanged**: without the flag the module is still eval-only, so every pre-2026-09-02 TAD artifact is reproducible bit-for-bit |
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

### 0.4 What `main` can and cannot run — read before copying any command

**The KIP flag set on this branch is exactly:**

```
kip.enabled  kip.pmg_only  kip.use_gate_shift  kip.use_lkin
kip.gate_signal  kip.on_raw_features
kip.folding_factor  kip.d_flow  kip.pmg_latent_dim  kip.align_proj_dim
```

Nothing else. `gate_type`, `const_shift_ratio`, `disable_pmg` are **branch `v3`
only**, and `train.checkpoint_every_steps` no longer exists anywhere.

Three consequences that reshape the campaign versus `TAD_V3_SETUP.md`:

1. **There is one gate, and it is never trained.** `KinematicShift` takes
   `s_t = (sigmoid(MLP(m)) * max_shift).floor().long()` and uses it as a slice
   index, so no gradient reaches the gate MLP — ever (lesson **C24**). Its input
   is min-max normalized, so `[0, 1]` is the whole reachable domain, and
   sweeping it moves `s_t` by **0–4 channels out of 128** across 8 seeds (seed 0:
   exactly 0), landing at `s ≈ 58–69`. **Every KIP-on arm on `main` is, in
   operation, a fixed ~50 % channel shift.** Never write "motion-gated" of any
   arm below. The four selectable gates live on `v3`; the gating question cannot
   be asked on this branch at all.

2. **Every KIP-on arm needs the flow cache** — `require_flow = cfg.kip.enabled`
   (`core/train.py:715`), full stop. There is no `disable_pmg` escape here, so
   v3's "decisive pair with no flow and no stage 1" does not exist. §9 (RAFT) is
   a **hard prerequisite** of M1, M2 and M3, and it moves earlier in the order of
   work than the v3 runbook puts it.

3. **At eval the checkpoint defines the architecture, not the CLI.**
   `adopt_checkpoint_architecture` (`core/inference.py:83`) replaces `cfg.model`
   and `cfg.kip` with the sections stored in the checkpoint, and an explicit
   `--set model.*` / `--set kip.*` that *contradicts* the checkpoint **raises**
   (lesson **C34**). This is the opposite of the v3 rule "repeat the gate flags
   on every eval": here you pass a `kip.*` flag at eval only when it agrees with
   how the arm was trained, and the safe default is to pass none. It also means
   v3's silent failure — scoring one gate's checkpoint under another — is
   structurally impossible on `main`.

A fourth, smaller one: `core/evaluate.py` on this branch has **no
`--dump-kip-diag`**, and a score `.npz` holds only `score`, `sim`,
`class_names`, `gt` (`core/inference.py:255`). The gate-verification cell from
`TAD_V3_SETUP.md` §4.2 cannot run here; §14.3 is its replacement.

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
$KATVAD_DATA_ROOT/TAD/train_ids.txt                             ~410 train ids (§5.2 only)
$KATVAD_CACHE_ROOT/clip/TAD_ncc/{id}.npy                        (L,512) no_center_crop
$KATVAD_CACHE_ROOT/flow/v1/TAD/{id}.npy                         (L,256) e_O, train ids only
$KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz                    DVS filler cache
$KATVAD_OUTPUT_ROOT/TAD/{seed}/{arm}/...                        every run in §13/§14
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
%cd /content/drive/MyDrive/Thesis/kat-vad
!git checkout main && git pull
!pip install -q "torch==2.4.*" "torchvision==0.19.*" "transformers==4.56.*" \
  "numpy<2" "av>=12" "einops>=0.8" "faiss-cpu>=1.8" "gdown>=5" \
  "matplotlib>=3.9" "opencv-python==4.11.*" "pyyaml>=6" \
  "requests>=2.32" "scikit-learn>=1.5" "torchmetrics>=1.4" "tqdm>=4.66"
```

**Restart the runtime** (`Runtime → Restart session`), then re-mount and re-run
§2.2. A batch `pip install` is atomic and a failed `!pip` does not stop the
notebook (lesson **C21**) — check the cell actually succeeded before moving on.

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

### 2.3 Branch preflight — confirm the tree has what §12 needs, and lacks what it must lack

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
git branch --show-current                                   # main

python -m core.data.tad       --help | grep -c with-train-split   # >= 1
python -m core.flow.raft_extract --help | grep -c frames-dir      # >= 1
python -m core.tools.eda report  --help | grep -c score-head-kernel # >= 1

python - <<'PY'
from core.config import load_config
for flag in ("kip.gate_type=constant", "kip.disable_pmg=true",
             "train.checkpoint_every_steps=100"):
    try:
        load_config(None, [flag]); print("UNEXPECTED: parsed", flag)
    except KeyError as e:
        print("ok, rejected:", e)
PY
```

If any of the three `KeyError` lines does **not** appear, you are not on `main`
and §12–§16 are the wrong runbook — use `core/docs/v3/setup/TAD_V3_SETUP.md`.

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
that reads as a broken download (lesson **C20**):

```bash
%%bash
# only if `ls /content/tad` printed something other than `frames`
mv /content/tad/*/frames /content/tad/frames_tmp && \
  rm -rf /content/tad/archived && mv /content/tad/frames_tmp /content/tad/frames
```

---

## 4. Ingest gates — run all three before extracting anything

An existing directory is not evidence of data (lesson **C10**). Count *files*.

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
   # (lesson C20). Idempotent — already-padded names are left alone.
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

### 5.1 Test split only — the cheap parse check

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
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

Then re-run without `--dry-run`. Writes `labels_train.json` (**empty** — that is
what eval-only means; §5.2 fills it), `frame_labels_test.json`, `defs.json`
(`["Normal", "Car Accident"]`), `meta.json`, `test_ids.txt`.

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

### 5.2 Train + test — this is the one training needs

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
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
is total (lesson **C19** cannot fire here). A folder under neither directory
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

Verify, and record the counts — §11 needs them:

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
the dataset name (lesson **C12**). Never put a raw-pooled TAD number and a
min-max-pooled DoTA number in the same column.

---

## 6. EDA pre-flight — characterize TAD before you trust a number measured on it

**New in this rewrite, and not optional.** The memory bank's trigger map is
explicit: *train on a new corpus → run `core.tools.eda` first*. DADA-2000 cost
this project a whole campaign because nobody did (lessons **C27**, **C28**), and
TAD is the first corpus profiled *before* training rather than after.

Run the label-only sections now (seconds, no feature cache needed):

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
python -m core.tools.eda report \
  --dataset TAD \
  --data-dir   "$KATVAD_DATA_ROOT/TAD" \
  --sections corpus,labels,protocol \
  --output-dir "$KATVAD_OUTPUT_ROOT/eda/TAD" \
  --plots
```

Re-run it **after §7** with `--clip-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc"` and
no `--sections` to add the feature sections and the linear probe — that is the
run that tells you whether a frozen-CLIP representation can support a
frame-level claim on TAD at all.

Read `eda_report.md` and answer three questions in the run notes **before**
training anything:

| Gate | Where | Pass condition | If it fails |
|---|---|---|---|
| **E-1 — clip-length leak (C28)** | §3.3 | length-only AUC **< 0.55**, no CRITICAL verdict | TAD's abnormal/normal videos differ in duration and a ruler beats the model. Fixed-length windows (`core/data/windows.py`) would be needed; that is DADA's Phase 2, and it is a corpus rebuild, not a flag |
| **E-2 — score-head span (C27)** | §1.2 | `score_head_kernel=9` covers **< 50 %** of the median clip | the 9-tap head sees the whole clip and every arm is a clip classifier. Lower `model.score_head_kernel` *for every arm equally*, and say so |
| **E-3 — clip oracle (C12)** | §3.4 | printed, **not gated** — a threshold here is unreachable by construction (lesson **C33**) | nothing; but the number goes beside every micro AUC you report |

TAD is expected to pass E-1 and E-2 comfortably — it is fixed-camera
surveillance with full-length videos in both classes, not DADA's trimmed
accident clips. **Expected is not measured.** If E-1 fires, stop and re-read
`DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` before spending a GPU hour.

Also record the median sampled clip length from §1.1: §11's sizing and the
E-2 reading both depend on it, and `data.max_vis_len = 512` only matters if TAD
clips exceed it.

---

## 7. CLIP features — `no_center_crop`, stride 8

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
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
  (lesson **C11**); re-running does only what is missing and logs
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

## 8. Gate T0 — reproduce the zero-shot number before you train anything

**Run this before §9 and everything after it.** It costs no training, and it
validates the entire ingest — unzip, frame order, label arithmetic, stride,
pooling — against a checkpoint whose TAD score is published. If T0 fails,
nothing downstream of it means anything.

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
python -m core.evaluate \
  --baseline-ckpt "$KATVAD_CKPT_ROOT/best.ckpt" \
  --set kip.enabled=false --set data.dataset=TAD \
  --data-dir "$KATVAD_DATA_ROOT/TAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD/gate_t0"
```

`--baseline-ckpt` takes the LaGoVAD compat loader, which carries no `config`
section, so §0.4's checkpoint-authoritative rule does not apply here and
`--set kip.enabled=false` is the right way to say "trunk only".

Reference: `TAD_ZERO_SHOT_AUC = 89.56`, `GATE_A_TOLERANCE = 0.5`
(`core/constants.py`).

**How to read the result, honestly** (lessons **C8** and **C8b**):

| Outcome | Reading |
|---|---|
| micro AUC within ±0.5 of 89.56 | ingest verified. Proceed. |
| 85–89 | plausible-but-unverified. The most likely cause is **sampling rate**: LaGoVAD scored precomputed `tad_features/*.npy` whose extraction interval is not recorded anywhere in its repo, and we sample stride 8. Try `--stride 1` labels + features on the 100 test clips only (cheap) and compare before concluding anything about the model. |
| ≪ 85, or ≈ 50 | a pipeline defect, not a protocol difference. Prime suspect: §4.3 frame ordering. Check that first. |

### 8.1 MEASURED 2026-09-14 — T0 = **0.7912**, and what that does *not* mean

| | |
|---|---:|
| micro AUC (raw pooling, stride 8) | **0.7912** |
| macro AUC (n = 60 two-class clips) | **0.7578** |
| `TAD_ZERO_SHOT_AUC` | 89.56 → **short by 10.4 points** |

By the table above that lands in "pipeline defect, prime suspect §4.3 frame
ordering". **Four of the five suspects are exonerated by measurement:**

| Check | Result | Verdict |
|---|---|---|
| ids vs `LaGoVAD-PreVAD/data/other_datasets/tad_test_anno.json` | 100/100, none extra, none missing | ✅ |
| frame labels rebuilt from that file's own `anomaly_span` | **0/100 mismatch** | ✅ label arithmetic exact |
| stride | ours 8 · baseline `full_length_eval.py:28` `interval=8` | ✅ identical |
| pooling | ours `none` · baseline raw | ✅ identical |
| **frame ordering** | `auc_macro = 0.7578`, `d = gap/σ = +0.90` | ✅ **shuffled frames give macro ≈ 0.50. Ordering is right.** |

> **Add this to the table above, it is the cheapest check in the runbook:**
> a macro AUC well clear of 0.50 exonerates frame ordering in one step. Read it
> *before* re-auditing the ingest. Do not send anyone to §4.3 on a micro number
> alone.

Two suspects survive, and the first explains most of the gap:

**1. micro AUC on TAD is a length-weighted clip classifier.** Measured from the
saved `.npz` of `gate_t0`:

| | |
|---|---:|
| micro as reported (frames weighted, i.e. by clip length) | 0.7912 |
| micro with every clip weighted **equally** | **0.6574** |
| micro of a "ruler" reading **only** the frame count | **0.8968** |
| constant-score-per-clip oracle (EDA §3.2) | **0.9226** |

Normal clips have median **139** sampled frames, abnormal **36** — 3.9× longer —
and **99.69 %** of positive/negative frame pairs span two clips. We unzip frame
folders; LaGoVAD ran `ffmpeg select mod(n,8)` on mp4s. A few frames of
difference in `T` per clip moves micro by this much with no bug anywhere.

**2. Transform.** Our cache is `no_center_crop`; `tad_features/*.npy` used
LaGoVAD's own transform (**C13**, **C2**). Same direction elsewhere: ncc costs
`best.ckpt` 0.8991 → 0.8949 on MSAD and 0.6142 → 0.6012 on DoTA.

**The reference for every later TAD arm is 0.7912** (lesson **C8b**). 89.56 does
not appear in any TAD table again. Note it is itself *below* this corpus's clip
oracle of 0.9226 — reproducing it perfectly would still be reproducing clip
classification.

**Gate against the released checkpoint's own number, not the printed one**
(lesson **C8b**). Whatever `best.ckpt` scores here *is* the reference every later
TAD arm is compared to. Record it in the run notes with the date; do not carry
89.56 forward as if we had reproduced it. MSAD is the precedent: we logged a
gate as failing at 0.8922 vs a published 0.9041 while `best.ckpt` itself scored
0.8991.

```bash
%%bash
python -m core.tools.rescore --run-dir "$KATVAD_OUTPUT_ROOT/TAD/gate_t0"
```

`rescore` recomputes the metric under every pooling rule from the saved `.npz`.
It is seconds, and it is the check that catches a silently wrong `auto`.

---

## 9. RAFT flow targets — train ids only, and **mandatory for M1/M2/M3**

`e_O` is a **train-time** target for `L_KIP_rec` / `L_KIP_align`. It is never on
the inference path, and the test split never needs it. Extract for the ~410
train ids only.

On `main` this step is not optional for any KIP-on arm: `require_flow =
cfg.kip.enabled` (`core/train.py:715`), so even M2 — which weights both flow
losses to zero — will not build its dataset without the cache (§0.4).

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
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
  `len(flow) == len(features)` per video and raises
  `Flow/feature length mismatch for {id}` otherwise
  (`core/data/dataset.py:124`). That assert is the only thing standing between
  you and a silent misalignment.
- The flow target has **23 effective dimensions** — frame-global scalars lifted
  to 256-d by a fixed seeded projection (`core/flow/raft_extract.py:52-81`).
  Never claim KIP localizes anything spatially.

---

## 10. DVS KNN filler cache

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
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

## 11. Sizing a TAD run — compute it, don't copy MSAD's

`DVSFeatureDataset.__len__` is `2 × num_abnormal_train`
(`core/data/dataset.py:98`), **not** the number of train videos. MSAD-full's
`num_epochs=125 ≈ 500 steps` does not carry over.

```python
import json, math, os
tr = json.load(open(f"{os.environ['KATVAD_DATA_ROOT']}/TAD/labels_train.json"))
A  = sum(tr.values())
BATCH, TARGET_STEPS = 64, 500
steps_per_epoch = math.ceil(2 * A / BATCH)
E = math.ceil(TARGET_STEPS / steps_per_epoch)
print(f'{A} abnormal train videos -> dataset len {2*A} -> {steps_per_epoch} steps/epoch')
print(f'E (num_epochs for ~{TARGET_STEPS} steps) = {E}')
```

**Write `E` down and use the same value in every arm and both stages.** An arm
trained for a different number of steps is not a control — it contains a
training-length difference the Δ cannot separate from the treatment. Warm-up
stays at 20 steps, peak LR 5e-5, batch 64, unchanged from the baseline
(`core/constants.py`).

**Step checkpoints no longer exist** (removed 2026-09-13, commit `cb7e2ac`): a
run writes only `checkpoint_last.pt`, and `train.checkpoint_every_steps` now
raises. Lesson **C16** still stands for any future trajectory probe —
step-uniform checkpoints undersample the loss range — but such a probe needs the
knob reintroduced deliberately rather than switched on.

---

## 12. The arm matrix on `main`

Four arms. They are **not** `TAD_V3_SETUP.md`'s A0–A4 and must never share a
table with them without the mapping column below: different branch, different
flags, different model.

Common flags for every stage-2 command:

```
--set train.stage=2 --set train.amp=true
--set data.dataset=TAD --set data.is_egocentric=false
--set train.num_epochs=$E --set train.seed=$S
```

| Arm | Extra flags | What it is | Stage 1? | Flow cache? |
|---|---|---|---|---|
| **M0** | `--set kip.enabled=false` | KIP-off trunk. The arm every Δ subtracts from | no (raises) | **no** |
| **M1** | *(none — KIP defaults)* | **KAT-VAD v1, full.** PMG head + shift + `L_KIP_rec`/`L_KIP_align`/`L_kin`. The arm every `RESULTS_*.md` number on this branch refers to | **yes** (§13.2) | yes |
| **M2** | `--set loss.lambda_rec=0 --set loss.lambda_align=0 --set kip.use_lkin=false` | **Smoother-only control.** The shift stays in the forward pass; every KIP loss is off, so no KIP parameter ever receives a gradient and the module is frozen at init | **no** | yes (loader only) |
| **M3** | `--set kip.pmg_only=true` | **Flow-objective-only control.** `use_gate_shift = cfg.use_gate_shift and not cfg.pmg_only` (`core/kip/kip_module.py:60`), so `v^k = v^t` — no smoothing at all — while `L_KIP_rec`/`L_KIP_align` still regularize the shared trunk | **yes** (§13.2) | yes |

`M2` and `M3` are a **decomposition**: M2 keeps the smoother and drops the flow
objective, M3 keeps the flow objective and drops the smoother, M1 has both, M0
neither. That is the cleanest attribution this branch can express, and it is the
`main`-side analogue of what v3 did with `gate_type=constant`.

### 12.1 Mapping to the v3 arms — for cross-branch reading only

| `main` arm | Nearest `v3` arm | Same? |
|---|---|---|
| M0 | A0 (`kip.enabled=false`) | **yes** — identical config, and `p1_ctrl` on DADA reproduced v3's A0 to 4 dp with 331/331 bitwise-identical curves (2026-09-12). A `main`-vs-`v3` Δ is valid **for `kip.enabled=false` only** |
| M1 | A3 (`gate_type=mlp_frozen`) | **yes in effect** — `mlp_frozen` *is* v1's gate |
| M2 | A2 (`gate_type=constant`, `const_shift_ratio=0.5`) | **no, only in effect.** v3 fixes the shift at 64/128 by construction; M2's shift is whatever the frozen random-init gate emits, measured at `s ≈ 58–69` with a span of 0–4 channels (C24). Verify it per run — §14.3 |
| M3 | *none* | v3 has no PMG-only arm |

### 12.2 What raises, so you recognise it

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Training set needs both classes: 0 normal, 0 abnormal` | `labels_train.json` is `{}` | you preprocessed **without** `--with-train-split`. §5.2 |
| `ValueError: Missing flow cache .../TAD/{id}.npy` | any KIP-on arm without §9 | run §9. There is no `require_flow=false` escape on `main` (§0.4) |
| `ValueError: Flow/feature length mismatch for {id}` | flow and CLIP caches built at different strides | rebuild one at the other's stride; nothing on disk records stride (**C2**) |
| `ValueError: ... neither of the split directories ... appears in its path` | a frame folder sits outside `abnormal/` and `normal/` | fix the layout; the directory is TAD's only train-split label |
| `ValueError: Stage 1 (KIP warm-up) requires kip.enabled=true` | `train.stage=1` with `kip.enabled=false` | M0 has no stage 1, by design |
| `KeyError: Unknown config key: kip.gate_type` | you copied a command out of `TAD_V3_SETUP.md` | use §13 |
| `ValueError: --set kip.X=... contradicts the checkpoint, which was trained with kip.X=...` | an eval flag disagrees with the arm | drop the flag. The checkpoint is authoritative (**C34**, §0.4) |

---

## 13. Training

Output dirs are the contract with §14. Train an arm elsewhere and §14's `--ckpt`
paths will not find it.

```
$KATVAD_OUTPUT_ROOT/TAD/$S/m0/stage2
$KATVAD_OUTPUT_ROOT/TAD/$S/m1/{stage1,stage2}
$KATVAD_OUTPUT_ROOT/TAD/$S/m2/stage2
$KATVAD_OUTPUT_ROOT/TAD/$S/m3/stage2          (warm-started from m1/stage1)
```

Everything lives under `TAD/`, including the DoTA evals (§14.1) — that is
deliberate. The MSAD campaign already owns `$KATVAD_OUTPUT_ROOT/DoTA_*`, and two
training corpora scoring the same benchmark must never share an output dir
(lesson **C17**).

Seeds **2024 2025 2026**, matching every other campaign. **One experiment = one
`--output-dir`**: `metrics.jsonl` appends, so a restart into the same dir
silently duplicates rows — that happened on MSAD A3 and had to be deduped by
hand.

### 13.0 The T-ladder — **run this before §13.1**

Plan: `.project/plans/katvad-tad-loss-ladder.md`. Three runs, seed 2024,
**504 steps each** — minutes of GPU, no stage 1, no flow cache, no
re-extraction. It tests the one defect §15.2 measured on `m0`: under clip-level
MIL nothing lowers a frame inside a positive bag, and TAD's abnormal clips are
only **33 %** positive, so **67 % of their frames get no downward pressure**.

| arm | `loss.dvs_anchor_mode` | `loss.bottomk_weight` | output dir |
|---|---|---|---|
| `t1_ctrl` | `span` | `0.0` | **= the existing `TAD/2024/m0`** — do not retrain |
| `t1_dvsignore` | **`ignore`** | `0.0` | `TAD/$S/t1_dvsignore/stage2` |
| `t1_bottomk` | `span` | **`1.0`** | `TAD/$S/t1_bottomk/stage2` |
| `t1_both` | **`ignore`** | **`1.0`** | `TAD/$S/t1_both/stage2` |

Names mirror DADA Phase 1 (`p1_ctrl` / `p1_dvsignore` / `p1_bottomk` / `p1_both`)
so the two tables join. `bottomk_weight=1.0` and `bottomk_topk_pct=16` are
**identical to DADA's** — do **not** sweep them (lesson **14**); the join is the
point, and 1.0 is a symmetry prior against `L_MIL`, not a tuned number.

> **Why this is not a repeat of DADA Phase 1, where both arms failed.** That run
> sat under an open precondition defect: DADA's median clip is **9** sampled
> frames under `score_head_kernel=9`, so no within-clip resolution existed for
> either term to act on (R2 / **C27**). By **C14** that makes it *unmeasured*,
> not *refuted*. TAD's median clip is **43** and kernel 9 spans **20.9 %** — the
> confound is absent here.

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
S=2024 ; E=72          # §11: 200 abnormal train -> DVS len 400 -> 7 steps/epoch -> 504 steps
                       # E MUST equal m0's num_epochs or m0 is not a control

COMMON="--set train.stage=2 --set train.amp=true --set data.dataset=TAD \
  --set data.is_egocentric=false --set train.num_epochs=$E --set train.seed=$S \
  --set kip.enabled=false \
  --data-dir  $KATVAD_DATA_ROOT/TAD \
  --clip-dir  $KATVAD_CACHE_ROOT/clip/TAD_ncc \
  --knn-cache $KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz"

# --- t1_dvsignore: stop training the 67% non-anomalous anchor frames to 1 (C29)
python -m core.train $COMMON \
  --set loss.dvs_anchor_mode=ignore \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD/$S/t1_dvsignore/stage2"

# --- t1_bottomk: the missing downward term on abnormal clips
python -m core.train $COMMON \
  --set loss.bottomk_weight=1.0 --set loss.bottomk_topk_pct=16 \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD/$S/t1_bottomk/stage2"

# --- t1_both
python -m core.train $COMMON \
  --set loss.dvs_anchor_mode=ignore \
  --set loss.bottomk_weight=1.0 --set loss.bottomk_topk_pct=16 \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD/$S/t1_both/stage2"
```

All three are `kip.enabled=false`, so **no `--flow-dir`, no stage 1** — §9's flow
cache is not needed for this ladder (§10's KNN cache still is).

**Where each kind of typo surfaces — verified on this branch 2026-09-14:**

| Typo | Raises | When |
|---|---|---|
| the *key* (`loss.bottmk_weight`) | `KeyError: Unknown config key` | at parse, instantly |
| the *value* of `dvs_anchor_mode` (`ignor`) | `ValueError: must be one of ('span', 'ignore')` | **`core/train.py:327` — at the first training step with DVS rows**, i.e. after model build and feature loading |
| **omitting `--set loss.bottomk_weight=1.0` entirely** | **nothing** | never — the arm silently trains as the control |

The third is the dangerous one, and it is why §15.1 diffs the `config.yaml`s
before any Δ is read. Spend ten seconds on this preflight first:

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
python - <<'PY'
from core.config import load_config
base = ["train.stage=2","data.dataset=TAD","train.num_epochs=72",
        "train.seed=2024","kip.enabled=false"]
arms = {"t1_ctrl":       [],
        "t1_dvsignore":  ["loss.dvs_anchor_mode=ignore"],
        "t1_bottomk":    ["loss.bottomk_weight=1.0","loss.bottomk_topk_pct=16"],
        "t1_both":       ["loss.dvs_anchor_mode=ignore",
                          "loss.bottomk_weight=1.0","loss.bottomk_topk_pct=16"]}
for n, extra in arms.items():
    c = load_config(None, base + extra)
    print(f"{n:<14} anchor={c.loss.dvs_anchor_mode:<7} bottomk_w={c.loss.bottomk_weight} "
          f"topk_pct={c.loss.bottomk_topk_pct} E={c.train.num_epochs}")
PY
```

Expected exactly:

```
t1_ctrl        anchor=span    bottomk_w=0.0 topk_pct=16 E=72
t1_dvsignore   anchor=ignore  bottomk_w=0.0 topk_pct=16 E=72
t1_bottomk     anchor=span    bottomk_w=1.0 topk_pct=16 E=72
t1_both        anchor=ignore  bottomk_w=1.0 topk_pct=16 E=72
```

`t1_ctrl`'s row must match `TAD/2024/m0/stage2/config.yaml` — that is what makes
the existing `m0` a valid control instead of a run that merely looks like one.

**One experiment = one `--output-dir`.** `metrics.jsonl` appends; restarting into
a used dir silently duplicates rows.

**Known in advance, recorded so it cannot become an excuse:** `_topk_k` computes
`k = max(1, n // topk_pct)`, so at TAD's median `n = 43` the bottom-k term pushes
down only **k = 2** frames per abnormal clip. If the ladder nulls, that is "weak
pressure", not "no mechanism" — say so, and do not respond by tuning `topk_pct`
against the result.

---

### 13.1 M0 and M2 — the cheap pair, run these first

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
S=2024 ; E=<from §11>
COMMON="--set train.stage=2 --set train.amp=true --set data.dataset=TAD \
  --set data.is_egocentric=false --set train.num_epochs=$E --set train.seed=$S \
  --data-dir  $KATVAD_DATA_ROOT/TAD \
  --clip-dir  $KATVAD_CACHE_ROOT/clip/TAD_ncc \
  --knn-cache $KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz"

# --- M0: KIP-off trunk (no flow-dir needed) ---
python -m core.train $COMMON \
  --set kip.enabled=false \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD/$S/m0/stage2"

# --- M2: smoother only, every KIP loss off, NO stage 1 ---
python -m core.train $COMMON \
  --set loss.lambda_rec=0 --set loss.lambda_align=0 --set kip.use_lkin=false \
  --flow-dir "$KATVAD_CACHE_ROOT/flow/v1/TAD" \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD/$S/m2/stage2"
```

M0 is cold by construction: stage 1 requires KIP (`core/train.py:187`) and
trains **only** `kip.*` parameters, so warm arms carry no trunk advantage and a
cold M0 is a fair comparator.

M2 skips stage 1 on purpose — with `lambda_rec = lambda_align = 0` and
`use_lkin=false` there is nothing to warm up, and the gate is non-differentiable
anyway (**C24**). Its `kip.*` weights stay at initialization for the whole run.
`kip_rec` and `kip_align` are still *computed* and multiplied by zero, which is
why the flow cache is still read; that is wasted compute, not a wrong gradient.

### 13.2 Stage 1 — KIP warm-up (M1 and M3 only)

Stage 1's loss is `lambda_rec·L_KIP_rec + lambda_align·L_KIP_align`. One run per
seed serves **both** M1 and M3: the two share a config except for `pmg_only`,
which only removes a module that holds no parameters of its own beyond the gate
MLP, and the checkpoint key layout is identical.

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
S=2024 ; E=<from §11>
python -m core.train \
  --set train.stage=1 --set data.dataset=TAD --set data.is_egocentric=false \
  --set train.num_epochs=$E --set train.seed=$S \
  --data-dir  "$KATVAD_DATA_ROOT/TAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/TAD" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD/$S/m1/stage1"
```

#### 13.2a Convergence check — a **stop**, not a warning

The spec's `assert stage1_final(L_KIP_rec) < τ_rec` is **not implemented**. No
gradient reaches `ê_O` through the shift, so a stage-2 arm will happily consume a
badly-trained `ê_O` and hand you a plausible number.

```python
import json, pathlib, os
S = 2024
p = (pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])
     / 'TAD' / str(S) / 'm1' / 'stage1' / 'metrics.jsonl')
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
rec  = [r['kip_rec'] for r in rows if 'kip_rec' in r]
n    = max(1, len(rec) // 10)
first, last = sum(rec[:n])/n, sum(rec[-n:])/n
print(f'first {n} mean {first:.3f} | last {n} mean {last:.3f} | ratio {last/first:.2f}')
assert last < 0.6 * first, (
    f'STOP: L_KIP_rec did not roughly halve ({first:.3f} -> {last:.3f}). '
    'M1 and M3 have no result until stage 1 converges.')
```

The MSAD/v1 reference trajectory was 19.2 → 9.7 with a plateau from ~step 200.
TAD's absolute scale will differ — its flow statistics are a different
distribution — so read the **ratio**, not the value. A last-window mean that has
not roughly halved means the arm is **blocked**, not weak (lesson **C14**):
report it as blocked, with no number.

### 13.3 M1 and M3 — stage 2, warm-started

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
S=2024 ; E=<from §11>
WARM="$KATVAD_OUTPUT_ROOT/TAD/$S/m1/stage1/checkpoint_last.pt"
COMMON="--set train.stage=2 --set train.amp=true --set data.dataset=TAD \
  --set data.is_egocentric=false --set train.num_epochs=$E --set train.seed=$S \
  --data-dir  $KATVAD_DATA_ROOT/TAD \
  --clip-dir  $KATVAD_CACHE_ROOT/clip/TAD_ncc \
  --flow-dir  $KATVAD_CACHE_ROOT/flow/v1/TAD \
  --knn-cache $KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz"

# --- M1: KAT-VAD v1, full KIP ---
python -m core.train $COMMON --init-weights "$WARM" \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD/$S/m1/stage2"

# --- M3: flow objective, no shift ---
python -m core.train $COMMON --init-weights "$WARM" \
  --set kip.pmg_only=true \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD/$S/m3/stage2"
```

**M3 sharing M1's stage 1 is deliberate** — it removes the one difference that is
not the treatment, which is the entire point of `M1 − M3`.

### 13.4 Resume, and the manual manifest

- Dead session: same command with `--init-weights ...` replaced by
  `--resume "<output-dir>/checkpoint_last.pt"`. They are mutually exclusive and
  the parser says so; resume carries the warm-started weights forward anyway.
- `--stop-after-epochs N` caps one invocation without shrinking the LR horizon.
- **Lesson C15:** checkpoints pickle `np.random.get_state()`. A Colab runtime that
  drifts to a different numpy major makes every checkpoint on Drive unloadable
  *before reaching a tensor*. Recover with the `find_class` unpickler in
  `COLAB.md` §A4.0; do not pin `numpy<2` away.
- **Lesson C17 — the run manifest is not built.** `config.yaml` records the
  config tree but not `--init-weights`, `--data-dir`, `--clip-dir`, `--flow-dir`,
  `--knn-cache`, the commit, or `sys.argv`. A warm-started arm's `config.yaml` is
  byte-identical to a cold one's. Write the manifest by hand, once per run:

```python
import json, os, subprocess, pathlib, sys
S, ARM = 2024, 'm2'
REPO = '/content/drive/MyDrive/Thesis/kat-vad'
run = pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])/'TAD'/str(S)/ARM/'stage2'
(run/'run_manifest.json').write_text(json.dumps({
    'corpus': 'TAD', 'branch': 'main', 'arm': ARM,
    'description': 'smoother only: lambda_rec=0, lambda_align=0, use_lkin=false',
    'seed': S, 'num_epochs': None,          # fill in E
    'init_weights': None,                    # m1/m3: the stage-1 path
    'clip_dir': f"{os.environ['KATVAD_CACHE_ROOT']}/clip/TAD_ncc",
    'flow_dir': f"{os.environ['KATVAD_CACHE_ROOT']}/flow/v1/TAD",
    'knn_cache': f"{os.environ['KATVAD_CACHE_ROOT']}/knn/TAD_ncc/knn_cache.npz",
    'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],
                  cwd=REPO).decode().strip(),
    'prediction': 'see §15 — pre-registered before the first eval',
}, indent=2))
```

---

## 14. Evaluation

Every trained arm gets **two** evals from the *same* `checkpoint_last.pt`:

| Eval | Benchmark | `--data-dir` | `--clip-dir` | `--score-norm auto` resolves to |
|---|---|---|---|---|
| **in-domain** | TAD test (100 clips, ~40 % normal) | `$KATVAD_DATA_ROOT/TAD` | `clip/TAD_ncc` | **raw** |
| **zero-shot** | DoTA val (~1,402 clips, ~all abnormal) | `$KATVAD_DATA_ROOT/DoTA/labels_s8` | `clip/DoTA_s8_ncc` | **min-max** |

**Those are two different metrics.** Never put a raw-pooled TAD number and a
min-max-pooled DoTA number in the same column, and always report `auc_macro`
beside the micro number (lesson **C12**).

The DoTA rows are reused from the MSAD campaign untouched — no transform, no
stride, no pooling changed on this branch, so lesson **C2** does not fire. Check
they exist before starting:

```bash
%%bash
for p in "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" "$KATVAD_DATA_ROOT/DoTA/labels_s8"; do
  n=$(ls "$p" 2>/dev/null | wc -l); echo "$n  $p"
done
```

### 14.1 The two eval blocks — all four arms, one seed

**Pass no `kip.*` flag.** The checkpoint carries its own `kip` section and it
wins; a contradicting flag raises (§0.4, lesson **C34**). This is the opposite of
the v3 runbook's rule, and the reason M0's eval below carries no
`kip.enabled=false` either.

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
S=2024
for ARM in m0 m1 m2 m3 ; do
  D="$KATVAD_OUTPUT_ROOT/TAD/$S/$ARM"
  CKPT="$D/stage2/checkpoint_last.pt"
  [ -f "$CKPT" ] || { echo "skip $ARM (not trained)"; continue; }

  # (a) TAD, in-domain -> raw pooling
  python -m core.evaluate --ckpt "$CKPT" \
    --set data.dataset=TAD \
    --data-dir "$KATVAD_DATA_ROOT/TAD" \
    --clip-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
    --score-norm auto --save-scores --output-dir "$D/eval_tad"

  # (b) DoTA, zero-shot transfer -> min-max pooling
  python -m core.evaluate --ckpt "$CKPT" \
    --set data.dataset=DoTA \
    --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
    --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
    --score-norm auto --save-scores --output-dir "$D/eval_dota"
done
```

### 14.1a The T-ladder evals (§13.0) — 6 evals, same two blocks

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
S=2024
for ARM in t1_dvsignore t1_bottomk t1_both ; do
  D="$KATVAD_OUTPUT_ROOT/TAD/$S/$ARM"
  CKPT="$D/stage2/checkpoint_last.pt"
  [ -f "$CKPT" ] || { echo "skip $ARM (not trained)"; continue; }

  python -m core.evaluate --ckpt "$CKPT" \
    --set data.dataset=TAD \
    --data-dir "$KATVAD_DATA_ROOT/TAD" \
    --clip-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
    --score-norm auto --save-scores --output-dir "$D/eval_tad"

  python -m core.evaluate --ckpt "$CKPT" \
    --set data.dataset=DoTA \
    --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
    --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
    --score-norm auto --save-scores --output-dir "$D/eval_dota"
done
```

`t1_ctrl`'s two evals already exist as `TAD/2024/m0/eval_tad` and
`.../eval_dota`. Do not re-run them — and do not move them either; §15.1 reads
`m0` by that name.

---

### 14.2 Reference arms — gate-independent, run once

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
# T0 — LaGoVAD's released trunk on TAD (§8). Re-run only if §8 was skipped.
# gate_a — the same trunk on DoTA; reuse the MSAD campaign's dir if it exists.
python -m core.evaluate --baseline-ckpt "$KATVAD_CKPT_ROOT/best.ckpt" \
  --set kip.enabled=false --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_gate_a/gate_a"
```

Gate against **the released checkpoint's own number**, not the paper's printed
89.56 / 62.60 (lesson **C8b**).

### 14.3 Verify the shift you actually trained — MANDATORY for M1 and M2

`main` has no `--dump-kip-diag` and the score `.npz` holds no KIP diagnostics
(§0.4), so the v3 gate-check cell does not run here. This is its replacement: it
reads the checkpoint's own config, then measures the realized shift on real
cached features through the public API.

The claim under test is C24's: that M1's and M2's gates are near-constant at
`s ≈ 58–69` out of 128, with a per-clip span of 0–4 channels. **If a TAD-trained
arm's span is large, C24 does not hold on this corpus and every "fixed smoother"
sentence in the write-up is wrong** — which would be a finding, not a bug.

```python
import json, os, pathlib, torch, numpy as np
from core.config import load_config
from core.inference import load_model_for_scoring

S, ARM = 2024, 'm2'
OUT  = pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])
CKPT = OUT/'TAD'/str(S)/ARM/'stage2'/'checkpoint_last.pt'
CLIP = pathlib.Path(os.environ['KATVAD_CACHE_ROOT'])/'clip'/'TAD_ncc'

cfg   = load_config(None, ['data.dataset=TAD'])
model = load_model_for_scoring(cfg, torch.device('cpu'), CKPT, None, 'stub', [])
assert model.kip is not None, f'{ARM}: KIP-off checkpoint — nothing to measure'
assert model.kip.shift is not None, f'{ARM}: pmg_only -> no shift (expected for m3)'

spans, means = [], []
for p in sorted(CLIP.glob('*.npy'))[:100]:
    v = torch.from_numpy(np.load(p).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        vt = model.temporal_encoder(v, torch.tensor([v.shape[1]]))
        eo = model.kip.pmg(vt)
        s  = model.kip.shift.compute_shift_counts(vt, eo)[0]
    spans.append(int(s.max() - s.min())); means.append(float(s.float().mean()))
print(f'{ARM}: span mean {np.mean(spans):.1f} of 128 '
      f'(min {min(spans)}, max {max(spans)}) | s mean {np.mean(means):.1f}')
```

> The cell passes `'stub'` as the text encoder, which skips the CLIP text
> weights entirely — nothing in this measurement touches the text branch, so
> there is no reason to download them.

| Arm | Expected span of 128 | `s` mean | If otherwise |
|---|---|---|---|
| **M1**, **M2** | **0–4** | ≈ 58–69 | C24's measurement does not transfer to TAD. Record the number, do not "fix" it |
| **M3** | the assert fires (`shift is None`) | — | `pmg_only` did not take |
| **M0** | `model.kip is None` | — | wrong checkpoint scored |

### 14.4 Rescore, and the score-count assertion

```bash
%%bash
for d in "$KATVAD_OUTPUT_ROOT"/TAD/*/m*/eval_* "$KATVAD_OUTPUT_ROOT"/TAD/gate_t0; do
  [ -d "$d/scores" ] && python -m core.tools.rescore --run-dir "$d"
done
```

**Before any Δ**, assert the score-file count against `results.json:num_videos`
in every arm (lesson **C11b**): `evaluate --save-scores` is not atomic, a 0-byte
`.npz` has been observed, and one of them perturbs **every** paired Δ in the
study, not one number.

```python
import json, os, pathlib
OUT = pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])
for d in sorted(OUT.glob('TAD/*/m*/eval_*')):
    r = d/'results.json'
    if not r.exists():
        continue
    n_files = len(list((d/'scores').glob('*.npz'))) if (d/'scores').is_dir() else 0
    n_rep   = json.load(open(r)).get('num_videos')
    zero    = [p.name for p in (d/'scores').glob('*.npz') if p.stat().st_size == 0]
    flag = '' if (n_files == n_rep and not zero) else f'   <-- MISMATCH {zero[:3]}'
    print(f'{d.relative_to(OUT)}: scores={n_files} results={n_rep}{flag}')
```

`core/evaluate.py` writes no config, so an eval dir does not record which arm
produced it. Write an `eval_manifest.json` in each with the arm name, the
checkpoint path, the pooling `auto` resolved to, and the §14.3 span.

---

## 15. Analysis and decision rules

Use `DOTA_EVAL.md` §4's analysis cell unchanged for the DoTA side — paired
bootstrap over clips, per-clip win/loss, ego/non-ego split, per-class deltas.
Report Δ per seed, then the **seed-level t-interval over n = 3**. Do not report
`3 seeds × 3 metrics × 2 benchmarks` as 18 CIs; there are 3 independent
replications.

| Comparison | What it answers | MSAD reference |
|---|---|---|
| **M2 − M0** (DoTA) | **The replication.** Does the smoother alone move zero-shot transfer when trained on TAD? | A2 − A0 = **+0.1025 ± 0.0350** (DADA inverted it: −0.0918) |
| **M1 − M2** (DoTA) | What the flow objective adds *on top of* the smoother | A2 − V1 = +0.0109, t95 [−0.0588, +0.0805] — indistinguishable |
| **M3 − M0** (DoTA) | What the flow objective buys as a pure trunk regularizer, with no smoothing | no v3 counterpart; nearest is A2b − A2 = −0.0105 |
| **M1 − M0** (DoTA) | The v1 headline. Uninterpretable without the three rows above | +0.0911 on MSAD (`RESULTS_NCC.md`) |
| **M2 − M0** (TAD, in-domain) | Does the smoother do anything in-domain? | MSAD in-domain was a bounded null, +0.0036 ± 0.0076 |

**Pre-registered predictions — write these into the run notes BEFORE the first
eval, not after:**

* **P1.** `M2 − M0 > 0` on zero-shot DoTA (the MSAD sign).
* **P2.** `M1 − M2 ≈ 0`, CI including zero (the flow objective adds nothing).
* **P3.** `M3 − M0 ≈ 0` or slightly negative.
* **P4.** Every in-domain TAD Δ is a **bounded null** at n = 3.

---

### 15.1 The T-ladder readout (§13.0) — **micro AUC is banned here**

TAD's micro AUC is a length-weighted clip classifier: the constant-per-clip
oracle scores **0.9226** and a frame-count-only ruler **0.8968** (`outputs/EDA/TAD`
§3.2–3.3, two CRITICAL verdicts). Print micro for the record; decide on these
four columns.

**What `t1_ctrl` (= `m0`, seed 2024) already measured, and the collapse it shows:**

| column | `gate_t0` (`best.ckpt`, zero-shot) | `t1_ctrl` (= `m0`, in-domain) | Δ |
|---|---:|---:|---:|
| micro AUC *(record only)* | 0.7912 | 0.9237 | +0.1325 |
| **clip-level AUC** (max score vs clip label) | 0.7687 | **0.9975** | **+0.2288** |
| **macro AUC** (n = 60) | **0.7578** | 0.6174 | **−0.1404** |
| **d = gap / within-clip σ** (**C31**) | **+0.903** | +0.419 | **−0.484** |
| **DoTA zero-shot macro** | **0.6158** | 0.5496 | **−0.0662** |

In-domain training bought near-perfect clip ranking and paid in localization
*and* transfer. `m0`'s micro 0.9237 sits just above the oracle's 0.9226 — to
three decimals it **is** a clip classifier.

**Pre-registered before the first T-ladder run (C33 — all four bars are reachable):**

* **H-T1.** `t1_bottomk` moves **all four** columns in the predicted direction
  *simultaneously*: clip-AUC ↓, macro ↑, d ↑, DoTA macro ↑. One column alone is
  noise at n = 1.
* **H-T2.** `t1_dvsignore` moves macro and d up, by less than `t1_bottomk` — it
  removes a wrong target rather than adding the missing one.
* **H-T3.** `t1_both` ≥ `t1_bottomk` on macro.
* **Worth-pursuing bar.** `t1_bottomk` or `t1_both` reaches **macro ≥ 0.65**
  (halfway to `gate_t0`'s 0.7578) **and** clip-level AUC **≤ 0.95**.
* **Falsification branch.** All four columns flat across all three arms ⇒ the
  missing-downward-term hypothesis is refuted on a corpus where DADA's C27
  confound is *absent*. Two corpora then agree, and the stride-2 re-extraction +
  windowed rebuild (plan §2) becomes justified expense rather than a guess.

**n = 1 seed. Report the sign pattern across 4 columns × 3 arms — never an
individual Δ, never a CI.** Seeds 2025/2026 only if the pattern is there.

```python
# Colab cell — the 4-column readout. Run after §14.1a.
import json, os, glob
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.environ['KATVAD_OUTPUT_ROOT'] + '/TAD'
ARMS = [('t1_ctrl (=m0)', f'{ROOT}/2024/m0'),
        ('t1_dvsignore',  f'{ROOT}/2024/t1_dvsignore'),
        ('t1_bottomk',    f'{ROOT}/2024/t1_bottomk'),
        ('t1_both',       f'{ROOT}/2024/t1_both')]
REF = ('gate_t0 (best.ckpt)', f'{ROOT}/gate_t0')

def cols(run):
    S = {}
    for f in glob.glob(f'{run}/scores/*.npz'):
        d = np.load(f, allow_pickle=True)
        S[os.path.basename(f)[:-4]] = (d['score'].astype(float), d['gt'].astype(float))
    if not S:
        return None
    sc = np.concatenate([s for s, _ in S.values()])
    gt = np.concatenate([g for _, g in S.values()])
    two = [(s, g) for s, g in S.values() if 0 < g.sum() < len(g)]
    macro = float(np.mean([roc_auc_score(g, s) for s, g in two]))
    lab = np.array([1.0 if g.sum() > 0 else 0.0 for _, g in S.values()])
    mx = np.array([s.max() for s, _ in S.values()])
    d = [ (s[g > 0].mean() - s[g == 0].mean()) / s.std()
          for s, g in two if s.std() > 0 ]
    return dict(micro=roc_auc_score(gt, sc), clip=roc_auc_score(lab, mx),
                macro=macro, d=float(np.mean(d)), n=len(two))

def dota_macro(run):
    f = f'{run}/eval_dota/results.json'
    return json.load(open(f))['auc_macro'] if os.path.exists(f) else float('nan')

print(f"{'arm':<20} {'clip AUC':>9} {'macro':>8} {'d=gap/s':>9} {'DoTA mac':>9} {'(micro)':>9}")
print('-' * 68)
name, run = REF
r = cols(run)
print(f"{name:<20} {r['clip']:>9.4f} {r['macro']:>8.4f} {r['d']:>+9.4f} "
      f"{0.6158:>9.4f} {r['micro']:>9.4f}   <- reference")
for name, run in ARMS:
    r = cols(run + '/eval_tad')
    if r is None:
        print(f'{name:<20}  (not evaluated)'); continue
    print(f"{name:<20} {r['clip']:>9.4f} {r['macro']:>8.4f} {r['d']:>+9.4f} "
          f"{dota_macro(run):>9.4f} {r['micro']:>9.4f}")
print('\nDirection sought:  clip AUC DOWN, macro UP, d UP, DoTA macro UP.')
print('Micro AUC is a clip classifier on this corpus (oracle 0.9226) — do not headline it.')
```

**Before reading any Δ, prove the arms differ only in the flag under test**
(`RESULTS_DADA_PHASE1.md` §3 ran the same check, lesson **C17**):

```bash
%%bash
S=2024
for ARM in t1_dvsignore t1_bottomk t1_both ; do
  echo "=== m0 vs $ARM ==="
  diff "$KATVAD_OUTPUT_ROOT/TAD/$S/m0/stage2/config.yaml" \
       "$KATVAD_OUTPUT_ROOT/TAD/$S/$ARM/stage2/config.yaml"
done
# Expect ONLY dvs_anchor_mode and/or bottomk_weight lines. Anything else -> stop.
```

Standing rules:

- **A small test set has wide intervals.** TAD test is **100 clips**. A Δ of
  ±0.02 in-domain will not clear the noise at n = 3. Say "bounded null below X",
  never "costs nothing".
- **In-domain TAD is not comparable to 89.56** (§0.1). Gate T0 is the only row in
  any table that may sit next to it.
- **Never say "motion-gated"** of M1 or M2. C24 measured the gate as a fixed
  ~50 % shift, and §14.3 re-measures it per run.
- **An A/B run under a known-open precondition defect measures the defect**
  (lesson **C14**). If §13.2a failed, M1 and M3 have no number — report blocked.
  If §6's E-1 fired, nothing in §15 means anything until the corpus is rebuilt.
- **The KIP A/B (M1/M2/M3) is BLOCKED until §15.1 reads** (lesson **C14**).
  `m0` is a clip classifier (clip-level AUC 0.9975, macro 0.6174); an arm stacked
  on it measures that collapse, not the smoother. Do not run §13.1's M2 for the
  `M2 − M0` replication before the T-ladder resolves.
- **Do not tune on any Δ.** The mechanism is what is under test.
- **`d = gap / within-clip σ`, not the raw gap** (lesson **C31**) if you report
  positive/negative score separation at all.

Record outcomes in a new `core/docs/RESULTS_TAD.md`. Do not retro-edit the MSAD
or DADA results documents — each was true for what ran on its corpus; add the
cross-corpus qualifier at the point of next citation. Then add the campaign to
`.project/memory-bank/{activeContext,progress}.md`: `outputs/` is gitignored, so
those files and `RESULTS_TAD.md` are the only durable record.

---

## 16. Order of work

```
--- session and data, no GPU -----------------------------------------------
[ ] §2        session, env vars, §2.3 BRANCH PREFLIGHT (three KeyErrors)
[ ] §3        unzip to /content
[ ] §4        three ingest gates — §4.3 frame order is the silent one
[ ] §5.1      test-split labels, --dry-run first
[ ] §5.2      train + test labels (--with-train-split); RECORD the counts
[ ] §6        EDA label-only run; read E-1 and E-2. STOP if E-1 fires.

--- GPU: caches and the gate -----------------------------------------------
[ ] §7        CLIP features, --ids-file test_ids.txt first, then all 510
[ ] §8        GATE T0 — reproduce the zero-shot number. STOP AND READ IT.
[ ] §6 again  EDA with --clip-dir: feature sections + linear probe
[ ] §9        RAFT flow, train ids only (mandatory for M1/M2/M3 on main)
[ ] §10       DVS KNN cache
[ ] §11       compute E once; write it down

--- the T-ladder: cheapest falsifiable thing on this corpus (NO GPU caches) --
[ ] §13.0     t1_dvsignore, t1_bottomk, t1_both -- seed 2024, 504 steps each
[ ] §14.1a    6 evals (TAD + DoTA per arm)
[ ] §15.1     config diff, then the 4-column cell. STOP AND READ IT.
[ ] branch:   sign pattern present -> seeds 2025 2026; flat -> plan katvad-tad-loss-ladder.md §2

--- the decisive pair (BLOCKED until the ladder reads -- C14) ---------------
[ ] §13.1     M0 and M2, seeds 2024 2025 2026
[ ] §14.1     eval both — TAD + DoTA, 3 seeds each (12 evals, minutes)
[ ] §14.3     shift verification on M2 (span 0-4, s ~ 58-69)
[ ] §14.2/4   gate_a reference, rescore, manifests, score-count assertion
[ ] §15       M2 - M0 on DoTA. STOP AND READ IT against MSAD's +0.1025.

--- the flow objective -----------------------------------------------------
[ ] §13.2     stage 1, 3 seeds
[ ] §13.2a    convergence assert per seed — a STOP, not a warning
[ ] §13.3     M1 and M3, 3 seeds each
[ ] §14       eval, verify, rescore
[ ] §15       M1 - M2 and M3 - M0

--- write-up ---------------------------------------------------------------
[ ] core/docs/RESULTS_TAD.md, then the memory bank
[ ] a lesson candidate per CLAUDE.md §7 if anything surprised you
```

M2 before M1 is deliberate, exactly as on MSAD and DADA. If the smoother
reproduces the gain on a third corpus, M1 answers a different question than you
think you are asking — and you want to know that before spending three stage-1
runs and a full RAFT pass on it. M0 and M2 also need no stage 1, so the pair is
the cheapest decisive thing in the program even though M2 must read the flow
cache.

---

## 17. Pitfalls, mapped to lessons

| Don't | Why | Lesson |
|---|---|---|
| Copy a command out of `TAD_V3_SETUP.md` | `kip.gate_type`, `kip.const_shift_ratio`, `kip.disable_pmg`, `train.checkpoint_every_steps` all raise here | §0.4, §2.3 |
| Put an in-domain TAD number next to 89.56 | 89.56 is zero-shot; in-domain beats it for free | §0.1, **C8b** |
| Say "motion-gated" of M1 or M2 | The gate is untrained and near-constant; it is a fixed ~50 % smoother | **C24**, §14.3 |
| Trust `0.jpg 1.jpg 10.jpg` ordering | Lexicographic ≠ temporal; the whole pipeline is then built on scrambled time, with no error anywhere | §4.3, **C10** |
| Extract before running §4's three gates | An existing directory is not evidence of data | **C10** |
| Train before running §6 | DADA cost a campaign for exactly this; C28 and C27 are corpus properties, not model properties | **C27**, **C28** |
| Unzip to Drive | ~150 k FUSE opens, paid twice; truncated extractions observed on DoTA | §3 |
| `mv "$DIR/prefix/"* "$DIR/"` | Dataset-sized glob overflows `ARG_MAX` and dies *after* extracting | **C20** |
| Center-crop the CLIP cache | Incompatible with every existing artifact, and breaks the appearance/flow FOV match | **C2**, **C13** |
| Change `--stride` after extracting | Invalidates every cache *and every metric measured on one* | **C2** |
| Min-max pool TAD scores | TAD test is ~40 % normal → raw pooling. Pooling follows the label distribution, not the dataset name | **C12** |
| Pass `--set kip.*` at eval to "make sure" | The checkpoint is authoritative; a contradicting flag raises, a matching one is noise | **C34**, §0.4 |
| Skip §9 because M2 zeroes the flow losses | `require_flow = cfg.kip.enabled`; the dataset will not build | §0.4 |
| Pad or patch a label whose span vanished at stride 8 | It is a stride problem. Patching it hides a real sampling defect | §5.1 |
| Vary `num_epochs` between arms | Then the Δ contains a training-length difference | §11 |
| Set `data.is_egocentric=true` | TAD is fixed-camera; that flag is DVS tuning for ego footage | §10 |
| Read anything into TAD's `mul` loss | `C = 2`, so `H_mul` is near-degenerate — close to a second binary head | §12 |
| Report a TAD Δ from one seed | 100 test clips is a small denominator; per-seed spread is not optional | §15 |
| Compute a Δ before asserting score-file counts | One 0-byte `.npz` perturbs every paired Δ, not one number | **C11b** |
| Restart training into an existing `--output-dir` | `metrics.jsonl` appends; MSAD A3 had to be deduped by hand | **C17** |
| Reuse `$KATVAD_OUTPUT_ROOT/DoTA_*` for a TAD-trained arm | Those dirs belong to the MSAD campaign. Two corpora, two trees | **C17** |

---

## 18. Out of scope here

- **The v3 gate matrix** — four selectable `gate_type`s, ECMR, `--dump-kip-diag`,
  the gate-type checkpoint guard: branch `v3`, and
  `core/docs/v3/setup/TAD_V3_SETUP.md`. Running those arms means checking out
  `v3`, and a `main`-vs-`v3` Δ is valid for `kip.enabled=false` only
  (§12.1).
- **P1 / P2 implementation.** Shipped 2026-09-02 (§0.3). Covered by
  `core/tests/test_tad.py` (`TestTrainSplit`, `TestTadTrains`) and
  `core/tests/test_extractors.py::TestRaftFrameExtraction`.
- **Checkpoint selection / validation split.** TAD ships no val split. Every arm
  reads `checkpoint_last`, deep in the overfit regime — a confound shared by all
  arms that cancels in a Δ but caps the absolute numbers.
- **Fixed-length re-sharding.** `windows.json` and `core/data/windows.py` exist
  on this branch but only `core.data.dada` writes them. TAD needs them only if
  §6's E-1 fires; sizing one is lesson **C32**'s territory (measure the length
  distribution **per class** first).
- **Alert-CLIP.** No public checkpoint exists. Every number here is stock
  CLIP ViT-B/16 at the pinned revision (**C4**).
- **PreVAD.** `core/docs/PREVAD_SETUP.md`. KIP-off trunk + Gate P0 only; it ships
  no pixels, so no KIP-on arm exists there, ever.
