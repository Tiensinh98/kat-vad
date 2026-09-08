# DADA-2000 — dataset setup on Colab (unzip → labels → CLIP → RAFT → KNN)

**Written 2026-09-03, against branch `v3`.** Dataset-level and version-agnostic:
everything here is about getting `data/DADA2000/` and the caches onto disk
correctly. The *experiment* on top of it — arms, seeds, ablations — lives in
`core/docs/v3/setup/DATA_V3_SETUP.md`.

Companion docs: `DATA_LAYOUT.md` (the on-disk contract this obeys),
`TAD_SETUP.md` (the closest sibling — another frame-folder dataset with a
directory-derived label), `DOTA_EVAL.md` §3.1 (the unzip/coverage playbook
this borrows), `COLAB.md` (session mechanics, checkpoint recovery).

Code: `core/data/dada.py` (`core/tests/test_dada.py` covers it, including
training under every KIP gate type).

---

## 0. Read this before you plan anything on DADA-2000

### 0.1 What DADA-2000 is *for* in this project

Spec §7.5 (`KAT_VAD_IMPLEMENTATION_SPEC.md`) originally scoped DADA-2000 as
**eval-only** — a second ego-centric zero-shot benchmark alongside DoTA, the
same role TAD started in. Unlike TAD and DoTA, though, DADA-2000 ships
**frame-level accident windows for (almost) every abnormal clip**, not just a
held-out test slice, so `core/data/dada.py` builds a real train/test split
with genuine frame-level test labels instead of a directory-only weak train
(TAD's `--with-train-split`) or an eval-only annotation (DoTA). That means
DADA-2000 can be:

1. **A third training corpus**, for the same replication purpose TAD serves
   (`TAD_SETUP.md` §0.2) — does the plain-TSM-smoother finding
   (`core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md`) hold on a corpus that is
   not MSAD *and* not TAD?
2. **Its own in-domain benchmark** (the held-out DADA test split).
3. **A zero-shot transfer target** for arms trained elsewhere (MSAD, TAD,
   PreVAD), exactly as spec §7.5 originally intended.

**There is no published DADA-2000 number pinned in this project**
(`core/constants.py` has `TAD_ZERO_SHOT_AUC` but nothing DADA-shaped) — so
unlike TAD's Gate T0, §6 below is a **sanity check**, not a reproduction gate.
Do not invent a target number to compare against; the released checkpoint's
own DADA-2000 score, once measured, becomes the reference for later arms
(lesson **8b**'s discipline: gate against what the checkpoint actually scores).

### 0.2 The three record kinds `core/data/dada.py` builds

| Kind | Source | Window known? | Split |
|---|---|---|---|
| **Annotated abnormal** | a `Cleaned_Metadata.csv` row, resolved against its frame folder | yes | train **or** test (seeded split) |
| **Weak abnormal** | a frame folder under `0_Non_Ego_Fault/` or `1_Ego_Fault/` with **no** CSV row | no | **always train** (like a windowless clip everywhere else in this project, it can never be a test candidate) |
| **Normal** | every frame folder under `0_Normal_Driving/` (carries no CSV rows at all) | n/a (all-zero) | train **or** test |

Weak supervision is preserved throughout: a **train**-split record's known
window (if it has one) is written to `meta.json` only, never into
`labels_train.json` — the same rule MSAD, DoTA and TAD all follow.

### 0.3 Do not run TAD's `--with-train-split` mental model here

TAD's train split is *entirely* unwindowed by construction — the annotation
covers only the 100-video test protocol. DADA-2000 is the opposite: most
abnormal clips carry a window, and `dada.py` **always** builds both splits —
there is no eval-only mode and no flag to request one. If you only want an
eval split (e.g. to score an externally-trained checkpoint zero-shot), pass
`--test-ratio-abnormal 1.0 --test-ratio-normal 1.0` so every candidate lands
in test — see §5.

---

## 1. What ships, and where it lands

Your archive, `$KATVAD_DATA_ROOT/DADA2000/archive.zip`:

```
Cleaned_Metadata.csv
0_Non_Ego_Fault/
  type10_vid001/     <- a DIRECTORY, images directly inside
    *.png
  ...
0_Normal_Driving/
  type10_vid001/
    *.png
  ...
1_Ego_Fault/
  type10_vid023/
    *.png
  ...
```

Three properties the code already handles, and one that needs checking:

- **`list_frame_folders(root, subdir=None)`** accepts any directory at any
  depth that directly holds images (`core/data/video_io.py`), so the three
  category directories are transparent — the same property TAD's
  `abnormal/`/`normal/` split relies on. Pass `--frames-dir .../DADA2000`
  (the root that directly contains the three category dirs), with **no**
  `--frames-subdir`.
- **Folder-name padding is not assumed.** `core/data/dada.py` parses
  `type<N>_vid<N>` with a regex and joins the CSV's `type`/`video` columns by
  **integer value**, not by reconstructing a zero-padded string — so it does
  not matter whether the archive pads `vid1` to `vid001` or not, as long as
  every folder matches `^type\d+_vid\d+$` exactly. A folder that does not
  raises immediately, naming the offender.
- **Folder names are NOT globally unique across the three directories** —
  confirmed on the real archive, not a hypothetical: `type10_vid001` exists
  under both `0_Non_Ego_Fault` and `0_Normal_Driving` (the accident-type
  taxonomy is shared across fault directories, and `0_Normal_Driving` reuses
  the same `type`/`vid` bucketing convention). `dada.py` handles this by
  construction, not by asserting it away: every `video_id` is
  `{fault_dirname}__{folder_name}` (e.g. `0_Non_Ego_Fault__type10_vid001`),
  globally unique regardless of how many bare-name collisions the archive has.
  Pass `--flat-frames-dir` (§5) so the extraction tools, which key their own
  folder scan on the bare on-disk name, can still resolve every id.
- ⚠️ **Frame ordering is unverified for this archive.** If the PNGs are not
  zero-padded (`0.png 1.png 10.png 2.png` sorts lexicographically, which is
  *not* temporal order), every downstream artifact is built on scrambled
  time — exactly the TAD pitfall (`TAD_SETUP.md` §4.3). Check §4.3 below
  before extracting anything.

Target layout (`DATA_LAYOUT.md`), where `{id}` is
`{fault_dirname}__{type<N>_vid<N>}` (§1 — the on-disk folder names alone are
not globally unique, so every artifact below is keyed by the disambiguated
form):

```
$KATVAD_DATA_ROOT/DADA2000/{0_Non_Ego_Fault,0_Normal_Driving,1_Ego_Fault}/type<N>_vid<N>/*.png
$KATVAD_DATA_ROOT/DADA2000/Cleaned_Metadata.csv
$KATVAD_DATA_ROOT/DADA2000/labels_train.json         {id: 0|1}
$KATVAD_DATA_ROOT/DADA2000/frame_labels_test.json    {id: [0,1,...]} per SAMPLED frame
$KATVAD_DATA_ROOT/DADA2000/defs.json                 ["Normal", "CarAccident"]
$KATVAD_DATA_ROOT/DADA2000/meta.json                 diagnostics only
$KATVAD_DATA_ROOT/DADA2000/train_ids.txt
$KATVAD_DATA_ROOT/DADA2000/test_ids.txt
/content/dada_flat/{id} -> symlink to the real type<N>_vid<N> folder (§5)
$KATVAD_CACHE_ROOT/clip/DADA2000/{id}.npy        (L,512) no_center_crop
$KATVAD_CACHE_ROOT/flow/v1/DADA2000/{id}.npy         (L,256) e_O, train ids only
$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz    DVS filler cache
```

Every artifact in this project since 2026-08-12 uses `no_center_crop`
(lessons **C2**, **C13**) — DADA-2000 is a new dataset, so there is no
center-crop legacy to preserve; it was extracted `--no-center-crop` from the
start (the cache dir name does **not** say so — see the marker below), no
alternative path.

### 1.1 `Cleaned_Metadata.csv` columns actually used

```
video, type, whether an accident occurred (1/0),
abnormal start frame, accident frame, abnormal end frame, total frames,
Fault_Label
```

`Fault_Label` ∈ `{0_Non_Ego_Fault, 1_Ego_Fault}` — every row in the shipped
file is an accident row (`whether an accident occurred (1/0)` is always
`1`); rows that are not are skipped with a warning, defensively, in case a
future export mixes in normal rows. `0_Normal_Driving` carries **no CSV rows
at all** — the directory is the whole label, same discipline as TAD's train
split (`TAD_SETUP.md` §5.2). The weather/light/scene/linear/texts/causes/
measures columns are not read by `dada.py`.

`(video, type)` joins to the on-disk folder `type{type}_vid{video}` **within
the directory named by that row's `Fault_Label`** — never searched across
directories, so an accidental cross-category name collision (§1) does not
silently misjoin a CSV row to the wrong folder.

---

## 2. Colab session setup

Identical to `TAD_SETUP.md` §2 — same install cell, same environment
variables, same restart discipline. Not repeated here.

```bash
!python -c "from core.kip import gate_shift; print(gate_shift.GATE_TYPES)"
# expect: ('rank', 'mlp_frozen', 'mlp_ste', 'constant')
```

---

## 3. Unzip

**Unzip to `/content` (VM-local NVMe), not to Drive** — same reasoning as
`TAD_SETUP.md` §3: the frames are an intermediate, `clip/DADA2000/*.npy`
and `flow/v1/DADA2000/*.npy` are what is worth persisting, and Drive's FUSE
mount pays a per-file-open cost you would otherwise pay twice.

```bash
%%bash
set -e
mkdir -p /content/dada
df -h /content | tail -1
unzip -q "$KATVAD_DATA_ROOT/DADA2000/archive.zip" -d /content/dada
ls /content/dada           # expect: Cleaned_Metadata.csv 0_Non_Ego_Fault 0_Normal_Driving 1_Ego_Fault
```

If the archive nests one level deeper, move the **directory**, never a glob
of its contents (`TAD_SETUP.md` §3, lesson **20** — a dataset-sized glob
overflows `ARG_MAX` and dies *after* a successful extraction).

---

## 4. Ingest gates — run all before extracting anything

An existing directory is not evidence of data (lesson **10**).

### 4.1 Coverage

```python
import os
ROOT = '/content/dada'
for d in ('0_Non_Ego_Fault', '0_Normal_Driving', '1_Ego_Fault'):
    folders = sorted(os.listdir(f'{ROOT}/{d}')) if os.path.isdir(f'{ROOT}/{d}') else []
    empty = [f for f in folders if not os.listdir(f'{ROOT}/{d}/{f}')]
    print(f'{d:18s} folders={len(folders):5d}  empty={len(empty):3d}  {empty[:3]}')
    assert not empty, f'{d}: empty folders = truncated unzip; re-run §3'
```

### 4.2 Every CSV row resolves to a folder

```python
import csv, os
ROOT = '/content/dada'
have = {}
for d in ('0_Non_Ego_Fault', '1_Ego_Fault'):
    have[d] = set(os.listdir(f'{ROOT}/{d}'))

import re
def type_vid(name):
    m = re.match(r'^type(\d+)_vid(\d+)$', name)
    return (int(m.group(1)), int(m.group(2))) if m else None

on_disk = {d: {type_vid(n) for n in have[d] if type_vid(n)} for d in have}
missing = []
with open(f'{ROOT}/Cleaned_Metadata.csv', newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
        if row['whether an accident occurred (1/0)'].strip() != '1':
            continue
        key = (int(row['type']), int(row['video']))
        d = row['Fault_Label'].strip()
        if key not in on_disk.get(d, set()):
            missing.append((d, key))
print(f'{len(missing)} CSV rows have no matching folder', missing[:5])
assert not missing, 'core.data.dada will raise on these; fix the unzip first (or pass --allow-missing-frames)'
```

`core/data/dada.py:resolve_annotated_records` raises on exactly this by
default. Better to see it here than three cells later.

### 4.3 ⚠️ Frame ordering — check before extracting anything

Same pitfall as TAD (`TAD_SETUP.md` §4.3): `list_frame_images` sorts by
filename, and unpadded numeric names sort wrong.

```python
import os, re
d = '/content/dada/0_Non_Ego_Fault'
sample = sorted(os.listdir(d))[0]
names = sorted(os.listdir(f'{d}/{sample}'))
print(sample, len(names), names[:5], '...', names[-3:])
lex = list(names)
num = sorted(names, key=lambda n: int(re.sub(r'\D', '', n) or 0))
print('lexicographic == numeric order:', lex == num)
```

**If that prints `False`, stop and zero-pad in place first** — the exact
renamer in `TAD_SETUP.md` §4.3 works here unchanged (just point `find` at
`/content/dada` instead of `/content/tad/frames`). Re-run §4.1 after any
rename: the file count must be unchanged.

---

## 5. Build the label files

`core/data/dada.py` always builds both splits — there is no eval-only mode.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.data.dada \
  --metadata   /content/dada/Cleaned_Metadata.csv \
  --frames-dir /content/dada \
  --out-dir    "$KATVAD_DATA_ROOT/DADA2000" \
  --flat-frames-dir /content/dada_flat \
  --stride 8 --seed 2024 \
  --test-ratio-abnormal 0.2 --test-ratio-normal 0.2 \
  --dry-run
```

`--dry-run` parses, resolves against disk and splits without writing (the
flat dir is not materialized either). Read the log line before dropping the
flag:

```
N abnormal frame folders have no CSV row -- weak-labeled, forced to train
Split: train=... (... abnormal) test=... (... abnormal)
DADA-2000: ... train (... abnormal: ... annotated + ... weak / ... normal), ... test (... abnormal / ... normal)
```

Then re-run without `--dry-run`. Writes `labels_train.json`,
`frame_labels_test.json`, `defs.json` (`["Normal", "CarAccident"]`),
`meta.json`, `train_ids.txt`, `test_ids.txt`, and **`--flat-frames-dir
/content/dada_flat`** — one symlink per record, named by the disambiguated
`{fault_dirname}__{type<N>_vid<N>}` id (§1). Point §7/§8's `--frames-dir` at
this flat dir, not `/content/dada` directly: `extract_clip_features.py` and
`raft_extract.py` key their own folder scan on the bare on-disk name, which
collides across categories exactly like §1 describes, and neither tool
asserts against it — an unprefixed `--frames-dir` there would silently
extract one clip's frames under the wrong id. Kept on `/content` (not Drive),
same reasoning as the frames themselves (§3): it is symlinks, not data, but
still one FUSE-mount round trip per file. Resumable: re-running only adds
symlinks for records that do not already have one.

> **If it raises `N abnormal test clips lose their anomaly window at stride
> 8`** (only reachable with `--strict`; the default warns and keeps going):
> a span rounds away to nothing at this stride. This is a stride problem —
> lower `--stride`, and understand doing so invalidates any cache already
> built at 8 (lesson **C2**). Without `--strict` these clips silently
> contribute only negative frames, matching DoTA's default behaviour
> (`core/data/dota.py:build_frame_labels`).

**The split is a project choice, not an official protocol.** DADA-2000 ships
no train/test split of its own for this weakly-supervised setting; `dada.py`
does a seeded, stratified (by `is_abnormal, Fault_Label`) split at
`--test-ratio-abnormal 0.2 --test-ratio-normal 0.2` by default, overridable
per run and fully overridable via `--split-file` (test ids, one per line) for
reproducibility across a seed sweep — pin it once and reuse it, the same way
`MSAD_ncc`'s `split_file` is pinned across seeds.

**A sibling project's precedent (`simple-tad`), and why it doesn't transfer
directly.** The original LOTVS-DADA release does ship an official train
assignment for its own risk-anticipation task. `simple-tad` (a sibling repo in
this workspace, not part of `core/`) reads it as-is:
`data_tools/dada/prepare_anno_dada2000.py` takes a file it calls
`orig_training.txt` (one accident clip per line, `type/video` directory plus
label/start/end/toa columns, from the *original* per-category layout
`frames/{type}/{video}`, zero-padded) as the literal train membership list, and
every remaining `type/video` pair in that release's `annotation/full_anno.csv`
(which — unlike this project's `Cleaned_Metadata.csv`, §1.1 — carries a row for
every clip, accident and normal alike) becomes validation. It is a **lookup
against the upstream benchmark's own split, not a split `simple-tad`
computes** (`data_tools/dada/halfsplit.py` is a separate, unrelated thing: a
50%-per-category random subsample of that official train set, used only to
cheapen finetuning — not a second split protocol).

That precedent does not carry over to this archive mechanically: `dada.py`
works from `Cleaned_Metadata.csv` plus the fault-attribution repackaging
(`0_Non_Ego_Fault` / `1_Ego_Fault` / `0_Normal_Driving`, §1), not the original
`frames/{type}/{video}` layout `orig_training.txt` indexes into, and whether a
given `(type, video)` pair keeps the same identity across that repackaging is
unverified — confirm it empirically (e.g. cross-check a handful of `(type,
video)` pairs' accident window against `Cleaned_Metadata.csv` before trusting
the mapping) rather than assuming it. If it holds and you want this project's
split to match the official one for comparability: obtain `orig_training.txt`
from the original LOTVS-DADA release, convert every `type/video` pair **not**
listed in it to this project's disambiguated id
(`{Fault_Label}__type{type}_vid{video}`, reading each pair's `Fault_Label` off
`Cleaned_Metadata.csv`), and pass the resulting file to `--split-file`.

Verify, and record the counts — §10 needs them:

```python
import json, os
D = f"{os.environ['KATVAD_DATA_ROOT']}/DADA2000"
tr = json.load(open(f'{D}/labels_train.json'))
te = json.load(open(f'{D}/frame_labels_test.json'))
A, N = sum(tr.values()), len(tr) - sum(tr.values())
print(f'train {len(tr)} ({A} abnormal / {N} normal)')
print(f'test  {len(te)} ({sum(1 for v in te.values() if any(v))} with a positive frame)')
assert A and N, 'DVSFeatureDataset needs both classes'
assert not (set(tr) & set(te)), 'LEAK: an id is in both splits'
frac_normal = sum(1 for v in te.values() if not any(v)) / len(te)
print(f'test normal fraction {frac_normal:.3f}  -> score-norm auto resolves to '
      f'{"minmax" if frac_normal < 0.05 else "RAW"}')
```

At the default 0.2/0.2 ratios the test split keeps roughly a fifth of the
normal videos, well above the 5 % `SCORE_NORM_AUTO_NORMAL_FRACTION`
threshold — so `--score-norm auto` should resolve to **raw** pooling on
DADA-2000's own test split (the MSAD/TAD protocol), never DoTA's per-clip
min-max. Confirm it with the printed fraction rather than assuming it —
pooling follows the label distribution, never the dataset name (lesson
**12**).

### 5.1 Audit the clips whose anomaly window vanished — every build

`--strict` **raises**; it is a gate, not a repair (`core/data/dada.py:417-432`).
Without it the build logs a WARNING and the affected clips enter the **test**
split carrying an all-zero label vector — they do not merely "contribute only
negative frames", they become indistinguishable from genuine `0_Normal_Driving`
clips. On a test set where 74 % of frames already come from all-normal clips,
that inflates both the micro AUC and the constant-score clip oracle
(`RESULTS_DADA.md` §4). **The 2026-09-06 build has 4 of them and they are still
in the published numbers.**

The train split is unaffected — `build_frame_labels` is called on the test
records only (`core/data/dada.py:557`) — so this is fixable **without
retraining anything**. Run the audit after every label build:

```python
import json, os
D = f"{os.environ['KATVAD_DATA_ROOT']}/DADA2000"
te = json.load(open(f'{D}/frame_labels_test.json'))
meta = json.load(open(f'{D}/meta.json'))
abnormal = {r['video_id'] for r in meta.get('records', []) if r.get('is_abnormal')} \
           if 'records' in meta else set()
vanished = sorted(v for v in te if v in abnormal and not any(te[v]))
print(f'{len(vanished)} abnormal test clips have an all-zero label vector')
for v in vanished:
    print('  ', v)
open(f'{D}/vanished_ids.txt', 'w').write('\n'.join(vanished) + '\n')
```

If `meta.json` carries no per-record flags, re-run `core.data.dada` with
`--strict` and read the ids straight out of the `ValueError` message — that is
the flag's only safe use.

**What to do with them.** Not `--stride 4`: that fires lesson **C2** and
invalidates every cached feature and every metric measured on it. Exclude them
at scoring time instead, and say so next to the number:

```bash
%%bash
# after any eval, before quoting a DADA micro AUC
for f in $(cat "$KATVAD_DATA_ROOT/DADA2000/vanished_ids.txt"); do
  echo "excluded from DADA metrics (window lost at stride 8): $f"
done
```

Then recompute with `core.tools.rescore` over the eval dir with those ids held
out, and report the clip-level oracle beside the micro number (lesson **12**).

**Done for the 2026-09-06 campaign (Phase 0.2, 2026-09-08).** The corrected
379-clip numbers for all seven arms are in `RESULTS_DADA.md` §3.1a — computed
offline from the saved `.npz` files, no retraining and no re-eval. Summary:
micro moves **+0.001 to +0.002** per arm; `auc_macro` is **unchanged** on every
arm (the four clips are single-class, so `macro_video_auc` already skipped
them); the clip oracle moves 0.9086 → **0.9082** and the length-only baseline
0.8654 → **0.8681**. Because the baseline rises more than any arm does, the best
arm's margin over a ruler *shrinks* to **+0.0075**. Quote the 379-clip column
and say which split a number came from.

**And print the length-only baseline too** (lesson **C28**). This corpus leaks
its label through clip length: abnormal clips top out at **17** stride-8 frames
while **107** normal clips are longer than any abnormal clip. `core.tools.eda`
computes it in §3.3 of its report:

```bash
%%bash
python -m core.tools.eda report --dataset DADA2000 \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" \
  --output-dir "$KATVAD_OUTPUT_ROOT/eda/DADA2000" --sections corpus,labels,protocol
# read §0 verdicts and §3.3; an arm that does not beat §3.3's micro AUC is unmeasured
```

---

## 6. Sanity check — a released checkpoint on DADA-2000

There is no published number to reproduce here (§0.1) — this step exists to
catch an ingest defect (frame order, label arithmetic, stride) before
spending GPU hours on training, exactly the role Gate T0 plays for TAD, minus
the reference number.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.evaluate \
  --baseline-ckpt "$KATVAD_CKPT_ROOT/best.ckpt" \
  --set kip.enabled=false --set data.dataset=DADA2000 \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --score-norm auto \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/gate_d0" --save-scores
```

Needs §7's CLIP cache first. **How to read it:** micro AUC clearly above
chance (~0.5) on a dashcam-accident benchmark this checkpoint never saw is
evidence the ingest is sound; a result at or near chance means check §4.3
(frame order) first, then the label arithmetic, before touching the model.
Whatever this scores becomes the reference for later DADA-2000 arms — record
it with the date; there is no printed paper number to fall back on.

```bash
%%bash
python -m core.tools.rescore --run-dir "$KATVAD_OUTPUT_ROOT/DADA2000/gate_d0"
```

---

## 7. CLIP features — `no_center_crop`, stride 8

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.tools.extract_clip_features \
  --frames-dir /content/dada_flat \
  --dataset DADA2000 --stride 8 --batch-size 64 --device cuda \
  --no-center-crop \
  --output-dir "$KATVAD_CACHE_ROOT/clip/DADA2000"
```

- **`--frames-dir` is the flat symlink dir from §5, not `/content/dada`.**
  `extract_frame_directory` keys its folder scan on the bare on-disk name
  (`core/tools/extract_clip_features.py:194`), which collides across DADA-2000's
  fault-attribution directories (§1) — pointed at `/content/dada` directly it
  would silently extract some clips under the wrong id, no error. The flat dir
  is already named by the disambiguated id, so this needs no other flag.
- **No `--frames-subdir`** — the PNGs sit directly in each clip folder.
- Confirm the log reads **`Transform: anisotropic resize`**.
- **The output dir is `clip/DADA2000`, not `clip/DADA2000_ncc`.** That is what
  `collab/DADA/v3/train.py` built and what every 2026-09-06 artifact reads;
  renaming it now orphans them. The cost is that the path no longer records its
  transform the way `MSAD_ncc` / `DoTA_s8_ncc` do, and
  `extract_clip_features.py` writes no manifest — pure lesson **C2** exposure.
  Drop a marker in instead, once, right after extraction:

  ```bash
  %%bash
  printf 'transform=anisotropic_resize (--no-center-crop)\nstride=8\nbuilt=2026-09\n' \
    > "$KATVAD_CACHE_ROOT/clip/DADA2000/TRANSFORM.txt"
  ```

  Read it before **any** re-extraction into this directory. A center-cropped
  rebuild here silently voids every number in `core/docs/v3/RESULTS_DADA.md`.
- Resumable per video with atomic `.part` writes (lesson **11**); re-running
  does only what is missing and logs `Resume: done/total`.
- To extract the test split first (for §6, before committing GPU hours to the
  full corpus): add `--ids-file "$KATVAD_DATA_ROOT/DADA2000/test_ids.txt"`.

Coverage check:

```python
import os, json, numpy as np
C = f"{os.environ['KATVAD_CACHE_ROOT']}/clip/DADA2000"
D = f"{os.environ['KATVAD_DATA_ROOT']}/DADA2000"
have = {p[:-4] for p in os.listdir(C) if p.endswith('.npy')}
te = json.load(open(f'{D}/frame_labels_test.json'))
print(f'{len(have)} feature files')
bad = [(v, len(te[v]), len(np.load(f'{C}/{v}.npy'))) for v in te
       if v in have and len(np.load(f'{C}/{v}.npy')) != len(te[v])]
print('length mismatches vs frame labels:', len(bad), bad[:3])
assert not bad, 'feature rows must line up with label rows, one per sampled frame'
```

---

## 8. RAFT flow targets — train ids only

`e_O` is a train-time target for `L_KIP_rec`/`L_KIP_align`, never on the
inference path. Extract for `train_ids.txt` only.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.flow.raft_extract \
  --frames-dir /content/dada_flat \
  --ids-file  "$KATVAD_DATA_ROOT/DADA2000/train_ids.txt" \
  --dataset DADA2000 \
  --cache-root "$KATVAD_CACHE_ROOT/flow/v1" \
  --stride 8 --batch-size 8 --device cuda
```

Same reason as §7 — `raft_extract.py:291` has the identical bare-name folder
scan, so `--frames-dir` must be the flat dir, not `/content/dada`.

Writes `flow/v1/DADA2000/{id}.npy` `(L,256)`, `{id}.stats.npy` `(L,23)`, and
`flow/v1/flow_projection.npz` if it does not exist yet (reused as-is from the
MSAD/TAD campaigns — never rebuild it per dataset, that would make the flow
caches incomparable across corpora).

- Frame-source and video-source flow are bit-identical
  (`core/tests/test_extractors.py::TestRaftFrameExtraction`), so this cache
  is directly comparable to MSAD's despite DADA-2000 shipping frames.
- Same stride as the CLIP cache — the dataset asserts
  `len(flow) == len(features)` per video and raises otherwise.
- **23 effective dimensions**, same as every other flow cache in this
  project — never claim spatial localization from it.

---

## 9. DVS KNN filler cache

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
python -m core.data.knn_cache \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" --dataset DADA2000 \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --output   "$KATVAD_CACHE_ROOT/knn/DADA2000/knn_cache.npz" \
  --motion-key
```

**Pass `--motion-key`**: DADA-2000 is ego-centric dashcam footage, the same
category as DoTA (spec §6.1 — the KNN key gets the coarse ego-motion
descriptor appended). Keep `data.is_egocentric=true` in every DADA-2000
training config for the same reason (selects `theta_ego`/`delta_m_ego` in
DVS) — this is the opposite default from TAD, which is fixed-camera.

---

## 10. Sizing a DADA-2000 run

`DVSFeatureDataset.__len__` is `2 x num_abnormal_train` — compute it, do not
copy another campaign's `num_epochs`.

```python
import json, math, os
tr = json.load(open(f"{os.environ['KATVAD_DATA_ROOT']}/DADA2000/labels_train.json"))
A = sum(tr.values())
BATCH, TARGET_STEPS = 64, 500
steps_per_epoch = math.ceil(2 * A / BATCH)
print(f'{A} abnormal train videos -> dataset len {2*A} -> {steps_per_epoch} steps/epoch')
print(f'num_epochs for ~{TARGET_STEPS} steps: {math.ceil(TARGET_STEPS / steps_per_epoch)}')
```

Use that `num_epochs` for both stages and every arm (`TAD_SETUP.md` §10's
reasoning applies unchanged). Warm-up stays at 20 steps, peak LR 5e-5.

**On the 2024 corpus build this printed 25 steps/epoch, so `E = 20`** — 500
optimizer steps, matching MSAD's 4 x 125. If it prints anything else, the split
changed and every number in `RESULTS_DADA.md` is about a different corpus.

### 10.1 Launch — the two arms this document is responsible for

**Ownership: `core/docs/v3/setup/DADA_V3_SETUP.md` §3 and §4 are the source of
truth for all six arms.** The two below are duplicated here because they are the
arms that *validate the data build* — no flow cache, no stage 1, so they run the
moment §5–§9 finish. If the two documents ever disagree, the v3 runbook wins.

Two seeds, one rule:

```
S=2024        # train seed. Change THIS for 2025 / 2026.
E=20          # never varies, across arms or stages
```

> **`core.data.dada --seed 2024` in §5 is a different seed and must not move.**
> That one draws the train/test split. Re-running the label build at `--seed
> 2025` re-draws the corpus, and every cross-seed Δ then compares two different
> datasets. Build the labels once; sweep `train.seed` only.

**A2 — plain-TSM control.** All six flags required; `disable_pmg=true` alone
does not remove the pathway, the three loss flags do (lesson **14**).

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

**A0 — KIP-off baseline.** The arm every Δ subtracts from.

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

**Both evals, either arm** — swap `CKPT`/`GATE` per the table below:

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024
ARM=constant                                    # or: kipoff
SUB=stage2_kip_on                               # kipoff uses: stage2
CKPT="$KATVAD_OUTPUT_ROOT/DADA2000/${ARM}_s$S/$SUB/checkpoint_last.pt"
GATE="--set kip.gate_type=constant --set kip.const_shift_ratio=0.5"
# kipoff instead:  GATE="--set kip.enabled=false"

python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DADA2000 \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/${ARM}_s$S/eval_dada"

python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DADA2000/${ARM}_s$S/eval_dota"
```

The **gate** flags repeat at eval; the **loss** flags and `data.is_egocentric`
must not — they are train-time only (`core/train.py:593`, `:600`) and change
neither the eval graph nor the state dict.

**The four gated arms — A1, A2b, A3, A4 — need §8's flow cache and a stage-1
run first. Their full commands are in `DADA_V3_SETUP.md` §3.2–§3.3, one
copy-paste block each.** Do not reconstruct them from this page.

| Arm | Gate flags | Stage 1 from | Full command |
|---|---|---|---|
| **A1** `rank` | `--set kip.gate_type=rank` | `rank_s$S/stage1` | `DADA_V3_SETUP.md` §3.3-A1 |
| **A2b** `constant`+PMG | `--set kip.gate_type=constant --set kip.const_shift_ratio=0.5` | `rank_s$S/stage1` | §3.3-A2b |
| **A3** `mlp_frozen` | `--set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm` | `mlp_frozen_s$S/stage1` | §3.3-A3 |
| **A4** `mlp_ste` | `--set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm` | `mlp_frozen_s$S/stage1` | §3.3-A4 |

---

## 11. Pitfalls, mapped to lessons

| Don't | Why | Lesson |
|---|---|---|
| Compare a DADA-2000 number to a printed paper AUC | No published number is pinned in this project for this protocol (§0.1) | §0.1 |
| Trust unpadded frame-name ordering | Lexicographic ≠ temporal; scrambles every downstream artifact silently | §4.3, **10** |
| Extract before running §4's gates | An existing directory is not evidence of data | **10** |
| Unzip to Drive | Per-file FUSE-open cost paid twice | §3 |
| `mv "$DIR/prefix/"* "$DIR/"` | Dataset-sized glob overflows `ARG_MAX`, dies mid-extraction | **20** |
| Center-crop the CLIP cache | Inconsistent with every other artifact in this project | **C2**, **C13** |
| Change `--stride` after extracting | Invalidates the cache and every metric measured on it | **C2** |
| Leave `--motion-key` off | DADA-2000 is ego-centric; the DVS KNN key should carry motion | §9, spec §6.1 |
| Set `data.is_egocentric=false` | DADA-2000 is dashcam footage, same category as DoTA | §9 |
| Change `core.data.dada --seed` when sweeping train seeds | That is the **split** seed; a new split re-draws the corpus and voids every cross-seed Δ | §10.1 |
| Leave the vanished-window clips in a quoted DADA micro AUC | They enter the test set as all-zero labels and read as genuine normal clips | §5.1, **12** |
| Re-extract into `clip/DADA2000` without reading `TRANSFORM.txt` | The dir name does not record `--no-center-crop`; a cropped rebuild voids `RESULTS_DADA.md` silently | §7, **C2** |
| Assume `clip/DADA2000_ncc` or `knn/DADA2000_ncc` exists | Earlier drafts of this doc prescribed those names; nothing was ever built there | §7, §9 |
| Reconstruct `type{T}_vid{V}` with an assumed pad width | Folder padding is not verified for this archive; `dada.py` joins by parsed int instead | §1 |
| Assume the score `.npz` set is complete | `evaluate --save-scores` is not atomic (lesson **11b**) | **11b** |
| Point §7/§8's `--frames-dir` at `/content/dada` instead of the flat dir | `type<N>_vid<N>` repeats across fault directories; the extractors key on bare folder name and would silently mis-extract, no error | §1, §5 |

---

## 12. Out of scope here

- **The experiment itself** — arms, seeds, gate types, ablations, decision
  rules: `core/docs/v3/setup/DATA_V3_SETUP.md`.
- **The preprocessor's implementation.** `core/data/dada.py`, covered by
  `core/tests/test_dada.py` (record resolution, the split, every failure
  mode, and training under every KIP gate).
- **Checkpoint selection / validation split.** DADA-2000 ships no official
  val split here either; every arm reads `checkpoint_last`.
- **Alert-CLIP.** No public checkpoint exists. Every number here is stock
  CLIP ViT-B/16.
