# PreVAD — download, setup, and training runbook

**Written:** 2026-08-23 · **Revised:** 2026-08-25 (§4 restructured: Drive holds
the zip and the whole `$KATVAD_DATA_ROOT/PreVAD` folder durably, `/content`
holds the extracted features per session; new §4.1 covers both roots and §4.1.1
the CSV staging step; §1.1 settles which files from the clone are actually
needed) · **Revised:** 2026-08-24 (trunk-transfer plan, §7; code prerequisites
G1–G3 implemented, §3)
**Status:** **executing.** No longer a proposal. As of 2026-08-25 the setup path
(§4.1.1 → §4.3 → §4.4 → §5 → §6) has run on Colab, and **Stage P (§7.1) is
mid-flight at epoch 26 of 40** — `global_step` 8,835, `checkpoint_last.pt`
459 MB on Drive. Everything from §7.2 (Stage T) onward is still unexecuted.

> **Gate P0 (§4.5) has no recorded result.** Its first attempt died on the
> environment (§4.0.1), and Stage P was started without a re-run being logged
> here. P0 was defined as the gate that blocks steps 9–13; the trunk is being
> trained ahead of it. Run it and record the number — if the features turn out
> incompatible, 13,360 batches were spent proving nothing. See §10.
**Inputs:** the `PreVAD/` clone at repo root, `LaGoVAD-PreVAD/` (read-only),
`core/` at `7da7b06`, and the `RESULTS_*.md` series.

PreVAD is LaGoVAD's *training* set (35,279 videos). Training on it is what turns
our port from "an MSAD-trained model" into "an open-world model comparable to
the paper's zero-shot table". This document covers what to download, what our
code is missing, and in which order to run it.

---

## 0. Verdict up front

**The released CLIP features are probably reusable, and there is a cheap decisive
test for it. They only get you a KIP-off run — which is fine, because KIP-off on
PreVAD is exactly the trunk this plan needs.**

| Question | Answer |
|---|---|
| Are the released `ViT-B-16-8p` features compatible with our `*_ncc` pipeline? | **Almost certainly yes** — same encoder, same transform, same interval. Three small deviations (§2). Verify with **Gate P0** (§4) before spending a GPU-hour. |
| Can we train KIP-**off** (the paper-comparable baseline) from the release alone? | **Yes.** `require_flow = cfg.kip.enabled` (`core/train.py:593`), so a KIP-off run never touches the flow cache. |
| Can we train KIP-**on** *on PreVAD*? | **No, and we are not going to try.** KIP needs per-frame RAFT targets `e_O`, which need pixels; PreVAD ships none, and re-downloading buys a 50–70 % subset whose clips cannot be paired with the released features anyway (§7.4). |
| Then what is the point? | **The trunk.** Flow is only needed where KIP is *attached*, not where the trunk is *pretrained*. Pretrain KIP-off on PreVAD, graft KIP onto that trunk, and run the A/B on MSAD where flow already exists. That measures KIP on top of a properly pretrained baseline — the comparison the current +0.09 is missing. **This is the plan: §7.** |
| Does the code run on PreVAD today? | **No.** Four concrete gaps, all in preprocessing/definitions, none in the model. See §3. |

```
PreVAD (released features, KIP-off)  ──►  trunk
                                            ├── MSAD stage 2, KIP-on   ─┐
                                            └── MSAD stage 2, KIP-off  ─┴─►  evaluate DoTA
```

Read §3 before planning any schedule: the missing pieces are half a day of code,
not a download.

---

## 1. What the release actually contains

`PreVAD/` at repo root is a clone of `https://huggingface.co/datasets/Kamino123/PreVAD`
with the large files still as **git-lfs pointers** (`git lfs` is not installed on
this machine — use the `hf` CLI instead, §4.2).

| Path | State | Real size | What it is |
|---|---|---|---|
| `train.csv` | present | 7.1 MB | 32,673 rows — 22,000 Normal / 10,673 abnormal |
| `test.csv` | present | 624 KB | 2,606 rows — 1,300 Normal / 1,306 abnormal |
| `annotations/prevad_test_anno.json` | present (1.1 MB) | 1.1 MB | same rows as `test.csv`, `v6/`-prefixed paths — **redundant, §1.1** |
| `annotations/prevad_train_anno.json` | present (11.8 MB) | 11.8 MB | same for train — **redundant, §1.1** |
| `annotations/data_sources.csv` | present | 4.3 MB | `video_id,source,url,comment` — the raw-video download manifest, **§7.4 only** |
| `features/ViT-B-16-8p-features.zip` | **LFS pointer** | **2.39 GiB** | the features we want |
| `features/ViT-L-14-8p-features.zip` | **LFS pointer** | 3.57 GiB | ViT-L/14 — **not** our backbone, skip |
| `train_search_cache.json` | **LFS pointer** | 397 MiB | baseline's retrieval-based DVS cache — **we do not need it** (§6) |
| raw videos | **absent** | — | "only private sharing due to restrictions of platforms" (baseline README) |

Row totals: 23,300 normal + 11,979 abnormal = **35,279**, matching the paper.

### CSV schema

```
video_id,path,video_path,class_name,superclass_name,descriptions,anomaly_span
```

- `video_id` **equals** the `.npy` stem for all 35,279 rows → a flat unzip needs
  **zero renaming** to satisfy our `cache/clip/{DATASET}/{video_id}.npy` contract.
- 1,027 ids contain `:` (e.g. `2N4dHs1Z4w4_00:42:43.600_00:42:52.234`). Fine on
  APFS/ext4; **breaks on Windows, FAT and exFAT**, and some zip tools mangle it.
  If the unzip lands on a Drive-mounted FAT volume, expect silent corruption.
- `anomaly_span` is a **list of normalized `[start, end]` fractions** — the same
  convention `core/data/dota.py` already consumes. Train rows have exactly one
  span; **104 test rows have 2+ spans** (§3, gap G2).
- `descriptions` is populated for **all 10,673** abnormal train rows and empty
  for every normal row.

### 1.1 Do we need `annotations/*.json` and the CSVs? — measured, 2026-08-25

**Short answer: the two CSVs, yes. Everything in `annotations/`, no.**

`core/data/prevad.py` takes `--train-csv` and `--test-csv` and nothing else
(`core/data/prevad.py:541-542`). The question is whether the JSONs carry anything
the CSVs do not. They do not — checked row by row across all 35,279 records:

| check | train | test |
|---|---|---|
| row count, CSV vs JSON | 32,673 = 32,673 | 2,606 = 2,606 |
| `video_id` ≠ `path` stem | **0** | **0** |
| `class_name` / `superclass_name` differ | **0** | **0** |
| `anomaly_span` differs | **0** | **0** |
| `descriptions` differ (JSON `null` ≡ CSV `""`) | **0** | **0** |

The CSV is a **strict superset**: same six fields, plus an explicit `video_id`
column. The JSON's only difference is a `v6/` prefix on `path`/`video_path` and
`null` where the CSV writes an empty cell. So:

| File in the clone | Needed? | Verdict |
|---|---|---|
| `train.csv`, `test.csv` | **yes** | The only inputs `core/data/prevad.py` reads. Stage them on Drive (§4.1.1). |
| `annotations/prevad_train_anno.json` | no | Byte-for-byte the same records as `train.csv`, minus `video_id`. Nothing reads it. |
| `annotations/prevad_test_anno.json` | no | Same, for `test.csv`. |
| `annotations/data_sources.csv` | no | Provenance/URL manifest for re-downloading raw video. That path is closed (§7.4). Keep it locally as the evidence behind §7.4's table; do not copy it to Drive. |
| `train_search_cache.json` | no | The baseline's retrieval-DVS cache; we build our own (§6). Still an LFS pointer — leave it that way. |
| `features/ViT-L-14-8p-features.zip` | no | 768-d — wrong backbone (`CLIP_FEATURE_DIM = 512`). |
| `features/ViT-B-16-8p-features.zip` | **yes** | The features. Already on Drive (§4.2). |

> Reproduce the table above with the loop in `core/tests/` style — read both
> files, zip the rows, compare the five fields. It takes ~10 s and is the only
> reason to open the JSONs at all.

If the release is ever re-pulled, `train.csv`/`test.csv` are plain files in the
clone (not LFS pointers), so a shallow `git clone` already gives you everything
this runbook needs except the 2.39 GiB feature zip.

---

## 2. Are the features reusable? The evidence

Our `*_ncc` caches and the released PreVAD features were built like this:

| | LaGoVAD (`tools/extract_feat_clip.py` + `src/utils/video_loader.py`) | Ours (`core/tools/extract_clip_features.py --no-center-crop`) |
|---|---|---|
| Encoder | OpenAI CLIP `ViT-B/16`, `jit=True` | HF `openai/clip-vit-base-patch16`, pinned rev, fp32 |
| Output | `encode_image` → 512-d, unnormalized | `.image_embeds` → 512-d, unnormalized |
| Transform | `no_center_crop`: `Resize((224,224))` anisotropic, then OpenAI mean/std | identical (`preprocess_frames(center_crop=False)`) |
| Sampling | `np.arange(interval//2, N, interval)` → **4, 12, 20, …** | `[::stride]` → **0, 8, 16, …** |
| Interval | 8 (`interval=8`; "8p" in the filename) | `FRAME_STRIDE = 8` |
| Saved dtype | whatever `encode_image` returns under jit — **possibly float16** | float32 |

**Three deviations, all benign or checkable:**

1. **Phase offset of 4 frames.** Their frame *k* is our frame *k* + 4. The model
   has no absolute-time prior, so this does not affect training. It *would*
   matter the day we extract flow ourselves and have to phase-match (§7).
2. **fp16 vs fp32.** `DVSFeatureDataset._load_features` does
   `np.load(...).astype(np.float32)` (`core/data/dataset.py:97`), so a float16
   cache loads correctly. Only disk size differs.
3. **HF vs OpenAI CLIP implementation.** Same weights, same projection; the
   `Normalize`-before-`Resize` ordering in their loader is mathematically
   identical to ours because bilinear resampling is a partition-of-unity linear
   filter and normalization is affine.

None of this is proof. **Gate P0 (§4.5) is the proof** — and it is the same
trick that caught the DoTA pooling bug (`RESULTS_DOTA.md`): run LaGoVAD's
released `best.ckpt` on the released PreVAD *test* features through our eval.
`best.ckpt` was trained on PreVAD train, so this is in-domain: a compatible
cache gives a high AUC, an incompatible one reads near chance.

> Per lesson C13, a feature cache is bound to the transform that built it. Treat
> the released features as a **third** cache alongside `*_ncc` and the retired
> center-crop set: give them their own directory, never mix.

---

## 3. Code gaps — what must be written before any of this runs

| # | Gap | Where | Effort |
|---|---|---|---|
| ~~**G1**~~ | ~~**No PreVAD preprocessor.**~~ **DONE 2026-08-24** — `core/data/prevad.py`, argparse CLI, modelled on `dota.py`. | `core/data/prevad.py` | done |
| ~~**G2**~~ | ~~**Multi-span labels unsupported.**~~ **DONE 2026-08-24** — every span filled independently and clamped to `[0, L]`. Confirmed: 104 multi-span test clips, up to **4** windows. Two further quirks found in the shipped table (below). Lesson **18/C18**. | `core/data/prevad.py:sampled_frame_labels` | done |
| ~~**G3**~~ | ~~**The class taxonomy in `definitions.py` is the wrong one.**~~ **DONE 2026-08-24** — `_PREVAD_CLS_DEFS` rewritten to the baseline's *live* 36-name `DEFAULT_CLASSES`; all 35 observed classes covered. **The claim that this would raise was wrong** — see below. Lesson **19/C19**. | `core/data/definitions.py:194` | done |
| **G4** | **Per-video descriptions are not wired.** LaGoVAD's whole point is definition-conditioning, and PreVAD is the only dataset we have that ships per-video `descriptions`. Our caption branch verbalizes *class names* (`core/train.py:257-267`); `loss.captions_from_definitions` is off because MSAD has no descriptions. | `DVSFeatureDataset` must carry a `caption` field; `train.py` must prefer it | medium — see §8 decision |

G1–G3 were **blocking** and are now closed. G4 is a scientific choice, not a
blocker (§8), and stays open by the §8 recommendation (option A first).

### Corrections to this section, from implementing it

**G3 would not have raised.** `class_index_tensor` (`core/train.py:136-148`)
only checks a class against `defs.json`, which we generate from the same CSV the
class names come from — it always agrees. And `verbalize_class_name`
(`core/data/definitions.py:342-346`) returns the *bare class name* for an unknown
class by design. Composed, they are silent: the text branch would have been
conditioned on `"Store Robbery"` instead of a definition sentence, voiding
LaGoVAD's central claim with no error anywhere. A silent G3 is worse than the
predicted crash, because a crash is a gate. `core/data/prevad.py` now asserts
definition coverage at preprocessing time, which is a real gate.

**The taxonomy is 35 in the data, 36 in the lookup.** The baseline's live
`DEFAULT_CLASSES` has 36 names; `Fire-related Accident` is a superclass with no
"Others" rows and appears only in `superclass_name`. `defs.json` therefore
carries the **35** that occur (so `H_mul` gets no permanently-negative column)
while `_PREVAD_CLS_DEFS` carries all 36 and can never under-cover. Six names are
simultaneously superclass and class (`Vehicle Accident`, `Violence`, `Robbery`,
`Production Accident`, `Animal-related Violence`, `Daily Accident`) — those rows
are the taxonomy's "Others" buckets and their definitions are deliberately broad.

**Two label defects in the shipped `test.csv`, neither documented anywhere.**
Measured, not assumed:

| quirk | count | worst case |
|---|---|---|
| spans ending past 1.0 | **440** | 1.2104 (21 % past the clip end) |
| spans with `end <= start` | **1** (`RrUW8ITUqx0_aug3`) | 0.9814 → 0.7110 |

Both are **absorbed, not repaired**. The baseline gets clamping for free by
filling into a `vis_max_len`-long buffer (`base.py:57-69`), so out-of-range ends
are truncated and reversed spans write nothing; the released ground truth is
*defined* by that behaviour. Rescaling or dropping those clips would diverge from
the labels the published number was measured against. `build_frame_labels` logs
all three counts (overflow, reversed, rounded-away) on every run.

**There is no `--stride` flag, deliberately.** Every other preprocessor derives
label length from a raw frame count. PreVAD ships no frames — the `.npy` *is* the
interval-8 sequence — so lengths are read from the array header, the only value
that can be right. `--clip-dir` is therefore **required**, not optional.

**§4.4's sanity checks are folded into the preprocessor** and can no longer be
skipped: feature width must be 512 (a 768-d array means the ViT-L/14 zip was
unpacked), arrays must be non-empty, and the mean L2 norm is sampled and warned
on if it looks pre-normalized. `--allow-missing-features` is the C10-style
escape hatch and logs coverage.

### What `core/data/prevad.py` must emit

Per the layout contract (`core/docs/DATA_LAYOUT.md`), with an `argparse` CLI:

```
data/PreVAD/labels_train.json        {video_id: 0|1}          from train.csv class_name != "Normal"
data/PreVAD/frame_labels_test.json   {video_id: [0,1,...]}    from test.csv anomaly_span x feature length
data/PreVAD/defs.json                ["Normal", ...34 more]   Normal FIRST (core/train.py:533)
data/PreVAD/meta.json                {video_id: {class_name, superclass_name, descriptions, anomaly_span, split}}
```

Two things it must get right:

- **Frame labels need the real feature length.** `dota.py` derives `L` from the
  frame count on disk. Here there are no frames — read `L` from the extracted
  `.npy` header (`np.load(..., mmap_mode="r").shape[0]`) so the labels match the
  cache exactly. Do **not** guess it from the timestamps in the id.
- **Rounding convention:** `round(frac * L)`, half-open fill, same as
  `dota.py:241` — that is what reproduced the published DoTA number.

---

## 4. Step-by-step setup

### 4.0 Keep the clone out of git

`PreVAD/` is an untracked **nested git repo** that will hold multiple GB. Add it
to `.gitignore` next to `LaGoVAD-PreVAD/` before the first `git add`.

### 4.0.1 Environment preconditions — verify, do not assume

Three of these bit in a row on 2026-08-25 (`faiss` missing, then
`transformers` too new, then the torch pin unobtainable). All three had the same
root cause and it is worth stating once:

> **A batch `pip install` is atomic, and a failing `!pip` does not stop the
> notebook.** One unsatisfiable requirement makes pip install **none** of the
> others, print `ERROR`, and exit non-zero — and the next cell runs anyway. The
> `torch==2.4.*` pin from `pyproject.toml` is unsatisfiable on Colab's current
> **Python 3.13** (wheels start at 2.5.0), so a cell asking for the full pinned
> set installs *nothing*: not `transformers==4.56.*`, not `numpy<2`. The symptom
> arrives much later as `AttributeError: 'CLIPTextModel' object has no attribute
> 'text_model'`. Lesson **21/C21**.

**What is installable on a Python 3.13 runtime**, and what to accept instead:

| `pyproject.toml` pin | On Colab py3.13 | Do |
|---|---|---|
| `torch==2.4.*`, `torchvision==0.19.*` | **unobtainable** — no cp313 wheels | keep Colab's torch; omit from the install |
| `numpy<2` | installable, but **breaks Colab's torch** (built on the numpy 2 ABI) | do not pin; accept numpy 2 |
| `transformers==4.56.*` | installable | **pin it** — `core/models/clip_text.py` reaches into 4.x `CLIPTextModel.text_model` internals at eight sites (`:77, 115, 122, 135, 141, 146, 151, 158`) |
| everything else | installable | pin as declared |

```python
!pip install -q \
  "transformers==4.56.*" \
  "av>=12" "einops>=0.8" "faiss-cpu==1.14.3" "gdown>=5" \
  "matplotlib>=3.9" "opencv-python==4.11.*" "pyyaml>=6" \
  "requests>=2.32" "scikit-learn>=1.5" "torchmetrics>=1.4" "tqdm>=4.66"
```

Then **Runtime → Restart session**, re-run the mount and §4.1 env cells, and
*verify* — never take the install's word for it:

```python
import faiss
import numpy
import torch
import transformers
from transformers import CLIPTextModel

print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| numpy', numpy.__version__, '| faiss', faiss.__version__)
print('text_model present:',
      'text_model' in CLIPTextModel.__init__.__code__.co_names)
```

Require `transformers 4.56.x` and `text_model present: True`. Anything else and
`core.evaluate` will die deep inside the frozen text encoder.

**The drift this leaves you with is real, so bound it.** Every number in
`RESULTS_{MSAD,DOTA,NCC,PHASE_A,ARM4_PROBE}.md` was measured on **torch 2.4 /
numpy<2**; this runtime is torch 2.9+ / numpy 2. Gate P0 (§4.5) is a
chance-vs-not-chance sanity gate and tolerates that. The Stage M A/B (§7.3) does
not. Before Stage M, re-run **one already-measured arm** as a control — the
cheapest is `best.ckpt` on DoTA, which is recorded at **0.6142** per-clip
min-max (`RESULTS_DOTA.md`). Reproduce it and the environment change is bounded
with a footnote; miss it and you found out before 13,360 steps, not after. This
is lesson 8b's discipline applied to the interpreter instead of the checkpoint.

### 4.1 Where everything lives

Two tiers, and the split is the whole point: **Drive holds what is expensive to
obtain, `/content` holds what is cheap to rebuild.** The 2.39 GiB zip and the
label files are durable; the 35,279 extracted `.npy` are per-session.

```python
import os

DRIVE = '/content/drive/MyDrive/kat-vad'          # same root as COLAB.md Step 2
os.environ['KATVAD_DATA_ROOT']   = f'{DRIVE}/data'
os.environ['KATVAD_CACHE_ROOT']  = f'{DRIVE}/cache'
os.environ['KATVAD_CKPT_ROOT']   = f'{DRIVE}/ckpts'
os.environ['KATVAD_OUTPUT_ROOT'] = f'{DRIVE}/outputs'

# PreVAD handles — every cell in this runbook uses these three
os.environ['PREVAD_ZIP']  = f"{os.environ['KATVAD_CACHE_ROOT']}/clip/PreVAD/ViT-B-16-8p-features.zip"
os.environ['PREVAD_DATA'] = f"{os.environ['KATVAD_DATA_ROOT']}/PreVAD"   # Drive, durable
os.environ['PREVAD_CLIP'] = '/content/cache/clip/PreVAD'                 # VM disk, per session

for v in ('KATVAD_DATA_ROOT', 'KATVAD_CACHE_ROOT', 'KATVAD_CKPT_ROOT',
          'KATVAD_OUTPUT_ROOT', 'PREVAD_DATA', 'PREVAD_CLIP'):
    os.makedirs(os.environ[v], exist_ok=True)
```

Target layout. `[once]` survives every session; `[session]` is rebuilt by §4.3
each time the VM is recycled:

```
DRIVE — durable
$KATVAD_CACHE_ROOT/clip/PreVAD/
└── ViT-B-16-8p-features.zip        2.39 GiB  [once]     the download      §4.2

$KATVAD_DATA_ROOT/PreVAD/                                ← $PREVAD_DATA
├── train.csv                        7.1 MB   [once]     staged from clone §4.1.1
├── test.csv                         624 KB   [once]     staged from clone §4.1.1
├── labels_train.json                         [once]     built             §5
├── frame_labels_test.json                    [once]     built             §5
├── defs.json                                 [once]     built             §5
├── meta.json                                 [once]     built             §5
└── test_ids.txt                              [once]     built             §5

$KATVAD_CACHE_ROOT/knn/PreVAD/knn_cache.npz   [once]     built             §6
$KATVAD_OUTPUT_ROOT/PreVAD/gate_p0/           [once]     Gate P0           §4.5
$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off/    [once]     the trunk         §7.1

VM DISK — rebuilt per session
/content/cache/clip/PreVAD/                              ← $PREVAD_CLIP
└── <video_id>.npy  × 35,279       ~2.5 GiB   [session]  inflated from the zip §4.3
```

Only §4.3 is per-session. Everything built downstream — label files, KNN cache,
checkpoints, eval scores — lands on Drive and is built once. **The label files
are valid across sessions** even though the features are not: `.npy` inflated
from the same archive are byte-identical every time, so the feature lengths §5
read from them never change.

> **Naming, and lesson C13.** Every other cache directory is named after the
> extractor that built it (`MSAD_ncc`, `DoTA_s8_ncc`), and an earlier draft
> called this one `PreVAD_rel`. Plain `PreVAD` is safe **only because a second
> PreVAD cache cannot exist** — the release ships no video (§7.4), so we can
> never self-extract a rival. If the authors ever send raw video (§10 step 1),
> the self-extracted cache must be `PreVAD_ncc` and this one keeps its name as
> the released set. Do not relax C13 anywhere else on the strength of this.

#### 4.1.1 Staging the two CSVs — the step that is easy to miss

`PreVAD/` is **gitignored**, so cloning this repo on Colab does **not** bring
`train.csv` and `test.csv` with it, and §5 cannot run without them. They are
7.7 MB total and belong on Drive next to the label files they produce. Put them
there once, from the local machine or by hand:

```bash
# local machine, with Drive for Desktop or the web UI — 7.7 MB
cp PreVAD/train.csv PreVAD/test.csv <Drive>/kat-vad/data/PreVAD/
```

```bash
# Colab, to confirm they arrived
!ls -l "$PREVAD_DATA"
# expect train.csv ~7,428,461 B and test.csv ~638,109 B
!head -1 "$PREVAD_DATA/train.csv"
# expect: video_id,path,video_path,class_name,superclass_name,descriptions,anomaly_span
```

Nothing else from the clone travels — see §1.1 for why the `annotations/` files
stay behind.

### 4.2 The features zip (already on Drive)

The zip is `$PREVAD_ZIP`. It stays on Drive permanently: it is the source §4.3
inflates from every session, and the only way to rebuild the feature directory.
Confirm it is the real file and not a 135-byte LFS pointer:

```bash
!ls -l "$KATVAD_CACHE_ROOT/clip/PreVAD/"
# expect ~2.39 GiB (2,563,392,012 bytes), not 135 B
!shasum -a 256 "$PREVAD_ZIP"
# expect 52fc1579ce23d628846f381ed5706b0425652f8c742a9151dba46e9cab1010a4
```

The `shasum` reads 2.39 GiB over FUSE — a few minutes. Run it once, when the zip
first lands; skip it on later sessions.

<details><summary>If the zip ever has to be re-fetched</summary>

`git lfs` is not installed on the local machine; use the `hf` CLI
(`huggingface_hub` 0.36.2 is in `.venv`, so `hf` is current and
`huggingface-cli` is the deprecated alias):

```bash
source .venv/bin/activate
hf download Kamino123/PreVAD --repo-type dataset \
    --include "features/ViT-B-16-8p-features.zip" --local-dir PreVAD/
```

Skip `ViT-L-14-8p-features.zip` (768-d — wrong backbone, our
`CLIP_FEATURE_DIM` is 512) and `train_search_cache.json` (397 MiB, §6).
2.39 GiB instead of 6.4 GiB.
</details>

### 4.3 Unzip to the local VM disk — every session

**This is the single biggest performance decision in the runbook.** 35,279 small
`.npy` on a Drive FUSE mount would make training I/O-bound: every `__getitem__`
is one small random read and DVS makes several per item, so Stage P alone is
~854k opens. Inflate to `/content` instead and pay a few minutes per session.

```bash
%%bash
set -euo pipefail
mkdir -p "$PREVAD_CLIP"
df -h /content | tail -1                     # need ~5 GiB free (~8 GiB while recovering)

if [ -f "$PREVAD_CLIP/.unzip_complete" ]; then
  echo "already extracted this session on $(cat "$PREVAD_CLIP/.unzip_complete")"
  exit 0
fi

# -j junks the archive's directory prefix so the .npy land flat right away.
# Do NOT flatten afterwards with `mv dir/* .` -- 35,279 arguments is past
# ARG_MAX and dies with "Argument list too long".
# -n = never overwrite: a cell that dies mid-unzip resumes, it does not restart.
time unzip -n -j -q "$PREVAD_ZIP" -d "$PREVAD_CLIP"

N=$(find "$PREVAD_CLIP" -maxdepth 1 -name '*.npy' | wc -l)
echo "extracted $N .npy files"
[ "$N" -eq 35279 ] || { echo "EXPECTED 35279 -- marker NOT written, re-run this cell"; exit 1; }

# only now, with 35,279 flat files confirmed: drop the nested copy an earlier
# non -j run left behind. Never run this before the count check.
if [ -d "$PREVAD_CLIP/ViT-B-16-8p-features" ]; then
  echo "removing the duplicate nested copy from the earlier run"
  rm -rf "${PREVAD_CLIP:?}/ViT-B-16-8p-features"
fi

du -sh "$PREVAD_CLIP"
```

> **Why `-j` and not a flatten step — measured, 2026-08-25.** The archive *does*
> carry a `ViT-B-16-8p-features/` prefix, and the first version of this cell
> flattened it with `mv "$PREVAD_CLIP/ViT-B-16-8p-features/"* "$PREVAD_CLIP/"`.
> The glob expands to **35,279 arguments** and `execve` refuses:
> `bash: line 15: /usr/bin/mv: Argument list too long`. `unzip -j` junks the
> prefix during extraction, so there is nothing to flatten. If a shell loop over
> a whole dataset ever *looks* necessary, that is the smell — let the tool that
> produced the paths handle them.

The marker is written by the **verification** cell below, not by the unzip —
a count of 35,279 is not evidence the files have content (lesson C10: a FUSE
unzip left five DoTA clips as empty directories and the folder count lied).
Together with `unzip -n`, that makes the whole of §4.3 idempotent: safe to
re-run, cheap when it has already succeeded.

```bash
%%bash
python - <<'PY'
import csv, datetime, os, pathlib, random

import numpy as np

cache = pathlib.Path(os.environ["PREVAD_CLIP"])
data = pathlib.Path(os.environ["PREVAD_DATA"])
ids = [r["video_id"] for f in ("train.csv", "test.csv")
       for r in csv.DictReader(open(data / f, newline="", encoding="utf-8"))]
print("ids in the CSVs:", len(ids))

missing = [i for i in ids if not (cache / f"{i}.npy").exists()]
print("missing:", len(missing), missing[:5])

colon = [i for i in ids if ":" in i]
print("colon ids:", len(colon),
      "missing among them:", sum(1 for i in colon if not (cache / f"{i}.npy").exists()))

bad = []
for i in random.Random(0).sample(ids, 256):        # header read, not a full load
    try:
        a = np.load(cache / f"{i}.npy", mmap_mode="r")
        if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != 512:
            bad.append((i, a.shape))
    except Exception as exc:                        # noqa: BLE001 - a notebook probe
        bad.append((i, repr(exc)))
print("bad headers among 256 sampled:", len(bad), bad[:5])

if not missing and not bad:
    stamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    (cache / ".unzip_complete").write_text(stamp)
    print("marker written — §4.3 is done for this session")
PY
```

Expect `ids 35279`, `missing 0`, `colon ids 1027` with **0** missing, and
`bad headers 0`. Anything else: delete `.unzip_complete` if it exists, re-run the
unzip cell (`-n` makes that cheap), and do not proceed.

Notes:

- Budget **~5 GiB free** on `/content` plus the 2.39 GiB zip on Drive. Measured
  2026-08-25: `overlay 226G 24G 203G 11% /` — ample. `df -h /content` is in the
  cell above for the sessions where it is not. The one-time recovery from the
  `mv` failure holds both copies at once; budget ~8 GiB for that run only.
- **Measured 2026-08-25: `real 0m14.5s` (`user 11.7s`) for the whole 2.39 GiB**
  — CPU-bound inflate, not a Drive read. The "few minutes per session" this
  section used to warn about is **fifteen seconds**. Whatever the argument for
  keeping the extracted features on Drive was, this is not it.
- **1,027 ids contain `:`** (§1). Fine on the VM's ext4; the verifier checks them
  by name every run. Never inflate onto a FAT/exFAT volume or into a folder
  mirrored to Windows by Drive for Desktop — colons are illegal there and the
  files are silently renamed or dropped.
- Nothing downstream cares that `$PREVAD_CLIP` is ephemeral: the label files
  (§5), the KNN cache (§6) and every checkpoint land on Drive, and `.npy`
  inflated from the same archive are byte-identical session to session.
- Extraction is the *only* per-session step. If you ever want it durable
  instead, the trade is stated in the second bullet above — it costs Stage P
  wall clock, not correctness.

### 4.4 Sanity checks (2 minutes, no GPU)

Coverage and headers are already covered by §4.3's verifier, which runs every
session. What is left is the scale check and the interval check — both are
properties of the archive, so once is enough:

```bash
source .venv/bin/activate && python - <<'PY'
import csv, os, pathlib
import numpy as np

cache = pathlib.Path(os.environ["PREVAD_CLIP"])
rows = list(csv.DictReader(open(pathlib.Path(os.environ["PREVAD_DATA"]) / "test.csv",
                                newline="", encoding="utf-8")))
a = np.load(cache / f"{rows[0]['video_id']}.npy")
print("shape", a.shape, "dtype", a.dtype,
      "L2 mean", float(np.linalg.norm(a, axis=1).mean()))
PY
```

Expect the second dim to be **512** and an L2 norm around 8–12 (**not** ~1.0 —
the baseline saves *unnormalized* `encode_image` output, and so do we; a mean of
1.0 means they are pre-normalized and our pipeline would need to match). A second
dim of 768 means the ViT-L/14 zip was unpacked by mistake.

**Interval check ("what does 8p mean?").** Many ids encode their clip window —
`IAGCgLLt7dM_54_86` is seconds 54→86, `2N4dHs1Z4w4_00:42:43.600_00:42:52.234` is
8.63 s. At ~30 fps and interval 8 those are ~120 and ~32 rows. If instead you
see ~960 and ~259, the release is interval 1 and everything downstream (labels,
`MAX_VIS_LEN`, DVS) changes.

### 4.5 Gate P0 — the decisive compatibility test

Build the label files (§3, after `prevad.py` exists), then run the released
checkpoint through our eval on PreVAD test:

```bash
source .venv/bin/activate && python -m core.evaluate \
    --baseline-ckpt ckpts/lagovad_best.ckpt \
    --data-dir "$PREVAD_DATA" --clip-dir "$PREVAD_CLIP" \
    --output-dir "$KATVAD_OUTPUT_ROOT/PreVAD/gate_p0" \
    --set data.dataset=PreVAD --score-norm auto --save-scores
```

- PreVAD test is **49.9 % normal**, so `--score-norm auto` resolves to **raw
  pooling** (`SCORE_NORM_AUTO_NORMAL_FRACTION = 0.05`) — the MSAD protocol, not
  the DoTA one. This is the right call here; do not min-max (lesson 12/C12).
- **Pass:** a high in-domain AUC — `best.ckpt` trained on this exact train split.
- **Fail (≈0.5):** the features are not compatible with the checkpoint. Stop and
  diagnose; do **not** proceed to training.

There is no published PreVAD-test AUC to gate against, so P0 is a
*sanity* gate (chance vs. clearly-not-chance), not a reproduction gate. Per
lesson 8b, the reproduction reference is the released checkpoint, never a
printed number.

---

## 5. Build the label files

```bash
source .venv/bin/activate && python -m core.data.prevad \
    --train-csv "$PREVAD_DATA/train.csv" --test-csv "$PREVAD_DATA/test.csv" \
    --clip-dir "$PREVAD_CLIP" \
    --out-dir "$PREVAD_DATA"
```

The CSVs are read from `$PREVAD_DATA` (staged in §4.1.1), not from the local
`PreVAD/` clone — that clone does not exist on Colab. Inputs and outputs share
the directory on purpose: `--out-dir` writes five files whose names cannot
collide with `train.csv`/`test.csv`, so `$PREVAD_DATA` ends up self-contained and
`--data-dir "$PREVAD_DATA"` is all any later command needs.

Expected output — verified against the real CSVs on 2026-08-24 (feature-dependent
lines need the unzipped cache):

```
INFO: Read 32673 train rows ... (22000 normal / 10673 abnormal)
INFO: Read 2606 test rows ... (1300 normal / 1306 abnormal)
INFO: 1 defined classes do not occur in the data ...: ['Fire-related Accident']
INFO: Mean feature L2 norm 1x.xxx over 32 clips (raw, as expected)
WARNING: 440 anomaly spans end past the clip (max end 1.2104) -- clamped ...
WARNING: 1 anomaly spans run backwards (end <= start) ...
WARNING: 1 abnormal clips have no positive frame after rounding ...: ['RrUW8ITUqx0_aug3']
INFO: PreVAD: 32673 train (10673 abnormal), 2606 test (1300 normal / 1306 abnormal),
      35 classes, ... test feature rows, ... positive
INFO: Test split is 49.9% normal videos -- --score-norm auto resolves to raw pooling
```

The three WARNING lines are **expected and correct** (§3). Their absence means
the CSVs changed. `--strict` turns the last one fatal.

---

## 6. DVS filler cache

The baseline's `train_search_cache.json` (397 MiB) belongs to its
`retrieval_based_synthesis` path, which we did not port. Ours is
`core/data/knn_cache.py` — faiss over central-frame CLIP features, K=10 — and it
is cheaper to rebuild than to download:

```bash
source .venv/bin/activate && python -m core.data.knn_cache \
    --data-dir "$PREVAD_DATA" --dataset PreVAD \
    --clip-dir "$PREVAD_CLIP"
```

32,673 × 512 float32 ≈ 67 MB in the index — seconds on CPU. Leave `--motion-key`
**off** (it needs the flow stats we do not have, and PreVAD is not egocentric).

---

## 7. Training — the trunk-transfer plan

```
PreVAD (released features, KIP-off)  ──►  trunk        §7.1   ~13,360 steps, once
        │
        └─ graft randomly-initialised KIP weights      §7.2   one cell, no GPU
                │
                ├── MSAD stage 2, KIP-on   ─┐
                └── MSAD stage 2, KIP-off  ─┴─► evaluate MSAD + DoTA   §7.3
```

**Why this shape.** Flow is needed only where KIP is *attached*, never where the
trunk is *pretrained* — `require_flow = cfg.kip.enabled` (`core/train.py:593`).
MSAD already has a flow cache (`cache/flow/v1/MSAD`). So a PreVAD-pretrained
trunk gives us the one thing the current result is missing: a Δ(on − off)
measured on top of a baseline trained the way the paper trains it, instead of on
top of an MSAD-only baseline.

**What it answers.** Today's headline is +0.09 on DoTA with both arms trained on
MSAD. There is a suggestive cross-check — our MSAD-trained KIP-on scores 0.6519
against LaGoVAD's PreVAD-trained `best.ckpt` at 0.6142, **+0.0508, CI
[+0.0355, +0.0659]** (`RESULTS_NCC.md`) — but that compares across two codebases
and two training sets at once. This plan makes it controlled: same code, same
eval, one variable.

Both outcomes are publishable and they say different things:

| result | reading |
|---|---|
| Δ(on − off) from the PreVAD trunk stays ≈ +0.09 | KIP's gain is **additive with data scale**. The strongest version of the claim. |
| Δ shrinks toward zero because KIP-off catches up | KIP is a **substitute for pretraining scale** — a cheap way to buy what 35k videos buy. Weaker, still useful, and far better found here than in review. |

### 7.1 Stage P — pretrain the trunk on PreVAD (KIP-off)

Our defaults already *are* LaGoVAD's PreVAD config: `lr 5e-5`, `batch 64`,
`40 epochs`, `warmup 20`, `vis_max_len 512`, seed 2024
(`LaGoVAD-PreVAD/src/configs/default.yaml` vs `core/constants.py`).

```bash
!python -m core.train \
  --data-dir "$PREVAD_DATA" --clip-dir "$PREVAD_CLIP" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/PreVAD/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off" \
  --set data.dataset=PreVAD --set data.is_egocentric=false \
  --set kip.enabled=false --set train.seed=2024 \
  --set loss.captions_from_definitions=true
```

`--flow-dir` is **omitted** on purpose: with `kip.enabled=false` no flow target is
consumed, and passing it only invites confusion about what the arm sees (same
convention as `COLAB.md` §A4).

**Cost.** `len(DVSFeatureDataset) = 2 x 10,673 = 21,346` items/epoch → **334
batches/epoch**, **13,360 batches** over 40 epochs (`grad_accum_steps = 1`, so
batches = optimizer steps). For scale: MSAD stage 2 is ~500 steps, so this is
~27x one MSAD arm. No video decoding — every step is `.npy` reads, ~854k of
them, which is why §4.3 insists on the local VM disk.

> **`data.num_workers` is dead config — do not reach for it.** It is declared at
> `core/config.py:114` and read **nowhere** in the tree. There is no
> `DataLoader` anywhere in `core/`: `core/train.py:470` iterates a per-epoch
> seeded permutation and builds each batch with
> `samples = [self.dataset[i] for i in indices]`, serially, in the main process
> — 64 items per batch, each 1-4 `np.load` calls once DVS fillers are counted.
> That serialization is what makes the resume in §7.1.1 exact; it is not a knob.
> An earlier revision of this section advised raising `num_workers` when the GPU
> idles. That advice was wrong and did nothing.

#### 7.1.1 Running it across sessions — stop, resume, and don't get killed

**Resume is exact.** `save_checkpoint`/`load_checkpoint`
(`core/train.py:408-441`) carry model, optimizer, scheduler, AMP scaler,
`epoch`, `global_step`, `batches_done` and **all four** RNG states (python,
numpy, torch, dataset + verbalizer). With no DataLoader and a seeded per-epoch
permutation, `--resume` replays the exact remaining data/DVS stream, mid-epoch
included; weights match a straight run within FP tolerance. `--resume` and
`--init-weights` are mutually exclusive (`core/train.py:581`).

**But the default granularity is one epoch, and epoch 0 saves nothing.**
`train.checkpoint_every_steps` defaults to **0 = per-epoch only**
(`core/config.py:130`), so the sole artifact is `checkpoint_last.pt`, written at
each epoch boundary (`core/train.py:522`). Kill the run during epoch 0 and there
is nothing to resume from — the whole session is lost. Add step checkpoints only
if an epoch turns out slow: `checkpoint_every_steps=N` writes
`checkpoint_step_{N}.pt` **and** `checkpoint_last.pt` on every hit
(`core/train.py:513-519`). **Measured 2026-08-25: one checkpoint is 459 MB**
(model + optimizer + scheduler + scaler), so `checkpoint_every_steps=100` over
this run means 133 files ≈ **61 GB** on Drive. Leave it at 0.

**Time-box it, do not kill it.** `--stop-after-epochs N` finishes epoch N
cleanly, saves, and returns without touching the LR-schedule horizon
(`core/train.py:525`):

```bash
python -m core.train ... --stop-after-epochs 8
# next session: the same command plus
#   --resume "$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off/checkpoint_last.pt"
```

**Run it detached.** 13,360 batches will outlive a browser tab, and `%%bash`
buffers, so a healthy run looks hung:

```python
!cd /content/drive/MyDrive/Thesis/kat-vad && nohup python -u -m core.train \
  --data-dir "$PREVAD_DATA" --clip-dir "$PREVAD_CLIP" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/PreVAD/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off" \
  --set data.dataset=PreVAD --set data.is_egocentric=false \
  --set kip.enabled=false --set train.seed=2024 \
  --set loss.captions_from_definitions=true \
  > "$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off/train.log" 2>&1 &
```

`-u` unbuffers. Poll from any later cell:

```bash
!tail -20 "$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off/train.log"
!wc -l    "$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off/metrics.jsonl"
!tail -1  "$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off/metrics.jsonl"
```

`metrics.jsonl` gets **one line per batch** carrying `epoch` and `batch` (of
334). Two `wc -l` calls a minute apart give batches/min and a real ETA — the
file has no timestamps, so that difference is the only rate signal.

> **`--resume` across sessions is where lesson 15/C15 bites.** `core/train.py:428`
> loads with `weights_only=False`, so the pickled `np.random.get_state()` has to
> survive. Within one session it round-trips fine. Across sessions on Colab's
> *unpinned* stack, a numpy major shift makes `--resume` die with
> `TypeError: _reconstruct` before it reaches a tensor. `COLAB.md` §A4.0's
> `_NumpyFreeUnpickler` shim recovers it; keep it pasted in the notebook rather
> than finding it mid-run.

**One trunk, three stage-2 seeds.** Seed variation in Phase A lived in stage 2,
and a second 13,360-step run to re-seed the trunk is not worth it now. State the
limitation plainly in whatever report follows: the trunk is n=1.

> Checkpoints from this run carry the numpy-RNG portability defect (lesson
> 15/C15) — see §7.2, which routes around it anyway.

#### 7.1.2 Resuming on a fresh VM — the checklist

**A new VM has wiped `$PREVAD_CLIP`.** It is on `/content` by design (§4.3), so
steps 1–4 below are preconditions, not optional warm-up: resume without them and
training dies on missing `.npy` — or worse, on an unpinned `transformers`, deep
inside the frozen text encoder. Steps 1–4 cost about two minutes.

**1 — mount + env** (identical to §4.1; repeated here so the block is
paste-complete):

```python
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE = '/content/drive/MyDrive/Thesis'
os.environ['KATVAD_DATA_ROOT']   = f'{DRIVE}/data'
os.environ['KATVAD_CACHE_ROOT']  = f'{DRIVE}/cache'
os.environ['KATVAD_CKPT_ROOT']   = f'{DRIVE}/ckpts'
os.environ['KATVAD_OUTPUT_ROOT'] = f'{DRIVE}/outputs'
os.environ['PREVAD_ZIP']  = f"{os.environ['KATVAD_CACHE_ROOT']}/clip/PreVAD/ViT-B-16-8p-features.zip"
os.environ['PREVAD_DATA'] = f"{os.environ['KATVAD_DATA_ROOT']}/PreVAD"
os.environ['PREVAD_CLIP'] = '/content/cache/clip/PreVAD'
for v in ('KATVAD_DATA_ROOT', 'KATVAD_CACHE_ROOT', 'KATVAD_CKPT_ROOT',
          'KATVAD_OUTPUT_ROOT', 'PREVAD_DATA', 'PREVAD_CLIP'):
    os.makedirs(os.environ[v], exist_ok=True)
```

**2 — install** (§4.0.1), then **Runtime → Restart session**, then re-run step 1:

```python
!pip install -q \
  "transformers==4.56.*" \
  "av>=12" "einops>=0.8" "faiss-cpu==1.14.3" "gdown>=5" \
  "matplotlib>=3.9" "opencv-python==4.11.*" "pyyaml>=6" \
  "requests>=2.32" "scikit-learn>=1.5" "torchmetrics>=1.4" "tqdm>=4.66"
```

**3 — verify.** Require `transformers 4.56.x` and `text_model present: True`:

```python
import numpy
import torch
import transformers
from transformers import CLIPTextModel

print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| numpy', numpy.__version__)
print('text_model present:',
      'text_model' in CLIPTextModel.__init__.__code__.co_names)
```

**4 — re-extract the features** (~15 s, §4.3):

```bash
%%bash
set -euo pipefail
mkdir -p "$PREVAD_CLIP"
[ -f "$PREVAD_CLIP/.unzip_complete" ] && { echo "already extracted"; exit 0; }
time unzip -n -j -q "$PREVAD_ZIP" -d "$PREVAD_CLIP"
N=$(find "$PREVAD_CLIP" -maxdepth 1 -name '*.npy' | wc -l)
echo "extracted $N"
[ "$N" -eq 35279 ] || { echo "EXPECTED 35279"; exit 1; }
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$PREVAD_CLIP/.unzip_complete"
```

**5 — resume.** No `--stop-after-epochs`: the loop runs `range(self.epoch,
num_epochs)` and stops itself at 40. Note `>>`, not `>` — truncating the log
loses the earlier sessions:

```python
!cd /content/drive/MyDrive/Thesis/kat-vad && nohup python -u -m core.train --data-dir "$PREVAD_DATA" --clip-dir "$PREVAD_CLIP" --knn-cache "$KATVAD_CACHE_ROOT/knn/PreVAD/knn_cache.npz" --output-dir "$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off" --resume "$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off/checkpoint_last.pt" --set data.dataset=PreVAD --set data.is_egocentric=false --set kip.enabled=false --set train.seed=2024 --set loss.captions_from_definitions=true >> "$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off/train.log" 2>&1 &
```

Confirm from the log, not from hope — `core/train.py:437` prints
`Resumed from … (epoch=N step=M batches_done=K)`. If that line is absent, the
process died before the loop.

**6 — poll:**

```bash
%%bash
OUT="$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off"
tail -5 "$OUT/train.log"
wc -l "$OUT/metrics.jsonl"; tail -1 "$OUT/metrics.jsonl"
ls -lh "$OUT"/checkpoint_last.pt
```

**Before killing anything, in either direction**, check for an orphan trainer —
two processes on one output directory interleave checkpoint writes and corrupt
`metrics.jsonl`:

```bash
!pgrep -af "core.train" || echo none
```

**Two hazards, both cheap to avoid:**

- **Never re-launch without `--resume` once `checkpoint_last.pt` exists.** The
  command is otherwise identical, so the mistake is invisible: it silently
  restarts at epoch 0 and overwrites two-thirds of a trunk. `_log_metrics` opens
  `metrics.jsonl` with `"a"`, so the restart also *appends* a second epoch 0 onto
  the first run's records. If you ever do restart deliberately, move the old
  `metrics.jsonl` aside first.
- **`TypeError: _reconstruct` on step 5 is lesson 15/C15**, not a corrupt
  checkpoint — Colab's numpy major moved since the last save, and
  `core/train.py:428` loads at `weights_only=False`. The weights are intact.
  Recover with `COLAB.md` §A4.0's `_NumpyFreeUnpickler`, re-save slim, resume
  from that. Do **not** pin `numpy<2`; it breaks the torch wheel's ABI.

### 7.2 Stage T — graft KIP onto the trunk

`warm_start_model` (`core/train.py:99-108`) is fail-loud on **missing** keys as
well as unexpected ones. A KIP-off checkpoint has no `kip.*` tensors, so feeding
it straight to a `kip.enabled=true` run raises. That guard is lesson 5 working as
designed — **do not weaken it.** Write an explicit merged copy instead: trained
trunk + freshly initialised KIP.

This is the mirror image of `COLAB.md` §A4.0, which *strips* `kip.*`; here we
*add* it. The unpickler shim is the same one and is required for the same reason
(`np.random.get_state()` in the payload aborts `torch.load` after a numpy major
version drift).

```python
import os, pickle, torch
from pathlib import Path
from core.config import load_config
from core.models.kat_vad import KATVAD


class _Discarded:
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


SRC = Path(os.environ['KATVAD_OUTPUT_ROOT']) / 'PreVAD/stage2_kip_off'
payload = torch.load(SRC / 'checkpoint_last.pt', map_location='cpu',
                     weights_only=False, pickle_module=_shim)
trunk = payload['model']

# a fresh KIP-enabled skeleton supplies the randomly-initialised kip.* tensors
cfg = load_config(SRC / 'config.yaml', ['kip.enabled=true'])
skeleton = KATVAD.from_config(cfg, load_clip=False).state_dict()

merged = dict(skeleton)
overlap = 0
for k, v in trunk.items():
    if k in merged and merged[k].shape == v.shape:
        merged[k] = v
        overlap += 1
    elif k in merged:
        raise ValueError(f'shape mismatch on {k}: {merged[k].shape} vs {v.shape}')

kip_keys = [k for k in merged if k.startswith('kip.')]
print(f'trunk tensors reused: {overlap}/{len(trunk)}; '
      f'kip.* left at init: {len(kip_keys)}; total: {len(merged)}')

slim = {'model': merged,
        'epoch': payload.get('epoch'),
        'global_step': payload.get('global_step'),
        'class_names': payload.get('class_names')}
torch.save(slim, SRC / 'checkpoint_last_withkip.pt')
```

Three checks before moving on:

- `overlap` must equal `len(trunk)`. Anything less means a tensor from PreVAD
  found no home in the KIP-enabled model, and the trunk is not fully transferring.
- `kip.* left at init` must be **> 0** — otherwise you loaded a KIP-on checkpoint
  by mistake and the "graft" is a no-op.
- `global_step` must match stage P's `config.yaml`; `None` means the load
  returned something that is not a trainer checkpoint.

**Why no shape mismatch across datasets.** PreVAD has 35 classes and MSAD has its
own list, but `MultiClassHead` is similarity-based — a single learned temperature
scalar, `einsum` against whatever text features arrive (`core/models/heads.py:113-137`).
No parameter is indexed by class count, so the head transfers unchanged. The
merge loop above will raise if that ever stops being true.

The KIP-**off** MSAD arm consumes `checkpoint_last.pt` directly — it needs no
prep, because a KIP-off model is exactly the shape the PreVAD run saved.

### 7.3 Stage M — the A/B from the shared trunk

Both arms start from the same PreVAD trunk. The only difference is KIP. Run three
seeds to match Phase A's evidence bar.

```bash
%%bash
for S in 2024 2025 2026; do
  DST="$KATVAD_OUTPUT_ROOT/MSAD_ncc_pv_s$S"
  TRUNK="$KATVAD_OUTPUT_ROOT/PreVAD/stage2_kip_off"

  # --- KIP-on: grafted trunk, flow consumed
  python -m core.train \
    --data-dir "$KATVAD_DATA_ROOT/MSAD" \
    --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
    --flow-dir "$KATVAD_CACHE_ROOT/flow/v1/MSAD" \
    --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
    --init-weights "$TRUNK/checkpoint_last_withkip.pt" \
    --output-dir "$DST/stage2_kip_on" \
    --set data.dataset=MSAD-full --set kip.enabled=true --set train.seed=$S

  # --- KIP-off: same trunk, no graft, no flow
  python -m core.train \
    --data-dir "$KATVAD_DATA_ROOT/MSAD" \
    --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
    --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
    --init-weights "$TRUNK/checkpoint_last.pt" \
    --output-dir "$DST/stage2_kip_off" \
    --set data.dataset=MSAD-full --set kip.enabled=false --set train.seed=$S
done
```

Then evaluate every arm on both benchmarks:

```bash
%%bash
for S in 2024 2025 2026; do
  DST="$KATVAD_OUTPUT_ROOT/MSAD_ncc_pv_s$S"
  for ARM in kip_on kip_off; do
    [ "$ARM" = "kip_off" ] && EXTRA="--set kip.enabled=false" || EXTRA="--set kip.enabled=true"

    python -m core.evaluate \
      --ckpt "$DST/stage2_$ARM/checkpoint_last.pt" $EXTRA \
      --set data.dataset=MSAD-full \
      --data-dir "$KATVAD_DATA_ROOT/MSAD" \
      --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
      --output-dir "$DST/eval_${ARM}_msad" --save-scores

    python -m core.evaluate \
      --ckpt "$DST/stage2_$ARM/checkpoint_last.pt" $EXTRA \
      --set data.dataset=DoTA \
      --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
      --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
      --output-dir "$DST/eval_${ARM}_dota" --save-scores
  done
done
```

> **Set `EXTRA` in the same cell that uses it.** An unset `EXTRA` silently
> evaluates a KIP-off checkpoint as KIP-on — the exact defect found in
> `collab/DoTA_ncc/evaluate_s2025_s2026.py` (`RESULTS_PHASE_A.md` §8). Here
> `strict=True` in `core/inference.py:64` would raise rather than corrupt, but
> do not lean on that.

**Cost:** 6 MSAD training runs (~500 steps each) + 12 evals, on top of stage P.
Small next to the 13,360-step trunk.

### 7.4 Why PreVAD-internal KIP-on stays out of reach

Recorded so the question is not reopened every few weeks. `L_KIP_rec` regresses
`e_O`, the cached RAFT embedding, which requires pixels. PreVAD ships none, and
`require_flow = cfg.kip.enabled` makes this a hard stop rather than a silent
degradation — `kip.enabled=true` without a flow cache raises `FileNotFoundError`
(`core/data/dataset.py:107`).

**Do not** "work around" it with `require_flow=False`. That zero-fills `e_O` and
`L_KIP_rec` then trains the PMG head to predict zeros — a wrong run that still
produces a checkpoint.

`LaGoVAD-PreVAD/data_download/` is a **collection** toolkit — the scripts the
authors used to *build* PreVAD — not a manifest that replays it.
`annotations/data_sources.csv` (35,279 rows) is the closest thing, and it does
not cover everything:

| `source` | rows | has a URL | Re-downloadable? |
|---|---|---|---|
| Existing video-text datasets | 20,092 | 18,460 | **Mostly yes** — VIDAL-10M (8,246), VATEX (5,299), VALOR-32K (4,915) carry direct YouTube URLs. MSR-VTT (1,632) has none. |
| China Expressway Camera | 3,800 | 3,800 | **No.** URL is a live portal; ids are wall-clock captures (`…_2024-09-19-12-08-01`). The scripts record streams **live** and cannot fetch September 2024. |
| bilibili long video | 3,647 | 3,647 | Yes — `yt-dlp` on the BVid, trimmed by offsets in the id. Region blocks apply. |
| YouTube video | 2,743 | 2,743 | Yes |
| YouTube long video | 1,487 | 1,487 | Yes |
| YouTube streaming | 1,464 | 1,464 | **Partly** — archived live streams; many gone. |
| bilibili video | 1,116 | 1,116 | Yes |
| RWF-2000 dataset | 930 | 0 | Separate dataset, request access. |

**~28,800 rows (82 %) carry a platform URL**; after multi-year link rot, region
blocks and dead stream archives, a realistic yield is **50–70 %** — and the 3,800
permanently lost ones are all highway cameras, i.e. exactly the traffic/motion
clips KIP cares about.

Two toolkit defects, recorded in case anyone tries anyway:

- **The `url` column is truncated for 3,974 YouTube rows.** Ids containing `-`
  are cut at the first dash: `first_person_walking-AHU0WncXt-c-1869.2-1879.0`
  yields `watch?v=AHU0WncXt` (9 chars) instead of `AHU0WncXt-c`. Rebuild the id
  from `video_id`.
- **`youtube/download_videos.py:70` wraps past one hour.** The trim range uses
  `time.strftime("%M:%S", ...)`, so a 5,113 s offset becomes `25:13`. Every
  long-video row past 3,600 s downloads the wrong segment, silently.

**And the download is the easy half.** Self-made flow cannot pair with the
released features: `yt-dlp --download-sections` snaps to keyframes, a re-download
years later is a different transcode at a possibly different fps, and the
`-Scene-NNN` ids imply an unpublished shot-detection pass. Lengths will differ,
and `core/data/dataset.py:112` raises on the mismatch. Any recovered subset would
need **both** features and flow re-extracted into a self-consistent cache, and a
KIP-on arm on a 60 % subset is not comparable to a KIP-off arm on the full
release — different training sets, uninterpretable Δ.

If the authors do send raw video, the picture changes and §7.1–7.3 become a
warm-up rather than the whole plan. Emailing them costs one message; it is step 1
of §10 for exactly that reason.

## 8. Decision: what conditions the text branch?

PreVAD is the first dataset we have with per-video `descriptions`, and
definition-conditioning is LaGoVAD's central claim. Two choices:

| | **A — class-name verbalizer** (fix G3 only) | **B — per-video descriptions** (fix G3 + G4) |
|---|---|---|
| Text per sample | one sampled definition sentence for the class | the video's own description |
| Faithfulness to LaGoVAD's PreVAD training | partial | **high** |
| Comparable to our MSAD/DoTA arms | **yes** — same conditioning everywhere | no — a new variable enters |
| Code | `definitions.py` only | + `DVSFeatureDataset` caption field, + `train.py` caption path, + DVS must decide what a *synthesized* clip's caption is |
| Risk | under-trains the open-world claim | the synthesis question is genuinely unsolved: which description survives when three clips are spliced? |

**Recommendation: A first, B as a follow-up run.** A is a clean re-use of the
existing pipeline and gives a like-for-like comparison against everything already
measured. B changes what the caption branch and `L_neg` see, and should be
measured as its own arm — not folded into the first PreVAD result.

Whichever is chosen, `loss.captions_from_definitions` is the switch that decides
whether the caption branch and `L_neg` are active at all; the baseline runs them
on PreVAD, so turn it **on** for a paper-comparable run.

---

## 9. Evaluation and what to compare against what

Five arms exist once §7 is done. Each benchmark keeps **its own** pooling
protocol (lesson 12/C12):

| Benchmark | Normal share of test | `--score-norm` |
|---|---|---|
| PreVAD test | 49.9 % | `auto` → raw |
| MSAD-full | 50.4 % | `auto` → raw |
| DoTA | 0.21 % | `auto` → **min-max** |

Never put raw-pooled and min-max-pooled numbers in one column — that is how a
released checkpoint reads as chance (`RESULTS_DOTA.md`).

### The comparisons that matter, in order

| # | Comparison | Answers |
|---|---|---|
| **C1** | PreVAD-trunk **KIP-on** vs PreVAD-trunk **KIP-off**, on DoTA, paired bootstrap, 3 seeds | **The headline.** Does the +0.09 survive a properly pretrained baseline? |
| **C2** | C1's Δ vs Phase A's Δ (+0.0915 ± 0.0088, MSAD-only trunks) | Additive with scale, or a substitute for it? |
| **C3** | PreVAD-trunk KIP-off vs MSAD-only KIP-off, on DoTA | What does PreVAD pretraining buy on its own? Sets the bar C1 has to clear. |
| **C4** | PreVAD-trunk arms vs `best.ckpt`, on MSAD-full | Recipe reproduction — the first time we run their objective on their data (§0). |
| **C5** | Gate P0 (§4.5) | Feature compatibility. Blocks everything; run first. |

C1 and C3 share the same eval, so the three-way table
{MSAD-only-off, PreVAD-off, PreVAD-on} on DoTA is the deliverable. Report the
**paired Δ with its CI**, never an arm alone — both DoTA arms drifted together
across seeds in Phase A while the gap held.

### Framing, stated up front in whatever report follows

- Everything in `RESULTS_{MSAD,DOTA,NCC,PHASE_A,ARM4_PROBE}.md` uses MSAD-only
  trunks. The new arms are not drop-in replacements for those numbers; they are a
  second, better-controlled measurement of the same quantity.
- The trunk is **n=1** (§7.1). Seed variation covers stage 2 only.
- The paper's printed numbers (MSAD 90.41, DoTA 62.60, TAD 89.56) are context,
  not gates. Lesson 8b: gate against the released checkpoint, which cannot reach
  them either.
- Per lesson 14, a disappointing C1 is **not** licence to tune KIP. The mechanism
  is still unexplained; tuning against either delta is fitting the benchmark.

---

## 10. Order of work

| # | Task | Blocking? | Cost |
|---|---|---|---|
| 1 | Email the authors for raw video — **day one**, the lead time is the point | no | one message |
| ~~2~~ | ~~`.gitignore` += `PreVAD/`~~ **DONE** — already present | no | done |
| 0 | Install + **verify** the environment (§4.0.1) — **DONE 2026-08-25**, repeat every VM | **yes** | minutes, every VM |
| ~~3a~~ | ~~Stage `train.csv` + `test.csv` into `$PREVAD_DATA` (§4.1.1)~~ **DONE** | **yes** | done |
| ~~3b~~ | ~~Verify the zip on Drive (§4.2)~~ **DONE** | **yes** | done |
| 3c | Unzip to `/content` + run the verifier (§4.3) — **every session**, ~15 s | **yes** | every session |
| ~~4~~ | ~~§4.4 sanity checks — dim 512, L2 norm, interval~~ **DONE** | **yes** | done |
| ~~5~~ | ~~Fix **G3** (35-class definitions)~~ **DONE 2026-08-24** | **yes** | done |
| ~~6~~ | ~~Write `core/data/prevad.py` (**G1**, **G2**) + tests~~ **DONE 2026-08-24** — 38 tests, 322 green, full §11 gate clean | **yes** | done |
| ~~7~~ | ~~Build label files (§5)~~ **DONE 2026-08-25** | **yes** | done |
| 8 | **Gate P0** — `best.ckpt` on PreVAD test (§4.5) — **NOT RECORDED**, and Stage P started without it | **yes** | ~1 GPU-hour |
| ~~9~~ | ~~KNN cache (§6)~~ **DONE 2026-08-25** | yes | done |
| 10 | **Stage P** — PreVAD KIP-off trunk (§7.1) — **IN PROGRESS**, epoch 26/40 at 2026-08-25 | **yes** | 13,360 batches |
| 11 | **Stage T** — graft KIP onto the trunk (§7.2) | **yes** | one cell |
| 12 | **Stage M** — 6 MSAD runs, 3 seeds × {on, off} (§7.3) | — | ~500 steps each |
| 13 | 12 evals + C1–C4 (§9) | — | ~4 GPU-hours |

Steps 5–8 are the real gate. If P0 fails, 9–13 are wasted.

**Where this actually stands (2026-08-25).** Steps 2–7 and 9 are done; step 10 is
running at epoch 26/40; **step 8 was skipped and is still owed.** The ordering
above exists because P0 is what makes Stage P worth running — a trunk trained on
features that turn out incompatible is 13,360 batches of nothing. Run P0 while
Stage P finishes: it needs the same label files and costs ~1 GPU-hour, and its
verdict decides whether §7.2 onward is meaningful. Do not report any Stage P
result until P0 has a recorded number.

**Every later session repeats step 3c only** (§7.1.2's checklist wraps 0 + 3c +
the resume into one paste).

Two natural stopping points, both defensible: after **step 8** you know the
release is usable; after **step 10** you have the recipe reproduction (C4) even
if the A/B never runs.

---

## Rules carried in from elsewhere

- `LaGoVAD-PreVAD/` is **read-only**. All code goes in `core/`.
- Every new entry point ships an `argparse` CLI (§14.6 of `CLAUDE.md`).
- Quality gate before commit: `ruff` / `mypy` / `bandit` / `pycycle` / `pyright`.
- ~~After `core/data/prevad.py` lands, add a lesson (CLAUDE.md §7)~~ **DONE** —
  lessons **18/C18** (widest-case label formats; the multi-span + clamp case)
  and **19/C19** (a graceful fallback hiding a missing taxonomy).
