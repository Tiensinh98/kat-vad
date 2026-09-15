# Gate D0 — DADA-2000 **original**, frame linear probe

**Run 2026-09-15 (Colab VM clock; 2026-09-16 01:21 local), branch `main`.**
Runbook `core/docs/DADA_ORIGIN_PHASE1.md` · plan
`.project/plans/katvad-dada-original-phase1-gate-d0.md` · notebook
`colab/DADA2000Origin/phase_1.ipynb` (outputs preserved in the file).

---

## 0. Verdict — **PASS**

**`auc_macro` = 0.6518** on the pipeline's own label convention, against a
pre-registered bar of **≥ 0.60 PASS / < 0.55 STOP**.

| arm | `auc_macro` | `auc_micro` | AP | AP baseline | clips | folds |
|---|---:|---:|---:|---:|---:|---:|
| **`d0_frac`** — fraction mapping, what Phase 4 will use | **0.6518** | 0.6207 | 0.4169 | 0.3079 | 400 | 5 |
| `d0_abs` — absolute indices, sanity arm | 0.6492 | 0.6214 | 0.4359 | 0.3252 | 400 | 5 |

`|Δ auc_macro| = 0.0026`, against the pre-registered **0.01** above which both
numbers would have been void. **Both arms clear 0.60 independently.**

### 0.1 The comparison that matters

Same frozen CLIP, same `no_center_crop` transform, same probe:

| corpus | frame probe `auc_macro` | clips |
|---|---:|---:|
| DoTA — the held-out benchmark | **0.6708** | 1,397 |
| **DADA-2000 original (this run)** | **0.6518** | 400 |
| DADA-2000 trimmed archive | 0.5228 | 383 |

**0.6518 sits 0.019 below DoTA and 0.129 above the trimmed archive.** The
CRITICAL verdict carried by `outputs/EDA/DADA2000/eda_report.md` — *"the features
carry no frame-level signal"* — is confirmed to be **a property of that
degenerate corpus, not of frozen CLIP**, exactly as the plan predicted from
DoTA's 0.6708. The EDA's own verdict line here reads INFO, not CRITICAL:

> *The features DO carry a frame-level signal … the deficit is SUPERVISION, not
> representation. A better head or denser labels can close it; a different
> backbone is not required.*

### 0.2 What this does NOT license

**A probe is a ceiling, not a promise.** It says a linear head *could* localize
on these features under full supervision. It says nothing about whether WS-MIL
training will: TAD had no representation problem and still collapsed into a clip
classifier (`TAD_SETUP.md` §15.1, lesson C14's family). **D0 licenses Phase 2, not
a claim.**

---

## 1. Population

- **400 clips**, seed **2024**, type-stratified — **52/52 types covered**.
- **15,961 sampled frames** at stride 8; **4,915 positive (30.8 %)** under
  `d0_frac` (5,190 / 32.5 % under `d0_abs` — see §3).
- Feature cache `clip/DADA2000_orig`, **32 MB**, coverage **400/400, 0 missing**.
- Transform **`no_center_crop`** (anisotropic resize to 224², lessons C2/C13).
- Extraction: 10 shards × 40 clips, frames deleted after each; free disk never
  dropped below 53 GiB.

### 1.1 `clip_linear_probe` did not run — expected

```
ran: false   reason: "every clip has the same clip-level label"
```

The original release has **zero normal videos**, so there is no clip-level target.
**This is the point of the corpus, not a defect:** there is no clip-level
shortcut for a model to find, unlike the trimmed archive (clip oracle 0.9086,
lesson C28) and TAD (0.9226). Do not "fix" it by importing `0_Normal_Driving` —
that pool is ~3× longer and *inverts* the length leak (0.8105 → 0.8539).

### 1.2 Feature-space facts worth carrying to Phase 2

- **Temporal autocorrelation is very high**: mean cosine **0.968** at lag 1,
  0.933 at lag 8. Consecutive sampled frames are nearly identical, so the
  *effective* sample size is far below 15,961 — the probe's grouped-by-clip CV is
  what keeps this honest, and any future frame-level split must group the same way.
- **Between-clip / within-clip feature variance = 1.97.** The embedding encodes
  scene identity more than dynamics. Relevant to KIP's premise, and to any claim
  that CLIP features carry motion.
- 420 of 512 dimensions hold 90 % of the variance — not a degenerate subspace.

---

## 2. P1 re-run at n = 400 — **supersedes Phase 0's n = 30**

```
P1 at n=400: 397 exact (99.2%), 397 within +-2 (99.2%), mean signed delta -0.28
largest frame deltas: [('t05_v040', -100), ('t10_v169', -10), ('t36_v002', -3), ...]
```

**99.2 % ≥ 95 %** → P1 passes at 13× Phase 0's sample, closing the Wilson-interval
gap that `DADA_ORIGIN_PHASE0.md` §8.1 flagged (29/30 had a 95 % lower bound of
≈ 0.83). **The original release is not trimmed.**

**Three clips are not exact, and they carry the whole deficit:**
−100 + −10 + −3 = **−113**, which reproduces the reported mean (−113/400 = −0.28)
exactly. 397 clips are off by **zero** frames.

`t05_v040` at **−100** is diagnosed in §2.2: the release itself trims that one
clip at the accident. It is the only clip in 400 whose labels are materially
misplaced by the pipeline's convention, and it is also the clip that produced the
worst-case number in §3.

### 2.1 The raw census was lost with the runtime — what survives, and what does not

`$D0` was `/content/d0`, VM-local. The runtime was recycled before
`meta.json` was copied, so **the per-clip frame census is gone**. This is a
process failure (the copy should have happened in the same cell that wrote this
report — now fixed), not a data loss that threatens the gate.

| artifact | where | status |
|---|---|---|
| **feature cache** `clip/DADA2000_orig`, 400 `.npy`, 32 MB | **Drive** | ✅ intact — the expensive part |
| `d0_{frac,abs}/eda_report.{json,md}` | repo `outputs/` | ✅ committed |
| notebook run log (the P1 line, the shard loop) | `colab/DADA2000Origin/phase_1.ipynb` | ✅ committed |
| **the 400-clip sample** | was `/content/d0_clips.json` | ✅ **regenerated** — `d0_clips_400.json`, this directory |
| `meta.json` per-clip census | VM | ❌ gone |
| `counts/*.json`, `dataset/d0_*/` label JSONs | VM | ❌ gone, rebuildable |

**The sample was recoverable because the pick is deterministic.**
`pick_probe.py --n 400 --seed 2024` against the committed
`data/DADA/dada标注.xlsx` reproduces it bit-for-bit — verified twice locally:
400 rows, 400 unique `(type, video)`, **52/52 types**, annotation `total frames`
median 314 (min 71, max 868). It is now committed as
**`d0_clips_400.json`**, so which clips were measured is no longer a fact that
lived only on a VM.

**What the lost census would have added** is each clip's on-disk count. The three
inexact clips and their deltas are recorded in the notebook output and in §2, and
`t05_v040` is fully reconstructed in §3 from the annotation alone. **`sampled_frames`
is still exactly recoverable without re-extracting anything** — it is the first
axis of each cached `.npy` — which pins `frames_on_disk` to a 7-frame window and
identifies any large mismatch unambiguously. §4 carries the recovery cell.

### 2.2 `t05_v040` — the −100 clip, diagnosed from the annotation

```
annotation:  total 482 frames, anomaly [285, 382]
on disk:     382 frames        <-- equals the anomaly's END, exactly
```

**The directory stops precisely where the annotated anomaly ends.** That is the
signature of a clip trimmed at the accident, i.e. the ~100 post-accident frames
are simply not in the release for this one clip. It is not an extraction failure:
the number lines up too well.

**Consequence: for this clip the `d0_abs` labels are right and `d0_frac`'s are
wrong.** The fraction convention rescales the window by `D/A = 0.79` and drops the
anomaly into the middle of the clip; the absolute convention puts it at the tail,
where it actually is. See §3.

---

## 3. The pre-registered frame-level check FIRED — and the threshold was mis-derived

```
frac vs abs labels: 256 clips differ, worst 17 frames, 311 frames total
WARNING worst-case disagreement 17 frames exceeds the pre-registered 2 (plan §3.2)
```

**This must be recorded, because the raw log says WARNING while §0 says PASS.**
Both are correct, and the reason is that the threshold was wrong, not the data.

**The mis-derivation.** Plan §3.2 argued that `sampled_frame_labels` *rounds* each
boundary while the absolute arm floors the start and ceils the end, so the two can
differ by "one frame per boundary = 2 per clip". **That holds only when the
on-disk frame count equals the annotation's.** The fraction mapping computes
`round((start/A) · L)` with `L = ceil(D/s)`; when `D ≠ A` the whole window is
rescaled by `D/A`, and the shift grows with the deficit:

```
shift ≈ (start / s) · |D − A| / A
```

**Reproduced exactly, from the annotation alone** (`t05_v040`: A = 482, D = 382,
s = 8, span 285–382):

```
L = ceil(382/8) = 48 sampled frames
frac window  (28, 38)  -> 10 positives      round((start/A)*L), round((end/A)*L)
abs  window  (35, 48)  -> 13 positives      start//s, ceil(end/s)
symmetric difference = 17 frames            <-- the reported worst, to the frame
```

So the "worst 17" is one clip, and it is the −100 clip. 253 of the 256 differing
clips differ by the rounding-only 1–2 frames; the tail is the three inexact clips
of §2. **And the fraction arm is the one that is wrong there** (§2.2): it places a
tail-of-clip anomaly in the middle.

**This is the second time in this plan that a threshold was written without
deriving its attainable range** — the same failure as C33, and the same one the
`≤ 1 frame` bar hit on 2026-09-15. Corrected bound, with the condition attached:

> **≤ 2 sampled frames for clips whose on-disk count matches the annotation;
> otherwise ~`(start/s)·|D−A|/A`.** The bar that decides anything is the AUC
> delta, and it was met with room to spare: **0.0026 vs 0.01**.

**Why the verdict stands.** The two label constructions disagree on 311 of 15,961
frame labels (**1.9 %**), concentrated in three clips, and the two independently
trained probes land 0.0026 apart — inside the pre-registered AUC bar and two
orders of magnitude below the 0.05-wide decision band. Neither arm's verdict
changes.

---

## 4. Reproducibility gaps to close (C17)

| gap | status |
|---|---|
| `commit` printed **empty** — `git rev-parse` returned nothing on the Drive mount | notebook cell now falls back to `UNKNOWN (<stderr>)` instead of a blank |
| `auc` column read **None** — the key is `auc_micro`, not `auc` | fixed; micro and AP recovered into §0 |
| `meta.json` not persisted before the runtime was recycled | **lost** (§2.1). The cell now copies it beside this report in the same step that writes the report |

None affects the number; all three affect whether the number can be cited later.

### 4.1 Recovering the census next session — no re-extraction

`sampled_frames` is the first axis of each cached `.npy`, and the cache is on
Drive. That pins `frames_on_disk` to `[8(L−1)+1, 8L]` and makes any large
mismatch unmistakable, which is all the P1 claim needs.

```python
import json, math, numpy as np, os
from pathlib import Path

CACHE = Path(os.environ['KATVAD_CACHE_ROOT']) / 'clip' / 'DADA2000_orig'
rows  = json.loads(Path('outputs/EDA/DADA2000Origin/d0_clips_400.json').read_text())

out, suspect = {}, []
for r in rows:
    vid = f"t{r['type']:02d}_v{r['video']:03d}"
    L = int(np.load(CACHE / f'{vid}.npy', mmap_mode='r').shape[0])
    expected_L = math.ceil(r['total'] / 8)          # if the counts matched
    out[vid] = {'type': r['type'], 'video': r['video'],
                'annotation_total_frames': r['total'], 'sampled_frames': L,
                'frames_on_disk_range': [8 * (L - 1) + 1, 8 * L],
                'annotation_span_frames': [r['start'], r['end']]}
    if L != expected_L:
        suspect.append((vid, r['total'], L, expected_L, 8 * (L - expected_L)))

Path('outputs/EDA/DADA2000Origin/d0_census_from_cache.json').write_text(
    json.dumps(out, indent=2))
print(f'{len(out)} clips; {len(suspect)} with a sampled length that cannot come '
      f'from the annotation count:')
for row in sorted(suspect, key=lambda x: x[-1]):
    print('   ', row)      # (vid, A, L_actual, L_expected, approx frame delta)
```

Clips whose `L` matches `ceil(A/8)` are exact to within ±7 frames; `t05_v040`
should appear with `L = 48` against an expected 61, i.e. ≈ −104 — consistent with
the measured −100.

---

## 5. Decision

**Gate D0 PASSES.** Proceed to **Phase 2** —
`.project/plans/katvad-dada-original-corpus.md` §5: build the T2 corpus
(W = 16, hop 8, `score_head_kernel` 9 → 3), sharded, then Phase 3's Gate W.

Before any `core/` code is written: `trace_call_path` on every symbol to be
touched, and its blast radius reported (plan §5.4, and `video_id_from_path`'s
5 CRITICAL hop-1 callers — do not touch it).
