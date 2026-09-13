# EDA — DADA2000

Data dir: `/content/drive/MyDrive/Thesis/data/DADA2000_w32s2` · score head kernel **9** · MIL top-k pct **16** · sections: corpus, labels, protocol, features, scores

Every number here is a property of the data on disk, not of a model.

## 0. Verdicts

| Level | Finding | Measurement | What to do |
|---|---|---|---|
| **HIGH** | Abnormal clips with an all-zero label vector | 3 clips declared abnormal in meta.json have no positive sampled frame; every metric counts them as normal. | Exclude them at scoring time (core/docs/DADA_SETUP.md §5.1). Do NOT 'fix' this with --strict: that flag raises, it does not repair. |
| **CRITICAL** | Micro AUC here is mostly clip classification (lesson C12) | A constant-score-per-clip oracle scores micro AUC 0.9766 with zero localization; 99.9% of the positive/negative pairs span two clips. | Report auc_macro as the headline and print this oracle beside any micro number. Never place a micro number from this corpus next to a published frame-level AUC. |

## 1. Corpus shape

| | clips | abnormal | normal | frames |
|---|---:|---:|---:|---:|
| train | 2,811 | 253 | 2,558 | — |
| test | 760 | 57 | 703 | 24,320 |

Test positive frames: **719** (2.96% of sampled frames)
DVS dataset length (2 x abnormal train): **506** → **8 steps/epoch** at batch 64.

### 1.1 Clip-length distribution (sampled frames)

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test sampled | 760 | 32.00 | 32.0 | 32.0 | 32.0 | 32.0 | 32.0 | 32.0 | 32.0 |
| train sampled | 2,811 | 32.00 | 32.0 | 32.0 | 32.0 | 32.0 | 32.0 | 32.0 | 32.0 |
| raw (pre-stride) | 3,571 | 225.55 | 75.5 | 142.5 | 209.0 | 299.0 | 415.0 | 63.0 | 579.0 |

### 1.2 Score-head receptive field (kernel 9) — lesson C27

- median clip length **32** sampled frames
- median fraction of a clip inside one output timestep: **28.1%**
- clips entirely inside the kernel: **0** (**0.0%**), holding **0.0%** of all test frames
- short-clip counts: `T<=3` → 0, `T<=5` → 0, `T<=9` → 0, `T<=13` → 0, `T<=17` → 0

### 1.3 MIL top-k floor (mil_topk_pct = 16)

- clips at `k = 1` (loss is a plain max): **0** (**0.0%**)
- median k: **2.0**
- k distribution: {'2': 760}

### 1.4 Subgroups by `fault_label` (3 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| 0_Non_Ego_Fault | 171 | 134 | 37 | 1,184 | 438 | 36.99% |
| 0_Normal_Driving | 3,244 | 2,544 | 700 | 22,400 | 0 | 0.00% |
| 1_Ego_Fault | 156 | 133 | 23 | 736 | 281 | 38.18% |

### 1.4 Subgroups by `ego_involve` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| False | 3,415 | 2,678 | 737 | 23,584 | 438 | 1.86% |
| True | 156 | 133 | 23 | 736 | 281 | 38.18% |

### 1.4 Subgroups by `class_name` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| CarAccident | 327 | 267 | 60 | 1,920 | 719 | 37.45% |
| Normal | 3,244 | 2,544 | 700 | 22,400 | 0 | 0.00% |

### 1.4 Subgroups by `split` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| test | 760 | 0 | 760 | 24,320 | 719 | 2.96% |
| train | 2,811 | 2,811 | 0 | 0 | 0 | — |

## 2. Label geometry

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| positives / test clip | 760 | 0.95 | 0.0 | 0.0 | 0.0 | 0.0 | 11.0 | 0.0 | 22.0 |
| positives / abnormal clip | 57 | 12.61 | 5.0 | 10.0 | 12.0 | 16.0 | 21.0 | 2.0 | 22.0 |
| positive frac. within abnormal | 57 | 0.39 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 0.1 | 0.7 |
| spans / abnormal clip | 57 | 1.00 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| span length | 57 | 12.61 | 5.0 | 10.0 | 12.0 | 16.0 | 21.0 | 2.0 | 22.0 |

- abnormal clips with **exactly one** positive frame: **0**
- abnormal clips with <= 2 positive frames: **1**
- multi-span abnormal clips: **0** (lesson C18 applies if > 0)
- **abnormal clips whose window vanished: 3**

<details><summary>Vanished-window ids (exclude these at scoring)</summary>

```
0_Non_Ego_Fault__type11_vid067__w000
0_Non_Ego_Fault__type3_vid001__w000
1_Ego_Fault__type18_vid006__w000
```
</details>

## 3. What the metric measures

### 3.1 Where the frames live

| clip kind | clips | frames | share of frames |
|---|---:|---:|---:|
| all_normal | 703 | 22,496 | 92.50% |
| mixed | 57 | 1,824 | 7.50% |
| all_positive | 0 | 0 | 0.00% |

### 3.2 The clip-level oracle (lesson C12)

A model emitting **one constant score per clip**, ranking clips perfectly and localizing nothing, scores:

- micro AUC **0.9766**, micro AP **0.3942**, macro AUC **0.5000** by construction
- 99.92% of positive/negative frame pairs span two clips (16,956,386 of 16,969,119); only 0.08% can be won by localization

**Print this oracle beside every micro AUC measured on this corpus.**

### 3.3 The clip-length leak (lesson C28)

A constant-score-per-clip detector whose **only** input is the clip's frame count — no pixels, no model — scores:

- clip-level AUC **0.5000** (**longer** clips are the abnormal ones); micro AUC **0.5000**, micro AP **0.0296**, macro AUC **0.5000** by construction
- abnormal clip length T: median **32.0**, min 32.0, max **32.0**
- normal clip length T: median **32.0**, min 32.0, max 32.0
- normal clips outside the abnormal length range entirely: **0** (0 frames)

**Any arm that does not beat this baseline is unmeasured.**

### 3.4 Per-clip AUC resolution

- two-class clips (the only ones `auc_macro` averages): **57**; single-class: 703

A clip with `p` positives and `n` negatives has `p*n` orderable pairs, so its AUC only takes values on a `1/(p*n)` grid. A coarse grid makes `auc_macro` honest but low-resolution — quote it with these counts.

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUC grid step 1/(pos*neg) | 57 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

- `--score-norm auto` resolves to **none** (normal-clip fraction 0.925 vs threshold 0.05)

## 4. The cached frozen-CLIP features

`/content/drive/MyDrive/Thesis/cache/clip/DADA2000_s2` — 0 of 760 test clips found, **760 missing**

### 4.1 Temporal autocorrelation (cosine between frame t and t+lag)

| lag (sampled frames) | lag1 | lag2 | lag3 | lag4 | lag5 | lag6 | lag7 | lag8 |
|---|---|---|---|---|---|---|---|---|
| mean cosine | — | — | — | — | — | — | — | — |

A value near 1.0 at lag 1 means consecutive sampled frames are nearly identical, i.e. the stride is finer than the content changes.

### 4.2 Supervised linear probe — the representation ceiling

Probe did not run: not requested.

