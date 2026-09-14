# EDA — TAD

Data dir: `/content/drive/MyDrive/Thesis/data/TAD` · score head kernel **9** · MIL top-k pct **16** · sections: corpus, labels, protocol

Every number here is a property of the data on disk, not of a model.

## 0. Verdicts

| Level | Finding | Measurement | What to do |
|---|---|---|---|
| **CRITICAL** | Micro AUC here is mostly clip classification (lesson C12) | A constant-score-per-clip oracle scores micro AUC 0.9226 with zero localization; 99.7% of the positive/negative pairs span two clips. | Report auc_macro as the headline and print this oracle beside any micro number. Never place a micro number from this corpus next to a published frame-level AUC. |
| **CRITICAL** | Clip length alone predicts the label (lesson C28) | A constant-score-per-clip detector reading only the clip's frame count scores clip-level AUC 0.6940 and micro AUC 0.8968 — shorter clips are the abnormal ones. 21 normal clips (8,049 frames) fall outside the abnormal length range entirely. | Rebuild the corpus into fixed-length windows with the anomaly at a random offset. Until then, print this baseline beside every micro AUC and treat any arm that fails to beat it as unmeasured. |

## 1. Corpus shape

| | clips | abnormal | normal | frames |
|---|---:|---:|---:|---:|
| train | 410 | 200 | 210 | — |
| test | 100 | 60 | 40 | 11,045 |

Test positive frames: **800** (7.24% of sampled frames)
DVS dataset length (2 x abnormal train): **400** → **7 steps/epoch** at batch 64.

### 1.1 Clip-length distribution (sampled frames)

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test sampled | 100 | 110.45 | 8.9 | 25.8 | 43.0 | 76.0 | 450.0 | 3.0 | 1362.0 |
| train sampled | 410 | 137.76 | 10.0 | 24.0 | 41.0 | 104.8 | 675.0 | 5.0 | 1875.0 |
| raw (pre-stride) | 510 | 1055.93 | 74.5 | 192.8 | 328.5 | 786.0 | 5260.0 | 21.0 | 15000.0 |

### 1.2 Score-head receptive field (kernel 9) — lesson C27

- median clip length **43** sampled frames
- median fraction of a clip inside one output timestep: **20.9%**
- clips entirely inside the kernel: **6** (**6.0%**), holding **0.3%** of all test frames
- short-clip counts: `T<=3` → 1, `T<=5` → 3, `T<=9` → 6, `T<=13` → 9, `T<=17` → 13

### 1.3 MIL top-k floor (mil_topk_pct = 16)

- clips at `k = 1` (loss is a plain max): **35** (**35.0%**)
- median k: **2.0**
- k distribution: {'1': 35, '2': 19, '3': 14, '4': 8, '5': 2, '6': 1, '7': 1, '10': 3, '11': 3, '12': 1, '14': 2, '15': 1, '17': 2, '19': 1, '28': 3, '34': 1, '44': 1, '70': 1, '85': 1}

### 1.4 Subgroups by `class_name` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| Car Accident | 260 | 200 | 60 | 2,385 | 800 | 33.54% |
| Normal | 250 | 210 | 40 | 8,660 | 0 | 0.00% |

### 1.4 Subgroups by `split` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| test | 100 | 0 | 100 | 11,045 | 800 | 7.24% |
| train | 410 | 410 | 0 | 0 | 0 | — |

## 2. Label geometry

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| positives / test clip | 100 | 8.00 | 0.0 | 0.0 | 6.0 | 14.0 | 23.1 | 0.0 | 41.0 |
| positives / abnormal clip | 60 | 13.33 | 3.0 | 8.8 | 12.5 | 17.0 | 28.2 | 1.0 | 41.0 |
| positive frac. within abnormal | 60 | 0.33 | 0.2 | 0.2 | 0.3 | 0.4 | 0.5 | 0.1 | 0.8 |
| spans / abnormal clip | 60 | 1.17 | 1.0 | 1.0 | 1.0 | 1.0 | 2.0 | 1.0 | 3.0 |
| span length | 70 | 11.43 | 2.5 | 6.0 | 11.0 | 15.0 | 22.0 | 1.0 | 26.0 |

- abnormal clips with **exactly one** positive frame: **2**
- abnormal clips with <= 2 positive frames: **2**
- multi-span abnormal clips: **9** (lesson C18 applies if > 0)
- **abnormal clips whose window vanished: 0**

## 3. What the metric measures

### 3.1 Where the frames live

| clip kind | clips | frames | share of frames |
|---|---:|---:|---:|
| all_normal | 40 | 8,660 | 78.41% |
| mixed | 60 | 2,385 | 21.59% |
| all_positive | 0 | 0 | 0.00% |

### 3.2 The clip-level oracle (lesson C12)

A model emitting **one constant score per clip**, ranking clips perfectly and localizing nothing, scores:

- micro AUC **0.9226**, micro AP **0.3354**, macro AUC **0.5000** by construction
- 99.69% of positive/negative frame pairs span two clips (8,170,717 of 8,196,000); only 0.31% can be won by localization

**Print this oracle beside every micro AUC measured on this corpus.**

### 3.3 The clip-length leak (lesson C28)

A constant-score-per-clip detector whose **only** input is the clip's frame count — no pixels, no model — scores:

- clip-level AUC **0.6940** (**shorter** clips are the abnormal ones); micro AUC **0.8968**, micro AP **0.2668**, macro AUC **0.5000** by construction
- abnormal clip length T: median **36.0**, min 5.0, max **99.0**
- normal clip length T: median **139.0**, min 3.0, max 1362.0
- normal clips outside the abnormal length range entirely: **21** (8,049 frames)

**Any arm that does not beat this baseline is unmeasured.**

### 3.4 Per-clip AUC resolution

- two-class clips (the only ones `auc_macro` averages): **60**; single-class: 40

A clip with `p` positives and `n` negatives has `p*n` orderable pairs, so its AUC only takes values on a `1/(p*n)` grid. A coarse grid makes `auc_macro` honest but low-resolution — quote it with these counts.

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUC grid step 1/(pos*neg) | 60 | 0.01 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 |

- `--score-norm auto` resolves to **none** (normal-clip fraction 0.400 vs threshold 0.05)

