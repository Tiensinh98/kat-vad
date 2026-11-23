# EDA — Nexar_window

Data dir: `/content/nexar_eda/data` · score head kernel **9** · MIL top-k pct **5** · sections: corpus, labels, protocol

Every number here is a property of the data on disk, not of a model.

## 0. Verdicts

| Level | Finding | Measurement | What to do |
|---|---|---|---|
| **CRITICAL** | Micro AUC here is mostly clip classification (lesson C12) | A constant-score-per-clip oracle scores micro AUC 0.9156 with zero localization; 100.0% of the positive/negative pairs span two clips. | Report auc_macro as the headline and print this oracle beside any micro number. Never place a micro number from this corpus next to a published frame-level AUC. |

## 1. Corpus shape

| | clips | abnormal | normal | frames |
|---|---:|---:|---:|---:|
| train | 7,311 | 1,689 | 5,622 | — |
| test | 1,788 | 434 | 1,354 | 35,760 |

Test positive frames: **3,179** (8.89% of sampled frames)
DVS dataset length (2 x abnormal train): **3,378** → **53 steps/epoch** at batch 64.

### 1.1 Clip-length distribution (sampled frames)

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test sampled | 1,788 | 20.00 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| train sampled | 7,311 | 20.00 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| raw (pre-stride) | 9,099 | 1186.01 | 1076.0 | 1198.0 | 1206.0 | 1225.0 | 1247.0 | 420.0 | 1800.0 |

### 1.2 Score-head receptive field (kernel 9) — lesson C27

- median clip length **20** sampled frames
- median fraction of a clip inside one output timestep: **45.0%**
- clips entirely inside the kernel: **0** (**0.0%**), holding **0.0%** of all test frames
- short-clip counts: `T<=3` → 0, `T<=5` → 0, `T<=9` → 0, `T<=13` → 0, `T<=17` → 0

### 1.3 MIL top-k floor (mil_topk_pct = 5)

- clips at `k = 1` (loss is a plain max): **0** (**0.0%**)
- median k: **4.0**
- k distribution: {'4': 1788}

### 1.4 Subgroups by `split` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| test | 1,788 | 0 | 1,788 | 35,760 | 3,179 | 8.89% |
| train | 7,311 | 7,311 | 0 | 0 | 0 | — |

## 2. Label geometry

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| positives / test clip | 1,788 | 1.78 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 | 0.0 | 17.0 |
| positives / abnormal clip | 434 | 7.32 | 1.0 | 4.0 | 8.0 | 10.0 | 14.0 | 1.0 | 17.0 |
| positive frac. within abnormal | 434 | 0.37 | 0.1 | 0.2 | 0.4 | 0.5 | 0.7 | 0.1 | 0.8 |
| spans / abnormal clip | 434 | 1.00 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| span length | 434 | 7.32 | 1.0 | 4.0 | 8.0 | 10.0 | 14.0 | 1.0 | 17.0 |

- abnormal clips with **exactly one** positive frame: **36**
- abnormal clips with <= 2 positive frames: **73**
- multi-span abnormal clips: **0** (lesson C18 applies if > 0)
- **abnormal source clips whose window vanished: 0**
- windows of abnormal clips holding no positive frame: **1,354** — these are *correct negatives*, not vanished windows; producing them is the point of re-sharding

## 3. What the metric measures

### 3.1 Where the frames live

| clip kind | clips | frames | share of frames |
|---|---:|---:|---:|
| all_normal | 1,354 | 27,080 | 75.73% |
| mixed | 434 | 8,680 | 24.27% |
| all_positive | 0 | 0 | 0.00% |

### 3.2 The clip-level oracle (lesson C12)

A model emitting **one constant score per clip**, ranking clips perfectly and localizing nothing, scores:

- micro AUC **0.9156**, micro AP **0.3662**, macro AUC **0.5000** by construction
- 99.97% of positive/negative frame pairs span two clips (103,541,720 of 103,574,999); only 0.03% can be won by localization

**Print this oracle beside every micro AUC measured on this corpus.**

### 3.3 The clip-length leak (lesson C28)

A constant-score-per-clip detector whose **only** input is the clip's frame count — no pixels, no model — scores:

- clip-level AUC **0.5000** (**longer** clips are the abnormal ones); micro AUC **0.5000**, micro AP **0.0889**, macro AUC **0.5000** by construction
- abnormal clip length T: median **20.0**, min 20.0, max **20.0**
- normal clip length T: median **20.0**, min 20.0, max 20.0
- normal clips outside the abnormal length range entirely: **0** (0 frames)

**Any arm that does not beat this baseline is unmeasured.**

### 3.4 Per-clip AUC resolution

- two-class clips (the only ones `auc_macro` averages): **434**; single-class: 1,354

A clip with `p` positives and `n` negatives has `p*n` orderable pairs, so its AUC only takes values on a `1/(p*n)` grid. A coarse grid makes `auc_macro` honest but low-resolution — quote it with these counts.

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUC grid step 1/(pos*neg) | 434 | 0.02 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.1 |

- `--score-norm auto` resolves to **none** (normal-clip fraction 0.757 vs threshold 0.05)

