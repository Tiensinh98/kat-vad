# EDA — DADA2000

Data dir: `/content/drive/MyDrive/Thesis/data/DADA2000_w24s1` · score head kernel **9** · MIL top-k pct **16** · sections: corpus, labels, protocol, features, scores

Every number here is a property of the data on disk, not of a model.

## 0. Verdicts

| Level | Finding | Measurement | What to do |
|---|---|---|---|
| **HIGH** | MIL top-k degenerates to a plain max | 100.0% of clips get k = 1 at mil_topk_pct = 16: one supervised frame per clip per step. | Lower loss.mil_topk_pct, or accept that supervision is clip-level here and stop reading frame-level claims into it. |
| **HIGH** | Abnormal source clips with an all-zero label vector | 3 source clips declared abnormal in meta.json have no positive sampled frame anywhere; every metric counts them as normal. | Exclude them at scoring time (core/docs/DADA_SETUP.md §5.1). Do NOT 'fix' this with --strict: that flag raises, it does not repair. |
| **CRITICAL** | Micro AUC here is mostly clip classification (lesson C12) | A constant-score-per-clip oracle scores micro AUC 0.8965 with zero localization; 100.0% of the positive/negative pairs span two clips. | Report auc_macro as the headline and print this oracle beside any micro number. Never place a micro number from this corpus next to a published frame-level AUC. |
| **INFO** | Frozen features are dominated by scene identity | Between-clip / within-clip feature variance = 5.7. | Temporal dynamics are a small part of this embedding; a head reading them is working against the representation's own scale. |
| **CRITICAL** | The features carry no frame-level signal | Even a supervised linear probe reaches only auc_macro 0.5146, while the clip-level probe reaches 0.5869. | No head on these features can localize. Frame-level work on this corpus needs a different backbone or a finer stride -- not another KIP variant (RESULTS_DADA.md §6, §10-C). |

## 1. Corpus shape

| | clips | abnormal | normal | frames |
|---|---:|---:|---:|---:|
| train | 4,631 | 1,481 | 3,150 | — |
| test | 1,161 | 406 | 755 | 27,864 |

Test positive frames: **5,016** (18.00% of sampled frames)
DVS dataset length (2 x abnormal train): **2,962** → **47 steps/epoch** at batch 64.

### 1.1 Clip-length distribution (sampled frames)

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test sampled | 1,161 | 24.00 | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |
| train sampled | 4,631 | 24.00 | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |
| raw (pre-stride) | 5,792 | 124.34 | 36.6 | 59.0 | 89.5 | 167.0 | 316.0 | 24.0 | 579.0 |

### 1.2 Score-head receptive field (kernel 9) — lesson C27

- median clip length **24** sampled frames
- median fraction of a clip inside one output timestep: **37.5%**
- clips entirely inside the kernel: **0** (**0.0%**), holding **0.0%** of all test frames
- short-clip counts: `T<=3` → 0, `T<=5` → 0, `T<=9` → 0, `T<=13` → 0, `T<=17` → 0

### 1.3 MIL top-k floor (mil_topk_pct = 16)

- clips at `k = 1` (loss is a plain max): **1,161** (**100.0%**)
- median k: **1.0**
- k distribution: {'1': 1161}

### 1.4 Subgroups by `fault_label` (3 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| 0_Non_Ego_Fault | 1,377 | 1,078 | 299 | 7,176 | 2,924 | 40.75% |
| 0_Normal_Driving | 3,341 | 2,692 | 649 | 15,576 | 0 | 0.00% |
| 1_Ego_Fault | 1,074 | 861 | 213 | 5,112 | 2,092 | 40.92% |

### 1.4 Subgroups by `ego_involve` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| False | 4,718 | 3,770 | 948 | 22,752 | 2,924 | 12.85% |
| True | 1,074 | 861 | 213 | 5,112 | 2,092 | 40.92% |

### 1.4 Subgroups by `class_name` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| CarAccident | 2,451 | 1,939 | 512 | 12,288 | 5,016 | 40.82% |
| Normal | 3,341 | 2,692 | 649 | 15,576 | 0 | 0.00% |

### 1.4 Subgroups by `split` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| test | 1,161 | 0 | 1,161 | 27,864 | 5,016 | 18.00% |
| train | 4,631 | 4,631 | 0 | 0 | 0 | — |

## 2. Label geometry

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| positives / test clip | 1,161 | 4.32 | 0.0 | 0.0 | 0.0 | 8.0 | 21.0 | 0.0 | 24.0 |
| positives / abnormal clip | 406 | 12.35 | 2.0 | 7.0 | 12.0 | 18.0 | 24.0 | 1.0 | 24.0 |
| positive frac. within abnormal | 406 | 0.51 | 0.1 | 0.3 | 0.5 | 0.8 | 1.0 | 0.0 | 1.0 |
| spans / abnormal clip | 406 | 1.00 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| span length | 406 | 12.35 | 2.0 | 7.0 | 12.0 | 18.0 | 24.0 | 1.0 | 24.0 |

- abnormal clips with **exactly one** positive frame: **16**
- abnormal clips with <= 2 positive frames: **26**
- multi-span abnormal clips: **0** (lesson C18 applies if > 0)
- **abnormal source clips whose window vanished: 3**
- windows of abnormal clips holding no positive frame: **103** — these are *correct negatives*, not vanished windows; producing them is the point of re-sharding

<details><summary>Vanished-window ids (exclude these at scoring)</summary>

```
0_Non_Ego_Fault__type13_vid002
0_Non_Ego_Fault__type1_vid017
0_Non_Ego_Fault__type6_vid112
```
</details>

## 3. What the metric measures

### 3.1 Where the frames live

| clip kind | clips | frames | share of frames |
|---|---:|---:|---:|
| all_normal | 755 | 18,120 | 65.03% |
| mixed | 370 | 8,880 | 31.87% |
| all_positive | 36 | 864 | 3.10% |

### 3.2 The clip-level oracle (lesson C12)

A model emitting **one constant score per clip**, ranking clips perfectly and localizing nothing, scores:

- micro AUC **0.8965**, micro AP **0.5148**, macro AUC **0.5000** by construction
- 99.97% of positive/negative frame pairs span two clips (114,565,694 of 114,605,568); only 0.03% can be won by localization

**Print this oracle beside every micro AUC measured on this corpus.**

### 3.3 The clip-length leak (lesson C28)

A constant-score-per-clip detector whose **only** input is the clip's frame count — no pixels, no model — scores:

- clip-level AUC **0.5000** (**longer** clips are the abnormal ones); micro AUC **0.5000**, micro AP **0.1800**, macro AUC **0.5000** by construction
- abnormal clip length T: median **24.0**, min 24.0, max **24.0**
- normal clip length T: median **24.0**, min 24.0, max 24.0
- normal clips outside the abnormal length range entirely: **0** (0 frames)

**Any arm that does not beat this baseline is unmeasured.**

### 3.4 Per-clip AUC resolution

- two-class clips (the only ones `auc_macro` averages): **370**; single-class: 791

A clip with `p` positives and `n` negatives has `p*n` orderable pairs, so its AUC only takes values on a `1/(p*n)` grid. A coarse grid makes `auc_macro` honest but low-resolution — quote it with these counts.

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUC grid step 1/(pos*neg) | 370 | 0.01 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

- `--score-norm auto` resolves to **none** (normal-clip fraction 0.650 vs threshold 0.05)

## 4. The cached frozen-CLIP features

`/content/drive/MyDrive/Thesis/cache/clip/DADA2000_s1` — 1,161 of 1,161 test clips found

- 27,864 frames x 512 dims; mean L2 norm 9.90
- dimensions holding 90 % of the variance: **419** of 512

### 4.1 Temporal autocorrelation (cosine between frame t and t+lag)

| lag (sampled frames) | lag1 | lag2 | lag3 | lag4 | lag5 | lag6 | lag7 | lag8 |
|---|---|---|---|---|---|---|---|---|
| mean cosine | 0.989 | 0.982 | 0.977 | 0.974 | 0.970 | 0.968 | 0.966 | 0.964 |

A value near 1.0 at lag 1 means consecutive sampled frames are nearly identical, i.e. the stride is finer than the content changes.

- between-clip / within-clip **feature** variance: **5.66** (high = the embedding encodes scene identity more than dynamics)

### 4.2 Supervised linear probe — the representation ceiling

| probe | AUC | macro AUC | AP | AP baseline | folds | units |
|---|---:|---:|---:|---:|---:|---:|
| frame-level | 0.5564 | 0.5146 | 0.2110 | 0.1800 | 5 | 27,864 frames |
| clip-level (mean-pooled) | 0.5869 | — | 0.4420 | 0.3497 | 5 | 1,161 clips |

Grouped cross-validation by clip, so no clip's frames score themselves. This is the *supervised ceiling* of these features: a trained arm cannot be expected to beat it, and a large gap below it is a supervision problem, not a representation problem.

