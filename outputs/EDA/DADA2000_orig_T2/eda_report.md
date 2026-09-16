# EDA — DADA2000_orig

Data dir: `/content/drive/MyDrive/Thesis/data/DADA2000_orig` · score head kernel **3** · MIL top-k pct **16** · sections: corpus, labels, protocol, features, scores

Every number here is a property of the data on disk, not of a model.

## 0. Verdicts

| Level | Finding | Measurement | What to do |
|---|---|---|---|
| **HIGH** | MIL top-k degenerates to a plain max | 100.0% of clips get k = 1 at mil_topk_pct = 16: one supervised frame per clip per step. | Lower loss.mil_topk_pct, or accept that supervision is clip-level here and stop reading frame-level claims into it. |
| **HIGH** | Abnormal source clips with an all-zero label vector | 2 source clips declared abnormal in meta.json have no positive sampled frame anywhere; every metric counts them as normal. | Exclude them at scoring time (core/docs/DADA_SETUP.md §5.1). Do NOT 'fix' this with --strict: that flag raises, it does not repair. |
| **CRITICAL** | Micro AUC here is mostly clip classification (lesson C12) | A constant-score-per-clip oracle scores micro AUC 0.7529 with zero localization; 100.0% of the positive/negative pairs span two clips. | Report auc_macro as the headline and print this oracle beside any micro number. Never place a micro number from this corpus next to a published frame-level AUC. |
| **CRITICAL** | The features carry no frame-level signal | Even a supervised linear probe reaches only auc_macro 0.5894, while the clip-level probe reaches 0.5589. | No head on these features can localize. Frame-level work on this corpus needs a different backbone or a finer stride -- not another KIP variant (RESULTS_DADA.md §6, §10-C). |

## 1. Corpus shape

| | clips | abnormal | normal | frames |
|---|---:|---:|---:|---:|
| train | 4,894 | 3,297 | 1,597 | — |
| test | 1,221 | 813 | 408 | 19,536 |

Test positive frames: **6,631** (33.94% of sampled frames)
DVS dataset length (2 x abnormal train): **6,594** → **104 steps/epoch** at batch 64.

### 1.1 Clip-length distribution (sampled frames)

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test sampled | 1,221 | 16.00 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 |
| train sampled | 4,894 | 16.00 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 |
| raw (pre-stride) | 6,115 | 364.98 | 202.0 | 288.0 | 360.0 | 428.0 | 548.0 | 121.0 | 1220.0 |

### 1.2 Score-head receptive field (kernel 3) — lesson C27

- median clip length **16** sampled frames
- median fraction of a clip inside one output timestep: **18.8%**
- clips entirely inside the kernel: **0** (**0.0%**), holding **0.0%** of all test frames
- short-clip counts: `T<=3` → 0, `T<=5` → 0, `T<=9` → 0, `T<=13` → 0, `T<=17` → 1221

### 1.3 MIL top-k floor (mil_topk_pct = 16)

- clips at `k = 1` (loss is a plain max): **1,221** (**100.0%**)
- median k: **1.0**
- k distribution: {'1': 1221}

### 1.4 Subgroups by `split` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| test | 1,221 | 0 | 1,221 | 19,536 | 6,631 | 33.94% |
| train | 4,894 | 4,894 | 0 | 0 | 0 | — |

## 2. Label geometry

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| positives / test clip | 1,221 | 5.43 | 0.0 | 0.0 | 5.0 | 10.0 | 14.0 | 0.0 | 16.0 |
| positives / abnormal clip | 813 | 8.16 | 2.0 | 5.0 | 8.0 | 11.0 | 15.0 | 1.0 | 16.0 |
| positive frac. within abnormal | 813 | 0.51 | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 0.1 | 1.0 |
| spans / abnormal clip | 813 | 1.00 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| span length | 813 | 8.16 | 2.0 | 5.0 | 8.0 | 11.0 | 15.0 | 1.0 | 16.0 |

- abnormal clips with **exactly one** positive frame: **35**
- abnormal clips with <= 2 positive frames: **76**
- multi-span abnormal clips: **0** (lesson C18 applies if > 0)
- **abnormal source clips whose window vanished: 2**
- windows of abnormal clips holding no positive frame: **402** — these are *correct negatives*, not vanished windows; producing them is the point of re-sharding

<details><summary>Vanished-window ids (exclude these at scoring)</summary>

```
t10_v100
t48_v056
```
</details>

## 3. What the metric measures

### 3.1 Where the frames live

| clip kind | clips | frames | share of frames |
|---|---:|---:|---:|
| all_normal | 408 | 6,528 | 33.42% |
| mixed | 787 | 12,592 | 64.46% |
| all_positive | 26 | 416 | 2.13% |

### 3.2 The clip-level oracle (lesson C12)

A model emitting **one constant score per clip**, ranking clips perfectly and localizing nothing, scores:

- micro AUC **0.7529**, micro AP **0.5098**, macro AUC **0.5000** by construction
- 99.95% of positive/negative frame pairs span two clips (85,533,658 of 85,573,055); only 0.05% can be won by localization

**Print this oracle beside every micro AUC measured on this corpus.**

### 3.3 The clip-length leak (lesson C28)

A constant-score-per-clip detector whose **only** input is the clip's frame count — no pixels, no model — scores:

- clip-level AUC **0.5000** (**longer** clips are the abnormal ones); micro AUC **0.5000**, micro AP **0.3394**, macro AUC **0.5000** by construction
- abnormal clip length T: median **16.0**, min 16.0, max **16.0**
- normal clip length T: median **16.0**, min 16.0, max 16.0
- normal clips outside the abnormal length range entirely: **0** (0 frames)

**Any arm that does not beat this baseline is unmeasured.**

### 3.4 Per-clip AUC resolution

- two-class clips (the only ones `auc_macro` averages): **787**; single-class: 434

A clip with `p` positives and `n` negatives has `p*n` orderable pairs, so its AUC only takes values on a `1/(p*n)` grid. A coarse grid makes `auc_macro` honest but low-resolution — quote it with these counts.

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUC grid step 1/(pos*neg) | 787 | 0.02 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.1 |

- `--score-norm auto` resolves to **none** (normal-clip fraction 0.334 vs threshold 0.05)

## 4. The cached frozen-CLIP features

`/content/drive/MyDrive/Thesis/cache/clip/DADA2000_orig` — 1,221 of 1,221 test clips found

- 19,536 frames x 512 dims; mean L2 norm 9.87
- dimensions holding 90 % of the variance: **420** of 512

### 4.1 Temporal autocorrelation (cosine between frame t and t+lag)

| lag (sampled frames) | lag1 | lag2 | lag3 | lag4 | lag5 | lag6 | lag7 | lag8 |
|---|---|---|---|---|---|---|---|---|
| mean cosine | 0.968 | 0.959 | 0.955 | 0.947 | 0.943 | 0.942 | 0.936 | 0.934 |

A value near 1.0 at lag 1 means consecutive sampled frames are nearly identical, i.e. the stride is finer than the content changes.

- between-clip / within-clip **feature** variance: **3.08** (high = the embedding encodes scene identity more than dynamics)

### 4.2 Supervised linear probe — the representation ceiling

| probe | AUC | macro AUC | AP | AP baseline | folds | units |
|---|---:|---:|---:|---:|---:|---:|
| frame-level | 0.5753 | 0.5894 | 0.4126 | 0.3394 | 5 | 19,536 frames |
| clip-level (mean-pooled) | 0.5589 | — | 0.7096 | 0.6658 | 5 | 1,221 clips |

Grouped cross-validation by clip, so no clip's frames score themselves. This is the *supervised ceiling* of these features: a trained arm cannot be expected to beat it, and a large gap below it is a supervision problem, not a representation problem.

