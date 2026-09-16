# EDA — DADA2000_orig

Data dir: `/content/drive/MyDrive/Thesis/data/DADA2000_orig` · score head kernel **3** · MIL top-k pct **16** · sections: corpus, labels, protocol, features, scores

Every number here is a property of the data on disk, not of a model.

## 0. Verdicts

| Level | Finding | Measurement | What to do |
|---|---|---|---|
| **HIGH** | MIL top-k degenerates to a plain max | 100.0% of clips get k = 1 at mil_topk_pct = 16: one supervised frame per clip per step. | Lower loss.mil_topk_pct, or accept that supervision is clip-level here and stop reading frame-level claims into it. |
| **HIGH** | Abnormal source clips with an all-zero label vector | 2 source clips declared abnormal in meta.json have no positive sampled frame anywhere; every metric counts them as normal. | Exclude them at scoring time (core/docs/DADA_SETUP.md §5.1). Do NOT 'fix' this with --strict: that flag raises, it does not repair. |
| **CRITICAL** | The features carry no frame-level signal | Even a supervised linear probe reaches only auc_macro 0.5983, while the clip-level probe reaches 0.5551. | No head on these features can localize. Frame-level work on this corpus needs a different backbone or a finer stride -- not another KIP variant (RESULTS_DADA.md §6, §10-C). |

## 1. Corpus shape

| | clips | abnormal | normal | frames |
|---|---:|---:|---:|---:|
| train | 4,401 | 3,242 | 1,159 | — |
| test | 1,106 | 805 | 301 | 22,120 |

Test positive frames: **7,340** (33.18% of sampled frames)
DVS dataset length (2 x abnormal train): **6,484** → **102 steps/epoch** at batch 64.

### 1.1 Clip-length distribution (sampled frames)

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test sampled | 1,106 | 20.00 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| train sampled | 4,401 | 20.00 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| raw (pre-stride) | 5,507 | 376.54 | 218.0 | 300.0 | 374.0 | 430.0 | 557.4 | 154.0 | 1220.0 |

### 1.2 Score-head receptive field (kernel 3) — lesson C27

- median clip length **20** sampled frames
- median fraction of a clip inside one output timestep: **15.0%**
- clips entirely inside the kernel: **0** (**0.0%**), holding **0.0%** of all test frames
- short-clip counts: `T<=3` → 0, `T<=5` → 0, `T<=9` → 0, `T<=13` → 0, `T<=17` → 0

### 1.3 MIL top-k floor (mil_topk_pct = 16)

- clips at `k = 1` (loss is a plain max): **1,106** (**100.0%**)
- median k: **1.0**
- k distribution: {'1': 1106}

### 1.4 Subgroups by `split` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| test | 1,106 | 0 | 1,106 | 22,120 | 7,340 | 33.18% |
| train | 4,401 | 4,401 | 0 | 0 | 0 | — |

## 2. Label geometry

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| positives / test clip | 1,106 | 6.64 | 0.0 | 0.0 | 7.0 | 11.0 | 16.0 | 0.0 | 20.0 |
| positives / abnormal clip | 805 | 9.12 | 2.0 | 6.0 | 9.0 | 12.0 | 16.0 | 1.0 | 20.0 |
| positive frac. within abnormal | 805 | 0.46 | 0.1 | 0.3 | 0.5 | 0.6 | 0.8 | 0.1 | 1.0 |
| spans / abnormal clip | 805 | 1.00 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| span length | 805 | 9.12 | 2.0 | 6.0 | 9.0 | 12.0 | 16.0 | 1.0 | 20.0 |

- abnormal clips with **exactly one** positive frame: **32**
- abnormal clips with <= 2 positive frames: **68**
- multi-span abnormal clips: **0** (lesson C18 applies if > 0)
- **abnormal source clips whose window vanished: 2**
- windows of abnormal clips holding no positive frame: **295** — these are *correct negatives*, not vanished windows; producing them is the point of re-sharding

<details><summary>Vanished-window ids (exclude these at scoring)</summary>

```
t05_v048
t10_v014
```
</details>

## 3. What the metric measures

### 3.1 Where the frames live

| clip kind | clips | frames | share of frames |
|---|---:|---:|---:|
| all_normal | 301 | 6,020 | 27.22% |
| mixed | 798 | 15,960 | 72.15% |
| all_positive | 7 | 140 | 0.63% |

### 3.2 The clip-level oracle (lesson C12)

A model emitting **one constant score per clip**, ranking clips perfectly and localizing nothing, scores:

- micro AUC **0.7037**, micro AP **0.4559**, macro AUC **0.5000** by construction
- 99.94% of positive/negative frame pairs span two clips (108,420,102 of 108,485,200); only 0.06% can be won by localization

**Print this oracle beside every micro AUC measured on this corpus.**

### 3.3 The clip-length leak (lesson C28)

A constant-score-per-clip detector whose **only** input is the clip's frame count — no pixels, no model — scores:

- clip-level AUC **0.5000** (**longer** clips are the abnormal ones); micro AUC **0.5000**, micro AP **0.3318**, macro AUC **0.5000** by construction
- abnormal clip length T: median **20.0**, min 20.0, max **20.0**
- normal clip length T: median **20.0**, min 20.0, max 20.0
- normal clips outside the abnormal length range entirely: **0** (0 frames)

**Any arm that does not beat this baseline is unmeasured.**

### 3.4 Per-clip AUC resolution

- two-class clips (the only ones `auc_macro` averages): **798**; single-class: 308

A clip with `p` positives and `n` negatives has `p*n` orderable pairs, so its AUC only takes values on a `1/(p*n)` grid. A coarse grid makes `auc_macro` honest but low-resolution — quote it with these counts.

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUC grid step 1/(pos*neg) | 798 | 0.01 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 |

- `--score-norm auto` resolves to **none** (normal-clip fraction 0.272 vs threshold 0.05)

## 4. The cached frozen-CLIP features

`/content/drive/MyDrive/Thesis/cache/clip/DADA2000_orig` — 1,106 of 1,106 test clips found

- 22,120 frames x 512 dims; mean L2 norm 9.87
- dimensions holding 90 % of the variance: **420** of 512

### 4.1 Temporal autocorrelation (cosine between frame t and t+lag)

| lag (sampled frames) | lag1 | lag2 | lag3 | lag4 | lag5 | lag6 | lag7 | lag8 |
|---|---|---|---|---|---|---|---|---|
| mean cosine | 0.968 | 0.959 | 0.956 | 0.948 | 0.944 | 0.942 | 0.937 | 0.934 |

A value near 1.0 at lag 1 means consecutive sampled frames are nearly identical, i.e. the stride is finer than the content changes.

- between-clip / within-clip **feature** variance: **2.77** (high = the embedding encodes scene identity more than dynamics)

### 4.2 Supervised linear probe — the representation ceiling

| probe | AUC | macro AUC | AP | AP baseline | folds | units |
|---|---:|---:|---:|---:|---:|---:|
| frame-level | 0.5910 | 0.5983 | 0.4299 | 0.3318 | 5 | 22,120 frames |
| clip-level (mean-pooled) | 0.5551 | — | 0.7504 | 0.7278 | 5 | 1,106 clips |

Grouped cross-validation by clip, so no clip's frames score themselves. This is the *supervised ceiling* of these features: a trained arm cannot be expected to beat it, and a large gap below it is a supervision problem, not a representation problem.

