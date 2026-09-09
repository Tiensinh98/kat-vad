# EDA — DoTA

Data dir: `/content/drive/MyDrive/Thesis/data/DoTA/labels_s8` · score head kernel **9** · MIL top-k pct **16** · sections: corpus, labels, protocol, features, scores

Every number here is a property of the data on disk, not of a model.

## 0. Verdicts

| Level | Finding | Measurement | What to do |
|---|---|---|---|
| **HIGH** | MIL top-k degenerates to a plain max | 99.9% of clips get k = 1 at mil_topk_pct = 16: one supervised frame per clip per step. | Lower loss.mil_topk_pct, or accept that supervision is clip-level here and stop reading frame-level claims into it. |
| **HIGH** | Abnormal clips with an all-zero label vector | 3 clips declared abnormal in meta.json have no positive sampled frame; every metric counts them as normal. | Exclude them at scoring time (core/docs/DADA_SETUP.md §5.1). Do NOT 'fix' this with --strict: that flag raises, it does not repair. |
| **INFO** | The features DO carry a frame-level signal | Supervised linear probe reaches auc_macro 0.6708 on the same cached features the trained arms saw. | The deficit is SUPERVISION, not representation. A better head or denser labels can close it; a different backbone is not required. |

## 1. Corpus shape

| | clips | abnormal | normal | frames |
|---|---:|---:|---:|---:|
| train | 0 | 0 | 0 | — |
| test | 1,397 | 1,394 | 3 | 18,369 |

Test positive frames: **6,086** (33.13% of sampled frames)
DVS dataset length (2 x abnormal train): **0** → **0 steps/epoch** at batch 64.

### 1.1 Clip-length distribution (sampled frames)

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test sampled | 1,397 | 13.15 | 8.0 | 11.0 | 13.0 | 15.0 | 19.0 | 1.0 | 36.0 |
| train sampled | 0 | — | — | — | — | — | — | — | — |
| raw (pre-stride) | 1,397 | 101.57 | 61.0 | 83.0 | 99.0 | 115.0 | 151.0 | 6.0 | 284.0 |

### 1.2 Score-head receptive field (kernel 9) — lesson C27

- median clip length **13** sampled frames
- median fraction of a clip inside one output timestep: **69.2%**
- clips entirely inside the kernel: **167** (**12.0%**), holding **7.3%** of all test frames
- short-clip counts: `T<=3` → 2, `T<=5` → 6, `T<=9` → 167, `T<=13` → 856, `T<=17` → 1219

### 1.3 MIL top-k floor (mil_topk_pct = 16)

- clips at `k = 1` (loss is a plain max): **1,395** (**99.9%**)
- median k: **1.0**
- k distribution: {'1': 1395, '2': 2}

### 1.4 Subgroups by `anomaly_class` (20 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| ego: lateral | 153 | 0 | 153 | 1,992 | 734 | 36.85% |
| ego: leave_to_left | 45 | 0 | 45 | 657 | 313 | 47.64% |
| ego: leave_to_right | 40 | 0 | 40 | 623 | 285 | 45.75% |
| ego: moving_ahead_or_waiting | 119 | 0 | 119 | 1,545 | 491 | 31.78% |
| ego: obstacle | 16 | 0 | 16 | 199 | 61 | 30.65% |
| ego: oncoming | 104 | 0 | 104 | 1,331 | 391 | 29.38% |
| ego: pedestrian | 18 | 0 | 18 | 232 | 64 | 27.59% |
| ego: start_stop_or_stationary | 11 | 0 | 11 | 141 | 38 | 26.95% |
| ego: turning | 279 | 0 | 279 | 3,309 | 1,040 | 31.43% |
| ego: unknown | 18 | 0 | 18 | 238 | 82 | 34.45% |
| other: lateral | 83 | 0 | 83 | 1,145 | 395 | 34.50% |
| other: leave_to_left | 54 | 0 | 54 | 741 | 288 | 38.87% |
| other: leave_to_right | 77 | 0 | 77 | 1,144 | 444 | 38.81% |
| other: moving_ahead_or_waiting | 73 | 0 | 73 | 1,002 | 310 | 30.94% |
| other: obstacle | 20 | 0 | 20 | 245 | 82 | 33.47% |
| other: oncoming | 30 | 0 | 30 | 393 | 98 | 24.94% |
| other: pedestrian | 21 | 0 | 21 | 284 | 75 | 26.41% |
| other: start_stop_or_stationary | 17 | 0 | 17 | 260 | 54 | 20.77% |
| other: turning | 205 | 0 | 205 | 2,683 | 765 | 28.51% |
| other: unknown | 14 | 0 | 14 | 205 | 76 | 37.07% |

### 1.4 Subgroups by `ego_involve` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| False | 594 | 0 | 594 | 8,102 | 2,587 | 31.93% |
| True | 803 | 0 | 803 | 10,267 | 3,499 | 34.08% |

## 2. Label geometry

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| positives / test clip | 1,397 | 4.36 | 1.0 | 3.0 | 4.0 | 5.0 | 9.0 | 0.0 | 26.0 |
| positives / abnormal clip | 1,394 | 4.37 | 1.0 | 3.0 | 4.0 | 5.0 | 9.0 | 1.0 | 26.0 |
| positive frac. within abnormal | 1,394 | 0.33 | 0.1 | 0.2 | 0.3 | 0.4 | 0.7 | 0.0 | 1.0 |
| spans / abnormal clip | 1,394 | 1.00 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| span length | 1,394 | 4.37 | 1.0 | 3.0 | 4.0 | 5.0 | 9.0 | 1.0 | 26.0 |

- abnormal clips with **exactly one** positive frame: **79**
- abnormal clips with <= 2 positive frames: **307**
- multi-span abnormal clips: **0** (lesson C18 applies if > 0)
- **abnormal clips whose window vanished: 3**

<details><summary>Vanished-window ids (exclude these at scoring)</summary>

```
90gyBengKDs_003658
bcpfYpmDcp0_000972
ezY7QrSWeWw_001685
```
</details>

## 3. What the metric measures

### 3.1 Where the frames live

| clip kind | clips | frames | share of frames |
|---|---:|---:|---:|
| all_normal | 3 | 42 | 0.23% |
| mixed | 1,392 | 18,319 | 99.73% |
| all_positive | 2 | 8 | 0.04% |

### 3.2 The clip-level oracle (lesson C12)

A model emitting **one constant score per clip**, ranking clips perfectly and localizing nothing, scores:

- micro AUC **0.5017**, micro AP **0.3321**, macro AUC **0.5000** by construction
- 99.93% of positive/negative frame pairs span two clips (74,704,547 of 74,754,338); only 0.07% can be won by localization

**Print this oracle beside every micro AUC measured on this corpus.**

### 3.3 The clip-length leak (lesson C28)

A constant-score-per-clip detector whose **only** input is the clip's frame count — no pixels, no model — scores:

- clip-level AUC **0.5280** (**shorter** clips are the abnormal ones); micro AUC **0.4993**, micro AP **0.3349**, macro AUC **0.5000** by construction
- abnormal clip length T: median **13.0**, min 1.0, max **36.0**
- normal clip length T: median **13.0**, min 9.0, max 20.0
- normal clips outside the abnormal length range entirely: **0** (0 frames)

**Any arm that does not beat this baseline is unmeasured.**

### 3.4 Per-clip AUC resolution

- two-class clips (the only ones `auc_macro` averages): **1,392**; single-class: 5

A clip with `p` positives and `n` negatives has `p*n` orderable pairs, so its AUC only takes values on a `1/(p*n)` grid. A coarse grid makes `auc_macro` honest but low-resolution — quote it with these counts.

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUC grid step 1/(pos*neg) | 1,392 | 0.04 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 1.0 |

- `--score-norm auto` resolves to **minmax** (normal-clip fraction 0.002 vs threshold 0.05)

## 4. The cached frozen-CLIP features

`/content/drive/MyDrive/Thesis/cache/clip/DoTA_s8_ncc` — 1,397 of 1,397 test clips found

- 18,369 frames x 512 dims; mean L2 norm 9.61
- dimensions holding 90 % of the variance: **419** of 512

### 4.1 Temporal autocorrelation (cosine between frame t and t+lag)

| lag (sampled frames) | lag1 | lag2 | lag3 | lag4 | lag5 | lag6 | lag7 | lag8 |
|---|---|---|---|---|---|---|---|---|
| mean cosine | 0.965 | 0.953 | 0.943 | 0.936 | 0.930 | 0.925 | 0.922 | 0.918 |

A value near 1.0 at lag 1 means consecutive sampled frames are nearly identical, i.e. the stride is finer than the content changes.

- between-clip / within-clip **feature** variance: **3.19** (high = the embedding encodes scene identity more than dynamics)

### 4.2 Supervised linear probe — the representation ceiling

| probe | AUC | macro AUC | AP | AP baseline | folds | units |
|---|---:|---:|---:|---:|---:|---:|
| frame-level | 0.6249 | 0.6708 | 0.4525 | 0.3313 | 5 | 18,369 frames |
| clip-level (mean-pooled) | 0.2176 | — | 0.9959 | 0.9979 | 3 | 1,397 clips |

Grouped cross-validation by clip, so no clip's frames score themselves. This is the *supervised ceiling* of these features: a trained arm cannot be expected to beat it, and a large gap below it is a supervision problem, not a representation problem.

