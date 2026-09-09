# EDA — DADA2000

Data dir: `/content/drive/MyDrive/Thesis/data/DADA2000` · score head kernel **9** · MIL top-k pct **16** · sections: corpus, labels, protocol, features, scores

Every number here is a property of the data on disk, not of a model.

## 0. Verdicts

| Level | Finding | Measurement | What to do |
|---|---|---|---|
| **CRITICAL** | Score head spans the clip (lesson C27) | 55.4% of test clips are <= kernel_size 9 (median T = 9); 24.3% of frames live in them. Every output timestep there sees the whole clip. | Lower model.score_head_kernel (try 3) or lower frame_stride so median T >> 9. Both fire lesson C2 on existing caches. |
| **HIGH** | MIL top-k degenerates to a plain max | 90.3% of clips get k = 1 at mil_topk_pct = 16: one supervised frame per clip per step. | Lower loss.mil_topk_pct, or accept that supervision is clip-level here and stop reading frame-level claims into it. |
| **HIGH** | Abnormal clips with an all-zero label vector | 4 clips declared abnormal in meta.json have no positive sampled frame; every metric counts them as normal. | Exclude them at scoring time (core/docs/DADA_SETUP.md §5.1). Do NOT 'fix' this with --strict: that flag raises, it does not repair. |
| **CRITICAL** | Micro AUC here is mostly clip classification (lesson C12) | A constant-score-per-clip oracle scores micro AUC 0.9086 with zero localization; 99.9% of the positive/negative pairs span two clips. | Report auc_macro as the headline and print this oracle beside any micro number. Never place a micro number from this corpus next to a published frame-level AUC. |
| **HIGH** | The scored run's curves are flat | Between-clip / within-clip score variance = 18.8; 3 clips have a literally constant curve. | This model has learned clip separation, not localization. Expect it to collapse under a per-clip min-max protocol (lesson C8). |
| **CRITICAL** | The features carry no frame-level signal | Even a supervised linear probe reaches only auc_macro 0.5228, while the clip-level probe reaches 0.6799. | No head on these features can localize. Frame-level work on this corpus needs a different backbone or a finer stride -- not another KIP variant (RESULTS_DADA.md §6, §10-C). |

## 1. Corpus shape

| | clips | abnormal | normal | frames |
|---|---:|---:|---:|---:|
| train | 1,530 | 780 | 750 | — |
| test | 383 | 191 | 192 | 5,244 |

Test positive frames: **476** (9.08% of sampled frames)
DVS dataset length (2 x abnormal train): **1,560** → **25 steps/epoch** at batch 64.

### 1.1 Clip-length distribution (sampled frames)

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test sampled | 383 | 13.69 | 3.0 | 6.0 | 9.0 | 19.0 | 37.9 | 1.0 | 72.0 |
| train sampled | 1,530 | 13.13 | 3.0 | 6.0 | 9.0 | 18.0 | 36.0 | 1.0 | 73.0 |
| raw (pre-stride) | 1,913 | 102.27 | 24.0 | 41.0 | 68.0 | 139.0 | 289.0 | 4.0 | 579.0 |

### 1.2 Score-head receptive field (kernel 9) — lesson C27

- median clip length **9** sampled frames
- median fraction of a clip inside one output timestep: **100.0%**
- clips entirely inside the kernel: **212** (**55.4%**), holding **24.3%** of all test frames
- short-clip counts: `T<=3` → 23, `T<=5` → 86, `T<=9` → 212, `T<=13` → 256, `T<=17` → 276

### 1.3 MIL top-k floor (mil_topk_pct = 16)

- clips at `k = 1` (loss is a plain max): **346** (**90.3%**)
- median k: **1.0**
- k distribution: {'1': 346, '2': 29, '3': 7, '4': 1}

### 1.4 Subgroups by `fault_label` (3 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| 0_Non_Ego_Fault | 567 | 454 | 113 | 799 | 272 | 34.04% |
| 0_Normal_Driving | 938 | 750 | 188 | 3,880 | 0 | 0.00% |
| 1_Ego_Fault | 408 | 326 | 82 | 565 | 204 | 36.11% |

### 1.4 Subgroups by `ego_involve` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| False | 1,505 | 1,204 | 301 | 4,679 | 272 | 5.81% |
| True | 408 | 326 | 82 | 565 | 204 | 36.11% |

### 1.4 Subgroups by `class_name` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| CarAccident | 975 | 780 | 195 | 1,364 | 476 | 34.90% |
| Normal | 938 | 750 | 188 | 3,880 | 0 | 0.00% |

### 1.4 Subgroups by `split` (2 groups)

| group | clips | train | test | test frames | positive frames | pos. frac |
|---|---:|---:|---:|---:|---:|---:|
| test | 383 | 0 | 383 | 5,244 | 476 | 9.08% |
| train | 1,530 | 1,530 | 0 | 0 | 0 | — |

## 2. Label geometry

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| positives / test clip | 383 | 1.24 | 0.0 | 0.0 | 0.0 | 2.0 | 4.0 | 0.0 | 7.0 |
| positives / abnormal clip | 191 | 2.49 | 1.0 | 1.0 | 2.0 | 3.0 | 5.0 | 1.0 | 7.0 |
| positive frac. within abnormal | 191 | 0.35 | 0.2 | 0.2 | 0.3 | 0.4 | 0.6 | 0.1 | 1.0 |
| spans / abnormal clip | 191 | 1.00 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| span length | 191 | 2.49 | 1.0 | 1.0 | 2.0 | 3.0 | 5.0 | 1.0 | 7.0 |

- abnormal clips with **exactly one** positive frame: **54**
- abnormal clips with <= 2 positive frames: **106**
- multi-span abnormal clips: **0** (lesson C18 applies if > 0)
- **abnormal clips whose window vanished: 4**

<details><summary>Vanished-window ids (exclude these at scoring)</summary>

```
0_Non_Ego_Fault__type1_vid017
0_Non_Ego_Fault__type1_vid052
0_Non_Ego_Fault__type6_vid112
1_Ego_Fault__type1_vid024
```
</details>

## 3. What the metric measures

### 3.1 Where the frames live

| clip kind | clips | frames | share of frames |
|---|---:|---:|---:|
| all_normal | 192 | 3,896 | 74.29% |
| mixed | 190 | 1,347 | 25.69% |
| all_positive | 1 | 1 | 0.02% |

### 3.2 The clip-level oracle (lesson C12)

A model emitting **one constant score per clip**, ranking clips perfectly and localizing nothing, scores:

- micro AUC **0.9086**, micro AP **0.3531**, macro AUC **0.5000** by construction
- 99.90% of positive/negative frame pairs span two clips (2,267,288 of 2,269,568); only 0.10% can be won by localization

**Print this oracle beside every micro AUC measured on this corpus.**

### 3.3 Per-clip AUC resolution

- two-class clips (the only ones `auc_macro` averages): **190**; single-class: 193

A clip with `p` positives and `n` negatives has `p*n` orderable pairs, so its AUC only takes values on a `1/(p*n)` grid. A coarse grid makes `auc_macro` honest but low-resolution — quote it with these counts.

| distribution | n | mean | p5 | p25 | **p50** | p75 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUC grid step 1/(pos*neg) | 190 | 0.15 | 0.0 | 0.1 | 0.1 | 0.2 | 0.3 | 0.0 | 1.0 |

- `--score-norm auto` resolves to **none** (normal-clip fraction 0.501 vs threshold 0.05)

### 3.4 A scored run's curves

`/content/drive/MyDrive/Thesis-V3/outputs/DADA2000/constant_s2024/eval_dada/scores` — 383 clips, auc_macro **0.5292** over 190 clips

- between-clip / within-clip score variance: **18.77**
- median within-clip score range: **0.113**
- literally constant curves: **3** clips

## 4. The cached frozen-CLIP features

`/content/drive/MyDrive/Thesis/cache/clip/DADA2000` — 383 of 383 test clips found

- 5,244 frames x 512 dims; mean L2 norm 9.91
- dimensions holding 90 % of the variance: **419** of 512

### 4.1 Temporal autocorrelation (cosine between frame t and t+lag)

| lag (sampled frames) | lag1 | lag2 | lag3 | lag4 | lag5 | lag6 | lag7 | lag8 |
|---|---|---|---|---|---|---|---|---|
| mean cosine | 0.963 | 0.953 | 0.950 | 0.942 | 0.939 | 0.939 | 0.934 | 0.931 |

A value near 1.0 at lag 1 means consecutive sampled frames are nearly identical, i.e. the stride is finer than the content changes.

- between-clip / within-clip **feature** variance: **2.72** (high = the embedding encodes scene identity more than dynamics)

### 4.2 Supervised linear probe — the representation ceiling

| probe | AUC | macro AUC | AP | AP baseline | folds | units |
|---|---:|---:|---:|---:|---:|---:|
| frame-level | 0.5429 | 0.5228 | 0.1058 | 0.0908 | 5 | 5,244 frames |
| clip-level (mean-pooled) | 0.6799 | — | 0.6670 | 0.4987 | 5 | 383 clips |

Grouped cross-validation by clip, so no clip's frames score themselves. This is the *supervised ceiling* of these features: a trained arm cannot be expected to beat it, and a large gap below it is a supervision problem, not a representation problem.

