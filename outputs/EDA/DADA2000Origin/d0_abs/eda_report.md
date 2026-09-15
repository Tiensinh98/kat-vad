# EDA — DADA2000_orig

Data dir: `/content/d0/dataset/d0_abs` · score head kernel **9** · MIL top-k pct **16** · sections: features

Every number here is a property of the data on disk, not of a model.

## 0. Verdicts

| Level | Finding | Measurement | What to do |
|---|---|---|---|
| **INFO** | The features DO carry a frame-level signal | Supervised linear probe reaches auc_macro 0.6492 on the same cached features the trained arms saw. | The deficit is SUPERVISION, not representation. A better head or denser labels can close it; a different backbone is not required. |
## 4. The cached frozen-CLIP features

`/content/drive/MyDrive/Thesis/cache/clip/DADA2000_orig` — 400 of 400 test clips found

- 15,961 frames x 512 dims; mean L2 norm 9.88
- dimensions holding 90 % of the variance: **420** of 512

### 4.1 Temporal autocorrelation (cosine between frame t and t+lag)

| lag (sampled frames) | lag1 | lag2 | lag3 | lag4 | lag5 | lag6 | lag7 | lag8 |
|---|---|---|---|---|---|---|---|---|
| mean cosine | 0.968 | 0.959 | 0.955 | 0.947 | 0.943 | 0.941 | 0.936 | 0.933 |

A value near 1.0 at lag 1 means consecutive sampled frames are nearly identical, i.e. the stride is finer than the content changes.

- between-clip / within-clip **feature** variance: **1.97** (high = the embedding encodes scene identity more than dynamics)

### 4.2 Supervised linear probe — the representation ceiling

| probe | AUC | macro AUC | AP | AP baseline | folds | units |
|---|---:|---:|---:|---:|---:|---:|
| frame-level | 0.6214 | 0.6492 | 0.4359 | 0.3252 | 5 | 15,961 frames |

Grouped cross-validation by clip, so no clip's frames score themselves. This is the *supervised ceiling* of these features: a trained arm cannot be expected to beat it, and a large gap below it is a supervision problem, not a representation problem.

