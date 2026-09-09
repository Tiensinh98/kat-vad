# EDA — cross-corpus comparison

| property | DADA2000 | DoTA |
|---|---|---|
| test clips | 383 | 1,397 |
| test sampled frames | 5,244 | 18,369 |
| test positive frames | 476 | 6,086 |
| all-normal test clips | 192 | 3 |
| frame share in all-normal clips | 0.7429 | 0.0023 |
| **median clip length T** | 9.0 | 13.0 |
| clips with T <= kernel | 0.5535 | 0.1195 |
| median kernel coverage | 1.0000 | 0.6923 |
| clips at MIL k = 1 | 0.9034 | 0.9986 |
| median positives / abnormal clip | 2.0 | 4.0 |
| vanished windows | 4 | 3 |
| **clip-oracle micro AUC** | 0.9086 | 0.5017 |
| **length-only micro AUC** | 0.8654 | 0.4993 |
| length-only clip AUC | 0.8105 | 0.5280 |
| cross-clip pair fraction | 0.9990 | 0.9993 |
| score-norm auto resolves to | none | minmax |
| feature between/within variance | 2.72 | 3.19 |
| CLIP cosine at lag 1 | 0.963 | 0.965 |
| **frame probe macro AUC** | 0.5228 | 0.6708 |
| clip probe AUC | 0.6799 | 0.2176 |

A corpus whose clip-oracle micro AUC is high and whose median T is at or below the score-head kernel cannot support a frame-level claim, whatever its micro AUC says (lessons C12, C27). A corpus whose length-only micro AUC is high does not support one at all: the label is readable without the pixels (lesson C28).
