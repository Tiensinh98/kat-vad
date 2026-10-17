# v4 Stage A: head selection (PREREG S1-S14)

Encoder `vit_s_k710_dl_from_giant`, 1272 T2-train sources (sha1 `cf7537843d152bf3f2bfdf6130aaa4fa66b293d8`), folds [255, 255, 254, 254, 254], seed 2024, commit `UNKNOWN`.
Steps: 10892 normal / 33324 abnormal labelled, 10223 not scored (S2).

**Selected (detrended, S9):** [14, 17, 19, 22, 23] = b2h2, b2h5, b3h1, b3h4, b3h5

**S10 stability:** mean fold Jaccard **0.180** vs floor 0.25 (chance 0.0397) → **FAIL**

Best block (S11): 3 → `ẽ_full` heads [18, 19, 20, 21, 22, 23]. A4-rand (S12): [30, 33, 68, 69, 71].

| fold | top-5 (detrended) | top-5 (raw) |
|---|---|---|
| 0 | [4, 11, 17, 19, 20] | [9, 14, 17, 21, 22] |
| 1 | [1, 10, 14, 17, 22] | [14, 17, 22, 23, 28] |
| 2 | [14, 17, 19, 21, 22] | [14, 17, 20, 23, 28] |
| 3 | [14, 15, 19, 22, 23] | [9, 10, 17, 20, 22] |
| 4 | [4, 10, 13, 33, 70] | [14, 16, 21, 22, 23] |

**Printed only (S13):** raw top-5 [14, 17, 21, 22, 23]; shared with selected [14, 17, 22, 23] (Jaccard 0.67); Spearman rho of RSS 0.816; raw fold Jaccard 0.349.

Next: **commit `head_selection.json`**, then run G1a with its sha256 (S14).
