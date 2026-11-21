# Nexar source-shortcut probe (`KIM_RUN.md` §38 S0)

Frozen CLIP, mean over each window, logistic probe, 5-fold GroupKFold by source video, seed 2024. Gate: S1 >= 0.90.

Input: `/content/source_probe/a2_s8` rows `[::1]`, 896 dims.

| pool | windows | videos |
| --- | --- | --- |
| A | 5622 | 451 |
| B | 6879 | 449 |
| P | 1689 | 451 |

A = normal windows inside positive videos (K5f's normals); B = negative-video windows (K5fno's normals); P = positive windows.

| probe | AUC | role |
| --- | --- | --- |
| S1 A vs B | 0.6935 | decides |
| S2 A vs B, B matched to A's scene x weather (5589 / 1331) | 0.6768 | printed |
| S3 scene x weather one-hot only | 0.6396 | printed |
| S4 P vs A (K5f's contrast) | 0.7082 | printed |
| S4 P vs B (K5fno's contrast) | 0.7673 | printed |

**no shortcut at the gate: K5fno may be built (stratified B printed)**

S4: if P vs B sits far above P vs A, K5fno's task is easier than K5f's by a margin that is not the event. Printed, never gating.
