# BDD-A negative-bag EDA -- read-out

G-I PASS (925 usable, 508 calm) | G-L (L-T2) length AUC 0.5000 PASS | T2 oracle mix0 0.7037
R0 0.676227 -- reproduces D2City's 0.676258: True

| quantity | all | calm (gate) |
|---|---:|---:|
| R0 auc_macro | 0.6762 | 0.6762 |
| X auc_macro | 0.5375 | 0.5492 |
| M auc_macro | 0.6714 | 0.6733 |
| S (DADA-pre vs BDD-A) | 1.0000 | 1.0000 |
| S-ref (DADA-pre vs DoTA) | 0.9999 | 0.9999 |
| shortcut AUC, R0 | 0.3741 | 0.3741 |
| shortcut AUC, X | 1.0000 | 1.0000 |
| shortcut AUC, M | 0.9995 | 0.9992 |
| G-X | FAIL | FAIL |
| G-M | True | True |
| Δ(X−R0) t95 | -0.1387 [-0.1535, -0.1240] | -0.1270 [-0.1532, -0.1008] |
| Δ(M−R0) t95 | -0.0048 [-0.0089, -0.0007] | -0.0029 [-0.0062, +0.0004] |

**Gate arm:** calm  ·  **Call:** NO-GO -- BDD-A teaches source, not accident
