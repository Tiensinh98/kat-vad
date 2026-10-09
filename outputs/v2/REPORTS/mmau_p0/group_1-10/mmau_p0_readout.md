# MM-AU / CAP-DATA Phase 0 — provenance read-out

Rule: .project/plans/katvad-mmau-phase0.md §3-§4. exact κ ≥ 0.99, near κ ≥ 0.95, top-K 5 (hard null = K-th), null flag > 1%, coverage bar 95%.

CAP clips cached: **1556 / 9768** (partial → no branch)

| query set | n | exact | near | none | exact cov. | near+exact cov. | κ q05/q25/q50/q75/q95 | null κ q05…q95 | null ≥ near | rate q05…q95 |
|---|---|---|---|---|---|---|---|---|---|---|
| DoTA (all) | 1397 | 214 | 31 | 1152 | 0.153 | 0.175 | 0.872 / 0.902 / 0.918 / 0.934 / 0.995 | 0.851 / 0.882 / 0.900 / 0.912 / 0.926 | 0.001 | 0.967 / 1.000 / 2.906 / 2.996 / 3.020 |
| DoTA-dev | 702 | 104 | 14 | 584 | 0.148 | 0.168 | 0.873 / 0.901 / 0.916 / 0.932 / 0.995 | 0.853 / 0.882 / 0.898 / 0.911 / 0.925 | 0.001 | 0.975 / 1.000 / 2.849 / 2.997 / 3.011 |
| DADA-2000 sources | 1945 | 0 | 0 | 1945 | 0.000 | 0.000 | 0.853 / 0.886 / 0.901 / 0.913 / 0.927 | 0.838 / 0.873 / 0.888 / 0.903 / 0.918 | 0.000 | — |

CAP clips hit by DoTA: 306 · by DADA: 0 · **clean: 1250** (197319 frames).

## Branch (plan §4): **FEASIBILITY_ONLY**
