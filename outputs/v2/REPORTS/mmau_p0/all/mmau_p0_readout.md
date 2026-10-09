# MM-AU / CAP-DATA Phase 0 — provenance read-out

Rule: .project/plans/katvad-mmau-phase0.md §3-§4. exact κ ≥ 0.99, near κ ≥ 0.95, top-K 5 (hard null = K-th), null flag > 1%, coverage bar 95%, near needs rate ≥ 0.9 (Amendment P0b-1).

CAP clips cached: **9768 / 9768**

| query set | n | exact | near | near rej. | none | exact cov. | near+exact cov. | κ q05/q25/q50/q75/q95 | null κ q05…q95 | null ≥ near | rate q05…q95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DoTA (all) | 1397 | 1248 | 58 | 7 | 91 | 0.893 | 0.935 | 0.937 / 0.993 / 0.994 / 0.995 / 0.996 | 0.869 / 0.899 / 0.913 / 0.924 / 0.938 | 0.006 | 0.982 / 0.999 / 1.001 / 2.983 / 3.008 |
| DoTA-dev | 702 | 625 | 28 | 1 | 49 | 0.890 | 0.930 | 0.931 / 0.993 / 0.994 / 0.995 / 0.996 | 0.870 / 0.898 / 0.912 / 0.923 / 0.937 | 0.000 | 0.986 / 0.999 / 1.001 / 2.987 / 3.007 |
| DADA-2000 sources | 1945 | 0 | 3 | 1 | 1942 | 0.000 | 0.002 | 0.863 / 0.893 / 0.908 / 0.921 / 0.934 | 0.850 / 0.882 / 0.899 / 0.912 / 0.926 | 0.000 | 3.938 / 5.204 / 6.786 / 7.043 / 7.250 |

CAP clips hit by DoTA: 1759 · by DADA: 3 · **clean: 8006** (1225036 frames).

## Branch (plan §4): **C**
