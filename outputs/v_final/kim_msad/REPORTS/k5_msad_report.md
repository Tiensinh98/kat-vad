# K5 vs LaGoVAD (released ckpt), zero-shot MSAD test — KIM_RUN.md §18

**PRELIMINARY (n = 1), not matched.** K5 s2024; nothing trained. Gate M0 (legacy shared-verbalizer read): LaGoVAD micro 0.8949 vs `full_gate_a` 0.8949 (gap -0.00000) PASS. The table's per-item read (definitions drawn per video, same draws for both rows) is 0.8986 (gap +0.00366, printed, not gated: the draws differ, C30). LaGoVAD's printed MSAD AUC is 0.9041 (beside its re-read only).

| Set | Metric | LaGoVAD | K5 | K5 - LaGoVAD |
|---|---|---|---|---|
| **MSAD-traffic** (140 videos, macro over 19) | AUC macro | 0.6247 [0.4766, 0.7674] | 0.7424 [0.6246, 0.8507] | +0.1177 [-0.0462, +0.2789] **not separated** |
| | AUC micro | 0.9375 | 0.6188 | -0.3187 |
| | micro, equal video weight | 0.9226 | 0.6332 | |
| | AP (base rate 0.0316) | 0.3361 | 0.0557 | |
| | clip AUC (peak) | 0.9279 | 0.5617 | |
| **MSAD-full** (240 videos, macro over 96) | AUC macro | 0.7176 [0.6625, 0.7705] | 0.6810 [0.6255, 0.7347] | -0.0366 [-0.1076, +0.0340] **not separated** |
| | AUC micro | 0.8986 | 0.5386 | -0.3600 |
| | micro, equal video weight | 0.8607 | 0.5512 | |
| | AP (base rate 0.2300) | 0.6554 | 0.2607 | |
| | clip AUC (peak) | 0.9450 | 0.5020 | |

## Rulers (no pixel)

| Set | clip oracle (micro) | length ruler (micro) | `t/N` ruler (macro) |
|---|---|---|---|
| MSAD-traffic | 0.9771 | 0.9639 | 0.5874 [0.4213, 0.7525] |
| MSAD-full | 0.8943 | 0.6466 | 0.7090 [0.6407, 0.7750] |

## Macro per class (MSAD-full, printed)

| Class | LaGoVAD | K5 |
|---|---|---|
| Assault | 0.5489 | 0.6545 |
| Explosion | 0.8160 | 0.6839 |
| Fighting | 0.8819 | 0.8493 |
| Fire | 0.8717 | 0.4718 |
| Object_falling | 0.7312 | 0.7701 |
| People_falling | 0.7373 | 0.5930 |
| Robbery | 0.5525 | 0.6627 |
| Shooting | 0.8776 | 0.7779 |
| Traffic_accident | 0.6247 | 0.7424 |
| Vandalism | 0.8796 | 0.6824 |
| Water_incident | 0.8512 | 0.7003 |

Census: 240 videos, fps spread {'10.0': 1, '15.0': 2, '25.0': 14, '29.0': 3, '29.87': 1, '29.95': 1, '29.97': 58, '29.98': 1, '30.0': 157, '6.0': 2}, 19 off 30 ± 1; label rows ≠ CLIP rows on 0 (truncated alike, §18 M2).

## How to read this

* The CCTV-traffic question is answered by **MSAD-traffic macro only** (MS6). MSAD-full is mostly classes outside the thesis scope: a cost there is not attributed to the camera.
* **Micro on MSAD-traffic says nothing**: the length ruler alone beats it. On MSAD-full the clip oracle sits at LaGoVAD's micro.
* Never put the paper's number beside K5; never write "beats LaGoVAD" (not matched).
