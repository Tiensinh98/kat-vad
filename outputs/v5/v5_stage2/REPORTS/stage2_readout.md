# v5 Stage 2 read-out

Rule: core/docs/v5/PREREG_V5.md S2-7, S2-9, S2-10, S2-14 (+ §2.4 B10). Set: DoTA-CAP (n/1397) dev, protocol B, 569 two-class clips, seeds s2024.

## Decision (S2-10): **PRELIMINARY NOT-ADOPTED (S2-9) (n = 1 seed)**

> **PRELIMINARY (§2.4 B11):** fewer seeds than signed; S2-7 (ii) is read on these seeds only and the paired t is not defined at n = 1. Report it as preliminary; the S2-10 decision needs every seed.

- candidate **R5d** (R5Δ replaced R5: R5 had no sentence)
- detection: no difference shown at n = 1; maps point: yes; guardrails hold: no
- next: not the S2-10 decision (§2.4 B11): train the remaining seeds of 2024, 2025, 2026 and re-read

## Detection (S2-7) — R - R1, cluster bootstrap 97.5 % over source videos

| row | macro | Δ per clip [97.5 %] | per-seed Δ | paired t (printed) | sentence |
|---|---|---|---|---|---|
| R1 | 0.8102 | — | — | — | control |
| R5 | 0.8140 | +0.0038 [-0.0038, +0.0116] | +0.0038 | — | no difference shown at n = 1 |
| R5d | 0.8139 | +0.0037 [-0.0039, +0.0115] | +0.0037 | — | no difference shown at n = 1 |

## Maps (S2-8) — 95 % cluster bootstrap; see `maps_readout.md`

| row | hit - chance | lift - lift_pre | hit - centre | points? | STAUC |
|---|---|---|---|---|---|
| R5 | +0.3070 [+0.2823, +0.3332] | +0.1039 [+0.0745, +0.1336] | +0.0044 [-0.0365, +0.0448] | no | 0.3168 |
| R5d | +0.3585 [+0.3359, +0.3832] | +0.1682 [+0.1422, +0.1952] | +0.0558 [+0.0176, +0.0930] | **yes** | 0.3261 |

## Guardrails (S2-9)

| row | nexar_val crop endpoint (seed mean) | Δ vs R1 | S1-G4 | hold |
|---|---|---|---|---|
| R1 | 0.8858 | — | — | — |
| R5 | 0.8737 | -0.0121 | pass | **no** |
| R5d | 0.8739 | -0.0119 | pass | **no** |

## Printed, never gating (S2-14)

- `p_T2` on these clips: 0.8220 (recorded 0.8220)
- v2 Nexar A2: seed-averaged macro over 5 seeds 0.8171; mean of s2024 0.8110
- R0 = v2 Nexar A0: seed-averaged macro over 5 seeds 0.7138; mean of s2024 0.6886
- `f_2` (DoTA-CAP placement) macro: R1 0.8528, R5 0.8528, R5d 0.8528; Δ vs R1 (95 %): R5 -0.0001 [-0.0035, +0.0035], R5d -0.0001 [-0.0035, +0.0034]
- `nexar_val` whole-video macro: R1 +0.9303 [+0.9111, +0.9475], R5 +0.9263 [+0.9084, +0.9425], R5d +0.9264 [+0.9084, +0.9426]
- S1-G6 R5/s2024: entropy [3.892, 3.891, 3.892, 3.892, 3.891, 3.892, 3.892, 3.892] — **did not leave the mean**
- S1-G6 R5d/s2024: entropy [3.892, 3.892, 3.892, 3.892, 3.891, 3.892, 3.892, 3.892] — **did not leave the mean**
- last-step motion share / ‖W_u‖: R1/s2024 None; R5/s2024 {'motion_share': 0.25814077258110046, 'w_u_norm': 1.237740397453308}; R5d/s2024 {'motion_share': 0.25895628333091736, 'w_u_norm': 1.237529993057251}
