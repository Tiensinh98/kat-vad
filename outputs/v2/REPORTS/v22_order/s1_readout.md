# v2.2 S1 — free diagnostics (no training)

Bar (§6.1): cluster-bootstrap 95 % low > 0 **and** mean ≥ +0.01, DoTA-CAP-dev, protocol B. DoTA-CAP (n/1397).

## S1a — change score `m5`: **PASS** (gates the TDP branch only; S2 runs regardless)

| read | DoTA-CAP-dev | T2-val (reported) |
|---|---|---|
| AUC(`m5`) | 0.7503 [0.7330, 0.7669] | 0.6943 [0.6657, 0.7208] |
| **AUC(`m5`) - 0.5** | **+0.2503 [+0.2330, +0.2669]** | +0.1943 [+0.1657, +0.2208] |
| **AUC(`m5`) - AUC(`m5`^π)** | **+0.2713 [+0.2537, +0.2891]** | +0.1977 [+0.1686, +0.2259] |
| AUC(`s`) - 0.5 (secondary) | +0.2161 [+0.1996, +0.2317] | +0.1663 [+0.1430, +0.1889] |
| AUC(`s`) - AUC(`s`^π) (secondary) | +0.2404 [+0.2255, +0.2547] | +0.1568 [+0.1344, +0.1791] |

Printed: per-clip corr(AUC(`m5`), AUC(`p_T2`)) = -0.053; ego 0.7847 [0.7658, 0.8031], non-ego 0.7056 [0.6780, 0.7318]; Δ vs twin ego +0.2959 [+0.2770, +0.3142], non-ego +0.2392 [+0.2090, +0.2688]; share of each clip's top-5 `m5` steps inside the span 0.593.
Sequences: DoTA {'sequences': 569, 'under_12': 0, 'under_3_excluded': 0, 'gap_counts': {6: 569}}; T2-val {'sequences': 219, 'under_12': 0, 'under_3_excluded': 0, 'gap_counts': {6: 219}}.

## S1b — Δu exit echo: Δu **NOT admitted (CLIP only)**

| stream | clips | Δ (+5..+6 - +10..+15) | ±0.1 SD margin | inside? | autocorr lags 3-7 |
|---|---|---|---|---|---|
| u (decides) | 120 | +2.8769 [+1.4177, +4.3616] | ±0.6378 | False | 0.14, 0.09, 0.05, -0.03, -0.06 |
| x (printed) | 120 | +1.2865 [+0.1876, +2.3723] | ±0.6840 | False | 0.36, 0.25, 0.16, 0.07, 0.01 |

## S1c — clip start: early-onset prediction **WITHDRAWN**

560 clips with normal steps 0-2. SMD (steps 0-2 vs normal steps ≥ 3): `s` +0.377, `m5` -0.346, `e` +0.196, `v_bar_mean` +0.395, `a_bar_mean` +0.080. Printed: max |SMD| over all κ channels 0.390, 61 channels above 0.1.
