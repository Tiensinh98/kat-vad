# DF8 / SC-1 — position swap (v2 addendum §24)

DoTA-CAP-dev shift subset, n = 491, 86 source clusters, protocol B per crop. Centres (p10/p50/p90): control [0.395, 0.513, 0.589]; early [0.2, 0.301, 0.406]; full [0.351, 0.519, 0.639]; late [0.594, 0.699, 0.8].

X10(b) full-arm max |diff| vs saved scores: A3ign 3.5e-07, A3 0.0e+00, A0 0.0e+00, pT2 0.0e+00.

## Verdict (X8): **R1>0, X7 fails**

> The prior's advantage does not survive a position swap, but the detector also depends on where the accident sits. (DoTA-CAP-dev shift subset, n = 491; accident centre 0.30-0.70)

* **R1** shifted(A3-ign) - shifted(`p_T2`): **+0.1201 [+0.0974, +0.1413]** (same Δ on the full clips: -0.0351 [-0.0613, -0.0098])
* R2 early - control: -0.1251 [-0.1405, -0.1096]
* R2 late - control: +0.0744 [+0.0573, +0.0912]
* R2 early - late: -0.1994 [-0.2254, -0.1732]
* X7 non-inferiority (both moved - control lower bounds > -0.01): **fails**
* R3 full - control (context cost): +0.0264 [+0.0142, +0.0383]

## Macro per arm (seed-averaged; R4 = the `p_T2` row)

| model | full | control | early | late |
|---|---|---|---|---|
| A3ign | 0.7934 [0.7745, 0.8119] | 0.7670 [0.7480, 0.7865] | 0.6419 [0.6148, 0.6691] | 0.8414 [0.8255, 0.8559] |
| A3 | 0.7588 [0.7412, 0.7763] | 0.7091 [0.6913, 0.7269] | 0.6234 [0.5942, 0.6523] | 0.8135 [0.7953, 0.8307] |
| A0 | 0.6731 [0.6491, 0.6955] | 0.6436 [0.6228, 0.6644] | 0.6018 [0.5759, 0.6275] | 0.7086 [0.6849, 0.7315] |
| pT2 | 0.8285 [0.8097, 0.8480] | 0.8742 [0.8613, 0.8875] | 0.3760 [0.3543, 0.3985] | 0.8672 [0.8519, 0.8825] |

## Printed only (X9)

`f_2` per arm: A3ign: full 0.8590 [0.8436, 0.8743], control 0.8617 [0.8472, 0.8761], early 0.4764 [0.4536, 0.4991], late 0.9126 [0.9011, 0.9245]; A3: full 0.8507 [0.8350, 0.8669], control 0.8647 [0.8510, 0.8788], early 0.4548 [0.4317, 0.4780], late 0.8863 [0.8728, 0.9004]; A0: full 0.8212 [0.8060, 0.8365], control 0.8419 [0.8282, 0.8557], early 0.4473 [0.4258, 0.4700], late 0.8893 [0.8756, 0.9029]

R1 by span length: 2-4s +0.1377 [+0.1075, +0.1677]; <2s +0.2012 [+0.1598, +0.2398]; >=4s -0.0146 [-0.0454, +0.0145]

D2 sensitivity, R1 with the leaky cache slice: +0.1329 [+0.1053, +0.1589]

Full-clip macros, kept vs dropped (78 dropped): A3ign kept 0.7934 [0.7745, 0.8119] / dropped 0.7769 [0.7237, 0.8265]; pT2 kept 0.8285 [0.8097, 0.8480] / dropped 0.7811 [0.7183, 0.8380]

Per-seed macros: A3ign/early: s2024 0.6429, s2025 0.6488, s2026 0.6442, s2027 0.6296, s2028 0.6254; A3ign/late: s2024 0.8280, s2025 0.8335, s2026 0.8238, s2027 0.8310, s2028 0.8321; A3/early: s2024 0.6168, s2025 0.6230, s2026 0.6016, s2027 0.5954, s2028 0.6022; A3/late: s2024 0.7913, s2025 0.7900, s2026 0.7930, s2027 0.7948, s2028 0.7975; A0/early: s2024 0.5620, s2025 0.5912, s2026 0.5803, s2027 0.5936, s2028 0.5992; A0/late: s2024 0.6525, s2025 0.7036, s2026 0.6617, s2027 0.7095, s2028 0.6975
