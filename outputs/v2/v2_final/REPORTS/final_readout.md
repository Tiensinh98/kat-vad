# KAT-VAD v2 — Final read-out (sealed sets, opened once)

Rules: PREREG_ADDENDUM.md §19 (Amendment 10, P1-P8); proposal §10.2. Seeds s2024, s2025, s2026, s2027, s2028. Adopted **A3** (`F` = A0), closed at E3 (P1): nothing below re-decides. P7 gate: PASS (22 dev means reproduced).

Never "motion" (G3 / D6). A2/A3: "O1 FAIL as registered; not collapsed under O1'" (§17 Q5). §18 provenance: rules frozen in `59ee17e`, signed after the E3 read-out.

## Position control (P4/P5) — DoTA-CAP-eval (n/1397)

* **A3's gain over A0 survives the T2 position prior.**
* **The video stream's gain on top of CRN (A3 - A1) survives the T2 position prior.**
* **A3 carries signal beyond position.**

| arm | raw | w=1 | w=2 |
|---|---|---|---|
| `p_T2` alone (DoTA-CAP-eval (n/1397)) | 0.8256 | | |
| A0 (DoTA-CAP-eval (n/1397)) | 0.6979 | 0.8015 | 0.8273 |
| A1 (DoTA-CAP-eval (n/1397)) | 0.6835 | 0.8090 | 0.8314 |
| A2 (DoTA-CAP-eval (n/1397)) | 0.7651 | 0.8220 | 0.8312 |
| A3 (DoTA-CAP-eval (n/1397)) | 0.7785 | 0.8451 | 0.8484 |
| `p_T2` alone (DoTA-CAP-dev (n/1397)) | 0.8220 | | |
| A0 (DoTA-CAP-dev (n/1397)) | 0.6768 | 0.7915 | 0.8194 |
| A1 (DoTA-CAP-dev (n/1397)) | 0.6577 | 0.7977 | 0.8245 |
| A2 (DoTA-CAP-dev (n/1397)) | 0.7536 | 0.8168 | 0.8264 |
| A3 (DoTA-CAP-dev (n/1397)) | 0.7576 | 0.8365 | 0.8424 |

| Δ (clip-level, cluster bootstrap) | set | raw | w=1 | w=2 |
|---|---|---|---|---|
| A3 - A0 | DoTA-CAP-eval (n/1397) | +0.0806 [+0.0568, +0.1051] | +0.0437 [+0.0270, +0.0608] | +0.0211 [+0.0087, +0.0336] |
| A2 - A0 | DoTA-CAP-eval (n/1397) | +0.0672 [+0.0410, +0.0946] | +0.0205 [+0.0015, +0.0393] | +0.0039 [-0.0104, +0.0178] |
| A3 - A1 | DoTA-CAP-eval (n/1397) | +0.0950 [+0.0769, +0.1145] | +0.0361 [+0.0226, +0.0508] | +0.0170 [+0.0078, +0.0270] |
| A0 + 2p - `p_T2` | DoTA-CAP-eval (n/1397) | +0.0017 [-0.0107, +0.0137] | | |
| A1 + 2p - `p_T2` | DoTA-CAP-eval (n/1397) | +0.0057 [-0.0061, +0.0173] | | |
| A2 + 2p - `p_T2` | DoTA-CAP-eval (n/1397) | +0.0055 [-0.0041, +0.0147] | | |
| A3 + 2p - `p_T2` | DoTA-CAP-eval (n/1397) | +0.0227 [+0.0144, +0.0312] | | |
| A3 - A0 | DoTA-CAP-dev (n/1397) | +0.0808 [+0.0613, +0.1000] | +0.0451 [+0.0304, +0.0594] | +0.0230 [+0.0130, +0.0328] |
| A2 - A0 | DoTA-CAP-dev (n/1397) | +0.0768 [+0.0546, +0.0999] | +0.0253 [+0.0094, +0.0417] | +0.0069 [-0.0043, +0.0188] |
| A3 - A1 | DoTA-CAP-dev (n/1397) | +0.0999 [+0.0833, +0.1172] | +0.0388 [+0.0264, +0.0509] | +0.0179 [+0.0087, +0.0268] |
| A0 + 2p - `p_T2` | DoTA-CAP-dev (n/1397) | -0.0026 [-0.0155, +0.0103] | | |
| A1 + 2p - `p_T2` | DoTA-CAP-dev (n/1397) | +0.0025 [-0.0099, +0.0150] | | |
| A2 + 2p - `p_T2` | DoTA-CAP-dev (n/1397) | +0.0043 [-0.0044, +0.0132] | | |
| A3 + 2p - `p_T2` | DoTA-CAP-dev (n/1397) | +0.0204 [+0.0116, +0.0292] | | |

## Macro per arm (seed-averaged) and optimism

| arm | set | eval macro | dev macro | optimism (dev - eval) | micro (min-max) | AP |
|---|---|---|---|---|---|---|
| A0 | DoTA-eval | 0.6971 [0.6770, 0.7158] | 0.6793 [0.6592, 0.6996] | -0.0178 | 0.6455 | 0.4508 |
| A0 | DoTA-CAP-eval (n/1397) | 0.6979 [0.6744, 0.7195] | 0.6768 [0.6538, 0.6994] | -0.0211 | 0.6514 | 0.4472 |
| A1 | DoTA-eval | 0.6840 [0.6653, 0.7020] | 0.6602 [0.6426, 0.6779] | -0.0239 | 0.6565 | 0.4552 |
| A1 | DoTA-CAP-eval (n/1397) | 0.6835 [0.6611, 0.7050] | 0.6577 [0.6379, 0.6766] | -0.0258 | 0.6583 | 0.4490 |
| A2 | DoTA-CAP-eval (n/1397) | 0.7651 [0.7430, 0.7859] | 0.7536 [0.7351, 0.7725] | -0.0115 | 0.7163 | 0.5320 |
| A3 | DoTA-CAP-eval (n/1397) | 0.7785 [0.7627, 0.7937] | 0.7576 [0.7404, 0.7745] | -0.0209 | 0.7304 | 0.5237 |

DoTA-eval and DoTA-CAP-eval are different clip sets (D14): never compare them.

* DoTA-eval: scored 695 / 700 clips; excluded without CLIP features (§20): TNZv-NBcV5U_002389, TNZv-NBcV5U_002660, W6YrlYyWguc_005597, W6YrlYyWguc_005927, nADqn-DZ-Dc_000075
* DoTA-CAP-eval (n/1397): scored 560 / 560 clips; excluded without CLIP features (§20): none
* DoTA-eval: `p_T2` 0.8239 [0.8065, 0.8408]; `t/N` (monotone) 0.5754 [0.5554, 0.5956]
* DoTA-CAP-eval (n/1397): `p_T2` 0.8256 [0.8078, 0.8435]; `t/N` (monotone) 0.5617 [0.5409, 0.5832]
* `p_CAP` (dev, in-domain ceiling): 0.852

## Contrasts (descriptive: paired t95 over seeds; the decision was E3's)

| contrast | set | mean Δ | t95 | clip-level Δ |
|---|---|---|---|---|
| A1 - A0 | DoTA-eval | -0.0072 | [-0.0337, +0.0193] | -0.0131 [-0.0296, +0.0035] |
| A2 - A0 | DoTA-CAP-eval (n/1397) | +0.0871 | [+0.0574, +0.1168] | +0.0672 [+0.0410, +0.0946] |
| A3 - A0 | DoTA-CAP-eval (n/1397) | +0.0900 | [+0.0654, +0.1146] | +0.0806 [+0.0568, +0.1051] |
| A3 - A1 | DoTA-CAP-eval (n/1397) | +0.0982 | [+0.0896, +0.1069] | +0.0950 [+0.0769, +0.1145] |
| A2 - A1 | DoTA-CAP-eval (n/1397) | +0.0953 | [+0.0820, +0.1086] | +0.0816 [+0.0536, +0.1101] |

## Macro per accident-share bin

| arm | set | <30 | 30-50 | 50-70 | >70 |
|---|---|---|---|---|---|
| A0 | DoTA-eval | 0.7047 [0.6754, 0.7331] | 0.6781 [0.6494, 0.7067] | 0.7120 [0.6701, 0.7523] | 0.7356 [0.6185, 0.8369] |
| A0 | DoTA-CAP-eval (n/1397) | 0.7013 [0.6686, 0.7326] | 0.6805 [0.6482, 0.7120] | 0.7217 [0.6762, 0.7635] | 0.7509 [0.6104, 0.8790] |
| A1 | DoTA-eval | 0.6838 [0.6530, 0.7153] | 0.6653 [0.6349, 0.6945] | 0.7048 [0.6582, 0.7510] | 0.8240 [0.7527, 0.8933] |
| A1 | DoTA-CAP-eval (n/1397) | 0.6842 [0.6511, 0.7170] | 0.6662 [0.6287, 0.7010] | 0.7021 [0.6550, 0.7497] | 0.8385 [0.7347, 0.9278] |
| A2 | DoTA-CAP-eval (n/1397) | 0.7590 [0.7303, 0.7870] | 0.7757 [0.7408, 0.8072] | 0.7589 [0.6944, 0.8154] | 0.7812 [0.6392, 0.9059] |
| A3 | DoTA-CAP-eval (n/1397) | 0.7767 [0.7516, 0.8008] | 0.7727 [0.7423, 0.8016] | 0.7872 [0.7474, 0.8271] | 0.8653 [0.7686, 0.9436] |

## T2-test (1,106 windows; key secondary, never an adoption)

| arm | micro | macro | clip oracle | window AUC | O1' pass (of 5) | macro Δ vs A0 (t95) |
|---|---|---|---|---|---|---|
| A0 | 0.6228 | 0.6307 | 0.7037 | 0.6667 | — | — |
| A1 | 0.6594 | 0.6377 | 0.7037 | 0.7164 | 5 | +0.0070 [+0.0003, +0.0138] |
| A2 | 0.7153 | 0.7080 | 0.7037 | 0.7715 | 5 | +0.0774 [+0.0633, +0.0914] |
| A3 | 0.7095 | 0.6950 | 0.7037 | 0.7706 | 5 | +0.0643 [+0.0510, +0.0777] |
