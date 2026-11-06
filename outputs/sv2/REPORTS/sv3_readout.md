# SV3 read-out: gated SA²-v2, Nexar-trained, seed 2024

Rule: `core/docs/final/SA2V2_RUN.md §10`, written before any SV3 number. PRELIMINARY (n = 1 seed): adopts nothing.

Gate tanh(g): first +0.0000 → last **-0.4843** (max |·| 0.4845) → **opened** (bar |tanh g| ≥ 0.2).

## DADA-2000 (370 sources, whole video, s8)

| Row | Macro |
| --- | --- |
| W0 | 0.7615 [0.7369, 0.7859] |
| W0D | 0.7529 [0.7283, 0.7773] |
| SV2 | 0.7133 [0.6883, 0.7379] |
| SV3 | 0.7287 [0.7035, 0.7533] |

| Contrast (per video) | What | Δ | Sentence |
| --- | --- | --- | --- |
| SV3 - W0D | gated module on the dropout host (primary) | -0.0242 [-0.0383, -0.0103] | cost shown |
| SV3 - SV2 | what the gate changed | +0.0154 [+0.0026, +0.0279] | gain shown |
| SV3 - W0 | printed | -0.0328 [-0.0507, -0.0152] | cost shown |

**Verdict (§10.3): opened and costs: training wants x^AS, transfer does not, PRELIMINARY (n = 1 seed).**

## DoTA-CAP-dev (protocol B)

| Row | Macro |
| --- | --- |
| W0 | 0.8110 [0.7949, 0.8271] |
| W0D | 0.8155 [0.7995, 0.8315] |
| SV2 | 0.7848 [0.7681, 0.8016] |
| SV3 | 0.7863 [0.7692, 0.8032] |

| Contrast (per clip) | What | Δ | Sentence |
| --- | --- | --- | --- |
| SV3 - W0D | gated module on the dropout host (primary) | -0.0292 [-0.0396, -0.0187] | cost shown |
| SV3 - SV2 | what the gate changed | +0.0015 [-0.0073, +0.0103] | no difference shown |
| SV3 - W0 | printed | -0.0247 [-0.0359, -0.0134] | cost shown |

**Verdict (§10.3): opened and costs: training wants x^AS, transfer does not, PRELIMINARY (n = 1 seed).**

