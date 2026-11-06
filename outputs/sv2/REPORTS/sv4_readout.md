# SV4 read-out: SA²-v2 through W_p, Nexar-trained, seed 2024

Rule: `core/docs/final/SA2V2_RUN.md §11`, written before any SV4 number. PRELIMINARY (n = 1 seed): adopts nothing.

W_p: share ‖W_p x^AS‖ / ‖c‖ over the last 100 steps **0.1247** → **used** (bar 0.05); ‖W_p‖ 1.3728.

`mil`, last 100 steps: SV4 0.1813, W0D 0.3003.

Paired with W0D at step 1 (|Δ| ≤ 0.001): **NO** (mil 0.69059 vs 0.69497 · dvs_sup 0.70385 vs 0.70297).

## DADA-2000 (370 sources, whole video, s8)

| Row | Macro |
| --- | --- |
| W0 | 0.7615 [0.7369, 0.7859] |
| W0D | 0.7529 [0.7283, 0.7773] |
| SV3 | 0.7287 [0.7035, 0.7533] |
| SV4 | 0.7390 [0.7146, 0.7620] |

| Contrast (per video) | What | Δ | Sentence |
| --- | --- | --- | --- |
| SV4 - W0D | W_p module on the dropout host (primary) | -0.0139 [-0.0274, -0.0007] | cost shown |
| SV4 - SV3 | matrix vs gate (printed; other scorer draws, D13) | +0.0103 [-0.0032, +0.0241] | no difference shown |
| SV4 - W0 | printed | -0.0225 [-0.0379, -0.0072] | cost shown |

**Verdict (§11.3): used and costs, PRELIMINARY (n = 1 seed).** Overfit tell: yes.

Printed: SV4's middle ruler (no pixels) 0.7342 [0.7076, 0.7594].

## DoTA-CAP-dev (protocol B)

| Row | Macro |
| --- | --- |
| W0 | 0.8110 [0.7949, 0.8271] |
| W0D | 0.8155 [0.7995, 0.8315] |
| SV3 | 0.7863 [0.7692, 0.8032] |
| SV4 | 0.8092 [0.7928, 0.8256] |

| Contrast (per clip) | What | Δ | Sentence |
| --- | --- | --- | --- |
| SV4 - W0D | W_p module on the dropout host (primary) | -0.0063 [-0.0153, +0.0026] | no difference shown |
| SV4 - SV3 | matrix vs gate (printed; other scorer draws, D13) | +0.0229 [+0.0122, +0.0334] | gain shown |
| SV4 - W0 | printed | -0.0018 [-0.0127, +0.0090] | no difference shown |

**Verdict (§11.3): used without a shown effect, PRELIMINARY (n = 1 seed).** Overfit tell: no.

**Next (§11.4): the token line closes: no further arm, gate, LR or fusion variant on tokens.**
