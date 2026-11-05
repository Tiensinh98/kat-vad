# W0D read-out: motion dropout vs spatial module, Nexar-trained, seed 2024

Rule: `core/docs/final/SA2V2_RUN.md §8`, written before any W0D number. PRELIMINARY (n = 1 seed): adopts nothing.

## DADA-2000 (370 sources, whole video, s8)

| Row | Macro |
| --- | --- |
| W0 | 0.7615 [0.7369, 0.7859] |
| W0D | 0.7529 [0.7283, 0.7773] |
| SV2 | 0.7133 [0.6883, 0.7379] |

| Contrast (per video) | What | Δ | Sentence |
| --- | --- | --- | --- |
| W0D - W0 | dropout alone | -0.0086 [-0.0204, +0.0027] | no difference shown |
| SV2 - W0D | spatial module on W0D | -0.0396 [-0.0555, -0.0238] | cost shown |
| SV2 - W0 | both (printed, = §7) | -0.0482 [-0.0674, -0.0290] | cost shown |

**Attribution (§8.2): the cost is the spatial module's; dropout alone shows none, PRELIMINARY (n = 1 seed).**

## DoTA-CAP-dev (protocol B)

| Row | Macro |
| --- | --- |
| W0 | 0.8110 [0.7949, 0.8271] |
| W0D | 0.8155 [0.7995, 0.8315] |
| SV2 | 0.7848 [0.7681, 0.8016] |

| Contrast (per clip) | What | Δ | Sentence |
| --- | --- | --- | --- |
| W0D - W0 | dropout alone | +0.0045 [-0.0021, +0.0110] | no difference shown |
| SV2 - W0D | spatial module on W0D | -0.0307 [-0.0430, -0.0183] | cost shown |
| SV2 - W0 | both (printed, = §7) | -0.0262 [-0.0400, -0.0122] | cost shown |

**Attribution (§8.2): the cost is the spatial module's; dropout alone shows none, PRELIMINARY (n = 1 seed).**

