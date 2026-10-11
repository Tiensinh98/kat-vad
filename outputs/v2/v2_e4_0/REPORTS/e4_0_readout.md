# v2.1 E4-0 — A3ign vs A3 on DoTA-CAP (n/1397) dev

Rule: core/docs/v2.1/KAT\_VAD\_PROPOSAL\_v2.1.md §11.4 (fixed before any A3-ign number). **Base B \= A3ign.**

## Decision

| check | result |
| :---- | :---- |
| macro\_t95\_above\_0 | PASS |
| f2\_point\_ge\_0 | PASS |
| guardrails\_hold | PASS |

&nbsp;

- macro A3 0.7576 · A3ign 0.7912 · `p_T2` 0.8220  
- paired t95 (A3ign \- A3): \+0.0398 \[+0.0347, \+0.0450\] (n \= 5); per seed s2024 \+0.0427, s2025 \+0.0420, s2026 \+0.0428, s2027 \+0.0332, s2028 \+0.0384  
- clip-level raw Δ (printed): \+0.0335 \[+0.0247, \+0.0420\]  
- `f_2` Δ: \+0.0102 \[+0.0048, \+0.0154\]

## Guardrails (T2-val, seed means)

| read | A3 | A3ign | pass |
| :---- | :---- | :---- | :---- |
| macro | 0.7034 | 0.7338 | True |
| normal-window peak | 0.6024 | 0.6151 | True |
| O1' every seed | — | {'s2024': True, 's2025': True, 's2026': True, 's2027': True, 's2028': True} | True |

&nbsp;

## Mechanism (DoTA-CAP-dev, seed-averaged, per-clip min-max; never moves B)

| read | A3 | A3ign | Δ \[95 % cluster CI\] |
| :---- | :---- | :---- | :---- |
| argmax\_in\_span | 0.6134 | 0.6643 | \+0.0510 \[+0.0197, \+0.0836\] |
| pre\_mean | 0.5270 | 0.2392 | \-0.2878 \[-0.3010, \-0.2744\] |
| in\_mean | 0.8635 | 0.7085 | \-0.1550 \[-0.1677, \-0.1432\] |
| tail-clip macro Δ (277 clips) |  |  | \+0.0315 \[+0.0203, \+0.0427\] |

&nbsp;

- pre\_mean\_falls: True  
- in\_mean\_does\_not\_fall: False  
- argmax\_in\_span\_rises: True  
- tail\_macro\_does\_not\_fall: True

## Claims

- Dropping the whole-anchor DVS label improves DoTA-CAP (n/1397) dev detection.  
- ... beyond the T2 position prior (f\_2 interval excludes 0).  
- No Gap S claim: not every §7 mechanism condition held.  
- A3-ign is the new best adoptable free arm F; later costly arms must beat it.