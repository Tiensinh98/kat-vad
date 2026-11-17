# K5Lnegu (one meaningless target per video) vs K5 and K5Lneg -- s2024

Rule: `core/docs/final/KIM_RUN.md §28.3`. PRELIMINARY, n = 1. NG4: K5's nexar_val re-read 0.8981 vs §13 0.8981.

**U2 verdict: MIXED** -- A iff K5Lnegu - K5 is below on none of ['nexar', 'dota', 'dada', 'dad']; B iff below on >= 3 and K5Lnegu - K5Lneg above on <= 1; else MIXED. Below K5 on ['nexar', 'dota', 'dada', 'dad']; above K5Lneg on ['nexar', 'dota', 'dada', 'dad'].

| set (macro; Nexar = val crops) | K5Lnegu - K5 | K5Lnegu - K5Lneg | K5Lneg - K5 (= §28.1) |
|---|---|---|---|
| nexar | -0.0696 [-0.0826, -0.0574] below | +0.0506 [+0.0342, +0.0673] above | -0.1202 [-0.1402, -0.1009] below |
| dota | -0.0252 [-0.0360, -0.0142] below | +0.0213 [+0.0134, +0.0296] above | -0.0465 [-0.0581, -0.0347] below |
| dada | -0.0526 [-0.0642, -0.0407] below | +0.0271 [+0.0157, +0.0387] above | -0.0797 [-0.0938, -0.0652] below |
| tad (printed) | -0.0308 [-0.0612, -0.0007] below | +0.0028 [-0.0245, +0.0300] not separated | -0.0335 [-0.0713, +0.0034] not separated |
| dad | -0.0387 [-0.0702, -0.0075] below | +0.0233 [+0.0009, +0.0455] above | -0.0620 [-0.0944, -0.0305] below |

Printed: P2 `sep` K5Lnegu - K5 = -1.7258 [-2.0369, -1.4464] (2.6290 -> 0.9032); FD6 NOT SUPPORTED.

A: the shared class target is what costs; good per-video captions *may* help (uniq is the extreme: real captions are correlated). B: the pooled-feature objective costs whatever the text; per-video captions under this loss form are closed. MIXED: no claim.
