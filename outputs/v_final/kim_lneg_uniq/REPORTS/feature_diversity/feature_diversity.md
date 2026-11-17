# Within-video feature diversity -- K5Lnegu vs K5, `nexar_val`

Rule: `core/docs/final/KIM_RUN.md §28.2`. n = 1 checkpoint per arm (K5 step 1749, K5Lnegu step 1749). Inference only.

**Verdict (FD6): NOT SUPPORTED** -- SUPPORTED iff Δ sep CI < 0 and Δ row_cos CI > 0; OPPOSITE iff Δ sep CI > 0.

| metric (per video, unit rows) | K5 | K5Lnegu | Δ K5Lnegu - K5 [95 %] | share of videos K5Lnegu lower |
|---|---|---|---|---|
| sep (decides) | 2.6290 | 0.9032 | -1.7258 [-2.0369, -1.4464] | 0.946 |
| row_cos (decides) | 0.7007 | 0.6319 | -0.0688 [-0.0786, -0.0584] | 0.884 |
| eff_rank (printed) | 135.3144 | 148.9000 | +13.5856 [+12.3126, +14.9724] | 0.000 |

Videos scored 112 of 112 positives; skipped 0 ([]). Frame labels (D-N1) measure only.
