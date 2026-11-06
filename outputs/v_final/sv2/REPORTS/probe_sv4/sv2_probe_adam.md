# SA²-v2 probe, AdamW state (rule `core/docs/final/SA2V2_RUN.md §9`)

`/content/drive/MyDrive/Thesis-Final/outputs/sv2/SV4/s2024` · step 1749 · lr 5e-05 · Σ lr = **0.04378** (the most one coordinate can move)

| param | numel | θ₀ (or ‖θ₀‖) | θ (or ‖θ‖) | RMS Δ / Σlr | grad RMS | share √v̂ < 10 eps | SNR (noise ≈) | reading |
|---|---|---|---|---|---|---|---|---|
| beta[0] | 1 | 1 | 0.9942 | 0.132 | 0.08581 | 0.000 | 0.087 (0.183) | noise: gradient alive, sign-incoherent |
| beta[1] | 1 | 0 | 0.002112 | 0.048 | 0.08346 | 0.000 | 0.090 (0.183) | noise: gradient alive, sign-incoherent |
| beta[2] | 1 | 0 | 0.0036 | 0.082 | 0.06173 | 0.000 | 0.002 (0.183) | noise: gradient alive, sign-incoherent |
| beta[3] | 1 | 0 | 0.001138 | 0.026 | 0.08241 | 0.000 | 0.138 (0.183) | noise: gradient alive, sign-incoherent |
| gamma | 1 | 0 | 0.0004043 | 0.009 | 0.1191 | 0.000 | 0.040 (0.183) | noise: gradient alive, sign-incoherent |
| log_temperature | 1 | 0 | 0.009058 | 0.207 | 0.2363 | 0.000 | 0.126 (0.183) | noise: gradient alive, sign-incoherent |
| bias | 1 | 2 | 1.996 | 0.098 | 0.01035 | 0.000 | 0.282 (0.183) | noise: gradient alive, sign-incoherent |
| w | 256 | 0 | 0.1182 | 0.169 | 0.1497 | 0.000 | 0.079 (0.183) | noise: gradient alive, sign-incoherent |
| w_cell | 393472 | 20.93 | 21.06 | 0.101 | 0.007278 | 0.000 | 0.131 (0.183) | noise: gradient alive, sign-incoherent |
| w_prior | 1280 | 5.29 | 5.271 | 0.075 | 0.03763 | 0.000 | 0.107 (0.183) | noise: gradient alive, sign-incoherent |
| layer | 527104 | 42.83 | 43.11 | 0.119 | 0.003061 | 0.000 | 0.134 (0.183) | noise: gradient alive, sign-incoherent |
| branch_head | 513 | 0.6055 | 0.7895 | 0.534 | 0.004514 | 0.000 | 0.510 (0.183) | moved |
| w_p | 262656 | 0 | 1.373 | 0.061 | 0.006044 | 0.000 | 0.135 (0.183) | noise: gradient alive, sign-incoherent |
| w_u | 197120 | 0 | 1.056 | 0.054 | 0.006399 | 0.000 | 0.157 (0.183) | noise: gradient alive, sign-incoherent |

Scalars print |θ|: a signed value is in the JSON as the cell logits use it.
