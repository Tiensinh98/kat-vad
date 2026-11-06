# SA²-v2 probe, AdamW state (rule `core/docs/final/SA2V2_RUN.md §9`)

`/content/drive/MyDrive/Thesis-Final/outputs/sv2/SV2/s2024` · step 1749 · lr 5e-05 · Σ lr = **0.04378** (the most one coordinate can move)

| param | numel | θ₀ (or ‖θ₀‖) | θ (or ‖θ‖) | RMS Δ / Σlr | grad RMS | share √v̂ < 10 eps | SNR (noise ≈) | reading |
|---|---|---|---|---|---|---|---|---|
| beta[0] | 1 | 1 | 0.9941 | 0.134 | 0.04742 | 0.000 | 0.243 (0.183) | noise: gradient alive, sign-incoherent |
| beta[1] | 1 | 0 | 0.0004057 | 0.009 | 0.08152 | 0.000 | 0.229 (0.183) | noise: gradient alive, sign-incoherent |
| beta[2] | 1 | 0 | 0.003465 | 0.079 | 0.03747 | 0.000 | 0.141 (0.183) | noise: gradient alive, sign-incoherent |
| beta[3] | 1 | 0 | 5.487e-05 | 0.001 | 0.07935 | 0.000 | 0.177 (0.183) | noise: gradient alive, sign-incoherent |
| gamma | 1 | 0 | 0.00102 | 0.023 | 0.1174 | 0.000 | 0.087 (0.183) | noise: gradient alive, sign-incoherent |
| log_temperature | 1 | 0 | 0.0208 | 0.475 | 0.1388 | 0.000 | 0.030 (0.183) | noise: gradient alive, sign-incoherent |
| bias | 1 | 2 | 2.001 | 0.013 | 0.01039 | 0.000 | 0.137 (0.183) | noise: gradient alive, sign-incoherent |
| w | 256 | 0 | 0.1449 | 0.207 | 0.07839 | 0.000 | 0.079 (0.183) | noise: gradient alive, sign-incoherent |
| w_cell | 393472 | 20.95 | 21.1 | 0.109 | 0.006512 | 0.000 | 0.126 (0.183) | noise: gradient alive, sign-incoherent |
| w_prior | 1280 | 5.379 | 5.357 | 0.079 | 0.03535 | 0.000 | 0.099 (0.183) | noise: gradient alive, sign-incoherent |
| layer | 527104 | 42.81 | 43.17 | 0.135 | 0.003028 | 0.000 | 0.136 (0.183) | noise: gradient alive, sign-incoherent |
| branch_head | 513 | 0.5757 | 0.7629 | 0.516 | 0.004586 | 0.000 | 0.548 (0.183) | moved |
| w_u | 197120 | 0 | 1.217 | 0.063 | 0.006343 | 0.000 | 0.152 (0.183) | noise: gradient alive, sign-incoherent |

Scalars print |θ|: a signed value is in the JSON as the cell logits use it.
