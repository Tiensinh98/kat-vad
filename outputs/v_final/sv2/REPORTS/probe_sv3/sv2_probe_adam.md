# SA²-v2 probe, AdamW state (rule `core/docs/final/SA2V2_RUN.md §9`)

`/content/drive/MyDrive/Thesis-Final/outputs/sv2/SV3/s2024` · step 1749 · lr 5e-05 · Σ lr = **0.04378** (the most one coordinate can move)

| param | numel | θ₀ (or ‖θ₀‖) | θ (or ‖θ‖) | RMS Δ / Σlr | grad RMS | share √v̂ < 10 eps | SNR (noise ≈) | reading |
|---|---|---|---|---|---|---|---|---|
| beta[0] | 1 | 1 | 0.9975 | 0.057 | 0.03617 | 0.000 | 0.389 (0.183) | lr: coherent gradient, budget unused |
| beta[1] | 1 | 0 | 0.002337 | 0.053 | 0.03766 | 0.000 | 0.164 (0.183) | noise: gradient alive, sign-incoherent |
| beta[2] | 1 | 0 | 0.001897 | 0.043 | 0.02879 | 0.000 | 0.291 (0.183) | noise: gradient alive, sign-incoherent |
| beta[3] | 1 | 0 | 0.001329 | 0.030 | 0.03693 | 0.000 | 0.177 (0.183) | noise: gradient alive, sign-incoherent |
| gamma | 1 | 0 | 0.001435 | 0.033 | 0.04483 | 0.000 | 0.012 (0.183) | noise: gradient alive, sign-incoherent |
| log_temperature | 1 | 0 | 0.03626 | 0.828 | 0.08917 | 0.000 | 0.547 (0.183) | moved |
| bias | 1 | 2 | 1.999 | 0.034 | 0.01075 | 0.000 | 0.104 (0.183) | noise: gradient alive, sign-incoherent |
| gate | 1 | 0 | 0.5286 | 0.483 | 0.2037 | 0.000 | 0.327 (0.183) | noise: gradient alive, sign-incoherent |
| w | 256 | 0 | 0.145 | 0.207 | 0.04905 | 0.000 | 0.255 (0.183) | noise: gradient alive, sign-incoherent |
| w_cell | 393472 | 20.95 | 21.11 | 0.110 | 0.005381 | 0.000 | 0.268 (0.183) | noise: gradient alive, sign-incoherent |
| w_prior | 1280 | 5.379 | 5.354 | 0.079 | 0.03011 | 0.000 | 0.397 (0.183) | lr: coherent gradient, budget unused |
| layer | 527104 | 42.81 | 43.19 | 0.139 | 0.002437 | 0.000 | 0.271 (0.183) | noise: gradient alive, sign-incoherent |
| branch_head | 513 | 0.5757 | 0.7153 | 0.441 | 0.004858 | 0.000 | 0.653 (0.183) | moved |
| w_u | 197120 | 0 | 1.121 | 0.058 | 0.008457 | 0.000 | 0.204 (0.183) | noise: gradient alive, sign-incoherent |

Scalars print |θ|: a signed value is in the JSON as the cell logits use it.
