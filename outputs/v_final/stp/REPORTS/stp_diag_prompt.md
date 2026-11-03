# LaGoVAD-STP diagnostic D-1 — did the prompt train? (rule: `core/docs/final/STP_BUILD.md §5`)

| run | param | numel | ‖θ₀‖ | ‖θ‖ | ‖θ - θ₀‖ | cos(θ, θ₀) | RMS Δ / Σlr | grad RMS | median √v̂ | share √v̂ < 10 eps | SNR (noise ≈) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R4 | prompt | 16384 | 126.4 | 126.4 | 0.2127 | 1.00000 | 0.038 | 9.528e-05 | 5.294e-05 | 0.000 | 0.276 (0.183) |
| R4 | sa2 | 131072 | 14.31 | 14.52 | 2.013 | 0.99035 | 0.127 | 0.0003876 | 0.0002309 | 0.000 | 0.216 (0.183) |
| R4 | w_u | 197120 | 0 | 1.407 | 1.407 | — | 0.072 | 0.009653 | 0.006686 | 0.000 | 0.220 (0.183) |
| R6 | prompt | 4096 | 1.261 | 1.262 | 0.04547 | 0.99935 | 0.016 | 0.00925 | 0.007733 | 0.000 | 0.136 (0.183) |
| R6 | sa2 | 131072 | 14.3 | 14.52 | 2.136 | 0.98913 | 0.135 | 0.0004142 | 0.0002492 | 0.000 | 0.195 (0.183) |
| R6 | w_u | 197120 | 0 | 1.435 | 1.435 | — | 0.074 | 0.009342 | 0.006687 | 0.000 | 0.220 (0.183) |

| run | prompt layout | steps | Σ lr | ‖θ₀‖ rebuilt / logged step 0 | reading (§5 D-1) |
|---|---|---|---|---|---|
| R4 | released | 1749 | 0.04378 | 126.421272 / 126.421272 (ok) | D1-noise: gradient alive but sign-incoherent -- an LR multiplier would random-walk |
| R6 | prefix | 1749 | 0.04378 | 1.261225 / 1.261225 (ok) | D1-noise: gradient alive but sign-incoherent -- an LR multiplier would random-walk |
