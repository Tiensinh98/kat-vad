# SA²-v2 probe, live gradients — init (rule `core/docs/final/SA2V2_RUN.md §9`)

At init (rebuilt as core.train builds it); 8 batches of 64, epoch-0 order, train mode, fp32, no optimizer step, on `cuda`.

## Gradient on each scalar, per term (mean ± std over batches; coherence |mean|/rms)

| param | host | patch | branch | total |
|---|---|---|---|---|
| beta[0] | -9.53e-03 ± 3.2e-03 (0.95) | +3.63e-02 ± 3.4e-02 (0.73) | +1.89e-04 ± 4.0e-04 (0.43) | +2.70e-02 ± 3.4e-02 (0.62) |
| beta[1] | -2.07e-03 ± 6.6e-03 (0.30) | +7.21e-03 ± 9.3e-03 (0.61) | +1.45e-04 ± 2.7e-04 (0.47) | +5.29e-03 ± 1.1e-02 (0.43) |
| beta[2] | -6.42e-03 ± 1.8e-03 (0.96) | +1.51e-02 ± 3.0e-02 (0.45) | +1.27e-04 ± 2.5e-04 (0.46) | +8.79e-03 ± 3.0e-02 (0.28) |
| beta[3] | -2.82e-03 ± 6.2e-03 (0.41) | +5.09e-03 ± 9.8e-03 (0.46) | +1.36e-04 ± 2.2e-04 (0.53) | +2.41e-03 ± 1.2e-02 (0.20) |
| gamma | -1.70e-02 ± 1.1e-02 (0.85) | -1.55e-03 ± 3.9e-03 (0.37) | -1.02e-04 ± 1.7e-04 (0.51) | -1.86e-02 ± 8.6e-03 (0.91) |
| log_temperature | +9.53e-03 ± 3.2e-03 (0.95) | +0.00e+00 ± 0.0e+00 (0.00) | -1.89e-04 ± 4.0e-04 (0.43) | +9.34e-03 ± 3.2e-03 (0.95) |
| bias | +0.00e+00 ± 0.0e+00 (0.00) | +9.35e-03 ± 1.6e-02 (0.51) | +0.00e+00 ± 0.0e+00 (0.00) | +9.35e-03 ± 1.6e-02 (0.51) |

## `w` (the residual scorer)

| term | ‖∇w‖ |
|---|---|
| host | 6.842e-02 |
| patch | 2.194e-01 |
| branch | 1.347e-03 |
| total | 2.272e-01 |

| pair | cos on ∇w |
|---|---|
| host|patch | -0.082 |
| host|branch | -0.063 |
| patch|branch | +0.059 |

## Which part of `l` ranks the cells (mean per-frame std across 49 cells)

| part | spread |
|---|---|
| prior | 1.0000 |
| text | 0.0000 |
| residual | 0.0000 |
| l | 1.0000 |

Weighted loss (mean): host 2.8913, patch 0.1490, branch 0.2084, total 3.2487
