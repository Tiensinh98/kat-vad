# SA²-v2 probe, live gradients — final (rule `core/docs/final/SA2V2_RUN.md §9`)

At /content/drive/MyDrive/Thesis-Final/outputs/sv2/SV2/s2024/checkpoint_last.pt; 8 batches of 64, epoch-0 order, train mode, fp32, no optimizer step, on `cuda`.

## Gradient on each scalar, per term (mean ± std over batches; coherence |mean|/rms)

| param | host | patch | branch | total |
|---|---|---|---|---|
| beta[0] | -4.71e-04 ± 4.8e-02 (0.01) | +8.18e-03 ± 1.2e-02 (0.56) | -5.22e-04 ± 7.4e-04 (0.58) | +7.18e-03 ± 5.4e-02 (0.13) |
| beta[1] | +1.80e-03 ± 7.1e-02 (0.03) | +1.25e-03 ± 5.3e-03 (0.23) | -1.06e-03 ± 2.8e-03 (0.35) | +1.99e-03 ± 7.7e-02 (0.03) |
| beta[2] | -6.61e-04 ± 4.0e-02 (0.02) | +7.02e-03 ± 1.4e-02 (0.46) | -3.77e-04 ± 5.7e-04 (0.55) | +5.98e-03 ± 5.0e-02 (0.12) |
| beta[3] | +1.15e-02 ± 7.3e-02 (0.16) | +4.87e-04 ± 5.7e-03 (0.08) | -9.08e-04 ± 3.0e-03 (0.29) | +1.10e-02 ± 8.0e-02 (0.14) |
| gamma | +4.93e-02 ± 1.2e-01 (0.39) | +4.22e-03 ± 6.0e-03 (0.58) | +1.94e-03 ± 3.5e-03 (0.49) | +5.54e-02 ± 1.2e-01 (0.42) |
| log_temperature | +3.54e-02 ± 1.4e-01 (0.24) | +0.00e+00 ± 0.0e+00 (0.00) | +1.08e-02 ± 4.0e-03 (0.94) | +4.62e-02 ± 1.5e-01 (0.30) |
| bias | +0.00e+00 ± 0.0e+00 (0.00) | +4.47e-03 ± 1.2e-02 (0.34) | +0.00e+00 ± 0.0e+00 (0.00) | +4.47e-03 ± 1.2e-02 (0.34) |

## `w` (the residual scorer)

| term | ‖∇w‖ |
|---|---|
| host | 1.027e+00 |
| patch | 1.752e-01 |
| branch | 8.286e-02 |
| total | 1.099e+00 |

| pair | cos on ∇w |
|---|---|
| host|patch | +0.131 |
| host|branch | +0.402 |
| patch|branch | -0.242 |

## Which part of `l` ranks the cells (mean per-frame std across 49 cells)

| part | spread |
|---|---|
| prior | 0.9919 |
| text | 0.0010 |
| residual | 3.2502 |
| l | 3.1685 |

Weighted loss (mean): host 0.5839, patch 0.1068, branch 0.1812, total 0.8718
