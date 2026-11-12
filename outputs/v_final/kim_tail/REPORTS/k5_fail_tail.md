# K5's failing tail (`KIM_RUN.md` §24)

PRELIMINARY (n = 1, K5 s2024). Fail = own frame AUC < 0.5. Cluster bootstrap 2000 draws, seed 2026. Lead rules FT5; about one false lead is expected by chance.

## DoTA-CAP-dev: 47 / 569 fail (0.083); 91 clusters

| attribute | stratum | n | fail share | mean AUC | Δ share vs rest [95 %] | lead |
|---|---|---:|---:|---:|---|---|
| ego_involve | False | 248 | 0.129 | 0.753 | +0.082 [+0.034, +0.135] |  |
| ego_involve | True | 321 | 0.047 | 0.871 | -0.082 [-0.135, -0.034] |  |
| night | False | 503 | 0.080 | 0.820 | -0.027 [-0.106, +0.042] |  |
| night | True | 66 | 0.106 | 0.820 | +0.027 [-0.042, +0.106] |  |
| accident_name | lateral | 106 | 0.066 | 0.831 | -0.020 [-0.079, +0.044] |  |
| accident_name | leave_to_left | 35 | 0.200 | 0.707 | +0.125 [+0.003, +0.250] | **lead** |
| accident_name | leave_to_right | 43 | 0.163 | 0.738 | +0.087 [-0.016, +0.202] |  |
| accident_name | moving_ahead_or_waiting | 64 | 0.078 | 0.811 | -0.005 [-0.076, +0.086] |  |
| accident_name | oncoming | 59 | 0.034 | 0.858 | -0.054 [-0.103, +0.010] |  |
| accident_name | other | 59 | 0.153 | 0.781 | +0.078 [-0.010, +0.174] |  |
| accident_name | turning | 203 | 0.049 | 0.853 | -0.052 [-0.091, -0.010] |  |

| covariate | mean (fail / rest) | AUC fail vs rest [95 %] | lead |
|---|---|---|---|
| length | 107.149 / 99.473 | 0.603 [0.506, 0.695] | **lead** |
| span share | 0.357 / 0.327 | 0.541 [0.445, 0.625] |  |
| span centre | 0.522 / 0.521 | 0.526 [0.440, 0.619] |  |
| span seconds | 3.991 / 3.210 | 0.564 [0.460, 0.657] |  |

Peak of the failing items: before span 0.447, inside span 0.191, after span 0.362.

Shared difficulty: Spearman rho(K5, LaGoVAD per-item AUC) = 0.134; LaGoVAD fails on 0.319 of K5's failing items against 0.295 overall.

## DADA-2000: 59 / 370 fail (0.159); 370 clusters

| attribute | stratum | n | fail share | mean AUC | Δ share vs rest [95 %] | lead |
|---|---|---:|---:|---:|---|---|
| light | 1 | 334 | 0.159 | 0.776 | -0.008 [-0.147, +0.110] |  |
| light | 2 | 36 | 0.167 | 0.752 | +0.008 [-0.110, +0.147] |  |
| weather | 1 | 345 | 0.157 | 0.777 | -0.043 [-0.215, +0.108] |  |
| weather | 2 | 22 | 0.182 | 0.729 | +0.024 [-0.133, +0.199] |  |
| weather | too small | 3 | 0.333 | 0.792 | +0.175 [-0.187, +0.851] |  |
| scenes | 1 | 33 | 0.242 | 0.735 | +0.091 [-0.052, +0.248] |  |
| scenes | 4 | 252 | 0.171 | 0.766 | +0.035 [-0.045, +0.109] |  |
| scenes | 5 | 70 | 0.100 | 0.825 | -0.073 [-0.150, +0.011] |  |
| scenes | too small | 15 | 0.067 | 0.751 | -0.097 [-0.195, +0.073] |  |
| linear | 1 | 164 | 0.189 | 0.755 | +0.053 [-0.018, +0.134] |  |
| linear | 3 | 116 | 0.181 | 0.755 | +0.031 [-0.048, +0.111] |  |
| linear | 4 | 71 | 0.085 | 0.851 | -0.093 [-0.168, -0.014] |  |
| linear | too small | 19 | 0.053 | 0.770 | -0.113 [-0.196, +0.011] |  |
| ego | False | 275 | 0.178 | 0.753 | +0.073 [-0.004, +0.147] |  |
| ego | True | 95 | 0.105 | 0.835 | -0.073 [-0.147, +0.004] |  |
| actor | motorbike | 92 | 0.152 | 0.760 | -0.010 [-0.095, +0.075] |  |
| actor | other | 97 | 0.113 | 0.838 | -0.062 [-0.139, +0.020] |  |
| actor | vehicle | 181 | 0.188 | 0.747 | +0.056 [-0.016, +0.133] |  |

| covariate | mean (fail / rest) | AUC fail vs rest [95 %] | lead |
|---|---|---|---|
| length | 46.610 / 42.894 | 0.557 [0.480, 0.633] |  |
| span share | 0.300 / 0.313 | 0.487 [0.415, 0.560] |  |
| span centre | 0.492 / 0.535 | 0.454 [0.368, 0.543] |  |
| span seconds | 3.530 / 3.311 | 0.547 [0.463, 0.629] |  |
| risk-phase seconds | 1.706 / 1.669 | 0.501 [0.423, 0.583] |  |

Peak of the failing items: before span 0.441, inside span 0.085, after span 0.475.

Shared difficulty: Spearman rho(K5, LaGoVAD per-item AUC) = 0.227; LaGoVAD fails on 0.492 of K5's failing items against 0.330 overall.

**Leads (FT5):** DoTA-CAP-dev accident_name = leave_to_left; DoTA-CAP-dev length.

**FT6:** leads found: Claude describes them and may propose one targeted check; the user decides.
