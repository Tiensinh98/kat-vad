# Micro level read (KIM_RUN.md §44.2) — printed, decides nothing, n = 1

## DoTA-CAP

| row | items | micro raw | low share | low share (min-max) | pos. frames in low | loud share | own AUC low / rest | low with own AUC ≥ 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K5f | 1129 | 0.6839 | 0.244 (275) | 0.094 | 0.250 | 0.176 | 0.688 / 0.866 | 0.785 |
| K5dfi | 1129 | 0.7207 | 0.190 (215) | 0.066 | 0.200 | 0.086 | 0.665 / 0.861 | 0.763 |

Overlap of low sets: both 200, K5f only 75, K5dfi only 15, Jaccard 0.690

### K5f: low share by stratum (Δ vs rest, 95 %)

| attribute | stratum | n | low share | mean own AUC | Δ | lead |
| --- | --- | --- | --- | --- | --- | --- |
| ego_involve | False | 502 | 0.434 | 0.749 | +0.343 [+0.290, +0.396] |  |
| ego_involve | True | 627 | 0.091 | 0.881 | -0.343 [-0.396, -0.290] |  |
| night | False | 996 | 0.253 | 0.820 | +0.080 [+0.018, +0.142] |  |
| night | True | 133 | 0.173 | 0.840 | -0.080 [-0.142, -0.018] |  |
| accident_name | lateral | 202 | 0.198 | 0.822 | -0.055 [-0.115, +0.008] |  |
| accident_name | leave_to_left | 69 | 0.348 | 0.757 | +0.111 [+0.016, +0.207] |  |
| accident_name | leave_to_right | 88 | 0.341 | 0.753 | +0.106 [+0.009, +0.211] |  |
| accident_name | moving_ahead_or_waiting | 154 | 0.182 | 0.823 | -0.072 [-0.132, -0.011] |  |
| accident_name | obstacle | 28 | 0.143 | 0.856 | -0.103 [-0.223, +0.037] |  |
| accident_name | oncoming | 111 | 0.072 | 0.868 | -0.190 [-0.242, -0.130] |  |
| accident_name | pedestrian | 27 | 0.407 | 0.773 | +0.168 [-0.028, +0.390] |  |
| accident_name | start_stop_or_stationary | 20 | 0.400 | 0.746 | +0.159 [-0.045, +0.370] |  |
| accident_name | too small | 19 | 0.421 | 0.761 | +0.181 [-0.029, +0.427] |  |
| accident_name | turning | 411 | 0.277 | 0.844 | +0.053 [-0.001, +0.101] |  |

| covariate | AUC low vs rest [95 %] | mean low / rest | lead |
| --- | --- | --- | --- |
| length | 0.506 [0.461, 0.549] | 101.262 / 100.491 |  |
| span share | 0.530 [0.491, 0.569] | 0.338 / 0.324 |  |
| span centre | 0.448 [0.405, 0.488] | 0.507 / 0.532 |  |
| span seconds | 0.532 [0.493, 0.569] | 3.374 / 3.253 |  |

### K5dfi: low share by stratum (Δ vs rest, 95 %)

| attribute | stratum | n | low share | mean own AUC | Δ | lead |
| --- | --- | --- | --- | --- | --- | --- |
| ego_involve | False | 502 | 0.359 | 0.751 | +0.303 [+0.249, +0.357] |  |
| ego_involve | True | 627 | 0.056 | 0.882 | -0.303 [-0.357, -0.249] |  |
| night | False | 996 | 0.196 | 0.823 | +0.045 [-0.018, +0.106] |  |
| night | True | 133 | 0.150 | 0.832 | -0.045 [-0.106, +0.018] |  |
| accident_name | lateral | 202 | 0.139 | 0.823 | -0.063 [-0.120, -0.008] |  |
| accident_name | leave_to_left | 69 | 0.319 | 0.755 | +0.137 [+0.035, +0.243] |  |
| accident_name | leave_to_right | 88 | 0.307 | 0.764 | +0.126 [+0.026, +0.224] |  |
| accident_name | moving_ahead_or_waiting | 154 | 0.143 | 0.816 | -0.055 [-0.107, -0.002] |  |
| accident_name | obstacle | 28 | 0.071 | 0.882 | -0.122 [-0.204, -0.011] |  |
| accident_name | oncoming | 111 | 0.054 | 0.870 | -0.151 [-0.199, -0.098] |  |
| accident_name | pedestrian | 27 | 0.185 | 0.781 | -0.005 [-0.165, +0.189] |  |
| accident_name | start_stop_or_stationary | 20 | 0.400 | 0.776 | +0.213 [-0.024, +0.448] |  |
| accident_name | too small | 19 | 0.211 | 0.773 | +0.020 [-0.148, +0.221] |  |
| accident_name | turning | 411 | 0.221 | 0.843 | +0.049 [+0.003, +0.095] |  |

| covariate | AUC low vs rest [95 %] | mean low / rest | lead |
| --- | --- | --- | --- |
| length | 0.528 [0.479, 0.577] | 102.791 / 100.182 |  |
| span share | 0.537 [0.492, 0.583] | 0.343 / 0.324 |  |
| span centre | 0.432 [0.389, 0.475] | 0.499 / 0.532 |  |
| span seconds | 0.546 [0.503, 0.588] | 3.450 / 3.243 |  |

Leads: none

## DADA-2000

| row | items | micro raw | low share | low share (min-max) | pos. frames in low | loud share | own AUC low / rest | low with own AUC ≥ 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K5f | 370 | 0.7065 | 0.235 (87) | 0.103 | 0.241 | 0.170 | 0.546 / 0.845 | 0.529 |
| K5dfi | 370 | 0.7218 | 0.186 (69) | 0.103 | 0.189 | 0.116 | 0.524 / 0.824 | 0.551 |

Overlap of low sets: both 66, K5f only 21, K5dfi only 3, Jaccard 0.733

### K5f: low share by stratum (Δ vs rest, 95 %)

| attribute | stratum | n | low share | mean own AUC | Δ | lead |
| --- | --- | --- | --- | --- | --- | --- |
| light | 1 | 334 | 0.249 | 0.776 | +0.137 [+0.021, +0.241] |  |
| light | 2 | 36 | 0.111 | 0.759 | -0.137 [-0.241, -0.021] |  |
| weather | 1 | 345 | 0.232 | 0.778 | -0.048 [-0.230, +0.126] |  |
| weather | 2 | 22 | 0.273 | 0.723 | +0.040 [-0.145, +0.240] |  |
| weather | too small | 3 | 0.333 | 0.792 | +0.099 [-0.264, +0.780] |  |
| scenes | 1 | 33 | 0.152 | 0.733 | -0.092 [-0.215, +0.054] |  |
| scenes | 4 | 252 | 0.266 | 0.768 | +0.096 [+0.008, +0.180] |  |
| scenes | 5 | 70 | 0.186 | 0.823 | -0.061 [-0.157, +0.050] |  |
| scenes | too small | 15 | 0.133 | 0.745 | -0.106 [-0.257, +0.113] |  |
| linear | 1 | 164 | 0.232 | 0.757 | -0.006 [-0.088, +0.085] |  |
| linear | 3 | 116 | 0.310 | 0.752 | +0.110 [+0.005, +0.207] |  |
| linear | 4 | 71 | 0.141 | 0.852 | -0.117 [-0.208, -0.019] |  |
| linear | too small | 19 | 0.158 | 0.770 | -0.081 [-0.242, +0.113] |  |
| ego | False | 275 | 0.269 | 0.752 | +0.132 [+0.040, +0.218] |  |
| ego | True | 95 | 0.137 | 0.840 | -0.132 [-0.218, -0.040] |  |
| actor | motorbike | 92 | 0.326 | 0.760 | +0.121 [+0.006, +0.226] |  |
| actor | other | 97 | 0.196 | 0.834 | -0.053 [-0.142, +0.041] |  |
| actor | vehicle | 181 | 0.210 | 0.750 | -0.049 [-0.133, +0.037] |  |

| covariate | AUC low vs rest [95 %] | mean low / rest | lead |
| --- | --- | --- | --- |
| length | 0.529 [0.462, 0.596] | 45.770 / 42.784 |  |
| span share | 0.495 [0.431, 0.562] | 0.301 / 0.314 |  |
| span centre | 0.409 [0.343, 0.479] | 0.480 / 0.543 |  |
| span seconds | 0.543 [0.477, 0.608] | 3.433 / 3.319 |  |

### K5dfi: low share by stratum (Δ vs rest, 95 %)

| attribute | stratum | n | low share | mean own AUC | Δ | lead |
| --- | --- | --- | --- | --- | --- | --- |
| light | 1 | 334 | 0.192 | 0.773 | +0.053 [-0.075, +0.166] |  |
| light | 2 | 36 | 0.139 | 0.721 | -0.053 [-0.166, +0.075] |  |
| weather | 1 | 345 | 0.186 | 0.770 | -0.014 [-0.197, +0.145] |  |
| weather | 2 | 22 | 0.227 | 0.727 | +0.043 [-0.136, +0.239] |  |
| weather | too small | 3 | 0.000 | 0.779 | -0.188 [-0.227, -0.149] |  |
| scenes | 1 | 33 | 0.152 | 0.724 | -0.038 [-0.164, +0.107] |  |
| scenes | 4 | 252 | 0.210 | 0.754 | +0.075 [-0.009, +0.150] |  |
| scenes | 5 | 70 | 0.143 | 0.835 | -0.054 [-0.142, +0.043] |  |
| scenes | too small | 15 | 0.067 | 0.773 | -0.125 [-0.223, +0.039] |  |
| linear | 1 | 164 | 0.189 | 0.740 | +0.005 [-0.070, +0.089] |  |
| linear | 3 | 116 | 0.250 | 0.761 | +0.093 [+0.003, +0.181] |  |
| linear | 4 | 71 | 0.099 | 0.836 | -0.109 [-0.189, -0.024] |  |
| linear | too small | 19 | 0.105 | 0.790 | -0.086 [-0.210, +0.085] |  |
| ego | False | 275 | 0.215 | 0.748 | +0.109 [+0.027, +0.187] |  |
| ego | True | 95 | 0.105 | 0.826 | -0.109 [-0.187, -0.027] |  |
| actor | motorbike | 92 | 0.261 | 0.752 | +0.099 [-0.004, +0.200] |  |
| actor | other | 97 | 0.144 | 0.840 | -0.057 [-0.142, +0.027] |  |
| actor | vehicle | 181 | 0.171 | 0.737 | -0.030 [-0.108, +0.052] |  |

| covariate | AUC low vs rest [95 %] | mean low / rest | lead |
| --- | --- | --- | --- |
| length | 0.516 [0.442, 0.589] | 45.116 / 43.113 |  |
| span share | 0.495 [0.422, 0.564] | 0.300 / 0.313 |  |
| span centre | 0.383 [0.313, 0.457] | 0.462 / 0.543 | LEAD |
| span seconds | 0.522 [0.451, 0.596] | 3.389 / 3.336 |  |

Leads: K5dfi span centre

