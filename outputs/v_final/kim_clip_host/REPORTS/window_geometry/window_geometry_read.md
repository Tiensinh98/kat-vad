# §67 Window-geometry read, DoTA-CAP-dev — **PRELIMINARY (n = 1 seed, in-domain); decides only §64.2's explanation, K5f stays**

569 clips (569 two-class), s1[::3]; windowed = 12 rows, hop 6, each window scored alone. Gate WG4 passed (whole = saved protocol B).

## Verdict (WG6): **PART GEOMETRY**

| contrast | Δ macro [95 % CI] | label |
|---|---|---|
| G = L5 windowed - L5 whole | +0.1775 [+0.1644, +0.1904] | GAIN |
| R = L5 windowed - L0 windowed | -0.0855 [-0.1013, -0.0696] | COST |
| L0 windowed - whole *(printed)* | -0.0007 [-0.0052, +0.0040] | NO DIFFERENCE |
| L5cap windowed - whole *(printed)* | -0.0070 [-0.0116, -0.0022] | NO DIFFERENCE |
| L5cap windowed - L0 windowed *(printed)* | +0.0080 [+0.0038, +0.0120] | NO DIFFERENCE |

## Rows

| row | whole macro | windowed macro | window AUC (top-k vs any-abnormal) | start spike, whole (rows 1-2 / median) |
|---|---|---|---|---|
| L5 = L5dfiDoTA | 0.4001 [0.3840, 0.4172] | 0.5777 [0.5643, 0.5907] | 0.6819 (2058/2867 positive) | 18.17 |
| L5cap = L5dficapDoTA | 0.6781 [0.6626, 0.6934] | 0.6711 [0.6565, 0.6853] | 0.6893 (2058/2867 positive) | 0.97 |
| L0 = L0dfiDoTArr | 0.6638 [0.6487, 0.6786] | 0.6631 [0.6490, 0.6771] | 0.6921 (2058/2867 positive) | 1.07 |

## Mean raw score by position inside a 12-row window (printed)

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L5 | 0.155 | 0.644 | 0.485 | 0.058 | 0.060 | 0.059 | 0.059 | 0.057 | 0.059 | 0.060 | 0.037 | 0.036 |
| L5cap | 0.186 | 0.211 | 0.220 | 0.230 | 0.229 | 0.234 | 0.236 | 0.242 | 0.239 | 0.245 | 0.250 | 0.249 |
| L0 | 0.150 | 0.165 | 0.176 | 0.190 | 0.192 | 0.196 | 0.199 | 0.205 | 0.205 | 0.215 | 0.219 | 0.213 |

Never: "KIM works on DoTA" from GEOMETRY (the deployable read is the whole clip); "the start zeros are the cause" (printed only); any zero-shot sentence; a seed claim.
