# DF8 — same frames, different position (post-hoc)

DoTA-CAP-dev shift subset, n = 491. post-hoc, printed only; decides nothing (v2 addendum §24's verdict is X8's).

Shift = AUC of the crop's own scores - AUC of the full clip's scores on the same frames (content and labels fixed). `pT2` is the pure-position ruler.

| model | control | early | late |
|---|---|---|---|
| A3ign | +0.0558 [+0.0449, +0.0677] | -0.0771 [-0.0892, -0.0653] | +0.0379 [+0.0268, +0.0497] |
| A3 | +0.0284 [+0.0163, +0.0405] | -0.0453 [-0.0616, -0.0297] | +0.0273 [+0.0145, +0.0401] |
| A0 | +0.0075 [-0.0020, +0.0169] | -0.0166 [-0.0297, -0.0037] | +0.0112 [+0.0016, +0.0208] |
| pT2 | +0.1484 [+0.1323, +0.1641] | -0.3282 [-0.3642, -0.2931] | -0.0530 [-0.0699, -0.0357] |

## Per-clip tracking of the ruler's shift (r, slope)

* A3ign: control r -0.031, slope -0.026; early r +0.208, slope +0.086; late r +0.232, slope +0.175
* A3: control r +0.040, slope +0.035; early r +0.100, slope +0.050; late r +0.206, slope +0.180
* A0: control r -0.126, slope -0.090; early r -0.072, slope -0.029; late r -0.048, slope -0.031

## early - late = frame set + move

| model | total (crop scores) | frame set (full-clip scores) | move (shift early - late) |
|---|---|---|---|
| A3ign | -0.1994 [-0.2254, -0.1732] | -0.0844 [-0.1098, -0.0589] | -0.1150 [-0.1308, -0.1003] |
| A3 | -0.1901 [-0.2213, -0.1596] | -0.1174 [-0.1536, -0.0816] | -0.0727 [-0.0918, -0.0535] |
| A0 | -0.1068 [-0.1365, -0.0770] | -0.0790 [-0.1142, -0.0446] | -0.0278 [-0.0436, -0.0122] |
| pT2 | -0.4912 [-0.5023, -0.4803] | -0.2160 [-0.2458, -0.1855] | -0.2752 [-0.3072, -0.2445] |
