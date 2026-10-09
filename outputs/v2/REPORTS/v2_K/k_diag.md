# Kill-switch K — K-pos diagnostic (printed, not gated)

Pre-registration: `core/docs/v2/PREREG_ADDENDUM.md` §6.2. K's verdict is unchanged.
Code: n/a

## (P) Pad drop — first 6 steps removed (295 two-class sources, 0 lost)

| probe | x | u | [x;u] | Δ([x;u] - x) |
|---|---|---|---|---|
| source-grouped CV (in-domain) | +0.6460 [+0.6209, +0.6705] | +0.7584 [+0.7354, +0.7818] | +0.7520 [+0.7292, +0.7750] | +0.1060 [+0.0802, +0.1303] |
| type-grouped CV (held-out accident types; NOT a domain transfer) | +0.6453 [+0.6201, +0.6701] | +0.7634 [+0.7403, +0.7871] | +0.7452 [+0.7219, +0.7690] | +0.0999 [+0.0744, +0.1258] |

Reading: **NOT_PAD**

## (Q) Beyond position — p = tau^1..tau^3, tau=(t+0.5)/L (295 two-class sources)

| probe | p | [x;p] | [x;u;p] | Δ([x;u;p] - [x;p]) |
|---|---|---|---|---|
| source-grouped CV (in-domain) | +0.7301 [+0.7015, +0.7582] | +0.7337 [+0.7113, +0.7552] | +0.7765 [+0.7567, +0.7966] | +0.0428 [+0.0230, +0.0640] |
| type-grouped CV (held-out accident types; NOT a domain transfer) | +0.7300 [+0.7007, +0.7587] | +0.7321 [+0.7083, +0.7544] | +0.7714 [+0.7507, +0.7920] | +0.0393 [+0.0176, +0.0608] |

Reading: **BEYOND_POSITION**

Null reading iff both Δ upper < +0.03 (K's bar); B = 2000.
