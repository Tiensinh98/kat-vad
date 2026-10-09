# v2 E2(d): encoder choice on DoTA-CAP-dev

Addendum §12. `dota_cap_dev` 569 clips (568 two-class), protocol B (stride 3); transfer fitted on 295 K sources (sha1 `634a105d7175`). Cluster bootstrap over source videos, B = 10000. Name it **DoTA-CAP (n/1397)** (D14).

| Encoder | Arm | Probe | CLIP-only | with u | **Δ** (decides) | Δ beyond position (printed) |
|---|---|---|---|---|---|---|
| vit_b_k710_dl_from_giant | A2 | in_domain | 0.6325 [0.6155, 0.6492] | 0.7946 [0.7808, 0.8085] | 0.1621 [0.1444, 0.1802] | -0.0033 [-0.0145, 0.0079] |
| vit_b_k710_dl_from_giant | A2 | transfer | 0.5690 [0.5512, 0.5867] | 0.7626 [0.7488, 0.7766] | 0.1936 [0.1767, 0.2112] | 0.0365 [0.0230, 0.0501] |
| vit_b_k710_dl_from_giant | A3 | in_domain | 0.6666 [0.6474, 0.6860] | 0.8010 [0.7868, 0.8150] | 0.1343 [0.1161, 0.1526] | -0.0059 [-0.0157, 0.0042] |
| vit_b_k710_dl_from_giant | A3 | transfer | 0.6219 [0.6051, 0.6384] | 0.7727 [0.7601, 0.7852] | 0.1508 [0.1366, 0.1653] | 0.0294 [0.0178, 0.0415] |
| vit_s_k710_dl_from_giant | A2 | in_domain | 0.6325 [0.6155, 0.6492] | 0.7814 [0.7679, 0.7948] | 0.1489 [0.1336, 0.1647] | 0.0043 [-0.0048, 0.0136] |
| vit_s_k710_dl_from_giant | A2 | transfer | 0.5690 [0.5512, 0.5867] | 0.7271 [0.7113, 0.7428] | 0.1581 [0.1415, 0.1750] | 0.0176 [0.0053, 0.0300] |
| vit_s_k710_dl_from_giant | A3 | in_domain | 0.6666 [0.6474, 0.6860] | 0.7920 [0.7786, 0.8054] | 0.1253 [0.1095, 0.1416] | 0.0058 [-0.0018, 0.0135] |
| vit_s_k710_dl_from_giant | A3 | transfer | 0.6219 [0.6051, 0.6384] | 0.7544 [0.7405, 0.7673] | 0.1325 [0.1180, 0.1465] | 0.0259 [0.0152, 0.0367] |

Rule (N6/N7): eligible iff in-domain Δ ≥ 0.1 or transfer Δ ≥ 0.03 (point estimate) for A2 or A3; pick = largest A3 transfer Δ, within 0.02 the cheaper encoder.

Eligible arms per encoder: {'vit_b_k710_dl_from_giant': ['A2', 'A3'], 'vit_s_k710_dl_from_giant': ['A2', 'A3']}

**Decision: vit_s_k710_dl_from_giant** (tie → cheaper encoder)

## Printed, never decided on

`u` alone and position `p` alone (macro):

| Encoder | Arm | Probe | u | p (position only) |
|---|---|---|---|---|
| vit_b_k710_dl_from_giant | A2 | in_domain | 0.8070 [0.7938, 0.8204] | 0.8522 [0.8369, 0.8672] |
| vit_b_k710_dl_from_giant | A2 | transfer | 0.7932 [0.7813, 0.8053] | 0.8471 [0.8310, 0.8630] |
| vit_b_k710_dl_from_giant | A3 | in_domain | 0.8193 [0.8061, 0.8323] | 0.8522 [0.8369, 0.8672] |
| vit_b_k710_dl_from_giant | A3 | transfer | 0.7949 [0.7826, 0.8070] | 0.8471 [0.8310, 0.8630] |
| vit_s_k710_dl_from_giant | A2 | in_domain | 0.7922 [0.7794, 0.8054] | 0.8522 [0.8369, 0.8672] |
| vit_s_k710_dl_from_giant | A2 | transfer | 0.7628 [0.7485, 0.7770] | 0.8471 [0.8310, 0.8630] |
| vit_s_k710_dl_from_giant | A3 | in_domain | 0.8075 [0.7952, 0.8203] | 0.8522 [0.8369, 0.8672] |
| vit_s_k710_dl_from_giant | A3 | transfer | 0.7798 [0.7663, 0.7933] | 0.8471 [0.8310, 0.8630] |

Δ per share bin:

| Encoder | Arm | Probe | <30 | 30-50 | 50-70 | >70 |
|---|---|---|---|---|---|---|
| vit_b_k710_dl_from_giant | A2 | in_domain | 0.1419 [0.1128, 0.1718] | 0.1771 [0.1469, 0.2065] | 0.1756 [0.1240, 0.2288] | 0.2736 [0.1585, 0.4060] |
| vit_b_k710_dl_from_giant | A2 | transfer | 0.1892 [0.1656, 0.2133] | 0.2000 [0.1706, 0.2296] | 0.1900 [0.1331, 0.2453] | 0.2048 [0.0747, 0.3483] |
| vit_b_k710_dl_from_giant | A3 | in_domain | 0.1175 [0.0906, 0.1453] | 0.1384 [0.1102, 0.1682] | 0.1717 [0.1169, 0.2289] | 0.2209 [0.1103, 0.3501] |
| vit_b_k710_dl_from_giant | A3 | transfer | 0.1451 [0.1238, 0.1661] | 0.1525 [0.1273, 0.1773] | 0.1646 [0.1160, 0.2122] | 0.1715 [0.0430, 0.3076] |
| vit_s_k710_dl_from_giant | A2 | in_domain | 0.1280 [0.1055, 0.1512] | 0.1650 [0.1365, 0.1939] | 0.1701 [0.1262, 0.2136] | 0.2242 [0.1014, 0.3716] |
| vit_s_k710_dl_from_giant | A2 | transfer | 0.1553 [0.1324, 0.1773] | 0.1598 [0.1290, 0.1915] | 0.1561 [0.1131, 0.2000] | 0.1963 [0.0962, 0.3211] |
| vit_s_k710_dl_from_giant | A3 | in_domain | 0.1045 [0.0834, 0.1267] | 0.1384 [0.1092, 0.1694] | 0.1606 [0.1177, 0.2018] | 0.1722 [0.0583, 0.3085] |
| vit_s_k710_dl_from_giant | A3 | transfer | 0.1248 [0.1077, 0.1420] | 0.1401 [0.1129, 0.1671] | 0.1363 [0.0950, 0.1773] | 0.1556 [0.0520, 0.2675] |

T2 side (K sources, source-grouped CV):

| Encoder | x | [x;u] | Δ |
|---|---|---|---|
| vit_b_k710_dl_from_giant | 0.6250 [0.6017, 0.6482] | 0.7619 [0.7407, 0.7821] | 0.1369 [0.1139, 0.1601] |
| vit_s_k710_dl_from_giant | 0.6250 [0.6017, 0.6482] | 0.7321 [0.7106, 0.7528] | 0.1071 [0.0864, 0.1272] |

D15 representativeness (all `dota_dev` vs its DoTA-CAP part vs dropped):

| Read | dev | cap | dropped |
|---|---|---|---|
| CLIP-only probe `x` | 0.6383 [0.6206, 0.6560] | 0.6359 [0.6165, 0.6550] | 0.6485 [0.6167, 0.6828] |
| CLIP-only probe `x_crn` | 0.6773 [0.6596, 0.6958] | 0.6730 [0.6538, 0.6926] | 0.6958 [0.6622, 0.7314] |
| A0 (E1 ckpts, protocol B) | 0.6628 [0.6436, 0.6823] | 0.6591 [0.6368, 0.6807] | 0.6787 [0.6340, 0.7245] |

| Side | clips | native frames q50 | share q50 | top categories |
|---|---|---|---|---|
| cap | 569 | 99 | 0.303 | ego: turning 115, other: turning 88, ego: lateral 68, ego: oncoming 48 |
| dropped | 133 | 105 | 0.318 | ego: turning 23, other: leave_to_right 13, ego: moving_ahead_or_waiting 12, ego: lateral 10 |
