# v2 D6: temporal-shuffle control on DoTA-CAP-dev (printed, never decided on)

Addendum §2 D6 / §12 N11. Encoder `vit_s_k710_dl_from_giant`, in-domain probe (N3), `dota_cap_dev` 569 clips (568 two-class), stride 3. Every window's 16 frames permuted by seed 2024: `[11, 13, 5, 7, 14, 15, 10, 2, 3, 12, 0, 1, 4, 9, 6, 8]`. Cluster bootstrap over source videos, B = 10000. Name it **DoTA-CAP (n/1397)** (D14).

| Arm | Set | ordered | shuffled | **ordered - shuffled** |
|---|---|---|---|---|
| A2 | u | 0.7922 [0.7794, 0.8054] | 0.7865 [0.7732, 0.7995] | 0.0057 [-0.0057, 0.0173] |
| A2 | with | 0.7814 [0.7678, 0.7948] | 0.7751 [0.7618, 0.7885] | 0.0063 [-0.0014, 0.0140] |
| A2 | with_p | 0.8365 [0.8228, 0.8503] | 0.8369 [0.8249, 0.8490] | -0.0004 [-0.0074, 0.0065] |
| A3 | u | 0.8075 [0.7952, 0.8203] | 0.8006 [0.7879, 0.8129] | 0.0069 [-0.0030, 0.0174] |
| A3 | with | 0.7920 [0.7786, 0.8054] | 0.7851 [0.7710, 0.7987] | 0.0069 [-0.0007, 0.0148] |
| A3 | with_p | 0.8403 [0.8265, 0.8542] | 0.8405 [0.8284, 0.8529] | -0.0003 [-0.0066, 0.0062] |

| Arm | CLIP-only | shuffled `with` - CLIP-only |
|---|---|---|
| A2 | 0.6326 [0.6156, 0.6494] | 0.1424 [0.1264, 0.1587] |
| A3 | 0.6666 [0.6474, 0.6860] | 0.1185 [0.1033, 0.1341] |

Reading (fixed in Amendment 6 before this number): ordered - shuffled ≈ 0 on `with_p` ⇒ the stream's gain is appearance, and the thesis may not call it motion.
