# DADA-2000 whole videos — 370 sources (370 abnormal, 370 two-class)

whole source video, every s8 row, labels = annotation span. Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant', 'stp_sa2': True, 'stp_top_k': 12, 'stp_motion': 'raw', 'stp_text_prompt': 'released'}`, input `/content/zs/bake/dada`, steps {'s2024': 1749}.

* **macro (seed-averaged, 95 % bootstrap over videos): 0.7579 [0.7334, 0.7820]**
* micro: 0.7048 · clip-level AUC: —
* rulers (no pixels): `t/N` 0.5514 [0.5247, 0.5782] · middle 0.7342 [0.7076, 0.7594]
* per-run macro: {'s2024': 0.7579}
