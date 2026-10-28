# v5 Stage 1 summary (PREREG_V5 §2.2)

- **g1_extract_s8**: PASS
- **g1_extract_s1**: PASS
- **g2_init**: PASS
- **g5_maps**: PASS
- **g4_provenance**: PASS

# S1-G1 grid extraction (stride 8)

**PASS** -- 563 / 563 videos stored, failed (length) 0 []

Printed only: min cos vs v2 `Nexar_s1_squash` 0.9999066445222236; videos with a row < 0.9999: 0

# S1-G1 grid extraction (stride 1)

**PASS** -- 225 / 225 videos stored, failed (length) 0 []

Printed only: min cos vs v2 `Nexar_s1_squash` 0.9994574354766979; videos with a row < 0.9999: 44

# S1-G2 same start: **PASS**

- (a) readout vs tile mean max |Δ| {'M1_content': 1.1920928955078125e-07, 'M1_change': 1.1920928955078125e-07, 'M8_content': 1.4901161193847656e-07, 'M8_change': 1.4901161193847656e-07} (≤ 1e-06); DVS splice keeps per-item ΔG: True
- (b) 100 Nexar windows, motion term vs R1's max |Δ| {'R1': 0.0, 'R3': 1.0728836059570312e-06, 'R5': 1.1324882507324219e-06, 'R5d': 1.1324882507324219e-06} (≤ 0.001)

# S1-G5 map tools: **PASS**

- (a) Stage 0 V7 D3 row reproduced: max |Δ| {'hit_minus_chance': 0.0, 'hit_minus_pre': 0.0} (≤ 0.0001)
- (b) random maps point: False (must be False); planted maps point: True (must be True)
- (c) STAUC port vs official: {'tarr': 0.0, 'stauc': 0.0, 'auc': 0.0} (≤ 1e-06); official {'stauc': 0.17144771670844555, 'auc': 0.5304259634888437}

# S1-G4 provenance: **PASS**

| run | ckpt step | metrics step | config Δ vs A2 | v5 mismatch | |
|---|---|---|---|---|---|
| R1/s2024 | 1749 | 1749 | — | — | PASS |
