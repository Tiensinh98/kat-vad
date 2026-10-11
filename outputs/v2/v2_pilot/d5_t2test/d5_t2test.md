# v2 pilot — D5 on T2-test (Amendment 7)

| checkpoint | T2-test micro | macro | phase-4 printed | T2-val micro | T2-val − T2-test |
|---|---|---|---|---|---|
| A0_s2099 | 0.6320 | 0.6389 | — | 0.6586 | +0.0266 |
| p4_s2024 | 0.6230 | 0.6334 | 0.6230 / 0.6334 | 0.7714 | +0.1484 |
| p4_s2025 | 0.6182 | 0.6347 | 0.6164 / 0.6352 | 0.7743 | +0.1561 |
| p4_s2026 | 0.6153 | 0.6078 | 0.6152 / 0.6057 | 0.7567 | +0.1414 |

**H1** (s2024 reproduces phase 4, tol 0.001): Δ micro +0.00000, Δ macro +0.00000 → PASS
**H2** (A0 vs ref mean on T2-test, margin 0.02): ref mean 0.6188 / 0.6253; Δ micro +0.0132, Δ macro +0.0136 → PASS
**H3** D5: FAIL on T2-val, EXPLAINED -- in-sample bias of the references. D5 no longer blocks P7.

H4 (printed): s2025/s2026 phase-4 numbers belong to the lost snapshots; the last column is the in-sample
inflation (references) vs A0 (T2-val unseen).
