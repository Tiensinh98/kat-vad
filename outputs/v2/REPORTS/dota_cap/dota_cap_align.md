# DoTA-CAP — alignment (L1-L3, CLIP caches only)

Rule: addendum §11. mean cos ≥ 0.99, every frame ≥ 0.95, irregular steps (|Δj - median| > 1) ≤ 5%.
Matches: `/content/drive/MyDrive/Thesis-V2/outputs/REPORTS/mmau_p0/all/mmau_p0_matches.json` (sha256 `11655304a79f`)

Reasons: cap_shorter 8, mean_cos 56, min_cos 9, ok 1129, out_of_range 46, p0_near 58, p0_none 91
Median CAP step of kept clips (Amendments 4a-4c): 1 → 817, 2 → 2, 3 → 310
Phase shift of kept clips (Amendment 4b, CAP frames): -1 → 6, -2 → 3, 0 → 866, 1 → 232, 2 → 22
Rate > 1, mean cos straight line → band, q50 (Amendment 4c): ok n=312 0.9882 → 0.9937; mean_cos n=44 0.9820 → 0.9878; min_cos n=2 0.9854 → 0.9917
Overshoot of `out_of_range` clips (CAP frames): 1 → 26, 15 → 1, 2 → 12, 3 → 2, 4 → 2, 6 → 2, 7 → 1

| set | kept | of | share |
|---|---|---|---|
| all | 1129 | 1397 | 0.808 |
| dev | 569 | 702 | 0.811 |
| rest_eval | 560 | 695 | 0.806 |
