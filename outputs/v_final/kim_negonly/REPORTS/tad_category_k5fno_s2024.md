# TAD by category (`KIM_RUN.md` §38.2 N4) — printed, decides nothing

Macro = mean frame AUC over the category's two-class videos. Clip AUC = peak score of the category's videos vs every Normal video's peak. Micro = the category's frames pooled with every Normal video's frames (raw). n = 1 seed; per-category counts are small.

| category | videos | K5f macro / clip AUC / micro | K5fno macro / clip AUC / micro |
| --- | --- | --- | --- |
| 01_Accident | 23 (23 two-class) | 0.8230 / 0.5902 / 0.6711 | 0.7881 / 0.4935 / 0.5659 |
| 02_IllegalTurn | 3 (3 two-class) | 0.8544 / 0.6750 / 0.8171 | 0.8409 / 0.4500 / 0.4898 |
| 03_IllegalOccupation | 3 (3 two-class) | 0.6425 / 0.6833 / 0.8181 | 0.5668 / 0.5750 / 0.7210 |
| 04_Retrograde | 3 (3 two-class) | 0.7883 / 0.7917 / 0.6986 | 0.6886 / 0.3667 / 0.2632 |
| 05_else | 14 (14 two-class) | 0.6626 / 0.6214 / 0.6324 | 0.7316 / 0.3357 / 0.4135 |
| 06_PedestrianOnRoad | 10 (10 two-class) | 0.4437 / 0.6150 / 0.6256 | 0.4929 / 0.3500 / 0.3906 |
| 07_RoadSpills | 4 (4 two-class) | 0.5974 / 0.6562 / 0.7465 | 0.5303 / 0.5062 / 0.5629 |
| all abnormal | 60 (60 two-class) | 0.6981 / 0.6250 / 0.6711 | 0.6951 / 0.4292 / 0.5008 |
