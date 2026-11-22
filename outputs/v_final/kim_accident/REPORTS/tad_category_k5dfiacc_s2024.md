# TAD by category (`KIM_RUN.md` §38.2 N4) — printed, decides nothing

Macro = mean frame AUC over the category's two-class videos. Clip AUC = peak score of the category's videos vs every Normal video's peak. Micro = the category's frames pooled with every Normal video's frames (raw). n = 1 seed; per-category counts are small.

| category | videos | K5dfi macro / clip AUC / micro | K5dfiACC macro / clip AUC / micro | L0dfi macro / clip AUC / micro | L0dfiACC macro / clip AUC / micro |
| --- | --- | --- | --- | --- | --- |
| 01_Accident | 23 (23 two-class) | 0.8383 / 0.5761 / 0.6166 | 0.7350 / 0.7272 / 0.7329 | 0.6399 / 0.6065 / 0.6857 | 0.7022 / 0.7152 / 0.7964 |
| 02_IllegalTurn | 3 (3 two-class) | 0.7903 / 0.6083 / 0.6901 | 0.7935 / 0.8833 / 0.7768 | 0.6913 / 0.4167 / 0.5553 | 0.7142 / 0.5750 / 0.6837 |
| 03_IllegalOccupation | 3 (3 two-class) | 0.6529 / 0.7333 / 0.8242 | 0.4735 / 0.3750 / 0.3568 | 0.5666 / 0.9167 / 0.9661 | 0.5047 / 0.5750 / 0.6577 |
| 04_Retrograde | 3 (3 two-class) | 0.7689 / 0.8167 / 0.6566 | 0.4914 / 0.4417 / 0.2760 | 0.7547 / 0.9333 / 0.8668 | 0.6896 / 0.7333 / 0.6908 |
| 05_else | 14 (14 two-class) | 0.7194 / 0.5696 / 0.5882 | 0.7709 / 0.5750 / 0.6004 | 0.4429 / 0.5661 / 0.6238 | 0.7127 / 0.6393 / 0.7562 |
| 06_PedestrianOnRoad | 10 (10 two-class) | 0.4746 / 0.5950 / 0.5012 | 0.5361 / 0.5950 / 0.5183 | 0.5795 / 0.7225 / 0.6107 | 0.6533 / 0.6725 / 0.7115 |
| 07_RoadSpills | 4 (4 two-class) | 0.6137 / 0.6687 / 0.6624 | 0.8973 / 0.8188 / 0.8887 | 0.6657 / 0.6312 / 0.7659 | 0.7588 / 0.8500 / 0.9140 |
| all abnormal | 60 (60 two-class) | 0.7198 / 0.6054 / 0.6119 | 0.6987 / 0.6517 / 0.6170 | 0.5902 / 0.6404 / 0.6733 | 0.6904 / 0.6863 / 0.7389 |
