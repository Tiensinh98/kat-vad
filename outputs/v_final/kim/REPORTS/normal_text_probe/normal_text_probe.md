# Normal-text probe (K5, inference only) — core/docs/final/KIM_RUN.md §15

## Step 1: H_bin's blend

Init: alpha 0.25, pre-fusion w 0.9241. Minor-text bar: w ≥ 0.9.

| run | step | alpha | pre-fusion w | text branch 1 - w |
|---|---|---|---|---|
| s2024 | 1749 | +0.2418 | 0.9182 | 0.0818 |
| s2025 | 1749 | +0.2526 | 0.9260 | 0.0740 |

text branch minor at every seed: **True**

## Step 2: dashcam - original Normal text (paired per clip, 95 % CI)

| set | seed | recorded | original (gate) | dashcam | Δ |
|---|---|---|---|---|---|
| dota | s2024 | 0.8195 | 0.8195 | 0.8197 | +0.0002 [-0.0001, +0.0005] |
| dota | s2025 | 0.8166 | 0.8166 | 0.8166 | -0.0000 [-0.0000, +0.0000] |
| dota | pooled | | | | +0.0001 [-0.0001, +0.0003] |
| dada | s2024 | 0.7740 | 0.7740 | 0.7737 | -0.0003 [-0.0009, +0.0003] |
| dada | s2025 | 0.7631 | 0.7631 | 0.7632 | +0.0001 [+0.0000, +0.0003] |
| dada | pooled | | | | -0.0001 [-0.0004, +0.0002] |

**Verdict: null** (margin 0.01) — the Normal wording does not matter at inference; nothing is trained.

Dashcam Normal sentences: `Normal driving` / `Normal dashcam footage of a vehicle driving on the road.` / `Normal driving video recorded by an in-car camera.` / `Normal driving recorded by a dashcam that does not involve any collision or unusual event.`
