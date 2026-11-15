# Zero-shot score curves per video (K5 vs LaGoVAD) — 2026-11-15

**Illustration only (s2024, n = 1). Nothing here decides anything; never quote a picture as a result.**

Tool: `core/tools/zs_curves.py` (+ `core/tests/test_zs_curves.py`). It draws the per-frame
scores the zero-shot reads already wrote (nothing is re-scored), from the same sources and
loader as `zs_arm_table` (§25.2 of `KIM_RUN.md`). It raises if two models scored different
videos, different frame labels, or a curve whose length differs from its labels.

## What it draws

| Output | Content |
| --- | --- |
| `grid_<set>.png` | `--n` videos (default 6) spread evenly over the `--rank-by` model's per-video frame AUC, worst → best, so the grid is not a best-case pick. On TAD / DAD the last cell is one all-normal video (false alarms). GT is shaded red, each model's per-video AUC is in the legend. |
| `onset_aligned.png` | Each model's mean score vs seconds from the GT onset (first abnormal row), over every two-class video, ±1 SEM. Each video is averaged per bin first, so long videos do not outweigh short ones. Bin = 2 score rows of the set; bins averaged over < 10 videos are hidden. |
| `zs_curves_<name>.json` | Picks, per-pick AUC, median per-video AUC per model, every onset bin (mean, SEM, videos per bin). |

Pooling follows the tables: **DoTA-CAP-dev and DADA after per-video min-max** (raw drawn faint
beneath, because min-max turns a near-zero raw curve into a full-height one); **TAD and DAD raw**
(min-max dashed, shape only). Time axis = seconds per score row: DoTA 0.1 s (protocol B,
interpolated to native 10 fps), DADA and TAD 0.267 s (s8, 30 fps assumed), DAD 0.28 s (s7 / 25 fps).

## Sources used for K5 vs LaGoVAD (local, 2026-11-15)

| Set | K5 | LaGoVAD (released) |
| --- | --- | --- |
| DoTA-CAP-dev | `outputs/v_final/kim/eval/dota/K5/clip_scores.npz` | `outputs/v_final/zs_lagovad/eval/dota/LaGoVAD/B/clip_scores.npz` (protocol B, as K5) |
| DADA-2000 | `outputs/v_final/kim/eval/dada/K5/clip_scores.npz` | `outputs/v_final/zs_lagovad/eval/dada/LaGoVAD/clip_scores.npz` |
| TAD | `outputs/v_final/kim/eval/tad/K5/s2024` | `outputs/v_final/zs_lagovad/eval/tad/LaGoVAD/released` |
| DAD | `outputs/v_final/kim/eval/dad/K5/s2024` | `outputs/v_final/zs_lagovad/eval/dad/LaGoVAD/released` |

Items and frame labels match on all four sets (569 / 370 / 100 / 466). Output:
`outputs/v_final/kim/REPORTS/zs_curves/` (gitignored).

```bash
python -m core.tools.zs_curves --rank-by K5 --name k5_lagovad_s2024 \
    --out-dir outputs/v_final/kim/REPORTS/zs_curves \
    --read dota K5 <K5 dota npz> --read dota LaGoVAD <LaGoVAD B npz> \
    --read dada K5 ... --read tad K5 ... --read dad K5 ...   # as in the table above
```

## What the pictures show (descriptive, n = 1)

- **K5's mean curve rises 2–3 s before the onset, peaks about 1.5–2 s after it, and falls while
  the GT is still on** (DoTA, DADA, TAD). K5 marks the start of the event, not the whole span.
  The early rise can be anticipation or a position prior (accidents sit mid-clip); this picture
  cannot tell them apart (see `CLAUDE.md` §14.5.1 on position).
- **LaGoVAD's mean curve is flatter**: higher before the onset, a lower peak, and a slower
  decay. On DAD (raw) it barely moves (≈ 0.05 → 0.07).
- Median per-video AUC (two-class videos): K5 / LaGoVAD = DoTA 0.884 / 0.666, DADA 0.865 /
  0.658, TAD 0.790 / 0.835, DAD 0.885 / 0.808. These are per-video medians, not the macro of
  the read-outs; quote `K5_ZS_REPORT.md` for those.
- **DAD's GT is the last 2 rows of a 15-row clip** (median onset at row 13), so its onset curve
  stops at +0.4 s. That is DAD's structure, not a drawing bug.
- Several videos have a raw curve near 0 throughout (e.g. DADA `t56_v021`, `t50_v090`); their
  min-max curve and per-video AUC rest on tiny differences.
