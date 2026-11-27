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
| `onset_aligned.png` | Each model's mean score vs seconds from the GT onset (first abnormal row), over every two-class video, ±1 SEM; dashed grey (right axis) = share of rows still inside the GT, averaged the same way. Each video is averaged per bin first, so long videos do not outweigh short ones. Bin = 2 score rows of the set; bins averaged over < 10 videos are hidden. |
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

## Every video (`--all`, 2026-11-15)

`--all` replaces the grid with one PNG per video, `<out-dir>/videos/<set>/<rank>_<video>.png`:
two-class videos worst -> best by the `--rank-by` model's per-video AUC (rank 1 = worst), then the
all-normal videos by id. `onset_aligned.png` and the json are written as before (the json's
`pick_aucs` then holds every video). `--all` and `--clips` are exclusive.

Wrapper `scripts/plot_zs_curves.py` (every video of one or more sets). K5 is drawn and ranked as
**`LaGoVAD+KIM (trained on <corpus>)`**: `--train nexar` (default) or `--train t2`, which the
figures show as **DADA2000** because T2 is cut from the DADA-2000 original release (`KIM_RUN.md`
§13). T2 has no DADA read (it trains on DADA-2000), so `--train t2 --set dada` is refused.
LaGoVAD = released checkpoint, DoTA under protocol B, the same for both corpora. Default sources:
`constants.ZS_CURVES_MAIN_SOURCES[train][set]` and `ZS_CURVES_LAGOVAD_SOURCES[set]`.
`outputs/v_final/kim/REPORTS/figures/` is gitignored.

```bash
source .venv/bin/activate
python scripts/plot_zs_curves.py --set dota dada tad dad            # trained on Nexar
python scripts/plot_zs_curves.py --train t2 --set dota tad dad      # trained on T2 (DADA2000)
python scripts/plot_zs_curves.py --set tad --tag my_tag --main <run dir> --lagovad <run dir>
# -> outputs/v_final/kim/REPORTS/figures/train_<train>/<set>/<tag>/videos/<set>/<rank>_<video>.png
#    tag defaults to lagovad_kim_vs_lagovad_s2024; --dry-run prints the zs_curves calls only
```

## What the pictures show (descriptive, n = 1)

- **K5's mean curve rises 2–3 s before the onset and peaks about 1.5–2 s after it.** The early
  rise can be anticipation, a late onset label or a position prior (accidents sit mid-clip);
  this picture cannot tell them apart (see `CLAUDE.md` §14.5.1 on position).
- **Corrected 2026-11-15 by the GT-share line** (dashed grey, right axis of `onset_aligned.png`:
  per bin, the share of rows still inside the GT, averaged over videos as the score is). On
  DoTA and DADA it stays ≈ 1 to about +1.5–2 s, then falls (DoTA ≈ 0.5 at +3 s, ≈ 0.2 at +5 s).
  K5's decay follows it, so most of the "fall while the GT is still on" read earlier is the GT
  ending, not K5 missing the span. This is a mean over videos: it does not show per video that
  K5 covers its own GT. LaGoVAD stays ≈ 0.45 at +5 s, where ≈ 80 % of rows are already outside
  the GT.
- **LaGoVAD's mean curve is flatter**: higher before the onset, a lower peak, and a slower
  decay. On DAD (raw) it barely moves (≈ 0.05 → 0.07).
- Median per-video AUC (two-class videos): K5 / LaGoVAD = DoTA 0.884 / 0.666, DADA 0.865 /
  0.658, TAD 0.790 / 0.835, DAD 0.885 / 0.808. These are per-video medians, not the macro of
  the read-outs; quote `K5_ZS_REPORT.md` for those.
- **DAD's GT is the last 2 rows of a 15-row clip** (median onset at row 13), so its onset curve
  stops at +0.4 s. That is DAD's structure, not a drawing bug.
- Several videos have a raw curve near 0 throughout (e.g. DADA `t56_v021`, `t50_v090`); their
  min-max curve and per-video AUC rest on tiny differences.

## Nexar val, K5 trained on T2 (2026-11-15, user; illustration, decides nothing)

The user asked to see K5 trained on T2 (shown as DADA2000) read zero-shot on `nexar_val`, on
the **uncut videos**. §13 of `KIM_RUN.md` already read that row on whole videos (`nexar_eval`
scores each whole video in one pass, `s1[::3]`, interpolated to native frames), so nothing is
re-assembled from crops. §13 kept no per-frame scores, so
`colab/final/k5_t2_nexar_curves.ipynb` scores LaGoVAD and K5-T2 again with `--save-scores` into
`outputs/kim_t2/eval/nexar_curves/<row>/scores/` (a new folder: §13's read-outs are not
overwritten), asserts that crops / whole macro / micro / AP reproduce §13 within 1e-4, and
draws with `zs_curves` (grid, onset, `--all`) into
`outputs/kim_t2/REPORTS/figures/train_t2/nexar/<tag>/`. `nexar_test` is never read.

- Set `nexar` in `zs_curves`: a `nexar_eval --save-scores` run dir, **raw** pooling (as
  `nexar_eval`), one row per native frame at `NEXAR_NOMINAL_FPS` = 30 (measured on
  `nexar_val`: 29.0–30.6, 123 / 225 at 30.0; ≤ ~2 % time error). Onset bins are 8 rows
  (0.27 s, `ZS_CURVES_ONSET_BIN_ROWS_BY_SET`) instead of 2.
- `scripts/plot_zs_curves.py --train t2 --set nexar` re-draws locally once
  `kim_t2/eval/nexar_curves/` is synced under `outputs/v_final/`.
- **In-domain, K5 trained on Nexar (`Nexar_window`, 20 rows at s8, hop 8), drawn 2026-11-15:**
  `python scripts/plot_zs_curves.py --set nexar --main-only` reads the scores already on disk
  (`outputs/v_final/kim_gate/eval/nexar/K5/scores/`, 225 videos, ckpt step 1,749) and titles
  the figures "in-domain" (`ZS_CURVES_IN_DOMAIN_PAIRS`). `--main-only` drops LaGoVAD (its Nexar
  scores are not local until the T2 notebook runs) and defaults the tag to `lagovad_kim_s2024`.
  That K5-only set was drawn once and deleted the same day (superseded by the two-model set
  below, once LaGoVAD's Nexar scores were synced). Median
  per-video AUC 0.967 (112 two-class videos, one below 0.5: `01028`) — in-domain and
  position-confounded, so it says nothing about localization; quote the crop endpoint 0.8981.
- **Position caveat.** On Nexar the event sits mid-video: the middle ruler `-|t/T − 0.5|`
  scores whole-video macro **0.9735**, above K5-T2's 0.8201 and LaGoVAD's 0.5832 (§13). A
  curve peaking in the GT span here does not show localization; the crop endpoint (0.7816) is
  the number to quote. Titles say "zero-shot": true for K5-T2 and LaGoVAD, the only rows drawn.
- **Run 2026-11-15 (commit `3855fb2`), synced and drawn locally.** Both rows reproduce §13
  exactly (max |Δ| 0.0 on crops / whole macro / micro / AP; 225 score files each). Drawn:
  `figures/train_t2/nexar/lagovad_kim_vs_lagovad_s2024/` (K5-T2 vs LaGoVAD) and
  `figures/train_nexar/nexar/lagovad_kim_vs_lagovad_s2024/` (K5 on Nexar vs LaGoVAD; titled
  "in-domain", which is true for K5 only — LaGoVAD is zero-shot on every set). Median per-video
  AUC (112 two-class): K5 (Nexar) 0.967, K5-T2 0.878, LaGoVAD 0.609. Onset picture: K5 (Nexar)
  sits near 0 before the onset and peaks ≈ 0.65 at +1.5–2 s; K5-T2 has the same shape on a
  raised floor (≈ 0.40 → 0.78), i.e. more false alarm level on normal Nexar rows; LaGoVAD is
  flat at ≈ 0.27. Position-confounded on Nexar (middle ruler 0.9735): shape only.

## Training-loss curves (`core.tools.loss_curves`, 2026-11-27)

Reads each run's own `metrics.jsonl` (+ `config.yaml` for the weights); nothing is re-run. One panel per key (`total`, `mil`, `mul_mil`, `dvs_sup`, `dvs_sup_mil`, `kim_mil`, `kim_nkc`, `lr`), runs overlaid on global step: centred rolling mean (25 steps) inside a band of ±1 rolling standard deviation over the same window (minibatch noise; per-step lines dropped 2026-11-27, user). The JSON holds first / last 50-step means, the smoothed minimum, and `total_residual` = max |total − Σ w·term| (≈ 0 means every term of `total` is logged).

    python -m core.tools.loss_curves --name k5dfi_nexar_vs_dota_s2024 \
        --out-dir outputs/v_final/kim/REPORTS/figures/train_nexar_vs_dota/losses/k5dfi_nexar_vs_dota_s2024 \
        --run "K5dfi (Nexar)" outputs/v_final/kim_dvs_ignore/K5dfi/K5/s2024 \
        --run "K5dfiDoTA (DoTA)" outputs/v_final/kim_dota_train/K5dfiDoTA/K5/s2024

**K5dfi Nexar vs DoTA (s2024, n = 1, illustration).** Same recipe, same LR schedule, ~1,750 steps each, but Nexar = 33 epochs × 53 steps, DoTA = 6 epochs × 294 steps (each Nexar window seen ~5.5× as often). `total_residual` 3e-7 on both. Last-50-step means Nexar / DoTA: total 0.79 / 1.63, `mil` 0.20 / 0.42, `mul_mil` 0.16 / 0.35, `kim_mil` 0.19 / 0.40, `dvs_sup_mil` 0.13 / 0.26, `dvs_sup` 0.04 / 0.07, `kim_nkc` 0.79 / 1.19 (starts at step 221 on both). Starts match (total 3.12 / 3.11). The DoTA row ends at about twice the Nexar loss on every term and flattens after ~750 steps; the Nexar row keeps falling. A training loss is not a fit to held-out data: lower Nexar loss is consistent with more passes over a smaller corpus, and says nothing about either row's zero-shot score.

**K5 / K0 / L0 per corpus (2026-11-27, shared terms `mil`, `mul_mil`, `dvs_sup`, `dvs_sup_mil` only — `total` is not comparable across rows: K5's adds `kim_mil` + 0.1·`kim_nkc`).** Figures `train_nexar/losses/k5dfi_k0dfi_l0dfi_s2024/`, `train_dota/losses/k5dfidota_k0dfidota_l0dfidota_s2024/`. Last-50-step `mil` / `mul_mil`: Nexar K5dfi 0.200 / 0.155, K0dfi 0.188 / 0.150, L0dfi 0.396 / 0.317; DoTA K5dfiDoTA 0.419 / 0.353, K0dfiDoTA 0.404 / 0.351, L0dfiDoTA 0.540 / 0.476. K5 and K0 overlap inside their bands on every term on both corpora (consistent with §53.1 / §58.1: KIM moves no trunk loss); the CLIP-only L0 ends clearly higher on every term, most on Nexar (`mil` ~2×), and its `mul_mil` leaves 0.69 later (~step 250 vs ~120). Training loss only, n = 1: "L0 fits the training windows less well" is licensed, "L0's lower zero-shot score is caused by underfitting" is not.
