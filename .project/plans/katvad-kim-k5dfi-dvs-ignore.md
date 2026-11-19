# Plan — K5dfi / K5dfie: K5df with `loss.dvs_anchor_mode=ignore`, and with `is_egocentric=true` (2026-11-19, CONFIRMED, BUILT, NOT RUN)

Branch `v_final_las_vad`. Rule text goes to `core/docs/final/KIM_RUN.md` §34 only after the user confirms the choices in §6.
Reopening a closed line needs a dated note from the user; the request of 2026-11-19 ("lập kế hoạch arm K5f với dvs ignore") is that note once confirmed.

## 0. Finding that changes the request

**"K5f + ignore" is a no-op.** K5f has DVS off (`dvs.theta = dvs.theta_ego = 1.0`).

- `core/data/dataset.py:183`: `if self._rng.random() < self.theta ...` is always true at theta 1.0, so no clip is ever spliced.
- `core/data/synthesis.py:80-84`: `pseudo` is all zeros unless `anchor_is_abnormal and fillers`, so with no fillers no frame is pseudo-positive.
- `core/losses/dvs.py:43-44`: `ignore_positive` masks only `frame_labels >= 0.5`. With none present, the mask and the loss are identical to `span`.
- `core/train.py` still builds `dvs_rows` from normal rows, so `dvs_sup` is logged (K0 = K5 = 0.7182), but its targets are all 0.

So the mode only acts when DVS is on. The arm that tests the question is **K5df + ignore**, named **K5dfi**:
K5df = K5d retrained on the §31 fixed labels (§32), DVS θ 0.7 / 0.85, `kim.seam_mask=true`. Its saved config has `dvs_anchor_mode: span`, as do all 13 saved arm configs.

Cheap proof to ship with the build (not a gate): unit test that, for a DVS-off batch, `supervised_loss(..., ignore_positive=True) == supervised_loss(..., ignore_positive=False)`.

## 1. Why this is worth one run (and the honest prior)

- Lesson (bj) / C29: `span` marks the whole spliced anchor positive. It is right only if the anchor is ~all anomaly. v2: `ignore` was worth +0.040 DoTA-CAP-dev macro (A3 → A3-ign), on a v2 T2-trained host.
- Counter-evidence: DADA archive Phase 1 (2026-09-08) `ignore` gave macro −0.0056 (§ RESULTS_DADA_PHASE1, "FALSIFIED"). Different corpus and host.
- KIM-side prior: DVS on top of KIM showed nothing (K5d − K5 DADA −0.0027, §12.1), so a DVS-label fix has little to act on. **Expectation (written before any number): K5dfi − K5df small, mostly "not separated"; a sign on DoTA-CAP-dev or DADA would not surprise.**
- Unmeasured assumption to print first (P0): what share of an anchor window in `Nexar_window` is truly anomalous. If it is close to 1.0, `span` is correct and `ignore` should do little. Printed, not gating.

## 2. Options considered

| Option | Verdict |
|---|---|
| A. K5f + ignore | Rejected: bit-identical loss to K5f (§0). Would waste a Colab run. |
| **B. K5dfi vs K5df (`span`), K5f printed** | **Chosen.** One flag differs; K5df s2024 already trained and read (`outputs/v_final/kim_lneg_cap_fixed/K5df`). |
| C. B + retrain K5df on the same runtime | Optional. §12.1 flag 1: run-to-run spread of the host is unmeasured (step-1 `mil` differed 0.6906 vs 0.6944 between runtimes). Costs a second run; only if the start check fails. |
| D. Ignore on the W0 host / without KIM (K0d-ign) | Out of scope: different question (KIM line is closed). |
| E. Add captions or `L_neg` | Out of scope: caption line closed 2026-11-18. |

## 3. Arm and guards

| item | K5dfi |
|---|---|
| Row | K5df's exact command (§32 R1: `kim.enabled`, `kim.prototypes`, `kim.edge_mask`, `kim.seam_mask`, `dvs.theta` 0.7, `dvs.theta_ego` 0.85, `Nexar_window`, A2 V2-S bake, s2024, 33 epochs, 1,749 steps) **plus `loss.dvs_anchor_mode=ignore`**. |
| G1 config diff | to K5df = `loss.dvs_anchor_mode` only (asserted; raises otherwise). |
| G2 J10 | ckpt `global_step` = last `metrics.jsonl` step = 1,749. |
| G3 start check | step-1 `mil` equal to K5df's within `KIM_START_TOL` (`ignore` changes only `dvs_sup`, not `mil`); step-1 `dvs_sup` **differs** (mask active). Printed. A `mil` mismatch flags the contrast (same-runtime retrain = option C), it does not stop the run. |
| G4 fix guard | last-epoch `mul_mil` > `KIM_FIXED_MUL_MIL_FLOOR` (0.01); never run on pre-§31 code. |
| G5 mask is live | `dvs_sup` is computed on a strictly smaller mask than K5df's: log the masked-positive share per epoch (expected ≈ share of spliced anchors × anchor length). If 0, the run is invalid. |

## 4. Reads and report

- Sets: Nexar val (crops, `--save-scores`, never `--final`), DoTA-CAP-dev (protocol B), DADA whole, TAD, DAD. **Macro and micro on all four zero-shot sets** (`zs_arm_table`), min-max on DoTA/DADA, raw on TAD/DAD.
- Contrasts, paired per item, cluster bootstrap: **K5dfi − K5df (primary)**, K5dfi − K5f, K5df − K5f (recomputed for reference).
- Post-hoc, printed, never decides: min-max span reads (`core.metrics.span_reads`, lesson bh: `ignore` lowers raw positive scores, so raw "pre-anomaly mean" would fall for free); Nexar val false alarms at cut 0.5 (not a ranking claim, §17.1).
- Curves: `core.tools.zs_curves` into `kim/REPORTS/figures/train_<train>/<set>/<tag>`.
- To check at build (not yet read): whether `zs_arm_table` / `kim_readout` accept K5df as the control, or whether `k5_lneg_cap_report` needs a `--control` option. If a symbol is edited, `trace_call_path` first and report blast radius.

## 5. Decision rule (proposed numbers; user to confirm before any number is read)

Margin = `V5_MIN_GAIN` = 0.01, the bar every read-out here uses; it is not lowered after the number (lesson 14).

- **GAIN**: K5dfi − K5df ≥ +0.01 with the interval above 0 on **DoTA-CAP-dev or DADA** (the DVS-sensitive sets), and below −0.01 on none of Nexar val / DoTA-CAP-dev / DADA / DAD.
- **COST**: ≤ −0.01 with the interval below 0 on any of those four.
- **NO DIFFERENCE**: anything else. TAD is printed, never decides.
- **Candidate** (C7 style): K5dfi − K5f above on ≥ 1 of the four and below on none. K5f stays reference + main model unless the user decides otherwise, whatever the sign.
- n = 1 adopts nothing. s2025 only if GAIN, and only on the user's word.
- Wording guard: if K5dfi is above K5df but still below K5f, say "`ignore` reduces DVS's cost", not "improves the model".

## 6. Choices for the user to confirm

1. Arm is **K5dfi = K5df + ignore** (not K5f).
2. Control is the existing K5df s2024 run, no same-runtime retrain unless G3 fails.
3. Margin 0.01 and the GAIN/COST sets in §5.
4. s2024 only, n = 1.

## 7. Work after confirmation

1. `KIM_RUN.md` §34 dated note (this rule, expectation, user's choices) before any number.
2. Test: DVS-off `ignore` == `span`; G5 masked-share logging if not already in `metrics.jsonl`.
3. Notebook `colab/final/k5_dvs_ignore.ipynb`: clone cells of `k5_lneg_cap_fixed.ipynb` (train one row with `K5D_SETS` + `loss.dvs_anchor_mode=ignore`, zero-shot reads, nexar_val reads, report). Related tests only, never the full suite.
4. Update `activeContext.md`, `CLAUDE.md` §14.0.1 line, `meta-index` untouched; lesson candidate to `pending.md`: "a loss flag is inert on a host that never creates its input (DVS-off × anchor mode)".
5. No commit unless the user asks.

Cost: one training run (1,749 steps, ≈ 2 s/step max, watch the log) plus reads, same Colab session shape as §32.

Runbook output dir: `Thesis-Final/outputs/kim_dvs_ignore/`.

## 8. Status 2026-11-19 (after the user confirmed §6 items 1-4 and added an arm)

- **Added arm K5dfie** = K5dfi + `data.is_egocentric=true`. Read from `dataset.py:107-111`: θ 0.7 → θ_ego 0.85 and `delta_m` 5 → `delta_m_ego` 2 (one filler, one seam), so it is lighter DVS + ignore; no matched start. Secondary contrast K5dfie − K5dfi, same R4 rule. `ignore` under `span` + ego (K5dfe) is not run (the 2 x 2 is incomplete).
- **G5 changed:** no masked-share log exists (adding one would edit `train.py`); the live-mask guard is K5dfi's step-1 `dvs_sup` != K5df's (report raises). P0 (anchor share) dropped: `windows.json` has no frame labels.
- **Built:** constants `KIM_DVS_IGNORE_*`; `core/tools/k5_dvs_ignore_report.py`; `core/tests/test_k5_dvs_ignore_report.py` (21 pass); `colab/final/k5_dvs_ignore.ipynb`; `KIM_RUN.md` §34. No existing symbol edited, so no `trace_call_path` was triggered (constants added only).
- **Not yet:** run (user, Colab); the report has not been rendered on real reads.
