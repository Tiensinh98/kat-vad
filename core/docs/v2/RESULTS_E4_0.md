# KAT-VAD v2 — E4-0 results (A3-ign vs A3, 5 seeds)

Pre-registration: `PREREG_ADDENDUM.md` **§22** (E1–E4: rationale, endpoints and guardrails, predictions, base rule,
claims, operational definitions), first committed in `b2b6512` / `2395e39` before any A3-ign number. A **post-Final v2
step on DoTA-CAP-dev**: it does not re-decide §19 (the Final read A3); any sealed-set number for A3-ign is a second look.
Raw outputs: `outputs/v2/v2_e4_0/` (`REPORTS/e4_0_readout.{md,json}`, `REPORTS/pb/A3ign/dota_cap_dev/`,
`A3ign/s{2024..2028}/{config.yaml,metrics.jsonl,diag/}`, `A3/s*/diag/`). Control = E3's A3 runs
(`outputs/v2/v2_e3/`). Runbook `colab/v2/e4_0_a3ign.ipynb`. Read-out tool `core/tools/e4_0_readout.py`.
Colab read-outs carry `commit: UNKNOWN` by design.

Arms: **A3** = CRN (R2) + V2-S, `loss.dvs_anchor_mode = span` (LaGoVAD default, C29) · **A3-ign** = A3 with
`dvs_anchor_mode = ignore`: the anchor steps of a DVS-spliced abnormal sequence leave the dense `L_dvs-sup` BCE;
fillers stay 0; `L_dvs-mil` and `L_MIL` supply the positives.

**Control integrity.** `config.yaml` of A3-ign differs from E3's A3 of the same seed in exactly one line
(`dvs_anchor_mode`; checked locally on s2026–s2028, by the notebook's preflight on all five). Every A3-ign run ends at
`global_step` 1740 in `metrics.jsonl`. A3 re-read by this harness = **0.7576**, E3's 0.758 → the harness reproduces.

## 1. Decision

**`B` = A3-ign.** All three §22 E3 checks pass. A3-ign is v2's adopted training recipe from here (same deployed
architecture as A3, one training label changed) and the new best adoptable free arm `F`: every later costly arm must
beat it.

| check (§22 E3) | value | result |
|---|---|---|
| 1. paired t95 of the DoTA-CAP-dev macro Δ above 0 | **+0.0398 [+0.0347, +0.0450]** (n = 5, t = 2.776) | PASS |
| 2. `f_2` Δ point ≥ 0 | +0.0102 | PASS |
| 3. guardrails vs A3 (T2-val) | below | PASS |

Per seed (2024…2028): +0.0427 / +0.0420 / +0.0428 / +0.0332 / +0.0384, sd 0.0041. Per-seed macro A3 0.7393 / 0.7377 /
0.7419 / 0.7485 / 0.7390 → A3-ign 0.7819 / 0.7797 / 0.7847 / 0.7816 / 0.7773.

**Two macro definitions — never mix them.** The headline macro (A3 **0.7576**, A3-ign **0.7912** [0.7740, 0.8077]) is
the per-clip AUC of the **seed-averaged** score, as in E3 and the Final. The per-seed means are 0.7413 / 0.7811. The
decision interval is on per-seed macros.

**Licensed sentences (§22 E3):**
- "Dropping the whole-anchor DVS label improves DoTA-CAP (n/1397) detection" (dev; any eval number is a **second look
  at an opened set**).
- "… beyond the T2 position prior": `f_2` Δ **+0.0102 [+0.0048, +0.0154]**, interval excludes 0 (`f_2` macro A3
  0.8424 → A3-ign 0.8526).
- **Not** "the whole-anchor label causes Gap S": one of the four mechanism conditions (§22 E2) fails (§3).

## 2. Guardrails (T2-val, seed means, re-diagnosed with the same code)

| read | A3 | A3-ign | bar | pass |
|---|---|---|---|---|
| macro | 0.7034 | **0.7338** | ≥ A3 − 0.01 | ✓ |
| normal-window peak (mean top-k `σ(y^bin)`, all-normal windows) | 0.6024 | 0.6151 (+0.0128) | ≤ A3 + 0.02 | ✓ |
| O1′ on every seed (vs E3's A0) | — | 5/5 | all | ✓ |

The peak uses 64 % of its margin. §4.1 prices it.

## 3. Mechanism (DoTA-CAP-dev, seed-averaged, per-clip min-max; never moves `B`)

| read | A3 | A3-ign | Δ [95 % cluster CI, 91 sources] | predicted | held |
|---|---|---|---|---|---|
| pre-span mean | 0.527 | 0.239 | **−0.288 [−0.301, −0.274]** | falls | ✓ |
| inside-span mean | 0.864 | 0.708 | **−0.155 [−0.168, −0.143]** | does not fall | **✗** |
| argmax-in-span share | 0.613 | 0.664 | +0.051 [+0.020, +0.084] | rises | ✓ |
| tail-clip macro (277 clips) | | | +0.032 [+0.020, +0.043] | does not fall | ✓ |

T2-val replicate (printed, per seed): pre 0.52 → 0.28–0.32, inside 0.76–0.77 → 0.63–0.68, argmax 0.65–0.69 →
0.69–0.71 — the same pattern in-domain, so it is not a DoTA artefact.

**Reading.** `span` trains the whole anchor to 1, so A3's curve is a broad plateau that leaks before the onset.
`ignore` leaves only top-k positive pressure inside the anchor, so the curve becomes a **taller, narrower peak on a
lower floor**: the floor before and after the span drops (pre −0.29, tail +0.03 — the opposite of the killed H2(a)
hold), the peak lands in the span more often, but the rest of the span falls relative to the peak. The gain is
**floor suppression, not span coverage** — which is why the Gap S sentence is not licensed.

Training losses (final epoch, mean over seeds): `dvs_sup` 0.093 → 0.017 is mechanical (fewer BCE terms), not a fit
signal. `mil` 0.014 → 0.029.

## 4. Exploratory reads (2026-10-11, printed, never gated)

Computed locally from the E3 / E4-0 score files and `diag.json` by a scratch script (not a tracked tool). Method check:
the onset-locked profile reproduces `REPORT_V2_RESULTS.md` §6.3's A3 row exactly (−1.02 … +0.64).

### 4.1 Is the normal-window peak a false-alarm cost?

| read | A3 | A3-ign |
|---|---|---|
| T2-val window-level AUC (seed mean) | 0.7566 | 0.7657 (+0.009; per seed +0.025 / +0.014 / +0.005 / +0.002 / −0.001) |
| T2-val micro (oracle 0.6986) | 0.7124 | 0.7427 |
| `y^bin` source-shortcut AUC (T2-val) | 0.541 | 0.509 |
| DoTA-CAP-dev pooled micro | 0.6944 | **0.7548** |
| DoTA-CAP-dev pooled AP | 0.4896 | **0.5697** |

Abnormal windows rose with the normal ones: window-level discrimination did not fall (4/5 seeds up, one −0.001).
The +0.013 peak is a level shift, not a lost ranking. Not a licence to claim "no cost" on unseen normal video —
no all-normal corpus is scored here.

### 4.2 Gap S re-measured on `B` (onset- and end-locked within-clip z-score, native frames, 10 fps)

| offset (frames) | −40 | −20 | −10 | 0 | +10 | +20 | +30 | +40 |
|---|---|---|---|---|---|---|---|---|
| A3 around start | −1.02 | −0.42 | −0.03 | +0.30 | +0.58 | +0.64 | +0.42 | +0.17 |
| **A3-ign around start** | −0.72 | −0.47 | **−0.28** | +0.20 | +0.75 | **+0.87** | +0.44 | **−0.04** |
| A3 around end | −0.06 | +0.49 | +0.54 | +0.43 | +0.24 | −0.05 | −0.37 | −0.59 |
| **A3-ign around end** | −0.12 | +0.49 | **+0.72** | +0.52 | +0.14 | **−0.35** | −0.48 | −0.50 |

| read | A3 | A3-ign |
|---|---|---|
| inside > before (share of clips) | 0.879 | 0.914 |
| median argmax offset from onset (frames) | 20 | 18 |
| span cover share* | 0.730 | 0.680, Δ **−0.051 [−0.070, −0.032]** |

\* *Ad hoc definition, this read only:* share of span frames above `med(normal frames) + ½·(1 − med)` on the min-max
curve.

**Gap S is not smaller on `B`; its shape changed.** The pre-onset half of Gap S (the 0.55 baseline) is largely gone.
The other half — the score is locked to the impact and fades within ~4 s, so most of a long span is under-scored — is
**sharper**: higher peak, faster fade, less span covered. A training-side fix for span coverage is still open; its
references are `B`'s numbers (§5).

### 4.3 Δ (A3-ign − A3) by accident position and ego class (raw per-clip AUC, cluster CI)

| stratum | n | A3 | A3-ign | `p_T2` | Δ |
|---|---|---|---|---|---|
| centre < 0.40 | 94 | 0.700 | 0.736 | 0.439 | +0.036 [+0.014, +0.059] |
| 0.40–0.55 | 229 | 0.768 | 0.808 | 0.856 | +0.040 [+0.028, +0.052] |
| 0.55–0.70 | 205 | 0.764 | 0.794 | 0.982 | +0.030 [+0.014, +0.046] |
| ≥ 0.70 | 41 | 0.802 | 0.812 | 0.709 | +0.010 [−0.017, +0.040] |
| ego | 321 | 0.804 | 0.843 | 0.839 | +0.039 [+0.027, +0.049] |
| non-ego | 248 | 0.697 | 0.724 | 0.800 | +0.027 [+0.014, +0.041] |

The gain is not concentrated where `p_T2` is strong (largest at < 0.40, where `p_T2` is 0.44); the per-clip
correlation of AUC with `p_T2`'s is unchanged (0.135 → 0.147). The ego/non-ego gap widens slightly (0.107 → 0.118):
Gap N is untouched, as expected.

## 5. Consequences

- **v2's training recipe is now A3-ign** (`loss.dvs_anchor_mode = ignore`). The Final's sealed-set numbers (§19) belong
  to A3 and stay as recorded; A3-ign has dev numbers only until its second look.
- **References move to `B` = A3-ign**, recomputed by the same code: argmax-in-span **0.664**, pre **0.239**, inside
  **0.708** (min-max, seed-averaged), tail clips 277, corr(AUC, `p_T2` AUC) **0.147**.
- **SG-NM closed 2026-10-11**: a training-only normality branch on `B` failed its pre-registered frozen-trunk gate
  (Δ rank-avg − `y^bin` −0.012 [−0.016, −0.008], `f_2` −0.003 [−0.006, −0.001], T2-val −0.012, 5/5 seeds negative;
  read-out `outputs/v2/v2_e4b/REPORTS/e4a_readout.md`). Not pursued; its code was never committed.
- **Second look run** (addendum §23): the gain holds on DoTA-CAP-eval and T2-test (§6). Labelled a second look.
- **Any arm trained after this date carries `dvs_anchor_mode = ignore`**, including Nexar (deferred) if it is meant to
  be comparable with `B`.

## 6. Second look on the opened sets (addendum §23, Amendment 12)

Pre-registered in `97f53e9` (2026-10-11T07:25:49Z); opened `2026-10-11T07:33:44Z`
(`outputs/v2/v2_e4_0/second_look/SECOND_LOOK_OPENED.json`). Read-out
`outputs/v2/v2_e4_0/second_look/REPORTS/second_look_readout.{md,json}`. Every S7 gate passed (same checkpoints at
step 1740, 560 clips, 1,106 T2-test windows, A0's T2-test macro == the Final's). Paired with the Final's A3 by seed.

| read (second look at an opened set) | A3-ign | A3 (Final) | Δ A3-ign − A3 |
|---|---|---|---|
| DoTA-CAP (n/1397) eval macro, seed-averaged | **0.8120** | 0.7785 | paired t95 **+0.0376 [+0.0326, +0.0426]**, 5/5 (per seed +0.042 / +0.039 / +0.040 / +0.034 / +0.033) |
| clip-level Δ (cluster CI, printed) | | | +0.0336 [+0.0255, +0.0415] |
| `f_2` macro (position-controlled) | 0.8640 | 0.8484 | **+0.0157 [+0.0104, +0.0207]** |
| optimism dev − eval | −0.0209 | −0.0209 | — |
| A3-ign − A0 (printed) | | | +0.1276 [+0.0991, +0.1561] |
| `p_T2` alone | 0.8256 | 0.8256 | — |
| T2-test macro (1,106 windows) | **0.7202** | 0.6950 | +0.0252 [+0.0170, +0.0333] |
| T2-test micro / window AUC | 0.7334 / 0.7714 | 0.7095 / 0.7706 | window AUC +0.001 |
| O1′ vs A0, T2-test | 5/5 | 5/5 | — |

**S5 sentences (both licensed):** "Dropping the whole-anchor DVS label improves DoTA-CAP (n/1397) detection, on dev
and on a second look at the opened eval set." · "… beyond the position prior (second look)." Nothing re-decides (S2).

**Reading.**
- **The dev gain replicates almost exactly** (dev +0.0398 → eval +0.0376, both 5/5), and the optimism is the same
  −0.021 for both arms: the eval set is easier for both, so the gap between them is not a selection artefact.
- **The beyond-position gain is larger on eval** (+0.016 vs +0.010 on dev).
- **`p_T2` (0.8256) still beats A3-ign (0.8120).** The gap shrank from 0.047 (A3) to 0.014 but did not close: never
  write "the model beats the position prior". The licensed form is that the model adds to it (`f_2` 0.864 > 0.826).
- **On T2-test the window-level AUC does not move** (0.7706 → 0.7714): the macro gain is within-window ranking, the
  same floor-suppression mechanism as on dev (§3), not better separation of abnormal from normal windows.
- T2-test micro 0.733 sits above the clip oracle 0.704: "O1 FAIL as registered; not collapsed under O1′" (5/5).
- The Gap S sentence stays refused (§3). No DoTA-eval number exists for A3-ign (no DoTA pixels for the video stream).

## 7. Prior evidence, revisited

`ignore` had been measured twice, both n = 1 in regimes that could not localize (DADA Phase 1 −0.0056, TAD +0.0035).
E4-0 is its first measurement where the model localizes (T2, kernel 3), and the first with 5 seeds: a +0.040 gain with
sd 0.004. Lesson C29 now has a measured cost on this project's adopted model.
