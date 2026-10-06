# KAT-VAD v2 — E3 results (2 × 2 factorial × 5 seeds)

Pre-registration: proposal §10.1–§10.3; `PREREG_ADDENDUM.md` §17 (O1′), **§18 Amendment 9 (M1–M10)**, §15 G3/G6–G7.
Raw outputs: `outputs/v2/v2_s3/` (gitignored; this file is the durable record): `REPORTS/e3_readout.{md,json}`,
`REPORTS/pb/A{0..3}/{dota_dev,dota_cap_dev}/`, `REPORTS/position_prior/`, `A{0..3}/s{2024..2028}/diag/`.
Runbook `colab/v2/e3_factorial.ipynb`. Colab read-outs carry `commit: UNKNOWN` by design.

**Provenance (§18).** M1–M10 were committed in `59ee17e` before any E3 number and not edited since; the runs were
made with the notebook's unsigned-§18 guard disabled by hand, and the user authorized §18 on 2026-10-06, after the
read-out. Cite this beside any E3 number.

Arms: **A0** = raw CLIP (KIP off) · **A1** = + CRN (R2) · **A2** = + V2-S video stream (`W_u`, zero-init) · **A3** =
CRN + V2-S. Seeds 2024–2028, 20 epochs, every run at `global_step` 1740 (J10). Protocol B (E1).

## 1. Decision

**Adopt A3** (`F` = A0). A3 − A0 on DoTA-CAP-dev (n/1397) macro **+0.0855 [+0.0573, +0.1137]**, 5/5 seeds positive;
A3 − `F` is the same interval; the guardrails (O1′, every seed) hold.

Licensed sentences (M8): "v2 as a whole (CRN + V2-S) improves DoTA-CAP (n/1397) over A0"; "the video stream (V2-S)
improves DoTA-CAP (n/1397) on top of CRN (A3 − A1)". **Never "motion"** (D6, §15 G3). A2/A3: "**O1 FAIL as
registered; not collapsed under O1′**" (§17 Q5).

## 2. Contrasts (decision interval = paired t95 over 5 seeds, t = 2.776; MDE_E3 = 0.0282, E3-internal)

| contrast | set | mean Δ | t95 | per seed (2024…2028) | clip-level Δ (printed) |
|---|---|---|---|---|---|
| A1 − A0 | DoTA-dev | −0.0151 | [−0.0372, +0.0070] | +0.005 / −0.023 / +0.003 / −0.028 / −0.033 | −0.0191 [−0.0358, −0.0028] |
| A2 − A0 | DoTA-CAP-dev | +0.0924 | [+0.0505, +0.1343] | +0.118 / +0.052 / +0.135 / +0.084 / +0.073 | +0.0768 [+0.0546, +0.0999] |
| A3 − A0 | DoTA-CAP-dev | **+0.0855** | **[+0.0573, +0.1137]** | +0.116 / +0.068 / +0.102 / +0.078 / +0.064 | +0.0808 [+0.0613, +0.1000] |
| A3 − A1 | DoTA-CAP-dev | **+0.0978** | **[+0.0863, +0.1093]** | +0.109 / +0.087 / +0.093 / +0.105 / +0.095 | +0.0999 [+0.0833, +0.1172] |
| A2 − A1 | DoTA-CAP-dev | +0.1047 | [+0.0796, +0.1298] | +0.112 / +0.071 / +0.125 / +0.112 / +0.104 | +0.0959 [+0.0713, +0.1204] |

Interaction (A3 − A2) − (A1 − A0) on DoTA-CAP-dev: **+0.0054** (descriptive). DoTA-dev = 702 clips, DoTA-CAP-dev =
569 clips: different sets (D14), A1 − A0 is never put beside a CAP number.

Readings:

* **CRN alone does not transfer.** A1 − A0 is not detectable at MDE_E3; its clip-level interval is *negative*
  (printed only). The free rule fails (DoTA-dev mean < 0; share bin 50–70 reversed, −0.048 [−0.092, −0.005]) →
  `F` = A0. E2(c)'s probe predicted CRN − raw **+0.037** on transfer; trained, the sign did not hold. CRN still does
  what it was built to do mechanically: `V^t` source-shortcut AUC **0.9998 → 0.569**. Removing corpus separability
  did not buy DoTA macro.
* **The video stream carries the gain.** A3 − A1 has the narrowest interval in the table and every seed in
  +0.087…+0.109. A2 ≈ A3 (A3 − A2 ≈ −0.007 by seed, +0.004 by clip); §10.3's order takes A3 first.
* **Additive.** The interaction is ≈ 0: CRN neither helps nor hurts the stream on DoTA macro.

## 3. Guardrails (T2-val, per seed)

| arm | micro (5 seeds) | macro (5 seeds) | window AUC Δ vs A0 | O1 as registered | O1′ (M2) |
|---|---|---|---|---|---|
| A0 | 0.642–0.659 | 0.633–0.664 | — | fails s2026 (macro < micro) | n/a (fallback) |
| A1 | 0.662–0.674 | 0.647–0.672 | +0.001…+0.032 | FAIL 4/5 (macro < micro) | **PASS 5/5** |
| A2 | 0.694–0.719 | 0.680–0.700 | +0.083…+0.106 | FAIL 5/5 | **PASS 5/5** |
| A3 | 0.710–0.715 | 0.689–0.710 | +0.060…+0.084 | FAIL 5/5 (micro > oracle 0.6986) | **PASS 5/5** |

No arm shows C14's signature (window AUC up **and** macro down): every macro Δ vs A0 is ≥ −0.0001. O1's
"macro ≥ micro" leg fails on A0 itself (s2026) and on A1 4/5 — it is noise when the two sit within ~0.02 (lesson
candidate (at), now confirmed on 20 runs). `ρ_u` (last) A2 0.133, A3 0.108: the stream is used.

## 4. Printed, never decided on (M6)

| arm | DoTA-dev macro | DoTA-CAP-dev macro | T2-val micro | T2-val macro | window AUC | shortcut `V^t` | shortcut `y^bin` | pos R² `V^t` | Spearman(y, `p_T2`) |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 0.6793 | 0.6768 | 0.6473 | 0.6528 | 0.6803 | 0.9998 | 0.5558 | −0.232 | 0.331 |
| A1 | 0.6602 | 0.6577 | 0.6681 | 0.6593 | 0.7008 | 0.5690 | 0.5301 | −0.031 | 0.276 |
| A2 | — | 0.7536 | 0.7081 | 0.6892 | 0.7734 | 0.9997 | 0.5278 | −0.097 | 0.586 |
| A3 | — | 0.7576 | 0.7124 | 0.7034 | 0.7566 | 0.5892 | 0.5410 | +0.015 | 0.506 |

Share bins (DoTA-CAP-dev macro, `<30 / 30–50 / 50–70 / >70`): A0 0.674 / 0.693 / 0.651 / 0.633 · A3 0.754 / 0.769 /
0.743 / 0.753. A3 gains in every bin; the `>70` bin has 19 clips and an interval ~0.23 wide.

**Position.** `p_T2` (cubic, fitted on 3,756 T2 training windows, no pixels): **DoTA-dev 0.8187 [0.7997, 0.8377],
DoTA-CAP-dev 0.8220 [0.8026, 0.8415]**; `p_CAP` (in-domain ceiling) 0.852; `t/N` (monotone) 0.566 / 0.557. G6's
sentence fired for both motion contrasts: Δ Spearman A2 − A0 **+0.254 [+0.217, +0.291]**, A3 − A1 **+0.230
[+0.202, +0.258]** — "this gain co-moves with the position prior". Position R² on `V^t` stays ≈ 0 for every arm: the
trunk does not carry `t/T` linearly, so the co-movement is not an explicit position channel; a better detector of
mid-clip accidents correlates with position by construction. G6 is a flag, not a verdict — §5 measures it.

## 5. Beyond position — post-hoc, dev only (printed, not gated)

**Computed after the read-out was seen.** Per clip, `f_w = z(ȳ) + w·z(p_T2)` with `ȳ` the seed-averaged `y^bin`
from `clip_scores.npz` and `z` standardized within the clip; macro over the 569 two-class clips; Δ by clip-level
cluster bootstrap (video clusters, B = 5,000, numpy seed 0). Scratch computation, not a tool; its Final version is
pre-registered as §19 P4 and its means are the regression target of §19 P7.

| arm | arm alone | arm + `p` (w = 1) | arm + 2`p` (w = 2) | clips where arm > `p_T2` |
|---|---|---|---|---|
| `p_T2` alone | 0.8220 | — | — | — |
| A0 | 0.6768 | 0.7915 | 0.8194 | 28.1 % |
| A1 | 0.6577 | 0.7977 | 0.8245 | 24.4 % |
| A2 | 0.7536 | 0.8168 | 0.8264 | 33.0 % |
| A3 | 0.7576 | 0.8365 | **0.8424** | 35.0 % |

| Δ | raw | w = 1 | w = 2 |
|---|---|---|---|
| A3 − A0 | +0.0808 [+0.0622, +0.1004] | +0.0451 [+0.0301, +0.0593] | **+0.0230 [+0.0131, +0.0324]** |
| A2 − A0 | +0.0768 [+0.0536, +0.1001] | +0.0253 [+0.0091, +0.0415] | +0.0069 [−0.0045, +0.0188] |
| A3 − A1 | +0.0999 [+0.0830, +0.1166] | +0.0388 [+0.0268, +0.0504] | +0.0179 [+0.0089, +0.0268] |

(The raw column reproduces the read-out's clip-level Δ to 4 decimals, so the scratch path reads the same scores.)

Readings:

* **No arm beats position alone.** `p_T2` 0.822 > A3 0.758. Protocol-B DoTA macro is dominated by *where* in the
  clip the accident sits; DoTA's clipping puts it mid-clip, and T2's windows teach the same hump.
* **A0 carries nothing beyond position** (A0 + 2`p` 0.819 ≈ `p` 0.822). **A3 does** (+0.020 over `p` alone).
* **A3's gain over A0 survives position control but shrinks to roughly ¼–½** (0.081 → 0.045 → 0.023, every interval
  > 0). **A2's does not** (→ +0.007, interval ∋ 0). So the arm that differs is the one with CRN: on dev, CRN is what
  keeps the stream's surplus separable from position, although CRN alone moves nothing. One dev set, a weight chosen
  after looking — §19 tests it on DoTA-CAP-eval.

## 6. What this changes for the thesis

* The adopted model is **A3 = CRN + a frozen VideoMAE V2-S stream**; its effect is real at n = 5 and not motion.
* Every DoTA-CAP macro carries `p_T2` beside it. Without position control the honest headline is +0.086; with it,
  +0.02…+0.045 (dev, post-hoc) until §19 reads eval.
* No DoTA-CAP number beside LaGoVAD's 62.60, full DoTA, or phase 4 (D14).

## 7. Next

§19 (Amendment 10, signed 2026-10-06) → Final harness built (`core.tools.final_readout`, `--final` on
`protocol_b_eval` / `position_prior`, `colab/v2/final.ipynb`; its P7 gate reproduces §5 22/22) → advisor briefed on D6 and `p_T2` (`ADVISOR_BRIEF_E3.md`) → Final run, once.
