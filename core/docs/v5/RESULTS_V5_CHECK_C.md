# v5 check C — change maps vs the centre prior on DoTA-CAP-dev boxes: **CLIP-BEATS-CENTRE**

Rule: `core/docs/v5/PREREG_V5.md` **§3.7** (written 2026-10-30 after the v5.1 PILOT-FAIL; post-hoc, disclosed).
Tool `core/tools/v5_centre_check.py`, runbook `colab/v5/check_c_centre.ipynb` (CPU, training-free). Raw read-out
`outputs/v5/v5_check_c/REPORTS/check_c_readout.md`.

## 1. Outcome

**CLIP-BEATS-CENTRE** (C-3: C14 hit − centre 95 % low > 0). Claude's pre-run prediction (C-7) was CENTRE-WINS — wrong.
Validity: 569 / 569 dev maps, 0 failed, 0 missing; **G-C0 PASS** (C14 hit − chance +0.2878 = Stage 0's, |Δ| 0).

| map | clips | hit − centre | hit − chance | centre − chance | mass − centre | mass − chance |
|---|---|---|---|---|---|---|
| **C14** (primary, 14 × 14) | 528 | **+0.1313** [+0.0929, +0.1691] | +0.2878 | +0.1565 | **+0.0197** [+0.0146, +0.0249] | +0.0551 |
| C14→7 (2 × 2 mean, 7 × 7) | 528 | **−0.0019** [−0.0423, +0.0372] | +0.3008 | +0.3028 | **+0.0154** [+0.0105, +0.0203] | +0.0493 |
| V7 | 528 | −0.2995 [−0.3372, −0.2624] | +0.0032 | +0.3028 | −0.0322 | +0.0018 |
| V14 | 528 | −0.1680 [−0.2056, −0.1305] | −0.0115 | +0.1565 | −0.0369 | −0.0015 |

By subset (hit − centre, printed): C14 ego +0.188, non-ego +0.067 [+0.017, +0.112]; C14→7 ego +0.090, non-ego
**−0.108**; box share < 0.5 %: C14 −0.044 (ns), C14→7 −0.310. V7 / V14 below centre in every subset.

## 2. Reading (printed, not gated)

1. **Most of the 14 × 14 margin is the prior's tie.** On an even grid the centre Gaussian's four middle tiles tie and
   `argmax` takes (6, 6), a 16-px tile up-left of the frame centre (disclosed in C-3, not adjusted). Its centre − chance
   halves (+0.303 at 7 × 7 → +0.157 at 14 × 14). With one true centre tile (7 × 7) the hit read **ties** (−0.002).
2. **The tie-free read is positive on both grids.** Mass-in-box uses the whole map (the symmetric Gaussian shares the
   four middle tiles): CLIP change puts +0.015 … +0.020 more of its mass in the box than the centre prior — about ⅓ of
   its +0.055 lift over a uniform map.
3. **VideoMAE V2-S change does not point** beyond the centre prior on any grid or subset (as Stage 0 and v5.1).

## 3. What is licensed

- "CLIP-patch change (lag 1, 14 × 14) carries spatial evidence beyond a centre prior on DoTA-CAP-dev boxes (hit −
  centre +0.131, inflated by the prior's even-grid tie; mass-in-box − centre +0.020); pooled to 7 × 7 the hit read
  ties the prior and only mass stays above it (+0.015)."
- Per C-6: R7 becomes **eligible** for an amendment, at most **MAPS-ONLY** — taken up as PREREG_V5 §4 (branch `v5-r7`).

**Not licensed:** "CLIP localizes the agent" without the centre-prior comparison; any 7 × 7 pointing claim beyond mass;
any detection claim; anything about non-ego or small-box clips beyond the printed ties / losses.
