# v2 P3 — E1: rate-matched DoTA evaluation, result (2026-09-30)

Tool `python -m core.tools.rate_matched_eval`, runbook `colab/v2/p3_e1.ipynb`. Rule: proposal
§7.3 / §10.1 as pre-registered in `PREREG_ADDENDUM.md` §8 (Amendment 2, J1–J8) and §9
(Amendment 3: D12, J9′, J10), all written before any E1 number. Read-out
`Thesis-V2/outputs/REPORTS/v2_E1/e1_readout.{json,md}` + `e1_run.log` (gitignored; this file is
the record). No training: three finished KIP-off checkpoints re-scored on DoTA-dev.

**Checkpoints (D12):** s2024 = phase 4 (`DADA2000_orig_phase4`), s2025 and s2026 = retrained with
phase 4's exact command (`colab/v2/p3_retrain_kipoff.ipynb`; config identical to phase 4 apart from
the v2-only keys at their off values). J10: all three at `global_step` 2040. J9′ (arm A's curve
from `s1[::8]` equals the curve on `DoTA_s8_ncc`, every step of every clip): passed on all three.
DoTA-dev: 702 clips, all two-class at native frames, 93 source-video clusters; s1 length equals the
annotation's `num_frames` on every clip.

## Verdict (mechanical, J7): **adopt B** — DoTA at stride 3, whole clip

| Arm | Input | Macro, seed-averaged [95 % cluster CI] | Δ vs A [95 % cluster CI] | Micro (min-max) |
|---|---|---|---|---|
| A | `s1[::8]`, whole clip (today's protocol) | 0.6294 [0.610, 0.648] | — | 0.6001 |
| **B** | `s1[::3]`, whole clip | **0.6628** [0.644, 0.682] | **+0.0334 [+0.019, +0.048]** | 0.6142 |
| C | `s1[::3]`, W = 20 hop 4, overlap-averaged | 0.6644 [0.645, 0.684] | +0.0350 [+0.020, +0.050] | 0.6150 |
| *position ruler `t/N` (no pixels)* | | *0.5663* [0.547, 0.586] | | |

Both B and C are eligible (lower bound > 0); their means differ by 0.0016 < 0.01, so J7 adopts **B**
(no windowing). All scores interpolated to native frames and read against the same native labels.

## What the numbers say

1. **The gain is consistent, not a seed or clip-sampling accident.** B − A per seed: +0.035
   (s2024), +0.033 (s2025), +0.042 (s2026); C − A: +0.034, +0.036, +0.045. The cluster interval
   resamples source videos (D3), so the 19-clip-per-video correlation is inside it.
2. **It is largest where the anomaly is short**, which is the mechanism the proposal named
   (row P: a 3× time-scale gap). B − A by accident-share bin: `<30 %` +0.040 (343 clips),
   `30–50 %` +0.031 (247), `50–70 %` +0.018 (93), `>70 %` +0.012 (19, interval ±0.13). The
   score head's kernel is 3 steps: 2.4 s at stride 8, 0.9 s at stride 3, versus 0.8 s in T2
   training. A short accident is averaged away by a 2.4 s kernel; a long one is not.
3. **It is not position.** Every arm is well above the no-pixel ruler (0.566), and B and A read the
   same labels, so a position prior cannot differ between them.
4. **Windowing adds nothing beyond the rate.** C − B = +0.0016, inside every interval. The trunk
   was trained on 20-step windows, yet whole DoTA clips at stride 3 (median ≈ 35 steps) score the
   same: sequence length is not the operative mismatch, step duration is.

## What the numbers do not say

- **It is an evaluation-protocol change, not a better model.** The checkpoints are unchanged; every
  arm, including the A0 baseline, gains from it. v2 contrasts are read *within* one protocol.
- **Never put protocol-B numbers beside LaGoVAD's published 62.60 or any phase-4 DoTA number.**
  Those are stride 8 on stride-8 labels over all 1,397 clips; E1's A (0.629) is stride 8 on native
  labels over DoTA-dev. Proposal §7.3: both protocols are reported.
- Micro is printed, never decided on (J5).

## Consequences

- **The DoTA protocol for every later v2 read-out (P6 read-out, E3) is B**: `s1[::3]` whole clip,
  interpolated to native frames; protocol A is reported beside it.
- **D11 fires**: E2(b) and E2(c) are re-read on DoTA-dev at stride 3 before P5 fixes the CRN
  reference (`colab/v2/p2_e2_s3.ipynb`). T2 is unchanged.
- **A1's DoTA input** is baked from `s1[::3]` (`build_v2_inputs apply --stride 3`), not from
  `DoTA_s8_ncc`.
- Motion arms (A2/A3) still have no DoTA endpoint (D9): VideoMAE needs pixels at any stride.
