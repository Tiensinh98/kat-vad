# W_u arms — non-zero init and a learning rate of its own

Written 2026-11-02, **before any W-arm number**. Branch `v_final`. User request: "thử W_u init khác không hoặc LR".
Code: `MotionResidual(init_std, init_seed)`, `model.motion_init_std`, `train.motion_lr_mult`
(`core.train.optimizer_param_groups`), constants `WU_*`. Tests `core/tests/test_wu_arms.py`.
Runbook `colab/final/wu_arms.ipynb` (CPU-free of patches: no SA², no frame archive). Read-out `core.tools.stp_readout`
with `--name wu_readout`.

## 1. Why this, and what it can and cannot say

`W_u` is the only learned motion path of the host: `h = x + W_u (c·ũ/σ_u)`, zero-init, AdamW 5e-5, 1,749 steps. Measured on
STP R4 / R6 (s2024, `STP_BUILD.md` §5.1): ‖W_u‖ 0 → 1.41, RMS Δ / Σ lr 0.072, grad RMS 9.6e-3, SNR 0.22 (noise 0.18),
motion share ρ_u ≈ 0.15. v2 knew this path matters (zero-shot DoTA 0.81 comes from it). Two competing readings:

* **H-W (under-trained):** a zero start plus a 5e-5 budget leaves `W_u` short of where it would settle, so a live start (WI)
  or a larger step (WL) should raise DoTA-CAP-dev macro.
* **H-0 (budget is not binding):** D-1 found the rows' gradients near the noise SNR; a larger step only random-walks, and
  v6 saw a branch fit Nexar better (mil 0.115 vs 0.200) and transfer worse. Then WI / WL are null or negative.

One seed cannot separate a +0.01 gain from the seed spread (v2 A2 seeds spread 0.866–0.886 on `nexar_val`).
**Everything below is PRELIMINARY (n = 1 seed) and adopts nothing.**

## 2. Arms (host = v2 Nexar A2's recipe, `stp.sa2 = false`, released prompts; Nexar `window`; s2024)

| arm | `W_u` init | LR on `W_u` | config overrides |
|---|---|---|---|
| **W0** (control) | 0 | 5e-5 | — (the shipped host, trained through this branch's code) |
| **WI** | N(0, 0.003²), bias 0 | 5e-5 | `model.motion_init_std=0.003` |
| **WL** | 0 | 5e-4 (×10) | `train.motion_lr_mult=10` |
| WIL (optional) | N(0, 0.003²) | 5e-4 | both |

* **0.003 is derived, not tuned:** the learned ‖W_u‖_F = 1.41 over 512 × 384 entries is 0.0032 per entry.
  ×10 is the budget rule of the Nov 2 proposal §3.5.1 (5e-5 → 5e-4). No other value is run, so no grid is read.
* The init draws from a private generator seeded with `train.seed`; **every other parameter equals W0's at init** and the
  global RNG stream (data order, DVS) is unchanged (`test_only_w_u_differs_from_the_zero_init_model`).
* The multiplier changes `W_u`'s group only (weight + bias); the warm-up / cosine factor is the same. Weight decay stays 0.01
  (AdamW decays by lr·wd, so `W_u`'s decay is also ×10 in WL / WIL).
* The control is trained here, not borrowed: **R1 s2024 (0.8102) is printed beside it**, and `|W0 − R1| > 0.01` is flagged
  "host does not reproduce R1" (a flag, not a stop; every contrast below is vs W0).

## 3. Read-out (rule fixed here)

* **Endpoint:** DoTA-CAP-dev protocol-B macro AUC (the 569 clips, `s1[::3]`, per-clip min-max). Δ = arm − W0 per clip,
  mean with a 95 % source-video cluster bootstrap (`stp_readout`).
* **Sentence:** "gain shown" iff the bootstrap low > 0 **and** mean ≥ +0.01 (`V5_MIN_GAIN`); "cost shown" iff high < 0;
  else "no difference shown". Always labelled `PRELIMINARY (n = 1 seed)`.
* **Printed, never deciding:** Δ on ego-involved / not, the t/N ruler, `p_T2` 0.822 (and the cubic position prior 0.852 on
  CAP-dev), and from `metrics.jsonl` the first → last ρ_u, ‖W_u‖ and `mil`.
* **Mechanics reading (printed, labels only):**
  * WI: ρ_u at step 0. ρ_u ≥ 0.5 at step 0 means the init swamps the CLIP stream; the arm is then reported as "init too
    large", not as a statement about live starts.
  * WL: ‖W_u‖ last vs W0's. ≤ 1.5× W0's means the larger step was not used, and a null says nothing about the budget.
  * Overfit tell (v6): `mil` last clearly below W0's together with a negative Δ.

## 4. What it licenses

| outcome | what next |
|---|---|
| WI or WL "gain shown" | the same arm on seeds 2025–2026 (3 seeds, 3/3 same sign, mean ≥ +0.01) before any claim |
| "no difference shown" / "cost shown" on both | closed: `W_u`'s zero start and 5e-5 are not the limit at one seed; no further LR / init sweep |
| WI ρ_u(0) ≥ 0.5 or WL ‖W_u‖ ≤ 1.5× W0 | the arm did not test its question; one corrected arm needs its own rule first |

**Not licensed by any outcome:** "the motion stream is under- / over-trained" without the mechanics above; any statement about
SA², the prefix or OSTP; any claim beyond DoTA-CAP-dev at n = 1; a pick between WI and WL by their Δ.
`stp_diag`'s D-1 budget for a `W_u` row assumes one LR; under WL its Σ lr must be multiplied by 10 before reading it.

## 5. Read-out (s2024, 1,749 steps; raw `outputs/v_final/w_u/wu_readout.md`) — **closed, PRELIMINARY (n = 1 seed)**

DoTA-CAP (n/1397) dev, protocol B, 569 clips. Printed: `p_T2` 0.8220 above every row.

| row | macro [95 % cluster CI] | Δ vs W0 | sentence | ρ_u first → last | ‖W_u‖ first → last | `mil` first → last |
|---|---|---|---|---|---|---|
| **W0** | 0.8110 [0.7948, 0.8269] | — | control | 0 → 0.146 | 0 → 1.26 | 0.694 → 0.176 |
| **WI** | 0.8106 [0.7948, 0.8260] | −0.0004 [−0.0032, +0.0021] | no difference shown | 0.058 → 0.157 | 1.33 → 1.84 | 0.695 → 0.177 |
| **WL** | 0.7693 [0.7491, 0.7889] | **−0.0417 [−0.0538, −0.0297]** | **cost shown** | 0 → 0.570 | 0 → 6.58 | 0.694 → **0.061** |
| R1 (printed) | 0.8102 [0.7942, 0.8260] | −0.0008 [−0.0026, +0.0009] | no difference shown | — | — | — |

* **Control:** |W0 − R1| = 0.0008 < 0.01, no flag. W0 equals v2 Nexar A2 s2024 to the last digit (`ZS_TAD_DADA.md` §2). WIL was not run.
* **Mechanics (§3), both arms tested their question:** WI ρ_u(0) = 0.058 < 0.5, so the init did not swamp CLIP; WL ‖W_u‖
  last = 5.2× W0's, so the larger step was used. **Overfit tell fires on WL:** `mil` 0.061 vs 0.176 with a negative Δ.
* **Printed by ego flag:** WL − W0 ego −0.0274 [−0.0395, −0.0159], non-ego −0.0602 [−0.0807, −0.0391]; WI − W0 both ∋ 0.
* **Outcome (§4, row 2):** closed. `W_u`'s zero start and 5e-5 are not the limit at one seed; no further LR / init sweep.
  This read supports H-0 over H-W at n = 1. Never write "the motion stream is over-trained" or a pick between WI and WL.
