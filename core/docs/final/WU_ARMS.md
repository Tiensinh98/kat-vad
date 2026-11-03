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
