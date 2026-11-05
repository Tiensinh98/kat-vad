# SA²-v2 (arm SV2): Nexar-trained, zero-shot DADA-2000, seed 2024

Written **before any SV2 number exists**. A defect found in this rule later becomes a dated note
below it, never a silent edit. Design: proposal `core/docs/final/` (LaGoVAD-STP-V2) §3.2.
Notebook: `colab/final/sa2v2_nexar_dada.ipynb`. Plan: `.project/plans/katvad-sa2v2-tokens.md`.

## 1. Question

With 7×7 pooled CLIP patch tokens as its local features, does SA²-v2 change zero-shot frame-level
detection on DADA-2000 against its host W0? Afterwards: does the same checkpoint read differently
on DoTA-CAP-dev, and does its map point at the agent there?

## 2. Arms

| Arm | What | Checkpoint |
| --- | --- | --- |
| **W0** (control) | v2 Nexar A2's function: CLIP + V2-S through `W_u` zero-init, no spatial module | `outputs/wu/W0/s2024`, reused; DADA read `outputs/zs/eval/dada/W0` |
| **SV2** | W0 + SA²-v2 = proposal row R7: scorer with priors and text term + `L_patch` + head on `x^AS` + motion dropout. No definition flip, no learnable prefix | `outputs/sv2/SV2/s2024` |
| **SV2_shuf** (C2) | SV2 with bag labels and class names permuted across bags, no KNN fillers | `outputs/sv2/SV2_shuf/s2024` |

Same recipe as v2 A2 s2024: 1,749 steps, batch 64, kernel 3, `mil_topk_pct` 5, `dvs span`, KIP off,
`is_egocentric` false. Only `stp.*` differs; the notebook asserts the config diff.

## 3. Read-out rule

**Primary, DADA-2000.** 370 sources of T2-test, whole video, every s8 row, `dada_whole_eval`, the
same protocol as W0's stored read. Per-video Δ = SV2 − W0 of the frame AUC over the two-class
videos, 95 % cluster bootstrap (cluster = the video).

* "gain shown" iff low > 0 and mean ≥ +0.01;
* "cost shown" iff high < 0;
* otherwise "no difference shown".

Every sentence carries **PRELIMINARY (n = 1 seed)** and adopts nothing.

**Secondary, DoTA-CAP-dev, run after DADA.** Protocol B (`s1[::3]`, whole clip, per-clip min-max,
native frames), per-clip Δ against W0's stored read, cluster = source video, same sentence.
STAUC and the pointing game of `q`, beside the uniform map, the centre prior and the untrained Mo
map on the same cells, with `v5_maps read` (same code as the earlier token test).

**Printed, never deciding:** micro, clip-level AUC, position rulers, R4's read if present.

## 4. Controls (proposal §4.3)

| Control | How, in this run | Reading |
| --- | --- | --- |
| C1 | Trained head on `x^AS` vs the same head on the 49-cell mean, per-video AUC on **DADA** | CI above 0 with mean ≥ +0.01: the weights add something to the pooled feature. Otherwise: no more than an average. Caveat: the head was trained on `x^AS` only, which favours `x^AS` |
| C2 | SV2_shuf, run **after** the main read | A movement counts only where it exceeds this run's |
| C3 | Top-3 `q` on abnormal vs normal rows within DADA's two-class videos, and on training bags | About equal: a peak in every frame, the map marks no anomaly |
| C4 | Final β₁…β₄, γ, T, b against C2 | A coefficient within C2's movement is not evidence |

## 5. Deviations and decisions

| # | Decision | Why |
| --- | --- | --- |
| D1 | Cells = 2×2 mean of plain CLIP's 14×14 tokens through `post_layernorm` + `visual_projection`, not ClearCLIP | User's choice 2026-11-05: 7×7, one CLIP pass. Known price in the proposal §3.2.1 and §3.2.6 |
| D2 | No pilot before training | User's decision 2026-11-05: a read on one set does not settle another |
| D3 | `τ_Δ` goes through CLIP's frozen `text_projection` | The host's text features are `CLIPTextModel.pooler_output`, not in `image_embeds`' space, where `P` lives. Without it the cosine compares two spaces and raises nothing. Host text path unchanged. The weights are a buffer, stored in the checkpoint |
| D4 | Level constants `m`, `s` of Mo and ρ frozen on Nexar's training sources, stored in the checkpoint | Proposal §3.2.2 |
| D5 | C1 on DADA, not on a Nexar validation split | No token cache for Nexar validation; user's decision. A zero-shot read of C1, not the proposal's |
| D6 | Motion dropout zeroes the whole `W_u` term, bias included, per bag | Zeroing the input would still add the bias |
| D7 | `W_u` host, not `tanh(α)·W·LN(v)` | `STP_BUILD.md` D1 |
| D8 | Encoder layer: GELU, no dropout | Not fixed by the proposal; chosen before the run |
| D9 | Bag length: at most 100 rows (5 clips × 20) | Measured from the construction; the proposal's old 120 was wrong |

## 6. What this run cannot license

* A re-run on tokens after the earlier negative (hit − chance −0.083) is a **forking path**, and is
  written as one (`RESULTS_V5_R7.md` §6).
* One seed: no adoption, no claim of an effect size.
* DADA has no boxes: DADA says nothing about where the map points.
* "SA²-v2 works on crops" or "on tokens" as a general claim: this run has 7×7 pooled tokens only.

## 7. Results

### 7.1 DADA-2000 (2026-11-05, seed 2024, read-out `REPORTS/sa2v2_readout.md`)

| Row | Macro | Micro | Middle ruler (no pixels) |
| --- | --- | --- | --- |
| W0 | 0.7615 [0.7369, 0.7859] | 0.6981 | 0.7342 [0.7076, 0.7594] |
| SV2 | 0.7133 [0.6883, 0.7379] | 0.6661 | same |

**SV2 − W0 (per video): −0.0482 [−0.0674, −0.0290] → cost shown, PRELIMINARY (n = 1 seed).**
Printed: SV2 − R4 −0.0446 [−0.0625, −0.0268]; SV2 sits below the middle ruler, W0 above it.
Checks: W0 equals its stored read; dump vs frame read max |Δ| = 0. Clip AUC empty (every source
is abnormal).

* C1: head on `x^AS` 0.6488 vs on the cell mean 0.4855, Δ +0.1634 [+0.138, +0.189] → gain by
  the rule. Caveat (§4): the mean is below chance, i.e. off the head's training distribution; and
  the branch alone (0.649) is below both W0 and the middle ruler.
* C3: top-3 `q` abnormal 0.553 vs normal 0.374, Δ +0.179 [+0.158, +0.199], 370 videos. DADA has
  no boxes; this says nothing about where the map points.
* Mechanics (C4 pending C2): β₂…β₄, γ moved ≤ 0.0035 from init, T 1.00 → 0.98, b unchanged; only
  the feature term moved (‖w‖ 0 → 0.145, spread 1.0 → 3.2, entropy 0.83 → 0.62). `‖x^AS‖/‖c_t‖`
  = 0.59. On training bags top-3 `q` rose on abnormal (0.53 → 0.66) and fell on normal
  (0.55 → 0.35); whether C2 does the same is open.

Not separable in this run: SV2 also carries motion dropout 0.3 (D6), which W0 does not, so the
cost is not attributed to the spatial module alone.

### 7.2 DoTA-CAP-dev (2026-11-05, same checkpoint)

Macro SV2 0.7848 [0.7681, 0.8016], W0 0.8110 [0.7949, 0.8271].
**SV2 − W0 (per clip): −0.0262 [−0.0400, −0.0122] → cost shown, PRELIMINARY (n = 1 seed).**

Maps (`v5_maps read`, 528 clips):

| Map | hit − chance | lift − lift_pre | hit − centre | STAUC |
| --- | --- | --- | --- | --- |
| Mo (untrained) | +0.293 [+0.271, +0.316] | +0.250 | −0.010 [−0.051, +0.030] | 0.334 |
| SV2 `q` | +0.148 [+0.120, +0.176] | +0.200 | **−0.154 [−0.201, −0.107]** | 0.284 |
| centre | +0.303 | −0.024 | 0 | 0.268 |
| uniform | −0.151 | −0.033 | −0.453 | 0.254 |

No map points by the rule. SV2's hits sit on ego accidents (+0.283) and agents > 8 % of the frame
(+0.319); non-ego −0.006, agents < 2 % ≈ 0 or below. The trained map is below the centre prior and
below the untrained Mo map it was given as a prior (β for Mo stayed at init, §7.1). The AUC column
is the same for every row, so it is not a property of the map and is not read.

**Both endpoints show a cost and the map does not point. Second negative on tokens after R7t
(`RESULTS_V5_R7.md`); §6 still applies.** C2 is not needed for this verdict (there is no positive
movement to protect); it would only explain the mechanics.


## 8. Arm W0D: is SV2's cost motion dropout's? (dated note 2026-11-05, before any W0D number)

Written after §7.1–§7.2 were read, so W0D is a post-hoc **attribution** arm: it can explain §7,
never reverse it, and it adopts nothing.

### 8.1 Arm

| Arm | What | Checkpoint |
| --- | --- | --- |
| **W0D** | W0 + `model.motion_dropout = 0.3` (= SV2's `stp.motion_dropout`, rule D6: the whole `W_u` term zero on 30 % of training bags). No SA²-v2, no cell cache | `outputs/sv2/W0D/s2024` |

Same recipe as W0 and SV2 (v2 A2 s2024, 1,749 steps); the notebook asserts that the config differs
from v2 A2 s2024 in `model.motion_dropout` only. Read with W0's own protocols: `dada_whole_eval`
on the A2 bake (no `--patch-dir`), protocol B on DoTA-CAP-dev.

Known difference beyond the dropout: the mask draws `torch.rand` from the global RNG, as SV2's does,
so batch-level randomness after the first draw differs from W0's. Same mechanism as SV2, so
W0D − W0 carries it and SV2 − W0D does not.

### 8.2 Read-out

Per set, with §3's sentence and intervals: **W0D − W0** (dropout alone) and **SV2 − W0D**
(spatial module on the dropout host); SV2 − W0 printed (= §7).

| W0D − W0 | SV2 − W0D | Sentence |
| --- | --- | --- |
| cost shown | not cost | the cost is motion dropout's; the spatial module shows none on top |
| not cost | cost shown | the cost is the spatial module's; dropout alone shows none |
| cost shown | cost shown | both carry a cost |
| not cost | not cost | neither change alone shows a cost |

DADA decides; DoTA-CAP-dev is read the same way and printed beside. One seed: PRELIMINARY, and
"not cost" never reads as "free".

## 9. Probe: why did β, γ, T, b not move? (dated note 2026-11-05, before any probe number)

`core/tools/sv2_probe.py`. Printed only; it changes no verdict of §7 and decides what a next arm
may change.

* **`adam`** on SV2's `checkpoint_last.pt`: per scalar coordinate (β₁…β₄, γ, log T, b) and per
  module (`w`, `w_cell`, `w_prior`, `pos`, `layer`, `branch_head`, `W_u`): RMS displacement / Σ lr,
  gradient RMS, share of `√v̂` < 10 eps, SNR beside the zero-mean-noise SNR. Reading (thresholds of
  `STP_BUILD.md` §5 D-1): *dead* (≥ half below 10 eps) → the gradient path is the defect;
  *noise* (SNR < 2× noise) → a larger LR would random-walk; *lr* (coherent, < 0.1 of the budget
  used) → the LR budget is the limit; *moved* otherwise.
* **`live`** at init and at the final checkpoint, 8 batches of epoch 0's order, fp32, train mode:
  each scalar's gradient from `host`, `λ_p L_patch`, `λ_b L_branch`, `total` (mean ± std,
  coherence |mean|/rms); ‖∇w‖ per term and the cosines between terms; and the per-frame spread
  across cells of `β·priors`, `γ z(cos)`, `wᵀH` and `l`.

Known before any probe number (arithmetic, not a measurement): at lr 5e-5 with 20 warm-up steps
and cosine decay over 1,749 steps, Σ lr ≈ 0.044, so no coordinate could have moved more than
≈ 0.044. β₁ starts at 1 on `z(Mo)` (`SV2_BETA_INIT`), so the scorer already started as the
per-frame motion prior.

### 8.3 W0D read-out (2026-11-05, seed 2024)

| Set | W0 | W0D | SV2 | W0D − W0 | SV2 − W0D |
| --- | --- | --- | --- | --- | --- |
| DADA (per video) | 0.7615 | 0.7529 | 0.7133 | −0.0086 [−0.0204, +0.0027] none | **−0.0396 [−0.0555, −0.0238] cost** |
| DoTA-CAP-dev (per clip) | 0.8110 | 0.8155 | 0.7848 | +0.0045 [−0.0021, +0.0110] none | **−0.0307 [−0.0430, −0.0183] cost** |

**§8.2 on both sets: the cost is the spatial module's; dropout alone shows none, PRELIMINARY
(n = 1 seed).** "None" for dropout is not "free" (§8.2).

### 9.1 Probe `adam` read-out (2026-11-05, SV2 s2024, step 1,749, Σ lr 0.04378)

Every SA²-v2 row reads **noise** by §9 (SNR < 2 × 0.183), none reads *dead* (no `√v̂` below
10 eps), none reads *lr*. The one row that **moved** is `branch_head` (SNR 0.548, 0.52 of the
budget), the training-only head on `x^AS`. SNR: β 0.14–0.24, γ 0.087, log T 0.030, b 0.137,
`w` 0.079, `w_cell` 0.126, `layer` 0.136, `w_prior` 0.099; host `W_u` 0.152.

Printed, not gating: Adam's moments are EMAs at the last step, where the cosine LR is ≈ 0, so the
SNR describes the end of training. Displacement says some rows moved earlier more than a random
walk would (log T 0.475, `w` 0.207 of the budget, against ≈ 0.05 for ±lr steps of random sign
over 1,749 steps); `live` at init reads the start.

