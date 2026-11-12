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

### 9.2 Probe `live` read-out (recorded 2026-11-05, SV2 s2024, `REPORTS/probe/live_{init,final}/`)

8 batches of epoch 0's order, fp32, train mode, no optimizer step. Printed only.

| Quantity | Init | Final |
| --- | --- | --- |
| β₁ gradient, host (coherence) | −9.53e-03 (0.95) | −4.71e-04 ± 4.8e-02 (0.01) |
| β₁ gradient, `L_patch` (coherence) | +3.63e-02 (0.73) | +8.18e-03 (0.56) |
| ‖∇w‖ host / patch / branch | 0.068 / 0.219 / 0.001 | 1.027 / 0.175 / 0.083 |
| cos(∇w) host, patch | −0.082 | +0.131 |
| Cell spread prior / text / residual | 1.000 / 0.000 / 0.000 | 0.992 / 0.001 / 3.250 |

* At init `L_patch` drives the scorer and pushes β₁ against the host, coherently. At the end the
  host dominates `w`, and its gradient on the scalars is incoherent (coherence ≤ 0.39 on every β, γ).
* The trained map is ranked by the residual, not by the priors, and the text term contributes
  almost nothing (spread 0.001).


## 10. Arm SV3: gated x^AS (dated note 2026-11-05, before any SV3 number)

Written after §7–§9 were read: §8.3 put SV2's cost on the spatial module, §9.1 read every
scorer row as noise. SV3 asks whether that cost goes away, and whether the host takes `x^AS`,
when it may refuse it. Post-hoc arm: it can explain §7, never reverse it; it adopts nothing.

### 10.1 Arm

| Arm | What | Checkpoint |
| --- | --- | --- |
| **SV3** | SV2 with `c̃ = c_t + tanh(g)·x^AS`, scalar `g = 0` at init (`stp.sv2_gate=true`). Everything else as SV2: scorer, priors, text term, `L_patch`, `L_branch` on the ungated `x^AS`, motion dropout 0.3 | `outputs/sv2/SV3/s2024` |

At step 0 SV3's detection path equals W0D's (same host, same dropout, `x^AS` × 0), so **W0D is the
control**. At `g = 0` no host loss reaches the scorer through `c̃`; the scorer learns from
`L_patch` and `L_branch` alone until `g` moves.

**D10. Gate LR = 25 × base, its own AdamW group** (`stp.gate_lr_mult`, `SV3_GATE_LR_MULT`).
Derived, not tuned: Σ lr at the base rate is 0.04378 (§9.1), so at ×1 `|tanh g|` could never pass
0.044 and the gate would read "shut" by construction. ×25 lets a coherent gradient reach `g ≈ 1`
(1 / 0.04378 = 22.8 → 25); a sign-incoherent one random-walks ≈ 25 × 5e-5 × √1749 ≈ 0.05.

### 10.2 Read-out

Per set, §3's sentence on **SV3 − W0D** (primary), and printed **SV3 − SV2** (what the gate
changed) and SV3 − W0. Gate: last logged `tanh(g)`; **opened** iff `|tanh g| ≥ 0.2`
(`SV3_GATE_OPEN`, ≈ 4× the random-walk reach of §10.1), else **shut**.

### 10.3 Verdict sentence (per set; DADA decides, DoTA-CAP-dev printed beside)

| Gate | SV3 − W0D | Sentence |
| --- | --- | --- |
| shut | not cost | the module stays unused when the host may refuse it |
| shut | cost | cost with a shut gate (not through `x^AS`: read the mechanics) |
| opened | gain | opened and helps |
| opened | no difference | opened without a shown effect |
| opened | cost | opened and costs: training wants `x^AS`, transfer does not |

One seed: PRELIMINARY. A shut gate does not show the module is useless, only that the host does
not take it under this objective; "opened and helps" adopts nothing without seeds and a
pre-registered confirmation. The pointing maps (`v5_maps read`) are read only if the gate opens.

### 10.4 Read-out (2026-11-05, seed 2024, `REPORTS/sv3_readout.md`)

Gate `tanh(g)`: 0 → **−0.4843**, opened by the rule (|·| ≥ 0.2).

| Set | W0 | W0D | SV2 | SV3 | SV3 − W0D (primary) | SV3 − SV2 |
| --- | --- | --- | --- | --- | --- | --- |
| DADA (per video) | 0.7615 | 0.7529 | 0.7133 | 0.7287 | **−0.0242 [−0.0383, −0.0103] cost** | +0.0154 [+0.0026, +0.0279] gain |
| DoTA-CAP-dev (per clip) | 0.8110 | 0.8155 | 0.7848 | 0.7863 | **−0.0292 [−0.0396, −0.0187] cost** | +0.0015 [−0.0073, +0.0103] none |

SV3 − W0: DADA −0.0328 [−0.0507, −0.0152], DoTA −0.0247 [−0.0359, −0.0134], both cost. Macros
recomputed from `clip_aucs.json` match; checkpoint and `metrics.jsonl` both end at step 1,749.

**§10.3 on both sets: opened and costs, PRELIMINARY (n = 1 seed). Third negative on tokens after
R7t and SV2.** Adopts nothing. Against SV2's cost on W0D, the gate recovers ≈ 39 % on DADA
(0.0154 / 0.0396) and ≈ 5 % on DoTA (0.0015 / 0.0307). SV3 sits below DADA's middle ruler
(0.7287 vs 0.7342, intervals overlap).

**Dated note on the rule's sentence (2026-11-05, after the read).** §10.3 did not consider the
gate's sign. "Training wants `x^AS`" reads as adding it; here the host **subtracts** it. The verdict
stands; read it as "training moved the coefficient to −0.48, transfer does not gain".

Printed post hoc, none changes the verdict:

* **The gate's motion is coherent.** Negative on 99.8 % of steps, |tanh g| ≥ 0.2 from step 543,
  |g| = 0.53 ≈ 10 × the random walk of §10.1. The probe's `noise` label on the gate describes the
  end of training (§9.1's caveat), not the trajectory. It flattens from step ≈ 1,200, when the LR
  left allows |Δg| ≤ ≈ 0.05, so this is not evidence of an equilibrium.
* **`x^AS` improves the training fit at either sign.** `mil`, steps 1,650–1,749: W0D 0.301, SV2
  0.221 (coefficient +1), SV3 0.223 (−0.48); at steps 250–350, with the gate still small: 0.595,
  0.557, 0.587. Together with the transfer cost this looks like overfitting through the token
  channel, but no Nexar validation exists in this run: a hypothesis.
* **The scorer's scalars are still at init**, as in SV2: β, γ within ≈ 0.0025; T 0.96. Map fit on
  training bags as SV2: top-3 `q` abnormal 0.673, normal 0.333 (SV2 0.664, 0.347). `q` is flatter
  (spread 1.75 vs 3.20, entropy 0.84 vs 0.62).
* **Position.** Mean within-video Spearman of the score with the middle ruler, W0D / SV2 / SV3: DADA
  0.364 / 0.300 / 0.311; DoTA 0.559 / 0.498 / 0.490. Part of the cost may be lost position
  alignment rather than lost detection. Not tested.

Limits:

* **W0D is not a paired control in practice.** §10.1's "equals W0D at step 0" holds for the form of
  the function only: step-1 `mil` 0.695 vs 0.706, `dvs_sup` 0.703 vs 0.551. Init and batch order were
  not separated.
* The intervals resample videos or clips, not runs: no run-to-run noise, so no effect size.
* **Still owed by §10.3:** `v5_maps read` on SV3. `colab/final/sv3_gate.ipynb` does not produce it.
* C2 (`SV2_shuf`) was not read: `REPORTS/sa2v2_readout.md` still prints `pending`.

## 11. Arm SV4: `W_p` on x^AS (dated note 2026-11-05, before any SV4 number)

Written after §10.4 was read, on the user's choice. **Question:** if the patch stream enters the
host the way the video stream does, through a learned zero-init matrix, does the cost of §10.4 go
away? Post-hoc arm: it can explain §7–§10, never reverse them; it adopts nothing.

**Prediction, recorded before the run:** "no difference shown" or "cost shown" on DADA, with `mil`
below W0D's. Reason: across WL, SV2 and SV3 the cost came with a better training fit, and R4, which
adds `x^AS` by a plain sum, shows no cost (`STP_BUILD.md` §3.1), so the form of the sum does not look
like the cause. This pattern was noticed after the numbers and is not tested.

### 11.1 Arm

| Arm | What | Checkpoint |
| --- | --- | --- |
| **SV4** | `c̃ = c_t + W_p x^AS`, `W_p ∈ ℝ^{512×512}` + bias, both zero at init (`stp.sv2_proj=true`), in place of SV3's gate. Padded rows stay `c_t`. Everything else as SV3: scorer, priors, text term, `L_patch`, `L_branch` on the unprojected `x^AS`, motion dropout 0.3 | `outputs/sv2/SV4/s2024` |

Same recipe as W0D, SV2 and SV3. Zero-shot reads with their protocols: DADA-2000 whole video and
DoTA-CAP-dev protocol B.

| # | Decision | Why |
| --- | --- | --- |
| D11 | `W_p` trains at the base LR, in the one AdamW group | As `W_u` at ×1, which reached `ρ_u` 0.15. WL (×10) cost −0.042 with `mil` 0.122 (last 100 steps), so ×10 is not the analogue of the host's path |
| D12 | `W_p` has a bias, masked on padded rows | As `W_u` |
| D13 | **SV4 builds the whole SA²-v2 module under a private RNG seeded with `train.seed`** (`torch.random.fork_rng`) | §10.4: SV3's step-1 losses differ from W0D's, because the module's init draws shift the host's init. With the fork, host init and data order equal W0D's, so W0D is a paired control. SV2 and SV3 keep their build, so their checkpoints and `sv2_probe` reads are unchanged |

Known limits: D13 means SV4's scorer starts from other draws than SV3's, so SV4 − SV3 is not a
one-change contrast (printed only). Pairing covers step 1; AMP and GPU rounding may still differ.

### 11.2 Read-out

Per set, §3's sentence on **SV4 − W0D** (primary); printed SV4 − SV3 and SV4 − W0.

Mechanics, printed: share `‖W_p x^AS‖ / ‖c‖` (`sv2_patch_share`) and `‖W_p‖` (`sv2_wp_norm`). `W_p` is
**used** iff the share's mean over the last 100 steps is ≥ 0.05 (`SV4_USED_SHARE`, ≈ ⅓ of W0's `ρ_u`
0.152), else **unused**. Also `mil` over the last 100 steps vs W0D's (last-batch values cannot rank
arms, `LOCAL_STEP1B.md` §5), and step-1 `mil` and `dvs_sup` vs W0D's, paired iff |Δ| ≤ 1e-3
(`SV4_PAIRED_TOL`). "Overfit tell" = cost shown with `mil` below W0D's.

### 11.3 Verdict sentence (per set; DADA decides, DoTA-CAP-dev printed beside)

| `W_p` | SV4 − W0D | Sentence |
| --- | --- | --- |
| unused | not cost | `W_p` stays small: the host does not take `x^AS` under this objective |
| unused | cost | cost with `W_p` unused (not through `x^AS`: read the mechanics) |
| used | gain | used and helps |
| used | no difference | used without a shown effect |
| used | cost | used and costs |

### 11.4 Stop rule (fixed now)

| DADA | DoTA-CAP-dev | Next |
| --- | --- | --- |
| gain shown | not cost | its own rule first, then seeds 2025–2026 of SV4 and W0D; `v5_maps read` on SV4 |
| anything else | any | **the token line closes: no further arm, gate, LR or fusion variant of SA²-v2 on tokens** |

One seed: PRELIMINARY. "Unused" does not show the patches are useless, and no outcome licenses a
claim about crops, objects or native resolution (§6).

### 11.5 Read-out (2026-11-06, seed 2024, `REPORTS/sv4_readout.md`)

`W_p` **used**: share `‖W_p x^AS‖ / ‖c‖` over the last 100 steps 0.1247 (bar 0.05), `‖W_p‖` 1.3728. `mil`,
last 100 steps: SV4 0.1813, W0D 0.3003. **Not a paired control in practice** even with D13: step-1 `mil` 0.69059 vs
0.69497 and `dvs_sup` 0.70385 vs 0.70297 (|Δ| > `SV4_PAIRED_TOL` 1e-3).

| Set | W0 | W0D | SV3 | SV4 | SV4 − W0D (primary) | SV4 − SV3 (printed) |
| --- | --- | --- | --- | --- | --- | --- |
| DADA (per video) | 0.7615 | 0.7529 | 0.7287 | 0.7390 | **−0.0139 [−0.0274, −0.0007] cost** | +0.0103 [−0.0032, +0.0241] none |
| DoTA-CAP-dev (per clip) | 0.8110 | 0.8155 | 0.7863 | 0.8092 | −0.0063 [−0.0153, +0.0026] none | +0.0229 [+0.0122, +0.0334] gain |

SV4 − W0: DADA −0.0225 [−0.0379, −0.0072] cost, DoTA −0.0018 [−0.0127, +0.0090] none. SV4's middle ruler
(no pixels) on DADA 0.7342 [0.7076, 0.7594].

**§11.3: DADA "used and costs" (overfit tell: yes, `mil` below W0D's); DoTA-CAP-dev "used without a shown
effect" (overfit tell: no). PRELIMINARY (n = 1 seed). Prediction of §11 (no difference or cost, `mil` below W0D's)
held.** The DADA interval's upper bound is −0.0007: the cost is barely shown and unpaired at step 1, so read it as
"no evidence that the matrix helps", not as a size.

**§11.4 fires ("anything else"): the token line closes.** See §12.

## 12. Control C2 (`SV2_shuf`) and closing the token line (2026-11-06)

### 12.1 C2 read-out (seed 2024, `REPORTS/sa2v2_readout.md`, `outputs/v_final/sv2/SV2_shuf/s2024`)

Run check: checkpoint and `metrics.jsonl` both end at step 1,749 (`verified.json`). The permutation lives in the
staged data (`shuffle_report.json`, not pushed): `config.yaml` has **no shuffle marker** and still shows
`knn_filler_ratio: 0.5` (as SV2's does), so "labels were shuffled" is read from the losses, not from the config.
The host cannot fit the permuted labels: `mil` over steps 1,650–1,749 0.717 (SV2 0.221, W0D 0.300), `sv2_branch`
0.692 ≈ ln 2.

First → mean of the last tenth of training:

| Diagnostic | SV2 | SV2_shuf | Reading |
| --- | --- | --- | --- |
| entropy of α | 0.828 → **0.625** | 0.828 → **0.960** | label-driven |
| logit spread | 1.00 → **3.20** | 1.00 → 0.63 | label-driven |
| top-3 `q` abnormal / normal | 0.533 / 0.547 → **0.663 / 0.348** | 0.529 / 0.527 → 0.514 / 0.490 | label-driven |
| `sv2_patch` / `sv2_branch` | 0.742 / 0.693 → **0.545 / 0.606** | 0.731 / 0.693 → 0.678 / 0.692 | label-driven |
| `sv2_w_norm` | 0 → 0.145 | 0 → 0.063 | label-driven |
| β₁…β₄, γ, T, b | each moves ≤ 0.02 | each moves ≤ 0.01 | **not separable from noise** |

The read-out tool's last column ("moved further than shuffled?") compares absolute displacement, so it prints
`True` for β₁, β₃, γ, T and b on movements of 0.001–0.02. **Do not cite that column for the scalars**: only the
rows above the last one carry a margin over C2. C1 (head on `x^AS` 0.649 vs 0.486 on the cell mean; the mean head
is below chance, a caveat) and C3 (top-3 `q` abnormal − normal rows +0.179 over 370 videos) are as in §7.1.

**Reading (PRELIMINARY, n = 1 seed).** SA²-v2 **did learn a label-dependent map**: entropy, spread, top-3 `q` and
the two module losses move in the real run and not in the permuted one. What it learned sits in the residual
`wᵀH` (§9.2: spread 3.25 vs priors 0.99, text 0.001); the scalar terms stayed at their start in both runs.
That map does not point (§7.2): SV2 hit − chance on boxes > 8 % +0.319 (Mo +0.316) and ego-involved +0.283, but
non-ego −0.006 and boxes < 0.5 % −0.032 (Mo +0.060 on the latter), STAUC 0.2844 vs the untrained Mo 0.3343. The
subsets where it fails are the ones H1 targets.

What C2 does **not** show: that the label-driven structure is a shortcut (a map that separates abnormal from
normal bags is also what a good map would do); that the fit is not overfit (the permuted run does not memorise
in 1,749 steps either, so the fall in `mil` is not capacity alone); or why the map misses small and non-ego
agents. The "learned something other than the agent's location" reading is a hypothesis; the position and ego
correlation of the map peak was not measured.

### 12.2 The token line closes

By §11.4 (SV4: DADA not a gain). **No further arm, gate, LR or fusion variant of SA²-v2 on 7×7 tokens.** Record:

| Arm | What | DADA vs W0D | DoTA-CAP-dev vs W0D |
| --- | --- | --- | --- |
| SV2 | module, `c̃ = c + x^AS` | −0.0396 cost | −0.0307 cost |
| SV3 | `tanh(g)` gate (g → −0.48) | −0.0242 cost | −0.0292 cost |
| SV4 | zero-init `W_p` | −0.0139 cost (barely) | −0.0063 none |

Together with R7t (`RESULTS_V5_R7.md`) and the separate-branch line on crops (`LOCAL_STEP1.md` §7,
`LOCAL_STEP1B.md` §5: L1 0.612, L1mo 0.704 against the training-free Mo 0.7132), every learned local branch
trained here on Nexar's bag labels ended no better than the training-free motion prior. n = 1 seed in every
row except the earlier crops reads; the intervals resample videos or clips, not runs.

**Not licensed by any outcome:** "patches / tokens are useless"; any claim about crops, objects, native
resolution or a larger training set; "SA²-v2 did not learn" (C2 shows it did); the size of any cost. **Still open
and not gating:** the SV3 maps (§10.3), and an optional free read of where the SV2 map peaks (frame position,
ego flag) on the existing outputs.

**2026-11-12, partial read (post-hoc, printed only; `KIM_RUN.md` §24.2):** detection AUC by ego flag. No SA²-v2 arm improves ego-uninvolved accidents (DADA: SV2 −0.045, SV3 −0.028, SV4 −0.020 against W0D, all excluding 0; DoTA: SV3 −0.031, SV2 / SV4 not separated). Where the map *peaks* by ego flag is still unread.

**Implication for the thesis claim (a decision for the user, not made here).** On the 528 boxed clips the
training-free Mo map has the highest STAUC (0.3343, against SV2 0.2844, centre 0.2677, uniform 0.2544) and ties
the centre prior on the pointing game (hit − centre −0.010 [−0.051, +0.030]); SV2 is below the centre prior on
that read (−0.154). H4 asks that the learned map beat an untrained change map: it does not. The gain claim of H1
has no supporting row either (§7, §10.4, §11.5).
