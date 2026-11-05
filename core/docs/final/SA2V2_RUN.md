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

Not run.
