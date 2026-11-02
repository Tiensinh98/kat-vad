# LaGoVAD-STP: build notes and the first-run rule

Source: `core/docs/final/Thesis proposal Deep learning-based traffic video anomaly detection.md`
(§3.2 STP, §3.3 training, §4.5 ablation rows). This file records **what was built**, **where it
deviates from the proposal**, and the **read-out rule for the first run**, written before any STP
number exists. Status: built 2026-11-01, not run.

## 1. What is built

| Part | Proposal | Code |
| --- | --- | --- |
| Patch features | 7×7 grid of 32×32 windows of the 224² resize, each re-encoded by frozen CLIP (§3.2.2) | `core/tools/clip_windows.py` (`--grids 1 7 --cells resized224`), `core/tools/stp_patches.py nexar` |
| Motion prior | `Mo = ‖2P̂_t − P̂_{t−1} − P̂_{t+1}‖`, top-K, K = 12; ego variant subtracts the frame-mean change | `core/data/stp_patches.py` (`motion_magnitude`, `select_top_k`), offline, per source video |
| SA² | `a = softmax((W_q c)ᵀ(W_k P)/√128)`, `x^AS = Σ a P`, `c̃ = c + x^AS`; `W_q` = 0, `W_k` Xavier | `core/models/sa2.py`, wired in `KATVAD._stp_rows` |
| Text prefix | l = 8 vectors before the class text, `N(0, 0.02²)`, definitions only (§3.2.3) | `SoftPromptCLIPTextModel(layout="prefix")`, config `stp.text_prompt` |
| Monitoring | attention entropy, `W_q` from step 0, prefix norm (§3.3 stage 3) | `metrics.jsonl`: `stp_entropy` (÷ log K, 1 = uniform), `stp_wq_norm`, `prompt_norm` |
| Maps | `E_t(j) = a_{t,j}` on `S_t` | indices stored per video (`idx/`), maps not built yet |

Config (`core/config.py` `STPConfig`): `stp.sa2`, `stp.top_k`, `stp.motion` (`raw` | `ego`),
`stp.text_prompt` (`released` | `prefix`), `stp.prefix_len`. Every default is the pre-STP
model, so every earlier checkpoint builds and loads unchanged. Train/eval refuse a patch cache
whose stride, motion prior or K differ from the checkpoint's (`--patch-dir`).

An STP input row is the host row followed by the selected patches:
`[c_t (512) ; c·(u − m_u)/σ_u (384) ; P_{t,S_t} (12 × 512)]`.

## 2. Deviations from the proposal (user decisions or forced by the data)

- **D1. Motion fusion = `W_u`, not `tanh(α)·W·LN(v)`** (user, 2026-11-01). The host is v2 Nexar
  A2 / v5 R1: `x_t = c̃_t + W_u (c·ũ_t/σ_u)`, `W_u` zero-init, no LayerNorm, no tanh gate —
  the code that produced the 0.81 macro. SA² feeds `c̃_t` into that sum.
- **D2. Training = Nexar only** (`Nexar_window`, v2 §29), not Nexar + CCD. Same seeds, schedule
  and recipe as v5 R1: stage 2, 1,749 steps, `score_head_kernel` 3, `mil_topk_pct` 5,
  `dvs_anchor_mode=span`, `is_egocentric=false`, KIP off, no CRN.
- **D3. Evaluation = DoTA-CAP (n/1397) dev, protocol B** (`s1[::3]`, whole clip, native frames,
  macro over two-class clips), not LaGoVAD's 1,402-clip list. DoTA's own pixels are gone; only
  the CAP-DATA copies of 1,129 clips can be re-cut into window crops. Never put this number next
  to LaGoVAD's 62.60.
- **D4. Neighbours of Mo = ±1 cache row**: Nexar `s1[::8]` at ~30 fps (0.27 s), DoTA-CAP `s1[::3]`
  at 10 fps (0.30 s). The proposal's "±1 sampled step".
- **D5. K = 12, l = 8, LR multiplier 1 for the new parameters**, fixed. The proposal's one-time
  search on a held-out 15 % is not run in this first pass.
- **D6. "Prefix off" = LaGoVAD's released 32 split soft prompts** (what R1 trains with), so R4
  differs from R1 only by SA², and R6 from R4 only by the prefix.
- **D7. Control = v5 Stage 1 R1 s2024** (DoTA-CAP-dev 0.8102, read in v5 Stage 2). R1 computes
  its motion term from the V7 grid's tile mean; that equals the baked `u` the STP arms read to
  ≤ 1e-6 (PREREG_V5 S1-G2(b)), so the hosts are the same function. The notebook also asserts that
  each STP run's `config.yaml` equals v2 Nexar A2 s2024's except the `stp` section.
- **D8. Captions**: the prefix touches definitions only. The caption path (`L_neg`) does not run
  on Nexar (`captions_from_definitions: false`), as in every v2/v5 row.

## 3. First-run read-out rule (written before any STP number)

Arms (`constants.STP_ARMS`): **R4** = SA², released prompts (prefix OFF); **R6** = SA² + prefix
(prefix ON, full STP); **R5** = prefix only (optional, no patches).

- Endpoint per arm: per-clip Δ (arm − R1) of protocol-B AUC over the two-class CAP-dev clips,
  mean with a 95 % source-video cluster bootstrap (10,000 resamples).
- Prefix contrast: R6 − R4, the same way.
- Sentence: **"gain shown"** iff the low bound > 0 and the mean ≥ +0.01; **"cost shown"** iff the
  high bound < 0; otherwise **"no difference shown"**.
- With 1 seed every sentence carries **PRELIMINARY (n = 1 seed)**. It adopts nothing; adoption
  needs the 3-seed run (2024–2026) under the same rule.
- Printed, never deciding: Δ on ego-involved / ego-uninvolved clips (H1), the t/N ruler, v2's
  cubic position prior `p_T2` = 0.822, and the training diagnostics. An `stp_entropy` that stays at
  1.0 to the last step means SA² stayed "selection + average" (§3.2.4): the arm is still read, and
  reported as such.

## 4. How to run

Notebook `colab/final/stp_nexar_dotacap.ipynb` (patch extraction → train → protocol B → read-out).
DoTA-CAP pixels come from `Thesis-Final/data/DoTA_CAP_frames/` (`{dota_id}.zip`, DoTA-native frames,
original CAP JPEG bytes, all 1,129 clips), built **once** on CPU by `colab/final/dota_cap_pack_frames.ipynb`
(`core/tools/dota_cap_frames.py pack` + `verify`). `clip_windows archive` reads it with the same gate as
`clip_windows cap`, so no later tool streams the CAP tars.
Paste `REPORTS/stp_readout.md` back. Tests: `core/tests/test_stp.py`.

## 5. First-run diagnostics D-1 / D-2 (rules written 2026-11-02, before any diagnostic number)

**Why.** The first run (s2024, `outputs/v_final/stp/REPORTS/stp_readout.md`, 2026-11-02) read
**"no difference shown — PRELIMINARY"** on all three contrasts: R4 − R1 +0.0009 [−0.0093, +0.0106],
R6 − R1 +0.0022 [−0.0083, +0.0122], R6 − R4 +0.0013 [−0.0045, +0.0073]; `p_T2` 0.822 above every arm.
Two training logs make the contrasts hard to read: R6's prefix `prompt_norm` moved **1.2612 → 1.2616**
(init N(0, 0.02²) over 8 × 512 ⇒ ≈ 1.28), and SA² entropy ended at **0.91 / 0.90** of log K (≈ 9.6 of
12 patches effective). The printed H1 split went the wrong way: R4 − R1 **+0.012 [+0.001, +0.020]** on
ego-involved clips, **−0.013 [−0.031, +0.004]** on the others. Both diagnostics are printed only: they
change no §3 verdict and decide only what the next arm may be (each next arm gets its own rule first).
Tool `core/tools/stp_diag.py`, notebook `colab/final/stp_diag.ipynb` (CPU), tests
`core/tests/test_stp_diag.py`.

**D-1 — did the prompt train?** (`stp_diag prompt`, R4 and R6 s2024.) θ₀ is rebuilt the way
`core.train.main` builds it (`set_global_seed`, then `KATVAD.from_config`); it counts as reproduced iff
its norm equals the run's batch-0 `prompt_norm` to `STP_DIAG_INIT_NORM_RTOL` = 1e-4. Read from AdamW's
state: grad RMS √v̂ (per coordinate, bias-corrected), `SNR = mean|m̂| / mean √v̂` (zero-mean noise gives
`√((1 − β₁)/(1 + β₁))·√(2/π)` ≈ 0.184 at β₁ = 0.9), and the per-coordinate RMS displacement over the LR
budget `Σ lr_s` (1 = the full Adam step every step, same direction). Reading of the **R6 prefix** row,
first match wins:

| reading | condition | what it licenses next |
|---|---|---|
| INIT NOT REPRODUCED | norm check fails | Adam columns only; no displacement claim |
| **D1-dead** | ≥ 50 % of coordinates with √v̂ < 10·eps | a code fix of the gradient path (no LR change) |
| **D1-noise** | SNR < 2 × the noise SNR | "the objective gives the prefix no consistent signal"; an LR multiplier is **not** licensed (it would random-walk) |
| **D1-lr** | coherent, RMS Δ / Σ lr < 0.1 | an amendment with a prefix LR multiplier + a mechanics gate on the logs |
| **D1-moved** | otherwise | `prompt_norm` was a misleading monitor; R6 − R4 stands as measured |

`W_q` (SA²) and `W_u` rows are printed beside as references (both left zero and reached ‖·‖ ≈ 1.4).

**D-2 — do the top-12 motion patches point at the anomaly?** (`stp_diag pointing`, DoTA-CAP-dev
patch cache `idx/` + `mo/`, the window cache for the ego-compensated Mo, DoTA boxes.) Row k ↔ native
frame 3k; box masks by `v5_stage0.tile_weights` on the 7 × 7 grid of the 224² squash. Per clip:
(i) **lift** = share of the top-12 tiles touching a box − the box's tile share, on anomaly rows with a
box; (ii) **lift − pre** = lift minus the same read on pre-anomaly rows under the first anomaly mask
(Amendment 1's control for growing / centred boxes); (iii) **region excess** of the top-12 over all
rows: border ring − 24/49, bottom row − 7/49, centre 3 × 3 − 9/49; (iv) Mo's own v5 Stage 0 reads (ρ,
top-1 hit − chance, hit − pre, training-free macro of the frame-mean and top-12-mean Mo). Every value
is a per-clip mean with a 95 % source-video cluster bootstrap, all / ego-involved / not. Readings:

* **"selection points at the agent"** iff (ii)'s low bound > 0 on all clips;
* **"selection sits on the border"** iff the border excess's low bound > 0;
* the ego-compensated Mo (= arm R7's selection) is **worth an arm** only if its (ii) on
  non-ego-involved clips beats the raw Mo's by a margin whose cluster CIs do not overlap; otherwise
  R7 is not licensed by D-2.

Paste `REPORTS/stp_diag_prompt.md` and `REPORTS/stp_diag_pointing.md` back.

### 5.1 Read-out (2026-11-02, s2024, 1,749 steps; raw `outputs/v_final/stp/REPORTS/stp_diag_{prompt,pointing}.{md,json}`)

Rules in §5 applied as written; nothing below changes a §3 verdict. The first run stays
`no difference shown — PRELIMINARY (n = 1 seed)`.

**D-1 → D1-noise** (both R4 and R6; init reproduced, ‖θ₀‖ 126.421272 / 1.261225 = logged step 0).
R6 prefix: SNR **0.136** (< the 0.184 noise SNR), RMS Δ / Σ lr 0.016, grad RMS 9.3e-3 (≈ W_u's 9.3e-3),
share √v̂ < 10 eps 0. The prefix moved 0.045 (3.6 % of ‖θ₀‖, cos 0.9994), so `prompt_norm`
1.2612 → 1.2616 was a poor monitor, but the movement is not coherent. Reference rows sit near the
noise SNR too: SA² 0.20 / 0.22, **W_u 0.22** (a row known to matter in v2), R4 prompt 0.28.
* Licensed by the rule: **no prefix LR multiplier** (it would random-walk).
* **Not licensed:** "the objective gives the prefix no signal", "the prefix has no effect". The 2×
  noise-SNR threshold does not separate "no signal" from "small batch + slow drift" — W_u fails it
  too. Writable: *the prefix did not demonstrably train in 1,749 steps at 1 seed, so R6 − R4 does
  not measure it.*

**D-2 → "selection points at the agent": YES; "selection sits on the border": NO; R7 not licensed.**
DoTA-CAP-dev (ego-involved 321 / not 248), raw Mo, top-12: lift − pre-anomaly **+0.132 [+0.122, +0.142]**
(lift +0.196); border −0.080 [−0.087, −0.073], bottom row −0.082, centre 3×3 +0.052 — the selection
avoids the border (the border-tile hypothesis for the wrong-way H1 split is refuted). Non-ego
(ii): raw +0.1007 [+0.086, +0.115] vs ego-compensated +0.1004 [+0.086, +0.115] → CIs overlap → no R7.
Mo's own training-free reads: ρ +0.505, top-1 hit − chance +0.314, macro global 0.718 / top-K 0.713.
* **Printed, post-hoc, 1 seed (never a verdict):** (ii) is larger on ego-involved (+0.160
  [+0.144, +0.176]) than non-ego (+0.101 [+0.086, +0.115]) — the direction of the H1 split in
  R4 − R1 (+0.012 ego, −0.013 non-ego).
* **Reading:** the selected patches carry where-information, yet R4 − R1 ≈ +0.001. A plausible but
  untested cause is SA² staying ≈ selection + average (entropy ≈ 0.90 of log K); `p_T2` 0.822 also
  stays above every arm.

**Outcome:** neither follow-up arm (prefix LR multiplier, R7) is licensed. Closed with this record
(option A, user 2026-11-02). Any further arm (e.g. making SA² leave the average) needs its own rule
written before its number.
