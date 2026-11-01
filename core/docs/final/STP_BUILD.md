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
