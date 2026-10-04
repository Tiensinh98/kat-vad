# v2 P4 — E2(d): encoder choice on DoTA-CAP-dev, result (2026-10-04)

Tool `python -m core.tools.e2d_probe`, runbook `colab/v2/p4_e2d.ipynb`. Rules: proposal §10.2 E2(d) as
pre-registered in `PREREG_ADDENDUM.md` §12 (N1–N12, `7b1889a`) on the endpoint of Amendment 4 (D13: DoTA-CAP),
all written before any VideoMAE feature of a DoTA clip was probed. Read-outs
`Thesis-V2/outputs/REPORTS/v2_E2d/e2d_readout.{json,md}` and `v2_A0_dev_protocolB/` (gitignored; this file is the
record). No training: frozen-feature logistic probes. Every number below is **DoTA-CAP (569/1397)** (D14) and is
never put beside a full-DoTA number or LaGoVAD's 62.60.

**Set-up.** `dota_cap_dev` = 569 clips (568 two-class), protocol B rows (`s1[::3]`, whole clip), 91 source-video
clusters, cluster bootstrap B = 10,000. In-domain probe = 5-fold CV grouped by source video (N3); transfer probe
= fitted on the 295 frozen K sources (T2-train, whole, s8; sha1 `634a105d7175`), scored on `dota_cap_dev` (N4).
A2 = `[x;u]` vs `x`; A3 = the same after R2 (per-clip median) CRN on both streams (N2).

## Verdict (mechanical, N6/N7): **V2-S** (`vit_s_k710_dl_from_giant`)

| Encoder | Arm | In-domain Δ ([x;u] − x) | Transfer Δ | Eligible |
|---|---|---|---|---|
| V2-B | A2 | +0.162 [+0.144, +0.180] | +0.194 [+0.177, +0.211] | yes |
| V2-B | A3 | +0.134 [+0.116, +0.153] | **+0.151** [+0.137, +0.165] | yes |
| V2-S | A2 | +0.149 [+0.134, +0.165] | +0.158 [+0.142, +0.175] | yes |
| V2-S | A3 | +0.125 [+0.110, +0.142] | **+0.132** [+0.118, +0.147] | yes |

Both encoders pass N6 on every row (thresholds: in-domain ≥ 0.10 or transfer ≥ 0.03). N7 ranks on the A3 transfer
Δ: B 0.1508, S 0.1325, gap **0.0183 < 0.02** → the cheaper encoder, **S** (384-d, `W_u` ≈ 0.20 M). The motion
stream is **not dropped**; A2/A3 stay in E3.

## What the numbers say (printed, N9–N10; none of this moves the verdict)

1. **Position alone beats everything on these clips.** The cubic position probe `p` (`τ, τ², τ³`, no pixels)
   reads **0.852** [0.837, 0.867] in-domain and **0.847** [0.831, 0.863] in transfer, above `u` alone
   (0.76–0.82) and above `[x;u]` (0.73–0.80). DoTA accidents sit at a consistent relative position (mid-clip),
   and a probe fitted on whole T2 sources learns the same prior. E1's `t/N` ruler read **0.566** on the same
   clips because it is monotone: it cannot draw a hump. **The `t/N` ruler understated position by ~0.29.**
2. **Beyond position, `u` adds ~nothing in-domain and a little in transfer** (`[x;u;p]` − `[x;p]`):

   | | B A2 | B A3 | S A2 | S A3 |
   |---|---|---|---|---|
   | in-domain | −0.003 [−0.015, +0.008] | −0.006 [−0.016, +0.004] | +0.004 [−0.005, +0.014] | +0.006 [−0.002, +0.014] |
   | transfer | +0.037 [+0.023, +0.050] | +0.029 [+0.018, +0.042] | +0.018 [+0.005, +0.030] | +0.026 [+0.015, +0.037] |

   The transfer rows agree with K-pos on T2 (+0.043 [+0.023, +0.064]): a few AUC points of motion signal survive
   position control, the +0.15 headline does not. **Caveat in both directions:** where accidents sit at a fixed
   relative position, real motion evidence and position are collinear, so this control *under*-credits `u`; it
   does not show `u` is useless. It shows DoTA-CAP-dev cannot separate the two by probing.
3. **`[x;p]` and `[x;u;p]` both read below `p` alone in-domain** (0.832–0.834 and 0.829–0.840 vs 0.852): 512+
   feature channels next to three position features cost the probe more than they give, so "beyond position"
   here is measured against a weakened base.
   Printed for the record; same probe for every row, so the contrasts stay paired.
4. **Ranked on position-controlled transfer, B is ahead of S** (A2 +0.037 vs +0.018), and on the T2 side
   (source-grouped CV on the K sources: B +0.137 [+0.114, +0.160], S +0.107 [+0.086, +0.127]). N7 does not rank
   on these and was committed before them; switching now would be a rule chosen after the number (lesson 14).
   Recorded as a limitation (Amendment 6, G1).
5. **DoTA-CAP is a fair stand-in for DoTA-dev (D15).** CLIP-only probe `x`: dev 0.638, CAP 0.636, dropped 0.649;
   `x_crn`: 0.677 / 0.673 / 0.696; A0 (E1's three KIP-off checkpoints, protocol B): **0.6628 / 0.6591 / 0.6787**.
   CAP is ~0.01–0.02 harder than the 133 dropped clips, every interval overlaps; median length 99 vs 105 native
   frames, accident share 0.303 vs 0.318, same top categories.
6. **The new protocol-B scorer reproduces E1 exactly.** `protocol_b_eval` on the three E1 checkpoints gives
   macro 0.6628 [0.6436, 0.6823] and per-seed 0.6291 / 0.6705 / 0.6457 — E1's arm B to four decimals. A motion
   arm scored by this path is on E1's ruler.

## What follows (Amendment 6, `PREREG_ADDENDUM.md` §15, written before any of these numbers)

* **G2** full-T2 V2-S extraction → **G4** bake A2/A3 → pilot batch 2 (`colab/v2/p4_s_full_t2_d6.ipynb`,
  `colab/v2/p6_pilot_batch2.ipynb`).
* **G3 / D6** temporal-shuffle control for S on `dota_cap_dev` (`core.tools.e2d_shuffle`): if ordered − shuffled
  on `[x;u;p]` has a CI containing 0, the thesis may not call the stream's DoTA-CAP gain "motion".
* **G6 / G7** every E3 DoTA-CAP macro is printed beside a cubic position probe fitted on T2 windows and with the
  per-arm score–position correlation; `t/N` is kept only as a labelled monotone ruler.
