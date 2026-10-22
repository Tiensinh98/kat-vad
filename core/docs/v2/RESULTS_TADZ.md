# KAT-VAD v2 — TAD results: zero-shot transfer (§26) and the clip-level gate (§27)

*2026-10-21 · branch `v2` · author: sinhpham (drafted with Claude)*

Two read-outs on TAD's official test protocol (`tad_test_anno.json`: 100 videos, 60 `Car Accident` + 40 `Normal`,
fixed traffic cameras), both written before their numbers:

- **§26 (Amendment 15, T1–T10): zero-shot transfer.** Do the T2-trained v2 checkpoints work on a fixed camera?
  Raw: `outputs/v2/v2_tadz/REPORTS/tadz_readout.{md,json}`, `tadz_frame_gate.json`. Tool `core/tools/tadz_readout.py`,
  runbook `colab/v2/tad_zeroshot.ipynb`, plan `.project/plans/katvad-v2-tad-zeroshot.md`.
- **§27 (Amendment 16, G1–G10): clip-level gate.** Can a gate fitted on MSAD repair §26's clip-level blindness?
  Raw: `outputs/v2/v2_tadg/REPORTS/tadg_readout.{md,json}`, `gate/{gate.npz,gate.json,c_*.json,dev_labels.json}`.
  Tool `core/tools/tad_clip_gate.py`, runbook `colab/v2/tad_clipgate.ipynb`, plan `.project/plans/katvad-v2-tad-clipgate.md`.

Neither moves `B` (T1, G1): **v2's training recipe stays A3-ign** and no DoTA number changes. Colab read-outs carry
`commit: UNKNOWN` by design. Both rules were authorized by the user and are not advisor-reviewed.

**Reading conventions.** *macro* = mean per-video frame AUC over the 60 two-class videos. *micro* = AUC over all
11,045 pooled frames. *clip-level AUC* = AUC of each video's max score against its video label. *t95* = paired t
interval over seeds 2024–2028 (t = 2.776). *[a, b]* for per-video reads = bootstrap with each video its own cluster,
B = 10,000. **Micro on TAD measures clip classification** (C14): the clip oracle scores 0.9226 and a length-only ruler
0.8968, so a TAD micro is never put beside the published 89.56 (C8b).

---

## 1. Verdicts

| Rule | Sentence (mechanical) | Numbers |
|---|---|---|
| §26 T5 endpoint | **"no transfer gain of v2's components is shown on TAD at n = 5"** | A3-ign − A0 macro +0.0218, t95 [−0.0032, +0.0467]; bootstrap +0.0327 [−0.0486, +0.1155] |
| §26 T6 (printed) | **"A3-ign and LaGoVAD's released checkpoint are not separated on TAD"** | +0.0015 [−0.0631, +0.0663]; not a matched comparison |
| §27 G5 dev gate | **PASS** | gate clip-level AUC on TAD-train 0.8051 ≥ 0.60 (length ruler 0.6541) |
| §27 G6 endpoint | **"a clip-level gate trained on MSAD restores clip-level discrimination on TAD"** | micro(s) − micro(A3-ign) +0.1361, t95 [+0.1323, +0.1398]; bootstrap +0.1569 [+0.1024, +0.2037] |

---

## 2. §26 — zero-shot transfer

**Protocol (T2–T4).** `core.evaluate --set data.dataset=TAD`, stride 8 (Gate T0's protocol; T2's s/step at an assumed
~30 fps, A-T1), whole video, raw scores, TAD's own definition. Arms (T3): the 15 deployed checkpoints, A0 / A3 from E3
and A3-ign from E4-0, all at step 1740. **No retraining; nothing tuned on TAD (T8).** A3 / A3-ign inputs: VideoMAE V2-S
extracted from the Kaggle frames (numeric-order symlink farm), CRN with each video's own R2 median, baked with the T2
statistics.

**Integrity (T9).** (a) T0 re-scores at micro **0.7912** (Gate T0); (b) 100 videos / 60 abnormal; (c) the Kaggle
frames are T0's frames: ⌈N/8⌉ == `TAD_ncc` rows on all 100, CLIP re-extracted on 5 videos matches at cos ≥ 0.99999;
(d) identical labels in every run; (e)–(f) J10 + arm config + bake manifest asserted by the notebook.

### 2.1 Levels (seed-averaged scores)

| Arm | macro [bootstrap] | per-seed macro (2024…2028) | clip-level AUC | micro |
|---|---|---|---:|---:|
| A0 (CLIP only) | 0.7268 [0.657, 0.793] | 0.720 / 0.738 / 0.725 / 0.709 / 0.730 | 0.578 | 0.493 |
| A3 (CRN + V2-S) | 0.7696 [0.712, 0.823] | 0.730 / 0.772 / 0.788 / 0.755 / 0.739 | 0.500 | 0.685 |
| **A3-ign** (`B`) | **0.7594 [0.707, 0.808]** | 0.740 / 0.751 / 0.768 / 0.749 / 0.723 | 0.544 | 0.678 |
| LaGoVAD released (T0) | 0.7579 [0.697, 0.814] | — | **0.769** | **0.791** |

Rulers (no pixels): `p_T2` macro 0.6885 [0.615, 0.758] · middle 0.683 · monotone `t/N` 0.587 · clip oracle micro
0.9226 · length ruler micro 0.8968 · clip-level length ruler (shorter = accident) 0.6940.

### 2.2 Contrasts (macro)

| Δ | paired t95 | per-video bootstrap | reading |
|---|---|---|---|
| **A3-ign − A0** (endpoint) | +0.0218 [−0.0032, +0.0467] | +0.0327 [−0.0486, +0.1155] | both contain 0 → no gain shown |
| A3 − A0 (printed) | +0.0324 [+0.0038, +0.0610] | +0.0429 [−0.0437, +0.1313] | bootstrap contains 0 → no sentence |
| A3-ign − A3 (printed) | −0.0107 [−0.0263, +0.0050] | −0.0102 [−0.0362, +0.0160] | the DoTA `ignore` gain does not carry over |
| A3-ign − LaGoVAD (printed) | — | +0.0015 [−0.0631, +0.0663] | not separated |

### 2.3 Post-hoc (printed, not gated; from the saved score files)

- **v2 is blind at clip level.** A3-ign's median max score is **1.000** on normal videos, the same as on accident
  videos; its mean on normal videos (0.352) is above the non-accident frames of accident videos (0.267). LaGoVAD keeps
  normals low (median max 0.013 vs 0.109).
- **The cause is the training data, not CRN.** A0, which has no CRN, is blind too (clip-level 0.578). T2 has no
  stand-alone normal video: its negatives are cut from inside accident videos, so the MIL objective learns *where* in a
  video the anomaly is, never *whether* a video has one. (An earlier draft blamed CRN; A0's number refutes that,
  lesson candidate (bp).)
- **Position does not explain TAD.** A3-ign − middle ruler +0.076 [−0.010, +0.163] (LaGoVAD +0.075); per-video AUC
  correlation with the ruler r = 0.01. Unlike DoTA-CAP, the position prior is below the model here.

---

## 3. §27 — the clip-level gate

**What it is.** `c(v) = σ(w · m̃(v) + b)`: logistic regression on the time-mean of a video's raw CLIP features
(`*_ncc`, no CRN), standardized; 513 parameters. It is trained separately from v2 (A3-ign is frozen), with
class-balanced binary cross-entropy + L2 (`C = 1.0`, lbfgs, max 2000 iterations), on **MSAD-train** (480 videos,
120 abnormal / 360 normal). MSAD was chosen because both classes come from the same cameras (no C38 source shortcut)
and `MSAD_ncc` is the same extractor as `TAD_ncc` (C2). **No TAD video is fitted on.** Mean pooling, not max, keeps it
length-blind by design.

**Fusion (G3).** `s(v, t) = c(v) · minmax_v(y_A3ign(v, t))`. Because `c > 0` rescales a whole video, every within-video
ranking is unchanged: **macro is exactly A3-ign's 0.7594** (checked per seed, |Δ| < 1e-9). Only the between-video
interleaving that micro reads can move. A3-ign localizes; the gate decides which video.

**Disclosure (§27).** The fusion form was first looked at post-hoc on TAD-test with LaGoVAD's scores as `c`
(c = mean T0: micro 0.752, clip 0.800). That look fixed the form; it fitted nothing. The gate is new.

### 3.1 Dev gate (G5) and sanity

| Read | gate clip-level AUC | length ruler |
|---|---:|---:|
| MSAD-train (fitted on) | 1.000 | — |
| MSAD-test (in-domain sanity) | 0.948 | — |
| **TAD-train** (410 videos, 200 abnormal; never fitted on) | **0.805** → PASS (bar 0.60) | 0.654 |
| TAD-test (printed) | 0.851 | 0.694 |

TAD-train labels from the id prefix (`Normal_*`), which reproduces the 100 test labels exactly (G9(c)).

### 3.2 Endpoint (G6) — TAD-test

| | macro | micro | clip-level AUC |
|---|---:|---:|---:|
| A3-ign (§26) | 0.7594 | 0.6779 | 0.5440 |
| **A3-ign × MSAD gate** | **0.7594** | **0.8347** | **0.8512** |
| LaGoVAD released (T0) | 0.7579 | 0.7912 | 0.7687 |
| A3-ign × mean(T0) (form seen on this set) | 0.7594 | 0.7524 | 0.8000 |

- micro(s) − micro(A3-ign): paired t95 **+0.1361 [+0.1323, +0.1398]** (per seed +0.136 / +0.141 / +0.134 / +0.136 /
  +0.133; fused per-seed micro 0.816 / 0.805 / 0.827 / 0.810 / 0.794); video bootstrap **+0.1569 [+0.1024, +0.2037]**.
- micro(s) − micro(LaGoVAD), printed: +0.0435 [−0.0313, +0.1294] → **not separated**, and not matched (T6).
- micro 0.835 is still below the length ruler (0.897) and the clip oracle (0.923).

### 3.3 Post-hoc: is the gate just reading video length? (printed, not gated)

The gate correlates with length (Spearman(c, L) −0.53 on both TAD splits; within normals −0.50 / −0.60), so this had
to be checked:

| | TAD-train | TAD-test |
|---|---:|---:|
| gate logit, clip-level AUC | 0.805 | 0.851 |
| length only (shorter = accident) | 0.654 | 0.694 |
| **gate residualized on log length** (linear fit on TAD-train) | **0.755** | **0.798** |
| within length terciles (edges from TAD-train) | 0.64 / 0.84 / 0.86 | 0.75 / 0.97 / 0.88 |

The gate carries signal beyond length. What is **not** ruled out: that it separates TAD's accident and normal footage
partly by recording style (compiled accident clips vs continuous normal recordings); TAD has no source metadata to test it.

---

## 4. What may and may not be claimed

| ✅ May say | ❌ May not say |
|---|---|
| "Zero-shot on fixed-camera TAD, A3-ign (macro 0.759) is not separated from LaGoVAD's released checkpoint (0.758)", with the training-data caveat | "v2's components transfer to TAD"; "v2 beats LaGoVAD on TAD" |
| "Trained on T2 alone, v2 cannot separate normal from accident videos (clip-level AUC 0.544); A0 cannot either (0.578)" | that CRN causes the clip-level blindness |
| "A clip-level gate fitted on MSAD (no TAD video) restores clip-level discrimination on TAD: micro 0.678 → 0.835, Δ +0.136 [+0.132, +0.140]; clip-level AUC 0.544 → 0.851; macro unchanged" | "the gate beats LaGoVAD" (+0.044, contains 0); any TAD micro beside 89.56; micro 0.835 without the length ruler 0.897 beside it |
| "the gate carries signal beyond video length" (residual 0.798) | "the gate detects car accidents" (it was trained on 11 mostly non-traffic MSAD classes; recording style not ruled out); `c` read as a calibrated probability |
| The deployed system on TAD is **two components**: A3-ign + an offline video-level gate | that it runs streaming (the gate needs the whole video's mean) |

## 5. Consequences

- **Recipe:** unchanged. `B` = A3-ign; the gate is an inference-time add-on, not a v2 training change.
- **DoTA:** unchanged by construction (macro invariant; DoTA-CAP is ~all-abnormal).
- **Report:** the TAD limitation of `REPORT_V2_RESULTS.md` §4.6 now has a measured repair (§4.7 there).
- **Open:** a causal (streaming) gate on a running mean; a gate trained on traffic-only normals; the recording-style
  confound. None is planned.
