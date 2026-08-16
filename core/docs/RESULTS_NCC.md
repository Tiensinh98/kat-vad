# `no_center_crop` rebuild — results across MSAD and DoTA

**Measured:** 2026-08-12, from the saved score curves in
`outputs/{MSAD_ncc,DoTA_ncc}/*/scores/*.npz`.
**Supersedes** the KIP verdict in `RESULTS_DOTA.md` §5–6, which was measured on
center-cropped features.

> **Partially superseded 2026-08-16 by `RESULTS_PHASE_A.md`.** Two claims below
> are now wrong:
>
> - **§4 "the MSAD reproduction gate no longer passes"** — wrong reference. The
>   released `best.ckpt` scores only 0.8991 (crop) / 0.8949 (ncc) through our own
>   eval, so it cannot reach the published 0.9041 either. Re-gated against the
>   checkpoint, all eight paired bootstraps include zero and the gate **passes**
>   (lesson 8b).
> - **§6 "Not established: 1. Reseeding, n = 1"** — now established. Seeds 2025
>   and 2026 replicate: Δ = +0.0915 ± 0.0088, nine CIs excluding zero. H2 rejected.
>
> Everything else here stands, including the D4 mechanism failure, which
> replicated three for three.

**One line:** removing the center crop flipped KIP's DoTA effect from
**−0.033 to +0.091** AUC, and it did so almost entirely through the KIP arm —
the baseline arms barely moved. Lesson C13 predicted this on theory; this is the
measurement. It costs ~2 pp of in-domain MSAD AUC and both mechanism checks came
back negative.

---

## 1. The grid

Four training runs: 2 transforms × KIP {on, off}. Same seed (2024), same
hyperparameters, same data split, same flow cache. Config diff between the
center-crop and `_ncc` runs is empty once paths are normalised — **the transform
is the only variable.**

### MSAD-full — in-domain, 240 test videos, raw pooling

| transform | KIP-off | KIP-on | Δ(on − off) |
|---|---|---|---|
| center-crop | 0.9052 / AP 0.7249 | 0.9064 / 0.7334 | +0.0012, CI [−0.0060, +0.0090] — null |
| `no_center_crop` | 0.8922 / 0.6587 | 0.8868 / 0.6574 | −0.0054, CI [−0.0155, +0.0033] — null |

Reference: `gate_a` (LaGoVAD released `best.ckpt`) 0.8991 → 0.8949.
LaGoVAD published 0.9041.

### DoTA — zero-shot, 1,397 clips (1,392 scorable), stride 8, per-clip min-max

| transform | KIP-off | KIP-on | Δ(on − off) |
|---|---|---|---|
| center-crop | 0.5539 (macro 0.5561) | 0.5215 (0.5232) | **−0.0324** |
| `no_center_crop` | 0.5607 (0.5638) | **0.6519 (0.6746)** | **+0.0911**, CI [+0.0799, +0.1019] |

`gate_a`: 0.6142 (macro 0.6328) → 0.6012 (0.6158). LaGoVAD published 0.6260.

**D3 verdict, `_ncc`:** micro Δ +0.0911 CI [+0.0799, +0.1019]; macro Δ +0.1108
CI [+0.0974, +0.1237]; AP Δ +0.0792 CI [+0.0684, +0.0895]. CI excludes zero,
mean is 6× the +0.015 bar. This is the top row of `DOTA_EVAL.md` §D3.

KIP-on also beats LaGoVAD's own released checkpoint: micro Δ +0.0508,
CI [+0.0355, +0.0659].

---

## 2. The crop removal worked *through* KIP, not around it

This is the load-bearing result. Same comparison, per arm, crop → `_ncc`:

| arm | Δ macro AUC (DoTA) | 95 % CI |
|---|---|---|
| **KIP-on** | **+0.1514** | [+0.1354, +0.1683] |
| KIP-off | +0.0077 | [−0.0065, +0.0225] — includes 0 |
| `gate_a` | −0.0170 | [−0.0301, −0.0037] — **negative** |

Only the arm that consumes motion moved. Lesson C13's mechanism —
`preprocess_for_raft` is full-frame while `preprocess_frames` center-cropped, so
KIP was regressing motion evidence cropped out of its own input — is now
measured, not argued.

Supporting detail: 349 of 1,402 DoTA clips are `ego/other: lateral`, i.e. the
evidence lives at the frame periphery that the crop discarded.

---

## 3. What it costs

| | Δ AUC | Δ AP |
|---|---|---|
| MSAD KIP-on, crop → ncc | −0.0196, CI [−0.0324, −0.0065] | −0.0761, CI [−0.1128, −0.0409] |
| MSAD KIP-off, crop → ncc | −0.0130, CI [−0.0260, +0.0003] | −0.0662, CI [−0.1021, −0.0320] |

Both AP drops are significant. Per-video, KIP-on loses 55 and wins 31, worst in
`People_falling` (−0.149), `Shooting` (−0.127), `Assault` (−0.112).

**The MSAD reproduction gate no longer passes.** `_ncc` KIP-off is 0.8922
against LaGoVAD's published 0.9041. The trade is ~2 pp in-domain AUC and ~7 pp
AP for ~9 pp out-of-domain AUC.

### 3.1 The transform-parity argument did not survive

The 2026-08-08 decision rested on two legs: KIP/RAFT field-of-view consistency,
and parity with the baseline's own `augmentation='no_center_crop'`. The
field-of-view leg is confirmed (§2). **The parity leg is refuted**: the released
`best.ckpt` got *worse* under `_ncc` on both benchmarks (DoTA macro −0.0170, CI
excludes zero; MSAD −0.0042, ns). If that checkpoint had been trained on
`no_center_crop` features, `_ncc` should have suited it better.

Either it was not, or our `_ncc` resize differs from the baseline's in some other
respect. This does not touch the KIP result — that is an A/B at fixed features —
but the parity claim should not be repeated.

---

## 4. Both mechanism checks came back negative

### D4 — the gain is not ego-concentrated

| group | KIP-off | KIP-on | Δ | 95 % CI |
|---|---|---|---|---|
| `ego:` (n = 802) | 0.5763 | 0.6732 | +0.0969 | [+0.0785, +0.1143] |
| `other:` (n = 590) | 0.5468 | 0.6765 | **+0.1296** | [+0.1098, +0.1497] |

`ego < other`, so per `DOTA_EVAL.md` §D4 row 3: **do not claim a kinematic
mechanism from this split.** Top classes are `other: leave_to_right` (+0.187),
`other: oncoming` (+0.165), `other: turning` (+0.144) — peripheral,
other-vehicle events. That is consistent with field-of-view restoration, which
is a real mechanism, just not ego-kinematics.

### D5 — the one live MSAD thread did not replicate

Multi-class frame accuracy on anomalous frames (n = 4,220):

| transform | KIP-off | KIP-on | Δ |
|---|---|---|---|
| center-crop | 0.4754 | 0.5116 | **+3.6 pp** |
| `no_center_crop` | 0.4758 | **0.4699** | **−0.6 pp** |

`RESULTS_MSAD.md` §3.5's strongest signal evaporated in exactly the run where
the DoTA gain appeared. **They are not one mechanism.** Do not write the
coherent-story paragraph §D5 anticipated.

---

## 5. Saturation — a caveat, not a defect

Both `_ncc` arms are extreme on DoTA: **89.1 %** (off) / **86.6 %** (on) of
frames score above 0.99; within-clip dynamic range median 1.8e-4 / 7.4e-4.
`gate_a` by contrast sits at mean 0.221 with range 0.038.

AUC is rank-based and per-clip min-max is monotone, so **macro AUC is
unaffected**. The one way saturation could fake this is float32 ties, so that
was checked directly:

| run | tied frames | clips with any tie |
|---|---|---|
| `_ncc` KIP-off | 270 / 18,369 (1.47 %) | 210 / 1,397 |
| `_ncc` KIP-on | 93 / 18,369 (0.51 %) | 84 / 1,397 |
| `_ncc` `gate_a` | 1 / 18,369 (0.01 %) | 1 / 1,397 |

~0.2 tied frames per clip. Far too few to manufacture a +0.11 macro delta.
**The effect is not a numerical artifact.**

What it does mean: the models are far outside their calibrated range, every
threshold-based metric is meaningless here, and no single absolute score from
these arms should be quoted on its own.

---

## 6. What this does and does not establish

**Established:** on `_ncc` features, at seed 2024, `checkpoint_last`, KIP-on
outperforms KIP-off on DoTA by a margin whose bootstrap CI is nowhere near zero,
and the gain is specific to the KIP arm.

**Not established:**

1. **Reseeding.** n = 1. MSAD's own KIP Δ moved +0.0012 → −0.0054 between
   transforms, so the per-run noise floor is not known to be small.
2. **The mechanism.** D4 and D5 both failed. The result is real; the *reason*
   given in the proposal is unsupported by these two checks.
3. **H3 — under-convergence.** The KIP-on model is less fitted to MSAD (final
   train `mil` 0.010 vs 0.0016; 21.2 % vs 24.5 % of MSAD frames above 0.99).
   Worse in-domain / better out-of-domain is the signature of a regularizer, not
   necessarily a motion module. Nothing here distinguishes the two.
4. **Warm-start asymmetry.** KIP-on warm-starts from stage 1, KIP-off starts
   cold, and stage-1 gradients reach the shared temporal encoder. The A/B is
   "KIP + warm start" vs "no KIP, cold".

Items 1 and 3 are what `.project/plans/msad-ncc-seeds-and-selection.md` is for
(Phases A and B); item 4 is its §4.

**Do not change KIP's architecture, losses, or hyperparameters on this result.**
It is a measurement with three open confounds.

---

## 7. Reproducing this document

All numbers come from the saved `.npz` score curves — no re-inference needed.
Load `score` and `gt` per clip, apply per-clip min-max for DoTA and none for
MSAD (or let `core.tools.rescore` do it), then:

- micro AUC = `roc_auc_score` over concatenated frames
- macro AUC = mean of per-clip `roc_auc_score`, skipping single-label clips
- CIs = paired bootstrap resampling **clips** (not frames), 1,000–4,000 draws,
  both arms scored on the same resample
- D4 groups come from `data/DoTA/metadata_val.json` → `anomaly_class`, split on
  the `ego:` / `other:` prefix
- D5 = `argmax` over the `sim` array on frames with `gt > 0.5`, compared against
  the class parsed from the video id
