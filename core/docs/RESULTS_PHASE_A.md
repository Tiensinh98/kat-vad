# Phase A — seed repeat (2024 / 2025 / 2026)

**Measured:** 2026-08-16, from the saved score curves in
`outputs/{MSAD_ncc,MSAD_ncc_s2025,MSAD_ncc_s2026,DoTA_ncc,DoTA_ncc_s2025,DoTA_ncc_s2026}/*/scores/*.npz`.
**Plan:** `.project/plans/msad-ncc-seeds-and-selection.md` §2.
**Extends** `RESULTS_NCC.md`, which is the same measurement at n = 1 seed.

**One line:** the DoTA gain replicates across all three seeds — Δ = **+0.0916 ±
0.0088** micro AUC, every bootstrap CI excluding zero — while MSAD stays null.
**H1 holds; H2 (seed luck) is rejected.** H3 and the warm-start confound remain
open, and the MSAD reproduction gate is redefined below.

> **Updated 2026-08-19 — §6's two open confounds are now closed.**
> `RESULTS_ARM4_PROBE.md` adds a fourth arm (KIP-off warm-started) and a
> trajectory probe: Δ(on − off_warm) = **+0.0988 ± 0.0148**, nine CIs excluding
> zero, so the warm start is not the mechanism; and KIP-off's DoTA AUC is flat
> at 0.559–0.561 across a 33× range of train `mil`, so **H3 is rejected**.
> Nothing in this document is superseded — the numbers below stand as measured.

---

## 1. What ran

Three seeds × {stage 1, stage 2 KIP-on, stage 2 KIP-off}. Seed 2024 was already
complete (`RESULTS_NCC.md`); 2025 and 2026 were added.

**Parity verified:** `config.yaml` for all nine runs is byte-identical across
seeds once Drive paths and `train.seed` are normalised. The data split, CLIP
cache, flow cache and KNN cache are shared and were not rebuilt. **The seed is
the only variable.**

Coverage is identical across every arm: MSAD 240/240 test videos (18,350 sampled
frames, 96 scorable for macro); DoTA 1,397 clips (18,369 frames, 1,392 scorable
for macro).

Runbooks: `collab/MSAD_ncc_s2025_2026/train.py`,
`collab/DoTA_ncc/evaluate_s2025_s2026.py`.

---

## 2. DoTA — zero-shot, per-clip min-max

| seed | metric | KIP-off | KIP-on | Δ(on − off) | 95 % CI |
|---|---|---|---|---|---|
| 2024 | micro AUC | 0.5609 | 0.6520 | **+0.0911** | [+0.0797, +0.1021] |
| | AP | 0.3516 | 0.4308 | +0.0792 | [+0.0685, +0.0896] |
| | macro AUC | 0.5638 | 0.6746 | +0.1108 | [+0.0977, +0.1244] |
| 2025 | micro AUC | 0.5587 | 0.6416 | **+0.0829** | [+0.0725, +0.0945] |
| | AP | 0.3453 | 0.4165 | +0.0712 | [+0.0622, +0.0810] |
| | macro AUC | 0.5532 | 0.6550 | +0.1018 | [+0.0889, +0.1156] |
| 2026 | micro AUC | 0.5284 | 0.6289 | **+0.1004** | [+0.0886, +0.1124] |
| | AP | 0.3247 | 0.4035 | +0.0788 | [+0.0690, +0.0885] |
| | macro AUC | 0.5190 | 0.6291 | +0.1101 | [+0.0957, +0.1240] |

**Across seeds:** micro Δ mean **+0.0915**, range [+0.0829, +0.1004], spread
±0.0088. Macro Δ mean +0.1076. AP Δ mean +0.0764.

Nine intervals, nine exclusions of zero, all three micro CIs mutually
overlapping — consistent with one underlying effect rather than three draws.
Plan §2.4's top row ("all three Δ positive, min > +0.03") is met with the
minimum at **+0.0829**, ~2.8× the bar.

**Both arms drift down together across seeds** (off 0.5609 → 0.5587 → 0.5284;
on 0.6520 → 0.6416 → 0.6289). The *level* is seed-sensitive; the *gap* is not.
This is why the paired Δ, not either arm's absolute number, is the reportable
quantity.

---

## 3. MSAD-full — in-domain, raw pooling

| seed | KIP-off | KIP-on | Δ AUC | 95 % CI | Δ AP | Δ macro |
|---|---|---|---|---|---|---|
| 2024 | 0.8922 | 0.8868 | −0.0054 | [−0.0154, +0.0036] | −0.0013 | −0.0055 |
| 2025 | 0.8857 | 0.8862 | +0.0005 | [−0.0066, +0.0073] | −0.0063 | +0.0372 |
| 2026 | 0.8824 | 0.8815 | −0.0008 | [−0.0096, +0.0082] | −0.0114 | +0.0326 |
| mean | 0.8868 | 0.8848 | **−0.0019** | — | −0.0063 | +0.0214 |

Micro AUC and AP are **null in all three seeds** — every CI includes zero, and
the sign is not even stable. KIP costs nothing in-domain and buys nothing.

**This asymmetry is the finding**, exactly as plan §2.4 anticipated: the motion
pathway pays where motion *is* the anomaly (DoTA, dashcam collisions) and is
inert where it is not (MSAD, a mixed-scenario surveillance set). That is a
sharper claim than a uniform win, because it names the condition.

### 3.1 A weaker signal, stated with its caveat

KIP-on's MSAD **macro** AUC is markedly more seed-stable than KIP-off's:

| | 2024 | 2025 | 2026 | spread |
|---|---|---|---|---|
| KIP-off macro | 0.7177 | 0.6785 | 0.6773 | 0.040 |
| KIP-on macro | 0.7122 | 0.7158 | 0.7099 | **0.006** |

Suggestive of a variance-reduction effect. But the per-seed Δ CIs are
[−0.0399, +0.0298], [−0.0016, +0.0799] and [+0.0005, +0.0665] — only the last
excludes zero, and barely. **At n = 3 this is an observation, not a result.**
Do not build an argument on it without more seeds.

---

## 4. The MSAD reproduction gate, redefined

The published LaGoVAD MSAD number is **0.9041**. Our `_ncc` arms land at
0.8815–0.8922, and `RESULTS_NCC.md` recorded the gate as failing on that basis.

**That was the wrong comparison.** LaGoVAD's own released `best.ckpt`, run
through our eval on the same features, does not reach 0.9041 either:

| transform | released `best.ckpt` | our KIP-off | published |
|---|---|---|---|
| center-crop | 0.8991 | 0.9052 | 0.9041 |
| `no_center_crop` | **0.8949** | 0.8922 | 0.9041 |

Paired bootstrap, every arm against the released checkpoint on the identical
protocol, 240 videos:

| comparison | Δ AUC | 95 % CI | Δ AP |
|---|---|---|---|
| crop, KIP-off − `best.ckpt` | +0.0061 | [−0.0201, +0.0304] | +0.0437 |
| crop, KIP-on − `best.ckpt` | +0.0073 | [−0.0171, +0.0308] | +0.0523 |
| ncc s2024, KIP-off − `best.ckpt` | −0.0027 | [−0.0246, +0.0197] | +0.0155 |
| ncc s2024, KIP-on − `best.ckpt` | −0.0081 | [−0.0306, +0.0139] | +0.0142 |
| ncc s2025, KIP-off − `best.ckpt` | −0.0092 | [−0.0335, +0.0143] | +0.0099 |
| ncc s2025, KIP-on − `best.ckpt` | −0.0087 | [−0.0312, +0.0133] | +0.0037 |
| ncc s2026, KIP-off − `best.ckpt` | −0.0125 | [−0.0374, +0.0108] | +0.0136 |
| ncc s2026, KIP-on − `best.ckpt` | −0.0134 | [−0.0377, +0.0101] | +0.0022 |

**Every CI includes zero, on both transforms, in all three seeds. AP is
consistently higher for our arms.** Our baseline port is statistically
indistinguishable from the checkpoint the authors shipped.

The 0.9041 is therefore **not a reachable gate with the published artifacts** —
whatever produced it (a different checkpoint, a different stride, a different
frame count, or an unreleased selection) is not in our hands. Note also that our
AUC is over 18,350 stride-8 sampled frames, not the paper's full frame count.

**Gate redefined:** the MSAD reproduction gate is *"our KIP-off arm matches
LaGoVAD's released `best.ckpt` under one identical protocol."* By that
definition it **passes**, everywhere measured. Quote 0.9041 only as the paper's
printed number, never as a target our tree failed to hit.

This generalises to lesson **8b** (an amendment to lesson 8, which as written had
no exit condition for a published number the released weights cannot reach).

---

## 5. Mechanism checks

### D4 — ego vs other (replicates; the ego claim is refuted)

Per-clip macro AUC Δ, split on `metadata_val.json` → `anomaly_class` prefix:

| seed | ego (n=802) | other (n=590) |
|---|---|---|
| 2024 | +0.0969 | **+0.1296** |
| 2025 | +0.0734 | **+0.1403** |
| 2026 | +0.0873 | **+0.1411** |

`other > ego` in **all three seeds**. At n = 1 this was a failed check; at n = 3
it is a settled negative. **The proposal's ego-kinematics story is refuted** —
do not claim KIP works by modelling the ego-vehicle's own motion.

Both groups gain strongly, though, so the effect is not one subgroup carrying
the average. Whatever KIP is doing, it helps on third-party motion *more*.

### H3 — under-convergence is still fully alive

Final stage-2 train `mil`:

| seed | KIP-on | KIP-off | ratio |
|---|---|---|---|
| 2024 | 0.0100 | 0.0016 | 6.3× |
| 2025 | 0.0065 | 0.0019 | 3.5× |
| 2026 | 0.0046 | 0.0008 | 5.5× |

KIP-on is **consistently less fitted to MSAD in every seed**. Worse in-domain /
better out-of-domain remains the signature of a regularizer as much as of a
motion module. Phase A does not touch this. See §6.

---

## 6. What Phase A establishes, and what is still open

**Established**

1. The DoTA gain is **not seed luck**. Three seeds, nine CIs, all excluding zero,
   min Δ +0.0829. H2 rejected.
2. The MSAD cost is **not real** — null in micro AUC and AP across all seeds.
3. Our baseline port **matches the released LaGoVAD checkpoint** on MSAD under
   an identical protocol (§4). The reproduction gate passes as redefined.
4. The ego-kinematics mechanism is **refuted**, three for three (§5).

**Still open**

1. ~~**H3 — under-convergence transfers.**~~ **CLOSED 2026-08-19 — rejected.**
   At train `mil` matched to KIP-on's endpoint, KIP-off reaches DoTA ≈ 0.560
   against KIP-on's 0.6519, and its transfer is flat over a 33× `mil` range.
   `RESULTS_ARM4_PROBE.md` §4.
2. ~~**Warm-start asymmetry.**~~ **CLOSED 2026-08-19 — not the mechanism.**
   Δ(on − off_warm) = +0.0988 ± 0.0148 over three seeds, nine CIs excluding
   zero, no shrinkage against the cold Δ; stage-1 trunk pretraining alone moves
   DoTA by ±0.02 with a seed-dependent sign. `RESULTS_ARM4_PROBE.md` §2.
3. **The mechanism generally.** D4 is refuted; D5 (MSAD multi-class accuracy)
   did not replicate under `_ncc` at seed 2024 and was not re-checked here. The
   result is real and its *reason* is unknown. **This is now the only open
   scientific question** — items 1 and 2 eliminated two rival explanations
   without confirming one.

Items 1 and 2 were what the next cycle was for — run, and reported in
`core/docs/RESULTS_ARM4_PROBE.md` (runbook: `COLAB.md` § "Arm 4 + trajectory
probe"). `.project/plans/msad-ncc-seeds-and-selection.md` §3 (Phase B) stays
deferred; A5 answered its question for ~1 % of the cost.

**Do not change KIP's architecture, losses, or hyperparameters on this result**
(lesson 14). Two confounds are open.

---

## 7. Reproducing this document

All numbers come from the saved `.npz` score curves — no re-inference. Method is
identical to `RESULTS_NCC.md` §7:

- micro AUC = `roc_auc_score` over concatenated frames; per-clip min-max for
  DoTA, raw for MSAD (`--score-norm auto` resolves this from the label
  distribution — lesson 12)
- macro AUC = mean per-clip `roc_auc_score`, skipping single-label clips
- CIs = paired bootstrap resampling **clips**, 2,000 draws, both arms scored on
  the same resample
- §4 uses the same bootstrap with `outputs/{MSAD,MSAD_ncc}/full_gate_a` as the
  reference arm
- D4 groups from `data/DoTA/metadata_val.json` → `anomaly_class`, split on the
  `ego:` / `other:` prefix

---

## 8. Defect found while auditing the runbook

`collab/DoTA_ncc/evaluate_s2025_s2026.py` lines 113–153 carry trailing per-seed
cells that reference `$EXTRA`, which is **unset in a fresh `%%bash` cell**. The
s2026 KIP-off cell would therefore have evaluated a KIP-off checkpoint with
`kip.enabled=true`.

**It did not corrupt these results.** `core/inference.py:64` calls
`model.load_state_dict(state)` at default `strict=True`, so the missing KIP keys
raise rather than half-loading (lesson 5 earning its keep). The reported numbers
came from the correct loop at lines 100–110, which sets `EXTRA` inline.

Delete those dead cells before anyone re-runs the notebook.
