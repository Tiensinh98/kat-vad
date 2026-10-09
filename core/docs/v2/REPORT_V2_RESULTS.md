# KAT-VAD v2 — Summary of experimental results (P0 → Final) and directions for improvement

*2026-10-07 · branch `v2` · code `472adb0` · author: sinhpham (drafted with Claude)*

**Sources.** Every number here comes from an existing read-out: `MOTION_STREAM_K.md`, `RESULTS_E2_CRN.md`,
`RESULTS_E1.md`, `RESULTS_E2D.md`, `RESULTS_P6_PILOT.md`, `RESULTS_E3.md` and
`outputs/v2/v2_final/REPORTS/final_readout.{md,json}`. `outputs/` is gitignored, so this file is the tracked record
of the Final. **§6 is new.** It is a per-clip error analysis run **on DoTA-CAP-dev**. It is exploratory and
**not pre-registered**: it points to directions, and nothing in it is a claim.

**Reading conventions.**
- *macro* = mean per-clip AUC over the two-class clips. It is the decision metric.
- *t95* = paired t interval over 5 seeds (2024–2028).
- *[a, b]* in clip-level tables = cluster bootstrap over source videos, B = 10,000.
- DoTA is read with **protocol B** (E1): `DoTA_s1_ncc[::3]`, whole clip, interpolated to native frames.
- "**DoTA-CAP (n/1397)**" is never put beside full DoTA, LaGoVAD's 62.60 or a phase-4 number (D14).

---

## 0. TL;DR

1. **Adopted model: A3 = CRN + a frozen VideoMAE V2-S video stream** (zero-init residual) on the LaGoVAD trunk.
   It has no KIP or RAFT and no new loss.
2. **On the sealed set (DoTA-CAP-eval, 560 clips, opened once):** A3 macro is **0.7785** vs A0 (raw CLIP) 0.6979.
   Δ = **+0.090 [+0.065, +0.115]**, positive on 5/5 seeds. **Optimism dev − eval is negative for every arm**, so the
   dev numbers were not inflated.
3. **Position is the central issue.** A cubic prior on relative position in the clip, fitted on T2 with no pixels
   (`p_T2`), scores **0.8256 > A3**. After controlling for position (`f_2`), A3's gain is
   **+0.021 [+0.009, +0.034]**. A2's gain (no CRN) does **not** survive (+0.004, CI contains 0).
4. **The video stream is not "motion".** Shuffling the 16 frames inside each window changes the result by
   ±0.0004 (D6). It is *a second, video-based appearance encoder*.
5. **CRN alone does not transfer** (A1 − A0 on DoTA-eval: −0.007, CI contains 0). It does remove the corpus-identity
   shortcut (0.9998 → 0.57), and it is the component that makes A3's gain separable from position.
6. **Dev error analysis (§6):**
   - Per clip, A3 and `p_T2` are **almost uncorrelated** (r = 0.14). A per-clip oracle max(A3, p) scores 0.898.
   - `p_T2` wins because of how DoTA clips are cut: accidents sit at 0.55–0.70 of the clip. When the accident is
     elsewhere, **A3 beats `p_T2` by a wide margin**.
   - Two main weaknesses: the score fades before the anomalous span ends, and **non-ego** accidents score clearly
     below ego accidents.

---

## 1. The v2 model and its four arms

```
frame ─► CLIP ViT-B/16 (frozen) ─► x ─► CRN ─► x̃ ─┐
causal 16-frame clip, 1.5 s ─► VideoMAE V2-S (frozen) ─► u ─► CRN ─► c·u/σ_u ─► W_u (0-init) ─(+)─► h
h ─► temporal encoder (2 layers, RoPE) ─► V^t ─► CoAttn(V^t, z) ─► H_bin → y^bin, H_mul → y^mul
```

| Arm | CRN (R2, per-clip median) | Video stream (V2-S, 384-d) | Note |
|---|:-:|:-:|---|
| A0 | – | – | = phase-4 KIP-off, the baseline |
| A1 | ✓ | – | the "free arm" |
| A2 | – | ✓ | |
| **A3** | ✓ | ✓ | **adopted** (E3) |

Training data: T2 (DADA-2000 original; windows W=20, hop 8; negatives cut from inside the accident videos), minus
T2-val. 20 epochs, step 1740, 5 seeds. No new loss. The only new parameters are `W_u` (≈ 0.20 M).

**Losses actually active** (verified from the pilot's `config.yaml` and `metrics.jsonl`; E3 configs differ from the
pilot only in `train.seed`, §18 M1):
- **On:** L_MIL, L_mul-MIL and **L_dvs**. `dvs_sup` and `dvs_sup_mil` are logged on 1,740/1,740 steps.
  DVS runs with **LaGoVAD's default** settings: θ = 0.7 (no-synthesis probability), δ_m = 5,
  `dvs_anchor_mode=span`, 50 % KNN fillers. This is because `data.is_egocentric=false`.
  The KAT-VAD spec's ego setting (θ_ego 0.85, δ_m 2, spec §6.1) is **not** used, and neither is the motion-aware
  KNN key, which needs RAFT `e_O`. Phase 4 ran the same way, so A0 ≡ phase-4 KIP-off holds.
- **Off: L_neg.** `captions_from_definitions=false`, so no caption features exist and the `cap_contrastive` term never
  runs (no such key in any log). `cap_contrastive_weight: 1.0` has no effect.
  "Losses unchanged from LaGoVAD" must therefore read "LaGoVAD's losses minus L_neg".

## 2. Data and protocol

| Set | Size | Role |
|---|---|---|
| T2-train − T2-val | 1,272 − 219 sources | training |
| T2-val | 219 sources / 645 windows | guardrail (O1′), pilot |
| T2-test | 1,106 windows | Final guardrail; never used to adopt |
| DoTA-dev / DoTA-eval | 702 / 700 clips (eval: 695 have features, §20) | endpoint for CRN and `F` (A0/A1) |
| **DoTA-CAP-dev / -eval** | **569 / 560 clips** | endpoint for arms with the video stream (DoTA pixels recovered from CAP-DATA, 1,129/1,397 = 80.8 %) |

DoTA-CAP is representative of DoTA (D15):
- CLIP-only probe: dev 0.638 / CAP 0.636 / dropped clips 0.649.
- A0: 0.6628 / 0.6591 / 0.6787.

CAP is about 0.01–0.02 harder than the dropped clips, and every interval overlaps.

---

## 3. Results in the order they were measured

| Step | Question | Main result | Conclusion |
|---|---|---|---|
| **P1 · K** (2026-09-28) | Does VideoMAE add anything over CLIP? (probe, 295 T2 sources) | `u` 0.760 vs `x` 0.615; Δ([x;u]−x) **+0.132 [+0.109, +0.155]** | GO |
| **P1 · K-pos** | Is it just position? | Position ruler `p` **0.730**; CLIP adds only **+0.004** over position; `u` beyond position **+0.043 [+0.023, +0.064]** | `u` is real, but it is only ⅓ of the headline |
| **P3 · E1** (09-30) | At which stride should DoTA be read? | s3 beats s8 by **+0.033 [+0.019, +0.048]**, 3/3 seeds. Largest on short accidents (<30 % share: +0.040) | adopt **protocol B**. This is a protocol change, not a better model |
| **P2 · E2(b,c) @ s3** (09-30) | Which reference for CRN? | R2 `r` 0.7092 vs R1 0.7081 (tie); E2(c) transfer **+0.037 [+0.027, +0.047]**; metric `r` dominated by `−f` (0.793) | **CRN = R2** |
| **P4 · E2(d)** (10-04) | V2-B or V2-S? | A3 transfer Δ: B 0.151 vs S 0.133 (gap 0.018 < 0.02) → S. **Cubic `p` = 0.852**, above every feature set | **V2-S** (tie rule picks the cheaper encoder) |
| **P4 · D6** (10-05) | Does temporal order contribute? | `[x;u;p]` ordered − shuffled **−0.0004 / −0.0003** (CI contains 0); shuffled `[x;u]` − `x` still +0.14 / +0.12 | **may not be called "motion"** |
| **P6 · pilot** (10-04/05, s2099) | Do the mechanisms work? Is there a C14-style collapse? | CRN: shortcut on `V^t` **0.9999 → 0.551**. A2/A3 fail O1 as registered, but window AUC and macro **rise together** → not C14 (Amendment 8, O1′). T2-test: O1′ PASS | All 4 arms kept for E3 |
| **E3** (10-06, 20 runs) | Which arm to adopt? (DoTA-CAP-dev) | A3 − A0 **+0.0855 [+0.057, +0.114]**; A3 − A1 **+0.098 [+0.086, +0.109]**; A1 − A0 (DoTA-dev) −0.015 (ns) | **adopt A3**, `F` = A0 |
| **Final** (10-06) | Does it hold on the sealed sets? | see §4 | **holds**, optimism < 0 |

## 4. Final — sealed sets, opened exactly once

### 4.1 Macro per arm (mean over 5 seeds)

| Arm | DoTA-CAP-eval | DoTA-CAP-dev | optimism (dev − eval) | DoTA-eval (695) | micro / AP (CAP-eval) |
|---|---|---|---|---|---|
| A0 | 0.6979 [0.674, 0.720] | 0.6768 | −0.021 | 0.6971 | 0.651 / 0.447 |
| A1 | 0.6835 [0.661, 0.705] | 0.6577 | −0.026 | 0.6840 | 0.658 / 0.449 |
| A2 | 0.7651 [0.743, 0.786] | 0.7536 | −0.012 | — | 0.716 / 0.532 |
| **A3** | **0.7785 [0.763, 0.794]** | 0.7576 | −0.021 | — | 0.730 / 0.524 |
| `p_T2` (position only, no pixels) | **0.8256** | 0.8220 | | 0.8239 | |
| `t/N` (monotone) | 0.562 | 0.557 | | 0.575 | |

### 4.2 Contrasts (paired t95 over seeds, n = 5)

| Contrast | Set | Δ | t95 | Per seed (2024…2028) |
|---|---|---|---|---|
| A1 − A0 | DoTA-eval | −0.0072 | [−0.034, +0.019] | +0.015 / −0.016 / +0.014 / −0.017 / −0.033 |
| A2 − A0 | CAP-eval | +0.0871 | [+0.057, +0.117] | +0.103 / +0.070 / +0.120 / +0.080 / +0.063 |
| **A3 − A0** | CAP-eval | **+0.0900** | **[+0.065, +0.115]** | +0.112 / +0.076 / +0.111 / +0.082 / +0.070 |
| **A3 − A1** | CAP-eval | **+0.0982** | **[+0.090, +0.107]** | +0.100 / +0.086 / +0.099 / +0.103 / +0.103 |
| A2 − A1 | CAP-eval | +0.0953 | [+0.082, +0.109] | |

### 4.3 Position control (§19 P4, `f_w = z(ȳ) + w·z(p_T2)`, w = 2 primary)

| Δ (clip-level) | Set | raw | w = 1 | **w = 2** |
|---|---|---|---|---|
| A3 − A0 | CAP-eval | +0.081 | +0.044 | **+0.021 [+0.009, +0.034]** |
| A3 − A1 | CAP-eval | +0.095 | +0.036 | **+0.017 [+0.008, +0.027]** |
| A2 − A0 | CAP-eval | +0.067 | +0.021 | +0.004 [−0.010, +0.018] |
| A3 + 2p − `p_T2` | CAP-eval | | | **+0.023 [+0.014, +0.031]** |
| A0 + 2p − `p_T2` | CAP-eval | | | +0.002 [−0.011, +0.014] |
| (dev, for reference) A3 − A0 | CAP-dev | +0.081 | +0.045 | +0.023 [+0.013, +0.033] |

All three P5 sentences fired positive:
- *"A3's gain over A0 survives the T2 position prior"*;
- *"the video stream's gain on top of CRN survives the T2 position prior"*;
- *"A3 carries signal beyond position"*.

Dev and eval agree to the third decimal.

### 4.4 T2-test (1,106 windows; guardrail)

| Arm | micro | macro | window AUC | macro Δ vs A0 (t95) | O1′ |
|---|---|---|---|---|---|
| A0 | 0.6228 | 0.6307 | 0.667 | — | — |
| A1 | 0.6594 | 0.6377 | 0.716 | +0.007 [+0.000, +0.014] | 5/5 |
| A2 | 0.7153 | **0.7080** | 0.771 | **+0.077 [+0.063, +0.091]** | 5/5 |
| A3 | 0.7095 | 0.6950 | 0.771 | +0.064 [+0.051, +0.078] | 5/5 |

Clip oracle 0.7037. In-domain, A2 > A3 by about 0.013: **CRN has a small in-domain cost on T2**, consistent with the
pilot.

---

## 5. What may and may not be claimed

| ✅ May say | ❌ May not say |
|---|---|
| "v2 as a whole (CRN + V2-S) improves DoTA-CAP (n/1397) over A0" (+0.090 on eval) | "motion" or "kinematics-aware" for the video stream (D6) |
| "the video stream (V2-S) improves DoTA-CAP on top of CRN" (+0.098) | A DoTA-CAP number beside 62.60 or full DoTA (D14) |
| The gain is **≈ +0.02** after controlling for the T2 position prior, on both dev and eval | "+0.09" without `p_T2` beside it |
| No selection optimism (dev ≤ eval for every arm) | "CRN improves transfer" (A1 − A0 contains 0) |
| CRN removes the T2↔DoTA corpus shortcut in `V^t` (0.9998 → 0.57) | A2/A3 "pass O1". The required wording is "O1 FAIL as registered; not collapsed under O1′" |

---

## 6. Per-clip error analysis — DoTA-CAP-dev (EXPLORATORY, not pre-registered)

*Inputs: seed-averaged scores of A0–A3 (E3 `clip_scores.npz`) and `p_T2` on the 569 dev clips. This runs on dev
because eval is already opened. Any improvement drawn from it must be tested on a set that has never been looked at
(see §8).*

### 6.1 A3 and position are complementary, not redundant

| Measure | Value |
|---|---|
| Correlation of per-clip AUC(A3) with AUC(`p_T2`) | **0.135** |
| Share of clips where A3 > `p_T2` | 35 % |
| Share of clips where A3 > A0 | 63 % |
| Per-clip oracle max(A3, `p_T2`) | **0.898** (A3 0.758, `p_T2` 0.822) |
| A3 + 2·p (`f_2`) | 0.842 |

### 6.2 `p_T2` wins through DoTA's clipping convention; away from it, A3 wins

| Accident centre (relative position) | n | A0 | **A3** | `p_T2` | A3 − p | % clips A3 > p |
|---|---|---|---|---|---|---|
| < 0.40 | 91 | 0.609 | **0.698** | 0.432 | **+0.266** | 80 % |
| 0.40–0.55 | 221 | 0.675 | 0.763 | 0.847 | −0.085 | 37 % |
| 0.55–0.70 | 215 | 0.683 | 0.768 | **0.982** | −0.214 | 7 % |
| ≥ 0.70 | 42 | 0.796 | **0.806** | 0.713 | **+0.093** | 71 % |

A3 is **almost flat across position** (0.70–0.81). `p_T2` ranges from 0.43 to 0.98.

So the 0.82 of `p_T2` is a property of the benchmark (DoTA places accidents around 0.55–0.70 of the clip), not
detection ability. In streaming or deployment there is no "relative position in the clip", so `p_T2` cannot be used.
**This is the strongest argument for defending A3 against the objection that "position beats the model".**

### 6.3 The score fires at onset and fades before the anomalous span ends

Mean within-clip z-score of A3 around the anomaly start and end (native frames, 10 fps):

| offset (frames) | −40 | −20 | −10 | 0 | +10 | +20 | +30 | +40 |
|---|---|---|---|---|---|---|---|---|
| A0 around **start** | −0.57 | −0.38 | −0.17 | +0.09 | +0.42 | +0.48 | +0.27 | +0.10 |
| A3 around **start** | −1.02 | −0.42 | −0.03 | +0.30 | +0.58 | **+0.64** | +0.42 | +0.17 |
| A3 around **end** | −0.06 | +0.49 | +0.54 | +0.43 | +0.24 | −0.05 | −0.37 | −0.59 |

- The peak comes about 1–2 s after onset, and by +4 s it has dropped to about a quarter. The score's argmax lies inside
  the span on **only 61 %** of clips.
- The mean score before the anomaly is **0.55**, versus 0.86 inside it. The two are separable (in > pre on 88 % of
  clips), but the baseline level is high.
- By share bin, A3 − `p_T2` goes −0.001 (<30 %) → −0.105 (30–50 %) → −0.173 (50–70 %) → −0.196 (>70 %).
  **The longer the span, the more A3 loses.** The model learns "the moment of impact"; DoTA labels the aftermath too.
- This is expected behaviour for MIL top-k (k = 4 over 20-step windows): the loss rewards only a few salient steps.
- **Note (2026-10-09, H2(a) read-out):** a causal hold of the score fixes none of this. 277/569 clips have ≥ 3 s of
  normal after the span, where the hold costs A3 −0.125: part of the "fade" is the model tracking the span's end. The
  share-bin slope above is largely position — at >70 % the few normal frames sit at the clip start, which `p_T2`
  ranks trivially. Read this bullet's numbers beside §6.2, not as a separate weakness.

### 6.4 Non-ego accidents are weaker than ego accidents

| | n | A0 | A3 | `p_T2` | A3 − A0 | A3 − p |
|---|---|---|---|---|---|---|
| ego | 321 | 0.737 | **0.804** | 0.839 | +0.068 | −0.034 |
| non-ego | 248 | 0.599 | **0.697** | 0.800 | +0.098 | −0.103 |

Non-ego accidents (other vehicles colliding, often small and far away in the frame) score **0.107** below ego.
Both encoders squash the *whole frame* to 224² (no crop) and produce one global vector, which dilutes small objects.

### 6.5 By accident class (n ≥ 15) and by onset time

| Class | n | A0 | A3 | A3 − A0 | A3 − p |
|---|---|---|---|---|---|
| turning | 203 | 0.712 | 0.767 | +0.055 | −0.057 |
| lateral | 106 | 0.646 | 0.782 | **+0.136** | −0.039 |
| moving_ahead_or_waiting | 64 | 0.633 | 0.720 | +0.087 | −0.118 |
| oncoming | 59 | 0.718 | **0.815** | +0.097 | **+0.006** |
| leave_to_right | 43 | 0.620 | 0.743 | +0.122 | −0.089 |
| leave_to_left | 35 | 0.642 | **0.682** | +0.040 | −0.174 |
| obstacle | 19 | 0.769 | 0.790 | +0.021 | −0.097 |
| pedestrian | 17 | 0.681 | 0.726 | +0.045 | −0.047 |

Anomalies that start very early (< 20 % of the clip, n = 54) are the hardest group: A0 0.560, A3 0.675. They have
little normal context before the onset.

---

## 7. Component ledger: what each part buys

| Component | Evidence for | Evidence against | Status |
|---|---|---|---|
| Protocol B (s3) | +0.033 on the same checkpoints, 3/3 seeds (E1) | — | kept; it is protocol |
| V2-S video stream | +0.098 over CRN (eval), 5/5 seeds; T2-test +0.077 | D6: not temporal order; most of it co-moves with position (G6) | kept, named "video appearance encoder" |
| CRN (R2) | shortcut 0.9998 → 0.57; lets A3 survive position control (A2 does not) | A1 − A0 contains 0 on DoTA-dev and -eval; ≈ 0.013 macro in-domain cost on T2 (A3 vs A2) | kept inside A3; no stand-alone claim |
| V2-S vs V2-B | cheaper; wins by the tie rule | B is ahead beyond position in transfer (+0.037 vs +0.018, E2(d)) | candidate to revisit |

---

## 8. Directions for improvement (ranked by evidence and cost)

> **Global constraint.** DoTA-CAP-eval and DoTA-eval are **already opened**. Any change chosen from §6 or from eval
> no longer has a clean test set on DoTA-CAP. A Final number for the next version needs one of:
> (a) pre-registration and a read on a new set (proposed in H1), or
> (b) an explicit label "second look at an opened set".
> Do not tune on a Δ (lesson 14). Write the decision rule for each direction below before running it.

| # | Direction | Evidence (numbers) | Concrete experiment | Cost | Kill criterion |
|---|---|---|---|---|---|
| **H1** | **Position-shift stress test** (a protocol, not yet a model change) | §6.2: A3 flat across position 0.70–0.81, `p_T2` 0.43–0.98; correlation 0.135 | Re-cut the DoTA-CAP clips (dev, then eval) so the accident centre is spread evenly over [0.1, 0.9]. **Design guard (pending (ay)):** crop both ends to a fixed length distribution; add a same-length crop with the accident at its original position as a control (position cost = control − shifted, context cost = full − control); keep `p_T2` frozen at its T2 fit. No retraining. **Deferred until v2 is closed (user, 2026-10-07)** | **Cheap but not free**: CRN must be re-baked on every crop and the 20 checkpoints re-run (forward pass on cached features; no training, no extraction) | If `p_T2` is still ≥ A3 on the shifted set, the "position is an artefact" argument is wrong |
| **H2** | **Scores that cover the span** (temporal precision) | §6.3: peak at +2 s, then fades; peak-in-span 61 %; A3 − p from −0.001 (<30 %) to −0.196 (>70 %) | (a) Post-hoc causal max-hold or EMA on `y^bin` (a free arm, no training). Fix the window by a rule, e.g. the median span length of **T2**, not of DoTA. (b) If (a) helps: MIL top-k as a **fraction** of window length (`mil_topk_pct`) instead of a fixed k = 4 | (a) cheap; (b) one small E3 (2 arms × 5 seeds) | (a) no macro gain in the >50 % share bins on dev → drop. **(a) RUN 2026-10-09: KILL** (`core/tools/posthoc_hold.py`, plan `katvad-v2-h2a-hold.md` App. A): causal max-hold w = 30 (T2 median span) on A3: Δ all −0.071, Δ share ≥ 0.5 +0.037 (∋ 0), Δ `f_2` −0.022; tail clips −0.125. **(b) not proposed**: `k = L // MIL_TOPK_PCT` already scales with `L`, and the motivating share-bin slope is largely position (see §6.3 note) |
| **H3** | **Region- or object-level features** for non-ego accidents | §6.4: non-ego 0.697 vs ego 0.804 (−0.107) | Add CLIP on a tile grid (e.g. 2×2 + global, max-pooled over tiles) or detector crops. Probe frozen features first, as in K/E2(d): `[x_tiles; u]` vs `[x; u]`, T2 → CAP-dev transfer, **with `p` printed beside it** | Medium: re-extract CLIP for T2 and DoTA. DoTA s1 survives on Drive, but tiles need pixels → only possible on DoTA-CAP | Beyond-position transfer of the probe < +0.01 → not worth building. **Probed 2026-10-09 (`text_window_probe.py`, pending (ba)): KILL.** Text-guided windows: transfer Δ vs `[x;u;p]` **−0.036 [−0.046, −0.026]**; the text margin is at chance zero-shot (0.504). Plain multi-scale tile mean-pool also fails this criterion (−0.012; non-ego −0.021). **→ CLIP tiles closed; only detector crops remain, as a v3 direction** |
| **H4** | **V2-B instead of V2-S** | E2(d): beyond-position transfer B +0.037 vs S +0.018; T2 side B +0.137 vs S +0.107 | Arm A3-B, 5 seeds, same recipe. Compare with A3-S on dev and on the H1-shifted set | 5 GPU runs. V2-B exists for DoTA-CAP and the 295 K sources, but only V2-S was extracted for full T2 (G2) → V2-B must be extracted for T2 | A3-B − A3-S `f_2` contains 0 → keep S |
| **H5** | **A stream that is actually order-sensitive** (only if the thesis wants to keep "kinematics") | D6: ordered − shuffled ±0.0004 | Difference features `u_t − u_{t−k}`, or VideoMAE on reversed clips / one frame repeated ×16 (motion-energy control). Probe first, with D6 as the gate | Cheap (probe) to medium | Shuffled control still contains 0 → drop the motion claim for good |
| H6 | Early-onset anomalies / short context | §6.5: start < 20 %: A3 0.675 (the lowest) | Check CRN reference R2 when most of the clip is anomalous; compare R2 vs R4 (past-only, eligible at s3) on this group | Cheap (probe on caches) | No difference → leave as is |

**Proposed order:**
- **H1 first once v2 is closed** (deferred, pending (ay)). It is cheap, and it decides how the whole position story is told in the thesis.
- **Then H2(a).** Free, and it targets the largest measured weakness.
- **Then H4.** Cheap, with existing evidence.
- **H3 and H5 are research directions for a v3.** Do them only if time allows, each through a probe gate before any
  training.

**Not proposed:**
- Feeding `p_T2` into the model to reach 0.84–0.90. That is benchmark fitting: it needs the clip's end point, fails in
  streaming, and §6.2 shows it collapses when the accident is off its usual position.
- More seeds or more T2-like data. The T2 learning curve is flat (+0.008 on DoTA over 4× the data).
- Foreign normal pools (D2City and BDD-A were both NO-GO, C38).

---

## 9. Appendix — where the source numbers live

| Step | File |
|---|---|
| K, K-pos | `core/docs/v2/MOTION_STREAM_K.md` |
| E2(a–c), D11 | `core/docs/v2/RESULTS_E2_CRN.md` |
| E1 | `core/docs/v2/RESULTS_E1.md` |
| E2(d), D6 | `core/docs/v2/RESULTS_E2D.md` |
| DoTA-CAP | `core/docs/v2/DOTA_CAP.md` |
| Pilot | `core/docs/v2/RESULTS_P6_PILOT.md` |
| E3 | `core/docs/v2/RESULTS_E3.md` |
| Final (raw) | `outputs/v2/v2_final/REPORTS/final_readout.{md,json}` (gitignored) |
| Pre-registered rules | `core/docs/v2/PREREG_ADDENDUM.md` §1–§20 |
| §6 (dev error analysis) | computed from `outputs/v2/v2_s3/REPORTS/pb/A*/dota_cap_dev/clip_scores.npz` + `position_prior/dota_cap_dev/position_prior.npz` + `data/DoTA/metadata_val.json` |
