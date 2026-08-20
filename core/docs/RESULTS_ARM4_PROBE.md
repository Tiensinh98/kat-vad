# Arm 4 + trajectory probe — closing the last two confounds

**Measured:** 2026-08-19, from the saved score curves in
`outputs/{MSAD_ncc,MSAD_ncc_s2025,MSAD_ncc_s2026}/eval_kip_off_warm/scores/*.npz`,
`outputs/{DoTA_ncc,DoTA_ncc_s2025,DoTA_ncc_s2026}/eval_kip_off_warm/scores/*.npz`
and `outputs/DoTA_ncc/probe/*/results.json`.
**Runbook:** `core/docs/COLAB.md` § "Arm 4 + trajectory probe (2026-08-16, post Phase A)".
**Extends** `RESULTS_PHASE_A.md`, which left exactly these two items open.

**One line:** the DoTA gain is **not** the warm start (Δ(on − off_warm) =
**+0.0988 ± 0.0148**, three seeds, nine CIs excluding zero) and **not**
under-convergence (KIP-off's DoTA AUC is flat at 0.559–0.561 across a **33×**
range of train `mil`, including the point where it is exactly as fitted as
KIP-on's endpoint). **H3 is rejected; the warm-start confound is closed.**
Δ ≈ +0.09 is attributable to the KIP module itself.

---

## 1. What ran

| arm | what it is | new? |
|---|---|---|
| `stage2_kip_on` | KIP-on, warm-started from stage 1 | from Phase A |
| `stage2_kip_off` | KIP-off, **cold** start | from Phase A |
| `stage2_kip_off_warm` | KIP-off, warm-started from the **same** stage-1 checkpoint KIP-on used | **A4 — new, 3 runs** |

Plus **A5**: 10 evals of seed 2024's saved `checkpoint_step_{100..500}.pt`
(both arms) on DoTA. No training, no code changes, cached features only.

**Parity verified.** `stage2_kip_off_warm/config.yaml` is byte-identical to
`stage2_kip_off/config.yaml` in every seed. The `--init-weights` flag is not
recorded in the config, so the warm start is verified from the loss instead —
step-1 train `mil`:

| seed | cold off | **off_warm** | on |
|---|---|---|---|
| 2024 | 0.86818 | **0.70148** | 0.70146 |
| 2025 | 0.68530 | **0.82810** | 0.81630 |
| 2026 | 0.78010 | **0.66500** | 0.66030 |

`off_warm` enters stage 2 next to `on` and away from cold `off` in every seed —
seed 2024 agrees with `on` to four decimals. The two warm arms share a trunk;
the cold arm does not.

Coverage is identical to Phase A across every arm: MSAD 240 videos / 18,350
frames / 96 macro-scorable; DoTA 1,397 clips / 18,369 frames / 1,392
macro-scorable.

Method (identical to `RESULTS_PHASE_A.md` §7): micro AUC over concatenated
frames, per-clip min-max for DoTA and raw for MSAD (`--score-norm auto`,
lesson 12); macro AUC = mean per-clip AUC over clips with both classes; CIs =
paired bootstrap resampling **clips**, 2,000 draws, `default_rng(0)`, both arms
scored on the same resample.

---

## 2. A4 — DoTA: the warm start is not the mechanism

| seed | metric | off_warm | KIP-on | Δ(on − off_warm) | 95 % CI |
|---|---|---|---|---|---|
| 2024 | micro AUC | 0.5482 | 0.6520 | **+0.1039** | [+0.0961, +0.1115] |
| | AP | 0.3409 | 0.4308 | +0.0899 | [+0.0823, +0.0981] |
| | macro AUC | 0.5470 | 0.6746 | +0.1277 | [+0.1179, +0.1374] |
| 2025 | micro AUC | 0.5313 | 0.6416 | **+0.1104** | [+0.1022, +0.1193] |
| | AP | 0.3279 | 0.4165 | +0.0887 | [+0.0810, +0.0972] |
| | macro AUC | 0.5289 | 0.6550 | +0.1260 | [+0.1157, +0.1370] |
| 2026 | micro AUC | 0.5466 | 0.6289 | **+0.0822** | [+0.0731, +0.0920] |
| | AP | 0.3382 | 0.4035 | +0.0653 | [+0.0571, +0.0737] |
| | macro AUC | 0.5398 | 0.6291 | +0.0891 | [+0.0772, +0.1009] |

**Across seeds:** micro Δ mean **+0.0988**, range [+0.0822, +0.1104], spread
±0.0148. AP Δ mean +0.0813. Macro Δ mean +0.1143. **Nine intervals, nine
exclusions of zero.**

Against the cold Δ from Phase A (mean +0.0915, range [+0.0829, +0.1004]): the
warm-corrected Δ is **not smaller — it is marginally larger**, and the per-seed
change goes in both directions (2024 +0.0911 → +0.1039; 2025 +0.0829 → +0.1104;
2026 +0.1004 → +0.0822).

This is the runbook's decision-table **row 1**: *Δ ≥ +0.05 with CIs excluding
zero → warm start is not the mechanism.* **The A/B is now "KIP vs no KIP"**,
both arms starting from the same stage-1 trunk, differing only in whether the
KIP module exists. That is the strongest version of the claim the artifacts
support, and it is the version to report.

### 2.1 Why it did not shrink — the trunk pretraining transfers nothing stable

Δ(off_warm − off_cold) on DoTA, i.e. what 125 epochs of `L_KIP_rec` /
`L_KIP_align` buy through the **shared temporal encoder alone**, with no KIP
module at inference:

| seed | Δ micro AUC | 95 % CI | Δ AP | Δ macro |
|---|---|---|---|---|
| 2024 | **−0.0128** | [−0.0229, −0.0030] | −0.0107 | −0.0169 |
| 2025 | **−0.0273** | [−0.0370, −0.0177] | −0.0174 | −0.0241 |
| 2026 | **+0.0181** | [+0.0079, +0.0274] | +0.0134 | +0.0208 |
| mean | −0.0073 | — | −0.0049 | −0.0067 |

Every CI excludes zero, **and the sign flips between seeds.** Stage-1
pretraining moves DoTA by ±0.02 in a seed-dependent direction — an order of
magnitude below KIP's +0.099, and not even reliably positive.

So the transferable thing is **the KIP module in the forward pass at
inference**, not the pretraining it delivered to the trunk. That is a sharper
and more defensible claim than "KIP + its pretraining helps", and it is only
available because arm 4 exists.

> The sign flip is the same pattern Phase A §2 recorded for the arms'
> *absolute* levels: what a seed moves is the level, not the paired gap. Here
> the warm start is a level effect; KIP is the gap.

---

## 3. A4 — MSAD: still null, still the finding

| seed | off_cold | off_warm | KIP-on | Δ(on − off_warm) | 95 % CI |
|---|---|---|---|---|---|
| 2024 | 0.8922 | 0.8866 | 0.8868 | +0.0001 | [−0.0047, +0.0045] |
| 2025 | 0.8857 | 0.8872 | 0.8862 | −0.0010 | [−0.0058, +0.0035] |
| 2026 | 0.8824 | 0.8844 | 0.8815 | −0.0030 | [−0.0078, +0.0015] |

ΔAP: +0.0047 / −0.0034 / +0.0057 — every CI includes zero. Δ(off_warm −
off_cold) is null too (−0.0056 / +0.0015 / +0.0020, all CIs including zero).

**The asymmetry survives arm 4 intact**: the motion pathway pays on the
motion-dominated benchmark and is inert in-domain, and neither half of that is
a warm-start artifact.

The weak macro signal from Phase A §3.1 also survives in the same weak form:
Δmacro(on − off_warm) is +0.0167 / +0.0164 / +0.0085 — positive in all three
seeds, **every CI including zero** (P>0 = 0.966 / 0.915 / 0.811). Still an
observation, still not a result.

---

## 4. A5 — the trajectory probe: H3 is rejected

H3: *KIP-on transfers better because it is less fitted to MSAD* (its final train
`mil` is 3.5–6.3× KIP-off's, `RESULTS_PHASE_A.md` §5). If true, an
equally-under-fitted KIP-off model should transfer just as well.

Seed 2024, both arms, `mil` averaged over the 20 steps ending at the checkpoint:

| arm | step | train `mil` | DoTA micro AUC | AP | macro |
|---|---|---|---|---|---|
| off | 100 | 0.0410 | 0.5589 | 0.3517 | 0.5620 |
| off | 200 | 0.0031 | 0.5604 | 0.3511 | 0.5630 |
| off | 300 | 0.0016 | 0.5604 | 0.3502 | 0.5623 |
| off | 400 | 0.0012 | 0.5608 | 0.3505 | 0.5640 |
| off | 500 | 0.0013 | 0.5607 | 0.3503 | 0.5638 |
| on | 100 | 0.0834 | 0.6097 | 0.3947 | 0.6186 |
| on | 200 | 0.0207 | 0.6412 | 0.4214 | 0.6587 |
| on | 300 | 0.0103 | 0.6482 | 0.4273 | 0.6691 |
| on | 400 | 0.0074 | 0.6514 | 0.4298 | 0.6737 |
| on | 500 | **0.0075** | 0.6519 | 0.4303 | 0.6746 |

Four independent readings, all pointing the same way:

1. **At matched convergence the gap is undiminished.** KIP-on's endpoint sits at
   `mil` ≈ 0.0075; KIP-off crosses that value at **step ≈ 161**, bracketed by
   step 100 (`mil` 0.0410, *less* fitted → 0.5589) and step 200 (`mil` 0.0031,
   *more* fitted → 0.5604). Interpolated KIP-off ≈ **0.560** against KIP-on's
   **0.6519** — a gap of **+0.092**, i.e. the entire headline Δ.
2. **KIP-off's transfer is flat.** Its `mil` falls **33×** (0.0410 → 0.0012)
   while DoTA AUC moves 0.5589 → 0.5607 — a span of **0.0019**. Convergence
   level does not index DoTA transfer for that arm at all.
3. **The direction is backwards for H3.** KIP-on *gains* +0.042 as it fits
   harder (0.6097 → 0.6519). H3 predicts less fitting → better transfer;
   KIP-on's own trajectory does the opposite.
4. **The curves do not overlap when plotted against `mil`.** KIP-on at its
   least-converged probed point (step 100, `mil` 0.0834 — 11× less fitted than
   KIP-off's endpoint) already scores 0.6097, above KIP-off's best-ever 0.5608.

This is decision-table **row 2** verbatim: *KIP-off stays at 0.53–0.56 along its
whole trajectory while KIP-on sits at 0.63.* **H3 is dead at matched
convergence.** Unambiguous, so the probe was not extended to seeds 2025/2026.

### 4.1 Integrity checks

- Probe step-500 reproduces the Phase A arms **exactly** — off 0.5607, on 0.6519
  — confirming `checkpoint_last` == `checkpoint_step_500` and that the probe
  loop scored the same models as the headline evals.
- All 10 runs cover 18,369 frames / 1,397 clips / 1,392 macro-scorable clips.
- The `off` runs **produced results at all**, which per lesson 5 (`strict=True`
  key checking) proves `EXTRA` was set: a KIP-off checkpoint loaded under
  `kip.enabled=true` raises. The `$EXTRA` defect of
  `RESULTS_PHASE_A.md` §8 did not recur.
- `results.json` records the original `checkpoint_step_*.pt` paths, not
  `_slim.pt`: `torch.load` worked in the user's session, so §A5.0 was correctly
  skipped (the runbook allows this — the numpy failure is environment drift,
  not a property of the artifacts).

### 4.2 The one segment the probe cannot see

`train.checkpoint_every_steps=100` over a 500-step run leaves **no checkpoint
below step 100**, where `mil` falls from 0.63 to 0.041 — 94 % of the loss range
is covered by the first (unprobed) 20 % of steps. So KIP-off's
*very*-under-converged regime is unmeasured and cannot be measured without
retraining.

It does not weaken the verdict: H3's claim is about convergence matched to
KIP-on's **endpoint**, and that point is bracketed. A KIP-off arm that is flat
to ±0.002 across a 33× `mil` range is not plausibly hiding a +0.07 jump just
below. But it is a sampling defect worth not repeating — see lesson **16**.

---

## 5. D4 re-checked on the warm comparison — refuted six for six

Per-clip macro AUC Δ(on − off_warm), split on `metadata_val.json` →
`anomaly_class` prefix:

| seed | ego (n=802) | other (n=590) | mean | win | loss |
|---|---|---|---|---|---|
| 2024 | +0.1134 | **+0.1469** | +0.1276 | 73.8 % | 13.1 % |
| 2025 | +0.1070 | **+0.1518** | +0.1260 | 71.1 % | 15.8 % |
| 2026 | +0.0577 | **+0.1323** | +0.0893 | 64.8 % | 20.6 % |

`other > ego` again in all three seeds — **six for six** counting Phase A's cold
comparison. **The ego-kinematics story is refuted and stays refuted.** Do not
claim KIP works by modelling the ego-vehicle's own motion.

Both groups still gain substantially, and per-clip wins outnumber losses ~4:1,
so this is a broad effect and not a subgroup carrying the average.

---

## 6. What this establishes, and what is left

**Established**

1. **The warm-start confound is closed.** Δ(on − off_warm) = +0.0988 ± 0.0148,
   nine CIs excluding zero, no shrinkage against the cold Δ. The comparison is
   now "KIP vs no KIP" from a shared trunk.
2. **Stage-1 trunk pretraining alone transfers nothing stable** — ±0.02, sign
   flipping across seeds. The gain lives in the module at inference.
3. **H3 is rejected.** At matched train `mil`, KIP-off reaches ≈ 0.560 against
   KIP-on's 0.6519. KIP-off's transfer is flat over a 33× `mil` range, and
   KIP-on's improves as it fits harder — the opposite of the regularizer story.
4. **MSAD stays null under arm 4**, and D4 stays refuted.

Combined with Phase A: **Δ ≈ +0.09 DoTA micro AUC is attributable to the KIP
module itself**, at +1.82 % inference parameters, RGB-only at test time, with no
in-domain cost.

**Still open**

1. **The mechanism.** Two rival explanations are eliminated; none is confirmed.
   D4 (ego-kinematics) is refuted, D5 (MSAD multi-class accuracy) did not
   replicate under `_ncc`. *What* KIP does is unknown. The next honest step is
   eval-time KIP diagnostics — save ŷ_O, the gate α and the shift magnitude,
   and look at where the gain concentrates.
2. **One benchmark.** The gain is DoTA-only, zero-shot from MSAD-trained
   models. PreVAD / TAD would say whether it is "motion" or "DoTA".
3. **`L_KIP_align` still runs at 3.66 vs chance 4.27** with both A11 mitigation
   knobs off (`RESULTS_MSAD.md` §6.5). The module works *despite* an alignment
   loss that is barely above chance, which is itself a mechanism clue.
4. **Checkpoint selection.** Every arm is `checkpoint_last` at train `mil` ≈
   0.001–0.01. Phase B's val split would fix that but on a 20 %-smaller training
   set whose numbers are not comparable to anything measured — it stays deferred
   (§7).

**Discipline note, to be carried into any paper text:** §4 reads DoTA *test*
scores along a training trajectory. It is a **diagnostic, not model selection**.
The reported arm is `checkpoint_last` in every table; the probe curve must never
become a headline number, or the result reads as tuned on test.

**Do not change KIP's architecture, losses, or hyperparameters on this result**
(lesson 14). The confounds are closed, but the mechanism is not identified, and
tuning against an unexplained delta is how you fit the benchmark.

---

## 7. Phase B stays deferred — now on evidence, not on a guess

`.project/plans/msad-ncc-seeds-and-selection.md` §3 specified a `--val-ratio`
three-way split, a KNN rebuild, `core/tools/select_checkpoint.py` and a full
retraining cycle, to answer H3.

A5 answered H3 for **10 evals on cached features**, on the training set
everything else was measured on, keeping every number comparable to Phase A.
Phase B's machinery is now needed only for *checkpoint selection as a
contribution*, not for this question. Build it when selection itself is the
deliverable.

---

## 8. Reproducing this document

All numbers come from the saved `.npz` score curves and `metrics.jsonl` — no
re-inference. Analysis script (one-off, scratchpad): loads
`{run}/scores/*.npz`, min-max normalizes per clip for DoTA, and runs the paired
clip bootstrap described in §1. The macro bootstrap resamples the **precomputed
per-clip AUC vector** rather than re-fitting 1,397 AUCs per draw — exactly
equivalent, ~200× faster.

- §2, §3 — `eval_kip_on` vs `eval_kip_off_warm`, and `eval_kip_off_warm` vs
  `eval_kip_off`, per seed
- §4 — `outputs/DoTA_ncc/probe/{on,off}_checkpoint_step_*/results.json` joined to
  `outputs/MSAD_ncc/stage2_kip_{on,off}/metrics.jsonl` by `global_step`
- §5 — per-clip AUC deltas grouped by `data/DoTA/metadata_val.json` →
  `anomaly_class`, split on the `ego:` / `other:` prefix
