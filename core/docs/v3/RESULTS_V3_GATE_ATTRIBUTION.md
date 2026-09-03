# v3 gate attribution — the plain-TSM control settles it

**Measured:** 2026-09-01, from the saved score curves in `outputs/v3/**` plus the
archived KIP-off and KIP-on arms in `outputs/{MSAD,DoTA}_ncc*`.
**Runbook:** `core/docs/v3/setup/MSAD_DOTA_V3_SETUP.md`.
**Notebook that ran it:** `collab/MSAD/v3/train.py` (commands verified against the
runbook line by line).
**Answers:** the pre-registered question in setup §2.2 / §6.12 and lesson **C24**.

> **One line.** A fixed 50 % channel shift — no flow, no PMG head, no KIP losses,
> no stage-1 warm-up — reproduces the **entire** +0.09 DoTA gain, and is
> **statistically indistinguishable** from the full v1 KIP (Δ = +0.0109,
> t95 [−0.0588, +0.0805], n = 3). The v3 **rank gate makes it worse**
> (−0.0683 vs the smoother, bootstrap CI [−0.0790, −0.0580], P>0 = 0.000).
> **The pre-registered prediction is confirmed. KIP's measured contribution is
> temporal smoothing.** The motion-induction claim is not supported by any
> ablation in this campaign.

---

## 1. What ran, and what the numbers are computed from

Six arms from setup §2.2, trained on MSAD-full, scored in-domain on MSAD-full and
zero-shot on DoTA. Seed 2024 is complete for all six; **A2 also has seeds 2025
and 2026.**

| Arm | Gate | Params @ inference | Stage 1 | Flow | Seeds |
|---|---|---:|---|---|---|
| **A0** KIP-off | — | 0 | n/a | no | 2024 (+2025/26 archived, §2.4) |
| **A1** rank (the v3 model) | `rank`, 0 params | 311,808 | yes | yes | 2024 |
| **A2** plain-TSM control | `constant`, r = 0.5 | 311,808 | **no** | **no** | **2024, 2025, 2026** |
| **A2b** constant + PMG trained | `constant`, r = 0.5 | 311,808 | yes | yes | 2024 |
| **A3** v1 bridge | `mlp_frozen`, 321 params | 312,129 | yes | yes | 2024 |
| **A4** gradient intervention | `mlp_ste`, 321 params | 312,129 | yes | yes | 2024 |
| *V1* archived KIP-on | `mlp_frozen` (pre-v3 tree) | 312,129 | yes | yes | 2024, 2025, 2026 |
| *gate_a* LaGoVAD `best.ckpt` | — | 0 | n/a | no | one-off |

Param counts measured directly (`KATVAD.from_config(..., training=False)`):
`rank`/`constant` carry **311,808 KIP params at inference and 0 on the score
path**; the MLP family adds 321. Setup §0 fact 3 holds.

### 1.1 Metric protocol — read this before quoting a number

Every number below is **recomputed in float64 from the `.npz` score curves**, with
per-clip min-max on DoTA (`(s−min)/(max−min)`, LaGoVAD's `offline_dota_eval.py`
convention, flat curves mapped to zeros) and **raw pooling on MSAD** — resolved
from the label distribution, not the dataset name (lesson **C12**).

**These are not the `results.json` numbers, and the difference is lesson C22.**
`core/evaluate.py` normalises in **float32**; we reproduced its output to the
last digit (A2 s2024: 0.642111 / AP 0.419967) and the float64 recompute of the
same files gives **0.642312 / AP 0.421551**. The gap is ΔAUC ≈ 0.0002,
ΔAP ≈ 0.0016 — ~50× smaller than the effect under test, so **no conclusion here
changes** — but it is real, C22 is still open in `normalize_scores`, and a table
must not mix the two. Everything in this document is on the float64 side.

**Never put a DoTA min-max number and an MSAD raw number in one column.**

### 1.2 One clip dropped, everywhere

`outputs/v3/DoTA_rank_s2024/eval_dota/scores/qzMjfBx1KI0_003085.npz` is
**0 bytes** — a truncated write, since `evaluate --save-scores` does not use the
atomic `.part`→rename path that `core/tools/feature_cache.py` gives the
extractors (lesson **C11**, now applying to a second writer). That clip
(`ego: leave_to_right`, 12 sampled frames) is therefore dropped from **all**
arms, so every comparison runs on an identical **1,396-clip / 18,357-frame**
DoTA set. 12 frames of 18,369 — 0.065 %. MSAD is complete at 240 clips.

### 1.3 The A0 seeds 2025 / 2026 are borrowed, and the substitution is measured

A0 was retrained under v3 only at seed 2024. Seeds 2025/2026 come from
`outputs/{MSAD,DoTA}_ncc_s{2025,2026}/eval_kip_off`. Two facts license that:

1. **Config parity.** The archived KIP-off configs differ from the v3 KIP-off
   config in `kip.*` keys only (`gate_type`, `const_shift_ratio`, `disable_pmg`,
   `ecmr_lambda`, `gate_signal`), all of which are inert under
   `kip.enabled=false`. `num_epochs`, `batch_size`, LR schedule, `amp`, dataset,
   split and every cache path match.
2. **A direct reproduction at the shared seed.** v3-trained A0 vs archived A0,
   both seed 2024, same protocol:

| | MSAD micro | DoTA min-max |
|---|---|---|
| A0 trained under v3 | 0.8924 | 0.5606 |
| A0 archived (pre-v3) | 0.8922 | 0.5611 |
| **Δ** | **+0.0002** | **−0.0005** (CI [−0.0008, −0.0002]) |

Two independent training runs of the same configuration on two code trees agree
to 5·10⁻⁴ AUC. **The substitution costs nothing measurable** — it is 180× smaller
than the effect under test. The same test passes on the KIP-on side: A3
(`mlp_frozen` retrained under v3) vs the archived V1 KIP-on arm, seed 2024 —
0.8868 vs 0.8868 MSAD, 0.6519 vs 0.6524 DoTA (Δ = **−0.0004**, CI [−0.0007,
−0.0002]). **The v3 tree reproduces the v1 tree on both arms.** Setup §0 fact 2's
bit-identity claim survives contact with a retrain.

---

## 2. Validity gates — all pass

| Gate | Where | Result |
|---|---|---|
| **Stage-1 convergence** (spec v3 §8, unenforced) | setup §5.1c | **PASS both families.** `L_KIP_rec` 16.56→9.37 (ratio 0.566, `rank`) and 16.54→9.33 (0.564, `mlp_frozen`); bar is < 0.60. Compare v1's 19.2→9.7 |
| **Gate identity** | setup §6.8 | **PASS all 16 KIP-on evals, 0 VOID.** See table below |
| **KIP-off carries no gate diagnostics** | setup §6.2 | **PASS** — A0's `.npz` hold `{class_names, gt, score, sim}` and no `kip_*` key, so the right checkpoint was scored |
| **Score-norm resolution** | setup §6.7 | **PASS** — DoTA `minmax` (1,394/1,397 abnormal), MSAD `none`, on every arm |
| **Protocol parity** | — | **PASS** — DoTA 18,369 frames / 1,397 videos and MSAD 18,350 / 240 identical across all 16 evals before the §1.2 drop |
| **Warm start actually happened** | setup §5.1 | **PASS by inference.** A1/A2b/A3/A4 open stage 2 at `kip_rec` ≈ 7.4–8.5 against a cold ≈ 16.5. Had to be inferred — `--init-weights` is still absent from `config.yaml` (lesson **C17**, still open) |
| **Training completion** | setup §5 | **PASS** — every stage-2 run reached step 500 |
| **D1 pipeline sanity** (`gate_a` ≥ 0.58 min-max) | `DOTA_EVAL.md` §5 | **PASS** — 0.6012 |

### 2.1 The §6.8 gate check, in full

Mean within-clip span of `s_t`, over 128 channels, first 200 clips per eval:

| Arm | Expected | Measured span | `s` mean | ratio |
|---|---|---|---|---|
| A2 `constant` (all 3 seeds) | 0 | **0.0** | 64.0 | 0.500 |
| A2b `constant` | 0 | **0.0** | 64.0 | 0.500 |
| A1 `rank` | 100–128 | **128.0** (min 128, max 128) | 63.5 | 0.500 |
| A3 `mlp_frozen` | 0–8 | **0.0** | 61.0 | 0.483 |
| A4 `mlp_ste` | 0–8 | **1.0** | 62.5 | 0.493 |
| A0 `enabled=false` | no `kip_*` | *(none)* | — | — |

Two things worth keeping:

- **`mlp_frozen`'s span is exactly 0 on every clip measured.** Lesson **C24** /
  H4′ is re-confirmed independently in this campaign: v1's "motion-gated adaptive
  temporal shift" was a **constant** shift at `s = 61`, not merely a
  near-constant one. `mlp_ste` moves it to a span of 1.
- **`rank` spans the full 128 on every single clip**, as designed. The v3 gate
  does exactly what it was built to do. That is what makes §5 damning rather than
  inconclusive.

---

## 3. Headline table

DoTA is per-clip min-max micro AUC; MSAD is raw-pooled micro AUC. `macro` is the
mean of per-clip AUCs (1,391 two-class clips on DoTA, 96 on MSAD).

| Arm | seed | MSAD micro | MSAD macro | MSAD AP | DoTA micro | DoTA macro | DoTA AP |
|---|---|---|---|---|---|---|---|
| **A0** KIP-off | 2024 | 0.8924 | 0.7183 | 0.6595 | 0.5606 | 0.5632 | 0.3509 |
| | 2025 | 0.8857 | 0.6785 | 0.6531 | 0.5588 | 0.5533 | 0.3453 |
| | 2026 | 0.8824 | 0.6773 | 0.6568 | 0.5286 | 0.5192 | 0.3247 |
| **A2** plain-TSM | 2024 | 0.8873 | 0.6887 | 0.6698 | **0.6423** | 0.6603 | 0.4215 |
| | 2025 | 0.8926 | 0.7164 | 0.6758 | **0.6416** | 0.6591 | 0.4170 |
| | 2026 | 0.8914 | 0.6998 | 0.6488 | **0.6716** | 0.6939 | 0.4503 |
| **A1** rank (v3) | 2024 | 0.8834 | 0.6655 | 0.6375 | **0.5740** | 0.5686 | 0.3536 |
| **A2b** const+PMG | 2024 | 0.8834 | 0.6943 | 0.6403 | 0.6318 | 0.6352 | 0.4102 |
| **A3** mlp_frozen | 2024 | 0.8868 | 0.7115 | 0.6573 | 0.6519 | 0.6739 | 0.4309 |
| **A4** mlp_ste | 2024 | 0.8867 | 0.7136 | 0.6574 | 0.6548 | 0.6782 | 0.4346 |
| *V1* archived KIP-on | 2024 | 0.8868 | 0.7122 | 0.6574 | 0.6524 | 0.6748 | 0.4310 |
| | 2025 | 0.8862 | 0.7158 | 0.6469 | 0.6416 | 0.6550 | 0.4164 |
| | 2026 | 0.8815 | 0.7099 | 0.6454 | 0.6289 | 0.6292 | 0.4034 |
| *gate_a* `best.ckpt` | — | 0.8949 | 0.7177 | 0.6432 | 0.6012 | 0.6159 | 0.3776 |

---

## 4. The decisive row: A2 − A0

Per §7 of the runbook the primary statistic is the **seed-level t-interval over
n = 3**, not a count of bootstrap CIs.

| Comparison | per-seed ΔAUC | mean ± sd | **t95 (n=3)** |
|---|---|---|---|
| **A2 − A0**, DoTA micro | +0.0817, +0.0828, +0.1430 | **+0.1025 ± 0.0350** | **[+0.0154, +0.1896]** |
| A2 − A0, DoTA macro | +0.0970, +0.1058, +0.1747 | +0.1259 ± 0.0425 | [+0.0202, +0.2316] |
| A2 − A0, DoTA AP | +0.0706, +0.0718, +0.1255 | +0.0893 ± 0.0314 | [+0.0113, +0.1673] |
| *V1 KIP-on − A0*, DoTA micro | +0.0918, +0.0829, +0.1002 | +0.0916 ± 0.0087 | [+0.0700, +0.1132] |

Per-seed paired bootstraps over clips (2,000 resamples) for A2 − A0 micro:
[+0.0690, +0.0941], [+0.0722, +0.0943], [+0.1317, +0.1546] — all exclude zero,
P(Δ>0) = 1.000 in all three.

**Read it as follows.**

1. **A plain 50 % temporal smoother, with the motion pathway provably removed,
   moves DoTA by +0.10 micro AUC.** The t-interval excludes zero on all three
   DoTA metrics.
2. **The archived V1 arm reproduces at +0.0916 ± 0.0087, t95 [+0.0700, +0.1132]**
   — the number `activeContext` and `RESULTS_PHASE_A.md` record. Our pipeline is
   validated against a known quantity before being used to make a new claim.
3. **A2's interval is wide (sd 0.0350) and the width is A0's fault, not A2's.**
   Seed 2026's A0 collapses to 0.5286 while A2 stays at 0.6716; A2's own spread
   across seeds is 0.6423 / 0.6416 / 0.6716. Do not quote "+0.1025" as though it
   were a tighter measurement than the v1 number — the honest form is §5's
   equivalence test, which pairs matched seeds on both sides.

---

## 5. The equivalence that settles the question

If the smoother *is* the mechanism, then A2 and the full v1 KIP should be
indistinguishable. Paired on matched seeds:

| A2 − V1 KIP-on | per-seed Δ | mean | t95 (n=3) | verdict |
|---|---|---|---|---|
| DoTA micro | −0.0101, −0.0000, +0.0427 | **+0.0109** | **[−0.0588, +0.0805]** | **includes zero** |
| DoTA macro | −0.0145, +0.0041, +0.0647 | +0.0181 | [−0.0848, +0.1210] | includes zero |
| DoTA AP | −0.0095, +0.0006, +0.0469 | +0.0127 | [−0.0620, +0.0873] | includes zero |
| MSAD micro | +0.0005, +0.0064, +0.0098 | +0.0056 | [−0.0061, +0.0173] | includes zero |
| MSAD AP | +0.0124, +0.0289, +0.0034 | +0.0149 | [−0.0172, +0.0470] | includes zero |

**Six of six metrics include zero.** Seed 2025 agrees to 4·10⁻⁵.

A2 differs from V1 KIP-on by: no RAFT flow targets, no `L_KIP_rec`, no
`L_KIP_align`, no `L_kin`, no stage-1 warm-up, and a gate that ignores `ê_O`
entirely. **Everything KIP was built to do is absent from A2, and DoTA cannot
tell them apart.**

This is an equivalence claim at n = 3, so state it with its power: the interval
admits differences up to ±0.08 micro AUC, which is the same order as the effect
itself. It rules out "the motion pathway contributes most of the +0.09"; it does
not rule out "the motion pathway contributes a little". What it does do is remove
any basis for attributing the +0.09 *to* the motion pathway — which is what three
campaigns and the proposal have claimed.

---

## 6. The v3 rank gate is worse than no gate at all

A1 is the arm v3 was built to deliver: a parameter-free, full-range,
ECMR-residual rank gate whose `s_t` provably spans all 128 channels on every clip
(§2.1). It is the only arm that could support a motion-gating claim.

| A1 rank − X, DoTA micro, seed 2024 | Δ | bootstrap 95 % CI | P(Δ>0) |
|---|---|---|---|
| **− A2 plain-TSM** | **−0.0683** | **[−0.0790, −0.0580]** | **0.000** |
| − A3 `mlp_frozen` (v1) | −0.0779 | — | — |
| − A0 KIP-off | +0.0134 | [+0.0015, +0.0254] | 0.987 |

Per-clip, A1 vs A2 over 1,391 two-class clips: mean **−0.0916**, median −0.0625,
**60.0 % of clips worse**, 26.7 % better. This is not a tail effect.

A1 also loses in-domain. On MSAD it is 0.8834 vs A0's 0.8924, macro 0.6655 vs
0.7183 (the worst macro of any arm), and **A1 − A2 on MSAD AP is −0.0323, CI
[−0.0602, −0.0029] — excluding zero**, the only in-domain comparison in this
campaign that does. On DoTA it is the only arm whose ΔAP against A0 includes zero
(+0.0028, CI [−0.0061, +0.0117]). The rank gate is the one intervention here that
is measurably harmful on *both* benchmarks.

**So the ranking is: fixed smoother > v1 near-constant gate ≫ v3 full-range
gate > no gate.** Making the gate actually vary with motion evidence *removed*
most of the benefit. That is the opposite of the v3 design hypothesis, and it is
the strongest single piece of evidence that what helps here is smoothing, not
motion conditioning: the more the shift count moves around, the worse the result.

**Caveat, stated plainly: A1 is n = 1 seed.** The bootstrap CI is over clips, not
seeds. The direction is unambiguous at seed 2024 and the per-clip loss rate makes
seed luck an implausible explanation for a −0.068 gap, but **A1 does not have a
seed-level result** and must not be reported as one.

---

## 7. The rest of the ladder

| Comparison | Δ DoTA micro (s2024) | bootstrap 95 % CI | What it says |
|---|---|---|---|
| **A3 − A0** (v1 bridge) | **+0.0913** | [+0.0806, +0.1023] | Lands on the archived **+0.0911 / +0.0918**. The bridge to every `RESULTS_*.md` number holds |
| **A3 − A2** (v1 gate vs constant) | +0.0096 | **[−0.0011, +0.0204]** — *includes zero* | **The trained 321-param gate buys nothing measurable over an explicit constant.** Exactly what a span of 0 predicts |
| **A2b − A2** (auxiliary losses) | **−0.0105** | **[−0.0207, −0.0011]**, P>0 = 0.016 | Training the PMG head against **real flow**, with all three KIP losses live, is **significantly worse** than not having it |
| **A4 − A3** (pure gradient) | +0.0029 | [+0.0019, +0.0039] | Unblocking the backward pass is a real but tiny effect. Forward is near-identical (span 0 → 1) |
| A2 − gate_a | +0.0411 | [+0.0250, +0.0563] | The smoother alone beats LaGoVAD's released trunk — and at **n = 3** this is +0.0506 ± 0.0171, **t95 [+0.0081, +0.0932]**, excluding zero |
| A1 − gate_a | **−0.0272** | [−0.0438, −0.0107] | The v3 model **loses to** the released trunk |

Two of those rows carry the argument. **A3 − A2's CI includes zero**: the gate v1
shipped is worth nothing over hard-coding its output. **A2b − A2's CI excludes
zero on the negative side**: adding the flow targets and all three auxiliary
losses back *costs* AUC. Between them they say the motion pathway is not merely
unproven — where it has been measured, it is neutral or harmful.

The whole KIP apparatus — flow extraction, the 311,808-param PMG head, three
auxiliary losses, a stage-1 warm-up — is worth **−0.0105** (A2b − A2) once the
gate is held constant.

**A2 is the cheapest arm in the program and it is not distinguishable from the
best.** At seed 2024 it ranks fourth of six on the point estimate (A4 0.6548 >
V1 0.6524 > A3 0.6519 > **A2 0.6423** > A2b 0.6318 > A1 0.5740), but the gap to
A3 has a CI that includes zero, and the two arms above A3 are A3 plus a +0.003
gradient effect. At seed 2025 it ties V1 to 4·10⁻⁵; at seed 2026 it beats V1 by
+0.0427. The correct summary is not "A2 wins" — it is "**nothing that costs more
than A2 reliably beats A2**".

---

## 8. MSAD stays a bounded null — including for the smoother

| Comparison | per-seed ΔAUC | mean ± sd | t95 (n=3) |
|---|---|---|---|
| A2 − A0, MSAD micro | −0.0051, +0.0069, +0.0090 | +0.0036 ± 0.0076 | [−0.0152, +0.0225] |
| A2 − A0, MSAD macro | −0.0295, +0.0378, +0.0225 | +0.0103 ± 0.0353 | [−0.0774, +0.0979] |
| A2 − A0, MSAD AP | +0.0103, +0.0226, −0.0079 | +0.0083 ± 0.0154 | [−0.0299, +0.0465] |
| V1 − A0, MSAD micro | −0.0056, +0.0005, −0.0008 | −0.0020 ± 0.0032 | [−0.0100, +0.0060] |

Every interval includes zero and the sign is not stable. The honest statement is
unchanged from `RESULTS_PHASE_A.md`: **any in-domain effect is smaller than
≈ 2 AUC points at n = 3** — a looser bound than the ±1 point the v1 arm supported,
because A2's seed spread is larger. Not "costs nothing".

The MSAD-null / DoTA-positive asymmetry survives — but it now belongs to a
**temporal smoother**, not to a motion pathway. It is evidence about DoTA (short,
ego-centric, motion-dominated clips where neighbouring-frame mixing helps
localization) rather than evidence about KIP.

**The reproduction gate still passes.** As redefined 2026-08-16 (lesson **8b**) —
"our arms match the released checkpoint under one identical protocol" — A2 vs
`gate_a` on MSAD is **−0.0045 ± 0.0027, t95 [−0.0113, +0.0024]** over three
seeds, including zero, with all three per-seed bootstrap CIs also including zero.
AP is *higher* for A2 (+0.0216). Our port remains statistically indistinguishable
from the checkpoint the authors shipped, and 0.9041 stays out of reach for
`best.ckpt` itself (0.8949 here).

**On DoTA the same comparison is positive and seed-level:** A2 − `gate_a` =
+0.0506 ± 0.0171, t95 [+0.0081, +0.0932]. A 50 % channel shift on top of our
trunk beats LaGoVAD's released checkpoint on its own transfer benchmark. That is
the *result* worth keeping from this campaign — it just is not a motion result.

---

## 9. Mechanism: D4 refuted again, now 3/3 for the smoother too

Δ micro AUC vs A0 on the ego / non-ego split (802 / 594 clips):

| Arm | seed | ego | other | |
|---|---|---|---|---|
| **A2** | 2024 | +0.0433 | **+0.1326** | other > ego |
| | 2025 | +0.0505 | **+0.1260** | other > ego |
| | 2026 | +0.1238 | **+0.1690** | other > ego |
| | **mean** | **+0.0725** | **+0.1425** | |
| V1 KIP-on | mean of 3 | +0.0712 | +0.1191 | other > ego |
| A1 rank | 2024 | **−0.0151** | +0.0519 | other > ego |
| A2b | 2024 | +0.0483 | +0.1018 | other > ego |
| A3 | 2024 | +0.0789 | +0.1091 | other > ego |
| A4 | 2024 | +0.0811 | +0.1129 | other > ego |

**`other` > `ego` in 10 of 10 arm-seeds. Do not claim ego-kinematics** — this is
now settled across two independent campaigns and a control that has no kinematic
pathway at all. Note A1's ego delta is *negative*: the full-range rank gate
actively hurts on the ego half.

Critically, **A2's ego/other signature (+0.0725 / +0.1425) is the same shape as
V1's (+0.0712 / +0.1191).** A control with no motion evidence reproduces the
mechanism signature that was previously offered as evidence *for* a motion
mechanism. That signature was never diagnostic.

Per-class (A2 − A0, mean of 3 seeds; classes with n ≥ 15), the gain is broad, not
localized: every one of 18 classes improves, from `ego: unknown` +0.031 to
`other: leave_to_right` +0.170. `other:` classes take the top four slots. A1 −A0
is *negative* on 9 of 18 classes.

Per-clip win rates for A2 − A0: 61.7 % / 61.4 % / 72.8 % win against 26.5 % /
26.1 % / 15.4 % loss across the three seeds — a broad shift, unlike the coin-flip
distribution the MSAD arms showed.

---

## 10. What this does to the thesis

**Confirmed, as pre-registered on 2026-08-30 before this eval ran:** `r = 0.5`
reproduces most of the +0.09. Per setup §2.2 the consequence was written down in
advance — *"the contribution is temporal smoothing and the thesis needs
restating."*

What is now established:

1. **The +0.09 DoTA effect is real, replicated, and survives every control.** It
   is not seed luck (n = 3, two independent arms), not the crop (`RESULTS_NCC`),
   not the warm start (`RESULTS_ARM4_PROBE`, and A2 has no warm start at all), and
   not under-convergence (H3 rejected).
2. **Its cause is a fixed ~50 % temporal channel shift.** A control containing
   nothing but that shift is statistically indistinguishable from full KIP on six
   of six metrics.
3. **No component of KIP has been attributed any positive contribution.** The
   auxiliary losses are −0.0105 (A2b − A2). The trained gate is +0.0096 over a
   constant (A3 − A2). The parameter-free full-range gate is −0.0683 (A1 − A2).
   The gradient fix is +0.0029 (A4 − A3).
4. **KIP as specified — "induce optical-flow motion evidence at train time to
   close LaGoVAD's motion weakness" — is not what produced the result.** The
   honest framing is: *a temporal shift module (TSM) applied between LaGoVAD's
   temporal encoder and its fusion buys ≈ +0.09 zero-shot DoTA micro AUC at zero
   inference cost on the score path and no in-domain penalty within ±0.02.* That
   is a smaller, cleaner, and defensible claim, and it is fully supported here.

Two honest readings of what to do with it, in the order I would put them:

- **Restate the contribution as temporal smoothing and own it.** It is a real,
  replicated, cheap, RGB-only zero-shot transfer gain over a strong baseline,
  with the mechanism *correctly identified* and the negative ablations reported.
  Papers are made of this. It requires dropping "kinematics-aware" from the
  method's identity, and it makes the RAFT pipeline, PMG head and three
  auxiliary losses dead weight to be deleted rather than defended.
- **Keep pursuing motion conditioning, but from zero.** Nothing in three
  campaigns supports it, and the one arm that genuinely varies the shift with
  motion evidence (A1) is the worst arm measured. That is a new project, not a
  repair, and it should not reuse the +0.09 as a starting point (lesson **14**).

**Do not tune A1 to beat A2.** That is exactly fitting the benchmark, and the
+0.09 is now known not to be a motion effect.

---

## 11. Limitations — read before citing

1. **A1, A2b, A3, A4 are n = 1 seed.** Only A0, A2 and V1 have three. Every
   single-seed Δ in §6–§7 carries a clip-level bootstrap CI, which is *not* a
   seed-level interval. §6's conclusion (A1 ≪ A2) is directionally strong but
   formally unreplicated.
2. **The n = 3 equivalence in §5 has ±0.08 resolution.** It refutes "motion is
   most of the effect", not "motion contributes nothing".
3. **Every threshold metric is void.** 84.5–89.3 % of DoTA frames score > 0.99
   (median 0.9993–0.9998). Micro/macro AUC and AP are rank-based and sound; any
   accuracy, F1, MCC or fixed-threshold number computed off these curves is not.
   `gate_a` sits at 0.0 % > 0.99 — that gap is a calibration difference between
   the released trunk and every arm trained here, and it is unexplained.
4. **All arms read `checkpoint_last`, deep in the overfit regime** (train `mil`
   0.001–0.010). No validation split, no model selection (`msad-ncc-seeds-and-selection.md`
   Phase B, deferred). The confound is shared by all arms and largely cancels in
   Δ, but it caps every absolute number.
5. **A2's seed-2026 A0 comparator is weak** (0.5286 vs 0.5606/0.5588), which
   inflates the A2 − A0 mean and its sd. §5's matched-seed equivalence is the
   robust framing; §4's +0.1025 is not tighter than V1's +0.0916.
6. **The rank gate resets at each `data.max_vis_len` window boundary** (setup
   §6.8). That is a real property of A1's scored model, not a logging artifact,
   and it is a candidate explanation for §6 that this campaign does not test.
7. **Stock CLIP ViT-B/16 throughout.** Alert-CLIP is deferred — no public
   checkpoint exists.
8. **`e_O` has 23 effective dimensions**, frame-global scalars lifted to 256-d by
   a fixed seeded projection. Nothing here localizes motion spatially, and no
   claim below should imply it.
9. **No *training*-side run manifest** (lesson C17). `--init-weights` is absent
   from every `config.yaml`, so the warm start in §2 is inferred from loss
   trajectories rather than read from a file. The eval side is now covered: the
   16 `eval_manifest.json` files from setup §6.11 were **retro-filled on
   2026-09-01**, each recording the arm, gate type, checkpoint, caches, score
   norm and its §6.8 span with the assert re-run at write time. `rank` vs
   `constant` provenance therefore rests on those manifests plus the gate check —
   not on anything the training run itself wrote.

---

## 12. Defects found in this campaign

| # | Defect | Impact | Lesson |
|---|---|---|---|
| 1 | `evaluate --save-scores` wrote a **0-byte `.npz`** (A1 DoTA) | 1 clip / 18,369 frames dropped from all arms. Silent — `exists()` calls it done | **C11**, extended to a second writer. `evaluate.py` should use `core/tools/feature_cache.py`'s atomic write |
| 2 | `results.json` metrics are **float32**-normalised | ΔAUC 0.0002 / ΔAP 0.0016 vs float64 on identical files; reproduced to the last digit | **C22**, still open in `normalize_scores` |
| 3 | `metrics.jsonl` for A3 stage 2 held **665 rows for 500 steps** (steps 1–165 twice, from a restart) | Any loss-trajectory read is wrong by 33 %. Duplicate copies agreed to 1e-6, so no corruption | Runbook §5.3 append behaviour; **deduped 2026-09-01**, `.jsonl.bak` retained |
| 4 | `--init-weights` still unrecorded | Warm start had to be inferred from loss values | **C17**, still open, now HIGH |
| 5 | One DoTA clip has a **flat score curve** (`W6YrlYyWguc_005367`) | Division by range without an eps guard yields NaN | `SCORE_NORM_EPS` exists for this; any offline analysis must branch on `hi > lo` |

---

## 13. What to run next

**Attribution is now done. The queue below is about consolidating a restated
claim, not about rescuing the old one.**

1. **Two more seeds for A1** (`rank`, 2025 + 2026 — needs §5.1a stage 1 for each).
   The one number that promotes §6 from "decisive at n = 1" to a result. It is
   also the only remaining way the motion-gating story could survive, so it
   should be run before any write-up asserts §6.
2. **Two more seeds for A3** (`mlp_frozen`), giving the v1 bridge a seed-level
   interval to sit beside A2's. Cheap, and it is what lets §5's equivalence be
   restated against a *v3-tree* arm rather than the archived one.
3. **Fix the two open metric defects** — atomic writes in `evaluate --save-scores`
   (C11) and float64 in `normalize_scores` (C22) — then re-`rescore --write`
   every run dir so `results.json` and any offline table agree by construction.
4. **The run manifest** (C17, ~30 LOC). Nothing else can distinguish a `rank` arm
   from a `constant` arm after the fact, and this campaign proves the gate check
   is the only current substitute.
5. **A2 at other ratios** (`const_shift_ratio` ∈ {0.125, 0.25, 0.75}) — the
   natural follow-up to a smoothing claim, and it characterises the *actual*
   mechanism instead of defending the abandoned one. Pre-register the reading:
   a smooth single-peaked curve in `r` supports smoothing; a flat one says
   something else is going on.
6. **Then decide the deletion.** If §13.1–2 hold, `core/flow/`, the PMG head,
   `L_KIP_rec`, `L_KIP_align` and `L_kin` are unused weight on the training path.
   Removing them is a large simplification — but it deletes the only
   infrastructure a future motion project would need, so it is the user's call,
   not a cleanup.

**Do not** retro-edit the existing `RESULTS_*.md` numbers. They were true for
what ran (`mlp_frozen`, which §2.1 now shows was a *strictly* constant shift);
add the qualifier at the point of next citation.

---

## 14. Provenance

- **Score curves:** `outputs/v3/{MSAD,DoTA}_{kipoff,rank,constant,constant_pmg,mlp_frozen,mlp_ste}_s2024/eval_*`,
  `outputs/v3/{MSAD,DoTA}_constant_s{2025,2026}/eval_*`,
  `outputs/{MSAD,DoTA}_ncc{,_s2025,_s2026}/eval_kip_{off,on}`,
  `outputs/MSAD_ncc/full_gate_a`, `outputs/DoTA_ncc/gate_a`.
  **`outputs/` is gitignored — this document is the durable record.**
- **`gate_a` was reused, not re-run.** Setup §6.9 asks for a fresh one; the v3
  diff to `core/models/kat_vad.py` is entirely KIP-scoped (a `training=` flag and
  `return_kip_diagnostics`), so a `kip.enabled=false` forward pass is unchanged
  and the archived arms are on the identical features, frame count and
  checkpoint. Verified by the §1.3 A0 reproduction on the same code paths.
- **Metrics:** recomputed float64 from `.npz`; DoTA per-clip min-max, MSAD raw;
  paired bootstrap 2,000 resamples over clips, `default_rng(0)`; seed-level
  t-intervals with `t₀.₉₇₅,₂ = 4.303`.
- **Supersedes nothing.** `RESULTS_NCC.md`, `RESULTS_PHASE_A.md` and
  `RESULTS_ARM4_PROBE.md` stand as measured; this document reinterprets what
  their +0.09 was caused by, and reproduces their central number (+0.0916 ±
  0.0087) as a validation check on its own pipeline.
