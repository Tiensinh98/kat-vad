# RESULTS — PreVAD: Gate P0, the PreVAD trunk, and the transfer campaign

**Compiled:** 2026-08-28 · **Scope:** every artifact under `outputs/PreVAD/**`,
`outputs/MSAD_ncc_pv_s{2024,2025,2026}/**` and `outputs/DoTA_ncc_pv_s{2024,2025,2026}/**`.
**Source of every number:** `results.json`, `scores/*.npz`, `metrics.jsonl`,
`config.yaml`, `PreVAD/{train,test}.csv`, `data/{MSAD,DoTA}/*`.
Nothing here was copied from a paper or estimated.

**Reads after:** `RESULTS_ARM4_PROBE.md` → `REPORT_KIP_MSAD_DOTA_PREVAD.md`.
**Protocol:** `PREVAD_SETUP.md` (§4.5 defines Gate P0), `DOTA_EVAL.md`, `TRAINING.md`.

---

## 0. Verdict, up front

Three things were measured. Two are clean results. One is **blocked** and must
not be reported as a KIP finding.

| # | Claim | Status |
|---|---|---|
| **1** | **Gate P0 passes.** The released `best.ckpt` scores **0.9031** micro AUC on PreVAD test through our eval; our PreVAD-trained trunk scores **0.9007** — Δ −0.0025, CI [−0.0103, +0.0047], and all three metric CIs include zero. The released `ViT-B-16-8p` features are compatible with our pipeline, and our port is indistinguishable from the released checkpoint on a **third** benchmark. | **Clean** |
| **2** | **PreVAD pretraining transfers.** Replacing the MSAD stage-1 trunk with a PreVAD stage-2 trunk moves the **KIP-off baseline** by **+0.0474 AP** on MSAD (seed-level t95 [+0.0318, +0.0630]) and **+0.0376 micro AUC** on DoTA (all three clip-level CIs exclude zero, sign stable). Our KIP-off arm now **beats the released checkpoint** on MSAD: AUC +0.0039, AP +0.0604, both seed-level intervals excluding zero. | **Clean** |
| **3** | ~~"The +0.09 DoTA gain disappears under a PreVAD trunk."~~ **Δ(on − off) is −0.0024 ± 0.0159 on DoTA and −0.0031 ± 0.0014 on MSAD — but the KIP module in these runs was never trained.** `L_KIP_rec` ends at 10.2–13.3 against the cold campaign's 4.91; `L_KIP_align` never moves off its initial 4.61 and sits **above** the 4.265 chance level. | **BLOCKED — lesson C14** |

**The one-line reading:** this campaign is an excellent trunk experiment and a
**void KIP ablation**. The PreVAD corpus is worth what it cost. The A/B in it is
not a refutation of the +0.09 — it is a measurement of a KIP whose flow head
still has its initialisation.

What it *does* establish, as a negative control: **the KIP architecture with an
untrained PMG head buys nothing.** The forward-pass wiring alone — PMG head,
frozen gate, adaptive shift, all present and active — moves DoTA by
−0.002 ± 0.016. Only a *trained* flow-regression head produced +0.09. That is
evidence against the strong form of **H4** ("KIP is a fixed temporal smoother
whose content does not matter"), stated with the confound in §7 attached.

---

## 1. What actually ran

### 1.1 The design, recovered from `collab/PreVAD/evaluate.py`

```
PreVAD train (32,673 clips)
  └─ stage 2, KIP-off, seed 2024, 40 epochs = 13,360 steps   →  outputs/PreVAD/stage2_kip_off
       │
       ├─ eval on PreVAD test (2,606 clips)                  →  outputs/PreVAD/eval_trunk
       │
       └─ graft: trunk tensors + a fresh KIP-enabled skeleton
          supplies randomly-initialised kip.*                →  checkpoint_last_withkip.pt
               │
               ├── MSAD-full stage 2, KIP-ON  (--init-weights …_withkip.pt)   ┐
               └── MSAD-full stage 2, KIP-OFF (--init-weights …_last.pt)      ├ × seeds 2024/2025/2026
                          40 epochs = 160 steps                               ┘
                              │
                              ├─ eval MSAD-full (in-domain)  → outputs/MSAD_ncc_pv_s{seed}/eval_kip_{on,off}
                              └─ eval DoTA     (zero-shot)   → outputs/DoTA_ncc_pv_s{seed}/eval_kip_{on,off}
```

Separately, the released LaGoVAD `best.ckpt` was run on PreVAD test →
`outputs/PreVAD/gate_p0` (Gate P0, `PREVAD_SETUP.md` §4.5).

### 1.2 What is *not* on disk — read this before quoting the campaign

- **There is no PreVAD KIP-on arm, and there never will be.** `outputs/PreVAD/`
  contains `stage2_kip_off` only. PreVAD ships CLIP features and no pixels, so
  the RAFT targets `L_KIP_rec` needs cannot be built — `PREVAD_SETUP.md` §7.4,
  settled by the user 2026-08-29. PreVAD is a **pretraining corpus and an
  in-domain reproduction gate**, permanently not a KIP A/B benchmark.
- **There is no per-seed PreVAD evaluation.** PreVAD test was scored exactly
  twice: `gate_p0` (released checkpoint) and `eval_trunk` (our seed-2024 trunk).
  n = 1, one arm. Any sentence of the form "KIP on PreVAD across 3 seeds" is
  unsupported by these artifacts.
- The 3 seeds × 2 arms live entirely in the **MSAD-full finetune**, and the
  evaluations are MSAD (in-domain) + DoTA (zero-shot).

### 1.3 Config parity — verified, and it is not clean

Within the campaign, parity is perfect: `config.yaml` for all six MSAD finetunes
is byte-identical once `train.seed` is normalised. Both arms of each seed pair
warm-start from the same PreVAD trunk (verified from the loss, §1.4).

Against the **cold campaign** the configs differ in three places, and this is
what blocks claim 3:

| field | cold (`MSAD_ncc*`) | this campaign (`MSAD_ncc_pv_*`) |
|---|---|---|
| trunk source | MSAD **stage 1** (KIP-only warm-up, 500 steps) | PreVAD **stage 2** (KIP-off, 13,360 steps) |
| `train.num_epochs` | 125 (**500 steps**) | 40 (**160 steps**) |
| `train.amp` | `true` | `false` |
| `train.checkpoint_every_steps` | 100 | 0 (no intermediate checkpoints) |

Every `kip.*`, `model.*`, `loss.*`, `dvs.*` and `data.*` field is identical.

**Three variables moved at once.** The trunk changed, the KIP stage-1 warm-up was
removed, and the step budget was cut 3.1×.

### 1.4 Warm start verified from the loss, not from the config

`--init-weights` is still absent from `config.yaml` (limitation 11 of the report,
lesson **C17** — still unfixed). Step-1 `mil` is the evidence:

| arm | seed 2024 | 2025 | 2026 |
|---|---|---|---|
| **PreVAD-trunk, KIP-off** | 0.1363 | 0.4282 | 0.5701 |
| **PreVAD-trunk, KIP-on** | 0.2021 | 0.4021 | 0.5317 |
| cold MSAD stage-1 trunk, KIP-on | 0.7015 | 0.8163 | — |
| cold, no warm start (KIP-off) | 0.8682 | 0.6853 | — |

Both PreVAD arms start well below every cold arm. The graft happened.

---

## 2. The PreVAD dataset, as measured

| Property | Train | Test |
|---|---:|---:|
| Clips | 32,673 | 2,606 |
| Normal clips | 22,000 (67.3 %) | 1,300 (**49.9 %**) |
| Abnormal clips | 10,673 | 1,306 |
| Distinct `class_name` | 34 | 35 |
| Distinct `superclass_name` | 8 | 8 |
| Sampled frames scored | — | 190,558 |
| Clips with both classes (macro AUC basis) | — | 1,078 |

Superclasses: Normal (22,000), Violence (2,889), Vehicle Accident (2,140),
Daily Accident (1,400), Fire-related Accident (1,177), Production Accident
(1,135), Animal-related Violence (1,096), Robbery (836).

**PreVAD test is 49.9 % normal**, so `--score-norm auto` resolves to **raw
pooling** — the MSAD protocol, not DoTA's. That is the correct call under lesson
**C12** (the rule reads the label distribution, not the dataset name), and
`results.json` records `score_norm: none` for both PreVAD runs. Do not min-max
PreVAD.

### 2.1 The label format — lesson C18 reproduces exactly

Re-counted from `PreVAD/test.csv`:

| | count |
|---|---:|
| total spans | 1,427 |
| clips with 0 / 1 / 2 / 3 / 4 spans | 1,300 / 1,202 / 88 / 15 / 1 |
| **multi-span clips** | **104** |
| **spans ending past 1.0** (max 1.2104) | **440** |
| **reversed spans** (`end < start`) | **1** |

C18's numbers are confirmed against the shipped CSV. The hull-taking failure
mode it warns about is real for 104 test clips, and `core/data/prevad.py`
fills every span independently and clamps to `[0, L]`.

---

## 3. Gate P0 — the compatibility gate passes

`PREVAD_SETUP.md` §4.5 defines P0 as a *sanity* gate — "high in-domain AUC"
passes, "≈ 0.5" fails and blocks everything downstream. There is no published
PreVAD-test AUC, so per lesson **C8b** the reference is the released checkpoint,
never a printed number.

| arm | checkpoint | micro AUC | AP | macro AUC (n = 1,078) |
|---|---|---|---|---|
| **Gate P0** | released `best.ckpt` | **0.9031** | **0.6910** | **0.6721** |
| **eval_trunk** | our PreVAD stage-2 KIP-off, seed 2024 | **0.9007** | **0.6902** | **0.6709** |

Raw pooling, 2,606 clips, 190,558 sampled frames, identical protocol and
identical ground truth for both arms.

**P0 passes by a wide margin** (0.9031 vs a 0.5 failure threshold). The released
`ViT-B-16-8p` features are compatible with our loaders, our label builder and our
eval. Steps 9–13 of `PREVAD_SETUP.md` §10 were unblocked retroactively — they
were run before P0 was recorded, which was a process error that happened to cost
nothing.

**And our trunk reproduces the checkpoint on a third benchmark.** Paired clip
bootstrap, our trunk minus `best.ckpt`, 2,606 clips:

| metric | Δ | 95 % CI |
|---|---:|---|
| micro AUC | −0.0025 | [−0.0103, +0.0047] |
| AP | −0.0008 | [−0.0266, +0.0232] |
| macro AUC | −0.0012 | [−0.0135, +0.0105] |

**All three CIs include zero.** Our port is statistically indistinguishable from
the released checkpoint on PreVAD, as it already was on MSAD. That is lesson
**C8b**'s gate passing on a third benchmark — and it is a stronger statement than
P0 required, since P0 only asked for "not chance".

### 3.1 Per-superclass, our trunk vs the released checkpoint

Mean per-clip AUC over clips containing both classes:

| superclass | n | `best.ckpt` | our trunk | Δ |
|---|---:|---:|---:|---:|
| Animal-related Violence | 71 | 0.5537 | 0.6180 | **+0.0642** |
| Fire-related Accident | 70 | 0.7706 | 0.8121 | **+0.0415** |
| Robbery | 116 | 0.6673 | 0.6669 | −0.0004 |
| Daily Accident | 174 | 0.6386 | 0.6287 | −0.0099 |
| Vehicle Accident | 232 | 0.6914 | 0.6816 | −0.0098 |
| Production Accident | 181 | 0.6862 | 0.6751 | −0.0111 |
| Violence | 234 | 0.6760 | 0.6644 | −0.0116 |
| **all** | **1,078** | **0.6721** | **0.6709** | **−0.0012** |

Mixed and small in every bucket. Nothing here is a result; it is a
distribution check, and it is consistent with "the two models are the same model".

**The easy classes are motion-textured, the hard ones are not** — Fire 0.877,
Mugging 0.816, Collapse 0.799, Explosion 0.767 at the top; Predation 0.510,
Fall into Water 0.534, Robbery 0.571, Daily Accident 0.576 at the bottom
(per-clip AUC, classes with n ≥ 15). Worth remembering when PreVAD is proposed
as the second transfer benchmark: the model's competence on it is uneven.

---

## 4. The PreVAD trunk — training

40 epochs, 13,360 steps, batch 64, lr 5e-5, seed 2024, `kip.enabled=false`,
`loss.captions_from_definitions=true`. 20-step trailing means:

| loss | first 20 | mid | last 20 |
|---|---:|---:|---:|
| `mil` | 1.4706 | 0.1878 | **0.0764** |
| `dvs_sup` | 1.4923 | 0.1224 | 0.1009 |
| `dvs_sup_mil` | 1.5802 | 0.1039 | 0.0637 |
| `mul_mil` | 6.9418 | 0.9677 | 0.7952 |
| `cap_contrastive` | 37.4637 | 2.8392 | 2.4943 |
| `total` | 48.9486 | 4.2209 | 3.5305 |

Final `mil` of **0.0764** is an order of magnitude *less* fitted than the MSAD
stage-2 arms (0.0012–0.0088) — 32,673 clips over 34 classes is a much harder fit
than 240 test / ~700 train MSAD clips. That is the point of the corpus.

---

## 5. Results — MSAD-full, in-domain

Raw pooling, 240 videos, 18,350 sampled frames, `checkpoint_last`.

### 5.1 Levels

| seed | arm | micro AUC | AP | macro AUC (n=96) |
|---|---|---:|---:|---:|
| 2024 | KIP-off | **0.8979** | **0.6990** | 0.7060 |
| 2024 | KIP-on | 0.8946 | 0.6888 | 0.7045 |
| 2025 | KIP-off | **0.8995** | **0.7056** | 0.7107 |
| 2025 | KIP-on | 0.8950 | 0.6983 | 0.7274 |
| 2026 | KIP-off | **0.8990** | **0.7061** | 0.7162 |
| 2026 | KIP-on | 0.8973 | 0.7027 | 0.7262 |
| — | released `best.ckpt` (ncc) | 0.8949 | 0.6432 | 0.7177 |
| — | cold KIP-off, 3 seeds | 0.8922 / 0.8857 / 0.8824 | 0.6587 / 0.6531 / 0.6568 | — |
| — | LaGoVAD paper (printed, not reachable) | 0.9041 | — | — |

**These are the best MSAD numbers the project has produced.** KIP-off mean AUC
0.8988 (cold: 0.8868), mean AP 0.7036 (cold: 0.6562). The spread across seeds
collapses to 0.0016 AUC, against the cold campaign's 0.0098.

### 5.2 Against the released checkpoint — our arm now wins

Paired clip bootstrap, every arm against `outputs/MSAD_ncc/full_gate_a`:

| arm | Δ AUC | seed-level t95 | Δ AP | seed-level t95 |
|---|---:|---|---:|---|
| PreVAD-trunk KIP-off | **+0.0039** ± 0.0008 | **[+0.0019, +0.0058]** | **+0.0604** ± 0.0039 | **[+0.0506, +0.0702]** |
| PreVAD-trunk KIP-on | +0.0008 ± 0.0014 | [−0.0028, +0.0044] | **+0.0534** ± 0.0071 | **[+0.0358, +0.0711]** |

Every per-seed clip-bootstrap CI on AUC *includes* zero (240 clips is not much),
but the effect is so consistent across seeds that the seed-level interval
excludes it. **Caveat, and it is the audit's own warning:** the three deltas
share one reference arm, so n = 3 here measures the seed-to-seed stability of
*our* arm, not three independent replications of the comparison. Read it as
"consistently a little better, and clearly better on AP", not as a p-value.

Under lesson C8b the MSAD reproduction gate passed before. It now passes with
room: our PreVAD-pretrained baseline is **above** the artifact the authors
shipped on both AUC and AP, on an identical protocol.

### 5.3 Δ(KIP-on − KIP-off) — small, negative, and void as a KIP claim

| seed | Δ AUC | 95 % CI | Δ AP | 95 % CI | Δ macro |
|---|---:|---|---:|---|---:|
| 2024 | −0.0033 | [−0.0084, +0.0017] | −0.0102 | [−0.0262, +0.0075] | −0.0015 |
| 2025 | −0.0044 | [−0.0101, +0.0012] | −0.0073 | [−0.0280, +0.0144] | +0.0167 |
| 2026 | −0.0016 | [−0.0065, +0.0030] | −0.0034 | [−0.0281, +0.0205] | +0.0100 |
| **seed-level** | **−0.0031 ± 0.0014** | **[−0.0066, +0.0004]** | **−0.0070 ± 0.0034** | **[−0.0155, +0.0016]** | +0.0084 ± 0.0092 |

Every clip-level CI includes zero. The sign is now *stable* and negative in all
three seeds on both AUC and AP — unlike the cold campaign, where MSAD Δ flipped
sign. But see §7: this is an untrained KIP, so the honest reading is "adding an
untrained 444 k-parameter module and 160 steps of three extra loss terms costs a
consistent ~0.3 AUC points in-domain", which is unsurprising and not a KIP result.

### 5.4 Per-class, seed 2024

Mean per-clip AUC (classes with n ≥ 3 two-class clips):

| class | n | pv off | pv on | Δ(pv) | cold off | cold on |
|---|---:|---:|---:|---:|---:|---:|
| Explosion | 6 | 0.9687 | 0.9646 | −0.0041 | 0.9254 | 0.9105 |
| Vandalism | 5 | 0.8795 | 0.8583 | −0.0213 | 0.8195 | 0.7981 |
| Fire | 5 | 0.8729 | 0.8813 | +0.0084 | 0.8389 | 0.8384 |
| Fighting | 5 | 0.8733 | 0.8694 | −0.0038 | 0.8485 | 0.8595 |
| Shooting | 6 | 0.7156 | 0.7374 | +0.0218 | 0.7882 | 0.6798 |
| Object_falling | 8 | 0.6951 | 0.6977 | +0.0026 | 0.5740 | 0.6300 |
| People_falling | 22 | 0.6845 | 0.6771 | −0.0074 | 0.6359 | 0.6518 |
| **Traffic_accident** | 19 | 0.6431 | 0.6434 | +0.0003 | 0.6865 | 0.7065 |
| Robbery | 11 | 0.5823 | 0.5996 | +0.0173 | 0.7463 | 0.6911 |
| Assault | 7 | 0.5325 | 0.5024 | −0.0301 | 0.6328 | 0.6226 |

**The PreVAD trunk is not uniformly better in-domain.** It gains on
Explosion / Fire / Fighting / Object_falling / People_falling / Vandalism and
loses on Assault / Robbery / Shooting / **Traffic_accident** — the class KAT-VAD
is nominally aimed at drops from 0.687–0.707 to 0.643. n is 5–22 per class; this
is a direction to check, not a measured effect.

---

## 6. Results — DoTA, zero-shot

Per-clip min-max pooling (lesson C12), 1,397 clips, 18,369 sampled frames.
Clip ids and ground truth verified byte-identical to the cold campaign, so every
cross-campaign delta below is exactly paired.

### 6.1 Levels

| seed | pv KIP-off | pv KIP-on | Δ(on − off) | cold KIP-off | cold KIP-on | cold Δ |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | 0.5864 | 0.5668 | **−0.0196** | 0.5607 | 0.6519 | +0.0911 |
| 2025 | 0.5876 | 0.5995 | **+0.0120** | 0.5585 | 0.6416 | +0.0829 |
| 2026 | 0.5863 | 0.5866 | **+0.0003** | 0.5283 | 0.6288 | +0.1004 |

Released `best.ckpt` on the same protocol scores **0.6012** (seed-independent).
The `cold` columns are the §7 arms of `REPORT_KIP_MSAD_DOTA_PREVAD.md`, quoted as
published there — the seed-2025 delta recomputes to +0.0831 at full precision,
a 4th-decimal rounding difference in that report's table, immaterial here.

### 6.2 Δ(KIP-on − KIP-off) — null, sign-unstable

| seed | Δ AUC | 95 % CI | Δ AP | 95 % CI | Δ macro | 95 % CI |
|---|---:|---|---:|---|---:|---|
| 2024 | −0.0196 | [−0.0260, −0.0132] | −0.0144 | [−0.0200, −0.0091] | −0.0198 | [−0.0280, −0.0122] |
| 2025 | +0.0120 | [+0.0054, +0.0193] | +0.0072 | [+0.0010, +0.0136] | +0.0139 | [+0.0055, +0.0222] |
| 2026 | +0.0003 | [−0.0062, +0.0072] | −0.0023 | [−0.0080, +0.0036] | +0.0026 | [−0.0054, +0.0105] |
| **seed-level** | **−0.0024 ± 0.0159** | **[−0.0420, +0.0372]** | −0.0032 ± 0.0108 | [−0.0300, +0.0237] | −0.0011 ± 0.0172 | [−0.0437, +0.0415] |

This is the exact signature the audit taught us to distrust in the *other*
direction: **clip-level CIs excluding zero in two of three seeds while the sign
flips across seeds.** Seed-level, it is a flat null. Per-clip win rate falls to
0.344 / 0.448 / 0.403 — KIP-on loses to KIP-off on 55–66 % of clips.

### 6.2b Against the released checkpoint on DoTA — both arms lose

`outputs/DoTA_ncc/gate_a` (released `best.ckpt`, ncc) scores **0.6012** micro
AUC. Paired against it:

| arm | Δ AUC (seed-level) | t95 | Δ macro | t95 |
|---|---:|---|---:|---|
| PreVAD-trunk KIP-off | −0.0145 ± 0.0007 | **[−0.0162, −0.0127]** | −0.0143 ± 0.0009 | **[−0.0166, −0.0120]** |
| PreVAD-trunk KIP-on | −0.0169 ± 0.0165 | [−0.0578, +0.0240] | −0.0154 ± 0.0180 | [−0.0602, +0.0294] |

The PreVAD trunk closes most, **but not all**, of the cold campaign's gap to the
released checkpoint on DoTA (cold KIP-off was ~0.04 below it). It does not reach
it, and the cold KIP-on arm — at 0.6288–0.6519 — still beats `best.ckpt` by a
margin no arm in this campaign approaches.

### 6.3 D4 (ego vs other) is uninformative here

Mean per-clip AUC gain Δ(on − off), split on the `anomaly_class` prefix:

| campaign | seed | ego (n=802) | other (n=590) | clip win rate |
|---|---|---:|---:|---:|
| **PreVAD trunk** | 2024 | −0.0155 | −0.0257 | 0.344 |
| | 2025 | −0.0020 | +0.0354 | 0.448 |
| | 2026 | −0.0074 | +0.0162 | 0.403 |
| cold | 2024 | +0.0969 | +0.1296 | 0.632 |
| | 2025 | +0.0734 | +0.1403 | 0.627 |
| | 2026 | +0.0873 | +0.1411 | 0.634 |

`other` is more extreme than `ego` in whichever direction the seed happens to
go — negative in 2024, positive in 2025/2026. That is a **gain-amplitude** effect,
not an ego/other mechanism. It neither supports nor refutes D4; with the overall
effect at zero there is nothing to split.

### 6.4 Saturation improved a great deal

Fraction of sampled frames scoring > 0.99:

| run | > 0.99 | < 0.01 | mean |
|---|---:|---:|---:|
| cold KIP-on (s2024, DoTA) | 0.866 | 0.000 | 0.981 |
| pv KIP-off (s2024, DoTA) | 0.481 | 0.037 | 0.854 |
| pv KIP-on (s2024, DoTA) | 0.382 | 0.017 | 0.869 |
| pv KIP-on (s2024, MSAD) | 0.179 | 0.609 | 0.315 |
| PreVAD trunk (PreVAD test) | 0.209 | 0.507 | 0.421 |
| released `best.ckpt` (PreVAD test) | 0.013 | 0.428 | 0.390 |

The report's limitation 8 (86–89 % of DoTA frames above 0.99, every
threshold/calibration metric void) is **substantially relieved** by the PreVAD
trunk: 38–48 %, and the distribution now has a low tail. Still too saturated to
trust a calibration metric, but this is the first arm where the question is
worth revisiting.

---

## 7. Why the KIP A/B is blocked — the module was never trained

This is the finding that governs how the rest of the campaign may be quoted.

20-step trailing means of KIP's own losses:

| run | `kip_rec` first → last | `kip_align` first → last | `kin` first → last | steps |
|---|---|---|---|---:|
| cold **stage 1** (KIP warm-up) | 16.54 → **10.52** | 4.63 → **4.08** | — | 500 |
| cold **stage 2**, KIP-on | 9.43 → **4.91** | 4.07 → **3.70** | 0.76 → **0.47** | 500 |
| pv stage 2, KIP-on, s2024 | 16.51 → **10.15** | 4.61 → **4.61** | 0.74 → 0.69 | 160 |
| pv stage 2, KIP-on, s2025 | 17.06 → **13.35** | 4.60 → **4.58** | 0.73 → 0.67 | 160 |
| pv stage 2, KIP-on, s2026 | 16.02 → **10.92** | 4.61 → **4.57** | 0.73 → 0.67 | 160 |

Three facts follow directly:

1. **`L_KIP_rec` in this campaign ends where the cold campaign's stage 1
   *ended*, and 2.1–2.7× worse than where its stage 2 ended.** The PMG flow head
   was grafted at random init (the notebook's graft cell prints exactly that:
   trunk tensors reused, `kip.*` left at init) and given 160 joint steps. It
   never learned to regress `e_O`.
2. **`L_KIP_align` never moves.** It starts at 4.605 and ends at 4.572–4.615 —
   a change of < 0.04 over the whole run — and sits **above** the 4.265 chance
   level the report quotes. In the cold campaign it reached 3.70. The alignment
   term contributed literally nothing here.
3. **`L_kin` barely moves** (0.73 → 0.67, against the cold campaign's 0.47).

**Therefore:** Δ(on − off) in this campaign compares "trunk" against "trunk +
an untrained KIP + three loss terms that did not converge, on a 3.1×-shorter
schedule". It is **not** a test of KIP. Recording its signed value as evidence
about the +0.09 would repeat lesson **C14** verbatim — an ablation run under a
known-open precondition defect measures the defect.

### 7.1 The budget cut alone does not explain the collapse

Seed 2024, from the A5 trajectory probe (`RESULTS_ARM4_PROBE.md`), DoTA micro
AUC at matched step count, linearly interpolated to step 160:

| arm | cold @ ~160 steps | PreVAD trunk @ 160 steps | difference |
|---|---:|---:|---:|
| KIP-off | ≈ 0.5596 (flat 0.5589→0.5604 over 100–500) | **0.5864** | **+0.027** |
| KIP-on | ≈ 0.6286 (0.6097@100 → 0.6412@200) | **0.5664** | **−0.062** |

So at a matched budget the PreVAD trunk **helps** KIP-off by ~0.027 and **hurts**
KIP-on by ~0.062. The shorter schedule is not the explanation for either.
(Probe checkpoints exist for seed 2024 only; treat this as one seed's arithmetic,
not a measured n=3 result.)

### 7.2 What this *does* license — a negative control on H4

**H4** (from the v2 code audit): the gate MLP's 321 parameters are frozen at
random init because `(ratio * max_shift).floor().long()` kills the gradient, so
KIP may be nothing but a fixed temporal smoother, and a fixed smoother is exactly
what buys AUC on a within-clip localization benchmark.

In this campaign the smoother is **fully present** — `use_gate_shift=true`, the
PMG head runs, the gate reads `flow_norm`, the shift applies — and it delivers
**−0.002 ± 0.016**. The only thing missing is a *trained* PMG head.

That is a real constraint on H4: the architecture alone does not produce the
gain. The gate's shift pattern is driven by `‖ê_O‖`, so a garbage `ê_O` yields a
different shift schedule than a trained one — the strong form of H4 ("the content
of `ê_O` is irrelevant") is hard to sustain against this.

**Do not over-read it.** The trunk changed at the same time, so this is
suggestive, not decisive. The clean version of this control is one run
(§9, item 1).

---

## 8. What the PreVAD trunk buys — the clean cross-campaign result

Paired clip bootstrap, PreVAD-trunk arm minus the cold arm of the same seed,
same arm type. Ids and ground truth verified identical.

### 8.1 MSAD in-domain

| arm | Δ AUC (seed-level) | t95 | Δ AP (seed-level) | t95 |
|---|---:|---|---:|---|
| KIP-off | +0.0120 ± 0.0056 | [−0.0020, +0.0260] | **+0.0474 ± 0.0063** | **[+0.0318, +0.0630]** |
| KIP-on | +0.0108 ± 0.0043 | [+0.0001, +0.0216] | **+0.0467 ± 0.0136** | **[+0.0131, +0.0804]** |

Per-seed AP deltas: +0.0404 / +0.0525 / +0.0493 (off), +0.0315 / +0.0515 /
+0.0573 (on). **Consistent, same-signed, ~+5 AP points in every seed and both
arms.** This is the campaign's strongest positive number.

### 8.2 DoTA zero-shot

| arm | seed 2024 | 2025 | 2026 | seed-level | t95 |
|---|---:|---:|---:|---:|---|
| **KIP-off** Δ AUC | +0.0257 | +0.0291 | +0.0580 | **+0.0376 ± 0.0177** | [−0.0065, +0.0817] |
| KIP-off Δ macro | +0.0369 | +0.0493 | +0.0824 | +0.0562 ± 0.0235 | [−0.0023, +0.1146] |
| **KIP-on** Δ AUC | −0.0851 | −0.0420 | −0.0423 | **−0.0565 ± 0.0248** | [−0.1180, +0.0051] |
| KIP-on Δ macro | −0.0938 | −0.0386 | −0.0251 | −0.0525 ± 0.0364 | [−0.1429, +0.0379] |

All twelve clip-level CIs exclude zero and **the sign is stable within each arm**
— the pattern that the stage-1-only trunk transfer conspicuously failed
(`RESULTS_ARM4_PROBE.md`: ±0.02 with a sign that flipped across seeds). At the
seed level the t-intervals are wide (n = 3, and the seed-2024 magnitudes are
outliers in both arms), so state it as: **direction established, magnitude not
pinned.**

**Read the two rows together:** a broad-domain trunk lifts the plain baseline's
zero-shot transfer by ~+0.04 AUC and drags the (untrained-)KIP arm down by
~−0.06. Both arms converge on ~0.586. The trunk did not "replace" KIP; the two
arms simply ended up at the same place from opposite directions, and only one of
those movements has an uncontaminated explanation.

---

## 9. What to run next — in this order

1. **The one run that unblocks everything: PreVAD trunk + KIP stage-1 warm-up +
   500 steps.** Same PreVAD trunk, but run the stage-1 KIP-only warm-up on MSAD
   before stage 2, and restore `num_epochs=125`. That is a single training arm
   per seed and it isolates *trunk* from *KIP pretraining* — the only reason
   claim 3 is blocked. Until it exists, the campaign has no KIP verdict.
2. ~~**A PreVAD KIP-on arm.**~~ **Ruled out permanently — do not reopen.**
   `L_KIP_rec` regresses `e_O`, a cached RAFT embedding, and RAFT needs pixels.
   **PreVAD ships CLIP features only** (`ViT-B-16-8p-features.zip`); there is no
   raw video, the release cannot be replayed from
   `annotations/data_sources.csv` (realistic yield 50–70 % after link rot, and
   the 3,800 permanently-dead rows are the highway-camera clips KIP cares about
   most), and a self-made download cannot be paired with the released features —
   different transcode, different fps, different lengths, and
   `core/data/dataset.py:112` raises on the mismatch. A KIP-on arm on a 60 %
   subset would not be comparable to a KIP-off arm on the full release anyway.
   Reasoning in full: `PREVAD_SETUP.md` §7.4. **Do not** work around it with
   `require_flow=False` — that zero-fills `e_O` and trains the PMG head to
   predict zeros, producing a wrong run that still writes a checkpoint.
   *Confirmed as the user's decision, 2026-08-29.* The remaining route to a
   second KIP benchmark is a dataset that ships pixels.
3. **Fix `--init-weights` provenance (lesson C17).** Three campaigns have now
   been reconstructed from step-1 loss values because `config.yaml` does not
   record the flag that defines the arm. ~30 LOC for a run manifest.
4. **Fix the float32/float64 pooling discrepancy** (§10.1). It is small but it
   makes two of our own tools disagree on the same `.npz`.
5. **Retire the "MSAD reproduction" framing.** With the PreVAD trunk our KIP-off
   arm is above the released checkpoint on AUC and AP. The interesting question
   is no longer "do we match LaGoVAD" but "how much of the gap to 0.9041 is
   pretraining corpus".

Standing discipline, unchanged: **do not tune KIP against any of these numbers.**

---

## 10. Defects found while analysing

### 10.1 `core.evaluate` and `core.tools.rescore` disagree on the same score files

`core.evaluate` min-max normalizes in **float32**; `core.tools.rescore` casts to
float64 first (`load_scores`, `core/tools/rescore.py:57`). On
`DoTA_ncc_pv_s2024/eval_kip_off`:

| path | micro AUC | AP |
|---|---:|---:|
| float32 (what wrote `results.json`) | 0.586406 | **0.369953** |
| float64 (what `rescore` prints) | 0.585956 | **0.366069** |

Verified by reproducing both from the `.npz` files. `auc_raw`/`ap_raw` are
identical to the last digit — the divergence is created *by the normalization*,
where float32 rounding manufactures ties among the saturated frames and sklearn
splits tie credit. **ΔAP = 0.0039 is the same order as several deltas in this
report.** Every number in this document uses the **float32** path so it matches
`results.json` and every prior `RESULTS_*.md`; the deltas are paired, so both
arms get the same treatment and no conclusion changes.

Fix: normalize in float64 in `core/metrics.py:normalize_scores` (or cast on load
in `core/evaluate.py`), then re-run `rescore --write` over every run dir. Lesson
candidate: *a metric's dtype is part of its protocol.*

### 10.2 90 PreVAD clip ids are silently mangled in `scores/` filenames

90 of 2,606 test ids contain `:` (e.g.
`2_KyluTt9Rk_00:13:32.467_00:13:42.733`). Their `.npz` files are written with
`:` → `_`. All 90 recover under that substitution and **no scores are lost** —
but a naive join of `scores/*.npz` stems against `test.csv:video_id` silently
drops those 90 clips (3.5 % of the test set), with no error. It cost me one
wrong per-superclass table before I noticed the `?` bucket.

Any analysis joining PreVAD score files to metadata must apply
`video_id.replace(":", "_")`. Lesson candidate: *an id that survives a filesystem
round-trip is not the same id.*

### 10.3 Gate P0 was recorded *after* the runs it was supposed to gate

`PREVAD_SETUP.md` §10 lists P0 as step 8, gating steps 9–13, and says "do not
report any Stage P result until P0 has a recorded number". Steps 9–13 ran first.
P0 passed, so nothing was wasted — but the gate did no gating.

---

## 11. Provenance

| Section | Artifacts |
|---|---|
| §1 design | `collab/PreVAD/evaluate.py`; `outputs/**/config.yaml` (6 MSAD finetunes + PreVAD trunk); step-1 `metrics.jsonl` |
| §2 dataset | `PreVAD/{train,test}.csv` (32,673 + 2,606 rows), re-counted spans |
| §3 Gate P0 | `outputs/PreVAD/{gate_p0,eval_trunk}/results.json` and `scores/*.npz` (2,606 each) |
| §4 trunk training | `outputs/PreVAD/stage2_kip_off/metrics.jsonl` (13,360 rows) |
| §5 MSAD | `outputs/MSAD_ncc_pv_s{2024,2025,2026}/eval_kip_{on,off}/**`; reference `outputs/MSAD_ncc/full_gate_a` |
| §6 DoTA | `outputs/DoTA_ncc_pv_s{2024,2025,2026}/eval_kip_{on,off}/**`; groups from `data/DoTA/metadata_val.json → anomaly_class` |
| §7 KIP losses | `outputs/MSAD_ncc_pv_s*/stage2_kip_on/metrics.jsonl`, `outputs/MSAD_ncc/stage{1,2_kip_on}/metrics.jsonl`; probe from `outputs/DoTA_ncc/probe/**` |
| §8 cross-campaign | paired against `outputs/{MSAD,DoTA}_ncc{,_s2025,_s2026}/eval_kip_{on,off}` |

**Statistics.** Micro AUC = `roc_auc_score` over concatenated sampled frames
under the resolved pooling rule. Macro AUC = mean per-clip `roc_auc_score`,
skipping single-label clips. Confidence intervals = paired bootstrap resampling
**clips** (not frames), 2,000 draws, `default_rng(0)`, both arms scored on the
same resample; the macro bootstrap resamples the precomputed per-clip AUC vector.
Seed-level intervals are two-sided 95 % t-intervals over the three per-seed
point estimates (t = 4.303, n = 3) — the audit's preferred summary, because
clip-level CIs over-count: within a seed the three metrics share one score curve.
Every arm-pair was checked for identical clip ids and identical ground truth
before differencing. **No re-inference was performed**; every figure derives from
saved score curves, `results.json` and `metrics.jsonl`.

**Reproducibility caveat.** `outputs/` is gitignored. This document and the
`RESULTS_*.md` chain are the durable record; the `.npz` score curves live on the
author's disk and Drive. The analysis scripts used here were scratch and are not
committed — §9 item 4 should promote the paired bootstrap into
`core/tools/` with tests so these numbers are re-derivable.
