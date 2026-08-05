# MSAD-full — Measured Results (commit `b9978ff`)

**Analysed:** 2026-08-01 · **Source:** `outputs/MSAD/**` (produced on Colab by
`collab/MSAD/train.py`) · **Analysis script:** re-derivable, see §9.

Everything below is recomputed from the per-video `.npz` score dumps, not copied
from the run logs. `results.json` values reproduce exactly.

---

## 1. Setup, as actually run

| Item | Value |
|---|---|
| Dataset | MSAD-full, 11 anomaly classes + Normal (`data/MSAD/meta.json`) |
| Train split | 480 videos (360 normal + 120 abnormal), video-level labels only |
| Test split | 240 videos (121 normal + 119 abnormal), 18,350 stride-8 frames, 23.0 % positive |
| Features | frozen CLIP ViT-B/16, `frame_stride=8`, `crop_size=224` (**center-crop** variant) |
| Flow targets | RAFT, cached, train-time only |
| Stage 1 | KIP warm-up, 125 epochs = 500 steps, lr 5e-5, bs 64, seed 2024, AMP off |
| Stage 2 | full objective, 125 epochs = 500 steps, seed 2024, AMP on, init from stage 1 |
| Eval ckpt | `checkpoint_last.pt` for both arms (**no validation-based selection**) |
| Metric | micro frame AUC/AP over concatenated **stride-8 sampled** frames |

Both stage-2 arms share seed, schedule, data and init policy; the KIP-off arm
differs only by `kip.enabled=false` (and therefore does not load flow).
This is a clean paired comparison — the only uncontrolled factor is the seed
itself (single seed, see §6.1).

---

## 2. Headline numbers

| Run | Checkpoint | AUC | AP |
|---|---|---:|---:|
| `full_gate_a` — released `best.ckpt` through our eval | `ckpts/best.ckpt` | 0.8991 | 0.6811 |
| `eval_kip_off` — our stage-2 baseline (KIP disabled) | `stage2_kip_off/checkpoint_last.pt` | 0.9052 | 0.7249 |
| `eval_kip_on` — our stage-2 + KIP | `stage2_kip_on/checkpoint_last.pt` | **0.9064** | **0.7334** |
| LaGoVAD paper, MSAD | — | 0.9041 | n/a |

**Reproduction verdict — PASS.** The KIP-off retrain (90.52) lands +0.11 pp over
the LaGoVAD published MSAD number (90.41). The baseline port is sound; the
"matches the paper" claim is now backed by a number.

**Gate (a) note.** The released checkpoint scores 89.91 through our pipeline,
0.5 pp *below* the paper. That gap is the price of our pipeline differences
(center-crop transform vs. the baseline's `no_center_crop`, stride 8, seeded
verbalizer, stride-8 metric). Useful as a bound: pipeline noise ≈ 0.5 pp, which
is **five times larger than the entire KIP effect**.
*Assumption to confirm: `ckpts/best.ckpt` is LaGoVAD's released checkpoint.*

---

## 3. Does KIP help? — the honest answer

### 3.1 Aggregate

| Metric | KIP-off | KIP-on | Δ | Verdict |
|---|---:|---:|---:|---|
| Frame AUC (all 240) | 0.9052 | 0.9064 | **+0.0012** | inside noise |
| Frame AP (all 240) | 0.7249 | 0.7334 | **+0.0086** | inside noise |
| AUC_A (abnormal videos only — *localization*) | 0.6944 | 0.6949 | +0.0005 | nil |
| AP_A | 0.7539 | 0.7657 | +0.0118 | inside noise |
| Video-level AUC (max score) | 0.9592 | 0.9639 | +0.0047 | inside noise |
| Multi-class frame acc. on anomalous frames | 0.4754 | 0.5116 | **+0.0362** | largest effect, still inside noise |

### 3.2 Paired bootstrap over test videos (10,000 resamples)

| Δ | mean | 95 % CI | P(Δ>0) |
|---|---:|---|---:|
| AUC | +0.0012 | [−0.0060, +0.0089] | 0.62 |
| AP | +0.0078 | [−0.0141, +0.0296] | 0.76 |
| Multi-class acc. | +0.0357 | [−0.0384, +0.1140] | 0.83 |

**Every confidence interval straddles zero.** Gate (c) ("KIP-on ≥ KIP-off")
*technically passes* on the point estimate — KIP-on wins on every single
aggregate metric, which is itself mildly encouraging — but the margin is not
statistically defensible from one seed. Do not put +0.12 pp AUC in a paper.

### 3.3 Per-class frame AUC/AP (class abnormal videos + all 121 normal videos)

| Class | n | AUC_off | AUC_on | ΔAUC | AP_off | AP_on | ΔAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Assault | 8 | 0.8671 | 0.8954 | **+0.0283** | 0.3523 | 0.3372 | −0.0151 |
| Shooting | 9 | 0.9398 | 0.9680 | **+0.0282** | 0.5631 | 0.5664 | +0.0033 |
| People_falling | 22 | 0.9237 | 0.9311 | +0.0074 | 0.5129 | 0.5136 | +0.0007 |
| Vandalism | 6 | 0.9624 | 0.9662 | +0.0038 | 0.5462 | 0.3895 | −0.1567 |
| Explosion | 7 | 0.9856 | 0.9867 | +0.0011 | 0.8095 | 0.8114 | +0.0019 |
| Object_falling | 10 | 0.9894 | 0.9880 | −0.0014 | 0.9062 | 0.8937 | −0.0126 |
| **Traffic_accident** | 20 | 0.9585 | 0.9565 | **−0.0020** | 0.3995 | 0.3965 | −0.0030 |
| Fighting | 6 | 0.9877 | 0.9826 | −0.0051 | 0.6982 | 0.6915 | −0.0067 |
| Fire | 12 | 0.9756 | 0.9702 | −0.0055 | 0.7934 | 0.7300 | −0.0634 |
| Water_incident | 5 | 0.9813 | 0.9749 | −0.0063 | 0.8901 | 0.8792 | −0.0109 |
| Robbery | 14 | 0.9399 | 0.9327 | −0.0072 | 0.6646 | 0.6246 | −0.0401 |

**The bad news, stated plainly:** `Traffic_accident` — the thesis's target class,
20 test videos — moves **−0.002 AUC**. KIP's gains sit on `Assault` and
`Shooting` (n=8, n=9 → ±1 video flips the sign). The motion-dominated classes
KIP was designed for show nothing on this benchmark.

### 3.4 Per-video AUC (96 abnormal videos with both labels present)

mean ΔAUC **+0.0119**, median **+0.0000**, wins 41.7 %, losses 38.5 %, ties 19.8 %.
Best: `Traffic_accident_26` +0.373, `Traffic_accident_4` +0.314, `Explosion_9`
+0.296. Worst: `People_falling_38` −0.396, `People_falling_28` −0.363,
`Object_falling_14` −0.198. **A coin flip with fat tails in both directions** —
the signature of a component that is changing the representation without
consistently improving it.

### 3.5 Where KIP *does* look real: the multi-class head

Frame-level class accuracy on anomalous frames, 0.4754 → 0.5116 (+7.6 % relative):

| Class | acc_off | acc_on | Δ |
|---|---:|---:|---:|
| Water_incident | 0.082 | 0.271 | **+0.188** |
| Fire | 0.674 | 0.848 | **+0.175** |
| Shooting | 0.245 | 0.364 | +0.119 |
| Fighting | 0.290 | 0.408 | +0.118 |
| Traffic_accident | 0.323 | 0.428 | **+0.105** |
| Object_falling | 0.703 | 0.715 | +0.011 |
| Assault | 0.073 | 0.079 | +0.007 |
| Explosion | 0.154 | 0.125 | −0.029 |
| Robbery | 0.796 | 0.710 | −0.085 |
| Vandalism | 0.452 | 0.360 | −0.091 |
| People_falling | 0.626 | 0.497 | −0.129 |

The classes that gain are the temporally-extended, motion-textured ones (water,
fire, fighting, traffic accident); the classes that lose are the ones with a
strong static-appearance cue (robbery, vandalism). **That is a mechanistically
plausible KIP signature** — it says the motion pathway is reshaping *which*
anomaly is recognised more than *whether* one is present. It is the most
promising thread in this whole run, and it is invisible in the headline AUC.

### 3.6 False alarms and curve dynamics

| Run | mean score on normals | frac > 0.5 | frac > 0.9 | mean\|Δscore\| abnormal | normal |
|---|---:|---:|---:|---:|---:|
| KIP-off | 0.0447 | 0.0397 | 0.0293 | 0.0072 | 0.0018 |
| KIP-on | 0.0465 | 0.0423 | 0.0320 | **0.0102** | 0.0017 |
| gate_a | 0.0698 | 0.0547 | 0.0170 | — | — |

KIP-on's curve is **42 % more temporally dynamic on abnormal videos** with no
extra jitter on normal videos (0.0018 → 0.0017). The motion pathway is doing
*something* — sharpening temporal response — it just isn't converting into AUC.

---

## 4. Training-side diagnostics

### 4.1 Loss trajectories (first-20-step mean → last-20-step mean)

| Run | loss | first 20 | last 20 | drop |
|---|---|---:|---:|---:|
| stage 1 | `kip_rec` | 16.33 | 9.95 | 39 % |
| stage 1 | `kip_align` | 4.63 | 4.02 | 13 % |
| stage 2 on | `kip_rec` | 9.29 | 4.84 | 48 % |
| stage 2 on | `kip_align` | 4.01 | 3.66 | 9 % |
| stage 2 on | `kin` | 0.755 | 0.424 | 44 % |
| stage 2 on | `mil` | 0.658 | 0.008 | 98.7 % |
| stage 2 off | `mil` | 0.740 | 0.001 | 99.8 % |

### 4.2 Three problems visible in these curves

**(a) `L_KIP_align` is running near chance.** The loss is a within-video
bidirectional InfoNCE over the video's own valid positions
(`core/kip/losses.py:44`). Mean sampled length on train is 78.2 frames, so
chance loss = E[ln n] = **4.265**. Final value **3.66** — only 0.6 nats (14 %)
below chance after 1,000 steps. The pseudo-flow and RGB embeddings are barely
distinguishable frame-to-frame, which is exactly the A11 failure mode the code
already anticipates: adjacent frames are near-duplicate *positives* being scored
as *negatives*, so the task is close to ill-posed. Both mitigation knobs
(`loss.align_subsample`, `loss.align_exclude_window`) are **off** (0/0).
The alignment term is currently contributing almost no learning signal.

**(b) Stage 2 memorizes the training set.** `mil` reaches 0.001 (off) / 0.008
(on) on 480 videos with video-level labels. The model has fully fitted the weak
labels well before step 500, and **evaluation reads `checkpoint_last.pt`** —
i.e. an arbitrary point deep inside the overfit regime, with no validation split
and no model selection. The KIP-on/off comparison is being taken at a point
neither arm was chosen for. This alone can produce ±0.5 pp swings.

**(c) `kip_rec` has not plateaued.** Its 20-step moving average first comes
within 5 % of its minimum at step ~446 of 500, in *both* stage 1 and stage 2 —
the reconstruction head is still improving when the schedule ends. Warm-up may
be undertrained (the COLAB note claims a plateau from ~step 200 on the small
28-video slice; that does not hold on MSAD-full).

### 4.3 Parameter and compute cost of KIP

| | params | vs. baseline |
|---|---:|---:|
| Baseline (KIP off), trainable | 18,923,524 | — |
| KIP total | 443,714 | +2.34 % |
| ├ `pmg` (PMGFlowHead) | 311,808 | inference path |
| ├ `shift` (KinematicShift gate) | 321 | inference path |
| ├ `mhead` (MotionScoreHead) | 33,025 | inference path |
| └ `proj_flow` + `proj_rgb` (align only) | 98,560 | **train-time only** |
| KIP on the inference path | 345,154 | **+1.82 %** |

**Efficiency verdict:** the design promise holds — inference stays RGB-only and
costs +1.8 % parameters (RAFT never runs at test time). The cost is paid
offline: RAFT flow extraction over all 720 videos, plus the stage-1 warm-up run.
For **+0.0012 AUC**, the current return on that cost is not defensible; for the
**+3.6 pp multi-class accuracy**, it might be. No wall-clock was logged
(`metrics.jsonl` has no timing field) — see next actions.

---

## 5. What this run does and does not establish

**Established**
1. The baseline port reproduces LaGoVAD on MSAD-full (90.52 vs 90.41 published).
2. The full KAT-VAD stack trains end-to-end on a real dataset without divergence;
   every KIP loss decreases monotonically-ish.
3. KIP's inference cost is +1.8 % params, RGB-only — the architectural claim holds.
4. KIP-on ≥ KIP-off on **every** aggregate metric measured (6/6), point-estimate.

**Not established**
1. That any of those margins is real. All 95 % CIs include zero; n=1 seed.
2. That KIP helps traffic anomalies — on the target class it is −0.002 AUC.
3. That the KIP mechanism (motion evidence) is what produced the deltas — no
   diagnostic of ŷ_O, gate α, or shift magnitude was saved at eval time.
4. Anything about DoTA/ego-centric motion, which is where the thesis claim lives.
   MSAD is fixed-camera; its anomalies are largely appearance-separable (the
   KIP-off model already hits 0.905). **This benchmark has little headroom for a
   motion pathway** — a null result here is weak evidence against KIP.

---

## 6. Next actions — ranked

### 6.1 P0 — Make gate (c) defensible: multi-seed (cheap, do it first)
Each stage-2 run is **500 optimizer steps**. Three seeds × {on, off} = 6 runs, plus
3 stage-1 warm-ups. Report mean ± std, and the paired per-seed Δ.
```bash
for s in 2024 2025 2026; do
  python -m core.train --set train.stage=2 --set train.seed=$s --set train.amp=true \
    --set data.dataset=MSAD-full --set train.num_epochs=125 \
    --init-weights "$OUT/MSAD/stage1_s$s/checkpoint_last.pt" \
    --output-dir "$OUT/MSAD/stage2_kip_on_s$s" ...   # and kip.enabled=false twin
done
```
**Decision rule:** if mean ΔAUC over 3 seeds < seed std, KIP does not help on
MSAD and the thesis narrative must move to DoTA (§6.6) or to the multi-class
story (§6.5). Do not iterate on KIP hyperparameters before this number exists.

### 6.2 P0 — Add a validation split and select checkpoints by it
Right now both arms are evaluated at `checkpoint_last.pt` with train `mil` ≈ 0.001
(§4.2b). Carve ~60 train videos (balanced) into a val split, log val AUC every N
steps, keep `checkpoint_best.pt`. Without this, every future ablation is measured
at an arbitrary point in an overfit regime. **This is the single largest source of
measurement noise in the current setup.**

### 6.3 P1 — Fix `L_KIP_align` (it is running at 14 % below chance)
Two one-line experiments, cheap, run against the multi-seed baseline:
- `--set loss.align_exclude_window=2` (drop near-duplicate temporal neighbours
  from the negative set — the documented A11 mitigation)
- `--set loss.lambda_align=0.5` (currently 0.1; the term barely moves the loss)

Success criterion: `kip_align` ends below ~3.0 (vs. chance 4.27) *without*
`kip_rec` regressing.

### 6.4 P1 — Instrument the eval so KIP's mechanism is observable
`core/evaluate.py` currently saves only `score`, `sim`, `gt`. Add (behind
`--save-scores`): `y_motion` (ŷ_O), the gate value α per frame, and the realised
shift magnitude. Then answer directly:
- Does ŷ_O correlate with cached RAFT flow norm on the **test** set? (Is PMG
  actually inducing motion, or just another learned feature?)
- Is the gate saturated? A dead gate (`shift` has 321 params, and the known
  `floor()` gradient issue is logged in `progress.md`) would explain a null result.

Also implement `AUC_A` and the MCC family in `evaluate.py` (currently
`NotImplementedError`); AUC_A = 0.694 is the number the KIP claim actually rides
on, and it should not require a side script.

### 6.5 P1 — Chase the multi-class result, it is the strongest signal here
+3.6 pp frame accuracy on anomalous frames, concentrated exactly on the
motion-textured classes (§3.5). Confirm across seeds, then produce a confusion
matrix (KIP-on vs off) and per-class AP. If it holds, "KIP improves anomaly
*categorisation* under fixed cameras, and *detection* under ego-motion" is a more
honest and more defensible thesis claim than a 0.1 pp AUC delta.

### 6.6 P2 — Move the headline claim to where motion has headroom
MSAD-full KIP-off already sits at 0.905. The proposal's headline is DoTA
(LaGoVAD 62.60, ego-centric, kinematics-dominated). Also run the MSAD-**highway**
stress slice (the proposal notes RTFM/MGFN/UR-DMU collapse to AP 1.4–4.1 there).
A +0.001 on saturated MSAD-full proves nothing either way; a delta on DoTA is
the whole thesis.

### 6.7 P2 — Spec §10 ablations (only after §6.1–6.2)
`kip.pmg_only=true`, `kip.use_gate_shift=false`, `kip.use_lkin=false`,
`kip.gate_signal=*`. With a val split and 3 seeds these become interpretable;
without them they are noise generators.

### 6.8 P3 — Housekeeping
- Log wall-clock/step and peak memory into `metrics.jsonl` — the efficiency claim
  ("RGB-only, +1.8 % params") deserves a measured FPS, not just a parameter count.
- Metrics are computed on 18,350 **stride-8 sampled** frames, not the 146,012 raw
  frames. LaGoVAD's published number is frame-level after expansion. Expand
  scores to full frame rate before quoting our 0.9052 against their 0.9041.
- `stage1` and `stage2_kip_off` share `outputs/` layout but not provenance
  metadata — write `git_sha`, seed, and the config hash into `results.json`.

---

## 7. Risks in the current story

| Risk | Impact | Mitigation |
|---|---|---|
| Reporting +0.0012 AUC as a KIP win | Reviewer kills the paper on one bootstrap | §6.1 multi-seed; report CI |
| `checkpoint_last` under train loss ≈ 0.001 | Every ablation measured at an arbitrary overfit point | §6.2 val split |
| `L_KIP_align` near chance | One of three KIP losses is inert; ablation would show "align doesn't matter" for the wrong reason | §6.3 |
| No eval-time KIP diagnostics | Cannot distinguish "KIP is wrong" from "KIP is disabled by a dead gate" | §6.4 |
| MSAD-full is saturated for this question | Null result reads as evidence against KIP when it is evidence of no headroom | §6.6 DoTA/highway |

---

## 8. Feature-cache caveat (unchanged, still live)

These numbers were produced with the **center-crop** CLIP transform
(`core/tools/extract_clip_features.py:41-64`), not the baseline's
`no_center_crop`. Every number in this document is bound to that cache. Changing
the transform invalidates all of it. See `.project/memory-bank/activeContext.md`
open question 2.

---

## 9. Reproducing this analysis

All figures above come from `outputs/MSAD/{eval_kip_on,eval_kip_off,full_gate_a}/scores/*.npz`
(keys: `score`, `sim` (L×12), `class_names`, `gt`) and the three `metrics.jsonl`
files. Metrics: `sklearn.roc_auc_score` / `average_precision_score` on
concatenated frames; CIs from a 10,000-sample paired bootstrap resampling
**videos** (not frames), seed 0. Parameter counts from
`KATVAD.from_config(load_clip=False)`.
