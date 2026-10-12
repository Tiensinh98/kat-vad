# Proposed Road-Traffic Video Anomaly Detection Architecture — **KAT-VAD v2.2**
### A staged test of temporal order between steps (D6-step), with a conditional Temporal-Difference Pathway from LAS-VAD's IAM

**Supersedes:** the first v2.2 draft of 2026-10-11 (TDP built directly, gated by a per-step linear probe), and `KAT_VAD_PROPOSAL_v2.md` as the working proposal. `claude/KAT_VAD_PROPOSAL_v2.1.md` (SG-NM) is closed: its gate failed (E4a).

**Evidence base:** the v2 results (`REPORT_V2_RESULTS.md`, updated 2026-10-11), quoted as "results §x"; the v2 proposal and architecture, quoted as "v2 §x" / "arch v2 §x"; the T2-study gaps G1–G5 as quoted in v2; the project paper summaries (LAS-VAD, DSANet, RefineVAD, LaGoVAD, SimpleTAD, Pi-VAD). Numbers marked *derived* are computed from those, with the arithmetic shown.

**Companion:** `KAT_VAD_v2.2_ARCHITECTURE.md` (the deployed network, the shuffle operator, the probe, the trunk arm and the conditional TDP, with tensor shapes).
**Date:** 2026-10-11 (after two review rounds the same day; §R).

> **Design rule (unchanged from v2).** A component enters only if (i) it targets a failure that was **measured**, (ii) a project paper supports the mechanism, and (iii) it can be checked cheaply **before** it is trained.
> - The scientific question, *does temporal order between steps carry detection signal?*, is answered first, on cached features, with numeric gates fixed below.
> - A new component (TDP) is built **only** if explicit velocity/acceleration features add signal beyond what a small temporal model already extracts from the same features.
> - **The deployed network is A3-ign** unless the last step adopts TDP. No new loss in any branch.

---

## R. Review history (2026-10-11)

| Round | Main points | Outcome in this version |
|---|---|---|
| **1** (9 points on the first draft) | Learned convs read across DVS splice junctions; the trunk *can* form signed differences (outer residual + linear kernel-3 head), so TDP adds only the magnitude and the second difference; Tπ leaked adjacency through in-model temporal ops; the per-step linear probe could not see the mechanism (transients, "spike then recover vs stay deviated"); VideoMAE windows overlap ≈ 80 %, so Δu echoes 1.5 s later; Δ_0 = 0 creates a clip-start acceleration spike; paired-seed integrity; an overused dev set; normal within-clip events compete with onsets | All accepted. TDP becomes **offline per-source features + a pointwise learned map** (no in-model temporal op, so splice-safe); **CLIP differences only** by default; **κ-level replicate padding**; a **fixed-capacity temporal probe** replaces the linear probe; the control is a **quantile-matched full-pipeline permutation**; RNG isolation and a **re-run control**. The overall verdict: answer the order question first (§3–§5) |
| **2** (agreement on ≈ 80 %; 5 gaps) | Whole-clip shuffling is confounded with position; probe receptive field miscounted (9 steps, ±4 ≈ ±1.2 s) and zero padding gives a window-position cue; the grad-clip rationale does not apply (no global clipping in `core/train.py`); TAD is a P0/P1 project in a different regime; the thesis-level risk that only one outcome yields a component | **Block shuffle** at the 1.5 s scale replaces whole-clip shuffling; **context-padded windows** for the probe; grad-clip rationale dropped; TAD deferred as a generalization study (§11); the **outcome table and an advisor check** before any training run (§10) |
| **3** (5 tightenings) | A random within-block permutation keeps ≈ 40 % of adjacencies (1.6 of 4); ragged final blocks; "+0.128 over A0" is not a measured contrast *(withdrawn in round 4)*; stop rules need numeric bars; probe intervals must come from the cluster bootstrap, not from probe seeds | **Adjacency-free permutations only** (14 for n = 5, also across block boundaries); a **ragged-tail rule** (§4.2); only measured contrasts quoted (§0); **one numeric bar** for every gate (§6.1); **cluster bootstrap** for probes, seeds as a stability check only |
| **4** (correction + 5 points) | **Correction:** "+0.1276 [+0.0991, +0.1561]" (results §4.5) *is* a notebook-printed paired t95 of A3-ign − A0; it equals 0.0900 + 0.0376 because paired mean differences are additive over shared seeds. (1) a scalar change score failing (S1a) does not show that order carries no signal; (2) `ũ_t` already summarises the true past 1.5 s as a bag, so S2 leans toward null; (3) κ was unstandardised on the model path, so probe ≠ model features; (4) S1b's "CI ∋ 0 → admit Δu" is an absence-of-evidence rule; (5) ambiguities: S1's deciding set, what `p` is, S1c multiplicity, the S4 trigger, short sequences in the twin | +0.128 restored as a measured contrast (§0); **S1a gates only the TDP branch**, S2 always runs (§5, §9, §10); S2's estimand says "beyond VideoMAE's 1.5 s bag" and a **CLIP-only secondary read** `[x̃; p]` is added (§3, §5); κ gets a **fixed T2-train standardisation, baked offline** (§7.1); S1b uses an **equivalence margin** (§5); S1 decides on DoTA-CAP-dev; `p` = cubic basis `[τ, τ², τ³]` (3 channels); S1c reads **five summary channels**; **S4 trigger = S2 passes and advisor approval**; short-sequence shares reported (§4.3) |

---

## A. The answer in brief (LAS-VAD and the common temporal front-end)

| Idea (source) | Decision | Reason | Detail |
|---|---|---|---|
| **IAM velocity / acceleration** (LAS-VAD §3.2) | **Conditional.** Built as TDP only if the κ-probe gate (S3) passes | It is the report's own direction H5, and free to compute from caches. But the trunk can already form signed differences, so what explicit features add must be measured, not assumed | §5 S3, §7 |
| IAM prototypes, `w_int`, **cross-intention contrastive loss `L_cst`** | **Decline** | Needs per-step class labels (83 fine DADA categories; hard mining selects exactly the boundary steps where pseudo-labels are wrong); unbounded as written (no temperature, no normalization, positive excluded from the denominator), which would inflate the un-normalized magnitudes KAT-VAD relies on; weight, α, M unspecified; an auxiliary trunk loss (G4); would confound T − Tπ; not ablated separately in LAS-VAD | §8.1 |
| **ACC** (LAS-VAD §3.3) | **Decline** | Not a motion component (ignores order). τ = 0.9 vs adjacent cosine 0.968 chains each window into one component; self-derived labels (cf. E4a); would undo E4-0 by re-labelling anchor steps densely; a new trunk loss | §8.2 |
| **Local Transformer** (DSANet; LAS-VAD via VadCLIP) | **Neither add nor replace** | Sequences are already local (20 training steps; whole DoTA clip ≈ 33); replacing the encoder breaks every paired control; no isolated ablation in the project's evidence | §8.3 |
| **Similarity GCN** (DSANet dual-channel; LAS-VAD global) | **Neither add nor replace** | On these features the softmax adjacency is near-uniform (*derived:* cosines in ≈ [0.9, 1] → weights differ by ≤ e^0.1 ≈ 1.105×), so it re-injects the window mean CRN removes; a smoother, and both smoothing remedies tried failed (H2(a), SG-NM); clip-length dependent (G3) | §8.4 |
| LLM attribute text | Decline | Not the measured bottleneck; text changes belong to DF3 | §8.5 |

---

## B. What changed

**v2 → v2.2.** The deployed network is unchanged (A3-ign) unless S5 adopts TDP. v2.2 adds a pre-registered, staged experiment on temporal order (S1–S5), a block-shuffle operator, a fixed-capacity temporal probe, a trunk-level order arm, and — conditionally — an offline-feature TDP.

**First v2.2 draft → this version.**

| First draft | This version | Why |
|---|---|---|
| TDP built first; gated by a per-step linear probe | Order question answered first (S1–S2); TDP gated by a **temporal** probe (S3) | R1 points 2, 4 |
| Learned projection `P`, learned causal gate convs, acceleration on gated velocity, all in the model | All temporal operations **offline per source video**; fixed PCA projection; fixed causal 3-step mean as the jitter filter; in the model only a **pointwise** MLP and a zero-init `W_k` | Splice junctions (R1.1); clean control (R1.3); probe = model features (R1.4) |
| Differences of CLIP **and** VideoMAE | **CLIP only** by default; Δu only if the echo check passes (S1b) | Moving-average echo (R1.5) |
| `Δ_0 = 0` | **κ-level replicate padding** (≤ 2 future steps at a clip start); padded steps excluded from medians | Clip-start spike (R1.6) |
| Tπ: permuted differences, temporal ops in true order | **Full-pipeline permutation** with far neighbours, **quantile-matched per clip and channel** to T | Adjacency leak (R1.3) |
| E4-0's A3-ign runs as the control | **A3-ign re-run in the same commit**; TDP initialised from its own seeded generator; zero-effect unit test | Paired-seed integrity (R1.7) |
| "the trunk can only read differences indirectly" | **Corrected:** `V^t = H + Enc(H)` carries `H` to a linear kernel-3 head, which can form signed differences; explicit features add the magnitude and the second difference | R1.2 |
| Bars: "interval above 0", "≥ +0.01" in places | **One bar everywhere** (§6.1); probe intervals from the **cluster bootstrap**; trained arms from paired t95 **and** the cluster bootstrap | R3.4, R3.5 |

---

## 0. Design summary

**Baseline.** LaGoVAD (ICLR 2026), the only venue-eligible weakly supervised baseline evaluated on real traffic data, stays the trunk.

**Where v2 stands (measured contrasts only).**
- A3 − A0 on DoTA-CAP-eval: **+0.090 [+0.065, +0.115]** (paired t95, 5/5 seeds; results §4.2). A3 − A1: +0.098 [+0.090, +0.107].
- A3-ign − A3, second look at DoTA-CAP-eval: **+0.0376 [+0.0326, +0.0426]** (5/5); `f_2` +0.0157 [+0.0104, +0.0207] (results §4.5).
- A3-ign − A0, second look at DoTA-CAP-eval: **+0.128 [+0.099, +0.156]** (paired t95, printed by the second-look notebook; results §4.5). It equals 0.0900 + 0.0376 because paired mean differences are additive over the same five seeds.
- Seed-averaged macros on DoTA-CAP-eval: A3-ign 0.8120 (second look), A3 0.7785, A0 0.6979; the position prior `p_T2` 0.8256. The difference of seed-averaged macros (0.114) is smaller than the paired per-seed mean (0.128) because averaging scores over seeds helps the noisier A0 more (A3 − A0: 0.081 vs 0.090, the same pattern).
- The video stream is **appearance, not motion**: shuffling the 16 frames inside each 1.5 s VideoMAE window changes the result by ±0.0004 (D6).

**The question.** v1 rested on "a crash is a velocity discontinuity". D6 tested order **inside** the 1.5 s video window and found none. Nobody has tested order **between** steps. v2.2 does, at the same 1.5 s scale, in three cheap steps before any training run:
1. **S1:** parameter-free change scores on cached features, against a matched order-free control.
2. **S2 (D6-step):** a small temporal probe on the features A3 feeds its trunk, true step order vs an adjacency-free block shuffle that keeps position.
3. **S3:** whether **explicit** velocity/acceleration features add to that temporal probe.

Then, after an advisor check, up to two training steps: **S4**, whether the trained trunk uses step order (only if S2 passes), and **S5**, TDP (only if S1a, S2 and S3 pass).

**What can and cannot be claimed.** Every outcome has pre-written wording (§9). TDP and every order test measure change in a **global scene representation**; no outcome licenses "vehicle motion", "optical flow", "trajectories" or "speed".

**LLM.** Unchanged: Holmes-VAU's ATS, off the critical path.

---

## 1. Gap analysis

### 1.1 Papers v2.2 draws on

| Paper | Traffic gap | Used | Declined |
|---|---|---|---|
| **LAS-VAD** (CVPR) | Static-camera benchmarks; semantic "velocity"; ego-motion inflates differences and makes ACC chain whole videos | IAM's velocity / acceleration construction (conditionally, as fixed offline features) | IAM prototypes, `w_int`, `L_cst`; ACC; attribute text; local Transformer + GCN (§8) |
| **RefineVAD** (AAAI 2026) | Motion as a variance proxy | Corroboration: consecutive feature differences as the "how much moves" signal | MoTAR's learned shift gate (its trainable-gate idea was the worst KIP arm on MSAD, −0.068) |
| **DSANet** (AAAI 2026) | Motion-blind | — | Local Transformer + dual-channel GCN; SG-NM (closed, E4a) |
| **LaGoVAD** (ICLR 2026) | No motion pathway | The baseline, unchanged | — |
| **SimpleTAD** (CVPR 2025) | Fully supervised | Motion-aware MVM objectives beat pixel/semantic ones on DoTA/DADA: a reason to test order at all; the 1.5 s window | — |
| **Pi-VAD** (arXiv 2025) | No traffic data | Motion gives the largest single-modality gain: a reason to test order at all | Induction from a non-motion input (the KIP lesson) |

### 1.2 The measured gap — **Gap M: no test of order between steps exists**

| Evidence | Value | Source |
|---|---|---|
| Order inside the video window carries nothing | D6: ordered − shuffled −0.0004 / −0.0003 (CI ∋ 0) | results §3 |
| Frame CLIP has no in-frame motion | lag-1 cosine 0.968; flow-direction R² 0.071; K 0.723 > W 0.654 | G1 |
| Consecutive steps still differ | *Derived:* for equal-norm vectors ‖x_t − x_{t−1}‖ = ‖x‖·√(2(1 − cos)) = 9.87 × √0.064 ≈ **2.5**, about a quarter of ‖x‖ | G1; v2 §4.1 |
| The trunk *can* form signed differences | `V^t = H + Enc(H)` carries `H` un-normalized to `H_bin`'s kernel-3 conv over time, which can compute `W(h_t − h_{t−1})`. What it cannot form linearly is a **magnitude** of change or a second difference | arch v2 §5, §8 |
| Whether order between steps matters is open | D6 shuffled frames inside each window at fixed step positions | results §8 (H5) |

### 1.3 The other v2 gaps and their status

| Gap | Evidence | v2.2 |
|---|---|---|
| **P** — position convention | `p_T2` 0.8256 > A3-ign 0.8120; A3 flat across position (0.70–0.81) | Not a model target. The block shuffle **preserves position by construction** (§4), and `f_2` is printed beside every Δ |
| **S** — score fades before the span ends | peak ≈ +2 s after onset; argmax in span 61 % | Not targeted. A change signal is expected to sharpen the onset, not cover the span; the tail-clip check guards against a hold-style failure (S5) |
| **N** — non-ego accidents | 0.697 vs 0.804 (ego) | Not targeted. Normal within-clip events (turns, tunnels, lighting, others braking) compete with distant collisions in a global vector; S1 measures how badly |
| **H6** — early-onset clips | onset < 20 %: A3 0.675 (n = 54) | A side prediction for S5, withdrawn if the clip-start check (S1c) fails |

---

## 2. Baseline: unchanged, LaGoVAD

| Criterion | Pi-VAD (arXiv) | DSANet (AAAI 26) | RefineVAD (AAAI 26) | **LaGoVAD (ICLR 26)** |
|---|---|---|---|---|
| WS · venue · official code | ✅ · ❌ · ✅ | ✅ · ✅ · ✅ | ✅ · ✅ · ✅ | ✅ · ✅ · ✅ |
| Real traffic evaluation | ❌ | ❌ | ❌ | ✅ DoTA / TAD |
| Measured in this project | — | SG-NM on our trunk: gate failed | its trainable-gate idea: worst KIP arm on MSAD | the trunk of A3-ign |

A trunk swap would change several things at once and make the order question unanswerable.

---

## 3. The questions and their estimands

Each step answers one precisely stated question. The estimand is part of the claim.

| Step | Question | Estimand (what the contrast measures) | What it does **not** measure |
|---|---|---|---|
| **S1** | Does the size of step-to-step change rank anomalous steps above normal ones within a clip? | Change magnitude **beyond each step's atypicality** (vs the quantile-matched full-pipeline permutation, §4.3) | Order at scales above 1.5 s |
| **S2** | Does step order carry signal in the features A3 feeds its trunk? | Order **finer than ≈ 1.5 s, beyond coarse position and beyond VideoMAE's 1.5 s bag**, readable by a fixed-capacity temporal model. Secondary read on `[x̃; p]` (CLIP only): the same without the bag | Lag-2 order (§4.1, residual); order across blocks |
| **S3** | Do explicit velocity / acceleration features add to that temporal model? | (a) **Value:** explicit κ beyond what the probe learns from the same features; (b) **order:** κ beyond its own atypicality | Whether the full trunk would gain (S5) |
| **S4** | Does the trained trunk use step order? | Fine-scale order beyond coarse position, **for A3-ign as trained** | Whether order *could* be used better (S3, S5) |
| **S5** | Does TDP improve the deployed model, and does the gain depend on order? | T − A3-ign; T − Tπ (order beyond atypicality) | Physical motion |

**Why "beyond VideoMAE's 1.5 s bag".** `ũ_t` summarises the frames of the true past 1.5 s as a bag (D6), and it travels with step `t` under the block shuffle. A probe can therefore compare `x̃_t` with `ũ_t` at a single step and read "recent change" without any step order. S2 on `F0` thus leans toward null by construction. The secondary read on `[x̃; p]` has no such bag; it is reported beside S2 and does not gate (the gate stays on the features the trunk actually receives).

The quantile-matched control (S1, S3b, S5) asks "what does order add beyond each step's atypicality". Accident steps are also highly atypical, so these contrasts are **expected to lean toward null**; that is the price of a control that keeps each step's content.

---

## 4. The two controls

### 4.1 Adjacency-free block shuffle (S2, S4)

**Blocks.** Steps are split into consecutive blocks of **5 steps** (≈ 1.35 s on DADA at 0.27 s/step, ≈ 1.5 s on DoTA at 0.30 s/step), the scale of D6's 16-frame window.

**Permutations.** Inside each block the steps are reordered by a permutation in which **no two originally adjacent steps end up adjacent, in either direction**. For n = 5 there are **14** such permutations of 120 (11.7 %). The same constraint holds **across block boundaries**: the last step of block j and the first step of block j + 1 must not be originally adjacent. Blocks are sampled left to right by rejection, with a fixed seed per sequence.

**Why the constraint.** In an unconstrained random permutation of 5 steps, each originally adjacent pair stays adjacent with probability 1/5 in the same direction and 1/5 reversed; `|Δ|` is reversal-symmetric, so on average **1.6 of the 4** within-block adjacencies keep their exact one-step change (*computed by enumeration*). That dilutes the test toward null.

**What survives (stated residual).** In the 14 valid permutations, slot-adjacent steps are on average **2.29 of 4** times originally 2 steps apart (≈ 0.6 s; *computed by enumeration*). No n = 5 permutation avoids lag-2 neighbours. The test therefore destroys one-step (≈ 0.3 s) order and leaves lag-2 neighbours; because adjacent features are highly similar, lag-2 changes resemble lag-1 changes, which also biases toward null. This is stated in every read-out.

**Position.** A step moves at most **4 slots** (*enumeration*). `p_T2` changes negligibly over 4 of ≈ 33 steps, edges stay edges, and the original position `p` moves with the step wherever `p` is a feature.

### 4.2 Ragged blocks and the unit of shuffling

| Sequence length `n` | Rule |
|---|---|
| multiple of 5 (e.g. training windows, `L` = 20 → 4 blocks) | blocks of 5 |
| `n = 5q + r`, `q ≥ 1`, `r ∈ {1, …, 4}` | the first `q − 1` blocks have 5 steps; the **last block has 5 + r steps (6–9)**. Valid permutations exist for every such size (90, 646, 5,242, 47,622 for 6–9; *enumeration*). Displacement in the last block is up to 8 steps (≈ 2.4 s) |
| `n = 4` | one block, one of the 2 valid permutations |
| `n ≤ 3` | **not shuffled** (no valid permutation exists for 2 or 3); the number of such sequences is reported |

**Unit of shuffling.**
- **Trunk (S4):** each training **segment** is shuffled on its own before splicing: an unspliced window is one segment; in a DVS-spliced sequence the anchor and each filler are shuffled separately, then spliced. A window's step set and window label are therefore identical to the control's; only the order differs. Test clips are shuffled per clip.
- **Probe (S2):** each source video (training) or clip (test) is shuffled once on its own timeline; context-padded probe windows are then cut from the shuffled sequence. The probe scores steps, not windows, so the window's step set may differ slightly from the unshuffled read; per-step labels and `p` move with their steps.
- In every case, scores are mapped back to the steps' original positions before any metric is computed.

### 4.3 Quantile-matched full-pipeline permutation (S1, S3b, S5's Tπ)

For a feature built from step-to-step change (‖Δx̃‖, κ), the order-free twin is built as follows:
1. Draw, per sequence and with a fixed seed, an ordering in which **neighbours are at least 6 steps apart in the original order** (beyond the 5-step block and the 1.5 s summaries). It exists for sequences of ≥ 12 steps and for none of 10 or 11 (*exhaustive search*); shorter sequences use the largest feasible gap, which is recorded. The twin is computed per **source video** (training) or **clip** (test), before windowing and splicing, so short DVS segments never need their own ordering; every read-out reports the share of T2-train sources, T2-val sources and DoTA-CAP-dev clips under 12 steps, and their gap distribution.
2. Run **the whole feature pipeline** (differences, smoothing, acceleration, causal max, trace) in that order, so no temporal operation ever reads a true neighbour.
3. Write the results back to the steps' original positions.
4. **Quantile-match per sequence and per channel to the true-order feature:** sort the true-order values of that channel in that sequence and assign them to steps in the rank order of the twin.

The twin then has exactly the true feature's values per sequence and channel, still tied to each step's own content (its distance from far-away steps, i.e. its atypicality), with no adjacency anywhere in its construction. The trunk or probe still sees `x̃`, `ũ` in true order in both arms, so their own access to order is held constant.

---

## 5. The staged plan

### S1 — free diagnostics on cached features (no training)

| Check | What | Decides |
|---|---|---|
| **S1a — change score** | Primary score: `m5_t`, the **1.5 s causal max** of the rate-normalized change ‖Δx̄_t‖ (CLIP only; architecture §3). Secondary, reported: ‖Δx̄_t‖ itself. Per-clip macro AUC against 0.5 and against its quantile-matched twin (§4.3). **Decides on DoTA-CAP-dev** (protocol B); T2-val reported | **Gates the TDP branch only** (S3, S5): both bars (§6.1) on AUC(`m5`) − 0.5 and AUC(`m5`) − AUC(`m5`^π) must pass. A scalar change score failing does **not** show that order carries no signal, so S2 runs either way |
| **S1b — Δu echo** | Event-triggered mean of rate-normalized ‖Δū‖ around span ends: steps +5…+6 after the end vs steps +10…+15 (post-span baseline), DoTA-CAP-dev; autocorrelation of ‖Δū‖ at lags 3–7 beside ‖Δx̄‖'s | **Equivalence rule:** Δu is admitted to κ (S3, S5) only if the cluster-bootstrap 95 % interval of (+5…+6 mean − baseline mean) lies **entirely within ±0.1 × SD** of ‖Δū‖ over post-span baseline steps. Otherwise CLIP only (the default). A noisy check therefore keeps the default instead of admitting Δu |
| **S1c — clip start** | On DoTA-CAP-dev **normal** steps: steps 0–2 vs steps ≥ 3, as a standardized mean difference, on **five summary channels only**: `s`, `m5`, `e`, the channel-mean of `v̄`, the channel-mean of `ā` (not all 131 channels, to avoid a multiplicity problem) | If any of the five has \|SMD\| > 0.1, the early-onset prediction (S5) is withdrawn. Does not stop the plan. Per-channel SMDs of all 131 are printed, not gated |

S1 also reports, for `m5`: its per-clip AUC correlation with `p_T2`, ego vs non-ego, and the share of each clip's top-5 `m5` steps inside the span (how often normal events win).

### S2 — D6-step probe (no trunk training)

- **Probe:** the fixed-capacity temporal probe of architecture §5: two Conv1d layers, kernel 5, 64 channels, ReLU, receptive field **9 steps (±4 ≈ ±1.1–1.2 s)**; trained on T2-train minus T2-val with the step labels earlier probes used; **context-padded windows** (4 real steps each side from the source video, only the central 20 scored), replicate padding only at the true ends of a clip.
- **Input:** `F0 = [x̃ ; ũ ; p]`, what A3 feeds its trunk plus the original position (D6's reference set). `p = [τ, τ², τ³]` with `τ = (t + 0.5)/n` of the step's **original** index (3 channels; the K-pos / E2(d) cubic basis, `core/tools/kill_switch_probe.py:position_features`); `p` moves with its step.
- **Secondary read (reported, not gated):** the same contrast on `[x̃ ; p]` (CLIP only), which has no VideoMAE bag (§3).
- **Contrast:** true order − adjacency-free block shuffle (§4.1), same architecture and seeds.
- **Read:** transfer to DoTA-CAP-dev (decides), in-domain T2-val (reported); `f_2` Δ beside.
- **Runs regardless of S1.**
- **Gate:** the bar (§6.1) on DoTA-CAP-dev, on `F0`. If it fails, **order finer than 1.5 s carries no signal readable by a temporal model in these features, beyond VideoMAE's bag**; S3–S5 are not run, and §9 row 1 applies.

### S3 — κ-probe (no trunk training; runs only if S1a **and** S2 pass)

- **Inputs:** `F0` vs `F0 + κ` vs `F0 + κ^π`, with κ the offline TDP features (architecture §3, standardised per row 10; CLIP only unless S1b admits Δu) and κ^π its quantile-matched twin (§4.3). All in true step order; same probe and seeds as S2.
- **Gates (both, on DoTA-CAP-dev, with the bar of §6.1):**
  - **(a) VALUE:** `(F0 + κ) − F0`. The probe already sees ±4 neighbours and has non-linearities, so this measures what explicit magnitude and second-difference features add beyond what it learns itself.
  - **(b) ORDER:** `(F0 + κ) − (F0 + κ^π)`.
- **Outcome:** both pass → S5 is eligible. (a) passes, (b) fails → the explicit features help through atypicality, not order; S5 is not run (a cheaper route to atypicality exists: CRN already supplies it).

### Advisor check (before any training run)

S1–S3 cost minutes on cached features. Before S4 (10 runs) or S5 (10 more), the outcome table (§10) goes to the advisor: whether a diagnostic chapter is an acceptable contribution when no new component results.

### S4 — D6-step on the trunk (10 runs)

- **Trigger (one rule):** S4 runs **only if S2 passes and the advisor approves** the training budget. It does not depend on S1a or S3. If S2 fails, S4 is not run.
- **Arms:** A3-ign **re-run in the current commit** (5 seeds) and A3-ign trained and tested with the adjacency-free block shuffle (§4.1–§4.2; segments shuffled before splicing; clips at test) (5 seeds). The re-run is the control for S5 as well.
- **Contrast:** A3-ign − A3-ign-shuffled on DoTA-CAP-dev macro, with the trained-arm bar (§6.1); `f_2` Δ beside; T2-val reported.
- **Reading:** a pass means *the trained trunk uses order finer than 1.5 s beyond coarse position*; the credit goes to LaGoVAD's trunk. A null with S2 positive means the information exists but this trunk does not use it, which strengthens the case for S5.

### S5 — TDP (10 runs; only if S1a, S2, S3 (a) and S3 (b) pass, and the advisor approves)

- **Arms:** **T** (A3-ign + TDP) and **Tπ** (same network, κ replaced by its quantile-matched twin), 5 seeds each; the S4 re-run of A3-ign is the shared control. 
- **Adoption of T:** the trained-arm bar on T − A3-ign; `f_2` Δ point estimate ≥ 0; guardrails (T2-val macro not more than 0.01 below A3-ign; O1′).
- **Order claim:** the trained-arm bar on **T − Tπ**, `f_2`(T − Tπ) ≥ 0, and the same sign on T2-val.
- **Mechanism metric (pre-registered):** macro AUC within **±3 s of onset** (onset-window AUC), with the trained-arm bar for any "onset" claim. Directional predictions, reported: onset latency falls; tail-clip macro (277 of 569 dev clips) does not fall; ego Δ ≥ non-ego Δ; early-onset Δ above the overall Δ (withdrawn if S1c fails); `p_T2` correlation of per-clip AUC does not rise.

Tπ is never adopted; a Tπ gain would be capacity, and the cheaper way to spend capacity is H4 (V2-B).

---

## 6. Statistics

### 6.1 The bar (one rule, fixed now)

| Step type | Uncertainty source | Bar to pass |
|---|---|---|
| **Parameter-free scores (S1)** | cluster bootstrap over source videos, B = 10,000 | 95 % lower bound **> 0** and point estimate **≥ +0.01** |
| **Probes (S2, S3)** | cluster bootstrap over source videos, B = 10,000, on **seed-averaged** probe scores. The 5 probe seeds are a **stability check only** (per-seed Δs reported; all five should share the sign) | 95 % lower bound **> 0** and point estimate **≥ +0.01** |
| **Trained arms (S4, S5)** | paired t95 over the 5 training seeds (training variance; v2's decision interval) **and** the cluster bootstrap on seed-averaged scores (clip sampling) | t95 lower bound **> 0**, cluster lower bound **> 0**, point estimate **≥ +0.01** |

+0.01 is the bar H3 used ("beyond-position transfer < +0.01 → not worth building"). Every Δ is printed with `p_T2` beside the macro and the `f_2` Δ beside the Δ.

### 6.2 Sets and their history

| Set | Status | Role in v2.2 |
|---|---|---|
| T2-train − T2-val (1,053 sources) | training | probe and trunk training |
| T2-val (219 sources) | guardrail, pilot | in-domain read; **same sign required** for any order claim |
| **DoTA-CAP-dev (569 clips)** | the decision set of ≈ 6 earlier choices | decides every gate |
| DoTA-CAP-eval, T2-test | opened (twice / once) | a read there is a **third look**, labelled; decides nothing |

**Why the dev overuse matters less here.** Earlier dev decisions shaped A3-ign, which both arms of every v2.2 contrast share; selection bias largely cancels in an arm contrast, unlike in an absolute score. Credibility is added by (i) the T2-val same-sign requirement, (ii) the specific, directional mechanism pattern of S5, and (iii) the DF8 position-shifted re-cut of CAP-dev, an evaluation condition nobody has looked at, on which the prediction "the S5 gain holds" is registered now if DF8 is run.

---

## 7. TDP — the conditional component (built only if S1a, S2 and S3 pass)

### 7.1 Definition

**Offline, per source video (training) or clip (test), before windowing and before DVS splicing:**

```
Δx_t = x_t − x_{t−1}                                       raw CLIP; identical to Δx̃ up to the scalar s (CRN cancels)
m_x = median_{t ≥ 1} ‖Δx_t‖                                per-clip rate scalar; padded steps excluded
Δx̄_t = (Δx_t / m_x) ⊘ σ_Δ                                  σ_Δ: per-channel std on T2-train, fixed
v_t = |Q Δx̄_t|                                             Q: fixed top-64 PCA basis of Δx̄ on T2-train      (64)
v̄_t = mean(v_{t−2}, v_{t−1}, v_t)                          fixed causal jitter filter (replaces IAM's gate)  (64)
ā_t = mean over t−2..t of |v̄_τ − v̄_{τ−1}|                 acceleration on the filtered velocity, as in IAM (64)
s_t = ‖Δx̄_t‖,  m5_t = max(s_{t−4..t}),  e_t = trace of s, half-life ≈ 1 s     causal summaries            (3)
κ_t = [ v̄_t ; ā_t ; s_t ; m5_t ; e_t ] ∈ ℝ^131            (+131 for Δu if S1b admits it)
replicate padding at the κ level: where an input is missing (t = 0 for velocity; t ≤ 1 for acceleration),
  copy the first valid value; partial windows elsewhere; ≤ 2 future steps used, at a clip start only
κ̂_t = (κ_t − μ_κ) ⊘ σ_κ                                   μ_κ, σ_κ ∈ ℝ^131: per-channel mean / std over T2-train steps, fixed
```

**Standardisation is baked offline.** The channels of κ have very different scales (`s`, `m5`, `e` are norms of a 512-d vector; `v̄`, `ā` follow the PCA eigen-scales), so κ̂, not κ, is stored in the cache and read by **both** the probe (S3) and the model (S5): the two see identical numbers. For Tπ, `K^π` holds `K`'s values per sequence and channel (§4.3), so the same `μ_κ`, `σ_κ` apply to it unchanged. The statistics are fitted once, on true-order κ over T2-train, and are shared by both arms.

**In the model (pointwise, no temporal operation):**

```
g_t = GELU( W_1 κ̂_t + b_1 )              W_1 ∈ ℝ^{128×131}, default init
h_t = x̃_t + W_u·(c·ũ_t ⊘ σ_u) + W_k g_t    W_k ∈ ℝ^{512×128} (+ bias), INITIALISED TO ZERO
```

Parameters: 16,896 + 66,048 = **82,944 ≈ 0.08 M** (*derived*).

### 7.2 Why each choice

| Choice | Reason |
|---|---|
| **Offline temporal ops, pointwise in the model** | No operation reads across a DVS splice junction or a window start; the probe (S3) and the model use **identical** features; the Tπ twin is exactly order-free (§4.3). Costs IAM's learned projection and learned gate, replaced by a fixed PCA basis and a fixed causal mean: a small price for a clean test |
| **IAM's velocity / acceleration, on the filtered velocity** | The paper-supported construction (LAS-VAD Sec. 3.2), kept in its fixed form |
| **Causal summaries `m5`, `e`** | Velocity is a transient at onset; the summaries let a step "remember" a recent change for ≈ 1–1.5 s, which the "spike then recover vs stay deviated" argument needs |
| **CLIP only by default** | Differences of a moving average (the 1.5 s VideoMAE window, ≈ 80 % overlap) spike when an event enters **and** when it leaves, ≈ 5 steps later; `|·|` makes both positive and the exit echo lands in post-span normal steps, where H2(a) lost −0.125. Single-frame CLIP has no echo |
| **First differences** | The other textbook fixed-effects estimator beside CRN: a per-clip constant cancels; no reference that can sit inside the accident; causal |
| **Per-clip rate normalization** | `|·|` gives κ a level, the clip's typical rate of change (ego speed, mounting, frame rate): a corpus signature of the kind CRN removes. The median is robust because impacts are short bursts |
| **Fixed per-channel standardisation of κ, baked offline** | Puts norm-type and PCA-type channels on one scale for `W_1`'s default init, and makes the probe's and the model's κ literally the same numbers. It is a fixed affine map per channel, so it does not touch the magnitude of change within a channel (unlike a per-token LayerNorm) |
| **Zero-init `W_k`, no LayerNorm** | Starts at A3-ign's function; the magnitude of change is the signal. The share `ρ_k = ‖W_k g‖ / ‖x̃‖` is logged every 50 steps |
| **Seeded generator for TDP init; zero-effect unit test; re-run control** | TDP's initialisation must not shift the RNG stream of the trunk. A unit test checks that a frozen TDP (`requires_grad = False`) with `W_k = 0` reproduces the control's first 100 losses. The control is re-run in the same commit anyway (S4) |
| **No new loss** | G4 |

### 7.3 Risks

| Risk | Caught by |
|---|---|
| Normal within-clip events (turns, tunnels, lighting, others braking) outrank onsets | S1a's top-5-in-span share; ego/non-ego split; the gates |
| The gain is capacity, not order | S3(b), S5's Tπ |
| Onset sharpening at the cost of span coverage | onset-window AUC and the tail-clip prediction (S5) |
| A clip-start artefact drives the early-onset result | S1c |
| A corpus shortcut through κ | rate normalization; DADA↔DoTA separability of κ reported in S3 |

---

## 8. Ideas declined

### 8.1 IAM's prototypes, `w_int` and the cross-intention contrastive loss `L_cst`

| Problem | Detail |
|---|---|
| **Needs frame-level class labels; hard mining picks the wrong ones** | Hard positive = least-similar same-class frame; hard negatives = most-similar other-class frames. With window labels only, about 46 % of an abnormal window's steps are positive (*derived*, T2 study). The most similar "other-class" step to an anomalous step is almost always its neighbour (cosine 0.968), at the label boundary where pseudo-labels are least reliable. E4-0 gained +0.040 by removing exactly such uncertain dense labels |
| **Classes do not support it** | 83 DADA category strings, the largest 12.2 %, ≈ 1,053 sources: most classes have 0–1 examples per batch of 64. Collapsing to normal/abnormal brings the label problem back: only "normal" step labels are trustworthy |
| **Broken as written** | No temperature, no feature normalization, positive excluded from the denominator: the loss is `−s⁺ + log Σ exp(s⁻)` on raw dot products and goes to −∞ by growing norms, the very magnitudes KAT-VAD keeps un-normalized. Missing from the paper's Eq. 9; weight, α, M unspecified |
| **An auxiliary trunk loss** | G4: ρ = 3.1, every task loss +24–32 %. Prototype normality (SG-NM) failed here: Δ −0.012, score correlation 0.85 with `y^bin` |
| **Would confound the order test** | It acts on the intention feature, which contains the velocity/acceleration channels, so it would reshape them differently in T and Tπ |
| **Benefit not isolated** | LAS-VAD ablates IAM as a whole (XD-Violence AVG mAP 24.24 → 29.98), not `L_cst` |

A repaired, binary, normalized version with trusted normal anchors and a ρ cap is a possible later experiment, not part of v2.2.

### 8.2 ACC

Not a motion component (order is ignored by construction). Its threshold τ = 0.9 against an adjacent cosine of 0.968 connects every adjacent pair, so transitive chaining puts a 20-step window, crash included, into one component (*derived*). Its labels are the model's own averaged scores (cf. E4a); it would put dense pseudo-labels back on the anchor steps E4-0 gained +0.040 by excluding; it is a new trunk loss without warm-up.

### 8.3 Local Transformer

Its purpose (keep attention local in videos of hundreds of snippets) is already met: 20-step training sequences (5.3 s), whole DoTA clips of ≈ 33 steps. Replacing the encoder breaks the 25 paired runs every comparison rests on. Its gain is not isolated in the project's evidence (LAS-VAD inherits it from VadCLIP).

### 8.4 Similarity GCN

On these features its softmax adjacency is near-uniform (≤ 1.105× spread, *derived*), so it approximates window mean-pooling and re-injects the fixed effect CRN removed (corpus shortcut on `V^t` 0.9998 → 0.57). It is a smoother; smoothing failed twice (H2(a) −0.071; SG-NM). Its strength depends on the number of steps (20 in training, ≈ 33 at test), the G3 failure mode. The temporal-distance channel duplicates RoPE and the kernel-3 head.

### 8.5 LLM attribute text

Not the measured bottleneck; `L_neg` is off; text changes belong to DF3.

---

## 9. Claim ladder (pre-written wording)

| Outcome | Permitted wording | Forbidden |
|---|---|---|
| **S2 fails** | "Neither frame order inside the 1.5 s video window (D6) nor step order at the same scale (D6-step) carries detection signal in the frozen features beyond VideoMAE's 1.5 s bag. KAT-VAD makes no motion claim." (If the CLIP-only secondary read is positive, add: "Without the video stream, step order is readable from CLIP features; the video stream's bag already supplies it.") | any motion / kinematic wording |
| **S1a fails, S2 passes** | "Step order finer than ≈ 1.5 s carries signal beyond coarse position, but the magnitude of step-to-step change alone does not rank anomalous steps beyond their atypicality." TDP is not built | "TDP" as a contribution |
| S2 passes, S3(a) fails | "Step order finer than ≈ 1.5 s carries signal beyond coarse position, but explicit velocity/acceleration features add nothing a small temporal model does not learn itself." | "TDP" as a contribution |
| S3(a) passes, S3(b) fails | "Explicit change features help, through atypicality rather than order." | "order-sensitive" |
| S4 passes | "The trained LaGoVAD trunk uses step order finer than ≈ 1.5 s, beyond coarse position." (credit: the baseline) | "KAT-VAD's motion component" |
| S5: T − A3-ign passes, T − Tπ fails | "TDP improves DoTA-CAP-dev detection; the gain is not attributable to order beyond atypicality." | motion / order wording |
| S5: both pass, T2-val same sign | "**Velocity and acceleration of the scene representation (from LAS-VAD's IAM) improve detection, and the gain depends on temporal order beyond each step's atypicality.**" | "vehicle motion", "optical flow", "trajectories", "speed" |
| … and `f_2`(T − A3-ign) interval excludes 0 | "… beyond the position prior" | — |
| … and the onset-window bar passes | "… mainly at the onset" | "covers the span" |
| any eval number | labelled "**third look at an opened set**" | an unlabelled eval number |

Every order statement carries its estimand: "finer than ≈ 1.5 s", "beyond coarse position", "beyond atypicality", and, for S2/S4, "lag-2 neighbours remain" (§4.1).

---

## 10. Outcomes and the thesis-level risk

| Outcome of S1–S3 | What the thesis gets | Training runs |
|---|---|---|
| Order carries no signal (S2 fails) | A clean two-level negative (D6 + D6-step) as an analysis chapter; the contribution remains v2's components (CRN + video stream: A3 − A0 +0.090, A3-ign − A3 +0.038, A3-ign − A0 +0.128 on the second look) and the position-controlled evaluation | 0 |
| Order carries signal; explicit change features do not (S1a or S3 fails) | Analysis chapter; S4 (if the advisor approves) to credit the trunk | 0–10 |
| Order carries signal; explicit κ adds beyond the temporal probe (S1a, S3 pass) | TDP as a component with a measured motivation | 20 (S4 control shared) |

Only the third row yields a new component. A pre-registered chain (KIP fails → the video stream works but is appearance → step order does or does not matter) is a legitimate master's contribution **on top of** v2's components; whether the programme accepts an analysis chapter in place of a second component is the **advisor's** decision, taken before any training run.

---

## 11. Multi-modality, LLM, data

- **Modalities:** unchanged (CLIP frames, VideoMAE V2-S clips, language definitions). κ is **not a modality**: a deterministic function of cached CLIP features. Flow, depth, pose, panoptic, audio: not used (v2 §5). Flow returns only if S5's order claim passes, as an input, never a regression target.
- **LLM:** Holmes-VAU ATS, unchanged and detachable.
- **Data and protocol:** unchanged. T2 (DADA-2000) windows W = 20, hop 8, stride 8 (0.27 s/step); DoTA-CAP at stride 3 (0.30 s/step), whole clip. Protocol B is what makes a per-step change comparable across corpora; the rate normalization absorbs the remaining 10 %. All new features come from **existing caches**: no extraction.
- **TAD (deferred, generalization study):** a test-only macro read is plausible (the recorded leak concerns training and micro AUC; the released checkpoint reproduced macro 0.7578), but it needs VideoMAE V2-S features, a rate-matched stride and a frozen split — a small P0/P1 project. TAD's fixed cameras have no ego-motion, so a global change signal is *cleaner* there and non-ego order signal easiest to see: a different regime, not a replication of a dashcam claim.

---

## 12. Justification ledger

| Decision | Rationale | Source |
|---|---|---|
| Keep LaGoVAD and A3-ign | Traffic-native WS baseline; 25 paired runs on this trunk | LaGoVAD; results §4 |
| Answer the order question before building | The trunk can already form signed differences; what explicit features add is empirical | R1.2; arch v2 §5, §8 |
| Block shuffle of 5 steps (≈ 1.5 s) | Preserves position (≤ 4 slots); destroys one-step order; D6's scale one level up | R2.1; D6 |
| Adjacency-free permutations (14 of 120), also across block boundaries | An unconstrained permutation keeps 1.6 of 4 adjacencies | R3.1; enumeration |
| Ragged-tail rule (last block 6–9 steps; n ≤ 3 unshuffled) | Valid permutations exist for 4–9, not for 2–3 | R3.2; enumeration |
| Shuffle segments before splicing; scores mapped back | Window step sets and labels identical to the control; no junction effects | R3.2 |
| Quantile-matched full-pipeline permutation | Order-free at every stage; identical values per sequence and channel; keeps per-step content | R1.3, R2 |
| Temporal probe (2 × kernel 5, RF 9), context-padded windows | Matches the trunk's access to neighbours; no padding cue | R1.2, R1.4, R2.2 |
| One bar (LB > 0, point ≥ +0.01); cluster bootstrap for probes; t95 + cluster for trained arms | Numeric, pre-registered; the right uncertainty source per step | R3.4, R3.5; H3 bar |
| Only measured contrasts quoted | Sums of paired contrasts have no interval | R3.3 |
| TDP offline + pointwise; CLIP only; κ-level replicate padding | Splice-safe; probe = model; no echo; no clip-start spike | R1.1, R1.5, R1.6 |
| S1a gates only the TDP branch; S2 always runs; S4 trigger = S2 passes + advisor | A scalar change score cannot speak for order; one unambiguous trigger per training step | R4.1, R4.5d |
| S2 estimand "beyond VideoMAE's bag"; CLIP-only secondary read | `ũ_t` carries the past 1.5 s as a bag through the shuffle | R4.2; D6 |
| κ standardised per channel on T2-train, baked offline | Probe and model read identical numbers; one scale for `W_1` | R4.3 |
| S1b equivalence margin (±0.1 SD) | A noisy check must not admit Δu | R4.4 |
| S1c on five summary channels | No 131-way multiplicity | R4.5c |
| Seeded TDP generator; zero-effect test; re-run control | Paired-seed integrity | R1.7, R2.3 |
| Advisor check before training | Only one outcome yields a component | R2.5 |
| Decline `L_cst`, prototypes, ACC, local Transformer, GCN, attribute text | §8 | G1, G2, G3, G4; E4a; E4-0; H2(a) |
| ATS off-path | Zero detection latency | Holmes-VAU |

---

## 13. Open hooks

- **K0 (descriptive):** permute steps at inference on the A3-ign checkpoints; confounded by distribution shift, decides nothing.
- **Longer-range order:** blocks of 10 steps (≈ 3 s) if S2 is null at 1.5 s and the advisor wants the scale question answered.
- **Signed differences** and **multi-lag velocity** (1, 3, 5 steps), only after S5 passes.
- **Streaming:** past-only medians for `m_x` and CRN (R4).
- **κ for the reporter:** pass the argmax of acceleration to ATS.
- **Unchanged from results §8:** H4 (V2-B), H1/DF8 (position shift), DF3 (`L_neg`), detector crops for non-ego (v3). None is combined with S4/S5.
