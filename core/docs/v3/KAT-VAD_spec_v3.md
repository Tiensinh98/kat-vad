# Proposed Road-Traffic Video Anomaly Detection Architecture — **KAT-VAD v3**
### (LaGoVAD baseline + Kinematic Induction Pathway — gate rebuilt, trunk decoupled, flow-bearing corpus expanded)

*Supersedes `KAT-VAD_spec_v2.md`. Revision driver: `REPORT_KIP_MSAD_DOTA_PREVAD.md` §12 and §§13–16, audited in `KAT-VAD_audit_addendum_PreVAD.md`. v2's §§1–6 and §§9–12 stand except where marked; §§4, 7, 8, 10 change materially.*

---

## What changed and why — v2 → v3 diff

| # | Change | Triggering evidence | Section |
|---:|---|---|---|
| **A** | **The gate is rebuilt, not just re-signalled.** v2 proposed feeding an ego-compensated residual into the existing MLP and called it "nested, so it cannot destroy the +0.09." **That safety argument is void:** the gate MLP receives no gradient (`(ratio*max_shift).floor().long()` is non-differentiable), so it is frozen at random init and cannot learn to weight a second input channel. v3 replaces it with a **parameter-free deterministic rank gate** (default) or an **STE-trainable MLP** (variant). | report §12.5 | §4.2 |
| **B** | **`s_t` logging promoted to the first experiment in the project.** A frozen random `σ(MLP(·))` on a scalar plausibly saturates ⇒ `s_t` ≈ constant per domain ⇒ KIP may be a *domain-conditioned constant smoother* (H4′), not an adaptive gate. Costs zero training and reorders everything downstream. | report §12.5 + §11.4 + §8.2 | §10.3 |
| **C** | **Plain-TSM control added as a required baseline.** If H4′ holds, the correct comparison is a fixed-ratio temporal shift with no flow at all. It has never been run. | H4′ | §10.3 |
| **D** | **PreVAD reclassified: trunk source + Gate-P0 benchmark only, permanently.** It ships CLIP features and no pixels; RAFT cannot be built; ~3,800 dead source rows are the traffic captures; re-downloads cannot be paired with the released features. v2's "move training to PreVAD scale" is **impossible for the KIP arm** and is withdrawn. | report §12.7 (settled 2026-08-29) | §7.1 |
| **E** | **New default training recipe: PreVAD trunk (KIP-off, features) → KIP stage-1 on flow-bearing data → joint stage-2.** This is the report's own §12.6 unblocking run promoted from "one run" to the standing recipe, because it is the only way to get broad-domain pretraining and a trained KIP simultaneously. | report §12.3, §12.6 | §8 |
| **F** | **Flow-bearing corpus expanded ~4–12× without PreVAD**: UCF-Crime (1,610), DoTA-train, BDD100K-normal, optionally XD-Violence / A3D / CCD / DAD — all ship pixels. This is the *only* remaining route off the 480-video KIP training corpus. | audit §5.1; report §12.7 | §7.2 |
| **G** | **Composition (trunk × KIP) elevated to a primary reported result**, not an ablation row. A 2×2: trunk {MSAD, PreVAD} × KIP {off, on}. KIP must clear **0.5868** DoTA from the PreVAD trunk to have a contribution at all. | report conclusion 8; audit addendum §5 | §10.2 |
| **H** | **Row-gated `L_KIP_rec` specified** for mixed feature-only / pixel-bearing corpora — *mask* the loss, never zero-fill `e_O`. | report §12.7's explicit trap | §8, §12 |
| **I** | **Precondition assertions added to the training pipeline** (module-trained gate, flow-cache coverage, transform parity, gate-gradient check) plus a run manifest. | 3 campaigns, 3 multi-variable changes, 2 void results; limitation 11 upgraded to High | §8 |
| **J** | **Calibration blocker on the LLM layer lifted for PreVAD-trunk arms**; MCC/AUCMCC become reportable there. | report limitation 8: 86–89 % → 38–48 % saturation | §6, §10 |
| **K** | **Metrics computed in float64**; the ours-vs-released claim restated only after rescoring. | report limitation 15 + §11.3 tie asymmetry (270 / 93 / 1) | §10.1 |
| — | *Unchanged from v2:* baseline (LaGoVAD), modality set, ECMR *signal* (§4.2), respecified `L_KIP_align` (§4.3), scale-free `L_kin` (§4.4), `mhead` off the inference graph, induction-parity rule, ATS pattern, Alert-CLIP swap, validation-split requirement. | | |

**Discipline unchanged:** no change here is aimed at raising the +0.09. Changes A–C are *diagnostic* — they exist to find out what the module does. D–F are forced by a hard data constraint. G–K are measurement hygiene.

---

## 0. Design summary

KAT-VAD is a **weakly-supervised, language-definition-conditioned traffic anomaly detector** on **LaGoVAD (ICLR 2026)**, whose documented weakness — no motion pathway, weakest score on ego-centric DoTA — is targeted by the **Kinematic Induction Pathway (KIP)**: Pi-VAD-style train-time-only optical-flow induction (RAFT supervises training; inference is RGB-only) driving a RefineVAD-style motion-adaptive temporal shift.

**v3's two substantive corrections.** First, the shift gate is rebuilt: it was a frozen randomly-initialised MLP that never received gradient, so v2's plan to teach it a new input signal could not have worked. It becomes a **parameter-free deterministic rank map over an ego-compensated motion residual**, which is trainable-free, seed-independent, loggable, and reduces KIP's score-path parameter count to **zero**. Second, the training recipe splits the two levers that were confounded: a **PreVAD-pretrained trunk** (features only — PreVAD ships no pixels and never will) supplies broad-domain semantics, and **flow-bearing corpora** (MSAD, UCF-Crime, DoTA-train, BDD100K-normal) supply the motion supervision KIP needs.

Modalities: **RGB (inference) + optical flow (train-time only, induced) + language definition**. LLM off the critical path (Holmes-VAU ATS). Backbone: Alert-CLIP swap — **deferred, see below**.

> **As-built status, 2026-08-30.** CHANGE A (the gate rebuild) is **implemented**: four selectable `kip.gate_type`s, `rank` the default, `mhead`/projections off the inference graph, `s_t` logged into every score `.npz`. See `.project/plans/katvad-v3-kip-gate-rebuild.md` Appendices A–E for the measurements. **The Alert-CLIP swap is deferred** — no public checkpoint exists — so the backbone is stock CLIP ViT-B/16 at the pinned revision and every v3 number is a stock-CLIP number. CHANGES D–K (training recipe, corpus, metrics) are **not** implemented; they are the next plan.

**Claim currently supported by evidence:** *from an MSAD-pretrained trunk with a stage-1 KIP warm-up, a train-time-only motion-induction pathway improves cross-domain temporal localization (fixed-camera → ego-centric) by ≈ 9 AUC points, seed-level CI [+0.070, +0.113], at no measurable in-domain cost.*
**Two sentences that must accompany it:** the gain has not been shown to survive a broad-domain trunk, and broad-domain pretraining moves the same benchmark on its own (+0.038, directional). And the module's inference-time gate has never received a gradient, so whether it is *adaptive* or a *domain-conditioned constant smoother* is undetermined.

---

## 1. Gap analysis

*Unchanged from v2* — see `KAT-VAD_spec_v2.md` §1. Two annotations added:

- **RefineVAD** — MoTAR Eq. 2 is also `s_t = ⌊r_t·D/K⌋`. Either the authors use a straight-through/soft relaxation not described, or the same dead-gate condition exists in the source paper. **Read their implementation before finalising §4.2** — it either validates the design or becomes a publishable observation about an AAAI 2026 method.
- **LaGoVAD** — its released `ViT-B-16-8p` features are now verified compatible with our pipeline end-to-end (Gate P0 passes on PreVAD test: Δ AUC −0.0025, CI [−0.0103, +0.0047]). PreVAD is a usable *trunk corpus*; it is not a usable *KIP corpus*.

---

## 2. Baseline selection

*Unchanged. LaGoVAD stands, and the evidence for the port strengthened again:* reproduction now passes on **three** benchmarks (MSAD, DoTA-protocol, PreVAD test), and with a PreVAD trunk our KIP-off arm moves **above** the released checkpoint in-domain (AUC +0.0039, AP +0.0604). **Caveat carried from the audit:** that comparison crosses a saturation boundary (tie counts 270 / 93 / 1), so the "paired ⇒ shared dtype bias" argument does not apply to it. Rescore in float64 before publishing the claim (§10.1).

---

## 3. Proposed architecture

Data flow unchanged from v2 except inside KIP:

```
video V ──► Frozen Alert-CLIP ViT-B/16 ──► F ──► Temporal encoder ──► v^t
                    ▼
   ┌───────────────────────── KIP ─────────────────────────────────┐
   │ (a) PMG-flow head:   v^t → ê_O ∈ ℝ^{L×256}   ← RAFT-supervised │
   │ (b) ECMR signal:     ê_O → μ_t → δ_t → m_t^res                 │  ◄── v2
   │ (c) GATE:            m_t^res → r_t                             │  ◄── REBUILT in v3
   │ (d) adaptive shift:  v^t, s_t=⌊r_t·D/K⌋ → v^k                  │  ← only score-path effect
   │ (e) motion head:     ê_O → ŷ_O                [TRAIN ONLY]     │
   └────────────────────────────────────────────────────────────────┘
                    ▼
   definition Z ──► Frozen CLIP text ──► z^t ──► Co-attention ──► v^u ──► H_bin → y^bin ; H_mul → y^mul
                                                                              ▼ (async) ATS → MLLM → report
```

Components (a), (d), (e) and everything outside KIP are **unchanged from v2**. (b) is v2's ECMR signal. (c) is new.

---

## 4. Novel component / loss

### 4.1 Evidence status, updated

| | Status |
|---|---|
| +0.0915 ± 0.0088 DoTA micro AUC, 3 seeds, cold and warm controls | **Established — under one trunk** |
| Warm-start and under-convergence eliminated | **Established** |
| Motion-mediated (transform ablation: +0.1514 to KIP-on, ≈0 to others) | **Strongly indicated**, n = 1 seed |
| No measurable in-domain cost | **Bounded null** (< ≈1 AUC point) |
| Ego-kinematics mechanism | **REFUTED 6/6** |
| **Survives a broad-domain trunk** | **UNTESTED** — the campaign that would have tested it removed KIP's stage-1 warm-up simultaneously; `L_KIP_rec` ended at 10.2–13.3 vs 4.91, `L_KIP_align` never left init |
| **The v1 gate is adaptive rather than a constant smoother** | **REFUTED, 2026-08-30.** Its input is min-max normalized, so `[0,1]` is the whole reachable domain; sweeping it moves `s_t` by 0–4 of 128 channels across 8 seeds (seed 0: exactly 0). v1 KIP-on was a fixed ~50 % smoother at `s ≈ 58–69`. **H4′ confirmed.** |
| KIP architecture with an *untrained* flow head | **Buys nothing** (−0.002 ± 0.016), confounded by the trunk moving too |
| Which sub-module produces the gain | **Unmeasured** |
| `L_KIP_align` contributes | **Refuted in practice** (13–15 % below chance) |
| FP suppression | **Unmeasured** (test set 99.8 % abnormal) |

### 4.2 CHANGE A — the gate, rebuilt

**Why v2's plan fails.** v2 proposed `r_t = σ(MLP([m^res_t, m^abs_t]))` and argued that zeroing the residual channel recovers v1, so the change was safe. **That argument required the MLP to learn.** It does not: `s_t = ⌊r_t·D/K⌋` is non-differentiable and used as a slice index, so no gradient reaches the gate MLP — or, through it, the PMG head. Adding a second input to a frozen random function does not degrade gracefully; it randomly perturbs the shift schedule. The nesting safety property is void.

**Default — deterministic rank gate (0 parameters).**

```
μ_t     = Σ_{τ≤t} (1−λ)λ^{t−τ} · ê_{O,τ}                 causal EMA, λ = 0.9        [0 params]
δ_t     = ê_{O,t} − μ_t                                  ego-compensated residual
m_t     = ‖δ_t‖₂
r_t     = rank_t(m) / (L−1) ∈ [0,1]                      within-clip rank map       [0 params]
s_t     = ⌊ r_t · D/K ⌋,  K = 4  ⇒ s_t ∈ [0,128]
v^k_t   = [ v^t_{t−1}(1:s_t), v^t_{t+1}(s_t:2s_t), v^t_t(2s_t:D) ]                  RefineVAD Eq. 3
```

Justification, point by point:

- **Removes a random seed-dependent function from the inference path.** Three seeds currently carry three *different* frozen gate functions, yet the delta is stable to ±0.009 — which is itself evidence that the specific random draw barely matters, and therefore that replacing it with something deterministic costs nothing and buys interpretability.
- **Rank-normalisation is inherently scale-free**, which supersedes v2's `median` normaliser and fixes the fixed-camera/ego-camera magnitude mismatch directly. It also guarantees `s_t` spans its full range on *every* clip, which structurally **rules out H4′ by construction** — a constant smoother is no longer expressible.
- **KIP's score-path parameter count becomes 0.** The efficiency claim strengthens from "+1.65 % params, 321 on the score path" to "**+1.55 % params, none on the score path**" — the entire inference-time effect is a deterministic reindexing driven by a reconstruction-trained flow estimator.
- **It is loggable and interpretable**, closing limitation 1 permanently rather than by one diagnostic run.
- **Source lineage intact:** the residual is DSANet's SG-NM logic (video-specific normal prototype, deviation = anomaly, training-only) applied to the motion channel; the shift is RefineVAD Eq. 3 verbatim; the *differential* form restores what MoTAR's `Var(x_t − x_{t−1})` always had and v1 discarded.

**Variant — STE-trainable MLP.** Keep v1's `σ(MLP(·))` but unblock the backward pass:

```
u_t = r_t · D/K ;   s_t = u_t + (⌊u_t⌋ − u_t).detach()
```

The forward pass is **bit-identical** to the current implementation, so this is a pure gradient intervention — the cleanest possible controlled experiment for the two readings in the audit addendum §3.2. Note the side effect: with the gradient unblocked, `L_MIL` also reaches the PMG head, which risks pulling `ê_O` off the flow manifold (Pi-VAD leaves `L_PMG` unweighted precisely to prevent that). Run it as an ablation, not a default.

**Prediction, pre-registered before running:** under the rank gate the `ego`/`other` asymmetry (§11.1) **shrinks by at least half** and `ego:` classes gain in absolute terms. If the asymmetry is unchanged, the ego-compensation hypothesis is wrong too and the search moves to spatial rather than temporal explanations.

### 4.3 `L_KIP_align` — respecified (unchanged from v2)

Snippet pooling over 4 positions (restoring Pi-VAD's 16-frame granularity), temporal exclusion window ±1 group, negatives subsampled to 64 **including cross-video negatives from the batch** (substituting for Pi-VAD's absent distillation teacher). **Decision rule unchanged:** if the repaired loss does not fall meaningfully below the new chance floor (`ln 64 ≈ 4.16`) and shows no ablation effect, **delete the term and report the negative result** — the +0.09 was obtained while it was inert, so it is provably not load-bearing. Deleting removes 98,560 train-time parameters.

### 4.4 `L_kin` — scale-free (unchanged from v2, now with a relieved precondition)

```
L_kin = L_MIL-topk(ŷ_O) + β · KL( sg(softmax(logit(y^bin))) ‖ softmax(logit(ŷ_O)) ),  β = 0.5
```

Softmax-over-time makes the agreement term depend on *where* the mass sits, not on absolute score level, so saturation cannot neutralise it. **Newly relieved:** saturation drops from 86–89 % to 38–48 % under a PreVAD trunk, so the raw-value form would now partially work — but the scale-free form is correct under both and should be adopted regardless.

`mhead` stays off the inference graph. Parameter accounting:

| | v1 claimed | v2 | **v3 (rank gate)** |
|---|---:|---:|---:|
| KIP inference-path params | 345,154 (+1.82 %) | 312,129 (+1.65 %) | **311,808 (+1.65 %)** |
| **Params on the score path** | — | 321 (0.0017 %) | **0** |
| KIP total (train), align retired | 443,714 | 345,154 | **345,154** |

### 4.5 `L_KIP_rec` — unchanged, still the only loss doing work

λ = 1.0, unweighted (Pi-VAD convention). Extend the stage-1 schedule: it reached within 5 % of its minimum only at step ≈ 446 of 500, so the whole +0.09 was achieved without it converging.

---

## 5. Multi-modality justification

*Unchanged from v2.* RGB (inference) + induced optical flow (train-time only) + language definition. Depth still deferred — adding a second induction head before the first is attributed compounds an unexplained effect.

---

## 6. LLM integration

*Unchanged in design; the blocker moves.* Holmes-VAU ATS, off the critical path, replacing VERA's midpoint prior with content-adaptive weighting on the detector's own peak evidence.

**v3 update:** v2 blocked Stage 3 on calibration, because inverse-CDF sampling over a curve that is >0.99 on 86–89 % of frames degenerates to uniform sampling. Under a PreVAD trunk saturation falls to **38–48 % with a real low tail**, so **ATS is unblocked on PreVAD-trunk arms specifically**. Do not build it on MSAD-trunk arms.

---

## 7. Datasets & preprocessing

### 7.1 CHANGE D — PreVAD's role, fixed permanently

> **PreVAD is a trunk source and a Gate-P0 reproduction benchmark. It is never a KIP training corpus.** `L_KIP_rec` regresses cached RAFT embeddings; RAFT needs pixels; PreVAD ships `ViT-B-16-8p` features only. `data_sources.csv` covers 82 % of rows with a URL, realistic post-link-rot yield is 50–70 %, and the ~3,800 permanently dead rows are the China Expressway Camera captures — the traffic/motion clips KIP cares about most. Even a successful re-download cannot be paired with the released features (different transcode, possibly different fps, `-Scene-NNN` ids imply an unpublished shot-detection pass). A KIP-on arm on a 60 % subset is not comparable to a KIP-off arm on the full release.
>
> `require_flow = cfg.kip.enabled` is the correct hard stop. **Never set `require_flow=False`** — that zero-fills `e_O` and trains the PMG head to predict zeros, producing a wrong run that still yields a checkpoint. The correct mechanism for mixed corpora is **row-gating the loss** (§8), not disabling the requirement.

*(Settled by the user, 2026-08-29; recorded so it is not proposed again.)*

### 7.2 CHANGE F — the flow-bearing corpus, expanded

The KIP module is still trained on **480 videos**, and PreVAD can never fix that. The fix is corpora that ship pixels:

| Corpus | Train videos | Ships pixels | Role |
|---|---:|:---:|---|
| **MSAD Protocol ii** | 480 | ✅ (flow cached) | current; in-domain fixed-camera |
| **UCF-Crime** | 1,610 | ✅ | **highest-value addition** — 4.4× the corpus for one extraction pass; every baseline reports on it |
| **DoTA train split** | ~3,275 | ✅ (frames) | ego-centric; enables a *supervised-domain* KIP arm, not just zero-shot |
| **BDD100K normal driving** | large | ✅ | Stage-0 DAPT + normal-clip negatives for FP measurement |
| **DADA-2000** | ~2,000 | ✅ | second ego-centric transfer benchmark (separates "motion" from "DoTA") |
| XD-Violence / A3D / CCD / DAD | ~4k / 1.5k / 1.75k | ✅ | optional further scaling |

**MSAD + UCF-Crime alone reaches ≈ 2,090 (4.4×); adding DoTA-train and XD-Violence reaches ≈ 9,300 (19×)** — a regime where the memorisation pathology (`mil` → 0.0012, extreme saturation) should not occur, and all of it is reachable with RAFT passes far smaller than PreVAD's 35k.

### 7.3 Full dataset plan

| Role | Dataset | Justification |
|---|---|---|
| **Trunk pretraining (features only)** | **PreVAD train** (32,673) | Worth +0.047 AP in-domain to the plain baseline (seed-level CI excludes zero); trunk already on disk, trained KIP-off |
| **KIP + task training (pixels required)** | **MSAD + UCF-Crime** (+ DoTA-train, XD-Violence) | The only route off 480 videos |
| **Validation / calibration** | **PreVAD val** (1,306 abnormal + 1,300 normal, frame labels) | Only split with frame labels *and* both classes |
| Reproduction gates | MSAD, DoTA-protocol, **PreVAD test** | All three now pass |
| Eval — ego-centric | **DoTA**, **DADA-2000**, A3D | DADA-2000 required (limitation 2) |
| Eval — fixed-camera traffic | **TAD**, MSAD traffic slice (`Traffic_accident` n=20, `road`/highway tags) | MSAD-highway collapses to AP 1.4–4.1 for current methods |
| **Normal-driving negative pool** | BDD100K normal + DADA normal segments + PreVAD traffic-cam feeds | Makes FP suppression measurable for the first time |
| Stage-0 DAPT (unlabeled) | BDD100K normal driving | Normal-only DAPT gives +2.8–4.0 AUCMCC; anomalies add nothing |

### 7.4 Preprocessing — the induction-parity rule (unchanged, one item still open)

> **Induction parity rule.** Any modality induced from a teacher must be supervised by targets computed on the **identical geometric transform** — same crop, same resize, same aspect handling — as the student's input. A cache is bound to its transform; mixing caches invalidates every metric measured on it.

**Still open:** the transform is anisotropic `Resize((224,224))`, squeezing 16:9 ≈ 1.78× horizontally. If RAFT ran at native aspect, horizontal flow is systematically mis-scaled against vertical — a bias, not noise, for a flow-regression target, bearing directly on the 349 lateral DoTA clips. Audit before assuming a defect; this belongs on the checklist that produced the +0.15.

---

## 8. Training pipeline — CHANGE E

**The recipe now separates the two levers that §12 confounded.**

```
S0  (optional)  DAPT — VideoMAE masked reconstruction, 75 % mask, BDD100K normal driving (pixels)
                → encoder internalises normal ego-motion, which is exactly the μ_t the ECMR gate subtracts
S1  TRUNK       PreVAD train, stage-2, KIP-OFF, features only            → broad-domain trunk  [already on disk]
S2  KIP WARM-UP MSAD (+ UCF-Crime) stage-1, KIP only, trunk frozen/low-LR
                L = L_KIP_rec + λ_al · L_KIP_align      ≥ 500 steps
                → trains PMG against the PreVAD trunk's v^t.  THIS IS THE STEP §12.4 OMITTED.
S3  JOINT       MSAD (+ UCF-Crime) stage-2, full objective, ≥ 500 steps, 3 seeds
S4  (optional)  ATS reasoning head, detector frozen — PreVAD-trunk arms only (§6)
```

**Why S1 and S2 must be separate stages rather than one corpus.** PreVAD supplies semantics but no pixels; MSAD/UCF supply pixels but little diversity. Training KIP on PreVAD is impossible (§7.1); training the trunk only on MSAD forfeits +0.047 AP. Splitting them is not a workaround — it is the only configuration in which both levers are available, and it makes the composition question (§10.2) directly measurable.

**Stage-3 objective (unchanged from v2):**

```
L_total = L_MIL + L_MIL-align + L_dvs-sup + L_dvs-supMIL + L_neg          ← LaGoVAD, unchanged
        + 1.0 · L_KIP_rec        (row-gated; see below)
        + 0.1 · L_KIP_align      (respecified §4.3; may be deleted)
        + 0.2 · L_kin            (scale-free §4.4)
```

**CHANGE H — row-gated `L_KIP_rec` for mixed corpora.** When a batch mixes pixel-bearing and feature-only samples, apply a per-sample mask:

```
L_KIP_rec = Σ_i  1[has_flow(i)] · MSE(ê_O^{(i)}, e_O^{(i)})  /  Σ_i 1[has_flow(i)]
```

**Mask the loss; never zero-fill `e_O`.** Zero-filling trains the PMG head to predict zeros — the exact trap §12.7 warns about. Row-gating already exists in the codebase for `L_dvs-sup`, so this is a small change. *Caveat:* on a feature-only batch the PMG head receives no reconstruction gradient and (under the rank gate, which is parameter-free) no task gradient either, so `ê_O` goes stale as `v^t` drifts. Mixed-corpus training is therefore an **iteration hook**, not a v3 default — S1/S2/S3 staging avoids the problem entirely.

**CHANGE I — preconditions that fail the run.** Three campaigns have produced two uninterpretable signed numbers from multi-variable changes; limitation 11 is now High and has cost three trunk reconstructions.

```
assert stage1_final(L_KIP_rec) < τ_rec               # "did KIP actually train?" → catches §12.4 at step 1
assert flow_cache_covers(train_ids)                   # no silent zero-fill
assert hash(feature_transform) == hash(flow_transform) # induction-parity rule
assert grad_norm(θ_gate) > 0   (first step, STE variant only)
emit run_manifest.json {init_weights, cache_hashes, transform, trunk_source, step_budget, seed}
```

The first assertion alone converts a six-run void campaign into a one-minute failure message.

**Also mandatory (carried from v2):** validation-based model selection on PreVAD val; per-epoch calibration logging (fraction of frames > 0.99, within-clip dynamic range) with >50 % treated as a training failure; seed-level t-intervals as the headline statistic with clip bootstrap as a within-seed secondary; **change one variable per campaign**.

---

## 9. Inference pipeline

1. Stream `V` + operator definition `Z` (set once, editable, no retraining).
2. Frozen Alert-CLIP per sampled frame → `F`; temporal encoder → `v^t`.
3. **KIP forward, RGB-only:** PMG regenerates `ê_O` from `v^t` — RAFT not loaded; EMA prototype → residual → **rank map** → `s_t`; adaptive shift → `v^k`. `mhead` not instantiated.
4. Frozen CLIP text on `Z` → `z^t` (cached per definition); co-attention → `v^u`.
5. `H_bin` → frame-level curve `y^bin`; `H_mul` → category probabilities.
6. Threshold + light Gaussian smoothing → incident windows.
7. *(Async, PreVAD-trunk arms only)* ATS → MLLM → incident report.

**Efficiency, restated:** **+311,808 params (+1.65 %)** over 18,923,524, **zero on the score path** (the gate is now a deterministic rank map). RAFT runs offline once as a cached target, never on the scoring path. Report GFLOPs/FPS against Pi-VAD (30.51 FPS / 19.88 GFLOPs) and SimpleTAD (up to 95 FPS); reasoning latency separately. **Also log `s_t` at inference** — it is free and it is the diagnostic (§10.3).

---

## 10. Evaluation

### 10.1 Protocol (v2 carried forward, two additions)

Label-distribution-driven pooling (per-clip min-max when normals < 5 %, raw otherwise) — validated on three datasets now, including PreVAD resolving to raw with no human decision. Keep a released checkpoint in every eval run. Verify label parity and report coverage. Identical clip subsets across arms.

**New: compute all metrics in float64.** `core.evaluate` normalises in float32 and manufactures ties among saturated frames (ΔAP up to 0.0039). The "all deltas are paired" defence holds for on-vs-off but **not** for ours-vs-released, because tie counts differ 270 / 93 / 1 across those arms — the bias is a function of saturation, and that comparison crosses a saturation boundary. `rescore --write` everywhere, then restate the +0.0039 AUC claim.

**New: apply `video_id.replace(":", "_")` on PreVAD joins** — 90 of 2,606 ids contain `:` and a naive join silently drops 3.5 % of the test set.

### 10.2 CHANGE G — composition is a primary result, not an ablation row

Report the full 2×2 as a headline table:

| DoTA micro AUC (3-seed means) | KIP-off | KIP-on |
|---|---:|---:|
| MSAD trunk | 0.5492 | **0.6408** (Δ +0.0916) |
| PreVAD trunk | **0.5868** (Δ +0.0376, directional) | **??? — S2/S3 measures this** |

Interpretation set in advance:

| Outcome | Reading |
|---|---|
| ≈ 0.678 | The levers are **additive**; KIP is a genuine, composable contribution. Strongest possible result. |
| ≈ 0.641 | **Redundant** — KIP and broad-domain pretraining buy the same thing. The contribution becomes "a cheap alternative to 32k pretraining clips," which is defensible but different. |
| ≈ 0.587 | **KIP contributes nothing over a good trunk.** The thesis needs a new contribution. |

**KIP must clear 0.5868 by a defensible margin or there is no contribution left.** This outranks the mechanism question: a method with an unknown mechanism but a real composable gain is publishable; a method with a known mechanism that a pretraining swap replicates is not.

### 10.3 Ablation ladder — reordered (CHANGE B/C at the top)

Diagnostics that cost no training come first, because they can change what everything below should measure.

| # | Run | Cost | Decides |
|---:|---|---|---|
| **1** | ~~Log `s_t`, `‖ê_O‖`, `μ_t`, gate α at eval~~ **— ANSWERED 2026-08-30 without needing checkpoints.** The gate never trains, so init *is* the function, and its input domain is bounded by construction; sweeping the domain settles it analytically. Logging shipped anyway (`--dump-kip-diag`, on by default in `evaluate.py`/`inference.py`) for the correlation-with-anomaly-window half. | **0 training** | **H4′ — CONFIRMED: constant smoother.** Limitation 1 closed. |
| **2** | Eval-time `use_gate_shift=false` on existing KIP-on checkpoints | **0 training** | Gate/shift vs KIP-trained trunk. Tests conclusion 2 directly. |
| **3** | **S1→S2→S3 (§12.6's run, 3 seeds)** | 3 runs | **The composition question (§10.2).** Closes limitation 14. |
| **4** | **Plain-TSM control** — `kip.gate_type=constant`, `kip.const_shift_ratio ∈ {0.1…0.9}`, `kip.disable_pmg=true` for the no-flow arm | 3–5 runs | **NOW THE TOP EXPERIMENT.** H4′ holds (row 1), so this *is* the correct baseline, and v1's gate sat at `r ≈ 0.5`. **Pre-registered prediction: `r = 0.5` reproduces most of the +0.0915.** If it does, the gain is temporal smoothing, not motion. One config flag; implemented 2026-08-30. |
| 5 | STE on the floor (forward bit-identical) | 3 runs | Bug-vs-feature reading of the frozen gate. |
| 6 | `use_lkin=false` / `pmg_only=true` / **shuffled-flow-target control** | 9–12 runs | Sub-module attribution; motion *content* vs capacity. |
| 7 | Rank gate vs frozen MLP; ECMR vs absolute norm | 6 runs | The mechanism hypothesis, with the pre-registered ego/other prediction. |
| 8 | Respecified `L_KIP_align` → keep or delete by the stated rule | 3 runs | Repair or retire. |
| 9 | Flow-bearing corpus expansion (+UCF-Crime) and DADA-2000 / TAD as second benchmarks | extraction + runs | Separates "motion" from "DoTA"; lifts KIP off 480 videos. |
| 10 | Alert-CLIP swap; Stage-0 DAPT on/off; longer stage-1 | 9 runs | Never-tested free upgrades. |

### 10.4 Metric suite

Frame-level AUC; AP; **AUC_A/AnoAUC**; **MCC / AUCMCC / MCC@0.5** (now meaningful on PreVAD-trunk arms, where saturation is 38–48 %); mAP@IoU 0.1–0.5; cross-dataset and drift@5; **raw-pooled and per-clip min-max reported side by side** (the gap *is* the calibration measurement); MSAD traffic-slice breakdown; efficiency (params/GFLOPs/FPS, reasoning latency separate); seed-level t-intervals as headline; in-domain nulls stated as equivalence bounds.

---

## 11. Justification ledger — v3 rows

*v1 and v2 rows carry forward unchanged; see `KAT-VAD_spec_v2.md` §11. New and revised rows only:*

| Decision | Rationale | Source |
|---|---|---|
| **Gate rebuilt as a parameter-free deterministic rank map** | The gate MLP never receives gradient (`floor` is non-differentiable on a slice index), so it is frozen at random init; v2's "nested, cannot destroy the +0.09" safety argument is void. A rank map is trainable-free, seed-independent, scale-free, loggable, and makes the score path 0-parameter | **report §12.5**; DSANet SG-NM; RefineVAD Eq. 3 |
| Rank map structurally rules out H4′ | A rank map spans [0,1] on every clip, so a constant smoother is not expressible | audit addendum §3.3 |
| STE variant kept as an ablation, not a default | Forward bit-identical ⇒ pure gradient intervention; but unblocking lets `L_MIL` reach the PMG head, risking departure from the flow manifold that Pi-VAD's unweighted `L_PMG` exists to prevent | Pi-VAD; report §12.5 |
| **`s_t` logging is experiment #1** | A frozen random `σ(MLP(scalar))` plausibly saturates ⇒ `s_t` ≈ constant per domain, which alone predicts the DoTA gain, the MSAD null and the crop sensitivity. Costs zero training | report §12.5, §11.4, §8.2 |
| **Plain-TSM control added** | If H4′ holds, a fixed-ratio temporal shift with no flow is the correct baseline, and it has never been run | H4′ |
| **PreVAD = trunk source only, permanently** | Ships CLIP features, no pixels; RAFT impossible; ~3,800 dead rows are the traffic captures; re-downloads unpairable with the released features | **report §12.7**, `PREVAD_SETUP.md` §7.4 |
| **Recipe = PreVAD trunk → KIP stage-1 on flow-bearing data → joint stage-2** | Only configuration where broad-domain pretraining and a trained KIP coexist; it is also §12.6's unblocking run | report §12.3, §12.6 |
| **Flow-bearing corpus expanded (UCF-Crime first)** | KIP is still trained on 480 videos and PreVAD cannot fix it; MSAD+UCF = 4.4×, +DoTA-train+XD = 19×, all ship pixels | audit §5.1; report §12.7 |
| **Composition promoted to a primary result** | Broad-domain pretraining is an independent lever on the same quantity (+0.038 DoTA, directional). KIP must clear 0.5868 from the PreVAD trunk or the contribution collapses | **report conclusion 8**; audit addendum §5 |
| **Row-gated `L_KIP_rec`; never `require_flow=False`** | Zero-filling trains the PMG head to predict zeros — a wrong run that still yields a checkpoint | report §12.7 |
| **Precondition assertions + run manifest** | 3 campaigns, 3 multi-variable changes, 2 void signed numbers; limitation 11 upgraded to High after costing a third trunk reconstruction | report §12.4, limitation 11 |
| **float64 metrics before the ours-vs-released claim** | Float32 tie-manufacturing is a *function of saturation*; tie counts differ 270/93/1, so that comparison crosses a saturation boundary and the paired-bias defence does not apply | report limitation 15 + §11.3 |
| **"+0.038 DoTA trunk gain" reported as directional, not established** | Seed-level t95 [−0.0065, +0.0817] includes zero; only the MSAD AP gain [+0.0318,+0.0630] is significant at n=3. Sign-stability ≠ significance — §7.4 already burned the project on that inference | audit addendum §4.2 |
| **ATS unblocked on PreVAD-trunk arms only** | Saturation 86–89 % → 38–48 % with a real low tail; inverse-CDF sampling over a saturated curve degenerates to uniform | report limitation 8 |
| Check RefineVAD's MoTAR implementation for the same `floor` | Either it uses an STE/soft relaxation, or the dead-gate condition exists in an AAAI 2026 method | RefineVAD Eq. 2 |

---

## 12. Open iteration hooks

- **Mixed-corpus training with row-gated `L_KIP_rec`** — the scaling story: *motion induction extends to feature-only corpora because flow supervision is only needed on a subset*. Blocked on solving PMG staleness when `v^t` drifts during feature-only batches.
- **DSANet-faithful DNP bank** replacing the EMA prototype (K=8 learnable queries over the lowest-scoring positions, with `L_compact`), making Gap F explicit.
- **Dual PMG placement** — Pi-VAD ablates early 87.14 / late 87.48 / **both 90.33**. KIP is spliced once. Run only after attribution.
- **Depth as a second induction head** (DepthAnythingV2) — Pi-VAD shows depth dominates AUC_A.
- **Score-fusion variant** (`ŷ_O` gates the final score) and the **traffic hard-negative text bank** — both unblocked once the normal-driving negative pool exists.
- **DoTA-supervised arm** — DoTA-train ships pixels, so a within-domain (not just zero-shot) KIP result is now reachable.
- **Pixel-level localization** (LAVIDA decoder + Street Scene RBDC/TBDC); **anticipation pivot** (DAD/CCD/A3D, time-to-accident).
- **Swap the baseline to RefineVAD** — still the documented fallback; the novel component would flip to the normality branch.

*Say which knob to turn and I will re-run only the affected sections and update the ledger.*
