# Audit Addendum — the PreVAD campaign (`REPORT_KIP_MSAD_DOTA_PREVAD.md`, rev. 2026-08-28)

*Companion to `KAT-VAD_experiment_audit.md`. §§1–11 of the report are unchanged and my earlier adjudication of them stands verbatim. This addendum covers §12 (new), §§13–16 (revised), and one technical fact inside §12.5 that outweighs everything else in the revision.*

---

## 1. Verdict on the PreVAD campaign

**Correctly executed and, more importantly, correctly reported.** Declaring the KIP A/B **blocked** rather than publishing the signed −0.0024 is the right call, and the loss table proves the precondition was open rather than merely suspected:

| run | `kip_rec` final | `kip_align` final | chance floor |
|---|---:|---:|---:|
| §9 cold stage-2 KIP-on | **4.91** | 3.70 | 4.265 |
| PreVAD-trunk stage-2 KIP-on | **10.2 / 13.3 / 10.9** | 4.57–4.61 | 4.265 |

`L_KIP_rec` ends where the *stage-1* warm-up ended in the earlier campaign, and `L_KIP_align` sits **above** chance — it never left initialisation. The arm labelled "KIP-on" carries an untrained KIP. Reporting that as a KIP result would have been the single most damaging error available, and the report refuses it. Same discipline as the crop episode, applied faster.

**Gate P0 is a genuine strengthening.** The port now reproduces the released checkpoint on a *third* benchmark (PreVAD test, Δ AUC −0.0025, CI [−0.0103, +0.0047]), and `--score-norm auto` resolved correctly on first contact with a 49.9 %-normal dataset with no human decision. That is the pooling rule generalising, which is exactly the test it needed.

---

## 2. What §12 resolves from my earlier audit — and what it does not

| My earlier finding | Status now |
|---|---|
| **§5.1 — trained at 1.5 % of design-point scale; rerun on PreVAD** | **Half-resolved, half-made-permanent.** The *trunk* is now broad-domain and it matters (+0.047 AP in-domain). But §12.7 establishes that a PreVAD KIP arm is **permanently impossible** — PreVAD ships CLIP features only, RAFT needs pixels, ~3,800 dead source rows are the China Expressway Camera captures, and re-downloads cannot be paired with the released features. **The KIP module itself is still trained on 480 videos, and PreVAD can never fix that.** This requires a design change, not another run (§5 below). |
| **§4.3 — score saturation disables `L_kin`'s anchor** | **Relieved on the new arms** (86–89 % → 38–48 % of DoTA frames > 0.99, with a real low tail). The scale-free respecification is still correct — it is robust either way — but the ATS blocker I put on the LLM layer is **lifted for PreVAD-trunk arms**, and threshold metrics (MCC/AUCMCC) become meaningful there for the first time. |
| **§4.2 — "the module in the forward pass" was never tested** | **First evidence exists, and it is confounded.** §12.5's untrained-PMG arm has the gate and shift fully active and delivers −0.002 ± 0.016 on DoTA. The report's own caveat is right: the trunk moved simultaneously, and an untrained `ê_O` produces a *different* smoother, not the same one. It constrains the strong form of H4 without settling it. |
| **§5.2 — FP suppression unmeasurable on a 99.8 %-abnormal set** | **Unchanged.** PreVAD test is 49.9 % normal, but it is a *pretraining* corpus, not a traffic benchmark. The normal-driving negative pool is still required. |
| **§5.3 — dtype/precision not audited** | **Superseded by the report's own limitation 15**, which found it independently. See §4 below — I disagree with one line of its impact analysis. |

---

## 3. The finding that matters most: **the gate MLP receives no gradient**

Buried in §12.5 as a premise for H4:

> the gate MLP's 321 parameters never receive gradient (`(ratio * max_shift).floor().long()` is non-differentiable)

This is the most consequential sentence in either report, and it deserves to be a numbered finding rather than a parenthetical.

### 3.1 Trace the consequence all the way through

```
v^t ──► pmg ──► ê_O ──► ‖·‖ ──► MLP(321p) ──► r ──► floor(r·D/K) ──► s ──► integer slice of v^t ──► v^k
                 │                              ▲                    ▲
                 │                        no gradient          non-differentiable
                 ├──► mhead ──► ŷ_O ──► L_kin          [trains pmg]
                 ├──► proj_flow ──► L_align            [trains pmg]
                 └──► L_KIP_rec                        [trains pmg]
```

`s` is used as an **index**, not a value. Gradient flows to the tensor being sliced (`v^t`) but never to the index. Therefore:

1. The gate MLP is **frozen at random initialisation** for the entire run.
2. **`pmg` receives no task gradient either.** `ê_O` reaches the score path *only* through `s`. So the flow head is trained purely by `L_KIP_rec` + `L_KIP_align` + `L_kin` — the MIL objective never touches it.
3. **KIP's entire inference-time contribution to the anomaly score is: a reconstruction-supervised flow-magnitude estimator, piped through a frozen random scalar function, floored to an integer, used to slice channels.** No part of that path was ever optimised for the detection task.

That reframes the efficiency claim too. The "321 parameters on the score path" from my earlier audit are not 321 *learned* parameters — they are 321 *random constants*.

### 3.2 Two readings, both live, both cheap to test

**Reading A — it is a bug.** My v1 spec wrote `r_t = σ(MLP(m_t))` intending the mapping from motion intensity to shift ratio to be *learned*, as RefineVAD's MoTAR intends. It never was. Fix: a straight-through estimator (`s = r·D/K + (⌊r·D/K⌋ − r·D/K).detach()`), which leaves the forward pass **bit-identical** and only unblocks the backward pass — an unusually clean controlled experiment.

**Reading B — it is why KIP works.** With the gradient blocked, `ê_O` is a *pure* motion estimator that the MIL loss cannot co-opt into an appearance shortcut. Pi-VAD deliberately leaves `L_PMG` unweighted for exactly this reason — pseudo-modalities must stay bounded to the real manifold. Unblocking the gradient may pull `ê_O` off the flow manifold and destroy the effect.

**Both readings predict opposite outcomes from the same single run.** That makes it a good experiment. It should not be assumed either way.

*Worth checking upstream:* RefineVAD's MoTAR Eq. 2 is also `s_t = ⌊r_t · D/K⌋`. Either the authors use an STE/soft relaxation not described in the summary, or the same dead-MLP condition exists in the source paper. Reading their implementation costs ten minutes and either validates the design or turns into a publishable observation about a AAAI 2026 method.

### 3.3 A new hypothesis this creates — **H4′, the domain-conditioned constant smoother**

A randomly-initialised MLP on a scalar input, with small init weights, produces `σ(≈0) ≈ 0.5` → `s ≈ ⌊0.5 · 512/4⌋ = 64` channels, roughly *constant*. If `‖ê_O‖` is large and above the MLP's dynamic range on dashcam footage, `σ` saturates and `s ≈ 128` (the maximum) nearly everywhere.

Under that condition, KIP reduces to:

> **apply heavy uniform bidirectional temporal channel-shifting on ego-centric video, and almost none on fixed-camera video** — because the scene moves a lot in one domain and barely at all in the other.

This single mechanism predicts, with nothing else added:

| Observation | H4′ prediction |
|---|---|
| +0.09 on DoTA | ✓ dashcam → high flow norm → large shift → strong temporal mixing that KIP-off lacks entirely |
| Null on MSAD | ✓ fixed camera → low flow norm → near-zero shift → KIP ≈ KIP-off |
| Crop removal worth +0.15 to KIP-on only | ✓ restoring peripheral flow raises `‖ê_O‖` → changes the shift schedule; no other arm has a shift |
| `other` > `ego` (6/6) | ✓ third-party events are shorter and sharper, so ±1-neighbour mixing helps them more |
| §12.5 untrained-PMG arm gains nothing | ✓ random `ê_O` has a different magnitude distribution → different (uninformative) schedule |
| KIP-on gains as it fits harder (probe) | ~ partially — the trunk improves while the schedule stabilises |

If H4′ is true, the contribution is not "a motion pathway" but "**a domain-adaptive temporal smoothing schedule derived from induced motion magnitude**" — still a real and defensible result, but a materially different thesis, and one whose baseline comparison must include *a plain TSM with a tuned constant shift ratio*, which nobody has run.

### 3.4 The one measurement that settles it — and it is nearly free

**Log `s_t` at evaluation time.** No training, no new arms. Three statistics per dataset:

1. **Distribution of `s_t`** on DoTA vs MSAD. If DoTA sits pinned near 128 and MSAD near 0, H4′ is essentially confirmed.
2. **Within-clip variance of `s_t`**. Near zero ⇒ constant smoother. Substantial ⇒ genuinely adaptive.
3. **Correlation of `s_t` with the ground-truth anomaly window**, per clip. This is the difference between "a smoother" and "a detector."

This also closes limitation 1 ("no eval-time diagnostics of ŷ_O, gate α or shift magnitude have been saved") with the highest information-per-GPU-second measurement available anywhere in the project — and it should run **before** the §12.6 unblocking run, because if `s_t` is constant the whole gate design changes and §12.6 would be measuring a superseded module.

**Companion sanity check, one step of training:** log `‖∂L/∂θ_gate‖`. If it is exactly zero, §12.5's code reading is confirmed empirically rather than by inspection.

---

## 4. Statistical and numerical corrections

### 4.1 The dtype argument does not hold for the ours-vs-released comparison

Limitation 15 says float32 min-max normalisation manufactures ties among saturated frames, worth up to **ΔAP 0.0039**, and concludes: *"All deltas are paired, so both arms carry the same bias."*

That is valid for KIP-on vs KIP-off (both saturated) and for trunk-vs-trunk. **It is not valid for ours vs the released checkpoint,** because §11.3 measures the tie counts as **270 / 93 / 1** for KIP-off / KIP-on / `best.ckpt`. The bias is a *function of saturation*, and these arms differ in saturation by two orders of magnitude. A comparison that crosses a saturation boundary does not carry a shared bias.

Concretely, this touches §12.3's headline: **PreVAD-trunk KIP-off beats the released checkpoint by AUC +0.0039**, seed-level t95 [+0.0019, +0.0058]. The AUC-side dtype effect measured in limitation 15 is ≈0.00045 — smaller than +0.0039, and ties *depress* the more-saturated arm, so the direction favours the claim. But the interval's lower bound is +0.0019, only ~4× the known artifact. **Rescore in float64 before making the "above the released checkpoint" claim.** It is one command and the report already identifies the fix.

### 4.2 "+0.038 DoTA AUC" is sign-stable but not significant at n = 3

§12.3 correctly warns that seed-level intervals are wide. Conclusion 8 then states the number as established fact. Recomputing:

| Trunk gain (KIP-off) | per-seed | mean ± sd | **seed-level t95** | |
|---|---|---|---|---|
| MSAD **AP** | +0.0404 / +0.0525 / +0.0493 | +0.0474 ± 0.0063 | **[+0.0318, +0.0630]** | **excludes 0** |
| MSAD AUC | +0.0057 / +0.0138 / +0.0166 | +0.0120 ± 0.0057 | [−0.0020, +0.0261] | includes 0 |
| **DoTA AUC** | +0.0257 / +0.0291 / +0.0580 | +0.0376 ± 0.0177 | **[−0.0065, +0.0817]** | **includes 0** |

So: the **in-domain AP gain is established**; the **zero-shot DoTA gain is directional only**. Conclusion 8 and the executive summary should carry that distinction, because the DoTA number is the one that competes with KIP's headline. Sign-stability across 3 seeds is encouraging evidence, not a significance claim — and this project has already been bitten once by exactly that inference (§7.4, where every bootstrap CI excluded zero while the sign flipped).

"Twelve clip-level CIs exclude zero" carries the same independence caveat as the earlier "18 CIs": 3 seeds × metrics × arms, sharing score curves and reference arms.

---

## 5. The question §12 opens, which is now the most important one in the project

Conclusion 8 identifies it exactly: broad-domain pretraining is **an independent lever on the same quantity KIP was built to move.** Quantified:

| DoTA micro AUC (3-seed means) | value |
|---|---:|
| MSAD-trunk, KIP-off | 0.5492 |
| MSAD-trunk, KIP-**on** | **0.6408** (KIP delta +0.0916) |
| PreVAD-trunk, KIP-off (measured) | **0.5868** (trunk delta +0.0376) |
| PreVAD-trunk, KIP-on — *if the levers are additive* | 0.6784 |
| PreVAD-trunk, KIP-on — *if fully redundant* | 0.6408 |

**KIP must clear 0.5868 from the PreVAD trunk, by a margin worth defending, or the contribution collapses into "an expensive substitute for more pretraining data."** That single number is the thesis's viability test, and it outranks the mechanism question: a method with an unknown mechanism but a real, composable gain is publishable; a method with a known mechanism that a cheap pretraining swap replicates is not.

The report is right that §12.6's run closes it. My only amendment is ordering: **run the `s_t` logging first** (§3.4), because if the gate is a constant smoother, §12.6 measures a module that is about to be redesigned.

---

## 6. A process defect worth naming, because it has now cost three campaigns

Three campaigns, three multi-variable changes, two uninterpretable signed numbers:

| Campaign | Variables changed at once | Cost |
|---|---|---|
| Center-crop → `_ncc` | transform + (implicitly) flow-target parity | −0.0324 recorded under an open precondition; sign later flipped |
| PreVAD trunk | trunk source **+** removal of KIP stage-1 **+** step budget 500→160 | 6 runs, zero KIP verdict |
| (limitation 11, all three) | `--init-weights` unrecorded | trunk provenance reconstructed from step-1 loss, three times |

The report upgrades limitation 11 to **High** and estimates ~30 LOC. I would go further and add **automated preconditions that fail the run**, not just a manifest:

```
assert stage1_final(L_KIP_rec) < τ_rec            # "did KIP actually train?"  → catches §12.4 at step 1
assert flow_cache_covers(train_ids)               # never zero-fill e_O (§12.7's trap)
assert hash(feature_transform) == hash(flow_transform)   # the induction-parity rule, §11.4
assert grad_norm(θ_gate) > 0  (first step, if STE enabled)
emit run_manifest.json {init_weights, cache_hashes, transform, trunk_source, step_budget}
```

The first assertion alone would have converted a six-run void campaign into a one-minute failure message. This is cheaper than any experiment in the queue and prevents the failure mode that has already recurred twice.

---

## 7. Revised experiment queue

Reordered against my earlier audit, with the new information folded in. Items 1–2 are diagnostics that cost no training and can change what the later items should even measure.

| # | Experiment | Cost | Decides |
|---:|---|---|---|
| **1** | **Log `s_t`, `‖ê_O‖`, gate α at eval** on existing checkpoints; distribution, within-clip variance, correlation with the anomaly window | **0 training** | **H4′.** Is KIP an adaptive gate or a domain-conditioned constant smoother? Closes limitation 1. Reorders everything below. |
| **2** | Eval-time `use_gate_shift=false` on existing KIP-on checkpoints | **0 training** | Is the gain the shift, or the KIP-trained trunk? Tests conclusion 2 directly. |
| **3** | **§12.6's unblocking run** — PreVAD trunk + KIP stage-1 on MSAD + 500 steps, 3 seeds | 3 runs | **The composition question (§5).** Does +0.09 survive a broad-domain trunk? Closes limitation 14. |
| 4 | Plain-TSM control: fixed shift ratio, no PMG, no flow, swept over `r ∈ {0.1…0.9}` | 3–5 runs | If H4′ holds, this is the correct baseline and it has never been run. A tuned constant TSM matching +0.09 would be decisive. |
| 5 | STE on the floor (forward bit-identical, backward unblocked) | 3 runs | Reading A vs Reading B (§3.2). |
| 6 | `use_lkin=false` / `pmg_only=true` / shuffled-flow-target control | 9–12 runs | Per-sub-module attribution; motion *content* vs capacity. |
| 7 | Respecified `L_KIP_align`; ECMR gate (after 1 & 5 settle the gate design) | 6 runs | Repair the two broken losses; test the mechanism hypothesis. |
| 8 | Expand the flow-bearing corpus (UCF-Crime + DoTA-train + BDD100K-normal) and add DADA-2000 / TAD as a second transfer benchmark | extraction + runs | Separates "motion" from "DoTA"; lifts KIP's own training corpus off 480 videos — the part PreVAD can never fix. |
| 9 | `rescore --write` in float64 everywhere; then restate the ours-vs-released claim | 1 command | Removes the one comparison that crosses a saturation boundary. |

---

## 8. Bottom line for this revision

The PreVAD campaign did what it could do and refused to claim what it couldn't. Three things follow:

1. **PreVAD is a trunk source and a Gate-P0 benchmark. It is not, and can never be, a KIP training corpus** (§12.7 is correct and should be treated as settled). The spec must stop planning around it.
2. **The KIP module's own training corpus is still 480 videos**, and the fix is expanding the *flow-bearing* corpus (UCF-Crime, DoTA-train, BDD100K-normal, A3D/CCD/DAD — all ship pixels), not PreVAD.
3. **The gate does not learn.** Until `s_t` is logged, nobody knows whether KIP is an adaptive kinematic gate or a domain-conditioned constant smoother — and that distinction determines both the correct baseline and the correct thesis claim.
