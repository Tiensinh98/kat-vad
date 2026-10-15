# Proposed Road-Traffic Video Anomaly Detection Architecture — **KAT-VAD v2.2** (rev. 2026-10-13c, review rounds 6–7)
### Two tracks: a **score track** (DF8 position read-out, H4 V2-B) and a **motion-claim track** (**EMS-v2.1**, ego-compensated epipolar motion evidence, gated by cheap kill-switches)

**Supersedes:** the v2.2 of 2026-10-13 (EMS-v2, round 5). `KAT_VAD_PROPOSAL_v2.md` remains the record of v2; `claude/KAT_VAD_PROPOSAL_v2.1.md` (SG-NM) is closed (E4a).

**Evidence base:**
- the **S1–S3 read-out of 2026-10-13** (`outputs/v2/REPORTS/v22_order/s{1,2,3}_readout.{md,json}`), quoted as "S1a" … "S3(b)", caveats "C-a" … "C-g";
- the v2 results (`REPORT_V2_RESULTS.md`), quoted as "results §x";
- the **round-6 review** (2026-10-13), quoted as "A1" … "A6", "B1" … "B3", "D1" … "D4", questions "Q1" … "Q14";
- project paper summaries: LAS-VAD (CVPR), Pi-VAD, SimpleTAD (CVPR 2025), LaGoVAD (ICLR 2026), RefineVAD and DSANet (AAAI 2026); and, for the degeneracy test only, Chum, Werner & Matas, *Two-view geometry estimation unaffected by a dominant plane* (CVPR 2005, "DEGENSAC").
- Numbers marked *derived* are computed from those, with the arithmetic shown. Numbers marked *judgement* are priors, not measurements.

**Companion:** `KAT_VAD_v2.2_ARCHITECTURE.md` (descriptor, controls, gates, arms, use tests, tensor shapes).
**Date:** 2026-10-13.

> **Design rule (v2, extended).** A component enters only if (i) it targets a **measured** failure, (ii) a project paper supports the mechanism, (iii) it is checked cheaply **before** training, (iv) it is shown to carry **information the existing inputs do not**, (v) the claim it licenses is decided on an **estimand where that mechanism is expected to act**, fixed before any outcome is seen, and — new in round 6 — (vi) its **goal is stated** (score or claim), and **adoption is decided on the benchmark metric only**.

---

## R. Review history

| Round | Main points | Outcome |
|---|---|---|
| 1–3 (2026-10-11) | splice-safe features; block shuffle; adjacency-free permutations; one bar; cluster bootstrap; advisor check | unchanged |
| 4 (2026-10-12) | S3 fails → TDP rejected; EMS-v1 introduced | superseded |
| 5 (2026-10-13) | root cause RC1/RC2/RC3; EMS-v1 audit (parallax) | EMS-v2; M0-R; ΔT; U0–U2; R0 |
| **6 (2026-10-13, this version)** | EMS-v2 serves a **claim, not the score** (prior span gain ≈ 0 to +0.015); nothing targets the open position gap. **A1** channel `e` switched meaning at `η_H`, and the homography fallback could re-import parallax; **A2** U0 (`ψ̃ := 0`) is not an exact ablation; **A3** CLIP *does* carry flow magnitude (R² 0.357); **A4** anisotropic squash differs by corpus; **A5** DADA 30 fps unverified; **A6** DoTA-CAP mixes 10 and 30 fps. **B1** two adoption routes; **B2** ΔT forking paths, no futility rule; **B3** ΔT rewards any onset spike. **D** cheapest-first order with two free kill-switches | **Two tracks** (§A, §8). **EMS-v2.1** (§4): `e^F` and `|r|` as separate, fixed-meaning channels with a validity bit and an off-plane-support (DEGENSAC-style) test instead of `η_H`; isotropic units; measured fps and pair interval in seconds; per-fps-group reporting. **U0** = within-clip circular shift. **Adoption on span only.** Forking-paths statement and **futility rule** for ΔT. STRUCTURE is the identifying contrast. New steps **R0-f** (DoTA-train check), **K1** (free global-flow probe), **K2** (quality-only pilot) |

---

## R6. Response to round 6, point by point

| Point | Accepted? | Change | Where |
|---|---|---|---|
| §0 Goal / prior | **Yes** | EMS-v2.1 is re-labelled the **motion-claim track**. Its expected span gain is stated as ≈ 0 to +0.015 (*judgement*, agreeing with the review). The **score track** (DF8 read-out, H4 V2-B) is added and runs first | §A, §0.1, §8 |
| §0.1 position untouched | **Yes** | Stated. No component in this revision targets position; DF8 decides how the gap is *told* (benchmark artefact vs model failure). Feeding `p_T2` to the model stays declined (results §8) | §3.4 |
| §0.2 probe headroom | **Yes** | Already in §2.3; now also in §0.1 as a reason the span bar is unlikely to pass | §0.1 |
| §0.3 change already exploited | **Yes** | Non-ego arithmetic added: +0.03 on non-ego → +0.013 overall (*derived*: 0.03 × 248/569) | §0.1 |
| §0.4 KIP precedent | **Yes** | Recorded as neither for nor against (flow was a target there, not an input) | §2.2 |
| **A1** `e` switches meaning | **Yes** | `e^F` (Sampson, epipolar) and `|r|` (homography residual) are **always separate channels**; `v_t` (F usable) and `π^H` are channels; `η_H` is **removed**. F usability = ≥ 15 F-inliers **and ≥ 8 F-inliers that are homography outliers** (off-plane support, DEGENSAC-style), so a road plane filling the lower half no longer forces a fallback when the rest of the scene has depth. When F is unusable, `e^F` **holds its last usable value** and its acceleration is **gated** (`v_t v_{t−1}`), so no switch becomes an event. Pilot **gates** the F-usable share (≥ 50 % of steps) and a **speed-coupling test** (`e^F` must be less coupled to ego speed than `|r|`) | §4.1, §4.3, §8 (K2) |
| **A2** U0 not exact | **Yes** | "Exact" dropped. **Primary U0 = within-clip circular shift** of `ψ̃` by ⌈T/3⌉…⌊2T/3⌋ steps (preserves the per-clip marginal, median and autocorrelation; breaks onset alignment by ≥ ≈ 3.3 s on a median DoTA clip). Within-clip permutation and `ψ̃ := 0` are reported (the latter labelled OOD-confounded, as K0). U2 uses the same shift per half | §6.2 |
| **A3** RC1 overstated | **Yes** | RC1 reworded: CLIP carries some flow **magnitude** (R² 0.357), little **direction** (0.071). M0-R reports R² by **direction-type** and **magnitude-type** channels; pre-registered expectation: the **ego block fails** M0-R (is reconstructable); the gate stays on the **local block** | §2.2, §4.4, §6.3 |
| **A4** anisotropic squash | **Yes** | Aspect-preserving resize to width 448 with padding (padding masked); all displacements in **frame-width units** for x and y (isotropic); grid defined in frame fractions; divergence uses the true cell aspect | §4.1 |
| **A5** DADA fps | **Yes** (user: *not verified*) | fps **measured** at K2/M0 (container metadata and frame count ÷ duration, both printed). Pair offset `k = round(0.1 s · fps)`; every displacement channel divided by the **actual** `Δt_f = k / fps` (units **per second**); `Δt_step` and ΔT windows use measured time. Gate: the two fps estimates must agree within 2 %, else the clip is flagged and excluded from M0 statistics | §4.1, §8 |
| **A6** DoTA-CAP mixed fps | **Yes** | K2 and M0 quality metrics, ΔT and span reported **per native-fps group** (10 / 30); F-usable gate applied per group | §4.3, §7 |
| **B1** two adoption routes | **Yes** | **Adopt on span only.** ΔT decides the motion claim, which is now **decoupled** from adoption (a motion claim on a non-adopted arm is allowed and worded as such) | §8, §11 |
| **B2** forking paths | **Yes** | Explicit statement in §5.3. **Futility rule, fixed before R0:** if ΔT(A3-ign) ≥ 0.90 on CAP-dev, or ΔT of the saved `F0` probe ≥ 0.90, ΔT is demoted to "reported" | §5.3 |
| **B3** ΔT rewards any spike | **Yes** | Stated: VALUE on ΔT licenses only "onset signal"; **STRUCTURE (ψ − δ) is the identifying contrast** for motion, at M1 and M2 | §5.4, §8 |
| **C** Q3 answered | — | C-b stands as written: the shuffled probe was **trained** on shuffled data (`core/tools/order_diagnostics.py:412-413`), so S2's Δ is not test-time distribution shift | §1 |
| **C** Q1 answered | — | No unscored subset inside DoTA-CAP. **R0-f** checks whether CAP's source contains DoTA-**train** clips; rule fixed now | §2.7, §7.2 |
| **D** order | **Yes** | R0 (+ futility) → **K1** global-flow probe on T2-val → **K2** quality-only pilot → full M0 → M0-R → M1 → advisor → M2 | §8 |

### Answers to the review's questions

| Q | Answer |
|---|---|
| **1** Goal | Two tracks. **Score:** DF8 (H1) read-out and H4 (A3-B) — they target the measured gaps (position story; B > S on transfer). **Claim:** EMS-v2.1, whose purpose is a defensible "motion-aware" claim; it is not expected to move span much and is not preferred over H4 for the score |
| **2** Expected gain | Span: ≈ 0 to +0.015 (*judgement*). With the gain concentrated on non-ego clips, +0.01 overall needs ≈ +0.023 on non-ego (*derived*: 0.01 × 569/248); +0.03 on non-ego gives +0.013 overall. The span bar is therefore **more likely to fail than pass**; ΔT is where a pass is plausible |
| **3** Homography-explained share | Unknown; plausibly high on dashcams (road plane + far scene). That is why `η_H` is gone: F usability depends on **off-plane support**, not on the homography's inlier ratio. If the share of steps with too little off-plane support is > 50 %, K2 fails and the design changes before extraction. `|r|` stays a separate channel with its parallax; K2's **speed-coupling test** checks that `e^F` removes most of it |
| **4** Switching channel | It no longer switches: fixed-meaning channels, validity as a channel, hold-last for `e^F` when unusable, gated acceleration |
| **5** M0-R expectations | Expected R² order: ego block > global magnitude > cell magnitude > direction. Pre-registered: **ego block fails** (R² > 0.5). The **gate applies to the local block** (all cell channels incl. `e^F`, `|r|`, `ℓ`, `λ`, `ν`, their accelerations and summaries), not to direction only: restricting it to direction would exclude `e^F` and `ℓ`, the claimed carriers. Direction and magnitude sub-blocks are reported |
| **6** λ, ν across corpora | Isotropic frame-width units, aspect-preserving resize, frame-fraction grid (A4) |
| **7** DADA fps | Not verified. Measured at K2; pairing is defined in seconds; at 25 fps, `k = 3` gives `Δt_f` = 0.12 s and `k = 2` gives 0.08 s (*derived*), and all displacements are divided by the actual `Δt_f`. Already-extracted VideoMAE windows are not re-extracted; if fps ≠ 30, their "1.5 s / 10 fps" descriptions are corrected in the text |
| **8** DoTA-CAP fps groups | Expected to differ (temporal resampling and compression). Reported per group throughout; the F-usable gate is per group |
| **9** Forking paths; futility | §5.3: stated openly; futility rule accepted and fixed now |
| **10** Adoption without span | No. Span only |
| **11** What decides "motion" | STRUCTURE (ψ − δ); VALUE alone licenses "onset signal" |
| **12** U0 | Not exact; primary U0 is now a within-clip circular shift |
| **13** DoTA-train clips | Unknown; R0-f checks and a rule decides (§7.2) |
| **14** Kill-switches | Accepted: K1 and K2 (§8) |

---

## R7. Round 7 — decisions taken (2026-10-13c; user: "decide what you can and run")

**These rows override any conflicting text below and in the architecture.** Not advisor-reviewed (user-authorized, as since 2026-09-29).

| # | Point | Decision | Overrides |
|---|---|---|---|
| **D1** | fps cannot be measured: DADA ships PNGs (no container, no duration); DoTA-CAP is image frames (`rebuild_folder` symlinks), its "10 / 30 fps" is the alignment rate `r`, and DoTA = 10 fps is itself the paper's | fps stays an **assumption**: **A1** DADA = 30 fps, **A1′** DoTA = 10 fps. Pairs are defined **in frames**: DADA `k = 3`; DoTA-CAP `r = 1` → `k = 1` (farm); `r = 3` → **`k = 3` on raw CAP frames** (D7); `Δt_f`, `Δt_step` and every "per second" are **conditional on A1/A1′** and labelled so. The "measured fps" gate is removed | §4.1 (fps lines), A5 row, arch §2.1 rows 1–2, K2 fps row |
| **D2** | R0-f cannot fire: DoTA test (1,402 clips) already spans **179 of DoTA's 184** YouTube sources (`metadata_val.json`, counted 2026-10-13), so a source-disjoint DoTA-train set has ≤ 5 sources (≈ 125 clips < 200); CAP carries no DoTA ids and DoTA-train has no features to match | **R0-f closed, no confirmatory set.** Any further condition is the DF8 swap, labelled "new condition on opened videos". §15 Q3 answered: no | §2.7 R0-f, §7.2 last row, §15 Q3 |
| **D3** | K2 speed coupling: `ρ(e^F)` on `v_t = 1` steps only vs `ρ(|r|)` on all steps → range restriction lowers `ρ(e^F)` mechanically | both ρ on the **same steps** (`v_t = 1`). Add a **falsification gate**: on near-stationary steps (`mean |f^H|` in the bottom 10 % of the pilot), F-usable share ≤ 10 %; else `n^P_min` is raised to the pilot's 99th percentile of `n^P` on those steps (fixed before any label is read) | §4.3, arch §4.2 |
| **D4** | STRUCTURE confounds structure with input width (224 vs 36) | δ is **width-matched**: same template as ψ — ego-like 12 (frame mean/p90 \|ΔI\|, mean ΔI, mean \|Δ∂_xI\|, mean \|Δ∂_yI\|, quadrant means ×4, ratio of changed pixels, 1 const-free spare = global p50) and per 4×4 cell 6 (mean/p90 \|ΔI\|, mean ΔI, mean \|Δ∂_xI\|, mean \|Δ∂_yI\|, p90 \|Δ∂I\|) + 2 globals, same `ā` and 5 summaries → **224**. Also reported (never deciding): `F0 + ψ̃_shift` (U0 shift applied to the probe input) | §6.1, arch §3, M1/M2 |
| **D5** | M1 probe vs M2 base mismatch if SC-2 adopts A3-B | **Motion-track base = A3-ign, fixed**, whatever SC-2 decides. EMS is not combined with V2-B in v2.2 | §3.4 last paragraph, §8 M2 |
| **D6** | K1: `flow/v1` pair `i` = frames `(i, i+1)` (`raft_extract.py:191`) → one-step look-ahead; interval 0.27 s | `stats23` **shifted one step later** (row t = pair `(t−1, t)`, row 0 replicates) before any use; read-out states the 0.27 s interval | §8 K1, arch §4.1 |
| **D7** | DoTA-CAP `r = 3` farm neighbours are 2–5 CAP frames apart (e.g. `0RJPQ_97dcs_000387`: steps 2, 4, 5 in `frame_map`) | for `r = 3` clips flow pairs are `(j_t − 3, j_t)` on **raw CAP frames**; the 3 dev clips at `r = 2` (and 2 overall) are a third group, reported, excluded from per-group gates. Dev groups: `r = 1` 395, `r = 3` 171, `r = 2` 3 | §4.1, arch §2.2 row 1, §7.1 per-group |
| **D8** | SC-1 wording needs the model's own position cost (DF8 plan R2), not only R1 | "… whereas the detector's score does" requires **R2**: both `control − early` and `control − late` for A3-ign have lower bound **> −0.01**. *(Sign corrected 2026-10-14, before any read, in v2 addendum §24 X7: the test is `early − control` and `late − control` lower bound > −0.01, i.e. moving the accident costs less than 0.01.)* DF8 choices D1–D6 taken as the plan recommends (dev only; re-extract V2-S heads; A3-ign primary, A3/A0 printed; K = 3, min context 10; 0.30 ↔ 0.70; R1 on the early/late mean) | §3.4, §11 SC-1 rows |
| **D9** | SC-2 re-opens E2(d)'s tie rule N7 | stated: **SC-2 re-opens N7**. N7 is a cost rule ("within 0.02 the cheaper encoder"), not evidence of equivalence; B was ahead on the beyond-position transfer read (+0.037 vs +0.018). Registered as a new arm on the same dev set | §3.4 |
| **D10** | K1's soft stop (Δ < +0.005) sits inside T2-val noise (S3 T2-val CI ±0.0105) | K1 null is **weak evidence**; it triggers the written advisor go-ahead, nothing stronger | §8 K1, §11 |
| **D11** | R0-c (κ regressor) only calibrates M0-R | **moved to the M0-R batch** (run together with the ψ regressor) | §2.7, §8 |
| **D12** | ΔT for saved native-frame scores vs step-level probes | ΔT is always computed **on the step grid**: saved protocol-B native scores are read at `native[::3]` (exactly the step scores: `to_native` interpolates through them); `Δt_step` = 0.30 s (A1′) on DoTA, 8/30 s (A1) on T2; T2-val read per **source** (as S2, O3) | §5.2, arch §5.1 |
| **D13** | Execution order | **Batch 1 (now):** R0-saved (local, saved scores) → R0-probe + K1 (Colab). **Batch 2:** SC-1 DF8 build. **Batch 3:** SC-2 A3-B (V2-B T2 extraction + 5 runs). **Batch 4:** EMS-v2.1 code + K2 pilot — **parked by D14**. M0 only after K2 | §8 order |
| **D14** | **R0-saved (run 2026-10-13, local, saved scores) falsifies ΔT's premise.** On DoTA-CAP-dev (557 eligible clips) **ΔT(`p_T2`) = 0.970** [0.958, 0.981] vs span 0.824 — the opposite of R0-a's prediction ("well below 0.822"). Positives always follow negatives in time, so any score rising through the onset wins ΔT. A pre-onset-detrended variant (G10) is no fix: `p_T2` 0.000 (a concave prior overshoots its own trend), A3 0.322, A3-ign 0.631 — it measures each score's pre-onset slope, not onset evidence. `f_2`-ΔT is 0.947–0.976 for every arm (the prior dominates) | **ΔT is retired as a decision estimand** (validity failure, measured before any EMS feature exists). The pre-registered fallback of §5.3 applies: the motion claim is decided on **span** (STRUCTURE / M − Mδ, `f_2` beside) plus the DF8 swap; ΔT and detrended ΔT are **reported** with `p_T2` beside. The futility rule is moot. Consequence for the motion track: its claim now rests on span, where the prior gain is ≈ 0 to +0.015 (§0.1) — the track is **parked after batch 1** (R0-probe, K1) unless K1 shows signal or the advisor asks for it; batches 2–3 (score track) go first | §5, §8 M1/M2, §11 ΔT rows, §12 |

---

## A. The answer in brief

**Two goals, two tracks, one dev set.**

| Track | Goal | Steps | Decides | Expected payoff |
|---|---|---|---|---|
| **Score** | the benchmark number and the position story | **SC-1** DF8 swap read-out (no training); **SC-2** H4: A3-B (V2-B extraction for T2 + 5 runs) | SC-1: how the `p_T2` > A3-ign gap is told. SC-2: adoption of A3-B on span | SC-2: ≈ 0 to +0.02 span (*judgement*; E2(d) beyond-position transfer B +0.037 vs S +0.018) |
| **Motion claim** | a defensible, pre-registered "motion-aware" claim (or a clean negative chain) | R0 → **K1** → **K2** → M0 → M0-R → M1 → advisor → M2 → U | the motion claim on ΔT (STRUCTURE); adoption only via span | span ≈ 0 to +0.015 (*judgement*); ΔT: open |

**Root cause of the motion null (unchanged, RC1 reworded).** Every earlier test (i) offered **no information the inputs lacked** — and the one thing CLIP clearly lacks is flow **direction** (KIP direction R² 0.071), while flow **magnitude** is partly present (0.357) — and (ii) was decided on an estimand where a transient onset cue is worth little.

| Idea | Status | Detail |
|---|---|---|
| TDP (IAM on the CLIP vector) | Rejected (S3) | §1, §2.2 |
| EMS-v1 | Superseded (parallax) | §2.6 |
| EMS-v2 (round 5) | **Revised → EMS-v2.1** (A1–A6) | §4 |
| **EMS-v2.1** | Candidate, motion-claim track, behind K1 and K2 | §4–§8 |
| **H4 / A3-B** | Candidate, score track | §3.4, §8 |
| **DF8 swap** | Read-out, score track | §3.4, §5.1 |
| Order claims (P1, S4) | Optional, analysis chapter | §8 |
| IAM prototypes / `L_cst`, ACC, local Transformer, GCN, attribute text | Declined | §10 |

---

## B. What changed (round 5 → round 6)

| Before (EMS-v2) | Now (EMS-v2.1 and plan) | Why |
|---|---|---|
| One plan, EMS-v2 the only next step | **Two tracks**; score track first | goal stated (Q1); review §0, D |
| Channel `e` = Sampson if `π^H < η_H`, else `|r|` | **`e^F` and `|r|` always separate**; `v_t`, `π^H` channels; `η_H` removed; off-plane support ≥ 8 for F usability; hold-last + gated `ā` | A1 |
| Homography-fallback share "reported" | **Gated** at K2: F-usable share ≥ 50 % per corpus and fps group; speed-coupling test | A1 |
| 448 × 224 squash; x ÷ W, y ÷ H | aspect-preserving resize + mask; **isotropic** width units; frame-fraction grid | A4 |
| "0.1 s = n − 3" on DADA | fps **measured**; `k = round(0.1·fps)`; displacements **per second** using actual `Δt_f` | A5 |
| Pooled DoTA-CAP | **per-fps-group** reports and gates | A6 |
| U0 = `ψ̃ := 0`, "exact" | U0 = **within-clip circular shift**; zeroing secondary | A2 |
| RC1: "flow is not in CLIP" | "flow **direction** is not in CLIP; magnitude partly is"; M0-R reports by type; ego block expected to fail | A3 |
| Adopt on span **or** (span non-inferior **and** ΔT) | **Adopt on span only**; motion claim decoupled | B1 |
| ΔT legitimacy paragraph | **Forking-paths statement + futility rule** | B2 |
| VALUE and STRUCTURE side by side | **STRUCTURE identifies motion**; VALUE = onset signal | B3 |
| Descriptor 185-d | **224-d** (ego 24, local 200) | A1 |
| — | **R0-f** (DoTA-train check), **K1** (global flow probe), **K2** (quality pilot) | C, D |

---

## 0. Design summary

**Baseline and deployed model (unchanged).** LaGoVAD (ICLR 2026) trunk; deployed **A3-ign** (CRN R2 + frozen CLIP ViT-B/16 + frozen VideoMAE V2-S through a zero-init residual; `dvs_anchor_mode = ignore`). A3 − A0 **+0.090 [+0.065, +0.115]** (CAP-eval, results §4.2); A3-ign − A3 **+0.0376 [+0.0326, +0.0426]** (second look, results §4.5). **`p_T2` 0.8256 > A3-ign 0.8120 on CAP-eval is the open position problem; no component in this revision targets it** (§3.4).

**Score track.** SC-1 runs the deferred DF8 plan (forward passes on the 491-clip early/late swap of CAP-dev). SC-2 runs H4: arm A3-B, trained as A3-ign with VideoMAE V2-B, 5 seeds, adopted on span only.

**Motion-claim track (EMS-v2.1, §4).** Per step, RAFT-small flow between the step's frame and the frame ≈ 0.1 s earlier (exact interval measured), aspect-preserved, on a frame-fraction grid, static mask applied, displacements in frame widths per second. From it: **ego-motion** (RANSAC homography as 4-point displacement, mean ego flow, inlier ratio, F-usable bit, off-plane support, ego jerk); **independent motion** `e^F` (Sampson distance to a RANSAC fundamental matrix, usable only with off-plane support); **homography residual** `|r|` (kept as its own channel); **looming excess** `ℓ`; **inward lateral** `λ`; **vertical** `ν`. 224-d descriptor, IAM acceleration, causal summaries, per-clip median reference, pointwise MLP, zero-init residual at `h`.

**Gates (motion track).** R0 (descriptive; ΔT eligibility and futility; DoTA-train check) → **K1** (free: does global flow add anything in-domain?) → **K2** (quality-only pilot; no labels read) → M0 (quality, corpus) → **M0-R** (non-redundancy of the local block) → **M1** (STRUCTURE on ΔT, with VALUE as prerequisite) → advisor → **M2** (base re-run, M, Mδ) → **U0–U2**.

**What a full pass licenses (§11):** *"Ego-motion-compensated, epipolar independent-motion features improve onset detection beyond appearance, position and frame-difference energy, and the trained detector relies on them."* Adoption into KAT-VAD additionally needs the span bar.

### 0.1 Expected effects (stated before any EMS data)

| Quantity | Prior | Basis |
|---|---|---|
| EMS span macro gain (M − base) | **≈ 0 to +0.015** (*judgement*) | probe headroom: `F0` 0.853 vs `p` 0.847 (≈ +0.006 for `x̃` + `ũ` inside the probe, *derived*); change already exploited (`m5` 0.750 alone; S3 shows the probe extracts it); flow's extra is direction and location of independent motion, mainly for non-ego (248/569 = 44 %, *derived*): +0.03 there → +0.013 overall |
| P(EMS span bar passes) | below even (*judgement*) | the bar is +0.01 with both lower bounds > 0 |
| EMS ΔT gain | open; plausible | ΔT removes most of the position floor (R0-a will show by how much) |
| A3-B span gain | ≈ 0 to +0.02 (*judgement*) | E2(d): B − S beyond-position transfer ≈ +0.019 at probe level (*derived*: 0.037 − 0.018); trained-arm gains are usually smaller |
| KIP precedent | neutral | RAFT flow as a **target** gave +0.0076 [−0.0002, +0.0154]; EMS uses flow as an **input** |

---

## 1. Results of S1–S3 (read-out 2026-10-13)

**Protocol.** Probes trained on T2-train − T2-val, decided on DoTA-CAP-dev (569 clips, 91 source clusters, protocol B). Bar: cluster-bootstrap 95 % lower bound > 0 **and** point ≥ +0.01, B = 10,000, seed-averaged scores (5 seeds).

| Step | Contrast | Result (DoTA-CAP-dev) | T2-val | Verdict |
|---|---|---|---|---|
| **S1a** | AUC(`m5`) − 0.5 | **+0.2503 [+0.2330, +0.2669]** (AUC 0.7503) | +0.1943 | PASS |
| | AUC(`m5`) − AUC(`m5^π`) | +0.2713 [+0.2537, +0.2891] | +0.1977 | (twin below chance, C-a) |
| **S1b** | Δu post-span vs margin ±0.1 SD | +2.877 [+1.418, +4.362] vs ±0.638 | — | Δu not admitted |
| **S1c** | SMD steps 0–2 vs ≥ 3 | `s` +0.377, `m5` −0.346, `v̄` +0.395; 61/131 κ channels > 0.1 | — | early-onset prediction withdrawn |
| **S2** | `F0` true − block shuffle | 0.8532 vs 0.8250, **+0.0283 [+0.0212, +0.0356]**, 5/5 seeds; `f_2` +0.0138 | +0.0165 | PASS |
| | CLIP-only (printed) | +0.0119 [+0.0015, +0.0230] | +0.0252 | — |
| **S3(a)** VALUE | `(F0+κ) − F0` | **−0.0007 [−0.0063, +0.0047]** | −0.0004 | **FAIL** |
| **S3(b)** ORDER | `(F0+κ) − (F0+κ^π)` | +0.0080 [+0.0021, +0.0140], `f_2` ∋ 0 | −0.0034 | **FAIL** |

Printed: per-clip corr(AUC(`m5`), AUC(`p_T2`)) −0.053; `m5` ego 0.785 vs non-ego 0.706; top-5 `m5` steps inside span 0.593; DADA↔DoTA shortcut AUC on κ 0.905; `F0` reproduces (0.8532 / 0.8538).

**Permitted wording (unchanged):** *Step order finer than ≈ 1.5 s carries signal beyond coarse position, but explicit velocity/acceleration features add nothing a small temporal model does not learn itself.*

**Caveats:** C-a (twin below chance); **C-b (confirmed in round 6):** true `F0` 0.853 ≈ cubic `p` 0.847, shuffled 0.825 below it; the shuffled probe was **trained** on shuffled train/val/DoTA parts alike (`order_diagnostics.py:412-413`), so the shortfall is not test-time distribution shift: a probe trained on shuffled context transfers worse than `p` alone; C-c (`m5` low at clip starts); C-d (κ padding did not remove the start spike); C-e (echo vs aftermath); C-f (`p_T2` beside macros); C-g (0.853 is not a detector result).

---

## 2. Root-cause analysis

### 2.1 What a motion claim needs

1. **New information** the inputs (`x̃`, `ũ`, `p`) do not determine.
2. **A place to act**: an estimand where a motion cue can change the ranking, with position controlled.
3. **Specificity**: motion *structure*, not any temporal pixel change.
4. **Use**: the trained detector's decisions depend on it.

| Attempt | (1) | (2) | (3) | (4) | Outcome |
|---|---|---|---|---|---|
| KIP (v1): flow as a **target** from CLIP | flow not an input | span | — | — | +0.0076 [−0.0002, +0.0154] |
| VideoMAE V2-S (v2) | new encoder, appearance (D6 ±0.0004) | span | D6 | — | +0.098, appearance |
| κ / TDP (S3) | ✗ function of `x̃` | span | κ^π | — | null |
| EMS-v1 | partly (parallax) | span | δ | ✗ | not run |
| **EMS-v2.1** | measured flow + **M0-R** (local block) | **ΔT** (+ futility) | **δ (identifying)** | **U0 shift**, U1, U2 | — |

### 2.2 RC1 — no new information entered (primary; reworded in round 6)

- κ = `f(x_t, …, x_{t−3})`; the probe reads `x̃` in order over 9 steps. By the data-processing inequality κ adds no label information; it can only ease learning. VALUE −0.0007 (width 0.011) and **the same null in-domain on T2-val (−0.0004)**: redundancy, not corpus shift.
- **What CLIP does and does not contain (corrected).** KIP Option A regressed RAFT statistics from CLIP: **magnitude block R² 0.357, direction block R² 0.071** (`RESULTS_DADA_ORIG_T2.md:75-76`). So flow **direction** is essentially absent from CLIP, while flow **magnitude** is partly recoverable. Many EMS channels are magnitudes (ego magnitude, mean `|f^H|`, `e^F`, `|r|`, `ρ`, `m5ρ`); they may be partly redundant. M0-R therefore reports **direction-type** and **magnitude-type** channels separately, and the **ego block is pre-registered to fail** M0-R (§4.4).
- The global change of an appearance embedding measures *what changed* (jolt, occlusion, new objects), not *how things move*: `m5` ego 0.785 > non-ego 0.706; both streams change after the span (S1b).
- **KIP precedent.** Flow as an auxiliary **target** gave +0.0076 [−0.0002, +0.0154]. That is not counter-evidence (EMS uses flow as an **input**), and not supporting evidence either.

**Implication.** A motion candidate must be computed by a displacement operator, and its **local** channels — where direction and the location of independent motion live — must be shown not reconstructable from `F0`.

### 2.3 RC2 — the estimand hides motion (primary; unchanged)

| Fact | Number | Consequence |
|---|---|---|
| Position prior alone | `p_T2` 0.822 (dev); cubic-`p` probe 0.847 | the probe starts at ≈ 0.85 |
| `F0` true order | 0.853 ≈ 0.847 (C-b) | `x̃` + `ũ` add ≈ +0.006 inside the probe (*derived*) |
| Shuffled `F0` | 0.825 < 0.847 (trained shuffled) | S2's Δ partly measures a broken context |
| Top-5 `m5` steps inside span | 59 % | precursors and manoeuvres become false positives |
| A3 peak | +1–2 s after onset; argmax in span 61 % | the model already learns the moment of impact |
| Aftermath | both streams change after the span | motion there is penalised |

Span AUC is the right **adoption** metric and the wrong place to **decide a motion claim**.

### 2.4 RC3 — learned motion weights are corpus-specific (contributing)

κ separates DADA from DoTA at 0.905; CRN brought `x̃` to 0.57. Frame rate, **aspect ratio** (2.4 vs 1.78) and step length enter motion statistics. Round 6 adds two self-inflicted sources, now fixed: the anisotropic squash (A4) and the unverified DADA fps (A5); and one within-DoTA source: native 10 vs 30 fps (A6).

### 2.5 What is *not* the root cause (unchanged)

No motion signal (S1a 0.750); bar too strict (S3(a) interval inside ±0.01); probe unstable (reproduces); VideoMAE should carry motion (D6).

### 2.6 Audit of EMS-v1 and of EMS-v2 (round 6)

| Defect | Version | Fix in EMS-v2.1 |
|---|---|---|
| Homography residual = parallax under forward motion | v1 | epipolar `e^F` (depth-free for static points) |
| Movers along epipolar lines invisible | v1/v2 | looming excess `ℓ` |
| Raw homography entries ill-conditioned | v1 | 4-point ego |
| **`e` changes meaning at `η_H`; `ā` turns the switch into an event** | **v2** | separate `e^F`, `|r|`; `v_t` as a channel; hold-last; gated `ā` |
| **Fallback by homography inlier ratio re-imports parallax on dashcams (road plane dominant)** | **v2** | F usability by **off-plane support** (DEGENSAC-style); K2 gates F-usable share and speed coupling |
| **Anisotropic, corpus-dependent squash** | **v2** | aspect-preserving resize; isotropic units |
| **Pair interval assumed** | **v2** | measured fps; per-second units |
| **U0 not an ablation** | **v2** | circular shift |

### 2.7 R0 — root-cause diagnostics (free, descriptive; run first)

Nothing here decides anything except the two ΔT demotion rules (§5.3) and the confirmatory-set rule (§7.2), all fixed now. **TDP is not re-opened** whatever R0 shows.

| # | Diagnostic | Prediction if RC1/RC2 are right |
|---|---|---|
| **R0-a** | ΔT of `m5`, `s`, `p_T2`, A3-ign on CAP-dev; eligible count; **per fps group** | ΔT(`m5`) > 0.750; ΔT(`p_T2`) well below 0.822. **Futility check** on A3-ign |
| **R0-b** | ΔT of saved S3 probes: `F0` vs `F0 + κ` | Δ ≈ 0. **Futility check** on `F0` |
| **R0-c** | M0-R regressor on κ | mean R² ≥ 0.5 (calibrates M0-R) |
| **R0-d** | Top-5 `m5` steps outside the span: share in [t_s − 2 s, t_s) | large precursor share |
| **R0-e** | `p`-only conv probe (5 seeds) | ≈ 0.85 |
| **R0-f** | **DoTA-train check:** count CAP-source clips whose DoTA ids lie in DoTA's train split; source-disjointness from CAP-dev/eval; whether any were ever scored or used in a fit (σ, shortcut tests) | — (decides §7.2 only) |

---

## 3. Alternatives considered

### 3.1 Motion routes (unchanged except the EMS row)

| Route | New information (RC1) | Ego vs independent | Corpus risk | Cost | Paper support | Verdict |
|---|---|---|---|---|---|---|
| TDP / κ variants | ✗ | ✗ | high (0.905) | free | LAS-VAD IAM | closed |
| Flow-stream encoder (I3D/X3D on flow) | ✓ | ✗ | high | medium | Pi-VAD | hook |
| Motion-aware MVM encoder | ? (D6) | ✗ | medium | high | SimpleTAD App. B | hook (DF6) |
| Object kinematics | ✓ | ✓ | low–medium | high | LAS-VAD summary | v3 |
| **EMS-v2.1** | direction ✓ (KIP); magnitude partly ✗ → M0-R by type | ✓ explicit | lower (isotropic, measured fps, gate) | low (224 floats/step) | LAS-VAD, Pi-VAD, SimpleTAD, RefineVAD | **chosen for the claim** |

**Why EMS over a flow-stream encoder** (unchanged): explicit ego/independent separation, physically named channels, an interpretable STRUCTURE control.

### 3.2 Fusion point (unchanged)
Early zero-init residual at `h`, with use tests; a late score head was declined.

### 3.3 Baseline (unchanged)
LaGoVAD trunk; every paired run is on it.

### 3.4 Score track (new in round 6)

| Option | Targets | Evidence | Cost | Verdict |
|---|---|---|---|---|
| **SC-1 DF8 swap read-out** (H1) | how the `p_T2` > A3-ign gap is told | results §6.2: the prior collapses off its usual position | forward passes only; plan `katvad-v2-df8-position-shift.md` | **run first** |
| **SC-2 H4: A3-B** | appearance capacity | E2(d): B +0.037 vs S +0.018 beyond position (transfer); T2 side +0.137 vs +0.107 | V2-B extraction for full T2; 5 runs | **run** |
| Feed `p_T2` to the model | position | would reach 0.84–0.90 | free | **declined** (needs the clip end; fails in streaming; collapses off-position: results §8) |
| A position-aware remedy | position | none measured yet | — | **deferred** until SC-1 shows whether the gap is a benchmark artefact |

**SC-1 registered readings.** (i) If `p_T2` falls below A3-ign on the swapped set (with lower bound > 0 for A3-ign − `p_T2`), the CAP-eval gap is told as **benchmark position structure**. (ii) If `p_T2` ≥ A3-ign there, the "position is an artefact" argument is withdrawn and a position remedy becomes a v3 question. SC-1 adopts nothing.

**Interaction of the tracks.** The base model for M2 is **frozen when M2 starts**: A3-ign, or A3-B if SC-2 has adopted it by then. M2's control is a re-run of that base. EMS and V2-B are never combined in any other way.

---

## 4. EMS-v2.1 — the Ego-Compensated, Epipolar Motion Stream

### 4.1 Definition (offline, per source video or clip)

```
per sequence:
  fps        = measured (container metadata AND frame count ÷ duration; both printed; must agree within 2 %)
  k          = round(0.1 s · fps) ; Δt_f = k / fps            (DoTA 10 fps: k = 1; 30 fps: k = 3; DADA: measured)
  resize     = aspect-preserving to width 448 (DADA 448 × 187 → pad 192; DoTA 448 × 252 → pad 256); padding masked
  coordinates: p = (x/W_native, y/W_native)   ← isotropic frame-WIDTH units for both axes
  grid       = 56 × 28 points, area-pooled over equal frame-FRACTION cells (same layout on both corpora)
  M          = static mask (hood, dashboard, overlays, padding), v1 rule

for each step t (time τ_t from measured fps):
  f_t        = RAFT-small(I_{t−k}, I_t) on the resized pair, pooled to the grid, ÷ Δt_f   (width units per second)
  -- ego-motion --
  H_t        = RANSAC homography on unmasked (p, p + f_tΔt_f); inlier ratio π^H_t
  ego_t      = (H_t(c_k) − c_k) / Δt_f at the 4 quadrant centres                         8 numbers
  -- epipolar --
  F_t        = RANSAC fundamental matrix on the same correspondences
  n^P_t      = # F-inliers that are H-outliers                  (off-plane support; DEGENSAC-style)
  v_t        = 1 iff ≥ 15 F-inliers AND n^P_t ≥ 8               (F usable; no η_H)
  e^F_t(p)   = Sampson distance of (p, p + f_tΔt_f) to F_t, ÷ Δt_f        if v_t
             = e^F_{t'}(p) of the last usable step t' < t        otherwise (hold-last; first usable value back-filled;
                                                                  clip with no usable step: 0 and flagged)
  -- homography residual (always) --
  r_t        = f_t − (H_t(p) − p)/Δt_f ;  |r_t|
  ℓ_t        = div r_t   (central differences with the true cell spacing in width units)
  λ_t(p)     = −sign(p_x − 0.5) · r_{t,x}(p)      inward lateral
  ν_t(p)     = r_{t,y}(p)                         vertical

  φ_t ∈ ℝ^110 : ego (12)   = ego_t (8), mean |ego flow| (1), π^H_t (1), v_t (1), n^P_t / #unmasked (1)
                cells (96) = per 4×4 cell: mean e^F, p90 e^F, p90 |r|, mean ℓ, mean λ, mean ν
                global (2) = ρ^F_t = max cell p90 e^F ; ρ^H_t = max cell p90 |r|
  ā_t ∈ ℝ^109 : mean over t−2..t of |φ_τ − φ_{τ−1}| / Δt_step  for every φ channel except v_t;
                for the 32 e^F cell channels and ρ^F the difference counts only if v_τ v_{τ−1} = 1 (else 0)
  summaries (5): 1.5 s causal max of ρ^F, of ρ^H, of ego jerk ‖Δego‖/Δt_step; 1 s traces of ρ^F, ρ^H
  ψ_t = [φ_t ; ā_t ; summaries] ∈ ℝ^224 ; replicate padding at the ψ level for t ≤ 1
  ψ̃_t = (ψ_t − median_τ ψ_τ) ⊘ σ_ψ        per-clip median; σ_ψ per channel on T2-train, fixed
```

**Channel subsets:** **ego** = 12 `φ` + 11 `ā` + ego-jerk max = **24**; **local** = 98 `φ` + 98 `ā` + 4 summaries = **200**. **Direction-type** (reported at M0-R): `ego_t`, `λ`, `ν` and their accelerations = 80; the rest are **magnitude-type** or geometry quality.

**In the model (pointwise; unchanged form):**

```
g_t = GELU(W_1 ψ̃_t + b_1)                 W_1 ∈ ℝ^{128×224}
h_t = x̃_t + W_u(c·ũ_t ⊘ σ_u) + W_m g_t     W_m ∈ ℝ^{512×128} (+ bias), INITIALISED TO ZERO
```

New parameters **94,848** (*derived*: 224·128 + 128 + 128·512 + 512). Zeroing `ψ̃` does **not** give an exact ablation: it feeds a constant `W_m GELU(b_1)` into an encoder with LayerNorm and attention, which is an out-of-distribution input, not a constant score offset (A2). Use is tested by shifting `ψ̃` in time (§6.2).

### 4.2 Why each choice

| Choice | Reason | Source |
|---|---|---|
| Flow as an inference input | direction is absent from CLIP (R² 0.071) | KIP; Pi-VAD |
| **`e^F` and `|r|` as separate fixed-meaning channels** | no channel changes units at a threshold; the MLP can weigh parallax-free and parallax-laden evidence (A1) | round 6 |
| **Off-plane support for F usability** | a dominant plane makes F estimation degenerate; F is usable when enough correspondences off the plane support it, regardless of how much of the image the road fills | DEGENSAC (Chum et al. 2005) |
| **Hold-last `e^F`; gated `ā`** | a lost F fit must not look like an event (A1) | round 6 |
| `v_t`, `n^P` as channels | geometry quality is information (impact, blur, occlusion collapse it), and the model must know when `e^F` is stale | — |
| Looming excess `ℓ` | covers along-axis movers (epipolar blind spot) | LAS-VAD weakness 1; IAM |
| Inward lateral `λ`, vertical `ν` | traffic-meaningful directions | Gap N |
| 4-point ego + ego jerk + inlier ratio | well-conditioned; impact = jolt | — |
| **Isotropic width units; aspect-preserving resize; frame-fraction grid** | removes the corpus-dependent x/y rescaling (A4) | round 6 |
| **Measured fps; per-second units** | the design no longer rests on DADA = 30 fps (A5) | round 6 |
| ≈ 0.1 s pairs, per-second acceleration | 10 fps on both corpora | SimpleTAD |
| 4×4 grid | keeps *where* | results §6.4 |
| Offline temporal ops; pointwise block; zero-init; separate generator | splice-safe; starts at the base | rounds 1, 4 |
| Per-clip median + fixed σ_ψ | removes clip constants | CRN |
| Synthetic-data RAFT weights; no new loss | no benchmark labels; G4 | v2 |

### 4.3 Risks and how each is caught

| Risk | Caught by |
|---|---|
| Flow failure (night, rain, wipers, glare, compression) | K2/M0: H-fit failure ≤ 20 % of steps per corpus **and per DoTA fps group** |
| **F unusable too often** (slow traffic, dominant plane) | **K2 gate:** F-usable share (`v_t = 1`) ≥ 50 % of steps per corpus and per fps group; else redesign before extraction |
| **`e^F` still carries parallax** | **K2 gate (speed coupling):** on all pilot steps (labels not read; most steps are normal), median per-clip Spearman ρ(cell p90 `e^F`, mean ego flow) ≤ median ρ(cell p90 `|r|`, mean ego flow) − 0.10 |
| fps misassumed | K2/M0: both fps estimates printed; disagreement > 2 % flags the clip |
| Descriptor = pixel change | δ (STRUCTURE) |
| Descriptor reconstructable from `F0` | M0-R (local block) |
| Corpus signature | M0 corpus gate (ψ̃ DADA-vs-DoTA AUC ≤ 0.70, else per-clip MAD scaling); within-DoTA 10-vs-30 fps AUC reported |
| Turns, lane changes | ego jerk; per-class report |
| Clip-start artefact | SMD of steps 0–2 vs later steps (K2, no labels) and vs later normal steps (M0) (κ reference: 61/131 channels > 0.1) |
| Gain from regularisation or OOD tolerance | U0 shift |
| Cost | M0: RAFT-small FLOPs/step vs ≈ 75 GFLOPs/step |

### 4.4 Registered expectations

**M0-R (A3, Q5).** Expected R² order: ego block > global magnitude (ρ, summaries) > cell magnitude (`e^F`, `|r|`, `ℓ`) > direction (`λ`, `ν`). **Pre-registered: the ego block fails (mean R² > 0.5).** This does not stop the plan (the gate is the local block), but it means any ego-side gain in M2 is read as *ease of learning*, not new information, and the U2-ego result is worded accordingly.

**Subgroups (reported, not gating).**

| Group (dev n) | Expected main carrier |
|---|---|
| non-ego (248) | `e^F`, `ℓ` |
| ego (321) | ego jerk, inlier collapse, `v_t` drop |
| lateral (106), turning (203), oncoming (59) | `e^F` |
| moving_ahead_or_waiting (64) | `ℓ` |
| leave_to_left / right (35 / 43) | ego channels |

---

## 5. Estimands

### 5.1 Three estimands, three jobs

| Estimand | Job |
|---|---|
| **Span macro** (per-clip AUC, protocol B) | **adoption, and only adoption** (both tracks) |
| **ΔT** onset-transition macro AUC | the **motion claim** (motion track) |
| **DF8 swap** span macro (491 CAP-dev clips) | the position story (SC-1) and "beyond the position prior" wording |

### 5.2 ΔT, fixed (unchanged windows)

Per clip with span `[t_s, t_e]`, times from measured fps:
- **positives** P: `t_s ≤ τ < min(t_s + 2.0 s, t_e)` (≈ 7 steps at 0.30 s);
- **negatives** N: normal steps with `t_s − 3.0 s ≤ τ < t_s − 0.5 s` (≈ 8 steps);
- eligible if |P| ≥ 2 and |N| ≥ 3; ΔT = mean per-clip AUC; cluster bootstrap over sources;
- `p_T2` and `f_2`-ΔT printed beside every ΔT; reported per fps group.

### 5.3 Forking paths and demotion rules (new in round 6)

**Statement for the committee.** ΔT is a **second estimand, introduced after the span estimand returned nulls** (S3). Its windows were set with knowledge of A3's peak timing, which was measured on CAP-dev (results §6.3). It was registered before any EMS feature existed and appeared earlier only as a reported directional prediction. We do not claim it was pre-specified from the start. What limits the forking paths:
1. ΔT never decides **adoption** (B1); the benchmark metric does.
2. One primary ΔT contrast per claim (STRUCTURE at M1; M − Mδ at M2); everything else is reported.
3. The windows are fixed now; R0 reads ΔT on old scores **descriptively** and cannot change them.
4. The motion claim needs the **same sign on T2-val**, a different corpus.
5. **Demotion rules**, fixed before R0 runs:
   - **Eligibility:** fewer than 200 eligible dev clips → ΔT demoted to "reported".
   - **Futility:** ΔT(A3-ign) ≥ **0.90** on CAP-dev, **or** ΔT(saved `F0` probe) ≥ **0.90** → ΔT demoted to "reported" (no room for a +0.01 gain to be meaningful).
   - If demoted, M1 and the M2 motion rule fall back to span (STRUCTURE on span; M − Mδ on span; U0 on span), labelled as the round-5 failure mode. The plan then expects a null and says so.
6. Wording: every ΔT result is printed as "ΔT (onset-transition estimand, registered after the span nulls)".

### 5.4 What ΔT can and cannot show (B3)

ΔT rewards **anything that spikes at the onset**, including CLIP change already inside `F0` (`x̃`) and frame-difference energy. Therefore:
- **VALUE** (`(F0+ψ̃) − F0` on ΔT) licenses only "**adds onset signal**".
- **STRUCTURE** (`(F0+ψ̃) − (F0+δ̃)` on ΔT) is the **identifying contrast for motion**: same step, same pathway, same estimand, the only difference being flow geometry versus pixel change. At M2 the same role is played by **M − Mδ**.

---

## 6. Controls

### 6.1 Frame-difference control δ (STRUCTURE)
Per-cell mean |I_t − I_{t−k}| (16) + frame mean (1); acceleration (17); two summaries → **36 dims**; **same resize, padding mask, frame-fraction cells and measured `k`** as EMS (round 6); pointwise block, zero-init (70,784 parameters). **ψ − δ = motion structure beyond pixel change.**

### 6.2 Use tests on the trained arm (inference only; revised in round 6)

| Test | Manipulation | Reads |
|---|---|---|
| **U0 (primary)** | **within-clip circular shift** of all `ψ̃` rows by `o` steps, `o` uniform in [⌈T/3⌉, ⌊2T/3⌋], 5 seeded offsets per clip, scores averaged | `ΔU0 = M − M[shift]` on ΔT and span |
| U0-perm (reported) | within-clip random permutation of rows (5 seeds) | — |
| U0-zero (reported, OOD-confounded) | `ψ̃ := 0` | as K0 |
| **U1** | `ψ̃` shifted later by 1.0 s (first rows replicate) | timing use |
| **U2** | circular shift (same offsets) of the ego channels only / the local channels only; zeroing reported | which half is used |

**Why a circular shift.** It keeps each clip's marginal distribution, median and autocorrelation, so the trunk sees an in-distribution sequence; it only breaks the alignment with the onset, by at least ⌈T/3⌉ steps (≈ 3.3 s on a median DoTA clip, *derived*: 11 × 0.30 s), more than the ΔT window. Its one artefact is a single wrap junction per clip.

### 6.3 Redundancy gate M0-R
The S2 probe architecture as a regressor (`F0 → d` targets), T2-train, R² per channel on T2-val. **Gate: mean R²(local block) ≤ 0.5**, valid only if κ's R² ≥ 0.5 (positive control, R0-c). **Reported:** ego block (expected > 0.5), direction-type and magnitude-type sub-blocks, δ.

---

## 7. Statistics

### 7.1 Bars (unchanged)

| Step type | Uncertainty | Bar |
|---|---|---|
| Probes (K1, M1) | cluster bootstrap on seed-averaged scores; 5 seeds = stability | lower bound **> 0**, point **≥ +0.01** |
| Trained arms (SC-2, M2) | paired t95 over 5 seeds **and** cluster bootstrap | both lower bounds **> 0**, point **≥ +0.01** |
| Use tests (U0) | cluster bootstrap on seed-averaged scores | lower bound **> 0**, point **≥ +0.01** |

Every macro is printed with `p_T2`; every Δ with its `f_2` Δ; **every CAP-dev result also per native-fps group** (A6; reported, not gating). Motion claims need the same sign on T2-val.

### 7.2 Sets, history and confirmation

| Set | Status | Role |
|---|---|---|
| T2-train − T2-val | training | probes, trunk |
| T2-val | guardrail | same-sign check; **K1 reads it** (a different, global feature; labelled) |
| DoTA-CAP-dev (569) | decision set of ≈ 6 earlier choices; now SC-2 and M2 too | decides every gate; arm contrasts share the base, so selection largely cancels |
| DoTA-CAP-eval, T2-test | opened | third look, labelled; decides nothing |
| **DoTA-train clips in CAP's source (R0-f)** | unknown | **Rule (fixed now):** if R0-f finds ≥ 200 ΔT-eligible clips that are source-disjoint from CAP-dev/eval and were never scored or used in any fit, they become a **sealed confirmatory set**: features extracted, labels sealed, only counts printed; **one read**, registered before M2, of (i) each adopted arm vs the base on span and (ii) M − Mδ on ΔT if the motion claim passed on dev. Otherwise the DF8 swap of CAP-eval is the only further condition and is labelled "**new condition on opened videos**" |

**Multiplicity.** Two adoption decisions (SC-2, M2) and one motion claim, each with one primary contrast. Everything else is reported.

---

## 8. The plan

**Order (cheapest first; the two tracks run in parallel where they do not share a decision).**

| Step | Track | What | Trains? | Decides |
|---|---|---|:-:|---|
| P0 | — | S1–S3 | no | done: TDP rejected |
| **R0** | motion | R0-a…f (§2.7); ΔT eligibility and **futility** | no | ΔT demotion (§5.3); confirmatory-set rule (§7.2) |
| **SC-1** | score | DF8 swap: forward passes of A3-ign, A3, `p_T2` (frozen T2 fit) on the 491-clip swap; CRN references recomputed per crop | no | how the position gap is told (§3.4) |
| **K1** | motion | **Free kill-switch.** S2 probe on `F0` vs `F0 + stats23` (the cached flow/v1 per-step RAFT statistics, incl. 16 direction bins), T2-train − T2-val, read on **T2-val** (span and ΔT); 5 seeds | no | **soft stop:** if both Δ points < +0.005, full M0 extraction needs a written advisor go-ahead first. Decides no claim (global, not per-cell) |
| **K2** | motion | **Quality-only pilot**: EMS-v2.1 and δ on 100 T2-**train** sources + 100 CAP-dev clips (50 per native-fps group); **no labels, no probe**. Gates: fps agreement; H-fit failure ≤ 20 %; **F-usable share ≥ 50 %** (per corpus, per fps group); **speed-coupling test**; clip-start SMD reported; FLOPs | no | pass → full M0; fail → descriptor redesign (allowed: no outcome seen) |
| **SC-2a** | score | Extract V2-B for full T2 (may share the M0 extraction job) | no | — |
| **M0** | motion | Extract ψ, δ for T2 and DoTA-CAP (+ DF8 crops); repeat K2 gates at full scale; σ_ψ, σ_δ, mask floor on T2-train; **corpus gate** | no | descriptor frozen |
| **SC-2b** | score | **A3-B**, 5 seeds, trained as A3-ign | yes (5) | **adopt A3-B iff** span A3-B − A3-ign passes the trained-arm bar, `f_2` point ≥ 0, T2-val macro ≥ A3-ign − 0.01, O1′. Read also on the DF8 swap (forward) |
| **M0-R** | motion | regressor on the local block, ego block, sub-blocks, δ, κ | no | proceed iff local mean R² ≤ 0.5 and κ ≥ 0.5 (if κ < 0.5: gate uninformative, M1 decides alone) |
| **M1** | motion | probes on CAP-dev: **STRUCTURE** `(F0+ψ̃) − (F0+δ̃)` on ΔT (**primary**); **VALUE** `(F0+ψ̃) − F0` on ΔT (prerequisite). Reported: both on span; ego/local splits; ego/non-ego; per class; per fps group | no | M2 eligible iff STRUCTURE and VALUE pass on ΔT and VALUE's span point ≥ 0 |
| Advisor | both | outcome table §12, including SC-1, SC-2, K1 | — | whether to train M2 |
| **M2** | motion | 5 seeds each: **base re-run**, **M**, **Mδ**; base = A3-ign or A3-B (frozen when M2 starts). Unit test: `W_m = 0` reproduces the re-run's first 100 losses | yes (15) | rules below |
| **U** | motion | U0–U2 on M; DF8 swap forward passes for the three arms | no | use and position rules |
| P1, S4 | — | optional order analysis | P1 no; S4 5 | analysis chapter |

**M2 rules (fixed now).**
- **Adopt M** iff span M − base passes the trained-arm bar, its `f_2` point ≥ 0, T2-val macro ≥ base − 0.01, O1′. **This is the only adoption route.**
- **"Improves detection"** wording: the span contrast passes.
- **Motion claim** (independent of adoption): ΔT **M − Mδ** passes the bar; `f_2`-ΔT point ≥ 0; same sign on T2-val; **U0 (circular shift) passes on ΔT**. If ΔT was demoted (§5.3), the same rule on span, labelled.
- **Timing:** U1's ΔT drop has lower bound > 0.
- **Beyond position:** DF8-swap span Δ (M − base) same sign, lower bound > 0.
- **Non-ego:** ΔT M − base on non-ego: lower bound > 0, point ≥ +0.01.
- Mδ is never adopted; if Mδ passes and M − Mδ fails, δ is a hook.

**Budget.** R0, SC-1, K1, K2, M0-R, M1, U: minutes to hours on caches. M0 + V2-B: one extraction campaign. SC-2b: 5 runs; M2: 15 runs (E3 was 20).

---

## 9. Why EMS-v2.1 is not KIP, TDP or EMS-v1 again

| | KIP | TDP | EMS-v1 | EMS-v2 | **EMS-v2.1** |
|---|---|---|---|---|---|
| Motion source | predicted from CLIP | differenced CLIP | flow | flow | flow, measured `Δt_f`, isotropic |
| New information shown first | no | no | no | M0-R | **K1** + **M0-R** (local block, by type) |
| Independent motion | — | — | `|r|` (parallax) | `e` switching at `η_H` | **`e^F` (off-plane-supported F) and `|r|`, separate** |
| Claim estimand | span | span | span | ΔT | ΔT **with futility rule** |
| Adoption | span | span | span | span or ΔT | **span only** |
| Identifying control | — | κ^π | δ | δ | **δ, declared identifying** |
| Use test | — | — | — | `ψ̃ := 0` | **circular shift** |

---

## 10. Ideas declined (unchanged)

IAM prototypes / `w_int` / `L_cst`; ACC; local Transformer; similarity GCN; LLM attribute text; TDP variants; feeding `p_T2` to the model.

---

## 11. Claim ladder (pre-written)

| Outcome | Permitted wording | Forbidden |
|---|---|---|
| Established (S1–S3) | "Step order finer than ≈ 1.5 s carries signal beyond coarse position, but explicit velocity/acceleration of the scene vector adds nothing a small temporal model does not learn." | motion wording; TDP |
| SC-1, `p_T2` < A3-ign on the swap | "The prior's advantage on DoTA-CAP is a property of where accidents sit in the clips; it does not survive a position swap, whereas the detector's score does." | "the model beats position" on the original set |
| SC-1, `p_T2` ≥ A3-ign on the swap | "A position prior remains competitive even when positions are swapped; the detector's use of position is an open problem." | "position is an artefact" |
| SC-2 adopts A3-B | "A larger frozen video encoder improves DoTA-CAP-dev detection." | — |
| K1 null | "Global flow statistics add nothing in-domain beyond appearance and position" (reported) | — |
| M0-R fails | "Local flow descriptors are predictable from appearance features; no motion claim." | motion wording |
| M1 STRUCTURE fails, VALUE passes | "Flow descriptors add onset signal, but no more than frame-difference energy does." | "motion-aware" |
| M2 motion rules pass, **adoption fails** | "A motion-aware variant (EMS) improves **onset** detection beyond appearance, position and frame-difference energy, and relies on its motion input; it does not improve the span metric, so the deployed model is unchanged." | "KAT-VAD is motion-aware"; "improves detection" |
| M2 motion rules pass **and** M adopted | "**Ego-motion-compensated, epipolar independent-motion features improve onset detection beyond appearance, position and frame-difference energy, and the trained detector relies on them.**" KAT-VAD may be called motion-aware (image-motion sense) | "vehicle speed", "trajectories", "time-to-collision", "vehicle kinematics" |
| … U1 / DF8 / non-ego pass | "… including when the motion happens" / "… and the gain holds under a position swap" / "… largest on non-ego accidents" | — |
| ΔT demoted | every ΔT number labelled "reported (demoted: eligibility/futility)" | ΔT-based claims |
| any eval number | "third look at an opened set" | unlabelled eval numbers |

---

## 12. Outcomes and thesis-level risk

| Outcome | Thesis gets | Runs |
|---|---|---|
| SC-1 only | a settled position story | 0 |
| SC-2 adopts | a better deployed model (score) | 5 |
| K1 null, advisor stops EMS | a complete negative chain: KIP (induction) → D6 (encoder) → S3 (redundancy, R0-c) → global flow adds nothing (K1) | 0 |
| K2 fails | a measured reason epipolar features are unreliable on these dashcams (F-usable share or speed coupling) — a finding in itself | 0 |
| M0-R or M1 fails | negative chain at the local-flow level, on the right estimand with the identifying control | 0 |
| M2 motion passes, not adopted | a motion-aware **variant** with value, structure and use evidence on onsets | 15 |
| M2 motion passes and adopted | a motion component in the deployed model | 15 |

---

## 13. Modalities, LLM, data

- **Modalities:** RGB → CLIP; RGB clips → VideoMAE V2-S (or V2-B if SC-2 adopts); language definitions; RGB frame pairs → flow → EMS-v2.1 (only if M is adopted).
- **LLM:** Holmes-VAU ATS, off the critical path. Hook: maximum ego jerk or looming step as "time of abrupt motion".
- **Data:** T2 training windows; DoTA-CAP-dev decides; protocol B. DF8 crops from CAP-dev. Possible confirmatory set per R0-f.
- **fps record:** DADA fps measured at K2 and written into `CLAUDE.md`, replacing assumption A1. If it is not 30, every "0.27 s step", "1.5 s window" and "10 fps" statement about DADA is corrected in the thesis text; cached features are not re-extracted.
- **TAD** (deferred): fixed cameras make F degenerate (`v_t` ≈ 0 throughout), so `|r|` carries the signal: a clean test of the residual channels as its own project.

---

## 14. Justification ledger

| Decision | Rationale | Source |
|---|---|---|
| Keep LaGoVAD / A3-ign as base | all paired runs on it | LaGoVAD; results §4 |
| **Two tracks; goal stated** | EMS serves a claim; H4 and DF8 serve the score and the position story | round 6 §0, Q1 |
| **Adoption on span only** | adopting a model that does not improve the benchmark is not a result | B1 |
| Root cause RC1 (reworded) + RC2 | in-domain S3 null; KIP magnitude 0.357 / direction 0.071 | §2; A3 |
| **`e^F` / `|r|` separate; off-plane support; hold-last; gated `ā`** | no threshold-switched channels; parallax not re-imported | A1; DEGENSAC |
| **Isotropic units; aspect-preserving resize** | removes corpus-specific x/y scaling | A4 |
| **Measured fps; per-second units** | no assumed frame rate | A5 |
| **Per-fps-group reporting** | DoTA-CAP mixes 10 and 30 fps | A6 |
| M0-R on the local block, by type; ego expected to fail | direction is the new information; magnitude partly redundant | A3 |
| ΔT for the claim, **with forking-paths statement and futility rule** | RC2; honesty about the order of estimands | B2 |
| **STRUCTURE identifies motion** | ΔT rewards any onset spike | B3 |
| **U0 = circular shift** | in-distribution manipulation that destroys only alignment | A2 |
| **K1, K2 before extraction** | cheapest information first | D |
| DoTA-train confirmatory rule | the only possible unscored DoTA data | Q1/Q13 |
| δ control; corpus gate; zero-init; pointwise; offline temporal ops | unchanged | rounds 1–5 |
| One bar per step type; T2-val same sign; third-look labels; advisor check | pre-registration | rounds 2–3 |

---

## 15. Open hooks and questions

**Questions still open (for you / the advisor):**
1. **K1 threshold.** The soft stop uses "both Δ points < +0.005" on T2-val. Is a written advisor go-ahead the right consequence, or should a K1 null stop the motion track outright?
2. **SC-2 extraction cost.** V2-B for full T2 is the largest single job in either track. If compute is tight, which runs first: V2-B (score) or EMS M0 (claim)? The plan assumes one shared campaign after K2.
3. **R0-f data access.** Does the DoTA download include the train split's videos and temporal annotations locally? If not, the confirmatory-set rule cannot fire and DF8-eval becomes the only further condition.

**Hooks:** object kinematics (v3; EMS's `e^F` map can propose regions); flow-stream encoder with a D6-style check; cheaper flow (Farneback, codec vectors) on DF8 or TAD; δ stream if Mδ passes and M − Mδ fails; streaming (past-only medians; flow and geometry are causal); a position-aware remedy after SC-1; DF3 (`L_neg`).
