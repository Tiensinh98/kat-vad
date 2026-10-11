# KAT-VAD v2.1 — E4a plan: the SG-NM gate on `B` = A3-ign

Status (2026-10-11): **PLAN — not built. Waits for the user's sign-off on §5 (K1–K8) and on the CRITICAL-impact
data-path change (§3.1).** Design source: `core/docs/v2.1/KAT_VAD_PROPOSAL_v2.1.md` §5.3, §7, §11.1–§11.2;
`KAT_VAD_v2.1_ARCHITECTURE.md` §9. Base `B` fixed by E4-0 (`core/docs/v2.1/RESULTS_E4_0.md`).

## 0. Status table

| step | what | state |
|---|---|---|
| P0 | segment labels in the DVS data path | not started (needs approval: CRITICAL blast radius) |
| P1 | `core/models/sg_nm.py` (branch + `L_compact`, `L_rec`, `S̃`, EMA) | not started |
| P2 | `core/tools/e4a_gate.py` (frozen trunk → train branch → score DoTA-CAP-dev + T2-val) | not started |
| P3 | `core/tools/e4a_readout.py` (5 seeds → gate decision) | not started |
| P4 | `colab/v2/e4a_gate.ipynb` (atomic publish, lesson (bi)) | not started |
| P5 | run (user) → read-out | — |

## 1. What E4a is (proposal §11.2, unchanged)

For each of `B`'s five checkpoints (`outputs/v2_e4_0/A3ign/s2024…s2028`): compute frozen `V^t` and `y^bin` on the
training stream (T2-train − T2-val, DVS on, `B`'s anchor mode), train **only** the SG-NM branch (`L_compact + L_rec`,
1,740 steps), then score whole DoTA-CAP-dev clips (protocol B) with label-free candidates (bottom 40 % `y^bin` per
clip) and frozen `μ_e`, `σ_e`. Per seed Δ = macro(rank-avg(`y^bin`, `S̃`)) − macro(`y^bin`); same Δ on `f_2`.
**Gate:** paired t95 of Δ above 0 **and** seed-averaged `f_2` Δ ≥ 0 → E4 runs; else SG-NM is dropped, v2.1 = `B`.
Reported: `S̃` alone macro, per-clip corr(`S̃` AUC, `p_T2` AUC) (ref **0.147** on `B`, drop-if > 0.4), corr with
`y^bin`, the same Δ on T2-val, `B`'s T2-val span profile (already in E4-0's diag), prototype usage.

## 2. What E4-0 changed in the reading (from `RESULTS_E4_0.md` §5)

- References move to `B`: argmax-in-span 0.664, pre 0.239, inside 0.708, tail clips 277.
- SG-NM's remaining target is span **coverage** (impact-locked fade is sharper on `B`); the "pull the 0.55 baseline
  down" argument is spent. No rule changes — this is how E4 is read, not how E4a is gated.

## 3. What the code lacks

### 3.1 Segment labels (P0) — CRITICAL blast radius

`compose_sequence` (`core/data/synthesis.py:54`) returns `pseudo_label` only, which is **all-zero on an unspliced
abnormal window** (weak supervision), so the per-segment candidate rule cannot be derived from it.
`trace_path(compose_sequence, inbound)` → `DVSFeatureDataset.__getitem__` at hop 1 = **CRITICAL** (every training run
goes through it).

**Change (additive only):** `SynthesizedSample` gains `segment_id (T,) long` (window index in the spliced order) and
`segment_abnormal (T,) float` (1 on every step of an abnormal anchor, spliced or not; 0 on fillers and normal
anchors); `truncate_sample` slices them; `DVSFeatureDataset` emits them; `collate_variable_length` pads them
(`segment_id` pad = −1). **No loss reads them.** RNG consumption unchanged.

**Proof of no behaviour change (tests, before any GPU):**
- for fixed seeds, every pre-existing batch field is bit-identical before/after (`v_feat`, `pseudo_frame_label`,
  `label`, `is_synthesized`, lengths) over a full epoch of a synthetic corpus;
- `test_baseline_parity` and the whole suite stay green (1017 / 0 at the E4-0 build);
- a trained-step regression: 3 steps of A3-ign config on the synthetic corpus give identical losses.

### 3.2 The branch (P1)

`core/models/sg_nm.py`, exactly architecture §9 stages 1–9: learned `Q ∈ ℝ^{2×512}`, MHA 8 heads with key mask `m_n`
→ LN → `P`; `L_compact`; decoder MLP 512→512→512 GELU, layer 1 **without residual from `G`**, layer 2 with
residuals, FFN 512→1024→512; `e = 1 − cos(R, V̄)`; `L_rec`; `S̃ = clip((e − μ_e)/(3σ_e), 0, 1)` with EMA 0.99 over
normal-segment valid steps; `L_consist` implemented but unused in E4a. ≈ 5.8 M params (asserted). Candidate selector
`ceil(0.4 · 20) = 8` per abnormal segment (constant `V2_1_SGNM_ABN_CANDIDATES`), all valid steps per normal segment;
the label-free variant for scoring (bottom 40 % of the clip). Prototype usage logged.

### 3.3 Gate tool (P2) and read-out (P3)

- Trunk = `load_finished_model` (J10 gate on `global_step`) in **eval mode, no grad** (K1); the training stream is
  `DVSFeatureDataset` with `B`'s config and seed, regenerated per epoch exactly as training does.
- Branch optimizer = trunk's: AdamW 5e-5, wd 0.01, 20-step warm-up, cosine, AMP; 1,740 steps; seed = `B`'s seed (K6).
- Scoring: protocol B inputs (`A3ign` baked DoTA-CAP cache, s1[::3], whole clip), `S̃` at steps → native frames with
  protocol B's interpolation; fusion on native frames (K2).
- Output per seed: `clip_scores.npz` for `S̃` and for the fused score (DoTA-CAP-dev + T2-val), `branch.pt`,
  `e4a_seed.json` (μ_e, σ_e, usage, losses). Read-out combines 5 seeds → `e4a_readout.{md,json}`.

### 3.4 Notebook (P4)

Atomic publish (`<dst>.partial` + fsync + `os.replace`), skip a seed only if its `e4a_seed.json` verifies; preflight
asserts every `B` checkpoint's step == 1740 and the baked cache manifest == `B`'s arm before GPU. Lesson (bi).

## 4. Debate

| option | pros | cons |
|---|---|---|
| **A. On-the-fly frozen trunk** each step (chosen) | same stochastic DVS stream as E4; no cache to build/version; trunk forward is cheap | trunk forward × 1,740 steps × 5 seeds on GPU (est. minutes per seed) |
| B. Pre-cache `V^t`, `y^bin` per epoch | branch trains on CPU-cheap tensors | 20 epochs × different DVS draws → large cache, a new binding to version; diverges from E4's stream |
| C. Train branch on unspliced windows only | no P0 data-path change | departs from §5.3 (fillers are the clean candidates); changes the gate's meaning → would need an amendment |

**Risks.** (R1) P0 touches the training path of every arm — mitigated by the bit-identical tests; if any field
changes, stop. (R2) Prototype collapse to K = 1 (> 95 % usage) — logged, not gated (proposal). (R3) `S̃` clipped at 0
gives ties on normal steps → rank average uses average ranks (K4). (R4) Eval-mode trunk features differ from E4's
train-mode features (dropout) → the gate measures the information in `B`'s *deployed* features; recorded (K1).
(R5) The 40 % share and K = 2 are fixed, not swept (lesson 14).

## 5. Operational choices to sign before any E4a number

| # | choice | proposed | alternative |
|---|---|---|---|
| K1 | trunk mode while producing the branch's training inputs | `eval()` (deterministic, = deployed features) | `train()` (= E4's dropout) |
| K2 | where `y^bin` and `S̃` are fused | native frames, after protocol B's interpolation of each | at steps, then interpolate |
| K3 | "seed-averaged `f_2` Δ" | `f_2` Δ on seed-averaged scores (as §11.1 defines it for E4-0/E4) | mean over seeds of per-seed `f_2` Δ |
| K4 | rank average | per clip, `(rank(y) + rank(S̃)) / 2`, average ranks for ties, divided by T | — |
| K5 | `S̃` scoring candidates on DoTA | bottom `ceil(0.4·T)` steps of the clip's `y^bin` (at steps) | — |
| K6 | branch seed | `B`'s seed (2024…2028), paired | fixed seed |
| K7 | T2-val Δ | same harness on T2-val windows, printed, not gated | — |
| K8 | P0 data-path change | additive fields + bit-identical tests (§3.1) | option C (needs an amendment) |

## 6. Appendix A — read-outs

(empty)
