# KAT-VAD v2.2 — batch 3 / SC-2: A3-B vs A3-ign (plan)

**Status: DEFERRED 2026-10-15 by the user** (largest GPU job of the project: V2-B on every T2 source + 5 runs). Built and committed; nothing run.

**Branch:** `v2`. **Rule:** v2 addendum **§25 (Amendment 14, Y1–Y11)**, written 2026-10-15 before any V2-B T2
feature beyond K's 300 sources and before any A3-B run. **Design:** v2.2 proposal §3.4, §7.1, §8 (SC-2a/b), §R7 D5,
D9. **Authority:** the user, 2026-10-15 ("commit đi rồi làm batch 3"), under round 7's "decide what you can and run"
(not advisor-reviewed). Batch context: `katvad-v2.2-two-tracks.md`.

## 0. Status

| step | what | where | state |
|---|---|---|---|
| P0 | rule → addendum §25 (Y1–Y11) | `core/docs/v2/PREREG_ADDENDUM.md` | **written 2026-10-15**; commit before the run |
| P1 | `core/tools/sc2_readout.py` — Y5–Y9 mechanically; `V22_SC2_*` constants | local | **built** (+11 tests) |
| P2 | `position_shift score --model A3B` (Y10) | local | **built** (`SCORED_MODELS`) |
| P3 | runbook `colab/v2/sc2_a3b.ipynb`: V2-B extraction → bake → train ×5 → diag → protocol B → read-out → (DF8 if adopted) | Colab | **built, DEFERRED (user, 2026-10-15)** |
| P4 | read-out → `core/docs/v2.2/RESULTS_SC2.md` | — | waits for P3 |

## 1. The question

E2(d) picked V2-S by a **cost** tie rule (N7: within 0.02 → cheaper encoder), while V2-B was ahead on the
beyond-position transfer read (+0.037 vs +0.018; A3 transfer 0.1508 vs 0.1325). SC-2 asks whether that probe-level
edge survives as a trained arm: **does A3-ign with V2-B beat A3-ign with V2-S on DoTA-CAP-dev span?** Expected
≈ 0 to +0.02 (*judgement*, proposal §0.1).

## 2. What already exists (measured 2026-10-15, no number read)

- V2-B DoTA-CAP cache: all 569 `dota_cap_dev` clips (`dota_cap_videomae.ipynb`, read by E2(d)).
- V2-B T2 cache: K's 300 sources only, K geometry (s8, causal 1.5 s, squash). SC-2a extends it exactly as §15 G2
  extended V2-S.
- Control: E4-0's five A3-ign runs (J10-verified), their protocol-B read and diags (`v2_e4_0/`).
- Every downstream tool is encoder-generic: `build_v2_inputs fit/apply`, `core.train` (`v2.motion`,
  `model.motion_dim`), `v2_diagnostics`, `protocol_b_eval`, `position_shift heads --encoder`.

## 3. Decisions taken (Y-numbers in §25)

| # | choice | alternative not taken |
|---|---|---|
| Y1 | config diff vs A3-ign = exactly `{v2.motion, model.motion_dim}`; CRN R2; same train split | re-tuning `c` / σ_u for the wider stream (changes two things at once) |
| Y2 | control = E4-0's runs, not retrained | a fresh A3-ign re-run (5 more runs, same seeds, no information) |
| Y5 | bar = §7.1 trained-arm bar: paired t95 **and** cluster bootstrap lower > 0, paired mean ≥ +0.01 | E4-0's lighter bar (t95 only) — proposal §7.1 fixed the stricter one |
| Y7 | guardrails = T2-val macro (−0.01) + O1′; peak printed | E4-0's peak guardrail (not in SC-2b's row) |
| Y10 | DF8 forward **only if adopted** | always (≈ 1.5–3 GPU-h of streaming for a model that may not be deployed; the user asked for fewer runs) |

## 4. Risks

| risk | effect | handling |
|---|---|---|
| V2-B extraction is the largest job (≈ 3–4× V2-S per window, all T2 sources) | one or two Colab sessions | resumable shards; wall time printed (Y9) |
| `σ_u`, `c` are refit for V2-B by `fit` (the scaling is part of the arm) | not a pure encoder swap at the input scale | the same rule that built A3's cache; recorded in the bake manifest |
| Dev set has been read ≈ 7 times | selection optimism | arm contrast shares the base; CAP-eval not read here; optimism dev − eval was negative for every v2 arm (Final) |
| `f_2` is placement-bound (pending (bn)) | Y6 reads a DoTA-placement quantity | qualified in the read-out sentence; it is a ≥ 0 floor, not the decision |

## 5. Budget

GPU: V2-B on the remaining T2 sources (one to two sessions) + 5 training runs (E4-0: ≈ 1 session) + diag + protocol B.
CPU: read-out. Step 8 (DF8 forward) only on adoption.

## Appendix A — read-outs

(empty)
