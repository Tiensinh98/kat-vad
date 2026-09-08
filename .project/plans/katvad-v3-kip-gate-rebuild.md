# KAT-VAD v3 — KIP gate rebuild (model architecture only)

**Status:** PLAN — not implemented. Written 2026-08-30.
**Source of truth:** `core/docs/v3/KAT-VAD-ARCHITECTURE.md` (Phase 3), `core/docs/v3/KAT-VAD_spec_v3.md` §4.2, §9.
**Scope decided with user 2026-08-30:** model architecture only · selectable `gate_type` enum · existing checkpoints stay loadable.
**Decisions settled 2026-08-30 (round 2):** fresh default `gate_type="rank"` · R2 accepted as a pre-existing invariant, not a new risk · `mlp_ste` and `constant` built now.
**Branch:** `v3` — checked out to hold v3-faithful code only. This is why the default is `rank` with no legacy fallback (T2.4).

> **Path correction.** The request said `core/docs/v2/KAT-VAD-ARCHITECTURE.md`. There is no `core/docs/v2/` in this tree — the docs `CLAUDE.md` §14.1 lists under `v2/` do not exist on disk. The file read is `core/docs/v3/KAT-VAD-ARCHITECTURE.md` (untracked, written 2026-08-29). `CLAUDE.md` §14.1 needs fixing (Phase 5).

---

## 1. Summary

- Replace KIP's **frozen randomly-initialised MLP gate** with the v3 **parameter-free deterministic rank map over an ego-compensated motion residual** (Phase 3b ECMR + Phase 3c rank gate). Score-path parameters go 321 → **0**.
- Keep the old gate reachable as `gate_type="mlp_frozen"` so every measured number in `RESULTS_*.md` stays reproducible, and add `mlp_ste` (spec §4.2 variant) and `constant` (plain-TSM control, ablation ladder #4) for free.
- Take **3e motion head** and **3f alignment projections** off the inference graph without breaking strict checkpoint loading.
- Emit `s_t`, `‖ê_O‖`, `‖μ_t‖` as forward diagnostics — this is what makes ablation-ladder rows 1 and 2 **zero-training** experiments.
- **Encoder unchanged.** Alert-CLIP is a docs-only line in v3; no checkpoint exists publicly. Phase 1 stays `openai/clip-vit-base-patch16` at the pinned revision. Documented as a deviation, not silently ignored.

---

## 2. Assumptions & Constraints

**Assumptions**
- A1. v3 supersedes v1/v2 for Phase 3 only. Phases 1, 2, 4, 5, 6a, 6b are unchanged in v3 and are **not touched**.
- A2. Phase 3d (adaptive bidirectional shift) as written in v3 is **already implemented verbatim** in `shift_channels_vectorized` (`core/kip/gate_shift.py:52`). `s_t ≤ 128` ⇒ `2s_t ≤ 256 < 512`, so the three slices never overlap. **Do not touch 3d.**
- A3. Phase 3a (PMG head, 311,808 params) is unchanged in v3. Not touched.
- A4. No training run is part of this plan. Deliverable is code + tests + docs that pass on CPU, data-free.

**Constraints (hard)**
- C-a. `LaGoVAD-PreVAD/` is read-only. All code in `core/`.
- C-b. Lesson **C14** — every number in `RESULTS_MSAD/DOTA/NCC/PHASE_A/ARM4_PROBE/PREVAD.md` was measured under the frozen-MLP gate. After this change those numbers describe `gate_type="mlp_frozen"` and **nothing else**. Any A/B that mixes gate types is void.
- C-c. Lesson **C5** — checkpoint key mismatches must raise. The compat work below is an *explicit allowlist*, never a relaxed loader.
- C-d. Lesson **C2/C13** — this change does not touch any transform or stride, so no cache is invalidated. Keep it that way.
- C-e. §10 code standards: type hints on public surface, `logging` not `print`, constants in `core/constants.py`, no magic numbers.
- C-f. §11 quality gate before commit; ≤20 files per commit; no `git add .`.

---

## 3. Plan Metadata

- **Plan type:** Architecture change / targeted refactor of one module
- **Size / scope:** Medium — 5 files modified, 2 added, ~450 LOC net, ~35 new tests
- **Estimated duration:** 3–5 working days (1 dev, no GPU)
- **Storage path:** `.project/plans/katvad-v3-kip-gate-rebuild.md`

---

## 4. Phases Overview

| Phase | Name | Goal | Duration | Depends on |
|---|---|---|---|---|
| 0 | Pre-flight & blast radius | Know what breaks before writing a line | 0.5 d | — |
| 1 | Parameter-free primitives | ECMR + rank map as pure, tested functions | 1 d | 0 |
| 2 | Gate dispatch + config surface | Four gate types selectable, defaults safe | 1 d | 1 |
| 3 | Train-only submodules off the inference graph | 3e/3f droppable, old ckpts still load | 1 d | 2 |
| 4 | Diagnostics plumbing | `s_t` observable → ablation rows 1/2 for free | 0.5 d | 2 |
| 5 | Docs, memory bank, lesson, gates | Ship it honestly | 0.5 d | 1–4 |

---

## 5. Detailed Tasks by Phase

### Phase 0 — Pre-flight & blast radius

**Goal:** establish the exact call graph and the exact set of state-dict keys at risk. No edits.

- [x] **P0.1** Run `trace_call_path` (direction `both`, depth 3) on `core.kip.gate_shift.KinematicShift`, `KinematicShift.compute_shift_counts`, `core.kip.kip_module.KIP.forward`, `core.kip.motion_head.MotionScoreHead`. Report the blast radius before editing (CLAUDE.md §13, mandatory). Expected from the read: `KinematicShift` → `KIP` → `KATVAD.forward` (`core/models/kat_vad.py:180`) → `core/train.py`, `core/inference.py`, `core/tests/test_kip_modules.py`. If the index is stale, `index_repository` first.
- [x] **P0.2** Dump the `kip.*` key list from one existing KIP-on checkpoint (e.g. `outputs/MSAD_ncc/stage2_kip_on/`) and record it verbatim in the plan appendix. This is the compat contract Phase 3 must satisfy.
- [x] **P0.3** Confirm the config-default risk: there is **no committed YAML** — `core/config.py` dataclass defaults are the only defaults, and every `outputs/*/config.yaml` is a gitignored run artifact. So flipping a dataclass default silently changes what a *re-run of an old config* does. Pin the mitigation now (Phase 2, T2.4).
- [~] **P0.4** *(BLOCKED — see appendix)* Read RefineVAD's released implementation of MoTAR Eq. 2 (spec §1 annotation) to settle whether they use a straight-through relaxation. **Time-boxed to 1 hour.** Outcome is one line in the lesson/doc either way — it does not block Phase 1.
- [x] **P0.5** Record the pre-change parameter counts by direct measurement (not from the doc table): `KIP` total, `pmg`, `shift.mlp`, `mhead`, `proj_flow`, `proj_rgb`. These become test assertions.

**Deliverables:** blast-radius report to user; `kip.*` key inventory; measured parameter baseline. → **all recorded in Appendix A**.

---

### Phase 1 — Parameter-free primitives (ECMR + rank map)

**Goal:** Phases 3b and 3c as pure functions with zero parameters, correct under padding, before anything is wired.

**New file: `core/kip/ecmr.py`**

- [x] **T1.1** `ego_compensated_residual(eo, mask, lam) -> tuple[Tensor, Tensor]` returning `(m (B,L), mu_norm (B,L))`.
  - Causal EMA `μ_t = Σ_{τ≤t}(1−λ)λ^{t−τ}·ê_{O,τ}`, `λ = 0.9`.
  - **Implementation choice: sequential recurrence over `L`** (`mu = lam*mu + (1-lam)*eo_t`), not the closed form. The closed form needs `λ^{-τ}`, which overflows float32 by `t ≈ 160`; `L` reaches 1536. Cost is `L` vectorized ops over `(B, d_O)` — negligible next to the transformer.
  - **Padding rule:** at a padded position the state must be *held*, not updated with zeros — otherwise the prototype decays toward zero inside padding and the first real position after a gap gets a fake residual. Use `torch.where(mask_t, new_state, prev_state)`.
  - **Warm-up rule:** at `t = 0`, `μ_0 = ê_{O,0}` (⇒ `m_0 = 0`), not `μ_0 = 0`. Initialising at zero makes the first frame the loudest position of every clip, which the rank map would then hand `s_0 = 128` — a boundary position that zero-pads its past neighbour. Bias, not noise.
  - `m_t = ‖δ_t‖₂`, forced to 0 at padded positions.
- [x] **T1.2** `rank_map(m, mask) -> Tensor` — Phase 3c.
  - `r_t = rank_t(m) / (n_valid − 1)` using **`n_valid`, not `L`**. With padding, `L−1` would compress the rank range by the padding fraction and make `s_t` depend on batch composition — an arm's score would change with its batch shuffle.
  - Ties: **ordinal ranking via double `argsort`**, deterministic and `torch.compile`-safe. Document that ties break by index order; on a constant `m` (a genuinely static clip) this still spans [0,1], which is the intended "rank map always spans its range" property but *is* arbitrary. Log a warning when `m.std() < eps` on a clip.
  - `n_valid == 1` ⇒ `r = 0`, not a divide-by-zero.
- [x] **T1.3** `shift_counts_from_ratio(r, max_shift, mask) -> Tensor` — `⌊r·D/K⌋`, clamped to `[0, max_shift]`, zeroed at padded positions. Shared by all four gate types so the floor/clamp semantics can't drift apart.
- [x] **T1.4** Constants into `core/constants.py`: `ECMR_EMA_LAMBDA = 0.9`, `RANK_TIE_EPS`. No magic numbers in the module.
- [x] **T1.5** Tests in a new `core/tests/test_kip_gate_v3.py`:
  - EMA against a hand-rolled numpy oracle, `L ∈ {1, 2, 7, 33}`.
  - `μ_t ≈ 0` residual identity: on a **constant** `ê_O` sequence, `m_t = 0` for all `t` — the static-camera degradation the doc claims.
  - **Mask invariance:** padding a batch to a longer `L` must not change `m`, `r`, or `s` at any valid position. This is the single highest-value test in the plan.
  - Rank output is a permutation of `{0…n−1}/(n−1)`; `s ∈ [0, 128]`; `min(s) = 0` and `max(s) = 128` on every clip with `n_valid ≥ 2` (the "rules out H4′ by construction" claim, asserted rather than believed).
  - Zero parameters: `sum(p.numel() for p in module.parameters()) == 0`.

**Deliverables:** `core/kip/ecmr.py`; ~15 tests; measured 0-parameter claim. → **DONE 2026-08-30, see Appendix B.**

---

### Phase 2 — Gate dispatch + config surface

**Goal:** four gate types, one shift implementation, safe defaults.

- [x] **T2.1** Add gate-type constants to `core/kip/gate_shift.py`: `GATE_TYPE_RANK` (v3 default), `GATE_TYPE_MLP_FROZEN` (v1, bit-identical to today), `GATE_TYPE_MLP_STE` (spec §4.2 variant), `GATE_TYPE_CONSTANT` (plain-TSM control).
- [x] **T2.2** Refactor `KinematicShift.compute_shift_counts` into a dispatch:
  - `rank` → `m` from `ecmr.ego_compensated_residual` → `rank_map` → `shift_counts_from_ratio`. `self.mlp` is **not constructed**.
  - `mlp_frozen` → today's path verbatim: `_intensity` → mask-aware min-max → `sigmoid(MLP(·))` → floor. Must stay **bit-identical**; guarded by a regression test (T2.6).
  - `mlp_ste` → forward bit-identical to `mlp_frozen`, backward unblocked via `u + (⌊u⌋ − u).detach()`. Note in the docstring: this also opens `L_MIL` → PMG, which Pi-VAD's unweighted `L_PMG` convention exists to prevent (spec §4.2). Ablation only.
  - `constant` → `r_t = const_ratio` for all `t`, `ê_O` ignored entirely.
  - Keep `gate_signal` (`flow_norm` / `feat_var`) meaningful for the MLP types only; **raise** if it is set alongside `gate_type="rank"` rather than silently ignoring it.
- [x] **T2.3** `KIPConfig` (`core/config.py:48`) gains: `gate_type: str = "rank"` (v3-faithful; the `v3` branch holds no v1 code path as a default), `ecmr_lambda: float = constants.ECMR_EMA_LAMBDA`, `const_shift_ratio: float = 0.5`, `disable_pmg: bool = False` (true plain-TSM control — no PMG, no flow requirement). Validate the enum in `__post_init__` and fail on an unknown value.
- [x] **T2.4** **Ambiguous-config guard (the C14 guard), revised.** A loaded config dict that has a `kip` section with `enabled: true` but **no `gate_type` key** predates this change and is genuinely ambiguous — it could only ever have meant `mlp_frozen`, but this branch is v3-faithful and holds no v1 default. Resolving it silently either way is the C14 defect. So `load_config` **raises**, with an actionable message: *"this config predates `kip.gate_type`; it ran the frozen-MLP gate. Add `gate_type: mlp_frozen` to reproduce it, or `gate_type: rank` to run v3 — they are different models."*
  - A fresh `KIPConfig()` in code defaults to `rank` and never raises. Only *deserialization of an ambiguous file* raises.
  - Consistent with C5 (fail loud on artifact/model mismatch) rather than inventing a legacy default on a branch that has no legacy.
  - Log the resolved gate type at INFO on every model construction, always, regardless of source.
- [x] **T2.5** Thread `disable_pmg` into `require_flow` at `core/train.py:593` → `cfg.kip.enabled and not cfg.kip.disable_pmg`. **Do not** touch the `require_flow=False` path — spec §7.1 is explicit that zero-filling `e_O` is the trap.
- [x] **T2.6** Tests:
  - **Bit-identity regression:** fixed seed, fixed input → `mlp_frozen` counts equal the counts produced by the pre-change `compute_shift_counts`. Pin the expected tensor as a literal fixture so the guard survives future refactors.
  - `mlp_ste` forward equals `mlp_frozen` forward exactly; and `grad` reaches `shift.mlp.*` under STE while it is `None` under `mlp_frozen`. (The latter asserts the known defect stays known.)
  - Under `gate_type="rank"`, backprop of a task-only loss leaves `kip.pmg.*` grads at `None` — the **0-parameter score path** claim, and the reason stage-1 warm-up becomes mandatory rather than optional.
  - Config round-trip: fresh dataclass → `rank`; dict with `kip.enabled: true` and no `gate_type` → **raises** with the T2.4 message; explicit `gate_type` of each of the four values round-trips; unknown value → raises; `kip.enabled: false` and no `gate_type` → does *not* raise (no gate exists to be ambiguous about).

**Deliverables:** four working gate types; config surface; ~12 tests; the C14 guard. → **DONE 2026-08-30, see Appendix C.**

---

### Phase 3 — Train-only submodules off the inference graph

**Goal:** honour v3's "`mhead` not instantiated" and "projections not instantiated" at inference *without* weakening `load_state_dict` (C5).

**The concrete problem.** `core/inference.py:64` calls `model.load_state_dict(state)` — strict. A KIP-on checkpoint contains `kip.mhead.*`, `kip.proj_flow.*`, `kip.proj_rgb.*`, and (for old arms) `kip.shift.mlp.*`. Drop those modules from the inference graph and strict loading fails on unexpected keys.

- [x] **T3.1** `KIP.__init__` gains `train_only_modules: bool = True`. When `False`, `mhead`, `proj_flow`, `proj_rgb` are not constructed and `forward` returns `motion_scores = None`.
- [x] **T3.2** `KIP.from_config` sets it from a new `training: bool` argument threaded from `KATVAD.from_config`, so `load_model_for_scoring` builds the inference graph and `train.py` builds the full one.
- [x] **T3.3** Add `core/models/ckpt_compat.load_kip_state_dict(model, state)` — an **explicit allowlist** of train-only key prefixes (`kip.mhead.`, `kip.proj_flow.`, `kip.proj_rgb.`, `kip.shift.mlp.`) that may be present in a checkpoint and absent from an inference-graph model. Every other mismatch still raises, and the function **logs the count of each prefix it dropped**. Never `strict=False`.
- [x] **T3.4** Gate-type mismatch must be **loud, not silent**. If the checkpoint has `kip.shift.mlp.*` and `cfg.kip.gate_type == "rank"`, raise with an actionable message: *"this checkpoint was trained with the frozen-MLP gate; score it with `--gate-type mlp_frozen` or it is a different model."* Silently scoring an mlp_frozen checkpoint under a rank gate is exactly the C13/C14 failure mode — a plausible number from the wrong model.
- [x] **T3.5** Add `--gate-type` to `core/inference.py`'s argparse (§14.6: every entry point ships a CLI flag for what it can vary).
- [x] **T3.6** Tests:
  - Round-trip: train-graph state dict loads into an inference-graph model; the allowlisted prefixes are reported dropped; an unrelated extra key still raises.
  - The gate-type mismatch raises with the expected message.
  - Inference-graph parameter count equals **311,808** under `gate_type="rank"` (measured, per P0.5), and the score-path count is 0.

**Deliverables:** inference graph matching v3's table; old checkpoints loadable under their own gate type; ~8 tests. → **DONE 2026-08-30, see Appendix D.**

---

### Phase 4 — Diagnostics plumbing

**Goal:** make ablation-ladder rows 1 and 2 runnable with **zero training**, which is the whole point of v3 CHANGE B.

- [x] **T4.1** Introduce `KIPOutput(NamedTuple)` with `vk, eo_hat, motion_scores, diagnostics`. Replaces the bare 3-tuple. One production call site (`core/models/kat_vad.py:180`) plus tests — small and typed, which a stashed `self._last_*` attribute would not be.
- [x] **T4.2** `diagnostics` carries `s` (int, `(B,L)`), `m` (`(B,L)`), `mu_norm` (`(B,L)`), `eo_norm` (`(B,L)`), all detached. Populated for every gate type so the four are directly comparable.
- [x] **T4.3** `KATVAD.forward` surfaces them under a `kip_diag` key, off by default via a `return_kip_diagnostics: bool = False` argument — no cost on the training path.
- [x] **T4.4** `core/inference.py` gains `--dump-kip-diag`, writing the per-clip arrays into the existing `.npz` alongside `scores`. Reuses `score_to_npz` (`core/inference.py:157`); no new artifact format, no new directory layout. **Decided 2026-08-30: defaults ON for `inference.py`, OFF for training** — the eval path is where ablation rows 1–2 read it, and the training path should carry no diagnostic cost. Pair with `--no-dump-kip-diag`.
- [x] **T4.5** Tests: diagnostics shapes and dtypes; padded positions are zero; `s` under `constant` is genuinely constant; `s` under `rank` spans `[0,128]`.

**Deliverables:** `s_t` observable end to end; ablation rows 1 and 2 become analysis, not training. → **DONE 2026-08-30, see Appendix E.**

---

### Phase 5 — Docs, memory bank, lesson, quality gates

- [x] **T5.1** Fix `CLAUDE.md` §14.1: the `core/docs/v2/` rows point at files that do not exist; repoint to `core/docs/v3/`. Add the three v3 docs to the table.
- [x] **T5.2** Add a **deviation note** to `core/docs/v3/KAT-VAD-ARCHITECTURE.md` Phase 1 and `KAT-VAD_spec_v3.md` §0: *the encoder is stock frozen CLIP ViT-B/16 at the pinned revision; Alert-CLIP is specified but no public checkpoint exists, so the swap is deferred and every v3 number is a stock-CLIP number.* Same note in `core/docs/TRAINING.md` deviations.
- [x] **T5.3** Update `core/docs/TRAINING.md` with the gate-type matrix and the stage-1 consequence: under `gate_type="rank"` the PMG head receives **no task gradient at all**, so a run without a converged stage-1 warm-up ranks noise. State it as a precondition, not a footnote.
- [x] **T5.4** Add a lesson (CLAUDE.md §7 — mandatory after a feature). Candidate, subject to the 5 gates in `GATES.md`: *[HIGH] Architecture — a non-differentiable op on the score path silently freezes everything upstream of it.* Triggers: `floor`, `argsort`, integer slicing, gate, shift, index. Bad: `s = (ratio*max).floor().long()` with a trainable MLP behind it. Good: assert `grad is not None` on every module claimed to be learned, in a test. Files: `core/kip/gate_shift.py:112`. Check `index.md` for a duplicate first — C14 is adjacent but is about *experiments*, not *architecture*.
- [x] **T5.5** Update `.project/memory-bank/activeContext.md` and `systemPatterns.md`: v3 Phase 3 landed, four gate types, the default flip and its C14 consequence, encoder deviation.
- [~] **T5.6** *(gates green + `detect_changes` clean; commit awaiting user go-ahead)* Quality gate: `source .venv/bin/activate && ruff check * && mypy * && bandit * && pycycle * && pyright *`, then the full 322+ test suite, then `detect_changes` before commit. Split into ≤20-file commits, added individually.

**Deliverables:** docs consistent with code; one lesson; memory bank current; green gates. → **DONE 2026-08-30, see Appendix F.**

---

## 5b. Appendix A — Phase 0 findings (measured 2026-08-30)

### A.1 Blast radius (P0.1) — LOW

`trace_path` on the `v3` branch index (1,617 nodes / 4,071 edges). The graph cannot see `nn.Module.__call__`, so production call sites were confirmed by text search; that limitation is recorded here rather than papered over.

| Symbol | Production callers | Test callers |
|---|---|---|
| `KinematicShift` | `KIP.__init__` (`core/kip/kip_module.py:44`) — **1** | 5 in `test_kip_modules.py` |
| `KIP` | `KATVAD.from_config` (`core/models/kat_vad.py:111`) — **1** | 6 in `test_kip_modules.py` |
| `KIP.forward` 3-tuple unpack | `core/models/kat_vad.py:180` — **1 site only** | `test_kip_modules.py`, `test_e2e_synthetic.py` |
| `MotionScoreHead` | `KIP.__init__` (`core/kip/kip_module.py:52`) — **1** | 1 |

**Verdict: LOW impact, no HIGH/CRITICAL warning to escalate.** The `KIPOutput` NamedTuple change (T4.1) touches exactly **one** production unpack site. `core/train.py` reads `outputs["eo_hat"] / ["vt"] / ["motion_scores"]` from the dict (`train.py:238-241, 346-349, 360`), never the tuple, so it is unaffected by the return-type change.

Ten test files mention `kip`; the directly affected one is `core/tests/test_kip_modules.py` (24 tests). Two existing tests are load-bearing for this plan and must keep passing unchanged:
- `TestKIP.test_gate_mlp_receives_no_gradient_spec_as_written` — already pins the frozen-gate defect.
- `TestKIP.test_gradients_reach_pmg_and_motion_head` — already pins that PMG *is* reached via `mhead`, i.e. the §6 correction was already encoded in the test suite.

### A.2 `kip.*` key inventory (P0.2) — the compat contract for T3.3

No checkpoint exists on local disk (`outputs/` holds configs, metrics and `.npz` only; checkpoints are on Drive). Inventory derived from `KIP().state_dict()` — the same object that writes them — which is equivalent for the compat contract.

```
kip.pmg.enc.weight          (128, 512, 3)     ┐
kip.pmg.enc.bias            (128,)            │
kip.pmg.translator.weight   (128, 128)        │ inference path — always present
kip.pmg.translator.bias     (128,)            │
kip.pmg.dec.weight          (256, 128, 3)     │
kip.pmg.dec.bias            (256,)            ┘
kip.shift.mlp.0.weight      (16, 1)           ┐
kip.shift.mlp.0.bias        (16,)             │
kip.shift.mlp.2.weight      (16, 16)          │ gate — ABSENT under gate_type=rank
kip.shift.mlp.2.bias        (16,)             │ presence ⇒ ckpt is an mlp_* arm (T3.4)
kip.shift.mlp.4.weight      (1, 16)           │
kip.shift.mlp.4.bias        (1,)              ┘
kip.mhead.net.0.weight      (128, 256)        ┐
kip.mhead.net.0.bias        (128,)            │
kip.mhead.net.2.weight      (1, 128)          │ train-only — allowlisted drops (T3.3)
kip.mhead.net.2.bias        (1,)              │
kip.proj_flow.weight        (128, 256)        │
kip.proj_flow.bias          (128,)            │
kip.proj_rgb.weight         (128, 512)        │
kip.proj_rgb.bias           (128,)            ┘
```

T3.3's allowlist is therefore exactly: `kip.mhead.`, `kip.proj_flow.`, `kip.proj_rgb.`, `kip.shift.mlp.`.

### A.3 Measured parameter baseline (P0.5) — every doc number confirmed

| Component | Measured | v3 doc claim | Match |
|---|---:|---:|:---:|
| `kip.pmg` | 311,808 | 311,808 | ✅ |
| `kip.shift.mlp` | 321 | 321 | ✅ |
| `kip.mhead` | 33,025 | 33,025 | ✅ |
| `kip.proj_flow` + `proj_rgb` | 32,896 + 65,664 = 98,560 | 98,560 | ✅ |
| **KIP total (v1, train)** | **443,714** | 443,714 | ✅ |
| KIP total (v3 rank gate, train) | 443,714 − 321 = **443,393** | 443,393 | ✅ |
| Inference path under rank gate | **311,808** | 311,808 | ✅ |
| Score path under rank gate | **0** | 0 | ✅ |

These become literal assertions in T3.6.

### A.4 Config-default risk (P0.3) — CONFIRMED, guard is necessary

`load_config` (`core/config.py:180-193`) applies each YAML section onto dataclass defaults via `_apply_section`; an absent key silently keeps the default. **22 run configs** under `outputs/*/*/config.yaml` carry a `kip:` section with no `gate_type`. Under a bare `rank` default, re-running any of them would silently produce a different model with a byte-identical-looking config — the C14 defect. **T2.4's raise is required, not optional.**

### A.5 P0.4 — BLOCKED, not skipped

Reading RefineVAD's MoTAR implementation for a straight-through relaxation could not be done in this environment: `RefineVAD_AAAI_2026.pdf` is present, but no PDF text extractor is available (`pypdf`, `pdfminer`, `PyPDF2`, `pdfplumber`, `fitz` all absent; `pdftotext`/`mutool`/`qpdf` not installed) and `pip install` has no network.

**Unblock with either:** `brew install poppler` (enables PDF reads), or `source .venv/bin/activate && pip install pypdf` from a networked shell.

**Impact: none on Phases 1–4.** It affects one documentation line (does the frozen gate reproduce a defect in the source paper, or is it ours?) and is carried into T5.4's lesson text. Phase 5 is the deadline, not Phase 1.

---

## 5c. Appendix B — Phase 1 results (2026-08-30)

**Shipped:** `core/kip/ecmr.py` (186 LOC, 3 public functions, **0 parameters**) and `core/tests/test_kip_gate_v3.py` (**31 tests**, all green first run). Constants added to `core/constants.py`: `ECMR_EMA_LAMBDA = 0.9`, `RANK_TIE_EPS = 1e-6`, `CONST_SHIFT_RATIO = 0.5` (pre-placed for T2.3).

**Suite: 322 → 353 passed, exit 0, no regressions.** Gates on the new files: `ruff` clean, `mypy` clean, `pyright` 0 errors, `bandit` clean.

### B.1 Decisions taken while implementing

| Decision | Why |
|---|---|
| First **valid** position seeds `μ`, tracked with a `started` flag rather than assuming `mask[:, 0] == 1` | Masks are left-aligned today, but the flag costs one bool tensor and removes a silent dependency on that convention. |
| Padded positions **hold** the EMA state via `torch.where`, never update it | Updating through padding decays `μ` toward zero, so the first real position after a gap gets a fabricated residual. |
| `argsort(..., stable=True)` on **both** passes | `torch.argsort` is *not* stable by default. Without this, tied residuals rank non-deterministically and two identical runs could diverge. Not in the plan; found while writing T1.2. |
| Padded positions sorted to `+inf` before ranking | Keeps valid positions on ranks `0…n_valid−1` with no gather/scatter. |
| `lam` validated to `[0, 1)` with a raise | A `lam` of exactly 1.0 freezes `μ` at the first sample and silently turns ECMR into "distance from frame 0". |
| Warning is per-batch at WARNING level | A genuinely constant residual means the shift schedule is arbitrary. Rare enough not to spam; loud enough to notice. |

### B.2 Identity confirmed by test, worth knowing

Because `μ_t` includes the current sample, the residual is **exactly** `δ_t = λ·(ê_{O,t} − μ_{t−1})`. Pinned by `test_residual_equals_lambda_times_previous_deviation`. The constant `λ` factor is immaterial — `rank_map` consumes only the ordering — but it means the residual measures deviation from the *previous* prototype, which is the intended "motion the clip's recent history does not explain" reading.

### B.3 Claims now measured rather than asserted

- **Static-camera degradation** (`test_constant_flow_gives_zero_residual`): a constant `ê_O` gives `m ≡ 0` exactly. The doc's "on a fixed camera `μ_t ≈ 0` and the residual reduces to raw magnitude" is *not* what the operator does — it reduces to **zero**, not to raw magnitude. The behaviour is still correct and graceful (a clip with no motion variation gets an arbitrary-but-harmless schedule), but the architecture doc's sentence is wrong and is corrected in Phase 5.
- **Scale-free** (`test_is_scale_free`): scaling `ê_O` by 1000× leaves `r` bit-identical.
- **Full-range span** (`test_spans_full_range_on_every_clip`): `min(s) = 0`, `max(s) = 128` on every clip with `n_valid ≥ 2`, including `n_valid = 2`. This is the H4′-by-construction claim, now a test rather than a paragraph.
- **Mask invariance** (`TestMaskInvariance`, 4 cases): a clip's `m`, `μ`, `r` and `s` are bit-identical whether it is scored alone or beside neighbours at three different padding widths.

### B.4 Measured cost of the sequential recurrence

`B=4, L=512, d=256` → **8.0 ms** per forward on CPU. Logged by `TestPerformance` on every run rather than assumed, with a generous 2 s regression ceiling.

Honest read: the plan called this "negligible next to the transformer". At `L = 512` it is small but **not** negligible on GPU, where `L` sequential kernel launches dominate. Typical `L` is far below the 1536 cap (stride-8 sampling; DoTA clips land in the tens), so this is not a Phase 1 problem. If a long-clip corpus makes it one, the fix is a causal `conv1d` with a truncated exponential kernel — `λ^k < 1e-7` by `k ≈ 153`, so a 160-tap kernel is exact to float32. **Not done now:** it complicates the seed and mask-hold rules, and optimising before there is a measurement showing it matters is the wrong order.

---

## 5d. Appendix C — Phase 2 results (2026-08-30)

**Shipped:** `core/kip/gate_shift.py` rebuilt around a 4-way gate dispatch; `KIPConfig` gained `gate_type`, `gate_signal: str | None`, `ecmr_lambda`, `const_shift_ratio`, `disable_pmg` plus a `validate()` called from both `__post_init__` and `load_config`; `load_config` raises on an ambiguous pre-v3 config; `require_flow` honours `disable_pmg`; `core/tests/test_kip_gate_types.py` (**35 tests**).

**Suite: 353 → 388 passed, exit 0.** `ruff` clean, `mypy` clean across **71** source files, `pyright` 0 errors, `bandit` clean, `pycycle` clean.

### C.1 THE HEADLINE — H4′ is confirmed, and it cost zero training

Ablation-ladder row 1 ("log `s_t`; decides H4′") is **answered**, because the gate MLP never trains, so its random init *is* the deployed function, and its input is min-max normalized, so `[0, 1]` is the *entire reachable input domain*.

Sweeping that whole domain under seed 2024, `m_hat: 0 → 1` maps to `ratio ∈ [0.4991, 0.5006]` — so `s_t ∈ {63, 64}`. **A span of 1 channel out of 128.**

Across 8 seeds, on `B=1, L=200`:

| seed | min | max | **span** | mean | std |
|---|---:|---:|---:|---:|---:|
| 2024 | 63 | 64 | **1** | 63.58 | 0.494 |
| 2025 | 68 | 71 | **3** | 69.42 | 0.711 |
| 2026 | 63 | 66 | **3** | 64.43 | 0.676 |
| 0 | 68 | 68 | **0** | 68.00 | 0.000 |
| 1 | 61 | 63 | **2** | 61.67 | 0.511 |
| 7 | 60 | 64 | **4** | 62.26 | 0.758 |
| 42 | 58 | 59 | **1** | 58.35 | 0.477 |
| 1234 | 65 | 66 | **1** | 65.02 | 0.140 |

Seed 0 gives a span of **exactly zero** — a literally constant smoother. Seeds 2024/2025/2026 are the three MSAD arms' seeds; their spans are 1, 3 and 3 channels out of 128.

**Mechanism:** the gate is `Linear(1,16) → GELU → Linear(16,16) → GELU → Linear(16,1)` at PyTorch default init. On a scalar input the composed map has an output magnitude of order `1e-3`, so `σ(·) ≈ 0.5` and `⌊0.5 × 128⌋ = 64` regardless of input.

Side-by-side at `L=200`, seed 2024:

| gate type | min | max | span | mean |
|---|---:|---:|---:|---:|
| `rank` (v3) | 0 | 128 | **128** | 63.5 |
| `mlp_frozen` (v1) | 63 | 64 | **1** | 63.5 |
| `constant` @ r=0.5 | 64 | 64 | **0** | 64.0 |

`mlp_frozen` and `constant` agree to within half a channel on the mean. **v1's "motion-gated adaptive temporal shift" was, in operation, a fixed ~50 % temporal shift.**

**What this changes.**
1. H4′ moves from *open hypothesis* to **confirmed**. It should no longer be listed as untested in `activeContext.md` or spec §4.1.
2. The **plain-TSM control (ablation 4) is now the decisive experiment**, and it has a sharp pre-registered prediction: `constant` at `r ≈ 0.5` should reproduce close to the full **+0.0915**. If it does, the +0.09 is a temporal-smoothing result, not a motion result, and `L_KIP_rec` matters only insofar as it keeps the trunk honest. That experiment is now one config flag.
3. The rank gate is a **genuine** change, not cosmetic: span 128 vs span ~1 on the same inputs.
4. **Caveat, stated plainly.** Measured at random init on random `ê_O`. Init *is* the deployed function (the MLP never trains) and min-max *does* guarantee the input domain, so both legs are solid — but a real checkpoint's `ê_O` on real features has not been swept. Phase 4's `--dump-kip-diag` closes that on existing checkpoints with no training.

### C.2 The spec's STE formula is a no-op — corrected in implementation

Spec §4.2 and plan T2.2 both specify `s_t = u_t + (⌊u_t⌋ − u_t).detach()` as the STE variant, described as "a pure gradient intervention".

**It does not work.** The formula does give `s_t` a gradient, but `s_t` is consumed only inside `channel < s` comparisons, and a comparison emits a boolean that no gradient flows through. Shipped as written, `mlp_ste` would have been indistinguishable from `mlp_frozen` — an ablation that silently measures nothing.

The fix applies the straight-through estimator to the channel **selection weights** instead (`shift_channels_straight_through`): a hard partition `{past, future, present}` in the forward pass with a smooth sigmoid surrogate in the backward pass, all three weights summing to 1 in both directions. Verified:

- **Forward is bit-identical to `mlp_frozen`** — `torch.equal`, max diff exactly `0.0`.
- Gradient now reaches `shift.mlp.*` **and** `ê_O` (both non-zero).

Cost: the STE path materialises three `(B, L, D)` weight tensors instead of two `torch.where` calls, which is why it stays an ablation rather than a default.

### C.3 Bit-identity of `mlp_frozen`, pinned as a literal

Captured from the pre-refactor code under `torch.manual_seed(1234)` **before** the file was touched, then asserted after: shift counts, `feat_var` counts, and the output checksum all match exactly. This is the C14 guard — every number in `RESULTS_*.md` remains reproducible by naming `gate_type: mlp_frozen`.

### C.4 Gradient topology — confirmed unchanged, as you said

Measured for all three hard gates (`rank`, `mlp_frozen`, `constant`): `grad → v^t` **present**, `grad → ê_O` **absent**, and for `mlp_frozen` every gate-MLP tensor is `None`. The rank gate deletes 321 dead parameters and changes no edge. Now a parametrized regression test.

### C.5 Judgement calls made while implementing

| Call | Why |
|---|---|
| `gate_signal` becomes `str \| None`, and is **rejected** (not ignored) for `rank`/`constant` | A silently-ignored key lets a config claim a gate signal that nothing read. Only possible because T2.4 forces every pre-v3 config to be edited anyway. |
| A file with **no** `kip:` section does *not* raise | It makes no claim about the gate; the ambiguity only exists in a file that configures KIP but omits `gate_type`. |
| `KIPConfig.validate()` called from `load_config`, not just `__post_init__` | `_apply_section` and the override loop both `setattr` onto a constructed dataclass, so `__post_init__` never re-runs. Without this, `--override kip.gate_type=bogus` would pass silently. |
| Three existing tests in `test_kip_modules.py` now name `mlp_frozen` explicitly | They pin v1 behaviour; under the new default they were testing a gate that owns no MLP. Intent preserved, target made explicit. |
| A test asserts **every archived run config on disk raises** | Turns the 22-config C14 exposure into a standing regression guard rather than a one-time check. |

---

## 5e. Appendix D — Phase 3 results (2026-08-30)

**Shipped:** `KIP.train_only_modules` (3e/3f droppable); `training` threaded through `KIP.from_config` → `KATVAD.from_config` → `core/inference.py`; `ckpt_compat.load_kip_state_dict` with an explicit allowlist and a gate-type mismatch guard; `--gate-type` on the inference CLI; `core/tests/test_kip_inference_graph.py` (**20 tests**).

**Suite: 388 → 408 passed, exit 0.** `ruff` clean, `mypy` clean across **72** source files, `pyright` 0 errors, `bandit` clean, `pycycle` clean.

### D.1 The inference graph now matches v3 §9 exactly

| | Training graph | Inference graph |
|---|---:|---:|
| `kip.pmg` (3a) | 311,808 | 311,808 |
| `kip.shift` (3b/3c/3d, rank) | 0 | 0 |
| `kip.mhead` (3e) | 33,025 | **not built** |
| `kip.proj_flow` + `proj_rgb` (3f) | 98,560 | **not built** |
| **total** | **443,393** | **311,808** |

`forward` returns `ŷ_O = None` on the inference graph, and `project_for_align` raises there rather than returning something meaningless. `test_scores_are_identical_across_graphs` asserts the anomaly curve is **bit-identical** across the two graphs — dropping 3e/3f must not move the score, because they were never on the score path.

### D.2 The compat loader is an allowlist, never `strict=False`

`load_kip_state_dict` drops exactly 8 tensors (`kip.mhead.` ×4, `kip.proj_flow.` ×2, `kip.proj_rgb.` ×2), logs the per-prefix counts, and raises on everything else. Tested: an unrelated extra key still raises; a missing required key still raises; train→train drops nothing; a `kip.enabled=false` model round-trips. `test_weights_actually_transfer` guards the failure where a drop-list quietly becomes a skip-everything.

### D.3 Gate-type mismatch is loud in both directions

- `mlp_frozen` checkpoint → `rank` model: raises, and the message names the remedy (`--gate-type mlp_frozen`), asserted by test.
- `rank` checkpoint → `mlp_frozen` model: raises (checkpoint carries no `kip.shift.mlp.*`).

**Documented limitation, with a test that says so.** `rank` and `constant` share an identical key layout — both are parameter-free — so keys alone *cannot* separate them, and `test_parameter_free_gates_are_interchangeable_by_keys_alone` pins that fact rather than leaving it to be discovered. The guard catches the mlp/non-mlp boundary only. Closing the rank-vs-constant gap needs the **run manifest** (lesson C17), which is deferred to the training-pipeline plan (§7 item 1). Worth weighting: the plain-TSM control (C.1) is `constant`, so this is exactly the comparison where the gap bites.

### D.4 Two things fixed during implementation

- **A nonsense ternary I wrote:** `cfg.kip.gate_signal = None if ... else None` — both branches identical. Replaced with the real intent: clear `gate_signal` only when the CLI switches to a gate that consumes no intensity signal, so validation does not fail on a key the override just made irrelevant.
- **`--dump-kip-diag` was added to the parser and then removed again.** It belongs to T4.4, where it is actually consumed; a flag that parses and does nothing is worse than an absent one. It lands in Phase 4.

### D.5 Optional-narrowing churn in existing tests

Making `mhead`/`proj_flow`/`proj_rgb` optional turned three existing assertions in `test_kip_modules.py` into `mypy` union-attr errors. Fixed with explicit `assert ... is not None` narrowing rather than `type: ignore`, which also documents at the call site that those tests exercise the **training** graph.

---

## 5f. Appendix E — Phase 4 results (2026-08-30)

**Shipped:** `KIPOutput` NamedTuple (`vk`, `eo_hat`, `motion_scores`, `diagnostics`); `KIP.forward(..., return_diagnostics=False)`; `KATVAD.forward(..., return_kip_diagnostics=False)` surfacing `kip_diag/*`; `sliding_window_scores` returning per-clip diagnostics across windows; `score_to_npz` writing them as `kip_*` keys; `--dump-kip-diag` / `--no-dump-kip-diag` on **both** `inference.py` and `evaluate.py`; `core/tests/test_kip_diagnostics.py` (**26 tests**).

**Suite: 408 → 434 passed, exit 0.** `ruff` clean, `mypy` clean across **73** source files, `pyright` 0 errors, `bandit` clean, `pycycle` clean.

### E.1 Scope extension, declared: `evaluate.py` also got the flag

T4.4 named only `core/inference.py`. **I wired `core/evaluate.py` too**, because `evaluate.py` is what writes `outputs/*/scores/*.npz` — the 21,281 files ablation-ladder rows 1-2 are supposed to read. Restricting the flag to `inference.py` would have satisfied the task text while leaving the deliverable unable to serve its stated purpose.

Guarded so it cannot surprise: `dump_diag = args.dump_kip_diag and args.save_scores and cfg.kip.enabled`. No `--save-scores`, no files, nothing collected.

### E.2 The decision, as implemented

| Path | Default | Mechanism |
|---|---|---|
| `core/inference.py` | **ON** | `--dump-kip-diag` / `--no-dump-kip-diag` |
| `core/evaluate.py` | **ON** (needs `--save-scores`) | same flag |
| `core/train.py` | **never** | no flag, no call — asserted by `test_training_never_collects_diagnostics`, which greps the module for both identifiers |

That last test is deliberately blunt: it fails if anyone later threads diagnostics into the training loop, which is the specific thing the decision ruled out.

### E.3 What lands in each `.npz`

Five arrays per clip, alongside the existing `score` / `sim` / `class_names` / `gt`:

| key | dtype | meaning |
|---|---|---|
| `kip_s` | **int16** | applied shift count, bounded by `D/K = 128` |
| `kip_gate_ratio` | float32 | `r_t` before the floor |
| `kip_m` | float32 | ECMR residual magnitude (rank gate) or raw intensity |
| `kip_mu_norm` | float32 | `‖μ_t‖`, the prototype; zero for non-ECMR gates |
| `kip_eo_norm` | float32 | `‖ê_O‖`, for the raw-vs-residual comparison |

No new artifact format and no new directory: `rescore`, `evaluate` and every existing reader are untouched, asserted by `test_npz_without_diagnostics_is_unchanged` (`{score, sim, class_names}` exactly).

### E.4 Two correctness guards worth naming

- **`test_reported_s_is_the_s_that_was_applied`** — recomputes the shift counts from the returned `ê_O` and asserts equality with the logged `s`. A diagnostic that silently disagrees with the forward pass is worse than no diagnostic, and this is a second gate evaluation, so the risk is real.
- **`test_scores_unchanged_when_diagnostics_requested`** / **`test_requesting_diagnostics_does_not_change_outputs`** — turning logging on must not move the curve.

### E.5 A windowing caveat that affects how the data is read

`sliding_window_scores` runs `max_vis_len`-sized windows, and the rank gate ranks **within a window**. So for a clip longer than `max_vis_len`, `s_t` reflects a per-window rank, not a global one. This is not a logging artifact — it is exactly what the scored model computes — but anyone plotting `s_t` against an anomaly window must know the ranking resets at each window boundary. Documented in the `sliding_window_scores` docstring.

### E.6 Blast radius of `KIPOutput`, as predicted

P0.1 predicted one production unpack site. Correct: `core/models/kat_vad.py` only. Everything else that broke was test code (7 tests across 3 files) plus `sliding_window_scores`'s own signature change, which rippled to `core/evaluate.py` and 2 tests. All compile-time, none silent.

---

## 5g. Appendix F — Phase 5 results (2026-08-30)

**Full quality gate, all green:** `ruff` clean · `mypy` **73 source files** clean · `pyright` 0 errors · `bandit` clean · `pycycle` clean · **`pytest` 434 passed**.

**`detect_changes`: 13 changed files, zero unexpected.** Impact confined to `core/kip/`, `core/models/`, `core/config.py`, `core/constants.py`, `core/inference.py`, `core/evaluate.py`, one line of `core/train.py`, plus tests and `TRAINING.md`.

### F.1 A second stale-reference class found in `CLAUDE.md`

T5.1 was scoped to repointing the `core/docs/v2/` rows. While doing it, **`.project/plans/katvad-v2-next-steps.md` turned out not to exist either** — `CLAUDE.md` cited it twice, including as "**The live plan.** Start here for 'what next'". The `.project/plans/` directory holds only `kat-vad-implementation.md`, `msad-ncc-seeds-and-selection.md`, and this plan.

So `CLAUDE.md` §14.1 was directing every session to two source-of-truth documents *and* a plan file, none of which are on disk. Both classes are now marked with an explicit "does not exist in this tree — do not cite" row rather than silently deleted, so the next reader who remembers them learns why they are gone.

### F.2 Docs corrected, with the reason recorded in place

| Doc | Correction |
|---|---|
| `v3/KAT-VAD-ARCHITECTURE.md` P1 | Alert-CLIP → **stock CLIP** deviation box; the "eyes tuned to abnormality" line in the mental model flagged aspirational |
| `v3/KAT-VAD-ARCHITECTURE.md` 3b | **The fixed-camera claim was wrong.** "residual reduces to the raw magnitude" → reduces to **exactly zero**; corrected with the test name |
| `v3/KAT-VAD-ARCHITECTURE.md` 3c | Measured H4′ box: why a learned MLP was replaced, with the 8-seed span table |
| `v3/KAT-VAD-ARCHITECTURE.md` 3d | Four gate types; the no-gradient-to-`ê_O` fact stated for all hard gates |
| `v3/KAT-VAD_spec_v3.md` §0 | As-built status: CHANGE A implemented, D–K not; Alert-CLIP deferred |
| `v3/KAT-VAD_spec_v3.md` §4.1 | "gate is adaptive" row: UNTESTED → **REFUTED** |
| `v3/KAT-VAD_spec_v3.md` §10.3 | Row 1 answered; **row 4 (plain-TSM) promoted to top experiment** with a pre-registered prediction |
| `core/docs/TRAINING.md` | New "The KIP gate (v3)" section: 4-type matrix, the two raising guards, **"stage-1 convergence is a precondition"**, gate diagnostics + the windowing caveat, and the stock-CLIP deviation |

### F.3 Lesson 24 — `[CRITICAL]`, all five gates passed

*A non-differentiable op on the score path silently freezes everything upstream of it.*

| Gate | Verdict |
|---|---|
| 1 REAL | 6 tensors `grad is None`; span 0–4/128 over 8 seeds; three campaigns mis-described |
| 2 RECURRABLE | `floor`/`round`/`argsort`/`argmax`/`topk`/`.long()` on any score path; the STE variant reproduced the same trap |
| 3 NON-OBVIOUS | `ruff`, `mypy`, `pyright` all pass; params exist, optimizer accepts them, losses fall, the model reaches a publishable number |
| 4 ENFORCEABLE | *"Assert a non-zero gradient on every module you describe as learned, and log the output of any hard/rounded decision variable on the score path."* |
| 5 NOT DUPLICATE | `grep -i` over `index.md` for gradient/floor/differentiable/frozen returned nothing covering it |

Severity `[CRITICAL]` per `GATES.md` ("silently invalid experimental results"), so it also carries an inline entry (**C24**), four trigger-map rows, and a new *Architecture / gradient flow* category in `meta-index.md`. `detailed.md` §24 records the seed table, why every check passed, and the STE trap in full.

### F.4 Memory bank

`activeContext.md` — new dated section leading with H4′ and the reordered queue (plain-TSM control promoted to #1 with its pre-registered prediction), what shipped, the two spec/doc errors, and what remains unimplemented. `systemPatterns.md` — "Two structural facts about KIP" → **three**, fact 1 rewritten from "H4 not refuted" to "**H4′ CONFIRMED**", fact 2 new (parameter-free gate, gradient topology unchanged, the STE no-op), plus four new rows in the patterns table.

### F.5 Commit not made — awaiting go-ahead

Gates and `detect_changes` are green, but committing was not requested. Proposed split, ≤20 files each, added individually, on branch `v3`:

1. **`feat(kip): v3 gate rebuild`** — `core/kip/ecmr.py`, `core/kip/gate_shift.py`, `core/kip/kip_module.py`, `core/kip/__init__.py`, `core/config.py`, `core/constants.py`, `core/models/kat_vad.py`, `core/models/ckpt_compat.py`, `core/inference.py`, `core/evaluate.py`, `core/train.py` (11)
2. **`test(kip): v3 gate, inference graph, diagnostics`** — 4 new test files + 2 updated (6)
3. **`docs(kip): v3 gate, H4′ confirmed, stock-CLIP deviation`** — `core/docs/TRAINING.md`, `core/docs/v3/` ×3 (4)

`.project/` is excluded per §10. `CLAUDE.md` is gitignored.

---

## 6. Risks & Mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | **The default flip silently re-defines every re-run.** `gate_type` defaults to `rank`; the ~20 `outputs/*/config.yaml` on disk have no such key. | Two arms with byte-identical configs would be different models — the class of defect that voided three campaigns (limitation 11). | T2.4: an ambiguous loaded config **raises** rather than resolving either way. Always log the resolved gate type. |
| R2 | **Stage-1 convergence is load-bearing** — PMG's only gradient sources are `L_KIP_rec`, `L_KIP_align` and `L_kin`-via-`mhead`. A run whose stage-1 has not converged ranks a poorly-trained `ê_O`. | An uninterpretable arm that still produces a checkpoint and a plausible number. | Accepted by the user 2026-08-30 as **pre-existing, not introduced here** — see the note below the table. T2.6 asserts the topology; T5.3 documents the precondition; spec §8's `assert stage1_final(L_KIP_rec) < τ_rec` is the enforcement and stays in the deferred training-pipeline scope (§7 item 1). |
| R3 | **Every published number becomes gate-type-qualified.** | `RESULTS_*.md` says "KIP-on"; that now means `mlp_frozen`. | T5.5 records it. Do **not** retro-edit the results docs — they were true for what ran. Add the qualifier at the point of next citation. |
| R4 | Rank-map tie behaviour on near-static clips is arbitrary (index order). | A genuinely static clip still gets a full 0→128 shift sweep. | T1.2 logs a warning at `m.std() < eps`. Worth watching in Phase 4 diagnostics on MSAD fixed-camera clips — it may be a real defect on that half of the corpus. |
| R5 | Masked EMA / rank off-by-one changes results with batch composition. | Non-reproducible arms; worse, quietly. | T1.5's mask-invariance test is the guard. Non-negotiable. |
| R6 | `KIPOutput` return-type change breaks call sites. | Compile-time, not silent. | One production call site; `trace_call_path` in P0.1 confirms before editing; mypy/pyright catch the rest. |
| R7 | Scope creep into the training pipeline (CHANGE E/H/I). | The reviewable unit stops being reviewable. | Explicitly out of scope. Deferred items listed in §7. |
| R8 | RefineVAD may itself use an STE not described in the paper (P0.4). | Changes whether the frozen gate is *our* bug or *theirs* — a publishable observation either way. | Time-boxed to 1 h; does not block. |

### Correction to R2 — the rank gate does not change the gradient topology

An earlier draft of this plan called PMG gradient starvation a consequence of the rank gate. **That was wrong**, and the record is corrected here so it is not repeated downstream.

Starvation was already total in v1. `s_t = (ratio * max_shift).floor().long()` is used as a slice index, so the backward pass stops there — `L_MIL` reached neither the gate MLP nor, through it, the PMG head. This is exactly what report §12.5 found (all 6 gate tensors `grad is None`). So in the runs that produced **+0.0915 ± 0.0088**, PMG was already trained by `L_KIP_rec` + `L_KIP_align` + `L_kin`-via-`mhead` and by nothing else.

The rank gate removes **321 already-dead parameters**. It does not add or delete one edge of the gradient graph. Consequences:

- The v1→v3 gate change is a **forward-pass change only**, which is what makes it a clean comparison.
- "Stage-1 must converge" is a **standing precondition of KIP as built**, not a new cost of v3 — and it was already known to be under-met (spec §4.5: `L_KIP_rec` reached within 5 % of its minimum only at step ≈446 of 500).
- T2.6's grad assertions therefore **pin an existing invariant**; they are regression guards, not proof of a new defect.

One real forward-pass difference remains, and it is the point of the change: the frozen MLP applied a random-but-fixed near-monotone map that could compress `s_t` into a narrow band, while the rank map is forced to span `[0, 128]` on every clip. That is the H4′ property, and it shows up in Phase 4's diagnostics, not in the gradients.

---

## 7. Explicitly out of scope (next plans)

Deferred by the user's scope decision, in priority order:

1. **CHANGE I preconditions + run manifest** — `assert stage1_final(L_KIP_rec) < τ_rec`, flow-cache coverage, transform-parity hash, run manifest. Also closes lesson **C17** (already HIGH in the live queue). **Highest value of everything deferred.** `assert stage1_final(L_KIP_rec) < τ_rec` is the enforcement for the standing stage-1 precondition (R2), which this plan documents but cannot enforce.
2. **CHANGE E staging** (S1 PreVAD trunk → S2 KIP warm-up → S3 joint) and **CHANGE H** row-gated `L_KIP_rec`.
3. **CHANGE F** flow-bearing corpus expansion (UCF-Crime, DoTA-train, BDD100K) — data engineering, weeks.
4. **CHANGE K** float64 metrics — already item 2 in the live `activeContext` queue, independent of this plan.
5. Alert-CLIP swap — blocked on a checkpoint existing.
6. Phase 7 ATS/MLLM — unblocked only on PreVAD-trunk arms (§6), and those don't exist yet.

---

## 8. Decisions settled, and what is left

### Settled 2026-08-30

| # | Decision | Outcome | Consequence in the plan |
|---:|---|---|---|
| 1 | Fresh-config default | **`gate_type="rank"`** — the `v3` branch holds v3-faithful code only, so there is no legacy default to preserve | T2.3 unchanged; T2.4 rewritten to **raise** on an ambiguous config rather than invent a legacy fallback |
| 2 | Accept R2 | **Accepted**, on the correct grounds: gradient starvation is pre-existing and total in v1, not introduced by the rank gate | R2 row rewritten; correction note added under §6; §7 item 1 reworded |
| 3 | `mlp_ste` + `constant` | **Build now** (my call) | Already in T2.2; see rationale below |

**On decision 3.** Both go in the Phase 2 dispatch now. Reasons, in order of weight:

- `constant` is the **plain-TSM control** (ablation ladder #4), and spec §10.3 says it *has never been run* despite being the correct baseline if H4′ holds. It costs ~15 LOC alongside the rank gate and nothing later.
- `mlp_ste` is forward-**bit-identical** to `mlp_frozen`, which makes it the cleanest controlled experiment in the whole program (ablation #5): a pure gradient intervention with zero forward-pass confound. That property only holds if it is built against the same dispatch as `mlp_frozen`; bolting it on later risks the two drifting.
- Building all four now means the four gate types share one `shift_counts_from_ratio` (T1.3), so the floor/clamp/mask semantics **cannot diverge** between arms. Added later, they would be four independent code paths and any cross-gate comparison would carry an implementation confound — the C14 pattern, one level down.

Cost of the decision: ~35 LOC and ~6 tests inside work already scheduled. No schedule change; still 3–5 days.

### Remaining actions

1. **Say go** — Phase 0 starts with the `trace_call_path` blast-radius report on `KinematicShift`, `KIP.forward` and `MotionScoreHead`, plus the `kip.*` key inventory from an existing KIP-on checkpoint. Nothing is edited until that report is back to you.
2. **One open question, low stakes:** should `--dump-kip-diag` (T4.4) default **on** for eval runs? It is a few KB per clip on top of the existing `.npz`, and ablation rows 1 and 2 need it across all existing checkpoints. My inclination is on-by-default for `inference.py`, off for training. Answerable at Phase 4, not now.
3. **P0.4 stays time-boxed to 1 hour** — read RefineVAD's MoTAR implementation for a straight-through relaxation. Either it validates §4.2 or it becomes a publishable observation about an AAAI 2026 method. Does not block Phase 1.
