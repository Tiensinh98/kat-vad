# System Patterns — architecture & design decisions

**Created:** 2026-07-31 (re-init from `b9978ff`, verified against the tree)
**Last reviewed:** 2026-09-08 (branch identity; v3-only components marked)

> **Branch `main` = KAT-VAD v1.** Anything below marked **[v3 only]** is *not in
> this tree* — it lives on branch `v3` (tip `bb1516c`). It is documented here
> because the design argument is expensive to redo and because the measured
> results depend on it.

## Forward pass

```
video ─► frozen CLIP ViT-B/16 ─► F (L×512)          core/tools/extract_clip_features.py (cached .npy)
                                  │
                                  ▼
        temporal encoder (RoFormer 2L/4H, RoPE, window-25 mask)   core/models/temporal_encoder.py
                                  │  v^t (L×512)
                                  ▼
        ┌──────────────── KIP (novel) ────────────────┐           core/kip/
        │ PMGFlowHead   v^t → ê_O (L×256)             │  pmg.py
        │ KinematicShift  gate + adaptive shift        │  gate_shift.py
        │ MotionScoreHead ê_O → ŷ_O                    │  motion_head.py
        └──────────────► v^k (L×512) ◄────────────────┘  kip_module.py
          on `main` the gate is the v1 frozen MLP, full stop:
          no `gate_type`, no `ecmr.py`  →  [v3 only]
                                  │
   definition Z ─► frozen CLIP text + 32 soft prompts ─► z^t (C×512)   core/models/clip_text.py
                                  │
                   co-attention fusion U ×2 ─► v^u, z^u  core/models/fusion.py
                                  │
                   H_bin → y^bin (L)   |   H_mul → y^mul (L×C)   core/models/heads.py
```

Assembly lives in `core/models/kat_vad.py`. **The only splice** is KIP between
temporal encoder and fusion: when `kip.enabled`, `v^k` feeds fusion, the H_bin
pre-path and `L_neg`; when off, `v^t` does. Text encoding is decoupled
(`encode_text`) so visual-path tests need no CLIP weights.

### Four structural facts that are easy to state wrongly

1. **H4′ is CONFIRMED for the v1 gate: it was a constant smoother** (measured
   2026-08-30, plan Appendix C). `compute_shift_counts` ends in
   `(ratio * max_shift).floor().long()`, which is not differentiable and is used
   as a slice index, so `KinematicShift.mlp`'s 321 parameters stay at random
   init for the entire run (all 6 tensors `grad is None`, pinned by
   `test_gate_mlp_receives_no_gradient_spec_as_written`). Random init is
   therefore the *deployed function* — and because the gate's input is min-max
   normalized, `[0, 1]` is its **entire reachable input domain**. Sweeping that
   whole domain moves `s_t` by **0–4 channels out of 128** across 8 seeds (seed
   0: exactly 0). `mlp_frozen` and `constant` at `r = 0.5` agree to within half a
   channel on the mean. **Never write "motion-gated" of `gate_type="mlp_frozen"`.**
   The decisive follow-up is the plain-TSM control — one config flag **on `v3`**;
   on `main` it needs a code edit, so **do not fake it with a hand-patched
   `compute_shift_counts`** (lesson C17: the config would not record it).
   → lesson **24 [CRITICAL]**.
2. **[v3 only — not in this tree] The v3 gate is parameter-free, and this changes no gradient edge.**
   `gate_type="rank"` (default) runs ECMR (causal EMA prototype → residual) into
   a within-clip rank map, spanning `[0, 128]` on every clip by construction —
   so a constant smoother is not expressible. `core/kip/ecmr.py`, 0 parameters.
   Gradient topology is **unchanged from v1**: `grad → v^t` present, `grad → ê_O`
   absent, for `rank`, `mlp_frozen` and `constant` alike (parametrized test).
   The rank gate removes 321 *already-dead* parameters; it does not introduce
   starvation. Consequence, unchanged and now documented: PMG is trained only by
   `L_KIP_rec`, `L_KIP_align` and `L_kin`-via-`mhead`, so **stage-1 convergence
   is a precondition**. `mlp_ste` unblocks the backward pass as an ablation —
   and the spec's formula for it (`s = u + (⌊u⌋−u).detach()`) is a **no-op**,
   because `s` is consumed only inside a comparison; the estimator must be
   applied to the selection weights (`shift_channels_straight_through`).
3. **`ê_O` has 23 effective dimensions and no spatial content.**
   `flow_statistics` (`core/flow/raft_extract.py:52-81`) pools each RAFT field
   into 23 frame-global scalars — magnitude mean/std/max, u mean/std, v mean/std,
   and a 16-bin L1-normalized magnitude-weighted angle histogram — and a fixed
   seeded Gaussian projection lifts them to 256-d. So `L_KIP_rec` regresses
   global motion statistics, not a motion field. Two corollaries: a
   "localized/peripheral event" mechanism is **not expressible in the target**
   (what can carry one globally is `mag_std`, `mag_max`, and a second mode in the
   angle histogram), and `‖ê_O‖₂` mixes an L1-normalized histogram with
   unnormalized magnitude scalars, so the gate signal is dominated by raw
   magnitude scale.

4. **The score head's receptive field can swallow a whole clip.**
   `H_bin` is `ConvScoreHead` with `score_head_layers=1`, i.e. a single
   `Conv1d(512 → 1, kernel_size=9, padding=4, padding_mode="replicate")`
   (`core/models/heads.py:21`). Median sequence length per corpus: **MSAD 86,
   DoTA 13, DADA-2000 9** (55 % of DADA clips are ≤ 9). On DADA every output
   timestep therefore sees **every** input frame, adjacent timesteps differ only
   by a fully-overlapping window slide, and the head degenerates into clip
   pooling — flat curves, `auc_macro` at chance, micro AUC that is really video
   classification. The supervision degenerates with it: `_topk_k` is
   `max(1, n // topk_pct)` (`core/losses/mil.py:22`), so `L_MIL` on a 9-frame
   clip is a plain max over 9. **`score_head_kernel` is a per-corpus decision,
   not a constant.** Measured 2026-09-06, `core/docs/v3/RESULTS_DADA.md` §5.

## Key decisions

| Decision | Where | Rationale |
|---|---|---|
| Plain PyTorch + argparse, no Lightning | `core/train.py` | baseline's Lightning coupling is not worth porting; full control over resume |
| Config = dataclasses + YAML + `--set a.b=v` | `core/config.py` | every spec §10 ablation is a typed flag; unknown keys **raise** |
| Constants centralized | `core/constants.py` | data-layout contract + baseline hyperparameters in one place |
| Env-var roots (`KATVAD_*_ROOT`) | `core/constants.py:15-20` | Colab points data/cache/ckpt/output at Drive without code edits |
| Versioned flow cache `cache/flow/v1/` | `core/flow/raft_extract.py` | the A10 projection is part of the cache identity; loaders fail loudly on mismatch |
| In-house cosine-warmup LambdaLR | `core/train.py` | avoids `transformers.get_scheduler` internal-API drift |
| Vectorized shift + loop oracle | `core/kip/gate_shift.py` | equivalence-tested; vectorized ≈8× faster on CPU |
| **[v3 only]** Four selectable gate types sharing one floor/clamp/mask step | `core/kip/gate_shift.py`, `core/kip/ecmr.py` — **absent on `main`** | `rank` / `mlp_frozen` / `mlp_ste` / `constant`; one `shift_counts_from_ratio` so cross-gate comparisons carry no implementation confound |
| **[v3 only]** Train-only KIP submodules off the inference graph | `core/kip/kip_module.py` (`train_only_modules`) — **absent on `main`**; here `mhead` is always instantiated | 3e/3f not instantiated when `training=False`; 311,808 params at inference, 0 on the score path; curve is bit-identical across graphs |
| Allowlist checkpoint loader, never `strict=False` | `core/models/ckpt_compat.py` (`load_kip_state_dict`) | drops exactly `kip.mhead.` / `kip.proj_flow.` / `kip.proj_rgb.`; **the gate-type mismatch guard is [v3 only]** — on `main` there is one gate, so a v3 checkpoint's `kip.*` keys will not map here. **Cannot separate `rank` from `constant`** — identical key layout; needs the run manifest (lesson 17) |
| **[v3 only]** Gate diagnostics into the existing `.npz` | `core/inference.py`, `core/evaluate.py` — **absent on `main`**: no `--dump-kip-diag`, no `kip_s` column | `--dump-kip-diag`, on for eval/inference, never for training; `kip_s` int16 + four float32 columns; no new artifact format |
| Fail-loud checkpoint mapping | `core/models/ckpt_compat.py` | unknown/missing/mis-shaped keys raise — no silent partial loads |
| Pooling resolved from labels, not dataset name | `core/metrics.py:resolve_score_norm` | `--score-norm auto` picks per-clip min-max when normal videos fall below 5 % of the test set; a new all-abnormal benchmark cannot silently inherit the wrong protocol (lesson 12) |
| Rescore from saved `.npz`, never re-infer | `core/tools/rescore.py` | model outputs are deterministic given features, so a protocol correction costs minutes instead of GPU hours |
| Atomic `.part` → rename cache writes | `core/tools/feature_cache.py` | skip-if-exists is not a resume: a killed `np.save` leaves a truncated `.npy` that `exists()` calls done (lesson 11) |
| One cache directory per transform | `cache/clip/{MSAD,MSAD_ncc,DoTA_s8_ncc,PreVAD_rel}` | a checkpoint is bound to the preprocessing that trained it; mixing them is a silent train/test mismatch (lesson 13). `PreVAD_rel` is named for the *release* that built it, not for our extractor |
| Label arity/bounds read off the widest shipped case | `core/data/prevad.py` | DoTA has one span per clip, PreVAD has up to four plus 440 out-of-range and 1 reversed; a single-span port mislabels silently (lesson 18) |
| Class-definition coverage asserted at preprocess time | `core/data/prevad.py:check_definition_coverage` | `verbalize_class_name` falls back to the bare class name by design, so a wrong taxonomy trains silently (lesson 19) |

## Loss composition (stage 2)

```
total = L_MIL
      + mul_weight             · L_MIL-align   (multi_class_mil_loss)
      + pseudo_sup_weight      · L_dvs-sup     (supervised_loss)      [row-gated]
      + pseudo_sup_mil_weight  · L_dvs-supMIL  (pseudo_sup_mil_loss)  [row-gated]
      + cap_contrastive_weight · L_neg         (CapContrastLoss)      [if captions]
      + lambda_rec             · L_KIP_rec
      + lambda_align           · L_KIP_align
      + gamma_kin              · L_kin         (main scores detached)
```

Stage 1 (`train.stage=1`) trains **KIP only** — everything else frozen, no text
encoded at all, objective = `lambda_rec·L_KIP_rec + lambda_align·L_KIP_align`.

Ablation gating: `kip.enabled=false` → pure baseline; `kip.pmg_only=true` →
rec+align only; `kip.use_lkin=false` → no `L_kin`.

## Scoring & evaluation patterns

Added after the DoTA protocol failure (2026-08-08) and hardened since.

- **`core/metrics.py`** is torch-free so both `core/evaluate.py` (which scores)
  and `core/tools/rescore.py` (which only reads `.npz`) can use it. It always
  reports `auc`, `ap`, `auc_raw`, `ap_raw` and `auc_macro` side by side, so a
  change of protocol is auditable instead of a silent redefinition.
- **Micro vs macro.** Micro AUC concatenates all frames, so each clip's absolute
  score scale enters the metric — signal on MSAD (50 % normal videos), noise on
  DoTA (0.2 % normal, pure within-clip localization). Macro AUC (mean per-clip
  AUC) needs no normalization and is reported alongside.
- **The constant-score-per-clip oracle bounds what micro AUC can mean.** Micro
  fails in *both* directions, not just DoTA's. On a corpus with many all-normal
  test clips it rewards pure clip classification: DADA-2000 is 74 % all-normal
  frames, and a model emitting one constant score per clip scores **0.9086**
  there while localizing nothing. Compute that oracle from the label vector alone
  and print it beside the micro number; if a measured AUC is within a few points
  of it, there is no localization result. Mirror of lesson C12.
- **`--save-scores` is the unit of durable evidence.** Every reportable number in
  `core/docs/RESULTS_*.md` is recomputed from `{run}/scores/*.npz`, never from a
  remembered console line. Paired bootstrap over *clips*, 2,000 draws, both arms
  on the same resample.

## Experiment layout convention

Run artifacts follow `outputs/{BENCH}[_ncc][_s{SEED}]/{stage|eval}_{arm}` — e.g.
`outputs/MSAD_ncc_s2025/stage2_kip_off_warm`, `outputs/DoTA_ncc/eval_kip_on`,
`outputs/DoTA_ncc/probe/{arm}_checkpoint_step_{N}`. Seed 2024 has **no `_s`
suffix** (it predates the seed sweep) — every loop over seeds needs that special
case.

**v3 runs use `outputs/v3/{BENCH}/{arm}_s{SEED}/{stage2[_kip_on]|eval_{bench}}`**
(e.g. `outputs/v3/DADA2000/constant_s2024/eval_dota`), which does carry `_s2024`.
The two conventions coexist; a loop over both needs to handle each.

Arms currently defined: `gate_a` / `full_gate_a` / `gate_d0` (released
`best.ckpt`), `kip_off` (cold), `kip_on` (warm from stage 1), `kip_off_warm` (the
arm-4 control) — **these are the v1 arms and the only ones `main` can run**; and
the v3 ladder **A0** `kipoff` / **A1** `rank` / **A2** `constant` / **A2b**
`constant_pmg` / **A3** `mlp_frozen` / **A4** `mlp_ste`, identical in name across
the MSAD, TAD and DADA campaigns so the rows line up. **A1–A4 require
`kip.gate_type` and therefore branch `v3`.** Note the mapping: `main`'s `kip_on`
*is* A3 `mlp_frozen` in v3's vocabulary, and (per the attribution) is
statistically the same arm as A2 `constant`. **A run's `config.yaml` does not record `--init-weights` or any path
flag** (lesson 17) — what distinguishes `kip_off` from `kip_off_warm` is visible
only in the loss trace, so provenance has to be argued, not read.

## Documented deviations from the baseline

Full text in `core/docs/TRAINING.md`. Three of them:

1. **`L_dvs` pair is row-gated to normal|synthesized rows.** Our DVS keeps `y^p`
   all-zero for un-synthesized abnormal clips (the true window is unknown under
   weak supervision); feeding those rows to `supervised_loss` would BCE every
   frame toward 0 and fight `L_MIL`. **First knob to revisit if gate (b) misses.**
2. **Captions off by default** on description-less datasets (MSAD ships none).
   `loss.captions_from_definitions=true` synthesizes them from the verbalizer —
   caveat: same-class videos then become InfoNCE false negatives.
3. **Scheduler** implemented in-house rather than via `transformers`.

## Data pipeline

`core/data/msad.py` (preprocessor) → `labels_train.json`,
`frame_labels_test.json`, `defs.json`, `meta.json`; frame labels are stride-8
aligned (`i*stride ∈ [start,end]`). Splits are seeded and stratified by
`(label, scenario)`; `--scenarios traffic` selects the spec §7.2 slice,
omitting it keeps the full 11-class benchmark.

`core/data/dota.py` is the zero-shot adapter: labels are
`round(normalized_span × feature_length)` half-open, features at stride 8,
`resolve_frame_counts` classifies unreadable clips as *absent* vs *empty* (a
Drive unzip leaves created-but-empty folders, so folder counts lie about
coverage — lesson 10). Derived spans equal the baseline's shipped `anomaly_span`
on all 1,402 val clips.

`core/data/prevad.py` (2026-08-24) is the PreVAD adapter over the released
`ViT-B-16-8p` features. It has **no `--stride`**: the release ships no frames, so
the `.npy` *is* the interval-8 sequence and label lengths come from the array
header. Every anomaly span is filled independently and clamped to `[0, L]`,
reproducing what the baseline gets for free by writing into a 512-long buffer.
Definition coverage and feature layout (width 512, non-empty, L2 norm sampled)
are asserted here so they cannot be skipped.

`core/data/tad.py` (P1, 2026-09-02) gained `--with-train-split`, building the
weakly-supervised train split from the frame folders the annotation does not
name; the default is unchanged, so every pre-2026-09-02 TAD artifact still
reproduces bit-for-bit. **The split directory is the label, never the filename
prefix** — TAD's 60 abnormal test videos carry seven prefixes and LaGoVAD labels
all of them `Car Accident`, so a prefix rule would mislabel 37 of 60
(`_label_from_directory` raises rather than defaulting). `core/flow/raft_extract.py`
gained matching `--frames-dir` / `--frames-subdir` / `--ids-file` (P2), sharing
`_extract_sources` with the video path so the two cannot drift; frames-vs-video
parity is asserted after `np.ascontiguousarray` fixed a 2.67e-5 kernel-dispatch
divergence (lesson C25).

`core/data/dada.py` (2026-09-03/04) is the DADA-2000 adapter. Unlike TAD/DoTA it
builds a **real seeded train/test split** (stratified by `(is_abnormal,
Fault_Label)`) because DADA ships frame-level ground truth for nearly every
abnormal clip. Three record kinds: annotated abnormal, weak abnormal (no CSV row
→ forced into train), normal (`0_Normal_Driving`, directory is the label).
`video_id` is `{fault_dirname}__{folder_name}` — the bare `type<N>_vid<N>` names
**repeat across the three fault directories on the real archive**, and
`extract_clip_features.py` / `raft_extract.py` build a plain dict keyed on
`Path.name` with no collision check, so `--flat-frames-dir` materializes a
symlink farm under the unique id and both tools are pointed at that instead
(lesson C26). Weak supervision is preserved: a train record's window goes into
`meta.json` only, never into `labels_train.json`.

`core/data/dataset.py` implements DVS: `θ` = *no-synthesis* probability,
`__len__ = 2 × num_anomaly`, fillers 50% KNN / 50% random-normal, yields
`v_feat / e_o / y^p / is_synthesized / cls_label`. `require_flow=False` yields
zero `e_O` for KIP-off runs — which is what makes a **PreVAD KIP-off trunk
trainable without any flow cache** (`core/docs/PREVAD_SETUP.md` §7).

## Testing pattern

**On `main` (verified 2026-09-08): 418 collected → 418 pass, 0 fail.** The 12
that used to fail were `TestDadaTrainsUnderEveryGate` (5) +
`TestTadTrainsUnderEveryGate` (7), written on `v3` and parametrized over the
v3-only `kip.gate_type`; `core/config.py` raises on unknown keys **by design**,
which was the row above working correctly. They are now
`TestDadaTrains` / `TestTadTrains`, exercising the one gate v1 ships. All
data-free and CPU-only.
Three tiers: unit (shapes, masks, gradients), **parity** against read-only
baseline modules with 1:1 state-dict copies on random weights, and a **synthetic
end-to-end** run (`core/tests/test_e2e_synthetic.py`) covering train →
checkpoint → kill → resume → infer → eval → visualize on a fabricated
mini-dataset. Grew 221 → 284 with the DoTA adapter, `core/metrics.py`,
`rescore.py` and `feature_cache.py`; 284 → 322 with the PreVAD adapter; then the
branches split — the 434 / 467 / 497 / 537 figures are **`v3`'s**.
`TestDadaTrains` / `TestTadTrains` exist to pin that the adapter's output
*trains*, so a DADA/TAD arm cannot fail for a data reason that was never
exercised. On `v3` they are `TestXTrainsUnderEveryGate` and sweep the gate enum;
on `main` there is only one gate type, so the matrix was collapsed rather than
the class deleted — `test_kip_off_trains` (arm A0), `test_stage1_warmup_runs`
and the `config.yaml`-recording tests all survive.

## Planned seams (not built — the v2 plan file does not exist in this tree; this section *is* the record)

Recorded here because each one is a *design decision already argued*, and the
argument is expensive to redo:

- **Eval-time gate bypass, not build-time removal.** `use_gate_shift=false` sets
  `self.shift = None` (`core/kip/kip_module.py:41`), and `core/inference.py:64`
  loads at `strict=True`, so a KIP-on checkpoint raises on unexpected
  `kip.shift.mlp.*` keys. That is lesson C5 working — **do not weaken it.** Add
  `kip.bypass_gate_shift` consumed inside `KIP.forward` (~15 LOC); the same hook
  carries the constant-shift and gate-reseed controls.
- **Shuffled flow targets as a config flag** (`data.shuffle_flow_targets`), not a
  scratch edit, so the run manifest records it (lesson 17). It is the control
  that decides whether the gain is motion *content* at all.
- **Run manifest before any new arm** — resolved `--init-weights` + step + hash,
  data/cache paths, git sha, `sys.argv` (lesson 17).
- **Aspect parity is a full re-extract**, new `*_ap` cache paths, old cache kept
  (lesson 2/13). Sequence it *after* the shuffled-target control: if the gain
  survives a permuted target, the re-extract answers nothing.
