# Training / inference / evaluation (Phase 5)

CLIs: `core/train.py`, `core/inference.py`, `core/evaluate.py`,
`core/tools/visualize.py`. All accept `--config <yaml>` plus repeatable
`--set section.key=value` overrides (see `core/config.py`). Colab usage:
`core/docs/COLAB.md`.

## Objective wiring (spec §8, stage 2)

```
total = L_MIL                                   (mil_loss; cls + cap branches)
      + mul_weight              · L_MIL-align   (multi_class_mil_loss)
      + pseudo_sup_weight       · L_dvs-sup     (supervised_loss)      [gated]
      + pseudo_sup_mil_weight   · L_dvs-supMIL  (pseudo_sup_mil_loss)  [gated]
      + cap_contrastive_weight  · L_neg         (CapContrastLoss)      [if captions]
      + lambda_rec              · L_KIP_rec
      + lambda_align            · L_KIP_align
      + gamma_kin               · L_kin         (main scores = sigmoid, detached)
```

Stage 1 (`train.stage=1`) trains **KIP only** (everything else
`requires_grad=False`) with `lambda_rec·L_KIP_rec + lambda_align·L_KIP_align`;
no text encoding happens at all.

KIP ablation gating (spec §10): `kip.enabled=false` → pure baseline;
`kip.pmg_only=true` → rec+align only; `kip.use_lkin=false` → no `L_kin`.

## The KIP gate (v3, 2026-08-30)

`kip.gate_type` selects how the per-position shift count `s_t` is produced. All
four share one floor/clamp/mask implementation
(`core.kip.ecmr.shift_counts_from_ratio`), so cross-gate comparisons carry no
implementation confound.

| `gate_type` | Params | `s_t` behaviour | Use for |
|---|---:|---|---|
| `rank` (**default**) | **0** | ECMR residual → within-clip rank → spans `[0, 128]` on every clip | v3 default; the only gate that is genuinely input-adaptive |
| `mlp_frozen` | 321 | **Near-constant, `s ≈ 58–69`, span 0–4/128** | Reproducing every number in `RESULTS_*.md` bit-for-bit |
| `mlp_ste` | 321 | Forward bit-identical to `mlp_frozen`; backward unblocked | Ablation 5 only (also opens `L_MIL` → PMG, which Pi-VAD's unweighted `L_PMG` exists to prevent) |
| `constant` | 0 | Fixed `const_shift_ratio`; `ê_O` ignored entirely | The plain-TSM control (ablation 4) |

`gate_signal` (`flow_norm` / `feat_var`) applies to the `mlp_*` types only;
setting it alongside `rank` or `constant` **raises** rather than being ignored.

> **A config file with a `kip:` section and no `gate_type` raises on load.**
> Every config written before 2026-08-30 ran `mlp_frozen`; resolving the omission
> silently either way would make two arms with identical-looking configs
> different models (lesson C14 / lesson 24). Add the key explicitly.

> **A checkpoint is bound to its gate type.** `load_kip_state_dict` refuses an
> `mlp_frozen` checkpoint under a `rank` model and vice versa, keyed on the
> presence of `kip.shift.mlp.*`. **It cannot tell `rank` from `constant`** —
> both are parameter-free with an identical key layout. Until the run manifest
> (lesson 17) lands, record the gate type of any `constant` arm by hand.

### Stage-1 convergence is a precondition, not a nicety

Under `rank`, `mlp_frozen` and `constant`, `s_t` is a hard integer used as a
slice index, so **no gradient reaches `ê_O` or the PMG head through the shift**.
This is unchanged from v1 — the rank gate removes 321 already-dead parameters
and alters no gradient edge. The PMG head is trained by `L_KIP_rec`,
`L_KIP_align` and `L_kin`-via-`mhead`, and by nothing else.

Consequence: a stage-2 run whose stage-1 has not converged ranks a poorly-trained
`ê_O`, and still produces a checkpoint and a plausible number. Spec v3 §8's
`assert stage1_final(L_KIP_rec) < tau_rec` is the enforcement and is **not yet
implemented** (deferred to the training-pipeline plan). Until it is, check
`metrics.jsonl` for the stage-1 `kip_rec` floor before trusting a stage-2 arm.

### Gate diagnostics

`--dump-kip-diag` (default **on** in `core/evaluate.py` with `--save-scores`,
and in `core/inference.py`; **never** collected during training) writes
`kip_s`, `kip_gate_ratio`, `kip_m`, `kip_mu_norm`, `kip_eo_norm` into each score
`.npz`. Note the windowing: the rank is computed *within* a `data.max_vis_len`
window, so `s_t` resets at each window boundary — which is what the scored model
does, not a logging artifact.

## Documented deviations from the baseline

1. **`L_dvs` pair is row-gated.** The baseline marks the *whole* abnormal
   anchor as pseudo-positive (all-ones `y^p` when un-synthesized). Our DVS
   (Phase 4, reviewed) keeps `y^p` all-zero for un-synthesized abnormal clips
   because the true window is unknown under weak supervision. Feeding those
   rows to `supervised_loss` would BCE every frame toward 0 and fight `L_MIL`,
   so `train.py` applies the pair only to rows that are **normal or
   synthesized**. If reproduction gate (b) misses, this is the first knob to
   revisit (plan §6).
2. **Captions on description-less datasets.** MSAD ships no per-video
   descriptions, so the caption branch (`cap_*` losses + `L_neg`) is inactive
   by default — the baseline behaves identically when `desc_label` is empty.
   `loss.captions_from_definitions=true` synthesizes captions from the
   class-definition verbalizer instead; caveat: same-class videos then share
   near-identical captions, which the InfoNCE treats as false negatives.
3. **Scheduler.** Cosine-with-warmup implemented in-house
   (`train.cosine_warmup_lambda`, unit-tested) instead of
   `transformers.get_scheduler` — avoids HF internal-API drift (lesson P4).

### Backbone: stock CLIP, not Alert-CLIP

Spec v3 §0 and the architecture doc's Phase 1 both name **Alert-CLIP** as the
frame encoder. **The implementation uses stock CLIP ViT-B/16** at the pinned
revision (`constants.CLIP_MODEL_NAME` / `CLIP_MODEL_REVISION`), because no
Alert-CLIP checkpoint is publicly available (checked 2026-08-30). The swap is
deferred, not rejected. **Every measured number in this repo is a stock-CLIP
number**; do not attribute any of them to abnormality-tuned features.

## Resumability contract

Checkpoints (`checkpoint_last.pt`, optional `checkpoint_step_*.pt` via
`train.checkpoint_every_steps`) store model/optimizer/scheduler/scaler,
epoch, global step, `batches_done` within the epoch, config, class names and
**all RNG states** (python / numpy / torch / cuda / mps / dataset-DVS /
verbalizer). Batches are drawn from a per-epoch seeded permutation with a
manual loop (no DataLoader worker processes), so `--resume` reproduces the
exact remaining batch sequence — mid-epoch step checkpoints included.
Weights after resume match a straight run within FP tolerance
(`core/tests/test_e2e_synthetic.py::TestKillAndResume`); strictly bitwise
equality is not guaranteed on backends with nondeterministic parallel
reductions (e.g. Apple Accelerate BLAS threads internally regardless of
`torch.set_num_threads`), where identical op streams differ at ULP level.
`--stop-after-epochs N` time-boxes one invocation without changing the
LR-schedule horizon.

**Device warning (pending lesson P6):** on this repo's graph, torch 2.4 **MPS
training diverges** (stage-1 `L_KIP_rec` climbs while CPU converges on
identical seeds/data). Pin `train.device=cpu` for local runs on Apple
Silicon; Phase-6 training targets CUDA.

## Text encoders

`--text-encoder clip` (default) uses the frozen CLIP tower + soft prompts and
needs HF weights (`core.tools.download clip`). `--text-encoder stub` produces
deterministic per-string Gaussian embeddings (sha256-seeded) — zero downloads;
used by the synthetic E2E and Colab dry-runs. Stage-1 training encodes no text
with either setting.

## Evaluation semantics (spec §10, A5)

Full-length videos are scored in `data.max_vis_len` windows and concatenated
(the baseline `full_length_eval.py` convention, incl. re-sampling verbalized
definitions per window; `--no-verbalize` disables). Micro AUC/AP are computed
over all concatenated sampled frames — equivalent under the uniform ×stride
clip→frame expansion (`core.inference.expand_to_frames`). AUC_A, the MCC
family and mAP@IoU raise `NotImplementedError` until Phase 7.

## Checkpoint compatibility (gate a)

`core/models/ckpt_compat.py` maps LaGoVAD `best.ckpt` onto `KATVAD`:
`temporal_encoder.*` → `temporal_encoder.encoder.*`, top-level `gate_alpha` →
`temporal_encoder.gate_alpha`, frozen `clip_text_model.model.*` skipped
(comes from HF), `fusion./bin_head./sim_head./clip_text_model.prompt_embedding.*`
1:1, `kip.*` stays initialized. Unknown or missing keys **raise** — no silent
partial loads. Used via `--baseline-ckpt` in `core/evaluate.py` /
`core/inference.py`.
