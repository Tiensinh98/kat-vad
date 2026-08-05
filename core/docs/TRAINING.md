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
