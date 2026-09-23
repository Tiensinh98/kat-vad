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
      + bottomk_weight          · L_bottomk     (abnormal_bottomk_loss)  [arm, 0]
```

Stage 1 (`train.stage=1`) trains **KIP only** (everything else
`requires_grad=False`) with `lambda_rec·L_KIP_rec + lambda_align·L_KIP_align`;
no text encoding happens at all.

KIP ablation gating (spec §10): `kip.enabled=false` → pure baseline;
`kip.pmg_only=true` → rec+align only; `kip.use_lkin=false` → no `L_kin`.

### The flow target and `lambda_rec` (lesson C37, Option A)

`L_KIP_rec` is a bare MSE against the cached `e_O`, so its size is set by the
**target's units**, not by how well PMG fits. Two cache versions exist:

| `--flow-dir` | `e_O` built from | global-mean MSE `V` | `lambda_rec` |
|---|---|---|---|
| `cache/flow/v1/{DS}` | 23 **raw** flow stats (pixel units) `@ M` | T2: **31.64** | `1.0` (default) — ~32× the O(1) task terms on T2 |
| `cache/flow/v2_zscore/{DS}` | the same stats **z-scored with train-split moments**, same `M` | ≈ 1 by construction | **`1/V` from `zscore_manifest.json`** |

Build v2 with `python -m core.flow.zscore_cache` (no frames, no RAFT; see
`COLAB.md` §4.3b). It writes `lambda_rec = round(1/V, 4)` into the manifest —
**derived, never swept** (lesson 14). The value lands near **1.0**: the weight
barely moves, the *loss* shrinks ~30× because the target changed. **Never pass
v1's 0.0316 to a v2 run** — that switches `L_KIP_rec` off.

A v2 cache is bound to one dataset's train split (`zscore_stats.npz` records its
SHA-1); using it for another corpus is a C2 violation. `.stats.npy` stays raw in
v2 — the motion-aware KNN key reads it. KIP-off runs never read flow or
`lambda_rec` (`build_trainer` passes `flow_dir=None`; pinned by
`core/tests/test_flow_zscore.py::TestKipOffIsInert`), so KIP-off arms are
comparable across the two caches. Read any `kip_rec` as `R² = 1 − kip_rec / V`
of the cache it trained on.

### Phase 1 arms — both default to the baseline

Added 2026-09-09 from `DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` §6. Neither
changes a single number unless you switch it on; the default graph and the
logged loss keys are byte-identical to before.

| Flag | Default | What it does |
|---|---|---|
| `loss.dvs_anchor_mode` | `span` | `span` = baseline: the whole spliced anchor is a dense positive in `supervised_loss`. **`ignore`** drops the anchor interior from that BCE — filler frames stay hard negatives and `pseudo_sup_mil_loss` supplies the positive pressure through its in-span top-k. Lesson **C29**. |
| `loss.bottomk_weight` | `0.0` | Weight of `abnormal_bottomk_loss`: pushes the **lowest**-k frames of each *abnormal* clip toward 0. The only term in the objective that lowers a frame inside a positive bag. Normal clips are excluded — `L_MIL` already bounds them. |
| `loss.bottomk_topk_pct` | `16` | `k = max(1, L // this)` for that term. |

**Why `ignore` exists.** `compose_sequence` marks the entire anchor clip
positive, which is right only when an abnormal clip is ~all anomaly. On
DADA-2000 the mean true positive fraction of an abnormal clip is **0.351**, so
the dense BCE trains **64.9 %** of those frames to 1 against a 0 annotation —
and it is the densest gradient in the objective. Measured consequence: normal
frames inside abnormal clips scored **0.4078** against **0.4076** for true
positives. Check the corpus's positive fraction (`core.tools.eda`, §2) before
leaving this on `span`.

`loss.dvs_anchor_mode` is validated at the point of use and **raises** on a
typo — an unrecognized value must not leave the arm silently off.

## Documented deviations from the baseline

1. **`L_dvs` pair is row-gated.** The baseline marks the *whole* abnormal
   anchor as pseudo-positive (all-ones `y^p` when un-synthesized). Our DVS
   (Phase 4, reviewed) keeps `y^p` all-zero for un-synthesized abnormal clips
   because the true window is unknown under weak supervision. Feeding those
   rows to `supervised_loss` would BCE every frame toward 0 and fight `L_MIL`,
   so `train.py` applies the pair only to rows that are **normal or
   synthesized**. If reproduction gate (b) misses, this is the first knob to
   revisit (plan §6). **The row gating is correct; the *span semantics* are
   the open problem** — see `loss.dvs_anchor_mode` above and lesson **C29**.
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

A run writes exactly one checkpoint, `checkpoint_last.pt`, at each epoch
boundary; `train.checkpoint_every_steps` and `checkpoint_step_*.pt` were
**removed on 2026-09-13** (see "Checkpoint writes" below). It stores
model/optimizer/scheduler/scaler, epoch, global step, `batches_done` within
the epoch, config, class names and
**all RNG states** (python / numpy / torch / cuda / mps / dataset-DVS /
verbalizer). Batches are drawn from a per-epoch seeded permutation with a
manual loop (no DataLoader worker processes), so `--resume` reproduces the
exact remaining batch sequence. (`batches_done` is retained so checkpoints
written before 2026-09-13 still resume correctly; new ones always record 0,
because a save now only happens at an epoch boundary.)
Weights after resume match a straight run within FP tolerance
(`core/tests/test_e2e_synthetic.py::TestKillAndResume`); strictly bitwise
equality is not guaranteed on backends with nondeterministic parallel
reductions (e.g. Apple Accelerate BLAS threads internally regardless of
`torch.set_num_threads`), where identical op streams differ at ULP level.
`--stop-after-epochs N` time-boxes one invocation without changing the
LR-schedule horizon.

## Checkpoint writes

**One checkpoint per run, written atomically.** `Trainer.save_checkpoint`
stages the payload to a `checkpoint_last.pt.part` sibling, `fsync`s it, and
only then renames it onto the target.

This is not decoration. `torch.save` straight onto the target opens it `"wb"`,
which truncates it to **zero bytes**, and only then streams several GB of
tensors in. A process killed inside that window leaves a 0-byte or half-written
`checkpoint_last.pt` that no existence check can distinguish from a finished
one — the failure surfaces hours later, at load time, with the run gone. The
window is wide on Colab and widens further when several training commands share
one GPU, because memory pressure is exactly what triggers the kill. Drive's
FUSE layer can also acknowledge a write whose bytes never land, hence the
`fsync` (lesson **C10**). After the rename the target is either the previous
checkpoint or the new one, never a torn file (lessons **C11 / C11b**).

`train.checkpoint_every_steps` was **removed**, not defaulted off. A config key
that parses and silently does nothing is the failure mode lessons **C19** and
**C24** are both about, so an old runbook that still passes it now fails loudly:

```
KeyError: Unknown config key: train.checkpoint_every_steps
```

Reintroducing step checkpoints for a trajectory probe is a deliberate change —
mind lesson **16** (step-uniform checkpoints undersample the loss range) and
the 459 MB-per-file cost measured in `PREVAD_SETUP.md`.

Pinned by `core/tests/test_e2e_synthetic.py::TestKillAndResume`:
`test_checkpoint_last_is_the_only_checkpoint`,
`test_checkpoint_every_steps_is_rejected`, `test_checkpoint_write_is_atomic`.

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

### Length-controlled evaluation (`--equalize-length`, lesson C28)

`core.evaluate --equalize-length N [--equalize-anchor center|start|end]` crops
every scored clip to exactly `N` sampled frames and **drops** the shorter ones,
so clip length carries no label information. Padding is deliberately not
offered: it would fabricate frames and interact with `ConvScoreHead`'s
`padding_mode="replicate"`.

Use it wherever the corpus's length distribution differs by class. On the
reconstructed DADA-2000 a detector reading only the frame count scores micro
AUC **0.8654**, so the uncontrolled number is not interpretable. `--equalize-anchor`
picks which window survives — **`end` on DADA-2000**, where the accident sits at
the end of the clip; `center` (the default) is the corpus-agnostic choice.

The run logs, and `results.json` records under `equalize`, how many clips were
kept vs dropped and **what fraction of the positive frames survived the crop**.
Read that retention number before reading the AUC: a control that deletes the
anomalies measures nothing.

## Checkpoint compatibility (gate a)

`core/models/ckpt_compat.py` maps LaGoVAD `best.ckpt` onto `KATVAD`:
`temporal_encoder.*` → `temporal_encoder.encoder.*`, top-level `gate_alpha` →
`temporal_encoder.gate_alpha`, frozen `clip_text_model.model.*` skipped
(comes from HF), `fusion./bin_head./sim_head./clip_text_model.prompt_embedding.*`
1:1, `kip.*` stays initialized. Unknown or missing keys **raise** — no silent
partial loads. Used via `--baseline-ckpt` in `core/evaluate.py` /
`core/inference.py`.
