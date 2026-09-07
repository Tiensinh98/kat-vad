# Plan — MSAD_ncc seed repeat + checkpoint selection

**Written:** 2026-08-12. **Supersedes** `activeContext.md` "Immediate next steps"
item 2, which folded seeds and the val split into one training cycle. They must
not share a cycle — see §1.2.

---

> ## STATUS 2026-08-19 — A4 + A5 done; both confounds closed
>
> **A4 (KIP-off warm-started) and A5 (trajectory probe): DONE.** Results in
> **`core/docs/RESULTS_ARM4_PROBE.md`**. Δ(on − off_warm) = **+0.0988 ± 0.0148**
> over three seeds, nine CIs excluding zero — the warm start is not the
> mechanism. KIP-off's DoTA AUC is flat at 0.559–0.561 across a 33× range of
> train `mil`, and at convergence matched to KIP-on's endpoint it reaches ≈ 0.560
> vs 0.6519 — **H3 rejected.**
>
> **Phase B (§3): DEFERRED on evidence, not on a guess.** The probe was
> unambiguous, so §3's val-split machinery is no longer on the critical path for
> any open question. Build it only when checkpoint selection is itself the
> deliverable.
>
> **No confound remains open.** What is open is the *mechanism* — see
> `RESULTS_ARM4_PROBE.md` §6.

---

> ## STATUS 2026-08-16
>
> **Phase A (§2): DONE.** Seeds 2025 and 2026 ran; results in
> **`core/docs/RESULTS_PHASE_A.md`**. Verdict is §2.4's top row —
> DoTA Δ = **+0.0915 ± 0.0088**, all three seeds positive, min +0.0829, nine
> bootstrap CIs all excluding zero. **H1 holds, H2 rejected.** MSAD is null in
> every seed, which is the asymmetry §2.4 hoped for.
>
> **§0's reference table is superseded** — see `RESULTS_PHASE_A.md` §4. The MSAD
> reproduction gate was measured against the wrong reference: the released
> `best.ckpt` cannot reach 0.9041 either (0.8991 crop / 0.8949 ncc), and all
> eight paired bootstraps of our arms against it include zero. The gate is
> re-defined against the checkpoint and **passes**. Lesson 8b.
>
> **Phase B (§3): DEFERRED, not cancelled.** Its val-split machinery
> (`--val-ratio`, KNN rebuild, `select_checkpoint.py`, a retraining cycle on a
> 20 %-smaller train set whose numbers §3.3 admits are incomparable) is the
> expensive way to answer H3. A **trajectory probe** over the already-saved
> `checkpoint_step_*.pt` answers the same question directly, with no code and no
> retraining, on the training set everything else was measured on. Build §3 only
> if the probe is ambiguous.
>
> **§4 (the warm-start confound) is promoted to next.** Phase A supporting H1 was
> its stated precondition. Note §4 understates the cost: `warm_start_model`
> (`core/train.py:98-108`) rejects unexpected keys, so a KIP-off arm **cannot**
> `--init-weights` from a KIP-enabled stage-1 checkpoint without stripping
> `kip.*` first.
>
> Runbook for both: **`core/docs/COLAB.md` § "Arm 4 + trajectory probe"**.

---

**Precondition:** the `no_center_crop` rebuild is done. `outputs/MSAD_ncc/**`
and `outputs/DoTA_ncc/**` exist and are analysed. Seed **2024** is complete.

---

## 0. What the measurement currently is

| | MSAD-full (in-domain, raw pooling) | DoTA (zero-shot, per-clip min-max) |
|---|---|---|
| KIP-off | 0.8922 / AP 0.6587 | 0.5607 (macro 0.5638) |
| KIP-on | 0.8868 / AP 0.6574 | **0.6519 (macro 0.6746)** |
| Δ(on − off) | −0.0054, CI [−0.0155, +0.0033] — **null** | **+0.0911, CI [+0.0799, +0.1019]** |

Reference arms: `gate_a` (LaGoVAD released `best.ckpt`) = 0.8949 MSAD,
0.6012 DoTA. LaGoVAD published: 0.9041 MSAD, 0.6260 DoTA.

All of it is **n = 1 seed**, from `checkpoint_last`, with no model selection.

---

## 1. The three hypotheses this has to separate

**H1 — KIP genuinely helps under domain shift.** The +0.091 is a property of the
method and survives reseeding.

**H2 — Seed luck.** One draw out of a variance band wide enough to contain
+0.091. MSAD's own KIP Δ moved +0.0012 → −0.0054 between transforms, so the
per-run noise floor is not obviously small.

**H3 — Under-convergence transfers.** The KIP-on model is simply less fitted to
MSAD (final train `mil` 0.010 vs KIP-off's 0.0016; 21.2 % vs 24.5 % of MSAD
frames above 0.99). Worse in-domain, better out-of-domain is exactly what
under-fitting the source domain buys. Under this hypothesis KIP is a
regularizer, not a motion module, and the DoTA gain is not evidence for the
proposal's claim.

**Seeds separate H1 from H2. Only checkpoint selection separates H1 from H3.**
Both are needed; they are not the same experiment.

### 1.2 Why they cannot share one cycle

A val split takes videos **out of training**. MSAD-full's train side is 480
videos / **120 abnormal**; a 20 % val slice leaves 96 abnormal. That is a
materially smaller training set, so every number from a val-split run is
**incomparable** to the 0.8922 / 0.8868 / 0.6519 already measured, and
incomparable to LaGoVAD's 0.9041.

Change seeds and training-set size together and a moved number has two possible
causes. Phase A therefore changes **nothing but the seed**.

---

## 2. Phase A — seed repeat (no code changes, start immediately)

### 2.1 What runs

Seed 2024 is done. Run **2025** and **2026**. Per seed: 1 stage-1 warm-up +
2 stage-2 arms = 3 runs; 6 runs total.

Everything is `--set train.seed=N`. `set_global_seed`, `epoch_permutation`, and
`DVSFeatureDataset(seed=...)` all read `cfg.train.seed`
(`core/train.py:77,130`, `core/train.py:main`). **No code touched.**

The **data split stays fixed** — it is seeded separately in
`core.data.msad --seed` and is not being re-run. Same train/test videos, same
KNN cache, same features across all three seeds. Only training stochasticity
moves.

### 2.2 Commands

Per `S` in `2025 2026`. Paths follow `collab/MSAD_ncc/train.py` exactly, with
`_s$S` appended to output dirs.

```bash
cd /content/drive/MyDrive/Thesis/kat-vad
S=2025   # then repeat with S=2026

# --- stage 1: KIP warm-up (needed only by the KIP-on arm) ---
python -m core.train \
  --set train.stage=1 --set data.dataset=MSAD-full \
  --set train.num_epochs=125 \
  --set train.seed=$S \
  --data-dir  "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/MSAD/MSAD-full" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_ncc/stage1_s$S"

# --- stage 2, KIP on ---
python -m core.train \
  --set train.stage=2 --set train.amp=true \
  --set data.dataset=MSAD-full \
  --set train.num_epochs=125 \
  --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --init-weights "$KATVAD_OUTPUT_ROOT/MSAD_ncc/stage1_s$S/checkpoint_last.pt" \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --flow-dir "$KATVAD_CACHE_ROOT/flow/v1/MSAD/MSAD-full" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_ncc/stage2_kip_on_s$S"

# --- stage 2, KIP off (no warm start, matching the seed-2024 run) ---
python -m core.train \
  --set train.stage=2 --set train.amp=true \
  --set kip.enabled=false \
  --set data.dataset=MSAD-full \
  --set train.num_epochs=125 \
  --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --data-dir "$KATVAD_DATA_ROOT/MSAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/MSAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_ncc/stage2_kip_off_s$S"
```

Then 4 MSAD evals and 4 DoTA evals (2 seeds × 2 arms each). Eval reads cached
features, so these are minutes, not hours.

```bash
for S in 2025 2026; do
  for ARM in on off; do
    EXTRA=""; [ $ARM = off ] && EXTRA="--set kip.enabled=false"
    python -m core.evaluate \
      --ckpt "$KATVAD_OUTPUT_ROOT/MSAD_ncc/stage2_kip_${ARM}_s$S/checkpoint_last.pt" \
      $EXTRA --set data.dataset=MSAD-full \
      --data-dir "$KATVAD_DATA_ROOT/MSAD" \
      --clip-dir "$KATVAD_CACHE_ROOT/clip/MSAD_ncc" \
      --output-dir "$KATVAD_OUTPUT_ROOT/MSAD_ncc/eval_kip_${ARM}_s$S" --save-scores

    python -m core.evaluate \
      --ckpt "$KATVAD_OUTPUT_ROOT/MSAD_ncc/stage2_kip_${ARM}_s$S/checkpoint_last.pt" \
      $EXTRA --set data.dataset=DoTA \
      --data-dir "$KATVAD_DATA_ROOT/DoTA" \
      --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
      --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_ncc/eval_kip_${ARM}_s$S" --save-scores
  done
done
```

`--score-norm` stays at its `auto` default: it resolves from the label
distribution, so MSAD gets `none` and DoTA gets `minmax` without being told
(lesson C12). Do not pass it explicitly.

### 2.3 Budget

- **Time:** unknown to this plan — time the seed-2025 stage-1 run and multiply.
  Stage 2 is 125 epochs × 4 batches = 500 steps, same as seed 2024.
- **Drive:** each checkpoint is **~230 MB** (19.4 M trainable params + Adam
  state, measured). `checkpoint_every_steps=100` over 500 steps → 5 step
  checkpoints + `checkpoint_last` ≈ **1.4 GB per stage-2 run**. Six stage-2 runs
  across all seeds ≈ 8.4 GB. Keep the step checkpoints until Phase B has used
  them for selection, then prune.

### 2.4 Decision rule

Report Δ(on − off) per seed, then mean ± the across-seed spread, on **both**
benchmarks. What matters is the DoTA Δ:

| Across 3 seeds | Verdict |
|---|---|
| all three Δ positive, min > +0.03 | **H1 holds.** DoTA becomes the headline benchmark; write the claim |
| mean positive, one seed near zero or negative | Real but unstable — report mean ± spread, never a single number |
| spread straddles zero | **H2.** The +0.091 was a draw. Stop; do not report it |

Also recompute the MSAD Δ per seed. If MSAD stays null across all three while
DoTA is consistently positive, that asymmetry **is** the finding, and it is a
better story than a uniform win: the motion pathway pays exactly where motion is
the anomaly.

---

## 3. Phase B — checkpoint selection (needs code; runs after Phase A)

Only start this if Phase A supports H1. If the Δ is seed noise, H3 does not
matter.

### 3.1 The constraint that shapes the design

**MSAD's train split has no frame-level anomaly windows.** `write_standard_files`
(`core/data/msad.py:315`) *raises* on abnormal test videos lacking a window, and
the full-MSAD path uses `--infer-abnormal-from-name` precisely because the train
side carries video-level labels only. `labels_train.json` is
`{video_id: int(is_abnormal)}` — 480 entries, 120 abnormal.

So a frame-AUC val monitor **cannot be built from held-out train videos**, and
carving one out of the 240-video test set would contaminate the only number
comparable to the paper. Rejected.

**Monitor: video-level AUC on held-out train videos.** Video score = mean of the
top-`loss.mil_topk_pct` % frame scores — the same MIL aggregate the model is
trained under, so the monitor is consistent with the objective and needs no
frame annotations.

### 3.2 Code changes

**C1 — three-way split in `core/data/msad.py`.**
Add `--val-ratio` (default `0.0`, preserving current behaviour exactly), a
`LABELS_VAL_FILENAME = "labels_val.json"` constant, and val stratification
inside `split_records` mirroring the existing (label, scenario) grouping.
`write_standard_files` writes val ids to `labels_val.json` and **omits them from
`labels_train.json`**.

That omission is the whole trick: `core/data/dataset.py:68` builds the training
set from `labels_train.json`, so val videos drop out of training with **zero
changes to the dataset or trainer**.

*Blast radius (`trace_path`, inbound):* `split_records` and
`write_standard_files` each have exactly **one caller** — `msad.preprocess`,
reached only from the `msad.py` CLI. The graph labels hop-1 "CRITICAL" by
distance, not severity; real scope is one module plus its tests. Nothing in
`core/train.py`, `core/evaluate.py`, or the model path is touched.

**C2 — rebuild the KNN cache.** `core/data/knn_cache.py:166` reads
`labels_train.json`. Rebuild it after C1 or DVS fillers will pull val videos
into training — a silent leak that would make the monitor optimistic.

**C3 — new `core/tools/select_checkpoint.py`** (new file, no blast radius).
Args: `--run-dir`, `--data-dir`, `--clip-dir`, `--flow-dir`, `--set` overrides,
`--topk-pct` (default `loss.mil_topk_pct`), `--output`. For each
`checkpoint_step_*.pt` and `checkpoint_last.pt`: score the `labels_val.json`
videos through the inference path, compute video-level AUC, write
`selection.json` (every candidate's score, plus the winner), and copy the winner
to `checkpoint_best.pt`. Needs an `argparse` CLI and tests per §10/§11.

**Explicitly not doing:** an in-loop validation pass with a `checkpoint_best`
monitor inside `Trainer.train`. It would modify the training loop for all
existing runs, and post-hoc selection over saved step checkpoints answers the
same question using the trusted scorer. Storage is the only cost, and it is
already being paid.

### 3.3 What Phase B measures

Phase B numbers are on a smaller training set and are **not comparable** to
Phase A's or to LaGoVAD's. The comparable quantity is the *within-Phase-B*
Δ(on − off), plus how far `checkpoint_best` sits from `checkpoint_last` in each
arm.

| Observation | Reading |
|---|---|
| best ≈ last in both arms | `checkpoint_last` was fine. **H3 rejected** — the Δ is not a convergence artifact |
| best ≫ last for KIP-off only | KIP-off was over-trained past its optimum. Part of the Δ was H3; re-measure the Δ at best-vs-best |
| best ≫ last in both, Δ survives | Selection helps both arms; the Δ is robust. Strongest result available |
| Δ collapses at best-vs-best | **H3 holds.** KIP is acting as a regularizer. Say that — it is still publishable, just not the proposal's claim |

---

## 4. Confound this plan does not fix

**KIP-on warm-starts from stage 1; KIP-off starts cold.** Stage 1 trains
`L_KIP_rec` / `L_KIP_align`, whose gradients flow back through the temporal
encoder — so the KIP-on trunk gets 125 epochs of pretraining the KIP-off trunk
never sees. The A/B is therefore "KIP + warm start" vs "no KIP, cold", not
"KIP vs no KIP".

The clean control is a **fourth arm: KIP-off, `--init-weights` from the matching
stage 1**, with `kip.enabled=false` at stage 2. Cost: +1 stage-2 run per seed.

Worth doing **only if Phase A supports H1** — at which point this becomes the
first question a reviewer asks, and it is cheap relative to being wrong about
the mechanism. Not worth doing if the Δ is noise.

---

## 5. Rules for this run

- Do **not** pass `--score-norm` explicitly; `auto` resolves it from the label
  distribution (C12).
- Do **not** quote any arm except `gate_a` against LaGoVAD's 62.60 / 90.41.
- Do **not** change KIP's architecture, losses, or hyperparameters during this
  run. It is a measurement, not a search.
- One cache dir per transform; every arm scores on `*_ncc` features only (C13).
- Record per arm: seed, checkpoint path, coverage (N evaluated / N in split),
  AUC, AP, macro AUC, Δ + CI. Coverage is not optional (C10).
- After Phase A lands, add the lesson: **the center-crop transform was hiding
  KIP's entire effect** — C13 predicted it on theory, the ncc rebuild measured
  it (crop→ncc moved KIP-on by +0.151 macro and KIP-off by +0.008).
