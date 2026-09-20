# KIP loss scale — D1/D2 diagnosis before any repair

**Status:** pre-registered 2026-09-18 · **RUN 2026-09-20** · read out 2026-09-21.
**Outcome: decision row `capture + orthogonal` → §5 Option A, NOT authorized.**
Results and the scoring of every bar: **Appendix A** at the foot of this file.
Record: `outputs/v1/DADA2000_orig_diag_kip_loss_scale/`.
Nothing in §1–§6 has been edited — the bars below are as written before the run.
**Branch:** `main` (KAT-VAD v1). **Corpus:** `DADA2000_orig` T2 (W=20 hop 8).
**Record:** `outputs/v1/DADA2000_orig_phase4/`, three seeds, paired KIP on/off.

> Bars in this document are written **before** the measurements. Where a
> threshold has a condition, the condition is written into the threshold and not
> into the prose beside it — third project-wide instance of **C33** was enough.

---

## 1. What is already measured, and what it does not settle

Phase 4's A/B is clean: `s{2024,2025,2026}/stage2_kip_{on,off}/config.yaml` differ
in exactly one line, `kip.enabled`.

| eval | metric | kip_off | kip_on | paired Δ | t95 (n=3) | seed signs |
|---|---|---:|---:|---:|---|---|
| T2 in-domain | `auc_macro` | 0.6248 | 0.6046 | −0.0201 | [−0.0698, +0.0296] | − − + |
| T2 in-domain | `auc` micro | 0.6182 | 0.6054 | **−0.0129** | **[−0.0249, −0.0008]** | − − − |
| DoTA 0-shot | `auc_macro` | 0.6113 | 0.6071 | −0.0042 | [−0.1024, +0.0941] | − − + |
| DoTA 0-shot | `auc` micro | 0.5856 | 0.5914 | +0.0058 | [−0.0790, +0.0905] | − − + |

**T2 micro is the only row whose interval excludes zero**, and all three seeds
agree on its sign. KIP-on is not "not helping much" in-domain — it is costing
≈0.013 micro. The DoTA rows decide nothing at this width; do not quote them.

Three facts from `metrics.jsonl` (per-epoch means, last epoch, all three seeds):

1. **`kip_rec` is 92.6–93.0 % of `total`** — 11.28/11.95/11.57 against a task-loss
   sum of 0.903/0.906/0.869, at `lambda_rec = 1.0`.
2. **Stage 1 froze everything outside `kip.*` and `kip_rec` plateaued at 20.2.
   Stage 2 unfroze the trunk and it fell to 11.3.** 44 % of stage 2's
   reconstruction gain came from rewriting `v^t`, not from the PMG head.
3. **Every task loss is worse with KIP on**, same seed, same data order:
   `mil` +0.098/+0.085/+0.079, `dvs_sup` +0.048/+0.038/+0.037, `dvs_sup_mil`
   +0.073/+0.063/+0.053 — a task-loss sum **+32.1 % / +26.0 % / +24.5 %** higher.

And three code facts behind them:

* `core/flow/raft_extract.py:flow_statistics` pools each flow field into 23
  frame-global scalars of which the first seven — `mag_mean`, `mag_std`,
  **`mag_max`**, `u_mean`, `u_std`, `v_mean`, `v_std` — are **raw pixel units**;
  the 16 angle bins are L1-normalized and bounded by 1.
* Nothing normalizes them afterwards: `make_projection` is a fixed Gaussian map
  scaled `1/sqrt(23)`, and `core/data/dataset.py:114` loads the `.npy` verbatim.
  `kip_reconstruction_loss` is a bare masked MSE.
* `PMGFlowHead` reads `v^t` (`kip.on_raw_features=false`) and stage 2 freezes
  nothing (`core/train.py:191-195` freezes only in stage 1), so `L_KIP_rec`'s
  gradient reaches the shared temporal encoder.

**What none of that settles.** A loss of 11.3 against a target whose own scale is
unknown is not evidence the head fits badly, and a loss *share* of 93 % is not
evidence of a *gradient* share of 93 %. Both gaps are closed below, and both are
closed **before** any weight is touched — tuning `lambda_rec` against the Δ above
would be fitting the benchmark (lesson **14**).

---

## 2. D1 — what a measured `kip_rec` is worth

**Question.** Is 11.6 a good fit, a bad fit, or beneath a constant?

**Instrument.** `core/eda/features.py:flow_stats`, extended 2026-09-18 with a
`target` block and a `raw_energy_share` table; rendered by the EDA report as
§4.3/§4.3.1. It streams both caches (O(dims) per item) and reports the MSE three
reference predictors score on the same `e_O` the loss sees:

| symbol | predictor | key |
|---|---|---|
| `Z` | all zeros | `mse_zero_predictor` |
| `V` | one global mean vector | `mse_global_mean_predictor` |
| `W` | each item's own mean | `mse_item_mean_predictor` |

`K` = the measured `kip_rec`, **11.60** (mean of the three seeds' last epoch).

**Command** (Colab, Drive mounted; the flow cache is the train-only one built for
Phase 4):

```bash
source .venv/bin/activate && python -m core.tools.eda report \
  --dataset DADA2000_orig \
  --data-dir   "$KATVAD_DATA_ROOT/DADA2000_orig" \
  --clip-dir   "$KATVAD_CACHE_ROOT/clip/DADA2000_orig" \
  --flow-dir   "$KATVAD_CACHE_ROOT/flow/v1/DADA2000_orig" \
  --output-dir "$KATVAD_OUTPUT_ROOT/DIAG/d1_flow_target" \
  --sections features --no-probe
```

`--no-probe` because D0 already answered the representation question (0.6518);
this run is about the flow cache only.

### Pre-registered bars

| id | quantity | bar | what it licenses |
|---|---|---|---|
| **D1-a** | `R²_global = 1 − K/V` | **≥ 0.20 PASS** · < 0.20 **STOP** | PASS: the head fits real structure and `lambda_rec` is a live variable. STOP: the target is noise at this capacity — normalizing it changes the number, not the situation; go to §5 option **D**. |
| **D1-b** | `K` vs `W` | **`K < W` PASS** · `K ≥ W` **REFUTES** | PASS: PMG beats an oracle that knows only which item this is, so some of the fit is within-item dynamics. REFUTE: every bit of `L_KIP_rec` is item identity — KIP's motion premise is unsupported here whatever `lambda_rec` is. |
| **D1-c** | `V` | ≥ 4.0 **CONFIRMS overweight** | Confirms `lambda_rec = 1.0` buys `L_KIP_rec` a multiple of the whole O(1) anomaly objective. The equalizing weight is `1/V`, reported, not adopted. |
| **D1-d** | `predicted_second_moment_from_raw / Z` | ∈ **[0.9, 1.1]** else **HARD STOP** | `predicted` is `mean_j E[s_j²]`, which matches the target's per-dim second moment only *in expectation over the projection*; the realized matrix is fixed, so its Gram deviates by `~1/sqrt(d_O) = 6 %` at `d_O = 256` and the band is ≈1.6 of those. Outside it, the cache and the seeded projection disagree and *every* `kip_rec` ever measured on them is uninterpretable. Investigate before anything else. |
| **D1-e** | top `raw_energy_share` row | reported, no bar | Names the stat that sets the scale. `mag_max` is the hypothesis; it is **not** measured yet and must not be asserted until this prints. |

The EDA verdict block fires `HIGH` on D1-c at `FLOW_TARGET_SCALE_WARN_MSE = 4.0`
and on D1-b's population form at `FLOW_TARGET_BETWEEN_ITEM_WARN = 0.5`.

---

## 3. D2 — who actually steers the trunk

**Question.** Does the 93 % loss share become a gradient share at the temporal
encoder, and are the two gradients in conflict or merely unequal?

**Instrument.** `core/tools/grad_probe.py` (new). It builds the *same* objects a
run does — via `core.train.build_trainer`, extracted from `train.main` so the two
cannot drift — computes the stage-2 loss once per batch, then takes a separate
`autograd.grad` per weighted term against four parameter groups
(`temporal_encoder`, `kip`, `fusion`, `heads`). Read-only: no optimizer step, no
checkpoint written. It re-sums its weighted terms and **raises** if they do not
reproduce `compute_losses`'s own `total`.

It reports, at the trunk:

* `rho = |g_KIP| / |g_task|`, per KIP term and summed;
* `cos(g_KIP, g_task)`.

**Command**, run at **two** points so the trajectory is visible:

```bash
for POINT in stage1 stage2_kip_on; do
  source .venv/bin/activate && python -m core.tools.grad_probe \
    --config     "$RUNS/s2024/stage2_kip_on/config.yaml" \
    --data-dir   "$KATVAD_DATA_ROOT/DADA2000_orig" \
    --clip-dir   "$KATVAD_CACHE_ROOT/clip/DADA2000_orig" \
    --flow-dir   "$KATVAD_CACHE_ROOT/flow/v1/DADA2000_orig" \
    --knn-cache  "$KATVAD_CACHE_ROOT/knn/DADA2000_orig.npz" \
    --checkpoint "$RUNS/s2024/$POINT/checkpoint_last.pt" \
    --output-dir "$KATVAD_OUTPUT_ROOT/DIAG/d2_grad/s2024_$POINT" \
    --num-batches 8
done
```

`--config` is deliberately the **kip_on** config in both runs: the stage-1
checkpoint is being probed *under the stage-2 objective*, which is the state
stage 2 started from. `--checkpoint` loads **model weights only**, through
`core.train.warm_start_model`: the stage-1 checkpoint has no `clip_text_model.*`
(stage 1 builds the model with `load_clip=False`) and the strict same-run resume
path refuses it over that absence alone — fixed 2026-09-20, lesson **C36**. Keep `--text-encoder clip` (the default) so the graph is the
one that trained; that needs `transformers==4.56.*` pinned (**C7**).

### Pre-registered bars

| id | quantity | bar | what it licenses |
|---|---|---|---|
| **D2-a** | `rho` (all KIP terms), at the stage-2 end point | **≥ 1.0 CONFIRMS** · < 0.30 **REFUTES** · else partial | CONFIRM: the trunk is majority-steered by KIP and "trunk capture" is the explanation of §1's +26 % task loss. REFUTE: the loss-scale story does **not** carry to gradients — the AUC drop has another cause and `lambda_rec` is the wrong lever; do not tune it. |
| **D2-b** | `cos(g_kip_rec, g_task)` | ≤ −0.10 **conflict** · \|cos\| < 0.10 **orthogonal** · ≥ +0.10 **aligned** | Conflict → lowering `lambda_rec` trades one objective for the other; the clean comparison is the detach arm (§5 **C**), not a weight sweep. Orthogonal → KIP spends trunk capacity without fighting; normalization + a principled weight is the repair. Aligned → gradient conflict is **not** the mechanism; look at the shift (**C24**) and at `L_kin`, not at the loss scale. |
| **D2-c** | `rho` at stage-1 end vs stage-2 end | reported, no bar | Whether the trunk was captured immediately or drifted into it over 2,040 steps. Informs whether a warm-up on `lambda_rec` is even coherent. |

`--num-batches 8` gives eight independent draws from the real epoch-0 order; the
JSON carries every per-batch record, so report the **mean and the spread**, not
the mean alone. n=8 batches on **one seed** — this is a mechanism probe, not an
effect size. Do not put a t-interval on it.

---

## 4. Decision table

Read D1-a/D1-b first; they can end the question without D2 mattering.

| D1-a | D1-b | D2-a | D2-b | verdict | next |
|---|---|---|---|---|---|
| STOP | — | — | — | The flow target is noise at this capacity. | §5 **D**: write the negative result. `lambda_rec` is not the story. |
| PASS | REFUTE | — | — | PMG fits item identity, not motion. | §5 **D**, and KIP's premise on T2 is refuted independently of weighting. |
| PASS | PASS | REFUTE | — | Loss share ≠ gradient share. | Stop tuning. Re-open the AUC drop against **C24** (the fixed ~50 % shift) instead. |
| PASS | PASS | CONFIRM | conflict | Trunk capture **and** direction conflict. | §5 **C** (detach arm) as the decisive control, then **A**. A weight sweep alone cannot separate them. |
| PASS | PASS | CONFIRM | orthogonal | Trunk capture, no direction fight. | §5 **A** (z-score `e_O`, new cache version) with `lambda_rec` re-derived, not swept. |
| PASS | PASS | partial | any | Mechanism present, not dominant. | Report both numbers; decide with the user. No arm fires automatically. |

---

## 5. The repair options this gates (none is authorized yet)

* **A — z-score `e_O`** per dimension from train-split statistics. Fixes the root:
  the loss returns to O(1) and `lambda_rec` becomes comparable across corpora.
  **Fires C2**: a new cache version, and every KIP-on number re-measured — stage 1
  and stage 2, three seeds. Never overwrite `flow/v1`.
* **B — lower `lambda_rec`** to ≈ `1/V`. One flag, no cache change, runnable
  immediately. But the weight stays corpus-specific, and adopting it off a Δ is
  lesson **14** unless the value is *derived* from D1-c rather than searched.
* **C — `detach()` `v^t` before PMG.** Isolates the trunk from `L_KIP_rec`
  exactly. Cheap and decisive as a **control**, but it removes KIP's whole
  premise — the remainder is the v3 **A2** arm (fixed shift, no flow), already
  measured indistinguishable from full KIP on MSAD. Diagnostic only.
* **D — conclude.** KIP v1 does not help on T2; record it as the project's third
  negative result and stop spending seeds on it.

---

## 6. Risks and known traps

1. **The checkpoints may not exist.** `results.json` records
   `/content/p4/runs/s2024/stage2_kip_on/checkpoint_last.pt` — a **VM-local**
   path, and that runtime is the kind that ate the Phase-0 frame census. **Verify
   the checkpoints are on Drive before planning D2**; if they are gone, D2 can
   still run at initialization, but the end-of-training point — the one that
   produced the AUC — is unrecoverable without a re-run.
2. **AMP vs fp32.** The probe runs fp32 where training used AMP. Ratios and
   cosines are scale-free, so a global loss-scale factor cancels; the *absolute*
   norms are not comparable to anything AMP produced.
3. **`--num-batches 8`, one seed.** A mechanism probe. Report the spread.
4. **Do not "fix" a D1-d failure by re-deriving the projection.** A mismatch
   means the cache and the projection that built it disagree; re-deriving hides
   it. `flow_projection.npz` is the artifact of record.
5. **C24 stands regardless.** Every KIP-on arm on `main` is a fixed ~50 % channel
   shift — the gate MLP never receives a gradient and there is no
   `kip.gate_type` on this branch. Whatever D1/D2 return, do not write
   "motion-gated" of these runs.

---

## 7. What changed in the tree to make this runnable

| file | change |
|---|---|
| `core/constants.py` | `FLOW_STAT_NAMES` — the 23 stats in emission order, derived from `FLOW_ANGLE_BINS` so it cannot drift. |
| `core/eda/features.py` | `_MomentAccumulator` (streaming moments, within-item pool kept apart), `_raw_energy_share`, and `flow_stats`'s new `target` block. |
| `core/eda/report.py` | §4.3 energy table + §4.3.1 baselines; two verdicts at `FLOW_TARGET_SCALE_WARN_MSE = 4.0` and `FLOW_TARGET_BETWEEN_ITEM_WARN = 0.5`. |
| `core/train.py` | `build_trainer()` extracted from `main()` — same wiring, one copy. `main()`'s CLI is unchanged. |
| `core/tools/grad_probe.py` | new: the D2 CLI. |
| `core/tests/test_eda.py` | 7 tests: target baselines, between/within split, energy share, projection round-trip, two verdict tests. |
| `core/tests/test_grad_probe.py` | 10 tests: term re-sum guard, reach/zero per group, `lambda_rec` linearity, KIP-off and stage-1 refusals, CLI artifacts. |

| `colab/DADA2000Origin/diag_kip_loss_scale.ipynb` | the runbook: staging, preflight, D1, D2, the decision table, the C17 manifest. |

Suite: **563 collected, 563 pass** (was 546). `ruff`, `mypy`, `bandit` clean.

**Dry-run 2026-09-18** on the Phase-4 synthetic fixture, both paths end to end: the EDA
report emits §4.3.1 and the notebook's parser reads it (`Z` 1.0818 / `V` 0.3654 /
`W` 0.3525, projection ratio 0.9473); `grad_probe` returns `rho` 0.035 and
`cos(kip_rec, task)` +0.037 there. Those are **fixture numbers with no pixel units in the
target** — they say the instruments run and discriminate, nothing about T2.


---

## Appendix A — the measurements (2026-09-20), scored against §2/§3

Record: `outputs/v1/DADA2000_orig_diag_kip_loss_scale/` —
`d1_flow_target/eda_report.{json,md}`,
`d2_grad/s{2024,2025,2026}_{stage1,stage2_kip_on}/grad_probe.{json,md}`,
`diag_manifest.json`. Environment: torch **2.11.0+cu128**, transformers
**4.56.2**, Python **3.13.15**, `commit: UNKNOWN` (Drive copy — `core_sha256`
`b85331c5…c16b924` stands in), notebook
`colab/DADA2000Origin/diag_kip_loss_scale.ipynb`.

### A.1 D1

`Z` **71.146** · `V` **31.642** · `W` **16.206** · `K` **11.620**
(per seed 11.317 / 11.959 / 11.583) · stage-1 plateau **20.542**.

| id | measured | bar | verdict |
|---|---:|---|---|
| **D1-a** `R²_global = 1 − K/V` | **0.6328** | ≥ 0.20 PASS | **PASS** |
| **D1-b** `K` vs `W` (`R²_item`) | 11.620 < 16.206 (**0.2830**) | `K < W` PASS | **PASS** |
| **D1-c** `V` | **31.642** | ≥ 4.0 confirms overweight | **CONFIRM**, ~32× an O(1) term |
| **D1-d** projection round-trip | **0.9239** | ∈ [0.9, 1.1] | **passes**, lower half of the band |
| **D1-e** top `raw_energy_share` | **`mag_max` 83.0 %** (mean 27.19, sd 22.72) | reported | the §1 hypothesis was right |

Equalizing weight `1/V` = **0.031603**. Between-item share **0.4878**, i.e. the
`FLOW_TARGET_BETWEEN_ITEM_WARN = 0.5` trip-wire was missed **by 0.012** — the
warning did not fire, which is not the same as clearing it.

### A.2 D2

Three seeds × two points, 8 batches of 64, `cuda`, fp32, no optimizer step.

| point | `rho_all` | `rho_kip_rec` (sd) | `cos(g_kip_rec, g_task)` | \|g_task\| |
|---|---:|---:|---:|---:|
| s2024 stage-1 | 11.055 | 10.581 (9.54) | +0.00017 | 4.906 |
| s2025 stage-1 | 12.409 | 11.915 (9.87) | +0.00596 | 7.237 |
| s2026 stage-1 | 11.434 | 11.184 (7.81) | −0.00027 | 6.209 |
| **s2024 stage-2** | **3.925** | 3.892 (1.94) | +0.00015 | 10.068 |
| **s2025 stage-2** | **3.027** | 2.997 (1.56) | −0.00211 | 12.461 |
| **s2026 stage-2** | **2.361** | 2.340 (1.66) | −0.00108 | 23.607 |

**D2-a** `rho_end` = **3.105** ≥ 1.0 → **CONFIRM**.
**D2-b** `cos_end` = **−0.0010**, |cos| < 0.10 → **ORTHOGONAL**.
**D2-c** 11.633 → 3.105: captured at step 1 and decaying — **but partly because
the denominator grew**; `|g_kip_rec|` itself went 51.9 → 39.2, 86.2 → 37.3,
69.4 → 55.2. A `lambda_rec` warm-up is therefore *coherent* but would be
chasing a ratio whose denominator is not stationary.

Loss share at stage 2: `kip_rec` **88.2–91.0 %** of `total`, reproducing the
`metrics.jsonl` 92.6–93.0 % of §1 to within the batch draw.

### A.3 The decision, and what was NOT done

`PASS · PASS · CONFIRM · orthogonal` → the §4 row **"Trunk capture, no direction
fight"** → **§5 Option A**: z-score `e_O` from train-split statistics into a
**new cache version**, `lambda_rec` re-derived as `1/V`, **not swept**.
§5 **C** (detach) drops to diagnostic-only: there is no direction conflict for it
to separate. **Option A is licensed by the table, not authorized by anyone.**
No `core/` file was changed on the strength of this appendix.

### A.4 Defects found in our own artifacts

1. `d1_flow_target/eda_report.md` prints `score head kernel 9 · MIL top-k pct 16`
   — `core.tools.eda` defaults, **not** T2's config (3 and 5). `--sections
   features` reads neither, so **no number is affected**; the header misleads a
   later reader. Same file records `Data dir: /content/p5/stage/corpus`, a
   VM-local path.
2. **D1-d must be re-printed on any new cache** Option A builds. At 0.9239 the
   current cache is 7.6 % below 1.0, ≈1.3× the expected `1/sqrt(256)` deviation.
3. The §3 command block probes stage-1 checkpoints under the stage-2 config; that
   path crashed on the day and produced **lesson C36** (`warm_start_model`, not
   `Trainer.load_checkpoint`). The numbers above are from the repaired run.
