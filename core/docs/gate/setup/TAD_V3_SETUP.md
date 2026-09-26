# KAT-VAD v3 — TAD training + evaluation runbook

**Written 2026-09-02, against branch `v3`.** The experiment layer on top of
`core/docs/TAD_SETUP.md`, which owns everything about getting TAD onto disk
(unzip, frame-order gate, labels, CLIP, RAFT, KNN, sizing). **Do §1–§10 of that
document first, including Gate T0.** Nothing here is meaningful without it.

Sibling runbook: `MSAD_DOTA_V3_SETUP.md`. This document deliberately mirrors its
structure and its arm names, because the whole point of the TAD campaign is that
the two are compared row for row.

> **The two code prerequisites shipped 2026-09-02** (`TAD_SETUP.md` §0.3):
> `core.data.tad --with-train-split` builds the weakly-supervised train split,
> and `core.flow.raft_extract --frames-dir` builds `e_O` from frame folders.
> Every arm below runs against branch `v3`, not this one. The train path is
> covered by `core/tests/test_tad.py::TestTadTrains`, so a TAD arm cannot fail
> for a *data* reason that was never exercised — but **on `main` that test
> exercises the v1 gate only**, because `kip.gate_type` does not exist here.

---

## 0. What this campaign is for

Not a headline. A **replication**.

`core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md` (2026-09-01) settled the
attribution on MSAD-full: a fixed 50 % channel shift with no flow, no PMG head
and no KIP losses (**A2**) is statistically indistinguishable from full v1 KIP on
zero-shot DoTA — Δ(A2 − V1) = +0.0109, t95 [−0.0588, +0.0805], 6/6 metrics
include zero — while A2 − A0 = **+0.1025 ± 0.0350**. The `rank` gate (**A1**) is
the *worst* KIP arm: A1 − A2 = −0.0683, CI [−0.0790, −0.0580]. No component of
KIP has been attributed a positive contribution.

TAD's job is to answer one question on a **second training corpus**:

> Does the smoother result replicate off MSAD? Specifically: is
> `A2 − A0 > 0`, `A1 − A2 ≤ 0`, on a model trained on TAD?

**Pre-registered prediction (2026-09-02, before any TAD arm is trained):**
A2 − A0 is positive on zero-shot DoTA and A1 − A2 is ≤ 0, reproducing the MSAD
ordering. Write this into the run notes **now**, not after the eval.

Three outcomes, all publishable, none of them "KIP works after all":

| Result | Reading |
|---|---|
| Ordering replicates | The smoother finding is corpus-independent. The restated claim in `activeContext.md` holds and gets stronger. |
| Ordering inverts (A1 > A2 on TAD) | Corpus dependence — genuinely new, and the only surviving route for a motion mechanism. Would need its own 3 seeds before anyone says it out loud. |
| Everything is null on TAD | TAD's train split (~410 clips, one anomaly class) may be too small or too homogeneous to move a transfer number. A bounded null, reported as one. |

**Do not tune anything on a TAD number** (lesson 14). The mechanism is what is
under test.

---

## 1. Preconditions

### 1.1 Code, environment, data

`TAD_SETUP.md` §2 (install + env vars) and §3–§9 (ingest, labels, caches). Then:

```bash
!python -c "from core.kip import gate_shift; print(gate_shift.GATE_TYPES)"
# expect: ('rank', 'mlp_frozen', 'mlp_ste', 'constant')
```

### 1.2 What must exist before §5

```bash
%%bash
for p in "$KATVAD_DATA_ROOT/TAD/labels_train.json" \
         "$KATVAD_DATA_ROOT/TAD/frame_labels_test.json" \
         "$KATVAD_DATA_ROOT/TAD/defs.json" \
         "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
         "$KATVAD_CACHE_ROOT/flow/v1/TAD" \
         "$KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz" \
         "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
         "$KATVAD_DATA_ROOT/DoTA/labels_s8"; do
  n=$(ls "$p" 2>/dev/null | wc -l); echo "$n  $p"
done
```

The two DoTA rows are the **zero-shot transfer** target and are reused from the
MSAD campaign untouched — v3 changed model code only, no transform, no stride,
no pooling, so lesson **C2** does not fire.

### 1.3 Two facts that decide every command below

- **`data.dataset=TAD`** resolves to verbalizer key `tad`
  (`core/data/definitions.py:DATASET_NAME_TO_ABBR`), whose taxonomy is exactly
  `["Normal", "Car Accident"]`. `C = 2`, so `H_mul` is near-degenerate on TAD —
  the multi-class head is close to a second binary head. Expect `mul` loss to
  behave differently than on MSAD-full's larger taxonomy, and do not read
  anything into it.
- **`data.is_egocentric=false`** (the default). TAD is fixed-camera surveillance.
  Setting it true selects `theta_ego` / `delta_m_ego`, which are tuned for
  DoTA-style ego footage and would change DVS behaviour for no reason.

### 1.4 Sizing — fill this in before writing any command

`TAD_SETUP.md` §10 computes `num_epochs` from `2 × num_abnormal_train`. Every
block below writes it as `E`. **Set `E` once, use the same value in every arm and
both stages.** An arm trained for a different number of steps is not a control.

```python
E = ...   # from TAD_SETUP.md §10
```

---

## 2. The arm matrix on TAD

Identical to `MSAD_DOTA_V3_SETUP.md` §2.2 — same names, same flags, so the two
campaigns' rows line up. What changes is the training corpus and the in-domain
benchmark.

| Arm | Flags added to the stage-2 command | Purpose | Needs |
|---|---|---|---|
| **A0** | `--set kip.enabled=false` | KIP-off trunk. The arm every Δ subtracts from | — |
| **A2** | `--set kip.gate_type=constant --set kip.const_shift_ratio=0.5 --set kip.disable_pmg=true --set loss.lambda_rec=0 --set loss.lambda_align=0 --set kip.use_lkin=false` | **Plain-TSM control.** A fixed 50 % channel shift and nothing else. **All six flags required** | — |
| **A1** | `--set kip.gate_type=rank` | The v3 model | stage 1 |
| **A2b** | `--set kip.gate_type=constant --set kip.const_shift_ratio=0.5` (+ `--flow-dir`) | Same smoother, PMG genuinely trained against real flow. Separates "the smoother did it" from "the aux losses did it" | stage 1 |
| **A3** | `--set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm` | The v1 bridge to every `RESULTS_*.md` number | stage 1 |
| **A4** | `--set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm` | Pure gradient intervention; forward bit-identical to A3 | stage 1 |

**A0 and A2 need neither a flow cache nor a stage 1.** That is the whole
decisive comparison, and it is the cheapest thing in the program. §6 exploits it.

> **`disable_pmg=true` does not disable the PMG head.** It sets
> `require_flow=False`, so the dataset hands back **zero** flow rows
> (`core/train.py:593`, `core/data/dataset.py:103`) while `L_KIP_rec` and
> `L_KIP_align` stay in the objective — training the 311,808-param PMG head to
> regress zeros. Zeroing `loss.lambda_rec` / `loss.lambda_align` and
> `kip.use_lkin` is what actually removes the pathway (lesson **14**).

### 2.1 What raises, so you recognise it

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Training set needs both classes: 0 normal, 0 abnormal` | `labels_train.json` is `{}` | you preprocessed **without** `--with-train-split`. `TAD_SETUP.md` §5.2 |
| `ValueError: Missing flow cache .../TAD/{id}.npy` | KIP on, `disable_pmg=false`, no flow | run `TAD_SETUP.md` §8. Do *not* reach for `require_flow=false` |
| `ValueError: ... neither of the split directories ... appears in its path` | a frame folder sits outside `abnormal/` and `normal/` | fix the layout; the directory is TAD's only train-split label |
| `KeyError: ...kip.enabled is true but kip.gate_type is missing` | pre-v3 KIP-on YAML | add `gate_type:`, or drop `--config` and use `--set` |
| `ValueError: kip.gate_signal=... is meaningless for kip.gate_type='rank'` | signal passed to a parameter-free gate | remove the flag |
| `KeyError: This checkpoint contains N kip.shift.mlp.* tensors ...` | scoring an `mlp_*` ckpt under `rank`/`constant` | build with the gate that trained it |
| `ValueError: Flow/feature length mismatch for {id}` | flow and CLIP caches built at different strides | rebuild one at the other's stride; nothing on disk records stride (**C2**) |

---

## 3. Training

Output dirs are the contract with §4. Train an arm elsewhere and §4's `--ckpt`
paths will not find it.

| Arm | Train dir (`$OUT` = `$KATVAD_OUTPUT_ROOT`) | Stage 1? | Flow? |
|---|---|---|---|
| **A0** | `$OUT/TAD_kipoff_s$S/stage2` | no (raises) | no |
| **A2** | `$OUT/TAD_constant_s$S/stage2_kip_on` | **no** | **no** |
| **A1** | `$OUT/TAD_rank_s$S/stage2_kip_on` | §3.2a | yes |
| **A2b** | `$OUT/TAD_constant_pmg_s$S/stage2_kip_on` | §3.2a | yes |
| **A3** | `$OUT/TAD_mlp_frozen_s$S/stage2_kip_on` | §3.2b | yes |
| **A4** | `$OUT/TAD_mlp_ste_s$S/stage2_kip_on` | §3.2b | yes |

Seeds `2024 2025 2026`, matching every other campaign. **One experiment = one
`--output-dir`** (`metrics.jsonl` appends; a restart into the same dir silently
duplicates rows — that happened on MSAD A3 and had to be deduped by hand).

### 3.1-A2 — plain-TSM control ⭐ RUN THIS FIRST

No stage 1, no flow, no KIP losses — just `labels_train.json` (`TAD_SETUP.md`
§5.2) and the CLIP cache. The cheapest run in the program.

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024 ; E=<from §1.4>

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=TAD \
  --set data.is_egocentric=false \
  --set train.num_epochs=$E --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.gate_type=constant --set kip.const_shift_ratio=0.5 \
  --set kip.disable_pmg=true \
  --set loss.lambda_rec=0 --set loss.lambda_align=0 --set kip.use_lkin=false \
  --data-dir  "$KATVAD_DATA_ROOT/TAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD_constant_s$S/stage2_kip_on"
```

### 3.1-A0 — KIP-off baseline

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024 ; E=<from §1.4>

python -m core.train \
  --set train.stage=2 --set train.amp=true --set data.dataset=TAD \
  --set data.is_egocentric=false \
  --set train.num_epochs=$E --set train.checkpoint_every_steps=100 \
  --set train.seed=$S \
  --set kip.enabled=false \
  --data-dir  "$KATVAD_DATA_ROOT/TAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD_kipoff_s$S/stage2"
```

Cold by construction: stage 1 requires KIP (`core/train.py:187`) and trains
**only** `kip.*` parameters, so warm arms carry no trunk advantage and cold A0 is
a fair comparator. No gate flags — a KIP-off config makes no claim about the gate.

### 3.2 Stage 1 — KIP warm-up

Stage 1's loss is `lambda_rec·L_KIP_rec + lambda_align·L_KIP_align`, neither of
which reads `v^k`, so **no gate type receives gradient in stage 1**. The run is
gate-independent in everything but its checkpoint key layout — hence **two**
stage-1 runs per seed, not four:

| Stage-1 run | Serves | Why |
|---|---|---|
| `gate_type=rank` | **A1, A2b** | parameter-free gates share a key layout, so `constant` loads it |
| `gate_type=mlp_frozen` | **A3, A4** | both own `kip.shift.mlp.*`; `mlp_ste` loads it |

#### 3.2a Parameter-free family (serves A1, A2b)

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024 ; E=<from §1.4>

python -m core.train \
  --set train.stage=1 --set data.dataset=TAD --set data.is_egocentric=false \
  --set train.num_epochs=$E --set train.seed=$S \
  --set kip.gate_type=rank \
  --data-dir  "$KATVAD_DATA_ROOT/TAD" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/TAD" \
  --knn-cache "$KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz" \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD_rank_s$S/stage1"
```

#### 3.2b MLP family (serves A3, A4)

Same command with `--set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm`
and `--output-dir "$KATVAD_OUTPUT_ROOT/TAD_mlp_frozen_s$S/stage1"`.

#### 3.2c Convergence check — a **stop**, not a warning

Spec v3 §8's `assert stage1_final(L_KIP_rec) < τ_rec` is **not implemented**.
No gradient reaches `ê_O` through the shift under `rank`/`mlp_frozen`/`constant`,
so a stage-2 arm will happily rank a badly-trained `ê_O` and hand you a plausible
number.

```python
import json, pathlib, os
FAM, S = 'rank', 2024          # or 'mlp_frozen'
p = (pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])
     / f'TAD_{FAM}_s{S}' / 'stage1' / 'metrics.jsonl')
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
rec  = [r['kip_rec'] for r in rows if 'kip_rec' in r]
n    = max(1, len(rec) // 10)
first, last = sum(rec[:n])/n, sum(rec[-n:])/n
print(f'first {n} mean {first:.3f} | last {n} mean {last:.3f} | ratio {last/first:.2f}')
assert last < 0.6 * first, (
    f'STOP: L_KIP_rec did not roughly halve ({first:.3f} -> {last:.3f}). '
    'This arm has no result until stage 1 converges.')
```

The MSAD/v1 reference trajectory was 19.2 → 9.7 with a plateau from ~step 200.
TAD's absolute scale will differ — its flow statistics are a different
distribution — so read the **ratio**, not the value. A last-window mean that has
not roughly halved means the arm is blocked, not weak (lesson **14**).

### 3.3 Remaining arms

A1, A2b, A3 and A4 are `MSAD_DOTA_V3_SETUP.md` §5.2-A1 / -A2b / -A3 / -A4
verbatim, with four substitutions:

```
--set data.dataset=MSAD-full        ->  --set data.dataset=TAD --set data.is_egocentric=false
--set train.num_epochs=125          ->  --set train.num_epochs=$E
--data-dir  .../MSAD                ->  --data-dir  "$KATVAD_DATA_ROOT/TAD"
--clip-dir  .../clip/MSAD_ncc       ->  --clip-dir  "$KATVAD_CACHE_ROOT/clip/TAD_ncc"
--flow-dir  .../flow/v1/MSAD/MSAD-full -> --flow-dir "$KATVAD_CACHE_ROOT/flow/v1/TAD"
--knn-cache .../knn/MSAD_ncc/...    ->  --knn-cache "$KATVAD_CACHE_ROOT/knn/TAD_ncc/knn_cache.npz"
--output-dir $OUT/MSAD_<gate>_s$S/… ->  --output-dir "$KATVAD_OUTPUT_ROOT/TAD_<gate>_s$S/…"
```

A1 and A2b warm-start from `TAD_rank_s$S/stage1/checkpoint_last.pt`; A3 and A4
from `TAD_mlp_frozen_s$S/stage1/checkpoint_last.pt`. **A4 shares A3's stage 1
deliberately** — that removes the one difference that is not the gradient, which
is the entire point of `A4 − A3`.

### 3.4 Resume, and the manual manifest

- Dead session: same command with `--init-weights ...` replaced by
  `--resume "<output-dir>/checkpoint_last.pt"`. Mutually exclusive; resume
  carries the warm-started weights forward anyway.
- `--stop-after-epochs N` caps one invocation without shrinking the LR horizon.
- **Lesson 15:** checkpoints pickle `np.random.get_state()`. A Colab runtime that
  drifts to a different numpy major makes every checkpoint on Drive unloadable
  *before reaching a tensor*. Recover with the `find_class` unpickler in
  `COLAB.md` §A4.0; do not pin `numpy<2` away.
- **Lesson 17 — the run manifest is not built.** `config.yaml` records
  `gate_type` but not `--init-weights`, `--data-dir`, `--clip-dir`,
  `--knn-cache`, the commit, or `sys.argv`. Write it by hand, once per run:

```python
import json, os, subprocess, pathlib
S, RUN, ARM, GATE = 2024, 'constant', 'A2 plain-TSM control', 'constant'
run = pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])/f'TAD_{RUN}_s{S}'/'stage2_kip_on'
(run/'run_manifest.json').write_text(json.dumps({
    'corpus': 'TAD', 'arm': ARM, 'gate_type': GATE, 'const_shift_ratio': 0.5,
    'disable_pmg': True, 'lambda_rec': 0.0, 'lambda_align': 0.0, 'use_lkin': False,
    'seed': S, 'num_epochs': None, 'init_weights': None,
    'clip_dir': f"{os.environ['KATVAD_CACHE_ROOT']}/clip/TAD_ncc",
    'flow_dir': None,
    'knn_cache': f"{os.environ['KATVAD_CACHE_ROOT']}/knn/TAD_ncc/knn_cache.npz",
    'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],
                  cwd='/content/drive/MyDrive/Thesis-V3/kat-vad').decode().strip(),
    'prediction': 'A2 - A0 > 0 and A1 - A2 <= 0 on zero-shot DoTA (pre-registered 2026-09-02)',
}, indent=2))
```

---

## 4. Evaluation

Every trained arm gets **two** evals from the *same* `checkpoint_last.pt`:

| Eval | Benchmark | `--data-dir` | `--clip-dir` | `--score-norm auto` resolves to |
|---|---|---|---|---|
| **in-domain** | TAD test (100 clips, ~40 % normal) | `$KATVAD_DATA_ROOT/TAD` | `clip/TAD_ncc` | **raw** |
| **zero-shot** | DoTA val (1,402 clips, ~all abnormal) | `$KATVAD_DATA_ROOT/DoTA/labels_s8` | `clip/DoTA_s8_ncc` | **min-max** |

**Those are two different metrics.** Never put a raw-pooled TAD number and a
min-max-pooled DoTA number in the same column, and always report `auc_macro`
beside the micro number (lesson **12**).

Two rules before you copy any block:

1. **The model is built from the CLI, not from the checkpoint.** The **gate**
   flags must repeat on *every* eval command. The **loss** flags must not —
   `lambda_rec`, `lambda_align`, `use_lkin` shape training only, and
   `disable_pmg` only controls whether the dataset reads flow.
2. **`rank` and `constant` checkpoints are key-identical.** Scoring an A1
   checkpoint under `--set kip.gate_type=constant` loads **cleanly**, raises
   nothing, and hands you a publishable number for a model never trained that
   way. `load_kip_state_dict` keys its check on `kip.shift.mlp.*`, and neither
   gate has any. **§4.2 is the only thing between you and that number.**

### 4.1 The two eval blocks (A2 shown; substitute per §4.3)

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
S=2024
CKPT="$KATVAD_OUTPUT_ROOT/TAD_constant_s$S/stage2_kip_on/checkpoint_last.pt"
GATE="--set kip.gate_type=constant --set kip.const_shift_ratio=0.5"

# --- TAD, in-domain ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=TAD \
  --data-dir "$KATVAD_DATA_ROOT/TAD" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD_constant_s$S/eval_tad"

# --- DoTA, zero-shot transfer ---
python -m core.evaluate --ckpt "$CKPT" $GATE \
  --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_from_TAD_constant_s$S/eval_dota"
```

`DoTA_from_TAD_*` in the output path is deliberate: the MSAD campaign already
owns `DoTA_constant_s$S`. Two different training corpora scoring the same
benchmark must never share an output dir.

### 4.2 Verify the gate you actually scored — MANDATORY

Run after **every** eval block. `--dump-kip-diag` defaults on with
`--save-scores`, so the diagnostics are already in each `.npz`.

```python
import numpy as np, pathlib, os

EXPECTED = {'rank': (100, 128), 'mlp_frozen': (0, 8),
            'mlp_ste': (0, 8), 'constant': (0, 0)}

def check(eval_dir: str, gate_type: str, n: int = 200) -> None:
    d = pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT']) / eval_dir / 'scores'
    files = sorted(d.glob('*.npz'))[:n]
    assert files, f'no scores in {d}'
    empty = [p.name for p in files if p.stat().st_size == 0]
    assert not empty, f'0-byte score files (lesson 11b): {empty[:3]}'
    spans, means, ratios = [], [], []
    for p in files:
        z = np.load(p)
        assert 'kip_s' in z, f'{p.name}: no kip_s — KIP-off arm, or diag was off'
        s = z['kip_s'].astype(np.int32)
        spans.append(s.max() - s.min()); means.append(s.mean())
        ratios.append(float(z['kip_gate_ratio'].mean()))
    lo, hi = EXPECTED[gate_type]; span_mean = float(np.mean(spans))
    print(f'{eval_dir}: span mean {span_mean:.1f} (min {min(spans)}, max {max(spans)}) '
          f'of 128 | s mean {np.mean(means):.1f} | ratio mean {np.mean(ratios):.3f}')
    assert lo <= span_mean <= hi, (
        f'GATE MISMATCH: {eval_dir} scored as {gate_type!r} but span {span_mean:.1f} '
        f'is outside [{lo}, {hi}]. This eval is VOID — do not read its AUC.')

S = 2024
for gate, run in [('constant', 'constant'), ('rank', 'rank'),
                  ('constant', 'constant_pmg'), ('mlp_frozen', 'mlp_frozen'),
                  ('mlp_ste', 'mlp_ste')]:
    for d in (f'TAD_{run}_s{S}/eval_tad', f'DoTA_from_TAD_{run}_s{S}/eval_dota'):
        if (pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])/d/'scores').is_dir():
            check(d, gate)
```

| Arm | Expected span of 128 | `s` mean | If otherwise |
|---|---|---|---|
| `constant` (A2, A2b) | exactly **0** | **64** at `r = 0.5` | a flag did not take |
| `rank` (A1) | near **128 on every clip** | spread | the rank gate is not working — stop |
| `mlp_frozen` (A3) | **0–4** | ≈ 58–69 | you are not running the gate you think |
| `mlp_ste` (A4) | **0–4** | ≈ 58–69 | forward is identical to A3 by construction |
| `enabled=false` (A0) | *no `kip_*` keys at all* | — | wrong checkpoint scored |

**A span of 0 on an arm you believe is `rank` is the failure this section exists
for**: the load succeeded and the AUC is meaningless.

**Windowing caveat:** the rank is computed *within* a `data.max_vis_len` window,
so `s_t` resets at each boundary. That is a real property of the method, not a
logging artifact — state it in any write-up.

### 4.3 Per-arm substitutions

| Arm | `--ckpt` from | Eval dirs | Gate flags at eval |
|---|---|---|---|
| **A0** | `TAD_kipoff_s$S/stage2` | `TAD_kipoff_s$S/eval_tad`, `DoTA_from_TAD_kipoff_s$S/eval_dota` | `--set kip.enabled=false` |
| **A2** | `TAD_constant_s$S/stage2_kip_on` | `TAD_constant_s$S/…` | `--set kip.gate_type=constant --set kip.const_shift_ratio=0.5` |
| **A1** | `TAD_rank_s$S/stage2_kip_on` | `TAD_rank_s$S/…` | `--set kip.gate_type=rank` |
| **A2b** | `TAD_constant_pmg_s$S/stage2_kip_on` | `TAD_constant_pmg_s$S/…` | same as A2 |
| **A3** | `TAD_mlp_frozen_s$S/stage2_kip_on` | `TAD_mlp_frozen_s$S/…` | `--set kip.gate_type=mlp_frozen --set kip.gate_signal=flow_norm` |
| **A4** | `TAD_mlp_ste_s$S/stage2_kip_on` | `TAD_mlp_ste_s$S/…` | `--set kip.gate_type=mlp_ste --set kip.gate_signal=flow_norm` |

`const_shift_ratio` is range-checked only (`core/config.py:110`) — passed beside
`rank` it is silently accepted and unused. `gate_signal` beside `rank`/`constant`
**raises**; it is the one gate flag the config protects you from.

### 4.4 Reference arms

Two, both gate-independent, both run once:

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad
# T0 — LaGoVAD's released trunk on TAD, zero-shot (TAD_SETUP.md §6)
python -m core.evaluate --baseline-ckpt "$KATVAD_CKPT_ROOT/best.ckpt" \
  --set kip.enabled=false --set data.dataset=TAD \
  --data-dir "$KATVAD_DATA_ROOT/TAD" --clip-dir "$KATVAD_CACHE_ROOT/clip/TAD_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/TAD_gate_t0/gate_t0"

# gate_a — the same trunk on DoTA; reuse the MSAD campaign's if it exists
python -m core.evaluate --baseline-ckpt "$KATVAD_CKPT_ROOT/best.ckpt" \
  --set kip.enabled=false --set data.dataset=DoTA \
  --data-dir "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --score-norm auto --save-scores \
  --output-dir "$KATVAD_OUTPUT_ROOT/DoTA_gate_a/gate_a"
```

Gate against **the released checkpoint's own number**, not the paper's printed
89.56 / 62.60 (lesson **8b**).

### 4.5 Rescore and manifest

```bash
%%bash
for d in "$KATVAD_OUTPUT_ROOT"/TAD_*_s*/eval_tad \
         "$KATVAD_OUTPUT_ROOT"/DoTA_from_TAD_*_s*/eval_dota \
         "$KATVAD_OUTPUT_ROOT"/TAD_gate_t0/gate_t0; do
  [ -d "$d/scores" ] && python -m core.tools.rescore --run-dir "$d"
done
```

`core/evaluate.py` writes no config, so an eval dir does not record which gate
produced it. Write an `eval_manifest.json` in each — `MSAD_DOTA_V3_SETUP.md`
§6.11's cell, with `TAD` paths and the §4.2 span pasted in.

**Before any Δ**, assert the score-file count against `results.json:num_videos`
in every arm (lesson **11b**): `evaluate --save-scores` is not atomic, and one
0-byte `.npz` perturbs *every* paired Δ in the study, not one number.

```python
import json, os, pathlib
OUT = pathlib.Path(os.environ['KATVAD_OUTPUT_ROOT'])
for d in sorted(OUT.glob('*_s*/eval_*')):
    r = d / 'results.json'
    if r.exists():
        n_files = len(list((d/'scores').glob('*.npz'))) if (d/'scores').is_dir() else 0
        n_rep = json.load(open(r)).get('num_videos')
        flag = '' if n_files == n_rep else '   <-- MISMATCH'
        print(f'{d.relative_to(OUT)}: scores={n_files} results={n_rep}{flag}')
```

---

## 5. Analysis and decision rules

Use `DOTA_EVAL.md` §4's analysis cell unchanged for the DoTA side — paired
bootstrap over clips, per-clip win/loss, ego/non-ego split, per-class deltas.
Report Δ per seed, then the **seed-level t-interval over n = 3**. Do not report
`3 seeds × 3 metrics × 2 controls` as 18 CIs; there are 3 independent
replications.

| Comparison | What it answers |
|---|---|
| **A2 − A0** (DoTA) | **The replication.** Does the fixed 50 % smoother move zero-shot transfer when trained on TAD, as it did on MSAD (+0.1025 ± 0.0350)? |
| **A1 − A2** (DoTA) | The only number that could support a motion-gating claim. On MSAD it was **−0.0683**, CI excluding zero |
| **A2 − A0** (TAD, in-domain) | On MSAD the in-domain effect was a bounded null (+0.0036 ± 0.0076). Does TAD agree? |
| **A1 − A0** | The v3 headline, uninterpretable without the two rows above |
| **A2b − A2** | What the auxiliary KIP losses buy with the gate fixed (MSAD: **−0.0105**, CI excluding zero) |
| **A3 − A0** | The v1 bridge |
| **A4 − A3** | Pure gradient effect; forward bit-identical |

Standing rules:

- **A small test set has wide intervals.** TAD test is **100 clips**. A Δ of
  ±0.02 in-domain will not clear the noise at n = 3. Say "bounded null below X",
  never "costs nothing".
- **In-domain TAD is not comparable to 89.56.** `TAD_SETUP.md` §0.1. Gate T0 is
  the only row in any table that may sit next to it.
- **An A/B run under a known-open precondition defect measures the defect**
  (lesson **14**). If §3.2c failed, the arm has no number — report it blocked.
- **Do not tune on any Δ.** The mechanism is what is under test.

Record outcomes in a new `core/docs/v3/RESULTS_V3_TAD.md`. Do not retro-edit
`RESULTS_V3_GATE_ATTRIBUTION.md` — it was true for what ran on MSAD; add the
cross-corpus qualifier at the point of next citation.

---

## 6. Order of work

```
--- data, no code changes needed -------------------------------------------
[ ] TAD_SETUP §2–§4   session, unzip, THREE ingest gates (esp. §4.3 frame order)
[ ] TAD_SETUP §5.1    test-split labels (works today)
[ ] TAD_SETUP §7      CLIP features, test ids first, then all 510
[ ] TAD_SETUP §6      GATE T0 — reproduce the zero-shot number. STOP AND READ IT.

--- the train split ---------------------------------------------------------
[ ] TAD_SETUP §5.2    train + test labels (--with-train-split); record the counts
[ ] TAD_SETUP §9      DVS KNN cache
[ ] §1.4              compute E once

--- the decisive pair, no flow, no stage 1 ----------------------------------
[ ] §3.1-A2   plain-TSM control, S = 2024, 2025, 2026
[ ] §3.1-A0   KIP-off baseline, same 3 seeds
[ ] §4.1      eval both — TAD + DoTA, 3 seeds each (12 evals, minutes)
[ ] §4.2      gate check: A2 span must be 0, s = 64 (assert, not eyeball)
[ ] §4.4      T0 + gate_a reference arms
[ ] §4.5      rescore, manifests, score-count assertion
[ ] §5        A2 − A0 on DoTA. STOP AND READ IT against MSAD's +0.1025.

--- flow, and the gated arms ------------------------------------------------
[ ] TAD_SETUP §8      flow targets from frame folders, train ids only
[ ] §3.2a + §3.2c     stage 1 (rank family), convergence assert per seed
[ ] §3.3-A1 + §4      the v3 model, 3 seeds
[ ] §5                A1 − A2 — the only motion-gating evidence there is

--- follow-up, as the results demand ----------------------------------------
[ ] A2b (needs §3.2a) | A3, A4 (need §3.2b)
```

A2 before A1 is deliberate, exactly as on MSAD. If the smoother reproduces the
gain on a second corpus, A1 answers a different question than you think you are
asking — and you want to know that before spending three stage-1 runs and a full
RAFT pass on it.

---

## 7. Pitfalls specific to this campaign

| Don't | Why | Lesson |
|---|---|---|
| Put an in-domain TAD number next to 89.56 | 89.56 is zero-shot; in-domain beats it for free | §0.1, **8b** |
| Reuse `DoTA_constant_s$S` for a TAD-trained arm | That dir belongs to the MSAD campaign. Two corpora, two dirs | **17** |
| Score a `rank` checkpoint under `constant` | Loads clean, raises nothing, wrong number. §4.2 is the only detector | **5**, **24** |
| Say "motion-gated" of `mlp_frozen` | It is a fixed ~50 % smoother, measured over 8 seeds | **24** |
| Vary `num_epochs` between arms | Then the Δ contains a training-length difference | §1.4 |
| Set `data.is_egocentric=true` | TAD is fixed-camera; that flag is DVS tuning for ego footage | §1.3 |
| Read anything into TAD's `mul` loss | `C = 2`; `H_mul` is near-degenerate on this taxonomy | §1.3 |
| Compute a Δ before asserting score-file counts | One 0-byte `.npz` perturbs every paired Δ | **11b** |
| `--set kip.disable_pmg=true` to dodge a missing flow cache | It trains the PMG head to regress zeros | **14** |
| Restart training into an existing `--output-dir` | `metrics.jsonl` appends; MSAD A3 had to be deduped by hand | **17** |

---

## 8. Out of scope here

- **Data preparation** — `core/docs/TAD_SETUP.md`, all of it.
- **P1 / P2 implementation.** Shipped 2026-09-02; see `TAD_SETUP.md` §0.3 and
  `core/tests/test_tad.py`.
- **The MSAD campaign** — `MSAD_DOTA_V3_SETUP.md`, and its results in
  `RESULTS_V3_GATE_ATTRIBUTION.md`.
- **PreVAD** — `core/docs/PREVAD_SETUP.md`. KIP-off trunk + Gate P0 only; it
  ships no pixels, so no KIP-on arm exists there, ever.
- **Checkpoint selection.** TAD ships no val split; every arm reads
  `checkpoint_last`. Shared confound, cancels in a Δ, caps absolute numbers.
