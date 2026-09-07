# EDA — characterize a corpus before you trust a number measured on it

**Tool:** `python -m core.tools.eda` · **Package:** `core/eda/` · **Tests:**
`core/tests/test_eda.py` (data-free) · **Written 2026-09-06.**

This exists because `core/docs/v3/RESULTS_DADA.md` established, *after* a full
seven-arm campaign, that two of its headline numbers were properties of the
corpus rather than of the model:

- DADA-2000's median clip is **9** stride-8 frames under a `Conv1d(kernel=9)`
  score head, so every output timestep sees the whole clip (lesson **C27**);
- **74 %** of its test frames come from clips that are negative in their
  entirety, so a constant-score-per-clip oracle scores **0.9069** micro AUC with
  zero localization (lesson **C12**).

Both are computable from the label files alone, in seconds, with no GPU. Run
this **before** the first training arm on any new corpus.

---

## 1. What it measures

| § of the report | Question it answers | Needs |
|---|---|---|
| **0. Verdicts** | Which structural red flags fire, and what to do about each | labels |
| **1. Corpus shape** | Split sizes, class balance, clip-length distribution, subgroups, and the DVS/steps-per-epoch budget | labels + `meta.json` |
| **1.2 Receptive field** | What fraction of a clip one score-head output timestep sees; how many clips fit entirely inside the kernel (**C27**) | labels |
| **1.3 MIL top-k floor** | How many clips get `k = 1`, i.e. where `L_MIL` is a plain max | labels |
| **2. Label geometry** | Positives per clip, span counts and lengths, multi-span clips (**C18**), and **windows that vanished at this stride** | labels + `meta.json` |
| **3. What the metric measures** | Frame share by clip kind, the **clip-level oracle**, the cross-clip pair fraction, per-clip AUC resolution, what `--score-norm auto` resolves to (**C12**) | labels |
| **3.4 A scored run** | Between/within score variance and constant-curve count for a trained arm | `--scores-dir` |
| **4. Features** | Norms, effective dimensionality, **temporal autocorrelation**, feature variance decomposition | `--clip-dir` |
| **4.2 Linear probe** | **The supervised ceiling of the cached features** — `RESULTS_DADA.md` §10-B | `--clip-dir` |
| **4.3 Flow** | Distribution of the 23-dim RAFT descriptors | `--flow-dir` |

Nothing here reads a checkpoint. Section 3.4 is the only part that touches a
model, and only through saved score curves.

---

## 2. Run it — DADA-2000 and DoTA

Paths are the **as-run** ones (`DADA_SETUP.md` §7 / §1.2a of `DADA_V3_SETUP.md`).

```bash
%%bash
cd /content/drive/MyDrive/Thesis-V3/kat-vad

python -m core.tools.eda report \
  --dataset DADA2000 \
  --data-dir  "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --flow-dir  "$KATVAD_CACHE_ROOT/flow/v1/DADA2000" \
  --output-dir "$KATVAD_OUTPUT_ROOT/eda/DADA2000" \
  --plots

python -m core.tools.eda report \
  --dataset DoTA \
  --data-dir  "$KATVAD_DATA_ROOT/DoTA/labels_s8" \
  --clip-dir  "$KATVAD_CACHE_ROOT/clip/DoTA_s8_ncc" \
  --output-dir "$KATVAD_OUTPUT_ROOT/eda/DoTA" \
  --plots

python -m core.tools.eda compare \
  "$KATVAD_OUTPUT_ROOT/eda/DADA2000/eda_report.json" \
  "$KATVAD_OUTPUT_ROOT/eda/DoTA/eda_report.json" \
  --output "$KATVAD_OUTPUT_ROOT/eda/compare_dada_dota.md"
```

DoTA has **no flow cache** (flow is train-time only and DoTA is eval-only here),
so it takes no `--flow-dir`.

Each `report` writes `eda_report.md` (read this), `eda_report.json` (script
this), and with `--plots` four PNGs under `plots/`: clip-length histogram with
the kernel marked, positives-per-clip histogram, the CLIP autocorrelation curve,
and frames-by-clip-kind with the oracle AUC in the title.

### 2.1 Fold in a trained arm's curves

```bash
python -m core.tools.eda report --dataset DADA2000 \
  --data-dir "$KATVAD_DATA_ROOT/DADA2000" \
  --clip-dir "$KATVAD_CACHE_ROOT/clip/DADA2000" \
  --scores-dir "$KATVAD_OUTPUT_ROOT/DADA2000/constant_s2024/eval_dada/scores" \
  --output-dir "$KATVAD_OUTPUT_ROOT/eda/DADA2000_A2_s2024"
```

Adds §3.4: between-clip / within-clip score variance and the constant-curve
count — the flatness diagnostic that `RESULTS_DADA.md` §4/§7.1 correlated at
Spearman +0.68 with DADA micro AUC and −0.82 with DoTA micro AUC.

### 2.2 Useful flags

| Flag | Why |
|---|---|
| `--sections corpus,labels,protocol` | Label-only run; no feature cache needed, ~1 s |
| `--no-probe` | Skip the linear probes (they dominate the runtime) |
| `--max-clips 200` | Cap the feature section on a huge corpus |
| `--score-head-kernel 3` | Ask "what *would* the coverage be if I changed the head?" before changing it |
| `--mil-topk-pct 4` | Same question for the MIL floor |

---

## 3. Reading §4.2 — the decisive experiment

The frame probe is a logistic regression on the **same cached features the
trained arms saw**, against the **real frame labels**, grouped-CV by clip so no
clip scores itself. It is the *supervised ceiling* of the representation.

| frame probe `auc_macro` | Reading | Consequence |
|---|---|---|
| **≥ 0.60** | The features carry a frame-level accident signal | The deficit is **supervision**, not representation. A denser label, a smaller kernel, or a finer stride can close it. Another KIP variant cannot. |
| **≈ 0.50** | Even full supervision cannot localize on these features | The **representation** is the ceiling. Frame-level work needs a different backbone — this is SimpleTAD's advantage (`RESULTS_DADA.md` §6), not a head problem. |

Read it beside the clip probe. A high clip AUC next to a chance frame AUC is the
representational form of the metric problem in §3: the corpus supports video
classification and not localization.

**The probe is an upper bound, not a target.** Per lesson **14**, do not tune
anything against it.

---

## 4. Thresholds the verdicts use

Defined at the top of `core/eda/report.py`; change them there, not inline.

| Constant | Default | Fires |
|---|---:|---|
| `FULLY_COVERED_WARN_FRACTION` | 0.25 | CRITICAL — score head spans the clip (**C27**) |
| `K1_WARN_FRACTION` | 0.50 | HIGH — MIL top-k is a plain max |
| `ORACLE_WARN_AUC` | 0.75 | CRITICAL — micro AUC is mostly clip classification (**C12**) |
| `BETWEEN_WITHIN_WARN` | 5.0 | HIGH (scores) / INFO (features) — flatness |
| `PROBE_SIGNAL_AUC` | 0.60 | Splits the two readings of §3 above |

---

## 5. Pitfalls

| Don't | Why |
|---|---|
| Read a micro AUC without the clip-oracle row from §3.2 | On a corpus with all-normal clips, micro is largely video classification (**C12**) |
| Treat the frame probe as a model result | It is a supervised ceiling on the features, with the labels the model never sees |
| Point `--clip-dir` at a cache built with a different stride or transform | The tool raises (lessons **C2**/**C13**) — that raise is the feature, do not work around it |
| Act on §1.2 by changing `frame_stride` or `score_head_kernel` casually | Both invalidate every cached feature **and every metric measured on it** (**C2**) |
| Compare `auc_macro` across corpora without §3.3's resolution row | A mean of per-clip AUCs on 9-frame clips with 2 positives is coarse, not wrong |
| Expect a flow-versus-label correlation | Flow is a train-time cache and the train split carries no frame labels |

---

## 6. Out of scope

- **Model evaluation** — `core/evaluate.py`, `core/tools/rescore.py`.
- **Building the datasets** — `core/docs/DADA_SETUP.md`, `DOTA_EVAL.md`,
  `TAD_SETUP.md`.
- **Campaign design and arms** — `core/docs/v3/setup/*_V3_SETUP.md`.
