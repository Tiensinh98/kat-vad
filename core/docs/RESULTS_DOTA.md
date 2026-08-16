# DoTA zero-shot — results, the protocol bug, and what they do and do not mean

**Written:** 2026-08-08 · **Protocol:** `core/docs/DOTA_EVAL.md` ·
**Companion:** `core/docs/RESULTS_MSAD.md`

> **⚠ The KIP verdict in this document is superseded.
> See `core/docs/RESULTS_NCC.md`.**
>
> Every number here was measured on **center-cropped** features. The Δ(on − off)
> of **−0.0329** recorded in §1 and §5 was a measurement of the crop, not of KIP:
> the same A/B on `no_center_crop` features gives **+0.0911**
> (CI [+0.0799, +0.1019]), and only the KIP arm moves (+0.1514 macro AUC vs
> +0.0077 for KIP-off). §4's diagnosis of the transform defect was right; the
> mistake was recording a signed delta while that defect was open — now lesson
> C14.
>
> **Still valid and unaffected:** §2's pooling analysis (lesson C12), the
> `gate_a` reproduction at 0.6142 min-max vs published 0.6260, and §3's
> saturation finding.

Three checkpoints scored on DoTA val (1,397 of 1,402 clips — 5 lost to a
truncated unzip, §1). No training. The first run reported all three at chance;
this document records what was actually wrong and what the corrected numbers
support.

---

## 1. Headline — the first run measured the metric, not the models

| Arm | Checkpoint | raw pooled | **min-max** | z-score | macro per-clip |
|---|---|---:|---:|---:|---:|
| `gate_a` | LaGoVAD released `best.ckpt` | 0.5055 | **0.6142** | 0.6236 | 0.6328 |
| `eval_kip_off` | ours, MSAD-full, KIP off | 0.5243 | **0.5539** | 0.5575 | 0.5561 |
| `eval_kip_on` | ours, MSAD-full, KIP on | 0.5126 | **0.5215** | 0.5215 | 0.5232 |

LaGoVAD's published DoTA number is **62.60**. `gate_a` under min-max is
**0.6142** — a reproduction. Under the raw pooling the first run used, it was
**0.5055**, i.e. chance.

**The reference arm is what made this findable.** A released checkpoint cannot
be at chance on the benchmark its paper reports, so the fault had to be
downstream of the model. Without `gate_a` in the run, three plausible-looking
bad numbers would have been read as a KIP result.

Bootstrap-paired over the 1,392 clips with both classes (5,000 resamples):

| Contrast | Δ mean per-clip AUC | 95 % CI | win rate |
|---|---:|---|---:|
| KIP-on − KIP-off | **−0.0329** | [−0.0513, −0.0138] | 43.6 % |
| ours (KIP-off) − `gate_a` | **−0.0767** | [−0.0920, −0.0615] | 34.4 % |

---

## 2. Defect 1 — micro AUC pooled raw scores across clips

`core/evaluate.py` concatenated every clip's scores and called `roc_auc_score`
once. That is right on MSAD and wrong on DoTA, and the reason is the label
distribution, not the dataset:

- **MSAD test is 50.4 % normal videos.** The between-video score scale is real
  signal — "this whole clip is calm" is a true statement the metric rewards.
- **DoTA val is 0.21 % normal** (3 clips of 1,397, and those 3 only because
  their anomaly window rounds away at stride 8). Every clip contains an
  anomaly; the only question is *when*. The between-clip scale carries no label
  information, so pooling it in lets a confident clip's **negatives** outrank a
  hesitant clip's **positives**.

Cost on the reference arm: **11 AUC points**.

LaGoVAD knows this — `src/offline_evals/offline_dota_eval.py` min-max
normalizes each clip before accumulating. Its generic `full_length_eval.py`
harness does not. Lesson C8 recorded the wrong one of the two as the published
protocol, from a code read that was never checked against a reproduction; C8 has
been corrected and C12 added.

**Fix:** `--score-norm {auto,none,minmax,zscore}`, default `auto`, which
resolves from the label distribution — min-max when normal videos are under 5 %
of the test set. Deciding from labels rather than a dataset name means a new
all-abnormal benchmark cannot silently inherit the wrong protocol. The threshold
is 5 %, not zero: an all-or-nothing rule let DoTA's 3 normal clips restore the
broken protocol, which is exactly what the first implementation did.

`results.json` now carries `auc` (resolved rule), `auc_raw`, `auc_macro` and
`score_norm`, so the two protocols stay comparable instead of one silently
replacing the other.

**Re-scoring costs nothing.** Pooling is a post-processing step over saved
score curves:

```bash
python -m core.tools.rescore --run-dir outputs/DoTA/gate_a          # compare rules
python -m core.tools.rescore --run-dir outputs/DoTA/gate_a --write  # rewrite results.json
```

---

## 3. Defect 2 — the MSAD checkpoints are saturated

Protocol alone does not rescue our arms. Per-clip score dynamic range:

| Arm | median range | frames > 0.99 | near-constant clips |
|---|---:|---:|---:|
| `gate_a` | 0.0418 | 0.0 % | 24 / 1,397 |
| `eval_kip_off` | 0.0199 | **43.9 %** | **238 / 1,397** |
| `eval_kip_on` | 0.1388 | 0.0 % | 1 / 1,397 |

`stage2_kip_off` emits ~1.0 on 44 % of all DoTA frames and is flat on 238 clips
— a model asserting "anomaly" everywhere, with no localization signal left for
any normalization to recover. This is the predicted consequence of the known
selection defect: eval reads `checkpoint_last` at train `mil ≈ 0.001`, with no
validation split and no model selection (`RESULTS_MSAD.md` §6, open item 2).

---

## 4. Defect 3 — the transform is inconsistent, in two directions

`gate_a` was scored on **center-crop** features, but LaGoVAD trained
`best.ckpt` on **`no_center_crop`** (anisotropic resize, full field of view —
`LaGoVAD-PreVAD/src/utils/video_loader.py:79`). That mismatch is a confound on
the one arm used as the reproduction reference, and plausibly explains the
residual 0.6142 vs 0.6260.

The internal inconsistency matters more. `core/flow/raft_extract.py:
preprocess_for_raft` resizes **full-frame** to 240×320; `preprocess_frames`
center-crops away roughly the left and right quarters of a 16:9 frame. So
`L_KIP_rec` trains the PMG-flow head to regress motion evidence removed from its
own input — and 349 of 1,402 DoTA clips are `ego: lateral` / `other: lateral`,
where that evidence lives precisely in the discarded strips. **KIP may have been
handicapped by construction**, independent of any baseline-parity argument.

Decision: move the whole pipeline to `no_center_crop` (§6). `--no-center-crop`
is now available on `core.tools.extract_clip_features`; the default is unchanged
so existing caches stay valid (lesson C2/C13).

---

## 5. What the KIP number means — and does not

KIP-on is **worse** than KIP-off by 0.033 per-clip AUC, CI excluding zero. On
the benchmark where the motion claim should be strongest. That is a real
negative measurement and it should be written down as one.

It is **not** a verdict on KIP, for three reasons that all have to be closed
first:

1. **Both checkpoints come from a defective selection regime** (`checkpoint_last`,
   no val split, n=1 seed). The two arms are not even at comparable points on
   the overfitting curve — KIP-off is saturated (43.9 % of frames > 0.99),
   KIP-on is not. That is not an apples-to-apples contrast.
2. **The features cropped away the evidence KIP exists to use** (§4).
3. **Zero-shot MSAD→DoTA transfer** compounds both: fixed-camera training,
   ego-camera evaluation.

Do not change KIP's architecture on the strength of this number. Fix the
measurement chain, then re-measure.

---

## 6. Next — the no_center_crop rebuild

Full command sequence: **`core/docs/COLAB.md` §"no_center_crop rebuild"**.
Everything lands in **new paths**; nothing overwrites the center-crop artifacts,
so the existing MSAD 0.9052 reproduction stays on disk for comparison.

| Artifact | Re-do? | Path |
|---|---|---|
| RAFT flow cache | **no** — already full-frame, independent of this transform | `cache/flow/v1/MSAD/` (unchanged) |
| MSAD CLIP features | yes (~1,080 videos) | `cache/clip/MSAD_ncc/` |
| KNN cache | yes (derived from CLIP) | `cache/knn/MSAD_ncc/` |
| stage1 / stage2 ×2 | yes | `outputs/MSAD_ncc/**` |
| MSAD eval | yes — re-measure the 0.9052 gate | `outputs/MSAD_ncc/eval_*` |
| DoTA CLIP features | yes (1,397 clips, stride 1 — the long pass) | `cache/clip/DoTA_ncc_s1` → `_s8` |
| DoTA eval, 3 arms | yes | `outputs/DoTA_ncc/**` |

Ordering notes:

- **Fold the validation split + `checkpoint_best` fix into this retrain.** It is
  already queued (`activeContext.md`), and doing it separately means paying for
  two full training cycles.
- **The MSAD reproduction becomes unverified until re-measured.** 0.9052 was
  obtained on the center-crop cache. It may not land there again; that is the
  accepted risk of the switch, not a surprise.
- A ~200-clip `gate_a` probe on `no_center_crop` features is the cheap
  go/no-go: if 0.6142 moves toward 0.6260, the transform is confirmed and the
  full rebuild is justified by evidence rather than by argument.

---

## 7. Reproducing this document

```bash
# corrected metrics for all three arms, from the saved score curves
for d in gate_a eval_kip_off eval_kip_on; do
  python -m core.tools.rescore --run-dir "outputs/DoTA/$d"
done
```

Every number above comes from `outputs/DoTA/*/scores/*.npz`, which are in the
repo. No GPU, no re-inference.
