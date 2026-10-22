# KAT-VAD v2 — TAD clip-level gate `s = c(v) · minmax_v(y_A3ign)` (branch `v2`)

Rule: `core/docs/v2/PREREG_ADDENDUM.md` **§27 (Amendment 16, G1–G10)**, signed 2026-10-21 before any gate was
fitted. Tool `core/tools/tad_clip_gate.py` (`fit` / `score` / `readout`, + `core/tests/test_tad_clip_gate.py`, 13
tests), constants `V2_TADG_*`, runbook `colab/v2/tad_clipgate.ipynb`.

## §0 Status

| step | state |
|---|---|
| Rule §27 | written + signed 2026-10-21 |
| Tool + tests | built 2026-10-21; suite 1184 / 0 fail. Smoke on §26's real score files with a **fake** gate: G3 macro identity holds (0.7594), the LaGoVAD-form row reproduces the post-hoc 0.7524 / 0.8000, a random dev gate stops at G5, readout ≈ 80 s CPU |
| Notebook | built 2026-10-21 (install cell fixed: `av` / `faiss` / `transformers` are pulled in by the readout helpers), run by the user 2026-10-21 on CPU |
| Read-out | **done 2026-10-21** → `core/docs/v2/RESULTS_TADZ.md` §3 |

## Why this design

* **Problem (§26 post-hoc).** A3-ign ranks *within* a TAD video as well as LaGoVAD (macro 0.759 vs 0.758) but
  cannot separate normal from accident videos (clip-level AUC 0.544; max score ≈ 1.0 on normals). **A0 is blind
  too (0.578)**, so the cause is the training data (T2 has no stand-alone normal video), not CRN.
* **Decoupling.** Macro is a mean of per-video AUCs, so any positive per-video rescaling leaves it unchanged.
  `c(v)` therefore can only move micro / clip-level; the DoTA story cannot change.
* **Gate on MSAD-train.** Both classes from the same cameras (no C38 source shortcut) and the same `*_ncc`
  extractor as `TAD_ncc` (C2). Mean pooling (length-blind; TAD normals are ~4× longer). Fixed logistic, nothing swept.
* **Dev gate on TAD-train**, read once before the test (bar 0.60); never fitted on, so the zero-shot claim holds.
* **The form was seen post-hoc on TAD-test with LaGoVAD as `c`** (disclosed in §27). The gate itself is new.

## Risks

* MSAD's anomaly classes are mostly non-traffic; the gate may learn "people/fire present" and miss car crashes →
  G5 fails or the endpoint is null. Reportable.
* MSAD normals are CCTV of mixed scenes; TAD normals are road cameras → domain shift on the normal side.
* TAD-train features may be missing from `TAD_ncc` → notebook falls back to Kaggle extraction (GPU).
* Not streaming: `c(v)` needs the whole video's mean. A causal running mean is a separate variant.

## Appendix A — read-out

Read 2026-10-21 (`outputs/v2/v2_tadg/REPORTS/tadg_readout.md`). Gate fitted on MSAD-train 120/360 (train AUC 1.000),
MSAD-test 0.948. **G5 PASS:** TAD-train (410, 200 abnormal, `TAD_ncc`, id-prefix labels) gate clip-level AUC 0.805 vs length
ruler 0.654. **G6: "a clip-level gate trained on MSAD restores clip-level discrimination on TAD"** — micro 0.678 → 0.835,
paired t95 +0.1361 [+0.1323, +0.1398] (5/5), video bootstrap +0.1569 [+0.1024, +0.2037]; clip-level AUC 0.544 → 0.851;
macro 0.7594 unchanged. vs LaGoVAD micro +0.044 [−0.031, +0.129] (not separated). Post-hoc: Spearman(c, L) −0.53, but the
gate residualized on log length still scores 0.798 (test) / 0.755 (dev); within length terciles 0.75 / 0.97 / 0.88.
Recording-style confound not testable (no source metadata).
