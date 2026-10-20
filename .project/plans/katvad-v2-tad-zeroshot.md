# KAT-VAD v2 — TAD zero-shot generalization probe (branch `v2`)

Rule: `core/docs/v2/PREREG_ADDENDUM.md` **§26 (Amendment 15, T1–T10)**, signed 2026-10-21 before any TAD
VideoMAE feature or v2-arm TAD score existed. Tool `core/tools/tadz_readout.py` (+ `core/tests/test_tadz_readout.py`),
constants `V2_TADZ_*`, runbook `colab/v2/tad_zeroshot.ipynb`.

## §0 Status

| step | state |
|---|---|
| Rule §26 | written + signed 2026-10-21 |
| Tool + tests | built 2026-10-21; T0 re-read locally from `outputs/v1/TAD/gate_t0` = micro 0.7912 / macro 0.7578 / clip 0.7687 / oracle 0.9226 / length ruler 0.8968 (all == TAD_SETUP §8.1) |
| Notebook | built 2026-10-21, **not run** |
| Read-out | — |

## Why this design

* **Question.** The user asked whether the deployed A3-ign "still works" on other datasets. Reframed (user agreed by
  asking for the notebook): *measure* transfer to a fixed camera, against its CLIP-only control, beside LaGoVAD and
  beside what position/length alone score. It never moves `B`.
* **TAD first, MSAD not.** TAD is traffic and has a reproduced reference (Gate T0). MSAD has non-traffic classes the
  one-definition T2 model was never asked about; deferred.
* **Stride 8, not protocol B's 3.** T0's protocol (LaGoVAD on identical rows; `TAD_ncc` already on Drive) and T2's
  0.27 s/step at an assumed ~30 fps (A-T1). E1's rate-matching logic picked s3 for 10 fps DoTA; the same logic
  picks s8 here. One endpoint only.
* **Only V2-S needs pixels.** CLIP = Drive `clip/TAD_ncc`; V2-S extracted at s8 from the Kaggle frames
  (`nikanvasei/traffic-anomaly-dataset-tad`) through a numeric-order symlink farm (TAD frame names are not
  zero-padded, TAD_SETUP §4.3). Gate T9(c) proves the Kaggle frames are T0's frames (⌈N/8⌉ == rows on all 100,
  CLIP re-extraction cos ≥ 0.999 on 5).
* **`core.evaluate`, not a new scorer.** It is T0's code path; the baked cache guard (`check_input_manifest`)
  refuses a mismatched arm/input.

## Risks

* Domain shift (dashcam ego-motion → static camera) may make the V2-S stream out of distribution: σ_u and the CRN
  scale are T2's. That is what the A3-ign − A0 contrast reads; a cost is a reportable limitation, not a bug.
* T0 (PreVAD-trained) vs A3-ign (≈800 T2 windows) is not a matched comparison; printed with that caveat (T6).
* Kaggle copy could differ from the archive T0 used → T9(c) stops the run before any score.

## Appendix A — read-out

(empty until the user pastes `v2_tadz/REPORTS/tadz_readout.md`)
