# KAT-VAD v2 — Amendment 20: can `B` drop CRN? (A2-ign vs A3-ign on T2)

Rule: `core/docs/v2/PREREG_ADDENDUM.md` **§31 (Z1–Z10)**, signed 2026-10-26 before any A2-ign run.
Tool `core/tools/crn_drop_readout.py`; runbook `colab/v2/crn_drop_a2ign.ipynb`.

## 0. Status

| step | state |
|---|---|
| §30 crn_ref seen (sentence (2), `f` = 0.17) | seen 2026-10-26; **not yet written** to `RESULTS_NEXAR.md` §5.2 (user's call) |
| §31 written + signed | done 2026-10-26 |
| tool + tests (`crn_drop_readout`, `sc2_readout.paired_reads` shared) | done |
| notebook | built, not run |
| user uploads code + runs notebook → pastes `crn_drop_readout.md` | **next** |
| `RESULTS_CRN_DROP.md` | after the read-out |

## 1. Why

- Nexar §29: CRN costs on every set when trained on Nexar (A3 − A2 −0.06 … −0.08).
- §30: the train/eval reference-unit mismatch explains only 17 % of that; the rest is CRN.
- T2 (where `B` is trained): CRN ≈ 0 on detection, but it is the one component under which the motion gain survives
  position control (E3 w = 2: A3 − A0 +0.023 vs A2 − A0 +0.007 ∋ 0; CAP-eval +0.004 ∋ 0).
- So "drop CRN from `B`" is a T2 question, answered with a non-inferiority rule (burden on the removal).

## 2. Design (one change)

A2-ign = E4-0 A3-ign with `v2.crn = none`. Inputs = E3's `A2_S` caches (no extraction, no bake). Control = E4-0's
A3-ign runs, never retrained. Cost: **5 training runs + 5 diags + 1 protocol-B call** (≈ A3-ign's E4-0 cost).

## 3. Rule (summary; §31 is authoritative)

Drop CRN iff, margin M = 0.01: paired t95 low > −M **and** cluster-bootstrap low > −M (DoTA-CAP-dev macro), `f_2`
cluster-bootstrap low > −M, T2-val macro ≥ A3-ign − M, O1′ every seed. Else A3-ign stays.

## 4. Risks

| risk | effect | mitigation |
|---|---|---|
| E3 predicts Z5 fails (beyond-position gap ≈ 0.016 > M) | A3-ign stays; 5 GPU runs buy a recorded "not shown harmless" | disclosed in §31 before the run; still the only T2 test of CRN under `ignore` |
| `A2_S` cache not on Drive / wrong split | preflight stops | Z10(c) asserts in step 1 |
| torn Drive checkpoint (lesson (bi)) | crash mid-campaign | atomic `publish`, quarantine `*.corrupt-*`, J10 |
| reading "keep" as "CRN helps" | over-claim | Z7 forbids it; the sentence says "not shown harmless" |

## Appendix A — read-outs

(empty)
