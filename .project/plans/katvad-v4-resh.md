# Plan — KAT-VAD v4 / ReSH (Reference-Steered Expert Heads)

**Branch:** `v4` (from `v2@5469bfd`). **Rules:** `core/docs/v4/PREREG.md` (binding). **Design:**
`core/docs/v4/KAT VAD v4 architecture proposal.md` rev 3 (§2 = review round 1, §2.1 = feasibility round 2).

## 0. Status

| Stage | What | State |
|---|---|---|
| V4-0 | rev 3 + PREREG | **written 2026-10-15, uncommitted** — commit before Stage A runs |
| V4-1 = A | hooked V2-S pass over T2, `select_heads`, G1a | **built 2026-10-15** (tools + 33 tests + `colab/v4/stage_a.ipynb`), not run |
| V4-2 = B | hooked pass over CAP-dev + DF8's 7,524 crop-boundary windows, G1b, G1c | not built (only after G1a PASS) |
| C | re-bake with `r̂` columns, ReSH module, A4-H + A4 × 5, DF8, permutation control | not built (only after G1b + G1c PASS) |
| D | A4-rand × 5, final read | not built |

## 1. Why this order (the decision gate)

V4's cheapest kill points come first. Stage A costs one hooked V2-S pass over T2 and CPU; Stage B one pass over
CAP-dev plus DF8's boundary windows. **No training before both gates pass.** Expected outcome (proposal §7.3): G1b
fails more often than not; a fail is a reportable bounded negative ("HeadHunt's dilution does not transfer to a
6-head video ViT here").

The most likely *expensive* failure is "steps 1–2 pass on CAP-dev, X7 fails": every adopted v2 step raised DF8's
position cost by ≈ 0.04 (A3 − A0 −0.041, A3-ign − A3 −0.041, paired t95 from `df8_readout.json` per-seed macros).
G1c is the probe-level early warning for that.

## 2. Stage A — what was built

| File | Role | PREREG |
|---|---|---|
| `core/models/videomae_heads.py` | `HeadTap`: `forward_pre_hook` on each block's `attn.proj`, token mean, `(B, 72, 64)`; encoder untouched | I2 |
| `core/tools/extract_head_features.py` | calls `extract_video_features.encode_frame_dir` **unchanged** inside the tap; writes `(T, 72·64)` float32 + `ucos/{id}.npy` (pooled-`u` row cosine vs the V2-S cache) atomically; `load_head_rows`, `ucos_failures` | I1–I4 |
| `core/tools/select_heads.py` | weak step labels, cubic detrend, 5 source folds, LDA / sym-KL / MMD² / K-means NMI, RSS, stability floor, best block, A4-rand + 20 draws → `head_selection.{json,md}` | S1–S14 |
| `core/tools/v4_gate.py g1a` | refuses an uncommitted selection (sha256); 32 logistic probes T2-train → T2-val; mechanical verdict | G1a |
| `core/constants.py` `V4_*` | every number above | — |
| `core/tests/test_v4_stage_a.py` | 33 tests: tap == per-head SDPA, output unchanged, resume sees head files as done, labels, detrend, measures, trend-vs-bump selection, draws disjoint, G1a logic, end-to-end glue | — |
| `colab/v4/stage_a.ipynb` | preflight → extraction (shards from `DADA2000.zip`) → I4 gate → selection → **STOP: commit** → G1a | — |

Not built on purpose (F8): the `r̂` re-bake, `σ_e`, `m_r`, `s_r`, PCA-32 — Stage C preparation.

## 3. Run instructions (Stage A)

1. Commit `core/docs/v4/` (PREREG first), upload branch `v4` to `Thesis-V2/kat-vad/`.
2. Run `colab/v4/stage_a.ipynb` §0–§4 (GPU for §2 only).
3. Download `outputs/v4/REPORTS/stage_a/head_selection.{json,md}`, commit, paste the sha256 into §5.
4. Run §6 (G1a). Paste `g1a_readout.md` back. PASS → Claude builds Stage B. FAIL → V4 stops.

## 4. Risks noticed while building

- `feature_cache.is_complete` accepts only 2-D arrays; a 3-D head cache would never read as complete, so every
  resume would silently re-extract everything. Heads are stored flat `(T, 72·64)` and reshaped on load (pending
  lesson candidate).
- A cubic absorbs ≈ 22 % of a 6-of-40-step bump (measured in `test_detrend_keeps_a_bump`): the F5 cost of
  detrending, which the S10 stability floor guards.

## Appendix A — read-outs

*(none yet)*
