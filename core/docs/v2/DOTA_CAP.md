# DoTA-CAP — DoTA pixels recovered from CAP-DATA (read-out, 2026-10-03)

The motion endpoint of KAT-VAD v2 (addendum §11, **Amendment 4**, D13–D15; construction L1–L7 with
Amendments 4a–4c). DoTA's own frames are gone; MM-AU's CAP-DATA re-hosts most DoTA clips. This page is
the durable record of the construction run — `outputs/` is gitignored. Tool: `python -m core.tools.dota_cap`
(`align` → `extract` → `finalize` → `freeze`), runbook `colab/v2/dota_cap_videomae.ipynb`.

**Name it "DoTA-CAP (n/1397)". Never put a DoTA-CAP number beside a full-DoTA number, LaGoVAD's 62.60
or a phase-4 DoTA number (D14)** — the dropped clips are not missing at random.

## 1. Frozen splits (L7)

Frozen with `python -m core.tools.dota_cap freeze` into `core/splits/v2/` beside the base splits, in
their own manifest `DOTA_CAP_MANIFEST.json` (the base `SPLITS_MANIFEST.json` is unchanged and
`freeze_splits --check` still passes). Read them only through `load_split`.

| split | rule | clips | sha1 |
|---|---|---|---|
| `dota_cap_dev` | DoTA-CAP ∩ `dota_dev` | **569** / 702 | `9425e4fe910b…` |
| `dota_cap_eval` (**sealed**) | DoTA-CAP minus `dota_dev` | 560 | `497e02df7efa…` |

The eval side is computed as a complement, so freezing never read DoTA-eval's ids. Provenance in the
manifest: finalized id list sha1 **`e1a37bbbb2eb…`**, alignment sha256 `5e8690e031ca…`, encoders
`vit_b_k710_dl_from_giant` + `vit_s_k710_dl_from_giant`, base `dota_dev` sha1 `b0adfcd0739a…`.
`load_split("dota_cap_eval")` raises `SealedSplitError` unless `final=True`.

## 2. Alignment (L1‴ + L2 + L3′, CLIP caches only)

Candidates = the 1,248 MM-AU P0 `exact` pairs (κ ≥ 0.99). Gates: mean aligned cosine ≥ 0.99, every frame
≥ 0.95, irregular steps ≤ 5 %.

| reason | clips |
|---|---|
| **ok** | **1,129** |
| `p0_none` (not in CAP) | 91 |
| `p0_near` (CAP re-processed, excluded by L0) | 58 |
| `mean_cos` | 56 |
| `out_of_range` | 46 |
| `min_cos` | 9 |
| `cap_shorter` | 8 |

| set | kept | of | share |
|---|---|---|---|
| all | 1,129 | 1,397 | 0.808 |
| dev | 569 | 702 | 0.811 |
| rest (eval) | 560 | 695 | 0.806 |

Kept clips by median CAP step (CAP stores DoTA at 10 or 30 fps): 1 → 817, 2 → 2, 3 → 310. Phase shift
(Amendment 4b, CAP frames): 0 → 866, +1 → 232, +2 → 22, −1 → 6, −2 → 3. Band (Amendment 4c) on rate > 1,
mean cosine q50 line → band: `ok` (n = 312) 0.9882 → 0.9937; `mean_cos` (n = 44) 0.9820 → 0.9878 (stay
out); `min_cos` (n = 2) 0.9854 → 0.9917.

## 3. Extraction (L4–L6, Colab GPU)

Per CAP group: `1-10` 174 · `11` 299 · `12-42` 243 · `43` 270 · `44-62` 143 = **1,129, all `ok`** — L4
(frame count) and L5 (pixel gate: CLIP `no_center_crop` of the rebuilt frames vs `DoTA_s1_ncc`) dropped
nothing. On the kept clips the pixel gate read mean cosine min **0.9900** / q50 0.9941, per-frame min
**0.9503** / q50 0.9821. VideoMAE V2-B and V2-S at DoTA's native 10 fps, 16 consecutive frames, causal,
clamped at DoTA frame 0, stride 1 (row-aligned with `DoTA_s1_ncc`):
`cache/video/<encoder>/DoTA_CAP_s1_squash/`.

## 4. What is not done yet

- **D15 representativeness** (A0/A1 on full DoTA-dev vs DoTA-CAP-dev; length, accident share and
  category of kept vs dropped dev clips) — printed by the E2(d) harness, never decided on.
- E2(d)'s implementation choices on DoTA-CAP-dev must be fixed in the addendum before any motion
  number is read.
