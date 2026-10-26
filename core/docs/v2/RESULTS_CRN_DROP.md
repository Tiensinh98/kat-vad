# KAT-VAD v2 — Amendment 20: can `B` drop CRN? (A2-ign vs A3-ign, §31)

Pre-registration: `PREREG_ADDENDUM.md` **§31 (Z1–Z10)**, signed 2026-10-26 before any A2-ign run. Runbook
`colab/v2/crn_drop_a2ign.ipynb`; tool `core/tools/crn_drop_readout.py`; plan `.project/plans/katvad-v2-crn-drop.md`.
Raw outputs (gitignored; this file is the durable record): `outputs/v2_crndrop/{A2ign/s*, REPORTS/}`. Colab read-outs
carry `commit: UNKNOWN` by design.

**Provenance.** A2-ign = E4-0's A3-ign with `v2.crn = none` (config diff `{v2.crn}` asserted per seed, planned and
written); inputs = E3's `A2_S` caches (manifest `crn = none`, V2-S, width 896, A3's `train_ids_sha1`). Five runs, seeds
2024–2028. Control = E4-0's five A3-ign runs, not retrained. Set: DoTA-CAP-dev (569 clips), protocol B.

## 1. Decision (Z7)

**Base = A3-ign. "Removing CRN is not shown to be harmless on DoTA-CAP-dev at n = 5."** All four checks fail
(margin M = 0.01):

| check | read | result |
|---|---|---|
| Z4a paired t95 of per-seed macro Δ (A2-ign − A3-ign), low > −M | +0.0023 [−0.0151, +0.0197] | FAIL |
| Z4b cluster bootstrap of per-clip Δ, low > −M | −0.0014 [−0.0165, +0.0135] | FAIL |
| Z5 `f_2` Δ cluster bootstrap, low > −M | **−0.0119 [−0.0204, −0.0033]** | FAIL |
| Z6 T2-val macro ≥ A3-ign − M; O1′ every seed | 0.7068 vs 0.7338 (−0.027); O1′ 5/5 | FAIL |

Per seed (Z4a): s2024 −0.0057, s2025 −0.0003, s2026 +0.0267, s2027 −0.0015, s2028 −0.0078.

## 2. Levels

| read | A3-ign | A2-ign | E3 A2 (`span`) | `p_T2` |
|---|---:|---:|---:|---:|
| DoTA-CAP-dev macro | 0.7912 | 0.7898 | 0.7536 | 0.8220 |
| `f_2` macro | 0.8526 | 0.8407 | — | — |
| T2-val macro (seed mean) | 0.7338 | 0.7068 | — | — |
| T2-val normal-window peak (printed) | 0.6151 | 0.4271 | — | — |
| source-shortcut AUC, `v^t` (printed) | 0.5838 | 0.9997 | — | — |
| position R², `v^t` (printed) | −0.0070 | −0.0820 | — | — |

Per-clip Δ by native-fps group (printed): `r1` +0.0012 [−0.0168, +0.0188]; `r3` −0.0079 [−0.0293, +0.0128]; other
+0.0255 [−0.1134, +0.1664].

## 3. Readings

1. **Detection: no loss shown, no non-inferiority shown.** The macro point estimates are ≈ 0 (+0.002 paired, −0.001 by
   clip), but both intervals reach below −0.01. Z4 fails on width (one seed, s2026, carries +0.027), not on a loss.
2. **Beyond position, A2-ign trails** (`f_2` whole interval below 0, −0.012), the size E3 predicted (≈ −0.016 at
   w = 2). This is the one claim CRN carried on T2, and it reproduces under `ignore`.
3. **In-domain, A2-ign trails** (T2-val −0.027). Its normal-window peak is lower (0.43 vs 0.62), printed only.
4. **CRN's mechanical effect reproduces:** without it `V^t` identifies the source video (shortcut 0.9997 vs 0.584).
5. **`ignore` and CRN add:** `ignore` without CRN lifts DoTA-CAP-dev +0.036 (E3 A2 0.7536 → 0.7898), with CRN +0.034
   (A3 0.7576 → A3-ign 0.7912, E4-0).

## 4. What may and may not be written

| may | may not |
|---|---|
| "Removing CRN is not shown to be harmless on DoTA-CAP-dev at n = 5" | "CRN improves detection" (macro Δ ≈ 0; Z7 forbids reading a failure as a gain) |
| "Without CRN, the position-controlled read (`f_2`, DoTA-CAP placement) and T2-val macro fall (printed)" | "CRN is useless" |
| "CRN's effect depends on the training corpus: on T2 it keeps the beyond-position signal and blocks the source shortcut; trained on Nexar it costs −0.06 … −0.08 (`RESULTS_NEXAR.md`)" | any A2-ign number on CAP-eval, T2-test, TAD or Nexar (Z9: a new amendment) |

**Nothing moves (Z9).** `B` = A3-ign; every earlier A3-ign read stands.
