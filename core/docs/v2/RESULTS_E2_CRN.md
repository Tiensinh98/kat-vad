# v2 P2 — E0, E2(a), E2(b), E2(c): result (2026-09-28)

Tool `python -m core.tools.crn_select` (references `core/crn/reference.py`). Rule: proposal
§4.2 steps 1–6; implementation choices I1–I9 in `PREREG_ADDENDUM.md` §7, written before any
E2 number. Read-out `outputs/v2/REPORTS/v2_E2abc/e2_readout.{json,md}` (gitignored; this file
is the record). Stride 8, CLIP `_ncc`, DoTA-dev 702 clips (700 two-class), T2-val 219 sources,
T2-train 1,272 sources.

## Verdict (mechanical): **R1**, veto passed. The choice between references is not identified.

| | R1 mean | R2 median | R3 robust | R4 past-only |
|---|---|---|---|---|
| DoTA-dev `r` macro (decision) | **0.6902** | 0.6899 | 0.6711 | 0.4956 (ineligible: reversal) |
| DoTA-dev **−f alone** (no pixel) | **0.7641** | 0.7634 | 0.7526 | 0.4334 |
| DoTA-dev `d` raw | 0.5224 | 0.5255 | 0.5392 | 0.5787 |
| DoTA-dev `d` stratified (step-3 cross-check) | 0.5926 | 0.6002 | 0.6054 | 0.5694 |
| E2(c) transfer Δ vs raw (95 % cluster CI) | +0.0302 [+0.020, +0.040] | +0.0300 [+0.019, +0.042] | +0.0250 [+0.014, +0.036] | +0.0335 [+0.022, +0.045] |

Position ruler `t/T`: DoTA-dev 0.567, T2-val 0.550; stratified 0.519 / 0.512.

## What the numbers say

1. **The E2(b) decision metric is dominated by a position prior that the rule itself injects.**
   `f` (a cubic on T2-val normal steps) is U-shaped for R1–R3: the deviation from a whole-clip
   mean is largest at both ends. So `r = d − f(t/T)` contains `−f`, a mid-clip tent, and
   **`−f` alone scores 0.764 on DoTA-dev, above `r` itself** (0.690) and far above `d` (0.522).
   For R4, `f` rises with `t`, so `−f` penalizes late steps (0.433). That, not the reference,
   is why R4 fails the no-reversal check. **The ranking R1 > R2 > R3 > R4 is a ranking of
   `−f`'s shapes.** Proposal §4.2 meant the residualization to *remove* the position artefact;
   on DoTA, whose accidents sit mid-to-late, subtracting a U-shaped normal trend *adds* one.
2. **The step-3 stratified cross-check is not position-free either:** `−f` scores 0.596 on it
   (five bins of width 0.2 do not flatten a steep tent), the same as `d`'s 0.59–0.61.
3. **E2(c) is the one position-clean read, and it favours CRN with every reference:** a linear
   probe on `x − μ^ref` cannot compute `t`, and it beats raw `x` on T2-train → DoTA-dev transfer
   by +0.025 to +0.034, every interval above 0. The four references are within 0.009 of each
   other with overlapping intervals.
4. **The defect is in the *selection metric*, not in the component.** The model gets
   `s·(x − μ^ref)` (§4.2), never `d` or `f`. Whether CRN gives the trunk a position channel is
   read by the §10.1 position probe on `V^t`, which is already required for every arm.

## E0 and E2(a) (record)

- **E0:** DoTA 10 fps → 0.800 s/step, median 13 steps = 10.4 s; T2 30 fps (assumed) → 0.267 s/step,
  median 41 steps = 10.9 s. Step ratio **3.00×** (by the fps assumptions; D8, switches nothing).
- **E2(a):** DoTA-dev bins 343 / 247 / 93 / **19** (= the freeze, §1), median share 0.306, 99.1 %
  start normal. T2-val 114 / 91 / 13 / **1**, median 0.294, 96.3 % start normal. Normal-step
  coverage of the last fifth (T2-val) 0.223 ≥ 0.05 → no fallback; decision metric = `r`.
- `>70` bin: 19 DoTA-dev clips; no reference's eligibility was decided by it alone.

## Open for the advisor

- Keep **R1** (the mechanical verdict; ≈ R2 ≈ R3 on every position-free read; the simplest), or
  amend the selection metric. Re-deciding on E2(c) now would pick R4 by 0.003, inside the
  noise — a post-hoc pick with no identifiable difference.
- R4 is the streaming/deployment form and is **not worse** on the clean read (E2(c) +0.034).
