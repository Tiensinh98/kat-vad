# v2 P2 — E0, E2(a), E2(b), E2(c): result (2026-09-28; D11 stride-3 re-read 2026-09-30)

Tool `python -m core.tools.crn_select` (references `core/crn/reference.py`). Rule: proposal
§4.2 steps 1–6; implementation choices I1–I9 in `PREREG_ADDENDUM.md` §7, written before any
E2 number. Read-out `outputs/v2/REPORTS/v2_E2abc/e2_readout.{json,md}` (gitignored; this file
is the record). Stride 8, CLIP `_ncc`, DoTA-dev 702 clips (700 two-class), T2-val 219 sources,
T2-train 1,272 sources.

> **Superseded for the CRN reference by §D11 below (2026-09-30): the reference is R2.** E1 adopted
> protocol B (DoTA at stride 3), so addendum D11 re-read E2(b)/(c) at stride 3. The stride-8 sections
> after §D11 stay as the P2 record.

## D11 — E2(b)/(c) re-read on DoTA-dev at stride 3 (2026-09-30): verdict **R2**, veto passed

Notebook `colab/v2/p2_e2_s3.ipynb`; `crn_select --dota-s1-dir clip/DoTA_s1_ncc --dota-stride 3`.
Read-outs `outputs/v2/REPORTS/v2_E2_s3/{gate_s8_path,gate_s1_stride8,s1_stride3}/`. The T2 side is
unchanged (D11), so the T2-val tables and `f` are the stride-8 run's.

**Regression gate (passed).** `s1[::8]` and the `DoTA_s8_ncc` path give identical E2(a)/(b)/(c)
tables and the same verdict (R1); the two read-outs differ only in the corpus label. Against the P2
read-out (`v2_E2abc`) E2(b) is identical and E2(c) moves by ≤ 0.0012 (raw 0.6277 → 0.6269), which
changes no verdict.

E0 at s3: DoTA-dev 0.300 s/step, median 33 steps = 9.9 s; T2-val 0.267 s/step → step ratio **1.12×**
(3.00× at s8). E2(a) at s3: DoTA-dev bins 343 / 247 / 93 / 19 (unchanged), 99.4 % start normal.

| | R1 mean | R2 median | R3 robust | R4 past-only |
|---|---|---|---|---|
| DoTA-dev `r` macro (decision) | 0.7081 | **0.7092** | 0.6997 | 0.7049 |
| `r` in 50–70 / >70 bins | 0.722 / 0.653 | 0.729 / 0.650 | 0.711 / 0.636 | 0.693 / 0.541 |
| eligible (no reversal, stratified ≥ 0.5) | yes | yes | yes | **yes** (no at s8) |
| DoTA-dev **−f alone** (no pixel) | **0.7933** | 0.7908 | 0.7837 | 0.4331 |
| DoTA-dev `d` raw | 0.5368 | 0.5386 | 0.5509 | 0.7512 |
| DoTA-dev `d` stratified (cross-check, printed) | 0.6044 | 0.6162 | **0.6257** | 0.6190 |
| E2(c) transfer macro (raw `x` 0.6234) | 0.6562 | **0.6602** | 0.6519 | 0.6545 |
| E2(c) Δ vs raw (95 % cluster CI) | +0.0329 [+0.023, +0.042] | **+0.0368 [+0.027, +0.047]** | +0.0286 [+0.019, +0.038] | +0.0312 [+0.019, +0.042] |

Position ruler `t/T` on DoTA-dev s3: 0.567 overall, 0.511 stratified.

**What it says.**

1. **The rule picks R2, and R2 is adopted** (proposal §4.2 step 4; D11 says the reference is taken at
   the stride E1 adopts). No reference's eligibility was decided by the `>70` bin alone.
2. **R1 and R2 are not distinguishable.** The gap is +0.0011 at s3 and −0.0003 at s8, against
   intervals of about ±0.015; the verdict flipped on a tie. The rule has no tie clause and none is
   added after the number (lesson 14), so the rule's pick stands.
3. **No position-free read contradicts R2.** E2(c), which cannot see `t`, ranks R2 first
   (+0.037, R1 +0.033); the printed stratified cross-check ranks R3 > R4 > R2 > R1. Median is also
   the reference that the proposal's breakdown argument (§4.2) favours over the mean.
4. **The E2(b) metric is still position-dominated, more so at s3:** `−f` alone reaches 0.793 (0.95
   in the `>70` bin), above every reference's `r` (0.70–0.71). As at s8, the claim that CRN helps
   rests on E2(c), not on E2(b).
5. **R4 becomes eligible at s3** (0.4956 → 0.7049): with a 33-step median clip the 8-step warm-up is
   a quarter of the clip instead of most of it. It still does not win; the streaming form remains
   viable at this rate.

## Stride-8 verdict (P2, 2026-09-28; superseded by §D11): **R1**, veto passed. The choice between references is not identified.

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

## Open for the advisor (stride 8; closed by §D11 — the reference is R2)

- Keep **R1** (the mechanical verdict; ≈ R2 ≈ R3 on every position-free read; the simplest), or
  amend the selection metric. Re-deciding on E2(c) now would pick R4 by 0.003, inside the
  noise — a post-hoc pick with no identifiable difference.
- R4 is the streaming/deployment form and is **not worse** on the clean read (E2(c) +0.034).
