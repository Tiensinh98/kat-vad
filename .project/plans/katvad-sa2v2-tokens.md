# SA²-v2 on 7×7 pooled CLIP tokens: Nexar → DADA-2000, then DoTA-CAP-dev, then C2

Rule: `core/docs/final/SA2V2_RUN.md`. Decisions by the user, 2026-11-05: 7×7; no pilot; edit
`KATVAD.forward`; DoTA-CAP-dev after DADA; C2 after the main read; C1 on DADA.

**Status 2026-11-06: CLOSED.** SV2, W0D, probe, SV3 and SV4 read out, all negative for the module
(`SA2V2_RUN.md` §7, §8.3, §9, §10.4, §11.5); SV4's stop rule §11.4 fired, so no further arm, gate, LR or fusion
variant on tokens. C2 (`SV2_shuf`) read: the module learned a label-dependent map (entropy, spread, top-3 `q`,
module losses beat the permuted run), in the residual `wᵀH`, and the map does not point (§12). Still open, not
gating: SV3 maps. Notebooks `colab/final/sv4_wp.ipynb`, `sa2v2_nexar_dada.ipynb` (part B = C2).

## Build

1. Model `core/models/sa2v2.py`; splice `KATVAD._sv2_rows`, motion dropout mask; off by default.
2. Data: `cell_columns` (per-source D before window cut), `EvalCells`, `DVSFeatureDataset(cell_dir)`.
3. Tools: `sa2v2_cells {pool,extract,stats,shuffle-labels}`, `sa2v2_eval dump`, `sa2v2_readout`.
4. Tests `core/tests/test_sa2v2.py` plus the existing STP, W_u, baseline parity tests.
5. Notebook `colab/final/sa2v2_nexar_dada.ipynb`.

## Run order (Colab; estimates, nothing measured)

| Step | Estimate |
| --- | --- |
| Preflight, pool Nexar tokens from `Thesis-V5`, level stats | 15–40 min |
| DADA tokens: one CLIP pass, 16,090 frames | 10–20 min |
| Train SV2 (1,749 steps). Stop and look if > 2 s/step | 15–45 min |
| DADA read + dump + read-out | 5–10 min |
| **Stop: paste the read-out.** Then C2 | 15–45 min |
| DoTA-CAP-dev pool, read, maps | 10–15 min |

## Risks

| Risk | Mitigation |
| --- | --- |
| `Thesis-V5` token caches missing | Preflight asserts paths and row counts before any GPU work |
| Host path changes | Default path keeps `mask` object and no new state keys; parity tests |
| `τ_Δ` in the wrong space | Frozen `text_projection` buffer; preflight asserts the checkpoint's is not identity |
| Memory at 6,400 valid rows × 49 tokens | Only valid rows go through the encoder; notebook prints peak |
