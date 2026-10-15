# KAT-VAD v2 — DF8 / H1: position-shift stress test (plan)

Status (2026-10-15): **RUN and READ OUT — X8 verdict "R1 > 0, X7 fails"** (App. A; `core/docs/v2.2/RESULTS_DF8.md`). Before that (2026-10-14): **BUILT, pre-registered as v2 addendum §24 (Amendment 13, X1–X10), not run.** Resumed as v2.2 batch 2 / SC-1 (round 7 D8, D13). Earlier (2026-10-11): deferred by the user. When resumed it waits for D1–D6 (§6). Once signed,
the rule (§4) becomes addendum §24 (Amendment 13) and is committed **before** any crop is scored.
Origin: `REPORT_V2_RESULTS.md` §8 H1, pending (ay) (design guard), RESULTS_E4_0.md §6 (`p_T2` 0.8256 > A3-ign 0.8120
on the second look). No training.

## 0. Status table

| step | what | state |
|---|---|---|
| P0 | signed choices D1–D6 → addendum §24 (X1–X10; X7 corrects D8's sign) | **done 2026-10-14** |
| P1 | `core/tools/position_shift.py plan` — crop table from labels (pure) | **built** (491 reproduced on the saved labels) |
| P2 | `position_shift heads` — re-extract the clamped V2-S windows from CAP frames (D2) + gate X10(c) | **built** (7,524 windows) |
| P3 | CRN per crop | **folded into `score`**: `bake_rows` on each crop's own `[::3]` rows with E3's T2 stats, in memory (no baked crop cache) |
| P4 | `position_shift score` — A3-ign / A3 / A0 × 5 on every arm, gate X10(b) (`full` == saved protocol-B scores) | **built** |
| P5 | `position_shift prior` + `readout` — §24 rule | **built** |
| P6 | `colab/v2/df8_position_shift.ipynb` | **run 2026-10-15**, X10 (a)–(d) pass |
| P7 | `position_shift same-frame` — post-hoc, printed only: same frames scored in the crop vs in the full clip, `p_T2` as ruler | **built + run locally 2026-10-15** (+2 tests) |

## 1. The question

`p_T2` — a cubic in relative position `t/T`, fit on T2, no pixels — beats every v2 model on DoTA-CAP (0.8256 vs
A3-ign 0.8120, second look). Per clip it is nearly uncorrelated with the model (r ≈ 0.14), and the dev error
analysis (`REPORT_V2_RESULTS.md` §6.2) says it wins through DoTA's cutting convention: accidents sit at 0.55–0.70 of
the clip, and where they do not, A3 wins by a wide margin (+0.266 when the centre is before 0.40). That is
observational: those clips differ in more than position. DF8 **moves the accident inside the same clip** and asks:

> Does `p_T2` still beat the model when the accident is not where DoTA puts it?

Proposal's sentence (kept): *if `p_T2` is still ≥ the model on the shifted set, the "position is an artefact"
argument is wrong.*

## 2. What the data allows (measured 2026-10-11 on the frozen labels, no scores read)

DoTA-CAP clips are short and their accidents long: native 10 fps, clip median **99** frames, span median **28**
(share 0.30), context before the span median **35**, after **29**. Measured on CAP-dev (569) / CAP-eval (560):

**The proposal's design is infeasible.** Placing the span *centre* at {0.1, 0.3, 0.5, 0.7, 0.9} of a fixed-length
crop with the whole span inside fits all five placements on **0–3 clips** for any L in 3–12 s (c = 0.1 needs the span
≤ 0.2 L *and* long post-context). Three placements {0.25, 0.5, 0.75} fit only 71–108 clips (and only short accidents —
a selection that would decide the answer).

**What fits: a same-length position swap inside each clip.** Keep the whole span; per clip set the context budget
`m = min(pre, post) + K` (K = 3 frames on the near side) and crop to `L = span + m` four ways:

| arm | frames kept before / after the span | achieved centre (CAP-dev, median [p10, p90]) |
|---|---|---|
| `full` | the whole clip (reference) | 0.51 |
| `control` | `m` split in the clip's own pre : post ratio | 0.51 |
| `early` | `K` / `m − K` | **0.30 [0.20, 0.41]** |
| `late` | `m − K` / `K` | **0.70 [0.59, 0.80]** |

With `min(pre, post) ≥ 10` frames (≥ 1 s of context each side): **491 / 569 CAP-dev clips**, L median 57 frames
(5.7 s). Every clip contributes all four arms, so every contrast is within-clip and paired.

The range is **0.30 ↔ 0.70, not 0.1 ↔ 0.9**. It still straddles `p_T2`'s peak (≈ 0.6) on one side and its fall-off on
the other, which is what the question needs; the thesis must say the range was limited by the data.

## 3. How each arm is built (no training)

- **CLIP rows `x`**: slice `DoTA_s1_ncc` to `[c0, c0 + L)` — exact.
- **V2-S rows `u`**: the cache is stride 1, so a step whose 1.5 s causal window lies inside the crop is exact by
  slicing. The first ≤ 15 frames of each crop are not: their window reaches back before `c0`, i.e. into context the
  crop removed (in `early`, into the frames just before the onset). **D2** decides how they are handled.
- **CRN R2** re-baked on each crop (`build_v2_inputs apply`, the E3 T2-train statistics, never `fit`): the per-clip
  median must be the crop's, or the comparison leaks the full clip (pending (ay)).
- **Scoring**: protocol B on the crop (`crop[::3]`, whole crop, interpolated to the crop's native frames), the same
  `score_steps` / `to_native` as `protocol_b_eval`.
- **`p_T2`**: the T2-fit cubic evaluated at `t / L` of the crop — never refit on DoTA (refitting is in-sample).
- **Labels**: the native labels sliced to the crop.

## 4. Read-out and the rule (to be frozen as addendum §24)

Primary model **A3-ign** (5 seeds, E4-0 checkpoints, seed-averaged scores). All intervals: cluster bootstrap over
source videos (B = 10,000), paired by clip; per-seed means printed.

- **R1 — the decisive read:** on the **shifted set** (each clip's mean of its `early` and `late` AUCs), the Δ
  `A3-ign − p_T2`:
  - entirely **above 0** → "On DoTA-CAP-dev, when the accident is moved within the clip, the model beats the position
    prior: `p_T2`'s advantage comes from where DoTA cuts its clips."
  - entirely **below 0** → "`p_T2` still beats the model with the accident moved: the clip-cutting argument is
    wrong" (the proposal's falsifier).
  - contains 0 → "Neither wins once the accident is moved."
- **R2 — the model's own position cost:** `control − early` and `control − late` for A3-ign (and `early − late`).
  Wording: "position-sensitive" only if an interval excludes 0; never "position-invariant" (a null is "not detectable
  at this size").
- **R3 — context cost:** `full − control` (shorter context, same position). Printed; it separates starvation from
  position (pending (ay)).
- **R4 — `p_T2` per arm**, printed: it must fall on `early` by construction; how far is the size of the artefact.
- **Printed only:** A3 and A0 per arm (15 more checkpoints), `f_2` per arm, by span-length bins, D2's sensitivity
  read, the 78 excluded clips' full-clip macros.
- Nothing here changes v2's model or any earlier verdict.

## 5. Risks and what each costs

| risk | effect | handling |
|---|---|---|
| **Content, not only position, moves**: `early` keeps the aftermath, `late` keeps pre-accident driving | an `early`/`late` gap can be "aftermath looks abnormal", not position | R1 averages the two mirror arms; R2 prints both sides; the thesis names the confound |
| Crops are short (5.7 s) and anomaly-heavy (share ≈ 0.5 vs 0.30) | CRN's median moves toward the anomaly; MIL-style scores lose contrast | `control` has the same L and share → R3 measures it; R1/R2 compare at equal L |
| Selection: 78 clips with < 1 s context on one side are dropped | the 491 are not all of CAP-dev | print `full` macros of kept vs dropped; never call the result "DoTA-CAP" without "(shift subset, n = 491)" |
| V2-S head windows (≤ 1.5 s) see removed frames | `early` would be partly un-shifted for `u` | D2 |
| Range 0.30–0.70 only | weaker than 0.1–0.9 | stated; it is what the data allows |
| A second read of DoTA-CAP-dev | dev was used for E3/E4-0 selection | no selection happens here (no arm is chosen); dev is the only open set; D1 |

## 6. Choices for the user (sign before build)

| # | choice | recommended | alternative |
|---|---|---|---|
| D1 | set | **DoTA-CAP-dev only** (open; nothing is selected on it) | also CAP-eval as a third look (not recommended: opened twice already) |
| D2 | V2-S rows in the first 1.5 s of a crop | **re-extract** those windows from the CAP frames with the extractor's own rule (clamp to the crop's first frame) = what a clip starting there would see; print the leaky slice beside it | slice the cache only (cheaper, leaks removed context into `early`'s onset) |
| D3 | checkpoints | **A3-ign ×5 primary; A3 ×5 and A0 ×5 printed** | A3-ign only |
| D4 | geometry | **K = 3 frames, min context 10 frames** (491 clips) | min 15 (440 clips, cleaner shift 0.29/0.71) |
| D5 | accept 0.30 ↔ 0.70 as the shift range | **yes** (measured: 0.1–0.9 fits ≤ 3 clips) | add an exploratory 3-placement set on the 71–108 clips that fit, printed only |
| D6 | R1's set definition | **mean of `early` and `late` per clip** | `early` alone (the side where `p_T2` is weak; biased toward R1 "above 0") |

## 7. Budget

No training. Per arm: slice + bake 491 crops; scoring 15 checkpoints × 3 new arms × 491 short clips (forward only,
~20 steps each) — minutes on a GPU. D2's re-extraction: ≤ 5 VideoMAE windows per crop × 491 × 3 arms ≈ 7 k windows,
plus reading the CAP frames of 491 clips. One Colab session.

## Appendix A — read-outs

### A.1 DF8 run, 2026-10-15 (Colab; `outputs/v2/REPORTS/v22_df8/df8_readout.md`)

- **X10 all pass:** J10 1740 × 15; `full` == saved scores (max 3.5e-07); heads 491/491, 7,524 windows, min cos
  0.99935 (thin margin over 0.999); 491 clips.
- **Verdict (X8): R1 > 0, X7 fails.** R1 **+0.1201 [+0.0974, +0.1413]** (full clips −0.0351 [−0.0613, −0.0098]).
  R2 early − control −0.1251 [−0.1405, −0.1096], late − control +0.0744 [+0.0573, +0.0912], early − late −0.1994.
  R3 +0.0264.
- Macro A3-ign full / control / early / late = 0.793 / 0.767 / 0.642 / 0.841; `p_T2` 0.829 / 0.874 / **0.376** /
  0.867. R1 is driven entirely by the prior's collapse on early; on control and late the prior still leads.
- X9: `f_2` early 0.45–0.48 for every model → `f_2` is tied to DoTA's placement (pending (bn)). R1 ≥ 4 s span
  −0.015 ∋ 0 (geometry). D2 sensitivity +0.133. Kept ≈ dropped.

### A.2 Post-hoc same-frame read, 2026-10-15 (local; `df8_same_frame.md`; printed only, decides nothing)

Asked by the user: "is A3-ign learning position?". Same frames, crop-scored minus full-clip-scored AUC:
A3-ign control / early / late **+0.056 / −0.077 / +0.038**; `p_T2` +0.148 / −0.328 / −0.053; A0 ≤ 0.017. Per-clip
tracking of the ruler: A3-ign r −0.03 / +0.21 / +0.23 (slope ≤ 0.18); A0 r ≤ 0. Early − late (−0.199) = frame set
−0.084 + move −0.115. **Reading:** not a position prior (≤ ¼–⅓ of the ruler's shift, opposite sign on late); a weak
position-like component remains (RoPE + 25-step window ≥ crop length → edges visible); A0 with the same trunk barely
moves → the extra sensitivity enters via CRN's crop median / V2-S's clamped window (inferred, not measured). The
direct test (CRN reference from the full clip) was not run, by the user's decision.
