# v7 Step 0 — Latent Surprise Predictor on V2-S vectors: **NO-GO**

Rules: `core/docs/v7/PREREG_V7.md` §1 (L0-1 … L0-12, signed 2026-11-01 before any number). Raw read-out:
`outputs/v7/v7_lsp/step0/REPORTS/step0_readout.md` (run on Colab 2026-11-01; Drive `Thesis-V7/outputs/v7/v7_lsp/step0/`).
No detector was trained.

## 1. Decision: **NO-GO** (D2 failed)

| check | rule | value | result |
|---|---|---|---|
| D1 | cos(k = Δ) < cos(k = 1) − 0.01 | 0.9325 vs 0.9766 | pass |
| **D2** | LSP ≤ 0.90 × best fixed predictor | gain **+0.0981** over context-mean (bar 0.10) | **FAIL** |
| D3 Nexar | AUC(max s) > AUC(max p) | 0.7006 vs 0.6854 | pass (point; CI ∋ 0) |
| D3 DoTA | mean per-clip AUC(s) − AUC(p) > 0 | +0.0159 [+0.0067, +0.0253] | pass |

D2 missed its bar by 0.0019. The bar was fixed before the number and is not moved after it (lesson 14), so the
verdict stands.

## 2. What the numbers say

* **The vectors barely move inside 1.6 s.** cos(u_t, u_{t+16}) = 0.933 against 0.977 at one step. Over one
  non-overlapping clip, a global V2-S vector stays close to where it was.
* **The predictor learns a smoother, not dynamics.** The context mean (L_rec 0.0651) beats copy-last (0.0692), so
  averaging the past already beats the last observation; constant velocity is far worse (0.1761). The LSP (0.0587,
  K = 16, chosen by L0-7) is 15 % below copy-last but only 9.8 % below the context mean. Most of what it learns over
  copy-last, a fixed average also gets.
* **Standalone surprise is weak on Nexar and inverted on DoTA.** On `nexar_val`, ŝ ≈ p (0.70 vs 0.69, the difference's
  CI includes 0), and corr(s, p) on normals is 0.91: the LSP's surprise is mostly raw change. On DoTA-CAP-dev both s
  (0.445) and p (0.430) sit **below 0.5** on valid frames. Only their log ratio ρ is above (0.545 [0.524, 0.566]). The
  cubic position prior alone scores 0.820 on the same frames.
* **The small DoTA gain of s over p comes from ego clips** (+0.031 [+0.021, +0.042]). Non-ego is −0.005 [−0.020,
  +0.011].
* **Surprise statistics identify the source** (D4, printed): Nexar normals vs DoTA lead-ins, AUC 0.993, 0.29 above
  D3's anomaly AUC. That is the domain-shift risk of proposal §4.3, flagged.
* **Surprise peaks are not in the lead-in** (D5): 6.2 % of clips (p 5.6 %). With per-clip AUC < 0.5, the peaks fall
  mostly *after* the labelled window or at frames labelled normal outside it. This is a reading of two printed numbers,
  not tested.

## 3. Caveats

* One seed; one predictor size; K = 32 was trained but not chosen (its held-out loss is in `REPORTS/train.json`, not
  read here).
* DoTA AUCs are on valid frames only (19 clips dropped; the first 3.1 s of each clip has no full context).
* D3/D4/D5 are standalone reads. Proposal §3.2.4 point 4 argues that surprise helps inside a supervised head even when
  it is weak alone. Step 0 was designed not to test that, and P3 was not run.

## 4. What is not licensed

* No claim that "LSP adds evidence". The predictor is not demonstrably more than a smoother of V2-S global vectors.
* No claim about LSP inside the detector: P2/P3 were not built or run.

---

## 5. Amendment 1 (§1.1) — the same Step 0 on CLIP frame vectors: **NO-GO**

Raw: `outputs/v7/v7_lsp/step0_clip/REPORTS/step0_readout.md` (Colab 2026-11-01). Post-hoc, disclosed (§1.1).

| check | value | result |
|---|---|---|
| D1 | cos 0.9835 → 0.9494 at Δ | pass |
| **D2** | LSP 0.04127 vs context-mean **0.04110** (gain **−0.0041**); copy-last 0.04915 | **FAIL** |
| D3 Nexar | AUC(max s) 0.536 vs p 0.514 (CI ∋ 0) | pass (point) |
| **D3 DoTA** | s − p **−0.0118 [−0.0180, −0.0055]** | **FAIL** |

* **On CLIP the predictor is the context mean.** Its held-out loss is slightly *above* a fixed average of the same
  16 vectors. CLIP is slower still than V2-S (cos 0.95 at 1.6 s), and per-frame jitter makes averaging the best rule.
* **The fixed CLIP change is informative on DoTA; the learned model takes it away.** Copy-last surprise p scores
  0.736 per-clip frame AUC on DoTA-CAP-dev valid frames (ego 0.774, non-ego 0.687), against 0.430 for V2-S. The LSP's
  s scores lower (0.724), most on non-ego clips (−0.021). The cubic position prior on the same frames is 0.820, above
  both. ρ = 0.32 (inverted: frames that are *more* predictable than their raw change are the anomalous ones).
* **Nexar video-level surprise is near chance** (s 0.536, p 0.514), and **D4 = 0.9992**: CLIP surprise statistics
  name the dataset almost perfectly (C38's pattern).

## 6. Joint reading (§1 + §1.1)

On Nexar, neither per-step global vector carries normal dynamics that a fixed rule does not already capture within 1.6 s.
V2-S: the LSP beats the context mean by 9.8 % (bar 10 %). CLIP: it ties it (−0.4 %). The only standalone signal seen is
a **fixed** cue, raw CLIP change, at 0.736 on DoTA-CAP-dev, and the position prior beats it there. This is the outcome
proposal §4.5 anticipated as reportable: "global latents carry no exploitable normal-dynamics structure at this
scale", extended here from V2-S to CLIP. It was measured before any detector, so it says nothing about LSP inside MIL.
