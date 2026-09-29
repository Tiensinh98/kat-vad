# KAT-VAD v2 — Exploration plan: freeze → kill-switch → E0–E2 → build → pilot → go/no-go for E3

**Branch:** `v2` (tách từ `main@cc39882`; code = v1 + tool v2). Mọi code của plan này nằm trên `v2`.
**Nguồn:** `core/docs/v2/KAT_VAD_PROPOSAL_v2.md` (§4, §7, §10) + `KAT_VAD_v2_ARCHITECTURE.md`.
**Pre-registration:** `core/docs/v2/PREREG_ADDENDUM.md` — D1–D6 (P0), Amendment 1 D7–D9 (§6), impl. choices I1–I9 (§7),
Amendment 2 D10–D11 + J1–J9 (§8). Proposal thắng plan ở phần *what*; addendum thắng proposal ở những điểm nó sửa.
**Last updated:** 2026-09-29.

---

## 0. Trạng thái hiện tại (đọc cái này trước)

Ký hiệu: ✅ xong · 🟡 đang làm / chờ chạy · ⛔ bị chặn · ⬜ chưa bắt đầu

| Phase | Trạng thái | Kết quả / việc còn lại | Commit |
|---|:-:|---|---|
| **P0** Freeze | ✅ | Splits T2-val 219 src, DoTA-dev 702 / eval 700 (sealed); addendum D1–D6 | `7422975` |
| **P1** Kill-switch K | ✅ | **GO** (+0.13), nhưng K-pos: chỉ **+0.043 [+0.023, +0.064]** vượt CLIP + position | `8001cf1`, `7b20444` |
| **P2** E0 / E2(a–c) | ✅ (s8) | CRN = **R1** (mechanical; metric E2(b) bị position confound), E2(c) transfer **+0.030 [+0.020, +0.040]**. Nhánh s3 (D11) chỉ chạy nếu E1 chọn B/C | `7b20444` |
| **P3** E1 | 🟡 | Code + notebook xong. **Chờ user chạy `colab/v2/p3_e1.ipynb`** | `0ab760d` |
| **P3** E0b | ⬜ | Chưa có notebook. Cần xác nhận ckpt phase-5 + lcurve 25/50 % trên Drive | — |
| **P4** Encoder E2(d) | ⛔ | Chặn bởi **D9** (endpoint motion chưa chốt). Chỉ extract được trên T2 | — |
| **P5** Build v2 | ⬜ | Không bị chặn — làm được ngay (local CPU) | — |
| **P6** Pilot | ⬜ | Cần P4 + P5 | — |
| **P7** Go/no-go | ⬜ | Cần P6 | — |

**Việc tiếp theo, theo thứ tự:**
1. **User:** chạy `colab/v2/p3_e1.ipynb`, dán `e1_readout.md` → điền Appendix A, quyết định có chạy D11 không.
2. **Claude (song song):** P5 build ở local.
3. **User chốt D9** (endpoint motion; đề xuất ở §6 P4) → mở khoá P4.
4. **Claude:** notebook E0b khi user xác nhận ckpt trên Drive.

**Test suite hiện tại:** 715 collected, 0 fail (2026-09-29).

---

## 1. Summary

- Mục tiêu: trả lời *"v2 có đáng chạy factorial 20 run (E3) không, và với cấu hình nào?"* (reference CRN, encoder,
  protocol DoTA) — **không train để quyết định**, chỉ probe + re-score + 1 pilot kỹ thuật.
- Rủi ro lớn nhất — motion stream không có signal — đã kiểm ở P1: **có signal, nhưng nhỏ** (≈ ⅓ con số K ban đầu, phần
  còn lại là position). Motion **không đo được trên DoTA** (không còn pixel) → mọi claim motion là in-domain T2.
- CRN đo được trên DoTA (CLIP-only). Protocol DoTA (stride/window) quyết định bởi E1.
- Không có loss mới, không tune gì theo Δ trên DoTA-dev (lesson 14).

## 2. Assumptions & Constraints

| # | Assumption / constraint | Trạng thái |
|---|---|---|
| A1 | DADA-2000 original = **30 fps** (chỉ có trong literature) | Vẫn là assumption; ghi trong mọi read-out |
| A2 | Data trên Drive | **DoTA: không còn pixel.** Còn CLIP `DoTA_s1_ncc` (1,397 clip, verified `s1[::8] == DoTA_s8_ncc` 30/30) + `DoTA_s8_ncc` + `labels_s8` → mọi stride CLIP đều dựng được; **VideoMAE trên DoTA thì không**. DADA original đủ frame |
| A3 | Compute = Colab (GPU); dev = macOS CPU, test không cần data | Mọi bước cần data thật → giao notebook Colab, không kéo data về local |
| A4 | DoTA: 1,402 clip / **179 video gốc** (max 19 clip/video) | Split và bootstrap group theo video gốc (D3) |
| C1 | Không đụng `LaGoVAD-PreVAD/`; code chỉ trong `core/` | — |
| C2 | Cache gắn transform + stride (C2/C13) | s3 = `s1[::3]` in-memory, không ghi cache mới |
| C3 | DoTA-eval và T2-test mở một lần (final, sau E3) | `load_split(..., final=True)` là cửa duy nhất (`SealedSplitError`) |

## 3. Deviations từ run order của thầy (§10.2) và các amendment

| # | Nội dung | Ở đâu |
|---|---|---|
| D1 | Kill-switch K chạy ngay sau freeze, trước E1 | addendum §2 |
| D2 | Reference CRN lấy ở stride mà E1 chọn (E2(b) ở cả s8 và s3) | §2; khôi phục có điều kiện bởi D11 |
| D3 | Mọi bootstrap DoTA resample **video gốc** (cluster), kể cả interval quyết định của E1 | §2 |
| D4 | Pilot P6: seed **2099**, 4 arm, chỉ đọc cơ chế + T2-val, **không in DoTA-dev** | §2 |
| D5 | A0 regression: A0 trên code v2 phải khớp KIP-off phase-4 | §2 |
| D6 | Temporal-shuffle control cho encoder được chọn (chỉ diagnostic) | §2 |
| D7 | K chạy trên T2 thay vì DoTA-dev (không có pixel DoTA) | Amendment 1, §6 |
| ~~D8~~ | ~~E1 bỏ~~ — **đảo lại bởi D10** | §6 → §8 |
| D9 | Motion arms (A2/A3) không có endpoint DoTA; **endpoint chưa chốt** | §6 — **đang mở** |
| D10 | E1 chạy lại từ `DoTA_s1_ncc` (user duyệt, chưa qua thầy) | Amendment 2, §8 |
| D11 | E2(b)/(c) đọc lại ở s3 **chỉ khi** E1 chọn B hoặc C | §8 |

Mọi rule quyết định khác (rule chọn reference, adoption E3, `F`) giữ nguyên proposal.

## 4. Plan Metadata

- **Plan type:** Research campaign — no-training probes + model build + engineering pilot
- **Storage path:** `@.project/plans/katvad-v2-e0-e2.md`

## 5. Phases Overview

| Phase | Goal | Phụ thuộc | Train? | Chạy ở |
|---|---|---|:-:|---|
| P0 Freeze | Split + rules + addendum; DoTA-eval guard | — | no | local |
| P1 Kill-switch K | VideoMAE-B có signal không (T2) | P0 | no | Colab |
| P2 Audits | E0, E2(a–c): reference CRN | P0 | no | local |
| P3 Protocol | E1 (A/B/C) + E0b, DoTA-dev | P0 | no | Colab |
| P4 Encoder | E2(d): B vs S, shuffle control, full T2 extract | P1 GO, **D9** | no | Colab |
| P5 Build v2 | Config, stats, dataset, CRN, `MotionResidual`, diagnostics, tests | P2 | — | local |
| P6 Pilot | 1 seed × 4 arm, cơ chế + guardrails + A0 regression | P4, P5 | yes (4 run) | Colab |
| P7 Go/no-go | Checklist → runbook E3 hoặc báo cáo dừng | P6 | no | — |
| (E3) | 2 × 2 × 5 seeds = 20 run | P7 GO | yes | Colab |

## 6. Detailed Tasks by Phase

### P0 — Freeze ✅ (`7422975`, 2026-09-27)

- [x] `core/tools/freeze_splits.py` + `core/data/v2_splits.py`: T2-val = 15 % source video T2-train (grouped theo type);
  DoTA-dev/eval 50/50 grouped theo video gốc, stratified theo share bin. Output `core/splits/v2/*.txt` + `SPLITS_MANIFEST.json` (sha1).
- [x] DoTA-eval guard: `load_split` raise `SealedSplitError` trừ khi `final=True`.
- [x] `core/docs/v2/PREREG_ADDENDUM.md` D1–D6 + rule K + pilot read-out + go/no-go.
- [x] CLAUDE.md §14.0.1 hàng `v2`, banner branch trong `activeContext.md`.

### P1 — Kill-switch K ✅ (`8001cf1` K, `7b20444` K-pos; 2026-09-28)

Thiết kế đã sửa theo D7 (addendum §6.1): ~300 source T2-train, whole source s8, probe (i′) source-grouped + (ii′) type-grouped.

- [x] VideoMAE V2 vendored: `core/models/videomae_v2.py` (pin HF commit + sha256, `weights_only=True`).
- [x] `core/tools/extract_video_features.py`: causal 16 f @ 10 fps, squash 224², pad đầu clip bằng lặp frame đầu, cache có manifest.
- [x] `core/tools/kill_switch_probe.py` (`pick` / `run` / `diag`) + tests; runbook `colab/v2/p1_kill_switch.ipynb`.
- [x] **K = GO**: `u` 0.760 vs CLIP 0.615, Δ +0.13; positive control qua.
- [x] **K-pos** (§6.2, printed): NOT_PAD; BEYOND_POSITION — `u` thêm **+0.043 [+0.023, +0.064]** vượt CLIP + position.
  Position ruler: 0.730 trên whole source, **0.575** trên T2-val windows.

### P2 — Audits E0 / E2(a–c) ✅ ở s8 (`7b20444`, 2026-09-28)

- [x] `core/crn/reference.py`: R1–R4 (R4 strictly-past, warm-up 8), dùng chung cho probe và model P5.
- [x] `core/tools/crn_select.py`: E0 (gộp vào đây thay vì `rate_audit.py`), E2(a), E2(b), E2(c) + tests. Impl. choices I1–I9 (§7).
- [x] **E0** 3.00× → E1 không bị skip.
- [x] **E2(a)** bins DoTA-dev = freeze (343/247/93/19); coverage 0.223 → metric `r`.
- [x] **E2(b)** → **R1** (mechanical). Caveat: `−f` alone 0.764 > `r` 0.690 → lựa chọn reference không được xác định (pending (ah)).
- [x] **E2(c)** CRN − raw transfer **+0.030 [+0.020, +0.040]** → veto qua. Mọi reference +0.025…+0.034.
- [ ] **(D11, có điều kiện)** Nếu E1 chọn B/C: thêm `--stride` cho `crn_select`, đọc lại E2(b)/(c) trên DoTA-dev ở s3
  (từ `s1[::3]`), notebook Colab. Nếu E1 giữ A: đóng task này.

Record: `core/docs/v2/RESULTS_E2_CRN.md`.

### P3 — Protocol: E1 🟡 + E0b ⬜

**E1** (Amendment 2, J1–J9, `0ab760d`):

- [x] `core/tools/rate_matched_eval.py`: A = `s1[::8]` whole · B = `s1[::3]` whole · C = `s1[::3]` W20 hop 4 overlap-average;
  nội suy về native frame; label native từ annotation theo độ dài s1; seed-average; paired Δ vs A, cluster bootstrap
  10k theo video gốc; rule J7; position ruler + per-seed + per-bin + micro in kèm (không quyết định).
- [x] J9 regression gate: A step-level phải khớp `max_score` phase-4 (atol 1e-4) trước khi chấm B/C.
- [x] `cluster_bootstrap_ci` chuyển vào `core/metrics.py` (crn_select dùng chung). +24 test; smoke end-to-end OK.
- [x] Runbook `colab/v2/p3_e1.ipynb` (stage 702 file dev về `/content`, spot check s1 vs s8).
- [ ] **User chạy notebook** → điền Appendix A. Nếu J9 fail: dừng, debug, không nới tolerance.
- [ ] Ghi `core/docs/v2/RESULTS_E1.md` + cập nhật memory bank.

**E0b** (descriptive, không quyết định gì):

- [ ] User xác nhận trên Drive: ckpt phase-4 KIP-on (v1), phase-5 KIP-on (v2 zscore), lcurve 25 % / 50 %.
- [ ] Notebook re-score các ckpt đó trên DoTA-dev (s8, protocol cũ) → pooled SD của 4 contrast (≤ 8 df) →
  `MDE_dev = 2.776 · SD_pooled / √5`. Tái dùng harness E1 (arm A) thay vì viết mới.

### P4 — Encoder choice E2(d) ⛔ (chặn bởi D9)

**D9 cần chốt trước** (người chốt: user; thầy chỉ khi user muốn). Đề xuất:
endpoint motion = **macro trên T2-val windows**, paired A2/A3 vs A0/A1 theo seed, **in position ruler 0.575 bên cạnh**.
Chốt thành Amendment 3 trong addendum **trước** khi đọc bất kỳ số motion nào.

Sau khi D9 chốt:

- [ ] Extract VideoMAE-B và -S trên T2-train subsample + T2-val (stride 8). **Không có DoTA** (không pixel).
- [ ] Probe đúng representation mỗi arm: A2 `c·(u − m_u)⊘σ_u`; A3 `c·ũ⊘σ_u` với reference R1. Rule eligibility thay
  cho rule DoTA-dev của proposal = rule theo endpoint D9.
- [ ] Lặp E2(b) trên `[x ; u]` với encoder đã chọn (in `−f` bên cạnh, pending (ah)).
- [ ] **D6 shuffle control** (xáo 16 frame trong clip) → chỉ report.
- [ ] Extract full T2 (train + val + test) cho encoder đã chọn.

**Deliverables:** `outputs/REPORTS/v2_E2d/` + read-out: encoder, eligibility, shuffle Δ; full T2 cache.

### P5 — Build v2 model ⬜ (không bị chặn; local CPU)

Trước khi sửa symbol nào: `trace_call_path` + báo blast radius; load `meta-index.md`; context7 cho API ngoài.

- [ ] Config `v2` trong `core/config.py`: `crn.enabled`, `crn.reference ∈ {R1..R4}` (mặc định R1), `crn.warmup`,
  `motion.enabled`, `motion.feature_dir`, `motion.encoder`, `motion.stats_path`. Raise nếu v2 bật cùng `kip.enabled`.
- [ ] `core/tools/build_v2_stats.py`: từ T2-train **trừ T2-val** → `s`, `m_u`, `σ_u`, `c`; manifest `train_ids_sha1`; mismatch → raise.
- [ ] Data layer: reference CRN tính trên source video (train) / clip (test) bằng `core/crn/reference.py`; load `u` song song `x`,
  assert độ dài khớp từng video.
- [ ] Model: `MotionResidual` (Linear `d_v→512`, weight + bias zero-init) cộng vào input trước temporal encoder; không LayerNorm.
  Log `ρ_u` và `‖W_u‖` mỗi 50 step.
- [ ] Checkpoint qua `ckpt_compat` (C5): A0 load ckpt KIP-off cũ được; A2/A3 thiếu `W_u` → raise.
- [ ] Diagnostics mọi arm: source-shortcut AUC trên `V^t` và `y^bin`, position probe (R² `t/T`) trên `V^t`, macro theo share bin.
- [ ] Tests: `TestV2A0IsKipOff`; `W_u = 0` ⇒ A2 ≡ A0 lúc init; `s` giữ `E‖x̃‖ = E‖x‖`; R4 streaming không nhìn tương lai;
  stats sha1 mismatch → raise.
- [ ] Quality gate §11 + full suite xanh (baseline 715). Docs `core/docs/v2/TRAINING_V2.md`. Lesson candidate.

### P6 — Pilot ⬜

**Goal:** chứng minh pipeline đúng về cơ chế. **Không** quyết định adoption.

- [ ] Seed **2099**, A0/A1/A2/A3, T2-train trừ T2-val, hyper-params proposal §8. Notebook Colab.
- [ ] Đọc **chỉ**: T2-val (macro, micro, clip oracle), guardrails (micro < oracle, macro ≥ micro, micro ≥ A0 − 0.01),
  `ρ_u` theo thời gian, `‖W_u‖` tăng từ 0, loss curves, clip-level AUC vs macro (C14), position R², source-shortcut.
- [ ] **DoTA-dev không in trong pilot** (lesson 14).
- [ ] **A0 regression (D5):** A0-pilot vs ckpt KIP-off phase-4 trên T2-val; chênh > ~0.02 → dừng, debug.
- [ ] `ρ_u` ≈ 0 suốt run → ghi nhận motion không được dùng; **không** sửa lr/`c`.

### P7 — Go/no-go cho E3 ⬜

- [ ] Checklist GO: splits committed; protocol chốt (E1); reference chốt (hoặc CRN dropped); encoder eligible theo D9
  (hoặc motion dropped); A0 regression pass; guardrails pass; không collapse; suite + quality gate xanh.
- [ ] Runbook E3 `colab/v2/e3_factorial.ipynb` (seeds 2024–2028 × arm còn sống), cost estimate từ pilot.
- [ ] Cập nhật memory bank; bảng quyết định P1–P6 (gửi thầy nếu user muốn).
- [ ] NO-GO → report bounded null / thành phần bị drop (proposal §10.3).

## 7. Risks & Mitigations

| Risk | Tác động | Giảm thiểu | Trạng thái |
|---|---|---|---|
| Motion signal chủ yếu là position | Claim "kinematics-aware" sai | K-pos; in position ruler cạnh mọi macro; D6 shuffle | Đã đo: ≈ ⅔ K là position |
| Motion không có endpoint DoTA | Không có claim zero-shot cho motion | D9 → endpoint T2-val windows | **Mở** |
| Metric detrended mang position prior (`−f`) | Chọn reference sai | In `−f`; E2(c) probe không thấy `t` | Đã thấy ở E2(b) |
| J9 fail (harness E1 lệch phase-4) | E1 không so được | Dừng, debug, không nới tolerance | Chờ chạy |
| Clip DoTA tương quan trong video | CI hẹp giả | D3 cluster bootstrap | Đã áp dụng |
| Dùng DoTA-dev nhiều lần (E1, E2, E3) | Selection optimism | DoTA-eval sealed; đo optimism dev − eval ở final | — |
| Sửa forward/dataset làm lệch baseline | Mọi Δ vô nghĩa | `TestV2A0IsKipOff` + A0 regression | P5/P6 |
| `ρ_u` ≈ 0 (motion không được dùng) | A2/A3 = A0 | Ghi nhận, không tune | P6 |
| Stats `σ_u/c/s` rò T2-val | Leak nhẹ | Build từ T2-train trừ val, bind sha1 | P5 |
| A1 (30 fps) sai | Clip motion sai tỉ lệ thời gian | Ghi assumption | Mở |

---

## Appendix A — Read-outs

| Phase | Kết quả | Commit | Ngày |
|---|---|---|---|
| P0 | T2-val 219 src / 645 win; DoTA-dev 702 clip / 93 vid, eval 700 / 86; `>70` bin chỉ 19 clip dev; addendum D1–D6 | `7422975` | 2026-09-27 |
| P1 K | **GO.** 295 src / 12,409 fr. `u` 0.760, `x` 0.615; Δ([x;u]−x) +0.132 [+0.109, +0.155] (i′), +0.126 [+0.101, +0.152] (ii′); control lower 0.740. **K-pos:** NOT_PAD (pad drop Δ +0.106 [+0.080, +0.130]); BEYOND_POSITION (`p` 0.730, `[x;p]` 0.734, `[x;u;p]` 0.777; Δ +0.043 [+0.023, +0.064] (i′), +0.039 [+0.018, +0.061] (ii′)). T2-val windows position ruler 0.575. `outputs/v2/DADA2000_orig/v2_K/` | Colab (code upload, `commit: UNKNOWN`) | 2026-09-28 |
| P2 E0/E2(a–c) | **R1 (mechanical), veto qua.** E2(b) `r` bị `−f` chi phối (0.764 > 0.690) → reference không được xác định. E2(c) +0.025…+0.034, mọi CI > 0. E0 3.00×. `core/docs/v2/RESULTS_E2_CRN.md` | `7b20444` | 2026-09-28 |
| P3 E1 | *Chờ chạy.* Label-only fact (smoke run): position ruler `t/N` trên DoTA-dev native ≈ 0.566 | `0ab760d` (harness) | — |
| P3 E0b | | | |
| P4 E2(d) | | | |
| P6 pilot | | | |
| P7 | | | |
