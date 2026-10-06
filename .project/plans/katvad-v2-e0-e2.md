# KAT-VAD v2 — Exploration plan: freeze → kill-switch → E0–E2 → build → pilot → go/no-go for E3

**Branch:** `v2` (tách từ `main@cc39882`; code = v1 + tool v2). Mọi code của plan này nằm trên `v2`.
**Nguồn:** `core/docs/v2/KAT_VAD_PROPOSAL_v2.md` (§4, §7, §10) + `KAT_VAD_v2_ARCHITECTURE.md`.
**Pre-registration:** `core/docs/v2/PREREG_ADDENDUM.md` — D1–D6 (P0), Amendment 1 D7–D9 (§6), impl. choices I1–I9 (§7),
Amendment 2 D10–D11 + J1–J9 (§8). Proposal thắng plan ở phần *what*; addendum thắng proposal ở những điểm nó sửa.
**Last updated:** 2026-10-04.

---

## 0. Trạng thái hiện tại (đọc cái này trước)

Ký hiệu: ✅ xong · 🟡 đang làm / chờ chạy · ⛔ bị chặn · ⬜ chưa bắt đầu

| Phase | Trạng thái | Kết quả / việc còn lại | Commit |
|---|:-:|---|---|
| **P0** Freeze | ✅ | Splits T2-val 219 src, DoTA-dev 702 / eval 700 (sealed); addendum D1–D6 | `7422975` |
| **P1** Kill-switch K | ✅ | **GO** (+0.13), nhưng K-pos: chỉ **+0.043 [+0.023, +0.064]** vượt CLIP + position | `8001cf1`, `7b20444` |
| **P2** E0 / E2(a–c) | ✅ | **D11 ở s3: CRN = R2** (`r` 0.7092 vs R1 0.7081 — hoà trong nhiễu, luật chọn R2), E2(c) R2 **+0.037 [+0.027, +0.047]**. s8 cũ = R1. `RESULTS_E2_CRN.md` §D11 | `7b20444`, `a20781b` + D11 record |
| **P3** E1 | ✅ | **Adopt B** (DoTA s3 whole clip): Δ vs A **+0.033 [+0.019, +0.048]**, C +0.035 (tie → B); 3/3 seed dương. `core/docs/v2/RESULTS_E1.md` | `0ab760d` + Amendment 3 (`7ce8bf5`) |
| **P3** E0b | ⬜ | Chưa có notebook. Ckpt phase-5 + lcurve có thể cũng là snapshot (pending (aj)) → kiểm `global_step` trước | — |
| **P4** Encoder E2(d) | ✅ | **E2(d) = V2-S** (tie rule N7: A3 transfer B 0.1508 vs S 0.1325). Motion giữ. **Position cubic `p` = 0.852** > `u`, `[x;u]`; vượt position chỉ transfer +0.02…+0.04. **D6 (2026-10-05): ordered − shuffled trên `[x;u;p]` −0.0004 / −0.0003, CI ∋ 0 → theo G3 không được gọi là "motion", mà là "second (video) appearance encoder"**; shuffled `[x;u]` − `x` vẫn +0.14 / +0.12. `core/docs/v2/RESULTS_E2D.md` §D6 | `b6d937f`, `7b1889a`, `7ea0882`, `ef24a7a` |
| **P5** Build v2 | ✅ (trừ bake A2/A3) | Model + input cache (`a20781b`), `protocol_b_eval` / `v2_dataset` / `v2_diagnostics` (`7ea0882`; `protocol_b_eval` tái tạo E1 B 0.6628 đúng 4 chữ số). 2026-10-04: `v2_diagnostics --dota-split`, `e2d_shuffle`, `dota_cap extract --shuffle-seed`. Bake A2/A3 nằm trong notebook batch 2 | `a20781b`, `7ea0882` + (chưa commit) |
| **P6** Pilot | ✅ | Batch 1 (A0+A1): guardrails PASS, CRN shortcut `V^t` 0.9999 → 0.551; D5 EXPLAINED (Amendment 7). Batch 2 (A2+A3, V2-S): **O1 FAIL như đăng ký**, không phải C14 → **Amendment 8 (§17, O1′)**. **T2-test (Q4, 2026-10-05): A1/A2/A3 đều PASS O1′** (A2 Δ window +0.101, macro +0.075; A3 +0.117 / +0.051) → survivors, **Motion Stream giữ**, E3 = {A0–A3}. `core/docs/v2/RESULTS_P6_PILOT.md` | `1a2a2c5`, `573d3d9`, `0aee92a` |
| **P7** Go/no-go | 🟡 | **GO** (checklist §18). Amendment 9 (M1–M10) **nháp chờ ký**; harness `position_prior` + `e3_readout` + `colab/v2/e3_factorial.ipynb` built (chưa commit) | — |

**Việc tiếp theo (2026-10-05, sau D6 + T2-test O1′):** P7 checklist → Amendment 9 (cách E3 được tính, viết trước mọi số E3) → `core.tools.e3_readout` + G6 position reads → runbook `colab/v2/e3_factorial.ipynb` → user chạy 20 run. **User báo thầy D6** (claim "kinematics-aware" → "video appearance + CRN") trước khi đốt GPU.

**Việc tiếp theo cũ (2026-10-04, sau batch 1):** (1) user commit + upload code (Amendment 6–7, tools, notebooks); (2) ~~`p6_d5_t2test.ipynb`~~ xong: D5 EXPLAINED; chạy **`p4_s_full_t2_d6.ipynb`**; (3) step 2 của P4-finish xong → `p6_pilot_batch2.ipynb`; (4) paste `d5_t2test.md`, `shuffle_readout.md`, `batch2_readout.md` → P7 checklist + E3 harness (G6 position reads).

**Việc tiếp theo cũ (2026-10-03, giữ làm lịch sử):**
1. **D9 chờ MM-AU Phase 0** (`.project/plans/katvad-mmau-phase0.md`, notebook `colab/v2/mmau_p0.ipynb`): nếu CAP-DATA
   chứa pixel DoTA (nhánh P) → D9 = DoTA như proposal; không thì T2-val + MMAU-clean (nhánh C). User chạy P0b (nhóm `1-10`) trước.
2. **Claude:** P5 diagnostics tool; đường chấm DoTA protocol B cho arm v2 (A1 trên cache bake, hiện `core.evaluate`
   chỉ chấm s8 và `rate_matched_eval` chỉ đọc CLIP thô); notebook E0b (sau khi user kiểm J10 của ckpt phase-5/lcurve).
3. A1 (R2) bake + pilot được ngay khi (2) xong.

**Test suite hiện tại:** 857 passed, 0 fail (2026-10-04).

**Drive layout (từ 2026-09-29):** code + outputs v2 ở `Thesis-V2/` (`kat-vad/`, `outputs/`); data, cache, ckpts và
run v1 (phase 4) ở `Thesis/`, chỉ đọc.

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
| D12 | Ckpt E1 = s2024 (phase 4) + s2025/s2026 **train lại** đúng lệnh phase 4; J9 → **J9′** (s1[::8] vs cache s8, cùng ckpt, không dùng `results.json` cũ); **J10** ckpt phải là bước cuối | Amendment 3, §9 |

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

### P2 — Audits E0 / E2(a–c) ✅ (s8 `7b20444` 2026-09-28; D11 s3 2026-09-30)

- [x] `core/crn/reference.py`: R1–R4 (R4 strictly-past, warm-up 8), dùng chung cho probe và model P5.
- [x] `core/tools/crn_select.py`: E0 (gộp vào đây thay vì `rate_audit.py`), E2(a), E2(b), E2(c) + tests. Impl. choices I1–I9 (§7).
- [x] **E0** 3.00× → E1 không bị skip.
- [x] **E2(a)** bins DoTA-dev = freeze (343/247/93/19); coverage 0.223 → metric `r`.
- [x] **E2(b)** → **R1** (mechanical). Caveat: `−f` alone 0.764 > `r` 0.690 → lựa chọn reference không được xác định (pending (ah)).
- [x] **E2(c)** CRN − raw transfer **+0.030 [+0.020, +0.040]** → veto qua. Mọi reference +0.025…+0.034.
- [x] **(D11 — KÍCH HOẠT, E1 chọn B) — code xong (uncommitted):** `crn_select --dota-s1-dir --dota-stride N`
  (`load_dota_dev_s1`: `s1[::N]`, label từ annotation qua `core.data.dota.resized_frame_labels`); rate audit theo stride
  từng corpus; `build_v2_inputs apply --stride`. +6 test. Notebook `colab/v2/p2_e2_s3.ipynb` có **cổng regression**
  (đường s8 == đường s1 ở stride 8, mọi số) — đã thử local bằng s1 dựng từ s8: **0 khác biệt** (label annotation ==
  `labels_s8` trên 702 clip).
- [x] **D11 read-out (s3):** cổng s8 == s1@8 qua (mọi bảng trùng, chỉ khác nhãn corpus). `r` DoTA-dev R1 0.7081 ·
  **R2 0.7092** · R3 0.6997 · R4 0.7049 (R4 giờ eligible). Verdict **R2**; E2(c) R2 +0.0368 [+0.027, +0.047] → veto qua.
  R1 vs R2 = hoà (+0.0011 s3, −0.0003 s8, CI ±0.015) — không thêm luật tie sau khi thấy số (lesson 14). `−f` 0.793 > mọi
  `r` → E2(b) vẫn bị position chi phối; claim CRN dựa vào E2(c). **Reference CRN = R2.**

Record: `core/docs/v2/RESULTS_E2_CRN.md`.

### P3 — Protocol: E1 🟡 + E0b ⬜

**E1** (Amendment 2, J1–J9, `0ab760d`):

- [x] `core/tools/rate_matched_eval.py`: A = `s1[::8]` whole · B = `s1[::3]` whole · C = `s1[::3]` W20 hop 4 overlap-average;
  nội suy về native frame; label native từ annotation theo độ dài s1; seed-average; paired Δ vs A, cluster bootstrap
  10k theo video gốc; rule J7; position ruler + per-seed + per-bin + micro in kèm (không quyết định).
- [x] J9 regression gate (bản đầu): A step-level khớp `max_score` phase-4 (atol 1e-4) trước khi chấm B/C.
- [x] `cluster_bootstrap_ci` chuyển vào `core/metrics.py` (crn_select dùng chung). +24 test; smoke end-to-end OK.
- [x] Runbook `colab/v2/p3_e1.ipynb` (layout Thesis / Thesis-V2, stream log subprocess, cài `av einops faiss-cpu`).
- [x] **Lần chạy 1 (2026-09-29): dừng ở J9.** s2024 pass 702/702 (harness = `core.evaluate`); s2025 fail 702/702.
  Nguyên nhân: ckpt Drive s2025 step 510, s2026 step 1530, `metrics.jsonl` đủ 2040 → bản cuối chưa từng lên Drive (pending (aj)).
  Chưa có số E1 nào được tính.
- [x] **Amendment 3** (§9): D12 train lại s2025/s2026; J9′ (s1[::8] vs cache s8 cùng ckpt, mọi step); J10 (ckpt step == metrics).
  Harness sửa theo (+5 test, smoke OK); runbook train lại `colab/v2/p3_retrain_kipoff.ipynb` (so `config.yaml` với phase 4,
  verify step + sha256 trên Drive).
- [x] User chạy `p3_retrain_kipoff.ipynb` (VERIFIED) → `p3_e1.ipynb`. J10 (3 × step 2040) và J9′ pass.
- [x] **Read-out (2026-09-30): adopt B.** A 0.6294 · B 0.6628 (Δ +0.0334 [+0.019, +0.048]) · C 0.6644 (Δ +0.0350);
  ruler 0.566. Δ theo bin: <30 % +0.040 → >70 % +0.012. `core/docs/v2/RESULTS_E1.md`.
- [x] Protocol DoTA cho P6/E3 = **B** (s1[::3] cả clip, nội suy native), in protocol A bên cạnh.

**E0b** (descriptive, không quyết định gì):

- [ ] User xác nhận trên Drive: ckpt phase-4 KIP-on (v1), phase-5 KIP-on (v2 zscore), lcurve 25 % / 50 % — **và mỗi ckpt
  qua J10** (`global_step` == dòng cuối `metrics.jsonl`); cùng code sync với phase 4 nên có thể cũng là snapshot.
  Contrast của E0b cần ckpt KIP-off: dùng s2024 + s2025/s2026 train lại.
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
- [x] E2(d) đọc xong 2026-10-04: **V2-S** (`RESULTS_E2D.md`). Amendment 6 (§15) chốt G1–G7.
- [ ] **D6 shuffle control** (xáo 16 frame trong clip) → chỉ report. Tool `e2d_shuffle` + `dota_cap extract --shuffle-seed`; notebook `p4_s_full_t2_d6.ipynb` step 3–4.
- [ ] Extract full T2 (train + val + test) cho encoder đã chọn — `p4_s_full_t2_d6.ipynb` step 2.

**Deliverables:** `outputs/REPORTS/v2_E2d/` + read-out: encoder, eligibility, shuffle Δ; full T2 cache.

### P5 — Build v2 model 🟡 (local CPU; **uncommitted** 2026-09-30)

Thiết kế (user duyệt 2026-09-29, addendum §10 K1–K6): CRN + motion scaling **bake offline** thành v2 input cache;
model chỉ thêm `W_u`. Không sửa `dataset.py` / `synthesis.py` / `collate.py`.

- [x] Config: `model.motion_dim` (0 = tắt) + section `v2` (`crn`, `motion`); `validate_v2` (giá trị hợp lệ, motion ⇔ dim,
  dim khớp encoder, raise nếu v2 + KIP). `"v2"` thêm vào `ARCH_SECTIONS` → evaluate đọc v2 từ checkpoint.
- [x] `core/data/v2_inputs.py`: `fit_stats` (`s`, `c`, `m_u`, `σ_u` trên T2-train trừ val), `bake_rows`, manifest,
  `check_input_manifest`. `core/tools/build_v2_inputs.py` (`fit` / `apply`), ghi atomic, manifest có `train_ids_sha1`.
  **Chạy thử local (A1 R1, trước D11; cache không giữ lại):** 1272 train + 219 val source, bake 1861, `c` 0.436 (spec ≈ 0.44), `s` 3.82; window T2 cắt
  từ cache bake khớp công thức (2e-7).
- [x] `core/models/motion_residual.py`: `Linear d_v→512` zero-init, không LayerNorm, padding không nhận motion, `ρ_u`.
  Gắn vào `KATVAD.forward` khi `motion_dim > 0`. `metrics.jsonl` có `motion_share` + `w_u_norm` mỗi batch (không vào loss).
- [x] `train` / `evaluate` kiểm manifest cache ↔ `cfg.v2` (K4).
- [x] Ckpt compat: ckpt cũ (không `motion_dim` / `v2`) load như A0; A0 state vào model motion (strict) → raise.
- [x] Tests `core/tests/test_v2_model.py` (28): config, zero-init ≡ no-motion (cùng weights), padding, `s` giữ
  `E‖x̃‖ = E‖x‖`, cột motion RMS = `c`, R4 không nhìn tương lai, manifest, **e2e A3 train → evaluate** trên fixture.
  Full suite 748 / 0 fail; ruff / mypy / pyright / bandit / pycycle sạch.
- [x] Docs `core/docs/v2/TRAINING_V2.md`. Notebook train lại: §3 so config theo YAML, cho phép key v2-only ở giá trị tắt
  (kiểm với config phase-4 thật: 0 khác biệt cho s2025/s2026).
- [x] Lesson candidate (ak) (quy ước `padding_mask`).
- [ ] **Diagnostics tool** (proposal §10.1): source-shortcut AUC trên `V^t` và `y^bin`, position probe (R² `t/T`) trên `V^t`,
  macro theo share bin — đọc checkpoint, dùng ở P6.
- [x] Reference = **R2** (D11): `TRAINING_V2.md` + tests e2e chuyển sang R2 (tests tham số hoá thêm R2/R3).
- [ ] Đường chấm DoTA protocol B (s1[::3], native frames) cho arm v2 — cần trước P6 để chấm A1 trên DoTA.
- [ ] Bake A2/A3 trên Colab — cần full-T2 VideoMAE cache (P4, chặn bởi D9). A1 (R2) bake được ngay (notebook P6).
- [ ] User review → commit.

### P6 — Pilot ⬜

**Goal:** chứng minh pipeline đúng về cơ chế. **Không** quyết định adoption.

- [ ] Seed **2099**, A0/A1/A2/A3, T2-train trừ T2-val, hyper-params proposal §8. Notebook Colab.
- [ ] Đọc **chỉ**: T2-val (macro, micro, clip oracle), guardrails (micro < oracle, macro ≥ micro, micro ≥ A0 − 0.01),
  `ρ_u` theo thời gian, `‖W_u‖` tăng từ 0, loss curves, clip-level AUC vs macro (C14), position R², source-shortcut.
- [ ] **DoTA-dev không in trong pilot** (lesson 14).
- [x] **A0 regression (D5):** đọc 2026-10-04 → **FAIL** trên T2-val (micro −0.109, in-sample bias như O6 báo trước). Giải thích theo Amendment 7 (H1 code identity s2024 == phase-4 T2-test; H2 D5 trên T2-test) — `p6_d5_t2test.ipynb`.
- [ ] `ρ_u` ≈ 0 suốt run → ghi nhận motion không được dùng; **không** sửa lr/`c`.

### P7 — Go/no-go cho E3 ⬜

- [x] Checklist GO (2026-10-05, addendum §18 đầu mục): mọi điều kiện §5 đạt; caveat: A2/A3 qua guardrail **chỉ nhờ O1′**
  (post-hoc), và D6 cấm chữ "motion". Suite 896 / 0 fail; gate sạch (bandit B614 cũ trong `core/tests/`).
- [x] **Amendment 9 (§18, M1–M10) — BẢN NHÁP, chờ user ký** (notebook từ chối chạy khi còn câu "Not in force").
- [x] Harness: `core.tools.position_prior` (G6 a), `core.tools.e3_readout` (M4–M8), `protocol_b_eval` ghi
  `clip_scores.npz`; tests `test_e3_readout.py`, `test_position_prior.py` (+26).
- [x] Runbook E3 `colab/v2/e3_factorial.ipynb` (4 arm × seeds 2024–2028, resumable, chia nhiều session).
  **Cost:** pilot không ghi wall time → notebook đo từng run và in ETA; ước lượng thô = 20 × thời gian 1 run pilot.
- [ ] User ký Amendment 9 → commit → upload → chạy E3 → paste `v2_e3/REPORTS/e3_readout.md`.
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
| P3 E1 | **Adopt B.** Ckpt: s2024 phase 4 + s2025/s2026 train lại (J10 2040 ×3, J9′ pass). Macro A 0.6294 · **B 0.6628 (Δ +0.0334 [+0.019, +0.048])** · C 0.6644 (Δ +0.0350 [+0.020, +0.050]); tie → B. Seed Δ +0.035/+0.033/+0.042. Ruler 0.566. `core/docs/v2/RESULTS_E1.md` | harness `0ab760d`, Amendment 3 `7ce8bf5`; Colab (code upload) | 2026-09-30 |
| P2 D11 (s3) | **R2**, veto qua. Cổng s8 == s1@8 qua. `r` R1 0.7081 · R2 0.7092 · R3 0.6997 · R4 0.7049 (eligible ở s3). E2(c) R2 +0.0368 [+0.027, +0.047] (R1 +0.033). R1 ≈ R2 (hoà). `−f` 0.793. `outputs/v2/REPORTS/v2_E2_s3/`, `RESULTS_E2_CRN.md` §D11 | Colab (code upload) | 2026-09-30 |
| P3 E0b | | | |
| P4 E2(d) | **V2-S** (N7 tie: A3 transfer B 0.1508, S 0.1325). Mọi arm eligible. Position cubic `p` **0.852** in-domain / 0.847 transfer > `u` (0.76–0.82) > `[x;u]`; `t/N` cũ chỉ 0.566. Vượt position: in-domain ≈ 0 (−0.006…+0.006), transfer B +0.029/+0.037, S +0.018/+0.026. T2 side B +0.137, S +0.107. D15: A0 dev 0.6628 / CAP 0.6591 / dropped 0.6787 (`protocol_b_eval` == E1 B). `core/docs/v2/RESULTS_E2D.md` | §12 `7b1889a`; Colab (code upload) | 2026-10-04 |
| P6 batch 1 | A0 / A1 (s2099, step 1740): micro 0.6586 / 0.6683, macro 0.6737 / 0.6705, oracle 0.6986, window AUC 0.686 / 0.683, guardrails PASS/PASS. Shortcut `V^t` 0.9999 → **0.551** (CRN), `y^bin` 0.54 / 0.52. Pos R² `V^t` −0.20 / −0.03. **D5 FAIL**: ref mean 0.7675 / 0.6810 (O1 FAIL ×3, window AUC 0.96–0.97, in-sample) → Amendment 7. `core/docs/v2/RESULTS_P6_PILOT.md` | Colab (code upload) | 2026-10-04 |
| P6 D5 (H1–H3) | **EXPLAINED.** H1 s2024 v2 == phase 4 (Δ 9e-9 / 0). H2 T2-test A0 0.6320 / 0.6389 vs ref 0.6188 / 0.6253, Δ +0.013 / +0.014 ≤ 0.02. In-sample gap ref +0.14…+0.16 vs A0 +0.027. D5 không còn chặn P7 | `1a2a2c5` + Colab | 2026-10-04 |
| P6 batch 2 | A2 / A3 (V2-S, s2099, step 1740) T2-val: micro 0.7120 / 0.7161, macro 0.6882 / 0.7052 > oracle 0.6986 → **O1 FAIL** (giữ nguyên, Q1); window AUC +0.090 / +0.077, macro +0.015 / +0.032 vs A0 → không phải C14 → Amendment 8. `ρ_u` 0.13 / 0.11, `‖W_u‖` 1.27 / 1.10. Shortcut `V^t` A2 0.9998 → A3 0.583 | `573d3d9` + Colab | 2026-10-05 |
| P4 D6 shuffle | **Không phải thứ tự thời gian.** `[x;u;p]` ordered − shuffled A2 −0.0004 [−0.0074, +0.0065], A3 −0.0003 [−0.0066, +0.0062]; `u` +0.006/+0.007 (CI ∋ 0); shuffled `[x;u]` − `x` +0.142 / +0.119. 569/569 qua pixel gate. G3 → "second (video) appearance encoder". `outputs/v2/REPORTS/v2_E2d_shuffle/` | Colab (code upload) | 2026-10-05 |
| P6 Q4 T2-test | **A1/A2/A3 PASS O1′** (T2-test + T2-val). T2-test micro/macro/window: A0 0.6320/0.6389/0.664 · A1 0.6558/0.6350/0.712 · A2 0.7192/**0.7136**/0.765 · A3 0.7090/0.6897/0.780; oracle 0.7037. O1 in ra: A1–A3 FAIL. A0 tool check == Amendment 7. CRN in-domain âm (A3 − A2 −0.024, 1 seed). `outputs/v2/v2_pilot/o1prime_t2test/` | Colab (code upload) | 2026-10-05 |
| P7 | | | |
