# KAT-VAD v2 — Exploration plan: freeze → kill-switch → E0–E2 → build → pilot → go/no-go for E3

**Status:** APPROVED 2026-09-27; P0 DONE (splits frozen). Next: P1.
**Branch:** `v2` (tách từ `main@cc39882`; code = v1 + docs v2). Mọi code của plan này nằm trên `v2`.
**Nguồn:** `core/docs/v2/KAT_VAD_PROPOSAL_v2.md` (§4, §7.2, §10) + `KAT_VAD_v2_ARCHITECTURE.md`.
Proposal thắng plan ở phần *what* (thành phần, rule adoption). Plan này đổi **thứ tự** và thêm
**kiểm tra kỹ thuật**. Mọi chỗ lệch khỏi §10.2 đều có lý do trong §3 và được commit vào addendum ở P0,
**trước** khi có bất kỳ con số nào.

---

## 1. Summary

- Mục tiêu: trả lời trong 2–3 tuần câu hỏi *"v2 có đáng chạy factorial 20 run (E3) không, và chạy với cấu hình nào?"*
  (reference CRN, encoder, protocol DoTA) — **không train để quyết định**, chỉ probe + re-score + 1 pilot kỹ thuật.
- Rủi ro lớn nhất của v2 là **motion stream không mang signal** (frozen VideoMAE V2 có thể chỉ là appearance).
  Nếu vậy, v2 co lại còn CRN + protocol. Vì thế kiểm tra nó **đầu tiên**, trên subset, bằng một kill-switch probe.
- Scope: P0 → P7 chi tiết. E3 chỉ là phase khung, được gate bởi P7.
- Không có loss mới, không tune gì theo Δ trên DoTA-dev (lesson 14).

## 2. Assumptions & Constraints

| # | Assumption / constraint | Hệ quả nếu sai |
|---|---|---|
| A1 | DADA-2000 original = **30 fps** (chỉ có trong literature; release là PNG, không đo được fps từ file) | clip 10 fps "mỗi frame thứ 3" sai tỉ lệ thời gian; ghi rõ là assumption trong mọi read-out |
| A2 | Frames DoTA + DADA original, CLIP caches `*_ncc`, checkpoints phase-4/5/lcurve đều nằm trên Drive | local chỉ có `results.json` (không có per-clip score) → E0b/E1 **bắt buộc chạy trên Colab** |
| A3 | Compute = Colab A100 40 GB; dev = macOS CPU, data-free tests | extraction VideoMAE bị I/O-bound từ Drive → copy frames sang SSD local của runtime trước |
| A4 | DoTA: 1,397 clip / **179 video gốc** (max 19 clip/video), 3 clip toàn normal | split và bootstrap phải **group theo video gốc**, không theo clip |
| C1 | Không đụng `LaGoVAD-PreVAD/`; code chỉ trong `core/` | — |
| C2 | Cache gắn với transform + stride (C2/C13): stride 3 = **cache mới**, không ghi đè `*_s8` | — |
| C3 | DoTA-eval và T2-test chỉ mở một lần (final, sau E3) | — |
| C4 | Timeline 2–3 tuần | P5 (build) chạy song song P3/P4 |

## 3. Deviations from the advisor's run order (§10.2) — and why

| # | Advisor | Plan này | Lý do |
|---|---|---|---|
| D1 | E0 → E0b+E1 → E2; VideoMAE extraction song song | **P1 kill-switch VideoMAE trên subset ngay sau freeze**, trước E1 | Rủi ro lớn nhất cần biết sớm nhất (~ngày 4). E1/E0b rẻ nhưng không đổi được go/no-go của cả v2. |
| D2 | E2(b) chọn CRN reference sau E1 | E2(b) chạy **ở cả stride 8 và stride 3**, lấy kết quả theo stride mà E1 chọn | E1 đổi step rate của DoTA → `t/T`, deviation, share bin đều đổi. Chọn reference ở stride 8 rồi eval ở stride 3 là chọn trên phân phối khác. Numpy, rẻ. |
| D3 | Clip-level paired bootstrap trên DoTA-dev | Bootstrap **theo video gốc** (cluster bootstrap) | 179 video, tới 19 clip/video tương quan → bootstrap theo clip thu hẹp CI giả tạo. Áp dụng cho E1 (quyết định) và mọi bootstrap báo cáo. |
| D4 | Không có | **P6 pilot** 1 seed × 4 arm (seed **2099**, ngoài 2024–2028), chỉ đọc cơ chế + T2-val, **không in DoTA-dev** | Bắt bug trước khi đốt 20 run. Seed ngoài bộ pre-registered nên không làm bẩn E3. |
| D5 | Không có | **A0 regression check**: A0 trên code v2 phải khớp KIP-off phase-4 | Code v2 sửa forward/dataset; A0 phải chứng minh vẫn là baseline, không thì mọi Δ vô nghĩa (C5 tinh thần). |
| D6 | E2(d) probe các encoder trên cùng feature | Thêm **temporal-shuffle control** (xáo 16 frame trong clip) cho encoder được chọn — chỉ diagnostic | Phân biệt "VideoMAE thêm motion" vs "VideoMAE chỉ là CLIP thứ hai". Không quyết định adoption, nhưng quyết định **cách viết** claim "kinematics-aware". |

Mọi rule quyết định khác (eligibility E2(d), rule chọn reference, adoption E3, `F`) **giữ nguyên proposal**.

## 4. Plan Metadata

- **Plan type:** Research campaign — no-training probes + model build + engineering pilot
- **Size / scope:** Large (≈ 6 module mới trong `core/`, 3 extraction, 1 pilot 4 run)
- **Estimated duration:** 13–16 ngày làm việc (2–3 tuần)
- **Storage path:** `@.project/plans/katvad-v2-e0-e2.md`

## 5. Phases Overview

| Phase | Goal | Ngày | Phụ thuộc | Train? |
|---|---|---|---|:-:|
| P0 Freeze | Split + rules + addendum commit; DoTA-eval guard | 1 | — | no |
| P1 Kill-switch K | VideoMAE-B có signal không, trên subset | 2–4 | P0 | no |
| P2 Local audits | E0 rate audit, E2(a) histograms, E2(b) ruler + CRN reference (2 stride), E2(c) veto | 2–5 (song song) | P0; E2(b) stride 3 cần cache P3 | no |
| P3 Protocol | E1 (A/B/C) + E0b trên một harness, DoTA-dev only | 3–7 | P0 | no |
| P4 Encoder choice | E2(d) full: B vs S, probe representation từng arm, shuffle control | 7–10 | P1 GO, P3 (stride) | no |
| P5 Build v2 | Config, input stats, dataset, `MotionResidual`, CRN, logging, tests | 5–11 (song song) | P2 interface CRN | — |
| P6 Pilot | 1 seed × 4 arm, cơ chế + guardrails + A0 regression | 11–14 | P4, P5 | yes (4 run) |
| P7 Go/no-go | Checklist → E3 runbook hoặc báo cáo dừng | 14–16 | P6 | no |
| (E3) | 2 × 2 × 5 seeds = 20 run | sau plan | P7 GO | yes |

## 6. Detailed Tasks by Phase

### P0 — Freeze (ngày 1)

**Goal:** mọi thứ quyết định sau này đọc trên split cố định và rule đã commit.

- [ ] `core/tools/freeze_splits.py` (argparse): tạo
  - `T2-val` = 15 % **source video** của T2-train, grouped, seed cố định trong `constants.py`;
  - `DoTA-dev / DoTA-eval` = 50/50, **grouped theo video gốc** (`clip_id.rsplit('_', 1)[0]`),
    stratified theo accident-share bin để hai nửa cân nhau; 3 clip toàn normal ghi rõ nằm bên nào.
  - Output `core/splits/v2/{t2_val_sources,dota_dev,dota_eval}.txt` + `SPLITS_MANIFEST.json` (sha1 từng list, seed, commit).
- [ ] DoTA-eval guard: loader nhận `--split dev|eval|full`; `eval`/`full` raise trừ khi có `--final`. Test: mặc định không thể in số eval.
- [ ] Addendum pre-registration `core/docs/v2/PREREG_ADDENDUM.md`: D1–D6 + kill-switch rule (P1) + pilot read-out (P6) + go/no-go (P7).
- [ ] Housekeeping: thêm hàng `v2` vào bảng CLAUDE.md §14.0.1 và banner branch trong `activeContext.md`.
- [ ] Commit (≤ 20 file, add từng file). Ghi hash vào mọi read-out sau.

**Deliverables:** split files + manifest, guard + test, addendum, commit hash.

### P1 — Kill-switch K: VideoMAE có signal không (ngày 2–4, Colab)

**Goal:** biết sớm motion stream có đáng làm tiếp không, trước khi đầu tư extraction đầy đủ và code model.

- [ ] `mcp context7` xác minh API load VideoMAE V2 distilled (`vit_b_k710_dl_from_giant`, `vit_s_k710_dl_from_giant`):
  có nằm trong `transformers 4.56` không, có cần `trust_remote_code` không. Nếu cần remote code → vendor
  định nghĩa model vào `core/` và pin revision/sha của weight (C4; bandit).
- [ ] `core/tools/extract_video_features.py` (argparse): causal 16 f @ 10 fps kết thúc tại step, squash 224², VideoMAE mean/std,
  mean-pool token → `u_t`. Đầu clip: pad bằng lặp frame đầu (ghi rõ). Cache `cache/video/<encoder>/<dataset>_s<stride>_squash/`,
  kèm manifest (encoder sha, stride, fps assumption, transform). Log throughput (steps/s) để ước tính P4.
- [ ] Tests (CPU, model giả): index causal (không lookahead), mapping stride/fps DADA (×3) vs DoTA (×1), shape, pad đầu clip.
- [ ] Subset: DoTA-dev ~200 clip (stratified theo share bin, grouped) + T2 ~300 source video (train phần, không đụng T2-val);
  **ViT-B** (ứng viên mạnh nhất: nếu B không có signal thì S cũng không), stride 8 để ghép với cache CLIP hiện có.
- [ ] Probe (tái dùng `core/eda/features.py`: `frame_linear_probe`, không viết lại):
  (i) in-domain DoTA-dev grouped CV: CLIP-only vs `u`-only vs `[x ; u]`; (ii) transfer T2-subset → DoTA-dev-subset.
  Cluster bootstrap theo video cho CI.
- [ ] **Rule (commit trong P0):**
  - **KILL motion stream** nếu cả hai Δ(`[x;u]` − CLIP) có point estimate ≤ 0 **và** cận trên CI < +0.03.
  - **Positive control:** `u`-only in-domain phải > 0.5 rõ ràng (CI loại 0.5). Nếu không → nghi bug pipeline, sửa rồi chạy lại, **không** KILL.
  - Còn lại → GO sang P4. K **không** quyết định eligibility (subset quá nhỏ); E2(d) ở P4 mới quyết định.

**Deliverables:** extractor + tests, subset caches, `outputs/REPORTS/v2_K/k_readout.md` (GO/KILL + throughput).
**Nếu KILL:** P4 bỏ, P5 bỏ motion stream (chỉ CRN), P6 còn 2 arm (A0, A1). Báo thầy trước khi đi tiếp.

### P2 — Local audits: E0, E2(a), E2(b), E2(c) (ngày 2–5, song song P1)

**Goal:** chốt reference CRN (hoặc bỏ CRN) chỉ bằng numpy trên cache CLIP.

- [ ] Kéo cache CLIP T2 + DoTA (s8, sau đó s3 từ P3) từ Drive về local (nhỏ, vài chục MB).
- [ ] **E0** `core/tools/rate_audit.py`: fps, stride, s/step, độ dài clip (giây) cho T2 và DoTA → xác nhận/bác gap ≈ 3×. Nếu không ≈ 3× → bỏ E1 (proposal).
- [ ] `core/crn/reference.py` — **một** implementation R1–R4 (R4 strictly-past, warm-up `N_w`=8), dùng chung cho probe và cho model ở P5 (DRY). Tests: R4 không nhìn tương lai; t < N_w dùng mean N_w bước đầu; R2/R3 đúng định nghĩa.
- [ ] **E2(a)** histogram accident-share trên DoTA-dev và DADA source video; tỉ lệ clip bắt đầu normal; coverage của normal steps ở fifth cuối `t/T` (quyết định có kích hoạt coverage fallback không).
- [ ] **E2(b)** `core/tools/crn_select.py`: position ruler theo share bin → `d_t`, `f` cubic fit trên normal steps của T2-val (clamp 5–95 pct) → `r_t` macro per bin + stratified AUC (min-max per clip trước khi pool) → rule chọn §4.2 bước 4–5. **Chạy ở stride 8 và stride 3** (D2).
- [ ] **E2(c)** transfer-probe veto (T2-train → DoTA-dev), reference đã chọn vs raw.

**Deliverables:** `outputs/REPORTS/v2_E2abc/` + read-out (reference được chọn cho mỗi stride, hoặc "CRN dropped").

### P3 — Protocol: E1 + E0b (ngày 3–7, Colab)

**Goal:** chốt stride/protocol DoTA; đo `MDE_dev`.

- [ ] Cache CLIP DoTA stride 3: `extract_clip_features --stride 3` → `cache/clip/DoTA_s3_ncc` (mới, C2); labels `dota.py --stride 3`.
- [ ] Evaluator: thêm chế độ sliding **W=20, hop 4, trung bình phần chồng** + **nội suy score về native frame** (hiện `sliding_window_scores` cắt theo `max_vis_len`, cần `trace_call_path` trước khi sửa). Arm A cũng đi qua cùng nội suy, để so cùng evaluator.
- [ ] Tests: nội suy về native frame giữ đúng độ dài; hop-average với W chia/không chia hết; stride 8 whole-clip qua harness mới khớp `results.json` cũ (regression).
- [ ] **E1:** 3 checkpoint KIP-off phase-4 × {A: s8 whole, B: s3 whole, C: s3 sliding W20} trên DoTA-dev. Quyết định bằng **cluster bootstrap theo video** (D3), rule proposal.
- [ ] **E0b:** re-score (s8, protocol cũ) phase-4 KIP-on v1, phase-5 KIP-on v2, lcurve 25 % / 50 % trên DoTA-dev → pooled SD, `MDE_dev` (≤ 8 df, descriptive). Kiểm tra trước là ckpt lcurve có trên Drive.
- [ ] In position ruler cạnh mọi macro.

**Deliverables:** `outputs/REPORTS/v2_E1_E0b/` + read-out (protocol chốt, `MDE_dev`).

### P4 — Encoder choice: E2(d) đầy đủ (ngày 7–10, Colab; chỉ khi P1 = GO)

- [ ] Extract VideoMAE-B và -S tại **stride chốt ở P3**: T2-train subsample, T2-val, DoTA-dev. (DoTA-eval: extract được nhưng không score.)
- [ ] Probe đúng representation mỗi arm đưa vào trunk: A2 `c·(u − m_u)⊘σ_u`; A3 `c·ũ⊘σ_u` với reference từ P2. Rule eligibility proposal: in-domain DoTA-dev ≥ +0.10 **hoặc** transfer ≥ +0.03 so với CLIP-only; chọn transfer tốt nhất, trong 0.02 thì encoder rẻ hơn thắng.
- [ ] Lặp E2(b) trên `[x ; u]` (proposal §4.2 bước 3) với encoder đã chọn.
- [ ] **D6 shuffle control:** encoder đã chọn, xáo 16 frame trong clip → probe lại. Chỉ report.
- [ ] Extract full T2 (train + val + test) cho encoder đã chọn.

**Deliverables:** `outputs/REPORTS/v2_E2d/` + read-out: encoder, eligibility, shuffle Δ; full T2 cache.

### P5 — Build v2 model (ngày 5–11, song song; local CPU)

Trước khi sửa bất kỳ symbol nào: `trace_call_path` + báo blast radius; load `meta-index.md`; context7 cho API ngoài.

- [ ] Config `v2` section trong `core/config.py`: `crn.enabled`, `crn.reference ∈ {mean, median, robust, past}`, `crn.warmup`,
  `motion.enabled`, `motion.feature_dir`, `motion.encoder`, `motion.stats_path`. Với v2 **ép `kip.enabled=false`** (raise nếu cả hai bật).
- [ ] `core/tools/build_v2_stats.py`: từ T2-train **trừ T2-val** → `s`, `m_u`, `σ_u`, `c`; manifest có `train_ids_sha1` (cùng mẫu với `zscore_manifest.json`). Stats bị bind vào split: mismatch → raise.
- [ ] Data: reference CRN tính trên **source video** (train) / **clip** (test) ở data layer (dùng `core/crn/reference.py`), không trong model. Load `u` song song `x`, assert độ dài khớp từng video (fail loud).
- [ ] Model: `MotionResidual` (Linear `d_v→512`, weight + bias zero-init) cộng vào input trước temporal encoder; không LayerNorm. Log `ρ_u` và `‖W_u‖` mỗi 50 step.
- [ ] Checkpoint: key mới qua `ckpt_compat` (C5) — A0 load ckpt KIP-off cũ không lỗi; A2/A3 thiếu `W_u` → raise.
- [ ] Diagnostics cho mọi arm (proposal §10.1): source-shortcut AUC trên `V^t` và `y^bin`, position probe (R² `t/T`) trên `V^t`, macro theo share bin.
- [ ] Tests:
  - `TestV2A0IsKipOff`: config A0 cho forward giống hệt KIP-off, cùng seed.
  - `W_u = 0` ⇒ output A2 ≡ A0 lúc init.
  - `s` giữ `E‖x̃‖ = E‖x‖` trên dữ liệu giả.
  - R4 streaming: score tại t không đổi khi thêm step sau t.
  - stats `train_ids_sha1` mismatch → raise.
- [ ] Quality gate §11 + full suite xanh (hiện 598). Docs: `core/docs/v2/TRAINING_V2.md` (flags, cache, stats).
- [ ] Lesson candidate sau khi build (§7 CLAUDE.md).

### P6 — Pilot (ngày 11–14, Colab; 4 run)

**Goal:** chứng minh pipeline đúng về cơ chế. **Không** quyết định adoption.

- [ ] Seed **2099**, A0/A1/A2/A3, T2-train trừ T2-val, hyper-params proposal §8.
- [ ] Đọc **chỉ**: T2-val (macro, micro, clip oracle), guardrails (micro < oracle, macro ≥ micro, micro ≥ A0 − 0.01),
  `ρ_u` theo thời gian, `‖W_u‖` tăng từ 0, loss curves, clip-level AUC vs macro (dấu hiệu collapse C14), position R², source-shortcut.
- [ ] **DoTA-dev không in trong pilot** (lesson 14: thấy Δ pilot là muốn tune).
- [ ] **A0 regression (D5):** so A0-pilot với ckpt KIP-off phase-4 re-score trên **T2-val** (T2-test vẫn đóng). Kỳ vọng ≈ −0.002 (mất 15 % data train). Chênh > ~0.02 → dừng, debug.
- [ ] Nếu `ρ_u` ≈ 0 suốt run → motion stream không được dùng; ghi nhận, không sửa lr/`c` (sẽ là tune).

**Deliverables:** `outputs/REPORTS/v2_pilot/` + read-out cơ chế.

### P7 — Go/no-go cho E3 (ngày 14–16)

- [ ] Checklist GO: splits committed; protocol chốt; reference chốt (hoặc CRN dropped); encoder eligible (hoặc motion dropped); A0 regression pass; guardrails pass cho mọi arm pilot; không collapse; suite xanh; quality gate xanh.
- [ ] Viết runbook E3 `colab/v2/e3_factorial.ipynb` (5 seeds 2024–2028 × arm còn sống), cost estimate từ throughput pilot.
- [ ] Cập nhật memory bank (`activeContext`, `progress`, `systemPatterns`), báo thầy bảng quyết định P1–P6.
- [ ] Nếu NO-GO: report bounded null / thành phần bị drop, theo proposal §10.3.

## 7. Risks & Mitigations

| Risk | Tác động | Giảm thiểu |
|---|---|---|
| VideoMAE V2 distilled không load được trong `transformers 4.56`, cần remote code | P1 trễ 1–2 ngày | context7 ngày 2; vendor model def + pin weight sha; fallback: `VideoMAEModel` HF gốc chỉ khi thầy đồng ý (khác ứng viên đã pre-register) |
| Kill-switch false negative trên subset | Giết oan motion stream | Rule KILL chỉ khi point ≤ 0 **và** CI upper < +0.03; positive control chặn trường hợp bug |
| I/O Drive → Colab chậm | Extraction mất nhiều ngày | Copy frames sang SSD runtime; extractor resumable (skip file có sẵn); đo throughput ở P1 |
| E1 đổi stride → CRN chọn trên phân phối sai | Reference sai | D2: chọn ở cả hai stride |
| Clip DoTA tương quan trong video | CI hẹp giả, E1 adopt oan | D3: cluster bootstrap |
| Dùng DoTA-dev nhiều lần (E1, E2, E3) | Selection optimism | DoTA-eval chỉ mở ở final; đo optimism bằng dev−eval |
| Sửa forward/dataset làm lệch baseline | Mọi Δ vô nghĩa | `TestV2A0IsKipOff` + A0 regression ở pilot |
| CRN reference trên source video chứa accident dài (T2 cắt từ trong accident video) | Đảo thứ hạng | E2(a) histogram + rule reversal; T2-val có bin > 50 % |
| A1 (30 fps) sai | Clip motion sai tỉ lệ thời gian | Ghi assumption; nếu có metadata fps từ xlsx thì đối chiếu ở E0 |
| 2–3 tuần không đủ | E3 trễ | P5 song song; nếu P1 KILL, P4 biến mất và plan ngắn lại ~4 ngày |
| Stats `σ_u/c/s` rò T2-val | Leak nhẹ vào quyết định in-domain | Build từ T2-train trừ val, bind sha1 |

## 8. Next Steps for the User

1. Duyệt D1–D6 (nhất là D3 cluster bootstrap và D4 seed pilot 2099). Nên cho thầy xem D1–D3 vì chúng sửa §10.2.
2. Xác nhận trên Drive có: frames DoTA + DADA original, checkpoint phase-4 (KIP-off/on), phase-5, lcurve 25/50 %.
3. Chốt có cho phép vendor code VideoMAE V2 (nếu không có trong `transformers 4.56`) hay không.
4. Cho em bắt đầu P0 (freeze split + addendum + housekeeping CLAUDE.md branch `v2`).

---

## Appendix A — Read-outs (điền sau khi chạy)

| Phase | Kết quả | Commit | Ngày |
|---|---|---|---|
| P0 | T2-val 219 src / 645 win; DoTA-dev 702 clip / 93 vid, eval 700 / 86; >70 bin chỉ 19 clip dev; addendum D1–D6 | (commit P0) | 2026-09-27 |
| P1 K | | | |
| P2 E0/E2(a–c) | | | |
| P3 E1/E0b | | | |
| P4 E2(d) | | | |
| P6 pilot | | | |
| P7 | | | |
