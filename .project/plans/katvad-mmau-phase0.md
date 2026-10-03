# MM-AU / CAP-DATA — Phase 0: provenance probe (DoTA pixel recovery, DADA overlap, clean remainder)

**Branch:** `v2`. **Viết:** 2026-09-30, **trước khi có bất kỳ con số so khớp nào**. User cho phép 2026-09-30
(không qua thầy). **Liên quan:** D9 (`core/docs/v2/PREREG_ADDENDUM.md` §6) — endpoint của motion arm A2/A3.
**Runbook:** `colab/v2/mmau_p0.ipynb`. **Tool:** `python -m core.tools.mmau_match`.

---

## 0. Trạng thái

| Bước | Trạng thái | Kết quả |
|---|:-:|---|
| P0a Mirror CAP-DATA HF → Drive + census | ◐ | 9768/9768 clip đã cache. `census.json` chỉ có nhóm `1-10`, `43`, `44-62` (lệch frame 2/1/3); nhóm `11`, `12-42` thiếu census (file per-group nằm trên `/content`, mất khi runtime reset). DoTA-CAP kiểm L4 cho đúng các clip nó dùng |
| P0b CLIP `_ncc` s1 nhóm `1-10` → match (khả thi?) | ✅ | **Khả thi** (2026-10-01): exact 214/1397 DoTA (dev 104/702), r ≈ 1 hoặc ≈ 3 ở 211/214, null ≥ near 0.07 %; DADA 0. Dải near lọt 9 cặp giả → **Amendment P0b-1** |
| P0c CLIP 4 nhóm còn lại → match toàn bộ (coverage) | ✅ | (2026-10-02) exact **1248/1397** (0.893), dev 625/702 (0.890); exact∪near 0.935 / 0.930; null 0.006; DADA 3 near; CAP sạch 8,006 |
| P0d Đọc nhánh D9 (§4) → Amendment 4 | ✅ | Nhánh **C** theo luật. User override (2026-10-03): D9 = **DoTA-CAP** (exact, căn từng frame, cổng pixel), addendum §11 D13–D15 + L0–L7 |
| P1 DoTA-CAP: align → extract VideoMAE B+S → finalize | ⬜ | `core/tools/dota_cap.py`, `colab/v2/dota_cap_videomae.ipynb`. Sau đó: freeze `dota_cap_ids.txt` vào `core/splits/v2/` |

## 1. Vì sao

- DoTA không còn pixel (Amendment 1) → A2/A3 không có endpoint zero-shot (D9 treo).
- MM-AU (`JeffreyChou/MM-AU`, HF, public, CC-BY-NC-4.0) = **CAP-DATA 9,768 video (267 GB)** + DADA-2000 1,962 video
  (124 GB, = corpus T2 của mình → **không tải**). Bài CAP (arXiv 2212.09381) ghi CAP-DATA gom **CCD, A3D, DoTA,
  DADA-2000** + web. `cap_text_annotations.xls` (sheet `annotation file`) **không có cột nguồn gốc** → provenance
  chỉ biết được bằng so khớp nội dung.
- Đo trước từ annotation (2026-09-30, scratchpad, chỉ mô tả): 9,768 video, 100 % accident, median 135 frame (tổng
  1,546,250), share abnormal median **0.494** (DoTA 0.306), 91.6 % bắt đầu normal, 16.7 % kết thúc trong accident,
  57 type.

Ba câu hỏi, không câu nào cần train:

1. **DoTA có nằm trong CAP không, và pixel có giống hệt không?** Có → khôi phục pixel DoTA → VideoMAE trên DoTA →
   D9 = DoTA như proposal gốc.
2. **CAP có chứa video nguồn của T2 (DADA-2000) không?** Có → clip đó bị loại khỏi mọi benchmark MM-AU.
3. **Phần sạch (không DoTA, không DADA) lớn cỡ nào?** → benchmark zero-shot phụ có motion.

## 2. Assumptions & constraints

| # | Nội dung |
|---|---|
| A1 | Layout sau giải nén: `.../{group}/{type}/{video:06d}/images/{frame:06d}.jpg` (README HF). Id = tên folder video (unique trong annotation, 1…14490). Tool không phụ thuộc độ sâu: dùng `list_frame_folders(root, "images")` |
| A2 | CLIP cache mới: `Thesis/cache/clip/MMAU_CAP_s1_ncc/` — **cùng transform `no_center_crop`, cùng backbone, stride 1** như `DoTA_s1_ncc` (C2). Stride nào khác đều dựng lại được từ s1 |
| A3 | `DoTA_s1_ncc` (1,397 clip) và `DADA2000_orig` (1,962 source, stride 8) là query; CAP là kho tham chiếu |
| C1 | DoTA-eval vẫn **sealed**: tool chỉ đọc DoTA-dev qua `load_split`; coverage in trên **toàn bộ DoTA** và **DoTA-dev**. Không đọc nhãn hay score nào — P0 chỉ đo sự tồn tại của pixel |
| C2 | Frame giải nén lên disk VM (`/content`), **không lên Drive** (FUSE + hàng triệu jpg, lesson C10); chỉ CLIP cache + raw part lên Drive |
| C3 | Raw part CAP được **mirror lên Drive** (quota 5 TB): link chết là mất pixel vĩnh viễn (tiền lệ DoTA) |

## 3. Luật so khớp (cố định trước khi đọc số)

Mọi embedding frame được chuẩn hoá L2. Với query `q` (một clip DoTA hoặc một source DADA) và clip CAP `c`:

- **Retrieval:** descriptor clip = mean các frame đã chuẩn hoá, chuẩn hoá lại. Lấy `K = 5` clip CAP có cosine
  descriptor cao nhất.
- **Containment** `κ(q, c)` = trung bình theo frame của `q` của `max_j cos(q_i, c_j)` — tỉ lệ nội dung `q` tìm thấy
  trong `c`, không phụ thuộc thứ tự, fps hay việc `c` dài hơn `q`.
- **Best match** = candidate có `κ` cao nhất. **Null cứng** = `κ` với candidate hạng `K` (giống về bối cảnh nhưng
  xếp hạng thấp nhất).
- **exact** nếu `κ ≥ 0.99` (cùng frame, chỉ khác mã hoá); **near** nếu `0.95 ≤ κ < 0.99` (cùng video, xử lý khác:
  resize, crop, fps); còn lại **none**.
- **Căn thời gian** (cặp exact/near): fit `j = r·i + o` (least squares) trên các frame có max-cos ≥ 0.95 → `r` =
  frame CAP / frame query (tỉ lệ fps), `o` = offset. In phân bố `r`.
- **Cờ độ tin cậy:** nếu > 1 % null ≥ 0.95 thì ngưỡng không tách được → **không đọc nhánh**, báo user.
- **Amendment P0b-1** (2026-10-01, user duyệt, không qua thầy; viết **sau** read-out P0b chỉ-khả-thi của nhóm
  `1-10`, **trước** P0c): cặp **near** còn phải căn thời gian được với `r ≥ 0.9` (`MMAU_NEAR_MIN_RATE`); không
  căn được hoặc `r < 0.9` → **none** (đếm ở cột `near rej.`), và không tính là hit khi loại clip CAP. Lý do vật lý:
  cùng một video thì CAP không thể có ít frame hơn query. Bằng chứng P0b: 9 cặp near κ 0.951–0.968 có `r` ∈
  [−0.31, 0.71], κ − null từ 0.000 đến 0.076, và clip CAP `002432` "khớp" 3 video YouTube khác nhau (cảnh dashcam
  giống nhau); 22 cặp near thật κ 0.980–0.990 đều có `r` ≈ 1 hoặc ≈ 3. **exact và cờ null không đổi.** Rủi ro chấp
  nhận: luật được đặt sau khi nhìn 16 % CAP; nó chỉ ảnh hưởng P′ và danh sách loại, không ảnh hưởng nhánh P.

## 4. Luật rẽ nhánh D9 (cố định trước khi đọc số)

| Nhánh | Điều kiện (sau P0c, toàn bộ CAP) | Hệ quả |
|---|---|---|
| **P — pixel recovery** | `exact` phủ ≥ 95 % của toàn bộ DoTA **và** ≥ 95 % của DoTA-dev | D9 = DoTA như proposal gốc: A2/A3 chấm trên DoTA-dev/eval bằng frame CAP đã căn. Trước khi dùng: cổng tái lập (CLIP của frame CAP đã căn == `DoTA_s1_ncc`, cos ≥ 0.99 trên mọi clip được dùng) |
| **P′ — near only** | `exact ∪ near` ≥ 95 % nhưng `exact` < 95 % | Pixel khác xử lý → VideoMAE nhìn input khác DoTA gốc. User quyết; ghi rõ là "DoTA re-encoded by CAP" |
| **C — clean benchmark** | không đạt P/P′ | D9 = T2-val (quyết định) + **MMAU-clean** (zero-shot phụ): clip CAP không near/exact với DoTA **hoặc** DADA |
| (mọi nhánh) | clip CAP khớp near/exact với một source DADA | **Loại** khỏi mọi benchmark MM-AU (nhiễm T2) |

Coverage một phần (< 95 %) **không** được dùng làm tập DoTA con cho motion arm: tập con được chọn theo việc CAP có
hay không, không ngẫu nhiên → không so được với A0/A1 trên toàn DoTA.

P0b (chỉ nhóm `1-10`, 1,556 video) chỉ trả lời "khả thi không" (phân bố κ, cờ null, có cặp exact không); **không**
quyết nhánh.

## 5. Các bước

### P0a — Mirror + census (CPU, Colab)
- `huggingface_hub.snapshot_download` (`allow_patterns`: `CAP-DATA_chunks/*`, `cap_text_annotations.xls`,
  `README.md`) → `Thesis/data/MMAU/raw/`. Kiểm **kích thước từng part** == HF API (`list_repo_tree`), ghi manifest.
- Census (sau giải nén): số folder mỗi nhóm == README (1556 / 3083 / 1629 / 2150 / 1350); số jpg mỗi video ==
  `total frames` trong annotation → in số lệch, không dừng.

### P0b — Nhóm `1-10` (L4) — ✅ 2026-10-01
- Đã chạy bằng `cat part_* | tar -xz` cả nhóm xuống `/content/mmau` rồi `extract_clip_features --frames-dir …
  --stride 1 --no-center-crop` → xoá frame. Cách này cần disk ≈ kích thước nhóm (nhóm `11` ≈ 96 GB → phải A100).
- **Từ P0c:** `python -m core.tools.stream_frames_clip --parts {g}.part_* …` đọc các part thành một luồng gzip,
  ghi frame từng video xuống `/content`, đủ `MMAU_STREAM_BATCH_GB` (8 GB) video xong thì encode bằng **đúng**
  `extract_frame_directory` (cùng transform, C2) rồi xoá. Disk đỉnh ≈ 1 lô + 1 video → T4/L4. Test chứng minh
  feature qua streaming == feature giải nén thường (bằng nhau từng bit).
- `python -m core.tools.mmau_match` → read-out `Thesis-V2/outputs/REPORTS/mmau_p0/group_1-10/`.

### P0c — 4 nhóm còn lại
- Từng nhóm bằng `stream_frames_clip` (preflight disk ≥ 3 × `BATCH_GB`); nhóm `1-10` đã cache thì bỏ qua.
- `mmau_match` trên toàn bộ CAP → read-out `mmau_p0/all/` → §4.

### P0d — Record
- `core/docs/v2/MMAU_P0.md` + Amendment 4 (D9) trong `PREREG_ADDENDUM.md`, theo đúng nhánh §4.

## 6. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| CAP resample/crop DoTA → κ rơi vào 0.95–0.99 | Nhánh P′ tách riêng; không gộp vào P |
| Nhiều video dashcam cùng bối cảnh → null cao | Cờ null > 1 % → không đọc nhánh |
| Một clip CAP chứa nhiều clip DoTA (DoTA = đoạn cắt từ video YouTube) | Containment theo hướng query ⊂ CAP; mỗi DoTA clip gán cho clip CAP tốt nhất, không ép 1-1 |
| Disk VM không đủ cho nhóm `11` | Streaming theo lô (≤ 8 GB + 1 video); tar không liền mạch theo video → tool raise |
| JPEG decode chậm (1.55M frame) | Resume theo video (`extract_clip_features` đã resumable); chạy từng nhóm |
