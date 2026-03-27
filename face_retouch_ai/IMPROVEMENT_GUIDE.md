# Hướng dẫn cải thiện Pipeline — Tiệm cận chất lượng Evoto.ai

Tài liệu này mô tả chi tiết cách cải thiện từng phần của pipeline retouch, dựa trên các gap đã được liệt kê. Mỗi phần bao gồm: vấn đề hiện tại, mục tiêu, các bước thực hiện cụ thể, và tham chiếu tới file trong project.

---

## Mục lục

1. [Blemish Segmentation — Phát hiện khuyết điểm](#1-blemish-segmentation--phát-hiện-khuyết-điểm)
2. [Inpainting và Harmonization — Xóa và cân màu da](#2-inpainting-và-harmonization--xóa-và-cân-màu-da)
3. [Tuning tham số theo loại khuyết điểm](#3-tuning-tham-số-theo-loại-khuyết-điểm)
4. [Sản phẩm và UX](#4-sản-phẩm-và-ux)
5. [Đánh giá có hệ thống](#5-đánh-giá-có-hệ-thống)

---

## 1. Blemish Segmentation — Phát hiện khuyết điểm

### Vấn đề hiện tại

- **Heuristic fallback** (`_heuristic_blemish` trong `pipelines/blemish_seg.py`) dựa trên LAB redness, Laplacian texture, và blob detection — không ổn định trên da tối màu, ánh sáng studio, hoặc makeup.
- **U-Net / SegFormer** chỉ chạy khi có checkpoint tại `models/blemish_seg/unet_blemish.pth` hoặc `segformer_acne/`. Nếu không có → luôn dùng heuristic.
- Thiếu phân biệt rõ ràng **mụn vs tàn nhang vs nốt ruồi** — Evoto cho phép giữ nốt ruồi, xóa mụn.

### Mục tiêu

- Có model segmentation AI chạy ổn định trên mọi loại da.
- Giảm false positive (tàn nhang, nốt ruồi, lỗ chân lông).
- Có thể tùy chọn "giữ nốt ruồi" trong UI.

### Các bước thực hiện

#### Bước 1.1: Chuẩn bị dataset chất lượng

- Tham khảo: `DATASET_PREP_GUIDE.md`, `KAGGLE_TRAINING_NOTE.md`.
- Cấu trúc cần có:
  - `datasets/acne_mega/images/` — ảnh chân dung.
  - `datasets/acne_mega/masks/` — mask nhị phân (0 = background, 255 = vùng cần xóa).
- Quy tắc mask:
  - Chỉ đánh dấu mụn, sẹo, vết đỏ — **không** đánh dấu nốt ruồi nếu muốn giữ.
  - Không bao gồm mắt, lông mày, môi, tóc, lỗ mũi.
  - Ưu tiên chất lượng hơn số lượng; tối thiểu 2.000 cặp, khuyến nghị 5.000+.
- Chạy validation trước khi train:
  ```powershell
  python scripts/validate_dataset.py --all
  ```

#### Bước 1.2: Train SegFormer (ưu tiên) hoặc U-Net

- **SegFormer** (khuyến nghị): `kaggle_train_segformer.py`, hướng dẫn chi tiết `KAGGLE_SEGFORMER_GUIDE.md`.
- **U-Net** (nhẹ hơn): dùng script `train_blemish_unet.py` trong `scripts/` nếu có.
- Checkpoint sau khi train:
  - SegFormer: copy thư mục `segformer_acne/` vào `models/blemish_seg/`.
  - U-Net: copy `unet_blemish.pth` hoặc `unet_blemish_best.pth` vào `models/blemish_seg/`.

#### Bước 1.3: Thêm logic "giữ nốt ruồi" (optional)

- Tạo dataset phụ với mask nốt ruồi (class riêng) hoặc dùng model phân lớp nốt ruồi vs mụn.
- Trong `blemish_seg.py`, sau khi có `blemish_mask`, trừ đi mask nốt ruồi:
  ```python
  # Pseudocode: blemish_mask = blemish_mask & (~mole_mask)
  ```
- Có thể thêm UI checkbox "Giữ nốt ruồi" trong `app.py` và truyền xuống pipeline.

#### Bước 1.4: Kiểm tra tham số inference

- File: `pipelines/blemish_seg.py` — hàm `detect_blemish_ai`.
- Tham số cần điều chỉnh khi đánh giá:
  - `threshold` (mặc định 0.5) — tăng nếu quá nhiều false positive.
  - `min_area`, `max_area` — lọc vùng quá nhỏ/lớn.
  - `dilate_px` — mở rộng nhẹ mask sau inference.

---

## 2. Inpainting và Harmonization — Xóa và cân màu da

### Vấn đề hiện tại

- **LaMa** (qua `simple-lama-inpainting`) xử lý tốt vùng nhỏ nhưng dễ lệch tone hoặc mất cấu trúc da với mask lớn hoặc ánh sáng phức tạp.
- **Skin Tone Harmonizer** (`pipelines/skin_tone_harmonizer.py`) tồn tại nhưng **chưa được tích hợp** vào pipeline chính (`main.py`).
- `app.py` có option tone unify nhưng `main.py` không gọi harmonizer.

### Mục tiêu

- Inpaint vùng da sau khi xóa mụn đồng đều tone với vùng xung quanh.
- Tích hợp Skin Tone Harmonizer vào pipeline chính.
- Cân nhắc inpainting chuyên biệt cho da (nếu có tài nguyên).

### Các bước thực hiện

#### Bước 2.1: Tích hợp Skin Tone Harmonizer vào `main.py`

- Import:
  ```python
  from pipelines.skin_tone_harmonizer import harmonize_skin_tone_model
  ```
- Gọi **sau Step 6 (inpaint)** và **trước Step 7 (skin smooth)**:
  ```python
  # Sau inpaint, trước smooth_skin
  inpainted_roi, _ = harmonize_skin_tone_model(
      inpainted_roi, skin_roi, strength=0.35
  )
  ```
- Nếu model không tồn tại (`models/skin_tone/harmonizer.pth`), hàm sẽ trả về ảnh gốc — cần xử lý fallback sạch.

#### Bước 2.2: Train Skin Tone Harmonizer (khi chưa có checkpoint)

- Tham khảo: `kaggle_train_tone_harmonizer.py`, `KAGGLE_TRAINING_NOTE.md`.
- Dataset: `input/` (before) + `target/` (after, Evoto-style) + `masks/` (skin mask, optional).
- Đặt checkpoint tại `models/skin_tone/harmonizer.pth` sau khi train xong.

#### Bước 2.3: Tinh chỉnh pre-dilation mask trước LaMa

- File: `pipelines/inpaint.py`.
- Hiện dùng kernel `(7, 7)` ellipsoid, 2 iterations. Có thể thử:
  - Kernel nhỏ hơn `(5, 5)` nếu vùng inpaint bị "bệt" quá lớn.
  - Tăng iterations nếu mép vùng xóa còn thấy biên.
- Lưu mask debug tại `outputs/debug/step6_inpaint_mask.png` để kiểm tra.

#### Bước 2.4: Cân nhắc inpainting chuyên biệt (long-term)

- Nghiên cứu mô hình diffusion inpainting có điều kiện (ví dụ Stable Diffusion inpainting) hoặc face-specific inpaint.
- Hiện tại LaMa + harmonizer là giải pháp cân bằng tốt chi phí/chất lượng.

---

## 3. Tuning tham số theo loại khuyết điểm

### Vấn đề hiện tại

- Các tham số `expand_px`, `dilate` trước LaMa, `strength` blend cuối, GFPGAN blend đều cố định.
- Evoto cung cấp nhiều preset (mụn nhẹ, sẹo nặng, đỏ da) với bộ tham số khác nhau.

### Mục tiêu

- Có preset hoặc slider cho từng loại khuyết điểm.
- Dễ dàng A/B test và chọn cấu hình tối ưu.

### Các bước thực hiện

#### Bước 3.1: Xác định tham số cần expose

| Tham số           | Vị trí                    | Mô tả ngắn                          |
|-------------------|---------------------------|-------------------------------------|
| `expand_px`       | `mask_expand.py`          | Độ mở rộng mask quanh khuyết điểm   |
| `dilate` iterations | `inpaint.py`           | Số lần dilate mask trước LaMa       |
| `strength`        | `main.py`                 | Cường độ blend cuối (0–1)           |
| GFPGAN `blend`    | `face_restore.py`         | Cường độ restore chi tiết           |
| `blur_sigma`      | `skin_retouch.py`         | Độ mịn da                           |
| `high_freq_weight`| `skin_retouch.py`         | Giữ texture vs làm mịn              |

#### Bước 3.2: Thêm preset vào `main.py`

- Định nghĩa dict preset:
  ```python
  PRESETS = {
      "light":  {"expand_px": 8,  "strength": 0.6, "blend": 0.15},
      "medium": {"expand_px": 12, "strength": 0.8, "blend": 0.20},
      "strong": {"expand_px": 16, "strength": 1.0, "blend": 0.25},
  }
  ```
- Thêm argument CLI `--preset light|medium|strong` và override tham số tương ứng.

#### Bước 3.3: Truyền tham số qua pipeline

- Sửa `run_pipeline(img_rgb, strength=0.8, ...)` để nhận thêm `expand_px`, `inpaint_dilate`, v.v.
- Truyền xuống `expand_mask`, `inpaint`, `restore_face` theo từng bước.

#### Bước 3.4: Thêm slider/radio trong Gradio (`app.py`)

- Thêm dropdown "Preset: Light / Medium / Strong".
- Khi chọn preset, cập nhật các slider tương ứng (hoặc dùng giá trị cố định của preset).

---

## 4. Sản phẩm và UX

### Vấn đề hiện tại

- Chỉ xử lý **face đầu tiên** (face index 0).
- Protect mask từ landmarks đã có nhưng có thể chưa mịn cho lông mày, ria, môi.
- Chưa tối ưu latency cho GPU / realtime preview.

### Mục tiêu

- Hỗ trợ đa khuôn mặt.
- Cải thiện protect mask (mịn hơn, tùy chọn bảo vệ ria/nốt ruồi).
- Giảm thời gian xử lý cho trải nghiệm realtime (nếu cần).

### Các bước thực hiện

#### Bước 4.1: Hỗ trợ đa khuôn mặt

- File: `main.py` — vòng lặp hiện chỉ xử lý `faces[0]`.
- Thay đổi:
  ```python
  for i, face in enumerate(faces):
      bbox = face["bbox"]
      # ... pad, crop face_roi ...
      # ... chạy pipeline cho face_roi ...
      result[y1:y2, x1:x2] = blended_roi
  ```
- Lưu ý: mỗi face cần ROI riêng; landmarks, parsing, skin mask cần crop theo từng bbox.

#### Bước 4.2: Cải thiện protect mask

- File: `pipelines/landmarks.py` — tạo `protect_mask_full` từ 478 điểm.
- Có thể:
  - Tăng vùng buffer quanh mắt, lông mày, môi (dilate polygon).
  - Thêm option exclude ria (vùng dưới mũi) nếu không muốn xóa râu.
- Export `protect_mask` ra debug để kiểm tra: `outputs/debug/step2_protect_mask.png`.

#### Bước 4.3: Tối ưu hiệu năng

- Chạy inference trên GPU (PyTorch, LaMa) — kiểm tra `torch.cuda.is_available()`.
- Giảm độ phân giải inference cho blemish model (ví dụ resize xuống 512 trước khi chạy U-Net/SegFormer).
- Cache model (đã có `_cached_unet`, `_cached_segformer`) — tránh load lại mỗi ảnh.
- Batch processing nếu xử lý nhiều ảnh liên tiếp.

#### Bước 4.4: Preview nhanh (optional)

- Chạy pipeline ở resolution thấp (ví dụ 512px) để preview, sau đó chạy full res khi user xác nhận.
- Hoặc bỏ qua GFPGAN ở bước preview để giảm latency.

---

## 5. Đánh giá có hệ thống

### Vấn đề hiện tại

- Chưa có script đánh giá metrics chuẩn hóa trong project.
- Chưa có bộ ảnh benchmark cố định để so sánh trước/sau mỗi thay đổi.
- Khó biết thay đổi nào thực sự cải thiện chất lượng.

### Mục tiêu

- Có script đánh giá LPIPS, SSIM trên vùng da (hoặc full image).
- Có bộ ảnh benchmark + bảng kết quả trước/sau.
- Quy trình A/B test khi thay đổi model hoặc tham số.

### Các bước thực hiện

#### Bước 5.1: Tạo bộ benchmark

- Tạo thư mục `benchmark/`:
  - `benchmark/images/` — 50–100 ảnh chân dung đại diện (nhiều loại da, ánh sáng, mức độ mụn).
  - `benchmark/reference/` (optional) — ảnh Evoto retouch tương ứng nếu có, để so sánh.
- Danh sách file lưu trong `benchmark/manifest.txt` hoặc đọc trực tiếp từ thư mục.

#### Bước 5.2: Script đánh giá metrics

- Tạo `scripts/evaluate_pipeline.py`:
  - Đọc từng ảnh, chạy `run_pipeline`, lưu output.
  - Tính LPIPS (perceptual similarity), SSIM so với input hoặc reference.
  - In ra bảng: tên file, LPIPS, SSIM, thời gian (s).
- Cài đặt: `pip install lpips scikit-image` (hoặc `piq`).

#### Bước 5.3: Đánh giá segmentation (riêng)

- Nếu có dataset có ground-truth mask (ví dụ `acne_mega`):
  - Chạy `detect_blemishes` trên ảnh test.
  - So sánh mask dự đoán với mask thật: IoU, Dice, precision, recall.
- File: có thể mở rộng `tests/test_blemish_detection.py` hoặc tạo `scripts/evaluate_segmentation.py`.

#### Bước 5.4: Blind A/B test (human evaluation)

- Xuất cặp ảnh (before, after A) và (before, after B) cho cùng input.
- Người đánh giá chọn phiên bản tốt hơn hoặc chấm điểm 1–5.
- Lưu kết quả vào CSV để phân tích thống kê.

#### Bước 5.5: Regression test

- Mỗi khi đổi code, chạy benchmark trên 10–20 ảnh cố định.
- So sánh metrics với baseline; cảnh báo nếu LPIPS/SSIM thay đổi > ngưỡng cho phép.
- Có thể tích hợp vào CI nếu dùng GitHub Actions hoặc tương tự.

---

## Tóm tắt checklist

| Phần                    | Ưu tiên | Trạng thái gợi ý |
|-------------------------|---------|-------------------|
| Train blemish model     | Cao     | Bắt buộc để thoát heuristic |
| Tích hợp tone harmonizer| Cao     | 1–2 giờ chỉnh `main.py`     |
| Preset tham số          | Trung bình | Vài giờ thêm CLI/UI     |
| Đa khuôn mặt            | Trung bình | Phụ thuộc use case        |
| Bộ benchmark + metrics  | Cao     | Nền tảng cho mọi cải tiến  |
| Tối ưu hiệu năng        | Thấp    | Khi cần realtime           |
| Giữ nốt ruồi            | Tùy chọn | Nếu product yêu cầu       |

---

*Tài liệu tham khảo: `main.py`, `DATASET_PREP_GUIDE.md`, `KAGGLE_TRAINING_NOTE.md`, `KAGGLE_SEGFORMER_GUIDE.md`.*
