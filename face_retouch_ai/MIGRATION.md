# Lịch sử gộp project

Nội dung từ **ai-retouch-test** đã được gộp vào **face_retouch_ai** (thư mục `ai-retouch-test` đã gỡ khỏi repo).

## Cấu trúc

```
face_retouch_ai/
├── app.py              # Gradio UI
├── main.py             # CLI pipeline
├── pipelines/          # Các bước xử lý
├── scripts/            # Huấn luyện, dataset
├── tests/
├── utils/              # download_models.py
├── models/             # Checkpoints
├── outputs/            # Kết quả, debug
└── requirements.txt
```

## Chạy ứng dụng

```bash
cd face_retouch_ai
pip install -r requirements.txt
python app.py          # Gradio UI
# hoặc
python main.py --input ảnh.jpg --output kết_quả.jpg   # CLI
```

## Models

Đặt các model vào `face_retouch_ai/models/`:

- `models/mediapipe/face_landmarker.task`
- `models/face_parsing/79999_iter.pth` hoặc `models/face_parsing_79999_iter.pth`
- `models/face_restore/GFPGANv1.4.pth`
- LaMa (auto-download), InsightFace buffalo_l, …

Chạy `python utils/download_models.py` để tải các model thiếu.
