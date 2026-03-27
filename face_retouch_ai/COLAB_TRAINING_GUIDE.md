# Hướng dẫn train SegFormer trên Google Colab

Dùng file **`colab_train_segformer.py`** — dataset đã tải từ Kaggle.

---

## 1. Chuẩn bị dataset

1. Tải **file zip** `acne_mega.zip` (hoặc thư mục) từ Kaggle Output (sau khi chạy `kaggle_prep_dataset.py` + `kaggle_zip_dataset.py`)
2. Upload lên Google Drive — ví dụ: `My Drive/acne_mega.zip`
3. Trong Cell 2, đặt `DATASET_ZIP = Path("/content/drive/MyDrive/acne_mega.zip")` — script sẽ tự giải nén rồi train

Nếu đã giải nén sẵn: `DATASET_ZIP = None` và sửa `DATASET_ROOT` cho đúng path thư mục (có `images/` và `masks/`).

---

## 2. Tạo Notebook trên Colab

1. Vào [colab.research.google.com](https://colab.research.google.com)
2. **File** → **New notebook**
3. **Runtime** → **Change runtime type** → **GPU** (T4) → **Save**

---

## 3. Chạy training (2 cell)

Mở file `colab_train_segformer.py`, copy vào Colab:

**Cell 1** (Mount Drive + cài thư viện):
```python
from google.colab import drive
drive.mount("/content/drive")

import subprocess
subprocess.run(["pip", "install", "evaluate", "accelerate", "transformers", "datasets", "-q"], check=False)
```

**Cell 2** (Config + Training): Copy **phần còn lại** từ `colab_train_segformer.py` (từ `from pathlib import Path` đến `main()`).

**Sửa trong Cell 2:**
```python
# Nếu dataset là file zip:
DATASET_ZIP = Path("/content/drive/MyDrive/acne_mega.zip")

# Nếu đã giải nén: DATASET_ZIP = None và sửa:
DATASET_ROOT = Path("/content/drive/MyDrive/acne_mega")

OUTPUT_DIR = Path("/content/drive/MyDrive/segformer_acne")
```

---

## 4. Tải model về máy

Sau khi train xong, model nằm trong Drive tại `OUTPUT_DIR`. Có thể:
- Mở Drive → tải thư mục `segformer_acne` về
- Hoặc: `!zip -r segformer_acne.zip {OUTPUT_DIR}` rồi tải file zip

Copy thư mục vào:
```
face_retouch_ai/models/blemish_seg/segformer_acne/
```

---

## 5. Lưu ý Colab

| Vấn đề | Giải pháp |
|--------|-----------|
| **Runtime ngắt sau ~12h** | Colab free giới hạn. Lưu checkpoint thường xuyên, dùng `save_strategy="steps"`, `save_steps=500` |
| **Drive chậm** | Copy dataset vào `/content`: `!cp -r /content/drive/MyDrive/acne_mega /content/` rồi `DATASET_ROOT = Path("/content/acne_mega")` |
| **GPU không bật** | Runtime → Change runtime type → GPU |
| **Hết RAM** | Giảm `BATCH_SIZE = 1`, `MAX_TRAIN_SAMPLES = 1000` |

---

## 6. Checklist nhanh

1. [ ] Bật GPU (Runtime → Change runtime type)
2. [ ] Mount Drive + cài thư viện (Cell 1)
3. [ ] Sửa `DATASET_ROOT` và `OUTPUT_DIR` trong Cell 2
4. [ ] `FORCE_CPU = False`
5. [ ] Chạy training (Cell 2)
6. [ ] Tải `segformer_acne` từ Drive về máy
