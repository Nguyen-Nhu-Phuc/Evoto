"""
Train SegFormer trên Google Colab - dataset đã tải từ Kaggle
============================================================
Hỗ trợ cả file zip hoặc thư mục đã giải nén.
Cấu trúc sau khi giải nén: .../images/ và .../masks/

Cách dùng:
1. Upload file acne_mega.zip (hoặc thư mục) lên Google Drive
2. Colab: Runtime -> Change runtime type -> GPU
3. Chạy lần lượt 2 cell bên dưới
"""

# ============== CELL 1: Mount Drive + cài thư viện ==============
from google.colab import drive
drive.mount("/content/drive")

# Cài thư viện (chạy xong chờ vài giây)
import subprocess
subprocess.run(["pip", "install", "evaluate", "accelerate", "transformers", "datasets", "-q"], check=False)

# ============== CELL 2: Config + Training (chạy sau Cell 1) ==============
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ColorJitter
from datasets import Dataset, Image as HFImage
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)
import evaluate

# === SỬA PATH CHO ĐÚNG ===
# Cách 1: Dataset là file zip → đặt path zip, script sẽ giải nén vào /content
DATASET_ZIP = Path("/content/drive/MyDrive/acne_mega.zip")  # None nếu dùng thư mục

# Cách 2: Dataset đã giải nén → để DATASET_ZIP = None, sửa DATASET_ROOT
DATASET_ROOT = Path("/content/drive/MyDrive/acne_mega")  # dùng khi DATASET_ZIP = None

# Thư mục giải nén tạm (nhanh hơn đọc từ Drive)
EXTRACT_DIR = Path("/content/acne_mega")

# Lưu model vào Drive để không mất khi tắt Colab
OUTPUT_DIR = Path("/content/drive/MyDrive/segformer_acne")
# ========================

CHECKPOINT = "nvidia/mit-b0"
FORCE_CPU = False  # Colab GPU thường ổn

MAX_TRAIN_SAMPLES = None   # None = dùng hết | 5000 = train nhanh
MAX_EVAL_SAMPLES = 2000
NUM_TRAIN_EPOCHS = 5
BATCH_SIZE = 4             # OOM thì hạ xuống 2

id2label = {0: "background", 1: "blemish"}
label2id = {v: k for k, v in id2label.items()}
num_labels = 2


def prepare_dataset() -> Path:
    """Giải nén zip nếu cần, trả về path thư mục dataset."""
    if DATASET_ZIP and DATASET_ZIP.exists():
        print(f"Đang giải nén {DATASET_ZIP} → {EXTRACT_DIR}...")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
            zf.extractall(EXTRACT_DIR)
        # Một số zip có thư mục con (acne_mega/images, acne_mega/masks)
        inner = EXTRACT_DIR / "acne_mega"
        if inner.exists() and (inner / "images").exists():
            return inner
        if (EXTRACT_DIR / "images").exists():
            return EXTRACT_DIR
        raise FileNotFoundError(f"Sau giải nén không tìm thấy images/ trong {EXTRACT_DIR}")
    return DATASET_ROOT


def load_mask_as_labels(mask_path: str) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 0).astype(np.int64)


def _ensure_rgb(img):
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    if hasattr(img, "convert"):
        return img.convert("RGB") if img.mode != "RGB" else img
    return img


def collect_pairs(root: Path):
    img_dir = root / "images"
    mask_dir = root / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Thiếu images/ hoặc masks/ trong {root}")
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = img_path.stem
        for ext in [".png", ".jpg"]:
            mask_path = mask_dir / f"{stem}{ext}"
            if mask_path.exists():
                pairs.append((str(img_path), str(mask_path)))
                break
    return pairs


def main():
    root = prepare_dataset()
    pairs = collect_pairs(root)
    print(f"Tìm thấy {len(pairs)} cặp image-mask")
    if len(pairs) < 10:
        raise ValueError("Dataset quá ít. Cần ít nhất 10 cặp.")

    np.random.seed(42)
    idx = np.random.permutation(len(pairs))
    split = int(0.8 * len(pairs))
    train_pairs = [pairs[i] for i in idx[:split]]
    val_pairs = [pairs[i] for i in idx[split:]]

    if MAX_TRAIN_SAMPLES and len(train_pairs) > MAX_TRAIN_SAMPLES:
        np.random.seed(44)
        shuf = np.random.permutation(len(train_pairs))
        train_pairs = [train_pairs[i] for i in shuf[:MAX_TRAIN_SAMPLES]]
        print(f"[Train] Giới hạn: {len(train_pairs)} mẫu")
    if MAX_EVAL_SAMPLES and len(val_pairs) > MAX_EVAL_SAMPLES:
        np.random.seed(43)
        vidx = np.random.permutation(len(val_pairs))[:MAX_EVAL_SAMPLES]
        val_pairs = [val_pairs[i] for i in vidx]
        print(f"[Eval] Giới hạn: {len(val_pairs)} mẫu")

    def create_dataset(pairs):
        return Dataset.from_dict({
            "image": [p[0] for p in pairs],
            "label_path": [p[1] for p in pairs],
        })

    train_ds = create_dataset(train_pairs).cast_column("image", HFImage())
    val_ds = create_dataset(val_pairs).cast_column("image", HFImage())

    image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    def _to_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    def train_transforms(example_batch):
        images = [_ensure_rgb(x) for x in _to_list(example_batch["image"])]
        images = [jitter(x) for x in images]
        labels = [Image.fromarray(load_mask_as_labels(p).astype(np.uint8)) for p in _to_list(example_batch["label_path"])]
        inputs = image_processor(images, labels, return_tensors="pt")
        return {k: v.squeeze(0) if v.dim() > 0 and v.shape[0] == 1 else v for k, v in inputs.items()}

    def val_transforms(example_batch):
        images = [_ensure_rgb(x) for x in _to_list(example_batch["image"])]
        labels = [Image.fromarray(load_mask_as_labels(p).astype(np.uint8)) for p in _to_list(example_batch["label_path"])]
        inputs = image_processor(images, labels, return_tensors="pt")
        return {k: v.squeeze(0) if v.dim() > 0 and v.shape[0] == 1 else v for k, v in inputs.items()}

    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)

    model = SegformerForSemanticSegmentation.from_pretrained(
        CHECKPOINT, id2label=id2label, label2id=label2id, num_labels=num_labels
    )
    if FORCE_CPU:
        model = model.to("cpu")
        print("[Train trên CPU]")

    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        if logits_tensor.dim() == 3:
            logits_tensor = logits_tensor.unsqueeze(0)
        pred_labels = torch.nn.functional.interpolate(
            logits_tensor, size=labels.shape[-2:],
            mode="bilinear", align_corners=False,
        ).argmax(dim=1).detach().cpu().numpy()
        m = metric.compute(predictions=pred_labels, references=labels, num_labels=num_labels, ignore_index=255, reduce_labels=False)
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in m.items()}

    class SegTrainer(Trainer):
        def _get_num_items_in_batch(self, batch_samples, device):
            try:
                n = sum(
                    b["labels"].shape[0] * b["labels"].shape[1] * b["labels"].shape[2]
                    for b in batch_samples if b.get("labels") is not None and hasattr(b["labels"], "shape")
                )
                return n if n > 0 else len(batch_samples)
            except Exception:
                return len(batch_samples)

    trainer = SegTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            learning_rate=6e-5,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            use_cpu=FORCE_CPU,
            dataloader_num_workers=2,
            save_total_limit=2,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=20,
            remove_unused_columns=False,
            report_to="none",
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    image_processor.save_pretrained(str(OUTPUT_DIR))

    print(f"\nĐã lưu model tại: {OUTPUT_DIR}")
    print("Tải thư mục segformer_acne từ Drive về máy.")
    print("Copy vào: face_retouch_ai/models/blemish_seg/segformer_acne/")


main()
