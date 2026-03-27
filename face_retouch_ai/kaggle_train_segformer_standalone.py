"""
Cell 2 - Train SegFormer cho blemish/acne segmentation trên Kaggle
==================================================================
Copy toàn bộ nội dung file này vào Cell 2 của Kaggle Notebook.

Yêu cầu: Cell 1 (kaggle_prep_dataset.py) đã chạy xong, hoặc dataset có sẵn images/ + masks/

Nếu lỗi "no kernel image": Bật FORCE_CPU=True rồi Restart kernel, Run All.
"""

import os
import subprocess
import sys

# Bật True nếu GPU báo "no kernel image" — train trên CPU (chậm)
FORCE_CPU = True
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

subprocess.run([sys.executable, "-m", "pip", "install", "evaluate", "accelerate", "-q"], check=False)

from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ColorJitter
from datasets import Dataset, Image as HFImage, DatasetDict
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)
import evaluate

# === SỬA PATH NÀY ===
# Nếu đã chạy Cell 1: dùng /kaggle/working/acne_mega
# Nếu dataset từ notebook khác: /kaggle/input/<output-slug>/acne_mega
DATASET_ROOT = Path("/kaggle/working/acne_mega")
# ====================

# Output: dùng /tmp nếu /kaggle/working hết chỗ (Errno 28)
OUTPUT_DIR = Path("/tmp/segformer_acne")
CHECKPOINT = "nvidia/mit-b0"

# Giới hạn mẫu (None = dùng hết). CPU: nên <= 2_000
MAX_TRAIN_SAMPLES = 2_000   # GPU: 5k~2h | CPU: 2k ~5-10h
MAX_EVAL_SAMPLES = 500     # giảm = ít RAM khi eval
NUM_TRAIN_EPOCHS = 3       # 3 epoch đủ thử
BATCH_SIZE = 2             # OOM nếu dùng 4+

# Binary: background (0), blemish (1)
id2label = {0: "background", 1: "blemish"}
label2id = {v: k for k, v in id2label.items()}
num_labels = 2


def load_mask_as_labels(mask_path: str) -> np.ndarray:
    """Load mask; convert 0/255 -> 0/1 (background/blemish)."""
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 0).astype(np.int64)


def _ensure_rgb(img):
    """Đảm bảo ảnh là PIL RGB — tránh lỗi 'Unable to infer channel dimension format'."""
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
    """Thu thập (image_path, mask_path) từ images/ và masks/."""
    img_dir = root / "images"
    mask_dir = root / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Thiếu images/ hoặc masks/ trong {root}. Chạy Cell 1 trước hoặc kiểm tra path."
        )
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
    pairs = collect_pairs(DATASET_ROOT)
    print(f"Tìm thấy {len(pairs)} cặp image-mask")
    if len(pairs) < 10:
        raise ValueError("Dataset quá ít. Cần ít nhất 10 cặp.")

    # Split 80/20
    np.random.seed(42)
    idx = np.random.permutation(len(pairs))
    split = int(0.8 * len(pairs))
    train_idx, val_idx = idx[:split], idx[split:]
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]

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

    # Tạo Dataset với image path và label (mask path — convert khi transform)
    def create_dataset(pairs):
        img_paths = [p[0] for p in pairs]
        mask_paths = [p[1] for p in pairs]
        return Dataset.from_dict({"image": img_paths, "label_path": mask_paths})

    train_ds = create_dataset(train_pairs)
    val_ds = create_dataset(val_pairs)

    # Cast image column
    train_ds = train_ds.cast_column("image", HFImage())
    val_ds = val_ds.cast_column("image", HFImage())

    # Load processor
    image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    def _to_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    def train_transforms(example_batch):
        images = _to_list(example_batch["image"])
        paths = _to_list(example_batch["label_path"])
        images = [_ensure_rgb(x) for x in images]
        images = [jitter(x) for x in images]
        labels = [Image.fromarray(load_mask_as_labels(p).astype(np.uint8)) for p in paths]
        inputs = image_processor(images, labels, return_tensors="pt")
        return {k: v.squeeze(0) if v.dim() > 0 and v.shape[0] == 1 else v for k, v in inputs.items()}

    def val_transforms(example_batch):
        images = _to_list(example_batch["image"])
        paths = _to_list(example_batch["label_path"])
        images = [_ensure_rgb(x) for x in images]
        labels = [Image.fromarray(load_mask_as_labels(p).astype(np.uint8)) for p in paths]
        inputs = image_processor(images, labels, return_tensors="pt")
        return {k: v.squeeze(0) if v.dim() > 0 and v.shape[0] == 1 else v for k, v in inputs.items()}

    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)

    # Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        CHECKPOINT, id2label=id2label, label2id=label2id, num_labels=num_labels
    )
    if FORCE_CPU:
        model = model.to("cpu")
        print("[Train trên CPU]")

    # Metric
    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        if logits_tensor.dim() == 3:
            logits_tensor = logits_tensor.unsqueeze(0)
        pred_labels = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1).detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics[k] = v.tolist()
        return metrics

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=6e-5,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        use_cpu=FORCE_CPU,         # ép CPU khi GPU lỗi
        dataloader_num_workers=0,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none",
    )

    # Tránh lỗi CUDA "no kernel image" — dùng shape thay vì .ne() trên GPU
    class SegTrainer(Trainer):
        def _get_num_items_in_batch(self, batch_samples, device):
            try:
                n = 0
                for batch in batch_samples:
                    L = batch.get("labels")
                    if L is not None and hasattr(L, "shape") and len(L.shape) >= 2:
                        n += L.shape[0] * L.shape[1] * L.shape[2]
                return n if n > 0 else len(batch_samples)
            except Exception:
                return len(batch_samples)

    trainer = SegTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    image_processor.save_pretrained(str(OUTPUT_DIR))

    # Copy sang /kaggle/working nếu đang dùng /tmp (để tải được)
    final_dir = Path("/kaggle/working/segformer_acne")
    if str(OUTPUT_DIR).startswith("/tmp"):
        try:
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(OUTPUT_DIR, final_dir)
            print(f"\nĐã copy model: {OUTPUT_DIR} -> {final_dir}")
            print("Download thư mục segformer_acne từ Output.")
        except OSError as e:
            print(f"\nModel tại {OUTPUT_DIR} (không copy được sang working: {e})")
    else:
        print(f"\nĐã lưu model tại: {OUTPUT_DIR}")
    print("Copy vào face_retouch_ai/models/blemish_seg/segformer_acne/")


if __name__ == "__main__":
    main()
