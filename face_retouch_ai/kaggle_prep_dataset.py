"""
Cell 1 - Chuẩn hóa dataset Kaggle về format images/ + masks/
============================================================
Copy nội dung file này vào Cell 1 của Kaggle Notebook. Chạy trước Cell 2 (training).

Hỗ trợ 1 hoặc nhiều dataset. Sửa INPUT_DATASETS và OUTPUT_ROOT.
"""

from pathlib import Path
import random
import shutil

# === SỬA CÁC BIẾN NÀY ===
INPUT_DATASETS = [
    Path("/kaggle/input/datasets/mosharofhossain/isic-2017-skin-lesion-segmentation-dataset"),
    Path("/kaggle/input/datasets/volodymyrpivoshenko/skin-cancer-lesions-segmentation"),
    Path("/kaggle/input/datasets/devdope/skin-lesion-dataset-using-segmentation"),
]
OUTPUT_ROOT = Path("/kaggle/working/acne_mega")

# --- Chọn một trong các chế độ ---

# True = mỗi nguồn lấy ĐÚNG bằng nhau (sau đó áp BALANCED_MAX_PER_SOURCE + MAX_TOTAL_PAIRS)
BALANCE_DATASETS_ABSOLUTE = True

# Trần mỗi nguồn khi cân bằng (nhỏ = ít disk, đủ chỗ copy + nén zip)
# 1200 x 3 nguồn = 3600 cặp ~1.2GB | 800 x 3 = 2400 ~0.8GB
BALANCED_MAX_PER_SOURCE = 1_200

# Trần tổng: mỗi nguồn <= MAX_TOTAL_PAIRS // số_nguồn
MAX_TOTAL_PAIRS = 3_600

# Khi BALANCE_DATASETS_ABSOLUTE = False
MAX_PER_DATASET = 3_000
# ===============================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg"}


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Tìm cặp (image, mask) — hỗ trợ nhiều format Kaggle phổ biến."""
    root = Path(root)

    # Thử lần lượt: root, root/ISIC_2017, root/data, root/Dataset (devdope)
    for r in [root, root / "ISIC_2017", root / "data", root / "Dataset"]:
        if r != root and not r.exists():
            continue
        pairs = _find_pairs_in(r)
        if pairs:
            return pairs
    return []


def _find_pairs_in(root: Path) -> list[tuple[Path, Path]]:
    """Tìm cặp (image, mask) trong root."""
    pairs = []
    seen = set()

    # Format A: images/ và masks/ có sẵn, stem trùng
    img_dir = root / "images"
    mask_dir = root / "masks"
    if img_dir.exists() and mask_dir.exists():
        for img in img_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                for ext in [".png", ".jpg"]:
                    m = mask_dir / f"{img.stem}{ext}"
                    if m.exists():
                        key = (str(img), str(m))
                        if key not in seen:
                            seen.add(key)
                            pairs.append((img, m))
                        break
        if pairs:
            return pairs

    # Format B: train/images + train/masks (hoặc valid, test)
    for split in ["train", "valid", "test"]:
        base = root / split
        imgs_dir = base / "images"
        masks_dir = base / "masks" if (base / "masks").exists() else base / "labels"
        if not imgs_dir.exists() or not masks_dir.exists():
            continue
        for img in imgs_dir.iterdir():
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            for ext in [".png", ".jpg"]:
                m = masks_dir / f"{img.stem}{ext}"
                if m.exists():
                    key = (str(img), str(m))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((img, m))
                    break

    # Format C.1: ISIC 2017 — training_images + training_masks, val_images + val_masks, test_images + test_masks
    for imgs_name, masks_name in [
        ("training_images", "training_masks"),
        ("val_images", "val_masks"),
        ("test_images", "test_masks"),
    ]:
        imgs_dir = root / imgs_name
        masks_dir = root / masks_name
        if not imgs_dir.exists() or not masks_dir.exists():
            continue
        for img in imgs_dir.iterdir():
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            # ISIC: mask có thể là stem.png hoặc stem_segmentation.png
            for mask_name in [f"{img.stem}.png", f"{img.stem}.jpg", f"{img.stem}_segmentation.png"]:
                m = masks_dir / mask_name
                if m.exists():
                    key = (str(img), str(m))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((img, m))
                    break

    # Format D: devdope — Dataset/Train/X, Val/X, Test/X; mỗi X có images/ + masks/ hoặc ảnh trực tiếp
    for split in ["Train", "Val", "Test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for sub in split_dir.iterdir():
            if not sub.is_dir():
                continue
            # X/images + X/masks
            imgs_dir = sub / "images"
            masks_dir = sub / "masks"
            if imgs_dir.exists() and masks_dir.exists():
                for img in imgs_dir.iterdir():
                    if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                        continue
                    for ext in [".png", ".jpg"]:
                        m = masks_dir / f"{img.stem}{ext}"
                        if m.exists():
                            key = (str(img), str(m))
                            if key not in seen:
                                seen.add(key)
                                pairs.append((img, m))
                            break
            else:
                # Ảnh và mask cùng thư mục (stem trùng)
                for img in sub.iterdir():
                    if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                        continue
                    if "_mask" in img.stem.lower() or "_seg" in img.stem.lower():
                        continue
                    for m in sub.glob(f"{img.stem}*"):
                        if m != img and m.suffix.lower() in MASK_EXTS:
                            key = (str(img), str(m))
                            if key not in seen:
                                seen.add(key)
                                pairs.append((img, m))
                            break

    # Format C: Cùng thư mục — image.jpg và image_Segmentation.png (ISIC style)
    if not pairs:
        for img in root.rglob("*"):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            if "_segmentation" in img.stem.lower() or "_mask" in img.stem.lower():
                continue  # skip file mask bị nhầm là image
            parent = img.parent
            for suf in ["_segmentation", "_Segmentation", "_mask", "_Mask"]:
                m = parent / f"{img.stem}{suf}.png"
                if not m.exists():
                    m = parent / f"{img.stem}{suf}.jpg"
                if m.exists():
                    key = (str(img), str(m))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((img, m))
                    break

    return pairs


if __name__ == "__main__":
    all_pairs = []

    if BALANCE_DATASETS_ABSOLUTE:
        per_source = []
        for ds_path in INPUT_DATASETS:
            if not ds_path.exists():
                print(f"[Bỏ qua - không tồn tại] {ds_path}")
                continue
            p = find_pairs(ds_path)
            if not p:
                print(f"[{ds_path.name}] 0 cặp — bỏ qua")
                continue
            per_source.append((ds_path.name, p))
            print(f"[{ds_path.name}] Tìm thấy {len(p)} cặp")

        if not per_source:
            pairs = []
        else:
            n_min = min(len(p) for _, p in per_source)
            n_take = n_min
            if BALANCED_MAX_PER_SOURCE is not None:
                n_take = min(n_take, BALANCED_MAX_PER_SOURCE)
            if MAX_TOTAL_PAIRS is not None:
                cap = MAX_TOTAL_PAIRS // len(per_source)
                n_take = min(n_take, cap)
            print(f"[Cân bằng] Mỗi nguồn lấy {n_take} cặp (n_min={n_min}, cap nguồn={BALANCED_MAX_PER_SOURCE})")
            for name, p in per_source:
                random.seed(hash(name) % 2**32)
                sampled = random.sample(p, n_take)
                all_pairs.extend(sampled)
                print(f"[{name}] Cân bằng: {len(sampled)}/{len(p)} cặp")
        pairs = all_pairs
        print(f"\nTổng cộng (cân bằng tuyệt đối): {len(pairs)} cặp")
    else:
        for ds_path in INPUT_DATASETS:
            if not ds_path.exists():
                print(f"[Bỏ qua - không tồn tại] {ds_path}")
                continue
            p = find_pairs(ds_path)
            n_orig = len(p)
            if MAX_PER_DATASET and n_orig > MAX_PER_DATASET:
                random.seed(hash(ds_path.name) % 2**32)
                p = random.sample(p, MAX_PER_DATASET)
                print(f"[{ds_path.name}] Lấy {len(p)}/{n_orig} cặp (MAX_PER_DATASET)")
            else:
                print(f"[{ds_path.name}] Tìm thấy {len(p)} cặp")
            all_pairs.extend(p)

        pairs = all_pairs
        print(f"\nTổng cộng: {len(pairs)} cặp image-mask")

        if MAX_TOTAL_PAIRS and len(pairs) > MAX_TOTAL_PAIRS:
            random.seed(42)
            pairs = random.sample(pairs, MAX_TOTAL_PAIRS)
            print(f"[Giới hạn] Lấy {len(pairs)} cặp (MAX_TOTAL_PAIRS)")

    if not pairs:
        raise SystemExit(
            "Không tìm thấy cặp nào. Kiểm tra INPUT_DATASETS và path.\n"
            "Chạy: !find /kaggle/input -maxdepth 5 -type d"
        )

    out_img = OUTPUT_ROOT / "images"
    out_mask = OUTPUT_ROOT / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    for i, (img_path, mask_path) in enumerate(pairs):
        shutil.copy2(img_path, out_img / f"{i:06d}{img_path.suffix.lower()}")
        shutil.copy2(mask_path, out_mask / f"{i:06d}.png")

    print(f"Đã chuẩn hóa vào {OUTPUT_ROOT}")
    print(f"  images: {len(list(out_img.iterdir()))}")
    print(f"  masks:  {len(list(out_mask.iterdir()))}")
    print(f"\nDATASET_ROOT cho Cell 2: {OUTPUT_ROOT}")
    print("Chạy kaggle_zip_dataset.py để nén thành zip.")
