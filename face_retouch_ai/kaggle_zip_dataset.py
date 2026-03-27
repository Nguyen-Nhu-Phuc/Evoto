"""
Cell 1b - Nén dataset đã chuẩn hóa thành zip
============================================
Chạy SAU kaggle_prep_dataset.py. Copy vào cell riêng trên Kaggle Notebook.

Lưu ý: Kaggle working dir ~20GB. Dataset 46k ảnh có thể ~10–15GB — không đủ chỗ
để tạo thêm file zip. Nếu lỗi "No space left", tải thư mục acne_mega trực tiếp từ Output.
"""

from pathlib import Path
import shutil

# === SỬA PATH NÀY ===
DATASET_ROOT = Path("/kaggle/working/acne_mega")
# ====================

if __name__ == "__main__":
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATASET_ROOT}. Chạy kaggle_prep_dataset.py trước."
        )

    # Kiểm tra dung lượng
    total = sum(f.stat().st_size for f in DATASET_ROOT.rglob("*") if f.is_file())
    size_gb = total / (1024**3)
    try:
        stat = shutil.disk_usage(DATASET_ROOT)
        free_gb = stat.free / (1024**3)
        print(f"Dataset: {size_gb:.1f} GB | Còn trống: {free_gb:.1f} GB")
        if free_gb < size_gb * 0.5:
            print("[Cảnh báo] Có thể không đủ chỗ cho file zip.")
    except Exception:
        pass

    zip_path = DATASET_ROOT.parent / f"{DATASET_ROOT.name}.zip"
    print(f"Đang nén {DATASET_ROOT} -> {zip_path} ...")

    try:
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", DATASET_ROOT.parent, DATASET_ROOT.name)
        print(f"Đã tạo {zip_path}")
    except OSError as e:
        if e.errno == 28:  # No space left on device
            print(
                "\n[Lỗi] Hết dung lượng. Kaggle working dir có giới hạn.\n"
                "Cách xử lý: Vào Output -> chọn thư mục acne_mega -> Download (Kaggle cho phép tải cả thư mục).\n"
                "Hoặc chạy notebook mới, chỉ Add output của run này rồi chạy Cell 1b (nhiều chỗ trống hơn)."
            )
        raise
