"""
view_results.py
===============
Hiển thị tất cả ảnh output trên giao diện cửa sổ.
Nhấn phím bất kỳ để xem ảnh tiếp theo, nhấn ESC để thoát.

Usage:
    python view_results.py
"""

import cv2
import sys
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

DISPLAY_ORDER = [
    ("1. Face Detection", "debug/face_detection.jpg"),
    ("2. Landmarks", "debug/landmarks.jpg"),
    ("3. Face Parsing - Mask", "debug/face_parsing_mask.jpg"),
    ("3. Face Parsing - Skin", "debug/skin_mask.png"),
    ("4. Blemish Mask", "debug/blemish_mask.png"),
    ("5. Inpainting", "debug/inpainted.jpg"),
    ("5. Inpainting - Comparison", "inpainted_comparison.jpg"),
    ("6. Skin Retouch", "debug/skin_retouched.jpg"),
    ("6. Skin Retouch - Comparison", "skin_retouch_comparison.jpg"),
    ("7. Texture Restore", "debug/texture_restored.jpg"),
    ("7. Texture Restore - Comparison", "texture_restore_comparison.jpg"),
    ("Final Result", "final_result.jpg"),
    ("Original | Final", "comparison.jpg"),
]


def main():
    found = []
    for title, filename in DISPLAY_ORDER:
        path = OUTPUTS_DIR / filename
        if path.exists():
            found.append((title, path))

    if not found:
        print("[ERROR] Không tìm thấy ảnh nào trong outputs/.")
        print("        Hãy chạy các test trước: python tests/test_*.py")
        sys.exit(1)

    print(f"Tìm thấy {len(found)} ảnh. Nhấn phím bất kỳ = ảnh tiếp, ESC = thoát.\n")

    for i, (title, path) in enumerate(found):
        img = cv2.imread(str(path))
        if img is None:
            continue

        # Resize nếu ảnh quá lớn (giữ tỉ lệ, max 1200px chiều rộng)
        h, w = img.shape[:2]
        max_w = 1200
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (max_w, int(h * scale)))

        window_name = "AI Retouch Viewer"
        print(f"  Hien thi: [{i+1}/{len(found)}] {title} ({path.name})")
        cv2.setWindowTitle(window_name, f"[{i+1}/{len(found)}] {title}")
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            print("\nDa thoat.")
            break
    else:
        print("\nDa xem het tat ca anh.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
