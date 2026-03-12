"""
Quick test script — runs pipeline on an image and prints a report.

Usage:
    python test_image.py                         # uses default inputs/portrait.jpg
    python test_image.py  path/to/photo.jpg      # any image
    python test_image.py  photo.jpg  0.6         # custom strength
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import run_pipeline

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test(img_path: str | None = None, strength: float = 0.8):
    # Resolve input
    if img_path is None:
        candidates = [
            Path(__file__).resolve().parent / "inputs" / "portrait.jpg",
            Path(__file__).resolve().parent / "inputs" / "portrait.png",
            Path(__file__).resolve().parent.parent / "ai-retouch-test" / "inputs" / "portrait.jpg",
        ]
        for c in candidates:
            if c.exists():
                img_path = str(c)
                break
        if img_path is None:
            print("ERROR: No input image found. Place a portrait in inputs/ or pass a path.")
            sys.exit(1)

    print(f"Testing with: {img_path}  (strength={strength})")

    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Cannot read: {img_path}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    out = run_pipeline(img_rgb, strength=strength)

    # Save
    stem = Path(img_path).stem
    out_path = OUTPUT_DIR / f"{stem}_retouched.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(out["result"], cv2.COLOR_RGB2BGR))
    print(f"\nSaved result: {out_path}")

    # Side-by-side comparison
    cmp = np.hstack([img_rgb, out["result"]])
    cmp_path = OUTPUT_DIR / f"{stem}_comparison.jpg"
    cv2.imwrite(str(cmp_path), cv2.cvtColor(cmp, cv2.COLOR_RGB2BGR))
    print(f"Saved comparison: {cmp_path}")

    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    test(path, s)
