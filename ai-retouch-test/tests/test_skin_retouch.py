"""
test_skin_retouch.py
====================
STEP 6 — Frequency-Separation Skin Smoothing.

Uses frequency separation:
  low  = gaussian(image, sigma=8)
  high = image − low
  smooth_low = bilateral(low, d=9, sigma=75)
  result = smooth_low + high * 0.8

This preserves pores and natural texture while smoothing discolouration.
Excludes facial features via no_smooth_mask / strict skin mask.

Output: outputs/debug/skin_retouched.jpg

Usage:
    python tests/test_skin_retouch.py
"""

import sys
import glob
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEBUG_DIR = OUTPUTS_DIR / "debug"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Frequency separation parameters
SMOOTH_SIGMA = 8       # Gaussian sigma for low-freq extraction
BILATERAL_D = 9
BILATERAL_SIGMA = 75
TEXTURE_WEIGHT = 0.8   # high * 0.8 preserves pores


def get_input_image() -> str:
    """Return path to the first image found in inputs/."""
    try:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            matches = glob.glob(str(INPUTS_DIR / ext))
            if matches:
                return matches[0]
        print("[ERROR] No image found in inputs/.")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


def load_mask(name: str, h: int, w: int) -> np.ndarray:
    """Load a grayscale mask by name from debug/ or outputs/."""
    try:
        for directory in (DEBUG_DIR, OUTPUTS_DIR):
            path = directory / name
            if path.exists():
                print(f"[INFO] Loading mask from {path}")
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return None
    except Exception as exc:
        print(f"[WARNING] Could not load {name}: {exc}")
        return None


def test_skin_retouch() -> None:
    """Frequency-separation smoothing: smooth base tones, preserve pore texture."""
    try:
        image_path = get_input_image()
        print(f"[INFO] Loading image: {image_path}")

        # Prefer inpainted output as input
        inpainted_path = DEBUG_DIR / "inpainted.jpg"
        if not inpainted_path.exists():
            inpainted_path = OUTPUTS_DIR / "inpainted.jpg"

        if inpainted_path.exists():
            print(f"[INFO] Using inpainted result: {inpainted_path}")
            img = cv2.imread(str(inpainted_path))
        else:
            img = cv2.imread(image_path)

        if img is None:
            print("[ERROR] Could not read image.")
            return

        h, w = img.shape[:2]
        print(f"[INFO] Image size: {w}x{h}")

        # Load strict skin mask (features already excluded)
        skin_mask = load_mask("skin_mask_strict.png", h, w)
        if skin_mask is None:
            skin_mask = load_mask("skin_mask.png", h, w)
        if skin_mask is None:
            skin_mask = load_mask("face_parsing_skin.jpg", h, w)
        if skin_mask is None:
            print("[WARNING] No skin mask found. Smoothing full image.")
            skin_mask = np.full((h, w), 255, dtype=np.uint8)

        # Exclude facial features if not already in strict mask
        no_smooth = load_mask("no_smooth_mask.png", h, w)
        if no_smooth is not None:
            skin_mask = cv2.subtract(skin_mask, no_smooth)
            print("[INFO] Excluded eyes/brows/lips from smoothing.")

        # === Frequency Separation ===
        img_f = img.astype(np.float64)

        # Low-frequency: Gaussian blur captures skin tone
        low_freq = cv2.GaussianBlur(img_f, (0, 0), sigmaX=SMOOTH_SIGMA)

        # High-frequency: pores, fine lines, natural texture
        high_freq = img_f - low_freq

        # Bilateral filter on low-frequency only
        low_u8 = np.clip(low_freq, 0, 255).astype(np.uint8)
        low_smoothed = cv2.bilateralFilter(low_u8, d=BILATERAL_D,
                                           sigmaColor=BILATERAL_SIGMA,
                                           sigmaSpace=BILATERAL_SIGMA)
        low_smoothed_f = low_smoothed.astype(np.float64)

        # Recombine: smoothed base + preserved texture
        retouched_f = low_smoothed_f + high_freq * TEXTURE_WEIGHT
        retouched = np.clip(retouched_f, 0, 255).astype(np.uint8)

        print(f"[INFO] Frequency separation: sigma={SMOOTH_SIGMA}, "
              f"bilateral d={BILATERAL_D}, texture_weight={TEXTURE_WEIGHT}")

        # Apply only within skin mask
        mask_3 = cv2.merge([skin_mask] * 3).astype(np.float64) / 255.0
        result = np.clip(
            retouched.astype(np.float64) * mask_3
            + img.astype(np.float64) * (1 - mask_3),
            0, 255,
        ).astype(np.uint8)

        # Save outputs
        cv2.imwrite(str(DEBUG_DIR / "skin_retouched.jpg"), result)
        cv2.imwrite(str(OUTPUTS_DIR / "skin_retouched.jpg"), result)
        print(f"[OK] Retouched result saved to {DEBUG_DIR / 'skin_retouched.jpg'}")

        comparison = np.hstack([img, result])
        cv2.imwrite(str(OUTPUTS_DIR / "skin_retouch_comparison.jpg"), comparison)
        print(f"[OK] Comparison saved to {OUTPUTS_DIR / 'skin_retouch_comparison.jpg'}")

    except Exception as exc:
        print(f"[ERROR] Skin retouch failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    test_skin_retouch()
