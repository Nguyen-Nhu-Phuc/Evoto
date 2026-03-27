"""
test_texture_restore.py
=======================
STEP 7 — Clamped High-Pass Texture Restoration.

Extracts high-frequency detail via Gaussian subtraction, clamps to ±20
to prevent re-introducing blemish contrast, and applies at 0.2 strength.
Blemish areas (dilated) are excluded from restoration.

Output: outputs/debug/texture_restored.jpg

Usage:
    python tests/test_texture_restore.py
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

# Clamped high-pass parameters
HP_SIGMA = 2.0       # Gaussian sigma for low-freq extraction
HP_CLAMP = 20        # max ±20 pixel values in high-pass
HP_AMOUNT = 0.2      # blending strength
BLEMISH_DILATE = 5   # extra dilation around blemish zones


def get_best_input() -> str:
    """Use retouched or inpainted output if available; else raw input."""
    try:
        candidates = [
            DEBUG_DIR / "skin_retouched.jpg",
            OUTPUTS_DIR / "skin_retouched.jpg",
            DEBUG_DIR / "inpainted.jpg",
            OUTPUTS_DIR / "inpainted.jpg",
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            matches = glob.glob(str(INPUTS_DIR / ext))
            if matches:
                return matches[0]
        print("[ERROR] No image found.")
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
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return None
    except Exception as exc:
        print(f"[WARNING] Could not load {name}: {exc}")
        return None


def test_texture_restore() -> None:
    """Clamped high-pass texture restoration, excluding blemish areas."""
    try:
        image_path = get_best_input()
        print(f"[INFO] Loading image: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image at {image_path}.")
            return

        h, w = img.shape[:2]
        print(f"[INFO] Image size: {w}x{h}")

        # --- Clamped high-pass ---
        img_f = img.astype(np.float64)
        low_freq = cv2.GaussianBlur(img_f, (0, 0), sigmaX=HP_SIGMA)
        high_pass = img_f - low_freq

        # Clamp to ±HP_CLAMP to avoid re-introducing blemish contrast
        high_pass = np.clip(high_pass, -HP_CLAMP, HP_CLAMP)

        print(f"[INFO] High-pass: sigma={HP_SIGMA}, clamp=±{HP_CLAMP}, amount={HP_AMOUNT}")

        # Build exclusion mask from blemish areas (dilated)
        blemish_mask = load_mask("blemish_mask.png", h, w)
        if blemish_mask is None:
            blemish_mask = load_mask("blemish_mask.jpg", h, w)

        exclude = np.ones((h, w), dtype=np.float64)
        if blemish_mask is not None:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (BLEMISH_DILATE * 2 + 1, BLEMISH_DILATE * 2 + 1))
            dilated = cv2.dilate(blemish_mask, kernel, iterations=1)
            exclude = 1.0 - (dilated.astype(np.float64) / 255.0)
            excluded_px = np.count_nonzero(dilated)
            print(f"[INFO] Excluding {excluded_px} blemish pixels from texture restore.")
        else:
            print("[INFO] No blemish mask found; restoring texture everywhere.")

        # Apply: img + high_pass * amount * exclusion
        exclude_3 = cv2.merge([exclude] * 3)
        result_f = img_f + high_pass * HP_AMOUNT * exclude_3
        result = np.clip(result_f, 0, 255).astype(np.uint8)

        # Save outputs
        cv2.imwrite(str(DEBUG_DIR / "texture_restored.jpg"), result)
        cv2.imwrite(str(OUTPUTS_DIR / "texture_restored.jpg"), result)
        print(f"[OK] Texture-restored result saved to {DEBUG_DIR / 'texture_restored.jpg'}")

        # Save debug high-pass visualization
        hp_vis = np.clip(high_pass + 128, 0, 255).astype(np.uint8)
        cv2.imwrite(str(DEBUG_DIR / "high_pass_clamped.jpg"), hp_vis)

        comparison = np.hstack([img, result])
        cv2.imwrite(str(OUTPUTS_DIR / "texture_restore_comparison.jpg"), comparison)
        print(f"[OK] Comparison saved to {OUTPUTS_DIR / 'texture_restore_comparison.jpg'}")

    except Exception as exc:
        print(f"[ERROR] Texture restoration failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    test_texture_restore()
