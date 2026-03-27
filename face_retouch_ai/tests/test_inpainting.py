"""
test_inpainting.py
==================
STEP 5 — Edge-Aware Inpainting with adaptive mask expansion.

Per-component dilation (radius = min(4, sqrt(area)/2)), blurred mask
for gradient preservation, gentle Telea inpainting with radius=3.

Output: outputs/debug/inpainted.jpg

Usage:
    python tests/test_inpainting.py
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
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

MAX_MASK_RATIO = 0.20   # max 20% of skin area
INPAINT_RADIUS = 3      # gentle radius
MAX_DILATE_RADIUS = 4   # max per-component dilation


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
    """Load a grayscale mask by name, trying debug/ then outputs/."""
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


def expand_mask_adaptive(mask: np.ndarray, skin_mask: np.ndarray = None) -> np.ndarray:
    """Expand each blemish component with adaptive radius = min(4, sqrt(area)/2)."""
    try:
        h, w = mask.shape[:2]
        original = np.count_nonzero(mask)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        expanded = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 8:
                continue
            radius = min(MAX_DILATE_RADIUS, max(1, int(np.sqrt(area) / 2)))
            component = (labels == i).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (radius * 2 + 1, radius * 2 + 1))
            dilated = cv2.dilate(component, kernel, iterations=1)
            expanded = cv2.bitwise_or(expanded, dilated)

        # Constrain to skin
        if skin_mask is not None:
            sk = (skin_mask > 127).astype(np.uint8) * 255
            expanded = cv2.bitwise_and(expanded, expanded, mask=sk)

        print(f"[INFO] Mask expansion: {original} → {np.count_nonzero(expanded)} px "
              f"({np.count_nonzero(expanded) / max(original, 1):.1f}x)")

        cv2.imwrite(str(DEBUG_DIR / "blemish_mask_expanded.png"), expanded)
        return expanded

    except Exception as exc:
        print(f"[WARNING] Mask expansion failed: {exc}")
        return mask


def test_inpainting() -> None:
    """Edge-aware inpainting: adaptive expansion + blurred mask + gentle radius."""
    try:
        image_path = get_input_image()
        print(f"[INFO] Loading image: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image at {image_path}.")
            return

        h, w = img.shape[:2]
        print(f"[INFO] Image size: {w}x{h}")

        # Load blemish mask
        blemish_mask = load_mask("blemish_mask.png", h, w)
        if blemish_mask is None:
            blemish_mask = load_mask("blemish_mask.jpg", h, w)
        if blemish_mask is None:
            print("[WARNING] No blemish mask found. Generating synthetic mask.")
            blemish_mask = np.zeros((h, w), dtype=np.uint8)
            rng = np.random.RandomState(42)
            for _ in range(5):
                cx = rng.randint(w // 4, 3 * w // 4)
                cy = rng.randint(h // 4, 3 * h // 4)
                r = rng.randint(5, max(6, min(h, w) // 30))
                cv2.circle(blemish_mask, (cx, cy), r, 255, -1)

        # Load skin mask
        skin_mask = load_mask("skin_mask_strict.png", h, w)
        if skin_mask is None:
            skin_mask = load_mask("skin_mask.png", h, w)
        if skin_mask is None:
            skin_mask = load_mask("face_parsing_skin.jpg", h, w)

        # Adaptive per-component expansion
        inp_mask = expand_mask_adaptive(blemish_mask, skin_mask)

        # Safety: limit mask to max 20% of skin area
        if skin_mask is not None:
            skin_pixels = np.count_nonzero(skin_mask)
            mask_pixels = np.count_nonzero(inp_mask)
            max_allowed = int(skin_pixels * MAX_MASK_RATIO)

            if mask_pixels > max_allowed and max_allowed > 0:
                print(f"[INFO] Mask too large ({mask_pixels} > {max_allowed}). Limiting.")
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    inp_mask, connectivity=8)
                limited = np.zeros_like(inp_mask)
                budget = max_allowed
                areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
                areas.sort(key=lambda x: x[1])
                for idx, area in areas:
                    if budget >= area:
                        limited[labels == idx] = 255
                        budget -= area
                inp_mask = limited

        if np.count_nonzero(inp_mask) == 0:
            print("[INFO] No blemish pixels to inpaint.")
            cv2.imwrite(str(OUTPUTS_DIR / "inpainted.jpg"), img)
            return

        # Edge-aware: blur the mask to preserve natural gradients
        blur_mask = cv2.GaussianBlur(inp_mask, (0, 0), sigmaX=1.5)
        blur_mask = (blur_mask > 32).astype(np.uint8) * 255

        # Run inpainting with gentle radius
        print(f"[INFO] Running edge-aware Telea inpainting (radius={INPAINT_RADIUS})...")
        result = cv2.inpaint(img, blur_mask, inpaintRadius=INPAINT_RADIUS,
                             flags=cv2.INPAINT_TELEA)

        # Save outputs
        cv2.imwrite(str(DEBUG_DIR / "inpainted.jpg"), result)
        cv2.imwrite(str(OUTPUTS_DIR / "inpainted.jpg"), result)
        cv2.imwrite(str(DEBUG_DIR / "inpaint_mask_used.png"), blur_mask)
        print(f"[OK] Inpainted result saved to {DEBUG_DIR / 'inpainted.jpg'}")

        comparison = np.hstack([img, result])
        cv2.imwrite(str(OUTPUTS_DIR / "inpainted_comparison.jpg"), comparison)
        print(f"[OK] Comparison saved to {OUTPUTS_DIR / 'inpainted_comparison.jpg'}")

    except Exception as exc:
        print(f"[ERROR] Inpainting failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    test_inpainting()
