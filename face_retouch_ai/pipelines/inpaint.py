"""
Step 6 — AI Inpainting using LaMa (via simple-lama-inpainting).
Auto-downloads model on first use (~200 MB to ~/.cache).
Fallback: OpenCV Telea inpainting.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

DEBUG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _lama_inpaint(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """LaMa inpainting via simple-lama-inpainting package."""
    try:
        from simple_lama_inpainting import SimpleLama

        h, w = img_rgb.shape[:2]
        lama = SimpleLama()
        pil_img = Image.fromarray(img_rgb)
        # SimpleLama expects a grayscale mask (0=keep, 255=inpaint)
        pil_mask = Image.fromarray(mask).convert("L")
        result = np.array(lama(pil_img, pil_mask))
        # Ensure output matches input dimensions (LaMa may pad internally)
        if result.shape[:2] != (h, w):
            result = cv2.resize(result, (w, h))
        return result
    except Exception as exc:
        print(f"  LaMa failed: {exc}")
        return None


def _opencv_inpaint(img_rgb: np.ndarray, mask: np.ndarray, radius: int = 7) -> np.ndarray:
    """OpenCV Telea inpainting fallback."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    result = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def inpaint(
    img_rgb: np.ndarray,
    blemish_mask: np.ndarray,
    soft_mask: np.ndarray,
):
    """
    Inpaint blemish regions, then alpha-blend with soft mask.

    Returns
    -------
    result : np.ndarray (H, W, 3) RGB uint8
    method : str — "lama" or "opencv"
    info   : str
    """
    print("[Step 6] AI Inpainting")

    # Binary mask for inpainting engine (need hard edges)
    hard_mask = (blemish_mask > 0).astype(np.uint8) * 255

    # Try LaMa first
    inpainted = _lama_inpaint(img_rgb, hard_mask)
    if inpainted is not None:
        method = "lama"
        print("  Using LaMa inpainting.")
    else:
        inpainted = _opencv_inpaint(img_rgb, hard_mask)
        method = "opencv"
        print("  Using OpenCV Telea fallback.")

    # Alpha-blend using soft mask (smooth transition at edges)
    alpha = soft_mask[:, :, np.newaxis]  # (H, W, 1)
    result = (
        img_rgb.astype(np.float32) * (1 - alpha)
        + inpainted.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)

    # Debug
    cv2.imwrite(
        str(DEBUG_DIR / "step6_inpainted_raw.jpg"),
        cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(DEBUG_DIR / "step6_inpainted_blended.jpg"),
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
    )

    diff = np.abs(result.astype(float) - img_rgb.astype(float))
    info = (
        f"Method: {method}\n"
        f"Mean diff: {diff.mean():.2f}/255\n"
        f"Max diff: {diff.max():.0f}\n"
    )
    print(f"[Step 6] Done — method={method}, mean_diff={diff.mean():.2f}")
    return result, method, info
