"""
Step 5 — Smart Mask Expansion.
Dilate blemish mask with 5×5 ellipse + GaussianBlur for soft edges.
"""

import cv2
import numpy as np
from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def expand_mask(
    blemish_mask: np.ndarray,
    dilate_size: int = 5,
    dilate_iters: int = 1,
    blur_ksize: int = 9,
):
    """
    Expand blemish mask with dilation and Gaussian blur.

    Returns
    -------
    soft_mask : np.ndarray (H, W) float32 in [0, 1]
    info : str
    """
    print("[Step 5] Mask Expansion")

    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
    )
    dilated = cv2.dilate(blemish_mask, kern, iterations=dilate_iters)
    blurred = cv2.GaussianBlur(dilated, (blur_ksize, blur_ksize), 0)
    soft_mask = blurred.astype(np.float32) / 255.0

    # Debug
    cv2.imwrite(str(DEBUG_DIR / "step5_mask_dilated.png"), dilated)
    cv2.imwrite(str(DEBUG_DIR / "step5_mask_soft.png"), (soft_mask * 255).astype(np.uint8))

    n_hard = np.count_nonzero(blemish_mask)
    n_dilated = np.count_nonzero(dilated)
    info = (
        f"Original blemish px: {n_hard}\n"
        f"After dilation: {n_dilated}\n"
        f"Soft mask range: [{soft_mask.min():.3f}, {soft_mask.max():.3f}]\n"
    )
    print(f"[Step 5] Done — {n_hard} → {n_dilated} px (dilated)")
    return soft_mask, info
