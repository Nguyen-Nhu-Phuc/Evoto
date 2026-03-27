"""
Step 5 — Smart Mask Expansion (Distance-Transform approach).
Expand blemish mask using distance transform for natural falloff,
then Gaussian blur for soft feathered edges.
"""

import cv2
import numpy as np
from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def expand_mask(
    blemish_mask: np.ndarray,
    expand_px: int = 12,
    blur_ksize: int = 11,
    max_expansion_ratio: float = 8.0,
):
    """
    Distance-transform mask expansion (Evoto-style).

    1. Compute distance transform from blemish boundary outward
    2. Expand mask to all pixels within `expand_px` of any blemish
    3. Gaussian blur for soft feathered edges

    This captures the surrounding redness/inflammation zone around
    each blemish, giving LaMa inpainting enough context to work with.

    Returns
    -------
    soft_mask : np.ndarray (H, W) float32 in [0, 1]
    info : str
    """
    try:
        if blemish_mask is None or blemish_mask.size == 0:
            return np.zeros((1, 1), dtype=np.float32), "No mask to expand."

        print(f"[Step 5] Distance-Transform Mask Expansion (expand_px={expand_px})")

        n_hard = np.count_nonzero(blemish_mask)

        # Distance transform from mask boundary outward
        # Invert: dist_transform measures distance from 0-pixels to nearest 255-pixel
        inverted = cv2.bitwise_not(blemish_mask)
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

        # Expand: include all pixels within expand_px distance of a blemish
        expanded = np.zeros_like(blemish_mask)
        expanded[dist < expand_px] = 255
        # Also keep the original mask
        expanded = cv2.bitwise_or(expanded, blemish_mask)

        n_expanded = np.count_nonzero(expanded)

        # Cap expansion ratio to prevent runaway
        if n_hard > 0 and n_expanded / n_hard > max_expansion_ratio:
            # Reduce expand_px iteratively
            cur_px = expand_px
            while cur_px > 2:
                cur_px -= 1
                expanded = np.zeros_like(blemish_mask)
                expanded[dist < cur_px] = 255
                expanded = cv2.bitwise_or(expanded, blemish_mask)
                if np.count_nonzero(expanded) / n_hard <= max_expansion_ratio:
                    break
            n_expanded = np.count_nonzero(expanded)
            print(f"  Capped expansion at {max_expansion_ratio}x (reduced to {cur_px}px)")

        # Soft feathered mask via Gaussian blur
        blurred = cv2.GaussianBlur(expanded, (blur_ksize, blur_ksize), 0)
        soft_mask = blurred.astype(np.float32) / 255.0

        # Debug saves
        cv2.imwrite(str(DEBUG_DIR / "step5_mask_expanded.png"), expanded)
        cv2.imwrite(str(DEBUG_DIR / "step5_mask_soft.png"), (soft_mask * 255).astype(np.uint8))
        # Distance heatmap for debugging
        dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(DEBUG_DIR / "step5_distance_map.png"), dist_vis)

        info = (
            f"Original blemish px: {n_hard}\n"
            f"After distance expansion ({expand_px}px): {n_expanded}\n"
            f"Expansion ratio: {n_expanded / max(n_hard, 1):.1f}x\n"
            f"Soft mask range: [{soft_mask.min():.3f}, {soft_mask.max():.3f}]\n"
        )
        print(f"[Step 5] Done - {n_hard} to {n_expanded} px (distance-transform)")
        return soft_mask, info

    except Exception as exc:
        print(f"[Step 5] ERROR: {exc}")
        h, w = blemish_mask.shape[:2] if blemish_mask is not None else (1, 1)
        return np.zeros((h, w), dtype=np.float32), f"Mask expansion error: {exc}"
