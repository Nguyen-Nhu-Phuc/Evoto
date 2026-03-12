"""
Step 8 — Clamped High-Pass Texture Restore.
Adds back fine skin detail lost during smoothing.
high_pass = GaussianBlur(gray, sigma) − gray
clamped = clamp(high_pass, −limit, +limit) * amount
"""

import cv2
import numpy as np
from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def restore_texture(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    sigma: float = 2.0,
    clamp_limit: float = 15.0,
    amount: float = 0.08,
):
    """
    Restore skin texture by adding back clamped high-pass detail.

    Parameters
    ----------
    sigma : float — Gaussian blur sigma for high-pass extraction
    clamp_limit : float — max ± intensity adjustment
    amount : float — strength of texture restoration [0–1]

    Returns
    -------
    result : np.ndarray (H, W, 3) RGB uint8
    info   : str
    """
    print("[Step 8] Texture Restore")

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ksize = int(sigma * 6) | 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    # High-pass = original − blurred (positive = detail)
    high_pass = gray - blurred
    clamped = np.clip(high_pass, -clamp_limit, clamp_limit) * amount

    # Apply within skin mask only
    mask_f = skin_mask.astype(np.float32) / 255.0
    clamped_masked = clamped * mask_f

    # Add to all channels
    result = img_rgb.astype(np.float32)
    for c in range(3):
        result[:, :, c] += clamped_masked
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Debug
    vis_hp = ((high_pass + clamp_limit) / (2 * clamp_limit) * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(DEBUG_DIR / "step8_high_pass.jpg"), vis_hp)
    cv2.imwrite(
        str(DEBUG_DIR / "step8_texture_restored.jpg"),
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
    )

    diff = np.abs(result.astype(float) - img_rgb.astype(float))
    info = (
        f"Sigma: {sigma}, clamp: ±{clamp_limit}, amount: {amount}\n"
        f"Mean texture adjust: {diff.mean():.3f}/255\n"
        f"Max adjust: {diff.max():.1f}\n"
    )
    print(f"[Step 8] Done — mean_adj={diff.mean():.3f}")
    return result, info
