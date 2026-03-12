"""
Step 7 — Skin Smoothing via Guided-Filter Frequency Separation.
Low freq  = GaussianBlur(sigma)
High freq = image − low
Smooth low = guidedFilter(low)
Result = smooth_low + high * texture_weight
"""

import cv2
import numpy as np
from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def smooth_skin(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    sigma: float = 7.0,
    guide_radius: int = 9,
    guide_eps: float = 0.01,
    texture_weight: float = 0.85,
):
    """
    Frequency-separation skin smoothing with guided filter.

    Parameters
    ----------
    sigma : float — Gaussian blur sigma for low-freq extraction
    guide_radius : int — guided filter radius
    guide_eps : float — guided filter regularisation
    texture_weight : float — how much high-freq texture to preserve [0–1]

    Returns
    -------
    result : np.ndarray (H, W, 3) RGB uint8
    info   : str
    """
    print("[Step 7] Skin Smoothing — Guided Filter")

    img_f = img_rgb.astype(np.float32) / 255.0

    # ── Frequency separation ──
    ksize = int(sigma * 6) | 1  # ensure odd
    low = cv2.GaussianBlur(img_f, (ksize, ksize), sigma)
    high = img_f - low

    # ── Smooth the low-frequency band with guided filter ──
    smooth_low = cv2.ximgproc.guidedFilter(
        guide=img_f, src=low, radius=guide_radius, eps=guide_eps
    )

    # ── Recombine ──
    recomposed = smooth_low + high * texture_weight
    recomposed = np.clip(recomposed * 255, 0, 255).astype(np.uint8)

    # ── Apply only within skin mask ──
    mask_f = (skin_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    result = (
        img_rgb.astype(np.float32) * (1 - mask_f)
        + recomposed.astype(np.float32) * mask_f
    ).clip(0, 255).astype(np.uint8)

    # Debug
    cv2.imwrite(
        str(DEBUG_DIR / "step7_low_freq.jpg"),
        cv2.cvtColor((low * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(DEBUG_DIR / "step7_high_freq.jpg"),
        cv2.cvtColor(
            ((high + 0.5) * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        ),
    )
    cv2.imwrite(
        str(DEBUG_DIR / "step7_smoothed.jpg"),
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
    )

    diff = np.abs(result.astype(float) - img_rgb.astype(float))
    info = (
        f"Sigma: {sigma}, guide_r: {guide_radius}, eps: {guide_eps}\n"
        f"Texture weight: {texture_weight}\n"
        f"Mean diff: {diff.mean():.2f}/255\n"
    )
    print(f"[Step 7] Done — mean_diff={diff.mean():.2f}")
    return result, info
