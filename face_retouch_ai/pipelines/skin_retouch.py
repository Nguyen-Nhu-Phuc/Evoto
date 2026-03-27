"""
Step 7 — Frequency-Separation Skin Smoothing (Evoto-style).
Splits image into low-frequency (colour/tone) and high-frequency (texture/pores).
Smooths only the low-frequency layer → preserves skin texture perfectly.
"""

import cv2
import numpy as np
from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def smooth_skin(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    blur_sigma: float = 10.0,
    guide_radius: int = 8,
    guide_eps: float = 0.02,
    high_freq_weight: float = 0.6,
):
    """
    Frequency-separation skin smoothing.

    1. Decompose: low = GaussianBlur(sigma=10), high = img - low
    2. Smooth only low-freq with guided filter (r=8, eps=0.02)
    3. Recompose: result = low_smooth + high * high_freq_weight

    This smooths colour unevenness (redness, dark spots) while preserving
    pore-level texture, matching professional retouching tools.

    Parameters
    ----------
    blur_sigma : float — Gaussian sigma for low/high frequency split
    guide_radius : int — guided filter radius for low-freq smoothing
    guide_eps : float — guided filter regularisation
    high_freq_weight : float — texture retention [0=plastic, 1=no smooth]

    Returns
    -------
    result : np.ndarray (H, W, 3) RGB uint8
    info   : str
    """
    print(f"[Step 7] Frequency-Separation Skin Smoothing (sigma={blur_sigma}, hf={high_freq_weight})")

    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, "Error: empty image."
        if skin_mask is None:
            skin_mask = np.full(img_rgb.shape[:2], 255, dtype=np.uint8)

        img_f = img_rgb.astype(np.float32) / 255.0

        # ── Step A: Frequency decomposition ──
        ksize = int(blur_sigma * 6) | 1  # ensure odd
        low_freq = cv2.GaussianBlur(img_f, (ksize, ksize), blur_sigma)
        high_freq = img_f - low_freq  # texture/pore detail

        # ── Step B: Smooth only the low-frequency layer ──
        low_smooth = cv2.ximgproc.guidedFilter(
            guide=low_freq, src=low_freq, radius=guide_radius, eps=guide_eps
        )

        # ── Step C: Recompose with controlled texture amount ──
        recomposed_f = low_smooth + high_freq * high_freq_weight
        recomposed = np.clip(recomposed_f * 255, 0, 255).astype(np.uint8)

        # ── Apply only within skin mask ──
        mask_f = (skin_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        result = (
            img_rgb.astype(np.float32) * (1 - mask_f)
            + recomposed.astype(np.float32) * mask_f
        ).clip(0, 255).astype(np.uint8)

        # Debug: save decomposition layers
        cv2.imwrite(
            str(DEBUG_DIR / "step7_low_freq.jpg"),
            cv2.cvtColor((low_freq * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        high_vis = ((high_freq + 0.5) * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(
            str(DEBUG_DIR / "step7_high_freq.jpg"),
            cv2.cvtColor(high_vis, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(DEBUG_DIR / "step7_smoothed.jpg"),
            cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
        )

        diff = np.abs(result.astype(float) - img_rgb.astype(float))
        info = (
            f"Freq-sep: blur_σ={blur_sigma}, guide_r={guide_radius}, eps={guide_eps}\n"
            f"High-freq weight: {high_freq_weight}\n"
            f"Mean diff: {diff.mean():.2f}/255\n"
        )
        print(f"[Step 7] Done — mean_diff={diff.mean():.2f}")
        return result, info

    except Exception as exc:
        print(f"[Step 7] ERROR: {exc}")
        return img_rgb, f"Skin smoothing error: {exc}"
