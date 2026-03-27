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


def _opencv_inpaint(img_rgb: np.ndarray, mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """OpenCV Telea inpainting fallback."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    result = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def inpaint(
    img_rgb: np.ndarray,
    blemish_mask: np.ndarray,
    soft_mask: np.ndarray,
    backend: str = "lama",
):
    """
    Inpaint blemish regions with pre-dilated mask for stronger removal,
    then alpha-blend with soft mask.

    Pre-dilation ensures LaMa sees a larger region around each blemish,
    producing cleaner fills that cover surrounding redness/scarring.

    Returns
    -------
    result : np.ndarray (H, W, 3) RGB uint8
    method : str — "lama" or "opencv"
    info   : str
    """
    backend = (backend or "lama").strip().lower()
    aliases = {
        "mat": "lama",
        "zits": "lama",
        "sd": "lama",
        "sd_inpaint": "lama",
        "stable-diffusion": "lama",
        "telea": "opencv",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"lama", "opencv"}:
        backend = "lama"

    print(f"[Step 6] AI Inpainting (pre-dilated) — backend={backend}")

    # Binary mask for inpainting engine
    hard_mask = (blemish_mask > 0).astype(np.uint8) * 255

    h, w = hard_mask.shape[:2]
    hard_pixels = int(np.count_nonzero(hard_mask))
    hard_ratio = hard_pixels / max(1, (h * w))

    # Adaptive parameters (key for "Evoto-like" naturalness)
    # Small coverage: keep dilation small to avoid creating plasticky patches.
    # Larger coverage: allow more context so redness/scar area is fully covered.
    if hard_ratio < 0.001:  # tiny spots
        dilate_iters = 1
        dilate_ksize = 5
        alpha_gain = 0.75
        telea_radius = 3
    elif hard_ratio < 0.01:
        dilate_iters = 2
        dilate_ksize = 7
        alpha_gain = 0.90
        telea_radius = 4
    else:
        dilate_iters = 2
        dilate_ksize = 9
        alpha_gain = 1.00
        telea_radius = 5

    # Pre-dilate mask before sending to LaMa — key improvement
    # This gives the inpainter a larger region to work with,
    # ensuring it covers the full blemish + surrounding redness
    dilate_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    inpaint_mask = cv2.dilate(hard_mask, dilate_kern, iterations=int(dilate_iters))

    # Small feather for smoother edges
    inpaint_mask = cv2.GaussianBlur(inpaint_mask, (5, 5), 1.5)
    inpaint_mask = (inpaint_mask > 32).astype(np.uint8) * 255

    # Debug: save the actual mask sent to inpainter
    cv2.imwrite(str(DEBUG_DIR / "step6_inpaint_mask.png"), inpaint_mask)

    # Run selected backend; auto-fallback to OpenCV if chosen backend fails.
    inpainted = None
    if backend == "lama":
        inpainted = _lama_inpaint(img_rgb, inpaint_mask)
        if inpainted is not None:
            method = "lama"
            print("  Using LaMa inpainting (with pre-dilated mask).")
        else:
            print("  LaMa unavailable, fallback to OpenCV Telea.")
    if inpainted is None:
        inpainted = _opencv_inpaint(img_rgb, inpaint_mask, radius=int(telea_radius))
        method = "opencv"
        print("  Using OpenCV Telea.")

    # Alpha-blend using soft mask (smooth transition at edges)
    # soft_mask is typically [0,1]. We modulate slightly to avoid over-smoothing on tiny spots.
    alpha = (soft_mask * float(alpha_gain)).clip(0.0, 1.0)[:, :, np.newaxis]  # (H, W, 1)
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
        f"Hard mask: {hard_pixels} px ({hard_ratio*100:.3f}%)\n"
        f"Adaptive: dilate={dilate_ksize}×{dilate_ksize} iters={dilate_iters}, alpha_gain={alpha_gain:.2f}, telea_r={telea_radius}\n"
        f"Mean diff: {diff.mean():.2f}/255\n"
        f"Max diff: {diff.max():.0f}\n"
    )
    print(f"[Step 6] Done — method={method}, mean_diff={diff.mean():.2f}")
    return result, method, info
