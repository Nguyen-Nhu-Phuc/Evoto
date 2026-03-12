"""
Step 4 — AI Blemish Segmentation.

Primary   : U-Net trained on Acne04 / DermNet (if model file exists).
Fallback  : LAB redness + Laplacian texture heuristic (works out-of-box).

Model auto-selected: if  models/blemish_seg/unet_blemish.pth  exists → U-Net,
otherwise → heuristic.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

import torch
import torch.nn as nn

DEBUG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "debug"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "blemish_seg"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_UNET_CKPT = MODELS_DIR / "unet_blemish.pth"


# ═══════════════════════════════ U-Net loading ═══════════════════════════

_cached_unet = None


def _get_unet():
    """Load U-Net model from checkpoint. Returns (model, device) or None."""
    global _cached_unet
    if _cached_unet is not None:
        return _cached_unet
    if not _UNET_CKPT.exists():
        return None

    try:
        # Import model class from scripts/
        sys.path.insert(0, str(SCRIPTS_DIR))
        from unet_model import UNetBlemish
    except ImportError as exc:
        raise RuntimeError(f"Cannot import UNetBlemish: {exc}") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetBlemish(in_ch=3, out_ch=1).to(device).eval()
    model.load_state_dict(
        torch.load(str(_UNET_CKPT), map_location=device, weights_only=True)
    )
    _cached_unet = (model, device)
    print("  [Blemish] Loaded U-Net model from", _UNET_CKPT)
    return _cached_unet


# ═══════════════════════════════ AI inference ═════════════════════════════

def detect_blemish_ai(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray = None,
    threshold: float = 0.5,
    min_area: int = 8,
    max_area: int = 300,
    dilate_px: int = 2,
) -> tuple[np.ndarray, dict]:
    """
    AI blemish detection using the trained U-Net.

    Steps:
      1. Resize image to 256×256
      2. Run U-Net inference
      3. Threshold probability map → binary mask
      4. Post-processing: component filtering, dilation, skin constraint

    Parameters
    ----------
    img_rgb    : (H, W, 3) uint8 RGB image
    skin_mask  : (H, W) uint8 optional — 255=skin region
    threshold  : float — sigmoid threshold (0–1)
    min_area   : int — remove regions smaller than this
    max_area   : int — remove regions larger than this
    dilate_px  : int — dilation kernel radius for mask refinement

    Returns
    -------
    blemish_mask : (H, W) uint8 — 255 at blemish pixels
    debug_info   : dict with blemish_pixels, region_count, mask_ratio, etc.
    """
    if img_rgb is None or img_rgb.size == 0:
        raise ValueError("Empty image passed to detect_blemish_ai")

    result = _get_unet()
    if result is None:
        raise RuntimeError(
            f"U-Net model not found at {_UNET_CKPT}.\n"
            "Train first:  python scripts/train_blemish_unet.py"
        )

    model, device = result
    h, w = img_rgb.shape[:2]

    # 1. Resize to 256×256
    inp = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
    inp_tensor = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # 2. Run model
    with torch.no_grad():
        prob = model(inp_tensor).squeeze().cpu().numpy()  # (256, 256) in [0, 1]

    # Resize probability map back to original size
    prob_full = cv2.resize(prob, (w, h))

    # 3. Threshold
    mask = (prob_full > threshold).astype(np.uint8) * 255

    # 4. Constrain to skin region
    if skin_mask is not None:
        mask = cv2.bitwise_and(mask, skin_mask)

    # 5. Connected-component noise removal
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    region_count = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            clean[labels == i] = 255
            region_count += 1

    # 6. Light dilation for coverage
    if dilate_px > 0:
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        clean = cv2.dilate(clean, kern, iterations=1)

    # Debug info
    blemish_pixels = int(np.count_nonzero(clean))
    mask_ratio = blemish_pixels / (h * w)
    debug_info = {
        "blemish_pixels": blemish_pixels,
        "region_count": region_count,
        "mask_ratio": mask_ratio,
        "threshold": threshold,
        "image_size": (h, w),
        "prob_min": float(prob_full.min()),
        "prob_max": float(prob_full.max()),
        "prob_mean": float(prob_full.mean()),
    }

    return clean, debug_info


# ═══════════════════════════════ Heuristic fallback ═════════════════════

def _heuristic_blemish(img_rgb: np.ndarray, skin_mask: np.ndarray,
                       a_thresh: int = 140, lap_thresh: float = 15.0):
    """
    LAB redness (A > a_thresh) + Laplacian texture (|lap| > lap_thresh).
    Connected-component filter: 8 ≤ area ≤ 300 px.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a_ch = lab[:, :, 1]

    # Redness condition (A channel > threshold)
    redness = (a_ch > a_thresh).astype(np.uint8) * 255

    # Texture condition (Laplacian magnitude)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    texture = (lap > lap_thresh).astype(np.uint8) * 255

    # Both conditions must be met
    raw = cv2.bitwise_and(redness, texture)

    # Constrain to skin
    if skin_mask is not None:
        raw = cv2.bitwise_and(raw, skin_mask)

    # Connected-component filter
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    clean = np.zeros_like(raw)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 8 <= area <= 300:
            clean[labels == i] = 255

    return clean


# ═══════════════════════════════ Public API ══════════════════════════════

def detect_blemishes(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray = None,
    protect_mask: np.ndarray = None,
):
    """
    Detect blemishes. Uses U-Net when available, falls back to heuristic.

    Returns
    -------
    blemish_mask : np.ndarray (H, W) uint8 — 255 at blemish pixels
    method       : str — "unet" or "heuristic"
    info         : str
    """
    print("[Step 4] Blemish Segmentation")
    h, w = img_rgb.shape[:2]

    # Try U-Net AI model first
    try:
        mask, debug = detect_blemish_ai(img_rgb, skin_mask)
        method = "unet"
        print(f"  Using U-Net AI model — {debug['region_count']} regions, "
              f"prob range [{debug['prob_min']:.3f}, {debug['prob_max']:.3f}]")
    except (RuntimeError, FileNotFoundError):
        mask = _heuristic_blemish(img_rgb, skin_mask)
        method = "heuristic"
        debug = None
        print("  Using heuristic fallback (LAB + Laplacian).")

    # Remove protected regions (eyes, brows, lips, nose)
    if protect_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(protect_mask))

    n_px = np.count_nonzero(mask)
    area_pct = n_px / (h * w) * 100

    # Safety cap — don't mask more than 45% of image
    if area_pct > 45:
        print(f"  WARNING: blemish mask covers {area_pct:.1f}% — clamping to 45%")
        # Keep only highest-confidence regions by eroding
        while np.count_nonzero(mask) / (h * w) * 100 > 45:
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        n_px = np.count_nonzero(mask)
        area_pct = n_px / (h * w) * 100

    # Debug
    cv2.imwrite(str(DEBUG_DIR / "step4_blemish_mask.png"), mask)
    vis = img_rgb.copy()
    vis[mask > 0] = [255, 0, 0]
    cv2.imwrite(
        str(DEBUG_DIR / "step4_blemish_overlay.jpg"),
        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
    )

    info = (
        f"Method: {method}\n"
        f"Blemish pixels: {n_px} ({area_pct:.2f}% of image)\n"
    )
    if debug is not None:
        info += (
            f"Regions: {debug['region_count']}\n"
            f"Prob range: [{debug['prob_min']:.3f}, {debug['prob_max']:.3f}]\n"
            f"Prob mean: {debug['prob_mean']:.4f}\n"
        )
    print(f"[Step 4] Done — {n_px} blemish px ({area_pct:.2f}%), method={method}")
    return mask, method, info
