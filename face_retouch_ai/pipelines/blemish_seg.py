"""
Step 4 — AI Blemish Segmentation.

Primary   : U-Net trained on Acne04 / DermNet (if model file exists).
Fallback  : LAB redness + Laplacian texture heuristic (works out-of-box).

Model auto-selected: if  models/blemish_seg/unet_blemish.pth  exists -> U-Net,
otherwise -> heuristic.
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

_UNET_CKPT_CANDIDATES = [
    MODELS_DIR / "unet_blemish.pth",
    MODELS_DIR / "unet_blemish_best.pth",
]
_SEGFORMER_CKPT_CANDIDATES = [
    MODELS_DIR / "segformer_acne",
    MODELS_DIR / "segformer_b0_acne",
]


# ═══════════════════════════════ U-Net loading ═══════════════════════════

_cached_unet = None
_cached_segformer = None


def _ensure_torch_compiler_compat():
    """Compat shim for older torch versions missing torch.compiler."""
    if hasattr(torch, "compiler"):
        if not hasattr(torch.compiler, "is_compiling"):
            setattr(torch.compiler, "is_compiling", lambda: False)
        return

    class _TorchCompilerShim:
        @staticmethod
        def is_compiling():
            return False

    torch.compiler = _TorchCompilerShim()  # type: ignore[attr-defined]


def _get_segformer():
    """
    Load SegFormer model from local checkpoint directory.
    Returns (model, processor, device, ckpt_path) or None.
    """
    global _cached_segformer
    if _cached_segformer is not None:
        return _cached_segformer
    ckpt_path = next((path for path in _SEGFORMER_CKPT_CANDIDATES if path.exists()), None)
    if ckpt_path is None:
        return None
    try:
        _ensure_torch_compiler_compat()
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    except Exception:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(str(ckpt_path), local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(ckpt_path),
        local_files_only=True,
    ).to(device).eval()
    _cached_segformer = (model, processor, device, ckpt_path)
    print("  [Blemish] Loaded SegFormer model from", ckpt_path)
    return _cached_segformer


def _get_unet():
    """Load U-Net model from checkpoint. Returns (model, device) or None."""
    global _cached_unet
    if _cached_unet is not None:
        return _cached_unet
    ckpt_path = next((path for path in _UNET_CKPT_CANDIDATES if path.exists()), None)
    if ckpt_path is None:
        return None

    try:
        # Import model class from scripts/
        sys.path.insert(0, str(SCRIPTS_DIR))
        from unet_model import UNetBlemish
    except ImportError as exc:
        raise RuntimeError(f"Cannot import UNetBlemish: {exc}") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetBlemish(in_ch=3, out_ch=1).to(device).eval()
    try:
        weights = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        weights = torch.load(str(ckpt_path), map_location=device)
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    if "head.weight" in weights and "final.weight" not in weights:
        weights = {
            (key.replace("head.", "final.", 1) if key.startswith("head.") else key): value
            for key, value in weights.items()
        }
    model.load_state_dict(weights)
    _cached_unet = (model, device)
    print("  [Blemish] Loaded U-Net model from", ckpt_path)
    return _cached_unet


# ═══════════════════════════════ AI inference ═════════════════════════════

def detect_blemish_ai(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray = None,
    threshold: float = 0.5,
    weak_threshold: float | None = None,
    strong_threshold: float | None = None,
    min_area: int = 10,
    max_area: int = 250,
    dilate_px: int = 2,
) -> tuple[np.ndarray, dict]:
    """
    AI blemish detection using the trained U-Net.

    Steps:
      1. Resize image to 256×256
      2. Run U-Net inference
      3. Threshold probability map -> binary mask
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

    h, w = img_rgb.shape[:2]
    model_name = "unet"

    # Priority 1: SegFormer (if available)
    segformer_result = _get_segformer()
    if segformer_result is not None:
        model_name = "segformer"
        model, processor, device, _ = segformer_result
        with torch.no_grad():
            proc = processor(images=img_rgb, return_tensors="pt")
            proc = {k: v.to(device) for k, v in proc.items()}
            logits = model(**proc).logits
            logits_up = torch.nn.functional.interpolate(
                logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.softmax(logits_up, dim=1)
            if probs.shape[1] >= 2:
                prob_full = probs[0, 1].cpu().numpy()
            else:
                prob_full = probs[0, 0].cpu().numpy()
    else:
        # Priority 2: U-Net fallback
        result = _get_unet()
        if result is None:
            raise RuntimeError(
                f"No SegFormer/U-Net model found in {MODELS_DIR}.\n"
                "Train first: python scripts/train_segformer_acne.py or python scripts/train_blemish_unet.py"
            )
        model, device = result

        inp = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
        inp_tensor = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(inp_tensor).squeeze().cpu().numpy()
        prob_full = cv2.resize(prob, (w, h))

    # 3. Multi-level thresholding
    # Weak mask: higher recall. Strong mask: higher precision.
    if weak_threshold is None:
        weak_threshold = float(max(0.05, min(0.95, threshold - 0.12)))
    if strong_threshold is None:
        strong_threshold = float(max(0.05, min(0.95, threshold + 0.12)))

    weak = (prob_full > weak_threshold).astype(np.uint8) * 255
    strong = (prob_full > strong_threshold).astype(np.uint8) * 255

    # 4. Constrain to skin region
    if skin_mask is not None:
        weak = cv2.bitwise_and(weak, skin_mask)
        strong = cv2.bitwise_and(strong, skin_mask)

    # 5. Connected-component filtering on STRONG (precision)
    n_labels_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
    strong_clean = np.zeros_like(strong)
    region_count = 0
    areas_kept: list[int] = []
    for i in range(1, n_labels_s):
        area = int(stats_s[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            strong_clean[labels_s == i] = 255
            region_count += 1
            areas_kept.append(area)

    # 6. Weak support only near strong (recall without runaway false positives)
    # If we have strong regions: allow weak pixels only in a small ring around them.
    if region_count > 0:
        grow = max(1, int(dilate_px))
        kern_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
        strong_neighborhood = cv2.dilate(strong_clean, kern_grow, iterations=1)
        weak_supported = cv2.bitwise_and(weak, strong_neighborhood)
        clean = cv2.bitwise_or(strong_clean, weak_supported)
    else:
        # No strong regions -> fall back to weak (but still filter components)
        n_labels_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
        clean = np.zeros_like(weak)
        for i in range(1, n_labels_w):
            area = int(stats_w[i, cv2.CC_STAT_AREA])
            if min_area <= area <= max_area:
                clean[labels_w == i] = 255

    # 7. Light dilation for coverage
    if dilate_px > 0:
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        clean = cv2.dilate(clean, kern, iterations=1)

    # Confidence map (for better downstream feathering / debug)
    # Map prob into [0,1] with weak_threshold as 0.
    conf = (prob_full - float(weak_threshold)) / max(1e-6, (1.0 - float(weak_threshold)))
    conf = np.clip(conf, 0.0, 1.0)
    conf_u8 = (conf * 255.0).astype(np.uint8)
    conf_u8 = cv2.bitwise_and(conf_u8, (clean > 0).astype(np.uint8) * 255)

    # Debug info
    blemish_pixels = int(np.count_nonzero(clean))
    mask_ratio = blemish_pixels / (h * w)
    debug_info = {
        "model": model_name,
        "blemish_pixels": blemish_pixels,
        "region_count": region_count,
        "mask_ratio": mask_ratio,
        "threshold": threshold,
        "weak_threshold": float(weak_threshold),
        "strong_threshold": float(strong_threshold),
        "image_size": (h, w),
        "prob_min": float(prob_full.min()),
        "prob_max": float(prob_full.max()),
        "prob_mean": float(prob_full.mean()),
        "kept_area_min": int(min(areas_kept)) if areas_kept else 0,
        "kept_area_max": int(max(areas_kept)) if areas_kept else 0,
    }

    debug_info["confidence_map_u8"] = conf_u8
    return clean, debug_info


# ═══════════════════════════════ Multi-stage heuristic ═══════════════════

def _heuristic_blemish(img_rgb: np.ndarray, skin_mask: np.ndarray,
                       redness_thresh: int = 8, sat_thresh: int = 40,
                       lap_thresh: float = 18.0):
    """
    Multi-stage acne/blemish detection (Evoto-style).

    Stage 1 — Redness: LAB A-channel relative threshold (A - 128 > redness_thresh)
    Stage 2 — Texture anomaly: Laplacian high-frequency bump detection
    Stage 3 — Blob detection: OpenCV LoG blob detector for round spots
    Stage 4 — Combine: (redness AND texture) OR blobs
    Stage 5 — Morphology cleanup + connected-component filter

    Catches small red pimples, textured bumps, and round dark spots.
    """
    h, w = img_rgb.shape[:2]

    # ── Stage 1: Redness detection (relative LAB A-channel) ──
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a_ch = lab[:, :, 1].astype(np.float32)
    # Relative redness — compare to local mean for adaptive threshold
    a_local_mean = cv2.GaussianBlur(a_ch, (51, 51), 0)
    redness_rel = a_ch - a_local_mean
    red_mask_rel = (redness_rel > redness_thresh).astype(np.uint8)
    # Absolute redness — catch obviously red spots
    red_mask_abs = ((a_ch - 128) > redness_thresh).astype(np.uint8)
    red_mask = cv2.bitwise_or(red_mask_rel, red_mask_abs)

    # ── HSV saturation — inflamed spots are more saturated ──
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat_ch = hsv[:, :, 1]
    sat_mask = (sat_ch > sat_thresh).astype(np.uint8)

    # Combine colour signals
    colour_mask = cv2.bitwise_or(red_mask, sat_mask)

    # ── Stage 2: Texture anomaly (Laplacian high-frequency) ──
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    lap = cv2.GaussianBlur(lap, (5, 5), 0)
    lap_norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
    texture_mask = (lap_norm > lap_thresh).astype(np.uint8)

    # ── Stage 3: Blob detection (LoG — catches round pimples) ──
    blob_mask = np.zeros((h, w), dtype=np.uint8)
    try:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255  # light blobs on inverted dark spots
        params.filterByArea = True
        params.minArea = 8
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.3
        params.filterByConvexity = False
        params.filterByInertia = False

        # Detect on redness map
        red_vis = ((a_ch - 128).clip(0, 40) * (255 / 40)).astype(np.uint8)
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(red_vis)

        for kp in keypoints:
            cx, cy = int(kp.pt[0]), int(kp.pt[1])
            r = max(int(kp.size / 2), 2)
            cv2.circle(blob_mask, (cx, cy), r, 255, -1)
    except Exception:
        pass  # blob detection is optional enhancement

    # ── Stage 4: Combine — (colour AND texture) OR blobs ──
    primary = (colour_mask & texture_mask) * 255
    raw = cv2.bitwise_or(primary, blob_mask)

    # Constrain to skin region
    if skin_mask is not None:
        raw = cv2.bitwise_and(raw, skin_mask)

    # ── Stage 5: Morphology cleanup ──
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kern_open)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kern_close)

    # Connected-component filter — keep reasonable sizes
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    clean = np.zeros_like(raw)
    kept = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 6 <= area <= 600:
            clean[labels == i] = 255
            kept += 1

    n_blobs = len(keypoints) if 'keypoints' in dir() else 0
    print(
        f"  [Heuristic] redness>{redness_thresh}, sat>{sat_thresh}, "
        f"lap>{lap_thresh}, blobs={n_blobs} -> {kept} regions"
    )
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
        # Slightly higher default threshold improves precision; weak/strong bands recover recall.
        mask, debug = detect_blemish_ai(img_rgb, skin_mask, threshold=0.56)
        method = debug.get("model", "ai")
        print(f"  Using {method} AI model — {debug['region_count']} regions, "
              f"prob range [{debug['prob_min']:.3f}, {debug['prob_max']:.3f}]")
        # Hybrid fallback when AI is too conservative on dense-acne photos.
        ai_px = int(np.count_nonzero(mask))
        ai_pct = ai_px / max(1, h * w) * 100.0
        if debug['region_count'] == 0 or ai_pct < 0.15:
            print("  AI mask is sparse - merging with heuristic fallback.")
            heur = _heuristic_blemish(img_rgb, skin_mask)
            mask = cv2.bitwise_or(mask, heur)
            method = f"{method}+heuristic"
    except (RuntimeError, FileNotFoundError):
        mask = _heuristic_blemish(img_rgb, skin_mask)
        method = "heuristic"
        debug = None
        print("  Using heuristic fallback (LAB + HSV + Laplacian).")

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
    if debug is not None and "confidence_map_u8" in debug:
        cv2.imwrite(str(DEBUG_DIR / "step4_blemish_confidence.png"), debug["confidence_map_u8"])
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
            f"Weak/strong thr: {debug.get('weak_threshold', 0):.3f}/{debug.get('strong_threshold', 0):.3f}\n"
        )
    print(f"[Step 4] Done — {n_px} blemish px ({area_pct:.2f}%), method={method}")
    return mask, method, info
