"""
test_blemish_detection.py
=========================
STEP 4 — Multi-Signal Blemish Detection.

Combines four local-anomaly signals inside the strict skin mask:
  1. LAB A-channel relative redness  (z-score vs local neighborhood)
  2. Absolute LAB A-channel floor gate
  3. HSV saturation anomaly          (local z-score)
  4. Darkness anomaly                (locally darker in L channel)

Weighted score fusion → IQR-based adaptive threshold → morphological cleanup.

Output: outputs/debug/blemish_mask.png

Usage:
    python tests/test_blemish_detection.py
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

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Absolute A-channel floor (lowered from 145 to catch more blemishes)
A_FLOOR = 133
# Minimum blemish component area in pixels
MIN_REGION_AREA = 5


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


def load_skin_mask(h: int, w: int) -> np.ndarray:
    """Load strict skin mask (already excludes features)."""
    try:
        candidates = [
            DEBUG_DIR / "skin_mask_strict.png",
            DEBUG_DIR / "skin_mask.png",
            OUTPUTS_DIR / "face_parsing_skin.jpg",
        ]
        for path in candidates:
            if path.exists():
                print(f"[INFO] Loading skin mask from {path}")
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        print("[WARNING] No skin mask found. Detecting on full image.")
        return np.full((h, w), 255, dtype=np.uint8)
    except Exception as exc:
        print(f"[WARNING] Could not load skin mask: {exc}")
        return np.full((h, w), 255, dtype=np.uint8)


def _compute_local_stats(channel: np.ndarray, ksize: int = 31):
    """Compute local mean and std via box filters.  O(1) per pixel."""
    try:
        ch_f = channel.astype(np.float32)
        local_mean = cv2.blur(ch_f, (ksize, ksize))
        local_sq = cv2.blur(ch_f * ch_f, (ksize, ksize))
        local_var = np.maximum(local_sq - local_mean * local_mean, 0)
        return local_mean, np.sqrt(local_var)
    except Exception as exc:
        mean_val = np.mean(channel).astype(np.float32)
        std_val = max(np.std(channel).astype(np.float32), 1.0)
        return (np.full_like(channel, mean_val, dtype=np.float32),
                np.full_like(channel, std_val, dtype=np.float32))


def test_blemish_detection() -> None:
    """Multi-signal blemish detection: redness + saturation + darkness + contrast."""
    try:
        image_path = get_input_image()
        print(f"[INFO] Loading image: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image at {image_path}.")
            return

        h, w = img.shape[:2]
        print(f"[INFO] Image size: {w}x{h}")

        # Load strict skin mask (features already excluded)
        skin_mask = load_skin_mask(h, w)
        skin_bin = (skin_mask > 127).astype(np.uint8)
        skin_pixels_mask = skin_bin == 1
        skin_bin_f = skin_bin.astype(np.float32)

        skin_count = np.count_nonzero(skin_pixels_mask)
        if skin_count == 0:
            print("[ERROR] No skin pixels — cannot detect blemishes.")
            return
        print(f"[INFO] Skin pixels: {skin_count}")

        # Adaptive neighborhood size (scales with face area)
        ksize = max(31, min(61, int(np.sqrt(skin_count) * 0.15)))
        if ksize % 2 == 0:
            ksize += 1

        # ── Signal 1: LAB A-channel relative redness ────────────
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L_ch = lab[:, :, 0].astype(np.float32)
        A_ch = lab[:, :, 1].astype(np.float32)

        a_mean, a_std = _compute_local_stats(A_ch, ksize)
        redness_signal = np.clip(
            (A_ch - a_mean) / np.maximum(a_std, 2.0), 0, 5.0
        )

        # Absolute A-channel floor gate
        abs_gate = (A_ch >= A_FLOOR).astype(np.float32)

        skin_a_vals = A_ch[skin_pixels_mask]
        print(f"[INFO] A-channel: mean={np.mean(skin_a_vals):.1f}, "
              f"std={np.std(skin_a_vals):.1f}, "
              f"floor pixels={int(np.sum(abs_gate * skin_bin_f))}")

        # ── Signal 2: HSV saturation anomaly ────────────────────
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        S_ch = hsv[:, :, 1].astype(np.float32)

        s_mean, s_std = _compute_local_stats(S_ch, ksize)
        saturation_signal = np.clip(
            (S_ch - s_mean) / np.maximum(s_std, 2.0), 0, 5.0
        )

        # ── Signal 3: Darkness anomaly (L channel, inverted) ────
        l_mean, l_std = _compute_local_stats(L_ch, ksize)
        darkness_signal = np.clip(
            (l_mean - L_ch) / np.maximum(l_std, 2.0), 0, 5.0
        )

        # ── Signal 4: Local contrast (grayscale variance) ───────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        _, gray_local_std = _compute_local_stats(gray, ksize=7)
        gstd_mean, gstd_std = _compute_local_stats(gray_local_std, ksize)
        contrast_signal = np.clip(
            (gray_local_std - gstd_mean) / np.maximum(gstd_std, 1.0), 0, 5.0
        )

        # ── Score fusion (weighted combination) ─────────────────
        score = (0.40 * redness_signal
                 + 0.25 * saturation_signal
                 + 0.20 * darkness_signal
                 + 0.15 * contrast_signal)

        # Gate: absolute A floor + skin mask
        score = score * abs_gate * skin_bin_f

        # ── Adaptive thresholding (IQR-based) ───────────────────
        skin_scores = score[skin_pixels_mask]
        nonzero_scores = skin_scores[skin_scores > 0.1]

        if len(nonzero_scores) > 50:
            p75 = np.percentile(nonzero_scores, 75)
            p95 = np.percentile(nonzero_scores, 95)
            iqr = p95 - p75
            adaptive_thresh = p75 + 0.5 * iqr
            score_thresh = max(adaptive_thresh, 0.8)
        else:
            score_thresh = 1.2

        print(f"[INFO] Score: max={np.max(skin_scores):.2f}, "
              f"mean={np.mean(skin_scores):.3f}, thresh={score_thresh:.2f}")

        mask = ((score > score_thresh) & skin_pixels_mask).astype(np.uint8) * 255

        # ── Morphological cleanup ───────────────────────────────
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # Remove tiny connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        removed = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < MIN_REGION_AREA:
                mask[labels == i] = 0
                removed += 1
        print(f"[INFO] Removed {removed} tiny regions (< {MIN_REGION_AREA} px)")

        # Statistics
        n_blemish = np.count_nonzero(mask)
        pct_skin = n_blemish / max(skin_count, 1) * 100
        print(f"[INFO] Blemish pixel count: {n_blemish}")
        print(f"[INFO] % of skin area: {pct_skin:.2f}%")

        # Save mask
        cv2.imwrite(str(DEBUG_DIR / "blemish_mask.png"), mask)
        print(f"[OK] Blemish mask saved to {DEBUG_DIR / 'blemish_mask.png'}")

        # Save score heatmap for debugging
        score_max_val = max(np.max(score), 1e-6)
        score_vis = np.clip(score / score_max_val * 255, 0, 255).astype(np.uint8)
        score_heatmap = cv2.applyColorMap(score_vis, cv2.COLORMAP_JET)
        score_heatmap[~skin_pixels_mask] = 0
        cv2.imwrite(str(DEBUG_DIR / "blemish_score_heatmap.jpg"), score_heatmap)
        print(f"[OK] Score heatmap saved to {DEBUG_DIR / 'blemish_score_heatmap.jpg'}")

        # Overlay visualization
        overlay = img.copy()
        coloured = np.zeros_like(img)
        coloured[:, :, 2] = mask
        overlay = cv2.addWeighted(overlay, 0.7, coloured, 0.3, 0)
        cv2.imwrite(str(OUTPUTS_DIR / "blemish_mask.jpg"), mask)
        cv2.imwrite(str(OUTPUTS_DIR / "blemish_overlay.jpg"), overlay)
        print(f"[OK] Blemish overlay saved to {OUTPUTS_DIR / 'blemish_overlay.jpg'}")

    except Exception as exc:
        print(f"[ERROR] Blemish detection failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    test_blemish_detection()
