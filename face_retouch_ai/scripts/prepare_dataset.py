"""
Download and prepare the Acne04 dataset for blemish segmentation training.

Since Acne04 does not include pixel-level segmentation masks, this script:
  1. Downloads the classification images from the Acne04 GitHub repo.
  2. Generates pseudo ground-truth masks using a heuristic detector
     (LAB redness + Laplacian texture), which can later be refined manually.
  3. Saves results in the expected  datasets/acne/images/ + masks/  layout.

If Acne04 is unavailable, a synthetic fallback generates training pairs
by creating skin-toned patches with randomised blemish spots.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --dataset-dir datasets/acne
    python scripts/prepare_dataset.py --synthetic 500
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DS_DIR = ROOT / "datasets" / "acne"
ACNE04_REPO = "https://github.com/xmu-xiaoma666/ACNE04.git"
IMG_SIZE = 256


# ─────────────────────── Heuristic mask generator ─────────────────────────

def generate_pseudo_mask(
    img_bgr: np.ndarray,
    a_thresh: int = 138,
    lap_thresh: float = 12.0,
    min_area: int = 6,
    max_area: int = 400,
) -> np.ndarray:
    """
    Generate a pseudo blemish mask using LAB redness + Laplacian texture.

    This is an approximation — human refinement improves training quality.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a_ch = lab[:, :, 1]

    # Redness
    redness = (a_ch > a_thresh).astype(np.uint8) * 255

    # Texture (laplacian)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    texture = (lap > lap_thresh).astype(np.uint8) * 255

    raw = cv2.bitwise_and(redness, texture)

    # Connected-component filtering
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    clean = np.zeros_like(raw)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            clean[labels == i] = 255

    # Light dilation for coverage
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.dilate(clean, kern, iterations=1)
    return clean


# ─────────────────────── Dataset download ─────────────────────────────────

def download_acne04(dest: Path):
    """Clone Acne04 repo (or a widely-available fork) into a temp directory."""
    tmp = dest.parent / "_acne04_raw"
    if tmp.exists():
        print(f"  Raw repo already exists at {tmp}, skipping clone.")
        return tmp

    print(f"  Cloning Acne04 repository…")
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", ACNE04_REPO, str(tmp)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("  ERROR: git not found. Install git or download Acne04 manually.")
        raise
    except subprocess.CalledProcessError:
        # Try alternative fork
        alt = "https://github.com/Acne04/Acne04.git"
        print(f"  Primary repo failed, trying {alt}…")
        try:
            subprocess.check_call(
                ["git", "clone", "--depth", "1", alt, str(tmp)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            print("  ERROR: Could not clone Acne04.")
            raise
    return tmp


def find_images_recursive(root: Path, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """Find all image files recursively."""
    images = []
    for ext in exts:
        images.extend(root.rglob(f"*{ext}"))
        images.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


# ─────────────────────── Synthetic fallback ──────────────────────────────

def _random_skin_bgr(rng: np.random.RandomState) -> tuple:
    """Return a random skin-tone BGR colour."""
    # Sample in HSV skin range, convert to BGR
    h = rng.randint(5, 25)
    s = rng.randint(50, 180)
    v = rng.randint(140, 240)
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_blemish(img: np.ndarray, mask: np.ndarray,
                  cx: int, cy: int, r: int,
                  rng: np.random.RandomState):
    """Draw one synthetic blemish (reddish ellipse + texture noise)."""
    # Slightly reddish colour relative to surrounding skin
    base = img[cy, cx].astype(np.int32)
    blem_b = int(np.clip(base[0] - rng.randint(30, 70), 0, 255))
    blem_g = int(np.clip(base[1] - rng.randint(20, 50), 0, 255))
    blem_r = int(np.clip(base[2] + rng.randint(10, 50), 0, 255))
    colour = (blem_b, blem_g, blem_r)

    # Random ellipse axes
    ax1 = max(r, 2)
    ax2 = max(int(r * rng.uniform(0.6, 1.4)), 2)
    angle = rng.randint(0, 180)
    cv2.ellipse(img, (cx, cy), (ax1, ax2), angle, 0, 360, colour, -1)
    cv2.ellipse(mask, (cx, cy), (ax1, ax2), angle, 0, 360, 255, -1)

    # Add subtle noise inside blemish region for texture
    y1, y2 = max(cy - r - 2, 0), min(cy + r + 2, img.shape[0])
    x1, x2 = max(cx - r - 2, 0), min(cx + r + 2, img.shape[1])
    patch = img[y1:y2, x1:x2].astype(np.int32)
    noise = rng.randint(-12, 13, patch.shape, dtype=np.int32)
    # Apply noise only where mask is set
    m_patch = mask[y1:y2, x1:x2]
    patch = np.where(m_patch[:, :, None] > 0, np.clip(patch + noise, 0, 255), patch)
    img[y1:y2, x1:x2] = patch.astype(np.uint8)


def generate_synthetic_pair(size: int, rng: np.random.RandomState):
    """
    Create one synthetic image–mask pair.

    Returns (img_bgr, mask) both of shape (size, size).
    """
    # Base skin-coloured image with slight gradient
    b, g, r = _random_skin_bgr(rng)
    img = np.full((size, size, 3), [b, g, r], dtype=np.uint8)
    # Add gradient
    grad = np.linspace(-15, 15, size).reshape(-1, 1).astype(np.int32)
    img = np.clip(img.astype(np.int32) + grad, 0, 255).astype(np.uint8)
    # Add Gaussian noise for skin texture
    noise = rng.normal(0, 4, img.shape).astype(np.int32)
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    # Slight blur
    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    mask = np.zeros((size, size), dtype=np.uint8)

    # Draw 3-25 blemishes
    n_blemishes = rng.randint(3, 26)
    for _ in range(n_blemishes):
        margin = 8
        cx = rng.randint(margin, size - margin)
        cy = rng.randint(margin, size - margin)
        radius = rng.randint(2, 10)
        _draw_blemish(img, mask, cx, cy, radius, rng)

    # Light Gaussian blur on mask edges for softer boundaries
    mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
    mask = (mask > 127).astype(np.uint8) * 255
    return img, mask


def generate_from_real_images(real_paths: list, count: int, size: int,
                              rng: np.random.RandomState):
    """
    Create augmented training pairs from real face images by overlaying
    synthetic blemishes.
    """
    pairs = []
    for i in range(count):
        src = real_paths[i % len(real_paths)]
        img = cv2.imread(str(src))
        if img is None:
            continue
        img = cv2.resize(img, (size, size))

        # Random brightness/contrast shift
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.randint(-20, 20)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Random flip
        if rng.rand() > 0.5:
            img = cv2.flip(img, 1)

        # Random crop-and-resize for variety
        if rng.rand() > 0.3:
            margin = rng.randint(10, 40)
            y1 = rng.randint(0, margin)
            x1 = rng.randint(0, margin)
            y2 = size - rng.randint(0, margin)
            x2 = size - rng.randint(0, margin)
            crop = img[y1:y2, x1:x2]
            img = cv2.resize(crop, (size, size))

        mask = np.zeros((size, size), dtype=np.uint8)
        n_blemishes = rng.randint(5, 30)
        for _ in range(n_blemishes):
            mrg = 8
            cx = rng.randint(mrg, size - mrg)
            cy = rng.randint(mrg, size - mrg)
            radius = rng.randint(2, 10)
            _draw_blemish(img, mask, cx, cy, radius, rng)

        mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
        mask = (mask > 127).astype(np.uint8) * 255
        pairs.append((img, mask))
    return pairs


def create_synthetic_dataset(ds_dir: Path, n_samples: int = 400):
    """Generate a synthetic blemish dataset (fallback when Acne04 unavailable)."""
    images_dir = ds_dir / "images"
    masks_dir = ds_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    count = 0

    # First: augment any real face images we have
    real_imgs = []
    for search_dir in [ROOT / "inputs", ROOT.parent / "ai-retouch-test" / "inputs"]:
        if search_dir.exists():
            real_imgs.extend(find_images_recursive(search_dir))

    if real_imgs:
        n_real = min(n_samples // 2, len(real_imgs) * 40)  # up to half from real
        print(f"  Generating {n_real} pairs from {len(real_imgs)} real image(s)…")
        real_pairs = generate_from_real_images(real_imgs, n_real, IMG_SIZE, rng)
        for img, mask in real_pairs:
            name = f"synth_{count:05d}"
            cv2.imwrite(str(images_dir / f"{name}.png"), img)
            cv2.imwrite(str(masks_dir / f"{name}.png"), mask)
            count += 1

    # Fill remaining with pure synthetic skin patches
    n_synth = n_samples - count
    print(f"  Generating {n_synth} pure synthetic pairs…")
    for _ in range(n_synth):
        img, mask = generate_synthetic_pair(IMG_SIZE, rng)
        name = f"synth_{count:05d}"
        cv2.imwrite(str(images_dir / f"{name}.png"), img)
        cv2.imwrite(str(masks_dir / f"{name}.png"), mask)
        count += 1

    print(f"  Synthetic dataset: {count} pairs → {ds_dir}")
    return count


# ─────────────────────── Main preparation ─────────────────────────────────

def prepare_dataset(ds_dir: Path, synthetic_count: int = 0):
    """Download images, generate pseudo masks, save in standard layout.
    
    If synthetic_count > 0, skip download and generate synthetic data.
    If Acne04 download fails, automatically falls back to synthetic.
    """
    images_dir = ds_dir / "images"
    masks_dir = ds_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Check if already prepared
    existing = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    if len(existing) >= 10:
        print(f"  Dataset already has {len(existing)} images, skipping.")
        return len(existing)

    # If explicit synthetic requested
    if synthetic_count > 0:
        print("  Using synthetic dataset generation.")
        return create_synthetic_dataset(ds_dir, synthetic_count)

    # Try Acne04 download
    try:
        raw_dir = download_acne04(ds_dir)
    except (SystemExit, subprocess.CalledProcessError, FileNotFoundError):
        print("  Acne04 unavailable — falling back to synthetic dataset.")
        return create_synthetic_dataset(ds_dir, 400)

    raw_images = find_images_recursive(raw_dir)
    print(f"  Found {len(raw_images)} raw images.")

    if not raw_images:
        print("  WARNING: No images found in cloned repo.")
        print("  Falling back to synthetic dataset.")
        return create_synthetic_dataset(ds_dir, 400)

    count = 0
    for src_path in raw_images:
        img = cv2.imread(str(src_path))
        if img is None:
            continue

        # Resize
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Generate pseudo mask
        mask = generate_pseudo_mask(img_resized)

        # Skip images with no detected blemishes (likely clean skin)
        if np.count_nonzero(mask) < 20:
            continue

        name = f"acne_{count:05d}"
        cv2.imwrite(str(images_dir / f"{name}.png"), img_resized)
        cv2.imwrite(str(masks_dir / f"{name}.png"), mask)
        count += 1

    print(f"  Prepared {count} image–mask pairs → {ds_dir}")

    # Clean up raw clone
    try:
        shutil.rmtree(raw_dir)
        print("  Cleaned up raw repo.")
    except Exception:
        pass

    return count


# ─────────────────────── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare Acne04 blemish dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DS_DIR),
        help="Output dataset directory",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="Generate N synthetic pairs instead of downloading (0 = try download first)",
    )
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir)
    print("=" * 60)
    print("  Blemish Dataset Preparation")
    print("=" * 60)
    n = prepare_dataset(ds_dir, synthetic_count=args.synthetic)
    print(f"\nDone — {n} training pairs ready at {ds_dir}")
    print("Structure:")
    print(f"  {ds_dir / 'images'}/  ({n} files)")
    print(f"  {ds_dir / 'masks'}/   ({n} files)")
    print()
    print("NOTE: Pseudo masks are generated by heuristic (LAB + Laplacian).")
    print("For best results, manually refine masks using a labeling tool")
    print("(e.g., CVAT, LabelMe, or Napari).")


if __name__ == "__main__":
    main()
