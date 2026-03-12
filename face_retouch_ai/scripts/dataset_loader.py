"""
Dataset loader for blemish segmentation training.

Supports:
  - Acne04 dataset (auto-generated pseudo-masks from heuristic)
  - Any dataset with images/ + masks/ structure

Augmentation via albumentations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False
    print("[WARN] albumentations not installed — no augmentation will be applied.")


# ─────────────────────────── Augmentation pipelines ───────────────────────

def get_train_transforms(img_size: int = 256):
    """Training augmentations: flip, brightness, contrast, noise, rotation."""
    if not HAS_ALBUM:
        return None
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = 256):
    """Validation: resize + normalize only."""
    if not HAS_ALBUM:
        return None
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ]
    )


# ─────────────────────────── Fallback transforms ─────────────────────────

def _simple_preprocess(image: np.ndarray, mask: np.ndarray, img_size: int = 256):
    """Fallback when albumentations is not installed."""
    image = cv2.resize(image, (img_size, img_size))
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image.transpose(2, 0, 1))
    mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
    return image, mask


# ─────────────────────────── Dataset class ────────────────────────────────

class BlemishDataset(Dataset):
    """
    PyTorch dataset for blemish segmentation.

    Expected folder layout:
        root/
          images/   *.jpg | *.png
          masks/    *.png  (0 = clean, 255 = blemish)

    Image–mask pairing by sorted filename (stem match or index match).
    """

    def __init__(
        self,
        root: str | Path,
        transform=None,
        img_size: int = 256,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.transform = transform

        img_dir = self.root / "images"
        mask_dir = self.root / "masks"
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        # Collect image paths
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in img_exts]
        )

        # Build mask lookup by stem
        mask_lookup = {}
        for p in mask_dir.iterdir():
            if p.suffix.lower() in {".png", ".jpg", ".bmp"}:
                mask_lookup[p.stem] = p

        # Pair images with masks
        self.pairs = []
        for img_path in self.images:
            mask_path = mask_lookup.get(img_path.stem)
            if mask_path is not None:
                self.pairs.append((img_path, mask_path))

        if not self.pairs:
            raise RuntimeError(
                f"No image–mask pairs found in {root}. "
                f"Found {len(self.images)} images, {len(mask_lookup)} masks."
            )

        print(f"[BlemishDataset] {len(self.pairs)} pairs from {root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        # Read image (BGR → RGB)
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask (grayscale, threshold to binary)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")
        mask = (mask > 127).astype(np.uint8) * 255

        # Apply augmentations
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image_t = augmented["image"].float()
            mask_t = augmented["mask"].float().unsqueeze(0) / 255.0
        else:
            image_t, mask_t = _simple_preprocess(image, mask, self.img_size)

        return image_t, mask_t


# ─────────────────────────── Utility: create splits ───────────────────────

def create_train_val_loaders(
    dataset_root: str | Path,
    img_size: int = 256,
    batch_size: int = 8,
    val_split: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    Build train + validation DataLoaders from a single dataset root.

    Returns (train_loader, val_loader, dataset_size)
    """
    from torch.utils.data import DataLoader, random_split

    full_ds = BlemishDataset(
        root=dataset_root,
        transform=get_train_transforms(img_size),
        img_size=img_size,
    )
    n = len(full_ds)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    # Override transform for validation set items
    val_ds_wrapper = _ValSubset(val_ds, get_val_transforms(img_size), img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds_wrapper,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] Train: {n_train}, Val: {n_val}, Batch: {batch_size}")
    return train_loader, val_loader, n


class _ValSubset(Dataset):
    """Wraps a random_split Subset and overrides the augmentation for validation."""

    def __init__(self, subset, val_transform, img_size: int):
        self.subset = subset
        self.val_transform = val_transform
        self.img_size = img_size

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Access the underlying dataset's pair directly
        real_idx = self.subset.indices[idx]
        img_path, mask_path = self.subset.dataset.pairs[real_idx]

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255

        if self.val_transform is not None:
            augmented = self.val_transform(image=image, mask=mask)
            image_t = augmented["image"].float()
            mask_t = augmented["mask"].float().unsqueeze(0) / 255.0
        else:
            image_t, mask_t = _simple_preprocess(image, mask, self.img_size)

        return image_t, mask_t
