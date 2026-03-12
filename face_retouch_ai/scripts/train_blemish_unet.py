"""
Training script for U-Net blemish segmentation model.

Usage:
    python scripts/train_blemish_unet.py
    python scripts/train_blemish_unet.py --epochs 50 --batch-size 16 --lr 1e-4
    python scripts/train_blemish_unet.py --dataset datasets/acne --resume

Saves best model to:
    models/blemish_seg/unet_blemish.pth
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from unet_model import UNetBlemish, BCEDiceLoss
from dataset_loader import create_train_val_loaders


CKPT_DIR = ROOT / "models" / "blemish_seg"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL = CKPT_DIR / "unet_blemish.pth"
LAST_MODEL = CKPT_DIR / "unet_blemish_last.pth"
DEFAULT_DS = ROOT / "datasets" / "acne"


# ─────────────────────────── Metrics ──────────────────────────────────────

def dice_score(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    """Compute Dice coefficient."""
    pred_bin = (pred > thresh).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    if union.item() == 0:
        return 1.0
    return (2.0 * intersection / union).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    """Compute IoU (Jaccard) score."""
    pred_bin = (pred > thresh).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    if union.item() == 0:
        return 1.0
    return (intersection / union).item()


# ─────────────────────────── Training loop ────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(preds.detach(), masks)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_dice / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        total_loss += loss.item()
        total_dice += dice_score(preds, masks)
        total_iou += iou_score(preds, masks)
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_dice / n, total_iou / n


# ─────────────────────────── Main ─────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, n_total = create_train_val_loaders(
        dataset_root=args.dataset,
        img_size=256,
        batch_size=args.batch_size,
        val_split=0.15,
        num_workers=args.workers,
    )

    # Model
    model = UNetBlemish(in_ch=3, out_ch=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: UNetBlemish — {n_params:,} parameters")

    # Resume
    if args.resume and BEST_MODEL.exists():
        model.load_state_dict(
            torch.load(str(BEST_MODEL), map_location=device, weights_only=True)
        )
        print(f"Resumed from {BEST_MODEL}")

    # Loss, optimizer, scheduler
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float("inf")
    no_improve = 0

    print()
    print("=" * 72)
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Dice':>10}  "
          f"{'Val Loss':>10}  {'Val Dice':>10}  {'Val IoU':>10}  {'LR':>10}")
    print("=" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Save best
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(BEST_MODEL))
            improved = " ★"
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"{epoch:5d}  {train_loss:10.4f}  {train_dice:10.4f}  "
            f"{val_loss:10.4f}  {val_dice:10.4f}  {val_iou:10.4f}  "
            f"{lr:10.6f}{improved}"
        )

        # Early stopping
        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Save last model
    torch.save(model.state_dict(), str(LAST_MODEL))

    print()
    print("=" * 72)
    print(f"  Training complete!")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Best model:     {BEST_MODEL}")
    print(f"  Last model:     {LAST_MODEL}")
    print("=" * 72)

    return model


# ─────────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train U-Net blemish segmentation")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DS),
                        help="Dataset root (with images/ and masks/ subdirs)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader num_workers (0 = main thread)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from best checkpoint")
    args = parser.parse_args()

    print("=" * 72)
    print("  U-Net Blemish Segmentation — Training")
    print("=" * 72)

    ds = Path(args.dataset)
    if not (ds / "images").exists() or not (ds / "masks").exists():
        print(f"\nERROR: Dataset not found at {ds}")
        print(f"Run first:  python scripts/prepare_dataset.py --dataset-dir {ds}")
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
