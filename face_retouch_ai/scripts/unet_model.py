"""
U-Net model for binary blemish segmentation.

Architecture: Lightweight encoder-decoder with skip connections.
Input:  (B, 3, 256, 256) — RGB face crop
Output: (B, 1, 256, 256) — sigmoid probability map
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetBlemish(nn.Module):
    """
    Light U-Net for binary blemish segmentation (256×256 input).

    Encoder:  64 → 128 → 256 → 512
    Bottleneck: 1024
    Decoder:  512 → 256 → 128 → 64
    Output:   1 channel, sigmoid activation
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bot(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))


# ─────────────────────────── Loss Functions ───────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1.0 - (
            (2.0 * intersection + self.smooth)
            / (pred_flat.sum() + target_flat.sum() + self.smooth)
        )


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_w * self.bce(pred, target) + self.dice_w * self.dice(pred, target)


# ─────────────────────────── Quick test ───────────────────────────────────

if __name__ == "__main__":
    model = UNetBlemish()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNetBlemish: {n_params:,} parameters")

    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}  range=[{y.min():.4f}, {y.max():.4f}]")

    criterion = BCEDiceLoss()
    target = torch.rand(2, 1, 256, 256)
    loss = criterion(y, target)
    print(f"Loss:   {loss.item():.4f}")
