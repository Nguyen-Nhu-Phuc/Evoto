"""
test_face_parsing.py
====================
STEP 3 — Face Parsing at 512x512 with morphological mask refinement.

Runs BiSeNet face parsing, extracts skin mask, applies OPEN+CLOSE
morphology to prevent mask leakage into background.

Output: outputs/debug/skin_mask.png, outputs/debug/face_parsing_mask.jpg

Usage:
    python tests/test_face_parsing.py
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
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

PARSING_CKPT = MODELS_DIR / "face_parsing_79999_iter.pth"

LABEL_NAMES = {
    0: "background", 1: "skin", 2: "l_brow", 3: "r_brow",
    4: "l_eye", 5: "r_eye", 6: "eyeglass", 7: "l_ear", 8: "r_ear",
    9: "earring", 10: "nose", 11: "mouth", 12: "u_lip", 13: "l_lip",
    14: "neck", 15: "necklace", 16: "cloth", 17: "hair", 18: "hat",
}


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


def _build_bisenet(n_classes: int = 19):
    """Build minimal BiSeNet matching face-parsing.PyTorch checkpoint."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import models
    except ImportError:
        print("[ERROR] torch/torchvision not installed.")
        sys.exit(1)

    class ConvBNReLU(nn.Module):
        def __init__(self, ci, co, ks=3, s=1, p=1):
            super().__init__()
            self.conv = nn.Conv2d(ci, co, ks, s, p, bias=False)
            self.bn = nn.BatchNorm2d(co)
        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    class ARM(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.conv = ConvBNReLU(ci, co)
            self.conv_atten = nn.Conv2d(co, co, 1, bias=False)
            self.bn_atten = nn.BatchNorm2d(co)
        def forward(self, x):
            f = self.conv(x)
            a = torch.sigmoid(self.bn_atten(self.conv_atten(
                torch.mean(f, dim=[2, 3], keepdim=True))))
            return f * a

    class ContextPath(nn.Module):
        def __init__(self):
            super().__init__()
            r = models.resnet18(weights=None)
            self.conv1, self.bn1, self.relu = r.conv1, r.bn1, r.relu
            self.maxpool = r.maxpool
            self.layer1, self.layer2 = r.layer1, r.layer2
            self.layer3, self.layer4 = r.layer3, r.layer4
            self.arm16 = ARM(256, 128)
            self.arm32 = ARM(512, 128)
            self.conv_head32 = ConvBNReLU(128, 128)
            self.conv_head16 = ConvBNReLU(128, 128)
            self.conv_avg = ConvBNReLU(512, 128, ks=1, s=1, p=0)
        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            f8 = self.layer2(x)
            f16 = self.layer3(f8)
            f32 = self.layer4(f16)
            avg = self.conv_avg(torch.mean(f32, dim=[2, 3], keepdim=True))
            f32_up = self.conv_head32(
                self.arm32(f32) + F.interpolate(avg, size=f32.shape[2:], mode="nearest"))
            f32_up = F.interpolate(f32_up, size=f16.shape[2:], mode="nearest")
            f16_up = self.conv_head16(self.arm16(f16) + f32_up)
            f16_up = F.interpolate(f16_up, size=f8.shape[2:], mode="nearest")
            return f8, f16_up, f32_up

    class SpatialPath(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvBNReLU(3, 64, 7, 2, 3)
            self.conv2 = ConvBNReLU(64, 64, 3, 2, 1)
            self.conv3 = ConvBNReLU(64, 64, 3, 2, 1)
            self.conv_out = ConvBNReLU(64, 128)
        def forward(self, x):
            return self.conv_out(self.conv3(self.conv2(self.conv1(x))))

    class FFM(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.convblk = ConvBNReLU(ci, co, ks=1, s=1, p=0)
            self.conv1 = nn.Conv2d(co, co // 4, 1, bias=False)
            self.conv2 = nn.Conv2d(co // 4, co, 1, bias=False)
        def forward(self, fsp, fcp):
            f = self.convblk(torch.cat([fsp, fcp], 1))
            a = torch.sigmoid(self.conv2(F.relu(self.conv1(
                torch.mean(f, dim=[2, 3], keepdim=True)))))
            return f + f * a

    class Out(nn.Module):
        def __init__(self, ci, mi, nc):
            super().__init__()
            self.conv = ConvBNReLU(ci, mi)
            self.conv_out = nn.Conv2d(mi, nc, 1, bias=False)
        def forward(self, x):
            return self.conv_out(self.conv(x))

    class BiSeNet(nn.Module):
        def __init__(self, nc=19):
            super().__init__()
            self.cp = ContextPath()
            self.sp = SpatialPath()
            self.ffm = FFM(256, 256)
            self.conv_out = Out(256, 256, nc)
            self.conv_out16 = Out(128, 64, nc)
            self.conv_out32 = Out(128, 64, nc)
        def forward(self, x):
            H, W = x.shape[2:]
            f8, fcp8, _ = self.cp(x)
            fsp = self.sp(x)
            return F.interpolate(
                self.conv_out(self.ffm(fsp, fcp8)),
                (H, W), mode="bilinear", align_corners=True,
            )

    return BiSeNet(n_classes)


def test_face_parsing() -> None:
    """Run BiSeNet at 512x512, extract skin mask, refine with morphology."""
    try:
        import torch
        from torchvision import transforms

        image_path = get_input_image()
        print(f"[INFO] Loading image: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image at {image_path}.")
            return

        h, w = img.shape[:2]
        print(f"[INFO] Image size: {w}x{h}")

        # Build model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_bisenet(19).to(device).eval()

        if PARSING_CKPT.exists():
            print(f"[INFO] Loading weights from {PARSING_CKPT}")
            sd = torch.load(str(PARSING_CKPT), map_location=device, weights_only=True)
            model.load_state_dict(sd, strict=False)
        else:
            print(f"[WARNING] Checkpoint not found at {PARSING_CKPT}. Using random weights.")

        # Preprocess at 512x512
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        inp = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

        # Inference
        print("[INFO] Running face parsing at 512x512...")
        with torch.no_grad():
            out = model(inp)
        parsing = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

        # Colour mask
        np.random.seed(42)
        palette = np.random.randint(0, 255, (20, 3), dtype=np.uint8)
        palette[0] = [0, 0, 0]
        colour_mask = cv2.resize(palette[parsing], (w, h), interpolation=cv2.INTER_NEAREST)

        # Skin mask (label 1) with morphological refinement
        skin_raw = np.zeros_like(parsing, dtype=np.uint8)
        skin_raw[parsing == 1] = 255
        skin_raw = cv2.resize(skin_raw, (w, h), interpolation=cv2.INTER_NEAREST)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_raw, cv2.MORPH_OPEN, kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        print(f"[INFO] Skin mask pixels: {np.count_nonzero(skin_mask)}")

        # Stats
        unique, counts = np.unique(parsing, return_counts=True)
        for u, c in zip(unique, counts):
            pct = c / parsing.size * 100
            print(f"  {LABEL_NAMES.get(u, f'class_{u}')}: {pct:.1f}%")

        # Save outputs
        cv2.imwrite(str(DEBUG_DIR / "face_parsing_mask.jpg"), colour_mask)
        cv2.imwrite(str(DEBUG_DIR / "skin_mask.png"), skin_mask)
        print(f"[OK] Segmentation mask saved to {DEBUG_DIR / 'face_parsing_mask.jpg'}")
        print(f"[OK] Skin mask saved to {DEBUG_DIR / 'skin_mask.png'}")

        # Backward compatibility
        cv2.imwrite(str(OUTPUTS_DIR / "face_parsing_mask.jpg"), colour_mask)
        cv2.imwrite(str(OUTPUTS_DIR / "face_parsing_skin.jpg"), skin_mask)

    except Exception as exc:
        print(f"[ERROR] Face parsing failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    test_face_parsing()
